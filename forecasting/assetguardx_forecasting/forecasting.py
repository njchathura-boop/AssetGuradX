"""Training, held-out evaluation and historical inference for the temporal GAT."""
from __future__ import annotations

import copy
import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from assetguardx_graph.artifacts import immutable_json

from .forecast_data import ASSETS, FAMILIES, iso, sha256, stamp
from .gat import Normalizer, TemporalGAT, onset_probabilities


def _json(path, value):
    immutable_json(Path(path), value)


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    handle = lock.open("x")
    temporary = None
    try:
        with handle:
            if path.exists():
                raise ValueError(f"refusing to overwrite model artifact: {path}")
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as staging:
                temporary = Path(staging.name)
            writer(temporary)
            os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def sample_indices(data, lookback, start, end, stride=1, *, labels=True):
    if lookback < 2 or stride < 1 or end <= start:
        raise ValueError("lookback >= 2, positive stride and increasing split required")
    a = data.arrays
    selected, masks = [], []
    for i in range(lookback - 1, len(a["times"])):
        time = int(a["times"][i])
        if not start <= time < end or int(time // 3600) % stride:
            continue
        window = a["times"][i - lookback + 1:i + 1] // 3600
        if not np.all(np.diff(window) == 1):
            continue
        mask = (a["labels"][i] >= 0) & (a["information_end"][i][None, :] < end)
        if labels and not mask.any():
            continue
        selected.append(i)
        masks.append(mask)
    shape = (0, len(ASSETS), 3)
    return np.asarray(selected, np.int64), np.stack(masks) if masks else np.zeros(shape, bool)


def split_samples(data, lookback, splits, stride):
    bounds = {key: stamp(value) for key, value in splits.items()}
    ordered = [bounds[key] for key in ("start", "train_end", "calibration_start", "validation_end", "test_end")]
    if ordered != sorted(set(ordered)):
        raise ValueError("train/selection/calibration/test boundaries must be strictly ordered")
    if stamp(data.metadata["schema_frozen_at"]) > bounds["train_end"]:
        raise ValueError("training ends before feature vocabulary freeze; prepare with an earlier --schema-until")
    intervals = {
        "train": (bounds["start"], bounds["train_end"]),
        "validation": (bounds["train_end"], bounds["calibration_start"]),
        "calibration": (bounds["calibration_start"], bounds["validation_end"]),
        "test": (bounds["validation_end"], bounds["test_end"]),
    }
    return {key: sample_indices(data, lookback, start, end, stride if key == "train" else 1)
            for key, (start, end) in intervals.items()}, intervals


def _constants(data, device):
    a = data.arrays
    return tuple(torch.as_tensor(a[key], dtype=torch.long, device=device) for key in
                 ("edge_index", "edge_type", "node_type", "target_index")) + (
        torch.as_tensor(FAMILIES, dtype=torch.long, device=device),)


def _batches(data, inputs, indices, masks, lookback, batch_size, device, order=None):
    node, edge = inputs
    if order is None:
        order = np.arange(len(indices))
    a = data.arrays
    for offset in range(0, len(order), batch_size):
        ids = order[offset:offset + batch_size]
        anchors = indices[ids]
        windows = anchors[:, None] - np.arange(lookback - 1, -1, -1)[None]
        yield tuple(torch.as_tensor(value, device=device) for value in (
            node[windows], edge[windows], a["edge_mask"][windows],
            a["labels"][anchors].astype(np.float32), a["severity"][anchors], masks[ids],
        ))


def forecast_loss(logits, severity, labels, targets, mask):
    probability = onset_probabilities(logits).clamp(1e-6, 1 - 1e-6)
    classification = F.binary_cross_entropy(probability[mask], labels[mask])
    # Percentage points, with a robust loss; no severity target is an input.
    regression = F.smooth_l1_loss(severity[mask] * 100, targets[mask] * 100)
    return classification + 0.1 * regression


def _infer(model, data, inputs, samples, lookback, batch_size, device):
    logits, severity = [], []
    constants = _constants(data, device)
    model.eval()
    with torch.no_grad():
        for node, edge, edge_mask, _, _, _ in _batches(data, inputs, *samples, lookback, batch_size, device):
            scores, amounts = model(node, edge, edge_mask, *constants)
            logits.append(scores.cpu())
            severity.append(amounts.cpu())
    empty = torch.empty((0, len(ASSETS), 3))
    return (torch.cat(logits) if logits else empty,
            torch.cat(severity) if severity else torch.empty((0, len(ASSETS), 3, 2)))


def _metric(labels, probabilities, severity, targets):
    from sklearn.metrics import average_precision_score
    if not len(labels):
        return {"count": 0, "average_precision": None, "reason": "no eligible observations"}
    both = len(np.unique(labels)) == 2
    bins = []
    for low in np.linspace(0, 0.9, 10):
        include = (probabilities >= low) & (probabilities < low + 0.1 + (1e-8 if low > 0.89 else 0))
        if include.any():
            bins.append({"lower": float(low), "count": int(include.sum()),
                         "mean_probability": float(probabilities[include].mean()),
                         "event_frequency": float(labels[include].mean())})
    return {"count": len(labels), "positives": int(labels.sum()), "prevalence": float(labels.mean()),
            "average_precision": float(average_precision_score(labels, probabilities)) if both else None,
            "ap_undefined_reason": None if both else "only one label class; event ranking is undefined",
            "brier": float(np.mean((probabilities - labels) ** 2)),
            "downside_mae_fraction": float(np.mean(abs(severity[:, 0] - targets[:, 0]))),
            "premium_mae_fraction": float(np.mean(abs(severity[:, 1] - targets[:, 1]))), "reliability_bins": bins}


def event_metrics(data, indices, masks, probabilities, start, end, alarm_threshold=0.5):
    a = data.arrays
    results = {}
    for k, horizon in enumerate((6, 12, 24)):
        per_asset = {}
        for asset_index, asset in enumerate(ASSETS):
            valid = masks[:, asset_index, k]
            available_anchors = indices[valid]
            times = a["times"][available_anchors]
            scores = probabilities[valid, asset_index, k]
            hits, leads, eligible_events = 0, [], 0
            episodes = a["times"][a["episode_onsets"][:, asset_index]]
            for onset in episodes[(episodes >= start) & (episodes < end)]:
                candidates = (times < onset) & (times >= onset - horizon * 3600)
                if not candidates.any():
                    continue
                eligible_events += 1
                warnings = times[candidates & (scores >= alarm_threshold)]
                if len(warnings):
                    hits += 1
                    leads.append(float((onset - warnings.min()) / 3600))
            false_times = times[(scores >= alarm_threshold) & (a["labels"][available_anchors, asset_index, k] == 0)]
            false_runs = int(len(false_times) > 0) + int((np.diff(false_times // 3600) > 1).sum())
            months = len(times) / (24 * 30.4375)
            per_asset[asset] = {"evaluable_episodes": eligible_events, "warned_episodes": hits,
                                "event_recall": hits / eligible_events if eligible_events else None,
                                "mean_warning_lead_hours": float(np.mean(leads)) if leads else None,
                                "false_alarm_episodes": false_runs,
                                "false_alarms_per_covered_asset_month": false_runs / months if months else None}
        results[str(horizon)] = per_asset
    return results


def metrics(data, samples, logits, severity, start, end, temperature=1.0, prevalence=None):
    indices, masks = samples
    probability = onset_probabilities(logits, temperature).numpy()
    severity = severity.numpy()
    labels, targets = data.arrays["labels"][indices], data.arrays["severity"][indices]
    report = {"snapshot_anchors": len(indices), "horizons": {}, "by_asset": {}}
    for k, horizon in enumerate((6, 12, 24)):
        mask = masks[:, :, k]
        report["horizons"][str(horizon)] = _metric(labels[:, :, k][mask], probability[:, :, k][mask],
                                                 severity[:, :, k][mask], targets[:, :, k][mask])
        if prevalence is not None and mask.any():
            report["horizons"][str(horizon)]["training_prevalence_baseline_brier"] = float(
                np.mean((labels[:, :, k][mask] - prevalence[k]) ** 2))
    for i, asset in enumerate(ASSETS):
        report["by_asset"][asset] = {}
        for k, horizon in enumerate((6, 12, 24)):
            mask = masks[:, i, k]
            report["by_asset"][asset][str(horizon)] = _metric(labels[:, i, k][mask], probability[:, i, k][mask],
                                                             severity[:, i, k][mask], targets[:, i, k][mask])
    report["events_at_fixed_alarm_threshold_0.5"] = event_metrics(data, indices, masks, probability, start, end)
    report["uncertainty"] = "No independent-event confidence intervals in this pilot; do not infer significance."
    return report


def _device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this Python environment; use --device cpu")
    return torch.device(value)


def code_hash():
    from assetguardx_graph import artifacts, temporal
    base = Path(__file__).parent
    sources = {name: sha256(base / name) for name in (
        "forecasting.py", "forecast_data.py", "gat.py", "labeling.py")}
    sources["graph/temporal.py"] = sha256(temporal.__file__)
    sources["graph/artifacts.py"] = sha256(artifacts.__file__)
    return temporal.digest(sources)


def train(data, output_dir, *, lookback=72, epochs=20, batch_size=8, hidden=32, heads=4,
          layers=2, dropout=0.1, architecture="gat", stride=6, seed=42, learning_rate=0.001,
          patience=5, device="cpu", allow_provisional=False, allow_canonical=False,
          splits=None, calibrate=False):
    if not allow_provisional and data.metadata["design"]["status"] != "approved":
        raise ValueError("label design is provisional; use --allow-provisional for a clearly marked pilot")
    if not data.metadata["database_verified"] and not allow_canonical:
        raise ValueError("dataset has no TigerGraph readback verification; use --allow-canonical for engineering only")
    if epochs < 1 or batch_size < 1 or hidden < 4 or heads < 1 or hidden % heads or patience < 1 or learning_rate <= 0:
        raise ValueError("invalid training sizes/rate")
    splits = copy.deepcopy(splits or data.metadata["design"]["splits"])
    samples, intervals = split_samples(data, lookback, splits, stride)
    if not len(samples["train"][0]) or not len(samples["validation"][0]):
        raise ValueError("no eligible training/validation windows; collect/cache more periods or explicitly set pilot split dates")
    anchors, mask = samples["train"]
    labels = data.arrays["labels"][anchors]
    positives = np.array([np.sum(labels[:, :, k][mask[:, :, k]]) for k in range(3)])
    totals = mask.sum(axis=(0, 1))
    if np.any(positives == 0) or np.any(positives == totals):
        raise ValueError("supervised onset training requires positives and negatives at every horizon; review forecast-census")
    normalizer = Normalizer.fit(data, *intervals["train"])
    inputs = normalizer.transform(data)
    device = _device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = TemporalGAT(node_features=inputs[0].shape[-1], node_kinds=len(data.metadata["node_kinds"]),
                        relation_count=len(data.metadata["relations"]), hidden=hidden, heads=heads,
                        layers=layers, dropout=dropout, architecture=architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    constants, rng = _constants(data, device), np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    _json(output_dir / "INCOMPLETE.json", {"status": "training", "seed": seed, "dataset_schema": data.metadata["tensor_schema_hash"]})
    best, best_epoch, best_state, history, waiting = math.inf, 0, None, [], 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, weight = 0, 0
        for node, edge, edge_mask, y, target, valid in _batches(data, inputs, *samples["train"], lookback, batch_size,
                                                              device, rng.permutation(len(anchors))):
            optimizer.zero_grad(set_to_none=True)
            logits, severity = model(node, edge, edge_mask, *constants)
            loss = forecast_loss(logits, severity, y, target, valid)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss; incomplete run retained")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total_loss += loss.item() * int(valid.sum())
            weight += int(valid.sum())
        selection_logits, selection_severity = _infer(model, data, inputs, samples["validation"], lookback, batch_size, device)
        ids, valid = samples["validation"]
        selection = float(forecast_loss(selection_logits, selection_severity,
                                        torch.tensor(data.arrays["labels"][ids], dtype=torch.float32),
                                        torch.tensor(data.arrays["severity"][ids]), torch.tensor(valid)))
        history.append({"epoch": epoch, "train_loss": total_loss / weight, "selection_loss": selection})
        print(f"epoch={epoch} train_loss={total_loss / weight:.6f} selection_loss={selection:.6f}", flush=True)
        if selection < best:
            best, best_epoch, waiting = selection, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            waiting += 1
            if waiting >= patience:
                break
    model.load_state_dict(best_state)
    temperature, calibration_status = 1.0, "not requested; probabilities are uncalibrated"
    if calibrate:
        ids, valid = samples["calibration"]
        if not len(ids) or len(np.unique(data.arrays["labels"][ids][valid])) < 2:
            calibration_status = "skipped: calibration period lacks both label classes"
        else:
            logits, _ = _infer(model, data, inputs, samples["calibration"], lookback, batch_size, device)
            y, valid_tensor = torch.tensor(data.arrays["labels"][ids], dtype=torch.float32), torch.tensor(valid)
            log_temperature = torch.tensor(0.0, requires_grad=True)
            calibrator = torch.optim.Adam([log_temperature], lr=0.03)
            for _ in range(100):
                calibrator.zero_grad()
                p = onset_probabilities(logits, log_temperature.exp().clamp(0.05, 20)).clamp(1e-6, 1 - 1e-6)
                error = F.binary_cross_entropy(p[valid_tensor], y[valid_tensor])
                error.backward()
                calibrator.step()
            temperature = float(log_temperature.detach().exp().clamp(0.05, 20))
            calibration_status = "temperature fitted only on separate calibration period; event adequacy remains a research gate"
    prevalence = (positives / totals).tolist()
    support = {asset: {str(h): {"eligible": int(mask[:, i, k].sum()),
                               "positive_asset_hours": int((labels[:, i, k][mask[:, i, k]] == 1).sum())}
                      for k, h in enumerate((6, 12, 24))} for i, asset in enumerate(ASSETS)}
    checkpoint = {"schema_version": "assetguardx-gat-checkpoint/v1", "model_settings": model.settings,
                  "state_dict": best_state, "normalizer": normalizer.state, "lookback_hours": lookback,
                  "tensor_schema_hash": data.metadata["tensor_schema_hash"], "data_metadata": data.metadata,
                  "splits": splits, "temperature": temperature, "calibration_status": calibration_status,
                  "seed": seed, "best_epoch": best_epoch, "training_support": support,
                  "prevalence_baseline": prevalence, "code_hash": code_hash()}
    model_path = output_dir / "model.pt"
    _atomic_write(model_path, lambda path: torch.save(checkpoint, path))
    scores, amounts = _infer(model, data, inputs, samples["validation"], lookback, batch_size, device)
    report = {"design_status": data.metadata["design"]["status"], "research_ready": False,
              "database_verified": data.metadata["database_verified"], "synthetic": data.metadata.get("synthetic", False),
              "architecture": model.settings, "parameter_count": sum(p.numel() for p in model.parameters()),
              "seed": seed, "device": str(device), "torch_version": str(torch.__version__),
              "code_hash": code_hash(), "best_epoch": best_epoch, "history": history, "splits": splits,
              "sample_counts": {key: len(value[0]) for key, value in samples.items()},
              "training_support": support, "calibration": calibration_status,
              "validation": metrics(data, samples["validation"], scores, amounts, *intervals["validation"], temperature, prevalence),
              "test_evaluated": False, "checkpoint_sha256": sha256(model_path)}
    _json(output_dir / "training_report.json", report)
    # Only this run's own marker is removed, after the checkpoint/report commit.
    (output_dir / "INCOMPLETE.json").unlink()
    return {"checkpoint": str(model_path.resolve()), "report": str((output_dir / "training_report.json").resolve()),
            "best_epoch": best_epoch, "parameter_count": report["parameter_count"], "test_evaluated": False,
            "design_status": report["design_status"], "database_verified": report["database_verified"]}


def load_model(checkpoint_path, data, device="cpu"):
    device = _device(device)
    checkpoint_path = Path(checkpoint_path)
    if (checkpoint_path.parent / "INCOMPLETE.json").exists():
        raise ValueError("model run is incomplete")
    report = json.loads((checkpoint_path.parent / "training_report.json").read_text(encoding="utf-8"))
    if sha256(checkpoint_path) != report["checkpoint_sha256"]:
        raise ValueError("model checkpoint checksum mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["schema_version"] != "assetguardx-gat-checkpoint/v1" or checkpoint["tensor_schema_hash"] != data.metadata["tensor_schema_hash"]:
        raise ValueError("model and input graph tensor schemas differ")
    if checkpoint["data_metadata"]["label_design_hash"] != data.metadata["label_design_hash"]:
        raise ValueError("model and input label definitions differ")
    model = TemporalGAT(**checkpoint["model_settings"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint, device


def predict(data, checkpoint_path, output, *, start=None, end=None, latest=False, batch_size=8, device="cpu"):
    import polars as pl
    model, checkpoint, device = load_model(checkpoint_path, data, device)
    lower = stamp(start) if start else int(data.arrays["times"][0])
    upper = stamp(end) if end else int(data.arrays["times"][-1]) + 1
    samples = sample_indices(data, checkpoint["lookback_hours"], lower, upper, labels=False)
    if latest:
        samples = samples[0][-1:], samples[1][-1:]
    if not len(samples[0]):
        raise ValueError("no complete input sequence in the requested prediction interval")
    inputs = Normalizer(checkpoint["normalizer"]).transform(data)
    logits, severity = _infer(model, data, inputs, samples, checkpoint["lookback_hours"], batch_size, device)
    probability = onset_probabilities(logits, checkpoint["temperature"]).numpy()
    severity = severity.numpy()
    model_hash, rows = sha256(checkpoint_path), []
    for j, index in enumerate(samples[0]):
        for i, asset in enumerate(ASSETS):
            state = data.arrays["episode_active"][index, i]
            deviation = data.arrays["label_observations"][index, i, 0]
            status = ("unknown_current_price" if not data.arrays["label_price_valid"][index, i] else
                      "unknown_episode_state" if state < 0 else "already_active" if state else
                      "anchor_breach" if abs(deviation) >= checkpoint["data_metadata"]["design"]["thresholds"][asset]
                      else "eligible_early_warning")
            for k, horizon in enumerate((6, 12, 24)):
                rows.append({"as_of": iso(data.arrays["times"][index]), "asset": asset,
                             "family": ("stablecoin", "liquid_staking", "btc_wrapper")[FAMILIES[i]],
                             "horizon_hours": horizon, "forecast_status": status,
                             "onset_probability": float(probability[j, i, k]) if status == "eligible_early_warning" else None,
                             "downside_severity_fraction": float(severity[j, i, k, 0]) if status == "eligible_early_warning" else None,
                             "premium_severity_fraction": float(severity[j, i, k, 1]) if status == "eligible_early_warning" else None,
                             "maximum_absolute_severity_fraction": float(severity[j, i, k].max()) if status == "eligible_early_warning" else None,
                             "training_positive_asset_hours": checkpoint["training_support"][asset][str(horizon)]["positive_asset_hours"],
                             "design_status": checkpoint["data_metadata"]["design"]["status"],
                             "database_verified": data.metadata["database_verified"],
                             "synthetic": data.metadata.get("synthetic", False),
                             "calibration_status": checkpoint["calibration_status"], "model_sha256": model_hash})
    frame = pl.DataFrame(rows).with_columns(pl.col("as_of").str.to_datetime(time_zone="UTC"))
    frame = frame.with_columns(pl.col(column).cast(pl.Float64) for column in (
        "onset_probability", "downside_severity_fraction", "premium_severity_fraction", "maximum_absolute_severity_fraction"))
    output = Path(output)
    if output.with_suffix(output.suffix + ".manifest.json").exists():
        raise ValueError("prediction manifest already exists; choose a new artifact path")
    _atomic_write(output, lambda path: frame.write_parquet(path, compression="zstd"))
    _json(output.with_suffix(output.suffix + ".manifest.json"), {
        "schema_version": "assetguardx-forecast-predictions/v1", "rows": len(rows),
        "data_file_sha256": sha256(output), "model_sha256": model_hash,
        "input_sources": data.metadata["sources"], "tensor_schema_hash": data.metadata["tensor_schema_hash"],
        "source_mode": data.metadata["source_mode"], "research_ready": False,
        "note": "historical forecasts from inputs available at as_of; predictive association, not intervention effects"})
    return {"predictions": str(output.resolve()), "rows": len(rows), "research_ready": False}


def evaluate(data, checkpoint_path, output_dir, *, split="test", batch_size=8, device="cpu"):
    model, checkpoint, device = load_model(checkpoint_path, data, device)
    samples, intervals = split_samples(data, checkpoint["lookback_hours"], checkpoint["splits"], 1)
    scores, amounts = _infer(model, data, Normalizer(checkpoint["normalizer"]).transform(data), samples[split],
                            checkpoint["lookback_hours"], batch_size, device)
    report = metrics(data, samples[split], scores, amounts, *intervals[split], checkpoint["temperature"],
                     checkpoint["prevalence_baseline"])
    report.update(split=split, model_sha256=sha256(checkpoint_path), source_mode=data.metadata["source_mode"],
                  design_status=data.metadata["design"]["status"], research_ready=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    _json(output_dir / "evaluation.json", report)
    return {"report": str((output_dir / "evaluation.json").resolve()), "split": split, "research_ready": False}

# ruff: noqa: E402 -- optional ML dependencies are skipped before importing the trainer
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pl = pytest.importorskip("polars")

from assetguardx_graph.artifacts import immutable_json
from assetguardx_forecasting.cli import main
from assetguardx_forecasting.demo import synthetic_forecast_release
from assetguardx_forecasting.forecast_data import (
    ForecastData, _cache_snapshots, _release_snapshots, census, episode_states, iso,
    market_census, prepare, read_design, sha256, stamp,
)
from assetguardx_forecasting.forecasting import evaluate, load_model, predict, sample_indices, train
from assetguardx_forecasting.gat import Normalizer, RelationalAttention, TemporalGAT, onset_probabilities


@pytest.fixture(scope="module")
def model_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("forecast")
    source = root / "synthetic.json"
    release = synthetic_forecast_release()
    immutable_json(source, release.to_dict())
    design = read_design(Path(__file__).parents[1] / "config" / "forecast.yaml")
    origin = stamp("2023-01-01T00:00:00Z")
    design["splits"] = {key: iso(origin + h * 3600) for key, h in (
        ("start", 0), ("train_end", 168), ("calibration_start", 224),
        ("validation_end", 280), ("test_end", 336))}
    design["schema_until"] = design["splits"]["train_end"]
    return prepare([source], design)


def test_attention_normalizes_valid_incoming_edges_and_ignores_missing_edges():
    torch.manual_seed(8)
    layer = RelationalAttention(8, 2, 1, dropout=0).eval()
    x = torch.randn(1, 3, 8)
    edges = torch.tensor([[0, 1], [2, 2]])
    types = torch.zeros(2, dtype=torch.long)
    attr = torch.randn(1, 2, 3)
    mask = torch.tensor([[True, False]])
    actual, weights = layer(x, edges, types, attr, mask, return_attention=True)
    assert torch.allclose(weights[:, 0], torch.ones_like(weights[:, 0]))
    assert torch.count_nonzero(weights[:, 1]) == 0
    changed = x.clone()
    changed[:, 1] += 100
    assert torch.allclose(actual[:, 2], layer(changed, edges, types, attr, mask)[:, 2])
    changed_attr = attr.clone()
    changed_attr[:, 0] += 10
    assert not torch.allclose(actual[:, 2], layer(x, edges, types, changed_attr, mask)[:, 2])
    actual.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters() if p.grad is not None)


def test_attention_is_equivariant_to_node_order_and_sensitive_to_relation():
    torch.manual_seed(9)
    layer = RelationalAttention(8, 2, 2, dropout=0).eval()
    x = torch.randn(2, 3, 8)
    edges = torch.tensor([[0, 1, 2], [2, 2, 0]])
    types = torch.tensor([0, 1, 0])
    attr = torch.randn(2, 3, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    actual = layer(x, edges, types, attr, mask)
    perm = torch.tensor([2, 0, 1])
    inverse = torch.argsort(perm)
    assert torch.allclose(actual[:, perm], layer(x[:, perm], inverse[edges], types, attr, mask), atol=1e-6)
    assert not torch.allclose(actual, layer(x, edges, 1 - types, attr, mask))


def test_probabilities_and_severity_are_monotone_across_horizons():
    model = TemporalGAT(node_features=6, node_kinds=2, relation_count=1, hidden=8, heads=2, dropout=0)
    node = torch.randn(2, 12, 3, 6)
    edge = torch.randn(2, 12, 2, 3)
    mask = torch.ones(2, 12, 2, dtype=torch.bool)
    scores, severity = model(node, edge, mask, torch.tensor([[0, 1], [1, 0]]), torch.zeros(2, dtype=torch.long),
                              torch.tensor([0, 0, 1]), torch.tensor([0, 1]), torch.tensor([0, 2]))
    probability = onset_probabilities(scores)
    assert torch.all((probability >= 0) & (probability <= 1))
    assert torch.all(torch.diff(probability, dim=-1) >= 0)
    assert torch.all(torch.diff(severity, dim=-2) >= 0)


def test_unknown_gaps_do_not_create_negative_episode_labels(model_data):
    states, starts = episode_states(np.array([0, 0, 0, .01, .01, 0, 0, 0, np.nan, 0, 0, 0]), .005)
    assert starts[3] and starts.sum() == 1
    assert states[4] == states[5] == states[6] == 1
    assert states[7] == 0 and states[8] == states[9] == states[10] == -1 and states[11] == 0
    data = deepcopy(model_data)
    gap = 26
    data.arrays["node_mask"][gap, data.arrays["target_index"][0]] = False
    from assetguardx_forecasting.forecast_data import add_labels
    add_labels(data)
    assert data.arrays["labels"][20, 0, 0] == -1  # sixth future hour is missing
    assert data.arrays["labels"][25, 0, 0] == -1
    assert data.arrays["episode_active"][26, 0] == -1


def test_labels_purge_actual_confirmation_timestamp(model_data):
    split_end = stamp("2023-01-08T00:00:00Z")
    ids, masks = sample_indices(model_data, 12, stamp("2023-01-01T00:00:00Z"), split_end)
    assert len(ids)
    for index, mask in zip(ids, masks):
        assert np.all(model_data.arrays["information_end"][index][None, :].repeat(8, axis=0)[mask] < split_end)
    assert not masks[ids == 161, :, 0].any()  # t=161 plus 6+1 confirmation crosses boundary


def test_scaler_statistics_ignore_future_values(model_data):
    data = deepcopy(model_data)
    start, end = stamp(data.metadata["design"]["splits"]["start"]), stamp(data.metadata["design"]["splits"]["train_end"])
    expected = Normalizer.fit(data, start, end).state
    data.arrays["node_values"][data.arrays["times"] >= end] += 1e10
    data.arrays["edge_values"][data.arrays["times"] >= end] += 1e10
    assert Normalizer.fit(data, start, end).state == expected


def test_label_threshold_uses_source_precision_and_confirmation_buffer(model_data):
    from assetguardx_forecasting.forecast_data import add_labels
    data = deepcopy(model_data)
    data.arrays["label_observations"][32:37, 0, 0] = .005
    add_labels(data)
    # Entry at hour 32 is the last forecast hour; hour 33 confirms the onset.
    assert data.arrays["labels"][26, 0, 0] == 1
    assert data.arrays["severity"][26, 0, 0, 1] == np.float32(.005)
    assert data.arrays["labels"][32, 0, 0] == -1  # already-breaching anchor


def test_explicit_price_age_excludes_stale_labels(model_data):
    from assetguardx_forecasting.forecast_data import add_labels
    data = deepcopy(model_data)
    data.arrays["label_observations"][26, 0, 3] = 1
    add_labels(data)
    assert data.arrays["labels"][20, 0, 0] == -1
    assert data.arrays["episode_active"][26, 0] == -1


def test_future_only_features_cannot_expand_training_vocabulary(tmp_path):
    release = synthetic_forecast_release(12)
    design = read_design(Path(__file__).parents[1] / "config" / "forecast.yaml")
    design["schema_until"] = "2023-01-01T06:00:00Z"
    future = next(r for r in release.records if r.source == r.target == "USDC" and r.available_at.hour == 8)
    new_record = replace(future, relation="future_only_feature", amount=Decimal(123))
    release = replace(release, records=release.records + (new_record,))
    path = tmp_path / "future.json"
    immutable_json(path, release.to_dict())
    data = prepare([path], design)
    assert all("future_only" not in key for key in data.metadata["feature_names"])
    assert data.metadata["ignored_records_outside_training_vocabulary"] > 0


def test_release_sweep_matches_canonical_graph_and_rejects_early_availability(tmp_path):
    release = synthetic_forecast_release(8)
    original = next(r for r in release.records if r.relation == "state_liquidity")
    delayed = replace(original, available_at=original.available_at + timedelta(hours=2),
                      valid_until=original.valid_until + timedelta(hours=3), amount=Decimal(99999))
    release = replace(release, records=tuple(delayed if r is original else r for r in release.records))
    path = tmp_path / "delayed.json"
    immutable_json(path, release.to_dict())
    for snapshot in _release_snapshots(path):
        assert snapshot.content_hash == release.graph_at(snapshot.timestamp, "exposure").content_hash
        if snapshot.timestamp < delayed.available_at:
            assert all(r.amount != Decimal(99999) for r in snapshot.records)


def _write_cache(path, snapshots):
    rows = []
    for s in snapshots:
        nodes = {n.entity_id: n for n in s.nodes}
        for r in s.records:
            rows.append({"timestamp": s.timestamp, "release_id": s.release_id, "release_hash": s.release_hash,
                "view": s.view, "source": r.source, "target": r.target,
                "source_kind": nodes[r.source].kind, "target_kind": nodes[r.target].kind,
                "source_name": nodes[r.source].name, "target_name": nodes[r.target].name,
                "relation": r.relation, "observed_at": r.observed_at, "available_at": r.available_at,
                "valid_until": r.valid_until, "valuation_at": r.valuation_at,
                "amount": str(r.amount) if r.amount is not None else None, "unit": r.unit,
                "meaning": r.meaning, "method": r.method, "quality": r.quality, "status": r.status,
                "evidence_json": json.dumps(r.evidence), "record_hash": r.content_hash})
    pl.DataFrame(rows).write_parquet(path)
    immutable_json(path.with_suffix(path.suffix + ".manifest.json"), {
        "schema_version": "assetguardx-tigergraph-snapshot-cache/v1", "release_id": snapshots[0].release_id,
        "release_hash": snapshots[0].release_hash, "view": "exposure", "every_hours": 1,
        "contains_future_available_records": False, "database_readback": "test fixture only",
        "direct_query_parity_checks": [snapshots[0].timestamp.isoformat()], "rows": len(rows),
        "data_file_sha256": sha256(path), "snapshots": len(snapshots),
        "snapshot_hashes": [{"timestamp": s.timestamp.isoformat(), "sha256": s.content_hash} for s in snapshots]})


def test_cache_roundtrip_and_corrupt_shard_detection(tmp_path):
    release = synthetic_forecast_release(4)
    times = sorted({r.available_at for r in release.records})
    snapshots = [release.graph_at(t, "exposure") for t in times]
    path = tmp_path / "cache.parquet"
    _write_cache(path, snapshots)
    assert [s.content_hash for s in _cache_snapshots(path)] == [s.content_hash for s in snapshots]
    with path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        list(_cache_snapshots(path))


def test_dataset_is_immutable_and_detects_corruption(model_data, tmp_path):
    path = tmp_path / "dataset.npz"
    model_data.save(path)
    loaded = ForecastData.load(path)
    assert np.array_equal(loaded.arrays["labels"], model_data.arrays["labels"])
    assert census(loaded)["observed_asset_episode_onsets"] > 0
    with pytest.raises(ValueError, match="overwrite"):
        model_data.save(path)
    assert not path.with_suffix(".npz.lock").exists()
    with path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        ForecastData.load(path)


def test_cli_prepare_and_census_are_connected(tmp_path):
    path = tmp_path / "synthetic.json"
    immutable_json(path, synthetic_forecast_release(48).to_dict())
    dataset = tmp_path / "prepared.npz"
    assert main(["forecast-prepare", "--releases", str(path), "--output", str(dataset)]) == 0
    assert main(["forecast-census", "--input", str(dataset)]) == 0
    assert ForecastData.load(dataset).metadata["database_verified"] is False


def test_market_census_matches_episode_definitions_without_graph_loading(model_data, tmp_path):
    rows = []
    for t in range(48):
        for i, asset in enumerate(model_data.metadata["design"]["assets"]):
            v = model_data.arrays["label_observations"][t, i]
            rows.append({"asset": asset, "hour": datetime.fromtimestamp(int(model_data.arrays["times"][t] // 3600 * 3600), timezone.utc),
                         "relative_deviation": v[0], "market_price_usd": v[1], "reference_price_usd": v[2],
                         "market_price_age_hours": v[3], "market_missing": False, "reference_missing": False})
    path = tmp_path / "processed" / "asset_hourly" / "year=2023" / "month=01" / "hourly.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(path)
    report = market_census(tmp_path, model_data.metadata["design"])
    assert report["hourly_time_points"] == 48 and report["stored_asset_hour_rows"] == 384
    assert report["observed_asset_episode_onsets"] == 8
    assert report["database_verified"] is False


def test_training_refuses_unreviewed_defaults_or_missing_event_class(model_data, tmp_path):
    with pytest.raises(ValueError, match="provisional"):
        train(model_data, tmp_path / "run")
    with pytest.raises(ValueError, match="readback"):
        train(model_data, tmp_path / "run", allow_provisional=True)
    data = deepcopy(model_data)
    data.arrays["labels"][data.arrays["labels"] >= 0] = 0
    with pytest.raises(ValueError, match="positives and negatives"):
        train(data, tmp_path / "run", lookback=12, allow_provisional=True, allow_canonical=True)


def test_train_reload_predict_and_test_future_injection(model_data, tmp_path):
    kwargs = dict(lookback=12, epochs=1, batch_size=4, hidden=8, heads=2, layers=1,
                  dropout=0, allow_provisional=True, allow_canonical=True)
    result = train(model_data, tmp_path / "first", **kwargs)
    assert result["test_evaluated"] is False
    assert not (tmp_path / "first" / "INCOMPLETE.json").exists()
    model, checkpoint, _ = load_model(tmp_path / "first" / "model.pt", model_data)
    assert checkpoint["best_epoch"] == 1
    changed = deepcopy(model_data)
    future = changed.arrays["times"] >= stamp(changed.metadata["design"]["splits"]["validation_end"])
    changed.arrays["node_values"][future] += 10000
    changed.arrays["labels"][future] = 0
    train(changed, tmp_path / "second", **kwargs)
    second, _, _ = load_model(tmp_path / "second" / "model.pt", changed)
    assert all(torch.equal(value, second.state_dict()[key]) for key, value in model.state_dict().items())
    out = tmp_path / "forecasts.parquet"
    predict(model_data, tmp_path / "first" / "model.pt", out, latest=True)
    frame = pl.read_parquet(out)
    assert frame.height == 24
    assert frame["synthetic"].all()
    for asset in frame["asset"].unique():
        p = frame.filter(pl.col("asset") == asset).sort("horizon_hours")["onset_probability"].to_numpy()
        assert np.all(np.diff(p) >= 0)
    active_out = tmp_path / "active.parquet"
    predict(model_data, tmp_path / "first" / "model.pt", active_out,
            start=iso(model_data.arrays["times"][34]), end=iso(model_data.arrays["times"][35]))
    active_frame = pl.read_parquet(active_out)
    assert active_frame["onset_probability"].null_count() == active_frame.height
    assert set(active_frame["forecast_status"]) == {"already_active"}
    evaluated = evaluate(model_data, tmp_path / "first" / "model.pt", tmp_path / "evaluation")
    report = json.loads(Path(evaluated["report"]).read_text())
    assert report["split"] == "test" and report["snapshot_anchors"] > 0
    assert main(["forecast-census", "--input", str(tmp_path / "missing.npz")]) == 2

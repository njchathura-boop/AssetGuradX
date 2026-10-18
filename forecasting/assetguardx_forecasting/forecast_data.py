"""Prepare compact, point-in-time graph tensors and provisional onset labels.

Economic entities are ML nodes. Self records are node features; cross-entity
records are typed edges with amounts, validity masks and observation ages.
Reverse edges enable message passing, not reverse financial transmission.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import yaml

from .labeling import label_window
from assetguardx_graph.artifacts import immutable_json
from assetguardx_graph.temporal import Entity, GraphRelease, Snapshot, TemporalRecord, digest


ASSETS = ("USDC", "USDT", "DAI", "FRAX", "LUSD", "stETH", "wstETH", "WBTC")
FAMILIES = (0, 0, 0, 0, 0, 1, 1, 2)
DATA_SCHEMA = "assetguardx-forecast-tensors/v1"
LABEL_KEYS = ("assets", "horizons_hours", "thresholds", "sustained_hours", "recovery_hours",
              "recovery_threshold_fraction", "maximum_market_price_age_hours")
TENSOR_KEYS = ("node_ids", "feature_names", "relations", "edges", "node_kinds")
LABEL_SOURCE_FIELDS = ("state_relative_deviation", "state_market_price_usd",
                       "state_reference_price_usd", "state_market_price_age_hours")


def stamp(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() != timedelta(0) or result.microsecond:
        raise ValueError("forecast timestamps must have an explicit UTC offset")
    return int(result.timestamp())


def iso(value):
    return datetime.fromtimestamp(int(value), timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_design(path):
    design = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if design.get("schema_version") != "assetguardx-forecast-design/v1":
        raise ValueError("unsupported forecast design")
    if design.get("status") not in {"provisional", "approved"}:
        raise ValueError("design status must be provisional or approved")
    if tuple(design["assets"]) != ASSETS or design["horizons_hours"] != [6, 12, 24]:
        raise ValueError("forecast v1 requires the eight target assets and horizons 6/12/24")
    if design["sustained_hours"] != 2 or design["recovery_hours"] != 3:
        raise ValueError("forecast v1 requires two-hour breaches and three-hour recovery")
    for asset in ASSETS:
        threshold = design["thresholds"][asset]
        if type(threshold) not in (int, float) or not np.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError(f"invalid threshold for {asset}")
    if not 0 < design["recovery_threshold_fraction"] < 1:
        raise ValueError("recovery threshold fraction must be between zero and one")
    if not np.isfinite(design["maximum_market_price_age_hours"]) or design["maximum_market_price_age_hours"] < 0:
        raise ValueError("invalid maximum market price age")
    dates = [stamp(design["splits"][key]) for key in ("start", "train_end", "calibration_start", "validation_end", "test_end")]
    if dates != sorted(set(dates)) or any(time % 3600 for time in dates) or stamp(design["schema_until"]) > dates[1]:
        raise ValueError("splits must be strictly ordered; vocabulary must freeze by train end")
    return design


def _release_snapshots(path):
    release = GraphRelease.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    # Sweep available records once rather than scanning a month for every hour.
    events = sorted(release.records, key=lambda r: (r.available_at, r.observed_at, r.key))
    by_hour = {}
    for record in events:
        if release.start <= record.available_at < release.end:
            hour = int(record.available_at.timestamp()) // 3600
            by_hour[hour] = max(by_hour.get(hour, record.available_at), record.available_at)
    nodes = {n.entity_id: n for n in release.nodes}
    selected, position = {}, 0
    for timestamp in sorted(by_hour.values()):
        while position < len(events) and events[position].available_at <= timestamp:
            record = events[position]
            prior = selected.get(record.key)
            if prior is None or (record.observed_at, record.available_at) > (prior.observed_at, prior.available_at):
                selected[record.key] = record
            position += 1
        records = tuple(r for _, r in sorted(selected.items()) if timestamp < r.valid_until and r.view == "exposure")
        active = sorted({v for r in records for v in (r.source, r.target)})
        yield Snapshot(release.release_id, release.content_hash, timestamp, "exposure", tuple(nodes[v] for v in active), records)


def _cache_snapshots(path):
    import polars as pl

    path = Path(path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "assetguardx-tigergraph-snapshot-cache/v1"
            or manifest.get("view") != "exposure" or manifest.get("every_hours") != 1
            or manifest.get("contains_future_available_records") is not False
            or not manifest.get("database_readback") or not manifest.get("direct_query_parity_checks")):
        raise ValueError("training cache requires hourly TigerGraph readback/parity evidence")
    if sha256(path) != manifest["data_file_sha256"]:
        raise ValueError("snapshot cache checksum mismatch")
    expected = {stamp(r["timestamp"]): r for r in manifest["snapshot_hashes"]}
    frame = pl.read_parquet(path).sort("timestamp")
    if frame.height != manifest["rows"]:
        raise ValueError("snapshot cache row count mismatch")
    seen = set()
    for timestamp, grouped in itertools.groupby(frame.iter_rows(named=True), key=lambda r: r["timestamp"]):
        rows = list(grouped)
        entities, records = {}, []
        for row in rows:
            if row["release_hash"] != manifest["release_hash"] or row["release_id"] != manifest["release_id"] or row["view"] != "exposure":
                raise ValueError("mixed snapshot release identities")
            for prefix in ("source", "target"):
                entity = Entity(row[prefix], row[prefix + "_kind"], row[prefix + "_name"])
                if entity.entity_id in entities and entities[entity.entity_id] != entity:
                    raise ValueError("conflicting node metadata")
                entities[entity.entity_id] = entity
            record = TemporalRecord(
                source=row["source"], target=row["target"], relation=row["relation"], view=row["view"],
                observed_at=row["observed_at"], available_at=row["available_at"], valid_until=row["valid_until"],
                valuation_at=row["valuation_at"], amount=Decimal(row["amount"]) if row["amount"] is not None else None,
                unit=row["unit"], meaning=row["meaning"], method=row["method"],
                evidence=tuple(json.loads(row["evidence_json"])), quality=row["quality"], status=row["status"],
            )
            if record.content_hash != row["record_hash"]:
                raise ValueError("cache record hash mismatch")
            records.append(record)
        snapshot = Snapshot(manifest["release_id"], manifest["release_hash"], timestamp, "exposure",
                            tuple(entities[k] for k in sorted(entities)), tuple(sorted(records, key=lambda r: r.key)))
        key = int(timestamp.timestamp())
        if key in seen or key not in expected or snapshot.content_hash != expected[key]["sha256"]:
            raise ValueError("cache snapshot hash/count mismatch")
        seen.add(key)
        yield snapshot
    if seen != set(expected) or len(seen) != manifest["snapshots"]:
        raise ValueError("snapshot cache has missing snapshots")


def iter_snapshots(paths, *, cached):
    loader = _cache_snapshots if cached else _release_snapshots
    for path in paths:
        yield from loader(path)


def _feature_key(record):
    return record.relation + "|" + record.unit


def _signed_log(value):
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("graph amount cannot be represented as a finite training value")
    return np.sign(result) * np.log1p(abs(result))


@dataclass
class ForecastData:
    arrays: dict
    metadata: dict

    def save(self, path):
        path = Path(path)
        manifest = path.with_suffix(path.suffix + ".manifest.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_suffix(path.suffix + ".lock")
        temporary = None
        handle = lock.open("x")
        try:
            with handle:
                if path.exists() or manifest.exists():
                    raise ValueError("refusing to overwrite an immutable modelling dataset")
                with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as handle:
                    temporary = Path(handle.name)
                    np.savez_compressed(handle, **self.arrays)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                immutable_json(manifest, {**self.metadata, "data_file_sha256": sha256(path)})
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
            lock.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        path = Path(path)
        metadata = json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text(encoding="utf-8"))
        if metadata["schema_version"] != DATA_SCHEMA or sha256(path) != metadata["data_file_sha256"]:
            raise ValueError("modelling dataset schema/checksum mismatch")
        if (metadata["tensor_schema_hash"] != digest({k: metadata[k] for k in TENSOR_KEYS})
                or metadata["design_hash"] != digest(metadata["design"])
                or metadata["label_design_hash"] != digest({k: metadata["design"][k] for k in LABEL_KEYS})):
            raise ValueError("modelling metadata hashes disagree")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        times = arrays["times"]
        if not len(times) or np.any(np.diff(times) <= 0) or len(set(times // 3600)) != len(times):
            raise ValueError("invalid hourly tensor timeline")
        if arrays["node_values"].shape != (len(times), len(metadata["node_ids"]), len(metadata["feature_names"])):
            raise ValueError("node tensor dimensions disagree with metadata")
        for key in ("node_values", "node_age", "edge_values"):
            if not np.isfinite(arrays[key]).all():
                raise ValueError("non-finite input tensor")
        if not np.isin(arrays["labels"], [-1, 0, 1]).all() or not np.isfinite(arrays["severity"][arrays["labels"] >= 0]).all():
            raise ValueError("invalid label tensor")
        return cls(arrays, metadata)


def episode_states(values, threshold, recovery_fraction=0.5):
    """Unknown coverage resets episode knowledge; three low hours re-establish it."""
    states = np.full(len(values), -1, np.int8)
    onsets = np.zeros(len(values), bool)
    state, low, breaches = -1, 0, 0
    for index, value in enumerate(values):
        if not np.isfinite(value):
            state, low, breaches = -1, 0, 0
        else:
            low = low + 1 if abs(value) < threshold * recovery_fraction else 0
            breaches = breaches + 1 if abs(value) >= threshold else 0
            if low >= 3:
                state = 0
            if breaches >= 2 and state != 1:
                if state == 0:
                    onsets[index - 1] = True
                state = 1
        states[index] = state
    return states, onsets


def add_labels(data):
    a, meta = data.arrays, data.metadata
    design = meta["design"]
    names = meta["feature_names"]
    def feature(relation):
        found = [i for i, key in enumerate(names) if key.startswith(relation + "|")]
        if len(found) != 1:
            raise ValueError(f"labels require exactly one node feature named {relation}")
        return found[0]
    deviation, market, reference, age = [feature("state_" + s) for s in (
        "relative_deviation", "market_price_usd", "reference_price_usd", "market_price_age_hours")]
    target = a["target_index"]
    # Labels retain source precision; float32/log-transformed neural inputs must
    # never move an observation across its declared event-entry threshold.
    values, prices, references, ages = (a["label_observations"][:, :, i].copy() for i in range(4))
    valid = (a["node_mask"][:, target, deviation] & a["node_mask"][:, target, market]
             & a["node_mask"][:, target, reference] & a["node_mask"][:, target, age]
             & (ages <= design["maximum_market_price_age_hours"]) & (prices > 0) & (references > 0))
    values[~valid] = np.nan
    a["label_price_valid"] = valid
    n, assets, horizons = len(a["times"]), len(ASSETS), design["horizons_hours"]
    labels = np.full((n, assets, len(horizons)), -1, np.int8)
    severity = np.full((n, assets, len(horizons), 2), np.nan, np.float32)
    info_end = np.zeros((n, len(horizons)), np.int64)
    active = np.full((n, assets), -1, np.int8)
    onsets = np.zeros((n, assets), bool)
    hour_to_index = {int(time // 3600): i for i, time in enumerate(a["times"])}
    # Run the episode state machine on an explicit hourly grid, including gaps.
    first, last = min(hour_to_index), max(hour_to_index)
    for asset_index, asset in enumerate(ASSETS):
        threshold = Decimal(str(design["thresholds"][asset]))
        grid = np.full(last - first + 1, np.nan)
        for hour, i in hour_to_index.items():
            grid[hour - first] = values[i, asset_index]
        states, starts = episode_states(grid, float(threshold), design["recovery_threshold_fraction"])
        deviations = {datetime.fromtimestamp(hour * 3600, timezone.utc):
                      Decimal(str(grid[hour - first])) if np.isfinite(grid[hour - first]) else None
                      for hour in range(first, last + 1)}
        for i, time in enumerate(a["times"]):
            hour = int(time // 3600)
            active[i, asset_index], onsets[i, asset_index] = states[hour - first], starts[hour - first]
            for h_index, horizon in enumerate(horizons):
                future_index = hour_to_index.get(hour + horizon + 1)
                info_end[i, h_index] = a["times"][future_index] if future_index is not None else time + (horizon + 1) * 3600
                result = label_window(datetime.fromtimestamp(hour * 3600, timezone.utc), deviations,
                                      horizon_hours=horizon, threshold=threshold,
                                      active_at_anchor=None if active[i, asset_index] < 0 else bool(active[i, asset_index]))
                if result.status == "valid":
                    labels[i, asset_index, h_index] = int(result.onset)
                    severity[i, asset_index, h_index] = float(result.downside), float(result.premium)
    a.update(labels=labels, severity=severity, information_end=info_end, episode_active=active, episode_onsets=onsets)


def prepare(paths, design, *, cached=False):
    paths = [Path(path).resolve() for path in paths]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("provide unique input graph files")
    freeze = stamp(design["schema_until"])
    source_hashes = {path: sha256(path) for path in paths}
    catalog, features, edges, times, release_hashes, synthetic_status = {}, set(), set(), set(), set(), set()
    # Vocabulary is learned from the training period only. No values are fitted.
    for snapshot in iter_snapshots(paths, cached=cached):
        time = int(snapshot.timestamp.timestamp())
        times.add(time)
        release_hashes.add(snapshot.release_hash)
        synthetic_status.add(snapshot.release_id.startswith("synthetic"))
        if time >= freeze:
            continue
        for node in snapshot.nodes:
            if node.entity_id in catalog and catalog[node.entity_id].kind != node.kind:
                raise ValueError("an entity changed kind across input releases")
            catalog[node.entity_id] = node
        for record in snapshot.records:
            key = _feature_key(record)
            if record.source == record.target:
                features.add(key)
            else:
                edges.add((record.source, record.target, key))
                edges.add((record.target, record.source, "reverse::" + key))
    if len(synthetic_status) > 1:
        raise ValueError("cannot mix synthetic fixtures with real graph inputs")
    if not times or not features or not set(ASSETS) <= set(catalog):
        raise ValueError("training schema period must contain all eight assets and state features")
    times = np.array(sorted(times), np.int64)
    if len(set(times // 3600)) != len(times):
        raise ValueError("multiple graph anchors in one hour across inputs")
    nodes, features, edges = sorted(catalog), sorted(features), sorted(edges)
    relations = sorted({edge[2] for edge in edges})
    kinds = sorted({node.kind for node in catalog.values()})
    ni, fi, ti = ({key: i for i, key in enumerate(items)} for items in (nodes, features, times.tolist()))
    ei = {edge: i for i, edge in enumerate(edges)}
    arrays = {
        "times": times,
        "node_values": np.zeros((len(times), len(nodes), len(features)), np.float32),
        "node_mask": np.zeros((len(times), len(nodes), len(features)), bool),
        "node_age": np.zeros((len(times), len(nodes), len(features)), np.float32),
        "edge_values": np.zeros((len(times), len(edges), 2), np.float32),
        "edge_mask": np.zeros((len(times), len(edges)), bool),
        "edge_index": np.array([(ni[s], ni[t]) for s, t, _ in edges], np.int64).reshape(-1, 2).T,
        "edge_type": np.array([relations.index(r) for _, _, r in edges], np.int64),
        "node_type": np.array([kinds.index(catalog[node].kind) for node in nodes], np.int64),
        "target_index": np.array([ni[asset] for asset in ASSETS], np.int64),
        "label_observations": np.full((len(times), len(ASSETS), len(LABEL_SOURCE_FIELDS)), np.nan, np.float64),
    }
    seen, ignored = {}, 0
    for snapshot in iter_snapshots(paths, cached=cached):
        time = int(snapshot.timestamp.timestamp())
        index = ti[time]
        slots = []
        for record in snapshot.records:
            if record.available_at > snapshot.timestamp or record.observed_at > snapshot.timestamp or record.valid_until <= snapshot.timestamp:
                raise ValueError("future or expired record entered a modelling snapshot")
            valid = record.quality == "valid" and record.amount is not None
            value = _signed_log(record.amount) if valid else 0.0
            age = 0.0 if record.unit == "structural_relation" else np.log1p((snapshot.timestamp - record.observed_at).total_seconds() / 3600)
            key = _feature_key(record)
            if record.source == record.target:
                if record.source in ASSETS and record.relation in LABEL_SOURCE_FIELDS and record.amount is not None:
                    arrays["label_observations"][index, ASSETS.index(record.source), LABEL_SOURCE_FIELDS.index(record.relation)] = float(record.amount)
                if record.source not in ni or key not in fi:
                    ignored += 1
                    continue
                n, f = ni[record.source], fi[key]
                slots.append(("node", n, f, value, valid, age))
            else:
                for edge in ((record.source, record.target, key), (record.target, record.source, "reverse::" + key)):
                    if edge not in ei:
                        ignored += 1
                        continue
                    slots.append(("edge", ei[edge], value, valid, age))
        signature = digest(sorted(slots))
        if index in seen:
            if seen[index] != signature:
                raise ValueError("overlapping graph releases disagree on hourly training features")
            continue
        seen[index] = signature
        for slot in slots:
            if slot[0] == "node":
                _, n, f, value, valid, age = slot
                arrays["node_values"][index, n, f] = value
                arrays["node_mask"][index, n, f] = valid
                arrays["node_age"][index, n, f] = age
            else:
                _, e, value, valid, age = slot
                arrays["edge_values"][index, e] = value, age
                arrays["edge_mask"][index, e] = valid
    if any(sha256(path) != source_hashes[path] for path in paths):
        raise ValueError("input graph artifact changed during preparation")
    metadata = {
        "schema_version": DATA_SCHEMA, "design": design, "design_hash": digest(design),
        "node_ids": nodes, "feature_names": features, "relations": relations, "edges": edges, "node_kinds": kinds,
        "source_mode": "tigergraph_verified_cache" if cached else "canonical_release_only",
        "database_verified": cached, "release_hashes": sorted(release_hashes),
        "sources": [{"path": str(path), "sha256": source_hashes[path]} for path in paths],
        "synthetic": True in synthetic_status,
        "schema_frozen_at": iso(freeze), "ignored_records_outside_training_vocabulary": ignored,
        "labels": "new sustained onset in (t,t+h]; h+1 confirmation; exclude active/breaching anchors",
        "reverse_edges": "message passing only; not reverse monetary transmission",
        "research_ready": False, "limitations": ["independent market-price reconciliation required", "event adequacy must be reviewed"],
    }
    metadata["tensor_schema_hash"] = digest({k: metadata[k] for k in TENSOR_KEYS})
    metadata["label_design_hash"] = digest({k: design[k] for k in LABEL_KEYS})
    data = ForecastData(arrays, metadata)
    add_labels(data)
    return data


def census(data):
    a, meta = data.arrays, data.metadata
    counts = {}
    for index, asset in enumerate(ASSETS):
        counts[asset] = {
            "observed_episode_onsets": int(a["episode_onsets"][:, index].sum()),
            "current_price_valid_hours": int(a["label_price_valid"][:, index].sum()),
            "unknown_episode_state_hours": int((a["episode_active"][:, index] < 0).sum()),
            "active_episode_hours": int((a["episode_active"][:, index] > 0).sum()),
            "horizons": {str(h): {"eligible": int((a["labels"][:, index, k] >= 0).sum()),
                                  "positive_asset_hours": int((a["labels"][:, index, k] == 1).sum())}
                         for k, h in enumerate(meta["design"]["horizons_hours"])},
        }
    starts = sorted(a["times"][np.any(a["episode_onsets"], axis=1)].tolist())
    clusters = sum(i == 0 or time - starts[i - 1] > 24 * 3600 for i, time in enumerate(starts))
    return {"snapshots": len(a["times"]), "asset_hours": len(a["times"]) * len(ASSETS),
            "start": iso(a["times"][0]), "end": iso(a["times"][-1]), "nodes": len(meta["node_ids"]),
            "typed_message_edges": len(meta["edges"]), "node_features": len(meta["feature_names"]),
            "design_status": meta["design"]["status"], "database_verified": meta["database_verified"],
            "observed_asset_episode_onsets": int(a["episode_onsets"].sum()), "onset_clusters_within_24h": clusters,
            "note": "clusters are an event census, not proof of statistically independent samples", "by_asset": counts}


def market_census(archive_root, design):
    """Read-only label adequacy check before building a year of graph caches.

Times here are the market table's hour labels, not model availability cutoffs.
This returns descriptive counts only; it cannot be saved as training tensors.
"""
    import polars as pl

    root = Path(archive_root) / "processed" / "asset_hourly"
    files = sorted(root.glob("year=*/month=*/*.parquet"))
    if not files:
        raise ValueError("no processed/asset_hourly Parquet files in this archive")
    columns = ["asset", "hour", "relative_deviation", "market_price_usd", "reference_price_usd",
               "market_price_age_hours", "market_missing", "reference_missing"]
    frame = pl.concat([pl.read_parquet(path, columns=columns) for path in files])
    frame = frame.filter(pl.col("asset").is_in(ASSETS)).with_columns(pl.col("hour").dt.epoch("s").alias("epoch"))
    if frame.select(pl.struct("asset", "epoch").n_unique()).item() != frame.height:
        raise ValueError("duplicate asset-hours in market census")
    if not frame.height or frame.filter(pl.col("epoch") % 3600 != 0).height:
        raise ValueError("market table must use nonempty whole-hour UTC labels")
    times = np.arange(frame["epoch"].min(), frame["epoch"].max() + 3600, 3600, dtype=np.int64)
    raw = np.full((len(times), len(ASSETS), 4), np.nan)
    mask = np.zeros(raw.shape, bool)
    for row in frame.iter_rows(named=True):
        index, asset = int((row["epoch"] - times[0]) // 3600), ASSETS.index(row["asset"])
        for feature, column in enumerate(columns[2:6]):
            value = row[column]
            if value is not None and np.isfinite(value):
                raw[index, asset, feature] = float(value)
                mask[index, asset, feature] = True
        if row["market_missing"] is not False:
            mask[index, asset, 1] = False
        if row["reference_missing"] is not False:
            mask[index, asset, 2] = False
    data = ForecastData({"times": times, "target_index": np.arange(len(ASSETS)),
                         "node_mask": mask, "label_observations": raw}, {
        "design": design, "feature_names": ["state_relative_deviation|fraction", "state_market_price_usd|USD_per_asset",
            "state_reference_price_usd|USD_per_asset", "state_market_price_age_hours|hours"],
        "node_ids": list(ASSETS), "edges": [], "database_verified": False})
    add_labels(data)
    report = census(data)
    report["hourly_time_points"] = report.pop("snapshots")
    report.pop("nodes")
    report.pop("typed_message_edges")
    return {**report, "source_mode": "processed_market_table_census_only", "stored_asset_hour_rows": frame.height,
            "timestamp_meaning": "market hour labels, not prediction availability cutoffs", "research_ready": False}

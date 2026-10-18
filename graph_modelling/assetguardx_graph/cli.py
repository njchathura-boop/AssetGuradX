"""Build, validate, load and export bounded AssetGuard-X graph releases."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

from assetguardx_graph.artifacts import immutable_json
from assetguardx_graph.demo import synthetic_release
from assetguardx_graph.real_sample import build_aave_sample
from assetguardx_graph.temporal import GraphRelease, canonical
from assetguardx_graph.tigergraph import TigerGraphStore


GRAPH_ROOT = Path(__file__).resolve().parents[1]


def read_release(path: Path) -> GraphRelease:
    return GraphRelease.from_dict(json.loads(path.read_text(encoding="utf-8")))




def _offline_demo(args):
    release = synthetic_release()
    snapshot = release.graph_at(release.start, "exposure")
    print(
        canonical(
            {
                "demonstration": "SYNTHETIC - NOT RESEARCH RESULTS",
                "snapshot_hash": snapshot.content_hash,
                "snapshot": snapshot.to_dict(),
            }
        )
    )
    if args.output:
        immutable_json(args.output, release.to_dict())
        print(f"saved_fixture={args.output.resolve()}")


def _real_sample(args):
    release = build_aave_sample(
        args.archive_root,
        first_day=date.fromisoformat(args.date),
        hours=args.hours,
        assets=tuple(args.assets),
    )
    immutable_json(args.output, release.to_dict())
    print(
        json.dumps(
            {
                "sample_file": str(args.output.resolve()),
                "release_id": release.release_id,
                "release_hash": release.content_hash,
                "nodes": len(release.nodes),
                "records": len(release.records),
                "source": "REAL processed Aave V2 reserve cash; not a stress model",
            },
            indent=2,
        )
    )


def _parse_utc(value: str, flag: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{flag} must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{flag} must include an explicit UTC offset")
    return parsed.astimezone(timezone.utc)


def _real_release(args):
    from assetguardx_graph.real_release import build_release

    if bool(args.quote_root) != bool(args.quote_release_id):
        raise ValueError("--quote-root and --quote-release-id must be supplied together")
    release = build_release(
        args.archive_root,
        start_time=_parse_utc(args.start_time, "--start-time"),
        end_time=_parse_utc(args.end_time, "--end-time"),
        context_hours=args.context_hours,
        include_quote_root=args.quote_root,
        quote_release_id=args.quote_release_id,
    )
    immutable_json(args.output, release.to_dict())
    print(json.dumps({
        "release_file": str(args.output.resolve()),
        "release_id": release.release_id,
        "release_hash": release.content_hash,
        "start": release.start.isoformat(),
        "end_exclusive": release.end.isoformat(),
        "entities": len(release.nodes),
        "temporal_records": len(release.records),
        "quote_release_id": args.quote_release_id,
        "source": "processed Drive tables and raw hourly block anchors",
        "note": "bounded historical release; reserve aggregates are not borrower positions",
    }, indent=2))


def _graph_cache(args):
    import pandas as pd

    release = read_release(args.input)
    start = _parse_utc(args.start_time, "--start-time")
    end = _parse_utc(args.end_time, "--end-time")
    if any((start.minute, start.second, start.microsecond, end.minute, end.second, end.microsecond)):
        raise ValueError("snapshot cache boundaries must be aligned to UTC hours")
    if end <= start or args.every_hours <= 0:
        raise ValueError("cache end must follow start and --every-hours must be positive")
    anchor_times = sorted({
        record.available_at
        for record in release.records
        if start <= record.available_at < end
    })
    if anchor_times:
        origin = anchor_times[0]
        timestamps = [
            timestamp for timestamp in anchor_times
            if int((timestamp - origin).total_seconds() // 3600) % args.every_hours == 0
        ]
    else:
        timestamps = []
    if not timestamps:
        raise ValueError("snapshot cache contains no timestamps")

    graph_config = yaml.safe_load(args.graph_config.read_text(encoding="utf-8"))
    db = TigerGraphStore(
        base_url=os.environ.get("ASSETGUARDX_TIGERGRAPH_URL", graph_config["restpp_url"])
    )
    snapshots = db.graph_snapshots(release, timestamps, args.view)
    by_timestamp = {snapshot.timestamp: snapshot for snapshot in snapshots}
    direct_checks = sorted({timestamps[0], timestamps[len(timestamps) // 2], timestamps[-1]})
    for timestamp in direct_checks:
        direct = db.graph_at(release, timestamp, args.view)
        if direct.content_hash != by_timestamp[timestamp].content_hash:
            raise RuntimeError(f"direct TigerGraph query differs from cached snapshot at {timestamp}")

    cache_rows = []
    for snapshot in snapshots:
        node_by_id = {node.entity_id: node for node in snapshot.nodes}
        node_index = {node.entity_id: index for index, node in enumerate(snapshot.nodes)}
        for record in snapshot.records:
            if record.available_at > snapshot.timestamp or record.observed_at > snapshot.timestamp:
                raise RuntimeError("future information detected while writing graph snapshot cache")
            cache_rows.append({
                "timestamp": snapshot.timestamp,
                "release_id": snapshot.release_id,
                "release_hash": snapshot.release_hash,
                "view": snapshot.view,
                "source": record.source,
                "source_index": node_index[record.source],
                "source_kind": node_by_id[record.source].kind,
                "source_name": node_by_id[record.source].name,
                "target": record.target,
                "target_index": node_index[record.target],
                "target_kind": node_by_id[record.target].kind,
                "target_name": node_by_id[record.target].name,
                "relation": record.relation,
                "observed_at": record.observed_at,
                "available_at": record.available_at,
                "valid_until": record.valid_until,
                "valuation_at": record.valuation_at,
                "amount": str(record.amount) if record.amount is not None else None,
                "unit": record.unit,
                "meaning": record.meaning,
                "method": record.method,
                "evidence_json": json.dumps(record.evidence, separators=(",", ":")),
                "quality": record.quality,
                "status": record.status,
                "age_seconds": int((snapshot.timestamp - record.observed_at).total_seconds()),
                "record_hash": record.content_hash,
            })
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".manifest.json").exists():
        raise ValueError("refusing to overwrite an existing immutable snapshot cache artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if partial.exists():
        raise ValueError(f"interrupted cache staging file already exists; inspect before retrying: {partial}")
    frame = pd.DataFrame(cache_rows)
    frame.to_parquet(partial, index=False, compression="zstd")
    partial.replace(args.output)
    from assetguardx_graph.real_release import _sha256

    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    immutable_json(manifest_path, {
        "schema_version": "assetguardx-tigergraph-snapshot-cache/v1",
        "release_id": release.release_id,
        "release_hash": release.content_hash,
        "view": args.view,
        "start_time": start.isoformat(),
        "end_time_exclusive": end.isoformat(),
        "every_hours": args.every_hours,
        "snapshots": len(snapshots),
        "rows": len(cache_rows),
        "direct_query_parity_checks": [item.isoformat() for item in direct_checks],
        "query_version": "agx_records_v1",
        "snapshot_hashes": [
            {
                "timestamp": item.timestamp.isoformat(),
                "sha256": item.content_hash,
                "node_count": len(item.nodes),
                "record_count": len(item.records),
            }
            for item in snapshots
        ],
        "contains_future_available_records": False,
        "data_file": args.output.name,
        "data_file_sha256": _sha256(args.output),
        "database_readback": "one bounded release query plus direct checks at first/middle/last snapshot",
    })
    print(json.dumps({
        "cache": str(args.output.resolve()),
        "manifest": str(manifest_path.resolve()),
        "snapshots": len(snapshots),
        "rows": len(cache_rows),
        "direct_query_parity_checks": len(direct_checks),
        "release_hash": release.content_hash,
        "future_information_check": "passed",
    }, indent=2))
def _graph(args):
    release = read_release(args.input)
    if args.command == "graph-check":
        print(
            json.dumps(
                {
                    "release_id": release.release_id,
                    "release_hash": release.content_hash,
                    "nodes": len(release.nodes),
                    "records": len(release.records),
                    "database_verified": False,
                },
                indent=2,
            )
        )
        return

    graph_config = yaml.safe_load(args.graph_config.read_text(encoding="utf-8"))
    db = TigerGraphStore(
        base_url=os.environ.get("ASSETGUARDX_TIGERGRAPH_URL", graph_config["restpp_url"])
    )
    if args.command == "graph-load":
        print(f"loaded_release_hash={db.load_graph_release(release)}")
        return

    at = datetime.fromisoformat(args.at.replace("Z", "+00:00"))
    snapshot = db.graph_at(release, at, args.view)
    if args.command == "graph-verify":
        print(
            json.dumps(
                {
                    "database_verified": True,
                    "release_id": release.release_id,
                    "release_hash": release.content_hash,
                    "snapshot_hash": snapshot.content_hash,
                    "at": at.isoformat(),
                    "view": args.view,
                    "nodes": len(snapshot.nodes),
                    "active_records": len(snapshot.records),
                    "sample_records": [
                        {
                            "source": record.source,
                            "target": record.target,
                            "relation": record.relation,
                            "amount": str(record.amount) if record.amount is not None else None,
                            "unit": record.unit,
                            "evidence": record.evidence,
                        }
                        for record in snapshot.records[:5]
                    ],
                },
                indent=2,
            )
        )
        return

    artifact = {
        "snapshot": snapshot.to_dict(),
        "snapshot_hash": snapshot.content_hash,
        "database_verified": True,
        "query_version": "agx_records_v1",
        "release_provenance": release.provenance,
    }
    immutable_json(args.output, artifact)
    print(f"export={args.output.resolve()}")
    print("Local file commit complete; confirm Google Drive sync before using it elsewhere.")




def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("offline-demo", help="print a synthetic graph fixture")
    demo.add_argument("--output", type=Path, help="optional path to save the synthetic release JSON")
    demo.set_defaults(handler=_offline_demo)

    sample = sub.add_parser(
        "real-sample", help="build a bounded real Aave V2 reserve sample from processed Parquet"
    )
    sample.add_argument("--archive-root", type=Path, required=True)
    sample.add_argument("--date", default="2023-01-01", help="UTC YYYY-MM-DD; stays within one month")
    sample.add_argument("--hours", type=int, default=24, help="1 to 168 hours, default 24")
    sample.add_argument("--assets", nargs="+", default=["USDC", "DAI", "WBTC"])
    sample.add_argument("--output", type=Path, required=True)
    sample.set_defaults(handler=_real_sample)

    historical = sub.add_parser(
        "real-release", help="build one point-in-time multi-protocol graph window from Drive Parquet"
    )
    historical.add_argument("--archive-root", type=Path, required=True)
    historical.add_argument("--start-time", required=True, help="evaluation start, aligned UTC hour")
    historical.add_argument("--end-time", required=True, help="exclusive evaluation end, max 30 days")
    historical.add_argument("--context-hours", type=int, default=168, help="preceding context, max 168")
    historical.add_argument("--quote-root", type=Path, help="supplemental/quote_sensitivity directory")
    historical.add_argument("--quote-release-id", help="complete supplemental quote release identifier")
    historical.add_argument("--output", type=Path, required=True)
    historical.set_defaults(handler=_real_release)

    for name in ("graph-check", "graph-load", "graph-verify", "graph-export"):
        command = sub.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument(
            "--graph-config", type=Path, default=GRAPH_ROOT / "config" / "graph.yaml"
        )
        command.set_defaults(handler=_graph)
        if name in {"graph-verify", "graph-export"}:
            command.add_argument("--at", required=True, help="UTC ISO timestamp")
            command.add_argument("--view", choices=["exposure", "correlation", "mstg"], required=True)
        if name == "graph-export":
            command.add_argument("--output", type=Path, required=True)

    cache = sub.add_parser(
        "graph-cache", help="cache point-in-time TigerGraph snapshots in a Parquet feature table"
    )
    cache.add_argument("--input", type=Path, required=True)
    cache.add_argument("--start-time", required=True, help="inclusive UTC hour")
    cache.add_argument("--end-time", required=True, help="exclusive UTC hour")
    cache.add_argument("--every-hours", type=int, default=1)
    cache.add_argument("--view", choices=["exposure"], required=True)
    cache.add_argument("--output", type=Path, required=True, help="new .parquet path")
    cache.add_argument("--graph-config", type=Path, default=GRAPH_ROOT / "config" / "graph.yaml")
    cache.set_defaults(handler=_graph_cache)


    args = parser.parse_args(argv)
    try:
        args.handler(args)
        return 0
    except ImportError as exc:
        print(f"ERROR: {exc}. Install graph_modelling/requirements.txt in this Python environment.", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare forecasting data, train temporal graph networks and evaluate predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from assetguardx_graph.artifacts import immutable_json

from .forecast_data import stamp

FORECAST_ROOT = Path(__file__).resolve().parents[1]


def _forecast_prepare(args):
    from .forecast_data import census, prepare, read_design
    design = read_design(args.design)
    if args.schema_until:
        design["schema_until"] = args.schema_until
        if stamp(args.schema_until) > stamp(design["splits"]["train_end"]):
            raise ValueError("feature vocabulary must freeze by train end")
    data = prepare(args.cache or args.releases, design, cached=bool(args.cache))
    data.save(args.output)
    print(json.dumps({"dataset": str(args.output.resolve()), **census(data)}, indent=2))


def _forecast_census(args):
    from .forecast_data import ForecastData, census, market_census, read_design
    report = census(ForecastData.load(args.input)) if args.input else market_census(args.archive_root, read_design(args.design))
    print(json.dumps(report, indent=2))


def _forecast_train(args):
    from .forecast_data import ForecastData
    from .forecasting import train
    data = ForecastData.load(args.input)
    splits = dict(data.metadata["design"]["splits"])
    for key in ("start", "train_end", "calibration_start", "validation_end", "test_end"):
        override = getattr(args, "split_start" if key == "start" else key)
        if override:
            splits[key] = override
    result = train(data, args.output_dir, lookback=args.lookback_hours, epochs=args.epochs,
                   batch_size=args.batch_size, hidden=args.hidden, heads=args.heads, layers=args.layers,
                   architecture=args.architecture, stride=args.train_stride_hours, seed=args.seed,
                   device=args.device, allow_provisional=args.allow_provisional,
                   allow_canonical=args.allow_canonical, splits=splits, calibrate=args.calibrate)
    print(json.dumps(result, indent=2))


def _forecast_predict(args):
    from .forecast_data import ForecastData
    from .forecasting import predict
    print(json.dumps(predict(ForecastData.load(args.input), args.checkpoint, args.output,
                             start=args.start_time, end=args.end_time, latest=args.latest,
                             batch_size=args.batch_size, device=args.device), indent=2))


def _forecast_evaluate(args):
    from .forecast_data import ForecastData
    from .forecasting import evaluate
    print(json.dumps(evaluate(ForecastData.load(args.input), args.checkpoint, args.output_dir,
                              split=args.split, batch_size=args.batch_size, device=args.device), indent=2))


def _forecast_smoke(args):
    from .demo import synthetic_forecast_release
    from .forecast_data import census, iso, prepare, read_design
    from .forecasting import predict, train
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    release = synthetic_forecast_release()
    source = root / "synthetic-release.json"
    immutable_json(source, release.to_dict())
    design = read_design(FORECAST_ROOT / "config" / "forecast.yaml")
    origin = int(release.start.timestamp()) // 3600 * 3600
    design["splits"] = {key: iso(origin + hour * 3600) for key, hour in
                        (("start", 0), ("train_end", 168), ("calibration_start", 224),
                         ("validation_end", 280), ("test_end", 336))}
    design["schema_until"] = design["splits"]["train_end"]
    data = prepare([source], design)
    data.save(root / "synthetic-dataset.npz")
    result = train(data, root / "model", lookback=12, epochs=2, batch_size=4, hidden=16,
                   heads=2, layers=1, allow_provisional=True, allow_canonical=True)
    predictions = predict(data, root / "model" / "model.pt", root / "synthetic-predictions.parquet", latest=True)
    print(json.dumps({"demonstration": "SYNTHETIC ENGINEERING ONLY - NOT RESEARCH RESULTS",
                      "census": census(data), "training": result, "predictions": predictions}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("forecast-prepare", help="tensorize historical graphs; build provisional sustained-onset labels")
    sources = prepare.add_mutually_exclusive_group(required=True)
    sources.add_argument("--cache", type=Path, nargs="+", help="verified hourly TigerGraph Parquet caches")
    sources.add_argument("--releases", type=Path, nargs="+", help="canonical JSON releases; engineering path, not DB-verified")
    prepare.add_argument("--design", type=Path, default=FORECAST_ROOT / "config" / "forecast.yaml")
    prepare.add_argument("--schema-until", help="UTC vocabulary freeze; must be no later than train end")
    prepare.add_argument("--output", type=Path, required=True, help="new immutable .npz artifact")
    prepare.set_defaults(handler=_forecast_prepare)

    census = sub.add_parser("forecast-census", help="count eligible targets and observed episodes before training")
    census_sources = census.add_mutually_exclusive_group(required=True)
    census_sources.add_argument("--input", type=Path)
    census_sources.add_argument("--archive-root", type=Path, help="read-only census of all processed market hours")
    census.add_argument("--design", type=Path, default=FORECAST_ROOT / "config" / "forecast.yaml")
    census.set_defaults(handler=_forecast_census)

    train = sub.add_parser("forecast-train", help="train GAT/mean-GNN/GRU with purged chronological splits")
    train.add_argument("--input", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True, help="new run directory")
    train.add_argument("--architecture", choices=["gat", "mean", "gru"], default="gat")
    train.add_argument("--lookback-hours", type=int, default=72)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--hidden", type=int, default=32)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--layers", type=int, choices=[1, 2], default=2)
    train.add_argument("--train-stride-hours", type=int, default=6)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    train.add_argument("--allow-provisional", action="store_true")
    train.add_argument("--allow-canonical", action="store_true", help="engineering-only training without TG readback")
    train.add_argument("--calibrate", action="store_true", help="fit temperature on separate calibration period")
    for flag in ("split-start", "train-end", "calibration-start", "validation-end", "test-end"):
        train.add_argument("--" + flag, help="explicit UTC pilot split override")
    train.set_defaults(handler=_forecast_train)

    predict = sub.add_parser("forecast-predict", help="write historical 6/12/24-hour forecasts from a saved checkpoint")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--latest", action="store_true", help="only last complete input sequence")
    predict.add_argument("--start-time")
    predict.add_argument("--end-time")
    predict.add_argument("--batch-size", type=int, default=8)
    predict.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    predict.set_defaults(handler=_forecast_predict)

    evaluate = sub.add_parser("forecast-evaluate", help="explicit held-out evaluation; test never used by training")
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--split", choices=["validation", "calibration", "test"], default="test")
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    evaluate.set_defaults(handler=_forecast_evaluate)

    smoke = sub.add_parser("forecast-smoke", help="train/reload/predict on explicitly synthetic engineering data")
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.set_defaults(handler=_forecast_smoke)

    args = parser.parse_args(argv)
    try:
        args.handler(args)
        return 0
    except ImportError as exc:
        print(f"ERROR: {exc}. Install forecasting/requirements.txt in this Python environment.", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

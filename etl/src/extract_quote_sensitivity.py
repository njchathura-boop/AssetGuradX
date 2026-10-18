"""Extract sampled, size-aware historical swap quotes from selected pools.

This produces a supplementary research release beside the already-frozen raw
archive. Every quote is an ``eth_call`` at a recorded historical block; it does
not submit a transaction or change Ethereum state. A quote is a pool simulation,
not a recorded trade or a guaranteed executable fill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
import yaml
from eth_abi import encode
from eth_utils import keccak

from extract_rpc_state import RpcClient, decode_result
from utils import (
    active_at,
    copy_to_drive_with_retries,
    load_yaml,
    parse_utc,
    resolve_storage_root,
    sha256_file,
    staging_path,
)


def _selector(signature: str) -> bytes:
    return keccak(text=signature)[:4]


def _decimal_amount(raw: str, decimals: int) -> tuple[int, str]:
    try:
        amount = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"invalid configured token amount: {raw}") from exc
    if not amount.is_finite() or amount <= 0:
        raise ValueError(f"configured token amount must be positive and finite: {raw}")
    scaled = amount * (Decimal(10) ** decimals)
    if scaled != scaled.to_integral_value():
        raise ValueError(f"configured amount {raw} exceeds {decimals} token decimals")
    integer = int(scaled)
    return integer, format(amount, "f")


def _size_class(symbol: str, registry: dict) -> str:
    asset = registry["assets"][symbol]
    return asset.get("family", "support")


def _amounts(symbol: str, registry: dict, cfg: dict) -> list[tuple[int, str]]:
    family = _size_class(symbol, registry)
    configured = cfg["input_amounts"].get(family)
    if configured is None:
        raise ValueError(f"no quote sizes configured for {symbol} (family {family})")
    decimals = int(registry["assets"][symbol]["decimals"])
    return [_decimal_amount(str(value), decimals) for value in configured]


def _curve_quote_specs(
    *, contract_key: str, meta: dict, registry: dict, block_row: dict, quote_cfg: dict
) -> list[dict]:
    specs = []
    for quote in meta.get("market_quotes", []):
        target, quote_asset = quote["target"], quote["quote"]
        method = quote.get("method", "get_dy")
        target_index, quote_index = int(quote["target_index"]), int(quote["quote_index"])
        directions = (
            (target, quote_asset, target_index, quote_index),
            (quote_asset, target, quote_index, target_index),
        )
        for token_in, token_out, index_in, index_out in directions:
            decimals = int(registry["assets"][token_in]["decimals"])
            for amount_raw, amount_text in _amounts(token_in, registry, quote_cfg):
                signature = f"{method}(int128,int128,uint256)"
                data = _selector(signature) + encode(
                    ["int128", "int128", "uint256"],
                    [index_in, index_out, amount_raw],
                )
                specs.append({
                    "protocol": "curve",
                    "contract_key": contract_key,
                    "contract_address": meta["address"].lower(),
                    "hour": block_row["hour"],
                    "snapshot_block_timestamp": block_row["block_timestamp"],
                    "block_number": int(block_row["block_number"]),
                    "input_asset": token_in,
                    "output_asset": token_out,
                    "input_decimals": decimals,
                    "amount_in_raw": str(amount_raw),
                    "amount_in": amount_text,
                    "call_kind": "curve_get_dy",
                    "to": meta["address"].lower(),
                    "data": "0x" + data.hex(),
                })
    return specs


def _uniswap_quote_specs(
    *, contract_key: str, meta: dict, registry: dict, block_row: dict, quote_cfg: dict
) -> list[dict]:
    quoter = quote_cfg["uniswap_v3_quoter_v2"]["address"].lower()
    signature = quote_cfg["uniswap_v3_quoter_v2"]["signature"]
    assets = registry["assets"]
    specs = []
    for token_in, token_out in ((meta["token0"], meta["token1"]), (meta["token1"], meta["token0"])):
        decimals = int(assets[token_in]["decimals"])
        for amount_raw, amount_text in _amounts(token_in, registry, quote_cfg):
            encoded = encode(
                ["(address,address,uint256,uint24,uint160)"],
                [(
                    assets[token_in]["address"],
                    assets[token_out]["address"],
                    amount_raw,
                    int(meta["fee"]),
                    0,
                )],
            )
            data = _selector(signature) + encoded
            specs.append({
                "protocol": "uniswap_v3",
                "contract_key": contract_key,
                "contract_address": meta["address"].lower(),
                "quoter_address": quoter,
                "hour": block_row["hour"],
                "snapshot_block_timestamp": block_row["block_timestamp"],
                "block_number": int(block_row["block_number"]),
                "input_asset": token_in,
                "output_asset": token_out,
                "input_decimals": decimals,
                "amount_in_raw": str(amount_raw),
                "amount_in": amount_text,
                "call_kind": "uniswap_v3_quoter_v2",
                "to": quoter,
                "data": "0x" + data.hex(),
            })
    return specs


def build_specs(blocks: pd.DataFrame, registry: dict, quote_cfg: dict) -> list[dict]:
    """Build deterministic historical calls; this function does no network I/O."""
    contracts = registry["contracts"]
    assets = registry["assets"]
    specs = []
    for block_row in blocks.to_dict(orient="records"):
        hour = pd.Timestamp(block_row["hour"]).to_pydatetime()
        for key, meta in contracts.items():
            if not active_at(meta, hour):
                continue
            if meta.get("state_kind") == "curve_pool":
                specs.extend(_curve_quote_specs(
                    contract_key=key, meta=meta, registry=registry,
                    block_row=block_row, quote_cfg=quote_cfg,
                ))
            elif meta.get("state_kind") == "uniswap_v3_pool":
                specs.extend(_uniswap_quote_specs(
                    contract_key=key, meta=meta, registry=registry,
                    block_row=block_row, quote_cfg=quote_cfg,
                ))
    return specs


def _is_execution_revert(error: str) -> bool:
    return bool(re.search(r"execution reverted|revert(?:ed)?", error, flags=re.IGNORECASE))


def flatten_results(results: list[dict], registry: dict) -> pd.DataFrame:
    rows = []
    for item in results:
        spec = item["spec"]
        decimals_in = int(spec["input_decimals"])
        decimals_out = int(registry["assets"][spec["output_asset"]]["decimals"])
        result = item.get("result")
        if item.get("error"):
            error = str(item["error"])
            if not _is_execution_revert(error):
                raise RuntimeError(
                    f"RPC transport/batch failure for {spec['contract_key']} at "
                    f"block {spec['block_number']}; month not committed."
                )
            rows.append({
                **{k: v for k, v in spec.items() if k not in {"data", "to"}},
                "quote_status": "execution_reverted",
                "amount_out_raw": None,
                "amount_out": None,
                "output_per_input": None,
                "sqrt_price_after_x96": None,
                "initialized_ticks_crossed": None,
                "gas_estimate": None,
                "error_class": "EVM_EXECUTION_REVERT",
            })
            continue
        if spec["call_kind"] == "curve_get_dy":
            amount_out_raw = int(decode_result(result, ["uint256"])[0])
            sqrt_after = ticks_crossed = gas_estimate = None
        else:
            decoded = decode_result(result, ["uint256", "uint160", "uint32", "uint256"])
            amount_out_raw, sqrt_after, ticks_crossed, gas_estimate = decoded
        amount_in_raw = int(spec["amount_in_raw"])
        amount_in = Decimal(amount_in_raw) / (Decimal(10) ** decimals_in)
        amount_out = Decimal(int(amount_out_raw)) / (Decimal(10) ** decimals_out)
        rows.append({
            **{k: v for k, v in spec.items() if k not in {"data", "to"}},
            "quote_status": "quoted" if amount_out_raw > 0 else "zero_output",
            "amount_out_raw": str(amount_out_raw),
            "amount_out": format(amount_out, "f"),
            "output_per_input": format(amount_out / amount_in, "f") if amount_in else None,
            "sqrt_price_after_x96": str(sqrt_after) if sqrt_after is not None else None,
            "initialized_ticks_crossed": int(ticks_crossed) if ticks_crossed is not None else None,
            "gas_estimate": str(gas_estimate) if gas_estimate is not None else None,
            "error_class": None,
        })
    return pd.DataFrame(rows)


def _release_id(start: datetime, end: datetime, step: int, cfg: dict, registry: dict) -> str:
    payload = {
        "schema": cfg["schema_version"],
        "start": start.isoformat(),
        "end": end.isoformat(),
        "every_hours": step,
        "quote_cfg": cfg,
        "contracts": registry["contracts"],
        "assets": registry["assets"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def _assert_utc_hour(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{label} must be an explicit UTC timestamp")
    if any((value.minute, value.second, value.microsecond)):
        raise ValueError(f"{label} must be aligned to an exact UTC hour")


def _load_blocks(root: Path, start: datetime, end: datetime, step: int) -> pd.DataFrame:
    frames = []
    for path in sorted((root / "raw" / "hourly_blocks").glob("year=*/month=*/*.parquet")):
        frame = pd.read_parquet(path, columns=["hour", "block_number", "block_timestamp"])
        frame["hour"] = pd.to_datetime(frame["hour"], utc=True)
        mask = (frame["hour"] >= start) & (frame["hour"] < end)
        if mask.any():
            frames.append(frame.loc[mask].copy())
    if not frames:
        raise RuntimeError("No hourly block anchors were found for the requested range")
    blocks = pd.concat(frames, ignore_index=True).sort_values("hour")
    if blocks["hour"].duplicated().any():
        raise ValueError("duplicate hourly block anchors in raw archive")
    elapsed_hours = ((blocks["hour"] - pd.Timestamp(start)).dt.total_seconds() / 3600).round().astype(int)
    blocks = blocks.loc[(elapsed_hours % step).eq(0)].copy()
    if blocks.empty:
        raise RuntimeError("No hourly anchors match the requested sampling interval")
    expected = pd.date_range(start=start, end=end, freq=f"{step}h", inclusive="left")
    actual = pd.DatetimeIndex(blocks["hour"])
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    if len(missing) or len(extra):
        first_missing = missing[0].isoformat() if len(missing) else None
        raise RuntimeError(
            "hourly block anchors do not completely cover the requested sampling grid "
            f"(missing={len(missing)}, extra={len(extra)}, first_missing={first_missing})"
        )
    blocks["block_timestamp"] = pd.to_datetime(blocks["block_timestamp"], utc=True)
    return blocks


def _verify_existing(path: Path, expected_keys: set[str]) -> dict:
    frame = pd.read_parquet(path, columns=["quote_key", "quote_status"])
    actual = set(frame["quote_key"].astype(str))
    if actual != expected_keys or not frame["quote_status"].isin(
        ["quoted", "zero_output", "execution_reverted"]
    ).all():
        raise RuntimeError(f"existing supplemental partition is incomplete or incompatible: {path}")
    return {
        "path": str(path), "rows": len(frame), "sha256": sha256_file(path),
        "quoted": int(frame["quote_status"].eq("quoted").sum()),
        "zero_output": int(frame["quote_status"].eq("zero_output").sum()),
        "execution_reverted": int(frame["quote_status"].eq("execution_reverted").sum()),
        "resumed": True,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-time", help="UTC inclusive; defaults to study start")
    parser.add_argument("--end-time", help="UTC exclusive; defaults to validation end")
    parser.add_argument("--every-hours", type=int, help="sampling interval; default from config")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--pause", type=float, default=0.25)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--execute", action="store_true", help="perform archive-RPC eth_call requests")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.retries <= 0 or args.timeout <= 0 or args.pause < 0:
        raise SystemExit("batch size/retries/timeout must be positive and pause nonnegative")

    etl_cfg = load_yaml("config/extraction.yaml")
    registry = load_yaml("config/contracts.yaml")
    quote_cfg = load_yaml("config/quote_sensitivity.yaml")
    default_start = parse_utc(etl_cfg["study_window"]["start"])
    default_end = parse_utc(etl_cfg["study_window"]["validation_end_exclusive"])
    start = parse_utc(args.start_time) if args.start_time else default_start
    end = parse_utc(args.end_time) if args.end_time else default_end
    _assert_utc_hour(start, "--start-time")
    _assert_utc_hour(end, "--end-time")
    if end <= start:
        raise SystemExit("--end-time must be later than --start-time")
    step = args.every_hours or int(quote_cfg["sampling"]["every_hours"])
    if step <= 0:
        raise SystemExit("--every-hours must be positive")

    root = resolve_storage_root(etl_cfg, require=True)
    blocks = _load_blocks(root, start, end, step)
    extractor_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    release_id = _release_id(start, end, step, quote_cfg, registry)
    release_id = hashlib.sha256(f"{release_id}|{extractor_hash}".encode()).hexdigest()[:20]
    by_month: dict[str, list[dict]] = defaultdict(list)
    for month, frame in blocks.groupby(blocks["hour"].dt.strftime("%Y-%m"), sort=True):
        by_month[month] = build_specs(frame, registry, quote_cfg)
    total_quotes = sum(len(specs) for specs in by_month.values())
    max_quotes = int(quote_cfg["sampling"]["max_quote_calls_per_release"])
    plan = {
        "schema_version": quote_cfg["schema_version"],
        "mode": "execute" if args.execute else "dry_run",
        "release_id": release_id,
        "start_time": start.isoformat(),
        "end_time_exclusive": end.isoformat(),
        "sample_every_hours": step,
        "sampled_hours": len(blocks),
        "quote_calls": total_quotes,
        "estimated_batch_requests": math.ceil(total_quotes / args.batch_size),
        "pool_count": len({spec["contract_key"] for specs in by_month.values() for spec in specs}),
        "maximum_quote_calls": max_quotes,
        "source": "Ethereum historical eth_call; no BigQuery scan",
    }
    if total_quotes > max_quotes:
        raise SystemExit(
            f"planned quote count {total_quotes:,} exceeds configured cap {max_quotes:,}; "
            "increase --every-hours or explicitly raise config/quote_sensitivity.yaml cap"
        )
    print(json.dumps(plan, indent=2))
    if not args.execute:
        print("Dry run only. Add --execute to make read-only historical quotes and write the supplement to Drive.")
        return 0

    rpc_url = __import__("os").getenv(etl_cfg["rpc"]["env_var"])
    if not rpc_url:
        raise SystemExit(f"Missing environment variable {etl_cfg['rpc']['env_var']}")
    rpc = RpcClient(
        rpc_url,
        timeout=args.timeout,
        retries=args.retries,
        batch_size=args.batch_size,
        pause=args.pause,
    )
    output_root = root / quote_cfg["storage"]["path"] / f"release_id={release_id}"
    file_manifest = []
    for month, specs in by_month.items():
        expected_keys = {
            hashlib.sha256(
                "|".join((pd.Timestamp(s["hour"]).isoformat(), s["contract_key"], s["input_asset"],
                           s["output_asset"], s["amount_in_raw"])).encode()
            ).hexdigest()
            for s in specs
        }
        out = output_root / f"year={month[:4]}" / f"month={month[5:7]}" / f"quote_sensitivity_{month}.parquet"
        if out.exists():
            file_manifest.append(_verify_existing(out, expected_keys))
            print(f"[RESUME] {month}: verified existing immutable partition")
            continue

        # Drop the internal calldata before converting results into research rows.
        call_results = rpc.batch_eth_call(specs)
        frame = flatten_results(call_results, registry)
        frame["quote_key"] = [
            hashlib.sha256(
                "|".join((pd.Timestamp(row.hour).isoformat(), row.contract_key, row.input_asset,
                           row.output_asset, str(row.amount_in_raw))).encode()
            ).hexdigest()
            for row in frame.itertuples(index=False)
        ]
        if set(frame["quote_key"]) != expected_keys:
            raise RuntimeError(f"quote-result identity mismatch for {month}; partition was not committed")
        stage = staging_path(etl_cfg, f"quote_sensitivity_{release_id}", month, out.name)
        partial = stage.with_suffix(stage.suffix + ".partial")
        stage.parent.mkdir(parents=True, exist_ok=True)
        if partial.exists():
            raise RuntimeError(f"interrupted local staging file exists; inspect before retrying: {partial}")
        frame.to_parquet(partial, index=False, compression=etl_cfg["output"].get("compression", "zstd"))
        partial.replace(stage)
        published = copy_to_drive_with_retries(stage, out, etl_cfg)
        entry = {
            "path": str(out), "rows": len(frame), "sha256": published["sha256"],
            "bytes": published["bytes"], "quoted": int(frame["quote_status"].eq("quoted").sum()),
            "execution_reverted": int(frame["quote_status"].eq("execution_reverted").sum()),
            "copy_attempts": published["copy_attempts"], "resumed": False,
        }
        file_manifest.append(entry)
        print(
            f"[OK] {month}: rows={len(frame):,}; quoted={entry['quoted']:,}; "
            f"execution_reverted={entry['execution_reverted']:,}; Drive={out}",
            flush=True,
        )

    manifest_path = root / quote_cfg["storage"]["manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as source:
            for line in source:
                try:
                    prior = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if prior.get("release_id") == release_id and prior.get("mode") == "complete":
                    print(f"[COMPLETE] supplemental release already manifested: {release_id}")
                    return 0
    manifest = {
        **plan,
        "mode": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "quoter_v2_address": quote_cfg["uniswap_v3_quoter_v2"]["address"],
        "quoter_deployment_source": quote_cfg["uniswap_v3_quoter_v2"]["source"],
        "extractor_sha256": extractor_hash,
        "files": file_manifest,
        "raw_frozen_manifest_modified": False,
        "note": "supplemental historical pool simulations; not executed swaps or realized prices",
    }
    # One JSONL row is the release commit marker. Monthly Parquet files are already
    # immutable and validated; no incomplete release is marked complete.
    from utils import append_jsonl
    append_jsonl(manifest_path, manifest)
    print(f"[COMPLETE] supplemental release={release_id}; manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

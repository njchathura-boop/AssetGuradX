"""Build a bounded, point-in-time graph release from processed Drive Parquet."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
import yaml

from assetguardx_graph.temporal import Entity, GraphRelease, TemporalRecord


def _utc(value, field: str) -> datetime:
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(stamp):
        raise ValueError(f"missing/invalid UTC timestamp in {field}")
    result = stamp.to_pydatetime()
    return result.replace(microsecond=0)


def _decimal(value, *, scale: int = 0) -> Decimal | None:
    if value is None or pd.isna(value):
        return None
    result = Decimal(str(value)) / (Decimal(10) ** scale)
    return result if result.is_finite() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _month_files(root: Path, dataset: str, start: datetime, end: datetime) -> list[Path]:
    directory = root / "processed" / dataset
    if not directory.exists():
        return []
    return [
        path
        for path in sorted(directory.glob("year=*/month=*/*.parquet"))
        if start.strftime("%Y-%m") <= path.stem[-7:] <= (end - timedelta(seconds=1)).strftime("%Y-%m")
    ]


def _read_window(root: Path, dataset: str, start: datetime, end: datetime, *, required=True):
    import pyarrow.parquet as pq

    paths = _month_files(root, dataset, start, end)
    if required and not paths:
        raise FileNotFoundError(f"processed/{dataset} is missing for the requested graph window")
    frames, provenance = [], []
    for path in paths:
        frame = pq.read_table(path).to_pandas()
        if "hour" not in frame:
            raise ValueError(f"processed table has no hour column: {path}")
        frame["hour"] = pd.to_datetime(frame["hour"], utc=True, errors="coerce")
        frame = frame.loc[(frame["hour"] >= start) & (frame["hour"] < end)].copy()
        if not frame.empty:
            frames.append(frame)
            provenance.append({
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
            })
    if not frames:
        if required:
            raise ValueError(f"processed/{dataset} contains no rows in the requested graph window")
        return pd.DataFrame(), provenance
    result = pd.concat(frames, ignore_index=True, sort=False)
    return result, provenance


def _evidence(dataset: str, source: dict, block: int | None = None) -> tuple[str, ...]:
    matches = [
        f"parquet:{item['path']}#sha256={item['sha256']}"
        for item in source.get(dataset, [])
    ]
    if block is not None:
        matches.append(f"ethereum:block={block}")
    if not matches:
        matches.append(f"processed_dataset:{dataset}")
    return tuple(matches)


def _quality(row: dict, *, missing=False, stale=False) -> str:
    if str(row.get("quality_status", "ok")) != "ok":
        return "source_failure"
    if missing:
        return "missing"
    if stale:
        return "stale"
    return "valid"


def _add(
    records: list[TemporalRecord], *, source: str, target: str, relation: str, view: str,
    observed: datetime, available: datetime, valid_until: datetime,
    amount: Decimal | None, unit: str, meaning: str, method: str,
    evidence: tuple[str, ...], quality: str = "valid", status: str = "observed",
    valuation: datetime | None = None,
) -> None:
    if available < observed:
        raise ValueError(f"future/invalid availability for {relation}: {observed} > {available}")
    records.append(TemporalRecord(
        source=source, target=target, relation=relation, view=view,
        observed_at=observed, available_at=available, valid_until=valid_until,
        valuation_at=valuation or observed, amount=amount, unit=unit,
        meaning=meaning, method=method, evidence=evidence, quality=quality, status=status,
    ))


def _entities(registry: dict, start: datetime, end: datetime) -> tuple[Entity, ...]:
    entities = [Entity(symbol, "asset", symbol) for symbol in registry["assets"]]
    entities.extend((Entity("USD", "reference_asset", "USD benchmark"),
                     Entity("BTC", "reference_asset", "BTC benchmark")))
    for key, meta in registry["contracts"].items():
        active_from = _utc(meta["active_from"], f"{key}.active_from") if meta.get("active_from") else None
        if active_from is not None and active_from >= end:
            continue
        kind = meta.get("state_kind")
        if kind in {"uniswap_v3_pool", "curve_pool"}:
            entity_kind, name = "dex_pool", meta.get("source", key)
        elif kind == "aave_v2_data_provider":
            continue
        elif kind == "chainlink_feed":
            entity_kind, name = "oracle_feed", meta.get("asset", key)
        elif kind == "lido_wsteth":
            entity_kind, name = "wrapping_contract", meta.get("source", key)
        elif kind in {"lido_steth", "lido_withdrawal_queue"}:
            entity_kind, name = "staking_contract", meta.get("source", key)
        else:
            continue
        entities.append(Entity(key, entity_kind, name))
    aave = next((v for v in registry["contracts"].values()
                 if v.get("state_kind") == "aave_v2_data_provider"), None)
    if aave:
        for reserve in aave.get("reserves", []):
            symbol = reserve["symbol"]
            key = f"aave_v2_{symbol}"
            entities.append(Entity(key, "lending_market", f"Aave V2 {symbol} reserve"))
    unique = {item.entity_id: item for item in entities}
    return tuple(unique[key] for key in sorted(unique))


def _topology(registry: dict, nodes: tuple[Entity, ...], start: datetime,
              end: datetime, evidence: tuple[str, ...]) -> list[TemporalRecord]:
    node_ids = {node.entity_id for node in nodes}
    records = []

    def edge(source: str, target: str, relation: str, meaning: str, active_from=None):
        if source not in node_ids or target not in node_ids:
            return
        observed = max(start, active_from) if active_from is not None else start
        if observed >= end:
            return
        _add(records, source=source, target=target, relation=relation, view="exposure",
             observed=observed, available=observed, valid_until=end, amount=Decimal(1),
             unit="structural_relation", meaning=meaning, method="versioned contract registry",
             evidence=evidence, status="reconstructed")

    for asset in registry["assets"]:
        if asset in {"USDC", "USDT", "DAI", "FRAX", "LUSD"}:
            edge(asset, "USD", "reference_link", "Stablecoin research benchmark is USD 1")
        elif asset in {"stETH", "wstETH", "WETH"}:
            edge(asset, "ETH", "reference_link", "Asset is benchmarked through ETH")
        elif asset == "WBTC":
            edge(asset, "BTC", "reference_link", "WBTC is benchmarked against BTC")

    for key, meta in registry["contracts"].items():
        kind = meta.get("state_kind")
        if kind == "uniswap_v3_pool":
            for asset in (meta["token0"], meta["token1"]):
                edge(asset, key, "traded_in_pool", f"{asset} is a token in the registered Uniswap V3 pool")
        elif kind == "curve_pool":
            for coin in meta.get("coins", []):
                edge(coin["symbol"], key, "pool_coin", f"{coin['symbol']} is a registered Curve pool coin")
            if meta.get("base_pool"):
                edge(meta["base_pool"], key, "curve_base_pool", "Curve metapool uses this registered base pool")
        elif kind == "aave_v2_data_provider":
            for reserve in meta.get("reserves", []):
                market = f"aave_v2_{reserve['symbol']}"
                edge(reserve["symbol"], market, "aave_reserve_asset", "Aave V2 reserve market for this asset")
        elif kind == "chainlink_feed":
            reference = "BTC" if str(meta.get("asset", "")).startswith("BTC") else "ETH"
            edge(key, reference, "provides_reference_price", f"Registered Chainlink {meta.get('asset')} feed")
        elif kind == "lido_steth":
            edge(key, "stETH", "accounts_for_staking_state", "Lido stETH accounting contract")
        elif kind == "lido_wsteth":
            edge(key, "wstETH", "accounts_for_wrapper_state", "Lido wstETH conversion contract")
            edge("wstETH", "stETH", "wraps_asset", "wstETH converts to stETH using the recorded historical rate")
        elif kind == "lido_withdrawal_queue":
            active_from = _utc(meta["active_from"], f"{key}.active_from") if meta.get("active_from") else None
            edge(key, "stETH", "withdrawal_queue_for",
                 "Lido withdrawal queue concerns stETH redemptions", active_from=active_from)
    return records


def _event_evidence(dataset: str, rows: list[dict], source_files: dict) -> tuple[str, ...]:
    fingerprints = sorted(
        str(row.get("source_event_fingerprint") or "")
        for row in rows
    )
    fingerprint_hash = hashlib.sha256("\n".join(fingerprints).encode()).hexdigest()
    tx_count = len({str(row.get("transaction_hash")) for row in rows if row.get("transaction_hash")})
    return (*_evidence(dataset, source_files), f"event_rows={len(rows)}",
            f"unique_transactions={tx_count}", f"event_fingerprints_sha256={fingerprint_hash}")


def _amount_from_raw(raw, decimals: int) -> Decimal | None:
    if raw is None or pd.isna(raw):
        return None
    return Decimal(str(raw)) / (Decimal(10) ** decimals)


def _uniswap_swap_flow(amount0: Decimal, amount1: Decimal, token0: str, token1: str):
    """Return the observed input/output direction without inferring zero output."""
    if amount0 > 0 and amount1 < 0:
        return token0, token1, amount0, -amount1
    if amount1 > 0 and amount0 < 0:
        return token1, token0, amount1, -amount0
    # Very small swaps can round the output down to zero in token base units.
    # Keep the actual pool input, but do not claim an output asset/amount.
    if amount0 > 0 and amount1 == 0:
        return token0, None, amount0, Decimal(0)
    if amount1 > 0 and amount0 == 0:
        return token1, None, amount1, Decimal(0)
    raise ValueError("Uniswap swap signed amounts do not identify a valid input/output flow")


def _add_swap_activity(records, table, source_files, snapshot_by_hour, next_by_hour, registry, known):
    if table.empty:
        return
    assets = registry["assets"]
    groups: dict[tuple, list[dict]] = {}
    for row in table.to_dict(orient="records"):
        if row.get("decode_ok") is False:
            raise ValueError("dex_swap_events contains a row marked decode_ok=false")
        hour = _utc(row.get("hour"), "dex_swap_events.hour")
        anchor = snapshot_by_hour.get(hour)
        if anchor is None:
            raise ValueError(f"DEX event has no hourly anchor: {hour.isoformat()}")
        event_at = _utc(row.get("block_timestamp"), "dex_swap_events.block_timestamp")
        if event_at < hour or event_at >= hour + timedelta(hours=1) or event_at > anchor:
            raise ValueError("DEX event timestamp is outside its hour or later than its hourly anchor")
        pool = str(row.get("pool") or row.get("contract_key"))
        if pool not in known:
            raise ValueError(f"DEX swap references unregistered pool {pool!r}")

        protocol = str(row.get("protocol", "")).lower()
        if protocol == "uniswap_v3":
            meta = registry["contracts"][pool]
            token0, token1 = meta["token0"], meta["token1"]
            if "amount0_raw" in row and "amount1_raw" in row:
                amount0 = _amount_from_raw(row.get("amount0_raw"), int(assets[token0]["decimals"]))
                amount1 = _amount_from_raw(row.get("amount1_raw"), int(assets[token1]["decimals"]))
            else:
                amount0, amount1 = _decimal(row.get("amount0")), _decimal(row.get("amount1"))
            if amount0 is None or amount1 is None:
                raise ValueError("Uniswap swap row is missing signed token amounts")
            sold_asset, bought_asset, sold_amount, bought_amount = _uniswap_swap_flow(
                amount0, amount1, token0, token1
            )
        else:
            sold_asset, bought_asset = row.get("sold_asset"), row.get("bought_asset")
            sold_asset, bought_asset = str(sold_asset), str(bought_asset)
            if sold_asset not in assets or bought_asset not in assets:
                raise ValueError("Curve swap references an asset absent from the registry")
            sold_amount = _amount_from_raw(
                row.get("tokens_sold_raw"), int(assets[sold_asset]["decimals"])
            )
            bought_amount = _amount_from_raw(
                row.get("tokens_bought_raw"), int(assets[bought_asset]["decimals"])
            )
            if sold_amount is None or bought_amount is None:
                raise ValueError("Curve swap row is missing sold/bought token amounts")

        if sold_asset not in known or (bought_asset is not None and bought_asset not in known):
            raise ValueError("DEX swap token endpoint is missing from the graph registry")
        key = (hour, pool, sold_asset, bought_asset)
        groups.setdefault(key, []).append({
            **row, "_sold_amount": sold_amount, "_bought_amount": bought_amount,
            "_event_at": event_at,
        })

    for (hour, pool, sold_asset, bought_asset), rows in groups.items():
        anchor = snapshot_by_hour[hour]
        evidence = _event_evidence("dex_swap_events", rows, source_files)
        sold_total = sum((row["_sold_amount"] for row in rows), Decimal(0))
        bought_total = sum((row["_bought_amount"] for row in rows), Decimal(0))
        common = {
            "observed": hour,
            "available": anchor,
            "valid_until": next_by_hour[hour],
            "evidence": evidence,
            "status": "derived",
        }
        if bought_asset is None:
            _add(records, source=sold_asset, target=pool,
                 relation="dex_swap_input_without_observed_output", view="exposure",
                 amount=sold_total, unit=sold_asset,
                 meaning="Observed positive Uniswap pool input; the other token amount rounded to zero",
                 method="aggregated decoded DEX swap events; output endpoint not inferred",
                 valuation=hour, **common)
            _add(records, source=pool, target=pool,
                 relation=f"dex_swap_unpaired_output_count_{sold_asset}", view="exposure",
                 amount=Decimal(len(rows)), unit="swap_events",
                 meaning="Decoded swap events with positive input and zero recorded output",
                 method="aggregated decoded DEX swap events", valuation=hour, **common)
            continue
        _add(records, source=sold_asset, target=pool,
             relation=f"dex_swap_volume_sold_to_{bought_asset}", view="exposure",
             amount=sold_total, unit=sold_asset,
             meaning=f"Hourly observed swap volume sold into {pool}",
             method="aggregated decoded DEX swap events", valuation=hour, **common)
        _add(records, source=pool, target=bought_asset,
             relation=f"dex_swap_volume_bought_from_{sold_asset}", view="exposure",
             amount=bought_total, unit=bought_asset,
             meaning=f"Hourly observed swap volume paid out by {pool}",
             method="aggregated decoded DEX swap events", valuation=hour, **common)
        _add(records, source=pool, target=pool,
             relation=f"dex_swap_count_{sold_asset}_to_{bought_asset}", view="exposure",
             amount=Decimal(len(rows)), unit="swap_events",
             meaning="Count of decoded swaps in this pool and direction during the UTC hour",
             method="aggregated decoded DEX swap events", valuation=hour, **common)


def _add_aave_liquidations(records, table, source_files, snapshot_by_hour, next_by_hour, registry, known):
    if table.empty:
        return
    liquidations = table.loc[
        table.get("event_signature", pd.Series(index=table.index, dtype="object"))
        .astype("string").str.startswith("LiquidationCall(", na=False)
    ]
    if liquidations.empty:
        return
    groups: dict[tuple, list[dict]] = {}
    for row in liquidations.to_dict(orient="records"):
        hour = _utc(row.get("hour"), "aave_events.hour")
        anchor = snapshot_by_hour.get(hour)
        if anchor is None:
            raise ValueError(f"Aave liquidation has no hourly anchor: {hour.isoformat()}")
        event_at = _utc(row.get("block_timestamp"), "aave_events.block_timestamp")
        if event_at < hour or event_at >= hour + timedelta(hours=1) or event_at > anchor:
            raise ValueError("Aave liquidation is outside its hour or later than its hourly anchor")
        collateral, debt = row.get("collateral_asset"), row.get("debt_asset")
        if collateral not in registry["assets"] or debt not in registry["assets"]:
            continue
        # The lending-pool contract emits events for every reserve. This graph
        # release intentionally models only the reserves registered in its
        # selected-market universe; unrelated reserves remain in the source
        # archive but are not graph edges.
        if f"aave_v2_{collateral}" not in known or f"aave_v2_{debt}" not in known:
            continue
        c_amount = _amount_from_raw(
            row.get("liquidatedCollateralAmount"),
            int(registry["assets"][collateral]["decimals"]),
        )
        d_amount = _amount_from_raw(
            row.get("debtToCover"),
            int(registry["assets"][debt]["decimals"]),
        )
        if c_amount is None:
            c_amount = _decimal(row.get("liquidated_collateral_normalized"))
        if d_amount is None:
            d_amount = _decimal(row.get("debt_to_cover_normalized"))
        groups.setdefault((hour, str(collateral), str(debt)), []).append({
            **row, "_collateral_amount": c_amount, "_debt_amount": d_amount,
        })

    for (hour, collateral, debt), rows in groups.items():
        collateral_market, debt_market = f"aave_v2_{collateral}", f"aave_v2_{debt}"
        evidence = _event_evidence("aave_events", rows, source_files)
        common = {
            "observed": hour,
            "available": snapshot_by_hour[hour],
            "valid_until": next_by_hour[hour],
            "evidence": evidence,
            "status": "derived",
        }
        collateral_total = sum(
            (row["_collateral_amount"] or Decimal(0) for row in rows), Decimal(0)
        )
        debt_total = sum((row["_debt_amount"] or Decimal(0) for row in rows), Decimal(0))
        q_collateral = "valid" if all(row["_collateral_amount"] is not None for row in rows) else "missing"
        q_debt = "valid" if all(row["_debt_amount"] is not None for row in rows) else "missing"
        _add(records, source=collateral_market, target=collateral,
             relation=f"aave_liquidated_collateral_for_{debt}", view="exposure",
             amount=collateral_total if q_collateral == "valid" else None, unit=collateral,
             meaning="Hourly collateral amount actually recorded in Aave liquidation events",
             method="aggregated decoded Aave V2 LiquidationCall events", valuation=hour,
             quality=q_collateral, **common)
        _add(records, source=debt_market, target=debt,
             relation=f"aave_debt_repaid_against_{collateral}", view="exposure",
             amount=debt_total if q_debt == "valid" else None, unit=debt,
             meaning="Hourly debt covered in Aave liquidation events",
             method="aggregated decoded Aave V2 LiquidationCall events", valuation=hour,
             quality=q_debt, **common)
        _add(records, source=debt_market, target=debt_market,
             relation=f"aave_liquidation_count_{collateral}_{debt}", view="exposure",
             amount=Decimal(len(rows)), unit="liquidation_events",
             meaning="Count of decoded Aave V2 liquidation events for this reserve pair",
             method="aggregated decoded Aave V2 LiquidationCall events", valuation=hour,
             **common)


def _add_aave_flows(records, table, source_files, snapshot_by_hour, next_by_hour, registry, known):
    if table.empty:
        return
    directions = {
        "Deposit(address,address,address,uint256,uint16)": ("deposit", "asset_to_market"),
        "Withdraw(address,address,address,uint256)": ("withdraw", "market_to_asset"),
        "Borrow(address,address,address,uint256,uint256,uint256,uint16)": ("borrow", "market_to_asset"),
        "Repay(address,address,address,uint256)": ("repay", "asset_to_market"),
    }
    action_rows = table.loc[table["event_signature"].isin(directions)]
    groups: dict[tuple, list[dict]] = {}
    for row in action_rows.to_dict(orient="records"):
        hour = _utc(row.get("hour"), "aave_events.hour")
        anchor = snapshot_by_hour.get(hour)
        if anchor is None:
            raise ValueError(f"Aave action has no hourly anchor: {hour.isoformat()}")
        event_at = _utc(row.get("block_timestamp"), "aave_events.block_timestamp")
        if event_at < hour or event_at >= hour + timedelta(hours=1) or event_at > anchor:
            raise ValueError("Aave action is outside its hour or later than its hourly anchor")
        action, direction = directions[row["event_signature"]]
        asset = row.get("asset")
        if asset not in registry["assets"]:
            continue
        market = f"aave_v2_{asset}"
        if market not in known:
            # Source logs cover the lending-pool contract, not just selected
            # markets. Keep out-of-scope reserves archived but out of this graph.
            continue
        groups.setdefault((hour, str(asset), action, direction), []).append(row)

    for (hour, asset, action, direction), rows in groups.items():
        market = f"aave_v2_{asset}"
        evidence = _event_evidence("aave_events", rows, source_files)
        decimals = int(registry["assets"][asset]["decimals"])
        amounts = [_amount_from_raw(row.get("amount"), decimals) for row in rows]
        amounts = [
            amount if amount is not None else _decimal(row.get("amount_normalized"))
            for amount, row in zip(amounts, rows)
        ]
        total = sum((amount or Decimal(0) for amount in amounts), Decimal(0))
        quality = "valid" if all(amount is not None for amount in amounts) else "missing"
        source, target = (asset, market) if direction == "asset_to_market" else (market, asset)
        common = {
            "observed": hour,
            "available": snapshot_by_hour[hour],
            "valid_until": next_by_hour[hour],
            "evidence": evidence,
            "status": "derived",
        }
        _add(records, source=source, target=target,
             relation=f"aave_{action}_volume", view="exposure",
             amount=total if quality == "valid" else None, unit=asset,
             meaning=f"Hourly amount in decoded Aave V2 {action} events",
             method="aggregated decoded Aave V2 action events", valuation=hour,
             quality=quality, **common)
        _add(records, source=market, target=market,
             relation=f"aave_{action}_count", view="exposure",
             amount=Decimal(len(rows)), unit="aave_events",
             meaning=f"Count of decoded Aave V2 {action} events for this reserve and hour",
             method="aggregated decoded Aave V2 action events", valuation=hour,
             **common)


def _verified_quote_manifest(quote_root: Path, release_id: str) -> dict:
    manifest_path = quote_root.parent.parent / "metadata" / "supplemental_quote_sensitivity.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"supplemental quote release manifest is missing: {manifest_path}")
    matches = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("release_id") == release_id and row.get("mode") == "complete":
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one complete quote release manifest for {release_id}")
    manifest = matches[0]
    if not manifest.get("files") or int(manifest.get("sample_every_hours", 0)) <= 0:
        raise ValueError("quote release manifest is missing file inventory or sampling interval")
    return manifest


def build_release(
    archive_root: Path, *, start_time: datetime, end_time: datetime,
    context_hours: int = 168, include_quote_root: Path | None = None,
    quote_release_id: str | None = None,
) -> GraphRelease:
    """Build a bounded complete graph window; ``start_time`` begins evaluation.

    The release includes preceding context hours for model lookbacks. A maximum
    30-day evaluation interval plus seven context days keeps the DB load bounded.
    """
    if start_time.tzinfo is None or end_time.tzinfo is None or start_time.utcoffset() != timedelta(0) or end_time.utcoffset() != timedelta(0):
        raise ValueError("graph release boundaries must be timezone-aware UTC")
    if start_time.minute or start_time.second or start_time.microsecond or end_time.minute or end_time.second or end_time.microsecond:
        raise ValueError("graph release boundaries must be aligned to UTC hours")
    if end_time <= start_time or (end_time - start_time) > timedelta(days=30):
        raise ValueError("evaluation window must be nonempty and at most 30 days")
    if type(context_hours) is not int or not 0 <= context_hours <= 24 * 7:
        raise ValueError("context_hours must be an integer from 0 through 168 (seven days)")
    window_start = start_time - timedelta(hours=context_hours)
    config_root = Path(__file__).resolve().parents[2] / "etl" / "config"
    registry_path = config_root / "contracts.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry_evidence = (
        f"registry:etl/config/contracts.yaml#sha256={_sha256(registry_path)}",
    )

    # Load anchors separately from raw, and fail when any requested UTC hour is absent.
    anchors, anchor_evidence = _read_raw_anchors(archive_root, window_start, end_time)
    source_files = {"hourly_blocks": anchor_evidence}
    expected = pd.date_range(window_start, end_time, freq="h", inclusive="left")
    actual = pd.DatetimeIndex(anchors["hour"])
    # Parquet commonly reads timestamps as datetime64[us], while date_range
    # defaults to datetime64[ns]. DatetimeIndex.equals() compares resolution
    # metadata too, so use timestamp value equality after checking lengths.
    if not _matches_expected_hour_grid(actual, expected):
        raise ValueError("raw hourly block anchors are incomplete, duplicated, or out of order")
    anchors["snapshot_block_timestamp"] = pd.to_datetime(anchors["block_timestamp"], utc=True)
    if (
        (anchors["snapshot_block_timestamp"] < anchors["hour"]).any()
        or (anchors["snapshot_block_timestamp"] >= anchors["hour"] + pd.Timedelta(hours=1)).any()
        or anchors["snapshot_block_timestamp"].is_monotonic_increasing is False
    ):
        raise ValueError("hourly block snapshots are outside their hour or not chronological")
    snapshot_by_hour = {
        row.hour.to_pydatetime(): row.snapshot_block_timestamp.to_pydatetime().replace(microsecond=0)
        for row in anchors.itertuples(index=False)
    }
    next_by_hour = {}
    ordered_hours = list(snapshot_by_hour)
    for index, hour in enumerate(ordered_hours):
        next_by_hour[hour] = snapshot_by_hour[ordered_hours[index + 1]] if index + 1 < len(ordered_hours) else end_time

    datasets = (
        "asset_hourly", "oracle_hourly", "uniswap_pool_state_hourly", "curve_pool_state_hourly",
        "aave_reserve_state_hourly", "lido_steth_state_hourly", "lido_wsteth_state_hourly",
        "lido_withdrawal_queue_hourly", "dex_swap_events", "aave_events",
    )
    tables = {}
    for dataset in datasets:
        table, evidence = _read_window(
            archive_root,
            dataset,
            window_start,
            end_time,
            required=dataset not in {"lido_withdrawal_queue_hourly", "dex_swap_events", "aave_events"},
        )
        tables[dataset] = table
        source_files[dataset] = evidence

    targets = [key for key, value in registry["assets"].items() if value.get("target_asset")]
    market = tables["asset_hourly"]
    if market.duplicated(["asset", "hour"]).any():
        raise ValueError("asset_hourly has duplicate asset-hours in the graph window")
    if set(market["asset"].unique()) != set(targets):
        raise ValueError("asset_hourly target universe disagrees with the versioned registry")
    expected_market_rows = len(targets) * len(expected)
    if len(market) != expected_market_rows:
        raise ValueError(f"asset_hourly coverage is incomplete: expected {expected_market_rows} rows, found {len(market)}")

    nodes = _entities(registry, window_start, end_time)
    known = {node.entity_id for node in nodes}
    records = _topology(registry, nodes, snapshot_by_hour[ordered_hours[0]], end_time, registry_evidence)

    def emit(row: dict, dataset: str, source: str, target: str, relation: str,
             value, unit: str, meaning: str, method: str, *, observed=None,
             quality=None, status="observed", scale=0):
        hour = _utc(row["hour"], f"{dataset}.hour")
        anchor = snapshot_by_hour.get(hour)
        if anchor is None:
            raise ValueError(f"no block anchor for {dataset} at {hour.isoformat()}")
        observed_at = _utc(observed, f"{dataset}.observed_at") if observed is not None else anchor
        val = _decimal(value, scale=scale)
        q = quality or _quality(row, missing=val is None)
        if q == "valid" and val is None:
            q = "missing"
        _add(records, source=source, target=target, relation=relation, view="exposure",
             observed=observed_at, available=anchor, valid_until=next_by_hour[hour],
             amount=val, unit=unit, meaning=meaning, method=method,
             evidence=_evidence(dataset, source_files, int(row["block_number"]) if "block_number" in row and pd.notna(row["block_number"]) else None),
             quality=q, status=status, valuation=observed_at)

    asset_feature_spec = (
        ("market_price_usd", "USD_per_asset", "Observed or explicitly stale market price"),
        ("reference_price_usd", "USD_per_asset", "Asset-specific reference benchmark"),
        ("relative_deviation", "fraction", "Signed deviation from the configured reference"),
        ("market_price_age_hours", "hours", "Age of the market-price source observation"),
        ("event_count", "asset_events", "Count of selected token activity events"),
        ("mint_count", "asset_events", "Count of selected token mints"),
        ("burn_count", "asset_events", "Count of selected token burns"),
        ("transfer_count", "asset_events", "Count of selected token transfers"),
        ("minted", "asset", "Hourly mint amount from selected token activity"),
        ("burned", "asset", "Hourly burn amount from selected token activity"),
        ("transferred", "asset", "Hourly selected token transfer amount"),
        ("net_issuance", "asset", "Mints minus burns in native token units"),
    )
    for row in market.to_dict(orient="records"):
        asset, hour = row["asset"], _utc(row["hour"], "asset_hourly.hour")
        if asset not in known:
            raise ValueError(f"unknown asset in asset_hourly: {asset}")
        forward_filled = row.get("market_price_forward_filled", False)
        is_stale = (False if pd.isna(forward_filled) else bool(forward_filled)) or bool(
            pd.notna(row.get("market_price_age_hours")) and float(row["market_price_age_hours"]) > 0
        )
        source_hour = row.get("market_observed_hour")
        market_observed_at = None
        if pd.notna(source_hour):
            source_hour = _utc(source_hour, "asset_hourly.market_observed_hour")
            market_observed_at = snapshot_by_hour.get(source_hour, source_hour)
        for field, unit, meaning in asset_feature_spec:
            if field not in row:
                continue
            q = _quality(row, missing=pd.isna(row.get(field)), stale=is_stale and field == "market_price_usd")
            emit(row, "asset_hourly", asset, asset, f"state_{field}", row.get(field), unit,
                 meaning, "canonical hourly table", quality=q,
                 observed=market_observed_at if field == "market_price_usd" else None)
        src = row.get("market_source")
        if isinstance(src, str) and pd.notna(row.get("market_price_usd")):
            for candidate in src.split("|"):
                pool = candidate.split(":", 1)[0]
                if pool in known:
                    emit(row, "asset_hourly", pool, asset, "market_quote_for_asset",
                         row["market_price_usd"], "USD_per_asset",
                         "Canonical market-price observation attributed to its registered pool source",
                         "canonical hourly table; source column retained", quality="stale" if is_stale else "valid",
                         observed=market_observed_at)

    # Keep actual DEX activity, but aggregate same-hour swaps by pool and token
    # direction. This is far smaller than putting every Ethereum log in the DB,
    # while retaining volume, count, exact temporal availability, and evidence.
    _add_swap_activity(
        records,
        tables["dex_swap_events"],
        source_files,
        snapshot_by_hour,
        next_by_hour,
        registry,
        known,
    )

    # Liquidation events are useful system-level stress indicators. These are
    # event aggregates, not reconstructed borrower positions or account health.
    _add_aave_liquidations(
        records,
        tables["aave_events"],
        source_files,
        snapshot_by_hour,
        next_by_hour,
        registry,
        known,
    )
    _add_aave_flows(
        records,
        tables["aave_events"],
        source_files,
        snapshot_by_hour,
        next_by_hour,
        registry,
        known,
    )

    # Pool state nodes retain native-unit liquidity/state; no USD exposure is inferred.
    for dataset in ("uniswap_pool_state_hourly", "curve_pool_state_hourly"):
        table = tables[dataset]
        if table.empty:
            continue
        key_col = "entity_key"
        if table.duplicated([key_col, "hour"]).any():
            raise ValueError(f"{dataset} has duplicate pool-hours")
        for row in table.to_dict(orient="records"):
            pool = row[key_col]
            if pool not in known:
                raise ValueError(f"unregistered pool in {dataset}: {pool}")
            if dataset == "uniswap_pool_state_hourly":
                measures = (
                    ("active_liquidity_raw", "uniswap_L_raw", "Uniswap V3 active liquidity state"),
                    ("sqrt_price_x96", "sqrt_price_x96", "Uniswap V3 pool square-root price state"),
                    ("tick", "tick", "Uniswap V3 current tick"),
                )
            else:
                measures = (("A_raw", "integer", "Curve amplification parameter"),
                            ("virtual_price_raw", "fixed_point_1e18", "Curve pool virtual price; not an external USD price"))
                meta = registry["contracts"][pool]
                for index, coin in enumerate(meta.get("coins", [])):
                    field = f"balance_{index}_raw"
                    if field in row:
                        measures += ((field, coin["symbol"], f"Curve pool balance of {coin['symbol']}"),)
            for field, unit, meaning in measures:
                if field in row:
                    emit(row, dataset, pool, pool, f"state_{field}", row.get(field), unit,
                         meaning, "historical eth_call state snapshot", scale=18 if field == "virtual_price_raw" else 0)

    aave = tables["aave_reserve_state_hourly"]
    if aave.duplicated(["entity_key", "hour"]).any():
        raise ValueError("aave_reserve_state_hourly has duplicate reserve-hours")
    for row in aave.to_dict(orient="records"):
        market_id, asset = row["entity_key"], row["asset"]
        decimals = int(row["decimals"])
        for field, relation, meaning in (
            ("available_liquidity_raw", "reserve_available_cash", "Aave reserve cash, not borrower collateral"),
            ("total_a_token_raw", "reserve_supplied", "Aggregate Aave aToken supply"),
            ("total_stable_debt_raw", "reserve_stable_debt", "Aggregate Aave stable debt"),
            ("total_variable_debt_raw", "reserve_variable_debt", "Aggregate Aave variable debt"),
        ):
            emit(row, "aave_reserve_state_hourly", market_id, asset, relation, row.get(field), asset,
                 meaning, "historical_rpc_erc20_total_supply_or_balance", scale=decimals)
        cash = _decimal(row.get("available_liquidity_raw"), scale=decimals)
        debt = (_decimal(row.get("total_stable_debt_raw"), scale=decimals) or Decimal(0)) + \
               (_decimal(row.get("total_variable_debt_raw"), scale=decimals) or Decimal(0))
        utilization = debt / (cash + debt) if cash is not None and cash + debt else None
        emit(row, "aave_reserve_state_hourly", market_id, market_id, "state_utilisation",
             utilization, "fraction", "Aggregate debt divided by available cash plus debt",
             "derived_from_historical_reserve_state", quality="valid" if utilization is not None else "missing")
        for field, unit, meaning in (
            ("ltv", "basis_points", "Aave reserve maximum loan-to-value parameter"),
            ("liquidation_threshold", "basis_points", "Aave reserve liquidation threshold parameter"),
            ("liquidation_bonus", "basis_points", "Aave reserve liquidation bonus parameter"),
            ("reserve_factor", "basis_points", "Aave reserve protocol fee parameter"),
        ):
            if field in row:
                emit(row, "aave_reserve_state_hourly", market_id, market_id,
                     f"state_{field}", row.get(field), unit, meaning,
                     "historical Aave V2 reserve configuration")
        for field, meaning in (
            ("usage_as_collateral_enabled", "Whether this reserve was enabled as collateral"),
            ("borrowing_enabled", "Whether borrowing this reserve was enabled"),
            ("stable_borrow_rate_enabled", "Whether stable-rate borrowing was enabled"),
            ("is_active", "Whether the Aave reserve was active"),
            ("is_frozen", "Whether the Aave reserve was frozen"),
        ):
            if field not in row:
                continue
            raw_flag = row.get(field)
            flag = None if raw_flag is None or pd.isna(raw_flag) else Decimal(int(bool(raw_flag)))
            emit(row, "aave_reserve_state_hourly", market_id, market_id,
                 f"state_{field}", flag, "boolean_0_or_1", meaning,
                 "historical Aave V2 reserve configuration")

    for dataset, value_fields in (
        ("lido_steth_state_hourly", (
            ("total_pooled_ether_raw", 18, "ETH", "Lido total pooled ether"),
            ("total_shares_raw", 18, "stETH_share", "Lido total share count"),
            ("total_supply_raw", 18, "stETH", "Lido stETH total supply"),
        )),
        ("lido_wsteth_state_hourly", (
            ("steth_per_wsteth_raw", 18, "stETH_per_wstETH", "Historical wstETH conversion rate"),
        )),
    ):
        for row in tables[dataset].to_dict(orient="records"):
            contract = row["entity_key"]
            asset = row.get("asset") or ("stETH" if "steth" in dataset else "wstETH")
            for field, scale, unit, meaning in value_fields:
                emit(row, dataset, contract, asset, f"state_{field}", row.get(field), unit,
                     meaning, "historical_lido_contract_state", scale=scale)

    for row in tables["lido_withdrawal_queue_hourly"].to_dict(orient="records"):
        contract = row["entity_key"]
        for field, scale, unit, meaning in (
            ("locked_ether_raw", 18, "ETH", "ETH currently locked for finalized Lido withdrawals"),
            ("unfinalized_steth_raw", 18, "stETH", "stETH backing requests not yet finalized"),
            ("unfinalized_request_number", 0, "withdrawal_requests", "Number of unfinalized Lido withdrawal requests"),
            ("last_request_id", 0, "request_id", "Latest submitted Lido withdrawal request ID"),
            ("last_finalized_request_id", 0, "request_id", "Latest finalized Lido withdrawal request ID"),
        ):
            if field in row:
                emit(row, "lido_withdrawal_queue_hourly", contract, "stETH",
                     f"state_{field}", row.get(field), unit, meaning,
                     "historical Lido withdrawal queue state", scale=scale)
        raw_bunker = row.get("is_bunker_mode")
        if raw_bunker is not None and not pd.isna(raw_bunker):
            emit(row, "lido_withdrawal_queue_hourly", contract, contract,
                 "state_is_bunker_mode", Decimal(int(bool(raw_bunker))), "boolean_0_or_1",
                 "Whether Lido withdrawal queue was in bunker mode",
                 "historical Lido withdrawal queue state")

    for row in tables["oracle_hourly"].to_dict(orient="records"):
        feed = row["entity_key"]
        reference = "BTC" if str(row.get("asset", "")).startswith("BTC") else "ETH"
        emit(row, "oracle_hourly", feed, reference, "oracle_reference_price_usd",
             row.get("reference_price"), "USD_per_reference_asset",
             "Chainlink feed answer with historical update time", "historical_rpc_latestRoundData")
        updated = row.get("oracle_updated_at_unix")
        if pd.notna(updated):
            snapshot_at = _utc(row["snapshot_block_timestamp"], "oracle snapshot")
            age = Decimal(str(snapshot_at.timestamp())) - Decimal(str(updated))
            if age < 0:
                raise ValueError("Chainlink updated_at is later than the historical block snapshot")
            emit(row, "oracle_hourly", feed, feed, "state_oracle_age_seconds", age, "seconds",
                 "Age of feed answer at its actual historical block", "snapshot_timestamp_minus_updated_at")

    # Aave aggregate values and pool liquidity are not account positions, realized
    # losses, or sufficient evidence of borrower liquidation health.
    quote_provenance = None
    if include_quote_root and quote_release_id:
        quote_manifest = _verified_quote_manifest(include_quote_root, quote_release_id)
        quote_files = sorted((include_quote_root / f"release_id={quote_release_id}").glob("year=*/month=*/*.parquet"))
        if not quote_files:
            raise FileNotFoundError(f"quote sensitivity release not found: {quote_release_id}")
        expected_quote_hashes = {
            str(Path(item["path"]).resolve()): item["sha256"]
            for item in quote_manifest.get("files", [])
        }
        for quote_file in quote_files:
            expected_hash = expected_quote_hashes.get(str(quote_file.resolve()))
            if expected_hash is None or _sha256(quote_file) != expected_hash:
                raise ValueError(f"quote release manifest/checksum mismatch: {quote_file}")
        source_files["quote_sensitivity"] = [
            {"path": path.relative_to(include_quote_root).as_posix(), "sha256": _sha256(path)}
            for path in quote_files
        ]
        quote_provenance = quote_release_id
        quote_frames = [pd.read_parquet(path) for path in quote_files]
        quotes = pd.concat(quote_frames, ignore_index=True)
        quotes["hour"] = pd.to_datetime(quotes["hour"], utc=True)
        quotes = quotes.loc[(quotes["hour"] >= window_start) & (quotes["hour"] < end_time)]
        for row in quotes.to_dict(orient="records"):
            pool, token_in, token_out = row["contract_key"], row["input_asset"], row["output_asset"]
            if pool not in known or token_in not in known or token_out not in known:
                raise ValueError("quote sensitivity contains an unregistered pool/token")
            observed = _utc(row["snapshot_block_timestamp"], "quote snapshot")
            _utc(row["hour"], "quote hour")
            amount_in = row["amount_in"]
            relation = f"quote_via_{pool}_{token_out}_per_{token_in}_for_{amount_in}"
            quote_status = row["quote_status"]
            valid = quote_status in {"quoted", "zero_output"}
            output_ratio = _decimal(row.get("output_per_input"))
            valid_until = min(observed + timedelta(hours=int(quote_manifest["sample_every_hours"])), end_time)
            _add(records, source=token_in, target=token_out, relation=relation, view="exposure",
                 observed=observed, available=observed, valid_until=valid_until,
                 amount=output_ratio, unit=f"{token_out}_per_{token_in}",
                 meaning=f"Read-only exact-input quote via {pool}: {amount_in} {token_in} to {token_out}",
                 method=f"historical_{row['call_kind']}",
                 evidence=_evidence_for_quote_row(quote_release_id, row),
                 quality="valid" if valid else "unsupported", status="derived")

    source_identity = json.dumps({
        "start": window_start.isoformat(), "end": end_time.isoformat(),
        "sources": source_files,
    }, sort_keys=True)
    release_id = "historical-" + hashlib.sha256(source_identity.encode()).hexdigest()[:20]
    return GraphRelease(
        release_id=release_id,
        start=snapshot_by_hour[ordered_hours[0]],
        end=end_time,
        nodes=nodes,
        records=tuple(records),
        provenance=json.dumps({
            "archive_root_name": archive_root.name,
            "start_time": start_time.isoformat(),
            "end_time_exclusive": end_time.isoformat(),
            "context_hours": context_hours,
            "source_file_hashes": source_files,
            "registry_sha256": _sha256(registry_path),
            "quote_release_id": quote_provenance,
        }, sort_keys=True),
    )


def _read_raw_anchors(root: Path, start: datetime, end: datetime):
    frames, provenance = [], []
    directory = root / "raw" / "hourly_blocks"
    for path in sorted(directory.glob("year=*/month=*/*.parquet")):
        frame = pd.read_parquet(path, columns=["hour", "block_number", "block_timestamp"])
        frame["hour"] = pd.to_datetime(frame["hour"], utc=True, errors="coerce")
        frame = frame.loc[(frame["hour"] >= start) & (frame["hour"] < end)].copy()
        if not frame.empty:
            frames.append(frame)
            provenance.append({"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)})
    if not frames:
        raise FileNotFoundError("raw/hourly_blocks has no rows for the requested graph window")
    return pd.concat(frames, ignore_index=True).sort_values("hour").reset_index(drop=True), provenance


def _matches_expected_hour_grid(actual: pd.DatetimeIndex, expected: pd.DatetimeIndex) -> bool:
    """Compare UTC hour values independently of pandas timestamp resolution."""
    return (
        not actual.has_duplicates
        and len(actual) == len(expected)
        and bool((actual == expected).all())
    )


def _evidence_for_quote_row(release_id: str, row: dict) -> tuple[str, ...]:
    return (
        f"quote_sensitivity_release:{release_id}",
        f"ethereum:block={int(row['block_number'])}",
        f"quote_key={row['quote_key']}",
    )



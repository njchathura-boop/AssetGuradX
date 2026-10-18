"""Build a bounded, real Aave V2 reserve sample for the existing graph schema.

This is a loader/integration fixture, not a validated systemic-risk graph.
The only economic quantity exported is observed reserve cash in native token
units. No account-level positions, liquidation estimates, or USD exposures are
implied.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, localcontext
from pathlib import Path

import pyarrow.parquet as pq

from assetguardx_graph.temporal import Entity, GraphRelease, TemporalRecord


_COLUMNS = (
    "hour",
    "snapshot_block_timestamp",
    "block_number",
    "entity_key",
    "state_kind",
    "protocol",
    "asset",
    "quality_status",
    "underlying_address",
    "token_decimals",
    "decimals",
    "available_liquidity_raw",
)


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _utc(value: datetime, name: str) -> datetime:
    if value is None or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be a timezone-aware UTC timestamp")
    if value.microsecond:
        raise ValueError(f"{name} must have whole-second precision")
    return value.astimezone(timezone.utc)


def build_aave_sample(
    archive_root: Path,
    *,
    first_day: date,
    hours: int = 24,
    assets: tuple[str, ...] = ("USDC", "DAI", "WBTC"),
) -> GraphRelease:
    """Convert only selected hourly reserve cash observations, failing on gaps."""
    if type(hours) is not int or not 1 <= hours <= 168:
        raise ValueError("sample hours must be between 1 and 168")
    if not assets or len(set(assets)) != len(assets):
        raise ValueError("select at least one unique asset")
    start = datetime.combine(first_day, datetime.min.time(), tzinfo=timezone.utc)
    end = start + timedelta(hours=hours)
    if (start.year, start.month) != ((end - timedelta(seconds=1)).year, (end - timedelta(seconds=1)).month):
        raise ValueError("sample must fit in one monthly Parquet file")

    relative = Path(
        "processed/aave_reserve_state_hourly"
    ) / f"year={start.year:04d}" / f"month={start.month:02d}" / (
        f"aave_reserve_state_hourly_{start.year:04d}-{start.month:02d}.parquet"
    )
    source = archive_root / relative
    if not source.is_file():
        raise FileNotFoundError(f"processed Aave reserve file not found: {source}")
    parquet = pq.ParquetFile(source)
    missing = set(_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Aave reserve schema missing: {', '.join(sorted(missing))}")
    checksum = _sha256(source)
    selected = set(assets)
    rows: dict[tuple[str, datetime], dict] = {}
    for batch in parquet.iter_batches(columns=list(_COLUMNS), batch_size=4096):
        for row in batch.to_pylist():
            if row["asset"] not in selected:
                continue
            hour = _utc(row["hour"], "hour")
            if not start <= hour < end:
                continue
            if hour.minute or hour.second:
                raise ValueError(f"non-hourly Aave timestamp for {row['asset']}")
            key = row["asset"], hour
            if key in rows:
                raise ValueError(f"duplicate Aave reserve observation: {key}")
            rows[key] = row

    expected = {(asset, start + timedelta(hours=i)) for asset in assets for i in range(hours)}
    absent = expected - rows.keys()
    if absent:
        raise ValueError(f"sample has {len(absent)} missing asset-hours; first: {min(absent)}")

    nodes = tuple(
        node
        for asset in sorted(selected)
        for node in (
            Entity(asset, "asset", asset),
            Entity(f"aave_v2_{asset}", "lending_market", f"Aave V2 {asset} reserve"),
        )
    )
    records = []
    for asset in sorted(selected):
        for i in range(hours):
            hour = start + timedelta(hours=i)
            row = rows[(asset, hour)]
            if (
                row["quality_status"] != "ok"
                or row["protocol"] != "aave_v2"
                or row["state_kind"] != "aave_v2_reserve"
                or row["entity_key"] != f"aave_v2_{asset}"
            ):
                raise ValueError(f"unverified/invalid Aave reserve state for {asset} at {hour}")
            observed = _utc(row["snapshot_block_timestamp"], "snapshot_block_timestamp")
            if not hour <= observed < hour + timedelta(hours=1):
                raise ValueError(f"snapshot time outside its hour for {asset} at {hour}")
            decimals = int(row["decimals"])
            if not 0 <= decimals <= 36 or row["token_decimals"] != decimals:
                raise ValueError(f"inconsistent token decimals for {asset} at {hour}")
            raw = row["available_liquidity_raw"]
            if not isinstance(raw, str) or not raw.isdecimal():
                raise ValueError(f"invalid reserve cash for {asset} at {hour}")
            if type(row["block_number"]) is not int or row["block_number"] < 1:
                raise ValueError(f"invalid source block for {asset} at {hour}")
            if not isinstance(row["underlying_address"], str) or not row["underlying_address"].startswith("0x"):
                raise ValueError(f"missing underlying address for {asset} at {hour}")
            with localcontext() as ctx:
                ctx.prec = max(50, len(raw) + decimals + 4)
                amount = Decimal(raw).scaleb(-decimals)
            if i + 1 < hours:
                following = _utc(
                    rows[(asset, hour + timedelta(hours=1))]["snapshot_block_timestamp"],
                    "next snapshot_block_timestamp",
                )
            else:
                following = observed + timedelta(hours=1)
            if following <= observed:
                raise ValueError(f"non-increasing source snapshots for {asset}")
            records.append(
                TemporalRecord(
                    source=asset,
                    target=f"aave_v2_{asset}",
                    relation="reserve_available_liquidity",
                    view="exposure",
                    observed_at=observed,
                    available_at=observed,
                    valid_until=following,
                    valuation_at=observed,
                    amount=amount,
                    unit=asset,
                    meaning="Aave V2 reserve cash available at its aToken contract; not account exposure or USD loss",
                    method="historical_rpc_erc20_balance_of_a_token",
                    evidence=(
                        f"parquet:{relative.as_posix()}#sha256={checksum}",
                        f"ethereum:block={row['block_number']}",
                        f"underlying:{row['underlying_address'].lower()}",
                    ),
                    quality="valid",
                    status="observed",
                )
            )
    asset_slug = "-".join(asset.lower() for asset in sorted(selected))
    return GraphRelease(
        release_id=f"real-aave-v2-sample-{first_day:%Y%m%d}-{hours}h-{asset_slug}-{checksum[:12]}",
        start=start,
        end=end,
        nodes=nodes,
        records=tuple(records),
        provenance=(
            f"REAL processed Aave V2 historical RPC sample; {relative.as_posix()}; "
            f"sha256={checksum}; reserve cash only; not a validated stress graph"
        ),
    )

"""Explicitly synthetic forecasting fixtures; never thesis results."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from assetguardx_graph.temporal import Entity, GraphRelease, TemporalRecord


def synthetic_forecast_release(hours=336):
    """Repeated artificial episodes exercise the trainer; no financial evidence."""
    from .forecast_data import ASSETS

    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    nodes = tuple(Entity(asset, "asset", asset + " (synthetic)") for asset in ASSETS) + (
        Entity("synthetic-pool", "dex_pool", "Synthetic test pool"),)
    records = []
    for offset in range(hours):
        time = start + timedelta(hours=offset, minutes=59, seconds=59)
        for asset in ASSETS:
            reference = 2000 if asset == "stETH" else 2300 if asset == "wstETH" else 30000 if asset == "WBTC" else 1
            deviation = -0.04 if 32 <= offset % 48 < 37 else 0.0001
            values = ((asset, asset, "state_market_price_usd", reference * (1 + deviation), "USD_per_asset"),
                      (asset, asset, "state_reference_price_usd", reference, "USD_per_asset"),
                      (asset, asset, "state_relative_deviation", deviation, "fraction"),
                      (asset, asset, "state_market_price_age_hours", 0, "hours"),
                      (asset, "synthetic-pool", "traded_in_pool", 1, "structural_relation"))
            for source, target, relation, amount, unit in values:
                records.append(TemporalRecord(source, target, relation, "exposure", time, time,
                    time + timedelta(hours=1), time, Decimal(str(amount)), unit,
                    "Artificial engineering observation", "synthetic_fixture", (f"synthetic:{offset}:{asset}",), status="derived"))
        records.append(TemporalRecord("synthetic-pool", "synthetic-pool", "state_liquidity", "exposure",
            time, time, time + timedelta(hours=1), time, Decimal(1000 - 10 * (offset % 48)),
            "synthetic_units", "Artificial liquidity", "synthetic_fixture", (f"synthetic:{offset}",), status="derived"))
    return GraphRelease("synthetic-forecast-smoke-v1", start + timedelta(minutes=59, seconds=59),
                        start + timedelta(hours=hours), nodes, tuple(records), "SYNTHETIC ENGINEERING ONLY; no research results")

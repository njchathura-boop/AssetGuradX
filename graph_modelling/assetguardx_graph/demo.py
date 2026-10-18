"""Explicitly synthetic graph fixture for offline engineering demonstrations."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from assetguardx_graph.temporal import Entity, GraphRelease, TemporalRecord


def synthetic_release():
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=7)
    records = []
    for offset in range(7 * 24):
        stamp = start + timedelta(hours=offset)
        records.append(
            TemporalRecord(
                source="USDC",
                target="synthetic-pool",
                relation="pool_inventory",
                view="exposure",
                observed_at=stamp,
                available_at=stamp,
                valid_until=stamp + timedelta(hours=1),
                valuation_at=stamp,
                amount=Decimal(1000 + offset),
                unit="USDC",
                meaning="Synthetic inventory, not financial evidence",
                method="engineering_fixture",
                evidence=(f"synthetic:hour:{offset}",),
                status="derived",
            )
        )
    return GraphRelease(
        "synthetic-seven-day-demo-v1",
        start,
        end,
        (
            Entity("USDC", "asset", "USDC (synthetic observations)"),
            Entity("synthetic-pool", "dex_pool", "Synthetic pool - not an Ethereum contract"),
        ),
        tuple(records),
        "SYNTHETIC ENGINEERING FIXTURE. Not downloaded data or thesis results.",
    )



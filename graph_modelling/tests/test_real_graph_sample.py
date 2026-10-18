from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from assetguardx_graph.temporal import GraphRelease, canonical
from assetguardx_graph.real_sample import build_aave_sample


START = datetime(2023, 1, 1, tzinfo=timezone.utc)


def source_file(tmp_path, *, missing_second=False, bad_quality=False, duplicate=False):
    path = (
        tmp_path
        / "processed"
        / "aave_reserve_state_hourly"
        / "year=2023"
        / "month=01"
        / "aave_reserve_state_hourly_2023-01.parquet"
    )
    path.parent.mkdir(parents=True)
    rows = []
    for i in range(2 if not missing_second else 1):
        hour = START + timedelta(hours=i)
        rows.append(
            {
                "hour": hour,
                "snapshot_block_timestamp": hour + timedelta(minutes=59, seconds=59),
                "block_number": 16308487 + i,
                "entity_key": "aave_v2_USDC",
                "state_kind": "aave_v2_reserve",
                "protocol": "aave_v2",
                "asset": "USDC",
                "quality_status": "bad" if bad_quality else "ok",
                "underlying_address": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
                "token_decimals": 6,
                "decimals": "6",
                "available_liquidity_raw": str(123456789 + i),
            }
        )
    if duplicate:
        rows.append(dict(rows[0]))
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_real_sample_is_bounded_traceable_and_point_in_time(tmp_path):
    path = source_file(tmp_path)
    release = build_aave_sample(tmp_path, first_day=date(2023, 1, 1), hours=2, assets=("USDC",))
    assert len(release.nodes) == 2
    assert len(release.records) == 2
    assert release.records[0].amount == Decimal("123.456789")
    assert release.records[0].unit == "USDC"
    assert release.records[0].evidence[0].startswith("parquet:processed/")
    assert "sha256=" in release.records[0].evidence[0]
    assert release.records[0].evidence[1] == "ethereum:block=16308487"
    assert release.graph_at(START, "exposure").records == ()
    assert release.graph_at(START + timedelta(hours=1), "exposure").records == (
        release.records[0],
    )
    assert release.graph_at(START + timedelta(hours=1, minutes=59, seconds=59), "exposure").records == (
        release.records[1],
    )
    assert GraphRelease.from_dict(__import__("json").loads(canonical(release.to_dict()))) == release
    assert path.is_file()


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"missing_second": True}, "missing asset-hours"),
        ({"bad_quality": True}, "unverified/invalid"),
        ({"duplicate": True}, "duplicate"),
    ],
)
def test_real_sample_rejects_incomplete_or_invalid_source(tmp_path, changes, match):
    source_file(tmp_path, **changes)
    with pytest.raises(ValueError, match=match):
        build_aave_sample(tmp_path, first_day=date(2023, 1, 1), hours=2, assets=("USDC",))


def test_real_sample_caps_size_and_rejects_month_crossing(tmp_path):
    with pytest.raises(ValueError, match="between 1 and 168"):
        build_aave_sample(tmp_path, first_day=date(2023, 1, 1), hours=169)
    with pytest.raises(ValueError, match="one monthly"):
        build_aave_sample(tmp_path, first_day=date(2023, 1, 31), hours=48)

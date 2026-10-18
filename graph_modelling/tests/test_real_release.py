import pandas as pd
import pytest
from decimal import Decimal

from assetguardx_graph.real_release import (
    _matches_expected_hour_grid,
    _uniswap_swap_flow,
)


def test_hour_grid_comparison_ignores_parquet_timestamp_resolution():
    expected = pd.date_range(
        "2023-01-01T00:00:00Z",
        "2023-01-01T03:00:00Z",
        freq="h",
        inclusive="left",
    )
    parquet_precision = expected.as_unit("us")

    assert parquet_precision.dtype != expected.dtype
    assert not parquet_precision.equals(expected)
    assert _matches_expected_hour_grid(parquet_precision, expected)
    assert not _matches_expected_hour_grid(parquet_precision[:-1], expected)
    assert not _matches_expected_hour_grid(
        parquet_precision.insert(1, parquet_precision[0]), expected
    )


def test_zero_output_uniswap_dust_is_kept_without_inventing_output_asset():
    assert _uniswap_swap_flow(Decimal("0"), Decimal("0.000000000000001483"), "USDC", "WETH") == (
        "WETH", None, Decimal("0.000000000000001483"), Decimal(0)
    )


def test_uniswap_swap_with_contradictory_signs_is_rejected():
    with pytest.raises(ValueError, match="valid input/output flow"):
        _uniswap_swap_flow(Decimal("1"), Decimal("2"), "USDC", "WETH")

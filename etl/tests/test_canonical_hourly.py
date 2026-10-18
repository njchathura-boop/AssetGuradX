import unittest
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from build_canonical_hourly import oracle_series, standardize_canonical_schema  # noqa: E402


class OracleFreshnessTests(unittest.TestCase):
    def test_bunker_mode_is_nullable_boolean_even_before_queue_activation(self):
        early = standardize_canonical_schema(pd.DataFrame({'is_bunker_mode': [None]}))
        late = standardize_canonical_schema(pd.DataFrame({'is_bunker_mode': [True]}))
        self.assertEqual(str(early['is_bunker_mode'].dtype), 'boolean')
        self.assertEqual(pa.Table.from_pandas(early).schema.field('is_bunker_mode').type, pa.bool_())
        self.assertEqual(pa.Table.from_pandas(late).schema.field('is_bunker_mode').type, pa.bool_())
        self.assertTrue(pd.isna(early.loc[0, 'is_bunker_mode']))

    def test_oracle_age_uses_microsecond_parquet_timestamp_units(self):
        # Parquet commonly loads UTC timestamps at microsecond resolution.
        snapshot = pd.Series(pd.to_datetime(['2023-01-01T00:59:59Z'], utc=True))
        snapshot = snapshot.astype('datetime64[us, UTC]')
        oracle = pd.DataFrame({
            'entity_key': ['chainlink_eth_usd'],
            'hour': pd.to_datetime(['2023-01-01T00:00:00Z'], utc=True),
            'reference_price': [1194.2699963],
            'oracle_updated_at_unix': [1672534535],  # 2023-01-01 00:55:35 UTC
            'snapshot_block_timestamp': snapshot,
        })

        result = oracle_series(oracle, 'chainlink_eth_usd', 'eth')

        self.assertEqual(result.loc[0, 'eth_oracle_age_seconds'], 264)

    def test_legacy_oracle_age_uses_end_of_hour_anchor_approximation(self):
        oracle = pd.DataFrame({
            'entity_key': ['chainlink_eth_usd'],
            'hour': pd.Series(pd.to_datetime(['2023-01-01T00:00:00Z'], utc=True))
                .astype('datetime64[us, UTC]'),
            'reference_price': [1194.2699963],
            'oracle_updated_at_unix': [1672534535],  # 2023-01-01 00:55:35 UTC
        })

        result = oracle_series(oracle, 'chainlink_eth_usd', 'eth')

        self.assertEqual(result.loc[0, 'eth_oracle_age_seconds'], 264)


if __name__ == '__main__':
    unittest.main()

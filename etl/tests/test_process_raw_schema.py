import sys
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from process_raw import EVENT_BASE_COLUMNS, EVENT_EXTRA_COLUMNS, standardize_event_schema  # noqa: E402


class EventSchemaTests(unittest.TestCase):
    def test_missing_event_fields_are_typed_and_ordered(self):
        early = standardize_event_schema('lido_events', pd.DataFrame({'amount': ['42']}))
        later = standardize_event_schema(
            'lido_events', pd.DataFrame({'postTotalEther': ['43'], 'amount': ['42']})
        )
        self.assertEqual(list(early), list(later))
        self.assertEqual(list(early), [*EVENT_BASE_COLUMNS, *EVENT_EXTRA_COLUMNS['lido_events']])
        self.assertTrue(pa.Table.from_pandas(early).schema.equals(pa.Table.from_pandas(later).schema))
        self.assertTrue(pd.isna(early.loc[0, 'postTotalEther']))
        self.assertEqual(later.loc[0, 'postTotalEther'], '43')

    def test_aave_bool_stays_bool_when_missing(self):
        early = standardize_event_schema('aave_events', pd.DataFrame({'amount': ['42']}))
        later = standardize_event_schema(
            'aave_events', pd.DataFrame({'receiveAToken': [True], 'amount': ['42']})
        )
        self.assertEqual(pa.Table.from_pandas(early).schema.field('receiveAToken').type, pa.bool_())
        self.assertTrue(pa.Table.from_pandas(early).schema.equals(pa.Table.from_pandas(later).schema))

    def test_unregistered_field_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'unregistered columns'):
            standardize_event_schema('lido_events', pd.DataFrame({'unexpected': ['x']}))


if __name__ == '__main__':
    unittest.main()

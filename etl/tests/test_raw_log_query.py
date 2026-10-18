from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from extract_bigquery import build_raw_protocol_logs_sql, output_path, query_params, raw_log_windows  # noqa: E402
from utils import load_yaml, parse_utc  # noqa: E402


class RawProtocolLogQueryTests(unittest.TestCase):
    def test_raw_log_windows_never_cross_a_calendar_month(self):
        windows = list(raw_log_windows(
            parse_utc('2023-01-26T00:00:00Z'),
            parse_utc('2023-02-08T00:00:00Z'), 7,
        ))
        self.assertEqual(windows, [
            (parse_utc('2023-01-26T00:00:00Z'), parse_utc('2023-02-01T00:00:00Z')),
            (parse_utc('2023-02-01T00:00:00Z'), parse_utc('2023-02-08T00:00:00Z')),
        ])
        first_path = output_path(Path('archive'), 'raw_protocol_logs', *windows[0])
        self.assertIn('20230126T000000Z_20230201T000000Z', first_path.name)
        self.assertNotEqual(first_path, output_path(Path('archive'), 'raw_protocol_logs',
                                                    windows[0][0], parse_utc('2023-02-02T00:00:00Z')))

    @classmethod
    def setUpClass(cls):
        cls.extraction = load_yaml('config/extraction.yaml')
        cls.registry = load_yaml('config/contracts.yaml')
        cls.contract_keys = cls.extraction['bigquery']['raw_log_contracts']

    def test_query_keeps_original_topics_and_payload_and_only_selected_addresses(self):
        sql, specs = build_raw_protocol_logs_sql(
            self.extraction['bigquery']['raw_logs_table'],
            self.registry['contracts'],
            self.contract_keys,
        )
        self.assertIn('topics,', sql)
        self.assertIn('data,', sql)
        self.assertIn('source_event_fingerprint', sql)
        self.assertIn('block_timestamp >= @start_ts', sql)
        self.assertIn('block_timestamp < @end_ts', sql)
        self.assertEqual(len(specs), 7)
        self.assertEqual(
            {spec['key'] for spec in specs},
            set(self.contract_keys),
        )
        self.assertNotIn('event_signature IN', sql)

    def test_deployment_interval_is_applied_to_withdrawal_queue(self):
        sql, specs = build_raw_protocol_logs_sql(
            self.extraction['bigquery']['raw_logs_table'],
            self.registry['contracts'],
            self.contract_keys,
        )
        queue = next(spec for spec in specs if spec['key'] == 'lido_withdrawal_queue')
        self.assertEqual(queue['active_from'], '2023-05-15T16:00:00Z')
        self.assertIn(f"block_timestamp >= @{queue['active_from_param']}", sql)

        params = query_params(
            parse_utc('2023-05-01T00:00:00Z'),
            parse_utc('2023-06-01T00:00:00Z'),
            specs,
        )
        active_param = next(param for param in params if param.name == queue['active_from_param'])
        self.assertEqual(active_param.value, parse_utc('2023-05-15T16:00:00Z'))

    def test_unknown_contract_and_duplicate_addresses_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown raw-log contract key'):
            build_raw_protocol_logs_sql('project.dataset.logs', {}, ['not_registered'])

        duplicate = {
            'one': {'address': '0x' + '1' * 40},
            'two': {'address': '0x' + '1' * 40},
        }
        with self.assertRaisesRegex(ValueError, 'Duplicate address'):
            build_raw_protocol_logs_sql('project.dataset.logs', duplicate, ['one', 'two'])


if __name__ == '__main__':
    unittest.main()

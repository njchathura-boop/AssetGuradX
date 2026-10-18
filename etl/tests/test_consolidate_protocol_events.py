from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from consolidate_protocol_events import (  # noqa: E402
    RAW_SOURCE,
    normalize_existing_event,
    normalize_raw_log,
    require_event_month,
    schema_for,
)


class ConsolidateProtocolEventsTests(unittest.TestCase):
    def test_rejects_cross_month_event_before_publishing(self):
        require_event_month({'block_timestamp': '2023-01-31T23:59:59Z'}, '2023-01', Path('source.parquet'))
        with self.assertRaisesRegex(RuntimeError, 'outside UTC month 2023-01'):
            require_event_month({'block_timestamp': '2023-02-01T00:00:00Z'}, '2023-01', Path('source.parquet'))

    def test_shared_schema_keeps_source_rows_and_raw_evidence(self):
        schema = schema_for('1.0.3', protocol_rows=1, raw_log_rows=1)
        self.assertEqual(schema.metadata[b'assetguardx_consolidation_version'], b'assetguardx-protocol-events/v2')
        self.assertEqual(schema.metadata[b'assetguardx_partition_by'], b'block_timestamp:calendar_month_utc')
        event = normalize_existing_event({
            'block_number': 10,
            'block_timestamp': '2023-01-01T00:00:00Z',
            'transaction_hash': '0x' + '11' * 32,
            'log_index': None,
            'source_table': 'decoded_events',
            'source_event_fingerprint': 'decoded-fingerprint',
            'contract_address': '0x' + '22' * 20,
            'contract_key': 'uniswap_pool',
            'event_signature': 'Swap(address,uint256)',
            'args_json': '[]',
        })
        log = normalize_raw_log({
            'block_number': 11,
            'block_timestamp': '2023-01-01T01:00:00Z',
            'transaction_hash': '0x' + '33' * 32,
            'log_index': 4,
            'source_event_fingerprint': 'raw-fingerprint',
            'contract_address': '0x' + '44' * 20,
            'contract_key': 'lido_withdrawal_queue',
            'topic0': '0x' + '55' * 32,
            'topics': ['0x' + '55' * 32, '0x' + '66' * 32],
            'data': '0x' + '77' * 32,
            'event_signature': 'WithdrawalsFinalized(uint256,uint256,uint256,uint256,uint256)',
            'args_json': '["1","2","3","4","5"]',
            'decoded_args_json': '{"fromRequestId":"1"}',
            'event_name': 'WithdrawalsFinalized',
            'event_family': 'withdrawal',
            'decode_status': 'decoded',
            'decode_error': None,
        }, '1.0.3')

        table = pa.Table.from_pylist([event, log], schema=schema)
        self.assertEqual(table.num_rows, 2)
        rows = table.to_pylist()
        self.assertEqual(rows[0]['source_table'], 'decoded_events')
        self.assertEqual(rows[1]['source_table'], RAW_SOURCE)
        self.assertEqual(rows[1]['topics'], ['0x' + '55' * 32, '0x' + '66' * 32])
        self.assertEqual(rows[1]['data'], '0x' + '77' * 32)
        self.assertEqual(rows[1]['decoder_version'], '1.0.3')

if __name__ == '__main__':
    unittest.main()

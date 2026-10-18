from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from eth_abi import encode as abi_encode
from eth_utils import keccak

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from decode_raw_logs import decode_log_record, make_contract_specs  # noqa: E402
from process_raw import decode_protocol_events, refresh_unknown_topics, standardize_event_schema  # noqa: E402
from utils import load_yaml  # noqa: E402


def word(abi_type: str, value) -> str:
    return '0x' + abi_encode([abi_type], [value]).hex()


class RawLogDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_yaml('config/extraction.yaml')
        cls.registry = load_yaml('config/contracts.yaml')
        cls.abi = load_yaml('config/raw_log_abis.yaml')
        cls.contract_specs = make_contract_specs(cls.registry, cls.abi)

    def base_row(self, key: str, signature: str, topics: list[str], data: bytes) -> dict:
        return {
            'block_number': 17_000_001,
            'block_timestamp': '2023-06-01T12:00:00+00:00',
            'transaction_hash': '0x' + 'ab' * 32,
            'log_index': 5,
            'contract_address': self.registry['contracts'][key]['address'],
            'contract_key': key,
            'topic0': '0x' + keccak(text=signature).hex(),
            'topics': ['0x' + keccak(text=signature).hex(), *topics],
            'data': '0x' + data.hex(),
            'source_event_fingerprint': 'test-fingerprint',
        }

    def test_aave_deposit_decodes_indexed_and_data_arguments(self):
        signature = 'Deposit(address,address,address,uint256,uint16)'
        reserve = '0x' + '11' * 20
        user = '0x' + '22' * 20
        beneficiary = '0x' + '33' * 20
        amount = 123_456_789_012_345_678
        referral = 7
        data = abi_encode(['address', 'uint256'], [user, amount])
        row = self.base_row(
            'aave_v2_lending_pool', signature,
            [word('address', reserve), word('address', beneficiary), word('uint16', referral)], data,
        )
        decoded = decode_log_record(row, self.contract_specs, self.abi['decoder_version'])
        self.assertEqual(decoded['decode_status'], 'decoded')
        args = json.loads(decoded['decoded_args_json'])
        self.assertEqual(args['reserve'], reserve.lower())
        self.assertEqual(args['user'], user.lower())
        self.assertEqual(args['onBehalfOf'], beneficiary.lower())
        self.assertEqual(args['amount'], str(amount))
        self.assertEqual(args['referral'], str(referral))

    def test_steth_transfer_and_withdrawal_nft_transfer_use_different_layouts(self):
        signature = 'Transfer(address,address,uint256)'
        sender = '0x' + '44' * 20
        receiver = '0x' + '55' * 20
        amount = 999_000_000_000_000_000
        steth = self.base_row(
            'lido_steth', signature,
            [word('address', sender), word('address', receiver)],
            abi_encode(['uint256'], [amount]),
        )
        steth_decoded = decode_log_record(steth, self.contract_specs, self.abi['decoder_version'])
        self.assertEqual(steth_decoded['decode_status'], 'decoded')
        self.assertEqual(json.loads(steth_decoded['decoded_args_json'])['value'], str(amount))

        token_id = 73
        queue = self.base_row(
            'lido_withdrawal_queue', signature,
            [word('address', sender), word('address', receiver), word('uint256', token_id)], b'',
        )
        queue_decoded = decode_log_record(queue, self.contract_specs, self.abi['decoder_version'])
        self.assertEqual(queue_decoded['decode_status'], 'decoded')
        self.assertEqual(json.loads(queue_decoded['decoded_args_json'])['tokenId'], str(token_id))
        self.assertEqual(queue_decoded['event_family'], 'withdrawal_nft_transfer')

    def test_curve_frax_single_coin_withdrawal_uses_the_pool_specific_four_argument_abi(self):
        signature = 'RemoveLiquidityOne(address,uint256,uint256,uint256)'
        provider = '0x' + '66' * 20
        token_amount, coin_amount, supply = 10**18, 999_000, 20 * 10**18
        data = abi_encode(['uint256', 'uint256', 'uint256'], [token_amount, coin_amount, supply])
        row = self.base_row(
            'curve_frax_usdc', signature,
            [word('address', provider)], data,
        )
        self.assertEqual(row['topic0'], '0x5ad056f2e28a8cec232015406b843668c1e36cda598127ec3b8c59b8c72773a0')
        decoded = decode_log_record(row, self.contract_specs, self.abi['decoder_version'])
        self.assertEqual(decoded['decode_status'], 'decoded')
        args = json.loads(decoded['decoded_args_json'])
        self.assertEqual(args['provider'], provider.lower())
        self.assertEqual(args['token_amount'], str(token_amount))
        self.assertEqual(args['coin_amount'], str(coin_amount))
        self.assertEqual(args['token_supply'], str(supply))

    def test_withdrawal_queue_approval_for_all_decodes_operator_boolean(self):
        signature = 'ApprovalForAll(address,address,bool)'
        owner = '0x' + '71' * 20
        operator = '0x' + '72' * 20
        row = self.base_row(
            'lido_withdrawal_queue', signature,
            [word('address', owner), word('address', operator)],
            abi_encode(['bool'], [True]),
        )
        decoded = decode_log_record(row, self.contract_specs, self.abi['decoder_version'])
        self.assertEqual(decoded['decode_status'], 'decoded')
        self.assertEqual(json.loads(decoded['decoded_args_json']), {
            'owner': owner.lower(), 'operator': operator.lower(), 'approved': True,
        })

    def test_withdrawals_finalized_decodes_both_indexed_request_ids(self):
        signature = 'WithdrawalsFinalized(uint256,uint256,uint256,uint256,uint256)'
        from_request_id, to_request_id = 120, 147
        amount_locked, shares_to_burn, timestamp = 10**18, 9 * 10**17, 1_682_000_000
        row = self.base_row(
            'lido_withdrawal_queue', signature,
            [word('uint256', from_request_id), word('uint256', to_request_id)],
            abi_encode(['uint256', 'uint256', 'uint256'], [amount_locked, shares_to_burn, timestamp]),
        )

        decoded = decode_log_record(row, self.contract_specs, self.abi['decoder_version'])

        self.assertEqual(decoded['decode_status'], 'decoded')
        self.assertEqual(json.loads(decoded['decoded_args_json']), {
            'fromRequestId': str(from_request_id),
            'toRequestId': str(to_request_id),
            'amountOfETHLocked': str(amount_locked),
            'sharesToBurn': str(shares_to_burn),
            'timestamp': str(timestamp),
        })

    def test_withdrawal_queue_token_and_operator_approvals_have_separate_parquet_schemas(self):
        import pandas as pd
        import pyarrow as pa

        owner = '0x' + '81' * 20
        approved_address = '0x' + '82' * 20
        operator = '0x' + '83' * 20
        rows = pd.DataFrame([
            {
                'block_number': 1, 'block_timestamp': '2023-06-01T12:00:00Z',
                'transaction_hash': '0x' + '91' * 32, 'log_index': 1,
                'source_table': 'raw_protocol_logs', 'source_event_fingerprint': 'token-approval',
                'contract_key': 'lido_withdrawal_queue',
                'contract_address': self.registry['contracts']['lido_withdrawal_queue']['address'],
                'event_signature': 'Approval(address,address,uint256)',
                'args_json': '[]',
                'decoded_args_json': json.dumps({
                    'owner': owner, 'approved': approved_address, 'tokenId': '123',
                }),
            },
            {
                'block_number': 2, 'block_timestamp': '2023-06-01T13:00:00Z',
                'transaction_hash': '0x' + '92' * 32, 'log_index': 2,
                'source_table': 'raw_protocol_logs', 'source_event_fingerprint': 'operator-approval',
                'contract_key': 'lido_withdrawal_queue',
                'contract_address': self.registry['contracts']['lido_withdrawal_queue']['address'],
                'event_signature': 'ApprovalForAll(address,address,bool)',
                'args_json': '[]',
                'decoded_args_json': json.dumps({
                    'owner': owner, 'operator': operator, 'approved': True,
                }),
            },
            {
                'block_number': 3, 'block_timestamp': '2023-06-01T14:00:00Z',
                'transaction_hash': '0x' + '93' * 32, 'log_index': 3,
                'source_table': 'protocol_events', 'source_event_fingerprint': 'legacy-token-approval',
                'contract_key': 'lido_withdrawal_queue',
                'contract_address': self.registry['contracts']['lido_withdrawal_queue']['address'],
                'event_signature': 'Approval(address,address,uint256)',
                'args_json': json.dumps([owner, approved_address, '456']),
                'decoded_args_json': None,
            },
        ])

        output = decode_protocol_events(rows, self.registry)
        token_approvals = output['lido_withdrawal_nft_token_approval_events']
        operator_approvals = output['lido_withdrawal_nft_operator_approval_events']
        self.assertEqual(token_approvals.iloc[0]['approved_address'], approved_address)
        self.assertEqual(token_approvals.iloc[0]['token_id'], '123')
        self.assertEqual(token_approvals.iloc[1]['approved_address'], approved_address)
        self.assertEqual(token_approvals.iloc[1]['token_id'], '456')
        self.assertTrue(operator_approvals.iloc[0]['approved'])
        pa.Table.from_pandas(token_approvals, preserve_index=False)
        pa.Table.from_pandas(operator_approvals, preserve_index=False)

    def test_unknown_and_malformed_logs_are_preserved_not_dropped(self):
        unknown = {
            'contract_key': 'curve_frax_usdc', 'topic0': '0x' + '99' * 32,
            'topics': ['0x' + '99' * 32], 'data': '0x1234',
        }
        result = decode_log_record(unknown, self.contract_specs, self.abi['decoder_version'])
        self.assertEqual(result['decode_status'], 'unknown_topic')
        self.assertEqual(result['topics'], unknown['topics'])
        self.assertEqual(result['data'], unknown['data'])

        signature = 'Withdraw(address,address,address,uint256)'
        malformed = self.base_row('aave_v2_lending_pool', signature, [], b'')
        result = decode_log_record(malformed, self.contract_specs, self.abi['decoder_version'])
        self.assertEqual(result['decode_status'], 'malformed')
        self.assertIn('Expected 4 topics', result['decode_error'])

    def test_every_registry_signature_for_raw_contracts_has_a_decoder_layout(self):
        for key in self.cfg['bigquery']['raw_log_contracts']:
            contract = self.registry['contracts'][key]
            indexed_topics = self.contract_specs[key]
            signatures = set(contract.get('decoded_event_signatures', []))
            signatures.update(['Transfer(address,address,uint256)', 'Approval(address,address,uint256)'])
            if key == 'lido_withdrawal_queue':
                signatures.add('ApprovalForAll(address,address,bool)')
            expected = {'0x' + keccak(text=signature).hex() for signature in signatures}
            missing = expected - set(indexed_topics)
            self.assertEqual(missing, set(), f'{key}: missing ABI layouts for {sorted(missing)}')

    def test_supplemental_events_decode_without_becoming_required_coverage(self):
        cases = [
            ('lido_steth', 'Unbuffered(uint256)', [], ['uint256'], [10**18]),
            ('lido_steth', 'DepositedValidatorsChanged(uint256)', [], ['uint256'], [25]),
            ('lido_steth', 'CLValidatorsUpdated(uint256,uint256,uint256)',
             [('uint256', 100)], ['uint256', 'uint256'], [24, 25]),
            ('lido_withdrawal_queue', 'BatchMetadataUpdate(uint256,uint256)',
             [], ['uint256', 'uint256'], [1, 2]),
            ('aave_v2_lending_pool', 'TokensRescued(address,address,uint256)',
             [('address', '0x' + '11' * 20), ('address', '0x' + '22' * 20)], ['uint256'], [7]),
            ('aave_v2_lending_pool', 'Paused()', [], [], []),
            ('aave_v2_lending_pool', 'Unpaused()', [], [], []),
            ('aave_v2_lending_pool', 'Upgraded(address)', [('address', '0x' + '33' * 20)], [], []),
            ('lido_steth', 'ContractVersionSet(uint256)', [], ['uint256'], [2]),
            ('lido_steth', 'EIP712StETHInitialized(address)', [], ['address'], ['0x' + '44' * 20]),
            ('lido_steth', 'LidoLocatorSet(address)', [], ['address'], ['0x' + '55' * 20]),
            ('curve_steth_eth', 'CommitNewFee(uint256,uint256,uint256)',
             [('uint256', 123)], ['uint256', 'uint256'], [4, 5]),
        ]
        for key, signature, indexed, data_types, values in cases:
            with self.subTest(signature=signature):
                row = self.base_row(key, signature, [word(t, v) for t, v in indexed],
                                    abi_encode(data_types, values))
                decoded = decode_log_record(row, self.contract_specs, self.abi['decoder_version'])
                self.assertEqual(decoded['decode_status'], 'decoded', decoded['decode_error'])
                self.assertIsNotNone(decoded['decoded_args_json'])
                self.assertNotIn(signature, self.registry['contracts'][key].get('decoded_event_signatures', []))

    def test_processor_overlays_frozen_unknown_rows_and_routes_metadata(self):
        import pandas as pd

        signature = 'BatchMetadataUpdate(uint256,uint256)'
        row = self.base_row('lido_withdrawal_queue', signature, [],
                            abi_encode(['uint256', 'uint256'], [10, 20]))
        frozen = pd.DataFrame([{**row, 'source_table': 'raw_protocol_logs',
                                'decode_status': 'unknown_topic', 'event_signature': None,
                                'event_name': None, 'event_family': 'unknown',
                                'decode_error': None, 'args_json': None,
                                'decoded_args_json': None, 'decoder_version': '1.0.3'}])
        refreshed = refresh_unknown_topics(frozen, self.contract_specs, self.abi['decoder_version'])
        self.assertEqual(frozen.iloc[0]['decode_status'], 'unknown_topic')
        self.assertEqual(refreshed.iloc[0]['decode_status'], 'decoded')
        self.assertEqual(refreshed.iloc[0]['decoder_version'], '1.0.4')
        routed = decode_protocol_events(refreshed, self.registry)
        self.assertEqual(routed['lido_withdrawal_metadata_events'].iloc[0]['fromTokenId'], '10')
        standardize_event_schema('lido_withdrawal_metadata_events',
                                 routed['lido_withdrawal_metadata_events'])

    def test_revised_curve_fee_fields_are_corrected_without_changing_raw(self):
        import pandas as pd

        signature = 'NewFee(uint256,uint256)'
        row = self.base_row('curve_steth_eth', signature, [],
                            abi_encode(['uint256', 'uint256'], [1_000_000, 5_000_000_000]))
        frozen = pd.DataFrame([{**row, 'source_table': 'raw_protocol_logs',
                                'decode_status': 'decoded', 'event_signature': signature,
                                'event_name': 'NewFee', 'event_family': 'dex_parameter',
                                'decode_error': None, 'args_json': '["1000000","5000000000"]',
                                'decoded_args_json': '{"old_fee":"1000000","new_fee":"5000000000"}',
                                'decoder_version': '1.0.3'}])
        revised = refresh_unknown_topics(frozen, self.contract_specs, self.abi['decoder_version'],
                                         self.abi['redecode_archived_signatures'])
        self.assertIn('old_fee', frozen.iloc[0]['decoded_args_json'])
        self.assertEqual(json.loads(revised.iloc[0]['decoded_args_json']),
                         {'fee': '1000000', 'admin_fee': '5000000000'})
        routed = decode_protocol_events(revised, self.registry)
        parameter = standardize_event_schema('curve_parameter_events', routed['curve_parameter_events'])
        self.assertEqual(parameter.iloc[0]['fee'], '1000000')
        self.assertEqual(parameter.iloc[0]['admin_fee'], '5000000000')

    def test_processor_routes_steth_transfer_separately_from_staking_accounting(self):
        signature = 'Transfer(address,address,uint256)'
        args = {'from': '0x' + '00' * 20, 'to': '0x' + '33' * 20, 'value': '1000000000000000000'}
        import pandas as pd

        rows = pd.DataFrame([{
            'block_number': 1, 'block_timestamp': '2023-06-01T12:00:00Z',
            'transaction_hash': '0x' + '12' * 32, 'log_index': 1,
            'source_table': 'raw_protocol_logs', 'source_event_fingerprint': 'fp',
            'contract_key': 'lido_steth',
            'contract_address': self.registry['contracts']['lido_steth']['address'],
            'event_signature': signature, 'args_json': json.dumps(list(args.values())),
            'decoded_args_json': json.dumps(args),
        }])
        output = decode_protocol_events(rows, self.registry)
        transfer = output['lido_token_transfer_events'].iloc[0]
        self.assertEqual(transfer['activity_type'], 'mint')
        self.assertEqual(transfer['value_raw'], '1000000000000000000')
        self.assertEqual(transfer['value_normalized'], 1.0)
        self.assertNotIn('lido_events', output)

if __name__ == '__main__':
    unittest.main()

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from eth_abi import decode, encode
from eth_utils import keccak

from utils import (
    active_at,
    append_jsonl,
    copy_to_drive_with_retries,
    load_yaml,
    resolve_storage_root,
    staging_path,
)


def selector(signature: str) -> bytes:
    return keccak(text=signature)[:4]


def calldata(signature: str, input_types: list[str] | None = None, args: list | None = None) -> str:
    data = selector(signature)
    if input_types:
        data += encode(input_types, args or [])
    return '0x' + data.hex()


def decode_result(raw_hex: str, output_types: list[str]):
    if raw_hex in (None, '0x'):
        raise ValueError('empty eth_call result')
    return decode(output_types, bytes.fromhex(raw_hex[2:]))


def jsonify(v):
    if isinstance(v, (bytes, bytearray)):
        return '0x' + bytes(v).hex()
    if isinstance(v, tuple):
        return [jsonify(x) for x in v]
    if isinstance(v, list):
        return [jsonify(x) for x in v]
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, int):
        return str(v)
    return str(v)


class RpcClient:
    def __init__(self, url: str, timeout=90, retries=6, batch_size=40, pause=0.05):
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self.batch_size = batch_size
        self.pause = pause
        self.session = requests.Session()
        self.next_id = 1

    def _post(self, payload):
        last = None
        for attempt in range(self.retries):
            try:
                r = self.session.post(self.url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                body = r.json()
                if self.pause:
                    time.sleep(self.pause)
                return body
            except Exception as exc:
                last = exc
                time.sleep(min(2 ** attempt, 20))
        raise RuntimeError(f'RPC request failed after {self.retries} attempts: {last}')

    def eth_call(self, to: str, data: str, block_number: int) -> str:
        rid = self.next_id; self.next_id += 1
        body = self._post({'jsonrpc':'2.0','id':rid,'method':'eth_call','params':[{'to':to,'data':data}, hex(int(block_number))]})
        if 'error' in body:
            raise RuntimeError(body['error'])
        return body['result']

    def batch_eth_call(self, specs: list[dict]) -> list[dict]:
        results = []
        total_batches = (len(specs) + self.batch_size - 1) // self.batch_size
        started = time.monotonic()
        for batch_number, i in enumerate(range(0, len(specs), self.batch_size), start=1):
            chunk = specs[i:i+self.batch_size]
            payload, id_to_spec = [], {}
            for spec in chunk:
                rid = self.next_id; self.next_id += 1
                id_to_spec[rid] = spec
                payload.append({
                    'jsonrpc':'2.0','id':rid,'method':'eth_call',
                    'params':[{'to':spec['to'],'data':spec['data']}, hex(int(spec['block_number']))]
                })
            try:
                body = self._post(payload)
                if not isinstance(body, list):
                    raise RuntimeError(f'Batch RPC returned non-list response: {body}')
                by_id = {x.get('id'): x for x in body}
                for rid, spec in id_to_spec.items():
                    item = by_id.get(rid)
                    if item is None:
                        results.append({'spec':spec,'error':'missing batch response'})
                    elif 'error' in item:
                        results.append({'spec':spec,'error':str(item['error'])})
                    else:
                        results.append({'spec':spec,'result':item.get('result')})
            except Exception as exc:
                # Fallback to individual calls so one provider/batch issue does not lose the whole month.
                for spec in chunk:
                    try:
                        raw = self.eth_call(spec['to'], spec['data'], spec['block_number'])
                        results.append({'spec':spec,'result':raw})
                    except Exception as indiv:
                        results.append({'spec':spec,'error':f'batch={exc}; individual={indiv}'})
            if batch_number == 1 or batch_number % 25 == 0 or batch_number == total_batches:
                elapsed = max(time.monotonic() - started, 0.001)
                rate = batch_number / elapsed
                remaining = (total_batches - batch_number) / rate if rate else 0
                print(
                    f'[RPC] batches {batch_number:,}/{total_batches:,} '
                    f'({batch_number / total_batches:.1%}); '
                    f'elapsed={elapsed / 60:.1f}m; eta={remaining / 60:.1f}m',
                    flush=True,
                )
        return results


def find_hourly_block_files(storage_root: Path) -> list[Path]:
    return sorted((storage_root / 'raw' / 'hourly_blocks').glob('year=*/month=*/hourly_blocks_*.parquet'))


def spec(entity_key, state_kind, hour, block, to, field_group, signature, output_types, output_names,
         input_types=None, args=None, extra=None):
    return {
        'entity_key': entity_key,
        'state_kind': state_kind,
        'hour': hour,
        'block_number': int(block),
        'to': to.lower(),
        'field_group': field_group,
        'signature': signature,
        'data': calldata(signature, input_types, args),
        'output_types': output_types,
        'output_names': output_names,
        'extra': extra or {},
    }


def discover_aave_tokens(rpc: RpcClient, meta: dict, block: int) -> dict:
    out = {}
    provider = meta['address']
    calls = []
    for reserve in meta['reserves']:
        calls.append(spec(
            f"aave_v2_{reserve['symbol']}", 'aave_v2_reserve', datetime.now(timezone.utc), block,
            provider, 'reserve_tokens', 'getReserveTokensAddresses(address)',
            ['address','address','address'], ['a_token','stable_debt_token','variable_debt_token'],
            ['address'], [reserve['address']], {'reserve': reserve}
        ))
    for item in rpc.batch_eth_call(calls):
        s = item['spec']; reserve = s['extra']['reserve']; symbol = reserve['symbol']
        if item.get('error'):
            raise RuntimeError(f'Aave reserve-token discovery failed for {symbol}: {item["error"]}')
        vals = decode_result(item['result'], s['output_types'])
        addresses = [str(v).lower() for v in vals]
        if any(a == '0x0000000000000000000000000000000000000000' for a in addresses):
            raise RuntimeError(f'Aave reserve {symbol} returned a zero debt/aToken address; registry may be wrong.')
        out[symbol] = {
            **reserve,
            'a_token': addresses[0],
            'stable_debt_token': addresses[1],
            'variable_debt_token': addresses[2],
        }
    return out


def build_month_specs(registry: dict, hours_df: pd.DataFrame, aave_tokens: dict) -> tuple[list[dict], dict]:
    contracts = registry['contracts']
    calls = []
    base_records = {}

    def record(hour, block, entity_key, state_kind, meta, extra=None):
        key = (pd.Timestamp(hour).isoformat(), entity_key)
        if key not in base_records:
            base_records[key] = {
                'hour': hour,
                'block_number': int(block),
                'entity_key': entity_key,
                'state_kind': state_kind,
                'protocol': meta.get('protocol'),
                'contract_address': meta.get('address'),
                'asset': meta.get('asset'),
                'values': {},
                'quality_status': 'ok',
                'errors': [],
                **(extra or {}),
            }
        return key

    for hr in hours_df.itertuples(index=False):
        hour = pd.Timestamp(hr.hour).to_pydatetime()
        block = int(hr.block_number)

        for entity_key, meta in contracts.items():
            kind = meta.get('state_kind')
            if not kind or not active_at(meta, hour):
                continue

            if kind == 'chainlink_feed':
                record(hour, block, entity_key, kind, meta)
                calls.append(spec(entity_key, kind, hour, block, meta['address'], 'round', 'latestRoundData()',
                                  ['uint80','int256','uint256','uint256','uint80'],
                                  ['round_id','answer_raw','started_at','updated_at','answered_in_round']))

            elif kind == 'lido_wsteth':
                record(hour, block, entity_key, kind, meta)
                calls.append(spec(entity_key, kind, hour, block, meta['address'], 'ratio', 'stEthPerToken()',
                                  ['uint256'], ['steth_per_wsteth_raw']))

            elif kind == 'lido_steth':
                record(hour, block, entity_key, kind, meta)
                calls.extend([
                    spec(entity_key, kind, hour, block, meta['address'], 'pooled', 'getTotalPooledEther()', ['uint256'], ['total_pooled_ether_raw']),
                    spec(entity_key, kind, hour, block, meta['address'], 'shares', 'getTotalShares()', ['uint256'], ['total_shares_raw']),
                    spec(entity_key, kind, hour, block, meta['address'], 'supply', 'totalSupply()', ['uint256'], ['total_supply_raw']),
                ])

            elif kind == 'lido_withdrawal_queue':
                record(hour, block, entity_key, kind, meta)
                calls.extend([
                    spec(entity_key, kind, hour, block, meta['address'], 'last_request', 'getLastRequestId()', ['uint256'], ['last_request_id']),
                    spec(entity_key, kind, hour, block, meta['address'], 'last_finalized', 'getLastFinalizedRequestId()', ['uint256'], ['last_finalized_request_id']),
                    spec(entity_key, kind, hour, block, meta['address'], 'locked', 'getLockedEtherAmount()', ['uint256'], ['locked_ether_raw']),
                    spec(entity_key, kind, hour, block, meta['address'], 'unfinalized_n', 'unfinalizedRequestNumber()', ['uint256'], ['unfinalized_request_number']),
                    spec(entity_key, kind, hour, block, meta['address'], 'unfinalized_steth', 'unfinalizedStETH()', ['uint256'], ['unfinalized_steth_raw']),
                    spec(entity_key, kind, hour, block, meta['address'], 'bunker', 'isBunkerModeActive()', ['bool'], ['is_bunker_mode']),
                ])

            elif kind == 'uniswap_v3_pool':
                record(hour, block, entity_key, kind, meta, {'token0':meta['token0'],'token1':meta['token1'],'fee':meta['fee']})
                calls.extend([
                    spec(entity_key, kind, hour, block, meta['address'], 'slot0', 'slot0()',
                         ['uint160','int24','uint16','uint16','uint16','uint8','bool'],
                         ['sqrt_price_x96','tick','observation_index','observation_cardinality','observation_cardinality_next','fee_protocol','unlocked']),
                    spec(entity_key, kind, hour, block, meta['address'], 'liquidity', 'liquidity()', ['uint128'], ['active_liquidity_raw']),
                ])

            elif kind == 'curve_pool':
                record(hour, block, entity_key, kind, meta, {'coins_json':json.dumps(meta['coins'])})
                for i, coin in enumerate(meta['coins']):
                    calls.append(spec(entity_key, kind, hour, block, meta['address'], f'balance_{i}', 'balances(uint256)',
                                      ['uint256'], [f'balance_{i}_raw'], ['uint256'], [i], {'coin':coin}))

                # Continuous hourly marginal market quotes.  These calls avoid relying
                # only on hours that happened to contain a swap.  We store both
                # directions so downstream code can form a bid/ask-style midpoint.
                for mq in meta.get('market_quotes', []):
                    target = mq['target']; quote = mq['quote']
                    method = mq.get('method', 'get_dy')
                    signature = f'{method}(int128,int128,uint256)'
                    target_dec = int(registry['assets'][target]['decimals'])
                    quote_dec = int(registry['assets'][quote]['decimals'])
                    ti = int(mq['target_index']); qi = int(mq['quote_index'])
                    safe_target = target.lower().replace('-', '_')
                    safe_quote = quote.lower().replace('-', '_')
                    calls.extend([
                        spec(entity_key, kind, hour, block, meta['address'],
                             f'quote_{safe_target}_{safe_quote}_forward', signature, ['uint256'],
                             [f'quote_{safe_target}_{safe_quote}_forward_raw'],
                             ['int128','int128','uint256'], [ti, qi, 10 ** target_dec]),
                        spec(entity_key, kind, hour, block, meta['address'],
                             f'quote_{safe_target}_{safe_quote}_reverse', signature, ['uint256'],
                             [f'quote_{safe_target}_{safe_quote}_reverse_raw'],
                             ['int128','int128','uint256'], [qi, ti, 10 ** quote_dec]),
                    ])

                calls.extend([
                    spec(entity_key, kind, hour, block, meta['address'], 'A', 'A()', ['uint256'], ['A_raw']),
                    spec(entity_key, kind, hour, block, meta['address'], 'fee', 'fee()', ['uint256'], ['fee_raw']),
                    spec(entity_key, kind, hour, block, meta['address'], 'virtual_price', 'get_virtual_price()', ['uint256'], ['virtual_price_raw']),
                ])

            elif kind == 'aave_v2_data_provider':
                pool = meta['lending_pool']
                for symbol, reserve in aave_tokens.items():
                    rkey = f'aave_v2_{symbol}'
                    record(hour, block, rkey, 'aave_v2_reserve', meta, {
                        'asset':symbol, 'underlying_address':reserve['address'].lower(), 'token_decimals':int(reserve['decimals']),
                        'a_token':reserve['a_token'], 'stable_debt_token':reserve['stable_debt_token'], 'variable_debt_token':reserve['variable_debt_token'],
                    })
                    # Snapshot economic exposure from token supplies + underlying cash.
                    calls.extend([
                        spec(rkey, 'aave_v2_reserve', hour, block, reserve['address'], 'available', 'balanceOf(address)', ['uint256'], ['available_liquidity_raw'], ['address'], [reserve['a_token']]),
                        spec(rkey, 'aave_v2_reserve', hour, block, reserve['a_token'], 'supply', 'totalSupply()', ['uint256'], ['total_a_token_raw']),
                        spec(rkey, 'aave_v2_reserve', hour, block, reserve['stable_debt_token'], 'stable_debt', 'totalSupply()', ['uint256'], ['total_stable_debt_raw']),
                        spec(rkey, 'aave_v2_reserve', hour, block, reserve['variable_debt_token'], 'variable_debt', 'totalSupply()', ['uint256'], ['total_variable_debt_raw']),
                        spec(rkey, 'aave_v2_reserve', hour, block, meta['address'], 'config', 'getReserveConfigurationData(address)',
                             ['uint256','uint256','uint256','uint256','uint256','bool','bool','bool','bool','bool'],
                             ['decimals','ltv','liquidation_threshold','liquidation_bonus','reserve_factor','usage_as_collateral_enabled','borrowing_enabled','stable_borrow_rate_enabled','is_active','is_frozen'],
                             ['address'], [reserve['address']]),
                        spec(rkey, 'aave_v2_reserve', hour, block, pool, 'normalized_income', 'getReserveNormalizedIncome(address)', ['uint256'], ['normalized_income_raw'], ['address'], [reserve['address']]),
                        spec(rkey, 'aave_v2_reserve', hour, block, pool, 'normalized_debt', 'getReserveNormalizedVariableDebt(address)', ['uint256'], ['normalized_variable_debt_raw'], ['address'], [reserve['address']]),
                    ])

    return calls, base_records


def apply_results(call_results: list[dict], records: dict):
    for item in call_results:
        s = item['spec']
        key = (pd.Timestamp(s['hour']).isoformat(), s['entity_key'])
        rec = records[key]
        if item.get('error'):
            rec['quality_status'] = 'partial'
            rec['errors'].append(f"{s['field_group']}: {item['error']}")
            continue
        try:
            vals = decode_result(item['result'], s['output_types'])
            if len(vals) != len(s['output_names']):
                raise ValueError('output length mismatch')
            for name, value in zip(s['output_names'], vals):
                rec['values'][name] = jsonify(value)
        except Exception as exc:
            rec['quality_status'] = 'partial'
            rec['errors'].append(f"{s['field_group']} decode: {exc}")


def flatten_records(records: dict, registry: dict) -> pd.DataFrame:
    rows = []
    chainlink_decimals = {
        k: int(v.get('decimals', 8)) for k,v in registry['contracts'].items() if v.get('state_kind') == 'chainlink_feed'
    }
    for rec in records.values():
        values = rec.pop('values')
        errors = rec.pop('errors')
        row = {**rec, 'values_json': json.dumps(values, sort_keys=True), 'errors_json': json.dumps(errors)}
        if rec['state_kind'] == 'chainlink_feed' and 'answer_raw' in values:
            d = chainlink_decimals.get(rec['entity_key'], 8)
            row['reference_price'] = int(values['answer_raw']) / (10 ** d)
            row['oracle_updated_at_unix'] = int(values.get('updated_at', '0'))
        if rec['state_kind'] == 'lido_wsteth' and 'steth_per_wsteth_raw' in values:
            row['steth_per_wsteth'] = int(values['steth_per_wsteth_raw']) / 1e18
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description='AssetGuardX hourly archive-RPC state snapshot extractor.')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('--month', help='Optional YYYY-MM repair/extraction filter. Other months are skipped.')
    p.add_argument('--batch-size', type=int, help='RPC eth_call batch size override for rate-limited providers.')
    p.add_argument('--pause', type=float, help='Seconds to pause between RPC batches.')
    p.add_argument('--retries', type=int, help='Maximum retries per RPC request.')
    p.add_argument('--timeout', type=int, help='HTTP timeout in seconds per RPC request.')
    args = p.parse_args()

    for name in ('batch_size', 'retries', 'timeout'):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise SystemExit(f'--{name.replace("_", "-")} must be positive')
    if args.pause is not None and args.pause < 0:
        raise SystemExit('--pause cannot be negative')

    cfg = load_yaml('config/extraction.yaml')
    registry = load_yaml('config/contracts.yaml')
    storage_root = resolve_storage_root(cfg, require=True)
    hb_files = find_hourly_block_files(storage_root)
    if args.month:
        hb_files = [f for f in hb_files if f.stem.endswith(args.month)]
    if not hb_files:
        raise SystemExit('No matching hourly_blocks files found. Complete BigQuery raw extraction first.')

    total_hours = sum(len(pd.read_parquet(f, columns=['hour'])) for f in hb_files)
    print(f'Hourly anchors: {total_hours:,}')
    print('State coverage: Chainlink, Uniswap V3, Curve, Aave V2 reserves, Lido stETH/wstETH, Lido withdrawal queue.')
    if not args.execute:
        print('Dry-run: no RPC calls made. Add --execute.')
        return

    url = os.getenv(cfg['rpc']['env_var'])
    if not url:
        raise SystemExit(f"Missing environment variable {cfg['rpc']['env_var']}")
    rpc = RpcClient(
        url,
        timeout=args.timeout if args.timeout is not None else int(cfg['rpc'].get('request_timeout_seconds', 90)),
        retries=args.retries if args.retries is not None else int(cfg['rpc'].get('max_retries', 6)),
        batch_size=args.batch_size if args.batch_size is not None else int(cfg['rpc'].get('batch_size', 40)),
        pause=args.pause if args.pause is not None else float(cfg['rpc'].get('pause_between_batches_seconds', 0.05)),
    )

    first_hb = pd.read_parquet(hb_files[0]).sort_values('hour').iloc[0]
    aave_meta = next(v for v in registry['contracts'].values() if v.get('state_kind') == 'aave_v2_data_provider')
    print('Discovering Aave V2 aToken/debt-token addresses on-chain...')
    aave_tokens = discover_aave_tokens(rpc, aave_meta, int(first_hb.block_number))
    print('Aave reserves:', ', '.join(aave_tokens))

    for hb_file in hb_files:
        month = hb_file.stem.replace('hourly_blocks_', '')
        hours_df = pd.read_parquet(hb_file).sort_values('hour')
        out = storage_root / 'raw' / 'rpc_hourly' / f'year={month[:4]}' / f'month={month[5:7]}' / f'rpc_hourly_{month}.parquet'
        stage = staging_path(cfg, 'rpc_hourly', month, f'rpc_hourly_{month}.parquet')
        if out.exists() and not args.force:
            try:
                existing = pd.read_parquet(out, columns=['quality_status'])
                partial = int((existing['quality_status'] != 'ok').sum()) if len(existing) else 0
                if len(existing) and partial == 0:
                    print(f'[SKIP] rpc_hourly {month} -> readable with no partial rows')
                    continue
                print(f'[REPAIR] rpc_hourly {month}: empty or partial_rows={partial}; regenerating only this month')
                out.unlink()
            except Exception as exc:
                print(f'[REPAIR] rpc_hourly {month}: unreadable existing file ({exc}); regenerating only this month')
                try: out.unlink()
                except OSError: pass
        if stage.exists() and not args.force:
            print(f'[RESUME-UPLOAD] rpc_hourly {month}')
            pub = copy_to_drive_with_retries(stage, out, cfg)
            append_jsonl(storage_root / cfg['output']['manifest'], {
                'dataset':'rpc_hourly','month':month,'file':str(out),'resumed_from_stage':True,
                'created_at_utc':datetime.now(timezone.utc).isoformat(), **pub,
            })
            continue

        calls, records = build_month_specs(registry, hours_df, aave_tokens)
        print(f'[RPC] {month}: {len(records):,} entity-hours; {len(calls):,} eth_call operations in batches')
        call_results = rpc.batch_eth_call(calls)
        apply_results(call_results, records)
        df = flatten_records(records, registry)
        # Preserve the actual last-block timestamp used for each hourly snapshot.
        # This is required for exact oracle/state freshness calculations later.
        anchors = hours_df[['block_number','block_timestamp']].copy()
        anchors = anchors.rename(columns={'block_timestamp':'snapshot_block_timestamp'})
        df = df.merge(anchors, on='block_number', how='left')
        stage.parent.mkdir(parents=True, exist_ok=True)
        partial = stage.with_suffix(stage.suffix + '.partial')
        partial.unlink(missing_ok=True)
        df.to_parquet(partial, index=False, compression=cfg['output'].get('compression','zstd'))
        stage.unlink(missing_ok=True); partial.replace(stage)
        failures = int((df['quality_status'] != 'ok').sum()) if len(df) else 0
        pub = copy_to_drive_with_retries(stage, out, cfg)
        append_jsonl(storage_root / cfg['output']['manifest'], {
            'dataset':'rpc_hourly','month':month,'file':str(out),'rows':len(df),'partial_rows':failures,
            'created_at_utc':datetime.now(timezone.utc).isoformat(), **pub,
        })
        print(f'[OK] rpc_hourly {month}: {len(df):,} rows; {failures:,} partial -> {out}')


if __name__ == '__main__':
    main()

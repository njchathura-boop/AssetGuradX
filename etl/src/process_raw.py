from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils import load_yaml, publish_dataframe, resolve_storage_root
from decode_raw_logs import decode_log_record, make_contract_specs


def monthly_files(root: Path, dataset: str):
    return sorted((root / 'raw' / dataset).glob('year=*/month=*/*.parquet'))


def month_of(path: Path) -> str:
    return path.stem.rsplit('_', 1)[-1]


def month_partition(path: Path) -> str:
    year = next((part[5:] for part in path.parts if part.startswith('year=')), None)
    month = next((part[6:] for part in path.parts if part.startswith('month=')), None)
    if not year or not month:
        return month_of(path)
    return f'{year}-{month}'


def files_by_month(root: Path, dataset: str, area: str = 'raw') -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted((root / area / dataset).glob('year=*/month=*/*.parquet')):
        grouped[month_partition(path)].append(path)
    return dict(grouped)


def load_args(raw) -> list:
    """Google decoded_events args are positional; tolerate future dict/list encodings too."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return []
    if isinstance(obj, list):
        vals = []
        for x in obj:
            if isinstance(x, dict):
                if 'value' in x:
                    vals.append(x['value'])
                elif len(x) == 1:
                    vals.append(next(iter(x.values())))
                else:
                    vals.append(x)
            else:
                vals.append(x)
        return vals
    if isinstance(obj, dict):
        # Preserve insertion order when a provider changes representation.
        return list(obj.values())
    return [obj]


ARG_NAMES = {
    'Swap(address,address,int256,int256,uint160,uint128,int24)':
        ['sender','recipient','amount0','amount1','sqrtPriceX96','liquidity','tick'],
    'Mint(address,address,int24,int24,uint128,uint256,uint256)':
        ['sender','owner','tickLower','tickUpper','amount','amount0','amount1'],
    'Burn(address,int24,int24,uint128,uint256,uint256)':
        ['owner','tickLower','tickUpper','amount','amount0','amount1'],
    'Flash(address,address,uint256,uint256,uint256,uint256)':
        ['sender','recipient','amount0','amount1','paid0','paid1'],
    'TokenExchange(address,int128,uint256,int128,uint256)':
        ['buyer','sold_id','tokens_sold','bought_id','tokens_bought'],
    'TokenExchangeUnderlying(address,int128,uint256,int128,uint256)':
        ['buyer','sold_id','tokens_sold','bought_id','tokens_bought'],
    'RemoveLiquidityOne(address,uint256,uint256)': ['provider','token_amount','coin_amount'],
    'RemoveLiquidityOne(address,uint256,uint256,uint256)':
        ['provider','token_amount','coin_amount','token_supply'],
    'NewFee(uint256,uint256)': ['fee', 'admin_fee'],
    'StopRampA(uint256,uint256)': ['current_a', 'timestamp'],
    'Deposit(address,address,address,uint256,uint16)':
        ['reserve','user','onBehalfOf','amount','referral'],
    'Withdraw(address,address,address,uint256)':
        ['reserve','user','to','amount'],
    'Borrow(address,address,address,uint256,uint256,uint256,uint16)':
        ['reserve','user','onBehalfOf','amount','borrowRateMode','borrowRate','referral'],
    'Repay(address,address,address,uint256)':
        ['reserve','user','repayer','amount'],
    'LiquidationCall(address,address,address,uint256,uint256,address,bool)':
        ['collateralAsset','debtAsset','user','debtToCover','liquidatedCollateralAmount','liquidator','receiveAToken'],
    'ReserveDataUpdated(address,uint256,uint256,uint256,uint256,uint256)':
        ['reserve','liquidityRate','stableBorrowRate','variableBorrowRate','liquidityIndex','variableBorrowIndex'],
    'ReserveUsedAsCollateralEnabled(address,address)': ['reserve','user'],
    'ReserveUsedAsCollateralDisabled(address,address)': ['reserve','user'],
    'Swap(address,address,uint256)': ['reserve','user','rateMode'],
    'RebalanceStableBorrowRate(address,address)': ['reserve','user'],
    'FlashLoan(address,address,address,uint256,uint256,uint16)': ['target','initiator','reserve','amount','premium','referralCode'],
    'Submitted(address,uint256,address)': ['sender','amount','referral'],
    'ETHDistributed(uint256,uint256,uint256,uint256,uint256,uint256)':
        ['reportTimestamp','preCLBalance','postCLBalance','withdrawalsWithdrawn','executionLayerRewardsWithdrawn','postBufferedEther'],
    'TokenRebased(uint256,uint256,uint256,uint256,uint256,uint256,uint256)':
        ['reportTimestamp','timeElapsed','preTotalShares','preTotalEther','postTotalShares','postTotalEther','sharesMintedAsFees'],
    'ELRewardsReceived(uint256)': ['amount'],
    'WithdrawalsReceived(uint256)': ['amount'],
    'TransferShares(address,address,uint256)': ['from','to','sharesValue'],
    'SharesBurnt(address,uint256,uint256,uint256)': ['account','preRebaseTokenAmount','postRebaseTokenAmount','sharesAmount'],
    'WithdrawalRequested(uint256,address,address,uint256,uint256)':
        ['requestId','requestor','owner','amountOfStETH','amountOfShares'],
    'WithdrawalsFinalized(uint256,uint256,uint256,uint256,uint256)':
        ['fromRequestId','toRequestId','amountOfETHLocked','sharesToBurn','timestamp'],
    'WithdrawalClaimed(uint256,address,address,uint256)':
        ['requestId','owner','receiver','amountOfETH'],
    'Transfer(address,address,uint256)': ['from','to','value'],
    'Approval(address,address,uint256)': ['owner','spender','value'],
    'ApprovalForAll(address,address,bool)': ['owner','operator','approved'],
}


# Event families contain several signatures. A month with no instance of one
# signature must still publish the same Arrow columns as every other month.
EVENT_BASE_COLUMNS = (
    'block_number', 'block_timestamp', 'hour', 'transaction_hash', 'log_index',
    'source_table', 'source_event_fingerprint', 'contract_key',
    'contract_address', 'event_signature', 'decode_ok',
)
EVENT_EXTRA_COLUMNS = {
    'dex_swap_events': (
        'protocol', 'pool', 'token0', 'token1', 'amount0', 'amount1',
        'amount0_raw', 'amount1_raw', 'sqrt_price_x96', 'liquidity_raw',
        'tick', 'execution_price_token0_per_token1',
        'execution_price_token1_per_token0', 'sold_asset', 'bought_asset',
        'tokens_sold', 'tokens_bought', 'tokens_sold_raw',
        'tokens_bought_raw', 'execution_price_bought_per_sold',
        'execution_price_sold_per_bought',
    ),
    'aave_events': (
        'reserve', 'user', 'onBehalfOf', 'to', 'repayer', 'amount',
        'referral', 'borrowRateMode', 'borrowRate', 'liquidityRate',
        'stableBorrowRate', 'variableBorrowRate', 'liquidityIndex',
        'variableBorrowIndex', 'collateralAsset', 'debtAsset',
        'debtToCover', 'liquidatedCollateralAmount', 'liquidator',
        'receiveAToken', 'initiator', 'premium', 'referralCode',
        'target', 'rateMode', 'asset', 'amount_normalized',
        'collateral_asset', 'liquidated_collateral_normalized',
        'debt_asset', 'debt_to_cover_normalized',
    ),
    'aave_administration_events': ('token', 'to', 'amount', 'implementation'),
    'curve_liquidity_events': (
        'provider', 'coin_amount', 'token_amount', 'token_supply',
        'fees', 'invariant', 'token_amounts',
    ),
    'lido_events': (
        'sender', 'amount', 'referral', 'from', 'to', 'sharesValue',
        'reportTimestamp', 'preCLBalance', 'postCLBalance',
        'withdrawalsWithdrawn', 'executionLayerRewardsWithdrawn',
        'postBufferedEther', 'timeElapsed', 'preTotalShares',
        'preTotalEther', 'postTotalShares', 'postTotalEther',
        'sharesMintedAsFees', 'account', 'preRebaseTokenAmount',
        'postRebaseTokenAmount', 'sharesAmount', 'depositedValidators',
        'preCLValidators', 'postCLValidators',
    ),
    'lido_administration_events': ('version', 'eip712StETH', 'lidoLocator'),
    'curve_parameter_events': (
        'new_fee', 'old_fee', 'future_time', 'initial_time', 'new_a', 'old_a',
        'deadline', 'fee', 'admin_fee', 'current_a', 'timestamp',
    ),
    'lido_withdrawal_events': (
        'requestId', 'requestor', 'owner', 'amountOfStETH',
        'amountOfShares', 'receiver', 'amountOfETH', 'fromRequestId',
        'toRequestId', 'amountOfETHLocked', 'sharesToBurn', 'timestamp',
    ),
    'lido_withdrawal_metadata_events': ('fromTokenId', 'toTokenId'),
}
EVENT_NON_STRING_COLUMNS = {
    'receiveAToken', 'amount0', 'amount1', 'tick',
    'execution_price_token0_per_token1',
    'execution_price_token1_per_token0', 'tokens_sold', 'tokens_bought',
    'execution_price_bought_per_sold',
    'execution_price_sold_per_bought', 'amount_normalized',
    'liquidated_collateral_normalized', 'debt_to_cover_normalized',
    'fees', 'token_amounts',
}


def standardize_event_schema(dataset: str, df: pd.DataFrame) -> pd.DataFrame:
    """Keep event-family column order and optional-field types stable by month."""
    if dataset not in EVENT_EXTRA_COLUMNS:
        return df
    columns = (*EVENT_BASE_COLUMNS, *EVENT_EXTRA_COLUMNS[dataset])
    unexpected = set(df.columns) - set(columns)
    if unexpected:
        raise ValueError(f'{dataset} has unregistered columns: {sorted(unexpected)}')
    result = df.reindex(columns=columns).copy()
    for name in EVENT_EXTRA_COLUMNS[dataset]:
        if name == 'receiveAToken':
            result[name] = result[name].astype('boolean')
        elif name in EVENT_NON_STRING_COLUMNS:
            continue
        else:
            result[name] = result[name].astype('string')
    return result


def typed_args(signature: str, raw, contract_key: str | None = None) -> tuple[dict, bool]:
    vals = load_args(raw)
    if contract_key == 'lido_withdrawal_queue' and signature == 'Approval(address,address,uint256)':
        # This ERC-721 contract names the second argument `approved` (an address)
        # and the third `tokenId`; the standard ERC-20 Approval layout differs.
        names = ['owner', 'approved', 'tokenId']
    else:
        names = ARG_NAMES.get(signature)
    if not names:
        return {'args_json': json.dumps(vals, default=str)}, False
    if len(vals) != len(names):
        return {'args_json': json.dumps(vals, default=str), 'expected_arg_count': len(names), 'actual_arg_count': len(vals)}, False
    return dict(zip(names, vals)), True


def refresh_unknown_topics(df: pd.DataFrame, contract_specs: dict, decoder_version: str,
                           revised_signatures=()) -> pd.DataFrame:
    """Re-decode unknown/revised archived rows, leaving frozen raw untouched."""
    if not {'source_table', 'decode_status', 'topics'}.issubset(df.columns):
        return df
    selected = df['source_table'].eq('raw_protocol_logs') & (
        df['decode_status'].eq('unknown_topic') | df['event_signature'].isin(revised_signatures)
    )
    if not selected.any():
        return df
    result = df.copy()
    fields = ('event_signature', 'event_name', 'event_family', 'decode_status',
              'decode_error', 'args_json', 'decoded_args_json', 'decoder_version')
    for index, row in result.loc[selected].iterrows():
        source = row.to_dict()
        topics = source.get('topics')
        if topics is not None and not isinstance(topics, str):
            source['topics'] = list(topics)
        decoded = decode_log_record(source, contract_specs, decoder_version)
        for field in fields:
            result.at[index, field] = decoded[field]
    return result


def build_token_hourly(df: pd.DataFrame) -> pd.DataFrame:
    if '_empty' in df.columns or df.empty:
        return pd.DataFrame()
    d = df.copy()
    d['hour'] = pd.to_datetime(d['block_timestamp'], utc=True).dt.floor('h')
    q = pd.to_numeric(d['quantity_raw'], errors='coerce')
    decimals = pd.to_numeric(d['token_decimals'], errors='coerce')
    d['quantity'] = q / (10.0 ** decimals)
    d['mint_amount'] = d['quantity'].where(d['activity_type'].eq('mint'), 0.0)
    d['burn_amount'] = d['quantity'].where(d['activity_type'].eq('burn'), 0.0)
    d['transfer_amount'] = d['quantity'].where(d['activity_type'].eq('transfer'), 0.0)
    out = d.groupby(['hour','asset','token_address','token_decimals'], as_index=False).agg(
        event_count=('activity_type','size'),
        mint_count=('activity_type', lambda x: int((x=='mint').sum())),
        burn_count=('activity_type', lambda x: int((x=='burn').sum())),
        transfer_count=('activity_type', lambda x: int((x=='transfer').sum())),
        minted=('mint_amount','sum'), burned=('burn_amount','sum'), transferred=('transfer_amount','sum'),
    )
    out['net_issuance'] = out['minted'] - out['burned']
    return out


def build_protocol_event_counts(df: pd.DataFrame) -> pd.DataFrame:
    if '_empty' in df.columns or df.empty:
        return pd.DataFrame()
    d = df.copy()
    d['hour'] = pd.to_datetime(d['block_timestamp'], utc=True).dt.floor('h')
    return d.groupby(['hour','contract_key','contract_address','event_signature'], as_index=False).agg(
        event_count=('transaction_hash','size'), unique_txs=('transaction_hash','nunique')
    )


def decode_protocol_events(df: pd.DataFrame, registry: dict) -> dict[str, pd.DataFrame]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    contracts = registry['contracts']
    assets = registry['assets']
    asset_by_addr = {v['address'].lower(): (k, int(v['decimals'])) for k,v in assets.items()}

    if '_empty' in df.columns or df.empty:
        return {}

    for r in df.itertuples(index=False):
        sig = r.event_signature
        raw_named_args = getattr(r, 'decoded_args_json', None)
        try:
            named_args = json.loads(raw_named_args) if isinstance(raw_named_args, str) and raw_named_args else None
        except (TypeError, json.JSONDecodeError):
            named_args = None
        if isinstance(named_args, dict):
            args, ok = named_args, True
        else:
            args, ok = typed_args(sig, r.args_json, r.contract_key)
        base = {
            'block_number': int(r.block_number),
            'block_timestamp': r.block_timestamp,
            'hour': pd.Timestamp(r.block_timestamp).floor('h'),
            'transaction_hash': r.transaction_hash,
            'log_index': getattr(r, 'log_index', None),
            'source_table': getattr(r, 'source_table', None),
            'source_event_fingerprint': getattr(r, 'source_event_fingerprint', None),
            'contract_key': r.contract_key,
            'contract_address': r.contract_address,
            'event_signature': sig,
            'decode_ok': ok,
        }
        meta = contracts.get(r.contract_key, {})

        if meta.get('protocol') == 'uniswap_v3':
            if sig.startswith('Swap(') and ok:
                t0, t1 = meta['token0'], meta['token1']
                d0, d1 = int(assets[t0]['decimals']), int(assets[t1]['decimals'])
                try:
                    a0_raw, a1_raw = int(args['amount0']), int(args['amount1'])
                    a0, a1 = a0_raw/(10**d0), a1_raw/(10**d1)
                    row = {**base, 'protocol':'uniswap_v3','pool':r.contract_key,
                           'token0':t0,'token1':t1,'amount0':a0,'amount1':a1,
                           'amount0_raw':str(a0_raw),'amount1_raw':str(a1_raw),
                           'sqrt_price_x96':str(args['sqrtPriceX96']),
                           'liquidity_raw':str(args['liquidity']),'tick':int(args['tick'])}
                    if a1 != 0: row['execution_price_token0_per_token1'] = abs(a0/a1)
                    if a0 != 0: row['execution_price_token1_per_token0'] = abs(a1/a0)
                    buckets['dex_swap_events'].append(row)
                except Exception:
                    buckets['decode_failures'].append({**base, 'args_json':r.args_json})
            elif sig.startswith(('Mint(', 'Burn(', 'Flash(')):
                buckets['uniswap_liquidity_events'].append({**base, **args})

        elif meta.get('protocol') == 'curve':
            if sig == 'Transfer(address,address,uint256)' and ok:
                buckets['curve_lp_token_transfer_events'].append({**base, **args})
            elif sig == 'Approval(address,address,uint256)' and ok:
                buckets['curve_lp_token_approval_events'].append({**base, **args})
            elif sig.startswith(('TokenExchange(', 'TokenExchangeUnderlying(')) and ok:
                coins = meta.get('underlying_coins') if sig.startswith('TokenExchangeUnderlying(') else meta.get('coins')
                try:
                    si, bi = int(args['sold_id']), int(args['bought_id'])
                    sc, bc = coins[si], coins[bi]
                    sold = int(args['tokens_sold'])/(10**int(sc['decimals']))
                    bought = int(args['tokens_bought'])/(10**int(bc['decimals']))
                    row = {**base, 'protocol':'curve','pool':r.contract_key,
                           'sold_asset':sc['symbol'],'bought_asset':bc['symbol'],
                           'tokens_sold':sold,'tokens_bought':bought,
                           'tokens_sold_raw':str(args['tokens_sold']),'tokens_bought_raw':str(args['tokens_bought'])}
                    if sold != 0: row['execution_price_bought_per_sold'] = bought/sold
                    if bought != 0: row['execution_price_sold_per_bought'] = sold/bought
                    buckets['dex_swap_events'].append(row)
                except Exception:
                    buckets['decode_failures'].append({**base, 'args_json':r.args_json})
            elif sig.startswith(('AddLiquidity(', 'RemoveLiquidity(', 'RemoveLiquidityOne(', 'RemoveLiquidityImbalance(')) and ok:
                buckets['curve_liquidity_events'].append({**base, **args})
            elif ok:
                buckets['curve_parameter_events'].append({**base, **args})
            else:
                buckets['decode_failures'].append({**base, 'args_json':r.args_json})

        elif meta.get('protocol') == 'aave_v2':
            if sig in ('Transfer(address,address,uint256)', 'Approval(address,address,uint256)') and ok:
                buckets['aave_auxiliary_token_events'].append({**base, **args})
                continue
            if sig in ('TokensRescued(address,address,uint256)', 'Paused()', 'Unpaused()', 'Upgraded(address)'):
                buckets['aave_administration_events'].append({**base, **args})
                continue
            row = {**base, **args}
            # Convert event amounts where the reserve address determines decimals.
            for field in ('reserve','collateralAsset','debtAsset'):
                if field in row and isinstance(row[field], str): row[field] = row[field].lower()
            if 'reserve' in row and str(row['reserve']).lower() in asset_by_addr and 'amount' in row:
                sym, dec = asset_by_addr[str(row['reserve']).lower()]
                row['asset'] = sym
                try: row['amount_normalized'] = int(row['amount'])/(10**dec)
                except Exception: pass
            if sig.startswith('LiquidationCall('):
                c = asset_by_addr.get(str(row.get('collateralAsset','')).lower())
                d = asset_by_addr.get(str(row.get('debtAsset','')).lower())
                if c:
                    row['collateral_asset'] = c[0]
                    try: row['liquidated_collateral_normalized'] = int(row['liquidatedCollateralAmount'])/(10**c[1])
                    except Exception: pass
                if d:
                    row['debt_asset'] = d[0]
                    try: row['debt_to_cover_normalized'] = int(row['debtToCover'])/(10**d[1])
                    except Exception: pass
            buckets['aave_events'].append(row)

        elif r.contract_key == 'lido_withdrawal_queue':
            if sig == 'BatchMetadataUpdate(uint256,uint256)' and ok:
                buckets['lido_withdrawal_metadata_events'].append({**base, **args})
                continue
            if sig == 'Transfer(address,address,uint256)' and ok:
                buckets['lido_withdrawal_nft_transfer_events'].append({**base, **args})
            elif sig == 'Approval(address,address,uint256)' and ok:
                # ERC-721 Approval's `approved` field is an address, not a bool.
                row = {**base, **args}
                row['approved_address'] = row.pop('approved', None)
                row['token_id'] = row.pop('tokenId', None)
                buckets['lido_withdrawal_nft_token_approval_events'].append(row)
            elif sig == 'ApprovalForAll(address,address,bool)' and ok:
                # ERC-721 operator approval has a boolean `approved` field.
                buckets['lido_withdrawal_nft_operator_approval_events'].append({**base, **args})
            else:
                buckets['lido_withdrawal_events'].append({**base, **args})
        elif meta.get('protocol') == 'lido':
            if sig in ('ContractVersionSet(uint256)', 'EIP712StETHInitialized(address)', 'LidoLocatorSet(address)'):
                buckets['lido_administration_events'].append({**base, **args})
                continue
            if sig == 'Transfer(address,address,uint256)' and ok:
                asset = meta.get('asset', 'stETH')
                decimals = int(assets.get(asset, {}).get('decimals', 18))
                try:
                    raw_value = int(args['value'])
                    normalized = raw_value / (10 ** decimals)
                    zero = '0x' + ('0' * 40)
                    sender = str(args['from']).lower()
                    receiver = str(args['to']).lower()
                    activity_type = 'mint' if sender == zero else 'burn' if receiver == zero else 'transfer'
                    buckets['lido_token_transfer_events'].append({
                        **base, **args, 'asset': asset, 'token_decimals': decimals,
                        'value_raw': str(raw_value), 'value_normalized': normalized,
                        'activity_type': activity_type,
                    })
                except Exception:
                    buckets['decode_failures'].append({**base, 'args_json':r.args_json})
            elif sig == 'Approval(address,address,uint256)' and ok:
                buckets['token_approval_events'].append({**base, **args, 'asset': meta.get('asset', 'stETH')})
            else:
                buckets['lido_events'].append({**base, **args})
        elif not ok:
            buckets['decode_failures'].append({**base, 'args_json':r.args_json})

    return {k: pd.DataFrame(v) for k,v in buckets.items()}


def build_dex_hourly(swaps: pd.DataFrame) -> pd.DataFrame:
    if swaps is None or swaps.empty:
        return pd.DataFrame()
    d = swaps.copy()
    rows = []
    # Summarize without forcing all protocols into one price orientation.
    for (hour, pool, protocol), g in d.groupby(['hour','pool','protocol'], dropna=False):
        rows.append({
            'hour':hour, 'pool':pool, 'protocol':protocol,
            'swap_count':len(g), 'unique_txs':g['transaction_hash'].nunique(),
        })
    return pd.DataFrame(rows)


def build_raw_log_decode_quality_hourly(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or '_empty' in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d['hour'] = pd.to_datetime(d['block_timestamp'], utc=True).dt.floor('h')
    return d.groupby(
        ['hour', 'contract_key', 'decode_status', 'event_family', 'topic0'],
        as_index=False, dropna=False,
    ).agg(log_count=('transaction_hash', 'size'), unique_txs=('transaction_hash', 'nunique'))


def expand_rpc(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    d = df[df['state_kind'].eq(kind)].copy()
    if d.empty:
        return d
    vals = d['values_json'].map(lambda x: json.loads(x) if isinstance(x,str) and x else {})
    expanded = pd.json_normalize(vals)
    expanded.index = d.index
    base_cols = [c for c in d.columns if c not in ('values_json','errors_json')]
    return pd.concat([d[base_cols].reset_index(drop=True), expanded.reset_index(drop=True)], axis=1)


def publish_month(root, cfg, dataset, month, df, force=False):
    if df is None or df.empty:
        return
    out = root / 'processed' / dataset / f'year={month[:4]}' / f'month={month[5:7]}' / f'{dataset}_{month}.parquet'
    if out.exists() and not force:
        print(f'[SKIP] {dataset} {month}')
        return
    df = standardize_event_schema(dataset, df)
    pub = publish_dataframe(df, cfg, f'processed_{dataset}', month, out, cfg['output'].get('compression','zstd'))
    print(f"[OK] {dataset} {month}: {len(df):,} rows -> {out} ({pub['bytes']:,} bytes)")


def standardize_existing_event_files(root: Path, cfg: dict) -> None:
    """Repair older derived partitions without re-reading or changing frozen raw."""
    changed = 0
    for dataset in EVENT_EXTRA_COLUMNS:
        for path in sorted((root / 'processed' / dataset).glob('year=*/month=*/*.parquet')):
            frame = pd.read_parquet(path)
            normalized = standardize_event_schema(dataset, frame)
            expected = pa.Table.from_pandas(normalized, preserve_index=False).schema.remove_metadata()
            if pq.read_schema(path).remove_metadata().equals(expected):
                continue
            publish_month(root, cfg, dataset, month_partition(path), normalized, force=True)
            changed += 1
    print(f'Event-schema standardization complete: {changed} derived partition(s) rewritten.')


def main():
    parser = argparse.ArgumentParser(description='Build intermediate tables from the immutable Drive archive only.')
    parser.add_argument('--force', action='store_true', help='Replace existing derived processed month files.')
    parser.add_argument('--events-only', action='store_true',
                        help='Refresh event derivatives only; leave hourly RPC and token-flow products untouched.')
    parser.add_argument('--standardize-existing', action='store_true',
                        help='Normalize existing event-table schemas without rebuilding from raw.')
    args = parser.parse_args()
    cfg = load_yaml('config/extraction.yaml')
    registry = load_yaml('config/contracts.yaml')
    abi_config = load_yaml('config/raw_log_abis.yaml')
    contract_specs = make_contract_specs(registry, abi_config)
    root = resolve_storage_root(cfg, require=True)

    if args.standardize_existing:
        standardize_existing_event_files(root, cfg)
        return

    # Everything below reads only the frozen raw archive. BigQuery is intentionally not imported.
    if not args.events_only:
        for f in monthly_files(root, 'token_activity_events'):
            month = month_of(f)
            publish_month(root, cfg, 'token_flows_hourly', month, build_token_hourly(pd.read_parquet(f)), args.force)

    protocol_groups = files_by_month(root, 'protocol_events')
    raw_log_groups = files_by_month(root, 'raw_protocol_logs')
    raw_logs_exist = bool(raw_log_groups)
    decoded_groups = files_by_month(root, 'decoded_raw_protocol_logs', area='processed')

    event_months = sorted(set(protocol_groups) | set(decoded_groups) | set(raw_log_groups))
    for month in event_months:
        sources = []
        unified_raw_parts = []
        for f in protocol_groups.get(month, []):
            source_part = pd.read_parquet(f)
            if '_empty' not in source_part.columns:
                if {'source_table', 'decode_status'}.issubset(source_part.columns):
                    source_part = refresh_unknown_topics(
                        source_part, contract_specs, abi_config['decoder_version'],
                        abi_config.get('redecode_archived_signatures', ()))
                    integrated = source_part['source_table'].eq('raw_protocol_logs')
                    if integrated.any():
                        unified_raw_parts.append(source_part.loc[integrated].copy())
                        # Preserve every raw log in the canonical archive, but only
                        # successfully decoded events may enter event-level products.
                        source_part = source_part.loc[~integrated | source_part['decode_status'].eq('decoded')]
                sources.append(source_part)
        if unified_raw_parts:
            raw_part = pd.concat(unified_raw_parts, ignore_index=True, sort=False)
            publish_month(root, cfg, 'raw_log_decode_quality_hourly', month,
                          build_raw_log_decode_quality_hourly(raw_part), args.force)
        else:
            if raw_logs_exist and not decoded_groups.get(month):
                raw_month_files = raw_log_groups.get(month, [])
                if raw_month_files:
                    raise RuntimeError(
                        f'Raw logs exist for {month} but no ABI-decoded outputs were found. '
                        'Run: python src/decode_raw_logs.py'
                    )
            for f in decoded_groups.get(month, []):
                raw_part = pd.read_parquet(f)
                # Legacy split layout: unknown and malformed topics remain in the
                # decoded derivative; only successfully decoded records are events.
                publish_month(root, cfg, 'raw_log_decode_quality_hourly', month,
                              build_raw_log_decode_quality_hourly(raw_part), args.force)
                recognized = raw_part[raw_part['decode_status'].eq('decoded')].copy()
                if not recognized.empty:
                    sources.append(recognized)
        if not sources:
            continue
        raw = pd.concat(sources, ignore_index=True, sort=False)
        # BigQuery decoded_events has no log_index. Its content fingerprint can
        # match for two distinct logs in the same transaction, so preserve both.
        # The archive validator checks identity where a real log_index exists.
        publish_month(root, cfg, 'protocol_event_counts_hourly', month,
                      build_protocol_event_counts(raw), args.force)
        decoded = decode_protocol_events(raw, registry)
        for dataset, frame in decoded.items():
            publish_month(root, cfg, dataset, month, frame, args.force)
        if 'dex_swap_events' in decoded:
            publish_month(root, cfg, 'dex_activity_hourly', month,
                          build_dex_hourly(decoded['dex_swap_events']), args.force)

    kinds = {
        'chainlink_feed':'oracle_hourly',
        'uniswap_v3_pool':'uniswap_pool_state_hourly',
        'curve_pool':'curve_pool_state_hourly',
        'aave_v2_reserve':'aave_reserve_state_hourly',
        'lido_steth':'lido_steth_state_hourly',
        'lido_wsteth':'lido_wsteth_state_hourly',
        'lido_withdrawal_queue':'lido_withdrawal_queue_hourly',
    }
    if not args.events_only:
        for f in monthly_files(root, 'rpc_hourly'):
            month = month_of(f)
            rdf = pd.read_parquet(f)
            for kind, dataset in kinds.items():
                publish_month(root, cfg, dataset, month, expand_rpc(rdf, kind), args.force)

    print('\nIntermediate processed tables created entirely from the raw Drive archive. No BigQuery was used.')
    print('These tables preserve enough information for later market/reference reconstruction, labels, and TigerGraph exports.')


if __name__ == '__main__':
    main()

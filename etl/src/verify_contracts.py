from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from extract_rpc_state import RpcClient, calldata, decode_result
from utils import copy_to_drive_with_retries, load_yaml, resolve_storage_root, staging_path

ZERO = '0x0000000000000000000000000000000000000000'


def latest_block(rpc: RpcClient) -> int:
    rid = rpc.next_id; rpc.next_id += 1
    body = rpc._post({'jsonrpc':'2.0','id':rid,'method':'eth_blockNumber','params':[]})
    return int(body['result'], 16)


def get_code(rpc: RpcClient, address: str, block: int) -> str:
    rid = rpc.next_id; rpc.next_id += 1
    body = rpc._post({'jsonrpc':'2.0','id':rid,'method':'eth_getCode','params':[address, hex(block)]})
    if 'error' in body:
        raise RuntimeError(body['error'])
    return body['result']


def call1(rpc, address, signature, out_types, block, input_types=None, args=None):
    raw = rpc.eth_call(address, calldata(signature, input_types, args), block)
    return decode_result(raw, out_types)


def main():
    p = argparse.ArgumentParser(description='Verify configured AssetGuardX contracts against Ethereum mainnet before extraction.')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    cfg = load_yaml('config/extraction.yaml')
    registry = load_yaml('config/contracts.yaml')
    if not args.execute:
        print('Dry-run. This check requires read-only RPC. Re-run with --execute.')
        return
    url = os.getenv(cfg['rpc']['env_var'])
    if not url:
        raise SystemExit(f"Missing {cfg['rpc']['env_var']}")
    rpc = RpcClient(url, timeout=90, retries=6, batch_size=20, pause=0.03)
    block = latest_block(rpc)
    report = {'verified_at_utc':datetime.now(timezone.utc).isoformat(),'block_number':block,'checks':[]}

    # ERC20 metadata for token assets. Native ETH uses a sentinel address and has no ERC20 bytecode.
    for symbol, asset in registry['assets'].items():
        addr = asset['address']
        if asset.get('native'):
            report['checks'].append({
                'kind':'native_asset','key':symbol,'address':addr,'ok':True,'issues':[],
                'note':'Native ETH sentinel; ERC20 metadata checks intentionally skipped.'
            })
            continue
        item = {'kind':'token','key':symbol,'address':addr,'ok':True,'issues':[]}
        try:
            code = get_code(rpc, addr, block)
            if code == '0x': item['issues'].append('no runtime code')
            try:
                dec = int(call1(rpc, addr, 'decimals()', ['uint8'], block)[0])
                item['decimals_onchain'] = dec
                if dec != int(asset['decimals']): item['issues'].append(f'decimals expected {asset["decimals"]}, got {dec}')
            except Exception as exc:
                item['issues'].append(f'decimals() failed: {exc}')
            try:
                sym = str(call1(rpc, addr, 'symbol()', ['string'], block)[0])
                item['symbol_onchain'] = sym
                if symbol not in ('3CRV',) and sym.lower() != symbol.lower(): item['issues'].append(f'symbol expected {symbol}, got {sym}')
            except Exception as exc:
                item['issues'].append(f'symbol() failed: {exc}')
        except Exception as exc:
            item['issues'].append(str(exc))
        item['ok'] = not item['issues']
        report['checks'].append(item)

    for key, meta in registry['contracts'].items():
        addr = meta['address']
        item = {'kind':meta.get('role','contract'),'key':key,'address':addr,'ok':True,'issues':[]}
        try:
            if get_code(rpc, addr, block) == '0x': item['issues'].append('no runtime code')
            if meta.get('state_kind') == 'uniswap_v3_pool':
                t0 = str(call1(rpc, addr, 'token0()', ['address'], block)[0]).lower()
                t1 = str(call1(rpc, addr, 'token1()', ['address'], block)[0]).lower()
                fee = int(call1(rpc, addr, 'fee()', ['uint24'], block)[0])
                exp0 = registry['assets'][meta['token0']]['address'].lower()
                exp1 = registry['assets'][meta['token1']]['address'].lower()
                item.update({'token0_onchain':t0,'token1_onchain':t1,'fee_onchain':fee})
                if t0 != exp0: item['issues'].append(f'token0 mismatch: expected {exp0}, got {t0}')
                if t1 != exp1: item['issues'].append(f'token1 mismatch: expected {exp1}, got {t1}')
                if fee != int(meta['fee']): item['issues'].append(f'fee mismatch: expected {meta["fee"]}, got {fee}')
            elif meta.get('state_kind') == 'curve_pool':
                seen = []
                for i, coin in enumerate(meta['coins']):
                    try:
                        c = str(call1(rpc, addr, 'coins(uint256)', ['address'], block, ['uint256'], [i])[0]).lower()
                        seen.append(c)
                        expected = str(coin.get('address','')).lower()
                        # Native ETH pools may use a protocol-specific sentinel/zero address,
                        # so exact address equality is enforced only for ERC-20 coins.
                        if expected and not registry['assets'].get(coin['symbol'], {}).get('native') and c != expected:
                            item['issues'].append(f'coins({i}) mismatch for {coin["symbol"]}: expected {expected}, got {c}')
                    except Exception as exc:
                        item['issues'].append(f'coins({i}) failed: {exc}')
                item['coins_onchain'] = seen

                # Verify every configured point-in-time market quote now, before any
                # expensive historical extraction. This catches wrong Curve pool,
                # index, function variant (get_dy vs get_dy_underlying), or decimals.
                quote_checks = []
                for mq in meta.get('market_quotes', []):
                    target = mq['target']; quote = mq['quote']
                    ti = int(mq['target_index']); qi = int(mq['quote_index'])
                    method = mq['method']
                    target_dec = int(registry['assets'][target]['decimals'])
                    quote_dec = int(registry['assets'][quote]['decimals'])
                    signature = f'{method}(int128,int128,uint256)'
                    qitem = {'target':target,'quote':quote,'method':method,'target_index':ti,'quote_index':qi}
                    try:
                        forward = int(call1(rpc, addr, signature, ['uint256'], block,
                                            ['int128','int128','uint256'], [ti, qi, 10 ** target_dec])[0])
                        reverse = int(call1(rpc, addr, signature, ['uint256'], block,
                                            ['int128','int128','uint256'], [qi, ti, 10 ** quote_dec])[0])
                        qitem.update({'forward_raw':str(forward),'reverse_raw':str(reverse)})
                        if forward <= 0 or reverse <= 0:
                            item['issues'].append(
                                f'{method} quote {target}<->{quote} returned non-positive value(s): {forward}, {reverse}'
                            )
                    except Exception as exc:
                        qitem['error'] = str(exc)
                        item['issues'].append(f'{method} quote verification failed for {target}<->{quote}: {exc}')
                    quote_checks.append(qitem)
                if quote_checks:
                    item['market_quote_checks'] = quote_checks
            elif meta.get('state_kind') == 'lido_withdrawal_queue':
                last_id = int(call1(rpc, addr, 'getLastRequestId()', ['uint256'], block)[0])
                last_finalized = int(call1(rpc, addr, 'getLastFinalizedRequestId()', ['uint256'], block)[0])
                item['last_request_id'] = last_id
                item['last_finalized_request_id'] = last_finalized
            elif meta.get('state_kind') == 'lido_wsteth':
                ratio = int(call1(rpc, addr, 'stEthPerToken()', ['uint256'], block)[0])
                item['steth_per_wsteth_raw'] = str(ratio)
                if ratio <= 0:
                    item['issues'].append('stEthPerToken() returned non-positive value')
            elif meta.get('state_kind') == 'lido_steth':
                pooled = int(call1(rpc, addr, 'getTotalPooledEther()', ['uint256'], block)[0])
                shares = int(call1(rpc, addr, 'getTotalShares()', ['uint256'], block)[0])
                supply = int(call1(rpc, addr, 'totalSupply()', ['uint256'], block)[0])
                item.update({'total_pooled_ether_raw':str(pooled),'total_shares_raw':str(shares),'total_supply_raw':str(supply)})
                if pooled <= 0 or shares <= 0 or supply <= 0:
                    item['issues'].append('Lido stETH accounting call returned non-positive value')
            elif meta.get('state_kind') == 'chainlink_feed':
                dec = int(call1(rpc, addr, 'decimals()', ['uint8'], block)[0])
                round_data = call1(rpc, addr, 'latestRoundData()', ['uint80','int256','uint256','uint256','uint80'], block)
                answer = int(round_data[1]); updated_at = int(round_data[3])
                item.update({'decimals_onchain':dec,'latest_answer_raw':str(answer),'latest_updated_at':updated_at})
                if dec != int(meta.get('decimals', dec)):
                    item['issues'].append(f'Chainlink decimals expected {meta.get("decimals")}, got {dec}')
                if answer <= 0 or updated_at <= 0:
                    item['issues'].append('Chainlink latestRoundData returned non-positive answer/timestamp')
            elif meta.get('state_kind') == 'aave_v2_data_provider':
                for reserve in meta['reserves']:
                    vals = call1(rpc, addr, 'getReserveTokensAddresses(address)', ['address','address','address'], block, ['address'], [reserve['address']])
                    if any(str(v).lower() == ZERO for v in vals):
                        item['issues'].append(f"Aave {reserve['symbol']} returned a zero reserve token address")
        except Exception as exc:
            item['issues'].append(str(exc))
        item['ok'] = not item['issues']
        report['checks'].append(item)

    report['all_ok'] = all(x['ok'] for x in report['checks'])
    for x in report['checks']:
        print(('[OK]   ' if x['ok'] else '[FAIL] ') + f"{x['key']} {x['address']}")
        for issue in x['issues']:
            print('       - ' + issue)

    storage_root = resolve_storage_root(cfg, require=True)
    stage = staging_path(cfg, 'metadata', 'verification', 'contract_verification.json')
    stage.write_text(json.dumps(report, indent=2), encoding='utf-8')
    out = storage_root / 'metadata' / 'contract_verification.json'
    copy_to_drive_with_retries(stage, out, cfg)
    print(f'Verification report -> {out}')
    if not report['all_ok']:
        raise SystemExit('One or more contract checks failed. Do not start the full extraction until reviewed.')


if __name__ == '__main__':
    main()

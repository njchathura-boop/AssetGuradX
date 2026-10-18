from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from utils import load_yaml, parse_utc, publish_dataframe, resolve_storage_root


def read_processed(root: Path, dataset: str) -> pd.DataFrame:
    files=sorted((root/'processed'/dataset).glob('year=*/month=*/*.parquet'))
    if not files: return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)


def asnum(s): return pd.to_numeric(s, errors='coerce')


def unix_seconds(timestamps: pd.Series) -> pd.Series:
    """Convert timezone-aware Parquet timestamps to Unix seconds independent of dtype unit."""
    parsed = pd.to_datetime(timestamps, utc=True, errors='coerce')
    return parsed.map(lambda value: value.timestamp() if pd.notna(value) else np.nan)


def hourly_grid(cfg, assets):
    start=parse_utc(cfg['study_window']['start'])
    end=parse_utc(cfg['study_window']['validation_end_exclusive'])
    hours=pd.date_range(start=start,end=end,freq='h',inclusive='left')
    return pd.MultiIndex.from_product([assets,hours],names=['asset','hour']).to_frame(index=False)


def oracle_series(oracle, entity_key, prefix):
    d=oracle[oracle['entity_key'].eq(entity_key)].copy()
    if d.empty: return pd.DataFrame(columns=['hour',f'{prefix}_usd',f'{prefix}_oracle_age_seconds'])
    d['hour']=pd.to_datetime(d['hour'],utc=True)
    d[f'{prefix}_usd']=asnum(d.get('reference_price'))
    updated=asnum(d.get('oracle_updated_at_unix'))
    if 'snapshot_block_timestamp' in d.columns:
        snapshot=pd.to_datetime(d['snapshot_block_timestamp'],utc=True,errors='coerce')
        snapshot_unix=unix_seconds(snapshot)
    else:
        # Older archives lack the exact anchor time. Use the end of the UTC hour
        # as an explicit approximation instead of treating the hour start as the snapshot.
        hour=pd.to_datetime(d['hour'],utc=True,errors='coerce')
        snapshot_unix=unix_seconds(hour) + 3599
    # Keep negative ages visible: they indicate an impossible/future oracle update
    # relative to the sampled historical block and must not be disguised as age zero.
    d[f'{prefix}_oracle_age_seconds']=snapshot_unix-updated
    return d[['hour',f'{prefix}_usd',f'{prefix}_oracle_age_seconds']].drop_duplicates('hour',keep='last')


def uniswap_market(state, registry, eth):
    if state.empty: return pd.DataFrame(columns=['asset','hour','market_price_usd','market_observed_hour','market_source'])
    assets=registry['assets']; rows=[]
    d=state.copy(); d['hour']=pd.to_datetime(d['hour'],utc=True)
    ethmap=eth.set_index('hour')['eth_usd']
    for r in d.itertuples(index=False):
        meta=registry['contracts'].get(r.entity_key,{})
        if meta.get('state_kind')!='uniswap_v3_pool': continue
        try:
            sqrt=float(getattr(r,'sqrt_price_x96'))/(2**96)
            t0,t1=meta['token0'],meta['token1']
            dec0=int(assets[t0]['decimals']); dec1=int(assets[t1]['decimals'])
            t1_per_t0=(sqrt*sqrt)*(10**(dec0-dec1))
            ethusd=ethmap.get(pd.Timestamp(r.hour))
            if pd.isna(ethusd): continue
            # Selected pools all quote target asset against WETH.
            if t1=='WETH' and assets.get(t0,{}).get('target_asset'):
                rows.append({'asset':t0,'hour':r.hour,'market_price_usd':t1_per_t0*ethusd,
                             'market_observed_hour':r.hour,'market_source':r.entity_key})
            elif t0=='WETH' and assets.get(t1,{}).get('target_asset') and t1_per_t0:
                rows.append({'asset':t1,'hour':r.hour,'market_price_usd':ethusd/t1_per_t0,
                             'market_observed_hour':r.hour,'market_source':r.entity_key})
        except Exception:
            continue
    return pd.DataFrame(rows)



def curve_state_market(state, registry, price_lookup):
    """Hourly Curve marginal prices from point-in-time get_dy/get_dy_underlying RPC quotes.

    Each configured quote stores both target->quote and quote->target for a one-token
    notional.  Their quote-per-target values are averaged to reduce one-sided fee/slippage
    bias.  Quote USD prices come only from the same/past hour anchors supplied in
    price_lookup; no future value is used.
    """
    if state.empty:
        return pd.DataFrame(columns=['asset','hour','market_price_usd','market_observed_hour','market_source'])
    rows=[]
    d=state.copy(); d['hour']=pd.to_datetime(d['hour'],utc=True)
    assets=registry['assets']
    for r in d.itertuples(index=False):
        meta=registry['contracts'].get(r.entity_key,{})
        for mq in meta.get('market_quotes',[]):
            target=mq['target']; quote=mq['quote']
            safe_target=target.lower().replace('-','_'); safe_quote=quote.lower().replace('-','_')
            fcol=f'quote_{safe_target}_{safe_quote}_forward_raw'
            rcol=f'quote_{safe_target}_{safe_quote}_reverse_raw'
            fraw=asnum(pd.Series([getattr(r,fcol,None)])).iloc[0]
            rraw=asnum(pd.Series([getattr(r,rcol,None)])).iloc[0]
            target_dec=int(assets[target]['decimals']); quote_dec=int(assets[quote]['decimals'])
            sides=[]
            if pd.notna(fraw) and fraw>0:
                sides.append(float(fraw)/(10**quote_dec))  # quote received for exactly 1 target
            if pd.notna(rraw) and rraw>0:
                target_received=float(rraw)/(10**target_dec)  # target received for exactly 1 quote
                if target_received>0:
                    sides.append(1.0/target_received)
            if not sides:
                continue
            quote_per_target=float(np.mean(sides))
            qusd=price_lookup.get((quote,pd.Timestamp(r.hour)))
            if qusd is None or pd.isna(qusd):
                continue
            rows.append({
                'asset':target,'hour':pd.Timestamp(r.hour),
                'market_price_usd':quote_per_target*float(qusd),
                'market_observed_hour':pd.Timestamp(r.hour),
                'market_source':f'{r.entity_key}:rpc_quote',
            })
    return pd.DataFrame(rows)

def curve_direct_observations(swaps, price_lookup):
    """Create target USD observations from Curve execution prices using already-anchored quote prices."""
    if swaps.empty: return pd.DataFrame()
    s=swaps[swaps['protocol'].eq('curve')].copy()
    s['hour']=pd.to_datetime(s['hour'],utc=True)
    rows=[]
    # Target, acceptable quote assets. Use causal hourly quote price supplied by price_lookup.
    rules={
        'DAI':['USDC'], 'USDT':['USDC'], 'FRAX':['USDC'], 'stETH':['ETH'],
        'LUSD':['USDC','DAI','USDT']
    }
    for r in s.itertuples(index=False):
        sold=getattr(r,'sold_asset',None); bought=getattr(r,'bought_asset',None)
        sold_amt=float(getattr(r,'tokens_sold',np.nan)); bought_amt=float(getattr(r,'tokens_bought',np.nan))
        if not (sold_amt>0 and bought_amt>0): continue
        for target,quotes in rules.items():
            quote=None; quote_per_target=None; target_amount=None; quote_amount=None
            if sold==target and bought in quotes:
                quote=bought; quote_per_target=bought_amt/sold_amt; target_amount=sold_amt; quote_amount=bought_amt
            elif bought==target and sold in quotes:
                quote=sold; quote_per_target=sold_amt/bought_amt; target_amount=bought_amt; quote_amount=sold_amt
            if quote is None: continue
            qkey=(quote,pd.Timestamp(r.hour))
            qusd=price_lookup.get(qkey)
            if qusd is None or pd.isna(qusd): continue
            rows.append({'asset':target,'hour':pd.Timestamp(r.hour),'market_price_usd':quote_per_target*float(qusd),
                         'market_observed_hour':pd.Timestamp(r.hour),'market_source':getattr(r,'pool','curve'),
                         'target_amount':target_amount,'quote_amount':quote_amount})
    if not rows: return pd.DataFrame()
    e=pd.DataFrame(rows)
    # Volume-weight swap observations by target amount.
    grouped=[]
    for (asset,hour),g in e.groupby(['asset','hour']):
        w=asnum(g['target_amount']).clip(lower=0)
        if w.sum()>0: px=np.average(asnum(g['market_price_usd']),weights=w)
        else: px=asnum(g['market_price_usd']).median()
        grouped.append({'asset':asset,'hour':hour,'market_price_usd':px,'market_observed_hour':hour,
                        'market_source':'|'.join(sorted(set(g['market_source'].astype(str))))})
    return pd.DataFrame(grouped)


def causal_fill_market(grid, observations):
    out=[]
    for asset,g in grid.groupby('asset',sort=False):
        x=g[['asset','hour']].copy().sort_values('hour')
        o=observations[observations['asset'].eq(asset)].sort_values('hour') if not observations.empty else pd.DataFrame()
        if not o.empty:
            o=o.drop_duplicates(['asset','hour'],keep='last')
            x=x.merge(o[['asset','hour','market_price_usd','market_observed_hour','market_source']],on=['asset','hour'],how='left')
        else:
            x['market_price_usd']=np.nan; x['market_observed_hour']=pd.NaT; x['market_source']=pd.NA
        observed=x['market_price_usd'].notna()
        x['market_price_observed_this_hour']=observed
        x['market_observed_hour']=x['market_observed_hour'].where(observed).ffill()
        x['market_source']=x['market_source'].astype('string').where(observed).ffill()
        x['market_price_usd']=x['market_price_usd'].ffill()
        x['market_price_age_hours']=(x['hour']-pd.to_datetime(x['market_observed_hour'],utc=True)).dt.total_seconds()/3600
        out.append(x)
    return pd.concat(out,ignore_index=True)


def add_reference(df, oracle, lido_steth, lido_wsteth):
    eth=oracle_series(oracle,'chainlink_eth_usd','eth')
    btc=oracle_series(oracle,'chainlink_btc_usd','btc')
    d=df.merge(eth,on='hour',how='left').merge(btc,on='hour',how='left')

    st=lido_steth.copy()
    if not st.empty:
        st['hour']=pd.to_datetime(st['hour'],utc=True)
        pooled=asnum(st.get('total_pooled_ether_raw'))
        supply=asnum(st.get('total_supply_raw'))
        st['steth_eth_reference_ratio']=pooled/supply.replace(0,np.nan)
        d=d.merge(st[['hour','steth_eth_reference_ratio']].drop_duplicates('hour',keep='last'),on='hour',how='left')
    else: d['steth_eth_reference_ratio']=np.nan
    d['steth_eth_reference_ratio']=d['steth_eth_reference_ratio'].fillna(1.0)

    ws=lido_wsteth.copy()
    if not ws.empty:
        ws['hour']=pd.to_datetime(ws['hour'],utc=True)
        if 'steth_per_wsteth' not in ws:
            ws['steth_per_wsteth']=asnum(ws.get('steth_per_wsteth_raw'))/1e18
        d=d.merge(ws[['hour','steth_per_wsteth']].drop_duplicates('hour',keep='last'),on='hour',how='left')
    else: d['steth_per_wsteth']=np.nan

    d['reference_price_usd']=np.nan
    stable=d['asset'].isin(['USDC','USDT','DAI','FRAX','LUSD'])
    d.loc[stable,'reference_price_usd']=1.0
    d.loc[d['asset'].eq('WBTC'),'reference_price_usd']=d.loc[d['asset'].eq('WBTC'),'btc_usd']
    stmask=d['asset'].eq('stETH')
    d.loc[stmask,'reference_price_usd']=d.loc[stmask,'steth_eth_reference_ratio']*d.loc[stmask,'eth_usd']
    wmask=d['asset'].eq('wstETH')
    d.loc[wmask,'reference_price_usd']=d.loc[wmask,'steth_eth_reference_ratio']*d.loc[wmask,'steth_per_wsteth']*d.loc[wmask,'eth_usd']
    d['relative_deviation']=(d['market_price_usd']-d['reference_price_usd'])/d['reference_price_usd']
    d['market_missing']=d['market_price_usd'].isna()
    d['reference_missing']=d['reference_price_usd'].isna()
    return d


def add_token_flows(df, flows):
    if flows.empty: return df
    f=flows.copy(); f['hour']=pd.to_datetime(f['hour'],utc=True)
    cols=['hour','asset','event_count','mint_count','burn_count','transfer_count','minted','burned','transferred','net_issuance']
    cols=[c for c in cols if c in f]
    d=df.merge(f[cols],on=['hour','asset'],how='left')
    fill=[c for c in cols if c not in ('hour','asset')]
    d[fill]=d[fill].fillna(0)
    return d


def add_aave(df, aave):
    if aave.empty: return df
    a=aave.copy(); a['hour']=pd.to_datetime(a['hour'],utc=True)
    dec=asnum(a['token_decimals'])
    for raw,name in [('available_liquidity_raw','aave_available'),('total_a_token_raw','aave_supplied'),
                     ('total_stable_debt_raw','aave_stable_debt'),('total_variable_debt_raw','aave_variable_debt')]:
        a[name]=asnum(a.get(raw))/(10.0**dec)
    a['aave_total_debt']=a['aave_stable_debt']+a['aave_variable_debt']
    denom=a['aave_available']+a['aave_total_debt']
    a['aave_utilisation']=a['aave_total_debt']/denom.replace(0,np.nan)
    keep=['hour','asset','aave_available','aave_supplied','aave_stable_debt','aave_variable_debt','aave_total_debt','aave_utilisation',
          'ltv','liquidation_threshold','liquidation_bonus','reserve_factor']
    keep=[c for c in keep if c in a]
    return df.merge(a[keep].drop_duplicates(['hour','asset'],keep='last'),on=['hour','asset'],how='left')


def add_lido_queue(df, q):
    if q.empty: return df
    q=q.copy(); q['hour']=pd.to_datetime(q['hour'],utc=True)
    q['lido_queue_unfinalized_steth']=asnum(q.get('unfinalized_steth_raw'))/1e18
    q['lido_queue_locked_eth']=asnum(q.get('locked_ether_raw'))/1e18
    q['lido_queue_unfinalized_requests']=asnum(q.get('unfinalized_request_number'))
    cols=['hour','lido_queue_unfinalized_steth','lido_queue_locked_eth','lido_queue_unfinalized_requests','is_bunker_mode']
    cols=[c for c in cols if c in q]
    q=q[cols].drop_duplicates('hour',keep='last')
    mask=df['asset'].isin(['stETH','wstETH'])
    left=df.loc[mask].merge(q,on='hour',how='left')
    return pd.concat([df.loc[~mask],left],ignore_index=True).sort_values(['asset','hour'])


def standardize_canonical_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Keep nullable fields typed even in months before their source was active."""
    df['is_bunker_mode'] = df.get(
        'is_bunker_mode', pd.Series(pd.NA, index=df.index)
    ).astype('boolean')
    return df


def main():
    parser=argparse.ArgumentParser(description='Build canonical hourly asset observations from processed archive data.')
    parser.add_argument('--force',action='store_true',help='Rebuild and replace existing processed/asset_hourly monthly outputs.')
    args=parser.parse_args()
    cfg=load_yaml('config/extraction.yaml'); reg=load_yaml('config/contracts.yaml')
    root=resolve_storage_root(cfg,require=True)
    target=[k for k,v in reg['assets'].items() if v.get('target_asset')]
    grid=hourly_grid(cfg,target)

    oracle=read_processed(root,'oracle_hourly')
    uni=read_processed(root,'uniswap_pool_state_hourly')
    curve_state=read_processed(root,'curve_pool_state_hourly')
    swaps=read_processed(root,'dex_swap_events')
    flows=read_processed(root,'token_flows_hourly')
    aave=read_processed(root,'aave_reserve_state_hourly')
    lst=read_processed(root,'lido_steth_state_hourly')
    wst=read_processed(root,'lido_wsteth_state_hourly')
    queue=read_processed(root,'lido_withdrawal_queue_hourly')
    required_tables={'oracle_hourly':oracle,'uniswap_pool_state_hourly':uni,'curve_pool_state_hourly':curve_state,
                     'lido_steth_state_hourly':lst,'lido_wsteth_state_hourly':wst,'aave_reserve_state_hourly':aave}
    missing=[name for name,frame in required_tables.items() if frame.empty]
    if missing:
        raise SystemExit('Required processed tables are missing: ' + ', '.join(missing) + '. Run run_process.ps1 first.')

    eth=oracle_series(oracle,'chainlink_eth_usd','eth')
    uni_obs=uniswap_market(uni,reg,eth)

    # Uniswap provides continuous hourly USD anchors for USDC, WBTC and wstETH.
    # ETH/USD comes directly from Chainlink.  Curve then supplies continuous marginal
    # hourly quotes for DAI, USDT, FRAX, LUSD and stETH.
    anchors=causal_fill_market(grid,uni_obs)
    lookup={(r.asset,pd.Timestamp(r.hour)):r.market_price_usd for r in anchors.itertuples(index=False) if pd.notna(r.market_price_usd)}
    for r in eth.itertuples(index=False):
        if pd.notna(r.eth_usd): lookup[('ETH',pd.Timestamp(r.hour))]=r.eth_usd

    curve_rpc_obs=curve_state_market(curve_state,reg,lookup)
    # Event execution prices are retained as a fallback/reconciliation source only.
    curve_swap_obs=curve_direct_observations(swaps,lookup)
    curve_rpc_keys=set(zip(curve_rpc_obs.get('asset',[]),curve_rpc_obs.get('hour',[]))) if not curve_rpc_obs.empty else set()
    if not curve_swap_obs.empty and curve_rpc_keys:
        keep=[(a,h) not in curve_rpc_keys for a,h in zip(curve_swap_obs['asset'],curve_swap_obs['hour'])]
        curve_swap_obs=curve_swap_obs.loc[keep]
    obs=pd.concat([uni_obs,curve_rpc_obs,curve_swap_obs],ignore_index=True,sort=False)
    if not obs.empty:
        obs=obs.drop_duplicates(['asset','hour'],keep='first')
    canon=causal_fill_market(grid,obs)
    canon=add_reference(canon,oracle,lst,wst)
    canon=add_token_flows(canon,flows)
    canon=add_aave(canon,aave)
    canon=add_lido_queue(canon,queue)
    canon=standardize_canonical_schema(canon)
    canon['market_price_forward_filled']=canon['market_price_usd'].notna() & canon['market_price_age_hours'].fillna(0).gt(0)

    # Publish monthly to keep the same resumable partition pattern.
    canon['month']=canon['hour'].dt.strftime('%Y-%m')
    for month,g in canon.groupby('month'):
        g=g.drop(columns=['month']).sort_values(['asset','hour'])
        out=root/'processed'/'asset_hourly'/f'year={month[:4]}'/f'month={month[5:7]}'/f'asset_hourly_{month}.parquet'
        if out.exists() and not args.force:
            print(f'[SKIP] asset_hourly {month}'); continue
        pub=publish_dataframe(g,cfg,'processed_asset_hourly',month,out,cfg['output'].get('compression','zstd'))
        print(f'[OK] asset_hourly {month}: {len(g):,} rows -> {out} ({pub["bytes"]:,} bytes)')
    print('\nCanonical asset_hourly created with explicit market-price age/missingness and hourly Curve RPC quotes. No future data was used for forward filling.')
    print('Review market-source coverage and staleness before generating final thesis labels.')


if __name__=='__main__':
    main()

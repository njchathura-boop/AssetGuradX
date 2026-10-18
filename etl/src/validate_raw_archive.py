from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from utils import active_at, load_yaml, month_windows, parse_utc, resolve_storage_root, iso_slug


CONSOLIDATION_VERSION = 'assetguardx-protocol-events/v2'


def duplicate_indexed_logs(df: pd.DataFrame, address_column: str) -> int:
    """Only an actual chain log index identifies a log within a transaction."""
    if 'log_index' not in df or not df['log_index'].notna().any():
        return 0
    indexed = df.loc[df['log_index'].notna()]
    return int(indexed.duplicated(['transaction_hash', 'log_index', address_column]).sum())


def expected_windows(cfg):
    s = parse_utc(cfg['study_window']['start'])
    e = parse_utc(cfg['study_window']['validation_end_exclusive'])
    return [(iso_slug(x), x, y) for x,y in month_windows(s,e)]


def path_for(root: Path, dataset: str, month: str) -> Path:
    return root / 'raw' / dataset / f'year={month[:4]}' / f'month={month[5:7]}' / f'{dataset}_{month}.parquet'


def main():
    p = argparse.ArgumentParser(description='Validate that AssetGuardX-Raw v2 is complete enough to freeze BigQuery usage.')
    p.add_argument('--deep', action='store_true', help='Open each Parquet and run row/duplicate/hour sanity checks.')
    args = p.parse_args()
    cfg = load_yaml('config/extraction.yaml')
    registry = load_yaml('config/contracts.yaml')
    root = resolve_storage_root(cfg, require=True)
    windows = expected_windows(cfg)
    required = ['protocol_events','token_activity_events','hourly_blocks','rpc_hourly']
    missing = []
    hard_errors = []
    print(f'Archive root: {root}')
    for month, start, end in windows:
        print(f'\n{month}')
        expected_hours = int((end-start).total_seconds() // 3600)
        for ds in required:
            pth = path_for(root, ds, month)
            if not pth.exists():
                print(f'  MISSING {ds}')
                missing.append((month,ds))
                continue
            size = pth.stat().st_size
            msg = f'  OK {ds}: {size/(1024**2):.1f} MiB'
            if args.deep:
                df = pd.read_parquet(pth)
                msg += f', {len(df):,} rows'
                time_column = 'hour' if ds in {'hourly_blocks', 'rpc_hourly'} else 'block_timestamp'
                if time_column in df:
                    times = pd.to_datetime(df[time_column], utc=True, errors='coerce')
                    outside = int((times.isna() | (times < start) | (times >= end)).sum())
                    if outside:
                        hard_errors.append((month, ds, f'outside_month_or_invalid_time={outside}'))
                        msg += f' ERROR outside_month_or_invalid_time={outside}'
                if ds == 'hourly_blocks' and len(df):
                    dup = int(df.duplicated(['hour']).sum())
                    if dup:
                        hard_errors.append((month,ds,f'duplicate_hours={dup}'))
                        msg += f' ERROR duplicate_hours={dup}'
                    if len(df) != expected_hours:
                        hard_errors.append((month,ds,f'hours={len(df)}, expected={expected_hours}'))
                        msg += f' ERROR expected_hours={expected_hours}'
                if ds == 'protocol_events' and len(df):
                    parquet = pq.ParquetFile(pth)
                    try:
                        metadata = parquet.schema_arrow.metadata or {}
                    finally:
                        parquet.close()
                    version = metadata.get(b'assetguardx_consolidation_version', b'').decode()
                    if version != CONSOLIDATION_VERSION:
                        hard_errors.append((month, ds, f'consolidation_version={version or "missing"}'))
                        msg += f' ERROR consolidation_version={version or "missing"}'
                    # decoded_events has no log_index. Several distinct logs in one
                    # transaction can therefore share its null-index key or content
                    # fingerprint; neither is evidence that a row is duplicated.
                    dup = duplicate_indexed_logs(df, 'contract_address')
                    if dup:
                        hard_errors.append((month, ds, f'duplicate_events={dup}'))
                        msg += f' ERROR duplicate_events={dup}'
                if ds == 'token_activity_events' and len(df):
                    dup = duplicate_indexed_logs(df, 'token_address')
                    if dup:
                        hard_errors.append((month, ds, f'duplicate_transfers={dup}'))
                        msg += f' ERROR duplicate_transfers={dup}'
                if ds == 'rpc_hourly' and len(df):
                    partial = int((df['quality_status'] != 'ok').sum()) if 'quality_status' in df else 0
                    dup = int(df.duplicated(['hour','entity_key']).sum()) if {'hour','entity_key'}.issubset(df.columns) else 0
                    if partial:
                        hard_errors.append((month,ds,f'partial_rows={partial}'))
                        msg += f' ERROR partial_rows={partial:,}'
                    if dup:
                        hard_errors.append((month,ds,f'duplicate_entity_hours={dup}'))
                        msg += f' ERROR duplicate_entity_hours={dup:,}'

                    # Every configured point-in-time state entity must have one row for every
                    # active hourly anchor. Aave expands one DataProvider into one row per reserve.
                    hb_path = path_for(root, 'hourly_blocks', month)
                    if hb_path.exists():
                        hb = pd.read_parquet(hb_path, columns=['hour'])
                        expected_rpc = 0
                        for hour_value in pd.to_datetime(hb['hour'], utc=True):
                            dt = hour_value.to_pydatetime()
                            for meta in registry['contracts'].values():
                                kind = meta.get('state_kind')
                                if not kind or not active_at(meta, dt):
                                    continue
                                if kind == 'aave_v2_data_provider':
                                    expected_rpc += len(meta.get('reserves', []))
                                else:
                                    expected_rpc += 1
                        if len(df) != expected_rpc:
                            hard_errors.append((month,ds,f'entity_hours={len(df)}, expected={expected_rpc}'))
                            msg += f' ERROR expected_entity_hours={expected_rpc:,}'
            print(msg)
    print('\n--- RESULT ---')
    if missing:
        print(f'NOT READY TO FREEZE: {len(missing)} monthly dataset files are missing.')
        raise SystemExit(2)
    if hard_errors:
        print(f'NOT READY TO FREEZE: {len(hard_errors)} hard validation errors found.')
        for x in hard_errors[:20]: print('  ',x)
        raise SystemExit(3)
    print('RAW ARCHIVE COMPLETE for the frozen scope. Canonical processing, labels, and TigerGraph exports can now be rebuilt from Drive without BigQuery.')


if __name__ == '__main__':
    main()

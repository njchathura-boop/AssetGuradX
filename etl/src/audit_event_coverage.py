from __future__ import annotations

import argparse
from collections import Counter
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from eth_utils import keccak

from utils import load_yaml, resolve_storage_root


def main():
    ap = argparse.ArgumentParser(description='Audit whether configured event sources actually produced rows in the frozen raw archive.')
    ap.add_argument('--strict', action='store_true', help='Fail if a required market/event contract has zero rows.')
    args = ap.parse_args()
    cfg = load_yaml('config/extraction.yaml')
    registry = load_yaml('config/contracts.yaml')
    root = resolve_storage_root(cfg, require=True)
    files = sorted((root/'raw'/'protocol_events').glob('year=*/month=*/*.parquet'))
    if not files:
        raise SystemExit('No raw protocol_events found.')

    # The selected raw-log contracts retain *all* topics. A configured event
    # absent there is different from an event absent only from decoded_events.
    raw_contracts = set(cfg['bigquery'].get('raw_log_contracts', []))
    raw_topic_rows = pl.scan_parquet([str(path) for path in files], glob=False).filter(
        pl.col('source_table') == 'raw_protocol_logs'
    ).group_by(['contract_key', 'topic0']).agg(pl.len().alias('rows')).collect()
    raw_topic_counts = {
        (row['contract_key'], row['topic0']): int(row['rows'])
        for row in raw_topic_rows.to_dicts()
    }

    contract_counts = Counter()
    signature_counts = Counter()
    source_counts = Counter()
    for f in files:
        columns = ['contract_key', 'event_signature']
        parquet_columns = set(pq.ParquetFile(f).schema_arrow.names)
        if 'source_table' in parquet_columns:
            columns.append('source_table')
        df = pd.read_parquet(f, columns=columns)
        contract_counts.update(df['contract_key'].astype(str))
        signature_counts.update(zip(df['contract_key'].astype(str), df['event_signature'].astype(str)))
        if 'source_table' in df.columns:
            source_counts.update(df['source_table'].fillna('unknown').astype(str))

    required_zero = []
    missing_decodes = []
    print('=== Contract/event coverage ===')
    for key, meta in registry['contracts'].items():
        sigs = meta.get('decoded_event_signatures') or []
        if not sigs:
            continue
        n = contract_counts[key]
        print(f'{key}: {n:,} selected event rows')
        if n == 0:
            required_zero.append(key)
        for sig in sigs:
            c = signature_counts[(key, sig)]
            if c:
                assessment = 'observed'
            elif key in raw_contracts:
                topic0 = '0x' + keccak(text=sig).hex()
                if raw_topic_counts.get((key, topic0), 0):
                    assessment = 'raw_topic_present_but_decoded_event_missing'
                    missing_decodes.append((key, sig))
                else:
                    assessment = 'zero_in_selected_raw_archive'
            else:
                assessment = 'unverified_no_raw_log_fallback'
            print(f'  [{assessment}] {sig}: {c:,}')

    print('\nZero-row configured event contracts:', ', '.join(required_zero) if required_zero else 'none')
    if source_counts:
        print('Rows by source table:', ', '.join(f'{source}: {count:,}' for source, count in sorted(source_counts.items())))
    if (required_zero or missing_decodes) and args.strict:
        raise SystemExit(2)
    print('Audit complete. Zero in the selected raw archive is not proof of zero on-chain activity;')
    print('unverified signatures require an independent source before claiming complete coverage.')


if __name__ == '__main__':
    main()

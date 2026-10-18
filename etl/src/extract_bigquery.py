from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import bigquery

from utils import (
    append_jsonl,
    copy_to_drive_with_retries,
    iso_slug,
    load_yaml,
    month_windows,
    parse_utc,
    quoted_sql_strings,
    resolve_storage_root,
    staging_path,
)

ZERO = '0x0000000000000000000000000000000000000000'


def build_protocol_events_sql(table: str, contracts: dict) -> str:
    filters, cases = [], []
    for key, meta in contracts.items():
        sigs = meta.get('decoded_event_signatures') or []
        if not sigs:
            continue
        addr = meta['address'].lower()
        filters.append(f"(LOWER(address) = '{addr}' AND event_signature IN ({quoted_sql_strings(sigs)}))")
        cases.append(f"WHEN LOWER(address) = '{addr}' THEN '{key}'")
    if not filters:
        raise ValueError('No decoded event contracts configured.')
    return f"""
SELECT
  block_number,
  block_timestamp,
  transaction_hash,
  CAST(NULL AS INT64) AS log_index,
  'decoded_events' AS source_table,
  TO_HEX(SHA256(CONCAT(
    CAST(block_number AS STRING), '|', transaction_hash, '|', LOWER(address), '|',
    event_signature, '|', TO_JSON_STRING(args)
  ))) AS source_event_fingerprint,
  LOWER(address) AS contract_address,
  CASE {' '.join(cases)} ELSE 'unknown' END AS contract_key,
  event_signature,
  TO_JSON_STRING(args) AS args_json
FROM `{table}`
WHERE block_timestamp >= @start_ts
  AND block_timestamp < @end_ts
  AND ({' OR '.join(filters)})
"""


def token_activity_entries(registry: dict) -> list[dict]:
    assets = registry['assets']
    out = []
    for symbol, policy in registry.get('token_activity', {}).items():
        asset = assets[symbol]
        out.append({
            'symbol': symbol,
            'address': asset['address'].lower(),
            'decimals': int(asset['decimals']),
            'mode': policy['mode'],
        })
    return out


def build_token_activity_sql(table: str, registry: dict) -> str:
    entries = token_activity_entries(registry)
    addresses = [x['address'] for x in entries]
    all_addresses = [x['address'] for x in entries if x['mode'] == 'all']
    sparse_addresses = [x['address'] for x in entries if x['mode'] == 'mint_burn_only']
    asset_case = ' '.join(f"WHEN LOWER(address) = '{x['address']}' THEN '{x['symbol']}'" for x in entries)
    dec_case = ' '.join(f"WHEN LOWER(address) = '{x['address']}' THEN {x['decimals']}" for x in entries)
    all_sql = quoted_sql_strings(all_addresses) if all_addresses else "''"
    sparse_sql = quoted_sql_strings(sparse_addresses) if sparse_addresses else "''"
    return f"""
SELECT
  block_number,
  block_timestamp,
  transaction_hash,
  CAST(NULL AS INT64) AS log_index,
  'token_transfers' AS source_table,
  TO_HEX(SHA256(CONCAT(
    CAST(block_number AS STRING), '|', transaction_hash, '|', LOWER(address), '|',
    LOWER(from_address), '|', LOWER(to_address), '|', CAST(quantity AS STRING)
  ))) AS source_event_fingerprint,
  LOWER(address) AS token_address,
  CASE {asset_case} ELSE 'unknown' END AS asset,
  CASE {dec_case} ELSE NULL END AS token_decimals,
  LOWER(from_address) AS from_address,
  LOWER(to_address) AS to_address,
  CAST(quantity AS STRING) AS quantity_raw,
  CASE
    WHEN LOWER(from_address) = '{ZERO}' THEN 'mint'
    WHEN LOWER(to_address) = '{ZERO}' THEN 'burn'
    ELSE 'transfer'
  END AS activity_type
FROM `{table}`
WHERE block_timestamp >= @start_ts
  AND block_timestamp < @end_ts
  AND LOWER(address) IN ({quoted_sql_strings(addresses)})
  AND (
    LOWER(address) IN ({all_sql})
    OR (
      LOWER(address) IN ({sparse_sql})
      AND (LOWER(from_address) = '{ZERO}' OR LOWER(to_address) = '{ZERO}')
    )
  )
"""


def build_hourly_blocks_sql(table: str) -> str:
    return f"""
WITH ranked AS (
  SELECT
    TIMESTAMP_TRUNC(block_timestamp, HOUR) AS hour,
    block_number,
    block_timestamp,
    ROW_NUMBER() OVER (
      PARTITION BY TIMESTAMP_TRUNC(block_timestamp, HOUR)
      ORDER BY block_timestamp DESC, block_number DESC
    ) AS rn
  FROM `{table}`
  WHERE block_timestamp >= @start_ts
    AND block_timestamp < @end_ts
)
SELECT hour, block_number, block_timestamp
FROM ranked
WHERE rn = 1
ORDER BY hour
"""


def build_raw_protocol_logs_sql(table: str, contracts: dict, contract_keys: list[str]) -> tuple[str, list[dict]]:
    """Build a bounded raw-log query, retaining all topics and data for audit/decode."""
    specs = []
    conditions = []
    cases = []
    seen_addresses = set()
    for key in contract_keys:
        meta = contracts.get(key)
        if not meta:
            raise ValueError(f'Unknown raw-log contract key in config: {key}')
        address = str(meta.get('address', '')).lower()
        if len(address) != 42 or not address.startswith('0x'):
            raise ValueError(f'Invalid Ethereum address for raw-log contract {key}: {address!r}')
        if address in seen_addresses:
            raise ValueError(f'Duplicate address in raw-log contract selection: {address}')
        seen_addresses.add(address)

        index = len(specs)
        address_param = f'contract_address_{index}'
        condition = f'LOWER(address) = @{address_param}'
        spec = {'key': key, 'address': address, 'address_param': address_param}
        active_from = meta.get('active_from')
        if active_from:
            active_param = f'contract_active_from_{index}'
            condition += f' AND block_timestamp >= @{active_param}'
            spec['active_from'] = active_from
            spec['active_from_param'] = active_param
        active_to = meta.get('active_to_exclusive')
        if active_to:
            active_param = f'contract_active_to_{index}'
            condition += f' AND block_timestamp < @{active_param}'
            spec['active_to_exclusive'] = active_to
            spec['active_to_param'] = active_param
        conditions.append(f'({condition})')
        cases.append(f"WHEN LOWER(address) = @{address_param} THEN '{key}'")
        specs.append(spec)

    if not specs:
        raise ValueError('At least one contract is required for raw-log extraction.')

    sql = f"""
SELECT
  block_number,
  block_timestamp,
  transaction_hash,
  log_index,
  LOWER(address) AS contract_address,
  CASE {' '.join(cases)} ELSE 'unknown' END AS contract_key,
  LOWER(topics[SAFE_OFFSET(0)]) AS topic0,
  topics,
  data,
  TO_HEX(SHA256(TO_JSON_STRING(STRUCT(
    block_number AS block_number,
    transaction_hash AS transaction_hash,
    log_index AS log_index,
    LOWER(address) AS contract_address,
    topics AS topics,
    data AS data
  )))) AS source_event_fingerprint
FROM `{table}`
WHERE block_timestamp >= @start_ts
  AND block_timestamp < @end_ts
  AND ({' OR '.join(conditions)})
"""
    return sql, specs



def table_columns(client: bigquery.Client, table_name: str) -> set[str]:
    table = client.get_table(table_name)
    return {field.name for field in table.schema}


def assert_columns(table_name: str, columns: set[str], required: set[str]) -> None:
    missing = sorted(required - columns)
    if missing:
        raise RuntimeError(f'BigQuery schema mismatch for {table_name}; missing required columns: {missing}')


def readable_parquet(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return False
        _ = pq.ParquetFile(path).metadata.num_rows
        return True
    except Exception:
        return False

def query_params(start: datetime, end: datetime, contract_specs: list[dict] | None = None):
    params = [
        bigquery.ScalarQueryParameter('start_ts', 'TIMESTAMP', start),
        bigquery.ScalarQueryParameter('end_ts', 'TIMESTAMP', end),
    ]
    for spec in contract_specs or []:
        params.append(bigquery.ScalarQueryParameter(spec['address_param'], 'STRING', spec['address']))
        if 'active_from_param' in spec:
            params.append(bigquery.ScalarQueryParameter(
                spec['active_from_param'], 'TIMESTAMP', parse_utc(spec['active_from'])
            ))
        if 'active_to_param' in spec:
            params.append(bigquery.ScalarQueryParameter(
                spec['active_to_param'], 'TIMESTAMP', parse_utc(spec['active_to_exclusive'])
            ))
    return params


def dry_run(
    client: bigquery.Client, sql: str, start: datetime, end: datetime,
    contract_specs: list[dict] | None = None,
) -> int:
    job = client.query(sql, job_config=bigquery.QueryJobConfig(
        dry_run=True, use_query_cache=False,
        query_parameters=query_params(start, end, contract_specs),
    ))
    return int(job.total_bytes_processed or 0)


def window_slug(start: datetime) -> str:
    return start.strftime('%Y%m%dT%H%M%SZ')


def raw_log_windows(start: datetime, end: datetime, window_days: int):
    """Bound raw-log queries by both batch size and UTC calendar month."""
    if window_days <= 0:
        raise ValueError('window_days must be positive')
    for month_start, month_end in month_windows(start, end):
        cursor = month_start
        while cursor < month_end:
            win_end = min(cursor + timedelta(days=window_days), month_end)
            yield cursor, win_end
            cursor = win_end


def output_path(storage_root: Path, dataset: str, start: datetime, end: datetime) -> Path:
    slug = (
        f'{window_slug(start)}_{window_slug(end)}'
        if dataset == 'raw_protocol_logs' else iso_slug(start)
    )
    return storage_root / 'raw' / dataset / f'year={start.year}' / f'month={start.month:02d}' / f'{dataset}_{slug}.parquet'


def write_query_to_local_parquet(
    client, sql, start, end, max_bytes, stage: Path, compression: str, page_size: int,
    contract_specs: list[dict] | None = None,
) -> dict:
    partial = stage.with_suffix(stage.suffix + '.partial')
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.unlink(missing_ok=True)
    job = client.query(sql, job_config=bigquery.QueryJobConfig(
        use_query_cache=True,
        maximum_bytes_billed=max_bytes,
        query_parameters=query_params(start, end, contract_specs),
    ))
    result = job.result(page_size=page_size)
    writer = None
    rows_written = 0
    try:
        for page in result.pages:
            rows = [dict(row.items()) for row in page]
            if not rows:
                continue
            table = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(partial, table.schema, compression=compression)
            writer.write_table(table)
            rows_written += len(rows)
        if writer is not None:
            writer.close()
            writer = None
        else:
            pq.write_table(pa.table({'_empty': pa.array([], type=pa.string())}), partial, compression=compression)
        stage.unlink(missing_ok=True)
        partial.replace(stage)
    except Exception:
        if writer is not None:
            writer.close()
        raise
    return {
        'job_id': job.job_id,
        'total_bytes_processed': int(job.total_bytes_processed or 0),
        'total_bytes_billed': int(job.total_bytes_billed or 0),
        'cache_hit': bool(job.cache_hit),
        'rows': rows_written,
    }


def publish_existing_stage(stage: Path, out: Path, storage_root: Path, cfg: dict, dataset: str, win_start: datetime, win_end: datetime, meta: dict | None = None):
    pub = copy_to_drive_with_retries(stage, out, cfg)
    record = {
        'dataset': dataset,
        'window_start': win_start.isoformat(),
        'window_end_exclusive': win_end.isoformat(),
        'file': str(out),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        **(meta or {}),
        **pub,
    }
    append_jsonl(storage_root / cfg['output']['manifest'], record)
    return record


def run_one(
    client, dataset, sql, win_start, win_end, cfg, execute, force, storage_root,
    contract_specs: list[dict] | None = None,
):
    cap = int(cfg['bigquery']['maximum_bytes_billed'][dataset])
    out = output_path(storage_root, dataset, win_start, win_end) if storage_root else None
    slug = (
        f'{window_slug(win_start)}_{window_slug(win_end)}'
        if dataset == 'raw_protocol_logs' else iso_slug(win_start)
    )
    stage = staging_path(cfg, dataset, slug, f'{dataset}_{slug}.parquet')

    if out is not None and out.exists() and not force:
        if readable_parquet(out):
            print(f'[SKIP] {dataset} {iso_slug(win_start)} -> valid Parquet already on Drive')
            return
        print(f'[REPAIR] {dataset} {iso_slug(win_start)} -> existing Drive file is unreadable/empty; replacing only this month')
        try:
            out.unlink()
        except OSError as exc:
            raise RuntimeError(f'Cannot remove invalid Drive output {out}: {exc}')

    if execute and stage.exists() and not force:
        print(f'[RESUME-UPLOAD] {dataset} {iso_slug(win_start)} from local staged file')
        record = publish_existing_stage(stage, out, storage_root, cfg, dataset, win_start, win_end, {'resumed_from_stage': True})
        print(f"[OK] {dataset} {iso_slug(win_start)} -> {out} ({record['bytes']:,} bytes)")
        return

    estimate = dry_run(client, sql, win_start, win_end, contract_specs)
    print(f'[DRY-RUN] {dataset} {iso_slug(win_start)}: {estimate/(1024**3):.3f} GiB estimated')
    if estimate > cap:
        raise RuntimeError(
            f'{dataset} {iso_slug(win_start)} estimate exceeds configured cap: '
            f'{estimate/(1024**3):.3f} > {cap/(1024**3):.3f} GiB'
        )
    if not execute:
        return

    meta = write_query_to_local_parquet(
        client, sql, win_start, win_end, cap, stage,
        cfg['output'].get('compression', 'zstd'),
        int(cfg['bigquery'].get(f'{dataset}_page_size', cfg['bigquery'].get('page_size', 10000))),
        contract_specs,
    )
    print(f"[STAGED] {dataset} {iso_slug(win_start)}: {meta['rows']:,} rows -> {stage}")
    record = publish_existing_stage(stage, out, storage_root, cfg, dataset, win_start, win_end, meta)
    print(f"[OK] {dataset} {iso_slug(win_start)}: {meta['rows']:,} rows -> {out}")


def main():
    p = argparse.ArgumentParser(description='AssetGuardX raw BigQuery extractor: local staging -> verified Drive publish.')
    p.add_argument('--dataset', choices=['protocol_events', 'token_activity_events', 'hourly_blocks', 'raw_protocol_logs', 'all'], default='all')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('--start')
    p.add_argument('--end')
    p.add_argument('--window-days', type=int, default=7,
                   help='UTC chunk size for raw_protocol_logs; other datasets remain month-chunked.')
    args = p.parse_args()

    cfg = load_yaml('config/extraction.yaml')
    registry = load_yaml('config/contracts.yaml')
    project = os.getenv('ASSETGUARDX_GCP_PROJECT', cfg['project_id'])
    client = bigquery.Client(project=project, location=cfg.get('location', 'US'))
    start = parse_utc(args.start or cfg['study_window']['start'])
    end = parse_utc(args.end or cfg['study_window']['validation_end_exclusive'])
    if end <= start:
        raise ValueError('--end must be later than --start (the end is exclusive).')
    if args.window_days <= 0:
        raise ValueError('--window-days must be a positive integer.')
    storage_root = resolve_storage_root(cfg, require=args.execute) if args.execute else resolve_storage_root(cfg, require=False)

    # Keep the existing broad run unchanged. Raw logs are opt-in and use smaller
    # fixed UTC windows so each result is independently resumable.
    wanted = ['protocol_events', 'token_activity_events', 'hourly_blocks'] if args.dataset == 'all' else [args.dataset]
    sqls = {}
    raw_contract_specs = None
    bq = cfg['bigquery']
    if 'protocol_events' in wanted:
        table = bq['decoded_events_table']
        assert_columns(table, table_columns(client, table),
                       {'block_number','block_timestamp','transaction_hash','address','event_signature','args'})
        sqls['protocol_events'] = build_protocol_events_sql(table, registry['contracts'])
    if 'token_activity_events' in wanted:
        table = bq['token_transfers_table']
        assert_columns(table, table_columns(client, table),
                       {'block_number','block_timestamp','transaction_hash','address','from_address','to_address','quantity'})
        sqls['token_activity_events'] = build_token_activity_sql(table, registry)
    if 'hourly_blocks' in wanted:
        table = bq['blocks_table']
        assert_columns(table, table_columns(client, table), {'block_number','block_timestamp'})
        sqls['hourly_blocks'] = build_hourly_blocks_sql(table)
    if 'raw_protocol_logs' in wanted:
        table = bq['raw_logs_table']
        assert_columns(table, table_columns(client, table),
                       {'block_number','block_timestamp','transaction_hash','log_index','address','topics','data'})
        sqls['raw_protocol_logs'], raw_contract_specs = build_raw_protocol_logs_sql(
            table, registry['contracts'], bq['raw_log_contracts']
        )

    print(f'Project: {project}')
    print(f'Window : {start.isoformat()} -> {end.isoformat()} (exclusive)')
    print(f'Mode   : {"EXECUTE" if args.execute else "DRY-RUN ONLY"}')
    print('Design : raw-first; hourly token aggregates are derived locally later, not queried twice.')
    if storage_root:
        print(f'Drive  : {storage_root}')

    if args.dataset == 'raw_protocol_logs':
        # Each shard belongs to exactly one UTC calendar month. This keeps the
        # decoder and consolidation stages calendar-correct without a repair pass.
        for win_start, win_end in raw_log_windows(start, end, args.window_days):
            print(f'\n=== raw logs {win_start.isoformat()} -> {win_end.isoformat()} ===')
            run_one(client, 'raw_protocol_logs', sqls['raw_protocol_logs'], win_start, win_end,
                    cfg, args.execute, args.force, storage_root, raw_contract_specs)
    else:
        # Month-first execution makes every completed month independently usable/resumable.
        for win_start, win_end in month_windows(start, end):
            print(f'\n=== {iso_slug(win_start)} ===')
            for dataset in wanted:
                run_one(client, dataset, sqls[dataset], win_start, win_end,
                        cfg, args.execute, args.force, storage_root)

    if not args.execute:
        print('\nDry-run complete. No rows downloaded.')


if __name__ == '__main__':
    main()

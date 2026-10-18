from __future__ import annotations

import argparse
import hashlib
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from utils import append_jsonl, copy_to_drive_with_retries, load_yaml, month_windows, parse_utc, resolve_scratch_root, resolve_storage_root, sha256_file


CONSOLIDATION_VERSION = 'assetguardx-protocol-events/v2'
PARTITION_RULE = 'block_timestamp:calendar_month_utc'
RAW_SOURCE = 'raw_protocol_logs'

BASE_FIELDS = [
    pa.field('block_number', pa.int64()),
    pa.field('block_timestamp', pa.timestamp('us', tz='UTC')),
    pa.field('transaction_hash', pa.string()),
    pa.field('log_index', pa.int64()),
    pa.field('source_table', pa.string()),
    pa.field('source_event_fingerprint', pa.string()),
    pa.field('contract_address', pa.string()),
    pa.field('contract_key', pa.string()),
    pa.field('event_signature', pa.string()),
    pa.field('args_json', pa.string()),
    # Raw evidence is retained for every record whose source exposes it.
    pa.field('topic0', pa.string()),
    pa.field('topics', pa.list_(pa.string())),
    pa.field('data', pa.string()),
    # Decoder results are populated for the raw-log source; null for BigQuery decoded rows.
    pa.field('event_name', pa.string()),
    pa.field('event_family', pa.string()),
    pa.field('decode_status', pa.string()),
    pa.field('decode_error', pa.string()),
    pa.field('decoded_args_json', pa.string()),
    pa.field('decoder_version', pa.string()),
]


def month_of(path: Path) -> str:
    year = next((part[5:] for part in path.parts if part.startswith('year=')), None)
    month = next((part[6:] for part in path.parts if part.startswith('month=')), None)
    if not year or not month:
        raise ValueError(f'Input is not partitioned by year/month: {path}')
    return f'{year}-{month}'


def files_by_month(paths: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(paths):
        grouped[month_of(path)].append(path)
    return dict(grouped)


def schema_for(decoder_version: str, protocol_rows: int, raw_log_rows: int) -> pa.Schema:
    metadata = {
        b'assetguardx_consolidation_version': CONSOLIDATION_VERSION.encode(),
        b'assetguardx_partition_by': PARTITION_RULE.encode(),
        b'assetguardx_decoder_version': decoder_version.encode(),
        b'assetguardx_protocol_event_rows': str(protocol_rows).encode(),
        b'assetguardx_raw_log_rows': str(raw_log_rows).encode(),
    }
    return pa.schema(BASE_FIELDS, metadata=metadata)


def parquet_num_rows(path: Path) -> int:
    parquet = pq.ParquetFile(path)
    try:
        return int(parquet.metadata.num_rows)
    finally:
        parquet.close()


def parquet_columns(path: Path) -> list[str]:
    parquet = pq.ParquetFile(path)
    try:
        return parquet.schema_arrow.names
    finally:
        parquet.close()


def normalize_timestamp(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if isinstance(value, datetime) and value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc) if isinstance(value, datetime) else value


def require_event_month(row: dict, month: str, path: Path) -> None:
    timestamp = normalize_timestamp(row.get('block_timestamp'))
    if not isinstance(timestamp, datetime) or timestamp.strftime('%Y-%m') != month:
        raise RuntimeError(
            f'{path}: event timestamp {timestamp!r} is outside UTC month {month}; '
            'the raw-log extractor must split windows at calendar boundaries.'
        )


def normalize_existing_event(row: dict) -> dict:
    """Project decoded_events rows onto the shared, lossless event schema."""
    return {
        'block_number': row.get('block_number'),
        'block_timestamp': normalize_timestamp(row.get('block_timestamp')),
        'transaction_hash': row.get('transaction_hash'),
        'log_index': row.get('log_index'),
        'source_table': row.get('source_table') or 'decoded_events',
        'source_event_fingerprint': row.get('source_event_fingerprint'),
        'contract_address': row.get('contract_address'),
        'contract_key': row.get('contract_key'),
        'event_signature': row.get('event_signature'),
        'args_json': row.get('args_json'),
        'topic0': row.get('topic0'),
        'topics': row.get('topics'),
        'data': row.get('data'),
        'event_name': row.get('event_name'),
        'event_family': row.get('event_family'),
        'decode_status': row.get('decode_status'),
        'decode_error': row.get('decode_error'),
        'decoded_args_json': row.get('decoded_args_json'),
        'decoder_version': row.get('decoder_version'),
    }


def normalize_raw_log(row: dict, decoder_version: str) -> dict:
    normalized = normalize_existing_event(row)
    normalized['source_table'] = RAW_SOURCE
    normalized['decoder_version'] = decoder_version
    return normalized


def parquet_rows(path: Path, columns: list[str] | None = None, batch_size: int = 20_000):
    parquet = pq.ParquetFile(path)
    try:
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            yield from batch.to_pylist()
    finally:
        parquet.close()


def fingerprint_digest(paths: list[Path], source_only: bool = False) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        columns = ['source_event_fingerprint']
        if 'source_table' in parquet.schema_arrow.names:
            columns.insert(0, 'source_table')
        elif source_only:
            raise RuntimeError(f'Cannot identify raw-log rows in {path}: source_table is missing.')
        try:
            for batch in parquet.iter_batches(batch_size=50_000, columns=columns):
                for row in batch.to_pylist():
                    if source_only and row.get('source_table') != RAW_SOURCE:
                        continue
                    fingerprint = row.get('source_event_fingerprint')
                    if not fingerprint:
                        raise RuntimeError(f'Missing source_event_fingerprint in {path}')
                    digest.update(fingerprint.encode('utf-8'))
                    digest.update(b'\n')
                    count += 1
        finally:
            parquet.close()
    return count, digest.hexdigest()


def raw_contract_keys(paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        for row in parquet_rows(path, ['contract_key']):
            if row.get('contract_key'):
                keys.add(str(row['contract_key']))
    return keys


def consolidated_metadata(path: Path) -> dict[str, str]:
    parquet = pq.ParquetFile(path)
    try:
        metadata = parquet.schema_arrow.metadata or {}
        return {key.decode(): value.decode() for key, value in metadata.items()}
    finally:
        parquet.close()


def is_currently_consolidated(path: Path, decoder_version: str) -> bool:
    try:
        meta = consolidated_metadata(path)
        return (
            meta.get('assetguardx_consolidation_version') == CONSOLIDATION_VERSION
            and meta.get('assetguardx_decoder_version') == decoder_version
        )
    except Exception:
        return False


def write_month(stage: Path, schema: pa.Schema, protocol_files: list[Path], decoded_files: list[Path], decoder_version: str, compression: str, month: str | None = None) -> dict:
    stage.parent.mkdir(parents=True, exist_ok=True)
    partial = stage.with_suffix(stage.suffix + '.partial')
    partial.unlink(missing_ok=True)
    writer = pq.ParquetWriter(partial, schema, compression=compression)
    protocol_rows = raw_rows = 0
    pending = []
    output = None

    def flush():
        if pending:
            writer.write_table(pa.Table.from_pylist(pending, schema=schema))
            pending.clear()

    def append_for_month(row: dict, path: Path) -> None:
        if month is not None:
            require_event_month(row, month, path)
        pending.append(row)

    try:
        for path in protocol_files:
            for row in parquet_rows(path):
                if row.get('source_table') == RAW_SOURCE:
                    raise RuntimeError(f'{path} already contains consolidated raw-log rows but lacks a trusted migration marker.')
                append_for_month(normalize_existing_event(row), path)
                protocol_rows += 1
                if len(pending) >= 10_000:
                    flush()
        for path in decoded_files:
            for row in parquet_rows(path):
                if row.get('decode_status') not in {'decoded', 'unknown_topic', 'malformed'}:
                    raise RuntimeError(f'Unexpected decode_status={row.get("decode_status")!r} in {path}')
                if not row.get('topics') or not row.get('data') or not row.get('topic0'):
                    raise RuntimeError(f'Raw evidence columns missing in decoded log row from {path}')
                append_for_month(normalize_raw_log(row, decoder_version), path)
                raw_rows += 1
                if len(pending) >= 10_000:
                    flush()
        flush()
        writer.close()
        writer = None
        output = pq.ParquetFile(partial)
        expected = protocol_rows + raw_rows
        actual = output.metadata.num_rows
        output.close()
        output = None
        if actual != expected:
            raise RuntimeError(f'Output row-count mismatch: got {actual:,}; expected {expected:,}')
        partial.replace(stage)
        return {'protocol_rows': protocol_rows, 'raw_log_rows': raw_rows, 'output_rows': expected}
    except Exception:
        if output is not None:
            output.close()
        if writer is not None:
            writer.close()
        # Preserve a closed partial file. A later run verifies it and can reuse
        # it if complete, or safely remove/rebuild it if incomplete.
        raise


def target_path(root: Path, month: str) -> Path:
    return root / 'raw' / 'protocol_events' / f'year={month[:4]}' / f'month={month[5:7]}' / f'protocol_events_{month}.parquet'


def record_consolidated_file(root: Path, cfg: dict, target: Path, month: str, verified: dict,
                             protocol_rows: int, decoder_version: str, raw_digest: str) -> None:
    append_jsonl(root / cfg['output']['manifest'], {
        'dataset': 'protocol_events',
        'window_start': f'{month}-01T00:00:00+00:00',
        'file': str(target),
        'rows': verified['rows'],
        'bytes': target.stat().st_size,
        'sha256': sha256_file(target) if cfg.get('output', {}).get('checksum_sha256', True) else None,
        'migration': CONSOLIDATION_VERSION,
        'decoder_version': decoder_version,
        'source_protocol_event_rows': protocol_rows,
        'source_raw_log_rows': verified['raw_log_rows'],
        'raw_log_fingerprint_sha256': raw_digest,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
    })


def verify_integrated_file(path: Path, decoder_version: str, expected_raw_rows: int | None = None,
                           expected_raw_digest: str | None = None) -> dict:
    if not is_currently_consolidated(path, decoder_version):
        raise RuntimeError(f'Consolidation marker/decoder version mismatch: {path}')
    metadata = consolidated_metadata(path)
    if metadata.get('assetguardx_partition_by') != PARTITION_RULE:
        raise RuntimeError(f'Calendar partition rule missing or incorrect: {path}')
    parquet = pq.ParquetFile(path)
    try:
        total_rows = parquet.metadata.num_rows
    finally:
        parquet.close()
    raw_count, raw_digest = fingerprint_digest([path], source_only=True)
    expected = int(metadata['assetguardx_raw_log_rows'])
    if raw_count != expected:
        raise RuntimeError(f'{path} has {raw_count:,} raw-log rows; metadata says {expected:,}')
    if expected_raw_rows is not None and raw_count != expected_raw_rows:
        raise RuntimeError(f'{path} raw-log rows={raw_count:,}; source rows={expected_raw_rows:,}')
    if expected_raw_digest is not None and raw_digest != expected_raw_digest:
        raise RuntimeError(f'{path} raw-log fingerprint sequence does not match the decoded/raw source.')
    return {'rows': total_rows, 'raw_log_rows': raw_count, 'raw_log_fingerprint_sha256': raw_digest}


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Merge ABI-decoded selected contract logs into the monthly raw/protocol_events Parquet files.'
    )
    parser.add_argument('--execute', action='store_true', help='Publish merged monthly Parquet files; default is a read-only plan.')
    parser.add_argument('--remove-separate-source', action='store_true',
                        help='After every merged month passes row/fingerprint validation, remove raw/raw_protocol_logs files.')
    parser.add_argument('--check-complete', action='store_true',
                        help='Exit successfully only if every configured month already has a v2 unified file.')
    args = parser.parse_args()
    if args.remove_separate_source and not args.execute:
        parser.error('--remove-separate-source requires --execute')

    cfg = load_yaml('config/extraction.yaml')
    abi_config = load_yaml('config/raw_log_abis.yaml')
    decoder_version = str(abi_config['decoder_version'])
    root = resolve_storage_root(cfg, require=args.execute)
    if root is None:
        raise RuntimeError('Drive archive root is not configured.')
    if args.check_complete:
        start = parse_utc(cfg['study_window']['start'])
        end = parse_utc(cfg['study_window']['validation_end_exclusive'])
        expected_months = [month_start.strftime('%Y-%m') for month_start, _ in month_windows(start, end)]
        complete = all(
            target_path(root, month).exists()
            and consolidated_metadata(target_path(root, month)).get('assetguardx_consolidation_version') == CONSOLIDATION_VERSION
            and consolidated_metadata(target_path(root, month)).get('assetguardx_partition_by') == PARTITION_RULE
            for month in expected_months
        )
        print('Unified protocol archive complete.' if complete else 'Unified protocol archive needs raw-log extraction and consolidation.')
        raise SystemExit(0 if complete else 3)
    raw_groups = files_by_month(list((root / 'raw' / 'raw_protocol_logs').glob('year=*/month=*/*.parquet')))
    decoded_groups = files_by_month(list((root / 'processed' / 'decoded_raw_protocol_logs').glob('year=*/month=*/*.parquet')))
    protocol_groups = files_by_month(list((root / 'raw' / 'protocol_events').glob('year=*/month=*/*.parquet')))

    if not raw_groups:
        migrated = [p for paths in protocol_groups.values() for p in paths if is_currently_consolidated(p, decoder_version)]
        if migrated:
            print(f'Already consolidated with decoder {decoder_version}: {len(migrated)} month file(s).')
            return
        raise RuntimeError(f'No raw source Parquet found under {root / "raw" / "raw_protocol_logs"}.')

    missing_decoded = sorted(set(raw_groups) - set(decoded_groups))
    if missing_decoded:
        raise RuntimeError(
            'Missing ABI-decoded raw logs for months: ' + ', '.join(missing_decoded)
            + '. Rebuild after correcting ABI config: python src/decode_raw_logs.py --force'
        )

    for month in sorted(raw_groups):
        decoded_files = decoded_groups[month]
        for path in decoded_files:
            found = consolidated_metadata(path).get('assetguardx_decoder_version', '')
            if found != decoder_version:
                raise RuntimeError(
                    f'{path} was decoded with version {found!r}, expected {decoder_version!r}. '
                    'Run: python src/decode_raw_logs.py --force'
                )
        raw_count, raw_digest = fingerprint_digest(decoded_files)
        remaining_source_rows = sum(parquet_num_rows(path) for path in raw_groups[month])
        if month not in protocol_groups:
            raise RuntimeError(f'{month}: no existing Uniswap/decoded protocol_events monthly file to merge into.')

        target = target_path(root, month)
        if len(protocol_groups[month]) != 1 or protocol_groups[month][0] != target:
            raise RuntimeError(f'{month}: expected one canonical monthly protocol_events file at {target}')
        if is_currently_consolidated(target, decoder_version):
            # A previous run may have published every month successfully, then
            # stopped partway through deleting the old source shards. In that
            # state the canonical file is checked against the complete decoded
            # month, while the leftover raw shard count may intentionally be lower.
            if remaining_source_rows > raw_count:
                raise RuntimeError(
                    f'{month}: remaining raw source rows exceed the verified decoded month '
                    f'({remaining_source_rows:,} > {raw_count:,}).'
                )
            verified = verify_integrated_file(target, decoder_version, raw_count, raw_digest)
            if args.execute:
                integrated = consolidated_metadata(target)
                record_consolidated_file(
                    root, cfg, target, month, verified,
                    int(integrated['assetguardx_protocol_event_rows']), decoder_version, raw_digest,
                )
            print(f'[ALREADY OK] {month}: {verified["rows"]:,} rows, {verified["raw_log_rows"]:,} raw logs')
            continue

        source_fingerprint_count, source_fingerprint_digest = fingerprint_digest(raw_groups[month])
        if raw_count != remaining_source_rows:
            raise RuntimeError(f'{month}: decoded rows={raw_count:,}, raw source rows={remaining_source_rows:,}')
        if source_fingerprint_count != raw_count or source_fingerprint_digest != raw_digest:
            raise RuntimeError(f'{month}: ABI-decoded fingerprints do not exactly match the archived raw logs.')

        if any('source_table' not in parquet_columns(path) for path in protocol_groups[month]):
            raise RuntimeError(f'{month}: existing protocol_events schema is missing source_table.')
        protocol_keys = raw_contract_keys(protocol_groups[month])
        raw_keys = raw_contract_keys(decoded_files)
        overlap = sorted(protocol_keys & raw_keys)
        if overlap:
            raise RuntimeError(f'{month}: contract keys occur in both sources; refusing unsafe merge: {overlap}')

        original_rows = sum(parquet_num_rows(path) for path in protocol_groups[month])
        output_rows = original_rows + raw_count
        print(f'{month}: existing={output_rows - raw_count:,}; raw_logs={raw_count:,}; merged={output_rows:,}; '
              f'fingerprints={raw_digest}')
        if not args.execute:
            continue

        stage = resolve_scratch_root(cfg) / 'protocol_events_consolidation' / month / f'protocol_events_{month}.parquet'
        partial = stage.with_suffix(stage.suffix + '.partial')
        metadata = None
        for candidate in (stage, partial):
            if not candidate.exists():
                continue
            try:
                recovered = verify_integrated_file(candidate, decoder_version, raw_count, raw_digest)
                candidate_metadata = consolidated_metadata(candidate)
                if (recovered['rows'] != output_rows
                        or int(candidate_metadata['assetguardx_protocol_event_rows']) != original_rows):
                    raise RuntimeError('Staged Parquet row counts do not match current inputs.')
                if candidate == partial:
                    partial.replace(stage)
                metadata = {
                    'protocol_rows': original_rows,
                    'raw_log_rows': raw_count,
                    'output_rows': output_rows,
                }
                print(f'[RECOVERED STAGE] {month}: verified and reusing {stage}')
                break
            except Exception:
                # An incomplete or stale stage is disposable; the archived inputs
                # remain untouched and the month will be rebuilt below.
                candidate.unlink(missing_ok=True)

        if metadata is None:
            metadata = write_month(
                stage, schema_for(decoder_version, original_rows, raw_count), protocol_groups[month], decoded_files,
                decoder_version, cfg['output'].get('compression', 'zstd'), month,
            )
        if metadata['raw_log_rows'] != raw_count:
            raise RuntimeError(f'{month}: staged merge raw row-count mismatch.')
        staged_digest_count, staged_digest = fingerprint_digest([stage], source_only=True)
        if staged_digest_count != raw_count or staged_digest != raw_digest:
            raise RuntimeError(f'{month}: staged raw log fingerprints do not match source.')

        # Keep the original Uniswap monthly file recoverable before replacing it.
        backup = root / 'metadata' / 'migration_backups' / 'protocol_events' / f'year={month[:4]}' / f'month={month[5:7]}' / f'protocol_events_{month}.original.parquet'
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(target, backup)
        if parquet_num_rows(backup) != original_rows:
            raise RuntimeError(f'{month}: protocol_events backup row count is not verified; refusing replacement.')
        try:
            published = copy_to_drive_with_retries(stage, target, cfg)
        except Exception:
            # copy_to_drive_with_retries replaces existing Drive files; restore the backup
            # if publication failed after removing the old target.
            if backup.exists():
                shutil.copy2(backup, target)
            raise
        verified = verify_integrated_file(target, decoder_version, raw_count, raw_digest)
        if verified['rows'] != metadata['output_rows']:
            shutil.copy2(backup, target)
            raise RuntimeError(f'{month}: published Parquet row count differs from staged merge; original restored.')

        record_consolidated_file(
            root, cfg, target, month, verified, metadata['protocol_rows'], decoder_version, raw_digest,
        )
        print(f'[MERGED] {month}: {verified["rows"]:,} total rows ({verified["raw_log_rows"]:,} raw logs), '
              f'{published["bytes"]:,} bytes')

    if args.execute and args.remove_separate_source:
        # Re-check every target before removing any of the original raw source files.
        for month in sorted(raw_groups):
            decoded_files = decoded_groups[month]
            raw_count, raw_digest = fingerprint_digest(decoded_files)
            verify_integrated_file(target_path(root, month), decoder_version, raw_count, raw_digest)
        removed = 0
        failed_to_remove = []
        for paths in raw_groups.values():
            for path in paths:
                for attempt in range(1, 6):
                    try:
                        path.unlink()
                        removed += 1
                        break
                    except PermissionError as exc:
                        if attempt == 5:
                            failed_to_remove.append((path, exc))
                        else:
                            print(f'[RETRY LOCK {attempt}/5] {path}; waiting for DriveFS/another reader to release it')
                            time.sleep(2)
        if failed_to_remove:
            print(f'Removed {removed} source file(s); {len(failed_to_remove)} remain locked.')
            for path, exc in failed_to_remove:
                print(f'[LOCKED] {path}: {exc}')
            raise SystemExit(4)
        print(f'Removed {removed} remaining source file(s) from raw/raw_protocol_logs after verified consolidation.')
    elif not args.execute:
        print('\nDry run only. No Drive files were changed. After verifying the plan, use --execute; '
              'add --remove-separate-source to remove the old raw log shards only after parity checks pass.')


if __name__ == '__main__':
    main()

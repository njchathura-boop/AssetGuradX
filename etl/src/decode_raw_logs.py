from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from eth_abi import decode as abi_decode
from eth_utils import keccak

from utils import append_jsonl, copy_to_drive_with_retries, load_yaml, resolve_storage_root, resolve_scratch_root


EXTRA_FIELDS = {
    'source_table': pa.string(),
    'event_signature': pa.string(),
    'event_name': pa.string(),
    'event_family': pa.string(),
    'decode_status': pa.string(),
    'decode_error': pa.string(),
    'args_json': pa.string(),
    'decoded_args_json': pa.string(),
    'decoder_version': pa.string(),
}


def event_types(signature: str) -> list[str]:
    match = re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*\((.*)\)', signature)
    if not match:
        raise ValueError(f'Invalid canonical event signature: {signature!r}')
    body = match.group(1)
    return [] if not body else body.split(',')


def topic_for(signature: str) -> str:
    return '0x' + keccak(text=signature).hex()


def _safe_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        # Decimal strings avoid precision loss in JSON consumers such as JavaScript.
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_safe_value(x) for x in value]
    if isinstance(value, bytes):
        return '0x' + value.hex()
    if hasattr(value, 'hex') and value.__class__.__name__ == 'HexBytes':
        return '0x' + value.hex()
    return value


def _decode_word(abi_type: str, value: str):
    raw = bytes.fromhex(value[2:] if value.startswith('0x') else value)
    if len(raw) != 32:
        raise ValueError(f'Indexed argument is not one ABI word ({len(raw)} bytes).')
    return _safe_value(abi_decode([abi_type], raw)[0])


def make_contract_specs(registry: dict, abi_config: dict) -> dict[str, dict[str, dict]]:
    global_specs = abi_config.get('events', {})
    overrides = abi_config.get('contract_overrides', {})
    supplemental = abi_config.get('supplemental_contract_events', {})
    unregistered = set(supplemental) - set(registry['contracts'])
    if unregistered:
        raise ValueError(f'Supplemental ABI contracts absent from registry: {sorted(unregistered)}')
    out = {}
    for contract_key, contract in registry['contracts'].items():
        signatures = set(contract.get('decoded_event_signatures') or [])
        # Observed administrative/metadata events are decodable without making
        # their occurrence a mandatory coverage requirement.
        signatures.update(supplemental.get(contract_key, []))
        signatures.update(['Transfer(address,address,uint256)', 'Approval(address,address,uint256)'])
        if contract_key == 'lido_withdrawal_queue':
            signatures.add('ApprovalForAll(address,address,bool)')
        selected = {}
        for signature in sorted(signatures):
            spec = overrides.get(contract_key, {}).get(signature, global_specs.get(signature))
            if spec is None:
                continue
            types = event_types(signature)
            names = spec.get('names', [])
            indexed = spec.get('indexed', [])
            if len(names) != len(types):
                raise ValueError(f'{contract_key}: {signature}: {len(names)} names for {len(types)} event fields')
            if any(i < 0 or i >= len(types) for i in indexed) or len(set(indexed)) != len(indexed):
                raise ValueError(f'{contract_key}: invalid indexed positions for {signature}')
            selected[topic_for(signature)] = {
                'signature': signature,
                'name': signature.split('(', 1)[0],
                'types': types,
                'names': names,
                'indexed': set(indexed),
                'family': spec.get('family', 'other'),
            }
        out[contract_key] = selected
    return out


def decode_log_record(row: dict, contract_specs: dict, decoder_version: str) -> dict:
    result = {
        **row,
        'source_table': 'raw_protocol_logs',
        'event_signature': None,
        'event_name': None,
        'event_family': 'unknown',
        'decode_status': 'unknown_topic',
        'decode_error': None,
        'args_json': None,
        'decoded_args_json': None,
        'decoder_version': decoder_version,
    }
    key = str(row.get('contract_key') or '')
    topics = row.get('topics') or []
    if isinstance(topics, str):
        try:
            topics = json.loads(topics)
        except Exception:
            topics = []
    topic0 = str(row.get('topic0') or (topics[0] if topics else '')).lower()
    spec = contract_specs.get(key, {}).get(topic0)
    if spec is None:
        return result

    result.update({
        'event_signature': spec['signature'],
        'event_name': spec['name'],
        'event_family': spec['family'],
    })
    try:
        if not topics or str(topics[0]).lower() != topic0:
            raise ValueError('topic0 does not match topics[0].')
        expected_topics = 1 + len(spec['indexed'])
        if len(topics) != expected_topics:
            raise ValueError(f'Expected {expected_topics} topics, got {len(topics)}.')

        data = str(row.get('data') or '0x')
        data_hex = data[2:] if data.startswith('0x') else data
        if len(data_hex) % 2:
            raise ValueError('Event data has an odd number of hex characters.')
        if data_hex and not re.fullmatch(r'[0-9a-fA-F]+', data_hex):
            raise ValueError('Event data contains non-hex characters.')
        data_bytes = bytes.fromhex(data_hex)

        args = [None] * len(spec['types'])
        topic_index = 1
        nonindexed_positions = []
        nonindexed_types = []
        for position, (abi_type, name) in enumerate(zip(spec['types'], spec['names'])):
            if position in spec['indexed']:
                if abi_type.endswith(']') or abi_type in ('string', 'bytes'):
                    raise ValueError(f'Cannot decode hashed indexed dynamic field {name!r} ({abi_type}).')
                args[position] = _decode_word(abi_type, str(topics[topic_index]))
                topic_index += 1
            else:
                nonindexed_positions.append(position)
                nonindexed_types.append(abi_type)
        if nonindexed_types:
            decoded = abi_decode(nonindexed_types, data_bytes)
            for position, value in zip(nonindexed_positions, decoded):
                args[position] = _safe_value(value)
        elif data_bytes:
            raise ValueError(f'Expected empty data but got {len(data_bytes)} bytes.')

        named = dict(zip(spec['names'], args))
        result.update({
            'decode_status': 'decoded',
            'args_json': json.dumps(args, separators=(',', ':')),
            'decoded_args_json': json.dumps(named, separators=(',', ':'), sort_keys=True),
        })
    except Exception as exc:
        result['decode_status'] = 'malformed'
        result['decode_error'] = f'{type(exc).__name__}: {exc}'
    return result


def _month_key(path: Path) -> str:
    year = next((part[5:] for part in path.parts if part.startswith('year=')), None)
    month = next((part[6:] for part in path.parts if part.startswith('month=')), None)
    if not year or not month:
        raise ValueError(f'Raw log path is not partitioned by year/month: {path}')
    return f'{year}-{month}'


def _output_path(root: Path, month: str) -> Path:
    year, number = month.split('-')
    return root / 'processed' / 'decoded_raw_protocol_logs' / f'year={year}' / f'month={number}' / f'decoded_raw_protocol_logs_{month}.parquet'


def _schema_with_extras(input_schema: pa.Schema, decoder_version: str) -> pa.Schema:
    fields = list(input_schema)
    present = {field.name for field in fields}
    for name, dtype in EXTRA_FIELDS.items():
        if name not in present:
            fields.append(pa.field(name, dtype))
    metadata = dict(input_schema.metadata or {})
    metadata[b'assetguardx_decoder_version'] = decoder_version.encode()
    return pa.schema(fields, metadata=metadata)


def _valid_existing_output(path: Path, row_count: int, decoder_version: str) -> bool:
    try:
        parquet = pq.ParquetFile(path)
        metadata = parquet.schema_arrow.metadata or {}
        version = metadata.get(b'assetguardx_decoder_version', b'').decode()
        return parquet.metadata.num_rows == row_count and version == decoder_version
    except Exception:
        return False


def audit_raw_logs(files_by_month: dict[str, list[Path]], contract_specs: dict, decoder_version: str) -> dict:
    """Decode all rows in memory-bounded batches and report coverage without writing files."""
    statuses = Counter()
    events = Counter()
    unknown_topics = Counter()
    malformed_topics = Counter()
    row_count = 0
    for month, files in sorted(files_by_month.items()):
        for path in files:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=5000):
                for source in batch.to_pylist():
                    result = decode_log_record(source, contract_specs, decoder_version)
                    row_count += 1
                    status = result['decode_status']
                    statuses[status] += 1
                    key = str(source.get('contract_key') or 'unknown')
                    topic = str(source.get('topic0') or 'null')
                    if status == 'decoded':
                        events[(key, result['event_signature'])] += 1
                    elif status == 'unknown_topic':
                        unknown_topics[(key, topic)] += 1
                    else:
                        malformed_topics[(key, topic, result['decode_error'])] += 1
    return {
        'decoder_version': decoder_version,
        'input_rows': row_count,
        'status_counts': dict(statuses),
        'decoded_events': [
            {'contract_key': key, 'event_signature': signature, 'rows': count}
            for (key, signature), count in sorted(events.items())
        ],
        'unknown_topics': [
            {'contract_key': key, 'topic0': topic, 'rows': count}
            for (key, topic), count in sorted(unknown_topics.items())
        ],
        'malformed_topics': [
            {'contract_key': key, 'topic0': topic, 'error': error, 'rows': count}
            for (key, topic, error), count in sorted(malformed_topics.items())
        ],
    }


def decode_month(root: Path, cfg: dict, files: list[Path], month: str, contract_specs: dict,
                 decoder_version: str, force: bool = False) -> dict:
    out = _output_path(root, month)
    input_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    if out.exists() and not force:
        if _valid_existing_output(out, input_rows, decoder_version):
            return {'month': month, 'status': 'skipped_verified', 'rows': input_rows, 'file': str(out)}
        raise RuntimeError(f'Existing decoded output is stale/unverified: {out}. Rebuild this derived month with --force.')

    if not files:
        return {'month': month, 'status': 'empty', 'rows': 0}
    first_schema = pq.ParquetFile(files[0]).schema_arrow
    output_schema = _schema_with_extras(first_schema, decoder_version)
    scratch = resolve_scratch_root(cfg) / 'decoded_raw_protocol_logs' / month
    scratch.mkdir(parents=True, exist_ok=True)
    staged = scratch / f'decoded_raw_protocol_logs_{month}.parquet'
    partial = staged.with_suffix('.parquet.partial')
    partial.unlink(missing_ok=True)
    counts = Counter()
    writer = pq.ParquetWriter(partial, output_schema, compression=cfg['output'].get('compression', 'zstd'))
    try:
        for path in files:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=5000):
                decoded_rows = [decode_log_record(row, contract_specs, decoder_version) for row in batch.to_pylist()]
                counts.update(row['decode_status'] for row in decoded_rows)
                table = pa.Table.from_pylist(decoded_rows, schema=output_schema)
                writer.write_table(table)
        writer.close()
        writer = None
        if pq.ParquetFile(partial).metadata.num_rows != input_rows:
            raise RuntimeError(f'Decoded row count mismatch for {month}; refusing to publish.')
        staged.unlink(missing_ok=True)
        partial.replace(staged)
        drive_info = copy_to_drive_with_retries(staged, out, cfg)
    except Exception:
        if writer is not None:
            writer.close()
        partial.unlink(missing_ok=True)
        raise

    record = {
        'dataset': 'decoded_raw_protocol_logs',
        'month': month,
        'decoder_version': decoder_version,
        'abi_config_sha256': hashlib.sha256((Path(__file__).resolve().parents[1] / 'config' / 'raw_log_abis.yaml').read_bytes()).hexdigest(),
        'contract_registry_sha256': hashlib.sha256((Path(__file__).resolve().parents[1] / 'config' / 'contracts.yaml').read_bytes()).hexdigest(),
        'input_files': [str(path) for path in files],
        'rows': input_rows,
        'decode_status_counts': dict(counts),
        'file': str(out),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        **drive_info,
    }
    append_jsonl(root / 'manifests' / 'raw_log_decoder_manifest.jsonl', record)
    return {'month': month, 'status': 'decoded', 'rows': input_rows, 'decode_status_counts': dict(counts), 'file': str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description='Decode archived Ethereum raw logs locally; never queries BigQuery or RPC.')
    parser.add_argument('--start-month', help='Inclusive YYYY-MM month bound.')
    parser.add_argument('--end-month', help='Inclusive YYYY-MM month bound.')
    parser.add_argument('--force', action='store_true', help='Rebuild existing derived decoder outputs for selected months.')
    parser.add_argument('--audit-only', action='store_true', help='Decode in bounded memory without writing any files.')
    args = parser.parse_args()

    cfg = load_yaml('config/extraction.yaml')
    registry = load_yaml('config/contracts.yaml')
    abi_config = load_yaml('config/raw_log_abis.yaml')
    root = resolve_storage_root(cfg, require=True)
    contracts = cfg['bigquery']['raw_log_contracts']
    contract_specs = make_contract_specs({'contracts': registry['contracts']}, abi_config)
    # Only include selected raw contracts, not the other contracts in the broad registry.
    contract_specs = {key: contract_specs.get(key, {}) for key in contracts}

    files_by_month: dict[str, list[Path]] = defaultdict(list)
    for path in sorted((root / 'raw' / 'raw_protocol_logs').glob('year=*/month=*/*.parquet')):
        month = _month_key(path)
        if args.start_month and month < args.start_month:
            continue
        if args.end_month and month > args.end_month:
            continue
        files_by_month[month].append(path)
    if args.start_month and not re.fullmatch(r'\d{4}-\d{2}', args.start_month):
        raise ValueError('--start-month must have form YYYY-MM')
    if args.end_month and not re.fullmatch(r'\d{4}-\d{2}', args.end_month):
        raise ValueError('--end-month must have form YYYY-MM')
    if args.start_month and args.end_month and args.end_month < args.start_month:
        raise ValueError('--end-month must not be earlier than --start-month')
    if not files_by_month:
        raise RuntimeError(f'No raw_protocol_logs Parquet files found under {root / "raw" / "raw_protocol_logs"}')

    print('Input: frozen raw Parquet only; no BigQuery or RPC access.')
    print(f'Drive archive: {root}')
    print(f'Decoder: {abi_config["schema_version"]} / {abi_config["decoder_version"]}')
    print(f'Months selected: {len(files_by_month)}')
    if args.audit_only:
        print(json.dumps(audit_raw_logs(files_by_month, contract_specs, abi_config['decoder_version']), indent=2, sort_keys=True))
        return
    for month, files in sorted(files_by_month.items()):
        result = decode_month(root, cfg, files, month, contract_specs, abi_config['decoder_version'], args.force)
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import pyarrow.parquet as pq

from utils import copy_to_drive_with_retries, load_yaml, resolve_storage_root, staging_path

CONSOLIDATION_VERSION = 'assetguardx-protocol-events/v2'
PARTITION_RULE = 'block_timestamp:calendar_month_utc'


def main():
    cfg = load_yaml('config/extraction.yaml')
    root = resolve_storage_root(cfg, require=True)
    validation = subprocess.run(
        [sys.executable, str(Path(__file__).with_name('validate_raw_archive.py')), '--deep'],
        cwd=Path(__file__).resolve().parent.parent,
        check=False,
    )
    if validation.returncode:
        raise SystemExit('Cannot freeze: deep raw archive validation failed. Resolve the errors above first.')
    datasets = ['protocol_events','token_activity_events','hourly_blocks','rpc_hourly']
    separate_raw_logs = sorted((root / 'raw' / 'raw_protocol_logs').glob('year=*/month=*/*.parquet'))
    if separate_raw_logs:
        raise SystemExit(
            f'Cannot freeze: {len(separate_raw_logs)} separate raw_protocol_logs file(s) remain. '
            'Complete run_raw.ps1 before freezing.'
        )
    checksum_by_file = {}
    manifest_path = root / cfg['output']['manifest']
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding='utf-8').splitlines():
            try:
                rec = json.loads(line)
                if rec.get('file') and rec.get('sha256'):
                    checksum_by_file[str(Path(rec['file']))] = rec['sha256']
            except Exception:
                pass
    entries = []
    for ds in datasets:
        files = sorted((root/'raw'/ds).glob('year=*/month=*/*.parquet'))
        for f in files:
            parquet = None
            try:
                parquet = pq.ParquetFile(f)
                rows = parquet.metadata.num_rows
                metadata = parquet.schema_arrow.metadata or {}
            except Exception as exc:
                raise SystemExit(f'Cannot read Parquet metadata for {f}: {exc}')
            finally:
                if parquet is not None:
                    parquet.close()
            entry = {
                'dataset':ds,'path':str(f.relative_to(root)),'bytes':f.stat().st_size,'rows':int(rows),
                'sha256': checksum_by_file.get(str(f))
            }
            if ds == 'protocol_events':
                consolidation = metadata.get(b'assetguardx_consolidation_version', b'').decode()
                if consolidation != CONSOLIDATION_VERSION:
                    raise SystemExit(
                        f'Cannot freeze {f}: protocol_events has no verified consolidation marker '
                        f'({consolidation or "missing"}). Rebuild the unified monthly protocol_events archive first.'
                    )
                partition_by = metadata.get(b'assetguardx_partition_by', b'').decode()
                if partition_by != PARTITION_RULE:
                    raise SystemExit(f'Cannot freeze {f}: calendar-month partition marker is missing or incorrect.')
                entry.update({
                    'consolidation_version': consolidation,
                    'partition_by': partition_by,
                    'decoder_version': metadata.get(b'assetguardx_decoder_version', b'').decode(),
                    'source_protocol_event_rows': int(metadata.get(b'assetguardx_protocol_event_rows', b'0')),
                    'source_raw_log_rows': int(metadata.get(b'assetguardx_raw_log_rows', b'0')),
                })
            entries.append(entry)

    manifest = {
        'assetguardx_raw_version':'v2',
        'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':{
            'chain':'Ethereum Mainnet',
            'start':cfg['study_window']['start'],
            'core_end_exclusive':cfg['study_window']['core_end_exclusive'],
            'validation_end_exclusive':cfg['study_window']['validation_end_exclusive'],
            'note':'After this marker, canonical processing/labels/graph exports must read the stored archive rather than re-query BigQuery unless scope changes.'
        },
        'files':entries,
        'totals':{
            'files':len(entries),
            'rows':sum(x['rows'] for x in entries),
            'bytes':sum(x['bytes'] for x in entries),
        }
    }
    stage = staging_path(cfg,'metadata','freeze','RAW_FROZEN.json')
    stage.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    out = root/'metadata'/'RAW_FROZEN.json'
    pub = copy_to_drive_with_retries(stage,out,cfg)
    print(f'RAW archive frozen: {out}')
    print(f"Files={manifest['totals']['files']:,} Rows={manifest['totals']['rows']:,} Bytes={manifest['totals']['bytes']:,}")
    print(f"Manifest copy SHA-256: {pub.get('sha256')}")


if __name__ == '__main__':
    main()

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from utils import ROOT, copy_to_drive_with_retries, load_yaml, resolve_storage_root, staging_path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    cfg = load_yaml('config/extraction.yaml')
    root = resolve_storage_root(cfg, require=True)
    names = [
        'config/extraction.yaml',
        'config/contracts.yaml',
        'config/market_sources.yaml',
        'config/labels.yaml',
        'requirements.txt',
    ]
    files = []
    for rel in names:
        p = ROOT / rel
        raw = p.read_bytes()
        files.append({
            'path': rel,
            'sha256': sha256_bytes(raw),
            'bytes': len(raw),
            'text': raw.decode('utf-8'),
        })

    snapshot = {
        'snapshot_version': 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Exact extraction definition used for AssetGuardX-Raw v2. Keep with the raw archive for reproducibility.',
        'files': files,
    }
    stage = staging_path(cfg, 'metadata', 'definition', 'extraction_definition_snapshot.json')
    stage.write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
    out = root / 'metadata' / 'extraction_definition_snapshot.json'
    pub = copy_to_drive_with_retries(stage, out, cfg)
    print(f'[OK] extraction definition snapshot -> {out} ({pub["bytes"]:,} bytes)')


if __name__ == '__main__':
    main()

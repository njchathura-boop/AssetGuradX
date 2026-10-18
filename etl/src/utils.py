from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with p.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def append_jsonl(path: Path, obj: dict) -> None:
    ensure_parent(path)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, sort_keys=True, default=str) + '\n')


def parse_utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace('Z', '+00:00')).astimezone(timezone.utc)


def month_windows(start: datetime, end_exclusive: datetime) -> Iterator[Tuple[datetime, datetime]]:
    cur = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    if start > cur:
        cur = start
    while cur < end_exclusive:
        if cur.month == 12:
            nxt = datetime(cur.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            nxt = datetime(cur.year, cur.month + 1, 1, tzinfo=timezone.utc)
        yield cur, min(nxt, end_exclusive)
        cur = nxt


def iso_slug(dt: datetime) -> str:
    return dt.strftime('%Y-%m')


def quoted_sql_strings(values: list[str]) -> str:
    def esc(v: str) -> str:
        return "'" + v.replace("'", "''") + "'"
    return ',\n        '.join(esc(v) for v in values)


def _windows_drive_candidates() -> list[Path]:
    candidates: list[Path] = []
    for letter in 'DEFGHIJKLMNOPQRSTUVWXYZ':
        root = Path(f'{letter}:\\')
        candidates.extend([root / 'My Drive', root / 'Google Drive' / 'My Drive'])
    user = Path(os.environ.get('USERPROFILE', str(Path.home())))
    candidates.extend([user / 'Google Drive' / 'My Drive', user / 'Google Drive', user / 'My Drive'])
    return candidates


def discover_google_drive_root(cfg: dict) -> Path | None:
    storage = cfg.get('storage', {})
    env_name = storage.get('drive_root_env', 'ASSETGUARDX_DRIVE_ROOT')
    explicit = os.getenv(env_name)
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return p.resolve()
        raise RuntimeError(f'{env_name} is set to a path that does not exist: {p}')

    if os.name == 'nt' and storage.get('auto_detect_windows', True):
        for p in _windows_drive_candidates():
            try:
                if p.exists() and p.is_dir():
                    return p.resolve()
            except OSError:
                continue
    return None


def resolve_storage_root(cfg: dict, require: bool = False) -> Path | None:
    drive = discover_google_drive_root(cfg)
    storage = cfg.get('storage', {})
    if drive is None:
        if require:
            env_name = storage.get('drive_root_env', 'ASSETGUARDX_DRIVE_ROOT')
            raise RuntimeError(
                'Google Drive mounted folder was not found. Use Google Drive for desktop in Stream files mode, then set:\n'
                f'  $env:{env_name} = "G:\\My Drive"\n'
                'or pass -DriveRoot to the PowerShell runner.'
            )
        return None
    return drive / Path(storage.get('project_subdir', 'AssetGuardX_IITM'))


def resolve_scratch_root(cfg: dict) -> Path:
    storage = cfg.get('storage', {})
    env_name = storage.get('scratch_root_env', 'ASSETGUARDX_SCRATCH_ROOT')
    explicit = os.getenv(env_name)
    if explicit:
        root = Path(os.path.expandvars(explicit)).expanduser()
    elif os.name == 'nt':
        raw = storage.get('scratch_default_windows', r'%LOCALAPPDATA%\AssetGuardX\scratch')
        root = Path(os.path.expandvars(raw)).expanduser()
    else:
        root = Path.home() / '.cache' / 'assetguardx' / 'scratch'
    root.mkdir(parents=True, exist_ok=True)
    return root


def staging_path(cfg: dict, dataset: str, month_slug: str, filename: str) -> Path:
    root = resolve_scratch_root(cfg)
    p = root / dataset / month_slug / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def free_space_gib(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def wait_for_file_size(path: Path, expected: int, attempts: int = 10, delay: float = 1.0) -> bool:
    """Allow DriveFS to publish a just-closed file's final size."""
    for attempt in range(attempts):
        try:
            if path.stat().st_size == expected:
                return True
        except FileNotFoundError:
            pass
        if attempt + 1 < attempts:
            time.sleep(delay)
    return False


def copy_to_drive_with_retries(local_path: Path, drive_path: Path, cfg: dict) -> dict:
    """Publish a closed local file to DriveFS. The local file is retained on any failure."""
    storage = cfg.get('storage', {})
    retries = int(storage.get('copy_retries', 8))
    delay = float(storage.get('copy_retry_seconds', 3))
    ensure_parent(drive_path)
    uploading = drive_path.with_suffix(drive_path.suffix + '.uploading')
    local_sha = sha256_file(local_path) if cfg.get('output', {}).get('checksum_sha256', True) else None
    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            if uploading.exists():
                uploading.unlink()
            # DriveFS can fail with WinError 1450 on long direct Parquet writes.
            # Copy only a fully closed local file, in bounded chunks.
            with local_path.open('rb') as src, uploading.open('wb') as dst:
                while True:
                    chunk = src.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
                dst.flush()
            if not wait_for_file_size(uploading, local_path.stat().st_size):
                raise IOError('Drive copy size mismatch')
            if drive_path.exists():
                drive_path.unlink()
            uploading.replace(drive_path)
            if not wait_for_file_size(drive_path, local_path.stat().st_size):
                raise IOError('Published Drive file size mismatch')
            if cfg.get('output', {}).get('deep_remote_checksum_on_copy', False) and local_sha:
                remote_sha = sha256_file(drive_path)
                if remote_sha != local_sha:
                    raise IOError('Published Drive SHA-256 mismatch')
            if storage.get('delete_scratch_after_verified_copy', True):
                local_path.unlink(missing_ok=True)
            return {'sha256': local_sha, 'bytes': drive_path.stat().st_size, 'copy_attempts': attempt}
        except Exception as exc:
            last_exc = exc
            try:
                if uploading.exists():
                    uploading.unlink()
            except OSError:
                pass
            if attempt < retries:
                time.sleep(delay * attempt)
    raise RuntimeError(
        f'Could not publish {local_path} to Google Drive after {retries} attempts. '
        f'The complete local staged file was kept for retry. Last error: {last_exc}'
    )


def active_at(meta: dict, hour: datetime) -> bool:
    active_from = meta.get('active_from')
    if active_from and hour < parse_utc(active_from):
        return False
    active_to = meta.get('active_to_exclusive')
    if active_to and hour >= parse_utc(active_to):
        return False
    return True


def publish_dataframe(df, cfg: dict, dataset: str, month_slug: str, drive_path: Path, compression: str = 'zstd') -> dict:
    stage = staging_path(cfg, dataset, month_slug, drive_path.name)
    partial = stage.with_suffix(stage.suffix + '.partial')
    partial.unlink(missing_ok=True)
    df.to_parquet(partial, index=False, compression=compression)
    stage.unlink(missing_ok=True)
    partial.replace(stage)
    return copy_to_drive_with_retries(stage, drive_path, cfg)

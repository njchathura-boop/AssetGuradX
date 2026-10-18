"""Immutable artifact commits shared by graph exports and their consumers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .temporal import canonical


def immutable_json(path: Path, value) -> None:
    """Commit immutable JSON with a same-directory atomic rename and writer lock."""
    payload = json.loads(canonical(value))
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    with lock.open("x", encoding="utf-8") as handle:
        handle.write("single-writer commit\n")
    temporary = None
    try:
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != payload:
                raise ValueError("refusing to overwrite immutable artifact with different content")
            return
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)

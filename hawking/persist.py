"""Leaf persistence helpers.

This module has no hawking imports so it can sit under dag_store, resources,
ledger, runtime, and grok_bridge without recreating the old SCC
(dag_store -> workunit -> resources -> max_policy -> dag_store).

``atomic_write_text`` is the only crash-safe writer in HAWKING-py.
``atomic_write_json`` is the JSON adapter. The read-only helpers centralize
the deliberately narrow fallback contracts used when optional receipts and
identity files are absent or malformed. Callers that used to ship private
``_atomic_write*``, JSON-copy, JSON-reader, or file-digest helpers re-export
these names.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union


def atomic_write_text(path: Union[str, Path], text: str) -> None:
    """Write UTF-8 text via a same-directory temp file, fsync, and ``os.replace``.

    A crash mid-write leaves the live path intact. JSON receipts, GOAL.md,
    mutation locks, and runtime ownership files all go through here.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = text if isinstance(text, str) else str(text)
    tmp_name = f".{dest.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    tmp_path = dest.parent / tmp_name
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, dest)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def atomic_write_bytes(path: Union[str, Path], payload: bytes) -> None:
    """Write bytes via the same-directory temp + fsync + replace contract."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(payload, bytes):
        raise TypeError(f"payload must be bytes, got {type(payload).__name__}")
    tmp_name = f".{dest.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    tmp_path = dest.parent / tmp_name
    try:
        with open(tmp_path, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, dest)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def atomic_write_json(path: Union[str, Path], obj: Any) -> None:
    """Write JSON via ``atomic_write_text`` (indent=2, sort_keys=True)."""
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True))


def json_compatible_copy(value: Any) -> Any:
    """Return a JSON-compatible copy, falling back to its string form."""
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)


def read_json_object_or_none(
        path: Path, *, maximum: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Read a UTF-8 JSON object, optionally refusing an over-limit file.

    ``maximum`` retains the bounded-read contract required by source/admission
    callers: it checks the reported size and reads at most one byte beyond the
    limit, so a racing or pseudo-file cannot turn an optional identity lookup
    into an unbounded allocation.
    """
    try:
        if maximum is None:
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            limit = int(maximum)
            if limit < 0 or path.stat().st_size > limit:
                return None
            with path.open("rb") as handle:
                raw = handle.read(limit + 1)
            if len(raw) > limit:
                return None
            value = json.loads(raw.decode("utf-8"))
    except (OSError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hexdigest for an in-memory bytes-like object."""
    return hashlib.sha256(data).hexdigest()


def sha256_file_or_none(path: Path) -> Optional[str]:
    """Stream a file in 1 MiB chunks, returning ``None`` when it cannot be read."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None

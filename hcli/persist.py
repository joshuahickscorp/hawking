"""Leaf persistence helpers.

This module has no hcli imports so it can sit under dag_store, resources,
ledger, runtime, and grok_bridge without recreating the old SCC
(dag_store -> workunit -> resources -> max_policy -> dag_store).

``atomic_write_text`` is the only crash-safe writer in HCLI-py.
``atomic_write_json`` is the JSON adapter. Callers that used to ship a
private ``_atomic_write*`` re-export these names.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Union


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

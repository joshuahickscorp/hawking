"""Sealed provenance pin helpers (C-SCI-R1)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

def pin_digest(obj: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(dict(obj)))

def verify_pin(obj: Mapping[str, Any], expected: str) -> bool:
    return pin_digest(obj) == expected

"""Canonical JSON, SHA checks, and time stamps for lineage machinery."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping


class LineageValueError(ValueError):
    """Malformed lineage input."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def digest(value: Any) -> str:
    data = value if isinstance(value, (bytes, bytearray)) else canonical(value)
    return hashlib.sha256(data).hexdigest()


def labeled_sha(label: str) -> str:
    return hashlib.sha256(f"hawking.lineage/{label}".encode("utf-8")).hexdigest()


def require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise LineageValueError(f"{name} must be a 64-character sha256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise LineageValueError(f"{name} must be a hexadecimal sha256") from exc
    return value.lower()


def require_nonempty_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LineageValueError(f"{name} must be a non-empty string")
    return value.strip()


def require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LineageValueError(f"{name} must be a mapping")
    return value


def bpw_key(value: object) -> str:
    """Exact-enough BPW compare key. Same literal pair compares equal."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise LineageValueError("representation_bpw must be a number or numeric string")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LineageValueError("representation_bpw must be numeric") from exc
    if number <= 0.0:
        raise LineageValueError("representation_bpw must be positive")
    return format(number, ".6f")

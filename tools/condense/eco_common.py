#!/usr/bin/env python3.12
"""Shared foundation for the Hawking Condenser Ecosystem Frontier scaffold.

This is the additive, default-off successor layer described in
`docs/plans/CONDENSER_ECOSYSTEM_FRONTIER.md`. It changes how proven artifacts are
wired, summoned, contextualized, and presented AFTER they exist. It does not
mutate, relabel, interrupt, or reinterpret the active Doctor result campaign.

Everything in the `eco_*` module family is:
  - additive: it creates new documents, schemas, state stores, and CLI surfaces;
  - default-off: nothing activates until the campaign supersession gate is signed;
  - non-interfering: it reads campaign artifacts by content hash, read-only, and
    never writes into a campaign-owned directory.

This module provides the canonical hashing/sealing form, atomic writes, safe JSON
reads, and the schema registry. The hashing form is byte-identical to the campaign
reporter's `_canonical` / `_hash_value` / `_sealed`, so an imported cell's declared
`result_sha256` / `disposition_sha256` validates here exactly as it does in the
campaign. Verified against real campaign seals on 2026-07-16.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any

# ── schema registry (every emitted artifact carries one) ──────────────────────────────
SCHEMA_PASSPORT = "hawking.eco.passport.v1"
SCHEMA_IDENTITY_EDGE = "hawking.eco.identity_edge.v1"
SCHEMA_PRIOR_LEDGER = "hawking.eco.prior_ledger.v1"
SCHEMA_ADAPTIVE_PLAN = "hawking.eco.adaptive_plan.v1"
SCHEMA_PIPELINE = "hawking.eco.pipeline.v1"
SCHEMA_PIPELINE_STATE = "hawking.eco.pipeline_state.v1"
SCHEMA_ACTIVATION = "hawking.eco.activation.v1"
SCHEMA_ACTIVATION_MANIFEST = "hawking.eco.activation_manifest.v1"
SCHEMA_ADMISSION_PLAN = "hawking.eco.admission_plan.v1"
SCHEMA_STATUS = "hawking.eco.status.v1"

# The campaign this scaffold supersedes (bound as immutable evidence, never mutated).
CAMPAIGN_PLAN_SHA256 = "3d254b5f7fcc5f02b55f2a71f306f7f6852839b699fd14ab4ddf5a05dbaa0106"

SHA256_RE = re.compile(r"[0-9a-f]{64}")
MAX_JSON_BYTES = 64 * 1024 * 1024  # generous; campaign_plan.json is ~1.5 MB


class EcoError(RuntimeError):
    """Any fail-closed error in the ecosystem-frontier scaffold."""


def repo_root() -> Path:
    """The repository root of THIS checkout (the worktree, when run from one)."""
    return Path(__file__).resolve().parents[2]


def now_iso() -> str:
    """UTC timestamp, seconds precision. Byte-identical to the campaign reporter."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ── canonical hashing / sealing (byte-identical to the campaign reporter) ──────────────
def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def hash_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sealed(value: dict[str, Any], field: str) -> bool:
    """True when `value[field]` equals the sha256 of the object minus that field.

    This is the campaign reporter's exact self-seal check; an imported result or
    disposition validates here iff it validates in the campaign.
    """
    if not isinstance(value, dict) or field not in value:
        return False
    return value.get(field) == hash_value({k: v for k, v in value.items() if k != field})


def seal_field(value: dict[str, Any], field: str) -> dict[str, Any]:
    """Return a copy of `value` with `field` set to the canonical self-seal."""
    body = {k: v for k, v in value.items() if k != field}
    out = dict(value)
    out[field] = hash_value(body)
    return out


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


# ── safe IO ────────────────────────────────────────────────────────────────────────────
def read_json_safe(path: str | os.PathLike[str], *, max_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    """Read a JSON object with symlink rejection and a size cap. No path confinement:
    the campaign root is an explicit, trusted config value that may legitimately live
    outside this checkout (the live campaign runs in the main working copy). Callers
    that need confinement should resolve+check against a trusted root separately.
    """
    p = Path(path)
    try:
        resolved = p.resolve(strict=True)
    except OSError as exc:
        raise EcoError(f"absent JSON input: {p}") from exc
    if p.is_symlink() or resolved.is_symlink():
        raise EcoError(f"symlink JSON input is forbidden: {p}")
    if not resolved.is_file():
        raise EcoError(f"not a regular file: {p}")
    if resolved.stat().st_size > max_bytes:
        raise EcoError(f"oversized JSON input: {p}")
    try:
        value = json.loads(resolved.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EcoError(f"invalid JSON {p}: {exc}") from exc
    if not isinstance(value, dict):
        raise EcoError(f"JSON root is not an object: {p}")
    return value


def sha_file(path: str | os.PathLike[str]) -> tuple[str, int]:
    """Streamed sha256 + byte count for a file (matches the campaign's file hashing)."""
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def atomic_write_json(path: str | os.PathLike[str], value: dict[str, Any]) -> Path:
    """Atomic, fsync'd JSON write (temp + os.replace). Matches the campaign idiom."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def eco_state_root() -> Path:
    """The additive report/state namespace for the ecosystem scaffold.

    Deliberately separate from `reports/condense/doctor_v5_ultra/` so nothing here
    can touch campaign-owned state.
    """
    return repo_root() / "reports" / "condense" / "frontier_eco"


__all__ = [
    "EcoError", "SCHEMA_PASSPORT", "SCHEMA_IDENTITY_EDGE", "SCHEMA_PRIOR_LEDGER",
    "SCHEMA_ADAPTIVE_PLAN", "SCHEMA_PIPELINE", "SCHEMA_PIPELINE_STATE",
    "SCHEMA_ACTIVATION", "SCHEMA_ACTIVATION_MANIFEST", "SCHEMA_ADMISSION_PLAN",
    "SCHEMA_STATUS", "CAMPAIGN_PLAN_SHA256",
    "repo_root", "now_iso", "canonical_bytes", "hash_value", "sealed", "seal_field",
    "is_sha256", "read_json_safe", "sha_file", "atomic_write_json", "eco_state_root",
]

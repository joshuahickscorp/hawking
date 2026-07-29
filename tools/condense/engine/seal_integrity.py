#!/usr/bin/env python3.12
"""Sealing and release integrity guards (engine-level).

These are properties of *sealing and release*, not of any one campaign.
Retired controllers keep their campaign-specific fixtures and tests; the
invariants themselves live here so every campaign can bind them.

Guards
------
* **reseal rejection** — a document that was mutated and re-sealed must not
  verify equal to the live rebuild of the same authority surface.
* **launcher node safety** — a release launcher must be a regular file, single
  hard-link, and carry the declared mode (no symlink / hardlink farm / 0700).
* **no-subprocess preflight** — pure verification paths must not shell out;
  callers assert this by never installing a subprocess runner into the verify
  path (the engine records the contract bit ``subprocess_used_by_verifier``).
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Callable, Mapping


class SealIntegrityError(RuntimeError):
    """A sealing / release integrity invariant failed."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def seal_document(value: Mapping[str, Any], *, seal_key: str = "seal_sha256") -> dict[str, Any]:
    unsigned = {k: v for k, v in value.items() if k != seal_key}
    return {
        **unsigned,
        seal_key: hashlib.sha256(canonical_json(unsigned)).hexdigest(),
    }


def verify_document_seal(
    value: Mapping[str, Any],
    *,
    label: str = "document",
    seal_key: str = "seal_sha256",
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SealIntegrityError(f"{label} is not a JSON object")
    recorded = value.get(seal_key)
    expected = seal_document(dict(value), seal_key=seal_key)[seal_key]
    if recorded != expected:
        raise SealIntegrityError(
            f"{label} seal mismatch: recorded={recorded!r} expected={expected}"
        )
    return dict(value)


def reject_resealed_substitution(
    observed: Mapping[str, Any],
    expected_builder: Callable[[], Mapping[str, Any]],
    *,
    label: str = "binding",
    match: str = "exact deterministic runtime",
) -> dict[str, Any]:
    """Reject a resealed document that does not match the live rebuild.

    Callers pass a zero-arg builder that re-derives the sealed authority surface
    from the trusted roots. Equality is over canonical JSON so key order and
    whitespace cannot mask a substitution.
    """
    verify_document_seal(observed, label=label)
    expected = expected_builder()
    if canonical_json(dict(observed)) != canonical_json(dict(expected)):
        raise SealIntegrityError(f"{label} is not the {match}")
    return dict(observed)


def inspect_launcher_node(
    path: Path,
    *,
    label: str = "launcher",
    expected_mode: int | None = 0o755,
    require_single_hard_link: bool = True,
    refuse_symlink: bool = True,
) -> os.stat_result:
    """Fail closed on symlink, multi-hard-link, or wrong-mode launcher nodes."""
    clean = Path(path)
    try:
        st = os.lstat(clean)
    except OSError as exc:
        raise SealIntegrityError(f"cannot stat {label}: {exc}") from exc
    if refuse_symlink and stat.S_ISLNK(st.st_mode):
        raise SealIntegrityError(f"{label} must be a regular file, not a symlink")
    if not stat.S_ISREG(st.st_mode):
        raise SealIntegrityError(f"{label} must be a regular file")
    if require_single_hard_link and st.st_nlink != 1:
        raise SealIntegrityError(f"{label} must not be a hard-link farm (nlink={st.st_nlink})")
    if expected_mode is not None and stat.S_IMODE(st.st_mode) != expected_mode:
        raise SealIntegrityError(
            f"{label} mode must be {expected_mode:04o}, got {stat.S_IMODE(st.st_mode):04o}"
        )
    return st


def preflight_must_not_use_subprocess(
    *,
    subprocess_used: bool,
    label: str = "preflight",
) -> None:
    """Record the engine contract that pure preflight never shells out."""
    if subprocess_used:
        raise SealIntegrityError(f"{label} must not call subprocess")

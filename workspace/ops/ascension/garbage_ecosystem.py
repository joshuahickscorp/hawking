"""Garbage ecosystem — PINNED / LEASED / EVICTABLE / QUARANTINED (bible §26).

Generalizes tonight's live reclaim operator:

    workspace/campaign/records/runs/frankenstein/reclaim_storage_keep_proto.py

That script already implements a real allow-listed DELETE_TARGETS set, a hard
ALLOWED_ROOTS confinement, and a protected PROTO_DIR that must never be
deleted. This module lifts that pattern into the full four-state model the
ascension bible requires, without mutating the live reclaim path (GLM recapture
still depends on the disk-floor safety of the original script).

Automatic deletion is fail-closed: every gate must pass. Unclassified paths are
QUARANTINED, never EVICTABLE.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


class ObjectClass(str, enum.Enum):
    """Bible §26 object classes."""

    PINNED = "PINNED"
    LEASED = "LEASED"
    EVICTABLE = "EVICTABLE"
    QUARANTINED = "QUARANTINED"


# Paths / name fragments that must never be auto-deleted (bible §26).
# Patterns are matched as path components or substrings of the resolved path.
NEVER_AUTO_DELETE_MARKERS: tuple[str, ...] = (
    "frankenstein",
    "proto-frankenstein",
    "hawking-frankenstein",
    # stable Hawking tree markers — code, not disposable cache
    ".git",
    # protected authorities / sole rollback
    "sole-rollback",
    "rollback",
    "receipt-authority",
    "protected-authority",
)

# Directory basenames treated as user / unknown worktrees (PINNED or QUARANTINED).
USER_OR_UNKNOWN_WORKTREE_MARKERS: tuple[str, ...] = (
    "worktrees",
    ".worktrees",
    "claude-grok",
)


@dataclass(frozen=True)
class ObjectRecord:
    """Classified storage object under the garbage ecosystem."""

    path: str
    object_class: ObjectClass
    sandbox_owned: bool
    reasons: tuple[str, ...] = ()
    active_references: int = 0
    receipt_sealed: bool = False
    successor_or_rejection_verified: bool = False
    rollback_preserved: bool = False
    remote_hash_verified: Optional[bool] = None  # None = not required
    labels: tuple[str, ...] = ()
    size_bytes: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["object_class"] = self.object_class.value
        return d


@dataclass(frozen=True)
class AutoDeleteDecision:
    """Outcome of the automatic-deletion gate evaluation."""

    path: str
    allowed: bool
    object_class: ObjectClass
    gates: Mapping[str, bool]
    refuse_reasons: tuple[str, ...]
    would_delete: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "allowed": self.allowed,
            "object_class": self.object_class.value,
            "gates": dict(self.gates),
            "refuse_reasons": list(self.refuse_reasons),
            "would_delete": self.would_delete,
        }


def _norm(path: str | os.PathLike[str]) -> str:
    return str(Path(path).expanduser())


def _resolved(path: str | os.PathLike[str]) -> str:
    try:
        return str(Path(path).expanduser().resolve())
    except OSError:
        return _norm(path)


def never_auto_delete_reason(
    path: str | os.PathLike[str],
    *,
    extra_protected: Sequence[str] = (),
) -> Optional[str]:
    """Return a refuse reason if ``path`` is in the never-auto-delete set.

    Bible §26 never auto-delete:
      Frankenstein, stable Hawking, protected authorities, user files,
      unknown worktrees, sole rollback, unclassified directories.
    """
    p = _resolved(path).lower()
    name = Path(path).name.lower()
    for marker in NEVER_AUTO_DELETE_MARKERS:
        if marker in p or marker == name:
            return f"never-auto-delete marker: {marker}"
    for marker in extra_protected:
        m = marker.lower()
        if m and (m in p or Path(path).name.lower() == Path(m).name.lower()):
            return f"explicitly protected: {marker}"
    # Unclassified directories are handled by classify → QUARANTINED; the
    # auto-delete gate still refuses anything not EVICTABLE.
    return None


def classify_object(
    path: str | os.PathLike[str],
    *,
    sandbox_roots: Sequence[str | os.PathLike[str]] = (),
    pinned_paths: Sequence[str | os.PathLike[str]] = (),
    leased_paths: Sequence[str | os.PathLike[str]] = (),
    evictable_paths: Sequence[str | os.PathLike[str]] = (),
    quarantine_paths: Sequence[str | os.PathLike[str]] = (),
    labels: Sequence[str] = (),
    active_references: int = 0,
    receipt_sealed: bool = False,
    successor_or_rejection_verified: bool = False,
    rollback_preserved: bool = False,
    remote_hash_verified: Optional[bool] = None,
    size_bytes: int = 0,
    known_partial_or_corrupt: bool = False,
    unknown_ownership: bool = False,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ObjectRecord:
    """Classify a path into PINNED / LEASED / EVICTABLE / QUARANTINED.

    Classification order (first match wins after quarantine hard-fails):

    1. QUARANTINED — partial/corrupt, unknown ownership, or never-auto-delete
       without an explicit pin (unclassified user/unknown worktree).
    2. PINNED — explicit pin list, never-auto-delete markers, user worktrees.
    3. LEASED — explicit lease list, or active_references > 0 under sandbox.
    4. EVICTABLE — explicit allow-list under sandbox roots only
       (generalizes reclaim_storage_keep_proto.DELETE_TARGETS + ALLOWED_ROOTS).
    5. QUARANTINED — default for anything unclassified.
    """
    raw = _norm(path)
    resolved = _resolved(path)
    label_t = tuple(labels)
    meta = dict(metadata or {})

    sandbox_resolved = [_resolved(r) for r in sandbox_roots]
    sandbox_owned = any(
        resolved == r or resolved.startswith(r + os.sep) for r in sandbox_resolved
    ) if sandbox_resolved else False

    pinned_set = {_resolved(p) for p in pinned_paths}
    leased_set = {_resolved(p) for p in leased_paths}
    evictable_set = {_resolved(p) for p in evictable_paths}
    quarantine_set = {_resolved(p) for p in quarantine_paths}

    reasons: list[str] = []

    # Hard quarantine conditions
    if known_partial_or_corrupt:
        reasons.append("partial atomic output or failed verification")
        return ObjectRecord(
            path=raw,
            object_class=ObjectClass.QUARANTINED,
            sandbox_owned=sandbox_owned,
            reasons=tuple(reasons),
            active_references=active_references,
            receipt_sealed=receipt_sealed,
            successor_or_rejection_verified=successor_or_rejection_verified,
            rollback_preserved=rollback_preserved,
            remote_hash_verified=remote_hash_verified,
            labels=label_t,
            size_bytes=size_bytes,
            metadata=meta,
        )
    if unknown_ownership or resolved in quarantine_set:
        reasons.append("unknown ownership or explicit quarantine")
        return ObjectRecord(
            path=raw,
            object_class=ObjectClass.QUARANTINED,
            sandbox_owned=sandbox_owned,
            reasons=tuple(reasons) or ("quarantine list",),
            active_references=active_references,
            receipt_sealed=receipt_sealed,
            successor_or_rejection_verified=successor_or_rejection_verified,
            rollback_preserved=rollback_preserved,
            remote_hash_verified=remote_hash_verified,
            labels=label_t,
            size_bytes=size_bytes,
            metadata=meta,
        )

    # PINNED
    never_reason = never_auto_delete_reason(resolved)
    if resolved in pinned_set:
        reasons.append("explicit PINNED list")
        return _record(
            raw, ObjectClass.PINNED, sandbox_owned, reasons, active_references,
            receipt_sealed, successor_or_rejection_verified, rollback_preserved,
            remote_hash_verified, label_t, size_bytes, meta,
        )
    if never_reason:
        reasons.append(never_reason)
        return _record(
            raw, ObjectClass.PINNED, sandbox_owned, reasons, active_references,
            receipt_sealed, successor_or_rejection_verified, rollback_preserved,
            remote_hash_verified, label_t, size_bytes, meta,
        )
    if any(m in resolved.lower() for m in USER_OR_UNKNOWN_WORKTREE_MARKERS):
        reasons.append("user or unknown worktree — never auto-delete")
        return _record(
            raw, ObjectClass.PINNED, sandbox_owned, reasons, active_references,
            receipt_sealed, successor_or_rejection_verified, rollback_preserved,
            remote_hash_verified, label_t, size_bytes, meta,
        )

    # LEASED
    if resolved in leased_set or active_references > 0:
        if active_references > 0:
            reasons.append(f"active_references={active_references}")
        if resolved in leased_set:
            reasons.append("explicit LEASED list")
        return _record(
            raw, ObjectClass.LEASED, sandbox_owned, reasons, active_references,
            receipt_sealed, successor_or_rejection_verified, rollback_preserved,
            remote_hash_verified, label_t, size_bytes, meta,
        )

    # EVICTABLE — only under sandbox roots + allow-list (reclaim pattern)
    if resolved in evictable_set:
        if not sandbox_owned and sandbox_resolved:
            reasons.append("evictable-listed but outside sandbox roots → QUARANTINED")
            return _record(
                raw, ObjectClass.QUARANTINED, False, reasons, active_references,
                receipt_sealed, successor_or_rejection_verified, rollback_preserved,
                remote_hash_verified, label_t, size_bytes, meta,
            )
        reasons.append("explicit EVICTABLE allow-list under sandbox")
        return _record(
            raw, ObjectClass.EVICTABLE, sandbox_owned or not sandbox_resolved,
            reasons, active_references, receipt_sealed,
            successor_or_rejection_verified, rollback_preserved,
            remote_hash_verified, label_t, size_bytes, meta,
        )

    # Default: unclassified → QUARANTINED
    reasons.append("unclassified path — default QUARANTINED")
    return _record(
        raw, ObjectClass.QUARANTINED, sandbox_owned, reasons, active_references,
        receipt_sealed, successor_or_rejection_verified, rollback_preserved,
        remote_hash_verified, label_t, size_bytes, meta,
    )


def _record(
    path: str,
    object_class: ObjectClass,
    sandbox_owned: bool,
    reasons: list[str],
    active_references: int,
    receipt_sealed: bool,
    successor_or_rejection_verified: bool,
    rollback_preserved: bool,
    remote_hash_verified: Optional[bool],
    labels: tuple[str, ...],
    size_bytes: int,
    metadata: Mapping[str, Any],
) -> ObjectRecord:
    return ObjectRecord(
        path=path,
        object_class=object_class,
        sandbox_owned=sandbox_owned,
        reasons=tuple(reasons),
        active_references=active_references,
        receipt_sealed=receipt_sealed,
        successor_or_rejection_verified=successor_or_rejection_verified,
        rollback_preserved=rollback_preserved,
        remote_hash_verified=remote_hash_verified,
        labels=labels,
        size_bytes=size_bytes,
        metadata=dict(metadata),
    )


def evaluate_auto_delete(
    record: ObjectRecord,
    *,
    apply: bool = False,
) -> AutoDeleteDecision:
    """Evaluate automatic-deletion gates (bible §26).

    All of the following must be true:

    - sandbox-owned
    - EVICTABLE
    - no active references
    - receipt sealed
    - successor or rejection verified
    - rollback preserved
    - remote hash verified when required (``remote_hash_verified is not False``
      when the field is set; ``None`` means not required)

    ``apply=True`` only reports ``would_delete``; this scaffold never performs
    filesystem deletion (live reclaim remains the sole mutator).
    """
    gates = {
        "sandbox_owned": bool(record.sandbox_owned),
        "evictable": record.object_class is ObjectClass.EVICTABLE,
        "no_active_references": record.active_references == 0,
        "receipt_sealed": bool(record.receipt_sealed),
        "successor_or_rejection_verified": bool(record.successor_or_rejection_verified),
        "rollback_preserved": bool(record.rollback_preserved),
        "remote_hash_ok": record.remote_hash_verified is not False,
    }
    refuse: list[str] = []
    never = never_auto_delete_reason(record.path)
    if never:
        refuse.append(never)
        gates["not_never_auto_delete"] = False
    else:
        gates["not_never_auto_delete"] = True

    for name, ok in gates.items():
        if not ok:
            refuse.append(f"gate failed: {name}")

    allowed = all(gates.values()) and not refuse
    # Strip duplicate refuse lines when never-reason already listed
    refuse_u = tuple(dict.fromkeys(refuse))
    return AutoDeleteDecision(
        path=record.path,
        allowed=allowed,
        object_class=record.object_class,
        gates=gates,
        refuse_reasons=refuse_u if not allowed else (),
        would_delete=bool(apply and allowed),
    )


def classify_many(
    paths: Iterable[str | os.PathLike[str]],
    **kwargs: Any,
) -> list[ObjectRecord]:
    """Classify a batch of paths with the same policy kwargs."""
    return [classify_object(p, **kwargs) for p in paths]


def build_cleanup_receipt(
    decisions: Sequence[AutoDeleteDecision],
    *,
    free_bytes_before: int,
    free_bytes_after: int,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Emit a cleanup receipt (bible §26 after-cleanup: prove free-space, receipt)."""
    body = {
        "schema": "hawking.ascension.cleanup_receipt.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "free_bytes_before": free_bytes_before,
        "free_bytes_after": free_bytes_after,
        "free_bytes_recovered": max(0, free_bytes_after - free_bytes_before),
        "decisions": [d.as_dict() for d in decisions],
        "deleted_count": sum(1 for d in decisions if d.would_delete),
        "refused_count": sum(1 for d in decisions if not d.allowed),
    }
    # Self-seal (campaign pattern from eco_common.seal_field)
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    body["receipt_sha256"] = digest
    return body


# ---------------------------------------------------------------------------
# Mapping from tonight's reclaim_storage_keep_proto constants (documentation
# helper — does not import or call the live script).
# ---------------------------------------------------------------------------

RECLAIM_PROVENANCE = {
    "source_file": (
        "workspace/campaign/records/runs/frankenstein/reclaim_storage_keep_proto.py"
    ),
    "generalized_concepts": {
        "DELETE_TARGETS": "evictable_paths allow-list",
        "ALLOWED_ROOTS": "sandbox_roots confinement",
        "PROTO_DIR": "PINNED + never_auto_delete (proto-frankenstein / Desktop)",
        "proto_present() receipt endpoint check": (
            "receipt_sealed + successor_or_rejection_verified gates"
        ),
        "dry-run default / --apply": "evaluate_auto_delete(apply=...) + cleanup receipt",
    },
}

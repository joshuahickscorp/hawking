"""Ascension notifications — bible §28 event vocabulary.

Generalizes tonight's live Telegram notifier:

    workspace/campaign/records/runs/frankenstein/v0_notifier.py

That daemon already implements a real external channel (Telegram via Keychain),
disk-floor alerts, lane start/finish, capture milestones, heartbeat, and
benchmark-complete pings. This module lifts its event surface to the full
section-28 vocabulary without starting a daemon or sending messages.

Hard rule (bible §28):
  No notification may declare completion solely because a sandbox model said so.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence


class NotificationKind(str, enum.Enum):
    """Bible §28 reportable events + a few operational extensions from v0_notifier."""

    # Section 28 required set
    TG_RUNG_CANDIDATE = "tg_rung_candidate"
    TG3_REVIEW_REQUIRED = "tg3_review_required"
    PARITY_REJECTION = "parity_rejection"
    REVIEWER_DISAGREEMENT = "reviewer_disagreement"
    REPEATED_FAILURE = "repeated_failure"
    MEMORY_DISK_PRESSURE = "memory_disk_pressure"
    NEW_MODEL_ADMITTED = "new_model_admitted"
    BENCHMARK_COMPLETE = "benchmark_complete"
    HUMAN_DECISION_REQUIRED = "human_decision_required"

    # Retained from v0_notifier operational surface (not completion authority)
    HEARTBEAT = "heartbeat"
    LANE_STATE = "lane_state"
    CAPTURE_MILESTONE = "capture_milestone"
    DISK_FLOOR = "disk_floor"
    PROTO_SEALED = "proto_sealed"  # sealed receipt endpoint, not model self-claim


class AuthoritySource(str, enum.Enum):
    """Who/what is allowed to underwrite a completion-shaped claim."""

    HUMAN = "human"
    SEALED_RECEIPT = "sealed_receipt"  # hash-sealed campaign receipt
    INDEPENDENT_HARNESS = "independent_harness"  # external verify bench
    PRESSURE_GOVERNOR = "pressure_governor"
    SUPERVISOR = "supervisor"  # non-sandbox orchestrator
    SANDBOX_MODEL = "sandbox_model"  # never sufficient alone for completion


# Kinds that assert a terminal / success outcome and therefore need non-sandbox authority.
_COMPLETION_SHAPED: frozenset[NotificationKind] = frozenset(
    {
        NotificationKind.BENCHMARK_COMPLETE,
        NotificationKind.NEW_MODEL_ADMITTED,
        NotificationKind.TG_RUNG_CANDIDATE,
        NotificationKind.PROTO_SEALED,
    }
)

_ALLOWED_COMPLETION_AUTHORITIES: frozenset[AuthoritySource] = frozenset(
    {
        AuthoritySource.HUMAN,
        AuthoritySource.SEALED_RECEIPT,
        AuthoritySource.INDEPENDENT_HARNESS,
        AuthoritySource.SUPERVISOR,
    }
)


@dataclass(frozen=True)
class NotificationEvent:
    """A single outbound notification candidate (not yet sent)."""

    kind: NotificationKind
    summary: str
    authority: AuthoritySource
    severity: str = "info"  # info | warn | error | critical
    details: Mapping[str, Any] = field(default_factory=dict)
    evidence_paths: tuple[str, ...] = ()
    repeated_count: int = 0
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    may_send: bool = True
    refuse_reason: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "summary": self.summary,
            "authority": self.authority.value,
            "severity": self.severity,
            "details": dict(self.details),
            "evidence_paths": list(self.evidence_paths),
            "repeated_count": self.repeated_count,
            "timestamp_utc": self.timestamp_utc,
            "may_send": self.may_send,
            "refuse_reason": self.refuse_reason,
        }

    def render_text(self, *, prefix: str = "[ascension]") -> str:
        bits = [prefix, self.kind.value.upper(), self.summary]
        if self.repeated_count > 1:
            bits.append(f"(×{self.repeated_count})")
        if self.severity in ("warn", "error", "critical"):
            bits.insert(1, self.severity.upper())
        return " ".join(bits)


def may_declare_completion(
    *,
    kind: NotificationKind,
    authority: AuthoritySource,
    evidence_paths: Sequence[str] = (),
    require_evidence: bool = True,
) -> tuple[bool, Optional[str]]:
    """Bible §28: no completion solely because a sandbox model said so.

    Returns ``(allowed, refuse_reason)``.
    """
    if kind not in _COMPLETION_SHAPED:
        return True, None
    if authority is AuthoritySource.SANDBOX_MODEL:
        return False, "completion-shaped event cannot be authorized by sandbox_model alone"
    if authority not in _ALLOWED_COMPLETION_AUTHORITIES:
        return False, f"authority {authority.value} cannot underwrite completion-shaped {kind.value}"
    if require_evidence and not evidence_paths and authority is not AuthoritySource.HUMAN:
        return False, "completion-shaped event requires evidence_paths (or human authority)"
    return True, None


def build_notification(
    kind: NotificationKind | str,
    summary: str,
    *,
    authority: AuthoritySource | str = AuthoritySource.SUPERVISOR,
    severity: str = "info",
    details: Optional[Mapping[str, Any]] = None,
    evidence_paths: Sequence[str] = (),
    repeated_count: int = 0,
    require_evidence_for_completion: bool = True,
) -> NotificationEvent:
    """Build a notification event, applying the completion-authority rule."""
    if isinstance(kind, str):
        kind = NotificationKind(kind)
    if isinstance(authority, str):
        authority = AuthoritySource(authority)

    ok, refuse = may_declare_completion(
        kind=kind,
        authority=authority,
        evidence_paths=evidence_paths,
        require_evidence=require_evidence_for_completion,
    )
    sev = severity
    if kind is NotificationKind.MEMORY_DISK_PRESSURE and sev == "info":
        sev = "warn"
    if kind is NotificationKind.HUMAN_DECISION_REQUIRED and sev == "info":
        sev = "warn"
    if kind is NotificationKind.TG3_REVIEW_REQUIRED and sev == "info":
        sev = "warn"

    return NotificationEvent(
        kind=kind,
        summary=summary,
        authority=authority,
        severity=sev,
        details=dict(details or {}),
        evidence_paths=tuple(evidence_paths),
        repeated_count=repeated_count,
        may_send=ok,
        refuse_reason=refuse,
    )


# Mapping from v0_notifier triggers → section-28 / extended kinds (docs helper).
V0_NOTIFIER_PROVENANCE = {
    "source_file": "workspace/campaign/records/runs/frankenstein/v0_notifier.py",
    "event_map": {
        "lane start/finish (grok-run status)": NotificationKind.LANE_STATE.value,
        "capture WINDOW/LAYER/SHARD milestones": NotificationKind.CAPTURE_MILESTONE.value,
        "5-min heartbeat": NotificationKind.HEARTBEAT.value,
        "DISK LOW free_gib floor": NotificationKind.DISK_FLOOR.value
        + " / "
        + NotificationKind.MEMORY_DISK_PRESSURE.value,
        "PROTO_FRANKENSTEIN_V0_FULL_LATENT_SEALED receipt endpoint": NotificationKind.PROTO_SEALED.value,
        "auto-bench summary (independent harness)": NotificationKind.BENCHMARK_COMPLETE.value,
    },
    "section_28_new_surface": [
        NotificationKind.TG_RUNG_CANDIDATE.value,
        NotificationKind.TG3_REVIEW_REQUIRED.value,
        NotificationKind.PARITY_REJECTION.value,
        NotificationKind.REVIEWER_DISAGREEMENT.value,
        NotificationKind.REPEATED_FAILURE.value,
        NotificationKind.NEW_MODEL_ADMITTED.value,
        NotificationKind.HUMAN_DECISION_REQUIRED.value,
    ],
    "hard_rule": (
        "No notification may declare completion solely because a sandbox model said so."
    ),
}


class NotificationBus:
    """In-memory bus for tests and future supervisors. Does not call Telegram."""

    def __init__(self) -> None:
        self.sent: list[NotificationEvent] = []
        self.refused: list[NotificationEvent] = []

    def publish(self, event: NotificationEvent) -> bool:
        if not event.may_send:
            self.refused.append(event)
            return False
        self.sent.append(event)
        return True

    def publish_built(self, *args: Any, **kwargs: Any) -> bool:
        return self.publish(build_notification(*args, **kwargs))

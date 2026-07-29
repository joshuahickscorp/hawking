#!/usr/bin/env python3.12
"""Typed campaign state machine — one FSM for every campaign family.

Phases are the shared lifecycle verbs. Campaign-specific variation is which
steps are declared in the ExperimentSpec, not a new controller class.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

from .spec import CampaignPhase, ExperimentSpec


class IllegalTransition(RuntimeError):
    """Raised when a transition is not permitted from the current phase."""


class Phase(str, Enum):
    """Runtime phase including bookends the spec does not list."""

    IDLE = "idle"
    PRECHECK = "precheck"
    MEASURE = "measure"
    ALLOCATE = "allocate"
    PACK = "pack"
    SEAL = "seal"
    MONITOR = "monitor"
    RESUME = "resume"
    REPORT = "report"
    COMPLETE = "complete"
    FAULT = "fault"


# Default forward edges. RESUME may enter any non-terminal phase that the
# checkpoint records as incomplete. FAULT is reachable from any active phase.
_FORWARD: dict[Phase, frozenset[Phase]] = {
    Phase.IDLE: frozenset({Phase.PRECHECK, Phase.RESUME, Phase.FAULT}),
    Phase.PRECHECK: frozenset({Phase.MEASURE, Phase.FAULT, Phase.COMPLETE}),
    Phase.MEASURE: frozenset({Phase.ALLOCATE, Phase.PACK, Phase.SEAL, Phase.FAULT}),
    Phase.ALLOCATE: frozenset({Phase.PACK, Phase.SEAL, Phase.FAULT}),
    Phase.PACK: frozenset({Phase.SEAL, Phase.MEASURE, Phase.FAULT}),
    Phase.SEAL: frozenset({Phase.MONITOR, Phase.REPORT, Phase.COMPLETE, Phase.FAULT}),
    Phase.MONITOR: frozenset({Phase.REPORT, Phase.COMPLETE, Phase.RESUME, Phase.FAULT}),
    Phase.RESUME: frozenset(
        {
            Phase.PRECHECK,
            Phase.MEASURE,
            Phase.ALLOCATE,
            Phase.PACK,
            Phase.SEAL,
            Phase.MONITOR,
            Phase.REPORT,
            Phase.FAULT,
        }
    ),
    Phase.REPORT: frozenset({Phase.COMPLETE, Phase.FAULT}),
    Phase.COMPLETE: frozenset(),
    Phase.FAULT: frozenset({Phase.RESUME, Phase.IDLE}),
}


def _phase(value: str | Phase | CampaignPhase) -> Phase:
    if isinstance(value, Phase):
        return value
    name = value.value if isinstance(value, CampaignPhase) else str(value)
    try:
        return Phase(name)
    except ValueError as exc:
        raise IllegalTransition(f"unknown phase {name!r}") from exc


@dataclass(frozen=True)
class Transition:
    source: Phase
    target: Phase
    claim_id: str
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class StateMachine:
    """In-memory typed FSM. Persistence is the checkpoint store's job."""

    campaign_id: str
    phase: Phase = Phase.IDLE
    completed_steps: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    fault_reason: str | None = None
    allowed: dict[Phase, frozenset[Phase]] = field(
        default_factory=lambda: {k: frozenset(v) for k, v in _FORWARD.items()}
    )

    @classmethod
    def for_spec(cls, spec: ExperimentSpec) -> "StateMachine":
        """Build an FSM whose forward path follows the spec's phase list."""
        phases = [_phase(p) for p in spec.phases]
        allowed: dict[Phase, frozenset[Phase]] = {
            k: frozenset(v) for k, v in _FORWARD.items()
        }
        # Tighten forward edges to the declared sequence when present.
        if phases:
            first = phases[0]
            allowed[Phase.IDLE] = frozenset({first, Phase.RESUME, Phase.FAULT})
            for left, right in zip(phases, phases[1:]):
                extra = {right, Phase.FAULT}
                if left == Phase.SEAL:
                    extra.add(Phase.COMPLETE)
                allowed[left] = frozenset(extra | {Phase.RESUME})
            last = phases[-1]
            allowed[last] = frozenset(
                (allowed.get(last) or frozenset()) | {Phase.COMPLETE, Phase.FAULT}
            )
        return cls(campaign_id=spec.campaign_id, allowed=allowed)

    def can(self, target: str | Phase | CampaignPhase) -> bool:
        return _phase(target) in self.allowed.get(self.phase, frozenset())

    def transition(
        self,
        target: str | Phase | CampaignPhase,
        *,
        claim_id: str,
        detail: Mapping[str, Any] | None = None,
    ) -> Transition:
        if not claim_id or not isinstance(claim_id, str):
            raise IllegalTransition("claim_id must be a non-empty string")
        if claim_id in self.claims:
            raise IllegalTransition(f"claim {claim_id!r} already consumed (one-use)")
        dest = _phase(target)
        if dest not in self.allowed.get(self.phase, frozenset()):
            raise IllegalTransition(
                f"illegal transition {self.phase.value!r} -> {dest.value!r} "
                f"for campaign {self.campaign_id!r}"
            )
        source = self.phase
        self.phase = dest
        self.claims.append(claim_id)
        if dest is Phase.FAULT:
            self.fault_reason = str((detail or {}).get("reason") or "fault")
        elif dest is not Phase.FAULT:
            # Leaving fault clears the reason only on successful resume path.
            if source is Phase.FAULT:
                self.fault_reason = None
        return Transition(
            source=source,
            target=dest,
            claim_id=claim_id,
            detail=dict(detail or {}),
        )

    def mark_step(self, step_id: str) -> None:
        if step_id not in self.completed_steps:
            self.completed_steps.append(step_id)

    def is_step_done(self, step_id: str) -> bool:
        return step_id in self.completed_steps

    def snapshot(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "phase": self.phase.value,
            "completed_steps": list(self.completed_steps),
            "claims": list(self.claims),
            "fault_reason": self.fault_reason,
        }

    @classmethod
    def from_snapshot(cls, raw: Mapping[str, Any]) -> "StateMachine":
        sm = cls(campaign_id=str(raw["campaign_id"]))
        sm.phase = _phase(str(raw.get("phase") or "idle"))
        sm.completed_steps = list(raw.get("completed_steps") or [])
        sm.claims = list(raw.get("claims") or [])
        sm.fault_reason = raw.get("fault_reason")
        return sm

    def restrict_to(self, phases: Iterable[str | Phase | CampaignPhase]) -> None:
        """Optional: further restrict allowed phases to a declared set."""
        allowed_set = {_phase(p) for p in phases}
        for source, targets in list(self.allowed.items()):
            self.allowed[source] = frozenset(
                t for t in targets if t in allowed_set or t in {Phase.FAULT, Phase.COMPLETE, Phase.RESUME}
            )

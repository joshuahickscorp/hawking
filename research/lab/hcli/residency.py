"""Residency modes state machine (Bible §25) — pure logic, no weight loading.

Campaign continuity does not require permanent model residency.

  Mode A — dual resident: 30B + 80B + target fit → pipeline execute N+1 while review N
  Mode B — executor resident: 30B + target fit; queue reviews until target unloads
  Mode C — fully phase separated:
      30B build → checkpoint/unload → target benchmark → seal → unload
      → 80B review → emit → unload → controller/human decide → 30B resumes

This module only tracks *logical* residency slots and legal transitions. Future
work will bind Slot load/unload to real Qwen/Gravity runtime handles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence
import uuid

from lab.receipts import seal

SCHEMA = "hawking.hcli.residency.v1"


class ResidencyMode(str, Enum):
    A_DUAL_RESIDENT = "A_dual_resident"
    B_EXECUTOR_RESIDENT = "B_executor_resident"
    C_PHASE_SEPARATED = "C_phase_separated"


class Slot(str, Enum):
    """Logical model/target occupancy slots (not device memory maps)."""

    EXECUTOR_30B = "executor_30b"
    REVIEWER_80B = "reviewer_80b"
    TARGET = "target"


class PhaseC(str, Enum):
    """Sub-phases only meaningful in Mode C (fully phase separated)."""

    IDLE = "idle"
    BUILD = "build"  # 30B resident
    CHECKPOINT_UNLOAD = "checkpoint_unload"
    BENCHMARK = "benchmark"  # target resident
    SEAL_EVIDENCE = "seal_evidence"
    UNLOAD_TARGET = "unload_target"
    REVIEW = "review"  # 80B resident
    EMIT_REVIEW = "emit_review"
    UNLOAD_REVIEWER = "unload_reviewer"
    DECIDE = "decide"  # controller/human; no model required
    RESUME = "resume"  # prepare to re-enter BUILD


class ResidencyRefusal(ValueError):
    """Illegal mode transition or slot occupancy."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# Which slots *must* be loadable for a mode to be entered.
MODE_REQUIRED_SLOTS: dict[ResidencyMode, frozenset[Slot]] = {
    ResidencyMode.A_DUAL_RESIDENT: frozenset(
        {Slot.EXECUTOR_30B, Slot.REVIEWER_80B, Slot.TARGET}
    ),
    ResidencyMode.B_EXECUTOR_RESIDENT: frozenset({Slot.EXECUTOR_30B, Slot.TARGET}),
    ResidencyMode.C_PHASE_SEPARATED: frozenset(),  # none permanently; phased
}

# Mode C: required resident slots at each sub-phase (empty = none).
PHASE_C_RESIDENTS: dict[PhaseC, frozenset[Slot]] = {
    PhaseC.IDLE: frozenset(),
    PhaseC.BUILD: frozenset({Slot.EXECUTOR_30B}),
    PhaseC.CHECKPOINT_UNLOAD: frozenset(),
    PhaseC.BENCHMARK: frozenset({Slot.TARGET}),
    PhaseC.SEAL_EVIDENCE: frozenset({Slot.TARGET}),
    PhaseC.UNLOAD_TARGET: frozenset(),
    PhaseC.REVIEW: frozenset({Slot.REVIEWER_80B}),
    PhaseC.EMIT_REVIEW: frozenset({Slot.REVIEWER_80B}),
    PhaseC.UNLOAD_REVIEWER: frozenset(),
    PhaseC.DECIDE: frozenset(),
    PhaseC.RESUME: frozenset(),
}

# Legal Mode C edges (from → frozenset of to).
PHASE_C_TRANSITIONS: dict[PhaseC, frozenset[PhaseC]] = {
    PhaseC.IDLE: frozenset({PhaseC.BUILD}),
    PhaseC.BUILD: frozenset({PhaseC.CHECKPOINT_UNLOAD}),
    PhaseC.CHECKPOINT_UNLOAD: frozenset({PhaseC.BENCHMARK}),
    PhaseC.BENCHMARK: frozenset({PhaseC.SEAL_EVIDENCE}),
    PhaseC.SEAL_EVIDENCE: frozenset({PhaseC.UNLOAD_TARGET}),
    PhaseC.UNLOAD_TARGET: frozenset({PhaseC.REVIEW}),
    PhaseC.REVIEW: frozenset({PhaseC.EMIT_REVIEW}),
    PhaseC.EMIT_REVIEW: frozenset({PhaseC.UNLOAD_REVIEWER}),
    PhaseC.UNLOAD_REVIEWER: frozenset({PhaseC.DECIDE}),
    PhaseC.DECIDE: frozenset({PhaseC.RESUME, PhaseC.IDLE}),
    PhaseC.RESUME: frozenset({PhaseC.BUILD, PhaseC.IDLE}),
}

# Legal mode switches (campaign-level). Any mode may re-enter itself (no-op).
MODE_TRANSITIONS: dict[ResidencyMode, frozenset[ResidencyMode]] = {
    ResidencyMode.A_DUAL_RESIDENT: frozenset(
        {
            ResidencyMode.A_DUAL_RESIDENT,
            ResidencyMode.B_EXECUTOR_RESIDENT,
            ResidencyMode.C_PHASE_SEPARATED,
        }
    ),
    ResidencyMode.B_EXECUTOR_RESIDENT: frozenset(
        {
            ResidencyMode.B_EXECUTOR_RESIDENT,
            ResidencyMode.A_DUAL_RESIDENT,
            ResidencyMode.C_PHASE_SEPARATED,
        }
    ),
    ResidencyMode.C_PHASE_SEPARATED: frozenset(
        {
            ResidencyMode.C_PHASE_SEPARATED,
            ResidencyMode.B_EXECUTOR_RESIDENT,
            ResidencyMode.A_DUAL_RESIDENT,
        }
    ),
}


@dataclass
class FitReport:
    """Memory-fit stub: which slots the host claims can be co-resident."""

    can_fit: frozenset[Slot]
    reason: str = ""

    def allows(self, required: Iterable[Slot]) -> bool:
        return frozenset(required) <= self.can_fit

    def to_dict(self) -> dict[str, Any]:
        return {
            "can_fit": sorted(s.value for s in self.can_fit),
            "reason": self.reason,
        }


@dataclass
class ReviewQueueItem:
    candidate_id: str
    enqueued_at: str = field(default_factory=_utc_now)
    drained_at: str | None = None

    @property
    def pending(self) -> bool:
        return self.drained_at is None


@dataclass
class PipelinePair:
    """Mode A pipelining: executor works N+1 while reviewer works N."""

    reviewing_candidate_id: str | None = None
    executing_candidate_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewing": self.reviewing_candidate_id,
            "executing": self.executing_candidate_id,
        }


@dataclass
class ResidencyState:
    mode: ResidencyMode
    loaded: set[Slot] = field(default_factory=set)
    phase_c: PhaseC = PhaseC.IDLE
    review_queue: list[ReviewQueueItem] = field(default_factory=list)
    pipeline: PipelinePair = field(default_factory=PipelinePair)
    fit: FitReport = field(
        default_factory=lambda: FitReport(can_fit=frozenset(Slot), reason="default_assume_full")
    )
    history: list[dict[str, Any]] = field(default_factory=list)
    campaign_id: str = field(default_factory=lambda: f"res-{uuid.uuid4().hex[:10]}")

    def _note(self, event: str, **extra: Any) -> None:
        self.history.append({"event": event, "at": _utc_now(), **extra})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "campaign_id": self.campaign_id,
            "mode": self.mode.value,
            "loaded": sorted(s.value for s in self.loaded),
            "phase_c": self.phase_c.value if self.mode is ResidencyMode.C_PHASE_SEPARATED else None,
            "review_queue_pending": [
                {"candidate_id": q.candidate_id, "enqueued_at": q.enqueued_at}
                for q in self.review_queue
                if q.pending
            ],
            "pipeline": self.pipeline.to_dict(),
            "fit": self.fit.to_dict(),
        }


class ResidencyStateMachine:
    """Well-defined transitions across Modes A/B/C. No real model I/O."""

    def __init__(
        self,
        *,
        initial_mode: ResidencyMode = ResidencyMode.C_PHASE_SEPARATED,
        fit: FitReport | None = None,
        campaign_id: str | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "mode": initial_mode,
            "fit": fit
            or FitReport(can_fit=frozenset(Slot), reason="default_assume_full"),
        }
        if campaign_id is not None:
            kwargs["campaign_id"] = campaign_id
        self.state = ResidencyState(**kwargs)
        self.state._note("init", mode=initial_mode.value)
        # Enter mode cleanly (loads required slots for A/B). Fit is enforced.
        if not self.mode_fits(initial_mode):
            raise ResidencyRefusal(
                f"cannot initialise in {initial_mode.value}: fit insufficient "
                f"(need {sorted(s.value for s in MODE_REQUIRED_SLOTS[initial_mode])})"
            )
        self._apply_mode_entry(initial_mode)

    # --- Fit -----------------------------------------------------------------

    def update_fit(self, fit: FitReport) -> None:
        self.state.fit = fit
        self.state._note("fit_update", **fit.to_dict())
        # If current mode no longer fits, caller must switch_mode explicitly.
        # We do not auto-degrade: silent residency changes are a campaign hazard.

    def mode_fits(self, mode: ResidencyMode) -> bool:
        required = MODE_REQUIRED_SLOTS[mode]
        if not required:
            return True
        return self.state.fit.allows(required)

    # --- Mode switches -------------------------------------------------------

    def switch_mode(self, mode: ResidencyMode) -> ResidencyState:
        if mode not in MODE_TRANSITIONS[self.state.mode]:
            raise ResidencyRefusal(
                f"illegal mode transition {self.state.mode.value} → {mode.value}"
            )
        if not self.mode_fits(mode):
            raise ResidencyRefusal(
                f"mode {mode.value} does not fit current FitReport "
                f"(need {sorted(s.value for s in MODE_REQUIRED_SLOTS[mode])}, "
                f"can_fit {sorted(s.value for s in self.state.fit.can_fit)})"
            )
        if mode is self.state.mode and mode is not ResidencyMode.C_PHASE_SEPARATED:
            # No-op re-entry for A/B.
            return self.state
        prev = self.state.mode
        # Unload everything before mode entry (clean boundary).
        self.state.loaded.clear()
        self.state.pipeline = PipelinePair()
        if mode is not ResidencyMode.B_EXECUTOR_RESIDENT:
            # Leaving B does not drop the queue; reviews still pending.
            pass
        self.state.mode = mode
        if mode is ResidencyMode.C_PHASE_SEPARATED:
            self.state.phase_c = PhaseC.IDLE
        self._apply_mode_entry(mode, force=True)
        self.state._note("switch_mode", from_mode=prev.value, to_mode=mode.value)
        return self.state

    def _apply_mode_entry(self, mode: ResidencyMode, *, force: bool = False) -> None:
        required = MODE_REQUIRED_SLOTS[mode]
        if required and not self.state.fit.allows(required) and not force:
            raise ResidencyRefusal(f"cannot enter {mode.value}: fit insufficient")
        if mode is ResidencyMode.A_DUAL_RESIDENT:
            self.state.loaded = set(required)
        elif mode is ResidencyMode.B_EXECUTOR_RESIDENT:
            self.state.loaded = set(required)
        elif mode is ResidencyMode.C_PHASE_SEPARATED:
            # Start empty at IDLE.
            self.state.loaded = set(PHASE_C_RESIDENTS[self.state.phase_c])

    # --- Slot primitives (logical) -------------------------------------------

    def load(self, slot: Slot) -> None:
        if slot in self.state.loaded:
            return
        # Fit check: proposed co-residency must still fit.
        proposed = set(self.state.loaded) | {slot}
        if not self.state.fit.allows(proposed):
            raise ResidencyRefusal(
                f"cannot load {slot.value}: would exceed fit "
                f"(loaded={sorted(s.value for s in self.state.loaded)})"
            )
        # Mode discipline
        if self.state.mode is ResidencyMode.B_EXECUTOR_RESIDENT and slot is Slot.REVIEWER_80B:
            # In B, reviewer loads only after target unloads (queue drain window).
            if Slot.TARGET in self.state.loaded:
                raise ResidencyRefusal(
                    "Mode B: cannot load reviewer while target is resident; "
                    "unload target and drain review queue first"
                )
        if self.state.mode is ResidencyMode.C_PHASE_SEPARATED:
            allowed = PHASE_C_RESIDENTS[self.state.phase_c]
            if slot not in allowed:
                raise ResidencyRefusal(
                    f"Mode C phase {self.state.phase_c.value} does not permit "
                    f"loading {slot.value} (allowed={sorted(s.value for s in allowed)})"
                )
        self.state.loaded.add(slot)
        self.state._note("load", slot=slot.value)

    def unload(self, slot: Slot) -> None:
        if slot not in self.state.loaded:
            return
        if self.state.mode is ResidencyMode.A_DUAL_RESIDENT:
            # Dual-resident invariant: all three stay up while in A.
            raise ResidencyRefusal(
                f"Mode A is dual-resident; cannot unload {slot.value} without switch_mode"
            )
        self.state.loaded.discard(slot)
        self.state._note("unload", slot=slot.value)

    # --- Mode A: pipelining --------------------------------------------------

    def pipeline_assign(
        self,
        *,
        executing: str | None,
        reviewing: str | None,
    ) -> PipelinePair:
        if self.state.mode is not ResidencyMode.A_DUAL_RESIDENT:
            raise ResidencyRefusal("pipeline_assign only valid in Mode A")
        required = MODE_REQUIRED_SLOTS[ResidencyMode.A_DUAL_RESIDENT]
        if not required <= self.state.loaded:
            raise ResidencyRefusal("Mode A pipeline requires 30B+80B+target loaded")
        if executing is not None and reviewing is not None and executing == reviewing:
            raise ResidencyRefusal(
                "pipeline requires distinct candidates (execute N+1, review N)"
            )
        self.state.pipeline = PipelinePair(
            executing_candidate_id=executing,
            reviewing_candidate_id=reviewing,
        )
        self.state._note(
            "pipeline_assign",
            executing=executing,
            reviewing=reviewing,
        )
        return self.state.pipeline

    # --- Mode B: review queue ------------------------------------------------

    def enqueue_review(self, candidate_id: str) -> ReviewQueueItem:
        if self.state.mode not in {
            ResidencyMode.B_EXECUTOR_RESIDENT,
            ResidencyMode.A_DUAL_RESIDENT,
        }:
            # Mode C reviews inline during REVIEW phase; queue is B-primary.
            if self.state.mode is ResidencyMode.C_PHASE_SEPARATED:
                raise ResidencyRefusal(
                    "Mode C does not queue reviews; reviews run in the REVIEW phase"
                )
        item = ReviewQueueItem(candidate_id=candidate_id)
        self.state.review_queue.append(item)
        self.state._note("enqueue_review", candidate_id=candidate_id)
        return item

    def begin_review_drain(self) -> list[str]:
        """Mode B: unload target, load reviewer, return pending candidate ids."""
        if self.state.mode is not ResidencyMode.B_EXECUTOR_RESIDENT:
            raise ResidencyRefusal("begin_review_drain only valid in Mode B")
        if Slot.TARGET in self.state.loaded:
            self.state.loaded.discard(Slot.TARGET)
            self.state._note("unload", slot=Slot.TARGET.value, reason="review_drain")
        # Executor may stay or go; typically unload to free space for 80B.
        if Slot.EXECUTOR_30B in self.state.loaded:
            self.state.loaded.discard(Slot.EXECUTOR_30B)
            self.state._note("unload", slot=Slot.EXECUTOR_30B.value, reason="review_drain")
        if not self.state.fit.allows({Slot.REVIEWER_80B}):
            raise ResidencyRefusal("reviewer 80B does not fit even after target unload")
        self.state.loaded.add(Slot.REVIEWER_80B)
        self.state._note("load", slot=Slot.REVIEWER_80B.value, reason="review_drain")
        pending = [q.candidate_id for q in self.state.review_queue if q.pending]
        self.state._note("review_drain_begin", pending=pending)
        return pending

    def complete_review(self, candidate_id: str) -> None:
        found = False
        for q in self.state.review_queue:
            if q.candidate_id == candidate_id and q.pending:
                q.drained_at = _utc_now()
                found = True
                break
        if not found:
            raise ResidencyRefusal(f"no pending review for {candidate_id!r}")
        self.state._note("review_complete", candidate_id=candidate_id)

    def pending_reviews(self) -> list[str]:
        return [q.candidate_id for q in self.state.review_queue if q.pending]

    # --- Mode C: phase chain -------------------------------------------------

    def advance_phase_c(self, to: PhaseC | str) -> PhaseC:
        if self.state.mode is not ResidencyMode.C_PHASE_SEPARATED:
            raise ResidencyRefusal("advance_phase_c only valid in Mode C")
        target = to if isinstance(to, PhaseC) else PhaseC(to)
        allowed = PHASE_C_TRANSITIONS[self.state.phase_c]
        if target not in allowed:
            raise ResidencyRefusal(
                f"illegal Mode C transition {self.state.phase_c.value} → {target.value}; "
                f"allowed={sorted(p.value for p in allowed)}"
            )
        # Apply unload/load to match target phase residents.
        desired = PHASE_C_RESIDENTS[target]
        # Unload anything not desired.
        for slot in list(self.state.loaded):
            if slot not in desired:
                self.state.loaded.discard(slot)
                self.state._note("unload", slot=slot.value, phase=target.value)
        # Load desired if fit allows.
        for slot in desired:
            if slot not in self.state.loaded:
                if not self.state.fit.allows(set(self.state.loaded) | {slot}):
                    raise ResidencyRefusal(
                        f"cannot enter phase {target.value}: {slot.value} does not fit"
                    )
                self.state.loaded.add(slot)
                self.state._note("load", slot=slot.value, phase=target.value)
        prev = self.state.phase_c
        self.state.phase_c = target
        self.state._note("phase_c", from_phase=prev.value, to_phase=target.value)
        return self.state.phase_c

    def run_phase_c_cycle(self) -> list[str]:
        """Drive one full Mode C cycle IDLE→…→DECIDE→RESUME for tests/demo."""
        if self.state.mode is not ResidencyMode.C_PHASE_SEPARATED:
            raise ResidencyRefusal("run_phase_c_cycle only valid in Mode C")
        if self.state.phase_c is not PhaseC.IDLE:
            raise ResidencyRefusal(
                f"full cycle starts at IDLE, currently {self.state.phase_c.value}"
            )
        path = [
            PhaseC.BUILD,
            PhaseC.CHECKPOINT_UNLOAD,
            PhaseC.BENCHMARK,
            PhaseC.SEAL_EVIDENCE,
            PhaseC.UNLOAD_TARGET,
            PhaseC.REVIEW,
            PhaseC.EMIT_REVIEW,
            PhaseC.UNLOAD_REVIEWER,
            PhaseC.DECIDE,
            PhaseC.RESUME,
        ]
        seen: list[str] = []
        for phase in path:
            self.advance_phase_c(phase)
            seen.append(phase.value)
        return seen

    # --- Introspection -------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return seal(self.state.to_dict())

    def legal_mode_targets(self) -> list[str]:
        return sorted(
            m.value
            for m in MODE_TRANSITIONS[self.state.mode]
            if m is not self.state.mode and self.mode_fits(m)
        )

    def legal_phase_c_targets(self) -> list[str]:
        if self.state.mode is not ResidencyMode.C_PHASE_SEPARATED:
            return []
        return sorted(p.value for p in PHASE_C_TRANSITIONS[self.state.phase_c])

    @staticmethod
    def transition_tables() -> dict[str, Any]:
        """Export the full transition relation for docs/tests."""
        return {
            "schema": SCHEMA,
            "mode_transitions": {
                m.value: sorted(x.value for x in targets)
                for m, targets in MODE_TRANSITIONS.items()
            },
            "mode_required_slots": {
                m.value: sorted(s.value for s in slots)
                for m, slots in MODE_REQUIRED_SLOTS.items()
            },
            "phase_c_transitions": {
                p.value: sorted(x.value for x in targets)
                for p, targets in PHASE_C_TRANSITIONS.items()
            },
            "phase_c_residents": {
                p.value: sorted(s.value for s in slots)
                for p, slots in PHASE_C_RESIDENTS.items()
            },
            "law": "campaign_continuity_does_not_require_permanent_model_residency",
        }

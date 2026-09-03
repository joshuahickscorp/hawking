"""Tests for residency modes state machine (Bible §25)."""
from __future__ import annotations

import pytest

from lab.hcli.residency import (
    MODE_TRANSITIONS,
    PHASE_C_TRANSITIONS,
    FitReport,
    PhaseC,
    ResidencyMode,
    ResidencyRefusal,
    ResidencyStateMachine,
    Slot,
)


def fit_all() -> FitReport:
    return FitReport(can_fit=frozenset(Slot), reason="roomy host")


def fit_executor_target_only() -> FitReport:
    return FitReport(
        can_fit=frozenset({Slot.EXECUTOR_30B, Slot.TARGET, Slot.REVIEWER_80B}),
        reason="can fit any single-or-dual combo in sequence; dual+target also listed",
    )


def fit_no_dual() -> FitReport:
    """30B+target fit; 80B does not co-reside with them (Mode B territory)."""
    return FitReport(
        can_fit=frozenset({Slot.EXECUTOR_30B, Slot.TARGET}),
        reason="Mode B envelope",
    )


def fit_reviewer_alone() -> FitReport:
    return FitReport(can_fit=frozenset({Slot.REVIEWER_80B}), reason="after unload")


def test_transition_tables_are_total_for_modes():
    tables = ResidencyStateMachine.transition_tables()
    for mode in ResidencyMode:
        assert mode.value in tables["mode_transitions"]
        assert mode.value in tables["mode_required_slots"]
    for phase in PhaseC:
        assert phase.value in tables["phase_c_transitions"]
        assert phase.value in tables["phase_c_residents"]


def test_mode_a_requires_three_slots_and_pipelines():
    sm = ResidencyStateMachine(initial_mode=ResidencyMode.A_DUAL_RESIDENT, fit=fit_all())
    assert sm.state.loaded == {
        Slot.EXECUTOR_30B,
        Slot.REVIEWER_80B,
        Slot.TARGET,
    }
    pair = sm.pipeline_assign(executing="cand-N+1", reviewing="cand-N")
    assert pair.executing_candidate_id == "cand-N+1"
    assert pair.reviewing_candidate_id == "cand-N"
    with pytest.raises(ResidencyRefusal, match="distinct"):
        sm.pipeline_assign(executing="same", reviewing="same")


def test_mode_a_forbids_partial_unload_without_mode_switch():
    sm = ResidencyStateMachine(initial_mode=ResidencyMode.A_DUAL_RESIDENT, fit=fit_all())
    with pytest.raises(ResidencyRefusal, match="dual-resident"):
        sm.unload(Slot.REVIEWER_80B)


def test_mode_a_entry_refused_when_fit_insufficient():
    sm = ResidencyStateMachine(
        initial_mode=ResidencyMode.C_PHASE_SEPARATED,
        fit=fit_no_dual(),
    )
    with pytest.raises(ResidencyRefusal, match="does not fit"):
        sm.switch_mode(ResidencyMode.A_DUAL_RESIDENT)


def test_mode_b_queues_reviews_until_drain():
    sm = ResidencyStateMachine(
        initial_mode=ResidencyMode.B_EXECUTOR_RESIDENT,
        fit=FitReport(
            can_fit=frozenset({Slot.EXECUTOR_30B, Slot.TARGET, Slot.REVIEWER_80B}),
            reason="sequential fit",
        ),
    )
    assert Slot.EXECUTOR_30B in sm.state.loaded
    assert Slot.TARGET in sm.state.loaded
    assert Slot.REVIEWER_80B not in sm.state.loaded

    sm.enqueue_review("cand-1")
    sm.enqueue_review("cand-2")
    assert sm.pending_reviews() == ["cand-1", "cand-2"]

    # Cannot load reviewer while target still resident.
    with pytest.raises(ResidencyRefusal, match="while target is resident"):
        sm.load(Slot.REVIEWER_80B)

    pending = sm.begin_review_drain()
    assert pending == ["cand-1", "cand-2"]
    assert Slot.TARGET not in sm.state.loaded
    assert Slot.REVIEWER_80B in sm.state.loaded

    sm.complete_review("cand-1")
    sm.complete_review("cand-2")
    assert sm.pending_reviews() == []


def test_mode_c_full_cycle_well_defined():
    sm = ResidencyStateMachine(
        initial_mode=ResidencyMode.C_PHASE_SEPARATED,
        fit=fit_all(),
    )
    assert sm.state.phase_c is PhaseC.IDLE
    assert sm.state.loaded == set()
    path = sm.run_phase_c_cycle()
    assert path == [
        "build",
        "checkpoint_unload",
        "benchmark",
        "seal_evidence",
        "unload_target",
        "review",
        "emit_review",
        "unload_reviewer",
        "decide",
        "resume",
    ]
    assert sm.state.phase_c is PhaseC.RESUME
    # After RESUME, may re-enter BUILD (campaign continuity without permanent residency).
    sm.advance_phase_c(PhaseC.BUILD)
    assert Slot.EXECUTOR_30B in sm.state.loaded
    assert Slot.REVIEWER_80B not in sm.state.loaded


def test_mode_c_illegal_skip_refused():
    sm = ResidencyStateMachine(initial_mode=ResidencyMode.C_PHASE_SEPARATED, fit=fit_all())
    sm.advance_phase_c(PhaseC.BUILD)
    with pytest.raises(ResidencyRefusal, match="illegal Mode C transition"):
        sm.advance_phase_c(PhaseC.REVIEW)  # skipped checkpoint/benchmark/seal


def test_mode_c_phase_residency_invariants():
    sm = ResidencyStateMachine(initial_mode=ResidencyMode.C_PHASE_SEPARATED, fit=fit_all())
    sm.advance_phase_c(PhaseC.BUILD)
    assert sm.state.loaded == {Slot.EXECUTOR_30B}
    sm.advance_phase_c(PhaseC.CHECKPOINT_UNLOAD)
    assert sm.state.loaded == set()
    sm.advance_phase_c(PhaseC.BENCHMARK)
    assert sm.state.loaded == {Slot.TARGET}
    sm.advance_phase_c(PhaseC.SEAL_EVIDENCE)
    assert sm.state.loaded == {Slot.TARGET}
    sm.advance_phase_c(PhaseC.UNLOAD_TARGET)
    assert sm.state.loaded == set()
    sm.advance_phase_c(PhaseC.REVIEW)
    assert sm.state.loaded == {Slot.REVIEWER_80B}


def test_mode_c_does_not_use_review_queue():
    sm = ResidencyStateMachine(initial_mode=ResidencyMode.C_PHASE_SEPARATED, fit=fit_all())
    with pytest.raises(ResidencyRefusal, match="does not queue"):
        sm.enqueue_review("cand-x")


def test_mode_switches_are_symmetric_and_fit_gated():
    sm = ResidencyStateMachine(initial_mode=ResidencyMode.C_PHASE_SEPARATED, fit=fit_all())
    sm.switch_mode(ResidencyMode.A_DUAL_RESIDENT)
    assert sm.state.mode is ResidencyMode.A_DUAL_RESIDENT
    sm.switch_mode(ResidencyMode.B_EXECUTOR_RESIDENT)
    assert sm.state.mode is ResidencyMode.B_EXECUTOR_RESIDENT
    sm.switch_mode(ResidencyMode.C_PHASE_SEPARATED)
    assert sm.state.mode is ResidencyMode.C_PHASE_SEPARATED
    assert sm.state.phase_c is PhaseC.IDLE


def test_every_mode_c_edge_is_exercisable():
    """Prove the transition graph has no dead declared edges."""
    sm = ResidencyStateMachine(initial_mode=ResidencyMode.C_PHASE_SEPARATED, fit=fit_all())
    # Walk every declared edge at least once via BFS from IDLE.
    from collections import deque

    seen_edges: set[tuple[str, str]] = set()
    # Reset helper: re-create machine when we need a fresh IDLE.
    queue: deque[PhaseC] = deque([PhaseC.IDLE])
    visited_states: set[PhaseC] = set()

    # Exhaustive walk by replaying path prefixes.
    def walk_to(target_path: list[PhaseC]) -> ResidencyStateMachine:
        m = ResidencyStateMachine(initial_mode=ResidencyMode.C_PHASE_SEPARATED, fit=fit_all())
        for phase in target_path:
            m.advance_phase_c(phase)
        return m

    # BFS over paths
    paths: deque[list[PhaseC]] = deque([[]])  # path of advances from IDLE
    while paths:
        path = paths.popleft()
        m = walk_to(path)
        current = m.state.phase_c
        if current in visited_states and path:
            # still explore unused edges
            pass
        visited_states.add(current)
        for nxt in sorted(PHASE_C_TRANSITIONS[current], key=lambda p: p.value):
            edge = (current.value, nxt.value)
            if edge in seen_edges:
                continue
            m2 = walk_to(path)
            m2.advance_phase_c(nxt)
            seen_edges.add(edge)
            paths.append(path + [nxt])

    declared = {
        (a.value, b.value)
        for a, targets in PHASE_C_TRANSITIONS.items()
        for b in targets
    }
    assert seen_edges == declared


def test_pipeline_only_in_mode_a():
    sm = ResidencyStateMachine(initial_mode=ResidencyMode.B_EXECUTOR_RESIDENT, fit=fit_all())
    with pytest.raises(ResidencyRefusal, match="only valid in Mode A"):
        sm.pipeline_assign(executing="a", reviewing="b")


def test_snapshot_sealed():
    sm = ResidencyStateMachine(initial_mode=ResidencyMode.C_PHASE_SEPARATED, fit=fit_all())
    snap = sm.snapshot()
    assert "seal_sha256" in snap
    assert snap["mode"] == ResidencyMode.C_PHASE_SEPARATED.value


def test_mode_transition_table_covers_all_pairs_declared():
    """Sanity: every mode lists itself (no-op) and the other two modes."""
    for mode, targets in MODE_TRANSITIONS.items():
        assert mode in targets
        assert len(targets) == 3

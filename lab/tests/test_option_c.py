"""Tests for Option-C sandbox (Bible §24) — structural identity with delegate/audit."""
from __future__ import annotations

import pytest

from lab.hcli.option_c import (
    MANDATORY_REVIEW_CATEGORIES,
    CandidatePhase,
    OptionCController,
    OptionCRefusal,
    OptionCSandbox,
    Role,
)


def _open_mandatory(sandbox: OptionCSandbox, category: str = "kernel_promotion") -> str:
    s = sandbox.open_candidate(category=category, worktree_ref="worktrees/oc-cand-1")
    return s.candidate_id


def _run_executor(sandbox: OptionCSandbox, cid: str) -> None:
    sandbox.executor_start(cid)
    sandbox.executor_emit(
        cid,
        summary="candidate kernel tweak",
        changes={"files": ["kernels/foo.metal"], "diff_lines": 12},
        tests_run=["pytest lab/tests/test_frontier_controller.py"],
        evidence_paths=["receipts/candidate.json"],
    )


def _run_reviewer(sandbox: OptionCSandbox, cid: str, recommendation: str = "APPROVE_WITH_GATES") -> None:
    sandbox.reviewer_start(cid)
    sandbox.reviewer_emit(
        cid,
        challenges=[{"axis": "parity", "claim": "numeric parity still holds"}],
        distinguishing_tests_requested=["parity_v2_1_subset"],
        severity_findings=[],
        recommendation=recommendation,
    )


def test_role_map_matches_tonight_pattern():
    mapping = OptionCSandbox.role_map_to_tonight()
    assert mapping["structural_identity"] is True
    assert mapping["logical_not_simultaneous"] is True
    m = mapping["mapping"]
    assert "grok-run delegate" in m[Role.EXECUTOR.value]["reference"]
    assert "grok-run audit" in m[Role.REVIEWER.value]["reference"]
    assert m[Role.PROTECTED_CONTROLLER.value]["never_transfers"] is True
    assert set(mapping["mandatory_review_categories"]) == set(MANDATORY_REVIEW_CATEGORIES)


def test_mandatory_categories_match_bible():
    expected = {
        "kernel_promotion",
        "quantization_change",
        "routing_change",
        "benchmark_change",
        "runtime_scheduling",
        "storage_deletion",
        "artifact_promotion",
        "effect_authority",
    }
    assert set(MANDATORY_REVIEW_CATEGORIES) == expected


def test_happy_path_promote_with_mandatory_review():
    sandbox = OptionCSandbox()
    cid = _open_mandatory(sandbox)
    _run_executor(sandbox, cid)
    _run_reviewer(sandbox, cid)
    controller = OptionCController(sandbox, controller_id="claude_controller")
    controller.begin_eval(cid)
    doc = controller.decide(
        cid,
        protected_parity={"pass": True},
        held_out_capability={"pass": True},
        clean_benchmark={"pass": True},
    )
    assert doc["decision"]["action"] == "PROMOTE"
    assert doc["decision"]["result_class"] == "PROMOTED_MECHANISM"
    assert doc["fabricated_promote"] is False
    assert "seal_sha256" in doc
    assert sandbox.sessions[cid].phase is CandidatePhase.PROMOTED


def test_mandatory_review_blocks_controller_without_review():
    sandbox = OptionCSandbox()
    cid = _open_mandatory(sandbox, category="storage_deletion")
    _run_executor(sandbox, cid)
    controller = OptionCController(sandbox)
    with pytest.raises(OptionCRefusal, match="mandatory"):
        controller.begin_eval(cid)


def test_non_mandatory_can_skip_review():
    sandbox = OptionCSandbox()
    s = sandbox.open_candidate(category="docs_typo", worktree_ref="wt/docs")
    cid = s.candidate_id
    _run_executor(sandbox, cid)
    controller = OptionCController(sandbox, controller_id="controller")
    controller.begin_eval(cid)
    doc = controller.decide(
        cid,
        protected_parity={"pass": True},
        held_out_capability={"pass": True},
        clean_benchmark={"pass": True},
    )
    assert doc["decision"]["action"] == "PROMOTE"


def test_missing_protected_gates_hold_not_promote():
    sandbox = OptionCSandbox()
    cid = _open_mandatory(sandbox)
    _run_executor(sandbox, cid)
    _run_reviewer(sandbox, cid)
    controller = OptionCController(sandbox, controller_id="controller")
    controller.begin_eval(cid)
    doc = controller.decide(cid)  # no gate bundles
    assert doc["decision"]["action"] == "HOLD"
    assert doc["decision"]["result_class"] == "INSUFFICIENT_EVIDENCE"
    assert sandbox.sessions[cid].phase is CandidatePhase.INSUFFICIENT_EVIDENCE


def test_cannot_force_promote_with_pending_or_fail():
    sandbox = OptionCSandbox()
    cid = _open_mandatory(sandbox)
    _run_executor(sandbox, cid)
    _run_reviewer(sandbox, cid)
    controller = OptionCController(sandbox, controller_id="controller")
    controller.begin_eval(cid)
    with pytest.raises(OptionCRefusal, match="PENDING"):
        controller.decide(
            cid,
            protected_parity={"pass": True},
            held_out_capability=None,
            clean_benchmark={"pass": True},
            force_action="PROMOTE",
        )
    with pytest.raises(OptionCRefusal, match="FAIL"):
        controller.decide(
            cid,
            protected_parity={"pass": False},
            held_out_capability={"pass": True},
            clean_benchmark={"pass": True},
            force_action="PROMOTE",
        )


def test_reviewer_independence_fence():
    sandbox = OptionCSandbox()
    cid = _open_mandatory(sandbox)
    _run_executor(sandbox, cid)
    with pytest.raises(OptionCRefusal, match="controller conclusions"):
        sandbox.reviewer_start(cid, controller_conclusions={"I already decided": "PROMOTE"})


def test_executor_forbidden_actions_in_changes():
    sandbox = OptionCSandbox()
    cid = _open_mandatory(sandbox)
    sandbox.executor_start(cid)
    with pytest.raises(OptionCRefusal, match="may not perform"):
        sandbox.executor_emit(
            cid,
            summary="bad",
            changes={"sign_own_results": True},
            tests_run=[],
        )


def test_reviewer_reject_blocks_promotion():
    sandbox = OptionCSandbox()
    cid = _open_mandatory(sandbox)
    _run_executor(sandbox, cid)
    _run_reviewer(sandbox, cid, recommendation="REJECT")
    controller = OptionCController(sandbox, controller_id="controller")
    controller.begin_eval(cid)
    doc = controller.decide(
        cid,
        protected_parity={"pass": True},
        held_out_capability={"pass": True},
        clean_benchmark={"pass": True},
    )
    assert doc["decision"]["action"] == "REJECT"
    assert sandbox.sessions[cid].phase is CandidatePhase.REJECTED


def test_phase_order_enforced():
    sandbox = OptionCSandbox()
    cid = _open_mandatory(sandbox)
    with pytest.raises(OptionCRefusal, match="EXECUTING"):
        sandbox.executor_emit(cid, summary="x", changes={}, tests_run=[])
    sandbox.executor_start(cid)
    with pytest.raises(OptionCRefusal, match="CANDIDATE_EMITTED"):
        sandbox.reviewer_start(cid)


def test_isolated_worktree_required():
    sandbox = OptionCSandbox()
    with pytest.raises(OptionCRefusal, match="worktree"):
        sandbox.open_candidate(category="kernel_promotion", worktree_ref="  ")


def test_logical_not_simultaneous_sequential_phases():
    """Option-C runs as a sequence of phases, not requiring co-residency."""
    sandbox = OptionCSandbox()
    cid = _open_mandatory(sandbox)
    assert sandbox.sessions[cid].phase is CandidatePhase.IDLE
    _run_executor(sandbox, cid)
    assert sandbox.sessions[cid].phase is CandidatePhase.CANDIDATE_EMITTED
    _run_reviewer(sandbox, cid)
    assert sandbox.sessions[cid].phase is CandidatePhase.REVIEW_EMITTED
    # Controller phase is a third sequential step.
    OptionCController(sandbox, controller_id="c").begin_eval(cid)
    assert sandbox.sessions[cid].phase is CandidatePhase.CONTROLLER_EVAL

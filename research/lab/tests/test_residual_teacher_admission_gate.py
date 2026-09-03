"""Deterministic tests for residual teacher admission gate (local evidence only)."""
from __future__ import annotations

from typing import Any

from lab.operators.residual_teacher_admission_gate import (
    ADMISSION_SCHEMA,
    PROTECTED_AXES,
    VERDICT_ADMIT,
    VERDICT_DEFERRED,
    VERDICT_REJECT,
    default_decision,
    evaluate_residual_teacher_admission,
)
from lab.receipts import verify


MEMBERSHIP = "a" * 64
OTHER_MEMBERSHIP = "b" * 64
SEAL = "c" * 64


def _glm_baseline(**overrides: Any) -> dict[str, Any]:
    base = {
        "sealed": True,
        "seal_sha256": SEAL,
        "teacher": "glm",
        "baseline_kind": "glm_only",
        "held_out_membership_hash": MEMBERSHIP,
        "status": "SEALED",
    }
    base.update(overrides)
    return base


def _kimi_ab(**overrides: Any) -> dict[str, Any]:
    base = {
        "teacher": "kimi",
        "includes_kimi": True,
        "held_out_membership_hash": MEMBERSHIP,
        "incremental_held_out_delta": 0.04,
        "comparison": {
            "baseline": 0.50,
            "with_kimi": 0.54,
            "delta": 0.04,
        },
    }
    base.update(overrides)
    return base


def _hypothesis(**overrides: Any) -> dict[str, Any]:
    base = {
        "name": "long-horizon agentic planning residual",
        "capability": "multi-step tool orchestration beyond GLM math bridge",
        "role": "residual_lane",
    }
    base.update(overrides)
    return base


def _provenance(**overrides: Any) -> dict[str, Any]:
    base = {
        "student_revision": "dsv4f-rev-001",
        "glm_revision": "glm-proto-rev-001",
        "kimi_revision": "kimi-k3-rev-001",
        "complete": True,
    }
    base.update(overrides)
    return base


def _no_regression(**axis_overrides: Any) -> dict[str, Any]:
    axes = {
        "math": {"gate": "PASS", "delta": 0.0},
        "coding": {"gate": "PASS", "delta": 0.01},
        "tool": {"gate": "PASS", "delta": 0.0},
        "agentic": {"gate": "PASS", "delta": 0.02},
    }
    axes.update(axis_overrides)
    return {"axes": axes}


def _forward_ready(**overrides: Any) -> dict[str, Any]:
    base = {"ready": True, "status": "READY", "architecture": "dsv4f"}
    base.update(overrides)
    return base


def _full_evidence(**overrides: Any) -> dict[str, Any]:
    evidence = {
        "glm_baseline_receipt": _glm_baseline(),
        "kimi_incremental_receipt": _kimi_ab(),
        "residual_hypothesis": _hypothesis(),
        "provenance": _provenance(),
        "no_regression": _no_regression(),
        "dsv4f_architecture_forward": _forward_ready(),
    }
    evidence.update(overrides)
    return evidence


def _by_name(decision: dict[str, Any], name: str) -> dict[str, Any]:
    for row in decision["checks"]:
        if row["name"] == name:
            return row
    raise AssertionError(f"check {name!r} missing: {decision['checks']}")


# ---------------------------------------------------------------------------
# Default defer
# ---------------------------------------------------------------------------


def test_default_outcome_is_deferred() -> None:
    decision = default_decision()
    verify(decision, label="residual-teacher-default")
    assert decision["schema"] == ADMISSION_SCHEMA
    assert decision["verdict"] == VERDICT_DEFERRED
    assert decision["claim_boundary"]["default_is_deferred"] is True
    assert decision["local_only"] is True
    assert decision["claim_boundary"]["networking_or_trainer_side_effects"] is False


def test_empty_mapping_is_deferred() -> None:
    decision = evaluate_residual_teacher_admission({})
    assert decision["verdict"] == VERDICT_DEFERRED


def test_partial_evidence_defers_without_reject() -> None:
    decision = evaluate_residual_teacher_admission(
        {
            "glm_baseline_receipt": _glm_baseline(),
            # no kimi, hypothesis, provenance, regression, forward
        }
    )
    assert decision["verdict"] == VERDICT_DEFERRED
    assert "FAIL" not in {c["status"] for c in decision["checks"]}
    assert _by_name(decision, "kimi_incremental_ab_receipt")["status"] == "DEFER"


# ---------------------------------------------------------------------------
# Membership mismatch â REJECT
# ---------------------------------------------------------------------------


def test_membership_mismatch_rejects() -> None:
    decision = evaluate_residual_teacher_admission(
        _full_evidence(
            kimi_incremental_receipt=_kimi_ab(
                held_out_membership_hash=OTHER_MEMBERSHIP
            )
        )
    )
    assert decision["verdict"] == VERDICT_REJECT
    row = _by_name(decision, "held_out_membership_match")
    assert row["status"] == "FAIL"
    assert "membership differs" in row["detail"]


# ---------------------------------------------------------------------------
# Regression on protected axes â REJECT
# ---------------------------------------------------------------------------


def test_protected_axis_regression_rejects() -> None:
    decision = evaluate_residual_teacher_admission(
        _full_evidence(
            no_regression=_no_regression(
                math={"gate": "FAIL", "delta": -0.03, "regressed": True}
            )
        )
    )
    assert decision["verdict"] == VERDICT_REJECT
    row = _by_name(decision, "protected_no_regression")
    assert row["status"] == "FAIL"
    assert "math" in row["detail"]


def test_coding_tool_agentic_regression_rejects() -> None:
    for axis in ("coding", "tool", "agentic"):
        decision = evaluate_residual_teacher_admission(
            _full_evidence(no_regression=_no_regression(**{axis: {"delta": -0.01}}))
        )
        assert decision["verdict"] == VERDICT_REJECT, axis
        assert axis in _by_name(decision, "protected_no_regression")["detail"]


# ---------------------------------------------------------------------------
# Absent / non-positive incremental improvement â REJECT
# ---------------------------------------------------------------------------


def test_absent_incremental_improvement_rejects() -> None:
    decision = evaluate_residual_teacher_admission(
        _full_evidence(
            kimi_incremental_receipt=_kimi_ab(
                incremental_held_out_delta=None,
                comparison={"baseline": 0.5, "with_kimi": 0.5},  # delta computed 0
            )
        )
    )
    # delta 0 is not positive
    assert decision["verdict"] == VERDICT_REJECT
    assert _by_name(decision, "incremental_held_out_improvement")["status"] == "FAIL"


def test_missing_delta_structure_rejects() -> None:
    decision = evaluate_residual_teacher_admission(
        _full_evidence(
            kimi_incremental_receipt={
                "teacher": "kimi",
                "includes_kimi": True,
                "held_out_membership_hash": MEMBERSHIP,
                # no delta, no comparison
            }
        )
    )
    assert decision["verdict"] == VERDICT_REJECT
    row = _by_name(decision, "incremental_held_out_improvement")
    assert row["status"] == "FAIL"
    assert "absent" in row["detail"]


# ---------------------------------------------------------------------------
# Forward / architecture not ready â REJECT
# ---------------------------------------------------------------------------


def test_dsv4f_forward_not_ready_rejects() -> None:
    decision = evaluate_residual_teacher_admission(
        _full_evidence(
            dsv4f_architecture_forward={
                "ready": False,
                "status": "DEEPSEEK_FORWARD_PENDING",
            }
        )
    )
    assert decision["verdict"] == VERDICT_REJECT
    assert _by_name(decision, "dsv4f_architecture_forward_ready")["status"] == "FAIL"


# ---------------------------------------------------------------------------
# Generic hypothesis / unsealed baseline â REJECT
# ---------------------------------------------------------------------------


def test_generic_more_distillation_hypothesis_rejects() -> None:
    decision = evaluate_residual_teacher_admission(
        _full_evidence(
            residual_hypothesis={"name": "more distillation", "role": "residual_lane"}
        )
    )
    assert decision["verdict"] == VERDICT_REJECT
    assert _by_name(decision, "named_residual_hypothesis")["status"] == "FAIL"


def test_unsealed_glm_baseline_rejects() -> None:
    decision = evaluate_residual_teacher_admission(
        _full_evidence(
            glm_baseline_receipt={
                "teacher": "glm",
                "held_out_membership_hash": MEMBERSHIP,
                "sealed": False,
            }
        )
    )
    assert decision["verdict"] == VERDICT_REJECT
    assert _by_name(decision, "glm_only_baseline_receipt")["status"] == "FAIL"


def test_baseline_with_kimi_is_not_glm_only() -> None:
    decision = evaluate_residual_teacher_admission(
        _full_evidence(
            glm_baseline_receipt=_glm_baseline(teacher="kimi+glm", teachers=["glm", "kimi"])
        )
    )
    assert decision["verdict"] == VERDICT_REJECT
    assert _by_name(decision, "glm_only_baseline_receipt")["status"] == "FAIL"


# ---------------------------------------------------------------------------
# Valid admission
# ---------------------------------------------------------------------------


def test_valid_admission_when_all_requirements_pass() -> None:
    decision = evaluate_residual_teacher_admission(_full_evidence())
    verify(decision, label="residual-teacher-admit")
    assert decision["verdict"] == VERDICT_ADMIT
    assert all(c["status"] == "PASS" for c in decision["checks"])
    assert decision["claim_boundary"]["admit_is_not_causal_proof"] is True
    assert decision["claim_boundary"]["admit_is_residual_lane_only"] is True
    assert decision["claim_boundary"]["not_duplicate_full_glm_transfer"] is True
    assert set(decision["protected_axes"]) == set(PROTECTED_AXES)
    assert "causal" in decision["reason"] or "not causal" in decision["reason"]


def test_keyword_overrides_match_bundle() -> None:
    decision = evaluate_residual_teacher_admission(
        glm_baseline_receipt=_glm_baseline(),
        kimi_incremental_receipt=_kimi_ab(),
        residual_hypothesis=_hypothesis(),
        provenance=_provenance(),
        no_regression=_no_regression(),
        dsv4f_architecture_forward=_forward_ready(),
    )
    assert decision["verdict"] == VERDICT_ADMIT


def test_admit_via_comparison_scores_without_explicit_delta_field() -> None:
    decision = evaluate_residual_teacher_admission(
        _full_evidence(
            kimi_incremental_receipt={
                "teacher": "kimi",
                "held_out_membership_hash": MEMBERSHIP,
                "comparison": {"baseline": 0.4, "with_kimi": 0.45},
            }
        )
    )
    assert decision["verdict"] == VERDICT_ADMIT
    assert _by_name(decision, "incremental_held_out_improvement")["status"] == "PASS"


def test_missing_provenance_defers_not_admits() -> None:
    evidence = _full_evidence()
    del evidence["provenance"]
    decision = evaluate_residual_teacher_admission(evidence)
    assert decision["verdict"] == VERDICT_DEFERRED
    assert _by_name(decision, "provenance_revision_identity")["status"] == "DEFER"

"""MATH/FORMAL lab: real local bounty through H.2 with z3 as checker."""
from __future__ import annotations

import pytest

from tools.theia.engine import execute_lab
from tools.theia.intake import INTAKE_ORDER, IntakeStage
from tools.theia.labs import LabKind
from tools.theia.math_lab import (
    SCHEMA,
    apple_threadgroup_ceiling,
    check_claims,
    checker_status,
    run_math_bounty,
)
from tools.theia.security import ActiveTestRefused, SecurityMachine, SecurityState
from tools.theia.value import ScheduleScore, VerifiedResult, accept_as_verified


def test_z3_checker_is_available_and_used():
    status = checker_status()
    assert status["z3py"]["available"] is True
    assert status["z3py"]["used"] is True
    assert status["lean"]["used"] is False


def test_apple_ceiling_calls_static_kernel_verify_symbol():
    from tools.future.static_kernel_verify import APPLE_MAX_THREADS_PER_THREADGROUP

    assert apple_threadgroup_ceiling() == APPLE_MAX_THREADS_PER_THREADGROUP
    assert apple_threadgroup_ceiling() == 1024


def test_check_claims_calls_z3_and_holds():
    claims = check_claims()
    ids = [c["id"] for c in claims]
    assert "APPLE_TG_PRODUCT_NEVER_EXCEEDS_CEILING" in ids
    assert "POWER_OF_TWO_THREADGROUP_IS_NOT_ALWAYS_LEGAL" in ids
    assert "H1_VALUE_STRICTLY_DECREASES_IN_RISK" in ids
    assert all(c["holds"] for c in claims)
    assert all(c["evidence_tier"] == "STATIC" for c in claims)
    assert all(c["checker"] == "z3.Solver.check" for c in claims)
    counter = next(c for c in claims if c["kind"] == "counterexample")
    assert counter["model"]["product"] == 2048
    assert counter["archived_as_negative_science"] is True


def test_math_lab_runs_h2_end_to_end():
    result = run_math_bounty(write=True)
    assert result.exit_code == 0, result.notes
    assert result.lab == LabKind.MATH_FORMAL.value
    assert result.final_stage is IntakeStage.TRAJECTORY_METHOD_NEGATIVE_SCIENCE
    assert list(result.stages_visited) == [s.value for s in INTAKE_ORDER]
    assert result.blocked is None
    assert result.security_halt is None
    assert isinstance(result.schedule_score, ScheduleScore)
    assert result.schedule_score.to_json_dict()["declares_result_true"] is False
    assert isinstance(result.verified_result, VerifiedResult)
    accept_as_verified(result.verified_result)
    assert result.verified_result.detail["schema"] == SCHEMA
    assert result.verified_result.detail["independent_module"] == (
        "tools.theia.math_lab.check_claims"
    )
    assert result.verified_result.detail["checker"] == "z3.Solver.check"
    assert result.verified_result.detail["n_claims"] >= 4
    assert "ACTIVE_TEST" not in result.stages_visited


def test_physics_and_open_source_stay_declared():
    with pytest.raises(RuntimeError, match="declared, not executed"):
        execute_lab(LabKind.PHYSICS_QUANTUM)
    with pytest.raises(RuntimeError, match="declared, not executed"):
        execute_lab(LabKind.OPEN_SOURCE)


def test_math_lab_does_not_implement_active_test():
    m = SecurityMachine()
    with pytest.raises(ActiveTestRefused):
        m.advance(SecurityState.ACTIVE_TEST)
    with pytest.raises(RuntimeError, match="refusal-only"):
        execute_lab(LabKind.AUTHORIZED_SECURITY)

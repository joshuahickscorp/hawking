"""SYSTEMS/COMPILER lab: real local bounty against this repo's own code."""
from __future__ import annotations

import pytest

from tools.future._common import REPO
from tools.theia.engine import execute_lab
from tools.theia.intake import INTAKE_ORDER, IntakeStage
from tools.theia.labs import LabKind
from tools.theia.security import ActiveTestRefused, SecurityMachine, SecurityState
from tools.theia.systems_lab import (
    SCHEMA,
    apple_threadgroup_ceiling,
    geometry_identities,
    parse_repo_shaders,
)
from tools.theia.value import ScheduleScore, VerifiedResult, accept_as_verified


def test_parse_metal_is_called_on_this_repo_shaders():
    from tools.future.static_kernel_verify import load_repo_sources, parse_metal

    metal, rust, _membership = load_repo_sources(REPO)
    assert metal, "crates/hawking-core/shaders must be on disk"
    parsed = parse_repo_shaders(metal)
    assert parsed["symbol"] == "tools.future.static_kernel_verify.parse_metal"
    assert parsed["n_kernels"] >= 1
    sample_path = next(iter(metal))
    ks, _sts = parse_metal(metal[sample_path], sample_path)
    assert parsed["kernels_by_file"][sample_path] == len(ks)
    assert rust, "crates/hawking-core rust hosts must be on disk"


def test_geometry_identities_are_static_and_hold():
    geo = geometry_identities()
    assert geo["evidence_tier"] == "STATIC"
    assert geo["not_a_hardware_occupancy_counter"] is True
    assert geo["occupancy_identity"]["holds"] is True
    assert geo["n_hold"] == geo["n_organs"]
    assert geo["n_organs"] == 11
    assert geo["apple_ceiling"] == apple_threadgroup_ceiling() == 1024
    assert geo["under_apple_ceiling"] is True


def test_systems_lab_runs_h2_end_to_end():
    """Calls execute_lab, which calls analyze/parse_metal/load_repo_sources."""
    result = execute_lab(LabKind.SYSTEMS_COMPILER)
    assert result.exit_code == 0, result.notes
    assert result.lab == LabKind.SYSTEMS_COMPILER.value
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
        "tools.future.static_kernel_verify.parse_metal"
    )
    assert result.verified_result.detail["analyze_symbol_was"] == (
        "tools.future.static_kernel_verify.analyze"
    )
    assert result.verified_result.detail["n_kernels"] >= 1
    assert result.verified_result.detail["n_geometry_hold"] == 11
    assert "ACTIVE_TEST" not in result.stages_visited
    assert result.notes["authorization"]["status"] == "HAWKING_INTERNAL"


def test_security_lab_still_cannot_take_active_test():
    m = SecurityMachine()
    with pytest.raises(ActiveTestRefused):
        m.advance(SecurityState.ACTIVE_TEST)
    with pytest.raises(RuntimeError, match="ACTIVE_TEST is unimplemented"):
        execute_lab(LabKind.AUTHORIZED_SECURITY)

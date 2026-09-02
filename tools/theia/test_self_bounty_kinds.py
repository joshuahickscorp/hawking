"""Each §19.12 self-bounty kind runs a real local receipt through H.2."""
from __future__ import annotations

import pytest

from tools.theia.engine import run_self_bounty_kind
from tools.theia.intake import INTAKE_ORDER, IntakeStage
from tools.theia.labs import LabKind, SelfBountyKind
from tools.theia.self_bounty import KIND_RECEIPTS, receipt_for_kind
from tools.theia.value import ScheduleScore, VerifiedResult, accept_as_verified


@pytest.mark.parametrize("kind", list(SelfBountyKind))
def test_every_self_bounty_kind_has_a_real_receipt(kind):
    path = receipt_for_kind(kind)
    assert path.is_file(), f"missing {KIND_RECEIPTS[kind]}"
    assert path.name == KIND_RECEIPTS[kind]


@pytest.mark.parametrize("kind", list(SelfBountyKind))
def test_every_self_bounty_kind_runs_h2_end_to_end(kind):
    result = run_self_bounty_kind(kind)
    assert result.exit_code == 0, result.notes
    assert result.lab == LabKind.HAWKING_SELF_BOUNTY.value
    assert result.self_bounty_kind == kind.value
    assert result.final_stage is IntakeStage.TRAJECTORY_METHOD_NEGATIVE_SCIENCE
    assert list(result.stages_visited) == [s.value for s in INTAKE_ORDER]
    assert result.blocked is None
    assert result.security_halt is None
    assert isinstance(result.schedule_score, ScheduleScore)
    assert result.schedule_score.value > 0
    assert isinstance(result.verified_result, VerifiedResult)
    accept_as_verified(result.verified_result)
    assert result.verified_result.detail.get("independent_module")
    assert "ACTIVE_TEST" not in result.stages_visited

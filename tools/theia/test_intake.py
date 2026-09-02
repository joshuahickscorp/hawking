"""Round-trip a Bounty through H.2 using a REAL receipt, not a fixture."""
from __future__ import annotations

import json
from pathlib import Path

from tools.future._common import REPO
from tools.theia.bounty import (
    Bounty,
    BountyClass,
    Budget,
    PublicOrPrivate,
    unpinned_scope,
)
from tools.theia.engine import DEFAULT_SELF_BOUNTY, run
from tools.theia.intake import INTAKE_ORDER, IntakeStage, run_intake
from tools.theia.labs import LabKind, SelfBountyKind
from tools.theia.self_bounty import bounty_from_receipt
from tools.theia.value import ScheduleScore, VerifiedResult, accept_as_verified

REAL_RECEIPT = REPO / "receipts" / "future" / "AUTONOMY_SCARS.json"


def test_named_receipt_is_the_real_autonomy_scars_artifact():
    assert REAL_RECEIPT == DEFAULT_SELF_BOUNTY
    assert REAL_RECEIPT.is_file()
    doc = json.loads(REAL_RECEIPT.read_text())
    assert doc["schema"] == "hawking.future.autonomy_scars.v1"
    assert doc["n_scars"] == 4


def test_round_trip_real_autonomy_scars_receipt_through_h2_intake():
    """Uses receipts/future/AUTONOMY_SCARS.json — a real Hawking artifact."""
    bounty, kind, _doc = bounty_from_receipt(REAL_RECEIPT)
    dumped = bounty.to_json_dict()
    assert Bounty.from_json_dict(dumped) == bounty
    result = run(REAL_RECEIPT)
    assert result.exit_code == 0
    assert result.source == str(REAL_RECEIPT.resolve())
    assert result.final_stage is IntakeStage.TRAJECTORY_METHOD_NEGATIVE_SCIENCE
    assert list(result.stages_visited) == [s.value for s in INTAKE_ORDER]
    assert result.self_bounty_kind == SelfBountyKind.NEGATIVE_SCIENCE.value
    assert result.lab == LabKind.HAWKING_SELF_BOUNTY.value
    assert result.blocked is None
    assert result.security_halt is None
    assert isinstance(result.schedule_score, ScheduleScore)
    assert result.schedule_score.value.numerator == 16
    assert result.schedule_score.value.denominator == 1
    assert isinstance(result.verified_result, VerifiedResult)
    accept_as_verified(result.verified_result)
    assert result.verified_result.detail["n_scars"] == 4
    assert kind is SelfBountyKind.NEGATIVE_SCIENCE
    assert bounty.bounty_class is BountyClass.HAWKING_INTERNAL_SELF_BOUNTY


def test_security_class_without_authority_is_blocked_rights(tmp_path):
    src = tmp_path / "note.json"
    src.write_text(json.dumps({"schema": "not-used", "seal_sha256": "x"}))
    bounty = Bounty(
        id="sec-unpinned",
        source=str(src),
        domain="security",
        question_or_target="allowed_targets: lab.example.local; in scope",
        monetary_reward=None,
        nonmonetary_value="none",
        authorization_scope=unpinned_scope(),
        rules=("follow program",),
        evidence_required=("program report",),
        verifier="tools.theia.intake.verify_receipt",
        budget=Budget(workunits=1),
        deadline=None,
        public_or_private=PublicOrPrivate.PRIVATE,
        submission_policy="private",
        success_conditions=("in-scope report",),
        stop_conditions=("BLOCKED_RIGHTS",),
        bounty_class=BountyClass.AUTHORIZED_BUG_BOUNTY_PROGRAM,
        lab=LabKind.AUTHORIZED_SECURITY.value,
    )
    result = run_intake(bounty, value_inputs_factory=lambda p: None)
    assert result.blocked is not None
    assert result.blocked.status == "BLOCKED_RIGHTS"
    assert result.blocked.reason == "unpinned"
    assert result.final_stage is IntakeStage.AUTHORITY_SCOPE_RESOLUTION
    assert IntakeStage.CHEAP_REPRODUCTION_SCREEN.value not in result.stages_visited

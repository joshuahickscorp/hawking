"""In-scope security bounty still cannot take ACTIVE_TEST."""
from __future__ import annotations

import json
from pathlib import Path

from tools.theia.authority import AUTHORITY_SCHEMA
from tools.theia.bounty import (
    Bounty,
    BountyClass,
    Budget,
    PublicOrPrivate,
    AuthorizationScope,
)
from tools.theia.intake import IntakeStage, run_intake
from tools.theia.labs import LabKind
from tools.theia.security import LAST_LEGAL_STATE


def test_in_scope_security_bounty_halts_before_active_test(tmp_path):
    src = tmp_path / "program.json"
    src.write_text(json.dumps({"schema": "unused", "seal_sha256": "x"}))
    authority = tmp_path / "authority.json"
    authority.write_text(
        json.dumps(
            {
                "schema": AUTHORITY_SCHEMA,
                "program_id": "test-program",
                "allowed_targets": ["lab.example.local"],
                "forbidden_test_classes": ["ACTIVE_TEST"],
                "operator": "pytest-operator",
            }
        )
    )
    bounty = Bounty(
        id="sec-in-scope",
        source=str(src),
        domain="security",
        question_or_target="please test lab.example.local",
        monetary_reward=None,
        nonmonetary_value="none",
        authorization_scope=AuthorizationScope(
            kind="PROGRAM", program_id="test-program"
        ),
        rules=("follow program",),
        evidence_required=("program report",),
        verifier="tools.theia.intake.verify_receipt",
        budget=Budget(workunits=1),
        deadline=None,
        public_or_private=PublicOrPrivate.PRIVATE,
        submission_policy="private",
        success_conditions=("in-scope report",),
        stop_conditions=("BLOCKED_RIGHTS", "ACTIVE_TEST"),
        bounty_class=BountyClass.AUTHORIZED_BUG_BOUNTY_PROGRAM,
        lab=LabKind.AUTHORIZED_SECURITY.value,
    )
    result = run_intake(
        bounty,
        value_inputs_factory=lambda p: None,
        authority_file=authority,
        declared_target="lab.example.local",
    )
    assert result.blocked is None
    assert result.security_halt is not None
    assert result.security_halt.status == "HALTED_BEFORE_ACTIVE_TEST"
    assert result.security_halt.last_legal_h3_state == LAST_LEGAL_STATE.value
    assert result.final_stage is IntakeStage.AUTHORITY_SCOPE_RESOLUTION
    assert IntakeStage.CHEAP_REPRODUCTION_SCREEN.value not in result.stages_visited
    assert result.schedule_score is None

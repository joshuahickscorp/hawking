"""Live execution path: ingest a real receipt, score, stage, exit 0."""
from __future__ import annotations

import json

from tools.future._common import RECEIPTS
from tools.theia.engine import (
    DEFAULT_SELF_BOUNTY,
    RECEIPT_NAME,
    main,
    run,
    write_engine_receipt,
)
from tools.theia.intake import IntakeStage
from tools.theia.labs import LabKind, SelfBountyKind


def test_engine_ingests_autonomy_scars_and_writes_receipt():
    result = run(DEFAULT_SELF_BOUNTY)
    assert result.exit_code == 0
    assert result.final_stage is IntakeStage.TRAJECTORY_METHOD_NEGATIVE_SCIENCE
    assert result.self_bounty_kind == SelfBountyKind.NEGATIVE_SCIENCE.value
    assert result.lab == LabKind.HAWKING_SELF_BOUNTY.value
    assert result.schedule_score is not None
    assert result.verified_result is not None
    out = write_engine_receipt(result)
    assert out == RECEIPTS / RECEIPT_NAME
    doc = json.loads(out.read_text())
    assert doc["schema"] == "hawking.theia.bounty_engine.v1"
    assert doc["halves"]["bounty_engine"]["status"] == "LIVE"
    assert doc["halves"]["model_ladder"]["status"] == "BLOCKED_EXTERNAL"
    assert doc["security"]["network_egress"] is False
    assert doc["security"]["active_test"] is False
    assert doc["self_bounty_run"]["source"].endswith("AUTONOMY_SCARS.json")
    assert doc["self_bounty_run"]["exit_code"] == 0


def test_cli_exits_0_on_real_receipt():
    code = main(["--self-bounty", str(DEFAULT_SELF_BOUNTY)])
    assert code == 0

"""Additive Odyssey candidate-board and active-owner guard tests."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import odyssey_ctl as ctl


def _override(path: Path, specimen: str) -> None:
    path.write_text(json.dumps({
        "preserved_evidence": {
            "flash_next": f"{specimen} remains the pristine-source Pulsar/Resident Star candidate",
        },
    }))


def _state():
    return {
        "schema": ctl.SCHEMA,
        "legacy_marker": {"must_survive": True},
        "patients": [{"oxx": "O099", "source": "Example/Model", "state": "READY"}],
        "work": [{"id": "legacy", "status": "VERIFIED"}],
    }


def test_board_adds_sealed_flash_owner_and_preserves_prior_rows(tmp_path):
    flash = "Qwen--Qwen3.8-Flash-Next@34567a4712bc"
    candidate = "LiquidAI--LFM2.5-1.2B-Instruct@0f604ada3f76"
    override = tmp_path / "override.json"
    _override(override, flash)
    state = _state()
    state["candidate_board"] = [{
        "specimen": candidate,
        "remote_evidence": [{"status": "OBSERVED"}],
    }]

    board = ctl.candidate_board(
        state,
        dispositions={
            flash: {"disposition": "RESEARCH SPECIMEN"},
            candidate: {"disposition": "PULSAR CANDIDATE"},
        },
        override_path=override,
    )
    by_name = {row["specimen"]: row for row in board}
    assert by_name[flash]["active_owner"] == ctl.FLASH_ACTIVE_OWNER
    assert by_name[flash]["owner_status"] == "SEALED"
    assert by_name[candidate]["active_owner"] == ctl.ACTIVE_OWNER_UNREGISTERED
    assert by_name[candidate]["remote_evidence"] == [{"status": "OBSERVED"}]


def test_owner_check_refuses_flash_collision_and_unregistered_deep_work(tmp_path):
    flash = "Qwen--Qwen3.8-Flash-Next@34567a4712bc"
    candidate = "Qwen--Qwen3-4B-Instruct-2507@cdbee75f17c0"
    override = tmp_path / "override.json"
    _override(override, flash)
    state = _state()
    dispositions = {
        flash: {"disposition": "RESEARCH SPECIMEN"},
        candidate: {"disposition": "PULSAR CANDIDATE"},
    }
    # The function's default override is production-only; use the derived
    # board explicitly here to exercise the same sealed owner fact in isolation.
    board = ctl.candidate_board(state, dispositions=dispositions, override_path=override)
    state["candidate_board"] = board

    flash_check = ctl.active_owner_check(flash, state=state, override_path=override)
    assert flash_check["deep_work_allowed"] is False
    assert flash_check["reason"] == "ACTIVE_OWNER_COLLISION"
    assert flash_check["active_owner"] == ctl.FLASH_ACTIVE_OWNER

    candidate_check = ctl.active_owner_check(
        "Qwen--Qwen3-4B-Instruct-2507@cdbee75f17c0",
        state=state,
        override_path=override,
    )
    assert candidate_check["deep_work_allowed"] is False
    assert candidate_check["reason"] == "ACTIVE_OWNER_UNREGISTERED"


def test_claim_is_additive_idempotent_and_does_not_write_injected_state(tmp_path):
    specimen = "LiquidAI--LFM2.5-1.2B-Instruct@0f604ada3f76"
    state = _state()
    state["candidate_board"] = ctl.candidate_board(
        state,
        dispositions={specimen: {"disposition": "PULSAR CANDIDATE"}},
    )
    real_state_path = ctl.STATE
    try:
        ctl.STATE = tmp_path / "must-not-be-created.json"
        first = ctl.claim_active_owner(specimen, "odyssey_remote", confirm=True, state=state)
        second = ctl.claim_active_owner(specimen, "odyssey_remote", confirm=True, state=state)
    finally:
        ctl.STATE = real_state_path

    assert first["changed"] is True
    assert second["idempotent"] is True
    assert len(state["active_owner_history"]) == 1
    assert not (tmp_path / "must-not-be-created.json").exists()
    assert state["legacy_marker"] == {"must_survive": True}
    assert ctl.active_owner_check(specimen, state=state)["deep_work_allowed"] is True


def test_sealed_owner_cannot_be_replaced_by_remote_claim(tmp_path):
    flash = "Qwen--Qwen3.8-Flash-Next@34567a4712bc"
    override = tmp_path / "override.json"
    _override(override, flash)
    state = _state()
    state["candidate_board"] = ctl.candidate_board(
        state,
        dispositions={flash: {"disposition": "RESEARCH SPECIMEN"}},
        override_path=override,
    )
    # The production helper reads the sealed override path; this test directly
    # verifies the collision behavior using the same derived board.
    assert state["candidate_board"][0]["active_owner"] == ctl.FLASH_ACTIVE_OWNER
    with pytest.raises(PermissionError, match="active owner collision"):
        ctl.claim_active_owner(
            flash,
            "odyssey_remote",
            confirm=True,
            state=state,
            override_path=override,
        )


def test_refresh_preserves_existing_state_keys_and_adds_owner_fields(tmp_path, monkeypatch):
    state_path = tmp_path / "ODYSSEY_STATE.json"
    monkeypatch.setattr(ctl, "STATE", state_path)
    ctl.save_state(_state())
    specimen = "Qwen--Qwen3-4B-Instruct-2507@cdbee75f17c0"
    monkeypatch.setattr(
        ctl,
        "MODELLAKE_DISPOSITIONS",
        tmp_path / "dispositions.json",
    )
    ctl.MODELLAKE_DISPOSITIONS.write_text(json.dumps({
        "dispositions": {specimen: {"disposition": "PULSAR CANDIDATE"}},
    }))

    payload = ctl.refresh_candidate_board(persist=True)
    saved = json.loads(state_path.read_text())
    assert payload["candidate_count"] == 1
    assert saved["legacy_marker"] == {"must_survive": True}
    assert saved["work"] == [{"id": "legacy", "status": "VERIFIED"}]
    assert saved["patients"][0]["active_owner"] == ctl.ACTIVE_OWNER_UNREGISTERED
    assert saved["candidate_board"][0]["specimen"] == specimen


def test_record_evidence_updates_only_the_observed_tier(tmp_path):
    specimen = "Qwen--Qwen3-14B@40c069824f42"
    state = _state()
    state["candidate_board"] = ctl.candidate_board(
        state,
        dispositions={specimen: {"disposition": "PARETO CANDIDATE"}},
    )
    ctl.claim_active_owner(specimen, "odyssey_remote", confirm=True, state=state)
    recorded = ctl.record_candidate_evidence(
        specimen,
        {
            "provider": "openrouter",
            "model": "openai/gpt-5.6-luna",
            "usage": {"total_tokens": 3, "cost_usd": 0.001},
            "tool_trace": [{"tool": "fs.read", "dispatched": True, "ok": True}],
            "observations": ["sealed identity observed"],
        },
        owner="odyssey_remote",
        tier="TIER0",
        confirm=True,
        state=state,
    )
    row = next(item for item in state["candidate_board"] if item["specimen"] == specimen)
    assert recorded["status"] == "OBSERVED"
    assert row["tier_state"]["tier0"] == "OBSERVED"
    assert row["tier_state"]["tier1"] == "PENDING"
    assert len(row["remote_evidence"]) == 1
    assert not (tmp_path / "ODYSSEY_STATE.json").exists()

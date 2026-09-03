"""Structural tests for Ascension platform / profiler / blocked-state contracts.

Bible §1, §11, §§29–31. No live Qwen work; no performance claims.
"""

from __future__ import annotations

import copy
import json

import pytest

from lab.operators.ascension_contracts import (
    BLOCKED_REGISTRY_PATH,
    MODEL_LADDER_PIPELINE,
    PLATFORM_DECISION_PATH,
    PROFILER_CONTRACT_PATH,
    REQUIRED_BLOCKER_FIELDS,
    REQUIRED_BOOTSTRAP_DISPLAY_NAMES,
    REQUIRED_BOOTSTRAP_ENTRY_IDS,
    AscensionContractError,
    entry_is_fully_blocked,
    load_blocked_state_registry,
    load_platform_decision,
    load_profiler_contract,
    validate_all_ascension_contracts,
    validate_blocked_state_registry,
    validate_overview_references_contracts,
    validate_platform_decision,
    validate_profiler_contract,
    validate_schedule_references_contracts,
)


def test_contract_files_exist_and_parse() -> None:
    for path in (
        PLATFORM_DECISION_PATH,
        PROFILER_CONTRACT_PATH,
        BLOCKED_REGISTRY_PATH,
    ):
        assert path.is_file(), path
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert "schema" in data


def test_platform_decision_validates() -> None:
    doc = validate_platform_decision()
    assert doc["stance"]["apple_first"] is True
    assert doc["stance"]["cuda_deferred"] is True
    assert doc["stance"]["cuda_rejected"] is False
    assert "lowest_common_denominator_kernel_mandate" in doc["forbidden"]
    assert doc["honesty"]["stage_ready"] is False
    assert doc["implementation_status"] == "NOT_STARTED"


def test_platform_decision_refuses_ready_claim() -> None:
    bad = load_platform_decision()
    bad["honesty"]["stage_ready"] = True
    with pytest.raises(AscensionContractError, match="stage_ready"):
        validate_platform_decision(bad)


def test_platform_decision_refuses_lcd_omission() -> None:
    bad = load_platform_decision()
    bad["forbidden"] = [x for x in bad["forbidden"] if "lowest" not in x]
    with pytest.raises(AscensionContractError, match="lowest_common_denominator"):
        validate_platform_decision(bad)


def test_profiler_contract_validates_and_denies_live_qwen_values() -> None:
    doc = validate_profiler_contract()
    assert doc["target_explained_percent"] == 98.0
    assert doc["live_qwen_values"]["exist"] is False
    assert doc["live_qwen_values"]["qwen3_coder_30b"] is None
    assert doc["live_qwen_values"]["qwen3_coder_next_80b"] is None
    assert doc["honesty"]["performance_claims"] is False
    assert doc["honesty"]["live_qwen_profile"] is False
    assert "PEAK_UTILIZATION" in doc["required_global_ledgers"]
    assert "FLOPS_PER_TOKEN" in doc["required_global_ledgers"]
    refusal_ids = {r["id"] for r in doc["refusal_conditions"]}
    assert "REFUSE_UNEXPLAINED_OTHER_ABOVE_TARGET" in refusal_ids
    assert "REFUSE_LIVE_QWEN_CLAIM_WITHOUT_RUNTIME" in refusal_ids
    assert "REFUSE_HIGHER_FLOPS_AS_AUTOMATIC_WIN" in refusal_ids


def test_profiler_contract_refuses_fabricated_qwen_values() -> None:
    bad = load_profiler_contract()
    bad["live_qwen_values"]["exist"] = True
    bad["live_qwen_values"]["qwen3_coder_30b"] = {"fake_tps": 999}
    with pytest.raises(AscensionContractError):
        validate_profiler_contract(bad)


def test_blocked_registry_names_both_qwen_models_blocked() -> None:
    doc = validate_blocked_state_registry()
    by_id = {e["id"]: e for e in doc["entries"]}
    assert set(REQUIRED_BOOTSTRAP_ENTRY_IDS) <= set(by_id)
    names = {e["display_name"] for e in doc["entries"]}
    for name in REQUIRED_BOOTSTRAP_DISPLAY_NAMES:
        assert name in names
    for entry_id in REQUIRED_BOOTSTRAP_ENTRY_IDS:
        entry = by_id[entry_id]
        assert entry["status"] == "BLOCKED"
        assert entry_is_fully_blocked(entry)
        assert tuple(doc["required_blocker_fields"]) == REQUIRED_BLOCKER_FIELDS
        for field in REQUIRED_BLOCKER_FIELDS:
            assert entry["blockers"][field]["status"] == "BLOCKED"
            assert entry["blockers"][field]["evidence_refs"] == []
            assert entry["blockers"][field]["cleared_at"] is None
            assert entry["blockers"][field]["cleared_by"] is None
    assert by_id["qwen3_coder_30b"]["display_name"] == "Qwen3-Coder-30B"
    assert by_id["qwen3_coder_next_80b"]["display_name"] == "Qwen3-Coder-Next-80B"
    assert by_id["qwen3_coder_30b"]["family"] == "QWEN3_MOE"
    assert by_id["qwen3_coder_next_80b"]["family"] == "QWEN3_NEXT"


def test_blocked_registry_pipeline_and_families() -> None:
    doc = validate_blocked_state_registry()
    assert tuple(doc["model_ladder_pipeline"]["phases"]) == MODEL_LADDER_PIPELINE
    families = doc["family_kernel_architecture"]["initial_families"]
    assert "QWEN3_MOE" in families
    assert "QWEN3_NEXT" in families
    assert "STATE_SPACE_HYBRID" in families
    assert "A" in doc["rotation_rule"]
    assert "B" in doc["rotation_rule"]
    assert "TG3" in doc["rotation_rule"]
    assert doc["honesty"]["any_bootstrap_model_unblocked"] is False
    assert doc["honesty"]["stage_ready"] is False


def test_blocked_registry_refuses_cleared_blocker_without_evidence_shape() -> None:
    bad = load_blocked_state_registry()
    entry = next(e for e in bad["entries"] if e["id"] == "qwen3_coder_30b")
    entry["blockers"]["tg3_approval"]["status"] = "CONTROLLER_CERTIFIED"
    entry["blockers"]["tg3_approval"]["cleared_by"] = "sandbox"
    with pytest.raises(AscensionContractError, match="must be BLOCKED"):
        validate_blocked_state_registry(bad)


def test_blocked_registry_refuses_missing_blocker_field() -> None:
    bad = load_blocked_state_registry()
    entry = next(e for e in bad["entries"] if e["id"] == "qwen3_coder_next_80b")
    del entry["blockers"]["profiler_evidence"]
    with pytest.raises(AscensionContractError, match="profiler_evidence"):
        validate_blocked_state_registry(bad)


def test_blocked_registry_refuses_unblocking_bootstrap_honesty() -> None:
    bad = load_blocked_state_registry()
    bad["honesty"]["any_bootstrap_model_unblocked"] = True
    with pytest.raises(AscensionContractError, match="any_bootstrap_model_unblocked"):
        validate_blocked_state_registry(bad)


def test_schedule_and_overview_reference_contracts() -> None:
    validate_schedule_references_contracts()
    validate_overview_references_contracts()


def test_validate_all_returns_honest_summary() -> None:
    summary = validate_all_ascension_contracts()
    assert summary["ok"] is True
    assert summary["live_qwen_values_exist"] is False
    assert summary["any_stage_ready"] is False
    assert summary["bootstrap_statuses"]["qwen3_coder_30b"] == "BLOCKED"
    assert summary["bootstrap_statuses"]["qwen3_coder_next_80b"] == "BLOCKED"


def test_tampered_schedule_missing_profiler_ref_fails() -> None:
    from lab.operators.ascension_contracts import load_master_schedule

    bad = load_master_schedule()
    for step in bad["steps"]:
        if step.get("id") == 13:
            step["companion_docs"] = [
                d
                for d in step.get("companion_docs", [])
                if "PROFILER" not in d and "COMPLETE_TOKEN" not in d
            ]
    with pytest.raises(AscensionContractError, match="profiler"):
        validate_schedule_references_contracts(bad)


def test_entry_is_fully_blocked_helper() -> None:
    reg = load_blocked_state_registry()
    entry = next(e for e in reg["entries"] if e["id"] == "qwen3_coder_30b")
    assert entry_is_fully_blocked(entry) is True
    mutated = copy.deepcopy(entry)
    mutated["status"] = "IN_PROGRESS"
    assert entry_is_fully_blocked(mutated) is False

from __future__ import annotations

from hcli.agentos.flash_science import (
    GRAVITY_LADDER,
    _accelerator_primitive_plan,
    _gravity_science_plan,
    _three_zero_questions,
)


def test_flash_gravity_ladder_is_complete_and_ordered():
    plan = _gravity_science_plan()
    assert plan["status"] == "PLAN_ONLY"
    assert plan["ladder"] == list(GRAVITY_LADDER)
    assert [row["stage"] for row in plan["stages"]] == list(GRAVITY_LADDER)
    assert [row["order"] for row in plan["stages"]] == list(range(1, len(GRAVITY_LADDER) + 1))
    assert all(row["status"] == "PLAN_ONLY" for row in plan["stages"])
    assert all(row["source_mutation_allowed"] is False for row in plan["stages"])


def test_flash_three_zero_questions_are_explicitly_unproven():
    questions = _three_zero_questions()
    assert set(questions) >= {"storage", "independent_information", "execution"}
    assert questions["status"] == "UNRESOLVED_PLAN"
    assert all(questions[key]["status"] == "NOT_PROVEN" for key in ("storage", "independent_information", "execution"))
    assert all(questions[key]["evidence_required"] for key in ("storage", "independent_information", "execution"))


def test_flash_accelerator_plan_names_capability_and_gap_for_each_candidate():
    plan = _accelerator_primitive_plan()
    entries = plan["entries"]
    assert plan["status"] == "PLAN_ONLY"
    assert plan["physical_execution_claim"] is False
    assert plan["candidate_classes"] == [
        "low-bit GEMV",
        "expert routing",
        "fused route/gather",
        "expert execution",
        "DeltaNet scan/state",
        "sparse attention",
        "MTP",
        "norms",
        "epilogues",
        "persistent state",
        "expert residency scheduling",
    ]
    assert len(entries) >= 15
    assert all(entry["status"] == "PLAN_ONLY" for entry in entries)
    assert all(entry["existing_capability"] and entry["gap"] for entry in entries)
    names = {entry["primitive"] for entry in entries}
    assert {"native_nf_expert_gemv", "router_topk_gather", "persistent_deltanet_state_update", "ngram_lookup_generator", "qsa_sparse_indexer_kv_gather", "mtp_accept_reject_rollback"} <= names


def test_flash_plan_does_not_claim_a_physical_accelerator():
    plan = _accelerator_primitive_plan()
    assert plan["physical_execution_claim"] is False
    assert "physical" in plan["claim_boundary"].lower()

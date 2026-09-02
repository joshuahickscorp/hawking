from __future__ import annotations

import json

from hcli.agentos.flash_executable import (
    _native_expert_composition_summary,
    _native_gate_up_swiglu_summary,
    _native_router_selection_summary,
    _native_routed_expert_dispatch_summary,
    _native_shared_expert_composition_summary,
    _native_shared_residual_hyperconnection_summary,
)
from hcli.agentos.handoff import _model_lake_summary
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


def test_native_router_selection_summary_preserves_bounded_claim_boundary(tmp_path):
    receipt = tmp_path / "native-router-selection.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "hawking.flash_noetic_router_selection_native.v1",
                "nomenclature_version": "HAWKING_NOMENCLATURE_V1",
                "status": "PASSED",
                "semantic_type": "NoeticExecutableCandidate",
                "compiler_stage": "HawkingAccelerator",
                "qualification": "BOUNDED_NATIVE_ROUTER_SELECTION",
                "repo": "Qwen/Qwen3.8-Flash-Next",
                "pinned_revision": "revision",
                "root": "/lake/specimen",
                "selection": {"expert_ids": [1, 2]},
                "native_selection_execution_observed": True,
                "native_loader": {"source_independent_execution": True},
                "source_selection_parity": {
                    "status": "MISMATCH",
                    "expert_ids_exact_match": False,
                },
                "native_source_authority_kernel": {
                    "kernel": "gemv_native_bf16_seq",
                    "source_payload_exact": True,
                    "source_guard_unchanged": True,
                },
                "native_source_authority_execution_observed": True,
                "source_payload_exact": True,
                "source_guard_unchanged": True,
                "source_native_selection": {"expert_ids": [1, 2]},
                "source_reference_parity": {
                    "status": "MATCH",
                    "expert_ids_exact_match": True,
                },
                "source_native_parity": {"within_tolerance": True},
                "whole_model_capability": "NOT_TESTED",
                "complete_token_runtime": "NOT_TESTED",
                "promotion_allowed": False,
                "physical_graph": {"fingerprint": "graph"},
                "claim_boundary": "bounded only",
                "next_action": "continue",
            }
        ),
        encoding="utf-8",
    )

    summary = _native_router_selection_summary(tmp_path, receipt)

    assert summary["status"] == "PASSED"
    assert summary["selection_status"] == "EXECUTED"
    assert summary["source_independent_execution"] is True
    assert summary["native_selection_execution_observed"] is True
    assert summary["source_selection_parity"]["status"] == "MISMATCH"
    assert summary["native_source_authority_execution_observed"] is True
    assert summary["native_source_authority_kernel"]["kernel"] == "gemv_native_bf16_seq"
    assert summary["source_native_selection"]["expert_ids"] == [1, 2]
    assert summary["source_reference_parity"]["status"] == "MATCH"
    assert summary["source_native_parity"]["within_tolerance"] is True
    assert summary["source_payload_exact"] is True
    assert summary["source_guard_unchanged"] is True
    assert summary["promotion_allowed"] is False


def test_native_routed_expert_dispatch_summary_preserves_bounded_claim_boundary(tmp_path):
    receipt = tmp_path / "native-routed-expert-dispatch.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "hawking.flash_noetic_routed_expert_dispatch_native.v1",
                "nomenclature_version": "HAWKING_NOMENCLATURE_V1",
                "status": "PASSED",
                "semantic_type": "NoeticExecutableCandidate",
                "compiler_stage": "HawkingAccelerator",
                "qualification": "BOUNDED_NATIVE_ROUTED_EXPERT_BODY_DISPATCH",
                "router_receipt": "router.json",
                "campaign_receipt": "campaign.json",
                "selection": {"expert_ids": [1, 2]},
                "source_selection_parity": {"status": "MISMATCH", "expert_ids_exact_match": False},
                "components": [
                    {"candidate_body": {"source_independent": True}},
                    {"candidate_body": {"source_independent": True}},
                ],
                "execution": {"selected_expert_count": 2, "dispatches_per_route": 2},
                "native_routed_body_dispatch_observed": True,
                "whole_model_capability": "NOT_TESTED",
                "complete_expert_runtime": "NOT_TESTED",
                "complete_token_runtime": "NOT_TESTED",
                "promotion_allowed": False,
                "physical_graph": {"fingerprint": "graph"},
                "claim_boundary": "bounded selected-body dispatch only",
                "next_action": "continue",
            }
        ),
        encoding="utf-8",
    )

    summary = _native_routed_expert_dispatch_summary(tmp_path, receipt)

    assert summary["status"] == "PASSED"
    assert summary["native_routed_body_dispatch_observed"] is True
    assert summary["source_independent_execution"] is True
    assert summary["source_selection_parity"]["status"] == "MISMATCH"
    assert summary["whole_model_capability"] == "NOT_TESTED"
    assert summary["complete_expert_runtime"] == "NOT_TESTED"
    assert summary["promotion_allowed"] is False


def test_native_gate_up_swiglu_summary_preserves_activation_boundary(tmp_path):
    receipt = tmp_path / "native-gate-up-swiglu.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "hawking.flash_noetic_routed_expert_gate_up_swiglu_native.v1",
                "nomenclature_version": "HAWKING_NOMENCLATURE_V1",
                "status": "PASSED",
                "semantic_type": "NoeticExecutableCandidate",
                "compiler_stage": "HawkingAccelerator",
                "qualification": "BOUNDED_NATIVE_ROUTED_EXPERT_GATE_UP_SWIGLU_ACTIVATION",
                "selection": {"expert_ids": [1, 2]},
                "source_selection_parity": {"status": "MISMATCH", "expert_ids_exact_match": False},
                "components": [{"candidate_body": {"source_independent": True}}],
                "execution": {"selected_expert_count": 2, "native_gate_up_swiglu_observed": True},
                "native_gate_up_swiglu_observed": True,
                "native_expert_gate_up_activation_observed": True,
                "noetic_ir": {"source_independent": True},
                "whole_model_capability": "NOT_TESTED",
                "complete_expert_runtime": "NOT_TESTED",
                "complete_token_runtime": "NOT_TESTED",
                "promotion_allowed": False,
                "physical_graph": {"fingerprint": "graph"},
                "claim_boundary": "gate/up only",
                "next_action": "continue",
            }
        ),
        encoding="utf-8",
    )

    summary = _native_gate_up_swiglu_summary(tmp_path, receipt)

    assert summary["status"] == "PASSED"
    assert summary["native_gate_up_swiglu_observed"] is True
    assert summary["native_expert_gate_up_activation_observed"] is True
    assert summary["source_independent_execution"] is True
    assert summary["complete_expert_runtime"] == "NOT_TESTED"
    assert summary["promotion_allowed"] is False


def test_native_expert_composition_summary_preserves_device_chain_boundary(tmp_path):
    receipt = tmp_path / "native-expert-composition.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "hawking.flash_noetic_routed_expert_composition_native.v1",
                "nomenclature_version": "HAWKING_NOMENCLATURE_V1",
                "status": "PASSED",
                "semantic_type": "NoeticExecutableCandidate",
                "compiler_stage": "HawkingAccelerator",
                "qualification": "BOUNDED_NATIVE_ROUTED_EXPERT_GATE_UP_DOWN_COMPOSITION",
                "execution": {
                    "selected_expert_count": 2,
                    "native_gate_up_swiglu_observed": True,
                    "native_down_projection_observed": True,
                    "native_expert_composition_observed": True,
                },
                "native_gate_up_swiglu_observed": True,
                "native_down_projection_observed": True,
                "native_expert_composition_observed": True,
                "bounded_selected_expert_output_observed": True,
                "noetic_ir": {"source_independent": True},
                "intermediate": {"device_resident": True, "host_roundtrip": False},
                "whole_model_capability": "NOT_TESTED",
                "complete_expert_runtime": "NOT_TESTED",
                "complete_token_runtime": "NOT_TESTED",
                "complete_system_ebpw": None,
                "flash_tps": None,
                "promotion_allowed": False,
                "physical_graph": {"fingerprint": "graph", "device_intermediate_no_host_roundtrip": True},
                "claim_boundary": "bounded composition only",
                "next_action": "continue",
            }
        ),
        encoding="utf-8",
    )

    summary = _native_expert_composition_summary(tmp_path, receipt)

    assert summary["status"] == "PASSED"
    assert summary["native_gate_up_swiglu_observed"] is True
    assert summary["native_down_projection_observed"] is True
    assert summary["native_expert_composition_observed"] is True
    assert summary["device_intermediate_no_host_roundtrip"] is True
    assert summary["source_independent_execution"] is True
    assert summary["complete_token_runtime"] == "NOT_TESTED"
    assert summary["complete_system_ebpw"] is None
    assert summary["promotion_allowed"] is False


def test_native_shared_expert_composition_summary_preserves_gated_device_chain_boundary(tmp_path):
    receipt = tmp_path / "native-shared-expert-composition.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "hawking.flash_noetic_shared_expert_composition_native.v1",
                "nomenclature_version": "HAWKING_NOMENCLATURE_V1",
                "status": "PASSED",
                "semantic_type": "NoeticExecutableCandidate",
                "compiler_stage": "HawkingAccelerator",
                "qualification": "BOUNDED_NATIVE_SHARED_EXPERT_Q4_G64_GATED_SWIGLU_COMPOSITION",
                "layer": 0,
                "execution": {
                    "complete_shared_expert_candidate_graph": True,
                    "native_shared_expert_gate_up_swiglu_observed": True,
                    "native_shared_expert_down_projection_observed": True,
                    "native_shared_expert_scalar_gate_observed": True,
                    "native_shared_expert_sigmoid_gate_observed": True,
                    "native_shared_expert_composition_observed": True,
                    "device_intermediate_no_host_roundtrip": True,
                    "dispatches_per_graph": 4,
                },
                "native_shared_expert_gate_up_swiglu_observed": True,
                "native_shared_expert_down_projection_observed": True,
                "native_shared_expert_scalar_gate_observed": True,
                "native_shared_expert_sigmoid_gate_observed": True,
                "native_shared_expert_composition_observed": True,
                "noetic_ir": {"source_independent": True},
                "intermediates": {"device_resident": True, "host_roundtrip": False},
                "whole_model_capability": "NOT_TESTED",
                "complete_expert_runtime": "NOT_TESTED",
                "complete_token_runtime": "NOT_TESTED",
                "complete_system_ebpw": None,
                "flash_tps": None,
                "promotion_allowed": False,
                "physical_graph": {"fingerprint": "graph", "device_intermediate_no_host_roundtrip": True},
                "claim_boundary": "bounded shared-expert composition only",
                "next_action": "continue",
            }
        ),
        encoding="utf-8",
    )

    summary = _native_shared_expert_composition_summary(tmp_path, receipt)

    assert summary["status"] == "PASSED"
    assert summary["native_shared_expert_gate_up_swiglu_observed"] is True
    assert summary["native_shared_expert_down_projection_observed"] is True
    assert summary["native_shared_expert_scalar_gate_observed"] is True
    assert summary["native_shared_expert_sigmoid_gate_observed"] is True
    assert summary["native_shared_expert_composition_observed"] is True
    assert summary["device_intermediate_no_host_roundtrip"] is True
    assert summary["source_independent_execution"] is True
    assert summary["complete_token_runtime"] == "NOT_TESTED"
    assert summary["complete_system_ebpw"] is None
    assert summary["promotion_allowed"] is False


def test_native_shared_residual_hyperconnection_summary_preserves_candidate_boundary(tmp_path):
    receipt = tmp_path / "native-shared-residual-hyperconnection.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "hawking.flash_noetic_shared_residual_hyperconnection_native.v1",
                "nomenclature_version": "HAWKING_NOMENCLATURE_V1",
                "status": "PASSED",
                "semantic_type": "NoeticExecutableCandidate",
                "compiler_stage": "HawkingAccelerator",
                "qualification": "BOUNDED_NATIVE_SHARED_EXPERT_RESIDUAL_HYPERCONNECTION_COMPOSITION",
                "layer": 0,
                "execution": {
                    "dispatches_per_graph": 9,
                    "native_hyperconnection_stream_injection_observed": True,
                    "native_hyperconnection_low_rank_down_observed": True,
                    "native_hyperconnection_low_rank_up_observed": True,
                    "native_hyperconnection_block_inject_observed": True,
                    "native_hyperconnection_residual_mix_observed": True,
                    "native_shared_residual_composition_observed": True,
                    "device_intermediate_no_host_roundtrip": True,
                },
                "native_shared_expert_gate_up_swiglu_observed": True,
                "native_shared_expert_down_projection_observed": True,
                "native_shared_expert_sigmoid_gate_observed": True,
                "native_hyperconnection_stream_injection_observed": True,
                "native_hyperconnection_low_rank_down_observed": True,
                "native_hyperconnection_low_rank_up_observed": True,
                "native_hyperconnection_block_inject_observed": True,
                "native_hyperconnection_residual_mix_observed": True,
                "native_shared_residual_composition_observed": True,
                "noetic_ir": {"source_independent": True},
                "physical_graph": {"fingerprint": "graph", "device_intermediate_no_host_roundtrip": True},
                "candidate_semantics": {"status": "BOUNDED_CANDIDATE_ONLY", "hc_norm": "NOT_LOADED"},
                "whole_model_capability": "NOT_TESTED",
                "complete_expert_runtime": "NOT_TESTED",
                "complete_token_runtime": "NOT_TESTED",
                "complete_system_ebpw": None,
                "flash_tps": None,
                "promotion_allowed": False,
                "claim_boundary": "bounded shared-expert-to-hyperconnection candidate only",
                "next_action": "qualify exact source semantics",
            }
        ),
        encoding="utf-8",
    )

    summary = _native_shared_residual_hyperconnection_summary(tmp_path, receipt)

    assert summary["status"] == "PASSED"
    assert summary["native_hyperconnection_stream_injection_observed"] is True
    assert summary["native_hyperconnection_low_rank_down_observed"] is True
    assert summary["native_hyperconnection_low_rank_up_observed"] is True
    assert summary["native_hyperconnection_block_inject_observed"] is True
    assert summary["native_hyperconnection_residual_mix_observed"] is True
    assert summary["native_shared_residual_composition_observed"] is True
    assert summary["device_intermediate_no_host_roundtrip"] is True
    assert summary["source_independent_execution"] is True
    assert summary["candidate_semantics"]["hc_norm"] == "NOT_LOADED"
    assert summary["complete_token_runtime"] == "NOT_TESTED"
    assert summary["promotion_allowed"] is False


def test_handoff_counts_nested_modellake_partials_without_ds_store(tmp_path):
    receipts = tmp_path / "receipts" / "headless"
    receipts.mkdir(parents=True)
    (receipts / "MODELLAKE_FLASH_NEXT_CENSUS.json").write_text(
        json.dumps(
            {
                "status": "PASSED",
                "partials": {
                    "entries": [
                        {"name": ".DS_Store"},
                        {"name": "partial-a"},
                        {"name": "partial-b"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    summary = _model_lake_summary(tmp_path)

    assert summary["census"]["partial_count"] == 2

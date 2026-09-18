"""Focused regression tests for the Flash whole-body NR rate-budget adapter."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import flash_complete_nr as flash_nr  # noqa: E402


def test_complete_nr_consumes_the_verified_current_execution_frontier() -> None:
    frontier = flash_nr.validate_execution_frontier()
    assert frontier["valid"] is True
    assert frontier["e02_replicated"] is False
    assert frontier["e02_independent_survivor_count"] == 2
    assert frontier["e06_candidate_for_separate_review"] is False
    assert frontier["e06_conditional_net_saving_bits_after_auxiliary_cost"] < 0
    assert frontier["e13_row_extraction_performed"] is False
    assert frontier["e13_tensor_payload_bytes_read"] == 0


def test_whole_body_budget_uses_ceiling_bytes_and_identifies_tail_blocker():
    budget = flash_nr.target_budget(
        parameter_count=100,
        family_source_bytes={
            "routed_experts": 80,
            "ngram_embedding": 80,
            "linear_attention_hyperconnection": 40,
        },
        raw_target="0.5",
    )
    # 100 * 0.5 bits is 50 bits; a physical representation needs seven bytes.
    assert budget["target_persistent_bytes_ceiling"] == 7
    assert budget["tail_source_bytes_if_left_bf16"] == 40
    assert budget["tail_alone_exceeds_target"] is True
    assert budget["remaining_for_dominant_bytes_if_tail_left_bf16"] == 0


def test_sub_ppm_target_is_refused_instead_of_silently_rounding():
    with pytest.raises(SystemExit, match="integer EBPW ppm"):
        flash_nr.target_ebpw_ppm("0.0000001")


def test_lowrank_activation_summary_keeps_teacher_control_and_target_boundary():
    document = {
        "schema": flash_nr.LOWRANK_ACTIVATION_SCREEN_SCHEMA,
        "status": "SOURCE_ACTIVATION_RATE_DISTORTION_ONLY",
        "rows": [
            {
                "rank": 1,
                "residual_fraction": 0.0,
                "complete_scoped_ebpw": 0.25,
                "relative_l2": 0.9,
                "source_activation_output": {"relative_l2": 0.8, "cosine": 0.6},
            },
            {
                "rank": 2,
                "residual_fraction": 0.0,
                "complete_scoped_ebpw": 0.5,
                "relative_l2": 0.8,
                "source_activation_output": {"relative_l2": 0.7, "cosine": 0.7},
            },
        ],
    }
    summary = flash_nr._best_lowrank_activation_rows(document)
    at_half = next(row for row in summary if row["target_scoped_ebpw"] == 0.5)
    assert at_half["best_candidate"]["rank"] == 2
    assert at_half["status"].startswith("SOURCE_ACTIVATION_CONTROL_ONLY")


def test_ngram_pq_summary_requires_the_static_screen_status():
    summary = flash_nr._ngram_pq_summary(
        {
            "schema": flash_nr.SHARDED_LOOKUP_PQ_SCREEN_SCHEMA,
            "status": "SAMPLED_STATIC_RATE_DISTORTION_ONLY",
            "target_summary": [{"target_projected_table_ebpw": 0.5, "best_candidate": None}],
        }
    )
    assert summary == [{"target_projected_table_ebpw": 0.5, "best_candidate": None}]
    assert flash_nr._ngram_pq_summary({"status": "PASSED"}) == []


def test_ple_access_trace_summary_preserves_its_output_boundary():
    document = {
        "schema": flash_nr.PLE_ACCESS_TRACE_SCHEMA,
        "status": "SOURCE_ALGORITHM_BOUND_TOKEN_ACCESS_TRACE__NOT_PLE_OUTPUT_PARITY",
        "sequence": {"binding": "STATEFUL_ACCEPTED_TOKEN_SEQUENCE_RECEIPT"},
        "active_lookup": {
            "token_count": 7,
            "lookup_events": 112,
            "lookup_events_per_token": 16,
            "unique_source_rows": 112,
            "unique_source_bytes": 35840,
            "source_bytes_referenced_across_events": 35840,
        },
    }
    summary = flash_nr._ple_access_trace_summary(document)
    assert summary is not None
    assert summary["lookup_events_per_token"] == 16
    assert summary["boundary"].endswith("NOT_PLE_OUTPUT_OR_CAPABILITY")
    assert flash_nr._ple_access_trace_summary({"status": "PASSED"}) is None


def test_ple_source_output_control_summary_preserves_its_parity_boundary():
    document = {
        "schema": flash_nr.PLE_SOURCE_OUTPUT_CONTROL_SCHEMA,
        "status": "SOURCE_BOUND_PLE_OUTPUT_CONTROL__NO_INDEPENDENT_PLE_OUTPUT_PARITY",
        "input_control": {
            "qualification": "EXACT_LAYER_SOURCE_PARITY",
            "state": {"sha256": "input-state"},
        },
        "active_lookup": {
            "token_count": 1,
            "lookup_events": 16,
            "lookup_events_per_token": 16,
            "unique_source_rows": 16,
            "unique_source_bytes": 5120,
        },
        "outputs": {
            "post_injection_state": {"sha256": "post-state"},
            "mutable_state_bytes": 368656,
        },
    }
    summary = flash_nr._ple_source_output_control_summary(document)
    assert summary is not None
    assert summary["input_qualification"] == "EXACT_LAYER_SOURCE_PARITY"
    assert summary["lookup_events_per_token"] == 16
    assert summary["boundary"].endswith("NOT_INDEPENDENT_PLE_PARITY_OR_CAPABILITY")
    assert flash_nr._ple_source_output_control_summary({"status": "PASSED"}) is None


def test_reference_formula_parity_summary_keeps_the_one_control_boundary():
    document = {
        "schema": flash_nr.PLE_REFERENCE_FORMULA_ORACLE_SCHEMA,
        "status": "REFERENCE_FRAMEWORK_PLE_FORMULA_PARITY_PASS",
        "source_control": {"seal_sha256": "source-control-seal"},
        "metrics": {
            "max_abs": 4.5e-8,
            "rmse": 1e-8,
            "relative_l2": 1.5e-7,
            "cosine": 1.0,
            "finite": True,
        },
    }
    summary = flash_nr._ple_reference_formula_parity_summary(document)
    assert summary is not None
    assert summary["source_control_seal"] == "source-control-seal"
    assert summary["relative_l2"] == 1.5e-7
    assert summary["boundary"].endswith("NOT_FULL_CHECKPOINT_OR_CAPABILITY")
    assert flash_nr._ple_reference_formula_parity_summary({"status": "PASSED"}) is None


def test_ple_output_pq_repair_summary_keeps_global_projection_and_scope_boundary():
    document = {
        "schema": flash_nr.PLE_LOOKUP_PQ_OUTPUT_SCREEN_SCHEMA,
        "status": "SOURCE_PLE_OUTPUT_PQ_PLUS_SPARSE_REPAIR_RATE_DISTORTION_ONLY",
        "source_control": {"seal_sha256": "source-control-seal"},
        "reference_parity": {"seal_sha256": "reference-parity-seal"},
        "target_summary": [
            {
                "target_projected_table_ebpw": 0.5,
                "best_candidate": {
                    "subdimension": 16,
                    "cardinality": 8,
                    "repair_count_per_row": 2,
                    "projected_complete_table_ebpw": 0.4875,
                    "ple_output_relative_l2": 0.8505,
                    "ple_output_cosine": 0.5381,
                },
            }
        ],
    }
    summary = flash_nr._ple_lookup_pq_output_screen_summary(document)
    assert summary is not None
    assert summary["target_summary"][0]["repair_count_per_row"] == 2
    assert summary["source_control_seal"] == "source-control-seal"
    assert summary["boundary"].endswith("NOT_COMPLETE_PLE_OR_NR")
    assert flash_nr._ple_lookup_pq_output_screen_summary({"status": "PASSED"}) is None


def test_ple_shard_local_pq_summary_keeps_its_separate_hierarchy_boundary():
    document = {
        "schema": flash_nr.PLE_LOOKUP_PQ_OUTPUT_SCREEN_SCHEMA,
        "status": flash_nr.PLE_SHARD_LOCAL_PQ_OUTPUT_SCREEN_STATUS,
        "source_control": {"seal_sha256": "source-control-seal"},
        "reference_parity": {"seal_sha256": "reference-parity-seal"},
        "target_summary": [
            {
                "target_projected_table_ebpw": 0.5,
                "best_candidate": {
                    "codebook_scope": "shard_local",
                    "subdimension": 8,
                    "cardinality": 8,
                    "repair_count_per_row": 0,
                    "projected_complete_table_ebpw": 0.375062,
                    "ple_output_relative_l2": 0.859731,
                    "ple_output_cosine": 0.532243,
                },
            }
        ],
    }
    summary = flash_nr._ple_shard_local_pq_output_screen_summary(document)
    assert summary is not None
    assert summary["target_summary"][0]["codebook_scope"] == "shard_local"
    assert summary["target_summary"][0]["repair_count_per_row"] == 0
    assert summary["boundary"].endswith("NOT_COMPLETE_PLE_OR_NR")
    assert flash_nr._ple_shard_local_pq_output_screen_summary({"status": "PASSED"}) is None


def test_ple_additive_pq_summary_keeps_its_separate_function_family_boundary():
    document = {
        "schema": flash_nr.PLE_LOOKUP_PQ_OUTPUT_SCREEN_SCHEMA,
        "status": flash_nr.PLE_ADDITIVE_PQ_OUTPUT_SCREEN_STATUS,
        "source_control": {"seal_sha256": "source-control-seal"},
        "reference_parity": {"seal_sha256": "reference-parity-seal"},
        "target_summary": [
            {
                "target_projected_table_ebpw": 0.5,
                "best_candidate": {
                    "codebook_scope": "global_shared",
                    "quantizer_family": "residual_additive",
                    "residual_stages": 2,
                    "subdimension": 10,
                    "cardinality": 4,
                    "repair_count_per_row": 0,
                    "projected_complete_table_ebpw": 0.400011,
                    "ple_output_relative_l2": 0.861647,
                    "ple_output_cosine": 0.558434,
                },
            }
        ],
    }
    summary = flash_nr._ple_additive_pq_output_screen_summary(document)
    assert summary is not None
    assert summary["target_summary"][0]["quantizer_family"] == "residual_additive"
    assert summary["target_summary"][0]["residual_stages"] == 2
    assert summary["boundary"].endswith("NOT_COMPLETE_PLE_OR_NR")
    assert flash_nr._ple_additive_pq_output_screen_summary({"status": "PASSED"}) is None


def test_ple_address_generator_summary_keeps_the_function_screen_boundary():
    document = {
        "schema": flash_nr.PLE_ADDRESS_GENERATOR_SCREEN_SCHEMA,
        "status": "SOURCE_PLE_OUTPUT_GENERATOR_RATE_DISTORTION_ONLY",
        "source_control": {"seal_sha256": "source-control-seal"},
        "reference_parity": {"seal_sha256": "reference-parity-seal"},
        "target_summary": [
            {
                "target_projected_table_ebpw": 0.5,
                "best_candidate": {
                    "feature_count": 65,
                    "projected_complete_table_ebpw": 0.000004,
                    "heldout_row_relative_l2": 1.02,
                    "ple_output_relative_l2": 0.95,
                    "ple_output_cosine": 0.30,
                },
            }
        ],
    }
    summary = flash_nr._ple_address_generator_screen_summary(document)
    assert summary is not None
    assert summary["target_summary"][0]["feature_count"] == 65
    assert summary["boundary"].endswith("NOT_COMPLETE_PLE_OR_NR")
    assert flash_nr._ple_address_generator_screen_summary({"status": "PASSED"}) is None


def test_ple_additive_hash_summary_refuses_a_fourier_or_unlabeled_screen():
    document = {
        "schema": flash_nr.PLE_ADDRESS_GENERATOR_SCREEN_SCHEMA,
        "status": "SOURCE_PLE_OUTPUT_GENERATOR_RATE_DISTORTION_ONLY",
        "representation": {"requested_family": "additive-hash"},
        "source_control": {"seal_sha256": "source-control-seal"},
        "reference_parity": {"seal_sha256": "reference-parity-seal"},
        "target_summary": [
            {
                "target_projected_table_ebpw": 0.1,
                "best_candidate": {
                    "family": "address_conditioned_additive_hash_factor_generator",
                    "configuration": {"bucket_count": 64, "hash_terms": 3},
                    "feature_count": 193,
                    "projected_complete_table_ebpw": 0.00001,
                    "heldout_row_relative_l2": 1.1,
                    "ple_output_relative_l2": 0.98,
                    "ple_output_cosine": 0.2,
                },
            }
        ],
    }
    summary = flash_nr._ple_additive_hash_function_screen_summary(document)
    assert summary is not None
    assert summary["target_summary"][0]["configuration"]["bucket_count"] == 64
    assert summary["target_summary"][0]["family"].endswith("factor_generator")

    document["representation"] = {"requested_family": "fourier"}
    assert flash_nr._ple_additive_hash_function_screen_summary(document) is None


def test_ple_context_response_summary_requires_a_valid_tangent_receipt() -> None:
    document = {
        "schema": flash_nr.PLE_CONTEXT_RESPONSE_SCREEN_SCHEMA,
        "status": flash_nr.PLE_CONTEXT_RESPONSE_SCREEN_STATUS,
        "source_control_batch_check": {"passed": True},
        "context_bank": {
            "direction_count": 8,
            "context_count": 17,
            "relative_rms": 0.01,
            "source_trajectory_distribution": False,
        },
        "address_context_response": {
            "cross_address_delta_spectrum": {
                "ranks_at_energy": {"0.9": 6, "0.95": 7, "0.99": 10},
                "numerical_rank": 78,
            }
        },
        "address_inputs": {
            "historical_stateful_address_sequence": {
                "historical_session_provenance": {"integrity_boundary": "inner seal withheld"}
            }
        },
    }
    document["seal_sha256"] = flash_nr.hashlib.sha256(
        flash_nr.json.dumps(document, sort_keys=True).encode()
    ).hexdigest()
    summary = flash_nr._ple_context_response_screen_summary(document)
    assert summary is not None
    assert summary["source_control_batch_passed"] is True
    assert summary["cross_address_delta_spectrum"]["ranks_at_energy"]["0.9"] == 6
    assert summary["boundary"].endswith("NOT_REAL_STATE_BANK_OR_REPRESENTATION")
    document["seal_sha256"] = "0" * 64
    assert flash_nr._ple_context_response_screen_summary(document) is None


def test_router_source_output_summary_preserves_route_failure_boundary() -> None:
    document = {
        "schema": flash_nr.ROUTER_SOURCE_OUTPUT_SCREEN_SCHEMA,
        "status": "FIXED_INT4_ROUTER_ROUTE_GATE_REJECTED",
        "source": {"router_tensor": "layers.0.mlp.gate.weight", "router_shape": [512, 2560]},
        "candidate": {
            "source_output": {
                "source_ids": [1] * 10,
                "candidate_ids": [2] * 10,
                "candidate_top10_symmetric_difference": 16,
                "candidate_top10_order_positions_changed": 8,
                "logits": {"relative_l2": 0.346322, "rmse": 2.39, "max_abs_error": 5.1},
                "route_weight_error": {"max_abs_error": 0.2},
                "source_top10_top11_margin": 0.05,
                "candidate_top10_top11_margin": 0.03,
            }
        },
        "matched_null": {
            "source_output": {
                "null_top10_symmetric_difference": 18,
                "logits": {"relative_l2": 1.2},
            }
        },
        "accounting": {"complete_scoped_ebpw": 4.4078125, "complete_candidate_bytes": 722176},
        "gate": {
            "source_control_recomputed_ids_match_bridge": True,
            "fixed_candidate_route_identity_pass": False,
            "tail_non_router_organs": "WITHHELD",
        },
    }
    summary = flash_nr._router_source_output_screen_summary(document)
    assert summary is not None
    assert summary["candidate_top10_symmetric_difference"] == 16
    assert summary["fixed_candidate_route_identity_pass"] is False
    assert summary["complete_scoped_ebpw"] == 4.4078125
    assert summary["boundary"].endswith("NOT_TAIL_COMPLETE_NR_OR_CAPABILITY")

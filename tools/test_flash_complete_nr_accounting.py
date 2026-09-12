"""Focused regression tests for the Flash whole-body NR rate-budget adapter."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import flash_complete_nr as flash_nr  # noqa: E402


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

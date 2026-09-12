from __future__ import annotations

import numpy as np

from tools.odyssey.ple_lookup_pq_output_screen import (
    apply_exact_sparse_repair,
    parse_candidates,
    parse_repair_counts,
    projected_table_accounting,
    target_summary,
)
from tools.odyssey.sharded_lookup_pq_screen import ProductQuantizer


def test_candidate_parser_deduplicates_and_rejects_bad_shapes() -> None:
    assert parse_candidates("32:8,16:8,32:8") == [(32, 8), (16, 8)]
    try:
        parse_candidates("32x8")
    except ValueError as exc:
        assert "subdimension:cardinality" in str(exc)
    else:
        raise AssertionError("malformed candidate must be rejected")


def test_projected_table_accounting_bills_codes_codebooks_and_decoder() -> None:
    quantizer = ProductQuantizer(
        subdimension=2,
        cardinality=4,
        codebooks=(np.zeros((4, 2), dtype=np.float32), np.zeros((4, 2), dtype=np.float32)),
    )
    bill = projected_table_accounting(quantizer, total_rows=10, width=4)
    assert bill["code_payload_bytes"] == 5
    assert bill["codebook_bytes"] == 32
    assert bill["complete_projected_bytes"] == (
        bill["code_payload_bytes"]
        + bill["codebook_bytes"]
        + bill["metadata_bytes"]
        + bill["decoder_support_bytes"]
    )


def test_sparse_repair_is_billed_and_restores_fixed_largest_errors() -> None:
    quantizer = ProductQuantizer(
        subdimension=2,
        cardinality=4,
        codebooks=(np.zeros((4, 2), dtype=np.float32), np.zeros((4, 2), dtype=np.float32)),
    )
    without_repair = projected_table_accounting(quantizer, total_rows=10, width=4)
    with_repair = projected_table_accounting(
        quantizer, total_rows=10, width=4, repair_count_per_row=1
    )
    assert with_repair["repair_payload_bytes"] == 23
    assert with_repair["complete_projected_bytes"] > without_repair["complete_projected_bytes"]
    source = np.array([[1.0, -5.0, 2.0, 3.0]], dtype=np.float32)
    base = np.zeros_like(source)
    repaired = apply_exact_sparse_repair(source, base, repair_count_per_row=1)
    np.testing.assert_array_equal(repaired, np.array([[0.0, -5.0, 0.0, 0.0]], dtype=np.float32))


def test_repair_parser_deduplicates_and_rejects_negative_counts() -> None:
    assert parse_repair_counts("0,1,2,1") == [0, 1, 2]
    try:
        parse_repair_counts("-1")
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative repair count must be rejected")


def test_target_summary_never_claims_a_materialized_representation() -> None:
    candidate = {
        "subdimension": 8,
        "cardinality": 8,
        "repair_count_per_row": 2,
        "projected_full_table": {"projected_complete_table_ebpw": 0.375},
        "active_row_metrics": {"global_relative_l2": 0.8},
        "ple_output_metrics": {"relative_l2": 0.7, "cosine": 0.8},
    }
    summary = target_summary([candidate])
    at_half = next(item for item in summary if item["target_projected_table_ebpw"] == 0.5)
    assert at_half["best_candidate"]["ple_output_relative_l2"] == 0.7
    assert at_half["best_candidate"]["repair_count_per_row"] == 2
    assert at_half["status"].endswith("NOT_A_REPRESENTATION_CLAIM")

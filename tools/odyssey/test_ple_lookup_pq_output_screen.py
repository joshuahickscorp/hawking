from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tools.odyssey.ple_lookup_pq_output_screen import (
    CODEBOOK_SCOPE_SHARD_LOCAL,
    QUANTIZER_FAMILY_RESIDUAL_ADDITIVE,
    QUANTIZER_FAMILY_SINGLE_STAGE,
    active_shard_local_representation_accounting,
    apply_exact_sparse_repair,
    encode_decode_shard_local_product_quantizers,
    parse_candidates,
    parse_repair_counts,
    partition_source_sample_rows_by_shard,
    projected_table_accounting,
    projected_shard_local_table_accounting,
    sampled_row_shard_ordinals,
    target_summary,
    validate_quantizer_configuration,
    validate_repair_counts_for_scope,
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


def test_shard_local_sampling_preserves_source_shard_ownership() -> None:
    shards = (
        SimpleNamespace(ordinal=4, tensor="source.shard_b", rows=8),
        SimpleNamespace(ordinal=9, tensor="source.shard_a", rows=8),
    )
    source_sample = {
        # This ordering deliberately differs from physical shard ordinal.
        "sampled_tensors": [
            {"tensor": "source.shard_a", "train_row_ids": [0, 4], "heldout_row_ids": [2]},
            {"tensor": "source.shard_b", "train_row_ids": [1], "heldout_row_ids": [3, 7]},
        ]
    }
    assert sampled_row_shard_ordinals(source_sample, shards, partition="train") == [9, 9, 4]
    rows = np.array([[90.0], [91.0], [40.0]], dtype=np.float32)
    by_shard = partition_source_sample_rows_by_shard(rows, source_sample, shards, partition="train")
    np.testing.assert_array_equal(by_shard[9], np.array([[90.0], [91.0]], dtype=np.float32))
    np.testing.assert_array_equal(by_shard[4], np.array([[40.0]], dtype=np.float32))


def test_shard_local_accounting_bills_each_codebook_and_directory() -> None:
    quantizer = ProductQuantizer(
        subdimension=2,
        cardinality=4,
        codebooks=(np.zeros((4, 2), dtype=np.float32), np.zeros((4, 2), dtype=np.float32)),
    )
    bill = projected_shard_local_table_accounting(
        {4: quantizer, 9: quantizer}, {4: 10, 9: 20}, width=4
    )
    assert bill["codebook_scope"] == "physical_shard_local"
    assert bill["code_payload_bytes"] == 15
    assert bill["codebook_bytes"] == 64
    assert bill["codebook_directory_bytes"] == 64
    assert bill["complete_projected_bytes"] == (
        bill["code_payload_bytes"]
        + bill["codebook_bytes"]
        + bill["codebook_directory_bytes"]
        + bill["metadata_bytes"]
        + bill["decoder_support_bytes"]
    )


def test_shard_local_decode_touches_only_bound_codebooks_and_bills_them() -> None:
    zero = ProductQuantizer(
        subdimension=2,
        cardinality=2,
        codebooks=(np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)),
    )
    one = ProductQuantizer(
        subdimension=2,
        cardinality=2,
        codebooks=(np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)),
    )
    reconstructed = encode_decode_shard_local_product_quantizers(
        np.zeros((3, 4), dtype=np.float32), [4, 9, 4], {4: zero, 9: one}
    )
    np.testing.assert_array_equal(
        reconstructed,
        np.array([[0.0] * 4, [1.0] * 4, [0.0] * 4], dtype=np.float32),
    )
    active = active_shard_local_representation_accounting([4, 9, 4], {4: zero, 9: one})
    assert active["active_code_bytes_for_this_control"] == 1
    assert active["active_codebook_bytes_for_this_control"] == 32
    assert active["active_codebook_directory_bytes_for_this_control"] == 64
    assert active["active_representation_bytes_for_this_control"] == 97


def test_shard_local_rejects_fixed_sparse_repair_hybrid() -> None:
    validate_repair_counts_for_scope(CODEBOOK_SCOPE_SHARD_LOCAL, [0])
    try:
        validate_repair_counts_for_scope(CODEBOOK_SCOPE_SHARD_LOCAL, [0, 1])
    except ValueError as exc:
        assert "separate hierarchy screen" in str(exc)
    else:
        raise AssertionError("shard-local screen must not silently repeat the repair hybrid")


def test_residual_additive_configuration_is_distinct_from_copied_source_repair() -> None:
    validate_quantizer_configuration(
        QUANTIZER_FAMILY_SINGLE_STAGE, residual_stages=1, repair_counts=[0, 1]
    )
    validate_quantizer_configuration(
        QUANTIZER_FAMILY_RESIDUAL_ADDITIVE, residual_stages=2, repair_counts=[0]
    )
    try:
        validate_quantizer_configuration(
            QUANTIZER_FAMILY_RESIDUAL_ADDITIVE, residual_stages=2, repair_counts=[1]
        )
    except ValueError as exc:
        assert "copied source-BF16 repair" in str(exc)
    else:
        raise AssertionError("additive PQ must not silently add copied source residuals")

from __future__ import annotations

import numpy as np

from tools.odyssey.ple_access_trace import LookupShard
from tools.odyssey.ple_address_generator_screen import (
    additive_hash_feature_matrix,
    address_feature_matrix,
    fit_address_generator,
    generator_accounting,
    parse_hash_bucket_counts,
    parse_feature_counts,
    round_bf16,
    target_summary,
)


def _shards() -> list[LookupShard]:
    return [
        LookupShard(0, "shard_0", __file__, 0, 0, 10, 4, 0),
        LookupShard(1, "shard_1", __file__, 0, 0, 10, 4, 10),
    ]


def test_address_features_are_deterministic_and_use_the_resolved_layout() -> None:
    rows = np.array([0, 5, 10, 19], dtype=np.int64)
    first = address_feature_matrix(rows, shards=_shards(), feature_count=9)
    second = address_feature_matrix(rows, shards=_shards(), feature_count=9)
    assert first.shape == (4, 9)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, 0], np.ones(4, dtype=np.float32))
    assert not np.array_equal(first[0], first[-1])


def test_generator_accounting_bills_all_static_generator_parts() -> None:
    accounting = generator_accounting(feature_count=5, total_rows=10, width=4)
    assert accounting["coefficient_bytes"] == 40
    assert accounting["complete_projected_bytes"] == (
        40 + accounting["metadata_bytes"] + accounting["runtime_support_bytes"]
    )
    assert accounting["projected_complete_table_ebpw"] > 0.0


def test_feature_parser_and_bf16_rounding_are_explicit() -> None:
    assert parse_feature_counts("5,9,5") == [5, 9]
    assert parse_hash_bucket_counts("8,16,8") == [8, 16]
    try:
        parse_feature_counts("0")
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("zero feature count must be rejected")
    try:
        parse_hash_bucket_counts("0")
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("zero hash buckets must be rejected")
    rounded = round_bf16(np.array([1.0, 1.0001], dtype=np.float32))
    assert rounded.dtype == np.float32
    assert rounded[0] == 1.0


def test_ridge_fit_is_finite_for_collinear_features() -> None:
    features = np.ones((4, 3), dtype=np.float32)
    rows = np.arange(8, dtype=np.float32).reshape(4, 2)
    coefficients = fit_address_generator(features, rows, ridge_l2=1e-3)
    assert coefficients.shape == (3, 2)
    assert np.isfinite(coefficients).all()


def test_additive_hash_features_are_deterministic_sparse_and_fit_their_own_family() -> None:
    rows = np.arange(32, dtype=np.int64)
    features = additive_hash_feature_matrix(rows, bucket_count=4, terms=2)
    repeated = additive_hash_feature_matrix(rows, bucket_count=4, terms=2)
    assert features.shape == (32, 9)
    np.testing.assert_array_equal(features, repeated)
    np.testing.assert_array_equal(features.sum(axis=1), np.full(32, 3.0, dtype=np.float32))

    coefficients = np.arange(features.shape[1] * 3, dtype=np.float32).reshape(features.shape[1], 3)
    targets = features @ coefficients
    recovered = fit_address_generator(features, targets, ridge_l2=1e-6)
    relative_l2 = np.linalg.norm(features @ recovered - targets) / np.linalg.norm(targets)
    assert relative_l2 < 0.01


def test_target_summary_retains_the_source_control_boundary() -> None:
    candidate = {
        "feature_count": 17,
        "projected_full_table": {"projected_complete_table_ebpw": 0.001},
        "heldout_row_metrics": {"global_relative_l2": 0.9},
        "ple_output_metrics": {"relative_l2": 0.8, "cosine": 0.6},
    }
    summary = target_summary([candidate])
    assert summary[0]["best_candidate"]["feature_count"] == 17
    assert summary[0]["status"].endswith("NOT_A_REPRESENTATION_CLAIM")

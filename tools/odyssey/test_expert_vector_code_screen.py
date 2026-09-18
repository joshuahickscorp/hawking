from __future__ import annotations

import numpy as np
import pytest

from tools.odyssey.expert_vector_code_screen import (
    _complete_accounting,
    _split_rows,
    screen_matrices,
)


def test_split_rows_is_disjoint_and_covers_even_matrix() -> None:
    train, heldout = _split_rows(8)
    np.testing.assert_array_equal(train, np.array([0, 2, 4, 6]))
    np.testing.assert_array_equal(heldout, np.array([1, 3, 5, 7]))
    assert not np.intersect1d(train, heldout).size
    assert sorted(np.concatenate([train, heldout]).tolist()) == list(range(8))


def test_complete_accounting_bills_shared_codebook_and_support() -> None:
    result = _complete_accounting(
        expert_count=2,
        rows=8,
        width=16,
        subdimension=4,
        cardinality=2,
    )
    assert result["bits_per_vector_code"] == 1
    assert result["code_payload_bytes"] == 8
    assert result["shared_bf16_codebook_bytes"] == 4 * 2 * 4 * 2
    assert result["expert_directory_bytes"] == 8
    assert result["complete_bytes"] > result["code_payload_bytes"]


def test_screen_is_output_bound_and_deterministic() -> None:
    rng = np.random.default_rng(20260912)
    weights = [rng.normal(size=(8, 16)).astype(np.float32) for _ in range(2)]
    activation = rng.normal(size=16).astype(np.float32)
    kwargs = {
        "expert_ids": [3, 7],
        "route_weights": [0.75, 0.25],
        "activation": activation,
        "subdimensions": [4],
        "cardinalities": [2],
        "iterations": 2,
        "seed": 11,
    }
    first = screen_matrices(weights, **kwargs)
    second = screen_matrices(weights, **kwargs)
    assert first[0]["accounting"]["complete_scoped_ebpw"] > 16.0
    assert first[0]["heldout_weight_metrics"]["finite"] is True
    assert first[0]["source_activation_output"]["route_weighted_sum"]["finite"] is True
    assert first[0]["source_activation_output"]["route_weighted_sum"]["relative_l2"] == pytest.approx(
        second[0]["source_activation_output"]["route_weighted_sum"]["relative_l2"]
    )
    assert first[0]["direct_execution"]["status"] == "NOT_ESTABLISHED"


def test_screen_rejects_non_dividing_subdimension() -> None:
    with pytest.raises(ValueError, match="does not divide"):
        screen_matrices(
            [np.ones((4, 6), dtype=np.float32)],
            expert_ids=[1],
            route_weights=[1.0],
            activation=np.ones(6, dtype=np.float32),
            subdimensions=[4],
            cardinalities=[2],
        )

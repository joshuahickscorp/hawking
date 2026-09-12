from __future__ import annotations

import numpy as np

from tools.odyssey.sharded_lookup_pq_screen import (
    encode_decode_product_quantizer,
    fit_product_quantizer,
    screen,
)


def test_pq_screen_is_deterministic_and_bills_the_projected_closure() -> None:
    train = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.1, 1.1, 2.1, 3.1],
            [4.0, 5.0, 6.0, 7.0],
            [4.1, 5.1, 6.1, 7.1],
        ],
        dtype=np.float32,
    )
    heldout = np.array([[0.05, 1.05, 2.05, 3.05], [4.05, 5.05, 6.05, 7.05]], dtype=np.float32)
    first = screen(
        train,
        heldout,
        total_rows=10,
        subdims=[2, 4],
        cards=[2, 4],
        iterations=3,
        seed=17,
    )
    second = screen(
        train,
        heldout,
        total_rows=10,
        subdims=[2, 4],
        cards=[2, 4],
        iterations=3,
        seed=17,
    )
    assert first == second
    for candidate in first:
        projected = candidate["projected_full_table"]
        assert projected["complete_projected_bytes"] == (
            projected["code_payload_bytes"]
            + projected["codebook_bytes"]
            + projected["metadata_bytes"]
            + projected["decoder_support_bytes"]
        )
        assert candidate["heldout_metrics"]["finite"] is True
        assert candidate["representation_materialized"] is False
        assert candidate["direct_execution"] is False


def test_pq_screen_rejects_incompatible_subdimensions() -> None:
    train = np.ones((4, 4), dtype=np.float32)
    heldout = np.ones((2, 4), dtype=np.float32)
    try:
        screen(
            train,
            heldout,
            total_rows=10,
            subdims=[3],
            cards=[2],
            iterations=1,
            seed=1,
        )
    except ValueError as exc:
        assert "does not divide" in str(exc)
    else:
        raise AssertionError("incompatible subdimension must be rejected")


def test_fitted_quantizer_reuses_the_screen_construction_deterministically() -> None:
    train = np.array(
        [[0.0, 1.0, 2.0, 3.0], [0.1, 1.1, 2.1, 3.1], [4.0, 5.0, 6.0, 7.0]],
        dtype=np.float32,
    )
    first = fit_product_quantizer(
        train, subdimension=2, cardinality=2, iterations=3, seed=17
    )
    second = fit_product_quantizer(
        train, subdimension=2, cardinality=2, iterations=3, seed=17
    )
    assert first.subdimension == 2
    assert first.cardinality == 2
    assert first.code_bits_per_subspace == 1
    for a, b in zip(first.codebooks, second.codebooks, strict=True):
        assert np.array_equal(a, b)
    np.testing.assert_array_equal(
        encode_decode_product_quantizer(train, first),
        encode_decode_product_quantizer(train, second),
    )

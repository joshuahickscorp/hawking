from __future__ import annotations

import numpy as np

from tools.odyssey.flash_router_source_output_screen import (
    accounting,
    rowwise_int4,
    screen_arrays,
    stable_topk,
)


def test_rowwise_int4_is_deterministic_and_finite() -> None:
    weights = np.asarray([[1.0, -2.0, 0.5], [0.0, 0.25, -0.75]], dtype=np.float32)
    weights = np.pad(weights, ((0, 0), (0, 1)))
    first = rowwise_int4(weights)
    second = rowwise_int4(weights)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    assert np.isfinite(first[2]).all()


def test_stable_topk_breaks_ties_by_lower_input_order() -> None:
    np.testing.assert_array_equal(stable_topk(np.asarray([1.0, 2.0, 2.0, 0.0]), 3), [1, 2, 0])


def test_router_accounting_bills_support_and_scales() -> None:
    bill = accounting(rows=512, width=2560)
    assert bill["source_bf16_bytes"] == 512 * 2560 * 2
    assert bill["int4_payload_bytes"] == 512 * 2560 // 2
    assert bill["row_scale_bf16_bytes"] == 512 * 2
    assert bill["complete_candidate_bytes"] == (
        512 * 2560 // 2 + 512 * 2 + 256 + 65536
    )
    assert 4.0 < bill["complete_scoped_ebpw"] < 4.6


def test_screen_requires_exact_ten_route_entries() -> None:
    router = np.eye(10, dtype=np.float32)
    activation = np.ones(10, dtype=np.float32)
    try:
        screen_arrays(router, activation, list(range(9)), [1.0] * 9)
    except ValueError as exc:
        assert "ten entries" in str(exc)
    else:
        raise AssertionError("short route control was accepted")


def test_screen_reports_exact_source_control_even_when_candidate_order_changes() -> None:
    router = (np.eye(10, dtype=np.float32) * 10.0) + 0.01
    activation = np.ones(10, dtype=np.float32)
    source_ids = stable_topk(router @ activation, 10).tolist()
    source_weights = [1.0 / 10.0] * 10
    result = screen_arrays(router, activation, source_ids, source_weights)
    assert result["source_control"]["source_ids_match_bridge"]
    assert set(result["candidate"]["source_output"]["candidate_ids"]) == set(source_ids)

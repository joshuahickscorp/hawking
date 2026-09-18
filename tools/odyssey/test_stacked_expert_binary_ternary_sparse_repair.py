from __future__ import annotations

import numpy as np

from tools.odyssey.stacked_expert_binary_ternary_sparse_repair import (
    _packed_bytes,
    _substrate,
    screen,
)


def test_packed_bytes_rounds_up_to_a_complete_byte():
    assert _packed_bytes(8, 1) == 1
    assert _packed_bytes(9, 1) == 2
    assert _packed_bytes(9, 2) == 3


def test_binary_and_ternary_screen_bill_all_scoped_parts():
    weight = np.array(
        [[1.0, -0.5, 0.0, 0.25], [-2.0, 1.0, 0.5, -0.25]],
        dtype=np.float32,
    )
    for base in ("binary", "ternary"):
        rows = screen(weight, base, [0.0, 0.5])
        assert rows[0]["complete_bytes"] == (
            rows[0]["packed_code_bytes"]
            + rows[0]["row_scale_bytes"]
            + rows[0]["metadata_bytes"]
            + rows[0]["decoder_support_bytes"]
        )
        assert rows[1]["complete_bytes"] > rows[0]["complete_bytes"]
        assert rows[1]["relative_l2"] <= rows[0]["relative_l2"]


def test_substrate_is_deterministic_and_shape_preserving():
    weight = np.arange(-12, 12, dtype=np.float32).reshape(4, 6)
    for base in ("binary", "ternary"):
        first = _substrate(weight, base)
        second = _substrate(weight, base)
        assert first.shape == weight.shape
        np.testing.assert_array_equal(first, second)

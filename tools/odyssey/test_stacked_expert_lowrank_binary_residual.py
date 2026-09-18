from __future__ import annotations

import numpy as np

from tools.odyssey.stacked_expert_lowrank_binary_residual import (
    _packed_bytes,
    screen,
)


def test_packed_binary_residual_billing_rounds_up():
    assert _packed_bytes(8, 1) == 1
    assert _packed_bytes(9, 1) == 2


def test_composition_bills_factors_and_dense_residual_for_each_rank():
    weight = np.array(
        [[1.0, -0.5, 0.0, 0.25], [-2.0, 1.0, 0.5, -0.25]],
        dtype=np.float32,
    )
    rows = screen(weight, [1, 2])
    assert rows[0]["complete_bytes"] == (
        rows[0]["factor_bytes"]
        + rows[0]["residual_code_bytes"]
        + rows[0]["residual_scale_bytes"]
        + rows[0]["metadata_bytes"]
        + rows[0]["decoder_support_bytes"]
    )
    assert rows[1]["factor_bytes"] > rows[0]["factor_bytes"]
    assert all(row["complete_scoped_ebpw"] > 0 for row in rows)


def test_composition_is_deterministic():
    weight = np.arange(-24, 24, dtype=np.float32).reshape(6, 8)
    first = screen(weight, [1, 2, 4])
    second = screen(weight, [1, 2, 4])
    assert first == second

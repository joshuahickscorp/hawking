from __future__ import annotations

import numpy as np
import pytest

from tools.odyssey.stacked_expert_lowrank_screen import _decode, curve


def test_bf16_decode_known_values() -> None:
    raw = np.array([0x3F80, 0x4000], dtype="<u2").tobytes()
    np.testing.assert_allclose(_decode(raw, "BF16"), np.array([1.0, 2.0], dtype=np.float32))


def test_lowrank_curve_has_monotonic_error_and_billed_factors() -> None:
    weight = np.diag(np.array([4.0, 2.0, 1.0], dtype=np.float32))
    rows = curve(weight, [1, 2, 3])
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert rows[0]["relative_frobenius_error"] > rows[1]["relative_frobenius_error"] > rows[2]["relative_frobenius_error"] - 1e-8
    assert rows[2]["relative_frobenius_error"] < 1e-6
    assert rows[1]["complete_scoped_ebpw"] > rows[0]["complete_scoped_ebpw"]


def test_curve_rejects_out_of_range_rank() -> None:
    with pytest.raises(ValueError):
        curve(np.ones((2, 3), dtype=np.float32), [3])

from __future__ import annotations

import numpy as np
import pytest

from tools.odyssey.ple_static_screen import _decode, sampled_windows, screen


def test_sampled_windows_cover_first_and_last_legal_region() -> None:
    assert sampled_windows(100, 3, 10) == [(0, 10), (45, 10), (90, 10)]
    with pytest.raises(ValueError):
        sampled_windows(9, 1, 10)


def test_bf16_decode_preserves_known_values() -> None:
    bits = np.array([0x3F80, 0xBF80, 0x0000], dtype="<u2").tobytes()
    np.testing.assert_allclose(_decode(bits, "BF16"), np.array([1.0, -1.0, 0.0], dtype=np.float32))


def test_screen_is_static_and_has_both_rate_distortion_bases() -> None:
    rng = np.random.default_rng(0)
    result = screen(rng.standard_normal((64, 8), dtype=np.float32))
    assert result["sample_rows"] == 64
    assert set(result["static_relative_l2"]) == {"binary_per_row", "ternary_per_row"}
    assert result["within_table_covariance"]["columns"] == 8

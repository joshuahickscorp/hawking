"""Unit tests for the Q80 capability-vs-BPW curve helpers. No source shards."""
from __future__ import annotations

import numpy as np

from lab.operators.q80_mixed_representation_pack import F_EXPERT, F_NONEXPERT
from lab.operators.q80_subbit_capability_curve import (
    BAR,
    complete_bpw,
    holdout_split,
    increment_ratios_from_growth,
    matched_magnitude_null,
    physical_bpw,
)


def test_identity_coefficients_have_not_moved() -> None:
    assert F_EXPERT == 0.9703169371044981
    assert F_NONEXPERT == 0.029683062895501933
    # Task-stated rounding is the first 5 decimals.
    assert abs(F_EXPERT - 0.97032) < 5e-6
    assert abs(F_NONEXPERT - 0.02968) < 5e-6


def test_sub_100_fs_expert_bpw_arithmetic() -> None:
    # complete = f_e * E + f_n * N. Solve E at complete=0.6552.
    ne8 = 8.250600705299505
    e_at_8 = (0.6552 - F_NONEXPERT * ne8) / F_EXPERT
    assert 0.40 < e_at_8 < 0.45
    ne4 = 4.2506
    e_at_4 = (0.6552 - F_NONEXPERT * ne4) / F_EXPERT
    assert 0.52 < e_at_4 < 0.56
    # Non-expert 8->4 buys ~0.12 complete BPW, not the 0.79 we need from 1.44.
    saved = F_NONEXPERT * (ne8 - ne4)
    assert 0.11 < saved < 0.13
    assert complete_bpw(1.2348805110280712, ne8) == 1.4431285748118732


def test_holdout_keeps_all_rows_when_starved() -> None:
    fit, hold, has = holdout_split(3, seed=1)
    assert not has
    assert fit.tolist() == [0, 1, 2]
    assert hold.tolist() == [0, 1, 2]


def test_holdout_is_seeded_and_disjoint() -> None:
    fit_a, hold_a, has_a = holdout_split(20, seed=7)
    fit_b, hold_b, has_b = holdout_split(20, seed=7)
    assert has_a and has_b
    assert fit_a.tolist() == fit_b.tolist()
    assert hold_a.tolist() == hold_b.tolist()
    assert set(fit_a).isdisjoint(set(hold_a))
    assert len(fit_a) + len(hold_a) == 20


def test_matched_magnitude_null_preserves_error_energy() -> None:
    rng = np.random.default_rng(0)
    src = rng.standard_normal((8, 16), dtype=np.float32)
    hat = src + 0.1 * rng.standard_normal((8, 16), dtype=np.float32)
    null = matched_magnitude_null(src, hat, seed=99)
    err = (hat - src).reshape(-1)
    nerr = (null - src).reshape(-1)
    assert np.allclose(np.sort(err), np.sort(nerr), atol=1e-6)
    assert not np.allclose(null, hat)


def test_physical_bpw_from_real_bytes() -> None:
    # binary group-128 on a 512x2048 organ: 1 sign bit + fp16 scale / 128.
    elems = 512 * 2048
    nbytes = 147708  # mixed-1p5 gate: 3630071808 bytes / 24576 tensors
    assert abs(physical_bpw(nbytes, elems) - 1.126922607421875) < 1e-12


def test_bar_is_the_d23_number() -> None:
    assert BAR == 0.8604


def test_increment_ratio_conversion_is_orthogonal_identity() -> None:
    # g = sqrt(1 + r^2)  =>  r = sqrt(g^2 - 1)
    g = math_sqrt = (1.0 + 0.5**2) ** 0.5
    r = increment_ratios_from_growth([1.0, g])
    assert abs(r[0] - 0.5) < 1e-12
    # shrinking residual (g<1) contributes r=0, not a NaN.
    assert increment_ratios_from_growth([1.0, 0.73]) == [0.0]

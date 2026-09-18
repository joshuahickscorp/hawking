"""Focused regression for the P6 gravity implementation contract v2.

Exercises the observable behavior of ``_spec_bits`` in
``hawking.gravity_gauntlet``: a spec that carries no precision information and a
spec that carries a degenerate zero-bit precision must both be reported as
"no precision information" (``None``), while a real positive precision must be
reported as its measured integer value.
"""
from __future__ import annotations

import pytest

from hawking.gravity_gauntlet import _spec_bits


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("q4", 4),
        ("q8-g2", 8),
        ("q1", 1),
        ("q0", None),
        ("q0-g2", None),
        ("no-precision-here", None),
        ("", None),
    ],
)
def test_spec_bits_reports_only_measured_positive_precision(spec, expected):
    assert _spec_bits(spec) == expected


def test_spec_bits_zero_precision_is_not_a_measured_candidate():
    # A zero-bit precision cannot store magnitude; it must not be confused with
    # a measured candidate the way an absent precision must not be.
    assert _spec_bits("q0") is None
    assert _spec_bits("q0") != 0
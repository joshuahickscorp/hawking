"""Focused regression for P15_AUTOTUNING_IMPLEMENTATION_CONTRACT_V3.

Exercises the newly added receipt-timestamp normalization helpers in
``hawking/frontier_scheduler.py`` (``_coerce_epoch_seconds`` and
``_age_seconds``). These are production behavior: every freshness probe in
this module depends on turning a producer's raw timestamp into a real age,
and the contract forbids inventing a number when the receipt is unusable.

No release or hardware qualification is claimed by this test.
"""
from __future__ import annotations

import pytest

from hawking.frontier_scheduler import _age_seconds, _coerce_epoch_seconds


@pytest.mark.parametrize(
    "raw,expected",
    [
        (1700000000, 1700000000.0),
        (1700000000.5, 1700000000.5),
        (1700000000000, 1700000000.0),  # milliseconds
        ("1700000000", 1700000000.0),
        ("2023-11-14T22:13:20Z", 1700000000.0),
        ("2023-11-14T22:13:20+00:00", 1700000000.0),
    ],
)
def test_coerce_epoch_seconds_accepts_real_producer_shapes(raw, expected):
    assert _coerce_epoch_seconds(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw",
    [None, "", "   ", "not-a-timestamp", True, False, [], {}, "2023-13-45T99:99:99Z"],
)
def test_coerce_epoch_seconds_reports_unknown_instead_of_guessing(raw):
    assert _coerce_epoch_seconds(raw) is None


def test_age_seconds_uses_injected_now_and_clamps_future_timestamps():
    assert _age_seconds(1700000000, now=1700000060.0) == pytest.approx(60.0)
    # A future-dated receipt must not yield a negative age.
    assert _age_seconds(1700000100, now=1700000000.0) == 0.0


def test_age_seconds_is_none_for_unusable_timestamp():
    assert _age_seconds("garbage", now=1700000000.0) is None
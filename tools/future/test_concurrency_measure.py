"""G009: occupancy is never the target; verified useful work per wall second is.

The concurrency Doctor already had the plan and the refusal semantics and
correctly refused to seal a law without protected observations. This harness
produces them, and the thing it must not do is rank on the wrong key: a second
session that halves everyone's rate while doubling GPU occupancy has bought
nothing.
"""
from __future__ import annotations

import pytest

from tools.future import concurrency_measure as cm


def _level(n, tokens, wall):
    return {
        "concurrency": n,
        "sessions_launched": n,
        "sessions_admitted": n,
        "cohort_wall_s": wall,
        "tokens_produced": tokens,
        "useful_tokens_per_wall_second": tokens / wall,
        "per_session": [],
        "failures": [],
    }


def test_a_second_session_that_halves_the_rate_buys_nothing():
    """Twice the sessions at half the rate is the same work per wall second."""
    got = cm.classify([_level(1, 32, 1.0), _level(2, 64, 2.0)])
    assert got["verdict"] == "NO_USEFUL_CONCURRENCY_HEADROOM"
    assert got["useful_work_ratio_vs_n1"][2] == pytest.approx(1.0)


def test_real_headroom_is_recognised():
    got = cm.classify([_level(1, 32, 1.0), _level(2, 64, 1.5)])
    assert got["verdict"] == "CONCURRENCY_HELPS"
    assert got["best_concurrency"] == 2


def test_a_gain_inside_five_percent_is_not_headroom():
    """The campaign's own bar elsewhere is "inside 5% of each other"."""
    got = cm.classify([_level(1, 32, 1.0), _level(2, 64, 1.94)])
    assert got["useful_work_ratio_vs_n1"][2] < 1.05
    assert got["verdict"] == "NO_USEFUL_CONCURRENCY_HEADROOM"


def test_no_baseline_refuses_rather_than_ranking():
    with pytest.raises(cm.ConcurrencyRefused, match="denominator"):
        cm.classify([_level(2, 64, 2.0)])


def test_a_zero_baseline_refuses_rather_than_dividing():
    with pytest.raises(cm.ConcurrencyRefused, match="against zero"):
        cm.classify([_level(1, 0, 1.0), _level(2, 64, 2.0)])


def test_the_law_refuses_to_universalise():
    doc = cm.build([_level(1, 32, 1.0), _level(2, 64, 2.0)])
    scope = doc["law_scope"]
    assert set(scope["refuses_to_universalise"]) == {"Flash", "M5", "FPGA", "CUDA"}
    for key in ("machine", "nx", "runtime"):
        assert scope[key]


def test_occupancy_is_named_as_not_the_target():
    doc = cm.build([_level(1, 32, 1.0), _level(2, 64, 2.0)])
    assert doc["classification"]["occupancy_is_not_the_target"] is True
    assert "occupancy is not available compute" in doc["claim_boundary"]


def test_the_cohort_wall_is_not_the_ranking_key():
    """It is 93% session startup even at 128 tokens.

    Ranking on cohort wall would have reported CONCURRENCY_HELPS on the strength
    of N model loads amortising, which is a fabricated finding about a real
    number. The ranking key is aggregate DECODE throughput.
    """
    rows = [
        {**_level(1, 32, 12.0), "aggregate_decode_tps": 36.5},
        # twice the sessions, each at half the decode rate: no overlap at all,
        # but the cohort wall barely moves because startup dominates.
        {**_level(2, 64, 13.0), "aggregate_decode_tps": 36.5},
    ]
    got = cm.classify(rows)
    assert got["verdict"] == "NO_USEFUL_CONCURRENCY_HEADROOM"
    assert got["useful_work_ratio_vs_n1"][2] == pytest.approx(1.0)
    assert "startup" in got["denominator_is_not_cohort_wall"]


def test_aggregate_decode_beats_cohort_wall_when_both_are_present():
    rows = [
        {**_level(1, 32, 12.0), "aggregate_decode_tps": 36.5},
        {**_level(2, 64, 13.0), "aggregate_decode_tps": 51.2},
    ]
    got = cm.classify(rows)
    assert got["verdict"] == "CONCURRENCY_HELPS"
    assert got["useful_work_ratio_vs_n1"][2] == pytest.approx(51.2 / 36.5, abs=1e-3)


def test_the_receipt_states_this_is_not_single_stream_tps():
    doc = cm.build([
        {**_level(1, 32, 12.0), "aggregate_decode_tps": 36.5},
        {**_level(2, 64, 13.0), "aggregate_decode_tps": 51.2},
    ])
    joined = " ".join(doc["what_this_does_not_mean"])
    assert "single-stream" in joined and "71" in joined

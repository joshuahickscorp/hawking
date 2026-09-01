"""The trial chooses, and the judge reads snapshots taken at the decision.

The defect this trial found in its own scheduler is the reason it exists: with
warmth unpriced and repeats undiscounted, the ranking was stable and the loop
returned to its cheapest specimen forever. A single-specimen run now fails.
"""
from __future__ import annotations

import json

import pytest

from tools.future import multi_model_trial as mt
from tools.future import specimen_scheduler as ss


def _trial():
    return json.loads((mt.REPO / mt.TRIAL_10M).read_text())


def test_an_unrun_trial_refuses_rather_than_reporting_a_pass():
    with pytest.raises(mt.TrialRefused, match="has not run"):
        mt.judge("receipts/future/NO_SUCH_TRIAL.json")


def test_the_ten_minute_trial_ran_and_passed():
    j = mt.judge(mt.TRIAL_10M)
    assert j["elapsed_seconds"] > 0
    assert j["passed"] is True, {k: v for k, v in j["checks"].items() if v is False}


def test_a_snapshot_exists_for_every_scheduling_decision():
    j = mt.judge(mt.TRIAL_10M)
    assert j["checks"]["recorded_a_snapshot_per_decision"] is True
    assert j["n_snapshots"] > 0


def test_snapshots_carry_runnable_blocked_and_resources():
    for s in _trial()["snapshots"][:20]:
        assert "n_runnable" in s and "n_blocked" in s
        assert "admissible_gb" in s and "free_gb" in s
        for b in s["blocked"]:
            assert b["reason"] in ("SCAR", "MEMORY", "?")


def test_the_judge_reads_snapshots_not_later_work():
    j = mt.judge(mt.TRIAL_10M)
    assert "Nothing here is inferred from work that appeared later" in j["judged_from"]


def test_loads_read_real_bytes_from_the_real_volume():
    j = mt.judge(mt.TRIAL_10M)
    assert j["checks"]["loaded_real_bytes"] is True
    assert j["n_loads"] > 0


def test_a_warm_vs_cold_check_with_too_few_cold_samples_is_untestable():
    """A check that cannot fail is not a check.

    After the first pass the OS page cache holds every specimen the trial
    touched, so "cold" reads are cached reads. The trial must say so rather
    than pass on noise between two cached numbers.
    """
    j = mt.judge(mt.TRIAL_10M)
    if j["n_cold"] < mt.MIN_COLD_SAMPLES:
        assert "warm_reload_was_faster_than_cold" in j["untestable_in_this_trial"]
        assert "warm_reload_was_faster_than_cold" not in j["checks"]
        assert "differing by noise" in j["why_untestable"]
    else:
        assert j["checks"]["warm_reload_was_faster_than_cold"] is True


def test_an_untestable_check_is_not_counted_as_a_pass():
    j = mt.judge(mt.TRIAL_10M)
    for name in j["untestable_in_this_trial"]:
        assert j["checks"].get(name) is not True
    if j["untestable_in_this_trial"]:
        assert "listed rather than counted as passes" in \
            j["passed_is_over_testable_checks_only"]


def test_the_warm_over_cold_claim_is_left_where_it_was_measured():
    j = mt.judge(mt.TRIAL_10M)
    if j["why_untestable"]:
        assert "SPECIMEN_LOAD_COST.json" in j["why_untestable"], \
            "the 142x figure belongs to the receipt that measured it"


def test_three_touch_states_are_recorded():
    j = mt.judge(mt.TRIAL_10M)
    assert "hid whether prefetch did anything" in j["why_three_states"]
    assert j["n_cold"] + j["n_prefetched"] + j["n_warm"] > 0


def test_a_dead_question_is_suppressed_without_loading_anything():
    j = mt.judge(mt.TRIAL_10M)
    assert j["n_questions_suppressed"] > 0
    ev = _trial()["events"]
    supp = [e for e in ev if e["kind"] == "QUESTION_SUPPRESSED"]
    assert all("nothing loaded" in e["why"] for e in supp)


def test_more_than_one_specimen_is_used():
    """The defect the trial found: a stable ranking never switches."""
    j = mt.judge(mt.TRIAL_10M)
    assert j["n_distinct_specimens"] > 1
    assert j["checks"]["used_more_than_one_specimen"] is True


def test_a_single_specimen_run_would_fail_the_judge():
    j = mt.judge(mt.TRIAL_10M)
    ev = [e for e in _trial()["events"]
          if e["kind"] != "LOAD_COMPLETED" or e["specimen"] ==
          [x for x in _trial()["events"] if x["kind"] == "LOAD_COMPLETED"][0]["specimen"]]
    one = {**_trial(), "events": ev}
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     dir=mt.REPO / "receipts" / "future") as f:
        json.dump(one, f)
        name = os.path.basename(f.name)
    try:
        assert mt.judge(f"receipts/future/{name}")["passed"] is False
    finally:
        os.unlink(mt.REPO / "receipts" / "future" / name)


def test_warmth_lowers_measured_cost_in_the_ranking():
    cold = ss.rank(hypothesis_family="x_none")
    warm = ss.rank(hypothesis_family="x_none",
                   warm={cold["ranked"][-1]["id"]})
    hot = next(r for r in warm["ranked"] if r["id"] == cold["ranked"][-1]["id"])
    cool = next(r for r in cold["ranked"] if r["id"] == cold["ranked"][-1]["id"])
    assert hot["measured_load_minutes"] < cool["measured_load_minutes"]
    assert hot["is_warm"] is True


def test_a_repeated_question_is_discounted_not_forbidden():
    base = ss.rank(hypothesis_family="x_none")
    top = base["ranked"][0]["id"]
    again = ss.rank(hypothesis_family="x_none", already_asked={("x_none", top)})
    row = next(r for r in again["ranked"] if r["id"] == top)
    assert row["already_asked"] is True
    assert row["score"] > 0, "discounted, not forbidden - a repeat can confirm"
    assert row["score"] < base["ranked"][0]["score"]


def test_the_bounded_slice_is_declared_not_hidden():
    b = mt.build()
    assert "not a sleep pretending to be I/O" in b["loads_are_bounded_slices"]
    assert "Nothing claims a full model was made resident" in \
        b["loads_are_bounded_slices"]

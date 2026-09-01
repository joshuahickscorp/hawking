"""G135 tests: a rate off a handful of turns is noise wearing a percent sign."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sovereign_trial as t  # noqa: E402


def test_both_arms_clear_the_minimum_sample():
    a = t.arms()
    assert a["before"]["n_iterations"] >= t.MIN_SAMPLE
    assert a["after"]["n_iterations"] >= t.MIN_SAMPLE


def test_a_tiny_arm_is_refused_not_reported(monkeypatch):
    real = t._iterations
    monkeypatch.setattr(t, "_iterations", lambda: real()[:t.RUN5_START + 2])
    with pytest.raises(t.TrialRefused, match="noise wearing a percent sign"):
        t.arms()


def test_an_empty_second_arm_refuses_rather_than_claiming_zero(monkeypatch):
    """An empty post-fix arm means the wrong log was read, not that the fixes
    did nothing. Reporting 0% would be the worse failure."""
    real = t._iterations
    monkeypatch.setattr(t, "_iterations", lambda: real()[:t.RUN5_START])
    with pytest.raises(t.TrialRefused, match="wrong log"):
        t.arms()


def test_the_rates_are_counted_from_the_log_not_typed():
    a = t.arms()
    its = t._iterations()
    before, after = its[:t.RUN5_START], its[t.RUN5_START:]
    for rows, arm in ((before, a["before"]), (after, a["after"])):
        assert arm["n_parsed"] == sum(1 for r in rows if r.get("parsed"))
        assert arm["parse_rate"] == pytest.approx(
            arm["n_parsed"] / arm["n_iterations"], abs=5e-4)


def test_executed_never_exceeds_parsed():
    """A turn whose reply did not parse cannot have run anything."""
    for arm in t.arms().values():
        assert arm["n_executed"] <= arm["n_parsed"] <= arm["n_iterations"]


def test_the_improvement_is_real_and_large():
    i = t.improvement()
    assert i["parse_rate_after"] > i["parse_rate_before"]
    assert i["parse_rate_ratio"] > 2.0


def test_the_dominant_fix_is_named_with_its_mechanism():
    w = t.what_changed()
    assert w["dominant"] == "IDENTICAL_REPLY_LOOP"
    assert "BYTE-IDENTICAL" in w["mechanism"]
    assert "55 consecutive unparsed" in w["the_signature_is_in_run4"]


def test_the_fixed_defects_are_read_from_the_attack_receipt():
    held = json.loads((t.REPO / t.ATTACKS_REL).read_text())["held_ids"]
    for d in t.what_changed()["defects_fixed_in_the_landing"]:
        assert d in held


def test_it_does_not_claim_a_smarter_model():
    n = t.what_this_does_not_claim()
    assert "same weights" in n["not_a_smarter_model"]
    assert "measured the harness" in n["not_a_smarter_model"]


def test_the_confound_is_named_not_argued_away():
    """Sequential runs, richer kernel by run 5. Saying so is the point."""
    n = t.what_this_does_not_claim()
    assert "not randomised" in n["not_a_controlled_experiment"]
    assert "richer kernel could raise the rate on its own" in \
        n["not_a_controlled_experiment"]


def test_parse_rate_is_not_offered_as_a_capability_result():
    n = t.what_this_does_not_claim()
    assert "says nothing about whether the science was any good" in \
        n["not_a_capability_result"]


def test_a_missing_log_refuses(monkeypatch):
    monkeypatch.setattr(t, "LOG_REL", "receipts/future/NO_SUCH.jsonl")
    with pytest.raises(t.TrialRefused, match="not on disk"):
        t._iterations()

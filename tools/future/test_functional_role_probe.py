"""The resident's second hypothesis, measured with matched arms.

The grade is not whether the gate proved important. It is whether the idea
became a controlled experiment and the belief moved on the result.
"""
from __future__ import annotations

import json

import pytest

from tools.future import functional_role_probe as fp


def test_a_missing_raw_refuses(monkeypatch):
    monkeypatch.setattr(fp, "RAW_REL", "receipts/future/NO_SUCH.json")
    with pytest.raises(fp.ProbeRefused, match="run the probe first"):
        fp.rows()


def test_the_arms_are_matched_on_elements_not_fraction():
    """gate/up are 17408x5120; down is 5120x17408. Fractions are not comparable."""
    m = fp.arms_are_matched()
    assert m["worst_element_count_spread"] < fp.MATCH_TOLERANCE
    assert m["n_groups_checked"] >= 6
    assert "same NUMBER of elements" in m["why_it_matters"]


def test_unmatched_arms_refuse(monkeypatch):
    bad = [dict(r) for r in fp.rows()]
    bad[0]["elements_destroyed"] = 1
    monkeypatch.setattr(fp, "rows", lambda: bad)
    with pytest.raises(fp.ProbeRefused, match="not a matched comparison"):
        fp.arms_are_matched()


def test_three_layers_and_four_strengths_were_measured():
    r = fp.rows()
    assert sorted({x["layer"] for x in r}) == [0, 21, 42]
    assert sorted({x["frac"] for x in r}) == [0.05, 0.1, 0.2, 0.4]


def test_the_hypothesis_is_refuted_as_stated():
    v = fp.verdict()
    assert v["status"] == "REFUTED"
    assert v["proposed_by"].startswith("the resident")
    lo, hi = v["gate_over_up_range"]
    assert hi < 2.0, "gate never reaches twice up's damage per matched element"


def test_down_is_usually_the_most_sensitive_and_it_was_in_the_discard_bucket():
    r = fp.ranking()
    assert fp.ROLE["down"] == "BULK_LINEAR", "the resident classed it as bulk"
    assert r["n_where_down_is_most_sensitive"] > r["n_points"] / 2


def test_the_measured_level_is_declared_and_is_not_capability():
    v = fp.verdict()
    assert v["measured_at_level"] == "LOCAL_FUNCTIONAL_FIDELITY"
    assert "not CAPABILITY" in v["what_this_does_not_close"]
    assert "has not been run" in v["what_this_does_not_close"]


def test_what_survives_the_refutation_is_stated():
    """A scar prunes a method, not the question behind it."""
    v = fp.verdict()
    assert "Functional role remains a legitimate allocation axis" in \
        v["what_survives_of_it"]
    assert "this particular assignment" in v["what_survives_of_it"]


def test_the_robustness_curve_is_reported_as_the_larger_result():
    b = fp.robustness()
    assert b["at_fraction_zeroed"] == 0.4
    assert b["worst_damage"] < 0.02, "40% of rows zeroed barely moves the hidden state"
    assert "how much can go, not which tensor goes first" in \
        b["why_this_is_the_bigger_result"]
    assert "licenses the next experiment, not a deletion" in b["caveat"]


def test_the_role_labels_are_flagged_as_a_hypothesis_layer():
    d = fp.build()
    assert "not source truth" in d["role_labels_are_a_hypothesis_layer"]

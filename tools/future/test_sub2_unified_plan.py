"""Three planners, and the disagreements are the useful part.

S029 asked for Claude's plan, Grok's plan, and the resident's own - then a
unification citing who contributed what and where they contradicted each other.
"""
from __future__ import annotations

import pytest

from tools.future import sub2_unified_plan as up


def test_all_three_planners_have_an_artifact_on_disk():
    p = up.planners_present()
    assert set(p) == set(up.PLANNERS)
    assert all(v["present"] for v in p.values())


def test_a_missing_plan_refuses_rather_than_unifying_two(monkeypatch):
    monkeypatch.setitem(up.SOURCES, "resident", "receipts/future/NO_SUCH.json")
    with pytest.raises(up.PlanRefused, match="fake completion"):
        up.planners_present()


def test_the_disagreements_are_recorded_with_a_winner():
    d = up.disagreements()
    assert len(d) == 3
    for x in d:
        assert x["winner"] in ("claude", "grok", "resident", "measurement", "both")
        assert x["why"] and x["consequence"]


def test_claude_lost_the_aux_arithmetic():
    d = {x["id"]: x for x in up.disagreements()}
    x = d["D1.mlp_2p25_is_aux_not_codes"]
    assert x["winner"] == "grok"
    assert "0.500 aux bits" in x["why"]
    assert "further below" in x["consequence"]


def test_the_resident_lost_the_allocation_claim_to_measurement():
    d = {x["id"]: x for x in up.disagreements()}
    x = d["D2.functional_role_allocation"]
    assert x["winner"] == "measurement"
    assert "discard bucket" in x["why"]
    assert "Scars prune methods, not goals" in x["consequence"]


def test_the_third_disagreement_resolves_as_both_true():
    d = {x["id"]: x for x in up.disagreements()}
    x = d["D3.what_2p0_would_buy_versus_whether_2p0_is_expressible"]
    assert x["winner"] == "both"
    assert "different questions" in x["resolved_by"]


def test_every_route_has_an_evidence_status_and_a_cheapest_falsifier():
    for r in up.routes():
        assert r["evidence_status"] in (
            "UNTESTED", "MEASURED_INSUFFICIENT", "REFUTED_AS_STATED")
        assert r["cheapest_falsifier"]
        assert r["from"]
        assert r["max_payoff"]


def test_the_refuted_route_is_kept_with_what_survives_of_it():
    r = {x["id"]: x for x in up.routes()}["R3.functional_role_allocation"]
    assert r["evidence_status"] == "REFUTED_AS_STATED"
    assert "AXIS survives" in r["why_live"]


def test_conventional_coding_is_marked_insufficient_not_untested():
    r = {x["id"]: x for x in up.routes()}["R2.conventional_coding"]
    assert r["evidence_status"] == "MEASURED_INSUFFICIENT"
    assert "2.5081" in r["why_live"]
    assert "already run" in r["cheapest_falsifier"]


def test_the_top_route_is_justified_by_the_hierarchy():
    b = up.build()
    assert b["top_route"] == "R1.capability_allocated_heterogeneous"
    assert "FIDELITY_HIERARCHY" in \
        {x["id"]: x for x in up.routes()}["R1.capability_allocated_heterogeneous"]["why_live"]
    assert "cannot speak to a capability claim" in b["why_that_route"]


def test_the_receipt_says_who_was_wrong():
    b = up.build()
    w = b["who_was_wrong_about_what"]
    assert "Claude was wrong" in w
    assert "resident was wrong" in w


def test_the_resident_is_not_graded_on_hit_rate():
    b = up.build()
    assert "hit rate is not the metric" in b["the_resident_is_not_scored_on_hit_rate"]

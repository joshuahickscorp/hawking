"""FAST_THOUGHT's metrics come from recorded events, or they do not get published.

The failure mode this module exists to avoid is a dashboard of plausible numbers
with no evidence under them. Every metric must refuse rather than default.
"""
from __future__ import annotations

import json

import pytest

from tools.future import fast_thought as ft


def test_the_latency_is_computed_from_real_recorded_pairs():
    l = ft.frontier_to_experiment_latency()
    assert l["n_pairs"] > 100, "this is measured, not sampled"
    assert l["median_s"] >= 0.0
    assert l["verdict"] == "GREEN"
    assert l["fraction_within_target"] == 1.0


def test_a_timeline_with_no_pairs_is_reported_not_silently_dropped():
    l = ft.frontier_to_experiment_latency()
    empty = [p for p in l["per_timeline"] if p["pairs"] == 0]
    assert empty, "some recorded runs have no ingest-then-launch pair"
    assert all("no ingest-then-launch pair" in p["note"] for p in empty)


def test_no_pairs_anywhere_refuses_rather_than_publishing_zero(monkeypatch):
    d = json.loads((ft.REPO / ft.TIMELINES[0]).read_text())
    stripped = {**d, "events": [e for e in d["events"]
                                if e.get("kind") not in (ft.INGEST, ft.LAUNCH)]}
    monkeypatch.setattr(ft, "_timeline", lambda rel: stripped)
    with pytest.raises(ft.FastThoughtRefused, match="must not be published as a number"):
        ft.frontier_to_experiment_latency()


def test_a_missing_timeline_refuses(monkeypatch):
    monkeypatch.setattr(ft, "TIMELINES", ("receipts/future/NO_SUCH.json",))
    with pytest.raises(ft.FastThoughtRefused, match="is not on disk"):
        ft.frontier_to_experiment_latency()


def test_the_latency_says_what_it_does_not_measure():
    l = ft.frontier_to_experiment_latency()
    assert "SCHEDULER's latency" in l["what_it_does_not_measure"]
    assert "nowhere near sufficient" in l["what_it_does_not_measure"]


def test_the_obligation_count_matches_the_hook_not_a_narrower_regex():
    """Counting only G-prefixed lines reported 85 against the hook's 144."""
    c = ft.claude_escalations_per_frontier_move()
    assert c["verified_obligations"] > 100
    assert c["unmet_obligations"] > 0
    assert c["commits"] > c["verified_obligations"]


def test_the_escalation_metric_is_labelled_a_proxy_with_its_direct_form():
    c = ft.claude_escalations_per_frontier_move()
    assert c["is_a_proxy"] is True
    assert "UPPER bound" in c["why_a_proxy"]
    assert "escalation event" in c["what_would_measure_it_directly"]


def test_scientific_tps_declares_its_denominator_problem():
    s = ft.scientific_tps()
    assert s["scientific_tps_per_hour"] > 0
    assert "predates this ultragoal" in s["the_denominator_is_the_whole_repo_history"]
    assert "floor, not" in s["the_denominator_is_the_whole_repo_history"]


def test_the_event_contract_admits_which_triggers_are_undemonstrated():
    e = ft.event_driven_contract()
    assert len(e["triggers"]) == 8
    assert "result_ingested" in e["observed_in_timelines"]
    assert "DECLARED, NOT DEMONSTRATED" in e["gap"]


def test_decision_compression_is_a_ranked_fact_set_not_a_narrative():
    d = ft.decision_compression()
    assert d["gap_to_60_ms"] > 0
    assert d["options"], "there must be options to choose between"
    assert all({"option", "max_ms_removable", "status"} <= set(o) for o in d["options"])
    assert d["dead_schools"], "what not to propose is part of the fact set"
    assert d["hard_bounds"]


def test_decision_compression_is_actually_small():
    """S026 §38 contrasts this with thousands of lines of history."""
    d = ft.build()["decision_compression"]
    assert d["size_bytes"] < 4000, "a compression that is not small is not one"


def test_the_summary_does_not_claim_fast_thought_is_achieved():
    h = ft.build()["honest_summary"]
    assert "What is slow is everything that still runs in a Claude turn" in h
    assert "instrumentation, not optimisation" in h

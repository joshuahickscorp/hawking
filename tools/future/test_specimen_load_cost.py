"""Load cost is measured on the real volume, and the model is scored held-out.

The failure guarded here is a cost model that never touched a disk. Every rate
comes from a timed read of real specimen bytes, and the prediction is scored on
a specimen it was not fitted to.
"""
from __future__ import annotations

import json

import pytest

from tools.future import specimen_load_cost as lc


def test_a_missing_rate_sample_refuses_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(lc, "RAW_REL", "receipts/future/NO_SUCH.json")
    with pytest.raises(lc.LoadCostRefused, match="a guess wearing a receipt"):
        lc.rates()


def test_an_absent_labelled_sample_refuses(monkeypatch):
    d = json.loads((lc.REPO / lc.RAW_REL).read_text())
    d["samples"] = [s for s in d["samples"] if s["label"] != "quiet_warm"]
    monkeypatch.setattr(lc, "_raw", lambda: d)
    with pytest.raises(lc.LoadCostRefused, match="sample quiet_warm is absent"):
        lc.rates()


def test_the_two_quiet_cold_samples_agree():
    r = lc.rates()
    assert r["quiet_cold_spread"] < 1.10, \
        "two different specimens must read at a similar cold rate"


def test_warm_dominates_cold_by_two_orders_of_magnitude():
    r = lc.rates()
    assert r["warm_over_cold"] > 100
    assert "Residency dominates contention" in r["reading"]


def test_the_contention_question_is_answered_with_a_number():
    a = lc.build()["ssd_contention_answer"]
    assert 1.0 < a["measured_contention_cost"] < 1.5
    assert a["answer"].startswith("BARELY")
    assert "spend the effort on residency instead" in a["so_the_scheduling_rule_is"]


def test_the_prediction_is_scored_on_a_specimen_it_was_not_fitted_to():
    s = lc.scored_prediction()
    assert s["fitted_on"] != s["scored_on"]
    assert s["within_10pct"] is True
    assert s["relative_error"] < 0.10
    assert "would measure nothing" in s["why_this_is_a_real_score"]


def test_one_held_out_point_is_not_called_a_validation():
    s = lc.scored_prediction()
    assert s["n_held_out_points"] == 1
    assert "not a validated cost model" in s["one_point_is_not_a_validation"]


def test_per_specimen_costs_scale_with_source_bytes():
    rows = lc.per_specimen()
    assert rows == sorted(rows, key=lambda r: r["source_gb"])
    assert rows[-1]["cold_load_minutes"] > rows[0]["cold_load_minutes"]
    assert all(r["warm_load_seconds"] < r["cold_load_seconds"] for r in rows)


def test_the_prefetch_case_is_made_with_hours_not_adjectives():
    p = lc.prefetch_case()
    assert p["total_cold_load_hours"] > 1.0
    assert p["worst_single_specimen"]["cold_minutes"] > 60
    assert "DO NOT WAIT UNTIL A MODEL IS NEEDED" in p["statement"]
    assert "seal backlog is therefore a scheduling cost" in \
        p["and_this_is_the_sealed_subset_only"]


def test_every_work_species_declares_its_resource_and_reversal():
    for name, spec in lc.WORK_SPECIES.items():
        assert spec["resource"], name
        assert spec["memory_effect"], name
        assert spec["reversible_by"], name


def test_the_module_says_these_are_a_floor_not_the_load_time():
    w = lc.build()["what_is_not_measured_here"]
    assert "DISK READ only" in w
    assert "FLOOR on load cost" in w

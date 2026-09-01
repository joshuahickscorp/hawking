"""The hoist is slower, the input never mattered, and warmup was the whole story.

This obligation was wrong twice. These tests pin the guard that would have
caught both: an arm measured outside steady state is refused, not averaged.
"""
from __future__ import annotations

import json

import pytest

from tools.future import dequant_hoist_ab as ab


def test_both_runs_are_at_adequate_warmup_and_in_steady_state():
    m = ab.measured()
    for run in (m["real_activation_run"], m["synthetic_control_run"]):
        assert run["warmup"] >= ab.MIN_WARMUP
        assert run["production_rep_spread"] < ab.STEADY_MAX_SPREAD


def test_an_underwarmed_receipt_is_refused_not_averaged(monkeypatch):
    d = json.loads((ab.REPO / ab.REAL_REL).read_text())
    monkeypatch.setattr(ab, "_raw", lambda rel: {**d, "warmup": 5})
    with pytest.raises(ab.AbRefused, match="coin flip between two modes"):
        ab.measured()


def test_a_bimodal_arm_is_refused_even_at_high_warmup(monkeypatch):
    d = json.loads((ab.REPO / ab.REAL_REL).read_text())
    bad = {**d, "mlp": {**d["mlp"], "production": {
        **d["mlp"]["production"], "rep_spread": 1.696}}}
    monkeypatch.setattr(ab, "_raw", lambda rel: bad)
    with pytest.raises(ab.AbRefused, match="rep spread 1.6960"):
        ab.measured()


def test_a_missing_arm_refuses():
    with pytest.raises(ab.AbRefused, match="not on disk"):
        ab._run("receipts/future/NO_SUCH.json", "x")


def test_the_hoist_is_slower_than_the_kernel_it_replaces():
    m = ab.measured()
    assert m["verdict"] == "SLOWER_THAN_PRODUCTION"
    assert m["headline_hoist_over_production"] < 1.0
    assert ab.build()["ms_per_token_saved"] == 0.0


def test_the_input_hypothesis_is_refuted_by_its_own_control():
    d = ab.measured()["the_input_does_not_matter"]
    assert d["relative_difference"] < 0.02, \
        "production reads the same on both inputs once it is warm"
    assert "REFUTED by this control" in d["reading"]


def test_the_real_cause_is_named_and_scoped_to_the_first_arm():
    w = ab.measured()["what_actually_moved"]
    assert w["cause"] == "INSUFFICIENT_WARMUP_ON_THE_FIRST_MEASURED_ARM"
    assert "252k" in w["signature"] and "428k" in w["signature"]
    assert "first arm timed" in w["why_production_specifically"]


def test_both_wrong_readings_are_kept_with_their_causes():
    h = ab.build()["correction_history"]
    assert [r["reading"] for r in h] == [1, 2, 3]
    assert "1.6338x" in h[0]["claim"]
    assert "synthetic input" in h[1]["claim"]
    assert all(r["cause"] for r in h)


def test_the_scar_names_the_receipt_it_hit_and_the_producer_fix():
    s = ab.scar()
    assert s["id"] == "WARMUP_5_LEAVES_THE_FIRST_MEASURED_ARM_OUTSIDE_STEADY_STATE"
    assert "MLP_ALU_ROOFLINE.json" in s["who_it_hit"]
    assert "1.696" in s["who_it_hit"] and "1.010" in s["who_it_hit"]
    assert "right by luck, not by construction" in s["who_it_hit"]
    assert "RECEIPT-ONLY FIXES ARE FORBIDDEN" in s["producer_fix"]


def test_the_fp_answer_stands_and_was_never_the_reason_to_reject():
    b = ab.bit_identity()
    assert b["all_exact"] is False
    assert b["max_rel_fro"] < 1e-6
    assert "never the reason to reject" in b["reading"]

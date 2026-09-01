"""The bitcast unpack is measured on the real graph, and the pair is matched.

A faster kernel that changes the output is a regression, and a faster kernel in
a different graph is not a comparison. Both refusals are pinned here.
"""
from __future__ import annotations

import json

import pytest

from tools.future import bitcast_dequant_ab as ab


def _ctrl():
    return json.loads((ab.REPO / ab.CTRL_REL).read_text())


def test_the_pair_is_matched_on_everything_but_the_unpack():
    m = ab.matched_pair()
    assert m["token_ids_identical"] is True
    assert m["n_tokens"] == 32
    assert m["reps"] == 9
    assert m["dispatches"] == 580
    assert m["dense_w_materialized"] == 0
    assert m["fallbacks"] == 0
    assert set(m["kernels_swapped"]) == set(ab.SWAPPED)


def test_a_token_divergence_refuses(monkeypatch):
    d = _ctrl()
    d["decode"][ab.LIVE_ARM]["new_token_ids"] = [1, 2, 3]
    monkeypatch.setattr(ab, "_raw",
                        lambda rel: d if rel == ab.CTRL_REL
                        else json.loads((ab.REPO / rel).read_text()))
    with pytest.raises(ab.AbRefused, match="not a win, it is a regression"):
        ab.matched_pair()


def test_a_changed_graph_refuses(monkeypatch):
    d = _ctrl()
    d["decode"][ab.LIVE_ARM]["theoretical_dispatches"] = 579
    monkeypatch.setattr(ab, "_raw",
                        lambda rel: d if rel == ab.CTRL_REL
                        else json.loads((ab.REPO / rel).read_text()))
    with pytest.raises(ab.AbRefused, match="not the same graph"):
        ab.matched_pair()


def test_a_fallback_refuses(monkeypatch):
    d = _ctrl()
    d["decode"][ab.LIVE_ARM]["fallbacks_reps"] = [0, 1, 0]
    monkeypatch.setattr(ab, "_raw",
                        lambda rel: d if rel == ab.CTRL_REL
                        else json.loads((ab.REPO / rel).read_text()))
    with pytest.raises(ab.AbRefused, match="took a fallback"):
        ab.matched_pair()


def test_an_extra_kernel_difference_refuses(monkeypatch):
    """If some other kernel also changed, the delta is not the unpack's."""
    d = _ctrl()
    d["decode"][ab.LIVE_ARM]["dispatched_kernels_rep0"].append("some_other_kernel")
    monkeypatch.setattr(ab, "_raw",
                        lambda rel: d if rel == ab.CTRL_REL
                        else json.loads((ab.REPO / rel).read_text()))
    with pytest.raises(ab.AbRefused, match="not exactly the two swapped unpacks"):
        ab.matched_pair()


def test_a_dense_w_materialization_refuses(monkeypatch):
    d = _ctrl()
    d["decode"][ab.LIVE_ARM]["dense_w_materialized"] = 1
    monkeypatch.setattr(ab, "_raw",
                        lambda rel: d if rel == ab.CTRL_REL
                        else json.loads((ab.REPO / rel).read_text()))
    with pytest.raises(ab.AbRefused, match="not in-register dequant"):
        ab.matched_pair()


def test_the_saving_is_material_and_both_arms_are_settled():
    t = ab.timing()
    assert t["ms_saved"] > 3.0
    assert t["speedup"] > 1.15
    assert t["control_rep_spread"] < 1.02
    assert t["bitcast_rep_spread"] < 1.02


def test_the_complete_token_crosses_forty_and_says_what_remains():
    c = ab.complete_token()
    assert c["tps_after"] > c["tps_before"]
    assert c["checkpoint_crossed"] == "40 TPS"
    assert c["still_short_of_60_by_ms"] > 0, "60 is not claimed"
    assert "CONSTRAINS, not one it proves" in c["why_the_host_gap_carries"]


def test_the_window_is_declared_contaminated_and_the_absolute_is_not_claimed():
    b = ab.claim_boundary()
    assert b["evidence_class"] == "DIAGNOSTIC_RELATIVE"
    assert "had a protected window" in b["what_is_not"]
    assert "no single absolute here is promotable" in b["what_is_not"]
    assert "resident_reprofile.py" in b["what_would_promote_it"]


def test_the_fp_boundary_separates_weight_identity_from_token_identity():
    f = ab.fp_boundary()
    assert f["bit_identical_weights"] is False
    assert f["token_identical"] is True
    assert "not a proof of identity over all inputs" in f["reading"]
    assert f["risk"]


def test_the_default_is_unchanged():
    """A lever that changes production by existing is not a lever."""
    assert ab.build()["default_is_unchanged"] is True
    assert "HAWKING_AFFINE2_GEO=bitcast" in ab.build()["lever"]


def test_two_independent_windows_agree_on_the_ratio():
    r = ab.independent_runs()
    assert len(r["runs"]) == 2
    assert r["gpu_speedup_spread"] < 0.01, \
        "the ratio must be stable across contention windows or it is not a ratio"
    lo, hi = r["bitcast_wall_tps_range"]
    assert lo > 40.0, "both windows cross the 40 TPS checkpoint"
    assert "measured here rather than assumed" in r["reading"]


def test_a_token_divergence_in_the_second_run_also_refuses(monkeypatch):
    d = json.loads((ab.REPO / ab.CTRL2_REL).read_text())
    d["decode"][ab.LIVE_ARM]["new_token_ids"] = [9, 9]
    real = ab._raw
    monkeypatch.setattr(ab, "_raw",
                        lambda rel: d if rel == ab.CTRL2_REL else real(rel))
    with pytest.raises(ab.AbRefused, match="run_2: the arms produced different"):
        ab.independent_runs()


def test_the_isolated_projection_understated_the_graph_and_says_so():
    p = ab.projection_vs_graph()
    assert p["graph_over_prediction"] > 1.0
    assert p["measured_ms_saved_in_the_graph"] > p["predicted_ms_saved"]
    assert "LOWER bound" in p["reading"]
    assert "Two observations are not a law" in p["do_not_invert_this"]


def test_the_absolute_is_reported_as_a_range_not_a_number():
    b = ab.claim_boundary()
    assert "RESPAWNED them" in b["what_is_not"]
    assert "a range, not a number" in b["what_is_not"]

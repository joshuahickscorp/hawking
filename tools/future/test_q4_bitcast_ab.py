"""q4's own ladder, because the scar says the split is kernel-specific.

The headline claims bit-identity, so the module must refuse to publish if the
arm is not bit-identical rather than quietly reporting an epsilon.
"""
from __future__ import annotations

import json

import pytest

from tools.future import q4_bitcast_ab as q4


def test_all_three_arms_ran_and_are_steady():
    a = q4.arms()
    for name in ("production", "bitcast", "arm_a_stripped"):
        assert a[name]["rep_spread"] < q4.STEADY_MAX_SPREAD


def test_an_unsteady_arm_refuses(monkeypatch):
    d = json.loads((q4.REPO / q4.RAW_REL).read_text())
    d["deltanet"]["production"]["rep_spread"] = 1.7
    monkeypatch.setattr(q4, "_raw", lambda: d)
    with pytest.raises(q4.Q4Refused, match="not in steady state"):
        q4.arms()


def test_a_missing_raw_refuses(monkeypatch):
    monkeypatch.setattr(q4, "RAW_REL", "receipts/future/NO_SUCH.json")
    with pytest.raises(q4.Q4Refused, match="not on disk"):
        q4.arms()


def test_the_candidate_is_faster_and_recovers_part_of_the_span():
    m = q4.measured()
    assert m["speedup"] > 1.10
    assert 0.0 < m["span_recovered"] < 1.0
    assert m["arm_a_over_production"] > 1.4


def test_the_bit_identity_claim_is_enforced_not_asserted(monkeypatch):
    """The docstring headline says BIT-IDENTICAL; publishing must depend on it."""
    m = q4.measured()
    assert m["bit_identical"] is True
    assert m["rel_fro"] == 0.0
    assert m["max_abs_err"] == 0.0

    d = json.loads((q4.REPO / q4.RAW_REL).read_text())
    d["deltanet"]["bitcast"]["output_compare"]["bit_identical"] = False
    d["deltanet"]["bitcast"]["output_compare"]["rel_fro"] = 1e-7
    monkeypatch.setattr(q4, "_raw", lambda: d)
    with pytest.raises(q4.Q4Refused, match="must be rewritten before it is published"):
        q4.measured()


def test_it_explains_why_q4_is_exact_and_q2_is_not():
    m = q4.measured()
    assert "contracts that to an FMA" in m["why_bit_identical_here_and_not_on_q2"]


def test_lm_head_is_excluded_from_the_projection_with_a_reason():
    t = q4.token_projection()
    assert "lm_head" not in t["organs_this_kernel_runs"]
    assert "different kernel this candidate does not touch" in t["lm_head_excluded_because"]


def test_the_projection_is_prospective_and_refuses_to_be_scaled_up():
    t = q4.token_projection()
    assert t["evidence_class"] == "PROSPECTIVE"
    assert t["below_the_materiality_threshold"] is True, \
        "counting DeltaNet by its in-projection put this under the 1 ms bar"
    assert "governs what to START, not what to discard" in t["why_keep_it_anyway"]
    assert "Two over, one under" in t["the_isolated_number_is_not_a_bound_in_either_direction"]


def test_the_scar_is_named_as_obeyed_not_cited_decoratively():
    b = q4.build()
    assert "kernel-specific" in b["the_scar_was_obeyed"]
    assert "measured q4's own production" in b["the_scar_was_obeyed"]


def test_the_second_fast_and_wrong_build_is_recorded():
    b = q4.build()["the_bug_that_the_output_compare_caught"]
    assert "DISCARDS BITS 13" in b
    assert "1.40x" in b and "0.877" in b
    assert "Second time" in b


def test_the_default_is_unchanged():
    assert q4.build()["default_is_unchanged"] is True
    assert "HAWKING_Q4_UNPACK=bitcast" in q4.build()["lever"]


def test_the_resident_ab_is_matched_and_token_identical():
    r = q4.resident_measured()
    assert r["both_arms_have_q2_bitcast_on"] is True, \
        "the difference must be the q4 unpack alone"
    assert r["token_identical"] is True
    assert r["dispatches"] == 580
    assert r["only_production_kernel_left"] == ["qwen_uniform_q4_embedding_lookup"]
    assert r["gpu_ms_saved"] > 0
    assert r["wall_tps_with"] > r["wall_tps_without"]


def test_a_leftover_production_matvec_refuses(monkeypatch):
    """A partial swap silently under-reports the candidate."""
    d = json.loads((q4.REPO / q4.RES_ON_REL).read_text())
    d["decode"][q4.LIVE_ARM]["dispatched_kernels_rep0"].append(
        "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128")
    monkeypatch.setattr(q4.json, "loads",
                        lambda s, _r=json.loads: d if '"decode"' in s else _r(s))
    with pytest.raises(q4.Q4Refused, match="still dispatched with the lever on"):
        q4.resident_measured()


def test_the_lower_bound_pattern_is_recorded_as_falsified():
    p = q4.projection_vs_graph()
    assert p["the_lower_bound_pattern_did_not_hold"] is True
    assert "Three observations, two directions" in p["reading"]
    assert "There is no bound" in p["reading"]
    assert "not established" in p["why_it_may_differ"], \
        "the cause is a hypothesis, not a finding"


def test_the_window_is_declared_unprotected():
    r = q4.resident_measured()
    assert r["evidence_class"] == "DIAGNOSTIC_RELATIVE"
    assert "not promotable" in r["window"]

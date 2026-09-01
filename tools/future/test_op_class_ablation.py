"""The arithmetic splits four ways, and the convert is the biggest piece.

The ladder's whole value is that it says WHICH class to attack. These tests pin
the decomposition's arithmetic and the guard that refuses an unsteady arm.
"""
from __future__ import annotations

import json

import pytest

from tools.future import op_class_ablation as oc


def test_every_arm_ran_and_was_in_steady_state():
    a = oc.arms()
    for name in ("production", "noaffine", "noconv", "nounpack",
                 "arm_a_stripped", "bitcast"):
        assert a[name]["steady_state"] is True
        assert a[name]["rep_spread"] < oc.STEADY_MAX_SPREAD


def test_an_unsteady_arm_is_refused(monkeypatch):
    d = json.loads((oc.REPO / oc.RAW_REL).read_text())
    bad = {**d, "mlp": {**d["mlp"], "production": {
        **d["mlp"]["production"], "rep_spread": 1.7}}}
    monkeypatch.setattr(oc, "_raw", lambda: bad)
    with pytest.raises(oc.AblationRefused, match="not in steady state"):
        oc.arms()


def test_a_missing_arm_refuses(monkeypatch):
    d = json.loads((oc.REPO / oc.RAW_REL).read_text())
    bad = {**d, "mlp": {**d["mlp"], "op_class_ablations": []}}
    monkeypatch.setattr(oc, "_raw", lambda: bad)
    with pytest.raises(oc.AblationRefused, match="ladder is incomplete"):
        oc.arms()


def test_the_ablations_are_flagged_as_not_computing_the_right_answer():
    a = oc.arms()
    for name in ("noaffine", "noconv", "nounpack"):
        assert a[name]["computes_the_right_answer"] is False
    assert a["bitcast"]["computes_the_right_answer"] is True


def test_the_convert_is_the_largest_class_and_none_dominates():
    d = oc.decomposition()
    assert d["largest_class"] == "uint_to_float_convert"
    assert d["no_single_class_dominates"] is True
    shares = {k: v["share_of_arithmetic"] for k, v in d["classes"].items()}
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-3)
    assert shares["uint_to_float_convert"] > shares["shift_and_mask"]
    assert shares["uint_to_float_convert"] > shares["both_fmas"]


def test_the_affine_fma_the_hoist_attacked_is_a_rounding_error():
    d = oc.decomposition()
    assert d["affine_fma_alone"]["share_of_arithmetic"] < 0.03
    assert "0.6% of its time" in d["why_the_hoist_bought_nothing"]


def test_a_negative_convert_residual_refuses_rather_than_reporting(monkeypatch):
    """If the classes are not additive the model is wrong, not the number."""
    a = oc.arms()
    # noconv keeps only shift+mask, so it cannot be SLOWER than production.
    bad = {**a, "noconv": {**a["noconv"], "effective_gb_s": 200.0}}
    monkeypatch.setattr(oc, "arms", lambda: bad)
    with pytest.raises(oc.AblationRefused, match="do not decompose additively"):
        oc.decomposition()



def test_the_bitcast_candidate_is_both_fast_and_correct():
    b = oc.bitcast_candidate()
    assert b["speedup_over_production"] > 1.20
    assert b["bit_identical"] is False, "refolded constants; exactness not expected"
    assert b["max_rel_fro"] < 1e-6, "but the error must sit at f32 epsilon"


def test_the_fast_and_wrong_first_version_is_recorded():
    b = oc.bitcast_candidate()
    assert "1.225x" in b["the_first_version_was_fast_and_wrong"]
    assert "rel_fro 1.26" in b["the_first_version_was_fast_and_wrong"]


def test_the_token_projection_is_labelled_prospective_with_its_promotion_path():
    t = oc.token_projection()
    assert t["evidence_class"] == "PROSPECTIVE"
    assert t["tps_after"] > t["tps_before"]
    assert t["ms_saved_if_it_lands"] > 1.0, "material by the S025 threshold"
    assert "has not yet been changed in the resident" in t["why_prospective"]
    assert "complete-token wall TPS" in t["what_would_make_it_measured"]


def test_the_scar_forbids_attacking_a_class_before_measuring_the_split():
    s = oc.build()["scar"]
    assert s["id"] == "REMOVING_ONE_OP_CLASS_IS_NOT_A_LEVER_WHEN_FOUR_SHARE_THE_COST"
    assert "MEASURE THE SPLIT BEFORE CHOOSING THE TARGET" in s["statement"]
    assert "q4 DeltaNet matvec" in s["reopen"], "the split is kernel-specific"

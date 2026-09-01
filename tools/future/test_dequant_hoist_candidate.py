"""A cheaper decode by algebra, not by a new format - and it is algebra only.

Every decode this campaign built changed WHERE the scale lives and made the tax
worse: LUT 2.0, native exp 2.5, incumbent 1.3333. This changes WHEN the affine
is applied, using an identity that is exact in exact arithmetic:

    sum_i (s*c_i + b) * x_i  ==  s * sum_i(c_i*x_i) + b * sum_i(x_i)

No kernel exists. Nothing was timed. An FMA count is not a GB/s.
"""
from __future__ import annotations

import pytest

from tools.future import dequant_hoist_candidate as dh


def test_the_incumbent_inner_loop_must_reconcile_before_algebra_is_built_on_it():
    acc = dh.accounting()
    inc = acc["incumbent"]
    assert (inc["dequant_fma"] + inc["mac_fma"]) / acc["weight_bytes_per_iteration"] == (
        pytest.approx(inc["fma_per_weight_byte"], abs=5e-4)
    )


def test_an_inner_loop_that_does_not_add_up_is_refused(monkeypatch, tmp_path):
    bad = tmp_path / "roof.json"
    bad.write_text(
        '{"mlp": {"decode_tax": {"inner_loop": {"weights_per_iteration": 8, '
        '"weight_bytes_per_iteration": 6, "dequant_fma": 8, "mac_fma": 8, '
        '"int_to_float": 8, "bitops": 16, "fma_per_weight_byte": 99.0}}}}'
    )
    monkeypatch.setattr(dh, "REPO", tmp_path)
    monkeypatch.setattr(dh, "ROOFLINE_REL", "roof.json")
    with pytest.raises(dh.CandidateRefused, match="does not reconcile"):
        dh.accounting()


def test_the_affine_is_amortised_over_the_CHUNK_not_the_group():
    """The kernel strides col += 512, so a thread handles one 8-weight chunk of a
    group and never returns to it. scale and bias are constant within the CHUNK.

    Assuming the group was how the first version of this receipt claimed 1.939x
    while describing the un-retiled loop.
    """
    acc = dh.accounting()
    assert acc["folded"]["dequant_fma"] == pytest.approx(2.0)
    assert acc["decode_cheapening"] == pytest.approx(4.0, abs=0.1)
    assert "col += 512" in acc["amortised_over"]


def test_the_retiling_variant_is_named_as_a_DIFFERENT_candidate():
    acc = dh.accounting()
    r = acc["a_retiling_would_do_better"]
    assert r["total_arithmetic_cheapening"] > acc["total_arithmetic_cheapening"]
    assert "DIFFERENT candidate" in r["but"]
    assert "did not state" in r["but"]


def test_it_clears_the_required_cheapening():
    doc = dh.build()
    m = doc["meets_the_requirement"]
    assert m["required_total_cheapening"] == 1.509
    assert m["offered_total_cheapening"] == pytest.approx(1.600, abs=1e-3)
    assert m["clears"] is True


def test_the_conversions_are_NOT_claimed_as_a_saving():
    """int_to_float stays. Claiming it would be the easiest overstatement."""
    acc = dh.accounting()
    assert acc["folded"]["int_to_float"] == acc["incumbent"]["int_to_float"]
    assert acc["folded"]["bitops"] == acc["incumbent"]["bitops"]
    assert "not the conversions" in acc["int_to_float_unchanged_because"]


def test_the_hoist_is_justified_by_x_being_row_independent():
    i = dh.identity()
    assert "NOT of the output row" in i["why_sum_x_is_free"]
    assert "17408" in i["why_sum_x_is_free"]


def test_it_inherits_the_fold_addqx_bar_rather_than_ignoring_it():
    """Summation order changes, so bit-identity is lost. That bar is named."""
    r = dh.risks()
    assert "fold_addqx" in r["not_bit_identical"]
    assert "PASS_JUSTIFIED_TOLERANCE" in r["not_bit_identical"]
    assert "22309" in r["not_bit_identical"]


def test_every_risk_is_named_including_the_unpredictable_one():
    r = dh.risks()
    for key in ("not_bit_identical", "precision", "register_pressure", "expressibility"):
        assert len(r[key]) > 60, key
    assert "UNTESTED" in r["precision"]
    assert "registers_per_thread is" in r["register_pressure"]
    assert r["status"] == "UNBUILT, UNMEASURED, arithmetic argument only"


def test_the_boundary_says_an_fma_count_is_not_a_gb_s():
    cb = dh.build()["claim_boundary"]
    assert "an FMA count is not a GB/s" in cb
    assert "candidate worth building, not a result" in cb


def test_the_candidate_is_bracketed_on_the_MEASURED_ladder_not_guessed():
    """The issue-rate ladder already swept FMA/byte on the real kernel."""
    b = dh.ladder_bracket()
    assert b["candidate_fma_per_weight_byte"] == pytest.approx(1.6667, abs=1e-3)
    assert b["bracketed_by"] == ["k6", "k4"]
    lo, hi = b["speedup_bracket"]
    assert lo == pytest.approx(1.2679, abs=1e-3)
    assert hi == pytest.approx(1.4256, abs=1e-3)


def test_it_is_a_bracket_and_not_an_interpolated_point():
    """The ladder's own verdict is that its shape is neither linear nor plateau."""
    b = dh.ladder_bracket()
    assert "false precision" in b["not_interpolated_because"]
    assert "0.8712" in b["not_interpolated_because"]
    lo, hi = b["mlp_ms_saved_bracket"]
    assert lo < hi, "a bracket must have two ends"
    assert lo == pytest.approx(3.3064, abs=1e-2)


def test_the_knee_is_named_so_nobody_chases_past_it():
    """k4 439.5 and k2 440.6 - halving the FMA again buys 0.25%."""
    knee = dh.ladder_bracket()["the_knee"]
    assert "439.5" in knee and "440.6" in knee
    assert "0.25%" in knee
    assert "just above the knee" in knee


def test_the_ladder_arms_are_not_bit_identical_and_neither_is_this():
    b = dh.ladder_bracket()
    assert "not-identical" in b["ladder_arms_are_not_bit_identical"]

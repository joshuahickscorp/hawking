"""The largest target on record is arithmetic, and it is a CEILING not a yield.

ARM A strips arithmetic at identical bytes and buys 1.51x; the ILP ladder makes
the same arithmetic parallel and buys nothing. So the cost is throughput, and
the roofline says half the inner-loop FMA is decode rather than compute.

The number that falls out is bigger than every representation lever combined,
which is exactly why it needs its caveat stated louder than itself.
"""
from __future__ import annotations

import pytest

from tools.future import decode_tax_target as dt


def test_half_the_inner_loop_is_decode():
    r = dt.requirement()
    assert r["production_decode_fma_per_weight_byte"] == pytest.approx(1.3333, abs=1e-3)
    assert r["production_fma_per_weight_byte"] == pytest.approx(2.6667, abs=1e-3)
    assert r["decode_share_of_inner_loop"] == pytest.approx(0.5, abs=0.01)


def test_the_requirement_is_a_specification_not_a_hope():
    r = dt.requirement()
    assert r["target_decode_fma_per_weight_byte"] == pytest.approx(0.8835, abs=1e-3)
    assert r["required_decode_cheapening"] == pytest.approx(1.509, abs=1e-3)
    assert "Not a promise that such a decode exists" in r["not_a_promise"]


def test_it_would_close_the_residual_gap():
    w = dt.worth()
    assert w["mlp_ms_saved"] == pytest.approx(5.2787, abs=1e-3)
    assert w["deltanet_ms_saved"] == pytest.approx(2.0313, abs=1e-3)
    assert w["total_ms_saved"] == pytest.approx(7.3099, abs=1e-3)
    assert w["closes_the_residual"] is True
    assert w["share_of_residual_gap"] > 1.0


def test_it_is_bigger_than_every_representation_lever_combined():
    rank = dt.against_everything_else()
    reps = (rank["entire_deltanet_representation_ms"]
            + rank["entire_auxiliary_school_ms"]
            + rank["mlp_entropy_floor_ms"])
    assert rank["decode_tax_at_arm_a_rate_ms"] > reps * 3


def test_the_catch_is_stated_and_is_the_load_bearing_part():
    """ARM A removes the decode ENTIRELY. No real representation can."""
    catch = dt.against_everything_else()["the_catch"]
    assert "removes the decode ENTIRELY" in catch
    assert "ceiling of a cheaper decode, not a candidate's yield" in catch
    assert "may not exist at this bit width" in catch


def test_the_auxiliary_school_is_priced_at_exactly_zero():
    """The lever the campaign spent weeks on. Kept in the ranking on purpose."""
    assert dt.against_everything_else()["entire_auxiliary_school_ms"] == 0.0


def test_nothing_is_measured_here_and_the_boundary_says_so():
    cb = dt.build()["claim_boundary"]
    assert "nothing is measured here" in cb
    assert "OPPORTUNITY BOUND" in cb
    assert "SELF_MEASURED_DIRTY" in cb and "RATIO is the usable part" in cb


def test_a_missing_receipt_refuses_rather_than_assuming_a_rate(monkeypatch):
    monkeypatch.setattr(dt, "ROOFLINE_REL", "receipts/future/NO_SUCH.json")
    with pytest.raises(dt.TargetRefused, match="read, not assumed"):
        dt.requirement()


def test_every_decode_tried_so_far_went_the_wrong_way():
    """Both alternative decodes cost MORE arithmetic than the incumbent."""
    p = dt.every_decode_tried_so_far()
    assert p["incumbent_decode_fma_per_weight_byte"] == pytest.approx(1.3333, abs=1e-3)
    for a in p["attempts"]:
        assert a["direction"] == "WORSE"
        assert a["decode_fma_per_weight_byte"] > p["incumbent_decode_fma_per_weight_byte"]
        assert a["source"].startswith("receipts/future/")


def test_the_target_is_below_anything_ever_built():
    p = dt.every_decode_tried_so_far()
    assert p["target"] < p["nothing_built_is_below"]
    assert p["nothing_built_is_below"] == pytest.approx(1.3333, abs=1e-3)


def test_the_law_names_the_trade_that_makes_it_negative():
    """Free bytes for binding arithmetic. This is aux_u8's slowdown explained."""
    law = dt.every_decode_tried_so_far()["law"]
    assert "FREE RESOURCE FOR THE BINDING ONE" in law
    assert "0.000 ms/GB" in law and "1.51x" in law
    assert "without a second theory" in law


def test_it_says_what_a_real_candidate_would_have_to_do():
    need = dt.every_decode_tried_so_far()["what_would_be_needed"]
    assert "cheaper than the incumbent" in need
    assert "not a different place to store the scale" in need

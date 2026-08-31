"""Price a perfect state machine before fitting three programs.

Oracle-first's economics half: IF AN UNFAIR ORACLE CANNOT MAKE THE ECONOMICS
WORK, DO NOT BUILD THE REAL VERSION. Three of the six S020 §16 families are
blocked on a fitted program each, and fitting three is real work - so the price
of perfect success comes first.
"""
from __future__ import annotations

import pytest

from tools.future import deltanet_state_machine_economics as ec


def test_the_byte_accounting_must_reconcile_before_it_is_priced():
    b = ec._bytes()
    assert b["weights"] == 2_961_659_904
    assert b["state"] == 150_994_944 + 5_898_240


def test_state_and_weights_are_priced_at_DIFFERENT_stream_rates():
    """The recurrent state is activation; the codes are weights. That is the
    whole reason for pricing them apart."""
    p = ec.price()
    assert p["rates_used"]["weight_codes"] == pytest.approx(0.547282)
    assert p["rates_used"]["activation"] == pytest.approx(2.906132)
    per_gb_weights = p["weights_ms_if_all_removed"] / p["weights_gb"]
    per_gb_state = p["state_ms_if_all_removed"] / p["state_gb"]
    assert per_gb_state > per_gb_weights * 4


def test_the_upper_bound_is_perfect_removal_not_a_speedup():
    p = ec.price()
    assert p["upper_bound_ms"] == pytest.approx(2.0768, abs=1e-3)
    assert p["halved_ms"] == pytest.approx(p["upper_bound_ms"] / 2, abs=1e-6)
    assert "OPPORTUNITY BOUND" in ec.build()["claim_boundary"]


def test_both_gaps_are_reported_not_the_flattering_one():
    p = ec.price()
    assert p["gap_to_71_raw_ms"] == pytest.approx(13.2051, abs=1e-3)
    assert p["gap_to_71_residual_after_everything_on_record_ms"] == pytest.approx(
        5.993, abs=1e-3
    )
    assert p["upper_bound_share_of_raw_gap"] < p["upper_bound_share_of_residual_gap"]
    assert "flattering" in p["two_gaps_because"]


def test_the_verdict_is_material_but_not_decisive():
    p = ec.price()
    assert p["verdict"] == "MATERIAL_NOT_DECISIVE"
    assert p["upper_bound_ms"] > p["material_bar_ms"]
    assert "nowhere near enough to close 71" in p["why"]


def test_it_says_what_the_byte_count_alone_would_have_claimed():
    lic = ec.what_this_licenses()
    assert "roughly 9 ms" in lic["and_the_byte_count_would_have_lied"]
    assert "BYTE_COUNT_TIMES_ORGAN_AVERAGE" in lic["and_the_byte_count_would_have_lied"]


def test_the_prior_from_the_two_fitted_families_is_carried():
    """Both families that HAVE been fitted came back MEASURED_NEGATIVE."""
    lic = ec.what_this_licenses()
    assert "MEASURED_NEGATIVE" in lic["but"]
    assert "not neutral" in lic["but"]


def test_it_ranks_itself_against_the_capacity_classes():
    lic = ec.what_this_licenses()
    assert "1.5x" in lic["ranking"] and "CONSTANT BYTES" in lic["ranking"]
    assert "no fit at all" in lic["ranking"]

"""FLASH_COMPLETE_EBPW_LE_1 cannot be reached by the lever being pulled at it.

The gate is the ONLY numeric acceptance spec in the 83-gate catalog, and the whole
FLASH program pushes one term at it: MLP density. Setting the entire MLP to zero
bytes -- a bound no compression scheme can beat, since stored weights cannot
occupy negative space -- still leaves complete_ebpw above the threshold.

These tests re-derive that from the committed ledger rather than asserting the
numbers, so if the body changes the claim changes with it instead of going stale.
"""
from __future__ import annotations

import pytest

from tools.future.complete_ebpw import bar_reachability


def test_the_bar_is_not_reachable_by_mlp_density_at_any_value():
    """The load-bearing claim. A free MLP is still not enough."""
    d = bar_reachability()
    assert d["reachable_by_mlp_density_alone"] is False
    zero = d["with_the_entire_mlp_at_zero_bytes"]
    assert zero["still_above_threshold"] is True
    assert zero["complete_ebpw"] > d["threshold"]


def test_the_arithmetic_is_self_consistent():
    """A ledger whose parts do not sum to its whole is not evidence of anything."""
    d = bar_reachability()
    parent = d["parent_params"]
    assert d["mlp"]["bytes"] + d["non_mlp"]["bytes"] == d["payload_bytes"]
    assert d["mlp"]["elements"] + d["non_mlp"]["params"] == parent
    # the report rounds to 6 decimals, so compare at that precision
    assert d["measured_complete_ebpw"] == pytest.approx(8.0 * d["payload_bytes"] / parent, abs=1e-6)
    assert d["budget_bytes_at_threshold"] == int(parent * d["threshold"] / 8.0)


def test_the_mlp_lever_is_monotone_and_bounded_below_by_the_zero_case():
    """Every density on the sensitivity curve must sit above the free-MLP floor."""
    d = bar_reachability()
    floor = d["with_the_entire_mlp_at_zero_bytes"]["complete_ebpw"]
    rows = d["mlp_density_sensitivity"]
    values = [r["complete_ebpw"] for r in rows]
    assert values == sorted(values, reverse=True), "denser MLP must not lower the total"
    assert all(v > floor for v in values)
    assert all(v > d["threshold"] for v in values), (
        "a sampled MLP density reaches the bar; the unreachability claim is wrong"
    )


def test_the_second_lever_is_named_and_is_below_q4():
    d = bar_reachability()
    lever = d["second_lever_required"]
    assert lever is not None
    assert lever["required_bpw_even_with_a_free_mlp"] < lever["current_bpw"]
    assert lever["required_bpw_even_with_a_free_mlp"] < 4.0, "Q4 is 4 bpw"


def test_a_threshold_that_is_reachable_reports_reachable():
    """The negative control: the analysis must not answer 'unreachable' always."""
    d = bar_reachability(threshold=8.0)
    assert d["reachable_by_mlp_density_alone"] is True
    assert d["second_lever_required"] is None
    assert d["with_the_entire_mlp_at_zero_bytes"]["still_above_threshold"] is False


def test_the_threshold_is_never_moved_by_this_analysis():
    """Redefining a bar after seeing a result is the failure this guards."""
    d = bar_reachability()
    assert d["threshold"] == 1.0
    assert d["threshold_untouched"] is True

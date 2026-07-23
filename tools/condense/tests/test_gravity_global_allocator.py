"""Tests for the Gravity global byte-allocator - enforce the budget, concentration, and
weak-domination invariants on synthetic organs. No real weights are touched."""
from __future__ import annotations

import os
import sys
from fractions import Fraction

import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gravity_global_allocator as ga  # noqa: E402


# ------------------------------------------------------------------------------------------------
# fixtures / helpers
# ------------------------------------------------------------------------------------------------

def _organs():
    return ga.default_gptoss_organs(scale=1)


def _mid_budget(organs):
    fb = ga.floor_bytes(organs)
    native = sum(o.bytes_at(ga.CANDIDATE_RATES[-1]) for o in organs)
    return fb + (native - fb) // 2


# ------------------------------------------------------------------------------------------------
# taxonomy / organ construction
# ------------------------------------------------------------------------------------------------

def test_taxonomy_covers_required_classes():
    names = {o.name for o in _organs()}
    required = {
        "embeddings", "lm_head_output_proj", "attn_q", "attn_k", "attn_v", "attn_o",
        "router", "mlp1_up_gate", "mlp2_down", "norms", "biases_scales",
        "shared_experts", "frequent_experts", "rare_experts", "runtime_metadata",
    }
    missing = required - names
    assert not missing, f"missing organ classes: {sorted(missing)}"


def test_organ_records_all_prior_fields():
    o = _organs()[0]
    for attr in ("param_count", "source_precision", "sensitivity", "activation_stat",
                 "router_or_expert_frequency", "representation_options", "doctor_reachable",
                 "min_protection_bpw", "quality_curve", "curve_points"):
        assert hasattr(o, attr)
    assert callable(o.quality_curve)
    assert all(isinstance(k, Fraction) for k in o.curve_points)


# ------------------------------------------------------------------------------------------------
# quality curve invariants
# ------------------------------------------------------------------------------------------------

def test_quality_curve_monotone_and_bounded():
    for o in _organs():
        vals = [o.quality_at(r) for r in ga.CANDIDATE_RATES]
        assert all(0.0 <= v <= 1.0 + 1e-9 for v in vals)
        for a, b in zip(vals, vals[1:]):
            assert b >= a - 1e-9, f"{o.name} quality not monotone in rate"


def test_quality_curve_concave_in_rate():
    """The genuine mathematical guarantee: Q(r) is concave in the retained fraction, so its exact
    secant slopes (no byte rounding involved) are non-increasing. This is what makes the greedy
    marginal allocator optimal and lets it weakly dominate a uniform baseline."""
    rates = ga.CANDIDATE_RATES
    for o in _organs():
        slopes = []
        for i in range(len(rates) - 1):
            dr = float(rates[i + 1] - rates[i])
            dq = o.quality_at(rates[i + 1]) - o.quality_at(rates[i])
            slopes.append(dq / dr)
        for a, b in zip(slopes, slopes[1:]):
            assert b <= a + 1e-9, f"{o.name} quality curve not concave in rate"


def test_marginal_utility_per_bit_non_increasing_up_to_rounding():
    """Byte-level marginal utility inherits the curve's concavity up to integer-byte quantization.
    Bracket gaps are unequal, so the ceil-to-whole-byte rounding injects sub-percent jitter; assert
    non-increasing within a rounding-aware relative tolerance rather than exactly."""
    rates = ga.CANDIDATE_RATES
    # only the non-pinned organs are ever stepped by the greedy; pinned/tiny organs have large
    # relative rounding jitter but are fixed at native and never allocated, so their discrete
    # byte-slopes are irrelevant to the allocator.
    for o in _organs():
        if o.pinned_native:
            continue
        us = []
        for i in range(len(rates) - 1):
            db = (o.bytes_at(rates[i + 1]) - o.bytes_at(rates[i])) * 8
            if db <= 0:
                continue
            us.append(ga.marginal_utility(o, rates[i], rates[i + 1], 1.0))
        for a, b in zip(us, us[1:]):
            assert b <= a * 1.02 + 1e-12, f"{o.name} marginal utility jumped beyond rounding jitter"


def test_sensitive_organ_curve_below_robust_at_low_rate():
    organs = {o.name: o for o in _organs()}
    lo = Fraction(1, 4)
    # mlp2/down (sensitive) recovers less at low rate than mlp1/up-gate (robust)
    assert organs["mlp2_down"].quality_at(lo) < organs["mlp1_up_gate"].quality_at(lo)


# ------------------------------------------------------------------------------------------------
# byte accounting invariants
# ------------------------------------------------------------------------------------------------

def test_bytes_strictly_increase_with_rate():
    for o in _organs():
        prev = -1
        for r in ga.CANDIDATE_RATES:
            b = o.bytes_at(r)
            assert b > prev, f"{o.name} bytes not strictly increasing at rate {r}"
            prev = b


def test_metadata_charge_is_never_free():
    for o in _organs():
        # even at the smallest rate the fixed metadata charge is present
        assert o.bytes_at(ga.CANDIDATE_RATES[0]) >= ga._METADATA_BYTES


def test_pinned_organs_are_source_native():
    organs = {o.name: o for o in _organs()}
    for name in ("router", "norms", "biases_scales", "attn_q", "attn_k", "attn_v", "attn_o"):
        assert organs[name].pinned_native, f"{name} should be pinned source-native by its floor"


# ------------------------------------------------------------------------------------------------
# greedy allocator: budget proof (invariant a)
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("frac", [0.10, 0.25, 0.5, 0.75, 0.9])
def test_greedy_respects_budget(frac):
    organs = _organs()
    fb = ga.floor_bytes(organs)
    native = sum(o.bytes_at(ga.CANDIDATE_RATES[-1]) for o in organs)
    budget = fb + int((native - fb) * frac)
    alloc = ga.greedy_allocate(organs, budget)
    assert alloc.total_bytes <= budget
    assert alloc.proof["within_budget"] is True
    assert alloc.proof["slack_bytes"] == budget - alloc.total_bytes
    assert sum(o.bytes for o in alloc.organs) == alloc.total_bytes


def test_budget_below_floor_raises():
    organs = _organs()
    fb = ga.floor_bytes(organs)
    with pytest.raises(ValueError):
        ga.greedy_allocate(organs, fb - 1)


def test_generous_budget_pushes_organs_up_but_never_over():
    organs = _organs()
    native = sum(o.bytes_at(ga.CANDIDATE_RATES[-1]) for o in organs)
    alloc = ga.greedy_allocate(organs, native)
    assert alloc.total_bytes <= native
    # with the full native budget every organ can reach 1/1
    assert all(a.rate == ga.CANDIDATE_RATES[-1] for a in alloc.organs)


# ------------------------------------------------------------------------------------------------
# concentration (invariant b)
# ------------------------------------------------------------------------------------------------

def test_bytes_concentrate_on_high_utility_organs():
    organs = _organs()
    budget = _mid_budget(organs)
    alloc = ga.greedy_allocate(organs, budget)
    conc = ga.concentration_report(organs, alloc)
    assert conc["total_discretionary_bytes"] > 0
    assert conc["top_half_share"] >= 0.5


def test_low_frequency_experts_get_no_more_than_frequent():
    organs = _organs()
    budget = _mid_budget(organs)
    alloc = ga.greedy_allocate(organs, budget)
    assert alloc.by_name("frequent_experts").rate >= alloc.by_name("rare_experts").rate


def test_sensitive_mlp2_protected_at_least_as_much_as_robust_mlp1():
    organs = _organs()
    budget = _mid_budget(organs)
    alloc = ga.greedy_allocate(organs, budget)
    assert alloc.by_name("mlp2_down").rate >= alloc.by_name("mlp1_up_gate").rate


# ------------------------------------------------------------------------------------------------
# weak domination over uniform (invariant c)
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("frac", [0.10, 0.20, 0.35, 0.5, 0.65, 0.8])
def test_greedy_weakly_dominates_uniform(frac):
    organs = _organs()
    fb = ga.floor_bytes(organs)
    native = sum(o.bytes_at(ga.CANDIDATE_RATES[-1]) for o in organs)
    budget = fb + int((native - fb) * frac)
    greedy = ga.greedy_allocate(organs, budget)
    uniform = ga.uniform_allocate(organs, budget)
    assert uniform.total_bytes <= budget
    assert greedy.projected_quality >= uniform.projected_quality - 1e-9


def test_greedy_strictly_beats_uniform_somewhere():
    """At some mid budget the non-uniform allocation should be strictly better, otherwise the
    whole premise (uniform BPW is suboptimal) would be empty on this organ set."""
    organs = _organs()
    fb = ga.floor_bytes(organs)
    native = sum(o.bytes_at(ga.CANDIDATE_RATES[-1]) for o in organs)
    strictly_better = False
    for frac in (0.15, 0.25, 0.35, 0.45, 0.55, 0.65):
        budget = fb + int((native - fb) * frac)
        g = ga.greedy_allocate(organs, budget)
        u = ga.uniform_allocate(organs, budget)
        if g.projected_quality > u.projected_quality + 1e-9:
            strictly_better = True
            break
    assert strictly_better


# ------------------------------------------------------------------------------------------------
# marginal utility
# ------------------------------------------------------------------------------------------------

def test_marginal_utility_zero_when_no_bits_added():
    o = _organs()[0]
    top = ga.CANDIDATE_RATES[-1]
    assert ga.marginal_utility(o, top, top, 1.0) == 0.0


def test_allocation_serializes():
    organs = _organs()
    alloc = ga.greedy_allocate(organs, _mid_budget(organs))
    d = alloc.to_dict()
    assert d["schema"] == ga.ALLOCATOR_SCHEMA
    assert d["total_bytes"] <= d["budget_bytes"]
    assert len(d["organs"]) == len(organs)
    # rates round-trip as exact-rational strings
    for row in d["organs"]:
        assert Fraction(row["rate"]) in ga.CANDIDATE_RATES


def test_demo_runs_green():
    assert ga.run_demo() == 0

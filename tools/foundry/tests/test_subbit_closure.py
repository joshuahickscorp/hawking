#!/usr/bin/env python3.12
"""Tests for the sub-bit closure program (tools/foundry/subbit_closure.py).

Four things must hold or the program is not science:
  1. every proposed variant is at or under the one-bit ceiling;
  2. a variant that would exceed 1.0 is REJECTED, including the historical A1_1p0;
  3. capability-tier compute is refused for a sub-1-bit probe until a one-bit method
     has been selected;
  4. the BPW arithmetic is exact rational, never a rounded decimal.
"""
import pathlib
import sys
from fractions import Fraction

FOUNDRY = pathlib.Path(__file__).resolve().parents[1]
if str(FOUNDRY) not in sys.path:
    sys.path.insert(0, str(FOUNDRY))

import pytest  # noqa: E402

import subbit_closure as sc  # noqa: E402

INV = sc.build_inventory()


# ── inventory ─────────────────────────────────────────────────────────────────
def test_inventory_matches_the_real_parent_byte_for_byte():
    assert INV.params == sc.SEALED_PARAMS
    assert INV.params * 2 == sc.SEALED_TOTAL_SIZE_BYTES
    assert INV.tensors == sc.SEALED_TENSOR_COUNT
    assert sum(o.params for o in INV.organs.values()) == INV.params


def test_inventory_cross_check_rejects_a_wrong_denominator():
    bad = {"metadata": {"total_size": sc.SEALED_TOTAL_SIZE_BYTES + 2}, "weight_map": {}}
    with pytest.raises(sc.ClosureError, match="cross-check failed"):
        sc.build_inventory(None, bad)


def test_every_tensor_must_be_allocated_exactly_once():
    half = sc.Variant(
        name="half_allocated", pressure_taken_from="n/a", rationale="n/a",
        bands=(sc.Band("norms", 10, sc.Native()),))
    with pytest.raises(sc.ClosureError, match="exactly once"):
        sc.bill(half, INV)


# ── 1. every variant passes the ceiling ───────────────────────────────────────
@pytest.mark.parametrize("variant", sc.VARIANTS, ids=lambda v: v.name)
def test_variant_is_legal_under_the_ceiling(variant):
    receipt = sc.check_ceiling(variant, INV)
    bpw = Fraction(receipt["complete_bits"], INV.params)
    assert bpw <= sc.CEILING
    assert receipt["legal"] is True
    assert receipt["headroom_bits"] >= 0


def test_there_are_at_least_three_legal_variants_taking_pressure_in_different_places():
    places = {v.pressure_taken_from for v in sc.VARIANTS}
    assert len(sc.VARIANTS) >= 3
    assert len(places) == len(sc.VARIANTS)


def test_nothing_is_excluded_as_overhead():
    """Every named slot is present and the total is their exact sum plus the reserve."""
    receipt = sc.bill(sc.V1, INV)
    slots = receipt["components_bits"]
    assert set(slots) == {
        "indices", "codebooks", "scales", "metadata", "alignment", "protected_islands",
        "doctor", "pass_through_tensors", "packaging", "runtime_tables"}
    assert receipt["complete_bits"] == sum(slots.values()) + receipt["reserve_bits"]
    # the organs that would be easiest to hide are all billed
    for slot in ("codebooks", "scales", "metadata", "alignment", "packaging",
                 "pass_through_tensors"):
        assert slots[slot] > 0, slot


def test_expert_only_bpw_is_never_the_model_rate():
    receipt = sc.bill(sc.V1, INV)
    expert_bits = sum(receipt["per_organ"][o]["bits"]
                      for o in ("expert_gate", "expert_up", "expert_down"))
    expert_params = sum(INV.organ(o).params
                        for o in ("expert_gate", "expert_up", "expert_down"))
    expert_only = Fraction(expert_bits, expert_params)
    whole = Fraction(receipt["complete_bits"], INV.params)
    assert expert_only < whole                      # the flattering number is the smaller one
    assert receipt["expert_only_bpw_is_not_the_model_rate"] is True


# ── 2. anything above 1.0 is rejected ─────────────────────────────────────────
def test_a1_1p0_replay_is_rejected():
    receipt = sc.bill(sc.A1_REPLAY, INV)
    assert receipt["legal"] is False
    assert Fraction(receipt["complete_bits"], INV.params) > sc.CEILING
    with pytest.raises(Exception) as exc:
        sc.check_ceiling(sc.A1_REPLAY, INV)
    assert "ceiling" in str(exc.value).lower()


def test_this_ledger_is_stricter_than_the_sealed_campaign_ledger():
    """A rebudget that is only legal under a slacker ruler is not a rebudget."""
    replay = Fraction(sc.bill(sc.A1_REPLAY, INV)["complete_bits"], INV.params)
    assert replay > sc.A1_SEALED_COMPLETE_BPW


def test_raising_any_expert_rate_one_notch_breaks_the_tightest_variant():
    """The budget is genuinely binding, not padded."""
    over = sc.Variant(
        name="C2_plus_one_notch", pressure_taken_from="test", rationale="test",
        bands=tuple(
            sc._experts(sc.VQ(16, 4, 64), sc.VQ(16, 4, 64), sc.VQ(32, 7, 4))
            + sc._nonexpert(sc.VQ(8, 1, 256), sc.VQ(8, 2, 256), sc.VQ(16, 3, 256))))
    with pytest.raises(Exception, match="(?i)ceiling"):
        sc.check_ceiling(over, INV)


def test_native_everything_is_wildly_illegal():
    native = sc.Variant(
        name="all_native", pressure_taken_from="nowhere", rationale="the parent itself",
        bands=tuple(sc.Band(n, o.tensors, sc.Native()) for n, o in INV.organs.items()))
    receipt = sc.bill(native, INV)
    assert receipt["complete_bpw_float"] > 16.0


# ── 3. the sub-bit scheduling rule ────────────────────────────────────────────
@pytest.mark.parametrize("rate", sc.SUBBIT_PROBE_RATES, ids=lambda q: f"{q.numerator}/{q.denominator}")
def test_subbit_probes_run_at_cheap_tiers_and_are_refused_at_expensive_ones(rate):
    for tier in sc.CHEAP_TIERS:
        assert sc.may_schedule(rate, tier)["allowed"] is True
    for tier in sc.EXPENSIVE_TIERS:
        verdict = sc.may_schedule(rate, tier)
        assert verdict["allowed"] is False
        assert "until a serious one-bit method is selected" in verdict["reason"]


def test_selecting_a_one_bit_method_unlocks_subbit_capability_compute():
    before = sc.may_schedule("1/2", "capability", one_bit_method_selected=False)
    after = sc.may_schedule("1/2", "capability", one_bit_method_selected=True)
    assert before["allowed"] is False
    assert after["allowed"] is True


def test_the_ceiling_rate_itself_may_spend_capability_compute():
    assert sc.may_schedule("1/1", "capability")["allowed"] is True


@pytest.mark.parametrize("rate", ["6/5", "5/4", "3/2", "2/1", "3/1"])
def test_no_tier_may_schedule_above_the_ceiling(rate):
    for tier in sc.FIDELITY_TIERS:
        verdict = sc.may_schedule(rate, tier, one_bit_method_selected=True)
        assert verdict["allowed"] is False
        assert "upward bracketing is REJECTED" in verdict["reason"]


def test_unknown_tier_is_refused_not_treated_as_cheap():
    assert sc.may_schedule("1/2", "vibes")["allowed"] is False


# ── 4. exact-fraction arithmetic ──────────────────────────────────────────────
def test_rates_are_exact_rationals_and_decimals_are_refused():
    with pytest.raises(sc.ClosureError, match="exact rational"):
        sc._parse_rate("0.85")
    assert sc._parse_rate("17/20") == Fraction(17, 20)


def test_vq_rate_is_exact_and_geometry_derived():
    assert sc.VQ(16, 4, 32).rate == Fraction(5, 4)
    assert sc.VQ(32, 1, 256).rate == Fraction(1, 4)
    assert sc.VQ(16, 2, 16).rate == Fraction(1, 2)
    with pytest.raises(sc.ClosureError):
        sc.VQ(16, 4, 30)            # k must be a power of two
    with pytest.raises(sc.ClosureError):
        sc.VQ(16, 17, 16)           # subspaces may not exceed dim


def test_complete_bpw_is_reported_as_an_exact_fraction():
    for variant in sc.VARIANTS:
        receipt = sc.bill(variant, INV)
        num, den = receipt["complete_bpw_exact"].split("/")
        exact = Fraction(int(num), int(den))
        assert exact == Fraction(receipt["complete_bits"], INV.params)
        assert exact.denominator > 1 or receipt["complete_bits"] % INV.params == 0
        # the float is a convenience only; the fraction is the identity
        assert abs(float(exact) - receipt["complete_bpw_float"]) < 1e-12


def test_bit_counts_are_integers_never_floats():
    receipt = sc.bill(sc.V5, INV)
    assert all(isinstance(v, int) for v in receipt["components_bits"].values())
    assert isinstance(receipt["complete_bits"], int)


def test_omission_does_not_shrink_the_denominator():
    receipt = sc.bill(sc.V4, INV)
    assert receipt["original_weight_count"] == INV.params
    gate = receipt["per_organ"]["expert_gate"]
    assert gate["params"] == INV.organ("expert_gate").params
    # surviving cells carry 1.5, higher than the illegal A1 rate, yet the whole model is legal
    assert gate["realized_organ_bpw_float"] < 1.5
    assert receipt["complete_bpw_float"] <= 1.0


# ── program assembly ──────────────────────────────────────────────────────────
def test_program_is_well_formed_and_orders_methods_decisive_first():
    prog = sc.build_program(INV)
    assert len(prog["legal_variants"]) == len(sc.VARIANTS)
    assert all(r["legal"] for r in prog["legal_variants"])
    assert any(not c["legal"] for c in prog["rejected_candidates"])
    ranks = [m["rank"] for m in prog["closure_methods"]]
    assert ranks == sorted(ranks) == list(range(1, len(ranks) + 1))
    assert prog["decisive_first_order"][0] == "M01_row_norm_stratified_codebooks"


def test_every_method_declares_the_five_required_fields():
    for m in sc.CLOSURE_METHODS:
        for key in ("changes", "why_not_falsified", "byte_cost", "first_tier",
                    "falsification", "changes_source"):
            assert m.get(key) not in (None, ""), (m["id"], key)
        assert m["first_tier"] in sc.FIDELITY_TIERS


def test_source_changing_methods_are_flagged():
    flagged = set(sc.source_changing_methods())
    assert {"M12_compressibility_training", "M13_quantization_aware_training",
            "M14_distillation_into_the_one_bit_student",
            "M15_learned_sharing_generated_weights",
            "M05_structured_expert_omission"} <= flagged
    assert "M01_row_norm_stratified_codebooks" not in flagged


def test_program_schedules_no_escape_hatch():
    prog = sc.build_program(INV)
    blob = str(prog).lower()
    for banned in ("escape receipt", "safety anchor", "quality anchor", "upward bracket"):
        # they may only appear in the FORBIDDEN list, never as a scheduled item
        assert banned not in str(prog["legal_variants"]).lower()
        assert banned not in str(prog["closure_methods"]).lower()
    assert "1.2" not in str([r["complete_bpw_float"] for r in prog["legal_variants"]])
    assert blob  # sanity


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

#!/usr/bin/env python3.12
"""Tests for the one-bit ceiling invariant (tools/foundry/one_bit_ceiling.py)."""
import pathlib
import sys
from fractions import Fraction

FOUNDRY = pathlib.Path(__file__).resolve().parents[1]
if str(FOUNDRY) not in sys.path:
    sys.path.insert(0, str(FOUNDRY))

import pytest  # noqa: E402

import one_bit_ceiling as obc  # noqa: E402

WEIGHTS = 1_000_000


def ledger(**over):
    """A complete ledger: every named slot declared, reserve declared, total 0 by default."""
    body = {c: 0 for c in obc.COMPONENTS}
    body[obc.RESERVE] = 0
    body.update(over)
    return obc.CompleteByteLedger(**body)


# ── the historical rejection ──────────────────────────────────────────────────
def test_a1_1p0_is_rejected_with_its_overage():
    with pytest.raises(obc.CeilingViolation) as exc:
        obc.assert_complete_bpw_le_one(obc.a1_1p0_ledger(), obc.A1_1P0_WEIGHTS)
    msg = str(exc.value)
    assert "1.007471652" in msg
    assert "overage 0.007471652 BPW = 1756537856 bits" in msg
    assert "do not raise the ceiling" in msg

    verdict = obc.a1_1p0_verdict()
    assert verdict["legal"] is False
    assert any("one-bit ceiling violated" in r for r in verdict["reasons"])


# ── completeness: undeclared is not zero ──────────────────────────────────────
def test_ledger_omitting_codebooks_is_incomplete_not_zero():
    fields = {c: 0 for c in obc.COMPONENTS if c != "codebooks"}
    fields[obc.RESERVE] = 0
    with pytest.raises(obc.IncompleteLedger, match="codebooks"):
        obc.CompleteByteLedger(**fields)


def test_missing_reserve_is_incomplete():
    with pytest.raises(obc.IncompleteLedger, match=obc.RESERVE):
        obc.CompleteByteLedger(**{c: 0 for c in obc.COMPONENTS})


def test_declared_but_unset_component_is_incomplete():
    with pytest.raises(obc.IncompleteLedger, match="undeclared is not zero"):
        ledger(doctor=None)


def test_unknown_component_is_refused():
    with pytest.raises(obc.IncompleteLedger, match="unknown ledger components"):
        ledger(misc_overhead=10)


def test_negative_component_cannot_pay_for_another():
    with pytest.raises(obc.IncompleteLedger, match="negative bits"):
        ledger(packaging=-8)


def test_incomplete_ledger_dict_fails_candidate_validation():
    fields = {c: 0 for c in obc.COMPONENTS if c != "codebooks"}
    fields[obc.RESERVE] = 0
    legal, reasons = obc.is_legal_candidate(
        {"original_weight_count": WEIGHTS, "ledger": fields}
    )
    assert legal is False
    assert "codebooks" in reasons[0]


def test_candidate_without_a_ledger_is_not_a_candidate():
    legal, reasons = obc.is_legal_candidate({"original_weight_count": WEIGHTS, "reported_bpw": "1/2"})
    assert legal is False
    assert "itemized ledger" in reasons[0]


# ── every one of the ten named components is billed ───────────────────────────
@pytest.mark.parametrize("component", obc.COMPONENTS)
def test_each_named_component_counts_against_the_ceiling(component):
    # the whole budget parked in this one slot: exactly 1.0 passes, one bit more does not
    assert obc.assert_complete_bpw_le_one(ledger(**{component: WEIGHTS}), WEIGHTS)["legal"] is True
    with pytest.raises(obc.CeilingViolation):
        obc.assert_complete_bpw_le_one(ledger(**{component: WEIGHTS + 1}), WEIGHTS)


def test_complete_bits_sums_all_eleven_slots():
    full = ledger(**{c: 3 for c in obc.COMPONENTS}, **{obc.RESERVE: 5})
    assert full.itemized_bits() == 3 * len(obc.COMPONENTS)
    assert full.complete_bits() == 3 * len(obc.COMPONENTS) + 5


# ── exact arithmetic ──────────────────────────────────────────────────────────
def test_one_plus_one_billionth_fails():
    w = 1_000_000_000
    over = ledger(indices=w + 1)
    assert over.complete_bpw(w) == Fraction(w + 1, w) == Fraction(1) + Fraction(1, 10 ** 9)
    with pytest.raises(obc.CeilingViolation):
        obc.assert_complete_bpw_le_one(over, w)

    # one bit over a budget where float() itself rounds the overage to exactly 1.0
    tiny = 10 ** 17
    assert float(Fraction(tiny + 1, tiny)) == 1.0  # a float gate would call this a tie
    with pytest.raises(obc.CeilingViolation):
        obc.assert_complete_bpw_le_one(ledger(indices=tiny + 1), tiny)


def test_float_overage_is_not_rounded_away():
    legal, reasons = obc.is_legal_candidate({
        "original_weight_count": WEIGHTS,
        "ledger": ledger(indices=WEIGHTS),
        "target_bpw": 1.0000001,
    })
    assert legal is False
    assert any("upward bracketing is REJECTED" in r for r in reasons)


def test_exactly_one_with_a_declared_reserve_passes():
    lg = ledger(indices=WEIGHTS - 4096, codebooks=3000, scales=1000, metadata=40,
                packaging=40, runtime_tables=16, **{obc.RESERVE: 0})
    assert lg.complete_bits() == WEIGHTS
    receipt = obc.assert_complete_bpw_le_one(lg, WEIGHTS)
    assert receipt["legal"] is True
    assert receipt["complete_bpw_exact"] == "1/1"
    legal, reasons = obc.is_legal_candidate({
        "candidate_id": "legal_at_one",
        "original_weight_count": WEIGHTS,
        "ledger": lg,
        "reported_bpw": "1/1",
        "target_bpw": "1/1",
    })
    assert (legal, reasons) == (True, [])


def test_reserve_is_charged_not_free():
    lg = ledger(indices=WEIGHTS, **{obc.RESERVE: 1})
    with pytest.raises(obc.CeilingViolation, match="overage"):
        obc.assert_complete_bpw_le_one(lg, WEIGHTS)
    legal, _ = obc.is_legal_candidate({
        "original_weight_count": WEIGHTS,
        "ledger": ledger(indices=WEIGHTS - 1, **{obc.RESERVE: 1}),
    })
    assert legal is True


def test_sub_bit_candidate_is_legal():
    lg = ledger(indices=WEIGHTS // 2, codebooks=1000, **{obc.RESERVE: 500})
    receipt = obc.assert_complete_bpw_le_one(lg, WEIGHTS)
    assert receipt["legal"] is True
    assert float(receipt["complete_bpw_float"]) < 0.51


# ── scope: no partial rate may stand in for the whole model ───────────────────
def test_expert_only_bpw_cannot_substitute_for_whole_model():
    lg = ledger(indices=WEIGHTS // 2, pass_through_tensors=WEIGHTS, **{obc.RESERVE: 0})
    legal, reasons = obc.is_legal_candidate({
        "candidate_id": "expert_only_claim",
        "original_weight_count": WEIGHTS,
        "ledger": lg,
        "expert_only_bpw": "1/2",
        "reported_bpw": "1/2",
    })
    assert legal is False
    assert any("understates the whole-model" in r for r in reasons)
    assert any("reported_bpw" in r and "!=" in r for r in reasons)
    assert any("one-bit ceiling violated" in r for r in reasons)


def test_partial_rate_without_a_whole_model_rate_is_refused():
    legal, reasons = obc.is_legal_candidate({
        "original_weight_count": WEIGHTS,
        "ledger": ledger(indices=WEIGHTS // 4, **{obc.RESERVE: 0}),
        "payload_only_bpw": "1/4",
    })
    assert legal is False
    assert any("may never stand in for the whole model" in r for r in reasons)


def test_non_whole_model_scope_is_refused():
    legal, reasons = obc.is_legal_candidate({
        "original_weight_count": WEIGHTS,
        "ledger": ledger(**{obc.RESERVE: 0}),
        "scope": "experts_only",
    })
    assert legal is False
    assert any("whole-model only" in r for r in reasons)


# ── forbidden anchors ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("anchor", ["6/5", "3/2", "2/1", "3/1"])
def test_upward_anchors_are_rejected(anchor):
    legal, reasons = obc.is_legal_candidate({
        "original_weight_count": WEIGHTS,
        "ledger": ledger(indices=WEIGHTS // 2, **{obc.RESERVE: 0}),
        "target_bpw": anchor,
    })
    assert legal is False
    assert any("upward bracketing is REJECTED" in r for r in reasons)


def test_zero_or_negative_weight_count_is_refused():
    with pytest.raises(obc.IncompleteLedger, match="must be positive"):
        ledger(**{obc.RESERVE: 0}).complete_bpw(0)

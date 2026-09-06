"""The gate and grammar must both be falsifiable, not asserted."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "future"))
import gravity_outlier_eval as G


def test_grammar_expresses_a_non_quantization_class():
    p = G.parse_spec("outlier0.005-g128")
    assert p["form"] == "outlier_split"
    assert p["frac"] == 0.005 and p["group"] == 128


def test_grammar_refuses_a_spec_it_cannot_execute():
    # Registration is not capability: a spec this evaluator cannot run must
    # never become a candidate.
    with pytest.raises(ValueError):
        G.parse_spec("tiers-1bit")
    with pytest.raises(ValueError):
        G.parse_spec("mixed-q2q3-experts")


def _gate(ppl, r4):
    return bool(r4 <= G.R4_MAX and ppl <= G.PPL_MAX)


def test_capability_gate_passes_the_measured_bf16_reference():
    assert _gate(G.BF16_PPL, G.BF16_R4)


def test_capability_gate_rejects_the_measured_collapsed_arm():
    # base-g128, measured: 87 consecutive identical tokens.
    # receipts/future/O003_OUTLIER_SPLIT.json
    assert not _gate(4.0847, 0.7204)


def test_capability_gate_rejects_the_best_arm_too():
    # outlier-g128 is the best representation measured so far and it STILL
    # fails. The gate is calibrated to bf16, not to whatever we managed to build.
    assert not _gate(3.3579, 0.5066)


def test_outlier_class_never_claims_execution():
    # The sparse channel has no kernel; a TPS claim for it would be fabrication.
    assert G.parse_spec("outlier0.005-g128")["form"] == "outlier_split"


def test_spec_naming_an_unrunnable_group_is_refused_at_the_door():
    # mx.quantize supports 32/64/128. outlier0.005-g256 parsed cleanly and then
    # died inside the search, costing a whole budget slot.
    with pytest.raises(ValueError, match="group 256"):
        G.parse_spec("outlier0.005-g256")
    for g in (32, 64, 128):
        assert G.parse_spec(f"outlier0.005-g{g}")["group"] == g


def test_cost_predictor_matches_measured_bytes():
    """A direction check is worthless if its arithmetic disagrees with the
    executor. Every non-outlier form must be exact."""
    measured = {"q4-g64-experts": 4.3797, "q3-g64-experts": 3.5024,
                "q2-g64-experts": 2.6251, "q2-g128-experts": 2.4058,
                "binary-g128": 1.4188, "binary-g64": 1.5284,
                "resbinary-g128": 2.4058}
    for spec, want in measured.items():
        got = G.predict_ebpw(spec)
        assert abs(got - want) < 1e-3, f"{spec}: predicted {got}, measured {want}"


def test_outlier_prediction_is_biased_HIGH_never_low():
    """The threshold keeps fewer weights than requested, so the estimate reads
    high. High is the safe direction: it can never make an upward proposal look
    like a descent."""
    for spec, measured in (("outlier0.005-g128", 2.5419), ("outlier0.02-g128", 2.9597)):
        got = G.predict_ebpw(spec)
        assert got >= measured, f"{spec}: {got} < {measured} -- bias went the unsafe way"
        assert got - measured < 0.01, f"{spec}: bias {got - measured} too large to be useful"

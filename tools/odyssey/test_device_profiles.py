"""G023 acceptance clause 4 pins."""
import json
from pathlib import Path

import pytest

RH = Path(__file__).resolve().parents[2] / "receipts/headless"
R = RH / "DEVICE_PROFILES.json"
pytestmark = pytest.mark.skipif(not R.is_file(), reason="profiles not qualified")


def rec():
    return json.load(open(R))


def test_two_profiles_were_qualified():
    d = rec()
    assert set(d["profiles"]) == {"INTERACTIVE", "MAXX"}
    for p in d["profiles"].values():
        assert p["winner"] is not None
    assert d["pass"] is True


def test_the_clause_is_answered_with_different_winners():
    """§71 asks for different winners OR an explicit finding that one dominates."""
    d = rec()
    assert d["clause_answer"]["different_winners"] is True
    assert d["profiles"]["INTERACTIVE"]["winner"] != d["profiles"]["MAXX"]["winner"]


def test_reproducibility_is_judged_per_metric_not_per_level():
    """A flat per-level gate failed everything: c1 aggregate carries model-load variance
    while TPOT repeats to 0.1%."""
    d = rec()
    assert "PER METRIC" in d["method"]
    r = d["reproducibility"]
    assert r["sealed-3.14"]["c1"]["tpot_ms"] < 1.0
    assert r["sealed-3.14"]["c1"]["aggregate_tps"] > 10.0


def test_each_profile_declares_the_metric_it_is_judged_on():
    for p in rec()["profiles"].values():
        assert p["headline_metric"] in ("tpot_ms", "aggregate_tps")
        assert p["why"]


def test_the_interactive_margin_exceeds_its_own_measurement_noise():
    p = rec()["profiles"]["INTERACTIVE"]
    assert p["margin_is_decisive"] is True
    for b, mm in p["per_body"].items():
        assert mm["tpot_ms"]["spread_pct"] < p["margin_pct"]


def test_a_body_that_does_not_reproduce_is_refused_qualification():
    """sealed's c4 aggregate swings 25.8% across three reps."""
    p = rec()["profiles"]["MAXX"]
    assert p["qualified"]["sealed-3.14"] is False
    assert p["qualified"]["variantB-2.76"] is True
    assert p["per_body"]["sealed-3.14"]["aggregate_tps"]["spread_pct"] > 15
    assert "does not repeat" in p["note"]


def test_every_cell_carries_its_reps_not_just_a_median():
    for b, lvls in rec()["profiles"]["MAXX"]["per_body"].items():
        for metric, v in lvls.items():
            assert len(v["reps"]) >= 3, (b, metric)


def test_the_mechanism_is_named_not_just_the_number():
    a = rec()["clause_answer"]
    assert "working set" in a["mechanism"]
    assert "admission ceiling" in a["mechanism"]


def test_it_does_not_claim_to_overturn_G040():
    """These are raw tokens, not verified WorkUnits."""
    b = rec()["bearing_on_G040"]
    assert b["does_it_overturn_the_selection"].startswith("NOT")
    assert "verified WorkUnits" in b["does_it_overturn_the_selection"]
    assert "unmeasured" in b["honest_status"]

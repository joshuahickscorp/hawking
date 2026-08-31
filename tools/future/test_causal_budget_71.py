"""The budget must not flatter itself, and 71 must stay honest."""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import causal_budget_71 as cb


def test_measured_now_matches_the_organ_trace():
    d = cb.build()["measured_now"]
    assert 28.0 < d["token_ms"] < 29.5
    assert 34.0 < d["tps"] < 36.0


def test_the_demonstrated_regime_is_below_50():
    """Matching the LM head on every organ is not enough for 50 TPS.

    This is the number that keeps the campaign honest: the entire granularity
    hypothesis, fully won, lands in the high forties.
    """
    t = cb.tps(cb.token_ms(cb.DEMONSTRATED_GB_S))
    assert 46.0 < t < 49.0, t


def test_71_is_recorded_as_unreachable_at_the_roof_on_todays_bytes():
    rows = {r["rung"]: r for r in cb.ladder()}
    roof = next(r for k, r in rows.items() if "clean GEMV roof" in k)
    assert roof["tps"] < 71.0, "the roof must not be quoted as 71"
    seventy_one = rows["71 TPS"]
    assert seventy_one["class"] == "NOT_REACHABLE_AT_THE_ROOF_ON_TODAYS_BYTES"
    assert "fewer" in seventy_one["requires"] or "host gap" in seventy_one["requires"]


def test_the_host_gap_lever_is_closed_not_ranked_as_work():
    e = next(x for x in cb.experiments() if x["id"] == "eliminate_all_host_gap")
    assert e["status"] == "CLOSED"
    assert e["cost"] == "NOT_WORTH_RUNNING"


def test_byte_levers_have_their_saving_applied():
    """A lever that saves a GB and reports 0.00 ms because the ARITHMETIC dropped
    it is a bug. One that reports 0.00 ms because its stream was MEASURED at
    0.000 ms/GB is a finding.

    This test used to assert every lever gains more than 0.5 TPS, which was right
    when the bug was that the counterfactual never applied the saving. Then
    ECONOMICS_CALIBRATION timed the streams separately - codes_keep_50 faster
    than 2*MAD, aux_keep_50 inside noise - and the auxiliary levers became a
    measured zero. Asserting they must gain would now pin the overcredit.

    The invariant that catches the original bug without contradicting the
    measurement: the reported saving must EQUAL gb_saved times its own stream's
    rate.
    """
    by_id = {l["id"]: l for l in cb.BYTE_LEVERS}
    seen = 0
    for e in cb.experiments():
        if not e.get("gb_saved"):
            continue
        seen += 1
        lever = by_id[e["id"]]
        assert e["ms_saved"] == pytest.approx(cb.lever_ms_saved(lever), abs=1e-3), e["id"]
        assert e["stream_class"] == lever["stream_class"]
    assert seen == len(by_id), "every lever must appear in the experiment table"


def test_every_rung_above_now_is_labelled_a_target():
    cbnd = cb.build()["claim_boundary"]
    assert "TARGET, not" in cbnd
    for r in cb.ladder():
        if r["rung"] != "measured now":
            assert r["class"] != "MEASURED"

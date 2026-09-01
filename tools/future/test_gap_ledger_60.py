"""The 60-TPS gap is derived, and the clock cannot be restarted into DISCOVERY.

The two ways this module could lie are a typed gap and a clock that resets on
every build. Both are pinned here.
"""
from __future__ import annotations

import json

import pytest

from tools.future import gap_ledger_60 as g


def test_the_gap_is_derived_from_the_live_baseline_not_typed():
    lv = g.live()
    a = json.loads((g.REPO / g.ABSOLUTE_REL).read_text())
    assert lv["ms_per_token"] == a["measured"]["gpu_ms_per_token"]
    assert g.build()["gap_to_60_ms"] == pytest.approx(
        lv["ms_per_token"] - 1000.0 / 60.0, abs=1e-4)


def test_a_self_inconsistent_budget_is_refused(monkeypatch):
    """Still enforced on the fallback path, which is the one that reads a wall
    figure and a wall TPS that could disagree."""
    monkeypatch.setattr(g, "ABSOLUTE_REL", "receipts/future/NO_SUCH_ABS.json")
    d = json.loads((g.REPO / g.BUDGET_REL).read_text())
    monkeypatch.setattr(g, "_budget", lambda: {**d, "decode_wall_tps": 99.0})
    with pytest.raises(g.GapRefused, match="self-inconsistent"):
        g.live()


def test_a_missing_budget_refuses_rather_than_defaulting(monkeypatch):
    """With both the promoted absolute and the pre-promotion budget gone, there
    is no baseline and the ledger must refuse rather than invent one."""
    monkeypatch.setattr(g, "ABSOLUTE_REL", "receipts/future/NO_SUCH_ABS.json")
    monkeypatch.setattr(g, "BUDGET_REL", "receipts/future/NO_SUCH.json")
    with pytest.raises(g.GapRefused, match="no live token budget"):
        g.live()


def test_the_baseline_is_the_promoted_gpu_absolute_not_the_stale_wall():
    """G131 rebased this. Every lever in the ledger is measured as GPU ms, so a
    gap denominated in WALL was comparing two different quantities - and the
    pre-promotion wall figure is 5.3 ms stale besides."""
    lv = g.live()
    assert lv["basis"] == "GPU"
    assert lv["source"] == g.ABSOLUTE_REL
    assert lv["levers_are_the_sealed_default"] is True
    assert lv["ms_per_token"] == lv["gpu_ms_per_token"]
    assert lv["wall_ms_per_token"] > lv["gpu_ms_per_token"]
    assert lv["host_gap_ms"] == pytest.approx(
        lv["wall_ms_per_token"] - lv["gpu_ms_per_token"], abs=5e-4)


def test_it_falls_back_to_the_pre_promotion_budget_and_says_so(monkeypatch):
    monkeypatch.setattr(g, "ABSOLUTE_REL", "receipts/future/NO_SUCH_ABS.json")
    lv = g.live()
    assert lv["basis"] == "WALL_PRE_PROMOTION_FALLBACK"
    assert lv["source"] == g.BUDGET_REL


def test_organs_are_priced_in_complete_token_tps_not_organ_percent():
    rows = g.organ_win_table()
    cur = g.live()["ms_per_token"]
    top = rows[0]
    assert top["organ"] == "mlp_gate_up", "the table must be ranked by cost"
    # Priced against the organ decomposition's OWN total, not the live baseline:
    # those organs were measured before the sealed-default promotion and a stale
    # numerator over a live denominator would inflate every figure.
    own = top["priced_against"]["token_ms"]
    assert own != cur
    assert top["tps_at_20pct_win"] == pytest.approx(
        1000.0 / (own - top["current_ms"] * 0.2), abs=1e-3)
    # A 20% win on the largest organ is still well short of 60.
    assert top["tps_at_20pct_win"] < 60.0


def test_the_stale_organ_decomposition_is_named_not_absorbed():
    st = g.organ_decomposition_is_stale()
    assert st["stale_by_ms"] > 4.0
    assert st["organ_rows_sum_ms"] > st["live_baseline_ms"]
    assert "re-run the organ decomposition" in st["next_measurement"]
    assert g.build()["organ_decomposition_is_stale"] == st


def test_immaterial_organs_are_named_as_not_worth_an_hour():
    rows = g.organ_win_table()
    small = [r for r in rows if not r["material"]]
    assert small, "embedding and sampling are below the threshold"
    assert all("do not" in r["why"] for r in small)


def test_the_three_dominant_kernels_do_not_reach_60_from_the_live_base():
    d = g.three_dominant()
    assert d["share_of_token"] > 0.75
    assert d["tps_at_20pct_across_all_three"] < 60.0
    assert "PROSPECTIVE composed path" in d["reading"], \
        "S026's 60-from-20% arithmetic used a base that is not qualified"


def test_the_open_sets_CEILING_now_exceeds_the_gap_to_60():
    """This flipped when G131 rebased the baseline, and the flip is a statement
    about OPPORTUNITY BOUNDS, not about reaching 60. sum_of_material_open_ms is
    what the open set could remove under PERFECT success; the ledger's own
    honest_reading says these are ceilings, not candidates."""
    b = g.build()
    assert b["does_the_open_set_reach_60"] is True
    assert b["sum_of_material_open_ms"] > b["gap_to_60_ms"]
    assert "CEILINGS, not candidates" in b["honest_reading"]


def test_perfect_arithmetic_removal_still_does_not_reach_60():
    """The arithmetic school stays bounded below 60 at the new baseline too, so
    the open set clears the gap only on its BANDWIDTH rungs."""
    a = g.build()["arithmetic_ceiling"]
    assert a["reaches_60"] is False
    assert a["still_short_of_60_by_ms"] > 0


def test_experiments_are_ranked_by_max_ms_removable():
    r = g.ranked_experiments()
    assert r == sorted(r, key=lambda x: -x["max_ms_removable"])
    assert r[0]["id"] == "reach_demonstrated_bandwidth_mlp"


def test_the_clock_start_is_stamped_once_and_never_rewritten():
    first = g._clock_start()
    second = g._clock_start()
    assert first["started_unix"] == second["started_unix"]
    assert (g.REPO / g.CLOCK_REL).is_file()


def test_the_clock_phase_advances_with_elapsed_time(monkeypatch):
    start = g._clock_start()
    for hours, want in ((0.1, "DISCOVERY"), (7, "SURVIVOR_OR_COLLAPSE"),
                        (13, "WIDEN"), (19, "DEPRIVILEGE")):
        monkeypatch.setattr(
            g.time, "time", lambda h=hours: float(start["started_unix"]) + h * 3600)
        assert g.escalation_clock()["phase"] == want


def test_the_clock_is_explicitly_not_an_abandonment():
    c = g.escalation_clock()
    assert "forbids complacency, not the incumbent" in c["not_an_abandonment"]
    assert c["phase_licenses"]


def test_the_whole_arithmetic_school_cannot_reach_sixty():
    """The decisive strategic fact, computed from measured arm_a rates."""
    a = g.arithmetic_ceiling()
    assert a["reaches_60"] is False
    assert a["still_short_of_60_by_ms"] > 0
    assert 50 < a["tps_after"] < 60, "perfect removal lands in the mid-50s"
    assert "CANNOT REACH 60 EVEN IF IT PERFECTLY SUCCEEDS" in a["verdict"]


def test_the_ceiling_does_not_dismiss_the_school_it_bounds_it():
    a = g.arithmetic_ceiling()
    assert "largest single block on the board" in a["what_this_does_not_say"]
    assert "must not be the ONLY thing running" in a["what_this_does_not_say"]


def test_deltanet_is_counted_by_its_in_projection_not_the_whole_organ():
    """Counting the whole 5.5971 ms organ would overstate the ceiling."""
    a = g.arithmetic_ceiling()
    q4 = next(p for p in a["parts"] if p["codec"] == "q4")
    assert "deltanet" not in q4["organs"], "the organ must not be counted whole"
    assert q4["deltanet_in_projection_only_ms"] == g.DN_INPROJ_MS
    assert "would overstate this ceiling" in q4["deltanet_note"]


def test_each_ratio_cites_the_receipt_it_was_measured_in():
    a = g.arithmetic_ceiling()
    for part in a["parts"]:
        assert part["source"].startswith("receipts/future/")
        assert part["arm_a_over_production"] > 1.4


def test_the_organ_level_estimate_is_labelled_as_one():
    a = g.arithmetic_ceiling()
    assert "estimate at the organ level" in a["assumption"]


def test_a_missing_organ_refuses(monkeypatch):
    d = g._budget()
    trimmed = {**d, "organs": {**d["organs"], "rows": [
        r for r in d["organs"]["rows"] if r["organ"] != "q4_remainder"]}}
    monkeypatch.setattr(g, "_budget", lambda: trimmed)
    with pytest.raises(g.GapRefused, match="budget has no rows for"):
        g.arithmetic_ceiling()


def test_sixty_needs_bytes_removed_and_says_how_many():
    b = g.bytes_required_after_arithmetic()
    assert b["further_ms_needed_for_60"] > 0
    assert 0.10 < b["fraction_of_matvec_bytes_to_remove"] < 0.30
    assert "must stop existing" in b["statement"]


def test_it_rules_out_entropy_coding_with_the_measured_entropy():
    b = g.bytes_required_after_arithmetic()
    w = b["why_entropy_coding_does_not_supply_it"]
    assert "1.87 bits" in w and "93.5%" in w
    assert "INFORMATION ELIMINATION" in w


def test_it_names_what_this_accounting_does_not_cover():
    b = g.bytes_required_after_arithmetic()
    o = b["the_other_place_to_look"]
    assert "DeltaNet state update" in o
    assert "lm_head" in o
    assert "G053" in o, "the open obligation aimed at that time"


def test_the_composition_assumption_is_declared():
    b = g.bytes_required_after_arithmetic()
    assert "do not automatically" in b["caveat"]
    assert "reopened after every representation change" in b["caveat"]

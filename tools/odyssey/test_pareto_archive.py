"""G040 pins."""
import json
from pathlib import Path

import pytest

R = Path(__file__).resolve().parents[2] / "receipts/headless/PARETO_ARCHIVE.json"
pytestmark = pytest.mark.skipif(not R.is_file(), reason="G040 receipt not built")


def rec():
    return json.load(open(R))


def test_capability_floor_is_applied_before_pareto():
    d = rec()["selection"]
    assert "CAPABILITY FLOOR FIRST" in d["rule"]
    assert d["rejected_by_floor"]


def test_the_floor_actually_rejected_the_density_and_speed_leaders():
    """This is why the order matters: both extremes are capability-dead."""
    d = rec()
    rejected = set(d["selection"]["rejected_by_floor"])
    assert d["archive"]["LOWEST_DENSITY"] in rejected
    assert d["archive"]["FASTEST_TPOT"] in rejected


def test_rejected_bodies_really_produce_no_verified_work():
    for k, v in rec()["selection"]["rejected_by_floor"].items():
        assert (v["capability_passed"] or 0) == 0
        assert v["hcli_wus_per_hour"] == 0.0


def test_all_six_archive_categories_are_present():
    a = rec()["archive"]
    assert set(a) == {"LOWEST_DENSITY", "FASTEST_TPOT", "LOWEST_TTFT", "BEST_CAPABILITY",
                      "BEST_HCLI_WUS_HOUR", "BEST_LONG_CONTEXT", "BEST_MULTISESSION"}


def test_unmeasured_categories_are_declared_not_guessed():
    d = rec()
    for k, v in d["archive"].items():
        if v is None:
            assert k in d["unmeasured_categories"], k
            assert len(d["unmeasured_categories"][k]) > 40


def test_the_pareto_front_is_not_a_singleton_and_neither_body_dominates():
    d = rec()["selection"]
    assert d["front_is_not_a_singleton"] is True
    assert len(d["pareto_front"]) >= 2


def test_selection_uses_work_per_resource_not_density():
    d = rec()["selection"]
    assert "per GB resident" in d["composite_metric"]
    assert "WUs/hour" in d["composite_metric"]


def test_the_margin_is_compared_against_measured_spread():
    """A 2% selection on a single run is exactly the mistake the concurrency sweep made."""
    c = rec()["selection"]["CONFIDENCE"]
    assert c["reps_per_body"]
    for body, n in c["reps_per_body"].items():
        assert n >= 3, f"{body} has only {n} reps"
    assert c["margin_exceeds_spread"] is True


def test_accepted_counts_are_stable_across_reps():
    """If acceptance varied run to run, the rate would not mean anything."""
    c = rec()["selection"]["CONFIDENCE"]
    assert all(c["accepted_stable_across_reps"].values())


def test_the_bench_limitation_is_stated_even_though_reps_are_tight():
    c = rec()["selection"]["CONFIDENCE"]
    assert "BENCH-LIMITED" in c["reading"]
    assert "no number of reps fixes that" in c["reading"]


def test_a_resident_was_actually_selected():
    d = rec()["selection"]
    assert d["provisional_resident"] in d["pareto_front"]
    assert rec()["pass"] is True

"""G038 pins. The trap: acceptance rate alone never proves a speedup. This codebase
already measured 87% acceptance at 0.91x."""
import json
from pathlib import Path

import pytest

R = Path(__file__).resolve().parents[2] / "receipts/headless/G038_DECODING_GRAVITY.json"
pytestmark = pytest.mark.skipif(not R.is_file(), reason="G038 receipt not built")


def rec():
    return json.load(open(R))


def test_the_census_actually_searched_for_each_named_mechanism():
    """S011 §20 names MTP, auxiliary heads, self-speculative paths and early-exit."""
    m = rec()["census"]["machinery_found"]
    for k in ("mtp", "medusa", "eagle", "exit", "draft"):
        assert k in m, k


def test_mtp_machinery_was_found_and_measured_not_assumed():
    c = rec()["census"]
    assert c["mtp_params"] > 0
    assert len(c["mtp_tensors"]) == 15
    for n, t in c["mtp_tensors"].items():
        assert t["params"] > 0 and t["shape"]


def test_the_census_records_that_it_is_unused():
    c = rec()["census"]
    assert "NEVER PACKED" in c["status_in_this_campaign"]


def test_every_draft_carries_a_cost_ratio_next_to_its_acceptance():
    """An acceptance rate published without the cost ratio is the 87%-at-0.91x mistake."""
    for name, d in rec()["draft_measurement"]["drafts"].items():
        assert "acceptance_rate" in d
        assert "cost_ratio_draft_over_verify" in d
        assert "predicted_speedup_x" in d
        assert "pays" in d


def test_break_even_is_the_stated_rule_and_is_applied_correctly():
    for name, d in rec()["draft_measurement"]["drafts"].items():
        expected = d["cost_ratio_draft_over_verify"] < d["acceptance_rate"]
        assert d["pays"] is expected, name


def test_a_dead_generator_really_does_carry_draft_signal():
    """S011 §21: reclassify failures by secondary role."""
    d = rec()["draft_measurement"]["drafts"]
    dead = {k: v for k, v in d.items() if v["is_dead_final_generator"]}
    assert dead, "no dead generator was tested as a draft"
    assert max(v["acceptance_rate"] for v in dead.values()) >= 0.5


def test_high_acceptance_is_not_reported_as_a_win():
    """variantA agrees 75% of the time and still loses. Say so."""
    d = rec()["draft_measurement"]["drafts"]["variantA-2.98"]
    assert d["acceptance_rate"] >= 0.7
    assert d["pays"] is False
    assert d["predicted_speedup_x"] < 1.0
    assert "NOT_ECONOMIC" in d["role"]


def test_no_full_size_draft_pays():
    """Same size class, same 964 dispatches -- drafting cannot be cheaper."""
    ds = rec()["draft_measurement"]["drafts"]
    assert not any(v["pays"] for v in ds.values())
    for v in ds.values():
        assert 0.9 < v["cost_ratio_draft_over_verify"] < 1.1


def test_the_round_trip_was_verified_rather_than_assumed():
    m = rec()["draft_measurement"]
    assert "round_trip_skipped" in m
    assert "excluded rather than guessed" in m["round_trip_note"]


def test_the_mtp_projection_is_labelled_a_projection():
    p = rec()["where_it_could_pay"]
    assert "IS_A_PROJECTION_NOT_A_MEASUREMENT" in p
    assert "never been packed" in p["IS_A_PROJECTION_NOT_A_MEASUREMENT"]
    assert p["next_step"]


def test_the_projection_uses_the_same_break_even_rule():
    p = rec()["where_it_could_pay"]
    for a, e in p["projected"].items():
        assert e["pays"] is (e["cost_ratio_draft_over_verify"] < e["acceptance"])


def test_the_prior_falsification_is_cited():
    r = rec()["reclassification_S011_21"]
    assert "0.91x" in r["prior_falsification"]

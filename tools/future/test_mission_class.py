"""Classification comes from what an obligation says it blocks, not from taste.

The dangerous failure is a classifier that quietly empties the launch gate. The
unknown case must BLOCK, and a declared class outside the four must refuse.
"""
from __future__ import annotations

import pytest

from tools.future import mission_class as mc


def test_every_unmet_obligation_is_classified():
    rows = mc.classified()
    assert rows, "the ledger has unmet obligations"
    assert all(r["mission_class"] in mc.CLASSES or
               r["mission_class"] == "UNCLASSIFIED" for r in rows)
    assert all(r["source"] in ("DECLARED", "DERIVED", "NONE") for r in rows)


def test_a_declared_class_wins_over_derivation():
    rows = {r["id"]: r for r in mc.classified()}
    declared = [r for r in rows.values() if r["source"] == "DECLARED"]
    assert declared, "the S026/S027 obligations declare their own class"


def test_an_invalid_declared_class_refuses():
    with pytest.raises(mc.ClassifyRefused, match="not one of"):
        mc.classify_one({"id": "G999", "text": "class: SOMEWHAT_IMPORTANT"})


def test_the_unknown_case_blocks_rather_than_defers():
    r = mc.classify_one({"id": "G999", "text": "an obligation with no signal at all"})
    assert r["mission_class"] == "UNCLASSIFIED"
    assert "NOT demoted" in r["why"]
    assert "becomes decorative" in r["why"]


def test_unclassified_counts_as_blocking_in_the_gate(monkeypatch):
    monkeypatch.setattr(mc, "_blocks",
                        lambda: [{"id": "G999", "text": "no signal here"}])
    g = mc.launch_gate()
    assert g["blocking_ids"] == ["G999"]
    assert g["green"] is False


def test_housekeeping_is_deferred_and_named():
    rows = {r["id"]: r for r in mc.classified()}
    assert rows["G021"]["mission_class"] == "DEFERRED", \
        "rewriting git history blocks no experiment"
    assert "changes no measurement" in rows["G021"]["why"]


def test_the_gate_does_not_clear_itself():
    """A classifier that turns the gate green on its own is the failure mode."""
    g = mc.launch_gate()
    assert g["green"] is False
    assert g["n_blocking"] > 0
    assert g["n_blocking"] < g["n_unmet_total"], "but it must move something"


def test_the_gate_states_what_it_changed():
    g = mc.launch_gate()
    assert "ceremonial gate" in g["what_this_changes"]
    assert "do not hold the launch" in g["rule"]


def test_the_derivation_is_declared_to_be_from_text_not_importance():
    d = mc.build()["derivation_is_from_the_text_not_from_importance"]
    assert "what its own text says it blocks" in d
    assert "No rule here encodes a judgement about how important" in d


def test_every_derivation_rule_carries_a_reason_and_a_class():
    for cls, why, needles in mc.DERIVE:
        assert cls in mc.CLASSES
        assert why and needles


def test_derivation_order_is_stable_for_a_text_matching_two_rules():
    """A text hitting both an autonomy and a tps needle must land LAUNCH_CRITICAL."""
    r = mc.classify_one({"id": "G999", "text": "a power torture that measures tps"})
    assert r["mission_class"] == "LAUNCH_CRITICAL"

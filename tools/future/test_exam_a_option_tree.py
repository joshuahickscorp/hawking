"""G115 tests: the exam must be able to FAIL, and must not contain its answer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exam_a_option_tree as ex  # noqa: E402


def test_the_pack_does_not_contain_the_answer():
    """S033: DO NOT FEED IT S032'S ANSWER. If the pack carries the routes from
    SUB2_UNIFIED_PLAN, the exam measures nothing."""
    c = ex.pack_contains_no_answer()
    assert c["clean"] is True
    assert c["leaked"] == []
    assert c["route_ids_checked"] > 0, "there must be ids to check against"


def test_a_leaked_route_id_refuses(monkeypatch):
    real = ex.pack
    leaked_id = json.loads(
        (ex.REPO / ex.FORBIDDEN_TO_PACK_REL).read_text())["routes"][0]["id"]
    monkeypatch.setattr(ex, "pack", lambda: real() + f"\nconsider {leaked_id}")
    with pytest.raises(ex.ExamRefused, match="answer"):
        ex.pack_contains_no_answer()


def test_the_pack_states_the_objective_and_the_conventional_floor():
    p = ex.pack()
    assert "complete EBPW" in p
    assert "2.0" in p
    assert "bottoms out at" in p, "the resident must know a better code cannot get there"
    assert "never reconstruct a dense parent" in p


def test_the_pack_carries_the_refuted_classes():
    p = ex.pack().lower()
    assert "shared linear low-rank" in p
    assert "entropy coding" in p
    assert "larger group sizes" in p


def test_the_pack_does_not_prescribe_a_route():
    """It may state EVIDENCE. It may not state a plan."""
    p = ex.pack().lower()
    for prescription in ("you should", "the answer is", "start by trying",
                         "the correct route"):
        assert prescription not in p


def test_the_refuted_class_patterns_actually_match_their_class():
    R = ex.REFUTED_CLASSES
    assert R["shared_linear_low_rank"].search("shared basis across layers")
    assert R["shared_linear_low_rank"].search("global low rank factorization")
    assert R["entropy_coding_the_code_stream"].search("huffman the code stream")
    assert R["larger_groups"].search("group size 1024")
    assert not R["larger_groups"].search("group size 64")


def test_the_schema_requires_a_falsifier_on_every_route():
    item = ex.OPTION_TREE_SCHEMA["properties"]["routes"]["items"]
    assert set(item["required"]) == {
        "id", "claim", "evidence_status", "cheapest_falsifier"}
    assert item["additionalProperties"] is False


def test_the_grade_checks_acceptance_not_the_science(monkeypatch):
    """G115 passes on a tree the RESIDENT generated where every route carries a
    status and a falsifier. Not on reaching 2.0."""
    fake = {"value": {"routes": [
        {"id": "r1", "claim": "c", "evidence_status": "UNTESTED",
         "cheapest_falsifier": "f"},
        {"id": "r2", "claim": "c2", "evidence_status": "MEASURED",
         "cheapest_falsifier": "f2"}],
        "which_first": "r1", "why_first": "cheapest"},
        "admit": {"ok": True}}
    monkeypatch.setattr(ex, "reply", lambda: fake)
    g = ex.grade()
    assert g["acceptance_met"] is True
    assert g["n_routes"] == 2
    assert g["every_route_carries_both"] is True
    assert g["which_first_is_one_of_its_own_routes"] is True
    assert "reaching 2.0" in g["what_acceptance_is_not"]


def test_a_route_missing_a_falsifier_fails_acceptance(monkeypatch):
    fake = {"value": {"routes": [
        {"id": "r1", "claim": "c", "evidence_status": "UNTESTED",
         "cheapest_falsifier": ""}],
        "which_first": "r1", "why_first": "x"}, "admit": {"ok": False}}
    monkeypatch.setattr(ex, "reply", lambda: fake)
    g = ex.grade()
    assert g["every_route_carries_both"] is False
    assert g["acceptance_met"] is False


def test_an_empty_tree_fails_acceptance(monkeypatch):
    monkeypatch.setattr(ex, "reply", lambda: {
        "value": {"routes": [], "which_first": "", "why_first": ""},
        "admit": {"ok": False}})
    g = ex.grade()
    assert g["resident_generated_the_tree"] is False
    assert g["acceptance_met"] is False


def test_picking_a_route_it_did_not_propose_fails_acceptance(monkeypatch):
    monkeypatch.setattr(ex, "reply", lambda: {
        "value": {"routes": [
            {"id": "r1", "claim": "c", "evidence_status": "X",
             "cheapest_falsifier": "f"}],
            "which_first": "r9", "why_first": "x"}, "admit": {"ok": True}})
    assert ex.grade()["which_first_is_one_of_its_own_routes"] is False
    assert ex.grade()["acceptance_met"] is False


def test_a_re_proposed_refuted_class_is_a_scar_feed_failure(monkeypatch):
    """Which is a finding about the PACK, not about the resident."""
    monkeypatch.setattr(ex, "reply", lambda: {
        "value": {"routes": [
            {"id": "sb", "claim": "use a shared basis across all layers",
             "evidence_status": "UNTESTED", "cheapest_falsifier": "f"}],
            "which_first": "sb", "why_first": "x"}, "admit": {"ok": True}})
    g = ex.grade()
    assert g["scar_feed_held"] is False
    assert "shared_linear_low_rank" in g["re_proposed_refuted_classes_UNAWARE"]


def test_a_KNOWING_revisit_of_a_refuted_class_is_not_a_failure(monkeypatch):
    """The resident named entropy coding and marked it REFUTED_IN_PART. That is
    the scar working; counting it against it would punish honesty."""
    monkeypatch.setattr(ex, "reply", lambda: {
        "value": {"routes": [
            {"id": "entropy", "claim": "entropy code the 2-bit stream",
             "evidence_status": "REFUTED_IN_PART", "cheapest_falsifier": "f"}],
            "which_first": "entropy", "why_first": "x"}, "admit": {"ok": True}})
    g = ex.grade()
    assert g["scar_feed_held"] is True
    assert "entropy_coding_the_code_stream" in \
        g["revisited_refuted_classes_KNOWINGLY"]
    assert g["re_proposed_refuted_classes_UNAWARE"] == {}


def test_the_grade_says_it_is_not_hit_rate():
    monkeys = ex.grade.__doc__ or ""
    assert "not Claude's opinion" in monkeys

"""Pareto comparison machinery + model-agnostic resident selection contract."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tools.pareto_table import (
    COMPARISON_AXES,
    Axis,
    candidate_identity,
    dominates,
    metrics_of,
    pareto_front,
    provenance_of,
    qualify,
)
from tools.selection_contract import (
    MACHINE_CLASSES,
    SelectionRefused,
    admit_candidate,
    bind_machine,
    bind_shadow,
    decide_from_table,
    promote,
    qualify as qualify_record,
    rollback,
)

CONTRACT = Path(__file__).resolve().parent / "selection_contract.py"

# Vendor/body names that must not be load-bearing in the contract.
FORBIDDEN_MODEL_NAMES = (
    "qwen",
    "llama",
    "glm",
    "mixtral",
    "deepseek",
    "falcon",
    "kimi",
    "mistral",
    "gpt-oss",
    "gpt2",
    "phi-3",
    "baichuan",
    "internlm",
    "noetic",
    "sealed-3.14",
    "variantb",
    "pocketaihub",
)


def test_candidate_identity_is_an_artifact_id_not_a_vendor_name():
    ident = candidate_identity("uniform-q4-v1", artifact_digest="abc", machine_class="UMA")
    assert ident.candidate_id == "uniform-q4-v1"
    assert ident.machine_class == "UMA"
    with pytest.raises(ValueError):
        candidate_identity("  ")


def test_dominates_skips_missing_cells_and_does_not_fabricate_zero():
    a = {"effective_bpw": 3.0, "token_ns": None, "tps": 10}
    b = {"effective_bpw": 4.0, "token_ns": 100, "tps": 10}
    assert dominates(a, b) is True
    # b is worse on bpw, equal tps, token_ns incomparable — does not dominate.
    assert dominates(b, a) is False
    empty = {"effective_bpw": None, "token_ns": None, "tps": None}
    other = {"effective_bpw": 4.0, "token_ns": 1, "tps": 1}
    assert dominates(empty, other) is False
    assert dominates(other, empty) is False


def test_dominates_requires_a_strict_win():
    a = {"effective_bpw": 4.0, "tps": 10, "token_ns": 100}
    b = dict(a)
    assert dominates(a, b) is False
    better = {"effective_bpw": 3.0, "tps": 10, "token_ns": 100}
    assert dominates(better, a) is True
    mixed = {"effective_bpw": 3.0, "tps": 1, "token_ns": 100}
    assert dominates(mixed, a) is False


def test_pareto_front_is_sorted_and_drops_dominated_bodies():
    cands = {
        "small-slow": {"effective_bpw": 2.0, "token_ns": 200, "tps": 5},
        "mid": {"effective_bpw": 3.0, "token_ns": 100, "tps": 10},
        "dominated": {"effective_bpw": 4.0, "token_ns": 200, "tps": 5},
    }
    front = pareto_front(cands)
    assert "dominated" not in front
    assert front == sorted(front)
    assert "small-slow" in front and "mid" in front


def test_qualify_fails_closed_on_empty_contract_and_missing_cells():
    empty = qualify({"effective_bpw": 3.0}, floors={})
    assert empty.passed is False
    assert "empty contract" in empty.failures[0]
    missing = qualify({"effective_bpw": None}, floors={"effective_bpw": 4.0})
    assert missing.passed is False
    assert any("missing" in f for f in missing.failures)
    ok = qualify({"effective_bpw": 3.0}, floors={"effective_bpw": 4.0})
    assert ok.passed is True
    flag_fail = qualify(
        {"effective_bpw": 3.0},
        floors={"effective_bpw": 4.0},
        flags={"doctor_pass": False},
    )
    assert flag_fail.passed is False


def test_metrics_and_provenance_are_first_class():
    m = metrics_of({"token_ns": 10}, sources={"token_ns": "G150.json"})
    assert m.get("token_ns") == 10
    p = provenance_of("G150.json", parent_digest="aa")
    assert p.receipt_refs == ("G150.json",)
    assert p.parent_digest == "aa"


def test_admit_shadow_qualify_promote_rollback_never_installs():
    rec = admit_candidate(
        "child-a",
        {"effective_bpw": 3.0, "token_ns": 80, "tps": 20},
        flags={
            "doctor_pass": True,
            "provenance_valid": True,
            "native_path": True,
            "no_hidden_fallback": True,
        },
    )
    assert rec["state"] == "CANDIDATE" and rec["installed"] is False
    rec = bind_shadow(rec, machine_class="UMA")
    assert rec["state"] == "SHADOW"
    assert rec["machine_binding"]["machine_class"] == "UMA"
    assert rec["installed"] is False
    rec = qualify_record(rec, floors={"effective_bpw": 4.0})
    assert rec["state"] == "QUALIFIED"
    assert rec["installed"] is False
    parent = admit_candidate(
        "parent",
        {"effective_bpw": 4.0, "token_ns": 100, "tps": 10},
    )
    promoted = promote(rec, parent)
    assert promoted["state"] == "PROMOTED"
    assert promoted["resident"] is True
    assert promoted["installed"] is False
    rolled = rollback(promoted, parent)
    assert rolled["state"] == "ROLLED_BACK"
    assert rolled["identity"]["candidate_id"] == "parent"
    assert rolled["installed"] is False
    assert rolled["rolled_back_from"]["candidate_id"] == "child-a"


def test_promote_refuses_a_dominated_child():
    child = admit_candidate("child", {"effective_bpw": 5.0, "token_ns": 200, "tps": 1})
    child["state"] = "QUALIFIED"
    parent = admit_candidate("parent", {"effective_bpw": 3.0, "token_ns": 80, "tps": 20})
    with pytest.raises(SelectionRefused):
        promote(child, parent)


def test_qualify_refuses_hard_gate_failure_before_qualified():
    rec = admit_candidate(
        "broken",
        {"effective_bpw": 1.0, "token_ns": 1, "tps": 100},
        flags={
            "doctor_pass": False,
            "provenance_valid": True,
            "native_path": True,
            "no_hidden_fallback": True,
        },
    )
    rec = bind_shadow(rec, machine_class="UMA")
    out = qualify_record(rec, floors={"effective_bpw": 4.0})
    assert out["state"] == "SHADOW"
    assert out["qualification"]["passed"] is False
    assert out["installed"] is False


def test_machine_binding_rejects_unknown_class_and_uses_wake_ids():
    rec = admit_candidate("x", {"effective_bpw": 3.0})
    with pytest.raises(SelectionRefused):
        bind_machine(rec, machine_class="not-a-device")
    bound = bind_machine(rec, machine_class="U50_PRESENT", present=False)
    assert bound["machine_binding"]["wake_condition"] == "U50_PRESENT"
    assert bound["machine_binding"]["present"] is False
    assert bound["installed"] is False
    assert "U50_PRESENT" in MACHINE_CLASSES
    assert "DGX_PRESENT" in MACHINE_CLASSES
    assert "NEW_M_SERIES_PRESENT" in MACHINE_CLASSES


def test_decide_from_table_fails_closed_without_floors_and_never_installs():
    table = {
        "a": {"effective_bpw": 3.0, "token_ns": 80, "tps": 20},
        "b": {"effective_bpw": 4.0, "token_ns": 100, "tps": 10},
    }
    decision = decide_from_table(table)
    assert decision["installed"] is False
    assert decision["selected"] is None
    assert decision["state"] == "CANDIDATE"


def test_decide_from_table_promotes_a_dominating_qualified_body():
    table = {
        "a": {"effective_bpw": 3.0, "token_ns": 80, "tps": 20},
        "b": {"effective_bpw": 5.0, "token_ns": 200, "tps": 5},
    }
    decision = decide_from_table(
        table,
        floors={"effective_bpw": 6.0},
    )
    # both pass the bpw floor (lower-is-better max 6); a dominates b.
    assert decision["installed"] is False
    assert "a" in decision["pareto_front"]
    assert "b" not in decision["pareto_front"] or decision["selected"] in (None, "a")
    if len(decision["pareto_front"]) == 1:
        assert decision["selected"] == "a"
        assert decision["state"] == "PROMOTED"
        assert decision["record"]["installed"] is False


def test_selection_contract_has_no_load_bearing_model_name():
    text = CONTRACT.read_text().lower()
    hits = []
    for name in FORBIDDEN_MODEL_NAMES:
        if name.lower() in text:
            hits.append(name)
    assert hits == [], f"load-bearing model names in selection_contract.py: {hits}"
    # Architectural fields must be identity / machine class / wake id.
    assert "candidate_id" in text
    assert "machine_class" in text
    assert "wake_condition" in text
    assert "installed" in text


def test_axis_rejects_unknown_evidence_tier():
    with pytest.raises(ValueError):
        Axis("x", "lower", evidence_tier="GUESSED")

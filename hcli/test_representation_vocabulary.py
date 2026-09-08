"""G031: the vocabulary must stay grounded, classified and falsifiable.

A catalogue of mechanisms with no measured numbers is a glossary. A finding with
no falsifier is a belief. These assert the difference.
"""
import json
from pathlib import Path

CAT = json.loads((Path(__file__).resolve().parent.parent
                  / "receipts" / "future" / "REPRESENTATION_VOCABULARY.json").read_text())


def test_every_mechanism_carries_a_measured_property():
    for m in CAT["MECHANISMS"]:
        assert m.get("property"), m
        assert m.get("measured_on"), f"{m['mechanism']} names no specimen"


def test_mechanisms_that_ran_carry_numbers():
    ran = [m for m in CAT["MECHANISMS"] if "best_arm" in m]
    assert len(ran) >= 7, f"only {len(ran)} mechanisms have a measured arm"
    for m in ran:
        assert isinstance(m.get("ebpw"), (int, float)), m["mechanism"]
        assert isinstance(m.get("ppl"), (int, float)), m["mechanism"]


def test_exactly_one_mechanism_passes_the_gate():
    passing = [m for m in CAT["MECHANISMS"] if m.get("passes_gate") is True]
    assert len(passing) == 1, f"expected one passing class, got {[m['mechanism'] for m in passing]}"
    assert passing[0]["mechanism"] == "affine quantized payload"


def test_every_finding_is_classified_with_a_falsifier_and_an_action():
    valid = {"ARCHITECTURE-FAMILY PRIOR", "ARCHITECTURE-FAMILY PRIOR (candidate)",
             "TRANSFERABLE LAW", "TRANSFERABLE LAW (candidate)",
             "PHYSICAL-MACHINE LAW", "SPECIMEN-SPECIFIC"}
    assert CAT["FINDINGS_CLASSIFIED"], "no findings classified"
    for f in CAT["FINDINGS_CLASSIFIED"]:
        assert f["class"] in valid, f["class"]
        assert f.get("falsifier"), f"{f['finding'][:40]} has no falsifier"
        assert f.get("action"), f"{f['finding'][:40]} has no action"
        assert f.get("evidence"), f"{f['finding'][:40]} has no evidence"


def test_the_family_prior_rests_on_two_specimens():
    p = [f for f in CAT["FINDINGS_CLASSIFIED"] if f["class"] == "ARCHITECTURE-FAMILY PRIOR"]
    assert p, "no promoted family prior"
    assert "O003" in p[0]["evidence"] and "Qwen" in p[0]["evidence"], p[0]["evidence"]


def test_the_gate_is_recorded_as_a_conjunction():
    g = CAT["capability_gate"]
    assert "ppl_max" in g and "median_4gram_max" in g
    assert "CONJUNCTION" in g["note"]

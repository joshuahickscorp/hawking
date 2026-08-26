"""Inheritance, not a log. Five fields, and evidence that resolves."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import transfer_report as tr

REPO = Path(__file__).resolve().parents[2]
R = REPO / "receipts/headless/QWEN_TRANSFER_REPORT.json"


def test_missing_fields_are_refused():
    try:
        tr.validate({"id": "X"})
    except tr.Refused as r:
        assert "missing field" in str(r)
    else:
        raise AssertionError("fieldless entry accepted")


def test_no_reopening_condition_is_refused():
    e = {f: ["x"] for f in tr.FIELDS}
    e.update(id="X", reopening_conditions=[], evidence=["receipts/headless/KERNEL_LIBRARY.json"])
    try:
        tr.validate(e)
    except tr.Refused as r:
        assert "reopening_conditions" in str(r)
    else:
        raise AssertionError("entry with no reopening condition accepted")


def test_no_measured_outcome_anywhere_is_refused():
    e = {f: ["x"] for f in tr.FIELDS}
    e.update(id="X", successful_architecture_classes=[], failed_architecture_classes=[],
             evidence=["receipts/headless/KERNEL_LIBRARY.json"])
    try:
        tr.validate(e)
    except tr.Refused as r:
        assert "no measured outcome" in str(r)
    else:
        raise AssertionError("entry with no measured outcome accepted")


def test_dead_evidence_is_refused():
    e = {f: ["x"] for f in tr.FIELDS}
    e.update(id="X", evidence=["receipts/headless/NOPE.json#a"])
    try:
        tr.validate(e)
    except tr.Refused:
        return
    raise AssertionError("dead evidence accepted")


def test_receipt_is_complete_and_every_entry_carries_all_five():
    d = json.load(open(R))
    assert d["pass"] is True and d["n_rejected"] == 0 and d["n_entries"] >= 20
    for e in d["entries"]:
        for f in tr.FIELDS:
            assert f in e, (e["id"], f)
        assert e["reopening_conditions"]


def test_negatives_carry_the_model_specific_scope_law():
    d = json.load(open(R))
    negs = [e for e in d["entries"] if e["id"].startswith("TR-NEG")]
    assert negs
    for e in negs:
        assert "never prunes" in e["scope_law"]

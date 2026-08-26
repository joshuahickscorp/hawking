"""No anonymous wins: an incomplete entry cannot enter, and a contract must be runnable."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import kernel_library as kl

REPO = Path(__file__).resolve().parents[2]
K = REPO / "receipts/headless/KERNEL_LIBRARY.json"
S = REPO / "receipts/headless/SUPEROPERATOR_LIBRARY.json"


def test_missing_field_is_refused():
    try:
        kl.check({"kernel_identity": "x"})
    except kl.Refused as r:
        assert "missing field" in str(r)
    else:
        raise AssertionError("incomplete entry accepted")


def test_blank_field_without_a_reason_is_refused():
    e = {f: {"kind": "CITED", "value": 1} for f in kl.REQUIRED}
    e["kernel_identity"] = "x"
    e["parity"] = {"kind": "ABSENT", "value": None}          # no absent_reason
    try:
        kl.check(e)
    except kl.Refused as r:
        assert "absent_reason" in str(r)
    else:
        raise AssertionError("blank field with no reason accepted")


def test_absent_with_a_reason_is_allowed():
    e = {f: {"kind": "CITED", "value": 1} for f in kl.REQUIRED}
    e["kernel_identity"] = "x"
    e["parity"] = {"kind": "ABSENT", "value": None, "absent_reason": "no sealed number"}
    kl.check(e)


def test_every_kernel_in_the_receipt_is_complete():
    d = json.load(open(K))
    assert d["n_rejected"] == 0 and d["n_complete"] == d["n_kernels"]
    for k in d["kernels"]:
        kl.check(k)


def test_contracts_that_ran_passed_and_the_rest_are_declared():
    d = json.load(open(K))
    for b, r in d["contract_runs"].items():
        assert r["runnable"] and r["passed"], (b, r)
    # kernels with no runnable contract are counted, never silently assumed correct
    assert d["n_kernels_without_a_runnable_contract"] == sum(
        1 for v in d["parity_contracts"].values() if not v)


def test_superoperators_are_separate_and_carry_the_refuted_megakernel():
    d = json.load(open(S))
    prim = {k["kernel_identity"] for k in json.load(open(K))["kernels"]}
    assert d["n_operators"] >= 3
    assert any(x["operator"] == "multi_layer_megakernel" for x in d["refuted"])
    for op in d["operators"]:
        assert op["semantic_justification"]
        assert set(op["kernels"]) <= prim, "a superoperator must name real primitive kernels"

"""The three laws this store exists to enforce, each watched failing."""
import json, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import negative_science as ns

REPO = Path(__file__).resolve().parents[2]
GOOD_EV = "receipts/headless/DECODING_GRAVITY.json#one_line"


def _entry(**over):
    e = dict(id="T", model="m", organ="o", technique="t", representation="r", kernel="k",
             machine="M", capability="c", physical_reason="p", reopen_condition="rc",
             level="MODEL_SPECIFIC", evidence=[GOOD_EV])
    e.update(over)
    return e


def test_nine_fields_enforced():
    for f in ns.FIELDS:
        e = _entry(); e.pop(f)
        try:
            ns.validate(e)
        except ns.Rejected as r:
            assert f in str(r)
        else:
            raise AssertionError(f"missing {f} was accepted")


def test_single_model_cannot_promote():
    for lvl in ("FAMILY", "GENERAL_PHYSICAL"):
        try:
            ns.validate(_entry(level=lvl))
        except ns.Rejected as r:
            assert "independently measured" in str(r)
        else:
            raise AssertionError(f"{lvl} accepted on one model")


def test_evidence_must_resolve():
    try:
        ns.validate(_entry(evidence=["receipts/headless/NOPE_DOES_NOT_EXIST.json"]))
    except ns.Rejected as r:
        assert "does not resolve" in str(r)
    else:
        raise AssertionError("nonexistent evidence accepted")
    try:
        ns.validate(_entry(evidence=["receipts/headless/DECODING_GRAVITY.json#no.such.key"]))
    except ns.Rejected as r:
        assert "does not resolve" in str(r)
    else:
        raise AssertionError("unresolvable json path accepted")


def test_receipt_holds_no_promoted_level():
    d = json.load(open(REPO / "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json"))
    assert d["counts"]["by_level"]["GENERAL_PHYSICAL"] == 0
    assert d["counts"]["rejected_at_admission"] == 0
    assert all(e.get(f) for e in d["entries"] for f in ns.FIELDS)


def test_rebuild_is_idempotent():
    """Re-migrating our own output once duplicated every entry. It must not again."""
    p = REPO / "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json"
    before = len(json.load(open(p))["entries"])
    subprocess.run([sys.executable, str(Path(__file__).parent / "negative_science.py"),
                    "--rebuild", str(p)], check=True, capture_output=True)
    assert len(json.load(open(p))["entries"]) == before

"""The matrix is only worth reading if a bad citation cannot get into it."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import organ_library as ol

REPO = Path(__file__).resolve().parents[2]
M = REPO / "receipts/headless/ORGAN_FRONTIER_MATRIX.json"


def test_missing_receipt_is_refused():
    try:
        ol.cite("receipts/headless/NO_SUCH_RECEIPT.json", "a.b")
    except ol.Refused as r:
        assert "missing receipt" in str(r)
    else:
        raise AssertionError("missing receipt accepted")


def test_unresolvable_path_is_refused():
    try:
        ol.cite("receipts/headless/ORGAN_DENSITY_FLOORS.json", "organs.nope.x")
    except ol.Refused as r:
        assert "no key" in str(r)
    else:
        raise AssertionError("unresolvable json path accepted")


def test_every_populated_cell_resolves():
    d = json.load(open(M))
    for e in d["organs"]:
        for k, v in e.items():
            if isinstance(v, dict) and v.get("cite"):
                rel, _, jp = v["cite"].partition("#")
                ol.cite(rel, jp)          # raises Refused if it does not resolve


def test_unmeasured_is_absent_not_interpolated():
    d = json.load(open(M))
    for e in d["organs"]:
        if e["status"] == "UNMEASURED":
            assert e["lowest_local_ebpw"]["value"] is None
            assert e["models_measured"] == []


def test_recognizer_consumes_the_library():
    """The consumer trace: the recognizer's notion of KNOWN comes from this matrix."""
    sys.path.insert(0, str(REPO / "tools/odyssey"))
    import arch_recognizer as ar
    known, declared = ar.known_organs()
    measured = {e["organ"] for e in json.load(open(M))["organs"] if e["status"] == "MEASURED"}
    assert known == measured and declared

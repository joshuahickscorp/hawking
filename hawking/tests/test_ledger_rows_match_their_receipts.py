"""A ledger row must not contradict the receipt it cites.

The campaign's ONLY nr_candidate row recorded uniform_capability_ok=True while
receipts/future/G017_ORGAN_ALLOCATION.json said capability_ok=False for every
arm. UNIFORM passes perplexity and FAILS 4-gram diversity, so it fails the
conjunction -- the row had copied ppl_ok into a field that means the whole
gate. That turned a total failure into an apparent pass, on the single row the
entire OII axis rested on.

The row was hand-assembled through the generic odyssey.record_measurement door
rather than produced by ingest_g017_into_ledger, which is how the two drifted.
This test is the guard: any capability field in a ledger row must equal the
conjunction in the receipt it names.
"""
import json
from pathlib import Path

import pytest

LEDGER = Path("receipts/future/G034_ODYSSEY_LEDGER.json")

ARM_FOR = {
    "uniform_capability_ok": "UNIFORM",
    "organ_weighted_capability_ok": "ORGAN_WEIGHTED",
    "inverted_capability_ok": "INVERTED",
}


def _measured_nr_rows():
    if not LEDGER.is_file():
        pytest.skip("ledger not present")
    doc = json.loads(LEDGER.read_text())
    return [s for s in doc.get("specimens", [])
            if (s.get("axes", {}).get("nr_candidate", {}) or {}).get("state") == "MEASURED"]


def test_there_is_at_least_one_measured_nr_candidate():
    rows = _measured_nr_rows()
    assert rows, "no MEASURED nr_candidate row: this guard would vacuously pass"


def test_every_capability_field_equals_its_receipts_conjunction():
    checked = 0
    for spec in _measured_nr_rows():
        cell = spec["axes"]["nr_candidate"]
        rp = cell.get("receipt")
        if not rp or not Path(rp).is_file():
            continue
        arms = (json.loads(Path(rp).read_text()).get("arms") or {})
        for field, arm in ARM_FOR.items():
            if field not in cell["value"] or arm not in arms:
                continue
            got, want = cell["value"][field], arms[arm].get("capability_ok")
            assert got == want, (
                f"{spec['slug']}: ledger {field}={got} but {rp} says "
                f"{arm}.capability_ok={want}. capability_ok is the ppl AND "
                f"diversity conjunction -- do not record ppl_ok here.")
            checked += 1
    assert checked, "no capability field was actually compared; the guard is vacuous"


def test_a_capability_pass_is_never_recorded_from_ppl_alone():
    """The exact defect: an arm that passes ppl and fails diversity is NOT ok."""
    for spec in _measured_nr_rows():
        cell = spec["axes"]["nr_candidate"]
        rp = cell.get("receipt")
        if not rp or not Path(rp).is_file():
            continue
        for arm, d in (json.loads(Path(rp).read_text()).get("arms") or {}).items():
            if d.get("ppl_ok") and not d.get("diversity_ok"):
                field = next((f for f, a in ARM_FOR.items() if a == arm), None)
                if field and field in cell["value"]:
                    assert cell["value"][field] is False, (
                        f"{spec['slug']} {arm}: passes ppl, fails diversity, so the "
                        f"conjunction is False -- but the ledger records {field}="
                        f"{cell['value'][field]}")


PPL_FOR = {
    "uniform_ppl": "UNIFORM",
    "organ_weighted_ppl": "ORGAN_WEIGHTED",
    "inverted_ppl": "INVERTED",
}


def test_a_row_does_not_discard_numbers_its_receipt_carries():
    """The boolean check passed while three ppl fields were null.

    The row was hand-assembled by a script that read arms[X]["ppl"], but the
    receipt nests it at arms[X]["capability"]["ppl"]. So every ppl came back
    None and the ledger published nulls where 5.2278 / 6.1738 / 12.84 existed.
    Nothing failed, because the guard only compared capability booleans.

    Silent data loss is the same disease as a contradicted boolean: the row
    stops being a faithful reading of the evidence it cites.
    """
    checked = 0
    for spec in _measured_nr_rows():
        cell = spec["axes"]["nr_candidate"]
        rp = cell.get("receipt")
        if not rp or not Path(rp).is_file():
            continue
        arms = (json.loads(Path(rp).read_text()).get("arms") or {})
        for field, arm in PPL_FOR.items():
            if field not in cell["value"] or arm not in arms:
                continue
            want = (arms[arm].get("capability") or {}).get("ppl", arms[arm].get("ppl"))
            if want is None:
                continue
            got = cell["value"][field]
            assert got is not None, (
                f"{spec['slug']}: ledger {field} is null but {rp} carries "
                f"{arm} ppl={want}. The row discarded a number its own receipt has.")
            assert abs(float(got) - float(want)) < 1e-9, (
                f"{spec['slug']}: ledger {field}={got} but receipt says {want}")
            checked += 1
    assert checked, "no ppl field was compared; this guard is vacuous"

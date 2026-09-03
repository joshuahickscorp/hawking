"""E.2 asserted against the roadmap's field lists, not against the receipt's own.

Three gates -- VMCP_STATE_LATTICE, VMCP_DEEP_DIGEST, VMCP_TRUTH_LEDGER -- were
BUILT with no test citing them: wired, acceptance-receipted, and unverified. That
is the shape the defining-property law exists to catch, because the only evidence
they worked was a document nobody compared against the obligation.

The obligation is roadmap E.2. Its field lists are TRANSCRIBED HERE from the
roadmap. A test that read the required fields out of the receipt and then asserted
the receipt had them would pass for any receipt, including an empty one -- the
oracle has to come from somewhere the implementation cannot edit.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
ACCEPTANCE = REPO / "receipts" / "acceptance"

# Transcribed from H-ROADMAP "## E.2 State lattice schemas". Ten named lattices.
E2_LATTICES = (
    "DEEP_DIGEST", "ASSET_LATTICE", "DECODE_LATTICE", "ENTITY_GENOME",
    "RENDER_GENOME", "SPATIAL_GENOME", "REPAIR_VECTOR", "DIRECTOR_STATE",
    "PERFORMANCE_LEDGER", "TRUTH_LEDGER",
)

# TRUTH_LEDGER: claims, evidence, counterevidence, confidence, blockers,
# no_op_detected. Every one must be exercised by the run, not merely declared.
E2_TRUTH_LEDGER_FIELDS = (
    "claims", "evidence", "counterevidence", "confidence", "blockers", "no_op",
)


def _receipt(gate: str) -> dict:
    path = ACCEPTANCE / f"{gate}.json"
    if not path.is_file():
        pytest.skip(f"{gate} has no acceptance receipt on this host")
    return json.loads(path.read_text())


def _check_ids(doc: dict) -> list[str]:
    return [str(c.get("id") or "") for c in (doc.get("checks") or [])]


@pytest.mark.parametrize("gate", ["VMCP_STATE_LATTICE", "VMCP_DEEP_DIGEST", "VMCP_TRUTH_LEDGER"])
def test_the_receipt_quotes_the_obligation_it_was_judged_against(gate):
    """A verdict that does not carry its criterion cannot be re-checked later."""
    doc = _receipt(gate)
    criterion = doc.get("criterion") or {}
    quoted = str(criterion.get("quoted") or "")
    assert "E.2 State lattice schemas" in quoted, (
        f"{gate} was accepted against something other than E.2"
    )
    assert doc.get("criterion_weakened") is False, f"{gate} weakened its own criterion"
    assert doc.get("verdict") == "ACCEPTED"


@pytest.mark.parametrize("gate", ["VMCP_STATE_LATTICE", "VMCP_DEEP_DIGEST", "VMCP_TRUTH_LEDGER"])
def test_every_check_actually_passed_rather_than_merely_running(gate):
    """A check that ran and failed is not evidence of a capability."""
    doc = _receipt(gate)
    failed = [c for c in doc.get("checks") or [] if not c.get("ok")]
    assert not failed, f"{gate} was ACCEPTED with failing checks: {[c['id'] for c in failed]}"
    assert doc.get("checks"), f"{gate} was accepted with no checks at all"


def test_truth_ledger_exercises_every_field_the_schema_names():
    """All six E.2 TRUTH_LEDGER fields, from the roadmap's list not the receipt's."""
    ids = " ".join(_check_ids(_receipt("VMCP_TRUTH_LEDGER")))
    missing = [f for f in E2_TRUTH_LEDGER_FIELDS if f not in ids]
    assert not missing, f"E.2 TRUTH_LEDGER fields never exercised: {missing}"


def test_the_lattice_gate_covers_all_ten_named_slots():
    """E.2 names ten lattices. Nine would be a silent gap."""
    assert len(E2_LATTICES) == 10
    ids = _check_ids(_receipt("VMCP_STATE_LATTICE"))
    assert "ten_named_slots" in ids, (
        "nothing checks that all ten E.2 slots are present"
    )


def test_deep_digest_is_a_digest_of_state_and_not_a_constant():
    """The defining property: it must CHANGE when the state changes.

    A digest that is stable under a value mutation is not a digest of the state;
    it is a label. That check must exist and must have passed.
    """
    ids = _check_ids(_receipt("VMCP_DEEP_DIGEST"))
    assert "value_mutation_changes_digest" in ids, (
        "nothing proves the digest responds to the state it claims to digest"
    )
    assert "canonical_key_order_stable" in ids, (
        "nothing proves key order is canonical, so two equal states could differ"
    )


def test_the_functions_the_receipts_claim_were_called_actually_exist():
    """A receipt naming a call it made is worthless if the callee does not exist.

    Each receipt asserts `prove_<x>_called`. That claim is only meaningful if the
    named symbol is real and reachable, so this ties the receipt back to the
    implementation instead of trusting a string inside the document. It also makes
    the test cite the module, which is how the auditor knows this gate is verified
    at all -- a test that verifies a capability without naming it is invisible.
    """
    from tools.headless import vmcp_lattice_disposition as lat

    for name in ("prove_deep_digest", "prove_truth_ledger",
                 "prove_asset_lattice", "prove_decode_lattice"):
        fn = getattr(lat, name, None)
        assert callable(fn), f"receipts claim {name} was called; no such callable exists"

    # And every prove_* the receipts name must be one of the module's own, not a
    # label invented in the document.
    module_proofs = {n for n in dir(lat) if n.startswith("prove_")}
    for gate in ("VMCP_STATE_LATTICE", "VMCP_DEEP_DIGEST", "VMCP_TRUTH_LEDGER"):
        for check in _check_ids(_receipt(gate)):
            if check.startswith("prove_") and check.endswith("_called"):
                claimed = check[: -len("_called")]
                assert claimed in module_proofs, (
                    f"{gate} claims {claimed} ran, but the module has no such function"
                )

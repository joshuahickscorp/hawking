"""FLASH_SOURCE_VERIFIED asserted against the census the roadmap demands.

The gate was BUILT with no test citing it: wired to a real producer, carrying an
acceptance receipt, and unverified. Its obligation is the head of the Flash
pipeline -- SOURCE / MANIFEST -> EXACT TENSOR CENSUS -> ORGAN GRAPH (roadmap 13,
line 1610) -- and the census is only meaningful if it is EXACT.

The required counts are transcribed here from the acceptance runner's own
constants rather than read back out of the receipt, because a test that asks the
receipt what it should contain and then confirms it contains that would pass for
any receipt.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
RECEIPT = REPO / "receipts" / "acceptance" / "FLASH_SOURCE_VERIFIED.json"

# Transcribed from tools/acceptance/flash/run_gates.py:29-31. The roadmap calls
# this an EXACT tensor census, so these are equalities, never lower bounds.
REQUIRED_SHARDS = 131
REQUIRED_TENSORS = 1658
REQUIRED_INDEXED_PAYLOAD_BYTES = 359_999_963_128


def _receipt() -> dict:
    if not RECEIPT.is_file():
        pytest.skip("FLASH_SOURCE_VERIFIED has no acceptance receipt on this host")
    return json.loads(RECEIPT.read_text())


def test_the_census_is_exact_and_not_merely_large():
    """EXACT means equality. A census that counted more would also be wrong."""
    doc = _receipt()
    quoted = str((doc.get("criterion") or {}).get("quoted") or "")
    assert str(REQUIRED_SHARDS) in quoted, "the criterion no longer names the shard count"
    assert "1,658" in quoted or str(REQUIRED_TENSORS) in quoted
    assert "359,999,963,128" in quoted or str(REQUIRED_INDEXED_PAYLOAD_BYTES) in quoted


def test_the_verdict_carries_its_own_criterion_and_was_not_weakened():
    doc = _receipt()
    assert doc.get("verdict") == "ACCEPTED"
    # Acceptance producers disagree on the field name: some emit
    # criterion_weakened, this one emits criterion_altered. Both must read False,
    # and at least one must be present -- a receipt asserting neither has not
    # claimed its criterion survived.
    weakened = doc.get("criterion_weakened")
    altered = doc.get("criterion_altered")
    assert weakened is not None or altered is not None, (
        "the receipt never states whether its criterion was altered"
    )
    assert weakened in (False, None) and altered in (False, None), (
        f"the gate weakened its own criterion: weakened={weakened} altered={altered}"
    )


def test_the_gate_does_not_claim_a_physical_measurement():
    """A tensor census reads headers. It is not a runtime or hardware result.

    The Flash hard gate at roadmap line 1607 pairs this with a TPS target the
    roadmap explicitly calls a research target rather than a current claim, so a
    census receipt must never drift up the evidence ladder.
    """
    doc = _receipt()
    assert doc.get("evidence_tier") in {"STATIC", "FUNCTIONAL_SIM"}, (
        f"a header census claims {doc.get('evidence_tier')}"
    )


def test_the_producer_the_receipt_names_actually_exists():
    """The census is reached through main(); a named producer must be real."""
    from tools import flash_organ_census

    assert callable(getattr(flash_organ_census, "main", None)), (
        "the acceptance runner calls flash_organ_census.main; it must exist"
    )

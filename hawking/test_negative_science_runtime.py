"""PROTECTED SOVEREIGN VERIFIER — G014

Negative science is durable, scoped, and reopenable.

RED GATE. This fails until the capability genuinely exists. It gates on a
durable receipt produced by a real run, never on a definition. A receipt with
missing, zero or placeholder measurements does not pass, and neither does a
model asserting completion.

If this gate is scientifically wrong it must be SUPERSEDED through protected
review with a negative control, not edited until it passes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

RECEIPT = Path("receipts/sovereign/G014_negative_science.json")


def _load():
    if not RECEIPT.is_file():
        pytest.fail(
            f"{RECEIPT} does not exist. G002 is not discharged. "
            "A verifier gates on durable evidence, not on intent."
        )
    try:
        return json.loads(RECEIPT.read_text(encoding="utf-8"))
    except ValueError as exc:
        pytest.fail(f"{RECEIPT} is not valid JSON: {exc}")


def _measured(doc, field):
    """A field that must be present, numeric and non-zero."""
    value = doc.get(field)
    assert value is not None, f"{field} is absent from {RECEIPT}"
    assert isinstance(value, (int, float)), f"{field} is not a measurement: {value!r}"
    assert value != 0, f"{field} is zero; a placeholder is not a measurement"
    return value


_STUB_MARKERS = ("stub", "fixture", "mock", "placeholder", "todo", "tbd",
                 "example", "sample", "negative-control", "dummy", "fake")


def test_producer_is_named_and_actually_ran():
    """A receipt nobody produced is not evidence, and a shape is not a run.

    A hand-written receipt with the right keys must NOT pass: the producer has
    to be named, non-stub, timestamped, and carry the command it ran.
    """
    doc = _load()
    producer = str(doc.get("produced_by") or "").strip()
    assert producer, "receipt has no `produced_by`; its producer is unknown"
    low = producer.lower()
    assert not any(m in low for m in _STUB_MARKERS), (
        f"producer {producer!r} names a stub; a fixture is not a run"
    )
    assert doc.get("schema"), "receipt has no schema"
    assert doc.get("produced_at"), "receipt has no `produced_at`; it may never have run"
    assert str(doc.get("command") or "").strip(), (
        "receipt records no `command`; nothing proves a producer executed"
    )


def test_claimed_completion_is_not_accepted_alone():
    """The negative control every gate shares: a status string proves nothing."""
    doc = _load()
    claim = str(doc.get("status") or "").lower()
    if claim in ("completed", "done", "ok", "success"):
        assert doc.get("evidence"), (
            "receipt claims completion but carries no `evidence`; "
            "a model saying done is not a result"
        )



REQUIRED = ("causal_question", "tested_family", "evidence", "reason_rejected",
            "scope", "reopen_if")


def test_every_scar_is_complete_and_scoped():
    doc = _load()
    scars = doc.get("scars")
    assert isinstance(scars, list) and scars, "no scars recorded"
    for s in scars:
        for f in REQUIRED:
            assert str(s.get(f) or "").strip(), f"scar missing {f}: {s.get('causal_question')}"
        assert s.get("scope") != "universal", (
            "a local failure was universalised; that is how a family gets wrongly condemned"
        )


def test_evidence_tier_is_explicit_and_physical_claims_have_physical_authority():
    doc = _load()
    scars = doc.get("scars") or []
    # Not vacuous: an empty list must not pass a loop-shaped assertion.
    assert scars, "no scars recorded; this gate cannot pass on an empty list"
    for s in scars:
        tier = str(s.get("evidence_tier") or "")
        assert tier, f"scar has no evidence tier: {s.get('causal_question')}"
        if s.get("physical_claim"):
            assert tier == "physical", (
                "a physical claim was promoted on non-physical evidence"
            )


def test_dead_families_are_not_reburned():
    doc = _load()
    assert "recomputed_dead_family_seconds" in doc, (
        "the field is absent; a default of 0 would pass this gate vacuously"
    )
    assert doc["recomputed_dead_family_seconds"] == 0, (
        "compute was spent re-testing an already-scarred family"
    )

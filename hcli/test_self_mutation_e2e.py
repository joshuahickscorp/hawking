"""PROTECTED SOVEREIGN VERIFIER — G003

One useful mutation originates in HCLI, is verified, and crosses the integration boundary.

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

RECEIPT = Path("receipts/sovereign/G003_self_mutation.json")


def _load():
    if not RECEIPT.is_file():
        pytest.fail(
            f"{RECEIPT} does not exist. G003 is not discharged. "
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



def test_the_mutation_originated_in_hcli_not_a_human():
    doc = _load()
    assert doc.get("origin") == "hcli_resident", (
        f"origin is {doc.get('origin')!r}; a human-authored change is not self-improvement"
    )
    assert str(doc.get("deficiency_measurement") or "").strip(), (
        "no measured deficiency motivated the mutation"
    )


def test_the_child_was_isolated_and_the_incumbent_untouched():
    doc = _load()
    assert doc.get("child_worktree"), "no isolated child worktree"
    assert doc.get("incumbent_rewritten_in_place") is False, (
        "the live incumbent was rewritten underneath itself"
    )
    assert doc.get("rollback_target"), "no rollback target preserved"


def test_it_crossed_the_real_integration_boundary():
    doc = _load()
    assert doc.get("crossed_integration_boundary") is True, (
        "READY_TO_LAND is not landed; the boundary was not crossed"
    )
    assert doc.get("commit"), "no durable commit provenance"
    assert doc.get("verified_by_protected_gate") is True, (
        "the child self-certified; a protected gate must have accepted it"
    )

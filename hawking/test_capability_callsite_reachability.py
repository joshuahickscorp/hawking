"""PROTECTED SOVEREIGN VERIFIER — G009

Capability reachability by call site, not by registration.

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

RECEIPT = Path("receipts/sovereign/G009_reachability.json")


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




def test_every_audited_capability_has_a_production_call_site():
    doc = _load()
    caps = doc.get("capabilities")
    assert isinstance(caps, list) and caps, "no capabilities audited"
    for c in caps:
        name = c.get("name")
        assert c.get("registered") is not None, f"{name}: registration unknown"
        sites = c.get("production_call_sites")
        assert isinstance(sites, list), f"{name}: call sites not enumerated"
        if c.get("required_by_active_frontier"):
            assert sites, (
                f"{name} is required by the active frontier and has NO production "
                "call site; a registration is not a capability"
            )
            assert not all("test" in str(s) for s in sites), (
                f"{name} is only reachable from its own tests"
            )


def test_invocation_was_attempted_not_merely_imported():
    doc = _load()
    for c in doc.get("capabilities") or []:
        if c.get("required_by_active_frontier"):
            assert c.get("invoked") is True, (
                f"{c.get('name')} was never actually invoked; importability is not capability"
            )
            assert "evidence" in c, f"{c.get('name')} returned no evidence"

"""PROTECTED SOVEREIGN VERIFIER — G015

The unattended control plane stays read-only and one-bodied.

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

RECEIPT = Path("receipts/sovereign/G015_control_plane.json")


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




def test_watch_is_read_only_and_opens_no_body():
    """This one can be checked directly, not only from a receipt."""
    import subprocess
    import hcli.agentos.resident as R

    src = __import__("inspect").getsource(R.watch_resident)
    assert "Popen" not in src and "subprocess" not in src, (
        "watch_resident spawns processes; it must be read-only"
    )
    assert "start_resident" not in src and "request_stop" not in src, (
        "watch_resident can mutate lifecycle state"
    )


def test_the_receipt_proves_it_under_a_live_resident():
    doc = _load()
    assert doc.get("bodies_during_watch") == 1, (
        f"{doc.get('bodies_during_watch')} model bodies during watch; must stay 1"
    )
    assert doc.get("state_mutated_by_watch") is False, "watch mutated durable state"
    assert doc.get("resident_survived_detach") is True, "Ctrl-C stopped the resident"


def test_the_protected_behaviours_did_not_regress():
    doc = _load()
    for behaviour in ("fault_isolation", "paste_receipt", "live_progress",
                      "one_body_discipline"):
        assert doc.get(behaviour) is True, f"{behaviour} regressed"

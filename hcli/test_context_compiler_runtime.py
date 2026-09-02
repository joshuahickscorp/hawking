"""PROTECTED SOVEREIGN VERIFIER — G004

GoalIR and the context compiler are on the live runtime path.

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

RECEIPT = Path("receipts/sovereign/G004_context_runtime.json")


def _load():
    if not RECEIPT.is_file():
        pytest.fail(
            f"{RECEIPT} does not exist. G004 is not discharged. "
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


FIELDS = ("source_tokens", "active_tokens", "retrieved_tokens",
          "tool_schema_tokens", "system_tokens", "generation_headroom")


def test_the_compiled_goal_reaches_the_posted_payload():
    doc = _load()
    assert doc.get("compiled_used_on_live_path") is True, (
        "compiled goal state is still discarded before the model call"
    )
    assert doc.get("raw_source_hash"), "raw source was not hashed"
    assert doc.get("raw_source_path"), "raw source has no durable location"


def test_the_accounting_is_measured():
    doc = _load()
    for f in FIELDS:
        _measured(doc, f)
    assert doc["active_tokens"] < doc["source_tokens"], (
        "active context is not smaller than source; nothing was compiled"
    )


def test_retrieval_closed_the_loop_without_inflating_every_turn():
    doc = _load()
    assert doc.get("retrieval_events"), "the model never retrieved an exact span"
    assert doc.get("retrieval_ns") is not None, "retrieval latency unmeasured"
    assert doc.get("steady_state_active_tokens") <= doc["active_tokens"], (
        "retrieved material permanently inflated the active window"
    )

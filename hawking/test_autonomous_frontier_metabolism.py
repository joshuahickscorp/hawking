"""PROTECTED SOVEREIGN VERIFIER — G008

Autonomous metabolism: evidence causes multiple replans with no human steering.

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

RECEIPT = Path("receipts/sovereign/G008_metabolism.json")


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




def test_distinct_causal_questions_not_prompt_count():
    doc = _load()
    q = doc.get("distinct_causal_questions")
    assert isinstance(q, int) and q >= 3, (
        f"only {q} distinct causal questions; launches are not questions"
    )


def test_replans_were_caused_by_evidence_and_unattended():
    doc = _load()
    replans = doc.get("replans")
    assert isinstance(replans, list) and len(replans) >= 2, (
        "fewer than two autonomous replans; this is not metabolism"
    )
    for r in replans:
        assert r.get("caused_by_evidence"), f"replan without evidence: {r}"
    assert doc.get("human_steering_events") == 0, (
        "a human steered the trial; it does not demonstrate autonomy"
    )


def test_a_stuck_frontier_did_not_stall_the_resident():
    doc = _load()
    assert doc.get("parked_frontiers") is not None, "no park behaviour recorded"
    assert doc.get("zero_progress_intervals") == 0, (
        "the resident spent intervals producing no accepted work and no new evidence"
    )

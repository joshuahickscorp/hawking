"""PROTECTED SOVEREIGN VERIFIER — G001

Goal compilation synthesises executable verifiers with no human-named test file.

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

RECEIPT = Path("receipts/sovereign/G001_verifier_synthesis.json")


def _load():
    if not RECEIPT.is_file():
        pytest.fail(
            f"{RECEIPT} does not exist. G001 is not discharged. "
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



def test_a_goal_with_no_test_filename_still_gets_a_real_verifier():
    """The whole point: today an obligation only gets a verifier if the prose
    literally names a test*.py file (`GoalCompiler._verify_command`)."""
    doc = _load()
    cases = doc.get("cases")
    assert isinstance(cases, list) and cases, "no synthesis cases recorded"
    for case in cases:
        text = str(case.get("goal_text") or "")
        assert "test" not in text.lower() or ".py" not in text, (
            "case named a test file; that is the mechanism being replaced"
        )
        assert str(case.get("verifier") or "").strip(), (
            f"no verifier synthesised for {text[:60]!r}"
        )
        assert case.get("red_before_green") is True, (
            "the synthesised gate was never proven to bite while red"
        )


def test_the_synthesised_verifier_is_not_a_tautology():
    doc = _load()
    for case in doc.get("cases") or []:
        cmd = str(case.get("verifier") or "")
        assert cmd.strip() not in ("true", ":", "exit 0", "/usr/bin/true"), cmd
        assert case.get("negative_control_failed") is True, (
            "the negative control did not fail; the gate cannot detect absence"
        )

"""PROTECTED SOVEREIGN VERIFIER — G010

ModelLake retained verified bytes per wall second.

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

RECEIPT = Path("receipts/sovereign/G010_modellake_retained.json")


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




def test_the_metric_is_retained_not_instantaneous():
    doc = _load()
    assert doc.get("metric") == "retained_verified_bytes_per_wall_second", (
        f"wrong primary metric: {doc.get('metric')!r}"
    )
    rate = _measured(doc, "retained_bytes_per_s")
    assert rate > 0, "retained rate is not positive; acquisition is not converging"


def test_the_window_is_long_enough_to_mean_anything():
    doc = _load()
    window = _measured(doc, "window_s")
    assert window >= 600, f"window {window}s is too short to dominate startup noise"
    assert doc.get("restarts") is not None, "restart count not recorded"


def test_lifecycle_defects_are_detected_not_assumed_absent():
    doc = _load()
    checks = doc.get("lifecycle_checks")
    assert isinstance(checks, dict), "no lifecycle audit"
    for name in ("complete_but_unpromoted", "stale_partial", "restart_data_loss",
                 "manifest_mismatch", "duplicate_acquisition"):
        assert name in checks, f"{name} was never checked"

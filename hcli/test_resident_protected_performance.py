"""PROTECTED SOVEREIGN VERIFIER — G012

Protected resident decode performance, measured now, not recalled.

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

RECEIPT = Path("receipts/sovereign/G012_resident_perf.json")


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




def test_the_authority_is_a_fresh_measurement():
    doc = _load()
    assert doc.get("measured_at"), "no measurement timestamp"
    assert doc.get("reused_historical_figure") is False, (
        "a historical 34/36/45/62 TPS figure was reused as current authority"
    )
    _measured(doc, "complete_token_ns")
    _measured(doc, "decode_tok_s")


def test_conditions_were_controlled_and_contamination_recorded():
    doc = _load()
    conds = doc.get("conditions")
    assert isinstance(conds, dict), "no controlled conditions recorded"
    for f in ("swap_used_bytes", "background_contamination", "free_bytes"):
        assert f in conds, f"condition {f} unrecorded"


def test_the_single_stream_saturation_claim_was_retested():
    doc = _load()
    agg = doc.get("aggregate_vs_single")
    assert isinstance(agg, dict) and agg.get("single_tok_s") and agg.get("aggregate_tok_s"), (
        "the multi-stream saturation claim was assumed rather than retested"
    )

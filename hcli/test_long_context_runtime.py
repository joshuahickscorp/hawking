"""PROTECTED SOVEREIGN VERIFIER — G006

Native long-context ladder at 131072 and 262144, measured.

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

RECEIPT = Path("receipts/sovereign/G006_long_context.json")


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



LADDER = ("131072", "262144")


def test_both_rungs_were_actually_served():
    doc = _load()
    rungs = doc.get("rungs")
    assert isinstance(rungs, dict), "no ladder recorded"
    for r in LADDER:
        assert r in rungs, f"rung {r} was never attempted"
        entry = rungs[r]
        assert entry.get("admitted_limit") == int(r), (
            f"rung {r} admitted {entry.get('admitted_limit')}; a clamp is still in force"
        )
        assert entry.get("generated") is True, f"rung {r} produced no generation"


def test_each_rung_carries_physical_measurements():
    doc = _load()
    for r in LADDER:
        e = doc["rungs"][r]
        for f in ("prefill_s", "decode_tok_s", "kv_bytes", "recurrent_state_bytes",
                  "total_memory_bytes"):
            v = e.get(f)
            assert isinstance(v, (int, float)) and v != 0, (
                f"rung {r} field {f} is not a measurement: {v!r}"
            )
        assert "swap_used_bytes" in e, f"rung {r} did not record swap behaviour"


def test_kv_accounting_was_re_derived_not_assumed():
    doc = _load()
    assert doc.get("kv_bytes_per_token_source") == "measured", (
        "KV per token was assumed from history rather than re-derived"
    )
    assert doc.get("sealed_artifact_mutated") is False, "a sealed artifact was mutated"

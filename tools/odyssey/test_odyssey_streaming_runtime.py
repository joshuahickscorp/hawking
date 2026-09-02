"""PROTECTED SOVEREIGN VERIFIER — G011

Odyssey streams I/II/III with no global barrier.

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

RECEIPT = Path("receipts/sovereign/G011_odyssey_streaming.json")


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




def test_the_three_streams_overlapped_in_wall_time():
    doc = _load()
    streams = doc.get("streams")
    assert isinstance(streams, dict), "no stream timing recorded"
    for s in ("I", "II", "III"):
        assert s in streams, f"stream {s} never ran"
        assert streams[s].get("start_s") is not None and streams[s].get("end_s"), s
    spans = [(streams[s]["start_s"], streams[s]["end_s"]) for s in ("I", "II", "III")]
    overlapped = any(
        a[0] < b[1] and b[0] < a[1] for i, a in enumerate(spans) for b in spans[i + 1:]
    )
    assert overlapped, "streams ran strictly serially; that is a global barrier"


def test_an_incomplete_specimen_did_not_block_independent_science():
    doc = _load()
    assert doc.get("blocked_on_incomplete_specimen") is False, (
        "science stalled waiting for a download"
    )
    assert doc.get("hcli_owned") is True, "Odyssey was driven by a human shell, not HCLI"


def test_progress_is_scientific_not_bookkeeping():
    doc = _load()
    assert _measured(doc, "laws_or_scars_added") > 0, (
        "no laws or scars were produced; WU count is not scientific movement"
    )

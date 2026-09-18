"""PROTECTED SOVEREIGN VERIFIER — G005

f32 GEMM prefill coverage, measured, with the dispatch gate passing honestly.

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

RECEIPT = Path("receipts/sovereign/G005_prefill_pipeline.json")


def _load():
    if not RECEIPT.is_file():
        pytest.fail(
            f"{RECEIPT} does not exist. G005 is not discharged. "
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



def test_the_f32_tensors_are_covered():
    doc = _load()
    census = doc.get("tensor_census")
    assert isinstance(census, dict), "no tensor census recorded"
    assert census.get("total"), "census has no total"
    uncovered = census.get("prefill_uncovered")
    assert uncovered is not None, "coverage was not measured"
    assert uncovered == 0, (
        f"{uncovered} tensors still fall through the non-batched path"
    )


def test_throughput_and_identity_are_both_proven():
    doc = _load()
    before = _measured(doc, "prefill_tok_s_before")
    after = _measured(doc, "prefill_tok_s_after")
    assert after > before, f"no prefill improvement: {before} -> {after}"
    assert doc.get("generated_token_identity") is True, (
        "batched and sequential paths do not produce identical tokens"
    )
    assert doc.get("decode_regressed") is False, "decode regressed"
    assert doc.get("metal_compiled") is True, "shaders did not compile on real Metal"


def test_the_dispatch_gate_passes_without_being_relaxed():
    doc = _load()
    assert doc.get("dispatch_gate_passed") is True, "the in-tree dispatch gate is red"
    assert doc.get("dispatch_gate_threshold_changed") is False, (
        "the gate was relaxed instead of satisfied"
    )

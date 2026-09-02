"""PROTECTED SOVEREIGN VERIFIER — G002

End-to-end time attribution for the ~2x HCLI control-plane gap.

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

RECEIPT = Path("receipts/sovereign/G002_hcli_overhead.json")


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


STAGES = (
    "context_construction_ns", "retrieval_ns", "provider_prepare_ns",
    "serialization_ns", "transport_ns", "native_prefill_ns", "native_decode_ns",
    "parse_ns", "tool_dispatch_ns", "verifier_ns", "evidence_ingest_ns",
    "schedule_ns",
)


def test_a_paired_comparison_exists():
    doc = _load()
    direct = _measured(doc, "direct_tok_s")
    hcli = _measured(doc, "hcli_tok_s")
    assert doc.get("paired") is True, "the two rates were not measured paired"
    assert direct > 0 and hcli > 0


def test_every_stage_is_attributed():
    """Plausibility is not attribution. Each stage carries measured nanoseconds."""
    doc = _load()
    stages = doc.get("stages")
    assert isinstance(stages, dict), "no per-stage attribution recorded"
    missing = [s for s in STAGES if s not in stages]
    assert not missing, f"unattributed stages: {missing}"
    total = sum(float(stages[s]) for s in STAGES)
    assert total > 0, "all stages are zero; nothing was measured"


def test_the_dominant_loss_is_identified_and_discriminated():
    doc = _load()
    dom = str(doc.get("dominant_cause") or "").strip()
    assert dom, "no dominant cause identified"
    assert dom in (doc.get("stages") or {}), f"{dom} is not one of the measured stages"
    hyps = doc.get("competing_hypotheses")
    assert isinstance(hyps, list) and len(hyps) >= 2, (
        "a single hypothesis is an assumption, not a discrimination"
    )
    assert doc.get("discriminator_command"), "no decisive discriminator was run"

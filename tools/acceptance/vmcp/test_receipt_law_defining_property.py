"""VMCP_RECEIPT_LAW asserted against its defining property, not against itself.

The gate is BUILT -- wired, with an acceptance receipt -- and until now NO test
cited it. That combination is the exact shape the defining-property law warns
about: a capability whose only evidence that it works is that something ran and
produced a document nobody checked against the obligation.

The obligation is roadmap E.4: a ToolReceipt carries

    tool, version, invocation, input_ids, input_hashes, output_ids,
    output_hashes, started_at, elapsed_ms, status, limitations, verifier, canary

and "a truth-affecting tool with no trace is not part of the proof."

The oracle here is that field list, transcribed from the roadmap rather than read
back out of the implementation. A test that asked the receipt module what fields
a receipt has, and then asserted the receipt has them, would pass for any receipt
whatsoever -- including an empty one.
"""
from __future__ import annotations

import pytest

from tools.vmcp import pty_eye, tool_doctor
from tools.vmcp.receipt import tool_receipt

# Transcribed from H-ROADMAP E.4. Independent of the implementation ON PURPOSE.
E4_REQUIRED_FIELDS = (
    "tool",
    "version",
    "invocation",
    "input_ids",
    "input_hashes",
    "output_ids",
    "output_hashes",
    "started_at",
    "elapsed_ms",
    "status",
    "limitations",
    "verifier",
    "canary",
)


def test_a_receipt_carries_every_field_the_law_names():
    receipt = tool_receipt(
        tool="probe",
        invocation=["/bin/echo", "hi"],
        status="ok",
        started_at="2026-01-01T00:00:00Z",
        elapsed_ms=1.5,
        verifier="tools.vmcp.receipt.tool_receipt",
    )
    missing = [f for f in E4_REQUIRED_FIELDS if f not in receipt]
    assert not missing, f"ToolReceipt is missing law-required fields: {missing}"


def test_a_truth_affecting_tool_leaves_a_trace():
    """The organs that make claims must emit a receipt, not merely return data.

    This is the half of the law that has teeth: "a truth-affecting tool with no
    trace is not part of the proof". A tool that answers correctly and records
    nothing fails the obligation even though its answer is right.
    """
    captured = pty_eye.capture(argv=["/bin/echo", "receipt-law"], timeout_s=10)
    assert captured.get("ok") is True
    receipt = captured.get("tool_receipt")
    assert receipt, "pty_eye.capture made a claim and left no ToolReceipt"
    missing = [f for f in E4_REQUIRED_FIELDS if f not in receipt]
    assert not missing, f"pty_eye receipt missing law-required fields: {missing}"

    profiled = tool_doctor.profile(argv=["/bin/echo", "receipt-law"])
    assert profiled.get("tool_receipt"), "tool_doctor.profile left no ToolReceipt"


def test_the_receipt_describes_THIS_invocation_and_not_a_generic_one():
    """A receipt that does not bind to its own call is a template, not a trace.

    Two different invocations must not produce the same trace. Without this, a
    hardcoded receipt would satisfy every other assertion in this file.
    """
    a = pty_eye.capture(argv=["/bin/echo", "alpha"], timeout_s=10)["tool_receipt"]
    b = pty_eye.capture(argv=["/bin/echo", "beta"], timeout_s=10)["tool_receipt"]
    assert a["invocation"] != b["invocation"], "the receipt does not record its own argv"
    assert "alpha" in str(a["invocation"])
    assert "beta" in str(b["invocation"])


def test_status_is_reported_not_assumed_successful():
    """A tool that always says ok cannot report a refusal, and refusals matter."""
    refused = tool_doctor.profile(argv=["curl", "https://example.com"])
    receipt = refused.get("tool_receipt")
    assert receipt, "a refusal must still leave a trace"
    assert receipt["status"] != "ok", (
        f"a network tool was not refused: status={receipt['status']}"
    )
    assert receipt["limitations"], "a refusal with no stated limitation explains nothing"

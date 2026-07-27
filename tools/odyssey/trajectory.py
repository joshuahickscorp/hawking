#!/usr/bin/env python3.12
"""Trajectory-loss interface for Odyssey T3 apparatus.

The *shape* of the loss is defined and unit-tested. Real trajectory work needs
parent trajectory traces. As of the teacher ledger assessment:

  - ledger lines: 122
  - per-layer captures (TEACHER_CAPTURED): 118
  - trajectory traces: 0

Therefore this module refuses to claim a real T3 loss over parent traces. It
will compute the metric over caller-supplied token sequences (including fixture
sequences) and always annotate the missing-trace condition.
"""
from __future__ import annotations

from typing import Any, Sequence

SCHEMA = "hawking.odyssey.trajectory_loss.interface.v1"

TEACHER_LEDGER_FACTS = {
    "ledger_lines": 122,
    "per_layer_captures": 118,
    "trajectory_traces": 0,
    "source": (
        "GLM52_TEACHER_EVIDENCE_LEDGER.jsonl "
        "(~/Library/Application Support/Hawking/GLM52Gravity/source_fetch/teacher/)"
    ),
    "implication": (
        "Real T3 trajectory stabilization cannot run: parent trajectory traces do not exist. "
        "Layer-scoped capsules (usable for T2-style work) are not trajectory traces."
    ),
}


class TrajectoryTracesMissing(RuntimeError):
    """Raised when a caller demands a real parent-trace loss and none exist."""


def trajectory_loss_shape(
    parent_tokens: Sequence[int],
    student_tokens: Sequence[int],
    *,
    source: str = "caller_supplied",
) -> dict[str, Any]:
    """Token-level divergence metric. Shape only when source is not real parent traces.

    Returns first divergence index, mismatch fraction, and lengths. Always
    includes the teacher-ledger fact that 0 trajectory traces exist.
    """
    parent = list(parent_tokens)
    student = list(student_tokens)
    n = max(len(parent), len(student))
    if n == 0:
        first = None
        mismatch_fraction = 0.0
        mismatches = 0
    else:
        first = None
        for i in range(min(len(parent), len(student))):
            if parent[i] != student[i]:
                first = i
                break
        if first is None and len(parent) != len(student):
            first = min(len(parent), len(student))
        mismatches = sum(
            1
            for i in range(n)
            if (parent[i] if i < len(parent) else None)
            != (student[i] if i < len(student) else None)
        )
        mismatch_fraction = mismatches / n

    is_fixture = source != "parent_trajectory_trace"
    return {
        "schema": SCHEMA,
        "status": "INTERFACE_ONLY" if is_fixture else "COMPUTED",
        "kind": "fixture_shape" if is_fixture else "parent_trace_loss",
        "first_divergence_index": first,
        "mismatch_fraction": mismatch_fraction,
        "n": n,
        "parent_len": len(parent),
        "student_len": len(student),
        "source": source,
        "teacher_ledger": dict(TEACHER_LEDGER_FACTS),
        "note": (
            "Trajectory-loss interface exercised on caller-supplied tokens. "
            "Real trajectory work needs parent traces which do not exist "
            f"({TEACHER_LEDGER_FACTS['trajectory_traces']} trajectory traces in the ledger)."
        ),
        "is_measurement_of_parent": False,
    }


def require_parent_traces_or_refuse() -> dict[str, Any]:
    """The honest gate: refuse real T3 until trajectory traces exist."""
    if TEACHER_LEDGER_FACTS["trajectory_traces"] == 0:
        raise TrajectoryTracesMissing(
            "REFUSED: 0 trajectory traces in teacher ledger "
            f"({TEACHER_LEDGER_FACTS['ledger_lines']} lines, "
            f"{TEACHER_LEDGER_FACTS['per_layer_captures']} per-layer captures). "
            "Cannot compute a real parent trajectory loss."
        )
    return {"status": "READY", "teacher_ledger": dict(TEACHER_LEDGER_FACTS)}

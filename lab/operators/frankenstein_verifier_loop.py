#!/usr/bin/env python3.12
"""Stage 8 — verified expert iteration loop interface (fail closed).

Loop: DSV4F attempt → verifier/tool → GLM critique on failure → repair →
verified trajectory → training.

No verifier/Lean/tool loop is wired yet → REQUIRES_VERIFIER.
No training loop → REQUIRES_TRAINING_LOOP for the train step.
Never fabricates trajectories or verifier outcomes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from lab.operators.frankenstein_gates import (
    REQUIRES_GLM_RUNTIME,
    REQUIRES_TRAINING_LOOP,
    REQUIRES_VERIFIER,
    fail_closed,
    gate_record,
)
from lab.receipts import seal


VERIFIER_LOOP_SCHEMA = "hawking.frankenstein.verified_expert_iteration.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def loop_interface_spec() -> dict[str, Any]:
    return seal(
        {
            "schema": VERIFIER_LOOP_SCHEMA,
            "recorded_at": _utc_now(),
            "status": "INTERFACE_ONLY",
            "steps": [
                {"id": 1, "name": "dsv4f_attempt", "gate": None},
                {"id": 2, "name": "verifier_or_tool", "gate": REQUIRES_VERIFIER},
                {
                    "id": 3,
                    "name": "glm_critique_on_failure",
                    "gate": REQUIRES_GLM_RUNTIME,
                },
                {"id": 4, "name": "repair", "gate": None},
                {"id": 5, "name": "verified_trajectory", "gate": REQUIRES_VERIFIER},
                {"id": 6, "name": "training", "gate": REQUIRES_TRAINING_LOOP},
            ],
            "gates": {
                REQUIRES_VERIFIER: gate_record(REQUIRES_VERIFIER),
                REQUIRES_GLM_RUNTIME: gate_record(REQUIRES_GLM_RUNTIME),
                REQUIRES_TRAINING_LOOP: gate_record(REQUIRES_TRAINING_LOOP),
            },
            "fabricated_trajectories": False,
            "note": (
                "Interface only.  run_verified_expert_iteration fails closed when "
                "verifier / GLM / training infra is absent."
            ),
        }
    )


def run_verified_expert_iteration(
    *,
    problem: Mapping[str, Any] | None = None,
    dsv4f_attempt_fn: Any = None,
    verifier_fn: Any = None,
    glm_critique_fn: Any = None,
    train_fn: Any = None,
) -> dict[str, Any]:
    """Execute the loop or fail closed at the first missing dependency."""

    if verifier_fn is None:
        closed = fail_closed(
            REQUIRES_VERIFIER,
            stage="8_verified_expert_iteration",
            operation="run_verified_expert_iteration",
        )
        closed["schema"] = VERIFIER_LOOP_SCHEMA
        closed["recorded_at"] = _utc_now()
        closed["fabricated"] = False
        return seal(closed)

    if dsv4f_attempt_fn is None:
        return seal(
            {
                "schema": VERIFIER_LOOP_SCHEMA,
                "status": "FAIL_CLOSED",
                "gate": "REQUIRES_DSV4F_ATTEMPT_FN",
                "executed": False,
                "fabricated": False,
                "recorded_at": _utc_now(),
                "note": "dsv4f_attempt_fn not provided",
            }
        )

    # Real path would call the functions; we only do so when all are present.
    attempt = dsv4f_attempt_fn(problem)
    verdict = verifier_fn(attempt)
    if not verdict.get("ok") and glm_critique_fn is None:
        closed = fail_closed(
            REQUIRES_GLM_RUNTIME,
            stage="8_verified_expert_iteration",
            operation="glm_critique_on_failure",
        )
        closed["schema"] = VERIFIER_LOOP_SCHEMA
        closed["attempt_present"] = True
        closed["verifier_ok"] = False
        return seal(closed)

    if train_fn is None:
        closed = fail_closed(
            REQUIRES_TRAINING_LOOP,
            stage="8_verified_expert_iteration",
            operation="train_on_verified_trajectory",
        )
        closed["schema"] = VERIFIER_LOOP_SCHEMA
        return seal(closed)

    # Fully wired path — still not inventing; just orchestrating callables.
    critique = None if verdict.get("ok") else glm_critique_fn(attempt, verdict)
    trajectory = {
        "attempt": attempt,
        "verifier": verdict,
        "critique": critique,
        "verified": bool(verdict.get("ok")),
    }
    train_result = train_fn(trajectory)
    return seal(
        {
            "schema": VERIFIER_LOOP_SCHEMA,
            "status": "EXECUTED",
            "recorded_at": _utc_now(),
            "trajectory": trajectory,
            "train_result_keys": sorted(train_result.keys())
            if isinstance(train_result, Mapping)
            else None,
            "fabricated": False,
        }
    )

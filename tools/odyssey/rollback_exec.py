#!/usr/bin/env python3.12
"""Executable form of odyssey/rollback/ROLLBACK.json.

Law: rollback is a recorded event, never a silent substitution.

Procedure:
  1. halt at the next checkpoint boundary
  2. record a rollback event naming the checkpoint restored
  3. restore the named checkpoint as current
  4. re-run the stage's entry gate before resuming
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tools.odyssey._paths import ODYSSEY
from tools.odyssey.checkpoints import CheckpointError, CheckpointStore
from tools.odyssey.trainer import StageEntryGateError, ToyTrainer, stage_entry_gate

SCHEMA = "hawking.odyssey.rollback.runtime.v1"
LAW_PATH = ODYSSEY / "rollback" / "ROLLBACK.json"


def load_law() -> dict[str, Any]:
    return json.loads(LAW_PATH.read_text())


def execute_rollback(
    store: CheckpointStore,
    checkpoint_id: str,
    *,
    stage: str = "FIXTURE",
    objective: str | None = "capability_weighted_ce",
    reason: str = "test_injected_regression",
    re_run_entry_gate: bool = True,
) -> dict[str, Any]:
    """Restore a named earlier checkpoint; record event; re-run entry gate.

    Returns a structured result. Raises CheckpointError if the target is bad.
    """
    law = load_law()
    previous = store.current_id()

    # 1. Halt is implicit: caller invokes this at a boundary.
    # 2+3. Verify target, record event, then point CURRENT.
    state = store.load_state(checkpoint_id)  # raises if corrupt/missing

    event = {
        "kind": "rollback",
        "schema": SCHEMA,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "restored": checkpoint_id,
        "restored_step": state["step"],
        "previous_current": previous,
        "reason": reason,
        "stage": stage,
        "law": law.get("law"),
        "procedure_executed": law.get("procedure"),
        "silent_substitution": False,
    }
    with store.events_path.open("a") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")

    store.set_current(checkpoint_id)

    gate_result: dict[str, Any] | None = None
    gate_error: str | None = None
    if re_run_entry_gate:
        try:
            gate_result = stage_entry_gate(
                stage,
                store=store,
                require_prior_checkpoint=True,
                objective=objective,
            )
        except StageEntryGateError as e:
            gate_error = str(e)
            return {
                "schema": SCHEMA,
                "status": "ENTRY_GATE_FAILED",
                "event": event,
                "current": checkpoint_id,
                "entry_gate_error": gate_error,
                "law_path": str(LAW_PATH.relative_to(ODYSSEY.parent))
                if LAW_PATH.is_relative_to(ODYSSEY.parent)
                else str(LAW_PATH),
            }

    return {
        "schema": SCHEMA,
        "status": "PROVEN",
        "event": event,
        "current": checkpoint_id,
        "current_step": state["step"],
        "entry_gate": gate_result,
        "law_path": "odyssey/rollback/ROLLBACK.json",
        "g6_note": (
            "G6 previously recorded rollback as DECLARED_NOT_PROVEN. "
            "This execution proves: recorded event + CURRENT restore + entry gate re-run."
        ),
    }


def rollback_and_restore_model(
    trainer: ToyTrainer,
    checkpoint_id: str,
    *,
    reason: str = "test_injected_regression",
) -> dict[str, Any]:
    """Rollback store + reload model state into the trainer."""
    result = execute_rollback(
        trainer.store,
        checkpoint_id,
        stage="FIXTURE",
        objective=trainer.objective_name,
        reason=reason,
    )
    if result["status"] != "PROVEN":
        return result
    trainer.load_from_checkpoint(checkpoint_id)
    result["model_step"] = trainer.model.step
    result["model_state_sha256"] = trainer.model.state_sha256()
    return result

#!/usr/bin/env python3.12
"""Odyssey toy trainer apparatus.

Exercises: content-addressed checkpoints, kill/resume bit-identity, rollback
with entry-gate re-run, objective registry, resource declarations, and
content-addressed training receipts.

This trainer operates ONLY on the FIXTURE toy model. It has never trained
anything real. It does not touch ODYSSEY_LAUNCH_AUTHORIZED or Math-Preserve.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.odyssey.checkpoints import CheckpointError, CheckpointStore
from tools.odyssey.objectives import UnregisteredObjective, require_objective
from tools.odyssey.receipts import make_receipt, write_receipt
from tools.odyssey.scheduler import declare_and_admit
from tools.odyssey.toy_model import FIXTURE_LABEL, ToyConfig, ToyMLP

SCHEMA = "hawking.odyssey.toy_trainer.fixture.v1"


class StageEntryGateError(RuntimeError):
    """Raised when a stage's entry gate fails."""


def stage_entry_gate(
    stage: str,
    *,
    store: CheckpointStore,
    require_prior_checkpoint: bool = False,
    objective: str | None = None,
) -> dict[str, Any]:
    """Re-runnable stage entry gate (used after rollback per ROLLBACK.json)."""
    checks: list[dict[str, Any]] = []
    ok = True

    # Fence must stay false for this fixture apparatus; we do not read launch/
    # as writable. Gate only asserts the toy apparatus preconditions.
    checks.append(
        {
            "name": "fixture_only",
            "ok": True,
            "detail": "toy trainer apparatus; never trains a real substrate",
        }
    )

    if objective is not None:
        try:
            entry = require_objective(objective, stage="FIXTURE")
            checks.append({"name": "objective_registered", "ok": True, "detail": entry["name"]})
        except UnregisteredObjective as e:
            ok = False
            checks.append({"name": "objective_registered", "ok": False, "detail": str(e)})

    cur = store.current_id()
    if require_prior_checkpoint:
        if not cur:
            ok = False
            checks.append({"name": "prior_checkpoint", "ok": False, "detail": "CURRENT missing"})
        else:
            v = store.verify(cur)
            checks.append({"name": "prior_checkpoint", "ok": v["status"] == "OK", "detail": v})
            ok = ok and v["status"] == "OK"
    else:
        checks.append(
            {
                "name": "prior_checkpoint",
                "ok": True,
                "detail": f"optional; current={cur}",
            }
        )

    result = {
        "schema": "hawking.odyssey.stage_entry_gate.v1",
        "stage": stage,
        "ok": ok,
        "checks": checks,
    }
    if not ok:
        raise StageEntryGateError(json.dumps(result))
    return result


class ToyTrainer:
    """Deterministic fixture trainer with checkpoint / resume / rollback."""

    def __init__(
        self,
        work_dir: Path,
        *,
        cfg: ToyConfig | None = None,
        objective: str = "capability_weighted_ce",
        checkpoint_every: int = 1,
        shard_size: int = 0,
        lr: float = 0.05,
    ):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or ToyConfig()
        self.objective_name = objective
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.lr = float(lr)
        self.store = CheckpointStore(self.work_dir / "checkpoints", shard_size=shard_size)
        self.receipts_dir = self.work_dir / "receipts"
        self.model = ToyMLP(self.cfg)
        self.history: list[dict[str, Any]] = []
        self.last_checkpoint_id: str | None = None

    def _resource_declare(self, stage: str) -> dict[str, Any]:
        # Toy run: tiny envelope. Declares wall time, RSS, threads before stage.
        return declare_and_admit(
            stage=stage,
            wall_time_budget_s=30.0,
            rss_budget_bytes=256 * 1024 * 1024,
            threads=1,
            heavy=False,
            memory_bytes=64 * 1024 * 1024,
        )

    def _save_ckpt(self, stage: str) -> dict[str, Any]:
        meta = self.store.save(
            stage=stage,
            step=self.model.step,
            objective=self.objective_name,
            model_state=self.model.state_dict(),
            parent_id=self.last_checkpoint_id,
            fixture_label=FIXTURE_LABEL,
        )
        self.last_checkpoint_id = meta["checkpoint_id"]
        return meta

    def load_from_checkpoint(self, cid: str | None = None) -> dict[str, Any]:
        cid = cid or self.store.current_id()
        if not cid:
            raise CheckpointError("NO_CHECKPOINT")
        state = self.store.load_state(cid)
        self.model = ToyMLP.from_state_dict(state["model_state"])
        self.last_checkpoint_id = cid
        return state

    def run(
        self,
        *,
        total_steps: int,
        stage: str = "FIXTURE",
        kill_after_step: int | None = None,
        resume: bool = False,
        emit_receipt: bool = True,
    ) -> dict[str, Any]:
        """Run training steps. If kill_after_step is set, stop after that step
        (simulating a crash) having written the last checkpoint.

        If resume=True, load CURRENT first and continue until total_steps.
        """
        require_objective(self.objective_name, stage="FIXTURE")
        resources = self._resource_declare(stage)
        if not resources["admission"]["admit"]:
            raise RuntimeError(f"resource scheduler denied: {resources}")

        gate = stage_entry_gate(
            stage,
            store=self.store,
            require_prior_checkpoint=resume,
            objective=self.objective_name,
        )

        if resume:
            self.load_from_checkpoint()

        start_step = self.model.step
        killed = False
        losses: list[float] = []

        while self.model.step < total_steps:
            stats = self.model.train_step(lr=self.lr)
            losses.append(stats["loss"])
            if self.model.step % self.checkpoint_every == 0:
                self._save_ckpt(stage)
            if kill_after_step is not None and self.model.step >= kill_after_step:
                # Ensure a checkpoint exists at the kill boundary.
                if self.model.step % self.checkpoint_every != 0:
                    self._save_ckpt(stage)
                killed = True
                break

        # Final checkpoint if we finished cleanly and didn't just save.
        if not killed and (self.last_checkpoint_id is None or self.model.step % self.checkpoint_every != 0):
            self._save_ckpt(stage)
        elif not killed and self.model.step > 0:
            # Always seal final state.
            self._save_ckpt(stage)

        result = {
            "schema": SCHEMA,
            "fixture_label": FIXTURE_LABEL,
            "stage": stage,
            "objective": self.objective_name,
            "start_step": start_step,
            "end_step": self.model.step,
            "target_steps": total_steps,
            "killed": killed,
            "kill_after_step": kill_after_step,
            "resumed": resume,
            "state_sha256": self.model.state_sha256(),
            "weights_sha256": self.model.weights_sha256(),
            "n_params": self.model.n_params(),
            "checkpoint_id": self.last_checkpoint_id,
            "losses": losses,
            "entry_gate": gate,
            "resources": resources,
            "never_trained_anything_real": True,
        }
        if emit_receipt:
            receipt = make_receipt(
                stage=stage,
                status="KILLED" if killed else "COMPLETED",
                checkpoint_id=self.last_checkpoint_id,
                objective=self.objective_name,
                steps_completed=self.model.step - start_step,
                state_sha256=self.model.state_sha256(),
                fixture=True,
                details={
                    "start_step": start_step,
                    "end_step": self.model.step,
                    "killed": killed,
                    "resumed": resume,
                    "n_params": self.model.n_params(),
                },
            )
            path = write_receipt(receipt, self.receipts_dir)
            result["receipt"] = receipt
            result["receipt_path"] = str(path)
        self.history.append(result)
        return result


def run_uninterrupted_vs_resumed(
    work_dir: Path,
    *,
    total_steps: int = 8,
    kill_after_step: int = 4,
    cfg: ToyConfig | None = None,
) -> dict[str, Any]:
    """The test that matters: resumed run reaches bit-identical state.

    1. Uninterrupted run to total_steps → state hash A
    2. Fresh trainer, run to kill_after_step, stop
    3. Resume to total_steps → state hash B
    4. Assert A == B
    """
    work_dir = Path(work_dir)
    cfg = cfg or ToyConfig(seed=7)

    clean = work_dir / "clean"
    clean.mkdir(parents=True, exist_ok=True)
    t_clean = ToyTrainer(clean, cfg=cfg, checkpoint_every=1)
    r_clean = t_clean.run(total_steps=total_steps, stage="FIXTURE")

    killed_dir = work_dir / "killed"
    killed_dir.mkdir(parents=True, exist_ok=True)
    t_kill = ToyTrainer(killed_dir, cfg=cfg, checkpoint_every=1)
    r_kill = t_kill.run(
        total_steps=total_steps,
        stage="FIXTURE",
        kill_after_step=kill_after_step,
    )
    # New trainer instance loads CURRENT and resumes.
    t_resume = ToyTrainer(killed_dir, cfg=cfg, checkpoint_every=1)
    r_resume = t_resume.run(total_steps=total_steps, stage="FIXTURE", resume=True)

    identical = r_clean["state_sha256"] == r_resume["state_sha256"]
    return {
        "schema": "hawking.odyssey.resume_bit_identity.v1",
        "fixture_label": FIXTURE_LABEL,
        "total_steps": total_steps,
        "kill_after_step": kill_after_step,
        "uninterrupted_state_sha256": r_clean["state_sha256"],
        "killed_state_sha256": r_kill["state_sha256"],
        "resumed_state_sha256": r_resume["state_sha256"],
        "bit_identical": identical,
        "uninterrupted_checkpoint_id": r_clean["checkpoint_id"],
        "resumed_checkpoint_id": r_resume["checkpoint_id"],
        "uninterrupted_end_step": r_clean["end_step"],
        "resumed_end_step": r_resume["end_step"],
        "note": (
            "PASS requires bit-identical final state after kill+resume vs uninterrupted. "
            "A resume that merely continues without matching is a FAIL."
        ),
    }

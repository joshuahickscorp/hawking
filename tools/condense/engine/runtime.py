#!/usr/bin/env python3.12
"""Campaign runtime — run / resume / report from an ExperimentSpec.

This is the single controller. Historical campaigns supply declarative specs
and optional handler hooks; they do not supply a new state machine.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .checkpoint import CheckpointStore
from .governor import ResourceGovernor, ResourceLimits
from .lease import LeaseError, SingletonLease
from .receipts import Receipt, ReceiptStore
from .scheduler import Scheduler, WorkStatus
from .spec import CampaignPhase, ExperimentSpec, load_spec
from .state_machine import IllegalTransition, Phase, StateMachine

Handler = Callable[["CampaignRuntime", Mapping[str, Any]], dict[str, Any]]


@dataclass
class RunResult:
    campaign_id: str
    phase: str
    status: str
    completed_steps: list[str]
    receipt_path: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "phase": self.phase,
            "status": self.status,
            "completed_steps": list(self.completed_steps),
            "receipt_path": self.receipt_path,
            "detail": dict(self.detail),
        }


# Built-in handlers: pure bookkeeping / record steps. Campaign-specific heavy
# work registers under explicit keys; missing handlers on optional steps skip.
def _handle_record(runtime: "CampaignRuntime", params: Mapping[str, Any]) -> dict[str, Any]:
    return {"recorded": True, **dict(params)}


def _handle_precheck_fences(
    runtime: "CampaignRuntime", params: Mapping[str, Any]
) -> dict[str, Any]:
    """Ensure authorization fences named in the spec stay false / absent."""
    closed: list[str] = []
    for fence in runtime.spec.authorization_fences:
        # Fences are closed unless an explicit truthy override is in params.
        overrides = params.get("allow") or {}
        if overrides.get(fence):
            raise RuntimeError(f"authorization fence forced open: {fence}")
        closed.append(fence)
    return {"fences_closed": closed}


BUILTIN_HANDLERS: dict[str, Handler] = {
    "record": _handle_record,
    "precheck.fences": _handle_precheck_fences,
    "report.summary": _handle_record,
    "seal.receipt": _handle_record,
}


class CampaignRuntime:
    """One runtime instance bound to a working directory and a spec."""

    def __init__(
        self,
        spec: ExperimentSpec,
        *,
        work_dir: Path,
        handlers: Mapping[str, Handler] | None = None,
        controller_epoch: str = "1",
        acquire_lease: bool = True,
    ) -> None:
        self.spec = spec
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.handlers: dict[str, Handler] = {**BUILTIN_HANDLERS, **dict(handlers or {})}
        self.controller_epoch = controller_epoch
        self.acquire_lease = acquire_lease

        limits = ResourceLimits.from_mapping(spec.metadata.get("resource_limits"))
        self.governor = ResourceGovernor(limits, root=self.work_dir)
        self.checkpoints = CheckpointStore(
            self.work_dir,
            campaign_id=spec.campaign_id,
            checkpoint_name=spec.checkpoint_name or "checkpoint.json",
        )
        self.receipts = ReceiptStore(self.work_dir / "receipts")
        self.lease = SingletonLease(
            self.work_dir / (spec.lease_name or f"{spec.campaign_id}.lease"),
            campaign_id=spec.campaign_id,
            controller_epoch=controller_epoch,
            owner=f"engine:{spec.campaign_id}",
        )
        self.machine = StateMachine.for_spec(spec)
        self.scheduler = Scheduler(spec)
        self._lease_held = False

    def _sync_scheduler_from_machine(self) -> None:
        self.scheduler = Scheduler(self.spec, completed=set(self.machine.completed_steps))

    def open(self) -> None:
        if self.acquire_lease:
            self.lease.acquire()
            self._lease_held = True
        resumed = self.checkpoints.resume_state()
        if resumed.get("phase") and resumed.get("phase") != "idle":
            self.machine = StateMachine.from_snapshot(resumed)
            if self.machine.phase not in {Phase.COMPLETE, Phase.IDLE}:
                # Mark resume entry without requiring a prior claim on cold start.
                if self.machine.phase is not Phase.RESUME:
                    try:
                        self.machine.transition(
                            Phase.RESUME,
                            claim_id=f"resume:{self.machine.phase.value}:{len(self.machine.claims)}",
                            detail={"from_checkpoint": True},
                        )
                    except IllegalTransition:
                        pass
        self._sync_scheduler_from_machine()
        self.checkpoints.record("open", self.machine.snapshot())

    def close(self) -> None:
        try:
            self.checkpoints.save(self.machine.snapshot())
        finally:
            if self._lease_held:
                self.lease.release()
                self._lease_held = False

    def __enter__(self) -> "CampaignRuntime":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def run_handler(self, step_id: str) -> dict[str, Any]:
        step = next((s for s in self.spec.steps if s.id == step_id), None)
        if step is None:
            raise KeyError(f"unknown step {step_id!r}")
        if self.machine.is_step_done(step_id) and step.idempotent:
            return {"skipped": True, "reason": "already_done"}
        name = step.handler or "record"
        handler = self.handlers.get(name)
        if handler is None:
            if step.optional:
                self.scheduler.mark_skipped(step_id, f"no handler {name}")
                self.machine.mark_step(step_id)
                return {"skipped": True, "reason": f"no handler {name}"}
            raise RuntimeError(f"no handler registered for {name!r} (step {step_id})")
        if self.acquire_lease:
            self.lease.assert_held()
        # Resource gate before non-light record steps.
        if name not in {"record", "precheck.fences", "report.summary"}:
            self.governor.require()
        result = handler(self, step.params)
        self.machine.mark_step(step_id)
        self.scheduler.mark_done(step_id, detail=result)
        self.checkpoints.record(
            "step_done",
            {"step_id": step_id, "handler": name, "result": result},
        )
        self.checkpoints.save(self.machine.snapshot())
        return result

    def advance_phase(self, target: str | Phase | CampaignPhase, *, claim_id: str) -> None:
        if self.acquire_lease:
            self.lease.assert_held()
        self.machine.transition(target, claim_id=claim_id)
        self.checkpoints.record(
            "phase_transition",
            {"to": self.machine.phase.value, "claim_id": claim_id},
        )
        self.checkpoints.save(self.machine.snapshot())

    def run(
        self,
        *,
        phases: list[str] | None = None,
        stop_on_fail: bool = True,
    ) -> RunResult:
        """Execute pending steps, optionally restricted to *phases*."""
        wanted = set(phases or self.spec.phases)
        # Enter first phase from idle/resume if needed.
        if self.machine.phase in {Phase.IDLE, Phase.RESUME}:
            first = next((p for p in self.spec.phases if p in wanted), None)
            if first is not None:
                try:
                    self.advance_phase(first, claim_id=f"enter:{first}")
                except IllegalTransition:
                    pass

        errors: list[str] = []
        for phase in self.spec.phases:
            if phase not in wanted:
                continue
            if self.machine.phase.value != phase:
                try:
                    self.advance_phase(phase, claim_id=f"phase:{phase}:{len(self.machine.claims)}")
                except IllegalTransition:
                    # Stay put if already past this phase.
                    if self.machine.phase is Phase.COMPLETE:
                        break
            for step in self.spec.steps_for(phase):
                if self.machine.is_step_done(step.id) and step.idempotent:
                    continue
                try:
                    self.run_handler(step.id)
                except Exception as exc:  # noqa: BLE001 - surface into FAULT
                    errors.append(f"{step.id}: {exc}")
                    self.scheduler.mark_failed(step.id, str(exc))
                    try:
                        self.advance_phase(
                            Phase.FAULT,
                            claim_id=f"fault:{step.id}:{len(self.machine.claims)}",
                        )
                    except IllegalTransition:
                        self.machine.fault_reason = str(exc)
                    if stop_on_fail:
                        return RunResult(
                            campaign_id=self.spec.campaign_id,
                            phase=self.machine.phase.value,
                            status="FAULT",
                            completed_steps=list(self.machine.completed_steps),
                            detail={"errors": errors},
                        )

        # Seal / report bookkeeping when all steps done.
        all_done = all(
            self.machine.is_step_done(s.id) or s.optional for s in self.spec.steps
        )
        receipt_path = None
        if all_done and not errors:
            if self.machine.phase is not Phase.COMPLETE:
                try:
                    if self.machine.phase is not Phase.REPORT:
                        if "report" in self.spec.phases and self.machine.can(Phase.REPORT):
                            self.advance_phase(
                                Phase.REPORT,
                                claim_id=f"report:{len(self.machine.claims)}",
                            )
                    if self.machine.can(Phase.COMPLETE):
                        self.advance_phase(
                            Phase.COMPLETE,
                            claim_id=f"complete:{len(self.machine.claims)}",
                        )
                except IllegalTransition:
                    pass
            receipt = Receipt(
                campaign_id=self.spec.campaign_id,
                status=self.spec.status,
                phase=self.machine.phase.value,
                summary={
                    "completed_steps": list(self.machine.completed_steps),
                    "title": self.spec.title,
                    "family": self.spec.family,
                },
                reproduction=self.spec.reproduction,
                artifacts=tuple(
                    a for a in (self.spec.receipt, self.spec.fixture) if a
                ),
            )
            path = self.receipts.write(receipt)
            receipt_path = str(path)
            self.checkpoints.record("receipt_written", {"path": receipt_path})
            self.checkpoints.save(self.machine.snapshot())

        status = "PASS" if not errors and all_done else (
            "FAULT" if errors else "PARTIAL"
        )
        return RunResult(
            campaign_id=self.spec.campaign_id,
            phase=self.machine.phase.value,
            status=status,
            completed_steps=list(self.machine.completed_steps),
            receipt_path=receipt_path,
            detail={"errors": errors} if errors else {},
        )

    def status(self) -> dict[str, Any]:
        return {
            "campaign_id": self.spec.campaign_id,
            "title": self.spec.title,
            "family": self.spec.family,
            "spec_status": self.spec.status,
            "phase": self.machine.phase.value,
            "completed_steps": list(self.machine.completed_steps),
            "fault_reason": self.machine.fault_reason,
            "scheduler": self.scheduler.snapshot(),
            "lease_held": self.lease.held,
            "reproduction": self.spec.reproduction,
            "receipt": self.spec.receipt,
            "reopen": [r.to_dict() for r in self.spec.reopen],
        }


def run_campaign(
    spec: ExperimentSpec | Mapping[str, Any] | str | Path,
    *,
    work_dir: Path,
    handlers: Mapping[str, Handler] | None = None,
    phases: list[str] | None = None,
    acquire_lease: bool = True,
) -> RunResult:
    """Convenience: load spec, run under lease, return result."""
    if not isinstance(spec, ExperimentSpec):
        spec = load_spec(spec)
    with CampaignRuntime(
        spec,
        work_dir=work_dir,
        handlers=handlers,
        acquire_lease=acquire_lease,
    ) as runtime:
        return runtime.run(phases=phases)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run a condense campaign from a spec")
    parser.add_argument("spec", type=Path, help="Path to ExperimentSpec JSON")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory for lease/checkpoint/receipts",
    )
    parser.add_argument("--status", action="store_true", help="Print status only")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument(
        "--no-lease",
        action="store_true",
        help="Skip exclusive lease (fixture / dry paths only)",
    )
    args = parser.parse_args(argv)
    spec = load_spec(args.spec)
    work_dir = args.work_dir or Path("reports/condense/engine") / spec.campaign_id
    with CampaignRuntime(
        spec, work_dir=work_dir, acquire_lease=not args.no_lease
    ) as runtime:
        if args.status:
            payload = runtime.status()
        else:
            payload = runtime.run().to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if isinstance(payload, dict) and payload.get("status") == "FAULT":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

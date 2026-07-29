#!/usr/bin/env python3.12
"""One scheduler for campaign work items.

Plans steps from an ExperimentSpec, skips completed idempotent work (resume),
and yields the next runnable unit. Does not execute handlers — that is runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Mapping

from .spec import ExperimentSpec, StepSpec


class WorkStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class WorkItem:
    step: StepSpec
    status: WorkStatus = WorkStatus.PENDING
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.step.id

    @property
    def phase(self) -> str:
        return self.step.phase


class Scheduler:
    """Deterministic, resume-safe work planner."""

    def __init__(
        self,
        spec: ExperimentSpec,
        *,
        completed: set[str] | None = None,
    ) -> None:
        self.spec = spec
        self.completed: set[str] = set(completed or ())
        self.items: list[WorkItem] = [WorkItem(step=s) for s in spec.steps]
        for item in self.items:
            if item.id in self.completed and item.step.idempotent:
                item.status = WorkStatus.DONE

    def pending(self) -> list[WorkItem]:
        return [i for i in self.items if i.status in {WorkStatus.PENDING, WorkStatus.READY}]

    def done(self) -> list[WorkItem]:
        return [i for i in self.items if i.status is WorkStatus.DONE]

    def mark_done(self, step_id: str, *, detail: Mapping[str, Any] | None = None) -> None:
        for item in self.items:
            if item.id == step_id:
                item.status = WorkStatus.DONE
                if detail:
                    item.detail.update(detail)
                self.completed.add(step_id)
                return
        raise KeyError(f"unknown step {step_id!r}")

    def mark_failed(self, step_id: str, reason: str) -> None:
        for item in self.items:
            if item.id == step_id:
                item.status = WorkStatus.FAILED
                item.detail["reason"] = reason
                return
        raise KeyError(f"unknown step {step_id!r}")

    def mark_skipped(self, step_id: str, reason: str) -> None:
        for item in self.items:
            if item.id == step_id:
                item.status = WorkStatus.SKIPPED
                item.detail["reason"] = reason
                return
        raise KeyError(f"unknown step {step_id!r}")

    def next_ready(self, *, phase: str | None = None) -> WorkItem | None:
        """Return the next runnable item whose inputs are satisfied."""
        done_ids = {i.id for i in self.items if i.status is WorkStatus.DONE}
        done_ids |= self.completed
        for item in self.items:
            if item.status not in {WorkStatus.PENDING, WorkStatus.READY}:
                continue
            if phase is not None and item.phase != phase:
                continue
            if all(inp in done_ids or inp in self.completed for inp in item.step.inputs):
                # Also treat pure artifact inputs (not step ids) as external —
                # only enforce inputs that match known step ids.
                needed = [inp for inp in item.step.inputs if any(i.id == inp for i in self.items)]
                if all(inp in done_ids for inp in needed):
                    item.status = WorkStatus.READY
                    return item
        return None

    def plan(self, *, phase: str | None = None) -> list[WorkItem]:
        """Materialize a ready-order plan without mutating running state."""
        planned: list[WorkItem] = []
        completed = set(self.completed)
        for item in self.items:
            if item.status is WorkStatus.DONE:
                completed.add(item.id)
                continue
            if phase is not None and item.phase != phase:
                continue
            needed = [inp for inp in item.step.inputs if any(i.id == inp for i in self.items)]
            if all(inp in completed for inp in needed):
                planned.append(item)
                if item.step.idempotent:
                    completed.add(item.id)
        return planned

    def walk(self) -> Iterator[WorkItem]:
        """Yield ready items until the plan is exhausted or blocked."""
        while True:
            item = self.next_ready()
            if item is None:
                return
            yield item
            if item.status is WorkStatus.READY:
                # Caller must mark_done / mark_failed; if left READY, stop to avoid loop.
                return

    def snapshot(self) -> dict[str, Any]:
        return {
            "campaign_id": self.spec.campaign_id,
            "completed": sorted(self.completed),
            "items": [
                {
                    "id": i.id,
                    "phase": i.phase,
                    "status": i.status.value,
                    "detail": dict(i.detail),
                }
                for i in self.items
            ],
        }

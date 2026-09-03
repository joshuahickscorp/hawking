from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from lab.layout import resolve_workspace_path
from lab.spec import CampaignPhase, ExperimentSpec, StepSpec

@dataclass(frozen=True)
class ResourceLimits:
    min_free_disk_bytes: int = 0
    min_free_inodes: int = 0
    max_concurrent_workers: int = 1
    require_path: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> 'ResourceLimits':
        raw = raw or {}
        return cls(
            min_free_disk_bytes=int(raw.get('min_free_disk_bytes') or 0),
            min_free_inodes=int(raw.get('min_free_inodes') or 0),
            max_concurrent_workers=int(raw.get('max_concurrent_workers') or 1),
            require_path=raw.get('require_path'),
        )

@dataclass(frozen=True)
class _ResourceSample:
    path: str
    free_disk_bytes: int
    total_disk_bytes: int
    free_inodes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'path': self.path,
            'free_disk_bytes': self.free_disk_bytes,
            'total_disk_bytes': self.total_disk_bytes,
            'free_inodes': self.free_inodes,
        }

class ResourceGovernor:

    def __init__(self, limits: ResourceLimits, *, root: Path | None=None) -> None:
        self.limits = limits
        self.root = Path(root or limits.require_path or '.')

    def sample(self, path: Path | None=None) -> _ResourceSample:
        target = Path(path or self.root)
        target.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(target)
        free_inodes: int | None = None
        try:
            st = os.statvfs(target)
            free_inodes = int(st.f_favail)
        except (AttributeError, OSError):
            free_inodes = None
        return _ResourceSample(
            path=str(target),
            free_disk_bytes=int(usage.free),
            total_disk_bytes=int(usage.total),
            free_inodes=free_inodes,
        )

    def evaluate(self, sample: _ResourceSample | None=None) -> list[str]:
        sample = sample or self.sample()
        failures: list[str] = []
        if self.limits.min_free_disk_bytes > 0 and sample.free_disk_bytes < self.limits.min_free_disk_bytes:
            failures.append(f'free_disk_bytes {sample.free_disk_bytes} < min {self.limits.min_free_disk_bytes}')
        if (
            self.limits.min_free_inodes > 0
            and sample.free_inodes is not None
            and sample.free_inodes < self.limits.min_free_inodes
        ):
            failures.append(f'free_inodes {sample.free_inodes} < min {self.limits.min_free_inodes}')
        if self.limits.require_path:
            req = Path(self.limits.require_path)
            if not req.is_absolute():
                workspace_req = resolve_workspace_path(req)
                if workspace_req.exists():
                    req = workspace_req
            if not req.exists():
                failures.append(f'require_path missing: {req}')
        return failures

    def allow(self) -> tuple[bool, _ResourceSample, list[str]]:
        sample = self.sample()
        failures = self.evaluate(sample)
        return (not failures, sample, failures)

    def require(self) -> _ResourceSample:
        ok, sample, failures = self.allow()
        if not ok:
            raise RuntimeError('resource governor refused: ' + '; '.join(failures))
        return sample

class WorkStatus(str, Enum):
    PENDING = 'pending'
    READY = 'ready'
    RUNNING = 'running'
    DONE = 'done'
    SKIPPED = 'skipped'
    FAILED = 'failed'

@dataclass
class _WorkItem:
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

    def __init__(self, spec: ExperimentSpec, *, completed: set[str] | None=None) -> None:
        self.spec = spec
        self.completed: set[str] = set(completed or ())
        self.items: list[_WorkItem] = [_WorkItem(step=s) for s in spec.steps]
        for item in self.items:
            if item.id in self.completed and item.step.idempotent:
                item.status = WorkStatus.DONE

    def _find(self, step_id: str) -> _WorkItem:
        for item in self.items:
            if item.id == step_id:
                return item
        raise KeyError(f'unknown step {step_id!r}')

    def mark_done(self, step_id: str, *, detail: Mapping[str, Any] | None=None) -> None:
        item = self._find(step_id)
        item.status = WorkStatus.DONE
        if detail:
            item.detail.update(detail)
        self.completed.add(step_id)

    def mark_failed(self, step_id: str, reason: str) -> None:
        item = self._find(step_id)
        item.status = WorkStatus.FAILED
        item.detail['reason'] = reason

    def mark_skipped(self, step_id: str, reason: str) -> None:
        item = self._find(step_id)
        item.status = WorkStatus.SKIPPED
        item.detail['reason'] = reason

    def next_ready(self, *, phase: str | None=None) -> _WorkItem | None:
        done_ids = {i.id for i in self.items if i.status is WorkStatus.DONE} | self.completed
        known = {i.id for i in self.items}
        for item in self.items:
            if item.status not in {WorkStatus.PENDING, WorkStatus.READY}:
                continue
            if phase is not None and item.phase != phase:
                continue
            needed = [inp for inp in item.step.inputs if inp in known]
            if all((inp in done_ids for inp in needed)):
                item.status = WorkStatus.READY
                return item
        return None

    def plan(self, *, phase: str | None=None) -> list[_WorkItem]:
        planned: list[_WorkItem] = []
        completed = set(self.completed)
        known = {i.id for i in self.items}
        for item in self.items:
            if item.status is WorkStatus.DONE:
                completed.add(item.id)
                continue
            if phase is not None and item.phase != phase:
                continue
            needed = [inp for inp in item.step.inputs if inp in known]
            if all((inp in completed for inp in needed)):
                planned.append(item)
                if item.step.idempotent:
                    completed.add(item.id)
        return planned

    def walk(self) -> Iterator[_WorkItem]:
        while True:
            item = self.next_ready()
            if item is None:
                return
            yield item
            if item.status is WorkStatus.READY:
                return

    def snapshot(self) -> dict[str, Any]:
        return {
            'campaign_id': self.spec.campaign_id,
            'completed': sorted(self.completed),
            'items': [
                {
                    'id': i.id,
                    'phase': i.phase,
                    'status': i.status.value,
                    'detail': dict(i.detail),
                }
                for i in self.items
            ],
        }

class IllegalTransition(RuntimeError):
    pass

class Phase(str, Enum):
    IDLE = 'idle'
    PRECHECK = 'precheck'
    MEASURE = 'measure'
    ALLOCATE = 'allocate'
    PACK = 'pack'
    VERIFY = 'verify'
    SEAL = 'seal'
    PROMOTE = 'promote'
    BURY = 'bury'
    MONITOR = 'monitor'
    RESUME = 'resume'
    REPORT = 'report'
    COMPLETE = 'complete'
    FAULT = 'fault'
_FORWARD: dict[Phase, frozenset[Phase]] = {
    Phase.IDLE: frozenset({Phase.PRECHECK, Phase.RESUME, Phase.FAULT}),
    Phase.PRECHECK: frozenset({Phase.MEASURE, Phase.FAULT, Phase.COMPLETE}),
    Phase.MEASURE: frozenset({
        Phase.ALLOCATE, Phase.PACK, Phase.VERIFY, Phase.SEAL, Phase.FAULT,
    }),
    Phase.ALLOCATE: frozenset({Phase.PACK, Phase.VERIFY, Phase.SEAL, Phase.FAULT}),
    Phase.PACK: frozenset({Phase.VERIFY, Phase.SEAL, Phase.MEASURE, Phase.FAULT}),
    Phase.VERIFY: frozenset({Phase.SEAL, Phase.PROMOTE, Phase.BURY, Phase.FAULT}),
    Phase.SEAL: frozenset({
        Phase.PROMOTE, Phase.BURY, Phase.MONITOR, Phase.REPORT, Phase.COMPLETE, Phase.FAULT,
    }),
    Phase.PROMOTE: frozenset({Phase.MONITOR, Phase.REPORT, Phase.COMPLETE, Phase.FAULT}),
    Phase.BURY: frozenset({Phase.REPORT, Phase.COMPLETE, Phase.FAULT}),
    Phase.MONITOR: frozenset({Phase.REPORT, Phase.COMPLETE, Phase.RESUME, Phase.FAULT}),
    Phase.RESUME: frozenset({
        Phase.PRECHECK, Phase.MEASURE, Phase.ALLOCATE, Phase.PACK, Phase.VERIFY,
        Phase.SEAL, Phase.PROMOTE, Phase.BURY, Phase.MONITOR, Phase.REPORT, Phase.FAULT,
    }),
    Phase.REPORT: frozenset({Phase.COMPLETE, Phase.FAULT}),
    Phase.COMPLETE: frozenset(),
    Phase.FAULT: frozenset({Phase.RESUME, Phase.IDLE}),
}

def _phase(value: str | Phase | CampaignPhase) -> Phase:
    if isinstance(value, Phase):
        return value
    name = value.value if isinstance(value, CampaignPhase) else str(value)
    try:
        return Phase(name)
    except ValueError as exc:
        raise IllegalTransition(f'unknown phase {name!r}') from exc

@dataclass(frozen=True)
class _Transition:
    source: Phase
    target: Phase
    claim_id: str
    detail: Mapping[str, Any] = field(default_factory=dict)

@dataclass
class StateMachine:
    campaign_id: str
    phase: Phase = Phase.IDLE
    completed_steps: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    fault_reason: str | None = None
    allowed: dict[Phase, frozenset[Phase]] = field(
        default_factory=lambda: {k: frozenset(v) for k, v in _FORWARD.items()}
    )

    @classmethod
    def for_spec(cls, spec: ExperimentSpec) -> 'StateMachine':
        phases = [_phase(p) for p in spec.phases]
        allowed: dict[Phase, frozenset[Phase]] = {k: frozenset(v) for k, v in _FORWARD.items()}
        if phases:
            first = phases[0]
            allowed[Phase.IDLE] = frozenset({first, Phase.RESUME, Phase.FAULT})
            for left, right in zip(phases, phases[1:]):
                extra = {right, Phase.FAULT, Phase.RESUME}
                if left in {Phase.SEAL, Phase.PROMOTE, Phase.BURY}:
                    extra.add(Phase.COMPLETE)
                allowed[left] = frozenset(extra)
            last = phases[-1]
            allowed[last] = frozenset((allowed.get(last) or frozenset()) | {Phase.COMPLETE, Phase.FAULT})
        return cls(campaign_id=spec.campaign_id, allowed=allowed)

    def can(self, target: str | Phase | CampaignPhase) -> bool:
        return _phase(target) in self.allowed.get(self.phase, frozenset())

    def transition(
        self,
        target: str | Phase | CampaignPhase,
        *,
        claim_id: str,
        detail: Mapping[str, Any] | None=None,
    ) -> _Transition:
        if not claim_id or not isinstance(claim_id, str):
            raise IllegalTransition('claim_id must be a non-empty string')
        if claim_id in self.claims:
            raise IllegalTransition(f'claim {claim_id!r} already consumed (one-use)')
        dest = _phase(target)
        if dest not in self.allowed.get(self.phase, frozenset()):
            raise IllegalTransition(
                f'illegal transition {self.phase.value!r} -> {dest.value!r} '
                f'for campaign {self.campaign_id!r}'
            )
        source = self.phase
        self.phase = dest
        self.claims.append(claim_id)
        if dest is Phase.FAULT:
            self.fault_reason = str((detail or {}).get('reason') or 'fault')
        elif source is Phase.FAULT:
            self.fault_reason = None
        return _Transition(source=source, target=dest, claim_id=claim_id, detail=dict(detail or {}))

    def mark_step(self, step_id: str) -> None:
        if step_id not in self.completed_steps:
            self.completed_steps.append(step_id)

    def is_step_done(self, step_id: str) -> bool:
        return step_id in self.completed_steps

    def snapshot(self) -> dict[str, Any]:
        return {
            'campaign_id': self.campaign_id,
            'phase': self.phase.value,
            'completed_steps': list(self.completed_steps),
            'claims': list(self.claims),
            'fault_reason': self.fault_reason,
        }

    @classmethod
    def from_snapshot(cls, raw: Mapping[str, Any]) -> 'StateMachine':
        sm = cls(campaign_id=str(raw['campaign_id']))
        sm.phase = _phase(str(raw.get('phase') or 'idle'))
        sm.completed_steps = list(raw.get('completed_steps') or [])
        sm.claims = list(raw.get('claims') or [])
        sm.fault_reason = raw.get('fault_reason')
        return sm

    def restrict_to(self, phases: Iterable[str | Phase | CampaignPhase]) -> None:
        allowed_set = {_phase(p) for p in phases}
        for source, targets in list(self.allowed.items()):
            self.allowed[source] = frozenset(
                t
                for t in targets
                if t in allowed_set
                or t in {Phase.FAULT, Phase.COMPLETE, Phase.RESUME}
            )

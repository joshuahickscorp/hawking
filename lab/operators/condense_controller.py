"""Offline, byte-accounted controller for bounded Condense/Gravity rotation.

This is deliberately a controller, not an executor: it opens no files, starts no
subprocesses, and has no network client. A caller performs fetch, fit, packing,
sealing, and eviction, then reports each completed bounded operation here. The
controller refuses transitions that could hold two source windows, two heavy
operations, or more than the declared incremental byte budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from threading import RLock
from typing import Any, Mapping


SCHEMA = "hawking.condense.rotation_controller.v1"
_SEAL_HEX = frozenset("0123456789abcdef")

# A controller instance has its own transition lock, but the heavy GPU is a
# process-wide resource.  This guard is deliberately retained until the
# caller has both sealed and evicted N.  If a caller abandons an active
# controller, later work fails closed instead of assuming the GPU is idle.
_HEAVY_OPERATION_LOCK = RLock()
_HEAVY_OPERATION_OWNER: tuple[int, str] | None = None


class ControllerError(RuntimeError):
    """The proposed lifecycle transition is not safely admissible."""


@dataclass(frozen=True)
class CondenseTask:
    """One source window and every artifact emitted from its single pack pass."""

    task_id: str
    source_bytes: int
    metadata_bytes: int
    artifact_bytes: Mapping[str, int]
    scratch_bytes: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id or self.task_id != self.task_id.strip():
            raise ControllerError("task_id must be a non-empty, trimmed string")
        for label, value in (("source_bytes", self.source_bytes), ("metadata_bytes", self.metadata_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ControllerError(f"{label} must be a non-negative integer")
        if isinstance(self.scratch_bytes, bool) or not isinstance(self.scratch_bytes, int) or self.scratch_bytes < 0:
            raise ControllerError("scratch_bytes must be a non-negative integer")
        if self.source_bytes == 0:
            raise ControllerError("source_bytes must be positive")
        artifacts = dict(self.artifact_bytes)
        if not artifacts:
            raise ControllerError("artifact_bytes must declare the one-pass artifact family")
        for artifact_id, byte_count in artifacts.items():
            if not isinstance(artifact_id, str) or not artifact_id or artifact_id != artifact_id.strip():
                raise ControllerError("artifact ids must be non-empty, trimmed strings")
            if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
                raise ControllerError("artifact byte counts must be positive integers")
        object.__setattr__(self, "artifact_bytes", artifacts)


@dataclass
class _TaskState:
    task: CondenseTask
    phase: str = "PENDING"
    metadata_owner: str | None = None
    metadata_resident: bool = False
    source_resident: bool = False
    reserved_artifact_bytes: int = 0
    packed_artifacts: dict[str, int] | None = None
    seal_sha256: str | None = None

    @property
    def resident_bytes(self) -> int:
        metadata = self.task.metadata_bytes if self.metadata_resident else 0
        source = self.task.source_bytes if self.source_resident else 0
        scratch = self.task.scratch_bytes if self.source_resident else 0
        return metadata + source + scratch + self.reserved_artifact_bytes


@dataclass
class _SharedStats:
    samples: int = 0
    tasks: set[str] = field(default_factory=set)
    sums: dict[str, float] = field(default_factory=dict)

    def add(self, task_id: str, metrics: Mapping[str, float]) -> None:
        self.samples += 1
        self.tasks.add(task_id)
        for name, value in metrics.items():
            self.sums[name] = self.sums.get(name, 0.0) + value

    def snapshot(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "tasks": sorted(self.tasks),
            "sums": dict(sorted(self.sums.items())),
            "means": {name: value / self.samples for name, value in sorted(self.sums.items())},
        }


class CondenseController:
    """Plan the ``N-1 evict -> N process -> N+1 metadata`` lifecycle.

    The controller has exactly one byte budget and one expected heavy-lease
    token. It never obtains that lease itself; an executor proves possession by
    supplying the expected token to each heavy operation.  A process-wide,
    fail-closed guard also prevents two controller instances from claiming the
    same heavy operation at once.  The executor remains responsible for
    obtaining the cross-process GPU lease before it presents that token.
    """

    def __init__(self, tasks: list[CondenseTask], *, byte_budget_bytes: int, heavy_lease_token: str) -> None:
        if isinstance(byte_budget_bytes, bool) or not isinstance(byte_budget_bytes, int) or byte_budget_bytes <= 0:
            raise ControllerError("byte_budget_bytes must be a positive integer")
        if not isinstance(heavy_lease_token, str) or not heavy_lease_token or heavy_lease_token != heavy_lease_token.strip():
            raise ControllerError("heavy_lease_token must be a non-empty, trimmed string")
        if not tasks:
            raise ControllerError("at least one CondenseTask is required")
        task_ids = [task.task_id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ControllerError("task ids must be unique")
        self._states = [_TaskState(task=task) for task in tasks]
        self._by_id = {state.task.task_id: state for state in self._states}
        self._budget = byte_budget_bytes
        self._heavy_lease_token = heavy_lease_token
        self._active_task_id: str | None = None
        self._stats: dict[tuple[str, str], _SharedStats] = {}
        self._events: list[dict[str, Any]] = []
        self._lock = RLock()

    def claim_metadata(self, worker_id: str) -> str | None:
        """Claim the one admissible metadata/range job (bootstrap or N+1)."""
        with self._lock:
            worker = self._worker(worker_id)
            state = self._metadata_candidate()
            if state is None or state.phase != "PENDING":
                return None
            state.phase = "METADATA_CLAIMED"
            state.metadata_owner = worker
            self._record("metadata_claimed", state, worker=worker, job_kind=self._metadata_kind(state))
            return state.task.task_id

    def steal_metadata(self, task_id: str, worker_id: str) -> None:
        """Explicitly transfer an unfinished eligible metadata job to an idle worker."""
        with self._lock:
            worker = self._worker(worker_id)
            state = self._state(task_id)
            if state is not self._metadata_candidate() or state.phase != "METADATA_CLAIMED":
                raise ControllerError("only the current unfinished metadata job may be stolen")
            if state.metadata_owner == worker:
                raise ControllerError("worker already owns this metadata job")
            previous_owner = state.metadata_owner
            state.metadata_owner = worker
            self._record("metadata_stolen", state, worker=worker, previous_owner=previous_owner)

    def complete_metadata(self, task_id: str, worker_id: str) -> None:
        """Account for a completed N+1 metadata/range fetch without fetching it."""
        with self._lock:
            worker = self._worker(worker_id)
            state = self._state(task_id)
            if state.phase != "METADATA_CLAIMED" or state.metadata_owner != worker:
                raise ControllerError("only the worker holding a claimed metadata job may complete it")
            self._assert_capacity(state.task.metadata_bytes, "metadata/range fetch")
            state.metadata_resident = True
            state.metadata_owner = None
            state.phase = "METADATA_READY"
            self._record("metadata_ready", state, worker=worker, job_kind=self._metadata_kind(state))

    def begin_processing(self, task_id: str, heavy_lease_token: str) -> None:
        """Materialize N only after N-1 was sealed and evicted."""
        with self._lock:
            state = self._state(task_id)
            self._assert_heavy_lease(heavy_lease_token)
            if self._active_task_id is not None:
                raise ControllerError("one-heavy-lease violation: another task is already processing")
            if state.phase != "METADATA_READY":
                raise ControllerError("processing requires a completed bounded metadata/range fetch")
            index = self._states.index(state)
            if index and self._states[index - 1].phase != "EVICTED":
                raise ControllerError("N-1 must be sealed and evicted before N may process")
            reserve = state.task.source_bytes + state.task.scratch_bytes + sum(state.task.artifact_bytes.values())
            self._assert_capacity(reserve, "source plus one-pass artifact reservation")
            self._claim_heavy_operation(state)
            state.source_resident = True
            state.reserved_artifact_bytes = sum(state.task.artifact_bytes.values())
            state.phase = "PROCESSING"
            self._active_task_id = state.task.task_id
            self._record(
                "processing_started",
                state,
                lease="held",
                reserved_artifact_bytes=state.reserved_artifact_bytes,
                scratch_bytes=state.task.scratch_bytes,
            )

    def finish_one_pass(self, task_id: str, heavy_lease_token: str, artifacts: Mapping[str, int]) -> None:
        """Record the one permitted multi-artifact pack pass for N."""
        with self._lock:
            state = self._state(task_id)
            self._assert_active(state, heavy_lease_token)
            if state.phase != "PROCESSING":
                raise ControllerError("one-pass packing is permitted exactly once after processing begins")
            actual = self._validate_artifacts(artifacts, state.task.artifact_bytes)
            state.packed_artifacts = actual
            state.reserved_artifact_bytes = sum(actual.values())
            state.phase = "PACKED"
            self._record("one_pass_multi_artifact_packed", state, artifact_ids=sorted(actual))

    def seal_and_evict(self, task_id: str, heavy_lease_token: str, seal_sha256: str) -> None:
        """Seal N and evict its source, metadata, and temporary artifact family."""
        with self._lock:
            state = self._state(task_id)
            self._assert_active(state, heavy_lease_token)
            if state.phase != "PACKED" or state.packed_artifacts is None:
                raise ControllerError("seal/evict requires the one-pass artifact family")
            self._validate_seal(seal_sha256)
            state.seal_sha256 = seal_sha256
            state.phase = "EVICTED"
            state.metadata_resident = False
            state.source_resident = False
            state.reserved_artifact_bytes = 0
            self._active_task_id = None
            self._release_heavy_operation(state)
            self._record("sealed_and_evicted", state, seal_sha256=seal_sha256)

    def record_profile_sample(self, task_id: str, *, rate_id: str, profile_id: str, metrics: Mapping[str, float]) -> None:
        """Share numeric rate/profile observations across all workers and tasks."""
        with self._lock:
            self._state(task_id)
            rate = self._identifier(rate_id, "rate_id")
            profile = self._identifier(profile_id, "profile_id")
            if not metrics:
                raise ControllerError("profile metrics must be non-empty")
            checked: dict[str, float] = {}
            for name, value in metrics.items():
                metric = self._identifier(name, "metric name")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)) or float(value) < 0:
                    raise ControllerError("profile metrics must be finite, non-negative numbers")
                checked[metric] = float(value)
            stats = self._stats.setdefault((rate, profile), _SharedStats())
            stats.add(task_id, checked)

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-ready accounting; this performs no disk or network I/O."""
        with self._lock:
            used = self._used_bytes()
            return {
                "schema": SCHEMA,
                "byte_budget_bytes": self._budget,
                "resident_bytes": used,
                "available_bytes": self._budget - used,
                "active_task_id": self._active_task_id,
                "heavy_lease": "HELD" if self._active_task_id is not None else "IDLE",
                "tasks": [
                    {"task_id": state.task.task_id, "phase": state.phase, "metadata_resident": state.metadata_resident,
                     "source_resident": state.source_resident, "resident_bytes": state.resident_bytes,
                     "scratch_bytes": state.task.scratch_bytes if state.source_resident else 0,
                     "packed_artifacts": dict(state.packed_artifacts or {}), "seal_sha256": state.seal_sha256}
                    for state in self._states
                ],
                "shared_rate_profile_stats": {f"{rate_id}/{profile_id}": stats.snapshot() for (rate_id, profile_id), stats in sorted(self._stats.items())},
                "events": [dict(event) for event in self._events],
            }

    def _metadata_candidate(self) -> _TaskState | None:
        if self._active_task_id is not None:
            active_index = next(index for index, state in enumerate(self._states) if state.task.task_id == self._active_task_id)
            next_index = active_index + 1
            return self._states[next_index] if next_index < len(self._states) else None
        for state in self._states:
            if state.phase != "EVICTED":
                return state
        return None

    def _metadata_kind(self, state: _TaskState) -> str:
        return "bootstrap_N" if self._active_task_id is None else "N_plus_1_metadata_range"

    def _state(self, task_id: str) -> _TaskState:
        try:
            return self._by_id[task_id]
        except KeyError as exc:
            raise ControllerError(f"unknown task_id: {task_id!r}") from exc

    def _assert_active(self, state: _TaskState, heavy_lease_token: str) -> None:
        self._assert_heavy_lease(heavy_lease_token)
        if self._active_task_id != state.task.task_id:
            raise ControllerError("one-heavy-lease violation: this task does not own the active lease")

    def _assert_heavy_lease(self, token: str) -> None:
        if token != self._heavy_lease_token:
            raise ControllerError("one-heavy-lease violation: expected lease token was not presented")

    def _claim_heavy_operation(self, state: _TaskState) -> None:
        """Reserve the process-wide heavy slot only after local admission passes."""
        global _HEAVY_OPERATION_OWNER
        with _HEAVY_OPERATION_LOCK:
            if _HEAVY_OPERATION_OWNER is not None:
                raise ControllerError(
                    "one-heavy-lease violation: a heavy operation is already active in another controller"
                )
            _HEAVY_OPERATION_OWNER = (id(self), state.task.task_id)

    def _release_heavy_operation(self, state: _TaskState) -> None:
        """Release the heavy slot only as part of a successful seal-and-evict."""
        global _HEAVY_OPERATION_OWNER
        expected = (id(self), state.task.task_id)
        with _HEAVY_OPERATION_LOCK:
            if _HEAVY_OPERATION_OWNER != expected:
                raise ControllerError("one-heavy-lease violation: process-wide heavy-slot ownership changed")
            _HEAVY_OPERATION_OWNER = None

    def _assert_capacity(self, additional_bytes: int, operation: str) -> None:
        required = self._used_bytes() + additional_bytes
        if required > self._budget:
            raise ControllerError(f"capacity violation during {operation}: need {required}, budget is {self._budget}")

    def _used_bytes(self) -> int:
        return sum(state.resident_bytes for state in self._states)

    def _validate_artifacts(self, artifacts: Mapping[str, int], expected: Mapping[str, int]) -> dict[str, int]:
        if set(artifacts) != set(expected):
            raise ControllerError("one-pass artifacts must exactly match the declared artifact family")
        actual: dict[str, int] = {}
        for artifact_id, limit in expected.items():
            value = artifacts[artifact_id]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ControllerError("actual artifact byte counts must be positive integers")
            if value > limit:
                raise ControllerError("actual artifact bytes exceed the byte-budgeted plan")
            actual[artifact_id] = value
        return actual

    def _worker(self, worker_id: str) -> str:
        return self._identifier(worker_id, "worker_id")

    @staticmethod
    def _identifier(value: str, label: str) -> str:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ControllerError(f"{label} must be a non-empty, trimmed string")
        return value

    @staticmethod
    def _validate_seal(value: str) -> None:
        if not isinstance(value, str) or len(value) != 64 or any(char not in _SEAL_HEX for char in value):
            raise ControllerError("seal_sha256 must be 64 lowercase hex characters")

    def _record(self, kind: str, state: _TaskState, **detail: Any) -> None:
        self._events.append({"n": len(self._events), "kind": kind, "task_id": state.task.task_id, **detail})

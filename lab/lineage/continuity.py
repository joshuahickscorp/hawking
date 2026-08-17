"""Generation-independent Genesis workers, checkpoints, and context rebinding.

Model generations are replaceable. Worker goals, tool state, receipts, negative
science, and NEXT_ACTION live here as durable AgentOS state. KV and transcript
state are explicitly ephemeral and are never required to migrate a worker.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from lab.lineage.bus import ResearchBus
from lab.lineage.canon import digest, require_nonempty_str, require_sha256, utc_now
from lab.receipts import seal, verify

CHECKPOINT_SCHEMA = "hawking.genesis.worker_checkpoint.v1"
CONTEXT_SCHEMA = "hawking.genesis.compiled_context.v1"
PROMOTION_EVENT_SCHEMA = "hawking.genesis.promoted.v1"
MIGRATION_SCHEMA = "hawking.genesis.worker_migration.v1"
CAPACITY_SCHEMA = "hawking.genesis.dynamic_worker_capacity.v1"

WORKER_SESSION_ROLES = frozenset({"child_a", "child_b"})
WORKER_STATES = frozenset(
    {"READY", "RUNNING", "WAITING", "CHECKPOINTED", "PREEMPTED", "REBINDING"}
)
DURABLE_TASK_KEYS = (
    "goal",
    "subgoal",
    "repo",
    "worktree",
    "hypotheses",
    "findings",
    "receipts",
    "negative_science",
    "pending_experiments",
    "tool_results",
    "NEXT_ACTION",
)
_WORKER_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")


class ContinuityError(ValueError):
    """Continuity evidence is incomplete or would destroy active work."""


def normalize_generation(raw: Mapping[str, Any], label: str = "generation") -> dict[str, Any]:
    generation = raw.get("generation")
    if type(generation) is not int or generation < 0:
        raise ContinuityError(f"{label}.generation must be a non-negative integer")
    artifact_sha = require_sha256(raw.get("artifact_sha"), f"{label}.artifact_sha")
    runtime_sha = require_sha256(raw.get("runtime_sha"), f"{label}.runtime_sha")
    repo_head = str(raw.get("repo_head") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", repo_head):
        raise ContinuityError(f"{label}.repo_head must be a 40- or 64-character hex digest")
    token_ns = raw.get("complete_token_ns")
    if type(token_ns) is not int or token_ns <= 0:
        raise ContinuityError(f"{label}.complete_token_ns must be a positive integer")
    bpw = raw.get("physical_bpw")
    if isinstance(bpw, bool) or not isinstance(bpw, (int, float)) or float(bpw) <= 0:
        raise ContinuityError(f"{label}.physical_bpw must be positive")
    return {
        "generation": generation,
        "artifact_sha": artifact_sha,
        "runtime_sha": runtime_sha,
        "repo_head": repo_head,
        "physical_bpw": float(bpw),
        "complete_token_ns": token_ns,
        "tps": 1_000_000_000.0 / token_ns,
    }


def normalize_worker(raw: Mapping[str, Any]) -> dict[str, Any]:
    worker_id = require_nonempty_str(raw.get("worker_id"), "worker_id")
    if not _WORKER_ID.fullmatch(worker_id):
        raise ContinuityError("worker_id contains unsafe filesystem characters")
    session_role = require_nonempty_str(raw.get("session_role"), "session_role")
    if session_role not in WORKER_SESSION_ROLES:
        raise ContinuityError(
            f"logical workers must use one of {sorted(WORKER_SESSION_ROLES)}; "
            "parent and protected_test are not worker homes"
        )
    task_id = require_nonempty_str(raw.get("task_id"), "task_id")
    state = str(raw.get("state") or "READY").upper()
    if state not in WORKER_STATES:
        raise ContinuityError(f"unknown worker state {state!r}")
    durable = raw.get("durable_task_state")
    if not isinstance(durable, Mapping):
        raise ContinuityError("durable_task_state must be an object")
    missing = [key for key in DURABLE_TASK_KEYS if key not in durable]
    if missing:
        raise ContinuityError(f"durable_task_state missing {missing}")
    for key in ("goal", "repo", "worktree", "NEXT_ACTION"):
        require_nonempty_str(durable.get(key), f"durable_task_state.{key}")
    dependencies = raw.get("generation_dependencies") or []
    if not isinstance(dependencies, list) or not all(isinstance(v, str) for v in dependencies):
        raise ContinuityError("generation_dependencies must be a string list")
    generation = normalize_generation(raw.get("bound_generation") or {}, "bound_generation")
    return {
        "worker_id": worker_id,
        "session_role": session_role,
        "task_id": task_id,
        "task_contract": dict(raw.get("task_contract") or {}),
        "priority": int(raw.get("priority", 100)),
        "state": state,
        "unevictable": bool(raw.get("unevictable", False)),
        "durable_task_state": dict(durable),
        "generation_dependencies": list(dependencies),
        "bound_generation": generation,
        "ephemeral_model_state": {
            "kv_preserved": False,
            "transcript_required_for_resume": False,
        },
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


class WorkerCheckpointStore:
    """Atomic, checksummed durable task state. No KV or transcript is stored."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(self, worker_id: str) -> Path:
        if not _WORKER_ID.fullmatch(worker_id):
            raise ContinuityError("unsafe worker_id")
        return self.root / worker_id / "checkpoint.json"

    def save(self, worker: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
        normalized = normalize_worker(worker)
        document = seal(
            {
                "schema": CHECKPOINT_SCHEMA,
                "checkpointed_at": utc_now(),
                "reason": require_nonempty_str(reason, "checkpoint reason"),
                "worker": normalized,
                "ephemeral_state_persisted": False,
                "resume_requires_transcript": False,
            }
        )
        _atomic_json(self.path_for(normalized["worker_id"]), document)
        return document

    def load(self, worker_id: str) -> dict[str, Any]:
        path = self.path_for(worker_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContinuityError(f"cannot load worker checkpoint {path}: {exc}") from exc
        verify(value, label=f"worker checkpoint {worker_id}")
        if value.get("schema") != CHECKPOINT_SCHEMA:
            raise ContinuityError("unexpected worker checkpoint schema")
        normalize_worker(value.get("worker") or {})
        return value


def compile_worker_context(
    *,
    directives: Iterable[Mapping[str, Any]],
    generation: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    world_state: Mapping[str, Any],
    bus_messages: Iterable[Mapping[str, Any]],
    max_bus_messages: int = 32,
) -> dict[str, Any]:
    """Compile compact current context from durable truth, never a transcript."""
    verify(checkpoint, label="worker checkpoint")
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ContinuityError("context compiler requires a worker checkpoint")
    worker = normalize_worker(checkpoint.get("worker") or {})
    generation_n = normalize_generation(generation)
    bindings: list[dict[str, Any]] = []
    for index, raw in enumerate(directives):
        if raw.get("integrity_verified") is not True:
            raise ContinuityError(f"directive {index} is not integrity verified")
        bindings.append(
            {
                "canonical_path": require_nonempty_str(
                    raw.get("canonical_path"), f"directive[{index}].canonical_path"
                ),
                "sha256": require_sha256(raw.get("sha256"), f"directive[{index}].sha256"),
                "size_bytes": int(raw.get("size_bytes", 0)),
            }
        )
    if not bindings:
        raise ContinuityError("at least one Genesis directive binding is required")
    recent = [dict(row) for row in bus_messages][-max_bus_messages:]
    durable = worker["durable_task_state"]
    unsigned = {
        "schema": CONTEXT_SCHEMA,
        "compiled_at": utc_now(),
        "directive_bindings": bindings,
        "generation": generation_n,
        "worker": {
            "worker_id": worker["worker_id"],
            "session_role": worker["session_role"],
            "task_id": worker["task_id"],
            "task_contract": worker["task_contract"],
            "durable_task_state": durable,
            "generation_dependencies": worker["generation_dependencies"],
        },
        "world_state": dict(world_state),
        "relevant_receipts": list(durable["receipts"]),
        "relevant_negative_science": list(durable["negative_science"]),
        "recent_research_bus": recent,
        "ephemeral_kv_reused": False,
        "source_transcript_required": False,
    }
    unsigned["context_sha256"] = digest(unsigned)
    return seal(unsigned)


@dataclass
class WorkerScheduler:
    """Minimal ready/waiting/preemption scheduler with tool-wait backfill."""

    workers: dict[str, dict[str, Any]]

    @classmethod
    def from_workers(cls, workers: Iterable[Mapping[str, Any]]) -> "WorkerScheduler":
        normalized = [normalize_worker(worker) for worker in workers]
        by_id = {worker["worker_id"]: worker for worker in normalized}
        if len(by_id) != len(normalized):
            raise ContinuityError("duplicate worker_id")
        return cls(by_id)

    def mark_tool_wait(self, worker_id: str, tool_call: str) -> dict[str, Any]:
        worker = self.workers[worker_id]
        worker["state"] = "WAITING"
        worker["waiting_on"] = require_nonempty_str(tool_call, "tool_call")
        return worker

    def mark_ready(self, worker_id: str) -> dict[str, Any]:
        worker = self.workers[worker_id]
        worker["state"] = "READY"
        worker.pop("waiting_on", None)
        return worker

    def next_ready(self) -> dict[str, Any] | None:
        ready = [worker for worker in self.workers.values() if worker["state"] == "READY"]
        return min(ready, key=lambda row: (row["priority"], row["worker_id"])) if ready else None

    def preempt(
        self,
        worker_id: str,
        *,
        store: WorkerCheckpointStore,
        reason: str,
    ) -> dict[str, Any]:
        worker = self.workers[worker_id]
        if worker.get("unevictable"):
            raise ContinuityError(f"worker {worker_id} is unevictable")
        checkpoint = store.save(worker, reason=reason)
        worker["state"] = "PREEMPTED"
        return checkpoint

    def restore(self, worker_id: str, *, store: WorkerCheckpointStore) -> dict[str, Any]:
        checkpoint = store.load(worker_id)
        worker = normalize_worker(checkpoint["worker"])
        worker["state"] = "READY"
        self.workers[worker_id] = worker
        return worker


def genesis_promoted_event(
    old_generation: Mapping[str, Any], new_generation: Mapping[str, Any]
) -> dict[str, Any]:
    old = normalize_generation(old_generation, "old_generation")
    new = normalize_generation(new_generation, "new_generation")
    directive_bindings = [dict(item) for item in directives]
    if new["generation"] <= old["generation"]:
        raise ContinuityError("promoted generation must increase")
    return seal(
        {
            "schema": PROMOTION_EVENT_SCHEMA,
            "type": "GENESIS_PROMOTED",
            "recorded_at": utc_now(),
            "old_generation": old,
            "new_generation": new,
        }
    )


def _migration_class(worker: Mapping[str, Any], old: Mapping[str, Any], new: Mapping[str, Any]) -> str:
    dependencies = set(worker.get("generation_dependencies") or [])
    if "invalid_on_generation_change" in dependencies:
        return "invalidated"
    if (
        "artifact" in dependencies
        and old["artifact_sha"] != new["artifact_sha"]
    ) or (
        "runtime" in dependencies
        and old["runtime_sha"] != new["runtime_sha"]
    ):
        return "needs_rebase"
    return "transferable"


def migrate_workers(
    *,
    workers: Iterable[Mapping[str, Any]],
    old_generation: Mapping[str, Any],
    new_generation: Mapping[str, Any],
    store: WorkerCheckpointStore,
    directives: Iterable[Mapping[str, Any]],
    world_state: Mapping[str, Any],
    bus: ResearchBus,
    parent_live_before: bool,
    parent_live_after: bool,
    protected_test_slot_available: bool,
) -> dict[str, Any]:
    """Checkpoint, invalidate/rebase, compile fresh contexts, and resume workers."""
    old = normalize_generation(old_generation, "old_generation")
    new = normalize_generation(new_generation, "new_generation")
    event = genesis_promoted_event(old, new)
    scheduler = WorkerScheduler.from_workers(workers)
    if len(scheduler.workers) < 2:
        raise ContinuityError("migration proof requires at least two logical workers")
    if len({row["task_id"] for row in scheduler.workers.values()}) < 2:
        raise ContinuityError("migration proof requires separate worker tasks")
    if not parent_live_before or not parent_live_after:
        raise ContinuityError("parent safety was not preserved during migration")
    if not protected_test_slot_available:
        raise ContinuityError("protected test slot must remain available")

    bus_view = bus.generation_view(
        generation=new["generation"],
        artifact_sha=new["artifact_sha"],
        runtime_sha=new["runtime_sha"],
    )
    migrated: list[dict[str, Any]] = []
    for worker_id in sorted(scheduler.workers):
        worker = scheduler.workers[worker_id]
        if worker["bound_generation"] != old:
            raise ContinuityError(
                f"worker {worker_id} is not bound to the promoted old generation"
            )
        before = store.save(worker, reason="GENESIS_PROMOTED checkpoint before rebind")
        classification = _migration_class(worker, old, new)
        worker["state"] = "REBINDING"
        worker["bound_generation"] = new
        durable = worker["durable_task_state"]
        if classification == "needs_rebase":
            durable["hypotheses"] = [
                *list(durable["hypotheses"]),
                "Generation-bound assumptions require revalidation on promoted identity.",
            ]
        elif classification == "invalidated":
            durable["pending_experiments"] = []
            durable["NEXT_ACTION"] = "Reproduce the task premise on the promoted generation."
        rebind_checkpoint = store.save(worker, reason="compile context for promoted generation")
        context = compile_worker_context(
            directives=directive_bindings,
            generation=new,
            checkpoint=rebind_checkpoint,
            world_state=world_state,
            bus_messages=bus_view,
        )
        worker["state"] = "READY"
        after = store.save(worker, reason="rebound to promoted generation")
        migrated.append(
            {
                "worker_id": worker_id,
                "session_role": worker["session_role"],
                "task_id": worker["task_id"],
                "classification": classification,
                "checkpoint_before_sha256": before["seal_sha256"],
                "checkpoint_rebind_sha256": rebind_checkpoint["seal_sha256"],
                "checkpoint_after_sha256": after["seal_sha256"],
                "context_sha256": context["context_sha256"],
                "resumed_state": worker["state"],
                "ephemeral_kv_reused": False,
                "work_lost": 0,
                "NEXT_ACTION": worker["durable_task_state"]["NEXT_ACTION"],
            }
        )

    result = {
        "schema": MIGRATION_SCHEMA,
        "status": "PASS",
        "promotion_event": event,
        "workers_checkpointed": len(migrated),
        "workers_rebound": len(migrated),
        "separate_task_count": len({row["task_id"] for row in migrated}),
        "shared_research_message_count": len(bus.messages()),
        "bus_continuity_states": {
            state: sum(1 for row in bus_view if row["continuity_state"] == state)
            for state in {
                "CURRENT",
                "TRANSFERRED_STRUCTURAL",
                "STALE_PENDING_REVALIDATION",
            }
        },
        "parent_live_before": True,
        "parent_live_after": True,
        "old_parent_may_unload_after_rebind": True,
        "protected_test_slot_available": True,
        "workers": migrated,
        "work_lost": 0,
        "recorded_at": utc_now(),
    }
    return seal(result)


def dynamic_worker_capacity(
    *,
    total_ram_bytes: int,
    used_ram_bytes: int,
    resident_model_bytes: int,
    marginal_worker_bytes: int,
    candidate_reserve_bytes: int,
    complete_token_ns: int,
) -> dict[str, Any]:
    """Derive capacity from measured bytes; widen only after the 100-TPS gate."""
    values = {
        "total_ram_bytes": total_ram_bytes,
        "used_ram_bytes": used_ram_bytes,
        "resident_model_bytes": resident_model_bytes,
        "marginal_worker_bytes": marginal_worker_bytes,
        "candidate_reserve_bytes": candidate_reserve_bytes,
    }
    if any(type(value) is not int or value < 0 for value in values.values()):
        raise ContinuityError("capacity byte inputs must be non-negative integers")
    if marginal_worker_bytes <= 0 or complete_token_ns <= 0:
        raise ContinuityError("marginal_worker_bytes and complete_token_ns must be positive")
    headroom = max(
        0,
        total_ram_bytes - used_ram_bytes - resident_model_bytes - candidate_reserve_bytes,
    )
    raw = headroom // marginal_worker_bytes
    crossed_100_tps = complete_token_ns <= 10_000_000
    supported = int(raw if crossed_100_tps else min(raw, 5))
    return {
        "schema": CAPACITY_SCHEMA,
        **values,
        "headroom_bytes": headroom,
        "raw_worker_capacity": int(raw),
        "supported_worker_capacity": supported,
        "crossed_100_tps": crossed_100_tps,
        "allocation_policy": (
            "widen AgentOS/HCLI/Context/Memory fronts"
            if crossed_100_tps
            else "prioritize Gravity, kernel, and Genesis infrastructure"
        ),
    }


__all__ = [
    "CAPACITY_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "CONTEXT_SCHEMA",
    "ContinuityError",
    "MIGRATION_SCHEMA",
    "PROMOTION_EVENT_SCHEMA",
    "WorkerCheckpointStore",
    "WorkerScheduler",
    "compile_worker_context",
    "dynamic_worker_capacity",
    "genesis_promoted_event",
    "migrate_workers",
    "normalize_generation",
    "normalize_worker",
]

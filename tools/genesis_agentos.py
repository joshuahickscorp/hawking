#!/usr/bin/env python3
"""Durable AgentOS turns over the already-resident Genesis worker sessions.

This is the execution bridge between the persisted logical-worker registry and
HCLI's model/tool plane.  It does not load a second model, run a protected GPU
benchmark, or mutate lineage.  A worker may inspect, edit, compile, and test in
its owned worktree, then use HCLI's ``submit_candidate`` tool to hand a concrete
child request to the external lifecycle controller.

The daemon invokes one bounded ``tick`` only when no protected GPU work is
reserved.  Each turn is checkpointed and persisted, so a crash loses neither
the worker's durable task state nor the fact that it attempted an action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lab.hcli.special_unit import (  # noqa: E402
    DEFAULT_TOOL_NAMES,
    TOOL_MAX_NEW_TOKENS,
    GenesisResidentBackend,
    ResourceGate,
    Session,
    SpecialUnit,
)
from lab.lineage.bus import ResearchBus  # noqa: E402
from lab.lineage.continuity import (  # noqa: E402
    WorkerCheckpointStore,
    compile_worker_context,
)
from lab.lineage.lifecycle import (  # noqa: E402
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_LINEAGE_PATH,
    DEFAULT_WORKER_REGISTRY,
    WorkerRegistry,
)
from lab.lineage.state import LineageState  # noqa: E402
from lab.receipts import seal  # noqa: E402
from tools.agentos.genesis_contract import contract_provenance  # noqa: E402


GPU_LOCK = Path("/tmp/hawking-gpu-lane.lock")
DEFAULT_SESSION_ROOT = REPO / "workspace" / "ops" / "genesis-agentos-sessions"
DEFAULT_STOPFILE = REPO / "workspace" / "ops" / "GENESIS_STOP"
SCHEMA = "hawking.genesis.agentos_tick.v1"
# A durable unattended worker gets the concrete edit/test plane, not an
# unrestricted shell.  That keeps lifecycle authority outside the model while
# still allowing source reads, writes, grep, CPU tests, Rust tests, and a
# one-way candidate handoff.
AGENTOS_TOOL_NAMES = tuple(name for name in DEFAULT_TOOL_NAMES if name != "bash")


class AgentOSTurn(Protocol):
    def to_dict(self) -> dict[str, Any]: ...


class UnitFactory(Protocol):
    def __call__(self, worker: Mapping[str, Any], worktree: Path) -> Any: ...


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _process_alive(pid: object) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except OSError:
        return False
    return True


def _load_lineage(path: Path) -> LineageState:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read lineage state {path}: {exc}") from exc
    return LineageState.from_dict(raw)


def _receipt_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _claim_worker(
    registry: WorkerRegistry,
    *,
    requested_worker: str | None,
) -> tuple[dict[str, Any], str] | None:
    """CAS-claim one logical worker, reclaiming only dead runner claims."""
    workers, previous_sha = registry.load()
    if previous_sha is None:
        raise RuntimeError("AgentOS worker registry is not bootstrapped")
    changed = False
    for worker in workers:
        durable = worker["durable_task_state"]
        claim = durable.get("agentos_claim")
        if worker.get("state") == "RUNNING" and isinstance(claim, Mapping):
            if not _process_alive(claim.get("pid")):
                worker["state"] = "READY"
                durable["agentos_recovered_claim"] = {
                    "at": _utc(),
                    "previous_claim": dict(claim),
                }
                durable.pop("agentos_claim", None)
                changed = True
    if changed:
        registry.replace(workers, expected_previous_sha256=previous_sha)
        workers, previous_sha = registry.load()
        if previous_sha is None:
            raise RuntimeError("worker registry vanished while recovering a stale claim")

    ready = [
        row
        for row in workers
        if row.get("state") == "READY"
        and (requested_worker is None or row.get("worker_id") == requested_worker)
    ]
    if not ready:
        return None
    worker = min(ready, key=lambda row: (int(row.get("priority", 100)), row["worker_id"]))
    worker["state"] = "RUNNING"
    worker["durable_task_state"]["agentos_claim"] = {
        "pid": os.getpid(),
        "claimed_at": _utc(),
        "worker_id": worker["worker_id"],
    }
    registry.replace(workers, expected_previous_sha256=previous_sha)
    _fresh, claimed_sha = registry.load()
    if claimed_sha is None:
        raise RuntimeError("worker registry disappeared after AgentOS claim")
    return dict(worker), claimed_sha


def _finish_worker(
    registry: WorkerRegistry,
    *,
    worker_id: str,
    claimed_sha: str,
    turn: Mapping[str, Any],
    checkpoint_root: Path,
) -> None:
    """Persist only actual tool/model observations, then release the worker."""
    workers, observed_sha = registry.load()
    if observed_sha != claimed_sha:
        raise RuntimeError("worker registry changed during AgentOS turn; refusing blind overwrite")
    worker = next((row for row in workers if row.get("worker_id") == worker_id), None)
    if worker is None or worker.get("state") != "RUNNING":
        raise RuntimeError("claimed worker is no longer RUNNING")
    durable = worker["durable_task_state"]
    results = list(durable.get("tool_results") or [])
    results.append(dict(turn))
    durable["tool_results"] = results[-24:]
    receipts = list(durable.get("receipts") or [])
    receipts.append(
        {
            "kind": "agentos_hcli_turn",
            "sha256": _receipt_digest(turn),
            "at": _utc(),
            "ok": bool(turn.get("ok")),
        }
    )
    durable["receipts"] = receipts[-48:]
    turn_digest = _receipt_digest(turn)
    durable["last_agentos_turn_receipt_sha256"] = turn_digest
    durable["last_agentos_turn"] = {
        "at": _utc(),
        "ok": bool(turn.get("ok")),
        "summary": str(turn.get("summary") or "")[:500],
    }
    if turn.get("outcome") == "TURN_COMPLETE" and turn.get("ok") is True:
        durable["NEXT_ACTION"] = (
            "Resume from actual AgentOS HCLI turn "
            f"{turn_digest[:16]}: inspect its recorded tool result and execute the next "
            "smallest falsifiable implementation or test step."
        )
    else:
        durable["NEXT_ACTION"] = (
            "Repair or falsify the failed AgentOS HCLI turn "
            f"{turn_digest[:16]} before broadening the search."
        )
    session_id = turn.get("hcli_session_id")
    if isinstance(session_id, str) and session_id:
        durable["hcli_session_id"] = session_id
    durable.pop("agentos_claim", None)
    worker["state"] = "READY"
    WorkerCheckpointStore(checkpoint_root).save(worker, reason="AgentOS HCLI turn persisted")
    registry.replace(workers, expected_previous_sha256=claimed_sha)


def _default_unit_factory(
    *,
    session_root: Path,
    candidate_root: Path,
    max_new_tokens: int,
) -> UnitFactory:
    def make(worker: Mapping[str, Any], worktree: Path) -> SpecialUnit:
        durable = worker["durable_task_state"]
        session_id = str(durable.get("hcli_session_id") or f"genesis-{worker['worker_id']}")
        session_path = session_root / session_id / "session.json"
        gate = ResourceGate(lock_path=GPU_LOCK)
        if session_path.is_file():
            unit = SpecialUnit.open(
                session_id,
                repo=worktree,
                session_root=session_root,
                owned_worktree=worktree,
            )
            unit.gate = gate
            unit.tools.gate = gate
            unit.tools.candidate_root = candidate_root
        else:
            unit = SpecialUnit(
                repo=worktree,
                session_root=session_root,
                session=Session(session_id=session_id),
                gate=gate,
                owned_worktree=worktree,
            )
            unit.tools.candidate_root = candidate_root
        unit.backend = GenesisResidentBackend(
            session_role=str(worker["session_role"]),
            gate=gate,
            max_new_tokens=max_new_tokens,
        )
        return unit

    return make


def _worker_prompt(
    *,
    worker: Mapping[str, Any],
    context: Mapping[str, Any],
    candidate_root: Path,
) -> str:
    durable = worker["durable_task_state"]
    compact = {
        "generation": context.get("generation"),
        "worker": context.get("worker"),
        "negative_science": context.get("relevant_negative_science"),
        "receipts": context.get("relevant_receipts"),
    }
    excerpt = json.dumps(compact, sort_keys=True, separators=(",", ":"))[:7_500]
    return (
        "You are a durable logical Genesis worker, not a new lineage child. "
        "Take one concrete implementation step in your OWNED WORKTREE. Use HCLI tools to "
        "inspect, edit, compile, or test; do not merely recommend a change. Your hard priority "
        "is physical BPW/unique-once weight bytes (Gravity) or complete-token runtime/kernel cost "
        "(kernel) until the protected system reaches 100 valid TPS. Do not run GPU benchmarks, "
        "touch protected gates/oracles, alter lineage, promote yourself, or claim a win without "
        "external protected evidence. Use prepare_candidate to turn real changed files into a "
        "complete but unqualified request; then use submit_candidate on that JSON only when ready. "
        "The external controller will independently test it. "
        "Candidate paths may be relative to your owned worktree: submission seals that worktree "
        "as the controller's resolution origin. "
        f"The inbox is external authority at {candidate_root}; never write it via bash.\n\n"
        f"WORKER: {worker['worker_id']} / {worker['session_role']}\n"
        f"GOAL: {durable['goal']}\nSUBGOAL: {durable['subgoal']}\n"
        f"NEXT_ACTION: {durable['NEXT_ACTION']}\n"
        f"DURABLE CONTEXT: {excerpt}\n\n"
        "Emit a well-formed tool_call now. Work in bounded steps and leave durable evidence for "
        "the next turn."
    )


def run_once(
    *,
    repo: Path = REPO,
    state_path: Path = DEFAULT_LINEAGE_PATH,
    worker_registry_path: Path = DEFAULT_WORKER_REGISTRY,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    candidate_root: Path = REPO / "workspace" / "ops" / "genesis-candidates",
    session_root: Path = DEFAULT_SESSION_ROOT,
    worker_id: str | None = None,
    max_rounds: int = 2,
    max_new_tokens: int = TOOL_MAX_NEW_TOKENS,
    unit_factory: UnitFactory | None = None,
) -> dict[str, Any]:
    """Run one guarded model/tool turn and persist its actual outcome."""
    if DEFAULT_STOPFILE.exists():
        return seal({"schema": SCHEMA, "outcome": "STOPPED", "recorded_at": _utc()})
    if GPU_LOCK.exists():
        return seal(
            {
                "schema": SCHEMA,
                "outcome": "DEFERRED_GPU_LANE_BUSY",
                "authority_moved": False,
                "recorded_at": _utc(),
            }
        )
    repo = Path(repo).resolve()
    registry = WorkerRegistry(Path(worker_registry_path))
    claimed = _claim_worker(registry, requested_worker=worker_id)
    if claimed is None:
        return seal(
            {
                "schema": SCHEMA,
                "outcome": "IDLE",
                "authority_moved": False,
                "recorded_at": _utc(),
            }
        )
    worker, claimed_sha = claimed
    durable = worker["durable_task_state"]
    worktree = Path(str(durable["worktree"])).resolve()
    raw_turn: dict[str, Any]
    try:
        if durable.get("worktree_isolated") is not True or worktree == repo:
            raise RuntimeError(
                "worker has no isolated worktree; run genesis_lifecycle provision-worktrees first"
            )
        if not worktree.is_dir():
            raise RuntimeError(f"worker owned worktree is unavailable: {worktree}")
        lineage = _load_lineage(Path(state_path))
        current = lineage.current
        if current is None or not current.valid:
            raise RuntimeError("CURRENT Genesis unavailable for AgentOS worker context")
        bound = worker["bound_generation"]
        if int(bound["generation"]) != current.generation or bound["artifact_sha"] != current.artifact_sha:
            raise RuntimeError("worker binding is stale; wait for protected continuity rebind")
        checkpoint = WorkerCheckpointStore(Path(checkpoint_root)).save(
            worker, reason="AgentOS HCLI turn before model action"
        )
        provenance = contract_provenance()
        directives = provenance.get("contracts")
        if not isinstance(directives, list):
            raise RuntimeError("canonical Genesis contract provenance is malformed")
        context = compile_worker_context(
            directives=[dict(row) for row in directives if isinstance(row, Mapping)],
            generation=bound,
            checkpoint=checkpoint,
            world_state={
                "CURRENT": current.to_dict(),
                "candidate_inbox": str(Path(candidate_root)),
            },
            bus_messages=ResearchBus().generation_view(
                generation=current.generation,
                artifact_sha=current.artifact_sha,
                runtime_sha=current.runtime_sha,
            ),
        )
        factory = unit_factory or _default_unit_factory(
            session_root=Path(session_root),
            candidate_root=Path(candidate_root),
            max_new_tokens=max_new_tokens,
        )
        unit = factory(worker, worktree)
        act = unit.act(
            _worker_prompt(worker=worker, context=context, candidate_root=Path(candidate_root)),
            known_tools=AGENTOS_TOOL_NAMES,
            max_new_tokens=max_new_tokens,
            max_rounds=max(1, min(int(max_rounds), 4)),
        )
        raw_turn = {
            "schema": SCHEMA,
            "outcome": "TURN_COMPLETE",
            # A model/tool turn is deliberately not a lineage transition.  Keep
            # that fact in every durable receipt so consumers never infer that
            # a successful edit, test, or candidate submission promoted it.
            "authority_moved": False,
            "worker_id": worker["worker_id"],
            "session_role": worker["session_role"],
            "ok": bool(act.ok),
            "summary": str(act.text)[:2_000],
            "act": act.to_dict(),
            "hcli_session_id": getattr(getattr(unit, "session", None), "session_id", None),
            "context_sha256": context.get("context_sha256"),
            "recorded_at": _utc(),
        }
    except Exception as exc:  # Persist a truthful failed attempt; do not strand RUNNING forever.
        raw_turn = {
            "schema": SCHEMA,
            "outcome": "TURN_FAILED",
            "authority_moved": False,
            "worker_id": worker["worker_id"],
            "session_role": worker["session_role"],
            "ok": False,
            "summary": f"{type(exc).__name__}: {exc}",
            "error_type": type(exc).__name__,
            "recorded_at": _utc(),
        }
    try:
        _finish_worker(
            registry,
            worker_id=worker["worker_id"],
            claimed_sha=claimed_sha,
            turn=raw_turn,
            checkpoint_root=Path(checkpoint_root),
        )
    except Exception as exc:
        raw_turn["persistence_error"] = f"{type(exc).__name__}: {exc}"
        raw_turn["outcome"] = "TURN_PERSISTENCE_FAILED"
        raw_turn["ok"] = False
    return seal(raw_turn)


def _status(registry_path: Path) -> dict[str, Any]:
    workers, _sha = WorkerRegistry(registry_path).load()
    return {
        "schema": SCHEMA,
        "workers": [
            {
                "worker_id": row["worker_id"],
                "session_role": row["session_role"],
                "state": row["state"],
                "generation": row["bound_generation"]["generation"],
                "next_action": row["durable_task_state"]["NEXT_ACTION"],
            }
            for row in workers
        ],
        "gpu_lane_busy": GPU_LOCK.exists(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="genesis_agentos")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("tick", "status"):
        item = sub.add_parser(name)
        item.add_argument("--repo", type=Path, default=REPO)
        item.add_argument("--state", type=Path, default=DEFAULT_LINEAGE_PATH)
        item.add_argument("--workers", type=Path, default=DEFAULT_WORKER_REGISTRY)
        item.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
        item.add_argument("--candidates", type=Path, default=REPO / "workspace" / "ops" / "genesis-candidates")
        item.add_argument("--sessions", type=Path, default=DEFAULT_SESSION_ROOT)
    tick = sub.choices["tick"]
    tick.add_argument("--worker", choices=("gravity", "kernel"), default=None)
    tick.add_argument("--max-rounds", type=int, default=2)
    tick.add_argument("--max-new-tokens", type=int, default=TOOL_MAX_NEW_TOKENS)
    args = parser.parse_args(argv)
    if args.command == "status":
        print(json.dumps(_status(args.workers), indent=2, sort_keys=True))
        return 0
    result = run_once(
        repo=args.repo,
        state_path=args.state,
        worker_registry_path=args.workers,
        checkpoint_root=args.checkpoints,
        candidate_root=args.candidates,
        session_root=args.sessions,
        worker_id=args.worker,
        max_rounds=args.max_rounds,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result.get("outcome") in {"TURN_FAILED", "TURN_PERSISTENCE_FAILED"} else 0


if __name__ == "__main__":
    raise SystemExit(main())

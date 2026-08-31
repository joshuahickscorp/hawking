"""NO-WAIT SCHEDULER — keep the resident moving while a detached unit is unfinished.

The autonomy loop calls subprocess.run and sits inside it. During a real 1-hour
trial that was tens of minutes inside one hashing call with runnable work sitting
in the same queue. A resident that waits synchronously while safe independent
work remains is an autonomy defect, not a slow resident.

This module does not replace detached.py, wakeup.py, workgraph.py, or
restart_supervisor.py. It sits on top of them:

    launch detached work
        -> persist the expected receipt contract
        -> release the reasoning context
        -> scheduler asks WHAT ELSE CAN RUN NOW (frontier / graph, not a guess)
        -> execute an independent WorkUnit (also detached — never subprocess.run)
        -> ingest the detached result when it lands, in landing order

The supervisor polls. The reasoning context is released before poll runs.
poll() inspects each handle once and returns; it is not a loop, not a wait,
and not a model-facing API (wakeup.py already refuses those aliases).

Nothing here measures hardware. A missing receipt is process_failed, never a
silent pass. The whole resident blocks only when there is no independent work
AND the exact dependency requires the in-flight result — that condition is
computed from the graph/frontier and recorded when it fires.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from hcli.persist import atomic_write_json
from hcli.resources import pid_is_alive
from hcli.workunit import WorkUnit
from tools.future._common import RECEIPTS, git, sha256_file, write_receipt
from tools.future.detached import (
    DetachedError,
    DetachedSupervisor,
    UnsafeCommandError,
    _as_mapping,
    _expected_receipt_of,
    identity_status,
)
from tools.future.restart_supervisor import rediscover_detached
from tools.future.wakeup import Watcher

RECEIPT = "NO_WAIT_SCHEDULER.json"
SCHEMA = "hawking.future.no_wait_scheduler.v1"
HANDLE_SCHEMA = "hawking.future.no_wait.handle.v1"
CONTRACT_SCHEMA = "hawking.future.no_wait.receipt_contract.v1"
BLOCK_SCHEMA = "hawking.future.no_wait.block.v1"
RECORDED_BY = "tools/future/no_wait_scheduler.py"
VERSION = 1

ERAS = (
    "I Genesis of the Laboratory",
    "II Compounding Civilization",
    "III Autonomous Science Civilization",
    "IV Synthetic Machine Civilization",
    "V Released Hawking Civilization",
)
ODYSSEYS = (
    "I WHAT IS TRUE?",
    "II WHAT DID HAWKING ALREADY LEARN?",
    "III WHERE IS HAWKING WRONG?",
)

# ingest_ready classification. open is not a landed row.
INGESTED = "ingested"
PROCESS_FAILED = "process_failed"
CANCELLED = "cancelled"
OPEN = "open"

# runnable_now. BLOCKED is the only whole-resident wait, and only with a name.
RUNNABLE = "RUNNABLE"
BLOCKED = "BLOCKED"
IDLE = "IDLE"
REFUSED = "REFUSED"

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. "
    "A supervision interval is process identity and disk landing, not a bench. "
    "poll is a supervisor operation; the reasoning context is already released."
)

# The model-facing completion bus (wakeup) refuses poll/wait. This poll is the
# supervisor entry that wakeup was written to keep off the generation loop.
SUPERVISOR_POLL_RULE = (
    "poll is a supervisor operation; the reasoning context is released before "
    "this runs. Do not call it from a model-reasoning loop."
)


class SchedulerError(RuntimeError):
    """Operational refusal with a reason. Never a success-shaped default."""

    def __init__(self, reason: str, *, fault: str = "refused") -> None:
        self.reason = reason
        self.fault = fault
        super().__init__(f"REFUSED [{fault}]: {reason}")


# ---------------------------------------------------------------------------
# Coercion. Absent identity is a refusal, not a generated id.
# ---------------------------------------------------------------------------


def _require_workspace(workspace: str | _os.PathLike[str] | None) -> Path:
    if workspace is None:
        raise SchedulerError(
            "workspace is required; refusing to detach into the live campaign",
            fault="workspace_required",
        )
    path = Path(workspace).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _as_handle(handle: Any) -> dict[str, Any]:
    if handle is None:
        raise SchedulerError("handle is required", fault="handle_required")
    if isinstance(handle, Mapping):
        row = dict(handle)
        if not str(row.get("job_id") or "").strip():
            raise SchedulerError("handle has no job_id", fault="handle_required")
        return row
    raise SchedulerError(
        f"handle must be a mapping, got {type(handle).__name__}",
        fault="handle_required",
    )


def _job_id_of(handle: Any) -> str:
    return str(_as_handle(handle)["job_id"])


def _blocked_ids(handles: Sequence[Any]) -> set[str]:
    ids: set[str] = set()
    for raw in handles:
        row = _as_handle(raw)
        for key in ("job_id", "unit_id", "workunit_id"):
            val = str(row.get(key) or "").strip()
            if val:
                ids.add(val)
    return ids


def _deps_of(unit: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for key in ("dependencies", "verification_depends_on", "depends_on"):
        raw = unit.get(key) or []
        if isinstance(raw, (str, bytes)):
            raise SchedulerError(
                f"{key} must be a list of ids, not a string",
                fault="bad_sequence",
            )
        if not isinstance(raw, (list, tuple)):
            continue
        for item in raw:
            text = str(item).strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
    return out


def _unit_id_of(unit: Mapping[str, Any]) -> str:
    return str(unit.get("id") or unit.get("unit_id") or unit.get("workunit_id") or "").strip()


def _receipt_present(path: Any) -> bool:
    if not path:
        return False
    try:
        return Path(str(path)).is_file()
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Receipt contract. Landing is a file at a named path, never a guessed success.
# ---------------------------------------------------------------------------


def expected_receipt(
    unit: Any,
    *,
    workspace: str | _os.PathLike[str],
    job_id: str | None = None,
) -> dict[str, Any]:
    """The contract that says what landing looks like. Does not spawn, does not wait."""
    root = _require_workspace(workspace)
    if unit is None:
        raise SchedulerError("unit is required", fault="unit_required")
    row = _as_mapping(unit)
    uid = _unit_id_of(row)
    if not uid:
        raise SchedulerError(
            "unit has no id; refusing to invent a receipt path",
            fault="unit_required",
        )
    token = str(job_id or uid)
    path = _expected_receipt_of(row, root, token)
    schema = row.get("output_schema") or row.get("receipt_schema")
    if schema is not None:
        schema = str(schema)
    return {
        "schema": CONTRACT_SCHEMA,
        "unit_id": uid,
        "job_id": token,
        "path": path,
        "required_schema": schema,
        "landing": (
            "a file exists at path; if required_schema is set the receipt's "
            "schema field must match. Absence is process_failed."
        ),
        "absent_is": PROCESS_FAILED,
        "never": "a silent pass, a default success, a guessed completion",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class NoWaitScheduler:
    """Supervisor-side no-wait glue over DetachedSupervisor.

    Does not block the resident on a child. Does not poll from a reasoning loop.
    Does not relaunch on rediscover. Does not invent a receipt.
    """

    def __init__(self, workspace: str | _os.PathLike[str]) -> None:
        self.workspace = _require_workspace(workspace)
        self.supervisor = DetachedSupervisor(self.workspace)
        self.root = self.workspace / "no_wait"
        self.handles_root = self.root / "handles"
        self.contracts_root = self.root / "contracts"
        self.blocks_root = self.root / "blocks"
        for path in (self.handles_root, self.contracts_root, self.blocks_root):
            path.mkdir(parents=True, exist_ok=True)
        self.watcher = Watcher(self.root / "wakeup_ledger.json", root=self.workspace)
        self.block_events: list[dict[str, Any]] = []

    def _handle_path(self, job_id: str) -> Path:
        return self.handles_root / f"{job_id}.json"

    def _contract_path(self, job_id: str) -> Path:
        return self.contracts_root / f"{job_id}.json"

    def _write_handle(self, handle: Mapping[str, Any]) -> None:
        atomic_write_json(self._handle_path(str(handle["job_id"])), dict(handle))

    def _write_contract(self, contract: Mapping[str, Any]) -> None:
        atomic_write_json(self._contract_path(str(contract["job_id"])), dict(contract))

    def _load_contract(self, job_id: str) -> dict[str, Any] | None:
        path = self._contract_path(job_id)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SchedulerError(
                f"receipt contract unreadable at {path}: {type(exc).__name__}: {exc}",
                fault="contract_unreadable",
            ) from exc
        if not isinstance(value, dict):
            raise SchedulerError(
                f"receipt contract is not an object: {path}",
                fault="contract_unreadable",
            )
        return value

    def launch_detached(
        self,
        unit: Any,
        *,
        env: Optional[Mapping[str, str]] = None,
    ) -> dict[str, Any]:
        """Start the work, persist the landing contract, return a handle. Do not wait."""
        if unit is None:
            raise SchedulerError("unit is required", fault="unit_required")
        row = _as_mapping(unit)
        uid = _unit_id_of(row)
        if not uid:
            raise SchedulerError(
                "unit has no id; refusing to launch an anonymous unit",
                fault="unit_required",
            )
        try:
            rec = self.supervisor.launch(row, env=env)
        except UnsafeCommandError as exc:
            sleeping = dict(exc.record or {})
            if sleeping.get("job_id"):
                contract = expected_receipt(
                    row, workspace=self.workspace, job_id=str(sleeping["job_id"])
                )
                self._write_contract(contract)
                handle = _handle_from_record(
                    sleeping, contract, launch_refused=True, unit=row
                )
                self._write_handle(handle)
            raise
        except DetachedError as exc:
            raise SchedulerError(str(exc), fault="launch_failed") from exc

        job_id = str(rec["job_id"])
        contract = {
            **expected_receipt(row, workspace=self.workspace, job_id=job_id),
            "path": rec.get("expected_receipt_path")
            or expected_receipt(row, workspace=self.workspace, job_id=job_id)["path"],
            "launched_at": rec.get("started_at") or time.time(),
        }
        self._write_contract(contract)
        try:
            self.watcher.register_expectation(
                unit_id=uid,
                path=str(contract["path"]),
                dependents=list(row.get("dependents") or ()),
                required_schema=contract.get("required_schema"),
            )
        except Exception as exc:
            raise SchedulerError(
                f"could not persist wakeup expectation: {type(exc).__name__}: {exc}",
                fault="contract_unreadable",
            ) from exc

        handle = _handle_from_record(rec, contract, launch_refused=False, unit=row)
        self._write_handle(handle)
        return handle

    def poll(self, handles: Sequence[Any]) -> list[dict[str, Any]]:
        """Supervisor-side, cheap, one inspect per handle. Never a busy loop."""
        snapshots: list[dict[str, Any]] = []
        for raw in handles:
            job_id = _job_id_of(raw)
            try:
                rec = self.supervisor.inspect(job_id)
            except FileNotFoundError:
                snapshots.append(
                    {
                        "job_id": job_id,
                        "state": "missing",
                        "terminal": None,
                        "pid": None,
                        "identity_status": "unknown",
                        "expected_receipt_path": None,
                        "receipt_present": False,
                        "finished_at": None,
                        "running_at": None,
                        "ingest": PROCESS_FAILED,
                        "reason": "supervision record absent; never silently forgotten",
                    }
                )
                continue
            expected = rec.get("expected_receipt_path")
            present = _receipt_present(expected)
            terminal = rec.get("terminal")
            snapshots.append(
                {
                    "job_id": job_id,
                    "unit_id": rec.get("workunit_id"),
                    "state": rec.get("state"),
                    "terminal": terminal,
                    "pid": rec.get("pid"),
                    "identity_status": rec.get("identity_status") or identity_status(rec),
                    "expected_receipt_path": expected,
                    "receipt_present": present,
                    "finished_at": rec.get("finished_at"),
                    "running_at": rec.get("running_at"),
                    "started_at": rec.get("started_at"),
                    "returncode": rec.get("returncode"),
                    "crash_reason": rec.get("crash_reason"),
                    "ingest": _classify_ingest(terminal, present, rec.get("state")),
                    "reason": rec.get("crash_reason"),
                    "supervisor_rule": SUPERVISOR_POLL_RULE,
                }
            )
        return snapshots

    def runnable_now(
        self,
        blocked: Sequence[Any],
        *,
        candidates: Sequence[Mapping[str, Any]] | None = None,
        graph: Any = None,
        lanes: Iterable[str] | str | None = None,
    ) -> dict[str, Any]:
        """What else can proceed while those handles are open. Never spins."""
        return runnable_now(
            blocked,
            candidates=candidates,
            graph=graph,
            lanes=lanes,
            scheduler=self,
        )

    def ingest_ready(self, handles: Sequence[Any]) -> dict[str, Any]:
        """Collect what has landed, in the order it landed. Open handles stay open."""
        snaps = self.poll(handles)
        landed: list[dict[str, Any]] = []
        still_open: list[dict[str, Any]] = []
        for snap in snaps:
            ingest = snap.get("ingest")
            if ingest == OPEN or ingest is None:
                still_open.append(snap)
                continue
            row = dict(snap)
            row["ingest"] = ingest
            if ingest == PROCESS_FAILED and not row.get("reason"):
                row["reason"] = _process_failed_reason(snap)
            landed.append(row)
        landed.sort(
            key=lambda r: (
                float(r["finished_at"])
                if isinstance(r.get("finished_at"), (int, float))
                else 0.0,
                str(r.get("job_id") or ""),
            )
        )
        forgotten = [
            s
            for s in snaps
            if s.get("state") == "missing" and s.get("ingest") != PROCESS_FAILED
        ]
        if forgotten:
            raise SchedulerError(
                f"handles vanished without classification: {[s.get('job_id') for s in forgotten]}",
                fault="forgotten_handle",
            )
        return {
            "landed": landed,
            "still_open": still_open,
            "n_landed": len(landed),
            "n_open": len(still_open),
            "order": "finished_at ascending (landing order)",
            "forgotten": [],
            "evidence_class": "STATIC_ONLY",
        }

    def cancel(self, handle: Any) -> dict[str, Any]:
        """Cancel and prove the process is gone. A live pid after cancel is a refusal."""
        row = _as_handle(handle)
        job_id = str(row["job_id"])
        try:
            rec = self.supervisor.cancel(job_id)
        except FileNotFoundError as exc:
            raise SchedulerError(
                f"cannot cancel {job_id}: supervision record absent",
                fault="handle_required",
            ) from exc
        pid = rec.get("pid") if rec.get("pid") is not None else row.get("pid")
        gone = not (isinstance(pid, int) and pid > 0 and pid_is_alive(pid))
        if rec.get("terminal") == "cancelled" and not gone:
            raise SchedulerError(
                f"cancel reported cancelled for {job_id} but pid {pid} is still alive",
                fault="cancel_did_not_stop",
            )
        handle_out = dict(row)
        handle_out.update(
            {
                "terminal": rec.get("terminal"),
                "state": rec.get("state"),
                "pid": rec.get("pid"),
                "process_gone": gone,
                "crash_reason": rec.get("crash_reason"),
                "finished_at": rec.get("finished_at"),
                "identity_status": rec.get("identity_status") or identity_status(rec),
            }
        )
        self._write_handle(handle_out)
        return handle_out

    def rediscover(self) -> dict[str, Any]:
        """Adopt handles that outlived a resident restart. Never relaunch."""
        report = rediscover_detached(detached=self.supervisor)
        if report.get("relaunched"):
            raise SchedulerError(
                "rediscover attempted a relaunch; this is a scheduler bug",
                fault="relaunch_forbidden",
            )
        handles: list[dict[str, Any]] = []
        for job in report.get("jobs") or []:
            jid = str(job.get("job_id") or "").strip()
            if not jid:
                continue
            try:
                rec = self.supervisor.inspect(jid)
            except FileNotFoundError:
                rec = dict(job)
            contract = self._load_contract(jid)
            if contract is None:
                contract = {
                    "schema": CONTRACT_SCHEMA,
                    "job_id": jid,
                    "unit_id": rec.get("workunit_id"),
                    "path": rec.get("expected_receipt_path"),
                    "required_schema": None,
                    "recovered_without_persisted_contract": True,
                    "absent_is": PROCESS_FAILED,
                }
            handle = _handle_from_record(
                rec,
                contract,
                launch_refused=bool(rec.get("launch_refused")),
                unit={"id": rec.get("workunit_id") or jid, "dependencies": []},
            )
            handle["rediscover_fate"] = job.get("fate")
            handle["relaunched"] = False
            self._write_handle(handle)
            handles.append(handle)
        return {
            "result": report.get("result"),
            "fates": report.get("fates"),
            "handles": handles,
            "n_handles": len(handles),
            "relaunched": False,
            "assumed_complete": False,
            "jobs": report.get("jobs"),
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
        }

    def record_block(self, view: Mapping[str, Any]) -> dict[str, Any]:
        """Persist a whole-resident BLOCKED event. Called when the condition fires."""
        event = {
            "schema": BLOCK_SCHEMA,
            "status": view.get("status"),
            "named_dependency": view.get("named_dependency"),
            "blocked_on": list(view.get("blocked_on") or []),
            "why": view.get("why"),
            "source": view.get("source"),
            "recorded_at": time.time(),
            "spin": False,
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
        }
        self.block_events.append(event)
        stamp = str(event["recorded_at"]).replace(".", "-")
        atomic_write_json(self.blocks_root / f"block-{stamp}.json", event)
        return event

    def reap_all(self) -> None:
        self.supervisor.reap_all()


def _handle_from_record(
    rec: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    launch_refused: bool,
    unit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": HANDLE_SCHEMA,
        "job_id": rec.get("job_id"),
        "unit_id": rec.get("workunit_id") or _unit_id_of(unit),
        "launched_at": rec.get("started_at") or rec.get("running_at") or time.time(),
        "pid": rec.get("pid"),
        "start_token": rec.get("start_token"),
        "expected_receipt_path": contract.get("path") or rec.get("expected_receipt_path"),
        "required_schema": contract.get("required_schema"),
        "dependencies": _deps_of(unit),
        "frontier_item": unit.get("frontier_item") or unit.get("frontier_id"),
        "state": rec.get("state"),
        "terminal": rec.get("terminal"),
        "launch_refused": bool(launch_refused),
        "timeout_s": rec.get("timeout_s"),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


def _classify_ingest(terminal: Any, receipt_present: bool, state: Any) -> str:
    if state == "missing":
        return PROCESS_FAILED
    if terminal is None:
        if state in {"SLEEPING"}:
            return PROCESS_FAILED
        return OPEN
    if terminal == "cancelled":
        return CANCELLED
    if terminal == "completed-with-receipt" and receipt_present:
        return INGESTED
    # timed_out / crashed / completed-without-receipt / unknown / claimed-receipt-missing
    return PROCESS_FAILED


def _process_failed_reason(snap: Mapping[str, Any]) -> str:
    terminal = snap.get("terminal")
    if snap.get("state") == "missing":
        return "supervision record absent; never silently forgotten"
    if terminal == "timed_out":
        return "timeout elapsed; expected receipt never appeared"
    if terminal == "completed-without-receipt":
        return "process exited without writing the expected receipt"
    if terminal == "completed-with-receipt" and not snap.get("receipt_present"):
        return "terminal claimed a receipt that is not on disk"
    if terminal == "crashed":
        return str(snap.get("crash_reason") or "process crashed")
    if terminal == "unknown":
        return str(snap.get("crash_reason") or "fate unknown; not assumed complete")
    if snap.get("state") == "SLEEPING":
        return str(snap.get("crash_reason") or "launch refused; unit SLEEPS")
    return f"terminal {terminal!r} is not a landed receipt"


# ---------------------------------------------------------------------------
# runnable_now — computed from the graph / frontier, never assumed
# ---------------------------------------------------------------------------


def _frontier_candidates(lanes: Iterable[str] | str | None) -> tuple[list[dict[str, Any]], str]:
    from tools.future import frontiers as fr

    available = lanes if lanes is not None else fr.THIS_HOST_LANES
    try:
        units = list(fr.next_work(available) or [])
    except Exception as exc:
        raise SchedulerError(
            f"frontier next_work failed: {type(exc).__name__}: {exc}",
            fault="frontier_unreadable",
        ) from exc
    return units, "frontiers.next_work"


def _graph_independent(graph: Any, blocked_ids: set[str]) -> list[dict[str, Any]]:
    """Ready set with in-flight units treated as running. Restores graph status."""
    units = getattr(graph, "units", None)
    if not isinstance(units, dict):
        raise SchedulerError(
            "graph has no units mapping; refusing to guess a ready set",
            fault="graph_unreadable",
        )
    saved: dict[str, Any] = {}
    for uid in list(blocked_ids):
        if uid in units:
            saved[uid] = units[uid].get("status")
            units[uid]["status"] = "running"
    try:
        ready = graph.compute_ready(mutate=False)
        return [dict(u) for u in ready if _unit_id_of(u) not in blocked_ids]
    finally:
        for uid, status in saved.items():
            units[uid]["status"] = status


def runnable_now(
    blocked: Sequence[Any],
    *,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    graph: Any = None,
    lanes: Iterable[str] | str | None = None,
    scheduler: NoWaitScheduler | None = None,
) -> dict[str, Any]:
    """Independent work that can proceed while `blocked` handles are open.

    BLOCKED fires only when nothing independent remains AND a named in-flight
    dependency is the reason. That is the only whole-resident wait. It is
    recorded when it fires. It never spins.
    """
    blocked_rows = [_as_handle(h) for h in (blocked or [])]
    blocked_ids = _blocked_ids(blocked_rows)
    source = "supplied_candidates"
    pool: list[dict[str, Any]] = []

    if graph is not None:
        try:
            pool = _graph_independent(graph, blocked_ids)
            source = "workgraph.compute_ready"
        except SchedulerError as exc:
            view = {
                "status": REFUSED,
                "runnable": [],
                "blocked_on": [r.get("job_id") for r in blocked_rows],
                "named_dependency": None,
                "why": exc.reason,
                "source": "workgraph",
                "spin": False,
            }
            if scheduler is not None:
                scheduler.record_block(view)
            return view
    elif candidates is not None:
        pool = [dict(c) for c in candidates]
        source = "supplied_candidates"
    else:
        try:
            pool, source = _frontier_candidates(lanes)
        except SchedulerError as exc:
            return {
                "status": REFUSED,
                "runnable": [],
                "blocked_on": [r.get("job_id") for r in blocked_rows],
                "named_dependency": None,
                "why": exc.reason,
                "source": "frontiers.next_work",
                "spin": False,
                "evidence_class": "STATIC_ONLY",
            }

    independent: list[dict[str, Any]] = []
    waiting: list[dict[str, Any]] = []
    for raw in pool:
        uid = _unit_id_of(raw)
        if uid and uid in blocked_ids:
            continue
        deps = _deps_of(raw)
        hit = [d for d in deps if d in blocked_ids]
        if hit:
            waiting.append({"id": uid, "blocked_on": hit})
            continue
        independent.append(dict(raw))

    blocked_on = [r.get("job_id") for r in blocked_rows]
    if independent:
        view = {
            "status": RUNNABLE,
            "runnable": independent,
            "blocked_on": blocked_on,
            "named_dependency": None,
            "waiting_on_inflight": waiting,
            "why": (
                f"{len(independent)} independent unit(s) can run while "
                f"{len(blocked_rows)} handle(s) stay open"
            ),
            "source": source,
            "spin": False,
            "evidence_class": "STATIC_ONLY",
        }
        return view

    if blocked_rows:
        named = None
        if waiting:
            named = waiting[0]["blocked_on"][0]
        else:
            named = (
                blocked_rows[0].get("unit_id")
                or blocked_rows[0].get("job_id")
            )
        view = {
            "status": BLOCKED,
            "runnable": [],
            "blocked_on": blocked_on,
            "named_dependency": named,
            "waiting_on_inflight": waiting,
            "why": (
                "no independent work available; the exact dependency "
                f"{named!r} requires the detached result"
            ),
            "source": source,
            "spin": False,
            "whole_resident_wait": True,
            "evidence_class": "STATIC_ONLY",
        }
        if scheduler is not None:
            scheduler.record_block(view)
        return view

    return {
        "status": IDLE,
        "runnable": [],
        "blocked_on": [],
        "named_dependency": None,
        "why": "nothing in flight and nothing to run",
        "source": source,
        "spin": False,
        "evidence_class": "STATIC_ONLY",
    }


# Module-level API the lane names. Workspace is required so a caller cannot
# accidentally detach into the live campaign working tree.


def launch_detached(
    unit: Any,
    *,
    workspace: str | _os.PathLike[str],
    env: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    return NoWaitScheduler(workspace).launch_detached(unit, env=env)


def poll(
    handles: Sequence[Any],
    *,
    workspace: str | _os.PathLike[str],
) -> list[dict[str, Any]]:
    return NoWaitScheduler(workspace).poll(handles)


def ingest_ready(
    handles: Sequence[Any],
    *,
    workspace: str | _os.PathLike[str],
) -> dict[str, Any]:
    return NoWaitScheduler(workspace).ingest_ready(handles)


def cancel(
    handle: Any,
    *,
    workspace: str | _os.PathLike[str],
) -> dict[str, Any]:
    return NoWaitScheduler(workspace).cancel(handle)


def rediscover(*, workspace: str | _os.PathLike[str]) -> dict[str, Any]:
    return NoWaitScheduler(workspace).rediscover()


# ---------------------------------------------------------------------------
# Proofs. Real processes, real timestamps. Intent is not evidence.
# ---------------------------------------------------------------------------


def _cpu_unit(
    name: str,
    dash_c: str,
    receipt: Path,
    *,
    extra_args: Sequence[str] = (),
    sleep_s: float | None = None,
    timeout_s: float = 20.0,
    dependencies: Sequence[str] = (),
    **over: Any,
) -> dict[str, Any]:
    body = dash_c
    if sleep_s is not None:
        body = (
            "import sys,time; from pathlib import Path; "
            f"time.sleep({float(sleep_s)}); "
            "Path(sys.argv[1]).write_text('{\"ok\": true}\\n')"
        )
    row: dict[str, Any] = {
        "id": name,
        "role": "science",
        "description": f"no-wait unit {name}",
        "command": [sys.executable, "-c", body, str(receipt), *list(extra_args)],
        "resource_class": "LIGHT_CONTROL",
        "output_receipt_path": str(receipt),
        "verifier": "future.no_wait_scheduler.ingest_ready",
        "classification": "STATIC_ONLY",
        "timeout_s": timeout_s,
        "dependencies": list(dependencies),
        "gpu_authority": False,
    }
    row.update(over)
    return row


def prove_overlap_interval(workspace: Path) -> dict[str, Any]:
    """One unit detached and unfinished WHILE another starts, progresses, completes.

    Overlap is read from timestamps, not from the intent to overlap.
    """
    sched = NoWaitScheduler(workspace)
    slow_path = workspace / "results" / "slow.json"
    fast_path = workspace / "results" / "fast.json"
    started_path = fast_path.with_suffix(".started")
    slow_path.parent.mkdir(parents=True, exist_ok=True)
    slow = _cpu_unit("slow-standin", "", slow_path, sleep_s=5.0, timeout_s=30.0)
    fast = _cpu_unit(
        "independent-cpu",
        "import sys,time; from pathlib import Path; "
        "p=Path(sys.argv[1]); p.with_suffix('.started').write_text(str(time.time())); "
        "time.sleep(0.25); p.write_text('{\"ok\": true}\\n')",
        fast_path,
        timeout_s=15.0,
        dependencies=(),
    )
    h_slow = sched.launch_detached(slow)
    t_slow = float(h_slow.get("launched_at") or time.time())
    view = sched.runnable_now(
        [h_slow],
        candidates=[slow, fast],
    )
    independent_named = [_unit_id_of(u) for u in view.get("runnable") or []]
    h_fast = sched.launch_detached(fast)
    t_fast_launch = float(h_fast.get("launched_at") or time.time())

    observed_progress = False
    t_progress = None
    t_fast_finish = None
    slow_open_at_progress = None
    slow_open_at_finish = None
    deadline = time.monotonic() + 12.0
    last_slow_terminal = None
    last_fast_terminal = None
    while time.monotonic() < deadline:
        snaps = sched.poll([h_slow, h_fast])
        by_id = {s["job_id"]: s for s in snaps}
        slow_snap = by_id[h_slow["job_id"]]
        fast_snap = by_id[h_fast["job_id"]]
        last_slow_terminal = slow_snap.get("terminal")
        last_fast_terminal = fast_snap.get("terminal")
        if started_path.is_file() and not observed_progress:
            observed_progress = True
            t_progress = time.time()
            slow_open_at_progress = slow_snap.get("terminal") is None
        if fast_snap.get("terminal") is not None and t_fast_finish is None:
            t_fast_finish = float(fast_snap.get("finished_at") or time.time())
            slow_open_at_finish = slow_snap.get("terminal") is None
            if slow_open_at_finish and observed_progress:
                break
        if last_slow_terminal is not None and last_fast_terminal is not None:
            break
        time.sleep(0.05)

    # The overlap facts are already captured above. Ingestion is a separate
    # condition and it was racing a 0.05s poll against a real subprocess: on a
    # loaded machine the fast job was terminal but its receipt had not been
    # polled yet, so a correct overlap reported passed=False about two runs in
    # three. Give ingestion a bounded chance instead of failing the proof for a
    # scheduling accident. The claim being proven does not change.
    fast_ingest = None
    ingest_deadline = time.time() + 5.0
    while time.time() < ingest_deadline:
        landed = sched.ingest_ready([h_slow, h_fast])
        fast_rows = [r for r in landed["landed"] if r["job_id"] == h_fast["job_id"]]
        if fast_rows:
            fast_ingest = fast_rows[0]["ingest"]
            if fast_ingest == INGESTED:
                break
        time.sleep(0.05)
    try:
        sched.cancel(h_slow)
    except (SchedulerError, DetachedError, FileNotFoundError, OSError):
        sched.reap_all()

    overlap = bool(
        observed_progress
        and slow_open_at_progress
        and t_fast_finish is not None
        and slow_open_at_finish
        and t_slow <= t_fast_launch <= t_fast_finish
        and fast_ingest == INGESTED
        and view.get("status") == RUNNABLE
        and any("independent-cpu" in str(n) for n in independent_named)
    )
    return {
        "passed": overlap,
        "slow_launched_at": t_slow,
        "fast_launched_at": t_fast_launch,
        "fast_progress_at": t_progress,
        "fast_finished_at": t_fast_finish,
        "slow_open_at_fast_progress": slow_open_at_progress,
        "slow_open_at_fast_finish": slow_open_at_finish,
        "slow_terminal_during_window": last_slow_terminal,
        "fast_terminal": last_fast_terminal,
        "fast_ingest": fast_ingest,
        "runnable_status_while_slow_open": view.get("status"),
        "independent_named": independent_named,
        "started_marker_present": started_path.is_file(),
        "fast_receipt_present": fast_path.is_file(),
        "clock": "time.time() unix seconds; overlap is an interval, not an intent",
    }


def prove_cancel_stops(workspace: Path) -> dict[str, Any]:
    sched = NoWaitScheduler(workspace)
    receipt = workspace / "results" / "cancel.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    unit = _cpu_unit("cancel-me", "", receipt, sleep_s=20.0, timeout_s=60.0)
    handle = sched.launch_detached(unit)
    deadline = time.monotonic() + 2.0
    pid = handle.get("pid")
    while time.monotonic() < deadline and not (isinstance(pid, int) and pid > 0):
        snaps = sched.poll([handle])
        pid = snaps[0].get("pid") if snaps else None
        time.sleep(0.02)
    cancelled = sched.cancel(handle)
    gone = bool(cancelled.get("process_gone"))
    still = bool(isinstance(pid, int) and pid > 0 and pid_is_alive(pid))
    passed = cancelled.get("terminal") == "cancelled" and gone and not still
    return {
        "passed": passed,
        "terminal": cancelled.get("terminal"),
        "process_gone": gone,
        "pid_still_alive": still,
        "pid": pid,
    }


def prove_blocked_names_dependency(workspace: Path) -> dict[str, Any]:
    from tools.future import workgraph as wg

    sched = NoWaitScheduler(workspace)
    graph = wg.WorkGraph(workspace=workspace, ncpu=2)
    graph.admit(
        wg.make_unit(
            id="WU.PRE",
            role="science",
            description="in-flight predecessor",
            dependencies=[],
            resource_lane="CPU_ANALYSIS",
            mutation_scope=[],
            verifier="future.no_wait.pre",
            expected_information_gain=2,
            cost_units=1,
            requires_hardware=False,
        )
    )
    graph.admit(
        wg.make_unit(
            id="WU.DEP",
            role="science",
            description="depends on the in-flight unit",
            dependencies=["WU.PRE"],
            resource_lane="CPU_ANALYSIS",
            mutation_scope=[],
            verifier="future.no_wait.dep",
            expected_information_gain=2,
            cost_units=1,
            requires_hardware=False,
        )
    )
    receipt = workspace / "results" / "pre.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    pre = _cpu_unit(
        "WU.PRE", "", receipt, sleep_s=8.0, timeout_s=30.0, dependencies=()
    )
    handle = sched.launch_detached(pre)
    view = sched.runnable_now([handle], graph=graph)
    try:
        passed = (
            view.get("status") == BLOCKED
            and view.get("named_dependency") == "WU.PRE"
            and view.get("spin") is False
            and view.get("runnable") == []
            and bool(sched.block_events)
        )
        return {
            "passed": passed,
            "status": view.get("status"),
            "named_dependency": view.get("named_dependency"),
            "spin": view.get("spin"),
            "why": view.get("why"),
            "n_block_events": len(sched.block_events),
            "source": view.get("source"),
        }
    finally:
        try:
            sched.cancel(handle)
        except (SchedulerError, DetachedError, FileNotFoundError, OSError):
            sched.reap_all()


def prove_missing_receipt_is_process_failed(workspace: Path) -> dict[str, Any]:
    sched = NoWaitScheduler(workspace)
    expected = workspace / "results" / "never.json"
    expected.parent.mkdir(parents=True, exist_ok=True)
    unit = {
        "id": "never-lands",
        "role": "science",
        "description": "sleeps past timeout and never writes the receipt",
        "command": ["/bin/sleep", "20"],
        "resource_class": "LIGHT_CONTROL",
        "output_receipt_path": str(expected),
        "verifier": "future.no_wait_scheduler.ingest_ready",
        "classification": "STATIC_ONLY",
        "timeout_s": 0.4,
        "dependencies": [],
    }
    handle = sched.launch_detached(unit)
    deadline = time.monotonic() + 6.0
    terminal = None
    while time.monotonic() < deadline:
        snaps = sched.poll([handle])
        terminal = snaps[0].get("terminal") if snaps else None
        if terminal is not None:
            break
        time.sleep(0.05)
    landed = sched.ingest_ready([handle])
    rows = landed["landed"]
    forgotten = landed.get("forgotten") or []
    ingest = rows[0]["ingest"] if rows else None
    present = expected.is_file()
    rediscovered = sched.rediscover()
    still_known = any(
        str(h.get("job_id")) == str(handle["job_id"])
        for h in rediscovered.get("handles") or []
    )
    passed = (
        ingest == PROCESS_FAILED
        and not present
        and not forgotten
        and still_known
        and terminal in {"timed_out", "crashed", "completed-without-receipt", "unknown"}
    )
    return {
        "passed": passed,
        "ingest": ingest,
        "terminal": terminal,
        "receipt_present": present,
        "forgotten": forgotten,
        "still_known_after_rediscover": still_known,
        "n_landed": landed["n_landed"],
        "reason": rows[0].get("reason") if rows else None,
    }


def prove_rediscover_adopts(workspace: Path) -> dict[str, Any]:
    first = NoWaitScheduler(workspace)
    receipt = workspace / "results" / "adopt.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    unit = _cpu_unit("adopt-me", "", receipt, sleep_s=12.0, timeout_s=40.0)
    handle = first.launch_detached(unit)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not handle.get("start_token"):
        snaps = first.poll([handle])
        if snaps:
            handle = {**handle, "pid": snaps[0].get("pid"), "start_token": None}
            rec = first.supervisor.inspect(handle["job_id"])
            handle["start_token"] = rec.get("start_token")
            handle["pid"] = rec.get("pid")
        time.sleep(0.02)
    second = NoWaitScheduler(workspace)
    report = second.rediscover()
    mine = [
        h for h in report.get("handles") or [] if h.get("job_id") == handle["job_id"]
    ]
    fate = None
    adopted_open = False
    if mine:
        fate = mine[0].get("rediscover_fate")
        adopted_open = mine[0].get("terminal") is None
    try:
        pid = handle.get("pid")
        live = bool(isinstance(pid, int) and pid_is_alive(pid))
        passed = bool(mine) and report.get("relaunched") is False and (
            fate == "ADOPTED" or (adopted_open and live)
        )
        return {
            "passed": passed,
            "fate": fate,
            "relaunched": report.get("relaunched"),
            "n_handles": report.get("n_handles"),
            "child_still_alive": live,
            "adopted_open": adopted_open,
        }
    finally:
        try:
            second.cancel(handle)
        except (SchedulerError, DetachedError, FileNotFoundError, OSError):
            second.reap_all()


def run_all_proofs() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="hawking-no-wait-") as td:
        root = Path(td)
        proofs = {
            "overlap_interval": prove_overlap_interval(root / "overlap"),
            "cancel_stops": prove_cancel_stops(root / "cancel"),
            "blocked_names_dependency": prove_blocked_names_dependency(root / "blocked"),
            "missing_receipt_is_process_failed": prove_missing_receipt_is_process_failed(
                root / "missing"
            ),
            "rediscover_adopts": prove_rediscover_adopts(root / "rediscover"),
        }
        names = sorted(proofs)
        failed = [name for name in names if not proofs[name].get("passed")]
        return {
            "proofs": proofs,
            "failed": failed,
            "all_passed": not failed,
            "n_proofs": len(names),
            "n_passed": len(names) - len(failed),
        }


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


RECOVERED_IMPLEMENTATION = [
    "tools/future/detached.py DetachedSupervisor.launch/inspect/cancel/adopt/list "
    "— spawn, identity (pid+start_token), timeout, cancel; this scheduler does not reimplement them",
    "tools/future/detached.py _expected_receipt_of / _as_mapping — landing path contract, not a second one",
    "tools/future/wakeup.py Watcher.register_expectation — receipt appearance is the completion event; "
    "Watcher refuses model-facing poll/wait aliases and that refusal is kept",
    "tools/future/workgraph.py WorkGraph.compute_ready — independence from dependency edges, not a hand order",
    "tools/future/frontiers.py next_work / THIS_HOST_LANES — runnable set when no graph is supplied",
    "tools/future/restart_supervisor.py rediscover_detached — ingest finished, adopt live, UNKNOWN otherwise; never relaunch",
    "tools/future/autonomy_run.py _invoke_bounded — the defect: subprocess.run blocks the loop; cited, not copied",
    "hcli/agentos/background.py start_new_session + persist + SIGTERM then SIGKILL",
    "hcli/resources.py pid_is_alive / process_start_token — pid alone is never identity",
    "hcli/persist.py atomic_write_json — handle and contract durability",
]

GAPS_CLOSED = [
    "launch_detached returns a handle without waiting; the expected receipt contract is on disk before the caller resumes",
    "runnable_now computes independent work from the graph/frontier while handles are open",
    "an overlap interval was observed: one detached unit still open while another independent unit started, progressed and completed (timestamps, not intent)",
    "the whole resident reports BLOCKED with the dependency named when nothing independent remains; it does not spin",
    "ingest_ready classifies a missing/never-written receipt as process_failed and does not drop the handle",
    "cancel proves the process is gone; a live pid after cancel is a refusal",
    "rediscover adopts handles that outlived a resident restart and never relaunches",
]

NEGATIVE_FINDINGS = [
    "autonomy_run.py still calls subprocess.run; this lane is not allowed to edit it, so the live loop is not yet wired to the scheduler",
    "wakeup.Watcher.poll remains a refused model-facing alias; supervisor poll lives here and is a single inspect pass",
    "workgraph.py still does not execute; this scheduler launches via detached.py and does not grow a second executor",
    "GPU/lease/cargo units still SLEEP in detached.py; this scheduler does not seize a lease to keep the resident busy",
    "a frontier that cannot be loaded is REFUSED, not replaced with a guessed candidate list",
]


def emit_resident_workunit() -> dict[str, Any]:
    unit = WorkUnit(
        id="future.no_wait_scheduler.overlap",
        role="science",
        description=(
            "Launch detached work, persist the expected receipt contract, "
            "release the reasoning context, ask what else can run now, "
            "execute an independent WorkUnit, ingest when it lands."
        ),
        resource_class="LIGHT_CONTROL",
        provider="future.no_wait_scheduler",
        verifier="future.no_wait_scheduler.overlap_interval",
        effect_class="READ_ONLY",
        workspace="repo-root",
        classification="STATIC_ONLY",
        status="pending",
    )
    row = unit.to_dict()
    row.update(
        {
            "command": [sys.executable, str(Path(__file__).resolve()), "--build"],
            "output_receipt_path": str(RECEIPTS / RECEIPT),
            "claim_boundary": CLAIM_BOUNDARY,
            "species": "independent_reproduction",
            "requires_quiescence": False,
            "may_promote": False,
            "may_modify_verifier": False,
            "gpu_authority": False,
            "timeout_s": 60.0,
        }
    )
    WorkUnit.from_dict(dict(row))
    return row


def build() -> Path:
    proofs = run_all_proofs()
    if not proofs["all_passed"]:
        raise SchedulerError(
            "negative-control proofs failed: " + ", ".join(proofs["failed"]),
            fault="proof_failed",
        )
    unit = emit_resident_workunit()
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Keep the resident moving while a detached unit is unfinished: "
            "launch, persist the landing contract, release, ask what else can "
            "run, execute independent work, ingest in landing order."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "module_sha256": sha256_file(Path(__file__)),
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "claim_boundary": CLAIM_BOUNDARY,
        "supervisor_poll_rule": SUPERVISOR_POLL_RULE,
        "does_not_call_subprocess_run_for_work": True,
        "blocks_whole_resident_only_when": (
            "no independent runnable work AND the exact named dependency "
            "requires the in-flight result; computed, then recorded"
        ),
        "api": {
            "launch_detached": "start, persist contract, return handle, do not wait",
            "expected_receipt": "landing contract; absence is process_failed",
            "poll": "supervisor-side, one inspect per handle, never a busy loop",
            "runnable_now": "independent work from graph/frontier while handles are open",
            "ingest_ready": "landed rows in finished_at order; missing receipt is process_failed",
            "cancel": "signal and prove the process is gone",
            "rediscover": "adopt outlived handles; never relaunch",
        },
        "proofs": proofs["proofs"],
        "proofs_all_passed": proofs["all_passed"],
        "overlap_is_an_interval": True,
        "recovered_implementation": RECOVERED_IMPLEMENTATION,
        "gaps_closed": GAPS_CLOSED,
        "negative_findings": NEGATIVE_FINDINGS,
        "resident_callable": {
            "entry_point": "tools.future.no_wait_scheduler.launch_detached(unit, workspace=...)",
            "workunit": unit["id"],
            "workunit_row": unit,
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.HCLI_SELF.no-launch",
            "fails_closed": (
                "absent unit/workspace/handle refuses; unsafe commands raise "
                "UnsafeCommandError; missing receipt is process_failed; cancel "
                "of a still-live pid raises; rediscover never relaunches; "
                "BLOCKED names the dependency and does not spin"
            ),
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

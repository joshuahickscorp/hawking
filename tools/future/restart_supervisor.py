"""RESIDENT RESTART SUPERVISOR — make a restart survivable and provable.

The mission is durable. The resident process is not, and must not be treated as
sacred: pathological state is checkpoint, kill, restart — not continuity theatre
around a broken process.

The resident may REQUEST a restart. Only the supervisor DECIDES and performs it.
A RestartRequest carries a reason and cannot execute anything; SupervisorDecision
is not user-constructible. Passing a request to restart_cycle is a TypeError.

This sidecar never starts a model, never takes a GPU/bench lease, and never
resolves an artifact at runtime. restart() execs the exact path+sha256 the
checkpoint named. rediscover_detached() ingests finished jobs, re-adopts live
ones, reports UNKNOWN when fate cannot be proven, and never relaunches.

    python3 tools/future/restart_supervisor.py --build
    python3 -m pytest tools/future/test_restart_supervisor.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from hcli.resources import pid_is_alive, process_start_token
from hcli.workunit import WorkUnit
from tools.future._common import RECEIPTS, REPO, git, sha256_file, write_receipt
from tools.future.detached import (
    DetachedError,
    DetachedSupervisor,
    identity_status,
    refuse_reason,
)
from tools.future.wakeup import (
    COMPLETED,
    PARTIAL_TRUNCATED,
    SEAL_MISMATCH,
    classify_receipt_bytes,
    write_sealed,
)

RECEIPT = "RESTART_SUPERVISOR.json"
SCHEMA = "hawking.future.restart_supervisor.v1"
CHECKPOINT_SCHEMA = "hawking.future.restart_supervisor.checkpoint.v1"
RESTORED_SCHEMA = "hawking.future.restart_supervisor.restored.v1"
RECORDED_BY = "tools/future/restart_supervisor.py"
VERSION = 1

CHECKPOINT_NAME = "RESTART_CHECKPOINT.json"
RESTORED_NAME = "RESTORED_STATE.json"
HANDLE_NAME = "RESIDENT_HANDLE.json"

# Recovered path. Importing autonomy_run pulls the whole loop; the constant is the
# contract. Content is never invented when the file is absent.
MISSION_STATE = RECEIPTS / "AUTONOMY_MISSION_STATE.json"

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. "
    "A restart verdict is process-identity and disk state, not a bench. "
    "A RestartRequest is not supervisor authority."
)

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

FATES = ("INGESTED", "ADOPTED", "UNKNOWN")
STOP_NEEDED = ("cooperative", "escalated", "already_dead", "refused")
DECISION_ACTIONS = ("RESTART", "REFUSE")
REQUIRED_CHECKPOINT_KEYS = (
    "schema",
    "complete",
    "artifact",
    "mission",
    "frontier",
    "in_flight",
    "queue",
    "detached_handles",
    "scar_consultations",
    "identity",
)

# Names a request must not expose. The absence is the type boundary.
_REQUEST_CANNOT = frozenset(
    {
        "restart",
        "stop",
        "restore",
        "restart_cycle",
        "perform",
        "execute",
        "run",
        "launch",
        "kill",
        "checkpoint",
        "rediscover_detached",
        "decide",
    }
)

# Short on purpose: proofs and tests must not wait a production grace.
DEFAULT_GRACE_S = 0.4
SPAWN_WAIT_S = 1.0

# os.kill(pid, 0) is true for zombies. Children we spawned must be waitpid'd
# after death or stop() will think SIGTERM failed and escalate a corpse.
_POPEN: dict[int, subprocess.Popen[Any]] = {}

_RESIDENT_SCRIPT = (
    "import pathlib, sys, time\n"
    "marker = pathlib.Path(sys.argv[1])\n"
    "marker.write_text('alive')\n"
    "stop = pathlib.Path(str(marker) + '.stop')\n"
    "while not stop.exists():\n"
    "    time.sleep(0.05)\n"
    "marker.write_text('stopped')\n"
)
_IGNORING_SCRIPT = (
    "import pathlib, signal, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "pathlib.Path(sys.argv[1]).write_text('alive')\n"
    "time.sleep(60)\n"
)


class RestartRefused(RuntimeError):
    """Operational refusal with a reason. Never a success-shaped default."""

    def __init__(self, reason: str, *, fault: str = "refused") -> None:
        self.reason = reason
        self.fault = fault
        super().__init__(f"REFUSED [{fault}]: {reason}")


class CheckpointCorrupt(RestartRefused):
    """Truncated, unsealed, or incomplete checkpoint. Do not half-read."""

    def __init__(self, reason: str, *, fault: str = "checkpoint_corrupt") -> None:
        super().__init__(reason, fault=fault)


class ArtifactMissing(RestartRefused):
    """The checkpoint named an artifact that is gone or no longer those bytes."""

    def __init__(self, reason: str, *, fault: str = "artifact_missing") -> None:
        super().__init__(reason, fault=fault)


class RestartAuthorityError(TypeError):
    """A request is not supervisor authority. The type, not a comment, refuses."""


# ---------------------------------------------------------------------------
# Authority boundary. Request cannot execute. Decision is issued, not constructed.
# ---------------------------------------------------------------------------


class RestartRequest:
    """Resident-issued. A reason, not an action. Has no restart path."""

    __slots__ = ("reason", "pathology", "requested_at", "requester")

    def __init__(
        self,
        reason: str,
        *,
        pathology: str = "",
        requester: str = "resident",
        requested_at: float | None = None,
    ) -> None:
        self.reason = str(reason or "")
        self.pathology = str(pathology or "")
        self.requester = str(requester or "resident")
        self.requested_at = time.time() if requested_at is None else float(requested_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "RestartRequest",
            "reason": self.reason,
            "pathology": self.pathology,
            "requester": self.requester,
            "requested_at": self.requested_at,
            "executable": False,
        }

    def __getattr__(self, name: str) -> Any:
        if name in _REQUEST_CANNOT or name.startswith("do_"):
            raise RestartAuthorityError(
                f"RestartRequest cannot {name}; a request is not supervisor authority"
            )
        raise AttributeError(name)

    def __repr__(self) -> str:
        return f"RestartRequest(reason={self.reason!r}, pathology={self.pathology!r})"


class SupervisorDecision:
    """Issued only by decide(). User construction is a type error."""

    __slots__ = ("request", "action", "reason")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RestartAuthorityError(
            "SupervisorDecision is not user-constructible; only decide() issues it"
        )

    @classmethod
    def _issue(
        cls, request: RestartRequest, action: str, reason: str
    ) -> SupervisorDecision:
        obj = object.__new__(cls)
        object.__setattr__(obj, "request", request)
        object.__setattr__(obj, "action", action)
        object.__setattr__(obj, "reason", reason)
        return obj

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "SupervisorDecision",
            "action": self.action,
            "reason": self.reason,
            "request": self.request.to_dict(),
        }


def request_restart(
    reason: str,
    *,
    pathology: str = "",
    requester: str = "resident",
) -> RestartRequest:
    """The only thing a resident may do: file a reason. Does not restart."""
    return RestartRequest(reason, pathology=pathology, requester=requester)


def decide(request: RestartRequest) -> SupervisorDecision:
    """Supervisor judgement. Empty reason is REFUSE. Never executes a restart."""
    if not isinstance(request, RestartRequest):
        raise RestartAuthorityError(
            f"decide() takes RestartRequest, got {type(request).__name__}"
        )
    if isinstance(request, SupervisorDecision):
        raise RestartAuthorityError("decide() does not re-wrap a decision")
    reason = request.reason.strip()
    if not reason:
        return SupervisorDecision._issue(
            request, "REFUSE", "empty reason is not grounds for a restart"
        )
    return SupervisorDecision._issue(
        request,
        "RESTART",
        f"supervisor accepts request: {reason}",
    )


# ---------------------------------------------------------------------------
# IO helpers. Fail closed. Never half-read a checkpoint.
# ---------------------------------------------------------------------------


def _as_workspace(workspace: str | os.PathLike[str] | None) -> Path:
    if workspace is None:
        raise RestartRefused(
            "workspace is required; refusing to write a checkpoint into the live campaign",
            fault="workspace_required",
        )
    path = Path(workspace).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _truncation_siblings(path: Path) -> list[str]:
    """atomic_write leaves .name.pid.uuid.tmp on a crash mid-write."""
    parent = path.parent
    if not parent.is_dir():
        return []
    hits: list[str] = []
    prefix = f".{path.name}."
    try:
        names = list(parent.iterdir())
    except OSError:
        return []
    for child in names:
        n = child.name
        if n.startswith(prefix) and n.endswith(".tmp"):
            hits.append(n)
        if n in {path.name + ".partial", path.name + ".tmp", "." + path.name + ".tmp"}:
            hits.append(n)
    return hits


def _read_bytes(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY)
    with os.fdopen(fd, "rb") as fh:
        return fh.read()


def load_checkpoint(path: str | os.PathLike[str] | Mapping[str, Any]) -> dict[str, Any]:
    """Load a sealed complete checkpoint, or refuse. Never returns a half object."""
    if isinstance(path, Mapping):
        doc = dict(path)
        if doc.get("schema") != CHECKPOINT_SCHEMA:
            raise CheckpointCorrupt(
                f"in-memory checkpoint schema {doc.get('schema')!r} != {CHECKPOINT_SCHEMA}"
            )
        if doc.get("complete") is not True:
            raise CheckpointCorrupt("in-memory checkpoint complete is not True")
        missing = [k for k in REQUIRED_CHECKPOINT_KEYS if k not in doc]
        if missing:
            raise CheckpointCorrupt(f"in-memory checkpoint missing keys: {missing}")
        return doc

    target = Path(path)
    siblings = _truncation_siblings(target)
    if not target.is_file():
        if siblings:
            raise CheckpointCorrupt(
                f"checkpoint missing at {target}; leftover tmp {siblings!r} "
                "is a mid-write; refusing to half-read",
                fault="truncated_mid_write",
            )
        raise CheckpointCorrupt(f"checkpoint not on disk: {target}", fault="checkpoint_missing")
    if siblings:
        # Live file plus a tmp is ambiguous: the tmp may be a newer torn write.
        raise CheckpointCorrupt(
            f"checkpoint {target.name} has truncation siblings {siblings!r}; refusing",
            fault="truncated_mid_write",
        )
    try:
        raw = _read_bytes(target)
    except OSError as exc:
        raise CheckpointCorrupt(f"checkpoint unreadable: {type(exc).__name__}: {exc}") from exc

    state, why = classify_receipt_bytes(raw, required_schema=CHECKPOINT_SCHEMA)
    if state == PARTIAL_TRUNCATED:
        raise CheckpointCorrupt(f"truncated checkpoint: {why}", fault="truncated_mid_write")
    if state == SEAL_MISMATCH:
        raise CheckpointCorrupt(f"checkpoint seal mismatch: {why}", fault="seal_mismatch")
    if state != COMPLETED:
        raise CheckpointCorrupt(f"checkpoint not complete ({state}): {why}")

    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointCorrupt(f"checkpoint parse failed after classify: {exc}") from exc
    if not isinstance(doc, dict):
        raise CheckpointCorrupt("checkpoint is not a JSON object")
    if doc.get("complete") is not True:
        raise CheckpointCorrupt("checkpoint complete is not True; refusing to half-read")
    missing = [k for k in REQUIRED_CHECKPOINT_KEYS if k not in doc]
    if missing:
        raise CheckpointCorrupt(f"checkpoint missing keys: {missing}")
    return doc


def _artifact_spec(artifact: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        raise RestartRefused(
            "artifact is required; refusing to resolve a resident binary at runtime",
            fault="artifact_required",
        )
    path_s = str(artifact.get("path") or "").strip()
    sha = str(artifact.get("sha256") or "").strip().lower()
    argv_raw = artifact.get("argv")
    if not path_s:
        raise RestartRefused("artifact.path is required", fault="artifact_required")
    if not sha or len(sha) != 64:
        raise RestartRefused("artifact.sha256 must be a 64-char hex digest", fault="artifact_required")
    if not isinstance(argv_raw, (list, tuple)) or not argv_raw:
        raise RestartRefused("artifact.argv must be a non-empty list", fault="artifact_required")
    argv = [str(x) for x in argv_raw]
    if any(not tok for tok in argv):
        raise RestartRefused("artifact.argv contains an empty token", fault="artifact_required")
    prog = Path(argv[0])
    if not prog.is_absolute():
        raise RestartRefused(
            "artifact.argv[0] is not absolute; refusing PATH resolution at restart",
            fault="runtime_resolution_refused",
        )
    cwd_raw = artifact.get("cwd")
    return {
        "path": path_s,
        "sha256": sha,
        "argv": argv,
        "cwd": str(cwd_raw) if cwd_raw else None,
    }


def _assert_artifact_present(spec: Mapping[str, Any]) -> Path:
    path = Path(str(spec["path"]))
    if not path.is_file():
        raise ArtifactMissing(f"named artifact no longer exists: {path}")
    live = sha256_file(path)
    if live.lower() != str(spec["sha256"]).lower():
        raise ArtifactMissing(
            f"artifact bytes do not match checkpoint sha256 at {path}; "
            "not the named artifact"
        )
    argv0 = Path(str(spec["argv"][0]))
    if not argv0.is_file() and not argv0.is_symlink():
        # Interpreter or binary named in argv[0] must still be that exact path.
        if not argv0.exists():
            raise ArtifactMissing(f"artifact.argv[0] no longer exists: {argv0}")
    return path


def _as_mission(mission: Any, *, workspace: Path) -> dict[str, Any]:
    if isinstance(mission, Mapping):
        body = dict(mission)
        if not (body.get("mission_id") or body.get("id")):
            raise RestartRefused(
                "mission mapping has no mission_id/id; refusing to invent one",
                fault="mission_required",
            )
        return body
    if mission is None:
        candidates = (workspace / "MISSION_STATE.json", MISSION_STATE)
        for cand in candidates:
            if cand.is_file():
                try:
                    loaded = json.loads(cand.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise RestartRefused(
                        f"mission state unreadable at {cand}: {type(exc).__name__}: {exc}",
                        fault="mission_unreadable",
                    ) from exc
                if not isinstance(loaded, dict):
                    raise RestartRefused(
                        f"mission state is not an object: {cand}", fault="mission_unreadable"
                    )
                loaded.setdefault("mission_id", loaded.get("id") or cand.name)
                loaded["_loaded_from"] = str(cand)
                return loaded
        raise RestartRefused(
            "mission is absent and AUTONOMY_MISSION_STATE.json is not on disk; "
            "refusing to invent a mission",
            fault="mission_required",
        )
    path = Path(mission)
    if not path.is_file():
        raise RestartRefused(f"mission file not on disk: {path}", fault="mission_required")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RestartRefused(
            f"mission file unreadable: {type(exc).__name__}: {exc}",
            fault="mission_unreadable",
        ) from exc
    if not isinstance(loaded, dict):
        raise RestartRefused("mission file is not a JSON object", fault="mission_unreadable")
    if not (loaded.get("mission_id") or loaded.get("id")):
        raise RestartRefused(
            "mission file has no mission_id/id; refusing to invent one",
            fault="mission_required",
        )
    loaded["_loaded_from"] = str(path)
    return loaded


def _identity_snapshot(identity: Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(identity, Mapping):
        blob = json.dumps(dict(identity), sort_keys=True, separators=(",", ":")).encode()
        return {
            "source": "supplied",
            "digest": hashlib.sha256(blob).hexdigest(),
            "keys": sorted(str(k) for k in identity.keys()),
        }
    try:
        from tools.future.resident_identity import load as load_ident

        ident = load_ident()
        blob = json.dumps(ident, sort_keys=True, separators=(",", ":"), default=str).encode()
        return {
            "source": "receipts/future/RESIDENT_IDENTITY.json",
            "digest": hashlib.sha256(blob).hexdigest(),
            "residency_status": ident.get("residency_status"),
            "executable_hash": ident.get("executable_hash"),
        }
    except Exception as exc:
        return {
            "source": "UNAVAILABLE",
            "digest": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _seq_of_ids(rows: Any) -> list[Any]:
    if rows is None:
        return []
    if isinstance(rows, (str, bytes)):
        raise RestartRefused("id sequence must not be a string", fault="bad_sequence")
    if not isinstance(rows, (list, tuple)):
        raise RestartRefused("id sequence must be a list", fault="bad_sequence")
    return list(rows)


def _handles_from(
    detached: DetachedSupervisor | None,
    supplied: Sequence[Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], str]:
    if supplied is not None:
        rows = [dict(h) for h in supplied]
        return rows, "supplied"
    if detached is None:
        return [], "NONE_SUPPLIED"
    try:
        live = detached.list()
    except (OSError, DetachedError) as exc:
        raise RestartRefused(
            f"detached list failed: {type(exc).__name__}: {exc}",
            fault="detached_unreadable",
        ) from exc
    return [dict(h) for h in live], "detached.list"


def _handle_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "job_id": row.get("job_id"),
        "pid": row.get("pid"),
        "start_token": row.get("start_token"),
        "expected_receipt_path": row.get("expected_receipt_path"),
        "state": row.get("state"),
        "terminal": row.get("terminal"),
        "identity_status": row.get("identity_status") or identity_status(row),
        "workunit_id": row.get("workunit_id"),
    }


def _signal_group(pid: int, sig: int) -> str:
    try:
        os.killpg(pid, sig)
        return "killpg"
    except OSError:
        try:
            os.kill(pid, sig)
            return "kill"
        except OSError as exc:
            return f"failed:{type(exc).__name__}"


def _child_dead(pid: int) -> bool:
    """True if the pid is gone, including a zombie we can reap.

    os.kill(pid, 0) is true for zombies, so pid_is_alive cannot be the only
    death test for children we spawned.
    """
    pop = _POPEN.get(pid)
    if pop is not None and pop.poll() is not None:
        _POPEN.pop(pid, None)
        try:
            pop.wait(timeout=0.05)
        except Exception:
            return True
        return True
    try:
        wpid, _status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return not pid_is_alive(pid)
    except OSError:
        return not pid_is_alive(pid)
    if wpid == pid:
        _POPEN.pop(pid, None)
        return True
    return not pid_is_alive(pid)


def _wait_dead(pid: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline:
        if _child_dead(pid):
            return True
        time.sleep(0.02)
    return _child_dead(pid)


def _waitpid(pid: int) -> None:
    pop = _POPEN.pop(pid, None)
    if pop is not None:
        try:
            pop.wait(timeout=0.5)
        except Exception:
            return
        return
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        return


def _reap_quietly(pid: Optional[int]) -> None:
    if not isinstance(pid, int) or pid <= 0:
        return
    if pid_is_alive(pid):
        _signal_group(pid, signal.SIGKILL)
        _wait_dead(pid, 1.0)
    _waitpid(pid)


# ---------------------------------------------------------------------------
# Public operations
# ---------------------------------------------------------------------------


def checkpoint(
    mission: Any,
    *,
    workspace: str | os.PathLike[str],
    artifact: Mapping[str, Any],
    detached: DetachedSupervisor | None = None,
    detached_handles: Sequence[Mapping[str, Any]] | None = None,
    identity: Mapping[str, Any] | None = None,
    frontier: Mapping[str, Any] | Sequence[Any] | None = None,
    queue: Sequence[Any] | None = None,
    in_flight: Sequence[Any] | None = None,
    scar_consultations: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Durable snapshot. Absent required inputs refuse; nothing is invented."""
    root = _as_workspace(workspace)
    spec = _artifact_spec(artifact)
    # Confirm the named bytes exist *now*, so a checkpoint cannot name fiction.
    _assert_artifact_present(spec)
    body_mission = _as_mission(mission, workspace=root)

    if frontier is None:
        fid = body_mission.get("frontier_ids") or body_mission.get("frontier")
        if isinstance(fid, Mapping):
            frontier_snap: dict[str, Any] = {
                "status": "FROM_MISSION",
                "item_ids": list(fid.get("item_ids") or fid.get("ids") or []),
                "items": fid.get("items") if isinstance(fid.get("items"), list) else [],
            }
        elif isinstance(fid, (list, tuple)):
            frontier_snap = {"status": "FROM_MISSION", "item_ids": list(fid), "items": []}
        else:
            frontier_snap = {
                "status": "UNAVAILABLE",
                "item_ids": [],
                "items": [],
                "reason": "no frontier supplied and mission carries none; not guessed from live book",
            }
    elif isinstance(frontier, Mapping):
        frontier_snap = {
            "status": str(frontier.get("status") or "SUPPLIED"),
            "item_ids": list(frontier.get("item_ids") or frontier.get("ids") or []),
            "items": list(frontier.get("items") or []),
        }
    else:
        frontier_snap = {"status": "SUPPLIED", "item_ids": list(frontier), "items": []}

    q = _seq_of_ids(queue if queue is not None else body_mission.get("queue") or [])
    inflight = _seq_of_ids(
        in_flight if in_flight is not None else body_mission.get("units") or body_mission.get("in_flight") or []
    )
    scars = _seq_of_ids(
        scar_consultations
        if scar_consultations is not None
        else body_mission.get("scar_consultations") or []
    )
    handles, handle_source = _handles_from(detached, detached_handles)

    doc = {
        "schema": CHECKPOINT_SCHEMA,
        "version": VERSION,
        "complete": True,
        "purpose": "durable resident-restart snapshot; not a hardware measurement",
        "artifact": spec,
        "mission": {
            "mission_id": body_mission.get("mission_id") or body_mission.get("id"),
            "phase": body_mission.get("phase"),
            "next_action": body_mission.get("next_action"),
            "body": body_mission,
        },
        "frontier": frontier_snap,
        "in_flight": inflight,
        "queue": q,
        "detached_handles": [_handle_snapshot(h) for h in handles],
        "detached_source": handle_source,
        "scar_consultations": scars,
        "identity": _identity_snapshot(identity),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }
    out = root / CHECKPOINT_NAME
    sealed = write_sealed(out, doc)
    sealed["path"] = str(out)
    return sealed


def stop(
    resident: Mapping[str, Any] | None,
    *,
    grace_s: float = DEFAULT_GRACE_S,
) -> dict[str, Any]:
    """Cooperative SIGTERM, then SIGKILL. Records which was needed. Never kills a reused pid."""
    if resident is None:
        raise RestartRefused(
            "resident handle is required; pass an already-dead handle rather than None",
            fault="resident_required",
        )
    record = dict(resident)
    pid_raw = record.get("pid")
    try:
        pid = int(pid_raw) if pid_raw is not None else 0
    except (TypeError, ValueError):
        pid = 0

    if pid > 0 and _child_dead(pid):
        return {
            "result": "ok",
            "stopped": True,
            "needed": "already_dead",
            "identity_status": "dead",
            "pid": pid,
            "signal": None,
            "reason": "resident pid is not alive; nothing signalled",
        }
    status = identity_status(record)
    if status == "dead":
        if pid > 0:
            _waitpid(pid)
        return {
            "result": "ok",
            "stopped": True,
            "needed": "already_dead",
            "identity_status": "dead",
            "pid": pid or record.get("pid"),
            "signal": None,
            "reason": "resident pid is not alive; nothing signalled",
        }
    if status != "match":
        # Pid reuse / missing token: signalling would be a foreign kill.
        return {
            "result": "refused",
            "stopped": False,
            "needed": "refused",
            "identity_status": status,
            "pid": pid or record.get("pid"),
            "signal": None,
            "reason": (
                f"identity_status={status}: will not signal a pid we cannot prove is the resident"
            ),
        }

    how = _signal_group(pid, signal.SIGTERM)
    if _wait_dead(pid, grace_s):
        _waitpid(pid)
        return {
            "result": "ok",
            "stopped": True,
            "needed": "cooperative",
            "identity_status": "dead",
            "pid": pid,
            "signal": "SIGTERM",
            "channel": how,
            "reason": "SIGTERM reaped the resident within grace",
        }
    how_k = _signal_group(pid, signal.SIGKILL)
    if _wait_dead(pid, 1.0):
        _waitpid(pid)
        return {
            "result": "ok",
            "stopped": True,
            "needed": "escalated",
            "identity_status": "dead",
            "pid": pid,
            "signal": "SIGKILL",
            "channel": how_k,
            "reason": "SIGTERM did not reap within grace; SIGKILL did",
        }
    return {
        "result": "refused",
        "stopped": False,
        "needed": "escalated",
        "identity_status": identity_status({"pid": pid, "start_token": record.get("start_token")}),
        "pid": pid,
        "signal": "SIGKILL",
        "reason": "resident still alive after SIGKILL; not claimed stopped",
    }


def restart(
    artifact: Mapping[str, Any],
    *,
    workspace: str | os.PathLike[str],
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Start the exact named artifact. No PATH lookup, no runtime resolution."""
    root = _as_workspace(workspace)
    spec = _artifact_spec(artifact)
    _assert_artifact_present(spec)
    argv = list(spec["argv"])
    unsafe = refuse_reason(argv)
    if unsafe:
        raise RestartRefused(f"artifact argv refused: {unsafe}", fault="unsafe_argv")
    cwd_s = spec.get("cwd") or str(root)
    cwd = Path(str(cwd_s))
    if not cwd.is_dir():
        raise RestartRefused(f"artifact cwd is not a directory: {cwd}", fault="cwd_missing")

    logs = root / "resident_logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_log = logs / "resident.stdout.log"
    stderr_log = logs / "resident.stderr.log"
    child_env = dict(os.environ)
    if env is not None:
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise RestartRefused("env must be a string mapping", fault="bad_env")
        child_env.update(env)
    source_root = str(Path(__file__).resolve().parents[2])
    pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = os.pathsep.join(
        [source_root] + ([pythonpath] if pythonpath else [])
    )

    try:
        out_fh = stdout_log.open("ab")
        err_fh = stderr_log.open("ab")
    except OSError as exc:
        raise RestartRefused(f"could not open resident logs: {exc}", fault="spawn_failed") from exc
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=out_fh,
            stderr=err_fh,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        raise RestartRefused(
            f"named artifact could not be spawned: {type(exc).__name__}: {exc}",
            fault="spawn_failed",
        ) from exc
    finally:
        try:
            out_fh.close()
        except OSError as exc:
            _ = exc
        try:
            err_fh.close()
        except OSError as exc:
            _ = exc
    _POPEN[proc.pid] = proc

    token = None
    deadline = time.monotonic() + SPAWN_WAIT_S
    while time.monotonic() < deadline:
        token = process_start_token(proc.pid)
        if token:
            break
        if proc.poll() is not None:
            break
        time.sleep(0.02)

    handle = {
        "pid": proc.pid,
        "pgid": proc.pid,
        "start_token": token,
        "argv": argv,
        "cwd": str(cwd),
        "artifact_path": spec["path"],
        "artifact_sha256": spec["sha256"],
        "alive": pid_is_alive(proc.pid),
        "returncode": proc.poll(),
        "identity_status": identity_status({"pid": proc.pid, "start_token": token}),
        "log_paths": {"stdout": str(stdout_log), "stderr": str(stderr_log)},
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }
    write_sealed(root / HANDLE_NAME, {**handle, "schema": "hawking.future.restart_supervisor.handle.v1", "complete": True})
    handle["result"] = "ok" if handle["alive"] else "refused"
    if not handle["alive"]:
        _waitpid(proc.pid)
        handle["reason"] = (
            f"named artifact exited immediately returncode={handle['returncode']}; "
            "not rounded into a running resident"
        )
        raise RestartRefused(handle["reason"], fault="exited_immediately")
    return handle


def restore(
    checkpoint_src: str | os.PathLike[str] | Mapping[str, Any],
    *,
    workspace: str | os.PathLike[str],
) -> dict[str, Any]:
    """Rebuild frontier and queue from a sealed checkpoint. Missing artifact refuses."""
    root = _as_workspace(workspace)
    doc = load_checkpoint(checkpoint_src)
    spec = _artifact_spec(doc.get("artifact") if isinstance(doc.get("artifact"), Mapping) else None)
    _assert_artifact_present(spec)
    frontier = doc.get("frontier") if isinstance(doc.get("frontier"), Mapping) else {}
    restored = {
        "schema": RESTORED_SCHEMA,
        "complete": True,
        "mission_id": (doc.get("mission") or {}).get("mission_id") if isinstance(doc.get("mission"), Mapping) else None,
        "frontier": dict(frontier),
        "queue": list(doc.get("queue") or []),
        "in_flight": list(doc.get("in_flight") or []),
        "scar_consultations": list(doc.get("scar_consultations") or []),
        "detached_handles": list(doc.get("detached_handles") or []),
        "identity": dict(doc.get("identity") or {}),
        "artifact": spec,
        "frontier_restored": frontier.get("status") not in {None, "UNAVAILABLE"},
        "queue_restored": True,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }
    sealed = write_sealed(root / RESTORED_NAME, restored)
    sealed["path"] = str(root / RESTORED_NAME)
    sealed["result"] = "ok"
    return sealed


def _classify_rediscover(
    job_id: str,
    *,
    detached: DetachedSupervisor | None,
    checkpoint_handle: Mapping[str, Any] | None,
) -> dict[str, Any]:
    record: dict[str, Any] | None = None
    if detached is not None:
        try:
            record = detached.inspect(job_id)
        except FileNotFoundError:
            record = None
        except (OSError, DetachedError, ValueError) as exc:
            return {
                "job_id": job_id,
                "fate": "UNKNOWN",
                "relaunched": False,
                "adopted": False,
                "assumed_complete": False,
                "reason": f"inspect failed: {type(exc).__name__}: {exc}",
            }
    if record is None and checkpoint_handle is not None:
        record = dict(checkpoint_handle)

    if record is None:
        return {
            "job_id": job_id,
            "fate": "UNKNOWN",
            "relaunched": False,
            "adopted": False,
            "assumed_complete": False,
            "reason": "job is in neither the checkpoint nor the live table as a readable record",
        }

    terminal = record.get("terminal")
    state = record.get("state")
    expected = record.get("expected_receipt_path")
    status = identity_status(record)

    if terminal == "completed-with-receipt":
        present = bool(expected) and Path(str(expected)).is_file()
        if not present:
            return {
                "job_id": job_id,
                "fate": "UNKNOWN",
                "terminal": terminal,
                "relaunched": False,
                "adopted": False,
                "assumed_complete": False,
                "observed_complete_with_receipt": False,
                "reason": "terminal says completed-with-receipt but the receipt file is absent",
            }
        return {
            "job_id": job_id,
            "fate": "INGESTED",
            "terminal": terminal,
            "relaunched": False,
            "adopted": False,
            "assumed_complete": False,
            "observed_complete_with_receipt": True,
            "expected_receipt_path": expected,
            "reason": "finished job ingested; not relaunched",
        }

    if terminal in {
        "completed-without-receipt",
        "crashed",
        "cancelled",
        "timed_out",
    }:
        return {
            "job_id": job_id,
            "fate": "INGESTED",
            "terminal": terminal,
            "relaunched": False,
            "adopted": False,
            "assumed_complete": False,
            "observed_complete_with_receipt": False,
            "reason": f"terminal {terminal} ingested; not assumed complete; not relaunched",
        }

    if terminal == "unknown":
        return {
            "job_id": job_id,
            "fate": "UNKNOWN",
            "terminal": "unknown",
            "relaunched": False,
            "adopted": False,
            "assumed_complete": False,
            "reason": "persisted terminal is unknown; not guessed complete; not relaunched",
        }

    if state == "SLEEPING" or record.get("launch_refused"):
        return {
            "job_id": job_id,
            "fate": "INGESTED",
            "terminal": terminal,
            "state": "SLEEPING",
            "relaunched": False,
            "adopted": False,
            "assumed_complete": False,
            "reason": "SLEEPING/refused launch kept as-is; not relaunched",
        }

    if status == "match" and detached is not None:
        adopted = detached.adopt(job_id)
        return {
            "job_id": job_id,
            "fate": "ADOPTED",
            "terminal": adopted.get("terminal"),
            "state": adopted.get("state"),
            "identity_status": adopted.get("identity_status"),
            "pid": adopted.get("pid"),
            "relaunched": False,
            "adopted": True,
            "assumed_complete": False,
            "reason": "still-running child re-adopted by pid+start_token; not duplicated",
        }

    if status == "match" and detached is None:
        return {
            "job_id": job_id,
            "fate": "ADOPTED",
            "identity_status": "match",
            "pid": record.get("pid"),
            "relaunched": False,
            "adopted": True,
            "assumed_complete": False,
            "reason": "identity matches; no DetachedSupervisor to persist adopt, not relaunched",
        }

    return {
        "job_id": job_id,
        "fate": "UNKNOWN",
        "terminal": terminal,
        "identity_status": status,
        "relaunched": False,
        "adopted": False,
        "assumed_complete": False,
        "reason": (
            f"fate undetermined (identity_status={status}, state={state}, "
            f"terminal={terminal}); not assumed complete; not relaunched"
        ),
    }


def rediscover_detached(
    checkpoint_src: str | os.PathLike[str] | Mapping[str, Any] | None = None,
    *,
    detached: DetachedSupervisor | None = None,
    live_table: Sequence[Mapping[str, Any]] | None = None,
    detached_handles: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Ingest finished jobs, re-adopt live ones, UNKNOWN otherwise. Never relaunch. Never double-count."""
    checkpoint_handles: list[dict[str, Any]] = []
    if checkpoint_src is not None:
        doc = load_checkpoint(checkpoint_src)
        raw_handles = doc.get("detached_handles") or []
        if not isinstance(raw_handles, list):
            raise CheckpointCorrupt("detached_handles is not a list")
        checkpoint_handles = [dict(h) for h in raw_handles if isinstance(h, Mapping)]
    if detached_handles is not None:
        checkpoint_handles = [dict(h) for h in detached_handles]

    if live_table is not None:
        live_rows = [dict(r) for r in live_table]
        live_source = "supplied_live_table"
    elif detached is not None:
        try:
            live_rows = [dict(r) for r in detached.list()]
        except (OSError, DetachedError) as exc:
            raise RestartRefused(
                f"live detached table unreadable: {type(exc).__name__}: {exc}",
                fault="detached_unreadable",
            ) from exc
        live_source = "detached.list"
    else:
        live_rows = []
        live_source = "NONE_SUPPLIED"

    mentions: list[str] = []
    for row in list(checkpoint_handles) + list(live_rows):
        jid = str(row.get("job_id") or "").strip()
        if jid:
            mentions.append(jid)

    by_id_checkpoint = {
        str(h.get("job_id")): h for h in checkpoint_handles if h.get("job_id")
    }
    seen: set[str] = set()
    unique_ids: list[str] = []
    for jid in mentions:
        if jid in seen:
            continue
        seen.add(jid)
        unique_ids.append(jid)

    jobs: list[dict[str, Any]] = []
    for jid in unique_ids:
        jobs.append(
            _classify_rediscover(
                jid,
                detached=detached,
                checkpoint_handle=by_id_checkpoint.get(jid),
            )
        )

    relaunched = [j for j in jobs if j.get("relaunched")]
    if relaunched:
        raise RestartRefused(
            f"rediscover attempted a relaunch of { [j['job_id'] for j in relaunched] }; "
            "this is a supervisor bug",
            fault="relaunch_forbidden",
        )

    return {
        "result": "ok",
        "n_input_mentions": len(mentions),
        "n_jobs": len(jobs),
        "n_duplicates_dropped": max(0, len(mentions) - len(jobs)),
        "fates": {f: sum(1 for j in jobs if j.get("fate") == f) for f in FATES},
        "jobs": jobs,
        "relaunched": False,
        "live_source": live_source,
        "assumed_complete": False,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


def restart_cycle(
    authority: Any,
    *,
    workspace: str | os.PathLike[str],
    mission: Any,
    artifact: Mapping[str, Any],
    resident: Mapping[str, Any] | None = None,
    already_dead: bool = False,
    detached: DetachedSupervisor | None = None,
    detached_handles: Sequence[Mapping[str, Any]] | None = None,
    identity: Mapping[str, Any] | None = None,
    frontier: Mapping[str, Any] | Sequence[Any] | None = None,
    queue: Sequence[Any] | None = None,
    in_flight: Sequence[Any] | None = None,
    scar_consultations: Sequence[Any] | None = None,
    grace_s: float = DEFAULT_GRACE_S,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Checkpoint, stop, restart the named artifact, restore, rediscover. Evidence per step."""
    if isinstance(authority, RestartRequest):
        raise RestartAuthorityError(
            "a RestartRequest cannot drive restart_cycle; only a SupervisorDecision can"
        )
    if not isinstance(authority, SupervisorDecision):
        raise RestartAuthorityError(
            f"restart_cycle requires SupervisorDecision, got {type(authority).__name__}"
        )

    steps: dict[str, Any] = {
        "checkpoint": {"result": "not_run"},
        "stop": {"result": "not_run"},
        "restart": {"result": "not_run"},
        "restore": {"result": "not_run"},
        "rediscover_detached": {"result": "not_run"},
    }
    verdict: dict[str, Any] = {
        "schema": "hawking.future.restart_supervisor.cycle.v1",
        "authority": authority.to_dict(),
        "steps": steps,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "relaunched_detached": False,
    }
    if authority.action != "RESTART":
        verdict["result"] = "refused"
        verdict["reason"] = authority.reason
        verdict["status"] = "REFUSED"
        return verdict

    root = _as_workspace(workspace)
    try:
        ckpt = checkpoint(
            mission,
            workspace=root,
            artifact=artifact,
            detached=detached,
            detached_handles=detached_handles,
            identity=identity,
            frontier=frontier,
            queue=queue,
            in_flight=in_flight,
            scar_consultations=scar_consultations,
        )
        steps["checkpoint"] = {
            "result": "ok",
            "path": ckpt.get("path"),
            "artifact_sha256": (ckpt.get("artifact") or {}).get("sha256"),
            "n_detached_handles": len(ckpt.get("detached_handles") or []),
            "n_queue": len(ckpt.get("queue") or []),
        }
    except RestartRefused as exc:
        steps["checkpoint"] = {"result": "refused", "fault": exc.fault, "reason": exc.reason}
        verdict["result"] = "refused"
        verdict["status"] = "REFUSED"
        verdict["reason"] = f"checkpoint: {exc.reason}"
        return verdict

    if already_dead and resident is None:
        steps["stop"] = {
            "result": "ok",
            "needed": "already_dead",
            "reason": "caller declared no live resident; nothing signalled",
        }
    else:
        try:
            stopped = stop(resident, grace_s=grace_s)
        except RestartRefused as exc:
            steps["stop"] = {"result": "refused", "fault": exc.fault, "reason": exc.reason}
            verdict["result"] = "refused"
            verdict["status"] = "REFUSED"
            verdict["reason"] = f"stop: {exc.reason}"
            return verdict
        steps["stop"] = {k: stopped[k] for k in stopped}
        if stopped.get("result") != "ok":
            verdict["result"] = "refused"
            verdict["status"] = "REFUSED"
            verdict["reason"] = f"stop: {stopped.get('reason')}"
            return verdict

    try:
        launched = restart(artifact, workspace=root, env=env)
        steps["restart"] = {
            "result": "ok",
            "pid": launched.get("pid"),
            "identity_status": launched.get("identity_status"),
            "artifact_path": launched.get("artifact_path"),
            "artifact_sha256": launched.get("artifact_sha256"),
            "argv": launched.get("argv"),
        }
        verdict["resident"] = {
            "pid": launched.get("pid"),
            "start_token": launched.get("start_token"),
        }
    except RestartRefused as exc:
        steps["restart"] = {"result": "refused", "fault": exc.fault, "reason": exc.reason}
        verdict["result"] = "refused"
        verdict["status"] = "REFUSED"
        verdict["reason"] = f"restart: {exc.reason}"
        return verdict

    try:
        restored = restore(ckpt.get("path") or (root / CHECKPOINT_NAME), workspace=root)
        steps["restore"] = {
            "result": "ok",
            "path": restored.get("path"),
            "queue": restored.get("queue"),
            "frontier_item_ids": (restored.get("frontier") or {}).get("item_ids"),
            "frontier_restored": restored.get("frontier_restored"),
            "queue_restored": restored.get("queue_restored"),
        }
    except RestartRefused as exc:
        steps["restore"] = {"result": "refused", "fault": exc.fault, "reason": exc.reason}
        verdict["result"] = "refused"
        verdict["status"] = "REFUSED"
        verdict["reason"] = f"restore: {exc.reason}"
        _reap_quietly((verdict.get("resident") or {}).get("pid"))
        return verdict

    try:
        discovered = rediscover_detached(
            ckpt.get("path") or (root / CHECKPOINT_NAME),
            detached=detached,
            detached_handles=detached_handles,
        )
        steps["rediscover_detached"] = {
            "result": "ok",
            "n_jobs": discovered.get("n_jobs"),
            "n_input_mentions": discovered.get("n_input_mentions"),
            "n_duplicates_dropped": discovered.get("n_duplicates_dropped"),
            "fates": discovered.get("fates"),
            "relaunched": discovered.get("relaunched"),
            "assumed_complete": discovered.get("assumed_complete"),
        }
        verdict["rediscover"] = discovered
    except RestartRefused as exc:
        steps["rediscover_detached"] = {
            "result": "refused",
            "fault": exc.fault,
            "reason": exc.reason,
        }
        verdict["result"] = "refused"
        verdict["status"] = "REFUSED"
        verdict["reason"] = f"rediscover_detached: {exc.reason}"
        _reap_quietly((verdict.get("resident") or {}).get("pid"))
        return verdict

    verdict["result"] = "ok"
    verdict["status"] = "PASSED"
    verdict["reason"] = "every step executed with evidence; no step was rounded into a pass"
    return verdict


class RestartSupervisor:
    """Workspace-bound supervisor. The only object that performs a restart."""

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        *,
        grace_s: float = DEFAULT_GRACE_S,
        detached: DetachedSupervisor | None = None,
    ) -> None:
        self.workspace = _as_workspace(workspace)
        self.grace_s = float(grace_s)
        self.detached = detached if detached is not None else DetachedSupervisor(self.workspace)

    def request_restart(self, reason: str, **kw: Any) -> RestartRequest:
        return request_restart(reason, **kw)

    def decide(self, request: RestartRequest) -> SupervisorDecision:
        return decide(request)

    def checkpoint(self, mission: Any, **kw: Any) -> dict[str, Any]:
        kw.setdefault("detached", self.detached)
        return checkpoint(mission, workspace=self.workspace, **kw)

    def stop(self, resident: Mapping[str, Any] | None, **kw: Any) -> dict[str, Any]:
        kw.setdefault("grace_s", self.grace_s)
        return stop(resident, **kw)

    def restart(self, artifact: Mapping[str, Any], **kw: Any) -> dict[str, Any]:
        return restart(artifact, workspace=self.workspace, **kw)

    def restore(self, checkpoint_src: Any, **kw: Any) -> dict[str, Any]:
        return restore(checkpoint_src, workspace=self.workspace, **kw)

    def rediscover_detached(self, checkpoint_src: Any = None, **kw: Any) -> dict[str, Any]:
        kw.setdefault("detached", self.detached)
        return rediscover_detached(checkpoint_src, **kw)

    def restart_cycle(self, authority: Any, **kw: Any) -> dict[str, Any]:
        kw.setdefault("detached", self.detached)
        kw.setdefault("grace_s", self.grace_s)
        return restart_cycle(authority, workspace=self.workspace, **kw)


# ---------------------------------------------------------------------------
# Proofs. Declared capability is not evidence; these actually fire.
# ---------------------------------------------------------------------------


def _write_artifact(root: Path, source: str, name: str = "resident_artifact.py") -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(source)
    argv0 = Path(sys.executable).resolve()
    marker = root / f"{path.stem}.marker"
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "argv": [str(argv0), str(path), str(marker)],
        "cwd": str(root),
    }


def _spawn_handle(artifact: Mapping[str, Any], workspace: Path) -> dict[str, Any]:
    marker = Path(artifact["argv"][2])
    if marker.is_file():
        marker.unlink()
    handle = restart(artifact, workspace=workspace)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if marker.is_file() and marker.read_text() == "alive":
            return handle
        time.sleep(0.02)
    return handle


def _mission() -> dict[str, Any]:
    return {
        "mission_id": "PROOF.RESTART",
        "phase": "running",
        "next_action": "drain then refill",
        "units": ["WU.PROOF.1"],
        "queue": [{"id": "WU.PROOF.2", "frontier_id": "FT.HCLI_SELF.no-launch"}],
        "frontier_ids": ["FT.HCLI_SELF.no-launch"],
        "scar_consultations": ["SCAR.PROOF"],
    }


def run_proofs(tmp: Path | None = None) -> dict[str, Any]:
    """Hermetic negative controls. A proof that does not fire is a failed proof."""
    own = tmp is None
    root = Path(tmp) if tmp is not None else Path(tempfile.mkdtemp(prefix="restart-sup-"))
    proofs: dict[str, Any] = {}
    failed: list[str] = []

    def _ok(name: str, cond: bool, detail: Any) -> None:
        proofs[name] = {"fires": bool(cond), "detail": detail}
        if not cond:
            failed.append(name)

    try:
        artifact = _write_artifact(root, _RESIDENT_SCRIPT)
        mission = _mission()
        sup = RestartSupervisor(root, grace_s=0.3)

        # Truncated mid-write is refused, not half-read.
        torn = root / "torn.json"
        torn.write_bytes(b'{"schema": "hawking.future.restart_supervisor.checkpoint.v1", "artifact": {')
        try:
            load_checkpoint(torn)
            _ok("truncated_checkpoint", False, "load_checkpoint accepted torn JSON")
        except CheckpointCorrupt as exc:
            _ok("truncated_checkpoint", exc.fault == "truncated_mid_write", exc.reason)

        # Leftover atomic_write tmp with no live file is a mid-write.
        ghost = root / "ghost.json"
        tmp_sib = ghost.parent / f".{ghost.name}.1.deadbeef.tmp"
        tmp_sib.write_text("{")
        try:
            load_checkpoint(ghost)
            _ok("truncated_tmp_sibling", False, "missing file with tmp sibling was accepted")
        except CheckpointCorrupt as exc:
            _ok("truncated_tmp_sibling", exc.fault == "truncated_mid_write", exc.reason)
        tmp_sib.unlink(missing_ok=True)

        # Missing named artifact refuses restore.
        ckpt = checkpoint(mission, workspace=root, artifact=artifact, frontier={"status": "SUPPLIED", "item_ids": ["FT.HCLI_SELF.no-launch"], "items": []})
        gone = Path(artifact["path"])
        saved = gone.read_text()
        gone.unlink()
        try:
            restore(ckpt["path"], workspace=root)
            _ok("restore_missing_artifact", False, "restore accepted a missing artifact")
        except ArtifactMissing as exc:
            _ok("restore_missing_artifact", True, exc.reason)
        gone.write_text(saved)
        # Hash changed after restore of the original bytes — rewrite different bytes.
        gone.write_text(saved + "#tamper\n")
        try:
            restore(ckpt["path"], workspace=root)
            _ok("restore_hash_mismatch", False, "restore accepted tampered bytes")
        except ArtifactMissing as exc:
            _ok("restore_hash_mismatch", True, exc.reason)
        gone.write_text(saved)

        # Request cannot execute.
        req = request_restart("hung event loop", pathology="hung")
        raised = False
        try:
            req.restart  # type: ignore[attr-defined]
        except RestartAuthorityError:
            raised = True
        cycle_raised = False
        try:
            restart_cycle(req, workspace=root, mission=mission, artifact=artifact, already_dead=True)
        except RestartAuthorityError:
            cycle_raised = True
        ctor_raised = False
        try:
            SupervisorDecision(req, "RESTART", "no")  # type: ignore[misc]
        except RestartAuthorityError:
            ctor_raised = True
        _ok("request_cannot_execute", raised and cycle_raised and ctor_raised,
            {"attr": raised, "cycle": cycle_raised, "ctor": ctor_raised})

        empty_dec = decide(request_restart(""))
        _ok("empty_reason_refuses", empty_dec.action == "REFUSE", empty_dec.reason)
        refused_cycle = restart_cycle(
            empty_dec, workspace=root, mission=mission, artifact=artifact, already_dead=True
        )
        _ok(
            "refuse_decision_does_not_restart",
            refused_cycle.get("status") == "REFUSED"
            and refused_cycle["steps"]["restart"]["result"] == "not_run",
            refused_cycle.get("reason"),
        )

        # Rediscover does not double-count a job in both tables.
        dsup = DetachedSupervisor(root)
        sleep_unit = {
            "id": "dup-job",
            "role": "science",
            "description": "sleep for rediscover",
            "command": ["/bin/sleep", "20"],
            "resource_class": "LIGHT_CONTROL",
            "verifier": "future.restart_supervisor.rediscover",
            "classification": "STATIC_ONLY",
        }
        rec = dsup.launch(sleep_unit)
        live = dsup.list()
        handle = {"job_id": rec["job_id"], "pid": rec.get("pid"), "start_token": rec.get("start_token")}
        discovered = rediscover_detached(
            detached_handles=[handle],
            detached=dsup,
            live_table=live,
        )
        _ok(
            "rediscover_no_double_count",
            discovered["n_jobs"] == 1 and discovered["n_input_mentions"] >= 2 and discovered["relaunched"] is False,
            {k: discovered[k] for k in ("n_jobs", "n_input_mentions", "n_duplicates_dropped", "fates")},
        )
        dsup.cancel(rec["job_id"])

        # Finished job is ingested, not relaunched.
        receipt_path = root / "results" / "done.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        done_unit = {
            "id": "done-job",
            "role": "science",
            "description": "write receipt and exit",
            "command": [
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{\"ok\":true}')",
                str(receipt_path),
            ],
            "resource_class": "LIGHT_CONTROL",
            "verifier": "future.restart_supervisor.ingest",
            "classification": "STATIC_ONLY",
            "output_receipt_path": str(receipt_path),
        }
        done_rec = dsup.launch(done_unit)
        terminal = dsup.wait_terminal(done_rec["job_id"], timeout_s=4.0)
        ingested = rediscover_detached(detached_handles=[_handle_snapshot(terminal)], detached=dsup)
        job0 = (ingested.get("jobs") or [{}])[0]
        _ok(
            "rediscover_ingests_finished",
            job0.get("fate") == "INGESTED" and job0.get("relaunched") is False and ingested["relaunched"] is False,
            job0,
        )

        ghost_job = rediscover_detached(
            detached_handles=[{"job_id": "ghost-no-record", "pid": 999999, "start_token": "nope"}]
        )
        _ok(
            "rediscover_unknown_undetermined",
            (ghost_job["jobs"] or [{}])[0].get("fate") == "UNKNOWN"
            and ghost_job["jobs"][0].get("assumed_complete") is False,
            ghost_job["jobs"][0],
        )

        # Cooperative stop.
        handle = _spawn_handle(artifact, root)
        stopped = stop(handle, grace_s=0.5)
        _ok(
            "stop_cooperative",
            stopped.get("needed") == "cooperative" and stopped.get("stopped") is True,
            stopped,
        )

        # Escalation when SIGTERM is ignored.
        ign = _write_artifact(root, _IGNORING_SCRIPT, name="ignore_term.py")
        ign_handle = _spawn_handle(ign, root)
        escalated = stop(ign_handle, grace_s=0.25)
        _ok(
            "stop_escalates",
            escalated.get("needed") == "escalated" and escalated.get("stopped") is True,
            escalated,
        )

        # Unproven identity is not signalled. A live foreign sleep + forged token.
        foreign = subprocess.Popen(
            ["/bin/sleep", "20"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            forged = {"pid": foreign.pid, "start_token": "not-the-real-token"}
            refused_stop = stop(forged, grace_s=0.2)
            still_alive = pid_is_alive(foreign.pid)
            _ok(
                "stop_refuses_unproven_identity",
                refused_stop.get("needed") == "refused" and still_alive,
                {"stop": refused_stop, "foreign_alive": still_alive},
            )
        finally:
            _reap_quietly(foreign.pid)

        # Relative argv[0] refused (would be a PATH lookup).
        rel = dict(artifact)
        rel["argv"] = ["python3", artifact["path"]]
        try:
            _artifact_spec(rel)
            _ok("refuse_runtime_path_lookup", False, "relative argv[0] accepted")
        except RestartRefused as exc:
            _ok("refuse_runtime_path_lookup", exc.fault == "runtime_resolution_refused", exc.reason)

        # Full cycle with a live resident.
        live_handle = _spawn_handle(artifact, root)
        decision = decide(request_restart("pathological event loop", pathology="hung"))
        cycle = restart_cycle(
            decision,
            workspace=root,
            mission=mission,
            artifact=artifact,
            resident=live_handle,
            detached=dsup,
            frontier={"status": "SUPPLIED", "item_ids": ["FT.HCLI_SELF.no-launch"], "items": []},
            queue=mission["queue"],
            in_flight=mission["units"],
            scar_consultations=mission["scar_consultations"],
            grace_s=0.5,
        )
        new_pid = (cycle.get("resident") or {}).get("pid")
        _ok(
            "restart_cycle_each_step",
            cycle.get("status") == "PASSED"
            and all(cycle["steps"][s]["result"] == "ok" for s in cycle["steps"])
            and cycle["steps"]["restore"]["queue_restored"] is True,
            {k: cycle["steps"][k]["result"] for k in cycle["steps"]},
        )
        _reap_quietly(new_pid)

        # Absent mission refuses.
        try:
            checkpoint(root / "no-mission.json", workspace=root, artifact=artifact)
            _ok("absent_mission_refuses", False, "missing mission was accepted")
        except RestartRefused as exc:
            _ok("absent_mission_refuses", exc.fault == "mission_required", exc.reason)

    finally:
        if own:
            # Best-effort: proofs must not leave sleeps behind.
            try:
                DetachedSupervisor(root).reap_all()
            except Exception as exc:
                proofs.setdefault("cleanup", {"fires": True, "detail": type(exc).__name__})

    proofs["all_passed"] = not failed
    proofs["failed"] = failed
    return proofs


def emit_resident_workunit() -> dict[str, Any]:
    unit = WorkUnit(
        id="future.restart-supervisor.restart-cycle",
        role="science",
        description=(
            "Checkpoint mission+artifact identity, cooperatively stop then escalate, "
            "restart the exact named artifact, restore frontier/queue, rediscover "
            "detached jobs without relaunching. CPU_ANALYSIS; no GPU lease."
        ),
        resource_class="LIGHT_CONTROL",
        provider="future.restart_supervisor",
        verifier="future.restart_supervisor.cycle_verdict",
        effect_class="REVERSIBLE",
        workspace="repo-root",
        classification="STATIC_ONLY",
        status="pending",
    )
    row = unit.to_dict()
    row.update(
        {
            "claim_boundary": CLAIM_BOUNDARY,
            "species": "HCLI_SELF_OPTIMIZE",
            "resource_class_note": "CPU_ANALYSIS protocol; LIGHT_CONTROL because this sidecar holds no GPU lease",
            "requires_quiescence": False,
            "may_promote": False,
            "may_modify_verifier": False,
            "gpu_authority": False,
        }
    )
    WorkUnit.from_dict(dict(row))
    return row


def recovered_implementation() -> list[str]:
    return [
        "tools/future/detached.py DetachedSupervisor.adopt/list/inspect/identity_status — re-adopt is pid+start_token; this lane calls it rather than forking a second supervisor",
        "tools/future/wakeup.py classify_receipt_bytes / write_sealed / PARTIAL_TRUNCATED — torn checkpoints are the same fault as torn receipts",
        "tools/future/resident_identity.py load() — restart-survivable identity document; snapshot is a digest, never a conversational reconstruction",
        "tools/future/resident_install.py restart slot no_silent_restart=True — silent restart remains forbidden",
        "tools/future/autonomy_run.py MISSION_STATE — durable mission path; absent file is a refusal, not an empty success",
        "tools/future/sandbox.py start_model raises AuthorityRefused — this supervisor does not start a model",
        "tools/future/frontiers.py next_work/refill — restore rebuilds the snapshotted queue; it does not call live refill as a substitute",
        "tools/future/succession.py checkpoint_incumbent — census-style, not a live process snapshot; this lane is the live-process snapshot that was missing",
        "tools/future/super_resident.py SilentRestartRefused — crash does not silently restart the body; this lane is the explicit, evidenced restart",
        "hcli/agentos/recovery.py — fixture recovery (read, not forked); production recovery remains Codex-owned",
        "hcli/persist.py atomic_write_json — crash-safe writer; leftover .tmp is treated as truncated mid-write",
        "hcli/resources.py pid_is_alive / process_start_token — pid alone is never identity",
    ]


def gaps_closed() -> list[str]:
    return [
        "no supervisor made a restart survivable: checkpoint of mission+frontier+queue+detached handles+artifact sha256, stop with cooperative-vs-escalated evidence, restart of the exact named artifact, restore, rediscover",
        "resident-issued restart was not a type boundary; RestartRequest now cannot execute and cannot be passed to restart_cycle",
        "detached jobs that outlived a dead resident had adopt() but nothing called it on restart, so a cycle could silently relaunch duplicates",
        "truncated checkpoint mid-write had no loader; wakeup.classify_receipt_bytes is now the gate",
        "restore had no artifact-existence check, so a checkpoint could restore a queue for a binary that was gone",
    ]


def negative_findings() -> list[str]:
    return [
        "does not start a model (sandbox forbids; no GPU lease; super_resident never starts a process)",
        "does not flock a bench lock or take a GPU lease; refuse_reason still gates argv",
        "live hcli/agentos/resident_gate.py and recovery.py remain Codex-owned; proofs run on fixture processes",
        "orchestration.py BINDINGS cannot be updated from this write partition, so the connector will report this module unbound until a later lane binds it",
        "pid wraparound was not waited out; reuse is proven by a live foreign process plus a forged start_token",
        "MISSION_STATE may be absent; checkpoint without a mission is refused rather than invented",
        "a still-alive-after-SIGKILL resident is recorded refused, not claimed stopped",
    ]


def resident_callable() -> dict[str, Any]:
    unit = emit_resident_workunit()
    return {
        "entry_point": "tools.future.restart_supervisor.restart_cycle(decision, workspace=...)",
        "workunit": (
            "one CPU_ANALYSIS unit; checkpoint/stop/restart/restore/rediscover_detached; "
            f"id={unit['id']}"
        ),
        "workunit_record": unit,
        "receipt": f"receipts/future/{RECEIPT}",
        "frontier": "FT.HCLI_SELF.no-launch",
        "fails_closed": (
            "RestartRequest cannot execute; truncated/unsealed checkpoints refuse; "
            "missing or hash-mismatched artifacts refuse; unproven pid identity is not signalled; "
            "undetermined detached jobs are UNKNOWN and never assumed complete; never relaunches"
        ),
        "callable": True,
        "python_api": {
            "checkpoint": "tools.future.restart_supervisor.checkpoint(mission, workspace=..., artifact=...)",
            "stop": "tools.future.restart_supervisor.stop(resident)",
            "restart": "tools.future.restart_supervisor.restart(artifact, workspace=...)",
            "restore": "tools.future.restart_supervisor.restore(checkpoint, workspace=...)",
            "rediscover_detached": "tools.future.restart_supervisor.rediscover_detached(...)",
            "request_restart": "tools.future.restart_supervisor.request_restart(reason) -> RestartRequest",
            "decide": "tools.future.restart_supervisor.decide(request) -> SupervisorDecision",
        },
    }


def build() -> Path:
    proofs = run_proofs()
    if not proofs.get("all_passed"):
        raise RestartRefused(
            "negative-control proofs failed: " + ", ".join(proofs.get("failed") or []),
            fault="proofs_failed",
        )
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Supervisor that makes a resident restart survivable and provable: "
            "durable checkpoint, cooperative-then-escalated stop, exact-artifact "
            "restart, restore of frontier/queue, rediscovery of detached jobs."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "status": "BUILT_NOT_PROMOTED",
        "built": True,
        "promoted": False,
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "module_sha256": sha256_file(Path(__file__)),
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "authority_boundary": {
            "resident_may": "file a RestartRequest with a reason",
            "resident_may_not": "execute stop/restart/restore/restart_cycle",
            "supervisor_decides": "decide(request) issues a SupervisorDecision",
            "supervisor_performs": "restart_cycle(decision, ...)",
            "user_constructible_decision": False,
            "request_executable": False,
        },
        "operations": ["checkpoint", "stop", "restart", "restore", "rediscover_detached", "restart_cycle"],
        "fates": list(FATES),
        "stop_needed": list(STOP_NEEDED),
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "required_checkpoint_keys": list(REQUIRED_CHECKPOINT_KEYS),
        "live_mission_state_present": MISSION_STATE.is_file(),
        "proofs": {k: v for k, v in proofs.items() if k not in {"failed"}},
        "proofs_all_passed": proofs["all_passed"],
        "negative_controls": {
            name: proofs[name]
            for name in (
                "truncated_checkpoint",
                "truncated_tmp_sibling",
                "restore_missing_artifact",
                "restore_hash_mismatch",
                "request_cannot_execute",
                "empty_reason_refuses",
                "refuse_decision_does_not_restart",
                "rediscover_no_double_count",
                "rediscover_ingests_finished",
                "rediscover_unknown_undetermined",
                "stop_cooperative",
                "stop_escalates",
                "stop_refuses_unproven_identity",
                "refuse_runtime_path_lookup",
                "restart_cycle_each_step",
                "absent_mission_refuses",
            )
            if name in proofs
        },
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
        "resident_callable": resident_callable(),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArtifactMissing",
    "CHECKPOINT_SCHEMA",
    "CheckpointCorrupt",
    "RECEIPT",
    "RestartAuthorityError",
    "RestartRefused",
    "RestartRequest",
    "RestartSupervisor",
    "SCHEMA",
    "SupervisorDecision",
    "build",
    "checkpoint",
    "decide",
    "load_checkpoint",
    "rediscover_detached",
    "request_restart",
    "restart",
    "restart_cycle",
    "restore",
    "selftest",
    "stop",
]

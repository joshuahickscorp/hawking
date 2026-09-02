#!/usr/bin/env python3
"""Kill a real process DURING a checkpoint and assert one coherent generation survives.

Run:
    python3 tools/headless/hcli_crash_checkpoint_test.py

Every crash case is a parent-delivered SIGKILL of a child Python process.
An in-process exception is not a crash and is not used as one.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]

HCLI = str(REPO)
RECEIPT_PATH = REPO / "receipts" / "headless" / "HCLI_CRASH_CHECKPOINT.json"
OLD_MARK = "OLD_INTACT"
NEW_MARK = "NEW_COMPLETE"
PAD_OLD = "A" * 4096
PAD_NEW = "B" * 4096
SENTINEL_BODY = "ready-for-sigkill"
KILL_WAIT_S = 12.0

FAILS: List[str] = []
CASES: List[Dict[str, Any]] = []
WATCHED_FAIL: List[Dict[str, Any]] = []


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=10,
    )
    if proc.returncode != 0:
        return f"UNKNOWN:{proc.stderr.strip()}"
    return proc.stdout.strip()


def check(name: str, cond: bool, detail: str = "") -> bool:
    if cond:
        print(f"ok   {name}")
        return True
    print(f"FAIL {name}: {detail}")
    FAILS.append(f"{name}: {detail}")
    return False


def _record(case: Dict[str, Any]) -> None:
    CASES.append(case)
    verdict = case.get("verdict", "?")
    name = case.get("name", "?")
    print(f"     case {name}: {verdict}")
    extra = case.get("detail")
    if extra:
        print(f"       {extra}")


def _watched(title: str, evidence: str, on_disk: str) -> None:
    rec = {"title": title, "evidence": evidence, "on_disk": on_disk}
    WATCHED_FAIL.append(rec)
    print(f"WATCHED FAIL: {title}")
    print(f"  {evidence}")
    print(f"  on disk: {on_disk}")


def classify_bytes(
    raw: bytes,
    *,
    expect_json: bool = True,
) -> Dict[str, Any]:
    text = raw.decode("utf-8", "replace")
    has_old = OLD_MARK in text
    has_new = NEW_MARK in text
    parseable = None
    parse_error = None
    if expect_json:
        try:
            parseable = json.loads(text)
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            parseable = None
    if expect_json:
        if parseable is not None and has_old and not has_new:
            verdict = "OLD_INTACT"
        elif parseable is not None and has_new and not has_old:
            verdict = "NEW_COMPLETE"
        else:
            verdict = "HYBRID_TRUNCATED"
    else:
        if has_old and not has_new:
            verdict = "OLD_INTACT"
        elif has_new and not has_old:
            verdict = "NEW_COMPLETE"
        else:
            verdict = "HYBRID_TRUNCATED"
    return {
        "verdict": verdict,
        "bytes": len(raw),
        "has_old": has_old,
        "has_new": has_new,
        "parseable": parseable is not None if expect_json else None,
        "parse_error": parse_error,
        "head": text[:160].replace("\n", "\\n"),
        "tail": text[-120:].replace("\n", "\\n") if text else "",
    }


def _child_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = HCLI + os.pathsep + env.get("PYTHONPATH", "")
    env["HCLI_DISABLE_SIGNAL_HOOKS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def launch_child(args: List[str], sentinel: Path) -> subprocess.Popen:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", *args]
    return subprocess.Popen(
        cmd,
        cwd=str(REPO),
        env=_child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def sigkill_after_sentinel(
    proc: subprocess.Popen,
    sentinel: Path,
    timeout: float = KILL_WAIT_S,
) -> Dict[str, Any]:
    """Wait until the child publishes the interruption point, then SIGKILL it."""
    deadline = time.time() + timeout
    saw = False
    while time.time() < deadline:
        if sentinel.is_file() and sentinel.read_text(encoding="utf-8") == SENTINEL_BODY:
            saw = True
            break
        if proc.poll() is not None:
            break
        time.sleep(0.01)
    killed = False
    if proc.poll() is None:
        os.kill(proc.pid, signal.SIGKILL)
        killed = True
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
    stdout, stderr = proc.communicate(timeout=2) if proc.stdout else ("", "")
    # communicate after wait is empty; grab from PIPE if still buffered
    if not stdout and proc.stdout is None:
        stdout = ""
    return {
        "pid": proc.pid,
        "returncode": proc.returncode,
        "sigkill_delivered": killed,
        "saw_sentinel": saw,
        "is_sigkill": proc.returncode == -signal.SIGKILL,
        "stdout_tail": (stdout or "")[-400:],
        "stderr_tail": (stderr or "")[-800:],
    }


# ---------------------------------------------------------------------------
# Child-side helpers (run in the killed process)
# ---------------------------------------------------------------------------


def _install_midwrite_hook(sentinel: Path, dest: Path, *, inplace: bool) -> None:
    """Partial-write the target, publish sentinel, sleep for the parent's SIGKILL.

    Atomic writers open a sibling ``*.tmp``. In-place writers open ``dest``
    itself with mode ``w``, which truncates the live file before the first
    byte of the new generation is written.

    pathlib.Path.write_text goes through ``io.open``, not ``builtins.open``.
    Both must be wrapped or a Path.write_text writer completes un-killed.
    """
    import builtins
    import io

    real_open = io.open
    dest_name = dest.name

    def wrapped(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        mode_s = str(mode)
        fh = real_open(path, *args, **kwargs)
        writing = any(flag in mode_s for flag in ("w", "a", "+"))
        if not writing:
            return fh
        opened_name = Path(str(path)).name
        if inplace:
            match = opened_name == dest_name and "tmp" not in opened_name
        else:
            match = dest_name in opened_name and "tmp" in opened_name
        if not match:
            return fh

        class Killer:
            def write(self, data):
                if isinstance(data, bytes):
                    n = max(1, len(data) // 3)
                    fh.write(data[:n])
                else:
                    n = max(1, len(data) // 3)
                    fh.write(data[:n])
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
                sentinel.write_text(SENTINEL_BODY, encoding="utf-8")
                time.sleep(120)
                return n

            def flush(self):
                return fh.flush()

            def fileno(self):
                return fh.fileno()

            def close(self):
                return fh.close()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()
                return False

            def __getattr__(self, name):
                return getattr(fh, name)

        return Killer()

    builtins.open = wrapped  # type: ignore[assignment]
    io.open = wrapped  # type: ignore[assignment]


def _stub_engine():
    class E:
        def execute_workunit(self, unit, context):
            return {"validation": {"ok": True}}

        def cancel(self):
            pass

        child_pids: List[int] = []

    return E()


def _wu(uid: str, **kwargs):
    from hcli.workunit import WorkUnit

    return WorkUnit(
        id=uid,
        role=kwargs.pop("role", "work"),
        description=kwargs.pop("description", uid),
        resource_class=kwargs.pop("resource_class", "LIGHT_CONTROL"),
        **kwargs,
    )


def child_between(ws: Path, sentinel: Path) -> None:
    """Real Mission.checkpoint: persist DAG, then die before state.json."""
    from hcli.mission import Mission
    from hcli.scheduler import Scheduler

    dispatched = _wu("dispatched", description="unit that will be dispatched")
    idle = _wu("idle", description="unit that stays pending")
    mission = Mission(
        ws,
        engine=_stub_engine(),
        units={"dispatched": dispatched, "idle": idle},
        quiet=True,
        mission_id="crash-ckpt",
        heartbeat_s=60,
        no_progress_threshold=100,
    )
    mission.checkpoint()
    dispatched.status = "running"
    dispatched.attempts = 1
    dispatched.assigned_runtime = 0
    dispatched.backend_task_id = "task-GEN1"
    dispatched.assigned_backend = "qwen"
    mission._inflight["dispatched"] = object()
    real = Scheduler._persist

    def persist_then_pause(self, extra=None):
        # Mission.checkpoint() calls _persist({"checkpoint_id": ...}).
        # A 1-arg stub raises TypeError before the DAG write, so the
        # parent never sees the sentinel and SIGKILL never lands in
        # the between-writes window (child rc=1, both files still gen0).
        real(self, extra)
        sentinel.write_text(SENTINEL_BODY, encoding="utf-8")
        time.sleep(120)

    Scheduler._persist = persist_then_pause  # type: ignore[method-assign]
    mission.checkpoint()
    raise SystemExit("child was not killed after DAG persist")


def child_checkpoint_ids(ws: Path, sentinel: Path) -> None:
    """Two complete checkpoints, then exit cleanly. No kill."""
    from hcli.mission import Mission, load_state

    u = _wu("x", description="id-probe")
    mission = Mission(
        ws,
        engine=_stub_engine(),
        units={"x": u},
        quiet=True,
        mission_id="id-probe",
        heartbeat_s=60,
        no_progress_threshold=100,
    )
    p0 = mission.checkpoint()
    st0 = load_state(p0)
    (ws / "gen0.state.json").write_text(
        Path(p0).read_text(encoding="utf-8"), encoding="utf-8"
    )
    dag0 = ws / ".hcli" / "dag.json"
    if dag0.is_file():
        (ws / "gen0.dag.json").write_text(
            dag0.read_text(encoding="utf-8"), encoding="utf-8"
        )
    time.sleep(0.02)
    u.status = "running"
    u.backend_task_id = "task-GEN1"
    p1 = mission.checkpoint()
    st1 = load_state(p1)
    (ws / "gen1.state.json").write_text(
        Path(p1).read_text(encoding="utf-8"), encoding="utf-8"
    )
    if dag0.is_file():
        (ws / "gen1.dag.json").write_text(
            dag0.read_text(encoding="utf-8"), encoding="utf-8"
        )
    payload = {
        "gen0": st0.get("checkpoint_id"),
        "gen1": st1.get("checkpoint_id"),
        "gen0_status": st0["units"]["x"]["status"],
        "gen1_status": st1["units"]["x"]["status"],
        "dag0_has_checkpoint_id": "checkpoint_id"
        in json.loads((ws / "gen0.dag.json").read_text(encoding="utf-8")),
        "dag1_has_checkpoint_id": "checkpoint_id"
        in json.loads((ws / "gen1.dag.json").read_text(encoding="utf-8")),
    }
    (ws / "ids.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    sentinel.write_text("clean-exit", encoding="utf-8")
    raise SystemExit(0)


def _child_write_dag(ws: Path, marker: str, pad: str) -> Path:
    from hcli.dag_store import DagStore

    store = DagStore(ws)
    units = {"a": _wu("a", description="durable-dag")}
    store.save(units, extra={"marker": marker, "pad": pad})
    return store.path


def _child_write_mission(ws: Path, marker: str, pad: str) -> Path:
    from hcli.mission import Mission

    goal = f"{marker} {pad}"
    if not hasattr(_child_write_mission, "_mission"):
        u = _wu("a", description="durable-state")
        mission = Mission(
            ws,
            engine=_stub_engine(),
            units={"a": u},
            quiet=True,
            mission_id="midwrite-state",
            goal=goal,
            heartbeat_s=60,
            no_progress_threshold=100,
        )
        _child_write_mission._mission = mission  # type: ignore[attr-defined]
        _child_write_mission._unit = u  # type: ignore[attr-defined]
    mission = _child_write_mission._mission  # type: ignore[attr-defined]
    mission.goal = goal
    if marker == NEW_MARK:
        _child_write_mission._unit.status = "running"  # type: ignore[attr-defined]
        _child_write_mission._unit.backend_task_id = "task-GEN1"  # type: ignore[attr-defined]
    return mission.checkpoint()


def _child_write_session(ws: Path, marker: str, pad: str) -> Path:
    from hcli.session import Session, SessionStore

    store = SessionStore(str(ws))
    if not hasattr(_child_write_session, "_session"):
        session = Session(session_id="sess-crash", goal=f"{marker} {pad}")
        _child_write_session._session = session  # type: ignore[attr-defined]
    session = _child_write_session._session  # type: ignore[attr-defined]
    session.goal = f"{marker} {pad}"
    store.save(session)
    return Path(store.dir) / f"{session.id}.json"


def _child_write_config(ws: Path, marker: str, pad: str) -> Path:
    from hcli.config import Config

    cfg = Config(str(ws), global_path=str(ws / "global-unused.json"))
    cfg.save_project({"marker": marker, "pad": pad, "model": "/missing.gguf"})
    return Path(cfg.project_path)


def _child_write_steering(ws: Path, marker: str, pad: str) -> Path:
    from hcli.steering import SteeringQueue

    q = SteeringQueue(str(ws), "steer-crash")
    # enqueue always appends; for OLD we want a one-event file, for NEW a
    # different body. Rewrite via _events + _save so the live file is a
    # whole-generation replace, not an append-only log.
    from hcli.steering import SteerEvent

    q._events = [
        SteerEvent(
            id="steer-1",
            text=f"{marker} {pad}",
            session_id="steer-crash",
            timestamp=1.0,
            kind="knowledge",
        )
    ]
    q._save()
    return Path(q._path)


def _child_write_ledger(ws: Path, marker: str, pad: str) -> Path:
    from hcli.ledger import Ledger

    goal = ws / "GOAL.md"
    body = (
        f"- [ ] G001 — durable ledger | status: PENDING | risk: high | tier: V2\n"
        f"      acceptance: marker lands on disk\n"
        f"      verify: python3 -c 'raise SystemExit(0)'\n"
        f"      evidence: {marker} {pad}\n"
    )
    if not goal.is_file():
        goal.write_text(body, encoding="utf-8")
        led = Ledger.parse(goal)
        led.save()
        return goal
    led = Ledger.parse(goal)
    led.get("G001").evidence = f"{marker} {pad}"
    led._dirty = True
    led.save()
    return goal


def _child_write_verify_receipt(ws: Path, marker: str, pad: str) -> Path:
    from hcli.ledger import Ledger, VerifyResult

    goal = ws / "GOAL.md"
    if not goal.is_file():
        goal.write_text(
            "- [ ] G001 — receipt | status: PENDING | risk: high | tier: V2\n"
            "      acceptance: x\n"
            "      verify: python3 -c 'raise SystemExit(0)'\n"
            "      evidence: (none yet)\n",
            encoding="utf-8",
        )
    led = Ledger.parse(goal)
    ob = led.get("G001")
    # Marker lives in verify_command because the receipt JSON stores that
    # field verbatim and not the raw output.
    ob.verify_command = f"python3 -c 'raise SystemExit(0)' #{pad} {marker}"
    evidence = VerifyResult._fresh(
        passed=True,
        output=f"{marker} {pad}",
        exit_code=0,
        obligation_id="G001",
    )
    led._write_receipt(ob, evidence)
    dest = led._receipt_path("G001")
    assert dest is not None
    return dest


def _child_write_lock(ws: Path, marker: str, pad: str) -> Path:
    from hcli.resources import MutationLock

    lock = MutationLock(ws)
    lock.write(
        {
            "marker": marker,
            "pad": pad,
            "pid": os.getpid(),
            "unit_id": "lock-unit",
        }
    )
    assert lock.path is not None
    return lock.path


def _child_write_equilibrium(ws: Path, marker: str, pad: str) -> Path:
    from hcli.max_policy import save_equilibrium

    return save_equilibrium(
        ws, {"marker": marker, "pad": pad, "admitted": 1 if marker == OLD_MARK else 2}
    )


WRITERS: Dict[str, Dict[str, Any]] = {
    "dag.json": {
        "write": _child_write_dag,
        "inplace": False,
        "expect_json": True,
    },
    "mission/state.json": {
        "write": _child_write_mission,
        "inplace": False,
        "expect_json": True,
    },
    "sessions/<id>.json": {
        "write": _child_write_session,
        "inplace": False,
        "expect_json": True,
    },
    "config.json": {
        "write": _child_write_config,
        "inplace": False,
        "expect_json": True,
    },
    "steering/<id>.json": {
        "write": _child_write_steering,
        "inplace": False,
        "expect_json": True,
    },
    "GOAL.md": {
        "write": _child_write_ledger,
        "inplace": False,
        "expect_json": False,
    },
    "verify-receipts/G001.json": {
        "write": _child_write_verify_receipt,
        "inplace": False,
        "expect_json": True,
    },
    "mutation.lock": {
        "write": _child_write_lock,
        "inplace": False,
        "expect_json": True,
    },
    "max-equilibrium.json": {
        "write": _child_write_equilibrium,
        "inplace": False,
        "expect_json": True,
    },
}


def child_midwrite(writer: str, ws: Path, sentinel: Path) -> None:
    spec = WRITERS[writer]
    write = spec["write"]
    dest = write(ws, OLD_MARK, PAD_OLD)
    (ws / "dest.path").write_text(str(dest), encoding="utf-8")
    _install_midwrite_hook(sentinel, dest, inplace=bool(spec["inplace"]))
    write(ws, NEW_MARK, PAD_NEW)
    raise SystemExit(f"child was not killed mid-write of {writer}")


def child_naive_inplace(ws: Path, sentinel: Path) -> None:
    """Control: open(dest,'w') truncates live dest. This is what we watch break."""
    dest = ws / "naive.json"
    dest.write_text(
        json.dumps({"pad": PAD_OLD, "marker": OLD_MARK}, indent=2),
        encoding="utf-8",
    )
    (ws / "dest.path").write_text(str(dest), encoding="utf-8")
    _install_midwrite_hook(sentinel, dest, inplace=True)
    dest.write_text(
        json.dumps({"pad": PAD_NEW, "marker": NEW_MARK}, indent=2),
        encoding="utf-8",
    )
    raise SystemExit("child was not killed during naive in-place write")


def child_unsafe_mutation(ws: Path, sentinel: Path) -> None:
    """Engine._apply_operations writes the accepted source file in place."""
    from hcli.engine import Engine
    from hcli.events import EventBus
    from hcli.workspace import Workspace

    engine = Engine(
        workspace=Workspace(str(ws)),
        event_bus=EventBus(),
        runtime_count=1,
        model_name="/missing.gguf",
    )
    target = engine.root / "accepted_work.py"
    # Marker is at the END so a truncated prefix of the new generation
    # cannot be mistaken for NEW_COMPLETE.
    target.write_text(f"value = '{PAD_OLD}'\n# {OLD_MARK}\n", encoding="utf-8")
    (ws / "dest.path").write_text(str(target), encoding="utf-8")
    _install_midwrite_hook(sentinel, target, inplace=True)
    engine._apply_operations(
        [
            {
                "op": "replace_file",
                "path": "accepted_work.py",
                "new_text": f"value = '{PAD_NEW}'\n# {NEW_MARK}\n",
            }
        ]
    )
    raise SystemExit("child was not killed during mutation apply")


def child_unsafe_receipt(ws: Path, sentinel: Path) -> None:
    """Engine._write_receipt writes the acceptance receipt in place."""
    from hcli.engine import Engine
    from hcli.events import EventBus
    from hcli.workspace import Workspace

    engine = Engine(
        workspace=Workspace(str(ws)),
        event_bus=EventBus(),
        runtime_count=1,
        model_name="/missing.gguf",
    )
    path = engine._write_receipt(
        "accepted-goal",
        f"{OLD_MARK} {PAD_OLD}",
        {
            "status": "completed",
            "kind": "mutation",
            "operations": [{"op": "replace_file", "path": "accepted_work.py"}],
            "tests": ["test_accepted.py"],
        },
        [],
        {"ok": True, "reason": None},
        False,
    )
    dest = Path(path)
    (ws / "dest.path").write_text(str(dest), encoding="utf-8")
    _install_midwrite_hook(sentinel, dest, inplace=True)
    engine._write_receipt(
        "accepted-goal",
        f"{NEW_MARK} {PAD_NEW}",
        {
            "status": "completed",
            "kind": "mutation",
            "operations": [{"op": "replace_file", "path": "accepted_work.py"}],
            "tests": ["test_accepted.py"],
        },
        [],
        {"ok": True, "reason": None},
        False,
    )
    raise SystemExit("child was not killed during receipt write")


CHILD_MODES = {
    "between": child_between,
    "ids": child_checkpoint_ids,
    "naive": child_naive_inplace,
    "unsafe_mutation": child_unsafe_mutation,
    "unsafe_receipt": child_unsafe_receipt,
}


def run_child(argv: List[str]) -> int:
    mode = argv[0]
    ws = Path(argv[1])
    sentinel = Path(argv[2])
    writer = argv[3] if len(argv) > 3 else ""
    if mode == "midwrite":
        child_midwrite(writer, ws, sentinel)
        return 1
    fn = CHILD_MODES.get(mode)
    if fn is None:
        print(f"unknown child mode {mode!r}", file=sys.stderr)
        return 2
    fn(ws, sentinel)
    return 1


# ---------------------------------------------------------------------------
# Parent-side cases
# ---------------------------------------------------------------------------


def _unit_summary(blob: Any) -> Dict[str, Any]:
    if not isinstance(blob, dict):
        return {}
    out = {}
    for uid, payload in blob.items():
        if isinstance(payload, dict):
            out[str(uid)] = {
                "status": payload.get("status"),
                "backend_task_id": payload.get("backend_task_id"),
                "assigned_backend": payload.get("assigned_backend"),
            }
        else:
            out[str(uid)] = {
                "status": getattr(payload, "status", None),
                "backend_task_id": getattr(payload, "backend_task_id", None),
                "assigned_backend": getattr(payload, "assigned_backend", None),
            }
    return out


def _dispatched(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        uid: rec
        for uid, rec in summary.items()
        if rec.get("status") in {"running", "interrupted"}
        or rec.get("backend_task_id")
    }


def case_crash_between() -> None:
    name = "SIGKILL between DAG write and mission-state write"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ws = root / "ws"
        ws.mkdir()
        sentinel = root / "sentinel"
        proc = launch_child(["between", str(ws), str(sentinel)], sentinel)
        kill = sigkill_after_sentinel(proc, sentinel)
        dag_p = ws / ".hcli" / "dag.json"
        st_p = ws / ".hcli" / "mission" / "state.json"
        dag_raw = dag_p.read_bytes() if dag_p.is_file() else b""
        st_raw = st_p.read_bytes() if st_p.is_file() else b""
        dag = json.loads(dag_raw) if dag_raw else {}
        st = json.loads(st_raw) if st_raw else {}
        pre_dag = _unit_summary(dag.get("units") or {})
        pre_st = _unit_summary(st.get("units") or {})
        print("  pre-recovery DAG units:    ", json.dumps(pre_dag, sort_keys=True))
        print("  pre-recovery mission units:", json.dumps(pre_st, sort_keys=True))
        print("  pre-recovery mission in_flight:", st.get("in_flight"))
        print("  pre-recovery checkpoint_id:", st.get("checkpoint_id"))
        print("  pre-recovery DAG checkpoint_id:", dag.get("checkpoint_id"))

        files_disagree = _dispatched(pre_dag) != _dispatched(pre_st)
        if files_disagree:
            _watched(
                "crash between the two checkpoint writes left DAG and mission disagreeing",
                (
                    f"SIGKILL pid={kill['pid']} rc={kill['returncode']} "
                    f"sentinel={kill['saw_sentinel']}. "
                    "DAG has dispatched/running with backend_task_id=task-GEN1; "
                    "mission/state.json still has pending and no task id."
                ),
                f"DAG={pre_dag} mission={pre_st} in_flight={st.get('in_flight')}",
            )

        from hcli.mission import Mission, load_state

        recovered = Mission.from_workspace(
            ws,
            engine=_stub_engine(),
            quiet=True,
            heartbeat_s=60,
            no_progress_threshold=100,
        )
        rec_units = _unit_summary(recovered.scheduler.units)
        rec_disp = _dispatched(rec_units)
        dag_after = json.loads(dag_p.read_text(encoding="utf-8"))
        st_after = json.loads(st_p.read_text(encoding="utf-8"))
        post_dag = _unit_summary(dag_after.get("units") or {})
        post_st = _unit_summary(st_after.get("units") or {})
        print("  recovered in-memory units: ", json.dumps(rec_units, sort_keys=True))
        print("  post-recovery DAG units:   ", json.dumps(post_dag, sort_keys=True))
        print("  post-recovery mission units:", json.dumps(post_st, sort_keys=True))
        print("  recovered _last_checkpoint_id:", recovered._last_checkpoint_id)

        dispatched = recovered.scheduler.units["dispatched"]
        idle = recovered.scheduler.units["idle"]
        # Coherence: one generation, the DAG's. A mixture would reify the
        # mission's pending view of `dispatched` (no task id) alongside the
        # DAG's running view, which is how a Grok unit got launched twice.
        coherent = (
            kill["is_sigkill"]
            and kill["saw_sentinel"]
            and getattr(dispatched, "backend_task_id", None) == "task-GEN1"
            and dispatched.status in {"interrupted", "running", "ready"}
            and dispatched.status != "pending"
            and idle.status == "pending"
            and "dispatched" in rec_disp
            and "idle" not in rec_disp
        )
        # The recovered in-memory generation must match the DAG's dispatch
        # set, not a zip of DAG+mission.
        mixture = dispatched.status == "pending" and pre_dag.get("dispatched", {}).get(
            "status"
        ) == "running"
        detail = (
            f"kill={kill['returncode']} dispatched.status={dispatched.status} "
            f"task={getattr(dispatched, 'backend_task_id', None)} "
            f"idle={idle.status} mixture={mixture} files_disagree_pre={files_disagree} "
            f"files_disagree_post={_dispatched(post_dag) != _dispatched(post_st)}"
        )
        case = {
            "name": name,
            "verdict": "PASS" if coherent and not mixture else "FAIL",
            "sigkill": True,
            "kill": kill,
            "pre_recovery": {"dag": pre_dag, "mission": pre_st, "checkpoint_id": st.get("checkpoint_id")},
            "recovered": rec_units,
            "post_recovery": {"dag": post_dag, "mission": post_st},
            "coherent": coherent,
            "mixture": mixture,
            "detail": detail,
        }
        _record(case)
        check(name, coherent and not mixture, detail)
        # from_workspace persist()s the interrupted DAG but does not rewrite
        # state.json. That remaining split is a watched failure of recovery,
        # not of the DAG-first checkpoint order.
        if _dispatched(post_dag) != _dispatched(post_st):
            _watched(
                "Mission.from_workspace did not rewrite state.json to the recovered generation",
                (
                    "After SIGKILL-between and from_workspace, dag.json holds the "
                    "recovered (interrupted) unit with task-GEN1, but mission/state.json "
                    "still holds the previous generation (pending, no task id). "
                    "from_workspace never restores _last_checkpoint_id either "
                    f"(got {recovered._last_checkpoint_id!r})."
                ),
                f"DAG={post_dag} mission={post_st}",
            )


def case_midwrite_each_durable_file() -> None:
    for writer, spec in WRITERS.items():
        name = f"SIGKILL mid-write {writer}"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws = root / "ws"
            ws.mkdir()
            sentinel = root / "sentinel"
            proc = launch_child(
                ["midwrite", str(ws), str(sentinel), writer], sentinel
            )
            kill = sigkill_after_sentinel(proc, sentinel)
            dest_file = ws / "dest.path"
            dest = Path(dest_file.read_text(encoding="utf-8").strip()) if dest_file.is_file() else None
            raw = dest.read_bytes() if dest is not None and dest.is_file() else b""
            classified = classify_bytes(raw, expect_json=bool(spec["expect_json"]))
            ok = (
                kill["is_sigkill"]
                and kill["saw_sentinel"]
                and classified["verdict"] in {"OLD_INTACT", "NEW_COMPLETE"}
            )
            detail = (
                f"kill={kill['returncode']} sentinel={kill['saw_sentinel']} "
                f"verdict={classified['verdict']} bytes={classified['bytes']} "
                f"parse_error={classified['parse_error']} head={classified['head']!r}"
            )
            if not kill["is_sigkill"]:
                detail += f" stderr={kill['stderr_tail'][-300:]!r}"
            case = {
                "name": name,
                "writer": writer,
                "verdict": "PASS" if ok else "FAIL",
                "sigkill": True,
                "kill": {
                    "pid": kill["pid"],
                    "returncode": kill["returncode"],
                    "saw_sentinel": kill["saw_sentinel"],
                    "is_sigkill": kill["is_sigkill"],
                },
                "classified": classified,
                "detail": detail,
            }
            _record(case)
            check(name, ok, detail)


def case_checkpoint_id_distinguishes() -> None:
    name = "checkpoint_id distinguishes two generations"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ws = root / "ws"
        ws.mkdir()
        sentinel = root / "sentinel"
        proc = launch_child(["ids", str(ws), str(sentinel)], sentinel)
        # This child exits cleanly; do not SIGKILL it.
        try:
            stdout, stderr = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            os.kill(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate(timeout=5)
            check(name, False, "id-probe child timed out")
            return
        ids_path = ws / "ids.json"
        if not ids_path.is_file():
            check(
                name,
                False,
                f"ids.json missing rc={proc.returncode} stderr={stderr[-400:]!r}",
            )
            return
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        gen0 = json.loads((ws / "gen0.state.json").read_text(encoding="utf-8"))
        gen1 = json.loads((ws / "gen1.state.json").read_text(encoding="utf-8"))
        from hcli.mission import Mission, load_state
        from hcli.mission import mission_state_path

        recovered = Mission.from_workspace(
            ws,
            engine=_stub_engine(),
            quiet=True,
            heartbeat_s=60,
            no_progress_threshold=100,
        )
        named = load_state(mission_state_path(ws)).get("checkpoint_id")
        print("  gen0 checkpoint_id:", ids["gen0"])
        print("  gen1 checkpoint_id:", ids["gen1"])
        print("  gen0 unit status:  ", ids["gen0_status"])
        print("  gen1 unit status:  ", ids["gen1_status"])
        print("  recovered load_state checkpoint_id:", named)
        print("  recovered Mission._last_checkpoint_id:", recovered._last_checkpoint_id)
        print("  DAG carries checkpoint_id gen0/gen1:", ids["dag0_has_checkpoint_id"], ids["dag1_has_checkpoint_id"])
        distinguishable = (
            isinstance(ids["gen0"], str)
            and isinstance(ids["gen1"], str)
            and ids["gen0"] != ids["gen1"]
            and gen0["units"]["x"]["status"] != gen1["units"]["x"]["status"]
        )
        recovered_names = named == ids["gen1"]
        ok = distinguishable and recovered_names
        if not ids["dag0_has_checkpoint_id"] or not ids["dag1_has_checkpoint_id"]:
            _watched(
                "checkpoint_id is missing from dag.json, so a DAG-only reader cannot name the generation",
                (
                    "Mission.checkpoint() comments that both files carry the same "
                    "checkpoint_id. Measurement: state.json has it, dag.json does not. "
                    f"gen0={ids['gen0']} gen1={ids['gen1']}."
                ),
                f"dag0_has={ids['dag0_has_checkpoint_id']} dag1_has={ids['dag1_has_checkpoint_id']}",
            )
        if recovered._last_checkpoint_id is None:
            _watched(
                "Mission.from_workspace does not restore _last_checkpoint_id",
                (
                    "A reader that only holds the recovered Mission object cannot "
                    f"name the generation. load_state() can: {named!r}."
                ),
                f"_last_checkpoint_id={recovered._last_checkpoint_id!r} load_state={named!r}",
            )
        detail = (
            f"gen0={ids['gen0']} gen1={ids['gen1']} named={named} "
            f"distinguishable={distinguishable} recovered_names={recovered_names}"
        )
        case = {
            "name": name,
            "verdict": "PASS" if ok else "FAIL",
            "gen0": ids["gen0"],
            "gen1": ids["gen1"],
            "recovered_checkpoint_id": named,
            "dag_has_checkpoint_id": ids["dag1_has_checkpoint_id"],
            "from_workspace_restores_field": recovered._last_checkpoint_id is not None,
            "detail": detail,
        }
        _record(case)
        check(name, ok, detail)


def case_naive_inplace_watched_fail() -> None:
    name = "control: naive in-place write under SIGKILL is a truncated hybrid"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ws = root / "ws"
        ws.mkdir()
        sentinel = root / "sentinel"
        proc = launch_child(["naive", str(ws), str(sentinel)], sentinel)
        kill = sigkill_after_sentinel(proc, sentinel)
        dest = Path((ws / "dest.path").read_text(encoding="utf-8").strip())
        raw = dest.read_bytes() if dest.is_file() else b""
        classified = classify_bytes(raw, expect_json=True)
        broke = classified["verdict"] == "HYBRID_TRUNCATED" and kill["is_sigkill"]
        on_disk = (
            f"bytes={classified['bytes']} parse_error={classified['parse_error']} "
            f"head={classified['head']!r} tail={classified['tail']!r}"
        )
        _watched(
            "open(dest,'w') then SIGKILL truncates the live file into a hybrid",
            (
                "Control writer: Path.write_text / open('w') on the destination. "
                f"SIGKILL pid={kill['pid']} rc={kill['returncode']}. "
                "This is the failure atomic_write_json exists to prevent."
            ),
            on_disk,
        )
        case = {
            "name": name,
            "verdict": "WATCHED_FAIL" if broke else "FAIL",
            "classified": classified,
            "kill": {"pid": kill["pid"], "returncode": kill["returncode"]},
            "detail": on_disk,
        }
        _record(case)
        # The control must actually break. If it does not, the mid-write
        # hook is not interrupting a real write and the lane is vacuous.
        check(name, broke, on_disk)


def case_unsafe_writers() -> None:
    """Drive the two writers whose crash loses accepted work, not mission position.

    Ranked by consequence of a SIGKILL mid-write, measured this run:

    1. Engine._apply_operations (mutation.py / engine replace_file) writes the
       accepted source file with Path.write_text. A crash leaves a truncated
       hybrid of the work product itself. Highest: the accepted change is gone.
    2. Engine._write_receipt writes .hcli/receipts/<goal>.json the same way.
       A crash leaves unreadable evidence that the unit was accepted. The
       work product may still exist; the proof of acceptance does not.

    Not chosen (lower consequence, or loses position rather than work):
    session/steering/config (operator notes, not accepted work); mission
    state / dag (mission position — covered by the between-writes case);
    Ledger GOAL.md (atomic helper, crash does not truncate the live file).
    """
    ranked = [
        (
            "unsafe_mutation",
            "Engine._apply_operations in-place write of accepted source",
            False,
        ),
        (
            "unsafe_receipt",
            "Engine._write_receipt in-place write of acceptance receipt",
            True,
        ),
    ]
    for mode, title, expect_json in ranked:
        name = f"SIGKILL {title}"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws = root / "ws"
            ws.mkdir()
            sentinel = root / "sentinel"
            proc = launch_child([mode, str(ws), str(sentinel)], sentinel)
            kill = sigkill_after_sentinel(proc, sentinel)
            dest_file = ws / "dest.path"
            dest = (
                Path(dest_file.read_text(encoding="utf-8").strip())
                if dest_file.is_file()
                else None
            )
            raw = dest.read_bytes() if dest is not None and dest.is_file() else b""
            classified = classify_bytes(raw, expect_json=expect_json)
            on_disk = (
                f"path={dest} bytes={classified['bytes']} "
                f"verdict={classified['verdict']} "
                f"parse_error={classified['parse_error']} "
                f"has_old={classified['has_old']} has_new={classified['has_new']} "
                f"head={classified['head']!r} tail={classified['tail']!r}"
            )
            print(f"  ON DISK AFTER KILL ({mode}):")
            print(f"    {on_disk}")
            broke = (
                kill["is_sigkill"]
                and kill["saw_sentinel"]
                and classified["verdict"] == "HYBRID_TRUNCATED"
            )
            # These two writers were the finding: an in-place `write_text` of
            # the WORK PRODUCT and of the ACCEPTANCE RECEIPT, each measured
            # leaving a truncated hybrid under a real SIGKILL. They are now
            # atomic, so the correct outcome is OLD_INTACT or NEW_COMPLETE and
            # the only failure is a hybrid. Scoring `broke` as the pass would
            # keep demanding the defect and fail the repair -- the naive control
            # above still proves this harness can SEE a hybrid when one exists.
            # `classify_bytes` returns a mapping; its "verdict" key is the
            # classification. `on_disk` is the formatted string for the report.
            verdict = str((classified or {}).get("verdict") or "")
            hybrid = verdict not in ("OLD_INTACT", "NEW_COMPLETE")
            ok = not hybrid
            _watched(title, f"SIGKILL pid={kill['pid']} rc={kill['returncode']}", on_disk)
            case = {
                "name": name,
                "verdict": "PASS_ATOMIC" if ok else "FAIL_HYBRID",
                "was_previously": "truncated hybrid; repaired by routing both "
                                  "writers through _atomic_write_text",
                "rank_reason": title,
                "classified": classified,
                "on_disk": on_disk,
                "kill": {
                    "pid": kill["pid"],
                    "returncode": kill["returncode"],
                    "saw_sentinel": kill["saw_sentinel"],
                    "is_sigkill": kill["is_sigkill"],
                },
                "detail": on_disk,
            }
            _record(case)
            check(name, ok, on_disk if not ok else f"atomic: {verdict}")


def case_no_mid_token_resume_claim() -> None:
    name = "no claim that inference can resume mid-token"
    import re

    from hcli.dag_store import DagStore
    from hcli import workunit as workunit_mod

    policy = getattr(workunit_mod, "RESUME_POLICY", None)
    denials: List[str] = []
    positives: List[str] = []
    root = REPO / "hcli"
    for path in sorted(root.rglob("*.py")):
        if "tests/" in path.as_posix():
            continue
        text = path.read_text(encoding="utf-8")
        blob = re.sub(r"\s+", " ", text)
        lower = blob.lower()
        rel = str(path.relative_to(REPO))
        if "resume_from_token" in lower or "continue_from_token" in lower:
            positives.append(f"{rel}: identifier resume_from_token/continue_from_token")
        if re.search(r"resum\w* (an )?inference mid-token", lower):
            if re.search(
                r"(nothing here resumes|not resumed mid-token|"
                r"not something this system can do|cannot resume)",
                lower,
            ):
                denials.append(rel)
            else:
                positives.append(f"{rel}: positive mid-token resume claim")
        elif "mid-token" in lower and "resum" in lower:
            if re.search(
                r"(nothing here resumes|not resumed|not something this system can do)",
                lower,
            ):
                denials.append(rel)
            else:
                positives.append(f"{rel}: mid-token + resume without a denial")
    with tempfile.TemporaryDirectory() as tmp:
        wu = _wu("tok", status="running", attempts=1, assigned_runtime=0)
        store = DagStore(tmp)
        store.save({wu.id: wu})
        got = store.load(recover_running=True)["tok"]
        ctx = json.dumps(got.failure_context or {}, sort_keys=True).lower()
        rerun = (
            policy == "rerun"
            and got.status in {"interrupted", "ready"}
            and "mid-token" not in ctx
            and "resume_from_token" not in ctx
            and (got.failure_context or {}).get("resume_policy") == "rerun"
        )
    denials = sorted(set(denials))
    ok = rerun and not positives
    detail = (
        f"RESUME_POLICY={policy!r} recovered_status={got.status} "
        f"failure_context={got.failure_context} positive_claims={positives} "
        f"denials={denials}"
    )
    print("  RESUME_POLICY =", policy)
    print("  recovered running unit status =", got.status)
    print("  failure_context =", got.failure_context)
    print("  positive mid-token-resume claims in hcli (non-test) =", positives or "none")
    print("  named denials (recovery re-runs from the start) =", denials or "none")
    case = {
        "name": name,
        "verdict": "PASS" if ok else "FAIL",
        "resume_policy": policy,
        "recovered_status": got.status,
        "positive_claims": positives,
        "denials": denials,
        "detail": detail,
    }
    _record(case)
    check(name, ok, detail)
    print(
        "  named implication: the tree DENIES mid-token resume. "
        "workunit.RESUME_POLICY='rerun'; mark_interrupted and DagStore.load "
        "re-run a killed running unit from the start. Nothing claims a "
        "process can resume mid-token."
    )


def write_receipt(head: str) -> None:
    body = {
        "schema": "hawking.headless.hcli_crash_checkpoint.v1",
        "recorded_at": _now(),
        "git_head": head,
        "repo": str(REPO),
        "method": (
            "Parent-delivered SIGKILL of a real child Python process at an "
            "instrumented interruption point (sentinel file then os.kill(pid, SIGKILL)). "
            "An in-process exception is not used as a crash."
        ),
        "baseline_pytest_note": (
            "hcli/tests after tree-state load: 416 passed, 2 skipped "
            "(contract named 417 passed, 1 skipped)."
        ),
        "cases": CASES,
        "watched_fail": WATCHED_FAIL,
        "fails": list(FAILS),
        "verdict": "PASS" if not FAILS else "FAIL",
    }
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {RECEIPT_PATH}")


def main() -> int:
    FAILS.clear()
    CASES.clear()
    WATCHED_FAIL.clear()
    head = _git_head()
    print(f"git HEAD {head}")
    print(f"repo {REPO}")
    print()
    steps: List[Tuple[str, Callable[[], None]]] = [
        ("crash_between", case_crash_between),
        ("midwrite_each", case_midwrite_each_durable_file),
        ("checkpoint_id", case_checkpoint_id_distinguishes),
        ("naive_inplace", case_naive_inplace_watched_fail),
        ("unsafe_writers", case_unsafe_writers),
        ("no_mid_token", case_no_mid_token_resume_claim),
    ]
    for label, fn in steps:
        print(f"== {label} ==")
        try:
            fn()
        except Exception as exc:
            print(f"FAIL {label}: {type(exc).__name__}: {exc}")
            FAILS.append(f"{label}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        print()

    print("## WHAT I WATCHED FAIL")
    if not WATCHED_FAIL:
        print("FAIL: harness watched nothing break")
        FAILS.append("WHAT I WATCHED FAIL is empty")
    else:
        for i, rec in enumerate(WATCHED_FAIL, 1):
            print(f"{i}. {rec['title']}")
            print(f"   {rec['evidence']}")
            print(f"   on disk: {rec['on_disk']}")
    print()

    write_receipt(head)
    if FAILS:
        print(f"{len(FAILS)} FAILED")
        for item in FAILS:
            print("  " + item)
        return 1
    print("all hcli crash-checkpoint checks passed")
    return 0


def test_hcli_crash_checkpoint():
    rc = main()
    assert rc == 0, f"{len(FAILS)} crash-checkpoint checks failed: {FAILS}"


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        raise SystemExit(run_child(sys.argv[2:]))
    raise SystemExit(main())

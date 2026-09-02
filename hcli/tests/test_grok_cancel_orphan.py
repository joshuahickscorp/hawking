"""Real-process proofs that cancelled/orphaned Grok tasks are not leaked.

These tests spawn OS processes (a grok-run-like child that ignores HUP/INT).
An in-process mock would pass on the broken code: Mission.cancel never
reached a Grok pid, grok-run cleanup does not kill, and from_workspace
failed adopted live units. Each case was first run against the unmodified
tree; that failure log is the evidence, not a comment.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

from hcli.grok_bridge import GrokBridge, process_alive
from hcli.mission import Mission
from hcli.workunit import WorkUnit


TASK_ID = "consult-orphan-20260101-000000"

# Mirrors grok-run's background wrapper: `trap '' HUP INT; execute_task ... &`
_GROK_LIKE_CHILD = (
    "import os, signal, time\n"
    "signal.signal(signal.SIGHUP, signal.SIG_IGN)\n"
    "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
    "print(os.getpid(), flush=True)\n"
    "time.sleep(3600)\n"
)

# Wrapper plus a child in its own session: killing only the wrapper leaks the child.
_GROK_LIKE_TREE = (
    "import os, signal, subprocess, sys, time\n"
    "signal.signal(signal.SIGHUP, signal.SIG_IGN)\n"
    "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
    "child = subprocess.Popen(\n"
    "    [sys.executable, '-c', "
    "'import signal,time; signal.signal(signal.SIGHUP, signal.SIG_IGN); time.sleep(3600)'],\n"
    "    start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,\n"
    ")\n"
    "print(os.getpid(), child.pid, flush=True)\n"
    "time.sleep(3600)\n"
)

_FAKE_GROK_RUN = """#!/usr/bin/env python3
import sys
cmd = sys.argv[1] if len(sys.argv) > 1 else ""
if cmd == "status":
    sys.stdout.write("status: running (exit -)\\n")
    raise SystemExit(0)
if cmd == "cleanup":
    # Real grok-run cleanup removes a worktree; it does not kill the pid.
    sys.stdout.write("task artifacts kept — delete manually if you want them gone\\n")
    raise SystemExit(0)
sys.stderr.write("unexpected argv %r\\n" % (sys.argv,))
raise SystemExit(1)
"""

_PARENT_WRITER = r"""
import json, os, sys, subprocess, threading, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hcli.grok_bridge import GrokBridge
from hcli.mission import Mission
from hcli.workunit import WorkUnit

ws = Path(sys.argv[2])
pidfile = Path(sys.argv[3])
task_id = sys.argv[4]
child_src = sys.argv[5]
child = subprocess.Popen(
    [sys.executable, "-c", child_src],
    start_new_session=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True,
)
line = child.stdout.readline().strip()
sleep_pid = int(line or child.pid)
wu = WorkUnit(
    id="g1",
    role="work",
    description="orphaned grok unit",
    resource_class="GROK",
    preferred_backend="grok",
    assigned_backend="grok",
    backend_task_id=task_id,
    status="running",
    attempts=1,
    assigned_runtime=0,
)
mission = Mission(
    ws,
    units={"g1": wu},
    quiet=True,
    no_progress_threshold=100,
    mission_id="m-orphan",
)
bridge = GrokBridge(ws)
bridge.receipts_dir.mkdir(parents=True, exist_ok=True)
(bridge.receipts_dir / f"{task_id}.json").write_text(
    json.dumps(
        {
            "task_id": task_id,
            "mode": "consult",
            "launch_pid": sleep_pid,
            "status": {"state": "running", "exit_code": None},
            "workspace": str(ws),
        }
    ),
    encoding="utf-8",
)
with mission._lock:
    mission._inflight["g1"] = threading.current_thread()
mission.phase = "running"
mission.checkpoint()
pidfile.write_text(
    json.dumps({"parent": os.getpid(), "sleep": sleep_pid, "task_id": task_id}),
    encoding="utf-8",
)
pidfile.replace(pidfile)  # same path; directory entry is already durable
os.fsync(open(pidfile, "r").fileno())
time.sleep(3600)
"""


def _kill_pid(pid: int) -> None:
    if not pid:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(int(pid), sig)
        except OSError:
            try:
                os.kill(int(pid), sig)
            except OSError:
                pass
        try:
            os.waitpid(int(pid), os.WNOHANG)
        except OSError:
            pass
        if not process_alive(pid):
            return
        time.sleep(0.05)
    try:
        os.waitpid(int(pid), os.WNOHANG)
    except OSError:
        pass


def _spawn_grok_like() -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-c", _GROK_LIKE_CHILD],
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    line = proc.stdout.readline().strip() if proc.stdout is not None else ""
    try:
        proc._reported_pid = int(line)  # type: ignore[attr-defined]
    except (TypeError, ValueError):
        proc._reported_pid = proc.pid  # type: ignore[attr-defined]
    return proc


def _install_fake_grok_run(root: Path) -> Path:
    path = root / "fake-grok-run"
    path.write_text(_FAKE_GROK_RUN, encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_receipt(bridge: GrokBridge, task_id: str, pid: int) -> Path:
    bridge.receipts_dir.mkdir(parents=True, exist_ok=True)
    dest = bridge.receipts_dir / f"{task_id}.json"
    dest.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "mode": "consult",
                "launch_pid": pid,
                "status": {"state": "running", "exit_code": None},
                "workspace": str(bridge.workspace),
            }
        ),
        encoding="utf-8",
    )
    return dest


def _running_grok_unit(task_id: str = TASK_ID) -> WorkUnit:
    return WorkUnit(
        id="g1",
        role="work",
        description="live grok unit",
        resource_class="GROK",
        preferred_backend="grok",
        assigned_backend="grok",
        backend_task_id=task_id,
        status="running",
        attempts=1,
        assigned_runtime=0,
    )


class _ProcessCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.fake = _install_fake_grok_run(self.root)
        self._prev_grok = os.environ.get("GROK_RUN")
        os.environ["GROK_RUN"] = str(self.fake)
        self._pids: list[int] = []
        self._procs: list = []

    def tearDown(self):
        for proc in self._procs:
            pid = int(getattr(proc, "_reported_pid", proc.pid) or 0)
            _kill_pid(pid)
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        for pid in self._pids:
            _kill_pid(pid)
        leftover = [pid for pid in self._pids if process_alive(pid)]
        if self._prev_grok is None:
            os.environ.pop("GROK_RUN", None)
        else:
            os.environ["GROK_RUN"] = self._prev_grok
        self._tmpdir.cleanup()
        if leftover:
            raise AssertionError(f"test left processes running: {leftover}")

    def _track(self, pid: int) -> int:
        self._pids.append(int(pid))
        return int(pid)

    def _track_proc(self, proc: subprocess.Popen) -> int:
        self._procs.append(proc)
        pid = int(getattr(proc, "_reported_pid", proc.pid))
        return self._track(pid)


class TestGrokCancelKillsRealProcess(_ProcessCase):
    def test_bridge_cancel_kills_launch_pid(self):
        proc = _spawn_grok_like()
        pid = self._track_proc(proc)
        self.assertTrue(process_alive(pid), f"child pid {pid} died before cancel")
        bridge = GrokBridge(self.root)
        _write_receipt(bridge, TASK_ID, pid)
        print(f"BEFORE GrokBridge.cancel pid={pid} alive={process_alive(pid)}", flush=True)
        result = bridge.cancel(TASK_ID)
        alive = process_alive(pid)
        print(
            f"AFTER GrokBridge.cancel pid={pid} alive={alive} result={result}",
            flush=True,
        )
        self.assertFalse(alive, f"Grok launch pid {pid} still alive after cancel")
        self.assertFalse(result.get("process_alive"))
        self.assertEqual(result.get("state"), "cancelled")
        self.assertEqual(result.get("launch_pid"), pid)

    def test_cancel_kills_descendant_in_its_own_session(self):
        """grok-run's wrapper dying must not leave the grok child running."""
        proc = subprocess.Popen(
            [sys.executable, "-c", _GROK_LIKE_TREE],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        line = proc.stdout.readline().strip() if proc.stdout is not None else ""
        parts = line.split()
        self.assertEqual(len(parts), 2, f"wrapper did not print pids: {line!r}")
        wrapper_pid = int(parts[0])
        child_pid = int(parts[1])
        proc._reported_pid = wrapper_pid  # type: ignore[attr-defined]
        self._track_proc(proc)
        self._track(child_pid)
        self.assertTrue(process_alive(wrapper_pid))
        self.assertTrue(process_alive(child_pid))
        bridge = GrokBridge(self.root)
        _write_receipt(bridge, TASK_ID, wrapper_pid)
        print(
            f"BEFORE tree-cancel wrapper={wrapper_pid} child={child_pid} "
            f"wrapper_alive={process_alive(wrapper_pid)} child_alive={process_alive(child_pid)}",
            flush=True,
        )
        result = bridge.cancel(TASK_ID)
        print(
            f"AFTER tree-cancel wrapper_alive={process_alive(wrapper_pid)} "
            f"child_alive={process_alive(child_pid)} pids={result.get('pids')}",
            flush=True,
        )
        self.assertFalse(process_alive(wrapper_pid), f"wrapper {wrapper_pid} still alive")
        self.assertFalse(
            process_alive(child_pid),
            f"descendant {child_pid} still alive after cancelling wrapper {wrapper_pid}",
        )

    def test_mission_cancel_kills_grok_pid_not_in_child_pids(self):
        proc = _spawn_grok_like()
        pid = self._track_proc(proc)
        wu = _running_grok_unit()
        mission = Mission(
            self.root,
            units={wu.id: wu},
            quiet=True,
            no_progress_threshold=100,
        )
        bridge = GrokBridge(self.root)
        _write_receipt(bridge, TASK_ID, pid)
        self.assertNotIn(pid, mission.child_pids)
        self.assertTrue(process_alive(pid), f"child pid {pid} died before cancel")
        print(
            f"BEFORE Mission.cancel pid={pid} in_child_pids={pid in mission.child_pids} "
            f"alive={process_alive(pid)}",
            flush=True,
        )
        mission.cancel("test-cancel-grok")
        alive = process_alive(pid)
        print(
            f"AFTER Mission.cancel pid={pid} alive={alive} child_pids={sorted(mission.child_pids)}",
            flush=True,
        )
        self.assertFalse(
            alive,
            f"Grok pid {pid} still alive after Mission.cancel; "
            f"child_pids={sorted(mission.child_pids)}",
        )

    def test_wait_returns_cancelled_and_process_is_dead(self):
        proc = _spawn_grok_like()
        pid = self._track_proc(proc)
        bridge = GrokBridge(self.root)
        _write_receipt(bridge, TASK_ID, pid)
        holder: dict = {}

        def waiter():
            try:
                holder["status"] = bridge.wait(TASK_ID, timeout=8.0, poll_interval=0.05)
            except Exception as exc:
                holder["error"] = exc

        thread = threading.Thread(target=waiter, daemon=True)
        thread.start()
        time.sleep(0.15)
        print(f"BEFORE wait-cancel pid={pid} alive={process_alive(pid)}", flush=True)
        bridge.cancel(TASK_ID)
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive(), "wait() did not return after cancel")
        status = holder.get("status") or {}
        alive = process_alive(pid)
        print(
            f"AFTER wait-cancel pid={pid} alive={alive} wait_state={status.get('state')}",
            flush=True,
        )
        self.assertIn(status.get("state"), ("cancelled", "stale-running", "failed"))
        self.assertFalse(alive, f"pid {pid} still alive after wait returned")


class TestOrphanAdoption(_ProcessCase):
    def test_sigkill_parent_does_not_fail_live_task_on_load(self):
        ws = self.root / "ws"
        ws.mkdir()
        pidfile = self.root / "pids.json"
        env = os.environ.copy()
        env["GROK_RUN"] = str(self.fake)
        env["PYTHONPATH"] = str(REPO)
        env["HCLI_DISABLE_SIGNAL_HOOKS"] = "1"
        parent = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _PARENT_WRITER,
                str(REPO),
                str(ws),
                str(pidfile),
                TASK_ID,
                _GROK_LIKE_CHILD,
            ],
            cwd=str(REPO),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._procs.append(parent)
        self._track(parent.pid)
        deadline = time.time() + 15
        while time.time() < deadline and not pidfile.is_file():
            if parent.poll() is not None:
                self.fail(
                    "parent exited before writing pidfile: "
                    f"rc={parent.returncode} stderr={parent.stderr.read() if parent.stderr else ''}"
                )
            time.sleep(0.05)
        self.assertTrue(pidfile.is_file(), "parent never wrote pidfile")
        meta = json.loads(pidfile.read_text(encoding="utf-8"))
        sleep_pid = self._track(int(meta["sleep"]))
        parent_pid = int(meta["parent"])
        self.assertTrue(
            process_alive(sleep_pid),
            f"grok-like child {sleep_pid} died before parent SIGKILL",
        )
        os.kill(parent_pid, signal.SIGKILL)
        try:
            parent.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.kill(parent.pid, signal.SIGKILL)
            parent.wait(timeout=5)
        self.assertTrue(
            process_alive(sleep_pid),
            f"child {sleep_pid} did not survive parent SIGKILL — orphan setup is wrong",
        )
        print(
            f"AFTER parent SIGKILL parent={parent_pid} sleep={sleep_pid} "
            f"sleep_alive={process_alive(sleep_pid)}",
            flush=True,
        )
        mission = Mission.from_workspace(
            ws, quiet=True, runtime_count=1, no_progress_threshold=100
        )
        wu = mission.scheduler.units["g1"]
        bridge = GrokBridge(ws)
        status = bridge.status(TASK_ID)
        print(
            f"on load unit.status={wu.status} process_alive={status.get('process_alive')} "
            f"grok_state={status.get('state')} launch_pid={status.get('launch_pid')}",
            flush=True,
        )
        self.assertEqual(
            wu.status,
            "running",
            f"live orphan was failed on load: status={wu.status} "
            f"process_alive={status.get('process_alive')}",
        )
        self.assertTrue(status.get("process_alive"), status)
        self.assertTrue(process_alive(sleep_pid))
        self.assertEqual(status.get("launch_pid"), sleep_pid)

    def test_dead_orphan_is_failed_on_load(self):
        ws = self.root / "ws"
        ws.mkdir()
        proc = _spawn_grok_like()
        pid = self._track_proc(proc)
        wu = _running_grok_unit()
        mission = Mission(
            ws,
            units={wu.id: wu},
            quiet=True,
            no_progress_threshold=100,
            mission_id="m-dead-orphan",
        )
        _write_receipt(GrokBridge(ws), TASK_ID, pid)
        with mission._lock:
            mission._inflight["g1"] = threading.current_thread()
        mission.phase = "running"
        mission.checkpoint()
        _kill_pid(pid)
        self.assertFalse(process_alive(pid))
        loaded = Mission.from_workspace(
            ws, quiet=True, runtime_count=1, no_progress_threshold=100
        )
        # INTERRUPTED, not failed. Two lanes disagreed here and the stricter
        # reading wins: a process that was killed did not fail its verifier, so
        # it must not consume a retry. The rule is terminal-grok -> failed;
        # stale-running, unobservable, or a non-grok crash -> interrupted. This
        # orphan is stale-running (its status file says running, its process is
        # gone), so it is interrupted and still retryable.
        self.assertEqual(loaded.scheduler.units["g1"].status, "interrupted")
        status = GrokBridge(ws).status(TASK_ID)
        print(
            f"dead orphan unit.status={loaded.scheduler.units['g1'].status} "
            f"process_alive={status.get('process_alive')} state={status.get('state')}",
            flush=True,
        )
        self.assertFalse(status.get("process_alive"))


if __name__ == "__main__":
    unittest.main()

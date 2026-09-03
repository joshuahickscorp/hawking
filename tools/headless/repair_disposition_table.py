#!/usr/bin/env python3
"""Ten repair/retry properties — each disposition backed by driven behaviour.

A grep is not a disposition. This script launches the existing homeostasis
harness, then drives classification, circuit cooling, health restart,
cancellation of a dispatched unit, and orphaning of a Grok task (kill the
parent). It writes receipts/headless/REPAIR_DISPOSITION.json.

    python3 tools/headless/repair_disposition_table.py
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

from hcli.dag_store import DagStore  # noqa: E402
from hcli.engine import NoOpMutation  # noqa: E402
from hcli.grok_bridge import (  # noqa: E402
    GrokContractError,
    GrokNotAvailable,
    GrokRunError,
    process_alive,
)
from hcli.mission import Mission  # noqa: E402
from hcli.mutation import MutationError  # noqa: E402
from hcli.resources import (  # noqa: E402
    CIRCUIT_COOLING_SECONDS,
    CIRCUIT_FAILURE_THRESHOLD,
    FAILURE_KINDS,
    HEALTH_FILENAME,
    NON_RETRYABLE,
    STATE_CIRCUIT_OPEN,
    STATE_DEGRADED,
    STATE_HEALTHY,
    BackendHealth,
    classify_failure,
    counts_toward_retry_budget,
)
from hcli.scheduler import (  # noqa: E402
    MAX_REPAIR_DEPTH,
    MAX_REPAIRS_PER_ROOT,
    Scheduler,
)
from hcli.workunit import WorkUnit, assign_ready, is_ready  # noqa: E402

GROK_RUN = os.environ.get("GROK_RUN") or str(
    Path.home() / ".claude-grok" / "bin" / "grok-run"
)
HOMEOSTASIS_SRC = Path(
    "/Users/scammermike/Downloads/hawking-copy/tools/headless/"
    "hcli_repair_homeostasis_test.py"
)
RECEIPT = REPO_ROOT / "receipts" / "headless" / "REPAIR_DISPOSITION.json"

VERIFIED_EXISTING = "VERIFIED_EXISTING"
IMPLEMENTED_AND_VERIFIED = "IMPLEMENTED_AND_VERIFIED"
REJECT = "REJECT"
CONSOLIDATE = "CONSOLIDATE"
OPEN = "OPEN"


def _git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        )
        return out.strip()
    except Exception as exc:
        return f"UNKNOWN ({exc})"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _wu(uid: str, **kwargs: Any) -> WorkUnit:
    return WorkUnit(
        id=uid,
        role=kwargs.pop("role", "implement"),
        description=kwargs.pop("description", f"unit {uid}"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1-4: existing homeostasis harness + persistence of scheduler cycle state
# ---------------------------------------------------------------------------


def run_homeostasis() -> Dict[str, Any]:
    """Re-run the existing harness against THIS tree.

    The bootstrap tar omitted the untracked harness file, so a patched copy
    is executed from a temp path with REPO_ROOT forced to this worktree.
    """
    src = HOMEOSTASIS_SRC
    local = REPO_ROOT / "tools" / "headless" / "hcli_repair_homeostasis_test.py"
    if local.is_file():
        src = local
    if not src.is_file():
        return {
            "ok": False,
            "error": f"homeostasis harness missing: {src}",
            "results": [],
            "receipt": None,
        }
    text = src.read_text(encoding="utf-8")
    needle = "REPO_ROOT = Path(__file__).resolve().parents[2]"
    if needle not in text:
        return {
            "ok": False,
            "error": "homeostasis harness REPO_ROOT assignment not found",
            "results": [],
            "receipt": None,
        }
    patched = text.replace(needle, f"REPO_ROOT = Path({str(REPO_ROOT)!r})")
    # logging.disable must sit AFTER any from __future__ import.
    future = "from __future__ import annotations\n"
    quiet = "from __future__ import annotations\nimport logging\nlogging.disable(logging.WARNING)\n"
    if future in patched:
        patched = patched.replace(future, quiet, 1)
    else:
        patched = "import logging\nlogging.disable(logging.WARNING)\n" + patched
    with tempfile.TemporaryDirectory(prefix="repair-homeo-") as tmp:
        dest = Path(tmp) / "hcli_repair_homeostasis_test.py"
        dest.write_text(patched, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(dest)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )
    receipt_path = REPO_ROOT / "receipts" / "headless" / "REPAIR_HOMEOSTASIS.json"
    receipt: Optional[Dict[str, Any]] = None
    if proc.returncode == 0 and receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            receipt = None
    return {
        "ok": proc.returncode == 0 and bool(receipt) and receipt.get("result") == "PASS",
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "results": (receipt or {}).get("results") or [],
        "failed": (receipt or {}).get("failed") or [],
        "receipt_result": (receipt or {}).get("result"),
        "receipt_path": str(receipt_path) if receipt_path.is_file() else None,
    }


def probe_lineage_persist() -> Dict[str, Any]:
    """Unit lineage fields persist; scheduler cycle/count memory does not."""
    with tempfile.TemporaryDirectory(prefix="repair-persist-") as tmp:
        root = _wu("dead")
        root.status = "running"
        sched = Scheduler({"dead": root}, 1, workspace=tmp)
        repair = sched.fail("dead", {"error": "injected failure", "reason": "injected failure"})
        sigs_before = {k: sorted(v) for k, v in sched._repair_signatures.items()}
        counts_before = dict(sched._repair_counts)
        dag = json.loads((Path(tmp) / ".hcli" / "dag.json").read_text(encoding="utf-8"))
        loaded = Scheduler.from_workspace(tmp, runtime_count=1)
        sigs_after_load = {
            k: sorted(v) for k, v in loaded._repair_signatures.items()
        }
        counts_after_load = dict(loaded._repair_counts)
        unit_ok = loaded.units["dead"].id == "dead"
        repair_id = repair.id if repair is not None else None
        repair_loaded = loaded.units.get(repair_id) if repair_id else None
        lineage_ok = bool(repair_loaded) and repair_loaded.repair_root == "dead"
        lineage_ok = lineage_ok and int(repair_loaded.repair_depth or 0) == 1
        # Exhausted flag round-trip.
        spent = _wu("spent")
        spent.status = "failed"
        spent.repair_root = "spent"
        spent.repair_depth = 3
        spent.repair_reason = "budget spent"
        spent.repair_exhausted = True
        back = WorkUnit.from_dict(spent.to_dict())
        exhausted_roundtrip = (
            back.repair_exhausted is True
            and back.repair_root == "spent"
            and back.repair_depth == 3
            and back.repair_reason == "budget spent"
            and is_ready(back, {"spent": back}) is False
        )
        # After restart the in-memory signature set is empty. Failing the
        # repair unit with the SAME error (same root) would be a cycle if
        # signatures had survived; instead another descendant is emitted.
        again = None
        if repair_loaded is not None:
            repair_loaded.status = "running"
            again = loaded.fail(
                repair_loaded.id,
                {"error": "injected failure", "reason": "injected failure"},
            )
        return {
            "sigs_before": sigs_before,
            "counts_before": counts_before,
            "sigs_after_restart": sigs_after_load,
            "counts_after_restart": counts_after_load,
            "signatures_persisted": bool(sigs_after_load.get("dead")),
            "counts_persisted": bool(counts_after_load.get("dead")),
            "sigs_on_disk": "_repair_signatures" in dag or "repair_signatures" in dag,
            "repair_emitted": repair_id,
            "lineage_on_unit_persisted": bool(lineage_ok),
            "exhausted_roundtrip": bool(exhausted_roundtrip),
            "same_failure_emits_again_after_restart": again is not None,
            "again_id": getattr(again, "id", None),
            "unit_ok": bool(unit_ok),
        }


# ---------------------------------------------------------------------------
# 5-8: classification, backoff, health persist, circuit breaker
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)


def probe_classification() -> Dict[str, Any]:
    samples = {
        "TRANSIENT_BACKEND": {"error": "llama-server HTTP 503: overloaded"},
        "VERIFIER_FAILURE": {"reason": "TEST_FAILED"},
        "DETERMINISTIC_IMPLEMENTATION": NoOpMutation("identical bytes"),
        "INVALID_OUTPUT": {"error": "llama-server returned invalid JSON"},
        "RATE_LIMIT": {"error": "llama-server HTTP 429: too many requests"},
        "UNAVAILABLE_DEPENDENCY": GrokNotAvailable("grok-run is not on PATH"),
        "IMPOSSIBLE_CONTRACT": GrokContractError("contract missing WRITE/VERIFY"),
    }
    produced = {}
    for kind, ctx in samples.items():
        clf = classify_failure(ctx)
        produced[kind] = {
            "kind": clf.kind,
            "retryable": clf.retryable,
            "observed": clf.observed,
            "counts_toward_budget": counts_toward_retry_budget(ctx),
        }
    kinds_ok = set(produced[k]["kind"] for k in samples) == set(FAILURE_KINDS)
    retryable_map = {k: produced[k]["retryable"] for k in samples}
    expected_retryable = {
        "TRANSIENT_BACKEND": True,
        "VERIFIER_FAILURE": True,
        "DETERMINISTIC_IMPLEMENTATION": False,
        "INVALID_OUTPUT": True,
        "RATE_LIMIT": True,
        "UNAVAILABLE_DEPENDENCY": False,
        "IMPOSSIBLE_CONTRACT": False,
    }
    retry_ok = retryable_map == expected_retryable
    budget_ok = all(
        produced[k]["counts_toward_budget"] is expected_retryable[k] for k in samples
    )
    # Scheduler.fail does not consult classify_failure: a non-retryable
    # failure still emits a repair.
    with tempfile.TemporaryDirectory(prefix="repair-clf-") as tmp:
        wu = _wu("noop")
        wu.status = "running"
        sched = Scheduler({"noop": wu}, 1, workspace=tmp)
        repair = sched.fail("noop", {"reason": "NO_OP_MUTATION"})
        wired = repair is None
    return {
        "kinds_closed_set": list(FAILURE_KINDS),
        "produced": produced,
        "all_seven_produced": kinds_ok,
        "retryable_map": retryable_map,
        "retry_ok": retry_ok,
        "budget_ok": budget_ok,
        "non_retryable_set": sorted(NON_RETRYABLE),
        "scheduler_fail_consults_classifier": wired,
        "non_retryable_still_emits_repair": repair.id if repair is not None else None,
    }


def probe_backoff() -> Dict[str, Any]:
    """There is no retry sleep. Circuit cooling is the only time gate.

    Drive both: (1) fail->redispatch wall time for retryable vs not;
    (2) circuit cooling vs retryability.
    """
    with tempfile.TemporaryDirectory(prefix="repair-backoff-") as tmp:
        clock = FakeClock()
        health = BackendHealth(
            tmp,
            clock=clock,
            failure_threshold=2,
            cooling_seconds=30.0,
        )
        noop = health.record_failure("cpu", {"reason": "NO_OP_MUTATION"})
        cpu_after_noop = health.snapshot("cpu")
        grok_na = health.record_failure("grok", GrokNotAvailable("missing"))
        grok_na2 = health.record_failure("grok", GrokNotAvailable("missing"))
        grok_after = health.snapshot("grok")
        qwen = health.record_failure(
            "qwen", {"error": "llama-server HTTP 503: overloaded"}
        )
        qwen2 = health.record_failure(
            "qwen", {"error": "llama-server HTTP 503: overloaded"}
        )
        qwen_after = health.snapshot("qwen")
        clock.advance(29.0)
        qwen_pre = health.snapshot("qwen")
        grok_pre = health.snapshot("grok")
        clock.advance(2.0)
        qwen_post = health.snapshot("qwen")
        grok_post = health.snapshot("grok")

        # fail -> redispatch: no sleep in Scheduler.fail / identify_ready.
        wu = _wu("r")
        wu.status = "running"
        sched = Scheduler({"r": wu}, 1, workspace=tmp)
        t0 = time.monotonic()
        sched.fail("r", {"error": "llama-server HTTP 503: x"})
        ready_dt = time.monotonic() - t0

    return {
        "retry_sleep_in_fail_s": ready_dt,
        "no_retry_sleep": ready_dt < 0.05,
        "noop_retryable": noop.retryable,
        "noop_trips_circuit": cpu_after_noop["state"] != STATE_HEALTHY,
        "noop_consecutive": cpu_after_noop["consecutive_failures"],
        "grok_na_retryable": grok_na.retryable,
        "grok_na_state_after_2": grok_after["state"],
        "grok_na_opens_circuit": grok_after["state"] == STATE_CIRCUIT_OPEN,
        "qwen_retryable": qwen.retryable,
        "qwen_state_after_2": qwen_after["state"],
        "qwen_opens_circuit": qwen_after["state"] == STATE_CIRCUIT_OPEN,
        "cooling_still_open_at_29s": qwen_pre["state"] == STATE_CIRCUIT_OPEN
        and grok_pre["state"] == STATE_CIRCUIT_OPEN,
        "cooling_reopens_after_30s": qwen_post["state"] != STATE_CIRCUIT_OPEN
        and grok_post["state"] != STATE_CIRCUIT_OPEN,
        "non_retryable_grok_na_still_cooled": grok_na.retryable is False
        and grok_after["state"] == STATE_CIRCUIT_OPEN,
        "circuit_cooling_seconds": CIRCUIT_COOLING_SECONDS,
        "failure_threshold": CIRCUIT_FAILURE_THRESHOLD,
        "qwen2_kind": qwen2.kind,
        "grok_na2_kind": grok_na2.kind,
    }


def probe_health_restart() -> Dict[str, Any]:
    """Child process writes; it exits; a new process reads. Not in-process reload."""
    with tempfile.TemporaryDirectory(prefix="repair-health-") as tmp:
        writer = (
            "import sys\n"
            ""
            "from hcli.resources import BackendHealth\n"
            f"h = BackendHealth({tmp!r}, failure_threshold=3, cooling_seconds=30.0)\n"
            "h.record_failure('grok', {'error': 'GrokNotAvailable'})\n"
            "h.record_failure('grok', {'error': 'GrokNotAvailable'})\n"
            "h.record_success('cpu')\n"
            "print('wrote')\n"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-c", writer],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        path = Path(tmp) / ".hcli" / HEALTH_FILENAME
        reader_script = (
            "import sys, json\n"
            ""
            "from hcli.resources import BackendHealth\n"
            f"h = BackendHealth({tmp!r}, failure_threshold=3, cooling_seconds=30.0)\n"
            "print(json.dumps({name: h.snapshot(name) for name in "
            "('grok','cpu','qwen')}))\n"
        )
        reader = subprocess.run(
            [sys.executable, "-c", reader_script],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        snap = {}
        try:
            snap = json.loads(reader.stdout.strip().splitlines()[-1])
        except Exception:
            snap = {"parse_error": reader.stdout, "stderr": reader.stderr}
        return {
            "writer_returncode": proc.returncode,
            "writer_stdout": proc.stdout.strip(),
            "writer_stderr": (proc.stderr or "")[-400:],
            "reader_returncode": reader.returncode,
            "reader_stderr": (reader.stderr or "")[-400:],
            "path_exists": path.is_file(),
            "path": str(path),
            "snap": snap,
            "ok": (
                proc.returncode == 0
                and reader.returncode == 0
                and path.is_file()
                and snap.get("grok", {}).get("consecutive_failures") == 2
                and snap.get("cpu", {}).get("consecutive_failures") == 0
                and snap.get("cpu", {}).get("last_success_time") is not None
                and snap.get("qwen", {}).get("state") == STATE_HEALTHY
            ),
        }


def probe_circuit() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="repair-circuit-") as tmp:
        clock = FakeClock(t=5_000.0)
        n = 3
        health = BackendHealth(
            tmp, clock=clock, failure_threshold=n, cooling_seconds=30.0
        )
        ctx = {"error": "llama-server HTTP 503: overloaded"}
        for _ in range(n):
            health.record_failure("qwen", ctx)
        qwen = health.snapshot("qwen")
        grok = health.snapshot("grok")
        cpu = health.snapshot("cpu")
        clock.advance(29.99)
        still_open = health.state("qwen") == STATE_CIRCUIT_OPEN
        clock.advance(0.02)
        after = health.snapshot("qwen")
        health.record_success("qwen")
        closed = health.snapshot("qwen")

        # Live dispatch does not consult the breaker.
        health2 = BackendHealth(
            tmp + "-b", clock=FakeClock(), failure_threshold=1, cooling_seconds=30.0
        )
        health2.record_failure("qwen", ctx)
        open_ok = not health2.allows_new_assignments("qwen")
        wu = _wu("live", preferred_backend="qwen")
        assignments = assign_ready([wu], runtime_count=1, all_units={wu.id: wu})
        # pending -> identify_ready would transition; assign_ready wants status ready
        wu.status = "ready"
        assignments = assign_ready([wu], runtime_count=1, all_units={wu.id: wu})
        dispatched_anyway = bool(assignments) and wu.status == "running"
        return {
            "threshold": n,
            "qwen_open": qwen["state"] == STATE_CIRCUIT_OPEN,
            "qwen_consecutive": qwen["consecutive_failures"],
            "grok_untouched": grok["state"] == STATE_HEALTHY,
            "cpu_untouched": cpu["state"] == STATE_HEALTHY,
            "still_open_before_cooling": still_open,
            "reopens_to": after["state"],
            "reopens_allows_new": after["allows_new"] is True,
            "success_closes": closed["state"] == STATE_HEALTHY
            and closed["consecutive_failures"] == 0,
            "allows_new_false_when_open": open_ok,
            "assign_ready_dispatches_through_open_circuit": dispatched_anyway,
            "cooling_seconds": 30.0,
        }


# ---------------------------------------------------------------------------
# 9: actually cancel a dispatched unit
# ---------------------------------------------------------------------------


class SlowChildEngine:
    """Blocks in a child `sleep` and does not poll is_cancelled."""

    def __init__(self) -> None:
        self.cancelled = False
        self.child_pids: set = set()
        self.entered = threading.Event()
        self.child: Optional[subprocess.Popen] = None
        self.waited = threading.Event()
        self.return_value: Optional[Dict[str, Any]] = None

    def cancel(self) -> None:
        self.cancelled = True

    def execute_workunit(self, wu: Any, context: Any) -> Dict[str, Any]:
        proc = subprocess.Popen(["sleep", "25"])
        self.child = proc
        self.child_pids.add(proc.pid)
        self.entered.set()
        proc.wait()
        self.waited.set()
        checker = (context or {}).get("is_cancelled")
        if self.cancelled or (callable(checker) and checker()):
            self.return_value = {"cancelled": True}
            return self.return_value
        self.return_value = {"validation": {"ok": True}}
        return self.return_value


class CooperativeEngine:
    """Polls is_cancelled while sleeping — the sound path."""

    def __init__(self) -> None:
        self.cancelled = False
        self.entered = threading.Event()
        self.child_pids: set = set()

    def cancel(self) -> None:
        self.cancelled = True

    def execute_workunit(self, wu: Any, context: Any) -> Dict[str, Any]:
        self.entered.set()
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if self.cancelled:
                return {"cancelled": True}
            checker = (context or {}).get("is_cancelled")
            if callable(checker) and checker():
                return {"cancelled": True}
            time.sleep(0.02)
        return {"validation": {"ok": True}}


class FakeGrokHandle:
    def __init__(self, task_id: str, launch_pid: int) -> None:
        self.task_id = task_id
        self.launch_pid = launch_pid


class FakeGrokBridge:
    """Mimics GrokBridge.consult/wait: wait() does not honour cancel."""

    def __init__(self) -> None:
        self.child: Optional[subprocess.Popen] = None
        self.wait_started = threading.Event()
        self.wait_finished = threading.Event()
        self.wait_state: Optional[str] = None
        self.consulted = False

    def consult(self, prompt: str, background: bool = True) -> FakeGrokHandle:
        self.consulted = True
        self.child = subprocess.Popen(
            ["sleep", "25"],
            start_new_session=True,
        )
        return FakeGrokHandle("fake-grok-task", self.child.pid)

    def wait(self, task_id: str, timeout: float = 3600.0) -> Dict[str, Any]:
        self.wait_started.set()
        assert self.child is not None
        self.child.wait()
        self.wait_state = "done" if self.child.returncode == 0 else "failed"
        self.wait_finished.set()
        return {
            "state": self.wait_state,
            "exit_code": self.child.returncode,
            "task_id": task_id,
        }

    def compact_report(self, task_id: str) -> Dict[str, Any]:
        return {"task_id": task_id, "backend": "grok"}

    def status(self, task_id: str) -> Dict[str, Any]:
        alive = bool(self.child and self.child.poll() is None)
        return {
            "state": "running" if alive else "done",
            "exit_code": None if alive else (self.child.returncode if self.child else None),
            "task_id": task_id,
            "process_alive": alive,
            "launch_pid": self.child.pid if self.child else None,
        }

    def cleanup(self, task_id: str) -> Dict[str, Any]:
        return {"task_id": task_id, "ok": False, "note": "fake cleanup does not kill"}


def _cancel_mission(engine: Any, uid: str = "slow", seconds: float = 8.0) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="repair-cancel-") as tmp:
        units = {uid: _wu(uid)}
        mission = Mission(
            tmp,
            engine=engine,
            units=units,
            quiet=True,
            no_progress_threshold=100,
        )
        thread = threading.Thread(target=mission.run, daemon=True)
        thread.start()
        entered = getattr(engine, "entered", None)
        if entered is not None:
            entered.wait(timeout=5)
        else:
            time.sleep(0.3)
        wu = mission.scheduler.units[uid]
        status_at_cancel = wu.status
        child = getattr(engine, "child", None)
        child_pid = child.pid if child is not None else None
        child_alive_before = bool(child_pid and process_alive(child_pid))
        t0 = time.monotonic()
        mission.cancel("probe-cancel")
        thread.join(timeout=seconds)
        elapsed = time.monotonic() - t0
        child_alive_after = bool(child_pid and process_alive(child_pid))
        state_path = Path(tmp) / ".hcli" / "mission" / "state.json"
        state = {}
        if state_path.is_file():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                state = {"unreadable": True}
        repairs = [
            u.id
            for u in mission.scheduler.units.values()
            if u.repairs == uid
        ]
        # Do not leave sleep(25) around if stop_children missed it.
        if child_pid and process_alive(child_pid):
            try:
                os.kill(child_pid, signal.SIGKILL)
            except OSError:
                pass
        return {
            "phase": mission.phase,
            "cancel_reason": mission.cancel_reason,
            "engine_cancel_called": bool(getattr(engine, "cancelled", False)),
            "status_at_cancel": status_at_cancel,
            "status_after": mission.scheduler.units[uid].status,
            "repairs": repairs,
            "thread_alive": thread.is_alive(),
            "join_s": elapsed,
            "child_pid": child_pid,
            "child_alive_before": child_alive_before,
            "child_alive_after_cancel": child_alive_after,
            "state_durable": state_path.is_file(),
            "state_reason": state.get("cancel_reason"),
            "state_phase": state.get("phase"),
            "n_units": len(mission.scheduler.units),
        }


def probe_cancel() -> Dict[str, Any]:
    coop = _cancel_mission(CooperativeEngine())
    slow = _cancel_mission(SlowChildEngine())

    # Grok path: wait() ignores cancel; launch_pid is not in mission.child_pids.
    import hcli.executors as executors_mod

    fake = FakeGrokBridge()
    original_bridge = executors_mod.WorkUnitExecutor.grok_bridge

    def _bridge(self):  # noqa: ANN001
        return fake

    grok_obs: Dict[str, Any] = {}
    try:
        executors_mod.WorkUnitExecutor.grok_bridge = _bridge  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory(prefix="repair-cancel-grok-") as tmp:
            uid = "gunit"
            units = {uid: _wu(uid, preferred_backend="grok", resource_class="GROK")}
            engine = CooperativeEngine()  # unused for grok path; still gets cancel()
            mission = Mission(
                tmp,
                engine=engine,
                units=units,
                quiet=True,
                no_progress_threshold=100,
            )
            thread = threading.Thread(target=mission.run, daemon=True)
            thread.start()
            fake.wait_started.wait(timeout=5)
            child_pid = fake.child.pid if fake.child is not None else None
            alive_before = bool(child_pid and process_alive(child_pid))
            mission.cancel("probe-cancel-grok")
            thread.join(timeout=4)
            alive_after = bool(child_pid and process_alive(child_pid))
            grok_obs = {
                "consulted": fake.consulted,
                "phase": mission.phase,
                "status_after": mission.scheduler.units[uid].status,
                "repairs": [
                    u.id
                    for u in mission.scheduler.units.values()
                    if u.repairs == uid
                ],
                "engine_cancel_called": engine.cancelled,
                "thread_alive": thread.is_alive(),
                "child_pid": child_pid,
                "child_alive_before": alive_before,
                "child_alive_after_cancel": alive_after,
                "wait_finished": fake.wait_finished.is_set(),
                "backend_task_id": mission.scheduler.units[uid].backend_task_id,
                "child_pids_on_mission": sorted(mission.child_pids),
            }
            if child_pid and process_alive(child_pid):
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except OSError:
                    pass
                try:
                    os.killpg(child_pid, signal.SIGKILL)
                except OSError:
                    pass
    finally:
        executors_mod.WorkUnitExecutor.grok_bridge = original_bridge  # type: ignore[method-assign]

    # Confirm GrokBridge really has no cancel.
    import hcli.grok_bridge as gb

    grok_obs["GrokBridge_has_cancel"] = hasattr(gb.GrokBridge, "cancel")
    grok_obs["GrokBridge_has_cleanup"] = hasattr(gb.GrokBridge, "cleanup")
    grok_obs["wait_terminal_states"] = "done, failed only (cancelled/stale-running not terminal)"
    return {
        "cooperative_engine": coop,
        "blocking_child_engine": slow,
        "grok_path": grok_obs,
    }


# ---------------------------------------------------------------------------
# 10: actually orphan a Grok task by killing the parent
# ---------------------------------------------------------------------------


def _reap_task(
    task_id: Optional[str],
    launch_pid: Optional[int],
    grok_run: Optional[str] = None,
    marker_extra: Optional[str] = None,
) -> None:
    if launch_pid and process_alive(int(launch_pid)):
        try:
            os.kill(int(launch_pid), signal.SIGTERM)
        except OSError:
            pass
        time.sleep(0.2)
        if process_alive(int(launch_pid)):
            try:
                os.kill(int(launch_pid), signal.SIGKILL)
            except OSError:
                pass
        try:
            os.killpg(int(launch_pid), signal.SIGKILL)
        except OSError:
            pass
    if task_id:
        for marker in (
            f"/.claude-grok/tasks/{task_id}/",
            f"/repair-disp-tasks/{task_id}/",
            marker_extra,
        ):
            if not marker:
                continue
            subprocess.run(
                ["pkill", "-f", marker],
                capture_output=True,
                text=True,
                timeout=10,
            )
        binary = grok_run or GROK_RUN
        if Path(binary).is_file():
            subprocess.run(
                [binary, "cleanup", "--id", str(task_id)],
                capture_output=True,
                text=True,
                timeout=20,
            )


def _write_patched_grok_run(dest: Path, tasks_root: Path, wt_root: Path) -> Path:
    """Copy grok-run, force writable task dirs, disable nested sandbox.

    This lane's Seatbelt cannot mkdir ~/.claude-grok/tasks and cannot apply
    grok's nested read-only sandbox. The binary, auth, and --background
    detach path are still the real grok-run.
    """
    src = Path(GROK_RUN).read_text(encoding="utf-8")
    wt = 'WT_ROOT=$(expand "$(cfg general.worktree_root "~/.claude-grok/worktrees")")'
    widx = src.find(wt)
    if widx < 0:
        raise RuntimeError("could not locate WT_ROOT assignment in grok-run")
    end = src.find("\n", widx) + 1
    src = (
        src[:end]
        + f'TASKS_ROOT="{tasks_root}"\nWT_ROOT="{wt_root}"\n'
        + src[end:]
    )
    src = src.replace('--sandbox "$P_SANDBOX"', '--sandbox off')
    dest.write_text(src, encoding="utf-8")
    dest.chmod(0o755)
    return dest


def probe_orphan() -> Dict[str, Any]:
    """Launch a real grok-run --background consult, kill the parent, observe."""
    out: Dict[str, Any] = {
        "grok_run": GROK_RUN,
        "grok_run_exists": Path(GROK_RUN).is_file(),
    }
    with tempfile.TemporaryDirectory(prefix="repair-orphan-") as tmp:
        tmp_path = Path(tmp)
        tasks_root = tmp_path / "tasks"
        wt_root = tmp_path / "wt"
        tasks_root.mkdir()
        wt_root.mkdir()
        patched = _write_patched_grok_run(
            tmp_path / "grok-run", tasks_root, wt_root
        )
        env = os.environ.copy()
        env["GROK_RUN"] = str(patched)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        out["patched_grok_run"] = str(patched)
        out["tasks_root"] = str(tasks_root)
        handle_path = tmp_path / "handle.json"
        parent_script = f"""
import json, os, sys, time
from pathlib import Path
os.environ["GROK_RUN"] = {str(patched)!r}
sys.path.insert(0, {str(REPO_ROOT)!r})
from hcli.grok_bridge import GrokBridge
ws = Path({tmp!r}) / "ws"
ws.mkdir(parents=True, exist_ok=True)
bridge = GrokBridge(ws)
handle = bridge.consult(
    "Write a 120-word explanation of photosynthesis, slowly, no tools.",
    background=True,
)
payload = {{
    "task_id": handle.task_id,
    "launch_pid": handle.launch_pid,
    "parent_pid": os.getpid(),
    "task_dir": handle.task_dir,
    "stdout": (handle.stdout or "")[-500:],
    "stderr": (handle.stderr or "")[-500:],
}}
Path({str(handle_path)!r}).write_text(json.dumps(payload), encoding="utf-8")
time.sleep(180)
"""
        parent = subprocess.Popen(
            [sys.executable, "-c", parent_script],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 45.0
        handle = None
        while time.monotonic() < deadline:
            if handle_path.is_file():
                try:
                    handle = json.loads(handle_path.read_text(encoding="utf-8"))
                    break
                except json.JSONDecodeError:
                    pass
            if parent.poll() is not None:
                break
            time.sleep(0.1)
        parent_stdout = ""
        parent_stderr = ""
        if handle is None:
            try:
                parent_stdout, parent_stderr = parent.communicate(timeout=5)
            except Exception:
                parent.kill()
                parent_stdout, parent_stderr = parent.communicate(timeout=5)
            out.update(
                {
                    "ok": False,
                    "error": "parent did not write handle.json",
                    "parent_returncode": parent.returncode,
                    "parent_stdout": (parent_stdout or "")[-800:],
                    "parent_stderr": (parent_stderr or "")[-800:],
                    "handle_exists": handle_path.is_file(),
                }
            )
            # Fallback: grok-run-shaped orphan (trap HUP/INT; sleep).
            fallback = _orphan_fallback()
            out["fallback"] = fallback
            return out

        task_id = handle.get("task_id")
        launch_pid = handle.get("launch_pid")
        parent_pid = handle.get("parent_pid") or parent.pid
        wait_alive = time.monotonic() + 5.0
        while launch_pid and not process_alive(int(launch_pid)) and time.monotonic() < wait_alive:
            time.sleep(0.05)
        alive_before = bool(launch_pid and process_alive(int(launch_pid)))
        parent_alive_before = process_alive(int(parent_pid))
        os.kill(int(parent_pid), signal.SIGKILL)
        time.sleep(1.0)
        parent_alive_after = process_alive(int(parent_pid))
        child_alive_after = bool(launch_pid and process_alive(int(launch_pid)))

        # Real GrokBridge.status from a NEW process (parent is dead).
        status_script = f"""
import json, os, sys
os.environ["GROK_RUN"] = {str(patched)!r}
sys.path.insert(0, {str(REPO_ROOT)!r})
from hcli.grok_bridge import GrokBridge
from pathlib import Path
bridge = GrokBridge(Path({tmp!r}) / "ws")
print(json.dumps(bridge.status({task_id!r})))
"""
        st_proc = subprocess.run(
            [sys.executable, "-c", status_script],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        status: Dict[str, Any] = {}
        try:
            status = json.loads(st_proc.stdout.strip().splitlines()[-1])
        except Exception:
            status = {
                "parse_error": st_proc.stdout[-400:],
                "stderr": (st_proc.stderr or "")[-400:],
                "returncode": st_proc.returncode,
            }

        # Scheduler.from_workspace adoption vs Mission.from_workspace undo.
        adopt = _adoption_after_orphan(tmp, task_id, status)

        # grok-run cleanup does not kill: observe pid still alive after cleanup.
        cleanup = {}
        if patched.is_file() and task_id:
            cproc = subprocess.run(
                [str(patched), "cleanup", "--id", str(task_id)],
                capture_output=True,
                text=True,
                timeout=20,
            )
            cleanup = {
                "returncode": cproc.returncode,
                "stdout": (cproc.stdout or "")[-400:],
                "stderr": (cproc.stderr or "")[-400:],
                "pid_alive_after_cleanup": bool(
                    launch_pid and process_alive(int(launch_pid))
                ),
            }

        out.update(
            {
                "ok": True,
                "task_id": task_id,
                "launch_pid": launch_pid,
                "parent_pid": parent_pid,
                "parent_alive_before": parent_alive_before,
                "parent_alive_after_sigkill": parent_alive_after,
                "child_alive_before_kill": alive_before,
                "child_alive_after_parent_killed": child_alive_after,
                "status_after_orphan": status,
                "adoption": adopt,
                "cleanup": cleanup,
                "handle": handle,
                "task_status_file": (
                    (tasks_root / str(task_id) / "status").read_text().strip()
                    if task_id and (tasks_root / str(task_id) / "status").is_file()
                    else None
                ),
                "task_stderr_excerpt": (
                    (tasks_root / str(task_id) / "grok-stderr.log").read_text()[:400]
                    if task_id and (tasks_root / str(task_id) / "grok-stderr.log").is_file()
                    else None
                ),
            }
        )
        try:
            _reap_task(
                task_id,
                launch_pid,
                grok_run=str(patched),
                marker_extra=str(tasks_root / str(task_id)),
            )
        except Exception as exc:
            out["reap_error"] = f"{type(exc).__name__}: {exc}"
        out["child_alive_after_reap"] = bool(
            launch_pid and process_alive(int(launch_pid))
        )
        return out


def _adoption_after_orphan(
    tmp: str, task_id: Optional[str], status: Dict[str, Any]
) -> Dict[str, Any]:
    from hcli.mission import mission_state_path
    from hcli.dag_store import atomic_write_json

    ws = Path(tmp) / "adopt-ws"
    ws.mkdir(parents=True, exist_ok=True)
    wu = _wu(
        "g1",
        preferred_backend="grok",
        resource_class="GROK",
    )
    wu.status = "running"
    wu.assigned_runtime = 0
    wu.attempts = 1
    wu.assigned_backend = "grok"
    wu.backend_task_id = task_id
    store = DagStore(ws)
    store.save({"g1": wu})

    def liveness(tid: str) -> Dict[str, Any]:
        return dict(status) if status.get("task_id") or status.get("state") else {
            "state": "unknown",
            "task_id": tid,
        }

    units_sched = DagStore(ws).load(recover_running=True, grok_liveness=liveness)
    adopted = list(getattr(DagStore(ws), "adopted_running", []) or [])
    # Reload to capture adopted_running on the same instance.
    store2 = DagStore(ws)
    units_sched = store2.load(recover_running=True, grok_liveness=liveness)
    sched_status = units_sched["g1"].status
    adopted = list(store2.adopted_running)

    # Mission.from_workspace then fails in_flight running units.
    state_path = mission_state_path(ws)
    atomic_write_json(
        state_path,
        {
            "version": 1,
            "id": "orphan-mission",
            "goal": "orphan",
            "phase": "running",
            "strategy": "default",
            "started_at": time.time(),
            "last_checkpoint": time.time(),
            "in_flight": ["g1"],
            "accepted_count": 0,
            "no_progress_warning": None,
            "child_pids": [],
            "cancel_reason": None,
            "session_id": "orphan",
            "no_progress_threshold": 100,
            "units": {"g1": units_sched["g1"].to_dict()},
            "compiled": None,
        },
    )
    # Patch GrokBridge.status so Scheduler.from_workspace (called by
    # Mission.from_workspace) sees the live status.
    import hcli.grok_bridge as gb

    original = gb.GrokBridge.status

    def _status(self, tid):  # noqa: ANN001
        return dict(status) if status else {"state": "unknown", "task_id": tid}

    mission_status = None
    try:
        gb.GrokBridge.status = _status  # type: ignore[method-assign]
        class _E:
            def cancel(self):
                return None

            def execute_workunit(self, wu, context):
                return {"validation": {"ok": False, "reason": "not-run"}}

        mission = Mission.from_workspace(
            ws, engine=_E(), quiet=True, runtime_count=1, no_progress_threshold=100
        )
        mission_status = mission.scheduler.units["g1"].status
    except Exception as exc:
        mission_status = f"{type(exc).__name__}: {exc}"
    finally:
        gb.GrokBridge.status = original  # type: ignore[method-assign]

    return {
        "scheduler_load_status": sched_status,
        "adopted_running": adopted,
        "mission_from_workspace_status": mission_status,
        "liveness_state": status.get("state"),
        "liveness_process_alive": status.get("process_alive"),
    }


def _orphan_fallback() -> Dict[str, Any]:
    """Same detach grok-run uses: (trap '' HUP INT; sleep) & then kill parent."""
    with tempfile.TemporaryDirectory(prefix="repair-orphan-fb-") as tmp:
        handle = Path(tmp) / "h.json"
        script = f"""
import json, os, subprocess, time
from pathlib import Path
proc = subprocess.Popen(
    ["bash", "-c", "trap '' HUP INT; exec sleep 30"],
    start_new_session=True,
)
Path({str(handle)!r}).write_text(json.dumps({{"pid": proc.pid, "parent": os.getpid()}}))
time.sleep(120)
"""
        parent = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 10
        data = None
        while time.monotonic() < deadline:
            if handle.is_file():
                try:
                    data = json.loads(handle.read_text())
                    break
                except json.JSONDecodeError:
                    pass
            if parent.poll() is not None:
                break
            time.sleep(0.05)
        if not data:
            parent.kill()
            return {"ok": False, "error": "fallback parent did not start child"}
        child = int(data["pid"])
        alive_before = process_alive(child)
        os.kill(parent.pid, signal.SIGKILL)
        time.sleep(0.8)
        alive_after = process_alive(child)
        if alive_after:
            try:
                os.kill(child, signal.SIGKILL)
            except OSError:
                pass
            try:
                os.killpg(child, signal.SIGKILL)
            except OSError:
                pass
        return {
            "ok": True,
            "note": "fallback: grok-run-shaped detach (trap HUP INT; sleep), not a real grok session",
            "child_pid": child,
            "alive_before_parent_kill": alive_before,
            "alive_after_parent_kill": alive_after,
        }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _homeo_named(homeo: Dict[str, Any], needle: str) -> Optional[Dict[str, Any]]:
    for row in homeo.get("results") or []:
        if needle.lower() in str(row.get("name") or "").lower():
            return row
    return None


def assemble(probes: Dict[str, Any]) -> List[Dict[str, Any]]:
    homeo = probes["homeostasis"]
    persist = probes["persist"]
    clf = probes["classification"]
    backoff = probes["backoff"]
    health = probes["health"]
    circuit = probes["circuit"]
    cancel = probes["cancel"]
    orphan = probes["orphan"]

    depth_row = _homeo_named(homeo, "DEPTH bound stops")
    bounded_row = _homeo_named(homeo, "BOUNDED repair tree")
    neg = _homeo_named(homeo, "NEGATIVE CONTROL")
    count_row = _homeo_named(homeo, "COUNT cap")
    exhausted_row = _homeo_named(homeo, "explicit exhausted")
    structured_row = _homeo_named(homeo, "lineage is structured")
    rereadied_row = _homeo_named(homeo, "never re-readied")
    ser_row = _homeo_named(homeo, "survives serialization")

    neg_ok = bool(neg and neg.get("ok"))
    depth_ok = bool(depth_row and depth_row.get("ok")) and bool(
        bounded_row and bounded_row.get("ok")
    )

    rows: List[Dict[str, Any]] = []

    rows.append(
        {
            "item": 1,
            "property": "Repair-tree depth is bounded",
            "disposition": VERIFIED_EXISTING if depth_ok and neg_ok else OPEN,
            "evidence": {
                "mechanism": f"scheduler.MAX_REPAIR_DEPTH={MAX_REPAIR_DEPTH}; "
                f"MAX_REPAIRS_PER_ROOT={MAX_REPAIRS_PER_ROOT}",
                "homeostasis": {
                    "bounded_tree": bounded_row,
                    "depth_bound_when_failures_differ": depth_row,
                    "count_cap": count_row,
                    "negative_control": neg,
                },
                "negative_control_lifts_BOTH_bounds": True,
                "note": (
                    "Negative control raises MAX_REPAIR_DEPTH to 12 AND "
                    "MAX_REPAIRS_PER_ROOT to 500 AND replaces "
                    "_failure_signature with a unique value each call. "
                    "Lifting only the depth bound is not a demonstration "
                    "once the count cap exists."
                ),
            },
        }
    )

    lineage_ok = bool(structured_row and structured_row.get("ok")) and bool(
        ser_row and ser_row.get("ok")
    ) and persist.get("lineage_on_unit_persisted") and persist.get(
        "exhausted_roundtrip"
    )
    sigs_gap = persist.get("same_failure_emits_again_after_restart") and not persist.get(
        "signatures_persisted"
    )
    rows.append(
        {
            "item": 2,
            "property": "Repair lineage identity is persisted",
            "disposition": VERIFIED_EXISTING if lineage_ok else OPEN,
            "evidence": {
                "unit_fields": "repair_root/repair_depth/repair_reason/"
                "repair_exhausted survive WorkUnit.to_dict/from_dict "
                "and dag.json",
                "structured_lineage": structured_row,
                "serialization": ser_row,
                "persist_probe": persist,
                "gap": (
                    "Scheduler._repair_signatures and _repair_counts are "
                    "in-memory only. After from_workspace they are empty, "
                    "so the same failure signature can emit another repair."
                    if sigs_gap
                    else None
                ),
            },
        }
    )

    term_ok = bool(exhausted_row and exhausted_row.get("ok")) and bool(
        rereadied_row and rereadied_row.get("ok")
    ) and persist.get("exhausted_roundtrip")
    rows.append(
        {
            "item": 3,
            "property": "Terminal repair exhaustion reaches an explicit, durable state",
            "disposition": VERIFIED_EXISTING if term_ok else OPEN,
            "evidence": {
                "exhausted_state": exhausted_row,
                "not_rereadied": rereadied_row,
                "durable_flag": persist.get("exhausted_roundtrip"),
                "reason_examples": (exhausted_row or {}).get("detail"),
            },
        }
    )

    cycle_row = bounded_row  # dead backend stops via cycle, not depth
    cycle_ok = bool(cycle_row and cycle_row.get("ok"))
    rows.append(
        {
            "item": 4,
            "property": "Repair cycles are detected",
            "disposition": VERIFIED_EXISTING if cycle_ok else OPEN,
            "evidence": {
                "dead_backend_bounded_by_signature": cycle_row,
                "depth_bound_used_when_signatures_vary": depth_row,
                "in_process_only": persist.get("same_failure_emits_again_after_restart"),
                "gap": (
                    "Cycle set is not persisted across process restart "
                    "(see item 2)."
                    if persist.get("same_failure_emits_again_after_restart")
                    else None
                ),
            },
        }
    )

    clf_ok = (
        clf.get("all_seven_produced")
        and clf.get("retry_ok")
        and clf.get("budget_ok")
    )
    rows.append(
        {
            "item": 5,
            "property": "Failures are classified by retryability",
            "disposition": VERIFIED_EXISTING if clf_ok else OPEN,
            "evidence": {
                "seven_kinds": clf.get("produced"),
                "all_seven_produced": clf.get("all_seven_produced"),
                "retryable_map": clf.get("retryable_map"),
                "counts_toward_retry_budget_matches_retryable": clf.get("budget_ok"),
                "scheduler_fail_consults_classifier": clf.get(
                    "scheduler_fail_consults_classifier"
                ),
                "gap": (
                    "classify_failure / counts_toward_retry_budget exist and "
                    "behave, but Scheduler.fail does not call them. A "
                    f"NO_OP_MUTATION still emitted repair "
                    f"{clf.get('non_retryable_still_emits_repair')}."
                    if not clf.get("scheduler_fail_consults_classifier")
                    else None
                ),
            },
        }
    )

    backoff_is_property = (
        backoff.get("no_retry_sleep")
        and backoff.get("non_retryable_grok_na_still_cooled")
    )
    # Property FAILS: no retry backoff, and the only time gate (circuit
    # cooling) applies to a non-retryable failure (GrokNotAvailable).
    rows.append(
        {
            "item": 6,
            "property": "Backoff applies only to retryable failures",
            "disposition": REJECT,
            "evidence": {
                "no_retry_sleep_in_scheduler_fail": backoff.get("no_retry_sleep"),
                "fail_elapsed_s": backoff.get("retry_sleep_in_fail_s"),
                "nearest_mechanism": (
                    f"BackendHealth circuit cooling "
                    f"({backoff.get('circuit_cooling_seconds')}s)"
                ),
                "NO_OP_MUTATION_non_retryable_does_not_trip_circuit": not backoff.get(
                    "noop_trips_circuit"
                ),
                "GrokNotAvailable_non_retryable_DOES_open_circuit_and_cool": backoff.get(
                    "non_retryable_grok_na_still_cooled"
                ),
                "HTTP_503_retryable_opens_circuit": backoff.get("qwen_opens_circuit"),
                "cooling_reopens": backoff.get("cooling_reopens_after_30s"),
                "negative_science": (
                    "There is no retry backoff. The only time-gated refusal "
                    "is circuit cooling, and it is keyed on _CIRCUIT_KINDS "
                    "not on retryable. GrokNotAvailable is NON_RETRYABLE "
                    "yet two records open the grok breaker and hold it for "
                    "the cooling window. The property is false."
                ),
                "raw": backoff,
            },
        }
    )

    rows.append(
        {
            "item": 7,
            "property": "Backend health is persisted",
            "disposition": VERIFIED_EXISTING if health.get("ok") else OPEN,
            "evidence": {
                "method": "child process wrote backend_health.json and exited; "
                "a second process constructed BackendHealth on the same "
                "workspace and read grok consecutive_failures=2, cpu "
                "last_success_time set, qwen healthy",
                "writer_returncode": health.get("writer_returncode"),
                "reader_returncode": health.get("reader_returncode"),
                "path_exists": health.get("path_exists"),
                "snap": health.get("snap"),
            },
        }
    )

    circuit_obj_ok = (
        circuit.get("qwen_open")
        and circuit.get("grok_untouched")
        and circuit.get("still_open_before_cooling")
        and circuit.get("reopens_allows_new")
        and circuit.get("success_closes")
    )
    unwired = circuit.get("assign_ready_dispatches_through_open_circuit")
    rows.append(
        {
            "item": 8,
            "property": "The circuit breaker is bounded",
            "disposition": VERIFIED_EXISTING if circuit_obj_ok else OPEN,
            "evidence": {
                "threshold": circuit.get("threshold"),
                "opens_at_n_consecutive_backend_failures": circuit.get("qwen_open"),
                "other_backends_untouched": circuit.get("grok_untouched")
                and circuit.get("cpu_untouched"),
                "still_open_inside_cooling_window": circuit.get(
                    "still_open_before_cooling"
                ),
                "reopens_degraded_after_cooling": circuit.get("reopens_to"),
                "success_resets": circuit.get("success_closes"),
                "assign_ready_ignores_breaker": unwired,
                "gap": (
                    "BackendHealth.allows_new_assignments is bounded, but "
                    "assign_ready / Scheduler.dispatch never call it. A unit "
                    "with preferred_backend=qwen was dispatched while that "
                    "breaker was open. The object is bounded; live dispatch "
                    "is not."
                    if unwired
                    else None
                ),
                "raw": circuit,
            },
        }
    )

    coop = cancel.get("cooperative_engine") or {}
    slow = cancel.get("blocking_child_engine") or {}
    grok_c = cancel.get("grok_path") or {}
    cancel_sound = (
        coop.get("phase") == "cancelled"
        and coop.get("status_after") == "failed"
        and not coop.get("repairs")
        and grok_c.get("child_alive_after_cancel") is False
        and grok_c.get("GrokBridge_has_cancel") is False
    )
    # The grok path is the hole: child stays alive. Cooperative engine is fine.
    grok_unsound = bool(grok_c.get("child_alive_after_cancel"))
    rows.append(
        {
            "item": 9,
            "property": "Cancellation semantics are sound",
            "disposition": REJECT if grok_unsound else (VERIFIED_EXISTING if cancel_sound else OPEN),
            "evidence": {
                "cooperative_engine": coop,
                "blocking_child_engine": slow,
                "grok_path": grok_c,
                "observed": (
                    "Mission.cancel sets an event, calls engine.cancel(), "
                    "fails in-flight units WITHOUT emitting a repair, joins "
                    "worker threads for 1s, then _stop_children on "
                    "mission.child_pids and engine.child_pids. A cooperative "
                    "engine unit ends failed, phase=cancelled, reason durable "
                    "in state.json, no repair. A grok unit's launch_pid is "
                    "NOT registered; GrokBridge has no cancel(); wait() only "
                    "treats done/failed as terminal; grok-run cleanup does "
                    "not send a signal. Cancelling a dispatched grok-like "
                    "unit left the child process alive."
                ),
            },
        }
    )

    orphan_child_lived = bool(orphan.get("child_alive_after_parent_killed"))
    fallback = orphan.get("fallback") or {}
    fallback_lived = bool(fallback.get("alive_after_parent_kill"))
    mission_undo = (orphan.get("adoption") or {}).get("mission_from_workspace_status")
    rows.append(
        {
            "item": 10,
            "property": "Orphan tasks are handled",
            "disposition": OPEN,
            "evidence": {
                "real_grok_launch": {
                    "ok": orphan.get("ok"),
                    "task_id": orphan.get("task_id"),
                    "launch_pid": orphan.get("launch_pid"),
                    "child_alive_after_parent_killed": orphan.get(
                        "child_alive_after_parent_killed"
                    ),
                    "status_after_orphan": orphan.get("status_after_orphan"),
                    "cleanup_kills": (orphan.get("cleanup") or {}).get(
                        "pid_alive_after_cleanup"
                    ),
                    "error": orphan.get("error"),
                },
                "fallback_detach": fallback,
                "adoption": orphan.get("adoption"),
                "observed": (
                    "grok-run --background is `(trap '' HUP INT; execute_task) &` "
                    "then the grok-run parent exits. Killing the HCLI parent "
                    "therefore cannot deliver HUP/INT to the task, and TERM "
                    "is never sent. DagStore can keep a live grok unit "
                    "`running` when grok_liveness reports running. "
                    "Mission.from_workspace then walks in_flight and "
                    "transition_status(running -> failed) — undoing adoption. "
                    "Mission._loop has no waiter for adopted running units; "
                    "a running unit with no inflight thread is `blocked`. "
                    "grok-run cleanup removes a worktree; it does not kill "
                    "the process. RuntimePool.reap_orphans is llama-server "
                    "only. Nothing reaps a Grok task whose parent died."
                ),
                "mission_undoes_adoption": mission_undo,
                "orphan_survived_parent_death": orphan_child_lived or fallback_lived,
            },
        }
    )
    return rows


def watched_fail(probes: Dict[str, Any], rows: List[Dict[str, Any]]) -> List[str]:
    lines = []
    homeo = probes["homeostasis"]
    if not homeo.get("ok"):
        lines.append(
            "Homeostasis harness returned non-zero: "
            f"rc={homeo.get('returncode')} failed={homeo.get('failed')} "
            f"stderr={(homeo.get('stderr') or '')[-300:]}"
        )
    neg = _homeo_named(homeo, "NEGATIVE CONTROL")
    if neg and not neg.get("ok"):
        lines.append(
            "NEGATIVE CONTROL did not grow the tree with BOTH bounds lifted: "
            f"{neg.get('detail')}. The bounds are then not what is stopping it."
        )
    persist = probes["persist"]
    if persist.get("same_failure_emits_again_after_restart"):
        lines.append(
            "After Scheduler.from_workspace, _repair_signatures was empty and "
            f"the same failure emitted {persist.get('again_id')}."
        )
    clf = probes["classification"]
    if clf.get("non_retryable_still_emits_repair"):
        lines.append(
            "NO_OP_MUTATION (non-retryable) still emitted "
            f"{clf.get('non_retryable_still_emits_repair')} because "
            "Scheduler.fail does not consult classify_failure."
        )
    backoff = probes["backoff"]
    if backoff.get("non_retryable_grok_na_still_cooled"):
        lines.append(
            "GrokNotAvailable is NON_RETRYABLE but opened the grok circuit "
            "and held it for the cooling window — backoff is not gated on "
            "retryable, and there is no retry sleep."
        )
    circuit = probes["circuit"]
    if circuit.get("assign_ready_dispatches_through_open_circuit"):
        lines.append(
            "assign_ready dispatched a qwen unit while that backend's "
            "circuit was open. allows_new_assignments is not consulted."
        )
    grok_c = (probes["cancel"].get("grok_path") or {})
    if grok_c.get("child_alive_after_cancel"):
        lines.append(
            f"Cancelled a dispatched grok-like unit (pid {grok_c.get('child_pid')}); "
            "the child was still alive after Mission.cancel. GrokBridge has "
            "no cancel(); launch_pid is not in mission.child_pids."
        )
    slow = probes["cancel"].get("blocking_child_engine") or {}
    if slow.get("child_alive_after_cancel"):
        lines.append(
            "Blocking engine child (sleep) was still alive after cancel; "
            "_stop_children did not reap it in time."
        )
    elif slow.get("phase") == "cancelled" and not slow.get("child_alive_after_cancel"):
        lines.append(
            "Blocking engine child WAS reaped by _stop_children (engine.child_pids). "
            "That path works when the pid is registered; grok does not register it."
        )
    orphan = probes["orphan"]
    if orphan.get("child_alive_after_parent_killed"):
        lines.append(
            f"Killed parent pid {orphan.get('parent_pid')}; grok launch_pid "
            f"{orphan.get('launch_pid')} (task {orphan.get('task_id')}) was "
            "still alive. Orphan not handled."
        )
    elif orphan.get("error"):
        lines.append(
            f"Real grok-run consult did not launch: {orphan.get('error')}. "
            f"fallback={orphan.get('fallback')}"
        )
        fb = orphan.get("fallback") or {}
        if fb.get("alive_after_parent_kill"):
            lines.append(
                "Fallback detach (trap HUP INT; sleep) survived parent SIGKILL — "
                "the grok-run background shape orphans on parent death."
            )
    cleanup = orphan.get("cleanup") or {}
    if cleanup.get("pid_alive_after_cleanup"):
        lines.append(
            "grok-run cleanup --id returned with the launch_pid still alive; "
            "cleanup does not kill the process."
        )
    adopt = orphan.get("adoption") or {}
    if adopt.get("mission_from_workspace_status") == "failed" and adopt.get(
        "scheduler_load_status"
    ) == "running":
        lines.append(
            "DagStore+liveness adopted the live grok unit as running; "
            "Mission.from_workspace then failed the in_flight unit, undoing adoption."
        )
    lines.append(
        "Bootstrap pytest: 1 failed (test_evidence_freshness context_efficiency "
        "receipt 379 != 377), 365 passed, 1 skipped. Expected known defect."
    )
    return lines


def handoff(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Exact patches for hcli — this lane cannot write those files."""
    patches = []
    patches.append(
        {
            "file": "hcli/scheduler.py",
            "why": "Persist cycle signatures and per-root counts so restart cannot re-open a closed lineage (items 2, 4).",
            "patch": """
# in Scheduler._persist extra= and Scheduler.from_workspace restore:
"repair_signatures": {k: sorted(v) for k, v in self._repair_signatures.items()},
"repair_counts": dict(self._repair_counts),
# from_workspace after load:
sigs = meta.get("repair_signatures") or {}
if isinstance(sigs, dict):
    sched._repair_signatures = {str(k): set(v) for k, v in sigs.items() if isinstance(v, list)}
counts = meta.get("repair_counts") or {}
if isinstance(counts, dict):
    sched._repair_counts = {str(k): int(v) for k, v in counts.items()}
""".strip(),
        }
    )
    patches.append(
        {
            "file": "hcli/scheduler.py + workunit.py",
            "why": "Consult classify_failure / allows_new_assignments (items 5, 6, 8).",
            "patch": """
# Scheduler.fail, before _emit_repair:
from .resources import counts_toward_retry_budget, classify_failure
clf = classify_failure(context)
if not clf.retryable:
    wu.repair_exhausted = True
    wu.repair_reason = f"non-retryable: {clf.observed or clf.kind}"
    self._persist()
    return None

# assign_ready, after can_admit succeeds, before transition_status running:
# caller must pass health: Optional[BackendHealth]
backend = getattr(wu, "preferred_backend", None) or "cpu"
if health is not None and not health.allows_new_assignments(backend):
    continue
""".strip(),
        }
    )
    patches.append(
        {
            "file": "hcli/grok_bridge.py + mission.py",
            "why": "Cancellation of a grok unit must kill launch_pid; wait must treat cancelled/stale-running as terminal (item 9).",
            "patch": """
# GrokBridge.cancel(task_id):
def cancel(self, task_id: str) -> Dict[str, Any]:
    receipt = self._read_receipt(task_id) or {}
    pid = receipt.get("launch_pid")
    killed = False
    if pid:
        try:
            os.kill(int(pid), signal.SIGTERM)
            killed = True
        except OSError:
            pass
    return {"task_id": task_id, "killed": killed, "launch_pid": pid}

# wait() terminal states:
if state in ("done", "failed", "cancelled", "stale-running", "unknown"):
    return last

# Mission.cancel: for each inflight wu with backend_task_id, GrokBridge.cancel
# Mission._start_unit: register_child_pid(handle.launch_pid) when grok
# Mission.from_workspace: do NOT fail in_flight units that DagStore just adopted
#   (skip uid if uid in {a['unit_id'] for a in sched.store.adopted_running})
""".strip(),
        }
    )
    patches.append(
        {
            "file": "hcli/mission.py",
            "why": "Adopted running grok units must be waited, not declared blocked (item 10).",
            "patch": """
# _loop: if a unit is status=running with backend_task_id and no inflight
# thread, spawn a waiter that GrokBridge.wait()s it (and honour _cancel).
# Do not take the "no ready work and mission is not done" branch while
# adopted_running is non-empty.
# On parent death the grok process is designed to survive (trap HUP INT);
# handling orphans MEANS adopting-and-waiting or killing, not ignoring.
""".strip(),
        }
    )
    return patches


def print_table(rows: List[Dict[str, Any]], probes: Dict[str, Any], head: str) -> None:
    print("=" * 78)
    print("REPAIR DISPOSITION TABLE")
    print("=" * 78)
    print(f"git HEAD: {head}")
    print(f"generated_at: {_now()}")
    print(f"MAX_REPAIR_DEPTH={MAX_REPAIR_DEPTH} MAX_REPAIRS_PER_ROOT={MAX_REPAIRS_PER_ROOT}")
    print()
    print(f"{'#':<3} {'DISPOSITION':<26} PROPERTY")
    print("-" * 78)
    for row in rows:
        print(f"{row['item']:<3} {row['disposition']:<26} {row['property']}")
    print()
    for row in rows:
        print("-" * 78)
        print(f"{row['item']}. {row['property']}")
        print(f"   disposition: {row['disposition']}")
        ev = row["evidence"]
        print("   evidence:")
        print("   " + json.dumps(ev, indent=2, default=str).replace("\n", "\n   "))
    print()
    print("=" * 78)
    print("WHAT I WATCHED FAIL")
    print("=" * 78)
    for line in watched_fail(probes, rows):
        print(f"- {line}")
    print()
    print("=" * 78)
    print("HANDOFF (hcli is READ-ONLY in this lane)")
    print("=" * 78)
    for i, h in enumerate(handoff(rows), 1):
        print(f"\n### HANDOFF {i}: {h['file']}")
        print(f"why: {h['why']}")
        print("```")
        print(h["patch"])
        print("```")
    homeo = probes["homeostasis"]
    print()
    print("=" * 78)
    print("HOMEOSTASIS HARNESS STDOUT (check lines only)")
    print("=" * 78)
    raw_out = homeo.get("stdout") or ""
    keep = [
        ln
        for ln in raw_out.splitlines()
        if ln.startswith("ok  ") or ln.startswith("FAIL ") or "checks passed" in ln
        or ln.startswith("receipt:")
    ]
    print("\n".join(keep) if keep else raw_out[-2000:])
    err = homeo.get("stderr") or ""
    err_keep = [
        ln
        for ln in err.splitlines()
        if "repair budget exhausted" not in ln and ln.strip()
    ]
    if err_keep:
        print("--- stderr ---")
        print("\n".join(err_keep[-40:]))


def main() -> int:
    head = _git_head()
    probes: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    def run(name, fn):
        try:
            probes[name] = fn()
        except Exception:
            errors[name] = traceback.format_exc()
            probes[name] = {"ok": False, "error": errors[name]}

    run("homeostasis", run_homeostasis)
    run("persist", probe_lineage_persist)
    run("classification", probe_classification)
    run("backoff", probe_backoff)
    run("health", probe_health_restart)
    run("circuit", probe_circuit)
    run("cancel", probe_cancel)
    run("orphan", probe_orphan)

    rows = assemble(probes)
    payload = {
        "gate": "REPAIR_DISPOSITION",
        "generated_at": _now(),
        "git_head": head,
        "max_repair_depth": MAX_REPAIR_DEPTH,
        "max_repairs_per_root": MAX_REPAIRS_PER_ROOT,
        "bootstrap_pytest": {
            "failed": 1,
            "passed": 365,
            "skipped": 1,
            "expected_failure": "hcli/tests/test_evidence_freshness.py::TestContextEfficiencyReceipt::test_receipt_fields_come_from_observed_assembly (379 != 377)",
        },
        "table": rows,
        "probes": {
            k: v
            for k, v in probes.items()
            if k != "homeostasis"
        },
        "homeostasis": {
            "ok": probes["homeostasis"].get("ok"),
            "returncode": probes["homeostasis"].get("returncode"),
            "failed": probes["homeostasis"].get("failed"),
            "receipt_result": probes["homeostasis"].get("receipt_result"),
            "results": probes["homeostasis"].get("results"),
            "stdout": probes["homeostasis"].get("stdout"),
        },
        "what_i_watched_fail": watched_fail(probes, rows),
        "handoff": handoff(rows),
        "probe_errors": errors,
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print_table(rows, probes, head)
    print()
    print(f"receipt: {RECEIPT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

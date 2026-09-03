#!/usr/bin/env python3
"""Bounded qualification of the existing HCLI resident control plane.

This is a qualification harness, not a new resident implementation.  The
first gate runs a useful source/test mission through AgentOS with a
deterministic local engine.  The second gate runs the real ResidentSupervisor
around a disposable AgentOS worker, kills only that worker, and proves that
the supervisor recovers the durable mission.  No model weights or GPU are
opened by this script.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
RECEIPT = REPO / "receipts" / "headless" / "HCLI_RESIDENT_CRASH_RECOVERY.json"
SCHEMA = "hcli.agentos.resident_qualification.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return default


def _write_json(path: Path, value: Any) -> None:
    from hcli.persist import atomic_write_json

    atomic_write_json(path, value)


def _wait_for(predicate, timeout: float = 30.0, interval: float = 0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _owned_alive(pid: Any) -> bool:
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


def _verifier(code: str) -> str:
    import shlex

    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def _check(checks: Dict[str, bool], name: str, value: Any) -> None:
    checks[name] = bool(value)


class SourceTestEngine:
    """A real, bounded HCLI worker that inspects source and runs one test."""

    def __init__(self, repo: Path, workspace: Path) -> None:
        self.repo = repo
        self.workspace = workspace
        self.qualification_root = workspace / ".hcli" / "qualification"
        self.source = repo / "hcli" / "agentos" / "resident.py"
        self.source_receipt = self.qualification_root / "source_test.json"
        self.child_receipt = self.qualification_root / "child_continuation.json"
        self.calls: List[str] = []

    def execute_workunit(self, unit: Any, _context: Mapping[str, Any]) -> Dict[str, Any]:
        self.calls.append(str(unit.id))
        self.qualification_root.mkdir(parents=True, exist_ok=True)
        if unit.id == "source-inspect":
            command = [
                sys.executable,
                "-m",
                "pytest",
                "hcli/tests/test_hcli_resident_daemon.py",
                "-q",
            ]
            proc = subprocess.run(
                command,
                cwd=str(self.repo),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            source_text = self.source.read_text(encoding="utf-8")
            tokens = {
                token: token in source_text
                for token in (
                    "class ResidentSupervisor",
                    "def resident_behavior",
                    "clean_room_requested",
                    "admit_evidence_children",
                )
            }
            _write_json(
                self.source_receipt,
                {
                    "schema": "hcli.resident.source_test.v1",
                    "command": command,
                    "returncode": proc.returncode,
                    "stdout_tail": (proc.stdout or "")[-4000:],
                    "stderr_tail": (proc.stderr or "")[-2000:],
                    "source_path": str(self.source),
                    "source_sha256": _sha256(self.source),
                    "required_symbols": tokens,
                    "tests_passed": proc.returncode == 0,
                },
            )
            child = {
                "id": "receipt-continuation",
                "role": "verification",
                "description": "Re-read the source/test receipt and compile the resident implementation.",
                "resource_class": "TEST",
                "verifier": _verifier(
                    "import json; from pathlib import Path; "
                    f"p=Path({str(self.child_receipt)!r}); "
                    "d=json.loads(p.read_text()); assert d['compile_returncode']==0; "
                    "assert d['parent_receipt_sha256']"
                ),
                "dependencies": ["source-inspect"],
            }
            return {
                "content": "source inspection and resident fixture test executed",
                "child_workunits": [child],
            }

        if unit.id == "receipt-continuation":
            parent = _read_json(self.source_receipt, {})
            proc = subprocess.run(
                [sys.executable, "-m", "py_compile", str(self.source)],
                cwd=str(self.repo),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            _write_json(
                self.child_receipt,
                {
                    "schema": "hcli.resident.child_continuation.v1",
                    "parent_receipt_sha256": _sha256(self.source_receipt),
                    "parent_tests_passed": parent.get("tests_passed") is True,
                    "compile_returncode": proc.returncode,
                    "compile_stderr": (proc.stderr or "")[-2000:],
                    "source_sha256": _sha256(self.source),
                },
            )
            return {"content": "parent receipt re-read and resident compiled"}

        return {"content": f"unexpected qualification unit {unit.id}"}


def run_real_mission_gate() -> Dict[str, Any]:
    """Run a useful source/test mission and continue into evidence-derived work."""
    from hcli.agentos import AgentOS, ResidentConfig, ResidentDaemon
    from hcli.workunit import WorkUnit

    workspace = Path(tempfile.mkdtemp(prefix="hcli-resident-real-mission-"))
    daemon = ResidentDaemon(workspace)
    daemon.configure(
        ResidentConfig(
            workspace=str(workspace),
            goal="qualify the resident with a bounded source and test mission",
            runtime_count=1,
        )
    )
    engine = SourceTestEngine(REPO, workspace)
    agent = AgentOS(workspace, engine=engine, repo_root=REPO)
    search = agent.invoke_tool(
        "filesystem.search",
        {
            "root": str(REPO / "hcli" / "agentos"),
            "pattern": "class ResidentSupervisor",
            "glob": "resident.py",
            "max_results": 4,
        },
    )
    source_hash = _sha256(engine.source)
    parent = WorkUnit(
        id="source-inspect",
        role="test",
        description="Inspect the resident implementation and run its focused qualification test.",
        resource_class="TEST",
        verifier=_verifier(
            "import json; from pathlib import Path; "
            f"p=Path({str(engine.source_receipt)!r}); d=json.loads(p.read_text()); "
            "assert d['tests_passed'] is True; "
            f"assert d['source_sha256']=={source_hash!r}; "
            "assert all(d['required_symbols'].values())"
        ),
    )
    agent.start_mission(
        "Qualify the resident source/test path with deterministic evidence.",
        units={parent.id: parent},
    )
    first = agent.run()
    first_mission_id = agent.mission.id
    refill = daemon.refill_from_evidence(agent.mission, first.get("evidence"))
    second = agent.continue_mission()
    units = agent.mission.scheduler.units
    report = {
        "schema": SCHEMA,
        "gate": "DURABLE_REAL_MISSION",
        "claim_boundary": "HCLI control-plane and deterministic source/test qualification; no model-body or GPU performance claim",
        "workspace": str(workspace),
        "tool_search": search.to_dict(),
        "first_result": first,
        "second_result": second,
        "mission_id_before_refill": first_mission_id,
        "mission_id_after_refill": agent.mission.id,
        "engine_calls": list(engine.calls),
        "refill": refill,
        "receipts": {
            "source_test": str(engine.source_receipt),
            "child_continuation": str(engine.child_receipt),
            "tool_receipt_dir": str(workspace / ".hcli" / "receipts" / "tools"),
            "mission_state": str(workspace / ".hcli" / "mission" / "state.json"),
        },
    }
    checks: Dict[str, bool] = {}
    _check(checks, "typed_source_search_passed", search.ok)
    _check(checks, "source_search_found_supervisor", bool((search.value or {}).get("matches")))
    _check(checks, "first_mission_completed", first.get("status") == "completed")
    _check(checks, "evidence_child_admitted", any(row.get("status") == "ADMITTED" for row in refill))
    _check(checks, "same_mission_id", first_mission_id == agent.mission.id)
    _check(checks, "child_completed", units.get("receipt-continuation").status == "completed")
    _check(checks, "no_duplicate_completed_work", engine.calls == ["source-inspect", "receipt-continuation"])
    _check(checks, "mission_receipt_exists", (workspace / ".hcli" / "mission" / "state.json").is_file())
    _check(checks, "source_receipt_exists", engine.source_receipt.is_file())
    _check(checks, "child_receipt_exists", engine.child_receipt.is_file())
    report["checks"] = checks
    report["status"] = "PASSED" if all(checks.values()) else "FAILED"
    return report


def _safe_memory(_self: Any, _config: Any) -> Dict[str, Any]:
    return {
        "safe": True,
        "reasons": [],
        "pressure": "qualification_override",
        "total_bytes": 1,
        "free_bytes": 1,
        "swap_used_bytes": 0,
        "reserve_bytes": 0,
        "swap_ceiling_bytes": None,
    }


def _qualification_spawn_worker(self: Any) -> None:
    from hcli.agentos.resident import _now
    from hcli.resources import process_start_token

    state = self.store.read()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPT), "--qualification-worker", str(self.state_path)],
        cwd=str(REPO),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    self._worker_process = proc
    self.store.update(
        state="RUNNING",
        worker_pid=proc.pid,
        worker_start_token=process_start_token(proc.pid),
        worker_started_at=_now(),
        worker_returncode=None,
        error=None,
        generation=int(state.get("generation") or 0) + 1,
        last_event="worker_started",
    )


def qualification_supervisor(state_path: str) -> int:
    from hcli.agentos.resident import ResidentSupervisor

    supervisor = ResidentSupervisor(state_path)
    supervisor._memory = types.MethodType(_safe_memory, supervisor)
    supervisor._spawn_worker = types.MethodType(_qualification_spawn_worker, supervisor)
    return supervisor.run()


def _append_call(path: Path, unit_id: str) -> None:
    old = _read_json(path, [])
    calls = old if isinstance(old, list) else []
    calls.append({"id": unit_id, "pid": os.getpid(), "at": time.time()})
    _write_json(path, calls)


def qualification_worker(state_path: str) -> int:
    from hcli.agentos import AgentOS, ResidentDaemon, ResidentConfig
    from hcli.workunit import WorkUnit

    state_file = Path(state_path).expanduser().resolve()
    workspace = state_file.parents[2]
    daemon = ResidentDaemon(workspace)
    state = daemon.store.read()
    config = ResidentConfig.from_mapping(state["config"])
    calls_path = workspace / ".hcli" / "qualification" / "worker_calls.json"
    calls_path.parent.mkdir(parents=True, exist_ok=True)
    generation = int(state.get("generation") or 0)

    class RecoveryEngine:
        def execute_workunit(self, unit: Any, _context: Mapping[str, Any]) -> Dict[str, Any]:
            _append_call(calls_path, unit.id)
            if unit.id == "slow" and generation == 1:
                (calls_path.parent / "slow-started").write_text(str(os.getpid()), encoding="utf-8")
                time.sleep(120)
            if unit.id == "slow":
                (calls_path.parent / "slow-resumed").write_text(str(os.getpid()), encoding="utf-8")
            return {"content": f"qualification worker completed {unit.id}"}

    first_marker = workspace / ".hcli" / "qualification" / "first-completed"
    units = {
        "first": WorkUnit(
            id="first",
            role="test",
            description="Create the durable pre-crash qualification marker.",
            resource_class="TEST",
            verifier=_verifier(
                "from pathlib import Path; "
                f"assert Path({str(first_marker)!r}).is_file()"
            ),
        ),
        "slow": WorkUnit(
            id="slow",
            role="test",
            description="Hold an active bounded unit long enough to qualify worker crash recovery.",
            dependencies=["first"],
            resource_class="TEST",
            verifier=_verifier(
                "from pathlib import Path; "
                f"assert Path({str(calls_path.parent / 'slow-resumed')!r}).is_file()"
            ),
        ),
    }

    engine = RecoveryEngine()
    agent = AgentOS(workspace, engine=engine, repo_root=REPO)
    mission_path = workspace / ".hcli" / "mission" / "state.json"
    if mission_path.is_file():
        agent.recover_mission()
    else:
        agent.start_mission(config.goal, units=units, runtime_count=1)
    original = engine.execute_workunit

    def execute(unit: Any, context: Mapping[str, Any]) -> Dict[str, Any]:
        result = original(unit, context)
        if unit.id == "first":
            first_marker.parent.mkdir(parents=True, exist_ok=True)
            first_marker.write_text("completed", encoding="utf-8")
        return result

    engine.execute_workunit = execute  # type: ignore[method-assign]
    result = agent.run()
    daemon.store.update(
        mission_id=agent.mission.id,
        worker_result=result,
        worker_returncode=0,
        last_event="worker_completed",
    )
    return 0


def run_crash_recovery_gate() -> Dict[str, Any]:
    from hcli.agentos import ResidentConfig, ResidentDaemon

    workspace = Path(tempfile.mkdtemp(prefix="hcli-resident-crash-recovery-"))
    daemon = ResidentDaemon(workspace)
    daemon.configure(
        ResidentConfig(
            workspace=str(workspace),
            goal="qualify resident worker crash recovery",
            runtime_count=1,
            interval_s=0.1,
            evacuation_grace_s=0.5,
            max_restarts=3,
        )
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    supervisor_process = subprocess.Popen(
        [sys.executable, str(SCRIPT), "--qualification-supervisor", str(daemon.store.state_path)],
        cwd=str(REPO),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    state_path = daemon.store.state_path
    checks: Dict[str, bool] = {}
    detail: Dict[str, Any] = {"workspace": str(workspace)}
    try:
        _check(checks, "supervisor_started", _wait_for(lambda: _owned_alive(_read_json(state_path, {}).get("supervisor_pid")), 10))
        initial = _read_json(state_path, {})
        supervisor_pid = initial.get("supervisor_pid")
        _check(checks, "mission_started", _wait_for(lambda: (workspace / ".hcli" / "mission" / "state.json").is_file(), 10))
        mission_path = workspace / ".hcli" / "mission" / "state.json"
        _check(checks, "slow_unit_active", _wait_for(lambda: (workspace / ".hcli" / "qualification" / "slow-started").is_file(), 20))
        _check(checks, "worker_registered", _wait_for(lambda: _read_json(state_path, {}).get("worker_pid") is not None, 5))
        before = _read_json(mission_path, {})
        mission_id = before.get("id")
        before_first = (before.get("units") or {}).get("first", {})
        worker_pid = _read_json(state_path, {}).get("worker_pid")
        worker_generation = _read_json(state_path, {}).get("generation")
        kill_delivered = False
        if _owned_alive(worker_pid):
            try:
                os.kill(int(worker_pid), signal.SIGKILL)
                kill_delivered = True
            except OSError:
                kill_delivered = False
        _check(checks, "worker_sigkill_delivered", kill_delivered)
        _check(checks, "supervisor_survived_worker_kill", _owned_alive(supervisor_pid))
        _check(checks, "restart_recorded", _wait_for(lambda: int(_read_json(state_path, {}).get("restart_count") or 0) >= 1, 15))
        _check(checks, "worker_relaunched", _wait_for(lambda: int(_read_json(state_path, {}).get("generation") or 0) > int(worker_generation or 0), 15))
        _check(checks, "mission_continued", _wait_for(lambda: (workspace / ".hcli" / "qualification" / "slow-resumed").is_file(), 25))
        _check(checks, "mission_finished_after_recovery", _wait_for(lambda: (_read_json(mission_path, {}).get("units") or {}).get("slow", {}).get("status") == "completed", 10))
        _check(checks, "supervisor_reconciled_recovered_worker", _wait_for(lambda: (
            _read_json(state_path, {}).get("worker_pid") is None
            and int(_read_json(state_path, {}).get("failure_streak") or 0) == 0
        ), 10))
        final = _read_json(mission_path, {})
        final_units = final.get("units") or {}
        calls = _read_json(workspace / ".hcli" / "qualification" / "worker_calls.json", [])
        call_ids = [item.get("id") for item in calls if isinstance(item, dict)]
        _check(checks, "same_mission_id", mission_id == final.get("id"))
        _check(checks, "completed_work_not_rerun", call_ids.count("first") == 1)
        _check(checks, "unfinished_work_recovered", call_ids.count("slow") >= 2)
        _check(checks, "first_unit_completed_before_and_after", before_first.get("status") == "completed" and final_units.get("first", {}).get("status") == "completed")
        _check(checks, "next_unit_completed", final_units.get("slow", {}).get("status") == "completed")
        current = _read_json(state_path, {})
        _check(checks, "restart_provenance_durable", int(current.get("restart_count") or 0) >= 1 and int(current.get("cycles") or 0) >= 1)
        _check(checks, "failure_streak_cleared_after_recovery", int(current.get("failure_streak") or 0) == 0)
        detail.update({
            "supervisor_pid": supervisor_pid,
            "killed_worker_pid": worker_pid,
            "initial_generation": worker_generation,
            "mission_id": mission_id,
            "calls": calls,
            "state": current,
            "mission": final,
        })
    finally:
        daemon.request_stop()
        _wait_for(lambda: not _owned_alive(_read_json(state_path, {}).get("supervisor_pid")), 10)
        if supervisor_process.poll() is None:
            supervisor_process.terminate()
            try:
                supervisor_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                supervisor_process.kill()
        detail["supervisor_returncode"] = supervisor_process.returncode
    return {
        "schema": SCHEMA,
        "gate": "RESIDENT_CRASH_RECOVERY",
        "claim_boundary": "supervisor/mission durability qualification with deterministic AgentOS worker; no model-body or GPU performance claim",
        "checks": checks,
        "status": "PASSED" if all(checks.values()) else "FAILED",
        **detail,
    }


def run() -> int:
    real = run_real_mission_gate()
    crash = run_crash_recovery_gate()
    report = {
        "schema": SCHEMA,
        "status": "PASSED" if real["status"] == "PASSED" and crash["status"] == "PASSED" else "FAILED",
        "claim_boundary": "bounded HCLI resident control-plane qualification only; no Qwen load, GPU, or performance claim",
        "real_mission": real,
        "crash_recovery": crash,
        "generated_at": time.time(),
    }
    _write_json(RECEIPT, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"receipt: {RECEIPT}")
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    if "--qualification-supervisor" in sys.argv:
        raise SystemExit(qualification_supervisor(sys.argv[sys.argv.index("--qualification-supervisor") + 1]))
    if "--qualification-worker" in sys.argv:
        raise SystemExit(qualification_worker(sys.argv[sys.argv.index("--qualification-worker") + 1]))
    raise SystemExit(run())

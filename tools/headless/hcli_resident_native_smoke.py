#!/usr/bin/env python3
"""Guarded production-path smoke for the HCLI resident and sealed Qwen 3.8.

The normal path is intentionally cheap: validate the sealed profile, run the
same host/runtime admission decisions used by HCLI, and prove that a packed
host leaves the native WorkUnit queued without starting a model worker.  If
both gates are safe, the script performs exactly one bounded production
resident mission in a disposable workspace.  It never overrides a memory or
swap refusal merely to make the smoke run.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

from hcli.persist import atomic_write_json


REPO = Path(__file__).resolve().parents[2]
PROFILE = (REPO / "hcli" / "hawking-native.sealed-3.14.json").resolve()
RECEIPT = REPO / "receipts" / "headless" / "HCLI_RESIDENT_NATIVE_DAEMON_SMOKE.json"
SCHEMA = "hcli.agentos.resident_native_smoke.v1"
NATIVE_TIMEOUT_S = 90.0


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wait_for(predicate: Any, timeout: float, interval: float = 0.05) -> bool:
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _check(checks: Dict[str, bool], name: str, value: Any) -> None:
    checks[name] = bool(value)


@contextmanager
def _bounded_generation_env() -> Iterator[None]:
    names = {
        "HCLI_HAWKING_NATIVE_MODE": "resident",
        "HCLI_MODEL_TOKENS": "16",
        "HCLI_STRUCTURED_OUTPUT_ATTEMPTS": "1",
        "HCLI_MODEL_TIMEOUT": str(int(NATIVE_TIMEOUT_S)),
        "HCLI_READY_TIMEOUT": "45",
    }
    old = {name: os.environ.get(name) for name in names}
    os.environ.update(names)
    try:
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _profile_preflight(profile_path: Path, workspace: Path) -> Dict[str, Any]:
    from hcli.hawking_native import config_for_model_path
    from hcli.machine import host_snapshot
    from hcli.runtime import RuntimePool

    config = config_for_model_path(str(profile_path))
    config.validate()
    identity = config.identity()
    snapshot = host_snapshot()
    from hcli.agentos.resident import memory_decision

    resident_memory = memory_decision(snapshot)
    # This is deliberately a dry plan. RuntimePool._plan_count performs the
    # exact GPU/host admission check but does not spawn a backend; calling
    # RuntimePool.start here would defeat the guard's safety contract.
    pool = RuntimePool(
        str(profile_path),
        requested_n=1,
        workspace=workspace,
        repo_root=REPO,
        topology="slot",
    )
    try:
        planned = pool._plan_count()  # qualification-only dry admission
        runtime_record = pool.admission_records[-1] if pool.admission_records else {}
        runtime_gate = {
            "planned": planned,
            "allow": planned >= 1,
            "refusal_reason": pool.refusal_reason,
            "record": runtime_record,
        }
    finally:
        pool.stop()
    return {
        "profile": str(profile_path),
        "profile_identity": {
            "model_id": identity.get("model_id"),
            "resident_identity": identity.get("resident_identity"),
            "runtime": identity.get("runtime"),
            "protocol": identity.get("protocol"),
            "artifact_inventory": identity.get("artifact_inventory"),
            "mode": identity.get("mode"),
        },
        "snapshot": snapshot,
        "resident_memory": resident_memory,
        "runtime_gate": runtime_gate,
        "safe_to_open_body": bool(resident_memory.get("safe")) and bool(runtime_gate["allow"]),
    }


def _seed_mission(workspace: Path) -> Dict[str, Any]:
    """Persist one native WorkUnit without constructing a native Controller."""
    from hcli.agentos import AgentOS
    from hcli.workunit import WorkUnit

    marker = workspace / "native-resident-marker.txt"
    marker.write_text("resident-native-smoke\n", encoding="utf-8")
    marker_hash = _sha256(marker)

    class SeedEngine:
        def execute_workunit(self, _unit: Any, _context: Mapping[str, Any]) -> Dict[str, Any]:
            raise AssertionError("seed engine must never execute the native WorkUnit")

    verifier = f"test -f {marker} && shasum -a 256 {marker} | grep -q {marker_hash}"
    unit = WorkUnit(
        id="native-resident-one-shot",
        role="resident-smoke",
        description=(
            "Return one concise confirmation that the fixed marker was read. "
            "Do not mutate files; the fixed verifier owns acceptance."
        ),
        resource_class="TEST",
        preferred_backend="resident",
        provider="resident",
        verifier=verifier,
    )
    agent = AgentOS(workspace, engine=SeedEngine(), repo_root=REPO)
    mission = agent.start_mission(
        "Qualify one production resident cognition call with a fixed verifier.",
        units={unit.id: unit},
    )
    return {
        "mission_id": mission.id,
        "marker": str(marker),
        "marker_sha256": marker_hash,
        "verifier": verifier,
        "unit_id": unit.id,
    }


def _stop_and_wait(daemon: Any, state_path: Path) -> Dict[str, Any]:
    try:
        daemon.request_stop()
    except Exception:
        pass
    _wait_for(lambda: _read_json(state_path, {}).get("supervisor_pid") is None, 10.0)
    state = _read_json(state_path, {})
    return {
        "state": state.get("state"),
        "supervisor_pid": state.get("supervisor_pid"),
        "worker_pid": state.get("worker_pid"),
        "body": _read_json(state_path.parent / "body.json", {}),
    }


def _guard_mission(preflight: Mapping[str, Any]) -> Dict[str, Any]:
    from hcli.agentos import ResidentConfig, ResidentDaemon, ResidentStore, start_resident
    from hcli.workunit import WorkUnit

    with tempfile.TemporaryDirectory(prefix="hcli-resident-native-guard-") as tmp:
        workspace = Path(tmp)
        daemon = ResidentDaemon(workspace)
        config = ResidentConfig(
            workspace=str(workspace),
            goal="guard the sealed Qwen 3.8 resident admission",
            model=str(PROFILE),
            repo_root=str(REPO),
            runtime_count=1,
            interval_s=0.15,
            evacuation_grace_s=0.5,
            reserve_bytes=preflight["resident_memory"].get("reserve_bytes"),
            swap_ceiling_bytes=preflight["resident_memory"].get("swap_ceiling_bytes"),
        )
        daemon.configure(config)
        daemon.enqueue_workunit(
            WorkUnit(
                id="native-admission-guard",
                role="resident-smoke",
                description="Remain queued until the sealed resident body is safely admitted.",
                resource_class="TEST",
                provider="resident",
                preferred_backend="resident",
            )
        )
        launch = start_resident(
            workspace,
            goal=config.goal,
            model=str(PROFILE),
            repo_root=REPO,
            runtime_count=1,
            interval_s=0.15,
            evacuation_grace_s=0.5,
            reserve_bytes=config.reserve_bytes,
            swap_ceiling_bytes=config.swap_ceiling_bytes,
        )
        state_path = daemon.store.state_path
        _wait_for(
            lambda: (_read_json(state_path, {}).get("behavior") or {}).get("action")
            in {"WAIT_FOR_MEMORY", "DISPATCH_WORK", "MONITOR_WORKER"},
            5.0,
        )
        state = ResidentStore(workspace).read()
        body = ResidentStore(workspace).body()
        behavior = state.get("behavior") or {}
        checks: Dict[str, bool] = {}
        _check(checks, "profile_uses_sealed_identity", "sealed-3.14" in str(PROFILE))
        _check(checks, "behavior_recorded", bool(behavior))
        _check(checks, "memory_guard_waited", behavior.get("action") == "WAIT_FOR_MEMORY")
        _check(checks, "model_load_disallowed", behavior.get("model_load_allowed") is False)
        _check(checks, "native_work_remained_queued", state.get("inbox_count") == 1)
        _check(checks, "worker_not_started", state.get("worker_pid") is None)
        _check(checks, "body_not_loaded", body.get("loaded") is not True)
        stopped = _stop_and_wait(daemon, state_path)
        _check(checks, "supervisor_stopped", stopped.get("supervisor_pid") is None)
        _check(checks, "worker_stopped", stopped.get("worker_pid") is None)
        _check(checks, "body_unloaded", stopped.get("body", {}).get("loaded") is not True)
        return {
            "mode": "SAFE_MEMORY_GUARD",
            "status": "PASSED" if all(checks.values()) else "FAILED",
            "claim_boundary": "sealed 3.8 profile and resident admission refusal only; native weights were not opened",
            "launch_state": launch.get("state"),
            "state": {
                "behavior": behavior,
                "memory": state.get("memory"),
                "inbox_count": state.get("inbox_count"),
                "resident_id": state.get("resident_id"),
            },
            "body": body,
            "stopped": stopped,
            "checks": checks,
        }


def _native_mission(preflight: Mapping[str, Any]) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="hcli-resident-native-mission-") as tmp:
        result = _native_mission_in_workspace(preflight, Path(tmp))
    result["workspace_cleaned"] = True
    return result


def _native_mission_in_workspace(preflight: Mapping[str, Any], workspace: Path) -> Dict[str, Any]:
    from hcli.agentos import ResidentConfig, ResidentDaemon, start_resident

    seed = _seed_mission(workspace)
    daemon = ResidentDaemon(workspace)
    config = ResidentConfig(
        workspace=str(workspace),
        goal="Qualify one production resident cognition call with a fixed verifier.",
        model=str(PROFILE),
        repo_root=str(REPO),
        runtime_count=1,
        interval_s=0.2,
        evacuation_grace_s=1.0,
        max_restarts=1,
        reserve_bytes=preflight["resident_memory"].get("reserve_bytes"),
        swap_ceiling_bytes=preflight["resident_memory"].get("swap_ceiling_bytes"),
    )
    daemon.configure(config)
    started = time.time()
    state_path = daemon.store.state_path
    with _bounded_generation_env():
        launch = start_resident(
            workspace,
            goal=config.goal,
            model=str(PROFILE),
            repo_root=REPO,
            runtime_count=1,
            interval_s=0.2,
            evacuation_grace_s=1.0,
            max_restarts=1,
            reserve_bytes=config.reserve_bytes,
            swap_ceiling_bytes=config.swap_ceiling_bytes,
        )
        finished = _wait_for(
            lambda: isinstance(_read_json(state_path, {}).get("worker_result"), dict)
            or _read_json(state_path, {}).get("state") in {"FAILED", "STOPPED"},
            NATIVE_TIMEOUT_S + 15.0,
            interval=0.1,
        )
        state_before_stop = _read_json(state_path, {})
        stopped = _stop_and_wait(daemon, state_path)
    mission = _read_json(workspace / ".hcli" / "mission" / "state.json", {})
    mission_log = workspace / ".hcli" / "mission" / "events.jsonl"
    log_text = mission_log.read_text(encoding="utf-8", errors="replace") if mission_log.is_file() else ""
    unit = (mission.get("units") or {}).get(seed["unit_id"], {})
    worker_result = state_before_stop.get("worker_result") or {}
    checks: Dict[str, bool] = {}
    _check(checks, "worker_finished", finished)
    _check(checks, "mission_identity_preserved", mission.get("id") == seed["mission_id"])
    _check(checks, "unit_completed", unit.get("status") == "completed")
    _check(checks, "worker_result_completed", worker_result.get("status") == "completed")
    _check(checks, "native_resident_call_recorded", "hawking-native://" in log_text)
    _check(checks, "sealed_identity_recorded", "sealed-3.14" in log_text)
    _check(checks, "no_worker_left", stopped.get("worker_pid") is None)
    _check(checks, "body_unloaded", stopped.get("body", {}).get("loaded") is not True)
    return {
        "mode": "LIVE_NATIVE_RESIDENT_MISSION",
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "claim_boundary": "one bounded native resident mission and lifecycle receipt; no quality or throughput claim",
        "workspace": str(workspace),
        "mission_id": seed["mission_id"],
        "launch_state": launch.get("state"),
        "elapsed_s": round(time.time() - started, 3),
        "state": {
            "state": state_before_stop.get("state"),
            "last_event": state_before_stop.get("last_event"),
            "worker_returncode": state_before_stop.get("worker_returncode"),
            "restart_count": state_before_stop.get("restart_count"),
            "body": state_before_stop.get("body"),
            "error": state_before_stop.get("error"),
        },
        "worker_result": worker_result,
        "unit": unit,
        "mission_log_tail": log_text[-12000:],
        "stopped": stopped,
        "checks": checks,
    }


def run() -> int:
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "started_at": time.time(),
        "profile_path": str(PROFILE),
        "claim_boundary": "guarded resident qualification; model body opens only after host and runtime admission pass",
    }
    try:
        # Force the profile's resident mode for both preflight and any child
        # process. The bounded environment is restored before this script
        # returns to its caller.
        with _bounded_generation_env():
            with tempfile.TemporaryDirectory(prefix="hcli-resident-native-preflight-") as tmp:
                preflight = _profile_preflight(PROFILE, Path(tmp))
            report["preflight"] = preflight
            if not preflight["safe_to_open_body"]:
                guard = _guard_mission(preflight)
                report["qualification"] = guard
                report["status"] = guard["status"]
            else:
                report["qualification"] = _native_mission(preflight)
            report["status"] = report["qualification"]["status"]
    except Exception as exc:  # noqa: BLE001 - receipt must expose the boundary
        report["status"] = "FAILED"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:3000]}
    report["finished_at"] = time.time()
    atomic_write_json(RECEIPT, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"receipt: {RECEIPT}")
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(run())

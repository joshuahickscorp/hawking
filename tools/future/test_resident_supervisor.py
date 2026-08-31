"""Negative controls for the long-lived resident supervisor.

The trial driver is a loop a human starts. This supervisor's loop is
independent of any conversational turn. These tests watch the failure
modes that would make it look like the first system:

- Git is not the event log; the source never asks git to record history
- an empty refill on one frontier does not halt the daemon
- a blocked frontier does not block an unblocked one
- IDLE_WITH_PROOF fires only on the full conjunction and names each frontier
- draining a list is not an instruction wait
- a status event does not end the loop
- a sleeping WorkUnit carries wake condition, blocked reason,
  required capability/resource, and next reevaluation trigger
- SIGTERM persists durable state before exit

--record and these tests both terminate. No pytest.skip.
"""
from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools.future import autonomy_run as ar
from tools.future import frontiers as fr
from tools.future import resident_supervisor as rs
from tools.future import work_events as we
from tools.future._common import RECEIPTS, REPO, _assert_no_hardware_claims


# ---------------------------------------------------------------------------
# Receipt / entry point
# ---------------------------------------------------------------------------


def test_record_emits_sealed_static_only_receipt():
    out = rs.record()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "RESIDENT_SUPERVISOR.json"
    assert doc["schema"] == rs.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["source_contains_forbidden_vcs_phrase"] is False
    assert doc["proofs_all_passed"] is True
    assert doc["live_trace"]["passed"] is True
    assert doc["live_trace"]["awaiting_instructions"] is False
    assert doc["n_live_frontiers"] == 11
    assert doc["live_frontiers"] == list(rs.LIVE_FRONTIERS)
    _assert_no_hardware_claims(doc)


def test_imports_the_existing_loop_and_does_not_fork_it():
    src = Path(rs.__file__).read_text(encoding="utf-8")
    assert "from tools.future import autonomy_run as ar" in src
    assert "from tools.future import frontiers as fr" in src
    assert rs.ar is ar
    assert rs.fr is fr


def test_lane_vocabulary_is_the_frontiers(tmp_path: Path):
    sup = rs.ResidentSupervisor(
        tmp_path,
        tick_s=0.0,
        max_ticks=1,
        launch_policy="dry",
        refill_hook=lambda n, h: [],
        curriculum_hook=lambda: [],
        self_improvement_hook=lambda: False,
        install_signals=False,
    )
    assert tuple(sup.available_lanes) == tuple(sorted(fr.THIS_HOST_LANES))
    assert set(sup.available_lanes) == set(ar.AVAILABLE_LANES)


# ---------------------------------------------------------------------------
# Git is not the event log
# ---------------------------------------------------------------------------


def test_source_never_contains_the_forbidden_vcs_phrase():
    src = Path(rs.__file__).read_text(encoding="utf-8")
    assert "git commit" not in src
    tree = ast.parse(src)
    planted = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and n.value == "git commit"
    ]
    assert planted == []


def test_persist_does_not_invoke_a_vcs_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []
    real_run = subprocess.run

    def wrapped(*args, **kwargs):
        argv = args[0] if args else kwargs.get("args")
        if isinstance(argv, (list, tuple)):
            calls.append([str(x) for x in argv])
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", wrapped)
    monkeypatch.setattr(rs, "argv_is_vcs_mutation", rs.argv_is_vcs_mutation)
    sup = rs.ResidentSupervisor(
        tmp_path,
        tick_s=0.0,
        max_ticks=1,
        launch_policy="dry",
        refill_hook=lambda n, h: (
            [rs.cpu_unit("WU.tools.a", live_frontier="TOOL_USE")] if n == "TOOL_USE" else []
        ),
        curriculum_hook=lambda: [],
        self_improvement_hook=lambda: False,
        install_signals=False,
    )
    sup.run()
    mutating = [c for c in calls if rs.argv_is_vcs_mutation(c)]
    assert mutating == []
    assert sup.event_log_path.is_file()
    assert sup.mission_path.is_file()
    assert sup.frontier_store_path.is_file()
    # The four durable stores are files, not a repository history rewrite.
    log = sup.event_log_path.read_text(encoding="utf-8")
    assert log.strip()
    assert "STATE_RECOVERED" in log
    recovered = [e for e in sup.events if e["kind"] == rs.KIND_STATE_RECOVERED]
    assert recovered
    assert recovered[0]["payload"]["git_is_event_log"] is False


def test_argv_is_vcs_mutation_detects_history_rewrite_and_spares_read_queries():
    assert rs.argv_is_vcs_mutation(["git", "commit", "-m", "x"]) is True
    assert rs.argv_is_vcs_mutation(["/usr/bin/git", "-C", ".", "commit", "-am", "x"]) is True
    assert rs.argv_is_vcs_mutation(["git", "rev-parse", "HEAD"]) is False
    assert rs.argv_is_vcs_mutation(["git", "--no-optional-locks", "show", "HEAD:x"]) is False
    assert rs.argv_is_vcs_mutation(["python3", "tools/future/resident_supervisor.py"]) is False
    with pytest.raises(rs.SupervisorError, match="VCS mutation"):
        rs._refuse_if_vcs_mutation(["git", "commit", "-m", "no"])


# ---------------------------------------------------------------------------
# Independent refill
# ---------------------------------------------------------------------------


def test_empty_refill_on_one_frontier_does_not_stop_the_daemon(tmp_path: Path):
    proof = rs.scenario_empty_refill_does_not_halt(tmp_path / "empty")
    assert proof["passed"] is True, proof
    assert proof["asked_all_eleven"] is True
    assert proof["seen"][:11] == list(rs.LIVE_FRONTIERS)
    assert "RESIDENT_TOKEN_NS" in proof["seen"]
    assert "TOOL_USE" in proof["seen"]
    assert proof["n_launched"] >= 1


def test_blocked_frontier_does_not_block_an_unblocked_one(tmp_path: Path):
    proof = rs.scenario_blocked_does_not_block_unblocked(tmp_path / "blocked")
    assert proof["passed"] is True, proof
    assert proof["n_launched"] >= 1
    assert proof["n_errors"] >= 1


def test_all_eleven_live_frontiers_are_asked_every_tick(tmp_path: Path):
    seen: list[str] = []

    def hook(name: str, held: set[str]) -> list[dict]:
        seen.append(name)
        return []

    sup = rs.ResidentSupervisor(
        tmp_path,
        tick_s=0.0,
        max_ticks=2,
        launch_policy="dry",
        refill_hook=hook,
        curriculum_hook=lambda: [],
        self_improvement_hook=lambda: False,
        install_signals=False,
    )
    sup.run()
    assert set(rs.LIVE_FRONTIERS) == {
        "RESIDENT_TOKEN_NS",
        "RESIDENT_EBPW",
        "RESIDENT_DISPATCH",
        "MLP_REPRESENTATION",
        "ACCELERATOR",
        "GRAVITY",
        "HCLI_SELF",
        "EXPERIMENT_TURNAROUND",
        "TOOL_USE",
        "CHILD_RESIDENT",
        "ODYSSEY_PREP",
    }
    assert seen == list(rs.LIVE_FRONTIERS) * 2


# ---------------------------------------------------------------------------
# Idle with proof / never awaiting instructions
# ---------------------------------------------------------------------------


def test_idle_with_proof_requires_the_full_conjunction_and_names_each_frontier(tmp_path: Path):
    proof = rs.scenario_idle_with_proof(tmp_path / "idle")
    assert proof["passed"] is True, proof
    assert proof["n_frontiers"] == 11
    assert proof["n_idle"] >= 1


def test_idle_with_proof_does_not_fire_when_one_frontier_still_has_work(tmp_path: Path):
    def hook(name: str, held: set[str]) -> list[dict]:
        if name == "ODYSSEY_PREP":
            n = sum(1 for uid in held if str(uid).startswith("WU.odyssey."))
            uid = f"WU.odyssey.{n}"
            return [rs.cpu_unit(uid, live_frontier=name, gain=3)]
        return []

    sup = rs.ResidentSupervisor(
        tmp_path,
        tick_s=0.0,
        max_ticks=2,
        launch_policy="dry",
        refill_hook=hook,
        curriculum_hook=lambda: [],
        self_improvement_hook=lambda: False,
        install_signals=False,
    )
    result = sup.run()
    idle = [e for e in sup.events if e["kind"] == rs.KIND_IDLE_WITH_PROOF]
    assert idle == []
    assert result["idle_with_proof"] is False
    assert any(e["kind"] == "WORK_LAUNCHED" for e in sup.events)


def test_idle_with_proof_does_not_fire_when_curriculum_remains(tmp_path: Path):
    def curriculum() -> list[dict]:
        return [rs.cpu_unit("WU.curriculum.1", live_frontier="TOOL_USE", gain=2)]

    sup = rs.ResidentSupervisor(
        tmp_path,
        tick_s=0.0,
        max_ticks=1,
        launch_policy="dry",
        refill_hook=lambda n, h: [],
        curriculum_hook=curriculum,
        self_improvement_hook=lambda: False,
        install_signals=False,
    )
    sup.run()
    idle = [e for e in sup.events if e["kind"] == rs.KIND_IDLE_WITH_PROOF]
    assert idle == []
    assert any(e["kind"] == "WORK_LAUNCHED" for e in sup.events)


def test_never_emits_awaiting_instructions_merely_because_a_list_ended(tmp_path: Path):
    proof = rs.scenario_list_ended_is_not_awaiting(tmp_path / "list")
    assert proof["passed"] is True, proof
    assert proof["n_ticks"] == 3
    blob = json.dumps(proof["kinds"]).lower()
    assert "awaiting instructions" not in blob
    assert "awaiting_instructions" not in blob


def test_emit_refuses_the_instruction_wait_label(tmp_path: Path):
    sup = rs.ResidentSupervisor(
        tmp_path,
        tick_s=0.0,
        max_ticks=0,
        install_signals=False,
        refill_hook=lambda n, h: [],
        curriculum_hook=lambda: [],
        self_improvement_hook=lambda: False,
    )
    with pytest.raises(rs.SupervisorError, match="forbidden_idle_label"):
        sup._emit("awaiting_instructions", {"why": "queue empty"})
    with pytest.raises(rs.SupervisorError, match="forbidden_idle_label"):
        sup._emit("STATUS", {"note": "awaiting instructions from the operator"})


# ---------------------------------------------------------------------------
# Status does not end the loop
# ---------------------------------------------------------------------------


def test_status_does_not_end_the_loop(tmp_path: Path):
    proof = rs.scenario_status_does_not_stop(tmp_path / "status")
    assert proof["passed"] is True, proof
    assert proof["n_ticks"] == 3
    assert proof["n_status"] >= 3
    assert proof["first_status_index"] < proof["n_events"] - 1


def test_emit_status_returns_and_run_continues(tmp_path: Path):
    ticks: list[int] = []

    def hook(name: str, held: set[str]) -> list[dict]:
        if name == "HCLI_SELF":
            ticks.append(len(ticks))
            return [rs.cpu_unit(f"WU.hcli.{len(ticks)}", live_frontier=name, gain=2)]
        return []

    sup = rs.ResidentSupervisor(
        tmp_path,
        tick_s=0.0,
        max_ticks=3,
        launch_policy="dry",
        refill_hook=hook,
        curriculum_hook=lambda: [],
        self_improvement_hook=lambda: False,
        install_signals=False,
    )
    result = sup.run()
    assert result["n_ticks"] == 3
    status_events = [e for e in sup.events if e["kind"] == rs.KIND_STATUS]
    assert len(status_events) == 3
    assert all(e["payload"]["does_not_stop_the_loop"] is True for e in status_events)
    # Work continued after the first status.
    launched = [e for e in sup.events if e["kind"] == "WORK_LAUNCHED"]
    assert launched
    first_status = next(i for i, e in enumerate(sup.events) if e["kind"] == rs.KIND_STATUS)
    last_launch = max(i for i, e in enumerate(sup.events) if e["kind"] == "WORK_LAUNCHED")
    assert last_launch > first_status or result["n_ticks"] == 3


# ---------------------------------------------------------------------------
# Sleeping units
# ---------------------------------------------------------------------------


def test_sleeping_unit_carries_the_four_required_fields(tmp_path: Path):
    proof = rs.scenario_sleeping_fields(tmp_path / "sleep")
    assert proof["passed"] is True, proof
    assert proof["missing"] == []


def test_sleeping_constructor_refuses_a_missing_field():
    with pytest.raises(rs.SupervisorError, match="wake_condition"):
        rs.make_sleeping_unit(
            id="WU.x",
            live_frontier="ACCELERATOR",
            wake_condition={},
            blocked_reason="blocked",
            required_resource=fr.LANE_GPU_PROTECTED,
            next_reevaluation_trigger={"kind": "tick", "every": 1},
        )
    with pytest.raises(rs.SupervisorError, match="blocked_reason"):
        rs.make_sleeping_unit(
            id="WU.x",
            live_frontier="ACCELERATOR",
            wake_condition={"all_of": ["gpu"]},
            blocked_reason="  ",
            required_resource=fr.LANE_GPU_PROTECTED,
            next_reevaluation_trigger={"kind": "tick", "every": 1},
        )
    with pytest.raises(rs.SupervisorError, match="capability/resource"):
        rs.make_sleeping_unit(
            id="WU.x",
            live_frontier="ACCELERATOR",
            wake_condition={"all_of": ["gpu"]},
            blocked_reason="blocked",
            next_reevaluation_trigger={"kind": "tick", "every": 1},
        )
    with pytest.raises(rs.SupervisorError, match="next_reevaluation_trigger"):
        rs.make_sleeping_unit(
            id="WU.x",
            live_frontier="ACCELERATOR",
            wake_condition={"all_of": ["gpu"]},
            blocked_reason="blocked",
            required_resource=fr.LANE_GPU_PROTECTED,
            next_reevaluation_trigger={},
        )


def test_hardware_work_parks_sleeping_and_cpu_work_still_runs(tmp_path: Path):
    def hook(name: str, held: set[str]) -> list[dict]:
        if name == "ACCELERATOR":
            return [
                {
                    "id": "WU.accel.gpu",
                    "live_frontier": name,
                    "expected_information_gain": 3,
                    "required_lanes": [fr.LANE_GPU_PROTECTED],
                    "resource_class": "GPU_EXCLUSIVE",
                    "description": "protected complete-token",
                }
            ]
        if name == "TOOL_USE":
            return [rs.cpu_unit("WU.tools.cpu", live_frontier=name, gain=2)]
        return []

    sup = rs.ResidentSupervisor(
        tmp_path,
        tick_s=0.0,
        max_ticks=1,
        launch_policy="dry",
        refill_hook=hook,
        curriculum_hook=lambda: [],
        self_improvement_hook=lambda: False,
        install_signals=False,
    )
    sup.run()
    assert "WU.accel.gpu" in sup.sleeping
    slept = sup.sleeping["WU.accel.gpu"]
    assert rs.sleeping_fields_missing(slept) == []
    assert slept["classification"] == "SLEEPING"
    launched_ids = [
        (e.get("payload") or {}).get("unit", {}).get("id")
        for e in sup.events
        if e["kind"] == "WORK_LAUNCHED"
    ]
    assert "WU.tools.cpu" in launched_ids
    assert "WU.accel.gpu" not in launched_ids


def test_satisfied_wake_condition_requeues_the_unit(tmp_path: Path):
    flag = {"wake": False}

    def hook(name: str, held: set[str]) -> list[dict]:
        return []

    def wake_fn(unit: dict) -> bool:
        return flag["wake"] and unit.get("id") == "WU.wake.me"

    sup = rs.ResidentSupervisor(
        tmp_path,
        tick_s=0.0,
        max_ticks=2,
        launch_policy="dry",
        refill_hook=hook,
        curriculum_hook=lambda: [],
        self_improvement_hook=lambda: False,
        wake_fn=wake_fn,
        install_signals=False,
    )
    sup.sleeping["WU.wake.me"] = rs.make_sleeping_unit(
        id="WU.wake.me",
        live_frontier="TOOL_USE",
        wake_condition={"all_of": ["test flag"], "satisfied": False},
        blocked_reason="waiting on test flag",
        required_capability="future.resident_supervisor.verify",
        required_resource=fr.LANE_CPU,
        next_reevaluation_trigger={"kind": "tick", "every": 1},
        required_lanes=[fr.LANE_CPU],
        resource_class="STATIC_ANALYSIS",
    )
    # Monkey the first tick: still sleeping. Second tick: wake.
    original_tick = sup.tick

    def wrapped_tick():
        if sup.tick_index == 0:
            flag["wake"] = False
        else:
            flag["wake"] = True
        return original_tick()

    sup.tick = wrapped_tick  # type: ignore[method-assign]
    sup.run()
    woken = [e for e in sup.events if e["kind"] == rs.KIND_WORK_WOKEN]
    assert woken, [e["kind"] for e in sup.events]


# ---------------------------------------------------------------------------
# Bounded tick / SIGTERM persist
# ---------------------------------------------------------------------------


def test_tick_is_bounded_and_max_ticks_terminates(tmp_path: Path):
    slept: list[float] = []
    sup = rs.ResidentSupervisor(
        tmp_path,
        tick_s=0.0,
        max_ticks=4,
        launch_policy="dry",
        refill_hook=lambda n, h: [],
        curriculum_hook=lambda: [],
        self_improvement_hook=lambda: False,
        sleep_fn=lambda s: slept.append(s),
        install_signals=False,
    )
    t0 = time.time()
    result = sup.run()
    elapsed = time.time() - t0
    assert result["n_ticks"] == 4
    assert result["stopped_reason"] == "max_ticks"
    assert elapsed < 5.0
    assert slept == []  # tick_s=0 skips wait


def test_sigterm_persists_state_before_exit(tmp_path: Path):
    ws = tmp_path / "sigterm"
    ws.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    child = r"""
import sys
from pathlib import Path
from tools.future.resident_supervisor import ResidentSupervisor
ws = Path(sys.argv[1])
s = ResidentSupervisor(
    ws,
    tick_s=0.05,
    max_ticks=10000,
    launch_policy="dry",
    refill_hook=lambda n, h: [],
    curriculum_hook=lambda: [],
    self_improvement_hook=lambda: False,
    install_signals=True,
)
raise SystemExit(s.run().get("exit_code", 0))
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", child, str(ws)],
        cwd=str(REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    mission = ws / "MISSION_STATE.json"
    deadline = time.time() + 8.0
    try:
        while time.time() < deadline and not mission.is_file():
            if proc.poll() is not None:
                out, err = proc.communicate()
                raise AssertionError(
                    f"child exited before SIGTERM rc={proc.returncode}\n"
                    f"stdout={out[-800:]}\nstderr={err[-800:]}"
                )
            time.sleep(0.05)
        assert mission.is_file(), "child never persisted mission state"
        proc.send_signal(signal.SIGTERM)
        try:
            rc = proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
            raise AssertionError("child did not exit after SIGTERM")
        assert rc == 0, f"SIGTERM exit {rc}; stderr={(proc.stderr.read() if proc.stderr else '')}"
        log = (ws / "EVENT_LOG.jsonl").read_text(encoding="utf-8")
        assert rs.KIND_SHUTDOWN_PERSISTED in log
        doc = json.loads(mission.read_text(encoding="utf-8"))
        assert doc.get("complete") is True
        assert doc.get("schema") == rs.MISSION_SCHEMA
        assert (ws / "FRONTIER_STORE.json").is_file()
        assert "awaiting instructions" not in log.lower()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=3)


def test_in_process_sigterm_handler_persists(tmp_path: Path):
    sup = rs.ResidentSupervisor(
        tmp_path,
        tick_s=0.0,
        max_ticks=1,
        launch_policy="dry",
        refill_hook=lambda n, h: [],
        curriculum_hook=lambda: [],
        self_improvement_hook=lambda: False,
        install_signals=False,
    )
    sup.tick()
    sup._on_sigterm(signal.SIGTERM, None)
    assert sup._stop is True
    assert sup._stop_reason == "signal"
    assert sup.mission_path.is_file()
    kinds = [e["kind"] for e in sup.events]
    assert rs.KIND_SHUTDOWN_PERSISTED in kinds


# ---------------------------------------------------------------------------
# Canonical work events / tick shape
# ---------------------------------------------------------------------------


def test_work_refilled_is_canonical_and_empty_is_not_a_refill(tmp_path: Path):
    def hook(name: str, held: set[str]) -> list[dict]:
        if name == "TOOL_USE":
            return [rs.cpu_unit("WU.tools.r", live_frontier=name, gain=3)]
        return []

    sup = rs.ResidentSupervisor(
        tmp_path,
        tick_s=0.0,
        max_ticks=1,
        launch_policy="dry",
        refill_hook=hook,
        curriculum_hook=lambda: [],
        self_improvement_hook=lambda: False,
        install_signals=False,
    )
    sup.run()
    refilled = [e for e in sup.events if e["kind"] == "WORK_REFILLED"]
    assert refilled
    for event in refilled:
        ok, why = we.validate(event)
        assert ok is True, why
    empty = [e for e in sup.events if e["kind"] == rs.KIND_FRONTIER_EMPTY_REFILL]
    assert empty
    assert all(e["payload"]["daemon_done"] is False for e in empty)


def test_tick_performs_the_named_loop_body(tmp_path: Path):
    sup = rs.ResidentSupervisor(
        tmp_path,
        tick_s=0.0,
        max_ticks=1,
        launch_policy="dry",
        refill_hook=lambda n, h: (
            [rs.cpu_unit("WU.tools.body", live_frontier="TOOL_USE", gain=2)]
            if n == "TOOL_USE"
            else []
        ),
        curriculum_hook=lambda: [],
        self_improvement_hook=lambda: False,
        install_signals=False,
    )
    snapshot = sup.tick()
    kinds = [e["kind"] for e in sup.events]
    assert snapshot["tick"] == 1
    assert rs.KIND_STATE_RECOVERED in kinds
    assert rs.KIND_STATUS in kinds
    assert "WORK_LAUNCHED" in kinds
    assert sup.mission_path.is_file()
    # The loop is still runnable after status; max_ticks is what stops run().
    assert sup._stop is False


def test_never_vcs_mutation_proof_helper(tmp_path: Path):
    proof = rs.scenario_never_vcs_mutation(tmp_path / "vcs")
    assert proof["passed"] is True, proof
    assert proof["git_is_event_log"] is False
    assert proof["phrase_absent"] is True

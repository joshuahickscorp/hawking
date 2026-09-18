"""Lightweight qualification of the resident control plane.

This suite never constructs Controller/RuntimePool and never opens a model.
It proves the lifecycle rules with a tiny fixture engine instead.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from hawking.agentos import AgentOS
from hawking.resident import (
    ResidentConfig,
    ResidentBodyRegistry,
    ResidentDaemon,
    ResidentStore,
    ResidentSupervisor,
    admit_evidence_children,
    memory_decision,
    resident_behavior,
)
from hawking.workunit import WorkUnit


class FixtureEngine:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute_workunit(self, unit, _context):
        self.calls.append(unit.id)
        raw = {
            "content": f"verified {unit.id}",
            "validation": {"ok": True, "verifier": "fixture"},
        }
        if unit.id == "parent":
            raw["child_workunits"] = [{
                "id": "child",
                "role": "research",
                "description": "inspect the verified parent receipt",
                "resource_class": "CPU_SHARED",
            }]
        return raw


def test_memory_pressure_waits_without_loading_a_model() -> None:
    unsafe = memory_decision(
        {
            "pressure": "high",
            "total_bytes": 100,
            "free_bytes": 1,
            "swap_used_bytes": 0,
        },
        reserve_bytes=20,
    )
    assert unsafe["safe"] is False
    assert "host memory pressure is high" in unsafe["reasons"]

    unknown = memory_decision({"pressure": "unknown", "total_bytes": 0})
    assert unknown["safe"] is False
    assert "memory admission is unknown" in unknown["reasons"][-1]

    swap_packed = memory_decision(
        {
            "pressure": "normal",
            "total_bytes": 100,
            "free_bytes": 80,
            "swap_used_bytes": 3 * 1024**3,
        },
        reserve_bytes=20,
    )
    assert swap_packed["safe"] is False
    assert swap_packed["swap_ceiling_bytes"] == 2 * 1024**3
    assert "swap" in swap_packed["reasons"][0]


def test_verified_evidence_refills_a_durable_mission() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        engine = FixtureEngine()
        parent = WorkUnit(id="parent", role="research", description="bounded parent")
        agent = AgentOS(workspace, engine=engine)
        agent.start_mission("resident fixture", units={"parent": parent})
        first = agent.run()

        assert first["status"] == "completed"
        assert first["evidence"][0]["accepted"] is True
        daemon = ResidentDaemon(workspace)
        rows = daemon.refill_from_evidence(agent.mission, first["evidence"])
        assert rows[0]["status"] == "ADMITTED"
        assert agent.mission.scheduler.units["child"].dependencies == ["parent"]

        second = agent.continue_mission()
        assert second["status"] == "completed"
        assert engine.calls == ["parent", "child"]

        restored = AgentOS(workspace, engine=FixtureEngine())
        recovered = restored.recover_mission()
        assert recovered.scheduler.units["child"].status == "completed"
        assert recovered.id == agent.mission.id
        assert (workspace / ".hawking" / "mission" / "state.json").is_file()


def test_verified_evidence_preserves_and_admits_more_than_eight_children() -> None:
    class ManyChildrenEngine(FixtureEngine):
        def execute_workunit(self, unit, context):
            raw = super().execute_workunit(unit, context)
            if unit.id == "parent":
                raw["child_workunits"] = [
                    {
                        "id": f"child-{index:02d}",
                        "role": "research",
                        "description": f"inspect receipt {index}",
                        "resource_class": "CPU_SHARED",
                    }
                    for index in range(16)
                ]
            return raw

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        parent = WorkUnit(id="parent", role="research", description="parent")
        agent = AgentOS(workspace, engine=ManyChildrenEngine())
        agent.start_mission("many children", units={"parent": parent})
        first = agent.run()
        assert len(first["evidence"][0]["child_workunits"]) == 16
        rows = ResidentDaemon(workspace).refill_from_evidence(
            agent.mission, first["evidence"]
        )
        assert len(rows) == 16
        assert all(row["status"] == "ADMITTED" for row in rows)


def test_unverified_evidence_cannot_create_children() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        parent = WorkUnit(id="parent", role="research", description="bounded parent")
        agent = AgentOS(tmp, engine=FixtureEngine())
        agent.start_mission("resident fixture", units={"parent": parent})
        rows = admit_evidence_children(
            agent.mission,
            {
                "unit_id": "parent",
                "accepted": False,
                "validation": {"ok": False},
                "child_workunits": [{"id": "bad", "description": "must not run"}],
            },
        )
        assert rows == []
        assert "bad" not in agent.mission.scheduler.units


def test_resident_children_are_explicitly_parented_and_durable() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        daemon = ResidentDaemon(workspace)
        daemon.configure(ResidentConfig(workspace=str(workspace), goal="fixture"))
        job = daemon.launch_child(
            [sys.executable, "-c", "print('child')"],
            timeout_s=5,
        )
        assert job["parent_job_id"].startswith("resident-")
        store = ResidentStore(workspace)
        assert job["job_id"] in store.read()["child_job_ids"]

        # A bound against hanging, NOT a performance assertion. Five seconds
        # was a guess at how long a `print('child')` needs to be observed
        # finished, and under the sharded runner the box is busy enough that
        # the guess failed about one run in six -- reporting the last polled
        # state, which was still RUNNING. The wait ends as soon as the job is
        # terminal, so a generous ceiling costs nothing when it works.
        from hawking.background import BackgroundJobStore

        deadline = time.time() + 60
        status = BackgroundJobStore(workspace).inspect(job["job_id"])
        while time.time() < deadline and status["state"] not in {"COMPLETED", "FAILED"}:
            time.sleep(0.05)
            status = BackgroundJobStore(workspace).inspect(job["job_id"])
        assert status["state"] == "COMPLETED", (
            f"child job ended in {status['state']!r}, not COMPLETED: {status}"
        )


def test_resident_behavior_and_clean_room_are_model_free() -> None:
    waiting = resident_behavior(
        {"clean_room_requested": True, "clean_room_reason": "GPU proof"},
        {"safe": True},
        mission_has_work=True,
        inbox_has_work=True,
        max_restarts=3,
    )
    assert waiting["action"] == "WAIT_FOR_CLEAN_ROOM"
    assert waiting["model_load_allowed"] is False
    assert waiting["unrelated_process_kill_allowed"] is False

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        daemon = ResidentDaemon(workspace)
        daemon.configure(ResidentConfig(workspace=str(workspace), goal="fixture"))
        queued = daemon.enqueue_workunit(
            WorkUnit(id="queued", role="research", description="queued fixture work")
        )
        assert queued["status"] == "QUEUED"
        assert ResidentStore(workspace).read_inbox()[0]["id"] == "queued"

        clean = daemon.request_clean_room("protected fixture")
        assert clean["clean_room_requested"] is True
        assert clean["clean_room_reason"] == "protected fixture"
        resumed = daemon.resume_clean_room()
        assert resumed["clean_room_requested"] is False

        body = ResidentBodyRegistry(workspace)
        body.mark_loading(pid=123)
        assert body.mark_loaded(pid=123)["status"] == "LOADED"
        unloaded = body.mark_unloaded()
        assert unloaded["loaded"] is False
        assert unloaded["worker_pid"] is None


def test_unbound_resident_never_spawns_a_worker_for_legacy_auto_discovery() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        daemon = ResidentDaemon(workspace)
        daemon.configure(ResidentConfig(workspace=str(workspace), goal="fixture"))
        supervisor = ResidentSupervisor(daemon.store.state_path)
        with patch("hawking.resident.subprocess.Popen") as spawn:
            supervisor._spawn_worker()

        spawn.assert_not_called()
        state = ResidentStore(workspace).read()
        assert state["state"] == "WAITING_FOR_NATIVE_MODEL"
        assert state["last_event"] == "waiting_for_native_model"


def test_explicit_model_waits_for_gpu_admission_without_spawning() -> None:
    repo = Path(__file__).resolve().parents[2]
    profile = repo / "hawking" / "hawking-native.sealed-3.14.json"
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        daemon = ResidentDaemon(workspace)
        config = ResidentConfig(
            workspace=str(workspace),
            goal="gpu admission fixture",
            model=str(profile),
            repo_root=str(repo),
        )
        daemon.configure(config)
        supervisor = ResidentSupervisor(daemon.store.state_path)
        snapshot = {
            "pressure": "normal",
            "total_bytes": 100 * 1024**3,
            "free_bytes": 80 * 1024**3,
            "swap_used_bytes": 0,
        }
        metal = {
            "recommendedMaxWorkingSetSize": 0,
            "currentAllocatedSize": 0,
            "source": "fixture",
        }
        with patch("hawking.machine.host_snapshot", return_value=snapshot), patch(
            "hawking.machine.metal_device_info", return_value=metal
        ):
            decision = supervisor._memory(config)

        assert decision["safe"] is False
        assert decision["runtime_gate"]["allow"] is False
        assert decision["runtime_gate"]["gate"] == "artifact"
        assert ResidentStore(workspace).read().get("worker_pid") is None

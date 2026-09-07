"""A steer injected from a DETACHED process must reach the running mission.

The Odyssey contract treats detached supervision as normal: Claude attaches,
injects a steer, detaches, and HCLI continues. Measured 2026-09-05, before this
fix:

    mission session_id                    3f254d6d-...
    the steer landed in                   c22b3aa4-...   (the INJECTING session)
    file for the mission's session        did not exist

`SteeringQueue` is keyed by session id and the mission polls only its OWN file, so
the steer was orphaned. The CLI printed "✓ Steer queued" either way, which is the
worst version of this bug: a silent success.
"""
from __future__ import annotations

import json
from pathlib import Path

import hcli.controller as ctrl


def _write_mission(root: Path, mission_id: str, session_id: str, phase: str = "running"):
    d = root / ".hcli" / "mission"
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps(
        {"id": mission_id, "session_id": session_id, "phase": phase}), encoding="utf-8")


def test_live_mission_identity_is_read_from_disk(tmp_path):
    _write_mission(tmp_path, "M-1", "S-1")
    got = ctrl._live_mission_identity(str(tmp_path))
    assert got == {"id": "M-1", "session_id": "S-1"}


def test_a_finished_mission_is_not_a_steer_target(tmp_path):
    """Negative control: only a LIVE mission may capture a detached steer."""
    _write_mission(tmp_path, "M-1", "S-1", phase="failed")
    assert ctrl._live_mission_identity(str(tmp_path)) is None


def test_no_mission_on_disk_is_not_an_error(tmp_path):
    assert ctrl._live_mission_identity(str(tmp_path)) is None


def test_a_detached_steer_lands_in_the_missions_queue_with_provenance(tmp_path):
    from hcli.steering import SteeringQueue

    _write_mission(tmp_path, "M-9", "S-9")
    live = ctrl._live_mission_identity(str(tmp_path))
    assert live is not None

    q = SteeringQueue(str(tmp_path), live["session_id"])
    q.enqueue("look at the miss path", kind="knowledge",
              mission_id=live["id"], source_session_id="SUPERVISOR-1")

    target = tmp_path / ".hcli" / "steering" / "S-9.json"
    assert target.is_file(), (
        "the steer did not land in the MISSION's queue; the mission polls only its "
        "own session file, so anywhere else is an orphan"
    )
    events = json.loads(target.read_text(encoding="utf-8"))
    assert events[0]["mission_id"] == "M-9", "the steer does not name the mission it steers"
    assert events[0]["source_session_id"] == "SUPERVISOR-1", (
        "the steer does not record who injected it"
    )


def test_a_running_queue_absorbs_a_steer_appended_by_another_process(tmp_path):
    """Routing the steer to the right FILE was only half the path.

    SteeringQueue loads once at construction and `all()` returned that cached
    list, so a running mission held a stale copy and never saw an externally
    injected steer. Mission._unit_context feeds `self._steering.all()` into the
    worker packet, so an unseen steer is an unread steer.
    """
    import time
    from hcli.steering import SteeringQueue

    mission_q = SteeringQueue(str(tmp_path), "S-1")
    mission_q.enqueue("the mission's own", kind="knowledge")
    assert len(mission_q.all()) == 1

    time.sleep(0.02)
    supervisor_q = SteeringQueue(str(tmp_path), "S-1")   # a DETACHED process
    supervisor_q.enqueue("injected from outside", kind="knowledge")

    seen = [e.text for e in mission_q.all()]
    assert "injected from outside" in seen, (
        "the running queue still cannot see a steer another process appended"
    )
    assert "the mission's own" in seen, (
        "absorbing external events dropped the queue's own; merge, never replace"
    )


def test_absorbing_does_not_duplicate_on_repeated_reads(tmp_path):
    """Negative control: all() is called per WorkUnit, so it must be idempotent."""
    from hcli.steering import SteeringQueue

    q = SteeringQueue(str(tmp_path), "S-2")
    q.enqueue("one", kind="knowledge")
    first = len(q.all())
    for _ in range(5):
        assert len(q.all()) == first, "repeated reads duplicated events"


def test_the_whole_chain_a_detached_steer_reaches_the_worker_packet(tmp_path):
    """End to end: inject from outside -> mission absorbs -> it lands in the packet.

    Each link was broken separately. Routing sent the steer to a file the mission
    never read; the queue then held a stale in-memory copy. This asserts the whole
    path, because two working halves and a broken joint is still a steer nobody
    acted on.
    """
    import time

    from hcli.goal import compile_worker_context
    from hcli.steering import SteeringQueue
    from hcli.workunit import WorkUnit

    mission_q = SteeringQueue(str(tmp_path), "S-LIVE")   # mission starts first
    time.sleep(0.02)
    SteeringQueue(str(tmp_path), "S-LIVE").enqueue(       # DETACHED supervisor
        "USE prefill_profile totals", kind="knowledge",
        mission_id="M-LIVE", source_session_id="SUPERVISOR")

    events = mission_q.all()          # exactly what Mission._unit_context passes
    assert [e.text for e in events] == ["USE prefill_profile totals"]
    assert events[0].mission_id == "M-LIVE"
    assert events[0].source_session_id == "SUPERVISOR"

    wu = WorkUnit(id="implement", role="implementation", description="do the thing")
    packet = compile_worker_context(
        wu, {}, phase="running", units={"implement": wu}, steering=events,
        failure_context=None, ledger=None, goal_ref="ref",
    )
    blob = packet if isinstance(packet, str) else str(packet)
    assert "USE prefill_profile totals" in blob, (
        "the steer reached the mission's queue but never entered the worker packet"
    )
    assert "[knowledge]" in blob, "the steer's kind was dropped on the way in"

"""Reissuing an unfinished mission's goal must ADOPT it, not replace it.

Before this, ``/mission <goal>`` -> ``Controller.run_mission`` always called
``Mission(...)``. That constructor builds a ``Scheduler``, whose ``__init__``
persists ``dag.json`` -- and ``content_identity()`` ignores status, so a
content-identical graph is neither an ``IdentityConflict`` nor a retirement.
Measured: completed units came back ``pending``, ``accepted_count`` went 1 -> 0,
and the prior mission id survived only in the append-only mission.log.
"""

from __future__ import annotations

import pytest

from hcli.controller import Controller
from hcli.mission import Mission

GOAL = "count the lines in one file and report the number"


def _prime(workspace, phase):
    """Leave a mission on disk with one unit already completed."""
    mission = Mission(workspace, goal=GOAL, quiet=True)
    mission.scheduler.units["implement"].status = "completed"
    mission.accepted_count = 1
    mission.phase = phase
    mission.checkpoint()
    return mission.id


def _run_without_engine(controller, goal):
    """Exercise run_mission's construction fork; Mission.run is not the subject."""
    original = Mission.run
    Mission.run = lambda self: {"status": "running"}
    try:
        controller.run_mission(goal, engine=object())
    finally:
        Mission.run = original
    return controller.mission


@pytest.mark.parametrize("phase", ["running", "evacuated", "idle"])
def test_unfinished_same_goal_is_resumed(tmp_path, phase):
    first = _prime(tmp_path, phase)
    mission = _run_without_engine(Controller(str(tmp_path)), GOAL)
    assert mission.id == first
    assert mission.scheduler.units["implement"].status == "completed"
    assert mission.accepted_count == 1


@pytest.mark.parametrize("phase", ["completed", "cancelled", "failed", "no_progress"])
def test_terminal_phase_is_a_restart_not_a_resume(tmp_path, phase):
    first = _prime(tmp_path, phase)
    mission = _run_without_engine(Controller(str(tmp_path)), GOAL)
    assert mission.id != first


def test_a_different_goal_never_adopts(tmp_path):
    first = _prime(tmp_path, "running")
    mission = _run_without_engine(
        Controller(str(tmp_path)), "a completely different objective: audit receipts"
    )
    assert mission.id != first


def test_an_unreadable_mission_still_runs(tmp_path, monkeypatch):
    """Restoring is the optimisation; a corrupt state must not kill /mission."""
    _prime(tmp_path, "running")

    def explode(*_a, **_kw):
        raise ValueError("state.json is not a mission")

    monkeypatch.setattr(Mission, "from_workspace", staticmethod(explode))
    mission = _run_without_engine(Controller(str(tmp_path)), GOAL)
    assert mission is not None
    assert mission.goal == GOAL


def test_an_adopted_mission_adopts_the_ledger_too(tmp_path):
    """/steer reads controller._ledger; adopting without it drops constraints."""
    _prime(tmp_path, "running")
    goal_md = tmp_path / ".hcli" / "GOAL.md"
    goal_md.parent.mkdir(parents=True, exist_ok=True)
    goal_md.write_text(f"# Ultragoal\n\n{GOAL}\n\n", encoding="utf-8")

    controller = Controller(str(tmp_path))
    assert controller._ledger is None
    _run_without_engine(controller, GOAL)
    assert controller._ledger is not None

"""A mission that has given up must stop the resident, not spin it.

`_mission_has_work` read UNIT status, and a terminal mission keeps `failed`
and `pending` units on disk forever. So the supervisor saw work every tick,
dispatched a worker, the worker recovered the dead mission, reported its
failure and exited 0 -- leaving `failure_streak` at 0, so `max_restarts` never
tripped. Observed live at cycles=60 and climbing, one dispatch every 15s, with
no path out and no signal to the operator.
"""
from __future__ import annotations

import json

import pytest

from hcli.agentos.resident import (
    BLOCKED_MISSION_PHASES,
    TERMINAL_MISSION_PHASES,
    _mission_has_work,
    mission_blocked_reason,
    resident_behavior,
)


def _write_mission(tmp_path, phase, units=None):
    d = tmp_path / ".hcli" / "mission"
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps({
        "version": 1,
        "id": "m-1",
        "goal": "g",
        "phase": phase,
        "units": units if units is not None else {
            "implement": {"id": "implement", "role": "research",
                          "description": "d", "status": "failed"},
            "validate": {"id": "validate", "role": "research",
                         "description": "d", "status": "pending"},
        },
    }))
    return tmp_path


@pytest.mark.parametrize("phase", sorted(TERMINAL_MISSION_PHASES))
def test_a_terminal_mission_is_not_available_work(tmp_path, phase):
    _write_mission(tmp_path, phase)
    assert _mission_has_work(tmp_path) is False, (
        f"phase={phase} still reads as work; this is the 15s dispatch spin"
    )


def test_a_running_mission_with_a_failed_unit_is_still_work(tmp_path):
    """The repair budget lives inside the mission; do not pre-empt it."""
    _write_mission(tmp_path, "running")
    assert _mission_has_work(tmp_path) is True


@pytest.mark.parametrize("phase", sorted(BLOCKED_MISSION_PHASES))
def test_a_blocked_mission_names_why_it_needs_a_human(tmp_path, phase):
    _write_mission(tmp_path, phase)
    reason = mission_blocked_reason(tmp_path)
    assert reason and phase in reason and "m-1" in reason


def test_completed_is_terminal_but_not_blocked(tmp_path):
    """A completed mission is re-run harmlessly; that is how a bank promotes."""
    _write_mission(tmp_path, "completed")
    assert _mission_has_work(tmp_path) is False
    assert mission_blocked_reason(tmp_path) is None


def test_absent_and_corrupt_missions_are_unchanged(tmp_path):
    assert mission_blocked_reason(tmp_path) is None
    assert _mission_has_work(tmp_path) is False
    d = tmp_path / ".hcli" / "mission"
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text("{not json")
    # Corrupt must still spawn a worker so the failure becomes visible.
    assert _mission_has_work(tmp_path) is True
    assert mission_blocked_reason(tmp_path) is None


def test_the_spin_is_gone_at_the_behavior_layer(tmp_path):
    """End of the chain: no mission work and no bank work means WAIT, not DISPATCH."""
    _write_mission(tmp_path, "failed")
    decision = resident_behavior(
        {"failure_streak": 0},
        {"safe": True},
        mission_has_work=_mission_has_work(tmp_path),
        inbox_has_work=False,
        max_restarts=3,
    )
    assert decision["action"] == "WAIT_FOR_WORK", decision
    assert decision["model_load_allowed"] is False

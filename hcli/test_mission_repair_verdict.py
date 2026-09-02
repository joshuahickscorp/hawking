"""A repair that SUCCEEDED must not still end the mission `failed`.

The repair budget exists so one failed WorkUnit is not the end of a run:
``scheduler.fail`` emits a descendant whose ``repairs`` field names the root. But
the end-of-run verdict counted every unit whose status is ``failed`` -- including
a root whose repair then completed. Nothing ever clears a root's status, so ANY
mission that repaired anything finished ``phase=failed``.

``failed`` is in ``BLOCKED_MISSION_PHASES``, so the supervisor stopped and asked
for a human. The repair budget could never pay out: a mission was allowed to
repair itself and then punished for having needed to.

The negative control is the second half of every test here -- an UNrepaired
failure, and a repair that itself failed, must both still be terminal, or this
change would just be "stop reporting failures".
"""
from __future__ import annotations

import json

import pytest

from hcli.agentos.resident import mission_blocked_reason
from hcli.mission import Mission
from hcli.workunit import WorkUnit


def _units(**statuses):
    """id -> WorkUnit, where a name containing '.repair.' repairs its prefix."""
    units = {}
    for uid, status in statuses.items():
        repairs = uid.split(".repair.")[0] if ".repair." in uid else None
        units[uid] = WorkUnit(
            id=uid,
            role="research",
            description=uid,
            status=status,
            repairs=repairs,
        )
    return units


def _mission(tmp_path, units):
    m = Mission(tmp_path, goal="g", units=units, engine=object(), install_signals=False)
    return m


def test_a_completed_repair_clears_its_root(tmp_path):
    m = _mission(tmp_path, _units(**{"a": "failed", "a.repair.1": "completed"}))
    assert m._unrepaired_failures() == []


def test_an_unrepaired_failure_is_still_a_failure(tmp_path):
    """Negative control. Without this the fix is just silence."""
    m = _mission(tmp_path, _units(**{"a": "failed"}))
    assert m._unrepaired_failures() == ["a"]


def test_a_repair_that_also_failed_is_still_a_failure(tmp_path):
    """The exhausted-lineage case. This is what keeps a dead mission terminal."""
    m = _mission(tmp_path, _units(**{"a": "failed", "a.repair.1": "failed"}))
    assert sorted(m._unrepaired_failures()) == ["a", "a.repair.1"]


def test_a_repair_still_running_does_not_yet_clear_its_root(tmp_path):
    m = _mission(tmp_path, _units(**{"a": "failed", "a.repair.1": "ready"}))
    assert m._unrepaired_failures() == ["a"]


def _write_mission(tmp_path, phase, units):
    d = tmp_path / ".hcli" / "mission"
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps({
        "version": 1, "id": "m-1", "goal": "g", "phase": phase, "units": units,
    }))


def test_the_supervisor_does_not_block_on_a_repaired_root(tmp_path):
    """mission_blocked_reason reads the same rule off disk, or the fix is half-done."""
    _write_mission(tmp_path, "completed", {
        "a": {"id": "a", "role": "research", "description": "d", "status": "failed"},
        "a.repair.1": {"id": "a.repair.1", "role": "research", "description": "d",
                       "status": "completed", "repairs": "a"},
    })
    assert mission_blocked_reason(tmp_path) is None


def test_the_supervisor_still_blocks_on_an_unrepaired_root(tmp_path):
    _write_mission(tmp_path, "completed", {
        "a": {"id": "a", "role": "research", "description": "d", "status": "failed"},
    })
    reason = mission_blocked_reason(tmp_path)
    assert reason is not None and "failed units" in reason


@pytest.mark.parametrize("phase", ["failed", "cancelled", "no_progress"])
def test_terminal_phases_are_untouched_by_this(tmp_path, phase):
    _write_mission(tmp_path, phase, {
        "a": {"id": "a", "role": "research", "description": "d", "status": "failed"},
        "a.repair.1": {"id": "a.repair.1", "role": "research", "description": "d",
                       "status": "completed", "repairs": "a"},
    })
    assert mission_blocked_reason(tmp_path) is not None

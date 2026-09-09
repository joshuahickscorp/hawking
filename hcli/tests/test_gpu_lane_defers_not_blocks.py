"""S006 21-25, 35: an externally-held GPU lane defers work; it does not fail it.

Admission counted only THIS scheduler's own running units, so a background
sweep or another session was invisible: the scheduler would start a GPU
measurement on top of one already running and contaminate both (S006 25).

The fix is two lines and they only work together. Deferring GPU units without
touching mission.py would be WORSE than the bug: the stall detector fails a
mission after 50 spins at 10 ms -- about half a second -- so the first deferred
unit would kill the run outright, which is the opposite of S006 35's "wait, then
return when the condition clears".
"""
import os
from pathlib import Path

import pytest

from hcli.resources import ResourceLimits
from hcli.workunit import (
    GPU_CLASSES,
    WorkUnit,
    assign_ready,
    gpu_lane_externally_held,
)


@pytest.fixture
def lane(tmp_path, monkeypatch):
    p = tmp_path / "gpu-lane.lock"
    monkeypatch.setenv("HAWKING_GPU_LANE_LOCK", str(p))
    return p


def _unit(uid, rc):
    return WorkUnit(id=uid, role="implement", description="d",
                    resource_class=rc, status="ready")


def _assign(units):
    return assign_ready(list(units.values()), 4, all_units=units,
                        limits=ResourceLimits.resolve())


def test_the_predicate_reads_the_lock_and_never_creates_it(lane):
    assert not gpu_lane_externally_held()
    assert not lane.exists(), "the check must READ the lock, never acquire it"
    lane.mkdir()
    assert gpu_lane_externally_held()


def test_a_regular_file_at_the_lock_path_is_not_a_lock(lane):
    """mkdir-atomic locks are DIRECTORIES. A stray file must not read as held,
    or a wedge would silently stop all GPU work forever."""
    lane.write_text("not a lock")
    assert not gpu_lane_externally_held()


@pytest.mark.parametrize("rc", sorted(GPU_CLASSES))
def test_gpu_work_is_deferred_while_the_lane_is_held(lane, rc):
    lane.mkdir()
    units = {"g": _unit("g", rc)}
    out = _assign(units)
    assert out == [], f"{rc} was admitted against a held lane: {out}"
    assert units["g"].status == "ready", "deferred work must stay READY, not fail"


@pytest.mark.parametrize("rc", sorted(GPU_CLASSES))
def test_it_is_admitted_again_the_moment_the_lane_clears(lane, rc):
    lane.mkdir()
    units = {"g": _unit("g", rc)}
    assert _assign(units) == []
    lane.rmdir()                                   # the external job finished
    out = _assign(units)
    assert out, "a cleared lane must let the unit through with no other prompt"


def test_non_gpu_work_runs_straight_through_a_held_lane(lane):
    """S006 36: independent branches. A held GPU lane must not stall CPU work."""
    lane.mkdir()
    units = {"c": _unit("c", "CPU_HEAVY"), "g": _unit("g", "GPU_DECODE")}
    out = _assign(units)
    assigned = {(t[0].id if hasattr(t[0], "id") else t[0]) for t in out}
    assert "c" in assigned, f"CPU work was blocked by a held GPU lane: {out}"
    assert "g" not in assigned, "GPU work should still be deferred"


def test_mission_treats_a_held_lane_as_wait_not_blocked():
    """The stall detector must not count an external hold toward 'blocked'."""
    src = Path("hcli/mission.py").read_text()
    i = src.index("idle_spins += 1")
    window = src[max(0, i - 1200):i]
    assert "gpu_lane_externally_held()" in window, (
        "the external-hold check must come BEFORE idle_spins accumulates, or a "
        "deferred GPU unit fails the mission in ~0.5s")
    assert "time.sleep(POLL_S)" in window, "an external hold must back off, not hot-spin"

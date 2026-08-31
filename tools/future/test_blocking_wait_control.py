"""G086's permanent negative control: the archived failure as a fixture.

S025 §29 names the scenario exactly.

    WorkUnit A launches a long subprocess.
    WorkUnits B/C/D are independently runnable.
    PASS only if A remains active WHILE B/C/D launch.
    If the supervisor waits on A: FAIL. No idle-event loophole.

The loophole is the part worth stating. G037's repair refuses an IDLE EVENT
taken while work is runnable, and a blocking subprocess wait never emits one -
so the condition passed while the instrument failed on the same timeline. This
control is written against the TIMELINE SHAPE rather than against any event the
driver chooses to emit.
"""
from __future__ import annotations

import pytest

from tools.future import no_wait_orchestration as nwo
from tools.future import runnability_snapshot as rs


def _launch(t, uid, detached=False):
    payload = {"unit": {"id": uid, "frontier_id": f"FT.{uid}"}}
    if detached:
        payload["launch"] = "detached"
    return {"kind": "workunit_launched", "t_s": t, "payload": payload}


def _started(t, uid):
    return {"kind": "detached_started", "t_s": t, "payload": {"unit_id": uid}}


def _done(t, uid):
    return {"kind": "result_ingested", "t_s": t, "payload": {"unit_id": uid}}


def _the_failure():
    """A launched, then nothing until A completes 300 s later.

    The timeline must SHOW that B/C/D were runnable during the wait, because the
    classifier rightly declines to convict a wait it cannot see work behind - it
    returns SLOW_BUT_CORRECT for "nothing runnable". That is the archived shape
    too: the 477 s run ended with twelve frontiers holding novel work, which is
    what made it a failure rather than a slow but honest wait.
    """
    return [
        _launch(0, "WU.A"),
        {"kind": "work_refilled", "t_s": 1,
         "payload": {"unit_ids": ["WU.B", "WU.C", "WU.D"], "n": 3}},
        _done(300, "WU.A"),
    ]


def _the_pass():
    """A stays active WHILE B, C and D launch."""
    return [
        _launch(0, "WU.A"),
        _launch(1, "WU.B", detached=True), _started(1, "WU.B"),
        _launch(2, "WU.C", detached=True), _started(2, "WU.C"),
        _launch(3, "WU.D", detached=True), _started(3, "WU.D"),
        _done(300, "WU.A"),
    ]


def test_the_archived_failure_shape_still_fails():
    """If this ever passes, the guard has been weakened, not the driver fixed."""
    got = nwo.classify(_the_failure())
    assert got["verdict"] == nwo.FAIL_NO_WAIT_ORCHESTRATION
    assert got["n_forcing_intervals"] >= 1


def test_launching_b_c_and_d_while_a_runs_passes():
    got = nwo.classify(_the_pass())
    assert got["n_forcing_intervals"] == 0, got.get("reason")


def test_the_pass_requires_the_detached_launches_to_be_PROVEN():
    """Relabelling A's neighbours as detached without a detached_started must not
    buy the pass - that is a driver talking its way out of an interval."""
    claimed_only = [
        _launch(0, "WU.A"),
        _launch(1, "WU.B", detached=True),   # claimed, never started
        _done(300, "WU.A"),
    ]
    ids = nwo._verified_detached_ids(claimed_only)
    assert ids == set(), "a claim without a detached_started is not proof"


def test_an_idle_event_cannot_launder_the_wait():
    """No idle-event loophole: the shape is judged, not the driver's narration."""
    with_idle = _the_failure()[:1] + [
        {"kind": "idle_justified", "t_s": 1,
         "payload": {"asked": [], "waiting_on": ["WU.A"]}},
    ] + _the_failure()[1:]
    got = nwo.classify(with_idle)
    assert got["verdict"] == nwo.FAIL_NO_WAIT_ORCHESTRATION, (
        "an idle_justified event must not excuse a blocking subprocess wait"
    )


def test_the_snapshot_would_have_separated_the_two_cases():
    """B/C/D runnable at the wait is the evidence the timelines never recorded."""
    snap = rs.snapshot(
        [{"id": "FT.WU.B", "entry_ids": ["WU.B"]},
         {"id": "FT.WU.C", "entry_ids": ["WU.C"]},
         {"id": "FT.WU.D", "entry_ids": ["WU.D"]}],
        already_launched=["WU.A"], scar_dead=[], t_s=1.0,
    )
    verdict = rs.classify_wait(snap, waiting_on=["WU.A"])
    assert verdict["verdict"] == rs.IDLE_WITH_RUNNABLE_WORK
    assert verdict["n_runnable"] == 3

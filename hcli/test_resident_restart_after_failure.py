"""`resident start` must be able to revive a resident that hit the limit.

`failure_streak` is what `resident_behavior` spends against `max_restarts`,
and it only ever reset when a worker exited 0. A resident at the limit is
FAILED and never spawns another worker, so the streak could not come down:
`hcli resident start` re-configured with `failure_streak` preserved, the
supervisor read `streak >= max_restarts` on its first pass, wrote
state=FAILED and exited. The only escape was hand-editing
`.hcli/resident/state.json`.
"""
from __future__ import annotations

from hcli.agentos.resident import (
    ResidentConfig,
    ResidentDaemon,
    resident_behavior,
)


def _config(tmp_path, goal="a goal"):
    return ResidentConfig(workspace=str(tmp_path), goal=goal)


def test_start_clears_a_streak_that_reached_the_restart_limit(tmp_path):
    daemon = ResidentDaemon(tmp_path)
    daemon.configure(_config(tmp_path))
    daemon.store.update(failure_streak=3, state="FAILED", restart_count=7)
    assert daemon.store.read()["failure_streak"] == 3

    state = daemon.configure(_config(tmp_path))

    assert state["failure_streak"] == 0, "start must clear the exhausted streak"
    assert state["restart_count"] == 7, "the lifetime counter is not a streak"

    # The behavioral consequence, not just the field: the supervisor's very
    # next decision must be work, not ESCALATE_FAILURE.
    decision = resident_behavior(
        state,
        {"safe": True},
        mission_has_work=True,
        inbox_has_work=False,
        max_restarts=3,
    )
    assert decision["action"] == "DISPATCH_WORK", decision


def test_a_streak_below_the_limit_still_escalates_when_it_reaches_it():
    """The budget itself is untouched: this is not a way to never fail."""
    decision = resident_behavior(
        {"failure_streak": 3},
        {"safe": True},
        mission_has_work=True,
        inbox_has_work=False,
        max_restarts=3,
    )
    assert decision["action"] == "ESCALATE_FAILURE", decision

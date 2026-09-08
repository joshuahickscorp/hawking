"""A stalled objective changes method instead of only demanding a human.

S036 s36/s68: the daemon should be addicted to FRONTIER MOVEMENT, not activity;
when nothing moves, the most productive action may be to abandon the current
method. The stall detector (Mission._on_no_progress) and the frontier scheduler
(frontier_scheduler.decide) both existed and never spoke -- a stall halted and
demanded a person. Now the stall consults the scheduler: if a different frontier
is READY, a replan target is recorded; if everything is parked, it is still a
halt, but an explained one, and the human path is preserved.
"""
from __future__ import annotations

import unittest
from unittest import mock

from hcli.frontier_scheduler import SchedulerDecision
from hcli.scheduler import NO_PROGRESS


class _StallHarness:
    """The minimum of Mission the stall handler touches."""

    def __init__(self):
        self.strategy = ""
        self.phase = ""
        self.no_progress_warning = None
        self.replan_target = None
        self._stop_reason = ""
        self.logs = []
        self.termed = []
        self.checkpointed = False

    def _term(self, msg):
        self.termed.append(msg)

    def _log(self, entry):
        self.logs.append(entry)

    def checkpoint(self):
        self.checkpointed = True

    # the two real methods under test, bound from Mission
    from hcli.mission import Mission
    _on_no_progress = Mission._on_no_progress
    _frontier_replan = Mission._frontier_replan


def _exc():
    return NO_PROGRESS("abc", 3, 3)


class TestStallReplans(unittest.TestCase):
    def test_a_ready_frontier_becomes_a_replan_target(self):
        harness = _StallHarness()
        decision = SchedulerDecision("RUN", "ODYSSEY", "odyssey has ready work",
                                     {}, ())
        with mock.patch("hcli.frontier_scheduler.decide", return_value=decision):
            harness._on_no_progress(_exc())
        self.assertEqual(harness.strategy, "replan_frontier")
        self.assertEqual(harness.replan_target["frontier"], "ODYSSEY")
        self.assertTrue(harness.checkpointed)
        self.assertIn("replan", harness.logs[-1])

    def test_everything_parked_stays_a_halt_and_preserves_the_human_path(self):
        harness = _StallHarness()
        decision = SchedulerDecision("PARK_ALL", None, "all frontiers waiting",
                                     {"ODYSSEY": "wake when GPU free"}, ())
        with mock.patch("hcli.frontier_scheduler.decide", return_value=decision):
            harness._on_no_progress(_exc())
        self.assertEqual(harness.strategy, "halt_no_progress")
        self.assertIsNone(harness.replan_target)
        self.assertNotIn("replan", harness.logs[-1])

    def test_a_scheduler_failure_does_not_turn_a_stall_into_a_crash(self):
        harness = _StallHarness()
        with mock.patch("hcli.frontier_scheduler.decide",
                        side_effect=RuntimeError("probe blew up")):
            harness._on_no_progress(_exc())  # must not raise
        self.assertEqual(harness.strategy, "halt_no_progress")
        self.assertTrue(any(e.get("event") == "frontier_replan_unavailable"
                            for e in harness.logs))

    def test_the_stall_is_always_checkpointed(self):
        # Whatever the scheduler says, the objective is preserved on disk.
        harness = _StallHarness()
        with mock.patch("hcli.frontier_scheduler.decide",
                        return_value=SchedulerDecision("PARK_ALL", None, "x", {}, ())):
            harness._on_no_progress(_exc())
        self.assertTrue(harness.checkpointed)


if __name__ == "__main__":
    unittest.main()

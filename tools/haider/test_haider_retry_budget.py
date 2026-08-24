#!/usr/bin/env python3
"""Deterministic tests for P0: bounded retry budget and mission resilience."""
from __future__ import annotations

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from haider import (
    WorkUnit,
    _transition_workunit_status,
    _workunit_is_ready,
    _identify_ready_workunits,
    _assign_ready_workunits,
    _mark_workunit_completed,
    _mark_workunit_failed,
    _build_scheduler_plan,
)


class TestRetryBudget(unittest.TestCase):
    """Prove that a failed transaction does not terminate the mission,
    and that retry budget is bounded."""

    def test_failed_workunit_can_retry(self):
        """A failed work unit transitions back to ready for retry."""
        wu = WorkUnit("wu-1", "implement", "do work")
        _transition_workunit_status(wu, "ready")
        _transition_workunit_status(wu, "running")
        _mark_workunit_failed(wu)
        self.assertEqual(wu.status, "failed")
        # Retry: failed -> ready
        self.assertTrue(_transition_workunit_status(wu, "ready"))
        self.assertEqual(wu.status, "ready")

    def test_retry_increments_attempts(self):
        """Each assignment increments the attempt counter."""
        wu = WorkUnit("wu-1", "implement", "do work")
        units = {"wu-1": wu}
        # First assignment
        ready = _identify_ready_workunits(units)
        assignments = _assign_ready_workunits(ready, 1)
        self.assertEqual(len(assignments), 1)
        self.assertEqual(wu.attempts, 1)
        # Fail and retry
        _mark_workunit_failed(wu)
        ready = _identify_ready_workunits(units)
        assignments = _assign_ready_workunits(ready, 1)
        self.assertEqual(len(assignments), 1)
        self.assertEqual(wu.attempts, 2)

    def test_bounded_retry_budget_exhausted(self):
        """After max_attempts, a failed work unit is not re-scheduled."""
        max_attempts = 3
        wu = WorkUnit("wu-1", "implement", "do work")
        units = {"wu-1": wu}
        for i in range(max_attempts):
            ready = _identify_ready_workunits(units)
            assignments = _assign_ready_workunits(ready, 1)
            self.assertEqual(len(assignments), 1)
            _mark_workunit_failed(wu)
        # Now wu is failed with attempts == max_attempts
        self.assertEqual(wu.attempts, max_attempts)
        self.assertEqual(wu.status, "failed")
        # _workunit_is_ready should return False for a failed unit
        self.assertFalse(_workunit_is_ready(wu, units))

    def test_independent_work_continues_after_failure(self):
        """When one work unit fails, independent ready work still proceeds."""
        wu_a = WorkUnit("wu-a", "implement", "task A")
        wu_b = WorkUnit("wu-b", "test", "task B")
        units = {"wu-a": wu_a, "wu-b": wu_b}
        # Both are pending with no deps, so both are ready
        ready = _identify_ready_workunits(units)
        self.assertEqual(len(ready), 2)
        # Assign both to 2 runtimes
        assignments = _assign_ready_workunits(ready, 2)
        self.assertEqual(len(assignments), 2)
        # Fail wu-a
        _mark_workunit_failed(wu_a)
        # Complete wu-b
        _mark_workunit_completed(wu_b)
        # wu-b is completed, wu-a is failed
        self.assertEqual(wu_a.status, "failed")
        self.assertEqual(wu_b.status, "completed")
        # A new work unit depending on wu-b should be ready
        wu_c = WorkUnit("wu-c", "verify", "verify B", dependencies=["wu-b"])
        units["wu-c"] = wu_c
        ready = _identify_ready_workunits(units)
        self.assertIn(wu_c, ready)

    def test_completed_work_preserved_after_rollback(self):
        """Completed validated work is not affected by a later rollback."""
        wu_done = WorkUnit("wu-done", "implement", "done work")
        wu_fail = WorkUnit("wu-fail", "implement", "failing work")
        units = {"wu-done": wu_done, "wu-fail": wu_fail}
        ready = _identify_ready_workunits(units)
        assignments = _assign_ready_workunits(ready, 2)
        _mark_workunit_completed(wu_done)
        _mark_workunit_failed(wu_fail)
        # wu-done remains completed
        self.assertEqual(wu_done.status, "completed")
        # wu-fail can be retried
        self.assertTrue(_transition_workunit_status(wu_fail, "ready"))

    def test_no_infinite_loop_on_retry(self):
        """A work unit that keeps failing does not loop forever."""
        wu = WorkUnit("wu-loop", "implement", "looping work")
        units = {"wu-loop": wu}
        max_attempts = 3
        for _ in range(max_attempts):
            ready = _identify_ready_workunits(units)
            assignments = _assign_ready_workunits(ready, 1)
            if not assignments:
                break
            _mark_workunit_failed(wu)
        # After max_attempts failures, no more assignments
        self.assertEqual(wu.attempts, max_attempts)
        ready = _identify_ready_workunits(units)
        assignments = _assign_ready_workunits(ready, 1)
        self.assertEqual(len(assignments), 0)

    def test_scheduler_plan_with_mixed_states(self):
        """Scheduler correctly handles mixed completed/failed/pending units."""
        wu1 = WorkUnit("wu-1", "implement", "step 1")
        wu2 = WorkUnit("wu-2", "test", "step 2", dependencies=["wu-1"])
        wu3 = WorkUnit("wu-3", "verify", "step 3", dependencies=["wu-2"])
        units = {"wu-1": wu1, "wu-2": wu2, "wu-3": wu3}
        # First cycle: only wu-1 is ready
        plan = _build_scheduler_plan(units, 1)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0][0].id, "wu-1")
        _mark_workunit_completed(wu1)
        # Second cycle: wu-2 is now ready
        plan = _build_scheduler_plan(units, 1)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0][0].id, "wu-2")
        _mark_workunit_failed(wu2)
        # wu-2 failed, wu-3 still pending (dep not met)
        plan = _build_scheduler_plan(units, 1)
        self.assertEqual(len(plan), 0)
        # Retry wu-2
        _transition_workunit_status(wu2, "ready")
        plan = _build_scheduler_plan(units, 1)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0][0].id, "wu-2")


if __name__ == "__main__":
    unittest.main()
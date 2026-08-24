#!/usr/bin/env python3
"""Deterministic tests for WorkUnit and scheduler primitives."""

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
)


class TestWorkUnitStatusTransitions(unittest.TestCase):
    def test_valid_transitions(self):
        wu = WorkUnit("w1", "core", "test")
        self.assertTrue(_transition_workunit_status(wu, "ready"))
        self.assertEqual(wu.status, "ready")
        self.assertTrue(_transition_workunit_status(wu, "running"))
        self.assertEqual(wu.status, "running")
        self.assertTrue(_transition_workunit_status(wu, "completed"))
        self.assertEqual(wu.status, "completed")

    def test_invalid_transition_pending_to_running(self):
        wu = WorkUnit("w2", "core", "test")
        self.assertFalse(_transition_workunit_status(wu, "running"))
        self.assertEqual(wu.status, "pending")

    def test_invalid_transition_completed_to_running(self):
        wu = WorkUnit("w3", "core", "test")
        _transition_workunit_status(wu, "ready")
        _transition_workunit_status(wu, "running")
        _transition_workunit_status(wu, "completed")
        self.assertFalse(_transition_workunit_status(wu, "running"))
        self.assertEqual(wu.status, "completed")

    def test_failed_to_ready_retry(self):
        wu = WorkUnit("w4", "core", "test")
        _transition_workunit_status(wu, "ready")
        _transition_workunit_status(wu, "running")
        _transition_workunit_status(wu, "failed")
        self.assertTrue(_transition_workunit_status(wu, "ready"))
        self.assertEqual(wu.status, "ready")


class TestWorkUnitDependencies(unittest.TestCase):
    def test_ready_when_no_deps(self):
        units = {"a": WorkUnit("a", "core", "no deps")}
        self.assertTrue(_workunit_is_ready(units["a"], units))

    def test_not_ready_when_dep_incomplete(self):
        units = {
            "a": WorkUnit("a", "core", "dep"),
            "b": WorkUnit("b", "core", "depends on a", dependencies=["a"]),
        }
        self.assertFalse(_workunit_is_ready(units["b"], units))

    def test_ready_when_dep_completed(self):
        units = {
            "a": WorkUnit("a", "core", "dep"),
            "b": WorkUnit("b", "core", "depends on a", dependencies=["a"]),
        }
        _transition_workunit_status(units["a"], "ready")
        _transition_workunit_status(units["a"], "running")
        _mark_workunit_completed(units["a"])
        self.assertTrue(_workunit_is_ready(units["b"], units))

    def test_not_ready_when_dep_missing(self):
        units = {"b": WorkUnit("b", "core", "depends on x", dependencies=["x"])}
        self.assertFalse(_workunit_is_ready(units["b"], units))


class TestSchedulerAssignment(unittest.TestCase):
    def test_assign_independent_work(self):
        units = {
            "a": WorkUnit("a", "core", "work a"),
            "b": WorkUnit("b", "test", "work b"),
        }
        ready = _identify_ready_workunits(units)
        self.assertEqual(len(ready), 2)
        assignments = _assign_ready_workunits(ready, runtime_count=2)
        self.assertEqual(len(assignments), 2)
        assigned_runtimes = {idx for _, idx in assignments}
        self.assertEqual(assigned_runtimes, {0, 1})

    def test_prevent_duplicate_runtime_assignment(self):
        units = {
            "a": WorkUnit("a", "core", "work a"),
            "b": WorkUnit("b", "test", "work b"),
            "c": WorkUnit("c", "adversary", "work c"),
        }
        ready = _identify_ready_workunits(units)
        assignments = _assign_ready_workunits(ready, runtime_count=2)
        self.assertEqual(len(assignments), 2)
        runtimes = [idx for _, idx in assignments]
        self.assertEqual(len(runtimes), len(set(runtimes)))

    def test_assignment_increments_attempts(self):
        units = {"a": WorkUnit("a", "core", "work a")}
        ready = _identify_ready_workunits(units)
        assignments = _assign_ready_workunits(ready, runtime_count=1)
        self.assertEqual(units["a"].attempts, 1)
        self.assertEqual(units["a"].assigned_runtime, 0)

    def test_mark_completed_clears_runtime(self):
        units = {"a": WorkUnit("a", "core", "work a")}
        ready = _identify_ready_workunits(units)
        _assign_ready_workunits(ready, runtime_count=1)
        self.assertEqual(units["a"].assigned_runtime, 0)
        _mark_workunit_completed(units["a"])
        self.assertIsNone(units["a"].assigned_runtime)
        self.assertEqual(units["a"].status, "completed")

    def test_mark_failed_clears_runtime(self):
        units = {"a": WorkUnit("a", "core", "work a")}
        ready = _identify_ready_workunits(units)
        _assign_ready_workunits(ready, runtime_count=1)
        _mark_workunit_failed(units["a"])
        self.assertIsNone(units["a"].assigned_runtime)
        self.assertEqual(units["a"].status, "failed")

    def test_serial_dependency_ordering(self):
        units = {
            "a": WorkUnit("a", "core", "first"),
            "b": WorkUnit("b", "test", "second", dependencies=["a"]),
        }
        ready = _identify_ready_workunits(units)
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].id, "a")
        assignments = _assign_ready_workunits(ready, runtime_count=2)
        self.assertEqual(len(assignments), 1)
        _mark_workunit_completed(units["a"])
        ready2 = _identify_ready_workunits(units)
        self.assertEqual(len(ready2), 1)
        self.assertEqual(ready2[0].id, "b")


class TestWorkUnitSerialization(unittest.TestCase):
    def test_roundtrip(self):
        wu = WorkUnit("w1", "core", "desc", dependencies=["x"])
        wu.status = "running"
        wu.assigned_runtime = 3
        wu.attempts = 2
        d = wu.to_dict()
        wu2 = WorkUnit.from_dict(d)
        self.assertEqual(wu2.id, wu.id)
        self.assertEqual(wu2.role, wu.role)
        self.assertEqual(wu2.description, wu.description)
        self.assertEqual(wu2.dependencies, wu.dependencies)
        self.assertEqual(wu2.status, wu.status)
        self.assertEqual(wu2.assigned_runtime, wu.assigned_runtime)
        self.assertEqual(wu2.attempts, wu.attempts)


if __name__ == "__main__":
    unittest.main()
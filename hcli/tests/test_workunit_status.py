from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli.workunit import (
    DEFAULT_RETRY_BUDGET,
    WorkUnit,
    identify_ready,
    is_ready,
    transition_status,
)


def _wu(uid: str, **kwargs) -> WorkUnit:
    return WorkUnit(id=uid, role="work", description=uid, **kwargs)


class TestWorkUnitStatusMachine(unittest.TestCase):
    def test_pending_ready_running_completed(self):
        wu = _wu("a")
        self.assertEqual(wu.status, "pending")
        self.assertTrue(transition_status(wu, "ready"))
        self.assertTrue(transition_status(wu, "running"))
        self.assertTrue(transition_status(wu, "completed"))
        self.assertFalse(transition_status(wu, "ready"))
        self.assertFalse(transition_status(wu, "failed"))

    def test_running_to_failed_to_ready(self):
        wu = _wu("a")
        transition_status(wu, "ready")
        transition_status(wu, "running")
        self.assertTrue(transition_status(wu, "failed"))
        self.assertTrue(transition_status(wu, "ready"))

    def test_illegal_transitions_rejected(self):
        wu = _wu("a")
        self.assertFalse(transition_status(wu, "running"))
        self.assertFalse(transition_status(wu, "completed"))
        self.assertFalse(transition_status(wu, "failed"))

    def test_dependency_not_ready_until_dep_completed(self):
        a = _wu("a")
        b = _wu("b", dependencies=["a"])
        units = {"a": a, "b": b}
        self.assertTrue(is_ready(a, units))
        self.assertFalse(is_ready(b, units))
        a.status = "completed"
        self.assertTrue(is_ready(b, units))

    def test_retry_budget_caps_failed_to_ready(self):
        wu = _wu("a", status="failed", attempts=DEFAULT_RETRY_BUDGET)
        self.assertFalse(is_ready(wu, {"a": wu}))
        wu.attempts = DEFAULT_RETRY_BUDGET - 1
        self.assertTrue(is_ready(wu, {"a": wu}))

    def test_identify_ready_includes_already_ready(self):
        a = _wu("a", status="ready")
        b = _wu("b")
        ready = identify_ready({"a": a, "b": b})
        ids = {wu.id for wu in ready}
        self.assertEqual(ids, {"a", "b"})
        self.assertEqual(b.status, "ready")

    def test_repair_unit_ready_when_original_failed(self):
        orig = _wu("orig", status="failed", attempts=1)
        repair = _wu("orig.repair.1", dependencies=["orig"], repairs="orig")
        units = {"orig": orig, repair.id: repair}
        self.assertTrue(is_ready(repair, units))
        self.assertFalse(is_ready(orig, units))

    def test_unknown_resource_class_normalizes_to_light_control(self):
        wu = _wu("a", resource_class="NOT_A_CLASS")
        self.assertEqual(wu.resource_class, "LIGHT_CONTROL")

    def test_roundtrip_dict_preserves_new_fields(self):
        wu = _wu(
            "a",
            resource_class="COMPILE",
            repairs="b",
            failure_context={"error": "x"},
            verifier="pytest -q",
            ready_at=1.5,
            running_at=2.5,
            finished_at=3.5,
        )
        wu.verification = {"ok": True}
        wu.backend_task_id = "task-1"
        restored = WorkUnit.from_dict(wu.to_dict())
        self.assertEqual(restored.resource_class, "COMPILE")
        self.assertEqual(restored.repairs, "b")
        self.assertEqual(restored.failure_context, {"error": "x"})
        self.assertEqual(restored.verifier, "pytest -q")
        self.assertEqual(restored.verification, {"ok": True})
        self.assertEqual(restored.backend_task_id, "task-1")
        self.assertEqual(restored.ready_at, 1.5)
        self.assertEqual(restored.running_at, 2.5)
        self.assertEqual(restored.finished_at, 3.5)

    def test_identify_ready_and_assign_stamp_clocks(self):
        from hcli.workunit import assign_ready

        a = _wu("a")
        units = {"a": a}
        ready = identify_ready(units)
        self.assertEqual(a.status, "ready")
        self.assertIsNotNone(a.ready_at)
        assigned = assign_ready(ready, 1, all_units=units)
        self.assertEqual(len(assigned), 1)
        self.assertEqual(a.status, "running")
        self.assertIsNotNone(a.running_at)
        self.assertGreaterEqual(a.running_at, a.ready_at)


if __name__ == "__main__":
    unittest.main()

import unittest
from tools.haider.haider import WorkUnit


class TestWorkUnit(unittest.TestCase):
    def test_initial_state(self):
        wu = WorkUnit(id="wu-1", role="CORE", description="test")
        self.assertEqual(wu.id, "wu-1")
        self.assertEqual(wu.role, "CORE")
        self.assertEqual(wu.status, "pending")
        self.assertEqual(wu.attempts, 0)
        self.assertIsNone(wu.assigned_runtime)
        self.assertEqual(wu.dependencies, [])

    def test_valid_transitions(self):
        wu = WorkUnit(id="wu-2", role="TEST", description="test")
        wu.transition("running", assigned_runtime=0)
        self.assertEqual(wu.status, "running")
        self.assertEqual(wu.assigned_runtime, 0)
        self.assertEqual(wu.attempts, 1)

        wu.transition("completed")
        self.assertEqual(wu.status, "completed")

    def test_invalid_transition(self):
        wu = WorkUnit(id="wu-3", role="ADVERSARY", description="test")
        with self.assertRaises(ValueError):
            wu.transition("completed")

    def test_failed_retry(self):
        wu = WorkUnit(id="wu-4", role="CORE", description="test")
        wu.transition("running", assigned_runtime=1)
        wu.transition("failed")
        self.assertEqual(wu.status, "failed")
        self.assertEqual(wu.attempts, 1)

        wu.transition("pending")
        self.assertEqual(wu.status, "pending")
        self.assertEqual(wu.attempts, 1)


if __name__ == "__main__":
    unittest.main()
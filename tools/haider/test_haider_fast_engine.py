import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.abspath(__file__)
    ),
)

import haider


class FastMissionStateTests(unittest.TestCase):
    def base(self):
        return {
            "mission_id": "x",
            "task": "DO THE MISSION",
            "source": "test",
            "status": "running",
            "cycle": 0,
            "max_cycles": 2,
        }

    def test_success_increments(self):
        state = haider._fast_state_transition(
            self.base(),
            mutation_ok=True,
            validation_ok=True,
        )

        self.assertEqual(
            state["cycle"],
            1,
        )

        self.assertEqual(
            state["status"],
            "running",
        )

    def test_mutation_failure_no_increment(self):
        state = haider._fast_state_transition(
            self.base(),
            mutation_ok=False,
            validation_ok=False,
        )

        self.assertEqual(
            state["cycle"],
            0,
        )

        self.assertEqual(
            state["status"],
            "failed",
        )

    def test_validation_failure_no_increment(self):
        state = haider._fast_state_transition(
            self.base(),
            mutation_ok=True,
            validation_ok=False,
        )

        self.assertEqual(
            state["cycle"],
            0,
        )

        self.assertEqual(
            state["status"],
            "failed",
        )

    def test_max_cycles(self):
        state = self.base()
        state["cycle"] = 1

        state = haider._fast_state_transition(
            state,
            mutation_ok=True,
            validation_ok=True,
        )

        self.assertEqual(
            state["cycle"],
            2,
        )

        self.assertEqual(
            state["status"],
            "max_cycles",
        )

    def test_complete(self):
        state = haider._fast_state_transition(
            self.base(),
            mutation_ok=True,
            validation_ok=True,
            complete=True,
        )

        self.assertEqual(
            state["status"],
            "complete",
        )

    def test_task_immutable(self):
        state = self.base()

        result = haider._fast_state_transition(
            state,
            mutation_ok=True,
            validation_ok=True,
        )

        self.assertEqual(
            result["task"],
            "DO THE MISSION",
        )

    def test_runtime_pool_rejects_zero(self):
        with self.assertRaises(ValueError):
            haider.FastRuntimePool(
                0,
                "model",
                "127.0.0.1",
                4096,
            )


if __name__ == "__main__":
    unittest.main()

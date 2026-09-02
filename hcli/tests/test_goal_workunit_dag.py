from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli.goal import (
    GoalCompiler,
    InvalidTransitionError,
    WorkUnitDAG,
)
from hcli.workunit import identify_ready, transition_status


class TestWorkUnitDAGWrapper(unittest.TestCase):
    def test_get_ready_units_routes_through_identify_ready(self):
        dag = WorkUnitDAG()
        dag.add_unit("a", "root")
        dag.add_unit("b", "child", ["a"])
        ready = dag.get_ready_units()
        self.assertEqual({wu.id for wu in ready}, {"a"})
        self.assertEqual(dag.units["a"].status, "ready")
        self.assertEqual(dag.units["b"].status, "pending")

    def test_mark_completed_uses_transition_status_and_refuses_repeat(self):
        dag = WorkUnitDAG()
        dag.add_unit("a", "root")
        dag.get_ready_units()
        self.assertEqual(dag.units["a"].status, "ready")
        dag.mark_completed("a")
        self.assertEqual(dag.units["a"].status, "completed")
        with self.assertRaises(InvalidTransitionError):
            dag.mark_completed("a")
        wu = dag.units["a"]
        self.assertFalse(transition_status(wu, "completed"))

    def test_mark_failed_walks_legal_path(self):
        dag = WorkUnitDAG()
        dag.add_unit("a", "root")
        dag.mark_failed("a")
        self.assertEqual(dag.units["a"].status, "failed")
        with self.assertRaises(InvalidTransitionError):
            dag.mark_failed("a")

    def test_repair_aware_readiness(self):
        dag = WorkUnitDAG()
        dag.add_unit("orig", "original")
        dag.units["orig"].status = "failed"
        dag.units["orig"].attempts = 1
        dag.add_unit("orig.repair.1", "repair", ["orig"])
        dag.units["orig.repair.1"].repairs = "orig"
        ready = {wu.id for wu in dag.get_ready_units()}
        self.assertEqual(ready, {"orig.repair.1"})
        self.assertEqual(dag.units["orig"].status, "failed")

    def test_compiler_ir_still_implement_then_validate(self):
        compiled = GoalCompiler().compile(
            "implement feature in foo.py. tests must pass."
        )
        dag = compiled["workunits"]
        self.assertEqual({wu.id for wu in dag.get_ready_units()}, {"implement"})
        dag.mark_completed("implement")
        self.assertEqual({wu.id for wu in dag.get_ready_units()}, {"validate"})

    def test_round_trip_dict_and_workspace(self):
        dag = WorkUnitDAG()
        dag.add_unit("a", "root")
        dag.add_unit("b", "child", ["a"])
        dag.get_ready_units()
        restored = WorkUnitDAG.from_dict(dag.to_dict())
        self.assertEqual(restored.to_dict(), dag.to_dict())
        with tempfile.TemporaryDirectory() as tmp:
            dag.save(tmp)
            loaded = WorkUnitDAG.from_workspace(tmp, recover_running=False)
            self.assertEqual(loaded.to_dict(), dag.to_dict())
            self.assertEqual(
                {wu.id for wu in identify_ready(loaded.units)},
                {wu.id for wu in loaded.get_ready_units()},
            )


if __name__ == "__main__":
    unittest.main()

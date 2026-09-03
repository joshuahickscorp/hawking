import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hcli.goal import GoalCompiler, WorkUnitDAG


class TestGoalCompiler(unittest.TestCase):
    def test_compile_extracts_invariants_and_workunits(self):
        goal_text = """Fix the parser. Do not touch persistence. Tests must pass.

Files: src/parser.py, tests/test_parser.py"""
        compiler = GoalCompiler()
        result = compiler.compile(goal_text)

        self.assertIn("invariants", result)
        self.assertIn("workunits", result)
        self.assertIn("acceptance_criteria", result)

        # Invariants should capture constraints
        invariants = result["invariants"]
        self.assertTrue(any("persistence" in inv.lower() for inv in invariants))

        # WorkUnits should be a DAG
        dag = result["workunits"]
        self.assertIsInstance(dag, WorkUnitDAG)
        self.assertGreater(len(dag.units), 0)

        # Acceptance criteria should reference tests
        criteria = result["acceptance_criteria"]
        self.assertTrue(any("test" in c.lower() for c in criteria))

    def test_focused_worker_context_excludes_full_goal(self):
        goal_text = "A very long goal description that should not be sent in full to every worker. " * 10
        compiler = GoalCompiler()
        result = compiler.compile(goal_text)
        dag = result["workunits"]

        # Get first workunit and build focused context
        first_wu = list(dag.units.values())[0]
        focused = compiler.build_focused_context(first_wu, result)

        # Focused context should be smaller than full goal
        self.assertLess(len(focused), len(goal_text))
        # Should contain the workunit description
        self.assertIn(first_wu.description, focused)

    def test_workunit_dag_dependency_readiness(self):
        dag = WorkUnitDAG()
        dag.add_unit("a", "task a", [])
        dag.add_unit("b", "task b", ["a"])
        dag.add_unit("c", "task c", ["a"])

        # Initially only 'a' is ready
        ready = dag.get_ready_units()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].id, "a")

        # After completing 'a', both 'b' and 'c' become ready
        dag.mark_completed("a")
        ready = dag.get_ready_units()
        self.assertEqual(len(ready), 2)
        self.assertIn("b", [u.id for u in ready])
        self.assertIn("c", [u.id for u in ready])


if __name__ == "__main__":
    unittest.main()
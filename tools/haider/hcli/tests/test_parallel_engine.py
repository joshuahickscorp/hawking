from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


from hcli.engine import Engine, EngineError
from hcli.events import EventBus
from hcli.workspace import Workspace


WRONG_ADD = "def add(a, b):\n    return a * b - 999\n"
RIGHT_ADD = "def add(a, b):\n    return a + b\n"
TEST_ADD = (
    "from calc import add\n\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
)


class TestParallelEngine(unittest.TestCase):
    """Real engine behaviour. Vacuous MagicMock-echo tests were deleted.

    Deleted (could not fail against real engine behaviour):
      - test_fixed_hcli_3_keeps_three_physical_descriptors
        Constructor echo of runtime_count plus a MagicMock the test built.
      - test_three_independent_ready_workunits_dispatch_concurrently
        Engine.execute makes one serial _call_model; the 0.35s bound cannot
        fail even if dispatch is deleted.
      - test_runtime_indices_differ / test_runtime_ports_differ
        Asserted fixture values the test itself wrote onto a MagicMock.
      - test_runtime_pool_remains_warm
        Asserted the same MagicMock fields, never touched by execute().
      - test_shutdown_leaves_zero_owned_children
        Called pool.stop() whose side_effect the test defined, then
        asserted those fields were zero.

    Rewritten:
      - test_scheduler_honors_dependencies — inspects the compiled DAG.
      - test_workers_cannot_race_overlapping_writers — two ops, disk order.
      - test_workers_return_proposals_rather_than_directly_mutating_disk
      - test_coordinator_combines_compatible_results — answer passthrough.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.workspace = Workspace(str(self.root))
        self.bus = EventBus()
        self.engine = Engine(
            workspace=self.workspace,
            event_bus=self.bus,
            runtime_count=3,
            model_name="/m.gguf",
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_scheduler_honors_dependencies(self):
        compiled = self.engine.goal_compiler.compile(
            "implement feature. tests must pass."
        )
        dag = compiled["workunits"]
        self.assertIn("implement", dag.units)
        self.assertIn("validate", dag.units)
        self.assertEqual(dag.units["validate"].dependencies, ["implement"])
        ready = dag.get_ready_units()
        ready_ids = {wu.id for wu in ready}
        self.assertEqual(ready_ids, {"implement"})
        dag.mark_completed("implement")
        ready_after = {wu.id for wu in dag.get_ready_units()}
        self.assertEqual(ready_after, {"validate"})

    def test_workers_cannot_race_overlapping_writers(self):
        target = self.engine.root / "a.py"
        target.write_text("x = 0\n", encoding="utf-8")
        self.engine._call_model = lambda p, e, c: {
            "kind": "mutation",
            "content": "two ordered writes",
            "operations": [
                {
                    "op": "replace",
                    "path": "a.py",
                    "old_text": "x = 0",
                    "new_text": "x = 1",
                },
                {
                    "op": "replace",
                    "path": "a.py",
                    "old_text": "x = 1",
                    "new_text": "x = 2",
                },
            ],
            "tests": [],
        }
        result = self.engine.execute("modify a.py twice")
        self.assertEqual(result["status"], "unverified")
        self.assertFalse(result.get("rolled_back"))
        self.assertEqual(target.read_text(encoding="utf-8"), "x = 2\n")

    def test_workers_return_proposals_rather_than_directly_mutating_disk(self):
        seen = {"during_call": None}

        def mock_call(prompt, evidence, compiled):
            seen["during_call"] = (self.engine.root / "new.py").exists()
            return {
                "kind": "mutation",
                "content": "create new.py",
                "operations": [
                    {
                        "op": "create",
                        "path": "new.py",
                        "new_text": "print(1)\n",
                    }
                ],
                "tests": [],
            }

        self.engine._call_model = mock_call
        result = self.engine.execute("create new.py")
        self.assertFalse(seen["during_call"])
        self.assertEqual(result["status"], "unverified")
        created = self.engine.root / "new.py"
        self.assertTrue(created.is_file())
        self.assertEqual(created.read_text(encoding="utf-8"), "print(1)\n")

    def test_coordinator_combines_compatible_results(self):
        self.engine._call_model = lambda p, e, c: {
            "kind": "answer",
            "content": "combined",
            "operations": [],
            "tests": [],
        }
        result = self.engine.execute("combine")
        self.assertEqual(result["kind"], "answer")
        self.assertEqual(result["content"], "combined")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result.get("operations"), [])

    def test_empty_tests_status_is_unverified_not_completed(self):
        (self.engine.root / "a.py").write_text("x = 1\n", encoding="utf-8")
        events: list[str] = []
        self.engine.event_bus.subscribe(lambda ev: events.append(ev.type))
        self.engine._call_model = lambda p, e, c: {
            "kind": "mutation",
            "content": "change x",
            "operations": [
                {
                    "op": "replace",
                    "path": "a.py",
                    "old_text": "x = 1",
                    "new_text": "x = 2",
                }
            ],
            "tests": [],
        }
        result = self.engine.execute("change a.py")
        self.assertEqual(result["status"], "unverified")
        self.assertNotIn("validation_passed", events)
        self.assertIn("validation_recorded", events)
        self.assertTrue(
            (self.engine.root / "a.py")
            .read_text(encoding="utf-8")
            .startswith("x = 2")
        )

    def test_create_missing_new_text_raises_and_writes_nothing(self):
        with self.assertRaises(EngineError) as ctx:
            self.engine._apply_operations(
                [{"op": "create", "path": "new.py", "content": "print(1)"}]
            )
        self.assertIn("new_text", str(ctx.exception))
        self.assertFalse((self.engine.root / "new.py").exists())

    def test_verified_repair_is_completed_and_red_before_green(self):
        (self.engine.root / "calc.py").write_text(WRONG_ADD, encoding="utf-8")
        (self.engine.root / "test_calc.py").write_text(TEST_ADD, encoding="utf-8")
        self.engine._call_model = lambda p, e, c: {
            "kind": "mutation",
            "content": "fix add",
            "operations": [
                {
                    "op": "replace",
                    "path": "calc.py",
                    "old_text": "return a * b - 999",
                    "new_text": "return a + b",
                }
            ],
            "tests": ["test_calc.py"],
        }
        result = self.engine.execute("fix add")
        self.assertEqual(result["status"], "completed")
        receipt = json.loads(Path(result["receipt"]).read_text(encoding="utf-8"))
        validation = receipt["validation"]
        self.assertTrue(validation.get("ok"))
        self.assertIs(validation.get("red_before_green"), True)
        self.assertTrue(validation.get("red_before_green_advisory"))
        self.assertEqual(
            (self.engine.root / "calc.py").read_text(encoding="utf-8"),
            RIGHT_ADD,
        )


if __name__ == "__main__":
    unittest.main()

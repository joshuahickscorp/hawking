from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli.engine import Engine, EngineError, NoOpMutation
from hcli.events import EventBus
from hcli.mutation import MutationError, _apply_insert, _apply_replace
from hcli.workspace import Workspace

WRONG_ADD = "def add(a, b):\n    return a * b - 999\n"
RIGHT_ADD = "def add(a, b):\n    return a + b\n"
TEST_ADD = (
    "from calc import add\n\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
)


class TestAcceptanceIntegrity(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.workspace = Workspace(str(self.root))
        self.bus = EventBus()
        self.engine = Engine(
            workspace=self.workspace,
            event_bus=self.bus,
            runtime_count=1,
            model_name="/m.gguf",
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write(self, name: str, text: str) -> Path:
        path = self.engine.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_single_anchor_still_required(self):
        self._write("a.py", "x = 1\nx = 1\n")
        with self.assertRaises(EngineError) as ctx:
            self.engine._apply_operations(
                [
                    {
                        "op": "replace",
                        "path": "a.py",
                        "old_text": "x = 1",
                        "new_text": "x = 2",
                    }
                ]
            )
        self.assertIn("exactly once", str(ctx.exception))
        self.assertNotIn("NO_OP_MUTATION", str(ctx.exception))

    def test_single_anchor_zero_occurrences(self):
        self._write("a.py", "x = 1\n")
        with self.assertRaises(EngineError) as ctx:
            self.engine._apply_operations(
                [
                    {
                        "op": "replace",
                        "path": "a.py",
                        "old_text": "y = 2",
                        "new_text": "y = 3",
                    }
                ]
            )
        self.assertIn("exactly once", str(ctx.exception))

    def test_safe_test_argv_admits_pytest_forms(self):
        self._write("calc.py", RIGHT_ADD)
        self._write("test_calc.py", TEST_ADD)
        for raw in (
            "test_calc.py",
            "pytest test_calc.py",
            "python -m pytest test_calc.py",
            "python3 -m pytest test_calc.py",
            "pytest -q test_calc.py",
        ):
            argv = self.engine._safe_test_argv(raw)
            self.assertIsNotNone(argv, raw)
            self.assertIn("-m", argv, raw)
            self.assertIn("pytest", argv, raw)

    def test_safe_test_argv_refuses_shell(self):
        self._write("a.py", "x = 1\n")
        self.assertIsNone(self.engine._safe_test_argv("make test"))
        self.assertIsNone(self.engine._safe_test_argv("pytest -q"))
        self.assertIsNone(self.engine._safe_test_argv("bash -c 'true'"))

    def test_zero_collected_is_no_evidence(self):
        self._write("empty.py", "x = 1\n")
        validation = self.engine._validate(
            [self.engine.root / "empty.py"],
            ["python -m pytest empty.py"],
        )
        self.assertFalse(validation.get("ok"))
        test_checks = [
            c for c in validation.get("checks", []) if c.get("kind") == "test"
        ]
        self.assertTrue(test_checks)
        self.assertEqual(test_checks[0].get("reason"), "NO_EVIDENCE")
        self.assertEqual(test_checks[0].get("collected"), 0)

    def test_apply_records_sha256_and_real_change_is_not_noop(self):
        self._write("a.py", "x = 1\n")
        result = self.engine._apply_operations(
            [
                {
                    "op": "replace",
                    "path": "a.py",
                    "old_text": "x = 1",
                    "new_text": "x = 2",
                }
            ]
        )
        self.assertTrue(result.get("changed"))
        files = result.get("files") or []
        self.assertEqual(len(files), 1)
        self.assertNotEqual(files[0]["sha256_before"], files[0]["sha256_after"])
        self.assertTrue(files[0]["changed"])

    def test_python_comment_only_change_is_a_noop_and_is_restored(self):
        self._write("a.py", "x = 1\n")
        with self.assertRaises(NoOpMutation):
            self.engine._apply_operations(
                [
                    {
                        "op": "append",
                        "path": "a.py",
                        "new_text": "# model marker\n",
                    }
                ]
            )
        self.assertEqual((self.engine.root / "a.py").read_text(), "x = 1\n")

    def test_append_empty_is_noop(self):
        self._write("a.py", "x = 1\n")
        with self.assertRaises(NoOpMutation):
            self.engine._apply_operations(
                [
                    {
                        "op": "append",
                        "path": "a.py",
                        "new_text": "",
                    }
                ]
            )

    def test_replace_file_identical_is_noop(self):
        self._write("a.py", "x = 1\n")
        with self.assertRaises(NoOpMutation):
            self.engine._apply_operations(
                [
                    {
                        "op": "replace_file",
                        "path": "a.py",
                        "new_text": "x = 1\n",
                    }
                ]
            )

    def test_empty_tests_receipt_records_no_evidence_but_keeps_mutation(self):
        self._write("a.py", "x = 1\n")
        self.engine._call_model = lambda prompt, evidence, compiled, **kwargs: {
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
        self.assertTrue(
            (self.engine.root / "a.py").read_text(encoding="utf-8").startswith("x = 2")
        )
        receipt = json.loads(Path(result["receipt"]).read_text(encoding="utf-8"))
        self.assertFalse(receipt["validation"]["ok"])
        self.assertEqual(receipt["validation"]["reason"], "NO_EVIDENCE")
        files = receipt["validation"].get("files") or []
        self.assertTrue(files)
        self.assertNotEqual(files[0]["sha256_before"], files[0]["sha256_after"])

    def test_mutation_replace_rejects_identical_strings(self):
        with self.assertRaises(MutationError) as ctx:
            _apply_replace("x = 1\n", "x = 1", "x = 1")
        self.assertIn("NO_OP_MUTATION", str(ctx.exception))

    def test_mutation_insert_rejects_empty_text(self):
        with self.assertRaises(MutationError) as ctx:
            _apply_insert("x = 1\n", "x = 1", "", "insert_after")
        self.assertIn("NO_OP_MUTATION", str(ctx.exception))

    def test_snapshot_restore_still_rolls_back_failed_test(self):
        self._write("calc.py", WRONG_ADD)
        self._write("test_calc.py", TEST_ADD)
        self.engine._call_model = lambda prompt, evidence, compiled, **kwargs: {
            "kind": "mutation",
            "content": "wrong fix",
            "operations": [
                {
                    "op": "replace",
                    "path": "calc.py",
                    "old_text": "return a * b - 999",
                    "new_text": "return a * b - 998",
                }
            ],
            "tests": ["test_calc.py"],
        }
        result = self.engine.execute("do not actually fix add")
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result.get("rolled_back"))
        self.assertEqual(
            (self.engine.root / "calc.py").read_text(encoding="utf-8"),
            WRONG_ADD,
        )


if __name__ == "__main__":
    unittest.main()

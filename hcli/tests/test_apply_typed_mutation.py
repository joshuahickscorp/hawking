"""The hands: one mutation transaction, reusable by any HCLI surface.

`Engine.execute()` already owned snapshot -> apply -> prove -> validate ->
accept-or-roll-back, but only as part of its own model loop. General HCLI (the
browser surface) has its own loop and its own tool dialect; what it lacked was
hands. `apply_typed_mutation` is that seam, and these pin the properties a
builder must never lose:

  * a rejected mutation leaves the file EXACTLY as it was
  * PROPOSED, APPLIED, VALIDATED and ACCEPTED stay distinguishable
  * an edit with no test is `unproven`, not `accepted`
  * a fragment that would not compile in its resulting file is refused BEFORE
    anything is written
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hcli.engine import Engine
from hcli.workspace import Workspace


class _Pool:
    model_path = "sealed-3.14"
    topology = "process"
    requested_n = 1
    admitted_n = 1
    repo_root = "."


def _engine(root: Path) -> Engine:
    return Engine(Workspace(str(root)), runtime_provider=lambda: _Pool())


class TestTheTransaction(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.engine = _engine(self.root)

    def test_a_create_with_no_test_is_unproven_not_accepted(self):
        got = self.engine.apply_typed_mutation([{
            "op": "create", "path": "made.py",
            "new_lines": ["VALUE = 1"]}])
        self.assertEqual(got["status"], "unproven",
                         "an edit with no test was reported as accepted")
        self.assertTrue(got["applied"])
        self.assertFalse(got["rolled_back"])
        self.assertEqual((self.root / "made.py").read_text().strip(), "VALUE = 1")

    def test_the_diff_shows_what_changed(self):
        got = self.engine.apply_typed_mutation([{
            "op": "create", "path": "made.py", "new_lines": ["VALUE = 1"]}])
        self.assertIn("VALUE = 1", got["diff"])
        self.assertIn("made.py", got["diff"])

    def test_no_operations_is_refused_without_touching_disk(self):
        got = self.engine.apply_typed_mutation([])
        self.assertEqual(got["status"], "rejected")
        self.assertFalse(got["applied"])

    def test_a_fragment_that_would_not_compile_is_refused_before_writing(self):
        target = self.root / "mod.py"
        target.write_text("def f():\n    return 0\n", encoding="utf-8")
        got = self.engine.apply_typed_mutation([{
            "op": "replace", "path": "mod.py",
            "old_lines": ["def f():", "    return 0"],
            "new_lines": ["def f():", "    return len(("]}])
        self.assertEqual(got["status"], "rejected")
        self.assertFalse(got["applied"], "invalid source was written to disk")
        self.assertEqual(target.read_text(), "def f():\n    return 0\n")

    def test_a_replace_whose_anchor_does_not_match_is_refused(self):
        target = self.root / "mod.py"
        target.write_text("def f():\n    return 0\n", encoding="utf-8")
        got = self.engine.apply_typed_mutation([{
            "op": "replace", "path": "mod.py",
            "old_lines": ["def nope():"], "new_lines": ["x = 1"]}])
        self.assertEqual(got["status"], "rejected")
        self.assertEqual(target.read_text(), "def f():\n    return 0\n",
                         "the file changed despite a rejected mutation")

    def test_a_failing_test_rolls_the_mutation_back(self):
        # The property that makes a builder safe: the repository is never left
        # half-mutated because a model turn ended badly.
        target = self.root / "mod.py"
        target.write_text("VALUE = 1\n", encoding="utf-8")
        proving = self.root / "test_mod.py"
        proving.write_text(
            "from mod import VALUE\n\n\ndef test_value():\n    assert VALUE == 1\n",
            encoding="utf-8")
        got = self.engine.apply_typed_mutation(
            [{"op": "replace", "path": "mod.py",
              "old_lines": ["VALUE = 1"], "new_lines": ["VALUE = 999"]}],
            tests=["test_mod.py"])
        self.assertEqual(got["status"], "rejected", got.get("reason"))
        self.assertTrue(got["rolled_back"])
        self.assertEqual(target.read_text(), "VALUE = 1\n",
                         "a failing test did not restore the original file")

    def test_a_passing_test_accepts_and_keeps_the_change(self):
        target = self.root / "mod.py"
        target.write_text("VALUE = 1\n", encoding="utf-8")
        proving = self.root / "test_mod.py"
        proving.write_text(
            "from mod import VALUE\n\n\ndef test_value():\n    assert VALUE == 2\n",
            encoding="utf-8")
        got = self.engine.apply_typed_mutation(
            [{"op": "replace", "path": "mod.py",
              "old_lines": ["VALUE = 1"], "new_lines": ["VALUE = 2"]}],
            tests=["test_mod.py"])
        self.assertIn(got["status"], ("accepted", "unproven"), got.get("reason"))
        self.assertEqual(target.read_text(), "VALUE = 2\n")
        self.assertFalse(got["rolled_back"])

    def test_the_verdict_never_says_accepted_when_it_rolled_back(self):
        target = self.root / "mod.py"
        target.write_text("VALUE = 1\n", encoding="utf-8")
        proving = self.root / "test_mod.py"
        proving.write_text(
            "from mod import VALUE\n\n\ndef test_value():\n    assert VALUE == 1\n",
            encoding="utf-8")
        got = self.engine.apply_typed_mutation(
            [{"op": "replace", "path": "mod.py",
              "old_lines": ["VALUE = 1"], "new_lines": ["VALUE = 999"]}],
            tests=["test_mod.py"])
        self.assertNotEqual(got["status"], "accepted")
        self.assertTrue(got["rolled_back"] and got["applied"])


if __name__ == "__main__":
    unittest.main()

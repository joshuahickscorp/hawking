"""DEBUG owns reproduce-and-localize, so the model reasons over a fact.

S036 s21: the loop is REPRODUCE -> LOCALIZE -> ... never "see error, guess fix".
A model shown a raw traceback reacts to whatever is salient; a model shown "this
assertion, this file:line, this function, reproduced on this command" spends
inference on the fix. These pin that the localizer finds the PROJECT frame (not
stdlib), reproduces only a real failure, and hands back a summary, not a dump.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hcli.capabilities import Failure, diagnose

PYTEST_FAIL = '''============================= test session starts ==============================
collected 1 item

test_adder.py F                                                          [100%]

=================================== FAILURES ===================================
_________________________________ test_add _____________________________________
test_adder.py:5: in test_add
    assert add(2, 3) == 5
E   assert -1 == 5
=========================== short test summary info ============================
FAILED test_adder.py::test_add - assert -1 == 5
============================== 1 failed in 0.02s ===============================
'''

TRACEBACK_THROUGH_STDLIB = '''Traceback (most recent call last):
  File "/opt/homebrew/lib/python3.12/json/__init__.py", line 346, in loads
    return _default_decoder.decode(s)
  File "myproj/parser.py", line 88, in parse
    return json.loads(text)
  File "/opt/homebrew/lib/python3.12/json/decoder.py", line 337, in decode
    obj, end = self.raw_decode(s)
E   json.decoder.JSONDecodeError: Expecting value: line 1 column 1
'''


class TestLocalization(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        (self.root / "test_adder.py").write_text(
            "from adder import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
            encoding="utf-8")

    def test_a_pytest_failure_localizes_to_the_project_file(self):
        f = diagnose(PYTEST_FAIL, command="pytest test_adder.py",
                     exit_code=1, root=self.root)
        self.assertTrue(f.reproduced)
        self.assertEqual(Path(f.file).name, "test_adder.py")
        self.assertEqual(f.line, 5)
        self.assertEqual(f.function, "test_add")

    def test_the_assertion_is_extracted(self):
        f = diagnose(PYTEST_FAIL, command="pytest", exit_code=1, root=self.root)
        self.assertIn("-1 == 5", f.message)

    def test_a_stdlib_frame_is_skipped_for_the_project_frame(self):
        # The deepest PROJECT frame is myproj/parser.py:88, not the json module.
        (self.root / "myproj").mkdir()
        (self.root / "myproj" / "parser.py").write_text(
            "import json\n\n\ndef parse(text):\n" + "\n" * 82 +
            "    return json.loads(text)\n", encoding="utf-8")
        f = diagnose(TRACEBACK_THROUGH_STDLIB, command="python -m myproj",
                     exit_code=1, root=self.root)
        self.assertEqual(Path(f.file).name, "parser.py",
                         "localized to the stdlib instead of the project")
        self.assertEqual(f.line, 88)

    def test_a_clean_run_is_not_a_failure(self):
        f = diagnose("2 passed in 0.01s", command="pytest",
                     exit_code=0, root=self.root)
        self.assertFalse(f.reproduced)
        self.assertIn("cleanly", f.note)

    def test_a_reproduced_failure_with_no_project_frame_says_so(self):
        f = diagnose("E   RuntimeError: something in a dependency",
                     command="pytest", exit_code=1, root=self.root)
        self.assertTrue(f.reproduced)
        self.assertIn("dependency or the harness", f.note)

    def test_the_summary_is_actionable_not_a_dump(self):
        f = diagnose(PYTEST_FAIL, command="pytest test_adder.py",
                     exit_code=1, root=self.root)
        summary = f.summary()
        self.assertIn("REPRODUCED", summary)
        self.assertIn("test_add()", summary)
        self.assertIn(">>", summary, "the failing line is not marked")
        self.assertLess(len(summary), 600)

    def test_a_large_output_goes_to_a_handle_not_context(self):
        class _Cache:
            class _Ref:
                id = "paste_x"
            def store(self, text):
                return self._Ref()
        big = PYTEST_FAIL + "x" * 3000
        f = diagnose(big, command="pytest", exit_code=1,
                     root=self.root, cache=_Cache())
        self.assertEqual(f.handle, "paste_x")
        self.assertIn("paste_x", f.summary())


if __name__ == "__main__":
    unittest.main()

"""A dropped bracket must be a retry, not a lost unit.

Measured on a live goal: the model produced a real mutation, patched the source
AND the test, and named a test. The 2-line source patch was CORRECT and
compiled. The 15-line test rewrite in the same reply dropped three closing
parens:

    self.assertTrue(w.get("truncated")
    self.assertIsNone(w.get("start_line")
    def test_a_whole_file_read_reports_total_lines(self:

py_compile failed AFTER the mutation was applied, so the whole unit -- including
the correct half -- was rolled back and lost, and the model never saw the error.

This is the one weakness measured that is actually about the model rather than
the harness: short patches correct, long verbatim code inside a JSON string
unreliable. Handing it its own SyntaxError, with the line, on the next attempt
is the cheapest thing that makes the failure fixable by the thing that caused it.
"""
from __future__ import annotations

import json
import unittest

from hcli.engine import _python_syntax_violation


def reply(path: str, body: str) -> str:
    return json.dumps({
        "kind": "mutation", "content": "x", "tests": [], "tool_calls": [],
        "operations": [{"op": "replace", "path": path, "new_text": body}],
    })


class TestSyntaxPreflight(unittest.TestCase):
    def test_the_live_failure_is_caught(self):
        broken = 'def test_x(self:\n    self.assertTrue(w.get("truncated")\n'
        got = _python_syntax_violation(reply("hcli/tests/test_x.py", broken))
        self.assertIsNotNone(got)
        self.assertIn("hcli/tests/test_x.py", got)
        self.assertIn("line", got)

    def test_valid_python_passes(self):
        """Negative control: correct code must not be refused."""
        self.assertIsNone(
            _python_syntax_violation(reply("a.py", "def f():\n    return 1\n"))
        )

    def test_non_python_is_not_judged_here(self):
        """A markdown file is the verifier's business, not the parser's."""
        self.assertIsNone(_python_syntax_violation(reply("d.md", "# ((( not python")))

    def test_a_non_json_reply_is_left_to_the_contract(self):
        self.assertIsNone(_python_syntax_violation("not json at all"))

    def test_an_empty_operation_is_not_a_syntax_error(self):
        self.assertIsNone(_python_syntax_violation(reply("a.py", "   ")))

    def test_the_message_tells_the_model_what_to_do(self):
        got = _python_syntax_violation(reply("a.py", "def f(:\n"))
        self.assertIn("rewrite that operation only", got)

    def test_it_is_wired_into_the_contract_retry(self):
        """At the call site, not just the helper.

        A helper that nothing calls is the failure mode this campaign hit four
        times, so the wiring is asserted rather than assumed.
        """
        import inspect

        from hcli.engine import Engine

        src = inspect.getsource(Engine._complete_with_schema_contract)
        self.assertIn("_python_syntax_violation(content)", src)
        self.assertIn("SchemaViolation(syntax", src)


if __name__ == "__main__":
    unittest.main()

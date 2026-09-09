"""A refusal the body cannot act on is churn, not a refusal.

repo.edit answers a bad `operations` with the same generic shape string every
time: `repo.edit needs {"operations": [...]}`. That message is correct when the
argument is genuinely the wrong TYPE. It is actively misleading when the body
sent a JSON array that failed to PARSE -- which is what a greedy body emitting a
whole file inline actually does, since a single unterminated string in the
generated content invalidates the entire payload.

Measured live: cycles 69-72 each carried a repo.edit call and each got the same
shape string back. The body had no way to distinguish "you sent the wrong kind
of thing" from "your JSON broke at character 812", so it re-sent the same broken
payload four times. The loop guard counts that as churn; the body experienced it
as an unexplained wall.

The fix is not to accept bad JSON. It is to say which failure it was.
"""
from __future__ import annotations

import unittest

from hcli.chat_tools import run_builder_tool


class _Engine:
    def apply_typed_mutation(self, operations, tests):
        raise AssertionError("must not reach the engine with an unparsed payload")


class TestARefusedBuilderCallSaysWhy(unittest.TestCase):
    def test_malformed_json_names_the_parse_error(self):
        broken = '[{"op": "create", "path": "x.py", "new_lines": ["unterminated]'
        got = run_builder_tool("repo.edit", {"operations": broken}, engine=_Engine())
        self.assertFalse(got.ok)
        self.assertIn("JSON", got.error.upper(),
                      "a parse failure must be named as one, not reported as a shape error")

    def test_the_parse_error_locates_the_problem(self):
        broken = '[{"op": "create", "path": "x.py", "new_lines": ["unterminated]'
        got = run_builder_tool("repo.edit", {"operations": broken}, engine=_Engine())
        self.assertRegex(got.error, r"(char|column|line)\s*\d+",
                         "the body needs a position, not just 'invalid'")

    def test_a_genuine_type_error_still_gets_the_shape(self):
        """Negative control: the existing message must survive for real type errors."""
        got = run_builder_tool("repo.edit", {"operations": 42}, engine=_Engine())
        self.assertFalse(got.ok)
        self.assertIn("operations", got.error)
        self.assertIn("insert_after", got.error, "the refusal still lists the accepted ops")

    def test_valid_json_string_is_not_refused_here(self):
        """A well-formed JSON string must reach the engine, not be rejected."""
        class Ok:
            def apply_typed_mutation(self, operations, tests):
                assert isinstance(operations, list), operations
                return {"status": "unproven", "applied": True, "rolled_back": False,
                        "reason": None, "diff": "", "paths": ["x.py"]}
        good = '[{"op": "create", "path": "x.py", "new_lines": ["X = 1"]}]'
        got = run_builder_tool("repo.edit", {"operations": good}, engine=Ok())
        self.assertTrue(got.ok, got.error)


if __name__ == "__main__":
    unittest.main()

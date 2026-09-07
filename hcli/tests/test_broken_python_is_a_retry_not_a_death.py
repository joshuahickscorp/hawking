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
    # `create`, so the body is judged as a whole module. A `replace` fragment is
    # spliced into a file and is judged by whether the RESULT compiles.
    return json.dumps({
        "kind": "mutation", "content": "x", "tests": [], "tool_calls": [],
        "operations": [{"op": "create", "path": path, "new_text": body}],
    })


class TestFragmentsAreNotCompiledAlone(unittest.TestCase):
    """A replace fragment is spliced into a file, not a module.

    The first version of this check compiled `new_text` on its own, so an
    indented block -- the correct replacement for an indented block -- was
    reported as "unexpected indent at line 1" and REJECTED. Three consecutive
    goals died being told to fix code that was not broken.

    A check that refuses correct work is worse than no check: it converts a
    working model into a failing one and hides the fact behind a plausible
    error message.
    """

    ANCHOR = "    files_seen = 0\n"

    def _op(self, new_text, op="replace"):
        return json.dumps({"kind": "mutation", "operations": [{
            "op": op, "path": "hcli/tool_registry.py",
            "old_text": self.ANCHOR, "new_text": new_text,
        }]})

    def test_an_indented_fragment_is_accepted(self):
        self.assertIsNone(
            _python_syntax_violation(self._op("    files_seen = 0\n    extra = 1\n"))
        )

    def test_a_fragment_that_breaks_the_file_is_still_caught(self):
        got = _python_syntax_violation(self._op("    files_seen = (0\n"))
        self.assertIsNotNone(got)
        self.assertIn("resulting file", got)

    def test_a_missing_anchor_is_a_RETRY_not_a_death(self):
        """Deliberately changed: this used to be left to the verifier.

        _apply_operations runs AFTER the contract accepts the reply, so an
        anchor that did not match killed the unit while holding a patch that
        was right except for its anchor. Measured: one literal backslash-n
        where a newline belonged, every other line correct. It is correctable,
        so it now goes back through the contract's retry with the file's real
        bytes attached.
        """
        payload = json.dumps({"kind": "mutation", "operations": [{
            "op": "replace", "path": "hcli/tool_registry.py",
            "old_text": "this text is not in the file anywhere",
            "new_text": "    x = 1\n",
        }]})
        got = _python_syntax_violation(payload)
        self.assertIsNotNone(got)
        self.assertIn("not one line of it", got)
        self.assertIn("trailing spaces", got)
        self.assertLess(len(got), 400)

    def test_an_ambiguous_anchor_says_so(self):
        """Matching many places needs the opposite fix from matching none."""
        payload = json.dumps({"kind": "mutation", "operations": [{
            "op": "replace", "path": "hcli/tool_registry.py",
            "old_text": "\n", "new_text": "x",
        }]})
        got = _python_syntax_violation(payload)
        self.assertIsNotNone(got)
        self.assertIn("must match", got)

    def test_a_create_still_compiles_standalone(self):
        """A created file IS a whole module, so the old rule is right there."""
        got = _python_syntax_violation(self._op("def f(:\n", op="create"))
        self.assertIsNotNone(got)


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
        self.assertIn("fix that operation", got)

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


class TestOnlyAMutationIsJudgedOnItsOperations(unittest.TestCase):
    """A tool_use reply carries placeholder operations. Judging them is a trap.

    Measured: the model asked to read hcli/tool_registry.py -- content "need
    exact unique anchor for _read_file total_lines" -- and filled old_text with
    "x" to satisfy the reply shape. The anchor preflight, added that same day to
    make anchors CORRECTABLE, refused it with "matches 497 places" on all three
    attempts. The model could never obtain the bytes the refusal demanded,
    because the request for them was what kept being refused.
    """

    def _reply(self, kind):
        return json.dumps({
            "kind": kind,
            "content": "need exact unique anchor for _read_file total_lines",
            "operations": [{
                "op": "replace", "path": "hcli/tool_registry.py",
                "old_text": "x", "new_text": "x",
            }],
            "tool_calls": [{"tool": "fs.read", "arguments": []}],
        })

    def test_a_tool_use_reply_is_not_judged_on_its_placeholder_operations(self):
        self.assertIsNone(_python_syntax_violation(self._reply("tool_use")))

    def test_an_answer_reply_is_not_judged_either(self):
        self.assertIsNone(_python_syntax_violation(self._reply("answer")))

    def test_a_mutation_reply_IS_still_judged(self):
        """The check must not be disabled, only aimed."""
        got = _python_syntax_violation(self._reply("mutation"))
        self.assertIsNotNone(got)
        self.assertIn("must match exactly one", got)

    def test_a_reply_with_no_kind_is_not_judged(self):
        """Absent kind is not 'mutation'; do not guess it into one."""
        self.assertIsNone(_python_syntax_violation(json.dumps({
            "operations": [{
                "op": "replace", "path": "hcli/tool_registry.py",
                "old_text": "x", "new_text": "x",
            }],
        })))


class TestCreatingAnExistingFileIsARetry(unittest.TestCase):
    """Offering to create a file that exists is correctable, not fatal.

    _apply_operations runs after the contract accepts, so this killed the unit
    outright. Measured twice on one goal: the model kept offering to author the
    RED spec file, which the goal had told it already existed, and each attempt
    died holding whatever else it had proposed alongside it.
    """

    def _payload(self, path):
        return json.dumps({"kind": "mutation", "operations": [
            {"op": "create", "path": path, "new_text": "x = 1\n"},
        ]})

    def test_creating_an_existing_file_returns_a_correction(self):
        got = _python_syntax_violation(self._payload("hcli/tool_registry.py"))
        self.assertIsNotNone(got)
        self.assertIn("already exists", got)
        self.assertIn("replace operation", got)

    def test_creating_a_new_file_is_still_allowed(self):
        self.assertIsNone(
            _python_syntax_violation(self._payload("hcli/tests/_no_such_file.py"))
        )

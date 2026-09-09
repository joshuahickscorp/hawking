"""A builder call whose arguments arrive as text must still reach the engine.

`coerce_arguments` exists because THE XML DIALECT CANNOT EXPRESS A NUMBER OR A
LIST: the artifact's own template emits `<parameter=operations>[...]</parameter>`,
so every value arrives as a string by construction. Its docstring records the
measured cost of that for `fs.list` -- the body sent max_results "200" twice and
got "expected integer, got string" both times.

But the coercion is driven by `registry.get(name)`, and BUILDER tools are not
registry tools. `repo.edit`'s registry spec carries no properties, so `props` is
empty, `coerce_arguments` returns the arguments untouched, `operations` stays a
string, and `run_builder_tool`'s `isinstance(operations, list)` check refuses it
with a shape error.

The consequence measured live: in cycles 69 and 70 the body finally reached for
`repo.edit` after 68 cycles of never trying -- with a correct-looking call
carrying full `new_lines`, `old_lines: []` and a `tests` argument -- and the
engine rejected both for malformed `operations`. The body was doing exactly what
it was asked; the substrate was dropping its arguments on the floor.

The same door for a registry tool works, because a registry tool has a schema.
So this is not "repo.edit is hard to call", it is "builder doors lose their
arguments in a way registry doors do not".
"""
from __future__ import annotations

import json
import unittest

from hcli.chat_tools import build_registry, coerce_arguments

OPERATIONS = [{"op": "create", "path": "x.py", "new_lines": ["X = 1"], "old_lines": []}]


class TestBuilderArgumentsSurviveTheXmlDialect(unittest.TestCase):
    def setUp(self):
        self.registry = build_registry(".", ".")

    def test_repo_edit_operations_arrive_as_a_list(self):
        got = coerce_arguments(self.registry, "repo.edit",
                               {"operations": json.dumps(OPERATIONS)})
        self.assertIsInstance(got["operations"], list,
                              "operations arrived as text and nothing converted it")
        self.assertEqual(got["operations"], OPERATIONS)

    def test_repo_edit_tests_arrive_as_a_list(self):
        got = coerce_arguments(self.registry, "repo.edit",
                               {"operations": json.dumps(OPERATIONS),
                                "tests": '["hcli/tests/test_x.py"]'})
        self.assertEqual(got["tests"], ["hcli/tests/test_x.py"])

    def test_a_bare_path_string_becomes_a_one_element_list(self):
        """The dialect may also send a single path with no brackets."""
        got = coerce_arguments(self.registry, "repo.edit",
                               {"operations": json.dumps(OPERATIONS),
                                "tests": "hcli/tests/test_x.py"})
        self.assertEqual(got["tests"], ["hcli/tests/test_x.py"])

    def test_a_registry_tool_is_unaffected(self):
        """Negative control: the existing registry path must not change."""
        got = coerce_arguments(self.registry, "fs.read",
                               {"path": "hcli/mutation.py", "start_line": "70"})
        self.assertEqual(got["start_line"], 70)
        self.assertEqual(got["path"], "hcli/mutation.py")

    def test_unconvertible_values_are_still_passed_through(self):
        """A value that is not valid JSON stays untouched so the tool's own
        validator still owns the refusal -- coercion must not invent data."""
        got = coerce_arguments(self.registry, "repo.edit",
                               {"operations": "not json at all"})
        self.assertEqual(got["operations"], "not json at all")


if __name__ == "__main__":
    unittest.main()

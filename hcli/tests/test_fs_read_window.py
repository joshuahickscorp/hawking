"""fs.read must be able to read a window, not only the top of a file.

`fs.search` reports the line a match is on. `fs.read` took only
`path, max_bytes, encoding`, so it returned the FIRST 4,001 bytes of a
188,062-byte engine.py and there was no way to reach anything further down.

A live goal died on exactly that. The model located `_record_model_call` at
line 3514 and then reported:

    Need to see the actual _record_model_call function and the model_call entry
    structure in engine.py to implement the grammar_enforced field correctly.

It could find the code and could not look at it. Any task touching a real file
is unreachable in that state, which is every level of the ladder above
read-and-explain.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from hcli.tool_registry import default_tool_registry

REPO = Path(__file__).resolve().parents[2]


def _reg():
    return default_tool_registry(REPO)


def _value(result):
    return getattr(result, "value", None) or {}


class TestReadWindow(unittest.TestCase):
    def test_search_then_read_reaches_a_deep_function(self):
        reg = _reg()
        matches = _value(
            reg.invoke("fs.search", {"pattern": "def _record_model_call", "path": "hcli"})
        ).get("matches") or []
        self.assertTrue(matches, "fs.search could not find the function")
        line = matches[0]["line"]
        self.assertGreater(line, 1000, "not the deep-file case this protects")

        window = _value(
            reg.invoke("fs.read", {"path": "hcli/engine.py",
                                   "start_line": line, "end_line": line + 20})
        )
        self.assertIn("_record_model_call", window.get("content") or "")
        self.assertEqual(window.get("start_line"), line)

    def test_the_window_reports_where_it_is(self):
        """A window with no coordinates cannot be reasoned about."""
        w = _value(_reg().invoke(
            "fs.read", {"path": "hcli/engine.py", "start_line": 10, "end_line": 12}
        ))
        self.assertEqual((w.get("start_line"), w.get("end_line")), (10, 12))
        self.assertGreater(w.get("total_lines") or 0, 12)
        self.assertEqual(len((w.get("content") or "").splitlines()), 3)

    def test_a_whole_file_read_is_unchanged(self):
        """Negative control: the default path must keep its old behaviour."""
        w = _value(_reg().invoke("fs.read", {"path": "hcli/engine.py"}))
        self.assertTrue(w.get("truncated"))
        self.assertIsNone(w.get("start_line"))

    def test_an_out_of_range_window_is_not_an_error(self):
        w = _value(_reg().invoke(
            "fs.read", {"path": "hcli/engine.py", "start_line": 10**9}
        ))
        self.assertEqual(w.get("content"), "")

    def test_an_out_of_range_window_SAYS_it_is_out_of_range(self):
        """Empty-because-it-does-not-exist must not read as empty-because-it-is-empty.

        A 66-cycle autonomous run died on this. The body believed a function lived
        past the end of hcli/mutation.py and walked the window forward looking for
        it -- cycle 47 asked for lines 1620-1720 of a 247-line file. Every one of
        those reads came back ok=True, content="", truncated=False: a SUCCESSFUL
        read that returned nothing, with an explicit claim that nothing was cut.
        Nothing in the payload said "that window does not exist", so the belief
        was never corrected by evidence and the body kept walking.

        `truncated` cannot carry this: it answers "was content cut?", and the
        honest answer here is no -- there was no content. The window itself is the
        thing that is wrong, so it needs its own signal.
        """
        w = _value(_reg().invoke(
            "fs.read", {"path": "hcli/mutation.py", "start_line": 1620, "end_line": 1720}
        ))
        total = w.get("total_lines") or 0
        self.assertGreater(total, 0)
        self.assertEqual(w.get("content"), "")          # unchanged: still not an error
        self.assertTrue(w.get("out_of_range"),
                        "an empty window past EOF must be flagged, not reported as a clean read")
        note = w.get("window_note") or ""
        self.assertIn(str(total), note, "the note must state the file's real length")
        self.assertIn("1620", note, "the note must name the window that does not exist")

    def test_the_reported_window_is_never_inverted(self):
        """start_line=1620,end_line=247 is not a coordinate pair, it is a contradiction."""
        w = _value(_reg().invoke(
            "fs.read", {"path": "hcli/mutation.py", "start_line": 1620, "end_line": 1720}
        ))
        self.assertLessEqual(
            w.get("start_line"), w.get("end_line"),
            "reported end_line must never precede start_line")

    def test_a_window_inside_the_file_is_not_flagged(self):
        """Negative control: the flag must not fire on an ordinary read."""
        w = _value(_reg().invoke(
            "fs.read", {"path": "hcli/mutation.py", "start_line": 70, "end_line": 80}
        ))
        self.assertFalse(w.get("out_of_range"))
        self.assertIsNone(w.get("window_note"))
        self.assertIn("_apply_replace", w.get("content") or "")

    def test_the_catalog_advertises_the_window(self):
        """A capability the model cannot see is a capability it does not have."""
        from hcli.engine import Engine

        catalog = Engine._compact_tool_catalog(_reg())
        self.assertIn("start_line", catalog)
        self.assertIn("end_line", catalog)


if __name__ == "__main__":
    unittest.main()

"""Inlined evidence must be the part of the file the request is about.

Evidence reached the model as the FIRST N characters of the file. A goal about
`_read_file` at line 509 of a 100 KB file received about the first 150 lines and
then had to spend a tool round -- roughly 150 s at 25 prompt tok/s -- asking for
the rest of a file it had already been handed.

Same defect as an fs.read with no offset: the interesting code is never at the
top. The window is centred on the densest run of identifiers the prompt and the
file share, with a DEFINITION outweighing a mention, because density alone chose
the block where fs.read is registered over the function that implements it.
"""
from __future__ import annotations

import unittest

from hcli.engine import _focused_excerpt

BODY = "\n".join(
    ["# header"] * 400
    + ["def pid_is_alive(pid):", "    return True"]
    + ["# tail"] * 400
) + "\n"


class TestFocusedExcerpt(unittest.TestCase):
    def test_the_window_finds_a_deep_definition(self):
        out = _focused_excerpt(BODY, "add a docstring to pid_is_alive", 2000, "x.py")
        self.assertIn("def pid_is_alive", out)

    def test_the_head_would_have_missed_it(self):
        """The failure being fixed: truncation from the top."""
        self.assertNotIn("def pid_is_alive", BODY[:2000])

    def test_the_excerpt_says_where_it_came_from(self):
        out = _focused_excerpt(BODY, "pid_is_alive", 2000, "x.py")
        self.assertIn("x.py lines", out)
        self.assertIn("of 802", out)

    def test_a_small_file_is_returned_whole(self):
        """Negative control: no windowing when none is needed."""
        small = "def f():\n    return 1\n"
        self.assertEqual(_focused_excerpt(small, "f", 5000, "s.py"), small)

    def test_it_respects_the_limit(self):
        out = _focused_excerpt(BODY, "pid_is_alive", 500, "x.py")
        self.assertLessEqual(len(out), 500 + 200)  # + the one-line header

    def test_no_match_falls_back_to_the_head(self):
        """No worse than the old behaviour when nothing matches."""
        out = _focused_excerpt(BODY, "zzz_nothing_matches_zzz", 300, "x.py")
        self.assertTrue(out.startswith("# header"))

    def test_a_definition_beats_a_mention(self):
        body = (
            "\n".join(["    call_thing()"] * 300)
            + "\n"
            + "\n".join(["# pad"] * 300)
            + "\ndef call_thing():\n    return 2\n"
        )
        out = _focused_excerpt(body, "change call_thing", 900, "y.py")
        self.assertIn("def call_thing", out)


class TestSymbolTargeting(unittest.TestCase):
    """`ast` knows where a definition is. Do not guess at it.

    Lexical density chose the block where fs.read is REGISTERED over the
    `_read_file` that implements it -- both mention the name, only one is the
    thing being changed. The parser has no such ambiguity, so exact symbol wins
    and density is the fallback.
    """

    def _engine(self):
        from pathlib import Path

        from hcli.engine import Engine

        eng = Engine.__new__(Engine)
        eng.MAX_EVIDENCE_FILES = 16
        eng.MAX_EVIDENCE_CHARS_PER_FILE = 24000
        eng.MAX_TOTAL_EVIDENCE_CHARS = 120000
        eng.root = Path(__file__).resolve().parents[2]
        eng._context_budget = lambda: type("B", (), {"usable_input_tokens": 5632})()
        eng._chars_per_token = None
        return eng

    def test_the_parser_finds_definitions(self):
        from hcli.engine import _python_symbol_lines

        src = "x = 1\n\n\ndef f():\n    pass\n\n\nclass C:\n    def m(self):\n        pass\n"
        got = _python_symbol_lines(src)
        self.assertEqual(got["f"], 4)
        self.assertEqual(got["C"], 8)
        self.assertEqual(got["m"], 9)
        self.assertEqual(got["x"], 1)

    def test_unparseable_python_is_not_fatal(self):
        """Negative control: a broken file must fall back, not raise."""
        from hcli.engine import _python_symbol_lines

        self.assertEqual(_python_symbol_lines("def f(:\n"), {})

    def test_exact_symbol_beats_the_registration_block(self):
        """The measured miss: `_read_file` at 509 vs its registration at ~1830."""
        got = self._engine()._gather_evidence(
            "hcli/tool_registry.py change _read_file to report total_lines"
        )
        self.assertTrue(got)
        self.assertIn("def _read_file", got[0].get("content") or "")

    def test_it_works_on_a_large_file(self):
        got = self._engine()._gather_evidence(
            "hcli/engine.py change _resolve_max_tokens"
        )
        self.assertTrue(got)
        self.assertIn("def _resolve_max_tokens", got[0].get("content") or "")


class TestItIsActuallyWired(unittest.TestCase):
    """The call site, not the helper.

    Third time in this campaign that a helper was tested while the wiring was
    not, and each time reverting the wiring left every test green: the prompt
    sanitizer, the JSON token mask, and now this. The helper is never the thing
    that failed in production.
    """

    def test_gather_evidence_calls_the_focuser(self):
        import inspect

        from hcli.engine import Engine

        src = inspect.getsource(Engine._gather_evidence)
        self.assertIn("_focused_excerpt(content, prompt", src)
        focus = inspect.getsource(__import__("hcli.engine", fromlist=["x"])._focused_excerpt)
        self.assertIn("_python_symbol_lines", focus)
        self.assertNotIn("content = content[\n                :per_file_limit", src)

    def test_a_deep_definition_survives_the_real_path(self):
        """End to end through _gather_evidence, on a real repository file."""
        from pathlib import Path

        from hcli.engine import Engine

        eng = Engine.__new__(Engine)
        eng.MAX_EVIDENCE_FILES = 16
        eng.MAX_EVIDENCE_CHARS_PER_FILE = 24000
        eng.MAX_TOTAL_EVIDENCE_CHARS = 120000
        eng.root = Path(__file__).resolve().parents[2]
        eng._context_budget = lambda: type("B", (), {"usable_input_tokens": 5632})()
        eng._chars_per_token = None

        got = eng._gather_evidence("hcli/resources.py add a docstring to pid_is_alive")
        self.assertTrue(got, "no evidence gathered at all")
        content = got[0].get("content") or ""
        self.assertIn("def pid_is_alive", content)


if __name__ == "__main__":
    unittest.main()

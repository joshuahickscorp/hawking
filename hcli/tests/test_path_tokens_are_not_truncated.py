"""A longer extension must not be truncated to a shorter known one.

The extension alternation had no trailing boundary, so `.jsonl` matched the
`.js`/`.json` alternative and stopped there. `COMPILE_ECONOMICS.jsonl` came out
as `COMPILE_ECONOMICS.js`, which does not exist, so `_safe_path` refused it and
the packet inlined NO evidence:

    packet.evidence_paths: ('COMPILE_ECONOMICS.js',)
    packet.evidence:       ()

The cost is a whole tool round. The packet names the file that matters, the
harness fails to read it, and the model spends about 150 s asking for a file we
had already identified. Measured shape of a unit: four calls, three of them
tool rounds before any work begins.
"""
from __future__ import annotations

import unittest

from hcli.engine import _PATH_TOKEN_RE


def tokens(text: str) -> list[str]:
    return [m.group("path") or m.group("qpath") for m in _PATH_TOKEN_RE.finditer(text)]


class TestPathTokens(unittest.TestCase):
    def test_jsonl_survives(self):
        self.assertEqual(
            tokens("read .hcli/school/COMPILE_ECONOMICS.jsonl now"),
            [".hcli/school/COMPILE_ECONOMICS.jsonl"],
        )

    def test_json_still_matches(self):
        """Negative control: bounding must not break the shorter extension."""
        self.assertEqual(tokens("open a.json"), ["a.json"])

    def test_js_still_matches(self):
        self.assertEqual(tokens("open a.js"), ["a.js"])

    def test_ordinary_source_paths_are_unaffected(self):
        self.assertEqual(
            tokens("edit hcli/engine.py and crates/x/src/lib.rs"),
            ["hcli/engine.py", "crates/x/src/lib.rs"],
        )

    def test_an_unknown_extension_is_not_a_path(self):
        self.assertEqual(tokens("not_a_path.zzz"), [])

    def test_a_known_extension_inside_a_longer_word_is_not_split(self):
        """The exact failure: .js must not be plucked out of .jsonl."""
        self.assertNotIn("x.js", tokens("x.jsonl"))

    def test_the_boundary_guards_extensions_NOT_in_the_list(self):
        """What the lookahead is for, as opposed to adding `jsonl` to the list.

        Adding one extension fixes one case. The boundary fixes the class: any
        longer extension that merely STARTS with a known one must not be
        truncated into a path that does not exist. `.pyc` and `.jsonc` are not
        source paths and must not be read as `.py` and `.json`.
        """
        self.assertEqual(tokens("build/mod.pyc"), [])
        self.assertEqual(tokens("cfg.jsonc"), [])
        self.assertEqual(tokens("a.tsv"), [])


class TestPacketFileExtractor(unittest.TestCase):
    """The extractor that actually feeds EVIDENCE_PATHS.

    Two extractors had this bug and only this one reaches the packet, so fixing
    the engine's regex alone changed nothing observable. In this one `js`
    preceded `json` in the alternation, so `.jsonl` truncated to `.js`.
    """

    def _files(self, text: str):
        from hcli.goal import GoalCompiler

        return GoalCompiler()._referenced_files(text)

    def test_jsonl_reaches_the_packet(self):
        self.assertIn(
            "workspace/campaign/odyssey/COMPILE_ECONOMICS.jsonl",
            self._files("read workspace/campaign/odyssey/COMPILE_ECONOMICS.jsonl"),
        )

    def test_it_does_not_also_yield_the_truncation(self):
        got = self._files("read a/COMPILE_ECONOMICS.jsonl")
        self.assertNotIn("a/COMPILE_ECONOMICS.js", got)

    def test_ordinary_paths_still_extract(self):
        """Negative control."""
        got = self._files("edit hcli/engine.py and docs/x.md")
        self.assertIn("hcli/engine.py", got)
        self.assertIn("docs/x.md", got)

    def test_a_longer_unknown_extension_is_refused(self):
        self.assertEqual(self._files("build/mod.pyc"), [])


if __name__ == "__main__":
    unittest.main()

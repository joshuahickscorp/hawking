"""fs.* tools must name their location the same way.

fs.read, fs.list and fs.write all take `path`. fs.search alone took `root`, so
the consistent guess was a hard schema error:

    $: unexpected properties ['path']    failure_class='INVALID_ARGUMENT'

The model made that guess, and then reported the result as ZERO MATCHES -- it
could not tell "you called it wrong" from "nothing found". It went on to answer
the question correctly but hedged it as unevidenced, because as far as it could
tell its own search had found nothing in the file it was asked about.

One inconsistent word cost a round and the confidence of a correct answer.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

from hcli.tool_registry import default_tool_registry

REPO = Path(__file__).resolve().parents[2]


def _registry():
    return default_tool_registry(REPO)


def _specs():
    out = {}
    for entry in _registry().discover():
        spec = ast.literal_eval(entry) if isinstance(entry, str) else entry
        out[spec.get("name")] = spec
    return out


class TestOneVocabulary(unittest.TestCase):
    def test_aliases_are_callable_but_not_separate_capabilities(self):
        registry = _registry()
        canonical = {entry["name"]: entry for entry in registry.discover()}
        self.assertEqual(canonical["fs.read"]["aliases"], ["filesystem.read"])
        self.assertNotIn("filesystem.read", canonical)
        self.assertEqual(registry.get("filesystem.read").alias_of, "fs.read")
        self.assertTrue(
            registry.invoke("filesystem.read", {"path": "hcli/__init__.py"}).ok
        )

    def test_every_fs_tool_accepts_path(self):
        specs = _specs()
        for name, spec in specs.items():
            if not name.startswith(("fs.", "filesystem.")):
                continue
            props = (spec.get("input_schema") or {}).get("properties") or {}
            self.assertIn(
                "path", props,
                f"{name} does not accept `path`, so the consistent guess is a "
                "schema error the model reads as an empty result",
            )

    def test_search_by_path_finds_what_is_there(self):
        result = _registry().invoke(
            "fs.search", {"pattern": "os.waitpid", "path": "hcli"}
        )
        self.assertTrue(getattr(result, "ok", False), getattr(result, "error", None))
        matches = (getattr(result, "value", None) or {}).get("matches") or []
        self.assertTrue(matches, "fs.search with `path` found nothing that exists")

    def test_root_still_works(self):
        """Negative control: the alias must not displace the original name."""
        result = _registry().invoke(
            "fs.search", {"pattern": "os.waitpid", "root": "hcli"}
        )
        self.assertTrue(getattr(result, "ok", False), getattr(result, "error", None))
        self.assertTrue((getattr(result, "value", None) or {}).get("matches"))

    def test_both_names_agree(self):
        reg = _registry()
        by_path = reg.invoke("fs.search", {"pattern": "os.waitpid", "path": "hcli"})
        by_root = reg.invoke("fs.search", {"pattern": "os.waitpid", "root": "hcli"})
        self.assertEqual(
            len((getattr(by_path, "value", None) or {}).get("matches") or []),
            len((getattr(by_root, "value", None) or {}).get("matches") or []),
        )


if __name__ == "__main__":
    unittest.main()

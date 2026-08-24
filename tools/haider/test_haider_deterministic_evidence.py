#!/usr/bin/env python3
"""Deterministic evidence discovery tests."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import haider


class TestDeterministicEvidence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.root.mkdir(parents=True, exist_ok=True)
        # Create a fake git repo structure
        (self.root / ".git").mkdir()
        (self.root / "src").mkdir()
        (self.root / "src" / "main.py").write_text("print('hello')\n")
        (self.root / "src" / "test_main.py").write_text("def test(): pass\n")
        (self.root / "README.md").write_text("# Test\n")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_deterministic_evidence_no_model(self):
        """Evidence discovery must work without any model call."""
        guard = haider.p0.RepositoryGuard(str(self.root))
        evidence = haider._deterministic_evidence(guard, "fix the tests")
        self.assertTrue(evidence["ok"])
        self.assertEqual(evidence["stats"]["model_turns"], 0)
        self.assertGreaterEqual(evidence["stats"]["fast_evidence_files"], 2)
        # Should find the test file
        paths = [f["path"] for f in evidence["files"]]
        self.assertTrue(any("test" in p for p in paths))

    def test_deterministic_evidence_named_file(self):
        """Named file evidence should be fast and deterministic."""
        guard = haider.p0.RepositoryGuard(str(self.root))
        evidence = haider._deterministic_evidence(guard, "fix src/main.py")
        self.assertTrue(evidence["ok"])
        self.assertEqual(evidence["stats"]["model_turns"], 0)
        paths = [f["path"] for f in evidence["files"]]
        self.assertIn("src/main.py", paths)

    def test_deterministic_evidence_git_tracked(self):
        """Git tracked files should be used for discovery."""
        guard = haider.p0.RepositoryGuard(str(self.root))
        evidence = haider._deterministic_evidence(guard, "implement feature")
        self.assertTrue(evidence["ok"])
        self.assertEqual(evidence["stats"]["model_turns"], 0)

    def test_deterministic_evidence_non_git(self):
        """Non-git directories should still work deterministically."""
        non_git = Path(self.tmpdir.name + "_nongit")
        non_git.mkdir()
        (non_git / "app.py").write_text("x = 1\n")
        guard = haider.p0.RepositoryGuard(str(non_git))
        evidence = haider._deterministic_evidence(guard, "fix app")
        self.assertTrue(evidence["ok"])
        self.assertEqual(evidence["stats"]["model_turns"], 0)


if __name__ == "__main__":
    unittest.main()
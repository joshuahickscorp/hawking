#!/usr/bin/env python3
import sys, os, tempfile, unittest
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hcli.workspace import Workspace

class TestWorkspace(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "ULTRAGOAL.md").write_text("# Goal\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_resolve(self):
        ws = Workspace(str(self.root))
        self.assertEqual(ws.resolve("ULTRAGOAL.md"), "ULTRAGOAL.md")

    def test_resolve_missing(self):
        ws = Workspace(str(self.root))
        self.assertIsNone(ws.resolve("MISSING.md"))

if __name__ == "__main__":
    unittest.main()

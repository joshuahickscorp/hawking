import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import haider
import p0_tool_bridge as p0


class MissionStateTests(unittest.TestCase):
    def test_create_state(self):
        state = haider.create_mission_state(
            "implement thing",
            "inline",
            8,
        )

        self.assertTrue(state["mission_id"])
        self.assertEqual(state["task"], "implement thing")
        self.assertEqual(state["source"], "inline")
        self.assertEqual(state["status"], "running")
        self.assertEqual(state["cycle"], 0)
        self.assertEqual(state["max_cycles"], 8)

    def test_bad_cycles(self):
        with self.assertRaises(ValueError):
            haider.create_mission_state("x", "inline", 0)

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as root:
            old = haider.MISSIONS_DIR

            try:
                haider.MISSIONS_DIR = Path(root)

                state = haider.create_mission_state(
                    "persist me",
                    "inline",
                    3,
                )

                path = haider.write_mission_state(state)
                loaded = haider.load_mission_state(path)

                self.assertEqual(
                    loaded["mission_id"],
                    state["mission_id"],
                )
                self.assertEqual(loaded["task"], "persist me")
                self.assertEqual(loaded["max_cycles"], 3)

                json.loads(path.read_text())

            finally:
                haider.MISSIONS_DIR = old

    def test_fast_evidence_named_file(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)

            (root_path / "tools").mkdir()
            target = root_path / "tools" / "thing.py"
            target.write_text("VALUE = 1\n")

            guard = p0.RepositoryGuard(root)

            evidence = haider.build_fast_mission_evidence(
                "Modify tools/thing.py to improve VALUE.",
                guard,
            )

            self.assertIsNotNone(evidence)
            self.assertEqual(
                evidence["stats"]["model_turns"],
                0,
            )
            self.assertIn(
                "VALUE = 1",
                evidence["final"],
            )


if __name__ == "__main__":
    unittest.main()

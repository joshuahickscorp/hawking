from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[4]

from hcli.grok_bridge import GrokBridge, GrokRunError
from hcli.report_compiler import compile_backend_report


FAKE_BIN = "/fake/grok-run"


class TestGrokWaitPolls(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.bridge = GrokBridge(self.root)
        self.which = patch(
            "hcli.grok_bridge.shutil.which",
            return_value=FAKE_BIN,
        )
        self.which.start()

    def tearDown(self):
        self.which.stop()
        self.tmpdir.cleanup()

    def test_wait_polls_status_never_invokes_grok_run_wait(self):
        states = [
            {"state": "running", "exit_code": None, "task_id": "tid-1"},
            {"state": "done", "exit_code": 0, "task_id": "tid-1"},
        ]

        def fake_status(task_id):
            return dict(states.pop(0))

        spawned = []

        def boom(*a, **k):
            spawned.append((a, k))
            raise AssertionError("subprocess spawned for wait")

        with patch.object(self.bridge, "status", side_effect=fake_status):
            with patch(
                "hcli.grok_bridge.subprocess.run",
                side_effect=boom,
            ):
                got = self.bridge.wait("tid-1", timeout=5, poll_interval=0.01)
        self.assertEqual(got["state"], "done")
        self.assertEqual(spawned, [])

    def test_wait_timeout_does_not_use_subprocess_timeout_120(self):
        def always_running(task_id):
            return {"state": "running", "exit_code": None, "task_id": task_id}

        with patch.object(self.bridge, "status", side_effect=always_running):
            with self.assertRaises(GrokRunError) as ctx:
                self.bridge.wait("tid-slow", timeout=0.05, poll_interval=0.01)
        self.assertIn("timed out", str(ctx.exception).lower())
        argv = ctx.exception.argv or []
        self.assertNotIn("wait", argv)

    def test_compact_report_does_not_dump_raw_trace(self):
        task_id = "slug-20260101-000000"
        task_dir = self.root / "task"
        task_dir.mkdir()
        raw = task_dir / "grok-report.md"
        raw.write_text(
            "SUMMARY: grounded.\n"
            "<think>secret chain of thought that must not leak</think>\n"
            '{"tool": "shell", "cmd": "cat /etc/passwd"}\n'
            "error: boom\n"
            "$ python3 -m pytest tools/haider/hcli/tests -q\n"
            "tools/haider/hcli/commands.py changed\n",
            encoding="utf-8",
        )
        receipt = {
            "task_id": task_id,
            "mode": "consult",
            "task_dir": str(task_dir),
            "report_path": str(raw),
            "workspace": str(self.root),
            "status": {"state": "done", "exit_code": 0},
        }
        self.bridge.receipts_dir.mkdir(parents=True, exist_ok=True)
        (self.bridge.receipts_dir / f"{task_id}.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
        compact = self.bridge.compact_report(task_id)
        self.assertEqual(compact["backend"], "grok")
        self.assertEqual(compact["task_id"], task_id)
        self.assertEqual(compact["raw_report_path"], str(raw))
        self.assertNotIn("/etc/passwd", compact["final_summary"])
        self.assertTrue(compact.get("errors"))
        dest = Path(compact["compact_path"])
        self.assertTrue(dest.is_file())
        blob = dest.read_text(encoding="utf-8")
        self.assertLess(len(blob), 8000)

    def test_compile_backend_report_shape(self):
        compact = compile_backend_report(
            backend="grok",
            task_id="t",
            raw_text="hello world\n- claim one\n",
            raw_report_path="/tmp/raw.md",
        )
        for key in (
            "backend",
            "task_id",
            "final_summary",
            "claims",
            "evidence_refs",
            "files_touched",
            "commands_executed",
            "verifier_inputs",
            "errors",
            "raw_report_path",
        ):
            self.assertIn(key, compact)


if __name__ == "__main__":
    unittest.main()

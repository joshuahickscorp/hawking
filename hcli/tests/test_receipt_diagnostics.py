from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli.engine import Engine
from hcli.events import EventBus
from hcli.workspace import Workspace


class TestReceiptDiagnostics(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.workspace = Workspace(str(self.root))
        self.bus = EventBus()
        self.engine = Engine(
            workspace=self.workspace,
            event_bus=self.bus,
            runtime_count=1,
            model_name="/m.gguf",
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def _receipts(self):
        receipt_dir = self.root / ".hcli" / "receipts"
        return list(receipt_dir.glob("*.json")) if receipt_dir.is_dir() else []

    def test_failed_receipt_has_error_fields(self):
        def boom(prompt, evidence, compiled):
            raise RuntimeError("runtime unreachable")

        self.engine._call_model = boom
        result = self.engine.execute("do a thing")
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["error"])
        self.assertEqual(result["error_type"], "RuntimeError")
        paths = self._receipts()
        self.assertEqual(len(paths), 1)
        receipt = json.loads(paths[0].read_text())
        self.assertTrue(receipt["error"])
        self.assertEqual(receipt["error_type"], "RuntimeError")
        self.assertTrue(receipt["error_traceback"])
        self.assertIn("RuntimeError", receipt["error_traceback"])

    def test_empty_str_exc_still_records_type(self):
        def interrupt(prompt, evidence, compiled):
            raise KeyboardInterrupt()

        self.engine._call_model = interrupt
        result = self.engine.execute("stop me")
        self.assertTrue(result["error"])
        self.assertEqual(result["error_type"], "KeyboardInterrupt")
        receipt = json.loads(self._receipts()[0].read_text())
        self.assertTrue(receipt["error"])
        self.assertEqual(receipt["error_type"], "KeyboardInterrupt")
        self.assertTrue(receipt["error_traceback"])

    def test_successful_receipt_has_no_error_key(self):
        self.engine._call_model = lambda p, e, c: {
            "kind": "answer",
            "content": "ok",
            "operations": [],
            "tests": [],
        }
        result = self.engine.execute("say ok")
        self.assertEqual(result["status"], "completed")
        receipt = json.loads(self._receipts()[0].read_text())
        self.assertNotIn("error", receipt)
        self.assertNotIn("error_type", receipt)
        self.assertNotIn("error_traceback", receipt)


if __name__ == "__main__":
    unittest.main()

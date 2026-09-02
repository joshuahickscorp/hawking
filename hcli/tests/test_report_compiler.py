from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hcli.report_compiler import (
    compile_backend_report,
    payload_dumps,
)


def _compile(raw, extra=None, path="/tmp/raw.md"):
    return compile_backend_report(
        backend="grok",
        task_id="t",
        raw_text=raw,
        raw_report_path=path,
        extra=extra or {},
    )


class TestCompileBackendReportShape(unittest.TestCase):
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


class TestReportCompactionSemantics(unittest.TestCase):
    def test_tiny_passthrough_does_not_inflate(self):
        raw = "nonce-99127db133804df3a2bd6d6b187b1870\n"  # 39 bytes, live shape
        c = _compile(raw, extra={"mode": "consult", "status": {"state": "done", "exit_code": 0}})
        self.assertEqual(c["raw_report_path"], "/tmp/raw.md")
        self.assertIn("nonce-99127db133804df3a2bd6d6b187b1870", c["final_summary"])
        n = len(payload_dumps(c))
        self.assertLess(n, max(len(raw), 64))
        self.assertTrue(c.get("passthrough") or n <= len(raw))

    def test_medium_keeps_decision_critical(self):
        raw = (
            "FINDING: buffer overflow in hcli/cli.py:12\n"
            "- claim: overflow is reachable from parse_args\n"
            "$ python3 -m pytest hcli/tests/test_cli.py -q\n"
            "error: boom\n"
            + "".join(f"step {i} details go here\n" for i in range(120))
        )
        c = _compile(
            raw,
            extra={
                "status": {"state": "done", "exit_code": 0},
                "verifier_inputs": ["python3 -m pytest hcli/tests/test_cli.py -q"],
            },
        )
        blob = payload_dumps(c)
        self.assertLess(len(blob), len(raw))
        self.assertIn("buffer overflow", c["final_summary"] + json.dumps(c["claims"]))
        self.assertTrue(any("cli.py" in f for f in c["files_touched"]))
        self.assertTrue(c["commands_executed"] or c["verifier_inputs"])
        self.assertTrue(c["errors"])
        self.assertEqual(c["status"]["state"], "done")
        self.assertEqual(c["raw_report_path"], "/tmp/raw.md")
        self.assertFalse(c.get("passthrough"))

    def test_large_collapses_and_stays_bounded(self):
        signal = "FINDING: buffer overflow in hcli/cli.py:12\n"
        raw = signal + ("trace line\n" * 4000)  # ~44KB
        c = _compile(raw, extra={"status": {"state": "done", "exit_code": 0}})
        blob = payload_dumps(c)
        self.assertLess(len(blob), 8000)
        self.assertLess(len(blob), int(0.1 * len(raw)))
        self.assertIn("buffer overflow", c["final_summary"])
        self.assertEqual(c["raw_report_path"], "/tmp/raw.md")

    def test_canary_material_change_changes_digest(self):
        # One-line diffs of finding / status / verifier. Pad past _TINY so
        # the record is compiled (not passthrough): a 60-byte finding is
        # below that floor and would drop status from payload_dumps.
        pad = "".join(f"context {i} unique detail\n" for i in range(20))
        base = "FINDING: buffer overflow in hcli/cli.py:12\n" + pad
        extra = {
            "status": {"state": "done", "exit_code": 0},
            "verifier_inputs": ["pytest -q"],
        }
        d0 = payload_dumps(_compile(base, extra=extra))
        d_find = payload_dumps(_compile(
            "FINDING: use after free in hcli/cli.py:12\n" + pad,
            extra=extra,
        ))
        d_state = payload_dumps(_compile(
            base, extra={**extra, "status": {"state": "failed", "exit_code": 1}}
        ))
        d_ver = payload_dumps(_compile(
            base, extra={**extra, "verifier_inputs": ["pytest -q --failed"]}
        ))
        self.assertFalse(_compile(base, extra=extra).get("passthrough"))
        self.assertNotEqual(d_find, d0)
        self.assertNotEqual(d_state, d0)
        self.assertNotEqual(d_ver, d0)

    def test_canary_repeated_noise_does_not_balloon(self):
        signal = "FINDING: buffer overflow in hcli/cli.py:12\n"
        extra = {"status": {"state": "done", "exit_code": 0}}
        small = _compile(signal + ("trace line\n" * 50), extra=extra)
        large = _compile(signal + ("trace line\n" * 5000), extra=extra)
        self.assertEqual(payload_dumps(small), payload_dumps(large))
        self.assertLess(len(payload_dumps(large)), 8000)
        self.assertLess(len(payload_dumps(large)), int(0.1 * len(signal + "trace line\n" * 5000)))
        self.assertIn("buffer overflow", large["final_summary"])


class TestProvenanceAddressability(unittest.TestCase):
    def test_raw_report_path_reads_back_through_compact(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "grok-report.md"
            body = "FINDING: buffer overflow in hcli/cli.py:12\n"
            raw_path.write_text(body, encoding="utf-8")
            compact = compile_backend_report(
                backend="grok",
                task_id="addr",
                raw_text=body,
                raw_report_path=str(raw_path),
                extra={"status": {"state": "done", "exit_code": 0}},
            )
            self.assertEqual(compact["raw_report_path"], str(raw_path))
            self.assertEqual(
                Path(compact["raw_report_path"]).read_text(encoding="utf-8"),
                body,
            )
            self.assertNotIn("raw_report_path", json.loads(payload_dumps(compact)))
            self.assertNotIn("workspace", json.loads(payload_dumps(compact)))

    def test_compact_path_resolves_and_raw_is_readable(self):
        from hcli.grok_bridge import GrokBridge

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_id = "slug-20260101-000000"
            task_dir = root / "task"
            task_dir.mkdir()
            raw = task_dir / "grok-report.md"
            body = "FINDING: buffer overflow in hcli/cli.py:12\n"
            raw.write_text(body, encoding="utf-8")
            bridge = GrokBridge(root)
            bridge.receipts_dir.mkdir(parents=True, exist_ok=True)
            (bridge.receipts_dir / f"{task_id}.json").write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "mode": "consult",
                        "task_dir": str(task_dir),
                        "report_path": str(raw),
                        "workspace": str(root),
                        "status": {"state": "done", "exit_code": 0},
                    }
                ),
                encoding="utf-8",
            )
            compact = bridge.compact_report(task_id)
            self.assertEqual(compact["raw_report_path"], str(raw))
            self.assertEqual(
                Path(compact["raw_report_path"]).read_text(encoding="utf-8"),
                body,
            )
            dest = Path(compact["compact_path"])
            self.assertTrue(dest.is_file())
            self.assertEqual(dest, bridge.receipts_dir / f"{task_id}.compact.json")
            stored = json.loads(dest.read_text(encoding="utf-8"))
            self.assertEqual(stored["raw_report_path"], str(raw))
            receipt = json.loads(
                (bridge.receipts_dir / f"{task_id}.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["compact_path"], str(dest))

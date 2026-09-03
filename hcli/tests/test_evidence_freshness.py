from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]

from hcli.config import Config
from hcli.engine import Engine
from hcli.events import EventBus
from hcli.workspace import Workspace


OLD = "snapshot-AAAA"
NEW = "snapshot-BBBB"
assert len(OLD) == len(NEW)


def _engine(root: Path) -> Engine:
    cfg = Config(str(root), global_path=str(root / "global-config.json"))
    runtime = type("Runtime", (), {})()
    runtime.index = 0
    runtime.pid = 1
    runtime.port = 18765
    runtime.active = True
    pool = type("Pool", (), {})()
    pool.runtimes = [runtime]
    return Engine(
        workspace=Workspace(str(root)),
        event_bus=EventBus(),
        runtime_provider=lambda: pool,
        runtime_state_provider=lambda: pool,
        runtime_count=1,
        model_name="local",
        config=cfg,
    )


def _rewrite(path: Path, text: str) -> None:
    old_mtime = path.stat().st_mtime_ns if path.exists() else None
    path.write_text(text, encoding="utf-8")
    st = path.stat()
    if old_mtime is not None and st.st_mtime_ns == old_mtime:
        os.utime(path, ns=(st.st_atime_ns, old_mtime + 1))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _payload_user(payload: dict) -> str:
    for message in payload.get("messages") or []:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _answer_stub(captured=None):
    def stub(endpoint, payload, timeout):
        if captured is not None:
            captured["payload"] = payload
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "kind": "answer",
                                "content": "ok",
                                "operations": [],
                                "tests": [],
                            }
                        )
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 4,
            },
        }

    return stub


class TestEvidenceIdentity(unittest.TestCase):
    def test_gather_stamps_path_mtime_size_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = _engine(root)
            notes = root / "notes.txt"
            notes.write_text(OLD, encoding="utf-8")
            items = engine._gather_evidence("read notes.txt")
            self.assertEqual(len(items), 1)
            item = items[0]
            st = notes.stat()
            self.assertEqual(item["path"], "notes.txt")
            self.assertEqual(item["content"], OLD)
            self.assertEqual(item["mtime_ns"], st.st_mtime_ns)
            self.assertEqual(item["size"], st.st_size)
            self.assertEqual(item["sha256"], _sha(OLD))


class TestEvidenceFreshness(unittest.TestCase):
    def test_stale_file_is_reread_into_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = _engine(root)
            notes = root / "notes.txt"
            notes.write_text(OLD, encoding="utf-8")
            items = engine._gather_evidence("read notes.txt")
            self.assertEqual(items[0]["content"], OLD)
            _rewrite(notes, NEW)
            payload = engine._build_model_payload("read notes.txt", items)
            user = _payload_user(payload)
            self.assertIn(NEW, user)
            self.assertNotIn(OLD, user)
            self.assertEqual(items[0]["content"], NEW)
            self.assertEqual(items[0]["sha256"], _sha(NEW))
            self.assertEqual(items[0]["mtime_ns"], notes.stat().st_mtime_ns)
            self.assertEqual(items[0]["size"], notes.stat().st_size)

    def test_unchanged_file_is_not_reread(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = _engine(root)
            notes = root / "notes.txt"
            notes.write_text(OLD, encoding="utf-8")
            target = notes.resolve()
            real = Path.read_text
            reads = {"n": 0}

            def counted(self, *args, **kwargs):
                if Path(self).resolve() == target:
                    reads["n"] += 1
                return real(self, *args, **kwargs)

            with patch.object(Path, "read_text", counted):
                items = engine._gather_evidence("read notes.txt")
                after_gather = reads["n"]
                self.assertGreaterEqual(after_gather, 1)
                payload = engine._build_model_payload("read notes.txt", items)
                self.assertEqual(reads["n"], after_gather)
            self.assertIn(OLD, _payload_user(payload))

    def test_execute_reruns_stale_evidence_before_http(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = _engine(root)
            notes = root / "notes.txt"
            notes.write_text(OLD, encoding="utf-8")
            captured: dict = {}
            orig_call = engine._call_model

            def wrapped(prompt, evidence, compiled=None, **kwargs):
                _rewrite(notes, NEW)
                return orig_call(prompt, evidence, compiled, **kwargs)

            engine._call_model = wrapped
            engine._post_completion = _answer_stub(captured)
            result = engine.execute("read notes.txt")
            self.assertEqual(result["status"], "completed")
            user = _payload_user(captured["payload"])
            self.assertIn(NEW, user)
            self.assertNotIn(OLD, user)
            receipt = json.loads(
                Path(result["receipt"]).read_text(encoding="utf-8")
            )
            identity = receipt["context_efficiency"]["evidence_identity"]
            self.assertEqual(len(identity), 1)
            self.assertEqual(identity[0]["path"], "notes.txt")
            self.assertEqual(identity[0]["sha256"], _sha(NEW))
            self.assertEqual(identity[0]["mtime_ns"], notes.stat().st_mtime_ns)
            self.assertEqual(identity[0]["size"], notes.stat().st_size)


class TestContextEfficiencyReceipt(unittest.TestCase):
    def test_receipt_fields_come_from_observed_assembly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = _engine(root)
            notes = root / "notes.txt"
            notes.write_text(OLD, encoding="utf-8")
            captured: dict = {}
            engine._post_completion = _answer_stub(captured)
            result = engine.execute("read notes.txt and also ./notes.txt")
            self.assertEqual(result["status"], "completed")
            receipt = json.loads(
                Path(result["receipt"]).read_text(encoding="utf-8")
            )
            self.assertIn("context_efficiency", receipt)
            ce = receipt["context_efficiency"]
            payload = captured["payload"]
            observed = engine._estimate_prompt_tokens(payload["messages"])
            self.assertEqual(ce["root_tokens"], observed)
            self.assertEqual(ce["worker_tokens"], observed)
            self.assertEqual(ce["evidence_bytes_inlined"], len(OLD.encode("utf-8")))
            self.assertEqual(ce["bytes_re_read_stale"], 0)
            self.assertEqual(
                ce["duplicated_bytes_avoided"],
                len(OLD.encode("utf-8")),
            )
            self.assertEqual(len(ce["evidence_identity"]), 1)
            ident = ce["evidence_identity"][0]
            self.assertEqual(ident["path"], "notes.txt")
            self.assertEqual(ident["sha256"], _sha(OLD))
            self.assertEqual(ident["mtime_ns"], notes.stat().st_mtime_ns)
            self.assertEqual(ident["size"], notes.stat().st_size)
            for key in (
                "root_tokens",
                "worker_tokens",
                "evidence_bytes_inlined",
                "bytes_re_read_stale",
                "duplicated_bytes_avoided",
            ):
                self.assertIsInstance(ce[key], int, key)
                self.assertNotEqual(ce[key], "unknown", key)

    def test_stale_reread_bytes_are_observed_not_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = _engine(root)
            notes = root / "notes.txt"
            notes.write_text(OLD, encoding="utf-8")
            items = engine._gather_evidence("read notes.txt")
            _rewrite(notes, NEW)
            payload = engine._build_model_payload("read notes.txt", items)
            ce = (engine._last_call_plan or {}).get("context_efficiency")
            self.assertIsInstance(ce, dict)
            self.assertEqual(
                ce["bytes_re_read_stale"],
                len(NEW.encode("utf-8")),
            )
            self.assertEqual(ce["evidence_bytes_inlined"], len(NEW.encode("utf-8")))
            self.assertEqual(ce["evidence_identity"][0]["sha256"], _sha(NEW))
            self.assertEqual(
                ce["root_tokens"],
                engine._last_call_plan["prompt_tokens_est"],
            )
            self.assertIn(NEW, _payload_user(payload))


if __name__ == "__main__":
    unittest.main()

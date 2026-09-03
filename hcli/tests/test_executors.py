from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]

from hcli.executors import (
    BACKEND_CPU,
    BACKEND_GROK,
    BACKEND_QWEN,
    WorkUnitExecutor,
    select_backend_name,
)
from hcli.grok_bridge import GrokRunHandle
from hcli.resources import MutationLock
from hcli.workunit import WorkUnit


def _wu(uid, **kwargs):
    return WorkUnit(id=uid, role="work", description=uid, **kwargs)


class TestBackendSelection(unittest.TestCase):
    def test_preferred_backend_wins(self):
        wu = _wu("a", resource_class="GPU_DECODE", preferred_backend="grok")
        self.assertEqual(select_backend_name(wu), BACKEND_GROK)

    def test_default_is_qwen(self):
        self.assertEqual(select_backend_name(_wu("a")), BACKEND_QWEN)

    def test_cpu_class_needs_preferred_backend(self):
        self.assertEqual(
            select_backend_name(_wu("a", resource_class="CPU_HEAVY")),
            BACKEND_QWEN,
        )
        self.assertEqual(
            select_backend_name(
                _wu("a", resource_class="CPU_HEAVY", preferred_backend="cpu")
            ),
            BACKEND_CPU,
        )


class TestCpuExecutorVerifier(unittest.TestCase):
    def test_cpu_unit_requires_real_command_and_records_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            nonce = Path(tmp) / "nonce.txt"
            nonce.write_text("secret-nonce-value", encoding="utf-8")
            wu = _wu(
                "cpu1",
                resource_class="TEST",
                preferred_backend="cpu",
                verifier=f"test -f {nonce} && grep -q secret-nonce-value {nonce}",
            )
            ex = WorkUnitExecutor(tmp)
            raw = ex.execute(wu, {})
            self.assertEqual(raw["backend"], BACKEND_CPU)
            self.assertTrue(raw["validation"]["ok"])
            self.assertEqual(raw["validation"]["exit_code"], 0)

    def test_failed_verifier_does_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            wu = _wu(
                "cpu2",
                resource_class="TEST",
                preferred_backend="cpu",
                verifier="python3 -c 'raise SystemExit(7)'",
            )
            raw = WorkUnitExecutor(tmp).execute(wu, {})
            self.assertFalse(raw["validation"]["ok"])
            self.assertEqual(raw["validation"]["exit_code"], 7)

    def test_missing_verifier_is_not_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            wu = _wu("cpu3", resource_class="TEST", preferred_backend="cpu")
            raw = WorkUnitExecutor(tmp).execute(wu, {})
            self.assertFalse(raw["validation"]["ok"])
            self.assertEqual(
                raw["validation"]["reason"], "NO_DETERMINISTIC_VALIDATION"
            )


class TestGrokExecutorRequiresVerifier(unittest.TestCase):
    def test_grok_text_cannot_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            wu = _wu("g1", preferred_backend="grok")
            handle = GrokRunHandle(
                task_id="g1-task",
                command_run=["grok-run", "consult"],
                started_at="now",
                mode="consult",
            )

            class Fake:
                def consult(self, prompt, **kwargs):
                    return handle

                def wait(self, task_id, timeout=3600.0, poll_interval=0.5):
                    return {"state": "done", "exit_code": 0, "task_id": task_id}

                def compact_report(self, task_id):
                    return {
                        "backend": "grok",
                        "task_id": task_id,
                        "final_summary": "I declare this VERIFIED",
                        "claims": [],
                        "evidence_refs": [],
                        "files_touched": [],
                        "commands_executed": [],
                        "verifier_inputs": [],
                        "errors": [],
                        "raw_report_path": None,
                    }

            raw = WorkUnitExecutor(tmp, grok_bridge=Fake()).execute(wu, {})
            self.assertEqual(raw["backend"], "grok")
            self.assertEqual(wu.backend_task_id, "g1-task")
            self.assertFalse(raw["validation"]["ok"])
            self.assertEqual(raw["validation"]["reason"], "GROK_REQUIRES_VERIFIER")


class TestSingleWriter(unittest.TestCase):
    def test_second_acquire_fails_while_held(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = MutationLock(tmp)
            self.assertTrue(lock.acquire("writer-a"))
            self.assertFalse(lock.acquire("writer-b"))
            rec = lock.read()
            self.assertEqual(rec["unit_id"], "writer-a")
            lock.release("writer-a")
            self.assertTrue(lock.acquire("writer-b"))
            lock.release("writer-b")


if __name__ == "__main__":
    unittest.main()

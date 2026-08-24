from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

from hcli.dag_store import DagCorruptError, DagStore, atomic_write_json
from hcli.scheduler import Scheduler
from hcli.workunit import WorkUnit, is_ready


def _wu(uid: str, **kwargs) -> WorkUnit:
    return WorkUnit(id=uid, role="work", description=uid, **kwargs)


class TestDagStore(unittest.TestCase):
    def test_atomic_write_uses_temp_and_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "dag.json"
            calls = []
            real = os.replace

            def spy(src, dst, *args, **kwargs):
                calls.append((str(src), str(dst)))
                return real(src, dst, *args, **kwargs)

            os.replace = spy
            try:
                atomic_write_json(target, {"ok": True})
            finally:
                os.replace = real
            self.assertTrue(calls)
            src, dst = calls[-1]
            self.assertIn(".tmp", Path(src).name)
            self.assertEqual(Path(dst), target)
            self.assertTrue(target.is_file())
            leftover = list(Path(tmp).glob("*.tmp"))
            self.assertEqual(leftover, [])

    def test_corrupted_dag_is_not_loaded_as_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DagStore(tmp)
            store.save({"a": _wu("a")})
            store.path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(DagCorruptError):
                store.load()
            store.path.write_text(json.dumps({"version": 1}), encoding="utf-8")
            with self.assertRaises(DagCorruptError):
                store.load()

    def test_running_unit_recovers_as_interrupted_not_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            units = {
                "a": _wu("a"),
                "b": _wu("b", dependencies=["a"]),
                "c": _wu("c"),
            }
            sched = Scheduler(units, runtime_count=2, workspace=tmp)
            sched.dispatch()
            self.assertEqual(units["a"].status, "running")
            self.assertEqual(units["c"].status, "running")
            sched.complete("a", verification={"ok": True})
            self.assertEqual(units["c"].status, "running")
            crashed_attempts = units["c"].attempts
            restarted = Scheduler.from_workspace(tmp, runtime_count=2)
            self.assertEqual(restarted.units["a"].status, "completed")
            self.assertIn(restarted.units["c"].status, ("interrupted", "ready"))
            self.assertNotEqual(restarted.units["c"].status, "completed")
            self.assertNotEqual(restarted.units["c"].status, "failed")
            self.assertEqual(restarted.units["c"].attempts, crashed_attempts)
            self.assertGreaterEqual(restarted.units["c"].attempts, 1)
            classification = getattr(restarted.units["c"], "classification", None)
            if classification is None and isinstance(
                restarted.units["c"].failure_context, dict
            ):
                classification = restarted.units["c"].failure_context.get(
                    "classification"
                )
            self.assertEqual(classification, "INTERRUPTED")

    def test_missing_dag_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                DagStore(tmp).load()


def _running_grok_unit(uid: str = "g1", task_id: str = "consult-20260101-000000") -> WorkUnit:
    wu = _wu(uid, status="running", attempts=1, assigned_runtime=0)
    wu.assigned_backend = "grok"
    wu.preferred_backend = "grok"
    wu.backend_task_id = task_id
    return wu


class TestGrokRestartAdoption(unittest.TestCase):
    """Live Grok units must be adopted on load, not failed-and-re-dispatched."""

    def _roundtrip(self, tmp: str, wu: WorkUnit) -> DagStore:
        store = DagStore(tmp)
        store.save({wu.id: wu})
        return store

    def test_live_grok_unit_is_adopted_not_redispatched(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_id = "consult-20260101-111111"
            wu = _running_grok_unit(task_id=task_id)
            store = self._roundtrip(tmp, wu)
            polls = []

            def liveness(tid: str):
                polls.append(tid)
                return {
                    "state": "running",
                    "exit_code": None,
                    "process_alive": True,
                    "task_id": tid,
                }

            units = store.load(recover_running=True, grok_liveness=liveness)
            self.assertEqual(units["g1"].status, "running")
            self.assertEqual(units["g1"].assigned_runtime, 0)
            self.assertEqual(getattr(units["g1"], "backend_task_id", None), task_id)
            adopted = getattr(store, "adopted_running", None)
            self.assertTrue(adopted, "live Grok unit must be recorded for polling")
            self.assertEqual(adopted[0]["backend_task_id"], task_id)
            self.assertEqual(polls, [task_id])

            sched = Scheduler(units, runtime_count=2, workspace=tmp)
            assignments = sched.dispatch()
            assigned_ids = [unit.id for unit, _slot in assignments]
            self.assertNotIn(
                "g1",
                assigned_ids,
                "adopting a live Grok unit must not launch a second dispatch",
            )
            self.assertEqual(sched.units["g1"].status, "running")
            self.assertEqual(polls, [task_id], "load checks liveness once; dispatch must not")

    def test_stale_running_is_interrupted_and_rerun_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_id = "consult-20260101-222222"
            wu = _running_grok_unit(task_id=task_id)
            store = self._roundtrip(tmp, wu)

            def liveness(tid: str):
                return {
                    "state": "stale-running",
                    "exit_code": None,
                    "process_alive": False,
                    "launch_pid": 9,
                    "stale_reason": "status file says running but launch pid 9 is gone",
                    "task_id": tid,
                }

            units = store.load(recover_running=True, grok_liveness=liveness)
            self.assertEqual(units["g1"].status, "interrupted")
            self.assertIsNone(units["g1"].assigned_runtime)
            self.assertFalse(getattr(store, "adopted_running", None))
            self.assertEqual(units["g1"].attempts, 1)
            self.assertTrue(
                is_ready(units["g1"], units),
                "stale-running is process death: interrupt and re-run, not fail",
            )

    def test_unobservable_liveness_is_interrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            wu = _running_grok_unit()
            store = self._roundtrip(tmp, wu)
            units = store.load(recover_running=True)
            self.assertEqual(units["g1"].status, "interrupted")
            self.assertIsNone(units["g1"].assigned_runtime)
            self.assertFalse(getattr(store, "adopted_running", None))

    def test_done_nonzero_grok_unit_is_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            # consult-20260822-223811 shape: grok-run status=done with exit_code=1
            wu = _running_grok_unit(task_id="consult-20260822-223811")
            store = self._roundtrip(tmp, wu)

            def liveness(tid: str):
                return {
                    "state": "failed",
                    "exit_code": 1,
                    "successful": False,
                    "task_id": tid,
                }

            units = store.load(recover_running=True, grok_liveness=liveness)
            self.assertEqual(units["g1"].status, "failed")
            self.assertFalse(getattr(store, "adopted_running", None))

    def test_non_grok_running_unit_recovers_as_interrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            wu = _wu("cpu", status="running", attempts=1, assigned_runtime=0)
            store = DagStore(tmp)
            store.save({"cpu": wu})

            def liveness(tid: str):
                raise AssertionError(f"liveness must not be consulted for a non-Grok unit: {tid}")

            units = store.load(recover_running=True, grok_liveness=liveness)
            self.assertEqual(units["cpu"].status, "interrupted")
            self.assertEqual(units["cpu"].attempts, 1)


if __name__ == "__main__":
    unittest.main()

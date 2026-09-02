from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli.max_policy import (
    grok_pool_snapshot,
    load_equilibrium,
    next_grok_rung,
    record_rung,
    run_grok_ramp,
)
from hcli.workunit import WorkUnit


def _wu(uid, **kwargs):
    return WorkUnit(id=uid, role="work", description=uid, **kwargs)


class TestGrokPoolSnapshot(unittest.TestCase):
    def test_live_running_is_active_stale_running_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "live-task-abc"
            stale = root / "stale-task-xyz"
            live.mkdir()
            stale.mkdir()
            (live / "status").write_text("running\n", encoding="utf-8")
            (stale / "status").write_text("running\n", encoding="utf-8")
            (live / "metadata.json").write_text(
                json.dumps({"started_at": "2026-08-22T00:00:00Z"}),
                encoding="utf-8",
            )
            snap = grok_pool_snapshot(
                tmp,
                tasks_root=root,
                is_live=lambda tid: tid == "live-task-abc",
            )
            self.assertEqual(snap["active"], 1)
            self.assertGreaterEqual(snap["failed"], 1)
            self.assertGreaterEqual(snap["stale"], 1)
            self.assertIsNotNone(snap["latency_s"])

    def test_active_is_not_taken_from_workunit_flags(self):
        class FakeWU:
            status = "running"
            resource_class = "GROK"
            role = "grok"
            backend_task_id = "wu-intent"

        class FakeSched:
            units = {"g": FakeWU()}

        class FakeMission:
            scheduler = FakeSched()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.mkdir(exist_ok=True)
            snap = grok_pool_snapshot(
                tmp,
                mission=FakeMission(),
                tasks_root=root,
                is_live=lambda tid: False,
            )
            self.assertEqual(snap["active"], 0)
            self.assertEqual(snap.get("wu_active"), 1)

    def test_done_and_failed_status_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, state in (("d1", "done"), ("f1", "failed"), ("q1", "queued")):
                d = root / name
                d.mkdir()
                (d / "status").write_text(state + "\n", encoding="utf-8")
            snap = grok_pool_snapshot(tmp, tasks_root=root, is_live=lambda tid: False)
            self.assertEqual(snap["done"], 1)
            self.assertEqual(snap["failed"], 1)
            self.assertEqual(snap["queued"], 1)
            self.assertEqual(snap["active"], 0)


class TestRecordRung(unittest.TestCase):
    def test_two_synthetic_rungs_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            u1 = _wu("a", status="completed", attempts=2, ready_at=1.0, running_at=2.0, finished_at=5.0)
            u2 = _wu("b", status="failed", attempts=1, ready_at=1.0, running_at=3.0, finished_at=4.0)
            record_rung(
                tmp,
                requested=1,
                admitted=1,
                actual=1,
                units={"a": u1, "b": u2},
                elapsed_s=3600,
                extra={"scheduler_overhead_s": 0.01, "throttle": None},
            )
            record_rung(
                tmp,
                requested=2,
                admitted=2,
                actual=2,
                units={"a": u1, "b": u2},
                elapsed_s=1800,
                extra={"rejected": 1, "verifier_wait_s": None},
            )
            payload = load_equilibrium(tmp)
            self.assertEqual(payload["source"], "rung-measure")
            self.assertEqual(len(payload["rungs"]), 2)
            self.assertEqual(payload["rungs"][0]["requested"], 1)
            self.assertEqual(payload["rungs"][1]["admitted"], 2)
            self.assertEqual(payload["rungs"][0]["verified"], 1)
            self.assertEqual(payload["rungs"][0]["failures"], 1)
            self.assertEqual(payload["rungs"][0]["retries"], 1)
            self.assertEqual(payload["rungs"][0]["queue_latency_s"], 2.0)
            self.assertEqual(payload["rungs"][0]["completion_latency_p50_s"], 1.0)
            self.assertEqual(payload["rungs"][0]["completion_latency_p95_s"], 3.0)
            self.assertEqual(payload["grok_admitted"], 2)
            self.assertEqual(payload["rungs"][1]["rejected"], 1)
            self.assertNotIn("qwen_resident", payload["rungs"][0])
            self.assertNotIn("cpu_validators", payload["rungs"][0])

    def test_run_grok_ramp_stops_when_rate_does_not_beat(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = os.environ.get("HCLI_GROK_ADMITTED")

            def run_rung(admitted):
                rate = {1: 10.0, 2: 20.0, 4: 15.0, 6: 12.0, 8: 8.0}[admitted]
                wu = _wu("x", status="completed", attempts=1)
                return {
                    "actual_active": admitted,
                    "units": {"x": wu},
                    "elapsed_s": 3600,
                    "verified_units_per_hour": rate,
                }

            try:
                payload = run_grok_ramp(tmp, run_rung, start=1)
            finally:
                if saved is None:
                    os.environ.pop("HCLI_GROK_ADMITTED", None)
                else:
                    os.environ["HCLI_GROK_ADMITTED"] = saved
            self.assertEqual(payload["grok_admitted"], 2)
            self.assertIn("did not beat", payload.get("stop_reason", ""))
            self.assertGreaterEqual(len(payload.get("rungs") or []), 3)
            self.assertEqual(next_grok_rung(8), None)
            self.assertEqual(next_grok_rung(1), 2)


if __name__ == "__main__":
    unittest.main()

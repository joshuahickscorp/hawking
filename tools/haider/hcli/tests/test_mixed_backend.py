from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

from hcli.controller import Controller
from hcli.mission import Mission
from hcli.resources import MutationLock
from hcli.workunit import WorkUnit, transition_status


def _wu(uid, **kwargs):
    return WorkUnit(id=uid, role="work", description=uid, **kwargs)


class StubEngine:
    def __init__(self):
        self.ran = []

    def execute_workunit(self, unit, context):
        self.ran.append(unit.id)
        return {"validation": {"ok": True}, "backend": "qwen"}

    def cancel(self):
        pass


class TestMixedBackendMission(unittest.TestCase):
    def test_cpu_verifier_completes_and_failed_verifier_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            nonce = Path(tmp) / "n.txt"
            nonce.write_text("abc", encoding="utf-8")
            units = {
                "ok": _wu(
                    "ok",
                    preferred_backend="cpu",
                    resource_class="TEST",
                    verifier=f"grep -q abc {nonce}",
                ),
                "bad": _wu(
                    "bad",
                    preferred_backend="cpu",
                    resource_class="TEST",
                    verifier="python3 -c 'raise SystemExit(3)'",
                ),
            }
            mission = Mission(
                tmp,
                engine=StubEngine(),
                units=units,
                quiet=True,
                no_progress_threshold=100,
            )
            result = mission.run()
            self.assertEqual(mission.scheduler.units["ok"].status, "completed")
            self.assertEqual(mission.scheduler.units["ok"].assigned_backend, "cpu")
            self.assertEqual(mission.scheduler.units["bad"].status, "failed")
            # A repair is emitted; the original cannot be completed by the fail.
            repairs = [
                u
                for u in mission.scheduler.units.values()
                if u.repairs == "bad"
            ]
            self.assertTrue(repairs)
            self.assertEqual(mission.scheduler.units["ok"].status, "completed")
            self.assertNotEqual(mission.scheduler.units["bad"].status, "completed")

    def test_qwen_path_still_uses_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = StubEngine()
            units = {"a": _wu("a"), "b": _wu("b")}
            Mission(
                tmp, engine=engine, units=units, quiet=True, no_progress_threshold=100
            ).run()
            self.assertEqual(set(engine.ran), {"a", "b"})
            self.assertEqual(engine.ran.count("a"), 1)

    def test_restart_keeps_ids_and_does_not_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            units = {
                "done": _wu("done"),
                "next": _wu("next", dependencies=["done"]),
            }
            engine = StubEngine()
            m1 = Mission(
                tmp,
                engine=engine,
                units=units,
                goal="durable mission",
                quiet=True,
                no_progress_threshold=100,
                mission_id="fixed-mission-id",
            )
            wu = m1.scheduler.units["done"]
            transition_status(wu, "ready")
            transition_status(wu, "running")
            # complete() now requires a passing verifier outcome. This test is
            # about restart and replay, not about completing unverified work,
            # so it supplies one rather than dropping the new gate.
            m1.scheduler.complete(
                "done", verification={"ok": True, "verifier": "test-fixture"}
            )
            m1.checkpoint()
            self.assertEqual(m1.scheduler.units["done"].status, "completed")
            m2 = Mission.from_workspace(
                tmp, engine=engine, quiet=True, runtime_count=1
            )
            self.assertEqual(m2.id, "fixed-mission-id")
            self.assertEqual(m2.scheduler.units["done"].status, "completed")
            self.assertIn("next", m2.scheduler.units)
            m2.run()
            self.assertEqual(engine.ran.count("done"), 0)
            self.assertIn("next", engine.ran)
            self.assertEqual(m2.scheduler.units["next"].status, "completed")

    def test_steer_does_not_fork_mission(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = Controller(workspace=tmp, runtime_count=1)
            first = ctrl.start_ultragoal("same mission after steer")
            mid = first["mission_id"]
            ctrl.queue_steer("constraint: add: extra future work")
            self.assertEqual(ctrl.mission.id, mid)
            pending = [
                u
                for u in ctrl.mission.scheduler.units.values()
                if u.status != "completed"
            ]
            self.assertTrue(any("steer" in u.description for u in pending))
            ctrl.handle_command("/steer correction: do not rewrite verified history")
            self.assertEqual(ctrl.mission.id, mid)

    def test_resume_restores_same_mission(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = Controller(workspace=tmp, runtime_count=1)
            created = ctrl.start_ultragoal("resume me")
            mid = created["mission_id"]
            ctrl.session_store.save(ctrl.session)
            sid = ctrl.session.id
            other = Controller(workspace=tmp, runtime_count=1)
            restored = other.handle_command(f"/resume {sid}")
            self.assertEqual(other.mission.id, mid)
            self.assertIn(mid, str(restored) + str(other.session.mission_id))


class TestMutationSerialized(unittest.TestCase):
    def test_two_writers_cannot_hold_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = MutationLock(tmp)
            b = MutationLock(tmp)
            self.assertTrue(a.acquire("wu-a"))
            self.assertFalse(b.acquire("wu-b"))
            a.release("wu-a")
            self.assertTrue(b.acquire("wu-b"))


if __name__ == "__main__":
    unittest.main()

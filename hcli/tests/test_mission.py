from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli.controller import Controller
from hcli.mission import Mission, MissionCorruptError
from hcli.steering import SteeringQueue
from hcli.workunit import WorkUnit
from hcli.workspace import Workspace


def _wu(uid, resource_class="LIGHT_CONTROL", **kwargs):
    return WorkUnit(
        id=uid,
        role=kwargs.pop("role", "work"),
        description=kwargs.pop("description", uid),
        resource_class=resource_class,
        **kwargs,
    )


class StubEngine:
    def __init__(self, workspace, delays=None, results=None, gates=None):
        self.workspace = Path(workspace)
        self.delays = dict(delays or {})
        self.results = dict(results or {})
        self.gates = dict(gates or {})
        self.lock = threading.Lock()
        self.events = []
        self.contexts = {}
        self.ran = []
        self.gpu_inflight = 0
        self.max_gpu = 0
        self.cpu_during_saturated_gpu = False
        self.cancelled = False
        self.prompts = []

    def cancel(self):
        self.cancelled = True
        for gate in self.gates.values():
            try:
                gate.set()
            except Exception:
                pass

    def execute_workunit(self, unit, context):
        uid = unit.id
        rc = getattr(unit, "resource_class", "LIGHT_CONTROL")
        with self.lock:
            self.events.append(("start", uid, time.perf_counter()))
            self.ran.append(uid)
            self.contexts[uid] = dict(context or {})
            self.prompts.append((uid, (context or {}).get("prompt")))
            if rc == "GPU_DECODE":
                self.gpu_inflight += 1
                self.max_gpu = max(self.max_gpu, self.gpu_inflight)
            elif self.gpu_inflight >= 2:
                self.cpu_during_saturated_gpu = True
        try:
            gate = self.gates.get(uid)
            if gate is not None:
                gate.wait(timeout=15)
            delay = float(self.delays.get(uid, 0.0))
            deadline = time.perf_counter() + delay
            while time.perf_counter() < deadline:
                if self.cancelled:
                    return {"cancelled": True}
                checker = (context or {}).get("is_cancelled")
                if callable(checker) and checker():
                    return {"cancelled": True}
                time.sleep(0.01)
            canned = self.results.get(uid)
            if canned is not None:
                out = dict(canned)
            else:
                (self.workspace / f"accepted_{uid}.txt").write_text(uid, encoding="utf-8")
                out = {"validation": {"ok": True}}
            val = out.get("validation")
            if isinstance(val, dict) and val.get("ok") is True:
                unit.verification = dict(val)
            return out
        finally:
            with self.lock:
                self.events.append(("finish", uid, time.perf_counter()))
                if rc == "GPU_DECODE":
                    self.gpu_inflight = max(0, self.gpu_inflight - 1)


def _event_time(events, kind, uid):
    for k, i, t in events:
        if k == kind and i == uid:
            return t
    return None


class TestMissionLoop(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("ACTIVE_DECODE_LIMIT")
        os.environ["ACTIVE_DECODE_LIMIT"] = "2"

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("ACTIVE_DECODE_LIMIT", None)
        else:
            os.environ["ACTIVE_DECODE_LIMIT"] = self._saved

    def test_no_barrier_across_independent_chains(self):
        with tempfile.TemporaryDirectory() as tmp:
            units = {
                "A1": _wu("A1"),
                "A2": _wu("A2", dependencies=["A1"]),
                "B1": _wu("B1"),
                "B2": _wu("B2", dependencies=["B1"]),
            }
            engine = StubEngine(tmp, delays={"B1": 0.3, "A1": 0.02, "A2": 0.02})
            mission = Mission(
                tmp, engine=engine, units=units, quiet=True, no_progress_threshold=100
            )
            mission.run()
            a2s = _event_time(engine.events, "start", "A2")
            b1f = _event_time(engine.events, "finish", "B1")
            self.assertIsNotNone(a2s)
            self.assertIsNotNone(b1f)
            self.assertLess(a2s, b1f)

    def test_decode_limit_and_cpu_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            units = {
                "g0": _wu("g0", "GPU_DECODE"),
                "g1": _wu("g1", "GPU_DECODE"),
                "g2": _wu("g2", "GPU_DECODE"),
                "c0": _wu("c0", "COMPILE"),
            }
            engine = StubEngine(
                tmp, delays={"g0": 0.18, "g1": 0.18, "g2": 0.18, "c0": 0.04}
            )
            mission = Mission(
                tmp,
                engine=engine,
                units=units,
                runtime_count=4,
                quiet=True,
                no_progress_threshold=100,
            )
            mission.run()
            self.assertEqual(engine.max_gpu, 2)

            def gpu_inflight_at(t):
                n = 0
                for kind, uid, ts in engine.events:
                    if not str(uid).startswith("g"):
                        continue
                    if kind == "start" and ts <= t:
                        n += 1
                    elif kind == "finish" and ts <= t:
                        n -= 1
                return n

            cpu_starts = [
                t
                for kind, uid, t in engine.events
                if kind == "start" and uid == "c0"
            ]
            self.assertTrue(any(gpu_inflight_at(t) >= 1 for t in cpu_starts), engine.events)

    def test_context_is_bounded_per_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            units = {"a": _wu("a"), "b": _wu("b", dependencies=["a"])}
            engine = StubEngine(tmp)
            mission = Mission(
                tmp, engine=engine, units=units, quiet=True, no_progress_threshold=100
            )
            mission.run()
            self.assertFalse(getattr(mission, "messages", None))
            a_ctx = engine.contexts["a"]
            b_ctx = engine.contexts["b"]
            self.assertNotIn("a-model-reply", json.dumps(b_ctx, default=str))
            self.assertEqual(a_ctx.get("unit_id"), "a")
            self.assertEqual(b_ctx.get("unit_id"), "b")
            self.assertNotEqual(a_ctx, b_ctx)

    def test_shared_runtime_pool_survives_mission_cleanup(self):
        class SharedPool:
            def __init__(self):
                self.runtimes = []
                self.stop_calls = 0

            def stop(self):
                self.stop_calls += 1

        with tempfile.TemporaryDirectory() as tmp:
            pool = SharedPool()
            mission = Mission(
                tmp,
                engine=StubEngine(tmp),
                units={"a": _wu("a")},
                runtime_pool=pool,
                stop_runtime_pool=False,
                quiet=True,
                no_progress_threshold=100,
            )
            mission.run()
            self.assertEqual(pool.stop_calls, 0)

    def test_repair_not_repetition(self):
        with tempfile.TemporaryDirectory() as tmp:
            units = {"orig": _wu("orig")}
            engine = StubEngine(
                tmp,
                results={
                    "orig": {
                        "validation": {"ok": False, "reason": "boom"},
                        "error": "boom",
                    }
                },
            )
            mission = Mission(
                tmp, engine=engine, units=units, quiet=True, no_progress_threshold=100
            )
            mission.run()
            orig = mission.scheduler.units["orig"]
            self.assertEqual(orig.status, "failed")
            repairs = [
                u for u in mission.scheduler.units.values() if getattr(u, "repairs", None) == "orig"
            ]
            self.assertTrue(repairs)
            self.assertEqual(engine.ran.count("orig"), 1)
            self.assertIn("boom", str(repairs[0].failure_context))

    def test_no_progress_halts_then_stays_quiet_on_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            units = {f"u{i}": _wu(f"u{i}") for i in range(3)}
            engine = StubEngine(tmp)
            mission = Mission(
                tmp,
                engine=engine,
                units=units,
                quiet=True,
                no_progress_threshold=3,
                fingerprint_fn=lambda: "same",
            )
            result = mission.run()
            self.assertTrue(mission.status().get("no_progress_warning"))
            self.assertTrue(
                mission.strategy != "default"
                or mission.phase in ("no_progress", "failed")
                or (result or {}).get("reason") == "no_progress"
            )

        with tempfile.TemporaryDirectory() as tmp:
            units = {f"v{i}": _wu(f"v{i}") for i in range(3)}
            engine = StubEngine(tmp)
            n = {"i": 0}

            def moving():
                n["i"] += 1
                return f"fp-{n['i']}"

            quiet = Mission(
                tmp,
                engine=engine,
                units=units,
                quiet=True,
                no_progress_threshold=3,
                fingerprint_fn=moving,
            )
            quiet.run()
            self.assertFalse(quiet.status().get("no_progress_warning"))
            self.assertNotEqual(quiet.phase, "no_progress")

    def test_model_claim_is_not_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            units = {"claim": _wu("claim")}
            engine = StubEngine(
                tmp,
                results={
                    "claim": {
                        "status": "completed",
                        "kind": "mutation",
                        "content": "done",
                        "operations": [],
                    }
                },
            )
            mission = Mission(
                tmp, engine=engine, units=units, quiet=True, no_progress_threshold=100
            )
            mission.run()
            self.assertNotEqual(mission.scheduler.units["claim"].status, "completed")

    def test_atomic_write_and_truncated_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            units = {"a": _wu("a")}
            engine = StubEngine(tmp)
            mission = Mission(
                tmp, engine=engine, units=units, quiet=True, no_progress_threshold=100
            )
            calls = []
            real = os.replace

            def spy(src, dst, *args, **kwargs):
                calls.append((str(src), str(dst)))
                return real(src, dst, *args, **kwargs)

            os.replace = spy
            try:
                mission.checkpoint()
            finally:
                os.replace = real
            state = Path(tmp) / ".hcli" / "mission" / "state.json"
            self.assertTrue(state.is_file())
            self.assertTrue(any(".tmp" in Path(src).name for src, _dst in calls))
            state.write_text("{not json", encoding="utf-8")
            with self.assertRaises(MissionCorruptError):
                Mission.from_workspace(tmp, engine=StubEngine(tmp), quiet=True)

    def test_cancel_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = threading.Event()
            units = {"slow": _wu("slow")}
            engine = StubEngine(tmp, delays={"slow": 8.0}, gates={"slow": gate})
            mission = Mission(
                tmp, engine=engine, units=units, quiet=True, no_progress_threshold=100
            )
            thread = threading.Thread(target=mission.run, daemon=True)
            thread.start()
            deadline = time.time() + 5
            while time.time() < deadline and "slow" not in engine.ran:
                time.sleep(0.02)
            mission.cancel("test-cancel")
            gate.set()
            thread.join(timeout=5)
            self.assertTrue(thread.is_alive() is False)
            self.assertEqual(mission.phase, "cancelled")
            self.assertTrue((Path(tmp) / ".hcli" / "mission" / "state.json").is_file())

    def test_steering_is_forward_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            units = {"A": _wu("A"), "B": _wu("B", dependencies=["A"])}
            engine = StubEngine(tmp)
            steered = {"done": False}

            def before_dispatch(mission):
                if steered["done"]:
                    return
                if mission.scheduler.units["A"].status == "completed":
                    mission.append_steer("use fewer comments", kind="correction")
                    steered["done"] = True

            mission = Mission(
                tmp,
                engine=engine,
                units=units,
                quiet=True,
                no_progress_threshold=100,
                session_id="s1",
                before_dispatch=before_dispatch,
            )
            mission.run()
            self.assertTrue(steered["done"])
            self.assertEqual(mission.scheduler.units["A"].status, "completed")
            self.assertIn(
                "use fewer comments",
                json.dumps(engine.contexts.get("B") or {}, default=str),
            )

    def test_status_fields_numeric(self):
        with tempfile.TemporaryDirectory() as tmp:
            units = {"a": _wu("a")}
            mission = Mission(
                tmp,
                engine=StubEngine(tmp),
                units=units,
                quiet=True,
                no_progress_threshold=100,
            )
            mission.run()
            snap = mission.status()
            for key in (
                "mission_id",
                "phase",
                "units_by_status",
                "active_runtimes",
                "active_decodes",
                "accepted_units_per_hour",
                "elapsed_wall",
                "last_checkpoint",
                "no_progress_warning",
            ):
                self.assertIn(key, snap)
            self.assertIsInstance(snap["units_by_status"], dict)
            self.assertIsInstance(snap["active_runtimes"], (int, float))
            self.assertIsInstance(snap["active_decodes"], (int, float))
            self.assertIsInstance(snap["accepted_units_per_hour"], (int, float))
            self.assertIsInstance(snap["elapsed_wall"], (int, float))

    def test_controller_status_pools_are_probed_not_wu_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            ws.mkdir()
            tasks = Path(tmp) / "tasks"
            live = tasks / "live-grok-1"
            stale = tasks / "stale-grok-1"
            live.mkdir(parents=True)
            stale.mkdir()
            (live / "status").write_text("running\n", encoding="utf-8")
            (stale / "status").write_text("running\n", encoding="utf-8")
            controller = Controller(workspace=str(ws), runtime_count=1)
            units = {"a": _wu("a"), "b": _wu("b")}
            units["a"].status = "running"
            units["b"].status = "running"
            mission = Mission(
                ws,
                engine=StubEngine(ws),
                units=units,
                quiet=True,
                no_progress_threshold=100,
                session_id=controller.session.id,
            )
            controller.mission = mission
            from hcli import max_policy as mp
            from unittest.mock import patch

            def isolated(workspace, mission=None, **kwargs):
                return mp.grok_pool_snapshot(
                    workspace,
                    mission=mission,
                    tasks_root=tasks,
                    is_live=lambda tid: tid == "live-grok-1",
                )

            with patch("hcli.controller.grok_pool_snapshot", isolated):
                snap = controller.status()
                rendered = controller.handle_command("/status")
            self.assertIn("qwen", snap)
            self.assertIn("grok", snap)
            self.assertIn("mutation", snap)
            self.assertIn("occupancy", snap)
            self.assertIn("blocked_units", snap)
            self.assertIn("checkpoint_age_s", snap)
            self.assertIn("verifier_backlog", snap)
            self.assertIn("watchdog", snap)
            self.assertEqual(snap["grok"]["active"], 1)
            self.assertGreaterEqual(snap["grok"]["failed"], 1)
            self.assertEqual(snap["occupancy"].get("LIGHT_CONTROL"), 2)
            self.assertNotEqual(snap["grok"]["active"], snap["occupancy"].get("LIGHT_CONTROL"))
            self.assertNotIn("failed_restarting", snap)
            self.assertNotIn("failed_restarting", snap["qwen"])
            self.assertIsInstance(rendered, dict)
            controller.shutdown()

    def test_controller_status_and_steer_wire_to_mission(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            ws.mkdir()
            controller = Controller(workspace=str(ws), runtime_count=1)
            units = {"a": _wu("a")}
            mission = Mission(
                ws,
                engine=StubEngine(ws),
                units=units,
                quiet=True,
                no_progress_threshold=100,
                session_id=controller.session.id,
            )
            controller.mission = mission
            snap = controller.status()
            self.assertIn("mission_id", snap)
            self.assertIn("phase", snap)
            self.assertIn("units_by_status", snap)
            event = controller.queue_steer("mid-flight note")
            self.assertTrue(event.text)
            pending = SteeringQueue(str(ws), controller.session.id).pending()
            self.assertTrue(any("mid-flight note" in e.text for e in pending))
            handled = controller.handle_command("/status")
            self.assertIsInstance(handled, dict)
            self.assertIn("mission_id", handled)
            controller.shutdown()


if __name__ == "__main__":
    unittest.main()

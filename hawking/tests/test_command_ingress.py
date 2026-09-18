from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]

from hawking.app import App
from hawking.cli import parse_hawking_args
from hawking.commands import REQUIRED_COMMANDS, CommandHandler
from hawking.controller import Controller
from hawking.events import EventBus
from hawking.models import ModelRegistry
from hawking.tui import TUI


class TestLiveIngressUnification(unittest.TestCase):
    def test_public_model_flag_refuses_unadmitted_legacy_shape_before_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "legacy-mlx"
            legacy.mkdir()
            (legacy / "config.json").write_text('{"model_type": "qwen"}', encoding="utf-8")
            (legacy / "model.safetensors").write_bytes(b"not-loaded")
            with self.assertRaises(SystemExit) as refused:
                parse_hawking_args(["--model", str(legacy), "hello"])
            self.assertEqual(refused.exception.code, 2)

    def test_unbound_app_stays_control_only_without_model_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(ModelRegistry, "resolve", side_effect=AssertionError("legacy discovery")):
                app = App(workspace=tmp, runtime_count=1)
            with self.assertRaisesRegex(RuntimeError, "admitted native artifact"):
                app.controller.ensure_runtime_pool()
            self.assertIsNone(app.controller.runtime_pool)
            app.controller.shutdown()

    def test_app_and_controller_share_one_dispatcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = App(workspace=tmp, runtime_count=1)
            ctrl = app.controller
            self.assertIs(ctrl.dispatcher(), ctrl.dispatcher())
            self.assertIsInstance(ctrl.dispatcher(), CommandHandler)
            self.assertEqual(app._handle_input("/help"), ctrl.handle_command("/help"))

    def test_required_commands_are_wired_on_tui_ingress(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = Controller(workspace=tmp, runtime_count=1)
            handler = ctrl.dispatcher()
            for cmd in REQUIRED_COMMANDS:
                meth = getattr(handler, f"_cmd_{cmd[1:]}", None)
                self.assertTrue(callable(meth), cmd)
                if cmd == "/grok":
                    continue
                if cmd == "/land":
                    # WIRING, not execution. `/land` re-verifies and re-runs the
                    # project's own test command, so asking "is it wired?" ran
                    # the whole suite inside one test: 300.55 s of a 433 s run,
                    # 70% of everything, for a dispatch check.
                    #
                    # Patching the commit path proves MORE than the old call
                    # did -- that dispatch actually reaches `_land_commit` --
                    # without performing a landing.
                    with patch.object(
                        CommandHandler, "_land_commit", return_value="LANDED"
                    ) as landed:
                        result = ctrl.handle_command(cmd)
                    self.assertTrue(landed.called, "/land did not reach _land_commit")
                    self.assertEqual(result, "LANDED")
                    continue
                result = ctrl.handle_command(cmd)
                if cmd == "/exit":
                    self.assertIs(result, False)
                    continue
                self.assertIsNotNone(result, cmd)
                if isinstance(result, str):
                    self.assertFalse(
                        result.startswith("Unknown command"),
                        f"{cmd} -> {result!r}",
                    )
                    self.assertNotIn("not connected yet", result)

    def test_tui_path_grok_is_retired_into_hawking(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = App(workspace=tmp, runtime_count=1)
            result = app._handle_input("/grok consult ping-from-tui")
            self.assertIsInstance(result, str)
            self.assertIn("retired", result)
            self.assertIn("Hawking owns cognition", result)


class TestClearPreservesDurableState(unittest.TestCase):
    def test_clear_wipes_transcript_keeps_goal_mission_dag_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            bus = EventBus()
            tui = TUI(bus, str(ws), "m", 1)
            bus.subscribe(tui._on_event)
            ctrl = Controller(workspace=str(ws), runtime_count=1, bus=bus)
            ctrl.session.messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]
            ctrl.session.steering.append("keep-steer")
            snap = ctrl.start_ultragoal(
                "Prove /clear does not forget the mission. Tests must pass."
            )
            mission_id = snap["mission_id"]
            dag_path = ws / ".hawking" / "dag.json"
            goal_path = ws / ".hawking" / "GOAL.md"
            receipt = ws / ".hawking" / "receipts" / "keep-me.json"
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps({"ok": True}), encoding="utf-8")
            self.assertTrue(dag_path.is_file())
            self.assertTrue(goal_path.is_file())
            tui.transcript = ["You: hello", "world"]
            result = ctrl.handle_command("/clear")
            self.assertEqual(ctrl.session.messages, [])
            self.assertEqual(tui.transcript, [])
            self.assertEqual(ctrl.session.goal, snap["goal"])
            self.assertEqual(ctrl.session.mission_id, mission_id)
            self.assertIsNotNone(ctrl.mission)
            self.assertEqual(ctrl.mission.id, mission_id)
            self.assertTrue(dag_path.is_file())
            reloaded = json.loads(dag_path.read_text(encoding="utf-8"))
            self.assertIn("units", reloaded)
            self.assertTrue(goal_path.is_file())
            self.assertTrue(receipt.is_file())
            self.assertEqual(ctrl.session.steering, ["keep-steer"])
            self.assertIsInstance(result, dict)
            self.assertTrue(result.get("cleared"))
            self.assertEqual(result.get("kind"), "transcript")

    def test_clear_does_not_drop_model_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = Controller(workspace=tmp, runtime_count=2, model="local")
            before = ctrl.model
            ctrl.session.messages.append({"role": "user", "content": "x"})
            ctrl.handle_command("/clear")
            self.assertEqual(ctrl.model, before)
            self.assertEqual(ctrl.runtime_count, 2)


class TestUltragoalIsNotASecondEngine(unittest.TestCase):
    def test_ultragoal_reuses_mission_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = Controller(workspace=tmp, runtime_count=1)
            first = ctrl.handle_command("/ultragoal keep the same mission identity")
            self.assertIsInstance(first, dict)
            mid = first["mission_id"]
            second = ctrl.handle_command("/ultragoal keep the same mission identity")
            self.assertEqual(second["mission_id"], mid)
            self.assertEqual(ctrl.mission.id, mid)


if __name__ == "__main__":
    unittest.main()

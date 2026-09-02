from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]

from hcli.app import App
from hcli.commands import REQUIRED_COMMANDS, CommandHandler
from hcli.controller import Controller
from hcli.events import EventBus
from hcli.grok_bridge import GrokBridge, GrokRunHandle
from hcli.tui import TUI


class TestLiveIngressUnification(unittest.TestCase):
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

    def test_tui_path_grok_reaches_bridge(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = App(workspace=tmp, runtime_count=1)
            calls = []

            def consult(self, prompt, **kwargs):
                calls.append(prompt)
                return GrokRunHandle(
                    task_id="consult-rec",
                    command_run=["grok-run", "consult"],
                    started_at="now",
                    mode="consult",
                    dry_run=True,
                )

            with patch.object(GrokBridge, "consult", consult):
                result = app._handle_input("/grok consult ping-from-tui")
            self.assertEqual(calls, ["ping-from-tui"])
            self.assertTrue(
                (isinstance(result, str) and "consult-rec" in result)
                or getattr(result, "task_id", None) == "consult-rec"
            )


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
            dag_path = ws / ".hcli" / "dag.json"
            goal_path = ws / ".hcli" / "GOAL.md"
            receipt = ws / ".hcli" / "receipts" / "keep-me.json"
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

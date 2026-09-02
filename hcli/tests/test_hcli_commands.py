from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from hcli.commands import CommandHandler


class TestCommandHandler(unittest.TestCase):
    def setUp(self):
        # spec= matters here. An unconstrained MagicMock answers to EVERY
        # attribute, so enrich_status_snapshot found "observed" fields that do
        # not exist on this controller and /status rendered a mission snapshot
        # full of MagicMock repr instead of the session summary the test asks
        # about. Constraining the mock to what a controller actually exposes
        # makes the test describe a real controller rather than a mock's
        # willingness to answer anything.
        self.controller = MagicMock(
            spec=[
                "session",
                "list_models",
                "select_model",
                "model_name",
                "set_goal",
                "queue_steer",
                "request_exit",
            ]
        )
        self.controller.session = MagicMock()
        self.controller.session.id = "sess-1"
        self.controller.session.goal = "test goal"
        self.controller.session.runtime_count = 2
        self.controller.session.model = "model.gguf"
        self.controller.session.messages = [1, 2, 3]
        self.handler = CommandHandler(self.controller)

    def test_non_command_returns_none(self):
        self.assertIsNone(self.handler.handle("hello"))

    def test_help(self):
        result = self.handler.handle("/help")
        self.assertIn("/status", result)
        self.assertIn("/exit", result)

    def test_status(self):
        result = self.handler.handle("/status")
        self.assertIn("sess-1", result)
        self.assertIn("test goal", result)

    def test_goal(self):
        self.handler.handle("/goal new goal")
        self.controller.set_goal.assert_called_with("new goal")

    def test_steer(self):
        self.controller.queue_steer.return_value = MagicMock(text="do x")
        result = self.handler.handle("/steer do x")
        self.assertIn("Steer queued", result)

    def test_exit(self):
        result = self.handler.handle("/exit")
        self.assertIsNone(result)
        self.controller.request_exit.assert_called_once()

    def test_unknown_command(self):
        result = self.handler.handle("/unknown")
        self.assertIn("Unknown", result)

    def test_model_no_arg(self):
        # Bare /model lists what is available rather than printing a usage
        # string; _cmd_model delegates to _cmd_models when there is no argument.
        self.controller.list_models.return_value = [
            {"name": "m1", "path": "/models/m1.gguf"}
        ]
        result = self.handler.handle("/model")
        self.assertIn("Available models:", result)
        self.assertIn("m1", result)

    def test_model_no_arg_with_nothing_discovered(self):
        self.controller.list_models.return_value = []
        self.assertIn("No models discovered", self.handler.handle("/model"))


if __name__ == "__main__":
    unittest.main()
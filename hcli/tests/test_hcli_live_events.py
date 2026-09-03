from __future__ import annotations

import inspect
import tempfile
import unittest
from unittest.mock import patch

from hcli.events import EventBus
from hcli.app import App


class TestEventCompatibility(unittest.TestCase):
    def test_eventbus_accepts_product_two_argument_form(self):
        bus = EventBus()

        bus.emit(
            "session_started",
            {"mode": "interactive"},
        )

        bus.emit(
            "user_message",
            {"text": "hello"},
        )

    def test_app_crosses_real_interactive_start_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = App(
                workspace=tmp,
                runtime_count=3,
                model=None,
                debug=False,
            )

            # Exercise:
            #
            # App.run
            #   -> _run_interactive
            #   -> TUI construction
            #   -> EventBus session_started emission
            #   -> TUI.run
            #   -> Controller.shutdown
            #
            # without blocking for terminal input.
            with patch(
                "hcli.app.TUI.run",
                return_value=0,
            ):
                rc = app.run()

            self.assertEqual(rc, 0)

    def test_headless_start_event_also_crosses_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = App(
                workspace=tmp,
                runtime_count=1,
                model=None,
                debug=False,
            )

            with patch.object(
                app.controller,
                "execute",
                return_value=0,
            ):
                rc = app.run(prompt="test")

            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()

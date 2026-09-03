from __future__ import annotations

import sys
import unittest
from pathlib import Path


from hcli.events import EventBus
from hcli.tui import TUI


class TestTuiEvents(unittest.TestCase):
    def test_final_response_reads_content_and_failure_events_surface(self):
        bus = EventBus()
        tui = TUI(bus, "/tmp/ws", "m", 1)
        bus.subscribe(tui._on_event)
        bus.emit(
            "final_response",
            {
                "goal_id": "g",
                "content": "THE ANSWER IS 42",
                "status": "completed",
            },
        )
        bus.emit("rollback", {"goal_id": "g", "reason": "validation failed"})
        bus.emit("validation_failed", {"goal_id": "g"})
        bus.emit("goal_completed", {"goal_id": "g", "status": "failed"})
        bus.emit("error", {"message": "boom"})
        self.assertIn("THE ANSWER IS 42", tui.transcript)
        joined = "\n".join(tui.transcript)
        self.assertIn("rollback", joined)
        self.assertIn("validation failed", joined)
        self.assertIn("goal failed", joined)
        self.assertIn("boom", joined)
        self.assertEqual(tui.status, "error")


if __name__ == "__main__":
    unittest.main()

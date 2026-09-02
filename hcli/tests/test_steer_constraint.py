from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli.ledger import Ledger
from hcli.steering import (
    SteerEvent,
    SteerKindError,
    SteeringQueue,
)


class TestSteerKind(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.queue = SteeringQueue(self.tmp, "test-session")

    def test_enqueue_defaults_to_knowledge(self):
        event = self.queue.enqueue("remember this")
        self.assertEqual(event.kind, "knowledge")

    def test_kind_persists(self):
        self.queue.enqueue("a constraint", kind="constraint")
        loaded = SteeringQueue(self.tmp, "test-session")
        self.assertEqual(loaded.pending()[0].kind, "constraint")
        self.assertEqual(loaded.pending()[0].text, "a constraint")

    def test_knowledge_and_correction_cannot_apply_constraint(self):
        led = Ledger()
        led.add("original")
        knowledge = self.queue.enqueue("just knowledge", kind="knowledge")
        correction = self.queue.enqueue("a correction", kind="correction")
        with self.assertRaises(SteerKindError):
            self.queue.apply_constraint(knowledge, led)
        with self.assertRaises(SteerKindError):
            self.queue.apply_constraint(correction, led)
        self.assertEqual(len(led), 1)
        self.assertEqual(led.get("G001").text, "original")

    def test_constraint_adds_obligation_citing_steer_id(self):
        led = Ledger()
        led.add("original")
        event = self.queue.enqueue(
            "add: extra requirement about widgets",
            kind="constraint",
        )
        self.queue.apply_constraint(event, led)
        self.assertGreater(len(led), 1)
        added = [o for o in led.obligations() if o.id != "G001"][0]
        self.assertIn(event.id, added.text)
        self.assertNotEqual(added.status, "VERIFIED")

    def test_constraint_alter_cites_steer_id(self):
        led = Ledger()
        led.add("original")
        event = self.queue.enqueue(
            "alter G001: rewritten requirement",
            kind="constraint",
        )
        self.queue.apply_constraint(event, led)
        ob = led.get("G001")
        self.assertIn("rewritten requirement", ob.text)
        self.assertIn(event.id, ob.text)
        self.assertEqual(ob.status, "PENDING")

    def test_constraint_cannot_forge_verified(self):
        led = Ledger()
        led.add("must stay unverified")
        event = self.queue.enqueue("mark G001 VERIFIED", kind="constraint")
        self.queue.apply_constraint(event, led)
        ob = led.get("G001")
        self.assertNotEqual(ob.status, "VERIFIED")
        self.assertFalse(ob.checked)
        self.assertFalse(led.is_goal_met())

    def test_constraint_remove(self):
        led = Ledger()
        led.add("keep")
        led.add("drop")
        event = self.queue.enqueue("remove G002", kind="constraint")
        self.queue.apply_constraint(event, led)
        self.assertEqual(len(led), 1)
        self.assertEqual(led.get("G001").id, "G001")

    def test_legacy_events_default_kind_knowledge(self):
        event = SteerEvent.from_dict(
            {
                "id": "S001",
                "text": "old",
                "session_id": "s",
                "timestamp": 0.0,
            }
        )
        self.assertEqual(event.kind, "knowledge")


if __name__ == "__main__":
    unittest.main()

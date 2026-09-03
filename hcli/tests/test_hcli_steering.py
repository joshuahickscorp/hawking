from __future__ import annotations

import os
import tempfile
import unittest

from hcli.steering import SteeringQueue, SteerEvent


class TestSteeringQueue(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.queue = SteeringQueue(self.tmp, "test-session")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_enqueue_and_pending(self):
        e = self.queue.enqueue("prioritize parser")
        self.assertEqual(len(self.queue.pending()), 1)
        self.assertEqual(self.queue.pending()[0].text, "prioritize parser")

    def test_apply_pending(self):
        self.queue.enqueue("don't touch persistence")
        applied = self.queue.apply_pending()
        self.assertEqual(len(applied), 1)
        self.assertTrue(applied[0].applied)
        self.assertEqual(len(self.queue.pending()), 0)

    def test_persistence(self):
        self.queue.enqueue("use existing scheduler")
        q2 = SteeringQueue(self.tmp, "test-session")
        self.assertEqual(len(q2.pending()), 1)
        self.assertEqual(q2.pending()[0].text, "use existing scheduler")

    def test_clear_applied(self):
        self.queue.enqueue("stop once tests pass")
        self.queue.apply_pending()
        self.queue.clear_applied()
        self.assertEqual(len(self.queue.all()), 0)

    def test_multiple_events(self):
        self.queue.enqueue("a")
        self.queue.enqueue("b")
        self.assertEqual(len(self.queue.pending()), 2)
        applied = self.queue.apply_pending()
        self.assertEqual(len(applied), 2)
        self.assertEqual(len(self.queue.pending()), 0)


if __name__ == "__main__":
    unittest.main()
"""Two defect shapes that keep recurring in this codebase, at new call sites.

1. os.kill(pid, 0) as "is it alive". It SUCCEEDS on a zombie, so a child that
   has already exited reads as alive until someone reaps it. This burned the
   full grace in _terminate_pids; pid_is_alive had the same shape untouched.

2. An unlocked read-modify-write over a file another process also writes. This
   let BackgroundJobStore._refresh clobber a completion record; GrokBridge's
   _update_receipt had the same shape, where a status() poll could reverse a
   concurrent cancel().
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

from hcli.resources import pid_is_alive


class TestZombieIsNotAlive(unittest.TestCase):
    def test_an_exited_child_is_not_reported_alive(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # The test must NOT reap: waitpid here is what the fix does, so calling
        # it first makes the assertion vacuous -- the first version of this test
        # passed with the fix reverted for exactly that reason. Wait for the
        # kernel to mark the pid a zombie instead, observed from outside.
        deadline = time.time() + 10
        state = ""
        while time.time() < deadline:
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(proc.pid)],
                capture_output=True, text=True,
            ).stdout.strip()
            if state.startswith("Z"):
                break
            time.sleep(0.005)
        self.assertTrue(
            state.startswith("Z"),
            f"pid {proc.pid} never became a zombie (state {state!r}); "
            "the test cannot exercise the defect",
        )
        self.assertFalse(
            pid_is_alive(proc.pid),
            f"pid {proc.pid} is a zombie but is reported alive",
        )
        proc.poll()

    def test_a_live_process_is_still_reported_alive(self):
        """Negative control: the reap must not make everything look dead."""
        self.assertTrue(pid_is_alive(os.getpid()))

    def test_nonsense_pids_are_not_alive(self):
        for bad in (0, -1, None, "x"):
            self.assertFalse(pid_is_alive(bad))  # type: ignore[arg-type]


class TestReceiptWriteIsSerialized(unittest.TestCase):
    def _bridge(self, tmp):
        from hcli.grok_bridge import GrokBridge

        return GrokBridge(workspace=Path(tmp))

    def test_concurrent_updates_leave_one_valid_receipt(self):
        """Concurrency smoke, NOT a lost-update detector.

        Honest about its own power: this passes with the lock removed. The
        window is narrow enough that six threads and forty writes each do not
        reliably lose one, so what it holds is that concurrent updates do not
        crash or leave a half-written receipt. `test_the_lock_is_actually_taken`
        is what fails when the lock goes.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bridge = self._bridge(tmp)
            bridge.receipts_dir.mkdir(parents=True, exist_ok=True)
            task = "t1"
            bridge.receipt_path(task).write_text(
                json.dumps({"task_id": task, "mode": "consult",
                            "status": {"state": "running"}, "note": 0}),
                encoding="utf-8",
            )

            errors = []

            def bump(n):
                try:
                    for i in range(40):
                        bridge._update_receipt(task, note=n * 1000 + i)
                except Exception as exc:  # pragma: no cover - surfaced below
                    errors.append(exc)

            threads = [threading.Thread(target=bump, args=(n,)) for n in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            self.assertEqual(errors, [], f"concurrent updates raised: {errors}")
            # The file must still be one valid receipt, not an interleaved one.
            data = json.loads(bridge.receipt_path(task).read_text())
            self.assertEqual(data["task_id"], task)
            self.assertIn("note", data)

    def test_the_lock_is_actually_taken(self):
        """Naming the guard, so removing it is a visible change."""
        import inspect

        from hcli.grok_bridge import GrokBridge

        src = inspect.getsource(GrokBridge._update_receipt)
        # `with self._receipt_lock(` and not just the name: "_receipt_lock" is a
        # substring of the helper "_update_receipt_locked", so the looser
        # assertion passed with the lock removed.
        self.assertIn("with self._receipt_lock(task_id):", src)
        self.assertIn("flock", inspect.getsource(GrokBridge._receipt_lock))


if __name__ == "__main__":
    unittest.main()

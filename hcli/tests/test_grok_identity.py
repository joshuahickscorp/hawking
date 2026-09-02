"""Grok task identity and liveness.

Both of these were watched failing against the pre-change bridge:

* every ``consult`` used the literal slug ``consult``, so grok-run's
  one-second-resolution id let two dispatches share one task directory. In the
  recorded mixed-max run ``grok1`` and ``grok2`` both resolved to
  ``consult-20260822-224557``; two WorkUnits were accepted off a single Grok
  execution and the second unit's prompt was never sent.
* ``status`` reported whatever the status FILE said, with no pid anywhere, so a
  task whose process had died still read ``running`` and held its scheduler slot
  for the full wait timeout.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path


from hcli.grok_bridge import (  # noqa: E402
    GrokBridge,
    extract_background_pid,
    process_alive,
    unique_task_slug,
)


class TestUniqueTaskSlug(unittest.TestCase):
    def test_slugs_are_distinct_within_one_second(self):
        slugs = [unique_task_slug("consult") for _ in range(500)]
        self.assertEqual(len(set(slugs)), len(slugs))

    def test_slugs_are_distinct_across_threads(self):
        out = []
        lock = threading.Lock()

        def work():
            got = [unique_task_slug("consult") for _ in range(200)]
            with lock:
                out.extend(got)

        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(set(out)), len(out))

    def test_slug_keeps_its_prefix(self):
        self.assertTrue(unique_task_slug("audit").startswith("audit-"))


class TestBackgroundPid(unittest.TestCase):
    def test_extracts_pid_from_real_grok_run_output(self):
        text = (
            "grok-run: sparse checkout: 43 path(s),  37M on disk\n"
            "hv3-ctxauth-20260822-233043\n"
            "grok-run: started in background (pid 92619) — poll with: "
            "grok-run wait --id hv3-ctxauth-20260822-233043\n"
        )
        self.assertEqual(extract_background_pid(text), 92619)

    def test_returns_none_when_absent(self):
        self.assertIsNone(extract_background_pid("nothing here"))
        self.assertIsNone(extract_background_pid(""))

    def test_process_alive_is_false_on_doubt(self):
        self.assertTrue(process_alive(os.getpid()))
        self.assertFalse(process_alive(None))
        self.assertFalse(process_alive(0))
        # A pid that cannot exist on macOS or Linux. os.kill raises
        # OverflowError rather than OSError for this, which the first version
        # of process_alive did not catch.
        self.assertFalse(process_alive(4_000_000_000))


class TestStaleRunningDetection(unittest.TestCase):
    """A status file left at `running` by a dead process is not `running`."""

    def setUp(self):
        # status() resolves the executable before it runs anything; point it at
        # the real one so the test exercises the liveness logic, not PATH.
        self._prev = os.environ.get("GROK_RUN")
        os.environ["GROK_RUN"] = str(Path.home() / ".claude-grok" / "bin" / "grok-run")

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("GROK_RUN", None)
        else:
            os.environ["GROK_RUN"] = self._prev

    def _bridge_with_receipt(self, tmp, task_id, pid):
        bridge = GrokBridge(tmp)
        bridge.receipts_dir.mkdir(parents=True, exist_ok=True)
        (bridge.receipts_dir / f"{task_id}.json").write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "mode": "consult",
                    "launch_pid": pid,
                    "status": {"state": "running", "exit_code": None},
                }
            ),
            encoding="utf-8",
        )
        return bridge

    def test_dead_pid_downgrades_running_to_stale_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_id = "consult-1x0x000001"
            bridge = self._bridge_with_receipt(tmp, task_id, 4_000_000_000)
            bridge._run = lambda *a, **k: type(
                "R", (), {"stdout": "status: running (exit -)", "stderr": "", "returncode": 0}
            )()
            got = bridge.status(task_id)
        self.assertEqual(got["state"], "stale-running")
        self.assertFalse(got["process_alive"])
        self.assertIn("4000000000", got["stale_reason"])

    def test_live_pid_keeps_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_id = "consult-1x0x000002"
            bridge = self._bridge_with_receipt(tmp, task_id, os.getpid())
            bridge._run = lambda *a, **k: type(
                "R", (), {"stdout": "status: running (exit -)", "stderr": "", "returncode": 0}
            )()
            got = bridge.status(task_id)
        self.assertEqual(got["state"], "running")
        self.assertTrue(got["process_alive"])

    def test_terminal_state_is_never_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_id = "consult-1x0x000003"
            bridge = self._bridge_with_receipt(tmp, task_id, 4_000_000_000)
            bridge._run = lambda *a, **k: type(
                "R", (), {"stdout": "status: done (exit 0)", "stderr": "", "returncode": 0}
            )()
            got = bridge.status(task_id)
        self.assertEqual(got["state"], "done")


if __name__ == "__main__":
    unittest.main()

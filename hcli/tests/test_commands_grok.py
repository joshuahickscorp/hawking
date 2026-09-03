from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]

from hcli.commands import CommandHandler
from hcli.grok_bridge import GrokBridge, GrokRunHandle

VALID_CONTRACT = """# WRITE
create hcli/commands.py

# VERIFY
`pytest hcli/tests -q` must exit 0

# ACCEPTANCE
pytest exits 0
"""


class FakeSession:
    def __init__(self):
        self.id = "sess-1"
        self.goal = "test goal"
        self.runtime_count = 1
        self.model = "model.gguf"
        self.messages = []


class FakeSteer:
    def __init__(self, text):
        self.text = text


class FakeLock:
    def __init__(self):
        self.acquires = []
        self.releases = []
        self.held = False

    def acquire(self, unit_id):
        self.acquires.append(unit_id)
        if self.held:
            return False
        self.held = True
        return True

    def release(self, unit_id=None):
        self.releases.append(unit_id)
        self.held = False


class FakeController:
    def __init__(self, workspace, mutation_lock=None, mission=None):
        self.workspace_root = str(workspace)
        self.session = FakeSession()
        self.mission = mission
        self.mutation_lock = mutation_lock

    def select_model(self, name):
        return name

    def queue_steer(self, text):
        return FakeSteer(text)

    def run_mission(self, goal):
        return {"mission_id": "m1", "status": "ok", "reason": "started"}

    def list_models(self):
        return []


def _handle_ret(mode):
    if mode in ("delegate", "audit", "consult"):

        def launch(self, *args, **kwargs):
            return GrokRunHandle(
                task_id=f"{mode}-rec",
                command_run=["grok-run", mode],
                started_at="now",
                mode=mode,
                dry_run=True,
            )

        return launch

    if mode in ("status", "wait"):

        def statusy(self, task_id, *args, **kwargs):
            return {"state": "done", "exit_code": 0, "task_id": task_id}

        return statusy

    if mode == "report":

        def report(self, task_id, *args, **kwargs):
            return f"report-for-{task_id}"

        return report

    def cleanup(self, task_id, *args, **kwargs):
        return {"task_id": task_id, "ok": True, "exit_code": 0, "stdout": "", "stderr": ""}

    return cleanup


@contextmanager
def record_bridge(methods=None):
    names = methods or [
        "delegate",
        "audit",
        "consult",
        "status",
        "wait",
        "report",
        "cleanup",
    ]
    calls = []
    patches = []

    def wrap(name):
        ret = _handle_ret(name)

        def impl(self, *args, **kwargs):
            calls.append({"method": name, "args": args, "kwargs": kwargs})
            return ret(self, *args, **kwargs)

        return impl

    try:
        for name in names:
            p = patch.object(GrokBridge, name, wrap(name))
            p.start()
            patches.append(p)
        yield calls
    finally:
        for p in reversed(patches):
            p.stop()


class TestCmdGrok(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.contract = self.root / "contract.md"
        self.contract.write_text(VALID_CONTRACT, encoding="utf-8")
        self.controller = FakeController(self.root)
        self.handler = CommandHandler(self.controller)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_help_lists_grok(self):
        result = self.handler.handle("/help")
        self.assertIn("/grok", result)

    def test_no_verb_returns_usage(self):
        result = self.handler.handle("/grok")
        self.assertIn("Commands:", result)
        self.assertIn("/grok delegate", result)
        self.assertIn("/grok consult", result)

    def test_unknown_verb_returns_usage(self):
        result = self.handler.handle("/grok frobnicate")
        self.assertIn("Commands:", result)
        self.assertIn("delegate", result)

    def test_missing_delegate_operands(self):
        result = self.handler.handle("/grok delegate")
        self.assertIn("Usage:", result)
        result = self.handler.handle("/grok delegate only-task")
        self.assertIn("Usage:", result)

    def test_missing_contract_file(self):
        result = self.handler.handle("/grok delegate t /no/such/contract.md")
        self.assertIn("not found", result.lower())

    def test_lazy_bridge_is_singleton_for_same_root(self):
        first = self.handler._grok_bridge()
        second = self.handler._grok_bridge()
        self.assertIs(first, second)
        self.assertEqual(Path(first.workspace), self.root)

    def test_bridge_rebuilds_when_workspace_changes(self):
        first = self.handler._grok_bridge()
        other = self.root / "other"
        other.mkdir()
        self.controller.workspace_root = str(other)
        second = self.handler._grok_bridge()
        self.assertIsNot(first, second)
        self.assertEqual(Path(second.workspace), other)

    def test_delegate_reads_contract_and_passes_none_lock(self):
        with record_bridge() as calls:
            result = self.handler.handle(
                f"/grok delegate slug-one {self.contract}"
            )
        self.assertIn("delegate-rec", result)
        hit = [c for c in calls if c["method"] == "delegate"]
        self.assertEqual(len(hit), 1)
        args, kwargs = hit[0]["args"], hit[0]["kwargs"]
        self.assertEqual(args[0], "slug-one")
        self.assertIn("WRITE", args[1])
        self.assertIsNone(kwargs.get("mutation_lock"))

    def test_delegate_relative_contract_under_workspace(self):
        with record_bridge() as calls:
            result = self.handler.handle("/grok delegate slug-rel contract.md")
        self.assertNotIn("not found", (result or "").lower())
        hit = [c for c in calls if c["method"] == "delegate"]
        self.assertEqual(len(hit), 1)
        self.assertIn("WRITE", hit[0]["args"][1])

    def test_wires_controller_mutation_lock(self):
        lock = FakeLock()
        handler = CommandHandler(FakeController(self.root, mutation_lock=lock))
        with record_bridge() as calls:
            handler.handle(f"/grok delegate slug-lock {self.contract}")
        hit = [c for c in calls if c["method"] == "delegate"]
        self.assertEqual(len(hit), 1)
        factory = hit[0]["kwargs"].get("mutation_lock")
        self.assertIsNotNone(factory)
        with factory():
            self.assertEqual(lock.acquires, ["hcli-grok-delegate"])
            self.assertTrue(lock.held)
        self.assertEqual(lock.releases, ["hcli-grok-delegate"])
        self.assertFalse(lock.held)

    def test_wires_mission_scheduler_lock_by_getattr(self):
        lock = FakeLock()

        class Sched:
            mutation_lock = lock

        class Mission:
            scheduler = Sched()

        handler = CommandHandler(FakeController(self.root, mission=Mission()))
        with record_bridge() as calls:
            handler.handle(f"/grok delegate slug-ms {self.contract}")
        factory = calls[0]["kwargs"].get("mutation_lock")
        self.assertIsNotNone(factory)
        with factory():
            self.assertEqual(lock.acquires, ["hcli-grok-delegate"])

    def test_audit_consult_status_wait_report_cleanup(self):
        with record_bridge() as calls:
            self.handler.handle(f"/grok audit a1 {self.contract}")
            self.handler.handle("/grok consult ping please")
            self.handler.handle("/grok status tid-1")
            self.handler.handle("/grok wait tid-2")
            report = self.handler.handle("/grok report tid-3")
            self.handler.handle("/grok cleanup tid-4")
        methods = [c["method"] for c in calls]
        self.assertEqual(
            methods, ["audit", "consult", "status", "wait", "report", "cleanup"]
        )
        self.assertEqual(calls[0]["args"][0], "a1")
        self.assertEqual(calls[1]["args"][0], "ping please")
        self.assertEqual(calls[2]["args"][0], "tid-1")
        self.assertEqual(calls[3]["args"][0], "tid-2")
        self.assertEqual(calls[4]["args"][0], "tid-3")
        self.assertEqual(calls[5]["args"][0], "tid-4")
        self.assertIn("report-for-tid-3", report)

    def test_existing_commands_still_work(self):
        model = self.handler.handle("/model foo")
        mission = self.handler.handle("/mission do it")
        steer = self.handler.handle("/steer remember")
        status = self.handler.handle("/status")
        self.assertIn("Switched", model)
        self.assertIn("m1", mission)
        self.assertIn("Steer queued", steer)
        self.assertIn("sess-1", status)


if __name__ == "__main__":
    unittest.main()

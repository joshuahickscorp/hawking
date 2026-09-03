from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli.commands import (
    MIN_ACCEPTED_RATE_WINDOW_S,
    CommandHandler,
    format_accepted_h,
    format_status,
)
from hcli.ledger import Ledger


GOAL_MD = """# GOAL

- [ ] G001 — first obligation | status: PENDING | risk: high | tier: V2
      acceptance: a exists
      verify: python3 -c "import sys; sys.exit(0)"
      evidence: (none yet)
- [ ] G002 — second obligation | status: ACTIVE | risk: medium | tier: V1
      acceptance: b exists
      verify: python3 -c "import sys; sys.exit(1)"
      evidence: (none yet)
- [x] G003 — third obligation | status: VERIFIED | risk: low | tier: V0
      acceptance: c exists
      verify: python3 -c "import sys; sys.exit(0)"
      evidence: already done
"""


class FakeSession:
    def __init__(self):
        self.id = "sess-1"
        self.goal = "test goal"
        self.runtime_count = 1
        self.model = "model.gguf"
        self.messages = []


class FakeScheduler:
    def __init__(self, last_dispatch=None):
        self.last_dispatch = last_dispatch
        self.units = {}


class FakeMission:
    def __init__(
        self,
        *,
        started_at,
        accepted_count,
        last_checkpoint=0.0,
        phase="running",
        goal="ship status",
        no_progress_warning=None,
        last_dispatch=None,
        ledger=None,
    ):
        self.id = "m-observe"
        self.phase = phase
        self.goal = goal
        self.started_at = started_at
        self.accepted_count = accepted_count
        self.last_checkpoint = last_checkpoint
        self.no_progress_warning = no_progress_warning
        self.scheduler = FakeScheduler(last_dispatch)
        self.ledger = ledger

    def status(self):
        elapsed = max(0.0, time.time() - float(self.started_at))
        # Deliberately reproduce the 1164 lie so /status has to refuse it.
        per_hour = (
            float(self.accepted_count) / (elapsed / 3600.0) if elapsed > 0 else 0.0
        )
        return {
            "mission_id": self.id,
            "phase": self.phase,
            "units_by_status": {
                "pending": 0,
                "ready": 1,
                "running": 0,
                "completed": int(self.accepted_count),
                "failed": 0,
            },
            "active_runtimes": 0,
            "active_decodes": 0,
            "accepted_units_per_hour": per_hour,
            "elapsed_wall": elapsed,
            "last_checkpoint": self.last_checkpoint,
            "no_progress_warning": self.no_progress_warning,
        }


class FakeController:
    def __init__(self, workspace, mission=None, ledger=None, status_snap=None):
        self.workspace_root = str(workspace)
        self.session = FakeSession()
        self.mission = mission
        self._ledger = ledger
        self._status_snap = status_snap

    def status(self):
        if self._status_snap is not None:
            return dict(self._status_snap)
        if self.mission is None:
            return {
                "workspace": self.workspace_root,
                "requested_runtimes": 1,
                "admitted_runtimes": 0,
                "model": None,
                "model_name": None,
                "runtimes": [],
                "engine_active": False,
                "shutdown": False,
            }
        snap = self.mission.status()
        snap["workspace"] = self.workspace_root
        return snap

    def list_models(self):
        return []


def _handler(workspace, **kwargs) -> CommandHandler:
    return CommandHandler(FakeController(workspace, **kwargs))


class TestFormatAcceptedH(unittest.TestCase):
    def test_short_window_prints_count_and_duration_not_rate(self):
        text = format_accepted_h(4, 12.367)
        self.assertEqual(text, "4 in 12s")
        self.assertNotIn("1164", text)
        self.assertNotIn("1164.3", format_status({
            "accepted_count": 4,
            "elapsed_wall": 12.367,
            "accepted_units_per_hour": 1164.349,
        }))

    def test_just_under_floor_still_raw(self):
        elapsed = MIN_ACCEPTED_RATE_WINDOW_S - 1
        text = format_accepted_h(4, elapsed)
        self.assertTrue(text.startswith("4 in "))
        self.assertNotIn(".", text.split(" in ", 1)[0])

    def test_at_floor_prints_rate(self):
        text = format_accepted_h(4, MIN_ACCEPTED_RATE_WINDOW_S)
        self.assertEqual(text, "48.0")

    def test_precomputed_rate_without_window_is_unknown(self):
        text = format_status({"accepted_units_per_hour": 1164.349})
        self.assertIn("accepted/h=unknown", text)
        self.assertNotIn("1164", text)


class TestStatusCompleteness(unittest.TestCase):
    def test_short_window_mission_refuses_annualisation(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = time.time()
            mission = FakeMission(
                started_at=now - 12.367,
                accepted_count=4,
                last_checkpoint=now - 5.0,
            )
            text = _handler(tmp, mission=mission).handle("/status")
            self.assertIn("accepted/h=4 in 12s", text)
            self.assertNotIn("1164", text)
            self.assertIn("ckpt=5.0s", text)
            self.assertLessEqual(len(text.splitlines()), 10)

    def test_long_window_prints_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = time.time()
            mission = FakeMission(
                started_at=now - 3600.0,
                accepted_count=12,
                last_checkpoint=now - 90.0,
            )
            text = _handler(tmp, mission=mission).handle("/status")
            self.assertIn("accepted/h=12.0", text)
            self.assertIn("ckpt=90s", text)

    def test_ledger_backlog_and_watchdog_from_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "GOAL.md"
            path.write_text(GOAL_MD, encoding="utf-8")
            ledger = Ledger.parse(path)
            now = time.time()
            mission = FakeMission(
                started_at=now - 400.0,
                accepted_count=2,
                last_checkpoint=now - 20.0,
                ledger=ledger,
            )
            text = _handler(tmp, mission=mission, ledger=ledger).handle("/status")
            self.assertIn("Verifier backlog=3", text)
            self.assertIn("watchdog=GOAL_NOT_MET", text)
            # THREE, not two. G003 is hand-marked `- [x] | status: VERIFIED`
            # with no verification sidecar, and this ledger deliberately parses
            # an unbacked VERIFIED as STALE rather than trusting it. A backlog
            # that believed the checkbox would be counting forgeries as done.
            self.assertEqual(len(ledger.unverified()), 3)

    def test_goal_md_on_disk_without_in_memory_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            goal = Path(tmp) / ".hcli" / "GOAL.md"
            goal.parent.mkdir(parents=True)
            goal.write_text(GOAL_MD, encoding="utf-8")
            text = _handler(tmp).handle("/status")
            self.assertIn("Verifier backlog=3", text)
            self.assertIn("watchdog=GOAL_NOT_MET", text)

    def test_checkpoint_and_accepted_from_state_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = time.time()
            state_dir = Path(tmp) / ".hcli" / "mission"
            state_dir.mkdir(parents=True)
            payload = {
                "id": "durable-m",
                "phase": "running",
                "goal": "durable goal",
                "started_at": now - 12.367,
                "last_checkpoint": now - 8.0,
                "accepted_count": 4,
            }
            (state_dir / "state.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            text = _handler(tmp).handle("/status")
            self.assertIn("mission durable-m", text)
            self.assertIn("accepted/h=4 in 12s", text)
            self.assertNotIn("1164", text)
            self.assertIn("ckpt=8.0s", text)
            self.assertIn("Goal: durable goal", text)

    def test_never_checkpointed_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = time.time()
            mission = FakeMission(
                started_at=now - 400.0,
                accepted_count=1,
                last_checkpoint=0.0,
            )
            text = _handler(tmp, mission=mission).handle("/status")
            self.assertIn("ckpt=unknown", text)

    def test_no_progress_watchdog(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = time.time()
            mission = FakeMission(
                started_at=now - 400.0,
                accepted_count=0,
                last_checkpoint=now - 30.0,
                phase="no_progress",
                no_progress_warning="fingerprint repeated",
            )
            text = _handler(tmp, mission=mission).handle("/status")
            self.assertIn("watchdog=no_progress", text)
            self.assertIn("no_progress: fingerprint repeated", text)

    def test_last_dispatch_without_clock_is_not_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = time.time()
            mission = FakeMission(
                started_at=now - 400.0,
                accepted_count=0,
                last_checkpoint=now - 10.0,
                last_dispatch={
                    "requested": {"LIGHT_CONTROL": 1},
                    "admitted": {"LIGHT_CONTROL": 1},
                    "overhead_s": 0.001,
                },
            )
            text = _handler(tmp, mission=mission).handle("/status")
            self.assertIn("watchdog=unknown", text)
            self.assertNotIn("watchdog=ok", text)

    def test_empty_sources_stay_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = _handler(tmp).handle("/status")
            self.assertIn("Verifier backlog=unknown", text)
            self.assertIn("accepted/h=unknown", text)
            self.assertIn("ckpt=unknown", text)
            self.assertIn("watchdog=unknown", text)
            self.assertLessEqual(len(text.splitlines()), 10)
            for line in text.splitlines():
                self.assertLessEqual(len(line), 80)

    def test_one_screen_and_truncated_goal(self):
        body = "line1 of the ultragoal\n" + ("x" * 400)
        text = format_status(
            {
                "mission_id": "m-1",
                "phase": "running",
                "goal": body,
                "verifier_backlog": 3,
                "accepted_count": 4,
                "elapsed_wall": 12.367,
                "checkpoint_age_s": 12,
                "watchdog": "GOAL_NOT_MET",
            }
        )
        lines = text.splitlines()
        self.assertLessEqual(len(lines), 10)
        self.assertTrue(lines[1].startswith("Goal: line1 of the ultragoal"))
        self.assertNotIn("xxx", text)
        self.assertIn("Verifier backlog=3", text)
        self.assertIn("accepted/h=4 in 12s", text)
        self.assertIn("ckpt=12s", text)
        self.assertIn("watchdog=GOAL_NOT_MET", text)

    def test_session_fallback_without_status_method(self):
        class Bare:
            def __init__(self, workspace):
                self.workspace_root = str(workspace)
                self.session = FakeSession()
                self.mission = None

        with tempfile.TemporaryDirectory() as tmp:
            text = CommandHandler(Bare(tmp)).handle("/status")
            self.assertIn("sess-1", text)
            self.assertIn("Session:", text)


if __name__ == "__main__":
    unittest.main()

"""Durable future goals are distinct from steering and survive restart."""
from __future__ import annotations

import json
import gzip

from hcli.controller import Controller
from hcli.goal_bank import GOAL_BANK_SCHEMA, GoalBank
from hcli.knowledge import KnowledgeStore
from hcli.persist import atomic_write_json
from hcli.tui import TUI
from hcli.events import EventBus


def test_goal_bank_is_fifo_and_persists_modes(tmp_path):
    bank = GoalBank(tmp_path)
    first = bank.add("ship the report")
    second = bank.add("run the overnight production mission", mode="mission")

    snapshot = bank.snapshot()
    assert snapshot["schema"] == GOAL_BANK_SCHEMA
    assert snapshot["queued_count"] == 2
    assert snapshot["next"]["id"] == first["id"]
    assert snapshot["queued"][1]["mode"] == "mission"

    claimed = bank.claim_next()
    assert claimed["id"] == first["id"]
    assert bank.snapshot()["running_count"] == 1

    # A live owner is not recovered by a second controller.
    assert GoalBank(tmp_path).recover_inflight() == 0
    bank.finish(first["id"], {"status": "completed", "goal_id": "g1"})
    assert bank.claim_next()["id"] == second["id"]


def test_dead_owner_returns_goal_to_queue(tmp_path):
    bank = GoalBank(tmp_path)
    item = bank.add("recover me")
    claimed = bank.claim_next()
    document = json.loads(bank.path.read_text(encoding="utf-8"))
    document["goals"][0]["owner_pid"] = 2**62
    atomic_write_json(bank.path, document)

    assert GoalBank(tmp_path).recover_inflight() == 1
    recovered = GoalBank(tmp_path).snapshot()["next"]
    assert recovered["id"] == item["id"]
    assert recovered["status"] == "queued"
    assert "returned to the bank" in recovered["last_error"]


class _CompletingEngine:
    def __init__(self):
        self.calls = []

    def execute(self, prompt, *, context_memory=None):
        self.calls.append((prompt, context_memory))
        return {
            "kind": "answer",
            "content": f"done: {prompt}",
            "operations": [],
            "tests": [],
            "status": "completed",
        }

    def cancel(self):
        return None


def test_completed_goal_drains_banked_goals_in_order(tmp_path):
    controller = Controller(tmp_path)
    engine = _CompletingEngine()
    controller.engine = engine
    first = controller.bank_goal("first future goal")
    second = controller.bank_goal("second future goal")
    try:
        result = controller.execute("current goal")
        assert [call[0] for call in engine.calls] == [
            "current goal",
            "first future goal",
            "second future goal",
        ]
        assert [row["id"] for row in result["bank_started"]] == [
            first["id"],
            second["id"],
        ]
        snapshot = controller.goal_bank_snapshot()
        assert snapshot["queued_count"] == 0
        assert [row["status"] for row in snapshot["recent"][:2]] == [
            "completed",
            "completed",
        ]
    finally:
        controller.shutdown()


def test_bank_command_and_tui_surface_are_visible(tmp_path):
    controller = Controller(tmp_path)
    try:
        queued = controller.handle_command("/bank make the next artifact")
        assert queued["goal"] == "make the next artifact"
        alias = controller.handle_command("\\bank make the alias artifact")
        assert alias["goal"] == "make the alias artifact"
        shown = controller.handle_command("/bank")
        assert shown["queued_count"] == 2
        assert shown["next"]["id"] == queued["id"]

        tui = TUI(
            EventBus(),
            str(tmp_path),
            "local",
            1,
            bank_snapshot_fn=controller.goal_bank_snapshot,
        )
        header = tui.render_header()
        assert "Bank: queued=2" in header
        assert "make the next artifact" in header
    finally:
        controller.shutdown()


def test_literal_bank_alias_uses_app_command_ingress(tmp_path):
    from hcli.app import App

    app = App(str(tmp_path))
    try:
        app._handle_input("\\bank route through the TUI ingress")
        snapshot = app.controller.goal_bank_snapshot()
        assert snapshot["queued_count"] == 1
        assert snapshot["next"]["goal"] == "route through the TUI ingress"
    finally:
        app.controller.shutdown()


def test_prior_knowledge_is_workspace_scoped_and_archived(tmp_path):
    store = KnowledgeStore(tmp_path)
    store.record_note("never discard the staged release", kind="constraint")
    store.record_checkpoint(
        {
            "active_goal": {"text": "ship the release"},
            "staging": {"staged": {"count": 2}},
            "mission": {"phase": "completed"},
        }
    )

    snapshot = KnowledgeStore(tmp_path).snapshot()
    assert snapshot["records"]
    assert any(
        item.get("kind") == "semantic_checkpoint" and item.get("verified") is True
        for item in snapshot["records"]
    )
    assert store.archive_path.is_file()
    assert b"ship the release" in gzip.decompress(store.archive_path.read_bytes())

    controller = Controller(tmp_path)
    try:
        assert controller.session.memory["prior_knowledge"]["records"]
        assert "memory=prior#" in controller.context_summary()
    finally:
        controller.shutdown()


def test_agentos_drains_the_same_bank_when_tui_is_closed(tmp_path):
    from hcli.agentos.runtime import AgentOS

    class FakeScheduler:
        units = {}

    class FakeMission:
        def __init__(self, goal):
            self.goal = goal
            self.id = f"mission-{goal}"
            self.phase = "completed"
            self.scheduler = FakeScheduler()

        def run(self):
            return {
                "status": "completed",
                "state": "VERIFIED",
                "mission_id": self.id,
            }

        def status(self):
            return {"status": "completed", "phase": "completed", "state": "VERIFIED"}

    agent = AgentOS(tmp_path, engine=object())
    first = agent.goal_bank.add("nightly report")
    second = agent.goal_bank.add("publish report", mode="mission")
    calls = []
    agent.mission = FakeMission("current")
    agent.start_mission = lambda goal, **_kwargs: (calls.append(goal) or FakeMission(goal))

    result = agent.run()
    assert calls == ["nightly report", "publish report"]
    assert [row["id"] for row in result["bank_started"]] == [first["id"], second["id"]]
    assert agent.goal_bank.snapshot()["queued_count"] == 0


def test_agentos_runs_one_banked_goal_while_active_mission_is_still_runnable(tmp_path):
    """A long resident mission must not make the bank write-only."""
    from hcli.agentos.runtime import AgentOS

    class ActiveScheduler:
        def is_done(self):
            return False

        class Unit:
            def to_dict(self):
                return {"id": "long-unit", "status": "running"}

        units = {"long-unit": Unit()}

    class ActiveMission:
        phase = "running"
        goal = "long mission"
        id = "long-mission"
        scheduler = ActiveScheduler()

        def run(self):
            return {"status": "failed", "reason": "bounded test stop"}

        def status(self):
            return {"status": "failed", "phase": "failed"}

    class Engine:
        def __init__(self):
            self.calls = []

        def execute(self, goal, *, context_memory=None):
            self.calls.append(goal)
            return {"status": "completed", "goal_id": "auto-goal"}

    engine = Engine()
    agent = AgentOS(tmp_path, engine=engine)
    item = agent.goal_bank.add("small self-verifying unit", mode="mission")
    agent.mission = ActiveMission()

    result = agent.run()

    assert engine.calls == ["small self-verifying unit"]
    assert result["status"] == "failed"
    assert result["bank_started"] == [{
        "id": item["id"],
        "goal": "small self-verifying unit",
        "mode": "mission",
        "status": "completed",
    }]
    assert agent.goal_bank.snapshot()["queued_count"] == 0
    assert agent.goal_bank.snapshot()["recent"][0]["status"] == "completed"


def test_inconclusive_completed_envelope_never_promotes_the_bank(tmp_path):
    from hcli.agentos.runtime import AgentOS

    agent = AgentOS(tmp_path, engine=object())
    agent.goal_bank.add("must wait for verified success")

    promoted = agent._drain_goal_bank({
        "status": "completed",
        "state": "INCONCLUSIVE",
        "verdict": "INCONCLUSIVE",
        "failed_units": ["G001.work"],
    })

    assert promoted == []
    assert agent.goal_bank.snapshot()["queued_count"] == 1


def test_resident_wakes_only_for_queued_bank_work(tmp_path):
    from hcli.agentos.resident import _goal_bank_has_work

    bank = GoalBank(tmp_path)
    item = bank.add("wake the resident")
    assert _goal_bank_has_work(tmp_path) is True
    claimed = bank.claim_next()
    assert claimed["id"] == item["id"]
    assert _goal_bank_has_work(tmp_path) is False

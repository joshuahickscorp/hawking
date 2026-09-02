"""Semantic compaction keeps state that a four-message slice would lose."""
from __future__ import annotations

import json
import subprocess

from hcli.controller import Controller
from hcli.engine import Engine
from hcli.knowledge import KnowledgeStore
from hcli.session import CONTEXT_MEMORY_SCHEMA, Session


def _git(root, *args):
    return subprocess.run(
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )


def test_session_round_trips_a_bounded_memory_checkpoint(tmp_path):
    session = Session(session_id="s1")
    for index in range(80):
        session.append_message("user", "x" * 1000 + str(index))
    assert len(session.messages) == 64
    session.set_memory(
        {
            "schema": CONTEXT_MEMORY_SCHEMA,
            "generation": 1,
            "staging": {"staged": {"count": 1}},
            "recent": [{"content": "r" * 1000}],
        }
    )
    restored = Session.from_dict(session.to_dict())
    assert restored.memory["staging"]["staged"]["count"] == 1
    assert restored.compaction_count == 0


def test_compaction_remembers_goal_steering_and_git_index_state(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "hcli@example.invalid")
    _git(tmp_path, "config", "user.name", "HCLI Test")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    tracked.write_text("staged plus worktree\n", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("new\n", encoding="utf-8")

    controller = Controller(tmp_path)
    try:
        controller.set_goal("Keep staged work safe while improving context compression.")
        controller.queue_steer("constraint: never discard staged changes")
        controller.session.append_message("user", "old discussion")
        controller.session.append_message("assistant", "old result")
        memory = controller.compact_context()
        assert memory["schema"] == CONTEXT_MEMORY_SCHEMA
        assert memory["active_goal"]["text"].startswith("Keep staged")
        assert memory["steering"]["events"][0]["kind"] == "constraint"
        assert memory["staging"]["staged"]["count"] == 1
        assert memory["staging"]["unstaged"]["count"] == 1
        assert memory["staging"]["untracked"]["count"] >= 1
        assert "untracked.txt" in memory["staging"]["untracked"]["paths"]
        assert memory["history_archive"]["records"] == 2
        assert controller.session_store.history_path(controller.session.id).is_file()
        assert len(controller.session_store.load_history(controller.session.id)) == 2
        assert len(controller.session.messages) == 2
        assert "memory=checkpoint#1" in controller.context_summary()

        loaded = controller.session_store.load(controller.session.id)
        assert loaded is not None
        assert loaded.memory["staging"]["staged"]["paths"] == ["tracked.txt"]

        controller.session.append_message("user", "new turn")
        second = controller.compact_context()
        assert second["history_archive"]["records"] == 1
        assert len(controller.session_store.load_history(controller.session.id)) == 3
        assert "memory=checkpoint#2" in controller.context_summary()
    finally:
        controller.shutdown()


def test_checkpoint_is_injected_as_bounded_memory_not_root_goal_replacement(tmp_path):
    from hcli.workspace import Workspace

    engine = Engine(Workspace(str(tmp_path)))
    payload = engine._build_model_payload(
        "continue the task",
        context_memory={
            "schema": CONTEXT_MEMORY_SCHEMA,
            "active_goal": {"text": "original durable goal"},
            "staging": {"staged": {"count": 2}},
        },
    )
    user = payload["messages"][1]["content"]
    assert "continue the task" in user
    assert "DURABLE CONTEXT CHECKPOINT" in user
    assert "original durable goal" in user
    assert len(json.dumps(payload, ensure_ascii=False)) < 20000


def test_prior_knowledge_snapshot_retrieves_relevant_facts_before_recent_noise(tmp_path):
    store = KnowledgeStore(tmp_path)
    store.record_note("unrelated compiler color preference", source="old")
    store.record_note(
        "production overnight campaign must preserve staged changes",
        kind="constraint",
        source="operator",
    )
    store.record_note("unrelated UI wording", source="new")

    snapshot = store.snapshot(limit=2, focus="overnight production campaign")

    assert snapshot["retrieval"]["mode"] == "focus_ranked"
    assert snapshot["records"]
    assert "production overnight campaign" in json.dumps(snapshot["records"][0])


def test_cold_recall_can_recover_a_fact_evicted_from_the_hot_index(tmp_path):
    store = KnowledgeStore(tmp_path)
    store.record_note("old production launch constraint", source="overnight")
    for index in range(30):
        store.record_note(f"unrelated later note {index}", source="noise")

    hot = store.snapshot(limit=24, focus="old production launch")
    cold = store.recall("old production launch", limit=4)

    assert not any("old production launch constraint" in json.dumps(item) for item in hot["records"])
    assert any("old production launch constraint" in json.dumps(item) for item in cold["records"])
    assert cold["retrieval"]["mode"] == "cold_recall"

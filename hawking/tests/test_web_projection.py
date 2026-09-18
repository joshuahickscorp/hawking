from __future__ import annotations

from pathlib import Path

from hawking.goal_surface import create_goal
from hawking.session import Session
from hawking.web_projection import WebEventJournal, project_snapshot
from hawking.web_release import WebReleaseStore


def _view(tmp_path: Path):
    goal = create_goal(tmp_path, objective="disposable recovery fixture")
    session = Session("web-c2", model="hawking/auto")
    session.ui = {"route_scope": "cloud"}
    return project_snapshot(tmp_path, session, [{"goal": goal, "workunit": {"workunit_id": goal["goal_id"], "state": "RUNNING"}, "result": None}])


def test_projection_is_canonical_and_event_replay_is_idempotent(tmp_path: Path) -> None:
    view = _view(tmp_path)
    assert view["schema"] == "hawking.web.contract.v1"
    assert view["goals"][0]["goal_id"].startswith("GOAL-")
    journal = WebEventJournal(tmp_path, "web-c2")
    journal.sync(view)
    first = journal.replay(0)
    assert first["events"]
    journal.sync(view)
    assert journal.replay(0)["events"] == first["events"]
    assert all("sequence" in event and "source_revision" in event for event in first["events"])


def test_session_message_cursor_changes_canonical_projection_event(tmp_path: Path) -> None:
    goal = create_goal(tmp_path, objective="message revision fixture")
    session = Session("web-message", model="hawking/auto")
    journal = WebEventJournal(tmp_path, session.id)
    initial = project_snapshot(tmp_path, session, [{"goal": goal, "workunit": {}, "result": None}])
    journal.sync(initial)
    cursor = journal.replay(0)["through_sequence"]
    session.append_message("user", "durably committed")
    updated = project_snapshot(tmp_path, session, [{"goal": goal, "workunit": {}, "result": None}])
    journal.sync(updated)
    replay = journal.replay(cursor)
    assert any(event["type"] == "session.updated" for event in replay["events"])


def test_goal_projection_preserves_parallel_frontier_without_exposing_scheduler_state(tmp_path: Path) -> None:
    goal = create_goal(tmp_path, objective="parallel lanes")
    goal.update({
        "active_workunit_id": "WU-A",
        "active_workunit_ids": ["WU-A", "WU-B"],
        "runnable_frontier": [
            {"workunit_id": "WU-A", "state": "RUNNABLE", "dependencies": []},
            {"workunit_id": "WU-B", "state": "RUNNABLE", "dependencies": ["WU-PREV"]},
        ],
        "frontier_waiting_dependencies": [
            {"workunit_id": "WU-C", "dependencies": ["WU-MISSING"]},
        ],
    })
    view = project_snapshot(
        tmp_path, Session("web-frontier", model="hawking/auto"),
        [{"goal": goal, "workunit": {"workunit_id": "WU-A", "state": "RUNNING"}, "result": None}],
    )
    projected = view["goals"][0]
    assert projected["active_workunit_id"] == "WU-A"
    assert projected["active_workunit_ids"] == ["WU-A", "WU-B"]
    assert [item["workunit_id"] for item in projected["runnable_frontier"]] == ["WU-A", "WU-B"]
    assert projected["frontier_waiting_dependencies"] == [
        {"workunit_id": "WU-C", "dependencies": ["WU-MISSING"]}
    ]


def test_projection_reports_a_gap_after_bounded_retention(tmp_path: Path) -> None:
    journal = WebEventJournal(tmp_path, "web-gap")
    for index in range(520):
        journal.append(event_type="worker.updated", identity={"session_id": "web-gap", "worker_id": str(index)}, state_rev=f"r{index}", payload={"index": index})
    replay = journal.replay(1)
    assert replay["gap"] is True
    assert replay["snapshot_required"] is True
    assert len(replay["events"]) <= 64


def test_web_release_preserves_current_candidate_previous(tmp_path: Path) -> None:
    source = tmp_path / "review.html"
    source.write_text("<main>one</main>", encoding="utf-8")
    releases = WebReleaseStore(tmp_path, source)
    first = releases.bootstrap()["current"]["release_id"]
    source.write_text("<main>two</main>", encoding="utf-8")
    assert releases.stage()["candidate"]["release_id"] != first
    live = releases.activate()
    assert live["previous"]["release_id"] == first
    assert b"two" in releases.current_bytes()[0]

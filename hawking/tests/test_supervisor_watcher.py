from __future__ import annotations

import json

from hawking.goal_surface import create_goal
from hawking.supervisor_watcher import SupervisorWatcher, batch_worker_results, cache_objective, objective_digest, supervisor_checkpoint
from hawking.auto_orchestration import recommend_concurrency


def test_objective_is_cached_by_digest(tmp_path):
    objective = tmp_path / "objective.md"
    objective.write_text("root objective\n", encoding="utf-8")
    first = cache_objective(tmp_path, "GOAL-WATCH", objective)
    second = cache_objective(tmp_path, "GOAL-WATCH", objective)
    assert first["objective_digest"] == second["objective_digest"]
    assert first["cached_at"] == second["cached_at"]
    objective.write_text("changed objective\n", encoding="utf-8")
    third = cache_objective(tmp_path, "GOAL-WATCH", objective)
    assert third["objective_digest"] != first["objective_digest"]


def test_unchanged_state_is_watcher_only_and_deltas_are_not_duplicated(tmp_path):
    goal = create_goal(str(tmp_path), objective="watch a bounded lane")
    watcher = SupervisorWatcher(tmp_path, goal["goal_id"])
    first = watcher.decide()
    assert first["wake_supervisor"] is True
    assert first["tier"] == "LIGHT_REVIEW"
    assert watcher.record_delta(first) is True
    second = watcher.decide()
    assert second["wake_supervisor"] is False
    assert second["tier"] == "WATCHER"
    assert watcher.record_delta(second) is False
    rows = (tmp_path / f".hawking/supervision/{goal['goal_id']}.deltas.jsonl").read_text().splitlines()
    assert len(rows) == 1


def test_real_event_wakes_without_state_change(tmp_path):
    goal = create_goal(str(tmp_path), objective="watch a bounded lane")
    watcher = SupervisorWatcher(tmp_path, goal["goal_id"])
    watcher.decide()
    decision = watcher.decide(events=["operator.steer"])
    assert decision["wake_supervisor"] is True
    assert "operator.steer" in decision["reasons"]


def test_worker_batch_suppresses_repeated_known_failure_but_wakes_new_class():
    repeated = batch_worker_results([
        {"workunit_id": "W1", "state": "OUTPUT_UNUSABLE", "failure_class": "PROVIDER_OUTPUT_UNUSABLE"},
        {"workunit_id": "W2", "state": "OUTPUT_UNUSABLE", "failure_class": "PROVIDER_OUTPUT_UNUSABLE"},
    ])
    assert repeated["known_failure_only"] is True
    assert repeated["wake_supervisor"] is False
    assert repeated["tier"] == "WATCHER"
    new = batch_worker_results([
        {"workunit_id": "W3", "state": "FAILED", "failure_class": "CREDENTIAL_UNAVAILABLE"},
    ])
    assert new["new_failure_classes"] == ["CREDENTIAL_UNAVAILABLE"]
    assert new["wake_supervisor"] is True


def test_autotuned_concurrency_is_class_sensitive_and_bounded():
    assert recommend_concurrency("research")["target"] == 10
    assert recommend_concurrency("planning")["target"] == 8
    assert recommend_concurrency("repo_engineering")["target"] == 6
    degraded = recommend_concurrency("research", attempts=4, accepted=0, failure_rate=1.0)
    assert degraded["target"] == 9
    assert recommend_concurrency("research", budget_remaining_usd=.10)["target"] == 2


def test_supervisor_checkpoint_is_reference_only(tmp_path):
    objective = tmp_path / "objective.md"
    objective.write_text("do not embed this body\n", encoding="utf-8")
    goal = create_goal(str(tmp_path), objective="compact supervision")
    checkpoint = supervisor_checkpoint(tmp_path, goal["goal_id"], objective_path=objective)
    assert checkpoint["schema"] == "hawking.supervisor_checkpoint.v1"
    assert checkpoint["root"] == goal["goal_id"]
    assert checkpoint["objective_digest"] == objective_digest(str(objective))
    assert "do not embed this body" not in json.dumps(checkpoint)

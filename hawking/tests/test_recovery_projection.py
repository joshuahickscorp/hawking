from __future__ import annotations

from pathlib import Path

from hawking.goal_surface import create_goal
from hawking.mutation import MutationIntentStore, build_mutation_request
from hawking.recovery import reconcile_recovery, recovery_status


def test_recovery_keeps_unverified_mutation_unknown(tmp_path: Path) -> None:
    goal = create_goal(tmp_path, objective="disposable crash boundary")
    request = build_mutation_request(
        tmp_path,
        [{"path": "fixture.txt"}],
        metadata={"request_id": "REQ-c2-unknown", "goal_id": goal["goal_id"]},
    )
    MutationIntentStore(tmp_path).write(request, status="UNKNOWN_OUTCOME", outcome={"stage": "published_before_verification"})
    # Simulate the exact crash boundary: a write reached the worktree, but the
    # process died before it could record/verifiably seal its post-image.
    (tmp_path / "fixture.txt").write_text("published but unverified\n", encoding="utf-8")
    result = reconcile_recovery(tmp_path)
    mutation = next(item for item in result["mutation_intents"] if item["request_id"] == "REQ-c2-unknown")
    assert mutation["status"] == "UNKNOWN_OUTCOME"
    assert mutation["reconciled"] is False
    assert recovery_status(tmp_path)["mutation_intents"][0]["status"] == "UNKNOWN_OUTCOME"

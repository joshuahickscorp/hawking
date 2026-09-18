from __future__ import annotations

from pathlib import Path

from hawking.goal_surface import create_goal, list_goals, write_result
from hawking.persist import atomic_write_json


def test_goal_listing_excludes_contract_and_result_companions(tmp_path: Path) -> None:
    goal = create_goal(tmp_path, objective="one canonical Goal")
    atomic_write_json(tmp_path / ".hawking" / "goals" / f"{goal['goal_id']}.contract.json", {"goal_id": goal["goal_id"]})
    write_result(tmp_path, goal["goal_id"], status="COMPLETE", summary="fixture")
    rows = list_goals(tmp_path)
    assert [row["goal_id"] for row in rows] == [goal["goal_id"]]

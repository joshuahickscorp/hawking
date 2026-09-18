from __future__ import annotations

import pytest

from hawking.sandbox_world import (
    SandboxWorldError,
    assimilate_finding,
    create_branch,
    create_world,
    dispose_branch,
    fail_branch,
    list_worlds,
    load_world,
    record_artifact,
    record_finding,
    register_candidate,
    register_evaluator,
    reserve_budget,
    settle_budget,
    snapshot_world,
)


def test_s0_persists_branches_findings_candidates_and_budget(tmp_path):
    world = create_world(
        tmp_path,
        world_id="S0-sandbox-fixture",
        objective="Find a falsifiable improvement",
        budget_usd=2.0,
        resource_policy={"device": "cpu", "max_parallel": 2},
    )
    alpha = create_branch(
        tmp_path,
        world["world_id"],
        branch_id="branch-alpha",
        worktree_ref=".worktrees/s0-alpha",
        source_revision="deadbeef",
    )
    sibling = create_branch(
        tmp_path,
        world["world_id"],
        branch_id="branch-sibling",
        worktree_ref=".worktrees/s0-sibling",
    )
    evaluator = register_evaluator(
        tmp_path,
        world["world_id"],
        evaluator_id="eval-software-1",
        kind="software",
        selector="pytest -q",
    )
    finding = record_finding(
        tmp_path,
        world["world_id"],
        branch_id=alpha["branch_id"],
        finding="bounded change may reduce a regression",
        falsifier="matching held-out test fails",
        action="run the independent test",
        evidence=["receipt:test-1"],
    )
    artifact = record_artifact(
        tmp_path,
        world["world_id"],
        branch_id=alpha["branch_id"],
        path=".worktrees/s0-alpha/result.json",
        digest="a" * 64,
    )
    candidate = register_candidate(
        tmp_path,
        world["world_id"],
        branch_id=alpha["branch_id"],
        category="software",
        summary="candidate remains unevaluated",
        evaluator_ids=[evaluator["evaluator_id"]],
        artifact_ids=[artifact["artifact_id"]],
    )
    budget = reserve_budget(
        tmp_path,
        world["world_id"],
        amount_usd=1.5,
        branch_id=alpha["branch_id"],
        reason="bounded test",
    )
    assert budget["available_usd"] == pytest.approx(0.5)
    budget = settle_budget(
        tmp_path,
        world["world_id"],
        amount_usd=1.25,
        reserved_usd=1.5,
        branch_id=alpha["branch_id"],
        reason="observed test cost",
    )
    assert budget["spent_usd"] == pytest.approx(1.25)
    assert budget["available_usd"] == pytest.approx(0.75)

    assimilated = assimilate_finding(
        tmp_path,
        world["world_id"],
        finding["finding_id"],
    )
    assert assimilated["assimilated"] is True
    snap = snapshot_world(tmp_path, world["world_id"])
    assert snap["root_persists"] is True
    assert snap["schema"] == "hawking.sandbox.world.v1"
    persisted = load_world(tmp_path, world["world_id"])
    assert persisted["execution_policy"]["deny_by_default"] is True
    assert persisted["candidate_lifecycle"]["promotion_authority"] == "protected_controller"
    assert snap["finding_count"] == 1
    assert snap["candidate_count"] == 1
    assert snap["evaluator_count"] == 1
    assert {row["branch_id"] for row in snap["branches"]} == {
        "branch-alpha", "branch-sibling"
    }
    assert list_worlds(tmp_path)[0]["world_id"] == world["world_id"]
    assert persisted["candidates"][0]["candidate_id"] == candidate["candidate_id"]


def test_child_failure_keeps_s0_root_and_siblings_alive(tmp_path):
    world = create_world(tmp_path, objective="isolate child failure")
    failed = create_branch(
        tmp_path, world["world_id"], branch_id="branch-failed", worktree_ref="failed"
    )
    sibling = create_branch(
        tmp_path, world["world_id"], branch_id="branch-live", worktree_ref="live"
    )
    failure = fail_branch(
        tmp_path,
        world["world_id"],
        failed["branch_id"],
        reason="provider output unusable",
        failure_class="OUTPUT_UNUSABLE",
    )
    assert failure["root_continues"] is True
    after = load_world(tmp_path, world["world_id"])
    assert after["state"] == "ACTIVE"
    assert after["branches"][failed["branch_id"]]["state"] == "FAILED"
    assert after["branches"][sibling["branch_id"]]["state"] == "ACTIVE"
    disposed = dispose_branch(tmp_path, world["world_id"], sibling["branch_id"])
    assert disposed["state"] == "DISPOSED"
    assert snapshot_world(tmp_path, world["world_id"])["child_failure_count"] == 1


def test_s0_refuses_unindependent_or_unbudgeted_authority(tmp_path):
    world = create_world(tmp_path, objective="negative controls", budget_usd=1.0)
    branch = create_branch(
        tmp_path, world["world_id"], branch_id="branch-negative", worktree_ref="negative"
    )
    with pytest.raises(SandboxWorldError, match="independent"):
        register_evaluator(
            tmp_path,
            world["world_id"],
            kind="research",
            owner="sandbox_model",
            independent=False,
        )
    with pytest.raises(SandboxWorldError, match="independent evaluator"):
        register_candidate(
            tmp_path,
            world["world_id"],
            branch_id=branch["branch_id"],
            category="research",
            summary="no evaluator",
            evaluator_ids=[],
        )
    with pytest.raises(SandboxWorldError, match="exceeds available"):
        reserve_budget(tmp_path, world["world_id"], amount_usd=1.01)
    evaluator = register_evaluator(
        tmp_path, world["world_id"], kind="research", evaluator_id="eval-research"
    )
    finding = record_finding(
        tmp_path,
        world["world_id"],
        branch_id=branch["branch_id"],
        finding="a candidate claim",
        falsifier="counterexample",
        action="test it",
    )
    with pytest.raises(SandboxWorldError, match="cannot assimilate"):
        assimilate_finding(
            tmp_path,
            world["world_id"],
            finding["finding_id"],
            controller_id="sandbox_model",
        )
    assert evaluator["independent"] is True
def test_safe_preserves_scalar_values() -> None:
    """Bounded world serialization must return scalars, not raise NameError."""
    import hawking.sandbox_world as sandbox_world

    assert sandbox_world._safe(None) is None
    assert sandbox_world._safe(True) is True
    assert sandbox_world._safe(7) == 7
    assert sandbox_world._safe(2.5) == 2.5
    assert sandbox_world._safe("ok") == "ok"
    assert sandbox_world._safe([1, "two", None]) == [1, "two", None]
    assert sandbox_world._safe({"a": 1, "token": "x"}) == {"a": 1}
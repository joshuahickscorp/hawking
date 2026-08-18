"""Executable continuity proof for sacrificial Genesis promotion."""
from __future__ import annotations

from pathlib import Path

from lab.lineage.bus import ResearchBus
from lab.lineage.canon import labeled_sha
from lab.lineage.continuity import (
    WorkerCheckpointStore,
    genesis_promoted_event,
    migrate_workers,
)
from lab.receipts import verify


def _generation(
    generation: int,
    *,
    artifact: str,
    runtime: str,
    complete_token_ns: int,
) -> dict:
    return {
        "generation": generation,
        "artifact_sha": labeled_sha(artifact),
        "runtime_sha": labeled_sha(runtime),
        "repo_head": "a" * 40,
        "physical_bpw": 4.25,
        "complete_token_ns": complete_token_ns,
    }


def _worker(worker_id: str, session_role: str, task_id: str, generation: dict) -> dict:
    return {
        "worker_id": worker_id,
        "session_role": session_role,
        "task_id": task_id,
        "task_contract": {"id": task_id, "version": 1},
        "priority": 10,
        "state": "READY",
        "durable_task_state": {
            "goal": "improve the current Genesis generation",
            "subgoal": "rebind after protected promotion",
            "repo": "/repo/hawking",
            "worktree": f"/worktrees/{worker_id}",
            "hypotheses": ["identity-coupled assumptions may require revalidation"],
            "findings": [],
            "receipts": [],
            "negative_science": [],
            "pending_experiments": ["measure the replacement on the protected path"],
            "tool_results": [],
            "NEXT_ACTION": "resume the independently assigned task",
        },
        "generation_dependencies": ["artifact", "runtime"],
        "bound_generation": generation,
    }


def test_promotion_checkpoints_and_rebinds_two_workers_with_generator_directives(
    tmp_path: Path,
) -> None:
    old = _generation(
        0,
        artifact="artifact/g0",
        runtime="runtime/g0",
        complete_token_ns=38_000_000,
    )
    new = _generation(
        1,
        artifact="artifact/g1",
        runtime="runtime/g1",
        complete_token_ns=9_000_000,
    )
    event = genesis_promoted_event(old, new)
    verify(event, label="GENESIS_PROMOTED")
    assert event["new_generation"]["generation"] == 1

    bus = ResearchBus()
    bus.publish_generation_aware(
        {
            "type": "HYPOTHESIS",
            "sender": "child-a",
            "artifact_sha": old["artifact_sha"],
            "runtime_sha": old["runtime_sha"],
            "epistemic_class": "HYPOTHESIS",
            "receipt_ref": "",
            "affected_subsystem": "deltanet",
            "body": {"text": "fuse activation tails only after current-main revalidation"},
            "generation": 0,
            "repo_head": old["repo_head"],
            "facet": "kernel",
            "invalidation_condition": "artifact or runtime identity changes",
            "generation_scope": "GENERATION_BOUND",
        }
    )
    directives = (
        {
            "canonical_path": "contracts/genesis/GENESIS_CONTINUITY_DIRECTIVE.md",
            "sha256": labeled_sha("continuity-directive"),
            "size_bytes": 1,
            "integrity_verified": True,
        }
        for _ in range(1)
    )
    result = migrate_workers(
        workers=[
            _worker("gravity", "child_a", "gravity-task", old),
            _worker("kernel", "child_b", "kernel-task", old),
        ],
        old_generation=old,
        new_generation=new,
        store=WorkerCheckpointStore(tmp_path / "checkpoints"),
        directives=directives,
        world_state={"current_artifact": new["artifact_sha"]},
        bus=bus,
        parent_live_before=True,
        parent_live_after=True,
        protected_test_slot_available=True,
    )

    verify(result, label="worker migration")
    assert result["status"] == "PASS"
    assert result["workers_checkpointed"] == result["workers_rebound"] == 2
    assert result["old_parent_may_unload_after_rebind"] is True
    assert {row["classification"] for row in result["workers"]} == {"needs_rebase"}
    assert all(row["resumed_state"] == "READY" for row in result["workers"])
    assert all(row["context_sha256"] for row in result["workers"])
    assert (tmp_path / "checkpoints" / "gravity" / "checkpoint.json").is_file()
    assert (tmp_path / "checkpoints" / "kernel" / "checkpoint.json").is_file()

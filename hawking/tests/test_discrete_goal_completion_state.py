from __future__ import annotations

import json

from hawking.workunit_owner import (
    WorkunitOwner,
    WorkunitRecord,
    _complete_discrete_goal,
    _unaccepted_effect_proposal,
)
from hawking.goal_surface import (
    create_goal,
    load_goal,
    reconcile_goal_frontier,
    update_goal,
)


def test_machine_completion_overrides_stale_provider_outcome(tmp_path) -> None:
    owner = WorkunitOwner(tmp_path)
    record = WorkunitRecord(
        workunit_id="GOAL-COMPLETION-STATE",
        goal_id="GOAL-COMPLETION-STATE",
        state="RUNNING",
        current_phase="BUILD",
        last_outcome="CONTINUE",
        worker_id="hawking-worker-test",
        worker_status="RUNNING",
        staging_workspace=str(tmp_path),
        checkpoint_path=str(tmp_path / "checkpoint.json"),
    )
    owner.save(record)

    _complete_discrete_goal(record, owner)
    restored = owner.load(record.workunit_id)

    assert restored.state == "COMPLETE"
    assert restored.last_outcome == "COMPLETE"
    assert restored.current_phase == "COMPLETE"
    assert restored.worker_status == "RELEASED"
    assert restored.worker_kill_reason == "goal_complete"
    assert restored.exact_next_action == "Goal complete; durable result packet is available"


def test_unaccepted_effect_proposal_stops_outer_owner_replay() -> None:
    proposal = json.dumps({
        "status": "MUTATION_PROPOSED",
        "operations": [{
            "op": "replace",
            "path": "hawking/example.py",
            "old_text": "old = 1",
            "new_text": "new = 2",
        }],
    })
    assert _unaccepted_effect_proposal(proposal, {
        "complete": False,
        "successful_tools": ["fs.read"],
    }) is True
    assert _unaccepted_effect_proposal(proposal, {
        "complete": False,
        "successful_tools": ["repo.edit"],
    }) is False
    assert _unaccepted_effect_proposal(json.dumps({
        "status": "MUTATION_PROPOSED",
        "operations": [{"op": "fs.read", "path": "hawking/example.py"}],
    }), {"complete": False, "successful_tools": ["fs.read"]}) is False
    assert _unaccepted_effect_proposal(json.dumps({
        "s": "MUTATE", "anchor": "SRC-A91F", "op": "replace", "body": "new = 2\n",
    }), {"complete": False, "successful_tools": ["fs.read"]}) is True


def test_compact_terminal_packets_are_machine_outcomes():
    from hawking.workunit_owner import TurnOutcome, parse_turn_outcome
    assert parse_turn_outcome('{"s":"BLOCKED","reason":"MISSING_SOURCE"}') == TurnOutcome.BLOCKED
    assert parse_turn_outcome('{"s":"DONE","evidence":["E-1"]}') == TurnOutcome.COMPLETE


def test_child_workunit_completion_preserves_parent_goal_phase(tmp_path) -> None:
    goal = create_goal(tmp_path, objective="A durable multi-tranche Goal.")
    update_goal(tmp_path, goal["goal_id"], status="RUNNING", phase="E_MACOS_SEMANTIC_ACTION_LEASE")
    owner = WorkunitOwner(tmp_path)
    record = WorkunitRecord(
        workunit_id="E-CHILD-WORKUNIT",
        goal_id=goal["goal_id"],
        state="RUNNING",
        current_phase="BUILD",
        worker_id="hawking-worker-child",
        worker_status="RUNNING",
        staging_workspace=str(tmp_path),
        checkpoint_path=str(tmp_path / "checkpoint.json"),
    )
    owner.save(record)

    _complete_discrete_goal(record, owner, terminal_goal=False)

    restored = owner.load(record.workunit_id)
    parent = load_goal(tmp_path, goal["goal_id"])
    assert restored.state == "COMPLETE"
    assert restored.current_phase == "WORKUNIT_COMPLETE"
    assert parent["status"] == "RUNNING"
    assert parent["phase"] == "E_MACOS_SEMANTIC_ACTION_LEASE"
    assert parent["last_workunit"]["workunit_id"] == record.workunit_id


def test_terminal_output_unusable_child_releases_parent_active_pointer(tmp_path) -> None:
    goal = create_goal(tmp_path, objective="A durable multi-tranche Goal.")
    update_goal(
        tmp_path,
        goal["goal_id"],
        status="RUNNING",
        phase="P1_CONTROL_PLANE_COLLAPSE",
        active_workunit_id="WU-OUTPUT",
        owner_job={"job_id": "job-output", "state": "RUNNING"},
    )
    contract_path = tmp_path / ".hawking" / "goals" / "WU-OUTPUT.contract.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps({
        "goal_id": goal["goal_id"],
        "workunit_id": "WU-OUTPUT",
        "goal_mode": "worker_research",
        "goal_terminal_on_workunit_complete": False,
    }), encoding="utf-8")
    owner = WorkunitOwner(tmp_path)
    owner.save(WorkunitRecord(
        workunit_id="WU-OUTPUT",
        goal_id=goal["goal_id"],
        goal_mode="worker_research",
        state="OUTPUT_UNUSABLE",
        classification="PROVIDER_OUTPUT_UNUSABLE",
        worker_id="hawking-worker-output",
        worker_status="WAITING_RETRY",
        background_job_id="job-output",
        contract_path=str(contract_path),
        checkpoint_path=str(tmp_path / "checkpoint.json"),
        staging_workspace=str(tmp_path),
    ))

    parent = load_goal(tmp_path, goal["goal_id"])
    assert parent["status"] == "RUNNING"
    assert parent["phase"] == "P1_CONTROL_PLANE_COLLAPSE"
    assert parent["active_workunit_id"] is None
    assert parent["owner_job"] is None
    assert parent["last_workunit"]["workunit_id"] == "WU-OUTPUT"
    assert parent["last_workunit"]["status"] == "OUTPUT_UNUSABLE"


def test_transient_provider_transport_pause_keeps_parent_lane_active(tmp_path, monkeypatch) -> None:
    goal = create_goal(tmp_path, objective="A retrying provider lane Goal.")
    update_goal(
        tmp_path,
        goal["goal_id"],
        status="RUNNING",
        active_workunit_id="WU-RETRY",
        active_workunit_ids=["WU-RETRY"],
        owner_job={"job_id": "job-retry", "workunit_id": "WU-RETRY", "state": "RUNNING"},
    )
    contract_path = tmp_path / ".hawking" / "goals" / "WU-RETRY.contract.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps({
        "goal_id": goal["goal_id"],
        "workunit_id": "WU-RETRY",
        "goal_mode": "worker_research",
        "goal_terminal_on_workunit_complete": False,
    }), encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        "hawking.auto_orchestration.refill_runnable_frontier",
        lambda workspace, goal_id: calls.append((workspace, goal_id)) or {"admitted": []},
    )
    owner = WorkunitOwner(tmp_path)
    record = WorkunitRecord(
        workunit_id="WU-RETRY",
        goal_id=goal["goal_id"],
        goal_mode="worker_research",
        state="PAUSED_PROVIDER",
        classification="PROVIDER_TRANSPORT_ERROR",
        worker_id="hawking-worker-retry",
        worker_status="WAITING_RETRY",
        worker_kill_reason="transport_error",
        background_job_id="job-retry",
        contract_path=str(contract_path),
        checkpoint_path=str(tmp_path / "WU-RETRY.checkpoint.json"),
        staging_workspace=str(tmp_path),
    )

    owner._reconcile_parent_after_terminal_child(record)

    parent = load_goal(tmp_path, goal["goal_id"])
    assert parent["active_workunit_ids"] == ["WU-RETRY"]
    assert parent["active_workunit_id"] == "WU-RETRY"
    assert calls == []


def test_terminal_parallel_child_releases_only_its_lane(tmp_path) -> None:
    goal = create_goal(tmp_path, objective="A parallel multi-tranche Goal.")
    update_goal(
        tmp_path,
        goal["goal_id"],
        status="RUNNING",
        phase="P4_AUTO",
        active_workunit_id="WU-A",
        active_workunit_ids=["WU-A", "WU-B"],
        owner_job={"job_id": "job-a", "workunit_id": "WU-A", "state": "RUNNING"},
        active_workunit_jobs=[
            {"job_id": "job-a", "workunit_id": "WU-A", "state": "RUNNING"},
            {"job_id": "job-b", "workunit_id": "WU-B", "state": "RUNNING"},
        ],
        runnable_frontier=[
            {"workunit_id": "WU-A", "state": "RUNNING"},
            {"workunit_id": "WU-B", "state": "RUNNING"},
        ],
    )
    owner = WorkunitOwner(tmp_path)
    for wid, job in (("WU-A", "job-a"), ("WU-B", "job-b")):
        contract_path = tmp_path / ".hawking" / "goals" / f"{wid}.contract.json"
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps({
            "goal_id": goal["goal_id"],
            "workunit_id": wid,
            "goal_mode": "worker_research",
            "goal_terminal_on_workunit_complete": False,
        }), encoding="utf-8")
        owner.save(WorkunitRecord(
            workunit_id=wid,
            goal_id=goal["goal_id"],
            goal_mode="worker_research",
            state="OUTPUT_UNUSABLE",
            classification="PROVIDER_OUTPUT_UNUSABLE",
            worker_id=f"worker-{wid.lower()}",
            worker_status="WAITING_RETRY",
            background_job_id=job,
            contract_path=str(contract_path),
            checkpoint_path=str(tmp_path / f"{wid}.checkpoint.json"),
            staging_workspace=str(tmp_path),
        ))
        parent = load_goal(tmp_path, goal["goal_id"])
        assert parent["status"] == "RUNNING"
        if wid == "WU-A":
            assert parent["active_workunit_ids"] == ["WU-B"]
            assert parent["active_workunit_id"] == "WU-B"
            assert parent["owner_job"]["job_id"] == "job-b"
            assert [item["workunit_id"] for item in parent["runnable_frontier"]] == ["WU-B"]
        else:
            assert parent["active_workunit_ids"] == []
            assert parent["active_workunit_id"] is None
            assert parent["owner_job"] is None
            assert parent["active_workunit_jobs"] == []
            assert parent["runnable_frontier"] == []


def test_dispatch_does_not_resurrect_scalar_stale_child(tmp_path, monkeypatch) -> None:
    from hawking.goal_surface import dispatch_child_workunit

    goal = create_goal(tmp_path, objective="A frontier compatibility Goal.")
    update_goal(
        tmp_path,
        goal["goal_id"],
        status="RUNNING",
        phase="P4_AUTO",
        active_workunit_id="WU-FINISHED",
        active_workunit_ids=[],
        allowed_models=["deepseek/deepseek-v4.1-flash"],
    )

    def fake_start(self, *, contract_path, workunit_id, model, background):
        return {
            "owner": {"checkpoint_path": str(tmp_path / "checkpoint.json")},
            "background": {
                "job_id": "job-new",
                "workunit_id": workunit_id,
                "state": "RUNNING",
            },
        }

    monkeypatch.setattr("hawking.workunit_owner.WorkunitOwner.start", fake_start)
    result = dispatch_child_workunit(
        tmp_path,
        goal["goal_id"],
        objective="Inspect one independent Auto seam.",
        acceptance=["Return bounded read-only evidence."],
        focus_files=["hawking/auto_orchestration.py"],
        goal_mode="worker_research",
        model_override="deepseek/deepseek-v4.1-flash",
        budget_usd=0.1,
        background=True,
    )
    parent = load_goal(tmp_path, goal["goal_id"])
    wid = result["workunit_id"]
    assert parent["active_workunit_ids"] == [wid]
    assert parent["active_workunit_id"] == wid
    assert "WU-FINISHED" not in parent["active_workunit_ids"]
    assert [item["workunit_id"] for item in parent["runnable_frontier"]] == [wid]


def test_terminal_child_releases_then_invokes_auto_refill_hook(tmp_path, monkeypatch) -> None:
    goal = create_goal(tmp_path, objective="A refill-aware multi-tranche Goal.")
    update_goal(
        tmp_path,
        goal["goal_id"],
        status="RUNNING",
        active_workunit_id="WU-TERMINAL",
        active_workunit_ids=["WU-TERMINAL"],
        owner_job={"job_id": "job-terminal", "workunit_id": "WU-TERMINAL"},
    )
    contract_path = tmp_path / ".hawking" / "goals" / "WU-TERMINAL.contract.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps({
        "goal_id": goal["goal_id"],
        "workunit_id": "WU-TERMINAL",
        "goal_mode": "worker_research",
        "goal_terminal_on_workunit_complete": False,
    }), encoding="utf-8")
    owner = WorkunitOwner(tmp_path)
    record = WorkunitRecord(
        workunit_id="WU-TERMINAL",
        goal_id=goal["goal_id"],
        goal_mode="worker_research",
        state="OUTPUT_UNUSABLE",
        classification="PROVIDER_OUTPUT_UNUSABLE",
        worker_id="hawking-worker-terminal",
        worker_status="WAITING_RETRY",
        background_job_id="job-terminal",
        contract_path=str(contract_path),
        checkpoint_path=str(tmp_path / "WU-TERMINAL.checkpoint.json"),
        staging_workspace=str(tmp_path),
    )
    calls = []
    monkeypatch.setattr(
        "hawking.auto_orchestration.refill_runnable_frontier",
        lambda workspace, goal_id: calls.append((workspace, goal_id)) or {"admitted": []},
    )

    owner._reconcile_parent_after_terminal_child(record)

    parent = load_goal(tmp_path, goal["goal_id"])
    assert parent["active_workunit_ids"] == []
    assert calls == [(tmp_path.resolve(), goal["goal_id"])]


def test_terminal_child_reconciles_auto_queue_without_promoting_provider_output(tmp_path, monkeypatch) -> None:
    goal = create_goal(tmp_path, objective="A queue-reconciled multi-tranche Goal.")
    update_goal(
        tmp_path,
        goal["goal_id"],
        status="RUNNING",
        active_workunit_id="WU-QUEUE",
        active_workunit_ids=["WU-QUEUE"],
        graph_runnable_frontier=["P-QUEUE"],
        graph_frontier_state={"P-QUEUE": {"state": "RUNNABLE_FRESH_ROUTE"}},
        auto_refill_queue=[{
            "tranche_id": "P-QUEUE",
            "state": "ADMITTED",
            "workunit_id": "WU-QUEUE",
            "writer_scope": "queue-scope",
        }],
    )
    contract_path = tmp_path / ".hawking" / "goals" / "WU-QUEUE.contract.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps({
        "goal_id": goal["goal_id"],
        "workunit_id": "WU-QUEUE",
        "goal_mode": "worker_research",
        "goal_terminal_on_workunit_complete": False,
    }), encoding="utf-8")
    monkeypatch.setattr(
        "hawking.auto_orchestration.refill_runnable_frontier",
        lambda workspace, goal_id: {"admitted": []},
    )
    owner = WorkunitOwner(tmp_path)
    record = WorkunitRecord(
        workunit_id="WU-QUEUE",
        goal_id=goal["goal_id"],
        goal_mode="worker_research",
        state="COMPLETE",
        classification="",
        worker_id="hawking-worker-queue",
        worker_status="RELEASED",
        background_job_id="job-queue",
        contract_path=str(contract_path),
        checkpoint_path=str(tmp_path / "WU-QUEUE.checkpoint.json"),
        staging_workspace=str(tmp_path),
    )

    owner._reconcile_parent_after_terminal_child(record)

    parent = load_goal(tmp_path, goal["goal_id"])
    queue = parent["auto_refill_queue"][0]
    assert parent["active_workunit_ids"] == []
    assert queue["state"] == "COMPLETE_PENDING_ACCEPTANCE"
    assert parent["graph_runnable_frontier"] == []
    assert parent["graph_frontier_state"]["P-QUEUE"]["state"] == (
        "WORKUNIT_COMPLETE_PENDING_ACCEPTANCE"
    )
    assert parent["graph_frontier_state"]["P-QUEUE"]["provider_output_promotion"] == "withheld"


def test_reconcile_frontier_discovers_queued_lanes_and_respects_dependencies(tmp_path) -> None:
    goal = create_goal(tmp_path, objective="A dependency-aware frontier Goal.")
    update_goal(
        tmp_path,
        goal["goal_id"],
        status="RUNNING",
        phase="P3_MEMORY_ARTIFACT_SOURCE",
        workunit_ids=["WU-QUEUED", "WU-DONE", "WU-DEPENDENT", "WU-WAITING", "WU-STALE"],
        active_workunit_id="WU-STALE",
        active_workunit_ids=["WU-STALE"],
        active_workunit_jobs=[{"workunit_id": "WU-STALE", "job_id": "job-stale"}],
    )
    owner = WorkunitOwner(tmp_path)
    common = {
        "goal_id": goal["goal_id"],
        "goal_mode": "worker_research",
        "worker_model": "deepseek/deepseek-v4.1-flash",
        "staging_workspace": str(tmp_path),
        "checkpoint_path": str(tmp_path / "checkpoints" / "{workunit_id}.json"),
    }
    owner.save(WorkunitRecord(
        workunit_id="WU-QUEUED", state="QUEUED",
        checkpoint_path=common["checkpoint_path"].format(workunit_id="WU-QUEUED"), **{
            key: value for key, value in common.items() if key != "checkpoint_path"
        },
    ))
    owner.save(WorkunitRecord(
        workunit_id="WU-DONE", state="COMPLETE",
        checkpoint_path=common["checkpoint_path"].format(workunit_id="WU-DONE"), **{
            key: value for key, value in common.items() if key != "checkpoint_path"
        },
    ))
    owner.save(WorkunitRecord(
        workunit_id="WU-DEPENDENT", state="QUEUED",
        parent_workunit_id="WU-DONE",
        checkpoint_path=common["checkpoint_path"].format(workunit_id="WU-DEPENDENT"), **{
            key: value for key, value in common.items() if key != "checkpoint_path"
        },
    ))
    owner.save(WorkunitRecord(
        workunit_id="WU-WAITING", state="QUEUED",
        parent_workunit_id="WU-MISSING",
        checkpoint_path=common["checkpoint_path"].format(workunit_id="WU-WAITING"), **{
            key: value for key, value in common.items() if key != "checkpoint_path"
        },
    ))
    owner.save(WorkunitRecord(
        workunit_id="WU-STALE", state="OUTPUT_UNUSABLE",
        checkpoint_path=common["checkpoint_path"].format(workunit_id="WU-STALE"), **{
            key: value for key, value in common.items() if key != "checkpoint_path"
        },
    ))

    projection = reconcile_goal_frontier(tmp_path, goal["goal_id"])
    parent = load_goal(tmp_path, goal["goal_id"])
    assert [item["workunit_id"] for item in projection["frontier"]] == [
        "WU-QUEUED", "WU-DEPENDENT",
    ]
    assert [item["workunit_id"] for item in projection["waiting"]] == ["WU-WAITING"]
    assert parent["active_workunit_ids"] == []
    assert parent["active_workunit_id"] is None
    assert parent["active_workunit_jobs"] == []
    assert [item["workunit_id"] for item in parent["runnable_frontier"]] == [
        "WU-QUEUED", "WU-DEPENDENT",
    ]


def test_reconcile_frontier_ignores_absent_dependency_metadata(tmp_path) -> None:
    goal = create_goal(tmp_path, objective="A frontier Goal with an implicit root lane.")
    update_goal(
        tmp_path,
        goal["goal_id"],
        status="RUNNING",
        phase="P0_REBIND",
        workunit_ids=["WU-ROOT-LANE"],
    )
    contract_path = tmp_path / ".hawking" / "goals" / "WU-ROOT-LANE.contract.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps({
        "goal_id": goal["goal_id"],
        "workunit_id": "WU-ROOT-LANE",
        "goal_mode": "worker_research",
    }), encoding="utf-8")
    WorkunitOwner(tmp_path).save(WorkunitRecord(
        workunit_id="WU-ROOT-LANE",
        goal_id=goal["goal_id"],
        goal_mode="worker_research",
        state="QUEUED",
        contract_path=str(contract_path),
        checkpoint_path=str(tmp_path / "WU-ROOT-LANE.checkpoint.json"),
        staging_workspace=str(tmp_path),
    ))

    projection = reconcile_goal_frontier(tmp_path, goal["goal_id"])

    assert [item["workunit_id"] for item in projection["frontier"]] == ["WU-ROOT-LANE"]
    assert projection["waiting"] == []


def test_relative_weight_artifact_is_not_a_remote_openrouter_model() -> None:
    from hawking.goal_surface import (
        _is_remote_openrouter_model,
        _openrouter_model_id,
    )

    # A relative path to a local artifact carries no provider authority.
    assert _is_remote_openrouter_model("models/hawking-p0.gguf") is False
    assert _is_remote_openrouter_model("weights/shard-1.safetensors") is False
    assert _is_remote_openrouter_model("runs/model.onnx") is False
    assert _is_remote_openrouter_model("runs/model.bin") is False
    assert _is_remote_openrouter_model("cache/model.npz") is False

    # An explicit, well-formed OpenRouter id is still recognized and unwrapped.
    assert _is_remote_openrouter_model("openrouter:moonshotai/kimi-k3") is True
    assert _openrouter_model_id("openrouter:moonshotai/kimi-k3") == "moonshotai/kimi-k3"
    assert _openrouter_model_id("moonshotai/kimi-k3") == "moonshotai/kimi-k3"

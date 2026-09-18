from __future__ import annotations

import json

from hawking.auto_orchestration import (
    BASE_MAX_PREMIUM_WORKERS,
    BASE_MAX_REMOTE_COGNITION_WORKERS,
    FRONTIER_BURST_FACTOR,
    MAX_ROUTE_REOPEN_DEPTH,
    MAX_AUTOTUNED_WORKERS,
    MAX_PREMIUM_WORKERS,
    MAX_REMOTE_COGNITION_WORKERS,
    build_cognition_plan,
    classify_request,
    packet_path,
    persist_worker_packet,
    record_model_outcome,
    refill_runnable_frontier,
    select_active_diversity_route,
    model_routing_value,
    select_measured_model,
    worker_packet,
)
from hawking.auto_orchestration import (
    _materialize_auto_continuations,
    _authoritative_active_model_counts,
    _reconcile_auto_queue_workunits,
    _route_reopen_compact_id,
    _route_reopen_depth,
    _shared_acceptance_circuit,
    _with_fresh_route_instruction,
)
from hawking.auto_mode import REMOTE_AUTO_ROSTER
from hawking.goal_surface import create_goal, dispatch_child_workunit, load_goal, update_goal


def test_qualified_alternate_model_route_legacy_is_canonical_alias():
    """P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2 regression guard.

    The legacy predicate name must resolve to the same function object as the
    canonical qualified-route predicate, and both must reject the bare
    ``openrouter:`` transport prefix while admitting a resolvable remote id.
    """
    import hawking.goal_surface as gs
    assert gs._is_qualified_alternate_model_route_legacy is (
        gs._is_qualified_alternate_model_route
    )
    assert not gs._is_qualified_alternate_model_route_legacy("openrouter:")
    assert not gs._is_qualified_alternate_model_route_legacy("")
    assert gs._is_qualified_alternate_model_route_legacy(
        "openrouter:deepseek/deepseek-v4.1-flash"
    )
    assert gs._is_qualified_alternate_model_route_legacy(
        "deepseek/deepseek-v4.1-flash"
    )


def test_auto_queue_reconciles_terminal_workunit_before_refill(tmp_path):
    goal = {
        "goal_id": "GOAL-QUEUE-RECONCILE",
        "auto_refill_queue": [{
            "tranche_id": "P18-AUTO",
            "state": "ADMITTED",
            "workunit_id": "WORKUNIT-TERMINAL",
        }],
    }
    workunits = tmp_path / ".hawking" / "workunits"
    workunits.mkdir(parents=True)
    (workunits / "WORKUNIT-TERMINAL.json").write_text(json.dumps({
        "state": "OUTPUT_UNUSABLE",
        "classification": "MUTATION_PAYLOAD_MISSING",
    }), encoding="utf-8")

    updated, reconciled = _reconcile_auto_queue_workunits(tmp_path, goal)

    assert reconciled == [{
        "tranche_id": "P18-AUTO",
        "workunit_id": "WORKUNIT-TERMINAL",
        "projected_state": "TERMINAL_PROVIDER_OUTCOME",
        "classification": "MUTATION_PAYLOAD_MISSING",
    }]
    assert updated["auto_refill_queue"][0]["state"] == "TERMINAL_PROVIDER_OUTCOME"
    assert updated["auto_refill_queue"][0]["released_workunit_state"] == "OUTPUT_UNUSABLE"


def test_active_frontier_capacity_has_a_verified_twenty_five_percent_step():
    assert FRONTIER_BURST_FACTOR == 1.25
    assert MAX_REMOTE_COGNITION_WORKERS is None
    assert MAX_PREMIUM_WORKERS is None
    assert MAX_AUTOTUNED_WORKERS == MAX_REMOTE_COGNITION_WORKERS
    assert BASE_MAX_REMOTE_COGNITION_WORKERS * FRONTIER_BURST_FACTOR == 15
    assert BASE_MAX_PREMIUM_WORKERS * FRONTIER_BURST_FACTOR == 5


def test_auto_scales_topology_from_deterministic_signals():
    simple = build_cognition_plan({"messages": [{"role": "user", "content": "fix one parser test"}]})
    assert simple["topology"] == "single worker"
    complex_plan = build_cognition_plan({
        "messages": [{
            "role": "user",
            "content": "Design the long-horizon Odyssey architecture, decompose it, implement it, and review the result with independent audits.",
        }]
    })
    assert complex_plan["topology"] == (
        "parallel development lanes; writers require disjoint effect scopes"
    )
    assert complex_plan["task"]["task_class"] == "repo_engineering"
    assert len(complex_plan["workers"]) == 6
    assert classify_request(text="multimodal document review")["task_class"] == "multimodal"


def test_worker_packet_is_redacted_and_persisted(tmp_path):
    goal = {
        "goal_id": "GOAL-PACKET",
        "objective": "Use Bearer secret-value and inspect the fixture",
        "acceptance": ["return evidence"],
        "authority_boundary": "isolated worktree",
        "budget_plan": {"build_usd": 1.0},
    }
    workunit = {
        "goal_id": "GOAL-PACKET",
        "workunit_id": "WU-PACKET",
        "worker_id": "hawking-worker-test",
        "worker_model": "deepseek/deepseek-v4.1-flash",
        "objective": goal["objective"],
        "state": "QUEUED",
    }
    packet = worker_packet(goal, workunit, role="implementer", plan_id="PLAN-1")
    assert "Bearer [redacted]" in packet["root_goal"]
    assert "secret-value" not in json.dumps(packet)
    path = persist_worker_packet(tmp_path, packet)
    assert path == packet_path(tmp_path, "WU-PACKET")
    assert json.loads(path.read_text(encoding="utf-8"))["digest"] == packet["digest"]


def test_model_outcome_is_durable(tmp_path):
    row = record_model_outcome(
        tmp_path,
        model="deepseek/deepseek-v4.1-flash",
        task_class="repo_engineering",
        accepted=True,
        cost_usd=0.0123,
        repaired=True,
    )
    assert row["attempts"] == 1
    assert row["accepted"] == 1
    assert json.loads(
        (tmp_path / ".hawking" / "auto" / "model-performance.json").read_text(encoding="utf-8")
    )["models"]["deepseek/deepseek-v4.1-flash::repo_engineering"]["repaired"] == 1


def test_model_outcome_persists_quality_and_transport_dimensions(tmp_path):
    record_model_outcome(
        tmp_path,
        model="z-ai/glm-5.3",
        task_class="source_mapping",
        accepted=False,
        wall_time_s=12.5,
        tool_compliance=False,
        source_grounding_quality=0.4,
        provider_semantic_success=False,
        terminal_packet_success=False,
        provider_transport_failure=True,
        failure_class="PROVIDER_OUTPUT_UNUSABLE",
        independence_value=0.8,
    )
    row = json.loads(
        (tmp_path / ".hawking" / "auto" / "model-performance.json").read_text(encoding="utf-8")
    )["models"]["z-ai/glm-5.3::source_mapping"]
    assert row["tool_compliance_samples"] == 1
    assert row["tool_compliance_successes"] == 0
    assert row["source_grounding_sum"] == 0.4
    assert row["provider_semantic_samples"] == 1
    assert row["terminal_packet_successes"] == 0
    assert row["provider_transport_failures"] == 1
    assert row["independence_value_sum"] == 0.8


def test_unmeasured_glm_gets_a_comparable_trial_before_rebalance(tmp_path):
    selected, route = select_measured_model(
        "source_mapping",
        REMOTE_AUTO_ROSTER,
        workspace=tmp_path,
        preferred="deepseek/deepseek-v4.1-flash",
        role="source_mapping",
    )
    assert selected == "z-ai/glm-5.3"
    assert route["reason"] == "unmeasured_glm_trial"
    assert route["value"]["exploration_eligible"] is True


def test_measured_value_can_rebalance_a_core_task_class(tmp_path):
    for _ in range(3):
        record_model_outcome(
            tmp_path,
            model="z-ai/glm-5.3",
            task_class="source_mapping",
            accepted=True,
            cost_usd=0.001,
            wall_time_s=1.0,
            tool_compliance=True,
            source_grounding_quality=1.0,
            provider_semantic_success=True,
            terminal_packet_success=True,
        )
    for _ in range(3):
        record_model_outcome(
            tmp_path,
            model="deepseek/deepseek-v4.1-flash",
            task_class="source_mapping",
            accepted=False,
            cost_usd=0.2,
            wall_time_s=60.0,
            tool_compliance=False,
            source_grounding_quality=0.2,
            provider_semantic_success=False,
            terminal_packet_success=False,
            failure_class="OUTPUT_UNUSABLE",
        )
    selected, route = select_measured_model(
        "source_mapping",
        REMOTE_AUTO_ROSTER,
        workspace=tmp_path,
        preferred="deepseek/deepseek-v4.1-flash",
        role="source_mapping",
    )
    assert selected == "z-ai/glm-5.3"
    assert route["reason"] == "measured_value"
    assert route["selected"]["routing_value"] > route["preferred_value"]["routing_value"]


def test_semantic_failure_pool_rotates_by_least_exposure(tmp_path):
    for model, attempts in (
        ("deepseek/deepseek-v4.1-flash", 4),
        ("moonshotai/kimi-k3", 2),
    ):
        for _ in range(attempts):
            record_model_outcome(
                tmp_path,
                model=model,
                task_class="routine_implementation",
                accepted=False,
                cost_usd=0.01,
                wall_time_s=2.0,
                provider_semantic_success=False,
                terminal_packet_success=False,
                failure_class="MUTATION_PAYLOAD_MISSING",
            )
    selected, route = select_measured_model(
        "routine_implementation",
        ["deepseek/deepseek-v4.1-flash", "moonshotai/kimi-k3"],
        workspace=tmp_path,
        preferred="deepseek/deepseek-v4.1-flash",
        role="implementation",
    )
    assert selected == "moonshotai/kimi-k3"
    assert route["reason"] == "semantic_failure_diversification"


def test_semantic_failure_pool_prefers_prior_accepted_route_over_novelty(tmp_path):
    for _ in range(4):
        record_model_outcome(
            tmp_path,
            model="deepseek/deepseek-v4.1-flash",
            task_class="routine_implementation",
            accepted=True,
            provider_semantic_success=False,
            terminal_packet_success=False,
        )
    for _ in range(2):
        record_model_outcome(
            tmp_path,
            model="moonshotai/kimi-k3",
            task_class="routine_implementation",
            accepted=False,
            provider_semantic_success=False,
            terminal_packet_success=False,
            failure_class="MUTATION_PAYLOAD_MISSING",
        )

    selected, route = select_measured_model(
        "routine_implementation",
        ["deepseek/deepseek-v4.1-flash", "moonshotai/kimi-k3"],
        workspace=tmp_path,
        preferred="moonshotai/kimi-k3",
        role="implementation",
    )

    assert selected == "deepseek/deepseek-v4.1-flash"
    assert route["reason"] == "semantic_failure_diversification"


def test_escalation_selection_is_explicit_and_information_gain_bounded(tmp_path):
    selected, route = select_measured_model(
        "falsification",
        REMOTE_AUTO_ROSTER,
        workspace=tmp_path,
        preferred="deepseek/deepseek-v4.1-flash",
        role="adversarial falsifier",
        escalation=True,
    )
    assert selected == "nvidia/nemotron-3-ultra-550b-a55b"
    assert route["reason"] == "information_gain_escalation"


def test_child_workunit_contract_carries_empirical_task_class(tmp_path, monkeypatch):
    goal = create_goal(
        tmp_path,
        objective="A durable measured-routing goal.",
        budget_usd=5.0,
        allowed_models=list(REMOTE_AUTO_ROSTER),
    )
    update_goal(tmp_path, goal["goal_id"], status="RUNNING")
    captured = {}

    def fake_dispatch(root, child, *, model, background):
        captured.update(child)
        return {"workunit_id": child["workunit_id"], "owner": {"state": "QUEUED"}}

    monkeypatch.setattr("hawking.goal_surface.dispatch_goal", fake_dispatch)
    result = dispatch_child_workunit(
        tmp_path,
        goal["goal_id"],
        objective="Map current source owners and focused tests.",
        acceptance=["return bounded source evidence"],
        goal_mode="worker_research",
        model_override="z-ai/glm-5.3",
    )
    assert result["auto_route"]["task_class"] == "repo_engineering"
    assert captured["task_class"] == "repo_engineering"
    assert captured["worker_role"] == "implementer"


def test_child_workunit_contract_preserves_explicit_measured_task_class(tmp_path, monkeypatch):
    goal = create_goal(
        tmp_path,
        objective="A durable measured-routing goal.",
        budget_usd=5.0,
        allowed_models=list(REMOTE_AUTO_ROSTER),
    )
    update_goal(tmp_path, goal["goal_id"], status="RUNNING")
    captured = {}

    def fake_dispatch(root, child, *, model, background):
        captured.update(child)
        return {"workunit_id": child["workunit_id"], "owner": {"state": "QUEUED"}}

    monkeypatch.setattr("hawking.goal_surface.dispatch_goal", fake_dispatch)
    result = dispatch_child_workunit(
        tmp_path,
        goal["goal_id"],
        objective="Inspect a bounded architecture seam.",
        acceptance=["return bounded source evidence"],
        goal_mode="worker_research",
        task_class="architecture_planning",
        model_override="moonshotai/kimi-k3",
    )
    assert result["auto_route"]["task_class"] == "architecture_planning"
    assert captured["task_class"] == "architecture_planning"


def test_child_workunit_auto_override_uses_measured_selection(tmp_path, monkeypatch):
    goal = create_goal(
        tmp_path,
        objective="A queued measured-route goal.",
        budget_usd=5.0,
        allowed_models=list(REMOTE_AUTO_ROSTER),
    )
    update_goal(tmp_path, goal["goal_id"], status="RUNNING")
    captured = {}

    def fake_dispatch(root, child, *, model, background):
        captured.update(child)
        captured["model"] = model
        return {"workunit_id": child["workunit_id"], "owner": {"state": "QUEUED"}}

    monkeypatch.setattr("hawking.goal_surface.dispatch_goal", fake_dispatch)
    result = dispatch_child_workunit(
        tmp_path,
        goal["goal_id"],
        objective="Inspect a bounded implementation seam.",
        acceptance=["return bounded evidence"],
        goal_mode="discrete_goal",
        model_override="hawking-auto",
    )
    assert captured["model"] in REMOTE_AUTO_ROSTER
    assert result["auto_route"]["selection"] == "hawking_auto_measured_override"


def test_auto_refill_admits_highest_priority_nonconflicting_ready_lanes(tmp_path, monkeypatch):
    goal = create_goal(tmp_path, objective="A durable V2 frontier.", budget_usd=5.0)
    update_goal(
        tmp_path,
        goal["goal_id"],
        status="RUNNING",
        allowed_models=["deepseek/deepseek-v4.1-flash", "moonshotai/kimi-k3"],
        graph_runnable_frontier=["P-APPLE", "P-NVIDIA", "P-APPLE-REVIEW"],
        auto_refill_queue=[
            {
                "tranche_id": "P-APPLE",
                "priority": 100,
                "writer_scope": "apple",
                "objective": "Read-only Apple backend map.",
                "acceptance": ["return structured source evidence"],
                "goal_mode": "worker_research",
                "model_override": "deepseek/deepseek-v4.1-flash",
            },
            {
                "tranche_id": "P-APPLE-REVIEW",
                "priority": 90,
                "writer_scope": "apple",
                "objective": "Independent Apple backend review.",
                "acceptance": ["return a falsifier"],
                "goal_mode": "worker_research",
                "model_override": "moonshotai/kimi-k3",
            },
            {
                "tranche_id": "P-NVIDIA",
                "priority": 80,
                "writer_scope": "nvidia",
                "objective": "Read-only NVIDIA backend map.",
                "acceptance": ["return structured source evidence"],
                "goal_mode": "worker_research",
                "model_override": "deepseek/deepseek-v4.1-flash",
            },
        ],
    )

    def fake_start(self, *, contract_path, workunit_id, model, background):
        return {
            "owner": {
                "workunit_id": workunit_id,
                "state": "RUNNING",
                "worker_model": model,
                "worker_id": f"worker-{workunit_id}",
                "checkpoint_path": str(tmp_path / f"{workunit_id}.checkpoint.json"),
            },
            "background": {"job_id": f"job-{workunit_id}", "state": "RUNNING"},
        }

    monkeypatch.setattr("hawking.workunit_owner.WorkunitOwner.start", fake_start)
    result = refill_runnable_frontier(tmp_path, goal["goal_id"], efficient_concurrency=3)

    assert [item["tranche_id"] for item in result["admitted"]] == ["P-APPLE", "P-NVIDIA"]
    assert any(
        item["tranche_id"] == "P-APPLE-REVIEW" and item["reason"] == "writer_scope_conflict"
        for item in result["skipped"]
    )
    parent = load_goal(tmp_path, goal["goal_id"])
    assert len(parent["active_workunit_ids"]) == 2
    queue = {item["tranche_id"]: item for item in parent["auto_refill_queue"]}
    assert queue["P-APPLE"]["state"] == "ADMITTED"
    assert queue["P-NVIDIA"]["state"] == "ADMITTED"
    assert queue["P-APPLE-REVIEW"].get("state", "READY") == "READY"


def test_auto_refill_fences_distinct_writer_scopes_that_target_same_file(tmp_path, monkeypatch):
    goal = create_goal(tmp_path, objective="A focus-safe frontier.", budget_usd=5.0)
    update_goal(
        tmp_path,
        goal["goal_id"],
        status="RUNNING",
        graph_runnable_frontier=["P-FIRST", "P-SECOND"],
        auto_refill_queue=[
            {
                "tranche_id": "P-FIRST",
                "priority": 100,
                "writer_scope": "first-scope",
                "focus_files": ["hawking/serve.py"],
                "objective": "First bounded server change.",
                "acceptance": ["return evidence"],
            },
            {
                "tranche_id": "P-SECOND",
                "priority": 90,
                "writer_scope": "second-scope",
                "focus_files": ["hawking/serve.py"],
                "objective": "Second bounded server change.",
                "acceptance": ["return evidence"],
            },
        ],
    )

    monkeypatch.setattr(
        "hawking.goal_surface.dispatch_child_workunit",
        lambda *args, **kwargs: {"workunit_id": "WORKUNIT-FIRST", "auto_route": {}},
    )
    result = refill_runnable_frontier(tmp_path, goal["goal_id"], efficient_concurrency=None)

    assert result["admitted"] == [{
        "tranche_id": "P-FIRST",
        "workunit_id": "WORKUNIT-FIRST",
        "writer_scope": "first-scope",
    }]
    assert any(
        item["tranche_id"] == "P-SECOND"
        and item["reason"] == "focus_file_conflict"
        and item["paths"] == ["hawking/serve.py"]
        for item in result["skipped"]
    )


def test_auto_refill_projects_accepted_dependencies_without_planning_contracts(tmp_path, monkeypatch):
    goal = create_goal(tmp_path, objective="A durable graph projection.", budget_usd=5.0)
    update_goal(
        tmp_path,
        goal["goal_id"],
        status="RUNNING",
        graph={"nodes": [
            {"id": "P-ROOT", "dependencies": []},
            {"id": "P-READY", "dependencies": ["P-ROOT"]},
            {"id": "P-NO-CONTRACT", "dependencies": ["P-ROOT"]},
        ]},
        accepted_heads=[{"tranche": "P-ROOT"}],
        graph_frontier_state={"P-ROOT": {"state": "ACCEPTED_READ_EVIDENCE"}},
        graph_runnable_frontier=[],
        auto_refill_queue=[{
            "tranche_id": "P-READY",
            "priority": 100,
            "writer_scope": "ready",
            "objective": "Use the already-planned contract.",
            "acceptance": ["return bounded evidence"],
        }],
    )

    monkeypatch.setattr(
        "hawking.goal_surface.dispatch_child_workunit",
        lambda *args, **kwargs: {"workunit_id": "WORKUNIT-READY", "auto_route": {}},
    )
    result = refill_runnable_frontier(tmp_path, goal["goal_id"], efficient_concurrency=None)

    assert result["admitted"] == [{
        "tranche_id": "P-READY",
        "workunit_id": "WORKUNIT-READY",
        "writer_scope": "ready",
    }]
    parent = load_goal(tmp_path, goal["goal_id"])
    assert parent["graph_runnable_frontier"] == ["P-READY", "P-NO-CONTRACT"]
    assert parent["graph_frontier_state"]["P-NO-CONTRACT"]["state"] == "RUNNABLE_AWAITING_CONTRACT"


def test_opted_in_auto_continuation_materializes_disjoint_implementation_contracts():
    goal = {
        "goal_id": "GOAL-CONTINUATION",
        "status": "RUNNING",
        "accepted_heads": [
            {"tranche": "P18_TUNNEL", "state": "ACCEPTED_READ_EVIDENCE"},
            {"tranche": "P19_SANDBOX", "state": "ACCEPTED_READ_EVIDENCE"},
        ],
        "graph": {"nodes": [
            {"id": "P18_TUNNEL", "dependencies": []},
            {"id": "P19_SANDBOX", "dependencies": []},
        ]},
        "auto_refill_queue": [
            {"tranche_id": "P18_TUNNEL", "state": "ACCEPTED_READ_EVIDENCE", "priority": 90},
            {"tranche_id": "P19_SANDBOX", "state": "ACCEPTED_READ_EVIDENCE", "priority": 80},
        ],
        "auto_continuation_policy": {
            "enabled": True,
            "source_tranches": ["P18_TUNNEL", "P19_SANDBOX"],
            "max_generated_per_refill": 2,
            "acceptance_canary_generation": "V1",
        },
    }

    updated, receipt = _materialize_auto_continuations(goal)

    assert [item["source_tranche"] for item in receipt["generated"]] == [
        "P18_TUNNEL", "P19_SANDBOX",
    ]
    generated = {
        item["tranche_id"]: item
        for item in updated["auto_refill_queue"]
        if item.get("generated_by")
    }
    assert set(generated) == {
        "P18_TUNNEL_AUTO_IMPLEMENTATION_V1",
        "P19_SANDBOX_AUTO_IMPLEMENTATION_V1",
    }
    assert {item["writer_scope"] for item in generated.values()} == {
        "auto-continuation/p18_tunnel",
        "auto-continuation/p19_sandbox",
    }
    assert all(item["goal_mode"] == "discrete_goal" for item in generated.values())
    assert all(item["dependencies"] == [item["source_tranche"]] for item in generated.values())
    assert all("substantive implementation" in item["objective"] for item in generated.values())
    assert all("qualification-support" not in item["objective"] for item in generated.values())
    assert all(item["mutation_plan"]["require_substantive_behavior"] is True for item in generated.values())
    assert all(item["mutation_plan"]["allow_existing_regression_tests"] is False for item in generated.values())
    assert all(item["mutation_plan"]["acceptance_canary"] is True for item in generated.values())
    assert all(item["mutation_plan"]["acceptance_canary_generation"] == "V1" for item in generated.values())
    assert all(
        item["focus_tests"] == [
            f"hawking/tests/test_auto_{item['source_tranche'].lower()}_regression.py"
        ]
        for item in generated.values()
    )
    assert all(
        "do not add phase, contract, qualification, or version markers as the implementation"
        in item["mutation_plan"]["non_goals"]
        for item in generated.values()
    )
    assert {node["id"] for node in updated["graph"]["nodes"]} >= set(generated)


def test_shared_acceptance_circuit_requires_a_later_accepted_workunit_to_close(tmp_path):
    workunits = tmp_path / ".hawking" / "workunits"
    workunits.mkdir(parents=True)
    for index in range(3):
        (workunits / f"WORKUNIT-REJECT-{index}.json").write_text(json.dumps({
            "goal_id": "GOAL-CIRCUIT",
            "workunit_id": f"WORKUNIT-REJECT-{index}",
            "state": "OUTPUT_UNUSABLE",
            "classification": "MUTATION_PROPOSAL_UNACCEPTED",
        }), encoding="utf-8")
    tripped = _shared_acceptance_circuit(tmp_path, "GOAL-CIRCUIT")
    assert tripped["open"] is True
    (workunits / "WORKUNIT-ACCEPTED.json").write_text(json.dumps({
        "goal_id": "GOAL-CIRCUIT",
        "workunit_id": "WORKUNIT-ACCEPTED",
        "state": "COMPLETE",
    }), encoding="utf-8")
    intents = tmp_path / ".hawking" / "mutation-intents"
    intents.mkdir()
    (intents / "REQ-ACCEPTED.json").write_text(json.dumps({
        "status": "APPLIED",
        "request": {
            "goal_id": "GOAL-CIRCUIT",
            "workunit_id": "WORKUNIT-ACCEPTED",
        },
    }), encoding="utf-8")
    cleared = _shared_acceptance_circuit(tmp_path, "GOAL-CIRCUIT")
    assert cleared["open"] is False
    assert cleared["accepted_after_failure"] is True


def test_auto_continuation_expands_only_accepted_heads_with_production_focus():
    goal = {
        "goal_id": "GOAL-EXPANDED-FOCUSED-HEADS",
        "status": "RUNNING",
        "accepted_heads": [
            {"tranche": "P8_NOVA_IMPLEMENTATION_CONTRACT_V2"},
            {"tranche": "P8_NOVA_TEST_EVIDENCE_V2"},
        ],
        "auto_refill_queue": [
            {
                "tranche_id": "P8_NOVA_IMPLEMENTATION_CONTRACT_V2",
                "state": "ACCEPTED_READ_EVIDENCE",
                "focus_files": [
                    "tools/future/gravity_nova_lineage.py",
                    "tools/future/test_gravity_nova_lineage.py",
                ],
                "focus_tests": ["tools/future/test_gravity_nova_lineage.py"],
            },
            {
                "tranche_id": "P8_NOVA_TEST_EVIDENCE_V2",
                "state": "ACCEPTED_READ_EVIDENCE",
                "focus_files": ["tools/future/test_gravity_nova_lineage.py"],
                "focus_tests": ["tools/future/test_gravity_nova_lineage.py"],
            },
        ],
        "auto_continuation_policy": {
            "enabled": True,
            "expand_accepted_focused_heads": True,
            "max_generated_per_refill": "unbounded",
        },
    }

    updated, receipt = _materialize_auto_continuations(goal)

    assert receipt["generated"] == [{
        "tranche_id": "P8_NOVA_IMPLEMENTATION_CONTRACT_V2_AUTO_IMPLEMENTATION_V1",
        "source_tranche": "P8_NOVA_IMPLEMENTATION_CONTRACT_V2",
        "writer_scope": "auto-continuation/p8_nova_implementation_contract_v2",
    }]
    generated = updated["auto_refill_queue"][-1]
    assert generated["task_class"] == "model_science"
    assert generated["focus_files"] == [
        "tools/future/gravity_nova_lineage.py",
        "tools/future/test_gravity_nova_lineage.py",
    ]
    assert generated["focus_tests"] == [
        "tools/future/test_auto_p8_nova_implementation_contract_v2_regression.py",
    ]
    assert generated["mutation_plan"]["dedicated_regression_test"] == generated["focus_tests"][0]


def test_auto_continuation_does_not_replay_terminal_or_unopted_work():
    base = {
        "goal_id": "GOAL-CONTINUATION-GUARD",
        "status": "RUNNING",
        "accepted_heads": [{"tranche": "P18_TUNNEL"}],
        "auto_refill_queue": [{"tranche_id": "P18_TUNNEL", "state": "ACCEPTED_READ_EVIDENCE"}],
        "blocked_tranches": [{"tranche": "P19_SANDBOX", "state": "TERMINAL_PROVIDER_OUTCOME"}],
    }
    unchanged, receipt = _materialize_auto_continuations(base)
    assert unchanged == base
    assert receipt["generated"] == []

    opted = {
        **base,
        "auto_continuation_policy": {
            "enabled": True,
            "source_tranches": ["P18_TUNNEL", "P19_SANDBOX"],
        },
        "auto_refill_queue": [
            {"tranche_id": "P18_TUNNEL", "state": "ACCEPTED_READ_EVIDENCE"},
            {"tranche_id": "P19_SANDBOX", "state": "ACCEPTED_READ_EVIDENCE"},
        ],
    }
    updated, receipt = _materialize_auto_continuations(opted)
    assert [item["source_tranche"] for item in receipt["generated"]] == ["P18_TUNNEL"]
    assert not any(item.get("source_tranche") == "P19_SANDBOX" for item in updated["auto_refill_queue"])


def test_auto_continuation_reopens_mutation_failure_only_on_fresh_route():
    goal = {
        "goal_id": "GOAL-CONTINUATION-REOPEN",
        "status": "RUNNING",
        "accepted_heads": [{"tranche": "P18_TUNNEL"}],
        "auto_refill_queue": [{
            "tranche_id": "P18_TUNNEL_AUTO_IMPLEMENTATION_V1",
            "state": "TERMINAL_PROVIDER_OUTCOME",
            "generated_by": "hawking.auto.continuation.v1",
            "source_tranche": "P18_TUNNEL",
            "workunit_id": "WORKUNIT-OLD",
            "released_classification": "MUTATION_PAYLOAD_MISSING",
            "objective": "Make the bounded tunnel delta.",
            "acceptance": ["run focused tests"],
        }],
        "auto_continuation_policy": {
            "enabled": True,
            "source_tranches": ["P18_TUNNEL"],
            "route_reopen": {
                "enabled": True,
                "failure_classes": ["MUTATION_PAYLOAD_MISSING"],
                "model": "moonshotai/kimi-k3",
                "fresh_route_receipt": "receipts/fresh-route.json",
            },
        },
    }
    updated, receipt = _materialize_auto_continuations(goal)
    assert receipt["generated"] == [{
        "tranche_id": "P18_TUNNEL_AUTO_IMPLEMENTATION_V1_ROUTE_REOPEN_V2",
        "source_tranche": "P18_TUNNEL",
        "reopen_of": "WORKUNIT-OLD",
        "model_override": "moonshotai/kimi-k3",
    }]
    reopened = updated["auto_refill_queue"][-1]
    assert reopened["state"] == "READY"
    assert reopened["model_override"] == "moonshotai/kimi-k3"
    assert reopened["reopen_of"] == "WORKUNIT-OLD"


def test_route_reopen_instruction_is_idempotent():
    base = "Make the bounded tunnel delta."
    once = _with_fresh_route_instruction(base)
    instruction = "Use the explicitly qualified alternate model route for this fresh attempt; do not replay the prior provider turn."
    many = _with_fresh_route_instruction(" ".join([once, instruction, instruction, instruction]))
    assert many == once


def test_auto_continuation_compacts_legacy_route_instruction_in_existing_queue():
    instruction = (
        "Use the explicitly qualified alternate model route for this fresh attempt; "
        "do not replay the prior provider turn."
    )
    goal = {
        "goal_id": "GOAL-CONTINUATION-COMPACTION",
        "status": "RUNNING",
        "accepted_heads": [],
        "auto_refill_queue": [{
            "tranche_id": "P18_TUNNEL_AUTO_IMPLEMENTATION_V1_ROUTE_REOPEN_V2_G4",
            "state": "READY",
            "objective": "Make the bounded tunnel delta. " + " ".join([instruction] * 12),
        }],
        "auto_continuation_policy": {"enabled": True},
    }

    updated, receipt = _materialize_auto_continuations(goal)

    compacted = updated["auto_refill_queue"][0]["objective"]
    assert compacted == "Make the bounded tunnel delta. " + instruction
    assert receipt["compacted_objectives"] == [{
        "tranche_id": "P18_TUNNEL_AUTO_IMPLEMENTATION_V1_ROUTE_REOPEN_V2_G4",
        "old_chars": len(goal["auto_refill_queue"][0]["objective"]),
        "new_chars": len(compacted),
    }]


def test_route_reopen_compacts_legacy_chain_without_changing_first_generation():
    legacy = "P18_TUNNEL_AUTO_IMPLEMENTATION_V1" + ("_ROUTE_REOPEN_V2" * 24)
    assert _route_reopen_depth(legacy) == 24
    compact = _route_reopen_compact_id(legacy, 25)
    assert compact == "P18_TUNNEL_AUTO_IMPLEMENTATION_V1_ROUTE_REOPEN_V2_G25"
    assert _route_reopen_depth(compact) == 25
    assert MAX_ROUTE_REOPEN_DEPTH == 1


def test_auto_continuation_compacts_stale_reopen_projection_without_losing_live_lane():
    goal = {
        "goal_id": "GOAL-CONTINUATION-REOPEN-COMPACTION",
        "status": "RUNNING",
        "accepted_heads": [{"tranche": "P18_TUNNEL"}],
        "graph": {"nodes": [
            {"id": "P18_TUNNEL", "dependencies": []},
            {"id": "P18_TUNNEL_AUTO_IMPLEMENTATION_V1_ROUTE_REOPEN_V2", "dependencies": ["P18_TUNNEL"]},
            {"id": "P18_TUNNEL_AUTO_IMPLEMENTATION_V1_ROUTE_REOPEN_V2_G2", "dependencies": ["P18_TUNNEL"]},
            {"id": "P18_TUNNEL_AUTO_IMPLEMENTATION_V1_ROUTE_REOPEN_V2_G3", "dependencies": ["P18_TUNNEL"]},
        ]},
        "graph_frontier_state": {
            "P18_TUNNEL_AUTO_IMPLEMENTATION_V1_ROUTE_REOPEN_V2": {"state": "TERMINAL_PROVIDER_OUTCOME"},
            "P18_TUNNEL_AUTO_IMPLEMENTATION_V1_ROUTE_REOPEN_V2_G2": {"state": "RUNNING"},
            "P18_TUNNEL_AUTO_IMPLEMENTATION_V1_ROUTE_REOPEN_V2_G3": {"state": "RUNNABLE"},
        },
        "graph_runnable_frontier": [
            "P18_TUNNEL_AUTO_IMPLEMENTATION_V1_ROUTE_REOPEN_V2_G3",
        ],
        "auto_refill_queue": [
            {
                "tranche_id": "P18_TUNNEL",
                "state": "ACCEPTED_READ_EVIDENCE",
            },
            {
                "tranche_id": "P18_TUNNEL_AUTO_IMPLEMENTATION_V1_ROUTE_REOPEN_V2",
                "state": "TERMINAL_PROVIDER_OUTCOME",
                "generated_by": "hawking.auto.continuation.v1",
                "reopen_of": "WORKUNIT-OLD",
                "source_tranche": "P18_TUNNEL",
            },
            {
                "tranche_id": "P18_TUNNEL_AUTO_IMPLEMENTATION_V1_ROUTE_REOPEN_V2_G2",
                "state": "ADMITTED",
                "generated_by": "hawking.auto.continuation.v1",
                "reopen_of": "WORKUNIT-RECOVERY",
                "source_tranche": "P18_TUNNEL",
            },
            {
                "tranche_id": "P18_TUNNEL_AUTO_IMPLEMENTATION_V1_ROUTE_REOPEN_V2_G3",
                "state": "READY",
                "generated_by": "hawking.auto.continuation.v1",
                "reopen_of": "WORKUNIT-STALE",
                "source_tranche": "P18_TUNNEL",
            },
        ],
        "auto_continuation_policy": {
            "enabled": True,
            "source_tranches": ["P18_TUNNEL"],
            "generated_tranches": ["P18_TUNNEL_AUTO_IMPLEMENTATION_V1"],
        },
    }

    updated, receipt = _materialize_auto_continuations(goal)

    assert [item["tranche_id"] for item in updated["auto_refill_queue"]] == [
        "P18_TUNNEL",
        "P18_TUNNEL_AUTO_IMPLEMENTATION_V1_ROUTE_REOPEN_V2_G2",
    ]
    assert [node["id"] for node in updated["graph"]["nodes"]] == [
        "P18_TUNNEL",
        "P18_TUNNEL_AUTO_IMPLEMENTATION_V1_ROUTE_REOPEN_V2_G2",
    ]
    assert updated["graph_runnable_frontier"] == []
    assert receipt["stale_route_reopen_projection"] == {
        "removed_count": 2,
        "removed_by_state": {"TERMINAL_PROVIDER_OUTCOME": 1, "READY": 1},
        "tranche_ids": [
            "P18_TUNNEL_AUTO_IMPLEMENTATION_V1_ROUTE_REOPEN_V2",
            "P18_TUNNEL_AUTO_IMPLEMENTATION_V1_ROUTE_REOPEN_V2_G3",
        ],
        "claim_boundary": "Only redundant queue/graph projections were removed; WorkUnit records and result/dead-letter packets remain authoritative.",
    }


def test_auto_continuation_compacts_terminal_versions_after_one_source_recovery():
    """One recovery is owned by the source tranche, not every V1..V4 row."""
    base = "P18_TUNNEL"
    recovered = f"{base}_AUTO_IMPLEMENTATION_V4_ROUTE_REOPEN_V2"
    goal = {
        "goal_id": "GOAL-CONTINUATION-SOURCE-RECOVERY",
        "status": "RUNNING",
        "accepted_heads": [{"tranche": base}],
        "graph": {"nodes": [
            {"id": base, "dependencies": []},
            {"id": f"{base}_AUTO_IMPLEMENTATION_V1", "dependencies": [base]},
            {"id": f"{base}_AUTO_IMPLEMENTATION_V4", "dependencies": [base]},
            {"id": recovered, "dependencies": [base]},
        ]},
        "auto_refill_queue": [
            {"tranche_id": base, "state": "ACCEPTED_READ_EVIDENCE"},
            {
                "tranche_id": f"{base}_AUTO_IMPLEMENTATION_V1",
                "state": "TERMINAL_PROVIDER_OUTCOME",
                "generated_by": "hawking.auto.continuation.v1",
                "source_tranche": base,
            },
            {
                "tranche_id": f"{base}_AUTO_IMPLEMENTATION_V4",
                "state": "TERMINAL_PROVIDER_OUTCOME",
                "generated_by": "hawking.auto.continuation.v1",
                "source_tranche": base,
            },
            {
                "tranche_id": recovered,
                "state": "READY",
                "generated_by": "hawking.auto.continuation.v1",
                "source_tranche": base,
                "reopen_of": "WORKUNIT-V4",
            },
        ],
        "auto_continuation_policy": {
            "enabled": True,
            "source_tranches": [base],
            "generated_tranches": [
                f"{base}_AUTO_IMPLEMENTATION_V1",
                f"{base}_AUTO_IMPLEMENTATION_V4",
            ],
        },
    }

    updated, receipt = _materialize_auto_continuations(goal)

    assert [item["tranche_id"] for item in updated["auto_refill_queue"]] == [
        base, recovered,
    ]
    assert receipt["stale_route_reopen_projection"]["removed_by_state"] == {
        "TERMINAL_PROVIDER_OUTCOME": 2,
    }


def test_auto_continuation_does_not_recursively_reopen_a_failed_recovery():
    goal = {
        "goal_id": "GOAL-CONTINUATION-NO-RECURSION",
        "status": "RUNNING",
        "accepted_heads": [{"tranche": "P18_TUNNEL"}],
        "auto_refill_queue": [{
            "tranche_id": "P18_TUNNEL_AUTO_IMPLEMENTATION_V1_ROUTE_REOPEN_V2",
            "state": "TERMINAL_PROVIDER_OUTCOME",
            "generated_by": "hawking.auto.continuation.v1",
            "source_tranche": "P18_TUNNEL",
            "reopen_of": "WORKUNIT-OLD",
            "workunit_id": "WORKUNIT-RECOVERY",
            "released_classification": "MUTATION_PAYLOAD_MISSING",
        }],
        "auto_continuation_policy": {
            "enabled": True,
            "source_tranches": ["P18_TUNNEL"],
            "route_reopen": {
                "enabled": True,
                "failure_classes": ["MUTATION_PAYLOAD_MISSING"],
                "model": "moonshotai/kimi-k3",
                "fresh_route_receipt": "receipts/fresh-route.json",
            },
        },
    }

    updated, receipt = _materialize_auto_continuations(goal)

    assert updated["auto_refill_queue"] == []
    assert receipt["generated"] == []
    assert receipt["stale_route_reopen_projection"]["removed_count"] == 1


def test_auto_continuation_reopens_only_newest_contract_per_writer_scope():
    def terminal(version, workunit_id):
        return {
            "tranche_id": f"P18_TUNNEL_AUTO_IMPLEMENTATION_V{version}",
            "state": "TERMINAL_PROVIDER_OUTCOME",
            "generated_by": "hawking.auto.continuation.v1",
            "source_tranche": "P18_TUNNEL",
            "writer_scope": "auto-continuation/p18_tunnel",
            "workunit_id": workunit_id,
            "released_classification": "MUTATION_PAYLOAD_MISSING",
            "objective": "Make the bounded tunnel delta.",
            "acceptance": ["run focused tests"],
        }

    goal = {
        "goal_id": "GOAL-CONTINUATION-NEWEST-ROUTE",
        "status": "RUNNING",
        "accepted_heads": [{"tranche": "P18_TUNNEL"}],
        "auto_refill_queue": [terminal(1, "WORKUNIT-V1"), terminal(4, "WORKUNIT-V4")],
        "auto_continuation_policy": {
            "enabled": True,
            "source_tranches": ["P18_TUNNEL"],
            "route_reopen": {
                "enabled": True,
                "failure_classes": ["MUTATION_PAYLOAD_MISSING"],
                "model": "moonshotai/kimi-k3",
                "fresh_route_receipt": "receipts/fresh-route.json",
            },
        },
    }

    updated, receipt = _materialize_auto_continuations(goal)

    assert receipt["generated"] == [{
        "tranche_id": "P18_TUNNEL_AUTO_IMPLEMENTATION_V4_ROUTE_REOPEN_V2",
        "source_tranche": "P18_TUNNEL",
        "reopen_of": "WORKUNIT-V4",
        "model_override": "moonshotai/kimi-k3",
    }]
    assert any(
        item.get("reason") == "route_reopen_scope_already_recovered"
        and item.get("tranche_id") == "P18_TUNNEL_AUTO_IMPLEMENTATION_V1"
        for item in receipt["skipped"]
    )
    assert updated["auto_refill_queue"][-1]["writer_scope"] == "auto-continuation/p18_tunnel"


def test_auto_refill_persists_stale_reopen_projection_compaction(tmp_path):
    goal = create_goal(tmp_path, objective="A bounded reopen projection.", budget_usd=5.0)
    stale = "P18_TUNNEL_AUTO_IMPLEMENTATION_V1_ROUTE_REOPEN_V2_G2"
    update_goal(
        tmp_path,
        goal["goal_id"],
        status="RUNNING",
        accepted_heads=[{"tranche": "P18_TUNNEL"}],
        graph={"nodes": [
            {"id": "P18_TUNNEL", "dependencies": []},
            {"id": stale, "dependencies": ["P18_TUNNEL"]},
        ]},
        graph_frontier_state={stale: {"state": "RUNNABLE"}},
        graph_runnable_frontier=[stale],
        auto_refill_queue=[
            {"tranche_id": "P18_TUNNEL", "state": "ACCEPTED_READ_EVIDENCE"},
            {
                "tranche_id": stale,
                "state": "READY",
                "generated_by": "hawking.auto.continuation.v1",
                "reopen_of": "WORKUNIT-STALE",
                "source_tranche": "P18_TUNNEL",
            },
        ],
        auto_continuation_policy={
            "enabled": True,
            "source_tranches": ["P18_TUNNEL"],
            "generated_tranches": ["P18_TUNNEL_AUTO_IMPLEMENTATION_V1"],
        },
    )

    result = refill_runnable_frontier(
        tmp_path, goal["goal_id"], efficient_concurrency=None, max_new=0,
    )

    assert result["admitted"] == []
    parent = load_goal(tmp_path, goal["goal_id"])
    assert [item["tranche_id"] for item in parent["auto_refill_queue"]] == ["P18_TUNNEL"]
    assert [node["id"] for node in parent["graph"]["nodes"]] == ["P18_TUNNEL"]
    assert parent["auto_continuation_last_run"]["stale_route_reopen_projection"]["removed_count"] == 1


def test_auto_continuation_compacts_stale_generated_id_index():
    current = "P18_TUNNEL_AUTO_IMPLEMENTATION_V4"
    stale = "P18_TUNNEL_AUTO_IMPLEMENTATION_V1" + ("_ROUTE_REOPEN_V2" * 32)
    goal = {
        "goal_id": "GOAL-CONTINUATION-GENERATED-ID-COMPACTION",
        "status": "RUNNING",
        "auto_refill_queue": [{
            "tranche_id": current,
            "state": "COMPLETE_PENDING_ACCEPTANCE",
            "generated_by": "hawking.auto.continuation.v1",
            "source_tranche": "P18_TUNNEL",
        }],
        "auto_continuation_policy": {
            "enabled": True,
            "generated_tranches": [current, stale],
        },
    }

    updated, receipt = _materialize_auto_continuations(goal)

    assert updated["auto_continuation_policy"]["generated_tranches"] == [current]
    assert receipt["stale_generated_tranche_projection"]["removed_count"] == 1
    assert receipt["stale_generated_tranche_projection"]["tranche_ids"] == [stale]


def test_active_route_diversity_keeps_qualified_alternate_alive():
    selected, receipt = select_active_diversity_route(
        "routine_implementation",
        ["deepseek/deepseek-v4.1-flash", "moonshotai/kimi-k3"],
        "deepseek/deepseek-v4.1-flash",
        active_model_counts={
            "deepseek/deepseek-v4.1-flash": 8,
            "moonshotai/kimi-k3": 1,
        },
    )
    assert selected == "moonshotai/kimi-k3"
    assert receipt["reason"] == "active_lane_diversification"


def test_authoritative_active_model_counts_override_stale_queue_route(tmp_path):
    workunits = tmp_path / ".hawking" / "workunits"
    workunits.mkdir(parents=True)
    for index in range(8):
        workunit_id = f"WORKUNIT-KIMI-{index}"
        (workunits / f"{workunit_id}.json").write_text(json.dumps({
            "workunit_id": workunit_id,
            "goal_id": "GOAL-LIVE",
            "state": "RUNNING",
            "worker_model": "moonshotai/kimi-k3",
        }), encoding="utf-8")
    (workunits / "WORKUNIT-DEEPSEEK.json").write_text(json.dumps({
        "workunit_id": "WORKUNIT-DEEPSEEK",
        "goal_id": "GOAL-LIVE",
        "state": "RUNNING",
        "worker_model": "deepseek/deepseek-v4.1-flash",
    }), encoding="utf-8")
    counts = _authoritative_active_model_counts(
        tmp_path,
        "GOAL-LIVE",
        [{
            "workunit_id": "WORKUNIT-KIMI-0",
            "state": "RUNNING",
            "selected_model": "deepseek/deepseek-v4.1-flash",
        }],
    )
    assert counts == {
        "moonshotai/kimi-k3": 8,
        "deepseek/deepseek-v4.1-flash": 1,
    }


def test_auto_continuation_measured_reopen_route_avoids_fixed_provider(tmp_path):
    performance = tmp_path / ".hawking" / "auto"
    performance.mkdir(parents=True)
    (performance / "model-performance.json").write_text(json.dumps({
        "models": {
            "deepseek/deepseek-v4.1-flash::routine_implementation": {
                "attempts": 2, "accepted": 1, "cost_usd": 0.01,
                "failure_classes": {}, "provider_transport_failures": 0,
            },
            "moonshotai/kimi-k3::routine_implementation": {
                "attempts": 8, "accepted": 0, "cost_usd": 0.20,
                "failure_classes": {"PROVIDER_TRANSPORT_ERROR": 8},
                "provider_transport_failures": 8,
            },
            "z-ai/glm-5.3::routine_implementation": {
                "attempts": 2, "accepted": 0, "cost_usd": 0.02,
                "failure_classes": {"PROVIDER_OUTPUT_UNUSABLE": 2},
                "provider_transport_failures": 0,
            },
        }
    }), encoding="utf-8")
    goal = {
        "goal_id": "GOAL-CONTINUATION-MEASURED-ROUTE",
        "status": "RUNNING",
        "accepted_heads": [{"tranche": "P18_TUNNEL"}],
        "auto_refill_queue": [{
            "tranche_id": "P18_TUNNEL_AUTO_IMPLEMENTATION_V1",
            "state": "TERMINAL_PROVIDER_OUTCOME",
            "generated_by": "hawking.auto.continuation.v1",
            "source_tranche": "P18_TUNNEL",
            "workunit_id": "WORKUNIT-OLD",
            "released_classification": "PROVIDER_TRANSPORT_ERROR",
            "task_class": "routine_implementation",
            "objective": "Make the bounded tunnel delta.",
            "acceptance": ["run focused tests"],
        }],
        "auto_continuation_policy": {
            "enabled": True,
            "source_tranches": ["P18_TUNNEL"],
            "route_reopen": {
                "enabled": True,
                "failure_classes": ["PROVIDER_TRANSPORT_ERROR"],
                "model": "hawking-auto",
                "model_pool": [
                    "deepseek/deepseek-v4.1-flash", "moonshotai/kimi-k3", "z-ai/glm-5.3",
                ],
                "fresh_route_receipt": "receipts/fresh-route.json",
            },
        },
    }
    updated, receipt = _materialize_auto_continuations(goal, workspace=tmp_path)
    reopened = updated["auto_refill_queue"][-1]
    assert reopened["model_override"] == "deepseek/deepseek-v4.1-flash"
    assert reopened["route_selection"]["reason"] == "measured_value"
    assert receipt["generated"][-1]["model_override"] == "deepseek/deepseek-v4.1-flash"


def test_graph_projection_does_not_reintroduce_terminal_provider_nodes(tmp_path, monkeypatch):
    goal = create_goal(tmp_path, objective="A terminal projection frontier.", budget_usd=5.0)
    update_goal(
        tmp_path,
        goal["goal_id"],
        status="RUNNING",
        graph={"nodes": [{"id": "P-TERMINAL", "dependencies": []}]},
        graph_frontier_state={
            "P-TERMINAL": {
                "state": "TERMINAL_PROVIDER_OUTCOME",
                "workunit_id": "WORKUNIT-OLD",
            }
        },
        graph_runnable_frontier=[],
        auto_refill_queue=[{
            "tranche_id": "P-TERMINAL",
            "priority": 100,
            "objective": "Do not replay terminal provider output.",
            "acceptance": ["return fresh evidence"],
        }],
    )
    monkeypatch.setattr(
        "hawking.goal_surface.dispatch_child_workunit",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("terminal node replayed")),
    )
    result = refill_runnable_frontier(tmp_path, goal["goal_id"], efficient_concurrency=None)
    assert result["admitted"] == []
    assert result["skipped"] == [{"tranche_id": "P-TERMINAL", "reason": "not_graph_runnable"}]
    parent = load_goal(tmp_path, goal["goal_id"])
    assert parent["graph_runnable_frontier"] == []
    assert parent["graph_frontier_state"]["P-TERMINAL"]["state"] == "TERMINAL_PROVIDER_OUTCOME"


def test_auto_refill_does_not_retry_known_terminal_tranche(tmp_path, monkeypatch):
    goal = create_goal(tmp_path, objective="A terminal-aware V2 frontier.", budget_usd=5.0)
    update_goal(
        tmp_path,
        goal["goal_id"],
        status="RUNNING",
        graph_runnable_frontier=["P-FAILED"],
        blocked_tranches=[{"tranche": "P-FAILED", "reason": "PROVIDER_OUTPUT_UNUSABLE"}],
        auto_refill_queue=[{
            "tranche_id": "P-FAILED",
            "priority": 100,
            "objective": "Do not replay this lane.",
            "acceptance": ["return evidence"],
        }],
    )
    monkeypatch.setattr(
        "hawking.goal_surface.dispatch_child_workunit",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("terminal lane retried")),
    )
    result = refill_runnable_frontier(tmp_path, goal["goal_id"])
    assert result["admitted"] == []
    assert result["skipped"] == [{"tranche_id": "P-FAILED", "reason": "known_terminal_or_admitted"}]


def test_auto_refill_reopens_terminal_tranche_only_with_fresh_route_receipt(tmp_path, monkeypatch):
    goal = create_goal(tmp_path, objective="A fresh-route reopen frontier.", budget_usd=5.0)
    receipt = tmp_path / "receipts" / "fresh-route.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text('{"tool_call_qualified": true}', encoding="utf-8")
    update_goal(
        tmp_path,
        goal["goal_id"],
        status="RUNNING",
        graph_runnable_frontier=["P-FAILED"],
        blocked_tranches=[{"tranche": "P-FAILED", "reason": "PROVIDER_OUTPUT_UNUSABLE"}],
        auto_refill_queue=[{
            "tranche_id": "P-FAILED",
            "priority": 100,
            "writer_scope": "failed-scope",
            "objective": "Run a materially fresh bounded packet.",
            "acceptance": ["return fresh evidence"],
            "reopen_of": "WORKUNIT-OLD",
            "fresh_route_receipt": "receipts/fresh-route.json",
        }],
    )

    def fake_dispatch(*args, **kwargs):
        return {"workunit_id": "WORKUNIT-FRESH", "auto_route": {"selected_model": "moonshotai/kimi-k3"}}

    monkeypatch.setattr("hawking.goal_surface.dispatch_child_workunit", fake_dispatch)
    result = refill_runnable_frontier(tmp_path, goal["goal_id"])
    assert result["admitted"] == [{
        "tranche_id": "P-FAILED",
        "workunit_id": "WORKUNIT-FRESH",
        "writer_scope": "failed-scope",
    }]


def test_auto_refill_admits_bounded_generated_reopen_by_source_lineage(tmp_path, monkeypatch):
    goal = create_goal(tmp_path, objective="A generated reopen continuation.", budget_usd=5.0)
    receipt = tmp_path / "receipts" / "fresh-route.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text('{"tool_call_qualified": true}', encoding="utf-8")
    continuation = "P-FAILED_AUTO_IMPLEMENTATION_V1_ROUTE_REOPEN_V2"
    update_goal(
        tmp_path,
        goal["goal_id"],
        status="RUNNING",
        graph={"nodes": [{"id": "P-FAILED", "dependencies": []}]},
        graph_frontier_state={"P-FAILED": {"state": "TERMINAL_PROVIDER_OUTCOME"}},
        graph_runnable_frontier=[],
        blocked_tranches=[{"tranche": "P-FAILED", "reason": "PROVIDER_OUTPUT_UNUSABLE"}],
        auto_refill_queue=[{
            "tranche_id": continuation,
            "priority": 100,
            "writer_scope": "failed-scope",
            "objective": "Run a fresh bounded continuation.",
            "acceptance": ["return fresh evidence"],
            "generated_by": "hawking.auto.continuation.v1",
            "reopen_of": "WORKUNIT-OLD",
            "fresh_route_receipt": "receipts/fresh-route.json",
            "mutation_plan": {"source_tranche": "P-FAILED"},
        }],
    )

    monkeypatch.setattr(
        "hawking.goal_surface.dispatch_child_workunit",
        lambda *args, **kwargs: {"workunit_id": "WORKUNIT-FRESH-CONTINUATION", "auto_route": {}},
    )
    result = refill_runnable_frontier(tmp_path, goal["goal_id"], efficient_concurrency=None)
    assert result["admitted"] == [{
        "tranche_id": continuation,
        "workunit_id": "WORKUNIT-FRESH-CONTINUATION",
        "writer_scope": "failed-scope",
    }]
def test_p17_collapse_contract_v2_qualification_support():
    """P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2 qualification-support guard.

    Bounded qualification-support check: asserts the collapse implementation
    contract v2 source marker is present and callable. This is a no-release,
    no-hardware-qualification guard only.
    """
    import hawking.goal_surface as goal_surface

    assert goal_surface.P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2 == (
        "P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2"
    )
    assert goal_surface.p17_collapse_contract_v2_marker() == (
        "P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2"
    )
def test_p17_collapse_implementation_contract_surface_present():
    """P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2 qualification-support check.

    Asserts the collapse implementation contract surface is importable and
    exposes its declared entry point and version. Bounded qualification-support
    delta; does not assert phase or release completion and does not exercise
    hardware.
    """
    import importlib

    module = importlib.import_module("hawking.goal_surface")
    assert module is not None

    assert module.P17_COLLAPSE_IMPLEMENTATION_CONTRACT_VERSION == "V2"
    assert callable(module.p17_collapse_implementation_contract)

    descriptor = module.p17_collapse_implementation_contract()
    assert descriptor["contract"] == "P17_COLLAPSE_IMPLEMENTATION_CONTRACT"
    assert descriptor["version"] == "V2"
    assert descriptor["entry_point"] == "p17_collapse_implementation_contract"
def test_p4_auto_implementation_contract_v2_scope_marker():
    from hawking.auto_orchestration import _p4_auto_implementation_contract_v2_scope_marker

    assert _p4_auto_implementation_contract_v2_scope_marker() == "P4_AUTO_IMPLEMENTATION_CONTRACT_V2"
def test_p4_auto_implementation_contract_probe_observable_behavior():
    from hawking.auto_orchestration import _p4_auto_implementation_contract_probe

    assert _p4_auto_implementation_contract_probe({"discrete_goal_complete": True}) is True
    assert _p4_auto_implementation_contract_probe({"discrete_goal_complete": False}) is False
    assert _p4_auto_implementation_contract_probe({}) is False
    assert _p4_auto_implementation_contract_probe(None) is False

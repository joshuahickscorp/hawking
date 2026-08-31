"""Tests for the static candidate staged planner.

The live physical qualification queue is Codex-owned and may only be visible
via another worktree of this repo. Tests that need the real 30-row set load it
read-only; they never write receipts/headless.
"""
from __future__ import annotations

import json
from itertools import combinations

import pytest

from tools.future import candidate_planner as cp
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims


def _cand(**kwargs):
    row = {
        "candidate_id": "x",
        "model": "Qwen27",
        "status": "READY_PROTECTED",
        "affected_physical_region": "region-a",
        "dependencies": [],
        "blocked_reason": None,
        "parity_contract": "same-parity",
        "capability_contract": "same-cap",
        "control_configuration": {"child_fusion_env": {}},
        "exact_mutation": {"child_fusion_env": {"HAWKING_FOO": "1"}},
        "expected_dispatch_reduction": "0",
        "expected_eliminated_work": "none",
        "expected_intermediate_byte_reduction": "0",
        "expected_active_byte_change": "unchanged",
        "expected_gpu_ns_mechanism": "geometry only",
    }
    row.update(kwargs)
    return row


def _stub_queue(rows):
    ids = [r["candidate_id"] for r in rows]
    return {
        "schema": "hawking.accelerator.physical_qualification_queue.v1",
        "version": 1,
        "fingerprint": "test",
        "candidates": rows,
        "funnel": {
            "static_validation": ids,
            "native_parity": [],
            "diagnostic_relative_ab": [],
            "protected_absolute_complete_wall": [
                r["candidate_id"] for r in rows if r.get("status") == "READY_PROTECTED"
            ],
            "promotion": [],
            "promotion_rule": (
                "only a protected complete-token receipt with capability and "
                "zero-fallback gates may promote"
            ),
        },
        "status_transitions": {
            "STATIC_ONLY": ["BLOCKED", "READY_DIAGNOSTIC", "READY_PROTECTED"],
            "READY_DIAGNOSTIC": ["BLOCKED", "DIAGNOSTIC_PASS", "DIAGNOSTIC_REJECT"],
            "DIAGNOSTIC_PASS": ["BLOCKED", "READY_PROTECTED"],
            "DIAGNOSTIC_REJECT": ["BLOCKED", "STATIC_ONLY"],
            "READY_PROTECTED": ["BLOCKED", "PROTECTED_PASS", "PROTECTED_REJECT"],
            "PROTECTED_PASS": ["BLOCKED", "INTEGRATED"],
            "PROTECTED_REJECT": ["BLOCKED", "STATIC_ONLY"],
            "INTEGRATED": [],
            "BLOCKED": ["STATIC_ONLY"],
        },
        "queue_policy": {
            "planning_is_side_effect_free": True,
            "protected_start_requires_existing_hcli_lease": True,
            "protected_start_requires_machine_quiescence": True,
            "diagnostic_results_do_not_promote": True,
        },
        "measurement_contract": {
            "protected_pass_requires_all_fields": True,
            "null_policy": "missing physical metrics remain null",
        },
    }


@pytest.fixture(scope="module")
def live_queue():
    return cp.load_queue()


@pytest.fixture(scope="module")
def live_rows(live_queue):
    return list(live_queue["candidates"])


def test_build_emits_sealed_receipt():
    out = cp.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "CANDIDATE_STAGED_PLAN.json"
    assert doc["schema"] == "hawking.future.candidate_planner.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        # A hardware key may exist as a nested string (mechanism text) but never
        # as a number anywhere in the document.
        pass


def test_selftest_is_callable():
    assert callable(cp.selftest)


def test_queue_has_the_campaign_shape(live_queue, live_rows):
    assert live_queue["schema"] == "hawking.accelerator.physical_qualification_queue.v1"
    # Disk is authority. The campaign brief said 30/12/14; the live queue may
    # have grown (pipeline-id-resolution was added on both models).
    n = len(live_rows)
    assert n == live_queue["counts"]["candidates"]
    assert n >= 30
    by_status = {}
    by_model = {}
    for row in live_rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        by_model[row["model"]] = by_model.get(row["model"], 0) + 1
    assert by_status.get("READY_PROTECTED") == live_queue["counts"]["by_status"]["READY_PROTECTED"]
    assert by_status["READY_PROTECTED"] >= 12
    assert by_status.get("BLOCKED", 0) >= 14
    assert by_model.get("Qwen27", 0) >= 15
    assert by_model.get("Flash", 0) >= 15


def test_dependency_graph_includes_declared_and_inferred(live_rows):
    graph = cp.build_graph(live_rows)
    declared = {(e["from"], e["to"]) for e in graph["declared_edges"]}
    assert ("qwen27-fast-profile", "qwen27-affine2-splitk4") in declared
    assert ("qwen27-affine2-splitk4", "qwen27-affine2-splitk4-vec") in declared
    assert ("flash-hc-staged-threadgroup", "flash-hc-router-topk-fusion") in declared
    assert ("flash-router-topk-fusion", "flash-hc-router-topk-fusion") in declared
    ids = {r["candidate_id"] for r in live_rows}
    if "qwen27-pipeline-id-resolution" in ids:
        assert ("qwen27-pipeline-cache-reuse", "qwen27-pipeline-id-resolution") in declared
    kinds = set()
    pair = None
    for edge in graph["conflict_edges"]:
        if {edge["a"], edge["b"]} == {"qwen27-affine2-splitk4", "qwen27-affine2-splitk4-vec"}:
            pair = edge
        for r in edge["reasons"]:
            kinds.add(r["kind"])
    assert pair is not None
    assert "same_region_incompatible_mutation" in {r["kind"] for r in pair["reasons"]}
    # Different regions, colliding env keys: affine2-splitk4 vs q2f-splitk4.
    env_pair = next(
        e
        for e in graph["conflict_edges"]
        if {e["a"], e["b"]} == {"qwen27-affine2-splitk4", "qwen27-q2f-splitk4"}
    )
    assert "env_key_collision" in {r["kind"] for r in env_pair["reasons"]}
    assert "distinct_model_executables" in kinds


def test_equivalence_geometry_refinement_has_evidence(live_rows):
    classes = cp.equivalence_classes(live_rows)
    by_member = {}
    for cl in classes:
        for m in cl["members"]:
            by_member[m] = cl
    affine = by_member["qwen27-affine2-splitk4"]
    assert "qwen27-affine2-splitk4-vec" in affine["members"]
    assert any(e["kind"] == "geometry_refinement" for e in affine["evidence"])
    q2f = by_member["qwen27-q2f-splitk4"]
    assert "qwen27-q2f-splitk4-vec" in q2f["members"]
    assert affine["class_id"] != q2f["class_id"]


def test_equivalence_does_not_merge_on_name_similarity(live_rows):
    classes = cp.equivalence_classes(live_rows)
    by_member = {m: cl["class_id"] for cl in classes for m in cl["members"]}
    # Both names contain "splitk4"; they are different regions and stay apart.
    assert by_member["qwen27-affine2-splitk4"] != by_member["qwen27-q2f-splitk4"]
    # Both names contain "router-topk-fusion"; the HC row is a composition, not a rename.
    assert by_member["flash-router-topk-fusion"] != by_member["flash-hc-router-topk-fusion"]


def test_equivalence_cross_model_twins_share_a_stem(live_rows):
    classes = cp.equivalence_classes(live_rows)
    by_member = {m: cl for cl in classes for m in cl["members"]}
    label = by_member["qwen27-encoder-label-elision"]
    assert "flash-encoder-label-elision" in label["members"]
    assert any(
        e["kind"] in {"shared_distinctive_stem", "cross_model_mechanism_jaccard"}
        for e in label["evidence"]
    )


def test_redundant_pruning_names_dominator_on_synthetic():
    weak = _cand(
        candidate_id="weak-geo",
        affected_physical_region="same-gemv",
        exact_mutation={"child_fusion_env": {"HAWKING_GEO": "tpr64"}},
        expected_dispatch_reduction="0",
        expected_eliminated_work="none; geometry-only candidate",
        expected_intermediate_byte_reduction="0",
    )
    strong = _cand(
        candidate_id="strong-geo",
        affected_physical_region="same-gemv",
        exact_mutation={"child_fusion_env": {"HAWKING_GEO": "fused4"}},
        expected_dispatch_reduction="reduce the sequence from five dispatches to one",
        expected_eliminated_work="four standalone dispatches",
        expected_intermediate_byte_reduction="remove staging traffic",
    )
    other = _cand(
        candidate_id="other-region",
        affected_physical_region="elsewhere",
        exact_mutation={"child_fusion_env": {"HAWKING_OTHER": "1"}},
        expected_dispatch_reduction="0",
    )
    pruned = {p["candidate_id"]: p for p in cp.prune_redundant([weak, strong, other])}
    assert pruned["weak-geo"]["redundant"] is True
    assert pruned["weak-geo"]["dominated_by"] == ["strong-geo"]
    assert pruned["strong-geo"]["redundant"] is False
    assert pruned["other-region"]["redundant"] is False


def test_parent_child_geometry_is_lineage_not_redundancy(live_rows):
    pruned = {p["candidate_id"]: p for p in cp.prune_redundant(live_rows)}
    # splitk4 is the control the vec child is measured against.
    assert pruned["qwen27-affine2-splitk4"]["redundant"] is False
    assert pruned["qwen27-affine2-splitk4-vec"]["redundant"] is False
    assert pruned["qwen27-q2f-splitk4"]["redundant"] is False


def test_incompatible_same_region_refusal_actually_fires(live_rows):
    """Negative control: the guard must raise on a known-bad pair.

    A helper that only returns True on happy input is not a guard.
    """
    index = {r["candidate_id"]: r for r in live_rows}
    a = index["qwen27-affine2-splitk4"]
    b = index["qwen27-affine2-splitk4-vec"]
    with pytest.raises(cp.IncompatibleMutationError) as fired:
        cp.assert_cell_compatible([a, b])
    msg = str(fired.value)
    assert "qwen27-affine2-splitk4" in msg
    assert "same_region_incompatible_mutation" in msg or "affected_physical_region" in msg
    # The same guard stays silent on a compatible disjoint pair, so a passing
    # raise is not "raises on everything".
    c = index["qwen27-encoder-label-elision"]
    cp.assert_cell_compatible([a, c])
    # Env-key collision across regions also fires.
    d = index["qwen27-q2f-splitk4"]
    with pytest.raises(cp.IncompatibleMutationError) as env_fired:
        cp.assert_cell_compatible([a, d])
    assert "env_key_collision" in str(env_fired.value)


def test_synthetic_same_region_incompatible_pair_is_refused():
    a = _cand(
        candidate_id="one",
        affected_physical_region="R",
        exact_mutation={"child_fusion_env": {"HAWKING_GEO": "a"}},
    )
    b = _cand(
        candidate_id="two",
        affected_physical_region="R",
        exact_mutation={"child_fusion_env": {"HAWKING_GEO": "b"}},
    )
    with pytest.raises(cp.IncompatibleMutationError):
        cp.assert_cell_compatible([a, b])
    # Compatible: different region, disjoint keys.
    c = _cand(
        candidate_id="three",
        affected_physical_region="S",
        exact_mutation={"child_fusion_env": {"HAWKING_OTHER": "1"}},
    )
    cp.assert_cell_compatible([a, c])


def test_staged_plan_dramatically_smaller_than_power_set(live_queue, live_rows):
    doc = cp.plan_from_queue(live_queue)
    plan = doc["staged_factorial_plan"]
    naive = plan["naive_power_set"]
    staged = plan["staged"]
    n_all = len(live_rows)
    n_ready = sum(1 for r in live_rows if r["status"] == "READY_PROTECTED")
    assert naive["n_all"] == n_all
    assert naive["size_all"] == 2**n_all
    assert naive["n_ready_protected_measurable"] == n_ready
    assert naive["size_ready_protected"] == 2**n_ready
    assert staged["cell_count"] < naive["size_ready_protected"]
    assert staged["cell_count"] * 8 < naive["size_ready_protected"]
    assert staged["cell_count"] * 1024 < naive["size_all"]
    assert staged["dramatically_smaller"] is True
    full_pairs = n_ready * (n_ready - 1) // 2
    assert staged["pair_cells"] < full_pairs
    assert staged["pair_cells"] < full_pairs // 2
    cp.assert_plan_dramatically_smaller(plan)


def test_real_plan_never_coschedules_incompatible_same_region_pair(live_queue, live_rows):
    doc = cp.plan_from_queue(live_queue)
    plan = doc["staged_factorial_plan"]
    cp.assert_no_incompatible_cell(plan, live_rows)
    forbidden = {("qwen27-affine2-splitk4", "qwen27-affine2-splitk4-vec")}
    forbidden.add(("qwen27-q2f-splitk4", "qwen27-q2f-splitk4-vec"))
    for cell in plan["cells"]:
        members = tuple(sorted(cell["candidates"]))
        if len(members) < 2:
            continue
        for a, b in combinations(members, 2):
            assert (a, b) not in forbidden
            assert (b, a) not in forbidden
    # Live same-region pair is split across stages: parent is a stage-1 single,
    # vec child is contingent on the parent surviving.
    stage_of = {}
    for cell in plan["cells"]:
        for ident in cell["candidates"]:
            stage_of.setdefault(ident, set()).add(cell["stage"])
    assert "1" in stage_of["qwen27-affine2-splitk4"]
    assert "C" in stage_of["qwen27-affine2-splitk4-vec"]
    assert stage_of["qwen27-affine2-splitk4"].isdisjoint(stage_of["qwen27-affine2-splitk4-vec"])


def test_predicted_additive_disjoint_pair_is_not_a_stage2_cell(live_queue, live_rows):
    doc = cp.plan_from_queue(live_queue)
    pair_cells = {
        frozenset(c["candidates"])
        for c in doc["staged_factorial_plan"]["cells"]
        if c["kind"] == "pair"
    }
    # Host encoder-label vs Q4 GEMV geometry: no shared resource tag, so a pair
    # cell would be a stage that cannot change an additivity decision.
    disjoint = frozenset({"qwen27-encoder-label-elision", "qwen27-q4-vecgroup-x64"})
    assert disjoint not in pair_cells
    # Fast-profile is subsumed by every child; pairing it with a child cannot
    # disambiguate anything the child single does not already measure.
    for cell in pair_cells:
        assert "qwen27-fast-profile" not in cell


def test_every_stage_names_what_it_disambiguates(live_queue):
    doc = cp.plan_from_queue(live_queue)
    plan = doc["staged_factorial_plan"]
    for stage in plan["stages"]:
        assert stage["disambiguates"]
    for cell in plan["cells"]:
        assert cell["disambiguates"]
        assert cell["executes_benchmark"] is False
        assert cell["acquires_lease"] is False


def test_lineage_scar_propagation(live_rows):
    scars = cp.lineage_scars(live_rows)
    fast = scars["qwen27-fast-profile"]
    assert "qwen27-affine2-splitk4" in fast["hard_invalidates_descendants"]
    assert "qwen27-affine2-splitk4-vec" in fast["hard_invalidates_descendants"]
    assert "qwen27-resident-untimed-decode" in fast["hard_invalidates_descendants"]
    affine = scars["qwen27-affine2-splitk4"]
    assert affine["hard_invalidates_descendants"] == ["qwen27-affine2-splitk4-vec"]
    # A Qwen rejection questions the Flash mechanism twin; it does not
    # hard-invalidate a different model (no Odyssey II law store).
    label = scars["qwen27-encoder-label-elision"]
    assert "flash-encoder-label-elision" in label["equivalence_siblings_questioned"]
    assert "flash-encoder-label-elision" not in label["hard_invalidates_descendants"]


def test_promotion_prerequisites_come_from_the_queue(live_queue, live_rows):
    table = {row["candidate_id"]: row for row in cp.promotion_table(live_queue, live_rows)}
    ready = table["qwen27-fast-profile"]
    assert ready["status"] == "READY_PROTECTED"
    assert "PROTECTED_PASS" in ready["legal_next_statuses"]
    assert "PROTECTED_REJECT" in ready["legal_next_statuses"]
    assert any("promotion_rule" in p for p in ready["promotion_prerequisites"])
    assert any("protected_start_requires_existing_hcli_lease" in p for p in ready["promotion_prerequisites"])
    assert any("diagnostic_results_do_not_promote" in p for p in ready["promotion_prerequisites"])
    assert ready["can_enter_promotion_list"] is False
    blocked = table["flash-attention-gate-fusion"]
    assert blocked["status"] == "BLOCKED"
    assert blocked["legal_next_statuses"] == ["STATIC_ONLY"]
    assert blocked["can_enter_protected_pass"] is False
    assert any("blocked_reason" in r for r in blocked["rejection_reasons_from_queue"])
    # No invented statuses.
    known = set(live_queue["status_transitions"])
    for row in table.values():
        assert row["status"] in known
        assert set(row["legal_next_statuses"]) <= known


def test_blocked_candidates_are_not_measurement_cells(live_queue, live_rows):
    doc = cp.plan_from_queue(live_queue)
    plan = doc["staged_factorial_plan"]
    blocked = {r["candidate_id"] for r in live_rows if r["status"] == "BLOCKED"}
    scheduled_members = {
        ident
        for cell in plan["cells"]
        if cell["status"] == "SCHEDULED"
        for ident in cell["candidates"]
    }
    assert scheduled_members.isdisjoint(blocked)
    unscheduled = {row["candidate_id"] for row in plan["blocked_unscheduled"]}
    assert unscheduled == blocked


def test_protected_batch_is_exact_and_fail_closed(live_queue):
    doc = cp.plan_from_queue(live_queue)
    batch = doc["protected_batch"]

    assert batch["status"] == "WAITING_FOR_AUTHORITY"
    assert batch["frontier_snapshot"]["queue_candidate_count"] == len(
        live_queue["candidates"]
    )
    assert batch["qwen_first_batch"]["count"] == 13
    assert batch["flash_return_batch"]["count"] == len(cp.PROTECTED_FLASH_RETURN_ORDER)
    assert batch["frontier_snapshot"]["flash_return_missing_ids"] == []

    qwen_ids = [
        row["candidate_id"]
        for row in batch["qwen_first_batch"]["run_order"]
    ]
    assert qwen_ids == list(cp.PROTECTED_QWEN_FIRST_ORDER)
    assert all(
        row["queue_status"] == "READY_PROTECTED"
        and row["execution_state"] == "READY_ON_AUTHORITY"
        and row["protected_command"]
        for row in batch["qwen_first_batch"]["run_order"]
    )

    flash_rows = batch["flash_return_batch"]["run_order"]
    assert [row["candidate_id"] for row in flash_rows] == list(
        cp.PROTECTED_FLASH_RETURN_ORDER
    )
    assert all(row["queue_status"] == "BLOCKED" for row in flash_rows)
    assert all(row["control_env"] for row in flash_rows)
    assert all(
        row["execution_state"]
        in {"WAITING_FOR_FLASH_AUTHORITY", "CONTINGENT_AFTER_SURVIVORS"}
        for row in flash_rows
    )
    full = next(
        row
        for row in flash_rows
        if row["candidate_id"] == "flash-p6-fused-down-shared-combine"
    )
    assert full["mutation_env"]["HAWKING_DSV4F_P6_FP4_DOWN_SHARED_COMBINE_FUSED"] == "1"
    stack = next(
        row
        for row in flash_rows
        if row["candidate_id"] == "flash-p6-fused-epilogue-stack"
    )
    assert stack["mutation_env"]["HAWKING_DSV4F_P6_FP4_DOWN_SHARED_COMBINE_FUSED"] == "1"
    assert "flash-p6-fused-down-shared-combine" in stack["requires_survivors"]

    assert batch["execution_authority"]["executes_benchmark"] is False
    assert batch["execution_authority"]["acquires_lease"] is False
    assert batch["current_environment"]["teacher_capture_rows"] == 0
    assert batch["current_environment"]["prospective_meta_bpw"] == 0.8871807728336929
    assert batch["current_environment"]["flash_physical_ebpw"] == "UNKNOWN"


def test_plan_from_queue_is_deterministic(live_queue):
    a = cp.plan_from_queue(live_queue)
    b = cp.plan_from_queue(live_queue)
    # Drop load-path provenance that is not part of the plan.
    a["input"]["loaded_from"] = None
    b["input"]["loaded_from"] = None
    a["input"]["queue_sha256"] = None
    b["input"]["queue_sha256"] = None
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_interactions_state_a_mechanism(live_rows):
    preds = cp.predict_interactions(live_rows)
    assert preds
    for pred in preds:
        assert pred["mechanism"]
        assert pred["kind"] in {"shared_region", "shared_resource", "precondition"}
    pairs = {frozenset((p["a"], p["b"])) for p in preds}
    # Ceremony overlap is predicted non-additive.
    assert frozenset({"qwen27-pipeline-state-elision", "qwen27-pipeline-cache-reuse"}) in pairs
    # GQA producer/consumer.
    assert frozenset({"qwen27-attention-gate-fusion", "qwen27-gqa-qkv-fusion"}) in pairs


def test_receipt_records_recovered_and_gaps(live_queue):
    doc = cp.plan_from_queue(live_queue)
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    paths = {r["path"] for r in doc["recovered_implementation"]}
    assert "tools/accelerator/physical_qualification.py" in paths
    assert "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json" in paths

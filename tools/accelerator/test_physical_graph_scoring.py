from architecture_atlas import build_atlas
from hcli.physical_graph import (
    NR_PRIMITIVES,
    apply_architecture_atlas,
    compile_physical_graph,
    score_physical_candidates,
)


def test_scoring_joins_representation_device_grouping_and_transfer_without_nominal_utilization():
    result = score_physical_candidates(
        [
            {
                "id": "metal-packed",
                "representation": "Q4",
                "device": "gpu",
                "grouping": "fused",
                "transfer_boundary": "device_resident",
                "complete_token_ns": 100,
                "resident_bytes": 200,
                "dispatches": 8,
                "capability_verified": True,
                "benchmark_class": "DIAGNOSTIC_RELATIVE",
                "nominal_utilization": 0.99,
            },
            {
                "id": "ane-slower",
                "representation": "Q4",
                "device": "ane",
                "grouping": "fused",
                "transfer_boundary": "host_roundtrip",
                "complete_token_ns": 120,
                "resident_bytes": 100,
                "dispatches": 3,
                "capability_verified": True,
                "benchmark_class": "PROTECTED_ABSOLUTE",
                "nominal_utilization": 0.10,
            },
        ],
        require_protected=True,
    )
    assert result["winner"] == "ane-slower"
    assert result["promotion_allowed"] is True
    assert result["nominal_utilization_is_not_authority"] is True
    assert result["candidates"][0]["ineligibility_reasons"] == ["protected_absolute_evidence_required"]


def test_scoring_rejects_unmeasured_or_fallback_candidates():
    result = score_physical_candidates(
        [
            {"id": "unknown", "capability_verified": True},
            {
                "id": "fallback",
                "complete_token_ns": 10,
                "capability_verified": True,
                "fallback_count": 1,
            },
        ]
    )
    assert result["winner"] is None
    assert result["promotion_allowed"] is False
    assert result["candidates"][0]["eligible_for_selection"] is False
    assert "complete_useful_work_unmeasured" in result["candidates"][0]["ineligibility_reasons"]
    assert "forbidden_or_unreported_fallback" in result["candidates"][1]["ineligibility_reasons"]


def test_scoring_accepts_existing_hcli_protected_boundary_alias():
    result = score_physical_candidates(
        [
            {
                "id": "protected-alias",
                "complete_token_ns": 42,
                "capability_verified": True,
                "benchmark_class": "QUALIFIED_PROTECTED",
            }
        ],
        require_protected=True,
    )
    assert result["winner"] == "protected-alias"
    assert result["promotion_allowed"] is True


def test_compiler_exposes_nr_primitives_and_scores_supplied_physical_plans():
    graph = compile_physical_graph(
        {
            "model_id": "flash-next",
            "organs": [],
            "physical_candidates": [
                {
                    "id": "candidate",
                    "representation": "literal",
                    "device": "gpu",
                    "operation_grouping": "species",
                    "complete_useful_ns": 1,
                    "capability_verified": True,
                    "benchmark_class": "PROTECTED_ABSOLUTE",
                }
            ],
        }
    )
    assert graph["representation"]["nr_primitives"] == list(NR_PRIMITIVES)
    assert graph["physical_plan_score"]["winner"] == "candidate"
    assert graph["physical_plan_score"]["promotion_allowed"] is True


def test_compiler_projects_atlas_behaviors_without_promoting_them():
    atlas = build_atlas()
    graph = compile_physical_graph(
        {"model_id": "Qwen3.8-27B sealed resident", "organs": []},
        architecture_atlas=atlas,
        backend="metal",
    )

    projection = graph["architecture_repatriation"]
    assert projection["atlas_fingerprint"] == atlas["fingerprint"]
    assert "persistent_physical_region" in projection["selected_behavior_ids"]
    assert "PersistentPhysicalRegion" in projection["selected_primitives"]
    assert graph["execution_policy"]["architecture_repatriation"]["stationarity_is_explicit"] is True
    assert graph["execution_policy"]["architecture_repatriation"]["move_or_recompute_is_explicit"] == "costed_dependency_query"
    assert graph["execution_policy"]["device_count_is_not_speed_authority"] is True
    assert "interference" in graph["execution_policy"]["selection_costs"]
    assert graph["representation"]["layout_algebra"]["logical_tensor"] == "organ_semantics"
    assert graph["representation"]["stationarity"]["status"] == "candidate_until_measured"
    assert graph["execution_policy"]["architecture_repatriation"]["measurement_authority"].startswith("protected")
    assert graph["architecture_repatriation"]["promotion"] == "not_allowed_without_protected_receipt"


def test_atlas_projection_refuses_rejected_and_blocked_entries():
    atlas = build_atlas()
    atlas["entries"] = [
        {**atlas["entries"][0], "status": "BLOCKED"},
        {**atlas["entries"][1], "status": "REJECTED"},
    ]
    graph = apply_architecture_atlas(
        {"model_id": "Flash", "representation": {}, "execution_policy": {}},
        atlas,
        backend="metal",
    )
    assert graph["architecture_repatriation"]["selected_behavior_ids"] == []

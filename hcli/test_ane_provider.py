from __future__ import annotations

import json

from hcli.ane_provider import ANEProvider
from hcli.physical_graph import compile_physical_graph


def test_ane_provider_requires_plan_and_measurement_for_promotion():
    provider = ANEProvider(
        {"neural_engine_present": True, "mlcomputeplan": {"status": "PLANNED", "operations": [{"operator": "matmul", "preferred": "NEURAL_ENGINE", "supported": ["NEURAL_ENGINE"]}]}},
        {"status": "MEASURED", "graphs": [{"operation": "matmul", "placement": {"preferred": "NEURAL_ENGINE"}}]},
    )
    result = provider.score_candidate(operation="matmul", shape=[1, 2560], complete_token_ns=100)
    assert result["eligible_for_promotion"] is True
    assert result["total_candidate_ns"] == 100


def test_ane_provider_stays_plan_only_without_evidence():
    provider = ANEProvider({"neural_engine_present": True}, {})
    result = provider.score_candidate(operation="sdpa", shape=[1, 24, 1, 256], complete_token_ns=None)
    assert result["eligible_for_promotion"] is False
    assert result["total_candidate_ns"] is None


def test_physical_graph_accepts_ane_provider_context():
    provider = ANEProvider({"neural_engine_present": True}, {})
    graph = compile_physical_graph({"model_id": "flash", "organs": []}, provider=provider)
    assert "ane" in graph["device_placement"]["candidates"]
    assert graph["provider_context"]["kind"] == "ANEProvider"
    assert graph["provider_context"]["private_interface_control"] == "forbidden"


def test_physical_graph_compiles_flash_acceleration_law():
    graph = compile_physical_graph({"model_id": "flash-next", "organs": []})
    policy = graph["execution_policy"]
    assert policy["process"] == "long_lived_executor"
    assert policy["state_handoff"]["fast_verified"] == "device_resident"
    assert policy["verification"]["divergence"].startswith("checkpoint_bisection")
    assert policy["promotion_metric"] == "measured_complete_useful_work"


def test_ane_characterization_is_model_scoped_and_keeps_missing_evidence_explicit():
    provider = ANEProvider(
        {
            "neural_engine_present": True,
            "mlcomputeplan": {
                "status": "PLANNED",
                "operations": [
                    {
                        "operator": "matmul",
                        "preferred": "NEURAL_ENGINE",
                        "supported": ["CPU", "NEURAL_ENGINE"],
                    }
                ],
            },
        },
        {"status": "MEASURED", "graphs": [{"operation": "matmul"}]},
    )
    result = provider.characterize_model(
        "flash-next",
        [
            {"id": "router", "operation": "matmul", "shape": [1, 512]},
            {"id": "norm", "operation": "rmsnorm", "shape": [1, 2560]},
        ],
        source_seal="flash-seal",
    )
    assert result["model_specific"] is True
    assert [row["organ_id"] for row in result["organs"]] == ["norm", "router"]
    assert result["promotion"]["status"] == "WITHHELD"
    assert "no_model_specific_protected_complete_work" in result["promotion"]["blockers"]
    assert result["mega_kernel"]["model_id"] == "flash-next"
    assert result["mega_kernel"]["not_a_single_vendor_kernel_claim"] is True
    router = next(row for row in result["organs"] if row["organ_id"] == "router")
    matrix = router["backend_matrix"]
    assert set(matrix) == {"cpu", "gpu", "ane"}
    assert matrix["cpu"]["measurement_state"] == "UNMEASURED"
    assert matrix["gpu"]["placement_evidence"] == "MODEL_SCOPED_RUN_REQUIRED"
    assert matrix["ane"]["placement_evidence"]["ane_supported"] is True
    assert all(not entry["promotion_eligible"] for entry in matrix.values())


def test_physical_graph_exposes_one_model_aware_mega_kernel_spine():
    graph = compile_physical_graph(
        {
            "model_id": "flash-next",
            "organs": [
                {"id": "z", "operation": "matmul"},
                {"id": "a", "operation": "rmsnorm"},
            ],
        },
        provider=ANEProvider({"neural_engine_present": True}, {}),
    )
    mega = graph["execution_policy"]["mega_kernel"]
    assert mega["schema"] == "hcli.physical_graph.mega_kernel.v1"
    assert [slot["organ_id"] for slot in mega["organ_slots"]] == ["a", "z"]
    assert graph["device_placement"]["mega_kernel"]["status"] == "PLAN_ONLY"
    assert "ane" in mega["backend_slots"]


def test_mega_kernel_models_recurrent_and_kv_state_as_persistent_decode_state():
    graph = compile_physical_graph(
        {
            "model_id": "flash-next",
            "organs": [
                {"id": "linear_attention_state", "operation": "state_update"},
                {"id": "full_attention_sdpa", "operation": "sdpa"},
            ],
        }
    )
    layout = graph["execution_policy"]["mega_kernel"]["decode_state_layout"]
    assert layout["position"] == 0
    assert layout["replay_is_not_decode"] is True
    assert "accepted_token" in layout["continuation_rule"]
    assert {row["kind"] for row in layout["regions"]} >= {
        "route_cache",
        "recurrent_state",
        "kv_cache",
    }
    assert "linear_attention_state.kv" not in {
        row["region_id"] for row in layout["regions"]
    }


def test_ane_model_card_carries_its_own_decode_state_layout():
    provider = ANEProvider({"neural_engine_present": True}, {})
    card = provider.characterize_model(
        "flash-next",
        [
            {"id": "linear_attention_state", "operation": "state_update"},
            {"id": "full_attention_sdpa", "operation": "sdpa"},
        ],
        source_seal="flash-seal",
    )
    layout = card["mega_kernel"]["decode_state_layout"]
    assert layout["replay_is_not_decode"] is True
    assert {row["kind"] for row in layout["regions"]} >= {
        "route_cache",
        "recurrent_state",
        "kv_cache",
    }

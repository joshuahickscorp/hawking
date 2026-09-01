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

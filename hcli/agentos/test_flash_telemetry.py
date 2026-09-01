"""Truth-bound tests for CUDA-architecture-to-Metal telemetry.

The graph is a planning surface, not a CUDA benchmark.  A source-native Metal
dispatch that has not achieved parity must stay diagnostic even when all of its
kernels were observed.
"""
from __future__ import annotations

import json

from hcli.agentos.flash_telemetry import emit_flash_telemetry


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _node(graph, node_id):
    return next(item for item in graph["nodes"] if item["id"] == node_id)


def test_blocked_source_native_layer_is_visible_but_not_promoted(tmp_path):
    root = tmp_path
    receipts = root / "receipts" / "headless"
    _write(receipts / "FLASH_NOETIC_EXACT_HYPERCONNECTION_NATIVE.json", {"status": "NOT_RUN"})
    _write(
        receipts / "FLASH_NOETIC_COMPLETE_LAYER0_NATIVE_PARITY.json",
        {
            "schema": "hawking.flash_complete_layer0.v1",
            "status": "BLOCKED",
            "qualification": "BLOCKED",
            "execution": {
                "provider": "apple_metal",
                "native_source_bf16": True,
                "fallback_count": 0,
            },
            "native_kernels": [
                "gemv_native_bf16_seq",
                "qwen_next_qkv_split_rearrange_conv_l2",
                "qwen_next_ba_split_to_decay_beta_source_bf16",
                "qwen_next_gated_delta_decode_single",
                "qwen_next_deltanet_source_bf16_gated_rmsnorm",
                "moe_topk_gate",
                "qwen_next_bf16_expert_gate_up_swiglu",
                "qwen_next_bf16_expert_down",
                "qwen_next_moe_weighted_sum",
                "qwen_next_moe_add_shared",
            ],
        },
    )

    output = emit_flash_telemetry(root)
    graph = json.loads((receipts / "CUDA_CAPABILITY_GRAPH.json").read_text(encoding="utf-8"))

    assert output["cuda_capability_graph"]["status"] == "PARTIAL_TRANSFER_MAP"
    assert graph["cuda_execution_observed"] is False
    assert graph["promotion_allowed"] is False
    assert graph["diagnostic_source_execution"]["observed"] is True
    for node_id in ("source_bf16_gemv", "recurrent_state_update", "source_native_moe_wave"):
        assert _node(graph, node_id)["metal_status"] == "DIAGNOSTIC_OBSERVED"


def test_missing_kernel_does_not_create_a_diagnostic_capability(tmp_path):
    root = tmp_path
    receipts = root / "receipts" / "headless"
    _write(receipts / "FLASH_NOETIC_EXACT_HYPERCONNECTION_NATIVE.json", {"status": "NOT_RUN"})
    _write(
        receipts / "FLASH_NOETIC_COMPLETE_LAYER0_NATIVE_PARITY.json",
        {
            "status": "BLOCKED",
            "execution": {
                "provider": "apple_metal",
                "native_source_bf16": True,
                "fallback_count": 0,
            },
            "native_kernels": ["gemv_native_bf16_seq"],
        },
    )

    emit_flash_telemetry(root)
    graph = json.loads((receipts / "CUDA_CAPABILITY_GRAPH.json").read_text(encoding="utf-8"))

    assert _node(graph, "source_bf16_gemv")["metal_status"] == "DIAGNOSTIC_OBSERVED"
    assert _node(graph, "recurrent_state_update")["metal_status"] == "ABSENT"
    assert _node(graph, "source_native_moe_wave")["metal_status"] == "ABSENT"


def test_passed_source_layer_promotes_only_bounded_apple_nodes(tmp_path):
    root = tmp_path
    receipts = root / "receipts" / "headless"
    _write(receipts / "FLASH_NOETIC_EXACT_HYPERCONNECTION_NATIVE.json", {"status": "NOT_RUN"})
    _write(
        receipts / "FLASH_NOETIC_COMPLETE_LAYER0_NATIVE_PARITY.json",
        {
            "status": "PASSED",
            "parity": {"passed": True},
            "execution": {"provider": "apple_metal", "native_source_bf16": True, "fallback_count": 0},
            "native_kernels": [
                "gemv_native_bf16_seq",
                "qwen_next_qkv_split_rearrange_conv_l2",
                "qwen_next_ba_split_to_decay_beta_source_bf16",
                "qwen_next_gated_delta_decode_single",
                "qwen_next_deltanet_source_bf16_gated_rmsnorm",
                "moe_topk_gate",
                "qwen_next_bf16_expert_gate_up_swiglu",
                "qwen_next_bf16_expert_down",
                "qwen_next_moe_weighted_sum",
                "qwen_next_moe_add_shared",
            ],
        },
    )

    emit_flash_telemetry(root)
    graph = json.loads((receipts / "CUDA_CAPABILITY_GRAPH.json").read_text(encoding="utf-8"))

    assert graph["cuda_execution_observed"] is False
    assert graph["promotion_allowed"] is False
    assert graph["diagnostic_source_execution"]["parity_passed"] is True
    for node_id in ("source_bf16_gemv", "recurrent_state_update", "source_native_moe_wave"):
        assert _node(graph, node_id)["metal_status"] == "FUNCTIONAL"


def test_additional_source_layer_receipts_are_visible_without_cuda_promotion(tmp_path):
    root = tmp_path
    receipts = root / "receipts" / "headless"
    _write(receipts / "FLASH_NOETIC_EXACT_HYPERCONNECTION_NATIVE.json", {"status": "NOT_RUN"})
    _write(receipts / "FLASH_NOETIC_COMPLETE_LAYER0_NATIVE_PARITY.json", {
        "status": "PASSED",
        "parity": {"passed": True},
        "source": {"layer_index": 0},
        "execution": {"provider": "apple_metal", "native_source_bf16": True, "fallback_count": 0},
        "native_kernels": [],
    })
    _write(receipts / "FLASH_NOETIC_COMPLETE_LAYER2_NATIVE_PARITY.json", {
        "status": "PASSED",
        "parity": {"passed": True},
        "source": {"layer_index": 2},
        "execution": {"provider": "apple_metal", "native_source_bf16": True, "fallback_count": 0},
    })

    emit_flash_telemetry(root)
    graph = json.loads((receipts / "CUDA_CAPABILITY_GRAPH.json").read_text(encoding="utf-8"))
    diagnostic = graph["diagnostic_source_execution"]
    assert diagnostic["source_parity_layer_count"] == 2
    assert diagnostic["additional_source_parity_layers"][0]["layer_index"] == 2
    assert diagnostic["additional_source_parity_layers"][0]["parity_passed"] is True
    assert graph["cuda_execution_observed"] is False
    assert graph["promotion_allowed"] is False


def test_timing_telemetry_marks_missing_source_bench_state_unknown(tmp_path):
    root = tmp_path
    exact = root / "exact.json"
    complete = root / "complete.json"
    _write(exact, {
        "status": "PASSED",
        "gpu_timing": {"device": "Apple test GPU", "graph_gpu_ns_median": 17},
        "physical_graph": {},
    })
    _write(complete, {"status": "BLOCKED"})

    output = emit_flash_telemetry(
        root,
        exact_receipt=exact,
        complete_layer0_receipt=complete,
        gpu_work_ledger=root / "ledger.json",
        token_critical_path=root / "critical.json",
        cuda_capability_graph=root / "graph.json",
    )

    for name in ("gpu_work_ledger", "token_critical_path", "cuda_capability_graph"):
        artifact = json.loads(open(output[name]["path"], encoding="utf-8").read())
        bench = artifact["bench"]
        assert bench["state"] == "UNKNOWN"
        assert bench["machine"] == "Apple test GPU"
        assert bench["recorded_at"].endswith("Z")

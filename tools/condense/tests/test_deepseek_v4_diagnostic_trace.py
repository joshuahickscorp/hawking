"""Focused contracts for DeepSeek-V4 diagnostic component execution evidence."""
from __future__ import annotations

import sys
from threading import Lock
from types import SimpleNamespace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import deepseek_v4_gravity as gravity


class _Tokenizer:
    def encode(self, _prompt: str, *, add_special_tokens: bool) -> SimpleNamespace:
        assert add_special_tokens is False
        return SimpleNamespace(ids=[17])

    def decode(self, ids: list[int]) -> str:
        return f"token-{ids[0]}"


def _runtime_with_component_stubs() -> gravity.DeepSeekV4DiagnosticRuntime:
    """Exercise the real trace wiring without allocating the 5.3 GiB artifact."""

    runtime = object.__new__(gravity.DeepSeekV4DiagnosticRuntime)
    runtime.hc_mult = 4
    runtime.hc_iters = 20
    runtime.norm_eps = 1e-6
    runtime.position = 0
    runtime.load_ms = 0.0
    runtime.eos_id = None
    runtime.tokenizer = _Tokenizer()
    runtime._lock = Lock()

    def reset() -> None:
        runtime.position = 0
        runtime.last_routes = []
        runtime.last_attention_execution = None
        runtime.last_moe_execution = None

    def embedding(token_id: int) -> np.ndarray:
        assert token_id == 17
        return np.asarray([1.0, 2.0], dtype=np.float32)

    def hc_pre(_hidden: np.ndarray, _prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray([1.0, 2.0], dtype=np.float32),
            np.ones(4, dtype=np.float32),
            np.eye(4, dtype=np.float32),
        )

    def hc_post(_update: np.ndarray, _residual: np.ndarray, _post: np.ndarray, _comb: np.ndarray) -> np.ndarray:
        return np.ones((4, 2), dtype=np.float32)

    def vector(_name: str) -> np.ndarray:
        return np.ones(2, dtype=np.float32)

    def attention(_values: np.ndarray, position: int) -> np.ndarray:
        runtime.last_attention_execution = {
            "executed": True,
            "kind": "sparse_compressed_attention_cpu_fallback",
            "position": position,
            "window_key_count": 5,
            "main_compressor_cache_before": 1,
            "main_compressor_emitted": False,
            "main_compressor_cache_after": 1,
            "index_compressor_cache_before": 1,
            "index_compressor_emitted": False,
            "index_compressor_cache_after": 1,
            "index_query_executed": True,
            "compressed_index_count": 1,
            "compressed_key_count": 1,
            "compressed_key_indices": [0],
            "attention_key_count": 6,
        }
        return np.asarray([1.0, 2.0], dtype=np.float32)

    def moe(_values: np.ndarray) -> np.ndarray:
        runtime.last_routes = [3, 5, 7, 11, 13, 17]
        runtime.last_moe_execution = {
            "selected_route_ids": runtime.last_routes.copy(),
            "routed_expert_count": len(runtime.last_routes),
            "shared_expert_executed": True,
        }
        return np.asarray([1.0, 2.0], dtype=np.float32)

    runtime.reset = reset
    runtime._embedding = embedding
    runtime._hc_pre = hc_pre
    runtime._hc_post = hc_post
    runtime._vector = vector
    runtime._attention = attention
    runtime._moe = moe
    runtime._head_logits = lambda _hidden: np.asarray([0.0, 1.0], dtype=np.float32)
    return runtime


def test_generate_trace_carries_explicit_component_execution_without_parity_claim() -> None:
    runtime = _runtime_with_component_stubs()

    result = runtime.generate("trace", 1)

    assert result["trace_schema"] == "hawking.gravity.deepseek_v4.diagnostic_forward_trace.v2"
    component = result["trace"][0]["component_execution"]
    assert component["schema"] == "hawking.gravity.deepseek_v4.diagnostic_component_execution.v1"
    assert component["diagnostic_only"] is True
    assert component["embedding"] == {
        "executed": True,
        "tensor": "embed.weight",
        "source_token_id": 17,
    }
    assert component["hc_attention"]["sinkhorn_iterations"] == 20
    assert component["attention"]["compressed_index_count"] == 1
    assert component["attention"]["compressed_key_count"] == 1
    assert component["router"]["selected_route_ids"] == [3, 5, 7, 11, 13, 17]
    assert component["shared_expert"]["executed"] is True
    assert component["routed_experts"]["selected_route_count"] == 6
    assert component["hc_ffn"]["executed"] is True
    assert component["head"]["executed"] is True
    assert component["decoder_mode"]["linear_execution"] == "cpu_numpy_float32_matvec"
    assert component["activation_qat"] is False
    assert component["numeric_parity_v2_1"] == "not_proven"
    assert component["metal_dispatches"] == 0

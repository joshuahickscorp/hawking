from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kimi_k26_adapter as kimi  # noqa: E402


def test_int4_compressed_tensors_round_trip_with_padding() -> None:
    rng = np.random.default_rng(7)
    values = rng.integers(-8, 8, size=(2, 35), dtype=np.int8)
    packed = kimi.pack_int4(values)
    assert packed.shape == (2, 5)
    assert packed.dtype == np.int32
    np.testing.assert_array_equal(kimi.unpack_int4(packed, values.shape), values)


def test_noaux_router_uses_original_scores_for_combine() -> None:
    x = np.array([[1.0, 0.0]], dtype=np.float32)
    gate = np.array([[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    # Bias changes selection, but official combine weights come from the unbiased sigmoid scores.
    bias = np.array([-10.0, 0.0, 0.0, 10.0], dtype=np.float32)
    indices, weights = kimi.route_noaux_tc(x, gate, bias, top_k=2, scaling=2.827)
    assert indices.tolist() == [[3, 1]]
    raw = 1.0 / (1.0 + np.exp(-np.array([1.0, 3.0], dtype=np.float32)))
    np.testing.assert_allclose(weights[0], raw / raw.sum() * 2.827, rtol=1e-6)


def test_official_config_binding(tmp_path: Path) -> None:
    config = {
        "model_type": "kimi_k25",
        "vision_config": {"vt_num_hidden_layers": 27},
        "text_config": {
            "model_type": "kimi_k2", "hidden_size": 7168, "num_hidden_layers": 61,
            "n_routed_experts": 384, "n_shared_experts": 1, "num_experts_per_tok": 8,
            "max_position_embeddings": 262144, "q_lora_rank": 1536, "kv_lora_rank": 512,
            "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
            "scoring_func": "sigmoid", "topk_method": "noaux_tc", "norm_topk_prob": True,
            "n_group": 1, "topk_group": 1, "routed_scaling_factor": 2.827,
            "rms_norm_eps": 1e-5, "rope_theta": 50000.0,
            "quantization_config": {
                "format": "pack-quantized", "quant_method": "compressed-tensors",
                "config_groups": {"group_0": {"weights": {
                    "num_bits": 4, "group_size": 32, "symmetric": True,
                }}},
            },
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    binding = kimi.bind_config(path)
    assert binding["claim_boundary"] == "TEXT_CORE_ONLY"
    assert binding["quantization"]["num_bits"] == 4

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import qwen3_moe_adapter as A  # noqa: E402
import qwen_bpw_budget as B  # noqa: E402


REAL_CONFIG = {
    "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
    "hidden_size": 4096, "vocab_size": 151936, "num_hidden_layers": 94,
    "num_attention_heads": 64, "num_key_value_heads": 4, "head_dim": 128,
    "num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 1536,
    "tie_word_embeddings": False, "torch_dtype": "bfloat16",
}


def _analytic_index():
    g = A.geometry_from_config(REAL_CONFIG)
    names = A.expected_tensor_names(g)
    total = sum(math.prod(A.expected_shape(g, name)) * 2 for name in names)
    return {"metadata": {"total_size": total},
            "weight_map": {name: "model-00001-of-00118.safetensors" for name in names}}


def test_whole_model_plan_reaches_120b_rate_without_hiding_nonexperts():
    plan = B.build_plan(REAL_CONFIG, _analytic_index())
    acct = plan["accounting"]
    assert plan["parent"]["parameters"] == 235_093_634_560
    assert acct["target_met"] is True
    assert acct["projected_whole_artifact_bpw"] <= 0.77
    assert acct["margin_bits"] > 0
    assert acct["container_metadata_reserve_bytes"] == 64 * 1024 * 1024
    # Attention/embed/head are explicitly compressed and billed; router/norm stability lanes stay BF16.
    assert plan["allocation"][A.ORGAN_Q]["spec"]["family"] == "product_quant"
    assert plan["allocation"][A.ORGAN_EMBED]["payload_bits"] > 0
    assert plan["allocation"][A.ORGAN_ROUTER]["spec"]["family"] == "kept_original"
    assert plan["allocation"][A.ORGAN_EXP_DOWN]["spec"]["k"] == 16


def test_expert_organ_allocation_is_below_target_but_not_mislabeled_whole_model():
    n = 4096 * 1536
    gate = B._pq_bits((1536, 4096), dim=32, subspaces=4, k=8)
    up = B._pq_bits((1536, 4096), dim=32, subspaces=4, k=8)
    down = B._pq_bits((4096, 1536), dim=16, subspaces=4, k=16, budget_frac=0.03)
    assert (gate + up + down) / (3 * n) < B.TARGET_WHOLE_BPW
    assert "expert-only BPW" in B.build_plan(REAL_CONFIG, _analytic_index())[
        "quality_strategy"
    ]["forbidden_shortcut"]

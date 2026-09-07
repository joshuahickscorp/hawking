"""G016 state-axis controls: GQA must not report MHA, MLA must not use n_heads*head_dim."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "future"))

import state_axis  # noqa: E402


def test_gqa_is_not_full_mha():
    """Qwen3-0.6B-shaped GQA: 28 layers, 16 heads, 8 kv, head_dim 128, bf16.

    Per-token state is n_layers * n_kv_heads * head_dim * 2 * dtype_bytes
    = 28 * 8 * 128 * 2 * 2 = 114688. Full MHA would be twice that (grouping 2).
    """
    cfg = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "torch_dtype": "bfloat16",
        "max_position_embeddings": 40960,
    }
    row = state_axis.measure_config(cfg, path="test-gqa")
    gqa = 28 * 8 * 128 * 2 * 2
    mha = 28 * 16 * 128 * 2 * 2
    assert gqa == 114688
    assert mha == 229376
    assert row["state_class"] == "KV_GQA"
    assert row["state_bytes_per_token"] == 114688
    assert row["state_bytes_per_token"] != mha
    assert mha / row["state_bytes_per_token"] == 2


def test_mla_is_not_sized_by_head_dim_times_n_heads():
    """Kimi-VL-shaped MLA: 27 layers, kv_lora_rank 512, qk_rope_head_dim 64, bf16.

    Per-token state is n_layers * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes
    = 27 * (512 + 64) * 2 = 31104. Sizing by n_heads * head_dim (16 * 128)
    would yield 27 * 16 * 128 * 2 * 2 = 221184.
    """
    cfg = {
        "architectures": ["KimiVLForConditionalGeneration"],
        "model_type": "kimi_vl",
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
        "hidden_size": 2048,
        "torch_dtype": "bfloat16",
        "max_position_embeddings": 131072,
    }
    row = state_axis.measure_config(cfg, path="test-mla")
    mla = 27 * (512 + 64) * 2
    wrong = 27 * 16 * 128 * 2 * 2
    assert mla == 31104
    assert wrong == 221184
    assert row["state_class"] == "KV_MLA"
    assert row["state_bytes_per_token"] == 31104
    assert row["state_bytes_per_token"] != wrong


def test_campaign_pareto_reports_non_null_state():
    """The frontier must actually ingest G016_STATE_AXIS, not leave `state` at zero."""
    import campaign_pareto as cp

    receipt = ROOT / "receipts" / "future" / "G016_STATE_AXIS.json"
    assert receipt.is_file(), "G016_STATE_AXIS.json missing"
    doc = json.loads(receipt.read_text())
    qwen = next(r for r in doc["rows"] if r["slug"].startswith("Qwen--Qwen3-0.6B@"))
    assert qwen["state_bytes_per_token"] == 114688

    harvested = cp.harvest()
    debt = harvested["measurement_debt"]
    assert "state" not in debt["unmeasured_for_everyone"], debt["unmeasured_for_everyone"]
    assert debt["n_with_axis"]["state"] > 0
    scored = harvested["scored"]
    with_state = [p for p in scored if "state" in p["values"]]
    assert with_state, "harvest produced no comparable state values"
    # The Qwen3-0.6B GQA number must reproduce from the receipt through harvest.
    hits = [p for p in with_state if "Qwen3-0.6B" in str(p.get("name", "")) or "Qwen3-0.6B" in str(p.get("slug", ""))]
    assert hits, [p["name"] for p in with_state[:8]]
    assert hits[0]["values"]["state"] == 114688.0

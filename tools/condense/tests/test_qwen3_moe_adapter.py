#!/usr/bin/env python3.12
"""Tests for the Qwen3-235B-A22B MoE organ adapter.

Two layers of validation:
  1. SYNTHETIC TWIN: a tiny qwen3_moe-shaped config (2 layers, 4 experts, top-2, hidden 16) plus a
     synthesized index. Validates organ mapping, shape derivation, name generation, the byte-exact
     cross-check, and top-k routing shape logic end-to-end in milliseconds (no weights, no source).
  2. REAL INDEX: assertions against the downloaded metadata in models/qwen3-235b-a22b/_meta/,
     confirming the real Qwen3 tensor-name scheme, the expert/router names, and the ~235B / ~438 GiB
     grand total. Skipped cleanly if _meta is not present.
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
from pathlib import Path

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import qwen3_moe_adapter as A   # noqa: E402

REPO = Path(__file__).resolve().parents[3]
META = REPO / "models" / "qwen3-235b-a22b" / "_meta"


# --------------------------------------------------------------------------- #
# Synthetic twin helpers
# --------------------------------------------------------------------------- #
def _tiny_config():
    return {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": 16,
        "vocab_size": 40,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,                 # decoupled: q_out = 4*8 = 32 != hidden 16
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 6,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }


def _synth_index(geom: A.Geometry):
    """Synthesize a HF-style index (weight_map + metadata.total_size) for the tiny config."""
    names = A.expected_tensor_names(geom)
    weight_map = {}
    total = 0
    for i, name in enumerate(sorted(names)):
        weight_map[name] = f"model-{(i % 2) + 1:05d}-of-00002.safetensors"
        total += math.prod(A.expected_shape(geom, name)) * geom.dtype_bytes
    return {"metadata": {"total_size": total}, "weight_map": weight_map}


# --------------------------------------------------------------------------- #
# Synthetic-twin tests
# --------------------------------------------------------------------------- #
def test_geometry_decoupled_head_dim():
    g = A.geometry_from_config(_tiny_config())
    assert g.q_out == 32 and g.kv_out == 16
    assert g.head_dim == 8 and g.hidden_size == 16


def test_classify_all_organ_classes():
    assert A.classify_tensor("model.embed_tokens.weight").organ_class == A.ORGAN_EMBED
    assert A.classify_tensor("lm_head.weight").organ_class == A.ORGAN_LM_HEAD
    assert A.classify_tensor("model.norm.weight").organ_class == A.ORGAN_FINAL_NORM
    c = A.classify_tensor("model.layers.3.self_attn.q_proj.weight")
    assert c.organ_class == A.ORGAN_Q and c.layer == 3 and c.expert is None
    c = A.classify_tensor("model.layers.1.mlp.gate.weight")
    assert c.organ_class == A.ORGAN_ROUTER and c.layer == 1
    c = A.classify_tensor("model.layers.2.mlp.experts.5.down_proj.weight")
    assert c.organ_class == A.ORGAN_EXP_DOWN and c.layer == 2 and c.expert == 5
    assert A.classify_tensor("model.layers.0.self_attn.q_norm.weight").organ_class == A.ORGAN_QNORM
    assert A.classify_tensor("model.layers.0.self_attn.k_norm.weight").organ_class == A.ORGAN_KNORM


def test_classify_rejects_garbage():
    with pytest.raises(A.Qwen3MoeAdapterError):
        A.classify_tensor("model.layers.0.self_attn.rotary.inv_freq")
    with pytest.raises(A.Qwen3MoeAdapterError):
        A.classify_tensor("totally.bogus.name")


def test_expected_shapes():
    g = A.geometry_from_config(_tiny_config())
    assert A.expected_shape(g, "model.embed_tokens.weight") == (40, 16)
    assert A.expected_shape(g, "lm_head.weight") == (40, 16)
    assert A.expected_shape(g, "model.layers.0.self_attn.q_proj.weight") == (32, 16)
    assert A.expected_shape(g, "model.layers.0.self_attn.k_proj.weight") == (16, 16)
    assert A.expected_shape(g, "model.layers.0.self_attn.o_proj.weight") == (16, 32)
    assert A.expected_shape(g, "model.layers.0.self_attn.q_norm.weight") == (8,)
    assert A.expected_shape(g, "model.layers.0.mlp.gate.weight") == (4, 16)
    assert A.expected_shape(g, "model.layers.0.mlp.experts.0.gate_proj.weight") == (6, 16)
    assert A.expected_shape(g, "model.layers.0.mlp.experts.0.up_proj.weight") == (6, 16)
    assert A.expected_shape(g, "model.layers.0.mlp.experts.0.down_proj.weight") == (16, 6)


def test_synthetic_inventory_and_cross_check():
    cfg = _tiny_config()
    g = A.geometry_from_config(cfg)
    idx = _synth_index(g)
    inv = A.build_inventory(cfg, idx)
    # every tensor in the index is classified and present in the inventory
    assert len(inv.tensors) == len(idx["weight_map"])
    # each layer contributes: q,k,v,o,q_norm,k_norm,iln,pln,gate = 9 attn/norm/router organs
    # plus 4 experts * 3 proj = 12 expert tensors -> 21 per layer; +3 top-level (embed, norm, lm_head)
    assert len(inv.tensors) == 2 * 21 + 3
    # expert organ classes aggregate across all layers*experts
    assert inv.per_class_count[A.ORGAN_EXP_GATE] == 2 * 4
    assert inv.per_class_count[A.ORGAN_ROUTER] == 2
    # byte-exact cross-check must hold for the synthesized index
    assert inv.cross_check_ok()
    assert inv.grand_bytes == idx["metadata"]["total_size"]
    assert inv.grand_bytes == inv.grand_params * g.dtype_bytes


def test_synthetic_name_verification():
    cfg = _tiny_config()
    g = A.geometry_from_config(cfg)
    idx = _synth_index(g)
    ver = A.verify_index_names(g, idx)
    assert ver.ok and not ver.missing and not ver.unexpected
    assert ver.router_present and ver.q_norm_present and ver.k_norm_present
    assert ver.router_example == "model.layers.0.mlp.gate.weight"


def test_synthetic_name_verification_detects_missing():
    cfg = _tiny_config()
    g = A.geometry_from_config(cfg)
    idx = _synth_index(g)
    idx["weight_map"].pop("model.layers.0.mlp.experts.0.gate_proj.weight")
    ver = A.verify_index_names(g, idx)
    assert not ver.ok
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in ver.missing


def test_topk_routing_shape():
    import numpy as np
    n_tok, n_exp, k = 5, 4, 2
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((n_tok, n_exp))
    idx, w = A.topk_route(logits, k, norm_topk_prob=True)
    assert idx.shape == (n_tok, k) and w.shape == (n_tok, k)
    # top-k indices are distinct per token and in range
    for row in idx:
        assert len(set(row.tolist())) == k and all(0 <= e < n_exp for e in row)
    # renormalized weights sum to ~1 per token
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)
    # each selected weight is the largest; without renorm weights are raw softmax probs <= 1
    idx2, w2 = A.topk_route(logits, k, norm_topk_prob=False)
    assert (w2 <= 1.0 + 1e-6).all()
    assert np.array_equal(idx, idx2)


def test_topk_rejects_bad_k():
    import numpy as np
    logits = np.zeros((3, 4))
    with pytest.raises(A.Qwen3MoeAdapterError):
        A.topk_route(logits, 5)     # k > n_experts
    with pytest.raises(A.Qwen3MoeAdapterError):
        A.topk_route(np.zeros(4), 2)  # not 2-D


def test_bounded_read_stub_untested_pending_source(tmp_path):
    cfg = _tiny_config()
    g = A.geometry_from_config(cfg)
    idx = _synth_index(g)
    inv = A.build_inventory(cfg, idx)
    plan = A.plan_bounded_read(inv, "model.layers.0.mlp.experts.0.gate_proj.weight", tmp_path)
    assert plan.status == "untested-pending-source"
    assert plan.shard_file == idx["weight_map"]["model.layers.0.mlp.experts.0.gate_proj.weight"]


def test_bounded_read_executes_on_synthetic_shard(tmp_path):
    """The read path is exercised against a tiny SYNTHETIC safetensors file, never the real source."""
    shard = tmp_path / "tiny.safetensors"
    tname = "model.layers.0.mlp.gate.weight"
    shape = [4, 16]
    nbytes = math.prod(shape) * 2  # BF16
    header = {tname: {"dtype": "BF16", "shape": shape, "data_offsets": [0, nbytes]}}
    hb = json.dumps(header).encode()
    with open(shard, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        fh.write(b"\x00" * nbytes)
    out = A.read_one_tensor_bytes(shard, tname, max_bytes=8)
    assert out["dtype"] == "BF16" and out["shape"] == (4, 16)
    assert out["byte_count"] == nbytes and out["read_bytes"] == 8
    # header-only mode reads zero data bytes
    out0 = A.read_one_tensor_bytes(shard, tname)
    assert out0["read_bytes"] == 0


# --------------------------------------------------------------------------- #
# Real-index tests (against the downloaded _meta)
# --------------------------------------------------------------------------- #
_HAS_META = (META / "config.json").is_file() and (META / "model.safetensors.index.json").is_file()
real = pytest.mark.skipif(not _HAS_META, reason=f"real _meta not present at {META}")


@pytest.fixture(scope="module")
def real_inv():
    cfg = A.load_config(META)
    idx = A.load_index(META)
    return cfg, idx, A.build_inventory(cfg, idx), A.geometry_from_config(cfg)


@real
def test_real_architecture(real_inv):
    _, _, _, g = real_inv
    assert g.architecture == "Qwen3MoeForCausalLM"
    assert g.model_type == "qwen3_moe"
    assert g.num_hidden_layers == 94 and g.num_experts == 128 and g.num_experts_per_tok == 8
    assert g.num_attention_heads == 64 and g.num_key_value_heads == 4 and g.head_dim == 128
    assert g.hidden_size == 4096 and g.moe_intermediate_size == 1536 and g.vocab_size == 151936
    assert g.tie_word_embeddings is False and g.torch_dtype == "bfloat16"
    assert g.q_out == 8192 and g.kv_out == 512


@real
def test_real_name_scheme(real_inv):
    _, idx, _, g = real_inv
    ver = A.verify_index_names(g, idx)
    assert ver.ok, f"missing={ver.missing[:5]} unexpected={ver.unexpected[:5]}"
    assert ver.router_present and ver.q_norm_present and ver.k_norm_present
    # the real expert/router names exist verbatim in the index
    wm = idx["weight_map"]
    assert "model.layers.0.mlp.gate.weight" in wm
    assert "model.layers.93.mlp.experts.127.down_proj.weight" in wm
    assert "model.layers.0.self_attn.q_norm.weight" in wm
    assert "lm_head.weight" in wm and "model.embed_tokens.weight" in wm


@real
def test_real_grand_total_235b_438gib(real_inv):
    _, idx, inv, _ = real_inv
    # byte-exact: analytic params * 2 bytes == index total_size
    assert inv.cross_check_ok()
    assert inv.grand_bytes == idx["metadata"]["total_size"] == 470187269120
    assert inv.grand_params == 235093634560
    assert 234e9 < inv.grand_params < 236e9          # ~235B
    gib = inv.grand_bytes / 1024 ** 3
    assert 437.0 < gib < 439.0                        # ~438 GiB


@real
def test_real_shard_plan_and_counts(real_inv):
    _, idx, inv, g = real_inv
    assert len(inv.shard_files) == 118
    assert len(inv.tensors) == len(idx["weight_map"]) == 36945
    # every tensor maps to a shard that appears in the index weight_map
    valid = set(idx["weight_map"].values())
    assert all(t.shard_file in valid for t in inv.tensors)
    # experts dominate the parameter count (A22B active, 235B total)
    exp_params = (inv.per_class_params[A.ORGAN_EXP_GATE]
                  + inv.per_class_params[A.ORGAN_EXP_UP]
                  + inv.per_class_params[A.ORGAN_EXP_DOWN])
    assert exp_params / inv.grand_params > 0.9


@real
def test_real_bounded_read_pending(real_inv):
    _, _, inv, _ = real_inv
    plan = A.plan_bounded_read(inv, "model.layers.0.mlp.experts.0.gate_proj.weight", META)
    # only _meta is staged; the weight shards are absent -> the read is pending source
    assert plan.status == "untested-pending-source"
    assert plan.shard_file.endswith(".safetensors")

#!/usr/bin/env python3.12
"""Streamed, validated DeepSeek-V4-Flash contextual forward.

The apparatus the correction called for. transformers 5.14.1 ships the official
``deepseek_v4`` modeling, so instead of reimplementing MLA + DSA + hyper-connections in
NumPy and validating each, this loads real dequantized weights into the official
``DeepseekV4DecoderLayer`` and runs it. The forward is validated by construction: it is the
reference code, on the reference arithmetic, with the checkpoint's own weights.

Memory is handled by streaming: one layer's weights are resident at a time. A layer's fp4
experts dequantize to a single ``[num_experts, 2*inter, hidden]`` bf16 parameter, the block
runs, and the layer is released before the next loads. So the 160 GB source runs on 96 GiB
without ever holding more than one block, which is exactly why the earlier "memory wall"
framing was wrong.

The checkpoint-to-module name and format map (all shapes verified against both sides):

    attn_norm/ffn_norm            -> input_layernorm/post_attention_layernorm   (bf16)
    hc_attn_*/hc_ffn_*            -> attn_hc.*/ffn_hc.*                          (f32, exact)
    attn.wq_a/wq_b/wkv/wo_a/wo_b  -> self_attn.q_a_proj/q_b_proj/kv_proj/o_a/o_b (fp8 dequant)
    attn.q_norm/kv_norm/attn_sink -> self_attn.q_a_norm/kv_norm/sinks
    attn.compressor.*             -> self_attn.compressor.*                      (bf16 + f32 ape)
    attn.indexer.*                -> self_attn.compressor.indexer.*              (bf16 + fp8 wq_b)
    ffn.gate.weight               -> mlp.gate.weight
    ffn.experts.N.w1/w3/w2        -> mlp.experts.gate_up_proj (concat w1,w3) / down_proj  (fp4)
    ffn.shared_experts.w1/w3/w2   -> mlp.shared_experts.gate/up/down_proj        (fp8)

    reference LAYER [TOKENS]   run a streamed forward to LAYER, seal pre-MoE hidden
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CONDENSE = Path(__file__).resolve().parent
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import deepseek_v4_moe as ds

SOURCE = ds.SOURCE
HASH_LAYERS = 3  # layers 0-2 are hash-routed


def _config():
    """The official config, built from the complete config.json.

    Transformers' own DeepseekV4Config.__init__ derives everything the modules need from the
    flat fields: the main/compress rope_parameters, num_local_experts, intermediate_size and
    the per-layer mlp_layer_types (hash_moe for the first three, moe after). Stripping fields
    here only breaks that derivation, so nothing is stripped.
    """
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    return DeepseekV4Config(**json.loads((SOURCE / "config.json").read_text()))


def _tensor(name: str, index: dict):
    meta, raw = ds._read(name, index)
    dtype = meta["dtype"]
    if dtype == "BF16":
        return ds._bf16(raw, meta["shape"])
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(meta["shape"]).copy()
    if dtype == "I64":
        return np.frombuffer(raw, dtype=np.int64).reshape(meta["shape"]).copy()
    if dtype == "F8_E4M3":
        scale = np.frombuffer(ds._read(name.replace(".weight", ".scale"), index)[1],
                              dtype=np.uint8)
        sm = ds._read(name.replace(".weight", ".scale"), index)[0]["shape"]
        return ds.dequant_fp8_e4m3(np.frombuffer(raw, dtype=np.uint8).reshape(meta["shape"]),
                                   scale.reshape(sm))
    raise ValueError(f"{name}: unhandled dtype {dtype}")


# Module parameter -> checkpoint tensor. Iterating the module's own parameters means each
# attention type (sliding / compressed_sparse / heavily_compressed) loads exactly the
# tensors it has: a sliding layer has no compressor, a heavily-compressed layer has a
# compressor but no indexer, and the map covers all three without per-type branches.
_PARAM_MAP = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "attn_hc.fn": "hc_attn_fn", "attn_hc.base": "hc_attn_base", "attn_hc.scale": "hc_attn_scale",
    "ffn_hc.fn": "hc_ffn_fn", "ffn_hc.base": "hc_ffn_base", "ffn_hc.scale": "hc_ffn_scale",
    "self_attn.sinks": "attn.attn_sink",
    "self_attn.q_a_proj.weight": "attn.wq_a.weight",
    "self_attn.q_a_norm.weight": "attn.q_norm.weight",
    "self_attn.q_b_proj.weight": "attn.wq_b.weight",
    "self_attn.kv_proj.weight": "attn.wkv.weight",
    "self_attn.kv_norm.weight": "attn.kv_norm.weight",
    "self_attn.o_a_proj.weight": "attn.wo_a.weight",
    "self_attn.o_b_proj.weight": "attn.wo_b.weight",
    "self_attn.compressor.position_bias": "attn.compressor.ape",
    "self_attn.compressor.kv_proj.weight": "attn.compressor.wkv.weight",
    "self_attn.compressor.gate_proj.weight": "attn.compressor.wgate.weight",
    "self_attn.compressor.kv_norm.weight": "attn.compressor.norm.weight",
    "self_attn.compressor.indexer.position_bias": "attn.indexer.compressor.ape",
    "self_attn.compressor.indexer.kv_proj.weight": "attn.indexer.compressor.wkv.weight",
    "self_attn.compressor.indexer.gate_proj.weight": "attn.indexer.compressor.wgate.weight",
    "self_attn.compressor.indexer.kv_norm.weight": "attn.indexer.compressor.norm.weight",
    "self_attn.compressor.indexer.q_b_proj.weight": "attn.indexer.wq_b.weight",
    "self_attn.compressor.indexer.scorer.weights_proj.weight": "attn.indexer.weights_proj.weight",
    "mlp.gate.weight": "ffn.gate.weight",
    "mlp.gate.tid2eid": "ffn.gate.tid2eid",
    "mlp.shared_experts.gate_proj.weight": "ffn.shared_experts.w1.weight",
    "mlp.shared_experts.up_proj.weight": "ffn.shared_experts.w3.weight",
    "mlp.shared_experts.down_proj.weight": "ffn.shared_experts.w2.weight",
}
# The stacked expert tensors are assembled rather than copied one-to-one.
_EXPERT_PARAMS = {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}


def _load_layer(layer: int, module, index: dict) -> dict:
    import torch
    prefix = f"layers.{layer}"
    loaded, skipped = [], []

    with torch.no_grad():
        for name, target in module.named_parameters():
            if name in _EXPERT_PARAMS:
                continue
            mapped = _PARAM_MAP.get(name)
            if mapped is None:
                skipped.append(name)
                continue
            checkpoint = f"{prefix}.{mapped}"
            if checkpoint not in index:
                skipped.append(name)
                continue
            array = _tensor(checkpoint, index)
            tensor = torch.from_numpy(np.ascontiguousarray(array)).to(target.dtype)
            if tuple(tensor.shape) != tuple(target.shape):
                raise ValueError(f"{name}: {tuple(tensor.shape)} != {tuple(target.shape)}")
            target.copy_(tensor)
            loaded.append(name)

        # Experts: gate_up_proj = concat(w1, w3) per expert; down_proj = w2.
        n = module.mlp.experts.num_experts
        gate_up = np.stack([np.concatenate([
            _expert_mat(f"{prefix}.ffn.experts.{e}", "w1", index),
            _expert_mat(f"{prefix}.ffn.experts.{e}", "w3", index)], axis=0) for e in range(n)])
        down = np.stack([_expert_mat(f"{prefix}.ffn.experts.{e}", "w2", index)
                        for e in range(n)])
        module.mlp.experts.gate_up_proj.copy_(
            torch.from_numpy(gate_up).to(module.mlp.experts.gate_up_proj.dtype))
        module.mlp.experts.down_proj.copy_(
            torch.from_numpy(down).to(module.mlp.experts.down_proj.dtype))
        loaded += list(_EXPERT_PARAMS)

    # A skipped param that is not a known-absent compressor/indexer/tid2eid is a real gap.
    unexpected = [s for s in skipped if not any(
        k in s for k in ("compressor", "indexer", "tid2eid"))]
    if unexpected:
        raise ValueError(f"layer {layer}: unmapped parameters {unexpected}")
    return {"loaded": len(loaded), "skipped_absent": skipped}


def _expert_mat(prefix: str, proj: str, index: dict, *, fp8: bool = False) -> np.ndarray:
    wm, wraw = ds._read(f"{prefix}.{proj}.weight", index)
    sm, sraw = ds._read(f"{prefix}.{proj}.scale", index)
    scale = np.frombuffer(sraw, dtype=np.uint8).reshape(sm["shape"])
    if wm["dtype"] == "I8":
        return ds.dequant_fp4(np.frombuffer(wraw, dtype=np.int8).reshape(wm["shape"]), scale)
    return ds.dequant_fp8_e4m3(np.frombuffer(wraw, dtype=np.uint8).reshape(wm["shape"]), scale)


def streamed_forward(token_ids: np.ndarray, up_to_layer: int, *, capture=()):
    """Run layers 0..up_to_layer with one block resident at a time; capture pre-MoE hidden.

    Pre-MoE hidden at a layer is ``post_attention_layernorm(collapsed)`` where collapsed is
    the ffn hyper-connection's collapse of the post-attention 4-wide stream. It is captured
    by a forward hook on the module so the reference math is untouched.
    """
    import torch
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M
    config = _config()
    config._attn_implementation = "eager"
    index = ds._index()
    torch.set_grad_enabled(False)

    embed_meta, embed_raw = ds._read("embed.weight", index)
    embed = ds._bf16(embed_raw, embed_meta["shape"])
    seq = token_ids.shape[0]
    tokens = torch.from_numpy(token_ids).long().unsqueeze(0)
    hidden = torch.from_numpy(embed[token_ids]).to(torch.bfloat16).unsqueeze(0)
    streams = hidden.unsqueeze(2).expand(-1, -1, config.hc_mult, -1).contiguous()

    rotary = M.DeepseekV4RotaryEmbedding(config)
    position_ids = torch.arange(seq).unsqueeze(0)
    position_embeddings = {
        "main": rotary(hidden, position_ids=position_ids, layer_type="main"),
        "compress": rotary(hidden, position_ids=position_ids, layer_type="compress"),
    }
    attention_mask = torch.triu(torch.full((1, 1, seq, seq), float("-inf")), diagonal=1)

    captured = {}
    for layer_idx in range(up_to_layer + 1):
        module = M.DeepseekV4DecoderLayer(config, layer_idx).to(torch.bfloat16).eval()
        _load_layer(layer_idx, module, index)
        if layer_idx in capture:
            def hook(mod, inp, out, li=layer_idx):
                captured[li] = out.detach().float().numpy()
            handle = module.post_attention_layernorm.register_forward_hook(hook)
        streams = module(streams, input_ids=tokens,
                        position_embeddings=position_embeddings,
                        position_ids=position_ids, attention_mask=attention_mask,
                        past_key_values=None)
        if layer_idx in capture:
            handle.remove()
        del module
    return streams, captured


if __name__ == "__main__":
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    n_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    tokens = np.random.default_rng(4).integers(0, 129280, n_tokens, dtype=np.int64)
    _, captured = streamed_forward(tokens, layer, capture=(layer,))
    pre_moe = captured[layer]
    out = Path(__file__).resolve().parents[2] / "reports" / "condense" / "deepseek_v4_flash"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"pre_moe_hidden_L{layer:02d}.npy", pre_moe)
    print(json.dumps({
        "layer": layer, "tokens": n_tokens,
        "pre_moe_shape": list(pre_moe.shape),
        "pre_moe_rms": float(np.sqrt(np.mean(pre_moe ** 2))),
        "finite": bool(np.isfinite(pre_moe).all()),
        "saved": str(out / f"pre_moe_hidden_L{layer:02d}.npy"),
    }))

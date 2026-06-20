#!/usr/bin/env python3
"""gguf_to_hf.py — dequantize a GGUF back to an HF safetensors dir.

This is the TRUE iso-harness bridge for the STRAND-vs-GGUF iso-bpw study:
a GGUF (Q2_K / Q3_K_M / IQ3_S / Q4_K_M / f16) is read, every tensor is
dequantized to f32 with the gguf python lib's reference dequantizer, the
GGUF tensor names are mapped back to HF names, and the result is written as
an HF model dir (config/tokenizer copied from a reference base dir) that
STRAND's ops/eval-ppl.py can load and score on the SAME perplexity harness.

Tied embeddings: Qwen2.5-0.5B ties lm_head to token_embd, so we only emit
model.embed_tokens.weight; transformers reties on load (tie_word_embeddings).

Usage:
  gguf_to_hf.py <in.gguf> <base_hf_dir> <out_hf_dir>
"""
import json
import os
import shutil
import sys

import gguf
import numpy as np
import torch
from safetensors.torch import save_file


def build_reverse_map(n_layers: int):
    """GGUF tensor name -> HF tensor name, via gguf's own TensorNameMap."""
    from gguf.constants import MODEL_ARCH
    nm = gguf.get_tensor_name_map(MODEL_ARCH.QWEN2, n_layers)
    rev = {}
    # nm maps HF-ish keys -> gguf base name; we want gguf full name -> HF full name.
    # Easiest: enumerate HF names we expect and ask nm for the gguf name.
    hf_suffixes = [
        ("model.embed_tokens.weight", "token_embd.weight"),
        ("model.norm.weight", "output_norm.weight"),
        ("lm_head.weight", "output.weight"),
    ]
    for hf, gg in hf_suffixes:
        rev[gg] = hf
    per_layer = {
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.q_proj.bias": "attn_q.bias",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.k_proj.bias": "attn_k.bias",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.v_proj.bias": "attn_v.bias",
        "self_attn.o_proj.weight": "attn_output.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
    }
    for i in range(n_layers):
        for hf_suf, gg_suf in per_layer.items():
            rev[f"blk.{i}.{gg_suf}"] = f"model.layers.{i}.{hf_suf}"
    return rev


def main():
    in_gguf, base_dir, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(base_dir, "config.json")) as f:
        cfg = json.load(f)
    n_layers = cfg["num_hidden_layers"]
    rev = build_reverse_map(n_layers)

    r = gguf.GGUFReader(in_gguf)
    tensors = {}
    skipped = []
    for t in r.tensors:
        hf_name = rev.get(t.name)
        if hf_name is None:
            skipped.append(t.name)
            continue
        deq = gguf.quants.dequantize(t.data, t.tensor_type)  # f32 numpy, already HF-shaped
        arr = np.ascontiguousarray(deq, dtype=np.float32)
        tensors[hf_name] = torch.from_numpy(arr).to(torch.bfloat16)

    if skipped:
        print(f"[gguf2hf] skipped (no HF map): {skipped}", file=sys.stderr)

    save_file(tensors, os.path.join(out_dir, "model.safetensors"))

    # copy config + tokenizer so transformers can load the dir
    for fn in os.listdir(base_dir):
        if fn == "model.safetensors":
            continue
        src = os.path.join(base_dir, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, fn))

    print(f"[gguf2hf] wrote {len(tensors)} tensors -> {out_dir}/model.safetensors")


if __name__ == "__main__":
    main()

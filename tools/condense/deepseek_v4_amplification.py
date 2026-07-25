#!/usr/bin/env python3.12
"""The contraction-first amplification gate on DeepSeek-V4-Flash.

GLM-5.2's residual stream amplified every functional-student error at every magnitude, at
every stratum, which is why its escape died at depth. DeepSeek's residual is structurally
different: manifold-constrained hyper-connections carry four parallel streams mixed by a
doubly-stochastic Sinkhorn matrix. Whether that is contractive is the whole cross-parent
question, and it is now measurable because the forward is validated.

Method, on the validated streamed forward:

    run layers 0..L, capturing the 4-wide stream entering L and the real pre-MoE hidden
    fit the functional student at L (real contextual input -> real MoE output)
    the error direction is (student MoE output - teacher MoE output)
    for each magnitude fraction, substitute teacher_moe + fraction * error at L via an mlp
        output hook, and run L and the following `depth` layers for both teacher and perturbed
    measure the relative L2 of the collapsed-hidden deviation entering and leaving each layer

A per-layer geometric-mean amplification below one is contractive: the first architecture
where a per-layer functional student could compose. Above one at every magnitude reproduces
GLM. The number decides the parent.

    run LAYER [DEPTH] [TOKENS]
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
import deepseek_v4_reference as ref
import glm52_moe_student as student

OUT = Path(__file__).resolve().parents[2] / "reports" / "condense" / "deepseek_v4_flash"
MAGNITUDES = (0.25, 0.5, 1.0)


def _collapsed(streams):
    """A single-stream view of the 4-wide hidden for a scale-stable L2 comparison: the mean
    over the hc_mult streams, which is what the streams represent before mixing."""
    return streams.mean(axis=2)


def run(layer: int, depth: int = 2, seq_len: int = 48) -> dict:
    import torch
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M

    config = ref._config()
    config._attn_implementation = "eager"
    index = ds._index()
    torch.set_grad_enabled(False)

    tokens = np.random.default_rng(41).integers(0, 129280, seq_len, dtype=np.int64)
    embed_meta, embed_raw = ds._read("embed.weight", index)
    embed = ds._bf16(embed_raw, embed_meta["shape"])
    tok_t = torch.from_numpy(tokens).long().unsqueeze(0)
    hidden0 = torch.from_numpy(embed[tokens]).to(torch.bfloat16).unsqueeze(0)
    streams = hidden0.unsqueeze(2).expand(-1, -1, config.hc_mult, -1).contiguous()
    rotary = M.DeepseekV4RotaryEmbedding(config)
    pid = torch.arange(seq_len).unsqueeze(0)
    pe = {"main": rotary(hidden0, position_ids=pid, layer_type="main"),
          "compress": rotary(hidden0, position_ids=pid, layer_type="compress")}
    mask = torch.triu(torch.full((1, 1, seq_len, seq_len), float("-inf")), diagonal=1)
    call = dict(input_ids=tok_t, position_embeddings=pe, position_ids=pid,
                attention_mask=mask, past_key_values=None)

    # Fit the student at L: need real pre-MoE hidden and real MoE output. Run 0..L capturing
    # the pre-MoE hidden, and keep the module for L resident to reuse.
    captured = {}
    modules = {}
    for li in range(layer):
        module = M.DeepseekV4DecoderLayer(config, li).to(torch.bfloat16).eval()
        ref._load_layer(li, module, index)
        streams = module(streams, **call)
        del module
    entering = streams.clone()  # 4-wide stream entering layer L

    # Load layer L, run it once to capture the real pre-MoE hidden, keep it for the
    # substituted teacher/perturbed runs below.
    modules[layer] = M.DeepseekV4DecoderLayer(config, layer).to(torch.bfloat16).eval()
    ref._load_layer(layer, modules[layer], index)

    def capture_hook(mod, inp, out):
        captured["pre_moe"] = out.detach().float().numpy()
    handle = modules[layer].post_attention_layernorm.register_forward_hook(capture_hook)
    modules[layer](entering, **call)
    handle.remove()

    pre_moe = captured["pre_moe"].reshape(-1, ds.DIM)
    experts = {e: ds._load_expert(f"layers.{layer}.ffn.experts.{e}", index)
               for e in range(ds.N_ROUTED)}
    teacher_moe = ds.moe_forward(pre_moe.astype(np.float32), layer, index,
                                 experts=experts)["post_moe"]
    fitted = student.fit(pre_moe.astype(np.float32), teacher_moe.astype(np.float32),
                         hidden=1024, seed=17, replaced_weights=1)
    student_moe = student.apply_student(fitted["blob"], pre_moe.astype(np.float32))
    error = student_moe - teacher_moe
    student_skill = 1.0 - float(np.sum((student_moe - teacher_moe) ** 2)
                               / max(np.sum((teacher_moe - teacher_moe.mean(0)) ** 2), 1e-9))

    def moe_substitute(sub_output_np):
        sub = torch.from_numpy(sub_output_np.reshape(1, seq_len, ds.DIM)).to(torch.bfloat16)

        def hook(mod, inp, out):
            return sub
        return hook

    # Run layer L for teacher and each magnitude, substituting the MoE output, then continue.
    def run_layer(module, streams_in, sub_np):
        handle = module.mlp.register_forward_hook(moe_substitute(sub_np))
        out = module(streams_in, **call)
        handle.remove()
        return out

    teacher_stream = run_layer(modules[layer], entering, teacher_moe)
    rows = []
    for fraction in MAGNITUDES:
        perturbed_moe = teacher_moe + fraction * error
        pert_stream = run_layer(modules[layer], entering, perturbed_moe)

        factors, t_state, p_state = [], teacher_stream, pert_stream
        entering_rl2 = float(np.linalg.norm(_collapsed(p_state).float().numpy()
                                            - _collapsed(t_state).float().numpy())
                             / max(np.linalg.norm(_collapsed(t_state).float().numpy()), 1e-9))
        for step in range(depth):
            nxt = layer + 1 + step
            module = modules.get(nxt)
            if module is None:
                module = M.DeepseekV4DecoderLayer(config, nxt).to(torch.bfloat16).eval()
                ref._load_layer(nxt, module, index)
                modules[nxt] = module
            before = _relative(p_state, t_state)
            t_state = module(t_state, **call)
            p_state = module(p_state, **call)
            after = _relative(p_state, t_state)
            factors.append(after / max(before, 1e-9))
        rows.append({"fraction": fraction, "entering_relative_l2": entering_rl2,
                     "geometric_mean_amplification": float(np.exp(np.mean(np.log(factors)))),
                     "per_layer": factors})

    gains = [r["geometric_mean_amplification"] for r in rows]
    verdict = ("CONTRACTIVE" if all(g < 1.0 for g in gains)
               else "THRESHOLD_STABLE" if any(g < 1.0 for g in gains)
               else "EXPANSIVE_AT_EVERY_TESTED_MAGNITUDE")
    result = {
        "schema": "hawking.deepseek_v4.amplification.v1",
        "parent": "deepseek-ai/DeepSeek-V4-Flash", "layer": layer, "depth": depth,
        "seq_len": seq_len, "hc_mult": config.hc_mult,
        "forward": "validated streamed forward, official DeepseekV4DecoderLayer, real weights",
        "student_local_skill_at_L": student_skill,
        "rows": rows, "verdict": verdict,
        "min_amplification": min(gains), "max_amplification": max(gains),
        "worse_at_smaller": gains[0] > gains[-1] if len(gains) > 1 else None,
        "vs_glm": "GLM was EXPANSIVE_AT_EVERY_TESTED_MAGNITUDE at strata 3/38/74; this is the "
                  "hyper-connection residual's answer to the same question",
        "not_evidence_of": "capability; bounded tokens, one stratum, few magnitudes.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"DEEPSEEK_V4_AMPLIFICATION_L{layer:02d}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=float))
    return result


def _relative(p_state, t_state) -> float:
    import numpy as _np
    p = _collapsed(p_state).float().numpy()
    t = _collapsed(t_state).float().numpy()
    return float(_np.linalg.norm(p - t) / max(_np.linalg.norm(t), 1e-9))


if __name__ == "__main__":
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    seq_len = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    r = run(layer, depth, seq_len)
    print(json.dumps({k: v for k, v in r.items()
                      if k in ("verdict", "student_local_skill_at_L", "min_amplification",
                               "max_amplification", "worse_at_smaller")}, indent=2, default=float))
    for row in r["rows"]:
        print(f"  fraction {row['fraction']:.2f}  entering_rl2 {row['entering_relative_l2']:.5f}"
              f"  geo-amp {row['geometric_mean_amplification']:.4f}  per-layer "
              f"{[round(x,3) for x in row['per_layer']]}")

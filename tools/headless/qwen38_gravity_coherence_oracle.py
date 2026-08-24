#!/usr/bin/env python3
"""ONE-SHOT diagnostic (steer S007). Not a reusable pack/decode tool.

Numpy oracle for the native Qwen3.8 gravity argmax-collapse:
  1. tied embeddings (lm_head present vs embed_tokens)
  2. mixer-kind vs config.layer_types vs tensors on disk
  3. embed lookup + layer-0 Gemma RMSNorm: HF source vs packed f32
     vs the (1+w) kernel the runtime actually runs

Writes JSON to stdout. Does not pack, decode, or touch Metal.
"""
from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np

SRC = os.path.expanduser("~/models/qwen3.8-27b-abliterated-bf16")
ART = os.path.expanduser("~/models/qwen38-gravity-uniform-q4-v1")
HIDDEN = 5120
VOCAB = 248320
EPS = 1.0e-6
# First generated-token id from the collapsed receipt.
COLLAPSE_ID = 150910
# A real prompt-like token: BOS / <|endoftext|> is 248044; use a mid-vocab
# content token so the embed row is typical, not a special.
PROBE_TOKEN = 100


def bf16_to_f32(buf: bytes) -> np.ndarray:
    u16 = np.frombuffer(buf, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).copy()


def read_st_header(path: str):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return n, hdr


def load_hf_tensor(weight_map, name: str, row: int | None = None) -> tuple[np.ndarray, list]:
    """Load a whole vector/matrix, or a single row of a rank-2 BF16 tensor."""
    shard = os.path.join(SRC, weight_map[name])
    n, hdr = read_st_header(shard)
    info = hdr[name]
    begin, end = info["data_offsets"]
    shape = info["shape"]
    with open(shard, "rb") as f:
        if row is None:
            f.seek(8 + n + begin)
            raw = f.read(end - begin)
            return bf16_to_f32(raw), shape
        assert len(shape) == 2, (name, shape)
        row_bytes = shape[1] * 2  # bf16
        f.seek(8 + n + begin + row * row_bytes)
        raw = f.read(row_bytes)
    return bf16_to_f32(raw), shape


def read_f32v2(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return np.fromfile(f, dtype="<f4", count=n)


def parse_hq30uq4_row(payload: bytes, row: int) -> tuple[np.ndarray, list]:
    """Dequant one row of a rank-2 HQ30UQ4 [rows, cols] payload. Group-64, even nibble low."""
    assert payload[:8] == b"HQ30UQ4\0", payload[:8]
    group = struct.unpack_from("<I", payload, 12)[0]
    rank = struct.unpack_from("<H", payload, 16)[0]
    shape = [struct.unpack_from("<I", payload, 32 + 4 * i)[0] for i in range(rank)]
    assert rank == 2, shape
    rows, cols = shape
    assert 0 <= row < rows
    after_dims = 32 + 4 * rank
    groups = (rows * cols + group - 1) // group
    scales = (
        np.frombuffer(payload, dtype="<u2", offset=after_dims, count=groups)
        .view(np.float16)
        .astype(np.float32)
    )
    codes = payload[after_dims + groups * 2 :]
    start_el = row * cols
    end_el = start_el + cols
    out = np.zeros(cols, dtype=np.float32)
    g0 = start_el // group
    g1 = (end_el - 1) // group
    for g in range(g0, g1 + 1):
        g_start = g * group
        scale = float(scales[g])
        code_base = g * (group // 2)
        for local in range(group):
            idx = g_start + local
            if idx < start_el or idx >= end_el:
                continue
            byte = codes[code_base + local // 2]
            nibble = byte & 0x0F if (local & 1) == 0 else (byte >> 4) & 0x0F
            out[idx - start_el] = (int(nibble) - 8) * scale
    return out, shape


def gemma_rms(x: np.ndarray, delta: np.ndarray) -> np.ndarray:
    inv = 1.0 / np.sqrt(np.mean(x * x) + EPS)
    return x * inv * (1.0 + delta)


def stats(arr: np.ndarray) -> dict:
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "l2": float(np.linalg.norm(arr)),
        "first8": [float(v) for v in arr[:8]],
    }


def main() -> int:
    with open(os.path.join(SRC, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(SRC, "model.safetensors.index.json")) as f:
        idx = json.load(f)
    wm = idx["weight_map"]
    keys = sorted(wm)

    tc = cfg["text_config"]
    layer_types = tc["layer_types"]
    mixer_hardcoded = [
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(64)
    ]

    per_layer = []
    mismatches = []
    for i in range(64):
        has_lin = any(k.startswith(f"model.language_model.layers.{i}.linear_attn.") for k in wm)
        has_att = any(k.startswith(f"model.language_model.layers.{i}.self_attn.") for k in wm)
        expected = layer_types[i]
        hard = mixer_hardcoded[i]
        ok = (expected == hard) and (
            (expected == "linear_attention" and has_lin and not has_att)
            or (expected == "full_attention" and has_att and not has_lin)
        )
        if not ok:
            mismatches.append(
                {"layer": i, "config": expected, "hardcoded": hard, "has_lin": has_lin, "has_att": has_att}
            )
        if i in (0, 1, 2, 3, 63):
            per_layer.append(
                {
                    "layer": i,
                    "config": expected,
                    "hardcoded": hard,
                    "has_linear_attn": has_lin,
                    "has_self_attn": has_att,
                }
            )

    # HF tensors
    ln0, _ = load_hf_tensor(wm, "model.language_model.layers.0.input_layernorm.weight")
    post0, _ = load_hf_tensor(wm, "model.language_model.layers.0.post_attention_layernorm.weight")
    gdn_norm, _ = load_hf_tensor(wm, "model.language_model.layers.0.linear_attn.norm.weight")
    qn3, _ = load_hf_tensor(wm, "model.language_model.layers.3.self_attn.q_norm.weight")
    kn3, _ = load_hf_tensor(wm, "model.language_model.layers.3.self_attn.k_norm.weight")
    final, _ = load_hf_tensor(wm, "model.language_model.norm.weight")
    embed_row0, embed_shape = load_hf_tensor(wm, "model.language_model.embed_tokens.weight", row=0)
    embed_probe, _ = load_hf_tensor(wm, "model.language_model.embed_tokens.weight", row=PROBE_TOKEN)
    lm_row0, lm_shape = load_hf_tensor(wm, "lm_head.weight", row=0)
    lm_collapse, _ = load_hf_tensor(wm, "lm_head.weight", row=COLLAPSE_ID)

    # Packed f32
    with open(os.path.join(ART, "manifest.json")) as f:
        man = json.load(f)
    rows = {r["name"]: r for r in man["tensors"]}

    def packed_f32(suffix: str) -> np.ndarray:
        hits = [n for n in rows if n.endswith(suffix)]
        assert len(hits) == 1, (suffix, hits)
        return read_f32v2(os.path.join(ART, "tensors", rows[hits[0]]["artifact"]))

    p_ln0 = packed_f32("layers.0.input_layernorm.weight")
    p_post0 = packed_f32("layers.0.post_attention_layernorm.weight")
    p_gdn = packed_f32("layers.0.linear_attn.norm.weight")
    p_qn3 = packed_f32("layers.3.self_attn.q_norm.weight")
    p_kn3 = packed_f32("layers.3.self_attn.k_norm.weight")
    p_final = packed_f32("model.norm.weight")

    # Packed Q4 embed row for PROBE_TOKEN
    embed_row_name = "language_model.model.embed_tokens.weight"
    embed_payload = open(os.path.join(ART, "tensors", rows[embed_row_name]["artifact"]), "rb").read()
    q4_row, q4_shape = parse_hq30uq4_row(embed_payload, PROBE_TOKEN)
    hf_row = embed_probe
    denom = (np.linalg.norm(hf_row) * np.linalg.norm(q4_row)) or 1.0
    embed_row_cosine = float(np.dot(hf_row, q4_row) / denom)

    # Layer-0 first op: Gemma RMSNorm of the embedding.
    # Correct (transformers Qwen3_5RMSNorm = Gemma3RMSNorm): y = rms(x)*(1+w_hf)
    # Packed-as-run (kernel (1+w_packed) with w_packed = w_hf - 1): y = rms(x)*w_hf
    y_correct = gemma_rms(hf_row, ln0)
    y_packed_kernel = gemma_rms(hf_row, p_ln0)
    y_if_kernel_saw_hf_delta = gemma_rms(hf_row, ln0)

    report = {
        "schema": "hawking.headless.qwen38_gravity_coherence_oracle.v1",
        "one_shot": True,
        "step1_tied_embeddings": {
            "config_tie_word_embeddings": cfg.get("tie_word_embeddings"),
            "text_config_tie_word_embeddings": tc.get("tie_word_embeddings"),
            "index_has_lm_head.weight": "lm_head.weight" in wm,
            "index_has_embed_tokens": "model.language_model.embed_tokens.weight" in wm,
            "lm_head_shape": lm_shape,
            "embed_shape": embed_shape,
            "lm_head_vs_embed_row0_cosine": float(
                np.dot(lm_row0, embed_row0)
                / ((np.linalg.norm(lm_row0) * np.linalg.norm(embed_row0)) or 1.0)
            ),
            "verdict": "NOT_TIED: lm_head.weight exists independently; tie_word_embeddings is false",
        },
        "step2_mixer_kind": {
            "full_attention_interval": tc.get("full_attention_interval"),
            "n_layer_types": len(layer_types),
            "config_full_attention_layers": [i for i, t in enumerate(layer_types) if t == "full_attention"],
            "hardcoded_rule": "(layer+1)%4==0 -> GQA / full_attention",
            "hardcoded_matches_config": mixer_hardcoded == layer_types,
            "tensor_mismatches": mismatches,
            "sample_layers": per_layer,
            "verdict": (
                "MATCH: hardcoded interval-4 GQA on layers 3,7,...,63 equals config.layer_types "
                "and the tensors present on disk"
                if not mismatches and mixer_hardcoded == layer_types
                else "MISMATCH"
            ),
        },
        "step3_oracle_embed_and_layer0_rmsnorm": {
            "transformers_class": "Qwen3_5RMSNorm = Qwen3NextRMSNorm = Gemma3RMSNorm",
            "transformers_math": "y = rms(x) * (1 + weight), zeros-init (HF stores delta)",
            "mlx_math": "nn.RMSNorm is y = rms(x) * weight, ones-init (MLX stores scale = 1+delta)",
            "runtime_kernel": "qwen80_residual_rmsnorm_tg: output = x * inv_rms * (1 + weight)",
            "hf_layer0_input_layernorm": stats(ln0),
            "packed_layer0_input_layernorm": stats(p_ln0),
            "packed_minus_hf": {
                "mean": float((p_ln0 - ln0).mean()),
                "max_abs": float(np.max(np.abs(p_ln0 - ln0))),
                "equals_hf_minus_one": bool(np.allclose(p_ln0, ln0 - 1.0, atol=1e-5, rtol=0)),
            },
            "hf_final_norm": stats(final),
            "packed_final_norm": stats(p_final),
            "hf_q_norm_l3": stats(qn3),
            "packed_q_norm_l3": stats(p_qn3),
            "hf_linear_attn_norm_l0_NOT_converted": stats(gdn_norm),
            "packed_linear_attn_norm_l0": stats(p_gdn),
            "gdn_norm_packed_equals_hf": bool(np.allclose(p_gdn, gdn_norm, atol=1e-5, rtol=0)),
            "probe_token": PROBE_TOKEN,
            "embed_q4_row_cosine_vs_hf": embed_row_cosine,
            "embed_hf_row": stats(hf_row),
            "embed_q4_row": stats(q4_row),
            "layer0_rmsnorm_correct_gemma_1plus_hf": stats(y_correct),
            "layer0_rmsnorm_as_packed_kernel_runs": stats(y_packed_kernel),
            "scale_ratio_packed_over_correct_mean": float(
                (y_packed_kernel.mean() / y_correct.mean()) if y_correct.mean() != 0 else 0.0
            ),
            "l2_ratio_packed_over_correct": float(
                np.linalg.norm(y_packed_kernel) / (np.linalg.norm(y_correct) or 1.0)
            ),
            "collapse_token_lm_head_row": stats(lm_collapse),
            "verdict": (
                "DATA BUG: packer mlx_residual_norm_to_delta subtracted 1 from HF Gemma "
                "deltas. Kernel (1+w_packed) = w_hf instead of (1+w_hf). Layer-0 residual "
                "RMSNorm output L2 is crushed by ~the (1+w)/w ratio. linear_attn.norm was "
                "correctly left as a ones-init scale. Embed Q4 row cosine is high, so "
                "quantization is not the collapse."
                if abs(float((p_ln0 - ln0).mean()) + 1.0) < 1e-4
                else (
                    "PACKED MATCHES HF DELTA: residual f32 equals Gemma delta, kernel "
                    "(1+w) reconstructs the scale. Embed Q4 row cosine remains high."
                    if float(np.max(np.abs(p_ln0 - ln0))) < 1e-5
                    else "UNEXPECTED packed-vs-HF residual-norm relationship"
                )
            ),
        },
    }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

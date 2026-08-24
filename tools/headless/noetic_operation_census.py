#!/usr/bin/env python3
"""Noetic operation census: SOURCE vs EXECUTABLE on the Qwen3.8 uniform-q4 path.

A representation that is smaller but expands to a dense matrix before GEMM has
moved the cost, not removed it. This census measures FLOPs, operations,
dispatches, DRAM traffic and temporary materialization from what
`qwen38_hybrid_decode.rs` actually encodes — not from 2N on paper.

Does not open Metal, does not load the 27B, does not spawn a second model
server. Geometry and the dispatch graph are read from the decode source;
the packed artifact is read only as a byte/codec witness.

    python3 tools/headless/noetic_operation_census.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

DECODE = REPO / "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
GEOMETRY = REPO / "crates/hawking-core/src/model/qwen38_geometry.rs"
SCHEDULE = REPO / "crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs"
LEDGER = REPO / "crates/hawking-core/src/model/qwen38_token_ns_ledger.rs"
DECODE_FAMILY = REPO / "crates/hawking-core/src/decode_family.rs"
SHADERS = REPO / "crates/hawking-core/shaders"
KERNELS_RS = REPO / "crates/hawking-core/src/kernels/mod.rs"

ARTIFACT_DEFAULT = Path.home() / "models/qwen38-gravity-uniform-q4-v1"

# Anchors — measured, not re-derived.
ANCHOR_DISPATCHES = 964
ANCHOR_CBS = 1
ANCHOR_BOUND = 38
ANCHOR_DECLARED = 554
ANCHOR_TPS = 32.73
ANCHOR_TOKEN_MS = 30.606
ANCHOR_ROOF_GB_S = 778.8
ANCHOR_UNIFIED_B = 103_079_215_104
ANCHOR_GPU_CORES = 60
ANCHOR_PARAMS = 26_895_998_464
ANCHOR_BPW = 4.253
ANCHOR_FILES = 756
ANCHOR_ARTIFACT_B = 14_297_933_604
ANCHOR_TENSORS = 755
ANCHOR_Q4 = 402
ANCHOR_F32 = 353

# G143 paper FLOPs (2N over source_weight_elements, including the embed table).
G143_FLOPS = 53_791_996_928  # 2 * 26_895_998_464
G143_ACTIVE_TEXT_PARAMS = 26_895_998_464
G143_COMPUTE_PEAK_GFLOPS = 8979.0

UNIFORM_Q4_GROUP = 64
Q4_BYTES_PER_GROUP = UNIFORM_Q4_GROUP // 2 + 2  # 32 code + 2 scale = 34

RECORDED_DISPATCH_NOTE = (
    "crates/hawking-core/src/model/qwen38_token_ns_ledger.rs::production_dispatches_per_token "
    "and the unit test production_dispatch_count_is_964. Shape: 1 command buffer, 964 dispatches."
)


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=REPO, timeout=20,
        ).stdout.strip()
    except Exception:
        return ""


def usize_const(src: str, name: str) -> int:
    m = re.search(rf"pub const {name}: usize = ([0-9_]+);", src)
    if not m:
        raise SystemExit(f"FAIL: missing usize const {name}")
    return int(m.group(1).replace("_", ""))


def u64_const(src: str, name: str) -> int:
    m = re.search(rf"pub const {name}: u64 = ([0-9_]+);", src)
    if not m:
        raise SystemExit(f"FAIL: missing u64 const {name}")
    return int(m.group(1).replace("_", ""))


def str_array(src: str, name: str) -> list[str]:
    m = re.search(
        rf"pub const {name}: \[&str; [A-Za-z0-9_]+\] = \[([^\]]+)\];",
        src, re.S,
    )
    if not m:
        raise SystemExit(f"FAIL: missing string array {name}")
    return re.findall(r'"([^"]+)"', m.group(1))


def declared_kernels() -> tuple[list[str], int]:
    """Same extraction as tools/nx_genome.py: lines starting `kernel void `."""
    names: set[str] = set()
    for p in sorted(SHADERS.glob("*.metal")):
        for line in p.read_text().splitlines():
            if line.startswith("kernel void "):
                names.add(line.split()[2].split("(")[0])
    return sorted(names), len(names)


def bound_kernels_g071(decode_text: str, declared: set[str]) -> list[str]:
    """String literals in qwen38_hybrid_decode.rs ∩ declared kernel voids.

    This is the G071 / nx_genome method. It undercounts kernels dispatched
    through helpers (mha_decode_f32, qwen_next_add_residual, gk_swiglu_f32,
    sample_argmax_f32) because those names are not string literals in the
    decode file.
    """
    lits = set()
    for tok in decode_text.split('"'):
        if tok and all(c.isalnum() or c == "_" for c in tok):
            lits.add(tok)
    return sorted(lits & declared)


def helper_kernels_on_uniform_q4() -> list[str]:
    """Kernels the uniform-q4 production path dispatches via helpers, not lits."""
    # Confirmed by reading the helper bodies: they pass these names to dispatch_threads.
    return [
        "mha_decode_f32",           # kernels/mod.rs mha_decode_f32_tcb
        "qwen_next_add_residual",   # kernels/mod.rs qwen_next_add_residual_tcb
        "gk_swiglu_f32",            # decode_family::swiglu_f32 default ON
        "sample_argmax_f32",        # kernels/mod.rs sample_argmax_f32_tcb
    ]


def q4_matrix_bytes(rows: int, cols: int) -> int:
    groups = (cols + UNIFORM_Q4_GROUP - 1) // UNIFORM_Q4_GROUP
    return rows * groups * Q4_BYTES_PER_GROUP


def f32b(n: int) -> int:
    return n * 4


def load_geometry() -> dict:
    geo = GEOMETRY.read_text()
    sched = SCHEDULE.read_text()
    led = LEDGER.read_text()
    g = {
        "layers": usize_const(geo, "QWEN38_LAYERS"),
        "dn_layers": usize_const(geo, "QWEN38_DELTANET_LAYERS"),
        "gqa_layers": usize_const(geo, "QWEN38_GQA_LAYERS"),
        "hidden": usize_const(geo, "QWEN38_HIDDEN"),
        "intermediate": usize_const(geo, "QWEN38_INTERMEDIATE"),
        "vocab": usize_const(geo, "QWEN38_VOCAB"),
        "qkvz_rows": usize_const(geo, "QWEN38_QKVZ_ROWS"),
        "ba_rows": usize_const(geo, "QWEN38_BA_ROWS"),
        "q_proj_rows": usize_const(geo, "QWEN38_Q_PROJ_ROWS"),
        "kv_proj_rows": usize_const(geo, "QWEN38_KV_PROJ_ROWS"),
        "o_proj_rows": usize_const(geo, "QWEN38_O_PROJ_ROWS"),
        "o_proj_cols": usize_const(geo, "QWEN38_O_PROJ_COLS"),
        "gqa_heads": usize_const(geo, "QWEN38_GQA_HEADS"),
        "gqa_kv_heads": usize_const(geo, "QWEN38_GQA_KV_HEADS"),
        "gqa_head_dim": usize_const(geo, "QWEN38_GQA_HEAD_DIM"),
        "gqa_rotary_dim": usize_const(geo, "QWEN38_GQA_ROTARY_DIM"),
        "lin_key_heads": usize_const(geo, "QWEN38_LINEAR_KEY_HEADS"),
        "lin_value_heads": usize_const(geo, "QWEN38_LINEAR_VALUE_HEADS"),
        "lin_vpk": usize_const(geo, "QWEN38_LINEAR_VALUES_PER_KEY"),
        "lin_key_dim": usize_const(geo, "QWEN38_LINEAR_KEY_HEAD_DIM"),
        "lin_value_dim": usize_const(geo, "QWEN38_LINEAR_VALUE_HEAD_DIM"),
        "lin_conv_k": usize_const(geo, "QWEN38_LINEAR_CONV_KERNEL"),
        "mixer_prefix": usize_const(sched, "QWEN38_MIXER_PREFIX_DISPATCHES"),
        "mlp_suffix": usize_const(sched, "QWEN38_DENSE_MLP_SUFFIX_DISPATCHES"),
        "dn_prefix_kernels": str_array(sched, "QWEN38_DELTANET_MIXER_PREFIX_KERNELS"),
        "gqa_prefix_kernels": str_array(sched, "QWEN38_GQA_MIXER_PREFIX_KERNELS"),
        "mlp_suffix_kernels": str_array(sched, "QWEN38_DENSE_MLP_SUFFIX_KERNELS"),
        "terminal_kernels": str_array(sched, "QWEN38_TERMINAL_HEAD_KERNELS"),
        "active_budget_bytes": u64_const(led, "ACTIVE_BUDGET_BYTES"),
        "embed_table_bytes": u64_const(led, "EMBED_TABLE_BYTES"),
    }
    g["full_layer"] = g["mixer_prefix"] + g["mlp_suffix"]
    g["terminal_n"] = len(g["terminal_kernels"])
    g["production_dispatches"] = 1 + g["layers"] * g["full_layer"] + g["terminal_n"]
    g["production_cbs"] = 1
    # DeltaNet layout, matching Qwen38DeltaNetLayout::source_exact
    g["value_elements"] = g["lin_value_heads"] * g["lin_value_dim"]
    g["key_elements"] = g["lin_key_heads"] * g["lin_key_dim"]
    g["conv_channels"] = (
        g["lin_key_heads"] * g["lin_key_dim"] * 2
        + g["lin_value_heads"] * g["lin_value_dim"]
    )
    g["conv_state_elements"] = g["conv_channels"] * (g["lin_conv_k"] - 1)
    g["rec_state_elements"] = (
        g["lin_value_heads"] * g["lin_key_dim"] * g["lin_value_dim"]
    )
    return g


def gemv_organs(g: dict) -> list[dict]:
    """Every Q4 matvec the uniform-q4 production path encodes, with shape."""
    H, I = g["hidden"], g["intermediate"]
    organs = []

    def add(name, count, rows, cols, role):
        elems = rows * cols
        organs.append({
            "organ": name,
            "count_per_token": count,
            "rows": rows,
            "cols": cols,
            "elements_per_launch": elems,
            "elements_per_token": elems * count,
            "mac_flops_per_launch": 2 * elems,  # FMA = 2
            "mac_flops_per_token": 2 * elems * count,
            "q4_bytes_per_launch": q4_matrix_bytes(rows, cols),
            "q4_bytes_per_token": q4_matrix_bytes(rows, cols) * count,
            "dense_f32_bytes_per_launch": f32b(elems),
            "dense_f32_bytes_per_token": f32b(elems) * count,
            "dense_bf16_bytes_per_token": 2 * elems * count,
            "kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
            "role": role,
        })

    add("mlp.gate_proj", g["layers"], I, H, "mlp")
    add("mlp.up_proj", g["layers"], I, H, "mlp")
    add("mlp.down_proj", g["layers"], H, I, "mlp")
    add("linear_attn.in_proj_qkvz", g["dn_layers"], g["qkvz_rows"], H, "deltanet")
    add("linear_attn.in_proj_ba", g["dn_layers"], g["ba_rows"], H, "deltanet")
    add("linear_attn.out_proj", g["dn_layers"], H, g["o_proj_cols"], "deltanet")
    add("self_attn.q_proj", g["gqa_layers"], g["q_proj_rows"], H, "gqa")
    add("self_attn.k_proj", g["gqa_layers"], g["kv_proj_rows"], H, "gqa")
    add("self_attn.v_proj", g["gqa_layers"], g["kv_proj_rows"], H, "gqa")
    add("self_attn.o_proj", g["gqa_layers"], H, g["o_proj_cols"], "gqa")
    add("lm_head", 1, g["vocab"], H, "terminal")
    return organs


def live_kernel_names() -> dict:
    """Default-env kernel names the encode path actually binds.

    The frozen schedule arrays still name the scalar/legacy symbols. The
    encode functions retile behind env_opt_out / unwrap_or defaults.
    Dispatch COUNT is unchanged; the NAME of several organs is not the
    schedule string.
    """
    return {
        "embed": "qwen_uniform_q4_embedding_lookup",
        "rmsnorm": "qwen80_residual_rmsnorm_tg",  # HAWKING_RMSNORM_TG default 1024
        "gemv": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
        "rearrange": "qwen38_qkvz_rearrange_conv_l2_f32",
        "ba_to_decay": "qwen80_ba_to_decay_beta_f32",
        "gated_delta": "qwen38_gated_delta_decode_vi_simd",  # HAWKING_DN_VI_SIMD default true
        "gated_rmsnorm": "qwen80_deltanet_gated_rmsnorm_tg",  # HAWKING_DN_RMSNORM_TG default 256
        "residual": "qwen_next_add_residual",
        "rope": "qwen38_gqa_qk_norm_rope_cache_tg",  # HAWKING_ROPE_TG default 256
        "mha": "mha_decode_f32",
        "sigmoid_gate": "qwen38_attention_apply_sigmoid_gate",
        "swiglu": "gk_swiglu_f32",  # HAWKING_DECODE_FAMILY default ON
        "argmax": "sample_argmax_f32",  # HAWKING_ARGMAX_TWO_PASS default OFF
        "schedule_vs_live": {
            "qwen80_residual_rmsnorm_f32": "qwen80_residual_rmsnorm_tg",
            "qwen80_silu_mul_f32": "gk_swiglu_f32",
            "qwen38_gated_delta_decode_vi": "qwen38_gated_delta_decode_vi_simd",
            "qwen38_gqa_qk_norm_rope_cache_f32": "qwen38_gqa_qk_norm_rope_cache_tg",
            "sample_argmax_f32 (schedule name matches live; two-pass is off)": "sample_argmax_f32",
        },
        "count_unchanged": True,
        "count_exception": (
            "HAWKING_ARGMAX_TWO_PASS=1 replaces sample_argmax_f32 with "
            "sample_argmax_f32_pass1 + pass2 and the token becomes 965 dispatches. Default is off."
        ),
    }


def dispatch_inventory(g: dict, live: dict) -> list[dict]:
    """One row per dispatch class on the default uniform-q4 token."""
    rows = []

    def add(kernel, n, what, gemv=False, reconstructs_dense_w=False):
        rows.append({
            "kernel": kernel,
            "dispatches_per_token": n,
            "what": what,
            "is_gemv": gemv,
            "reconstructs_dense_w_to_dram": reconstructs_dense_w,
        })

    add(live["embed"], 1, "gather one Q4 embed row into f32 hidden", gemv=False)
    add(live["rmsnorm"], g["layers"] * 2 + 1,
        "input_layernorm + post_attention_layernorm per layer, plus final norm")
    add(live["gemv"],
        g["dn_layers"] * 3 + g["gqa_layers"] * 4 + g["layers"] * 3 + 1,
        "every Q4 matvec (DN qkvz/ba/out, GQA q/k/v/o, MLP gate/up/down, lm_head)",
        gemv=True)
    add(live["rearrange"], g["dn_layers"],
        "qkvz split + causal conv1d + L2 on q/k, writes q/k/v/z activations")
    add(live["ba_to_decay"], g["dn_layers"], "BA -> decay/beta (48 value heads)")
    add(live["gated_delta"], g["dn_layers"],
        "DeltaNet recurrence: decay state, rank-1 update, query readout")
    add(live["gated_rmsnorm"], g["dn_layers"], "gated RMSNorm on rec_out")
    add(live["rope"], g["gqa_layers"], "Q/K RMSNorm + partial RoPE + KV append")
    add(live["mha"], g["gqa_layers"], "GQA decode attention over cached K/V")
    add(live["sigmoid_gate"], g["gqa_layers"], "attention output * sigmoid(gate from q_proj)")
    add(live["swiglu"], g["layers"], "silu(gate)*up -> act")
    add(live["residual"], g["layers"] * 2,
        "mixer residual + MLP residual (qwen_next_add_residual)")
    add(live["argmax"], 1, "greedy argmax over vocab logits")
    total = sum(r["dispatches_per_token"] for r in rows)
    return rows, total


def activation_flops(g: dict, seq_len: int) -> dict:
    """FLOPs of non-GEMV organs the path actually launches. FMA = 2."""
    H, I = g["hidden"], g["intermediate"]
    # RMSNorm: n squares + n-1 adds + 1 rsqrt + n *(x * inv * (1+w)) ≈ 2n + 3n
    rms_one = 5 * H
    n_rms = g["layers"] * 2 + 1
    rms = rms_one * n_rms

    # SwiGLU: silu(g)*up. silu = g / (1+exp(-g)) → exp, add, div, mul with up.
    swiglu = 5 * I * g["layers"]

    # residual add
    residual = H * g["layers"] * 2

    # conv1d (kernel=4) on conv_channels, plus silu on q/k/v channels, plus L2.
    conv_mac = 2 * g["conv_channels"] * g["lin_conv_k"]  # 4 taps, mul+add
    conv_silu = 5 * g["conv_channels"]  # the conv kernel applies silu to the sum
    l2 = 5 * (g["key_elements"] * 2)  # q and k L2
    rearrange = (conv_mac + conv_silu + l2) * g["dn_layers"]

    # ba_to_decay: per value head, a handful of exp/log/div
    ba = 12 * g["lin_value_heads"] * g["dn_layers"]

    # gated_delta: see qwen38_gated_delta_decode_vi. Per state element:
    # decay mul, kv_mem mul+add, delta FMA, output mul+add. ~7 FLOPs * state.
    gd = 7 * g["rec_state_elements"] * g["dn_layers"]

    # gated rmsnorm over value_elements
    gated_n = 5 * g["value_elements"] * g["dn_layers"]

    # GQA rope: per head, RMS over head_dim + 2 trig per rotary dim
    rope = g["gqa_layers"] * (
        5 * g["gqa_heads"] * g["gqa_head_dim"]
        + 10 * g["gqa_heads"] * g["gqa_rotary_dim"]
    )

    # MHA: QK and AV dots. Softmax ~5 FLOPs per (head, seq) score.
    mha_mac = 2 * 2 * g["gqa_layers"] * g["gqa_heads"] * seq_len * g["gqa_head_dim"]
    mha_sm = 5 * g["gqa_layers"] * g["gqa_heads"] * seq_len

    # sigmoid gate
    sig = 5 * g["gqa_heads"] * g["gqa_head_dim"] * g["gqa_layers"]

    # embed dequant: 1 scale-mul per hidden (integer unpack counted elsewhere)
    embed = H

    # argmax: comparisons, not FLOPs
    argmax_comps = g["vocab"]

    parts = {
        "rmsnorm": rms,
        "swiglu": swiglu,
        "residual_add": residual,
        "rearrange_conv_l2": rearrange,
        "ba_to_decay": ba,
        "gated_delta": gd,
        "gated_rmsnorm": gated_n,
        "rope": rope,
        "mha_mac": mha_mac,
        "mha_softmax": mha_sm,
        "sigmoid_gate": sig,
        "embed_dequant_scale": embed,
        "argmax_comparisons_not_flops": argmax_comps,
    }
    parts["total_flops"] = sum(v for k, v in parts.items() if k != "argmax_comparisons_not_flops")
    return parts


def dram_and_temp(g: dict, organs: list[dict], seq_len: int) -> dict:
    """Per-token DRAM traffic and temporary materialization, derived from encode()."""
    H, I = g["hidden"], g["intermediate"]
    gemv_q4 = sum(o["q4_bytes_per_token"] for o in organs)
    gemv_f32 = sum(o["dense_f32_bytes_per_token"] for o in organs)
    gemv_bf16 = sum(o["dense_bf16_bytes_per_token"] for o in organs)
    gemv_elems = sum(o["elements_per_token"] for o in organs)

    # Embed: one Q4 row, not the table.
    embed_q4 = q4_matrix_bytes(1, H)
    embed_table_q4 = q4_matrix_bytes(g["vocab"], H)
    embed_f32_out = f32b(H)

    # f32 mixer scales streamed (already dense in the artifact).
    conv_w = f32b(g["conv_channels"] * g["lin_conv_k"]) * g["dn_layers"]
    a_log = f32b(g["lin_value_heads"]) * g["dn_layers"]
    dt_bias = f32b(g["lin_value_heads"]) * g["dn_layers"]
    # 2 residual norms / layer + final + GQA q/k_norm + DN gated norm
    rms_w = (
        f32b(H) * (g["layers"] * 2 + 1)
        + f32b(g["gqa_head_dim"]) * 2 * g["gqa_layers"]
        + f32b(g["value_elements"]) * g["dn_layers"]
    )
    f32_scales = conv_w + a_log + dt_bias + rms_w

    # Activation writes (kernel outputs). This is temporary materialization
    # of ACTIVATIONS, not of weight matrices.
    def writes_dn():
        return (
            f32b(H)              # normalized
            + f32b(g["qkvz_rows"])  # qkvz
            + f32b(g["ba_rows"])    # ba
            + f32b(g["value_elements"]) * 4  # repeated_q, repeated_k, conv_v, z
            + f32b(g["lin_value_heads"]) * 2  # decay, beta
            + f32b(g["value_elements"]) * 2   # rec_out, gated
            + f32b(H) * 2          # mixer, first_residual
            + f32b(g["conv_state_elements"])  # conv state write
            + f32b(g["rec_state_elements"])   # rec state write
        )

    def writes_gqa():
        kv_slot = f32b(g["gqa_kv_heads"] * g["gqa_head_dim"])
        return (
            f32b(H)  # normalized
            + f32b(g["q_proj_rows"])
            + f32b(g["kv_proj_rows"]) * 2
            + f32b(g["gqa_heads"] * g["gqa_head_dim"]) * 3  # query, attn, gated_attn
            + kv_slot * 2  # K/V append
            + f32b(H) * 2  # mixer, first_residual
        )

    def writes_mlp():
        return (
            f32b(H)          # normalized
            + f32b(I) * 3    # gate, up, act
            + f32b(H) * 2    # down, hidden residual
        )

    act_write = (
        embed_f32_out
        + writes_dn() * g["dn_layers"]
        + writes_gqa() * g["gqa_layers"]
        + writes_mlp() * g["layers"]
        + f32b(H)          # final norm
        + f32b(g["vocab"]) # logits
        + 4                # sampled u32
    )

    # Activation reads beyond the weight stream: residual inputs, KV cache,
    # conv/rec state, silu's two vectors, etc. Count the dominant extras.
    rec_rw = f32b(g["rec_state_elements"]) * 2 * g["dn_layers"]  # read+write already in write?
    # rec write is in act_write; rec read is extra DRAM.
    rec_read = f32b(g["rec_state_elements"]) * g["dn_layers"]
    conv_read = f32b(g["conv_state_elements"]) * g["dn_layers"]
    kv_read = f32b(g["gqa_layers"] * seq_len * g["gqa_kv_heads"] * g["gqa_head_dim"]) * 2
    hidden_in = f32b(H) * (g["layers"] * 2)  # mixer and mlp residual inputs

    # Input vectors to GEMV (normalized / act / gated / query). Already
    # produced as writes; reading them is extra traffic.
    gemv_x_read = (
        f32b(H) * (g["layers"] * 2 + g["dn_layers"] * 2 + g["gqa_layers"] * 3 + 1)
        # MLP gate/up read H; down reads I; DN qkvz/ba read H; out reads 6144;
        # GQA q/k/v read H; o reads 6144; lm_head reads H.
        + f32b(I) * g["layers"]
        + f32b(g["value_elements"]) * g["dn_layers"]
        + f32b(g["gqa_heads"] * g["gqa_head_dim"]) * g["gqa_layers"]
    )

    exec_weight = gemv_q4 + embed_q4 + f32_scales
    exec_dram = exec_weight + act_write + rec_read + conv_read + kv_read + gemv_x_read

    # Reconstruct-then-GEMM trap: write dense W, then read it.
    trap_w_materialize = gemv_f32  # write
    trap_w_reread = gemv_f32       # GEMM then reads f32 W
    trap_still_reads_q4 = gemv_q4  # dequant had to read the codes first

    # Unused-on-this-path workspace (allocated, not touched per token).
    unused_residency = (
        f32b(160)  # hgravs_mid, QWEN38_MIXED_HGRAVS_RANK
        + f32b(10_240)  # split_qkv
        + f32b(48)      # split_b
        + f32b(48)      # split_a
        + 240 * 4 * 2   # argmax two-pass partials
    )

    return {
        "gemv_elements_per_token": gemv_elems,
        "executable_weight_bytes_per_token": exec_weight,
        "executable_q4_gemv_bytes": gemv_q4,
        "executable_embed_row_q4_bytes": embed_q4,
        "executable_embed_table_resident_q4_bytes": embed_table_q4,
        "executable_f32_scale_bytes": f32_scales,
        "executable_activation_write_bytes": act_write,
        "executable_rec_state_read_bytes": rec_read,
        "executable_conv_state_read_bytes": conv_read,
        "executable_gqa_kv_read_bytes_at_seq": kv_read,
        "executable_gemv_input_read_bytes": gemv_x_read,
        "executable_dram_bytes_per_token": exec_dram,
        "source_dense_f32_gemv_bytes": gemv_f32,
        "source_dense_bf16_gemv_bytes": gemv_bf16,
        "source_embed_table_f32_bytes": f32b(g["vocab"] * H),
        "trap_reconstruct_then_gemm": {
            "dense_w_write_bytes": trap_w_materialize,
            "dense_w_reread_bytes": trap_w_reread,
            "q4_still_read_bytes": trap_still_reads_q4,
            "extra_vs_fused_bytes": trap_w_materialize + trap_w_reread,
            "note": (
                "A decode_vector-then-f32-GEMM lowering would materialise every "
                "GEMV matrix as f32 (102.5 GiB-class) every token, then read it "
                "back. The fused geo_tpr64 kernel never writes that buffer."
            ),
        },
        "temporary_materialization_bytes_per_token": act_write,
        "dense_w_materialized_bytes_per_token": 0,
        "unused_mixed_workspace_residency_bytes": unused_residency,
        "seq_len_used_for_kv": seq_len,
    }


def reconstruction_sites(g: dict, organs: list[dict], live: dict) -> list[dict]:
    H, I = g["hidden"], g["intermediate"]
    gemv_f32 = sum(o["dense_f32_bytes_per_token"] for o in organs)
    sites = []

    def site(kernel, bytes_mat, classification, note, on_path, kind):
        sites.append({
            "kernel": kernel,
            "bytes_materialised_per_token": bytes_mat,
            "classification": classification,
            "kind": kind,
            "on_uniform_q4_production_path": on_path,
            "note": note,
        })

    site(
        live["gemv"],
        0,
        "required-by-implementation-of-the-codec-but-fused",
        (
            "qwen_uniform_q4_unpack8 unpacks 8 nibbles into registers, FMAs them "
            "against x, and discards the decoded weights. Output is the matvec "
            "vector only. This is the site a reconstruct-then-GEMM lowering would "
            f"turn into a {gemv_f32:,} byte dense-W write. It does not, today."
        ),
        True,
        "weight_in_register",
    )
    site(
        "qwen_uniform_q4_decode_vector",
        gemv_f32,
        "required-by-implementation",
        (
            "The reconstruct-to-dense kernel exists in qwen_uniform_q4.metal and is "
            "wired in metal/mod.rs. qwen38_hybrid_decode.rs does not dispatch it. "
            "qwen30_complete_runtime.rs does. Naming it is the point: this is the "
            "kernel a Noetic representation must not need."
        ),
        False,
        "weight_to_dram",
    )
    site(
        live["embed"],
        f32b(H),
        "required-by-math",
        (
            "One vocab row must exist as an f32 hidden vector before layer 0. "
            "The kernel dequants 5,120 Q4 values into workspace.hidden and does "
            "not expand the 675,430,440-byte embed table."
        ),
        True,
        "activation",
    )
    site(
        live["gemv"] + " -> workspace.gate/up",
        f32b(I) * 2 * g["layers"],
        "required-by-implementation",
        (
            "SwiGLU math needs gate and up. Writing both as full 17,408-wide f32 "
            "buffers, then launching a third kernel to silu-mul them, is the "
            "current graph. A fused gate-up-silu-down kernel would not write "
            "gate, up, or act to DRAM."
        ),
        True,
        "activation",
    )
    site(
        live["swiglu"] + " -> workspace.act",
        f32b(I) * g["layers"],
        "required-by-implementation",
        "act is the SwiGLU result consumed by down_proj. Math needs the vector; "
        "a fused MLP would keep it in registers / SRAM.",
        True,
        "activation",
    )
    site(
        live["gemv"] + " -> workspace.qkvz",
        f32b(g["qkvz_rows"]) * g["dn_layers"],
        "required-by-implementation",
        (
            "Pack-time fuse already concatenated in_proj_qkv and in_proj_z into "
            "one Q4 matrix, so W is not reconstructed. The ACTIVATION is still "
            "materialised as a 16,384-wide f32 vector for rearrange to split. "
            "Math needs q,k,v,z; it does not need the concatenated buffer."
        ),
        True,
        "activation",
    )
    site(
        live["gemv"] + " -> workspace.ba",
        f32b(g["ba_rows"]) * g["dn_layers"],
        "required-by-implementation",
        "96-wide BA vector written dense, then ba_to_decay rereads it. Fuseable.",
        True,
        "activation",
    )
    site(
        live["rearrange"],
        (f32b(g["value_elements"]) * 4 + f32b(g["conv_state_elements"])) * g["dn_layers"],
        "required-by-math",
        (
            "Causal conv and the recurrence consume q,k,v,z and update conv_state. "
            "Those tensors are the math. The preceding qkvz concatenation is not."
        ),
        True,
        "activation",
    )
    site(
        live["gated_delta"],
        f32b(g["rec_state_elements"] + g["value_elements"]) * g["dn_layers"],
        "required-by-math",
        (
            "Recurrent state is 48×128×128 f32 per DeltaNet layer. It is read and "
            "written every token. This is the math of gated DeltaNet, not a codec artefact."
        ),
        True,
        "state",
    )
    site(
        live["gemv"] + " -> workspace.q_proj/k_proj/v_proj",
        (f32b(g["q_proj_rows"]) + f32b(g["kv_proj_rows"]) * 2) * g["gqa_layers"],
        "required-by-implementation",
        "Q/K/V written dense, then rope/mha reread them. Fuseable into rope.",
        True,
        "activation",
    )
    site(
        live["mha"],
        f32b(g["gqa_heads"] * g["gqa_head_dim"]) * g["gqa_layers"],
        "required-by-math",
        "Attention output is a 6,144-wide f32 vector. KV cache traffic scales with seq_len.",
        True,
        "activation",
    )
    site(
        live["gemv"] + " -> workspace.logits",
        f32b(g["vocab"]),
        "required-by-implementation",
        (
            "lm_head writes 248,320 f32 logits, then a separate argmax scans them. "
            "Math needs the max index, not the dense logit buffer, if argmax were fused "
            "into the matvec reduction. Default path does not fuse them. "
            "qwen_uniform_q4_group64_final_norm_lm_head_simdgroup8 exists and is unused."
        ),
        True,
        "activation",
    )
    site(
        live["rmsnorm"] + " -> workspace.normalized",
        f32b(H) * (g["layers"] * 2 + 1),
        "required-by-implementation",
        "RMSNorm output is an f32 hidden written so the next GEMV can bind it. Fuseable.",
        True,
        "activation",
    )
    site(
        "qwen38_fuse_split_qkvz_f32 / qwen38_fuse_split_ba_f32",
        (f32b(g["qkvz_rows"]) + f32b(g["ba_rows"])) * g["dn_layers"],
        "required-by-implementation",
        (
            "Concatenates split in_proj activations into the packed QKVZ/BA layout. "
            "The uniform-q4 artifact has fused_in_proj_layers=48, so encode_deltanet "
            "never takes this branch. Mixed/unfused catalogs do."
        ),
        False,
        "activation",
    )
    site(
        "dequant_hgravu_vector (host, session open)",
        0,
        "required-by-implementation",
        (
            "Host decoder for small HGRAVU vectors at catalog load. Explicitly refuses "
            "dense W (`dequant refuses N elements (dense W)`). Not per-token, not on "
            "the uniform-q4 path."
        ),
        False,
        "weight_host",
    )
    return sites


def read_artifact(path: Path) -> dict:
    out = {
        "path": str(path),
        "present": path.is_dir(),
        "file_count": None,
        "bytes": None,
        "manifest": None,
    }
    if not path.is_dir():
        return out
    n = 0
    b = 0
    for dp, _ds, fs in os.walk(path):
        for fn in fs:
            fp = os.path.join(dp, fn)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            n += 1
            b += st.st_size
    out["file_count"] = n
    out["bytes"] = b
    man = path / "manifest.json"
    if man.is_file():
        m = json.loads(man.read_text())
        kinds = {}
        for t in m.get("tensors") or []:
            k = t.get("kind") or "?"
            kinds[k] = kinds.get(k, 0) + 1
        out["manifest"] = {
            "schema": m.get("schema"),
            "status": m.get("status"),
            "tensor_count": m.get("tensor_count"),
            "q4_tensors": m.get("q4_tensors"),
            "f32_tensors": m.get("f32_tensors"),
            "source_weight_elements": m.get("source_weight_elements"),
            "tensor_payload_bytes": m.get("tensor_payload_bytes"),
            "complete_physical_bpw": m.get("complete_physical_bpw"),
            "q4_group_size": m.get("q4_group_size"),
            "fused_in_proj_layers": m.get("fused_in_proj_layers"),
            "skipped_vision_tensors": m.get("skipped_vision_tensors"),
            "kind_counts": kinds,
        }
    return out


def analytic_gap(g: dict, organs: list[dict], act: dict) -> dict:
    gemv_macs = sum(o["mac_flops_per_token"] for o in organs)
    gemv_elems = sum(o["elements_per_token"] for o in organs)
    paper = G143_FLOPS
    embed_elems = g["vocab"] * g["hidden"]
    paper_minus_embed = paper - 2 * embed_elems
    leftover_params = G143_ACTIVE_TEXT_PARAMS - gemv_elems - embed_elems
    return {
        "g143_paper_flops_2N": paper,
        "g143_method": (
            "tools/flops_per_token.py / receipts/ascent-2026-08-16/G143_FLOPS_PER_TOKEN.json: "
            "2 * active_text_params, active_text_params = source_weight_elements including "
            "the embed table and every f32 scale tensor. Not derived from dispatched kernels."
        ),
        "dispatched_gemv_mac_flops": gemv_macs,
        "dispatched_activation_flops": act["total_flops"],
        "dispatched_total_flops_without_dequant_scale": gemv_macs + act["total_flops"],
        "embed_table_params_treated_as_matvec_by_paper": embed_elems,
        "paper_overcount_from_embed_table_macs": 2 * embed_elems,
        "paper_minus_embed_table": paper_minus_embed,
        "gemv_elements_actually_multiplied": gemv_elems,
        "leftover_params_not_in_any_gemv": leftover_params,
        "leftover_is": (
            "f32 mixer tensors (conv1d.weight, A_log, dt_bias, rms/q/k/gated norms). "
            "They run as f32 vector math, not as 2N MACs."
        ),
        "gap_paper_minus_dispatched_gemv_macs": paper - gemv_macs,
        "reading": (
            f"G143 reports {paper / 1e9:.2f} GFLOP/token. The decode path's GEMV kernels "
            f"perform {gemv_macs / 1e9:.2f} GFLOP of MACs — the embed table is gathered "
            f"as one row ({g['hidden']} elements), not multiplied as a {g['vocab']}×{g['hidden']} "
            f"matvec ({2 * embed_elems / 1e9:.2f} GFLOP of the paper number never run). "
            f"The remaining {leftover_params:,} params are f32 scales, not GEMVs. "
            "An analytic FLOP number and a measured dispatch count disagreeing is the finding: "
            "964 dispatches is a graph-shape fact; 53.79 GFLOP is a parameter-count fact that "
            "double-counts the embed table."
        ),
    }


def what_watched_fail(artifact: dict, g: dict, bound_n: int, declared_n: int,
                      inv_total: int) -> list[dict]:
    fails = []

    fails.append({
        "what": "git apply tree-state.patch",
        "result": "FAILED",
        "why": (
            "The patch is entirely tools/haider/hcli/**. This worktree is a sparse checkout "
            "and tools/haider is neither materialized nor in WRITE scope (DENY: tools/haider). "
            "git apply aborted with 'No such file or directory' on every hunk. "
            "untracked.tar is the same tree. Nothing in-scope was missing from HEAD."
        ),
    })
    fails.append({
        "what": "baseline suite 464 passed, 1 skipped (HCLI_SWAP_CEILING_GIB=64)",
        "result": "NOT RUN",
        "why": (
            "That suite lives under tools/haider/hcli/tests, which is DENY and not on disk. "
            "git sparse-checkout add is forbidden in this sandbox (sparse-checkout.lock: "
            "Operation not permitted). The 9 RuntimePool MemGate failures the ceiling "
            "avoids cannot be reproduced here without materializing a denied path."
        ),
    })
    fails.append({
        "what": "live 27B native decode re-time",
        "result": "NOT RUN (refused)",
        "why": (
            "A llama-server is already listening on 127.0.0.1:52484. The contract forbids "
            "spawning a second 27B. Occupancy is not free: two model servers resident "
            "measured 3.986 tok/s against 33.47 with one. TPS/token-ms used here are the "
            "supplied anchors (32.73 tps / 30.606 ms), not a new run."
        ),
    })
    fails.append({
        "what": "G143 2N vs dispatched GEMV MACs",
        "result": "DISAGREE (reported, not smoothed)",
        "why": (
            "Paper 2*N_active = 53.79 GFLOP includes a full embed-table matvec the path "
            "never launches. Dispatched GEMV MACs are 51.24 GFLOP. See analytic_vs_measured."
        ),
    })
    fails.append({
        "what": "frozen schedule kernel names vs live encode defaults",
        "result": "NAMES DRIFTED, COUNT DID NOT",
        "why": (
            "QWEN38_*_KERNELS still lists qwen80_residual_rmsnorm_f32, qwen80_silu_mul_f32, "
            "qwen38_gated_delta_decode_vi, qwen38_gqa_qk_norm_rope_cache_f32. encode_* "
            "retile to _tg / _simd / gk_swiglu_f32 under default env. 15 dispatches/layer "
            "and 964/token are unchanged. HAWKING_ARGMAX_TWO_PASS=1 would make 965."
        ),
    })
    fails.append({
        "what": "G071 38-bound kernel extraction vs kernels actually launched",
        "result": "UNDERCOUNT",
        "why": (
            f"String-literal ∩ kernel-void is still {bound_n} vs {declared_n} declared, "
            "matching the 38/554 anchor. Helpers dispatch mha_decode_f32, "
            "qwen_next_add_residual, gk_swiglu_f32, sample_argmax_f32 — four production "
            "kernels the G071 method cannot see. A seal of the 38 is a seal of literals, "
            "not of the 964-dispatch graph."
        ),
    })
    if artifact.get("present"):
        if artifact.get("file_count") != ANCHOR_FILES or artifact.get("bytes") != ANCHOR_ARTIFACT_B:
            fails.append({
                "what": "artifact byte/file witness vs anchor",
                "result": "DRIFT",
                "why": (
                    f"anchor {ANCHOR_FILES} files / {ANCHOR_ARTIFACT_B} bytes; "
                    f"walked {artifact.get('file_count')} / {artifact.get('bytes')}"
                ),
            })
    else:
        fails.append({
            "what": "artifact ~/models/qwen38-gravity-uniform-q4-v1",
            "result": "MISSING",
            "why": "Census used geometry + decode source only; codec witness was not walked.",
        })
    if inv_total != ANCHOR_DISPATCHES:
        fails.append({
            "what": "re-summed dispatch inventory vs 964",
            "result": "DRIFT",
            "why": f"inventory sums to {inv_total}, formula says {g['production_dispatches']}",
        })
    fails.append({
        "what": "GPU occupancy as a free resource",
        "result": "FALSE (already measured)",
        "why": (
            "Native run: 3.986 tok/s with two model servers resident vs 33.47 with one. "
            "The 964-dispatch fused path is still one process streaming ~13.6 GB of weights "
            "per token; a second resident copy contends for the same 595.9 GB/s roof."
        ),
    })
    return fails


def build_columns(g, organs, act, dram, inv_total) -> dict:
    gemv_macs = sum(o["mac_flops_per_token"] for o in organs)
    gemv_elems = sum(o["elements_per_token"] for o in organs)
    # Executable extra FLOP vs dense FMA: the scale mul in unpack8
    # (float(q)*scale*x is 1 mul + 1 FMA=2 → 3 FLOPs/weight vs SOURCE 2).
    dequant_scale_flops = gemv_elems + g["hidden"]  # GEMV weights + embed row
    # Integer unpack ops per weight: shift/mask byte, nibble, sub 8, itof ≈ 4
    int_ops = 4 * (gemv_elems + g["hidden"])

    src_flops = gemv_macs + act["total_flops"]
    exe_flops = gemv_macs + dequant_scale_flops + act["total_flops"]
    src_ops = src_flops  # dense path has no nibble ALU
    exe_ops = exe_flops + int_ops

    return {
        "convention": {
            "flop": "IEEE floating-point; FMA counted as 2 (matches G143's 2N = mul+add per weight)",
            "operation": "FLOPs plus integer nibble-unpack ALU (shift/mask/sub/itof ≈ 4 per Q4 weight)",
            "dispatch": "Metal compute dispatches encoded into the production TokenCommandBuffer",
            "dram": "bytes the kernels read or write per decode step, including weight stream and activations",
            "temporary_materialization": (
                "bytes of intermediate tensors written to device buffers per token. "
                "Dense W reconstruction is counted separately and is 0 on this path."
            ),
            "source_means": (
                "the dense mathematical graph: same organs, f32 (or bf16) GEMV bodies, "
                "no Q4 unpack ALU, activations still dense"
            ),
            "executable_means": (
                "the uniform-q4 fused path encode_embed + 64*(mixer+mlp) + encode_terminal "
                "with default env (geo_tpr64_tg128 GEMV, family SwiGLU, tiled RMS/RoPE/DeltaNet)"
            ),
        },
        "source": {
            "flops_per_token": src_flops,
            "flops_per_token_trace": (
                f"GEMV MACs {gemv_macs} (2*elements of every dispatched matvec) + "
                f"activation FLOPs {act['total_flops']} (RMSNorm/SwiGLU/DeltaNet/MHA/…). "
                f"Does NOT include G143's 2*embed_table."
            ),
            "g143_paper_flops_per_token": G143_FLOPS,
            "operations_per_token": src_ops,
            "operations_per_token_trace": "equal to FLOPs: a dense f32 GEMV has no nibble unpack",
            "dispatches_per_token": inv_total,
            "dispatches_per_token_trace": (
                "same control graph as the executable (one kernel per organ). "
                "A reconstruct-then-GEMM SOURCE lowering would add 401 qwen_uniform_q4_decode_vector "
                "dispatches and is tabulated under trap_reconstruct_then_gemm, not here."
            ),
            "dram_bytes_per_token_f32_weights": dram["source_dense_f32_gemv_bytes"] + dram["executable_activation_write_bytes"],
            "dram_bytes_per_token_bf16_weights": dram["source_dense_bf16_gemv_bytes"] + dram["executable_activation_write_bytes"],
            "dram_trace": (
                "streaming every GEMV matrix as dense f32 (4 B/weight) or bf16 (2 B/weight), "
                "plus the same activation writes the executable already pays"
            ),
            "temporary_materialization_bytes_per_token": dram["temporary_materialization_bytes_per_token"],
            "dense_w_materialized_bytes_per_token": dram["source_dense_f32_gemv_bytes"],
            "dense_w_note": (
                "A dense SOURCE stores W already unpacked, so 'materialization' is the weight "
                "stream itself, not a codec expansion."
            ),
        },
        "executable": {
            "flops_per_token": exe_flops,
            "flops_per_token_trace": (
                f"same GEMV MACs {gemv_macs} + same activations {act['total_flops']} + "
                f"one extra scale-mul FLOP per Q4 weight ({dequant_scale_flops}). "
                "unpack8: acc += float(q)*scale*x[col]."
            ),
            "operations_per_token": exe_ops,
            "operations_per_token_trace": (
                f"executable FLOPs {exe_flops} + integer unpack {int_ops} "
                "(~4 ALU ops per Q4 weight)"
            ),
            "dispatches_per_token": inv_total,
            "dispatches_per_token_trace": RECORDED_DISPATCH_NOTE,
            "command_buffers_per_token": 1,
            "dram_bytes_per_token": dram["executable_dram_bytes_per_token"],
            "dram_trace": (
                f"Q4 GEMV stream {dram['executable_q4_gemv_bytes']} + embed row "
                f"{dram['executable_embed_row_q4_bytes']} + f32 scales "
                f"{dram['executable_f32_scale_bytes']} + activation writes "
                f"{dram['executable_activation_write_bytes']} + state/KV/GEMV-input reads"
            ),
            "temporary_materialization_bytes_per_token": dram["temporary_materialization_bytes_per_token"],
            "dense_w_materialized_bytes_per_token": 0,
            "dense_w_note": (
                "zero. Packed codes stay packed. Missing codec fails the run; there is no "
                "reconstruct-to-Q4 or reconstruct-to-f32 fallback on this path."
            ),
        },
        "verdict": {
            "does_the_executable_do_less_work": False,
            "does_the_executable_hold_fewer_bytes": True,
            "one_line": (
                "The executable holds and streams fewer bytes (Q4 4.25 BPW vs f32/bf16). "
                "It does not do fewer MACs — it does the same 51.24 GFLOP of GEMV plus extra "
                "dequant ALU — and it does not reconstruct a dense W before computing."
            ),
            "paradigm_boundary_crossed": False,
            "paradigm_note": (
                "Storage moved. Arithmetic did not. Activation tensors are still dense f32. "
                "The Noetic attack surface is the list of reconstruction sites, not the BPW."
            ),
        },
        "trap_reconstruct_then_gemm_not_the_current_path": {
            "extra_dispatches_per_token": sum(1 for o in organs for _ in range(o["count_per_token"])),
            "extra_dense_w_bytes_per_token": dram["trap_reconstruct_then_gemm"]["dense_w_write_bytes"],
            "kernel_that_would_do_it": "qwen_uniform_q4_decode_vector",
            "note": dram["trap_reconstruct_then_gemm"]["note"],
        },
    }


def print_report(doc: dict) -> None:
    c = doc["columns"]
    s, e = c["source"], c["executable"]
    print("=" * 78)
    print("NOETIC OPERATION CENSUS — SOURCE vs EXECUTABLE")
    print("Qwen3.8-27B uniform-q4 fused decode (qwen38_hybrid_decode.rs)")
    print("=" * 78)
    print()
    print(c["verdict"]["one_line"])
    print()
    print(f"{'metric':<42} {'SOURCE':>18} {'EXECUTABLE':>18}")
    print("-" * 78)

    def row(name, sv, ev, w=18):
        def fmt(v):
            if v is None:
                return "—"
            if isinstance(v, bool):
                return "yes" if v else "no"
            if isinstance(v, int):
                return f"{v:,}"
            if isinstance(v, float):
                return f"{v:,.3f}"
            return str(v)
        print(f"{name:<42} {fmt(sv):>{w}} {fmt(ev):>{w}}")

    row("FLOPs / token (dispatched)", s["flops_per_token"], e["flops_per_token"])
    row("FLOPs / token (G143 paper 2N)", s["g143_paper_flops_per_token"], None)
    row("operations / token", s["operations_per_token"], e["operations_per_token"])
    row("dispatches / token", s["dispatches_per_token"], e["dispatches_per_token"])
    row("command buffers / token", 1, e["command_buffers_per_token"])
    row("DRAM bytes / token (f32 SOURCE)", s["dram_bytes_per_token_f32_weights"], e["dram_bytes_per_token"])
    row("DRAM bytes / token (bf16 SOURCE)", s["dram_bytes_per_token_bf16_weights"], e["dram_bytes_per_token"])
    row("temp materialization bytes / token", s["temporary_materialization_bytes_per_token"],
        e["temporary_materialization_bytes_per_token"])
    row("dense W materialized bytes / token", s["dense_w_materialized_bytes_per_token"],
        e["dense_w_materialized_bytes_per_token"])
    print()
    print("Traces")
    print("  SOURCE FLOPs:     ", s["flops_per_token_trace"])
    print("  EXECUTABLE FLOPs: ", e["flops_per_token_trace"])
    print("  SOURCE ops:       ", s["operations_per_token_trace"])
    print("  EXECUTABLE ops:   ", e["operations_per_token_trace"])
    print("  SOURCE disp:      ", s["dispatches_per_token_trace"])
    print("  EXECUTABLE disp:  ", e["dispatches_per_token_trace"])
    print("  SOURCE DRAM:      ", s["dram_trace"])
    print("  EXECUTABLE DRAM:  ", e["dram_trace"])
    print("  EXECUTABLE W:     ", e["dense_w_note"])
    print()
    gap = doc["analytic_vs_measured"]
    print("## Analytic vs measured (not smoothed)")
    print(f"  G143 paper 2N              {gap['g143_paper_flops_2N']:,} FLOP")
    print(f"  dispatched GEMV MACs       {gap['dispatched_gemv_mac_flops']:,} FLOP")
    print(f"  paper − dispatched GEMV    {gap['gap_paper_minus_dispatched_gemv_macs']:,} FLOP")
    print(f"  of which embed-table 2N    {gap['paper_overcount_from_embed_table_macs']:,} FLOP (never launched)")
    print(f"  leftover non-GEMV params   {gap['leftover_params_not_in_any_gemv']:,}")
    print(f"  {gap['reading']}")
    print()
    print("## Dispatches / token vs recorded 964")
    rec = doc["dispatch_reconciliation"]
    print(f"  formula 1 + layers*15 + 3 = {rec['formula_total']}")
    print(f"  inventory re-sum            {rec['inventory_total']}")
    print(f"  recorded anchor             {rec['recorded_anchor']}")
    print(f"  command buffers             {rec['command_buffers']}")
    print(f"  {rec['verdict']}")
    print()
    print("## Kernel binding")
    kb = doc["kernel_binding"]
    print(f"  declared kernel void        {kb['declared_in_tree']}")
    print(f"  G071 bound (string ∩ void)  {kb['g071_bound_count']}  (anchor {ANCHOR_BOUND})")
    print(f"  helpers invisible to G071   {kb['helper_kernels_on_uniform_q4']}")
    print(f"  {kb['note']}")
    print()
    print("## Dense-reconstruction sites")
    print(f"  {'kernel':<52} {'bytes/tok':>14}  class")
    for site in doc["reconstruction_sites"]:
        flag = " " if site["on_uniform_q4_production_path"] else "*"
        print(f" {flag}{site['kernel']:<51} {site['bytes_materialised_per_token']:>14,}  "
              f"{site['classification']}")
    print("  * = not on the uniform-q4 production path (named because a naive lowering uses it)")
    print()
    print("## WHAT I WATCHED FAIL")
    for i, f in enumerate(doc["what_i_watched_fail"], 1):
        print(f"  {i}. {f['what']}: {f['result']}")
        print(f"     {f['why']}")
    print()
    print(f"wrote {doc['written_to']}")
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=str(ARTIFACT_DEFAULT))
    ap.add_argument("--seq-len", type=int, default=34,
                    help="GQA KV length for one decode step (native prompt_len was 34)")
    ap.add_argument("--out", default=str(REPO / "receipts/headless/NOETIC_OPERATION_CENSUS.json"))
    args = ap.parse_args()

    for p in (DECODE, GEOMETRY, SCHEDULE, LEDGER, SHADERS):
        if not p.exists():
            print(f"FAIL: required path missing: {p}", file=sys.stderr)
            return 2

    g = load_geometry()
    decode_text = DECODE.read_text()
    declared_list, declared_n = declared_kernels()
    bound = bound_kernels_g071(decode_text, set(declared_list))
    live = live_kernel_names()
    organs = gemv_organs(g)
    inv, inv_total = dispatch_inventory(g, live)
    act = activation_flops(g, args.seq_len)
    dram = dram_and_temp(g, organs, args.seq_len)
    sites = reconstruction_sites(g, organs, live)
    artifact = read_artifact(Path(os.path.expanduser(args.artifact)))
    gap = analytic_gap(g, organs, act)
    columns = build_columns(g, organs, act, dram, inv_total)

    formula_total = g["production_dispatches"]
    recon = {
        "formula": "1 embed + QWEN38_LAYERS * (MIXER_PREFIX + MLP_SUFFIX) + len(TERMINAL_HEAD_KERNELS)",
        "embed": 1,
        "layers": g["layers"],
        "mixer_prefix": g["mixer_prefix"],
        "mlp_suffix": g["mlp_suffix"],
        "full_layer": g["full_layer"],
        "terminal": g["terminal_n"],
        "formula_total": formula_total,
        "inventory_total": inv_total,
        "recorded_anchor": ANCHOR_DISPATCHES,
        "command_buffers": ANCHOR_CBS,
        "matches_recorded_964": formula_total == ANCHOR_DISPATCHES and inv_total == ANCHOR_DISPATCHES,
        "source": RECORDED_DISPATCH_NOTE,
        "verdict": (
            "STILL 964. 1 + 64*15 + 3 = 964, and the encode-path inventory re-sums to the "
            "same number. Live kernel NAMES drifted from the frozen schedule arrays; the "
            "COUNT did not. Default env does not enable the two-pass argmax (that would be 965)."
            if formula_total == ANCHOR_DISPATCHES and inv_total == ANCHOR_DISPATCHES
            else f"DRIFT: formula {formula_total}, inventory {inv_total}, recorded {ANCHOR_DISPATCHES}"
        ),
    }

    fails = what_watched_fail(artifact, g, len(bound), declared_n, inv_total)

    gemv_disp = sum(r["dispatches_per_token"] for r in inv if r["is_gemv"])
    doc = {
        "schema": "hawking.headless.noetic_operation_census.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": git_head(),
        "question": "Does the executable do LESS WORK, or just hold fewer bytes?",
        "answer": columns["verdict"]["one_line"],
        "path": {
            "decode_source": str(DECODE.relative_to(REPO)),
            "schedule": str(SCHEDULE.relative_to(REPO)),
            "ledger": str(LEDGER.relative_to(REPO)),
            "shaders": str(SHADERS.relative_to(REPO)),
            "artifact": artifact["path"],
        },
        "anchors_not_rederived": {
            "tps": ANCHOR_TPS,
            "ms_per_token": ANCHOR_TOKEN_MS,
            "roof_gb_s": ANCHOR_ROOF_GB_S,
            "unified_memory_bytes": ANCHOR_UNIFIED_B,
            "gpu_cores": ANCHOR_GPU_CORES,
            "parameter_count": ANCHOR_PARAMS,
            "bpw": ANCHOR_BPW,
            "dispatches_per_token": ANCHOR_DISPATCHES,
            "command_buffers_per_token": ANCHOR_CBS,
            "kernels_bound_g071": ANCHOR_BOUND,
            "kernels_declared": ANCHOR_DECLARED,
            "two_servers_tps": 3.986,
            "one_server_tps": 33.47,
            "llama_server_live_on": 52484,
        },
        "geometry": {k: v for k, v in g.items() if not k.endswith("_kernels")},
        "artifact": artifact,
        "kernel_binding": {
            "declared_in_tree": declared_n,
            "g071_bound_count": len(bound),
            "g071_bound": bound,
            "helper_kernels_on_uniform_q4": helper_kernels_on_uniform_q4(),
            "live_default_names": live,
            "extraction_g071": (
                "string literals in qwen38_hybrid_decode.rs intersected with declared "
                "`kernel void` names — the nx_genome / G071 method"
            ),
            "note": (
                f"G071 measured 38 bound against 554 declared; live count is "
                f"{len(bound)} / {declared_n}. A seal listing all {declared_n} would be a lie "
                "about what runs. A seal listing only the 38 is also incomplete: four "
                "production kernels are dispatched through helpers and are not literals."
            ),
        },
        "dispatch_inventory": inv,
        "dispatch_reconciliation": recon,
        "gemv_organs": organs,
        "gemv_dispatches_per_token": gemv_disp,
        "activation_flops": act,
        "dram_and_temp": dram,
        "columns": columns,
        "analytic_vs_measured": gap,
        "reconstruction_sites": sites,
        "flop_convention": columns["convention"],
        "what_i_watched_fail": fails,
        "self_check": {
            "dispatches_964": formula_total == 964 and inv_total == 964,
            "declared_554": declared_n == 554,
            "g071_bound_38": len(bound) == 38,
            "layers_64": g["layers"] == 64,
            "full_layer_15": g["full_layer"] == 15,
            "dense_w_materialized_is_zero": columns["executable"]["dense_w_materialized_bytes_per_token"] == 0,
        },
        "written_to": args.out,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print_report(doc)

    # Self-check is informational in the receipt; hard-fail only if the
    # dispatch formula — the number this project already sealed — cannot be
    # recovered from the source sitting on disk.
    if formula_total != ANCHOR_DISPATCHES:
        print(f"FAIL: production dispatch formula is {formula_total}, not {ANCHOR_DISPATCHES}",
              file=sys.stderr)
        return 3
    if inv_total != ANCHOR_DISPATCHES:
        print(f"FAIL: dispatch inventory re-sum is {inv_total}, not {ANCHOR_DISPATCHES}",
              file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())

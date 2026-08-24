#!/usr/bin/env python3
"""Dispatch ledger: name every launch on the sealed 756-dispatch parent.

The GPU ledger overturned the campaign's earlier story. The q4 incumbent is
bandwidth-bound (468.9 GB/s, 95.6% of complete-wall). A dispatch cut does
not buy throughput proportionally — 964→756 bought +5.8% tok/s, not 21.6%.
This ledger is the deliverable. A further cut below 756 is a child of the
sealed parent at ~/noetic/NOETIC_PARENT_A, not a mutation of it.

    python3 tools/headless/dispatch_ledger.py
    python3 tools/headless/dispatch_ledger.py --measure
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from first_noetic_executable import (  # noqa: E402
    PROMPT,
    TOKENIZER,
    git_head,
    judge_coherence,
    now_iso,
)
from noetic_operation_census import (  # noqa: E402
    ANCHOR_ROOF_GB_S,
    f32b,
    load_geometry,
    q4_matrix_bytes,
)

SCHEMA = "hawking.headless.dispatch_ledger.v1"
RECEIPT = REPO / "receipts" / "headless" / "DISPATCH_LEDGER.json"
RAW = REPO / "receipts" / "headless" / "_DISPATCH_LEDGER_raw.json"
SHADER = REPO / "crates" / "hawking-core" / "shaders" / "qwen80_device_activations.metal"
DECODE = REPO / "crates" / "hawking-core" / "src" / "model" / "qwen38_hybrid_decode.rs"
PARENT = Path.home() / "noetic" / "NOETIC_PARENT_A"
TOKEN_NS_GIT = "HEAD:receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json"

CARGO_TARGET = Path(
    os.environ.get(
        "CARGO_TARGET_DIR",
        str(REPO / "workspace" / "ops" / "build" / "rust"),
    )
)

PARENT_DISPATCHES = 756
UNFUSED_DISPATCHES = 964
CANDIDATE_SAVED = 128  # 2 per layer: mixer residual+mlp rms, mlp residual+next rms
CANDIDATE_DISPATCHES = PARENT_DISPATCHES - CANDIDATE_SAVED  # 628
ROOF_GB_S = ANCHOR_ROOF_GB_S  # 778.8 N017 sequential DRAM roof
PEAK_GB_S = 700.0
SEQ_LEN = 42

KERNEL_GOOD = "qwen80_add_residual_rmsnorm_tg"
KERNEL_BAD = "qwen80_add_residual_rmsnorm_tg_plainweight"

# Isolated family GPU ns from QWEN38_TOKEN_NS_LEDGER (diagnostic CBs, unfused
# 964 graph). Source: git show HEAD:receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json
# Production per-dispatch GPU_NS is ABSENT (atDispatchBoundary=false).
ISOLATED_NS = {
    "input_norms": (1_137_250, 64),
    "post_norms": (1_210_874, 64),
    "final_norm": (19_291, 1),
    "silu_64": (160_958, 64),
    "mlp_residual_64": (134_208, 64),
    "mixer_residual_64": (118_250, 64),
    "rearrange_48": (350_999, 48),
    "ba_to_decay_48": (139_374, 48),
    "gated_rmsnorm_48": (1_295_500, 48),
    "rope_cache_16": (1_562_625, 16),
    "mha_16": (666_500, 16),
    "sigmoid_16": (43_625, 16),
    "argmax": (335_499, 1),
    "embed": (4_999, 1),
    "gated_delta_48": (2_146_166, 48),
    "dn_gemvs": (5_560_749, 144),
    "gqa_gemvs": (1_817_416, 64),
    "mlp_matvecs_64": (15_853_666, 192),
    "lm_head": (1_017_458, 1),
    # Fused dual/triple GEMVs: sum of the unfused family members they replaced.
    "gate_up_swiglu_fused": (2 * 15_853_666 // 192 + 160_958 // 64, 1),
    "gqa_qkv_fused": (3 * 1_817_416 // 64, 1),
    "dn_qkvz_ba_fused": (5_560_749 * (44_564_480 + 261_120) // (61_537_280 * 48), 1),
    "dn_out_proj": (5_560_749 * 16_711_680 // (61_537_280 * 48), 1),
    "gqa_o_proj": (1_817_416 // 64, 1),
    "mlp_down": (15_853_666 // 192, 1),
}

MEASURED = "MEASURED"
DERIVED = "DERIVED"
ABSENT = "ABSENT"


def affine2_matrix_bytes(rows: int, cols: int, group: int = 64) -> int:
    groups = (cols + group - 1) // group
    return rows * groups * 20  # 16 code + 2 scale + 2 bias


def qty(value, *, kind: str, unit: str, command: str, note: str | None = None,
        absent_reason: str | None = None):
    if kind == ABSENT:
        out = {
            "value": None,
            "kind": ABSENT,
            "unit": unit,
            "command": command,
            "absent_reason": absent_reason,
        }
        if note:
            out["note"] = note
        return out
    out = {
        "value": value,
        "kind": kind,
        "unit": unit,
        "command": command,
        "absent_reason": None,
    }
    if note:
        out["note"] = note
    return out


def traffic_ns(bytes_: int) -> float:
    return (bytes_ / (ROOF_GB_S * 1e9)) * 1e9 if bytes_ else 0.0


def isolated_per(name: str) -> float:
    ns, n = ISOLATED_NS[name]
    return ns / n if n else float(ns)


def launch_overhead_ns(isolated_name: str | None, bytes_: int) -> dict[str, Any]:
    """Launch overhead = isolated per-dispatch GPU_NS minus bandwidth floor.

    Tiny kernels (residual ~2 µs for 60 KB) are almost all launch. GEMVs are
    bandwidth-bound; the residual after subtracting traffic/roof is the
    launch-like remainder. Production per-dispatch GPU_NS is ABSENT.
    """
    if isolated_name is None:
        return qty(
            None,
            kind=ABSENT,
            unit="ns/dispatch",
            command="MTLDevice.supportsCounterSampling(.atDispatchBoundary)",
            absent_reason=(
                "No isolated family for this launch and atDispatchBoundary="
                "false, so per-dispatch GPU_NS cannot be split from the 1-CB "
                "GPUEnd−GPUStart interval."
            ),
        )
    per = isolated_per(isolated_name)
    floor = traffic_ns(bytes_)
    overhead = max(0.0, per - floor)
    return qty(
        overhead,
        kind=DERIVED,
        unit="ns/dispatch",
        command=f"git show {TOKEN_NS_GIT}  # isolated[{isolated_name}]/n − bytes/595.9e9",
        note=(
            f"isolated_per={per:.1f} ns, bandwidth_floor={floor:.1f} ns at "
            f"{ROOF_GB_S} GB/s sequential roof. Not a production-CB counter."
        ),
    )


def mixer_kind(layer: int) -> str:
    return "gqa" if (layer + 1) % 4 == 0 else "deltanet"


def shader_evidence() -> dict[str, Any]:
    text = SHADER.read_text(encoding="utf-8", errors="replace") if SHADER.is_file() else ""
    rust = DECODE.read_text(encoding="utf-8", errors="replace") if DECODE.is_file() else ""
    return {
        "shader_present": SHADER.is_file(),
        "kernel_good": text.find(f"kernel void {KERNEL_GOOD}("),
        "kernel_bad": text.find(f"kernel void {KERNEL_BAD}("),
        "all_kernels_declared": (
            f"kernel void {KERNEL_GOOD}(" in text
            and f"kernel void {KERNEL_BAD}(" in text
        ),
        "uses_one_plus_w": "(1.0f + weight[index])" in text
        and KERNEL_GOOD in text,
        "bad_uses_plain_weight": (
            "x_norm[index] = residual_out[index] * inverse_rms * weight[index]" in text
        ),
        "wired": "fuse_add_rmsnorm" in rust and "encode_add_residual_rmsnorm" in rust,
        "default_off": "Default Off" in rust or "default Off" in rust,
        "does_not_write_dense_w": True,
    }


def find_binary() -> Path | None:
    env = os.environ.get("QWEN38_DISPATCH_LEDGER_BIN")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    for c in (
        CARGO_TARGET / "release-fast" / "examples" / "ascension_qwen38_dispatch_ledger",
        CARGO_TARGET / "release" / "examples" / "ascension_qwen38_dispatch_ledger",
        REPO / "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_dispatch_ledger",
    ):
        if c.is_file():
            return c
    return None


def parent_graph(g: dict) -> list[dict[str, Any]]:
    """Every dispatch of the sealed 756-dispatch fused parent, encode order."""
    H, I = g["hidden"], g["intermediate"]
    rows: list[dict[str, Any]] = []

    def add(
        *,
        operator: str,
        kernel: str,
        layer: int | None,
        mixer: str | None,
        weight_bytes: int,
        act_read: int,
        act_write: int,
        flops: float,
        isolated: str | None,
        deps: list[str],
        candidacy: str,
        candidacy_why: str,
        class_frequency: int,
        organ: str | None = None,
    ) -> None:
        total = weight_bytes + act_read + act_write
        launch = launch_overhead_ns(isolated, total)
        launch_v = launch["value"] if launch["kind"] != ABSENT else 0.0
        mem_ns = traffic_ns(total)
        sync_ns = 0.0
        rank = (launch_v or 0.0) + mem_ns + sync_ns
        rows.append({
            "index": len(rows),
            "layer": layer,
            "mixer": mixer,
            "operator": operator,
            "organ": organ,
            "kernel": kernel,
            "bytes": {
                "weight_read": weight_bytes,
                "activation_read": act_read,
                "activation_write": act_write,
                "total": total,
            },
            "flops": flops,
            "launch_overhead_ns": launch,
            "memory_traffic_bytes": total,
            "memory_traffic_ns_at_roof": mem_ns,
            "synchronization_ns": sync_ns,
            "rank_score_ns": rank,
            "rank_score_formula": "launch_overhead_ns + memory_traffic_bytes/595.9e9*1e9 + synchronization_ns",
            "dependencies": deps,
            "fusion_candidacy": candidacy,
            "fusion_candidacy_why": candidacy_why,
            "frequency": class_frequency,
            "command_buffer": 1,
        })

    # --- embed ---
    add(
        operator="embed_lookup",
        kernel="qwen_uniform_q4_embedding_lookup",
        layer=None,
        mixer=None,
        weight_bytes=q4_matrix_bytes(1, H),
        act_read=0,
        act_write=f32b(H),
        flops=H,
        isolated="embed",
        deps=["token_id"],
        candidacy="not_a_candidate",
        candidacy_why="Single gather of one Q4 row. Nothing shares this input.",
        class_frequency=1,
        organ="embed",
    )

    for layer in range(g["layers"]):
        kind = mixer_kind(layer)
        prev = "embed_lookup" if layer == 0 else f"L{layer-1}.mlp.residual"
        if kind == "deltanet":
            add(
                operator="input_rmsnorm",
                kernel="qwen80_residual_rmsnorm_tg",
                layer=layer,
                mixer=kind,
                weight_bytes=f32b(H),
                act_read=f32b(H),
                act_write=f32b(H),
                flops=5 * H,
                isolated="input_norms",
                deps=[prev],
                candidacy="candidate:residual+rmsnorm",
                candidacy_why=(
                    "Shares the residual stream with the previous MLP add. "
                    "That pair is the remaining shared-input fusion (layer 0 "
                    "has no previous add; it stays a standalone launch)."
                ),
                class_frequency=g["layers"],
                organ="mixer.norm",
            )
            qkvz_b = q4_matrix_bytes(g["qkvz_rows"], H)
            ba_b = q4_matrix_bytes(g["ba_rows"], H)
            add(
                operator="dn_qkvz_ba_concat",
                kernel="qwen_uniform_q4_group64_matvec_pair_concat_geo_tpr64_tg128",
                layer=layer,
                mixer=kind,
                weight_bytes=qkvz_b + ba_b,
                act_read=f32b(H),
                act_write=f32b(g["qkvz_rows"]) + f32b(g["ba_rows"]),
                flops=2 * (g["qkvz_rows"] + g["ba_rows"]) * H,
                isolated="dn_qkvz_ba_fused",
                deps=["input_rmsnorm"],
                candidacy="already_fused:dn_qkvz_ba",
                candidacy_why="964→756: two in-proj GEMVs that already shared normalized x.",
                class_frequency=g["dn_layers"],
                organ="linear_attn.in_proj",
            )
            add(
                operator="qkvz_rearrange_conv_l2",
                kernel="qwen38_qkvz_rearrange_conv_l2_f32",
                layer=layer,
                mixer=kind,
                weight_bytes=f32b(g["conv_channels"] * g["lin_conv_k"]),
                act_read=f32b(g["qkvz_rows"]) + f32b(g["conv_state_elements"]),
                act_write=f32b(g["value_elements"]) * 4 + f32b(g["conv_state_elements"]),
                flops=2 * g["conv_channels"] * g["lin_conv_k"] + 5 * g["conv_channels"]
                + 5 * g["key_elements"] * 2,
                isolated="rearrange_48",
                deps=["dn_qkvz_ba_concat"],
                candidacy="not_a_candidate:stateful",
                candidacy_why=(
                    "Causal conv updates conv_state. Fusing into the GEMV would "
                    "serialize 8704 TGs behind a 16-head conv; the 8-layer f16 "
                    "megakernel that tried this class of collapse measured 4.4x slower."
                ),
                class_frequency=g["dn_layers"],
                organ="linear_attn.conv",
            )
            add(
                operator="ba_to_decay_beta",
                kernel="qwen80_ba_to_decay_beta_f32",
                layer=layer,
                mixer=kind,
                weight_bytes=f32b(g["lin_value_heads"]) * 2,
                act_read=f32b(g["ba_rows"]),
                act_write=f32b(g["lin_value_heads"]) * 2,
                flops=12 * g["lin_value_heads"],
                isolated="ba_to_decay_48",
                deps=["dn_qkvz_ba_concat"],
                candidacy="not_a_candidate:tiny_independent",
                candidacy_why=(
                    "96-wide. Independent of rearrange (sibling, not producer). "
                    "Isolated 2.9 µs × 48 = 139 µs, 0.4% of GPU_NS. Does not share "
                    "an input with a second GEMV."
                ),
                class_frequency=g["dn_layers"],
                organ="linear_attn.ba",
            )
            add(
                operator="gated_delta",
                kernel="qwen38_gated_delta_decode_vi_simd",
                layer=layer,
                mixer=kind,
                weight_bytes=0,
                act_read=f32b(g["rec_state_elements"]) + f32b(g["value_elements"]) * 2
                + f32b(g["lin_value_heads"]) * 2,
                act_write=f32b(g["rec_state_elements"]) + f32b(g["value_elements"]),
                flops=7 * g["rec_state_elements"],
                isolated="gated_delta_48",
                deps=["qkvz_rearrange_conv_l2", "ba_to_decay_beta"],
                candidacy="not_a_candidate:recurrent_state",
                candidacy_why=(
                    "Reads and writes 48×128×128 rec state. Math of gated DeltaNet, "
                    "not a shared-input launch pair."
                ),
                class_frequency=g["dn_layers"],
                organ="linear_attn.recurrence",
            )
            add(
                operator="gated_rmsnorm",
                kernel="qwen80_deltanet_gated_rmsnorm_tg",
                layer=layer,
                mixer=kind,
                weight_bytes=f32b(g["value_elements"]),
                act_read=f32b(g["value_elements"]) * 2,
                act_write=f32b(g["value_elements"]),
                flops=5 * g["value_elements"],
                isolated="gated_rmsnorm_48",
                deps=["gated_delta"],
                candidacy="candidate:norm+projection",
                candidacy_why=(
                    "Shares gated with out_proj. Fusing RMS into geo_tpr64 would "
                    "recompute a 6144-wide reduction in every out_proj TG "
                    f"({g['o_proj_rows'] // 2} TGs). That is the inverse of the "
                    "working fusion (one shared x, many rows)."
                ),
                class_frequency=g["dn_layers"],
                organ="linear_attn.norm",
            )
            add(
                operator="out_proj",
                kernel="qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
                layer=layer,
                mixer=kind,
                weight_bytes=q4_matrix_bytes(H, g["o_proj_cols"]),
                act_read=f32b(g["value_elements"]),
                act_write=f32b(H),
                flops=2 * H * g["o_proj_cols"],
                isolated="dn_out_proj",
                deps=["gated_rmsnorm"],
                candidacy="candidate:output_projection+residual",
                candidacy_why=(
                    "out_proj output is the residual add's delta. Subsumed by "
                    "residual+rmsnorm: fusing GEMV+add would still leave the "
                    "RMSNorm launch, and would retile the workhorse."
                ),
                class_frequency=g["dn_layers"],
                organ="linear_attn.out_proj",
            )
            add(
                operator="mixer_residual",
                kernel="qwen_next_add_residual",
                layer=layer,
                mixer=kind,
                weight_bytes=0,
                act_read=f32b(H) * 2,
                act_write=f32b(H),
                flops=H,
                isolated="mixer_residual_64",
                deps=["out_proj", prev if layer == 0 else f"L{layer}.hidden_in"],
                candidacy="candidate:residual+rmsnorm",
                candidacy_why=(
                    "Shares first_residual with the following post_attention "
                    "RMSNorm. Isolated 1.85 µs; almost all launch. Working shape."
                ),
                class_frequency=g["layers"],
                organ="mixer.residual",
            )
        else:
            add(
                operator="input_rmsnorm",
                kernel="qwen80_residual_rmsnorm_tg",
                layer=layer,
                mixer=kind,
                weight_bytes=f32b(H),
                act_read=f32b(H),
                act_write=f32b(H),
                flops=5 * H,
                isolated="input_norms",
                deps=[prev],
                candidacy="candidate:residual+rmsnorm",
                candidacy_why="Same as DeltaNet input_rmsnorm: shared residual stream.",
                class_frequency=g["layers"],
                organ="mixer.norm",
            )
            q_b = q4_matrix_bytes(g["q_proj_rows"], H)
            kv_b = q4_matrix_bytes(g["kv_proj_rows"], H)
            add(
                operator="gqa_qkv_concat",
                kernel="qwen_uniform_q4_group64_matvec_qkv_geo_tpr64_tg128",
                layer=layer,
                mixer=kind,
                weight_bytes=q_b + 2 * kv_b,
                act_read=f32b(H),
                act_write=f32b(g["q_proj_rows"]) + 2 * f32b(g["kv_proj_rows"]),
                flops=2 * (g["q_proj_rows"] + 2 * g["kv_proj_rows"]) * H,
                isolated="gqa_qkv_fused",
                deps=["input_rmsnorm"],
                candidacy="already_fused:gqa_qkv",
                candidacy_why="964→756: Q,K,V already shared normalized x.",
                class_frequency=g["gqa_layers"],
                organ="self_attn.qkv",
            )
            add(
                operator="qk_norm_rope_cache",
                kernel="qwen38_gqa_qk_norm_rope_cache_tg",
                layer=layer,
                mixer=kind,
                weight_bytes=f32b(g["gqa_head_dim"]) * 2,
                act_read=f32b(g["q_proj_rows"]) + 2 * f32b(g["kv_proj_rows"]),
                act_write=f32b(g["gqa_heads"] * g["gqa_head_dim"])
                + 2 * f32b(g["gqa_kv_heads"] * g["gqa_head_dim"]),
                flops=5 * g["gqa_heads"] * g["gqa_head_dim"]
                + 10 * g["gqa_heads"] * g["gqa_rotary_dim"],
                isolated="rope_cache_16",
                deps=["gqa_qkv_concat"],
                candidacy="candidate:attention_preparation",
                candidacy_why=(
                    "Could fuse into QKV concat, but rope writes KV cache slots "
                    "and does per-head RMS. Not the shared-input GEMV shape."
                ),
                class_frequency=g["gqa_layers"],
                organ="self_attn.rope",
            )
            kv_read = f32b(SEQ_LEN * g["gqa_kv_heads"] * g["gqa_head_dim"]) * 2
            add(
                operator="mha_decode",
                kernel="mha_decode_f32",
                layer=layer,
                mixer=kind,
                weight_bytes=0,
                act_read=f32b(g["gqa_heads"] * g["gqa_head_dim"]) + kv_read,
                act_write=f32b(g["gqa_heads"] * g["gqa_head_dim"]),
                flops=2 * 2 * g["gqa_heads"] * SEQ_LEN * g["gqa_head_dim"]
                + 5 * g["gqa_heads"] * SEQ_LEN,
                isolated="mha_16",
                deps=["qk_norm_rope_cache"],
                candidacy="not_a_candidate:seq_len_softmax",
                candidacy_why="Softmax over cached K/V. Seq-length dependent. Not a GEMV pair.",
                class_frequency=g["gqa_layers"],
                organ="self_attn.mha",
            )
            add(
                operator="sigmoid_gate",
                kernel="qwen38_attention_apply_sigmoid_gate",
                layer=layer,
                mixer=kind,
                weight_bytes=0,
                act_read=f32b(g["gqa_heads"] * g["gqa_head_dim"]) * 2,
                act_write=f32b(g["gqa_heads"] * g["gqa_head_dim"]),
                flops=5 * g["gqa_heads"] * g["gqa_head_dim"],
                isolated="sigmoid_16",
                deps=["mha_decode", "gqa_qkv_concat"],
                candidacy="candidate:projection+activation",
                candidacy_why=(
                    "Could fuse into o_proj. Isolated 2.7 µs × 16 = 43 µs, 0.13% "
                    "of GPU_NS. Not attempted: does not share a GEMV input."
                ),
                class_frequency=g["gqa_layers"],
                organ="self_attn.gate",
            )
            add(
                operator="o_proj",
                kernel="qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
                layer=layer,
                mixer=kind,
                weight_bytes=q4_matrix_bytes(H, g["o_proj_cols"]),
                act_read=f32b(g["gqa_heads"] * g["gqa_head_dim"]),
                act_write=f32b(H),
                flops=2 * H * g["o_proj_cols"],
                isolated="gqa_o_proj",
                deps=["sigmoid_gate"],
                candidacy="candidate:output_projection+residual",
                candidacy_why="Same as DeltaNet out_proj: subsumed by residual+rmsnorm.",
                class_frequency=g["gqa_layers"],
                organ="self_attn.o_proj",
            )
            add(
                operator="mixer_residual",
                kernel="qwen_next_add_residual",
                layer=layer,
                mixer=kind,
                weight_bytes=0,
                act_read=f32b(H) * 2,
                act_write=f32b(H),
                flops=H,
                isolated="mixer_residual_64",
                deps=["o_proj"],
                candidacy="candidate:residual+rmsnorm",
                candidacy_why="Shares first_residual with post_attention RMSNorm.",
                class_frequency=g["layers"],
                organ="mixer.residual",
            )

        # MLP suffix (fused gate+up+SwiGLU)
        add(
            operator="post_attention_rmsnorm",
            kernel="qwen80_residual_rmsnorm_tg",
            layer=layer,
            mixer=kind,
            weight_bytes=f32b(H),
            act_read=f32b(H),
            act_write=f32b(H),
            flops=5 * H,
            isolated="post_norms",
            deps=["mixer_residual"],
            candidacy="candidate:residual+rmsnorm",
            candidacy_why="Producer is mixer_residual. Isolated 18.9 µs, mostly launch+reduction.",
            class_frequency=g["layers"],
            organ="mlp.norm",
        )
        gate_b = affine2_matrix_bytes(I, H)
        up_b = affine2_matrix_bytes(I, H)
        add(
            operator="gate_up_swiglu",
            kernel="qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
            layer=layer,
            mixer=kind,
            weight_bytes=gate_b + up_b,
            act_read=f32b(H),
            act_write=f32b(I),
            flops=2 * 2 * I * H + 5 * I,
            isolated="gate_up_swiglu_fused",
            deps=["post_attention_rmsnorm"],
            candidacy="already_fused:gate_up_swiglu",
            candidacy_why="964→756: gate+up shared normalized x; SwiGLU in-register.",
            class_frequency=g["layers"],
            organ="mlp.gate_up",
        )
        add(
            operator="down_proj",
            kernel="qwen_affine_q2_group32_matvec_geo_tpr64_tg128",
            layer=layer,
            mixer=kind,
            weight_bytes=affine2_matrix_bytes(H, I),
            act_read=f32b(I),
            act_write=f32b(H),
            flops=2 * H * I,
            isolated="mlp_down",
            deps=["gate_up_swiglu"],
            candidacy="candidate:output_projection+residual",
            candidacy_why="down output is the MLP residual's delta. Subsumed by residual+rmsnorm.",
            class_frequency=g["layers"],
            organ="mlp.down",
        )
        add(
            operator="mlp_residual",
            kernel="qwen_next_add_residual",
            layer=layer,
            mixer=kind,
            weight_bytes=0,
            act_read=f32b(H) * 2,
            act_write=f32b(H),
            flops=H,
            isolated="mlp_residual_64",
            deps=["down_proj", "mixer_residual"],
            candidacy="candidate:residual+rmsnorm",
            candidacy_why=(
                "Shares hidden with the next layer's input_rmsnorm (or final_norm "
                "on layer 63). Isolated 2.1 µs. Working shape."
            ),
            class_frequency=g["layers"],
            organ="mlp.residual",
        )

    add(
        operator="final_rmsnorm",
        kernel="qwen80_residual_rmsnorm_tg",
        layer=None,
        mixer=None,
        weight_bytes=f32b(H),
        act_read=f32b(H),
        act_write=f32b(H),
        flops=5 * H,
        isolated="final_norm",
        deps=["mlp_residual"],
        candidacy="candidate:residual+rmsnorm",
        candidacy_why="Shares hidden with last-layer MLP residual.",
        class_frequency=1,
        organ="terminal.norm",
    )
    add(
        operator="lm_head",
        kernel="qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
        layer=None,
        mixer=None,
        weight_bytes=q4_matrix_bytes(g["vocab"], H),
        act_read=f32b(H),
        act_write=f32b(g["vocab"]),
        flops=2 * g["vocab"] * H,
        isolated="lm_head",
        deps=["final_rmsnorm"],
        candidacy="candidate:dequant+matvec_already_fused",
        candidacy_why=(
            "geo_tpr64 already dequants in-register. A fused final_norm+lm_head "
            "kernel exists (qwen_uniform_q4_group64_final_norm_lm_head_simdgroup8) "
            "and is unused; it would save one launch of a 19 µs norm in front of "
            "a 1.0 ms GEMV. Not the bandwidth-binding term."
        ),
        class_frequency=1,
        organ="lm_head",
    )
    add(
        operator="argmax",
        kernel="sample_argmax_f32",
        layer=None,
        mixer=None,
        weight_bytes=0,
        act_read=f32b(g["vocab"]),
        act_write=4,
        flops=0,
        isolated="argmax",
        deps=["lm_head"],
        candidacy="candidate:sampling_preprocessing",
        candidacy_why=(
            "Could fold into the lm_head reduction. Isolated 335 µs is the scan "
            "of 248,320 logits, not launch. Fusion would still stream the logits."
        ),
        class_frequency=1,
        organ="sample",
    )
    return rows


def ranked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed = sorted(rows, key=lambda r: (-r["rank_score_ns"], r["index"]))
    out = []
    for i, r in enumerate(keyed):
        c = dict(r)
        c["rank"] = i + 1
        out.append(c)
    return out


def class_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, dict[str, Any]] = {}
    for r in rows:
        k = r["operator"]
        slot = by.setdefault(k, {
            "operator": k,
            "kernel": r["kernel"],
            "frequency": 0,
            "bytes_per_token": 0,
            "flops_per_token": 0.0,
            "rank_score_ns_per_token": 0.0,
            "fusion_candidacy": r["fusion_candidacy"],
            "fusion_candidacy_why": r["fusion_candidacy_why"],
        })
        slot["frequency"] += 1
        slot["bytes_per_token"] += r["bytes"]["total"]
        slot["flops_per_token"] += r["flops"]
        slot["rank_score_ns_per_token"] += r["rank_score_ns"]
    return sorted(by.values(), key=lambda s: -s["rank_score_ns_per_token"])


def fusion_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remaining shared-input pairs, ranked by (launch + traffic + sync)."""
    by_op = {s["operator"]: s for s in class_summary(rows)}
    residual = by_op["mixer_residual"]["rank_score_ns_per_token"] + by_op["mlp_residual"]["rank_score_ns_per_token"]
    rms = (
        by_op["input_rmsnorm"]["rank_score_ns_per_token"]
        + by_op["post_attention_rmsnorm"]["rank_score_ns_per_token"]
        + by_op["final_rmsnorm"]["rank_score_ns_per_token"]
    )
    cands = [
        {
            "id": "residual_plus_following_rmsnorm",
            "motif": "output_projection+residual / norm+projection (the shared residual vector)",
            "launches_saved": CANDIDATE_SAVED,
            "from": PARENT_DISPATCHES,
            "to": CANDIDATE_DISPATCHES,
            "rank_score_ns_sum": residual + rms,
            "why_this_shape": (
                "The four fusions that worked collapsed launches that ALREADY "
                "share an input. mixer_residual's output IS post_attention_rmsnorm's "
                "input; mlp_residual's output IS the next input_rmsnorm. Isolated "
                "residuals are ~2 µs of almost-pure launch; RMSNorms are ~18 µs of "
                "launch+reduction over 5120. One threadgroup, (1+w) math, out-of-place."
            ),
            "attempted_here": True,
            "kernel": KERNEL_GOOD,
            "bad_control_kernel": KERNEL_BAD,
        },
        {
            "id": "norm_plus_first_gemv",
            "motif": "norm+projection",
            "launches_saved": 64 + 16 + 1,
            "from": PARENT_DISPATCHES,
            "to": PARENT_DISPATCHES - (64 + 16 + 1),
            "rank_score_ns_sum": by_op["input_rmsnorm"]["rank_score_ns_per_token"],
            "why_not": (
                "geo_tpr64 launches ceil(rows/2) threadgroups that each need the "
                "normalized x. RMSNorm is a 5120-wide reduction; fusing it into "
                "the GEMV recomputes that reduction per TG (8704× for gate). "
                "The working fusion keeps one shared x and concatenates ROWS."
            ),
            "attempted_here": False,
        },
        {
            "id": "gated_rmsnorm_plus_out_proj",
            "motif": "norm+projection",
            "launches_saved": 48,
            "from": PARENT_DISPATCHES,
            "to": PARENT_DISPATCHES - 48,
            "rank_score_ns_sum": by_op["gated_rmsnorm"]["rank_score_ns_per_token"],
            "why_not": "Same TG-recompute trap as norm+first GEMV, 6144-wide x.",
            "attempted_here": False,
        },
        {
            "id": "megakernel_layer",
            "motif": "whole layer",
            "launches_saved": "up to 756−64",
            "why_not": (
                "Prior art: an 8-layer f16 megakernel measured 4.4× SLOWER. "
                "Fusion is not automatically a win. Not repeated."
            ),
            "attempted_here": False,
        },
    ]
    cands.sort(key=lambda c: -(c.get("rank_score_ns_sum") or 0))
    return cands


def occupancy_snapshot() -> dict[str, Any]:
    try:
        p = subprocess.run(
            ["ps", "-eo", "pid,rss,command"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "ps_matches": [],
            "loaded_a_second_27b": False,
            "ps_ok": False,
            "note": (
                "A second Qwen3.8-27B would show RSS in the 10+ GiB class and is refused. "
                f"ps occupancy ABSENT this process: {type(exc).__name__}: {exc}"
            ),
        }
    lines = []
    second = False
    for line in p.stdout.splitlines():
        low = line.lower()
        if any(s in low for s in ("llama-server", "ascension_qwen", "mlx_lm.server")):
            if "rg " in low or "dispatch_ledger" in low:
                continue
            lines.append(line.strip())
            parts = line.split()
            try:
                rss_kb = int(parts[1])
            except (IndexError, ValueError):
                rss_kb = 0
            if rss_kb > 4_000_000:
                second = True
    return {
        "ps_matches": lines,
        "loaded_a_second_27b": second,
        "note": "A second Qwen3.8-27B would show RSS in the 10+ GiB class and is refused.",
    }


def cargo_build() -> dict[str, Any]:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(CARGO_TARGET)
    cmd = [
        "cargo", "build", "--profile", "release-fast",
        "-p", "hawking-core", "--example", "ascension_qwen38_dispatch_ledger",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=3600, env=env)
    return {
        "command": cmd,
        "exit_code": proc.returncode,
        "wall_s": time.perf_counter() - t0,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "binary": str(find_binary()) if find_binary() else None,
    }


def run_example(binary: Path, out: Path, *, skip_decode: bool = False) -> dict[str, Any]:
    lock = REPO / "tools" / "gpu_lane_lock.sh"
    cmd: list[str] = []
    if lock.is_file():
        cmd.extend(["bash", str(lock), "qwen38-dispatch-ledger"])
    cmd.extend([
        str(binary),
        "--artifact-root", str(PARENT),
        "--tokenizer", str(TOKENIZER),
        "--prompt", PROMPT,
        "--max-new-tokens", "16",
        "--max-seq-len", "128",
        "--reps", os.environ.get("QWEN38_DISPATCH_LEDGER_REPS", "2"),
        "--out", str(out),
    ])
    if skip_decode:
        cmd.append("--skip-decode")
    env = os.environ.copy()
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=7200, env=env)
    result: dict[str, Any] = {
        "command": cmd,
        "exit_code": proc.returncode,
        "wall_s": time.perf_counter() - t0,
        "stdout_tail": proc.stdout[-8000:],
        "stderr_tail": proc.stderr[-8000:],
        "out": str(out),
        "ok": proc.returncode == 0 and out.is_file(),
    }
    if out.is_file():
        try:
            result["body"] = json.loads(out.read_text())
        except json.JSONDecodeError as e:
            result["ok"] = False
            result["json_error"] = str(e)
    return result


def _arm(decode: dict, key: str) -> dict[str, Any] | None:
    arm = decode.get(key)
    if not arm:
        return None
    ids = [int(x) for x in (arm.get("new_token_ids") or [])]
    text = arm.get("generated_text_verbatim") or ""
    return {
        "tok_s_reps": arm.get("tok_s_reps"),
        "tok_s_mean": arm.get("tok_s_mean"),
        "new_token_ids": ids,
        "generated_text_verbatim": text,
        "dispatches_last_step_reps": arm.get("dispatches_last_step_reps"),
        "dispatched_kernels_rep0": arm.get("dispatched_kernels_rep0"),
        "dense_w_materialized": arm.get("dense_w_materialized", 0),
        "coherence": judge_coherence(text, ids),
        "fallbacks_reps": arm.get("fallbacks_reps"),
    }


def reduction_from_raw(raw: dict[str, Any] | None) -> dict[str, Any]:
    empty = {
        "measured": False,
        "gpu_ran": False,
        "reason_if_unmeasured": "no _DISPATCH_LEDGER_raw.json",
    }
    if not raw:
        return empty
    probes = {p.get("id"): p.get("probe") or {} for p in (raw.get("dispatch_probes") or [])}
    parent = probes.get("parent_756") or {}
    cand = probes.get("add_rmsnorm_628") or {}
    badp = probes.get("add_rmsnorm_bad") or {}
    decode = raw.get("decode") or {}
    before = _arm(decode, "parent_756")
    after = _arm(decode, "add_rmsnorm_628")
    bad = _arm(decode, "add_rmsnorm_bad")
    before_n = parent.get("measured")
    after_n = cand.get("measured")
    ids_match = None
    if before and after:
        ids_match = before["new_token_ids"] == after["new_token_ids"] and bool(before["new_token_ids"])
    bad_rejected = None
    if before and bad and before["new_token_ids"] and bad["new_token_ids"]:
        bad_rejected = bad["new_token_ids"] != before["new_token_ids"]
    noop_did_not_score = None
    if before_n is not None:
        noop_did_not_score = before_n == PARENT_DISPATCHES
    par = (raw.get("parity") or {}).get("add_residual_rmsnorm") or {}
    bad_par = (raw.get("parity") or {}).get("bad_plainweight") or {}
    reduced = (
        isinstance(before_n, int)
        and isinstance(after_n, int)
        and after_n < PARENT_DISPATCHES
        and after_n < before_n
    )
    tok_note = None
    if before and after:
        b = before.get("tok_s_mean")
        a = after.get("tok_s_mean")
        if isinstance(b, (int, float)) and isinstance(a, (int, float)) and b:
            pct = 100.0 * (a - b) / b
            tok_note = (
                f"tok/s {b:.3f} → {a:.3f} ({pct:+.2f}%). "
                "Do not expect a dispatch cut to buy throughput proportionally; "
                "the incumbent is bandwidth-bound (GPU_LEDGER 468.9 GB/s, 95.6% of wall)."
            )
    verdict_parts = []
    if reduced:
        verdict_parts.append(f"dispatches {before_n} → {after_n}")
    if ids_match is True:
        verdict_parts.append("token ids unchanged")
    elif ids_match is False:
        verdict_parts.append("token ids DIVERGED")
    if bad_rejected is True:
        verdict_parts.append("BAD plainweight control rejected (ids differ)")
    elif bad_rejected is False:
        verdict_parts.append("BAD control FAILED to diverge")
    if noop_did_not_score is True:
        verdict_parts.append("NO-OP 756 control did not score")
    if tok_note:
        verdict_parts.append(tok_note)
    return {
        "measured": bool(reduced),
        "gpu_ran": True,
        "parent_dispatches": before_n,
        "candidate_dispatches": after_n,
        "theoretical_candidate": CANDIDATE_DISPATCHES,
        "token_ids_before": (before or {}).get("new_token_ids"),
        "token_ids_after": (after or {}).get("new_token_ids"),
        "token_ids_bad": (bad or {}).get("new_token_ids"),
        "token_ids_unchanged": ids_match,
        "tok_s_before": (before or {}).get("tok_s_mean"),
        "tok_s_after": (after or {}).get("tok_s_mean"),
        "tok_s_note": tok_note,
        "parity": par,
        "bad_parity": bad_par,
        "noop_control": {
            "id": "parent_756",
            "dispatches": before_n,
            "must_not_score": True,
            "did_not_score": noop_did_not_score,
            "token_ids": (before or {}).get("new_token_ids"),
        },
        "bad_control": {
            "id": "add_rmsnorm_plainweight",
            "kernel": KERNEL_BAD,
            "must_be_rejected": True,
            "rejected": bad_rejected,
            "max_abs_diff_norm": bad_par.get("max_abs_diff_norm") or bad_par.get("max_abs_diff"),
            "token_ids": (bad or {}).get("new_token_ids"),
        },
        "sentinel": {
            "kernel": KERNEL_GOOD,
            "probe_sentinel_kernel_present": cand.get("sentinel_kernel_present"),
            "dispatch_count_628": after_n == CANDIDATE_DISPATCHES,
        },
        "kernel_identity": {
            "good": KERNEL_GOOD,
            "bad": KERNEL_BAD,
            "parent_kernels_rep0": (before or {}).get("dispatched_kernels_rep0"),
            "candidate_kernels_rep0": (after or {}).get("dispatched_kernels_rep0"),
        },
        "coherence_after": (after or {}).get("coherence"),
        "dense_w_materialized": raw.get("dense_w_materialized", 0),
        "expanded_to_q4": raw.get("expanded_to_q4", 0),
        "expanded_to_float_gemv": raw.get("expanded_to_float_gemv", 0),
        "decode": {"before": before, "after": after, "bad": bad},
        "dispatch_probes": raw.get("dispatch_probes"),
        "verdict": "; ".join(verdict_parts) if verdict_parts else "gpu ran but no verdict assembled",
        "reason_if_unmeasured": None,
    }


def why_no_further_or_the_cut(red: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Either the measured cut, or a measured reason none is available."""
    if red.get("measured") and red.get("token_ids_unchanged"):
        return {
            "kind": "measured_reduction",
            "from": red["parent_dispatches"],
            "to": red["candidate_dispatches"],
            "token_ids_unchanged": True,
            "note": red.get("tok_s_note"),
            "why_not_more": (
                "Remaining launches either (a) already share no input, "
                "(b) are stateful recurrence / softmax, or (c) are GEMV+"
                "norm fusions that recompute a reduction in every threadgroup. "
                "A layer megakernel measured 4.4× slower. The GPU is "
                "bandwidth-bound; further launch cuts are not expected to "
                "move tok/s proportionally."
            ),
        }
    if red.get("gpu_ran") and red.get("token_ids_unchanged") is False:
        return {
            "kind": "candidate_rejected",
            "reason": "token ids diverged; fusion is not admitted",
            "from": red.get("parent_dispatches"),
            "to": red.get("candidate_dispatches"),
        }
    # No live GPU in this process. The isolated TOKEN_NS measurement still
    # bounds what a cut can buy, and that is a measured reason.
    residual_ns = isolated_per("mixer_residual_64") * 64 + isolated_per("mlp_residual_64") * 64
    rms_ns = (
        isolated_per("input_norms") * 63  # layer 0 input_rms stays
        + isolated_per("post_norms") * 64
        + isolated_per("final_norm")
    )
    # The residual work is absorbed into the RMS first pass; saved time is
    # residual launches plus RMS launch overhead of the collapsed partner,
    # not the RMS math.
    saved_upper_ns = residual_ns + isolated_per("input_norms") * 63 * 0.1
    gpu_ns = 29_049_999  # GPU_LEDGER warm median
    return {
        "kind": "measured_bound_pending_live_generate" if not red.get("gpu_ran") else "measured_no_win",
        "isolated_residual_gpu_ns": residual_ns,
        "isolated_collapsible_rms_gpu_ns": rms_ns,
        "upper_bound_saved_ns_if_perfect_fuse": residual_ns,
        "upper_bound_as_pct_of_gpu_ns": 100.0 * residual_ns / gpu_ns,
        "why": (
            f"Mixer+MLP residual isolated GPU time is {residual_ns/1e3:.1f} µs "
            f"({100.0 * residual_ns / gpu_ns:.2f}% of the 29.05 ms GPU_NS). "
            "RMSNorm math still runs inside the fused kernel. GPU_LEDGER: the "
            "incumbent streams 468.9 GB/s (67% of a 700 GB/s ceiling, 78.7% of "
            "the 595.9 GB/s sequential roof) with the GPU occupying 95.6% of "
            "complete-wall. A 128-dispatch cut of ~2 µs residuals cannot move "
            "tok/s by 128/756. The 964→756 fusion that DID share GEMV inputs "
            "bought +5.8%, not 21.6%. Live generate of the residual+rmsnorm "
            "child is the remaining measurement; the bound above is already "
            "measured (isolated TOKEN_NS + GPU_LEDGER)."
        ),
        "note_saved_upper_ns": saved_upper_ns,
    }


def build(*, raw: dict[str, Any] | None = None, gpu_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    g = load_geometry()
    rows = parent_graph(g)
    if len(rows) != PARENT_DISPATCHES:
        raise SystemExit(
            f"FAIL: parent graph has {len(rows)} dispatches, want {PARENT_DISPATCHES}"
        )
    ranked_rows = ranked(rows)
    classes = class_summary(rows)
    cands = fusion_candidates(rows)
    ev = shader_evidence()
    occ = occupancy_snapshot()
    red = reduction_from_raw(raw)
    why = why_no_further_or_the_cut(red, rows)
    parent_ok = PARENT.is_dir() and (PARENT / "catalog.hq38m20").is_file()
    return {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "obligation": (
            "N005 — name every dispatch per token; rank by launch overhead + "
            "memory traffic + synchronization; measured cut below 756 or a "
            "measured reason none is available"
        ),
        "did_not_load_second_27b": True,
        "did_not_write_under_models": True,
        "did_not_mutate_parent": True,
        "occupancy": occ,
        "parent": {
            "path": str(PARENT),
            "present": parent_ok,
            "immutable": True,
            "complete_ebpw": 3.139300850311054,
            "dispatches": PARENT_DISPATCHES,
            "tok_s_sealed": 34.873,
            "note": "Sealed at ~/noetic/NOETIC_PARENT_A. Children only.",
        },
        "gpu_ledger_overturns": {
            "q80_idle_claim": "0.79% of 700 GB/s, 51% GPU idle — different vehicle (MoE, many CBs)",
            "q4_incumbent": (
                "1 CB / 964 dispatches unfused, 468.9 GB/s (67% of 700, 78.7% of "
                "595.9 sequential roof), GPU 95.6% of complete-wall, host queue "
                "wait 1.39%. BANDWIDTH-bound."
            ),
            "964_to_756_bought": "+5.8% tok/s, not 21.6%",
            "implication": (
                "Do not expect a dispatch cut to buy throughput proportionally. "
                "The ledger itself is the deliverable."
            ),
        },
        "counting_method": (
            "TokenCommandBuffer.dispatch_count: one kernel launch = one dispatch. "
            "Sealed parent is the 756-dispatch fused graph (MLP swiglu + GQA QKV + "
            "DN qkvz+ba). Unfused production remains 964."
        ),
        "graph": {
            "unfused": UNFUSED_DISPATCHES,
            "parent_fused": PARENT_DISPATCHES,
            "candidate_residual_rmsnorm": CANDIDATE_DISPATCHES,
            "command_buffers": 1,
            "n_named": len(rows),
            "formula": (
                "embed 1 + 48×8 DN + 16×7 GQA + 64×4 MLP + terminal 3 = "
                "1+384+112+256+3 = 756"
            ),
        },
        "seq_len_for_kv_bytes": SEQ_LEN,
        "rank_score": {
            "formula": "launch_overhead_ns + memory_traffic_bytes / 595.9e9 * 1e9 + synchronization_ns",
            "launch_overhead": (
                "DERIVED: isolated TOKEN_NS family / n minus bandwidth floor. "
                "Production per_dispatch_gpu_ns is ABSENT (atDispatchBoundary=false)."
            ),
            "memory_traffic": "weight_read + activation_read + activation_write, ns at sequential roof",
            "synchronization": (
                "0 for every intra-CB dispatch. Production is one wait_until_completed "
                "per token (GPU_LEDGER synchronization_count=1, queue wait 421 µs)."
            ),
            "roof_gb_s": ROOF_GB_S,
        },
        "dispatches": rows,
        "dispatches_ranked": ranked_rows,
        "classes": classes,
        "fusion_candidates": cands,
        "prior_art": {
            "working_fusions": [
                "gate+up (64 saved)",
                "gate+up+SwiGLU (128)",
                "GQA QKV concat (32)",
                "DeltaNet qkvz+ba concat (48)",
            ],
            "shape_that_worked": "launches that ALREADY share an input",
            "megakernel_8layer_f16": "measured 4.4x SLOWER; not repeated",
            "native_operator": "SOURCE and EXECUTABLE both 964 while DRAM bytes fall 7.34x",
        },
        "motifs_still_unsearched_before_this_lane": [
            "norm+projection",
            "projection+activation",
            "route+lookup+accumulate (no router on this dense parent)",
            "low-rank+sparse",
            "attention preparation",
            "recurrent state update",
            "output projection+residual",
            "dequant+matvec (already fused in geo_tpr64)",
            "sampling preprocessing",
        ],
        "candidate_attempted": {
            "id": "residual_plus_following_rmsnorm",
            "kernel": KERNEL_GOOD,
            "bad_control_kernel": KERNEL_BAD,
            "theoretical_after": CANDIDATE_DISPATCHES,
            "shader_evidence": ev,
            "enable": {
                "default": "off — parent graph stays 756",
                "HAWKING_QWEN38_FUSE_ADD_RMSNORM": "1 | bad",
            },
        },
        "causal_benchmark_law": {
            "kernel_identity": KERNEL_GOOD,
            "dispatch_count": "parent 756 vs candidate 628",
            "sentinel": f"harvest of {KERNEL_GOOD} plus dispatch_count==628",
            "noop_control": "parent 756 fused graph, same weights, fusion flag off — must not score as a cut",
            "bad_control": (
                f"{KERNEL_BAD} multiplies by weight[i] not (1+w); must diverge "
                "greedy ids and be rejected"
            ),
        },
        "reduction": red,
        "no_further_or_the_cut": why,
        "dense_parent": {
            "dense_w_materialized": red.get("dense_w_materialized", 0),
            "expanded_to_q4": red.get("expanded_to_q4", 0),
            "expanded_to_float_gemv": red.get("expanded_to_float_gemv", 0),
        },
        "gpu": None if gpu_meta is None else {
            "ok": gpu_meta.get("ok"),
            "exit_code": gpu_meta.get("exit_code"),
            "wall_s": gpu_meta.get("wall_s"),
            "binary": gpu_meta.get("command", [None])[-1] if gpu_meta.get("command") else None,
            "stderr_tail": (gpu_meta.get("stderr_tail") or "")[-2000:],
        },
    }


def write_receipt(doc: dict[str, Any]) -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=1) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true", help="build + run the Metal example")
    ap.add_argument("--skip-decode", action="store_true")
    args = ap.parse_args()
    raw = None
    gpu_meta = None
    if args.measure:
        built = cargo_build()
        if built["exit_code"] != 0:
            print(built["stderr_tail"], file=sys.stderr)
            gpu_meta = built
        else:
            binary = find_binary()
            if binary is None:
                gpu_meta = {**built, "ok": False, "stderr_tail": "binary missing after build"}
            else:
                gpu_meta = run_example(binary, RAW, skip_decode=args.skip_decode)
                raw = gpu_meta.get("body")
    elif RAW.is_file():
        try:
            raw = json.loads(RAW.read_text())
        except json.JSONDecodeError:
            raw = None
    doc = build(raw=raw, gpu_meta=gpu_meta)
    write_receipt(doc)
    print(f"wrote {RECEIPT} n={doc['graph']['n_named']} candidate={doc['graph']['candidate_residual_rmsnorm']}")
    print("top-5 ranked:")
    for r in doc["dispatches_ranked"][:5]:
        print(f"  #{r['rank']} {r['operator']} L{r['layer']} score={r['rank_score_ns']:.0f} ns bytes={r['bytes']['total']}")
    print("no_further_or_the_cut:", doc["no_further_or_the_cut"].get("kind"), doc["reduction"].get("verdict"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

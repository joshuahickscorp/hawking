#!/usr/bin/env python3
"""Qwen3.8 descent-below-2.0 codec screen on REAL activations.

CPU-only. Does not pack a competing mixed-2p0 artifact (sibling lane owns that).
Does not use the GPU. Scores cheap-to-reconstruct codecs against the captured
BF16 post-norm hidden states, plus post-SwiGLU X built from those hiddens and
the real gate/up weights.

Physical BPW is 8 * payload_bytes / elements (container + scales + codes).
Output cosine is holdout: y = X @ W.T vs X @ W_hat.T on tokens never used as
a codec parameter (the cheap codecs here are weight-only; the split still
guards against over-reading a 256-token capture).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators.ascension_dual_gravity_worker import (  # noqa: E402
    GROUP_BINARY,
    GROUP_UNIFORM,
    _additive_residual_codec,
    _binary_codec,
    _hadamard_lattice_codec,
    _ternary_codec,
    _uniform_codec,
)
from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map  # noqa: E402
from lab.operators.residual_compact_codec import (  # noqa: E402
    decode_residual_compact,
    encode_residual_compact,
)
from lab.receipts import seal  # noqa: E402

MAIN_HAWKING = Path("/Users/scammermike/Downloads/hawking")
MODEL_DIR = MAIN_HAWKING / "workspace/campaign/records/runs/qwen38-27b/bf16"
CAPTURE_DIR = MAIN_HAWKING / "workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
DEFAULT_OUT = REPO_ROOT / "receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json"

SCHEMA = "hawking.special_unit.qwen38_bpw_descent.v1"
MEASURED_MS = 33.537
CURRENT_BPW = 4.252735126866492
F_MLP = 0.6363
F_ATTN = 0.2692
F_EMB = 0.0945
N_TOKENS = 256
HIDDEN = 5120
FIT_N = 192
HOLD_N = 64
Q80_BAR = 0.8604
TIGHT_BAR = 0.95
MODERATE_BAR = 0.90

# Q80 mixed reconstruction slowdown-per-byte, measured on the mixed vehicle.
Q80_MIXED_SLOWDOWN_PER_BYTE = 5.9

LAYERS = (0, 3, 15, 31, 47, 63)


def silu(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    num = float(left @ right)
    den = float(np.linalg.norm(left) * np.linalg.norm(right))
    if den <= 1e-12:
        return 1.0 if num == 0.0 else 0.0
    return num / den


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    ref = np.asarray(a, dtype=np.float64).reshape(-1)
    hat = np.asarray(b, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(ref))
    if n <= 1e-12:
        return 0.0
    return float(np.linalg.norm(ref - hat) / n)


def load_hidden(layer: int) -> np.ndarray:
    path = CAPTURE_DIR / "hidden" / f"L{layer:02d}.f32"
    x = np.fromfile(path, dtype=np.float32)
    if x.size != N_TOKENS * HIDDEN:
        raise RuntimeError(f"{path} has {x.size} floats, expected {N_TOKENS * HIDDEN}")
    return np.ascontiguousarray(x.reshape(N_TOKENS, HIDDEN), dtype=np.float32)


def is_gqa(layer: int) -> bool:
    return int(layer) % 4 == 3


def project_ms(bpw: float, recon_penalty: float = 1.0) -> float:
    return MEASURED_MS * (float(bpw) / CURRENT_BPW) * float(recon_penalty)


def project_tps(ms: float) -> float:
    return 1000.0 / ms if ms > 0 else 0.0


def _rice(values: np.ndarray) -> tuple[bytes, np.ndarray]:
    result = encode_residual_compact(
        values,
        outlier_ratio=0.02,
        group_size=GROUP_BINARY,
        index_mode="rice",
        value_bits=1,
        value_scale="rms",
    )
    return result.payload, decode_residual_compact(result.payload)


CODECS: list[dict[str, Any]] = [
    {
        "name": "uniform_q4_g64",
        "cost_class": "CHEAP_INREGISTER",
        "recon_penalty": 1.0,
        "note": "Incumbent family. Hawking Qwen3.8 q4 path is 406 GB/s.",
        "encode": lambda W: (
            (r := _uniform_codec(W, bits=4, group_size=GROUP_UNIFORM)).payload,
            r.reconstruction,
        ),
    },
    {
        "name": "uniform_q3_g64",
        "cost_class": "CHEAP_INREGISTER",
        "recon_penalty": 1.0,
        "note": "Same affine dequant as q4; one fewer bit.",
        "encode": lambda W: (
            (r := _uniform_codec(W, bits=3, group_size=GROUP_UNIFORM)).payload,
            r.reconstruction,
        ),
    },
    {
        "name": "uniform_q2_g64",
        "cost_class": "CHEAP_INREGISTER",
        "recon_penalty": 1.0,
        "note": "Same affine dequant. MLX isolated eval on 5120x17408 was NOT faster than q4 (289 vs 276 us host-wall) — kernel must stay bandwidth-bound.",
        "encode": lambda W: (
            (r := _uniform_codec(W, bits=2, group_size=GROUP_UNIFORM)).payload,
            r.reconstruction,
        ),
    },
    {
        "name": "binary_g128",
        "cost_class": "CHEAP_INREGISTER",
        "recon_penalty": 1.0,
        "note": "Sign * group mean-abs. Decode is one AND + FMA. Cheapest reconstruct.",
        "encode": lambda W: (
            (r := _binary_codec(W, group_size=GROUP_BINARY)).payload,
            r.reconstruction,
        ),
    },
    {
        "name": "ternary_t0.7_g128",
        "cost_class": "CHEAP_TERNARY",
        "recon_penalty": 1.05,
        "note": "2-bit codes + scale; threshold stored but decode is still in-register.",
        "encode": lambda W: (
            (r := _ternary_codec(W, threshold_multiplier=0.7, group_size=GROUP_BINARY)).payload,
            r.reconstruction,
        ),
    },
    {
        "name": "hadamard_q2_g128",
        "cost_class": "MEDIUM_TRANSFORM",
        "recon_penalty": 1.15,
        "note": "Affine q2 in Walsh-Hadamard domain. Extra butterfly per group.",
        "encode": lambda W: (
            (r := _hadamard_lattice_codec(W, bits=2, group_size=GROUP_BINARY)).payload,
            r.reconstruction,
        ),
    },
    {
        "name": "additive_q2q2_g64",
        "cost_class": "MEDIUM_TWO_STAGE",
        "recon_penalty": 1.20,
        "note": "Two q2 codebooks. ~4-bit payload, two lookups.",
        "encode": lambda W: (
            (r := _additive_residual_codec(W, group_size=GROUP_UNIFORM)).payload,
            r.reconstruction,
        ),
    },
    {
        "name": "rice_q1_rms_2pct",
        "cost_class": "EXPENSIVE_SPARSE",
        "recon_penalty": Q80_MIXED_SLOWDOWN_PER_BYTE,
        "note": "Sibling up_proj recipe. Rice bitstream + scatter. Q80 paid 5.9x / byte.",
        "encode": _rice,
    },
]


def score_pair(
    W: np.ndarray,
    X_fit: np.ndarray,
    X_hold: np.ndarray,
    Y_fit_ref: np.ndarray,
    Y_hold_ref: np.ndarray,
    codec: dict[str, Any],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    payload, W_hat = codec["encode"](W)
    encode_s = time.perf_counter() - t0
    elements = int(W.size)
    bpw = 8.0 * len(payload) / elements
    weight_cos = cosine(W, W_hat)
    weight_rel = rel_l2(W, W_hat)
    t1 = time.perf_counter()
    Y_fit = X_fit @ W_hat.T
    Y_hold = X_hold @ W_hat.T
    gemm_s = time.perf_counter() - t1
    return {
        "codec": codec["name"],
        "cost_class": codec["cost_class"],
        "recon_penalty_assumed": codec["recon_penalty"],
        "payload_bytes": int(len(payload)),
        "elements": elements,
        "physical_bpw": float(bpw),
        "weight_cosine": float(weight_cos),
        "weight_rel_l2": float(weight_rel),
        "fit_output_cosine": float(cosine(Y_fit_ref, Y_fit)),
        "hold_output_cosine": float(cosine(Y_hold_ref, Y_hold)),
        "fit_output_rel_l2": float(rel_l2(Y_fit_ref, Y_fit)),
        "hold_output_rel_l2": float(rel_l2(Y_hold_ref, Y_hold)),
        "clears_q80_bar": bool(cosine(Y_hold_ref, Y_hold) >= Q80_BAR),
        "clears_moderate": bool(cosine(Y_hold_ref, Y_hold) >= MODERATE_BAR),
        "clears_tight": bool(cosine(Y_hold_ref, Y_hold) >= TIGHT_BAR),
        "encode_s": float(encode_s),
        "gemm_s": float(gemm_s),
    }


def score_weight_only(W: np.ndarray, codec: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    payload, W_hat = codec["encode"](W)
    encode_s = time.perf_counter() - t0
    elements = int(W.size)
    bpw = 8.0 * len(payload) / elements
    wcos = cosine(W, W_hat)
    return {
        "codec": codec["name"],
        "cost_class": codec["cost_class"],
        "recon_penalty_assumed": codec["recon_penalty"],
        "payload_bytes": int(len(payload)),
        "elements": elements,
        "physical_bpw": float(bpw),
        "weight_cosine": float(wcos),
        "weight_rel_l2": float(rel_l2(W, W_hat)),
        "fit_output_cosine": None,
        "hold_output_cosine": None,
        "fit_output_rel_l2": None,
        "hold_output_rel_l2": None,
        "clears_q80_bar": None,
        "clears_moderate": None,
        "clears_tight": None,
        "encode_s": float(encode_s),
        "gemm_s": 0.0,
        "quality_space": "weight_only",
    }


def organ_rows(
    layer: int, role: str, W: np.ndarray, X: np.ndarray | None
) -> dict[str, Any]:
    output_space = (
        X is not None and X.ndim == 2 and X.shape[1] == W.shape[1]
    )
    rows = []
    if output_space:
        assert X is not None
        X_fit, X_hold = X[:FIT_N], X[FIT_N : FIT_N + HOLD_N]
        Y_fit_ref = X_fit @ W.T
        Y_hold_ref = X_hold @ W.T
    else:
        X_fit = X_hold = Y_fit_ref = Y_hold_ref = None
    for codec in CODECS:
        if output_space:
            row = score_pair(W, X_fit, X_hold, Y_fit_ref, Y_hold_ref, codec)
            row["quality_space"] = "output"
        else:
            row = score_weight_only(W, codec)
        row["layer"] = int(layer)
        row["role"] = role
        row["W_shape"] = [int(W.shape[0]), int(W.shape[1])]
        row["n_fit"] = FIT_N if output_space else 0
        row["n_hold"] = HOLD_N if output_space else 0
        rows.append(row)
        hold = row["hold_output_cosine"]
        hold_s = f"{hold:.4f}" if isinstance(hold, float) else "weight-only"
        print(
            f"  L{layer:02d} {role:16s} {codec['name']:22s}  "
            f"bpw={row['physical_bpw']:.4f}  hold={hold_s}  "
            f"{codec['cost_class']}",
            flush=True,
        )
    return {
        "layer": int(layer),
        "role": role,
        "W_shape": [int(W.shape[0]), int(W.shape[1])],
        "quality_space": "output" if output_space else "weight_only",
        "candidates": rows,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, dict[str, Any]] = {}
    for organ in rows:
        role = organ["role"]
        for c in organ["candidates"]:
            slot = by.setdefault(
                f"{role}|{c['codec']}",
                {
                    "role": role,
                    "codec": c["codec"],
                    "cost_class": c["cost_class"],
                    "recon_penalty_assumed": c["recon_penalty_assumed"],
                    "hold": [],
                    "fit": [],
                    "bpw": [],
                    "weight_cos": [],
                },
            )
            if c["hold_output_cosine"] is not None:
                slot["hold"].append(c["hold_output_cosine"])
                slot["fit"].append(c["fit_output_cosine"])
            slot["bpw"].append(c["physical_bpw"])
            slot["weight_cos"].append(c["weight_cosine"])
    out = []
    for slot in by.values():
        hold = np.asarray(slot["hold"], dtype=np.float64) if slot["hold"] else None
        rec: dict[str, Any] = {
            "role": slot["role"],
            "codec": slot["codec"],
            "cost_class": slot["cost_class"],
            "recon_penalty_assumed": slot["recon_penalty_assumed"],
            "n_bpw": int(len(slot["bpw"])),
            "n_hold": int(0 if hold is None else hold.size),
            "physical_bpw_mean": float(np.mean(slot["bpw"])),
            "physical_bpw_min": float(np.min(slot["bpw"])),
            "physical_bpw_max": float(np.max(slot["bpw"])),
            "weight_cosine_mean": float(np.mean(slot["weight_cos"])),
            "weight_cosine_min": float(np.min(slot["weight_cos"])),
        }
        if hold is not None and hold.size:
            rec.update(
                {
                    "hold_mean": float(np.mean(hold)),
                    "hold_min": float(np.min(hold)),
                    "hold_p10": float(np.quantile(hold, 0.10)),
                    "hold_max": float(np.max(hold)),
                    "fit_mean": float(np.mean(slot["fit"])),
                    "frac_clears_q80_bar": float(np.mean(hold >= Q80_BAR)),
                    "frac_clears_moderate": float(np.mean(hold >= MODERATE_BAR)),
                    "frac_clears_tight": float(np.mean(hold >= TIGHT_BAR)),
                }
            )
        else:
            rec.update(
                {
                    "hold_mean": None,
                    "hold_min": None,
                    "quality_space": "weight_only",
                    "note": "out_proj in-dim is 6144 (value width), not captured hidden 5120",
                }
            )
        out.append(rec)
    out.sort(key=lambda r: (r["role"], r["physical_bpw_mean"]))
    return {"by_role_codec": out}


def pick(summary: list[dict[str, Any]], role: str, codec: str) -> dict[str, Any]:
    for row in summary:
        if row["role"] == role and row["codec"] == codec:
            return row
    raise KeyError(f"{role} {codec}")


def mean_roles(summary: list[dict[str, Any]], roles: tuple[str, ...], codec: str, field: str) -> float:
    vals = [pick(summary, role, codec)[field] for role in roles]
    return float(np.mean(vals))


def recipe_row(
    *,
    name: str,
    mlp_bpw: float,
    attn_bpw: float,
    emb_bpw: float,
    cost_class: str,
    recon_penalty: float,
    quality: str,
    verdict: str,
    note: str,
) -> dict[str, Any]:
    complete = F_MLP * mlp_bpw + F_ATTN * attn_bpw + F_EMB * emb_bpw
    ms_invar = project_ms(complete, 1.0)
    ms_adj = project_ms(complete, recon_penalty)
    return {
        "codec": name,
        "mlp_bpw": round(mlp_bpw, 4),
        "attn_bpw": round(attn_bpw, 4),
        "emb_bpw": round(emb_bpw, 4),
        "projected_bpw": round(complete, 4),
        "reconstruction_cost_class": cost_class,
        "recon_penalty": recon_penalty,
        "projected_ms_token_invariant": round(ms_invar, 3),
        "projected_tps_invariant": round(project_tps(ms_invar), 2),
        "projected_ms_token_cost_adjusted": round(ms_adj, 3),
        "projected_tps_cost_adjusted": round(project_tps(ms_adj), 2),
        "clears_2p0": bool(complete <= 2.0),
        "clears_50tps_invariant": bool(project_tps(ms_invar) >= 50.0),
        "clears_50tps_adjusted": bool(project_tps(ms_adj) >= 50.0),
        "quality_evidence": quality,
        "verdict": verdict,
        "note": note,
    }


def build_recipes(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mlp_roles = ("gate_proj", "up_proj", "down_proj")
    attn_bpw_roles = ("attn_in", "attn_out")
    attn_hold_roles = ("attn_in",)

    def mlp(codec: str) -> float:
        return mean_roles(summary, mlp_roles, codec, "physical_bpw_mean")

    def attn(codec: str) -> float:
        return mean_roles(summary, attn_bpw_roles, codec, "physical_bpw_mean")

    def mlp_hold(codec: str) -> tuple[float, float]:
        holds = [pick(summary, r, codec)["hold_min"] for r in mlp_roles]
        means = [pick(summary, r, codec)["hold_mean"] for r in mlp_roles]
        return float(np.mean(means)), float(np.min(holds))

    def attn_hold(codec: str) -> tuple[float, float]:
        holds = [pick(summary, r, codec)["hold_min"] for r in attn_hold_roles]
        means = [pick(summary, r, codec)["hold_mean"] for r in attn_hold_roles]
        return float(np.mean(means)), float(np.min(holds))

    q4_mlp_h, q4_mlp_min = mlp_hold("uniform_q4_g64")
    q3_mlp_h, q3_mlp_min = mlp_hold("uniform_q3_g64")
    q2_mlp_h, q2_mlp_min = mlp_hold("uniform_q2_g64")
    bin_mlp_h, bin_mlp_min = mlp_hold("binary_g128")
    q2_attn_h, q2_attn_min = attn_hold("uniform_q2_g64")
    q3_attn_h, q3_attn_min = attn_hold("uniform_q3_g64")
    bin_attn_h, bin_attn_min = attn_hold("binary_g128")
    rice_up = pick(summary, "up_proj", "rice_q1_rms_2pct")
    bin_gate = pick(summary, "gate_proj", "binary_g128")
    bin_down = pick(summary, "down_proj", "binary_g128")
    q2_down = pick(summary, "down_proj", "uniform_q2_g64")
    q3_down = pick(summary, "down_proj", "uniform_q3_g64")

    # Sibling L00 measured (partial pack): do not re-derive, cite.
    sib_gate, sib_up, sib_down = 1.1250234267290902, 1.287504487879136, 0.13161719827090992
    sib_mlp = (sib_gate + sib_up + sib_down) / 3.0
    sib_complete_if_global = F_MLP * sib_mlp + (F_ATTN + F_EMB) * 4.250011256679038
    # Penalty if rice (up only) inherits Q80 5.9x and the rest stays cheap.
    sib_up_byte_frac = (F_MLP * sib_up) / sib_complete_if_global
    sib_penalty_rice_only = sib_up_byte_frac * Q80_MIXED_SLOWDOWN_PER_BYTE + (1.0 - sib_up_byte_frac) * 1.0

    q4_emb = pick(summary, "gate_proj", "uniform_q4_g64")["physical_bpw_mean"]  # same codec family
    # embed not swept in output-space; use measured q4 physical from incumbent (~4.25)
    emb_q4 = 4.250011256679038
    emb_q3 = pick(summary, "gate_proj", "uniform_q3_g64")["physical_bpw_mean"]
    emb_q2 = pick(summary, "gate_proj", "uniform_q2_g64")["physical_bpw_mean"]

    recipes = [
        recipe_row(
            name="incumbent_uniform_q4_all",
            mlp_bpw=mlp("uniform_q4_g64"),
            attn_bpw=attn("uniform_q4_g64"),
            emb_bpw=emb_q4,
            cost_class="CHEAP_INREGISTER",
            recon_penalty=1.0,
            quality=(
                f"Q4 hold mean MLP {q4_mlp_h:.4f} min {q4_mlp_min:.4f}; "
                f"bring-up generate coherent, q4_min_cosine_vs_bf16=0.98948"
            ),
            verdict="BASELINE — 4.25 BPW, 29.8 TPS measured. Bring-up only; fails G016 2.0.",
            note="Measured 33.537 ms / 406.2 GB/s. Not a descent candidate.",
        ),
        recipe_row(
            name="REF_sibling_q80_recipe_binary_rice_lowrank_q4rest",
            mlp_bpw=sib_mlp,
            attn_bpw=4.250011256679038,
            emb_bpw=emb_q4,
            cost_class="EXPENSIVE_SPARSE+LOWRANK",
            recon_penalty=round(sib_penalty_rice_only, 3),
            quality=(
                "Sibling L00 pack: gate 1.125 / up 1.288 / down 0.132, "
                "mean_component_cosine 0.9289 on 17 tensors. Generation gate not yet closed. "
                "This lane did not re-pack."
            ),
            verdict=(
                "REFERENCE ONLY (sibling owns the pack). Bytes land near 2.09 if L00 "
                "ratios hold — may MISS 2.0 unless later layers or attention move. "
                "Cost-adjusted TPS can regress vs Q4 if rice scatter inherits Q80's 5.9x."
            ),
            note=(
                "down_proj rank-160 on 17408-wide is 0.13 BPW. Consume as L@(R@x), "
                "never reconstruct W. Rice on up is the reconstruction risk."
            ),
        ),
        recipe_row(
            name="cheap_binary_mlp_q4_rest",
            mlp_bpw=mlp("binary_g128"),
            attn_bpw=attn("uniform_q4_g64"),
            emb_bpw=emb_q4,
            cost_class="CHEAP_INREGISTER",
            recon_penalty=1.0,
            quality=f"binary MLP hold mean {bin_mlp_h:.4f} min {bin_mlp_min:.4f}",
            verdict="ABOVE 2.0. Cheap reconstruct. Only pursue if binary MLP hold stays useful.",
            note="Cannot break 2.0 without compressing attention or embed.",
        ),
        recipe_row(
            name="cheap_binary_mlp_q3_attn_q4_emb",
            mlp_bpw=mlp("binary_g128"),
            attn_bpw=attn("uniform_q3_g64"),
            emb_bpw=emb_q4,
            cost_class="CHEAP_INREGISTER",
            recon_penalty=1.0,
            quality=(
                f"binary MLP hold mean {bin_mlp_h:.4f} min {bin_mlp_min:.4f}; "
                f"q3 attn hold mean {q3_attn_h:.4f} min {q3_attn_min:.4f}"
            ),
            verdict="Primary cheap path at the 2.0 line. Wins iff binary MLP and q3 attn both stay coherent.",
            note="All in-register affine/sign-scale. No rice, no SVD.",
        ),
        recipe_row(
            name="cheap_binary_mlp_q2_attn_q4_emb",
            mlp_bpw=mlp("binary_g128"),
            attn_bpw=attn("uniform_q2_g64"),
            emb_bpw=emb_q4,
            cost_class="CHEAP_INREGISTER",
            recon_penalty=1.0,
            quality=(
                f"binary MLP hold mean {bin_mlp_h:.4f} min {bin_mlp_min:.4f}; "
                f"q2 attn hold mean {q2_attn_h:.4f} min {q2_attn_min:.4f}"
            ),
            verdict="Best cheap density if q2 attention holds. Below 2.0 on arithmetic.",
            note="Attention-density sibling owns a fuller attention census; this is the MLP-led recipe that needs their organ.",
        ),
        recipe_row(
            name="cheap_q2_mlp_q4_rest",
            mlp_bpw=mlp("uniform_q2_g64"),
            attn_bpw=attn("uniform_q4_g64"),
            emb_bpw=emb_q4,
            cost_class="CHEAP_INREGISTER",
            recon_penalty=1.0,
            quality=f"q2 MLP hold mean {q2_mlp_h:.4f} min {q2_mlp_min:.4f}",
            verdict="Above 2.0 and likely misses 50 TPS. Safer quality than binary, worse density.",
            note="q2 alone on 63.6% mass cannot break 2.0.",
        ),
        recipe_row(
            name="cheap_q2_all_except_emb_q4",
            mlp_bpw=mlp("uniform_q2_g64"),
            attn_bpw=attn("uniform_q2_g64"),
            emb_bpw=emb_q4,
            cost_class="CHEAP_INREGISTER",
            recon_penalty=1.0,
            quality=(
                f"q2 MLP hold mean {q2_mlp_h:.4f} min {q2_mlp_min:.4f}; "
                f"q2 attn hold mean {q2_attn_h:.4f} min {q2_attn_min:.4f}"
            ),
            verdict="Still above 2.0. Uniform-q2 floor is ~2.25 BPW; cannot reach 2.0 without a 1-bit organ.",
            note="MLX isolated q2 on this shape was not faster than q4 (host-wall).",
        ),
        recipe_row(
            name="cheap_q3_all",
            mlp_bpw=mlp("uniform_q3_g64"),
            attn_bpw=attn("uniform_q3_g64"),
            emb_bpw=emb_q3,
            cost_class="CHEAP_INREGISTER",
            recon_penalty=1.0,
            quality=f"q3 MLP hold mean {q3_mlp_h:.4f} min {q3_mlp_min:.4f}",
            verdict="Quality-preserving cheap step, but ~3.25 BPW misses 50 TPS (42 TPS invariant).",
            note="Useful as a sanity rung, not a G016 closer.",
        ),
        recipe_row(
            name="cheap_binary_all_except_emb_q4",
            mlp_bpw=mlp("binary_g128"),
            attn_bpw=attn("binary_g128"),
            emb_bpw=emb_q4,
            cost_class="CHEAP_INREGISTER",
            recon_penalty=1.0,
            quality=(
                f"binary MLP hold mean {bin_mlp_h:.4f} min {bin_mlp_min:.4f}; "
                f"binary attn hold mean {bin_attn_h:.4f} min {bin_attn_min:.4f}"
            ),
            verdict="Deepest cheap descent (~1.4 BPW). Q30 died at <=1.5. Only real if binary attention holds.",
            note="Density Law: lower BPW is better ONLY while the model remains useful.",
        ),
        recipe_row(
            name="cheap_binary_gate_up_q2_down_q2_attn_q4_emb",
            mlp_bpw=(
                pick(summary, "gate_proj", "binary_g128")["physical_bpw_mean"]
                + pick(summary, "up_proj", "binary_g128")["physical_bpw_mean"]
                + pick(summary, "down_proj", "uniform_q2_g64")["physical_bpw_mean"]
            )
            / 3.0,
            attn_bpw=attn("uniform_q2_g64"),
            emb_bpw=emb_q4,
            cost_class="CHEAP_INREGISTER",
            recon_penalty=1.0,
            quality=(
                f"binary gate hold min {bin_gate['hold_min']:.4f}; "
                f"binary up min {pick(summary, 'up_proj', 'binary_g128')['hold_min']:.4f}; "
                f"q2 down min {q2_down['hold_min']:.4f}; "
                f"q2 attn min {q2_attn_min:.4f}"
            ),
            verdict="Cheap mixed without rice. Use if binary down fails and q2 down holds.",
            note="Exploits dense-every-row: no weight-space SVD fallback.",
        ),
        recipe_row(
            name="cheap_binary_gate_rice_up_q3_down_q4_rest_NO_PACK",
            mlp_bpw=(
                bin_gate["physical_bpw_mean"]
                + rice_up["physical_bpw_mean"]
                + q3_down["physical_bpw_mean"]
            )
            / 3.0,
            attn_bpw=attn("uniform_q4_g64"),
            emb_bpw=emb_q4,
            cost_class="EXPENSIVE_SPARSE",
            recon_penalty=round(
                (F_MLP * rice_up["physical_bpw_mean"])
                / (
                    F_MLP
                    * (
                        bin_gate["physical_bpw_mean"]
                        + rice_up["physical_bpw_mean"]
                        + q3_down["physical_bpw_mean"]
                    )
                    / 3.0
                    + (F_ATTN + F_EMB) * 4.25
                )
                * Q80_MIXED_SLOWDOWN_PER_BYTE
                + (
                    1.0
                    - (F_MLP * rice_up["physical_bpw_mean"])
                    / (
                        F_MLP
                        * (
                            bin_gate["physical_bpw_mean"]
                            + rice_up["physical_bpw_mean"]
                            + q3_down["physical_bpw_mean"]
                        )
                        / 3.0
                        + (F_ATTN + F_EMB) * 4.25
                    )
                ),
                3,
            ),
            quality=(
                f"rice up hold mean {rice_up['hold_mean']:.4f} min {rice_up['hold_min']:.4f}; "
                f"q3 down min {q3_down['hold_min']:.4f}"
            ),
            verdict="DO NOT BUILD — sibling already owns rice+lowrank. Shown only to price reconstruction.",
            note="If adjusted TPS < Q4 TPS, rice is a regression on this bandwidth-bound model.",
        ),
    ]
    # bytes-saved / reconstruction-ns proxy: higher is better.
    # Use (current_bpw - target) / (ms_adj) as a comparable score.
    for r in recipes:
        saved = CURRENT_BPW - r["projected_bpw"]
        r["bytes_saved_over_cost_adjusted_ms"] = round(
            saved / r["projected_ms_token_cost_adjusted"], 4
        )
    recipes.sort(key=lambda r: r["bytes_saved_over_cost_adjusted_ms"], reverse=True)
    return recipes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--layers", type=str, default=",".join(str(x) for x in LAYERS))
    args = parser.parse_args()
    layers = tuple(int(x) for x in args.layers.split(",") if x.strip())

    capture_meta = json.loads((CAPTURE_DIR / "capture-result.json").read_text())
    if capture_meta.get("status") != "CAPTURED_REAL_BF16_POST_NORM_HIDDEN":
        raise SystemExit(f"capture is not real: {capture_meta.get('status')}")
    if capture_meta.get("source", {}).get("not_synthetic") is not True:
        raise SystemExit("capture claims synthetic — refuse")

    weight_map = load_weight_map(MODEL_DIR)
    organs: list[dict[str, Any]] = []
    t_all = time.perf_counter()

    for layer in layers:
        print(f"=== layer {layer} gqa={is_gqa(layer)} ===", flush=True)
        X = load_hidden(layer)
        prefix = f"language_model.model.layers.{layer}."
        gate = np.ascontiguousarray(
            load_tensor(MODEL_DIR, weight_map, prefix + "mlp.gate_proj.weight"),
            dtype=np.float32,
        )
        organs.append(organ_rows(layer, "gate_proj", gate, X))
        up = np.ascontiguousarray(
            load_tensor(MODEL_DIR, weight_map, prefix + "mlp.up_proj.weight"),
            dtype=np.float32,
        )
        organs.append(organ_rows(layer, "up_proj", up, X))
        print(f"  computing post-SwiGLU X for L{layer} …", flush=True)
        t_sw = time.perf_counter()
        x_swiglu = silu(X @ gate.T) * (X @ up.T)
        print(f"  post-SwiGLU {x_swiglu.shape} in {time.perf_counter() - t_sw:.2f}s", flush=True)
        del gate, up
        down = np.ascontiguousarray(
            load_tensor(MODEL_DIR, weight_map, prefix + "mlp.down_proj.weight"),
            dtype=np.float32,
        )
        organs.append(organ_rows(layer, "down_proj", down, x_swiglu))
        del down, x_swiglu

        if is_gqa(layer):
            win = np.ascontiguousarray(
                load_tensor(MODEL_DIR, weight_map, prefix + "self_attn.q_proj.weight"),
                dtype=np.float32,
            )
            organs.append(organ_rows(layer, "attn_in", win, X))
            del win
            wout = np.ascontiguousarray(
                load_tensor(MODEL_DIR, weight_map, prefix + "self_attn.o_proj.weight"),
                dtype=np.float32,
            )
            organs.append(organ_rows(layer, "attn_out", wout, X))
            del wout
        else:
            win = np.ascontiguousarray(
                load_tensor(MODEL_DIR, weight_map, prefix + "linear_attn.in_proj_qkv.weight"),
                dtype=np.float32,
            )
            organs.append(organ_rows(layer, "attn_in", win, X))
            del win
            wout = np.ascontiguousarray(
                load_tensor(MODEL_DIR, weight_map, prefix + "linear_attn.out_proj.weight"),
                dtype=np.float32,
            )
            organs.append(organ_rows(layer, "attn_out", wout, X))
            del wout
        del X

    summary = summarize(organs)
    recipes = build_recipes(summary["by_role_codec"])

    receipt = {
        "schema": SCHEMA,
        "date": "2026-08-16",
        "lane": "qwen38-bpw-descent",
        "did_not_duplicate_sibling_mixed_pack": True,
        "sibling_pack": "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-2p0-v1",
        "activation": {
            "path": str(CAPTURE_DIR),
            "schema": capture_meta.get("schema"),
            "status": capture_meta.get("status"),
            "n_tokens": int(capture_meta.get("n_tokens", N_TOKENS)),
            "not_synthetic": True,
            "fit_kind": capture_meta.get("fit_kind"),
            "sha256_self": capture_meta.get("sha256_self"),
            "fit_n": FIT_N,
            "hold_n": HOLD_N,
            "down_proj_x": "real post-SwiGLU silu(X@Wg.T)*(X@Wu.T) from captured hidden + BF16 gate/up",
            "every_row_has_activations": True,
            "weight_space_svd_fallback_used": False,
        },
        "baseline": {
            "measured_ms": MEASURED_MS,
            "current_bpw": CURRENT_BPW,
            "achieved_gb_s": 406.2,
            "source": "receipts/ascent-2026-08-16/QWEN38_BANDWIDTH_BOUND.json",
            "invariant": "ms_at_target = measured_ms * (target_bpw / current_bpw)",
        },
        "mass_fractions": {"mlp": F_MLP, "attention_norms": F_ATTN, "embed_lm_head": F_EMB},
        "bars": {
            "q80_residual_identity": Q80_BAR,
            "moderate": MODERATE_BAR,
            "tight": TIGHT_BAR,
            "note": "0.8604 was pessimistic on Q80 (generation worked at down_proj 0.768). Generation is the gate.",
        },
        "layers": list(layers),
        "codec_catalog": [
            {k: c[k] for k in ("name", "cost_class", "recon_penalty", "note")} for c in CODECS
        ],
        "organs": organs,
        "summary": summary,
        "candidate_table": recipes,
        "wall_s": time.perf_counter() - t_all,
        "claim_boundary": {
            "full_model_not_packed": True,
            "generation_not_run": True,
            "gpu_not_used": True,
            "projections_are_byte_count_invariant_times_stated_penalty": True,
            "attention_census_not_exhaustive": True,
            "lm_head_not_output_scored": True,
            "sibling_owns_q80_recipe_pack": True,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sealed = seal(receipt)
    args.out.write_text(json.dumps(sealed, indent=2) + "\n")
    print(f"wrote {args.out} wall={sealed['wall_s']:.1f}s")
    print("=== CANDIDATE TABLE ===")
    for r in recipes:
        print(
            f"{r['codec']:48s} bpw={r['projected_bpw']:.3f}  "
            f"{r['reconstruction_cost_class']:28s}  "
            f"ms={r['projected_ms_token_invariant']:.2f}/{r['projected_ms_token_cost_adjusted']:.2f}  "
            f"tps={r['projected_tps_invariant']:.1f}/{r['projected_tps_cost_adjusted']:.1f}  "
            f"{r['verdict'][:80]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

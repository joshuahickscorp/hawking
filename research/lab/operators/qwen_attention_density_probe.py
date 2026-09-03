"""CPU representation probe: can Q80 / Qwen3.8 attention + lm_head compress?

Uses REAL captured activations (Q80 all-layer BF16 route capture; Qwen3.8
mlx BF16 post-norm hidden). Refuses synthetic / Gaussian X.

This is a representation measurement, not a kernel or GPU benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from lab.operators.ascension_dual_gravity_worker import (
    GROUP_BINARY,
    GROUP_UNIFORM,
    _binary_codec,
    _factor_codec,
    _hadamard_lattice_codec,
    _ternary_codec,
    _uniform_codec,
)
from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map
from lab.operators.residual_compact_codec import encode_residual_compact


REPO = Path(__file__).resolve().parents[2]
Q80_SRC = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen-80b/Qwen3-Coder-Next"
)
Q80_CAP = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "ascension-sandbox/physical/qwen80/quality-diagnostics/source-bf16-capture-n192-scale64"
)
Q38_SRC = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
)
Q38_CAP = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
)
Q38_Q4_MANIFEST = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json"
)

# Quality bars. Qwen3.8 uniform-Q4 vs BF16 min cosine is 0.98948
# (receipts/ascent-2026-08-16/THREE_MODEL_REGIME_SPLIT.json). Expert-MLP
# bar 0.8604 is intentionally NOT reused: attention is HIGH sensitivity.
BAR_MATCH_Q4 = 0.990
BAR_TIGHT = 0.995
BAR_INTEREST = 0.970

# Q80 Q4 attention GEMV mass (codes+f16 scales only). Equals
# receipts/QWEN80_TOKEN_NS_LEDGER.json theoretical_weight_bytes.attention_deltanet_gqa_bytes.
Q80_ATTN_Q4_BYTES = 818_036_736
Q80_LM_HEAD_Q4_BYTES = 165_306_368
Q80_ATTN_ELEMS = 1_539_833_856  # 818036736 * 8 / 4.25
Q80_LM_HEAD_ELEMS = 151_936 * 2_048
Q80_TOTAL_ELEMS = 79_674_391_296
Q80_Q4_TOKEN_BYTES = {
    "attention": 818_151_424,  # G013 verified class
    "lm_head": 165_329_552,
    "routed_experts": 802_160_640,  # Q4 vehicle; mixed is ~100.9M
    "router": 26_742_448,
    "shared_expert": 80_216_064,  # Q4 ledger shared_expert_bytes
}
Q80_MIXED_TOKEN_EXPERT_BYTES = 100_915_200

# Qwen3.8 measured active bytes (receipts/ascent-2026-08-16/QWEN38_ACTIVE_BUDGET_MEASURED.json)
Q38_ACTIVE = {
    "mlp": 9_091_161_600,
    "linear_attn": 2_961_704_064,
    "full_attn": 891_325_184,
    "lm_head": 675_430_440,
    "norms": 2_642_952,
}
Q38_TOTAL_ELEMS = 26_895_998_464
Q38_LM_HEAD_ELEMS = 248_320 * 5_120

SCHEMA = "hawking.ascension.qwen_attention_density_probe.v1"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mean_row_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
        b = b[None, :]
    num = np.sum(a * b, axis=1)
    da = np.linalg.norm(a, axis=1)
    db = np.linalg.norm(b, axis=1)
    den = da * db
    ok = den > 1e-12
    if not np.any(ok):
        return 0.0
    return float(np.mean(num[ok] / den[ok]))


def _min_row_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
        b = b[None, :]
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = den > 1e-12
    if not np.any(ok):
        return 0.0
    return float(np.min(num[ok] / den[ok]))


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False).reshape(-1)
    b = b.astype(np.float64, copy=False).reshape(-1)
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(a))
    return num / den if den > 1e-12 else num


def _silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def physical_bpw(nbytes: int, elements: int) -> float:
    if elements <= 0:
        return 0.0
    return 8.0 * float(nbytes) / float(elements)


def weight_stats(W: np.ndarray) -> dict[str, Any]:
    flat = np.ascontiguousarray(W, dtype=np.float32).reshape(-1)
    abs_w = np.abs(flat)
    med = float(np.median(abs_w))
    p = np.percentile(abs_w, [50, 90, 99, 99.9, 100]).astype(float)
    # excess kurtosis of raw weights
    mu = float(np.mean(flat))
    c = flat.astype(np.float64) - mu
    m2 = float(np.mean(c * c))
    m4 = float(np.mean(c * c * c * c))
    kurt = (m4 / (m2 * m2) - 3.0) if m2 > 0 else 0.0
    out = {
        "shape": [int(x) for x in W.shape],
        "elements": int(flat.size),
        "dtype_loaded": "float32_from_bf16",
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "rms": float(np.sqrt(np.mean(np.square(flat, dtype=np.float64)))),
        "mean_abs": float(np.mean(abs_w)),
        "median_abs": med,
        "p50_abs": float(p[0]),
        "p90_abs": float(p[1]),
        "p99_abs": float(p[2]),
        "p99_9_abs": float(p[3]),
        "max_abs": float(p[4]),
        "dynamic_range_max_over_median": (float(p[4]) / med) if med > 0 else None,
        "frac_abs_gt_10x_median": float(np.mean(abs_w > (10.0 * med))) if med > 0 else 0.0,
        "frac_abs_gt_100x_median": float(np.mean(abs_w > (100.0 * med))) if med > 0 else 0.0,
        "excess_kurtosis": kurt,
        "finite": bool(np.isfinite(flat).all()),
    }
    if W.ndim == 2:
        out_ch = np.sqrt(np.mean(np.square(W, dtype=np.float64), axis=1))
        in_ch = np.sqrt(np.mean(np.square(W, dtype=np.float64), axis=0))
        def _spread(v: np.ndarray) -> dict[str, float]:
            v = np.asarray(v, dtype=np.float64)
            med_v = float(np.median(v))
            return {
                "min": float(np.min(v)),
                "median": med_v,
                "max": float(np.max(v)),
                "cv": float(np.std(v) / (np.mean(v) + 1e-12)),
                "max_over_median": float(np.max(v) / med_v) if med_v > 0 else 0.0,
            }
        out["per_output_channel_rms"] = _spread(out_ch)
        out["per_input_channel_rms"] = _spread(in_ch)
    return out


def channel_sensitivity(W: np.ndarray, X: np.ndarray) -> dict[str, Any]:
    """Per-input-channel |W| * rms(X) — AWQ-style sensitivity, real X."""
    if W.ndim != 2 or X.ndim != 2 or X.shape[1] != W.shape[1]:
        return {"status": "geometry_mismatch"}
    w_col = np.mean(np.abs(W), axis=0).astype(np.float64)
    x_rms = np.sqrt(np.mean(np.square(X, dtype=np.float64), axis=0))
    s = w_col * x_rms
    med = float(np.median(s))
    return {
        "n_in": int(s.size),
        "median": med,
        "p90": float(np.percentile(s, 90)),
        "p99": float(np.percentile(s, 99)),
        "max": float(np.max(s)),
        "max_over_median": (float(np.max(s)) / med) if med > 0 else 0.0,
        "frac_gt_10x_median": float(np.mean(s > 10.0 * med)) if med > 0 else 0.0,
        "frac_gt_4x_median": float(np.mean(s > 4.0 * med)) if med > 0 else 0.0,
        "note": "sensitivity = mean_abs(W[:,j]) * rms(X[:,j]) on real captured rows",
    }


def load_q80_hidden(layer: int, n: int = 384) -> np.ndarray:
    root = Q80_CAP / "hidden" / f"L{layer:02d}"
    files: list[Path] = []
    for probe in sorted(p for p in root.iterdir() if p.is_dir()):
        for fn in sorted(probe.glob("*.f32le")):
            files.append(fn)
            if len(files) >= n:
                break
        if len(files) >= n:
            break
    if len(files) < 8:
        raise RuntimeError(f"Q80 hidden L{layer} only found {len(files)} rows")
    rows = [np.fromfile(f, dtype="<f4") for f in files[:n]]
    X = np.stack(rows, axis=0)
    if X.shape[1] != 2048:
        raise RuntimeError(f"Q80 hidden width {X.shape[1]} != 2048")
    return np.ascontiguousarray(X, dtype=np.float32)


def load_q38_hidden(layer: int) -> np.ndarray:
    path = Q38_CAP / "hidden" / f"L{layer:02d}.f32"
    raw = np.fromfile(path, dtype="<f4")
    if raw.size != 256 * 5120:
        raise RuntimeError(f"Qwen3.8 L{layer} hidden size {raw.size} != 256*5120")
    return np.ascontiguousarray(raw.reshape(256, 5120), dtype=np.float32)


def fuse_q38_qkvz(qkv: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Match crates/hawking-core/src/model/qwen38_geometry.rs fuse_in_proj_qkvz."""
    hidden = 5120
    key_heads = 16
    key_dim = 128
    values_per_key = 3
    value_dim = 128
    key_elements = key_heads * key_dim
    value_rows = values_per_key * value_dim
    qkvz_rows_per_key = key_dim * 2 + value_rows * 2
    fused = np.empty((key_heads * qkvz_rows_per_key, hidden), dtype=np.float32)
    for kh in range(key_heads):
        dst = kh * qkvz_rows_per_key
        q_src = kh * key_dim
        k_src = key_elements + kh * key_dim
        v_src = key_elements * 2 + kh * value_rows
        z_src = kh * value_rows
        fused[dst : dst + key_dim] = qkv[q_src : q_src + key_dim]
        fused[dst + key_dim : dst + 2 * key_dim] = qkv[k_src : k_src + key_dim]
        fused[dst + 2 * key_dim : dst + 2 * key_dim + value_rows] = qkv[v_src : v_src + value_rows]
        fused[dst + 2 * key_dim + value_rows : dst + qkvz_rows_per_key] = z[z_src : z_src + value_rows]
    return fused


def fuse_q38_ba(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    hidden = 5120
    key_heads = 16
    vpk = 3
    fused = np.empty((key_heads * vpk * 2, hidden), dtype=np.float32)
    for kh in range(key_heads):
        src = kh * vpk
        dst = kh * vpk * 2
        fused[dst : dst + vpk] = b[src : src + vpk]
        fused[dst + vpk : dst + 2 * vpk] = a[src : src + vpk]
    return fused


def deltanet_out_proxy(X: np.ndarray, W_qkvz: np.ndarray, *, key_heads: int, values_per_key: int, key_dim: int = 128, value_dim: int = 128) -> np.ndarray:
    """Real-derived out_proj X: v * silu(z) from source in_proj on captured hidden.

    Not the recurrent DeltaNet mix. Same width and same bilinear gate
    structure the mixer consumes. Labelled as a site proxy in the receipt.
    """
    y = X @ W_qkvz.T
    value_rows = values_per_key * value_dim
    per_key = key_dim * 2 + value_rows * 2
    y3 = y.reshape(X.shape[0], key_heads, per_key)
    v = y3[:, :, key_dim * 2 : key_dim * 2 + value_rows].reshape(X.shape[0], -1)
    z = y3[:, :, key_dim * 2 + value_rows :].reshape(X.shape[0], -1)
    return np.ascontiguousarray(v * _silu(z), dtype=np.float32)


def gqa_out_proxy(X: np.ndarray, W_q: np.ndarray, W_v: np.ndarray, *, n_heads: int, n_kv: int, head_dim: int = 256) -> np.ndarray:
    """Real-derived o_proj X: GQA-repeat(v) * sigmoid(q_gate).

    Not the softmax mix. Gate layout matches qwen38_attention_apply_sigmoid_gate
    / qwen80_gqa_apply_sigmoid_gate: per head [q | gate].
    """
    qg = X @ W_q.T
    v = X @ W_v.T
    qg = qg.reshape(X.shape[0], n_heads, 2, head_dim)
    gate = _sigmoid(qg[:, :, 1, :])
    v = v.reshape(X.shape[0], n_kv, head_dim)
    repeat = n_heads // n_kv
    v_rep = np.repeat(v, repeat, axis=1)
    return np.ascontiguousarray((v_rep * gate).reshape(X.shape[0], n_heads * head_dim), dtype=np.float32)


def act_weighted_rsvd(W: np.ndarray, X: np.ndarray, rank: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """HGRAVS01 factors: minimize ||(W-LR) X||_F via activation-weighted rSVD."""
    Xf = np.ascontiguousarray(X, dtype=np.float32)
    n = max(1, Xf.shape[0])
    gram = (Xf.T @ Xf) / float(n)
    ridge = 1e-5 * float(np.trace(gram) / max(gram.shape[0], 1)) + 1e-8
    gram = gram + ridge * np.eye(gram.shape[0], dtype=np.float32)
    evals, evecs = np.linalg.eigh(gram.astype(np.float64))
    evals = np.clip(evals, 1e-12, None)
    sqrt_g = ((evecs * np.sqrt(evals)) @ evecs.T).astype(np.float32)
    inv_sqrt_g = ((evecs * (1.0 / np.sqrt(evals))) @ evecs.T).astype(np.float32)
    actual = min(max(1, int(rank)), W.shape[0], W.shape[1])
    over = min(12, max(0, min(W.shape) - actual))
    rng = np.random.default_rng(seed)
    k = actual + over
    omega = rng.standard_normal((W.shape[1], k), dtype=np.float32)
    y = W @ (sqrt_g @ omega)
    q, _ = np.linalg.qr(y, mode="reduced")
    b = (q.T @ W) @ sqrt_g
    u, s, vt = np.linalg.svd(b, full_matrices=False)
    left = np.ascontiguousarray(q @ (u[:, :actual] * s[:actual]), dtype=np.float32)
    right = np.ascontiguousarray(vt[:actual, :] @ inv_sqrt_g, dtype=np.float32)
    return left, right


def eval_pair(Y: np.ndarray, Yh: np.ndarray, W: np.ndarray, Wh: np.ndarray | None) -> dict[str, float]:
    out: dict[str, float] = {
        "output_cosine": _mean_row_cosine(Y, Yh),
        "output_cosine_min_row": _min_row_cosine(Y, Yh),
        "output_rel_l2": _rel_l2(Y, Yh),
    }
    if Wh is not None:
        out["weight_cosine"] = _mean_row_cosine(W.reshape(1, -1), Wh.reshape(1, -1))
        out["weight_rel_l2"] = _rel_l2(W, Wh)
    return out


def uniform_act_scaled(W: np.ndarray, X: np.ndarray, bits: int) -> tuple[np.ndarray, int]:
    """Column-scale by real activation RMS, then HGRAVU01. Cheap reconstruct."""
    s = np.sqrt(np.mean(np.square(X, dtype=np.float64), axis=0)).astype(np.float32)
    s = np.maximum(s, 1e-8)
    packed = _uniform_codec(W * s[None, :], bits=bits, group_size=GROUP_UNIFORM)
    what = packed.reconstruction.reshape(W.shape) / s[None, :]
    extra = int(s.size * 2)  # f16 per-in-channel scales
    return what, len(packed.payload) + extra


def propose_hgravs01(W: np.ndarray, X_fit: np.ndarray, X_hold: np.ndarray, rank: int, bits: int = 3) -> dict[str, Any]:
    left, right = act_weighted_rsvd(W, X_fit, rank)
    l_body, l_hat, _ = _factor_codec(left, bits=bits)
    r_body, r_hat, _ = _factor_codec(right, bits=bits)
    # header is billed in the real container; ~0.6-2 KB. Use bodies + 2048 slack.
    nbytes = len(l_body) + len(r_body) + 2048
    Y = X_hold @ W.T
    Yh = (X_hold @ r_hat.T) @ l_hat.T
    q = eval_pair(Y, Yh, W, None)
    q["bpw"] = physical_bpw(nbytes, int(W.size))
    q["bytes"] = int(nbytes)
    q["rank"] = int(left.shape[1])
    q["factor_bits"] = bits
    return q


def run_codecs(
    W: np.ndarray,
    X: np.ndarray,
    *,
    skip_heavy: bool,
    do_hgravs: bool,
    hgravs_ranks: list[int],
) -> list[dict[str, Any]]:
    """Score existing Gravity families on this (W, X). X is the true in-dim."""
    n = X.shape[0]
    # Deterministic even/odd split so fit/hold are interleaved across the
    # prompt stream rather than first-half vs last-half of one probe.
    fit_idx = np.arange(0, n, 2)
    hold_idx = np.arange(1, n, 2)
    if hold_idx.size < 8:
        hold_idx = np.arange(n)
        fit_idx = np.arange(n)
    X_fit, X_hold = X[fit_idx], X[hold_idx]
    Y_hold = X_hold @ W.T
    rows: list[dict[str, Any]] = []

    def add(name: str, family: str, recon: str, payload_bytes: int, Wh: np.ndarray | None, Yh: np.ndarray, **extra: Any) -> None:
        rec = {
            "codec": name,
            "family": family,
            "reconstruction_cost_class": recon,
            "payload_bytes": int(payload_bytes),
            "bpw": physical_bpw(payload_bytes, int(W.size)),
            **eval_pair(Y_hold, Yh, W, Wh),
            **extra,
        }
        rows.append(rec)

    # HGRAVU01 family — CHEAP, shipping Q4 kernel geometry.
    bit_list = (8, 6, 5, 4, 3, 2) if not skip_heavy else (8, 4, 3)
    for bits in bit_list:
        packed = _uniform_codec(W, bits=bits, group_size=GROUP_UNIFORM)
        Wh = packed.reconstruction.reshape(W.shape)
        add(
            f"HGRAVU01_q{bits}_g64",
            "HGRAVU01",
            "CHEAP_INREGISTER",
            len(packed.payload),
            Wh,
            X_hold @ Wh.T,
            shipping_kernel="qwen_uniform_q4_group64_matvec (same family, bits vary)",
        )
        del packed, Wh

    # Activation-scaled uniform (AWQ-style). Still CHEAP_INREGISTER.
    for bits in (4, 3, 2):
        Wh, nbytes = uniform_act_scaled(W, X_fit, bits)
        add(
            f"HGRAVU01_q{bits}_g64_act_colscale",
            "HGRAVU01_act_scaled",
            "CHEAP_INREGISTER",
            nbytes,
            Wh,
            X_hold @ Wh.T,
            note="column scales from real X_fit RMS; folded at pack time",
        )
        del Wh

    # HGRAVB01 — algorithmically cheap; no attention kernel ships.
    packed = _binary_codec(W, group_size=GROUP_BINARY)
    Wh = packed.reconstruction.reshape(W.shape)
    add(
        "HGRAVB01_binary_g128",
        "HGRAVB01",
        "CHEAP_INREGISTER_NO_ATTN_KERNEL",
        len(packed.payload),
        Wh,
        X_hold @ Wh.T,
        shipping_kernel="q80_binary_group_matvec exists for expert gate ONLY",
    )
    del packed, Wh

    # HGRAVR02 shipping recipe — EXPENSIVE scatter.
    if not skip_heavy:
        packed = encode_residual_compact(
            W, outlier_ratio=0.02, group_size=GROUP_BINARY, index_mode="rice", value_bits=1, value_scale="rms"
        )
        Wh = packed.reconstruction.reshape(W.shape)
        add(
            "HGRAVR02_binary_rice_q1_rms_2pct",
            "HGRAVR02",
            "EXPENSIVE_SCATTER",
            len(packed.payload),
            Wh,
            X_hold @ Wh.T,
            shipping_kernel="q80_binary_group_matvec + q80_sparse_q1_apply_csr (expert up ONLY)",
        )
        del packed, Wh

    # Hadamard lattice — extra ALU, still in-register. Skip lm_head-sized.
    if not skip_heavy and W.size <= 40_000_000:
        for bits in (4, 3):
            packed = _hadamard_lattice_codec(W, bits=bits, group_size=128)
            Wh = packed.reconstruction.reshape(W.shape)
            add(
                f"HGRAVH01_hadamard_q{bits}_g128",
                "HGRAVH01",
                "MEDIUM_INREGISTER_TRANSFORM",
                len(packed.payload),
                Wh,
                X_hold @ Wh.T,
            )
            del packed, Wh
        packed = _ternary_codec(W, threshold_multiplier=0.55, group_size=GROUP_BINARY)
        Wh = packed.reconstruction.reshape(W.shape)
        add(
            "HGRAVT01_ternary_t0.55_g128",
            "HGRAVT01",
            "CHEAP_INREGISTER_NO_ATTN_KERNEL",
            len(packed.payload),
            Wh,
            X_hold @ Wh.T,
        )
        del packed, Wh

    if do_hgravs:
        for rank in hgravs_ranks:
            if rank >= min(W.shape):
                continue
            q = propose_hgravs01(W, X_fit, X_hold, rank, bits=3)
            rows.append(
                {
                    "codec": f"HGRAVS01_r{q['rank']}_b3",
                    "family": "HGRAVS01",
                    "reconstruction_cost_class": "CONDITIONAL_TWOSTAGE",
                    "payload_bytes": q["bytes"],
                    "bpw": q["bpw"],
                    "output_cosine": q["output_cosine"],
                    "output_cosine_min_row": q["output_cosine_min_row"],
                    "output_rel_l2": q["output_rel_l2"],
                    "rank": q["rank"],
                    "factor_bits": 3,
                    "shipping_kernel": "q80_hgravs01_factor_matvec exists for expert down [2048,512] r160 ONLY",
                    "note": "cheap iff fused y=L@(R@x); LOSS if W is materialized (mixed vehicle 5.9x slower/byte)",
                }
            )
    return rows


def pick_winner(candidates: list[dict[str, Any]], *, bar: float, cheap_only: bool) -> dict[str, Any] | None:
    pool = []
    for c in candidates:
        if float(c.get("output_cosine") or 0) < bar:
            continue
        if cheap_only and not str(c.get("reconstruction_cost_class", "")).startswith("CHEAP_INREGISTER"):
            continue
        pool.append(c)
    if not pool:
        return None
    return min(pool, key=lambda c: (float(c["bpw"]), -float(c["output_cosine"])))


def q80_census() -> dict[str, Any]:
    classes = {
        "linear_attn.in_proj_qkvz.weight": {"n": 36, "shape": [12288, 2048], "q4": True},
        "linear_attn.in_proj_ba.weight": {"n": 36, "shape": [64, 2048], "q4": True},
        "linear_attn.out_proj.weight": {"n": 36, "shape": [2048, 4096], "q4": True},
        "linear_attn.conv1d.weight": {"n": 36, "shape": [8192, 1, 4], "q4": False},
        "linear_attn.A_log": {"n": 36, "shape": [32], "q4": False},
        "linear_attn.dt_bias": {"n": 36, "shape": [32], "q4": False},
        "linear_attn.norm.weight": {"n": 36, "shape": [128], "q4": False},
        "self_attn.q_proj.weight": {"n": 12, "shape": [8192, 2048], "q4": True},
        "self_attn.k_proj.weight": {"n": 12, "shape": [512, 2048], "q4": True},
        "self_attn.v_proj.weight": {"n": 12, "shape": [512, 2048], "q4": True},
        "self_attn.o_proj.weight": {"n": 12, "shape": [2048, 4096], "q4": True},
        "self_attn.q_norm.weight": {"n": 12, "shape": [256], "q4": False},
        "self_attn.k_norm.weight": {"n": 12, "shape": [256], "q4": False},
        "lm_head.weight": {"n": 1, "shape": [151936, 2048], "q4": True},
    }
    rows = []
    q4_attn = 0
    q4_lm = 0
    for name, spec in classes.items():
        el = int(np.prod(spec["shape"])) * spec["n"]
        today = int(round(el * 4.25 / 8.0)) if spec["q4"] else int(el * 4)  # f32-ish / bf16 source
        # Today on the Q4 vehicle the GEMV class is 4.25; 1-d stay tiny.
        if spec["q4"]:
            today = int(round(el * 4.25 / 8.0))
            if name == "lm_head.weight":
                q4_lm += today
            else:
                q4_attn += today
        else:
            today = 0  # not in the 818 MB GEMV class
        rows.append(
            {
                "tensor_class": name,
                "count": spec["n"],
                "shape": spec["shape"],
                "elements_total": el,
                "in_q4_attention_or_lm_head_mass": spec["q4"],
                "bytes_today_q4_vehicle": today,
            }
        )
    return {
        "model": "Qwen3-Coder-Next (Q80)",
        "geometry": {
            "layers": 48,
            "hidden": 2048,
            "deltanet_layers": 36,
            "gqa_layers": 12,
            "gqa_rule": "layer % 4 == 3",
            "linear_key_heads": 16,
            "linear_value_heads": 32,
            "gqa_heads": 16,
            "gqa_kv_heads": 2,
            "gqa_head_dim": 256,
            "q_proj_includes_sigmoid_gate": True,
            "q_proj_layout": "per-head [q_256 | gate_256]",
        },
        "classes": rows,
        "q4_attention_gemv_bytes": q4_attn,
        "q4_lm_head_bytes": q4_lm,
        "ledger_attention_bytes": Q80_ATTN_Q4_BYTES,
        "ledger_lm_head_bytes": Q80_LM_HEAD_Q4_BYTES,
        "identity_check_attn": q4_attn == Q80_ATTN_Q4_BYTES,
    }


def q38_census() -> dict[str, Any]:
    man = json.loads(Q38_Q4_MANIFEST.read_text())
    by: dict[str, dict[str, Any]] = {}
    for t in man["tensors"]:
        name = t["name"]
        if "linear_attn" in name:
            key = "linear_attn." + name.split("linear_attn.", 1)[1]
        elif "self_attn" in name:
            key = "self_attn." + name.split("self_attn.", 1)[1]
        elif name.endswith("lm_head.weight"):
            key = "lm_head.weight"
        else:
            continue
        rec = by.setdefault(key, {"bytes": 0, "elements": 0, "n": 0, "shapes": set(), "kind": t["kind"]})
        rec["bytes"] += int(t["bytes"])
        rec["elements"] += int(t["elements"])
        rec["n"] += 1
        rec["shapes"].add(tuple(t["shape"]))
    rows = []
    for key, rec in sorted(by.items()):
        rows.append(
            {
                "tensor_class": key,
                "count": rec["n"],
                "shapes": [list(s) for s in sorted(rec["shapes"])],
                "elements_total": rec["elements"],
                "bytes_today_q4": rec["bytes"],
                "kind": rec["kind"],
                "bpw": physical_bpw(rec["bytes"], rec["elements"]),
            }
        )
    return {
        "model": "Qwen3.8-27B language-only",
        "geometry": {
            "layers": 64,
            "hidden": 5120,
            "linear_attn_layers": 48,
            "full_attn_layers": 16,
            "gqa_rule": "(layer+1)%4==0 i.e. layer%4==3",
            "linear_key_heads": 16,
            "linear_value_heads": 48,
            "gqa_heads": 24,
            "gqa_kv_heads": 4,
            "gqa_head_dim": 256,
            "q_proj_includes_sigmoid_gate": True,
            "runtime_fuses_split_in_proj": True,
            "source_split": ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"],
            "runtime_fused": ["in_proj_qkvz [16384,5120]", "in_proj_ba [96,5120]"],
        },
        "classes": rows,
        "measured_active": Q38_ACTIVE,
        "q4_complete_bpw": man["complete_physical_bpw"],
    }


def describe_activation_sites() -> dict[str, Any]:
    q80_result = Q80_CAP / "capture-result.json"
    q38_result = Q38_CAP / "capture-result.json"
    q38 = json.loads(q38_result.read_text())
    return {
        "q80": {
            "path": str(Q80_CAP),
            "schema": "hawking.ascension.qwen80_broad_activation_all_layer_route_capture_result.v1",
            "n_tokens": 25258,
            "site": "BF16-source layer-streamed post-attention RMSNorm (router-input)",
            "site_is_attention_in_proj_input": False,
            "site_is_same_width_rms_normalized_hidden": True,
            "hidden_width": 2048,
            "not_synthetic": True,
            "lm_head_final_norm_executed": False,
            "capture_result_sha256": "17a1e9b60a53cc491601a549880c2d215ff16395ee36abaa05fb95eb7fe2aabe",
            "caveat": (
                "In-proj functional scores use REAL post-attention RMSNorm rows, "
                "not input_layernorm residual. Wrong residual point, real distribution. "
                "Gaussian-proxy is the known-invalid class; this is not that class. "
                "out_proj uses a real-derived mixer-site proxy (v*silu(z) or GQA-repeat(v)*sigmoid(gate))."
            ),
        },
        "qwen38": {
            "path": str(Q38_CAP),
            "schema": q38["schema"],
            "status": q38["status"],
            "n_tokens": q38["n_tokens"],
            "hidden_width": q38["hidden"],
            "forward": q38["source"]["forward"],
            "not_synthetic": q38["source"]["not_synthetic"],
            "fit_kind": q38["fit_kind"],
            "site": "CAPTURED_REAL_BF16_POST_NORM_HIDDEN, 256 rows x 5120, 5 native prompts",
            "site_is_attention_in_proj_input": "UNCONFIRMED_POST_NORM",
            "sha256_self": q38.get("sha256_self"),
            "caveat": (
                "Schema name says post_swiglu but stored width is 5120 (hidden), "
                "so these are residual-width post-norm hiddens, not the 17408 SwiGLU intermediate. "
                "Used as real same-width X for in-proj / lm_head. out_proj uses real-derived proxy."
            ),
        },
    }


def probe_one(
    *,
    model: str,
    label: str,
    W: np.ndarray,
    X: np.ndarray,
    bytes_today: int,
    n_replicas: int,
    x_site: str,
    skip_heavy: bool,
    do_hgravs: bool,
    hgravs_ranks: list[int],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    stats = weight_stats(W)
    sens = channel_sensitivity(W, X)
    cands = run_codecs(W, X, skip_heavy=skip_heavy, do_hgravs=do_hgravs, hgravs_ranks=hgravs_ranks)
    cheap_q4 = pick_winner(cands, bar=BAR_MATCH_Q4, cheap_only=True)
    any_q4 = pick_winner(cands, bar=BAR_MATCH_Q4, cheap_only=False)
    cheap_tight = pick_winner(cands, bar=BAR_TIGHT, cheap_only=True)
    # Q4 itself is the incumbent; find it
    q4 = next((c for c in cands if c["codec"] == "HGRAVU01_q4_g64"), None)
    proposed = cheap_q4 or q4
    proj_bpw = float(proposed["bpw"]) if proposed else 4.25
    proj_one = int(round(W.size * proj_bpw / 8.0))
    proj_all = proj_one * n_replicas
    today_all = bytes_today * n_replicas if bytes_today > 0 else int(round(W.size * n_replicas * 4.25 / 8.0))
    return {
        "model": model,
        "tensor": label,
        "n_replicas": n_replicas,
        "shape": [int(x) for x in W.shape],
        "elements_one": int(W.size),
        "bytes_today_one": int(bytes_today if bytes_today > 0 else round(W.size * 4.25 / 8.0)),
        "bytes_today_all_replicas": int(today_all),
        "x_site": x_site,
        "n_x_rows": int(X.shape[0]),
        "weight_stats": stats,
        "per_channel_sensitivity": sens,
        "candidates": cands,
        "winner_cheap_at_0p990": cheap_q4,
        "winner_any_at_0p990": any_q4,
        "winner_cheap_at_0p995": cheap_tight,
        "proposed": proposed,
        "proposed_codec": None if proposed is None else proposed["codec"],
        "proposed_bpw": proj_bpw,
        "proposed_bytes_all_replicas": proj_all,
        "byte_delta_all_replicas": proj_all - today_all,
        "wall_s": time.perf_counter() - t0,
    }


def load_q80_tensor(wmap: dict[str, str], name: str) -> np.ndarray:
    return np.ascontiguousarray(load_tensor(Q80_SRC, wmap, name), dtype=np.float32)


def load_q38_tensor(wmap: dict[str, str], name: str) -> np.ndarray:
    return np.ascontiguousarray(load_tensor(Q38_SRC, wmap, name), dtype=np.float32)


def run(out_path: Path, *, quick: bool) -> dict[str, Any]:
    t_all = time.perf_counter()
    q80_wmap = load_weight_map(Q80_SRC)
    q38_wmap = load_weight_map(Q38_SRC)
    sites = describe_activation_sites()
    census = {"q80": q80_census(), "qwen38": q38_census()}

    # Cache hiddens
    q80_h: dict[int, np.ndarray] = {}
    q38_h: dict[int, np.ndarray] = {}

    def q80x(layer: int) -> np.ndarray:
        if layer not in q80_h:
            q80_h[layer] = load_q80_hidden(layer, n=256 if quick else 384)
        return q80_h[layer]

    def q38x(layer: int) -> np.ndarray:
        if layer not in q38_h:
            q38_h[layer] = load_q38_hidden(layer)
        return q38_h[layer]

    probes: list[dict[str, Any]] = []
    ckpt = out_path.with_suffix(".partial.json")

    def add(p: dict[str, Any]) -> None:
        print(
            f"  {p['tensor']:48s} cheap@0.99={p['winner_cheap_at_0p990']['codec'] if p['winner_cheap_at_0p990'] else 'NONE'}"
            f" bpw={p['proposed_bpw']:.3f}  wall={p['wall_s']:.1f}s",
            flush=True,
        )
        probes.append(p)
        ckpt.write_text(json.dumps({"n": len(probes), "probes": probes}, indent=2, default=str))

    # ---- Q80 DeltaNet L0 + L36 ----
    q80_dn_layers = [0] if quick else [0, 36]
    for L in q80_dn_layers:
        print(f"Q80 DeltaNet L{L}", flush=True)
        X = q80x(L)
        Wq = load_q80_tensor(q80_wmap, f"model.layers.{L}.linear_attn.in_proj_qkvz.weight")
        add(
            probe_one(
                model="q80",
                label=f"q80.L{L}.linear_attn.in_proj_qkvz",
                W=Wq,
                X=X,
                bytes_today=int(round(Wq.size * 4.25 / 8.0)),
                n_replicas=36,
                x_site="q80_post_attn_rmsnorm_router_input",
                skip_heavy=quick,
                do_hgravs=True,
                hgravs_ranks=[64, 128, 256, 512] if not quick else [128, 256],
            )
        )
        Wb = load_q80_tensor(q80_wmap, f"model.layers.{L}.linear_attn.in_proj_ba.weight")
        add(
            probe_one(
                model="q80",
                label=f"q80.L{L}.linear_attn.in_proj_ba",
                W=Wb,
                X=X,
                bytes_today=int(round(Wb.size * 4.25 / 8.0)),
                n_replicas=36,
                x_site="q80_post_attn_rmsnorm_router_input",
                skip_heavy=False,
                do_hgravs=True,
                hgravs_ranks=[8, 16, 32],
            )
        )
        Wo = load_q80_tensor(q80_wmap, f"model.layers.{L}.linear_attn.out_proj.weight")
        Xp = deltanet_out_proxy(X, Wq, key_heads=16, values_per_key=2)
        add(
            probe_one(
                model="q80",
                label=f"q80.L{L}.linear_attn.out_proj",
                W=Wo,
                X=Xp,
                bytes_today=int(round(Wo.size * 4.25 / 8.0)),
                n_replicas=36,
                x_site="real_derived_v_silu_z_from_source_in_proj_qkvz",
                skip_heavy=quick,
                do_hgravs=True,
                hgravs_ranks=[64, 128, 256, 512] if not quick else [128, 256],
            )
        )
        del Wq, Wb, Wo, Xp

    # ---- Q80 GQA L3 + L47 ----
    q80_gqa_layers = [3] if quick else [3, 47]
    for L in q80_gqa_layers:
        print(f"Q80 GQA L{L}", flush=True)
        X = q80x(L)
        Wq = load_q80_tensor(q80_wmap, f"model.layers.{L}.self_attn.q_proj.weight")
        Wk = load_q80_tensor(q80_wmap, f"model.layers.{L}.self_attn.k_proj.weight")
        Wv = load_q80_tensor(q80_wmap, f"model.layers.{L}.self_attn.v_proj.weight")
        Wo = load_q80_tensor(q80_wmap, f"model.layers.{L}.self_attn.o_proj.weight")
        for suffix, W, reps in (
            ("q_proj", Wq, 12),
            ("k_proj", Wk, 12),
            ("v_proj", Wv, 12),
        ):
            add(
                probe_one(
                    model="q80",
                    label=f"q80.L{L}.self_attn.{suffix}",
                    W=W,
                    X=X,
                    bytes_today=int(round(W.size * 4.25 / 8.0)),
                    n_replicas=reps,
                    x_site="q80_post_attn_rmsnorm_router_input",
                    skip_heavy=quick and suffix == "q_proj",
                    do_hgravs=True,
                    hgravs_ranks=[64, 128, 256, 512] if suffix == "q_proj" else [32, 64, 128, 256],
                )
            )
        Xp = gqa_out_proxy(X, Wq, Wv, n_heads=16, n_kv=2)
        add(
            probe_one(
                model="q80",
                label=f"q80.L{L}.self_attn.o_proj",
                W=Wo,
                X=Xp,
                bytes_today=int(round(Wo.size * 4.25 / 8.0)),
                n_replicas=12,
                x_site="real_derived_gqa_repeat_v_times_sigmoid_qgate",
                skip_heavy=quick,
                do_hgravs=True,
                hgravs_ranks=[64, 128, 256, 512] if not quick else [128, 256],
            )
        )
        del Wq, Wk, Wv, Wo, Xp

    # ---- Q80 lm_head ----
    print("Q80 lm_head", flush=True)
    # Last captured hidden is L47 post-attn RMSNorm; not post-final-norm.
    Xl = q80x(47)
    Wl = load_q80_tensor(q80_wmap, "lm_head.weight")
    add(
        probe_one(
            model="q80",
            label="q80.lm_head",
            W=Wl,
            X=Xl,
            bytes_today=Q80_LM_HEAD_Q4_BYTES,
            n_replicas=1,
            x_site="q80_L47_post_attn_rmsnorm_NOT_final_norm",
            skip_heavy=True,  # skip residual/hadamard on 311M weights
            do_hgravs=True,
            hgravs_ranks=[128, 256, 512, 1024] if not quick else [256, 512],
        )
    )
    del Wl

    # ---- Qwen3.8 DeltaNet L0 + L32 ----
    q38_dn_layers = [0] if quick else [0, 32]
    for L in q38_dn_layers:
        print(f"Qwen3.8 DeltaNet L{L}", flush=True)
        X = q38x(L)
        pfx = f"language_model.model.layers.{L}.linear_attn."
        qkv = load_q38_tensor(q38_wmap, pfx + "in_proj_qkv.weight")
        z = load_q38_tensor(q38_wmap, pfx + "in_proj_z.weight")
        a = load_q38_tensor(q38_wmap, pfx + "in_proj_a.weight")
        b = load_q38_tensor(q38_wmap, pfx + "in_proj_b.weight")
        Wq = fuse_q38_qkvz(qkv, z)
        Wb = fuse_q38_ba(b, a)
        # also score the source splits — they may differ
        for suffix, W, today_el_scale, reps in (
            ("in_proj_qkv", qkv, 1.0, 48),
            ("in_proj_z", z, 1.0, 48),
        ):
            add(
                probe_one(
                    model="qwen38",
                    label=f"qwen38.L{L}.linear_attn.{suffix}",
                    W=W,
                    X=X,
                    bytes_today=int(round(W.size * 4.25 / 8.0)),
                    n_replicas=reps,
                    x_site="qwen38_post_norm_hidden_5120",
                    skip_heavy=quick,
                    do_hgravs=True,
                    hgravs_ranks=[128, 256, 512] if not quick else [128, 256],
                )
            )
        add(
            probe_one(
                model="qwen38",
                label=f"qwen38.L{L}.linear_attn.in_proj_qkvz_fused",
                W=Wq,
                X=X,
                bytes_today=int(round(Wq.size * 4.25 / 8.0)),
                n_replicas=48,
                x_site="qwen38_post_norm_hidden_5120",
                skip_heavy=quick,
                do_hgravs=True,
                hgravs_ranks=[128, 256, 512] if not quick else [128, 256],
            )
        )
        add(
            probe_one(
                model="qwen38",
                label=f"qwen38.L{L}.linear_attn.in_proj_ba_fused",
                W=Wb,
                X=X,
                bytes_today=int(round(Wb.size * 4.25 / 8.0)),
                n_replicas=48,
                x_site="qwen38_post_norm_hidden_5120",
                skip_heavy=False,
                do_hgravs=True,
                hgravs_ranks=[8, 16, 32],
            )
        )
        Wo = load_q38_tensor(q38_wmap, pfx + "out_proj.weight")
        Xp = deltanet_out_proxy(X, Wq, key_heads=16, values_per_key=3)
        add(
            probe_one(
                model="qwen38",
                label=f"qwen38.L{L}.linear_attn.out_proj",
                W=Wo,
                X=Xp,
                bytes_today=int(round(Wo.size * 4.25 / 8.0)),
                n_replicas=48,
                x_site="real_derived_v_silu_z_from_fused_in_proj_qkvz",
                skip_heavy=quick,
                do_hgravs=True,
                hgravs_ranks=[128, 256, 512] if not quick else [128, 256],
            )
        )
        del qkv, z, a, b, Wq, Wb, Wo, Xp

    # ---- Qwen3.8 GQA L3 + L63 ----
    q38_gqa_layers = [3] if quick else [3, 63]
    for L in q38_gqa_layers:
        print(f"Qwen3.8 GQA L{L}", flush=True)
        X = q38x(L)
        pfx = f"language_model.model.layers.{L}.self_attn."
        Wq = load_q38_tensor(q38_wmap, pfx + "q_proj.weight")
        Wk = load_q38_tensor(q38_wmap, pfx + "k_proj.weight")
        Wv = load_q38_tensor(q38_wmap, pfx + "v_proj.weight")
        Wo = load_q38_tensor(q38_wmap, pfx + "o_proj.weight")
        for suffix, W in (("q_proj", Wq), ("k_proj", Wk), ("v_proj", Wv)):
            add(
                probe_one(
                    model="qwen38",
                    label=f"qwen38.L{L}.self_attn.{suffix}",
                    W=W,
                    X=X,
                    bytes_today=int(round(W.size * 4.25 / 8.0)),
                    n_replicas=16,
                    x_site="qwen38_post_norm_hidden_5120",
                    skip_heavy=quick and suffix == "q_proj",
                    do_hgravs=True,
                    hgravs_ranks=[128, 256, 512] if suffix == "q_proj" else [32, 64, 128, 256],
                )
            )
        Xp = gqa_out_proxy(X, Wq, Wv, n_heads=24, n_kv=4)
        add(
            probe_one(
                model="qwen38",
                label=f"qwen38.L{L}.self_attn.o_proj",
                W=Wo,
                X=Xp,
                bytes_today=int(round(Wo.size * 4.25 / 8.0)),
                n_replicas=16,
                x_site="real_derived_gqa_repeat_v_times_sigmoid_qgate",
                skip_heavy=quick,
                do_hgravs=True,
                hgravs_ranks=[128, 256, 512] if not quick else [128, 256],
            )
        )
        del Wq, Wk, Wv, Wo, Xp

    print("Qwen3.8 lm_head", flush=True)
    Xl = q38x(63)
    Wl = load_q38_tensor(q38_wmap, "language_model.lm_head.weight")
    add(
        probe_one(
            model="qwen38",
            label="qwen38.lm_head",
            W=Wl,
            X=Xl,
            bytes_today=Q38_ACTIVE["lm_head"],
            n_replicas=1,
            x_site="qwen38_L63_post_norm_hidden_NOT_confirmed_final_norm",
            skip_heavy=True,
            do_hgravs=True,
            hgravs_ranks=[256, 512, 1024] if not quick else [256, 512],
        )
    )
    del Wl

    receipt = {
        "schema": SCHEMA,
        "date": "2026-08-16",
        "status": "MEASURED_REPRESENTATION",
        "claim_boundary": {
            "not_a_kernel": True,
            "not_a_gpu_benchmark": True,
            "not_a_generation_coherence_claim": True,
            "used_real_activations": True,
            "used_synthetic_or_gaussian": False,
            "attention_in_proj_x_site_is_post_norm_not_input_layernorm": True,
            "out_proj_x_is_real_derived_mixer_proxy": True,
            "expert_mlp_recipe_not_assumed_to_transfer": True,
        },
        "quality_bars": {
            "match_q4": BAR_MATCH_Q4,
            "tight": BAR_TIGHT,
            "interest": BAR_INTEREST,
            "why_not_expert_bar_0p8604": "attention and lm_head are HIGH sensitivity; Qwen3.8 Q4 vs bf16 min cosine is 0.98948",
        },
        "activation_sites": sites,
        "census": census,
        "probes": probes,
        "wall_s": time.perf_counter() - t_all,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, default=str))
    tmp.replace(out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes) in {receipt['wall_s']:.1f}s", flush=True)
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_PROBE.json",
    )
    ap.add_argument("--quick", action="store_true", help="one layer per class, fewer ranks")
    args = ap.parse_args()
    run(args.out, quick=args.quick)


if __name__ == "__main__":
    main()

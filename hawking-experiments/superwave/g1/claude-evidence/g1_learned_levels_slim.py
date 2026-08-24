#!/usr/bin/env python3
"""Slim follow-up: frozen winners on remaining tensors + all 64 downs.

Reuses helpers from g1_learned_levels.py. CPU only. No GPU.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "/tmp")
from g1_learned_levels import (  # noqa: E402
    CAP,
    FIT_N,
    G64,
    G128,
    HIDDEN,
    OUT,
    assign_levels_chunked,
    bpw_codes,
    col_energy_for_groups,
    decode_class,
    down_x,
    group_pad,
    is_gqa,
    load_hidden,
    load_site_x,
    load_tensor,
    log,
    mixer_x,
    recon_binary_meanabs,
    recon_fixed_grid,
    recon_learned_shared,
    recon_lloyd,
    recon_ternary_sym,
    recon_ternary_t07,
    recon_uniform_clip,
    rss_gb,
    score,
    snap_f16,
    tname,
    unpad,
    weight_stats,
)

# Freeze the L0-gate winners. p=1.35 and mu=2.0 were the grid opts there.
P_STAR = 1.35
MU_STAR = 2.0
SCALE_SLIM = np.array([0.70, 0.82, 0.88, 0.96, 1.00], dtype=np.float32)


def power_levels(k: int, p: float) -> np.ndarray:
    u = (np.arange(k, dtype=np.float32) - 0.5 * (k - 1)) / max(0.5 * (k - 1), 1e-6)
    return (np.sign(u) * np.power(np.abs(u), float(p))).astype(np.float32)


def mu_levels(k: int, mu: float) -> np.ndarray:
    u = (np.arange(k, dtype=np.float32) - 0.5 * (k - 1)) / max(0.5 * (k - 1), 1e-6)
    return (np.sinh(float(mu) * u) / math.sinh(float(mu))).astype(np.float32)


def recon_levels_search(W: np.ndarray, levels: np.ndarray, g: int, wcost) -> np.ndarray:
    padded, n = group_pad(W, g)
    amax = np.max(np.abs(padded), axis=1)
    # slim multiplier grid
    lev_max = float(np.max(np.abs(levels))) if levels.size else 1.0
    denom = max(lev_max, 1e-12)
    best_cost = None
    best_recon = None
    for m in SCALE_SLIM:
        s = snap_f16((amax * float(m)) / denom)
        recon = assign_levels_chunked(padded, levels, s)
        err = padded - recon
        if wcost is None:
            cost = float(np.sum(np.square(err, dtype=np.float64)))
        else:
            cost = float(np.sum(np.square(err, dtype=np.float64) * wcost))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_recon = recon
    assert best_recon is not None
    return unpad(best_recon, n, W.shape)


def run_slim(W, X_fit, X_hold, Y_hold, R_hold, *, want_lloyd: bool) -> list[dict]:
    n_out, n_in = W.shape
    n = int(W.size)
    wcost = None
    if X_fit is not None and X_fit.shape[1] == n_in and n_in % G64 == 0:
        energy = np.sum(np.square(X_fit, dtype=np.float64), axis=0)
        wcost = col_energy_for_groups(energy, n_out, n_in, G64)
    rows = []

    def add(name, family, bits, g, nf16, extra, Wh, extra_d=None):
        rec = {
            "name": name,
            "family": family,
            "bits_nominal": bits,
            "group": g,
            "decode": decode_class(name),
            "complete_bpw": bpw_codes(n, g, bits, nf16, extra),
            "n_f16_per_group": nf16,
            "extra_f16": extra,
        }
        if family == "ternary":
            rec["complete_bpw_2bit_store"] = bpw_codes(n, g, 2.0, nf16, extra)
            rec["complete_bpw_packed5"] = bpw_codes(n, g, 1.6, nf16, extra)
            rec["complete_bpw_shannon"] = bpw_codes(n, g, math.log2(3.0), nf16, extra)
        rec.update(score(W, Wh, X_hold, Y_hold, R_hold))
        if extra_d:
            rec.update(extra_d)
        rows.append(rec)
        log(f"    {name} cos={rec.get('hold_output_cosine'):.6f} bpw={rec['complete_bpw']:.4f} {rec['decode']}")

    add("binary_meanabs_g128", "1bit", 1.0, G128, 1, 0, recon_binary_meanabs(W, G128))
    add("ternary_t0.7_g128", "ternary", 2.0, G128, 2, 0, recon_ternary_t07(W, G128, 0.7))
    add("ternary_sym_act_g64", "ternary", math.log2(3.0), G64, 1, 0, recon_ternary_sym(W, G64, "act", wcost))
    add("uniform_q2_clip_g64", "2bit", 2.0, G64, 1, 0, recon_uniform_clip(W, 2, G64))
    add("pow2_m1012_g64", "2bit", 2.0, G64, 1, 0, recon_fixed_grid(W, G64, np.array([-1, 0, 1, 2], dtype=np.float32), wcost))
    add("power_q2_p1.35_g64", "2bit", 2.0, G64, 1, 0, recon_levels_search(W, power_levels(4, P_STAR), G64, wcost), {"p": P_STAR})
    Wh, lev = recon_learned_shared(W, 4, G64, wcost, scale_search=True)
    add("learned_4_act_g64", "2bit", 2.0, G64, 1, 4, Wh, {"levels_norm": lev})
    add("uniform_q3_clip_g64", "3bit", 3.0, G64, 1, 0, recon_uniform_clip(W, 3, G64))
    add("power_q3_p1.35_g64", "3bit", 3.0, G64, 1, 0, recon_levels_search(W, power_levels(8, P_STAR), G64, wcost), {"p": P_STAR})
    add("mulaw_q3_mu2_g64", "3bit", 3.0, G64, 1, 0, recon_levels_search(W, mu_levels(8, MU_STAR), G64, wcost), {"mu": MU_STAR})
    Wh, lev = recon_learned_shared(W, 8, G64, wcost, scale_search=True)
    add("learned_8_act_g64", "3bit", 3.0, G64, 1, 8, Wh, {"levels_norm": lev})
    if want_lloyd:
        add("lloyd_4_act_g64", "2bit", 2.0, G64, 4, 0, recon_lloyd(W, 4, G64, wcost))
        add("lloyd_8_act_g64", "3bit", 3.0, G64, 8, 0, recon_lloyd(W, 8, G64, wcost))
    return rows


PHASE1_SLIM = [
    (0, "mlp.down_proj.weight", "swiglu", True),
    (0, "linear_attn.in_proj_qkv.weight", "hidden", False),
    (0, "linear_attn.out_proj.weight", "mixer", True),
    (3, "self_attn.q_proj.weight", "hidden", False),
    (3, "self_attn.o_proj.weight", "mixer", False),
    (31, "mlp.up_proj.weight", "hidden", False),
    (31, "mlp.down_proj.weight", "swiglu", False),
    (54, "mlp.down_proj.weight", "swiglu", False),
    (58, "mlp.down_proj.weight", "swiglu", True),
    (62, "mlp.down_proj.weight", "swiglu", True),
    (63, "mlp.gate_proj.weight", "hidden", False),
    (63, "mlp.down_proj.weight", "swiglu", True),
    (63, "self_attn.o_proj.weight", "mixer", False),
]


def already_done(label: str) -> bool:
    return (OUT / f"phase1_{label.replace('.', '_')}.json").exists()


def eval_one(layer: int, suffix: str, site: str, want_lloyd: bool, tag: str) -> dict:
    label = f"L{layer}.{suffix}"
    log(f"{tag} {label} site={site} lloyd={want_lloyd}")
    t0 = time.time()
    Xh = load_hidden(layer)
    X = load_site_x(layer, site, Xh)
    W = load_tensor(tname(layer, suffix))
    X_fit, X_hold = X[:FIT_N], X[FIT_N:]
    Y_hold = X_hold @ W.T
    R_hold = Xh[FIT_N:] if Y_hold.shape[1] == HIDDEN else None
    rec = {
        "label": label,
        "layer": layer,
        "suffix": suffix,
        "site": site,
        "shape": [int(x) for x in W.shape],
        "x_shape": [int(x) for x in X.shape],
        "rows_per_dim": float(X.shape[0]) / float(W.shape[1]),
        "weight_stats": weight_stats(W),
        "variants": run_slim(W, X_fit, X_hold, Y_hold, R_hold, want_lloyd=want_lloyd),
        "wall_s": time.time() - t0,
        "rss_gb": rss_gb(),
        "slim": True,
    }
    path = OUT / f"phase1_{label.replace('.', '_')}.json"
    path.write_text(json.dumps(rec))
    del W, X, Xh, Y_hold
    return rec


def run_phase2_variants(W, X_fit, X_hold, Y_hold, R_hold) -> list[dict]:
    """Faster 6-method set for the 64-down census."""
    n_out, n_in = W.shape
    n = int(W.size)
    energy = np.sum(np.square(X_fit, dtype=np.float64), axis=0)
    wcost = col_energy_for_groups(energy, n_out, n_in, G64)
    rows = []

    def add(name, family, bits, g, nf16, extra, Wh, extra_d=None):
        rec = {
            "name": name,
            "family": family,
            "bits_nominal": bits,
            "group": g,
            "decode": decode_class(name),
            "complete_bpw": bpw_codes(n, g, bits, nf16, extra),
        }
        rec.update(score(W, Wh, X_hold, Y_hold, R_hold))
        if extra_d:
            rec.update(extra_d)
        rows.append(rec)

    add("binary_meanabs_g128", "1bit", 1.0, G128, 1, 0, recon_binary_meanabs(W, G128))
    add("ternary_sym_act_g64", "ternary", math.log2(3.0), G64, 1, 0, recon_ternary_sym(W, G64, "act", wcost))
    add("uniform_q2_clip_g64", "2bit", 2.0, G64, 1, 0, recon_uniform_clip(W, 2, G64))
    add("power_q2_p1.35_g64", "2bit", 2.0, G64, 1, 0, recon_levels_search(W, power_levels(4, P_STAR), G64, wcost), {"p": P_STAR})
    add("uniform_q3_clip_g64", "3bit", 3.0, G64, 1, 0, recon_uniform_clip(W, 3, G64))
    add("power_q3_p1.35_g64", "3bit", 3.0, G64, 1, 0, recon_levels_search(W, power_levels(8, P_STAR), G64, wcost), {"p": P_STAR})
    return rows


def phase2_downs() -> list[dict]:
    out = []
    for layer in range(64):
        log(f"PHASE2 down L{layer}")
        t0 = time.time()
        Xh = load_hidden(layer)
        X = down_x(Xh, layer)
        W = load_tensor(tname(layer, "mlp.down_proj.weight"))
        X_fit, X_hold = X[:FIT_N], X[FIT_N:]
        Y_hold = X_hold @ W.T
        R_hold = Xh[FIT_N:]
        rec = {
            "label": f"L{layer}.mlp.down_proj.weight",
            "layer": layer,
            "organ": "down_proj",
            "suffix": "mlp.down_proj.weight",
            "site": "swiglu",
            "shape": [int(x) for x in W.shape],
            "rows_per_dim": float(X.shape[0]) / float(W.shape[1]),
            "variants": run_phase2_variants(W, X_fit, X_hold, Y_hold, R_hold),
            "wall_s": time.time() - t0,
            "rss_gb": rss_gb(),
        }
        out.append(rec)
        (OUT / "phase2.json").write_text(json.dumps({"tensors": out}))
        by = {v["name"]: v["hold_output_cosine"] for v in rec["variants"]}
        rp = {v["name"]: v.get("residual_proxy_cosine") for v in rec["variants"]}
        log(
            f"  L{layer} q3={by['uniform_q3_clip_g64']:.5f} p3={by['power_q3_p1.35_g64']:.5f} "
            f"q2={by['uniform_q2_clip_g64']:.5f} p2={by['power_q2_p1.35_g64']:.5f} "
            f"ter={by['ternary_sym_act_g64']:.5f} bin={by['binary_meanabs_g128']:.5f} "
            f"res_p3={rp['power_q3_p1.35_g64']}"
        )
        del W, X, Xh, Y_hold
    return out


def main() -> None:
    t0 = time.time()
    log("START slim follow-up")
    p1 = []
    for layer, suffix, site, lloyd in PHASE1_SLIM:
        label = f"L{layer}.{suffix}"
        path = OUT / f"phase1_{label.replace('.', '_')}.json"
        if path.exists():
            old = json.loads(path.read_text())
            # rerun if the existing file is the slow full sweep but we still want slim? keep full
            log(f"SKIP existing {label} nvar={len(old.get('variants', []))}")
            p1.append(old)
            continue
        p1.append(eval_one(layer, suffix, site, lloyd, "PHASE1S"))
    (OUT / "phase1_slim.json").write_text(json.dumps({"tensors": [r["label"] for r in p1]}))
    p2 = phase2_downs()
    summary = {
        "schema": "hawking.g1.learned_levels.slim.v1",
        "wall_s": time.time() - t0,
        "rss_max_gb": rss_gb(),
        "p_star": P_STAR,
        "mu_star": MU_STAR,
        "phase1_n": len(p1),
        "phase2_n": len(p2),
    }
    (OUT / "summary_slim.json").write_text(json.dumps(summary, indent=2))
    log(f"DONE slim wall={summary['wall_s']:.1f}s rss={summary['rss_max_gb']:.3f}G")


if __name__ == "__main__":
    main()

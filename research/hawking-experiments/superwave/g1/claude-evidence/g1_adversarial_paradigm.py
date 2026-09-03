#!/usr/bin/env python3
"""Adversarial-paradigm measurements. CPU only. No GPU. No pack. No generate."""
from __future__ import annotations

import json
import os
import resource
import struct
import time
from pathlib import Path

import numpy as np

BF16 = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16")
ACT = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1")
G0_MANIFEST = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json"
)
OUT = Path("/tmp/g1_adversarial_paradigm.json")

N = 26_895_998_464
E_MLP = 17_112_760_320
E_ATTN = 7_237_795_840
E_TAB = 2_542_796_800
E_SMALL = 2_645_504
assert E_MLP + E_ATTN + E_TAB + E_SMALL == N

GB_S = 639.2522341137478  # sealed weight_addressing, CITED TOKEN_NS_QWEN38
NS_PER_BYTE = 1e9 / (GB_S * 1e9)

# CITED contract screen products
PROD = {
    "G0": 0.4078534106896186,
    "q3mlp": 0.009305905311825565,
    "q4down": 8.359763329144191e-10,
    "mixed2p0": 2.3203991109789943e-15,
    "q4_mse_g128": 0.47754866875899726,
}


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


def complete(b_mlp, b_attn, b_tab, b_small=32.00853977162764) -> float:
    return (E_MLP * b_mlp + E_ATTN * b_attn + E_TAB * b_tab + E_SMALL * b_small) / N


def load_bf16(shard: Path, name: str) -> np.ndarray:
    with open(shard, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        info = header[name]
        start, end = info["data_offsets"]
        f.seek(8 + header_len + start)
        raw = f.read(end - start)
    u16 = np.frombuffer(raw, dtype="<u2").copy()
    f32 = np.empty(u16.shape[0], dtype=np.float32)
    # bf16 -> f32 via left-shift; view as f32
    tmp = u16.astype(np.uint32) << 16
    f32 = tmp.view(np.float32).reshape(info["shape"])
    return np.array(f32, dtype=np.float32, copy=True)


def x_stats(path: Path) -> dict:
    x = np.fromfile(path, dtype="<f4").reshape(256, 5120)
    # economy SVD
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    energy = s.astype(np.float64) ** 2
    tot = float(energy.sum())
    csum = np.cumsum(energy)
    # numerical rank: s > max(s)*eps*max(m,n) * 10 (generous)
    eps = np.finfo(np.float32).eps
    thresh = float(s[0]) * eps * max(x.shape) * 10.0
    rank = int((s > thresh).sum())
    # also a 1e-4 relative energy rank
    rank_1e4 = int((csum / tot < 1.0 - 1e-4).sum()) + 1
    return {
        "shape": list(x.shape),
        "rms": float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))),
        "s0": float(s[0]),
        "s_last": float(s[-1]),
        "s0_over_s255": float(s[0] / s[-1]) if s[-1] > 0 else None,
        "energy_frac_top8": float(energy[:8].sum() / tot),
        "energy_frac_top32": float(energy[:32].sum() / tot),
        "energy_frac_top64": float(energy[:64].sum() / tot),
        "energy_frac_top128": float(energy[:128].sum() / tot),
        "rank_f32_thresh": rank,
        "rank_energy_1e-4": min(rank_1e4, 256),
        "null_dim_hidden": 5120 - 256,
        "rows_per_dim": 256 / 5120,
        "invisible_frac_if_isotropic": 1.0 - 256 / 5120,
    }


def goodhart_gate() -> dict:
    name = "language_model.model.layers.0.mlp.gate_proj.weight"
    shard = BF16 / "model-00001-of-00011.safetensors"
    W = load_bf16(shard, name)  # [17408, 5120]
    X = np.fromfile(ACT / "hidden" / "L00.f32", dtype="<f4").reshape(256, 5120)
    # hold split matches descent: last 64 of 256
    X_hold = X[192:]
    # SVD of full 256-row capture (the screen's entire X)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = 256
    V = Vt[:r]  # [256, 5120]
    # W_vis = (W @ V.T) @ V   visible to the 256-token span
    WV = W @ V.T  # [17408, 256]
    W_vis = WV @ V
    W_null = W - W_vis
    nW = float(np.linalg.norm(W, ord="fro"))
    nVis = float(np.linalg.norm(W_vis, ord="fro"))
    nNull = float(np.linalg.norm(W_null, ord="fro"))
    # output cosines
    Y = X_hold @ W.T
    Y_vis = X_hold @ W_vis.T
    Y_null = X_hold @ W_null.T

    def cos_mat(A, B) -> float:
        a = A.reshape(-1).astype(np.float64)
        b = B.reshape(-1).astype(np.float64)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    def row_cos_mean(A, B) -> float:
        # mean row cosine
        acc = 0.0
        for i in range(A.shape[0]):
            a = A[i].astype(np.float64)
            b = B[i].astype(np.float64)
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            acc += float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0
        return acc / A.shape[0]

    # random probe in the same ambient space (not the capture)
    rng = np.random.default_rng(160)
    X_rand = rng.standard_normal((64, 5120), dtype=np.float64).astype(np.float32)
    Y_rand = X_rand @ W.T
    Y_rand_vis = X_rand @ W_vis.T

    # destroy-visible keep-null: the complementary Goodhart fail
    return {
        "tensor": name,
        "W_shape": list(W.shape),
        "||W||_F": nW,
        "||W_vis||_F": nVis,
        "||W_null||_F": nNull,
        "vis_energy_frac": (nVis / nW) ** 2,
        "null_energy_frac": (nNull / nW) ** 2,
        "isotropic_vis_expectation": 256 / 5120,
        "weight_cosine(W, W_vis)": nVis / nW,
        "weight_cosine(W, W_null)": nNull / nW,
        "hold_flat_cos(W, W_vis)": cos_mat(Y, Y_vis),
        "hold_row_cos(W, W_vis)": row_cos_mean(Y, Y_vis),
        "hold_flat_cos(W, W_null)": cos_mat(Y, Y_null),
        "hold_row_cos(W, W_null)": row_cos_mean(Y, Y_null),
        "rand64_flat_cos(W, W_vis)": cos_mat(Y_rand, Y_rand_vis),
        "rand64_row_cos(W, W_vis)": row_cos_mean(Y_rand, Y_rand_vis),
        "note": "W_vis is the unique projection of W onto the 256-token row-span of X. X@W.T == X@W_vis.T algebraically. Random X is iid N(0,1), not a generate claim.",
    }


def screen_stats() -> dict:
    out = {}
    for n in (192, 196, 208, 304, 402, 496, 498, 755, 851):
        row = {}
        for k, p in PROD.items():
            if p <= 0:
                row[k] = None
                continue
            row[k] = float(p ** (1.0 / n))
        out[str(n)] = row
    # implied n from contract 0.9953 and 0.9764
    out["implied_n_from_G0_0.9953"] = float(np.log(PROD["G0"]) / np.log(0.9953))
    out["implied_n_from_q3mlp_0.9764"] = float(np.log(PROD["q3mlp"]) / np.log(0.9764))
    out["bracket_ratio_G0_over_q3mlp"] = PROD["G0"] / PROD["q3mlp"]
    out["any_threshold_in_(q3mlp, G0)_is_unidentified"] = True
    # four negatives + one positive cannot locate a cut
    # any strictly monotone f with f(G0)>f(neg) "fits"
    negs = [PROD["q3mlp"], PROD["q4down"], PROD["mixed2p0"]]
    out["named_negatives_in_contract"] = ["q3mlp", "q4down", "mixed2p0"]
    out["named_positive"] = ["G0"]
    out["q4_mse_g128_beats_G0_on_product"] = PROD["q4_mse_g128"] > PROD["G0"]
    out["product_ordering"] = sorted(PROD.items(), key=lambda kv: kv[1], reverse=True)
    # if q3mlp generate is COHERENT, the original 4neg/1pos fit is stale
    out["if_q3mlp_coherent_then_two_positives_span"] = PROD["G0"] / PROD["q3mlp"]
    return out


def dictionary_accounting() -> dict:
    # classic shared codebook on one 17408x5120 gate, then 64 of them
    m, kdim = 17408, 5120
    n = m * kdim
    cases = []
    for d, K in ((2, 256), (4, 256), (4, 1024), (8, 256), (8, 1024), (16, 256), (32, 256)):
        assert kdim % d == 0
        n_sub = n // d
        index_bits = n_sub * np.log2(K)
        cb_bytes = K * d * 2  # fp16
        # one tensor
        bpw_one = (index_bits + 8 * cb_bytes) / n
        # 64 tensors, one shared book
        bpw_64 = (64 * index_bits + 8 * cb_bytes) / (64 * n)
        cases.append(
            {
                "d": d,
                "K": K,
                "index_bpw": float(np.log2(K) / d),
                "cb_bytes": cb_bytes,
                "bpw_one_tensor_incl_cb": float(bpw_one),
                "bpw_64_shared_cb": float(bpw_64),
                "cb_bpw_on_one": float(8 * cb_bytes / n),
                "cb_bpw_on_64": float(8 * cb_bytes / (64 * n)),
            }
        )
    # shared hidden basis rank r, 64 gate layers: V is kdim x r, coeffs m x r, both f16
    basis = []
    for r in (8, 32, 64, 160, 256):
        v_bytes = kdim * r * 2
        c_bytes = m * r * 2
        factor_bpw_shared_v = 8 * (v_bytes + 64 * c_bytes) / (64 * n)
        factor_bpw_per_layer = 8 * (v_bytes + c_bytes) / n
        # residual still ~ (1 - e_r) of mass; if residual kept at Q4 4.25:
        # use shared-basis residuals from g1-shared-basis (CITED)
        basis.append(
            {
                "r": r,
                "shared_V_bytes": v_bytes,
                "per_layer_coeff_bytes": c_bytes,
                "factor_bpw_shared_V_64gates": float(factor_bpw_shared_v),
                "factor_bpw_per_layer_V": float(factor_bpw_per_layer),
            }
        )
    # alignment: one 5120x5120 f16 rotation per layer
    rot_bytes = 64 * kdim * kdim * 2
    rot_bpw_on_N = 8 * rot_bytes / N
    rot_bpw_on_gates = 8 * rot_bytes / (64 * n)
    return {
        "pq_family": cases,
        "shared_basis_factors_f16": basis,
        "per_layer_hidden_rotation_f16": {
            "bytes": rot_bytes,
            "complete_bpw_on_N": float(rot_bpw_on_N),
            "bpw_on_64_gates": float(rot_bpw_on_gates),
            "note": "A runtime basis change that is not folded must stream this or apply it as a 5120x5120 GEMV per use.",
        },
    }


def generator_vs_bandwidth() -> dict:
    # time to read Q4 of one gate vs eval of rank-r generator
    gate_n = 17408 * 5120
    q4_bytes = 40 + (gate_n // 64) * (32 + 2)  # header + codes + scales
    t_q4_ns = q4_bytes * NS_PER_BYTE * 1e9 / 1e0
    # NS_PER_BYTE is seconds/byte; convert
    t_q4_s = q4_bytes / (GB_S * 1e9)
    rows = []
    for r in (1, 8, 64, 160, 256):
        factor_bytes = r * (17408 + 5120) * 2  # f16 A,B
        t_fac_s = factor_bytes / (GB_S * 1e9)
        # FLOPs: R@x is r*K, L@(.) is M*r; vs dense M*K
        flops_gen = r * (17408 + 5120)
        flops_dense = 17408 * 5120
        rows.append(
            {
                "r": r,
                "factor_bytes": factor_bytes,
                "factor_read_s_at_639": t_fac_s,
                "factor_read_us": t_fac_s * 1e6,
                "q4_read_us": t_q4_s * 1e6,
                "read_saving_us_if_no_residual": (t_q4_s - t_fac_s) * 1e6,
                "flops_gen": flops_gen,
                "flops_dense": flops_dense,
                "flops_ratio": flops_gen / flops_dense,
                "factor_bpw": 8 * factor_bytes / gate_n,
            }
        )
    # PQ reject geometry
    llama_mn = 14336 * 4096
    llama_q4_bytes = llama_mn * 4.25 / 8
    llama_q4_us = llama_q4_bytes / (GB_S * 1e9) * 1e6
    pq_us = 460.041
    attn_elems = 7_214_202_880  # GEMV projections, CITED g1-vector-quantization
    attn_q4_us = (attn_elems * 4.25 / 8) / (GB_S * 1e9) * 1e6
    attn_pq_proj_us = (attn_elems / llama_mn) * pq_us
    return {
        "GB_s": GB_S,
        "ns_per_byte": 1e9 / (GB_S * 1e9) * 1e9,
        "ps_per_f16": 2 / (GB_S * 1e9) * 1e12,
        "gate_q4_bytes": q4_bytes,
        "gate_q4_read_us": t_q4_s * 1e6,
        "rank_sweep": rows,
        "pq_reject": {
            "geometry": "14336x4096",
            "q4_bytes": llama_q4_bytes,
            "q4_read_us_at_639": llama_q4_us,
            "measured_1stage_pq_us": pq_us,
            "pq_over_q4_read": pq_us / llama_q4_us,
            "attn_elems": attn_elems,
            "attn_q4_read_us_at_639": attn_q4_us,
            "attn_if_every_gemv_pays_460us": attn_pq_proj_us,
        },
        "q3mlp_token_vs_G0": {
            "q3mlp_wall_ns": 148_588_917,
            "G0_TOKEN_NS": 39_326_090,
            "ratio": 148_588_917 / 39_326_090,
            "note": "Lower complete BPW (3.614 vs 4.253) is SLOWER. Density is not token cost. simd3 path.",
        },
    }


def conditional_simd() -> dict:
    # 32-wide simdgroup, island fraction f
    rows = []
    for f in (1e-5, 1e-4, 1e-3, 0.01, 0.03, 0.1):
        p_div32 = 1.0 - (1.0 - f) ** 32
        p_div64 = 1.0 - (1.0 - f) ** 64  # TPR64 row
        # expected extra path executions if divergent group runs both
        extra32 = p_div32  # fraction of groups that pay a second path
        rows.append(
            {
                "island_frac": f,
                "P_simd32_diverges": p_div32,
                "P_tpr64_row_has_island": p_div64,
                "bytes_avoided_vs_q4_if_rest_binary": f * (4.25 - 1.125),
            }
        )
    # rice 1% index cost from sparse-islands (CITED, recomputed formula)
    n = 10240 * 5120
    k = int(0.01 * n)
    val_bpw = 16.0 * k / n
    # rice of gaps at density 0.01 is ~0.082 CITED; also fixed log2 n
    fixed_idx_bpw = k * np.log2(n) / n
    return {
        "simdgroup": 32,
        "tpr64_threads_per_row": 64,
        "divergence_table": rows,
        "L0_qkv_1pct_island": {
            "n": n,
            "k": k,
            "value_bf16_bpw": val_bpw,
            "fixed_log2n_index_bpw": float(fixed_idx_bpw),
            "rice_index_bpw_CITED": 0.08243,
            "scheme_bpw_binary_plus_rice_CITED": 1.3669,
            "omit_index_lie": "quoting 1.125+0.16=1.285 hides 0.082 rice; bitmap would hide 1.0",
        },
        "in_kernel_if_row_3994": "KILLS TPR64 word-load; 8-unpack of group 62 contains col 3994 as local 26",
    }


def arithmetic() -> dict:
    b_small = 32.00853977162764
    b_tab_q4 = 4.250000251691366
    b_attn_q4 = 4.250009196169866
    g0 = 8 * 14_297_694_680 / N
    cells = {}
    for name, (bm, ba, bt) in {
        "G0_all_q4": (4.250003590303309, b_attn_q4, b_tab_q4),
        "mlp0_attnQ4_tabQ4": (0.0, b_attn_q4, b_tab_q4),
        "mlp0_attn0_tabQ4": (0.0, 0.0, b_tab_q4),
        "mlp0_attn0_tab0_small32": (0.0, 0.0, 0.0),
        "tab_bf16_rest0": (0.0, 0.0, 16.0),
        "tab_f32_rest0": (0.0, 0.0, 32.0),
        "mixed2p0": (0.8480504639008466, b_attn_q4, b_tab_q4),
        "q3mlp": (3.2500251321231617, 4.25009205012337, 4.250001799593266),
        "hadamardQ4_attn_mlpQ4_tabQ4": (4.25, 4.125, 4.25),
        "hadamardQ4_attn_only_delta_from_G0": (4.250003590303309, 4.125, b_tab_q4),
        "island_k1_on_q3_body": (3.25 + 0.00249 * N / E_MLP, 3.25, 3.25),  # placeholder
        "attn_to_0_rest_G0": (4.250003590303309, 0.0, b_tab_q4),
        "tables_to_0_rest_G0": (4.250003590303309, b_attn_q4, 0.0),
        "mlp_to_0_rest_G0": (0.0, b_attn_q4, b_tab_q4),
        "target_0p7_mlp0_attn0_tabQ4": (0.0, 0.0, b_tab_q4),
        "target_1p5_mlp0p848_tabQ4_attn2p064": (0.8480504639008466, 2.064157091228481, b_tab_q4),
        "hetero15_all_binary_small32": (1.125, 1.125, 1.125),
        "entropy_shannon_G0_indices": (3.732, 3.732, 3.732),  # whole-model cited ceiling
    }.items():
        cells[name] = complete(bm, ba, bt, b_small)

    # inversion: max b_attn for complete < 1.5
    inv = {}
    ceiling_bits = 1.5 * N
    small_bits = 8 * 10_584_840
    tab_bits = 8 * 1_350_860_880
    for bm in (0.0, 0.131617, 0.5, 0.8, 0.8480504639008466, 1.0, 1.125, 3.2500251321231617):
        rem = ceiling_bits - E_MLP * bm - tab_bits - small_bits
        inv[str(bm)] = rem / E_ATTN

    # small-class max save from Q4 to 0
    small_classes = {
        "gqa.k_proj": 83_886_080,
        "gqa.v_proj": 83_886_080,
        "gqa.k+v": 167_772_160,
        "dn.in_proj_a+b": 23_592_960,
        "dn.conv1d": 1_966_080,
        "all_small_f32": E_SMALL,
        "channel_3994_k1_elems": 5_252_608,
        "lm_head": 1_271_398_400,
        "embed": 1_271_398_400,
        "tables": E_TAB,
        "attention_GEMV": E_ATTN,
        "one_gate_tensor": 17_408 * 5_120,
        "all_64_gate": E_MLP // 3,
    }
    saves = {
        name: {"elems": e, "frac": e / N, "save_Q4_to_0_bpw": 4.25 * e / N, "save_bf16_to_0_bpw": 16.0 * e / N}
        for name, e in small_classes.items()
    }
    # hadamard 0.125 on attention
    had = 0.125 * E_ATTN / N
    # entropy 0.521 on GEMV codes
    ent = 0.5211 * (N - E_SMALL) / N
    return {
        "N": N,
        "mass_frac": {"mlp": E_MLP / N, "attn": E_ATTN / N, "tab": E_TAB / N, "small": E_SMALL / N},
        "G0_complete_from_bytes": g0,
        "cells": cells,
        "max_b_attn_for_complete_lt_1p5_tabQ4": inv,
        "small_class_saves": saves,
        "hadamard_Q4_complete_delta": had,
        "entropy_0p521_complete_delta": ent,
        "kill_tab_bf16": cells["tab_bf16_rest0"] > 1.5,
        "kill_crush_mlp_only": cells["mlp0_attnQ4_tabQ4"] > 1.5,
        "kill_0p7_without_touching_tables_if_mlp_attn_zero": cells["target_0p7_mlp0_attn0_tabQ4"],
    }


def sidecar() -> dict:
    # mixed-q3mlp leftover 2p0 MLP payloads
    leftover_q3mlp = 1_814_060_541  # CITED g1-bracket-bisection
    leftover_q4down = 93_847_197
    catalog_q3mlp = 180_124
    scale_plane = 800_686_080
    vision_elems = 460_730_096
    return {
        "q3mlp_leftover_2p0_mlp_bytes_in_dir": leftover_q3mlp,
        "q3mlp_leftover_bpw_if_counted": 8 * leftover_q3mlp / N,
        "q3mlp_leftover_in_catalog_nbytes": False,
        "q4down_leftover_S01_down_bytes": leftover_q4down,
        "q4down_leftover_bpw_if_counted": 8 * leftover_q4down / N,
        "catalog_q3mlp_bytes": catalog_q3mlp,
        "catalog_bpw": 8 * catalog_q3mlp / N,
        "G0_scale_plane_GEMV_bytes": scale_plane,
        "scale_plane_bpw": 8 * scale_plane / N,
        "scale_plane_is_dormant": False,
        "vision_elems_excluded": vision_elems,
        "vision_bf16_bytes": vision_elems * 2,
        "vision_if_loaded_bf16_not_in_language_BPW": True,
        "sub15_expand_to_Q4_reconstructed_vehicle": "sidecar that made generate unattributable; standing rule rejects",
    }


def main() -> None:
    t0 = time.time()
    report = {
        "schema": "hawking.g1.adversarial_paradigm.measure.v1",
        "identity": {
            "N": N,
            "E_MLP": E_MLP,
            "E_ATTN": E_ATTN,
            "E_TAB": E_TAB,
            "E_SMALL": E_SMALL,
            "G0_manifest_bpw": None,
        },
    }
    g0m = json.loads(G0_MANIFEST.read_text())
    report["identity"]["G0_manifest_bpw"] = g0m["complete_physical_bpw"]
    report["identity"]["G0_payload_bytes"] = g0m["tensor_payload_bytes"]
    report["identity"]["G0_elements"] = g0m["source_weight_elements"]
    report["identity"]["recompute_8b_over_n"] = 8 * g0m["tensor_payload_bytes"] / g0m["source_weight_elements"]

    report["arithmetic"] = arithmetic()
    report["screen"] = screen_stats()
    report["dictionary"] = dictionary_accounting()
    report["generator"] = generator_vs_bandwidth()
    report["conditional"] = conditional_simd()
    report["sidecar"] = sidecar()

    xs = {}
    for layer, fn in ((0, "L00.f32"), (6, "L06.f32"), (32, "L32.f32"), (63, "L63.f32")):
        xs[str(layer)] = x_stats(ACT / "hidden" / fn)
    report["capture_X"] = xs
    # L7 channel 3994 identically 0
    x7 = np.fromfile(ACT / "hidden" / "L07.f32", dtype="<f4").reshape(256, 5120)
    report["L7_ch3994"] = {
        "min": float(x7[:, 3994].min()),
        "max": float(x7[:, 3994].max()),
        "nnz": int(np.count_nonzero(x7[:, 3994])),
    }

    report["goodhart_L0_gate"] = goodhart_gate()
    report["wall_s"] = time.time() - t0
    report["rss_max_gb"] = rss_gb()
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False))
    print("WROTE", OUT, "wall", report["wall_s"], "rss", report["rss_max_gb"])
    # compact human dump of the numbers the report will cite
    print("=== ARITH ===")
    a = report["arithmetic"]
    print("mass", a["mass_frac"])
    print("G0", a["G0_complete_from_bytes"])
    for k, v in a["cells"].items():
        print(f"  {k:48s} {v:.12f}")
    print("max b_attn", a["max_b_attn_for_complete_lt_1p5_tabQ4"])
    print("hadamard delta", a["hadamard_Q4_complete_delta"])
    print("entropy delta", a["entropy_0p521_complete_delta"])
    print("=== SCREEN geom means n=192 ===")
    print(report["screen"]["192"])
    print("implied n", report["screen"]["implied_n_from_G0_0.9953"], report["screen"]["implied_n_from_q3mlp_0.9764"])
    print("bracket", report["screen"]["bracket_ratio_G0_over_q3mlp"])
    print("=== X rank ===")
    for k, v in xs.items():
        print(k, {kk: v[kk] for kk in ("rms", "rank_f32_thresh", "rank_energy_1e-4", "energy_frac_top32", "s0_over_s255", "invisible_frac_if_isotropic")})
    print("=== GOODHART ===")
    g = report["goodhart_L0_gate"]
    for k in (
        "vis_energy_frac",
        "null_energy_frac",
        "weight_cosine(W, W_vis)",
        "hold_flat_cos(W, W_vis)",
        "hold_row_cos(W, W_vis)",
        "hold_flat_cos(W, W_null)",
        "rand64_flat_cos(W, W_vis)",
        "rand64_row_cos(W, W_vis)",
    ):
        print(f"  {k} {g[k]}")
    print("=== GEN ===")
    print("ps/f16", report["generator"]["ps_per_f16"])
    print("gate q4 us", report["generator"]["gate_q4_read_us"])
    print("pq over q4", report["generator"]["pq_reject"]["pq_over_q4_read"])
    print("attn pq proj us", report["generator"]["pq_reject"]["attn_if_every_gemv_pays_460us"])
    print("q3/G0 token", report["generator"]["q3mlp_token_vs_G0"]["ratio"])
    print("=== SIDECAR ===")
    print(report["sidecar"])
    print("=== DICT rot ===")
    print(report["dictionary"]["per_layer_hidden_rotation_f16"])
    print("pq d8K256 one/64", report["dictionary"]["pq_family"][3])
    print("basis r256", report["dictionary"]["shared_basis_factors_f16"][4])


if __name__ == "__main__":
    main()

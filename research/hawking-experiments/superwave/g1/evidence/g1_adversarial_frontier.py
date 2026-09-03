#!/usr/bin/env python3
"""Adversarial measurements for Gravity-1 wave families.

CPU/numpy only. No GPU, no generate, no pack, no resident. Peak RSS tracked.
Writes /tmp/g1_adversarial_frontier.json
"""
from __future__ import annotations
import json, math, os, resource, sys, time
import numpy as np

ABS_BF16 = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
ABS_CAPTURE = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
N = 26_895_998_464
E_MLP = 17_112_760_320
E_ATTN = 7_237_795_840
E_TAB = 2_542_796_800
E_SMALL = 2_645_504
BANDWIDTH_GB_S = 639.2522341137478  # CITED TOKEN_NS_QWEN38 weight_addressing
SLACK_BYTES_Q3MLP = 1_814_060_541   # MEASURED g1-artifact-inventory.md

HERE = os.path.dirname(os.path.abspath(__file__))
# tools live in the worktree, not /tmp
WT = os.environ.get("G1_WT", os.getcwd())
sys.path.insert(0, os.path.join(WT, "tools"))

import gravity_doctor_gate as dg
import gravity_ir as gir
from gravity_doctor_gate import (
    axes, gate, c_uniform, c_faithful_q4, c_visible_subspace, _probe, _rowcos, _worst_unit,
)
from gravity_ir import (
    Program, quant_tensor, dense_tensor, shared_basis, sparse_correction,
    exact_island, generated_block, SOURCE_PARAM_COUNT,
)


def align_costs(n_in, n_out):
    perm_bits = math.lgamma(n_in + 1) / math.log(2)
    return {
        "permutation_bytes": perm_bits / 8,
        "sign_bytes": n_in / 8,
        "channel_scale_bytes": 2 * n_in,
        "dense_orthogonal_bytes": 2 * n_in * n_in,
    }


def spectrum_agreement(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    rel = float(np.max(np.abs(a - b) / (np.maximum(a, b) + 1e-9)))
    return cos, rel


def canon(W):
    W = W.copy()
    ci = np.argsort(-np.linalg.norm(W, axis=0))
    W = W[:, ci]
    ri = np.argsort(-np.linalg.norm(W, axis=1))
    W = W[ri]
    sgn = np.sign(W.sum(axis=0) + 1e-30)
    return W * sgn


def flatcos(A, B):
    a, b = A.ravel(), B.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

dg.BF16 = ABS_BF16
dg.CAPTURE = ABS_CAPTURE
_orig_load_tensor = dg.load_tensor
_orig_load_X = dg.load_X


def load_tensor(name, root=ABS_BF16):
    return _orig_load_tensor(name, root=root)


def load_X(layer, capture=ABS_CAPTURE):
    return _orig_load_X(layer, capture=capture)


dg.load_tensor = load_tensor
dg.load_X = load_X


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)


def now():
    return time.perf_counter()


def tag(v, kind="MEASURED"):
    return {"value": v, "tag": kind}


def entropy_from_counts(c):
    t = int(c.sum())
    if t == 0:
        return 0.0
    p = c[c > 0].astype(np.float64) / t
    return float(-(p * np.log2(p)).sum())


def fwht_last256(x):
    """Orthonormal FWHT along last axis; last dim must be 256."""
    x = x.copy()
    n = x.shape[-1]
    assert n == 256
    h = 1
    lead = int(np.prod(x.shape[:-1]))
    x = x.reshape(lead, n)
    while h < n:
        x = x.reshape(lead, n // (2 * h), 2, h)
        a = x[:, :, 0, :] + x[:, :, 1, :]
        b = x[:, :, 0, :] - x[:, :, 1, :]
        x = np.stack([a, b], axis=2)
        h *= 2
    x = x.reshape(lead, n) * (1.0 / np.sqrt(n))
    return x


def block_hadamard_in(W, block=256):
    """W_h = W @ H, H = blkdiag(H_block, ...). H is orthonormal, H^2 = I."""
    rows, cols = W.shape
    assert cols % block == 0
    nblk = cols // block
    X = W.reshape(rows, nblk, block)
    Y = fwht_last256(X)
    return Y.reshape(rows, cols)


def apply_block_H(X, block=256):
    """X_h = X @ H for activations (n, cols)."""
    n, cols = X.shape
    assert cols % block == 0
    nblk = cols // block
    return fwht_last256(X.reshape(n, nblk, block)).reshape(n, cols)


def rsvd(W, k=256, p=16, seed=0):
    """Randomized SVD; returns s (k,), Vh (k, n_in)."""
    rng = np.random.default_rng(seed)
    n_in = W.shape[1]
    G = rng.standard_normal((n_in, k + p)).astype(np.float32)
    Y = W @ G
    Q, _ = np.linalg.qr(Y, mode="reduced")
    B = Q.T @ W
    _, s, Vh = np.linalg.svd(B, full_matrices=False)
    return s[:k].astype(np.float64), Vh[:k]


def subspace_overlap(Va, Vb):
    """||Va Vb^T||_F^2 / k  with Va, Vb shape (k, n)."""
    k = Va.shape[0]
    M = Va @ Vb.T
    return float(np.square(M).sum() / k)


def rel_l2_spectra(sa, sb):
    a = sa / (np.linalg.norm(sa) + 1e-30)
    b = sb / (np.linalg.norm(sb) + 1e-30)
    return float(np.linalg.norm(a - b))


def quant_hist(W, bits=4, group=128):
    lim = (1 << (bits - 1)) - 1
    d = W.shape[1]
    codes = []
    Wh = W.astype(np.float32)
    for s in range(0, d - d % group, group):
        blk = Wh[:, s:s + group]
        amax = np.abs(blk).max(axis=1, keepdims=True) + 1e-30
        step = amax / lim
        q = np.clip(np.round(blk / step), -lim, lim).astype(np.int16)
        codes.append(q.ravel())
    q = np.concatenate(codes)
    offset = q - q.min()
    counts = np.bincount(offset.astype(np.int32), minlength=int(q.max() - q.min()) + 1)
    return entropy_from_counts(counts), counts, int(q.min()), int(q.max())


def nkp_residual(W, p, q, r, s, rank=8, seed=0):
    """Nearest sum-of-`rank` Kroneckers via unfolding SVD. W is (p*r, q*s)."""
    assert W.shape == (p * r, q * s)
    # rearrange to (p*q, r*s): W[a,b] with a=i*r+k, b=j*s+l -> M[i*q+j, k*s+l]
    T = W.reshape(p, r, q, s).transpose(0, 2, 1, 3).reshape(p * q, r * s)
    rng = np.random.default_rng(seed)
    k = min(rank, T.shape[0], T.shape[1])
    G = rng.standard_normal((T.shape[1], k + 8)).astype(np.float32)
    Y = T @ G
    Q, _ = np.linalg.qr(Y, mode="reduced")
    B = Q.T @ T
    Uhat, sig, Vh = np.linalg.svd(B, full_matrices=False)
    U = Q @ Uhat[:, :k]
    # reconstruct rank-k unfolding
    That = (U[:, :k] * sig[:k]) @ Vh[:k]
    What = That.reshape(p, q, r, s).transpose(0, 2, 1, 3).reshape(p * r, q * s)
    num = float(np.linalg.norm(W - What))
    den = float(np.linalg.norm(W)) + 1e-30
    params = k * (p * q + r * s)
    return {
        "factors": [p, q, r, s],
        "rank": int(k),
        "rel_fro": num / den,
        "params": int(params),
        "param_frac": params / W.size,
        "bytes_f16": int(params * 2),
        "bpw_on_tensor": 16.0 * params / W.size,
    }, What


def progressive_binary_planes(W, n_planes, per="row"):
    """Greedy binary (sign) planes with a scale. Returns list of (scale, P) recon."""
    R = W.astype(np.float32).copy()
    acc = np.zeros_like(R)
    out = []
    for i in range(n_planes):
        P = np.sign(R)
        P[P == 0] = 1.0
        if per == "row":
            s = (R * P).sum(1, keepdims=True) / (np.square(P).sum(1, keepdims=True) + 1e-30)
        elif per == "group128":
            s = np.zeros((R.shape[0], R.shape[1]), dtype=np.float32)
            g = 128
            for c0 in range(0, R.shape[1], g):
                blk = R[:, c0:c0 + g]
                Pb = P[:, c0:c0 + g]
                sc = (blk * Pb).sum(1, keepdims=True) / (np.square(Pb).sum(1, keepdims=True) + 1e-30)
                s[:, c0:c0 + g] = sc
            P = P  # noqa
            acc = acc + s * P
            R = W.astype(np.float32) - acc
            out.append(acc.copy())
            continue
        else:
            s = float((R * P).sum() / (np.square(P).sum() + 1e-30))
        acc = acc + s * P
        R = W.astype(np.float32) - acc
        out.append(acc.copy())
    return out


def progressive_ternary_planes(W, n_planes):
    """Greedy per-row ternary planes: q in {-1,0,1} * row-scale.

    Scale chosen as row RMS so the 0-bin is the natural inner third of
    a unit-scale ternary quantizer (threshold 0.5 after /s).
    """
    R = W.astype(np.float32).copy()
    acc = np.zeros_like(R)
    out = []
    sparsities = []
    for i in range(n_planes):
        rms = np.sqrt((R * R).mean(1, keepdims=True)) + 1e-30
        q = np.clip(np.round(R / rms), -1, 1)
        # refit scale on the chosen support
        supp = q != 0
        s = np.zeros_like(rms)
        num = (R * q).sum(1, keepdims=True)
        den = np.square(q).sum(1, keepdims=True) + 1e-30
        s = num / den
        acc = acc + s * q
        R = W.astype(np.float32) - acc
        out.append(acc.copy())
        sparsities.append(float(1.0 - supp.mean()))
    return out, sparsities


def plane_cost_bpw(elements, n_planes, kind, n_out, n_in, group=128):
    """Complete-accounting BPW of a plane ladder on ONE tensor, then the
    same structure applied to a named mass (caller multiplies)."""
    if kind == "binary_row":
        # 1 bit/weight/plane + f16 scale per row per plane + 40 B header/plane
        bits = n_planes * 1.0
        extra = n_planes * (2 * n_out + 40)
        return bits + 8 * extra / elements
    if kind == "binary_g128":
        groups = n_out * (n_in // group)
        extra = n_planes * (2 * groups + 40)
        return n_planes * 1.0 + 8 * extra / elements
    if kind == "ternary_dense2":
        # 2-bit dense ternary + row scale
        extra = n_planes * (2 * n_out + 40)
        return n_planes * 2.0 + 8 * extra / elements
    if kind == "ternary_sparse":
        # caller supplies nnz via sparsity
        raise ValueError("use plane_cost_ternary_sparse")
    raise ValueError(kind)


def plane_cost_ternary_sparse(elements, sparsities, n_out, index_bits=None):
    """Each plane stores nnz f16 values + indices. index_bits default log2(elements)."""
    if index_bits is None:
        index_bits = math.ceil(math.log2(elements))
    total_bytes = 0
    for sp in sparsities:
        nnz = (1.0 - sp) * elements
        total_bytes += nnz * 2 + nnz * index_bits / 8 + 40
    return 8 * total_bytes / elements


def mse_scale_q(W, bits, group, X_fit):
    """8-point MSE scale vs absmax, fitted on X_fit. Returns Wh."""
    lim = (1 << (bits - 1)) - 1
    multipliers = (0.50, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50, 2.00)
    d = W.shape[1]
    Wh = np.empty_like(W, dtype=np.float32)
    # Precompute per-group Gram of X (group x group) — shared across rows
    grams = []
    for s in range(0, d - d % group, group):
        Xg = X_fit[:, s:s + group]  # (t, g)
        grams.append(Xg.T @ Xg)     # (g, g)
    for s, G in zip(range(0, d - d % group, group), grams):
        blk = W[:, s:s + group].astype(np.float32)
        amax = np.abs(blk).max(axis=1, keepdims=True) + 1e-30
        s0 = amax / lim
        best_e = None
        best = None
        for m in multipliers:
            step = s0 * m
            q = np.clip(np.round(blk / step), -lim, lim)
            recon = q * step
            e = blk - recon
            # e^T G e per row: (e @ G) * e sum
            quad = np.einsum("ig,gh,ih->i", e, G, e, optimize=True)
            if best_e is None or True:
                if best is None:
                    best = recon.copy()
                    best_e = quad.copy()
                else:
                    take = quad < best_e
                    best[take] = recon[take]
                    best_e[take] = quad[take]
        Wh[:, s:s + group] = best
    return Wh


def stats_W(W):
    w = W.ravel()
    # subsample if huge
    if w.size > 5_000_000:
        rng = np.random.default_rng(0)
        w = w[rng.choice(w.size, 5_000_000, replace=False)]
    mu = float(w.mean())
    sd = float(w.std()) + 1e-30
    m4 = float(np.mean((w - mu) ** 4))
    kurt = m4 / (sd ** 4)
    rms = float(np.sqrt(np.mean(w * w)))
    mx = float(np.max(np.abs(w)))
    return {
        "mean": mu, "std": sd, "kurtosis": kurt, "max_over_rms": mx / (rms + 1e-30),
        "frac_gt_6rms": float(np.mean(np.abs(w) > 6 * rms)),
        "frac_gt_10rms": float(np.mean(np.abs(w) > 10 * rms)),
    }


def score_pack(W, Wh, X, ref):
    g = gate(W, Wh, X, ref=ref)
    return {
        "observed": g["observed"],
        "probed": g["probed"],
        "worst_unit": g["worst_unit"],
        "gate": g["gate"],
        "worst_axis": g.get("worst_axis"),
        "healthy": g["healthy"],
        "rel_fro": float(np.linalg.norm(W - Wh) / (np.linalg.norm(W) + 1e-30)),
        "energy_ratio": float(np.linalg.norm(Wh) / (np.linalg.norm(W) + 1e-30)),
        "weight_cos": float(flatcos(W, Wh)),
    }


def main():
    t0 = now()
    out = {
        "schema": "hawking.gravity1.adversarial_frontier.measure.v1",
        "N": N,
        "bandwidth_gb_s_cited": BANDWIDTH_GB_S,
        "rss_gb_start": rss_gb(),
    }

    # ------------------------------------------------------------------
    # 0. Arithmetic identities (exact on measured integers)
    # ------------------------------------------------------------------
    frac = {
        "mlp": E_MLP / N,
        "attn": E_ATTN / N,
        "tab": E_TAB / N,
        "small": E_SMALL / N,
    }
    b_small = 32.00853977162764  # MEASURED G0
    small_contrib = frac["small"] * b_small

    def complete(b_mlp, b_attn, b_tab):
        return frac["mlp"] * b_mlp + frac["attn"] * b_attn + frac["tab"] * b_tab + small_contrib

    # inversion: complete < 1.0
    #  f_m b_m + f_a b_a + f_t b_t < 1 - small
    budget = 1.0 - small_contrib
    inversions = {}
    # tables held at Q4 g64 = 4.25
    # f_m b_m + f_a b_a < budget - f_t*4.25
    rem_q4tab = budget - frac["tab"] * 4.25
    inversions["tables_q4_4.25"] = {
        "remainder_for_mlp_attn": rem_q4tab,
        "if_attn_stays_q4_max_b_mlp": None,  # will be negative
        "if_mlp_stays_q4_max_b_attn": None,
        "if_mlp_zero_max_b_attn": rem_q4tab / frac["attn"],
        "if_attn_zero_max_b_mlp": rem_q4tab / frac["mlp"],
        "if_equal_max_b": rem_q4tab / (frac["mlp"] + frac["attn"]),
        "mlp_only_to_zero_complete": complete(0.0, 4.25, 4.25),
        "attn_only_to_zero_complete": complete(4.25, 0.0, 4.25),
        "tab_only_to_zero_complete": complete(4.25, 4.25, 0.0),
        "mlp_and_attn_zero_tables_q4": complete(0.0, 0.0, 4.25),
    }
    inversions["tables_q4_4.25"]["if_attn_stays_q4_max_b_mlp"] = (
        (rem_q4tab - frac["attn"] * 4.25) / frac["mlp"]
    )
    inversions["tables_q4_4.25"]["if_mlp_stays_q4_max_b_attn"] = (
        (rem_q4tab - frac["mlp"] * 4.25) / frac["attn"]
    )
    inversions["all_equal_b_max"] = budget / (frac["mlp"] + frac["attn"] + frac["tab"])
    inversions["per_weight_1bit_g128"] = 1.0 + 16.0 / 128.0
    inversions["per_weight_unreachable"] = inversions["per_weight_1bit_g128"] > 1.0

    # class-only max save from G0 4.252735
    g0 = 4.252735126866492
    class_elems = {
        "mlp": E_MLP,
        "attn": E_ATTN,
        "tab": E_TAB,
        "gqa_kv": 83_886_080 * 2,
        "gqa_q": 1_006_632_960,
        "gqa_all": 1_677_721_600,
        "linear_attn": 5_562_051_072,
        "embed": 1_271_398_400,
        "lm_head": 1_271_398_400,
        "conv1d": 1_966_080,
        "in_proj_ba": 23_592_960,
        "self_attn_v": 83_886_080,
    }
    class_cap = {}
    for k, e in class_elems.items():
        f = e / N
        class_cap[k] = {
            "elements": e,
            "frac": f,
            "max_save_if_zero_from_4.25": f * 4.25,
            "complete_if_class_zero_rest_q4": g0 - f * 4.25 + f * 0.0,  # approx; small already in g0
            "capable_of_sub1_alone": (g0 - f * 4.25) < 1.0,
        }

    # IR floor: all 498 GEMV at 2b g128, tables held 6b, small f32
    # recount headers exactly
    p_floor = Program("allocator_floor_2bit")
    # 192 mlp + 208 fused attn? source 304 attn GEMV names; allocator inventory is 2-D language
    # 48*8 + 16*7 + 2 = 498 including tables
    # allocatable = 496 (no embed/lm_head curves)
    for n_t, e_each, bits, name in (
        (192, E_MLP // 192, 2, "mlp"),
        (48, 10_240 * 5_120, 2, "in_proj_qkv"),
        (48, 6_144 * 5_120, 2, "in_proj_z"),
        (48, 48 * 5_120, 2, "in_proj_a"),
        (48, 48 * 5_120, 2, "in_proj_b"),
        (48, 5_120 * 6_144, 2, "out_proj"),
        (16, 12_288 * 5_120, 2, "q_proj"),
        (16, 1_024 * 5_120, 2, "k_proj"),
        (16, 1_024 * 5_120, 2, "v_proj"),
        (16, 5_120 * 6_144, 2, "o_proj"),
        (2, E_TAB // 2, 6, "tables_held"),
    ):
        node = quant_tensor(e_each, bits, 128, "geo_tpr64_g128")
        for i in range(n_t):
            p_floor.add(f"{name}.{i}", e_each, [node])
    # small
    # 353 tensors, 2645504 elems total — use one dense blob + per-tensor headers via measured 10584840
    p_floor.add("small_f32", E_SMALL, [dense_tensor(E_SMALL, 4, "f32v2_direct", header=8 * 353)])
    # wait dense_tensor adds elements*4 + header, header=8*353 bills all small headers
    floor_bpw_ir = p_floor.complete_bpw()

    # simpler closed form matching allocator.bytes_at
    def bytes_at(elements, bits, group=128, header=40):
        return quant_tensor(elements, bits, group, "x", header=header).stored_bytes

    gemv_alloc_elems = E_MLP + E_ATTN  # 24350556160
    n_alloc = 496
    held_tab_bytes = 2 * bytes_at(E_TAB // 2, 6)
    alloc_2b = bytes_at(gemv_alloc_elems, 2, header=0) + 40 * n_alloc
    # bytes_at on the SUM under-counts groups? groups = elements//128, same as sum if each divisible
    small_bytes = 10_584_840
    floor_bytes = alloc_2b + held_tab_bytes + small_bytes
    floor_bpw = 8 * floor_bytes / N

    # all-498 at 2b
    all2_bytes = bytes_at(E_MLP + E_ATTN + E_TAB, 2, header=0) + 40 * 498 + small_bytes
    all2_bpw = 8 * all2_bytes / N

    slack_bpw = 8 * SLACK_BYTES_Q3MLP / N
    contract_slack = 0.539646522788
    contract_slack_bytes = contract_slack * N / 8

    out["arithmetic"] = {
        "frac": frac,
        "complete_fn_examples": {
            "g0_style_q4": complete(4.25, 4.25, 4.25),
            "mlp0_attn_tab_q4": complete(0.0, 4.25, 4.25),
            "attn0_mlp_tab_q4": complete(4.25, 0.0, 4.25),
            "tab0_mlp_attn_q4": complete(4.25, 4.25, 0.0),
            "mlp0_attn0_tab_q4": complete(0.0, 0.0, 4.25),
            "all_1.125": complete(1.125, 1.125, 1.125),
            "all_0.997": complete(0.997, 0.997, 0.997),
        },
        "inversions_sub1": inversions,
        "class_capability": class_cap,
        "allocator_floor": {
            "held_tables_6b_alloc_2b_bpw": floor_bpw,
            "held_tables_6b_alloc_2b_bytes": floor_bytes,
            "all_498_at_2b_bpw": all2_bpw,
            "ir_program_bpw": floor_bpw_ir,
            "note": "2.5065 is tables held at 6-bit g128 + 496 GEMVs at 2-bit + small f32",
        },
        "sidecar_q3mlp_slack": {
            "slack_bytes_cited_inventory": SLACK_BYTES_Q3MLP,
            "slack_bpw": slack_bpw,
            "contract_figure": contract_slack,
            "contract_figure_implies_bytes": contract_slack_bytes,
            "delta_bytes": contract_slack_bytes - SLACK_BYTES_Q3MLP,
        },
    }

    # alignment economics (exact)
    c5120 = align_costs(5120, 17408)
    out["alignment_costs_n_in_5120"] = {
        **c5120,
        "bpw_over_64_sites": {k: 8 * v * 64 / N for k, v in c5120.items()},
    }

    # IR priced mechanisms (description cost, no quality claim)
    def ir_shared_v(rank, coeff_bits, n_sites=64, rows=17408, cols=5120):
        p = Program(f"shared_V_r{rank}_c{coeff_bits}")
        cid = p.pool.put("SharedBasis", nbytes=rank * cols * 2, rank=rank, dtype="f16")
        # per-site: rows*rank coeffs at coeff_bits, plus optional residual not included
        for i in range(n_sites):
            p.add(f"s{i}", rows * cols, [
                shared_basis(rows * rank, coeff_bits, cid, "fused_basis_gemv"),
            ])
        # alignment perm+sign+scale per site
        align_b = n_sites * (c5120["permutation_bytes"] + c5120["sign_bytes"] + c5120["channel_scale_bytes"])
        return {
            "complete_bpw_coeffs_plus_V": p.complete_bpw(),
            "shared_bytes": p.total_bytes() - p.site_bytes(),
            "site_bytes": p.site_bytes(),
            "align_perm_sign_scale_bytes": align_b,
            "align_bpw": 8 * align_b / N,
            "total_with_align_bpw": p.complete_bpw() + 8 * align_b / N,
            "covers_sites": n_sites,
            "covers_elems": n_sites * rows * cols,
            "covers_frac": n_sites * rows * cols / N,
            "residual_not_billed": True,
        }

    out["ir_priced"] = {
        "shared_V_r32_c4_64gates": ir_shared_v(32, 4),
        "shared_V_r256_c4_64gates": ir_shared_v(256, 4),
        "shared_V_r256_c2_64gates": ir_shared_v(256, 2),
        "dense_ortho_64_sites_bpw": 8 * c5120["dense_orthogonal_bytes"] * 64 / N,
        "butterfly_angles_n4096": {
            "angles": 2048 * 12,
            "bytes_f16_per_site": 2048 * 12 * 2,
            "bpw_64_sites": 8 * 2048 * 12 * 2 * 64 / N,
        },
        "block_hadamard_256_description_bytes": 0,
        "monarch_b256_n5120_params": 2 * 5120 * 256,
        "monarch_b256_bytes_f16_per_site": 2 * 5120 * 256 * 2,
        "monarch_b256_bpw_64_sites": 8 * 2 * 5120 * 256 * 2 * 64 / N,
    }

    # ------------------------------------------------------------------
    # 1. Load L0 gate + X  (mass-dominant class, has capture)
    # ------------------------------------------------------------------
    print(f"load L0 gate  rss={rss_gb():.3f}G", flush=True)
    W = dg.load_tensor("language_model.model.layers.0.mlp.gate_proj.weight")
    X = dg.load_X(0)
    print(f"  W {W.shape} X {X.shape} rss={rss_gb():.3f}G", flush=True)
    assert X.shape[1] == W.shape[1] == 5120

    ref_W = c_faithful_q4(W, group=128)
    ref = axes(W, ref_W, X)
    out["L0_gate"] = {
        "shape": list(W.shape),
        "elements": int(W.size),
        "ref_q4_g128": {k: float(ref[k]) for k in ("observed", "probed", "worst_unit")},
        "x_rank": int(np.linalg.matrix_rank(X, tol=1e-3 * np.linalg.norm(X, 2))),
    }

    constructions = {}

    # honest Qn
    for bits in (2, 3, 4):
        Wh = c_uniform(W, bits, 128)
        constructions[f"uniform_q{bits}_g128"] = score_pack(W, Wh, X, ref)
        constructions[f"uniform_q{bits}_g128"]["bpw_ir"] = 8 * quant_tensor(W.size, bits, 128, "x").stored_bytes / W.size

    # visible subspace (incumbent screen Goodhart)
    Wvis = c_visible_subspace(W, X)
    constructions["visible_subspace_X_only"] = score_pack(W, Wvis, X, ref)

    # span-fit to X ∪ P_seed0  — Goodhart of the 3-axis gate
    P0 = _probe(W.shape[1], n=256, seed=0)
    P1 = _probe(W.shape[1], n=256, seed=1)
    XP = np.concatenate([X, P0], axis=0)
    # project rows of W (i.e. columns of W.T) onto row-span of XP
    # W_hat = W @ (B^T B) where B = top-r right singular of XP
    _, _, Vt = np.linalg.svd(XP.astype(np.float32), full_matrices=False)
    r_xp = int(np.linalg.matrix_rank(XP, tol=1e-3 * np.linalg.norm(XP, 2)))
    B = Vt[:r_xp]
    Wspan = W @ (B.T @ B)
    constructions["spanfit_X_and_P0"] = score_pack(W, Wspan, X, ref)
    # held-out probe (seed 1) — not inside axes()
    constructions["spanfit_X_and_P0"]["probed_heldout_seed1"] = _rowcos(P1 @ W.T, P1 @ Wspan.T)
    constructions["spanfit_X_and_P0"]["worst_unit_heldout"] = _worst_unit(P1 @ W.T, P1 @ Wspan.T)
    constructions["spanfit_X_and_P0"]["x_plus_p0_rank"] = r_xp

    # Q4-error rearrangement: permute output-row errors
    Eq = ref_W - W
    rng = np.random.default_rng(0)
    perm = rng.permutation(W.shape[0])
    Wperm = W + Eq[perm]
    constructions["q4_error_row_permuted"] = score_pack(W, Wperm, X, ref)
    constructions["q4_error_row_permuted"]["probed_heldout_seed1"] = _rowcos(P1 @ W.T, P1 @ Wperm.T)

    # same-split vs held-out function-fitted scales (Q3, more room to move)
    X_even, X_odd = X[0::2], X[1::2]
    Wmse_even = mse_scale_q(W, bits=3, group=128, X_fit=X_even)
    sc_same = score_pack(W, Wmse_even, X_even, axes(W, c_uniform(W, 3, 128), X_even))
    sc_hold = score_pack(W, Wmse_even, X_odd, axes(W, c_uniform(W, 3, 128), X_odd))
    sc_probe = score_pack(W, Wmse_even, X, ref)  # ref is Q4; also score vs Q3
    ref_q3 = axes(W, c_uniform(W, 3, 128), X)
    constructions["mse_q3_fit_even"] = {
        "on_even_same_split": sc_same,
        "on_odd_heldout": sc_hold,
        "on_full_vs_q4ref": sc_probe,
        "on_full_vs_q3ref": score_pack(W, Wmse_even, X, ref_q3),
        "absmax_q3_on_even": score_pack(W, c_uniform(W, 3, 128), X_even, axes(W, c_uniform(W, 3, 128), X_even)),
        "absmax_q3_on_odd": score_pack(W, c_uniform(W, 3, 128), X_odd, axes(W, c_uniform(W, 3, 128), X_odd)),
    }

    # Hadamard reshape then Qn, effective Wh = Q(W H) H
    print("  hadamard", flush=True)
    WH = block_hadamard_in(W, 256)
    out["L0_gate"]["weight_stats_native"] = stats_W(W)
    out["L0_gate"]["weight_stats_hadamard"] = stats_W(WH)
    H_X = apply_block_H(X, 256)
    for bits in (2, 3, 4):
        Qh = c_uniform(WH, bits, 128)
        # effective matrix on original x: Qh @ H, since y = Qh @ (H x) = (Qh H) x
        Weff = block_hadamard_in(Qh, 256)  # Qh @ H  (H self-inverse)
        constructions[f"hadamard256_then_q{bits}_g128"] = score_pack(W, Weff, X, ref)
        constructions[f"hadamard256_then_q{bits}_g128"]["bpw_codes"] = (
            8 * quant_tensor(W.size, bits, 128, "x").stored_bytes / W.size
        )
        constructions[f"hadamard256_then_q{bits}_g128"]["runtime_Hx_adds_per_token"] = 20 * 256 * 8  # FWHT butterflies
        constructions[f"hadamard256_then_q{bits}_g128"]["description_bytes"] = 0

    # entropy of Q4 codes native vs Hadamard
    H_nat, _, _, _ = quant_hist(W, 4, 128)
    H_had, _, _, _ = quant_hist(WH, 4, 128)
    H_nat2, _, _, _ = quant_hist(W, 2, 128)
    H_had2, _, _, _ = quant_hist(WH, 2, 128)
    out["L0_gate"]["index_entropy"] = {
        "q4_native": H_nat, "q4_hadamard": H_had,
        "q2_native": H_nat2, "q2_hadamard": H_had2,
        "ans_g64_flush": 32.0 / 64.0,
        "ans_g128_flush": 32.0 / 128.0,
        "ans_g256_flush": 32.0 / 256.0,
    }

    # progressive planes
    print("  planes", flush=True)
    bin_row = progressive_binary_planes(W, 6, per="row")
    bin_g = progressive_binary_planes(W, 4, per="group128")
    ter, ter_sp = progressive_ternary_planes(W, 4)
    plane_scores = {"binary_row": [], "binary_g128": [], "ternary_row": []}
    for i, acc in enumerate(bin_row, 1):
        sc = score_pack(W, acc, X, ref)
        sc["n_planes"] = i
        sc["bpw"] = plane_cost_bpw(W.size, i, "binary_row", W.shape[0], W.shape[1])
        plane_scores["binary_row"].append(sc)
    for i, acc in enumerate(bin_g, 1):
        sc = score_pack(W, acc, X, ref)
        sc["n_planes"] = i
        sc["bpw"] = plane_cost_bpw(W.size, i, "binary_g128", W.shape[0], W.shape[1])
        plane_scores["binary_g128"].append(sc)
    for i, acc in enumerate(ter, 1):
        sc = score_pack(W, acc, X, ref)
        sc["n_planes"] = i
        sc["sparsity"] = ter_sp[i - 1]
        sc["bpw_dense2"] = plane_cost_bpw(W.size, i, "ternary_dense2", W.shape[0], W.shape[1])
        sc["bpw_sparse"] = plane_cost_ternary_sparse(W.size, ter_sp[:i], W.shape[0])
        plane_scores["ternary_row"].append(sc)
    constructions["planes"] = plane_scores

    # Kronecker / tensor operator
    print("  kronecker", flush=True)
    # 17408=136*128, 5120=64*80
    nkp_jobs = [
        (136, 64, 128, 80, 1),
        (136, 64, 128, 80, 8),
        (136, 64, 128, 80, 32),
        (17, 16, 1024, 320, 8),   # skinny
        (68, 20, 256, 256, 8),    # head_dim aligned
    ]
    nkp_out = []
    for p, q, r, s, rk in nkp_jobs:
        if p * r != W.shape[0] or q * s != W.shape[1]:
            nkp_out.append({"factors": [p, q, r, s], "skip": "shape mismatch"})
            continue
        info, What = nkp_residual(W, p, q, r, s, rank=rk)
        info.update(score_pack(W, What.astype(np.float32), X, ref))
        # sequential GEMM count
        info["sequential_gemms"] = 2 * rk
        info["factor_read_bytes_f16"] = info["bytes_f16"]
        info["factor_read_ns_at_639"] = info["bytes_f16"] / (BANDWIDTH_GB_S * 1e9) * 1e9
        info["dense_q4_read_bytes"] = W.size / 2 + 2 * (W.size // 128)
        info["dense_q4_read_ns_at_639"] = info["dense_q4_read_bytes"] / (BANDWIDTH_GB_S * 1e9) * 1e9
        nkp_out.append(info)
        del What
    constructions["kronecker"] = nkp_out

    # block-diagonal energy (256×256 blocks on a 68×20 tiling)
    T4 = W.reshape(68, 256, 20, 256)
    diag = 0.0
    for i in range(20):
        diag += float(np.square(T4[i, :, i, :]).sum())
    tot = float(np.square(W).sum())
    constructions["block_diag_256"] = {
        "diag_energy_frac": diag / tot,
        "params": 20 * 256 * 256,
        "param_frac": (20 * 256 * 256) / W.size,
        "rel_fro_if_offdiag_zero": math.sqrt(1.0 - diag / tot),
    }

    # monarch-scale param count already in ir_priced; residual of
    # two block-diagonal factors is not fitted (expensive). Energy
    # in 256-blocks is the cheap proxy.

    out["L0_gate"]["constructions"] = constructions
    del WH, Wvis, Wspan, Wperm, Wmse_even, Eq, ref_W
    print(f"  L0 done rss={rss_gb():.3f}G", flush=True)

    # ------------------------------------------------------------------
    # 2. Cross-layer: spectrum affinity vs subspace overlap
    # ------------------------------------------------------------------
    def pair_compare(cls, la, lb, k=256):
        print(f"  pair {cls} {la},{lb}", flush=True)
        A = dg.load_tensor(f"language_model.model.layers.{la}.{cls}.weight")
        sa, Va = rsvd(A, k=k)
        # keep A for canon/raw — canon on 17408×5120 is a couple argsorts
        B = dg.load_tensor(f"language_model.model.layers.{lb}.{cls}.weight")
        sb, Vb = rsvd(B, k=k)
        raw = flatcos(A, B)
        # spectrum metrics
        scos, srel = spectrum_agreement(sa / (sa[0] + 1e-30), sb / (sb[0] + 1e-30))
        rl2 = rel_l2_spectra(sa, sb)
        ov32 = subspace_overlap(Va[:32], Vb[:32])
        ov256 = subspace_overlap(Va[:256], Vb[:256])
        # energy in top-k
        # rsvd s^2 / ||W||^2 is approximate
        eA = float(np.square(sa[:32]).sum() / (np.square(A).sum() + 1e-30))
        eB = float(np.square(sb[:32]).sum() / (np.square(B).sum() + 1e-30))
        eA256 = float(np.square(sa[:256]).sum() / (np.square(A).sum() + 1e-30))
        eB256 = float(np.square(sb[:256]).sum() / (np.square(B).sum() + 1e-30))
        # shared-V residual: project B onto Va
        # B ≈ (B Va^T) Va
        coef = B @ Va[:32].T
        Bhat = coef @ Va[:32]
        shared32 = float(np.linalg.norm(B - Bhat) / (np.linalg.norm(B) + 1e-30))
        coef = B @ Va[:256].T
        Bhat = coef @ Va[:256]
        shared256 = float(np.linalg.norm(B - Bhat) / (np.linalg.norm(B) + 1e-30))
        # local rank-32 residual of B
        coef = B @ Vb[:32].T
        Bhat = coef @ Vb[:32]
        local32 = float(np.linalg.norm(B - Bhat) / (np.linalg.norm(B) + 1e-30))
        coef = B @ Vb[:256].T
        Bhat = coef @ Vb[:256]
        local256 = float(np.linalg.norm(B - Bhat) / (np.linalg.norm(B) + 1e-30))
        # null: random orthonormal k-frames
        rng = np.random.default_rng(0)
        R1 = rng.standard_normal(A.shape).astype(np.float32)
        R2 = rng.standard_normal(A.shape).astype(np.float32)
        sr1, Vr1 = rsvd(R1, k=k, seed=1)
        sr2, Vr2 = rsvd(R2, k=k, seed=2)
        rl2_null = rel_l2_spectra(sr1, sr2)
        # also real vs random
        rl2_vs_rand = rel_l2_spectra(sa, sr1)
        ov32_null = subspace_overlap(Vr1[:32], Vr2[:32])
        scos_null, _ = spectrum_agreement(sr1 / (sr1[0] + 1e-30), sr2 / (sr2[0] + 1e-30))
        can = flatcos(canon(A), canon(B))
        del A, B, R1, R2, Bhat, coef
        return {
            "cls": cls, "pair": [la, lb],
            "raw_cos": raw, "canon_cos": can,
            "spectrum_cos": scos, "spectrum_cos_null": scos_null,
            "spectrum_rel_l2": rl2, "spectrum_rel_l2_null": rl2_null,
            "spectrum_rel_l2_vs_random": rl2_vs_rand,
            "affinity_x_vs_null": (rl2_null / rl2) if rl2 > 0 else None,
            "affinity_x_vs_realrand": (rl2_vs_rand / rl2) if rl2 > 0 else None,
            "overlap_k32": ov32, "overlap_k256": ov256, "overlap_k32_null": ov32_null,
            "energy_k32": [eA, eB], "energy_k256": [eA256, eB256],
            "sharedV_rel_k32": shared32, "sharedV_rel_k256": shared256,
            "localV_rel_k32": local32, "localV_rel_k256": local256,
        }

    pairs = [
        pair_compare("mlp.gate_proj", 30, 31),
        pair_compare("mlp.gate_proj", 15, 47),
        pair_compare("mlp.down_proj", 30, 31),
        pair_compare("self_attn.q_proj", 31, 35),
        pair_compare("self_attn.v_proj", 31, 35),
    ]
    out["cross_layer"] = pairs

    # ------------------------------------------------------------------
    # 3. Tensor-operator sequential cost (arithmetic on measured bandwidth)
    # ------------------------------------------------------------------
    # decode batch=1 is bandwidth-bound at 639.25 GB/s. Extra sequential
    # stages add launch + intermediate traffic. No GPU timing here.
    q4_gate_bytes = W.size / 2 + 2 * (W.size // 128)
    q4_gate_ns = q4_gate_bytes / (BANDWIDTH_GB_S * 1e9) * 1e9
    # 64-layer MLP three GEMVs
    mlp_q4_bytes = 3 * 64 * q4_gate_bytes
    out["operator_cost_model"] = {
        "one_gate_q4_bytes": q4_gate_bytes,
        "one_gate_q4_ns_at_639": q4_gate_ns,
        "mlp_all_q4_ns_at_639": mlp_q4_bytes / (BANDWIDTH_GB_S * 1e9) * 1e9,
        "launch_ns_est_low": 2000,
        "launch_ns_est_high": 5000,
        "two_stage_lowrank_r256_factor_bytes": (17408 + 5120) * 256 * 2,
        "note": "A k-stage operator that does not reduce unique bytes cannot beat geo_tpr64; each extra launch is 2-5 us ESTIMATED plus intermediate traffic. FLOP reduction is free on a bandwidth-bound decode and does not move TOKEN_NS.",
        "Hx_runtime_adds": 20 * 256 * 8,
        "Hx_is_register_butterfly": True,
        "dense_rotate_5120_bytes_f16": 5120 * 5120 * 2,
        "dense_rotate_ns_at_639": (5120 * 5120 * 2) / (BANDWIDTH_GB_S * 1e9) * 1e9,
        "dense_rotate_64_layers_both_sides_ms": 64 * 2 * (5120 * 5120 * 2) / (BANDWIDTH_GB_S * 1e9) * 1e3,
    }

    # entropy RA vs SEQ accounting on L0 numbers
    H4 = out["L0_gate"]["index_entropy"]["q4_native"]
    out["entropy_accounting"] = {
        "H4_L0_gate": H4,
        "g64_ans_index_bpw": H4 + 32 / 64,
        "g128_ans_index_bpw": H4 + 32 / 128,
        "g256_ans_index_bpw": H4 + 32 / 256,
        "seq_global_index_bpw": H4,
        "complete_if_all_GEMV_at_Hg64_plus_scale_g64": None,
    }
    # production H is 3.4789 element-weighted (CITED). Use that for complete.
    H_prod = 3.478937682977414
    # complete = (H_index + scale_bpw) * E_gemv / N + small
    E_gemv = N - E_SMALL
    for grain, flush, scale_g in (
        ("seq", 0.0, 64),
        ("G64", 32 / 64, 64),
        ("G128", 32 / 128, 128),
        ("G256", 32 / 256, 256),
    ):
        index = H_prod + flush
        scale = 16.0 / scale_g
        complete_e = (index + scale) * E_gemv / N + small_contrib
        out["entropy_accounting"][f"complete_{grain}_citedH"] = complete_e

    out["rss_gb_peak"] = rss_gb()
    out["wall_s"] = now() - t0
    path = "/tmp/g1_adversarial_frontier.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"wrote {path} wall={out['wall_s']:.2f}s rss={out['rss_gb_peak']:.3f}G", flush=True)
    # compact stdout
    print("=== ARITH ===")
    print(json.dumps(out["arithmetic"]["complete_fn_examples"], indent=2))
    print("inversions", json.dumps(out["arithmetic"]["inversions_sub1"], indent=2))
    print("floor", json.dumps(out["arithmetic"]["allocator_floor"], indent=2))
    print("sidecar", json.dumps(out["arithmetic"]["sidecar_q3mlp_slack"], indent=2))
    print("=== L0 REF ===", out["L0_gate"]["ref_q4_g128"], "rank", out["L0_gate"]["x_rank"])
    for k, v in out["L0_gate"]["constructions"].items():
        if k == "planes":
            print("PLANES")
            for kind, rows in v.items():
                for r in rows:
                    print(f"  {kind} p={r.get('n_planes')} healthy={r.get('healthy')} obs={r.get('observed'):.6f} pr={r.get('probed'):.6f} wu={r.get('worst_unit'):.6f} fro={r.get('rel_fro'):.4f} bpw={r.get('bpw', r.get('bpw_dense2'))}")
        elif k == "kronecker":
            print("KRON")
            for r in v:
                print(f"  {r.get('factors')} r={r.get('rank')} fro={r.get('rel_fro')} obs={r.get('observed')} healthy={r.get('healthy')} params={r.get('params')}")
        elif k == "mse_q3_fit_even":
            print("MSEFIT even", v["on_even_same_split"]["observed"], "odd", v["on_odd_heldout"]["observed"],
                  "absmax_even", v["absmax_q3_on_even"]["observed"], "absmax_odd", v["absmax_q3_on_odd"]["observed"])
        else:
            print(f"{k}: h={v.get('healthy')} obs={v.get('observed')} pr={v.get('probed')} wu={v.get('worst_unit')} fro={v.get('rel_fro')} E={v.get('energy_ratio')} extra={ {kk:v[kk] for kk in v if kk.startswith('probed_') or kk.startswith('x_plus') or kk=='bpw_ir'} }")
    print("=== PAIRS ===")
    for p in pairs:
        print(json.dumps({k: p[k] for k in p if k not in ()}, default=float))
    print("=== ENTROPY ===", out["L0_gate"]["index_entropy"], out["entropy_accounting"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

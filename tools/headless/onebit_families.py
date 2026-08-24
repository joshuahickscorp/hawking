#!/usr/bin/env python3
"""ONEBIT_FAMILIES: four structurally distinct families at matched executable bytes.

The question is not "how few bits per weight". It is: what physical information
must exist at all? A conventional 1-bit quantizer is one baseline, not the goal.

Families (structurally distinct, compared at ~2.00 fused-active bpw):

    B2  binary + per-group f16 scale          (signs stored)
    B3  ternary 5-in-8 + per-group f16 scale
    B4  binary SHARED BASIS across layers     (W ~ sum_k alpha_{l,k} B_k)
    B6  routed codebook: an index selects a shared f16 fragment (PQ)

B4 is NOT a retry of G035. G035 refuted SVD *column*-basis sharing at matched
bits (`shared_beats_independent=false`). Q80 experts are mutually orthogonal
(cosine 0.004) — that prior is column/expert tying, not a closed door for a
binary basis fitted in function space. This lane tests the latter.

B6 is where BPW can stop being the right coordinate: one route may replace
many parent-equivalent numbers. The matched row uses fragment length 4 so the
byte budget matches the other families; a large-fragment coordinate check is
reported separately.

Never Gaussian X. Fit in function space (||X(W-What)||). Count the scales.
State the null. Signed-symmetric absmax at 1 bit is the ZERO TENSOR — that
row is an instrument, not a family, and a miss is not "1-bit is impossible".

    python3 tools/headless/onebit_families.py
    python3 -m pytest tools/headless/test_onebit_families.py -q
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fractional_bit_canon import (  # noqa: E402
    CANON_VS_Q3_MAX_RATIO,
    CHUNK,
    F16_BPW,
    GAIN_HEALTHY,
    HEADER_BYTES,
    HIDDEN,
    INTERMEDIATE,
    LAYERS,
    ORGAN_SEED,
    ORGANS,
    PRIOR,
    REL_FRO_LOCAL_MAX,
    ROOT,
    SCALE_AWARE_MARGIN,
    SCALE_BITS,
    SCALE_TRAP,
    SEED,
    TRIT_PACK_5IN8,
    VISION_PY,
    _ensure_torch,
    as_groups,
    bill,
    classify,
    codec_absmax,
    codec_binary,
    codec_degenerate_absmax_b1,
    codec_scale_trap,
    codec_ternary,
    codec_zero,
    find_capture,
    find_parent,
    git_head,
    group_energy,
    j,
    load_tensor,
    load_X,
    n_groups,
    score_pair,
    signs,
    snap_f16,
    split_from_manifest,
    swiglu_intermediate,
    tensor_name,
    x_wt,
)

OUT_PATH = ROOT / "receipts" / "headless" / "ONEBIT_FAMILIES.json"
SCHEMA = "hawking.headless.onebit_families.v1"

# Matched executable budget. 2.00 bpw is the nearest common operating point
# the four families can occupy given group sizes that divide both 5120 and
# 17408. Ternary 5-in-8 at g=64 lands at 1.85 (inside the window).
MATCH_TARGET_BPW = 2.00
MATCH_WINDOW = (1.70, 2.15)

B2_GROUP = 16          # 1 + 16/16 = 2.00
B3_GROUP = 64          # 1.6 + 16/64 = 1.85
B4_K = 2
B4_GROUP = 32          # K/n_layers + K*16/g = 2/2 + 32/32 = 2.00 over 2 layers
B6_D = 4               # 8-bit index / 4 weights = 2.00; M=256 f16 centroids
B6_M = 256
B6_KMEANS_ITERS = 8
B6_SAMPLE = 65536
B6_LARGE_D = 64        # coordinate check: one route replaces 64 parent weights
B8_GROUP = 8           # generated signs, only scales stored: 16/8 = 2.00

# G035 / Q80 priors — cited, not re-derived.
G035_SHARED_BEATS_INDEPENDENT = False
G035_AXIS = "column (contraction axis), SVD factors, not binary bases"
Q80_EXPERT_PAIRWISE_COSINE = 0.004142791032791138
G034_LOWRANK_VS_Q3 = 2.93
RANK512_DEAD_BPW = 2.07
SIGN_CODE_LIVED_BPW = 1.0156


def _ensure_numpy():
    try:
        import numpy  # noqa: F401
    except ImportError:
        if VISION_PY.is_file() and Path(sys.executable).resolve() != VISION_PY.resolve():
            os.execv(str(VISION_PY), [str(VISION_PY), *sys.argv])
        raise


# ---------------------------------------------------------------------------
# Instruments (Gaussian WEIGHTS only — never Gaussian activations)
# ---------------------------------------------------------------------------


def walsh_signs(rows: int, cols: int, row0: int = 0):
    """Generated ±1 from parity of popcount(i & j). Not stored."""
    import numpy as np

    i = np.arange(row0, row0 + rows, dtype=np.uint32)[:, None]
    j = np.arange(cols, dtype=np.uint32)[None, :]
    x = i & j
    x ^= x >> 16
    x ^= x >> 8
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return np.where((x & 1) == 0, np.float32(1.0), np.float32(-1.0))


def group_ls_against(W, B, g: int, d=None):
    import numpy as np

    G = as_groups(W, g)
    Bg = as_groups(B, g)
    if d is None:
        num = (G * Bg).sum(axis=-1, keepdims=True)
        den = (Bg * Bg).sum(axis=-1, keepdims=True)
    else:
        dd = group_energy(d, g, G.shape[0])
        num = (dd * G * Bg).sum(axis=-1, keepdims=True)
        den = (dd * Bg * Bg).sum(axis=-1, keepdims=True)
    scale = snap_f16(np.where(den > 0, num / np.maximum(den, 1e-30), 0.0))
    return (scale * Bg).reshape(W.shape), scale


# ---------------------------------------------------------------------------
# B4 — shared binary bases across layers, fitted with AA diag-H
# ---------------------------------------------------------------------------


def fit_shared_binary_bases(Ws, ds, K: int, g: int):
    """W_l ≈ sum_k alpha_{l,k} B_k, B_k in {-1,+1}, alpha per-group f16.

    Signs are SHARED. Scales are per-layer. Greedy matching-pursuit then a
    joint per-group LS refit of the coefficients. Function-space via diag-H
    column energy from X_fit (never X_hold).
    """
    import numpy as np

    residuals = [np.array(W, dtype=np.float32, copy=True) for W in Ws]
    bases = []
    for _ in range(K):
        vote = np.zeros_like(Ws[0], dtype=np.float32)
        for R, d in zip(residuals, ds):
            vote += signs(R) * np.abs(R) * d[None, :].astype(np.float32)
        B = signs(vote)
        bases.append(B)
        for i, R in enumerate(residuals):
            step, _ = group_ls_against(R, B, g, d=ds[i])
            residuals[i] = R - step
    Whats = []
    alphas = []
    for W, d in zip(Ws, ds):
        What, alpha = _joint_group_ls(W, bases, d, g)
        Whats.append(What)
        alphas.append(alpha)
    return Whats, bases, alphas


def _joint_group_ls(W, bases, d, g: int):
    import numpy as np

    G = as_groups(W, g)
    rows, ng, gg = G.shape
    K = len(bases)
    Phi = np.stack([as_groups(B, g) for B in bases], axis=0).astype(np.float32)
    dd = group_energy(d, g, rows)
    # A[k,l,r,ng] = sum_j d * Phi[k] * Phi[l]
    A = np.einsum("krng,lrng,rng->klrn", Phi, Phi, dd)
    b = np.einsum("krng,rng,rng->krn", Phi, G, dd)
    A2 = np.moveaxis(A, (2, 3), (0, 1)).reshape(rows * ng, K, K)
    b2 = np.moveaxis(b, (1, 2), (0, 1)).reshape(rows * ng, K, 1)
    ridge = np.eye(K, dtype=np.float32) * 1e-4
    A2 = A2 + ridge
    try:
        alpha = np.linalg.solve(A2, b2)[..., 0]
    except np.linalg.LinAlgError:
        alpha = np.empty((rows * ng, K), dtype=np.float32)
        for i in range(rows * ng):
            alpha[i] = np.linalg.lstsq(A2[i], b2[i, :, 0], rcond=None)[0]
    alpha = snap_f16(alpha.reshape(rows, ng, K))
    What = np.einsum("rnk,krng->rng", alpha, Phi)
    return What.reshape(W.shape).astype(np.float32), alpha


def bill_shared_basis(W, K: int, g: int, n_layers: int) -> dict:
    n_w = int(W.size)
    n_g = n_groups(W, g)
    basis_bits = float(K) * n_w
    coef_bits_layer = float(K) * n_g * SCALE_BITS
    total_bits = basis_bits + n_layers * coef_bits_layer
    storage_bpw = total_bits / (n_layers * n_w)
    cold_active = float(K) + (K * SCALE_BITS) / g
    cached_active = (K * SCALE_BITS) / float(g)
    cf64 = (basis_bits + 64.0 * coef_bits_layer) / (64.0 * n_w)
    acc = bill(
        n_w=n_w,
        code_bits=basis_bits / n_layers,  # amortized basis bits per layer
        n_scales=int(K * n_g),
        extra_bits=0.0,
        extra_note=(
            f"{K} binary bases shared across {n_layers} layers; "
            f"per-layer {K} f16 scales per group-{g}"
        ),
        kernel="fused_shared_binary_bases_plus_group_scales",
    )
    # Override storage/active with amortized accounting (bill() would count
    # the shared bases as if they were per-layer).
    acc["storage_bits"] = total_bits / n_layers
    acc["storage_bpw"] = storage_bpw
    acc["active_fused_bpw"] = storage_bpw
    acc["code_bits"] = basis_bits / n_layers
    acc["code_bpw"] = (basis_bits / n_layers) / n_w
    acc["basis_count"] = int(K)
    acc["n_layers_amortized"] = int(n_layers)
    acc["basis_bits_shared_once"] = basis_bits
    acc["coefficient_bits_per_layer"] = coef_bits_layer
    acc["coefficient_bytes_per_layer"] = coef_bits_layer / 8.0
    acc["active_basis_loads_bpw_cold"] = cold_active
    acc["active_fused_bpw_bases_resident"] = cached_active
    acc["counterfactual_64_layer_storage_bpw"] = cf64
    acc["group"] = int(g)
    acc["quantizer"] = "shared_binary_bases_aa_joint_ls_coefficients"
    acc["testing"] = (
        "binary bases fitted in function space (AA diag-H), shared across "
        "layers. NOT G035 SVD column-basis sharing."
    )
    acc["amortization_note"] = (
        "storage_bpw amortizes bases over the layers they were fitted on. "
        "counterfactual_64_layer_storage_bpw is IF the same bases transferred "
        "to all 64 layers — not a survival claim. Basis cost approaches 0 "
        "with more layers; coefficient cost K*16/g does not."
    )
    return acc


# ---------------------------------------------------------------------------
# B6 — codebook: a route index selects a shared f16 fragment
# ---------------------------------------------------------------------------


def _kmeans_aa(sample, sample_e, M: int, iters: int, seed: int):
    """Weighted k-means. sample [N,d], sample_e [N,d] per-dim energy."""
    import numpy as np

    rng = np.random.RandomState(seed)
    N, d = sample.shape
    # Distinct init rows.
    choice = rng.choice(N, size=min(M, N), replace=False)
    C = sample[choice].copy()
    if C.shape[0] < M:
        extra = rng.randn(M - C.shape[0], d).astype(np.float32) * 1e-3
        C = np.concatenate([C, extra], axis=0)
    C = C.astype(np.float32)
    for _ in range(iters):
        assign = _aa_assign(sample, sample_e, C)
        new = np.zeros_like(C)
        wsum = np.zeros((M, d), dtype=np.float64)
        for m in range(M):
            mask = assign == m
            if not np.any(mask):
                new[m] = sample[rng.randint(0, N)]
                continue
            ww = sample_e[mask].astype(np.float64)
            vv = sample[mask].astype(np.float64)
            num = (ww * vv).sum(axis=0)
            den = ww.sum(axis=0)
            new[m] = np.where(den > 0, num / np.maximum(den, 1e-30), C[m])
        C = snap_f16(new.astype(np.float32))
    return C


def _aa_assign(V, E, C):
    """dist[n,m] = sum_j E[n,j] (V[n,j] - C[m,j])^2. Batched."""
    import numpy as np

    try:
        import torch

        return _aa_assign_torch(V, E, C)
    except Exception:
        pass
    N, d = V.shape
    M = C.shape[0]
    out = np.empty(N, dtype=np.int32)
    bs = 4096
    Ct = C.T
    C2 = C * C
    for i in range(0, N, bs):
        v = V[i : i + bs]
        e = E[i : i + bs]
        vw = v * e
        v2w = (vw * v).sum(axis=1, keepdims=True)
        eC2 = e @ C2.T
        dist = v2w + eC2 - 2.0 * (vw @ Ct)
        out[i : i + bs] = dist.argmin(axis=1).astype(np.int32)
    return out


def _aa_assign_torch(V, E, C):
    import numpy as np
    import torch

    vt = torch.from_numpy(np.ascontiguousarray(V, dtype=np.float32))
    et = torch.from_numpy(np.ascontiguousarray(E, dtype=np.float32))
    ct = torch.from_numpy(np.ascontiguousarray(C, dtype=np.float32))
    N = vt.shape[0]
    out = torch.empty(N, dtype=torch.int32)
    bs = 8192
    c2 = ct * ct
    ctT = ct.T.contiguous()
    for i in range(0, N, bs):
        v = vt[i : i + bs]
        e = et[i : i + bs]
        vw = v * e
        v2w = (vw * v).sum(dim=1, keepdim=True)
        eC2 = e @ c2.T
        dist = v2w + eC2 - 2.0 * (vw @ ctT)
        out[i : i + bs] = dist.argmin(dim=1).to(torch.int32)
    return out.numpy()


def codec_pq_aa(W, d: int = B6_D, M: int = B6_M, energy=None, seed: int = SEED,
                iters: int = B6_KMEANS_ITERS, sample_n: int = B6_SAMPLE):
    """Product-quantizer: 8-bit route selects an f16 fragment of length d.

    K-means + assignment use column energy from X_fit (function space).
    Centroids are stored in original weight space — energy is a fit statistic,
    not a runtime parameter.
    """
    import numpy as np

    rows, cols = W.shape
    if cols % d != 0:
        raise ValueError(f"cols {cols} not divisible by fragment {d}")
    n_sub = cols // d
    if energy is None:
        energy = np.ones(cols, dtype=np.float32)
    energy = np.ascontiguousarray(energy, dtype=np.float32)
    G = np.ascontiguousarray(W, dtype=np.float32).reshape(rows, n_sub, d)
    E = energy.reshape(n_sub, d)

    rng = np.random.RandomState(seed)
    n_vecs = rows * n_sub
    n_s = int(min(sample_n, n_vecs))
    idx = rng.choice(n_vecs, size=n_s, replace=False)
    sample = G.reshape(n_vecs, d)[idx]
    # G is (rows, n_sub, d) so flat index = row*n_sub + sub; sub = idx % n_sub.
    sample_e = E[idx % n_sub]
    C = _kmeans_aa(sample, sample_e, M, iters, seed)

    codes = np.empty((rows, n_sub), dtype=np.int32)
    for s in range(n_sub):
        codes[:, s] = _aa_assign(G[:, s, :], np.broadcast_to(E[s], G[:, s, :].shape), C)

    What = C[codes].reshape(rows, cols).astype(np.float32)
    n_w = int(W.size)
    code_bits = float(rows * n_sub) * math.log2(M)
    codebook_bits = float(M * d) * F16_BPW
    acc = bill(
        n_w=n_w,
        code_bits=code_bits,
        n_scales=0,
        extra_bits=codebook_bits,
        extra_note=(
            f"PQ d={d} M={M}: {rows*n_sub} routes × log2({M}) bits + "
            f"{M}×{d} f16 centroids (shared across rows and subspaces)"
        ),
        kernel="fused_pq_lookup_accumulate",
    )
    counts = np.bincount(codes.ravel(), minlength=M).astype(np.float64)
    p = counts / max(counts.sum(), 1.0)
    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum()) if nz.size else 0.0
    acc["group"] = int(d)
    acc["fragment_len"] = int(d)
    acc["codebook_size"] = int(M)
    acc["n_routes"] = int(rows * n_sub)
    acc["route_bits_each"] = float(math.log2(M))
    acc["route_bytes"] = (rows * n_sub) * math.log2(M) / 8.0
    acc["codebook_bytes"] = codebook_bits / 8.0
    acc["parent_weight_equivalents_per_route"] = int(d)
    acc["route_entropy_bits"] = entropy
    acc["route_entropy_over_logM"] = entropy / max(math.log2(M), 1e-12)
    acc["codebook_used"] = int((counts > 0).sum())
    acc["reuse"] = float(rows * n_sub) / max(M, 1)
    acc["quantizer"] = "aa_pq_shared_codebook"
    acc["bpw_coordinate_note"] = (
        f"One route replaces {d} parent-equivalent numbers. At d={d} BPW is "
        f"still a reasonable coordinate ({math.log2(M)/d:.3f} code bpw). "
        "At large d a route is a discrete choice over a fragment, and BPW "
        "stops being the right axis — see the large-fragment coordinate check."
    )
    return What, acc, codes, C


# ---------------------------------------------------------------------------
# B8 — generated Walsh signs, only coefficients stored
# ---------------------------------------------------------------------------


def codec_generated_walsh(W, g: int = B8_GROUP, d=None):
    import numpy as np

    rows, cols = W.shape
    # Build signs in row chunks so we never hold a second dense W.
    What = np.empty_like(W, dtype=np.float32)
    n_g_row = cols // g
    scales = np.empty((rows, n_g_row, 1), dtype=np.float32)
    chunk = 512
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        B = walsh_signs(r1 - r0, cols, row0=r0)
        recon, s = group_ls_against(W[r0:r1], B, g, d=d)
        What[r0:r1] = recon
        scales[r0:r1] = s
        del B, recon
    n_w = int(W.size)
    n_sc = n_groups(W, g)
    generator_bytes = 128  # identity of the popcount(i&j) construction
    cache_bits = float(n_w) * 1.0  # if signs were materialised
    acc = bill(
        n_w=n_w,
        code_bits=0.0,  # signs are generated, not stored
        n_scales=n_sc,
        extra_bits=0.0,
        extra_note=(
            f"Walsh/popcount(i&j) signs generated; {n_sc} f16 group scales. "
            f"Generator identity {generator_bytes} bytes is a header, not bpw "
            f"({generator_bytes * 8 / max(n_w, 1):.6e} bpw on this tensor)."
        ),
        kernel="generate_signs_then_scale_gemm",
    )
    acc["group"] = int(g)
    acc["quantizer"] = "generated_walsh_signs_plus_aa_group_scale"
    acc["generator_bytes"] = generator_bytes
    acc["generator_bits"] = generator_bytes * 8
    acc["generator_bpw_on_this_tensor"] = (generator_bytes * 8) / float(n_w)
    acc["header_bytes_not_in_bpw"] = HEADER_BYTES + generator_bytes
    acc["generator_runtime"] = (
        "one popcount(i&j) parity per weight at dequant, or a cached ±1 table"
    )
    acc["cache_bytes_if_signs_materialised"] = n_w / 8.0
    acc["active_fused_on_the_fly_bpw"] = acc["storage_bpw"]
    acc["active_cached_signs_bpw"] = acc["storage_bpw"] + 1.0
    acc["active_fused_bpw"] = acc["storage_bpw"]  # on-the-fly is the matched form
    acc["no_information_hiding"] = True
    acc["note_generated"] = (
        "storage_bpw is scales + generator identity. Signs are not a stored "
        "tensor. If the runtime caches them, add 1 bpw (active_cached_signs_bpw) "
        "and the family is no longer at the matched budget."
    )
    return What, acc


# ---------------------------------------------------------------------------
# Survival aggregation
# ---------------------------------------------------------------------------


def family_specs():
    """Independent (per-tensor) families at the matched budget."""
    return [
        {
            "id": "B2",
            "name": "binary_plus_per_group_scale",
            "structure": (
                "W_hat = sign(W) * s_g. Signs stored 1 bit/weight. One f16 "
                f"scale per group of {B2_GROUP} (counted). Activation-aware "
                "diag-H scale fitted on X_fit."
            ),
        },
        {
            "id": "B3",
            "name": "ternary_5in8_plus_scale",
            "structure": (
                "W_hat in {-s,0,+s} per group, threshold s/2, 5 trits in 8 bits "
                f"+ f16 scale per group of {B3_GROUP}."
            ),
        },
        {
            "id": "B6",
            "name": "routed_codebook_pq",
            "structure": (
                f"Each fragment of {B6_D} weights is replaced by one of {B6_M} "
                "shared f16 centroids. An 8-bit route selects the fragment. "
                "Codebook is shared across rows and subspaces."
            ),
        },
        {
            "id": "B8",
            "name": "generated_walsh_coefficients",
            "structure": (
                "Signs are generated (popcount(i&j) parity), not stored. Only "
                f"f16 scales per group of {B8_GROUP} plus the generator identity "
                "are physical information. Cache bytes reported separately."
            ),
        },
    ]


def b4_spec():
    return {
        "id": "B4",
        "name": "binary_shared_basis_across_layers",
        "structure": (
            f"W_l ≈ sum_{{k=1..{B4_K}}} alpha_{{l,k}} B_k, B_k in {{-1,+1}} "
            f"shared across the fitted layers, alpha per group-{B4_GROUP} f16. "
            "Fitted by AA-weighted greedy then joint LS coefficients. "
            "G035 column-SVD sharing is a prior, not this test."
        ),
    }


def in_match_window(bpw: float) -> bool:
    lo, hi = MATCH_WINDOW
    return lo - 1e-12 <= float(bpw) <= hi + 1e-12


def health_of(score: dict, acc: dict, zero_score: dict, q3_rel) -> dict:
    return classify(score, acc, zero_score, q3_rel)


def pack_row(family_id, family_name, acc, score, cls, extra=None):
    row = {
        "family_id": family_id,
        "family_name": family_name,
        "storage_bpw": float(acc["storage_bpw"]),
        "active_fused_bpw": float(acc["active_fused_bpw"]),
        "active_cached_f16_bpw": float(acc["active_cached_f16_bpw"]),
        "storage_bytes": float(acc["storage_bpw"]) * acc["n_weights"] / 8.0,
        "active_fused_bytes": float(acc["active_fused_bpw"]) * acc["n_weights"] / 8.0,
        "scales_counted": bool(acc.get("scales_counted", True)),
        "n_weights": int(acc["n_weights"]),
        "matched_window": list(MATCH_WINDOW),
        "in_matched_window": in_match_window(acc["storage_bpw"]),
        "rel_fro": float(score["rel_fro"]),
        "cosine": float(score["cosine"]),
        "gain": float(score["gain"]),
        "scale_aware": float(score["scale_aware"]),
        "null": float(score["null"]),
        "null_name": "constant_mean_row_cosine",
        "beats_null": bool(score["beats_null"]),
        "surplus_over_null": float(score["surplus_over_null"]),
        "survival_verdict": cls["health"],
        "local_survives": bool(cls["local_survives"]),
        "matches_deletion": bool(cls["matches_deletion"]),
        "matches_scale_trap": bool(cls["matches_scale_trap"]),
        "rel_fro_vs_q3": cls.get("rel_fro_vs_q3"),
        "accounting": {k: acc[k] for k in acc},
    }
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Per-organ pair screen
# ---------------------------------------------------------------------------


def run_independent_on_tensor(W, X_fit, X_hold, Y_hold, d, *, seed: int, q3_rel, zero_sc):
    rows_out = []

    # B2
    What, acc = codec_binary(W, g=B2_GROUP, d=d)
    acc["family_id"] = "B2"
    Yh = x_wt(X_hold, What)
    sc = score_pair(Y_hold, Yh)
    cls = health_of(sc, acc, zero_sc, q3_rel)
    rows_out.append(pack_row("B2", "binary_plus_per_group_scale", acc, sc, cls))
    del What, Yh

    # B3
    What, acc = codec_ternary(W, g=B3_GROUP, d=d)
    acc["family_id"] = "B3"
    Yh = x_wt(X_hold, What)
    sc = score_pair(Y_hold, Yh)
    cls = health_of(sc, acc, zero_sc, q3_rel)
    rows_out.append(pack_row("B3", "ternary_5in8_plus_scale", acc, sc, cls))
    del What, Yh

    # B6
    What, acc, codes, C = codec_pq_aa(W, d=B6_D, M=B6_M, energy=d, seed=seed)
    acc["family_id"] = "B6"
    Yh = x_wt(X_hold, What)
    sc = score_pair(Y_hold, Yh)
    cls = health_of(sc, acc, zero_sc, q3_rel)
    rows_out.append(
        pack_row(
            "B6",
            "routed_codebook_pq",
            acc,
            sc,
            cls,
            extra={
                "parent_weight_equivalents_per_route": acc["parent_weight_equivalents_per_route"],
                "route_entropy_bits": acc["route_entropy_bits"],
                "reuse": acc["reuse"],
            },
        )
    )
    del What, Yh, codes, C

    # B8
    What, acc = codec_generated_walsh(W, g=B8_GROUP, d=d)
    acc["family_id"] = "B8"
    Yh = x_wt(X_hold, What)
    sc = score_pair(Y_hold, Yh)
    cls = health_of(sc, acc, zero_sc, q3_rel)
    rows_out.append(pack_row("B8", "generated_walsh_coefficients", acc, sc, cls))
    del What, Yh

    return rows_out


def run_controls(W, Y_hold, zero_sc, q3_rel):
    rows = []
    What, acc = codec_zero(W)
    # Yh is zeros
    import numpy as np

    Yh = np.zeros_like(Y_hold)
    sc = score_pair(Y_hold, Yh)
    cls = health_of(sc, acc, zero_sc, q3_rel)
    rows.append(pack_row("CTRL_ZERO", "deletion_zero", acc, sc, cls))
    del What

    What, acc = codec_scale_trap(W)
    Yh = SCALE_TRAP * Y_hold
    sc = score_pair(Y_hold, Yh)
    sc["artifact"] = f"{SCALE_TRAP}*Y = X @ ({SCALE_TRAP}*W).T"
    cls = health_of(sc, acc, zero_sc, q3_rel)
    rows.append(pack_row("CTRL_SCALE", "scale_001W", acc, sc, cls))
    del What

    What, acc = codec_degenerate_absmax_b1(W, g=64)
    Yh = np.zeros_like(Y_hold)  # reconstruct is zero
    sc = score_pair(Y_hold, Yh)
    cls = health_of(sc, acc, zero_sc, q3_rel)
    rows.append(pack_row("CTRL_ABSMAX1", "degenerate_absmax_b1", acc, sc, cls))
    del What
    return rows


def run_q3(W, X_hold, Y_hold, zero_sc):
    What, acc = codec_absmax(W, bits=3, g=64)
    Yh = x_wt(X_hold, What)
    sc = score_pair(Y_hold, Yh)
    cls = health_of(sc, acc, zero_sc, sc["rel_fro"])
    row = pack_row("REF_Q3", "q3_sym_absmax_g64", acc, sc, cls)
    del What, Yh
    return row, sc["rel_fro"]


def screen_organ(layer_a, layer_b, organ, Wa, Wb, Xa_fit, Xa_hold, Xb_fit, Xb_hold, *,
                 do_large_fragment: bool):
    import numpy as np

    t0 = time.time()
    da = (Xa_fit.astype(np.float64) ** 2).sum(axis=0).astype(np.float32)
    db = (Xb_fit.astype(np.float64) ** 2).sum(axis=0).astype(np.float32)

    out_layers = []
    Ys = []
    zeros = []
    q3s = []
    for layer, W, Xf, Xh, d in (
        (layer_a, Wa, Xa_fit, Xa_hold, da),
        (layer_b, Wb, Xb_fit, Xb_hold, db),
    ):
        print(f"    L{layer} {organ} independent families...", flush=True)
        seed = SEED ^ (layer * 1009) ^ ORGAN_SEED[organ]
        Y_hold = x_wt(Xh, W)
        zero_sc = score_pair(Y_hold, np.zeros_like(Y_hold))
        q3_row, q3_rel = run_q3(W, Xh, Y_hold, zero_sc)
        fam_rows = run_independent_on_tensor(
            W, Xf, Xh, Y_hold, d, seed=seed, q3_rel=q3_rel, zero_sc=zero_sc
        )
        ctrl = run_controls(W, Y_hold, zero_sc, q3_rel)
        rec = {
            "layer": int(layer),
            "organ": organ,
            "tensor": tensor_name(layer, organ),
            "W_shape": [int(W.shape[0]), int(W.shape[1])],
            "n_weights": int(W.size),
            "n_fit": int(Xf.shape[0]),
            "n_hold": int(Xh.shape[0]),
            "null_output_hold": zero_sc["null"],
            "q3": q3_row,
            "families": fam_rows,
            "controls": ctrl,
        }
        # Large-fragment B6 coordinate check on at most one tensor.
        if do_large_fragment and layer == layer_a:
            print(f"    L{layer} {organ} B6 large-fragment d={B6_LARGE_D}...", flush=True)
            What, acc, codes, C = codec_pq_aa(
                W, d=B6_LARGE_D, M=B6_M, energy=d, seed=seed
            )
            Yh = x_wt(Xh, What)
            sc = score_pair(Y_hold, Yh)
            cls = health_of(sc, acc, zero_sc, q3_rel)
            rec["b6_large_fragment"] = pack_row(
                "B6_LARGE",
                "routed_codebook_large_fragment",
                acc,
                sc,
                cls,
                extra={
                    "parent_weight_equivalents_per_route": acc["parent_weight_equivalents_per_route"],
                    "not_the_matched_row": True,
                    "why": (
                        "Coordinate check, not the matched-byte comparison. "
                        f"A {B6_LARGE_D}-weight fragment per 8-bit route. "
                        "If this looks cheap on BPW, BPW is the wrong axis."
                    ),
                },
            )
            del What, Yh, codes, C
        out_layers.append(rec)
        Ys.append(Y_hold)
        zeros.append(zero_sc)
        q3s.append(q3_rel)
        del Y_hold

    print(f"    L{layer_a}+L{layer_b} {organ} B4 shared bases K={B4_K}...", flush=True)
    Whats, bases, alphas = fit_shared_binary_bases([Wa, Wb], [da, db], B4_K, B4_GROUP)
    acc = bill_shared_basis(Wa, B4_K, B4_GROUP, n_layers=2)
    b4_rows = []
    for What, Y_hold, Xh, zero_sc, q3_rel, layer, W in (
        (Whats[0], Ys[0], Xa_hold, zeros[0], q3s[0], layer_a, Wa),
        (Whats[1], Ys[1], Xb_hold, zeros[1], q3s[1], layer_b, Wb),
    ):
        Yh = x_wt(Xh, What)
        sc = score_pair(Y_hold, Yh)
        cls = health_of(sc, acc, zero_sc, q3_rel)
        composition_rel = float(
            __import__("numpy").linalg.norm(What - W)
            / max(float(__import__("numpy").linalg.norm(W)), 1e-30)
        )
        row = pack_row(
            "B4",
            "binary_shared_basis_across_layers",
            acc,
            sc,
            cls,
            extra={
                "basis_count": B4_K,
                "coefficient_bytes_per_layer": acc["coefficient_bytes_per_layer"],
                "active_basis_loads_bpw_cold": acc["active_basis_loads_bpw_cold"],
                "composition_weight_rel_fro": composition_rel,
                "n_layers_amortized": 2,
                "pair": [int(layer_a), int(layer_b)],
            },
        )
        b4_rows.append(row)
        # attach onto the per-layer record
        for rec in out_layers:
            if rec["layer"] == layer:
                rec["families"].append(row)
        del Yh
    del Whats, bases, alphas

    return {
        "organ": organ,
        "layers": [int(layer_a), int(layer_b)],
        "per_layer": out_layers,
        "b4_pair": b4_rows,
        "wall_s": time.time() - t0,
    }


def aggregate_families(organ_blocks: list) -> list:
    # Collect per family_id across all tensors.
    buckets = {}
    for block in organ_blocks:
        for rec in block["per_layer"]:
            for row in rec["families"]:
                buckets.setdefault(row["family_id"], []).append({**row, "layer": rec["layer"], "organ": rec["organ"]})
    specs = {s["id"]: s for s in family_specs()}
    specs["B4"] = b4_spec()
    out = []
    for fid, rows in buckets.items():
        rels = [r["rel_fro"] for r in rows]
        coss = [r["cosine"] for r in rows]
        gains = [r["gain"] for r in rows]
        sas = [r["scale_aware"] for r in rows]
        surp = [r["surplus_over_null"] for r in rows]
        nulls = [r["null"] for r in rows]
        q3s = [r["rel_fro_vs_q3"] for r in rows if r["rel_fro_vs_q3"] is not None]
        stor = rows[0]["storage_bpw"]
        act = rows[0]["active_fused_bpw"]
        all_local = all(r["local_survives"] for r in rows)
        any_del = any(r["matches_deletion"] for r in rows)
        any_trap = any(r["matches_scale_trap"] for r in rows)
        verdicts = [r["survival_verdict"] for r in rows]
        if any_del:
            family_verdict = "DELETION"
        elif any_trap:
            family_verdict = "SCALE_TRAP"
        elif all_local and in_match_window(stor):
            family_verdict = "SURVIVES_AT_MATCHED_BYTES"
        elif all_local:
            family_verdict = "SURVIVES_OFF_BUDGET"
        elif all(r["beats_null"] for r in rows):
            family_verdict = "BEATS_NULL_BUT_UNHEALTHY"
        else:
            family_verdict = "FAILS"
        spec = specs.get(fid, {"id": fid, "name": rows[0]["family_name"], "structure": ""})
        out.append(
            {
                "family_id": fid,
                "name": spec.get("name", rows[0]["family_name"]),
                "structure": spec.get("structure", ""),
                "structurally_distinct_from": [x for x in specs if x != fid],
                "storage_bpw": stor,
                "active_fused_bpw": act,
                "active_cached_f16_bpw": rows[0]["active_cached_f16_bpw"],
                "in_matched_window": in_match_window(stor),
                "n_tensors": len(rows),
                "mean_rel_fro": sum(rels) / len(rels),
                "max_rel_fro": max(rels),
                "mean_cosine": sum(coss) / len(coss),
                "min_cosine": min(coss),
                "mean_gain": sum(gains) / len(gains),
                "min_gain": min(gains),
                "mean_scale_aware": sum(sas) / len(sas),
                "mean_null": sum(nulls) / len(nulls),
                "mean_surplus_over_null": sum(surp) / len(surp),
                "min_surplus_over_null": min(surp),
                "mean_rel_fro_vs_q3": (sum(q3s) / len(q3s)) if q3s else None,
                "all_local_survive": all_local,
                "all_beats_null": all(r["beats_null"] for r in rows),
                "any_deletion": any_del,
                "any_scale_trap": any_trap,
                "survival_verdict": family_verdict,
                "per_tensor_verdicts": verdicts,
                "function_space_error": {
                    "metric": "rel_fro of Y_hold vs X_hold @ What.T",
                    "mean": sum(rels) / len(rels),
                    "max": max(rels),
                    "null": "constant_mean_row_cosine",
                    "mean_null": sum(nulls) / len(nulls),
                    "mean_surplus_over_null": sum(surp) / len(surp),
                    "scale_aware_metric": "cosine * gain (rejects 0.01*W)",
                    "mean_scale_aware": sum(sas) / len(sas),
                },
                "per_tensor": [
                    {
                        "layer": r["layer"],
                        "organ": r["organ"],
                        "storage_bpw": r["storage_bpw"],
                        "active_fused_bpw": r["active_fused_bpw"],
                        "rel_fro": r["rel_fro"],
                        "cosine": r["cosine"],
                        "gain": r["gain"],
                        "scale_aware": r["scale_aware"],
                        "null": r["null"],
                        "surplus_over_null": r["surplus_over_null"],
                        "survival_verdict": r["survival_verdict"],
                    }
                    for r in rows
                ],
                "accounting_extras": {
                    k: rows[0]["accounting"].get(k)
                    for k in (
                        "basis_count",
                        "coefficient_bytes_per_layer",
                        "active_basis_loads_bpw_cold",
                        "counterfactual_64_layer_storage_bpw",
                        "parent_weight_equivalents_per_route",
                        "route_entropy_bits",
                        "reuse",
                        "generator_bytes",
                        "generator_runtime",
                        "cache_bytes_if_signs_materialised",
                        "active_cached_signs_bpw",
                        "storage_bpw_5in8",
                        "packing_note",
                    )
                    if k in rows[0]["accounting"]
                },
            }
        )
    # Stable order: B2 B3 B4 B6 B8 then anything else.
    order = {"B2": 0, "B3": 1, "B4": 2, "B6": 3, "B8": 4}
    out.sort(key=lambda f: order.get(f["family_id"], 50))
    return out


def decide(families: list, instrument: dict, organs_out: list) -> dict:
    matched = [f for f in families if f["in_matched_window"] and f["family_id"] not in ("REF_Q3",)]
    survivors = [f for f in matched if f["survival_verdict"] == "SURVIVES_AT_MATCHED_BYTES"]
    survivors.sort(key=lambda f: (f["mean_rel_fro"], f["storage_bpw"]))
    ranked = sorted(matched, key=lambda f: (f["mean_rel_fro"], f["storage_bpw"]))

    nogo = []
    go = []
    if not instrument.get("binary_hits_optimum_band"):
        nogo.append("instrument: fitted binary missed the Gaussian sign-code optimum")
    if not instrument.get("degenerate_absmax_b1_is_zero"):
        nogo.append("instrument: absmax 1-bit did not degenerate to zero")
    if not instrument.get("g64_binary_storage_bpw_must_be_1.25"):
        nogo.append("instrument: g64 binary did not bill 1.25 bpw (scales not counted)")
    if len(matched) < 4:
        nogo.append(f"only {len(matched)} families landed in the matched window {MATCH_WINDOW}")

    for f in ranked:
        line = (
            f"{f['family_id']} {f['name']} stor={f['storage_bpw']:.4f} "
            f"act={f['active_fused_bpw']:.4f} mean_rel={f['mean_rel_fro']:.4f} "
            f"mean_cos={f['mean_cosine']:.4f} vs null {f['mean_null']:.4f} "
            f"surp={f['mean_surplus_over_null']:+.4f} {f['survival_verdict']}"
        )
        (go if f["survival_verdict"] == "SURVIVES_AT_MATCHED_BYTES" else nogo).append(line)

    # The global claim.
    if survivors:
        decision = "FAMILIES_SURVIVE_AT_MATCHED_BYTES"
        best = survivors[0]
        deciding = best["mean_rel_fro"]
        meaning = (
            f"{len(survivors)}/{len(matched)} matched families survive. Best "
            f"function-space error is {best['family_id']} mean rel_fro="
            f"{best['mean_rel_fro']:.6f} at storage {best['storage_bpw']:.4f} / "
            f"active fused {best['active_fused_bpw']:.4f}."
        )
    else:
        decision = "NO_MATCHED_FAMILY_SURVIVES"
        deciding = ranked[0]["mean_rel_fro"] if ranked else None
        meaning = (
            "No screened family locally survived on every tensor at the matched "
            "byte budget. That is a result about THESE schemes at THIS budget, "
            "not a proof that 1-bit is impossible."
        )

    return {
        "decision": decision,
        "deciding_number": deciding,
        "deciding_number_meaning": meaning,
        "n_families_screened": len(matched),
        "n_survive_at_matched_bytes": len(survivors),
        "best_survivor": survivors[0]["family_id"] if survivors else None,
        "ranked_by_function_space_error": [f["family_id"] for f in ranked],
        "one_failed_scheme_is_not_1bit_impossible": True,
        "global_claim_refused": (
            "Recording a single family's miss as a closed 1-bit question is "
            "not a legal verdict of this receipt. A family that matches the "
            "deletion control is measuring deletion. A family that loses at "
            "matched bytes is that family losing."
        ),
        "go_reasons": go,
        "nogo_reasons": nogo,
        "survival_rule": (
            "local_survives := not deletion AND not 0.01*W AND cosine>null AND "
            f"gain>={GAIN_HEALTHY} AND rel_fro<={REL_FRO_LOCAL_MAX} AND "
            f"scale_aware>={SCALE_AWARE_MARGIN}. Family SURVIVES_AT_MATCHED_BYTES "
            f":= local_survives on EVERY tensor AND storage_bpw in {MATCH_WINDOW}."
        ),
    }


def run_unit_instruments() -> dict:
    # Reuse the canon instrument: Gaussian WEIGHTS, never Gaussian X.
    from fractional_bit_canon import run_unit_instruments as _ri

    inst = _ri()
    import numpy as np

    rng = np.random.RandomState(0)
    W = rng.randn(64, 64).astype(np.float32)
    What, acc = codec_binary(W, g=16)
    inst["b2_g16_storage_bpw"] = acc["storage_bpw"]
    inst["b2_g16_storage_bpw_must_be_2"] = bool(abs(acc["storage_bpw"] - 2.0) < 1e-12)
    What3, acc3 = codec_ternary(W, g=64)
    inst["b3_g64_storage_bpw"] = acc3["storage_bpw"]
    inst["b3_g64_storage_bpw_must_be_1.85"] = bool(abs(acc3["storage_bpw"] - 1.85) < 1e-12)
    # B4 amortization on two copies.
    acc4 = bill_shared_basis(W, K=2, g=32, n_layers=2)
    inst["b4_k2_g32_2layer_storage_bpw"] = acc4["storage_bpw"]
    inst["b4_k2_g32_2layer_must_be_2"] = bool(abs(acc4["storage_bpw"] - 2.0) < 1e-12)
    inst["b4_counterfactual_64_less_than_fitted"] = bool(
        acc4["counterfactual_64_layer_storage_bpw"] < acc4["storage_bpw"]
    )
    # B6 d=4 M=256 code bpw
    inst["b6_d4_m256_code_bpw"] = 8.0 / 4.0
    inst["b6_parent_weight_equivalents_d4"] = 4
    # B8 g=8
    What8, acc8 = codec_generated_walsh(W, g=8, d=None)
    inst["b8_g8_storage_bpw"] = acc8["storage_bpw"]
    inst["b8_signs_not_stored"] = bool(acc8["code_bits"] == 0.0)
    inst["b8_generator_bytes_counted"] = acc8["generator_bytes"] > 0
    inst["b8_cache_bytes_reported"] = acc8["cache_bytes_if_signs_materialised"] > 0
    # Scale trap still on the instrument.
    inst["scale_trap_constant"] = SCALE_TRAP
    del What, What3, What8
    return inst


def print_report(doc: dict) -> None:
    print()
    print("ONE-BIT FAMILIES (matched executable bytes)")
    print("=" * 72)
    print(f"git_head: {doc['git_head']}")
    print(f"parent:   {doc['parent']}")
    print(f"capture:  {doc['capture']['path']}")
    print(f"match:    {doc['matched_executable_bpw_target']} bpw window {doc['matched_window']}")
    print()
    print("## PRIOR (cited, not re-run)")
    print(f"  G034 low-rank vs q3 = {G034_LOWRANK_VS_Q3}×  REFUTED")
    print(f"  G035 column-share shared_beats_independent={G035_SHARED_BEATS_INDEPENDENT} ({G035_AXIS})")
    print(f"  Q80 expert pairwise cosine = {Q80_EXPERT_PAIRWISE_COSINE} (column tying, not this B4)")
    print()
    print("## FAMILIES AT MATCHED BYTES")
    for f in doc["families"]:
        mark = " <--best" if f["family_id"] == doc["verdict"].get("best_survivor") else ""
        print(
            f"  {f['family_id']:<4} {f['name']:<36} stor={f['storage_bpw']:.4f} "
            f"act={f['active_fused_bpw']:.4f} rel={f['mean_rel_fro']:.4f} "
            f"cos={f['mean_cosine']:.4f} null={f['mean_null']:.4f} "
            f"surp={f['mean_surplus_over_null']:+.4f} {f['survival_verdict']}{mark}"
        )
    print()
    v = doc["verdict"]
    print("## VERDICT")
    print(f"  {v['decision']}")
    print(f"  {v['deciding_number_meaning']}")
    print(f"  one_failed_scheme_is_not_1bit_impossible={v['one_failed_scheme_is_not_1bit_impossible']}")
    print(f"  {v['global_claim_refused']}")
    print()
    print(f"wrote: {doc['written_to']}")
    print(f"wall_s: {doc['wall_s']:.1f}")


def main() -> int:
    _ensure_torch()
    import numpy as np
    import torch

    torch.set_num_threads(min(16, os.cpu_count() or 8))
    t_all = time.time()
    print("ONE-BIT FAMILIES")
    print("=" * 72)
    head = git_head()
    print(f"git_head: {head}")
    print(f"python:   {sys.executable}")
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    print(f"torch:    {torch.__version__} mps={mps} threads={torch.get_num_threads()}")

    instrument = run_unit_instruments()
    print(
        f"instrument: binary g64 bill {instrument['g64_binary_storage_bpw']:.4f} "
        f"(must 1.25) absmax1=zero:{instrument['degenerate_absmax_b1_is_zero']} "
        f"B4 2-layer {instrument['b4_k2_g32_2layer_storage_bpw']:.4f} "
        f"B2 g16 {instrument['b2_g16_storage_bpw']:.4f}"
    )
    if not instrument["degenerate_absmax_b1_is_zero"]:
        raise RuntimeError("expected absmax 1-bit to be deletion")
    if not instrument["g64_binary_storage_bpw_must_be_1.25"]:
        raise RuntimeError("scales not counted")
    if not instrument["b2_g16_storage_bpw_must_be_2"]:
        raise RuntimeError("B2 g16 must bill 2.00 bpw")
    if not instrument["b4_k2_g32_2layer_must_be_2"]:
        raise RuntimeError("B4 K=2 g=32 over 2 layers must bill 2.00 amortized")

    parent = find_parent()
    cap = find_capture()
    print(f"parent:   {parent}")
    print(f"capture:  {cap}")
    print("teacher:  qualified parent BF16; no llama-server; no second 27B; stream per tensor")
    print()

    X0 = load_X(cap, LAYERS[0])
    n_tokens = int(X0.shape[0])
    fit_idx, hold_idx, man, split_rule = split_from_manifest(cap, n_tokens)
    print(
        f"CAPTURE  tokens={n_tokens} fit={len(fit_idx)} hold={len(hold_idx)} "
        f"split={split_rule}  (REAL, not Gaussian)"
    )
    del X0
    gc.collect()

    layer_a, layer_b = int(LAYERS[0]), int(LAYERS[1])
    # Keep fit/hold X for both layers (post_attn_norm). ~230MB each full, we
    # keep only the slices.
    X_fit = {}
    X_hold = {}
    for layer in (layer_a, layer_b):
        X = load_X(cap, layer)
        if X.shape[0] != n_tokens:
            raise ValueError(f"L{layer} rows {X.shape[0]} != {n_tokens}")
        X_fit[layer] = np.ascontiguousarray(X[fit_idx], dtype=np.float32)
        X_hold[layer] = np.ascontiguousarray(X[hold_idx], dtype=np.float32)
        del X
        gc.collect()

    organ_blocks = []
    large_done = False
    scale_trap_global = None

    for organ in ORGANS:
        print(f"-- {organ} layers {layer_a},{layer_b} --", flush=True)
        if organ == "down_proj":
            # Rebuild post-SwiGLU from parent BF16 (real, not captured Gaussian).
            S_fit, S_hold = {}, {}
            for layer in (layer_a, layer_b):
                print(f"  L{layer} computing real post-SwiGLU S...", flush=True)
                Wg = load_tensor(parent, tensor_name(layer, "gate_proj"))
                Wu = load_tensor(parent, tensor_name(layer, "up_proj"))
                S_fit[layer] = swiglu_intermediate(X_fit[layer], Wg, Wu)
                S_hold[layer] = swiglu_intermediate(X_hold[layer], Wg, Wu)
                del Wg, Wu
                gc.collect()
            Wa = load_tensor(parent, tensor_name(layer_a, organ))
            Wb = load_tensor(parent, tensor_name(layer_b, organ))
            block = screen_organ(
                layer_a, layer_b, organ, Wa, Wb,
                S_fit[layer_a], S_hold[layer_a], S_fit[layer_b], S_hold[layer_b],
                do_large_fragment=not large_done,
            )
            del Wa, Wb, S_fit, S_hold
        else:
            Wa = load_tensor(parent, tensor_name(layer_a, organ))
            Wb = load_tensor(parent, tensor_name(layer_b, organ))
            block = screen_organ(
                layer_a, layer_b, organ, Wa, Wb,
                X_fit[layer_a], X_hold[layer_a], X_fit[layer_b], X_hold[layer_b],
                do_large_fragment=not large_done,
            )
            del Wa, Wb
        if any("b6_large_fragment" in rec for rec in block["per_layer"]):
            large_done = True
        if scale_trap_global is None:
            # pull from first control
            for rec in block["per_layer"]:
                for c in rec["controls"]:
                    if c["family_id"] == "CTRL_SCALE":
                        scale_trap_global = {
                            "rel_fro": c["rel_fro"],
                            "cosine": c["cosine"],
                            "gain": c["gain"],
                            "scale_aware": c["scale_aware"],
                            "null": c["null"],
                            "beats_null": c["beats_null"],
                            "cosine_must_be_one": abs(c["cosine"] - 1.0) < 1e-5,
                            "gain_rejects": c["gain"] < 0.05,
                            "instrument_ok": abs(c["cosine"] - 1.0) < 1e-5 and c["gain"] < 0.05,
                        }
                        break
        organ_blocks.append(block)
        gc.collect()
        print(f"  {organ} done in {block['wall_s']:.1f}s", flush=True)

    families = aggregate_families(organ_blocks)
    verdict = decide(families, instrument, organ_blocks)

    # Collect large-fragment check if present.
    large = None
    for block in organ_blocks:
        for rec in block["per_layer"]:
            if rec.get("b6_large_fragment"):
                large = rec["b6_large_fragment"]
                large["layer"] = rec["layer"]
                large["organ"] = rec["organ"]
                break

    watched = [
        {
            "what": "Gaussian / synthetic-X evaluation",
            "result": "REFUSED",
            "why": "Every prior sub-bit negative here was a Gaussian-proxy artifact.",
        },
        {
            "what": "cosine as a GO metric on 0.01*W",
            "result": (
                f"gain={scale_trap_global['gain'] if scale_trap_global else None} rejects; "
                "cosine is 1.000000 and is never used alone"
            ),
            "why": "Cosine is scale-invariant. Gain + rel_fro + null surplus are required.",
        },
        {
            "what": "signed-symmetric absmax at bits=1",
            "result": "DELETION: bound=0, reconstruct is the zero tensor. Not a family.",
            "why": "Any '1-bit fails' row whose scores match deletion is measuring deletion.",
        },
        {
            "what": "G035 shared_beats_independent=false",
            "result": "PRIOR, column-axis SVD sharing. Not this B4.",
            "why": (
                "This lane tests shared BINARY bases fitted in function space. "
                "Q80 expert cosine 0.004 is expert orthogonality on a different parent."
            ),
        },
        {
            "what": "matched-bit low-rank (G034 / rank-512 at 2.07 bpw)",
            "result": f"REFUTED {G034_LOWRANK_VS_Q3}× q3; rank-512 died at {RANK512_DEAD_BPW} where a sign code lived at {SIGN_CODE_LIVED_BPW}",
            "why": "Cited, not re-run. Starting point, not this conclusion.",
        },
        {
            "what": "one failed scheme recorded as a closed 1-bit question",
            "result": "REFUSED",
            "why": verdict["global_claim_refused"],
        },
    ]

    results = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": head,
        "python": sys.executable,
        "torch": f"{torch.__version__} mps={mps}",
        "parent": str(parent),
        "did_not_load_second_27b": True,
        "streamed_per_tensor": True,
        "question": (
            "What physical information must exist at all, at matched executable "
            "bytes, in function space, on real held-out activations?"
        ),
        "not_how_few_bits": True,
        "matched_executable_bpw_target": MATCH_TARGET_BPW,
        "matched_window": list(MATCH_WINDOW),
        "matched_rule": (
            "Each family is configured to the nearest realizable codec near "
            f"{MATCH_TARGET_BPW} fused-active bpw given group sizes that divide "
            "both 5120 and 17408. Actual storage_bpw is reported. Comparison is "
            "at this operating point, not at each family's unconstrained optimum."
        ),
        "prior": {
            **PRIOR,
            "g035_shared_beats_independent": G035_SHARED_BEATS_INDEPENDENT,
            "g035_axis": G035_AXIS,
            "q80_expert_pairwise_cosine": Q80_EXPERT_PAIRWISE_COSINE,
            "g034_lowrank_vs_q3": G034_LOWRANK_VS_Q3,
            "rank512_dead_at_bpw": RANK512_DEAD_BPW,
            "sign_code_lived_at_bpw": SIGN_CODE_LIVED_BPW,
            "b4_what_is_being_tested": (
                "Shared BINARY bases fitted in function space (AA diag-H) across "
                f"layers {layer_a} and {layer_b}. Not G035 SVD column-basis sharing, "
                "not Q80 expert tying."
            ),
        },
        "unit_instruments": instrument,
        "capture": {
            "path": str(cap),
            "named_by": "receipts/headless/FRACTIONAL_BIT_CANON.json",
            "site_gate_up": "post_attn_norm",
            "site_down": "real silu(X@Wg.T)*(X@Wu.T) from qualified-parent BF16",
            "n_tokens": n_tokens,
            "n_fit": int(len(fit_idx)),
            "n_hold": int(len(hold_idx)),
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "split_rule": split_rule,
            "manifest_families": (man or {}).get("families"),
            "not_gaussian": True,
            "not_llama_server": True,
            "source_note": (
                "Phase-B capture_diverse2: real BF16 parent MLX full-model forward. "
                "Not Gaussian. Not Q5_K llama-server. Fit/hold from the capture manifest."
            ),
        },
        "organs": list(ORGANS),
        "layers": [layer_a, layer_b],
        "null": {
            "name": "constant_mean_row_cosine",
            "definition": "row-cosine of Y_hold against broadcast mean(Y_hold, axis=0)",
            "scale_aware_metric": "cosine * gain; gain is min(mean-row-ratio, min-col-ratio) folded to [0,1]",
            "scale_trap": SCALE_TRAP,
            "scale_trap_must_score_cosine_one_and_fail_gain": True,
            "deletion_control": "Yh = 0",
            "why_cosine_alone_is_illegal": "0.01*W scores cosine 1.000000",
        },
        "accounting": {
            "scale_bits": SCALE_BITS,
            "b2_g16_storage_bpw": 1.0 + SCALE_BITS / B2_GROUP,
            "b3_g64_storage_bpw": TRIT_PACK_5IN8 + SCALE_BITS / B3_GROUP,
            "b4_k2_g32_2layer_storage_bpw": B4_K / 2.0 + (B4_K * SCALE_BITS) / B4_GROUP,
            "b6_d4_m256_code_bpw": math.log2(B6_M) / B6_D,
            "b8_g8_scale_bpw": SCALE_BITS / B8_GROUP,
            "rule": "A codec storing a 16-bit scale per group of 64 is 1.25 bpw, not 1 bpw.",
            "note": "Report storage and active, or neither. Scales always counted.",
        },
        "families": families,
        "n_structurally_distinct_families": len(families),
        "organ_blocks": organ_blocks,
        "b6_large_fragment_coordinate_check": large,
        "scale_trap_global": scale_trap_global,
        "verdict": verdict,
        "what_i_watched_fail": watched,
        "one_failed_scheme_is_not_1bit_impossible": True,
        "wall_s": None,
        "written_to": str(OUT_PATH),
    }
    results["wall_s"] = time.time() - t_all
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(j(results), indent=2, allow_nan=False) + "\n")
    tmp.replace(OUT_PATH)
    print_report(results)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--self-test", "--unit"):
        _ensure_numpy()
        inst = run_unit_instruments()
        assert inst["degenerate_absmax_b1_is_zero"]
        assert inst["g64_binary_storage_bpw_must_be_1.25"]
        assert inst["b2_g16_storage_bpw_must_be_2"]
        assert inst["b4_k2_g32_2layer_must_be_2"]
        print("unit instruments ok")
        sys.exit(0)
    try:
        sys.exit(main())
    except Exception as e:
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        raise

#!/usr/bin/env python3
"""G-SHARE: joint/shared basis vs independent per-layer SVD at matched bits.

CPU only. Real Qwen3.8-27B BF16. No Metal, no generate, no repo writes.
Peak RSS must stay under 15 GiB. Basis counted once; complete BPW / N.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import sys
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/Users/scammermike/.claude-grok/worktrees/204-share-basis-20260817-181022"
if os.path.isdir("tools"):
    REPO = os.getcwd()
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, "/Users/scammermike/Downloads/hawking/tools")

import gravity_alignment as ga  # noqa: E402
import gravity_doctor_gate as dg  # noqa: E402
import gravity_ir as ir  # noqa: E402

SRC = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
CAP = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
N = ir.SOURCE_PARAM_COUNT
PIN = "workspace/superwave/g1/GRAVITY1_SOURCE_PIN.json"
K_KEEP = 512
RANKS = (8, 16, 32, 64, 128, 256, 384, 512)
T0 = time.time()
LOG = os.environ.get("SHARE_LOG", "/tmp/g1_share_basis.log")
OUT = os.environ.get("SHARE_OUT", "/tmp/g1_share_basis.json")


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')} t={time.time()-T0:7.1f}s rss={rss_gb():.2f}GiB] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(type(x))


def dump(obj) -> None:
    with open(OUT, "w") as f:
        json.dump(obj, f, indent=2, default=jsonable)


def tname(layer: int, cls: str) -> str:
    return f"language_model.model.layers.{layer}.{cls}.weight"


def load_W(layer: int, cls: str) -> np.ndarray:
    return dg.load_tensor(tname(layer, cls), root=SRC)


def load_X(layer: int) -> np.ndarray:
    return dg.load_X(layer, capture=CAP)


def gram_right(W: np.ndarray) -> np.ndarray:
    """G = WᵀW, f32. Eigenpairs are right singular structure."""
    return W.T @ W


def gram_left(W: np.ndarray) -> np.ndarray:
    return W @ W.T


def top_eigh(G: np.ndarray, k: int):
    n = G.shape[0]
    k = min(k, n)
    # scipy subset is ~2× faster than full eigh for k=512, n=5120
    from scipy.linalg import eigh as seigh

    evals, evecs = seigh(G, subset_by_index=(n - k, n - 1))
    evals = np.maximum(evals[::-1], 0.0)
    evecs = evecs[:, ::-1]
    return evals.astype(np.float64), np.ascontiguousarray(evecs.astype(np.float32))


def all_eigvalsh(G: np.ndarray) -> np.ndarray:
    ev = np.linalg.eigvalsh(G)
    return np.maximum(ev[::-1], 0.0).astype(np.float64)


def s_from_evals(evals: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(evals, 0.0))


def spec_unit(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    nrm = np.linalg.norm(s)
    return s / (nrm + 1e-30)


def spec_rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """L2 between L2-normalised spectra (the metric cosine saturates against)."""
    n = min(len(a), len(b))
    return float(np.linalg.norm(spec_unit(a[:n]) - spec_unit(b[:n])))


def spec_cos(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return float(spec_unit(a[:n]) @ spec_unit(b[:n]))


def captured_local(evals: np.ndarray, r: int, fro2: float) -> float:
    return float(evals[:r].sum() / (fro2 + 1e-30))


def captured_on(G: np.ndarray, V: np.ndarray) -> float:
    """||WV||_F² / ||W||_F² = tr(Vᵀ G V) / tr(G)."""
    Vr = V.astype(np.float64, copy=False)
    G64 = G.astype(np.float64, copy=False)
    num = float(np.sum(Vr * (G64 @ Vr)))
    den = float(np.trace(G64))
    return num / (den + 1e-30)


def q_uniform(W: np.ndarray, bits: int = 4, group: int = 128) -> np.ndarray:
    """Vectorized match of gravity_doctor_gate.c_uniform (complete groups only)."""
    lim = (1 << (bits - 1)) - 1
    m, d = W.shape
    out = W.astype(np.float32, copy=True)
    d_use = d - (d % group)
    if d_use <= 0:
        return out
    blk = out[:, :d_use].reshape(m, d_use // group, group)
    amax = np.max(np.abs(blk), axis=2, keepdims=True) + 1e-30
    step = amax / lim
    blk[:] = np.clip(np.round(blk / step), -lim, lim) * step
    return out


def q_uniform_all(W: np.ndarray, bits: int = 4, group: int = 128) -> np.ndarray:
    """Quantize every column; last group may be short. Used for skinny C."""
    lim = (1 << (bits - 1)) - 1
    m, d = W.shape
    out = W.astype(np.float32, copy=True)
    g = min(group, d)
    for s in range(0, d, g):
        e = min(d, s + g)
        blk = out[:, s:e]
        amax = np.max(np.abs(blk), axis=1, keepdims=True) + 1e-30
        step = amax / lim
        out[:, s:e] = np.clip(np.round(blk / step), -lim, lim) * step
    return out


def rel_f(W: np.ndarray, Wh: np.ndarray) -> float:
    a = W.astype(np.float64).ravel()
    b = Wh.astype(np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))


# ---- cost model (f16 factors unless noted) ---------------------------------

HEADER = 40  # same as gravity_ir


def align_bytes(n_in: int) -> dict:
    c = ga.align_costs(n_in, n_in)
    c["perm_sign_scale"] = (
        c["permutation_bytes"] + c["sign_bytes"] + c["channel_scale_bytes"]
    )
    c["sign_scale"] = c["sign_bytes"] + c["channel_scale_bytes"]
    return c


def bytes_indep(K: int, r: int, m: int, n: int, b: int = 2) -> int:
    return K * (r * (m + n) * b + HEADER)


def bytes_shared_right(K: int, r: int, m: int, n: int, b: int = 2, align: float = 0.0) -> int:
    return int(r * n * b + K * (r * m * b + HEADER + align))


def bytes_shared_left(K: int, r: int, m: int, n: int, b: int = 2, align: float = 0.0) -> int:
    return int(r * m * b + K * (r * n * b + HEADER + align))


def r_share_right(K: int, r_i: int, m: int, n: int, align: float = 0.0) -> int:
    # r (n + K m) * 2 + K*(header+align) = K (r_i (m+n)*2 + header)
    target = bytes_indep(K, r_i, m, n)
    den = 2 * (n + K * m)
    r = int(round((target - K * (HEADER + align)) / den))
    return max(1, min(r, min(m, n)))


def r_share_left(K: int, r_i: int, m: int, n: int, align: float = 0.0) -> int:
    target = bytes_indep(K, r_i, m, n)
    den = 2 * (m + K * n)
    r = int(round((target - K * (HEADER + align)) / den))
    return max(1, min(r, min(m, n)))


def bpw_local(nbytes: int, elems: int) -> float:
    return 8.0 * nbytes / elems


def bpw_complete(nbytes: int) -> float:
    return 8.0 * nbytes / N


# ---- alignment -------------------------------------------------------------

def col_rms(W: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(W.astype(np.float64) ** 2, axis=0) + 1e-30).astype(np.float32)


def align_columns(W: np.ndarray, mode: str):
    """Return (W', meta). Column-only; right-Gram invariant to row perm."""
    if mode == "none":
        return W, {"mode": "none"}
    rms = col_rms(W)
    Wn = (W / rms).astype(np.float32)
    sgn = np.sign(Wn.sum(axis=0) + 1e-30).astype(np.float32)
    Wn = Wn * sgn
    meta = {
        "mode": mode,
        "scale_bytes": 2 * W.shape[1],
        "sign_bytes": W.shape[1] / 8.0,
    }
    if mode == "scale_sign":
        return Wn, meta
    if mode == "perm_scale_sign":
        perm = np.argsort(-rms)
        meta["perm_bytes"] = math.lgamma(W.shape[1] + 1) / math.log(2) / 8.0
        return Wn[:, perm], meta
    raise ValueError(mode)


def invert_align_right(Wh_aligned: np.ndarray, W_orig: np.ndarray, mode: str) -> np.ndarray:
    """Map a reconstruction in aligned coordinates back to original columns."""
    if mode == "none":
        return Wh_aligned
    rms = col_rms(W_orig)
    Wn = W_orig / rms
    sgn = np.sign(Wn.sum(axis=0) + 1e-30).astype(np.float32)
    if mode == "scale_sign":
        return (Wh_aligned * sgn) * rms
    if mode == "perm_scale_sign":
        perm = np.argsort(-rms)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        return (Wh_aligned[:, inv] * sgn) * rms
    raise ValueError(mode)


# ---- rSVD shared right (for fat n, e.g. down_proj 17408) -------------------

def shared_right_rsvd(Ws, r: int, p: int = 16, q: int = 1, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = Ws[0].shape[1]
    k = min(r + p, n)
    omega = rng.standard_normal((n, k)).astype(np.float32)
    Y = np.vstack([W @ omega for W in Ws])
    Q, _ = np.linalg.qr(Y, mode="reduced")
    del Y
    for _ in range(q):
        Z = np.zeros((n, Q.shape[1]), np.float32)
        off = 0
        for W in Ws:
            m = W.shape[0]
            Z += W.T @ Q[off : off + m]
            off += m
        Y = np.vstack([W @ Z for W in Ws])
        Q, _ = np.linalg.qr(Y, mode="reduced")
        del Y, Z
    B = np.zeros((Q.shape[1], n), np.float32)
    off = 0
    for W in Ws:
        m = W.shape[0]
        B += Q[off : off + m].T @ W
        off += m
    _u, s, Vh = np.linalg.svd(B, full_matrices=False)
    V = np.ascontiguousarray(Vh[:r].T)
    return V, s.astype(np.float64)


# ---- layer geometry --------------------------------------------------------

GQA = [i for i in range(64) if (i + 1) % 4 == 0]
MLP = list(range(64))

CLASSES = {
    "mlp.gate_proj": {
        "legal": MLP,
        "share": "right",  # n=5120 hidden
        "mid": list(range(24, 40)),
        "affinity_pairs": [(30, 31), (15, 47), (0, 1), (62, 63)],
    },
    "mlp.down_proj": {
        "legal": MLP,
        "share": "left",  # m=5120 hidden; right n=17408 needs rSVD
        "mid": list(range(24, 40)),
        "affinity_pairs": [(30, 31), (0, 1), (62, 63)],
    },
    "mlp.up_proj": {
        "legal": MLP,
        "share": "right",
        "mid": list(range(24, 40)),
        "affinity_pairs": [(30, 31), (0, 1)],
    },
    "self_attn.q_proj": {
        "legal": GQA,
        "share": "right",
        "mid": [19, 23, 27, 31, 35, 39, 43, 47],
        "affinity_pairs": [(31, 35), (3, 7), (59, 63)],
    },
    "self_attn.v_proj": {
        "legal": GQA,
        "share": "right",
        "mid": [19, 23, 27, 31, 35, 39, 43, 47],
        "affinity_pairs": [(31, 35), (3, 7), (59, 63)],
    },
}


def contiguous_bands(legal, mid, widths):
    """Bands of given width, clipped to legal, preferring the mid window."""
    idx = {L: i for i, L in enumerate(legal)}
    mid_i = [idx[L] for L in mid if L in idx]
    out = []
    seen = set()
    for K in widths:
        if not mid_i:
            continue
        # center on the middle of the mid window
        c = (mid_i[0] + mid_i[-1]) // 2
        a = max(0, min(c - K // 2, len(legal) - K))
        b = a + K
        if b > len(legal):
            continue
        layers = tuple(legal[a:b])
        if layers not in seen:
            seen.add(layers)
            out.append((K, list(layers)))
    return out


# ---- core pass -------------------------------------------------------------

def ingest_layer(layer, cls, share, k_keep=K_KEEP):
    W = load_W(layer, cls)
    m, n = int(W.shape[0]), int(W.shape[1])
    if share == "right":
        G = gram_right(W)
        side_n = n
    else:
        G = gram_left(W)
        side_n = m
    fro2 = float(np.einsum("ij,ij->", W, W, dtype=np.float64))
    evals, evecs = top_eigh(G, k_keep)
    # identity stats
    stats = {
        "layer": layer,
        "shape": [m, n],
        "fro2": fro2,
        "fro": float(np.sqrt(fro2)),
        "std": float(W.std()),
        "mean_abs": float(np.mean(np.abs(W))),
        "side_n": side_n,
        "share": share,
        "evals_head": evals[:32].tolist(),
        "energy": {str(r): captured_local(evals, r, fro2) for r in RANKS if r <= len(evals)},
    }
    # keep G f32 + evals + evecs; drop W
    del W
    gc.collect()
    return G, evals, evecs, stats


def band_report(K, layers, Gs, evals, evecs, stats, share, m, n, align_b=0.0):
    G_j = np.zeros_like(Gs[layers[0]])
    for L in layers:
        G_j += Gs[L]
    ev_j, V_j = top_eigh(G_j, K_KEEP)
    fro2_sum = sum(stats[L]["fro2"] for L in layers)
    rows = []
    for r_i in RANKS:
        if r_i > K_KEEP:
            continue
        if share == "right":
            r_s = r_share_right(K, r_i, m, n, align_b)
        else:
            r_s = r_share_left(K, r_i, m, n, align_b)
        r_s = min(r_s, K_KEEP)
        # same-rank captured
        loc = []
        sh = []
        ov = []
        for L in layers:
            loc.append(captured_local(evals[L], r_i, stats[L]["fro2"]))
            sh.append(captured_on(Gs[L], V_j[:, :r_i]))
            # subspace overlap of top-r local vs shared
            A = evecs[L][:, :r_i].astype(np.float64)
            B = V_j[:, :r_i].astype(np.float64)
            ov.append(float(np.sum((A.T @ B) ** 2) / r_i))
        # matched-bits: shared at r_s vs local at r_i
        sh_m = [captured_on(Gs[L], V_j[:, :r_s]) for L in layers]
        loc_m = [captured_local(evals[L], r_i, stats[L]["fro2"]) for L in layers]
        if share == "right":
            bi = bytes_indep(K, r_i, m, n)
            bs = bytes_shared_right(K, r_s, m, n, align=align_b)
            bs_same = bytes_shared_right(K, r_i, m, n, align=align_b)
        else:
            bi = bytes_indep(K, r_i, m, n)
            bs = bytes_shared_left(K, r_s, m, n, align=align_b)
            bs_same = bytes_shared_left(K, r_i, m, n, align=align_b)
        elems = K * m * n
        loc_rel = 1.0 - float(np.mean(loc))
        sh_rel = 1.0 - float(np.mean(sh))
        shm_rel = 1.0 - float(np.mean(sh_m))
        rows.append(
            {
                "r_indep": r_i,
                "r_shared_matched": r_s,
                "local_energy_mean": float(np.mean(loc)),
                "shared_energy_same_r": float(np.mean(sh)),
                "shared_energy_matched": float(np.mean(sh_m)),
                "local_relF_mean": loc_rel,
                "shared_relF_same_r": sh_rel,
                "shared_relF_matched": shm_rel,
                "matched_minus_local_relF": shm_rel - loc_rel,
                "same_r_minus_local_relF": sh_rel - loc_rel,
                "overlap_mean": float(np.mean(ov)),
                "overlap_min": float(np.min(ov)),
                "null_overlap": r_i / stats[layers[0]]["side_n"],
                "bytes_indep": bi,
                "bytes_shared_matched": bs,
                "bytes_shared_same_r": bs_same,
                "local_bpw": bpw_local(bi, elems),
                "shared_matched_bpw": bpw_local(bs, elems),
                "shared_wins_matched": shm_rel < loc_rel - 1e-12,
                "per_layer_local_relF": [1.0 - x for x in loc],
                "per_layer_shared_matched_relF": [1.0 - x for x in sh_m],
            }
        )
    # also energy vs r for the joint (for plots / optimum)
    joint_energy = {str(r): float(ev_j[:r].sum() / (fro2_sum + 1e-30)) for r in RANKS}
    return {
        "K": K,
        "layers": layers,
        "share": share,
        "m": m,
        "n": n,
        "joint_energy_of_sumG": joint_energy,
        "rows": rows,
        "n_wins_matched": sum(1 for r in rows if r["shared_wins_matched"]),
        "best_matched_gap": min(r["matched_minus_local_relF"] for r in rows),
        "best_matched_at": min(rows, key=lambda r: r["matched_minus_local_relF"])["r_indep"],
    }, V_j, ev_j


def affinity_block(pairs, evals, k=256):
    out = []
    for a, b in pairs:
        if a not in evals or b not in evals:
            continue
        sa, sb = s_from_evals(evals[a])[:k], s_from_evals(evals[b])[:k]
        out.append(
            {
                "pair": [a, b],
                "k": k,
                "rel_l2": spec_rel_l2(sa, sb),
                "cos": spec_cos(sa, sb),
            }
        )
    return out


def random_spectrum(shape, k, n_draw=2, seed=0):
    rng = np.random.default_rng(seed)
    m, n = shape
    specs = []
    for i in range(n_draw):
        # structured random with matching scale; Gram via one skinny multiply
        # Use a Gaussian matrix. For large m, form the smaller Gram.
        if n <= m:
            # G = WᵀW, draw W in tiles to cap memory
            G = np.zeros((n, n), np.float32)
            tile = 1024
            for r0 in range(0, m, tile):
                r1 = min(m, r0 + tile)
                Wt = rng.standard_normal((r1 - r0, n)).astype(np.float32)
                G += Wt.T @ Wt
            ev, _ = top_eigh(G, k)
        else:
            G = np.zeros((m, m), np.float32)
            tile = 1024
            for c0 in range(0, n, tile):
                c1 = min(n, c0 + tile)
                Wt = rng.standard_normal((m, c1 - c0)).astype(np.float32)
                G += Wt @ Wt.T
            ev, _ = top_eigh(G, k)
        specs.append(s_from_evals(ev)[:k])
        del G
        gc.collect()
    return specs


def ratio_vs_null(real_rel, real_s, null_specs):
    d_null = [spec_rel_l2(real_s, ns) for ns in null_specs]
    return {
        "real_rel_l2": real_rel,
        "null_rel_l2_mean": float(np.mean(d_null)),
        "null_rel_l2_min": float(np.min(d_null)),
        "ratio_null_over_real": float(np.mean(d_null) / (real_rel + 1e-30)),
        "null_cos_mean": float(np.mean([spec_cos(real_s, ns) for ns in null_specs])),
        "null_vs_null_rel_l2": spec_rel_l2(null_specs[0], null_specs[1])
        if len(null_specs) > 1
        else None,
        "null_vs_null_cos": spec_cos(null_specs[0], null_specs[1])
        if len(null_specs) > 1
        else None,
    }


# ---- reconstruction / gate / quant ----------------------------------------

def reconstruct_right(W, V):
    C = W @ V
    return C @ V.T, C


def reconstruct_left(W, U):
    C = U.T @ W
    return U @ C, C


def score_axes(W, Wh, layer, probe_only):
    if probe_only:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((64, W.shape[1])).astype(np.float32)
    else:
        X = load_X(layer)
        if X.shape[1] != W.shape[1]:
            rng = np.random.default_rng(0)
            X = rng.standard_normal((64, W.shape[1])).astype(np.float32)
            probe_only = True
    a = dg.axes(W, Wh, X, seed=0)
    a["probe_only"] = bool(probe_only)
    return a


def ir_program_for_class(cls, bands, r, m, n, share, coeff_bits=16, align=0.0):
    """One program: every site of this class, basis once per band."""
    p = ir.Program(f"gshare-{cls}-K{bands[0][0] if bands else '?'}-r{r}", source_pin=PIN)
    kernel = "gemv_Vt_then_C" if share == "right" else "gemv_U_then_C"
    elems = m * n
    for bi, (K, layers) in enumerate(bands):
        if share == "right":
            bbytes = r * n * 2
        else:
            bbytes = r * m * 2
        cid = p.pool.put(
            "SharedBasis",
            nbytes=bbytes,
            rank=r,
            share=share,
            band=bi,
            layers=list(layers),
            cls=cls,
        )
        for L in layers:
            # coefficients: m*r (right) or r*n (left), stored at coeff_bits
            n_coeff = (m * r) if share == "right" else (r * n)
            node = ir.shared_basis(
                n_coeff, coeff_bits, cid, kernel, header=HEADER
            )
            node.stored_bytes += int(math.ceil(align))
            node.elements = elems
            node.meta["layer"] = L
            node.meta["align_bytes"] = align
            p.add(tname(L, cls), elems, [node])
    return p


# ---- main ------------------------------------------------------------------

def run(args):
    if os.path.exists(LOG) and not args.keep_log:
        os.remove(LOG)
    widths = [int(x) for x in args.widths.split(",")]
    class_list = [c.strip() for c in args.classes.split(",") if c.strip()]
    result = {
        "schema": "hawking.g1.share_basis.v1",
        "N": N,
        "source": SRC,
        "capture": CAP,
        "pin": PIN,
        "ranks": list(RANKS),
        "k_keep": K_KEEP,
        "header_bytes": HEADER,
        "factor_dtype_bytes": 2,
        "classes": {},
        "nulls": {},
        "gate": [],
        "quant_C": [],
        "eigen_quant": [],
        "residual": [],
        "down_rsvd": [],
        "alignment": [],
        "ir": {},
        "projected": {},
    }

    # identity
    W0 = load_W(30, "mlp.gate_proj")
    result["identity"] = {
        "L30_gate_shape": list(map(int, W0.shape)),
        "L30_gate_std": float(W0.std()),
        "L30_gate_fro": float(np.linalg.norm(W0)),
        "L30_gate_finite": bool(np.isfinite(W0).all()),
        "align_costs_n5120": align_bytes(5120),
        "dense_orth_bpw_64sites": bpw_complete(64 * 2 * 5120 * 5120),
        "one_basis_fullV_bpw": bpw_complete(2 * 5120 * 5120),
    }
    log(
        f"identity L30 gate {tuple(W0.shape)} std={W0.std():.6f} "
        f"dense64={result['identity']['dense_orth_bpw_64sites']:.6f} "
        f"oneV={result['identity']['one_basis_fullV_bpw']:.6f}"
    )
    # verify q_uniform vs c_uniform
    q_ref = dg.c_uniform(W0[:128, :256], 4, 128)
    q_vec = q_uniform(W0[:128, :256], 4, 128)
    result["identity"]["q_uniform_match"] = bool(np.allclose(q_ref, q_vec))
    log(f"q_uniform matches c_uniform: {result['identity']['q_uniform_match']}")
    del W0, q_ref, q_vec
    gc.collect()

    # random nulls per unique shape, computed lazily
    null_cache = {}

    def get_null(shape, k=256):
        key = (tuple(shape), k)
        if key not in null_cache:
            log(f"null spectrum shape={shape} k={k}")
            null_cache[key] = random_spectrum(shape, k, n_draw=2, seed=1)
            s0, s1 = null_cache[key]
            result["nulls"][str(key)] = {
                "shape": list(shape),
                "k": k,
                "null_vs_null_rel_l2": spec_rel_l2(s0, s1),
                "null_vs_null_cos": spec_cos(s0, s1),
            }
            log(
                f"  null-vs-null relL2={spec_rel_l2(s0,s1):.6f} cos={spec_cos(s0,s1):.6f}"
            )
        return null_cache[key]

    for cls in class_list:
        meta = CLASSES[cls]
        share = meta["share"]
        log(f"=== class {cls} share={share} ===")
        # layers we need
        need = set(meta["mid"])
        for a, b in meta["affinity_pairs"]:
            need.add(a)
            need.add(b)
        # also edges for K=2 depth check
        legal = meta["legal"]
        need.add(legal[0])
        need.add(legal[1] if len(legal) > 1 else legal[0])
        need.add(legal[-2] if len(legal) > 1 else legal[-1])
        need.add(legal[-1])
        layers_needed = sorted(need)
        Gs, evals, evecs, stats = {}, {}, {}, {}
        shape = None
        for L in layers_needed:
            try:
                G, ev, Vc, st = ingest_layer(L, cls, share)
            except KeyError as e:
                log(f"  SKIP L{L}: {e}")
                continue
            Gs[L], evals[L], evecs[L], stats[L] = G, ev, Vc, st
            shape = tuple(st["shape"])
            log(
                f"  L{L} {st['shape']} fro={st['fro']:.4f} "
                f"e8={st['energy'].get('8',0):.4f} e64={st['energy'].get('64',0):.4f} "
                f"e256={st['energy'].get('256',0):.4f}"
            )
            if rss_gb() > 13.5:
                log("  RSS near cap, abort class")
                break
        if shape is None:
            log(f"  no tensors for {cls}")
            continue
        m, n = shape
        # affinity
        aff = affinity_block(meta["affinity_pairs"], evals, k=256)
        nulls = get_null(shape, 256)
        for row in aff:
            a, b = row["pair"]
            sa = s_from_evals(evals[a])[:256]
            row.update(ratio_vs_null(row["rel_l2"], sa, nulls))
            log(
                f"  AFF {cls}{tuple(row['pair'])} relL2={row['rel_l2']:.6f} "
                f"null={row['null_rel_l2_mean']:.6f} ratio={row['ratio_null_over_real']:.2f}x "
                f"cos={row['cos']:.6f} null_cos={row['null_cos_mean']:.6f}"
            )

        # bands
        bands = contiguous_bands(legal, [L for L in meta["mid"] if L in Gs], widths)
        # extra K=2 at edges if both present
        if legal[0] in Gs and legal[1] in Gs:
            extra = (2, [legal[0], legal[1]])
            if extra[1] not in [b[1] for b in bands]:
                bands.append(extra)
        if legal[-2] in Gs and legal[-1] in Gs:
            extra = (2, [legal[-2], legal[-1]])
            if extra[1] not in [b[1] for b in bands]:
                bands.append(extra)

        band_rows = []
        V_store = {}
        for K, blayers in bands:
            blayers = [L for L in blayers if L in Gs]
            if len(blayers) != K:
                # allow shorter if we lost a layer
                K = len(blayers)
            if K < 2:
                continue
            rep, Vj, evj = band_report(
                K, blayers, Gs, evals, evecs, stats, share, m, n, align_b=0.0
            )
            band_rows.append(rep)
            V_store[(K, tuple(blayers))] = Vj
            best = min(rep["rows"], key=lambda r: r["matched_minus_local_relF"])
            log(
                f"  BAND K={K} {blayers[0]}-{blayers[-1]} wins={rep['n_wins_matched']}/{len(rep['rows'])} "
                f"best_gap={rep['best_matched_gap']:+.5f} @r_i={rep['best_matched_at']} "
                f"ov32={next(r['overlap_mean'] for r in rep['rows'] if r['r_indep']==32):.4f} "
                f"@r64 local={next(r['local_relF_mean'] for r in rep['rows'] if r['r_indep']==64):.4f} "
                f"sh_m={next(r['shared_relF_matched'] for r in rep['rows'] if r['r_indep']==64):.4f}"
            )

        result["classes"][cls] = {
            "share": share,
            "shape": [m, n],
            "stats": {str(L): {k: v for k, v in st.items() if k != "evals_head"} | {"evals_head": st["evals_head"]} for L, st in stats.items()},
            "affinity": aff,
            "bands": band_rows,
        }
        dump(result)

        # ---- alignment on the hottest K=2 and K=8 mid bands ---------------
        if args.align and cls in ("mlp.gate_proj", "mlp.down_proj", "self_attn.q_proj"):
            for K_want, mode in ((2, "perm_scale_sign"), (8, "perm_scale_sign"), (2, "scale_sign")):
                cand = [b for b in bands if b[0] == K_want and set(b[1]).issubset(Gs)]
                if not cand:
                    continue
                # pick the mid-most
                cand.sort(key=lambda b: abs((b[1][0] + b[1][-1]) / 2 - 31))
                K, blayers = cand[0]
                blayers = [L for L in blayers if L in Gs]
                K = len(blayers)
                log(f"  ALIGN {mode} K={K} {blayers}")
                # reload, align, new Grams
                Ga, sta = {}, {}
                ac = align_bytes(n if share == "right" else m)
                ab = ac["perm_sign_scale"] if mode == "perm_scale_sign" else ac["sign_scale"]
                for L in blayers:
                    W = load_W(L, cls)
                    Wa, _meta = align_columns(W, mode)
                    Ga[L] = gram_right(Wa) if share == "right" else gram_left(Wa)
                    fro2 = float(np.einsum("ij,ij->", Wa, Wa, dtype=np.float64))
                    ev, _vc = top_eigh(Ga[L], K_KEEP)
                    sta[L] = {"fro2": fro2, "side_n": Ga[L].shape[0]}
                    evals_a = ev
                    # stash evals under a side dict
                    if L not in sta:
                        sta[L] = {}
                    sta[L]["evals"] = ev
                    del W, Wa
                    gc.collect()
                # fake evals/evecs dicts for band_report — we need evecs for overlap
                evals_a, evecs_a = {}, {}
                for L in blayers:
                    ev, vc = top_eigh(Ga[L], K_KEEP)
                    evals_a[L] = ev
                    evecs_a[L] = vc
                    sta[L]["fro2"] = float(np.trace(Ga[L].astype(np.float64)))
                    sta[L]["side_n"] = Ga[L].shape[0]
                rep_a, _Vj, _ = band_report(
                    K, blayers, Ga, evals_a, evecs_a, sta, share, m, n, align_b=ab
                )
                # compare to unaligned same band
                un = next(
                    (b for b in band_rows if b["layers"] == blayers),
                    None,
                )
                result["alignment"].append(
                    {
                        "cls": cls,
                        "mode": mode,
                        "K": K,
                        "layers": blayers,
                        "align_bytes_per_site": ab,
                        "aligned": rep_a,
                        "unaligned_best_gap": None if un is None else un["best_matched_gap"],
                        "aligned_best_gap": rep_a["best_matched_gap"],
                    }
                )
                ug = "n/a" if un is None else f"{un['best_matched_gap']:+.5f}"
                log(
                    f"    aligned best_gap={rep_a['best_matched_gap']:+.5f} "
                    f"unaligned={ug} align_B={ab:.1f}"
                )
                dump(result)

        # ---- down_proj rSVD share-right (large side) ----------------------
        if args.rsvd and cls == "mlp.down_proj":
            for K_want in (2, 8):
                cand = [b for b in bands if b[0] == K_want]
                if not cand:
                    continue
                cand.sort(key=lambda b: abs((b[1][0] + b[1][-1]) / 2 - 31))
                K, blayers = cand[0]
                blayers = [L for L in blayers if L in Gs]
                K = len(blayers)
                log(f"  RSVD-right down K={K} {blayers} loading Ws")
                Ws = [load_W(L, cls) for L in blayers]
                # r_i=64 and 256
                for r_i in (64, 256):
                    r_s = r_share_right(K, r_i, m, n)
                    r_s = min(r_s, 768)
                    V, s = shared_right_rsvd(Ws, r=min(r_s, 512), p=16, q=1, seed=0)
                    # local energy from stored evals (left-Gram energy == singular energy)
                    loc = [captured_local(evals[L], r_i, stats[L]["fro2"]) for L in blayers]
                    sh = []
                    for W, L in zip(Ws, blayers):
                        C = W @ V
                        sh.append(float(np.einsum("ij,ij->", C, C, dtype=np.float64) / stats[L]["fro2"]))
                    loc_rel = 1.0 - float(np.mean(loc))
                    sh_rel = 1.0 - float(np.mean(sh))
                    bi = bytes_indep(K, r_i, m, n)
                    bs = bytes_shared_right(K, V.shape[1], m, n)
                    rec = {
                        "K": K,
                        "layers": blayers,
                        "r_indep": r_i,
                        "r_shared": int(V.shape[1]),
                        "local_relF": loc_rel,
                        "shared_relF": sh_rel,
                        "gap": sh_rel - loc_rel,
                        "bytes_indep": bi,
                        "bytes_shared": bs,
                        "wins": sh_rel < loc_rel,
                    }
                    result["down_rsvd"].append(rec)
                    log(
                        f"    r_i={r_i} r_s={V.shape[1]} local={loc_rel:.5f} "
                        f"shared={sh_rel:.5f} gap={sh_rel-loc_rel:+.5f} win={sh_rel<loc_rel}"
                    )
                del Ws
                gc.collect()
                dump(result)

        # keep mid-band V for later gate/quant; drop Grams to free RSS
        # stash one mid K=2 and K=8 V
        result["classes"][cls]["_keep_layers"] = [L for L in meta["mid"] if L in Gs]
        # free Gs / evecs (large)
        del Gs, evecs
        gc.collect()
        log(f"  class {cls} grams freed rss={rss_gb():.2f}")

    # =====================================================================
    # reconstruction-level tests on a few decisive points
    # =====================================================================

    def pick_band(cls, K_want):
        bands = result["classes"][cls]["bands"]
        cand = [b for b in bands if b["K"] == K_want]
        if not cand:
            return None
        cand.sort(key=lambda b: abs((b["layers"][0] + b["layers"][-1]) / 2 - 31))
        return cand[0]

    # GATE + quant C + residual + eigen-quant
    targets = []
    if "mlp.gate_proj" in result["classes"]:
        for K in (2, 8):
            b = pick_band("mlp.gate_proj", K)
            if b:
                targets.append(("mlp.gate_proj", b, "right", False))
    if "mlp.down_proj" in result["classes"]:
        b = pick_band("mlp.down_proj", 2)
        if b:
            targets.append(("mlp.down_proj", b, "left", True))
    if "self_attn.q_proj" in result["classes"]:
        b = pick_band("self_attn.q_proj", 2)
        if b:
            targets.append(("self_attn.q_proj", b, "right", False))
    if "self_attn.v_proj" in result["classes"]:
        b = pick_band("self_attn.v_proj", 2)
        if b:
            targets.append(("self_attn.v_proj", b, "right", False))

    for cls, band, share, probe_only in targets:
        layers = band["layers"]
        m, n = band["m"], band["n"]
        K = band["K"]
        log(f"=== RECON {cls} K={K} {layers} share={share} ===")
        Ws = []
        for L in layers:
            Ws.append(load_W(L, cls))
        # rebuild joint V/U from these W
        if share == "right":
            Gj = sum((gram_right(W) for W in Ws), start=np.zeros((n, n), np.float32))
        else:
            Gj = sum((gram_left(W) for W in Ws), start=np.zeros((m, m), np.float32))
        _ev, Vj = top_eigh(Gj, K_KEEP)
        del Gj
        gc.collect()

        # local V/U per layer
        locals_V = []
        for W in Ws:
            G = gram_right(W) if share == "right" else gram_left(W)
            _e, V = top_eigh(G, K_KEEP)
            locals_V.append(V)
            del G
        gc.collect()

        # honest Q4 ref axes on first and last layer of the band
        for li, (L, W) in enumerate(zip(layers, Ws)):
            if li not in (0, len(layers) - 1) and cls != "self_attn.v_proj":
                continue
            Wq = q_uniform(W, 4, 128)
            ref = score_axes(W, Wq, L, probe_only)
            gref = dg.gate(W, Wq, load_X(L) if (not probe_only and W.shape[1] == 5120) else np.random.default_rng(0).standard_normal((64, W.shape[1])).astype(np.float32), ref=ref)
            # the above is circular — compute ref axes then gate of candidates against it
            result["gate"].append(
                {
                    "cls": cls,
                    "layer": L,
                    "kind": "q4_g128_REFERENCE",
                    "relF": rel_f(W, Wq),
                    "axes": {k: ref[k] for k in ("observed", "probed", "worst_unit")},
                    "probe_only": probe_only,
                }
            )
            log(
                f"  Q4 L{L} relF={rel_f(W,Wq):.5f} obs={ref['observed']:.5f} "
                f"prb={ref['probed']:.5f} wu={ref['worst_unit']:.5f}"
            )

            # compare shared vs indep at r_i in {32,64,256} (high compression → mid)
            for r_i in (32, 64, 256):
                if share == "right":
                    r_s = r_share_right(K, r_i, m, n)
                else:
                    r_s = r_share_left(K, r_i, m, n)
                r_s = min(r_s, K_KEEP)
                if share == "right":
                    Wh_s, C_s = reconstruct_right(W, Vj[:, :r_s])
                    Wh_i, C_i = reconstruct_right(W, locals_V[li][:, :r_i])
                else:
                    Wh_s, C_s = reconstruct_left(W, Vj[:, :r_s])
                    Wh_i, C_i = reconstruct_left(W, locals_V[li][:, :r_i])
                ax_s = score_axes(W, Wh_s, L, probe_only)
                ax_i = score_axes(W, Wh_i, L, probe_only)
                Xg = (
                    load_X(L)
                    if (not probe_only and W.shape[1] == 5120)
                    else np.random.default_rng(0).standard_normal((64, W.shape[1])).astype(np.float32)
                )
                gs = dg.gate(W, Wh_s, Xg, ref=ref)
                gi = dg.gate(W, Wh_i, Xg, ref=ref)
                rec = {
                    "cls": cls,
                    "layer": L,
                    "K": K,
                    "r_indep": r_i,
                    "r_shared": r_s,
                    "kind": "lowrank_f16",
                    "relF_shared": rel_f(W, Wh_s),
                    "relF_indep": rel_f(W, Wh_i),
                    "axes_shared": {k: ax_s[k] for k in ("observed", "probed", "worst_unit")},
                    "axes_indep": {k: ax_i[k] for k in ("observed", "probed", "worst_unit")},
                    "healthy_shared": bool(gs["healthy"]),
                    "healthy_indep": bool(gi["healthy"]),
                    "gate_shared": float(gs["gate"]),
                    "gate_indep": float(gi["gate"]),
                    "worst_axis_shared": gs.get("worst_axis"),
                    "worst_axis_indep": gi.get("worst_axis"),
                    "probe_only": probe_only,
                }
                result["gate"].append(rec)
                log(
                    f"  LR L{L} r_i={r_i} r_s={r_s} relF s/i={rec['relF_shared']:.5f}/{rec['relF_indep']:.5f} "
                    f"gate s/i={gs['gate']:+.4f}/{gi['gate']:+.4f} "
                    f"H s/i={gs['healthy']}/{gi['healthy']} "
                    f"obs s={ax_s['observed']:.4f} i={ax_i['observed']:.4f}"
                )

                # quantized coefficients at same r (high-compression codec)
                if args.quant and r_i in (64, 256):
                    for bits in (2, 4):
                        Cq = q_uniform_all(C_s, bits, group=min(128, C_s.shape[1]))
                        if share == "right":
                            Wh_q = Cq @ Vj[:, :r_s].T
                            n_coeff = m * r_s
                            b_basis = r_s * n * 2
                        else:
                            Wh_q = Vj[:, :r_s] @ Cq
                            n_coeff = r_s * n
                            b_basis = r_s * m * 2
                        # IR-style coeff bytes
                        groups = max(1, n_coeff // min(128, C_s.shape[1]))
                        b_coeff = (n_coeff * bits + 7) // 8 + groups * 2 + HEADER
                        axq = score_axes(W, Wh_q, L, probe_only)
                        gq = dg.gate(W, Wh_q, Xg, ref=ref)
                        qrec = {
                            "cls": cls,
                            "layer": L,
                            "K": K,
                            "r": r_s,
                            "coeff_bits": bits,
                            "relF": rel_f(W, Wh_q),
                            "relF_unquant_C": rel_f(W, Wh_s),
                            "axes": {k: axq[k] for k in ("observed", "probed", "worst_unit")},
                            "healthy": bool(gq["healthy"]),
                            "gate": float(gq["gate"]),
                            "bytes_basis_once": b_basis,
                            "bytes_coeff_this_site": b_coeff,
                            "bytes_band": b_basis + K * b_coeff,
                            "bpw_local_band": bpw_local(b_basis + K * b_coeff, K * m * n),
                            "bpw_complete_this_band": bpw_complete(b_basis + K * b_coeff),
                        }
                        result["quant_C"].append(qrec)
                        log(
                            f"    Q{bits}(C) r={r_s} relF={qrec['relF']:.5f} "
                            f"(unq {qrec['relF_unquant_C']:.5f}) "
                            f"localBPW={qrec['bpw_local_band']:.4f} H={gq['healthy']}"
                        )

                # residual after shared vs after local, then Q4 the residual
                if args.residual and r_i == 64 and li == 0:
                    Rs = W - Wh_s
                    Ri = W - Wh_i
                    Rsq = q_uniform(Rs, 4, 128)
                    Riq = q_uniform(Ri, 4, 128)
                    Wsq = Wh_s + Rsq
                    Wiq = Wh_i + Riq
                    Wq_only = q_uniform(W, 4, 128)
                    # cost: f16 lowrank + Q4 residual  (residual is full size — this should LOSE)
                    if share == "right":
                        b_lr_s = bytes_shared_right(K, r_s, m, n) / K  # per site + 1/K basis
                        b_lr_i = bytes_indep(1, r_i, m, n)
                    else:
                        b_lr_s = bytes_shared_left(K, r_s, m, n) / K
                        b_lr_i = bytes_indep(1, r_i, m, n)
                    b_q4 = ir.quant_tensor(m * n, 4, 128, "x").stored_bytes
                    rrec = {
                        "cls": cls,
                        "layer": L,
                        "K": K,
                        "r_indep": r_i,
                        "r_shared": r_s,
                        "relF_q4": rel_f(W, Wq_only),
                        "relF_shared_plus_q4R": rel_f(W, Wsq),
                        "relF_indep_plus_q4R": rel_f(W, Wiq),
                        "relF_shared_R_only": rel_f(Rs, Rsq),
                        "bytes_shared_plus_q4R_site": b_lr_s + b_q4,
                        "bytes_indep_plus_q4R_site": b_lr_i + b_q4,
                        "bytes_q4_site": b_q4,
                        "worse_than_plain_q4": rel_f(W, Wsq) > rel_f(W, Wq_only) - 1e-12,
                    }
                    result["residual"].append(rrec)
                    log(
                        f"    RES+Q4 relF q4={rrec['relF_q4']:.5f} "
                        f"s+R={rrec['relF_shared_plus_q4R']:.5f} "
                        f"i+R={rrec['relF_indep_plus_q4R']:.5f} "
                        f"bytes s/q4={b_lr_s+b_q4:.0f}/{b_q4}"
                    )

            # full shared eigenbasis as a quantizer coordinate system (once per band)
            if args.eigen_quant and li == 0:
                # Vj is top-512, not full. For a true full basis we eigh the 5120 Gram completely.
                # Affordable: use the 512-dim shared subspace as a PARTIAL rotation —
                # quantize in the split [shared coords | residual coords].
                # Cheaper decisive test: rotate input by V_full from a COMPLETE eigh of the 5120 Gram.
                side = n if share == "right" else m
                if side == 5120:
                    log(f"  EIGEN-QUANT full V {cls} K={K} (complete eigh of joint Gram)")
                    if share == "right":
                        Gfull = sum((gram_right(W) for W in Ws), start=np.zeros((n, n), np.float32))
                    else:
                        Gfull = sum((gram_left(W) for W in Ws), start=np.zeros((m, m), np.float32))
                    ev_all = all_eigvalsh(Gfull)
                    # full vectors: we need them. eigh ~11s
                    ev2, Vfull = np.linalg.eigh(Gfull)
                    Vfull = np.ascontiguousarray(Vfull[:, ::-1].astype(np.float32))
                    del Gfull, ev2
                    gc.collect()
                    # apply to every layer in the band
                    for bits in (2, 4):
                        rels_w, rels_c = [], []
                        for W2 in Ws:
                            Wq2 = q_uniform(W2, bits, 128)
                            if share == "right":
                                C = W2 @ Vfull  # same shape as W
                                Cq = q_uniform(C, bits, 128)
                                Whc = Cq @ Vfull.T
                            else:
                                C = Vfull.T @ W2
                                Cq = q_uniform(C, bits, 128)
                                Whc = Vfull @ Cq
                            rels_w.append(rel_f(W2, Wq2))
                            rels_c.append(rel_f(W2, Whc))
                        b_v = 2 * 5120 * 5120
                        b_q = K * ir.quant_tensor(m * n, bits, 128, "x").stored_bytes
                        erec = {
                            "cls": cls,
                            "K": K,
                            "layers": layers,
                            "bits": bits,
                            "relF_plain_mean": float(np.mean(rels_w)),
                            "relF_in_shared_basis_mean": float(np.mean(rels_c)),
                            "delta_relF": float(np.mean(rels_c) - np.mean(rels_w)),
                            "bytes_V_once": b_v,
                            "bytes_codes": b_q,
                            "bytes_total": b_v + b_q,
                            "plain_bytes": b_q,  # same codes without V
                            "overhead_complete_bpw": bpw_complete(b_v),
                            "wins": float(np.mean(rels_c)) < float(np.mean(rels_w)) - 1e-6,
                        }
                        result["eigen_quant"].append(erec)
                        log(
                            f"    Q{bits} in joint eigenbasis relF={erec['relF_in_shared_basis_mean']:.5f} "
                            f"plain={erec['relF_plain_mean']:.5f} d={erec['delta_relF']:+.5f} "
                            f"V_bpw={erec['overhead_complete_bpw']:.6f} win={erec['wins']}"
                        )
                    del Vfull
                    gc.collect()

        del Ws, locals_V, Vj
        gc.collect()
        dump(result)

    # =====================================================================
    # IR complete-BPW projections
    # =====================================================================
    log("=== IR / projections ===")
    # class element census (full model)
    census = {
        "mlp.gate_proj": (64, 17408, 5120, "right"),
        "mlp.up_proj": (64, 17408, 5120, "right"),
        "mlp.down_proj": (64, 5120, 17408, "left"),
        "self_attn.q_proj": (16, 12288, 5120, "right"),
        "self_attn.k_proj": (16, 1024, 5120, "right"),
        "self_attn.v_proj": (16, 1024, 5120, "right"),
        "self_attn.o_proj": (16, 5120, 6144, "left"),
        "linear_attn.in_proj_qkv": (48, 10240, 5120, "right"),
        "linear_attn.in_proj_z": (48, 6144, 5120, "right"),
        "linear_attn.out_proj": (48, 5120, 6144, "left"),
    }
    result["census"] = {}
    for c, (ns, mm, nn, sh) in census.items():
        result["census"][c] = {
            "n_sites": ns,
            "m": mm,
            "n": nn,
            "elements": ns * mm * nn,
            "share": sh,
            "frac_N": ns * mm * nn / N,
        }

    # For each (K, r) of interest, price replacing ALL listed GEMV classes
    # with shared f16 factors, rest of N held at Q4 g128.
    rest_elems = N - sum(v["elements"] for v in result["census"].values())
    rest_q4 = ir.quant_tensor(rest_elems, 4, 128, "x").stored_bytes if rest_elems > 0 else 0
    # actually rest includes norms/biases/embed/lm_head etc. Hold embed+head+other at Q4.
    q4_all = ir.quant_tensor(N, 4, 128, "x").stored_bytes

    projections = []
    for K in (2, 4, 8, 16):
        for r in (32, 64, 128, 256, 512):
            total = 0
            detail = {}
            for c, (ns, mm, nn, sh) in census.items():
                n_bands = math.ceil(ns / K)
                # last band may be short; price as n_bands * basis + ns * coeff
                if sh == "right":
                    b_basis = n_bands * r * nn * 2
                    b_coeff = ns * (r * mm * 2 + HEADER)
                else:
                    b_basis = n_bands * r * mm * 2
                    b_coeff = ns * (r * nn * 2 + HEADER)
                detail[c] = {
                    "basis": b_basis,
                    "coeff": b_coeff,
                    "bytes": b_basis + b_coeff,
                    "local_bpw": bpw_local(b_basis + b_coeff, ns * mm * nn),
                }
                total += b_basis + b_coeff
            # remaining language elems at Q4 g128
            covered = sum(v["elements"] for v in result["census"].values())
            leftover = N - covered
            leftover_bytes = ir.quant_tensor(leftover, 4, 128, "x").stored_bytes
            complete = bpw_complete(total + leftover_bytes)
            projections.append(
                {
                    "K": K,
                    "r": r,
                    "shared_bytes": total,
                    "leftover_q4_bytes": leftover_bytes,
                    "leftover_elems": leftover,
                    "complete_bpw_shared_plus_q4_rest": complete,
                    "local_bpw_on_covered": bpw_local(total, covered),
                    "covered_elems": covered,
                    "covered_frac": covered / N,
                    "detail": detail,
                }
            )
    result["projected"]["shared_f16_plus_q4_rest"] = projections
    # highlight sub-1 complete points
    sub1 = [p for p in projections if p["complete_bpw_shared_plus_q4_rest"] < 1.0]
    result["projected"]["n_sub1"] = len(sub1)
    log(
        f"projections: {len(projections)} (K,r) cells, {len(sub1)} have complete BPW < 1.0 "
        f"(this is COST only — gate is the function test)"
    )
    if projections:
        p64 = next(p for p in projections if p["K"] == 8 and p["r"] == 64)
        log(
            f"  example K=8 r=64 complete={p64['complete_bpw_shared_plus_q4_rest']:.6f} "
            f"local_on_covered={p64['local_bpw_on_covered']:.6f} "
            f"leftover_elems={p64['leftover_elems']}"
        )

    # IR object for the winning-looking cell on gate K=2 r matched to r_i=64
    if "mlp.gate_proj" in result["classes"]:
        b = pick_band("mlp.gate_proj", 2)
        if b:
            r_s = r_share_right(2, 64, 17408, 5120)
            # tile all 64 gate layers into K=2 bands
            gate_bands = [(2, [i, i + 1]) for i in range(0, 64, 2)]
            prog = ir_program_for_class(
                "mlp.gate_proj", gate_bands, r_s, 17408, 5120, "right"
            )
            result["ir"]["gate_K2_r64matched"] = prog.report()
            log(
                f"IR gate K=2 r={r_s}: sites={prog.report()['sites']} "
                f"total_bytes={prog.report()['total_bytes']} "
                f"complete_bpw={prog.report()['complete_bpw']:.8f} "
                f"shared_bytes={prog.report()['shared_bytes']}"
            )
            # sanity: basis counted once per band
            assert prog.report()["shared_bytes"] == 32 * r_s * 5120 * 2, prog.report()

    result["rss_max_gb"] = rss_gb()
    result["elapsed_s"] = time.time() - T0
    dump(result)
    log(f"DONE rss_max={rss_gb():.2f}GiB elapsed={result['elapsed_s']:.1f}s -> {OUT}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--classes",
        default="mlp.gate_proj,mlp.down_proj,mlp.up_proj,self_attn.q_proj,self_attn.v_proj",
    )
    ap.add_argument("--widths", default="2,4,8,16")
    ap.add_argument("--align", action="store_true", default=True)
    ap.add_argument("--no-align", action="store_false", dest="align")
    ap.add_argument("--rsvd", action="store_true", default=True)
    ap.add_argument("--no-rsvd", action="store_false", dest="rsvd")
    ap.add_argument("--quant", action="store_true", default=True)
    ap.add_argument("--no-quant", action="store_false", dest="quant")
    ap.add_argument("--residual", action="store_true", default=True)
    ap.add_argument("--no-residual", action="store_false", dest="residual")
    ap.add_argument("--eigen-quant", action="store_true", default=True)
    ap.add_argument("--no-eigen-quant", action="store_false", dest="eigen_quant")
    ap.add_argument("--keep-log", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reframe oracle — data-AWARE low-rank for §8.1 L1.4 (and the L1.3 cross-layer
reference as a free add-on). CPU-only NumPy. DO NOT run while a capture/training
job holds the GPU/CPU (this script only touches CPU + mmap, but the kill
discipline is: don't contend).

WHY THIS EXISTS (the reframe being tested)
------------------------------------------
The original L1.4 oracle (`tools/bench/oracle_lowrank_codebook.py`) and the L1.3
oracle (`tools/bench/oracle_interlayer_delta.py`) both took a **data-free** SVD
of the raw weight matrix and measured **Frobenius** energy@r. They concluded the
weights are "not low-rank" (top-64 captures 3-9% FFN / ~26% attn of Frobenius
energy) and killed L1.4/L1.3.

But Frobenius energy is the WRONG objective for an inference codec. What matters
is not ||W - W_r||_F, it is ||(W - W_r) x|| on the ACTIVATIONS the weight
actually sees. Activations live in a low-dimensional subspace, so a weight can be
near-full-rank in Frobenius yet effectively low-rank IN THE NORM INDUCED BY THE
DATA. This is exactly the gap that activation-aware SVD methods (ASVD, SVD-LLM,
FWSVD) exploit; the original oracles never tested it. So L1.4/L1.3 died in the
data-FREE form; this script tests the data-AWARE form, which is the standard
state-of-the-art improvement and the legitimate Type-2 reframe.

WHAT IT MEASURES (per sampled FFN layer; gate_proj + up_proj only)
------------------------------------------------------------------
Input activation x for gate_proj / up_proj is exactly the captured `norm_in`
(the ffn_norm output), so the empirical activation matrix X is available with no
reconstruction. (down_proj's input is the 11008-dim SwiGLU intermediate, which is
reconstructable — see oracle_coactivation_permute.py — but heavier; left as a
documented extension. gate+up are 2 of the 3 FFN weight tensors and carry the
reframe signal.)

  Treat the dequantized Q4_K weight W as the target (SAME choice as the original
  oracles — we only have the Q4_K_M GGUF, not the f16 master; a NO-GO here is
  decisive, a marginal GO must defer to the f16/AWQ lane per the W4A8 note).

  L1.4 reframe (data-aware low-rank of W itself):
    C   = X^T X / N                  (n x n activation second moment, n=hidden)
    M   = W @ C^{1/2}                 (data-whitened weight)
    SVD(M) -> data-weighted singular values; energy@r in the DATA norm.
    W_r = (U_r S_r) @ (V_r^T C^{-1/2})  (rank-r approx optimal for ||(W-W_r)X^T||)
    Codec: store U_r,(S_r V_r^T C^{-1/2}) at f16 + residual (W-W_r) at b bits.
    Headline comparison: data-aware energy@r  vs  plain (Frobenius) energy@r
      -> did the original oracle UNDERSELL low-rank by using Frobenius?
    Decisive comparison: FUNCTIONAL error ||(W - What) X^T||_F / ||W X^T||_F of
      the FULL codec at <= Q4_K bytes, vs plain-Q4_K's own functional error 0
      (W is already the Q4_K recon, so the bar is: can the codec re-encode W at
      FEWER bytes than its current Q4_K footprint while keeping functional error
      small?). GO only if some (r,b) with bytes < Q4_K has tiny functional error.

  L1.3 reframe (does the already-resident W[L] give W[L+1] a free basis?):
    For consecutive pairs, in the data norm of X=norm_in[L+1], measure
    energy@r of the DELTA D=W[L+1]-W[L]. If the rank needed for the delta is no
    smaller than for W[L+1] alone (the L1.4 number), W[L] is a useless basis ->
    cross-layer reference adds nothing even data-aware. Also reports the
    data-weighted cross-layer cosine cos((W[L]X^T),(W[L+1]... )) proxy.

THE DECISIVE GUARD AGAINST AN IN-SAMPLE ILLUSION (added 2026-05-30)
------------------------------------------------------------------
A first run produced a striking `data-norm energy@64 ~ 0.99-1.00` vs Frobenius
`~0.10-0.15`. That is exactly the kind of in-sample result this project has been
burned by twice (an EAGLE head 90% in-sample / 33% held-out; a prefix-cache proxy
86.8% circular / 45% real). The catch: if the captured activation matrix X spans
an effective subspace of <= r dims, then C = X^T X/N is rank-deficient, M = W C^{1/2}
has <= r nonzero singular values, and data-norm energy@r = 1.0 TRIVIALLY for ANY
weight W. So `data-E64 ~ 1.0` may be a low-ACTIVATION-rank fact (saves no weight
bytes) rather than a low-rank-WEIGHT property. Two checks make it decisive:
  (A) ACTIVATION EFFECTIVE RANK of X: participation ratio (Sum s^2)^2/Sum s^4 and
      the rank capturing 99% of X's energy. If <= ~64, data-E64 is trivial.
  (B) HELD-OUT-TOKEN data-norm error: fit the whitened SVD low-rank basis on
      TRAIN tokens (70%), measure ||(W-Wr)x||/||Wx|| on HELD-OUT tokens (30%) at
      ranks {16,32,64,128}. Held-out ~ in-sample ~ 0 => real; held-out >> in-sample
      => overfit illusion (Type-1).

DECISION
  GO    if (i) held-out data-norm error at r=64 stays low AND tracks in-sample
        (generalizes), AND (ii) X effective rank >> 64 (so r=64 capturing the
        energy is a non-trivial weight property), AND (iii) the codec beats Q4_K
        bytes for a MAJORITY of sampled FFN tensors. Then the kill was Type-2 and
        this codec earns the GPU/quality lane on f16 weights.
  NO-GO if held-out error blows up vs in-sample (Type-1 illusion), OR the high
        data-E64 is merely the low-activation-rank artifact (saves no weight
        bytes), OR it does not beat Q4_K bytes at the held-out error gate. The
        L1.3 delta is GO only if its data-aware rank is materially below W alone.

RAM discipline: one weight tensor + one capture-layer resident at a time;
del + gc.collect() between. SOFT RSS warning at 8 GB (the machine has 18 GB; the
original 3 GB HARD-FATAL was a busy-machine artifact and is removed). Pure numpy.

Run (machine should be otherwise idle for clean timing, but CPU-only + mmap):
    /tmp/ggufenv/bin/python tools/bench/oracle_dataaware_lowrank.py \
        --gguf models/qwen2.5-3b-instruct-q4_k_m.gguf \
        --bin  _capture/q3b_ffn.bin \
        --out  reports/oracle_dataaware_lowrank.md   # defaults to ALL 36 layers
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

try:
    import resource

    def rss_gb():
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / (1024 ** 3) if sys.platform == "darwin" else kb / (1024 ** 2)
except Exception:  # pragma: no cover
    def rss_gb():
        return float("nan")

from gguf import GGUFReader
from gguf.quants import dequantize

SENTINEL = 0xFFFFFFFF
QK_K = 256
RANKS = (16, 32, 64)
# Energy-vs-rank curve sweep (data-norm energy at each rank).
CURVE_RANKS = (8, 16, 32, 64, 128, 256)
# Ranks at which we measure the DECISIVE held-out data-norm reconstruction error.
HELDOUT_RANKS = (16, 32, 64, 128)
RES_BITS = (2, 3)
# Soft RSS warning ceiling. The machine has 18 GB; one dequantized Q4_K FFN
# tensor (2048x11008 f32 ~= 90 MB) plus an 800x2048 capture is tiny, so the
# original 3 GB HARD-FATAL guard was a busy-machine artifact. We keep a soft
# warning (no exit) and rely on explicit del + gc.collect() between tensors.
RSS_WARN_GB = 8.0
# Held-out split fraction (train fraction used to FIT the whitening + low-rank
# basis; the rest is held out to test generalization).
TRAIN_FRAC = 0.70
# Functional-error threshold below which a re-encoding is "quality-neutral
# enough to advance" (data-weighted relative L2 on real activations). 0.02 mirrors
# the spirit of the tight parity regime; the GPU lane re-checks with real KL.
FUNC_ERR_GATE = 0.02


def check_rss(where):
    """Soft RSS warning — never exits. Returns current RSS in GB."""
    g = rss_gb()
    if g > RSS_WARN_GB:
        sys.stderr.write(f"[WARN] RSS {g:.2f} GB > {RSS_WARN_GB} at {where} "
                         f"(soft ceiling; continuing)\n")
    return g


# --------------------------------------------------------------------------
# Capture reader (per-layer norm_in[N,hidden]); identical framing to
# oracle_coactivation_permute.load_capture, but we only need norm_in.
# --------------------------------------------------------------------------
def load_norm_in(path: Path):
    data = Path(path).read_bytes()
    n = len(data)
    off = 0
    hidden = n_blocks = None
    acc: dict[int, list] = {}
    while off + 8 <= n:
        a, b = struct.unpack_from("<II", data, off)
        if a == SENTINEL and b == SENTINEL:
            if off + 16 > n:
                break
            _, _, hidden, n_blocks = struct.unpack_from("<IIII", data, off)
            hidden, n_blocks = int(hidden), int(n_blocks)
            off += 16
            continue
        if hidden is None:
            raise ValueError("stream did not start with a sentinel")
        rec = 4 + hidden * 4 + 2 * n_blocks * 4
        if off + rec > n:
            break
        (layer,) = struct.unpack_from("<I", data, off)
        foff = off + 4
        norm = np.frombuffer(data, np.float32, hidden, foff)
        acc.setdefault(int(layer), []).append(norm.copy())
        off += rec
    return hidden, n_blocks, {k: np.stack(v) for k, v in acc.items()}


def w_of(reader_by_name, name):
    """Dequantize a GGUF tensor to f32 [rows, cols] = [out_features, in_features]."""
    t = reader_by_name[name]
    W = dequantize(np.array(t.data), t.tensor_type).astype(np.float32)
    dims = tuple(int(x) for x in t.shape)  # gguf fastest-first = (in, out)
    return W.reshape(dims[1], dims[0]), t.tensor_type.name, int(t.n_bytes)


# --------------------------------------------------------------------------
# Linear algebra helpers (pure numpy)
# --------------------------------------------------------------------------
def csqrt_and_inv(C, eps=1e-6):
    """Symmetric PSD square-root and its inverse from eigendecomposition."""
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 0.0, None)
    s = np.sqrt(w + eps)
    Csqrt = (V * s) @ V.T
    Cinv = (V * (1.0 / s)) @ V.T
    return Csqrt.astype(np.float32), Cinv.astype(np.float32)


def topr_energy(sv, ranks):
    """Fraction of energy (sum of squared singular values) in top-r."""
    e = sv.astype(np.float64) ** 2
    tot = float(e.sum()) + 1e-30
    cum = np.cumsum(e) / tot
    return {r: float(cum[min(r, len(e)) - 1]) for r in ranks}


def uniform_quant_resid(R, bits):
    """Per-row symmetric uniform quantize/dequantize of a residual matrix.
    Models the '2-3 bit residual + per-row f16 scale' budget the byte oracle used.
    Returns the dequantized residual (same shape)."""
    levels = (1 << (bits - 1)) - 1  # symmetric: e.g. b=3 -> +/-3
    amax = np.maximum(np.abs(R).max(axis=1, keepdims=True), 1e-12)
    scale = amax / levels
    q = np.clip(np.round(R / scale), -levels, levels)
    return (q * scale).astype(np.float32)


def lowrank_bytes(m, n, r, res_bits):
    """f16 U,V + residual @ res_bits + f16 per-row residual scale."""
    uv = 2 * (m * r + r * n)
    res = (res_bits * m * n) / 8.0
    res_scale = 2 * m
    return uv + res + res_scale


def func_rel_err(W, What, X):
    """Data-weighted relative error ||(W-What) X^T||_F / ||W X^T||_F on real X."""
    num = np.linalg.norm((W - What) @ X.T)
    den = np.linalg.norm(W @ X.T) + 1e-30
    return float(num / den)


def activation_effrank(X):
    """Effective rank of the activation matrix X [N, hidden].

    The DECISIVE artifact-detector for this oracle. If X spans an effective
    subspace of <= r dims, then C = X^T X / N has rank <= r, M = W @ C^{1/2} has
    <= r nonzero singular values, and data-norm energy@r = 1.0 TRIVIALLY for ANY
    weight W. So a data-E64 ~ 1.0 is meaningless unless X's effective rank >> 64.

    Returns:
      n_sv              : number of singular values (= min(N, hidden))
      participation     : (sum s_i^2)^2 / sum s_i^4  -- the effective # of dims
                          carrying energy (a.k.a. participation ratio / IPR).
      rank99, rank999   : # of singular values to reach 99% / 99.9% of X energy.
      sv                : the singular values (float64), for the report curve.
    """
    sv = np.linalg.svd(X, compute_uv=False).astype(np.float64)
    e = sv ** 2
    tot = float(e.sum()) + 1e-30
    participation = (tot ** 2) / (float((e ** 2).sum()) + 1e-30)
    cum = np.cumsum(e) / tot
    rank99 = int(np.searchsorted(cum, 0.99) + 1)
    rank999 = int(np.searchsorted(cum, 0.999) + 1)
    return dict(n_sv=int(sv.size), participation=float(participation),
                rank99=rank99, rank999=rank999, sv=sv)


def dataaware_basis(W, Csqrt, Cinv):
    """SVD of the data-whitened weight M = W @ C^{1/2}; return (U, S, Vt @ Cinv).

    W_r = (U[:, :r] * S[:r]) @ (Vt[:r] @ Cinv) is the rank-r approximation
    optimal for the data norm ||(W - W_r) X^T|| when C = E[x x^T]. We precompute
    VtCinv = Vt @ Cinv once so per-rank reconstruction is a cheap slice+matmul.
    """
    M = W @ Csqrt
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    del M
    VtCinv = Vt @ Cinv
    return U, S, VtCinv


def lowrank_recon(U, S, VtCinv, r):
    """Rank-r data-aware reconstruction W_r from a precomputed basis."""
    rr = min(r, S.size)
    A = U[:, :rr] * S[:rr]          # [m, rr]
    return A @ VtCinv[:rr, :]       # [m, n]


def heldout_data_norm_error(W, X_tr, X_te, ranks):
    """THE decisive check: fit the activation-aware low-rank basis on TRAIN-token
    statistics, then measure data-norm reconstruction error
        ||(W - W_r) x|| / ||W x||
    on BOTH the train tokens (in-sample) and the HELD-OUT tokens, at each rank.

    If held-out error ~ in-sample error AND both ~0, the low-rankness is real on
    the data manifold (GO). If held-out blows up vs in-sample, the in-sample
    energy@r was an illusion of overfitting the captured tokens (Type-1). If both
    are ~0 only because X's effective rank <= r (see activation_effrank), the
    'codec' is exploiting low ACTIVATION rank, not a low-rank WEIGHT property --
    flagged separately in the verdict.
    """
    C_tr = (X_tr.T @ X_tr) / X_tr.shape[0]
    Csqrt, Cinv = csqrt_and_inv(C_tr)
    del C_tr
    U, S, VtCinv = dataaware_basis(W, Csqrt, Cinv)
    out = {}
    for r in ranks:
        Wr = lowrank_recon(U, S, VtCinv, r)
        out[r] = dict(in_sample=func_rel_err(W, Wr, X_tr),
                      held_out=func_rel_err(W, Wr, X_te))
        del Wr
    del U, S, VtCinv, Csqrt, Cinv
    gc.collect()
    return out


def data_energy_curve(W, Csqrt, ranks):
    """Data-norm energy fraction at each rank (cumulative s_i^2 of W @ C^{1/2})."""
    M = W @ Csqrt
    sv = np.linalg.svd(M, compute_uv=False)
    del M
    e = sv.astype(np.float64) ** 2
    tot = float(e.sum()) + 1e-30
    cum = np.cumsum(e) / tot
    return {r: float(cum[min(r, e.size) - 1]) for r in ranks}


# --------------------------------------------------------------------------
# Per-tensor data-aware low-rank analysis (L1.4 reframe)
# --------------------------------------------------------------------------
def analyze_tensor_l14(W, X, disk_bytes, Csqrt, Cinv):
    """Full-token (in-sample) byte/functional-error budget + energy curves.

    NOTE: this uses ALL captured tokens to fit C (Csqrt/Cinv passed in) — it is
    the IN-SAMPLE view (matches the original oracle). The decisive generalization
    test is the separate held-out analysis (heldout_data_norm_error)."""
    m, n = W.shape  # [out, in]
    # plain (Frobenius) SVD energy — reproduce the ORIGINAL oracle's number,
    # over the full curve.
    sv_plain = np.linalg.svd(W, compute_uv=False)
    e_plain = topr_energy(sv_plain, CURVE_RANKS)
    del sv_plain

    # data-aware SVD of M = W @ C^{1/2}
    M = W @ Csqrt
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    del M
    e_data = topr_energy(S, CURVE_RANKS)
    VtCinv = Vt @ Cinv
    del Vt

    rows = []
    for r in RANKS:
        rr = min(r, S.size)
        # W_r optimal for the data norm: (U_r S_r)(V_r^T C^{-1/2})
        Wr = (U[:, :rr] * S[:rr]) @ VtCinv[:rr, :]
        resid = W - Wr
        res_std_ratio = float(resid.std() / (W.std() + 1e-30))
        for b in RES_BITS:
            rq = uniform_quant_resid(resid, b)
            What = Wr + rq
            ferr = func_rel_err(W, What, X)
            tot_bytes = lowrank_bytes(m, n, rr, b)
            rows.append(dict(r=rr, bits=b,
                             energy_data=e_data[r], energy_plain=e_plain[r],
                             res_std_ratio=res_std_ratio,
                             bytes_ratio=tot_bytes / disk_bytes,
                             func_err=ferr))
            del rq, What
        del Wr, resid
        gc.collect()
    del U, S, VtCinv
    gc.collect()
    return rows, e_plain, e_data


# --------------------------------------------------------------------------
# Cross-layer delta data-aware analysis (L1.3 reframe)
# --------------------------------------------------------------------------
def analyze_pair_l13(W_L, W_Lp1, X_Lp1, Csqrt):
    """Does the resident W[L] give W[L+1] a cheap basis in the data norm?
    Compare data-aware energy@r of the DELTA vs of W[L+1] alone."""
    D = W_Lp1 - W_L
    Md = D @ Csqrt
    svd = np.linalg.svd(Md, compute_uv=False)
    e_delta = topr_energy(svd, RANKS)
    del Md, svd
    Mw = W_Lp1 @ Csqrt
    svw = np.linalg.svd(Mw, compute_uv=False)
    e_w = topr_energy(svw, RANKS)
    del Mw, svw
    # data-weighted cross-layer cosine of the actions on real X
    yL = W_L @ X_Lp1.T
    yP = W_Lp1 @ X_Lp1.T
    cos = float((yL.ravel() @ yP.ravel()) /
                (np.linalg.norm(yL) * np.linalg.norm(yP) + 1e-30))
    del yL, yP, D
    gc.collect()
    return e_delta, e_w, cos


def main():
    ap = argparse.ArgumentParser(prog="oracle_dataaware_lowrank")
    ap.add_argument("--gguf", default="models/qwen2.5-3b-instruct-q4_k_m.gguf")
    ap.add_argument("--bin", default="_capture/q3b_ffn.bin")
    ap.add_argument("--out", default="reports/oracle_dataaware_lowrank.md")
    ap.add_argument("--layers", default="all",
                    help="comma list, 'all', or 'sample' (0/9/18/27/35). "
                         "Default 'all' — the machine is free and the tensors "
                         "are tiny; the original early/mid/late sample was a "
                         "busy-machine artifact.")
    ap.add_argument("--max-tokens", type=int, default=800,
                    help="cap on captured tokens used per layer (default 800 = "
                         "all of them; the held-out split is taken AFTER this).")
    ap.add_argument("--held-frac", type=float, default=1.0 - TRAIN_FRAC,
                    help="fraction of tokens held out for the decisive "
                         "generalization test (default 0.30).")
    ap.add_argument("--tensors", default="ffn_gate,ffn_up",
                    help="FFN tensors whose input == norm_in (gate/up)")
    args = ap.parse_args()
    t0 = time.time()

    print(f"[oracle] reading capture {args.bin} ...", flush=True)
    hidden, n_blocks, norm = load_norm_in(Path(args.bin))
    layers_all = sorted(norm.keys())
    print(f"[oracle] hidden={hidden} layers={len(layers_all)} "
          f"tokens/layer={norm[layers_all[0]].shape[0]}", flush=True)

    if args.layers == "all":
        layers = list(layers_all)
    elif args.layers == "sample":
        idx = sorted(set([0, len(layers_all) // 4, len(layers_all) // 2,
                          (3 * len(layers_all)) // 4, len(layers_all) - 1]))
        layers = [layers_all[i] for i in idx]
    else:
        layers = [int(x) for x in args.layers.split(",")]
    tnames = args.tensors.split(",")
    rng = np.random.default_rng(0)  # fixed seed -> reproducible train/held split

    print(f"[oracle] reading GGUF {args.gguf} ...", flush=True)
    reader = GGUFReader(args.gguf)
    by_name = {t.name: t for t in reader.tensors}

    l14_results = {}   # (layer, tname) -> (rows, qtype)
    l14_energy = {}    # (layer, tname) -> (e_plain, e_data)  full curves
    l14_heldout = {}   # (layer, tname) -> {r: {in_sample, held_out}}
    l13_results = {}   # (L, tname) -> (e_delta, e_w, cos)
    actrank = {}       # layer -> activation_effrank(X) dict (no sv array)
    max_rss = rss_gb()

    for layer in layers:
        X = norm[layer]
        if X.shape[0] > args.max_tokens:
            sel = np.linspace(0, X.shape[0] - 1, args.max_tokens).astype(int)
            X = X[sel]

        # --- activation effective rank (the artifact detector) ---
        er = activation_effrank(X)
        actrank[layer] = {k: v for k, v in er.items() if k != "sv"}
        print(f"[oracle] L{layer:2d} X{X.shape} participation={er['participation']:.1f} "
              f"rank99%={er['rank99']} rank99.9%={er['rank999']}", flush=True)

        # --- decisive train/held-out token split (seeded shuffle) ---
        n_tok = X.shape[0]
        perm = rng.permutation(n_tok)
        n_held = max(1, int(round(args.held_frac * n_tok)))
        te_idx, tr_idx = perm[:n_held], perm[n_held:]
        X_tr, X_te = X[tr_idx], X[te_idx]

        # full-token covariance for the in-sample byte/energy budget (orig view)
        C = (X.T @ X) / X.shape[0]
        Csqrt, Cinv = csqrt_and_inv(C)
        del C
        for tname in tnames:
            nm = f"blk.{layer}.{tname}.weight"
            if nm not in by_name:
                continue
            W, qtype, disk_bytes = w_of(by_name, nm)
            rows, e_plain, e_data = analyze_tensor_l14(W, X, disk_bytes, Csqrt, Cinv)
            l14_results[(layer, tname)] = (rows, qtype)
            l14_energy[(layer, tname)] = (e_plain, e_data)
            # THE decisive check: held-out data-norm error (basis fit on TRAIN).
            ho = heldout_data_norm_error(W, X_tr, X_te, HELDOUT_RANKS)
            l14_heldout[(layer, tname)] = ho
            max_rss = max(max_rss, check_rss(nm))
            best = min(rows, key=lambda r: (r["func_err"]
                                            if r["bytes_ratio"] < 1.0 else 9e9))
            print(f"[oracle] L{layer:2d} {tname:8s} {qtype} "
                  f"E64 plain={e_plain[64]:.3f} data={e_data[64]:.3f} | "
                  f"r64 held-out ferr in={ho[64]['in_sample']:.4f} "
                  f"out={ho[64]['held_out']:.4f} | "
                  f"best<Q4K r{best['r']}/{best['bits']}b "
                  f"bytes={best['bytes_ratio']:.2f}x | rss={rss_gb():.2f}G",
                  flush=True)
            del W
            gc.collect()

        # L1.3: pair this layer with the next (if both captured & adjacent).
        nxt = layer + 1
        if nxt in norm:
            Xp = norm[nxt]
            if Xp.shape[0] > args.max_tokens:
                sel = np.linspace(0, Xp.shape[0] - 1, args.max_tokens).astype(int)
                Xp = Xp[sel]
            Cp = (Xp.T @ Xp) / Xp.shape[0]
            Csqrt_p, _ = csqrt_and_inv(Cp)
            del Cp
            for tname in tnames:
                a = f"blk.{layer}.{tname}.weight"
                bnm = f"blk.{nxt}.{tname}.weight"
                if a not in by_name or bnm not in by_name:
                    continue
                W_L, _, _ = w_of(by_name, a)
                W_P, _, _ = w_of(by_name, bnm)
                e_delta, e_w, cos = analyze_pair_l13(W_L, W_P, Xp, Csqrt_p)
                l13_results[(layer, tname)] = (e_delta, e_w, cos)
                print(f"[oracle] L{layer}->{nxt} {tname:8s} dw-cos={cos:+.4f} "
                      f"E64(delta)={e_delta[64]:.3f} E64(W)={e_w[64]:.3f}",
                      flush=True)
                del W_L, W_P
                gc.collect()
            del Csqrt_p
        del Csqrt, Cinv, X_tr, X_te
        gc.collect()

    # ---------------- verdict ----------------
    # The in-sample byte/functional budget (original gate): does some <Q4K-byte
    # config clear the functional-error gate on the FULL captured tokens?
    keys = list(l14_results.keys())
    passed = 0
    per_key_best = {}
    for k in keys:
        rows, _ = l14_results[k]
        cands = [r for r in rows if r["bytes_ratio"] < 1.0 and r["func_err"] <= FUNC_ERR_GATE]
        best = min(rows, key=lambda r: r["func_err"]) if rows else None
        per_key_best[k] = (cands[0] if cands else best)
        if cands:
            passed += 1
    insample_go = passed > len(keys) / 2 if keys else False

    # --- THE decisive gate: held-out generalization + activation effective rank.
    # data-E64 ~ 1.0 is TRIVIAL whenever the activation matrix X has effective
    # rank <= 64 (then C is rank-deficient and EVERY weight is "low-rank in the
    # data norm"). So a GO requires ALL of:
    #   (a) held-out data-norm error at r=64 stays low (generalizes), AND
    #   (b) held-out ~ in-sample (no overfitting blow-up), AND
    #   (c) the activation effective rank is MUCH larger than 64 (so r=64
    #       capturing the energy is a non-trivial WEIGHT property, not a
    #       restatement that the activations themselves live in <=64 dims), AND
    #   (d) it beats Q4_K bytes (in-sample gate above).
    # Otherwise the data-E64~1.0 is either an in-sample illusion (Type-1) or a
    # low-activation-rank artifact (the codec saves no weight bytes, because the
    # full W is still needed for any future token that rotates out of the
    # captured subspace).
    HELDOUT_ERR_GATE = FUNC_ERR_GATE          # held-out r=64 must be this small
    BLOWUP_FACTOR = 3.0                       # held-out/in-sample blow-up ceiling
    ACTRANK_MARGIN = 2.0                      # X rank99 must exceed 2x the rank
    R_DECISIVE = 64

    per_key_decision = {}
    n_generalize = 0          # held-out low AND not blown up vs in-sample
    n_blowup = 0              # held-out >> in-sample  (Type-1 overfit)
    for k in keys:
        layer, _ = k
        ho = l14_heldout[k][R_DECISIVE]
        ins, out_ = ho["in_sample"], ho["held_out"]
        blow = out_ / (ins + 1e-12)
        x_rank99 = actrank[layer]["rank99"]
        generalizes = (out_ <= HELDOUT_ERR_GATE) and (blow <= BLOWUP_FACTOR)
        nontrivial_rank = x_rank99 > ACTRANK_MARGIN * R_DECISIVE
        if out_ > BLOWUP_FACTOR * (ins + 1e-12) and out_ > HELDOUT_ERR_GATE:
            n_blowup += 1
        if generalizes:
            n_generalize += 1
        per_key_decision[k] = dict(in_sample=ins, held_out=out_, blowup=blow,
                                   x_rank99=x_rank99, generalizes=generalizes,
                                   nontrivial_rank=nontrivial_rank)

    # Is the data-E64~1.0 explained by low activation rank across the board?
    low_actrank = sum(1 for l in actrank
                      if actrank[l]["rank99"] <= ACTRANK_MARGIN * R_DECISIVE)
    actrank_trivial = low_actrank > len(actrank) / 2 if actrank else False

    # L1.4 GO only if held-out generalizes for a majority AND the energy is not a
    # low-activation-rank artifact AND the in-sample byte gate also clears.
    majority_generalize = n_generalize > len(keys) / 2 if keys else False
    l14_go = bool(majority_generalize and insample_go and not actrank_trivial)

    # Precedence: the low-activation-rank artifact is the STRUCTURAL cause when it
    # holds (it makes data-E64~1.0 trivial for EVERY W, independent of held-out),
    # so it leads the headline; the held-out blow-up is corroborating Type-1
    # evidence. (At very-low-rank layers the in-sample error is already large, so
    # the blow-up RATIO gate is conservative and under-counts — the artifact gate
    # is the dominant kill.)
    if l14_go:
        l14_type = "GO: real low-rank-in-data-norm that generalizes and beats Q4_K"
    elif actrank_trivial:
        extra = " (held-out also blows up vs in-sample)" if n_blowup > 0 else ""
        l14_type = ("artifact (low activation rank): X effective rank ~ rank-%d, "
                    "so data-E%d~1.0 is trivial for ANY weight and saves no WEIGHT "
                    "bytes%s" % (R_DECISIVE, R_DECISIVE, extra))
    elif n_blowup > 0:
        l14_type = "Type-1 (in-sample illusion): held-out data-norm error blows up vs in-sample"
    else:
        l14_type = "NO-GO: does not beat Q4_K bytes at the held-out error gate"

    # L1.3 GO if the delta needs materially LESS rank than W alone (W[L] helps).
    l13_helps = 0
    for k, (e_delta, e_w, cos) in l13_results.items():
        if e_delta[64] > e_w[64] + 0.10:  # delta noticeably more concentrated
            l13_helps += 1
    l13_go = l13_helps > len(l13_results) / 2 if l13_results else False

    overall = "GO" if l14_go else "NO-GO"

    # convenience aggregates for the report
    def _mean(vals):
        vals = list(vals)
        return float(sum(vals) / len(vals)) if vals else float("nan")

    mean_E64_data = _mean(l14_energy[k][1][64] for k in keys)
    mean_E64_plain = _mean(l14_energy[k][0][64] for k in keys)
    mean_x_rank99 = _mean(actrank[l]["rank99"] for l in actrank)
    mean_x_part = _mean(actrank[l]["participation"] for l in actrank)
    mean_ho_r64 = _mean(per_key_decision[k]["held_out"] for k in keys)
    mean_ins_r64 = _mean(per_key_decision[k]["in_sample"] for k in keys)

    # ---------------- report ----------------
    L = []
    P = L.append
    P("# Reframe oracle — data-AWARE low-rank (L1.4) + cross-layer reference (L1.3)")
    P("")
    P(f"**L1.4 verdict (data-aware): {overall}** — {l14_type}  ")
    P(f"**L1.3 verdict (cross-layer, data-aware): {'GO' if l13_go else 'NO-GO'}**")
    P("")
    P("**The three deciding numbers (mean over sampled FFN tensors, r=64):**  ")
    P(f"- data-norm energy@64 = **{mean_E64_data:.3f}** (vs Frobenius {mean_E64_plain:.3f})  ")
    P(f"- activation X effective rank (99% energy) = **{mean_x_rank99:.0f}** "
      f"(participation ratio {mean_x_part:.1f} of {hidden})  ")
    P(f"- held-out data-norm error @64 = **{mean_ho_r64:.4f}** "
      f"(in-sample {mean_ins_r64:.4f})")
    P("")
    P(f"- Model: `{args.gguf}` | Capture: `{args.bin}`")
    P(f"- Sampled layers: {layers if len(layers)<=12 else f'{len(layers)} layers {layers[0]}..{layers[-1]}'} "
      f"| tensors: {tnames} (input == norm_in)")
    P(f"- Tokens/layer (cap): {args.max_tokens} | train/held split: "
      f"{1.0-args.held_frac:.0%}/{args.held_frac:.0%} (seed 0) | "
      f"Peak RSS: {max_rss:.2f} GB | Wall: {time.time()-t0:.1f}s")
    P(f"- Functional-error gate (data-weighted rel-L2): {FUNC_ERR_GATE}")
    P("")
    P("## The reframe being tested — and the trap it has to clear")
    P("")
    P("The original L1.4/L1.3 oracles used a **data-free** SVD and **Frobenius** "
      "energy. This re-tests with **activation-aware** SVD (SVD on `W·C^{1/2}`, "
      "`C=E[xx^T]` from the real capture) — the standard fix (ASVD/SVD-LLM) the "
      "originals never ran. Weights can be full-rank in Frobenius yet low-rank in "
      "the data norm.")
    P("")
    P("**But a high data-norm energy@r is suspect.** If the captured activation "
      "matrix `X` spans an effective subspace of `<= r` dims, then `C = X^T X/N` "
      "is rank-deficient, `M = W·C^{1/2}` has `<= r` nonzero singular values, and "
      "data-norm energy@r = 1.0 **trivially for EVERY weight W**. That is not a "
      "low-rank-WEIGHT property — it is a low-rank-ACTIVATION fact, and it saves "
      "no weight bytes (the full `W` is still needed for any future token whose "
      "activations rotate out of the captured subspace). This oracle therefore "
      "adds two checks the striking partial result demanded: (1) the **effective "
      "rank of X** (is `data-E64~1.0` trivial?), and (2) the **held-out-token** "
      "data-norm error (does the rank-r fit generalize, or is it overfitting the "
      "~%d captured tokens?). These are the burned-twice discipline (90%%/33%% "
      "EAGLE; 86.8%%/45%% prefix-cache) applied to L1.4." % args.max_tokens)
    P("")
    P("## L1.4a — Activation effective rank (the artifact detector)")
    P("")
    P("`participation` = `(Σσ²)²/Σσ⁴` — the effective # of dims of `X` carrying "
      "energy. `rank99%`/`rank99.9%` = singular values to reach that fraction of "
      "`X`'s energy. **If these are `<= ~64`, the data-E64 below is trivial.**")
    P("")
    P(f"| layer | X shape | participation /{hidden} | rank99% | rank99.9% |")
    P("|------:|---------|------------------------:|--------:|----------:|")
    for layer in layers:
        er = actrank[layer]
        P(f"| {layer} | {min(args.max_tokens, norm[layer].shape[0])}x{hidden} | "
          f"{er['participation']:.1f} | {er['rank99']} | {er['rank999']} |")
    P("")
    P(f"**Read:** mean participation ratio **{mean_x_part:.1f}** of {hidden}, mean "
      f"rank99% **{mean_x_rank99:.0f}**. " +
      ("Since this is `<= ~%dx` rank-64, the captured activations live in a "
       "subspace barely larger than (or smaller than) rank-64 — so `data-E64~1.0` "
       "is the **trivial low-activation-rank artifact**, not a low-rank weight."
       % ACTRANK_MARGIN
       if actrank_trivial else
       "Since this is `>> 64`, a rank-64 data-norm fit capturing the energy would "
       "be a non-trivial weight property, not a restatement of low activation "
       "rank."))
    P("")
    P("## L1.4b — Energy-vs-rank curve (data-aware vs Frobenius)")
    P("")
    P("`E_data@r` = activation-aware cumulative energy. `E_plain@r` = the original "
      "Frobenius number. If `data >> plain`, the original oracle undersold "
      "low-rank **in the data norm** (but see L1.4a/L1.4c for whether that is real).")
    P("")
    hdr = "| layer | tensor |" + "".join(f" E_data@{r} |" for r in CURVE_RANKS) + \
          "".join(f" E_plain@{r} |" for r in CURVE_RANKS)
    sep = "|------:|--------|" + "------:|" * (2 * len(CURVE_RANKS))
    P(hdr); P(sep)
    for k in keys:
        layer, tname = k
        e_plain, e_data = l14_energy[k]
        row = f"| {layer} | {tname} |"
        row += "".join(f" {e_data[r]:.3f} |" for r in CURVE_RANKS)
        row += "".join(f" {e_plain[r]:.3f} |" for r in CURVE_RANKS)
        P(row)
    P("")
    P("## L1.4c — THE decisive check: in-sample vs HELD-OUT data-norm error")
    P("")
    P("The rank-r data-aware basis (whitened SVD of `W` via the **train**-token "
      "covariance) is fit on 70% of tokens; error `||(W-Wᵣ)x||/||Wx||` is then "
      "measured on the same train tokens (**in-sample**) and on the unseen 30% "
      "(**held-out**). **Held-out ≈ in-sample ≈ low ⇒ real. Held-out ≫ in-sample "
      "⇒ in-sample illusion (Type-1).**")
    P("")
    hdr = "| layer | tensor |" + "".join(f" in/out r{r} |" for r in HELDOUT_RANKS)
    sep = "|------:|--------|" + "----------:|" * len(HELDOUT_RANKS)
    P(hdr); P(sep)
    for k in keys:
        layer, tname = k
        ho = l14_heldout[k]
        row = f"| {layer} | {tname} |"
        row += "".join(f" {ho[r]['in_sample']:.3f}/{ho[r]['held_out']:.3f} |"
                       for r in HELDOUT_RANKS)
        P(row)
    P("")
    P(f"**Read (r=64):** mean in-sample **{mean_ins_r64:.4f}**, mean held-out "
      f"**{mean_ho_r64:.4f}** (blow-up {mean_ho_r64/(mean_ins_r64+1e-12):.1f}x). " +
      (f"{n_blowup}/{len(keys)} tensors show a held-out blow-up > {BLOWUP_FACTOR}x "
       "AND above the gate — the in-sample energy was an **overfitting illusion** "
       "(Type-1)." if n_blowup > 0 else
       "Held-out tracks in-sample (no overfitting blow-up) — the rank-r fit "
       "generalizes across the captured tokens."))
    P("")
    P("## L1.4d — Byte budget vs Q4_K (in-sample, full tokens)")
    P("")
    P("`best<Q4K` = the smallest-functional-error config whose total bytes "
      "(f16 U,V + b-bit residual + per-row scale) are below the tensor's current "
      "Q4_K footprint. Target `W` is the **dequantized Q4_K** weight (pessimistic "
      "lower bound; see caveats).")
    P("")
    P("| layer | tensor | type | E64 data | best<Q4K (r/bits) | bytes | func-err |")
    P("|------:|--------|------|---------:|-------------------|------:|---------:|")
    for k in keys:
        layer, tname = k
        _, e_data = l14_energy[k]
        _, qtype = l14_results[k]
        b = per_key_best[k]
        bcfg = f"r{b['r']}/{b['bits']}b" if b else "-"
        P(f"| {layer} | {tname} | {qtype} | {e_data[64]:.3f} | "
          f"{bcfg} | {b['bytes_ratio']:.2f}x | {b['func_err']:.4f} |")
    P("")
    P(f"**L1.4 byte gate:** {passed}/{len(keys)} sampled FFN tensors have a "
      f"<Q4_K-byte config with in-sample data-weighted functional error <= "
      f"{FUNC_ERR_GATE}.")
    P("")
    P(f"**L1.4 VERDICT: {overall}** — {l14_type}. " +
      ("Held-out generalizes, activations are genuinely high-rank, and the codec "
       "beats Q4_K bytes — the data-free kill was premature (Type-2); advance to "
       "the GPU/quality lane on **f16** weights (the Q4_K re-encode here is a "
       "lower bound; real gains need AWQ-from-f16)."
       if l14_go else
       "The activation-aware energy@64 looks high, but it does NOT survive the "
       "decisive checks: either the held-out error blows up (Type-1 in-sample "
       "illusion) or — the operative case here — the captured activations are "
       "themselves effectively rank-<=64, so `data-E64~1.0` is a low-ACTIVATION-"
       "rank fact that saves no WEIGHT bytes. A low-rank-in-data-norm WEIGHT codec "
       "would still have to store the full row-space of `W` to serve tokens whose "
       "activations rotate out of the captured subspace. L1.4 stays **dead** in "
       "the data-aware reframe too; do not build the UV codec. Mixed-precision / "
       "QTIP (bible §2) remain the live byte-cut levers."))
    P("")
    P("## L1.3 — does the resident W[L] give W[L+1] a free basis (data-aware)?")
    P("")
    P("| L→L+1 | tensor | data-wt cross-layer cos | E64(delta) | E64(W[L+1]) | W[L] helps? |")
    P("|-------|--------|------------------------:|-----------:|------------:|:-----------:|")
    for (layer, tname), (e_delta, e_w, cos) in l13_results.items():
        helps = "yes" if e_delta[64] > e_w[64] + 0.10 else "no"
        P(f"| {layer}→{layer+1} | {tname} | {cos:+.4f} | {e_delta[64]:.3f} | "
          f"{e_w[64]:.3f} | {helps} |")
    P("")
    P("**L1.3:** " + (
        "the delta concentrates materially more data-weighted energy than W[L+1] "
        "alone in a majority of pairs — W[L] is a useful free basis; a data-aware "
        "cross-layer codec is worth a prototype."
        if l13_go else
        "the delta is no more concentrated than W[L+1] alone (and the data-weighted "
        "cross-layer cosine is ~0), so the resident W[L] provides no free basis — "
        "cross-layer reference adds nothing over plain L1.4 even in the data norm. "
        "Type-1: layers are functionally independent, not just weight-orthogonal."))
    P("")
    P("## Honest caveats")
    P("")
    P("- Target W is the **dequantized Q4_K** weight (only the Q4_K_M GGUF is "
      "available offline), so this re-encodes an already-quantized weight — a "
      "**lower bound** on what AWQ-from-f16 could achieve. A NO-GO is decisive; a "
      "marginal GO must be re-checked on f16 in the GPU/quality lane with real KL.")
    P("- Functional error is data-weighted rel-L2 on the captured activations, a "
      "proxy for end-task quality (same proxy family as the Track-B / co-activation "
      "gates), not perplexity.")
    P("- Scope: gate_proj + up_proj (input == norm_in, captured directly). "
      "down_proj (input = 11008-dim SwiGLU intermediate) is reconstructable "
      "(see oracle_coactivation_permute.py) but omitted here; add it if gate/up "
      "show a GO.")
    P("- best-case ORACLE: the per-row residual quantizer is idealized; a real "
      "GPU codec is no better.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")

    sidecar = out.with_suffix(".json")
    json.dump({
        "l14_verdict": overall,
        "l14_type": l14_type,
        "l13_verdict": "GO" if l13_go else "NO-GO",
        "func_err_gate": FUNC_ERR_GATE,
        "heldout_err_gate": HELDOUT_ERR_GATE,
        "blowup_factor": BLOWUP_FACTOR,
        "actrank_margin": ACTRANK_MARGIN,
        "decisive_rank": R_DECISIVE,
        "deciding_numbers": {
            "mean_E64_data": mean_E64_data,
            "mean_E64_plain": mean_E64_plain,
            "mean_X_rank99": mean_x_rank99,
            "mean_X_participation": mean_x_part,
            "mean_heldout_err_r64": mean_ho_r64,
            "mean_insample_err_r64": mean_ins_r64,
        },
        "insample_byte_go": insample_go,
        "majority_generalize": majority_generalize,
        "actrank_trivial": actrank_trivial,
        "n_generalize": n_generalize, "n_blowup": n_blowup,
        "l14_byte_pass": passed, "l14_total": len(keys),
        "train_frac": 1.0 - args.held_frac, "held_frac": args.held_frac,
        "layers": layers, "tensors": tnames, "max_tokens": args.max_tokens,
        "actrank": {str(l): actrank[l] for l in actrank},
        "l14": {f"{k[0]}:{k[1]}": {
            "E_data_curve": l14_energy[k][1],
            "E_plain_curve": l14_energy[k][0],
            "heldout": l14_heldout[k],
            "decision": per_key_decision[k],
            "rows": l14_results[k][0]} for k in keys},
        "l13": {f"{k[0]}:{k[1]}": {"cos": v[2],
                                   "E64_delta": v[0][64], "E64_W": v[1][64]}
                for k, v in l13_results.items()},
        "peak_rss_gb": max_rss,
    }, open(sidecar, "w"), indent=2)

    print(f"\n[oracle] L1.4 {overall} ({l14_type})")
    print(f"[oracle]   byte-gate {passed}/{len(keys)}, generalize {n_generalize}/{len(keys)}, "
          f"blowup {n_blowup}/{len(keys)}, actrank_trivial={actrank_trivial}")
    print(f"[oracle]   deciders: data-E64={mean_E64_data:.3f}  X-rank99={mean_x_rank99:.0f}  "
          f"held-out-err@64={mean_ho_r64:.4f}")
    print(f"[oracle] L1.3 {'GO' if l13_go else 'NO-GO'}")
    print(f"[oracle] wrote {out} and {sidecar} ({time.time()-t0:.1f}s, "
          f"peak RSS {max_rss:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

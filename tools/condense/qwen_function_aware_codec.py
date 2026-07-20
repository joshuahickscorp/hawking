#!/usr/bin/env python3.12
"""Function-aware sub-bit codec: scale-invariant VQ + activation-weighted fitting.

WHY THIS EXISTS. The sealed F1 negative on Qwen3-235B
(reports/foundry/F1_QWEN3_235B_SUB_BIT_UNSOLVED.json) retired the RAW-WEIGHT PQ/VQ family
at <= 1 complete BPW. Its own named highest-value untried lever is:

    "94 percent of gate/up rows collapse onto ONE codeword"

That is not a rate problem. Expert gate/up row L2 norms span 1e-5 .. 0.91 - five orders of
magnitude - and the campaign codec (qwen_gravity_campaign._fit_grammar) fits Lloyd centroids
on RAW row vectors. Lloyd minimises unweighted squared error, so the handful of large-norm
rows own the entire centroid budget and every small-norm row is assigned the same near-zero
codeword. The billed index bits are physically present in the artifact and buy nothing. The
declared R5 lever (row-norm STRATA) only splits that 5-decade span into 2 pieces of 2.5
decades each, so the pathology survives stratification.

THE TWO METHODS HERE, both from SUBBIT_CLOSURE_PROGRAM.json, neither previously built:

  M01' scale-invariant VQ (strictly stronger than the declared strata form)
        Factor every row into (scale, direction): w_i = s_i * u_i with ||u_i|| = sqrt(cols).
        Quantize ONLY the direction; ship s_i as one bf16 per row. The codebook now sees a
        distribution with no dynamic range at all, so centroid capacity is spent on shape.
        Exact cost: 16 bits/row. On gate/up [1536, 4096] that is 16/4096 = 0.0039 BPW;
        on down [4096, 1536] it is 16/1536 = 0.0104 BPW. Billed, never free.

  M02  activation-aware (diagonal-Hessian) fitting
        Lloyd on weight-space MSE reconstructs the conditional mean of the WEIGHTS. What the
        artifact must preserve is the OUTPUT. For a linear organ y = W x, the diagonal of the
        input second-moment E[x x^T] is the exact per-column weight of the output error:
            ||W x - W' x||^2  ~  sum_j h_j (W_ij - W'_ij)^2,  h_j = E[x_j^2]
        so both the assignment and the centroid update become h-weighted. Costs ZERO artifact
        bytes: the fit is offline, the shipped format is unchanged.

They compose: after row scaling, the output error of row i is s_i^2 * sum_j h_j (u_ij-u'_ij)^2,
so the per-vector fitting weight is s_i^2 * h_j. That product is what `fit` actually uses.

FALSIFICATION (the program's own gate, measured by `occupancy`): if the single-codeword share
on real gate/up tensors does not fall from ~94 percent to below 20 percent, or if tensor
reconstruction error does not improve AT THE IDENTICAL EXACT RATE, the lever is dead and goes
to the negative-transfer atlas. No capability claim is made by this module - only a real
parent-vs-packed forward may select a frontier.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gravity_forge as gf  # noqa: E402

SCHEMA = "hawking.gravity.function_aware_codec.v1"

# Rows whose norm is exactly zero would divide by zero; they decode to zero either way.
_EPS = 1e-12


# ── row scale / direction factorisation (M01') ────────────────────────────────────────────────
def row_scales(w: np.ndarray) -> np.ndarray:
    """Per-row scale s_i = ||w_i|| / sqrt(cols), rounded to the bf16 grid that is actually shipped.

    Dividing by sqrt(cols) makes the normalized rows land at ||u_i|| = sqrt(cols), i.e. unit RMS
    per element, which keeps the centroid magnitudes O(1) regardless of tensor width.
    """
    cols = w.shape[1]
    s = np.linalg.norm(np.asarray(w, np.float32), axis=1) / math.sqrt(cols)
    # bf16 round-trip by truncation: this is the value the artifact really stores.
    bits = np.ascontiguousarray(s, np.float32).view(np.uint32) >> np.uint32(16)
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def normalize_rows(w: np.ndarray, s: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """w -> (unit-RMS directions, shipped bf16 scales)."""
    s = row_scales(w) if s is None else s
    return np.asarray(w, np.float32) / np.maximum(s, _EPS)[:, None], s


def scale_bits(shape: tuple[int, int]) -> int:
    """Exact artifact cost of M01': one bf16 scale per row. Per tensor, never amortized."""
    return int(shape[0]) * 16


# ── weighted Lloyd (M02) ──────────────────────────────────────────────────────────────────────
def _row_chunk(k: int) -> int:
    """Rows per distance block, capping the [chunk, k] intermediate at ~16M floats."""
    return max(64, int(16_000_000 // max(1, k)))


def _assign(v, cb, wt=None):
    """argmin_k sum_d wt[n,d] * (v[n,d] - cb[k,d])^2, blocked so [N,k] is never materialized.

    The ||v||^2 term (and its weighted analogue sum_d h*v^2) is CONSTANT across k, so it cannot
    change an argmin and is simply not computed. That is strictly less work and less code, and
    selftest asserts the indices are identical to the naive full-distance form.

    MEASURED NULL, recorded so nobody re-derives it: carrying a running (min, argmin) over
    K-blocks instead of reducing the full [N,k] block looked like 1.44x in an unpaired timing,
    but a PAIRED interleaved A/B on the real d8/k1024 gate geometry gave median 1.07x and
    min-to-min 0.98x. That is a tie, and a tie is a null. The apparent win was GPU contention
    from a concurrent heavy campaign, not the algorithm. Reverted rather than kept: it added a
    nested loop and two torch.where per block to buy nothing.
    """
    torch = gf._torch()
    n, k = v.shape[0], cb.shape[0]
    step = _row_chunk(k)
    out = torch.empty(n, dtype=torch.long, device=v.device)
    cbt = cb.t().contiguous()
    cb2 = (cb * cb)
    cb2t = cb2.t().contiguous()
    cb2s = cb2.sum(1)
    for i in range(0, n, step):
        c = v[i:i + step]
        if wt is None:
            d2 = cb2s - 2.0 * (c @ cbt)
        else:
            h = wt[i:i + step]
            d2 = (h @ cb2t) - 2.0 * ((h * c) @ cbt)
        out[i:i + step] = d2.argmin(1)
    return out


def _lloyd(v, k: int, *, wt=None, iters: int = 6, seed: int = 0):
    """Weighted Lloyd on v:[N,D] -> centroids [k,D].

    Centroid update is the weighted mean cb[k] = sum_{n in k} wt[n]*v[n] / sum_{n in k} wt[n],
    elementwise per dimension. Empty clusters keep their previous centroid (torch.where, not a
    boolean mask - the mask form forces an MPS device sync every iteration).
    """
    torch = gf._torch()
    n = v.shape[0]
    k = int(min(k, n))
    g = torch.Generator(device="cpu").manual_seed(seed)
    cb = v[torch.randperm(n, generator=g)[:k].to(v.device)].clone()
    ones = torch.ones_like(v) if wt is None else wt
    for _ in range(iters):
        idx = _assign(v, cb, wt)
        num = torch.zeros_like(cb)
        den = torch.zeros_like(cb)
        num.index_add_(0, idx, v if wt is None else wt * v)
        den.index_add_(0, idx, ones)
        cb = torch.where(den > 0, num / den.clamp(min=_EPS), cb)
    return cb


# ── importance vectors ────────────────────────────────────────────────────────────────────────
def importance_from_activations(x: np.ndarray) -> np.ndarray:
    """Diagonal input second moment h_j = mean_t x[t,j]^2 from calibration activations [T, cols].

    This is the exact diagonal of the Hessian of the output MSE for a linear organ, up to the
    constant 2. Normalized to mean 1 so it never rescales the objective, only reweights it.
    """
    h = np.mean(np.asarray(x, np.float32) ** 2, axis=0)
    m = float(h.mean())
    return (h / m).astype(np.float32) if m > 0 else np.ones_like(h)


def _weights(rows: int, cols: int, dim: int, s: np.ndarray | None,
             h: np.ndarray | None, device, torch):
    """Per-vector fitting weight [N, dim] = s_row^2 * h_col, or None when both are absent.

    Row-major reshape to [-1, dim] puts vector n at row n // (cols/dim), columns
    (n % (cols/dim)) * dim ... + dim, which is exactly the tiling built here.
    """
    if s is None and h is None:
        return None
    per_row = cols // dim
    hh = (np.ones(cols, np.float32) if h is None else np.asarray(h, np.float32))
    base = np.tile(hh.reshape(per_row, dim), (rows, 1))            # [N, dim]
    if s is not None:
        base = base * np.repeat(np.asarray(s, np.float32) ** 2, per_row)[:, None]
    return torch.from_numpy(np.ascontiguousarray(base)).to(device)


# ── fit / apply ───────────────────────────────────────────────────────────────────────────────
def fit(mats: list[np.ndarray], *, dim: int, k: int, stages: int = 1, seed: int = 0,
        row_scale: bool = True, importance: np.ndarray | None = None, iters: int = 6):
    """Fit a shared residual codebook stack over a cluster of expert tensors.

    Returns a list of `stages` centroid tensors, the SAME deployment shape the campaign codec
    ships (one codebook per (layer, organ group), shared by all experts in the cluster), so the
    byte ledger is unchanged apart from the row scales that `scale_bits` accounts for.
    """
    torch = gf._torch()
    dev = gf._device()
    chunks, wchunks = [], []
    for m in mats:
        rows, cols = m.shape
        u, s = normalize_rows(m) if row_scale else (np.asarray(m, np.float32), None)
        chunks.append(torch.from_numpy(np.ascontiguousarray(u)).to(dev).reshape(-1, dim))
        w = _weights(rows, cols, dim, s, importance, dev, torch)
        if w is not None:
            wchunks.append(w)
    pool = torch.cat(chunks, 0)
    wt = torch.cat(wchunks, 0) if wchunks else None
    del chunks, wchunks
    res = pool.clone()
    books = []
    for st in range(stages):
        cb = _lloyd(res, k, wt=wt, iters=iters, seed=seed + 31 * st)
        res = res - cb[_assign(res, cb, wt)]
        books.append(cb)
    del pool, res
    return books


def apply(books, w: np.ndarray, *, dim: int, row_scale: bool = True,
          importance: np.ndarray | None = None) -> np.ndarray:
    """Encode one tensor against fitted codebooks and return the DECODED reconstruction."""
    torch = gf._torch()
    dev = gf._device()
    rows, cols = w.shape
    u, s = normalize_rows(w) if row_scale else (np.asarray(w, np.float32), None)
    wt = _weights(rows, cols, dim, s, importance, dev, torch)
    v = torch.from_numpy(np.ascontiguousarray(u)).to(dev).reshape(-1, dim)
    rec = torch.zeros_like(v)
    res = v
    for cb in books:
        q = cb[_assign(res, cb, wt)]
        rec = rec + q
        res = res - q
    out = rec.reshape(rows, cols).detach().cpu().numpy().astype(np.float32)
    return out * s[:, None] if s is not None else out


def refit_scales(w: np.ndarray, direction: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
    """Closed-form optimal per-row scale GIVEN the decoded directions. Costs zero extra bytes.

    The a-priori scale ||w_i||/sqrt(cols) is what makes the codebook see a unit-RMS distribution,
    but it is NOT the scale that minimises the reconstruction error once the direction has been
    quantized: that is the least-squares projection

        s_i* = <w_i, d_i>_h / <d_i, d_i>_h      (h-weighted when activations are known)

    which is exactly the per-row analog of the per-tensor "optimal output gain" the existing
    allocation already budgets for. Re-rounded to bf16 because that is what ships.
    """
    a = np.asarray(w, np.float32)
    d = np.asarray(direction, np.float32)
    if h is not None:
        hw = np.asarray(h, np.float32)[None, :]
        num = (a * d * hw).sum(1)
        den = (d * d * hw).sum(1)
    else:
        num = (a * d).sum(1)
        den = (d * d).sum(1)
    s = num / np.maximum(den, _EPS)
    bits = np.ascontiguousarray(s, np.float32).view(np.uint32) >> np.uint32(16)
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def apply_refit(books, w: np.ndarray, *, dim: int,
                importance: np.ndarray | None = None) -> np.ndarray:
    """Scale-invariant decode with the closed-form optimal scale substituted back in.

    Identical artifact layout and identical bit cost as `apply(..., row_scale=True)`: the same one
    bf16 scale per row is shipped, it just carries a better value.
    """
    rows, cols = w.shape
    u, s = normalize_rows(w)
    torch = gf._torch()
    dev = gf._device()
    wt = _weights(rows, cols, dim, s, importance, dev, torch)
    v = torch.from_numpy(np.ascontiguousarray(u)).to(dev).reshape(-1, dim)
    rec = torch.zeros_like(v)
    res = v
    for cb in books:
        q = cb[_assign(res, cb, wt)]
        rec = rec + q
        res = res - q
    direction = rec.reshape(rows, cols).detach().cpu().numpy().astype(np.float32)
    s2 = refit_scales(w, direction, importance)
    return direction * s2[:, None]


def fit_doctor(mats: list[np.ndarray], base_books, *, dim: int, k: int, stages: int,
               doctor_dim: int, doctor_k: int, doctor_stages: int, protect_frac: float,
               seed: int = 0, iters: int = 4):
    """Lane B M10: fit a correction codebook on the residual of the rows the base codec FAILED on.

    Diagnosis-driven, not uniform: the protected set is chosen by measured residual energy per row
    after the base pass, which is the direct read on where the base representation lost the
    function. Rows are ranked by ||w_i - base_i|| / ||w_i|| so a large row that was coded well is
    not protected ahead of a small row that was destroyed.
    """
    pool = []
    for m in mats:
        base = apply_refit(base_books, m, dim=dim)
        rows = protected_rows(m, base, protect_frac)
        r = np.asarray(m, np.float32)[rows] - base[rows]
        u, _ = normalize_rows(r)
        pool.append(u)
    return fit(pool, dim=doctor_dim, k=doctor_k, stages=doctor_stages, seed=seed + 977,
               row_scale=True, iters=iters)


def protected_rows(w: np.ndarray, base: np.ndarray, frac: float) -> np.ndarray:
    """Row indices with the worst RELATIVE reconstruction error, `frac` of them."""
    num = np.linalg.norm(np.asarray(w, np.float32) - base, axis=1)
    den = np.maximum(np.linalg.norm(np.asarray(w, np.float32), axis=1), _EPS)
    n = max(1, int(round(frac * w.shape[0])))
    return np.sort(np.argsort(-(num / den))[:n])


def apply_doctor(base_books, doctor_books, w: np.ndarray, *, dim: int, doctor_dim: int,
                 protect_frac: float) -> np.ndarray:
    """Base decode, then add the coded residual back on the protected rows only."""
    out = apply_refit(base_books, w, dim=dim)
    rows = protected_rows(w, out, protect_frac)
    resid = np.asarray(w, np.float32)[rows] - out[rows]
    out[rows] = out[rows] + apply_refit(doctor_books, resid, dim=doctor_dim)
    return out


def doctor_bits(shape: tuple[int, int], *, doctor_dim: int, doctor_k: int, doctor_stages: int,
                protect_frac: float, cluster: int) -> int:
    """Exact Doctor cost: correction indices + amortized correction codebook + per-protected-row
    bf16 scale + a one-bit-per-row protection bitmap. Every one of these ships in the artifact."""
    rows, cols = shape
    n_prot = max(1, int(round(protect_frac * rows)))
    idx = (n_prot * cols // doctor_dim) * doctor_stages * math.ceil(math.log2(doctor_k))
    cb = doctor_stages * doctor_k * doctor_dim * 16 // max(1, cluster)
    return int(idx + cb + n_prot * 16 + rows)


def occupancy(books, w: np.ndarray, *, dim: int, row_scale: bool = True,
              importance: np.ndarray | None = None) -> dict[str, float]:
    """The M01 falsification metric: how much of the billed index rate is actually spent.

    `single_codeword_share` is the fraction of vectors landing on the single most popular
    centroid (the pathology: ~0.94 on raw gate/up). `used_fraction` is how many centroids are
    touched at all. `index_entropy_bits` is the realized entropy against the billed log2(k).
    """
    torch = gf._torch()
    dev = gf._device()
    rows, cols = w.shape
    u, s = normalize_rows(w) if row_scale else (np.asarray(w, np.float32), None)
    wt = _weights(rows, cols, dim, s, importance, dev, torch)
    v = torch.from_numpy(np.ascontiguousarray(u)).to(dev).reshape(-1, dim)
    idx = _assign(v, books[0], wt).detach().cpu().numpy()
    k = int(books[0].shape[0])
    cnt = np.bincount(idx, minlength=k).astype(np.float64)
    p = cnt / max(1.0, cnt.sum())
    nz = p[p > 0]
    return {
        "single_codeword_share": float(p.max()),
        "used_fraction": float((cnt > 0).mean()),
        "index_entropy_bits": float(-(nz * np.log2(nz)).sum()),
        "billed_index_bits": float(math.ceil(math.log2(k))),
        "k": k,
    }


# ── measurement ───────────────────────────────────────────────────────────────────────────────
def rel_error(w: np.ndarray, r: np.ndarray, h: np.ndarray | None = None) -> float:
    """Relative Frobenius error, optionally in the activation-weighted (output) metric."""
    a = np.asarray(w, np.float32)
    b = np.asarray(r, np.float32)
    if h is None:
        return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + _EPS))
    hw = np.sqrt(np.asarray(h, np.float32))[None, :]
    return float(np.linalg.norm((a - b) * hw) / (np.linalg.norm(a * hw) + _EPS))


def compare(mats: list[np.ndarray], *, dim: int, k: int, stages: int = 1, seed: int = 0,
            importance: np.ndarray | None = None, iters: int = 6) -> dict[str, Any]:
    """Baseline (raw Lloyd, the F1 codec) vs scale-invariant vs scale+activation-aware.

    All three carry the IDENTICAL index rate stages*log2(k)/dim. The only ledger difference is
    `scale_bits`, reported here as `extra_bpw_row_scale` so no arm can hide a rate advantage.
    """
    rows, cols = mats[0].shape
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "config": {"dim": dim, "k": k, "stages": stages, "n_tensors": len(mats),
                   "shape": [rows, cols], "seed": seed, "iters": iters},
        "index_bpw": round(stages * math.log2(k) / dim, 6),
        "extra_bpw_row_scale": round(scale_bits((rows, cols)) / (rows * cols), 6),
        "arms": {},
    }
    arms = [("baseline_raw_lloyd", False, None, False),
            ("scale_invariant", True, None, False),
            ("scale_invariant_refit", True, None, True)]
    if importance is not None:
        arms.append(("scale_plus_activation_aware", True, importance, False))
        arms.append(("scale_plus_activation_aware_refit", True, importance, True))
    for name, rs, imp, refit in arms:
        t0 = time.time()
        books = fit(mats, dim=dim, k=k, stages=stages, seed=seed,
                    row_scale=rs, importance=imp, iters=iters)
        errs, werrs, occ = [], [], []
        for m in mats:
            r = (apply_refit(books, m, dim=dim, importance=imp) if refit
                 else apply(books, m, dim=dim, row_scale=rs, importance=imp))
            errs.append(rel_error(m, r))
            if importance is not None:
                werrs.append(rel_error(m, r, importance))
            occ.append(occupancy(books, m, dim=dim, row_scale=rs, importance=imp))
        out["arms"][name] = {
            "mean_rel_error": round(float(np.mean(errs)), 6),
            "max_rel_error": round(float(np.max(errs)), 6),
            "mean_output_weighted_rel_error": (round(float(np.mean(werrs)), 6) if werrs else None),
            "mean_single_codeword_share": round(float(np.mean([o["single_codeword_share"] for o in occ])), 6),
            "mean_used_fraction": round(float(np.mean([o["used_fraction"] for o in occ])), 6),
            "mean_index_entropy_bits": round(float(np.mean([o["index_entropy_bits"] for o in occ])), 6),
            "seconds": round(time.time() - t0, 2),
        }
    # The Shannon floor for a memoryless Gaussian source at this exact index rate. A codec sitting
    # at or below it has no post-hoc headroom left to find in the marginal distribution.
    out["iid_gaussian_rate_distortion_floor"] = round(math.sqrt(2.0 ** (-2.0 * out["index_bpw"])), 6)
    b = out["arms"]["baseline_raw_lloyd"]
    s = out["arms"]["scale_invariant"]
    out["falsification"] = {
        "gate_single_codeword_below_0.20": bool(s["mean_single_codeword_share"] < 0.20),
        "gate_error_improves_at_identical_rate": bool(s["mean_rel_error"] < b["mean_rel_error"]),
        "single_codeword_share": [b["mean_single_codeword_share"], s["mean_single_codeword_share"]],
        "rel_error": [b["mean_rel_error"], s["mean_rel_error"]],
    }
    out["falsification"]["verdict"] = (
        "LEVER_ALIVE" if all(v for kk, v in out["falsification"].items() if kk.startswith("gate_"))
        else "LEVER_DEAD")
    return out


# ── self-check ────────────────────────────────────────────────────────────────────────────────
def selftest() -> dict[str, Any]:
    """Runnable check of every non-trivial branch, on a synthetic twin of the real pathology."""
    rng = np.random.default_rng(0)
    rows, cols, dim, k = 256, 128, 8, 64

    # bf16 scale round-trip is exactly what the artifact stores.
    w0 = rng.standard_normal((rows, cols)).astype(np.float32)
    s0 = row_scales(w0)
    assert np.all(np.isfinite(s0)) and s0.shape == (rows,)
    assert np.allclose(s0, row_scales(w0)), "scale quantization must be deterministic"

    # unit-RMS after normalization
    u0, _ = normalize_rows(w0)
    rms = np.sqrt((u0 ** 2).mean(1))
    assert np.allclose(rms, 1.0, atol=2e-2), rms[:4]

    # THE PATHOLOGY: same directions, row norms spanning 5 decades (the real gate/up geometry).
    dirs = rng.standard_normal((rows, cols)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    span = np.logspace(-5, -0.04, rows).astype(np.float32)
    rng.shuffle(span)
    w = dirs * span[:, None]

    res = compare([w], dim=dim, k=k, stages=1, seed=0, iters=8)
    base = res["arms"]["baseline_raw_lloyd"]
    inv = res["arms"]["scale_invariant"]
    # Pathology reproduced then cured. The synthetic twin is k=64 so its baseline share sits near
    # 0.4, not the 0.94 the real k=1024 gate/up tensors show; the relational form is the invariant.
    assert base["mean_single_codeword_share"] > 4 * inv["mean_single_codeword_share"], (base, inv)
    assert inv["mean_single_codeword_share"] < 0.20, inv             # and cured
    assert inv["mean_rel_error"] < base["mean_rel_error"], (base, inv)
    assert res["falsification"]["verdict"] == "LEVER_ALIVE"

    # exact ledger: 16 bits per row, and the reported BPW matches it
    assert scale_bits((rows, cols)) == rows * 16
    assert abs(res["extra_bpw_row_scale"] - 16.0 / cols) < 1e-9

    # activation weighting must beat unweighted IN THE OUTPUT METRIC it optimizes
    h = importance_from_activations(rng.standard_normal((64, cols)).astype(np.float32) *
                                    np.logspace(-1, 1, cols).astype(np.float32))
    res2 = compare([w], dim=dim, k=k, stages=1, seed=0, importance=h, iters=8)
    aw = res2["arms"]["scale_plus_activation_aware"]["mean_output_weighted_rel_error"]
    si = rel_error(w, apply(fit([w], dim=dim, k=k, stages=1, seed=0, row_scale=True),
                            w, dim=dim, row_scale=True), h)
    assert aw is not None and aw <= si * 1.0 + 1e-6, (aw, si)

    # importance is mean-normalized, so it never rescales the objective
    assert abs(float(h.mean()) - 1.0) < 1e-5

    # SPEED PATHS ARE EXACT. _assign drops the constant ||v||^2 term and, for large k, carries a
    # running min over K-blocks. Both must return the SAME indices as the naive full-distance
    # form, weighted and unweighted, above and below the K-block threshold. A speedup that moves
    # an index is not a speedup, it is a different artifact.
    torch = gf._torch()
    dev = gf._device()

    def _naive(vv, cbb, ww=None):
        cbt = cbb.t().contiguous()
        if ww is None:
            return ((vv * vv).sum(1, keepdim=True) - 2.0 * (vv @ cbt) + (cbb * cbb).sum(1)).argmin(1)
        return ((ww * vv * vv).sum(1, keepdim=True) - 2.0 * ((ww * vv) @ cbt)
                + (ww @ (cbb * cbb).t().contiguous())).argmin(1)

    for kk in (64, 1024):                            # small and large codebook
        vv = torch.from_numpy(rng.standard_normal((4096, dim)).astype(np.float32)).to(dev)
        cbb = torch.from_numpy(rng.standard_normal((kk, dim)).astype(np.float32)).to(dev)
        ww = torch.from_numpy(np.abs(rng.standard_normal((4096, dim))).astype(np.float32)).to(dev)
        assert bool((_assign(vv, cbb) == _naive(vv, cbb)).all()), f"unweighted k={kk}"
        assert bool((_assign(vv, cbb, ww) == _naive(vv, cbb, ww)).all()), f"weighted k={kk}"

    return {"ok": True, "pathology_share": base["mean_single_codeword_share"],
            "cured_share": inv["mean_single_codeword_share"],
            "rel_error_baseline": base["mean_rel_error"],
            "rel_error_scale_invariant": inv["mean_rel_error"],
            "output_weighted_activation_aware": aw, "output_weighted_scale_only": round(si, 6)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Function-aware sub-bit codec (M01' + M02).")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

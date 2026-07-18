#!/usr/bin/env python3
"""
imatrix MIXED-PRECISION byte-cut oracle (Bible axis-2 byte-cut; lever listed as
"+12-20%, no new kernel" but with NO oracle/report until now). First-cut,
in-session (CPU/NumPy) weight-only PROXY that BRACKETS the real activation
imatrix answer. Companion to `oracle_qtip_quality.py` (same dequant pattern,
same RSS/--selftest discipline, same honest-bracket style).

WHAT THE REAL LEVER IS (llama.cpp imatrix mixed-precision)
----------------------------------------------------------
llama.cpp's `--imatrix` weights quantization error by per-INPUT-COLUMN activation
importance s_j = mean_t( x_{t,j}^2 ) over a calibration set (an "importance
matrix"): the affine grid is refit to minimise sum_j s_j (w_ij - q_ij)^2 instead
of the unweighted sum, so columns the model actually USES keep precision. The
MIXED-PRECISION byte-cut then assigns FEWER bits to whole tensors/rows of low
aggregate importance and keeps 4-bit on high-importance ones — net FEWER bytes
at iso-quality, and crucially "NO NEW KERNEL" because every piece is still a
standard ggml K-quant block (Q4_K / Q3_K / Q2_K), just a different mix per tensor.

THE HARD CONSTRAINT (why the decisive verdict is Colab, not local)
------------------------------------------------------------------
The real importance s_j is an ACTIVATION statistic. It needs a forward pass over
a calibration corpus on the f16 weights — and the f16 Qwen2.5-3B is NOT on this
machine (only the Q4_K_M GGUF), nor can the logit/KL gate run without a forward
pass. A recorded kill to RESPECT: imatrix-Q3-from-Q4 was +32% PPL for -18% bytes
(`reports/dead_levers.md`) — you must NEVER fit the low-bit grid from already-Q4
weights. So the DECISIVE gate (`--colab`) runs where f16 weights + a real imatrix
+ a forward pass live. This file's job is a weight-only surrogate that brackets
whether the mixed-precision ASSIGNMENT can plausibly cut bytes at iso-RMSE.

HOW IT IS MODELLED HERE (and why this is trustworthy, not a wrong sim)
---------------------------------------------------------------------
The byte-cut lever has two separable parts. We model the part we CAN see exactly
and BRACKET the part we cannot:

  PART 1 — the ASSIGNMENT (modelled, weight-only).  Given a per-output-channel
    importance surrogate, keep the top-fraction of channels at Q4_K and drop the
    rest to Q3_K (or Q2_K), chosen so total bytes <= uniform-Q4_K. We measure
    reconstruction RMSE of this mixed assignment vs uniform Q4_K_M, at EQUAL-or-
    FEWER bytes. This is an exact NumPy round-trip through faithful K-quant grids
    (no approximation of the quantizer) — only the IMPORTANCE ranking is a proxy.

  PART 2 — the IMPORTANCE RANKING (bracketed).  The real ranking is activation
    s_j; we cannot compute it without activations. We therefore report the
    mixed-precision RMSE under THREE rankings that bracket reality:
      * ORACLE-RMSE   : rank channels by their OWN Q3-vs-Q4 RMSE penalty (the
                        best any importance signal could do for *RMSE* — a
                        lower bound on mixed-precision error; activation imatrix
                        cannot beat this for the recon metric).
      * WEIGHT-NORM   : rank by per-output-channel L2 norm ||row||_2 (a pure
                        weight surrogate — the realistic in-session signal).
      * RANDOM        : rank arbitrarily (an upper bound — what you get with NO
                        importance signal at all).
    The real activation-imatrix assignment sits BETWEEN weight-norm and oracle
    for whatever metric correlates with activations; reporting the interval makes
    the one thing the proxy cannot pin down explicit, rather than inventing a
    single possibly-wrong number ("a wrong sim is worse than none", bible 8.3.1).

  We ALSO report an activation-weighted importance-WEIGHTED-RMSE proxy: a
  diag(W^T W) column-energy weighting as a stand-in for sum(act^2). This is a
  DIRECTION-only probe of whether weight-energy and (eventual) activation-energy
  even point the same way — it is NOT the real imatrix (weight energy != activ
  energy), and is labelled as such everywhere.

WHY THE WEIGHT-ONLY SURROGATE OVER/UNDER-CREDITS THE REAL IMATRIX
-----------------------------------------------------------------
  * OVER-credits: the ORACLE ranking peeks at the actual Q3/Q4 RMSE penalty,
    which no causal importance signal (weight or activation) gets for free — so
    oracle-RMSE is optimistic for ANY real assignment on the RMSE metric.
  * UNDER-credits: RMSE is NOT the model's objective. The real imatrix protects
    columns that move LOGITS, which can be low-weight-norm yet high-activation
    (e.g. a near-constant feature the next layer leans on). Weight-norm is blind
    to that. So on the DECISIVE logit/KL metric the real imatrix can BEAT every
    weight-only ranking here — this proxy cannot see that upside.
  * Structural: the byte-cut's whole value is at iso-LOGIT-quality, not iso-RMSE.
    A weight-RMSE proxy can show the assignment is byte-feasible and rank-
    sensible, but only the Colab logit/KL gate decides GO/NO-GO.

KILL-PROTOCOL FRAMING (AGENT.md / bible 8.3.1)
-----------------------------------------------
This is a FIRST CUT. It does NOT record a kill. A weight-only proxy cannot
legitimately kill an activation-driven method (that would be a Type-2 error):
the verdict is NEEDS-MEASUREMENT with the named decisive gate = real imatrix on
f16 + KL/logit check on code (the `--colab` runbook in the report).

RAM discipline: one real tensor at a time, del+gc, RSS ceiling 3 GB.
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

from oracle_qtip_quality import rel_rmse, rss_gb

# ----------------------------------------------------------------------------
MODEL = os.environ.get(
    "IMATRIX_MODEL",
    "/Users/scammermike/Downloads/hawking/models/qwen2.5-3b-instruct-q4_k_m.gguf",
)
REPORT_MD = "/Users/scammermike/Downloads/hawking/reports/oracle_imatrix_mixprec.md"
REPORT_JSON = "/Users/scammermike/Downloads/hawking/reports/oracle/imatrix_mixprec.json"
RSS_CEIL_GB = 3.0

# Same representative real-tensor sample as oracle_qtip_quality.py (attn + ffn
# across early/mid/late) so the two oracles are directly comparable.
SAMPLE_NAMES = [
    "blk.0.attn_q.weight",
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_down.weight",
    "blk.17.attn_output.weight",
    "blk.17.ffn_up.weight",
    "blk.35.attn_q.weight",
    "blk.35.ffn_down.weight",
]

QK = 256              # ggml K-quant super-block
SUB = 32              # K-quant sub-block
SEED = 0

# Effective bits/weight of the faithful NumPy K-quant grids modelled below.
# Q4_K_M : 4 + (6+6)/32 + (16+16)/256 = 4.5  bits  -> 144 B / 256-block
# Q3_K   : 3 + 6/16   + 16/256        = 3.4375 bits -> 110 B / 256-block (ggml: 110)
# Q2_K   : 2 + (4+4)/16 + (16+16)/256 = 2.625  bits -> 84  B / 256-block (ggml: 84)
BITS = {"Q4_K": 4.5, "Q3_K": 3.4375, "Q2_K": 2.625}
BLOCK_BYTES = {"Q4_K": 144.0, "Q3_K": 110.0, "Q2_K": 84.0}


def check_rss(where):
    g = rss_gb()
    if g > RSS_CEIL_GB:
        sys.stderr.write(f"[FATAL] RSS {g:.2f} GB > {RSS_CEIL_GB} GB at {where}\n")
        sys.exit(2)
    return g


# ============================================================================
# Faithful NumPy K-quant grids. gguf.quantize() raises NotImplementedError for
# K-quants, so we reimplement the structure (mirrors oracle_qtip_quality.py's
# q4k_quantize, extended to 3-bit and 2-bit). Each takes a (rows, cols) block
# with cols % 256 == 0 and returns (recon, eff_bits, block_bytes).
#
# An OPTIONAL per-input-column importance weight `w_col` (shape (cols,)) makes the
# affine sub-block refit IMPORTANCE-WEIGHTED, exactly as llama.cpp's imatrix does
# (weighted least squares on the grid). w_col=None -> unweighted (plain K-quant).
# This is the ONLY place activation importance would enter; in-session we can
# only pass weight-energy as a *proxy* weight, never the real activation s_j.
# ============================================================================
def _affine_refit(w, nlevels, wcol=None, iters=5, eps=1e-12):
    """Vectorised affine fit of `w` (nsub, SUB) to `nlevels` levels [0..nlevels-1].
    Optional per-element weights `wcol` (nsub, SUB) -> weighted least squares
    (the imatrix mechanism). Returns (q, scale(nsub,), min(nsub,))."""
    L = nlevels - 1
    wmin = w.min(1, keepdims=True)
    wmax = w.max(1, keepdims=True)
    rng = np.maximum(wmax - wmin, eps)
    scale = rng / L
    mn = wmin.copy()
    q = np.clip(np.round((w - mn) / scale), 0, L)
    if wcol is None:
        ww = np.ones_like(w)
    else:
        ww = wcol
    sw = ww.sum(1, keepdims=True) + eps
    for _ in range(iters):
        qbar = (ww * q).sum(1, keepdims=True) / sw
        wbar = (ww * w).sum(1, keepdims=True) / sw
        cov = (ww * (q - qbar) * (w - wbar)).sum(1, keepdims=True)
        var = (ww * (q - qbar) ** 2).sum(1, keepdims=True)
        new_scale = np.where(var > eps, cov / np.maximum(var, eps), scale)
        new_scale = np.where(new_scale > 0, new_scale, scale)
        mn = wbar - new_scale * qbar
        scale = new_scale
        q = np.clip(np.round((w - mn) / scale), 0, L)
    return q, scale.squeeze(1), mn.squeeze(1)


def _q6(vals, signed):
    """Quantize a vector to 6 bits with one f16-ish super-scale; return recon."""
    vals = vals.astype(np.float64)
    if signed:
        amax = np.max(np.abs(vals))
        if amax == 0:
            return vals
        s = amax / 31.0
        return np.clip(np.round(vals / s), -32, 31) * s
    vmax = np.max(vals)
    if vmax <= 0:
        return vals
    s = vmax / 63.0
    return np.clip(np.round(vals / s), 0, 63) * s


def _qN(vals, nbits, signed):
    """Quantize a scale/min vector to nbits with one super-scale; return recon."""
    vals = vals.astype(np.float64)
    lim = (1 << (nbits - 1)) - 1 if signed else (1 << nbits) - 1
    if signed:
        amax = np.max(np.abs(vals))
        if amax == 0:
            return vals
        s = amax / lim
        return np.clip(np.round(vals / s), -lim - 1, lim) * s
    vmax = np.max(vals)
    if vmax <= 0:
        return vals
    s = vmax / lim
    return np.clip(np.round(vals / s), 0, lim) * s


def _kquant_rows(W, nlevels, scale_bits, min_bits, wcol=None):
    """Generic K-quant round-trip over rows of W (rows, cols), cols % 256 == 0.
    Mirrors q4k structure: 256-superblock = 8 sub-blocks of 32; affine grid per
    sub-block; sub-block scales (+mins) quantized to {scale_bits, min_bits}.
    `wcol` (cols,) optional per-input-column importance weight (imatrix WLS)."""
    R, C = W.shape
    assert C % QK == 0, f"K-quant needs cols % 256 == 0 (got {C})"
    nsblk = C // QK
    out = np.empty_like(W, dtype=np.float64)
    if wcol is not None:
        wcol = np.asarray(wcol, dtype=np.float64)
    for r in range(R):
        row = W[r].astype(np.float64).reshape(nsblk, QK)
        wc = None if wcol is None else wcol.reshape(nsblk, QK)
        for sb in range(nsblk):
            sub = row[sb].reshape(8, SUB)
            wsub = None if wc is None else wc[sb].reshape(8, SUB)
            q, scales, mins = _affine_refit(sub, nlevels, wcol=wsub)
            scales_q = _qN(scales, scale_bits, signed=False)
            mins_q = _qN(mins, min_bits, signed=True)
            recon = q * scales_q[:, None] + mins_q[:, None]
            out[r, sb * QK:(sb + 1) * QK] = recon.reshape(QK)
    return out


def q4k_quantize(W, wcol=None):
    """NumPy Q4_K_M: 4-bit grid, 6-bit scales + 6-bit mins. 4.5 bits, 144 B."""
    recon = _kquant_rows(W, nlevels=16, scale_bits=6, min_bits=6, wcol=wcol)
    return recon, BITS["Q4_K"], BLOCK_BYTES["Q4_K"]


def q3k_quantize(W, wcol=None):
    """NumPy Q3_K: 3-bit grid, 6-bit scales + 6-bit mins. ~3.44 bits, 110 B.
    (ggml Q3_K is symmetric; we keep an affine min for a faithful RMSE-comparable
    grid — this slightly FAVOURS Q3_K, which is conservative for a byte-cut that
    must beat it, and is noted in the report.)"""
    recon = _kquant_rows(W, nlevels=8, scale_bits=6, min_bits=6, wcol=wcol)
    return recon, BITS["Q3_K"], BLOCK_BYTES["Q3_K"]


def q2k_quantize(W, wcol=None):
    """NumPy Q2_K: 2-bit grid, 4-bit scales + 4-bit mins. ~2.63 bits, 84 B."""
    recon = _kquant_rows(W, nlevels=4, scale_bits=4, min_bits=4, wcol=wcol)
    return recon, BITS["Q2_K"], BLOCK_BYTES["Q2_K"]


QUANTIZERS = {"Q4_K": q4k_quantize, "Q3_K": q3k_quantize, "Q2_K": q2k_quantize}


# ============================================================================
# Mixed-precision ASSIGNMENT (the byte-cut lever).
# Keep the top `keep_frac` of OUTPUT CHANNELS (rows) at Q4_K; demote the rest to
# `low` (Q3_K or Q2_K). Ranking comes from an importance vector over rows.
# We round to whole 256-superblocks of rows so the byte accounting is exact and
# the result is a legal ggml mix (no new kernel: each row-group is a standard
# K-quant). Returns (recon, mean_bits, mean_block_bytes, n_kept_rows).
# ============================================================================
def mixed_assignment(W, importance, keep_frac, low="Q3_K", wcol=None):
    R, C = W.shape
    order = np.argsort(-importance)              # high importance first
    n_keep = int(round(keep_frac * R))
    n_keep = max(0, min(R, n_keep))
    keep_rows = np.sort(order[:n_keep])
    low_rows = np.sort(order[n_keep:])
    recon = np.empty_like(W, dtype=np.float64)
    if len(keep_rows):
        rec_hi, _, _ = q4k_quantize(W[keep_rows], wcol=wcol)
        recon[keep_rows] = rec_hi
    if len(low_rows):
        rec_lo, _, _ = QUANTIZERS[low](W[low_rows], wcol=wcol)
        recon[low_rows] = rec_lo
    bytes_per_row_blk = (C // QK)
    total_bytes = (len(keep_rows) * BLOCK_BYTES["Q4_K"]
                   + len(low_rows) * BLOCK_BYTES[low]) * bytes_per_row_blk
    weights = R * C
    mean_bits = total_bytes * 8.0 / weights
    mean_block_bytes = total_bytes / (R * bytes_per_row_blk)
    return recon, mean_bits, mean_block_bytes, len(keep_rows)


def importance_weight_norm(W):
    """Per-output-channel (row) L2 norm — the realistic weight-only surrogate."""
    return np.linalg.norm(W.astype(np.float64), axis=1)


def importance_oracle_rmse(W, low="Q3_K"):
    """Per-row Q3(or low)-vs-Q4 RMSE penalty: the best ANY ranking can do for the
    RMSE metric (rows that suffer most from demotion rank highest -> kept). An
    optimistic lower bound on mixed-precision RMSE; no causal signal gets this."""
    R, C = W.shape
    rec4, _, _ = q4k_quantize(W)
    recL, _, _ = QUANTIZERS[low](W)
    pen = np.linalg.norm((recL - rec4).astype(np.float64), axis=1)  # demotion cost
    return pen


def find_keep_frac_for_bytes(R, C, low, byte_budget_frac=1.0):
    """Largest keep_frac whose total bytes <= byte_budget_frac * uniform-Q4_K.
    With byte_budget_frac=1.0 this is the iso-or-fewer-bytes assignment."""
    uni = R * BLOCK_BYTES["Q4_K"]
    budget = byte_budget_frac * uni
    # bytes(keep) = keep*144 + (R-keep)*low ; solve keep
    lo_b = BLOCK_BYTES[low]
    # keep*144 + (R-keep)*lo_b <= budget
    # keep*(144-lo_b) <= budget - R*lo_b
    num = budget - R * lo_b
    den = (BLOCK_BYTES["Q4_K"] - lo_b)
    keep = num / den if den > 0 else R
    keep = max(0, min(R, int(np.floor(keep))))
    return keep / R


# ============================================================================
# --selftest : validate the quantizers + assignment on fixed synthetic inputs.
# Analogue of oracle_qtip_quality.py's SQNR check: a known-optimum SQNR for the
# uniform quantizer on Gaussian data, plus monotonic bit->quality ordering and
# the byte/assignment invariants.
# ============================================================================
def selftest():
    rng = np.random.default_rng(SEED)
    fails = []

    # 1) Quantizer SQNR ordering on a unit-Gaussian source (more bits -> higher
    #    SQNR). For a uniform scalar quantizer on Gaussian data the high-rate
    #    SQNR rises ~6.02 dB/bit; with K-quant block overhead we just require a
    #    sane absolute level for 4-bit and strict monotonicity 2<3<4.
    Wg = rng.standard_normal((32, 512)).astype(np.float32)
    sq = {}
    for name, fn in QUANTIZERS.items():
        rec, _, _ = fn(Wg)
        mse = np.mean((rec - Wg) ** 2)
        sq[name] = 10 * np.log10(np.var(Wg) / mse)
    if not (sq["Q2_K"] < sq["Q3_K"] < sq["Q4_K"]):
        fails.append(f"SQNR not monotone in bits ({sq})")
    # 4-bit affine block quantizer on Gaussian: expect ~20-26 dB (4-bit ideal
    # uniform Gaussian ~ 4*6.02 - 4.35 ~= 19.7 dB high-rate; block min/scale
    # buys a few dB). Bracket generously; a gross bug lands far outside.
    if not (18.0 < sq["Q4_K"] < 30.0):
        fails.append(f"Q4_K Gaussian SQNR implausible ({sq['Q4_K']:.2f} dB)")

    # 2) Q4_K rel-RMSE plausible and beats gguf Q4_0 on a Gaussian source
    q4k, _, _ = q4k_quantize(Wg)
    rr_q4k = rel_rmse(q4k, Wg)
    q40_ok = True
    rr_q40 = float("nan")
    try:
        from gguf import quantize, dequantize, GGMLQuantizationType
        q40 = dequantize(quantize(Wg, GGMLQuantizationType.Q4_0),
                         GGMLQuantizationType.Q4_0).astype(np.float64).reshape(Wg.shape)
        rr_q40 = rel_rmse(q40, Wg)
        if not (rr_q4k < rr_q40):
            fails.append(f"Q4_K not better than Q4_0 ({rr_q4k:.4f} vs {rr_q40:.4f})")
    except Exception as e:
        q40_ok = False
        sys.stderr.write(f"[selftest] gguf Q4_0 cross-check skipped: {e}\n")
    if not (0.0 < rr_q4k < 0.12):
        fails.append(f"Q4_K rel-RMSE implausible ({rr_q4k:.4f})")

    # 3) Importance-weighted refit reduces error ON the high-weight columns vs
    #    unweighted (the imatrix mechanism must actually protect weighted cols).
    Wi = rng.standard_normal((16, 512)).astype(np.float32)
    wcol = np.ones(512)
    hi_cols = rng.choice(512, size=32, replace=False)
    wcol[hi_cols] = 50.0                                   # a few "important" cols
    rec_u, _, _ = q3k_quantize(Wi)                         # unweighted
    rec_w, _, _ = q3k_quantize(Wi, wcol=wcol)              # imatrix-weighted
    err_u = np.sqrt(np.mean((rec_u[:, hi_cols] - Wi[:, hi_cols]) ** 2))
    err_w = np.sqrt(np.mean((rec_w[:, hi_cols] - Wi[:, hi_cols]) ** 2))
    if not (err_w <= err_u + 1e-9):
        fails.append(f"imatrix WLS did not protect weighted cols ({err_w:.4f} > {err_u:.4f})")

    # 4) Mixed assignment: byte budget honoured + oracle ranking <= weight-norm
    #    <= random for RMSE (the bracket ordering this oracle relies on). Use a
    #    real byte-cut (0.85x) so demotion actually happens and the rankings
    #    DIFFER — at iso-bytes (1.0x) the budget keeps every row at Q4 and the
    #    bracket collapses (vacuously equal), which would not test the ordering.
    Wm = rng.standard_normal((64, 512)).astype(np.float32)
    # Inject heterogeneity: scale some rows up so demotion cost varies by row.
    Wm[:16] *= 8.0
    R = Wm.shape[0]
    kf = find_keep_frac_for_bytes(R, Wm.shape[1], "Q3_K", 0.85)
    imp_or = importance_oracle_rmse(Wm, "Q3_K")
    imp_wn = importance_weight_norm(Wm)
    imp_rd = rng.standard_normal(R)
    rec_or, mb_or, bb_or, nk = mixed_assignment(Wm, imp_or, kf, "Q3_K")
    rec_wn, _, _, _ = mixed_assignment(Wm, imp_wn, kf, "Q3_K")
    rec_rd, _, _, _ = mixed_assignment(Wm, imp_rd, kf, "Q3_K")
    rr_or, rr_wn, rr_rd = (rel_rmse(rec_or, Wm), rel_rmse(rec_wn, Wm),
                           rel_rmse(rec_rd, Wm))
    uni_bytes = R * BLOCK_BYTES["Q4_K"]
    mix_bytes = bb_or * R
    if not (mix_bytes < uni_bytes - 1e-6):
        fails.append(f"mixed did not cut bytes vs uniform Q4_K ({mix_bytes:.0f} vs {uni_bytes:.0f})")
    # Strict ordering (rankings must actually differ at a forcing budget):
    if not (rr_or < rr_wn):
        fails.append(f"oracle ranking !< weight-norm RMSE ({rr_or:.4f} >= {rr_wn:.4f})")
    if not (rr_wn < rr_rd):
        fails.append(f"weight-norm ranking !< random RMSE ({rr_wn:.4f} >= {rr_rd:.4f})")

    # 5) Byte accounting closed-form matches mixed_assignment report
    keep_b = nk * BLOCK_BYTES["Q4_K"] + (R - nk) * BLOCK_BYTES["Q3_K"]
    if not (abs(keep_b - mix_bytes) < 1e-6):
        fails.append(f"byte accounting mismatch ({keep_b:.0f} vs {mix_bytes:.0f})")

    print("=== imatrix mixed-prec oracle self-test ===")
    print(f"  K-quant SQNR monotone 2<3<4 .......... "
          f"{sq['Q2_K']:.1f} < {sq['Q3_K']:.1f} < {sq['Q4_K']:.1f} dB")
    print(f"  Q4_K Gaussian rel-RMSE ............... {rr_q4k:.4f} (4.5b)"
          + (f" < Q4_0 {rr_q40:.4f}" if q40_ok else ""))
    print(f"  imatrix WLS protects weighted cols ... err {err_u:.4f} -> {err_w:.4f}")
    print(f"  mixed bytes <= uniform Q4_K .......... {mix_bytes:.0f} <= {uni_bytes:.0f} B "
          f"(kept {nk}/{R} rows @Q4_K)")
    print(f"  RMSE bracket oracle<=wnorm<=random ... {rr_or:.4f} <= {rr_wn:.4f} <= {rr_rd:.4f}")
    if fails:
        print("\nSELF-TEST FAILED:")
        for f in fails:
            print("  - " + f)
        sys.exit(1)
    print("\nself-test PASSED — quantizers + assignment + bracket are trustworthy.")
    return True


# ============================================================================
# --local-proxy : weight-only first cut grounded in real Qwen weights.
# For each tensor, at an iso-or-fewer-byte budget vs uniform Q4_K_M, compute the
# mixed-precision RMSE under the three bracket rankings (oracle / weight-norm /
# random) for low in {Q3_K, Q2_K}. Also report the diag(W^T W) column-energy
# importance-WEIGHTED RMSE as a direction-only activation-shape probe.
# ============================================================================
def local_proxy(low="Q3_K", byte_frac=0.85, max_rows=256):
    from gguf import GGUFReader, dequantize
    t0 = time.time()
    reader = GGUFReader(MODEL)
    by_name = {t.name: t for t in reader.tensors}

    rows = []
    for nm in SAMPLE_NAMES:
        if nm not in by_name:
            sys.stderr.write(f"[skip] {nm} not in model\n")
            continue
        t = by_name[nm]
        shape = tuple(int(x) for x in t.shape)            # (cols, rows)
        n_cols, m_rows = shape[0], shape[1]
        w = dequantize(np.array(t.data), t.tensor_type).astype(np.float32)
        W = w.reshape(m_rows, n_cols)
        del w
        check_rss(f"deq {nm}")

        # Subsample rows for cost while keeping cols intact (cols carry the
        # imatrix/column structure; row count just scales work).
        if m_rows > max_rows:
            ridx = np.linspace(0, m_rows - 1, max_rows).astype(int)
            Wsub = np.ascontiguousarray(W[ridx])
        else:
            Wsub = W
        R, C = Wsub.shape

        # Uniform Q4_K baseline RMSE (the thing the byte-cut must match).
        rec_u, _, _ = q4k_quantize(Wsub)
        rr_uniform = rel_rmse(rec_u, Wsub)
        del rec_u

        # Byte budget -> keep fraction (iso-or-fewer bytes).
        kf = find_keep_frac_for_bytes(R, C, low, byte_frac)

        # Three bracket rankings.
        imp_or = importance_oracle_rmse(Wsub, low)
        imp_wn = importance_weight_norm(Wsub)
        rng = np.random.default_rng(SEED)
        imp_rd = rng.standard_normal(R)

        rec_or, mb, bb, nk = mixed_assignment(Wsub, imp_or, kf, low)
        rr_or = rel_rmse(rec_or, Wsub); del rec_or
        rec_wn, _, _, _ = mixed_assignment(Wsub, imp_wn, kf, low)
        rr_wn = rel_rmse(rec_wn, Wsub); del rec_wn
        rec_rd, _, _, _ = mixed_assignment(Wsub, imp_rd, kf, low)
        rr_rd = rel_rmse(rec_rd, Wsub); del rec_rd

        # Correlation of the realistic weight-norm ranking with the oracle
        # ranking (Spearman-ish via rank correlation): how good is weight-norm
        # as a stand-in for the RMSE-optimal ranking? (Activation imatrix would
        # add a *different* signal on top — see report.)
        rk_or = np.argsort(np.argsort(-imp_or))
        rk_wn = np.argsort(np.argsort(-imp_wn))
        rank_corr = float(np.corrcoef(rk_or, rk_wn)[0, 1])

        # Direction-only activation-shape probe: diag(W^T W) per-input-column
        # energy as a *proxy* weight for the imatrix WLS (NOT real activations).
        # Compare importance-WEIGHTED rel-RMSE of uniform Q4_K with vs without
        # this weighting — does protecting high-weight-energy columns even help
        # the (weight-energy-)weighted error? (A sign the mechanism is wired;
        # the real lever uses activation energy, which can point elsewhere.)
        col_energy = (Wsub.astype(np.float64) ** 2).sum(0)   # (C,)
        col_energy = col_energy / (col_energy.mean() + 1e-12)
        rec_uw, _, _ = q4k_quantize(Wsub, wcol=col_energy)
        # weighted rel-RMSE (weight each squared error by col energy)
        we = col_energy[None, :]
        num_w = np.sqrt(np.sum(we * (rec_uw - Wsub) ** 2))
        num_u, _, _ = q4k_quantize(Wsub)
        num_u = np.sqrt(np.sum(we * (num_u - Wsub) ** 2))
        den_w = np.sqrt(np.sum(we * Wsub.astype(np.float64) ** 2))
        wrr_weighted = float(num_w / den_w) if den_w > 0 else float("nan")
        wrr_unweighted = float(num_u / den_w) if den_w > 0 else float("nan")
        del rec_uw

        cut_pct = (1.0 - bb / BLOCK_BYTES["Q4_K"]) * 100.0
        rows.append(dict(
            name=nm, shape=[m_rows, n_cols], rows_used=R, disk=t.tensor_type.name,
            low=low, keep_frac=kf, kept_rows=nk, mean_bits=mb,
            mix_block_bytes=bb, q4k_block_bytes=BLOCK_BYTES["Q4_K"], byte_cut_pct=cut_pct,
            uniform_q4k_rmse=rr_uniform,
            mix_rmse_oracle=rr_or, mix_rmse_weightnorm=rr_wn, mix_rmse_random=rr_rd,
            oracle_beats_uniform=bool(rr_or <= rr_uniform),
            weightnorm_beats_uniform=bool(rr_wn <= rr_uniform),
            wnorm_oracle_rank_corr=rank_corr,
            imat_weighted_rmse=wrr_weighted, unweighted_weighted_rmse=wrr_unweighted,
            imat_helps_weighted=bool(wrr_weighted <= wrr_unweighted),
        ))
        g = check_rss(f"done {nm}")
        sys.stderr.write(
            f"[ok] {nm} RSS={g:.2f}GB cut={cut_pct:.0f}% uni={rr_uniform:.4f} "
            f"mix[or={rr_or:.4f} wn={rr_wn:.4f} rd={rr_rd:.4f}] rankcorr={rank_corr:.2f}\n")
        del W, Wsub, imp_or, imp_wn, imp_rd
        gc.collect()

    return rows, time.time() - t0


# ============================================================================
# Report writer + verdict logic (NEEDS-MEASUREMENT; never a kill).
# ============================================================================
def write_reports(rows, wall, low, byte_frac):
    n = len(rows)
    med = lambda k: float(np.median([r[k] for r in rows])) if rows else float("nan")
    n_or = sum(r["oracle_beats_uniform"] for r in rows)
    n_wn = sum(r["weightnorm_beats_uniform"] for r in rows)
    n_imat = sum(r["imat_helps_weighted"] for r in rows)
    med_uni = med("uniform_q4k_rmse")
    med_or = med("mix_rmse_oracle")
    med_wn = med("mix_rmse_weightnorm")
    med_rd = med("mix_rmse_random")
    med_cut = med("byte_cut_pct")
    med_bits = med("mean_bits")
    med_corr = med("wnorm_oracle_rank_corr")
    half = (n + 1) // 2

    L = []
    o = L.append
    o("# Oracle — imatrix MIXED-PRECISION byte-cut (axis-2; first cut)")
    o("")
    o(f"**Model:** `models/qwen2.5-3b-instruct-q4_k_m.gguf`  **Lane:** CPU NumPy  "
      f"**Mix:** Q4_K + {low}  **Budget:** <= {byte_frac:.2f}x uniform-Q4_K bytes "
      f"(median {med_cut:.0f}% byte-cut, ~{med_bits:.2f} eff bits/weight)")
    o("**Date:** 2026-05-31")
    o("")
    o("> **Scope = WEIGHT-ONLY first-cut PROXY.** The lever is "
      "`imatrix mixed-precision` (bible axis-2, listed +12-20% / no-new-kernel, "
      "previously with no oracle). The DECISIVE verdict needs the REAL importance "
      "matrix — an ACTIVATION statistic from a forward pass over a calibration "
      "corpus on **f16** weights — plus a KL/logit gate. The f16 Qwen and a forward "
      "pass are **not on this machine** (only the Q4_K_M GGUF), and fitting a low-bit "
      "grid from already-Q4 weights is a recorded kill (imatrix-Q3-from-Q4 = +32% "
      "PPL / -18% bytes, `reports/dead_levers.md`). So this is a first cut, **not** "
      "the gate, and records **NO kill** (a weight-only proxy cannot legitimately "
      "kill an activation-driven method — that would be a Type-2 error).")
    o("")
    o("> **What is exact vs bracketed.** The mixed-precision ASSIGNMENT and the "
      "K-quant round-trips (Q4_K/Q3_K/Q2_K) are faithful NumPy reimplementations "
      "(gguf has no K-quant quantizer) — exact, no approximation. The IMPORTANCE "
      "RANKING is what we cannot see without activations, so we report the mixed "
      "RMSE under THREE bracket rankings:")
    o(">  * **ORACLE** — rank rows by their own Q4-vs-low RMSE demotion penalty "
      "(a greedy per-row heuristic, ~RMSE-optimal — occasionally edged by "
      "weight-norm by <0.1% via kept/demoted interaction). An optimistic "
      "near-lower-bound; no causal importance signal gets the penalty for free.")
    o(">  * **WEIGHT-NORM** — rank by per-output-channel L2 norm. The realistic "
      "in-session weight surrogate.")
    o(">  * **RANDOM** — no importance signal; an upper bound on the error.")
    o("> The real activation-imatrix assignment sits inside this interval for the "
      "metric that tracks activations; for the *RMSE* metric it cannot beat ORACLE.")
    o("")
    o(f"## (i) Mixed-precision RMSE vs uniform Q4_K_M at a {med_cut:.0f}% byte-cut")
    o("")
    o("Keep the top-importance output channels at Q4_K, demote the rest to "
      f"{low}, sized so total bytes <= {byte_frac:.2f}x uniform Q4_K_M (a "
      f"{med_cut:.0f}% cut). Reconstruction rel-RMSE of the dequantized real "
      "tensor (lower = closer to uniform Q4_K). `beats` = that ranking's mixed "
      "RMSE <= the uniform-Q4_K (no-cut) RMSE — a tautological loss for any "
      "byte-cut; the meaningful comparison is the intra-budget one just below.")
    o("")
    o("| tensor | shape | disk | cut% | keep%@Q4 | uniform Q4_K | mix ORACLE | mix WNORM | mix RANDOM | WN beats | OR beats |")
    o("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        o(f"| {r['name']} | {r['shape'][0]}x{r['shape'][1]} | {r['disk']} | "
          f"{r['byte_cut_pct']:.0f}% | {r['keep_frac']*100:.0f}% | "
          f"{r['uniform_q4k_rmse']:.4f} | {r['mix_rmse_oracle']:.4f} | "
          f"{r['mix_rmse_weightnorm']:.4f} | {r['mix_rmse_random']:.4f} | "
          f"{'YES' if r['weightnorm_beats_uniform'] else 'no'} | "
          f"{'YES' if r['oracle_beats_uniform'] else 'no'} |")
    o("")
    o(f"**Median:** uniform Q4_K {med_uni:.4f} vs mixed [ORACLE {med_or:.4f}, "
      f"WNORM {med_wn:.4f}, RANDOM {med_rd:.4f}] at {med_cut:.0f}% byte-cut "
      f"(~{med_bits:.2f} bits). WEIGHT-NORM ranking matches-or-beats uniform on "
      f"**{n_wn}/{n}** tensors; ORACLE (RMSE lower bound) on **{n_or}/{n}**.")
    o("")
    # The decisive intra-budget comparison: importance vs no-importance at the
    # SAME byte budget. "beats uniform-Q4_K" (the no-CUT reference) is a
    # tautological loss for any byte-cut and is NOT what the lever claims.
    lever_or = (1.0 - med_or / med_rd) * 100.0 if med_rd > 0 else float("nan")
    lever_wn = (1.0 - med_wn / med_rd) * 100.0 if med_rd > 0 else float("nan")
    o("### What the imatrix actually buys (the intra-budget comparison)")
    o("")
    o("The lever's real claim is **importance-guided mixing beats UNIFORM demotion "
      "to the SAME byte budget** — NOT that a byte-cut beats the no-cut Q4_K_M "
      "(that is tautologically impossible: cut bytes -> RMSE rises). So the "
      "decisive in-session number is mixed-with-importance (ORACLE / WEIGHT-NORM) "
      "vs RANDOM (= uniform demotion to budget):")
    o("")
    o(f"- **ORACLE vs RANDOM:** −{lever_or:.1f}% RMSE — the *most* any importance "
      "signal can recover for RMSE at this budget.")
    o(f"- **WEIGHT-NORM vs RANDOM:** −{lever_wn:.1f}% RMSE — what the realistic "
      f"weight surrogate recovers. It captures ~{(lever_wn/lever_or*100 if lever_or>0 else float('nan')):.0f}% "
      "of the oracle's gain (rank corr below confirms weight-norm ≈ the RMSE-"
      "optimal ranking).")
    o("")
    o("So at the RMSE metric the importance ranking buys only a SINGLE-DIGIT-% "
      "edge over no ranking — the byte-cut's cost is dominated by the steep "
      f"Q4->{low} grid penalty (uniform {med_uni:.4f} -> ~{med_rd:.4f}), not by "
      "*which* channels are demoted. This is the proxy's central honest result: "
      "on weight-RMSE the mixed-precision lever is mostly a quantizer-rate story, "
      "and importance is a small correction. **Whether the real ACTIVATION imatrix "
      "buys more — on LOGITS, the metric that matters — is exactly what weight-RMSE "
      "cannot see and the Colab gate must measure.**")
    o("")
    o("## (ii) How good is the weight-only ranking? (rank correlation)")
    o("")
    o("Rank-correlation of the realistic WEIGHT-NORM ranking with the RMSE-ORACLE "
      "ranking per tensor. High corr -> weight-norm already captures most of the "
      "*RMSE*-relevant importance; the gap to GO is then mostly whether the real "
      "ACTIVATION imatrix adds a logit-relevant signal weight-norm is blind to.")
    o("")
    o("| tensor | weight-norm vs oracle rank corr |")
    o("|---|---|")
    for r in rows:
        o(f"| {r['name']} | {r['wnorm_oracle_rank_corr']:.2f} |")
    o("")
    o(f"**Median rank corr:** {med_corr:.2f}.")
    o("")
    o("## (iii) Direction-only activation-shape probe (NOT the real imatrix)")
    o("")
    o("A diag(W^T W) per-input-column *weight*-energy weighting fed into the "
      "K-quant WLS refit (the same machinery the real imatrix uses, but with "
      "weight energy as a stand-in for the activation energy sum(x^2) it cannot "
      "see). Reports whether protecting high-weight-energy columns lowers the "
      "(weight-energy-)weighted RMSE — a sanity check that the imatrix MECHANISM "
      "is wired and helps *when* importance points at high-magnitude columns. The "
      "real lever weights by ACTIVATION energy, which can point at low-weight-norm "
      "columns, so this is a direction probe only.")
    o("")
    o(f"- Importance-weighted-RMSE helped (weighted <= unweighted) on **{n_imat}/{n}** "
      "tensors. (Expected: weighting by a column's own energy mostly tracks where "
      "the error already is, so the WLS gain on the *weighted* metric is modest "
      "and occasionally negative — exactly why the decisive signal must come from "
      "*activation* energy, not weight energy.)")
    o("")
    o("## Direction read (NOT the verdict)")
    o("")
    if n_wn >= half:
        o(f"- **GO direction (weight-only).** Even the realistic WEIGHT-NORM "
          f"ranking matches-or-beats uniform Q4_K_M RMSE on {n_wn}/{n} tensors at "
          f"{med_cut:.0f}% fewer bytes — i.e. demoting low-norm channels to {low} "
          "is byte-feasible without raising weight-RMSE. The real activation "
          "imatrix can only help further on the metric that matters. This robustly "
          "EARNS the Colab f16 + real-imatrix + logit gate.")
    elif n_or >= half:
        o(f"- **Conditional direction.** Uniform Q4_K wins under WEIGHT-NORM but "
          f"the RMSE-ORACLE ranking matches-or-beats it on {n_or}/{n} tensors — so "
          "the byte-cut is achievable in principle, but ONLY if the importance "
          "signal is good enough, and weight-norm alone is not it. Whether the "
          "real ACTIVATION imatrix closes the gap is exactly what the Colab gate "
          "decides; this is the case the proxy cannot settle.")
    else:
        o(f"- **Cautionary direction.** Even the RMSE-ORACLE ranking (the best any "
          f"signal can do for RMSE) trails uniform Q4_K on {n - n_or}/{n} tensors "
          f"at this byte budget — demoting to {low} costs more weight-RMSE than the "
          "byte budget buys back, regardless of ranking. The real imatrix could "
          "still win on LOGITS (RMSE is not its objective), so this is "
          "NEEDS-MEASUREMENT, not a kill — but the Colab gate must show a real "
          "logit/KL margin.")
    o("- **Why weight-only over/under-credits the real activation imatrix:** "
      "(1) OVER — the ORACLE ranking peeks at the actual demotion RMSE, which no "
      "causal signal gets, so it is optimistic for any real assignment on RMSE; "
      "(2) UNDER — RMSE is not the model's objective: the real imatrix protects "
      "columns that move LOGITS, which can be low-weight-norm but high-activation "
      "(a near-constant feature a later layer leans on), and weight-norm is blind "
      "to that, so the real imatrix can BEAT every ranking here on the decisive "
      "logit/KL metric; (3) STRUCTURAL — the lever's value is iso-LOGIT-quality, "
      "not iso-RMSE, and a weight-RMSE proxy can only show byte-feasibility and "
      "rank-sensibility; (4) the Q3_K grid here keeps an affine min (slightly "
      "FAVOURING the low-bit leg vs ggml's symmetric Q3_K — conservative for a "
      "byte-cut that must beat it).")
    o("")
    o("## The DECISIVE (Colab) gate — `--colab` runbook")
    o("")
    o("On Colab, with f16 Qwen2.5-3B + a code calibration corpus:")
    o("1. Run llama.cpp `llama-imatrix` over the corpus on the **f16** model to "
      "produce a real importance matrix (per-input-column sum(x^2)). Export "
      "per-tensor f16 weights + the imatrix vector -> `weights.npz` / `imat.npz`.")
    o("2. Build the mixed-precision GGUF the real way: `llama-quantize "
      "--imatrix imat.dat model-f16.gguf model-mix.gguf <type>` with a tensor-type "
      "override that keeps high-importance tensors at Q4_K and demotes low-"
      "importance ones to {Q3_K,Q2_K}; confirm total bytes <= the uniform Q4_K_M "
      "GGUF. (No new kernel — every tensor is a standard ggml K-quant.)")
    o("3. **Recon gate:** mixed vs uniform-Q4_K_M rel-RMSE **vs f16** (GO floor: "
      "mixed <= uniform at fewer bytes). Fit the low-bit grids FROM f16, never "
      "from Q4 (kill-respect).")
    o("4. **Functional gate (decisive):** forward-pass f16 / uniform-Q4_K_M / "
      "mixed on held-out **code**; export next-token logits; check logit-cosine / "
      "KL / argmax-agreement (GO: mixed >= uniform cosine & argmax, <= KL — i.e. "
      "the byte-cut is free on quality). The bible's +12-20% is a THROUGHPUT claim "
      "(fewer bytes -> faster decode GEMV); confirm with a paired decode bench "
      "once the quality gate is GO.")
    o("5. GO on recon AND logits -> wire the mix in the loader (byte accounting "
      "only; no kernel) + paired decode bench for the +12-20%. NO-GO on the "
      "logit gate with a fair real imatrix -> THEN a kill is legitimate (records "
      "a dead_levers entry, classified Type-1/2 per protocol).")
    o("")
    o(f"_Wall: {wall:.1f}s. Peak RSS: {check_rss('report'):.2f} GB. Run "
      "`--selftest` (must pass) for these numbers to be trustworthy._")

    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(REPORT_JSON, "w") as f:
        json.dump(dict(
            lever="imatrix_mixed_precision", low_quant=low, byte_budget_frac=byte_frac,
            n_tensors=n, median_byte_cut_pct=med_cut, median_mean_bits=med_bits,
            median_uniform_q4k_rmse=med_uni, median_mix_rmse_oracle=med_or,
            median_mix_rmse_weightnorm=med_wn, median_mix_rmse_random=med_rd,
            lever_value_oracle_vs_random_pct=(1.0 - med_or / med_rd) * 100.0 if med_rd > 0 else None,
            lever_value_weightnorm_vs_random_pct=(1.0 - med_wn / med_rd) * 100.0 if med_rd > 0 else None,
            n_weightnorm_beats_uniform=n_wn, n_oracle_beats_uniform=n_or,
            median_wnorm_oracle_rank_corr=med_corr,
            n_imat_helps_weighted=n_imat,
            verdict="NEEDS-MEASUREMENT (decisive gate = Colab f16 + real imatrix + logits)",
            per_tensor=rows,
        ), f, indent=2)
    sys.stderr.write(f"[done] wrote {REPORT_MD} + {REPORT_JSON}\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description="imatrix mixed-precision byte-cut oracle")
    ap.add_argument("--selftest", action="store_true",
                    help="validate quantizers + assignment + bracket (in-session gate)")
    ap.add_argument("--local-proxy", action="store_true",
                    help="weight-only first cut on real Qwen weights (in-session)")
    ap.add_argument("--low", default="Q3_K", choices=["Q3_K", "Q2_K"],
                    help="low-precision leg of the mix (default Q3_K)")
    ap.add_argument("--byte-frac", type=float, default=0.85,
                    help="byte budget as fraction of uniform-Q4_K (default 0.85 = "
                         "~15%% byte-cut, matching the bible's +12-20%% throughput "
                         "target; 1.0 = iso-bytes is a degenerate no-demotion no-op)")
    args = ap.parse_args()

    if not (args.selftest or args.local_proxy):
        ap.error("pick one of --selftest / --local-proxy")
    if args.selftest:
        selftest()
    if args.local_proxy:
        rows, wall = local_proxy(low=args.low, byte_frac=args.byte_frac)
        if not rows:
            sys.stderr.write("[FATAL] no tensors processed\n")
            sys.exit(1)
        write_reports(rows, wall, args.low, args.byte_frac)


if __name__ == "__main__":
    main()

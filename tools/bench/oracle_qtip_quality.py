#!/usr/bin/env python3
"""
QTIP byte-cut QUALITY oracle (Bible §8.1 L1.5 surviving reframe; axis-2 deep
byte-cut). Designed in `plans/qtip_bytecut_design_2026_05_31.md` §5.1.

WHAT THIS IS
------------
The offline quality gate that decides whether QTIP (incoherence-processed
trellis-coded quantization, ~3.0 bits) is USABLE as the deep byte-cut codec for
Qwen2.5-3B decode — i.e. whether a ~3-bit trellis MATCHES-OR-BEATS Q4_K_M's
~4.5-bit grid on the metrics that matter (reconstruction RMSE, and on Colab the
logit-cosine / KL / argmax-agreement family the W4A8 work uses). Oracle-before-
body discipline (AGENT.md): no QTIP kernel is written until this clears.

THE HARD CONSTRAINT (why the decisive verdict is Colab, not local)
------------------------------------------------------------------
QTIP's prize is real ONLY when the codec is fit FROM f16 weights — never
requant-from-Q4_K (a recorded kill to respect: imatrix-Q3-from-Q4 was +32% PPL
for -18% bytes; `reports/dead_levers.md`). The f16 Qwen2.5-3B is NOT on this
machine (only the Q4_K_M GGUF is), and the logit-cosine/KL gate needs a forward
pass. So the DECISIVE gate (`--colab`) runs where the f16 weights + a GPU live.

HOW QTIP QUALITY IS MODELED HERE (and why this is trustworthy, not a wrong sim)
------------------------------------------------------------------------------
QTIP = incoherence rotation (RHT) + a trellis-coded quantizer. This oracle models
the RHT EXACTLY (it is an orthogonal transform — no approximation) and BRACKETS
the trellis, rather than reimplementing a Viterbi codec that could be subtly
wrong ("a wrong trellis sim is worse than none", bible §8.3.1). For a STORED rate
of k bits/weight the trellis buys some coding gain over a memoryless k-bit grid
(QTIP/TCQ literature: up to ~1 bit). So real QTIP-k quality lies between:
  * LOWER bound  = RHT + optimal k-bit scalar quantizer   (no trellis gain), and
  * UPPER bound  = RHT + optimal (k+1)-bit scalar quantizer (full ~1-bit gain),
both at the SAME stored ~k-bit byte cut. Reporting the interval makes the one
thing the proxy cannot pin down explicit, instead of a single possibly-wrong
number. The decisive Colab run uses the REAL QTIP quantizer, not this model.

What runs IN-SESSION (CPU/NumPy, contamination-immune):
  * `--selftest`     : validates RHT, the scalar quantizers (Lloyd-Max optimal,
                       monotone, bracket ordering), the NumPy Q4_K_M baseline,
                       and the logit metrics. Exits non-zero on any failure.
  * `--local-proxy`  : grounds a DIRECTION-ONLY first cut in REAL Qwen weights:
                       (i) RHT-Gaussianization headroom (raw vs post-RHT excess-
                           kurtosis / max-mean — how much outlier tax RHT removes,
                           robust to Q4 noise: a distribution-shape stat, NOT
                           requant-from-Q4 quantization); and
                       (ii) a Q4_K vs QTIP[lower,upper] RMSE race on a CLEAN f32
                           bootstrap of the real weight marginal (fresh i.i.d.
                           draw -> Q4_K pays real quant error incl. the outlier
                           tax; NOT the forbidden requant-from-already-Q4).

KILL-PROTOCOL FRAMING (AGENT.md / bible §8.3.1)
------------------------------------------------
QTIP is the named, alive reframe of the Type-1 L1.5 gather-wall kill. This is its
named cheap quality oracle. GO floors (design §5.1): QTIP-3.0 RMSE <= Q4_K_M RMSE
at FEWER bytes (decisive: f16 source); QTIP logit-cosine >= Q4_K_M's, KL <=, and
argmax >=, ON CODE. NO-GO on the decisive (Colab) run => quality Type-1; records a
dead_levers entry and closes the byte-cut axis (with the §5.0/§6 speed gate).

RAM discipline: one real tensor at a time, del+gc, RSS ceiling 3 GB.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
MODEL = os.environ.get(
    "QTIP_MODEL", str(ROOT / "models/qwen2.5-3b-instruct-q4_k_m.gguf")
)
REPORT_MD = str(ROOT / "reports/oracle_qtip_quality.md")
REPORT_JSON = str(ROOT / "reports/oracle/qtip_quality.json")
RSS_CEIL_GB = 3.0

# Representative real tensors (attn + ffn across early/mid/late), retained from
# the original feasibility sweep. Q4_K + Q6_K both appear on disk.
SAMPLE_NAMES = [
    "blk.0.attn_q.weight",
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_down.weight",
    "blk.17.attn_output.weight",
    "blk.17.ffn_up.weight",
    "blk.35.attn_q.weight",
    "blk.35.ffn_down.weight",
]

RHT_BLOCK = 256       # incoherence rotation block width (power of 2)
QK = 256              # ggml K-quant super-block
SUB = 32              # K-quant sub-block
SEED = 0

try:
    import resource

    def rss_gb():
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / (1024**3) if sys.platform == "darwin" else kb / (1024**2)
except Exception:  # pragma: no cover
    def rss_gb():
        return float("nan")


def check_rss(where):
    g = rss_gb()
    if g > RSS_CEIL_GB:
        sys.stderr.write(f"[FATAL] RSS {g:.2f} GB > {RSS_CEIL_GB} GB at {where}\n")
        sys.exit(2)
    return g


# ============================================================================
# Incoherence preprocessing (RHT): random sign diagonal + block Walsh-Hadamard.
# Orthogonal and EXACT (||Rx|| == ||x||, R^-1 R = I). QTIP folds this offline
# into adjacent linears for ~zero decode cost; here we only need its quality
# effect (Gaussianization of the per-block weight distribution).
# ============================================================================
def fwht(a):
    """Normalized fast Walsh-Hadamard transform along the last axis (len = 2^p).
    Batch-robust: collapses leading axes, transforms, restores shape."""
    a = a.astype(np.float64)
    orig = a.shape
    n = orig[-1]
    a = a.reshape(-1, n).copy()
    h = 1
    while h < n:
        a = a.reshape(-1, n // (2 * h), 2, h)
        x = a[:, :, 0, :]
        y = a[:, :, 1, :]
        a = np.concatenate([x + y, x - y], axis=-1).reshape(-1, n)
        h *= 2
    return (a / np.sqrt(n)).reshape(orig)


def rht_signs(block, seed=SEED):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=block).astype(np.float64) * 2.0 - 1.0


def rht_apply(blocks, signs):
    """blocks: (..., B) with B == len(signs). Returns rotated (orthogonal)."""
    return fwht(blocks * signs)


def rht_inverse(blocks, signs):
    """Inverse of rht_apply. Normalized FWHT is its own inverse; undo signs."""
    return fwht(blocks) * signs


# ============================================================================
# Distribution-shape stats (robust to Q4 noise -> honest on real Q4_K-dequant).
# ============================================================================
def excess_kurtosis(x):
    x = x.ravel().astype(np.float64)
    x = x - x.mean()
    s = x.std()
    return float(np.mean((x / s) ** 4) - 3.0) if s > 0 else 0.0


def max_over_mean(x):
    x = np.abs(x.ravel().astype(np.float64))
    m = x.mean()
    return float(x.max() / m) if m > 0 else float("inf")


def rel_rmse(recon, ref):
    recon = recon.ravel().astype(np.float64)
    ref = ref.ravel().astype(np.float64)
    denom = np.linalg.norm(ref)
    return float(np.linalg.norm(recon - ref) / denom) if denom > 0 else float("nan")


# ============================================================================
# Optimal scalar quantizer for a unit Gaussian (Lloyd-Max), cached per #levels.
# Models the RHT-Gaussianized block's per-element quantizer. The trellis gain is
# bracketed by running this at nlevels = 2^k (lower) and 2^(k+1) (upper).
# ============================================================================
_LLOYD_CACHE = {}


def lloyd_max_gaussian(nlevels, iters=80, samples=400000, seed=SEED):
    if nlevels in _LLOYD_CACHE:
        return _LLOYD_CACHE[nlevels]
    rng = np.random.default_rng(seed)
    x = np.sort(rng.standard_normal(samples))
    p = (np.arange(nlevels) + 0.5) / nlevels         # init at quantile midpoints
    lv = np.sqrt(2.0) * _erfinv(2.0 * p - 1.0)
    for _ in range(iters):
        b = (lv[:-1] + lv[1:]) / 2.0                 # decision boundaries
        idx = np.searchsorted(b, x)
        sums = np.bincount(idx, weights=x, minlength=nlevels)
        cnts = np.bincount(idx, minlength=nlevels)
        new = np.where(cnts > 0, sums / np.maximum(cnts, 1), lv)
        if np.allclose(new, lv, atol=1e-8):
            lv = new
            break
        lv = new
    _LLOYD_CACHE[nlevels] = lv
    return lv


def _erfinv(y):
    try:
        from scipy.special import erfinv
        return erfinv(y)
    except Exception:
        a = 0.147                                    # Winitzki approx (~1e-3)
        ln = np.log(np.maximum(1 - y * y, 1e-300))
        t = 2 / (np.pi * a) + ln / 2
        return np.sign(y) * np.sqrt(np.sqrt(t * t - ln / a) - t)


def scalar_quantize(y, levels):
    """Nearest-level quantize y (any shape) to sorted `levels`."""
    b = (levels[:-1] + levels[1:]) / 2.0
    idx = np.searchsorted(b, y.ravel())
    return levels[idx].reshape(y.shape)


# ============================================================================
# QTIP quality model: RHT + scalar Lloyd-Max at the bracketed rate.
# Stored rate = `store_bits` (the byte cut); modeled quality = `quality_bits`
# (store_bits for the lower bound, store_bits+1 for the upper bound).
# ============================================================================
def qtip_quantize(W, store_bits=3, quality_bits=None, rht_block=RHT_BLOCK,
                  seed=SEED, scale_bits=16):
    if quality_bits is None:
        quality_bits = store_bits
    levels = lloyd_max_gaussian(1 << int(quality_bits))
    signs = rht_signs(rht_block, seed)
    R, C = W.shape
    nblk = C // rht_block
    recon = np.array(W, dtype=np.float64, copy=True)
    if nblk:
        seg = W[:, :nblk * rht_block].reshape(R, nblk, rht_block)
        rot = rht_apply(seg, signs)                   # (R, nblk, B) Gaussianized
        sigma = rot.std(axis=-1, keepdims=True)
        sigma = np.where(sigma == 0, 1.0, sigma)
        qhat = scalar_quantize(rot / sigma, levels) * sigma
        derot = rht_inverse(qhat, signs)
        recon[:, :nblk * rht_block] = derot.reshape(R, nblk * rht_block)
    eff_bits = store_bits + scale_bits / rht_block
    block_bytes = (store_bits * QK) / 8.0 + (scale_bits / 8.0) * (QK / rht_block)
    return recon, eff_bits, block_bytes


# ============================================================================
# NumPy Q4_K_M baseline (gguf.quantize does NOT implement K-quants — verified
# NotImplementedError for Q4_K/Q5_K/Q6_K). Faithful structure: 256-superblock =
# 8 sub-blocks of 32; per sub-block an affine 4-bit grid (iterative LS refit,
# like llama.cpp make_qkx2); the 8 scales + 8 mins quantized to 6 bits each with
# f16 super-scales. Effective 4 + (6+6)/32 + (16+16)/256 = 4.5 bits/weight.
# ============================================================================
def _affine4_refit(w, iters=5, eps=1e-12):
    """Vectorized 4-bit affine fit over sub-blocks. w: (nsub, SUB)."""
    wmin = w.min(1, keepdims=True)
    wmax = w.max(1, keepdims=True)
    rng = np.maximum(wmax - wmin, eps)
    scale = rng / 15.0
    mn = wmin.copy()
    q = np.clip(np.round((w - mn) / scale), 0, 15)
    for _ in range(iters):
        qbar = q.mean(1, keepdims=True)
        wbar = w.mean(1, keepdims=True)
        cov = ((q - qbar) * (w - wbar)).sum(1, keepdims=True)
        var = ((q - qbar) ** 2).sum(1, keepdims=True)
        new_scale = np.where(var > eps, cov / np.maximum(var, eps), scale)
        new_scale = np.where(new_scale > 0, new_scale, scale)
        mn = wbar - new_scale * qbar
        scale = new_scale
        q = np.clip(np.round((w - mn) / scale), 0, 15)
    return q, scale.squeeze(1), mn.squeeze(1)


def _q6(vals, signed):
    """Quantize a vector to 6 bits with a single f16-ish super-scale; return recon."""
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


def q4k_quantize(W):
    """Round-trip W (rows, cols, cols % 256 == 0) through a NumPy Q4_K_M-style
    grid. Returns (recon, eff_bits=4.5, block_bytes=144)."""
    R, C = W.shape
    assert C % QK == 0, f"Q4_K needs cols % 256 == 0 (got {C})"
    nsblk = C // QK
    out = np.empty_like(W, dtype=np.float64)
    for r in range(R):
        row = W[r].astype(np.float64).reshape(nsblk, QK)
        for sb in range(nsblk):
            sub = row[sb].reshape(8, SUB)             # 8 sub-blocks of 32
            q, scales, mins = _affine4_refit(sub)
            scales_q = _q6(scales, signed=False)      # 6-bit scales (>=0)
            mins_q = _q6(mins, signed=True)           # 6-bit mins (signed)
            recon = q * scales_q[:, None] + mins_q[:, None]
            out[r, sb * QK:(sb + 1) * QK] = recon.reshape(QK)
    return out, 4.5, 144.0


# ============================================================================
# Logit-domain metrics (the DECISIVE functional gate; consumed on Colab from
# exported next-token logits of the f16 / Q4_K_M / QTIP models on the code corpus).
# ============================================================================
def logit_cosine(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return float(np.mean(num / np.maximum(den, 1e-12)))


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def kl_div(p_logits, q_logits):
    """Mean KL(P || Q) over rows, P=softmax(p_logits) (the f16 reference)."""
    p = _softmax(p_logits.astype(np.float64))
    q = _softmax(q_logits.astype(np.float64))
    return float(np.mean(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), axis=-1)))


def argmax_agree(ref_logits, other_logits):
    return float(np.mean(np.argmax(ref_logits, -1) == np.argmax(other_logits, -1)))


# ============================================================================
# --selftest : validate the codec model + metrics on fixed synthetic inputs.
# ============================================================================
def selftest():
    rng = np.random.default_rng(SEED)
    fails = []

    # 1) RHT orthogonality + exact inverse
    signs = rht_signs(RHT_BLOCK)
    x = rng.standard_normal((4, RHT_BLOCK))
    rx = rht_apply(x, signs)
    if not np.allclose(np.linalg.norm(rx, axis=1), np.linalg.norm(x, axis=1), atol=1e-9):
        fails.append("RHT not norm-preserving")
    if not np.allclose(rht_inverse(rx, signs), x, atol=1e-9):
        fails.append("RHT inverse != identity")

    # 2) RHT Gaussianizes a heavy-tailed block
    heavy = rng.standard_t(3, size=(8, RHT_BLOCK))
    k_raw, k_rot = excess_kurtosis(heavy), excess_kurtosis(rht_apply(heavy, signs))
    if not (k_rot < k_raw):
        fails.append(f"RHT did not reduce kurtosis ({k_raw:.2f}->{k_rot:.2f})")

    # 3) Lloyd-Max levels optimal-ish: 8-level unit-Gaussian SQNR ~14.6 dB
    lv3 = lloyd_max_gaussian(8)
    g = rng.standard_normal(200000)
    mse3 = np.mean((g - scalar_quantize(g, lv3)) ** 2)
    sqnr_db = 10 * np.log10(1.0 / mse3)
    if not (13.5 < sqnr_db < 15.5):
        fails.append(f"8-level Gaussian SQNR off ({sqnr_db:.2f} dB, expect ~14.6)")

    # 4) bracket ordering: (k+1)-bit quality strictly better than k-bit
    mse4 = np.mean((g - scalar_quantize(g, lloyd_max_gaussian(16))) ** 2)
    if not (mse4 < mse3):
        fails.append(f"bracket not ordered (4b {mse4:.4f} !< 3b {mse3:.4f})")

    # 5) QTIP model on heavy-tailed matrix: upper bound beats lower at same bytes
    Wt = rng.standard_t(4, size=(32, 512)).astype(np.float32)
    lo, eb, bb = qtip_quantize(Wt, store_bits=3, quality_bits=3)
    hi, _, _ = qtip_quantize(Wt, store_bits=3, quality_bits=4)
    rr_lo, rr_hi = rel_rmse(lo, Wt), rel_rmse(hi, Wt)
    if not (rr_hi < rr_lo):
        fails.append(f"QTIP upper !< lower ({rr_hi:.4f} vs {rr_lo:.4f})")
    if not (90 <= bb <= 110):
        fails.append(f"QTIP-3 block bytes off ({bb:.0f}, expect ~98)")

    # 6) NumPy Q4_K_M baseline round-trips, beats gguf Q4_0 on a Gaussian source
    Wg = rng.standard_normal((16, 512)).astype(np.float32)
    q4k, q4k_eb, q4k_bb = q4k_quantize(Wg)
    rr_q4k = rel_rmse(q4k, Wg)
    q40_ok = True
    try:
        from gguf import quantize, dequantize, GGMLQuantizationType
        q40 = dequantize(quantize(Wg, GGMLQuantizationType.Q4_0),
                         GGMLQuantizationType.Q4_0).astype(np.float64).reshape(Wg.shape)
        rr_q40 = rel_rmse(q40, Wg)
        if not (rr_q4k < rr_q40):
            fails.append(f"Q4_K_M not better than Q4_0 ({rr_q4k:.4f} vs {rr_q40:.4f})")
    except Exception as e:
        q40_ok = False
        sys.stderr.write(f"[selftest] gguf Q4_0 cross-check skipped: {e}\n")
    if not (0.0 < rr_q4k < 0.12):
        fails.append(f"Q4_K_M rel-RMSE implausible ({rr_q4k:.4f})")

    # 7) logit metrics
    lg = rng.standard_normal((16, 100))
    if not (abs(logit_cosine(lg, lg) - 1) < 1e-9 and kl_div(lg, lg) < 1e-9
            and argmax_agree(lg, lg) == 1.0):
        fails.append("logit metrics wrong on identical input")

    print("=== QTIP oracle self-test ===")
    print(f"  RHT orthogonal + exact inverse ....... OK")
    print(f"  RHT Gaussianizes heavy tail .......... exkurt {k_raw:.2f} -> {k_rot:.2f}")
    print(f"  Lloyd-Max 8-level Gaussian SQNR ...... {sqnr_db:.2f} dB (expect ~14.6)")
    print(f"  bracket ordering 4b<3b MSE ........... {mse4:.4f} < {mse3:.4f}")
    print(f"  QTIP upper<lower @ {bb:.0f}B/blk ........... rmse {rr_hi:.4f} < {rr_lo:.4f} ({eb:.2f} eff bits)")
    print(f"  NumPy Q4_K_M baseline ................ rel-RMSE {rr_q4k:.4f} (4.5b)"
          + (f" < Q4_0 {rr_q40:.4f}" if q40_ok else ""))
    print(f"  logit cos/KL/argmax plumbing ......... OK")
    if fails:
        print("\nSELF-TEST FAILED:")
        for f in fails:
            print("  - " + f)
        sys.exit(1)
    print("\nself-test PASSED — codec model + metrics are trustworthy.")
    return True


# ============================================================================
# --local-proxy : DIRECTION-ONLY first cut grounded in real Qwen weights.
# ============================================================================
def local_proxy(store_bits=3, max_rows=48, boot_cols=512):
    from gguf import GGUFReader, dequantize
    t0 = time.time()
    reader = GGUFReader(MODEL)
    by_name = {t.name: t for t in reader.tensors}
    rng = np.random.default_rng(SEED)

    rows = []
    for nm in SAMPLE_NAMES:
        if nm not in by_name:
            sys.stderr.write(f"[skip] {nm} not in model\n")
            continue
        t = by_name[nm]
        shape = tuple(int(x) for x in t.shape)        # (cols, rows)
        n_cols, m_rows = shape[0], shape[1]
        w = dequantize(np.array(t.data), t.tensor_type).astype(np.float32)
        W = w.reshape(m_rows, n_cols)
        del w
        check_rss(f"deq {nm}")

        # (i) RHT-Gaussianization headroom (distribution shape; Q4-noise robust)
        nb = (n_cols // RHT_BLOCK) * RHT_BLOCK
        flat = W[:, :nb].reshape(-1, RHT_BLOCK)
        signs = rht_signs(RHT_BLOCK)
        raw_k, raw_mm = excess_kurtosis(flat), max_over_mean(flat)
        rot = rht_apply(flat, signs)
        rht_k, rht_mm = excess_kurtosis(rot), max_over_mean(rot)
        del rot

        # (ii) Q4_K vs QTIP[lower,upper] RMSE on a CLEAN bootstrap of the real
        #      marginal (fresh i.i.d. -> Q4_K pays real quant error incl. outlier
        #      tax; NOT requant-from-already-Q4). Small sub-matrix for cost.
        vals = W[:, :nb].ravel()
        sub_rows = min(max_rows, m_rows)
        cols = (boot_cols // QK) * QK
        boot = rng.choice(vals, size=sub_rows * cols, replace=True).astype(
            np.float32).reshape(sub_rows, cols)
        del vals
        q4, q4_bits, q4_bytes = q4k_quantize(boot)
        rr_q4 = rel_rmse(q4, boot)
        del q4
        lo, qt_bits, qt_bytes = qtip_quantize(boot, store_bits=store_bits,
                                              quality_bits=store_bits)
        rr_lo = rel_rmse(lo, boot)
        del lo
        hi, _, _ = qtip_quantize(boot, store_bits=store_bits,
                                 quality_bits=store_bits + 1)
        rr_hi = rel_rmse(hi, boot)
        del hi, boot

        rows.append(dict(
            name=nm, shape=[m_rows, n_cols], disk=t.tensor_type.name,
            raw_exkurt=raw_k, raw_maxmean=raw_mm,
            rht_exkurt=rht_k, rht_maxmean=rht_mm,
            kurt_drop=(raw_k - rht_k), mm_drop=(raw_mm - rht_mm),
            q4k_rmse=rr_q4, q4k_bits=q4_bits, q4k_bytes=q4_bytes,
            qtip_rmse_lower=rr_lo, qtip_rmse_upper=rr_hi,
            qtip_bits=qt_bits, qtip_bytes=qt_bytes,
            qtip_lower_beats_q4k=bool(rr_lo <= rr_q4),
            qtip_upper_beats_q4k=bool(rr_hi <= rr_q4),
        ))
        g = check_rss(f"done {nm}")
        sys.stderr.write(f"[ok] {nm} RSS={g:.2f} GB exkurt {raw_k:.2f}->{rht_k:.2f} "
                         f"rmse q4k={rr_q4:.4f} qtip[{rr_lo:.4f},{rr_hi:.4f}]\n")
        del W, flat
        gc.collect()

    return rows, time.time() - t0


# ============================================================================
# Report writers + verdict logic.
# ============================================================================
def write_reports(rows, wall, store_bits):
    n = len(rows)
    n_lo = sum(r["qtip_lower_beats_q4k"] for r in rows)
    n_hi = sum(r["qtip_upper_beats_q4k"] for r in rows)
    med = lambda key: float(np.median([r[key] for r in rows])) if rows else float("nan")
    med_kurt_drop, med_mm_drop = med("kurt_drop"), med("mm_drop")
    med_q4, med_lo, med_hi = med("q4k_rmse"), med("qtip_rmse_lower"), med("qtip_rmse_upper")
    qt_bytes = rows[0]["qtip_bytes"] if rows else float("nan")
    q4_bytes = rows[0]["q4k_bytes"] if rows else 144.0
    eff = rows[0]["qtip_bits"] if rows else float("nan")
    cut = (1.0 - qt_bytes / q4_bytes) * 100 if rows else float("nan")

    L = []
    o = L.append
    o("# Oracle — QTIP byte-cut QUALITY (L1.5 reframe, axis-2 deep byte-cut)")
    o("")
    o(f"**Model:** `models/qwen2.5-3b-instruct-q4_k_m.gguf`  **Lane:** CPU NumPy  "
      f"**Target store:** ~{store_bits}.0 bits (~{eff:.2f} eff, ~{qt_bytes:.0f} B/256-blk)")
    o("**Date:** 2026-05-31")
    o("")
    o("> **Scope = DIRECTION-ONLY local proxy.** The DECISIVE quality verdict "
      "(recon-RMSE vs **f16** + logit-cosine/KL/argmax on code) is **Colab** — QTIP "
      "must be fit **from f16**, never requant-from-Q4_K (a recorded kill), and the "
      "f16 Qwen + a forward pass are not on this machine. Numbers below are grounded "
      "in REAL Qwen weights but are a first cut, not the gate.")
    o("")
    o("> **QTIP quality is BRACKETED** [lower, upper] = RHT + optimal {k, k+1}-bit "
      "scalar quantizer, both at the SAME ~k-bit byte cut. The trellis coding gain "
      "(≤~1 bit, QTIP/TCQ literature) lives inside this interval; the proxy models "
      "RHT exactly and refuses to invent a single trellis number ('a wrong trellis "
      "sim is worse than none').")
    o("")
    o("## (i) RHT-Gaussianization headroom on real weights")
    o("")
    o("How much outlier tax the incoherence rotation removes — QTIP's core lever. "
      "Excess-kurtosis and max/mean are distribution-SHAPE stats (robust to Q4 "
      "dequant noise; not requant-from-Q4). Gaussian ref: exkurt 0, max/mean ~6 "
      "per 256-block.")
    o("")
    o("| tensor | shape | disk | raw exkurt | RHT exkurt | raw max/mean | RHT max/mean |")
    o("|---|---|---|---|---|---|---|")
    for r in rows:
        o(f"| {r['name']} | {r['shape'][0]}x{r['shape'][1]} | {r['disk']} | "
          f"{r['raw_exkurt']:.2f} | {r['rht_exkurt']:.2f} | "
          f"{r['raw_maxmean']:.1f} | {r['rht_maxmean']:.1f} |")
    o("")
    o(f"**Median reduction:** excess-kurtosis −{med_kurt_drop:.2f}, max/mean "
      f"−{med_mm_drop:.1f}. RHT materially Gaussianizes (the precondition for a "
      "low-bit quantizer to hit its rate-distortion target), but real weights stay "
      "somewhat heavier-tailed than ideal Gaussian after a single 256-RHT.")
    o("")
    o("## (ii) Q4_K vs QTIP RMSE — clean bootstrap of the real marginal")
    o("")
    o(f"Fresh i.i.d. resample of each tensor's real weight values (clean f32 -> Q4_K "
      f"pays real quant error incl. the outlier tax; **not** the forbidden requant-"
      f"from-already-Q4). QTIP[lower,upper] at ~{qt_bytes:.0f} B/256-blk vs Q4_K_M "
      f"(NumPy, 4.5 bits, {q4_bytes:.0f} B).")
    o("")
    o("| tensor | Q4_K_M rmse | QTIP lower | QTIP upper | lower≤Q4K | upper≤Q4K |")
    o("|---|---|---|---|---|---|")
    for r in rows:
        o(f"| {r['name']} | {r['q4k_rmse']:.4f} | {r['qtip_rmse_lower']:.4f} | "
          f"{r['qtip_rmse_upper']:.4f} | {'YES' if r['qtip_lower_beats_q4k'] else 'no'} | "
          f"{'YES' if r['qtip_upper_beats_q4k'] else 'no'} |")
    o("")
    bits_needed = float(np.log2(med_lo / med_q4)) if (med_lo > 0 and med_q4 > 0) else float("nan")
    o(f"**Median:** Q4_K_M {med_q4:.4f} vs QTIP [{med_lo:.4f}, {med_hi:.4f}] at "
      f"−{cut:.0f}% bytes. QTIP-lower ≤ Q4_K on **{n_lo}/{n}** tensors; "
      f"QTIP-upper ≤ Q4_K on **{n_hi}/{n}**.")
    o("")
    o(f"**Bits-equivalent gap:** to MATCH Q4_K_M weight-RMSE at this byte budget, QTIP "
      f"must extract **~{bits_needed:.2f} bits** of combined RHT-whitening + trellis "
      f"coding gain over a {store_bits}-bit scalar (each ~1 bit ≈ 6 dB ≈ ×½ RMSE). "
      f"That is at the **upper edge** of the TCQ envelope (~0.5–1 bit typical, ~1.2 "
      f"deep) — the modeled +1-bit upper bound still falls ~{(med_hi/med_q4-1)*100:.0f}% "
      f"short. So matching Q4_K_M on RMSE is possible only if the real trellis lands "
      f"near its best-case gain AND RHT whitens more than the single 256-rotation here.")
    o("")
    o("## Direction read (NOT the verdict)")
    o("")
    half = (n + 1) // 2
    if n_lo >= half:
        o(f"- **Strong GO direction.** Even the LOWER bound (no trellis gain) "
          f"matches-or-beats Q4_K_M RMSE on {n_lo}/{n} tensors at −{cut:.0f}% bytes — "
          "RHT's outlier-tax removal alone covers the 1.5-bit deficit. Real QTIP "
          "(with trellis gain) is better still. This robustly EARNS the Colab f16 + "
          "logit gate.")
    elif n_hi >= half:
        o(f"- **Conditional GO direction.** Q4_K_M wins on the lower bound but the "
          f"UPPER bound (full ~1-bit trellis gain) matches-or-beats it on {n_hi}/{n} "
          "tensors — so whether QTIP-3.0 clears Q4_K depends on how much of the "
          "trellis coding gain is real. The Colab f16 gate (with the REAL QTIP "
          "quantizer) is exactly what settles it; this is the case the proxy cannot "
          "decide.")
    else:
        o(f"- **Cautionary direction.** Even the UPPER bound (full ~1-bit trellis "
          f"gain) trails Q4_K_M on {n - n_hi}/{n} tensors. RHT + ~3-bit does not "
          "cover the 1.5-bit deficit on this proxy. The Colab f16 gate must clear a "
          "real margin or QTIP's quality Type-1 fires (dead_levers entry; closes the "
          "byte-cut axis with the §5.0/§6 speed gate).")
    o("- **Caveats (why direction-only):** (1) bootstrap destroys per-channel "
      "spatial structure; (2) RMSE ≠ logit quality — the decisive metric is logit-"
      "cosine/KL/argmax on code (Colab); (3) the QTIP source here is a resample of "
      "Q4_K-dequant values, not true f16 — the f16 source is the only fair quantizer "
      "input (kill-respect #2); (4) the Q4_K_M baseline is a faithful NumPy "
      "reimplementation (gguf has no K-quant quantizer), a touch below llama.cpp's "
      "importance-weighted optimum.")
    o("")
    o("## How the DECISIVE (Colab) gate fires — `--colab` runbook")
    o("")
    o("On Colab, with f16 Qwen2.5-3B + a code corpus:")
    o("1. Export per-tensor f16 weights -> `weights.npz`; run "
      "`oracle_qtip_quality.py --colab weights.npz` for **recon-RMSE vs f16** "
      "(QTIP-3.0/3.25 vs real Q4_K_M; **GO floor: QTIP ≤ Q4_K_M at fewer bytes**). "
      "Swap in the REAL QTIP quantizer for the codec there.")
    o("2. Forward-pass f16 / Q4_K_M / QTIP on held-out **code** tokens; export "
      "next-token logits -> three `.npy`; run `--colab ... --logits f16 q4k qtip` "
      "for **logit-cosine / KL / argmax** (GO: QTIP ≥ Q4_K_M cosine & argmax, ≤ KL).")
    o("3. GO on both -> §5.2 on-GPU decode-cost oracle (trellis BW-bound on M3?). "
      "NO-GO on either -> quality Type-1 kill; byte-cut axis closes.")
    o("")
    o(f"_Wall: {wall:.1f}s. Peak RSS: {check_rss('report'):.2f} GB. Run `--selftest` "
      "(must pass) for these numbers to be trustworthy._")

    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(REPORT_JSON, "w") as f:
        json.dump(dict(
            target_store_bits=store_bits, n_tensors=n,
            n_qtip_lower_beats_q4k=n_lo, n_qtip_upper_beats_q4k=n_hi,
            median_kurtosis_drop=med_kurt_drop, median_maxmean_drop=med_mm_drop,
            median_q4k_rmse=med_q4, median_qtip_rmse_lower=med_lo,
            median_qtip_rmse_upper=med_hi, qtip_block_bytes=qt_bytes,
            q4k_block_bytes=q4_bytes,
            verdict="DIRECTION-ONLY (decisive gate = Colab f16 + logits)",
            per_tensor=rows,
        ), f, indent=2)
    sys.stderr.write(f"[done] wrote {REPORT_MD} + {REPORT_JSON}\n")
    print("\n".join(L))


def colab_eval(weights_npz, logits=None, store_bits=3):
    """DECISIVE gate — runs on Colab against f16-exported tensors (+ optional
    logits). Off-Colab (no f16 weights here) it fails loudly."""
    if not os.path.exists(weights_npz):
        sys.stderr.write(
            f"[colab] {weights_npz} not found. This is the Colab f16 path: export "
            "per-tensor f16 Qwen weights to an .npz first (see the report runbook). "
            "It cannot run locally — only the Q4_K_M GGUF is on this machine. For the "
            "REAL verdict, swap the bracketed scalar model for the QTIP quantizer.\n")
        sys.exit(2)
    data = np.load(weights_npz)
    res = []
    for name in data.files:
        Wf16 = data[name].astype(np.float32)
        if Wf16.ndim != 2 or Wf16.shape[1] % QK:
            continue
        q4, _, q4b = q4k_quantize(Wf16)
        lo, _, qtb = qtip_quantize(Wf16, store_bits=store_bits, quality_bits=store_bits)
        hi, _, _ = qtip_quantize(Wf16, store_bits=store_bits, quality_bits=store_bits + 1)
        res.append(dict(name=name, q4k_rmse=rel_rmse(q4, Wf16),
                        qtip_lower=rel_rmse(lo, Wf16), qtip_upper=rel_rmse(hi, Wf16),
                        qtip_block_bytes=qtb, q4k_block_bytes=q4b))
        del Wf16, q4, lo, hi
        gc.collect()
    mq4 = float(np.median([r["q4k_rmse"] for r in res]))
    mlo = float(np.median([r["qtip_lower"] for r in res]))
    mhi = float(np.median([r["qtip_upper"] for r in res]))
    go = mlo <= mq4
    print(f"[colab recon gate] median RMSE  Q4_K_M={mq4:.4f}  QTIP=[{mlo:.4f},{mhi:.4f}]  "
          f"-> lower {'≤' if go else '>'} Q4_K  ({'GO' if go else 'check upper/logits'})")
    if logits and len(logits) == 3:
        f16, q4k, qtp = (np.load(p) for p in logits)
        print(f"[colab logit gate] cos(QTIP,f16)={logit_cosine(qtp,f16):.5f} "
              f"cos(Q4K,f16)={logit_cosine(q4k,f16):.5f} | "
              f"KL(f16||QTIP)={kl_div(f16,qtp):.5f} KL(f16||Q4K)={kl_div(f16,q4k):.5f} | "
              f"argmax(QTIP,f16)={argmax_agree(f16,qtp):.4f} "
              f"argmax(Q4K,f16)={argmax_agree(f16,q4k):.4f}")
    return res


def main():
    ap = argparse.ArgumentParser(description="QTIP byte-cut quality oracle")
    ap.add_argument("--selftest", action="store_true",
                    help="validate codec model + metrics on synthetic (in-session gate)")
    ap.add_argument("--local-proxy", action="store_true",
                    help="direction-only first cut on real Qwen weights (in-session)")
    ap.add_argument("--colab", metavar="WEIGHTS_NPZ",
                    help="DECISIVE gate: f16-exported tensors (runs on Colab)")
    ap.add_argument("--logits", nargs=3, metavar=("F16", "Q4K", "QTIP"),
                    help="three logit .npy for the functional gate (with --colab)")
    ap.add_argument("--bits", type=int, default=3, help="QTIP stored bits/weight")
    args = ap.parse_args()

    if not (args.selftest or args.local_proxy or args.colab):
        ap.error("pick one of --selftest / --local-proxy / --colab")
    if args.selftest:
        selftest()
    if args.colab:
        colab_eval(args.colab, args.logits, store_bits=args.bits)
    if args.local_proxy:
        rows, wall = local_proxy(store_bits=args.bits)
        if not rows:
            sys.stderr.write("[FATAL] no tensors processed\n")
            sys.exit(1)
        write_reports(rows, wall, args.bits)


if __name__ == "__main__":
    main()

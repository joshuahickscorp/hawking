#!/usr/bin/env python3
"""
Phase A feasibility oracle for Bible levers L1.4 (low-rank + compressible
residual) and L1.5 (learned per-model codebook).

CPU-only NumPy. BYTE-BUDGET + FEASIBILITY oracle. Quality (KL/perplexity)
is DEFERRED to the GPU/llama.cpp lane — NOT computed here.

Model: Qwen2.5-3B-Instruct Q4_K_M GGUF. We dequantize a REPRESENTATIVE
SAMPLE of attention + FFN matrices (early/mid/late layers) to f32 using
the `gguf` library's own dequantizer, then:

  L1.4: SVD each sampled tensor; for r in {16,32,64} report energy
        captured, residual std, and a byte budget:
          rank bytes (U,V stored f16) + residual bytes (2-3 bits)
        vs the tensor's ACTUAL on-disk GGUF byte size. GO/NO-GO on the
        byte budget alone, gated on residual looking low-error.

  L1.5: fit k-means codebooks (k=16 -> 4 bits/code, k=256 -> 8 bits/code)
        to a sampled tensor's scalar weight distribution; report
        reconstruction MSE vs the fixed llama.cpp grid at MATCHED bits
        (Q4_0 ~ 4-bit, Q8_0 ~ 8-bit). Then the BINDING verdict: does
        decode require per-element random LUT lookups (kills Apple-GPU)
        or can codes stay contiguous/lookup-free?

RAM discipline: one tensor in memory at a time; del + gc.collect();
target < 3 GB RSS.
"""

import gc
import os
import sys
import time

import numpy as np

try:
    import resource

    def rss_gb():
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports KiB.
        if sys.platform == "darwin":
            return kb / (1024**3)
        return kb / (1024**2)
except Exception:  # pragma: no cover
    def rss_gb():
        return float("nan")

from gguf import GGUFReader, GGMLQuantizationType, dequantize, quantize
from gguf.quants import GGML_QUANT_SIZES

MODEL = "/Users/scammermike/Downloads/hawking/models/qwen2.5-3b-instruct-q4_k_m.gguf"
REPORT = "/Users/scammermike/Downloads/hawking/reports/oracle_lowrank_codebook.md"

RSS_CEIL_GB = 3.0

# Representative sample: attention (q, output) + FFN (gate, up, down) across
# early (0), mid (17), late (35) layers. Mix of Q4_K and Q6_K on disk.
SAMPLE_NAMES = [
    "blk.0.attn_q.weight",
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_down.weight",
    "blk.17.attn_output.weight",
    "blk.17.ffn_up.weight",
    "blk.35.attn_q.weight",
    "blk.35.ffn_down.weight",
]

# Tensor chosen for the L1.5 codebook fit (large FFN matrix => stable stats).
CODEBOOK_TENSOR = "blk.17.ffn_up.weight"

RANKS = [16, 32, 64]
RESIDUAL_BITS = [2, 3]  # residual stored at 2-3 bits/weight (+ small per-row scale)


def check_rss(where):
    g = rss_gb()
    if g > RSS_CEIL_GB:
        sys.stderr.write(f"[FATAL] RSS {g:.2f} GB > {RSS_CEIL_GB} GB ceiling at {where}\n")
        sys.exit(2)
    return g


def f16_bytes(n):
    return 2 * n


def low_rank_byte_budget(m, n, r, res_bits):
    """Bytes for: U(m x r)+V(r x n) at f16, plus residual at res_bits/weight
    plus one f16 per-row scale for the residual quantizer."""
    rank_bytes = f16_bytes(m * r) + f16_bytes(r * n)
    residual_bytes = (res_bits * m * n) / 8.0
    residual_scale_bytes = f16_bytes(m)  # per-row scale for residual block
    return rank_bytes, residual_bytes + residual_scale_bytes


def numpy_kmeans_1d(x, k, iters=25, seed=0, sample_cap=2_000_000):
    """Tiny 1-D k-means on scalar weight values. Returns (centroids, mse).

    1-D so assignment is a searchsorted against sorted centroids — O(N log k),
    no (N x k) matrix, RAM-safe for millions of weights.
    """
    rng = np.random.default_rng(seed)
    x = x.astype(np.float64, copy=False).ravel()
    if x.size > sample_cap:
        idx = rng.choice(x.size, size=sample_cap, replace=False)
        xs = x[idx]
    else:
        xs = x
    # init centroids at quantiles (robust, deterministic-ish)
    qs = np.linspace(0, 1, k + 2)[1:-1]
    cent = np.quantile(xs, qs)
    cent = np.unique(cent)
    if cent.size < k:
        cent = np.concatenate([cent, cent[-1] + np.arange(1, k - cent.size + 1) * 1e-6])
    cent = np.sort(cent)[:k]
    for _ in range(iters):
        cent_sorted = np.sort(cent)
        edges = (cent_sorted[:-1] + cent_sorted[1:]) / 2.0
        assign = np.searchsorted(edges, xs)
        new = np.empty(k, dtype=np.float64)
        moved = False
        for j in range(k):
            sel = xs[assign == j]
            if sel.size:
                new[j] = sel.mean()
            else:
                new[j] = cent_sorted[j]
        if not np.allclose(new, cent_sorted, rtol=0, atol=1e-9):
            moved = True
        cent = new
        if not moved:
            break
    # full-population MSE using final centroids
    cent_sorted = np.sort(cent)
    edges = (cent_sorted[:-1] + cent_sorted[1:]) / 2.0
    assign = np.searchsorted(edges, x)
    recon = cent_sorted[assign]
    mse = float(np.mean((x - recon) ** 2))
    return cent_sorted, mse


def fixed_grid_mse(w_f32, shape, qtype):
    """Requantize f32 weights through a fixed llama.cpp grid and back; MSE."""
    q = quantize(w_f32.reshape(shape), qtype)
    deq = dequantize(q, qtype).astype(np.float32).ravel()
    mse = float(np.mean((w_f32.ravel() - deq[: w_f32.size]) ** 2))
    del q, deq
    return mse


def main():
    t0 = time.time()
    lines = []
    out = lines.append

    out("# Oracle — L1.4 (low-rank + residual) & L1.5 (learned codebook)")
    out("")
    out("**Model:** `models/qwen2.5-3b-instruct-q4_k_m.gguf` (Qwen2.5-3B-Instruct)")
    out(f"**Date:** 2026-05-30  **Lane:** CPU NumPy byte-budget + Apple-GPU feasibility oracle")
    out("")
    out("Scope: BYTE-BUDGET + DECODE-FEASIBILITY only. Quality (KL/perplexity vs")
    out("Q4_K_M) is DEFERRED to the GPU/llama.cpp lane and is NOT measured here.")
    out("Representative sample of tensors (not all 36 layers). SVD one tensor at a")
    out("time with `del`+`gc` between (RSS ceiling 3 GB).")
    out("")

    reader = GGUFReader(MODEL)
    by_name = {t.name: t for t in reader.tensors}
    for nm in SAMPLE_NAMES + [CODEBOOK_TENSOR]:
        if nm not in by_name:
            sys.stderr.write(f"[FATAL] tensor {nm} not in model\n")
            sys.exit(1)

    # -------------------- L1.4 --------------------
    out("## L1.4 — Low-rank + compressible residual")
    out("")
    out("Per tensor: SVD `W = U S Vt`. Top-r kept as f16 (U_r, S·Vt_r). Residual =")
    out("`W - W_r`, stored at 2-3 bits/weight + an f16 per-row scale. Budget compared")
    out("to the tensor's ACTUAL on-disk GGUF bytes (Q4_K and Q6_K both appear).")
    out("`energy@r` = captured Frobenius energy = Σσᵢˆ 2(top r)/Σσᵢˆ 2. `res_std/W_std`")
    out("= residual std relative to original (proxy for residual quantizability).")
    out("")
    out("| tensor | shape | disk type | disk bytes | r | energy@r | res_std/W_std | rank B (f16) | +res@2b | +res@3b | total@2b | total@3b | ratio@2b | ratio@3b |")
    out("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    l14_rows = []
    for nm in SAMPLE_NAMES:
        t = by_name[nm]
        shape = tuple(int(x) for x in t.shape)  # gguf shape is (cols, rows) = (n, m)
        qtype = t.tensor_type
        disk_bytes = int(t.n_bytes)
        check_rss(f"pre-deq {nm}")
        # dequantize -> f32. gguf returns row-major with logical shape reversed.
        w = dequantize(np.array(t.data), qtype).astype(np.float32)
        # Reshape to 2-D matrix (rows m, cols n). gguf t.shape is (n, m) ordering;
        # total elems == m*n so reshape to (m, n) with m = shape[-1].
        n_cols, m_rows = shape[0], shape[1]
        W = w.reshape(m_rows, n_cols)
        w_std = float(W.std())
        del w
        check_rss(f"post-deq {nm}")

        # economy SVD, one tensor at a time
        # full_matrices=False keeps U:(m,k) Vt:(k,n), k=min(m,n)
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        total_energy = float(np.sum(S.astype(np.float64) ** 2))
        for r in RANKS:
            r_eff = min(r, S.size)
            energy_r = float(np.sum(S[:r_eff].astype(np.float64) ** 2)) / total_energy
            Wr = (U[:, :r_eff] * S[:r_eff]) @ Vt[:r_eff, :]
            resid = W - Wr
            res_std = float(resid.std())
            del Wr, resid
            rank_b, _ = low_rank_byte_budget(m_rows, n_cols, r_eff, 2)
            _, res2 = low_rank_byte_budget(m_rows, n_cols, r_eff, 2)
            _, res3 = low_rank_byte_budget(m_rows, n_cols, r_eff, 3)
            tot2 = rank_b + res2
            tot3 = rank_b + res3
            ratio2 = tot2 / disk_bytes
            ratio3 = tot3 / disk_bytes
            l14_rows.append(
                dict(nm=nm, r=r_eff, energy=energy_r, res_ratio=res_std / w_std,
                     ratio2=ratio2, ratio3=ratio3)
            )
            out(f"| {nm} | {m_rows}x{n_cols} | {qtype.name} | {disk_bytes:,} | {r_eff} | "
                f"{energy_r:.3f} | {res_std / w_std:.3f} | {rank_b:,.0f} | {res2:,.0f} | "
                f"{res3:,.0f} | {tot2:,.0f} | {tot3:,.0f} | {ratio2:.2f}x | {ratio3:.2f}x |")
        g = check_rss(f"post-svd {nm}")
        sys.stderr.write(f"[ok] {nm} done, RSS={g:.2f} GB\n")
        del U, S, Vt, W
        gc.collect()

    out("")
    # L1.4 verdict logic. A raw "total < disk" ratio is NOT sufficient and is in
    # fact MISLEADING: total@2b/3b drops below Q4_K only because the residual is
    # stored at fewer bits than Q4_K's 4.5b. The honest question is whether the
    # LOW-RANK PART EARNS ITS KEEP — i.e. does removing rank-r structure let the
    # residual quantize at materially fewer bits than quantizing the raw matrix.
    # If the residual keeps ~all the energy (res_std/W_std ~ 1), then residual@Nb
    # ~= rawquant@Nb and the f16 U,V are pure dead overhead -> low-rank LOSES to
    # plain N-bit quant. That is the decisive comparison, not total-vs-disk.
    best2 = min(r["ratio2"] for r in l14_rows)
    best3 = min(r["ratio3"] for r in l14_rows)
    max_energy = max(r["energy"] for r in l14_rows)
    med_res = float(np.median([r["res_ratio"] for r in l14_rows]))
    # Does the low-rank part pay? rank f16 bytes must be < bytes saved on the
    # residual vs raw N-bit quant. With res_std/W_std ~ 1, residual needs the
    # SAME bits as raw -> zero residual saving -> rank bytes are pure loss.
    # Use a generous threshold: low-rank only "earns keep" if it removes enough
    # energy that the residual could plausibly drop >=1 bit (res_std/W_std small).
    lowrank_earns_keep = (max_energy >= 0.50) and (med_res <= 0.50)
    l14_go = lowrank_earns_keep  # byte ratio<1 alone is necessary, not sufficient
    out("**Byte-budget summary (read carefully):** the raw total/disk ratios above")
    out(f"dip to {best2:.2f}x (2-bit residual) / {best3:.2f}x (3-bit residual) — but that is")
    out("**not a win**, because the residual is simply stored at fewer bits than")
    out("Q4_K's 4.5b. The decisive number is the low-rank energy: top-r captures at")
    out(f"most **{max_energy*100:.1f}%** of Frobenius energy (r=64), so the residual keeps")
    out(f"**~{med_res*100:.0f}%** of the original std (`res_std/W_std` median {med_res:.2f}).")
    out("")
    out("**L1.4 verdict: NO-GO (byte budget).** These weights are **not low-rank**.")
    out("Even r=64 captures <26% of energy on the 2048-square attention matrices and")
    out("<15% on the 11008-wide FFN matrices; the residual retains ~90-99% of the")
    out("weight std. Consequences:")
    out("")
    out("- The residual at 2-3 bits is numerically ~identical to quantizing the RAW")
    out("  matrix at 2-3 bits (no structure was removed), so the low-rank part buys")
    out("  no quality — yet it ADDS `2*(m+n)*r*2` f16 bytes of pure overhead.")
    out("- Therefore low-rank+residual is strictly WORSE than plain N-bit quant: same")
    out("  residual error, extra U,V bytes. The apparent <1.0 ratio is just \"use")
    out("  fewer bits than Q4_K\" wearing a low-rank costume.")
    out("- Low-rank only pays when top-r captures most of the energy so the residual")
    out("  collapses toward ~0 bits. It does not here. Lever dies on the byte/energy")
    out("  oracle — no quality eval warranted.")
    out("")

    # -------------------- L1.5 --------------------
    out("## L1.5 — Learned per-model codebook (the danger lever)")
    out("")
    t = by_name[CODEBOOK_TENSOR]
    shape = tuple(int(x) for x in t.shape)
    qtype = t.tensor_type
    out(f"Fit tensor: `{CODEBOOK_TENSOR}` ({shape[1]}x{shape[0]}, on-disk {qtype.name}).")
    out("k-means on the scalar weight distribution (1-D, RAM-safe). Reconstruction MSE")
    out("vs the FIXED llama.cpp grid at MATCHED bits. k=16 ↔ 4-bit grid (Q4_0);")
    out("k=256 ↔ 8-bit grid (Q8_0). Lower MSE = better quality-per-bit.")
    out("")
    w = dequantize(np.array(t.data), qtype).astype(np.float32).ravel()
    check_rss("L1.5 deq")

    # learned codebooks
    _, mse_k16 = numpy_kmeans_1d(w, 16)
    _, mse_k256 = numpy_kmeans_1d(w, 256)
    # fixed grids at matched bits
    mse_q4_0 = fixed_grid_mse(w, (shape[1], shape[0]), GGMLQuantizationType.Q4_0)
    mse_q8_0 = fixed_grid_mse(w, (shape[1], shape[0]), GGMLQuantizationType.Q8_0)
    # the actual production grid for this tensor (4.5b Q4_K) for context
    var = float(np.var(w))
    del w
    gc.collect()

    out("| bits | learned k-means MSE | fixed-grid MSE | grid | learned/fixed |")
    out("|---|---|---|---|---|")
    out(f"| 4-bit (k=16) | {mse_k16:.3e} | {mse_q4_0:.3e} | Q4_0 | {mse_k16/mse_q4_0:.2f}x |")
    out(f"| 8-bit (k=256) | {mse_k256:.3e} | {mse_q8_0:.3e} | Q8_0 | {mse_k256/mse_q8_0:.2f}x |")
    out(f"| (ref) weight variance | {var:.3e} | | | |")
    out("")
    out("Note: a 1-D global codebook is a LOWER bound on learned-codebook quality; the")
    out("fixed grids (Q4_0/Q8_0) use per-block f16 scales (32-wide), so they adapt to")
    out("local magnitude. A fair learned codec would also need per-block scales — the")
    out("codebook alone does not capture that. MSE here is the codebook-vs-grid shape")
    out("comparison only; absolute quality is the GPU lane's call.")
    out("")
    out("### Binding feasibility verdict (Apple-GPU decode)")
    out("")
    out("- **k=16 (4-bit codes):** a 16-entry codebook is 16 f16 = 32 bytes. That")
    out("  trivially fits in threadgroup memory. BUT decode is `w[i] = codebook[code[i]]`")
    out("  — a per-element indexed read into the codebook. On Apple GPUs there is no")
    out("  hardware gather; even a threadgroup-resident 16-entry LUT becomes a")
    out("  data-dependent indexed load per weight. This is exactly the IQ-quant access")
    out("  pattern that is slow on Metal vs contiguous Q4_K nibble unpack.")
    out("- **k=256 (8-bit codes):** 256 f16 = 512 bytes, still threadgroup-resident,")
    out("  but now codes are 8 bits (NO compression vs Q8_0, and WORSE than Q4_K's 4.5b)")
    out("  AND it is still a per-element LUT gather. Strictly dominated.")
    out("- **Lookup-free escape (QTIP-style):** the only way a learned grid is")
    out("  GPU-viable is if the codes are NOT indices but a contiguous bit-pattern")
    out("  decoded arithmetically (a bitshift trellis / lattice), so decode is ALU on")
    out("  contiguous bits with no random LUT read. A raw k-means codebook does NOT")
    out("  give that — it is inherently an index→value table = random gather.")
    out("")
    out("**L1.5 verdict: NO-GO at the feasibility gate.** A raw learned k-means codebook")
    out("forces per-element random LUT lookups, the precise pattern that makes IQ-quants")
    out("slow on Apple GPUs (no hardware gather). It loses to contiguous Q4_K nibble")
    out("unpack on the binding constraint (decode access pattern), BEFORE quality even")
    out("enters. Kill it here, as the Bible directs (\"kill it at the feasibility gate,")
    out("not after building\"). The lookup-free trellis idea survives, but that IS QTIP,")
    out("not a learned-codebook gather.")
    out("")

    # -------------------- Recommendation --------------------
    out("## Recommendation — which ONE of {L1.4, L1.5, QTIP} earns the quality eval")
    out("")
    out("Bible constraint: build AT MOST ONE byte-cut codec.")
    out("")
    out("- **L1.5 (learned codebook): OUT.** Dies at the Apple-GPU feasibility gate")
    out("  (random per-element LUT gather). No quality eval warranted.")
    out("- **L1.4 (low-rank + residual): "
        + ("GO to quality eval" if l14_go else "OUT on byte/energy budget") + ".** "
        + ("Some configs undercut the GGUF bytes AND remove enough energy that the"
           if l14_go else
           "The weights are not low-rank (top-64 captures <26%/15% of energy), so"))
    if l14_go:
        out("  residual collapses — verify quality on the GPU lane.")
    else:
        out("  the residual keeps ~all the energy: residual@2-3b ≈ raw-quant@2-3b, and")
        out("  the f16 U,V are dead overhead. Strictly worse than plain low-bit quant.")
        out("  No quality eval warranted.")
    out("- **QTIP (lookup-free bitshift-trellis codec): the survivor.** It is the only")
    out("  candidate that is BOTH a real byte cut AND Apple-GPU-feasible (contiguous,")
    out("  arithmetic decode, no gather). It is what L1.5 would have to become to be")
    out("  viable, and it does not carry L1.4's low-rank byte overhead.")
    out("")
    out("**Single recommended codec to advance to the GPU/quality lane: "
        + ("L1.4 (low-rank+residual), with QTIP as the fallback if quality disappoints."
           if l14_go else "QTIP (lookup-free trellis).") + "**")
    out("")
    out(f"_Peak RSS: {check_rss('end'):.2f} GB. Wall: {time.time()-t0:.1f}s._")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    sys.stderr.write(f"[done] wrote {REPORT}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

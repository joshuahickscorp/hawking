# Oracle — L1.4 (low-rank + residual) & L1.5 (learned codebook)

**Model:** `models/qwen2.5-3b-instruct-q4_k_m.gguf` (Qwen2.5-3B-Instruct)
**Date:** 2026-05-30  **Lane:** CPU NumPy byte-budget + Apple-GPU feasibility oracle

Scope: BYTE-BUDGET + DECODE-FEASIBILITY only. Quality (KL/perplexity vs
Q4_K_M) is DEFERRED to the GPU/llama.cpp lane and is NOT measured here.
Representative sample of tensors (not all 36 layers). SVD one tensor at a
time with `del`+`gc` between (RSS ceiling 3 GB).

## L1.4 — Low-rank + compressible residual

Per tensor: SVD `W = U S Vt`. Top-r kept as f16 (U_r, S·Vt_r). Residual =
`W - W_r`, stored at 2-3 bits/weight + an f16 per-row scale. Budget compared
to the tensor's ACTUAL on-disk GGUF bytes (Q4_K and Q6_K both appear).
`energy@r` = captured Frobenius energy = Σσᵢˆ 2(top r)/Σσᵢˆ 2. `res_std/W_std`
= residual std relative to original (proxy for residual quantizability).

| tensor | shape | disk type | disk bytes | r | energy@r | res_std/W_std | rank B (f16) | +res@2b | +res@3b | total@2b | total@3b | ratio@2b | ratio@3b |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blk.0.attn_q.weight | 2048x2048 | Q4_K | 2,359,296 | 16 | 0.155 | 0.919 | 131,072 | 1,052,672 | 1,576,960 | 1,183,744 | 1,708,032 | 0.50x | 0.72x |
| blk.0.attn_q.weight | 2048x2048 | Q4_K | 2,359,296 | 32 | 0.193 | 0.898 | 262,144 | 1,052,672 | 1,576,960 | 1,314,816 | 1,839,104 | 0.56x | 0.78x |
| blk.0.attn_q.weight | 2048x2048 | Q4_K | 2,359,296 | 64 | 0.257 | 0.862 | 524,288 | 1,052,672 | 1,576,960 | 1,576,960 | 2,101,248 | 0.67x | 0.89x |
| blk.0.ffn_gate.weight | 11008x2048 | Q4_K | 12,681,216 | 16 | 0.058 | 0.971 | 417,792 | 5,658,112 | 8,476,160 | 6,075,904 | 8,893,952 | 0.48x | 0.70x |
| blk.0.ffn_gate.weight | 11008x2048 | Q4_K | 12,681,216 | 32 | 0.092 | 0.953 | 835,584 | 5,658,112 | 8,476,160 | 6,493,696 | 9,311,744 | 0.51x | 0.73x |
| blk.0.ffn_gate.weight | 11008x2048 | Q4_K | 12,681,216 | 64 | 0.148 | 0.923 | 1,671,168 | 5,658,112 | 8,476,160 | 7,329,280 | 10,147,328 | 0.58x | 0.80x |
| blk.0.ffn_down.weight | 2048x11008 | Q6_K | 18,493,440 | 16 | 0.030 | 0.985 | 417,792 | 5,640,192 | 8,458,240 | 6,057,984 | 8,876,032 | 0.33x | 0.48x |
| blk.0.ffn_down.weight | 2048x11008 | Q6_K | 18,493,440 | 32 | 0.052 | 0.974 | 835,584 | 5,640,192 | 8,458,240 | 6,475,776 | 9,293,824 | 0.35x | 0.50x |
| blk.0.ffn_down.weight | 2048x11008 | Q6_K | 18,493,440 | 64 | 0.089 | 0.955 | 1,671,168 | 5,640,192 | 8,458,240 | 7,311,360 | 10,129,408 | 0.40x | 0.55x |
| blk.17.attn_output.weight | 2048x2048 | Q4_K | 2,359,296 | 16 | 0.087 | 0.956 | 131,072 | 1,052,672 | 1,576,960 | 1,183,744 | 1,708,032 | 0.50x | 0.72x |
| blk.17.attn_output.weight | 2048x2048 | Q4_K | 2,359,296 | 32 | 0.145 | 0.925 | 262,144 | 1,052,672 | 1,576,960 | 1,314,816 | 1,839,104 | 0.56x | 0.78x |
| blk.17.attn_output.weight | 2048x2048 | Q4_K | 2,359,296 | 64 | 0.234 | 0.875 | 524,288 | 1,052,672 | 1,576,960 | 1,576,960 | 2,101,248 | 0.67x | 0.89x |
| blk.17.ffn_up.weight | 11008x2048 | Q4_K | 12,681,216 | 16 | 0.029 | 0.986 | 417,792 | 5,658,112 | 8,476,160 | 6,075,904 | 8,893,952 | 0.48x | 0.70x |
| blk.17.ffn_up.weight | 11008x2048 | Q4_K | 12,681,216 | 32 | 0.051 | 0.974 | 835,584 | 5,658,112 | 8,476,160 | 6,493,696 | 9,311,744 | 0.51x | 0.73x |
| blk.17.ffn_up.weight | 11008x2048 | Q4_K | 12,681,216 | 64 | 0.090 | 0.954 | 1,671,168 | 5,658,112 | 8,476,160 | 7,329,280 | 10,147,328 | 0.58x | 0.80x |
| blk.35.attn_q.weight | 2048x2048 | Q4_K | 2,359,296 | 16 | 0.099 | 0.949 | 131,072 | 1,052,672 | 1,576,960 | 1,183,744 | 1,708,032 | 0.50x | 0.72x |
| blk.35.attn_q.weight | 2048x2048 | Q4_K | 2,359,296 | 32 | 0.149 | 0.923 | 262,144 | 1,052,672 | 1,576,960 | 1,314,816 | 1,839,104 | 0.56x | 0.78x |
| blk.35.attn_q.weight | 2048x2048 | Q4_K | 2,359,296 | 64 | 0.224 | 0.881 | 524,288 | 1,052,672 | 1,576,960 | 1,576,960 | 2,101,248 | 0.67x | 0.89x |
| blk.35.ffn_down.weight | 2048x11008 | Q6_K | 18,493,440 | 16 | 0.027 | 0.986 | 417,792 | 5,640,192 | 8,458,240 | 6,057,984 | 8,876,032 | 0.33x | 0.48x |
| blk.35.ffn_down.weight | 2048x11008 | Q6_K | 18,493,440 | 32 | 0.046 | 0.977 | 835,584 | 5,640,192 | 8,458,240 | 6,475,776 | 9,293,824 | 0.35x | 0.50x |
| blk.35.ffn_down.weight | 2048x11008 | Q6_K | 18,493,440 | 64 | 0.080 | 0.959 | 1,671,168 | 5,640,192 | 8,458,240 | 7,311,360 | 10,129,408 | 0.40x | 0.55x |

**Byte-budget summary (read carefully):** the raw total/disk ratios above
dip to 0.33x (2-bit residual) / 0.48x (3-bit residual) — but that is
**not a win**, because the residual is simply stored at fewer bits than
Q4_K's 4.5b. The decisive number is the low-rank energy: top-r captures at
most **25.7%** of Frobenius energy (r=64), so the residual keeps
**~95%** of the original std (`res_std/W_std` median 0.95).

**L1.4 verdict: NO-GO (byte budget).** These weights are **not low-rank**.
Even r=64 captures <26% of energy on the 2048-square attention matrices and
<15% on the 11008-wide FFN matrices; the residual retains ~90-99% of the
weight std. Consequences:

- The residual at 2-3 bits is numerically ~identical to quantizing the RAW
  matrix at 2-3 bits (no structure was removed), so the low-rank part buys
  no quality — yet it ADDS `2*(m+n)*r*2` f16 bytes of pure overhead.
- Therefore low-rank+residual is strictly WORSE than plain N-bit quant: same
  residual error, extra U,V bytes. The apparent <1.0 ratio is just "use
  fewer bits than Q4_K" wearing a low-rank costume.
- Low-rank only pays when top-r captures most of the energy so the residual
  collapses toward ~0 bits. It does not here. Lever dies on the byte/energy
  oracle — no quality eval warranted.

## L1.5 — Learned per-model codebook (the danger lever)

Fit tensor: `blk.17.ffn_up.weight` (11008x2048, on-disk Q4_K).
k-means on the scalar weight distribution (1-D, RAM-safe). Reconstruction MSE
vs the FIXED llama.cpp grid at MATCHED bits. k=16 ↔ 4-bit grid (Q4_0);
k=256 ↔ 8-bit grid (Q8_0). Lower MSE = better quality-per-bit.

| bits | learned k-means MSE | fixed-grid MSE | grid | learned/fixed |
|---|---|---|---|---|
| 4-bit (k=16) | 9.652e-06 | 5.152e-06 | Q4_0 | 1.87x |
| 8-bit (k=256) | 6.821e-07 | 2.081e-08 | Q8_0 | 32.78x |
| (ref) weight variance | 6.999e-04 | | | |

Note: a 1-D global codebook is a LOWER bound on learned-codebook quality; the
fixed grids (Q4_0/Q8_0) use per-block f16 scales (32-wide), so they adapt to
local magnitude. A fair learned codec would also need per-block scales — the
codebook alone does not capture that. MSE here is the codebook-vs-grid shape
comparison only; absolute quality is the GPU lane's call.

### Binding feasibility verdict (Apple-GPU decode)

- **k=16 (4-bit codes):** a 16-entry codebook is 16 f16 = 32 bytes. That
  trivially fits in threadgroup memory. BUT decode is `w[i] = codebook[code[i]]`
  — a per-element indexed read into the codebook. On Apple GPUs there is no
  hardware gather; even a threadgroup-resident 16-entry LUT becomes a
  data-dependent indexed load per weight. This is exactly the IQ-quant access
  pattern that is slow on Metal vs contiguous Q4_K nibble unpack.
- **k=256 (8-bit codes):** 256 f16 = 512 bytes, still threadgroup-resident,
  but now codes are 8 bits (NO compression vs Q8_0, and WORSE than Q4_K's 4.5b)
  AND it is still a per-element LUT gather. Strictly dominated.
- **Lookup-free escape (QTIP-style):** the only way a learned grid is
  GPU-viable is if the codes are NOT indices but a contiguous bit-pattern
  decoded arithmetically (a bitshift trellis / lattice), so decode is ALU on
  contiguous bits with no random LUT read. A raw k-means codebook does NOT
  give that — it is inherently an index→value table = random gather.

**L1.5 verdict: NO-GO at the feasibility gate.** A raw learned k-means codebook
forces per-element random LUT lookups, the precise pattern that makes IQ-quants
slow on Apple GPUs (no hardware gather). It loses to contiguous Q4_K nibble
unpack on the binding constraint (decode access pattern), BEFORE quality even
enters. Kill it here, as the Bible directs ("kill it at the feasibility gate,
not after building"). The lookup-free trellis idea survives, but that IS QTIP,
not a learned-codebook gather.

## Recommendation — which ONE of {L1.4, L1.5, QTIP} earns the quality eval

Bible constraint: build AT MOST ONE byte-cut codec.

- **L1.5 (learned codebook): OUT.** Dies at the Apple-GPU feasibility gate
  (random per-element LUT gather). No quality eval warranted.
- **L1.4 (low-rank + residual): OUT on byte/energy budget.** The weights are not low-rank (top-64 captures <26%/15% of energy), so
  the residual keeps ~all the energy: residual@2-3b ≈ raw-quant@2-3b, and
  the f16 U,V are dead overhead. Strictly worse than plain low-bit quant.
  No quality eval warranted.
- **QTIP (lookup-free bitshift-trellis codec): the survivor.** It is the only
  candidate that is BOTH a real byte cut AND Apple-GPU-feasible (contiguous,
  arithmetic decode, no gather). It is what L1.5 would have to become to be
  viable, and it does not carry L1.4's low-rank byte overhead.

**Single recommended codec to advance to the GPU/quality lane: QTIP (lookup-free trellis).**

_Peak RSS: 2.19 GB. Wall: 48.0s._

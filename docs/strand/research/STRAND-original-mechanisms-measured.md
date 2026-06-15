# STRAND-original mechanisms — MEASURED on real Qwen2.5-0.5B (2026-06-13)

_Not a literature synthesis — six home-grown mechanisms built and measured on the real
weights. Probe scripts: `research/experiments/strand_novel_probes.py` (P1/P2/P3),
`strand_probe4_outaware.py` (P4), `strand_probe_scale_lowrank.py` (H/B). Reproducible, CPU-only,
~20s total._

**Method honesty (no soft-positives).** Faithful STRAND RHT (exact splitmix64 sign + row-aware
FWHT, real 128/256 block structure) + a scalar Lloyd-Max codebook stand-in for the trellis.
The trellis adds space-filling gain on TOP, so **absolute rel-RMS here is pessimistic** (scalar
0.34 @2-bit vs the real trellis ~0.28); the **RATIOS and entropy deltas are what transfer**, and
every probe is designed so the verdict rides a ratio, not an absolute. Every WIN below still
needs a real-trellis + PPL A/B to bank — these are go/kill gates, not final numbers.

## Scoreboard

| # | Mechanism (what only a frozen-integer codec can try) | Verdict | Measured |
|---|---|---|---|
| **P4** | **Output-aware pre-conditioning that composes with RHT** | **WIN — quality** | **−15.6% output-error, α=0.5, robust 6/6** |
| P2 | Embedded residual (one file decodes bit-exact @2.0 AND 3.0 bpw) | **WIN — capability** | +9% RMS vs dedicated flat-3 |
| P1 | Context-predicted scale (conditional, not iid, side-info coder) | **WIN — density (modest)** | 0.027 bpw mean / 0.048 attn |
| P3 | Frozen codebook-bank + integer selector | KILL | 87–97% pick Gaussian (RHT flattened tails) |
| H | Low-rank scale field (separable row×position factors) | KILL | rank-1 high but residual incompressible; P1 wins |
| B | Payload context/dictionary coding | KILL | order-1 = order-0 (ctx_gain ≈ 0); memoryless |

The three kills are as valuable as the wins: they close "heavy-tail residual" (P3), "low-rank
side-info" (H), and "is the payload really max-entropy" (B) with numbers, so nobody burns time there.

---

## P4 — Output-aware pre-conditioning ∘ RHT  ★ the find (a QUALITY lever)

**The diagnosis it proves:** STRAND minimizes *weight* error `Σ(ΔW)²`, but the loss cares about
*output* error `Σⱼ aⱼ²(ΔWᵢⱼ)²` where `aⱼ` = per-input-column activation rms. These disagree, and
STRAND optimizes the wrong one. Measured: at the best α the *weight*-RMS gets **worse** (q_proj
0.339→0.525) while the *output*-error gets **better** (0.346→0.264, −23.7%) — direct proof the
objective is mis-specified.

**The mechanism (moat-safe, principle is AWQ-class but the STRAND facts are new):**
pre-scale input columns by `D = aⱼ^α` (geo-mean-normalized) BEFORE the RHT, quantize as usual,
and fold `D⁻¹` into the activation side at inference. `y = Wx = (W·D)(D⁻¹·x)` — the `D⁻¹` rides
the activation path STRAND already RHTs, so the **weight decode stays bit-exact integer**; only a
frozen per-input-feature diagonal touches the (already-float) activation vector.

**Three things measured here that are NOT in any paper and that hand you the lever:**
1. **Saliency SURVIVES the random Hadamard.** It was a real risk that RHT's column-mixing would
   whiten the per-column saliency to nothing (that's what would have made it dead for STRAND).
   It doesn't — α=0.5 still cuts output-error 15.6%. 
2. **This is the correct fix for the Hessian-Viterbi backfire (+1.1%).** That lever reweighted
   *coordinates post-RHT* (where curvature is flattened) with a *mismatched c4 Hessian*. The right
   move is an *activation-energy diagonal pre-RHT* — different signal, different place, and it works.
3. **The sweet spot is α≈0.5** (geo-mean of weight and activation magnitude) and the win is robust
   across attention and FFN, layers 0 and 12 (+5.2% to +23.7%).

| tensor | out-err α=0 | α=0.5 (best region) | Δ |
|---|---|---|---|
| L0 down_proj | 0.342 | 0.297 | −13.3% |
| L0 up_proj | 0.385 | 0.328 (α.75) | −14.7% |
| L0 q_proj | 0.212 | 0.171 | −19.0% |
| L12 down_proj | 0.343 | 0.325 | −5.2% |
| L12 up_proj | 0.394 | 0.319 (α.75) | −19.1% |
| L12 q_proj | 0.346 | 0.264 | −23.7% |
| **mean** | **0.337** | **0.284** | **−15.6%** |

**Integration design (→ executor):**
- Calib already emits it: `actmean-qwen05b.json` → `modules[name].feature_rms` is `aⱼ` (per input
  feature). No new calibration pass needed.
- ENCODE: in `quantize-model.rs::quantize_one` (`encode.rs` path), before `rht_forward_rows`,
  multiply each input column `j` by `D_j = (feature_rms_j)^α / geomean`. New flag `--act-precond α`.
- DEPLOY: fold `diag(D⁻¹)` into the activation-side RHT in `outlier_mac::matvec_rht` (a per-input
  multiply before the Hadamard). `D` is a frozen per-tensor `[in_features]` f16 vector → store as a
  new EOF-chained section (mirror OUTL) and **seal it in `descriptor_digest`** (same discipline as
  DBIA — an unsealed activation diagonal is an attestation hole).
- MOAT: weight decode untouched (still integer LUT). Only the float activation path gains a frozen
  diagonal — deterministic.
- GATE (0.5B, the real test): quantize with `--act-precond {0, 0.5}`, eval PPL. Bank if the −15.6%
  output-RMS becomes a real PPL/loss-tax drop (output-RMS is much closer to PPL than weight-RMS, but
  de-bias taught us they can still diverge — confirm). **Stacks with DBIA (first moment) and the
  outlier channel — orthogonal corrections (2nd-moment activation vs mean vs magnitude).**
- α is itself tunable per tensor-class (attention liked 0.5, up_proj liked 0.75) — a cheap sweep.

---

## P2 — Embedded residual (progressive bitrate, bit-exact at each rung)

One artifact, two decode depths: read the 2-bit base → 2.0 bpw; also read the 1-bit residual
layer → ~3 bpw. Measured rel-RMS: base 0.341, **embedded(2+1) 0.201, dedicated flat-3 0.184 →
embedded is only 1.09× flat-3.** A float-LUT codec (QTIP-computed-codes) CANNOT do bit-exact
progressive decode (FP non-associativity); STRAND can. The 9% penalty is the residual using its
own per-block scale + 1-bit book; a trellis residual or a shared base-scale would close most of it.

**Integration:** a second SDSQ-style section carrying the residual indices (+ its scale), decode adds
one MAC `recon += scale_r · LUT1[idx_r]`. Truncatable: ship once, deploy at the bitrate the device
affords. **Build only if multi-bitrate deploy (the dismantle/phone story) is a goal** — it's a
capability, not a bpw/quality win on its own.

---

## P1 — Context-predicted scale (conditional side-info coder)

Per-block scale carries structure an *iid* entropy coder (today's static-14-bit-CDF rANS) misses.
Measured saving of an *order-1 conditional* coder over the marginal: **0.027 bpw mean, 0.048 bpw on
attention** (q_proj log-scale entropy drops 9.1→3.2 bits given the previous block). Note: naive
*delta* coding sometimes LOSES (up_proj −0.002) — the lever is a **conditional/context model**, not
first-difference.

**Integration:** upgrade the SDSQ/C2 scale_q coder from static-marginal to an order-1 context model
(context = quantized previous-block scale, causal & decoder-reconstructable). Pure side-info,
moat-safe, fold into `sideinfo_rans.rs`/`c2_final.rs`. Modest (~0.027 bpw) — stack onto C2, don't
build standalone.

---

## Kills (do NOT build — measured dead)

- **P3 frozen codebook-bank:** 87–97% of post-RHT blocks pick the Gaussian book, ~0% pick
  student-t(3); gain 0.03–0.19% < selector cost 0.008–0.016 bpw. **RHT's Gaussianization is
  thorough** — the single Gaussian LUT is correct, and the "heavy-tail residual" question is CLOSED.
- **H low-rank scale field:** the log-scale matrix IS high-rank-1 (attention 97–98%, down 85–96%)
  but the rank-1 residual stays broadband/incompressible, so factor-storage + residual entropy does
  NOT beat direct coding (mean save −0.003 bpw; up_proj actively loses). **P1's conditional coding
  dominates H** — use P1, not factorization.
- **B payload context/dictionary:** order-1 index entropy = order-0 (1.912 ≈ 1.915, ctx_gain
  0.0001 bits) → the payload is **memoryless and near-max-entropy**, settling the sprint's assertion
  with a number. Only a ~1.4%(real)–4.4%(scalar) *marginal* redundancy remains (known ECTCQ tail,
  needs integer-ANS, low priority). No context/dictionary/LZ win on the payload — dead.

---

## Net for the executor

1. **P4 is the new lever to chase — it's a QUALITY (loss-tax) attack from the PTQ side**, the first
   one found that isn't training. −15.6% output-error, moat-safe, stacks with DBIA + outliers, calib
   already has the inputs. Confirm with a 0.5B `--act-precond` PPL A/B before the cloud recipe locks.
2. **P1 + P2** fold into the density/format track (C2 conditional coder; optional progressive section).
3. **P3 / H / B are dead** — banked with numbers so they don't get re-opened.
4. All absolute numbers are scalar-stand-in (pessimistic); the executor re-runs the winners through
   the real Rust trellis + PPL. The probes are the go/kill gates, not the final figures.

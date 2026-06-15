# Oracle — imatrix MIXED-PRECISION byte-cut (axis-2; first cut)

**Model:** `models/qwen2.5-3b-instruct-q4_k_m.gguf`  **Lane:** CPU NumPy  **Mix:** Q4_K + Q3_K  **Budget:** <= 0.85x uniform-Q4_K bytes (median 15% byte-cut, ~3.82 eff bits/weight)
**Date:** 2026-05-31

> **Scope = WEIGHT-ONLY first-cut PROXY.** The lever is `imatrix mixed-precision` (bible axis-2, listed +12-20% / no-new-kernel, previously with no oracle). The DECISIVE verdict needs the REAL importance matrix — an ACTIVATION statistic from a forward pass over a calibration corpus on **f16** weights — plus a KL/logit gate. The f16 Qwen and a forward pass are **not on this machine** (only the Q4_K_M GGUF), and fitting a low-bit grid from already-Q4 weights is a recorded kill (imatrix-Q3-from-Q4 = +32% PPL / -18% bytes, `reports/dead_levers.md`). So this is a first cut, **not** the gate, and records **NO kill** (a weight-only proxy cannot legitimately kill an activation-driven method — that would be a Type-2 error).

> **What is exact vs bracketed.** The mixed-precision ASSIGNMENT and the K-quant round-trips (Q4_K/Q3_K/Q2_K) are faithful NumPy reimplementations (gguf has no K-quant quantizer) — exact, no approximation. The IMPORTANCE RANKING is what we cannot see without activations, so we report the mixed RMSE under THREE bracket rankings:
>  * **ORACLE** — rank rows by their own Q4-vs-low RMSE demotion penalty (a greedy per-row heuristic, ~RMSE-optimal — occasionally edged by weight-norm by <0.1% via kept/demoted interaction). An optimistic near-lower-bound; no causal importance signal gets the penalty for free.
>  * **WEIGHT-NORM** — rank by per-output-channel L2 norm. The realistic in-session weight surrogate.
>  * **RANDOM** — no importance signal; an upper bound on the error.
> The real activation-imatrix assignment sits inside this interval for the metric that tracks activations; for the *RMSE* metric it cannot beat ORACLE.

## (i) Mixed-precision RMSE vs uniform Q4_K_M at a 15% byte-cut

Keep the top-importance output channels at Q4_K, demote the rest to Q3_K, sized so total bytes <= 0.85x uniform Q4_K_M (a 15% cut). Reconstruction rel-RMSE of the dequantized real tensor (lower = closer to uniform Q4_K). `beats` = that ranking's mixed RMSE <= the uniform-Q4_K (no-cut) RMSE — a tautological loss for any byte-cut; the meaningful comparison is the intra-budget one just below.

| tensor | shape | disk | cut% | keep%@Q4 | uniform Q4_K | mix ORACLE | mix WNORM | mix RANDOM | WN beats | OR beats |
|---|---|---|---|---|---|---|---|---|---|---|
| blk.0.attn_q.weight | 2048x2048 | Q4_K | 15% | 36% | 0.0444 | 0.1049 | 0.1048 | 0.1162 | no | no |
| blk.0.ffn_gate.weight | 11008x2048 | Q4_K | 15% | 36% | 0.0443 | 0.1107 | 0.1108 | 0.1181 | no | no |
| blk.0.ffn_down.weight | 2048x11008 | Q6_K | 15% | 36% | 0.0789 | 0.1282 | 0.1283 | 0.1328 | no | no |
| blk.17.attn_output.weight | 2048x2048 | Q4_K | 15% | 36% | 0.0458 | 0.1080 | 0.1081 | 0.1176 | no | no |
| blk.17.ffn_up.weight | 11008x2048 | Q4_K | 15% | 36% | 0.0431 | 0.1115 | 0.1116 | 0.1167 | no | no |
| blk.35.attn_q.weight | 2048x2048 | Q4_K | 15% | 36% | 0.0441 | 0.1073 | 0.1075 | 0.1174 | no | no |
| blk.35.ffn_down.weight | 2048x11008 | Q6_K | 15% | 36% | 0.0808 | 0.1328 | 0.1329 | 0.1358 | no | no |

**Median:** uniform Q4_K 0.0444 vs mixed [ORACLE 0.1107, WNORM 0.1108, RANDOM 0.1176] at 15% byte-cut (~3.82 bits). WEIGHT-NORM ranking matches-or-beats uniform on **0/7** tensors; ORACLE (RMSE lower bound) on **0/7**.

### What the imatrix actually buys (the intra-budget comparison)

The lever's real claim is **importance-guided mixing beats UNIFORM demotion to the SAME byte budget** — NOT that a byte-cut beats the no-cut Q4_K_M (that is tautologically impossible: cut bytes -> RMSE rises). So the decisive in-session number is mixed-with-importance (ORACLE / WEIGHT-NORM) vs RANDOM (= uniform demotion to budget):

- **ORACLE vs RANDOM:** −5.9% RMSE — the *most* any importance signal can recover for RMSE at this budget.
- **WEIGHT-NORM vs RANDOM:** −5.8% RMSE — what the realistic weight surrogate recovers. It captures ~99% of the oracle's gain (rank corr below confirms weight-norm ≈ the RMSE-optimal ranking).

So at the RMSE metric the importance ranking buys only a SINGLE-DIGIT-% edge over no ranking — the byte-cut's cost is dominated by the steep Q4->Q3_K grid penalty (uniform 0.0444 -> ~0.1176), not by *which* channels are demoted. This is the proxy's central honest result: on weight-RMSE the mixed-precision lever is mostly a quantizer-rate story, and importance is a small correction. **Whether the real ACTIVATION imatrix buys more — on LOGITS, the metric that matters — is exactly what weight-RMSE cannot see and the Colab gate must measure.**

## (ii) How good is the weight-only ranking? (rank correlation)

Rank-correlation of the realistic WEIGHT-NORM ranking with the RMSE-ORACLE ranking per tensor. High corr -> weight-norm already captures most of the *RMSE*-relevant importance; the gap to GO is then mostly whether the real ACTIVATION imatrix adds a logit-relevant signal weight-norm is blind to.

| tensor | weight-norm vs oracle rank corr |
|---|---|
| blk.0.attn_q.weight | 0.98 |
| blk.0.ffn_gate.weight | 0.93 |
| blk.0.ffn_down.weight | 0.99 |
| blk.17.attn_output.weight | 0.98 |
| blk.17.ffn_up.weight | 0.91 |
| blk.35.attn_q.weight | 0.97 |
| blk.35.ffn_down.weight | 0.88 |

**Median rank corr:** 0.97.

## (iii) Direction-only activation-shape probe (NOT the real imatrix)

A diag(W^T W) per-input-column *weight*-energy weighting fed into the K-quant WLS refit (the same machinery the real imatrix uses, but with weight energy as a stand-in for the activation energy sum(x^2) it cannot see). Reports whether protecting high-weight-energy columns lowers the (weight-energy-)weighted RMSE — a sanity check that the imatrix MECHANISM is wired and helps *when* importance points at high-magnitude columns. The real lever weights by ACTIVATION energy, which can point at low-weight-norm columns, so this is a direction probe only.

- Importance-weighted-RMSE helped (weighted <= unweighted) on **4/7** tensors. (Expected: weighting by a column's own energy mostly tracks where the error already is, so the WLS gain on the *weighted* metric is modest and occasionally negative — exactly why the decisive signal must come from *activation* energy, not weight energy.)

## Direction read (NOT the verdict)

- **Cautionary direction.** Even the RMSE-ORACLE ranking (the best any signal can do for RMSE) trails uniform Q4_K on 7/7 tensors at this byte budget — demoting to Q3_K costs more weight-RMSE than the byte budget buys back, regardless of ranking. The real imatrix could still win on LOGITS (RMSE is not its objective), so this is NEEDS-MEASUREMENT, not a kill — but the Colab gate must show a real logit/KL margin.
- **Why weight-only over/under-credits the real activation imatrix:** (1) OVER — the ORACLE ranking peeks at the actual demotion RMSE, which no causal signal gets, so it is optimistic for any real assignment on RMSE; (2) UNDER — RMSE is not the model's objective: the real imatrix protects columns that move LOGITS, which can be low-weight-norm but high-activation (a near-constant feature a later layer leans on), and weight-norm is blind to that, so the real imatrix can BEAT every ranking here on the decisive logit/KL metric; (3) STRUCTURAL — the lever's value is iso-LOGIT-quality, not iso-RMSE, and a weight-RMSE proxy can only show byte-feasibility and rank-sensibility; (4) the Q3_K grid here keeps an affine min (slightly FAVOURING the low-bit leg vs ggml's symmetric Q3_K — conservative for a byte-cut that must beat it).

## The DECISIVE (Colab) gate — `--colab` runbook

On Colab, with f16 Qwen2.5-3B + a code calibration corpus:
1. Run llama.cpp `llama-imatrix` over the corpus on the **f16** model to produce a real importance matrix (per-input-column sum(x^2)). Export per-tensor f16 weights + the imatrix vector -> `weights.npz` / `imat.npz`.
2. Build the mixed-precision GGUF the real way: `llama-quantize --imatrix imat.dat model-f16.gguf model-mix.gguf <type>` with a tensor-type override that keeps high-importance tensors at Q4_K and demotes low-importance ones to {Q3_K,Q2_K}; confirm total bytes <= the uniform Q4_K_M GGUF. (No new kernel — every tensor is a standard ggml K-quant.)
3. **Recon gate:** mixed vs uniform-Q4_K_M rel-RMSE **vs f16** (GO floor: mixed <= uniform at fewer bytes). Fit the low-bit grids FROM f16, never from Q4 (kill-respect).
4. **Functional gate (decisive):** forward-pass f16 / uniform-Q4_K_M / mixed on held-out **code**; export next-token logits; check logit-cosine / KL / argmax-agreement (GO: mixed >= uniform cosine & argmax, <= KL — i.e. the byte-cut is free on quality). The bible's +12-20% is a THROUGHPUT claim (fewer bytes -> faster decode GEMV); confirm with a paired decode bench once the quality gate is GO.
5. GO on recon AND logits -> wire the mix in the loader (byte accounting only; no kernel) + paired decode bench for the +12-20%. NO-GO on the logit gate with a fair real imatrix -> THEN a kill is legitimate (records a dead_levers entry, classified Type-1/2 per protocol).

_Wall: 26.7s. Peak RSS: 0.81 GB. Run `--selftest` (must pass) for these numbers to be trustworthy._

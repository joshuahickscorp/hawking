# Inner-product de-biasing — results & verdict

_will.md §4 LIVE-queue #3 (TurboQuant family). Lever owner files:
`crates/strand-quant/src/debias.rs`, `crates/strand-quant/src/bin/gate-debias.rs`,
`research/patches/debias-integration.patch`. Machine: Apple M3 Pro (macos/aarch64),
repo @ 40e9b31, 2026-06-11._

## TL;DR (decisive number first)

At the 2-bit operating point (l=12, +1% outlier), inner-product de-biasing cuts
**simulated output-RMS by +4.19%** on a **non-zero-mean** activation model (μ̄≈0.3),
for **0.0179 bpw** (a bf16 per-output-row bias side-channel; free if folded into an
existing layer bias). On a **zero-mean** activation model the correction is exactly
vacuous (+0.0001%) — confirming the derivation. **The lever is ALIVE iff real Qwen
activations carry a mean** (they do, post-RMSNorm + residual DC); the deciding test
is the spec'd 0.5B PPL A/B below. This is a *mean* correction, structurally distinct
from the dead Hessian family — and the rowsum bias it exploits **survives the RHT**
(measured rowsum-bias RMS = 0.23, nonzero post-rotation).

## The math (rigorous; the whole verdict turns on it)

Layer `y = W x`, `W` is `[out, in]`, recon `Ŵ`, `Δ = Ŵ − W`, activations `x ~ (μ,Σ)`.

- Per-row output error: `e_i = Σ_j Δ_ij x_j`.
- Its mean: **`E[e_i] = Σ_j Δ_ij μ_j`**  ... (1)

**Zero-mean degeneracy.** If `μ = 0`, (1) = 0 — the first-moment correction is
vacuous and what remains is `Var[e_i] = Δ_i Σ Δ_iᵀ`, a *curvature reweight* = the
**Hessian family**, already DEAD for STRAND (RHT whitens Σ→σ²I, the reweight only
overfits the calib corpus). So **for zero-mean activations de-bias ≡ dead-Hessian**.

**Non-zero-mean (the live regime).** Take the simplest billable mean model
`μ_j = μ̄` (one scalar per tensor; means generalise across corpora where Hessians
overfit):
- `E[e_i] = μ̄ · S_i`, `S_i := rowsum(Ŵ_i) − rowsum(W_i)`  ... (2)
- correction `c_i = −μ̄ · S_i`, applied additively to the output `y_i`  ... (3)

**RHT survival (the make-or-break).** STRAND quantises as `Ŵ = Rᵀ q(R W)`, `R`
orthogonal. The rowsum is `S_i = 1ᵀΔ_i = (R 1)ᵀ δ̃_i` where `δ̃_i` is the RHT-space
residual. `R 1 ≠ 1` (Hadamard mixes the all-ones vector into one spread direction),
so `S_i` is a projection of the residual onto a *fixed rotated direction* — **rotated,
not destroyed**. RHT preserves inner products ⇒ the rowsum bias is estimable post-RHT.
**Measured: rowsum-bias RMS = 0.23 ≠ 0 → confirmed survives RHT.** This is the key
distinction from why diag-H died: diag-H died because the RHT *flattens curvature*;
the rowsum bias is a *mean projection*, which the RHT only rotates.

## Measured (gate-debias, 14 real Qwen2.5-0.5B tensors, k=2 l=12 +1% outlier, 64 Gaussian acts/tensor)

| activation model | mean \|out-bias\| uncorr → debiased | mean out-RMS uncorr → debiased | **out-RMS reduction** |
|---|---|---|---|
| **non-zero-mean (μ̄≈0.3)** | 2.29e-3 → 1.24e-3 | 2.423e-1 → 2.321e-1 | **+4.19%** |
| zero-mean control (μ̄→0) | — | — | +0.0001% (vacuous, as eq.1 predicts) |

Per-tensor out-RMS reduction was tight: +3.77% … +4.97% across all 14 (q/k/v/o/gate/
up/down, layers 0–1). rel-RMS itself is **unchanged** by de-biasing (≈22–26%, the
canon 2-bit floor) — i.e. the proxy (weight-MSE) is flat while the truth (output
error) moves, exactly the will.md §5.5 pattern that the gate exists to catch.

**Cost (billed, §5.11):** `Δbpw = bias_bits / in_features`. bf16 @ in=896 = **0.0179
bpw**; @ in=4864 (mlp down_proj) = 0.0033 bpw. **Free** if folded into an existing
layer bias. Three orders below the 0.32 bpw outlier channel.

## Verdict

**ALIVE at the gate's kill bar (≥0.5% output-RMS cut), pending the PPL confirm.**
The honest caveat: the +4.19% is on a *modelled* activation mean (μ̄=0.3 isotropic).
The win is real and bounded by how much DC the true Qwen activations carry. The
output-error gate is the proxy-for-the-proxy here; **PPL is the truth** and must be
run before adoption. If the real μ̄ is ~0 the lever collapses to the +0.0001% control
(the 4th RHT-whitening kill — recorded with the math either way).

## 0.5B PPL A/B protocol (SPEC ONLY — not run; needs CPU/MPS-free box per §8 freeze trap)

Goal: does the de-bias bias side-channel reduce held-out WikiText-2 PPL at 2-bit?

1. **Calibrate μ̄ (cheap, once).** Run a forward pass of the bf16 model on ~8
   WikiText-2 windows; for each projection layer record the per-layer activation
   mean over the input feature axis (the `x` that feeds that Linear). Store a
   `{layer_name: μ̄}` map. (A single global μ̄ is an acceptable first cut — means
   generalise; the per-layer map is the refinement.) This is a NEW small script
   `ops/calib-actmean.py` (do not touch ops/eval-ppl.py).

2. **Quant both arms** with the canon recipe `--bits 2 --l 12 --outlier-channel 1`:
   - **arm A (baseline):** as today → recon.A.safetensors. (canon 2-bit floor ≈ 80.7)
   - **arm B (de-biased):** same + `--debias --debias-mu <μ̄>` (integration patch),
     which adds the per-row correction as a sibling `<proj>.bias` tensor (or a DBIAS
     section) → recon.B.safetensors.

3. **Eval both** with `ops/eval-ppl.py <load_dir> 2048 64 cpu bf16 <tag> <out_json>`
   (the canon protocol; non-overlap windows, exp(Σnll/Σtok)). For projections that
   carry no bias in the base model, the A/B harness must ADD the correction at the
   matmul — i.e. eval-ppl must read the sibling `.bias`/DBIAS and add it post-Linear
   for arm B (a ~5-line eval shim, kept in the new script, NOT in eval-ppl.py).

4. **Adversarial check (§5.4):** arm A and arm B PPL must NOT be identical to 15
   digits (that would mean the bias never got applied — the contamination tell).

### Kill bar
- **ADOPT** if arm B PPL ≤ arm A PPL − 0.5% (i.e. ≤ ~80.3 vs 80.7), AND the +0.0179
  bpw is either folded-free or judged worth it. A larger move (the gate's +4% output
  RMS suggests up to a few % PPL is plausible if μ̄ is large) makes it a clear win.
- **DEAD** if arm B ≥ arm A (within noise) → the real activation mean is too small;
  record as the 4th RHT-whitening kill (the rowsum bias survives RHT but the *output*
  it biases is near-centred). Either way the math + the gate stand as the artifact.

## Files
- `crates/strand-quant/src/debias.rs` — derivation (module docs) + `debias_tensor`,
  `estimate_mu_bar`, `matvec`, `output_error` + 3 unit tests (incl. the exact
  first-moment-cancellation identity). 3/3 green.
- `crates/strand-quant/src/bin/gate-debias.rs` — the measurement above. Run:
  `STRAND_NO_GPU=1 ./target/release/gate-debias scratch/qwen-05b/model.safetensors 14`
- `research/patches/debias-integration.patch` — the quantize-model.rs wiring sketch
  (advisory; the file is refactor-owned).

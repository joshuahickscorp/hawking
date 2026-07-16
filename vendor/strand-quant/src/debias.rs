//! Inner-product de-biasing for STRAND recon (will.md §4 queue #3 / TurboQuant family).
//!
//! # The lever
//! MSE-optimal quantizers minimise `E[(W - Ŵ)^2]` but do NOT preserve the matmul:
//! the quantity the network actually consumes is the inner product `y_i = <W_i, x>`,
//! and a per-row reconstruction error `Δ_i = Ŵ_i - W_i` biases it by
//! `e_i = <Δ_i, x>`. This module estimates and removes the *mean* of that bias.
//!
//! # The derivation (be rigorous — record the math, win or dead)
//! For a linear layer `y = W x`, `W` is `[out, in]`, activations `x ~ (μ, Σ)`.
//! Row `i` output error:
//!     e_i = Σ_j Δ_ij x_j .
//! Its expectation over the activation model:
//!     E[e_i] = Σ_j Δ_ij μ_j .                                            (1)
//!
//! ## Zero-mean case (the honest crux)
//! If `μ = 0`, (1) is **exactly zero** — the first-moment correction is vacuous.
//! What is left is the *second* moment, `Var[e_i] = Δ_i Σ Δ_iᵀ`, which is a
//! curvature/covariance reweight — i.e. the **Hessian family**, already DEAD for
//! STRAND (will.md §4: the RHT whitens, so Σ ≈ σ²I in RHT space and the reweight
//! only overfits the calib corpus). So: **for zero-mean activations the de-bias
//! lever degenerates to the dead Hessian lever.** This is the 4th RHT-whitening
//! kill and is recorded as such — UNLESS activations carry a mean.
//!
//! ## Non-zero-mean case (where the lever can live)
//! LLM activations are NOT zero-mean: RMSNorm rescales but does not centre, and
//! the residual stream carries a persistent DC component. Take the simplest
//! *billable* mean model, an isotropic scalar `μ_j = μ̄` (one number per tensor;
//! generalises across corpora far better than a full per-channel Hessian):
//!     E[e_i] = μ̄ · Σ_j Δ_ij = μ̄ · S_i ,   S_i := rowsum(Ŵ_i) - rowsum(W_i).  (2)
//! The unbiasing correction is therefore a **per-output-row additive constant**
//!     c_i = - μ̄ · S_i                                                       (3)
//! applied to the layer output `y_i`. KEY: this is a *mean* correction (depends on
//! the rowsum bias `S_i` and the activation mean `μ̄`), NOT a curvature reweight —
//! it is the structurally-distinct cousin of the dead Hessian family.
//!
//! ## Does it survive the RHT? (the make-or-break question)
//! STRAND quantises in RHT space: `Ŵ = Rᵀ q(R W)` where `R` is the per-row signed
//! Walsh-Hadamard rotation (orthogonal, `RᵀR = I`). The rowsum is a single inner
//! product with the all-ones vector `1`:
//!     S_i = 1ᵀ Δ_i = 1ᵀ Rᵀ (q(RW)_i - (RW)_i) = (R 1)ᵀ δ̃_i ,
//! where `δ̃_i` is the quantization residual in RHT space. `R 1` is NOT `1` (the
//! Hadamard mixes the all-ones vector into one spread direction), so `S_i` is a
//! projection of the RHT-space residual onto a fixed rotated direction. It is
//! generically **nonzero and estimable post-RHT** — the RHT preserves inner
//! products, so the rowsum bias is rotated, not destroyed. Whether it is *large
//! enough to matter for output error* is the empirical question `gate-debias`
//! answers (rel-RMS is the proxy; simulated `Wx` vs `Ŵx` error is the truth).
//!
//! # Billing (will.md §5.11 — bill everything)
//! The correction is a length-`out` vector. Folded into an existing layer bias it
//! is free; Qwen projections have NO bias, so it is a billed side-channel:
//!     Δbpw = (out · bias_bits) / (out · in) = bias_bits / in .
//! At in=896 (the 0.5B), bf16 bias = 16/896 = **0.0179 bpw** — three orders below
//! the 0.32 bpw outlier channel. Effectively free. (A per-row f32 fold into the
//! per-block scale is NOT available: the correction is additive in output space,
//! the scale is multiplicative in weight space — they do not compose. So the
//! honest cost is the tiny bias vector, not zero.)

/// Per-tensor de-bias result: the additive output correction `c` (length `out`)
/// and the diagnostics needed to decide adopt/dead.
#[derive(Clone, Debug)]
pub struct DebiasResult {
    /// `c_i = -mu_bar * S_i` — add to layer output `y_i` at inference (eq. 3).
    pub bias_correction: Vec<f32>,
    /// Per-row reconstruction rowsum bias `S_i` (eq. 2), independent of `mu_bar`.
    pub rowsum_bias: Vec<f32>,
    /// The scalar activation-mean model `mu_bar` used.
    pub mu_bar: f32,
    /// Billed cost of the side-channel in bits-per-weight.
    pub bpw_cost: f64,
}

/// Compute the de-bias correction from the original weights `w` and recon `recon`
/// (both `[out, in]`, row-major), an activation-mean model `mu_bar`, and the bias
/// payload width in bits (16 = bf16 side-channel; pass any existing bias width if
/// folding). Pure, deterministic, float-encode-side (decode stays integer/LUT).
pub fn debias_tensor(w: &[f32], recon: &[f32], in_features: usize, mu_bar: f32, bias_bits: u32) -> DebiasResult {
    assert_eq!(w.len(), recon.len(), "w/recon length mismatch");
    assert!(in_features > 0 && w.len() % in_features == 0, "ragged tensor");
    let out = w.len() / in_features;
    let mut rowsum_bias = vec![0.0f32; out];
    let mut bias_correction = vec![0.0f32; out];
    for i in 0..out {
        let base = i * in_features;
        // Kahan-summed rowsum delta for numerical honesty on long rows.
        let mut s = 0.0f64;
        let mut comp = 0.0f64;
        for j in 0..in_features {
            let d = recon[base + j] as f64 - w[base + j] as f64;
            let y = d - comp;
            let t = s + y;
            comp = (t - s) - y;
            s = t;
        }
        rowsum_bias[i] = s as f32;
        bias_correction[i] = (-(mu_bar as f64) * s) as f64 as f32;
    }
    let bpw_cost = bias_bits as f64 / in_features as f64;
    DebiasResult { bias_correction, rowsum_bias, mu_bar, bpw_cost }
}

/// Estimate `mu_bar` from a sample of activation vectors (each length `in_features`):
/// the grand mean over all sampled entries. Falls back to 0 on empty input (=>
/// degenerate zero correction, the honest zero-mean verdict).
pub fn estimate_mu_bar(samples: &[Vec<f32>]) -> f32 {
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for s in samples {
        for &v in s {
            sum += v as f64;
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        (sum / n as f64) as f32
    }
}

/// Simulated output: `y = W x` for one activation vector (row-major `[out,in]`).
pub fn matvec(w: &[f32], x: &[f32], in_features: usize) -> Vec<f32> {
    let out = w.len() / in_features;
    let mut y = vec![0.0f32; out];
    for i in 0..out {
        let base = i * in_features;
        let mut acc = 0.0f64;
        for j in 0..in_features {
            acc += w[base + j] as f64 * x[j] as f64;
        }
        y[i] = acc as f32;
    }
    y
}

/// Mean / RMS output error of a recon (optionally de-biased) over a batch of
/// activation vectors. Returns (mean_signed_error, rms_error) aggregated over all
/// (row, sample) pairs. `correction` (if Some) is added to each `y_recon`.
pub fn output_error(w: &[f32], recon: &[f32], in_features: usize, xs: &[Vec<f32>], correction: Option<&[f32]>) -> (f64, f64) {
    let out = w.len() / in_features;
    let mut sum_signed = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut count = 0usize;
    for x in xs {
        let y_ref = matvec(w, x, in_features);
        let y_rec = matvec(recon, x, in_features);
        for i in 0..out {
            let mut e = y_rec[i] as f64 - y_ref[i] as f64;
            if let Some(c) = correction {
                e += c[i] as f64;
            }
            sum_signed += e;
            sum_sq += e * e;
            count += 1;
        }
    }
    let n = count.max(1) as f64;
    (sum_signed / n, (sum_sq / n).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correction_cancels_first_moment_on_constant_x() {
        // x = mu_bar * 1  => y_recon - y_ref = mu_bar * S_i, and c_i = -mu_bar*S_i
        // so the corrected output error is EXACTLY zero per row (eq. 2/3 identity).
        let in_f = 8usize;
        let out = 3usize;
        let w: Vec<f32> = (0..out * in_f).map(|k| ((k as f32) * 0.31).sin()).collect();
        let recon: Vec<f32> = w.iter().map(|&v| (v * 7.0).round() / 7.0).collect();
        let mu_bar = 0.5f32;
        let r = debias_tensor(&w, &recon, in_f, mu_bar, 16);
        let x = vec![mu_bar; in_f];
        let (_, rms_uncorr) = output_error(&w, &recon, in_f, &[x.clone()], None);
        let (mean_corr, rms_corr) = output_error(&w, &recon, in_f, &[x], Some(&r.bias_correction));
        assert!(rms_uncorr > 1e-6, "need a real bias to cancel");
        assert!(rms_corr < 1e-4, "corrected rms should vanish on constant x: {rms_corr}");
        assert!(mean_corr.abs() < 1e-4);
    }

    #[test]
    fn zero_mean_makes_correction_zero() {
        // mu_bar = 0 => correction is identically zero (the dead-Hessian degeneracy).
        let in_f = 16usize;
        let w: Vec<f32> = (0..4 * in_f).map(|k| (k as f32 * 0.7).cos()).collect();
        let recon: Vec<f32> = w.iter().map(|&v| (v * 5.0).round() / 5.0).collect();
        let r = debias_tensor(&w, &recon, in_f, 0.0, 16);
        assert!(r.bias_correction.iter().all(|&c| c == 0.0));
        // but the rowsum bias itself is generically nonzero (it is the estimable
        // quantity the RHT rotates, not destroys).
        assert!(r.rowsum_bias.iter().any(|&s| s.abs() > 1e-6));
    }

    #[test]
    fn billing_is_inverse_in_features() {
        let r = debias_tensor(&[1.0; 32], &[1.0; 32], 8, 0.1, 16);
        assert!((r.bpw_cost - 16.0 / 8.0).abs() < 1e-12);
    }
}

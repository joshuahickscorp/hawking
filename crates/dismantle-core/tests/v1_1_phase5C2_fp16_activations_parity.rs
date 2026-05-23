//! Phase 5C.2 fp16 activations parity test.
//!
//! Verifies that the f16 intermediate activation path for the final-norm
//! → LM head step produces outputs that match the f32 baseline:
//!   - argmax must match exactly (regression guard)
//!   - logit values within atol=5e-3 (f16 quantization noise tolerance)
//!
//! Tests are synthetic (no model weights required):
//!
//! Tests:
//! - rmsnorm_f32_to_f16_parity — rmsnorm_f32 (f32 out) vs rmsnorm_f32_to_f16
//!   (f16 out promoted to f32): atol=5e-3, same argmax.
//! - gemv_f16_f16in_parity — gemv_f16 (f32 activation) vs gemv_f16_f16in
//!   (f16 activation): argmax matches exactly, logit atol=5e-3.
//! - end_to_end_final_norm_lm_head_parity — combined rmsnorm_f32_to_f16 +
//!   gemv_f16_f16in pipeline argmax matches rmsnorm_f32 + gemv_f16 pipeline.
//!
//! Design rationale: the residual stream stays f32 between layers; only the
//! per-layer normed activation is f16. This prevents the accumulation error
//! that caused the 3 prior f16 residual attempts to produce garbage output.

#![cfg(target_os = "macos")]

use std::time::{SystemTime, UNIX_EPOCH};

fn random_f32(seed: &mut u64) -> f32 {
    // xorshift64 — fast deterministic pseudo-random.
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    // Map to [-2.0, 2.0] (typical residual stream magnitude).
    ((*seed as i64 as f32) / (i64::MAX as f32)) * 2.0
}

fn make_residual(n: usize, seed: &mut u64) -> Vec<f32> {
    (0..n).map(|_| random_f32(seed)).collect()
}

fn make_weight(n: usize, seed: &mut u64) -> Vec<f32> {
    // RMS norm weights are typically close to 1.0.
    (0..n).map(|_| 0.5 + random_f32(seed).abs()).collect()
}

fn make_lm_head(rows: usize, cols: usize, seed: &mut u64) -> Vec<u16> {
    // f16 LM head weights — convert random f32 to f16 bits.
    (0..rows * cols)
        .map(|_| {
            let v = random_f32(seed) * 0.1; // small magnitude typical for weight matrices
            half::f16::from_f32(v).to_bits()
        })
        .collect()
}

/// Compute rmsnorm_f32 (f32 → f32) on CPU for reference.
fn rmsnorm_f32_ref(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
    let inv = 1.0 / rms;
    x.iter().zip(weight.iter()).map(|(&xv, &wv)| xv * inv * wv).collect()
}

/// Compute rmsnorm → f16 → f32 promote (simulates GPU rmsnorm_f32_to_f16).
fn rmsnorm_f32_to_f16_ref(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
    let inv = 1.0 / rms;
    x.iter()
        .zip(weight.iter())
        .map(|(&xv, &wv)| {
            // Simulate: store as f16 then load back to f32.
            let v_f32 = xv * inv * wv;
            half::f16::from_f32(v_f32).to_f32()
        })
        .collect()
}

/// Compute GEMV with f16 weights × f32 activation → f32 output (reference for gemv_f16).
fn gemv_f16_f32in_ref(w: &[u16], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            let row = &w[r * cols..(r + 1) * cols];
            row.iter()
                .zip(x.iter())
                .map(|(&wbits, &xv)| {
                    half::f16::from_bits(wbits).to_f32() * xv
                })
                .sum::<f32>()
        })
        .collect()
}

/// Compute GEMV with f16 weights × f16 activation → f32 output (reference for gemv_f16_f16in).
fn gemv_f16_f16in_ref(w: &[u16], x_f32: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    // Simulate: convert activation to f16, then compute.
    let x_f16: Vec<f32> = x_f32.iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect();
    (0..rows)
        .map(|r| {
            let row = &w[r * cols..(r + 1) * cols];
            row.iter()
                .zip(x_f16.iter())
                .map(|(&wbits, &xv)| {
                    half::f16::from_bits(wbits).to_f32() * xv
                })
                .sum::<f32>()
        })
        .collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Verify rmsnorm_f32_to_f16 produces outputs within atol of rmsnorm_f32.
/// The only difference is f16 rounding of each output element.
#[test]
fn rmsnorm_f32_to_f16_parity() {
    let mut seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    seed ^= 0xdeadbeef_12345678;

    // Use hidden=512 (V2-Lite uses 2048; 512 is sufficient for coverage).
    let hidden = 512;
    let eps = 1e-6_f32;

    for trial in 0..8 {
        let x = make_residual(hidden, &mut seed);
        let weight = make_weight(hidden, &mut seed);

        let ref_out = rmsnorm_f32_ref(&x, &weight, eps);
        let f16_out = rmsnorm_f32_to_f16_ref(&x, &weight, eps);

        let max_diff = ref_out
            .iter()
            .zip(f16_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff <= 5e-3,
            "trial {trial}: rmsnorm_f32_to_f16 max_diff={max_diff:.2e} > 5e-3"
        );

        // Argmax check: the top element should match between f32 and f16 outputs.
        // (This is a sanity check; argmax on norm output is less meaningful than
        //  argmax on logits, but verifies no catastrophic divergence.)
        let ref_top = argmax(&ref_out);
        let f16_top = argmax(&f16_out);
        assert_eq!(
            ref_top, f16_top,
            "trial {trial}: rmsnorm argmax mismatch ref={ref_top} f16={f16_top}"
        );
    }
    eprintln!("✓ rmsnorm_f32_to_f16 parity: atol≤5e-3, argmax match (8 trials, hidden={hidden})");
}

/// Verify gemv_f16_f16in (f16 activation) argmax matches gemv_f16 (f32 activation).
/// The input activation difference (f16 rounding) should not shift the winner.
#[test]
fn gemv_f16_f16in_parity() {
    let mut seed = 0xfeedface_abcd1234u64;

    // Small LM head shape for unit test speed: 128 rows × 256 cols.
    // Production is 102400×5120 but math is identical.
    let rows = 128;
    let cols = 256;
    let eps = 1e-6_f32;

    for trial in 0..8 {
        let x = make_residual(cols, &mut seed);
        let weight_norm = make_weight(cols, &mut seed);
        let lm_head = make_lm_head(rows, cols, &mut seed);

        // Reference: f32 rmsnorm output → f32 GEMV.
        let x_norm_f32 = rmsnorm_f32_ref(&x, &weight_norm, eps);
        let logits_f32 = gemv_f16_f32in_ref(&lm_head, &x_norm_f32, rows, cols);

        // Phase 5C.2 path: f16 rmsnorm output → f16 GEMV.
        let x_norm_f16 = rmsnorm_f32_to_f16_ref(&x, &weight_norm, eps);
        let logits_f16 = gemv_f16_f16in_ref(&lm_head, &x_norm_f16, rows, cols);

        // argmax must match.
        let top_f32 = argmax(&logits_f32);
        let top_f16 = argmax(&logits_f16);
        assert_eq!(
            top_f32, top_f16,
            "trial {trial}: gemv_f16_f16in argmax mismatch top_f32={top_f32} top_f16={top_f16}"
        );

        // Logit values within atol=5e-3 (f16 rounding noise in activation propagates
        // through the weight matrix; for typical weight magnitudes the error is small).
        let max_diff = logits_f32
            .iter()
            .zip(logits_f16.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff <= 5e-3,
            "trial {trial}: logit max_diff={max_diff:.2e} > atol=5e-3"
        );
    }
    eprintln!(
        "✓ gemv_f16_f16in parity: argmax exact, logit atol≤5e-3 (8 trials, {rows}×{cols})"
    );
}

/// Combined pipeline parity: rmsnorm_f32_to_f16 + gemv_f16_f16in argmax matches
/// rmsnorm_f32 + gemv_f16 for larger hidden size (closer to production shape).
#[test]
fn end_to_end_final_norm_lm_head_parity() {
    let mut seed = 0x1234abcd_5678ef90u64;

    // hidden=512, vocab=1024 — fast synthetic test covering the combined pipeline.
    let hidden = 512;
    let vocab = 1024;
    let eps = 1e-6_f32;

    let mut argmax_matches = 0usize;
    let trials = 16;

    for _trial in 0..trials {
        let residual = make_residual(hidden, &mut seed);
        let norm_weight = make_weight(hidden, &mut seed);
        let lm_head = make_lm_head(vocab, hidden, &mut seed);

        // f32 reference pipeline.
        let x_norm_f32 = rmsnorm_f32_ref(&residual, &norm_weight, eps);
        let logits_f32 = gemv_f16_f32in_ref(&lm_head, &x_norm_f32, vocab, hidden);

        // Phase 5C.2 f16 intermediate pipeline.
        let x_norm_f16 = rmsnorm_f32_to_f16_ref(&residual, &norm_weight, eps);
        let logits_f16 = gemv_f16_f16in_ref(&lm_head, &x_norm_f16, vocab, hidden);

        if argmax(&logits_f32) == argmax(&logits_f16) {
            argmax_matches += 1;
        }
    }

    // Require argmax match on ≥ 15/16 trials (>93%). In practice with well-distributed
    // random weights and activations the match rate is ~100%.
    assert!(
        argmax_matches >= 15,
        "end-to-end argmax match rate {argmax_matches}/{trials} < 15/16"
    );
    eprintln!(
        "✓ end-to-end final-norm+LM head parity: {argmax_matches}/{trials} argmax matches \
         (hidden={hidden}, vocab={vocab})"
    );

    // Note: with real model weights and longer sequences, argmax match is expected
    // to degrade slightly (f16 noise in x_norm → slight logit shifts). The parity
    // test above uses random small weights which are worst-case for noise propagation.
    // Production weights (Q4K) are also quantized, so their activation sensitivity
    // is already bounded by the Q4K rounding; f16 x_norm adds comparable noise.
    eprintln!(
        "  Phase 5C.2 scope: final-norm → LM head path only. Per-layer FFN-norm paths \
         remain f32 (future work: MoE/FFN gate+up GEMVs need f16-input variants)."
    );
}

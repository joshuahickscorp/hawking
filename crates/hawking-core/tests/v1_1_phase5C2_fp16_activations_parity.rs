#![cfg(target_os = "macos")]
use std::time::{SystemTime, UNIX_EPOCH};
fn random_f32(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed as i64 as f32) / (i64::MAX as f32)) * 2.0
}
fn make_residual(n: usize, seed: &mut u64) -> Vec<f32> {
    (0..n).map(|_| random_f32(seed)).collect()
}
fn make_weight(n: usize, seed: &mut u64) -> Vec<f32> {
    (0..n).map(|_| 0.5 + random_f32(seed).abs()).collect()
}
fn make_lm_head(rows: usize, cols: usize, seed: &mut u64) -> Vec<u16> {
    (0..rows * cols)
        .map(|_| {
            let v = random_f32(seed) * 0.1; // small magnitude typical for weight matrices
            half::f16::from_f32(v).to_bits()
        })
        .collect()
}
fn rmsnorm_f32_ref(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
    let inv = 1.0 / rms;
    x.iter()
        .zip(weight.iter())
        .map(|(&xv, &wv)| xv * inv * wv)
        .collect()
}
fn rmsnorm_f32_to_f16_ref(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
    let inv = 1.0 / rms;
    x.iter()
        .zip(weight.iter())
        .map(|(&xv, &wv)| {
            let v_f32 = xv * inv * wv;
            half::f16::from_f32(v_f32).to_f32()
        })
        .collect()
}
fn gemv_f16_f32in_ref(w: &[u16], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            let row = &w[r * cols..(r + 1) * cols];
            row.iter()
                .zip(x.iter())
                .map(|(&wbits, &xv)| half::f16::from_bits(wbits).to_f32() * xv)
                .sum::<f32>()
        })
        .collect()
}
fn gemv_f16_f16in_ref(w: &[u16], x_f32: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let x_f16: Vec<f32> = x_f32
        .iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect();
    (0..rows)
        .map(|r| {
            let row = &w[r * cols..(r + 1) * cols];
            row.iter()
                .zip(x_f16.iter())
                .map(|(&wbits, &xv)| half::f16::from_bits(wbits).to_f32() * xv)
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
#[test]
fn rmsnorm_f32_to_f16_parity() {
    let mut seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    seed ^= 0xdeadbeef_12345678;
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
        let ref_top = argmax(&ref_out);
        let f16_top = argmax(&f16_out);
        assert_eq!(
            ref_top, f16_top,
            "trial {trial}: rmsnorm argmax mismatch ref={ref_top} f16={f16_top}"
        );
    }
}
#[test]
fn gemv_f16_f16in_parity() {
    let mut seed = 0xfeedface_abcd1234u64;
    let rows = 128;
    let cols = 256;
    let eps = 1e-6_f32;
    for trial in 0..8 {
        let x = make_residual(cols, &mut seed);
        let weight_norm = make_weight(cols, &mut seed);
        let lm_head = make_lm_head(rows, cols, &mut seed);
        let x_norm_f32 = rmsnorm_f32_ref(&x, &weight_norm, eps);
        let logits_f32 = gemv_f16_f32in_ref(&lm_head, &x_norm_f32, rows, cols);
        let x_norm_f16 = rmsnorm_f32_to_f16_ref(&x, &weight_norm, eps);
        let logits_f16 = gemv_f16_f16in_ref(&lm_head, &x_norm_f16, rows, cols);
        let top_f32 = argmax(&logits_f32);
        let top_f16 = argmax(&logits_f16);
        assert_eq!(
            top_f32, top_f16,
            "trial {trial}: gemv_f16_f16in argmax mismatch top_f32={top_f32} top_f16={top_f16}"
        );
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
}
#[test]
fn end_to_end_final_norm_lm_head_parity() {
    let mut seed = 0x1234abcd_5678ef90u64;
    let hidden = 512;
    let vocab = 1024;
    let eps = 1e-6_f32;
    let mut argmax_matches = 0usize;
    let trials = 16;
    for _trial in 0..trials {
        let residual = make_residual(hidden, &mut seed);
        let norm_weight = make_weight(hidden, &mut seed);
        let lm_head = make_lm_head(vocab, hidden, &mut seed);
        let x_norm_f32 = rmsnorm_f32_ref(&residual, &norm_weight, eps);
        let logits_f32 = gemv_f16_f32in_ref(&lm_head, &x_norm_f32, vocab, hidden);
        let x_norm_f16 = rmsnorm_f32_to_f16_ref(&residual, &norm_weight, eps);
        let logits_f16 = gemv_f16_f16in_ref(&lm_head, &x_norm_f16, vocab, hidden);
        if argmax(&logits_f32) == argmax(&logits_f16) {
            argmax_matches += 1;
        }
    }
    assert!(
        argmax_matches >= 15,
        "end-to-end argmax match rate {argmax_matches}/{trials} < 15/16"
    );
}

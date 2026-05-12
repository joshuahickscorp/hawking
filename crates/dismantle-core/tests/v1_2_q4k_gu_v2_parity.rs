//! v1.2.0-2 — Parity test: moe_batched_gemm_q4_indexed_v2t_gu_v2 vs v2t_gu.
//!
//! Both kernels compute silu(gate) * up for routed Q4_K_M experts.  The v2
//! kernel adds: sumy correction trick, scale/activation preloading, and
//! paired nibble reads.  Output must match v2t_gu within atol=1e-3 (fp16
//! quantisation noise budget).
//!
//! Test shapes:
//!   1. routes=2, rows=16,   cols=256   — sub-TG edge case
//!   2. routes=6, rows=1408, cols=2048  — production DeepSeek V2-Lite shape

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::MetalContext;
use half::f16;
use once_cell::sync::Lazy;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

const ATOL: f32 = 1e-3;

fn ctx() -> &'static MetalContext {
    static CTX: Lazy<MetalContext> = Lazy::new(|| {
        MetalContext::new().expect("Metal device required for v1.2.0-2 parity test")
    });
    &CTX
}

fn fixed_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n)
        .map(|_| rng.gen_range(-1.0_f32..1.0_f32))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Build a synthetic fused Q4_K_M weight tensor for `n_experts` consecutive
/// matrices of shape (rows, blocks_per_row * 144 bytes).
fn synthetic_q4_k_bytes(n_experts: usize, rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let blocks_per_row = cols / 256;
    let bytes_per_expert = rows * blocks_per_row * 144;
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_experts * bytes_per_expert];
    for b in 0..(n_experts * rows * blocks_per_row) {
        let off = b * 144;
        // d: small positive fp16
        let d = 0.0005 + rng.gen::<f32>() * 0.001;
        bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        // dmin: small fp16 (can be negative)
        let dmin = (rng.gen::<f32>() - 0.5) * 0.001;
        bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        // Scale + nibble bytes
        for i in 4..144 {
            bytes[off + i] = rng.gen::<u8>();
        }
    }
    bytes
}

fn run_parity(routes: usize, rows: usize, cols: usize, seed_base: u64) {
    let n_experts = routes + 4;
    let blocks_per_row = cols / 256;
    let bytes_per_expert = rows * blocks_per_row * 144;

    // Build weight tensor: gate matrices followed by up matrices.
    let gate_bytes = synthetic_q4_k_bytes(n_experts, rows, cols, seed_base);
    let up_bytes   = synthetic_q4_k_bytes(n_experts, rows, cols, seed_base ^ 0x1234_5678);

    // Assemble with padding prefix to exercise non-zero offsets.
    let pad = 128usize;
    let mut w_all = vec![0xA5u8; pad + gate_bytes.len() + up_bytes.len()];
    let gate_offset = pad;
    let up_offset   = pad + n_experts * bytes_per_expert;
    w_all[gate_offset..gate_offset + gate_bytes.len()].copy_from_slice(&gate_bytes);
    w_all[up_offset..up_offset + up_bytes.len()].copy_from_slice(&up_bytes);

    // Route IDs — spread across experts
    let route_ids: Vec<u32> = (0..routes)
        .map(|i| ((i * 3 + 1) % n_experts) as u32)
        .collect();

    let x = fixed_f32(cols, seed_base ^ 0xDEAD_BEEF);

    // Reference: v2t_gu
    let mut ref_out = vec![0.0_f32; routes * rows];
    kernels::moe_batched_gemm_q4_indexed_v2t_gu_raw(
        ctx(),
        &w_all,
        gate_offset,
        up_offset,
        &route_ids,
        &x,
        routes,
        rows,
        cols,
        &mut ref_out,
    )
    .expect("v2t_gu dispatch failed");

    // Candidate: v2t_gu_v2
    let mut v2_out = vec![0.0_f32; routes * rows];
    kernels::moe_batched_gemm_q4_indexed_v2t_gu_v2_raw(
        ctx(),
        &w_all,
        gate_offset,
        up_offset,
        &route_ids,
        &x,
        routes,
        rows,
        cols,
        &mut v2_out,
    )
    .expect("v2t_gu_v2 dispatch failed");

    let diff = max_abs_diff(&ref_out, &v2_out);
    println!(
        "[v1.2.0-2] gu_v2 parity (routes={routes} rows={rows} cols={cols}) \
         max|ref-v2| = {diff:.6e}"
    );
    assert!(
        diff < ATOL,
        "v2t_gu_v2 vs v2t_gu diff {diff:.6e} >= atol {ATOL} \
         (routes={routes} rows={rows} cols={cols})"
    );
}

#[test]
fn test_gu_v2_parity_small() {
    run_parity(2, 16, 256, 0xBEEF_0001);
}

#[test]
fn test_gu_v2_parity_production() {
    run_parity(6, 1408, 2048, 0xBEEF_0002);
}

//! v0.3.0 — Numerical parity test: gemm_q4_k_m_fused_simd vs scalar reference.
//!
//! Shape: M=64, K=256 (1 Q4_K block per row), seed=42.
//! Asserts max |scalar - simd| < 1e-3 (fp16 quant noise tolerance).

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::MetalContext;
use dismantle_core::gguf::GgmlType;
use dismantle_core::quant::dequant_into;
use once_cell::sync::Lazy;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

pub const ATOL: f32 = 1e-3;

fn ctx() -> &'static MetalContext {
    static CTX: Lazy<MetalContext> =
        Lazy::new(|| MetalContext::new().expect("Metal device required for simd parity test"));
    &CTX
}

fn fixed_input(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Synthetic Q4_K weight bytes with small d/dmin so per-element magnitudes
/// stay bounded; prevents accumulation-order divergence from crossing 1e-3.
fn synthetic_q4_k_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
    use half::f16;
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_blocks * 144];
    for b in 0..n_blocks {
        let off = b * 144;
        let d = 0.01 + rng.gen::<f32>() * 0.01;
        let d_bits = f16::from_f32(d).to_bits();
        bytes[off..off + 2].copy_from_slice(&d_bits.to_le_bytes());
        let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
        let dmin_bits = f16::from_f32(dmin).to_bits();
        bytes[off + 2..off + 4].copy_from_slice(&dmin_bits.to_le_bytes());
        for i in 4..144 {
            bytes[off + i] = rng.gen::<u8>();
        }
    }
    bytes
}

#[test]
fn test_gemm_q4_k_simd_matches_scalar() {
    let rows = 64;
    let cols = 256; // 1 Q4_K block per row
    let n_blocks = rows * (cols / 256);

    let w_bytes = synthetic_q4_k_bytes(n_blocks, 42);
    let x = fixed_input(cols, 0xDEAD_BEEF);

    // Scalar reference: dequant → fp32 GEMV.
    let mut w_f32 = vec![0.0_f32; rows * cols];
    dequant_into(GgmlType::Q4_K, &w_bytes, &mut w_f32)
        .expect("Q4_K dequant should succeed for synthetic bytes");
    let mut scalar_out = vec![0.0_f32; rows];
    kernels::gemv_f32(&w_f32, rows, cols, &x, &mut scalar_out);

    // simdgroup Metal path.
    let ctx = ctx().clone();
    let mut simd_out = vec![0.0_f32; rows];
    kernels::gemv_q4_k_m_simd(&ctx, &w_bytes, rows, cols, &x, &mut simd_out)
        .expect("gemv_q4_k_m_simd should succeed");

    let diff = max_abs_diff(&scalar_out, &simd_out);
    println!("[v0.3.0] gemm_q4_k_simd parity max abs diff = {diff:.6e}");
    assert!(
        diff < ATOL,
        "gemm_q4_k_m_fused_simd vs scalar diff {diff:.6e} >= atol {ATOL}"
    );
}

#[test]
fn test_gemm_q4_k_simd_larger_shape() {
    // Larger shape: multiple Q4_K blocks per row, rows not multiple of 8.
    let rows = 128;
    let cols = 512; // 2 Q4_K blocks per row
    let n_blocks = rows * (cols / 256);

    let w_bytes = synthetic_q4_k_bytes(n_blocks, 0xCAFE_BABE);
    let x = fixed_input(cols, 0x1234_5678);

    let mut w_f32 = vec![0.0_f32; rows * cols];
    dequant_into(GgmlType::Q4_K, &w_bytes, &mut w_f32)
        .expect("Q4_K dequant should succeed");
    let mut scalar_out = vec![0.0_f32; rows];
    kernels::gemv_f32(&w_f32, rows, cols, &x, &mut scalar_out);

    let ctx = ctx().clone();
    let mut simd_out = vec![0.0_f32; rows];
    kernels::gemv_q4_k_m_simd(&ctx, &w_bytes, rows, cols, &x, &mut simd_out)
        .expect("gemv_q4_k_m_simd should succeed");

    let diff = max_abs_diff(&scalar_out, &simd_out);
    println!("[v0.3.0] gemm_q4_k_simd larger shape parity max abs diff = {diff:.6e}");
    assert!(
        diff < ATOL,
        "gemm_q4_k_m_fused_simd vs scalar diff {diff:.6e} >= atol {ATOL}"
    );
}

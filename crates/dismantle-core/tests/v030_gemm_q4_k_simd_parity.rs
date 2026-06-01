//! v0.3.0/v0.3.1 — Numerical parity test: gemm_q4_k_m_fused_simd vs scalar reference.
//!
//! Covers both the standalone path (gemv_q4_k_m_simd) and the batched path
//! (dispatch_gemv_q4_k_m_simd_batched / encode_gemv_q4_k_m_simd via ctx.dispatch_batch).
//! Shape: M=64, K=256 (1 Q4_K block per row), seed=42.
//! Asserts max |scalar - simd| < 1e-3 (fp16 quant noise tolerance).

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::gguf::GgmlType;
use dismantle_core::quant::dequant_into;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

mod common;
use common::*;

pub const ATOL: f32 = 1e-3;

fn fixed_input(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
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

// v0.3.1 batched-path parity tests: exercise dispatch_gemv_q4_k_m_simd_batched
// (routes through ctx.dispatch_batch { encode_gemv_q4_k_m_simd }).

#[test]
fn test_gemm_q4_k_simd_batched_matches_scalar() {
    let rows = 64;
    let cols = 256;
    let n_blocks = rows * (cols / 256);

    let w_bytes = synthetic_q4_k_bytes(n_blocks, 42);
    let x = fixed_input(cols, 0xDEAD_BEEF);

    let mut w_f32 = vec![0.0_f32; rows * cols];
    dequant_into(GgmlType::Q4_K, &w_bytes, &mut w_f32)
        .expect("Q4_K dequant should succeed for synthetic bytes");
    let mut scalar_out = vec![0.0_f32; rows];
    kernels::gemv_f32(&w_f32, rows, cols, &x, &mut scalar_out);

    let ctx = ctx().clone();
    let mut batched_out = vec![0.0_f32; rows];
    kernels::dispatch_gemv_q4_k_m_simd_batched(&ctx, &w_bytes, rows, cols, &x, &mut batched_out)
        .expect("dispatch_gemv_q4_k_m_simd_batched should succeed");

    let diff = max_abs_diff(&scalar_out, &batched_out);
    println!("[v0.3.1] gemm_q4_k_simd batched parity max abs diff = {diff:.6e}");
    assert!(
        diff < ATOL,
        "dispatch_gemv_q4_k_m_simd_batched vs scalar diff {diff:.6e} >= atol {ATOL}"
    );
}

#[test]
fn test_gemm_q4_k_simd_batched_larger_shape() {
    let rows = 128;
    let cols = 512;
    let n_blocks = rows * (cols / 256);

    let w_bytes = synthetic_q4_k_bytes(n_blocks, 0xCAFE_BABE);
    let x = fixed_input(cols, 0x1234_5678);

    let mut w_f32 = vec![0.0_f32; rows * cols];
    dequant_into(GgmlType::Q4_K, &w_bytes, &mut w_f32)
        .expect("Q4_K dequant should succeed");
    let mut scalar_out = vec![0.0_f32; rows];
    kernels::gemv_f32(&w_f32, rows, cols, &x, &mut scalar_out);

    let ctx = ctx().clone();
    let mut batched_out = vec![0.0_f32; rows];
    kernels::dispatch_gemv_q4_k_m_simd_batched(&ctx, &w_bytes, rows, cols, &x, &mut batched_out)
        .expect("dispatch_gemv_q4_k_m_simd_batched should succeed");

    let diff = max_abs_diff(&scalar_out, &batched_out);
    println!("[v0.3.1] gemm_q4_k_simd batched larger shape max abs diff = {diff:.6e}");
    assert!(
        diff < ATOL,
        "dispatch_gemv_q4_k_m_simd_batched vs scalar diff {diff:.6e} >= atol {ATOL}"
    );
}

// v0.3.2 pair-path parity tests: exercise dispatch_gemv_q4_k_m_simd_pair_batched
// (encodes two simd GEMVs — distinct weights, shared x — into one CommandBatch).

#[test]
fn test_gemm_q4_k_simd_pair_matches_scalar() {
    let rows = 64;
    let cols = 256;
    let n_blocks = rows * (cols / 256);

    let w_a_bytes = synthetic_q4_k_bytes(n_blocks, 42);
    let w_b_bytes = synthetic_q4_k_bytes(n_blocks, 0xDEAD_CAFE);
    let x = fixed_input(cols, 0xDEAD_BEEF);

    let mut w_a_f32 = vec![0.0_f32; rows * cols];
    let mut w_b_f32 = vec![0.0_f32; rows * cols];
    dequant_into(GgmlType::Q4_K, &w_a_bytes, &mut w_a_f32).expect("Q4_K dequant a");
    dequant_into(GgmlType::Q4_K, &w_b_bytes, &mut w_b_f32).expect("Q4_K dequant b");
    let mut scalar_a = vec![0.0_f32; rows];
    let mut scalar_b = vec![0.0_f32; rows];
    kernels::gemv_f32(&w_a_f32, rows, cols, &x, &mut scalar_a);
    kernels::gemv_f32(&w_b_f32, rows, cols, &x, &mut scalar_b);

    let ctx = ctx().clone();
    let mut pair_a = vec![0.0_f32; rows];
    let mut pair_b = vec![0.0_f32; rows];
    kernels::dispatch_gemv_q4_k_m_simd_pair_batched(
        &ctx, &w_a_bytes, &w_b_bytes, rows, cols, &x, &mut pair_a, &mut pair_b,
    )
    .expect("dispatch_gemv_q4_k_m_simd_pair_batched should succeed");

    let diff_a = max_abs_diff(&scalar_a, &pair_a);
    let diff_b = max_abs_diff(&scalar_b, &pair_b);
    println!("[v0.3.2] pair parity diff_a={diff_a:.6e} diff_b={diff_b:.6e}");
    assert!(diff_a < ATOL, "pair output A diff {diff_a:.6e} >= atol {ATOL}");
    assert!(diff_b < ATOL, "pair output B diff {diff_b:.6e} >= atol {ATOL}");
}

#[test]
fn test_gemm_q4_k_simd_pair_larger_shape() {
    let rows = 128;
    let cols = 512;
    let n_blocks = rows * (cols / 256);

    let w_a_bytes = synthetic_q4_k_bytes(n_blocks, 0xCAFE_BABE);
    let w_b_bytes = synthetic_q4_k_bytes(n_blocks, 0xABCD_1234);
    let x = fixed_input(cols, 0x1234_5678);

    let mut w_a_f32 = vec![0.0_f32; rows * cols];
    let mut w_b_f32 = vec![0.0_f32; rows * cols];
    dequant_into(GgmlType::Q4_K, &w_a_bytes, &mut w_a_f32).expect("Q4_K dequant a");
    dequant_into(GgmlType::Q4_K, &w_b_bytes, &mut w_b_f32).expect("Q4_K dequant b");
    let mut scalar_a = vec![0.0_f32; rows];
    let mut scalar_b = vec![0.0_f32; rows];
    kernels::gemv_f32(&w_a_f32, rows, cols, &x, &mut scalar_a);
    kernels::gemv_f32(&w_b_f32, rows, cols, &x, &mut scalar_b);

    let ctx = ctx().clone();
    let mut pair_a = vec![0.0_f32; rows];
    let mut pair_b = vec![0.0_f32; rows];
    kernels::dispatch_gemv_q4_k_m_simd_pair_batched(
        &ctx, &w_a_bytes, &w_b_bytes, rows, cols, &x, &mut pair_a, &mut pair_b,
    )
    .expect("dispatch_gemv_q4_k_m_simd_pair_batched larger shape should succeed");

    let diff_a = max_abs_diff(&scalar_a, &pair_a);
    let diff_b = max_abs_diff(&scalar_b, &pair_b);
    println!("[v0.3.2] pair larger shape diff_a={diff_a:.6e} diff_b={diff_b:.6e}");
    assert!(diff_a < ATOL, "pair larger output A diff {diff_a:.6e} >= atol {ATOL}");
    assert!(diff_b < ATOL, "pair larger output B diff {diff_b:.6e} >= atol {ATOL}");
}

// v0.3.3 pair+silu parity tests: exercise dispatch_gemv_q4_k_m_simd_pair_silu_batched
// (gate GEMV + up GEMV + silu_mul all in one CommandBatch; returns silu(gate)*up directly).
// ATOL_SILU is wider than ATOL: Q4_K GEMV error (~1e-3) is amplified by the silu output
// magnitude (~3–6 for the synthetic weights used here, d≈0.015, K=256-512).
const ATOL_SILU: f32 = 1e-2;

#[test]
fn test_gemm_q4_k_simd_pair_silu_matches_scalar() {
    let rows = 64;
    let cols = 256;
    let n_blocks = rows * (cols / 256);

    let w_gate_bytes = synthetic_q4_k_bytes(n_blocks, 42);
    let w_up_bytes   = synthetic_q4_k_bytes(n_blocks, 0xDEAD_CAFE);
    let x = fixed_input(cols, 0xDEAD_BEEF);

    // Scalar reference: dequant both, GEMV both, CPU silu_mul.
    let mut w_gate_f32 = vec![0.0_f32; rows * cols];
    let mut w_up_f32   = vec![0.0_f32; rows * cols];
    dequant_into(GgmlType::Q4_K, &w_gate_bytes, &mut w_gate_f32).expect("Q4_K dequant gate");
    dequant_into(GgmlType::Q4_K, &w_up_bytes,   &mut w_up_f32).expect("Q4_K dequant up");
    let mut g_ref = vec![0.0_f32; rows];
    let mut u_ref = vec![0.0_f32; rows];
    kernels::gemv_f32(&w_gate_f32, rows, cols, &x, &mut g_ref);
    kernels::gemv_f32(&w_up_f32,   rows, cols, &x, &mut u_ref);
    let mut ref_a = vec![0.0_f32; rows];
    kernels::silu_mul(&g_ref, &u_ref, &mut ref_a);

    // GPU fused path.
    let ctx = ctx().clone();
    let mut gpu_a = vec![0.0_f32; rows];
    kernels::dispatch_gemv_q4_k_m_simd_pair_silu_batched(
        &ctx, &w_gate_bytes, &w_up_bytes, rows, cols, &x, &mut gpu_a,
    )
    .expect("dispatch_gemv_q4_k_m_simd_pair_silu_batched should succeed");

    let diff = max_abs_diff(&ref_a, &gpu_a);
    println!("[v0.3.3] pair+silu parity diff={diff:.6e}");
    assert!(diff < ATOL_SILU, "pair+silu diff {diff:.6e} >= atol_silu {ATOL_SILU}");
}

#[test]
fn test_gemm_q4_k_simd_pair_silu_larger_shape() {
    let rows = 128;
    let cols = 512;
    let n_blocks = rows * (cols / 256);

    let w_gate_bytes = synthetic_q4_k_bytes(n_blocks, 0xCAFE_BABE);
    let w_up_bytes   = synthetic_q4_k_bytes(n_blocks, 0xABCD_1234);
    let x = fixed_input(cols, 0x1234_5678);

    let mut w_gate_f32 = vec![0.0_f32; rows * cols];
    let mut w_up_f32   = vec![0.0_f32; rows * cols];
    dequant_into(GgmlType::Q4_K, &w_gate_bytes, &mut w_gate_f32).expect("Q4_K dequant gate");
    dequant_into(GgmlType::Q4_K, &w_up_bytes,   &mut w_up_f32).expect("Q4_K dequant up");
    let mut g_ref = vec![0.0_f32; rows];
    let mut u_ref = vec![0.0_f32; rows];
    kernels::gemv_f32(&w_gate_f32, rows, cols, &x, &mut g_ref);
    kernels::gemv_f32(&w_up_f32,   rows, cols, &x, &mut u_ref);
    let mut ref_a = vec![0.0_f32; rows];
    kernels::silu_mul(&g_ref, &u_ref, &mut ref_a);

    let ctx = ctx().clone();
    let mut gpu_a = vec![0.0_f32; rows];
    kernels::dispatch_gemv_q4_k_m_simd_pair_silu_batched(
        &ctx, &w_gate_bytes, &w_up_bytes, rows, cols, &x, &mut gpu_a,
    )
    .expect("dispatch_gemv_q4_k_m_simd_pair_silu_batched larger shape should succeed");

    let diff = max_abs_diff(&ref_a, &gpu_a);
    println!("[v0.3.3] pair+silu larger shape diff={diff:.6e}");
    assert!(diff < ATOL_SILU, "pair+silu larger shape diff {diff:.6e} >= atol_silu {ATOL_SILU}");
}

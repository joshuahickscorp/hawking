//! Phase 2 / WB — Weight-pinning parity tests.
//!
//! WB pre-uploads kernel-bound weight tensors as `metal::Buffer`
//! once at model load time and adds `*_pinned` variants of the
//! kernel entry points that reference the buffer instead of memcpy'ing
//! a host slice on every dispatch. The byte-slice variants stay as the
//! parity-test surface — this file proves the pinned and byte-slice
//! paths produce bit-identical output for the same input.
//!
//! Coverage (incremental — extends as more weights are pinned):
//! - `gemv_f16_metal_pinned` vs `gemv_f16_metal` — LM head shape
//!   (~vocab × hidden) and a smaller fixed shape that matches the
//!   `phase1_kernel_parity` test.

#![cfg(target_os = "macos")]

use half::f16;
use hawking_core::kernels;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

mod common;
use common::*;

fn fixed_input(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
}

fn gemv_f16_pinned_check(rows: usize, cols: usize, seed_x: u64, seed_w: u64) {
    let x = fixed_input(cols, seed_x);
    let w_f32 = fixed_input(rows * cols, seed_w);
    let w_f16: Vec<f16> = w_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let w_bytes: &[u8] = bytemuck::cast_slice(&w_f16);

    let ctx = ctx().clone();

    // Path A: byte-slice (parity-test reference path).
    let mut out_byte_slice = vec![0.0_f32; rows];
    kernels::gemv_f16_metal(&ctx, w_bytes, rows, cols, &x, &mut out_byte_slice)
        .expect("gemv_f16_metal byte-slice path");

    // Path B: pinned (production path with WB).
    let w_buf = ctx.new_buffer_with_bytes(w_bytes);
    let mut out_pinned = vec![0.0_f32; rows];
    kernels::gemv_f16_metal_pinned(&ctx, &w_buf, rows, cols, &x, &mut out_pinned)
        .expect("gemv_f16_metal_pinned");

    // Pinned and byte-slice paths must be bit-identical: same kernel,
    // same input bytes, just a different upload path. ATOL is generous
    // (1e-3) but anything beyond fp32 representation noise is a bug.
    let max_diff = out_byte_slice
        .iter()
        .zip(out_pinned.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("[WB] gemv_f16 ({rows}x{cols}) byte-slice vs pinned max abs diff = {max_diff:.6}");
    assert!(
        max_diff < ATOL,
        "gemv_f16 pinned/byte-slice diverged: {max_diff} >= atol {ATOL}"
    );
}

#[test]
fn test_gemv_f16_pinned_matches_byte_slice_small() {
    // Small shape that exercises kernel reduction without depending on
    // the LM head being loaded. Mirrors the size used by
    // `phase1_kernel_parity::test_gemv_f16_matches_cpu`.
    gemv_f16_pinned_check(4096, 2048, 0xA1A1_A1A1, 0xB2B2_B2B2);
}

#[test]
fn test_gemv_f16_pinned_matches_byte_slice_lm_head_shape() {
    // DeepSeek-V2-Lite LM-head shape: vocab=102400, hidden=2048.
    // This is the production target — the buffer that lives on
    // `DeepSeekV2.lm_head_buf` is exactly this size.
    gemv_f16_pinned_check(102400, 2048, 0xC3C3_C3C3, 0xD4D4_D4D4);
}

fn gemv_f32_attn_pinned_check(rows: usize, cols: usize, seed_x: u64, seed_w: u64) {
    let x = fixed_input(cols, seed_x);
    let w = fixed_input(rows * cols, seed_w);
    let w_bytes: &[u8] = bytemuck::cast_slice(&w);

    let ctx = ctx().clone();

    let mut out_byte_slice = vec![0.0_f32; rows];
    kernels::gemv_f32_attn_metal(&ctx, &w, rows, cols, &x, &mut out_byte_slice)
        .expect("gemv_f32_attn_metal byte-slice path");

    let w_buf = ctx.new_buffer_with_bytes(w_bytes);
    let mut out_pinned = vec![0.0_f32; rows];
    kernels::gemv_f32_attn_metal_pinned(&ctx, &w_buf, rows, cols, &x, &mut out_pinned)
        .expect("gemv_f32_attn_metal_pinned");

    let max_diff = out_byte_slice
        .iter()
        .zip(out_pinned.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!(
        "[WB] gemv_f32_attn ({rows}x{cols}) byte-slice vs pinned max abs diff = {max_diff:.6}"
    );
    assert!(
        max_diff < ATOL,
        "gemv_f32_attn pinned/byte-slice diverged: {max_diff} >= atol {ATOL}"
    );
}

#[test]
fn test_gemv_f32_attn_pinned_q_a_proj() {
    gemv_f32_attn_pinned_check(1536, 2048, 0xE1E1_E1E1, 0xE2E2_E2E2);
}

#[test]
fn test_gemv_f32_attn_pinned_q_b_proj() {
    gemv_f32_attn_pinned_check(3072, 1536, 0xF1F1_F1F1, 0xF2F2_F2F2);
}

#[test]
fn test_gemv_f32_attn_pinned_kv_a_proj() {
    gemv_f32_attn_pinned_check(576, 2048, 0x1010_1010, 0x2020_2020);
}

#[test]
fn test_gemv_f32_attn_pinned_kv_b_proj() {
    gemv_f32_attn_pinned_check(2048, 512, 0x3030_3030, 0x4040_4040);
}

#[test]
fn test_gemv_f32_attn_pinned_o_proj() {
    // hidden × (n_heads × v_head_dim) = 2048 × 2048
    gemv_f32_attn_pinned_check(2048, 2048, 0x5050_5050, 0x6060_6060);
}

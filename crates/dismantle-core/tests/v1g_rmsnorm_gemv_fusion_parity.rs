//! Wedge G parity: fused rmsnorm+gemv TCB matches CPU reference.
//!
//! Tests:
//!   1. `wedge_g_rmsnorm_gemv_f32_attn_pinned_tcb_matches_cpu` — fused dispatch
//!      produces same output as sequential CPU rmsnorm + gemv_f32 within atol 1e-3.
//!   2. `wedge_g_fused_pair_matches_cpu` — two fused calls in one TCB (simulating
//!      q_a + kv_a dispatch) both match their CPU references.
//!   3. `wedge_g_fused_argmax_agrees_with_unfused` — argmax of fused and unfused
//!      GEMV agree (temp=0 token parity).
#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::TokenCommandBuffer;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

mod common;
use common::*;

fn fixed_f32_positive(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n).map(|_| rng.gen_range(0.5_f32..1.5_f32)).collect()
}

fn cpu_rmsnorm_gemv_f32(w: &[f32], x: &[f32], weight: &[f32], eps: f32, rows: usize, cols: usize) -> Vec<f32> {
    let mut x_norm = vec![0.0f32; cols];
    kernels::rmsnorm(x, weight, eps, &mut x_norm);
    let mut out = vec![0.0f32; rows];
    kernels::gemv_f32(w, rows, cols, &x_norm, &mut out);
    out
}

// ─────────────────────────────────────────────────────────────────────────────

/// rmsnorm_gemv_f32_attn_pinned_tcb matches CPU rmsnorm + gemv_f32.
/// Shape: rows=64 (q_lora_rank analogue), cols=256 (hidden analogue).
#[test]
fn wedge_g_rmsnorm_gemv_f32_attn_pinned_tcb_matches_cpu() {
    let ctx = ctx();
    let rows = 64usize;
    let cols = 256usize;
    let eps = 1e-6f32;

    let w = fixed_f32(rows * cols, 0xA1B2_C3D4);
    let x = fixed_f32(cols, 0xE5F6_0718);
    let weight = fixed_f32_positive(cols, 0x1234_5678);

    let cpu_out = cpu_rmsnorm_gemv_f32(&w, &x, &weight, eps, rows, cols);

    let w_buf = new_f32_buf(ctx, &w);
    let x_buf = new_f32_buf(ctx, &x);
    let weight_buf = new_f32_buf(ctx, &weight);
    let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::rmsnorm_gemv_f32_attn_pinned_tcb(
        &mut tcb, &w_buf, &x_buf, &weight_buf, eps, &out_buf, rows, cols,
    )
    .expect("rmsnorm_gemv_f32_attn_pinned_tcb");
    tcb.commit_and_wait().expect("commit");

    let gpu_out = read_f32_buf(&out_buf, rows);
    let diff = max_abs_diff(&cpu_out, &gpu_out);
    assert!(
        diff < 1e-3,
        "rmsnorm_gemv rows={rows} cols={cols}: max_abs_diff={diff:.2e} > 1e-3"
    );
}

/// Two fused calls in one TCB (q_a + kv_a analogue) both match CPU reference.
/// Mirrors the actual attention_tcb_inner Phase 1 usage: same x_buf, different w.
#[test]
fn wedge_g_fused_pair_matches_cpu() {
    let ctx = ctx();
    let rows_a = 48usize;  // q_lora_rank analogue
    let rows_b = 64usize;  // kv_a_dim analogue
    let cols = 128usize;   // hidden analogue
    let eps = 1e-5f32;

    let w_a = fixed_f32(rows_a * cols, 0xAAAA_1111);
    let w_b = fixed_f32(rows_b * cols, 0xBBBB_2222);
    let x = fixed_f32(cols, 0xCCCC_3333);
    let weight = fixed_f32_positive(cols, 0xDDDD_4444);

    let cpu_out_a = cpu_rmsnorm_gemv_f32(&w_a, &x, &weight, eps, rows_a, cols);
    let cpu_out_b = cpu_rmsnorm_gemv_f32(&w_b, &x, &weight, eps, rows_b, cols);

    let w_a_buf = new_f32_buf(ctx, &w_a);
    let w_b_buf = new_f32_buf(ctx, &w_b);
    let x_buf = new_f32_buf(ctx, &x);
    let weight_buf = new_f32_buf(ctx, &weight);
    let out_a_buf = ctx.new_buffer(rows_a * std::mem::size_of::<f32>());
    let out_b_buf = ctx.new_buffer(rows_b * std::mem::size_of::<f32>());

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::rmsnorm_gemv_f32_attn_pinned_tcb(
        &mut tcb, &w_a_buf, &x_buf, &weight_buf, eps, &out_a_buf, rows_a, cols,
    )
    .expect("rmsnorm_gemv q_a");
    kernels::rmsnorm_gemv_f32_attn_pinned_tcb(
        &mut tcb, &w_b_buf, &x_buf, &weight_buf, eps, &out_b_buf, rows_b, cols,
    )
    .expect("rmsnorm_gemv kv_a");
    tcb.commit_and_wait().expect("commit");

    let gpu_out_a = read_f32_buf(&out_a_buf, rows_a);
    let gpu_out_b = read_f32_buf(&out_b_buf, rows_b);

    let diff_a = max_abs_diff(&cpu_out_a, &gpu_out_a);
    let diff_b = max_abs_diff(&cpu_out_b, &gpu_out_b);
    assert!(
        diff_a < 1e-3,
        "q_a fused: max_abs_diff={diff_a:.2e} > 1e-3"
    );
    assert!(
        diff_b < 1e-3,
        "kv_a fused: max_abs_diff={diff_b:.2e} > 1e-3"
    );
}

/// Argmax of fused and unfused GEMV agree (temperature=0 token parity).
/// Ensures Wedge G does not change the winner of the projected output.
#[test]
fn wedge_g_fused_argmax_agrees_with_unfused() {
    let ctx = ctx();
    let rows = 128usize;
    let cols = 256usize;
    let eps = 1e-6f32;

    let w = fixed_f32(rows * cols, 0xF00D_CAFE);
    let x = fixed_f32(cols, 0xDEAD_BEEF);
    let weight = fixed_f32_positive(cols, 0xBEEF_CAFE);

    // CPU: rmsnorm → gemv_f32 → argmax.
    let cpu_out = cpu_rmsnorm_gemv_f32(&w, &x, &weight, eps, rows, cols);
    let cpu_winner = cpu_out
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();

    // GPU fused TCB.
    let w_buf = new_f32_buf(ctx, &w);
    let x_buf = new_f32_buf(ctx, &x);
    let weight_buf = new_f32_buf(ctx, &weight);
    let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::rmsnorm_gemv_f32_attn_pinned_tcb(
        &mut tcb, &w_buf, &x_buf, &weight_buf, eps, &out_buf, rows, cols,
    )
    .expect("rmsnorm_gemv_f32_attn_pinned_tcb");
    tcb.commit_and_wait().expect("commit");

    let gpu_out = read_f32_buf(&out_buf, rows);
    let gpu_winner = gpu_out
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();

    assert_eq!(
        gpu_winner, cpu_winner,
        "argmax winner differs: gpu={gpu_winner} cpu={cpu_winner}"
    );
}

//! Wedge H parity: simdgroup_matrix GEMV matches CPU reference.
//!
//! Tests:
//!   1. `wedge_h_simdgroup_f32_basic` — basic rows×cols shapes; atol 1e-5.
//!   2. `wedge_h_simdgroup_f32_qb_shape` — q_b_proj analogue shape (rows=256, cols=64).
//!   3. `wedge_h_simdgroup_f32_argmax_agrees` — argmax of simdgroup and scalar GEMV agree.
#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::TokenCommandBuffer;

mod common;
use common::*;

// ─────────────────────────────────────────────────────────────────────────────

/// simdgroup_f32 GEMV matches CPU gemv_f32 at atol 1e-4 for square shapes.
#[test]
fn wedge_h_simdgroup_f32_basic() {
    let ctx = ctx();

    for &(rows, cols) in &[(8usize, 8usize), (16, 8), (8, 16), (32, 64), (64, 32)] {
        let w = fixed_f32(rows * cols, 0xA1B2_C3D4 ^ rows as u64);
        let x = fixed_f32(cols, 0xE5F6_0718 ^ cols as u64);

        let mut cpu_out = vec![0.0f32; rows];
        kernels::gemv_f32(&w, rows, cols, &x, &mut cpu_out);

        let w_buf = new_f32_buf(ctx, &w);
        let x_buf = new_f32_buf(ctx, &x);
        let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_simdgroup_f32_tcb(&mut tcb, &w_buf, &x_buf, &y_buf, rows, cols)
            .unwrap_or_else(|e| panic!("gemv_simdgroup_f32_tcb rows={rows} cols={cols}: {e}"));
        tcb.commit_and_wait().expect("commit");

        let gpu_out = read_f32_buf(&y_buf, rows);
        let diff = max_abs_diff(&cpu_out, &gpu_out);
        assert!(
            diff < 1e-4,
            "rows={rows} cols={cols}: max_abs_diff={diff:.2e} > 1e-4"
        );
    }
}

/// q_b_proj analogue: rows=256 (heads×head_dim proxy), cols=64 (q_lora proxy).
/// Simulates the actual Phase 2 mini-TCB shape in attention_tcb_inner.
#[test]
fn wedge_h_simdgroup_f32_qb_shape() {
    let ctx = ctx();
    let rows = 256usize;
    let cols = 64usize;

    let w = fixed_f32(rows * cols, 0xBEEF_CAFE);
    let x = fixed_f32(cols, 0xDEAD_BEEF);

    let mut cpu_out = vec![0.0f32; rows];
    kernels::gemv_f32(&w, rows, cols, &x, &mut cpu_out);

    let w_buf = new_f32_buf(ctx, &w);
    let x_buf = new_f32_buf(ctx, &x);
    let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_simdgroup_f32_tcb(&mut tcb, &w_buf, &x_buf, &y_buf, rows, cols)
        .expect("gemv_simdgroup_f32_tcb");
    tcb.commit_and_wait().expect("commit");

    let gpu_out = read_f32_buf(&y_buf, rows);
    let diff = max_abs_diff(&cpu_out, &gpu_out);
    assert!(
        diff < 1e-3,
        "q_b_shape rows={rows} cols={cols}: max_abs_diff={diff:.2e} > 1e-3"
    );
}

/// Argmax of simdgroup GEMV matches CPU gemv_f32 argmax (token parity at temp=0).
#[test]
fn wedge_h_simdgroup_f32_argmax_agrees() {
    let ctx = ctx();
    let rows = 128usize;
    let cols = 64usize;

    let w = fixed_f32(rows * cols, 0xF00D_1234);
    let x = fixed_f32(cols, 0xCAFE_5678);

    let mut cpu_out = vec![0.0f32; rows];
    kernels::gemv_f32(&w, rows, cols, &x, &mut cpu_out);
    let cpu_winner = cpu_out
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();

    let w_buf = new_f32_buf(ctx, &w);
    let x_buf = new_f32_buf(ctx, &x);
    let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_simdgroup_f32_tcb(&mut tcb, &w_buf, &x_buf, &y_buf, rows, cols)
        .expect("gemv_simdgroup_f32_tcb");
    tcb.commit_and_wait().expect("commit");

    let gpu_out = read_f32_buf(&y_buf, rows);
    let gpu_winner = gpu_out
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();

    assert_eq!(
        gpu_winner, cpu_winner,
        "argmax: gpu={gpu_winner} cpu={cpu_winner}"
    );
}

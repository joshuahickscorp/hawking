#![cfg(target_os = "macos")]
//! Track 3.15-style FFN tail parity:
//! `ffn_down_swiglu + add_rmsnorm_ffn` must match the old
//! `silu_mul + ffn_down + add_rmsnorm` sequence for Q4_K predec weights.

use hawking_core::kernels;
use hawking_core::metal::{MetalContext, TokenCommandBuffer};

mod common;
use common::*;

fn make_q4k_weights(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    let blocks_per_row = cols / 256;
    let total_bytes = rows * blocks_per_row * 144;
    let w: Vec<u8> = (0..total_bytes)
        .map(|i| ((i as u32).wrapping_mul(2246822519u32).wrapping_add(seed)) as u8)
        .collect();
    let n_scales = rows * blocks_per_row * 16;
    let s: Vec<f32> = (0..n_scales)
        .map(|i| {
            let v = ((i as u32).wrapping_mul(2654435761u32).wrapping_add(seed)) as f32
                / u32::MAX as f32;
            v * 2.0 - 1.0
        })
        .collect();
    (w, s)
}

fn rand_vec(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(2654435761u32).wrapping_add(seed)) as f32;
            (x / u32::MAX as f32) * 4.0 - 2.0
        })
        .collect()
}

fn run_ref(
    ctx: &MetalContext,
    w_q4: &[u8],
    scales: &[f32],
    gate: &[f32],
    up: &[f32],
    x: &[f32],
    norm_weight: &[f32],
    rows: usize,
    cols: usize,
    b: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let w_buf = ctx.new_buffer_with_bytes(w_q4);
    let sc_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
    let gate_buf = new_f32_buf(ctx, gate);
    let up_buf = new_f32_buf(ctx, up);
    let x_buf = new_f32_buf(ctx, x);
    let norm_buf = new_f32_buf(ctx, norm_weight);
    let act_buf = ctx.new_buffer(b * cols * 4);
    let down_buf = ctx.new_buffer(b * rows * 4);
    let xnorm_buf = ctx.new_buffer(b * rows * 4);
    let w_bytes = rows * (cols / 256) * 144;

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::silu_mul_tcb(&mut tcb, &gate_buf, &up_buf, &act_buf, b * cols).unwrap();
    if b == 1 {
        kernels::gemv_q4_k_v4_predec_pinned_tcb(
            &mut tcb, &w_buf, 0, w_bytes, &sc_buf, 0, rows, cols, &act_buf, &down_buf,
        )
        .unwrap();
    } else if b <= 4 {
        kernels::gemm_q4_k_m_batched_v4r_predec_pinned_tcb(
            &mut tcb, &w_buf, 0, w_bytes, &sc_buf, 0, rows, cols, b, &act_buf, &down_buf,
        )
        .unwrap();
    } else {
        kernels::gemm_q4_k_m_batched_v3w_predec_pinned_tcb(
            &mut tcb, &w_buf, 0, w_bytes, &sc_buf, 0, rows, cols, b, &act_buf, &down_buf,
        )
        .unwrap();
    }
    kernels::add_rmsnorm_fused_batched_tcb(
        &mut tcb, &x_buf, &down_buf, &norm_buf, &xnorm_buf, 1e-6, rows, b,
    )
    .unwrap();
    tcb.commit_and_wait().unwrap();

    (
        read_f32_buf(&down_buf, b * rows),
        read_f32_buf(&x_buf, b * rows),
        read_f32_buf(&xnorm_buf, b * rows),
    )
}

fn run_fused(
    ctx: &MetalContext,
    w_q4: &[u8],
    scales: &[f32],
    gate: &[f32],
    up: &[f32],
    x: &[f32],
    norm_weight: &[f32],
    rows: usize,
    cols: usize,
    b: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let w_buf = ctx.new_buffer_with_bytes(w_q4);
    let sc_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
    let gate_buf = new_f32_buf(ctx, gate);
    let up_buf = new_f32_buf(ctx, up);
    let x_buf = new_f32_buf(ctx, x);
    let norm_buf = new_f32_buf(ctx, norm_weight);
    let down_buf = ctx.new_buffer(b * rows * 4);
    let xnorm_buf = ctx.new_buffer(b * rows * 4);
    let w_bytes = rows * (cols / 256) * 144;

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::ffn_down_swiglu_add_rmsnorm_ffn_q4k_predec_batched_tcb(
        &mut tcb, &w_buf, 0, w_bytes, &sc_buf, 0, rows, cols, b, &gate_buf, &up_buf, &x_buf,
        &norm_buf, &xnorm_buf, 1e-6, &down_buf,
    )
    .unwrap();
    tcb.commit_and_wait().unwrap();

    (
        read_f32_buf(&down_buf, b * rows),
        read_f32_buf(&x_buf, b * rows),
        read_f32_buf(&xnorm_buf, b * rows),
    )
}

#[test]
fn q4k_predec_swiglu_add_rmsnorm_tail_matches_ref() {
    let ctx = ctx();
    let rows = 512;
    let cols = 1024;
    let (w, scales) = make_q4k_weights(rows, cols, 0x3150);

    for b in [1usize, 3, 5] {
        let gate = rand_vec(b * cols, 0x5151 + b as u32);
        let up = rand_vec(b * cols, 0x6161 + b as u32);
        let x = rand_vec(b * rows, 0x7171 + b as u32);
        let norm_weight: Vec<f32> = rand_vec(rows, 0x8181 + b as u32)
            .into_iter()
            .map(|v| v.abs() + 0.5)
            .collect();

        let (ref_down, ref_x, ref_xnorm) = run_ref(
            ctx,
            &w,
            &scales,
            &gate,
            &up,
            &x,
            &norm_weight,
            rows,
            cols,
            b,
        );
        let (fused_down, fused_x, fused_xnorm) = run_fused(
            ctx,
            &w,
            &scales,
            &gate,
            &up,
            &x,
            &norm_weight,
            rows,
            cols,
            b,
        );

        let down_diff = max_abs_diff(&ref_down, &fused_down);
        let x_diff = max_abs_diff(&ref_x, &fused_x);
        let xnorm_diff = max_abs_diff(&ref_xnorm, &fused_xnorm);
        assert!(
            down_diff < 1e-4 && x_diff < 1e-4 && xnorm_diff < 1e-4,
            "B={b}: tail diffs down={down_diff:.2e} x={x_diff:.2e} xnorm={xnorm_diff:.2e}"
        );
        eprintln!(
            "tail swiglu+add_rmsnorm B={b}: down={down_diff:.2e} x={x_diff:.2e} xnorm={xnorm_diff:.2e} OK"
        );
    }
}

#![cfg(target_os = "macos")]
use hawking_core::kernels;
use hawking_core::metal::{MetalContext, TokenCommandBuffer};
mod common;
use common::*;
fn run_ref_v3w(
    ctx: &MetalContext,
    w_q4: &[u8],
    scales: &[f32],
    gate: &[f32],
    up: &[f32],
    rows: usize,
    cols: usize,
    b: usize,
) -> Vec<f32> {
    let w_buf = ctx.new_buffer_with_bytes(w_q4);
    let sc_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
    let gate_buf = new_f32_buf(ctx, gate);
    let up_buf = new_f32_buf(ctx, up);
    let act_buf = ctx.new_buffer(b * cols * 4);
    let y_buf = ctx.new_buffer(b * rows * 4);
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::silu_mul_tcb(&mut tcb, &gate_buf, &up_buf, &act_buf, b * cols).unwrap();
    let w_bytes = rows * (cols / 256) * 144;
    kernels::gemm_q4_k_m_batched_v3w_predec_pinned_tcb(
        &mut tcb, &w_buf, 0, w_bytes, &sc_buf, 0, rows, cols, b, &act_buf, &y_buf,
    )
    .unwrap();
    tcb.commit_and_wait().unwrap();
    read_f32_buf(&y_buf, b * rows)
}
fn run_fused_v3w(
    ctx: &MetalContext,
    w_q4: &[u8],
    scales: &[f32],
    gate: &[f32],
    up: &[f32],
    rows: usize,
    cols: usize,
    b: usize,
) -> Vec<f32> {
    let w_buf = ctx.new_buffer_with_bytes(w_q4);
    let sc_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
    let gate_buf = new_f32_buf(ctx, gate);
    let up_buf = new_f32_buf(ctx, up);
    let y_buf = ctx.new_buffer(b * rows * 4);
    let mut tcb = TokenCommandBuffer::new(ctx);
    let w_bytes = rows * (cols / 256) * 144;
    kernels::gemm_q4_k_m_batched_v3w_predec_swiglu_pinned_tcb(
        &mut tcb, &w_buf, 0, w_bytes, &sc_buf, 0, rows, cols, b, &gate_buf, &up_buf, &y_buf,
    )
    .unwrap();
    tcb.commit_and_wait().unwrap();
    read_f32_buf(&y_buf, b * rows)
}
fn run_ref_v4r(
    ctx: &MetalContext,
    w_q4: &[u8],
    scales: &[f32],
    gate: &[f32],
    up: &[f32],
    rows: usize,
    cols: usize,
    b: usize,
) -> Vec<f32> {
    let w_buf = ctx.new_buffer_with_bytes(w_q4);
    let sc_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
    let gate_buf = new_f32_buf(ctx, gate);
    let up_buf = new_f32_buf(ctx, up);
    let act_buf = ctx.new_buffer(b * cols * 4);
    let y_buf = ctx.new_buffer(b * rows * 4);
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::silu_mul_tcb(&mut tcb, &gate_buf, &up_buf, &act_buf, b * cols).unwrap();
    let w_bytes = rows * (cols / 256) * 144;
    kernels::gemm_q4_k_m_batched_v4r_predec_pinned_tcb(
        &mut tcb, &w_buf, 0, w_bytes, &sc_buf, 0, rows, cols, b, &act_buf, &y_buf,
    )
    .unwrap();
    tcb.commit_and_wait().unwrap();
    read_f32_buf(&y_buf, b * rows)
}
fn run_fused_v4r(
    ctx: &MetalContext,
    w_q4: &[u8],
    scales: &[f32],
    gate: &[f32],
    up: &[f32],
    rows: usize,
    cols: usize,
    b: usize,
) -> Vec<f32> {
    let w_buf = ctx.new_buffer_with_bytes(w_q4);
    let sc_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
    let gate_buf = new_f32_buf(ctx, gate);
    let up_buf = new_f32_buf(ctx, up);
    let y_buf = ctx.new_buffer(b * rows * 4);
    let mut tcb = TokenCommandBuffer::new(ctx);
    let w_bytes = rows * (cols / 256) * 144;
    kernels::gemm_q4_k_m_batched_v4r_predec_swiglu_pinned_tcb(
        &mut tcb, &w_buf, 0, w_bytes, &sc_buf, 0, rows, cols, b, &gate_buf, &up_buf, &y_buf,
    )
    .unwrap();
    tcb.commit_and_wait().unwrap();
    read_f32_buf(&y_buf, b * rows)
}
fn rand_vec(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(2654435761u32).wrapping_add(seed)) as f32;
            (x / u32::MAX as f32) * 4.0 - 2.0
        })
        .collect()
}
#[test]
fn swiglu_fused_v3w_matches_ref() {
    let ctx = ctx();
    let rows = 2048;
    let cols = 11008;
    let (w, scales) = make_q4k_predec_pm1(rows, cols, 0xABCD);
    for b in [5usize, 6, 7, 8] {
        let gate = rand_vec(b * cols, 0xDEAD + b as u32);
        let up = rand_vec(b * cols, 0xBEEF + b as u32);
        let ref_out = run_ref_v3w(ctx, &w, &scales, &gate, &up, rows, cols, b);
        let fused_out = run_fused_v3w(ctx, &w, &scales, &gate, &up, rows, cols, b);
        let max_diff = ref_out
            .iter()
            .zip(&fused_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "B={b}: v3w swiglu max_diff={max_diff} > atol 1e-4"
        );
    }
}
#[test]
fn swiglu_fused_v4r_matches_ref() {
    let ctx = ctx();
    let rows = 2048;
    let cols = 11008;
    let (w, scales) = make_q4k_predec_pm1(rows, cols, 0x1234);
    for b in [2usize, 3, 4, 5, 6, 7, 8] {
        let gate = rand_vec(b * cols, 0xCAFE + b as u32);
        let up = rand_vec(b * cols, 0xF00D + b as u32);
        let ref_out = run_ref_v4r(ctx, &w, &scales, &gate, &up, rows, cols, b);
        let fused_out = run_fused_v4r(ctx, &w, &scales, &gate, &up, rows, cols, b);
        let max_diff = ref_out
            .iter()
            .zip(&fused_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "B={b}: v4r swiglu max_diff={max_diff} > atol 1e-4"
        );
    }
}

#![cfg(target_os = "macos")]
//! Track 3.5 parity: SwiGLU-fused ffn_down must be bit-identical to
//! (silu_mul + separate ffn_down) for both v3w (B=5..8) and v4r (B=2..4) paths.
//!
//! Saves 1 dispatch/layer × 28 layers = 28 dispatches on Qwen-3B.
//! Parity gate: `atol = 0` (bit-identical, same arithmetic in same order).

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, TokenCommandBuffer};

mod common;
use common::*;

/// Run the reference path: silu_mul + separate v3w_predec ffn_down.
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
    let w_buf    = ctx.new_buffer_with_bytes(w_q4);
    let sc_buf   = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
    let gate_buf = new_f32_buf(ctx, gate);
    let up_buf   = new_f32_buf(ctx, up);
    let act_buf  = ctx.new_buffer(b * cols * 4);
    let y_buf    = ctx.new_buffer(b * rows * 4);

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::silu_mul_tcb(&mut tcb, &gate_buf, &up_buf, &act_buf, b * cols).unwrap();
    let w_bytes = rows * (cols / 256) * 144;
    kernels::gemm_q4_k_m_batched_v3w_predec_pinned_tcb(
        &mut tcb, &w_buf, 0, w_bytes, &sc_buf, 0, rows, cols, b, &act_buf, &y_buf,
    ).unwrap();
    tcb.commit_and_wait().unwrap();
    read_f32_buf(&y_buf, b * rows)
}

/// Run the fused path: swiglu v3w_predec ffn_down.
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
    let w_buf    = ctx.new_buffer_with_bytes(w_q4);
    let sc_buf   = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
    let gate_buf = new_f32_buf(ctx, gate);
    let up_buf   = new_f32_buf(ctx, up);
    let y_buf    = ctx.new_buffer(b * rows * 4);

    let mut tcb = TokenCommandBuffer::new(ctx);
    let w_bytes = rows * (cols / 256) * 144;
    kernels::gemm_q4_k_m_batched_v3w_predec_swiglu_pinned_tcb(
        &mut tcb, &w_buf, 0, w_bytes, &sc_buf, 0, rows, cols, b,
        &gate_buf, &up_buf, &y_buf,
    ).unwrap();
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
    let w_buf    = ctx.new_buffer_with_bytes(w_q4);
    let sc_buf   = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
    let gate_buf = new_f32_buf(ctx, gate);
    let up_buf   = new_f32_buf(ctx, up);
    let act_buf  = ctx.new_buffer(b * cols * 4);
    let y_buf    = ctx.new_buffer(b * rows * 4);

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::silu_mul_tcb(&mut tcb, &gate_buf, &up_buf, &act_buf, b * cols).unwrap();
    let w_bytes = rows * (cols / 256) * 144;
    kernels::gemm_q4_k_m_batched_v4r_predec_pinned_tcb(
        &mut tcb, &w_buf, 0, w_bytes, &sc_buf, 0, rows, cols, b, &act_buf, &y_buf,
    ).unwrap();
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
    let w_buf    = ctx.new_buffer_with_bytes(w_q4);
    let sc_buf   = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
    let gate_buf = new_f32_buf(ctx, gate);
    let up_buf   = new_f32_buf(ctx, up);
    let y_buf    = ctx.new_buffer(b * rows * 4);

    let mut tcb = TokenCommandBuffer::new(ctx);
    let w_bytes = rows * (cols / 256) * 144;
    kernels::gemm_q4_k_m_batched_v4r_predec_swiglu_pinned_tcb(
        &mut tcb, &w_buf, 0, w_bytes, &sc_buf, 0, rows, cols, b,
        &gate_buf, &up_buf, &y_buf,
    ).unwrap();
    tcb.commit_and_wait().unwrap();
    read_f32_buf(&y_buf, b * rows)
}

fn make_q4k_weights(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    let blocks_per_row = cols / 256;
    let block_bytes = 144;
    let total_bytes = rows * blocks_per_row * block_bytes;
    let w: Vec<u8> = (0..total_bytes).map(|i| ((i as u32).wrapping_mul(2246822519u32).wrapping_add(seed)) as u8).collect();
    // Predec scale table: 16 f32 per block (8 d,m pairs)
    let n_scales = rows * blocks_per_row * 16;
    let s: Vec<f32> = (0..n_scales).map(|i| {
        let v = ((i as u32).wrapping_mul(2654435761u32).wrapping_add(seed)) as f32 / u32::MAX as f32;
        v * 2.0 - 1.0
    }).collect();
    (w, s)
}

fn rand_vec(n: usize, seed: u32) -> Vec<f32> {
    (0..n).map(|i| {
        let x = ((i as u32).wrapping_mul(2654435761u32).wrapping_add(seed)) as f32;
        (x / u32::MAX as f32) * 4.0 - 2.0
    }).collect()
}

#[test]
fn swiglu_fused_v3w_matches_ref() {
    let ctx = ctx();
    // Qwen-3B-like: intermediate=11008, hidden=2048
    let rows = 2048; let cols = 11008;
    let (w, scales) = make_q4k_weights(rows, cols, 0xABCD);

    for b in [5usize, 6, 7, 8] {
        let gate = rand_vec(b * cols, 0xDEAD + b as u32);
        let up   = rand_vec(b * cols, 0xBEEF + b as u32);
        let ref_out   = run_ref_v3w(ctx, &w, &scales, &gate, &up, rows, cols, b);
        let fused_out = run_fused_v3w(ctx, &w, &scales, &gate, &up, rows, cols, b);
        let max_diff = ref_out.iter().zip(&fused_out).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "B={b}: v3w swiglu max_diff={max_diff} > atol 1e-4");
        eprintln!("v3w swiglu B={b}: max_diff={max_diff:.2e} OK");
    }
}

#[test]
fn swiglu_fused_v4r_matches_ref() {
    let ctx = ctx();
    let rows = 2048; let cols = 11008;
    let (w, scales) = make_q4k_weights(rows, cols, 0x1234);

    // Wave-R0: extended to B=5..8 to gate the DISMANTLE_QWEN_MULTISEQ_V4R_HIGHB
    // route (fused v4r swiglu must match its f32 ref on the ffn_down shape at high B).
    for b in [2usize, 3, 4, 5, 6, 7, 8] {
        let gate = rand_vec(b * cols, 0xCAFE + b as u32);
        let up   = rand_vec(b * cols, 0xF00D + b as u32);
        let ref_out   = run_ref_v4r(ctx, &w, &scales, &gate, &up, rows, cols, b);
        let fused_out = run_fused_v4r(ctx, &w, &scales, &gate, &up, rows, cols, b);
        let max_diff = ref_out.iter().zip(&fused_out).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "B={b}: v4r swiglu max_diff={max_diff} > atol 1e-4");
        eprintln!("v4r swiglu B={b}: max_diff={max_diff:.2e} OK");
    }
}

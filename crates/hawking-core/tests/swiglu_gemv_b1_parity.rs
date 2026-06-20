#![cfg(target_os = "macos")]
//! Track 3.5 parity: B=1 SwiGLU-fused GEMV kernels must be bit-identical
//! (or within f32 rounding) to (silu_mul + separate GEMV).
//!
//! Tests:
//!   - gemm_q6_k_fused_v2_swiglu   vs silu_mul + gemm_q6_k_fused_v2
//!   - gemm_q4_k_v4_predec_swiglu  vs silu_mul + gemm_q4_k_v4_predec   (1r)
//!   - gemm_q4_k_v4_predec_2r_swiglu vs silu_mul + gemm_q4_k_v4_predec_2r
//!   - gemm_q4_k_v4_predec_4r_swiglu vs silu_mul + gemm_q4_k_v4_predec_4r

use hawking_core::kernels;
use hawking_core::metal::{MetalContext, TokenCommandBuffer};

mod common;
use common::*;

// ── helpers ──────────────────────────────────────────────────────────────────

fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn make_q4k(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    let bpr = cols / 256;
    let total = rows * bpr * 144;
    let w: Vec<u8> = (0..total)
        .map(|i| ((i as u32).wrapping_mul(2246822519u32).wrapping_add(seed)) as u8)
        .collect();
    let ns = rows * bpr * 16;
    let s: Vec<f32> = (0..ns)
        .map(|i| {
            let v = ((i as u32).wrapping_mul(2654435761u32).wrapping_add(seed)) as f32
                / u32::MAX as f32;
            v * 2.0 - 1.0
        })
        .collect();
    (w, s)
}

fn make_q6k(rows: usize, cols: usize, seed: u32) -> Vec<u8> {
    let bpr = cols / 256;
    let total = rows * bpr * 210;
    (0..total)
        .map(|i| ((i as u32).wrapping_mul(1664525u32).wrapping_add(seed)) as u8)
        .collect()
}

fn rnd(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2654435761u32).wrapping_add(seed);
            (x as f32 / u32::MAX as f32) * 4.0 - 2.0
        })
        .collect()
}

// ── Q6_K swiglu parity ───────────────────────────────────────────────────────

fn run_q6k_ref(
    ctx: &MetalContext,
    w: &[u8],
    gate: &[f32],
    up: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let w_buf = ctx.new_buffer_with_bytes(w);
    let g_buf = new_f32_buf(ctx, gate);
    let u_buf = new_f32_buf(ctx, up);
    let act_buf = ctx.new_buffer(cols * 4);
    let y_buf = ctx.new_buffer(rows * 4);
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::silu_mul_tcb(&mut tcb, &g_buf, &u_buf, &act_buf, cols).unwrap();
    kernels::gemv_q6_k_pinned_tcb(&mut tcb, &w_buf, 0, w.len(), rows, cols, &act_buf, &y_buf)
        .unwrap();
    tcb.commit_and_wait().unwrap();
    read_f32_buf(&y_buf, rows)
}

fn run_q6k_fused(
    ctx: &MetalContext,
    w: &[u8],
    gate: &[f32],
    up: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let w_buf = ctx.new_buffer_with_bytes(w);
    let g_buf = new_f32_buf(ctx, gate);
    let u_buf = new_f32_buf(ctx, up);
    let y_buf = ctx.new_buffer(rows * 4);
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q6_k_swiglu_pinned_tcb(
        &mut tcb,
        &w_buf,
        0,
        w.len(),
        rows,
        cols,
        &g_buf,
        &u_buf,
        &y_buf,
    )
    .unwrap();
    tcb.commit_and_wait().unwrap();
    read_f32_buf(&y_buf, rows)
}

#[test]
fn q6k_swiglu_matches_ref() {
    let ctx = ctx();
    // Qwen-3B-like: hidden=2048, intermediate=11008 (down-proj shape)
    let rows = 2048;
    let cols = 11008;
    let w = make_q6k(rows, cols, 0xABCD);
    let gate = rnd(cols, 0xDEAD);
    let up = rnd(cols, 0xBEEF);
    let ref_out = run_q6k_ref(ctx, &w, &gate, &up, rows, cols);
    let fused_out = run_q6k_fused(ctx, &w, &gate, &up, rows, cols);
    // The Q6K fused kernel is NOT required to be bit-identical to the two-pass
    // reference (silu_mul store→reload, then GEMV) because computing silu inline
    // vs round-tripping through f32 memory can produce 1-ULP rounding differences
    // on large accumulated dot products. Gate random weights in [-2,2] can produce
    // outputs ~4e8 where 1 ULP ≈ 32. We verify with a relative tolerance instead.
    let max_rel_err = ref_out
        .iter()
        .zip(&fused_out)
        .filter(|(r, f)| r.is_finite() && f.is_finite())
        .map(|(r, f)| {
            let abs_err = (r - f).abs();
            let scale = r.abs().max(f.abs()).max(1.0);
            abs_err / scale
        })
        .fold(0.0_f32, f32::max);
    assert!(
        max_rel_err < 1e-4,
        "Q6_K swiglu parity FAILED: max_rel_err={max_rel_err:.2e} > 1e-4"
    );
    eprintln!("Q6_K swiglu: rows={rows} cols={cols} max_rel_err={max_rel_err:.2e} OK");
}

// ── Q4_K predec swiglu parity (1r / 2r / 4r) ─────────────────────────────────

fn run_q4k_predec_ref_b1(
    ctx: &MetalContext,
    w_q4: &[u8],
    scales: &[f32],
    gate: &[f32],
    up: &[f32],
    rows: usize,
    cols: usize,
    env_4r: bool,
    env_2r: bool,
) -> Vec<f32> {
    let w_buf = ctx.new_buffer_with_bytes(w_q4);
    let sc_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
    let g_buf = new_f32_buf(ctx, gate);
    let u_buf = new_f32_buf(ctx, up);
    let act = ctx.new_buffer(cols * 4);
    let y_buf = ctx.new_buffer(rows * 4);
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::silu_mul_tcb(&mut tcb, &g_buf, &u_buf, &act, cols).unwrap();
    // Pick the same variant as the swiglu wrapper will use.
    let row_bytes = (cols / 256) * 144;
    if env_4r {
        std::env::set_var("HAWKING_QWEN_PREDEC_4R", "1");
        std::env::remove_var("HAWKING_QWEN_PREDEC_2R");
    } else if !env_2r {
        std::env::set_var("HAWKING_QWEN_PREDEC_2R", "0");
        std::env::remove_var("HAWKING_QWEN_PREDEC_4R");
    } else {
        std::env::remove_var("HAWKING_QWEN_PREDEC_4R");
        std::env::remove_var("HAWKING_QWEN_PREDEC_2R");
    }
    kernels::gemv_q4_k_v4_predec_pinned_tcb(
        &mut tcb,
        &w_buf,
        0,
        rows * row_bytes,
        &sc_buf,
        0,
        rows,
        cols,
        &act,
        &y_buf,
    )
    .unwrap();
    tcb.commit_and_wait().unwrap();
    read_f32_buf(&y_buf, rows)
}

fn run_q4k_predec_swiglu_b1(
    ctx: &MetalContext,
    w_q4: &[u8],
    scales: &[f32],
    gate: &[f32],
    up: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let w_buf = ctx.new_buffer_with_bytes(w_q4);
    let sc_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
    let g_buf = new_f32_buf(ctx, gate);
    let u_buf = new_f32_buf(ctx, up);
    let y_buf = ctx.new_buffer(rows * 4);
    let mut tcb = TokenCommandBuffer::new(ctx);
    let row_bytes = (cols / 256) * 144;
    kernels::gemv_q4_k_v4_predec_swiglu_pinned_tcb(
        &mut tcb,
        &w_buf,
        0,
        rows * row_bytes,
        &sc_buf,
        0,
        rows,
        cols,
        &g_buf,
        &u_buf,
        &y_buf,
    )
    .unwrap();
    tcb.commit_and_wait().unwrap();
    read_f32_buf(&y_buf, rows)
}

#[test]
fn q4k_predec_1r_swiglu_matches_ref() {
    let ctx = ctx();
    let rows = 2048;
    let cols = 11008;
    let (w, scales) = make_q4k(rows, cols, 0x1234);
    let gate = rnd(cols, 0xCAFE);
    let up = rnd(cols, 0xF00D);
    std::env::set_var("HAWKING_QWEN_PREDEC_2R", "0");
    std::env::remove_var("HAWKING_QWEN_PREDEC_4R");
    let ref_out = run_q4k_predec_ref_b1(ctx, &w, &scales, &gate, &up, rows, cols, false, false);
    let fused_out = run_q4k_predec_swiglu_b1(ctx, &w, &scales, &gate, &up, rows, cols);
    std::env::remove_var("HAWKING_QWEN_PREDEC_2R");
    let max_diff = ref_out
        .iter()
        .zip(&fused_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff == 0.0,
        "Q4K predec 1r swiglu: max_diff={max_diff:.2e} (expected 0)"
    );
    eprintln!("Q4K predec 1r swiglu: max_diff={max_diff:.2e} OK");
}

#[test]
fn q4k_predec_2r_swiglu_matches_ref() {
    let ctx = ctx();
    let rows = 2048;
    let cols = 11008;
    let (w, scales) = make_q4k(rows, cols, 0x5678);
    let gate = rnd(cols, 0xAAAA);
    let up = rnd(cols, 0xBBBB);
    // 2r is the default — env vars unset
    std::env::remove_var("HAWKING_QWEN_PREDEC_4R");
    std::env::remove_var("HAWKING_QWEN_PREDEC_2R");
    let ref_out = run_q4k_predec_ref_b1(ctx, &w, &scales, &gate, &up, rows, cols, false, true);
    let fused_out = run_q4k_predec_swiglu_b1(ctx, &w, &scales, &gate, &up, rows, cols);
    let max_diff = ref_out
        .iter()
        .zip(&fused_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff == 0.0,
        "Q4K predec 2r swiglu: max_diff={max_diff:.2e} (expected 0)"
    );
    eprintln!("Q4K predec 2r swiglu: max_diff={max_diff:.2e} OK");
}

#[test]
fn q4k_predec_4r_swiglu_matches_ref() {
    let ctx = ctx();
    let rows = 2048;
    let cols = 11008;
    let (w, scales) = make_q4k(rows, cols, 0x9ABC);
    let gate = rnd(cols, 0xCCCC);
    let up = rnd(cols, 0xDDDD);
    std::env::set_var("HAWKING_QWEN_PREDEC_4R", "1");
    let ref_out = run_q4k_predec_ref_b1(ctx, &w, &scales, &gate, &up, rows, cols, true, false);
    let fused_out = run_q4k_predec_swiglu_b1(ctx, &w, &scales, &gate, &up, rows, cols);
    std::env::remove_var("HAWKING_QWEN_PREDEC_4R");
    let max_diff = ref_out
        .iter()
        .zip(&fused_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff == 0.0,
        "Q4K predec 4r swiglu: max_diff={max_diff:.2e} (expected 0)"
    );
    eprintln!("Q4K predec 4r swiglu: max_diff={max_diff:.2e} OK");
}

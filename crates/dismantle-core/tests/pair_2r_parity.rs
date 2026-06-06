#![cfg(target_os = "macos")]
//! Track A7 parity: `gemm_q4_k_v4_predec_pair_2r` must produce bit-identical
//! outputs to `gemm_q4_k_v4_predec_pair` (1r) for all production-relevant shapes.
//!
//! The 2r kernel amortises the activation x across 4 partial sums instead of 2
//! but uses identical FMA order per row, so outputs must match exactly.

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, TokenCommandBuffer};

mod common;
use common::*;

/// Make a synthetic Q4K predec weight buffer (rows × cols, 144 B/block).
fn make_q4k_predec(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    let bpr = cols / 256;
    let total_bytes = rows * bpr * 144;
    let w: Vec<u8> = (0..total_bytes)
        .map(|i| ((i as u32).wrapping_mul(2_246_822_519).wrapping_add(seed)) as u8)
        .collect();
    let n_scales = rows * bpr * 16;
    let s: Vec<f32> = (0..n_scales)
        .map(|i| {
            let v = ((i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed)) as f32
                / u32::MAX as f32;
            // Typical scale range: [-0.5, 0.5]
            v - 0.5
        })
        .collect();
    (w, s)
}

fn rand_vec(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(1_664_525).wrapping_add(seed);
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn run_pair_1r(
    ctx: &MetalContext,
    wg: &[u8],
    wu: &[u8],
    g_scales: &[f32],
    u_scales: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<f32>, Vec<f32>) {
    let _wg_buf = ctx.new_buffer_with_bytes(wg);
    let _wu_buf = ctx.new_buffer_with_bytes(wu);
    let gs_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(g_scales));
    let us_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(u_scales));
    let x_buf = new_f32_buf(ctx, x);
    let yg_buf = ctx.new_buffer(rows * 4);
    let yu_buf = ctx.new_buffer(rows * 4);

    // 1r pair uses shared model_buf with separate offsets; here gate and up are
    // separate buffers so we reuse wg_buf for both matrix pointers (gate at off=0,
    // up at off=0 of wu_buf). Use the wrapper's model_buf + offset convention:
    // gate: model_buf=wg_buf off=0, up: model_buf=wu_buf off=0.
    // The wrapper takes a single model_buf + two offsets into it. For the test
    // we need two distinct buffers (gate ≠ up), so call the raw dispatch twice
    // for 1r parity, or use a combined buffer with offsets.
    //
    // Simplest: build a combined buffer [gate_bytes || up_bytes] and set offsets.
    let w_bytes = rows * (cols / 256) * 144;
    let mut combined = Vec::with_capacity(wg.len() + wu.len());
    combined.extend_from_slice(wg);
    combined.extend_from_slice(wu);
    let combined_buf = ctx.new_buffer_with_bytes(&combined);

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q4_k_v4_predec_pair_pinned_tcb(
        &mut tcb,
        &combined_buf,
        0,
        w_bytes,
        &gs_buf,
        0,
        w_bytes,
        w_bytes,
        &us_buf,
        0,
        rows,
        cols,
        &x_buf,
        &yg_buf,
        &yu_buf,
    )
    .expect("1r pair dispatch");
    tcb.commit_and_wait().expect("1r pair wait");

    (read_f32_buf(&yg_buf, rows), read_f32_buf(&yu_buf, rows))
}

fn run_pair_2r(
    ctx: &MetalContext,
    wg: &[u8],
    wu: &[u8],
    g_scales: &[f32],
    u_scales: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<f32>, Vec<f32>) {
    let w_bytes = rows * (cols / 256) * 144;
    let mut combined = Vec::with_capacity(wg.len() + wu.len());
    combined.extend_from_slice(wg);
    combined.extend_from_slice(wu);
    let combined_buf = ctx.new_buffer_with_bytes(&combined);
    let gs_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(g_scales));
    let us_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(u_scales));
    let x_buf = new_f32_buf(ctx, x);
    let yg_buf = ctx.new_buffer(rows * 4);
    let yu_buf = ctx.new_buffer(rows * 4);

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q4_k_v4_predec_pair_2r_pinned_tcb(
        &mut tcb,
        &combined_buf,
        0,
        w_bytes,
        &gs_buf,
        0,
        w_bytes,
        w_bytes,
        &us_buf,
        0,
        rows,
        cols,
        &x_buf,
        &yg_buf,
        &yu_buf,
    )
    .expect("2r pair dispatch");
    tcb.commit_and_wait().expect("2r pair wait");

    (read_f32_buf(&yg_buf, rows), read_f32_buf(&yu_buf, rows))
}

#[test]
fn pair_2r_matches_pair_1r_multiple_shapes() {
    let ctx = ctx();

    // (rows, cols, seed) — rows=16 tests the boundary (2 TGs for 2r, each just
    // covers the 16-row stride). rows=48 tests the non-multiple-of-16 boundary.
    // rows=512, cols=2048 approximates production gate/up shapes at smaller scale.
    let cases: &[(usize, usize, u32)] = &[
        (16, 256, 0xA701),
        (32, 512, 0xA702),
        (48, 256, 0xA703), // non-multiple of 16: last TG processes 12 rows (has1 path)
        (128, 512, 0xA704),
        (512, 2048, 0xA705),
        (1024, 512, 0xA706),
    ];

    for &(rows, cols, seed) in cases {
        let (wg, g_scales) = make_q4k_predec(rows, cols, seed);
        let (wu, u_scales) = make_q4k_predec(rows, cols, seed ^ 0xFFFF);
        let x = rand_vec(cols, seed ^ 0x1234);

        let (ref_g, ref_u) = run_pair_1r(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);
        let (got_g, got_u) = run_pair_2r(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);

        let diff_g = max_abs_diff(&ref_g, &got_g);
        let diff_u = max_abs_diff(&ref_u, &got_u);
        assert_eq!(
            diff_g, 0.0,
            "rows={rows} cols={cols}: gate max_diff={diff_g:.2e} (must be 0)"
        );
        assert_eq!(
            diff_u, 0.0,
            "rows={rows} cols={cols}: up   max_diff={diff_u:.2e} (must be 0)"
        );
        eprintln!(
            "pair_2r rows={rows} cols={cols}: gate_diff={diff_g:.2e} up_diff={diff_u:.2e} OK"
        );
    }
}

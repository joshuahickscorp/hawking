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

fn run_pair_2r_inline(
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
    kernels::gemv_q4_k_v4_predec_pair_2r_inline_pinned_tcb(
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
    .expect("2r inline pair dispatch");
    tcb.commit_and_wait().expect("2r inline pair wait");

    (read_f32_buf(&yg_buf, rows), read_f32_buf(&yu_buf, rows))
}

fn run_pair_2r_inline_nox(
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
    kernels::gemv_q4_k_v4_predec_pair_2r_inline_nox_pinned_tcb(
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
    .expect("2r inline nox pair dispatch");
    tcb.commit_and_wait().expect("2r inline nox pair wait");

    (read_f32_buf(&yg_buf, rows), read_f32_buf(&yu_buf, rows))
}

fn run_pair_4r(
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
    kernels::gemv_q4_k_v4_predec_pair_4r_pinned_tcb(
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
    .expect("4r pair dispatch");
    tcb.commit_and_wait().expect("4r pair wait");

    (read_f32_buf(&yg_buf, rows), read_f32_buf(&yu_buf, rows))
}

fn run_pair_3r(
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
    kernels::gemv_q4_k_v4_predec_pair_3r_pinned_tcb(
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
    .expect("3r pair dispatch");
    tcb.commit_and_wait().expect("3r pair wait");

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

/// Track E3: `gemm_q4_k_v4_predec_pair_2r_inline` must be bit-identical to
/// `pair_2r`; only scale-load style changes.
#[test]
fn pair_2r_inline_matches_pair_2r_multiple_shapes() {
    let ctx = ctx();

    let cases: &[(usize, usize, u32)] = &[
        (16, 256, 0xE301),
        (17, 256, 0xE302),
        (32, 512, 0xE303),
        (48, 256, 0xE304),
        (128, 512, 0xE305),
        (512, 2048, 0xE306),
        (1024, 512, 0xE307),
    ];

    for &(rows, cols, seed) in cases {
        let (wg, g_scales) = make_q4k_predec(rows, cols, seed);
        let (wu, u_scales) = make_q4k_predec(rows, cols, seed ^ 0xBEEF);
        let x = rand_vec(cols, seed ^ 0x9876);

        let (ref_g, ref_u) = run_pair_2r(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);
        let (got_g, got_u) =
            run_pair_2r_inline(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);

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
            "pair_2r_inline rows={rows} cols={cols}: gate_diff={diff_g:.2e} up_diff={diff_u:.2e} OK"
        );
    }
}

/// Track F1: `gemm_q4_k_v4_predec_pair_2r_inline_nox` must be bit-identical to
/// `pair_2r` — it only drops the xl[8] activation preload (reads x per-pi), with
/// identical per-accumulator FMA order and identical x values (xl[k] was just
/// x[b*256 + k*32 + lane]). Boundary cases (17, 48) exercise the has1 OOB guard.
#[test]
fn pair_2r_inline_nox_matches_pair_2r_multiple_shapes() {
    let ctx = ctx();

    let cases: &[(usize, usize, u32)] = &[
        (16, 256, 0xF101),
        (17, 256, 0xF102),
        (32, 512, 0xF103),
        (48, 256, 0xF104),
        (128, 512, 0xF105),
        (512, 2048, 0xF106),
        (1024, 512, 0xF107),
        (11008, 2048, 0xF108), // production ffn gate/up shape
    ];

    for &(rows, cols, seed) in cases {
        let (wg, g_scales) = make_q4k_predec(rows, cols, seed);
        let (wu, u_scales) = make_q4k_predec(rows, cols, seed ^ 0xCAFE);
        let x = rand_vec(cols, seed ^ 0x4321);

        let (ref_g, ref_u) = run_pair_2r(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);
        let (got_g, got_u) =
            run_pair_2r_inline_nox(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);

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
            "pair_2r_inline_nox rows={rows} cols={cols}: gate_diff={diff_g:.2e} up_diff={diff_u:.2e} OK"
        );
    }
}

/// Track E2: `gemm_q4_k_v4_predec_pair_3r` must be bit-identical to `pair_2r`.
/// Boundary cases exercise the non-power-of-two row geometry: 24 rows/TG.
#[test]
fn pair_3r_matches_pair_2r_multiple_shapes() {
    let ctx = ctx();

    let cases: &[(usize, usize, u32)] = &[
        (24, 256, 0xE201), // exactly 1 TG for 3r
        (25, 256, 0xE202), // 2 TGs, second has 1 row
        (32, 256, 0xE203), // second TG has 8 rows
        (33, 512, 0xE204), // second TG has 9 rows
        (48, 512, 0xE205), // exactly 2 TGs
        (128, 512, 0xE206),
        (512, 2048, 0xE207),
        (1024, 512, 0xE208),
    ];

    for &(rows, cols, seed) in cases {
        let (wg, g_scales) = make_q4k_predec(rows, cols, seed);
        let (wu, u_scales) = make_q4k_predec(rows, cols, seed ^ 0xDEAD);
        let x = rand_vec(cols, seed ^ 0x5678);

        let (ref_g, ref_u) = run_pair_2r(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);
        let (got_g, got_u) = run_pair_3r(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);

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
            "pair_3r rows={rows} cols={cols}: gate_diff={diff_g:.2e} up_diff={diff_u:.2e} OK"
        );
    }
}

/// Track B2: `gemm_q4_k_v4_predec_pair_4r` must be bit-identical to `pair_2r`
/// (same per-accumulator FMA order, only scale access style differs: inline
/// vs preloaded).  Also includes a boundary case where rows is not a multiple
/// of 32 to exercise the `has1/has2/has3` out-of-bounds guards.
#[test]
fn pair_4r_matches_pair_2r_multiple_shapes() {
    let ctx = ctx();

    // (rows, cols, seed) — rows=48 tests non-multiple of 32 (last TG covers
    // only 16 rows → has2/has3 are false for simd_id 2..7, verifying guards).
    // rows=11008 approximates the production ffn gate/up shape on Qwen2.5-3B.
    let cases: &[(usize, usize, u32)] = &[
        (32, 256, 0xB201),
        (48, 256, 0xB202), // non-multiple of 32: tests has1/has2/has3 boundary
        (64, 512, 0xB203),
        (128, 512, 0xB204),
        (512, 2048, 0xB205),
        (1024, 512, 0xB206),
    ];

    for &(rows, cols, seed) in cases {
        let (wg, g_scales) = make_q4k_predec(rows, cols, seed);
        let (wu, u_scales) = make_q4k_predec(rows, cols, seed ^ 0xDEAD);
        let x = rand_vec(cols, seed ^ 0x5678);

        let (ref_g, ref_u) = run_pair_2r(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);
        let (got_g, got_u) = run_pair_4r(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);

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
            "pair_4r rows={rows} cols={cols}: gate_diff={diff_g:.2e} up_diff={diff_u:.2e} OK"
        );
    }
}

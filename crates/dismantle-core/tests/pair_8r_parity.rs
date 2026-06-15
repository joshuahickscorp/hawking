#![cfg(target_os = "macos")]
//! Track E1 parity: `gemm_q4_k_v4_predec_pair_8r` must produce outputs
//! matching `gemm_q4_k_v4_predec_pair_4r` (the default-on reference) within
//! a tight tolerance.
//!
//! The 8r kernel handles 8 rows per simdgroup (64 rows/TG) vs 4 rows for the
//! 4r kernel (32 rows/TG), halving TG count again for the Qwen-3B gate+up
//! shape (11008 rows → 172 TGs vs 344). Both kernels use the same per-element
//! FMA path; the only difference is loop unrolling depth and row-stride. FMA
//! reordering between 4 and 8 independent accumulators may differ by ~2 ULPs;
//! we gate at 1e-5 to allow that while catching any structural error.

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, TokenCommandBuffer};

mod common;
use common::*;

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

/// Run gate+up pair with 4r kernel (reference for E1).
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

/// Run gate+up pair with 8r kernel (Track E1).
fn run_pair_8r(
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
    kernels::gemv_q4_k_v4_predec_pair_8r_pinned_tcb(
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
    .expect("8r pair dispatch");
    tcb.commit_and_wait().expect("8r pair wait");
    (read_f32_buf(&yg_buf, rows), read_f32_buf(&yu_buf, rows))
}

/// E1 quality gate: 8r must agree with 4r within 1e-5.
/// 8 independent FMA chains (gate0-7 + up0-7) may reorder vs 4 chains in 4r,
/// producing ULP-level differences. 1e-5 is tight enough to catch any structural
/// error (wrong row index, scale misread, missing accumulator) while tolerating
/// FMA reassociation.
///
/// Key boundary cases:
///  - rows=64:  exactly 1 TG (8 simdgroups × 8 rows)
///  - rows=65:  2 TGs; second TG has 1 active row (tests has1-7 all false for simd_id>0)
///  - rows=72:  2 TGs; second TG has 8 rows (simd_id=0 only has row0, has1-7 false for rest)
///  - rows=128: exactly 2 TGs
///  - rows=512, 1024: larger shapes
#[test]
fn e1_pair_8r_matches_pair_4r() {
    let ctx = ctx();
    // max_abs_diff gate: 1e-5 allows FMA reorder ULPs, rejects any structural bug
    const MAX_DIFF: f32 = 1e-5;

    let cases: &[(usize, usize, u32)] = &[
        (64, 256, 0xE101),   // exactly 1 TG
        (65, 256, 0xE102),   // 2 TGs, second has 1 row: tests has1-7 guards for simd_id>0
        (72, 256, 0xE103),   // 2 TGs, second has 8 rows: tests has1-7 for simd_id=0
        (128, 256, 0xE104),  // exactly 2 TGs
        (256, 512, 0xE105),  // 4 TGs
        (512, 2048, 0xE106), // larger shape
        (1024, 512, 0xE107), // 16 TGs
    ];

    for &(rows, cols, seed) in cases {
        let (wg, g_scales) = make_q4k_predec(rows, cols, seed);
        let (wu, u_scales) = make_q4k_predec(rows, cols, seed ^ 0xDEAD);
        let x = rand_vec(cols, seed ^ 0x5678);

        let (ref_g, ref_u) = run_pair_4r(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);
        let (got_g, got_u) = run_pair_8r(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);

        let diff_g = max_abs_diff(&ref_g, &got_g);
        let diff_u = max_abs_diff(&ref_u, &got_u);
        assert!(
            diff_g <= MAX_DIFF,
            "E1 rows={rows} cols={cols}: gate max_diff={diff_g:.2e} > {MAX_DIFF:.2e}"
        );
        assert!(
            diff_u <= MAX_DIFF,
            "E1 rows={rows} cols={cols}: up   max_diff={diff_u:.2e} > {MAX_DIFF:.2e}"
        );
        eprintln!(
            "E1 pair_8r rows={rows} cols={cols}: gate_diff={diff_g:.2e} up_diff={diff_u:.2e} OK"
        );
    }
}

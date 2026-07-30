#![cfg(target_os = "macos")]
use hawking_core::kernels;
use hawking_core::metal::MetalContext;
mod common;
use common::*;
fn run_pair(
    ctx: &MetalContext,
    wg: &[u8],
    wu: &[u8],
    g_scales: &[f32],
    u_scales: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
    is_8r: bool,
) -> (Vec<f32>, Vec<f32>) {
    run_predec_pair_combined(
        ctx,
        wg,
        wu,
        g_scales,
        u_scales,
        x,
        rows,
        cols,
        |tcb, c, wb, gs, us, xb, yg, yu| {
            let r = if is_8r {
                kernels::gemv_q4_k_v4_predec_pair_8r_pinned_tcb(
                    tcb, c, 0, wb, gs, 0, wb, wb, us, 0, rows, cols, xb, yg, yu,
                )
            } else {
                kernels::gemv_q4_k_v4_predec_pair_4r_pinned_tcb(
                    tcb, c, 0, wb, gs, 0, wb, wb, us, 0, rows, cols, xb, yg, yu,
                )
            };
            r.expect("pair dispatch");
        },
    )
}
#[test]
fn e1_pair_8r_matches_pair_4r() {
    let ctx = ctx();
    const MAX_DIFF: f32 = 1e-5;
    let cases: &[(usize, usize, u32)] = &[
        (64, 256, 0xE101),
        (65, 256, 0xE102),
        (72, 256, 0xE103),
        (128, 256, 0xE104),
        (256, 512, 0xE105),
        (512, 2048, 0xE106),
        (1024, 512, 0xE107),
    ];
    for &(rows, cols, seed) in cases {
        let (wg, g_scales) = make_q4k_predec_pm05(rows, cols, seed);
        let (wu, u_scales) = make_q4k_predec_pm05(rows, cols, seed ^ 0xDEAD);
        let x = lcg_f32(cols, seed ^ 0x5678, -1.0, 1.0);
        let (ref_g, ref_u) = run_pair(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols, false);
        let (got_g, got_u) = run_pair(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols, true);
        let diff_g = max_abs_diff(&ref_g, &got_g);
        let diff_u = max_abs_diff(&ref_u, &got_u);
        assert!(
            diff_g <= MAX_DIFF,
            "E1 rows={rows} cols={cols}: gate max_diff={diff_g:.2e}"
        );
        assert!(
            diff_u <= MAX_DIFF,
            "E1 rows={rows} cols={cols}: up max_diff={diff_u:.2e}"
        );
    }
}

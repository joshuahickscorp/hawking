#![cfg(target_os = "macos")]
use hawking_core::kernels;
use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
mod common;
use common::*;
type PairOut = (Vec<f32>, Vec<f32>);
fn run_pair(
    ctx: &MetalContext,
    wg: &[u8],
    wu: &[u8],
    g_scales: &[f32],
    u_scales: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
    dispatch: impl FnOnce(
        &mut TokenCommandBuffer,
        &PinnedBuffer,
        usize,
        &PinnedBuffer,
        &PinnedBuffer,
        &PinnedBuffer,
        &PinnedBuffer,
        &PinnedBuffer,
    ),
) -> PairOut {
    run_predec_pair_combined(ctx, wg, wu, g_scales, u_scales, x, rows, cols, dispatch)
}
fn run_1r(
    ctx: &MetalContext,
    wg: &[u8],
    wu: &[u8],
    gs: &[f32],
    us: &[f32],
    x: &[f32],
    r: usize,
    c: usize,
) -> PairOut {
    run_pair(
        ctx,
        wg,
        wu,
        gs,
        us,
        x,
        r,
        c,
        |tcb, m, wb, gsb, usb, xb, yg, yu| {
            kernels::gemv_q4_k_v4_predec_pair_pinned_tcb(
                tcb, m, 0, wb, gsb, 0, wb, wb, usb, 0, r, c, xb, yg, yu,
            )
            .expect("1r");
        },
    )
}
fn run_2r(
    ctx: &MetalContext,
    wg: &[u8],
    wu: &[u8],
    gs: &[f32],
    us: &[f32],
    x: &[f32],
    r: usize,
    c: usize,
) -> PairOut {
    run_pair(
        ctx,
        wg,
        wu,
        gs,
        us,
        x,
        r,
        c,
        |tcb, m, wb, gsb, usb, xb, yg, yu| {
            kernels::gemv_q4_k_v4_predec_pair_2r_pinned_tcb(
                tcb, m, 0, wb, gsb, 0, wb, wb, usb, 0, r, c, xb, yg, yu,
            )
            .expect("2r");
        },
    )
}
fn run_2r_inline(
    ctx: &MetalContext,
    wg: &[u8],
    wu: &[u8],
    gs: &[f32],
    us: &[f32],
    x: &[f32],
    r: usize,
    c: usize,
) -> PairOut {
    run_pair(
        ctx,
        wg,
        wu,
        gs,
        us,
        x,
        r,
        c,
        |tcb, m, wb, gsb, usb, xb, yg, yu| {
            kernels::gemv_q4_k_v4_predec_pair_2r_inline_pinned_tcb(
                tcb, m, 0, wb, gsb, 0, wb, wb, usb, 0, r, c, xb, yg, yu,
            )
            .expect("2r_inline");
        },
    )
}
fn run_2r_inline_nox(
    ctx: &MetalContext,
    wg: &[u8],
    wu: &[u8],
    gs: &[f32],
    us: &[f32],
    x: &[f32],
    r: usize,
    c: usize,
) -> PairOut {
    run_pair(
        ctx,
        wg,
        wu,
        gs,
        us,
        x,
        r,
        c,
        |tcb, m, wb, gsb, usb, xb, yg, yu| {
            kernels::gemv_q4_k_v4_predec_pair_2r_inline_nox_pinned_tcb(
                tcb, m, 0, wb, gsb, 0, wb, wb, usb, 0, r, c, xb, yg, yu,
            )
            .expect("2r_inline_nox");
        },
    )
}
fn run_3r(
    ctx: &MetalContext,
    wg: &[u8],
    wu: &[u8],
    gs: &[f32],
    us: &[f32],
    x: &[f32],
    r: usize,
    c: usize,
) -> PairOut {
    run_pair(
        ctx,
        wg,
        wu,
        gs,
        us,
        x,
        r,
        c,
        |tcb, m, wb, gsb, usb, xb, yg, yu| {
            kernels::gemv_q4_k_v4_predec_pair_3r_pinned_tcb(
                tcb, m, 0, wb, gsb, 0, wb, wb, usb, 0, r, c, xb, yg, yu,
            )
            .expect("3r");
        },
    )
}
fn run_4r(
    ctx: &MetalContext,
    wg: &[u8],
    wu: &[u8],
    gs: &[f32],
    us: &[f32],
    x: &[f32],
    r: usize,
    c: usize,
) -> PairOut {
    run_pair(
        ctx,
        wg,
        wu,
        gs,
        us,
        x,
        r,
        c,
        |tcb, m, wb, gsb, usb, xb, yg, yu| {
            kernels::gemv_q4_k_v4_predec_pair_4r_pinned_tcb(
                tcb, m, 0, wb, gsb, 0, wb, wb, usb, 0, r, c, xb, yg, yu,
            )
            .expect("4r");
        },
    )
}
fn assert_bit_identical(
    label: &str,
    cases: &[(usize, usize, u32)],
    seed_up_xor: u32,
    seed_x_xor: u32,
    reference: fn(&MetalContext, &[u8], &[u8], &[f32], &[f32], &[f32], usize, usize) -> PairOut,
    candidate: fn(&MetalContext, &[u8], &[u8], &[f32], &[f32], &[f32], usize, usize) -> PairOut,
) {
    let ctx = ctx();
    for &(rows, cols, seed) in cases {
        let (wg, g_scales) = make_q4k_predec_pm05(rows, cols, seed);
        let (wu, u_scales) = make_q4k_predec_pm05(rows, cols, seed ^ seed_up_xor);
        let x = lcg_f32(cols, seed ^ seed_x_xor, -1.0, 1.0);
        let (ref_g, ref_u) = reference(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);
        let (got_g, got_u) = candidate(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);
        let diff_g = max_abs_diff(&ref_g, &got_g);
        let diff_u = max_abs_diff(&ref_u, &got_u);
        assert_eq!(
            diff_g, 0.0,
            "{label} rows={rows} cols={cols}: gate max_diff={diff_g:.2e}"
        );
        assert_eq!(
            diff_u, 0.0,
            "{label} rows={rows} cols={cols}: up max_diff={diff_u:.2e}"
        );
    }
}
#[test]
fn pair_2r_matches_pair_1r_multiple_shapes() {
    assert_bit_identical(
        "pair_2r",
        &[
            (16, 256, 0xA701),
            (32, 512, 0xA702),
            (48, 256, 0xA703),
            (128, 512, 0xA704),
            (512, 2048, 0xA705),
            (1024, 512, 0xA706),
        ],
        0xFFFF,
        0x1234,
        run_1r,
        run_2r,
    );
}
#[test]
fn pair_2r_inline_matches_pair_2r_multiple_shapes() {
    assert_bit_identical(
        "pair_2r_inline",
        &[
            (16, 256, 0xE301),
            (17, 256, 0xE302),
            (32, 512, 0xE303),
            (48, 256, 0xE304),
            (128, 512, 0xE305),
            (512, 2048, 0xE306),
            (1024, 512, 0xE307),
        ],
        0xBEEF,
        0x9876,
        run_2r,
        run_2r_inline,
    );
}
#[test]
fn pair_2r_inline_nox_matches_pair_2r_multiple_shapes() {
    assert_bit_identical(
        "pair_2r_inline_nox",
        &[
            (16, 256, 0xF101),
            (17, 256, 0xF102),
            (32, 512, 0xF103),
            (48, 256, 0xF104),
            (128, 512, 0xF105),
            (512, 2048, 0xF106),
            (1024, 512, 0xF107),
            (11008, 2048, 0xF108),
        ],
        0xCAFE,
        0x4321,
        run_2r,
        run_2r_inline_nox,
    );
}
#[test]
fn pair_3r_matches_pair_2r_multiple_shapes() {
    assert_bit_identical(
        "pair_3r",
        &[
            (24, 256, 0xE201),
            (25, 256, 0xE202),
            (32, 256, 0xE203),
            (33, 512, 0xE204),
            (48, 512, 0xE205),
            (128, 512, 0xE206),
            (512, 2048, 0xE207),
            (1024, 512, 0xE208),
        ],
        0xDEAD,
        0x5678,
        run_2r,
        run_3r,
    );
}
#[test]
fn pair_4r_matches_pair_2r_multiple_shapes() {
    assert_bit_identical(
        "pair_4r",
        &[
            (32, 256, 0xB201),
            (48, 256, 0xB202),
            (64, 512, 0xB203),
            (128, 512, 0xB204),
            (512, 2048, 0xB205),
            (1024, 512, 0xB206),
        ],
        0xDEAD,
        0x5678,
        run_2r,
        run_4r,
    );
}

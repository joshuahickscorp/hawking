#![cfg(target_os = "macos")]
use half::f16;
use hawking_core::kernels;
use hawking_core::metal::{MetalContext, TokenCommandBuffer};
mod common;
use common::*;
fn make_q4k_predec(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    let bpr = cols / 256;
    let w: Vec<u8> = (0..rows * bpr * 144)
        .map(|i| ((i as u32).wrapping_mul(2246822519).wrapping_add(seed)) as u8)
        .collect();
    let s: Vec<f32> = (0..rows * bpr * 16)
        .map(|i| {
            let v = ((i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(seed ^ 0xAB)) as f32
                / u32::MAX as f32;
            0.1 + v * 1.9
        })
        .collect();
    (w, s)
}
fn f32_to_f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| f16::from_f32(x).to_le_bytes())
        .collect()
}
fn rel_l2(reference: &[f32], got: &[f32]) -> f64 {
    let num: f64 = reference
        .iter()
        .zip(got)
        .map(|(&r, &g)| ((r - g) as f64).powi(2))
        .sum();
    let den: f64 = reference
        .iter()
        .map(|&r| (r as f64).powi(2))
        .sum::<f64>()
        .max(1e-30);
    (num / den).sqrt()
}
fn run_2r_add_f32(
    ctx: &MetalContext,
    w: &[u8],
    scales: &[f32],
    x: &[f32],
    residual: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let w_buf = ctx.new_buffer_with_bytes(w);
    let sc_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
    let x_buf = new_f32_buf(ctx, x);
    let res_buf = new_f32_buf(ctx, residual);
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q4_k_v4_predec_2r_add_pinned_tcb(
        &mut tcb,
        &w_buf,
        0,
        w.len(),
        &sc_buf,
        0,
        rows,
        cols,
        &x_buf,
        &res_buf,
    )
    .unwrap();
    tcb.commit_and_wait().unwrap();
    read_f32_buf(&res_buf, rows)
}
fn run_2r_add_f16s(
    ctx: &MetalContext,
    w: &[u8],
    scales_f16: &[u8],
    x: &[f32],
    residual: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let w_buf = ctx.new_buffer_with_bytes(w);
    let sc_buf = ctx.new_buffer_with_bytes(scales_f16);
    let x_buf = new_f32_buf(ctx, x);
    let res_buf = new_f32_buf(ctx, residual);
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q4_k_v4_predec_2r_add_f16s_pinned_tcb(
        &mut tcb,
        &w_buf,
        0,
        w.len(),
        &sc_buf,
        0,
        rows,
        cols,
        &x_buf,
        &res_buf,
    )
    .unwrap();
    tcb.commit_and_wait().unwrap();
    read_f32_buf(&res_buf, rows)
}
fn run_4r_add_f32(
    ctx: &MetalContext,
    w: &[u8],
    scales: &[f32],
    x: &[f32],
    residual: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let w_buf = ctx.new_buffer_with_bytes(w);
    let sc_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
    let x_buf = new_f32_buf(ctx, x);
    let res_buf = new_f32_buf(ctx, residual);
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q4_k_v4_predec_4r_add_pinned_tcb(
        &mut tcb,
        &w_buf,
        0,
        w.len(),
        &sc_buf,
        0,
        rows,
        cols,
        &x_buf,
        &res_buf,
    )
    .unwrap();
    tcb.commit_and_wait().unwrap();
    read_f32_buf(&res_buf, rows)
}
fn run_4r_add_f16s(
    ctx: &MetalContext,
    w: &[u8],
    scales_f16: &[u8],
    x: &[f32],
    residual: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let w_buf = ctx.new_buffer_with_bytes(w);
    let sc_buf = ctx.new_buffer_with_bytes(scales_f16);
    let x_buf = new_f32_buf(ctx, x);
    let res_buf = new_f32_buf(ctx, residual);
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q4_k_v4_predec_4r_add_f16s_pinned_tcb(
        &mut tcb,
        &w_buf,
        0,
        w.len(),
        &sc_buf,
        0,
        rows,
        cols,
        &x_buf,
        &res_buf,
    )
    .unwrap();
    tcb.commit_and_wait().unwrap();
    read_f32_buf(&res_buf, rows)
}
#[test]
fn d6_predec_2r_add_f16s_quality_gate() {
    let ctx = ctx();
    const MAX_REL_L2: f64 = 1e-2;
    let cases: &[(usize, usize, u32)] = &[
        (2048, 2048, 0xD600),
        (2048, 2048, 0xD601),
        (33, 256, 0xD602),
        (48, 256, 0xD603),
        (512, 2048, 0xD604),
    ];
    for &(rows, cols, seed) in cases {
        let (w, sc) = make_q4k_predec(rows, cols, seed);
        let sc_f16 = f32_to_f16_bytes(&sc);
        let x: Vec<f32> = (0..cols)
            .map(|i| {
                ((i as u32).wrapping_mul(1664525).wrapping_add(seed) as f32 / u32::MAX as f32) * 2.0
                    - 1.0
            })
            .collect();
        let residual: Vec<f32> = (0..rows)
            .map(|i| {
                ((i as u32).wrapping_mul(1103515245).wrapping_add(seed) as f32 / u32::MAX as f32)
                    * 2.0
                    - 1.0
            })
            .collect();
        let ref_res = run_2r_add_f32(ctx, &w, &sc, &x, &residual, rows, cols);
        let got_res = run_2r_add_f16s(ctx, &w, &sc_f16, &x, &residual, rows, cols);
        let r = rel_l2(&ref_res, &got_res);
        assert!(
            r < MAX_REL_L2,
            "2r_add_f16s rows={rows} cols={cols}: rel_L2={r:.4e} >= {MAX_REL_L2}"
        );
    }
}
#[test]
fn d6_predec_4r_add_f16s_quality_gate() {
    let ctx = ctx();
    const MAX_REL_L2: f64 = 1e-2;
    const MAX_CROSS_F16: f64 = 2e-5; // 2r_f16s vs 4r_f16s should be near-identical
    let cases: &[(usize, usize, u32)] = &[
        (2048, 2048, 0xD610),
        (2048, 2048, 0xD611),
        (33, 256, 0xD612),
        (49, 256, 0xD613),
        (512, 2048, 0xD614),
    ];
    for &(rows, cols, seed) in cases {
        let (w, sc) = make_q4k_predec(rows, cols, seed);
        let sc_f16 = f32_to_f16_bytes(&sc);
        let x: Vec<f32> = (0..cols)
            .map(|i| {
                ((i as u32).wrapping_mul(1664525).wrapping_add(seed) as f32 / u32::MAX as f32) * 2.0
                    - 1.0
            })
            .collect();
        let residual: Vec<f32> = (0..rows)
            .map(|i| {
                ((i as u32).wrapping_mul(1103515245).wrapping_add(seed) as f32 / u32::MAX as f32)
                    * 2.0
                    - 1.0
            })
            .collect();
        let ref_res = run_4r_add_f32(ctx, &w, &sc, &x, &residual, rows, cols);
        let got_4r = run_4r_add_f16s(ctx, &w, &sc_f16, &x, &residual, rows, cols);
        let got_2r = run_2r_add_f16s(ctx, &w, &sc_f16, &x, &residual, rows, cols);
        let r4f = rel_l2(&ref_res, &got_4r);
        let cross = rel_l2(&got_2r, &got_4r);
        assert!(
            r4f < MAX_REL_L2,
            "4r_add_f16s rows={rows} cols={cols}: rel_L2 vs f32={r4f:.4e} >= {MAX_REL_L2}"
        );
        assert!(
            cross < MAX_CROSS_F16,
            "2r_f16s vs 4r_f16s rows={rows} cols={cols}: cross={cross:.4e} >= {MAX_CROSS_F16}"
        );
    }
}

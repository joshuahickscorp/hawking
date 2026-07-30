#![cfg(target_os = "macos")]
use hawking_core::kernels;
use hawking_core::metal::{MetalContext, TokenCommandBuffer};
mod common;
use common::*;
fn positive_vec(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(1_103_515_245).wrapping_add(seed);
            0.5 + (x as f32 / u32::MAX as f32)
        })
        .collect()
}
fn run_reference(
    ctx: &MetalContext,
    w_q4: &[u8],
    scales: &[f32],
    attn_out: &[f32],
    residual: &[f32],
    norm_weight: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let w_buf = ctx.new_buffer_with_bytes(w_q4);
    let scales_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
    let attn_buf = new_f32_buf(ctx, attn_out);
    let residual_buf = new_f32_buf(ctx, residual);
    let norm_buf = new_f32_buf(ctx, norm_weight);
    let oproj_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let x_norm_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q4_k_v4_predec_pinned_tcb(
        &mut tcb,
        &w_buf,
        0,
        w_q4.len(),
        &scales_buf,
        0,
        rows,
        cols,
        &attn_buf,
        &oproj_buf,
    )
    .expect("reference o_proj predec GEMV");
    kernels::add_rmsnorm_fused_tcb(
        &mut tcb,
        &residual_buf,
        &oproj_buf,
        &norm_buf,
        &x_norm_buf,
        eps,
        rows,
    )
    .expect("reference add_rmsnorm");
    tcb.commit_and_wait().expect("reference commit");
    (
        read_f32_buf(&residual_buf, rows),
        read_f32_buf(&x_norm_buf, rows),
    )
}
fn run_fused(
    ctx: &MetalContext,
    w_q4: &[u8],
    scales: &[f32],
    attn_out: &[f32],
    residual: &[f32],
    norm_weight: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let w_buf = ctx.new_buffer_with_bytes(w_q4);
    let scales_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
    let attn_buf = new_f32_buf(ctx, attn_out);
    let residual_buf = new_f32_buf(ctx, residual);
    let norm_buf = new_f32_buf(ctx, norm_weight);
    let x_norm_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q4_k_v4_predec_2r_add_rmsnorm_tcb(
        &mut tcb,
        &w_buf,
        0,
        w_q4.len(),
        &scales_buf,
        0,
        rows,
        cols,
        &attn_buf,
        &residual_buf,
        &norm_buf,
        &x_norm_buf,
        eps,
    )
    .expect("fused o_proj add rmsnorm");
    tcb.commit_and_wait().expect("fused commit");
    (
        read_f32_buf(&residual_buf, rows),
        read_f32_buf(&x_norm_buf, rows),
    )
}
fn run_2r_add_direct(
    ctx: &MetalContext,
    w_q4: &[u8],
    scales: &[f32],
    x: &[f32],
    residual: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let w_buf = ctx.new_buffer_with_bytes(w_q4);
    let scales_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
    let x_buf = new_f32_buf(ctx, x);
    let res_buf = new_f32_buf(ctx, residual);
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q4_k_v4_predec_2r_add_pinned_tcb(
        &mut tcb,
        &w_buf,
        0,
        w_q4.len(),
        &scales_buf,
        0,
        rows,
        cols,
        &x_buf,
        &res_buf,
    )
    .expect("2r_add dispatch");
    tcb.commit_and_wait().expect("2r_add wait");
    read_f32_buf(&res_buf, rows)
}
fn run_4r_add_direct(
    ctx: &MetalContext,
    w_q4: &[u8],
    scales: &[f32],
    x: &[f32],
    residual: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let w_buf = ctx.new_buffer_with_bytes(w_q4);
    let scales_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
    let x_buf = new_f32_buf(ctx, x);
    let res_buf = new_f32_buf(ctx, residual);
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q4_k_v4_predec_4r_add_pinned_tcb(
        &mut tcb,
        &w_buf,
        0,
        w_q4.len(),
        &scales_buf,
        0,
        rows,
        cols,
        &x_buf,
        &res_buf,
    )
    .expect("4r_add dispatch");
    tcb.commit_and_wait().expect("4r_add wait");
    read_f32_buf(&res_buf, rows)
}
#[test]
fn q4k_predec_4r_add_matches_2r_add() {
    let ctx = ctx();
    let cases: &[(usize, usize, u32)] = &[
        (32, 256, 0xB400),
        (48, 256, 0xB401), // non-multiple of 32 — tests has1/has2/has3 guards
        (64, 512, 0xB402),
        (128, 512, 0xB403),
        (512, 2048, 0xB404),
        (1024, 512, 0xB405),
    ];
    for &(rows, cols, seed) in cases {
        let (w_q4, scales) = make_q4k_predec_pm025(rows, cols, seed);
        let x = lcg_f32(cols, seed ^ 0x9000, -1.0, 1.0);
        let residual = lcg_f32(rows, seed ^ 0xA000, -1.0, 1.0);
        let ref_res = run_2r_add_direct(ctx, &w_q4, &scales, &x, &residual, rows, cols);
        let got_res = run_4r_add_direct(ctx, &w_q4, &scales, &x, &residual, rows, cols);
        let diff = max_abs_diff(&ref_res, &got_res);
        assert_eq!(
            diff, 0.0,
            "rows={rows} cols={cols}: max_diff={diff:.2e} (must be 0)"
        );
    }
}
#[test]
fn q4k_predec_2r_oproj_add_rmsnorm_matches_reference() {
    std::env::remove_var("HAWKING_QWEN_PREDEC_2R");
    std::env::set_var("HAWKING_QWEN_PREDEC_4R", "0");
    let ctx = ctx();
    let eps = 1e-6_f32;
    for (rows, cols, seed) in [(256usize, 512usize, 0xA11C_E001), (130, 768, 0xA11C_E002)] {
        let (w_q4, scales) = make_q4k_predec_pm025(rows, cols, seed);
        let attn_out = lcg_f32(cols, seed ^ 0x1111, -1.0, 1.0);
        let residual = lcg_f32(rows, seed ^ 0x2222, -1.0, 1.0);
        let norm_weight = positive_vec(rows, seed ^ 0x3333);
        let (ref_residual, ref_norm) = run_reference(
            ctx,
            &w_q4,
            &scales,
            &attn_out,
            &residual,
            &norm_weight,
            rows,
            cols,
            eps,
        );
        let (fused_residual, fused_norm) = run_fused(
            ctx,
            &w_q4,
            &scales,
            &attn_out,
            &residual,
            &norm_weight,
            rows,
            cols,
            eps,
        );
        let residual_diff = max_abs_diff(&ref_residual, &fused_residual);
        let norm_diff = max_abs_diff(&ref_norm, &fused_norm);
        assert_eq!(
            residual_diff, 0.0,
            "rows={rows} cols={cols}: residual max_diff={residual_diff:.2e}"
        );
        assert_eq!(
            norm_diff, 0.0,
            "rows={rows} cols={cols}: x_norm max_diff={norm_diff:.2e}"
        );
    }
}

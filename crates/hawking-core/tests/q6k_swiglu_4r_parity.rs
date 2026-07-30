#![cfg(target_os = "macos")]
use hawking_core::kernels;
use hawking_core::metal::{MetalContext, TokenCommandBuffer};
use hawking_core::quant;
mod common;
use common::*;
fn make_q6k(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let w_f32 = fixed_f32(rows * cols, seed);
    let blocks = (rows * cols) / 256;
    let mut w_q6 = vec![0u8; blocks * quant::Q6_K_BLOCK_BYTES];
    quant::quantize_q6_k(&w_f32, &mut w_q6).expect("Q6_K quant");
    w_q6
}
fn run_2r(
    ctx: &MetalContext,
    w_q6: &[u8],
    gate: &[f32],
    up: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let model_buf = ctx.new_buffer_with_bytes(w_q6);
    let gate_buf = new_f32_buf(ctx, gate);
    let up_buf = new_f32_buf(ctx, up);
    let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q6_k_swiglu_2r_direct_tcb(
        &mut tcb,
        &model_buf,
        0,
        w_q6.len(),
        rows,
        cols,
        &gate_buf,
        &up_buf,
        &out_buf,
    )
    .unwrap();
    tcb.commit_and_wait().unwrap();
    read_f32_buf(&out_buf, rows)
}
fn run_4r(
    ctx: &MetalContext,
    w_q6: &[u8],
    gate: &[f32],
    up: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let model_buf = ctx.new_buffer_with_bytes(w_q6);
    let gate_buf = new_f32_buf(ctx, gate);
    let up_buf = new_f32_buf(ctx, up);
    let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q6_k_swiglu_4r_direct_tcb(
        &mut tcb,
        &model_buf,
        0,
        w_q6.len(),
        rows,
        cols,
        &gate_buf,
        &up_buf,
        &out_buf,
    )
    .unwrap();
    tcb.commit_and_wait().unwrap();
    read_f32_buf(&out_buf, rows)
}
#[test]
fn d8_q6k_swiglu_4r_matches_2r() {
    let ctx = ctx();
    const MAX_DIFF: f32 = 1e-4;
    let cases: &[(usize, usize, u64)] = &[
        (32, 256, 0xD800), // exactly 1 TG for 4r (32 rows/TG)
        (33, 256, 0xD801), // 2 TGs, 2nd has 1 row: tests has1/has2/has3 guards
        (40, 256, 0xD802), // 2 TGs, 2nd has 8 rows
        (64, 512, 0xD803), // exactly 2 TGs
        (128, 512, 0xD804),
        (256, 512, 0xD805),
        (17, 256, 0xD806),
        (25, 256, 0xD807),
        (48, 512, 0xD808),
    ];
    for &(rows, cols, seed) in cases {
        let w_q6 = make_q6k(rows, cols, seed);
        let gate = fixed_f32(cols, seed ^ 0x1000);
        let up = fixed_f32(cols, seed ^ 0x2000);
        let ref_out = run_2r(ctx, &w_q6, &gate, &up, rows, cols);
        let got_out = run_4r(ctx, &w_q6, &gate, &up, rows, cols);
        let diff = max_abs_diff(&ref_out, &got_out);
        assert!(
            diff <= MAX_DIFF,
            "D8 rows={rows} cols={cols}: 4r vs 2r diff={diff:.2e} > {MAX_DIFF:.2e}"
        );
    }
}

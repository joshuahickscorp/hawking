#![cfg(target_os = "macos")]
use half::f16;
use hawking_core::kernels;
use hawking_core::metal::{MetalContext, TokenCommandBuffer};
mod common;
use common::*;
fn make_q4k_predec(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    let bpr = cols / 256;
    let total_w = rows * bpr * 144;
    let w: Vec<u8> = (0..total_w)
        .map(|i| ((i as u32).wrapping_mul(2246822519u32).wrapping_add(seed)) as u8)
        .collect();
    let ns = rows * bpr * 16;
    let s: Vec<f32> = (0..ns)
        .map(|i| {
            let v = ((i as u32)
                .wrapping_mul(2654435761u32)
                .wrapping_add(seed ^ 0xAB)) as f32
                / u32::MAX as f32;
            0.1 + v * 1.9 // in [0.1, 2.0]
        })
        .collect();
    (w, s)
}
fn f32_to_f16_scales(scales: &[f32]) -> Vec<u8> {
    scales
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
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
fn rel_l2(reference: &[f32], got: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&r, &g) in reference.iter().zip(got) {
        let d = (r - g) as f64;
        num += d * d;
        den += (r as f64) * (r as f64);
    }
    (num / den.max(1e-30)).sqrt()
}
fn run_f32(
    ctx: &MetalContext,
    w: &[u8],
    scales: &[f32],
    gate: &[f32],
    up: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let w_buf = ctx.new_buffer_with_bytes(w);
    let s_buf = new_f32_buf(ctx, scales);
    let g_buf = new_f32_buf(ctx, gate);
    let u_buf = new_f32_buf(ctx, up);
    let y_buf = ctx.new_buffer(rows * 4);
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q4_k_v4_predec_swiglu_pinned_tcb(
        &mut tcb,
        &w_buf,
        0,
        w.len(),
        &s_buf,
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
fn run_f16s(
    ctx: &MetalContext,
    w: &[u8],
    scales_f16: &[u8],
    gate: &[f32],
    up: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let w_buf = ctx.new_buffer_with_bytes(w);
    let s_buf = ctx.new_buffer_with_bytes(scales_f16);
    let g_buf = new_f32_buf(ctx, gate);
    let u_buf = new_f32_buf(ctx, up);
    let y_buf = ctx.new_buffer(rows * 4);
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q4_k_v4_predec_f16s_swiglu_pinned_tcb(
        &mut tcb,
        &w_buf,
        0,
        w.len(),
        &s_buf,
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
fn predec_f16s_4r_swiglu_rel_l2_quality_gate() {
    let ctx = ctx();
    const MAX_REL_L2: f64 = 1e-2;
    let cases: &[(usize, usize, u32)] = &[
        (256, 2048, 0xD110), // 8 blocks — enough to average
        (1024, 2048, 0xD111),
        (2048, 2048, 0xD112),
        (2048, 11008, 0xD113), // Qwen-3B production ffn_down shape
    ];
    for &(rows, cols, seed) in cases {
        let (w, scales) = make_q4k_predec(rows, cols, seed);
        let scales_f16 = f32_to_f16_scales(&scales);
        let gate = rnd(cols, seed ^ 0x11);
        let up = rnd(cols, seed ^ 0x22);
        let ref_out = run_f32(&ctx, &w, &scales, &gate, &up, rows, cols);
        let got_out = run_f16s(&ctx, &w, &scales_f16, &gate, &up, rows, cols);
        let rel = rel_l2(&ref_out, &got_out);
        assert!(
            rel < MAX_REL_L2,
            "rows={rows} cols={cols}: rel_L2={rel:.4e} >= {MAX_REL_L2:.4e}"
        );
    }
}
#[test]
fn predec_f16s_4r_swiglu_small_shapes_smoke() {
    let ctx = ctx();
    const MAX_ABS_FACTOR: f32 = 0.5; // allow 50% of the reference magnitude as abs diff
    let cases: &[(usize, usize, u32)] = &[(256, 256, 0xD100), (512, 512, 0xD101)];
    for &(rows, cols, seed) in cases {
        let (w, scales) = make_q4k_predec(rows, cols, seed);
        let scales_f16 = f32_to_f16_scales(&scales);
        let gate = rnd(cols, seed ^ 0x11);
        let up = rnd(cols, seed ^ 0x22);
        let ref_out = run_f32(&ctx, &w, &scales, &gate, &up, rows, cols);
        let got_out = run_f16s(&ctx, &w, &scales_f16, &gate, &up, rows, cols);
        for &v in &got_out {
            assert!(!v.is_nan(), "rows={rows} cols={cols}: NaN in f16s output");
        }
        let ref_norm: f32 = ref_out.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let abs_diff = max_abs_diff(&ref_out, &got_out);
        assert!(
            ref_norm == 0.0 || abs_diff < MAX_ABS_FACTOR * ref_norm.max(1.0),
            "rows={rows} cols={cols}: abs_diff={abs_diff:.4} too large vs ref_norm={ref_norm:.4}"
        );
    }
}

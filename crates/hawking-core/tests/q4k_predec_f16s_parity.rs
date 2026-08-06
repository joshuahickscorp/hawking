#![cfg(target_os = "macos")]
use half::f16;
use hawking_core::kernels::{self, predecode_q4_k_scale_table_f16};
use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use rand::Rng;
use rand_pcg::Pcg64Mcg;
mod common;
use common::*;
fn make_x(cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..cols)
        .map(|_| rng.gen_range(-3.0_f32..3.0_f32))
        .collect()
}
fn new_f16_buf(ctx: &MetalContext, data: &[f16]) -> PinnedBuffer {
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|h| h.to_bits().to_le_bytes())
        .collect();
    ctx.new_buffer_with_bytes(&bytes)
}
#[test]
fn q4k_v4_predec_f16s_relative_parity() {
    let rows = 2048_usize;
    let cols = 2048_usize;
    let ctx = ctx();
    let w_bytes = make_q4k_bytes_pcg(rows, cols, 0xF165_8E1E);
    let model_buf = ctx.new_buffer_with_bytes(&w_bytes);
    let x = make_x(cols, 0xCAFE_F00D);
    let x_buf = new_f32_buf(ctx, &x);
    let scales_f32 = kernels::predecode_q4_k_scale_table(&w_bytes);
    let scales_f32_buf = new_f32_buf(ctx, &scales_f32);
    let y_ref_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_pinned_tcb(
            &mut tcb,
            &model_buf,
            0,
            w_bytes.len(),
            &scales_f32_buf,
            0,
            rows,
            cols,
            &x_buf,
            &y_ref_buf,
        )
        .expect("f32 predec encode");
        tcb.commit_and_wait().expect("f32 predec commit");
    }
    let y_ref = read_f32_buf(&y_ref_buf, rows);
    let scales_f16 = predecode_q4_k_scale_table_f16(&w_bytes);
    assert_eq!(
        scales_f16.len(),
        rows * (cols / 256) * 16,
        "predecode_q4_k_scale_table_f16 length mismatch"
    );
    let scales_f16_buf = new_f16_buf(ctx, &scales_f16);
    let y_f16_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_2r_f16s_pinned_tcb(
            &mut tcb,
            &model_buf,
            0,
            w_bytes.len(),
            &scales_f16_buf,
            0,
            rows,
            cols,
            &x_buf,
            &y_f16_buf,
        )
        .expect("f16s predec encode");
        tcb.commit_and_wait().expect("f16s predec commit");
    }
    let y_f16 = read_f32_buf(&y_f16_buf, rows);
    let mut num = 0.0_f64; // ||ref - f16||^2
    let mut den = 0.0_f64; // ||ref||^2
    let mut max_abs = 0.0_f32;
    for i in 0..rows {
        let d = (y_ref[i] - y_f16[i]) as f64;
        num += d * d;
        den += (y_ref[i] as f64) * (y_ref[i] as f64);
        max_abs = max_abs.max((y_ref[i] - y_f16[i]).abs());
    }
    let rel_l2 = (num / den.max(1e-30)).sqrt();
    assert!(
        rel_l2 < 1e-2,
        "f16-scales predec rel_L2 {rel_l2:.3e} exceeds the 1e-2 f16 precision budget"
    );
}

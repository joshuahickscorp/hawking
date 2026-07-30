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
fn rel_l2(reference: &[f32], test: &[f32]) -> (f64, f32, f64) {
    let mut num = 0.0_f64; // ||ref - test||^2
    let mut den = 0.0_f64; // ||ref||^2
    let mut max_abs = 0.0_f32;
    for i in 0..reference.len() {
        let d = (reference[i] - test[i]) as f64;
        num += d * d;
        den += (reference[i] as f64) * (reference[i] as f64);
        max_abs = max_abs.max((reference[i] - test[i]).abs());
    }
    ((num / den.max(1e-30)).sqrt(), max_abs, den.sqrt())
}
#[test]
fn q4k_v4_predec_pair_f16s_relative_parity() {
    let rows = 2048_usize;
    let cols = 2048_usize;
    let ctx = ctx();
    let wg_bytes = make_q4k_bytes_pcg(rows, cols, 0x6A7E_5CA1);
    let wu_bytes = make_q4k_bytes_pcg(rows, cols, 0x0DD0_5CA1);
    let mut model = wg_bytes.clone();
    let u_offset = model.len();
    model.extend_from_slice(&wu_bytes);
    let model_buf = ctx.new_buffer_with_bytes(&model);
    let x = make_x(cols, 0xCAFE_F00D);
    let x_buf = new_f32_buf(ctx, &x);
    let g_scales_f32 = kernels::predecode_q4_k_scale_table(&wg_bytes);
    let u_scales_f32 = kernels::predecode_q4_k_scale_table(&wu_bytes);
    let g_scales_f32_buf = new_f32_buf(ctx, &g_scales_f32);
    let u_scales_f32_buf = new_f32_buf(ctx, &u_scales_f32);
    let yg_ref_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let yu_ref_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_pair_pinned_tcb(
            &mut tcb,
            &model_buf,
            0,
            wg_bytes.len(),
            &g_scales_f32_buf,
            0,
            u_offset,
            wu_bytes.len(),
            &u_scales_f32_buf,
            0,
            rows,
            cols,
            &x_buf,
            &yg_ref_buf,
            &yu_ref_buf,
        )
        .expect("f32 pair encode");
        tcb.commit_and_wait().expect("f32 pair commit");
    }
    let yg_ref = read_f32_buf(&yg_ref_buf, rows);
    let yu_ref = read_f32_buf(&yu_ref_buf, rows);
    let g_scales_f16 = predecode_q4_k_scale_table_f16(&wg_bytes);
    let u_scales_f16 = predecode_q4_k_scale_table_f16(&wu_bytes);
    assert_eq!(
        g_scales_f16.len(),
        rows * (cols / 256) * 16,
        "predecode_q4_k_scale_table_f16 length mismatch (gate)"
    );
    let g_scales_f16_buf = new_f16_buf(ctx, &g_scales_f16);
    let u_scales_f16_buf = new_f16_buf(ctx, &u_scales_f16);
    let yg_f16_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let yu_f16_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_pair_f16s_pinned_tcb(
            &mut tcb,
            &model_buf,
            0,
            wg_bytes.len(),
            &g_scales_f16_buf,
            0,
            u_offset,
            wu_bytes.len(),
            &u_scales_f16_buf,
            0,
            rows,
            cols,
            &x_buf,
            &yg_f16_buf,
            &yu_f16_buf,
        )
        .expect("f16s pair encode");
        tcb.commit_and_wait().expect("f16s pair commit");
    }
    let yg_f16 = read_f32_buf(&yg_f16_buf, rows);
    let yu_f16 = read_f32_buf(&yu_f16_buf, rows);
    let yg_inline_f16_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let yu_inline_f16_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_pair_2r_inline_f16s_pinned_tcb(
            &mut tcb,
            &model_buf,
            0,
            wg_bytes.len(),
            &g_scales_f16_buf,
            0,
            u_offset,
            wu_bytes.len(),
            &u_scales_f16_buf,
            0,
            rows,
            cols,
            &x_buf,
            &yg_inline_f16_buf,
            &yu_inline_f16_buf,
        )
        .expect("2r inline f16s pair encode");
        tcb.commit_and_wait().expect("2r inline f16s pair commit");
    }
    let yg_inline_f16 = read_f32_buf(&yg_inline_f16_buf, rows);
    let yu_inline_f16 = read_f32_buf(&yu_inline_f16_buf, rows);
    let yg_f16s_nox_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let yu_f16s_nox_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_pair_f16s_nox_pinned_tcb(
            &mut tcb,
            &model_buf,
            0,
            wg_bytes.len(),
            &g_scales_f16_buf,
            0,
            u_offset,
            wu_bytes.len(),
            &u_scales_f16_buf,
            0,
            rows,
            cols,
            &x_buf,
            &yg_f16s_nox_buf,
            &yu_f16s_nox_buf,
        )
        .expect("f16s nox pair encode");
        tcb.commit_and_wait().expect("f16s nox pair commit");
    }
    let yg_f16s_nox = read_f32_buf(&yg_f16s_nox_buf, rows);
    let yu_f16s_nox = read_f32_buf(&yu_f16s_nox_buf, rows);
    let yg_halfreg_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let yu_halfreg_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_pair_f16s_halfreg_pinned_tcb(
            &mut tcb,
            &model_buf,
            0,
            wg_bytes.len(),
            &g_scales_f16_buf,
            0,
            u_offset,
            wu_bytes.len(),
            &u_scales_f16_buf,
            0,
            rows,
            cols,
            &x_buf,
            &yg_halfreg_buf,
            &yu_halfreg_buf,
        )
        .expect("f16s halfreg pair encode");
        tcb.commit_and_wait().expect("f16s halfreg pair commit");
    }
    let yg_halfreg = read_f32_buf(&yg_halfreg_buf, rows);
    let yu_halfreg = read_f32_buf(&yu_halfreg_buf, rows);
    let (g_rel, g_max, g_norm) = rel_l2(&yg_ref, &yg_f16);
    let (u_rel, u_max, u_norm) = rel_l2(&yu_ref, &yu_f16);
    let g_inline_max = max_abs_diff(&yg_f16, &yg_inline_f16);
    let u_inline_max = max_abs_diff(&yu_f16, &yu_inline_f16);
    let g_nox_max = max_abs_diff(&yg_f16, &yg_f16s_nox);
    let u_nox_max = max_abs_diff(&yu_f16, &yu_f16s_nox);
    let g_halfreg_max = max_abs_diff(&yg_f16, &yg_halfreg);
    let u_halfreg_max = max_abs_diff(&yu_f16, &yu_halfreg);
    assert!(
        g_rel < 1e-2,
        "f16s pair GATE rel_L2 {g_rel:.3e} exceeds the 1e-2 f16 precision budget"
    );
    assert!(
        u_rel < 1e-2,
        "f16s pair UP rel_L2 {u_rel:.3e} exceeds the 1e-2 f16 precision budget"
    );
    assert_eq!(
        g_inline_max, 0.0,
        "2r-inline f16s GATE max_abs {g_inline_max:.3e} differs from pair_f16s"
    );
    assert_eq!(
        u_inline_max, 0.0,
        "2r-inline f16s UP max_abs {u_inline_max:.3e} differs from pair_f16s"
    );
    assert_eq!(
        g_nox_max, 0.0,
        "f16s_nox GATE max_abs {g_nox_max:.3e} differs from pair_f16s"
    );
    assert_eq!(
        u_nox_max, 0.0,
        "f16s_nox UP max_abs {u_nox_max:.3e} differs from pair_f16s"
    );
    assert_eq!(
        g_halfreg_max, 0.0,
        "f16s_halfreg GATE max_abs {g_halfreg_max:.3e} differs from pair_f16s"
    );
    assert_eq!(
        u_halfreg_max, 0.0,
        "f16s_halfreg UP max_abs {u_halfreg_max:.3e} differs from pair_f16s"
    );
}

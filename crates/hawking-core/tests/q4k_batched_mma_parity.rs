#![cfg(target_os = "macos")]
use half::f16;
use hawking_core::kernels;
use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use rand::Rng;
use rand_pcg::Pcg64Mcg;
mod common;
use common::*;
const ATOL: f32 = 1e-3;
const RTOL: f32 = 1e-4;
fn newf(ctx: &MetalContext, d: &[f32]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(d))
}
fn readf(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
    let p = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
}
fn check_close(label: &str, rows: usize, cols: usize, batch: usize, a: &[f32], b: &[f32]) {
    let mut worst_abs = 0.0_f32;
    let mut worst_rel = 0.0_f32;
    let mut viol: Option<(usize, f32, f32, f32)> = None;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        let rel = d / a[i].abs().max(1e-6);
        worst_abs = worst_abs.max(d);
        worst_rel = worst_rel.max(rel);
        if d > ATOL + RTOL * a[i].abs() && viol.is_none() {
            viol = Some((i, d, a[i], b[i]));
        }
    }
    if let Some((i, d, av, bv)) = viol {
        panic!(
            "{label} {rows}x{cols} batch={batch}: abs diff {d} > atol {ATOL} + rtol {RTOL}*|a| \
             (worst @ {i}: ref={av:e} mma={bv:e}); max_abs={worst_abs:e} max_rel={worst_rel:e}"
        );
    }
}
fn check_shape_mma(ctx: &MetalContext, rows: usize, cols: usize, seed: u64) {
    let w = make_q4k_bytes_pcg(rows, cols, seed);
    let wbuf = ctx.new_buffer_with_bytes(&w);
    let mut rng = Pcg64Mcg::new(seed as u128 ^ 0xA5A5_A5A5);
    for batch in [1usize, 2, 4, 8] {
        let x: Vec<f32> = (0..batch * cols)
            .map(|_| rng.gen_range(-3.0_f32..3.0))
            .collect();
        let xbuf = newf(ctx, &x);
        let y_ref = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemm_q4_k_m_batched_v3w_pinned_tcb(
                &mut tcb,
                &wbuf,
                0,
                w.len(),
                rows,
                cols,
                batch,
                &xbuf,
                &y_ref,
            )
            .expect("v3w encode");
            tcb.commit_and_wait().expect("v3w commit");
        }
        let y_mma = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemm_q4_k_m_batched_v3w_mma_pinned_tcb(
                &mut tcb,
                &wbuf,
                0,
                w.len(),
                rows,
                cols,
                batch,
                &xbuf,
                &y_mma,
            )
            .expect("mma encode");
            tcb.commit_and_wait().expect("mma commit");
        }
        let a = readf(&y_ref, batch * rows);
        let bb = readf(&y_mma, batch * rows);
        check_close("mma vs v3w", rows, cols, batch, &a, &bb);
    }
}
fn check_shape_mma_predec(ctx: &MetalContext, rows: usize, cols: usize, seed: u64) {
    let w = make_q4k_bytes_pcg(rows, cols, seed);
    let wbuf = ctx.new_buffer_with_bytes(&w);
    let scales = kernels::predecode_q4_k_scale_table(&w);
    let sbuf = newf(ctx, &scales);
    let mut rng = Pcg64Mcg::new(seed as u128 ^ 0x1234_9876);
    for batch in [1usize, 2, 4, 8] {
        let x: Vec<f32> = (0..batch * cols)
            .map(|_| rng.gen_range(-3.0_f32..3.0))
            .collect();
        let xbuf = newf(ctx, &x);
        let y_ref = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemm_q4_k_m_batched_v3w_predec_pinned_tcb(
                &mut tcb,
                &wbuf,
                0,
                w.len(),
                &sbuf,
                0,
                rows,
                cols,
                batch,
                &xbuf,
                &y_ref,
            )
            .expect("v3w_predec encode");
            tcb.commit_and_wait().expect("v3w_predec commit");
        }
        let y_mma = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemm_q4_k_m_batched_v3w_mma_predec_pinned_tcb(
                &mut tcb,
                &wbuf,
                0,
                w.len(),
                &sbuf,
                0,
                rows,
                cols,
                batch,
                &xbuf,
                &y_mma,
            )
            .expect("mma_predec encode");
            tcb.commit_and_wait().expect("mma_predec commit");
        }
        let a = readf(&y_ref, batch * rows);
        let bb = readf(&y_mma, batch * rows);
        check_close("mma_predec vs v3w_predec", rows, cols, batch, &a, &bb);
    }
}
#[test]
fn mma_matches_v3w_winning_shape() {
    check_shape_mma(ctx(), 11008, 2048, 0xBEEF_1234);
}
#[test]
fn mma_matches_v3w_sanity_tile() {
    check_shape_mma(ctx(), 512, 512, 0x0512_0512);
}
#[test]
fn mma_predec_matches_v3w_predec_winning_shape() {
    check_shape_mma_predec(ctx(), 11008, 2048, 0xFEED_4321);
}
#[test]
fn mma_predec_matches_v3w_predec_sanity_tile() {
    check_shape_mma_predec(ctx(), 512, 512, 0x0512_0513);
}

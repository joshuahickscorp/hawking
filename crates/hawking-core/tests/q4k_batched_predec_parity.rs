#![cfg(target_os = "macos")]
use half::f16;
use hawking_core::kernels;
use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use rand::Rng;
use rand_pcg::Pcg64Mcg;
mod common;
use common::*;
fn newf(ctx: &MetalContext, d: &[f32]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(d))
}
fn readf(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
    let p = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
}
#[test]
fn batched_predec_bit_identical_to_v3w() {
    let rows = 2048_usize;
    let cols = 2048_usize;
    let ctx = ctx();
    let w = make_q4k_bytes_pcg(rows, cols, 0xBEEF_1234);
    let wbuf = ctx.new_buffer_with_bytes(&w);
    let scales = kernels::predecode_q4_k_scale_table(&w);
    let sbuf = newf(ctx, &scales);
    let mut rng = Pcg64Mcg::new(0x5EED_5EED);
    for batch in 1..=8usize {
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
        let y_predec = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
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
                &y_predec,
            )
            .expect("predec encode");
            tcb.commit_and_wait().expect("predec commit");
        }
        let a = readf(&y_ref, batch * rows);
        let b = readf(&y_predec, batch * rows);
        let mut diffs = 0usize;
        let mut first = None;
        for i in 0..a.len() {
            if a[i].to_bits() != b[i].to_bits() {
                diffs += 1;
                if first.is_none() {
                    first = Some((i, a[i], b[i]));
                }
            }
        }
        if let Some((i, av, bv)) = first {
            panic!(
                "batch={batch}: {diffs}/{} differ; first @ {i} v3w={av:e} predec={bv:e}",
                a.len()
            );
        }
    }
}
#[test]
fn batched_predec_bit_identical_ffn_down_shape() {
    let rows = 2048_usize;
    let cols = 11008_usize;
    let ctx = ctx();
    let w = make_q4k_bytes_pcg(rows, cols, 0xFADE_9988);
    let wbuf = ctx.new_buffer_with_bytes(&w);
    let scales = kernels::predecode_q4_k_scale_table(&w);
    let sbuf = newf(ctx, &scales);
    let mut rng = Pcg64Mcg::new(0xC0FF_EE11);
    for batch in 1..=8usize {
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
        let y_predec = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
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
                &y_predec,
            )
            .expect("predec encode");
            tcb.commit_and_wait().expect("predec commit");
        }
        let a = readf(&y_ref, batch * rows);
        let b = readf(&y_predec, batch * rows);
        let mut first = None;
        for i in 0..a.len() {
            if a[i].to_bits() != b[i].to_bits() {
                first = Some((i, a[i], b[i]));
                break;
            }
        }
        if let Some((i, av, bv)) = first {
            panic!("ffn_down batch={batch}: first diff @ {i} v3w={av:e} predec={bv:e}");
        }
    }
}

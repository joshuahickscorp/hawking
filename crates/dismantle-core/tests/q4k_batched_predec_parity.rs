//! Bit-identical parity: gemm_q4_k_m_batched_v3w_predec vs the non-predec
//! gemm_q4_k_m_batched_v3w, across batch B=1..=8. The predec variant only
//! pre-decodes the sub-block scales host-side; it does the same fp32 math in
//! the same order, so outputs must be bit-identical. Validates the batched
//! predec kernel in isolation before it's wired into the decode/verify path.

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use half::f16;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

mod common;
use common::*;

fn make_q4k_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let n_blocks = rows * (cols / 256);
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_blocks * 144];
    for b in 0..n_blocks {
        let off = b * 144;
        let d = 0.01_f32 + rng.gen::<f32>() * 0.01;
        let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
        bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        for i in 4..144 {
            bytes[off + i] = rng.gen::<u8>();
        }
    }
    bytes
}

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
    let w = make_q4k_bytes(rows, cols, 0xBEEF_1234);
    let wbuf = ctx.new_buffer_with_bytes(&w);
    let scales = kernels::predecode_q4_k_scale_table(&w);
    let sbuf = newf(ctx, &scales);

    let mut rng = Pcg64Mcg::new(0x5EED_5EED);
    for batch in 1..=8usize {
        // x_batch: (batch, cols) contiguous.
        let x: Vec<f32> = (0..batch * cols)
            .map(|_| rng.gen_range(-3.0_f32..3.0))
            .collect();
        let xbuf = newf(ctx, &x);

        let y_ref = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemm_q4_k_m_batched_v3w_pinned_tcb(
                &mut tcb, &wbuf, 0, w.len(), rows, cols, batch, &xbuf, &y_ref,
            )
            .expect("v3w encode");
            tcb.commit_and_wait().expect("v3w commit");
        }
        let y_predec = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemm_q4_k_m_batched_v3w_predec_pinned_tcb(
                &mut tcb, &wbuf, 0, w.len(), &sbuf, 0, rows, cols, batch, &xbuf, &y_predec,
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
        eprintln!("[batched-predec parity] batch={batch}: {} elems bit-identical", a.len());
    }
}

/// Phase-1 ffn_down shape (rows=h=2048, cols=intermediate=11008). Exercises
/// the requant'd-ffn_down predec wire-up's exact dispatch shape so the
/// large-cols path is covered, not just the square q_proj shape above.
#[test]
fn batched_predec_bit_identical_ffn_down_shape() {
    let rows = 2048_usize;
    let cols = 11008_usize;
    let ctx = ctx();
    let w = make_q4k_bytes(rows, cols, 0xFADE_9988);
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
                &mut tcb, &wbuf, 0, w.len(), rows, cols, batch, &xbuf, &y_ref,
            )
            .expect("v3w encode");
            tcb.commit_and_wait().expect("v3w commit");
        }
        let y_predec = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemm_q4_k_m_batched_v3w_predec_pinned_tcb(
                &mut tcb, &wbuf, 0, w.len(), &sbuf, 0, rows, cols, batch, &xbuf, &y_predec,
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
        eprintln!("[batched-predec parity ffn_down] batch={batch}: {} elems bit-identical", a.len());
    }
}

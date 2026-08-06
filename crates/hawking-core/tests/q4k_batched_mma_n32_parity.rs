#![cfg(target_os = "macos")]
//! K6 — the N-tiled MMA kernel must agree with the proven v3w kernel.
//!
//! `gemm_q4_k_m_batched_v3w_mma_n32` exists to lift batched prefill off the
//! B=8 ceiling, which is why prefill currently runs near decode speed. Lifting
//! that ceiling is only worth anything if the wider kernel computes the same
//! thing, so this checks it against `v3w` across the overlap (B<=8) and then
//! across the range only it can reach (B up to 32).
//!
//! The overlap cases matter most: if the N tiling were wrong, the padded
//! columns or the per-tile accumulators would show up there first.

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

fn assert_close(label: &str, batch: usize, a: &[f32], b: &[f32]) {
    let mut worst = 0.0_f32;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        worst = worst.max(d);
        assert!(
            d <= ATOL + RTOL * a[i].abs(),
            "{label} batch={batch}: element {i} differs by {d} (ref={:e} n32={:e})",
            a[i],
            b[i]
        );
    }
    eprintln!("  {label} batch={batch}: max abs diff {worst:e}");
}

/// Reference is the v3w kernel, which the existing MMA parity test already
/// treats as truth.
fn reference(
    ctx: &MetalContext,
    w: &PinnedBuffer,
    wlen: usize,
    rows: usize,
    cols: usize,
    batch: usize,
    x: &PinnedBuffer,
) -> Vec<f32> {
    let y = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemm_q4_k_m_batched_v3w_pinned_tcb(&mut tcb, w, 0, wlen, rows, cols, batch, x, &y)
        .expect("v3w reference dispatch");
    let _ = tcb.commit_and_wait();
    readf(&y, batch * rows)
}

fn n32(
    ctx: &MetalContext,
    w: &PinnedBuffer,
    wlen: usize,
    rows: usize,
    cols: usize,
    batch: usize,
    x: &PinnedBuffer,
) -> Vec<f32> {
    let y = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemm_q4_k_m_batched_v3w_mma_n32_pinned_tcb(
        &mut tcb, w, 0, wlen, rows, cols, batch, x, &y,
    )
    .expect("n32 dispatch");
    let _ = tcb.commit_and_wait();
    readf(&y, batch * rows)
}

fn check_shape(ctx: &MetalContext, rows: usize, cols: usize, seed: u64, batches: &[usize]) {
    let w = make_q4k_bytes_pcg(rows, cols, seed);
    let wbuf = ctx.new_buffer_with_bytes(&w);
    let mut rng = Pcg64Mcg::new(seed as u128 ^ 0x5A5A_5A5A);
    for &batch in batches {
        let x: Vec<f32> = (0..batch * cols)
            .map(|_| rng.gen_range(-3.0_f32..3.0))
            .collect();
        let xbuf = newf(ctx, &x);
        let want = reference(ctx, &wbuf, w.len(), rows, cols, batch, &xbuf);
        let got = n32(ctx, &wbuf, w.len(), rows, cols, batch, &xbuf);
        assert_close(&format!("{rows}x{cols}"), batch, &want, &got);
    }
}

#[test]
fn n32_matches_v3w_on_the_overlapping_batches() {
    let Some(ctx) = MetalContext::new().ok() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    // B<=8 is where both kernels are defined. A wrong tiling shows here first.
    for (rows, cols) in [(64usize, 256usize), (128, 512), (256, 256)] {
        check_shape(&ctx, rows, cols, 0xC0FFEE, &[1, 2, 4, 8]);
    }
}

#[test]
fn n32_is_correct_above_the_old_ceiling() {
    let Some(ctx) = MetalContext::new().ok() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    // v3w cannot run these, so compare against a CPU reference instead.
    let (rows, cols) = (64usize, 256usize);
    let w = make_q4k_bytes_pcg(rows, cols, 0xBEEF);
    let wbuf = ctx.new_buffer_with_bytes(&w);
    let mut rng = Pcg64Mcg::new(0x1234);
    for batch in [9usize, 16, 32] {
        let x: Vec<f32> = (0..batch * cols)
            .map(|_| rng.gen_range(-3.0_f32..3.0))
            .collect();
        let xbuf = newf(&ctx, &x);
        let got = n32(&ctx, &wbuf, w.len(), rows, cols, batch, &xbuf);

        // Each column n must equal the single-token result for that column,
        // which v3w can compute. This checks the tiling without needing a
        // second wide kernel to compare against.
        for n in 0..batch {
            let xn = newf(&ctx, &x[n * cols..(n + 1) * cols]);
            let want = reference(&ctx, &wbuf, w.len(), rows, cols, 1, &xn);
            let slice = &got[n * rows..(n + 1) * rows];
            for r in 0..rows {
                let d = (want[r] - slice[r]).abs();
                assert!(
                    d <= ATOL + RTOL * want[r].abs(),
                    "batch={batch} column {n} row {r}: {d} (single={:e} wide={:e})",
                    want[r],
                    slice[r]
                );
            }
        }
        eprintln!("  {rows}x{cols} batch={batch}: all {batch} columns match single-token v3w");
    }
}

#[test]
fn batch_above_32_is_refused_rather_than_silently_wrong() {
    let Some(ctx) = MetalContext::new().ok() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let (rows, cols) = (64usize, 256usize);
    let w = make_q4k_bytes_pcg(rows, cols, 1);
    let wbuf = ctx.new_buffer_with_bytes(&w);
    let x = vec![0.0_f32; 33 * cols];
    let xbuf = newf(&ctx, &x);
    let y = ctx.new_buffer(33 * rows * std::mem::size_of::<f32>());
    let mut tcb = TokenCommandBuffer::new(&ctx);
    let err = kernels::gemm_q4_k_m_batched_v3w_mma_n32_pinned_tcb(
        &mut tcb,
        &wbuf,
        0,
        w.len(),
        rows,
        cols,
        33,
        &xbuf,
        &y,
    )
    .expect_err("batch 33 must be refused");
    assert!(
        format!("{err}").contains("1..=32"),
        "error should name the real ceiling, got: {err}"
    );
}

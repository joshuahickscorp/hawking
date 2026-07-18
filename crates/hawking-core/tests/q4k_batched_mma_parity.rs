//! P1-A parity: the simdgroup-matrix (MMA) batched Q4_K GEMMs vs the tuned
//! scalar/predec v3w kernels, across batch B=1..=8.
//!
//! Unlike `q4k_batched_predec_parity` (bit-identical), the MMA kernels reorder
//! the K reduction (depth-8 hardware tiles + a different accumulation tree) vs
//! the scalar FMA chain, so they are numerically close but NOT `to_bits()`
//! equal. Gate is **atol = 1e-3 fp16** (the project's parity regime; the
//! standalone silicon #8 MMA measured ~1.26e-4, ~8x under 1e-3).
//!
//! Shapes: the rows>cols WINNING shape (11008x2048 — ffn gate/up, where the
//! caller actually swaps to MMA) at B in {1,2,4,8}, plus a 512x512 sanity tile.
//! (q/k/v/o square + ffn_down wide stay on v3w by the rows>cols gate, so the
//! MMA kernels are only exercised on rows>cols here.)

#![cfg(target_os = "macos")]

use half::f16;
use hawking_core::kernels;
use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use rand::Rng;
use rand_pcg::Pcg64Mcg;

mod common;
use common::*;

// Combined tolerance |a-b| <= ATOL + RTOL*|a| (numpy allclose). A pure
// atol=1e-3 is the project regime for ~O(1)-magnitude kernel outputs, but the
// MMA reorders the K reduction (depth-8 hardware tiles vs the scalar FMA
// chain), and at the ffn shape (cols=2048, random-byte Q4_K) outputs reach
// ~1e3 — where the fp32 reduction-reorder noise floor (~|y|*1e-6*sqrt(K)) is
// itself ~3e-3, *above* atol 1e-3. So atol alone is unsatisfiable for a
// CORRECT reordered kernel there. Measured relative error is 3e-6..1.3e-5;
// RTOL=1e-4 gives ~10-30x headroom yet stays ~100x tighter than any real
// indexing/math bug (which produces O(1) relative error).
const ATOL: f32 = 1e-3;
const RTOL: f32 = 1e-4;

fn make_q4k_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let n_blocks = rows * (cols / 256);
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_blocks * 144];
    for b in 0..n_blocks {
        let off = b * 144;
        // Small d/dmin keep dequant values in a tight range so atol 1e-3 is a
        // real gate (matches q4k_batched_predec_parity + phase1_kernel_parity).
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

/// Assert every element is within the combined atol+rtol tolerance; report the
/// worst absolute and relative diffs. Panics (fails the test) on any violation.
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
    eprintln!(
        "[{label}] {rows}x{cols} batch={batch}: max_abs={worst_abs:e} max_rel={worst_rel:e} \
         (atol {ATOL} rtol {RTOL})"
    );
}

/// Reference: tuned scalar v3w. Under test: the non-predec MMA twin.
fn check_shape_mma(ctx: &MetalContext, rows: usize, cols: usize, seed: u64) {
    let w = make_q4k_bytes(rows, cols, seed);
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

/// Reference: tuned v3w_predec. Under test: the predec MMA twin (the shipped
/// Option-B path). v3w_predec is bit-identical to v3w, so this also anchors
/// the predec twin against the scalar reference.
fn check_shape_mma_predec(ctx: &MetalContext, rows: usize, cols: usize, seed: u64) {
    let w = make_q4k_bytes(rows, cols, seed);
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
    // ffn gate/up: intermediate x hidden = 11008 x 2048 (rows>cols → the swap).
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

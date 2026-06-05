//! Parity: `gemm_q4_k_m_batched_v4r_predec` vs `gemm_q4_k_m_batched_v3w_predec`.
//!
//! v4r removes all threadgroup barriers and uses 16 rows/TG (was 8) + 2-row ILP.
//! The per-element Q4_K decode and FMA accumulation order are unchanged, so
//! outputs are expected to be bit-identical to v3w_predec.
//!
//! Shapes: 2048×2048 (q/k/v/o), 11008×2048 (ffn gate/up), 2048×11008 (ffn_down).
//! Batch B ∈ {2, 4, 8} (B=1 routes to the predec GEMV, never reaches v4r).
//!
//! Run:
//!   cargo test --release -p dismantle-core --test gemm_q4k_v4r_predec_parity \
//!     -- --ignored --nocapture

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::{PinnedBuffer, TokenCommandBuffer};
use half::f16;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

mod common;
use common::*;

const ATOL: f32 = 1e-3;

fn make_q4k_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let n_blocks = rows * (cols / 256);
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_blocks * 144];
    for b in 0..n_blocks {
        let off = b * 144;
        let d: f16 = f16::from_f32(0.01 + rng.gen::<f32>() * 0.01);
        let dmin: f16 = f16::from_f32((rng.gen::<f32>() - 0.5) * 0.01);
        bytes[off..off + 2].copy_from_slice(&d.to_bits().to_le_bytes());
        bytes[off + 2..off + 4].copy_from_slice(&dmin.to_bits().to_le_bytes());
        for i in 4..144 {
            bytes[off + i] = rng.gen::<u8>();
        }
    }
    bytes
}

fn run_v3w(
    wbuf: &PinnedBuffer, wlen: usize, sbuf: &PinnedBuffer,
    rows: usize, cols: usize, batch: usize, xbuf: &PinnedBuffer,
) -> Vec<f32> {
    let ctx = ctx();
    let ybuf = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemm_q4_k_m_batched_v3w_predec_pinned_tcb(
        &mut tcb, wbuf, 0, wlen, sbuf, 0, rows, cols, batch, xbuf, &ybuf,
    ).expect("v3w encode");
    tcb.commit_and_wait().expect("v3w commit");
    let p = ybuf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(p, batch * rows) }.to_vec()
}

fn run_v4r(
    wbuf: &PinnedBuffer, wlen: usize, sbuf: &PinnedBuffer,
    rows: usize, cols: usize, batch: usize, xbuf: &PinnedBuffer,
) -> Vec<f32> {
    let ctx = ctx();
    let ybuf = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemm_q4_k_m_batched_v4r_predec_pinned_tcb(
        &mut tcb, wbuf, 0, wlen, sbuf, 0, rows, cols, batch, xbuf, &ybuf,
    ).expect("v4r encode");
    tcb.commit_and_wait().expect("v4r commit");
    let p = ybuf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(p, batch * rows) }.to_vec()
}

fn check_parity(label: &str, rows: usize, cols: usize, batch: usize, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let mut n_bit_diff = 0usize;
    let mut max_abs = 0.0f32;
    for (&av, &bv) in a.iter().zip(b.iter()) {
        if av.to_bits() != bv.to_bits() { n_bit_diff += 1; }
        max_abs = max_abs.max((av - bv).abs());
    }
    if max_abs > ATOL {
        panic!("{label} {rows}×{cols} B={batch}: max_abs={max_abs:.3e} > atol {ATOL}");
    }
    println!(
        "[{label}] {rows}×{cols} B={batch}: max_abs={max_abs:.3e} bit_diffs={n_bit_diff}/{} — PASS",
        a.len()
    );
}

fn parity_shape(rows: usize, cols: usize, seed: u64) {
    let ctx = ctx();
    let w = make_q4k_bytes(rows, cols, seed);
    let wbuf = ctx.new_buffer_with_bytes(&w);
    let scales = kernels::predecode_q4_k_scale_table(&w);
    let sbuf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(&scales));
    let wlen = w.len();

    let mut rng = Pcg64Mcg::new((seed ^ 0x1357_2468u64) as u128);
    for &batch in &[2usize, 4, 8] {
        let x: Vec<f32> = (0..batch * cols).map(|_| rng.gen_range(-2.0f32..2.0)).collect();
        let xbuf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(&x));
        let a = run_v3w(&wbuf, wlen, &sbuf, rows, cols, batch, &xbuf);
        let b = run_v4r(&wbuf, wlen, &sbuf, rows, cols, batch, &xbuf);
        check_parity("v4r vs v3w", rows, cols, batch, &a, &b);
    }
}

#[test]
#[ignore]
fn v4r_parity_attn_square() {
    parity_shape(2048, 2048, 0x1001);
}

#[test]
#[ignore]
fn v4r_parity_ffn_up() {
    parity_shape(11008, 2048, 0x1002);
}

#[test]
#[ignore]
fn v4r_parity_ffn_down() {
    parity_shape(2048, 11008, 0x1003);
}

#[test]
#[ignore]
fn v4r_parity_small() {
    parity_shape(512, 512, 0x1004);
}

#![cfg(target_os = "macos")]
use half::f16;
use hawking_core::kernels;
use hawking_core::metal::{PinnedBuffer, TokenCommandBuffer};
use rand::Rng;
use rand_pcg::Pcg64Mcg;
mod common;
use common::*;
const ATOL: f32 = 1e-3;
fn run_v3w(
    wbuf: &PinnedBuffer,
    wlen: usize,
    sbuf: &PinnedBuffer,
    rows: usize,
    cols: usize,
    batch: usize,
    xbuf: &PinnedBuffer,
) -> Vec<f32> {
    let ctx = ctx();
    let ybuf = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemm_q4_k_m_batched_v3w_predec_pinned_tcb(
        &mut tcb, wbuf, 0, wlen, sbuf, 0, rows, cols, batch, xbuf, &ybuf,
    )
    .expect("v3w encode");
    tcb.commit_and_wait().expect("v3w commit");
    let p = ybuf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(p, batch * rows) }.to_vec()
}
fn run_v4r(
    wbuf: &PinnedBuffer,
    wlen: usize,
    sbuf: &PinnedBuffer,
    rows: usize,
    cols: usize,
    batch: usize,
    xbuf: &PinnedBuffer,
) -> Vec<f32> {
    let ctx = ctx();
    let ybuf = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemm_q4_k_m_batched_v4r_predec_pinned_tcb(
        &mut tcb, wbuf, 0, wlen, sbuf, 0, rows, cols, batch, xbuf, &ybuf,
    )
    .expect("v4r encode");
    tcb.commit_and_wait().expect("v4r commit");
    let p = ybuf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(p, batch * rows) }.to_vec()
}
fn check_parity(label: &str, rows: usize, cols: usize, batch: usize, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let mut n_bit_diff = 0usize;
    let mut max_abs = 0.0f32;
    for (&av, &bv) in a.iter().zip(b.iter()) {
        if av.to_bits() != bv.to_bits() {
            n_bit_diff += 1;
        }
        max_abs = max_abs.max((av - bv).abs());
    }
    if max_abs > ATOL {
        panic!("{label} {rows}×{cols} B={batch}: max_abs={max_abs:.3e} > atol {ATOL}");
    }
}
fn parity_shape(rows: usize, cols: usize, seed: u64) {
    let ctx = ctx();
    let w = make_q4k_bytes_pcg(rows, cols, seed);
    let wbuf = ctx.new_buffer_with_bytes(&w);
    let scales = kernels::predecode_q4_k_scale_table(&w);
    let sbuf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(&scales));
    let wlen = w.len();
    let mut rng = Pcg64Mcg::new((seed ^ 0x1357_2468u64) as u128);
    for &batch in &[2usize, 4, 8] {
        let x: Vec<f32> = (0..batch * cols)
            .map(|_| rng.gen_range(-2.0f32..2.0))
            .collect();
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

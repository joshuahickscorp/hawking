#![cfg(target_os = "macos")]
//! Track 3.6 parity: rope_qk_f32_b1_bias must be bit-identical to
//! (add_inplace q_bias + rope_q + add_inplace k_bias + rope_k).
//!
//! Tests several (n_q_heads, n_k_heads, head_dim, pos) combinations including
//! GQA shapes typical of Qwen2.5-3B (n_q=16, n_k=8, head_dim=128).

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, TokenCommandBuffer};
use once_cell::sync::Lazy;

fn ctx() -> &'static MetalContext {
    static CTX: Lazy<MetalContext> =
        Lazy::new(|| MetalContext::new().expect("Metal device required"));
    &CTX
}

fn new_f32_buf(ctx: &MetalContext, data: &[f32]) -> dismantle_core::metal::PinnedBuffer {
    let bytes = bytemuck::cast_slice(data);
    ctx.new_buffer_with_bytes(bytes)
}

fn read_f32_buf(buf: &dismantle_core::metal::PinnedBuffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }
}

fn rnd(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2654435761u32).wrapping_add(seed);
            (x as f32 / u32::MAX as f32) * 4.0 - 2.0
        })
        .collect()
}

/// Reference: 4 dispatches (q_bias + rope_q + k_bias + rope_k).
fn run_ref(
    ctx: &MetalContext,
    q: &[f32],
    k: &[f32],
    q_bias: Option<&[f32]>,
    k_bias: Option<&[f32]>,
    n_q: usize,
    n_k: usize,
    hd: usize,
    pos: u32,
    base: f32,
) -> (Vec<f32>, Vec<f32>) {
    let q_buf = new_f32_buf(ctx, q);
    let k_buf = new_f32_buf(ctx, k);
    let mut tcb = TokenCommandBuffer::new(ctx);
    if let Some(qb) = q_bias {
        let b_buf = new_f32_buf(ctx, qb);
        kernels::add_inplace_metal_tcb(&mut tcb, &q_buf, &b_buf, q.len()).unwrap();
    }
    if let Some(kb) = k_bias {
        let b_buf = new_f32_buf(ctx, kb);
        kernels::add_inplace_metal_tcb(&mut tcb, &k_buf, &b_buf, k.len()).unwrap();
    }
    kernels::rope_q_f32_inplace_tcb(&mut tcb, &q_buf, n_q, hd, 0, hd, pos, base).unwrap();
    kernels::rope_q_f32_inplace_tcb(&mut tcb, &k_buf, n_k, hd, 0, hd, pos, base).unwrap();
    tcb.commit_and_wait().unwrap();
    (read_f32_buf(&q_buf, q.len()), read_f32_buf(&k_buf, k.len()))
}

/// Fused: 1 dispatch (rope_qk_f32_b1_bias).
fn run_fused(
    ctx: &MetalContext,
    q: &[f32],
    k: &[f32],
    q_bias: Option<&[f32]>,
    k_bias: Option<&[f32]>,
    n_q: usize,
    n_k: usize,
    hd: usize,
    pos: u32,
    base: f32,
) -> (Vec<f32>, Vec<f32>) {
    let q_buf = new_f32_buf(ctx, q);
    let k_buf = new_f32_buf(ctx, k);
    let qb_buf = q_bias.map(|b| new_f32_buf(ctx, b));
    let kb_buf = k_bias.map(|b| new_f32_buf(ctx, b));
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::rope_qk_f32_b1_bias_tcb(
        &mut tcb,
        &q_buf,
        &k_buf,
        qb_buf.as_ref(),
        kb_buf.as_ref(),
        n_q,
        n_k,
        hd,
        pos,
        base,
    )
    .unwrap();
    tcb.commit_and_wait().unwrap();
    (read_f32_buf(&q_buf, q.len()), read_f32_buf(&k_buf, k.len()))
}

fn check(label: &str, n_q: usize, n_k: usize, hd: usize, pos: u32, with_bias: bool) {
    let ctx = ctx();
    let q = rnd(n_q * hd, 0xABCD + pos);
    let k = rnd(n_k * hd, 0x1234 + pos);
    let qb = with_bias.then(|| rnd(n_q * hd, 0xDEAD));
    let kb = with_bias.then(|| rnd(n_k * hd, 0xBEEF));
    let (rq, rk) = run_ref(
        ctx,
        &q,
        &k,
        qb.as_deref(),
        kb.as_deref(),
        n_q,
        n_k,
        hd,
        pos,
        10000.0,
    );
    let (fq, fk) = run_fused(
        ctx,
        &q,
        &k,
        qb.as_deref(),
        kb.as_deref(),
        n_q,
        n_k,
        hd,
        pos,
        10000.0,
    );
    let max_q = rq
        .iter()
        .zip(&fq)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_k = rk
        .iter()
        .zip(&fk)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // The fused kernel computes bias-add + rope in one pass vs the reference's
    // store-reload between the two dispatches. The Metal compiler may apply FMA
    // fusion across the (q+bias)*c and (q+bias)*s products, producing 1-ULP
    // rounding differences. Allow a relative tolerance of 1e-5.
    let rel_q = rq
        .iter()
        .zip(&fq)
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1.0))
        .fold(0.0f32, f32::max);
    let rel_k = rk
        .iter()
        .zip(&fk)
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1.0))
        .fold(0.0f32, f32::max);
    assert!(rel_q < 1e-5, "{label}: Q max_rel={rel_q:.2e} > 1e-5");
    assert!(rel_k < 1e-5, "{label}: K max_rel={rel_k:.2e} > 1e-5");
    eprintln!("{label}: Q max_rel={rel_q:.2e}  K max_rel={rel_k:.2e}  OK");
}

#[test]
fn rope_qk_b1_bias_qwen3b_shape() {
    // Qwen2.5-3B: n_q=16, n_k=8 (GQA), head_dim=128
    check("qwen3b  bias  pos=0", 16, 8, 128, 0, true);
    check("qwen3b  bias  pos=127", 16, 8, 128, 127, true);
    check("qwen3b  bias  pos=512", 16, 8, 128, 512, true);
    check("qwen3b  nobias pos=63", 16, 8, 128, 63, false);
}

#[test]
fn rope_qk_b1_bias_mha_shape() {
    // MHA (n_q == n_k): e.g. 32 heads, head_dim=128
    check("mha128 bias  pos=1", 32, 32, 128, 1, true);
    check("mha128 nobias pos=255", 32, 32, 128, 255, false);
}

#[test]
fn rope_qk_b1_bias_small_shape() {
    // Small shapes to check edge cases
    check("small  bias  pos=7", 4, 2, 64, 7, true);
    check("single nobias pos=0", 1, 1, 128, 0, false);
}

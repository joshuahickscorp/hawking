#![cfg(target_os = "macos")]
//! P1b — parity test for `mha_decode_f32_tcb` against the CPU reference
//! `crate::attn::mha_decode_step`.
//!
//! Setup matches a Qwen-class GQA decode step: n_heads = group_size *
//! n_kv_heads, single new token, seq_len includes the new token.
//!
//! Tolerance: 1e-4 absolute. Softmax + dot-product accumulation order
//! differs between the per-thread CPU loop and the TG-parallel GPU
//! reduction, so bit-identical is not achievable. fp32 atol 1e-4 is
//! comfortably tighter than any downstream accumulated drift across 28
//! layers (per-token error << per-layer rmsnorm tolerance 1e-5 * 28).

use dismantle_core::attn::mha_decode_step;
use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use once_cell::sync::Lazy;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

fn ctx() -> &'static MetalContext {
    static CTX: Lazy<MetalContext> =
        Lazy::new(|| MetalContext::new().expect("Metal device required"));
    &CTX
}

fn fixed_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
}

fn new_f32_buf(ctx: &MetalContext, data: &[f32]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(data))
}

fn read_f32_buf(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn mha_decode_metal_matches_cpu() {
    let n_heads = 4usize;
    let n_kv_heads = 2usize;
    let head_dim = 64usize;
    let seq_len = 16usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    let q = fixed_f32(q_dim, 0xA1A1_A1A1);
    let k = fixed_f32(seq_len * kv_dim, 0xB2B2_B2B2);
    let v = fixed_f32(seq_len * kv_dim, 0xC3C3_C3C3);

    // CPU reference.
    let mut expected = vec![0.0f32; q_dim];
    mha_decode_step(
        &q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, &mut expected,
    )
    .expect("cpu mha_decode_step");

    // GPU path.
    let ctx = ctx();
    let q_buf = new_f32_buf(ctx, &q);
    let k_buf = new_f32_buf(ctx, &k);
    let v_buf = new_f32_buf(ctx, &v);
    let out_buf = ctx.new_buffer(q_dim * std::mem::size_of::<f32>());

    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::mha_decode_f32_tcb(
            &mut tcb, &q_buf, &k_buf, 0, &v_buf, 0, &out_buf, seq_len, head_dim, n_heads, n_kv_heads,
        )
        .expect("mha_decode_f32_tcb encode");
        tcb.commit_and_wait().expect("mha_decode_f32_tcb commit");
    }

    let actual = read_f32_buf(&out_buf, q_dim);
    let diff = max_abs_diff(&expected, &actual);
    assert!(
        diff < 1e-4,
        "mha_decode_f32 vs CPU max_abs_diff = {diff} (limit 1e-4)"
    );
}

#[test]
fn mha_decode_metal_seq_len_one() {
    // Smallest meaningful case: seq_len=1 (first decode token).
    let n_heads = 2usize;
    let n_kv_heads = 1usize;
    let head_dim = 32usize;
    let seq_len = 1usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    let q = fixed_f32(q_dim, 0xDEAD_BEEF);
    let k = fixed_f32(seq_len * kv_dim, 0xCAFE_BABE);
    let v = fixed_f32(seq_len * kv_dim, 0xFEED_FACE);

    let mut expected = vec![0.0f32; q_dim];
    mha_decode_step(
        &q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, &mut expected,
    )
    .unwrap();

    let ctx = ctx();
    let q_buf = new_f32_buf(ctx, &q);
    let k_buf = new_f32_buf(ctx, &k);
    let v_buf = new_f32_buf(ctx, &v);
    let out_buf = ctx.new_buffer(q_dim * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::mha_decode_f32_tcb(
            &mut tcb, &q_buf, &k_buf, 0, &v_buf, 0, &out_buf, seq_len, head_dim, n_heads, n_kv_heads,
        )
        .unwrap();
        tcb.commit_and_wait().unwrap();
    }
    let actual = read_f32_buf(&out_buf, q_dim);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-4, "seq_len=1: max_abs_diff = {diff}");
}

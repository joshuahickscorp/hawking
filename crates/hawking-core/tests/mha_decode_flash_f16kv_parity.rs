#![cfg(target_os = "macos")]
use half::f16;
use hawking_core::attn::mha_decode_step;
use hawking_core::kernels;
use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
mod common;
use common::*;
const ATOL: f32 = 1e-3;
const RTOL: f32 = 1e-4;
fn new_f16_buf(ctx: &MetalContext, data: &[f32]) -> PinnedBuffer {
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|&x| f16::from_f32(x).to_bits().to_le_bytes())
        .collect();
    ctx.new_buffer_with_bytes(&bytes)
}
fn run_flash_f16kv(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    let q_dim = n_heads * head_dim;
    let ctx = ctx();
    let q_buf = new_f32_buf(ctx, q);
    let k_buf = new_f16_buf(ctx, k);
    let v_buf = new_f16_buf(ctx, v);
    let out_buf = ctx.new_buffer(q_dim * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::mha_decode_flash_f16kv_tcb(
            &mut tcb, &q_buf, &k_buf, 0, &v_buf, 0, &out_buf, seq_len, head_dim, n_heads,
            n_kv_heads,
        )
        .expect("mha_decode_flash_f16kv_tcb encode");
        tcb.commit_and_wait()
            .expect("mha_decode_flash_f16kv_tcb commit");
    }
    read_f32_buf(&out_buf, q_dim)
}
fn check_geometry(n_heads: usize, n_kv_heads: usize, head_dim: usize, seq_len: usize) {
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let seed = (seq_len as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ ((n_heads as u64) << 17)
        ^ ((head_dim as u64) << 33);
    let q = fixed_f32(q_dim, seed ^ 0xA1);
    let k = fixed_f32(seq_len * kv_dim, seed ^ 0xB2);
    let v = fixed_f32(seq_len * kv_dim, seed ^ 0xC3);
    let k_rt = f16_round_trip(&k);
    let v_rt = f16_round_trip(&v);
    let mut cpu = vec![0.0f32; q_dim];
    mha_decode_step(
        &q, &k_rt, &v_rt, n_heads, n_kv_heads, head_dim, seq_len, &mut cpu,
    )
    .expect("cpu mha_decode_step");
    let flash = run_flash_f16kv(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len);
    let (vf, i) = worst_violation(&flash, &cpu, ATOL, RTOL);
    assert!(
        vf <= 0.0,
        "flash_f16kv vs CPU(f16-rt): seq={seq_len} h={n_heads} kvh={n_kv_heads} hd={head_dim}: \
         violation {vf} beyond atol={ATOL}+rtol={RTOL} at i={i} (flash={} cpu={})",
        flash[i],
        cpu[i]
    );
}
#[test]
fn flash_f16kv_matches_ref_qwen_geometry_multi_tile() {
    let (n_heads, n_kv_heads, head_dim) = (16usize, 2usize, 128usize);
    for &seq_len in &[1usize, 128, 129, 384, 4096] {
        check_geometry(n_heads, n_kv_heads, head_dim, seq_len);
    }
}
#[test]
fn flash_f16kv_full_mha_tile_boundary() {
    check_geometry(4, 4, 128, 129);
}
#[test]
fn flash_f16kv_long_context_4k() {
    check_geometry(16, 2, 128, 4096);
}

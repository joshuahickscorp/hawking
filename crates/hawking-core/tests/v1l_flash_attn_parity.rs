#![cfg(target_os = "macos")]
use hawking_core::kernels;
use hawking_core::metal::PinnedBuffer;
mod common;
use common::*;
fn run_parity(
    label: &str,
    n_heads: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    seq_len: usize,
) {
    let ctx = ctx();
    let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    let scale = 1.0_f32 / (q_head_dim as f32).sqrt();
    let q = fixed_f32(n_heads * q_head_dim, 0xDEAD_BEEF);
    let c_kv = fixed_f32(seq_len * kv_lora_rank, 0xCAFE_BABE);
    let k_pe = fixed_f32(seq_len * qk_rope_head_dim, 0x1234_5678);
    let kv_b_raw = fixed_f32(
        n_heads * (qk_nope_head_dim + v_head_dim) * kv_lora_rank,
        0xABCD_EF01,
    );
    let kv_b_buf: PinnedBuffer =
        ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&kv_b_raw));
    let mut mla_out = vec![0.0f32; n_heads * v_head_dim];
    kernels::mla_decode_metal(
        ctx,
        &q,
        &c_kv,
        &k_pe,
        &kv_b_buf,
        n_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        kv_lora_rank,
        seq_len,
        scale,
        &mut mla_out,
    )
    .expect("mla_decode_metal");
    let mut flash_out = vec![0.0f32; n_heads * v_head_dim];
    kernels::flash_attn_decode_metal(
        ctx,
        &q,
        &c_kv,
        &k_pe,
        &kv_b_buf,
        n_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        kv_lora_rank,
        seq_len,
        scale,
        &mut flash_out,
    )
    .expect("flash_attn_decode_metal");
    let diff = max_abs_diff(&mla_out, &flash_out);
    assert!(diff < 1e-3, "{label}: flash vs mla diff {diff:.2e} >= 1e-3");
}
#[test]
fn v1l_flash_vs_mla_small() {
    run_parity(
        "small(heads=4,nope=16,rope=8,v=16,lora=32,seq=64)",
        4,
        16,
        8,
        16,
        32,
        64,
    );
}
#[test]
fn v1l_flash_vs_mla_realistic() {
    run_parity(
        "realistic(heads=16,nope=64,rope=32,v=64,lora=64,seq=256)",
        16,
        64,
        32,
        64,
        64,
        256,
    );
}
#[test]
fn v1l_flash_vs_mla_seq_one() {
    run_parity(
        "seq1(heads=4,nope=16,rope=8,v=16,lora=32,seq=1)",
        4,
        16,
        8,
        16,
        32,
        1,
    );
}
#[test]
fn v1l_flash_vs_mla_multi_tile() {
    run_parity(
        "multi_tile(heads=8,nope=32,rope=16,v=32,lora=32,seq=384)",
        8,
        32,
        16,
        32,
        32,
        384,
    );
}

#![cfg(target_os = "macos")]
use half::f16;
use hawking_core::attn::mha_decode_step;
use hawking_core::kernels;
use hawking_core::metal::TokenCommandBuffer;
mod common;
use common::*;
const N_HEADS: usize = 16;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
fn new_f16_buf(
    ctx: &hawking_core::metal::MetalContext,
    data: &[f32],
) -> hawking_core::metal::PinnedBuffer {
    let mut bytes = vec![0u8; data.len() * std::mem::size_of::<u16>()];
    for (i, &x) in data.iter().enumerate() {
        let bits = f16::from_f32(x).to_bits();
        bytes[2 * i..2 * i + 2].copy_from_slice(&bits.to_le_bytes());
    }
    ctx.new_buffer_with_bytes(&bytes)
}
fn run_f16kv_single(label: &str, seq_len: usize, atol: f32) {
    let ctx = ctx();
    let q_dim = N_HEADS * HEAD_DIM;
    let kv_dim = N_KV_HEADS * HEAD_DIM;
    let q = fixed_f32(q_dim, 0xF16C_0DE0 ^ seq_len as u64);
    let k = fixed_f32(seq_len * kv_dim, 0x0B2B_2B2B ^ seq_len as u64);
    let v = fixed_f32(seq_len * kv_dim, 0x0C3C_3C3C ^ seq_len as u64);
    let k_rt = f16_round_trip(&k);
    let v_rt = f16_round_trip(&v);
    let q_buf = new_f32_buf(ctx, &q);
    let k_ref_buf = new_f32_buf(ctx, &k_rt);
    let v_ref_buf = new_f32_buf(ctx, &v_rt);
    let ref_out_buf = ctx.new_buffer(q_dim * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::mha_decode_f32_tcb(
            &mut tcb,
            &q_buf,
            &k_ref_buf,
            0,
            &v_ref_buf,
            0,
            &ref_out_buf,
            seq_len,
            HEAD_DIM,
            N_HEADS,
            N_KV_HEADS,
        )
        .expect("mha_decode_f32_tcb encode");
        tcb.commit_and_wait().expect("mha_decode_f32_tcb commit");
    }
    let ref_gpu = read_f32_buf(&ref_out_buf, q_dim);
    let mut ref_cpu = vec![0.0f32; q_dim];
    mha_decode_step(
        &q,
        &k_rt,
        &v_rt,
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        seq_len,
        &mut ref_cpu,
    )
    .expect("cpu mha_decode_step");
    let k_f16_buf = new_f16_buf(ctx, &k);
    let v_f16_buf = new_f16_buf(ctx, &v);
    let out_buf = ctx.new_buffer(q_dim * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::mha_decode_f16kv_tcb(
            &mut tcb, &q_buf, &k_f16_buf, 0, &v_f16_buf, 0, &out_buf, seq_len, HEAD_DIM, N_HEADS,
            N_KV_HEADS,
        )
        .expect("mha_decode_f16kv_tcb encode");
        tcb.commit_and_wait().expect("mha_decode_f16kv_tcb commit");
    }
    let actual = read_f32_buf(&out_buf, q_dim);
    let diff_gpu = max_abs_diff(&ref_gpu, &actual);
    let diff_cpu = max_abs_diff(&ref_cpu, &actual);
    assert!(
        diff_gpu < atol,
        "{label}: f16kv vs f32-GPU diff {diff_gpu:.3e} >= {atol:.0e}"
    );
    assert!(
        diff_cpu < atol,
        "{label}: f16kv vs CPU diff {diff_cpu:.3e} >= {atol:.0e}"
    );
}
#[test]
fn f16kv_single_seq1() {
    run_f16kv_single("seq=1", 1, ATOL);
}
#[test]
fn f16kv_single_seq64() {
    run_f16kv_single("seq=64", 64, ATOL);
}
#[test]
fn f16kv_single_seq512() {
    run_f16kv_single("seq=512", 512, ATOL);
}
#[test]
fn f16kv_single_seq2048_long_context() {
    run_f16kv_single("seq=2048 long-ctx", 2048, ATOL);
}
fn run_f16kv_batched(label: &str, p0: usize, b: usize, atol: f32) {
    let ctx = ctx();
    let q_dim = N_HEADS * HEAD_DIM;
    let kv_dim = N_KV_HEADS * HEAD_DIM;
    let total_seq = p0 + b; // cache must cover the largest batch's causal prefix
    let q = fixed_f32(b * q_dim, 0xBA7C_0DE0 ^ (p0 as u64));
    let k = fixed_f32(total_seq * kv_dim, 0x0B2B_2B2B ^ (p0 as u64));
    let v = fixed_f32(total_seq * kv_dim, 0x0C3C_3C3C ^ (p0 as u64));
    let k_rt = f16_round_trip(&k);
    let v_rt = f16_round_trip(&v);
    let q_buf = new_f32_buf(ctx, &q);
    let k_ref_buf = new_f32_buf(ctx, &k_rt);
    let v_ref_buf = new_f32_buf(ctx, &v_rt);
    let ref_out_buf = ctx.new_buffer(b * q_dim * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::mha_decode_f32_batched_tcb(
            &mut tcb,
            &q_buf,
            &k_ref_buf,
            0,
            &v_ref_buf,
            0,
            &ref_out_buf,
            p0,
            b,
            HEAD_DIM,
            N_HEADS,
            N_KV_HEADS,
        )
        .expect("mha_decode_f32_batched_tcb encode");
        tcb.commit_and_wait()
            .expect("mha_decode_f32_batched_tcb commit");
    }
    let ref_gpu = read_f32_buf(&ref_out_buf, b * q_dim);
    let mut ref_cpu = vec![0.0f32; b * q_dim];
    for bi in 0..b {
        let seq_bi = p0 + bi + 1;
        let q_bi = &q[bi * q_dim..(bi + 1) * q_dim];
        let out_bi = &mut ref_cpu[bi * q_dim..(bi + 1) * q_dim];
        mha_decode_step(
            q_bi,
            &k_rt[..seq_bi * kv_dim],
            &v_rt[..seq_bi * kv_dim],
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            seq_bi,
            out_bi,
        )
        .expect("cpu mha_decode_step (batched elem)");
    }
    let k_f16_buf = new_f16_buf(ctx, &k);
    let v_f16_buf = new_f16_buf(ctx, &v);
    let out_buf = ctx.new_buffer(b * q_dim * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::mha_decode_f16kv_batched_tcb(
            &mut tcb, &q_buf, &k_f16_buf, 0, &v_f16_buf, 0, &out_buf, p0, b, HEAD_DIM, N_HEADS,
            N_KV_HEADS,
        )
        .expect("mha_decode_f16kv_batched_tcb encode");
        tcb.commit_and_wait()
            .expect("mha_decode_f16kv_batched_tcb commit");
    }
    let actual = read_f32_buf(&out_buf, b * q_dim);
    let diff_gpu = max_abs_diff(&ref_gpu, &actual);
    let diff_cpu = max_abs_diff(&ref_cpu, &actual);
    assert!(
        diff_gpu < atol,
        "{label}: f16kv-batched vs f32-GPU diff {diff_gpu:.3e} >= {atol:.0e}"
    );
    assert!(
        diff_cpu < atol,
        "{label}: f16kv-batched vs CPU diff {diff_cpu:.3e} >= {atol:.0e}"
    );
}
#[test]
fn f16kv_batched_p0_0_b8() {
    run_f16kv_batched("p0=0 b=8", 0, 8, ATOL);
}
#[test]
fn f16kv_batched_p0_64_b4() {
    run_f16kv_batched("p0=64 b=4", 64, 4, ATOL);
}
#[test]
fn f16kv_batched_p0_2048_b4_long_context() {
    run_f16kv_batched("p0=2048 b=4 long-ctx", 2048, 4, ATOL);
}
#[test]
fn f16_kv_append_matches_cpu_round_trip() {
    let ctx = ctx();
    let kv_dim = N_KV_HEADS * HEAD_DIM; // 256 elems = one token's K (or V) slice
    let max_seq = 8usize;
    let seq_slot = 3usize;
    let src = fixed_f32(kv_dim, 0x0A11_CE00);
    let src_buf = new_f32_buf(ctx, &src);
    let cache_elems = max_seq * kv_dim;
    let dst_buf = ctx.new_buffer(cache_elems * std::mem::size_of::<u16>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::memcpy_f32_to_f16_off_tcb(
            &mut tcb,
            &src_buf,
            &dst_buf,
            0,
            seq_slot * kv_dim,
            kv_dim,
        )
        .expect("memcpy_f32_to_f16_off encode");
        tcb.commit_and_wait().expect("memcpy_f32_to_f16_off commit");
    }
    let dst_bits: Vec<u16> = {
        let ptr = dst_buf.contents() as *const u16;
        unsafe { std::slice::from_raw_parts(ptr, cache_elems) }.to_vec()
    };
    let base = seq_slot * kv_dim;
    for (i, &x) in src.iter().enumerate() {
        let expect = f16::from_f32(x).to_bits();
        let got = dst_bits[base + i];
        assert_eq!(
            got, expect,
            "f16 KV-append bit mismatch at elem {i}: gpu={got:#06x} cpu={expect:#06x}"
        );
    }
    for s in 0..max_seq {
        if s == seq_slot {
            continue;
        }
        let slot = &dst_bits[s * kv_dim..(s + 1) * kv_dim];
        assert!(
            slot.iter().all(|&b| b == 0),
            "slot {s} was not supposed to be written"
        );
    }
}

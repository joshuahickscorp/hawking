#![cfg(target_os = "macos")]
use hawking_core::kernels;
use hawking_core::metal::PinnedBuffer;
use hawking_core::quant::{quantize_q8_0, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS};
mod common;
use common::*;
#[test]
fn kv_append_q8_gpu_matches_cpu_quantize() {
    let ctx = ctx();
    let kv_lora_rank = 512usize;
    let qk_rope_head_dim = 64usize;
    let max_seq = 8usize;
    let c_kv_normed = fixed_f32(kv_lora_rank, 0xA11CE);
    let mut kv_a_out = vec![0.0f32; kv_lora_rank + qk_rope_head_dim];
    let pe_src = fixed_f32(qk_rope_head_dim, 0xB0B);
    kv_a_out[kv_lora_rank..].copy_from_slice(&pe_src);
    let n_blocks = kv_lora_rank / Q8_0_BLOCK_ELEMS;
    let row_bytes = n_blocks * Q8_0_BLOCK_BYTES;
    let mut gpu_cache = vec![0u8; max_seq * row_bytes];
    let mut gpu_kpe = vec![0.0f32; max_seq * qk_rope_head_dim];
    let seq_slot = 3usize;
    kernels::kv_append_q8_0_f32_metal(
        ctx,
        &c_kv_normed,
        &kv_a_out,
        &mut gpu_cache,
        &mut gpu_kpe,
        seq_slot,
        kv_lora_rank,
        qk_rope_head_dim,
        max_seq,
    )
    .expect("gpu kv_append_q8");
    let mut cpu_row = vec![0u8; row_bytes];
    quantize_q8_0(&c_kv_normed, &mut cpu_row).expect("cpu quantize");
    let gpu_row = &gpu_cache[seq_slot * row_bytes..(seq_slot + 1) * row_bytes];
    let diff_bytes: Vec<usize> = gpu_row
        .iter()
        .zip(cpu_row.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();
    assert!(
        diff_bytes.is_empty(),
        "GPU/CPU Q8 quantize differ at byte offsets: {diff_bytes:?}"
    );
    let gpu_pe = &gpu_kpe[seq_slot * qk_rope_head_dim..(seq_slot + 1) * qk_rope_head_dim];
    for (i, (a, b)) in gpu_pe.iter().zip(pe_src.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-9,
            "k_pe element {i} mismatch: gpu={a} cpu={b}"
        );
    }
    for s in 0..max_seq {
        if s == seq_slot {
            continue;
        }
        let row = &gpu_cache[s * row_bytes..(s + 1) * row_bytes];
        assert!(
            row.iter().all(|&b| b == 0),
            "slot {s} wasn't supposed to be written"
        );
    }
}
const N_HEADS: usize = 16;
const QK_NOPE: usize = 128;
const QK_ROPE: usize = 64;
const V_HEAD: usize = 128;
const KV_LORA: usize = 512;
fn run_q8_vs_f32(label: &str, seq_len: usize, c_kv_scale: f32, atol: f32) {
    let ctx = ctx();
    let q_head_dim = QK_NOPE + QK_ROPE;
    let scale = 1.0_f32 / (q_head_dim as f32).sqrt();
    let q = fixed_f32(N_HEADS * q_head_dim, 0xDEAD ^ seq_len as u64);
    let c_kv: Vec<f32> = fixed_f32(seq_len * KV_LORA, 0xBEEF ^ seq_len as u64)
        .into_iter()
        .map(|x| x * c_kv_scale)
        .collect();
    let k_pe = fixed_f32(seq_len * QK_ROPE, 0xCAFE ^ seq_len as u64);
    let kv_b = fixed_f32(N_HEADS * (QK_NOPE + V_HEAD) * KV_LORA, 0xABCD);
    let kv_b_buf: PinnedBuffer = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&kv_b));
    let mut ref_out = vec![0.0f32; N_HEADS * V_HEAD];
    kernels::mla_decode_metal(
        ctx,
        &q,
        &c_kv,
        &k_pe,
        &kv_b_buf,
        N_HEADS,
        QK_NOPE,
        QK_ROPE,
        V_HEAD,
        KV_LORA,
        seq_len,
        scale,
        &mut ref_out,
    )
    .expect("mla_decode_metal");
    let n_blocks_per_row = KV_LORA / Q8_0_BLOCK_ELEMS;
    let row_bytes = n_blocks_per_row * Q8_0_BLOCK_BYTES;
    let mut c_kv_q8 = vec![0u8; seq_len * row_bytes];
    for t in 0..seq_len {
        let src_row = &c_kv[t * KV_LORA..(t + 1) * KV_LORA];
        let dst_row = &mut c_kv_q8[t * row_bytes..(t + 1) * row_bytes];
        quantize_q8_0(src_row, dst_row).expect("quantize_q8_0");
    }
    let mut q8_out = vec![0.0f32; N_HEADS * V_HEAD];
    kernels::mla_decode_q8kv_metal(
        ctx,
        &q,
        &c_kv_q8,
        &k_pe,
        &kv_b_buf,
        N_HEADS,
        QK_NOPE,
        QK_ROPE,
        V_HEAD,
        KV_LORA,
        seq_len,
        scale,
        &mut q8_out,
    )
    .expect("mla_decode_q8kv_metal");
    let diff = max_abs_diff(&ref_out, &q8_out);
    assert!(diff < atol, "{label}: diff {diff:.3e} >= {atol:.0e}");
}
#[test]
fn q8kv_seq256_worst_case() {
    run_q8_vs_f32("seq=256 worst", 256, 1.0, 0.30);
}
#[test]
fn q8kv_seq1024_worst_case() {
    run_q8_vs_f32("seq=1024 worst", 1024, 1.0, 0.30);
}
#[test]
fn q8kv_seq2048_worst_case() {
    run_q8_vs_f32("seq=2048 worst", 2048, 1.0, 0.30);
}
#[test]
fn q8kv_seq256_realistic() {
    run_q8_vs_f32("seq=256 real", 256, 0.1, 0.03);
}
#[test]
fn q8kv_seq1024_realistic() {
    run_q8_vs_f32("seq=1024 real", 1024, 0.1, 0.03);
}
#[test]
fn q8kv_seq2048_realistic() {
    run_q8_vs_f32("seq=2048 real", 2048, 0.1, 0.03);
}

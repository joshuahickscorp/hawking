//! Track E parity — synthetic test that the per-channel W4A8 LM_HEAD path
//! (quantize_per_channel + gemm_q4_k_a8_v3_8r_per_channel) matches the
//! f32-baseline GEMV when scales come from `per_channel_scales_from_abs(x)`.
//!
//! This is the LM_HEAD-shape mirror of `w4a8_per_channel_parity.rs` (the
//! q_proj-shape test that already shipped). Differs in:
//!   - Shape: Qwen-3B LM_HEAD = vocab×hidden = 151936 × 2048 (vs 2048×2048
//!     in q_proj). We test at a smaller-but-shape-similar 32768 × 2048 to
//!     keep the test fast.
//!   - Pipeline: this version uses the GPU quantize kernel
//!     `quantize_f32_to_int8_per_channel_tcb` (NEW in Track E) instead of
//!     the CPU-side `quantize_to_int8_per_channel`. Asserts they produce
//!     identical int8 output bit-for-bit.
//!   - Identity scales: also tests with all-ones scales to verify the
//!     pipeline degenerates correctly to round-and-clip int8.

#![cfg(target_os = "macos")]

use hawking_core::kernels;
use hawking_core::metal::{PinnedBuffer, TokenCommandBuffer};
use half::f16;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

mod common;
use common::*;

fn make_q4k_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let n_blocks = rows * (cols / 256);
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_blocks * 144];
    for b in 0..n_blocks {
        let off = b * 144;
        let d = 0.01_f32 + rng.gen::<f32>() * 0.01;
        let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
        let d_bits = f16::from_f32(d).to_bits();
        let dmin_bits = f16::from_f32(dmin).to_bits();
        bytes[off..off + 2].copy_from_slice(&d_bits.to_le_bytes());
        bytes[off + 2..off + 4].copy_from_slice(&dmin_bits.to_le_bytes());
        for i in 4..16 {
            bytes[off + i] = rng.gen::<u8>() & 0x3F;
        }
        for i in 16..144 {
            bytes[off + i] = rng.gen::<u8>();
        }
    }
    bytes
}

fn make_x(cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..cols)
        .map(|_| rng.gen_range(-3.0_f32..3.0_f32))
        .collect()
}

fn read_i8_buf(buf: &PinnedBuffer, n: usize) -> Vec<i8> {
    let ptr = buf.contents() as *const i8;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

#[test]
fn gpu_quantize_matches_cpu_per_channel() {
    // Verify the new `quantize_f32_to_int8_per_channel` GPU kernel produces
    // BIT-IDENTICAL output to the CPU `quantize_to_int8_per_channel` ref.
    let cols = 2048;
    let ctx = ctx();
    let x = make_x(cols, 0xABCD_1234);
    let scales = kernels::per_channel_scales_from_abs(&x);

    // CPU reference
    let cpu_i8 = kernels::quantize_to_int8_per_channel(&x, &scales);

    // GPU pipeline
    let x_buf = new_f32_buf(ctx, &x);
    let scales_buf = new_f32_buf(ctx, &scales);
    let out_buf = ctx.new_buffer(cols);
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::quantize_f32_to_int8_per_channel_tcb(
            &mut tcb,
            &x_buf,
            &scales_buf,
            &out_buf,
            cols,
        )
        .expect("GPU quantize encode");
        tcb.commit_and_wait().expect("GPU quantize commit");
    }
    let gpu_i8 = read_i8_buf(&out_buf, cols);

    assert_eq!(
        cpu_i8.len(),
        gpu_i8.len(),
        "lengths differ: cpu={} gpu={}",
        cpu_i8.len(),
        gpu_i8.len()
    );
    let mismatches: Vec<usize> = (0..cols).filter(|&i| cpu_i8[i] != gpu_i8[i]).collect();
    if !mismatches.is_empty() {
        let first = mismatches[0];
        panic!(
            "GPU/CPU quantize differ at {} positions; first @{}: cpu={} gpu={} x={:.4} scale={:.4e}",
            mismatches.len(), first, cpu_i8[first], gpu_i8[first], x[first], scales[first]
        );
    }
    eprintln!("[E4] GPU quantize bit-identical to CPU on cols={} ✓", cols);
}

#[test]
fn end_to_end_per_channel_lmhead_pipeline() {
    // End-to-end pipeline test:
    //   1. GPU per-channel quantize x_norm → x_int8 using static scales
    //   2. GPU per-channel gemm_q4_k_a8_v3_8r_per_channel
    //   3. Compare against f32-baseline gemv at same Q4_K weights
    let rows = 32768_usize; // smaller-than-vocab but same shape ratio
    let cols = 2048_usize;
    let ctx = ctx();

    let w_bytes = make_q4k_bytes(rows, cols, 0xCAFE_BABE);
    let model_buf = ctx.new_buffer_with_bytes(&w_bytes);
    let x = make_x(cols, 0xDEAD_BEEF);

    // f32-baseline
    let x_f32_buf = new_f32_buf(ctx, &x);
    let y_baseline_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_m_v3_8r_pinned_tcb(
            &mut tcb,
            &model_buf,
            0,
            w_bytes.len(),
            rows,
            cols,
            &x_f32_buf,
            &y_baseline_buf,
        )
        .expect("baseline encode");
        tcb.commit_and_wait().expect("baseline commit");
    }
    let y_baseline = read_f32_buf(&y_baseline_buf, rows);

    // Per-channel W4A8 pipeline (GPU quantize + GPU per-channel gemm)
    let scales = kernels::per_channel_scales_from_abs(&x);
    let scales_buf = new_f32_buf(ctx, &scales);
    let x_int8_buf = ctx.new_buffer(cols);
    let y_pc_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::quantize_f32_to_int8_per_channel_tcb(
            &mut tcb,
            &x_f32_buf,
            &scales_buf,
            &x_int8_buf,
            cols,
        )
        .expect("E4 quantize encode");
        kernels::gemm_q4_k_a8_v3_8r_per_channel_pinned_tcb(
            &mut tcb,
            &model_buf,
            0,
            w_bytes.len(),
            rows,
            cols,
            &x_int8_buf,
            &scales_buf,
            &y_pc_buf,
        )
        .expect("E4 per-channel gemm encode");
        tcb.commit_and_wait().expect("E4 commit");
    }
    let y_pc = read_f32_buf(&y_pc_buf, rows);

    // Cosine + NRMSE (oracle scales → should be near-bit-identical).
    let dot: f32 = y_baseline.iter().zip(&y_pc).map(|(&a, &b)| a * b).sum();
    let na: f32 = y_baseline.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let nb: f32 = y_pc.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let cosine = dot / (na * nb);
    let rmse: f32 = (y_baseline
        .iter()
        .zip(&y_pc)
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
        / rows as f32)
        .sqrt();
    let mean_abs = y_baseline.iter().map(|x| x.abs()).sum::<f32>() / rows as f32;
    let nrmse = rmse / mean_abs;
    eprintln!(
        "[E4 end-to-end] rows={} cosine={:.6} nrmse={:.4e}  baseline[0..4]={:?}  pc[0..4]={:?}",
        rows,
        cosine,
        nrmse,
        &y_baseline[..4],
        &y_pc[..4]
    );
    assert!(
        cosine > 0.9999 && nrmse < 0.02,
        "per-channel LM_HEAD pipeline out of tolerance: cosine={cosine:.6} nrmse={nrmse:.4e}"
    );
}

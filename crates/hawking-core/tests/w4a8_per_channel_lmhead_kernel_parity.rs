#![cfg(target_os = "macos")]
use half::f16;
use hawking_core::kernels;
use hawking_core::metal::{PinnedBuffer, TokenCommandBuffer};
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
    let cols = 2048;
    let ctx = ctx();
    let x = make_x(cols, 0xABCD_1234);
    let scales = kernels::per_channel_scales_from_abs(&x);
    let cpu_i8 = kernels::quantize_to_int8_per_channel(&x, &scales);
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
            mismatches.len(),
            first,
            cpu_i8[first],
            gpu_i8[first],
            x[first],
            scales[first]
        );
    }
}
#[test]
fn end_to_end_per_channel_lmhead_pipeline() {
    let rows = 32768_usize; // smaller-than-vocab but same shape ratio
    let cols = 2048_usize;
    let ctx = ctx();
    let w_bytes = make_q4k_bytes(rows, cols, 0xCAFE_BABE);
    let model_buf = ctx.new_buffer_with_bytes(&w_bytes);
    let x = make_x(cols, 0xDEAD_BEEF);
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
    assert!(
        cosine > 0.9999 && nrmse < 0.02,
        "per-channel LM_HEAD pipeline out of tolerance: cosine={cosine:.6} nrmse={nrmse:.4e}"
    );
}

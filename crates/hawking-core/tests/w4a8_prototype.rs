#![cfg(target_os = "macos")]
use half::f16;
use hawking_core::kernels;
use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use rand::Rng;
use rand_pcg::Pcg64Mcg;
use std::time::Instant;
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
fn new_buf_bytes(ctx: &MetalContext, bytes: &[u8]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytes)
}
#[test]
fn w4a8_parity_and_bw_saving() {
    let rows = 2048_usize;
    let cols = 2048_usize;
    let ctx = ctx();
    let w_bytes = make_q4k_bytes(rows, cols, 0xDEAD_BEEF);
    let model_buf = ctx.new_buffer_with_bytes(&w_bytes);
    let x = make_x(cols, 0xCAFE_F00D);
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
    let (x_int8, x_scales) = kernels::quantize_to_int8_per_block(&x, 256);
    assert_eq!(x_int8.len(), cols);
    assert_eq!(x_scales.len(), cols / 256);
    let x_int8_buf = new_buf_bytes(ctx, bytemuck::cast_slice(&x_int8));
    let x_scales_buf = new_f32_buf(ctx, &x_scales);
    let y_w4a8_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemm_q4_k_a8_v3_8r_pinned_tcb(
            &mut tcb,
            &model_buf,
            0,
            w_bytes.len(),
            rows,
            cols,
            &x_int8_buf,
            &x_scales_buf,
            &y_w4a8_buf,
        )
        .expect("W4A8 encode");
        tcb.commit_and_wait().expect("W4A8 commit");
    }
    let y_w4a8 = read_f32_buf(&y_w4a8_buf, rows);
    let dot_ab: f32 = y_baseline.iter().zip(&y_w4a8).map(|(&a, &b)| a * b).sum();
    let norm_a: f32 = y_baseline.iter().map(|&a| a * a).sum::<f32>().sqrt();
    let norm_b: f32 = y_w4a8.iter().map(|&a| a * a).sum::<f32>().sqrt();
    let cosine = dot_ab / (norm_a * norm_b);
    let rmse: f32 = (y_baseline
        .iter()
        .zip(&y_w4a8)
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
        / y_baseline.len() as f32)
        .sqrt();
    let mean_abs_baseline =
        y_baseline.iter().map(|x| x.abs()).sum::<f32>() / y_baseline.len() as f32;
    let nrmse = rmse / mean_abs_baseline;
    let mut max_recover_err = 0.0f32;
    for i in 0..cols {
        let b = i / 256;
        let recovered = x_int8[i] as f32 * x_scales[b];
        let err = (recovered - x[i]).abs();
        if err > max_recover_err {
            max_recover_err = err;
        }
    }
    assert!(
        cosine > 0.999 && nrmse < 0.05,
        "W4A8 output out of tolerance: cosine={cosine:.6} nrmse={nrmse:.4e}"
    );
    let warmup = 40;
    let calls = 200;
    {
        let mut run = || {
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
            .unwrap();
            tcb.commit_and_wait().unwrap();
        };
        for _ in 0..warmup {
            run();
        }
        let t0 = Instant::now();
        for _ in 0..calls {
            run();
        }
        let us = t0.elapsed().as_micros() as f64 / calls as f64;
    }
    {
        let mut run = || {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemm_q4_k_a8_v3_8r_pinned_tcb(
                &mut tcb,
                &model_buf,
                0,
                w_bytes.len(),
                rows,
                cols,
                &x_int8_buf,
                &x_scales_buf,
                &y_w4a8_buf,
            )
            .unwrap();
            tcb.commit_and_wait().unwrap();
        };
        for _ in 0..warmup {
            run();
        }
        let t0 = Instant::now();
        for _ in 0..calls {
            run();
        }
        let us = t0.elapsed().as_micros() as f64 / calls as f64;
    }
    {
        let mut run = || {
            let (x_q, x_s) = kernels::quantize_to_int8_per_block(&x, 256);
            let xq_buf = new_buf_bytes(ctx, bytemuck::cast_slice(&x_q));
            let xs_buf = new_f32_buf(ctx, &x_s);
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemm_q4_k_a8_v3_8r_pinned_tcb(
                &mut tcb,
                &model_buf,
                0,
                w_bytes.len(),
                rows,
                cols,
                &xq_buf,
                &xs_buf,
                &y_w4a8_buf,
            )
            .unwrap();
            tcb.commit_and_wait().unwrap();
        };
        for _ in 0..warmup {
            run();
        }
        let t0 = Instant::now();
        for _ in 0..calls {
            run();
        }
        let us = t0.elapsed().as_micros() as f64 / calls as f64;
    }
}

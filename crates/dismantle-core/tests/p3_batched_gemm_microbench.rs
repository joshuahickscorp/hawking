//! P3 — Microbench the batched Q4_K GEMM kernel against B back-to-back
//! single-vector GEMVs at production Qwen-3B shapes. Tells us whether
//! the kernel itself delivers the expected BW amortization or whether
//! the gap to 4× is in the surrounding orchestration.
//!
//! Shapes:
//!   - ffn_gate/up: rows=11008, cols=2048   (8 MB Q4_K weight)
//!   - ffn_down:    rows=2048,  cols=11008  (8 MB)
//!   - q/o_proj:    rows=2048,  cols=2048   (1.5 MB)

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use half::f16;
use once_cell::sync::Lazy;
use rand::Rng;
use rand_pcg::Pcg64Mcg;
use std::time::Instant;

fn ctx() -> &'static MetalContext {
    static CTX: Lazy<MetalContext> =
        Lazy::new(|| MetalContext::new().expect("Metal device required"));
    &CTX
}

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

fn new_f32_buf(ctx: &MetalContext, n: usize, seed: u64) -> PinnedBuffer {
    let mut rng = Pcg64Mcg::new(seed as u128);
    let data: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect();
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(&data))
}

fn bench_shape(rows: usize, cols: usize, calls: usize, warmup: usize) {
    let ctx = ctx();
    let w_bytes = make_q4k_bytes(rows, cols, 0xDEAD_BEEF + (rows + cols) as u64);
    let model_buf = ctx.new_buffer_with_bytes(&w_bytes);
    let w_offset = 0usize;
    let w_byte_size = w_bytes.len();

    let weight_mb = w_byte_size as f64 / (1024.0 * 1024.0);
    eprintln!("\n=== rows={rows} cols={cols} weight={weight_mb:.1} MiB ===");

    let f32_bytes = std::mem::size_of::<f32>();

    // B=1 baseline: gemv_q4_k_m_v3_8r_pinned_tcb
    {
        let x = new_f32_buf(ctx, cols, 0x1111);
        let y = ctx.new_buffer(rows * f32_bytes);
        let mut run = || {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_q4_k_m_v3_8r_pinned_tcb(
                &mut tcb, &model_buf, w_offset, w_byte_size,
                rows, cols, &x, &y,
            ).expect("gemv v3");
            tcb.commit_and_wait().expect("commit");
        };
        for _ in 0..warmup { run(); }
        let t0 = Instant::now();
        for _ in 0..calls { run(); }
        let mean_us = t0.elapsed().as_micros() as f64 / calls as f64;
        eprintln!("[B=1 v3_8r]              mean={:.1} us/call", mean_us);
    }

    // B=4 sequential (4 × gemv v3) — orchestration baseline for the
    // batched dispatcher under the same TCB.
    {
        let x_batch = new_f32_buf(ctx, 4 * cols, 0x2222);
        let y_batch = ctx.new_buffer(4 * rows * f32_bytes);
        let x_stride = cols * f32_bytes;
        let y_stride = rows * f32_bytes;
        let mut run = || {
            let mut tcb = TokenCommandBuffer::new(ctx);
            for bi in 0..4 {
                kernels::gemv_q4_k_m_v3_8r_pinned_off_tcb(
                    &mut tcb, &model_buf, w_offset, w_byte_size,
                    rows, cols,
                    &x_batch, bi * x_stride,
                    &y_batch, bi * y_stride,
                ).expect("gemv off");
            }
            tcb.commit_and_wait().expect("commit");
        };
        for _ in 0..warmup { run(); }
        let t0 = Instant::now();
        for _ in 0..calls { run(); }
        let mean_us = t0.elapsed().as_micros() as f64 / calls as f64;
        eprintln!("[B=4 sequential v3]      mean={:.1} us/call", mean_us);
    }

    // B=4 batched v2: gemm_q4_k_m_batched_v2_pinned_tcb
    {
        let x_batch = new_f32_buf(ctx, 4 * cols, 0x3333);
        let y_batch = ctx.new_buffer(4 * rows * f32_bytes);
        let mut run = || {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemm_q4_k_m_batched_v2_pinned_tcb(
                &mut tcb, &model_buf, w_offset, w_byte_size,
                rows, cols, 4, &x_batch, &y_batch,
            ).expect("gemm batched v2");
            tcb.commit_and_wait().expect("commit");
        };
        for _ in 0..warmup { run(); }
        let t0 = Instant::now();
        for _ in 0..calls { run(); }
        let mean_us = t0.elapsed().as_micros() as f64 / calls as f64;
        eprintln!("[B=4 batched v2]         mean={:.1} us/call", mean_us);
    }

    // B=4 batched v3: shmem-staged activation tile.
    {
        let x_batch = new_f32_buf(ctx, 4 * cols, 0x4444);
        let y_batch = ctx.new_buffer(4 * rows * f32_bytes);
        let mut run = || {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemm_q4_k_m_batched_v3_pinned_tcb(
                &mut tcb, &model_buf, w_offset, w_byte_size,
                rows, cols, 4, &x_batch, &y_batch,
            ).expect("gemm batched v3");
            tcb.commit_and_wait().expect("commit");
        };
        for _ in 0..warmup { run(); }
        let t0 = Instant::now();
        for _ in 0..calls { run(); }
        let mean_us = t0.elapsed().as_micros() as f64 / calls as f64;
        eprintln!("[B=4 batched v3 (shmem)] mean={:.1} us/call", mean_us);
    }
}

#[test]
fn p3_batched_gemm_microbench() {
    let calls = 200;
    let warmup = 40;
    // FFN gate/up shape (8 MB Q4_K weight) — dominant.
    bench_shape(11008, 2048, calls, warmup);
    // FFN down shape (Q4_K requant) — same weight bytes, transposed.
    bench_shape(2048, 11008, calls, warmup);
    // Attention q/o projection shape — smaller.
    bench_shape(2048, 2048, calls, warmup);
}

#![cfg(target_os = "macos")]
use half::f16;
use hawking_core::kernels::{self, predecode_q4_k_scale_table_f16};
use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use rand::Rng;
use rand_pcg::Pcg64Mcg;
use std::time::Instant;
mod common;
use common::*;
fn make_x(cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..cols)
        .map(|_| rng.gen_range(-3.0_f32..3.0_f32))
        .collect()
}
fn new_f16_buf(ctx: &MetalContext, data: &[f16]) -> PinnedBuffer {
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|h| h.to_bits().to_le_bytes())
        .collect();
    ctx.new_buffer_with_bytes(&bytes)
}
const WARMUP: usize = 30;
const ITERS: usize = 200;
fn time_dispatch<F>(label: &str, mut encode: F) -> f64
where
    F: FnMut(&mut TokenCommandBuffer<'_>),
{
    let ctx = ctx();
    for _ in 0..WARMUP {
        let mut tcb = TokenCommandBuffer::new(ctx);
        encode(&mut tcb);
        tcb.commit_and_wait().expect("warmup commit");
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let mut tcb = TokenCommandBuffer::new(ctx);
        encode(&mut tcb);
        tcb.commit_and_wait().expect("timed commit");
    }
    let elapsed = t0.elapsed();
    let us_per_call = elapsed.as_secs_f64() * 1e6 / ITERS as f64;
    us_per_call
}
fn bench_shape(rows: usize, cols: usize, tag: &str) {
    let ctx = ctx();
    let blocks = rows * (cols / 256);
    let w_bytes = make_q4k_bytes_pcg(rows, cols, 0xF165_8E1E ^ (rows as u64));
    let model_buf = ctx.new_buffer_with_bytes(&w_bytes);
    let x = make_x(cols, 0xCAFE_F00D);
    let x_buf = new_f32_buf(ctx, &x);
    let scales_f32 = kernels::predecode_q4_k_scale_table(&w_bytes);
    let scales_f32_buf = new_f32_buf(ctx, &scales_f32);
    let scales_f16 = predecode_q4_k_scale_table_f16(&w_bytes);
    let scales_f16_buf = new_f16_buf(ctx, &scales_f16);
    let y_f32_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let y_f16_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let wlen = w_bytes.len();
    let us_f32 = time_dispatch("f32 scales (2r)", |tcb| {
        kernels::gemv_q4_k_v4_predec_pinned_tcb(
            tcb,
            &model_buf,
            0,
            wlen,
            &scales_f32_buf,
            0,
            rows,
            cols,
            &x_buf,
            &y_f32_buf,
        )
        .expect("f32 predec encode");
    });
    let us_f16 = time_dispatch("f16 scales (2r)", |tcb| {
        kernels::gemv_q4_k_v4_predec_2r_f16s_pinned_tcb(
            tcb,
            &model_buf,
            0,
            wlen,
            &scales_f16_buf,
            0,
            rows,
            cols,
            &x_buf,
            &y_f16_buf,
        )
        .expect("f16s predec encode");
    });
    let weights = (blocks * 144) as f64;
    let x_bytes = (cols * 4) as f64;
    let y_bytes = (rows * 4) as f64;
    let bytes_f32 = weights + (blocks * 16 * 4) as f64 + x_bytes + y_bytes;
    let bytes_f16 = weights + (blocks * 16 * 2) as f64 + x_bytes + y_bytes;
    let gbps_f32 = bytes_f32 / (us_f32 * 1e3); // bytes / (µs*1e3 ns) -> GB/s
    let gbps_f16 = bytes_f16 / (us_f16 * 1e3);
    let speedup = (us_f32 - us_f16) / us_f32 * 100.0;
}
#[test]
#[ignore = "microbench — run with --ignored --nocapture; needs a free GPU"]
fn q4k_predec_f16s_bandwidth_bench() {
    bench_shape(2048, 2048, "attn-square 2048x2048");
    bench_shape(11008, 2048, "ffn-up 11008x2048");
    bench_shape(2048, 11008, "ffn-down 2048x11008");
}

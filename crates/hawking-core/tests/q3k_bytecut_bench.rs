#![cfg(target_os = "macos")]
use half::f16;
use hawking_core::kernels;
use hawking_core::metal::TokenCommandBuffer;
use hawking_core::quant::predecode_q3_k_scale_table;
use rand::Rng;
use rand_pcg::Pcg64Mcg;
use std::time::Instant;
mod common;
use common::*;
fn make_q3k_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let n_blocks = rows * (cols / 256);
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_blocks * 110];
    for b in 0..n_blocks {
        let off = b * 110;
        for i in 0..108 {
            bytes[off + i] = rng.gen::<u8>();
        }
        let d = 0.004 + rng.gen::<f32>() * 0.004;
        bytes[off + 108..off + 110].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
    }
    bytes
}
fn make_x(cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..cols)
        .map(|_| rng.gen_range(-3.0_f32..3.0_f32))
        .collect()
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
fn gbps(bytes: f64, us: f64) -> f64 {
    bytes / (us * 1e3)
}
fn bench_shape(rows: usize, cols: usize, tag: &str) {
    let ctx = ctx();
    let blocks = rows * (cols / 256); // blocks per output row-vector total
    let q3_w = make_q3k_bytes(rows, cols, 0x3D15_8E1E ^ (rows as u64));
    let q3_buf = ctx.new_buffer_with_bytes(&q3_w);
    let q3_scales = predecode_q3_k_scale_table(&q3_w); // 16 f32/block
    let q3_scales_buf = new_f32_buf(ctx, &q3_scales);
    let q4_w = make_q4k_bytes_pcg(rows, cols, 0xF165_8E1E ^ (rows as u64));
    let q4_buf = ctx.new_buffer_with_bytes(&q4_w);
    let q4_scales = kernels::predecode_q4_k_scale_table(&q4_w); // 16 f32/block
    let q4_scales_buf = new_f32_buf(ctx, &q4_scales);
    let x = make_x(cols, 0xCAFE_F00D);
    let x_buf = new_f32_buf(ctx, &x);
    let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let q3_wlen = q3_w.len();
    let q4_wlen = q4_w.len();
    let us_q3_fused = time_dispatch("Q3_K fused", |tcb| {
        kernels::gemv_q3_k_pinned_tcb(tcb, &q3_buf, 0, q3_wlen, rows, cols, &x_buf, &y_buf)
            .expect("q3_k fused encode");
    });
    let us_q3_fused_2r = time_dispatch("Q3_K fused 2r", |tcb| {
        kernels::gemv_q3_k_fused_2r_pinned_tcb(
            tcb, &q3_buf, 0, q3_wlen, rows, cols, &x_buf, &y_buf,
        )
        .expect("q3_k fused 2r encode");
    });
    let us_q3_predec = time_dispatch("Q3_K predec", |tcb| {
        kernels::gemv_q3_k_v4_predec_pinned_tcb(
            tcb,
            &q3_buf,
            0,
            q3_wlen,
            &q3_scales_buf,
            0,
            rows,
            cols,
            &x_buf,
            &y_buf,
        )
        .expect("q3_k predec encode");
    });
    let us_q4_predec = time_dispatch("Q4_K predec", |tcb| {
        kernels::gemv_q4_k_v4_predec_pinned_tcb(
            tcb,
            &q4_buf,
            0,
            q4_wlen,
            &q4_scales_buf,
            0,
            rows,
            cols,
            &x_buf,
            &y_buf,
        )
        .expect("q4_k predec encode");
    });
    let x_bytes = (cols * 4) as f64;
    let y_bytes = (rows * 4) as f64;
    let scale_f32_per_block = 16 * 4; // 16 f32 pre-decoded scales = 64 B
    let bpb_q3_fused = 110.0; // weights only (fused + fused_2r read identical bytes)
    let bpb_q3_predec = 110.0 + scale_f32_per_block as f64; // 174 B
    let bpb_q4_predec = 144.0 + scale_f32_per_block as f64; // 208 B
    let bytes_q3_fused = blocks as f64 * bpb_q3_fused + x_bytes + y_bytes;
    let bytes_q3_predec = blocks as f64 * bpb_q3_predec + x_bytes + y_bytes;
    let bytes_q4_predec = blocks as f64 * bpb_q4_predec + x_bytes + y_bytes;
    let mut q3_winner = "fused";
    let mut q3_us = us_q3_fused;
    if us_q3_fused_2r < q3_us {
        q3_winner = "fused_2r";
        q3_us = us_q3_fused_2r;
    }
    if us_q3_predec < q3_us {
        q3_winner = "predec";
        q3_us = us_q3_predec;
    }
    let r2_vs_fused = (us_q3_fused - us_q3_fused_2r) / us_q3_fused * 100.0; // + => 2r faster
    let bytecut = (us_q4_predec - q3_us) / us_q4_predec * 100.0; // + => Q3 faster (byte-cut holds)
    let r2_vs_q4 = (us_q4_predec - us_q3_fused_2r) / us_q4_predec * 100.0; // + => Q3 fused_2r faster
}
#[test]
#[ignore = "microbench — run with --ignored --nocapture; needs a free GPU"]
fn q3k_bytecut_gemv_bench() {
    bench_shape(2048, 2048, "attn-square 2048x2048");
    bench_shape(11008, 2048, "ffn-up 11008x2048");
    bench_shape(2048, 11008, "ffn-down 2048x11008");
}

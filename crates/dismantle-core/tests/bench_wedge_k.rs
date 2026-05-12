//! Wedge K timing micro-bench
//! Compares gemm_q4_k_m_fused_v2 vs gemm_q4_k_m_simdmat on realistic shapes.
//! Run via: cargo test --test bench_wedge_k -- --nocapture --ignored
#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, PinnedBuffer};
use once_cell::sync::Lazy;
use rand::Rng;
use rand_pcg::Pcg64Mcg;
use std::time::Instant;

fn ctx() -> &'static MetalContext {
    static CTX: Lazy<MetalContext> =
        Lazy::new(|| MetalContext::new().expect("Metal device required"));
    &CTX
}

fn fixed_input(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
}

fn synthetic_q4_k_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
    use half::f16;
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_blocks * 144];
    for b in 0..n_blocks {
        let off = b * 144;
        let d = 0.01 + rng.gen::<f32>() * 0.01;
        bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
        bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        for i in 4..144 { bytes[off + i] = rng.gen::<u8>(); }
    }
    bytes
}

fn bench_kernel(label: &str, iters: u32, mut f: impl FnMut()) -> f64 {
    // warmup
    for _ in 0..10 { f(); }
    let t0 = Instant::now();
    for _ in 0..iters { f(); }
    let elapsed = t0.elapsed();
    let mean_us = elapsed.as_secs_f64() * 1e6 / iters as f64;
    println!("  {label}: {mean_us:.1}us/call  ({iters} iters, {:.1}ms total)", elapsed.as_secs_f64()*1e3);
    mean_us
}

#[test]
#[ignore]
fn wedge_k_timing_bench() {
    let ctx = ctx();
    let iters = 200u32;

    // cols must be a multiple of 256 (Q4_K block = 256 elems).
    // DeepSeek-V2 Lite: gate/up rows=1408 cols=2048; shared FFN rows=2048 cols=10944≈cols via Q4K.
    for (rows, cols) in [(1408usize, 2048usize), (2048, 2048), (2048, 7168)] {
        println!("\n[WedgeK bench] rows={rows} cols={cols}");
        let n_blocks = rows * (cols / 256);
        let w_bytes = synthetic_q4_k_bytes(n_blocks, 42);
        let x = fixed_input(cols, 0xDEAD_BEEF);
        let model_buf = ctx.new_buffer_with_bytes(&w_bytes);
        let mut out_v2 = vec![0.0f32; rows];
        let mut out_sm = vec![0.0f32; rows];

        let v2_us = bench_kernel("gemm_q4_k_m_fused_v2 (pinned)", iters, || {
            kernels::gemv_q4_k_m_v2_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut out_v2).unwrap();
        });

        let sm_us = bench_kernel("gemm_q4_k_m_simdmat (pinned)", iters, || {
            kernels::gemv_q4_k_m_simdmat_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut out_sm).unwrap();
        });

        println!("  speedup: {:.2}x  (v2={v2_us:.1}us → simdmat={sm_us:.1}us)", v2_us / sm_us);
    }
}

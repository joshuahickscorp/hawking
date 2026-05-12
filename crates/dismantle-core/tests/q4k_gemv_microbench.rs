//! Micro-bench: Q4_K_M GEMV variants at rows=1408, cols=2048.
//! Measures mean_us per call across 200 warm + 1000 timed calls.
//! Run with: cargo test --release --test q4k_gemv_microbench -- --nocapture
#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, PinnedBuffer};
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

fn make_x(cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..cols).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
}

fn bench_variant(
    name: &str,
    rows: usize,
    cols: usize,
    warmup: usize,
    calls: usize,
    run: &mut dyn FnMut() -> (),
) {
    for _ in 0..warmup {
        run();
    }
    let t0 = Instant::now();
    for _ in 0..calls {
        run();
    }
    let elapsed_us = t0.elapsed().as_micros() as f64;
    let mean_us = elapsed_us / calls as f64;
    println!(
        "[microbench] {name:30} rows={rows} cols={cols}  mean={mean_us:.1} us/call  ({calls} calls)"
    );
}

#[test]
fn q4k_gemv_microbench_all() {
    let rows = 1408_usize;
    let cols = 2048_usize;
    let warmup = 100;
    let calls = 500;

    let ctx = ctx();
    let w_bytes = make_q4k_bytes(rows, cols, 0xDEAD_BEEF);
    let x = make_x(cols, 0x1234_5678);
    let model_buf = ctx.new_buffer_with_bytes(&w_bytes);
    let w_offset = 0usize;
    let w_byte_size = w_bytes.len();

    // v2_pinned
    {
        let mut out = vec![0.0f32; rows];
        bench_variant("gemv_q4_k_m_v2_pinned", rows, cols, warmup, calls, &mut || {
            kernels::gemv_q4_k_m_v2_pinned(ctx, &model_buf, w_offset, w_byte_size, rows, cols, &x, &mut out)
                .expect("v2_pinned");
        });
    }

    // simdmat
    {
        let mut out = vec![0.0f32; rows];
        bench_variant("gemv_q4_k_m_simdmat_pinned", rows, cols, warmup, calls, &mut || {
            kernels::gemv_q4_k_m_simdmat_pinned(ctx, &model_buf, w_offset, w_byte_size, rows, cols, &x, &mut out)
                .expect("simdmat");
        });
    }

    // v3_8r
    {
        let mut out = vec![0.0f32; rows];
        bench_variant("gemv_q4_k_m_v3_8r_pinned", rows, cols, warmup, calls, &mut || {
            kernels::gemv_q4_k_m_v3_8r_pinned(ctx, &model_buf, w_offset, w_byte_size, rows, cols, &x, &mut out)
                .expect("v3_8r");
        });
    }

    // v3_dual
    {
        let mut out = vec![0.0f32; rows];
        bench_variant("gemv_q4_k_m_v3_dual_pinned", rows, cols, warmup, calls, &mut || {
            kernels::gemv_q4_k_m_v3_dual_pinned(ctx, &model_buf, w_offset, w_byte_size, rows, cols, &x, &mut out)
                .expect("v3_dual");
        });
    }

    // v3_llama (Approach 3: 4 rows/simdgroup, sumy trick)
    {
        let mut out = vec![0.0f32; rows];
        bench_variant("gemv_q4_k_m_v3_llama_pinned", rows, cols, warmup, calls, &mut || {
            kernels::gemv_q4_k_m_v3_llama_pinned(ctx, &model_buf, w_offset, w_byte_size, rows, cols, &x, &mut out)
                .expect("v3_llama");
        });
    }

    // fused_simd (simdgroup_matrix, copy path — Approach 2 reference, last to avoid thermal inflation)
    {
        let mut out = vec![0.0f32; rows];
        bench_variant("gemv_q4_k_m_simd (simdmat-hw)", rows, cols, warmup, 100, &mut || {
            kernels::gemv_q4_k_m_simd(ctx, &w_bytes, rows, cols, &x, &mut out)
                .expect("fused_simd");
        });
    }
}

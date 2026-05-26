//! GPU `quantize_f32_to_int8_per_block_scaled` must bit-match the CPU
//! reference `quantize_to_int8_per_block_scaled`. The AWQ Option B path
//! folds an activation-side divide (x / s) into the existing per-block
//! int8 quantize; this test ensures the fused GPU variant produces the
//! same int8 bytes and per-block scales as the explicit divide-then-
//! quantize CPU pipeline.
//!
//! Range chosen so the scaled values stay well within the int8 working
//! range; if a future smoothing JSON pushes outside, raise the range.

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use once_cell::sync::Lazy;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

fn ctx() -> &'static MetalContext {
    static CTX: Lazy<MetalContext> =
        Lazy::new(|| MetalContext::new().expect("Metal device required"));
    &CTX
}

fn new_f32_buf(ctx: &MetalContext, data: &[f32]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(data))
}

fn read_i8(buf: &PinnedBuffer, n: usize) -> Vec<i8> {
    let ptr = buf.contents() as *const i8;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

fn read_f32(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

/// Generate a smoothing vector with realistic AWQ-style range
/// (most channels ~1.0, some outliers up to ~5×) given a seed.
fn make_smoothing(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128 ^ 0xA_5BAEu128);
    (0..n)
        .map(|i| {
            // 5% of channels get an "outlier" factor up to 5x; rest cluster
            // around 1.0 with mild jitter — matches the layer_0 stats from
            // profiles/qwen3b_awq_smoothing.json.
            if i % 20 == 0 {
                rng.gen_range(2.0..5.0)
            } else {
                rng.gen_range(0.3..1.6)
            }
        })
        .collect()
}

fn run_one(ctx: &MetalContext, n: usize, seed: u64, range: f32) {
    let mut rng = Pcg64Mcg::new(seed as u128);
    let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-range..range)).collect();
    let s = make_smoothing(n, seed);
    let (cpu_int8, cpu_scales) =
        kernels::quantize_to_int8_per_block_scaled(&x, &s, 256);

    let x_buf = new_f32_buf(ctx, &x);
    let s_buf = new_f32_buf(ctx, &s);
    let int8_buf = ctx.new_buffer(n);
    let scales_buf = ctx.new_buffer((n / 256) * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::quantize_f32_to_int8_per_block_scaled_tcb(
            &mut tcb, &x_buf, &s_buf, &int8_buf, &scales_buf, n,
        )
        .expect("encode");
        tcb.commit_and_wait().expect("commit");
    }
    let gpu_int8 = read_i8(&int8_buf, n);
    let gpu_scales = read_f32(&scales_buf, n / 256);

    assert_eq!(
        cpu_scales, gpu_scales,
        "scales mismatch at n={n} seed={seed}"
    );
    let mut diffs = 0usize;
    let mut first_bad = None;
    for i in 0..n {
        if cpu_int8[i] != gpu_int8[i] {
            diffs += 1;
            if first_bad.is_none() {
                first_bad = Some((
                    i,
                    cpu_int8[i],
                    gpu_int8[i],
                    x[i],
                    s[i],
                    cpu_scales[i / 256],
                ));
            }
        }
    }
    assert_eq!(
        diffs, 0,
        "int8 mismatch at n={n} seed={seed}: {diffs} elems differ; first bad: {:?}",
        first_bad,
    );
}

#[test]
fn quantize_int8_scaled_kernel_matches_cpu_small() {
    run_one(ctx(), 256, 0xA11CE, 3.0);
}

#[test]
fn quantize_int8_scaled_kernel_matches_cpu_hidden_2048() {
    run_one(ctx(), 2048, 0xBEEF, 3.0);
}

#[test]
fn quantize_int8_scaled_kernel_matches_cpu_intermediate_11008() {
    run_one(ctx(), 11008, 0xC0FFEE, 1.5);
}

#[test]
fn quantize_int8_scaled_kernel_handles_all_zero_block() {
    let ctx = ctx();
    let mut x = vec![0.0f32; 2048];
    for i in 512..768 {
        x[i] = 1.5 * ((i as f32) - 640.0) / 128.0;
    }
    let s = make_smoothing(2048, 0xD15EA5E);
    let (cpu_int8, cpu_scales) =
        kernels::quantize_to_int8_per_block_scaled(&x, &s, 256);
    let x_buf = new_f32_buf(ctx, &x);
    let s_buf = new_f32_buf(ctx, &s);
    let int8_buf = ctx.new_buffer(2048);
    let scales_buf = ctx.new_buffer(8 * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::quantize_f32_to_int8_per_block_scaled_tcb(
            &mut tcb, &x_buf, &s_buf, &int8_buf, &scales_buf, 2048,
        )
        .expect("encode");
        tcb.commit_and_wait().expect("commit");
    }
    assert_eq!(cpu_scales, read_f32(&scales_buf, 8), "scales");
    assert_eq!(cpu_int8, read_i8(&int8_buf, 2048), "int8");
}

#[test]
fn quantize_int8_scaled_kernel_handles_zero_smoothing_channels() {
    // Degenerate s entries (≤ 1e-12) must clamp to inv_s = 0 on both
    // CPU and GPU so the result is just int8 zero at that channel — the
    // bake tool will never emit a zero, but we don't want a divergent
    // NaN if it ever does.
    let ctx = ctx();
    let n = 1024;
    let mut rng = Pcg64Mcg::new(0xBADu128);
    let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0f32..2.0)).collect();
    let mut s = make_smoothing(n, 0xBAD ^ 0x5EED);
    // Zero out a sprinkling of channels across two blocks.
    for i in [3, 17, 100, 257, 600] {
        s[i] = 0.0;
    }
    let (cpu_int8, cpu_scales) =
        kernels::quantize_to_int8_per_block_scaled(&x, &s, 256);
    let x_buf = new_f32_buf(ctx, &x);
    let s_buf = new_f32_buf(ctx, &s);
    let int8_buf = ctx.new_buffer(n);
    let scales_buf = ctx.new_buffer((n / 256) * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::quantize_f32_to_int8_per_block_scaled_tcb(
            &mut tcb, &x_buf, &s_buf, &int8_buf, &scales_buf, n,
        )
        .expect("encode");
        tcb.commit_and_wait().expect("commit");
    }
    assert_eq!(cpu_scales, read_f32(&scales_buf, n / 256), "scales");
    assert_eq!(cpu_int8, read_i8(&int8_buf, n), "int8");
}

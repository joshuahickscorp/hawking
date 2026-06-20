//! GPU `quantize_f32_to_int8_per_block` must bit-match the CPU reference
//! `quantize_to_int8_per_block`. Both use the same `scale = max|x|/127`
//! formula and round-to-nearest-with-clamp, so the only sources of
//! divergence would be: (a) a different per-block reduction order on the
//! GPU producing a different max_abs (only possible if any FP add went
//! into the reduction — it doesn't; we reduce fabs(x) with `max`, which
//! is order-independent for floats not involving NaN), or (b) a kernel
//! bug. We assert bit-identical bytes + scales.

#![cfg(target_os = "macos")]

use hawking_core::kernels;
use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use rand::Rng;
use rand_pcg::Pcg64Mcg;

mod common;
use common::*;

fn read_i8(buf: &PinnedBuffer, n: usize) -> Vec<i8> {
    let ptr = buf.contents() as *const i8;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

fn read_f32(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

fn run_one(ctx: &MetalContext, n: usize, seed: u64, range: f32) {
    let mut rng = Pcg64Mcg::new(seed as u128);
    let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-range..range)).collect();
    let (cpu_int8, cpu_scales) = kernels::quantize_to_int8_per_block(&x, 256);

    let x_buf = new_f32_buf(ctx, &x);
    let int8_buf = ctx.new_buffer(n);
    let scales_buf = ctx.new_buffer((n / 256) * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::quantize_f32_to_int8_per_block_tcb(&mut tcb, &x_buf, &int8_buf, &scales_buf, n)
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
                first_bad = Some((i, cpu_int8[i], gpu_int8[i], x[i], cpu_scales[i / 256]));
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
fn quantize_int8_kernel_matches_cpu_small() {
    run_one(ctx(), 256, 0xA11CE, 3.0);
}

#[test]
fn quantize_int8_kernel_matches_cpu_hidden_2048() {
    run_one(ctx(), 2048, 0xBEEF, 3.0);
}

#[test]
fn quantize_int8_kernel_matches_cpu_intermediate_11008() {
    run_one(ctx(), 11008, 0xC0FFEE, 1.5);
}

#[test]
fn quantize_int8_kernel_handles_all_zero_block() {
    // A degenerate block (all zeros) → scale becomes 1.0 fallback. Ensure
    // CPU/GPU agree on the fallback path.
    let ctx = ctx();
    let mut x = vec![0.0f32; 2048];
    // Non-zero middle block so we exercise both paths in one dispatch.
    for i in 512..768 {
        x[i] = 1.5 * ((i as f32) - 640.0) / 128.0;
    }
    let (cpu_int8, cpu_scales) = kernels::quantize_to_int8_per_block(&x, 256);
    let x_buf = new_f32_buf(ctx, &x);
    let int8_buf = ctx.new_buffer(2048);
    let scales_buf = ctx.new_buffer(8 * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::quantize_f32_to_int8_per_block_tcb(&mut tcb, &x_buf, &int8_buf, &scales_buf, 2048)
            .expect("encode");
        tcb.commit_and_wait().expect("commit");
    }
    assert_eq!(cpu_scales, read_f32(&scales_buf, 8), "scales");
    assert_eq!(cpu_int8, read_i8(&int8_buf, 2048), "int8");
}

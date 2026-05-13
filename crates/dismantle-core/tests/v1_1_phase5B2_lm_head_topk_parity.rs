//! Phase 5B.2 parity: gemv_f16_metal_pinned_topk_tcb matches CPU reference.
//!
//! Tests:
//!   1. `phase5b2_topk_extraction_synthetic` — LM-head GEMV + top-K extraction
//!      via TCB produces top-K indices and values that match a CPU reference
//!      (full GEMV, then argsort).

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use half::f16;
use once_cell::sync::Lazy;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

fn ctx() -> &'static MetalContext {
    static CTX: Lazy<MetalContext> =
        Lazy::new(|| MetalContext::new().expect("Metal device required"));
    &CTX
}

fn fixed_f16(n: usize, seed: u64) -> Vec<f16> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n)
        .map(|_| f16::from_f32(rng.gen_range(-1.0_f32..1.0_f32)))
        .collect()
}

fn fixed_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
}

fn new_f16_buf(ctx: &MetalContext, data: &[f16]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(data))
}

fn new_f32_buf(ctx: &MetalContext, data: &[f32]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(data))
}

fn read_f32_buf(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

fn cpu_gemv_f16(w: &[f16], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    for r in 0..rows {
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += f32::from(w[r * cols + c]) * x[c];
        }
        out[r] = acc;
    }
    out
}

fn cpu_topk(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut indexed: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    indexed.sort_by(|a, b| {
        // Descending by value; ascending by index on ties.
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    indexed.into_iter().take(k).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

// ─────────────────────────────────────────────────────────────────────────────

/// Synthetic parity test: LM-head GEMV + top-K extraction.
/// Uses a small synthetic weight matrix (1024 rows × 128 cols) to avoid
/// loading 8GB of model weights.
#[test]
fn phase5b2_topk_extraction_synthetic() {
    let ctx = ctx();

    // Synthetic shape: 1024 vocab × 128 hidden
    let rows = 1024usize;
    let cols = 128usize;
    let k = 128usize;

    // Generate synthetic data with fixed seeds.
    let w = fixed_f16(rows * cols, 0xDEADBEEF);
    let x = fixed_f32(cols, 0xCAFEBABE);

    // CPU reference: full GEMV then top-K.
    let cpu_logits = cpu_gemv_f16(&w, rows, cols, &x);
    let cpu_topk = cpu_topk(&cpu_logits, k);

    // GPU path: TCB-encoded GEMV + top-K.
    let w_buf = new_f16_buf(ctx, &w);
    let x_buf = new_f32_buf(ctx, &x);
    let logits_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let topk_buf = ctx.new_buffer(k * 2 * std::mem::size_of::<f32>());

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_f16_metal_pinned_topk_tcb(
        &mut tcb, &w_buf, rows, cols, k, &x_buf, &logits_buf, &topk_buf,
    )
    .expect("gemv_f16_metal_pinned_topk_tcb");
    tcb.commit_and_wait().expect("commit_and_wait");

    // Read back results.
    let gpu_logits = read_f32_buf(&logits_buf, rows);
    let gpu_topk_raw = read_f32_buf(&topk_buf, k * 2);

    // Parse top-K output: interleaved [idx0, val0, idx1, val1, ...]
    let mut gpu_topk = Vec::new();
    for i in 0..k {
        let idx = gpu_topk_raw[2 * i] as u32;
        let val = gpu_topk_raw[2 * i + 1];
        gpu_topk.push((idx, val));
    }

    // Verify 1: Full GEMV output matches (atol=1e-3 for fp16 quantization).
    let logits_diff = max_abs_diff(&cpu_logits, &gpu_logits);
    println!("Logits max diff: {}", logits_diff);
    assert!(
        logits_diff < 1e-3,
        "GEMV mismatch: diff={} (atol=1e-3)",
        logits_diff
    );

    // Verify 2: Top-K indices match exactly.
    for (i, (cpu_pair, gpu_pair)) in cpu_topk.iter().zip(gpu_topk.iter()).enumerate() {
        assert_eq!(
            cpu_pair.0, gpu_pair.0,
            "Top-K index mismatch at position {}: cpu={} gpu={}",
            i, cpu_pair.0, gpu_pair.0
        );
    }

    // Verify 3: Top-K values match (atol=1e-3 for fp16).
    for (i, (cpu_pair, gpu_pair)) in cpu_topk.iter().zip(gpu_topk.iter()).enumerate() {
        let diff = (cpu_pair.1 - gpu_pair.1).abs();
        assert!(
            diff < 1e-3,
            "Top-K value mismatch at position {}: cpu={} gpu={} diff={}",
            i, cpu_pair.1, gpu_pair.1, diff
        );
    }

    println!(
        "✓ phase5b2_topk_extraction_synthetic: rows={} cols={} k={} PASS",
        rows, cols, k
    );
}

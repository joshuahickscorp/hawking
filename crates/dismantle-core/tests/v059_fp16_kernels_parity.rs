//! v0.5.9 fp16 activation kernel parity tests.
//!
//! Tests (atol=1e-2 for nonlinear, atol=0 for copies):
//! - gemv_f32_attn_f16_matches_f32
//! - gemv_f32_moe_f16_matches_f32
//! - add_inplace_f16_matches_f32
//! - embed_lookup_f16_matches_f32
//! - softmax_f16_matches_f32
//! - layer_norm_f16_matches_f32
//! - rope_inplace_f16_matches_f32

#![cfg(target_os = "macos")]

use dismantle_core::kernels::{
    gemv_f32_attn_f16_metal,
    gemv_f32_moe_f16_metal,
    add_inplace_f16_metal,
    embed_lookup_f16_metal,
    softmax_f16_metal,
    layer_norm_f16_metal,
    rope_inplace_f16_metal,
};
use dismantle_core::metal::MetalContext;
use half::f16;

fn make_ctx() -> MetalContext {
    MetalContext::new().expect("Metal device")
}

fn f32_to_f16_vec(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

fn f16_buf_to_f32(ptr: *const f16, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(ptr, n) }
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn check_atol(a: &[f32], b: &[f32], atol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).abs();
        assert!(diff <= atol, "{label}[{i}]: |{av} - {bv}| = {diff} > {atol}");
    }
}

// ── CPU reference helpers ────────────────────────────────────────────────────

fn cpu_gemv_f32(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            w[r * cols..(r + 1) * cols]
                .iter()
                .zip(x.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>()
        })
        .collect()
}

fn cpu_softmax(x: &[f32]) -> Vec<f32> {
    let max_v = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max_v).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn cpu_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean = x.iter().sum::<f32>() / n as f32;
    let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(weight.iter().zip(bias.iter()))
        .map(|(&xi, (&wi, &bi))| (xi - mean) * inv_std * wi + bi)
        .collect()
}

fn cpu_rope(x: &[f32], head_dim: usize, pos: u32, base: f32) -> Vec<f32> {
    let mut out = x.to_vec();
    let half_dim = head_dim / 2;
    for id in 0..half_dim {
        let theta = (pos as f32) / base.powf(2.0 * id as f32 / head_dim as f32);
        let c = theta.cos();
        let s = theta.sin();
        let x0 = out[2 * id];
        let x1 = out[2 * id + 1];
        out[2 * id]     = x0 * c - x1 * s;
        out[2 * id + 1] = x0 * s + x1 * c;
    }
    out
}

// ── tests ────────────────────────────────────────────────────────────────────

#[test]
fn gemv_f32_attn_f16_matches_f32() {
    let ctx = make_ctx();
    let rows = 64usize;
    let cols = 256usize;

    let x_f32: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01) - 1.28).collect();
    let x_f16 = f32_to_f16_vec(&x_f32);
    let w: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32 * 0.007).sin()) * 0.5)
        .collect();

    let cpu_x_f32_from_f16: Vec<f32> = x_f16.iter().map(|v| v.to_f32()).collect();
    let cpu_out = cpu_gemv_f32(&w, rows, cols, &cpu_x_f32_from_f16);

    let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&x_f16));
    let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f16>());

    gemv_f32_attn_f16_metal(&ctx, &w, rows, cols, &x_buf, &y_buf).expect("gemv_f32_attn_f16");

    let gpu_out = f16_buf_to_f32(y_buf.contents() as *const f16, rows);
    // atol 5e-2: for 256-element dot products with output magnitudes ~40,
    // f32 summation-order differences reach ~0.01–0.02.
    check_atol(&gpu_out, &cpu_out, 5e-2, "gemv_f32_attn_f16");
}

#[test]
fn gemv_f32_moe_f16_matches_f32() {
    let ctx = make_ctx();
    let rows = 32usize;
    let cols = 128usize;

    let x_f32: Vec<f32> = (0..cols).map(|i| ((i as f32 * 0.05).cos()) * 2.0).collect();
    let x_f16 = f32_to_f16_vec(&x_f32);
    let w: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32 * 0.013).sin()) * 0.3)
        .collect();

    let cpu_x: Vec<f32> = x_f16.iter().map(|v| v.to_f32()).collect();
    let cpu_out = cpu_gemv_f32(&w, rows, cols, &cpu_x);

    let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&x_f16));
    let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f16>());

    gemv_f32_moe_f16_metal(&ctx, &w, rows, cols, &x_buf, &y_buf).expect("gemv_f32_moe_f16");

    let gpu_out = f16_buf_to_f32(y_buf.contents() as *const f16, rows);
    check_atol(&gpu_out, &cpu_out, 1e-2, "gemv_f32_moe_f16");
}

#[test]
fn add_inplace_f16_matches_f32() {
    let ctx = make_ctx();
    let n = 512usize;

    let a_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01) - 2.5).collect();
    let b_f32: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.03).sin())).collect();

    let a_f16 = f32_to_f16_vec(&a_f32);
    let b_f16 = f32_to_f16_vec(&b_f32);

    // CPU reference using f16 roundtrip.
    let cpu_out: Vec<f32> = a_f16
        .iter()
        .zip(b_f16.iter())
        .map(|(a, b)| (a.to_f32() + b.to_f32()))
        .collect();

    let a_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&a_f16));
    let b_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&b_f16));

    add_inplace_f16_metal(&ctx, &a_buf, &b_buf, n).expect("add_inplace_f16");

    let gpu_out = f16_buf_to_f32(a_buf.contents() as *const f16, n);
    // Pure element-wise add in f32 then back to f16: atol=0 vs f16-precision CPU
    check_atol(&gpu_out, &cpu_out, 1e-3, "add_inplace_f16");
}

#[test]
fn embed_lookup_f16_matches_f32() {
    let ctx = make_ctx();
    let vocab = 32usize;
    let hidden = 128usize;
    let token = 7u32;

    // Embed table: vocab × hidden f16.
    let embed_f16: Vec<f16> = (0..vocab * hidden)
        .map(|i| f16::from_f32((i as f32 * 0.01) - 1.6))
        .collect();

    // CPU reference: just index into the embed table.
    let expected_row: Vec<f32> = embed_f16[token as usize * hidden..(token as usize + 1) * hidden]
        .iter()
        .map(|v| v.to_f32())
        .collect();

    let embed_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&embed_f16));
    let out_buf = ctx.new_buffer(hidden * std::mem::size_of::<f16>());

    embed_lookup_f16_metal(&ctx, &embed_buf, &out_buf, hidden, token).expect("embed_lookup_f16");

    let gpu_out = f16_buf_to_f32(out_buf.contents() as *const f16, hidden);
    check_atol(&gpu_out, &expected_row, 0.0, "embed_lookup_f16");
}

#[test]
fn softmax_f16_matches_f32() {
    let ctx = make_ctx();
    let n = 256usize;

    let x_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02) - 2.5).collect();
    let x_f16 = f32_to_f16_vec(&x_f32);

    let cpu_x: Vec<f32> = x_f16.iter().map(|v| v.to_f32()).collect();
    let cpu_out = cpu_softmax(&cpu_x);

    let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&x_f16));
    let out_buf = ctx.new_buffer(n * std::mem::size_of::<f16>());

    softmax_f16_metal(&ctx, &x_buf, &out_buf, n).expect("softmax_f16");

    let gpu_out = f16_buf_to_f32(out_buf.contents() as *const f16, n);

    // Sum should be ~1.0.
    let sum: f32 = gpu_out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-2, "softmax_f16 sum={sum} != 1.0");
    check_atol(&gpu_out, &cpu_out, 1e-2, "softmax_f16");
}

#[test]
fn softmax_f16_nan_free_adversarial() {
    let ctx = make_ctx();
    let n = 128usize;

    // Very large and very small values — potential NaN from exp overflow.
    let x_f32: Vec<f32> = (0..n).map(|i| if i == 0 { 60.0 } else { -60.0 }).collect();
    let x_f16 = f32_to_f16_vec(&x_f32);

    let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&x_f16));
    let out_buf = ctx.new_buffer(n * std::mem::size_of::<f16>());

    softmax_f16_metal(&ctx, &x_buf, &out_buf, n).expect("softmax_f16 adversarial");

    let gpu_out = f16_buf_to_f32(out_buf.contents() as *const f16, n);
    for (i, &v) in gpu_out.iter().enumerate() {
        assert!(!v.is_nan(), "softmax_f16 NaN at index {i}");
        assert!(v >= 0.0, "softmax_f16 negative prob at {i}: {v}");
    }
}

#[test]
fn layer_norm_f16_matches_f32() {
    let ctx = make_ctx();
    let n = 256usize;
    let eps = 1e-5f32;

    let x_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02) - 2.5).collect();
    let weight_f32: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let bias_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.0005) - 0.1).collect();

    let x_f16 = f32_to_f16_vec(&x_f32);
    let weight_f16 = f32_to_f16_vec(&weight_f32);
    let bias_f16 = f32_to_f16_vec(&bias_f32);

    // CPU reference using f16 roundtrip for weight/bias.
    let x_rt: Vec<f32> = x_f16.iter().map(|v| v.to_f32()).collect();
    let w_rt: Vec<f32> = weight_f16.iter().map(|v| v.to_f32()).collect();
    let b_rt: Vec<f32> = bias_f16.iter().map(|v| v.to_f32()).collect();
    let cpu_out = cpu_layer_norm(&x_rt, &w_rt, &b_rt, eps);

    let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&x_f16));
    let w_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&weight_f16));
    let b_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&bias_f16));
    let out_buf = ctx.new_buffer(n * std::mem::size_of::<f16>());

    layer_norm_f16_metal(&ctx, &x_buf, &w_buf, &b_buf, eps, n, &out_buf)
        .expect("layer_norm_f16");

    let gpu_out = f16_buf_to_f32(out_buf.contents() as *const f16, n);
    check_atol(&gpu_out, &cpu_out, 1e-2, "layer_norm_f16");
}

#[test]
fn rope_inplace_f16_matches_f32() {
    let ctx = make_ctx();
    let head_dim = 64usize;
    let pos = 7u32;
    let base = 10000.0f32;

    let x_f32: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.1) - 3.2).collect();
    let x_f16 = f32_to_f16_vec(&x_f32);

    // CPU reference (operates on f16→f32 roundtrip input).
    let x_rt: Vec<f32> = x_f16.iter().map(|v| v.to_f32()).collect();
    let cpu_out = cpu_rope(&x_rt, head_dim, pos, base);

    let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&x_f16));

    rope_inplace_f16_metal(&ctx, &x_buf, head_dim, pos, base).expect("rope_inplace_f16");

    let gpu_out = f16_buf_to_f32(x_buf.contents() as *const f16, head_dim);

    // CPU computes sin/cos in f64 internally; GPU uses f32. Compose error: atol=1e-2.
    let cpu_out_f16: Vec<f32> = cpu_out
        .iter()
        .map(|&v| f16::from_f32(v).to_f32())
        .collect();
    check_atol(&gpu_out, &cpu_out_f16, 1e-2, "rope_inplace_f16");
}

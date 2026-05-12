//! v0.5.8 RMSNorm-GEMV fusion parity tests.
//!
//! Tests:
//! - rmsnorm_gemv_f32_attn_fused_matches_separate — atol=1e-3
//! - rmsnorm_gemv_q4k_pair_fused_matches_separate — atol=1e-3

#![cfg(target_os = "macos")]

use dismantle_core::kernels::{
    rmsnorm_gemv_f32_attn_pinned_metal,
    rmsnorm_gemv_q4k_pair_metal,
};
use dismantle_core::metal::MetalContext;
use half::f16;

fn make_ctx() -> MetalContext {
    MetalContext::new().expect("Metal device")
}

// ── CPU reference helpers ────────────────────────────────────────────────────

fn cpu_rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / x.len() as f32 + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi * inv_rms * wi)
        .collect()
}

fn cpu_gemv_f32(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            let row = &w[r * cols..(r + 1) * cols];
            row.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum::<f32>()
        })
        .collect()
}

/// Q4_K_M dequant: decode one 256-element block.
/// Mirrors the kernel's per-thread logic across all 256 elements.
fn cpu_dequant_q4k_block(block: &[u8]) -> Vec<f32> {
    assert_eq!(block.len(), 144);
    let d_bits: u16 = (block[0] as u16) | ((block[1] as u16) << 8);
    let dmin_bits: u16 = (block[2] as u16) | ((block[3] as u16) << 8);
    let d = f16::from_bits(d_bits).to_f32();
    let dmin = f16::from_bits(dmin_bits).to_f32();

    let mut out = vec![0.0f32; 256];
    for tid in 0..256usize {
        let sub = tid >> 5;
        let (s_byte, m_byte): (u8, u8) = if sub < 4 {
            (block[4 + sub] & 0x3F, block[4 + 4 + sub] & 0x3F)
        } else {
            let j = sub - 4;
            (
                (block[4 + 8 + j] & 0x0F) | ((block[4 + j] >> 6) << 4),
                (block[4 + 8 + j] >> 4) | ((block[4 + 4 + j] >> 6) << 4),
            )
        };
        let pair = sub >> 1;
        let upper = (sub & 1) != 0;
        let i = tid & 31;
        let q = block[16 + pair * 32 + i];
        let nib: u32 = if upper { (q as u32 >> 4) & 0x0F } else { q as u32 & 0x0F };
        out[tid] = d * (s_byte as f32) * (nib as f32) - dmin * (m_byte as f32);
    }
    out
}

fn cpu_gemv_q4k(w_bytes: &[u8], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    let blocks_per_row = cols / 256;
    assert_eq!(w_bytes.len(), rows * blocks_per_row * 144);
    (0..rows)
        .map(|r| {
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let bo = (r * blocks_per_row + b) * 144;
                let weights = cpu_dequant_q4k_block(&w_bytes[bo..bo + 144]);
                let x_off = b * 256;
                for k in 0..256 {
                    acc += weights[k] * x[x_off + k];
                }
            }
            acc
        })
        .collect()
}

fn check_atol(a: &[f32], b: &[f32], atol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).abs();
        assert!(
            diff <= atol,
            "{label}[{i}]: |{av} - {bv}| = {diff} > {atol}"
        );
    }
}

// ── rmsnorm_gemv_f32_attn_fused_matches_separate ────────────────────────────

#[test]
fn rmsnorm_gemv_f32_attn_fused_matches_separate() {
    let ctx = make_ctx();
    let rows = 128usize;
    let cols = 512usize;
    let eps = 1e-5f32;

    // Fixed-seed pseudo-random data.
    let x: Vec<f32> = (0..cols).map(|i| ((i as f32 * 0.1731).sin()) * 2.0).collect();
    let weight: Vec<f32> = (0..cols).map(|i| 1.0 + (i as f32 * 0.0013).cos() * 0.1).collect();
    let w: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32 * 0.0071).sin()) * 0.5)
        .collect();

    // CPU reference: separate rmsnorm then gemv.
    let x_norm = cpu_rmsnorm(&x, &weight, eps);
    let cpu_out = cpu_gemv_f32(&w, rows, cols, &x_norm);

    // GPU fused path.
    let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&x));
    let weight_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&weight));
    let w_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&w));
    let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

    rmsnorm_gemv_f32_attn_pinned_metal(
        &ctx, &w_buf, &x_buf, &weight_buf, eps, &out_buf, rows, cols,
    )
    .expect("rmsnorm_gemv_f32_attn_pinned_metal");

    let gpu_out: Vec<f32> = {
        let ptr = out_buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, rows) }.to_vec()
    };

    check_atol(&gpu_out, &cpu_out, 1e-3, "rmsnorm_gemv_f32_attn_fused");
}

// ── rmsnorm_gemv_q4k_pair_fused_matches_separate ────────────────────────────

fn make_q4k_random_bytes(rows: usize, cols: usize, seed: u32) -> Vec<u8> {
    let blocks_per_row = cols / 256;
    let total_bytes = rows * blocks_per_row * 144;
    let mut v = vec![0u8; total_bytes];
    let mut s = seed;
    for b in v.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (s >> 16) as u8;
    }
    // Fix d/dmin f16 bytes to small positive values to avoid NaN/Inf.
    for block in 0..(rows * blocks_per_row) {
        let bo = block * 144;
        // d = 0.1 as f16 = 0x2E66
        v[bo]     = 0x66;
        v[bo + 1] = 0x2E;
        // dmin = 0.05 as f16 = 0x2A66
        v[bo + 2] = 0x66;
        v[bo + 3] = 0x2A;
    }
    v
}

#[test]
fn rmsnorm_gemv_q4k_pair_fused_matches_separate() {
    let ctx = make_ctx();
    let rows = 32usize;
    let cols = 256usize;
    let eps = 1e-5f32;

    let x: Vec<f32> = (0..cols).map(|i| ((i as f32 * 0.2431).sin()) * 1.5).collect();
    let weight_f32: Vec<f32> = (0..cols)
        .map(|i| 1.0 + (i as f32 * 0.0019).cos() * 0.05)
        .collect();
    let weight_f16: Vec<f16> = weight_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let w_gate_bytes = make_q4k_random_bytes(rows, cols, 0xDEAD_BEEF);
    let w_up_bytes   = make_q4k_random_bytes(rows, cols, 0xCAFE_BABE);

    // CPU reference: rmsnorm then two separate Q4_K GEMVs.
    // Use f16→f32 converted weights to match GPU precision exactly.
    let weight_from_f16: Vec<f32> = weight_f16.iter().map(|v| v.to_f32()).collect();
    let x_norm = cpu_rmsnorm(&x, &weight_from_f16, eps);
    let cpu_gate = cpu_gemv_q4k(&w_gate_bytes, rows, cols, &x_norm);
    let cpu_up   = cpu_gemv_q4k(&w_up_bytes,   rows, cols, &x_norm);

    // GPU fused path.
    let x_buf        = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&x));
    let gate_out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let up_out_buf   = ctx.new_buffer(rows * std::mem::size_of::<f32>());

    rmsnorm_gemv_q4k_pair_metal(
        &ctx,
        &weight_f16,
        eps,
        &w_gate_bytes,
        &w_up_bytes,
        &gate_out_buf,
        &up_out_buf,
        &x_buf,
        rows,
        cols,
    )
    .expect("rmsnorm_gemv_q4k_pair_metal");

    let gpu_gate: Vec<f32> = {
        let ptr = gate_out_buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, rows) }.to_vec()
    };
    let gpu_up: Vec<f32> = {
        let ptr = up_out_buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, rows) }.to_vec()
    };

    check_atol(&gpu_gate, &cpu_gate, 1e-3, "q4k_pair_gate");
    check_atol(&gpu_up,   &cpu_up,   1e-3, "q4k_pair_up");
}

#[test]
fn rmsnorm_gemv_q4k_pair_larger() {
    let ctx = make_ctx();
    let rows = 64usize;
    let cols = 512usize;
    let eps = 1e-5f32;

    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.007 - 1.5)).collect();
    let weight_f32: Vec<f32> = vec![1.0f32; cols];
    let weight_f16: Vec<f16> = weight_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let w_gate_bytes = make_q4k_random_bytes(rows, cols, 0x1234_5678);
    let w_up_bytes   = make_q4k_random_bytes(rows, cols, 0x8765_4321);

    let weight_from_f16: Vec<f32> = weight_f16.iter().map(|v| v.to_f32()).collect();
    let x_norm = cpu_rmsnorm(&x, &weight_from_f16, eps);
    let cpu_gate = cpu_gemv_q4k(&w_gate_bytes, rows, cols, &x_norm);
    let cpu_up   = cpu_gemv_q4k(&w_up_bytes,   rows, cols, &x_norm);

    let x_buf        = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&x));
    let gate_out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let up_out_buf   = ctx.new_buffer(rows * std::mem::size_of::<f32>());

    rmsnorm_gemv_q4k_pair_metal(
        &ctx, &weight_f16, eps, &w_gate_bytes, &w_up_bytes,
        &gate_out_buf, &up_out_buf, &x_buf, rows, cols,
    )
    .expect("rmsnorm_gemv_q4k_pair_metal larger");

    let gpu_gate: Vec<f32> = {
        let ptr = gate_out_buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, rows) }.to_vec()
    };
    let gpu_up: Vec<f32> = {
        let ptr = up_out_buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, rows) }.to_vec()
    };

    // Tolerance 5e-3: fp32 accumulation in 512-element blocks yields ~0.002 error
    // for output magnitudes ~2600; the quantization error budget is much larger.
    check_atol(&gpu_gate, &cpu_gate, 5e-3, "q4k_pair_larger_gate");
    check_atol(&gpu_up,   &cpu_up,   5e-3, "q4k_pair_larger_up");
}

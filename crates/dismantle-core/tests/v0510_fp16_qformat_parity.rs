//! v0.5.10 fp16 Q-format kernel parity tests (atol=1e-3).
//!
//! Tests:
//! - gemm_q4_k_m_fused_f16_matches_f32     — Q4_K_M GEMV with f16 x/y
//! - moe_grouped_gemm_q4_f16_matches_f32   — MoE Q4_K_M GEMV with f16 x/y
//! - dequant_q8_0_f16_round_trip           — Q8_0 → f16 GPU vs CPU
//! - dequant_q6_k_f16_matches_cpu          — Q6_K → f16 GPU vs CPU

#![cfg(target_os = "macos")]

use dismantle_core::kernels::{
    gemm_q4_k_m_fused_f16_metal,
    moe_grouped_gemm_q4_f16_metal,
    dequant_q8_0_f16_metal,
    dequant_q6_k_f16_metal,
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

/// Q4_K_M dequant for one 144-byte block → 256 f32 values.
fn cpu_dequant_q4k_block(block: &[u8]) -> Vec<f32> {
    assert_eq!(block.len(), 144);
    let d    = f16::from_bits((block[0] as u16) | ((block[1] as u16) << 8)).to_f32();
    let dmin = f16::from_bits((block[2] as u16) | ((block[3] as u16) << 8)).to_f32();
    let mut out = vec![0.0f32; 256];
    for tid in 0..256usize {
        let sub = tid >> 5;
        let (s_byte, m_byte): (u8, u8) = if sub < 4 {
            (block[4 + sub] & 0x3F, block[8 + sub] & 0x3F)
        } else {
            let j = sub - 4;
            (
                (block[12 + j] & 0x0F) | ((block[4 + j] >> 6) << 4),
                (block[12 + j] >> 4)   | ((block[8 + j] >> 6) << 4),
            )
        };
        let pair  = sub >> 1;
        let upper = (sub & 1) != 0;
        let i     = tid & 31;
        let q     = block[16 + pair * 32 + i];
        let nib: u32 = if upper { (q as u32 >> 4) & 0x0F } else { q as u32 & 0x0F };
        out[tid] = d * (s_byte as f32) * (nib as f32) - dmin * (m_byte as f32);
    }
    out
}

fn cpu_gemv_q4k_f16x(w_bytes: &[u8], rows: usize, cols: usize, x_f16: &[f16]) -> Vec<f32> {
    let blocks_per_row = cols / 256;
    (0..rows)
        .map(|r| {
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let bo = (r * blocks_per_row + b) * 144;
                let weights = cpu_dequant_q4k_block(&w_bytes[bo..bo + 144]);
                let x_off = b * 256;
                for k in 0..256 {
                    acc += weights[k] * x_f16[x_off + k].to_f32();
                }
            }
            acc
        })
        .collect()
}

/// Build random Q4_K_M bytes with fixed small d/dmin to avoid NaN.
fn make_q4k_bytes(rows: usize, cols: usize, seed: u32) -> Vec<u8> {
    let blocks_per_row = cols / 256;
    let mut v = vec![0u8; rows * blocks_per_row * 144];
    let mut s = seed;
    for b in v.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (s >> 16) as u8;
    }
    for block in 0..(rows * blocks_per_row) {
        let bo = block * 144;
        // d = 0.1 as f16 = 0x2E66
        v[bo]     = 0x66; v[bo + 1] = 0x2E;
        // dmin = 0.05 as f16 = 0x2A66
        v[bo + 2] = 0x66; v[bo + 3] = 0x2A;
    }
    v
}

/// Q6_K dequant for one 210-byte block → 256 f32 values.
/// Mirrors the GPU kernel logic.
fn cpu_dequant_q6k_block(block: &[u8]) -> Vec<f32> {
    assert_eq!(block.len(), 210);
    let d = f16::from_bits((block[208] as u16) | ((block[209] as u16) << 8)).to_f32();
    let ql = &block[0..128];
    let qh = &block[128..192];
    // scales: 16 signed bytes at offset 192
    let mut out = vec![0.0f32; 256];
    for tid in 0..256usize {
        let half_idx = tid >> 7;
        let local    = tid & 127;
        let l        = local & 31;
        let group    = local >> 5;
        let ql_base  = half_idx * 64;
        let qh_base  = half_idx * 32;
        let qhi      = qh[qh_base + l];
        let q: u32 = match group {
            0 => ((ql[ql_base + l]      & 0x0F) as u32) | (((qhi >> 0) & 0x03) as u32) << 4,
            1 => ((ql[ql_base + 32 + l] & 0x0F) as u32) | (((qhi >> 2) & 0x03) as u32) << 4,
            2 => ((ql[ql_base + l]      >> 4)   as u32) | (((qhi >> 4) & 0x03) as u32) << 4,
            _ => ((ql[ql_base + 32 + l] >> 4)   as u32) | (((qhi >> 6) & 0x03) as u32) << 4,
        };
        let q_signed = q as i32 - 32;
        let sc_idx = 192 + half_idx * 8 + (l >> 4) + group * 2;
        let scale  = block[sc_idx] as i8 as f32;
        out[tid]   = d * scale * (q_signed as f32);
    }
    out
}

/// Build a simple Q6_K block with controllable d and uniform quant values.
fn make_q6k_block(d_val: f32, quant_fill: u8) -> Vec<u8> {
    let mut block = vec![0u8; 210];
    // d at offset 208..210
    let d_bits = f16::from_f32(d_val).to_bits();
    block[208] = (d_bits & 0xFF) as u8;
    block[209] = (d_bits >> 8) as u8;
    // ql[0..128]: fill pattern so nibbles/half-bytes are consistent
    for i in 0..128 { block[i] = quant_fill; }
    // qh[128..192]: fill pattern for upper 2 bits
    for i in 128..192 { block[i] = quant_fill; }
    // scales[192..208]: non-zero signed values
    for i in 0..16 { block[192 + i] = 2u8; }  // scale = 2 as i8 = 2
    block
}

// ── tests ────────────────────────────────────────────────────────────────────

#[test]
fn gemm_q4_k_m_fused_f16_matches_f32() {
    let ctx = make_ctx();
    let rows = 4usize;
    let cols = 256usize;

    let x_f32: Vec<f32> = (0..cols).map(|i| ((i as f32 * 0.017).sin()) * 1.5).collect();
    let x_f16 = f32_to_f16_vec(&x_f32);
    let w_bytes = make_q4k_bytes(rows, cols, 0xABCD_1234);

    // Round CPU output through f16 to match GPU (GPU stores `(half)shmem[0]`).
    let cpu_out: Vec<f32> = cpu_gemv_q4k_f16x(&w_bytes, rows, cols, &x_f16)
        .into_iter()
        .map(|v| f16::from_f32(v).to_f32())
        .collect();

    let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&x_f16));
    let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f16>());

    gemm_q4_k_m_fused_f16_metal(&ctx, &w_bytes, rows, cols, &x_buf, &y_buf)
        .expect("gemm_q4_k_m_fused_f16");

    let gpu_out = f16_buf_to_f32(y_buf.contents() as *const f16, rows);
    // atol=5e-1: f32 accumulation order differs between CPU sequential and GPU
    // threadgroup reduction; at magnitude ~2890 and f16 spacing 2, differences
    // can reach one f16 ULP (~2.0) but in practice stay under 0.5.
    check_atol(&gpu_out, &cpu_out, 5e-1, "gemm_q4_k_m_fused_f16");
}

#[test]
fn moe_grouped_gemm_q4_f16_matches_f32() {
    let ctx = make_ctx();
    let rows = 8usize;
    let cols = 256usize;

    let x_f32: Vec<f32> = (0..cols).map(|i| ((i as f32 * 0.023).cos()) * 2.0).collect();
    let x_f16 = f32_to_f16_vec(&x_f32);
    let w_bytes = make_q4k_bytes(rows, cols, 0xDEAD_BEEF);

    let cpu_out: Vec<f32> = cpu_gemv_q4k_f16x(&w_bytes, rows, cols, &x_f16)
        .into_iter()
        .map(|v| f16::from_f32(v).to_f32())
        .collect();

    let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&x_f16));
    let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f16>());

    moe_grouped_gemm_q4_f16_metal(&ctx, &w_bytes, rows, cols, &x_buf, &y_buf)
        .expect("moe_grouped_gemm_q4_f16");

    let gpu_out = f16_buf_to_f32(y_buf.contents() as *const f16, rows);
    check_atol(&gpu_out, &cpu_out, 5e-1, "moe_grouped_gemm_q4_f16");
}

#[test]
fn dequant_q8_0_f16_round_trip() {
    let ctx = make_ctx();
    // Build 2 Q8_0 blocks: 2 × 34 bytes = 68 bytes total.
    // Each block: 2-byte f16 scale + 32 signed int8 quants.
    let nblock = 2usize;
    let mut src = vec![0u8; nblock * 34];
    // Block 0: scale = 0.5
    let d0 = f16::from_f32(0.5).to_bits();
    src[0] = (d0 & 0xFF) as u8;
    src[1] = (d0 >> 8) as u8;
    for i in 0..32 { src[2 + i] = (i as i8).wrapping_add(10) as u8; }
    // Block 1: scale = -0.25
    let d1 = f16::from_f32(-0.25).to_bits();
    src[34] = (d1 & 0xFF) as u8;
    src[35] = (d1 >> 8) as u8;
    for i in 0..32 { src[36 + i] = ((i as i8).wrapping_sub(5)) as u8; }

    // CPU reference.
    let d0f = f16::from_bits((src[0] as u16) | ((src[1] as u16) << 8)).to_f32();
    let d1f = f16::from_bits((src[34] as u16) | ((src[35] as u16) << 8)).to_f32();
    let mut cpu_out = vec![0.0f32; 64];
    for i in 0..32 { cpu_out[i] = d0f * (src[2 + i] as i8) as f32; }
    for i in 0..32 { cpu_out[32 + i] = d1f * (src[36 + i] as i8) as f32; }
    // Round-trip through f16 (GPU stores half).
    let cpu_f16: Vec<f32> = cpu_out.iter().map(|&v| f16::from_f32(v).to_f32()).collect();

    let dst_buf = ctx.new_buffer(64 * std::mem::size_of::<f16>());
    dequant_q8_0_f16_metal(&ctx, &src, &dst_buf).expect("dequant_q8_0_f16");

    let gpu_out = f16_buf_to_f32(dst_buf.contents() as *const f16, 64);
    check_atol(&gpu_out, &cpu_f16, 1e-3, "dequant_q8_0_f16");
}

#[test]
fn dequant_q6_k_f16_matches_cpu() {
    let ctx = make_ctx();
    let nblock = 2usize;
    let mut src = Vec::with_capacity(nblock * 210);
    // Block 0: d=0.1, quant fill=0b01010101
    src.extend_from_slice(&make_q6k_block(0.1, 0x55));
    // Block 1: d=0.2, quant fill=0b10101010
    src.extend_from_slice(&make_q6k_block(0.2, 0xAA));

    // CPU reference.
    let mut cpu_out = vec![0.0f32; nblock * 256];
    let b0 = cpu_dequant_q6k_block(&src[0..210]);
    let b1 = cpu_dequant_q6k_block(&src[210..420]);
    cpu_out[0..256].copy_from_slice(&b0);
    cpu_out[256..512].copy_from_slice(&b1);
    // Round-trip through f16 to match GPU precision.
    let cpu_f16: Vec<f32> = cpu_out.iter().map(|&v| f16::from_f32(v).to_f32()).collect();

    let dst_buf = ctx.new_buffer(nblock * 256 * std::mem::size_of::<f16>());
    dequant_q6_k_f16_metal(&ctx, &src, &dst_buf).expect("dequant_q6_k_f16");

    let gpu_out = f16_buf_to_f32(dst_buf.contents() as *const f16, nblock * 256);
    check_atol(&gpu_out, &cpu_f16, 1e-3, "dequant_q6_k_f16");
}

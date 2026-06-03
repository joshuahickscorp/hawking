//! Quantization formats: Q3_K, Q4_K_M, Q5_K_M, Q8_0, plus dequant helpers.
//!
//! CPU dequant materializes fp32 weights from the GGUF mmap on demand (reference path).
//! The Metal fast path fuses dequant inside the FMA loop in threadgroup memory so DRAM
//! only ships 4-bit weights. Metal kernel sources live in `shaders/quant.metal`.

use crate::gguf::{GgmlType, TensorInfo};
use crate::{Error, Result};
use half::f16;

/// Number of elements per K-quants super-block.
pub const Q_K: usize = 256;

/// Dequantize a tensor's bytes into a freshly allocated `Vec<f32>`.
/// Element count is read from the dims.
pub fn dequant_to_f32(info: &TensorInfo, bytes: &[u8]) -> Result<Vec<f32>> {
    let n_elems: usize = info.dims.iter().product::<u64>() as usize;
    let mut out = vec![0.0f32; n_elems];
    dequant_into(info.dtype, bytes, &mut out)?;
    Ok(out)
}

/// Dequantize a tensor's bytes into a freshly allocated `Vec<f16>`.
pub fn dequant_to_f16(info: &TensorInfo, bytes: &[u8]) -> Result<Vec<f16>> {
    let n_elems: usize = info.dims.iter().product::<u64>() as usize;
    let mut tmp = vec![0.0f32; n_elems];
    dequant_into(info.dtype, bytes, &mut tmp)?;
    Ok(tmp.into_iter().map(f16::from_f32).collect())
}

/// Dequantize raw quantized bytes into the provided fp32 destination.
/// `out.len()` is the canonical element count; it must match the
/// number implied by the bytes for a given dtype.
pub fn dequant_into(dtype: GgmlType, bytes: &[u8], out: &mut [f32]) -> Result<()> {
    match dtype {
        GgmlType::F32 => copy_f32(bytes, out),
        GgmlType::F16 => copy_f16(bytes, out),
        GgmlType::BF16 => copy_bf16(bytes, out),
        GgmlType::Q4_0 => dequant_q4_0(bytes, out),
        GgmlType::Q4_1 => dequant_q4_1(bytes, out),
        GgmlType::Q5_0 => dequant_q5_0(bytes, out),
        GgmlType::Q5_1 => dequant_q5_1(bytes, out),
        GgmlType::Q8_0 => dequant_q8_0(bytes, out),
        GgmlType::Q3_K => dequant_q3_k_into(bytes, out),
        GgmlType::Q4_K => dequant_q4_k(bytes, out),
        GgmlType::Q5_K => dequant_q5_k(bytes, out),
        GgmlType::Q6_K => dequant_q6_k(bytes, out),
        other => Err(Error::Kernel(format!(
            "dequant: type {:?} not implemented yet (Phase 0 covers F16/BF16/F32/\
             Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q3_K/Q4_K/Q5_K/Q6_K)",
            other
        ))),
    }
}

// ---------- Plain copies -------------------------------------------------

fn copy_f32(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    let need = out.len() * 4;
    if bytes.len() < need {
        return Err(Error::Kernel(format!(
            "copy_f32: have {}B need {}B",
            bytes.len(),
            need
        )));
    }
    for i in 0..out.len() {
        out[i] = f32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
    }
    Ok(())
}

fn copy_f16(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    let need = out.len() * 2;
    if bytes.len() < need {
        return Err(Error::Kernel(format!(
            "copy_f16: have {}B need {}B",
            bytes.len(),
            need
        )));
    }
    for i in 0..out.len() {
        let bits = u16::from_le_bytes(bytes[2 * i..2 * i + 2].try_into().unwrap());
        out[i] = f16::from_bits(bits).to_f32();
    }
    Ok(())
}

fn copy_bf16(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    let need = out.len() * 2;
    if bytes.len() < need {
        return Err(Error::Kernel(format!(
            "copy_bf16: have {}B need {}B",
            bytes.len(),
            need
        )));
    }
    for i in 0..out.len() {
        let bits = u16::from_le_bytes(bytes[2 * i..2 * i + 2].try_into().unwrap());
        // bf16 is the upper 16 bits of an fp32; pad with zero mantissa.
        let f = f32::from_bits((bits as u32) << 16);
        out[i] = f;
    }
    Ok(())
}

/// Validate a dequant target and return its block count: `out` must be a whole
/// number of blocks and `bytes` must hold at least that many block-bytes.
/// Consolidates the identical guard prologue across the `dequant_q*` family;
/// error strings are byte-identical to the inlined originals.
#[inline]
fn block_count(
    tag: &str,
    out_len: usize,
    bytes_len: usize,
    block_elems: usize,
    block_bytes: usize,
) -> Result<usize> {
    if out_len % block_elems != 0 {
        return Err(Error::Kernel(format!(
            "{tag}: out len not multiple of {block_elems}"
        )));
    }
    let nb = out_len / block_elems;
    if bytes_len < nb * block_bytes {
        return Err(Error::Kernel(format!(
            "{tag}: have {}B need {}B",
            bytes_len,
            nb * block_bytes
        )));
    }
    Ok(nb)
}

// ---------- Q8_0 ---------------------------------------------------------
//
// Block layout: { f16 d; int8 qs[32] }   total 34 bytes per 32 elems.

fn dequant_q8_0(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    const BLOCK_BYTES: usize = 34;
    const BLOCK_ELEMS: usize = 32;
    let nb = block_count("q8_0", out.len(), bytes.len(), BLOCK_ELEMS, BLOCK_BYTES)?;
    for b in 0..nb {
        let off = b * BLOCK_BYTES;
        let d =
            f16::from_bits(u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap())).to_f32();
        for i in 0..BLOCK_ELEMS {
            let q = bytes[off + 2 + i] as i8 as f32;
            out[b * BLOCK_ELEMS + i] = d * q;
        }
    }
    Ok(())
}


// ---------- Inverse quantize helpers (Q8_0 / Q4_K / Q6_K) ---------------
//
// Required by mixed_quant_store re-quant + the Qwen-dense LM head /
// FFN-down requant fast paths.

pub const Q8_0_BLOCK_ELEMS: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

pub fn quantize_q8_0(src: &[f32], dst: &mut [u8]) -> Result<()> {
    if src.len() % Q8_0_BLOCK_ELEMS != 0 {
        return Err(Error::Kernel(
            "q8_0 quantize: src len not multiple of 32".into(),
        ));
    }
    let nb = src.len() / Q8_0_BLOCK_ELEMS;
    let need = nb * Q8_0_BLOCK_BYTES;
    if dst.len() < need {
        return Err(Error::Kernel(format!(
            "q8_0 quantize: dst {}B need {}B",
            dst.len(), need
        )));
    }
    for b in 0..nb {
        let block = &src[b * Q8_0_BLOCK_ELEMS..(b + 1) * Q8_0_BLOCK_ELEMS];
        let amax = block.iter().copied().fold(0.0_f32, |m, v| m.max(v.abs()));
        let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
        let off = b * Q8_0_BLOCK_BYTES;
        let d_f16 = f16::from_f32(d);
        dst[off..off + 2].copy_from_slice(&d_f16.to_bits().to_le_bytes());
        for i in 0..Q8_0_BLOCK_ELEMS {
            let q = (block[i] * inv_d).round().clamp(-127.0, 127.0) as i8;
            dst[off + 2 + i] = q as u8;
        }
    }
    Ok(())
}

pub const Q4_K_BLOCK_BYTES: usize = 144;
pub const Q6_K_BLOCK_BYTES: usize = 210;

pub fn quantize_q4_k(src: &[f32], dst: &mut [u8]) -> Result<()> {
    if src.len() % Q_K != 0 {
        return Err(Error::Kernel("q4_K quantize: src len not multiple of 256".into()));
    }
    let nb = src.len() / Q_K;
    let need = nb * Q4_K_BLOCK_BYTES;
    if dst.len() < need {
        return Err(Error::Kernel(format!("q4_K quantize: dst {}B need {}B", dst.len(), need)));
    }
    for b in 0..nb {
        let block = &src[b * Q_K..(b + 1) * Q_K];
        let off = b * Q4_K_BLOCK_BYTES;
        for byte in &mut dst[off + 16..off + 144] { *byte = 0; }
        let mut sub_scale = [0.0f32; 8];
        let mut sub_min = [0.0f32; 8];
        for s in 0..8 {
            let vals = &block[s * 32..(s + 1) * 32];
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for &v in vals { if v < mn { mn = v; } if v > mx { mx = v; } }
            if mn > 0.0 { mn = 0.0; }
            if mx < 0.0 { mx = 0.0; }
            if (mx - mn).abs() < 1e-30 {
                sub_scale[s] = 0.0; sub_min[s] = -mn;
            } else {
                sub_scale[s] = (mx - mn) / 15.0; sub_min[s] = -mn;
            }
        }
        let max_scale = sub_scale.iter().copied().fold(0.0f32, f32::max);
        let max_min = sub_min.iter().copied().fold(0.0f32, f32::max);
        let d = if max_scale > 0.0 { max_scale / 63.0 } else { 0.0 };
        let dmin = if max_min > 0.0 { max_min / 63.0 } else { 0.0 };
        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
        let inv_dmin = if dmin > 0.0 { 1.0 / dmin } else { 0.0 };
        let mut sc_u6 = [0u8; 8];
        let mut mn_u6 = [0u8; 8];
        for s in 0..8 {
            sc_u6[s] = (sub_scale[s] * inv_d).round().clamp(0.0, 63.0) as u8;
            mn_u6[s] = (sub_min[s] * inv_dmin).round().clamp(0.0, 63.0) as u8;
        }
        dst[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        dst[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        encode_q_k_scale_min(&sc_u6, &mn_u6, &mut dst[off + 4..off + 16]);
        let qs = &mut dst[off + 16..off + 144];
        for s in 0..8 {
            let eff_scale = d * sc_u6[s] as f32;
            let eff_min = dmin * mn_u6[s] as f32;
            let inv_eff = if eff_scale > 0.0 { 1.0 / eff_scale } else { 0.0 };
            let pair = s / 2;
            let upper = (s % 2) == 1;
            let qbase = pair * 32;
            for i in 0..32 {
                let x = block[s * 32 + i];
                let nib = ((x + eff_min) * inv_eff).round().clamp(0.0, 15.0) as u8;
                let byte_idx = qbase + i;
                if upper {
                    qs[byte_idx] = (qs[byte_idx] & 0x0F) | ((nib & 0xF) << 4);
                } else {
                    qs[byte_idx] = (qs[byte_idx] & 0xF0) | (nib & 0xF);
                }
            }
        }
    }
    Ok(())
}

fn encode_q_k_scale_min(scales: &[u8; 8], mins: &[u8; 8], dst: &mut [u8]) {
    debug_assert!(dst.len() >= 12);
    for j in 0..4 {
        dst[j] = (scales[j] & 0x3F) | (((scales[4 + j] >> 4) & 0x3) << 6);
        dst[4 + j] = (mins[j] & 0x3F) | (((mins[4 + j] >> 4) & 0x3) << 6);
        dst[8 + j] = (scales[4 + j] & 0x0F) | ((mins[4 + j] & 0x0F) << 4);
    }
}

pub fn quantize_q6_k(src: &[f32], dst: &mut [u8]) -> Result<()> {
    if src.len() % Q_K != 0 {
        return Err(Error::Kernel("q6_K quantize: src len not multiple of 256".into()));
    }
    let nb = src.len() / Q_K;
    let need = nb * Q6_K_BLOCK_BYTES;
    if dst.len() < need {
        return Err(Error::Kernel(format!("q6_K quantize: dst {}B need {}B", dst.len(), need)));
    }
    for b in 0..nb {
        let block = &src[b * Q_K..(b + 1) * Q_K];
        let off = b * Q6_K_BLOCK_BYTES;
        for byte in &mut dst[off..off + 208] { *byte = 0; }
        let mut local_scale = [0.0f32; 16];
        for s in 0..16 {
            let mut amax = 0.0f32;
            for &v in &block[s * 16..(s + 1) * 16] {
                let av = v.abs();
                if av > amax { amax = av; }
            }
            if amax > 0.0 { local_scale[s] = amax / 32.0; }
        }
        let max_scale = local_scale.iter().copied().fold(0.0f32, f32::max);
        let d = if max_scale > 0.0 { max_scale / 127.0 } else { 0.0 };
        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
        let mut sc_i8 = [0i8; 16];
        for s in 0..16 {
            sc_i8[s] = (local_scale[s] * inv_d).round().clamp(0.0, 127.0) as i8;
        }
        for half in 0..2 {
            let ql_off = off + half * 64;
            let qh_off = off + 128 + half * 32;
            let base = half * 128;
            for l in 0..32 {
                for group in 0..4u32 {
                    let off_in_half = l + (group as usize) * 32;
                    let e = base + off_in_half;
                    let sub = e / 16;
                    let eff = d * sc_i8[sub] as f32;
                    let inv_eff = if eff != 0.0 { 1.0 / eff } else { 0.0 };
                    let q_signed = (block[e] * inv_eff).round().clamp(-32.0, 31.0) as i32;
                    let u = (q_signed + 32) as u8;
                    let low4 = u & 0xF;
                    let high2 = (u >> 4) & 0x3;
                    let ql_idx_in_half = if group & 1 == 0 { l } else { 32 + l };
                    let ql_byte = &mut dst[ql_off + ql_idx_in_half];
                    if group < 2 {
                        *ql_byte = (*ql_byte & 0xF0) | low4;
                    } else {
                        *ql_byte = (*ql_byte & 0x0F) | (low4 << 4);
                    }
                    let shift = (group * 2) as u8;
                    let qh_byte = &mut dst[qh_off + l];
                    *qh_byte = (*qh_byte & !(0x3 << shift)) | (high2 << shift);
                }
            }
        }
        for s in 0..16 {
            dst[off + 192 + s] = sc_i8[s] as u8;
        }
        dst[off + 208..off + 210].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
    }
    Ok(())
}


// ---------- Q4_0 ---------------------------------------------------------
//
// Block layout: { f16 d; uint8 qs[16] }   total 18 bytes per 32 elems.
// The 16 quant bytes encode 32 nibbles arranged so that elems j and
// j+16 share the same byte: low nibble → j, high nibble → j+16.
// Quants are signed (-8..7) after subtracting the bias.

fn dequant_q4_0(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    const BLOCK_BYTES: usize = 18;
    const BLOCK_ELEMS: usize = 32;
    let nb = block_count("q4_0", out.len(), bytes.len(), BLOCK_ELEMS, BLOCK_BYTES)?;
    for b in 0..nb {
        let off = b * BLOCK_BYTES;
        let d =
            f16::from_bits(u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap())).to_f32();
        let qs = &bytes[off + 2..off + 18];
        let dst = &mut out[b * BLOCK_ELEMS..(b + 1) * BLOCK_ELEMS];
        for j in 0..16 {
            let lo = (qs[j] & 0x0F) as i32 - 8;
            let hi = ((qs[j] >> 4) & 0x0F) as i32 - 8;
            dst[j] = d * lo as f32;
            dst[j + 16] = d * hi as f32;
        }
    }
    Ok(())
}

// ---------- Q4_1 ---------------------------------------------------------
//
// Block layout: { f16 d; f16 m; uint8 qs[16] }   total 20 bytes / 32 elems.
// Quants are unsigned (0..15); reconstructed value is d*q + m.

fn dequant_q4_1(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    const BLOCK_BYTES: usize = 20;
    const BLOCK_ELEMS: usize = 32;
    let nb = block_count("q4_1", out.len(), bytes.len(), BLOCK_ELEMS, BLOCK_BYTES)?;
    for b in 0..nb {
        let off = b * BLOCK_BYTES;
        let d =
            f16::from_bits(u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap())).to_f32();
        let m = f16::from_bits(u16::from_le_bytes(
            bytes[off + 2..off + 4].try_into().unwrap(),
        ))
        .to_f32();
        let qs = &bytes[off + 4..off + 20];
        let dst = &mut out[b * BLOCK_ELEMS..(b + 1) * BLOCK_ELEMS];
        for j in 0..16 {
            let lo = (qs[j] & 0x0F) as f32;
            let hi = ((qs[j] >> 4) & 0x0F) as f32;
            dst[j] = d * lo + m;
            dst[j + 16] = d * hi + m;
        }
    }
    Ok(())
}

// ---------- Q5_0 ---------------------------------------------------------
//
// Block layout: { f16 d; uint8 qh[4]; uint8 qs[16] }   total 22 bytes.
// `qh` packs 32 high bits — bit i is the 5th bit of element i.
// Low 4 bits live in `qs` with the same low/high split as Q4_0:
//   qs[j] low  → element j     ; qs[j] high → element j+16
// Quants are signed: subtract 16 after combining the 5 bits.

fn dequant_q5_0(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    const BLOCK_BYTES: usize = 22;
    const BLOCK_ELEMS: usize = 32;
    let nb = block_count("q5_0", out.len(), bytes.len(), BLOCK_ELEMS, BLOCK_BYTES)?;
    for b in 0..nb {
        let off = b * BLOCK_BYTES;
        let d =
            f16::from_bits(u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap())).to_f32();
        let qh = u32::from_le_bytes(bytes[off + 2..off + 6].try_into().unwrap());
        let qs = &bytes[off + 6..off + 22];
        let dst = &mut out[b * BLOCK_ELEMS..(b + 1) * BLOCK_ELEMS];
        for j in 0..16 {
            let lo = (qs[j] & 0x0F) as i32;
            let hi = ((qs[j] >> 4) & 0x0F) as i32;
            let h_lo = ((qh >> j) & 0x1) as i32;
            let h_hi = ((qh >> (j + 16)) & 0x1) as i32;
            let q_lo = (lo | (h_lo << 4)) - 16;
            let q_hi = (hi | (h_hi << 4)) - 16;
            dst[j] = d * q_lo as f32;
            dst[j + 16] = d * q_hi as f32;
        }
    }
    Ok(())
}

// ---------- Q5_1 ---------------------------------------------------------
//
// Block layout: { f16 d; f16 m; uint8 qh[4]; uint8 qs[16] } 24 bytes.
// Same nibble layout as Q5_0 but quants are unsigned (0..31) and the
// reconstructed value is d*q + m.

fn dequant_q5_1(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    const BLOCK_BYTES: usize = 24;
    const BLOCK_ELEMS: usize = 32;
    let nb = block_count("q5_1", out.len(), bytes.len(), BLOCK_ELEMS, BLOCK_BYTES)?;
    for b in 0..nb {
        let off = b * BLOCK_BYTES;
        let d =
            f16::from_bits(u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap())).to_f32();
        let m = f16::from_bits(u16::from_le_bytes(
            bytes[off + 2..off + 4].try_into().unwrap(),
        ))
        .to_f32();
        let qh = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
        let qs = &bytes[off + 8..off + 24];
        let dst = &mut out[b * BLOCK_ELEMS..(b + 1) * BLOCK_ELEMS];
        for j in 0..16 {
            let lo = (qs[j] & 0x0F) as i32;
            let hi = ((qs[j] >> 4) & 0x0F) as i32;
            let h_lo = ((qh >> j) & 0x1) as i32;
            let h_hi = ((qh >> (j + 16)) & 0x1) as i32;
            let q_lo = lo | (h_lo << 4);
            let q_hi = hi | (h_hi << 4);
            dst[j] = d * q_lo as f32 + m;
            dst[j + 16] = d * q_hi as f32 + m;
        }
    }
    Ok(())
}

// ---------- Q4_K (a.k.a. Q4_K_M) ---------------------------------------
//
// Super-block of 256 elements; layout:
//   f16 d;            // super-block scale
//   f16 dmin;         // super-block min
//   uint8 scales[12]; // 8x6-bit packed scales+mins (4 of each per byte)
//   uint8 qs[128];    // 256 4-bit quants
//
// Per 32-elem sub-block i (0..7):
//   x[j] = d * scale[i] * (q[j] & 0xF) - dmin * min[i]    for low nibble
//   x[j] = d * scale[i] * (q[j] >> 4)  - dmin * min[i]    for high nibble

fn dequant_q4_k(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    const BLOCK_BYTES: usize = 144;
    let nb = block_count("q4_k", out.len(), bytes.len(), Q_K, BLOCK_BYTES)?;
    for b in 0..nb {
        let off = b * BLOCK_BYTES;
        let d =
            f16::from_bits(u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap())).to_f32();
        let dmin = f16::from_bits(u16::from_le_bytes(
            bytes[off + 2..off + 4].try_into().unwrap(),
        ))
        .to_f32();
        let scales = &bytes[off + 4..off + 16];
        let qs = &bytes[off + 16..off + 144];

        // Decode the 8 (scale, min) pairs.
        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        decode_q_k_scale_min(scales, &mut sc, &mut mn);

        let dst = &mut out[b * Q_K..(b + 1) * Q_K];
        for sub in 0..8 {
            let s = d * sc[sub] as f32;
            let m = dmin * mn[sub] as f32;
            // Each sub-block is 32 elems; q4 packs 2 elems per byte, so 16
            // bytes per sub-block. Pairs (low, high) live in the same
            // 64-byte half: subs 0..4 share bytes 0..63, subs 4..8 share 64..127.
            // Specifically, low nibble of qs[sub*16+i] gives x[sub*32+i]
            // when sub<4, but for the K-quants Q4_K layout, low/high nibbles
            // alternate between sub-block pairs:
            //   sub=2k:   low nibble  of qs[k*32+i .. k*32+i+15]   -> elems 0..32
            //   sub=2k+1: high nibble of qs[k*32+i .. k*32+i+15]   -> elems 0..32
            let pair = sub / 2;
            let upper = (sub % 2) == 1;
            let qbase = pair * 32;
            for i in 0..32 {
                let q = qs[qbase + i];
                let nib = if upper { (q >> 4) & 0xF } else { q & 0xF };
                dst[sub * 32 + i] = s * nib as f32 - m;
            }
        }
    }
    Ok(())
}

// ---------- Q3_K ---------------------------------------------------------
//
// Super-block of 256 elements; layout (110 bytes):
//   uint8 hmask[32];  // high/sign bit, eight bit-planes over 32 columns
//   uint8 qs[64];     // low 2 bits
//   uint8 scales[12]; // 16x6-bit packed signed scales, stored +32
//   f16   d;          // super-block scale
//
// Per 16-elem sub-block i (0..15):
//   scale = unpacked_scale[i] - 32
//   q     = low2 - (high_bit_set ? 0 : 4)   // q in [-4, 3]
//   x     = d * scale * q
pub fn dequant_q3_k_into(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    const BLOCK_BYTES: usize = 110;
    let nb = block_count("q3_k", out.len(), bytes.len(), Q_K, BLOCK_BYTES)?;

    for b in 0..nb {
        let off = b * BLOCK_BYTES;
        let hmask = &bytes[off..off + 32];
        let qs = &bytes[off + 32..off + 96];
        let scale_bytes = &bytes[off + 96..off + 108];
        let d = f16::from_bits(u16::from_le_bytes(
            bytes[off + 108..off + 110].try_into().unwrap(),
        ))
        .to_f32();

        let mut scales = [0i8; 16];
        decode_q3_k_scales(scale_bytes, &mut scales);

        let dst = &mut out[b * Q_K..(b + 1) * Q_K];
        for half in 0..2 {
            let half_q = half * 32;
            let bit_base = half * 4;
            let elem_base = half * 128;
            for j in 0..4 {
                let shift = j * 2;
                let high_mask = 1u8 << (bit_base + j);
                for lane in 0..16 {
                    let q0_idx = half_q + lane;
                    let q1_idx = half_q + 16 + lane;
                    let q0 = ((qs[q0_idx] >> shift) & 0x03) as i8
                        - if (hmask[lane] & high_mask) != 0 { 0 } else { 4 };
                    let q1 = ((qs[q1_idx] >> shift) & 0x03) as i8
                        - if (hmask[16 + lane] & high_mask) != 0 { 0 } else { 4 };
                    let s0 = d * scales[half * 8 + j * 2] as f32;
                    let s1 = d * scales[half * 8 + j * 2 + 1] as f32;
                    dst[elem_base + j * 32 + lane] = s0 * q0 as f32;
                    dst[elem_base + j * 32 + 16 + lane] = s1 * q1 as f32;
                }
            }
        }
    }
    Ok(())
}

fn decode_q3_k_scales(src: &[u8], scales: &mut [i8; 16]) {
    debug_assert!(src.len() >= 12);
    let aux0 = u32::from_le_bytes(src[0..4].try_into().unwrap());
    let aux1 = u32::from_le_bytes(src[4..8].try_into().unwrap());
    let aux2 = u32::from_le_bytes(src[8..12].try_into().unwrap());
    let decoded = [
        (aux0 & 0x0f0f_0f0f) | (((aux2 >> 0) & 0x0303_0303) << 4),
        (aux1 & 0x0f0f_0f0f) | (((aux2 >> 2) & 0x0303_0303) << 4),
        ((aux0 >> 4) & 0x0f0f_0f0f) | (((aux2 >> 4) & 0x0303_0303) << 4),
        ((aux1 >> 4) & 0x0f0f_0f0f) | (((aux2 >> 6) & 0x0303_0303) << 4),
    ];
    for (chunk, word) in decoded.iter().enumerate() {
        let bytes = word.to_le_bytes();
        for (i, &v) in bytes.iter().enumerate() {
            scales[chunk * 4 + i] = v as i8 - 32;
        }
    }
}

/// Pre-decoded Q3_K sub-block scale table (worklist 1.6): for each 110-byte
/// block, the 16 `d * scale[i]` f32 values. Q3_K is symmetric (no min term,
/// unlike Q4_K's (ds, dm) pairs), so it's 16 f32/block. A Q3_K predec GEMV
/// reads these instead of unpacking the packed 6-bit scales + super-block `d`
/// on every call. Build once at load; pair with a `gemm_q3_k_v4_predec` kernel.
pub fn predecode_q3_k_scale_table(bytes: &[u8]) -> Vec<f32> {
    const BLOCK_BYTES: usize = 110;
    debug_assert_eq!(
        bytes.len() % BLOCK_BYTES,
        0,
        "predecode_q3_k_scale_table: len {} not a multiple of 110",
        bytes.len()
    );
    let nb = bytes.len() / BLOCK_BYTES;
    let mut out = vec![0.0f32; nb * 16];
    for b in 0..nb {
        let off = b * BLOCK_BYTES;
        let d = f16::from_bits(u16::from_le_bytes(
            bytes[off + 108..off + 110].try_into().unwrap(),
        ))
        .to_f32();
        let mut scales = [0i8; 16];
        decode_q3_k_scales(&bytes[off + 96..off + 108], &mut scales);
        for i in 0..16 {
            out[b * 16 + i] = d * scales[i] as f32;
        }
    }
    out
}

/// Decode the 12-byte packed (scale, min) array into two 8-element u8
/// arrays. Layout matches ggml's `get_scale_min_k4` exactly:
///
///   For j < 4:   scale[j] = q[j]   & 0x3F
///                min[j]   = q[j+4] & 0x3F
///   For j >= 4:  scale[j] = (q[j+4] & 0x0F) | ((q[j-4] >> 6) << 4)
///                min[j]   = (q[j+4] >>  4)  | ((q[j]   >> 6) << 4)
///
/// i.e. the upper 2 bits of bytes[0..8] are scattered into the high
/// bits of scales[4..8] and mins[4..8]; bytes[8..12] hold the low 4
/// bits of those same scales/mins.
fn decode_q_k_scale_min(src: &[u8], scales: &mut [u8; 8], mins: &mut [u8; 8]) {
    for j in 0..4 {
        scales[j] = src[j] & 0x3F;
        mins[j] = src[4 + j] & 0x3F;
    }
    for j in 0..4 {
        scales[4 + j] = (src[8 + j] & 0x0F) | ((src[j] >> 6) << 4);
        mins[4 + j] = (src[8 + j] >> 4) | ((src[4 + j] >> 6) << 4);
    }
}

// ---------- Q5_K ---------------------------------------------------------
//
// Super-block of 256 elements; layout (176 bytes):
//   f16 d;
//   f16 dmin;
//   uint8 scales[12]; // same packing as Q4_K
//   uint8 qh[32];     // 1 high bit per element  (256 bits / 8)
//   uint8 qs[128];    // 4 low bits per element  (same as Q4_K)

fn dequant_q5_k(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    const BLOCK_BYTES: usize = 176;
    let nb = block_count("q5_k", out.len(), bytes.len(), Q_K, BLOCK_BYTES)?;
    for b in 0..nb {
        let off = b * BLOCK_BYTES;
        let d =
            f16::from_bits(u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap())).to_f32();
        let dmin = f16::from_bits(u16::from_le_bytes(
            bytes[off + 2..off + 4].try_into().unwrap(),
        ))
        .to_f32();
        let scales = &bytes[off + 4..off + 16];
        let qh = &bytes[off + 16..off + 48];
        let qs = &bytes[off + 48..off + 176];

        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        decode_q_k_scale_min(scales, &mut sc, &mut mn);

        let dst = &mut out[b * Q_K..(b + 1) * Q_K];
        // Q5_K's `qh` stores 1 bit per (sub-block, column): bit `sub`
        // of qh[col] is the 5th bit of element `sub * 32 + col`.
        // Different layout from "1 bit per flat element"; matches
        // ggml's u1<<=2 stepping in dequantize_row_q5_K.
        for sub in 0..8 {
            let s = d * sc[sub] as f32;
            let m = dmin * mn[sub] as f32;
            let pair = sub / 2;
            let upper = (sub % 2) == 1;
            let qbase = pair * 32;
            for i in 0..32 {
                let lo = qs[qbase + i];
                let nib = if upper { (lo >> 4) & 0xF } else { lo & 0xF };
                let hi_bit = (qh[i] >> sub) & 0x1;
                let q5 = nib | (hi_bit << 4);
                let elem_idx = sub * 32 + i;
                dst[elem_idx] = s * q5 as f32 - m;
            }
        }
    }
    Ok(())
}

// ---------- Q6_K ---------------------------------------------------------
//
// Super-block of 256 elements; layout (210 bytes):
//   uint8 ql[128];    // lower 4 bits of each quant
//   uint8 qh[64];     // upper 2 bits of each quant
//   int8  scales[16]; // per-16-element scales
//   f16   d;          // super-block scale

fn dequant_q6_k(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    const BLOCK_BYTES: usize = 210;
    let nb = block_count("q6_k", out.len(), bytes.len(), Q_K, BLOCK_BYTES)?;
    for b in 0..nb {
        let off = b * BLOCK_BYTES;
        let ql = &bytes[off..off + 128];
        let qh = &bytes[off + 128..off + 192];
        let sc_bytes = &bytes[off + 192..off + 208];
        let d = f16::from_bits(u16::from_le_bytes(
            bytes[off + 208..off + 210].try_into().unwrap(),
        ))
        .to_f32();

        let dst = &mut out[b * Q_K..(b + 1) * Q_K];
        // Per upstream layout (ggml `dequantize_row_q6_K`): 256 elems
        // are processed in two 128-element halves. Each half consumes
        // 64 ql bytes, 32 qh bytes, and 8 int8 scales. Within a half,
        // for each column l in 0..32, four output elements are produced
        // at offsets l, l+32, l+64, l+96 with the four nibble/qh-shift
        // combinations below.
        for half in 0..2 {
            let ql_half = &ql[half * 64..half * 64 + 64];
            let qh_half = &qh[half * 32..half * 32 + 32];
            let sc_half = &sc_bytes[half * 8..half * 8 + 8];
            let base = half * 128;
            for l in 0..32 {
                let qhi = qh_half[l];
                let q1 = ((ql_half[l] & 0x0F) | (((qhi >> 0) & 0x3) << 4)) as i32 - 32;
                let q2 = ((ql_half[32 + l] & 0x0F) | (((qhi >> 2) & 0x3) << 4)) as i32 - 32;
                let q3 = ((ql_half[l] >> 4) | (((qhi >> 4) & 0x3) << 4)) as i32 - 32;
                let q4 = ((ql_half[32 + l] >> 4) | (((qhi >> 6) & 0x3) << 4)) as i32 - 32;
                let is = l / 16;
                let s1 = sc_half[is + 0] as i8 as f32;
                let s2 = sc_half[is + 2] as i8 as f32;
                let s3 = sc_half[is + 4] as i8 as f32;
                let s4 = sc_half[is + 6] as i8 as f32;
                dst[base + l + 0] = d * s1 * q1 as f32;
                dst[base + l + 32] = d * s2 * q2 as f32;
                dst[base + l + 64] = d * s3 * q3 as f32;
                dst[base + l + 96] = d * s4 * q4 as f32;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_zeros_round_trip() {
        // One block: f16 scale = 1.0, all quants zero.
        let mut bytes = vec![0u8; 34];
        bytes[0..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        let mut out = vec![1.0f32; 32];
        dequant_q8_0(&bytes, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q3_k_known_pattern() {
        let mut bytes = vec![0u8; 110];
        // Give all 16 sub-blocks scale byte 33 -> signed scale +1.
        for j in 0..16 {
            let l = 33u8;
            if j < 8 {
                bytes[96 + j] |= l & 0x0f;
            } else {
                bytes[96 + j - 8] |= (l & 0x0f) << 4;
            }
            bytes[96 + 8 + j % 4] |= (l >> 4) << (2 * (j / 4));
        }
        bytes[108..110].copy_from_slice(&f16::from_f32(2.0).to_bits().to_le_bytes());

        // Element 0: low2=3 and hmask bit set -> q=+3, value=6.
        bytes[0] = 0x01;
        bytes[32] = 0x03;
        // Element 1: low2=2 and hmask bit clear -> q=-2, value=-4.
        bytes[33] = 0x02;

        let mut out = vec![0.0f32; 256];
        dequant_q3_k_into(&bytes, &mut out).unwrap();
        assert_eq!(out[0], 6.0);
        assert_eq!(out[1], -4.0);
    }

    #[test]
    fn copy_f16_round_trip() {
        let src: Vec<f16> = [1.0, -1.5, 2.25]
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let bytes: Vec<u8> = src.iter().flat_map(|h| h.to_bits().to_le_bytes()).collect();
        let mut out = vec![0.0f32; 3];
        copy_f16(&bytes, &mut out).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-3);
        assert!((out[1] + 1.5).abs() < 1e-3);
        assert!((out[2] - 2.25).abs() < 1e-3);
    }

    #[test]
    fn q3_k_predec_table_matches_decode() {
        // Two blocks of pseudo-random bytes with known super-block d values.
        let mut bytes = vec![0u8; 220];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ((i * 37 + 11) & 0xFF) as u8;
        }
        bytes[108..110].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        bytes[218..220].copy_from_slice(&f16::from_f32(-0.25).to_bits().to_le_bytes());

        let table = predecode_q3_k_scale_table(&bytes);
        assert_eq!(table.len(), 32);
        for blk in 0..2 {
            let off = blk * 110;
            let d = f16::from_bits(u16::from_le_bytes(
                bytes[off + 108..off + 110].try_into().unwrap(),
            ))
            .to_f32();
            let mut scales = [0i8; 16];
            decode_q3_k_scales(&bytes[off + 96..off + 108], &mut scales);
            for i in 0..16 {
                assert_eq!(table[blk * 16 + i], d * scales[i] as f32, "block {blk} sub {i}");
            }
        }
    }
}

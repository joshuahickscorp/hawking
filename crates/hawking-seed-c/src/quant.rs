//! The four GGML dequantizers this fixture needs: Q8_0, Q5_0, Q6_K, Q4_K, plus F32/F16 passthrough.
//! Byte layouts and per-element formulas are the canonical GGML formats, matched to the predecessor's
//! `quant.rs` so reconstruction is bit-identical. f16<->f32 uses the `half` crate (IEEE primitive).

use crate::gguf::GgmlType;
use crate::{Error, Result};
use half::f16;

const QK: usize = 256; // K-quant super-block

#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}
#[inline]
pub fn f32_to_f16_bits(x: f32) -> u16 {
    f16::from_f32(x).to_bits()
}
#[inline]
fn rd_f16(b: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[off], b[off + 1]]))
}

/// Q8_0 — 34 bytes / 32 elems: {f16 d; i8 qs[32]}; y = d * qs[i].
pub fn dequant_q8_0(bytes: &[u8], out: &mut [f32]) {
    const BB: usize = 34;
    const BE: usize = 32;
    for b in 0..out.len() / BE {
        let off = b * BB;
        let d = rd_f16(bytes, off);
        for i in 0..BE {
            out[b * BE + i] = d * (bytes[off + 2 + i] as i8 as f32);
        }
    }
}

/// Q5_0 — 22 bytes / 32 elems: {f16 d; u32 qh; u8 qs[16]}; 5th bit from qh, quant signed -16.
pub fn dequant_q5_0(bytes: &[u8], out: &mut [f32]) {
    const BB: usize = 22;
    const BE: usize = 32;
    for b in 0..out.len() / BE {
        let off = b * BB;
        let d = rd_f16(bytes, off);
        let qh = u32::from_le_bytes(bytes[off + 2..off + 6].try_into().unwrap());
        let qs = &bytes[off + 6..off + 22];
        let dst = &mut out[b * BE..(b + 1) * BE];
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
}

/// Q6_K — 210 bytes / 256 elems: {u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d}; 6-bit signed -32.
pub fn dequant_q6_k(bytes: &[u8], out: &mut [f32]) {
    const BB: usize = 210;
    for b in 0..out.len() / QK {
        let off = b * BB;
        let ql = &bytes[off..off + 128];
        let qh = &bytes[off + 128..off + 192];
        let sc = &bytes[off + 192..off + 208]; // int8
        let d = rd_f16(bytes, off + 208);
        let dst = &mut out[b * QK..(b + 1) * QK];
        for half in 0..2 {
            let ql_h = &ql[half * 64..half * 64 + 64];
            let qh_h = &qh[half * 32..half * 32 + 32];
            let sc_h = &sc[half * 8..half * 8 + 8];
            let base = half * 128;
            for l in 0..32 {
                let qhi = qh_h[l];
                let q1 = ((ql_h[l] & 0x0F) | (((qhi >> 0) & 0x3) << 4)) as i32 - 32;
                let q2 = ((ql_h[32 + l] & 0x0F) | (((qhi >> 2) & 0x3) << 4)) as i32 - 32;
                let q3 = ((ql_h[l] >> 4) | (((qhi >> 4) & 0x3) << 4)) as i32 - 32;
                let q4 = ((ql_h[32 + l] >> 4) | (((qhi >> 6) & 0x3) << 4)) as i32 - 32;
                let is = l / 16;
                let s1 = sc_h[is] as i8 as f32;
                let s2 = sc_h[is + 2] as i8 as f32;
                let s3 = sc_h[is + 4] as i8 as f32;
                let s4 = sc_h[is + 6] as i8 as f32;
                dst[base + l] = d * s1 * q1 as f32;
                dst[base + l + 32] = d * s2 * q2 as f32;
                dst[base + l + 64] = d * s3 * q3 as f32;
                dst[base + l + 96] = d * s4 * q4 as f32;
            }
        }
    }
}

/// The 6-bit scale/min unpack from Q4_K's 12 scale bytes (ggml get_scale_min_k4).
fn q4k_scale_min(src: &[u8], sc: &mut [u8; 8], mn: &mut [u8; 8]) {
    for j in 0..4 {
        sc[j] = src[j] & 0x3F;
        mn[j] = src[4 + j] & 0x3F;
    }
    for j in 0..4 {
        sc[4 + j] = (src[8 + j] & 0x0F) | ((src[j] >> 6) << 4);
        mn[4 + j] = (src[8 + j] >> 4) | ((src[4 + j] >> 6) << 4);
    }
}

/// Q4_K — 144 bytes / 256 elems: {f16 d; f16 dmin; u8 scales[12]; u8 qs[128]}; y = d*sc*nib - dmin*mn.
pub fn dequant_q4_k(bytes: &[u8], out: &mut [f32]) {
    const BB: usize = 144;
    for b in 0..out.len() / QK {
        let off = b * BB;
        let d = rd_f16(bytes, off);
        let dmin = rd_f16(bytes, off + 2);
        let scales = &bytes[off + 4..off + 16];
        let qs = &bytes[off + 16..off + 144];
        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        q4k_scale_min(scales, &mut sc, &mut mn);
        let dst = &mut out[b * QK..(b + 1) * QK];
        for sub in 0..8 {
            let s = d * sc[sub] as f32;
            let m = dmin * mn[sub] as f32;
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
}

/// Dequantize a whole tensor's bytes into `out` (length = n_elements) by its GGML type.
pub fn dequant(dtype: GgmlType, bytes: &[u8], out: &mut [f32]) -> Result<()> {
    match dtype {
        GgmlType::F32 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = rd_f16(bytes, i * 2);
            }
        }
        GgmlType::Q8_0 => dequant_q8_0(bytes, out),
        GgmlType::Q5_0 => dequant_q5_0(bytes, out),
        GgmlType::Q6_K => dequant_q6_k(bytes, out),
        GgmlType::Q4_K => dequant_q4_k(bytes, out),
        other => return Err(Error::Gguf(format!("unsupported dequant type {other:?}"))),
    }
    Ok(())
}

/// Direct-quant: dequantize a SINGLE row (`cols` elements) of a row-major weight into `out`, reading
/// only that row's blocks from the compressed view. Requires `cols % block_size == 0` (always true in
/// GGUF). This is the bounded-tile primitive the direct operators use — the whole tensor is never
/// expanded to f32.
pub fn dequant_row(dtype: GgmlType, bytes: &[u8], row: usize, cols: usize, out: &mut [f32]) -> Result<()> {
    let (bs, bb) = dtype.block_layout();
    let (bs, bb) = (bs as usize, bb as usize);
    if bs == 0 || cols % bs != 0 {
        return Err(Error::Gguf(format!("row dequant needs cols%block==0 (cols={cols}, block={bs})")));
    }
    let blocks_per_row = cols / bs;
    let start = row * blocks_per_row * bb;
    let end = start + blocks_per_row * bb;
    dequant(dtype, &bytes[start..end], &mut out[..cols])
}

/// The tied embedding lookup with the predecessor's f16 round-trip: Q8_0 -> f32 -> round to f16 -> f32.
/// Reproduces Candidate B's f16-rounded embedding WITHOUT materializing the whole f16 table.
pub fn embed_row_f16(dtype: GgmlType, bytes: &[u8], token: usize, hidden: usize, out: &mut [f32]) -> Result<()> {
    dequant_row(dtype, bytes, token, hidden, out)?;
    for v in out.iter_mut().take(hidden) {
        *v = f16::from_f32(*v).to_f32();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_dequant_matches_whole_tensor() {
        // build a 4-row x 32-col Q8_0 tensor (each row = 1 block); row dequant must match whole dequant
        let (rows, cols) = (4usize, 32usize);
        let mut bytes = vec![0u8; rows * 34];
        for r in 0..rows {
            let d = f32_to_f16_bits(1.0).to_le_bytes();
            bytes[r * 34] = d[0];
            bytes[r * 34 + 1] = d[1];
            for i in 0..32 {
                bytes[r * 34 + 2 + i] = (r * 32 + i) as u8 as i8 as u8;
            }
        }
        let mut whole = vec![0f32; rows * cols];
        dequant(GgmlType::Q8_0, &bytes, &mut whole).unwrap();
        for r in 0..rows {
            let mut row = vec![0f32; cols];
            dequant_row(GgmlType::Q8_0, &bytes, r, cols, &mut row).unwrap();
            assert_eq!(&row[..], &whole[r * cols..(r + 1) * cols]);
        }
    }

    #[test]
    fn q8_0_roundtrip_simple() {
        // block: d = 1.0 (f16), qs = 0,1,2,...  -> y = 0,1,2,...
        let mut bytes = vec![0u8; 34];
        let d = f32_to_f16_bits(1.0).to_le_bytes();
        bytes[0] = d[0];
        bytes[1] = d[1];
        for i in 0..32 {
            bytes[2 + i] = i as u8;
        }
        let mut out = vec![0f32; 32];
        dequant_q8_0(&bytes, &mut out);
        for i in 0..32 {
            assert_eq!(out[i], i as f32);
        }
    }

    #[test]
    fn q5_0_center_is_zero() {
        // d=1, all nibbles 0 and high bits so quant = 0 -> value should be -16 (0 - 16)
        let mut bytes = vec![0u8; 22];
        let d = f32_to_f16_bits(1.0).to_le_bytes();
        bytes[0] = d[0];
        bytes[1] = d[1];
        let mut out = vec![0f32; 32];
        dequant_q5_0(&bytes, &mut out);
        assert_eq!(out[0], -16.0);
    }

    #[test]
    fn f16_primitive_roundtrips_scales() {
        for x in [0.0f32, 1.0, -2.5, 0.015625, 100.0] {
            let b = f32_to_f16_bits(x);
            assert_eq!(f16_to_f32(b), half::f16::from_f32(x).to_f32());
        }
    }
}

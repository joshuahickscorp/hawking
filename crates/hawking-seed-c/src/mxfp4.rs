//! MXFP4 dequantization (OCP microscaling FP4). GPT-OSS stores each MoE expert weight as `*.blocks`
//! (FP4 = E2M1, 2 codes per byte) plus `*.scales` (UE8 = E8M0, one shared power-of-two scale per block
//! of 32 values). Dequant: value = LUT_E2M1[code] * 2^(scale_byte - 127). Bounded, direct — the whole
//! expert set is never materialized; callers dequant one expert (or a bounded slice) at a time.

/// E2M1 magnitude+sign lookup for the 16 FP4 codes.
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, // +
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, // -
];

const BLOCK: usize = 32;

#[inline]
fn ue8_scale(byte: u8) -> f32 {
    // E8M0: biased exponent, value = 2^(e-127); 255 is reserved (treat as 0 to fail safe).
    if byte == 255 {
        0.0
    } else {
        // 2^(byte-127) via bit construction to avoid powf cost where possible
        f32::from_bits(((byte as u32) << 23).wrapping_add(0)) // exponent field = byte, mantissa 0 -> 2^(byte-127)
    }
}

/// Dequantize one row of `n` MXFP4 values: `blocks` = ceil(n/2) packed bytes, `scales` = ceil(n/32)
/// UE8 bytes. Writes `n` f32 into `out`.
pub fn dequant_row(blocks: &[u8], scales: &[u8], n: usize, out: &mut [f32]) {
    for i in 0..n {
        let byte = blocks[i / 2];
        let code = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 } as usize;
        let s = ue8_scale(scales[i / BLOCK]);
        out[i] = E2M1[code] * s;
    }
}

/// Dequantize a full `rows x n` MXFP4 weight (row-major) into f32. Each row: `row_block_bytes` packed
/// bytes + `row_scale_bytes` UE8 scales.
pub fn dequant_matrix(blocks: &[u8], scales: &[u8], rows: usize, n: usize, out: &mut [f32]) {
    let rbb = n.div_ceil(2);
    let rsb = n.div_ceil(BLOCK);
    for r in 0..rows {
        dequant_row(&blocks[r * rbb..(r + 1) * rbb], &scales[r * rsb..(r + 1) * rsb], n, &mut out[r * n..(r + 1) * n]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ue8_is_power_of_two() {
        assert_eq!(ue8_scale(127), 1.0); // 2^0
        assert_eq!(ue8_scale(128), 2.0); // 2^1
        assert_eq!(ue8_scale(126), 0.5); // 2^-1
    }

    #[test]
    fn e2m1_codes_and_row() {
        // one block of 2 values, codes: [+1.0 (2), +6.0 (7)] packed in one byte (low=2, high=7) => 0x72
        let blocks = [0x72u8];
        let scales = [127u8]; // scale 1.0
        let mut out = [0f32; 2];
        dequant_row(&blocks, &scales, 2, &mut out);
        assert_eq!(out, [1.0, 6.0]);
        // scale 2.0
        let scales2 = [128u8];
        dequant_row(&blocks, &scales2, 2, &mut out);
        assert_eq!(out, [2.0, 12.0]);
    }
}

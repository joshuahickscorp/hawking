//! Q80 mixed-representation kernel-facing packers and CPU oracles.
//!
//! These oracles execute **what the packed artifact encodes**. They are not a
//! claim about BF16 parent fidelity — that belongs to Gravity. A kernel that
//! matches this module is a correct kernel even if the representation is bad.
//!
//! # Pack-lane contract
//!
//! Pack must emit the existing Gravity containers. Kernels consume the
//! **bodies** after the 8-byte magic + u32le header-length + JSON header.
//! Do not invent a second on-disk family.
//!
//! | organ | container | body |
//! |---|---|---|
//! | `gate_proj` | `HGRAVB01` / `hawking.gravity.binary_sign_scale.v1` | `fp16 scales[groups] \|\| LSB-first signs` |
//! | `up_proj` | `HGRAVR02` / `hawking.gravity.binary_outlier_residual.v2` rice_q1_rms @ 2% | binary body + `u32 first` + rice(diffs) + `fp16 rms` + 1-bit signs |
//! | `down_proj` | `HGRAVS01` / `hawking.gravity.activation_weighted_svd_low_rank.v1` r160 b3 | left body then right body; each `fp16 scales \|\| packed codes` |
//!
//! Geometry the kernels assume (Qwen3-Coder-Next routed expert):
//! `gate`/`up` = `[512, 2048]`, `down` = `[2048, 512]`, `group_size` binary=128,
//! hgravs factor `bits=3` `group_size=64` `rank=160`.
//!
//! Forbidden token path: packed → dense `(rows×cols)` temporary → matvec.

use crate::{Error, Result};
use half::f16;

pub const Q80_BINARY_GROUP_SIZE: usize = 128;
pub const Q80_HGRAVS_GROUP_SIZE: usize = 64;
pub const Q80_HGRAVS_BITS: u8 = 3;
pub const Q80_HGRAVS_RANK: usize = 160;
pub const Q80_RICE_Q1_OUTLIER_RATIO: f64 = 0.02;
pub const Q80_GATE_ROWS: usize = 512;
pub const Q80_GATE_COLS: usize = 2048;
pub const Q80_DOWN_ROWS: usize = 2048;
pub const Q80_DOWN_COLS: usize = 512;

pub const MAGIC_BINARY: [u8; 8] = *b"HGRAVB01";
pub const MAGIC_RESIDUAL_COMPACT: [u8; 8] = *b"HGRAVR02";
pub const MAGIC_HGRAVS01: [u8; 8] = *b"HGRAVS01";

#[derive(Clone, Debug, PartialEq)]
pub struct BinaryGroupPacked {
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    pub groups_per_row: usize,
    pub scales_f16: Vec<u16>,
    pub signs: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RiceQ1Packed {
    pub binary: BinaryGroupPacked,
    pub first_index: u32,
    pub rice_k: u32,
    pub rice_bytes: Vec<u8>,
    pub outlier_count: usize,
    pub residual_scale_f16: u16,
    pub residual_signs: Vec<u8>,
    pub indices: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UniformFactorPacked {
    pub rows: usize,
    pub cols: usize,
    pub bits: u8,
    pub group_size: usize,
    pub groups: usize,
    pub bound: u16,
    pub scales_f16: Vec<u16>,
    pub codes: Vec<u8>,
}

pub fn packed_byte_count(count: usize, bits: u8) -> Result<usize> {
    if bits == 0 || bits > 8 {
        return Err(Error::Model("packed bit width must be 1..=8".into()));
    }
    count
        .checked_mul(usize::from(bits))
        .map(|bits_total| bits_total.div_ceil(8))
        .ok_or_else(|| Error::Model("packed byte count overflow".into()))
}

/// LSB-first unsigned pack. Same wire as the Q80 hgravs01 factor body.
pub fn pack_unsigned_lsb(codes: &[u8], bits: u8) -> Result<Vec<u8>> {
    pack_unsigned(codes, bits)
}

/// Inverse of [`pack_unsigned_lsb`]. `count` is the number of codes, including
/// any retained group padding the packer emitted.
pub fn unpack_unsigned_lsb(packed: &[u8], count: usize, bits: u8) -> Result<Vec<u8>> {
    if bits == 0 || bits > 8 {
        return Err(Error::Model("unsigned unpack bits must be 1..=8".into()));
    }
    let expected = packed_byte_count(count, bits)?;
    if packed.len() < expected {
        return Err(Error::Model(
            "unsigned unpack payload shorter than packed geometry".into(),
        ));
    }
    let mut out = Vec::with_capacity(count);
    for element in 0..count {
        let value = extract_unsigned(packed, element, bits);
        out.push(u8::try_from(value).map_err(|_| {
            Error::Model("unsigned unpack produced a code that does not fit u8".into())
        })?);
    }
    Ok(out)
}

/// MSB-first unsigned pack of the same codes. Decode must extract MSB-first
/// too; this is a bit-plane experiment, not a different codebook.
pub fn pack_unsigned_msb(codes: &[u8], bits: u8) -> Result<Vec<u8>> {
    if bits == 0 || bits > 8 {
        return Err(Error::Model("unsigned pack bits must be 1..=8".into()));
    }
    let mut bit_iter = Vec::with_capacity(codes.len() * usize::from(bits));
    for &code in codes {
        for bit in (0..bits).rev() {
            bit_iter.push(((code >> bit) & 1) != 0);
        }
    }
    Ok(pack_bits_lsb(bit_iter))
}

pub fn split_gravity_container<'a>(payload: &'a [u8], magic: &[u8; 8]) -> Result<(&'a [u8], &'a [u8])> {
    if payload.len() < 12 || payload[..8] != magic[..] {
        return Err(Error::Model("gravity container magic mismatch".into()));
    }
    let header_len = u32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]) as usize;
    let body_offset = 12usize
        .checked_add(header_len)
        .ok_or_else(|| Error::Model("gravity container header length overflows".into()))?;
    if body_offset > payload.len() {
        return Err(Error::Model("gravity container header exceeds payload".into()));
    }
    Ok((&payload[12..body_offset], &payload[body_offset..]))
}

fn pack_bits_lsb(bits: impl IntoIterator<Item = bool>) -> Vec<u8> {
    let mut out = Vec::new();
    let mut acc = 0u8;
    let mut filled = 0u8;
    for bit in bits {
        if bit {
            acc |= 1u8 << filled;
        }
        filled += 1;
        if filled == 8 {
            out.push(acc);
            acc = 0;
            filled = 0;
        }
    }
    if filled > 0 {
        out.push(acc);
    }
    out
}

fn pack_unsigned(codes: &[u8], bits: u8) -> Result<Vec<u8>> {
    if bits == 0 || bits > 8 {
        return Err(Error::Model("unsigned pack bits must be 1..=8".into()));
    }
    let mut bit_iter = Vec::with_capacity(codes.len() * usize::from(bits));
    for &code in codes {
        for bit in 0..bits {
            bit_iter.push(((code >> bit) & 1) != 0);
        }
    }
    Ok(pack_bits_lsb(bit_iter))
}

/// Binary sign + stored group scale. Scale is mean-abs of the group, stored
/// as fp16 — same as `lab.operators.ascension_dual_gravity_worker._binary_codec`.
pub fn pack_binary_group(
    weights: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> Result<BinaryGroupPacked> {
    if rows == 0 || cols == 0 || group_size == 0 {
        return Err(Error::Model("binary_group requires positive geometry".into()));
    }
    if weights.len() != rows * cols {
        return Err(Error::Model("binary_group weight length disagrees with shape".into()));
    }
    if cols % group_size != 0 {
        return Err(Error::Model(
            "binary_group kernel requires cols to be a multiple of group_size".into(),
        ));
    }
    if weights.iter().any(|v| !v.is_finite()) {
        return Err(Error::Model("binary_group refuses non-finite weights".into()));
    }
    let groups_per_row = cols / group_size;
    let groups = rows * groups_per_row;
    let mut scales_f16 = Vec::with_capacity(groups);
    let mut sign_bits = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for group in 0..groups_per_row {
            let start = row * cols + group * group_size;
            let slice = &weights[start..start + group_size];
            let mut sum_abs = 0.0f64;
            for &value in slice {
                sum_abs += f64::from(value.abs());
                sign_bits.push(value >= 0.0);
            }
            let mean = sum_abs / group_size as f64;
            scales_f16.push(f16::from_f32(mean as f32).to_bits());
        }
    }
    Ok(BinaryGroupPacked {
        rows,
        cols,
        group_size,
        groups_per_row,
        scales_f16,
        signs: pack_bits_lsb(sign_bits),
    })
}

pub fn binary_group_weight(packed: &BinaryGroupPacked, row: usize, col: usize) -> f32 {
    let flat = row * packed.cols + col;
    let scale = f16::from_bits(packed.scales_f16[row * packed.groups_per_row + col / packed.group_size])
        .to_f32();
    let positive = ((packed.signs[flat >> 3] >> (flat & 7)) & 1) != 0;
    if positive {
        scale
    } else {
        -scale
    }
}

/// Serial left-to-right f32 matvec of the packed binary_group artifact.
pub fn binary_group_matvec_f32(packed: &BinaryGroupPacked, input: &[f32]) -> Result<Vec<f32>> {
    if input.len() != packed.cols || input.iter().any(|v| !v.is_finite()) {
        return Err(Error::Model("binary_group matvec input is not finite cols".into()));
    }
    let mut output = vec![0.0f32; packed.rows];
    for row in 0..packed.rows {
        let mut sum = 0.0f32;
        for col in 0..packed.cols {
            sum += binary_group_weight(packed, row, col) * input[col];
        }
        output[row] = sum;
    }
    Ok(output)
}

struct BitWriter {
    buf: Vec<u8>,
    acc: u8,
    filled: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            acc: 0,
            filled: 0,
        }
    }

    fn write_bit(&mut self, bit: u8) {
        self.acc |= (bit & 1) << self.filled;
        self.filled += 1;
        if self.filled == 8 {
            self.buf.push(self.acc);
            self.acc = 0;
            self.filled = 0;
        }
    }

    fn write_ones(&mut self, mut count: u32) {
        while count > 0 {
            let room = 8 - self.filled;
            let take = count.min(u32::from(room)) as u8;
            let mask = if take == 8 { 0xffu8 } else { (1u8 << take) - 1 };
            self.acc |= mask << self.filled;
            self.filled += take;
            count -= u32::from(take);
            if self.filled == 8 {
                self.buf.push(self.acc);
                self.acc = 0;
                self.filled = 0;
            }
        }
    }

    fn write_lsbs(&mut self, value: u32, bits: u32) {
        for i in 0..bits {
            self.write_bit(((value >> i) & 1) as u8);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.filled > 0 {
            self.buf.push(self.acc);
        }
        self.buf
    }
}

fn best_rice_k(values: &[u32]) -> u32 {
    if values.is_empty() {
        return 0;
    }
    let n = values.len() as u64;
    let mut best_k = 0u32;
    let mut best_bits = u64::MAX;
    for k in 0..16u32 {
        let mut q_sum = 0u64;
        for &value in values {
            q_sum += u64::from(value >> k);
        }
        let bits = q_sum + n * (1 + u64::from(k));
        if bits < best_bits {
            best_k = k;
            best_bits = bits;
        }
    }
    best_k
}

fn pack_rice(values: &[u32], k: u32) -> Vec<u8> {
    let mut writer = BitWriter::new();
    let mask = if k == 0 { 0 } else { (1u32 << k) - 1 };
    for &value in values {
        writer.write_ones(value >> k);
        writer.write_bit(0);
        if k > 0 {
            writer.write_lsbs(value & mask, k);
        }
    }
    writer.finish()
}

pub fn unpack_rice(payload: &[u8], count: usize, k: u32) -> Result<Vec<u32>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut byte = 0usize;
    let mut bit = 0u8;
    let mut read_bit = || -> Result<u32> {
        if byte >= payload.len() {
            return Err(Error::Model("rice stream overran its payload".into()));
        }
        let value = u32::from((payload[byte] >> bit) & 1);
        bit += 1;
        if bit == 8 {
            bit = 0;
            byte += 1;
        }
        Ok(value)
    };
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut q = 0u32;
        while read_bit()? == 1 {
            q = q
                .checked_add(1)
                .ok_or_else(|| Error::Model("rice quotient overflow".into()))?;
        }
        let mut rem = 0u32;
        for i in 0..k {
            rem |= read_bit()? << i;
        }
        out.push((q << k) | rem);
    }
    Ok(out)
}

/// Binary base + global top-2% |residual| stored as rice indices + 1-bit
/// (sign × RMS of the selected residuals). Matches
/// `lab.operators.residual_compact_codec.encode_residual_compact`
/// (`index_mode=rice`, `value_bits=1`, `value_scale=rms`).
pub fn pack_binary_rice_q1(
    weights: &[f32],
    rows: usize,
    cols: usize,
    outlier_ratio: f64,
) -> Result<RiceQ1Packed> {
    if !(0.0 < outlier_ratio && outlier_ratio <= 0.1) {
        return Err(Error::Model("rice_q1 outlier ratio must be in (0, 0.1]".into()));
    }
    let binary = pack_binary_group(weights, rows, cols, Q80_BINARY_GROUP_SIZE)?;
    let n = rows * cols;
    let mut residual = vec![0.0f32; n];
    for row in 0..rows {
        for col in 0..cols {
            let flat = row * cols + col;
            residual[flat] = weights[flat] - binary_group_weight(&binary, row, col);
        }
    }
    let count = ((n as f64) * outlier_ratio).ceil() as usize;
    let count = count.max(1).min(n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        residual[b]
            .abs()
            .partial_cmp(&residual[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut indices: Vec<u32> = order[..count].iter().map(|&i| i as u32).collect();
    indices.sort_unstable();
    let selected: Vec<f32> = indices.iter().map(|&i| residual[i as usize]).collect();
    let rms = if selected.is_empty() {
        0.0f32
    } else {
        let mean_sq = selected
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            / selected.len() as f64;
        mean_sq.sqrt() as f32
    };
    let mut scale = rms;
    if !scale.is_finite() || scale <= 0.0 {
        scale = 1.0;
    }
    let residual_scale_f16 = f16::from_f32(scale).to_bits();
    let residual_signs = pack_bits_lsb(selected.iter().map(|v| *v >= 0.0));
    let first_index = indices[0];
    let (rice_k, rice_bytes) = if indices.len() == 1 {
        (0u32, Vec::new())
    } else {
        let mut diffs = Vec::with_capacity(indices.len() - 1);
        for pair in indices.windows(2) {
            if pair[1] <= pair[0] {
                return Err(Error::Model("rice residual requires strictly increasing indices".into()));
            }
            diffs.push(pair[1] - pair[0]);
        }
        let k = best_rice_k(&diffs);
        (k, pack_rice(&diffs, k))
    };
    Ok(RiceQ1Packed {
        binary,
        first_index,
        rice_k,
        rice_bytes,
        outlier_count: count,
        residual_scale_f16,
        residual_signs,
        indices,
    })
}

/// CSR row pointers over already-sorted flat indices. Bind-time only.
pub fn rice_q1_row_ptr(indices: &[u32], rows: usize, cols: usize) -> Result<Vec<u32>> {
    if cols == 0 {
        return Err(Error::Model("rice CSR requires positive cols".into()));
    }
    let mut row_ptr = vec![0u32; rows + 1];
    for &flat in indices {
        let row = (flat as usize) / cols;
        if row >= rows {
            return Err(Error::Model("rice CSR index row out of range".into()));
        }
        row_ptr[row + 1] += 1;
    }
    for i in 0..rows {
        row_ptr[i + 1] = row_ptr[i + 1]
            .checked_add(row_ptr[i])
            .ok_or_else(|| Error::Model("rice CSR row pointer overflow".into()))?;
    }
    if row_ptr[rows] as usize != indices.len() {
        return Err(Error::Model("rice CSR count disagrees with outlier count".into()));
    }
    Ok(row_ptr)
}

pub fn expand_rice_indices(packed: &RiceQ1Packed) -> Result<Vec<u32>> {
    if packed.outlier_count == 0 {
        return Err(Error::Model("rice residual requires at least one outlier".into()));
    }
    if packed.outlier_count == 1 {
        return Ok(vec![packed.first_index]);
    }
    let diffs = unpack_rice(&packed.rice_bytes, packed.outlier_count - 1, packed.rice_k)?;
    let mut indices = Vec::with_capacity(packed.outlier_count);
    let mut acc = u64::from(packed.first_index);
    indices.push(packed.first_index);
    for diff in diffs {
        acc = acc
            .checked_add(u64::from(diff))
            .ok_or_else(|| Error::Model("rice index overflow".into()))?;
        if acc > u64::from(u32::MAX) {
            return Err(Error::Model("rice index exceeds u32".into()));
        }
        indices.push(acc as u32);
    }
    Ok(indices)
}

/// `y = binary_group(x) + rice_q1 residual corrections`, serial in index order.
pub fn binary_rice_q1_matvec_f32(packed: &RiceQ1Packed, input: &[f32]) -> Result<Vec<f32>> {
    let mut output = binary_group_matvec_f32(&packed.binary, input)?;
    let indices = expand_rice_indices(packed)?;
    if indices.len() != packed.outlier_count {
        return Err(Error::Model("rice expand produced a different outlier count".into()));
    }
    let cols = packed.binary.cols;
    let scale = f16::from_bits(packed.residual_scale_f16).to_f32();
    for (n, &flat) in indices.iter().enumerate() {
        let flat = flat as usize;
        if flat / cols >= packed.binary.rows || flat % cols >= cols {
            return Err(Error::Model("rice residual index out of range".into()));
        }
        let positive = ((packed.residual_signs[n >> 3] >> (n & 7)) & 1) != 0;
        let value = if positive { scale } else { -scale };
        output[flat / cols] += value * input[flat % cols];
    }
    Ok(output)
}

pub fn pack_uniform_factor(
    values: &[f32],
    rows: usize,
    cols: usize,
    bits: u8,
    group_size: usize,
) -> Result<UniformFactorPacked> {
    if bits < 2 || bits > 8 || group_size == 0 || rows == 0 || cols == 0 {
        return Err(Error::Model("uniform factor geometry is invalid".into()));
    }
    if values.len() != rows * cols {
        return Err(Error::Model("uniform factor length disagrees with shape".into()));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error::Model("uniform factor refuses non-finite values".into()));
    }
    let elements = rows * cols;
    let groups = elements.div_ceil(group_size);
    let bound = (1u16 << (bits - 1)) - 1;
    let mut scales_f16 = Vec::with_capacity(groups);
    let mut codes = Vec::with_capacity(groups * group_size);
    for group in 0..groups {
        let start = group * group_size;
        let end = (start + group_size).min(elements);
        let max_abs = values[start..end]
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        let scale = max_abs / f32::from(bound.max(1));
        scales_f16.push(f16::from_f32(scale).to_bits());
        let denom = if scale > 0.0 { scale } else { 1.0 };
        for index in 0..group_size {
            let value = values.get(start + index).copied().unwrap_or(0.0);
            let signed = (value / denom).round().clamp(-(bound as f32), bound as f32) as i16;
            codes.push((signed + bound as i16) as u8);
        }
    }
    let packed_codes = pack_unsigned(&codes, bits)?;
    Ok(UniformFactorPacked {
        rows,
        cols,
        bits,
        group_size,
        groups,
        bound,
        scales_f16,
        codes: packed_codes,
    })
}

pub(super) fn extract_unsigned(codes: &[u8], element: usize, bits: u8) -> u16 {
    let bit0 = element * usize::from(bits);
    let mut value = 0u16;
    for b in 0..usize::from(bits) {
        let bit_index = bit0 + b;
        let byte = codes[bit_index >> 3];
        let bit = u16::from((byte >> (bit_index & 7)) & 1);
        value |= bit << b;
    }
    value
}

pub fn uniform_factor_value(packed: &UniformFactorPacked, row: usize, col: usize) -> f32 {
    let element = row * packed.cols + col;
    let group = element / packed.group_size;
    let scale = f16::from_bits(packed.scales_f16[group]).to_f32();
    let code = extract_unsigned(&packed.codes, element, packed.bits);
    let signed = i32::from(code) - i32::from(packed.bound);
    signed as f32 * scale
}

pub fn uniform_factor_matvec_f32(packed: &UniformFactorPacked, input: &[f32]) -> Result<Vec<f32>> {
    if input.len() != packed.cols || input.iter().any(|v| !v.is_finite()) {
        return Err(Error::Model("uniform factor matvec input is not finite cols".into()));
    }
    let mut output = vec![0.0f32; packed.rows];
    for row in 0..packed.rows {
        let mut sum = 0.0f32;
        for col in 0..packed.cols {
            sum += uniform_factor_value(packed, row, col) * input[col];
        }
        output[row] = sum;
    }
    Ok(output)
}

/// Native two-stage `y = L @ (R @ x)` of packed factors. Never forms dense W.
pub fn hgravs01_two_stage_matvec_f32(
    left: &UniformFactorPacked,
    right: &UniformFactorPacked,
    input: &[f32],
) -> Result<Vec<f32>> {
    if left.cols != right.rows {
        return Err(Error::Model("hgravs01 factor rank disagrees".into()));
    }
    if right.cols != input.len() {
        return Err(Error::Model("hgravs01 right factor cols disagree with input".into()));
    }
    let mid = uniform_factor_matvec_f32(right, input)?;
    uniform_factor_matvec_f32(left, &mid)
}

pub fn deterministic_matrix(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * cols);
    for i in 0..rows * cols {
        let phase = ((i as u32).wrapping_mul(1103515245).wrapping_add(seed)) as f32;
        out.push(((phase % 251.0) - 125.0) / 251.0);
    }
    out
}

pub fn deterministic_input(cols: usize) -> Vec<f32> {
    (0..cols)
        .map(|index| {
            let phase = (index % 17) as f32;
            (phase * 0.07 - 0.5).sin()
        })
        .collect()
}

pub fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_group_roundtrip_signs_and_scales() {
        let rows = 4;
        let cols = 256;
        let w = deterministic_matrix(rows, cols, 7);
        let packed = pack_binary_group(&w, rows, cols, Q80_BINARY_GROUP_SIZE).unwrap();
        assert_eq!(packed.groups_per_row, 2);
        assert_eq!(packed.signs.len(), (rows * cols).div_ceil(8));
        let x = deterministic_input(cols);
        let y = binary_group_matvec_f32(&packed, &x).unwrap();
        assert_eq!(y.len(), rows);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rice_q1_indices_roundtrip() {
        let rows = 8;
        let cols = 128;
        let w = deterministic_matrix(rows, cols, 11);
        let packed = pack_binary_rice_q1(&w, rows, cols, 0.02).unwrap();
        let expanded = expand_rice_indices(&packed).unwrap();
        assert_eq!(expanded, packed.indices);
        assert!(packed.outlier_count >= 1);
        let x = deterministic_input(cols);
        let y = binary_rice_q1_matvec_f32(&packed, &x).unwrap();
        assert_eq!(y.len(), rows);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn hgravs_two_stage_does_not_need_dense_w() {
        let rows = 16;
        let cols = 32;
        let rank = 8;
        let left_vals = deterministic_matrix(rows, rank, 3);
        let right_vals = deterministic_matrix(rank, cols, 5);
        let left = pack_uniform_factor(&left_vals, rows, rank, 3, 64).unwrap();
        let right = pack_uniform_factor(&right_vals, rank, cols, 3, 64).unwrap();
        let x = deterministic_input(cols);
        let y = hgravs01_two_stage_matvec_f32(&left, &right, &x).unwrap();
        assert_eq!(y.len(), rows);
        assert!(y.iter().all(|v| v.is_finite()));
        // Token-path temporary is mid[rank], never dense W[rows*cols].
        assert_eq!(rank * 4, 32);
        assert!(rows * cols * 4 > rank * 4);
    }

    #[test]
    fn rice_q1_fused_correction_does_not_form_dense_w() {
        let rows = 8;
        let cols = 128;
        let w = deterministic_matrix(rows, cols, 11);
        let packed = pack_binary_rice_q1(&w, rows, cols, 0.02).unwrap();
        let x = deterministic_input(cols);
        let y = binary_rice_q1_matvec_f32(&packed, &x).unwrap();
        let mut rebuilt = binary_group_matvec_f32(&packed.binary, &x).unwrap();
        let scale = f16::from_bits(packed.residual_scale_f16).to_f32();
        for (n, &flat) in packed.indices.iter().enumerate() {
            let positive = ((packed.residual_signs[n >> 3] >> (n & 7)) & 1) != 0;
            let value = if positive { scale } else { -scale };
            rebuilt[(flat as usize) / cols] += value * x[(flat as usize) % cols];
        }
        assert_eq!(max_abs_error(&y, &rebuilt), 0.0);
    }
}

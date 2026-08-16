//! Switching-activity analysis for Q80 mixed packed streams.
//!
//! Dynamic CMOS energy of a packed-weight fetch is, to first order, proportional
//! to the Hamming distance between consecutively transmitted words — not the
//! number of words. For a codebook codec the assignment of binary codes to
//! centroids is a free variable: permute the codes and the lookup table
//! together and decoded values are bit-identical, BPW is unchanged, and the
//! bit patterns on the bus change.
//!
//! Fetch order is the order the Q80 mixed kernels address packed bytes
//! (`shaders/q80_mixed_decode.metal`), not the on-disk JSON envelope.

use super::q80_mixed_decode::extract_unsigned;
use super::{
    pack_unsigned_lsb, pack_unsigned_msb, packed_byte_count, unpack_unsigned_lsb,
    BinaryGroupPacked, RiceQ1Packed, UniformFactorPacked, MAGIC_HGRAVS01,
};
use crate::{Error, Result};
use serde_json::{Map, Value};

pub const RANDOM_ALPHA: f64 = 0.5;

#[derive(Clone, Debug, PartialEq)]
pub struct WordActivity {
    pub word_bits: u32,
    pub words: u64,
    pub transitions: u64,
    pub mean_hamming: f64,
    pub transitions_per_byte: f64,
    pub alpha: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CodeActivity {
    pub bits: u8,
    pub codes: u64,
    pub transitions: u64,
    pub mean_hamming: f64,
    pub alpha: f64,
    pub histogram: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Assignment {
    /// `symbol_to_code[old_code] = new_code`.
    pub symbol_to_code: Vec<u8>,
    pub bits: u8,
    pub method: &'static str,
    pub cost: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RemappedFactor {
    pub packed: UniformFactorPacked,
    /// `lut[new_code] = signed q` such that `q * scale` is the original value.
    pub lut: Vec<i32>,
    pub assignment: Assignment,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Hgravs01CodeView {
    pub rank: usize,
    pub bits: u8,
    pub group_size: usize,
    pub matrix_shape: [usize; 2],
    pub left_shape: [usize; 2],
    pub right_shape: [usize; 2],
    pub left_elements: usize,
    pub right_elements: usize,
    pub left_codes: Vec<u8>,
    pub right_codes: Vec<u8>,
    pub left_code_bytes: Vec<u8>,
    pub right_code_bytes: Vec<u8>,
    pub left_scale_bytes: Vec<u8>,
    pub right_scale_bytes: Vec<u8>,
}

pub fn hamming_u32(a: u32, b: u32) -> u32 {
    (a ^ b).count_ones()
}

pub fn activity_from_transitions(word_bits: u32, words: u64, transitions: u64) -> WordActivity {
    let pairs = words.saturating_sub(1);
    let mean = if pairs == 0 {
        0.0
    } else {
        transitions as f64 / pairs as f64
    };
    let bytes = (words as f64) * (f64::from(word_bits) / 8.0);
    let transitions_per_byte = if bytes == 0.0 {
        0.0
    } else {
        transitions as f64 / bytes
    };
    let alpha = if pairs == 0 || word_bits == 0 {
        0.0
    } else {
        mean / f64::from(word_bits)
    };
    WordActivity {
        word_bits,
        words,
        transitions,
        mean_hamming: mean,
        transitions_per_byte,
        alpha,
    }
}

/// Consecutive-word Hamming of a byte stream, little-endian `word_bits`-wide
/// words (8, 32, or 128). Partial tail words are zero-padded.
pub fn packed_word_activity(bytes: &[u8], word_bits: u32) -> WordActivity {
    assert!(
        word_bits == 8 || word_bits == 32 || word_bits == 128,
        "word_bits must be 8, 32, or 128"
    );
    let width = (word_bits / 8) as usize;
    if bytes.is_empty() {
        return activity_from_transitions(word_bits, 0, 0);
    }
    let words = bytes.len().div_ceil(width);
    let mut transitions = 0u64;
    let mut prev = read_le_word(bytes, 0, width);
    for index in 1..words {
        let word = read_le_word(bytes, index * width, width);
        transitions += u64::from(hamming_word(prev, word, word_bits));
        prev = word;
    }
    activity_from_transitions(word_bits, words as u64, transitions)
}

fn read_le_word(bytes: &[u8], offset: usize, width: usize) -> u128 {
    let mut word = 0u128;
    for (i, byte) in bytes.iter().skip(offset).take(width).enumerate() {
        word |= u128::from(*byte) << (8 * i);
    }
    word
}

fn hamming_word(a: u128, b: u128, word_bits: u32) -> u32 {
    let xor = a ^ b;
    if word_bits <= 64 {
        let mask = if word_bits == 64 {
            u64::MAX
        } else {
            (1u64 << word_bits) - 1
        };
        ((xor as u64) & mask).count_ones()
    } else {
        let lo = xor as u64;
        let hi = (xor >> 64) as u64;
        let hi_bits = word_bits - 64;
        let hi_mask = if hi_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << hi_bits) - 1
        };
        lo.count_ones() + (hi & hi_mask).count_ones()
    }
}

/// Kernel extract order for a row-major factor: element `row * cols + col`.
/// The Q80 serial and simd factor kernels both walk columns left-to-right
/// inside a row; simdgroups only change which rows are concurrent.
pub fn unpacked_code_activity(codes: &[u8], bits: u8) -> CodeActivity {
    let alphabet = 1usize << bits;
    let mut histogram = vec![0u64; alphabet];
    for &code in codes {
        if (code as usize) < alphabet {
            histogram[code as usize] += 1;
        }
    }
    let mut transitions = 0u64;
    if codes.len() >= 2 {
        for pair in codes.windows(2) {
            transitions += u64::from(hamming_u32(u32::from(pair[0]), u32::from(pair[1])));
        }
    }
    let pairs = codes.len().saturating_sub(1) as u64;
    let mean = if pairs == 0 {
        0.0
    } else {
        transitions as f64 / pairs as f64
    };
    let alpha = if bits == 0 || pairs == 0 {
        0.0
    } else {
        mean / f64::from(bits)
    };
    CodeActivity {
        bits,
        codes: codes.len() as u64,
        transitions,
        mean_hamming: mean,
        alpha,
        histogram,
    }
}

pub fn transition_matrix(codes: &[u8], bits: u8) -> Vec<Vec<u64>> {
    let n = 1usize << bits;
    let mut matrix = vec![vec![0u64; n]; n];
    for pair in codes.windows(2) {
        let a = pair[0] as usize;
        let b = pair[1] as usize;
        if a < n && b < n {
            matrix[a][b] += 1;
        }
    }
    matrix
}

pub fn assignment_cost(matrix: &[Vec<u64>], symbol_to_code: &[u8]) -> u64 {
    let n = matrix.len();
    let mut cost = 0u64;
    for i in 0..n {
        for j in 0..n {
            let count = matrix[i][j];
            if count == 0 {
                continue;
            }
            cost += count
                * u64::from(hamming_u32(
                    u32::from(symbol_to_code[i]),
                    u32::from(symbol_to_code[j]),
                ));
        }
    }
    cost
}

pub fn identity_assignment(bits: u8) -> Assignment {
    let n = 1u8 << bits;
    let symbol_to_code: Vec<u8> = (0..n).collect();
    Assignment {
        symbol_to_code,
        bits,
        method: "identity",
        cost: 0,
    }
}

pub fn gray_assignment(bits: u8) -> Assignment {
    let n = 1u8 << bits;
    let symbol_to_code: Vec<u8> = (0..n).map(binary_to_gray).collect();
    Assignment {
        symbol_to_code,
        bits,
        method: "reflected_gray",
        cost: 0,
    }
}

/// Map offset-binary (`q = code - bound`) onto `bits`-bit two's complement.
pub fn twos_complement_assignment(bits: u8) -> Assignment {
    let n = 1i32 << bits;
    let bound = (1i32 << (bits - 1)) - 1;
    let mut symbol_to_code = vec![0u8; n as usize];
    for old in 0..n {
        let q = old - bound;
        let coded = if q >= 0 { q } else { n + q };
        symbol_to_code[old as usize] = coded as u8;
    }
    Assignment {
        symbol_to_code,
        bits,
        method: "twos_complement",
        cost: 0,
    }
}

/// Sign-magnitude of the signed level, 1 sign bit + (bits-1) magnitude.
pub fn sign_magnitude_assignment(bits: u8) -> Assignment {
    let n = 1i32 << bits;
    let bound = (1i32 << (bits - 1)) - 1;
    let mut symbol_to_code = vec![0u8; n as usize];
    for old in 0..n {
        let q = old - bound;
        let mag = q.unsigned_abs() as u8;
        let sign = if q < 0 { 1u8 << (bits - 1) } else { 0 };
        symbol_to_code[old as usize] = sign | mag;
    }
    Assignment {
        symbol_to_code,
        bits,
        method: "sign_magnitude",
        cost: 0,
    }
}

pub fn binary_to_gray(value: u8) -> u8 {
    value ^ (value >> 1)
}

pub fn gray_to_binary(mut value: u8) -> u8 {
    let mut mask = value >> 1;
    while mask != 0 {
        value ^= mask;
        mask >>= 1;
    }
    value
}

/// Exact minimum-Hamming assignment for alphabets of size <= 8 (8! = 40320).
/// Larger alphabets use greedy construction plus deterministic annealing.
pub fn minimize_assignment(matrix: &[Vec<u64>], bits: u8) -> Assignment {
    let n = 1usize << bits;
    assert_eq!(matrix.len(), n);
    let mut candidates = vec![
        with_cost(identity_assignment(bits), matrix),
        with_cost(gray_assignment(bits), matrix),
        with_cost(twos_complement_assignment(bits), matrix),
        with_cost(sign_magnitude_assignment(bits), matrix),
        with_cost(greedy_assignment(matrix, bits), matrix),
    ];
    if bits <= 3 {
        candidates.push(exact_assignment(matrix, bits));
    } else {
        let seed = with_cost(greedy_assignment(matrix, bits), matrix);
        candidates.push(anneal_assignment(matrix, bits, seed.symbol_to_code, 48_000));
        candidates.push(anneal_assignment(
            matrix,
            bits,
            gray_assignment(bits).symbol_to_code,
            24_000,
        ));
    }
    candidates
        .into_iter()
        .min_by(|a, b| {
            a.cost
                .cmp(&b.cost)
                .then_with(|| method_preference(a.method).cmp(&method_preference(b.method)))
        })
        .expect("at least identity")
}

fn method_preference(method: &str) -> u8 {
    match method {
        "exact_brute_force" => 0,
        "annealed" => 1,
        "greedy" => 2,
        _ => 3,
    }
}

fn with_cost(mut assignment: Assignment, matrix: &[Vec<u64>]) -> Assignment {
    assignment.cost = assignment_cost(matrix, &assignment.symbol_to_code);
    assignment
}

fn exact_assignment(matrix: &[Vec<u64>], bits: u8) -> Assignment {
    let n = 1usize << bits;
    let mut perm: Vec<u8> = (0..n as u8).collect();
    let mut best = perm.clone();
    let mut best_cost = assignment_cost(matrix, &perm);
    heap_permute(&mut perm, n, matrix, &mut best, &mut best_cost);
    Assignment {
        symbol_to_code: best,
        bits,
        method: "exact_brute_force",
        cost: best_cost,
    }
}

fn heap_permute(
    perm: &mut [u8],
    k: usize,
    matrix: &[Vec<u64>],
    best: &mut [u8],
    best_cost: &mut u64,
) {
    if k == 1 {
        let cost = assignment_cost(matrix, perm);
        if cost < *best_cost {
            *best_cost = cost;
            best.copy_from_slice(perm);
        }
        return;
    }
    heap_permute(perm, k - 1, matrix, best, best_cost);
    for i in 0..k - 1 {
        if k % 2 == 0 {
            perm.swap(i, k - 1);
        } else {
            perm.swap(0, k - 1);
        }
        heap_permute(perm, k - 1, matrix, best, best_cost);
    }
}

fn greedy_assignment(matrix: &[Vec<u64>], bits: u8) -> Assignment {
    let n = 1usize << bits;
    let mut degree = vec![0u64; n];
    for i in 0..n {
        for j in 0..n {
            degree[i] += matrix[i][j] + matrix[j][i];
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| degree[*b].cmp(&degree[*a]).then(a.cmp(b)));
    let mut symbol_to_code = vec![0u8; n];
    let mut used = vec![false; n];
    let mut assigned = vec![false; n];
    if let Some(&first) = order.first() {
        symbol_to_code[first] = 0;
        used[0] = true;
        assigned[first] = true;
    }
    for &symbol in &order[1..] {
        let mut best_code = 0u8;
        let mut best = u64::MAX;
        for code in 0..n {
            if used[code] {
                continue;
            }
            let mut cost = 0u64;
            for other in 0..n {
                if !assigned[other] {
                    continue;
                }
                let ham = u64::from(hamming_u32(code as u32, u32::from(symbol_to_code[other])));
                cost += matrix[symbol][other] * ham;
                cost += matrix[other][symbol] * ham;
            }
            if cost < best {
                best = cost;
                best_code = code as u8;
            }
        }
        symbol_to_code[symbol] = best_code;
        used[best_code as usize] = true;
        assigned[symbol] = true;
    }
    Assignment {
        symbol_to_code,
        bits,
        method: "greedy",
        cost: 0,
    }
}

fn anneal_assignment(matrix: &[Vec<u64>], bits: u8, start: Vec<u8>, steps: u32) -> Assignment {
    let n = start.len();
    let mut perm = start;
    let mut cost = assignment_cost(matrix, &perm);
    let mut best = perm.clone();
    let mut best_cost = cost;
    let mut rng = Lcg(0xC0DEC0DE_u64.wrapping_add(u64::from(bits) << 32));
    for step in 0..steps {
        let i = (rng.next() as usize) % n;
        let j = (rng.next() as usize) % n;
        if i == j {
            continue;
        }
        perm.swap(i, j);
        let next = assignment_cost(matrix, &perm);
        let temp = 1.0 - (f64::from(step) / f64::from(steps.max(1)));
        let accept = next <= cost || {
            let delta = (next - cost) as f64;
            let threshold = ((rng.next() >> 11) as f64) / ((1u64 << 53) as f64);
            (-delta / (temp * (best_cost.max(1) as f64 / 8.0 + 1.0))).exp() > threshold
        };
        if accept {
            cost = next;
            if cost < best_cost {
                best_cost = cost;
                best.copy_from_slice(&perm);
            }
        } else {
            perm.swap(i, j);
        }
    }
    Assignment {
        symbol_to_code: best,
        bits,
        method: "annealed",
        cost: best_cost,
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
}

pub fn apply_assignment(codes: &[u8], symbol_to_code: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(codes.len());
    for &code in codes {
        let mapped = *symbol_to_code
            .get(code as usize)
            .ok_or_else(|| Error::Model("code is outside the assignment domain".into()))?;
        out.push(mapped);
    }
    Ok(out)
}

/// Build the signed-q lookup that makes remapped codes decode to the original
/// values under `value = lut[code] * scale`.
pub fn lut_from_assignment(assignment: &Assignment, bound: i32) -> Vec<i32> {
    let n = 1usize << assignment.bits;
    let mut lut = vec![0i32; n];
    for (old, &new_code) in assignment.symbol_to_code.iter().enumerate() {
        lut[new_code as usize] = old as i32 - bound;
    }
    lut
}

pub fn decode_codes_offset(codes: &[u8], bound: i32) -> Vec<i32> {
    codes.iter().map(|&c| i32::from(c) - bound).collect()
}

pub fn decode_codes_lut(codes: &[u8], lut: &[i32]) -> Result<Vec<i32>> {
    let mut out = Vec::with_capacity(codes.len());
    for &code in codes {
        let q = *lut
            .get(code as usize)
            .ok_or_else(|| Error::Model("code is outside the decode LUT".into()))?;
        out.push(q);
    }
    Ok(out)
}

pub fn remap_uniform_factor(
    packed: &UniformFactorPacked,
    assignment: &Assignment,
) -> Result<RemappedFactor> {
    if assignment.bits != packed.bits {
        return Err(Error::Model(
            "assignment bit width disagrees with the packed factor".into(),
        ));
    }
    let count = packed.groups * packed.group_size;
    let codes = unpack_unsigned_lsb(&packed.codes, count, packed.bits)?;
    let remapped = apply_assignment(&codes, &assignment.symbol_to_code)?;
    let new_bytes = pack_unsigned_lsb(&remapped, packed.bits)?;
    if new_bytes.len() != packed.codes.len() {
        return Err(Error::Model(
            "remapped pack changed the packed byte count (BPW must be invariant)".into(),
        ));
    }
    let lut = lut_from_assignment(assignment, i32::from(packed.bound));
    let original_q = decode_codes_offset(&codes, i32::from(packed.bound));
    let remapped_q = decode_codes_lut(&remapped, &lut)?;
    if original_q != remapped_q {
        return Err(Error::Model(
            "remapped codes + LUT are not bit-identical to offset-binary decode".into(),
        ));
    }
    let mut new_packed = packed.clone();
    new_packed.codes = new_bytes;
    Ok(RemappedFactor {
        packed: new_packed,
        lut,
        assignment: assignment.clone(),
    })
}

/// Decode every stored element of a uniform factor (including group padding)
/// with either offset-binary or an explicit LUT. Used as the bit-identity gate.
pub fn uniform_factor_signed_levels(
    packed: &UniformFactorPacked,
    lut: Option<&[i32]>,
) -> Result<Vec<i32>> {
    let count = packed.groups * packed.group_size;
    let codes = unpack_unsigned_lsb(&packed.codes, count, packed.bits)?;
    match lut {
        None => Ok(decode_codes_offset(&codes, i32::from(packed.bound))),
        Some(table) => decode_codes_lut(&codes, table),
    }
}

pub fn binary_sign_bits(packed: &BinaryGroupPacked) -> Vec<u8> {
    let n = packed.rows * packed.cols;
    let mut bits = Vec::with_capacity(n);
    for flat in 0..n {
        bits.push((packed.signs[flat >> 3] >> (flat & 7)) & 1);
    }
    bits
}

pub fn invert_sign_bytes(signs: &[u8], bit_count: usize) -> Vec<u8> {
    let mut out = signs.to_vec();
    for (index, byte) in out.iter_mut().enumerate() {
        let used = bit_count.saturating_sub(index * 8).min(8);
        if used == 8 {
            *byte = !*byte;
        } else if used > 0 {
            let mask = (1u8 << used) - 1;
            *byte ^= mask;
        }
    }
    out
}

/// Simd/two-stage tile fetch of packed factor bytes.
///
/// One simdgroup owns one row and walks columns in 32-wide tiles. A threadgroup
/// of 8 simdgroups therefore issues, for each tile `t`, eight `tile_bytes`-wide
/// reads at row stride `row_bytes`. That is the address order the memory
/// controller sees, not a single-row serial walk.
pub fn kernel_simd_tile_bytes(
    packed_codes: &[u8],
    rows: usize,
    cols: usize,
    bits: u8,
    simd_width: usize,
    rows_per_tg: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(packed_codes.len());
    if rows == 0 || cols == 0 || bits == 0 {
        return out;
    }
    let bits_usize = usize::from(bits);
    let tiles = cols.div_ceil(simd_width);
    let row_bits = cols * bits_usize;
    let tile_bits = simd_width * bits_usize;
    for row0 in (0..rows).step_by(rows_per_tg) {
        let row_end = (row0 + rows_per_tg).min(rows);
        for tile in 0..tiles {
            for row in row0..row_end {
                let bit0 = row * row_bits + tile * tile_bits;
                let remaining_cols = cols - tile * simd_width;
                let this_bits = remaining_cols.min(simd_width) * bits_usize;
                let byte0 = bit0 / 8;
                let byte1 = (bit0 + this_bits).div_ceil(8).min(packed_codes.len());
                if byte0 < packed_codes.len() {
                    out.extend_from_slice(&packed_codes[byte0..byte1]);
                }
            }
        }
    }
    out
}

pub fn binary_kernel_sign_bytes(packed: &BinaryGroupPacked) -> Vec<u8> {
    // Serial after alignment, simd_bytes, and tg256 all consume one sign byte
    // per 8 consecutive columns of a row. That is storage order of `signs`.
    packed.signs.clone()
}

pub fn rice_kernel_bytes(packed: &RiceQ1Packed) -> Vec<u8> {
    packed.rice_bytes.clone()
}

pub fn scale_bytes_from_f16(scales: &[u16]) -> Vec<u8> {
    scales.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Peek an on-disk HGRAVS01 container and return unpacked factor codes in
/// kernel extract order (flattened row-major, first `elements` codes only).
pub fn peek_hgravs01_codes(payload: &[u8]) -> Result<Hgravs01CodeView> {
    if payload.len() < 12 || payload[..8] != MAGIC_HGRAVS01 {
        return Err(Error::Model("HGRAVS01 peek magic mismatch".into()));
    }
    let header_len =
        u32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]) as usize;
    let body_offset = 12usize
        .checked_add(header_len)
        .ok_or_else(|| Error::Model("HGRAVS01 peek header length overflows".into()))?;
    if body_offset > payload.len() {
        return Err(Error::Model("HGRAVS01 peek header exceeds payload".into()));
    }
    let header: Value = serde_json::from_slice(&payload[12..body_offset])
        .map_err(|error| Error::Model(format!("HGRAVS01 peek JSON: {error}")))?;
    let object = header
        .as_object()
        .ok_or_else(|| Error::Model("HGRAVS01 peek header is not an object".into()))?;
    require_str(
        object,
        "schema",
        "hawking.gravity.activation_weighted_svd_low_rank.v1",
    )?;
    let rank = json_usize(object, "rank")?;
    let bits = u8::try_from(json_u64(object, "factor_bits")?)
        .map_err(|_| Error::Model("HGRAVS01 peek factor_bits do not fit u8".into()))?;
    let group_size = json_usize(object, "factor_group_size")?;
    let matrix = json_pair(object, "matrix_shape")?;
    let left_meta = json_object(object, "left")?;
    let right_meta = json_object(object, "right")?;
    let left_shape = json_pair_value(left_meta, "shape")?;
    let right_shape = json_pair_value(right_meta, "shape")?;
    let left_elements = json_usize_value(left_meta, "elements")?;
    let right_elements = json_usize_value(right_meta, "elements")?;
    let left_scale_bytes = json_usize_value(left_meta, "scale_bytes")?;
    let left_code_bytes = json_usize_value(left_meta, "code_bytes")?;
    let right_scale_bytes = json_usize_value(right_meta, "scale_bytes")?;
    let right_code_bytes = json_usize_value(right_meta, "code_bytes")?;
    let left_groups = json_usize_value(left_meta, "groups")?;
    let right_groups = json_usize_value(right_meta, "groups")?;
    let body = &payload[body_offset..];
    let left_body = left_scale_bytes + left_code_bytes;
    let right_body = right_scale_bytes + right_code_bytes;
    if body.len() != left_body + right_body {
        return Err(Error::Model(
            "HGRAVS01 peek body length disagrees with factor ledgers".into(),
        ));
    }
    let left_scales = body[..left_scale_bytes].to_vec();
    let left_packed = body[left_scale_bytes..left_body].to_vec();
    let right_scales = body[left_body..left_body + right_scale_bytes].to_vec();
    let right_packed = body[left_body + right_scale_bytes..].to_vec();
    let left_all = unpack_unsigned_lsb(&left_packed, left_groups * group_size, bits)?;
    let right_all = unpack_unsigned_lsb(&right_packed, right_groups * group_size, bits)?;
    if left_all.len() < left_elements || right_all.len() < right_elements {
        return Err(Error::Model(
            "HGRAVS01 peek unpacked fewer codes than elements".into(),
        ));
    }
    Ok(Hgravs01CodeView {
        rank,
        bits,
        group_size,
        matrix_shape: matrix,
        left_shape,
        right_shape,
        left_elements,
        right_elements,
        left_codes: left_all[..left_elements].to_vec(),
        right_codes: right_all[..right_elements].to_vec(),
        left_code_bytes: left_packed,
        right_code_bytes: right_packed,
        left_scale_bytes: left_scales,
        right_scale_bytes: right_scales,
    })
}

fn require_str(object: &Map<String, Value>, key: &str, expected: &str) -> Result<()> {
    match object.get(key).and_then(Value::as_str) {
        Some(value) if value == expected => Ok(()),
        _ => Err(Error::Model(format!(
            "HGRAVS01 peek {key} is not {expected:?}"
        ))),
    }
}

fn json_u64(object: &Map<String, Value>, key: &str) -> Result<u64> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| Error::Model(format!("HGRAVS01 peek {key} is not u64")))
}

fn json_usize(object: &Map<String, Value>, key: &str) -> Result<usize> {
    usize::try_from(json_u64(object, key)?)
        .map_err(|_| Error::Model(format!("HGRAVS01 peek {key} does not fit usize")))
}

fn json_object<'a>(object: &'a Map<String, Value>, key: &str) -> Result<&'a Map<String, Value>> {
    object
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| Error::Model(format!("HGRAVS01 peek {key} is not an object")))
}

fn json_pair(object: &Map<String, Value>, key: &str) -> Result<[usize; 2]> {
    let array = object
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Model(format!("HGRAVS01 peek {key} is not an array")))?;
    if array.len() != 2 {
        return Err(Error::Model(format!("HGRAVS01 peek {key} is not length 2")));
    }
    let a = array[0]
        .as_u64()
        .ok_or_else(|| Error::Model(format!("HGRAVS01 peek {key}[0] is not u64")))?;
    let b = array[1]
        .as_u64()
        .ok_or_else(|| Error::Model(format!("HGRAVS01 peek {key}[1] is not u64")))?;
    Ok([
        usize::try_from(a).map_err(|_| Error::Model("shape does not fit usize".into()))?,
        usize::try_from(b).map_err(|_| Error::Model("shape does not fit usize".into()))?,
    ])
}

fn json_pair_value(object: &Map<String, Value>, key: &str) -> Result<[usize; 2]> {
    json_pair(object, key)
}

fn json_usize_value(object: &Map<String, Value>, key: &str) -> Result<usize> {
    json_usize(object, key)
}

pub fn factor_elements_used(packed: &UniformFactorPacked) -> Result<Vec<u8>> {
    let count = packed.groups * packed.group_size;
    let codes = unpack_unsigned_lsb(&packed.codes, count, packed.bits)?;
    let used = packed.rows * packed.cols;
    if codes.len() < used {
        return Err(Error::Model(
            "uniform factor has fewer codes than rows*cols".into(),
        ));
    }
    Ok(codes[..used].to_vec())
}

pub fn json_activity(activity: &WordActivity) -> Value {
    serde_json::json!({
        "word_bits": activity.word_bits,
        "words": activity.words,
        "transitions": activity.transitions,
        "mean_hamming": activity.mean_hamming,
        "transitions_per_byte": activity.transitions_per_byte,
        "alpha": activity.alpha,
        "alpha_vs_random_0_5": activity.alpha / RANDOM_ALPHA,
    })
}

pub fn json_code_activity(activity: &CodeActivity) -> Value {
    serde_json::json!({
        "bits": activity.bits,
        "codes": activity.codes,
        "transitions": activity.transitions,
        "mean_hamming": activity.mean_hamming,
        "alpha": activity.alpha,
        "alpha_vs_random_0_5": activity.alpha / RANDOM_ALPHA,
        "histogram": activity.histogram,
    })
}

/// Re-pack the same codes MSB-first and report packed-word activity. Decode
/// must extract MSB-first; decoded *values* stay identical iff the extract
/// matches. Packed byte count is unchanged.
pub fn msb_pack_activity(codes: &[u8], bits: u8) -> Result<(Vec<u8>, WordActivity, WordActivity)> {
    let lsb = pack_unsigned_lsb(codes, bits)?;
    let msb = pack_unsigned_msb(codes, bits)?;
    if lsb.len() != msb.len() {
        return Err(Error::Model(
            "MSB-first pack changed the packed byte count".into(),
        ));
    }
    let expected = packed_byte_count(codes.len(), bits)?;
    if lsb.len() != expected {
        return Err(Error::Model(
            "LSB pack length disagrees with packed geometry".into(),
        ));
    }
    let lsb_act = packed_word_activity(&lsb, 32);
    let msb_act = packed_word_activity(&msb, 32);
    Ok((msb, lsb_act, msb_act))
}

pub fn extract_unsigned_pub(codes: &[u8], element: usize, bits: u8) -> u16 {
    extract_unsigned(codes, element, bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::qwen_complete_binary::{
        deterministic_matrix, pack_binary_group, pack_uniform_factor, uniform_factor_value,
        Q80_BINARY_GROUP_SIZE,
    };

    #[test]
    fn remap_is_bit_identical_and_byte_count_invariant() {
        let values = deterministic_matrix(32, 64, 3);
        let packed = pack_uniform_factor(&values, 32, 64, 3, 64).unwrap();
        let used = factor_elements_used(&packed).unwrap();
        let matrix = transition_matrix(&used, 3);
        let assignment = minimize_assignment(&matrix, 3);
        let remapped = remap_uniform_factor(&packed, &assignment).unwrap();
        assert_eq!(remapped.packed.codes.len(), packed.codes.len());
        let original_q = uniform_factor_signed_levels(&packed, None).unwrap();
        let remapped_q =
            uniform_factor_signed_levels(&remapped.packed, Some(&remapped.lut)).unwrap();
        assert_eq!(original_q, remapped_q);
        for row in 0..packed.rows {
            for col in 0..packed.cols {
                let element = row * packed.cols + col;
                let scale =
                    half::f16::from_bits(packed.scales_f16[element / packed.group_size]).to_f32();
                let expected = original_q[element] as f32 * scale;
                assert_eq!(
                    uniform_factor_value(&packed, row, col).to_bits(),
                    expected.to_bits()
                );
                let got = remapped.lut[extract_unsigned(
                    &remapped.packed.codes,
                    element,
                    remapped.packed.bits,
                ) as usize] as f32
                    * scale;
                assert_eq!(got.to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn polarity_flip_preserves_binary_hamming() {
        let values = deterministic_matrix(8, 256, 1);
        let packed = pack_binary_group(&values, 8, 256, Q80_BINARY_GROUP_SIZE).unwrap();
        let bits = binary_sign_bits(&packed);
        let original = unpacked_code_activity(&bits, 1);
        let flipped = invert_sign_bytes(&packed.signs, bits.len());
        let mut flipped_bits = Vec::with_capacity(bits.len());
        for flat in 0..bits.len() {
            flipped_bits.push((flipped[flat >> 3] >> (flat & 7)) & 1);
        }
        let inverted = unpacked_code_activity(&flipped_bits, 1);
        assert_eq!(original.transitions, inverted.transitions);
        assert_eq!(original.alpha, inverted.alpha);
    }

    #[test]
    fn exact_assignment_never_worse_than_identity() {
        let codes = vec![0u8, 1, 3, 2, 6, 7, 5, 4, 0, 1];
        let matrix = transition_matrix(&codes, 3);
        let identity = with_cost(identity_assignment(3), &matrix);
        let best = minimize_assignment(&matrix, 3);
        assert!(best.cost <= identity.cost);
        if best.cost < identity.cost {
            assert_ne!(best.method, "identity");
        }
    }

    #[test]
    fn peaked_zero_level_assigns_all_zero_code() {
        // Real hgravs factors pile on the offset-binary zero (code = bound).
        // The useful free move is to give that mode the all-zero codeword.
        let mut codes = vec![3u8; 200];
        for (i, slot) in codes.iter_mut().enumerate() {
            if i % 11 == 0 {
                *slot = 2;
            } else if i % 13 == 0 {
                *slot = 4;
            }
        }
        let matrix = transition_matrix(&codes, 3);
        let best = minimize_assignment(&matrix, 3);
        let mode_code = best.symbol_to_code[3];
        assert!(
            mode_code == 0 || mode_code == 7,
            "mode should land on an all-equal codeword, got {mode_code}"
        );
        let identity = assignment_cost(&matrix, &identity_assignment(3).symbol_to_code);
        assert!(best.cost < identity);
    }

    #[test]
    fn gray_roundtrip() {
        for value in 0..16u8 {
            assert_eq!(gray_to_binary(binary_to_gray(value)), value);
        }
    }

    #[test]
    fn msb_pack_preserves_byte_count() {
        let codes = vec![0u8, 1, 2, 3, 4, 5, 6, 3, 2, 1];
        let (msb, lsb_act, msb_act) = msb_pack_activity(&codes, 3).unwrap();
        assert_eq!(msb.len(), pack_unsigned_lsb(&codes, 3).unwrap().len());
        assert!(lsb_act.words > 0);
        assert!(msb_act.words > 0);
    }
}

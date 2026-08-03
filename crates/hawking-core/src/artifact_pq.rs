//! gravity-pq CPU codec (GLM52CPK).
use crate::{Error, Result};
use half::f16;
const PQ_MAGIC: &[u8; 8] = b"GLM52CPK";
/// Llama residual-PQ grammar: each chunk selects one codeword from every
/// additive stage. This is deliberately distinct from `GLM52CPK`, whose
/// codebooks address disjoint subspaces and cannot represent residual stages.
const RESIDUAL_PQ_MAGIC: &[u8; 8] = b"LLM52RPK";
const PQ_HEADER_LEN: usize = 64;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PqHeader {
    pub d: u16,
    pub s: u16,
    pub sub: u16,
    pub card: u16,
    pub rows: u32,
    pub cols: u32,
    pub nchunk: u32,
    pub seed: u32,
    pub bits: u16,
    pub rotate: u8,
    pub n_codebooks: u8,
}

/// Fixed 64-byte header for `llama.residual-pq.v1` payloads.
///
/// Codebooks are `[stage][card][D]` fp16 and indices are MSB-first in
/// `[row][chunk][stage]` order. `D` is the complete source chunk width for
/// every stage; unlike ordinary PQ, stages never split the input vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualPqHeader {
    pub d: u16,
    pub stages: u16,
    pub card: u16,
    pub rows: u32,
    pub cols: u32,
    pub nchunk: u32,
    pub seed: u32,
    pub bits: u16,
}

pub fn parse_residual_pq_header(payload: &[u8]) -> Result<ResidualPqHeader> {
    if payload.len() < PQ_HEADER_LEN {
        return Err(Error::Gravity(format!(
            "residual pq short header: {} bytes, need {PQ_HEADER_LEN}",
            payload.len()
        )));
    }
    require_magic!(payload, RESIDUAL_PQ_MAGIC, "llama residual-pq")?;
    let f = &payload[8..];
    let (d, stages, card) = (le_u16!(f, 0), le_u16!(f, 2), le_u16!(f, 4));
    let (rows, cols, nchunk, seed) = (
        le_u32!(f, 8),
        le_u32!(f, 12),
        le_u32!(f, 16),
        le_u32!(f, 20),
    );
    let bits = le_u16!(f, 24);
    if d == 0 || stages == 0 || card < 2 || bits == 0 || bits > 16 {
        return Err(Error::Gravity(format!(
            "residual pq invalid geometry d={d} stages={stages} card={card} bits={bits}"
        )));
    }
    if card as u32 > (1u32 << bits) || cols != nchunk.saturating_mul(d as u32) {
        return Err(Error::Gravity(format!(
            "residual pq invalid card/shape card={card} bits={bits} cols={cols} nchunk={nchunk} d={d}"
        )));
    }
    if f[26] != 0 || f[27] != stages as u8 {
        return Err(Error::Gravity(
            "residual pq flags/codebook count are not canonical".into(),
        ));
    }
    Ok(ResidualPqHeader {
        d,
        stages,
        card,
        rows,
        cols,
        nchunk,
        seed,
        bits,
    })
}
pub fn parse_pq_header(payload: &[u8]) -> Result<PqHeader> {
    if payload.len() < PQ_HEADER_LEN {
        return Err(Error::Gravity(format!(
            "pq short header: {} bytes, need {PQ_HEADER_LEN}",
            payload.len()
        )));
    }
    require_magic!(payload, PQ_MAGIC, "gravity-pq")?;
    let f = &payload[8..];
    let (d, s, sub, card) = (le_u16!(f, 0), le_u16!(f, 2), le_u16!(f, 4), le_u16!(f, 6));
    let (rows, cols, nchunk, seed) = (
        le_u32!(f, 8),
        le_u32!(f, 12),
        le_u32!(f, 16),
        le_u32!(f, 20),
    );
    let bits = le_u16!(f, 24);
    let (rotate, n_codebooks) = (f[26], f[27]);
    if rotate > 1 {
        return Err(Error::Gravity(format!("pq rotate {rotate} not 0/1")));
    }
    if d != s.wrapping_mul(sub) {
        return Err(Error::Gravity(format!("pq D {d} != S {s} * sub {sub}")));
    }
    if cols != nchunk.wrapping_mul(d as u32) {
        return Err(Error::Gravity(format!(
            "pq cols {cols} != nchunk {nchunk} * D {d}"
        )));
    }
    if n_codebooks as u16 != s {
        return Err(Error::Gravity(format!(
            "pq n_codebooks {n_codebooks} != S {s}"
        )));
    }
    Ok(PqHeader {
        d,
        s,
        sub,
        card,
        rows,
        cols,
        nchunk,
        seed,
        bits,
        rotate,
        n_codebooks,
    })
}
fn unpack_bits(stream: &[u8], count: usize, bits: u32) -> Result<Vec<u32>> {
    let need_bytes = (count as u64 * bits as u64).div_ceil(8);
    if (stream.len() as u64) < need_bytes {
        return Err(Error::Gravity(format!(
            "pq idx short: have {} bytes, need {need_bytes}",
            stream.len()
        )));
    }
    let mask: u64 = if bits >= 32 {
        u32::MAX as u64
    } else {
        (1u64 << bits) - 1
    };
    let mut out = Vec::with_capacity(count);
    let mut acc: u64 = 0;
    let mut nbits: u32 = 0;
    let mut pos: usize = 0;
    for _ in 0..count {
        while nbits < bits {
            acc = (acc << 8) | stream[pos] as u64;
            nbits += 8;
            pos += 1;
        }
        out.push(((acc >> (nbits - bits)) & mask) as u32);
        nbits -= bits;
    }
    Ok(out)
}

/// Random-access bit-packed index; must match sequential `unpack_bits`.
///
/// Only the test module reads it, and CI runs clippy without `--all-targets`.
#[cfg_attr(not(test), allow(dead_code))]
fn index_at(stream: &[u8], i: usize, bits: u32) -> u32 {
    let bitoff = i * bits as usize;
    let (mut acc, mut taken, mut byte, skip) = (0u64, 0u32, bitoff / 8, (bitoff % 8) as u32);
    while taken < skip + bits {
        acc = (acc << 8) | *stream.get(byte).unwrap_or(&0) as u64;
        taken += 8;
        byte += 1;
    }
    let mask = if bits >= 32 {
        u32::MAX as u64
    } else {
        (1u64 << bits) - 1
    };
    ((acc >> (taken - skip - bits)) & mask) as u32
}
pub fn pq_sections(payload: &[u8]) -> Result<(&[u8], &[u8])> {
    let h = parse_pq_header(payload)?;
    if h.rotate != 0 {
        return Err(Error::Gravity("pq rotate=1 unsupported".into()));
    }
    let cb_values = h.n_codebooks as usize * h.card as usize * h.sub as usize;
    let cb_end = section_end!(PQ_HEADER_LEN, cb_values, 2, "pq cb");
    let idx_count = h.rows as usize * h.nchunk as usize * h.s as usize;
    let idx_bytes = (idx_count as u64 * h.bits as u64).div_ceil(8) as usize;
    let idx_end = section_end!(cb_end, idx_bytes, 1, "pq idx");
    if payload.len() < idx_end {
        return Err(Error::Gravity(format!(
            "pq short: have {} bytes, need {idx_end}",
            payload.len()
        )));
    }
    Ok((&payload[PQ_HEADER_LEN..cb_end], &payload[cb_end..idx_end]))
}

pub fn residual_pq_sections(payload: &[u8]) -> Result<(&[u8], &[u8])> {
    let h = parse_residual_pq_header(payload)?;
    let cb_values = h.stages as usize * h.card as usize * h.d as usize;
    let cb_end = section_end!(PQ_HEADER_LEN, cb_values, 2, "residual pq cb");
    let idx_count = h.rows as usize * h.nchunk as usize * h.stages as usize;
    let idx_bytes = (idx_count as u64 * h.bits as u64).div_ceil(8) as usize;
    let idx_end = section_end!(cb_end, idx_bytes, 1, "residual pq idx");
    if payload.len() < idx_end {
        return Err(Error::Gravity(format!(
            "residual pq short: have {} bytes, need {idx_end}",
            payload.len()
        )));
    }
    Ok((&payload[PQ_HEADER_LEN..cb_end], &payload[cb_end..idx_end]))
}

pub fn pq_row(payload: &[u8], index: usize) -> Result<Vec<f32>> {
    PqTensor::from_payload(payload)?.row(index)
}
pub struct PqTensor {
    pub header: PqHeader,
    codebooks: Vec<f32>,
    indices: Vec<u16>,
}

/// CPU authority for `llama.residual-pq.v1`. It is intentionally a direct
/// compact execution, never a dense reconstruction cache.
pub struct ResidualPqTensor {
    pub header: ResidualPqHeader,
    codebooks: Vec<f32>,
    indices: Vec<u16>,
}

impl ResidualPqTensor {
    pub fn from_payload(payload: &[u8]) -> Result<ResidualPqTensor> {
        let header = parse_residual_pq_header(payload)?;
        let (cb, codes) = residual_pq_sections(payload)?;
        let mut codebooks = vec![0.0f32; cb.len() / 2];
        for (index, value) in codebooks.iter_mut().enumerate() {
            *value = f16::from_bits(u16::from_le_bytes(
                cb[index * 2..index * 2 + 2].try_into().unwrap(),
            ))
            .to_f32();
        }
        let index_count = header.rows as usize * header.nchunk as usize * header.stages as usize;
        let indices = unpack_bits(codes, index_count, header.bits as u32)?
            .into_iter()
            .map(|value| value as u16)
            .collect();
        Ok(ResidualPqTensor {
            header,
            codebooks,
            indices,
        })
    }

    pub fn matvec(&self, x: &[f32]) -> Result<Vec<f32>> {
        let h = self.header;
        if x.len() != h.cols as usize {
            return Err(Error::Gravity(format!(
                "residual pq matvec x {} != cols {}",
                x.len(),
                h.cols
            )));
        }
        let (d, stages, card, rows, chunks) = (
            h.d as usize,
            h.stages as usize,
            h.card as usize,
            h.rows as usize,
            h.nchunk as usize,
        );
        let mut y = vec![0.0f32; rows];
        for row in 0..rows {
            let mut sum = 0.0f32;
            for chunk in 0..chunks {
                for stage in 0..stages {
                    let code = self.indices[(row * chunks + chunk) * stages + stage] as usize;
                    let base = (stage * card + code) * d;
                    for j in 0..d {
                        sum += self.codebooks[base + j] * x[chunk * d + j];
                    }
                }
            }
            y[row] = sum;
        }
        Ok(y)
    }

    pub fn row(&self, row: usize) -> Result<Vec<f32>> {
        let h = self.header;
        let (d, stages, card, rows, chunks) = (
            h.d as usize,
            h.stages as usize,
            h.card as usize,
            h.rows as usize,
            h.nchunk as usize,
        );
        if row >= rows {
            return Err(Error::Gravity(format!("residual pq row {row} OOR {rows}")));
        }
        let mut out = vec![0.0f32; chunks * d];
        for chunk in 0..chunks {
            for stage in 0..stages {
                let code = self.indices[(row * chunks + chunk) * stages + stage] as usize;
                let base = (stage * card + code) * d;
                for j in 0..d {
                    out[chunk * d + j] += self.codebooks[base + j];
                }
            }
        }
        Ok(out)
    }
}
impl PqTensor {
    pub fn from_payload(payload: &[u8]) -> Result<PqTensor> {
        let h = parse_pq_header(payload)?;
        if h.rotate != 0 {
            return Err(Error::Gravity("pq rotate=1 unsupported".into()));
        }
        if h.bits > 16 {
            return Err(Error::Gravity(format!("pq bits {} >16", h.bits)));
        }
        let (sub, card, rows, nchunk) = (
            h.sub as usize,
            h.card as usize,
            h.rows as usize,
            h.nchunk as usize,
        );
        let cb_values = h.n_codebooks as usize * card * sub;
        let cb_end = section_end!(PQ_HEADER_LEN, cb_values, 2, "pq cb");
        if payload.len() < cb_end {
            return Err(Error::Gravity(format!(
                "pq short cb: have {} bytes, need {cb_end}",
                payload.len()
            )));
        }
        let mut codebooks = vec![0f32; cb_values];
        for (i, cbv) in codebooks.iter_mut().enumerate() {
            *cbv = f16::from_bits(u16::from_le_bytes(
                payload[PQ_HEADER_LEN + i * 2..PQ_HEADER_LEN + i * 2 + 2]
                    .try_into()
                    .unwrap(),
            ))
            .to_f32();
        }
        let idx_count = rows * nchunk * h.s as usize;
        let indices = unpack_bits(&payload[cb_end..], idx_count, h.bits as u32)?
            .into_iter()
            .map(|v| v as u16)
            .collect();
        Ok(PqTensor {
            header: h,
            codebooks,
            indices,
        })
    }
    fn matvec_acc<T, F>(&self, x: &[f32], zero: T, mut add: F) -> Result<Vec<T>>
    where
        T: Copy,
        F: FnMut(T, f32, f32) -> T,
    {
        let h = &self.header;
        if x.len() != h.cols as usize {
            return Err(Error::Gravity(format!(
                "pq matvec x {} != cols {}",
                x.len(),
                h.cols
            )));
        }
        let (d, s, sub, card, rows, nchunk) = (
            h.d as usize,
            h.s as usize,
            h.sub as usize,
            h.card as usize,
            h.rows as usize,
            h.nchunk as usize,
        );
        let mut y = vec![zero; rows];
        for sub_idx in 0..s {
            let cb_base = sub_idx * card * sub;
            let x_off = sub_idx * sub;
            for r in 0..rows {
                for c in 0..nchunk {
                    let code = self.indices[(r * nchunk + c) * s + sub_idx] as usize;
                    let cb_row = cb_base + code * sub;
                    let x_base = c * d + x_off;
                    for j in 0..sub {
                        y[r] = add(y[r], self.codebooks[cb_row + j], x[x_base + j]);
                    }
                }
            }
        }
        Ok(y)
    }
    pub fn matvec(&self, x: &[f32]) -> Result<Vec<f32>> {
        self.matvec_acc(x, 0f32, |acc, a, b| acc + a * b)
    }
    pub fn row(&self, index: usize) -> Result<Vec<f32>> {
        let h = &self.header;
        let (d, s, sub, card, rows, nchunk) = (
            h.d as usize,
            h.s as usize,
            h.sub as usize,
            h.card as usize,
            h.rows as usize,
            h.nchunk as usize,
        );
        if index >= rows {
            return Err(Error::Gravity(format!("pq row {index} OOR {rows} rows")));
        }
        let mut out = vec![0f32; nchunk * d];
        for c in 0..nchunk {
            for sub_idx in 0..s {
                let flat = (index * nchunk + c) * s + sub_idx;
                let code = self.indices[flat] as usize;
                let cb_row = sub_idx * card * sub + code * sub;
                let dst = c * d + sub_idx * sub;
                out[dst..dst + sub].copy_from_slice(&self.codebooks[cb_row..cb_row + sub]);
            }
        }
        Ok(out)
    }
}
pub fn pq_matvec(payload: &[u8], x: &[f32]) -> Result<Vec<f32>> {
    PqTensor::from_payload(payload)?.matvec(x)
}
pub fn pq_matvec_f64_authority(payload: &[u8], x: &[f32]) -> Result<Vec<f64>> {
    PqTensor::from_payload(payload)?.matvec_acc(x, 0f64, |acc, a, b| acc + (a as f64) * (b as f64))
}
pub(super) fn pack_indices(indices: &[u32], bits: u32) -> Result<Vec<u8>> {
    if bits == 0 || bits > 32 {
        return Err(Error::Gravity(format!(
            "pack bits {bits} out of range 1..32"
        )));
    }
    let max_val = if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    for (i, &v) in indices.iter().enumerate() {
        if v > max_val {
            return Err(Error::Gravity(format!("pack idx[{i}]={v} > {bits} bits")));
        }
    }
    let need_bytes = (indices.len() as u64 * bits as u64).div_ceil(8) as usize;
    let mut out = Vec::with_capacity(need_bytes);
    let mut acc: u64 = 0;
    let mut nbits: u32 = 0;
    for &v in indices {
        acc = (acc << bits) | (v as u64);
        nbits += bits;
        while nbits >= 8 {
            let shift = nbits - 8;
            out.push(((acc >> shift) & 0xff) as u8);
            nbits -= 8;
            acc = if nbits == 0 {
                0
            } else {
                acc & ((1u64 << nbits) - 1)
            };
        }
    }
    if nbits > 0 {
        out.push(((acc << (8 - nbits)) & 0xff) as u8);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{index_at, pack_indices, unpack_bits, ResidualPqTensor, RESIDUAL_PQ_MAGIC};
    #[test]
    fn unpack_bits_matches_hand_packed_7bit_values() {
        let p: [u8; 7] = [0x00, 0x07, 0xF7, 0xF8, 0x00, 0xC1, 0x01];
        assert_eq!(
            unpack_bits(&p, 8, 7).unwrap(),
            vec![0, 1, 126, 127, 64, 3, 2, 1]
        );
    }
    #[test]
    fn index_at_matches_sequential_unpack() {
        let p: [u8; 7] = [0x00, 0x07, 0xF7, 0xF8, 0x00, 0xC1, 0x01];
        let seq = unpack_bits(&p, 8, 7).unwrap();
        for (i, &w) in seq.iter().enumerate() {
            assert_eq!(index_at(&p, i, 7), w, "index {i}");
        }
    }
    #[test]
    fn unpack_bits_rejects_short_stream() {
        assert!(unpack_bits(&[0x00, 0x07, 0xF7, 0xF8, 0x00, 0xC1], 8, 7).is_err());
    }

    #[test]
    fn residual_pq_executes_additive_stages_without_dense_reconstruction() {
        let mut payload = vec![0u8; 64];
        payload[..8].copy_from_slice(RESIDUAL_PQ_MAGIC);
        payload[8..10].copy_from_slice(&2u16.to_le_bytes()); // D
        payload[10..12].copy_from_slice(&2u16.to_le_bytes()); // stages
        payload[12..14].copy_from_slice(&2u16.to_le_bytes()); // card
        payload[16..20].copy_from_slice(&2u32.to_le_bytes()); // rows
        payload[20..24].copy_from_slice(&4u32.to_le_bytes()); // cols
        payload[24..28].copy_from_slice(&2u32.to_le_bytes()); // chunks
        payload[32..34].copy_from_slice(&1u16.to_le_bytes()); // index bits
        payload[35] = 2; // canonical codebook count == stages
        for value in [1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0] {
            payload.extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
        }
        // [row][chunk][stage], MSB-first: r0=(0,1,1,0), r1=(1,1,0,0).
        payload.extend_from_slice(&pack_indices(&[0, 1, 1, 0, 1, 1, 0, 0], 1).unwrap());
        let packed = ResidualPqTensor::from_payload(&payload).unwrap();
        assert_eq!(packed.row(0).unwrap(), vec![31.0, 42.0, 13.0, 24.0]);
        assert_eq!(
            packed.matvec(&[1.0, 2.0, 3.0, 4.0]).unwrap(),
            vec![250.0, 242.0]
        );
    }
}

//! `.gravity` container reader + `gravity-pq` tensor codec.
//!
//! Mirrors the container framing in `tools/condense/gravity_format.py` and
//! the `gravity-pq` tensor payload codec in `tools/condense/glm52_pack.py`.
//! [`pq_matvec`] mirrors the CPU reference `pq_execute` in
//! `tools/condense/gravity_forge.py`.
//!
//! Container layout (little-endian throughout):
//! ```text
//! bytes 0..8    magic == b"GRAVITY\0"
//! bytes 8..12   format_version: u32   (reject > 1)
//! bytes 12..20  header_len: u64
//! bytes 20..20+header_len   UTF-8 JSON header
//! bytes 20+header_len..     tensor payload body (descriptor offsets are
//!                           relative to this point)
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use half::f16;
use memmap2::Mmap;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{Error, Result};

const MAGIC: &[u8; 8] = b"GRAVITY\0";
const MAX_FORMAT_VERSION: u32 = 1;
const PREFIX_LEN: usize = 20;

/// One tensor's location + integrity info inside a `.gravity` shard's
/// `tensors` header array.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorDescriptor {
    pub name: String,
    pub codec: String,
    /// Byte offset relative to the body (i.e. relative to `20 + header_len`).
    pub offset: u64,
    pub bytes: u64,
    /// Hex-encoded SHA-256 of this tensor's payload bytes.
    pub sha256: String,
    pub shape: Vec<u64>,
    pub elements: u64,
}

#[derive(Deserialize)]
struct GravityHeader {
    tensors: Vec<TensorDescriptor>,
}

/// An mmap-backed `.gravity` shard. `open` parses only the prefix + JSON
/// header (cheap); tensor bytes are read/copied on demand via
/// [`GravityShard::read_tensor`].
pub struct GravityShard {
    mmap: Mmap,
    body_offset: u64,
    tensors: HashMap<String, TensorDescriptor>,
    tensor_order: Vec<String>,
    /// The parsed JSON header with `tensors` removed (that part is modeled
    /// above). Carries whatever else the writer put there — `model`,
    /// `compression`, `integrity`, `architecture`, `tokenizer`, `shard`,
    /// `schema`, etc. — verbatim and untyped.
    pub extra: serde_json::Value,
}

impl GravityShard {
    pub fn open(path: &Path) -> Result<GravityShard> {
        let f = File::open(path)?;
        // Safety: treated as read-only for the lifetime of `GravityShard`;
        // truncation under us is caller error, same contract as gguf.rs.
        let mmap = unsafe { Mmap::map(&f)? };
        Self::from_mmap(mmap)
    }

    fn from_mmap(mmap: Mmap) -> Result<GravityShard> {
        if mmap.len() < PREFIX_LEN {
            return Err(Error::Gravity(format!(
                "file too short for prefix: {} bytes",
                mmap.len()
            )));
        }
        let magic = &mmap[0..8];
        if magic != MAGIC {
            return Err(Error::Gravity(format!(
                "bad magic {magic:?}, expected {MAGIC:?}"
            )));
        }
        let format_version = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        if format_version > MAX_FORMAT_VERSION {
            return Err(Error::Gravity(format!(
                "unsupported format_version {format_version} (max {MAX_FORMAT_VERSION})"
            )));
        }
        let header_len = u64::from_le_bytes(mmap[12..20].try_into().unwrap());
        let header_end = (PREFIX_LEN as u64)
            .checked_add(header_len)
            .ok_or_else(|| Error::Gravity("header_len overflow".into()))?;
        if header_end > mmap.len() as u64 {
            return Err(Error::Gravity(format!(
                "header end {header_end} past file length {}",
                mmap.len()
            )));
        }
        let header_bytes = &mmap[PREFIX_LEN..header_end as usize];

        let mut header_value: serde_json::Value = serde_json::from_slice(header_bytes)
            .map_err(|e| Error::Gravity(format!("header JSON parse: {e}")))?;
        let header: GravityHeader = serde_json::from_value(header_value.clone())
            .map_err(|e| Error::Gravity(format!("header `tensors` parse: {e}")))?;
        if let Some(obj) = header_value.as_object_mut() {
            obj.remove("tensors");
        }

        let mut tensors = HashMap::with_capacity(header.tensors.len());
        let mut tensor_order = Vec::with_capacity(header.tensors.len());
        for d in header.tensors {
            tensor_order.push(d.name.clone());
            tensors.insert(d.name.clone(), d);
        }

        Ok(GravityShard {
            mmap,
            body_offset: header_end,
            tensors,
            tensor_order,
            extra: header_value,
        })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensor_order.iter().map(String::as_str)
    }

    pub fn descriptor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.get(name)
    }

    /// Read one tensor's raw payload bytes (untouched — codec-specific
    /// decoding, e.g. `gravity-pq`, is the caller's job). When
    /// `verify_hash` is set, the payload's SHA-256 is checked against the
    /// descriptor's `sha256` and a mismatch is an error.
    pub fn read_tensor(&self, name: &str, verify_hash: bool) -> Result<Vec<u8>> {
        let d = self
            .descriptor(name)
            .ok_or_else(|| Error::Gravity(format!("no such tensor {name:?}")))?;
        let start = self
            .body_offset
            .checked_add(d.offset)
            .ok_or_else(|| Error::Gravity(format!("tensor {name}: offset overflow")))?;
        let end = start
            .checked_add(d.bytes)
            .ok_or_else(|| Error::Gravity(format!("tensor {name}: end overflow")))?;
        if end > self.mmap.len() as u64 {
            return Err(Error::Gravity(format!(
                "tensor {name}: end {end} past file length {}",
                self.mmap.len()
            )));
        }
        let payload = &self.mmap[start as usize..end as usize];
        if verify_hash {
            let mut h = Sha256::new();
            h.update(payload);
            let digest = h.finalize();
            let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
            if hex != d.sha256 {
                return Err(Error::Gravity(format!(
                    "tensor {name}: sha256 mismatch: expected {}, got {hex}",
                    d.sha256
                )));
            }
        }
        Ok(payload.to_vec())
    }
}

// ---------------------------------------------------------------------
// `gravity-pq` tensor payload codec.
// ---------------------------------------------------------------------

const PQ_MAGIC: &[u8; 8] = b"GLM52CPK";
/// Fixed size of the `gravity-pq` payload header (8B magic + 28B packed
/// fields + 28B zero padding), i.e. where the codebooks start.
const PQ_HEADER_LEN: usize = 64;
/// Size in bytes of the `"<HHHHIIIIH?B"` packed field block that follows
/// the magic.
const PQ_FIELDS_LEN: usize = 28;

/// Parsed `gravity-pq` payload header (bytes `8..36` of the payload).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PqHeader {
    pub d: u16,
    /// Subspaces. Production artifacts use 1.
    pub s: u16,
    pub sub: u16,
    /// Codebook cardinality, e.g. 128 or 256.
    pub card: u16,
    pub rows: u32,
    pub cols: u32,
    pub nchunk: u32,
    pub seed: u32,
    /// Index width in bits: 7 for card=128, 8 for card=256.
    pub bits: u16,
    /// 0 or 1. Rotated payloads (`rotate == 1`) are not yet supported.
    pub rotate: u8,
    /// Always equals `s`.
    pub n_codebooks: u8,
}

/// Parse a `gravity-pq` payload's fixed 64-byte header.
pub fn parse_pq_header(payload: &[u8]) -> Result<PqHeader> {
    if payload.len() < PQ_HEADER_LEN {
        return Err(Error::Gravity(format!(
            "gravity-pq payload too short for header: {} bytes, need {PQ_HEADER_LEN}",
            payload.len()
        )));
    }
    let magic = &payload[0..8];
    if magic != PQ_MAGIC {
        return Err(Error::Gravity(format!(
            "bad gravity-pq magic {magic:?}, expected {PQ_MAGIC:?}"
        )));
    }
    let f = &payload[8..8 + PQ_FIELDS_LEN];
    let d = u16::from_le_bytes(f[0..2].try_into().unwrap());
    let s = u16::from_le_bytes(f[2..4].try_into().unwrap());
    let sub = u16::from_le_bytes(f[4..6].try_into().unwrap());
    let card = u16::from_le_bytes(f[6..8].try_into().unwrap());
    let rows = u32::from_le_bytes(f[8..12].try_into().unwrap());
    let cols = u32::from_le_bytes(f[12..16].try_into().unwrap());
    let nchunk = u32::from_le_bytes(f[16..20].try_into().unwrap());
    let seed = u32::from_le_bytes(f[20..24].try_into().unwrap());
    let bits = u16::from_le_bytes(f[24..26].try_into().unwrap());
    let rotate = f[26];
    let n_codebooks = f[27];

    if rotate > 1 {
        return Err(Error::Gravity(format!(
            "gravity-pq header: rotate byte {rotate} is not 0/1"
        )));
    }
    if d != s.wrapping_mul(sub) {
        return Err(Error::Gravity(format!(
            "gravity-pq header: D {d} != S {s} * sub {sub}"
        )));
    }
    if cols != nchunk.wrapping_mul(d as u32) {
        return Err(Error::Gravity(format!(
            "gravity-pq header: cols {cols} != nchunk {nchunk} * D {d}"
        )));
    }
    if n_codebooks as u16 != s {
        return Err(Error::Gravity(format!(
            "gravity-pq header: n_codebooks {n_codebooks} != S {s}"
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

/// Unpack `count` MSB-first, `bits`-wide unsigned values from a
/// numpy-`packbits`-order bitstream: value `i` occupies stream bits
/// `[i*bits, (i+1)*bits)`, and stream bit `k` is bit `7 - k%8` of byte
/// `k/8`. Trailing bits in the final byte (if any) are ignored.
fn unpack_bits(stream: &[u8], count: usize, bits: u32) -> Result<Vec<u32>> {
    let need_bits = count as u64 * bits as u64;
    let need_bytes = need_bits.div_ceil(8);
    if (stream.len() as u64) < need_bytes {
        return Err(Error::Gravity(format!(
            "gravity-pq index bitstream too short: have {} bytes, need {need_bytes}",
            stream.len()
        )));
    }
    // Rolling MSB-first window: feed whole bytes into `acc`, take `bits`
    // off the top of the `nbits` live ones. `nbits` never exceeds
    // `bits + 7 <= 39`, so shifting `acc` left by 8 cannot lose a live bit,
    // and the mask discards the consumed ones still sitting above them.
    let mask: u64 = if bits >= 32 { u32::MAX as u64 } else { (1u64 << bits) - 1 };
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

/// Byte spans of a `gravity-pq` payload's two sections: the f16 codebooks
/// and the packed index stream. Both are returned verbatim — the GPU path
/// uploads them as-is, since the kernel reads `half` and walks the packed
/// stream itself, so nothing is widened or unpacked on the way in.
pub fn pq_sections(payload: &[u8]) -> Result<(&[u8], &[u8])> {
    let h = parse_pq_header(payload)?;
    if h.rotate != 0 {
        return Err(Error::Gravity(
            "rotated gravity-pq artifacts (rotate=1) are not yet supported".into(),
        ));
    }
    let cb_values = h.n_codebooks as usize * h.card as usize * h.sub as usize;
    let cb_end = PQ_HEADER_LEN
        .checked_add(cb_values.checked_mul(2).unwrap_or(usize::MAX))
        .ok_or_else(|| Error::Gravity("gravity-pq codebook size overflow".into()))?;
    let idx_count = h.rows as usize * h.nchunk as usize * h.s as usize;
    let idx_bytes = (idx_count as u64 * h.bits as u64).div_ceil(8) as usize;
    let idx_end = cb_end
        .checked_add(idx_bytes)
        .ok_or_else(|| Error::Gravity("gravity-pq index size overflow".into()))?;
    if payload.len() < idx_end {
        return Err(Error::Gravity(format!(
            "gravity-pq payload too short: have {} bytes, need {idx_end}",
            payload.len()
        )));
    }
    Ok((&payload[PQ_HEADER_LEN..cb_end], &payload[cb_end..idx_end]))
}

/// Read one `bits`-wide MSB-first value at position `i` of a packed index
/// stream, without walking the values before it.
fn index_at(stream: &[u8], i: usize, bits: u32) -> u32 {
    let bitoff = i * bits as usize;
    let mut acc: u64 = 0;
    let mut taken = 0u32;
    let mut byte = bitoff / 8;
    let skip = (bitoff % 8) as u32;
    // Pull whole bytes until `skip + bits` of them are in hand.
    while taken < skip + bits {
        acc = (acc << 8) | *stream.get(byte).unwrap_or(&0) as u64;
        taken += 8;
        byte += 1;
    }
    let mask: u64 = if bits >= 32 { u32::MAX as u64 } else { (1u64 << bits) - 1 };
    ((acc >> (taken - skip - bits)) & mask) as u32
}

/// Decode a single row of a `gravity-pq` payload straight from its bytes —
/// `cols` values, touching only that row's chunk codes. This is the
/// embedding-lookup path: materializing a `[vocab, hidden]` matrix to read
/// one row of it would defeat the point of the format.
pub fn pq_row(payload: &[u8], index: usize) -> Result<Vec<f32>> {
    let h = parse_pq_header(payload)?;
    let (cb, codes) = pq_sections(payload)?;
    if index >= h.rows as usize {
        return Err(Error::Gravity(format!(
            "pq_row: index {index} out of range for {} rows",
            h.rows
        )));
    }
    let (d, s, sub, card, nchunk) = (
        h.d as usize,
        h.s as usize,
        h.sub as usize,
        h.card as usize,
        h.nchunk as usize,
    );
    let mut out = vec![0f32; nchunk * d];
    for c in 0..nchunk {
        for sub_idx in 0..s {
            let flat = (index * nchunk + c) * s + sub_idx;
            let code = index_at(codes, flat, h.bits as u32) as usize;
            if code >= card {
                return Err(Error::Gravity(format!(
                    "pq_row: code {code} exceeds codebook cardinality {card}"
                )));
            }
            let cb_row = (sub_idx * card + code) * sub;
            let dst = c * d + sub_idx * sub;
            for j in 0..sub {
                let off = (cb_row + j) * 2;
                out[dst + j] =
                    f16::from_bits(u16::from_le_bytes(cb[off..off + 2].try_into().unwrap()))
                        .to_f32();
            }
        }
    }
    Ok(out)
}

/// A `gravity-pq` payload decoded once into the two things execution
/// actually needs: codebooks widened to f32, and the index bitstream
/// unpacked to one value per (row, chunk, subspace).
///
/// [`pq_matvec`] decodes on every call, which is correct but re-walks the
/// bitstream bit by bit each time. A forward pass hits the same tensor
/// once per token, so anything holding weights across tokens wants this
/// instead.
pub struct PqTensor {
    pub header: PqHeader,
    /// `n_codebooks * card * sub` f32 values, flat index `(s*card + code)*sub + j`.
    codebooks: Vec<f32>,
    /// `rows * nchunk * s` codes, flat index `(r*nchunk + c)*s + subspace`.
    /// `u16` covers every `card` the header can express (`bits <= 16`).
    indices: Vec<u16>,
}

impl PqTensor {
    /// Decode a `gravity-pq` payload. Rejects rotated payloads and any
    /// `bits > 16` rather than guessing either construction.
    pub fn from_payload(payload: &[u8]) -> Result<PqTensor> {
        let h = parse_pq_header(payload)?;
        if h.rotate != 0 {
            // TODO(rotation): port `_pq_rotation_np(D, seed)` from
            // tools/condense/gravity_forge.py — do not guess the construction.
            return Err(Error::Gravity(
                "rotated gravity-pq artifacts (rotate=1) are not yet supported".into(),
            ));
        }
        if h.bits > 16 {
            return Err(Error::Gravity(format!(
                "gravity-pq bits {} exceeds the 16-bit index width this decoder stores",
                h.bits
            )));
        }

        let sub = h.sub as usize;
        let card = h.card as usize;
        let rows = h.rows as usize;
        let nchunk = h.nchunk as usize;

        // Codebooks: `n_codebooks` back to back, each `card * sub` f16 values.
        let cb_values = h.n_codebooks as usize * card * sub;
        let cb_bytes = cb_values
            .checked_mul(2)
            .ok_or_else(|| Error::Gravity("gravity-pq codebook size overflow".into()))?;
        let cb_start = PQ_HEADER_LEN;
        let cb_end = cb_start
            .checked_add(cb_bytes)
            .ok_or_else(|| Error::Gravity("gravity-pq codebook size overflow".into()))?;
        if payload.len() < cb_end {
            return Err(Error::Gravity(format!(
                "gravity-pq payload too short for codebooks: have {} bytes, need {cb_end}",
                payload.len()
            )));
        }
        // Widen f16 -> f32 once. Flat index (s*card + code)*sub + j.
        let mut codebooks = vec![0f32; cb_values];
        for (i, cbv) in codebooks.iter_mut().enumerate() {
            let off = cb_start + i * 2;
            let bits = u16::from_le_bytes(payload[off..off + 2].try_into().unwrap());
            *cbv = f16::from_bits(bits).to_f32();
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

    pub fn rows(&self) -> usize {
        self.header.rows as usize
    }

    pub fn cols(&self) -> usize {
        self.header.cols as usize
    }

    /// `y = W @ x`, accumulating in f32 strictly left-to-right so the
    /// result is bit-identical to [`pq_matvec`] on the same payload.
    pub fn matvec(&self, x: &[f32]) -> Result<Vec<f32>> {
        let h = &self.header;
        if x.len() != h.cols as usize {
            return Err(Error::Gravity(format!(
                "pq matvec: x.len() {} != cols {}",
                x.len(),
                h.cols
            )));
        }
        let d = h.d as usize;
        let s = h.s as usize;
        let sub = h.sub as usize;
        let card = h.card as usize;
        let rows = h.rows as usize;
        let nchunk = h.nchunk as usize;

        // xc[c][j] = x[c*D + j] (rotate==0, checked at decode, so no
        // rotation applied).
        // y[r] = sum_s sum_c sum_j codebook[s][index(r,c,s)][j] * xc[c][s*sub+j]
        let mut y = vec![0f32; rows];
        for sub_idx in 0..s {
            let cb_base = sub_idx * card * sub;
            let x_off = sub_idx * sub;
            for r in 0..rows {
                for c in 0..nchunk {
                    let flat = (r * nchunk + c) * s + sub_idx;
                    let code = self.indices[flat] as usize;
                    let cb_row = cb_base + code * sub;
                    let x_base = c * d + x_off;
                    for j in 0..sub {
                        y[r] += self.codebooks[cb_row + j] * x[x_base + j];
                    }
                }
            }
        }
        Ok(y)
    }

    /// Decode a single row of the encoded matrix — `cols` values. Used for
    /// embedding lookup, where materializing the whole `[vocab, hidden]`
    /// matrix to read one row would be absurd. Mirrors
    /// `gravity_llama_reference.py::GravityWeights.row`.
    pub fn row(&self, index: usize) -> Result<Vec<f32>> {
        let h = &self.header;
        let rows = h.rows as usize;
        if index >= rows {
            return Err(Error::Gravity(format!(
                "pq row: index {index} out of range for {rows} rows"
            )));
        }
        let d = h.d as usize;
        let s = h.s as usize;
        let sub = h.sub as usize;
        let card = h.card as usize;
        let nchunk = h.nchunk as usize;

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

/// CPU matvec over a `gravity-pq` payload: `y = W @ x` where `W` is the
/// `[rows, cols]` matrix the payload encodes. `x.len()` must equal
/// `cols`; returns `rows` values. Mirrors `gravity_forge.py::pq_execute`,
/// accumulating in f32.
pub fn pq_matvec(payload: &[u8], x: &[f32]) -> Result<Vec<f32>> {
    PqTensor::from_payload(payload)?.matvec(x)
}

// ---------------------------------------------------------------------
// Architecture-independent weight access.
// ---------------------------------------------------------------------

/// One tensor as a forward pass consumes it.
pub enum Tensor {
    Pq(PqTensor),
    Dense(Vec<f32>),
}

/// Widen a `native.<dtype>` payload to f32. An unknown dtype is an error
/// rather than a reinterpretation of the bytes as something plausible.
pub fn widen_native(codec: &str, blob: &[u8]) -> Result<Vec<f32>> {
    let dtype = codec.split_once('.').map(|(_, d)| d).unwrap_or("");
    let (unit, name) = match dtype {
        "bf16" | "f16" => (2usize, dtype),
        "f32" => (4usize, dtype),
        other => {
            return Err(Error::Gravity(format!(
                "unsupported native tensor dtype {other:?} (codec {codec:?})"
            )))
        }
    };
    if blob.len() % unit != 0 {
        return Err(Error::Gravity(format!(
            "native.{name} payload length {} is not a multiple of {unit}",
            blob.len()
        )));
    }
    Ok(match dtype {
        "bf16" => blob
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        "f16" => blob
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        _ => blob
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    })
}

fn matvec_dense(w: &[f32], x: &[f32], name: &str) -> Result<Vec<f32>> {
    if x.is_empty() || w.len() % x.len() != 0 {
        return Err(Error::Gravity(format!(
            "tensor {name:?}: {} values is not a whole number of {}-wide rows",
            w.len(),
            x.len()
        )));
    }
    Ok(w.chunks_exact(x.len())
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect())
}

fn row_dense(w: &[f32], index: usize, cols: usize, name: &str) -> Result<Vec<f32>> {
    let start = index * cols;
    if start + cols > w.len() {
        return Err(Error::Gravity(format!(
            "tensor {name:?}: row {index} out of range"
        )));
    }
    Ok(w[start..start + cols].to_vec())
}

/// Where a [`GravityWeights`] actually gets its bytes.
enum Source {
    /// Every tensor of a single shard, decoded once at open. Right for an
    /// artifact small enough that "decode everything up front" is cheap —
    /// the Llama and tiny-GLM fixtures are tens to hundreds of MB.
    Eager(HashMap<String, Tensor>),
    /// Every tensor's owning shard *filename* known; no shard opened and no
    /// payload touched until a tensor from it is actually asked for.
    ///
    /// Right for a flagship-scale MoE artifact: only 8 of 256 experts
    /// activate per layer, so eagerly decoding all of them would both do
    /// ~32x the necessary work and — worse — exceed physical memory, since a
    /// `PqTensor`'s u16-widened indices run close to 2x the packed size.
    /// Opening all 282 shards up front would be comparatively cheap (mmap is
    /// virtual address space, not resident pages) but still unnecessary for
    /// a short run that never touches most of them, so shards open lazily
    /// into `open_shards` on first use and stay there for the model's
    /// lifetime. A tensor's payload bytes are read, decoded, used, and
    /// dropped on every call — same as the Python `GravityGlmSource` this
    /// mirrors. `RefCell` rather than `Mutex`: this path has no `Send+Sync`
    /// requirement, unlike the GPU model's resident buffers.
    Lazy {
        shard_dir: std::path::PathBuf,
        tensor_shard: HashMap<String, String>,
        open_shards: std::cell::RefCell<HashMap<String, GravityShard>>,
        verify_hash: bool,
    },
}

/// Every tensor of a `.gravity` model, addressed by name.
///
/// Architecture-independent on purpose: the Llama and GLM adapters differ in
/// what they compute, not in how they reach a weight, and a second loader
/// would be a second place for a codec or a hash check to be forgotten.
pub struct GravityWeights {
    source: Source,
    /// The shard header minus `tensors` — architecture, compression,
    /// integrity, tokenizer, as the writer left them. Identical across every
    /// shard of one model by construction, so shard 0's copy speaks for all.
    pub header: serde_json::Value,
}

impl GravityWeights {
    pub fn open(path: &Path, verify_hash: bool) -> Result<GravityWeights> {
        let shard = GravityShard::open(path)?;
        let names: Vec<String> = shard.tensor_names().map(str::to_string).collect();
        let mut tensors = HashMap::with_capacity(names.len());
        for name in &names {
            let codec = shard
                .descriptor(name)
                .expect("name came from tensor_names")
                .codec
                .clone();
            let blob = shard.read_tensor(name, verify_hash)?;
            let t = if codec == "gravity-pq" {
                Tensor::Pq(PqTensor::from_payload(&blob)?)
            } else if codec.starts_with("native.") {
                Tensor::Dense(widen_native(&codec, &blob)?)
            } else {
                return Err(Error::Gravity(format!(
                    "tensor {name}: unsupported codec {codec:?}"
                )));
            };
            tensors.insert(name.clone(), t);
        }
        Ok(GravityWeights {
            source: Source::Eager(tensors),
            header: shard.extra,
        })
    }

    /// Open a multi-shard model in `dir`, indexing which shard owns which
    /// tensor without opening any shard or decoding any payload yet. See
    /// [`Source::Lazy`] for why this stays lazy.
    ///
    /// Prefers `dir/model.gravity.index.json` — the assembler's own manifest,
    /// written once when it graded this exact directory's coverage complete
    /// against the official tensor count. That manifest carries the
    /// synthesized full architecture (twenty fields) rather than the five a
    /// single shard's own header holds, and reading one JSON file beats
    /// opening all 282 shards just to learn what they contain. Falls back to
    /// scanning shard headers directly for a `model-*.gravity` directory that
    /// was never assembled — self-sufficient, just slower to open.
    pub fn open_dir(dir: &Path, verify_hash: bool) -> Result<GravityWeights> {
        let index_path = dir.join("model.gravity.index.json");
        if index_path.is_file() {
            let manifest: serde_json::Value = serde_json::from_slice(
                &std::fs::read(&index_path)
                    .map_err(|e| Error::Gravity(format!("{}: {e}", index_path.display())))?,
            )
            .map_err(|e| Error::Gravity(format!("{}: {e}", index_path.display())))?;
            let tensor_shard: HashMap<String, String> = manifest
                .get("weight_map")
                .and_then(|v| v.as_object())
                .ok_or_else(|| {
                    Error::Gravity(format!("{}: no weight_map", index_path.display()))
                })?
                .iter()
                .filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string())))
                .collect();
            let header = manifest
                .get("architecture")
                .map(|a| serde_json::json!({"architecture": a}))
                .ok_or_else(|| {
                    Error::Gravity(format!("{}: no architecture block", index_path.display()))
                })?;
            return Ok(GravityWeights {
                source: Source::Lazy {
                    shard_dir: dir.to_path_buf(),
                    tensor_shard,
                    open_shards: std::cell::RefCell::new(HashMap::new()),
                    verify_hash,
                },
                header,
            });
        }

        let mut names: Vec<std::ffi::OsString> = std::fs::read_dir(dir)
            .map_err(|e| Error::Gravity(format!("{}: {e}", dir.display())))?
            .filter_map(|e| e.ok())
            .map(|e| e.file_name())
            .filter(|n| {
                let s = n.to_string_lossy();
                s.starts_with("model-") && s.ends_with(".gravity")
            })
            .collect();
        names.sort();
        if names.is_empty() {
            return Err(Error::Gravity(format!(
                "{}: no model.gravity.index.json and no model-*.gravity shards found",
                dir.display()
            )));
        }
        let mut tensor_shard = HashMap::new();
        let mut header = None;
        for name in &names {
            let filename = name.to_string_lossy().into_owned();
            let shard = GravityShard::open(&dir.join(name))?;
            if header.is_none() {
                header = Some(shard.extra.clone());
            }
            for tname in shard.tensor_names() {
                tensor_shard.insert(tname.to_string(), filename.clone());
            }
        }
        Ok(GravityWeights {
            source: Source::Lazy {
                shard_dir: dir.to_path_buf(),
                tensor_shard,
                open_shards: std::cell::RefCell::new(HashMap::new()),
                verify_hash,
            },
            header: header.expect("names is non-empty"),
        })
    }

    pub fn contains(&self, name: &str) -> bool {
        match &self.source {
            Source::Eager(t) => t.contains_key(name),
            Source::Lazy { tensor_shard, .. } => tensor_shard.contains_key(name),
        }
    }

    /// Open `name`'s owning shard if it is not already open, then run `f`
    /// against it. The shard is inserted into `open_shards` before `f` runs
    /// and stays there — a shard opened for one tensor is likely to be asked
    /// for a sibling tensor later in the same forward pass (an artifact's
    /// tensors are grouped by source shard, and a layer's projections
    /// typically land together), so keeping it open trades a little
    /// resident memory (mmap bookkeeping, not tensor bytes) for not
    /// re-opening the same file repeatedly.
    fn with_lazy_shard<T>(
        shard_dir: &Path,
        tensor_shard: &HashMap<String, String>,
        open_shards: &std::cell::RefCell<HashMap<String, GravityShard>>,
        name: &str,
        f: impl FnOnce(&GravityShard) -> Result<T>,
    ) -> Result<T> {
        let filename = tensor_shard
            .get(name)
            .ok_or_else(|| Error::Gravity(format!("artifact has no tensor {name:?}")))?;
        if !open_shards.borrow().contains_key(filename) {
            let shard = GravityShard::open(&shard_dir.join(filename))?;
            open_shards.borrow_mut().insert(filename.clone(), shard);
        }
        f(open_shards
            .borrow()
            .get(filename)
            .expect("just opened or already present"))
    }

    /// A natively-carried 1D tensor: norm weights, biases, router
    /// corrections. Packing these is refused upstream, so finding one packed
    /// here means the artifact disagrees with the runtime about what it is.
    /// Owned rather than borrowed so the same signature covers both a
    /// pre-decoded eager tensor and one decoded fresh on this call.
    pub fn dense(&self, name: &str) -> Result<Vec<f32>> {
        match &self.source {
            Source::Eager(tensors) => match tensors.get(name) {
                Some(Tensor::Dense(v)) => Ok(v.clone()),
                Some(Tensor::Pq(_)) => Err(Error::Gravity(format!(
                    "tensor {name:?} is packed; expected a natively-carried dense tensor"
                ))),
                None => Err(Error::Gravity(format!("artifact has no tensor {name:?}"))),
            },
            Source::Lazy {
                shard_dir,
                tensor_shard,
                open_shards,
                verify_hash,
            } => Self::with_lazy_shard(shard_dir, tensor_shard, open_shards, name, |shard| {
                let codec = &shard
                    .descriptor(name)
                    .expect("name came from this shard's own index")
                    .codec;
                if !codec.starts_with("native.") {
                    return Err(Error::Gravity(format!(
                        "tensor {name:?} is packed; expected a natively-carried dense tensor"
                    )));
                }
                widen_native(codec, &shard.read_tensor(name, *verify_hash)?)
            }),
        }
    }

    /// `y = W @ x` for a 2D weight, whichever codec carries it.
    pub fn matvec(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        match &self.source {
            Source::Eager(tensors) => match tensors.get(name) {
                Some(Tensor::Pq(t)) => t.matvec(x),
                Some(Tensor::Dense(w)) => matvec_dense(w, x, name),
                None => Err(Error::Gravity(format!("artifact has no tensor {name:?}"))),
            },
            Source::Lazy {
                shard_dir,
                tensor_shard,
                open_shards,
                verify_hash,
            } => Self::with_lazy_shard(shard_dir, tensor_shard, open_shards, name, |shard| {
                let codec = shard
                    .descriptor(name)
                    .expect("name came from this shard's own index")
                    .codec
                    .clone();
                let blob = shard.read_tensor(name, *verify_hash)?;
                if codec == "gravity-pq" {
                    PqTensor::from_payload(&blob)?.matvec(x)
                } else if codec.starts_with("native.") {
                    matvec_dense(&widen_native(&codec, &blob)?, x, name)
                } else {
                    Err(Error::Gravity(format!(
                        "tensor {name}: unsupported codec {codec:?}"
                    )))
                }
            }),
        }
    }

    /// One row of a 2D weight — the embedding-lookup path. `cols` is needed
    /// only for the dense case, where the payload carries no shape.
    pub fn row(&self, name: &str, index_: usize, cols: usize) -> Result<Vec<f32>> {
        match &self.source {
            Source::Eager(tensors) => match tensors.get(name) {
                Some(Tensor::Pq(t)) => t.row(index_),
                Some(Tensor::Dense(w)) => row_dense(w, index_, cols, name),
                None => Err(Error::Gravity(format!("artifact has no tensor {name:?}"))),
            },
            Source::Lazy {
                shard_dir,
                tensor_shard,
                open_shards,
                verify_hash,
            } => Self::with_lazy_shard(shard_dir, tensor_shard, open_shards, name, |shard| {
                let codec = shard
                    .descriptor(name)
                    .expect("name came from this shard's own index")
                    .codec
                    .clone();
                let blob = shard.read_tensor(name, *verify_hash)?;
                if codec == "gravity-pq" {
                    pq_row(&blob, index_)
                } else if codec.starts_with("native.") {
                    row_dense(&widen_native(&codec, &blob)?, index_, cols, name)
                } else {
                    Err(Error::Gravity(format!(
                        "tensor {name}: unsupported codec {codec:?}"
                    )))
                }
            }),
        }
    }
}

/// Metal `GravityPQParams` mirror (`shaders/gravity_pq.metal`): eight
/// `uint`s in declaration order, 32 bytes total, `#[repr(C)]` so a raw
/// pointer cast is a valid `set_bytes` payload.
#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GravityPqParams {
    dim: u32,
    subspaces: u32,
    sub: u32,
    card: u32,
    rows: u32,
    cols: u32,
    nchunk: u32,
    bits: u32,
}

/// GPU counterpart of [`pq_matvec`]: dispatches `gravity_pq_matvec`
/// (`shaders/gravity_pq.metal`), one SIMD group per output row, 8 SIMD
/// groups (256 threads) per threadgroup. Same shape/rotate contract as
/// the CPU path; results differ only in the last bit or two because the
/// kernel's per-row reduction (`fma` chain + `simd_sum`) reassociates
/// sums the CPU path performs strictly left-to-right.
#[cfg(target_os = "macos")]
pub fn pq_matvec_metal(
    ctx: &crate::metal::MetalContext,
    payload: &[u8],
    x: &[f32],
) -> Result<Vec<f32>> {
    let h = parse_pq_header(payload)?;
    if x.len() != h.cols as usize {
        return Err(Error::Gravity(format!(
            "pq_matvec_metal: x.len() {} != cols {}",
            x.len(),
            h.cols
        )));
    }
    if h.rotate != 0 {
        return Err(Error::Gravity(
            "rotated gravity-pq artifacts (rotate=1) are not yet supported".into(),
        ));
    }

    let card = h.card as usize;
    let sub = h.sub as usize;
    let rows = h.rows as usize;
    let nchunk = h.nchunk as usize;

    // Codebooks: `n_codebooks` back to back, each `card * sub` f16 values,
    // uploaded byte-for-byte -- the kernel reads `half` directly, no host
    // widening.
    let cb_values = h.n_codebooks as usize * card * sub;
    let cb_bytes = cb_values
        .checked_mul(2)
        .ok_or_else(|| Error::Gravity("gravity-pq codebook size overflow".into()))?;
    let cb_start = PQ_HEADER_LEN;
    let cb_end = cb_start
        .checked_add(cb_bytes)
        .ok_or_else(|| Error::Gravity("gravity-pq codebook size overflow".into()))?;
    if payload.len() < cb_end {
        return Err(Error::Gravity(format!(
            "gravity-pq payload too short for codebooks: have {} bytes, need {cb_end}",
            payload.len()
        )));
    }

    // Index bitstream: same byte span `unpack_bits` would consume on the
    // CPU path, plus 4 zero bytes of tail padding so the kernel's whole-word
    // read at the last index's byte offset never runs past the buffer.
    let idx_count = rows * nchunk * h.s as usize;
    let need_bytes = (idx_count as u64 * h.bits as u64).div_ceil(8) as usize;
    if payload.len() < cb_end + need_bytes {
        return Err(Error::Gravity(format!(
            "gravity-pq index bitstream too short: have {} bytes, need {need_bytes}",
            payload.len() - cb_end
        )));
    }
    let mut codes = payload[cb_end..cb_end + need_bytes].to_vec();
    codes.extend_from_slice(&[0u8; 4]);

    let codebooks_buf = ctx.new_buffer_with_bytes(&payload[cb_start..cb_end]);
    let codes_buf = ctx.new_buffer_with_bytes(&codes);
    let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
    let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

    let params = GravityPqParams {
        dim: h.d as u32,
        subspaces: h.s as u32,
        sub: h.sub as u32,
        card: h.card as u32,
        rows: h.rows,
        cols: h.cols,
        nchunk: h.nchunk,
        bits: h.bits as u32,
    };

    // One SIMD group (32 lanes) per output row, 8 SIMD groups (256 threads)
    // per threadgroup; the kernel guards `row >= rows` for the boundary
    // threadgroup. `dispatch_threads` takes a total-thread grid, so scale
    // the threadgroup count back up by the threadgroup size.
    const TG: u32 = 256;
    let n_tg = h.rows.div_ceil(8);
    ctx.dispatch_threads("gravity_pq_matvec", (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
        enc.set_buffer(0, Some(&codebooks_buf), 0);
        enc.set_buffer(1, Some(&codes_buf), 0);
        enc.set_buffer(2, Some(&x_buf), 0);
        enc.set_buffer(3, Some(&y_buf), 0);
        enc.set_bytes(
            4,
            std::mem::size_of::<GravityPqParams>() as u64,
            &params as *const GravityPqParams as *const _,
        );
    })?;

    let y_ptr = y_buf.contents() as *const f32;
    Ok(unsafe { std::slice::from_raw_parts(y_ptr, rows) }.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `[0,1,126,127,64,3,2,1]` at 7 bits each, MSB-first, packed
    /// numpy-`packbits`-style (56 bits = exactly 7 bytes, no trailing
    /// padding needed):
    ///
    /// ```text
    /// 0000000 0000001 1111110 1111111 1000000 0000011 0000010 0000001
    /// -> 00000000 00000111 11110111 11111000 00000000 11000001 00000001
    /// -> 0x00     0x07     0xF7     0xF8     0x00     0xC1     0x01
    /// ```
    #[test]
    fn unpack_bits_matches_hand_packed_7bit_values() {
        let packed: [u8; 7] = [0x00, 0x07, 0xF7, 0xF8, 0x00, 0xC1, 0x01];
        let got = unpack_bits(&packed, 8, 7).expect("unpack");
        assert_eq!(got, vec![0, 1, 126, 127, 64, 3, 2, 1]);
    }

    /// `index_at` must agree with the sequential walk for every position,
    /// including the ones straddling a byte boundary — it is the only
    /// reader on the embedding path, where nothing else would catch a
    /// one-bit skew.
    #[test]
    fn index_at_matches_sequential_unpack() {
        let packed: [u8; 7] = [0x00, 0x07, 0xF7, 0xF8, 0x00, 0xC1, 0x01];
        let seq = unpack_bits(&packed, 8, 7).expect("unpack");
        for (i, &want) in seq.iter().enumerate() {
            assert_eq!(index_at(&packed, i, 7), want, "index {i}");
        }
    }

    #[test]
    fn unpack_bits_rejects_short_stream() {
        let packed: [u8; 6] = [0x00, 0x07, 0xF7, 0xF8, 0x00, 0xC1];
        assert!(unpack_bits(&packed, 8, 7).is_err());
    }
}

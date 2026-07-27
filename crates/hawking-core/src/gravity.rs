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

use std::collections::{HashMap, HashSet};
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
        use crate::cost_ledger::{self, Bucket};

        let d = {
            let _lookup = cost_ledger::Scope::new(Bucket::ContainerLookup);
            self.descriptor(name)
                .ok_or_else(|| Error::Gravity(format!("no such tensor {name:?}")))?
        };
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
        let payload = {
            let _lookup = cost_ledger::Scope::new(Bucket::ContainerLookup);
            &self.mmap[start as usize..end as usize]
        };
        if verify_hash {
            let _verify = cost_ledger::Scope::new(Bucket::ArtifactVerificationAndSha);
            cost_ledger::record_sha_verification();
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
        // Host-side copy out of the mmap. Counts as a hot-loop allocation when
        // the cost ledger is recording a decode token.
        cost_ledger::record_allocation(payload.len() as u64);
        Ok(payload.to_vec())
    }

    /// Copy only the first `prefix_len` bytes of one tensor payload.
    ///
    /// This deliberately does **not** claim SHA-256 verification: a digest
    /// covers the complete payload, so a prefix cannot authenticate it.
    /// Header-only admission may use this to reject unsupported codecs before
    /// allocating runtime state; execution must still reach [`read_tensor`]
    /// and its ordinary full-payload verification before using the weight.
    fn read_tensor_prefix_unverified(&self, name: &str, prefix_len: usize) -> Result<Vec<u8>> {
        use crate::cost_ledger::{self, Bucket};

        let d = {
            let _lookup = cost_ledger::Scope::new(Bucket::ContainerLookup);
            self.descriptor(name)
                .ok_or_else(|| Error::Gravity(format!("no such tensor {name:?}")))?
        };
        if prefix_len as u64 > d.bytes {
            return Err(Error::Gravity(format!(
                "tensor {name}: payload {} bytes is shorter than requested {prefix_len}-byte prefix",
                d.bytes
            )));
        }
        let start = self
            .body_offset
            .checked_add(d.offset)
            .ok_or_else(|| Error::Gravity(format!("tensor {name}: offset overflow")))?;
        let tensor_end = start
            .checked_add(d.bytes)
            .ok_or_else(|| Error::Gravity(format!("tensor {name}: end overflow")))?;
        if tensor_end > self.mmap.len() as u64 {
            return Err(Error::Gravity(format!(
                "tensor {name}: end {tensor_end} past file length {}",
                self.mmap.len()
            )));
        }
        let prefix_end = start
            .checked_add(prefix_len as u64)
            .ok_or_else(|| Error::Gravity(format!("tensor {name}: prefix end overflow")))?;
        let prefix = {
            let _lookup = cost_ledger::Scope::new(Bucket::ContainerLookup);
            &self.mmap[start as usize..prefix_end as usize]
        };
        cost_ledger::record_allocation(prefix.len() as u64);
        Ok(prefix.to_vec())
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

    /// Numeric Parity V2.1 authority for this compact matrix.
    ///
    /// Codebook values are stored as f16 and therefore widen exactly to
    /// f32/f64. Activations are f32 and promote exactly to f64. The products
    /// and the left-to-right sum are then evaluated in f64, so neither the
    /// host f32 reduction nor any Metal reduction order is treated as the
    /// oracle.
    pub fn matvec_f64_authority(&self, x: &[f32]) -> Result<Vec<f64>> {
        let h = &self.header;
        if x.len() != h.cols as usize {
            return Err(Error::Gravity(format!(
                "pq f64 authority: x.len() {} != cols {}",
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
        let mut y = vec![0.0f64; rows];
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
                        y[r] += (self.codebooks[cb_row + j] as f64)
                            * (x[x_base + j] as f64);
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

/// FP64 authority for a compact `gravity-pq` matvec under Numeric Parity
/// V2.1. See [`PqTensor::matvec_f64_authority`].
pub fn pq_matvec_f64_authority(payload: &[u8], x: &[f32]) -> Result<Vec<f64>> {
    PqTensor::from_payload(payload)?.matvec_f64_authority(x)
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

pub(crate) fn matvec_dense(w: &[f32], x: &[f32], name: &str) -> Result<Vec<f32>> {
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

/// Host oracle for device `gemv_native_bf16_seq`: widen little-endian bf16
/// the same way as [`widen_native`], then left-to-right f32 Σ per row via
/// [`matvec_dense`]. Public so parity tests (and the GPU module) share one
/// definition.
pub fn matvec_bf16_host(weight_le: &[u8], cols: usize, x: &[f32]) -> Result<Vec<f32>> {
    if x.len() != cols {
        return Err(Error::Gravity(format!(
            "matvec_bf16_host: x.len() {} != cols {cols}",
            x.len()
        )));
    }
    if cols == 0 || weight_le.len() % (cols * 2) != 0 {
        return Err(Error::Gravity(format!(
            "matvec_bf16_host: payload {} B is not a whole number of {cols}-wide bf16 rows",
            weight_le.len()
        )));
    }
    let w = widen_native("native.bf16", weight_le)?;
    matvec_dense(&w, x, "lm_head.weight")
}

/// Additive, default-off accumulation candidates for native-BF16 GEMV.
///
/// `Sequential` is the existing left-to-right f32 contract. The other
/// variants are exposed only to parity tests and explicit microbenchmarks;
/// no runtime selection consults this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeBf16Accumulation {
    Sequential,
    Neumaier,
    /// Neumaier summation plus the residual of every rounded f32 product,
    /// recovered with an explicit fused multiply-add.
    NeumaierCompensatedProduct,
}

impl NativeBf16Accumulation {
    pub const ALL: [Self; 3] = [
        Self::Sequential,
        Self::Neumaier,
        Self::NeumaierCompensatedProduct,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Neumaier => "neumaier",
            Self::NeumaierCompensatedProduct => "neumaier_compensated_product",
        }
    }

    #[cfg(target_os = "macos")]
    pub const fn metal_kernel(self) -> &'static str {
        match self {
            Self::Sequential => "gemv_native_bf16_seq",
            Self::Neumaier => "gemv_native_bf16_neumaier",
            Self::NeumaierCompensatedProduct => {
                "gemv_native_bf16_neumaier_compensated_product"
            }
        }
    }
}

/// Matching f32 host comparator for the additive native-BF16 accumulation
/// candidates. Bounds and authority remain external: this function changes
/// arithmetic only, never parity policy.
pub fn matvec_bf16_host_accumulation(
    weight_le: &[u8],
    cols: usize,
    x: &[f32],
    accumulation: NativeBf16Accumulation,
) -> Result<Vec<f32>> {
    if accumulation == NativeBf16Accumulation::Sequential {
        return matvec_bf16_host(weight_le, cols, x);
    }
    if x.len() != cols {
        return Err(Error::Gravity(format!(
            "matvec_bf16_host_accumulation: x.len() {} != cols {cols}",
            x.len()
        )));
    }
    if cols == 0 || weight_le.len() % (cols * 2) != 0 {
        return Err(Error::Gravity(format!(
            "matvec_bf16_host_accumulation: payload {} B is not a whole number of \
             {cols}-wide bf16 rows",
            weight_le.len()
        )));
    }

    let weights = widen_native("native.bf16", weight_le)?;
    Ok(weights
        .chunks_exact(cols)
        .map(|row| {
            let mut sum = 0.0f32;
            let mut correction = 0.0f32;
            for (&weight, &activation) in row.iter().zip(x) {
                let product = weight * activation;
                let product_residual =
                    if accumulation == NativeBf16Accumulation::NeumaierCompensatedProduct {
                        weight.mul_add(activation, -product)
                    } else {
                        0.0
                    };
                let next = sum + product;
                let addition_residual = if sum.abs() >= product.abs() {
                    let delta = sum - next;
                    delta + product
                } else {
                    let delta = product - next;
                    delta + sum
                };
                correction += addition_residual;
                correction += product_residual;
                sum = next;
            }
            sum + correction
        })
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

/// Default byte budget for the Lazy-path native dense/row decoded memo.
///
/// The campaign carries a few hundred `native.*` tensors (norms, biases,
/// router corrections) — about 0.1% of flagship weights. 256 MiB is a
/// generous ceiling for those widened f32 vectors while still bounding
/// residency the way the GPU weight cache does.
pub const DEFAULT_NATIVE_DENSE_MEMO_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

/// Snapshot of the Lazy-path native dense/row memo. Surfaced so a long run
/// has an explicit residency number rather than an unbounded HashMap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeDenseMemoStats {
    pub budget_bytes: u64,
    pub resident_bytes: u64,
    pub high_water_bytes: u64,
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    /// Times a payload SHA-256 was actually computed on the dense/row path.
    /// Integrity still requires **at least one** check per tensor per process
    /// when `verify_hash` is on; this counter proves repeats are not free.
    pub verifications: u64,
    /// Distinct tensors that have been integrity-checked (or admitted after
    /// a checked load) on the dense/row path.
    pub verified_tensors: usize,
    pub evictions: u64,
}

struct MemoEntry {
    value: Vec<f32>,
    bytes: u64,
    last_tick: u64,
}

/// Byte-budgeted memo of decoded `native.*` vectors plus a one-shot
/// verification record for tensors too large to keep decoded (e.g. an
/// embedding table reached via [`GravityWeights::row`]).
///
/// Decode+hash of the same handful of norm/bias tensors used to dominate
/// warm decode when hashing was on: 78 layers × several tensors × every
/// token re-read, re-widened, and re-hashed the same bytes. This memo keeps
/// the widened `Vec<f32>` after the first access so subsequent `dense` /
/// `row` calls are a clone from RAM. Large tensors skip the decoded cache
/// (holding a full embedding table would defeat the budget) but still mark
/// verification done so SHA-256 runs at most once per process.
struct NativeDenseMemo {
    decoded: HashMap<String, MemoEntry>,
    /// Names whose payload has been integrity-checked (or loaded under
    /// `verify_hash == false`). Large row() targets land here without a
    /// decoded entry.
    verified: HashSet<String>,
    budget_bytes: u64,
    resident_bytes: u64,
    high_water_bytes: u64,
    clock: u64,
    hits: u64,
    misses: u64,
    verifications: u64,
    evictions: u64,
}

impl NativeDenseMemo {
    fn new(budget_bytes: u64) -> Self {
        Self {
            decoded: HashMap::new(),
            verified: HashSet::new(),
            budget_bytes,
            resident_bytes: 0,
            high_water_bytes: 0,
            clock: 0,
            hits: 0,
            misses: 0,
            verifications: 0,
            evictions: 0,
        }
    }

    fn stats(&self) -> NativeDenseMemoStats {
        NativeDenseMemoStats {
            budget_bytes: self.budget_bytes,
            resident_bytes: self.resident_bytes,
            high_water_bytes: self.high_water_bytes,
            entries: self.decoded.len(),
            hits: self.hits,
            misses: self.misses,
            verifications: self.verifications,
            verified_tensors: self.verified.len(),
            evictions: self.evictions,
        }
    }

    fn is_verified(&self, name: &str) -> bool {
        self.verified.contains(name)
    }

    fn has_decoded(&self, name: &str) -> bool {
        self.decoded.contains_key(name)
    }

    /// Memo hit: clone the decoded vector and refresh LRU. `None` on miss.
    fn take_decoded(&mut self, name: &str) -> Option<Vec<f32>> {
        if let Some(e) = self.decoded.get_mut(name) {
            self.clock = self.clock.saturating_add(1);
            e.last_tick = self.clock;
            self.hits = self.hits.saturating_add(1);
            Some(e.value.clone())
        } else {
            None
        }
    }

    fn note_miss(&mut self) {
        self.misses = self.misses.saturating_add(1);
    }

    fn record_verification(&mut self, name: &str) {
        self.verified.insert(name.to_string());
        self.verifications = self.verifications.saturating_add(1);
    }

    /// Remember that `name` is safe to re-read without hashing (either we
    /// just verified it, or `verify_hash` is off so integrity was waived).
    fn mark_verified_without_hash(&mut self, name: &str) {
        self.verified.insert(name.to_string());
    }

    /// Admit a decoded native vector under the byte budget. Oversized
    /// tensors (or a full cache that cannot free room) skip residency but
    /// remain in `verified` so hashing is still one-shot.
    fn admit_decoded(&mut self, name: &str, value: Vec<f32>) {
        if self.decoded.contains_key(name) {
            self.clock = self.clock.saturating_add(1);
            if let Some(e) = self.decoded.get_mut(name) {
                e.last_tick = self.clock;
            }
            return;
        }
        let bytes = (value.len() as u64).saturating_mul(4);
        self.verified.insert(name.to_string());
        if bytes > self.budget_bytes {
            return;
        }
        while self.resident_bytes.saturating_add(bytes) > self.budget_bytes {
            if !self.evict_one() {
                return;
            }
        }
        self.clock = self.clock.saturating_add(1);
        self.decoded.insert(
            name.to_string(),
            MemoEntry {
                value,
                bytes,
                last_tick: self.clock,
            },
        );
        self.resident_bytes = self.resident_bytes.saturating_add(bytes);
        if self.resident_bytes > self.high_water_bytes {
            self.high_water_bytes = self.resident_bytes;
        }
    }

    fn evict_one(&mut self) -> bool {
        let victim = self
            .decoded
            .iter()
            .min_by_key(|(_, e)| e.last_tick)
            .map(|(k, _)| k.clone());
        let Some(name) = victim else {
            return false;
        };
        if let Some(e) = self.decoded.remove(&name) {
            self.resident_bytes = self.resident_bytes.saturating_sub(e.bytes);
            self.evictions = self.evictions.saturating_add(1);
            true
        } else {
            false
        }
    }
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
    /// lifetime.
    ///
    /// `native.*` tensors reached via [`GravityWeights::dense`] / [`row`]
    /// are memoized after first decode (byte-budgeted); integrity hashing
    /// still runs at least once per tensor when `verify_hash` is on, but
    /// not on every subsequent access. Packed `matvec` payloads remain
    /// decode-on-call (or GPU-cached by the adapter).
    ///
    /// `Mutex` rather than `RefCell`: an `Engine` implementor must be
    /// `Send + Sync` (the GPU weight cache wraps a `GravityWeights` for its
    /// `dense`/`row` delegation), and an uncontended lock on a
    /// single-threaded CPU forward costs nothing worth avoiding.
    Lazy {
        shard_dir: std::path::PathBuf,
        tensor_shard: HashMap<String, String>,
        open_shards: std::sync::Mutex<HashMap<String, GravityShard>>,
        verify_hash: bool,
        dense_memo: std::sync::Mutex<NativeDenseMemo>,
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
                    open_shards: std::sync::Mutex::new(HashMap::new()),
                    verify_hash,
                    dense_memo: std::sync::Mutex::new(NativeDenseMemo::new(
                        DEFAULT_NATIVE_DENSE_MEMO_BUDGET_BYTES,
                    )),
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
                open_shards: std::sync::Mutex::new(HashMap::new()),
                verify_hash,
                dense_memo: std::sync::Mutex::new(NativeDenseMemo::new(
                    DEFAULT_NATIVE_DENSE_MEMO_BUDGET_BYTES,
                )),
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

    /// Every tensor name this model declares. Eager mode only — used to
    /// enumerate a fixture's tensors for testing; the flagship (Lazy) is
    /// walked by the forward pass name-by-name, never enumerated wholesale.
    pub fn tensor_names(&self) -> Vec<String> {
        match &self.source {
            Source::Eager(t) => t.keys().cloned().collect(),
            Source::Lazy { tensor_shard, .. } => tensor_shard.keys().cloned().collect(),
        }
    }

    /// Residency of the Lazy-path native dense/row memo. Eager mode returns
    /// a zeroed snapshot (everything was decoded at open; there is no memo).
    pub fn dense_memo_stats(&self) -> NativeDenseMemoStats {
        match &self.source {
            Source::Eager(_) => NativeDenseMemoStats {
                budget_bytes: 0,
                resident_bytes: 0,
                high_water_bytes: 0,
                entries: 0,
                hits: 0,
                misses: 0,
                verifications: 0,
                verified_tensors: 0,
                evictions: 0,
            },
            Source::Lazy { dense_memo, .. } => dense_memo
                .lock()
                .expect("gravity dense-memo mutex")
                .stats(),
        }
    }

    /// Raw, undecoded payload bytes plus codec string — what a GPU cache
    /// wants to upload verbatim, as opposed to [`dense`]/[`matvec`]/[`row`]
    /// which decode for CPU execution. Lazy mode only: the flagship is the
    /// only artifact large enough to need a GPU-resident cache in the first
    /// place, and Eager mode never retains raw bytes past decode.
    pub fn raw_payload(&self, name: &str) -> Result<(String, Vec<u8>)> {
        let (codec, blob, _shape) = self.raw_payload_with_shape(name)?;
        Ok((codec, blob))
    }

    /// Like [`raw_payload`], but also returns the descriptor shape so a
    /// device-resident dense path (e.g. `lm_head.weight` as `native.bf16`)
    /// can size the GEMV without decoding.
    pub fn raw_payload_with_shape(&self, name: &str) -> Result<(String, Vec<u8>, Vec<u64>)> {
        match &self.source {
            Source::Eager(_) => Err(Error::Gravity(
                "raw_payload: not available in Eager mode (decoded at open, raw bytes discarded)"
                    .into(),
            )),
            Source::Lazy {
                shard_dir,
                tensor_shard,
                open_shards,
                verify_hash,
                ..
            } => Self::with_lazy_shard(shard_dir, tensor_shard, open_shards, name, |shard| {
                let d = shard
                    .descriptor(name)
                    .expect("name came from this shard's own index");
                let codec = d.codec.clone();
                let shape = d.shape.clone();
                let blob = shard.read_tensor(name, *verify_hash)?;
                Ok((codec, blob, shape))
            }),
        }
    }

    /// Read just a `gravity-pq` tensor's fixed header plus descriptor shape.
    ///
    /// Lazy mode copies 64 payload bytes and does not hash the remaining
    /// payload. This is an admission-only view: callers can reject an
    /// incompatible artifact before allocating device/session state, while
    /// the later ordinary payload load still performs the configured full
    /// SHA-256 verification before execution. Eager mode returns the header
    /// already decoded during [`open`](Self::open).
    pub fn pq_header_prefix_unverified_with_shape(
        &self,
        name: &str,
    ) -> Result<(PqHeader, Vec<u64>)> {
        match &self.source {
            Source::Eager(tensors) => match tensors.get(name) {
                Some(Tensor::Pq(tensor)) => Ok((
                    tensor.header,
                    vec![tensor.header.rows as u64, tensor.header.cols as u64],
                )),
                Some(Tensor::Dense(_)) => Err(Error::Gravity(format!(
                    "tensor {name}: compact admission requires gravity-pq, found native tensor"
                ))),
                None => Err(Error::Gravity(format!("artifact has no tensor {name:?}"))),
            },
            Source::Lazy {
                shard_dir,
                tensor_shard,
                open_shards,
                ..
            } => Self::with_lazy_shard(shard_dir, tensor_shard, open_shards, name, |shard| {
                let d = shard
                    .descriptor(name)
                    .expect("name came from this shard's own index");
                if d.codec != "gravity-pq" {
                    return Err(Error::Gravity(format!(
                        "tensor {name}: compact admission requires gravity-pq, found {:?}",
                        d.codec
                    )));
                }
                let prefix = shard.read_tensor_prefix_unverified(name, PQ_HEADER_LEN)?;
                Ok((parse_pq_header(&prefix)?, d.shape.clone()))
            }),
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
        open_shards: &std::sync::Mutex<HashMap<String, GravityShard>>,
        name: &str,
        f: impl FnOnce(&GravityShard) -> Result<T>,
    ) -> Result<T> {
        let filename = tensor_shard
            .get(name)
            .ok_or_else(|| Error::Gravity(format!("artifact has no tensor {name:?}")))?;
        if !open_shards
            .lock()
            .expect("gravity lazy-shard mutex")
            .contains_key(filename)
        {
            let shard = GravityShard::open(&shard_dir.join(filename))?;
            open_shards
                .lock()
                .expect("gravity lazy-shard mutex")
                .insert(filename.clone(), shard);
        }
        f(open_shards
            .lock()
            .expect("gravity lazy-shard mutex")
            .get(filename)
            .expect("just opened or already present"))
    }

    /// A natively-carried 1D tensor: norm weights, biases, router
    /// corrections. Packing these is refused upstream, so finding one packed
    /// here means the artifact disagrees with the runtime about what it is.
    /// Owned rather than borrowed so the same signature covers both a
    /// pre-decoded eager tensor and one decoded fresh on this call.
    ///
    /// Lazy mode memoizes the widened vector after the first access (under
    /// a byte budget). Integrity hashing still runs on the first load when
    /// `verify_hash` is on, then is skipped for the rest of the process.
    pub fn dense(&self, name: &str) -> Result<Vec<f32>> {
        use crate::cost_ledger::{self, Bucket};
        cost_ledger::record_dense_call();
        match &self.source {
            Source::Eager(tensors) => match tensors.get(name) {
                Some(Tensor::Dense(v)) => {
                    cost_ledger::record_allocation((v.len() * 4) as u64);
                    Ok(v.clone())
                }
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
                dense_memo,
            } => {
                {
                    let mut memo = dense_memo.lock().expect("gravity dense-memo mutex");
                    if let Some(v) = memo.take_decoded(name) {
                        return Ok(v);
                    }
                    memo.note_miss();
                }
                let need_verify = {
                    let memo = dense_memo.lock().expect("gravity dense-memo mutex");
                    *verify_hash && !memo.is_verified(name)
                };
                let (codec, blob) =
                    Self::with_lazy_shard(shard_dir, tensor_shard, open_shards, name, |shard| {
                        let codec = shard
                            .descriptor(name)
                            .expect("name came from this shard's own index")
                            .codec
                            .clone();
                        if !codec.starts_with("native.") {
                            return Err(Error::Gravity(format!(
                                "tensor {name:?} is packed; expected a natively-carried dense tensor"
                            )));
                        }
                        let blob = shard.read_tensor(name, need_verify)?;
                        Ok((codec, blob))
                    })?;
                let v = {
                    let _decode = cost_ledger::Scope::new(Bucket::PackedIndexDecode);
                    widen_native(&codec, &blob)?
                };
                {
                    let mut memo = dense_memo.lock().expect("gravity dense-memo mutex");
                    if need_verify {
                        memo.record_verification(name);
                    } else if !*verify_hash {
                        memo.mark_verified_without_hash(name);
                    }
                    // Re-check: another thread may have admitted since our miss.
                    if let Some(cached) = memo.take_decoded(name) {
                        return Ok(cached);
                    }
                    memo.admit_decoded(name, v.clone());
                }
                Ok(v)
            }
        }
    }

    /// `y = W @ x` for a 2D weight, whichever codec carries it.
    pub fn matvec(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        use crate::cost_ledger::{self, Bucket};
        cost_ledger::record_matvec_call();
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
                ..
            } => Self::with_lazy_shard(shard_dir, tensor_shard, open_shards, name, |shard| {
                let codec = shard
                    .descriptor(name)
                    .expect("name came from this shard's own index")
                    .codec
                    .clone();
                let blob = shard.read_tensor(name, *verify_hash)?;
                if codec == "gravity-pq" {
                    let t = {
                        let _decode = cost_ledger::Scope::new(Bucket::PackedIndexDecode);
                        PqTensor::from_payload(&blob)?
                    };
                    // Exact payload extent (descriptor.bytes); not a page, not
                    // a whole shard. mmap may still fault full pages into RSS
                    // — that shows up in page_faults_*, not here.
                    cost_ledger::record_active_bytes_for(name, blob.len() as u64);
                    t.matvec(x)
                } else if codec.starts_with("native.") {
                    let w = {
                        let _decode = cost_ledger::Scope::new(Bucket::PackedIndexDecode);
                        widen_native(&codec, &blob)?
                    };
                    cost_ledger::record_active_bytes_for(name, (w.len() * 4) as u64);
                    matvec_dense(&w, x, name)
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
    ///
    /// Lazy `native.*` rows share the dense memo when the full tensor fits
    /// the budget (same decoded vector `dense` would keep). Oversized
    /// tensors — embedding tables — only memoize verification: the full
    /// table is not retained, but SHA-256 still runs at most once per
    /// process. Packed (`gravity-pq`) rows verify once the same way, then
    /// re-read without re-hashing; row-level decoded caching would be wrong
    /// for a vocab-scale table.
    pub fn row(&self, name: &str, index_: usize, cols: usize) -> Result<Vec<f32>> {
        use crate::cost_ledger::{self, Bucket};
        cost_ledger::record_row_call();
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
                dense_memo,
            } => {
                {
                    let mut memo = dense_memo.lock().expect("gravity dense-memo mutex");
                    if let Some(w) = memo.take_decoded(name) {
                        return row_dense(&w, index_, cols, name);
                    }
                }
                let need_verify = {
                    let memo = dense_memo.lock().expect("gravity dense-memo mutex");
                    *verify_hash && !memo.is_verified(name)
                };
                let (codec, blob) =
                    Self::with_lazy_shard(shard_dir, tensor_shard, open_shards, name, |shard| {
                        let codec = shard
                            .descriptor(name)
                            .expect("name came from this shard's own index")
                            .codec
                            .clone();
                        let blob = shard.read_tensor(name, need_verify)?;
                        Ok((codec, blob))
                    })?;
                {
                    let mut memo = dense_memo.lock().expect("gravity dense-memo mutex");
                    if need_verify {
                        memo.record_verification(name);
                    } else if !*verify_hash {
                        memo.mark_verified_without_hash(name);
                    }
                }
                if codec == "gravity-pq" {
                    // Packed embeddings: verify-once only. Caching every row
                    // of a vocab-scale table would blow the budget; caching
                    // the whole decoded matrix is worse.
                    let _decode = cost_ledger::Scope::new(Bucket::PackedIndexDecode);
                    pq_row(&blob, index_)
                } else if codec.starts_with("native.") {
                    let w = {
                        let _decode = cost_ledger::Scope::new(Bucket::PackedIndexDecode);
                        widen_native(&codec, &blob)?
                    };
                    let row = row_dense(&w, index_, cols, name)?;
                    {
                        let mut memo = dense_memo.lock().expect("gravity dense-memo mutex");
                        if !memo.has_decoded(name) {
                            memo.note_miss();
                            // Small native matrices share the dense memo;
                            // oversize (embedding-scale) tensors skip residency
                            // but stay in `verified` via admit_decoded.
                            memo.admit_decoded(name, w);
                        }
                    }
                    Ok(row)
                } else {
                    Err(Error::Gravity(format!(
                        "tensor {name}: unsupported codec {codec:?}"
                    )))
                }
            }
        }
    }
}

/// Explicit Metal kernel candidates for `gravity-pq`.
///
/// [`Generic`](Self::Generic) is the existing production kernel and remains
/// the default used by [`pq_matvec_metal`]. The other variants are additive
/// and must be named by a benchmark or caller; no environment variable or
/// model path silently opts into them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PqMetalKernelVariant {
    /// Existing packed-bit extractor and one SIMD group per row.
    Generic,
    /// `bits=8` direct `codes[flat]` lookup; scalar FMA order within a lane.
    Bits8Direct,
    /// Direct byte lookup plus four independent `float4` accumulators.
    Bits8Vec4,
    /// Direct byte lookup plus an f32 double-single accumulator/reduction.
    ///
    /// This is an unpromoted numeric candidate. It has no throughput claim
    /// and is selected only through the explicit variant/autotune APIs.
    Bits8DoubleSingle,
    /// 2D row × four chunk slices, then ordered slice reduction.
    Bits8Vec4Split4,
    /// 2D row × eight chunk slices, then ordered slice reduction.
    Bits8Vec4Split8,
}

impl PqMetalKernelVariant {
    pub const ALL: [Self; 6] = [
        Self::Generic,
        Self::Bits8Direct,
        Self::Bits8Vec4,
        Self::Bits8DoubleSingle,
        Self::Bits8Vec4Split4,
        Self::Bits8Vec4Split8,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Generic => "generic",
            Self::Bits8Direct => "bits8-direct",
            Self::Bits8Vec4 => "bits8-vec4",
            Self::Bits8DoubleSingle => "bits8-double-single",
            Self::Bits8Vec4Split4 => "bits8-2d-split4",
            Self::Bits8Vec4Split8 => "bits8-2d-split8",
        }
    }

    pub const fn kernel_name(self) -> &'static str {
        match self {
            Self::Generic => "gravity_pq_matvec",
            Self::Bits8Direct => "gravity_pq_matvec_bits8_direct",
            Self::Bits8Vec4 => "gravity_pq_matvec_bits8_vec4",
            Self::Bits8DoubleSingle => "gravity_pq_matvec_bits8_double_single",
            Self::Bits8Vec4Split4 | Self::Bits8Vec4Split8 => "gravity_pq_matvec_bits8_2d",
        }
    }

    pub const fn split_count(self) -> Option<u32> {
        match self {
            Self::Bits8Vec4Split4 => Some(4),
            Self::Bits8Vec4Split8 => Some(8),
            _ => None,
        }
    }

    pub const fn dispatches_per_matvec(self) -> usize {
        if self.split_count().is_some() {
            2
        } else {
            1
        }
    }

    /// Whether this candidate can interpret the tensor without changing
    /// artifact semantics. Vector candidates require aligned 4-wide sections;
    /// direct-byte candidates require exactly one byte per index.
    pub fn supports(self, h: &PqHeader) -> bool {
        match self {
            Self::Generic => true,
            Self::Bits8Direct | Self::Bits8DoubleSingle => h.bits == 8,
            Self::Bits8Vec4 | Self::Bits8Vec4Split4 | Self::Bits8Vec4Split8 => {
                h.bits == 8 && h.d % 4 == 0 && h.sub % 4 == 0
            }
        }
    }

    pub fn validate(self, h: &PqHeader) -> Result<()> {
        if self.supports(h) {
            return Ok(());
        }
        Err(Error::Gravity(format!(
            "PQ kernel {} does not support D={}, S={}, sub={}, card={}, bits={}; \
             byte variants require bits=8 and vector variants also require D/sub multiples of 4",
            self.as_str(),
            h.d,
            h.s,
            h.sub,
            h.card,
            h.bits
        )))
    }
}

impl std::fmt::Display for PqMetalKernelVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for PqMetalKernelVariant {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let normalized = s.trim().to_ascii_lowercase();
        Self::ALL
            .into_iter()
            .find(|v| v.as_str() == normalized)
            .ok_or_else(|| {
                format!(
                    "unknown PQ kernel variant {s:?}; expected {}",
                    Self::ALL
                        .iter()
                        .map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            })
    }
}

/// Summary of repeated synchronized matvec timings.
#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
pub struct PqMetalTimingSummary {
    pub min_us: f64,
    pub median_us: f64,
    pub p95_us: f64,
    pub mean_us: f64,
}

/// One bounded candidate measurement. `gpu` is populated when the process
/// starts with `HAWKING_TCB_TRACE=gpu_prod` on a counter-capable device.
#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub struct PqMetalBenchmark {
    pub variant: PqMetalKernelVariant,
    pub warmup: usize,
    pub iterations: usize,
    pub wall: PqMetalTimingSummary,
    pub gpu: Option<PqMetalTimingSummary>,
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

#[cfg(target_os = "macos")]
impl From<PqHeader> for GravityPqParams {
    fn from(h: PqHeader) -> Self {
        Self {
            dim: h.d as u32,
            subspaces: h.s as u32,
            sub: h.sub as u32,
            card: h.card as u32,
            rows: h.rows,
            cols: h.cols,
            nchunk: h.nchunk,
            bits: h.bits as u32,
        }
    }
}

/// A compact PQ matrix uploaded once for parity and kernel-autotune runs.
/// Weight/code buffers stay resident across candidates and iterations, so
/// timings cover dispatch + execution rather than artifact upload.
#[cfg(target_os = "macos")]
pub struct PqMetalMatrix {
    header: PqHeader,
    params: GravityPqParams,
    codebooks: metal::Buffer,
    codes: metal::Buffer,
}

#[cfg(target_os = "macos")]
impl PqMetalMatrix {
    pub fn from_payload(ctx: &crate::metal::MetalContext, payload: &[u8]) -> Result<Self> {
        let header = parse_pq_header(payload)?;
        if header.rotate != 0 {
            return Err(Error::Gravity(
                "rotated gravity-pq artifacts (rotate=1) are not yet supported".into(),
            ));
        }
        let (cb, packed_codes) = pq_sections(payload)?;
        // The generic kernel reads a four-byte MSB window. Keep its established
        // tail-padding contract even though direct-byte variants do not need it.
        let mut codes_padded = Vec::with_capacity(packed_codes.len() + 4);
        codes_padded.extend_from_slice(packed_codes);
        codes_padded.extend_from_slice(&[0u8; 4]);
        Ok(Self {
            header,
            params: header.into(),
            codebooks: ctx.new_buffer_with_bytes_checked(cb)?,
            codes: ctx.new_buffer_with_bytes_checked(&codes_padded)?,
        })
    }

    pub const fn header(&self) -> PqHeader {
        self.header
    }

    fn prepare(
        &self,
        ctx: &crate::metal::MetalContext,
        variant: PqMetalKernelVariant,
    ) -> Result<()> {
        variant.validate(&self.header)?;
        let _ = ctx.pipeline(variant.kernel_name())?;
        if variant.split_count().is_some() {
            let _ = ctx.pipeline("gravity_pq_reduce_2d")?;
        }
        Ok(())
    }

    fn encode(
        &self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        variant: PqMetalKernelVariant,
        x: &metal::Buffer,
        y: &metal::Buffer,
        partials: Option<&metal::Buffer>,
    ) -> Result<()> {
        const ROW_TG: u32 = 256;
        let params = self.params;
        if let Some(splits) = variant.split_count() {
            let scratch = partials.ok_or_else(|| {
                Error::Gravity(format!("PQ kernel {variant} requires a partials buffer"))
            })?;
            // One 32-thread SIMD group per (row, chunk-slice) pair.
            tcb.dispatch_threads(
                variant.kernel_name(),
                (params.rows * 32, splits, 1),
                (32, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&self.codebooks), 0);
                    enc.set_buffer(1, Some(&self.codes), 0);
                    enc.set_buffer(2, Some(x), 0);
                    enc.set_buffer(3, Some(scratch), 0);
                    enc.set_bytes(
                        4,
                        std::mem::size_of::<GravityPqParams>() as u64,
                        &params as *const GravityPqParams as *const _,
                    );
                    enc.set_bytes(5, 4, &splits as *const u32 as *const _);
                },
            )?;
            tcb.dispatch_threads(
                "gravity_pq_reduce_2d",
                (params.rows.div_ceil(ROW_TG) * ROW_TG, 1, 1),
                (ROW_TG, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(scratch), 0);
                    enc.set_buffer(1, Some(y), 0);
                    enc.set_bytes(
                        2,
                        std::mem::size_of::<GravityPqParams>() as u64,
                        &params as *const GravityPqParams as *const _,
                    );
                    enc.set_bytes(3, 4, &splits as *const u32 as *const _);
                },
            )
        } else {
            // Existing mapping: one SIMD group per row, 8 groups/TG.
            let n_tg = params.rows.div_ceil(8);
            tcb.dispatch_threads(
                variant.kernel_name(),
                (n_tg * ROW_TG, 1, 1),
                (ROW_TG, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&self.codebooks), 0);
                    enc.set_buffer(1, Some(&self.codes), 0);
                    enc.set_buffer(2, Some(x), 0);
                    enc.set_buffer(3, Some(y), 0);
                    enc.set_bytes(
                        4,
                        std::mem::size_of::<GravityPqParams>() as u64,
                        &params as *const GravityPqParams as *const _,
                    );
                },
            )
        }
    }

    fn activation_buffers(
        &self,
        ctx: &crate::metal::MetalContext,
        variant: PqMetalKernelVariant,
        x: &[f32],
    ) -> Result<(metal::Buffer, metal::Buffer, Option<metal::Buffer>)> {
        if x.len() != self.header.cols as usize {
            return Err(Error::Gravity(format!(
                "PQ Metal matvec: x.len() {} != cols {}",
                x.len(),
                self.header.cols
            )));
        }
        self.prepare(ctx, variant)?;
        let x_buf = ctx.new_buffer_with_bytes_checked(bytemuck::cast_slice::<f32, u8>(x))?;
        let y_buf =
            ctx.new_buffer_checked(self.header.rows as usize * std::mem::size_of::<f32>())?;
        let partials = variant
            .split_count()
            .map(|splits| {
                ctx.new_buffer_checked(
                    self.header.rows as usize
                        * splits as usize
                        * std::mem::size_of::<f32>(),
                )
            })
            .transpose()?;
        Ok((x_buf, y_buf, partials))
    }

    /// Run one explicit candidate and read back its result.
    pub fn matvec(
        &self,
        ctx: &crate::metal::MetalContext,
        variant: PqMetalKernelVariant,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        let (x_buf, y_buf, partials) = self.activation_buffers(ctx, variant, x)?;
        let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
        self.encode(&mut tcb, variant, &x_buf, &y_buf, partials.as_ref())?;
        tcb.commit_and_wait()?;
        let ptr = y_buf.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(ptr, self.header.rows as usize) }.to_vec())
    }

    /// Benchmark a candidate with resident weight, activation, output, and
    /// scratch buffers. Every sample is one synchronized matvec command
    /// buffer; a 2D candidate includes its ordered reduction dispatch.
    pub fn benchmark(
        &self,
        ctx: &crate::metal::MetalContext,
        variant: PqMetalKernelVariant,
        x: &[f32],
        warmup: usize,
        iterations: usize,
    ) -> Result<PqMetalBenchmark> {
        use std::time::Instant;

        if iterations == 0 {
            return Err(Error::Gravity(
                "PQ Metal benchmark requires at least one iteration".into(),
            ));
        }
        let (x_buf, y_buf, partials) = self.activation_buffers(ctx, variant, x)?;
        let run_once = || -> Result<()> {
            let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
            self.encode(&mut tcb, variant, &x_buf, &y_buf, partials.as_ref())?;
            tcb.commit_and_wait()
        };
        let _ = ctx.drain_trace();
        for _ in 0..warmup {
            run_once()?;
        }
        let _ = ctx.drain_trace();

        let mut wall_us = Vec::with_capacity(iterations);
        let mut gpu_us = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let t0 = Instant::now();
            run_once()?;
            wall_us.push(t0.elapsed().as_secs_f64() * 1e6);
            let samples = ctx.drain_trace();
            let gpu: Vec<u64> = samples.iter().filter_map(|s| s.gpu_us).collect();
            if gpu.len() == variant.dispatches_per_matvec() {
                gpu_us.push(gpu.into_iter().sum::<u64>() as f64);
            }
        }
        Ok(PqMetalBenchmark {
            variant,
            warmup,
            iterations,
            wall: summarize_timings(&wall_us),
            gpu: (gpu_us.len() == iterations).then(|| summarize_timings(&gpu_us)),
        })
    }
}

#[cfg(target_os = "macos")]
fn summarize_timings(samples: &[f64]) -> PqMetalTimingSummary {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let percentile = |p: f64| {
        let i = ((sorted.len() - 1) as f64 * p).round() as usize;
        sorted[i.min(sorted.len() - 1)]
    };
    PqMetalTimingSummary {
        min_us: sorted[0],
        median_us: percentile(0.50),
        p95_us: percentile(0.95),
        mean_us: sorted.iter().sum::<f64>() / sorted.len() as f64,
    }
}

/// GPU counterpart of [`pq_matvec`] using an explicit additive candidate.
#[cfg(target_os = "macos")]
pub fn pq_matvec_metal_with_variant(
    ctx: &crate::metal::MetalContext,
    payload: &[u8],
    x: &[f32],
    variant: PqMetalKernelVariant,
) -> Result<Vec<f32>> {
    PqMetalMatrix::from_payload(ctx, payload)?.matvec(ctx, variant, x)
}

/// GPU counterpart of [`pq_matvec`]. This remains pinned to
/// [`PqMetalKernelVariant::Generic`], preserving the established production
/// kernel and launch exactly; autotune results require an explicit later
/// promotion before they can affect the default.
#[cfg(target_os = "macos")]
pub fn pq_matvec_metal(
    ctx: &crate::metal::MetalContext,
    payload: &[u8],
    x: &[f32],
) -> Result<Vec<f32>> {
    pq_matvec_metal_with_variant(ctx, payload, x, PqMetalKernelVariant::Generic)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

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

    /// Minimal multi-shard Lazy fixture: one `model-*.gravity` with native.f32
    /// tensors so `open_dir` takes the Lazy path (Eager `open` would decode
    /// once at load and never exercise the memo).
    fn write_lazy_native_fixture(
        dir: &Path,
        tensors: &[(&str, &[f32])],
    ) -> std::path::PathBuf {
        let mut body = Vec::new();
        let mut descs = Vec::new();
        for &(name, vals) in tensors {
            let mut blob = Vec::with_capacity(vals.len() * 4);
            for &v in vals {
                blob.extend_from_slice(&v.to_le_bytes());
            }
            let mut h = Sha256::new();
            h.update(&blob);
            let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
            descs.push(serde_json::json!({
                "name": name,
                "codec": "native.f32",
                "offset": body.len() as u64,
                "bytes": blob.len() as u64,
                "sha256": hex,
                "shape": [vals.len() as u64],
                "elements": vals.len() as u64,
            }));
            body.extend_from_slice(&blob);
        }
        let header = serde_json::json!({
            "schema": "hawking.gravity.shard_header.v1",
            "format_version": 1,
            "model": {"name": "dense-memo-fixture"},
            "architecture": {},
            "tokenizer": {},
            "compression": {"codec": "native"},
            "shard": {"index": 1, "count": 1},
            "integrity": {"tensor_count": descs.len()},
            "tensors": descs,
        });
        let header_bytes = serde_json::to_vec(&header).expect("header json");
        let path = dir.join("model-00001-of-00001.gravity");
        let mut f = File::create(&path).expect("create shard");
        f.write_all(MAGIC).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        f.write_all(&header_bytes).unwrap();
        f.write_all(&body).unwrap();
        path
    }

    /// One header-only PQ tensor with a deliberately false payload digest.
    /// Admission must be able to inspect the 64-byte header without claiming
    /// integrity; the ordinary full-payload access must still reject it.
    fn write_lazy_unverified_pq_header_fixture(dir: &Path, name: &str) {
        let mut payload = Vec::with_capacity(PQ_HEADER_LEN);
        payload.extend_from_slice(PQ_MAGIC);
        payload.extend_from_slice(&32u16.to_le_bytes()); // D
        payload.extend_from_slice(&1u16.to_le_bytes()); // S
        payload.extend_from_slice(&32u16.to_le_bytes()); // sub
        payload.extend_from_slice(&256u16.to_le_bytes()); // card
        payload.extend_from_slice(&2u32.to_le_bytes()); // rows
        payload.extend_from_slice(&32u32.to_le_bytes()); // cols
        payload.extend_from_slice(&1u32.to_le_bytes()); // nchunk
        payload.extend_from_slice(&7u32.to_le_bytes()); // seed
        payload.extend_from_slice(&8u16.to_le_bytes()); // bits
        payload.push(0); // rotate
        payload.push(1); // n_codebooks
        payload.resize(PQ_HEADER_LEN, 0);

        let header = serde_json::json!({
            "schema": "hawking.gravity.shard_header.v1",
            "format_version": 1,
            "model": {"name": "pq-header-prefix-fixture"},
            "architecture": {},
            "tokenizer": {},
            "compression": {"codec": "gravity-pq"},
            "shard": {"index": 1, "count": 1},
            "integrity": {"tensor_count": 1},
            "tensors": [{
                "name": name,
                "codec": "gravity-pq",
                "offset": 0,
                "bytes": payload.len() as u64,
                "sha256": "00".repeat(32),
                "shape": [2, 32],
                "elements": 64,
            }],
        });
        let header_bytes = serde_json::to_vec(&header).expect("header json");
        let path = dir.join("model-00001-of-00001.gravity");
        let mut f = File::create(path).expect("create PQ prefix shard");
        f.write_all(MAGIC).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        f.write_all(&header_bytes).unwrap();
        f.write_all(&payload).unwrap();
    }

    #[test]
    fn pq_admission_header_prefix_does_not_claim_full_payload_verification() {
        let dir = tempfile::tempdir().expect("tempdir");
        let name = "model.layers.0.self_attn.kv_b_proj.weight";
        write_lazy_unverified_pq_header_fixture(dir.path(), name);

        let weights = GravityWeights::open_dir(dir.path(), true).expect("open_dir");
        let (header, shape) = weights
            .pq_header_prefix_unverified_with_shape(name)
            .expect("header-only admission");
        assert_eq!(
            header,
            PqHeader {
                d: 32,
                s: 1,
                sub: 32,
                card: 256,
                rows: 2,
                cols: 32,
                nchunk: 1,
                seed: 7,
                bits: 8,
                rotate: 0,
                n_codebooks: 1,
            }
        );
        assert_eq!(shape, vec![2, 32]);

        let err = weights
            .raw_payload_with_shape(name)
            .expect_err("ordinary payload load must still enforce full SHA-256");
        assert!(
            err.to_string().contains("sha256 mismatch"),
            "full load did not preserve verification: {err}"
        );
    }

    #[test]
    fn dense_memo_verifies_once_across_many_calls() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vals: Vec<f32> = (0..64).map(|i| (i as f32) * 0.125 - 1.0).collect();
        write_lazy_native_fixture(dir.path(), &[("norm.weight", &vals)]);

        let weights = GravityWeights::open_dir(dir.path(), true).expect("open_dir");
        let first = weights.dense("norm.weight").expect("dense #1");
        assert_eq!(first, vals);

        let after_first = weights.dense_memo_stats();
        assert_eq!(after_first.verifications, 1, "first dense must verify");
        assert_eq!(after_first.misses, 1);
        assert_eq!(after_first.hits, 0);
        assert_eq!(after_first.entries, 1);
        assert_eq!(after_first.resident_bytes, 64 * 4);
        assert!(after_first.budget_bytes >= after_first.resident_bytes);

        for _ in 0..50 {
            let again = weights.dense("norm.weight").expect("dense repeat");
            assert_eq!(again, vals);
        }

        let stats = weights.dense_memo_stats();
        assert_eq!(
            stats.verifications, 1,
            "subsequent dense calls must not re-verify"
        );
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 50);
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.verified_tensors, 1);
    }

    #[test]
    fn dense_memo_returns_bit_identical_values() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Include subnormals / signed zeros / extremes so widen+cache cannot
        // quietly re-quantize.
        let vals = vec![
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            f32::from_bits(0x0000_0001), // smallest positive subnormal
            f32::MIN,
            f32::MAX,
            std::f32::consts::PI,
        ];
        write_lazy_native_fixture(
            dir.path(),
            &[
                ("a.weight", &vals),
                ("b.bias", &[0.5, -0.25, 2.0]),
            ],
        );

        let weights = GravityWeights::open_dir(dir.path(), true).expect("open_dir");
        let a0 = weights.dense("a.weight").expect("a cold");
        let b0 = weights.dense("b.bias").expect("b cold");
        let a1 = weights.dense("a.weight").expect("a warm");
        let b1 = weights.dense("b.bias").expect("b warm");

        assert_eq!(a0, vals);
        assert_eq!(a1, a0);
        assert_eq!(b0, vec![0.5, -0.25, 2.0]);
        assert_eq!(b1, b0);
        // Bit-identical, not merely approximate.
        for (i, (&x, &y)) in a0.iter().zip(a1.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "a.weight[{i}] bits");
        }

        let stats = weights.dense_memo_stats();
        assert_eq!(stats.verifications, 2);
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.entries, 2);
    }

    #[test]
    fn dense_memo_row_hits_same_decoded_entry() {
        let dir = tempfile::tempdir().expect("tempdir");
        // 3×4 matrix laid out row-major as native.f32.
        let matrix: Vec<f32> = (0..12).map(|i| i as f32).collect();
        write_lazy_native_fixture(dir.path(), &[("embed.weight", &matrix)]);

        let weights = GravityWeights::open_dir(dir.path(), true).expect("open_dir");
        let row1 = weights.row("embed.weight", 1, 4).expect("row cold");
        assert_eq!(row1, vec![4.0, 5.0, 6.0, 7.0]);

        let stats_after_row = weights.dense_memo_stats();
        assert_eq!(stats_after_row.verifications, 1);
        assert_eq!(stats_after_row.entries, 1);

        // dense of the same name must hit the memo, not re-verify.
        let full = weights.dense("embed.weight").expect("dense after row");
        assert_eq!(full, matrix);
        let stats = weights.dense_memo_stats();
        assert_eq!(stats.verifications, 1, "row already verified");
        assert_eq!(stats.hits, 1);
    }

    #[test]
    fn dense_memo_oversized_tensor_skips_residency_but_verifies_once() {
        let dir = tempfile::tempdir().expect("tempdir");
        // 8 f32s = 32 bytes; budget of 16 bytes cannot hold the decoded vec.
        let vals: Vec<f32> = (0..8).map(|i| i as f32).collect();
        write_lazy_native_fixture(dir.path(), &[("big.weight", &vals)]);

        // Build Lazy source with a tiny budget by opening normally then
        // swapping is not possible; instead exercise NativeDenseMemo directly
        // for the oversize rule, and the public path for verify-once via a
        // normal open (full budget). Direct unit for admit:
        let mut memo = NativeDenseMemo::new(16);
        memo.record_verification("big.weight");
        memo.admit_decoded("big.weight", vals.clone());
        assert!(!memo.has_decoded("big.weight"));
        assert!(memo.is_verified("big.weight"));
        assert_eq!(memo.stats().entries, 0);
        assert_eq!(memo.stats().verifications, 1);
        assert_eq!(memo.stats().verified_tensors, 1);

        let weights = GravityWeights::open_dir(dir.path(), true).expect("open_dir");
        for _ in 0..5 {
            assert_eq!(
                weights.dense("big.weight").expect("dense"),
                vals
            );
        }
        let stats = weights.dense_memo_stats();
        assert_eq!(stats.verifications, 1);
        assert_eq!(stats.hits, 4);
        assert_eq!(stats.misses, 1);
    }
}

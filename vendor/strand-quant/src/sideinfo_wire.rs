//! SDSQ — the entropy-coded per-block `scale_q` side-info section for the
//! `.strand` v2 (STR2) container. This is the **wire** half of sprint Lever 1
//! (C2 side-info rANS); the codec half is [`crate::sideinfo_rans`].
//!
//! # What this carries
//! One rANS-coded stream of every block's `scale_q` integer, in archive
//! tensor-then-block order. `sideinfo_rans` measured `scale_q` at ~9.8 bits/sym
//! against the 32 bits it is billed at in the compact ship representation (and
//! 32 of the 128 bits/block the v2 seek table spends, `BlockOffsetRecord`), so
//! coding it recovers ~0.084 bpw at **zero quality cost** — the decoded `scale_q`
//! is byte-identical to the stored value, so the integer LUT reconstruct
//! (`decode::reconstruct_q` over `eff_scale_q`) — and therefore every device's
//! decode and the SPRV block hashes — are unchanged.
//!
//! # Why a section and not a seek-table edit (the moat clause)
//! The v2 seek table (`format.rs` ~:330, `BlockOffsetRecord`) is the random-access
//! index for `deploy v2`; it is left **byte-untouched**. SDSQ is an additive,
//! EOF-chained side-channel appended to a finished STR2 archive, exactly like
//! OUTL/SPRV/SDSC. On decode the SDSQ stream is decoded and used to **overwrite**
//! each block's `scale_q`; because the value is recovered losslessly, the overwrite
//! is value-for-value identical to the inline seek-table `scale_q` — the win is
//! that the *ship* representation can carry `scale_q` here (≈10 bits/block) instead
//! of inline 32-bit. (`sub_scale`, the 0.022-bpw stream, is intentionally NOT here:
//! it lives in the per-tensor sideinfo plane, not the seek table, and folding it
//! would touch that plane's reconstruct — out of scope; see the sprint doc.)
//!
//! # Wire layout — mirrors OUTL/DBIA exactly (outlier_wire.rs / debias_wire.rs)
//! The section is appended page-aligned with a 16-byte EOF trailer
//! `(offset:u64 LE, bytes:u32 LE, magic b"SDSQ")` so it chains the same way OUTL /
//! SPRV / SDSC / RSLT do.
//!
//! ```text
//! SDSQ section (page-aligned, [section_off, section_off+section_bytes)):
//!   header (32 bytes)
//!     +0   magic     b"SDSQ"
//!     +4   version   u32 LE  (== SDSQ_VERSION)
//!     +8   n_tensors u32 LE  (== archive tensor count)
//!     +12  flags     u32 LE  (reserved, must be 0)
//!     +16  16 reserved bytes (must be 0)
//!   then a per-tensor block-count table, n_tensors entries:
//!     +0   n_blocks  u64 LE  (== archive tensor n_blocks)
//!   then the rANS scale_q stream:
//!     +0   stream_len u32 LE
//!     +4   stream     sideinfo_rans::encode_scale_q(all blocks' scale_q,
//!                     concatenated in tensor-then-block order)
//!   then zero pad up to the trailer
//!   trailer (16 bytes): offset:u64 LE | bytes:u32 LE | b"SDSQ"
//! ```
//!
//! # Chain position chosen (LOAD-BEARING — the silent-drop hazard)
//! Canonical append order is `base -> [SDSC] -> [OUTL] -> SDSQ -> SPRV`, i.e. SDSQ
//! sits **above OUTL and below the SPRV seal** (SPRV/RSLT are outermost-only seals;
//! parse_sprv_section / parse_rslt_section require their trailer at EOF, so every
//! data section MUST be appended BEFORE the seal). Because SDSQ can sit ABOVE OUTL
//! and ABOVE SDSC, both of those readers' trailer-chain walkers are taught to step
//! over the SDSQ magic (see `outlier_wire::read_outl_bytes` and `selfdesc`'s
//! `chain_scan` / `read_sdsc_bytes`), exactly as `debias_wire::read_dbia_bytes`
//! already steps over the data magics under it. `read_sdsq_bytes` itself steps over
//! { SPRV, OUTL, RSLT, SDSC } to reach SDSQ beneath any of them. SPRV/RSLT readers
//! are outermost-only and never walk, so they need no change as long as SDSQ is
//! appended BEFORE the seal (enforced by the `append_sdsq` SPRV-seal guard).
//!
//! Determinism: encode counts are integers, decode is the integer-only
//! `sideinfo_rans::decode_scale_q`, serialise/parse round-trips the exact bytes,
//! and the apply is a plain `i32` field overwrite (no float anywhere). Two appends
//! of the same scale_q onto the same base produce byte-identical files.

use std::fs;
use std::io::Write as _;
use std::path::Path;

use crate::format::{flags_v2, read_strand_v2, read_strand_v2_header, OwnedTensorV2, StrandV2Header, PAGE};
use crate::sideinfo_rans::{decode_scale_q, encode_scale_q};

pub const SDSQ_MAGIC: &[u8; 4] = b"SDSQ";

/// Magics this reader steps *over* while walking the EOF trailer chain to reach an
/// SDSQ trailer that sits underneath them. SDSQ is the innermost-but-one data
/// section; the seals (SPRV/RSLT) and the other data sections (OUTL/SDSC) can sit
/// above it. Mirrors the step-over set in `debias_wire::read_dbia_bytes`.
const SPRV_MAGIC: &[u8; 4] = b"SPRV";
const OUTL_MAGIC: &[u8; 4] = b"OUTL";
const RSLT_MAGIC: &[u8; 4] = b"RSLT";
const SDSC_MAGIC: &[u8; 4] = b"SDSC";

pub const SDSQ_VERSION: u32 = 1;

pub const SDSQ_HEADER_BYTES: usize = 32;

pub const SDSQ_TRAILER_BYTES: usize = 16;

/// Parsed SDSQ section: the per-tensor block-count split plus the decoded
/// `scale_q` stream. `block_counts[i]` is tensor `i`'s `n_blocks`; the flat
/// `scale_q` runs in tensor-then-block order and its length equals
/// `block_counts.iter().sum()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SdsqSection {
    pub block_counts: Vec<usize>,
    pub scale_q: Vec<i32>,
}

impl SdsqSection {
    /// Total blocks coded (== `scale_q.len()`).
    pub fn total_blocks(&self) -> usize {
        self.scale_q.len()
    }

    /// Borrow tensor `i`'s `scale_q` slice from the flat stream, or `None` if `i`
    /// is out of range. (Walks the prefix sum; archives have few tensors.)
    pub fn tensor_scale_q(&self, i: usize) -> Option<&[i32]> {
        if i >= self.block_counts.len() {
            return None;
        }
        let start: usize = self.block_counts[..i].iter().sum();
        let n = self.block_counts[i];
        self.scale_q.get(start..start + n)
    }
}

#[inline]
fn page_align(x: usize) -> usize {
    (x + PAGE - 1) & !(PAGE - 1)
}

fn sdsq_section_bytes(block_counts: &[usize], scale_q: &[i32]) -> Result<Vec<u8>, String> {
    let total: usize = block_counts.iter().sum();
    if total != scale_q.len() {
        return Err(format!("sdsq: block_counts sum {total} != scale_q len {}", scale_q.len()));
    }
    let mut o = Vec::new();
    o.extend_from_slice(SDSQ_MAGIC);
    o.extend_from_slice(&SDSQ_VERSION.to_le_bytes());
    o.extend_from_slice(&(block_counts.len() as u32).to_le_bytes());
    o.extend_from_slice(&0u32.to_le_bytes()); // flags (reserved)
    o.extend_from_slice(&[0u8; 16]); // reserved
    debug_assert_eq!(o.len(), SDSQ_HEADER_BYTES);

    for &nb in block_counts {
        o.extend_from_slice(&(nb as u64).to_le_bytes());
    }

    let stream = encode_scale_q(scale_q);
    let stream_len: u32 = stream.len().try_into().map_err(|_| format!("sdsq: scale_q stream is {} bytes — exceeds the u32 field", stream.len()))?;
    o.extend_from_slice(&stream_len.to_le_bytes());
    o.extend_from_slice(&stream);
    Ok(o)
}

/// Append an SDSQ section to a finished v2 archive. Mirrors `append_outl` /
/// `append_dbia`: refuses to append behind a SPRV seal (SDSQ must go on BEFORE the
/// seal), refuses a double-append, page-aligns the section, and leaves all prior
/// bytes — including the seek table — untouched.
pub fn append_sdsq(path: impl AsRef<Path>, scale_q: &[i32]) -> Result<(), String> {
    let path = path.as_ref();
    let buf = fs::read(path).map_err(|e| format!("sdsq: read {path:?}: {e}"))?;

    if buf.len() >= SDSQ_TRAILER_BYTES && &buf[buf.len() - 4..] == &SPRV_MAGIC[..] {
        return Err("sdsq: file already has an SPRV trailer — SDSQ must be appended BEFORE SPRV \
             (the seal is outermost; data sections stack under it)"
            .into());
    }
    if buf.len() >= SDSQ_TRAILER_BYTES && &buf[buf.len() - 4..] == &RSLT_MAGIC[..] {
        return Err("sdsq: file already has an RSLT trailer — SDSQ must be appended BEFORE the RSLT seal".into());
    }
    match read_sdsq_bytes(&buf, true) {
        Ok(Some(_)) => return Err("sdsq: file already has an SDSQ section (double-append rejected)".into()),
        Err(e) => {
            return Err(format!(
                "sdsq: file ends in SDSQ magic but the section is invalid — refusing to \
                 append a second trailer behind it: {e}"
            ))
        }
        Ok(None) => {}
    }

    let hdr = read_strand_v2_header(&buf)?;
    let block_counts: Vec<usize> = hdr.tensors.iter().map(|t| t.n_blocks).collect();
    let total: usize = block_counts.iter().sum();
    if scale_q.len() != total {
        return Err(format!("sdsq: {} scale_q values, archive has {total} blocks across {} tensors", scale_q.len(), hdr.tensors.len()));
    }

    let section = sdsq_section_bytes(&block_counts, scale_q)?;
    let sdsq_bytes: u32 = section.len().try_into().map_err(|_| format!("sdsq: section is {} bytes — exceeds the u32 field", section.len()))?;

    let sdsq_offset = page_align(buf.len());
    let lead_pad = sdsq_offset - buf.len();
    let end = page_align(sdsq_offset + section.len() + SDSQ_TRAILER_BYTES);
    let tail_pad = end - SDSQ_TRAILER_BYTES - sdsq_offset - section.len();

    let mut tail = Vec::with_capacity(lead_pad + section.len() + tail_pad + SDSQ_TRAILER_BYTES);
    tail.resize(lead_pad, 0u8);
    tail.extend_from_slice(&section);
    tail.resize(tail.len() + tail_pad, 0u8);
    tail.extend_from_slice(&(sdsq_offset as u64).to_le_bytes());
    tail.extend_from_slice(&sdsq_bytes.to_le_bytes());
    tail.extend_from_slice(SDSQ_MAGIC);

    let mut f = fs::OpenOptions::new().append(true).open(path).map_err(|e| format!("sdsq: open {path:?} for append: {e}"))?;
    f.write_all(&tail).map_err(|e| format!("sdsq: append to {path:?}: {e}"))?;
    Ok(())
}

fn parse_sdsq_section(buf: &[u8], sdsq_offset: usize, sdsq_bytes: usize, trailer_end: usize) -> Result<SdsqSection, String> {
    if sdsq_offset % PAGE != 0 {
        return Err(format!("sdsq: sdsq_offset {sdsq_offset} not page-aligned"));
    }
    let min_end = sdsq_offset.checked_add(sdsq_bytes).and_then(|x| x.checked_add(SDSQ_TRAILER_BYTES)).ok_or("sdsq: sdsq_offset + sdsq_bytes overflows")?;
    if min_end > trailer_end || trailer_end % PAGE != 0 {
        return Err(format!(
            "sdsq: section [{sdsq_offset}, +{sdsq_bytes}] + trailer does not fit the \
             page-aligned region ending at {trailer_end}"
        ));
    }
    if sdsq_bytes < SDSQ_HEADER_BYTES {
        return Err("sdsq: section shorter than the 32-byte header".into());
    }
    // padding between section end and the trailer must be zero (byte-stability)
    if buf[sdsq_offset + sdsq_bytes..trailer_end - SDSQ_TRAILER_BYTES].iter().any(|&b| b != 0) {
        return Err("sdsq: nonzero bytes in section padding".into());
    }

    let v2 = read_strand_v2_header(buf)?;

    let s = &buf[sdsq_offset..sdsq_offset + sdsq_bytes];
    if &s[0..4] != &SDSQ_MAGIC[..] {
        return Err("sdsq: bad section header magic".into());
    }
    let version = u32::from_le_bytes(s[4..8].try_into().unwrap());
    if version != SDSQ_VERSION {
        return Err(format!("sdsq: version {version} != {SDSQ_VERSION}"));
    }
    let n_tensors = u32::from_le_bytes(s[8..12].try_into().unwrap()) as usize;
    if n_tensors != v2.tensors.len() {
        return Err(format!("sdsq: section n_tensors {n_tensors} != archive's {}", v2.tensors.len()));
    }
    let flags = u32::from_le_bytes(s[12..16].try_into().unwrap());
    if flags != 0 {
        return Err(format!("sdsq: reserved flag bits set: {flags:#x}"));
    }
    if s[16..32].iter().any(|&b| b != 0) {
        return Err("sdsq: header reserved bytes not zero".into());
    }

    let mut p = SDSQ_HEADER_BYTES;
    let take = |p: &mut usize, n: usize| -> Result<&[u8], String> {
        let end = p.checked_add(n).filter(|&e| e <= s.len()).ok_or("sdsq: section truncated")?;
        let sl = &s[*p..end];
        *p = end;
        Ok(sl)
    };

    let mut block_counts = Vec::with_capacity(n_tensors);
    for (i, desc) in v2.tensors.iter().enumerate() {
        let nb = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap()) as usize;
        if nb != desc.n_blocks {
            return Err(format!("sdsq: tensor record {i}: n_blocks {nb} != archive's {}", desc.n_blocks));
        }
        block_counts.push(nb);
    }

    let stream_len = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap()) as usize;
    let stream = take(&mut p, stream_len)?;
    if p != sdsq_bytes {
        return Err(format!("sdsq: {} trailing bytes after the scale_q stream", sdsq_bytes - p));
    }

    let mut spos = 0usize;
    let scale_q = decode_scale_q(stream, &mut spos)?;
    if spos != stream.len() {
        return Err(format!("sdsq: {} trailing bytes inside the rANS stream", stream.len() - spos));
    }
    let total: usize = block_counts.iter().sum();
    if scale_q.len() != total {
        return Err(format!("sdsq: decoded {} scale_q values, block_counts sum to {total}", scale_q.len()));
    }
    Ok(SdsqSection { block_counts, scale_q })
}

/// Read the SDSQ section from a buffer, walking back through any SPRV / OUTL /
/// RSLT / SDSC trailers that sit on top of it. Returns `Ok(None)` if there is no
/// SDSQ section. With `strict = true` a present-but-corrupt SDSQ trailer is an
/// `Err`; with `strict = false` it degrades to `Ok(None)`.
pub fn read_sdsq_bytes(buf: &[u8], strict: bool) -> Result<Option<SdsqSection>, String> {
    let mut end = buf.len();
    // Bounded walk: at most a few stacked sections (SDSC/OUTL/SDSQ/RSLT/SPRV). 6 is slack.
    for _ in 0..6 {
        if end < SDSQ_TRAILER_BYTES {
            return Ok(None);
        }
        let t = &buf[end - SDSQ_TRAILER_BYTES..end];
        let magic = &t[12..16];
        if magic == &SDSQ_MAGIC[..] {
            let parse = (|| -> Result<SdsqSection, String> {
                let sdsq_offset = u64::from_le_bytes(t[0..8].try_into().unwrap());
                let sdsq_bytes = u32::from_le_bytes(t[8..12].try_into().unwrap());
                let sdsq_offset: usize = sdsq_offset.try_into().map_err(|_| "sdsq: sdsq_offset exceeds address space".to_string())?;
                parse_sdsq_section(buf, sdsq_offset, sdsq_bytes as usize, end)
            })();
            return match parse {
                Ok(s) => Ok(Some(s)),
                Err(e) if strict => Err(e),
                Err(_) => Ok(None),
            };
        } else if magic == &SPRV_MAGIC[..] || magic == &OUTL_MAGIC[..] || magic == &RSLT_MAGIC[..] || magic == &SDSC_MAGIC[..] {
            // Every stacked section's trailer stores its own section start at [0..8].
            // Step back to it and keep looking for SDSQ underneath.
            let off = u64::from_le_bytes(t[0..8].try_into().unwrap());
            let Ok(off) = usize::try_from(off) else {
                return Ok(None);
            };
            if off >= end {
                return Ok(None);
            }
            end = off;
        } else {
            return Ok(None);
        }
    }
    Ok(None)
}

/// Read the SDSQ section from a file path.
pub fn read_sdsq(path: impl AsRef<Path>) -> Result<Option<SdsqSection>, String> {
    let path = path.as_ref();
    let buf = fs::read(path).map_err(|e| format!("sdsq: read {path:?}: {e}"))?;
    read_sdsq_bytes(&buf, true)
}

/// Decode-apply: overwrite each block's `scale_q` in a parsed v2 header from the
/// SDSQ section. The header's per-tensor `n_blocks` must match the section's
/// `block_counts` (already validated at parse time against the same archive).
/// After this, `hdr.tensors[t].table[b].scale_q` carries the SDSQ-decoded value —
/// byte-identical to the inline seek-table value, so the integer reconstruct is
/// unchanged. This is the reference apply the loader uses when an SDSQ section is
/// present; it touches only the in-memory `scale_q` field, never the seek-table
/// bytes on disk.
pub fn apply_sdsq_to_header(hdr: &mut StrandV2Header, sdsq: &SdsqSection) -> Result<(), String> {
    if hdr.tensors.len() != sdsq.block_counts.len() {
        return Err(format!("sdsq apply: header has {} tensors, SDSQ has {}", hdr.tensors.len(), sdsq.block_counts.len()));
    }
    let mut pos = 0usize;
    for (i, t) in hdr.tensors.iter_mut().enumerate() {
        let nb = sdsq.block_counts[i];
        if t.n_blocks != nb || t.table.len() != nb {
            return Err(format!("sdsq apply: tensor {i}: header n_blocks {}/{} != SDSQ {nb}", t.n_blocks, t.table.len()));
        }
        let slice = sdsq.scale_q.get(pos..pos + nb).ok_or("sdsq apply: scale_q stream shorter than block_counts")?;
        for (rec, &sq) in t.table.iter_mut().zip(slice.iter()) {
            rec.scale_q = sq;
        }
        pos += nb;
    }
    Ok(())
}

/// Decode-apply for the **full-read** (`read_strand_v2`) path: overwrite every
/// `OwnedTensorV2`'s per-block `scale_q` — in BOTH the reconstructed
/// `base.enc.blocks` and the `table` seek records — from the SDSQ section. The
/// full reader builds `EncodedTensor.blocks`, which is what `block_hashes`/SPRV
/// and the decode kernels read, so SPRV verification and any `read_strand_v2`
/// consumer need this fill when [`flags_v2::SCALEQ_IN_SDSQ`] is set. Byte-identical
/// to the value the legacy 16-byte record would have carried.
pub fn apply_sdsq_to_tensors(tensors: &mut [OwnedTensorV2], sdsq: &SdsqSection) -> Result<(), String> {
    if tensors.len() != sdsq.block_counts.len() {
        return Err(format!("sdsq apply: {} tensors, SDSQ has {}", tensors.len(), sdsq.block_counts.len()));
    }
    let mut pos = 0usize;
    for (i, t) in tensors.iter_mut().enumerate() {
        let nb = sdsq.block_counts[i];
        if t.base.enc.blocks.len() != nb || t.table.len() != nb {
            return Err(format!("sdsq apply: tensor {i}: blocks {}/table {} != SDSQ {nb}", t.base.enc.blocks.len(), t.table.len()));
        }
        let slice = sdsq.scale_q.get(pos..pos + nb).ok_or("sdsq apply: scale_q stream shorter than block_counts")?;
        for ((blk, rec), &sq) in t.base.enc.blocks.iter_mut().zip(t.table.iter_mut()).zip(slice.iter()) {
            blk.scale_q = sq;
            rec.scale_q = sq;
        }
        pos += nb;
    }
    Ok(())
}

/// Read a v2 header and, if the archive carries [`flags_v2::SCALEQ_IN_SDSQ`],
/// source `scale_q` from the SDSQ section so `table[..].scale_q` is correct.
///
/// **Hard-errors** if the flag is set but no SDSQ section is present (or it is
/// corrupt): a packed archive whose side-info is missing is unrecoverable and must
/// never silently decode with the `0` placeholder. For a legacy (16-byte-record)
/// archive this is exactly `read_strand_v2_header` — SDSQ, if present, is ignored
/// because the inline `scale_q` is already authoritative.
pub fn read_strand_v2_header_applied(buf: &[u8]) -> Result<StrandV2Header, String> {
    let mut hdr = read_strand_v2_header(buf)?;
    if hdr.flags & flags_v2::SCALEQ_IN_SDSQ != 0 {
        let sdsq = read_sdsq_bytes(buf, true)?.ok_or(
            "sdsq: archive sets SCALEQ_IN_SDSQ but has no SDSQ section — scale_q is \
             unrecoverable (refusing to decode with a 0 placeholder)",
        )?;
        apply_sdsq_to_header(&mut hdr, &sdsq)?;
    }
    Ok(hdr)
}

/// Full-read variant of [`read_strand_v2_header_applied`]: parse every tensor and,
/// for a packed archive, fill `scale_q` from SDSQ in both `enc.blocks` and `table`.
/// Hard-errors if the flag is set but SDSQ is absent/corrupt.
pub fn read_strand_v2_applied(buf: &[u8]) -> Result<Vec<OwnedTensorV2>, String> {
    let mut tensors = read_strand_v2(buf)?;
    let flags = read_strand_v2_header(buf)?.flags;
    if flags & flags_v2::SCALEQ_IN_SDSQ != 0 {
        let sdsq = read_sdsq_bytes(buf, true)?.ok_or(
            "sdsq: archive sets SCALEQ_IN_SDSQ but has no SDSQ section — scale_q is \
             unrecoverable (refusing to decode with a 0 placeholder)",
        )?;
        apply_sdsq_to_tensors(&mut tensors, &sdsq)?;
    }
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{encode_tensor_with, EncodeOpts};
    use crate::format::{read_strand_v2, write_strand_v2, PackedTensor, PackedTensorV2};
    use crate::trellis::TrellisConfig;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn tmp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("strand-sdsq-{tag}-{}-{}.strand", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed)))
    }

    struct TmpFile(PathBuf);
    impl Drop for TmpFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn test_weights(n: usize, seed: u64) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5).collect()
    }

    fn build_test_archive() -> Vec<u8> {
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc_a = encode_tensor_with(&test_weights(1024, 11), &cfg, &EncodeOpts::default());
        let enc_b = encode_tensor_with(&test_weights(768, 23), &cfg, &EncodeOpts::default());
        let shape_a = [4u64, 256u64];
        let shape_b = [3u64, 256u64];
        let tensors = [
            PackedTensorV2 {
                base: PackedTensor { name: "model.layers.0.q_proj", shape: &shape_a, rht_seed: 0, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc: &enc_a },
                block_len: cfg.block_len as u32,
            },
            PackedTensorV2 {
                base: PackedTensor { name: "model.layers.0.down_proj", shape: &shape_b, rht_seed: 0, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc: &enc_b },
                block_len: cfg.block_len as u32,
            },
        ];
        write_strand_v2(&tensors, [9u8; 32], true).expect("write v2")
    }

    /// The exact per-block scale_q the producer would feed `append_sdsq`.
    fn archive_scale_q(buf: &[u8]) -> Vec<i32> {
        let hdr = read_strand_v2_header(buf).unwrap();
        hdr.tensors.iter().flat_map(|t| t.table.iter().map(|r| r.scale_q)).collect()
    }

    #[test]
    fn sdsq_round_trip_and_v2_reader_compat() {
        let buf = build_test_archive();
        let path = tmp_path("roundtrip");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();

        assert_eq!(read_sdsq(&path).unwrap(), None, "plain v2 must read as absent");

        let scale_q = archive_scale_q(&buf);
        assert!(!scale_q.is_empty(), "fixture must have blocks");
        append_sdsq(&path, &scale_q).expect("append sdsq");

        let back = read_sdsq(&path).unwrap().expect("section found");
        assert_eq!(back.scale_q, scale_q, "SDSQ scale_q must round-trip byte-identically");
        let hdr = read_strand_v2_header(&buf).unwrap();
        assert_eq!(back.block_counts, hdr.tensors.iter().map(|t| t.n_blocks).collect::<Vec<_>>());

        // append must not touch the v2 bytes (the seek table included), and the v2
        // reader still works under the SDSQ trailer.
        let trailered = std::fs::read(&path).unwrap();
        assert_eq!(&trailered[..buf.len()], &buf[..], "append must not touch v2 bytes");
        assert_eq!(trailered.len() % PAGE, 0, "SDSQ end must be page-aligned (stacking)");
        let full = read_strand_v2(&trailered).expect("full v2 read under SDSQ trailer");
        assert_eq!(full.len(), 2);

        // double-append rejected, file untouched
        let before = std::fs::read(&path).unwrap();
        assert!(append_sdsq(&path, &scale_q).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    #[test]
    fn sdsq_apply_overwrites_scale_q_byte_identically() {
        let buf = build_test_archive();
        let path = tmp_path("apply");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();

        let scale_q = archive_scale_q(&buf);
        append_sdsq(&path, &scale_q).expect("append sdsq");
        let sdsq = read_sdsq(&path).unwrap().expect("sdsq present");

        // Take a header whose scale_q we deliberately CLOBBER, then apply SDSQ and
        // assert it recovers the original seek-table values exactly.
        let mut hdr = read_strand_v2_header(&buf).unwrap();
        for t in hdr.tensors.iter_mut() {
            for rec in t.table.iter_mut() {
                rec.scale_q = 0x7eadbeefu32 as i32; // poison
            }
        }
        apply_sdsq_to_header(&mut hdr, &sdsq).expect("apply");

        let want = read_strand_v2_header(&buf).unwrap();
        for (t, w) in hdr.tensors.iter().zip(want.tensors.iter()) {
            for (rec, wr) in t.table.iter().zip(w.table.iter()) {
                assert_eq!(rec.scale_q, wr.scale_q, "scale_q must be byte-identical post-apply");
            }
        }
    }

    #[test]
    fn append_is_byte_deterministic() {
        let buf = build_test_archive();
        let p1 = tmp_path("det1");
        let p2 = tmp_path("det2");
        let _g1 = TmpFile(p1.clone());
        let _g2 = TmpFile(p2.clone());
        std::fs::write(&p1, &buf).unwrap();
        std::fs::write(&p2, &buf).unwrap();
        let scale_q = archive_scale_q(&buf);
        append_sdsq(&p1, &scale_q).unwrap();
        append_sdsq(&p2, &scale_q).unwrap();
        assert_eq!(std::fs::read(&p1).unwrap(), std::fs::read(&p2).unwrap());
    }

    #[test]
    fn sdsq_validates_count() {
        let buf = build_test_archive();
        let path = tmp_path("validate");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();

        let mut scale_q = archive_scale_q(&buf);
        scale_q.push(123); // one too many
        assert!(append_sdsq(&path, &scale_q).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), buf, "rejected append leaves file intact");
    }

    #[test]
    fn sdsq_corrupt_trailer_is_error_not_crash() {
        let buf = build_test_archive();
        let path = tmp_path("corrupt");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();
        append_sdsq(&path, &archive_scale_q(&buf)).expect("append");
        let clean = std::fs::read(&path).unwrap();

        // corrupt the trailer's byte-count field
        let mut c1 = clean.clone();
        let pb_pos = c1.len() - 8;
        c1[pb_pos] ^= 0xFF;
        assert!(read_sdsq_bytes(&c1, true).is_err());
        assert_eq!(read_sdsq_bytes(&c1, false).unwrap(), None, "lenient degrades to None");

        // corrupt a header byte inside the section
        let sdsq_off = {
            let t = &clean[clean.len() - SDSQ_TRAILER_BYTES..];
            u64::from_le_bytes(t[0..8].try_into().unwrap()) as usize
        };
        let mut c2 = clean.clone();
        c2[sdsq_off + 4] ^= 0xFF; // version field
        assert!(read_sdsq_bytes(&c2, true).is_err());

        // degenerate inputs do not panic
        assert_eq!(read_sdsq_bytes(b"", true).unwrap(), None);
        assert_eq!(read_sdsq_bytes(b"SDSQ", true).unwrap(), None);
        let mut tiny = vec![0u8; SDSQ_TRAILER_BYTES];
        tiny[12..].copy_from_slice(SDSQ_MAGIC);
        assert!(read_sdsq_bytes(&tiny, true).is_err());
        assert_eq!(read_sdsq_bytes(&tiny, false).unwrap(), None);
    }
}

//! DBIA — the de-bias side-info section for the `.strand` v2 container.
//!
//! # What this carries
//! The ADOPTED inner-product de-bias win (see `debias.rs` for the derivation):
//! a per-output-row additive correction `c_i` (eq. 3 there, `c_i = -mu_bar * S_i`)
//! that is added to the layer output `y_i` at inference. The encode side already
//! computes these vectors (`quantize-model --actmean` -> `<out>.debias.json`,
//! `debias_correction()` in that bin). This module is the *wire* half: it serialises
//! the per-tensor `c` vectors into a DBIA section appended to a finished v2 archive,
//! and parses them back, so deploy never needs the JSON sidecar or the base weights.
//!
//! # Why a section and not a fold
//! Qwen projections have no bias to fold `c` into, and the correction is additive in
//! *output* space while the per-block scale is multiplicative in *weight* space — they
//! do not compose (debias.rs §billing). So `c` is a billed side-channel: one bf16 per
//! output row, `Δbpw = 16 / in_features` (≈0.0179 at in=896 — three orders below the
//! outlier channel; on dp_d4_r2 the measured payload was ~0.0136 bpw).
//!
//! # Wire layout — mirrors OUTL exactly (outlier_wire.rs)
//! The section is appended to a finished STR2 archive, page-aligned, with a 16-byte
//! EOF trailer `(offset:u64 LE, bytes:u32 LE, magic b"DBIA")` so it chains the same
//! way OUTL/SPRV/RSLT do. One fixed-size record per archive tensor; tensors with no
//! correction are encoded as a zero-length record (mirrors OUTL's `None`).
//!
//! ```text
//! DBIA section (page-aligned, [section_off, section_off+section_bytes)):
//!   header  (32 bytes)
//!     +0   magic    b"DBIA"
//!     +4   version  u32 LE  (== DBIA_VERSION)
//!     +8   n_tensors u32 LE (== archive tensor count)
//!     +12  flags    u32 LE  (reserved, must be 0)
//!     +16  16 reserved bytes (must be 0)
//!   then n_tensors records, in archive tensor order, each:
//!     +0   len      u32 LE  (number of bf16 entries == out_features, or 0 = absent)
//!     +4   reserved u32 LE  (must be 0)
//!     +8   len * u16 LE bf16 payload (round-to-nearest-even of c_i, big rows first)
//!   (records are NOT individually padded; the section as a whole is page-padded by
//!    the appender, exactly like OUTL.)
//!   then zero pad up to the trailer
//!   trailer (16 bytes): offset:u64 LE | bytes:u32 LE | b"DBIA"
//! ```
//!
//! # Float order (documented, load-bearing for byte-stability)
//! - Storage dtype is **bf16** (top 16 bits of the IEEE-754 f32, round-to-nearest-even).
//! - Encode: `f32 -> bf16` via [`f32_to_bf16_round`] (ties-to-even on the dropped 16
//!   mantissa bits; +inf/-inf/NaN pass through their top half unchanged).
//! - Decode: `bf16 -> f32` via [`bf16_to_f32`] == `f32::from_bits((bits as u32) << 16)`
//!   (identical to `safetensor_io::bf16_to_f32`).
//! - Apply: the correction is added to the f32 accumulator in the MAC epilogue AFTER
//!   the full inner product (and after any outlier residual term), one add per output
//!   row: `y[o] = y[o] + bf16_to_f32(c_bits[o])`. There is no per-element float
//!   reordering — the only float op on the decode side is this single deterministic add,
//!   so two runs on the same bytes produce bit-identical `y`.
//!
//! Determinism: encode (`f32_to_bf16_round`) and decode (`bf16_to_f32`) are pure
//! integer bit-twiddles; serialise/parse round-trips the exact `u16` payload; the apply
//! is a single f32 add with no accumulation order to vary. See the unit tests.

use std::fs;
use std::io::Write as _;
use std::path::Path;

use crate::format::{read_strand_v2_header, PAGE};

pub const DBIA_MAGIC: &[u8; 4] = b"DBIA";

/// Magics this reader steps *over* while walking the EOF trailer chain to reach a
/// DBIA trailer that sits underneath them. Mirrors the way `read_outl_bytes` steps
/// over SPRV. The canonical append order is base -> OUTL -> DBIA -> SPRV, so DBIA can
/// be found beneath SPRV and/or RSLT; OUTL is included for completeness (DBIA-then-OUTL
/// is not produced by the canonical appender but the walk tolerates it).
const SPRV_MAGIC: &[u8; 4] = b"SPRV";
const OUTL_MAGIC: &[u8; 4] = b"OUTL";
const RSLT_MAGIC: &[u8; 4] = b"RSLT";

pub const DBIA_VERSION: u32 = 1;

pub const DBIA_HEADER_BYTES: usize = 32;

pub const DBIA_TRAILER_BYTES: usize = 16;

/// Per-tensor record fixed prefix: `len: u32` + `reserved: u32`.
pub const DBIA_RECORD_FIXED_BYTES: usize = 8;

/// Deterministic f32 -> bf16 with round-to-nearest, ties-to-even.
///
/// bf16 keeps the sign, the 8-bit exponent and the top 7 mantissa bits of f32; the
/// low 16 bits are dropped. We round to nearest on those 16 dropped bits and break
/// ties toward an even low mantissa bit. NaN/Inf keep their top 16 bits unchanged
/// (the standard "truncate, do not round into Inf" behaviour for the non-finite class).
#[inline]
pub fn f32_to_bf16_round(x: f32) -> u16 {
    let bits = x.to_bits();
    // NaN / Inf: exponent all ones. Preserve the top half verbatim (for NaN this keeps
    // it a NaN; for Inf the low bits are already zero so rounding is a no-op anyway).
    if (bits & 0x7f80_0000) == 0x7f80_0000 {
        return (bits >> 16) as u16;
    }
    // round-to-nearest-even on the 16 dropped bits
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}

/// bf16 -> f32. Byte-identical to `safetensor_io::bf16_to_f32`.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// One archive tensor's de-bias correction, stored as bf16.
///
/// `c_bits.len()` is `out_features` (= `shape[0]`). The decode-side apply does
/// `y[o] += bf16_to_f32(c_bits[o])`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DebiasWire {
    /// One bf16 per output row, in row order (row 0 first).
    pub c_bits: Vec<u16>,
}

impl DebiasWire {
    /// Build from an f32 correction vector (`debias.rs` / `debias_correction()` output),
    /// rounding each entry to bf16. `c.len()` must equal the tensor's `out_features`.
    pub fn from_f32(c: &[f32]) -> Self {
        DebiasWire { c_bits: c.iter().map(|&v| f32_to_bf16_round(v)).collect() }
    }

    /// Dequantised corrections in row order. The decode apply consumes exactly this.
    pub fn dequant(&self) -> impl Iterator<Item = f32> + '_ {
        self.c_bits.iter().map(|&b| bf16_to_f32(b))
    }

    /// Serialised payload byte count for this record (fixed prefix + bf16 entries).
    #[inline]
    pub fn wire_bytes(&self) -> usize {
        DBIA_RECORD_FIXED_BYTES + self.c_bits.len() * 2
    }
}

/// The parsed DBIA section: one optional correction per archive tensor (index-aligned
/// with `StrandV2Header.tensors`). `None` = that tensor carries no correction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DbiaSection {
    pub tensors: Vec<Option<DebiasWire>>,
}

impl DbiaSection {
    /// Number of tensors that carry a correction.
    pub fn n_with_correction(&self) -> usize {
        self.tensors.iter().filter(|t| t.is_some()).count()
    }

    /// Total bf16 bias entries across all tensors.
    pub fn total_entries(&self) -> usize {
        self.tensors.iter().filter_map(|t| t.as_ref().map(|w| w.c_bits.len())).sum()
    }
}

#[inline]
fn page_align(x: usize) -> usize {
    (x + PAGE - 1) & !(PAGE - 1)
}

/// out_features for a tensor header shape: `shape[0]`. v2 STRICT tensors are 2-D
/// `[out, in]`; a 1-D tensor has `out = shape[0]` and no de-bias is meaningful, but we
/// still accept a zero-length (absent) record for it.
#[inline]
fn out_features_of(shape: &[u64]) -> Option<usize> {
    shape.first().map(|&d| d as usize)
}

fn dbia_section_bytes(wires: &[Option<DebiasWire>], out_features: &[usize]) -> Result<Vec<u8>, String> {
    debug_assert_eq!(wires.len(), out_features.len());
    let mut o = Vec::new();
    o.extend_from_slice(DBIA_MAGIC);
    o.extend_from_slice(&DBIA_VERSION.to_le_bytes());
    o.extend_from_slice(&(wires.len() as u32).to_le_bytes());
    o.extend_from_slice(&0u32.to_le_bytes()); // flags (reserved)
    o.extend_from_slice(&[0u8; 16]); // reserved
    debug_assert_eq!(o.len(), DBIA_HEADER_BYTES);

    for (i, (w, &out)) in wires.iter().zip(out_features.iter()).enumerate() {
        match w {
            None => {
                o.extend_from_slice(&0u32.to_le_bytes()); // len = 0 (absent)
                o.extend_from_slice(&0u32.to_le_bytes()); // reserved
            }
            Some(w) => {
                if w.c_bits.len() != out {
                    return Err(format!("dbia: tensor record {i}: correction has {} rows, tensor has out_features {out}", w.c_bits.len()));
                }
                if w.c_bits.is_empty() {
                    return Err(format!("dbia: tensor record {i}: Some(empty) is ambiguous — use None for absent"));
                }
                let len: u32 = w.c_bits.len().try_into().map_err(|_| format!("dbia: tensor record {i}: too many rows for u32 len"))?;
                o.extend_from_slice(&len.to_le_bytes());
                o.extend_from_slice(&0u32.to_le_bytes()); // reserved
                for &b in &w.c_bits {
                    o.extend_from_slice(&b.to_le_bytes());
                }
            }
        }
    }
    Ok(o)
}

/// Append a DBIA section to a finished v2 archive. Mirrors `append_outl`:
/// refuses to append behind a SPRV seal (DBIA must go on BEFORE SPRV), refuses a
/// double-append, page-aligns the section, leaves all prior bytes untouched.
pub fn append_dbia(path: impl AsRef<Path>, wires: &[Option<DebiasWire>]) -> Result<(), String> {
    let path = path.as_ref();
    let buf = fs::read(path).map_err(|e| format!("dbia: read {path:?}: {e}"))?;

    if buf.len() >= DBIA_TRAILER_BYTES && &buf[buf.len() - 4..] == &SPRV_MAGIC[..] {
        return Err("dbia: file already has an SPRV trailer — DBIA must be appended BEFORE SPRV \
             (sections stack as OUTL then DBIA then SPRV, SPRV outermost)"
            .into());
    }
    match read_dbia_bytes(&buf, true) {
        Ok(Some(_)) => return Err("dbia: file already has a DBIA section (double-append rejected)".into()),
        Err(e) => {
            return Err(format!(
                "dbia: file ends in DBIA magic but the section is invalid — refusing to \
                 append a second trailer behind it: {e}"
            ))
        }
        Ok(None) => {}
    }

    let hdr = read_strand_v2_header(&buf)?;
    if wires.len() != hdr.tensors.len() {
        return Err(format!("dbia: {} wire records, archive has {} tensors", wires.len(), hdr.tensors.len()));
    }
    let out_features: Vec<usize> = hdr.tensors.iter().map(|t| out_features_of(&t.shape).unwrap_or(0)).collect();
    let section = dbia_section_bytes(wires, &out_features)?;
    let dbia_bytes: u32 = section.len().try_into().map_err(|_| format!("dbia: section is {} bytes — exceeds the u32 field", section.len()))?;

    let dbia_offset = page_align(buf.len());
    let lead_pad = dbia_offset - buf.len();
    let end = page_align(dbia_offset + section.len() + DBIA_TRAILER_BYTES);
    let tail_pad = end - DBIA_TRAILER_BYTES - dbia_offset - section.len();

    let mut tail = Vec::with_capacity(lead_pad + section.len() + tail_pad + DBIA_TRAILER_BYTES);
    tail.resize(lead_pad, 0u8);
    tail.extend_from_slice(&section);
    tail.resize(tail.len() + tail_pad, 0u8);
    tail.extend_from_slice(&(dbia_offset as u64).to_le_bytes());
    tail.extend_from_slice(&dbia_bytes.to_le_bytes());
    tail.extend_from_slice(DBIA_MAGIC);

    let mut f = fs::OpenOptions::new().append(true).open(path).map_err(|e| format!("dbia: open {path:?} for append: {e}"))?;
    f.write_all(&tail).map_err(|e| format!("dbia: append to {path:?}: {e}"))?;
    Ok(())
}

fn parse_dbia_section(buf: &[u8], dbia_offset: usize, dbia_bytes: usize, trailer_end: usize) -> Result<DbiaSection, String> {
    if dbia_offset % PAGE != 0 {
        return Err(format!("dbia: dbia_offset {dbia_offset} not page-aligned"));
    }
    let min_end = dbia_offset.checked_add(dbia_bytes).and_then(|x| x.checked_add(DBIA_TRAILER_BYTES)).ok_or("dbia: dbia_offset + dbia_bytes overflows")?;
    if min_end > trailer_end || trailer_end % PAGE != 0 {
        return Err(format!(
            "dbia: section [{dbia_offset}, +{dbia_bytes}] + trailer does not fit the \
             page-aligned region ending at {trailer_end}"
        ));
    }
    if dbia_bytes < DBIA_HEADER_BYTES {
        return Err("dbia: section shorter than the 32-byte header".into());
    }
    // padding between section end and the trailer must be zero (byte-stability)
    if buf[dbia_offset + dbia_bytes..trailer_end - DBIA_TRAILER_BYTES].iter().any(|&b| b != 0) {
        return Err("dbia: nonzero bytes in section padding".into());
    }

    let v2 = read_strand_v2_header(buf)?;

    let s = &buf[dbia_offset..dbia_offset + dbia_bytes];
    if &s[0..4] != &DBIA_MAGIC[..] {
        return Err("dbia: bad section header magic".into());
    }
    let version = u32::from_le_bytes(s[4..8].try_into().unwrap());
    if version != DBIA_VERSION {
        return Err(format!("dbia: version {version} != {DBIA_VERSION}"));
    }
    let n_tensors = u32::from_le_bytes(s[8..12].try_into().unwrap()) as usize;
    if n_tensors != v2.tensors.len() {
        return Err(format!("dbia: section n_tensors {n_tensors} != archive's {}", v2.tensors.len()));
    }
    let flags = u32::from_le_bytes(s[12..16].try_into().unwrap());
    if flags != 0 {
        return Err(format!("dbia: reserved flag bits set: {flags:#x}"));
    }
    if s[16..32].iter().any(|&b| b != 0) {
        return Err("dbia: header reserved bytes not zero".into());
    }

    let mut p = DBIA_HEADER_BYTES;
    let take = |p: &mut usize, n: usize| -> Result<&[u8], String> {
        let end = p.checked_add(n).filter(|&e| e <= s.len()).ok_or("dbia: section truncated")?;
        let sl = &s[*p..end];
        *p = end;
        Ok(sl)
    };

    let mut tensors = Vec::with_capacity(n_tensors);
    for (i, desc) in v2.tensors.iter().enumerate() {
        let len = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap()) as usize;
        let reserved = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap());
        if reserved != 0 {
            return Err(format!("dbia: tensor record {i}: reserved field not zero"));
        }
        if len == 0 {
            tensors.push(None);
            continue;
        }
        let out = out_features_of(&desc.shape).unwrap_or(0);
        if len != out {
            return Err(format!("dbia: tensor record {i}: len {len} != tensor out_features {out}"));
        }
        let raw = take(&mut p, len * 2)?;
        let mut c_bits = Vec::with_capacity(len);
        for chunk in raw.chunks_exact(2) {
            c_bits.push(u16::from_le_bytes([chunk[0], chunk[1]]));
        }
        tensors.push(Some(DebiasWire { c_bits }));
    }
    if p != dbia_bytes {
        return Err(format!("dbia: {} trailing bytes after the last record", dbia_bytes - p));
    }
    Ok(DbiaSection { tensors })
}

/// Read the DBIA section from a buffer, walking back through any SPRV/OUTL/RSLT
/// trailers that sit on top of it. Returns `Ok(None)` if there is no DBIA section.
/// With `strict = true` a present-but-corrupt DBIA trailer is an `Err`; with
/// `strict = false` it degrades to `Ok(None)`.
pub fn read_dbia_bytes(buf: &[u8], strict: bool) -> Result<Option<DbiaSection>, String> {
    let mut end = buf.len();
    // Bounded walk: at most a few stacked sections (OUTL/DBIA/RSLT/SPRV). 6 is slack.
    for _ in 0..6 {
        if end < DBIA_TRAILER_BYTES {
            return Ok(None);
        }
        let t = &buf[end - DBIA_TRAILER_BYTES..end];
        let magic = &t[12..16];
        if magic == &DBIA_MAGIC[..] {
            let parse = (|| -> Result<DbiaSection, String> {
                let dbia_offset = u64::from_le_bytes(t[0..8].try_into().unwrap());
                let dbia_bytes = u32::from_le_bytes(t[8..12].try_into().unwrap());
                let dbia_offset: usize = dbia_offset.try_into().map_err(|_| "dbia: dbia_offset exceeds address space".to_string())?;
                parse_dbia_section(buf, dbia_offset, dbia_bytes as usize, end)
            })();
            return match parse {
                Ok(s) => Ok(Some(s)),
                Err(e) if strict => Err(e),
                Err(_) => Ok(None),
            };
        } else if magic == &SPRV_MAGIC[..] || magic == &OUTL_MAGIC[..] || magic == &RSLT_MAGIC[..] {
            // Every stacked section's trailer stores its own section start at [0..8].
            // Step back to it and keep looking for DBIA underneath.
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

/// Read the DBIA section from a file path.
pub fn read_dbia(path: impl AsRef<Path>) -> Result<Option<DbiaSection>, String> {
    let path = path.as_ref();
    let buf = fs::read(path).map_err(|e| format!("dbia: read {path:?}: {e}"))?;
    read_dbia_bytes(&buf, true)
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
        std::env::temp_dir().join(format!("strand-dbia-{tag}-{}-{}.strand", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed)))
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

    // Two tensors: q_proj [4,256] (out=4), down_proj [3,300] (out=3).
    fn build_test_archive() -> Vec<u8> {
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc_a = encode_tensor_with(&test_weights(1024, 11), &cfg, &EncodeOpts::default());
        let enc_b = encode_tensor_with(&test_weights(900, 23), &cfg, &EncodeOpts::default());
        let shape_a = [4u64, 256u64];
        let shape_b = [3u64, 300u64];
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
        write_strand_v2(&tensors, [9u8; 32], false).expect("write v2")
    }

    fn sample_wires() -> Vec<Option<DebiasWire>> {
        vec![
            // out=4 correction with a representative spread of magnitudes/signs
            Some(DebiasWire::from_f32(&[1.5e-3, -2.0e-4, 0.0, 7.125e-2])),
            None, // down_proj carries no correction
        ]
    }

    #[test]
    fn f32_bf16_round_trip_is_top16_and_ties_even() {
        // exact bf16-representable values survive byte-exact
        for &v in &[0.0f32, 1.0, -1.0, 0.5, 2.0, -0.25, 7.125e-2_f32] {
            let b = f32_to_bf16_round(v);
            // top 16 bits of an exactly-representable value, possibly +1 from rounding
            assert_eq!(bf16_to_f32(b).to_bits() >> 16, f32_to_bf16_round(bf16_to_f32(b)) as u32, "bf16 must be a fixed point of the round");
        }
        // ties-to-even: 0x0000_8000 (exactly half a bf16 ulp above an even mantissa)
        // rounds DOWN (stays even); one ulp up rounds to even as well.
        let down = f32::from_bits(0x3f80_8000); // 1.0 + 0.5ulp, even low bit -> round down
        assert_eq!(f32_to_bf16_round(down), 0x3f80);
        let up = f32::from_bits(0x3f81_8000); // 1.0+1.5ulp, odd low bit -> round up to even
        assert_eq!(f32_to_bf16_round(up), 0x3f82);
        // NaN/Inf preserve their top half
        assert_eq!(f32_to_bf16_round(f32::INFINITY), (f32::INFINITY.to_bits() >> 16) as u16);
        assert!(bf16_to_f32(f32_to_bf16_round(f32::NAN)).is_nan());
    }

    #[test]
    fn dbia_round_trip_and_v2_reader_compat() {
        let buf = build_test_archive();
        let path = tmp_path("roundtrip");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();

        assert_eq!(read_dbia(&path).unwrap(), None, "plain v2 must read as absent");

        let wires = sample_wires();
        append_dbia(&path, &wires).expect("append dbia");

        let back = read_dbia(&path).unwrap().expect("section found");
        assert_eq!(back.tensors, wires, "DBIA must round-trip exactly");
        let w0 = back.tensors[0].as_ref().unwrap();
        assert_eq!(w0.c_bits.len(), 4);
        // decode matches the bf16 round of the inputs
        let got: Vec<u32> = w0.dequant().map(|v| v.to_bits()).collect();
        let want: Vec<u32> = [1.5e-3f32, -2.0e-4, 0.0, 7.125e-2].iter().map(|&v| bf16_to_f32(f32_to_bf16_round(v)).to_bits()).collect();
        assert_eq!(got, want);
        assert!(back.tensors[1].is_none());
        assert_eq!(back.n_with_correction(), 1);
        assert_eq!(back.total_entries(), 4);

        // append must not touch the v2 bytes, and the v2 reader still works under DBIA
        let trailered = std::fs::read(&path).unwrap();
        assert_eq!(&trailered[..buf.len()], &buf[..], "append must not touch v2 bytes");
        assert_eq!(trailered.len() % PAGE, 0, "DBIA end must be page-aligned (stacking)");
        let full = read_strand_v2(&trailered).expect("full v2 read under DBIA trailer");
        assert_eq!(full.len(), 2);

        // double-append is rejected and the file is left untouched
        let before = std::fs::read(&path).unwrap();
        assert!(append_dbia(&path, &wires).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    #[test]
    fn append_is_byte_deterministic() {
        // Two independent appends of the same wires onto the same base produce
        // byte-identical files — the section is fully deterministic.
        let buf = build_test_archive();
        let p1 = tmp_path("det1");
        let p2 = tmp_path("det2");
        let _g1 = TmpFile(p1.clone());
        let _g2 = TmpFile(p2.clone());
        std::fs::write(&p1, &buf).unwrap();
        std::fs::write(&p2, &buf).unwrap();
        append_dbia(&p1, &sample_wires()).unwrap();
        append_dbia(&p2, &sample_wires()).unwrap();
        assert_eq!(std::fs::read(&p1).unwrap(), std::fs::read(&p2).unwrap());
    }

    #[test]
    fn dbia_validates_shape_and_count() {
        let buf = build_test_archive();
        let path = tmp_path("validate");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();

        // wrong record count
        assert!(append_dbia(&path, &sample_wires()[..1]).is_err());
        // wrong out_features (q_proj is out=4, not 5)
        let mut w = sample_wires();
        w[0] = Some(DebiasWire::from_f32(&[0.0, 1.0, 2.0, 3.0, 4.0]));
        assert!(append_dbia(&path, &w).is_err());
        // Some(empty) is ambiguous
        let mut w = sample_wires();
        w[0] = Some(DebiasWire { c_bits: vec![] });
        assert!(append_dbia(&path, &w).is_err());

        assert_eq!(std::fs::read(&path).unwrap(), buf, "rejected appends leave file intact");
    }

    #[test]
    fn dbia_corrupt_trailer_is_error_not_crash() {
        let buf = build_test_archive();
        let path = tmp_path("corrupt");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();
        append_dbia(&path, &sample_wires()).expect("append");
        let clean = std::fs::read(&path).unwrap();

        // corrupt the trailer's byte-count field
        let mut c1 = clean.clone();
        let pb_pos = c1.len() - 8;
        c1[pb_pos] ^= 0xFF;
        assert!(read_dbia_bytes(&c1, true).is_err());
        assert_eq!(read_dbia_bytes(&c1, false).unwrap(), None, "lenient degrades to None");

        // corrupt a header byte inside the section
        let dbia_off = {
            let t = &clean[clean.len() - DBIA_TRAILER_BYTES..];
            u64::from_le_bytes(t[0..8].try_into().unwrap()) as usize
        };
        let mut c2 = clean.clone();
        c2[dbia_off + 4] ^= 0xFF; // version field
        assert!(read_dbia_bytes(&c2, true).is_err());

        // degenerate inputs do not panic
        assert_eq!(read_dbia_bytes(b"", true).unwrap(), None);
        assert_eq!(read_dbia_bytes(b"DBIA", true).unwrap(), None);
        let mut tiny = vec![0u8; DBIA_TRAILER_BYTES];
        tiny[12..].copy_from_slice(DBIA_MAGIC);
        assert!(read_dbia_bytes(&tiny, true).is_err());
        assert_eq!(read_dbia_bytes(&tiny, false).unwrap(), None);
    }

    /// The decode-side apply, in isolation (the reference for the MAC epilogue add).
    /// `y[o] += bf16_to_f32(c_bits[o])`, one add per output row, after the inner product.
    fn apply_debias_epilogue(y: &mut [f32], wire: &DebiasWire) {
        debug_assert_eq!(y.len(), wire.c_bits.len());
        for (yo, &cb) in y.iter_mut().zip(wire.c_bits.iter()) {
            *yo += bf16_to_f32(cb);
        }
    }

    #[test]
    fn apply_is_deterministic_and_matches_dequant() {
        let wire = DebiasWire::from_f32(&[1.5e-3, -2.0e-4, 0.0, 7.125e-2]);
        let base = [0.1f32, -0.2, 0.3, -0.4];

        let mut y1 = base;
        apply_debias_epilogue(&mut y1, &wire);
        let mut y2 = base;
        apply_debias_epilogue(&mut y2, &wire);
        // bit-identical across runs (single deterministic add, no accumulation order)
        let b1: Vec<u32> = y1.iter().map(|v| v.to_bits()).collect();
        let b2: Vec<u32> = y2.iter().map(|v| v.to_bits()).collect();
        assert_eq!(b1, b2);

        // matches base + dequant(c), computed the documented way
        let c: Vec<f32> = wire.dequant().collect();
        for o in 0..4 {
            assert_eq!(y1[o].to_bits(), (base[o] + c[o]).to_bits());
        }
    }
}

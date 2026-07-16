use std::fs;
use std::io::Write as _;
use std::path::Path;

use crate::format::{read_strand_v2_header, PAGE};

pub const OUTL_MAGIC: &[u8; 4] = b"OUTL";

const SPRV_MAGIC: &[u8; 4] = b"SPRV";

/// SDSQ (sprint Lever 1 side-info rANS) is appended ABOVE OUTL and below the SPRV
/// seal, so the OUTL trailer-chain walk must step over an SDSQ trailer to find OUTL
/// beneath it — exactly as it already steps over SPRV. Without this, an SDSQ-on-top
/// archive would make `read_outl_bytes` halt on the unknown magic and silently
/// report OUTL as absent (a correctness break, not a missing feature).
const SDSQ_MAGIC: &[u8; 4] = b"SDSQ";

pub const OUTL_VERSION: u32 = 1;

/// OUTL section header flag: the outlier **positions** are NOT bit-packed inline
/// (`idx_bits` per entry); instead each tensor's sorted positions are gap-coded
/// with the [`crate::c2_final`] entropy coder and appended after that tensor's
/// value-only packed codes. The per-record `idx_bits` field is `0` as a sentinel.
///
/// This is the C2F outlier-position lever (the largest measured side-info channel,
/// ~0.148 bpw on Qwen2.5-0.5B q2): inline positions cost `idx_bits` (~0.226 bpw)
/// where the gap distribution entropy-codes to ~0.083 bpw. It is **container-only**:
/// the decoded positions are byte-identical to the inline ones, so `OutlierWire`
/// (and therefore the sparse-outlier MAC and SPRV block hashes) are unchanged.
/// `read_outl_bytes` reconstructs full `entries` either way, so no downstream
/// reader (loader, decode, provenance) needs to know which encoding was used.
pub const OUTL_FLAG_POS_RANS: u32 = 1 << 0;

pub const OUTL_HEADER_BYTES: usize = 32;

pub const OUTL_TRAILER_BYTES: usize = 16;

pub const OUTL_RECORD_FIXED_BYTES: usize = 24;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutlierWire {
    pub omax_bits: u32,

    pub entries: Vec<(u32, i32)>,

    pub idx_bits: u32,

    pub val_bits: u32,
}

#[inline]
pub fn idx_bits_for(n: usize) -> u32 {
    if n <= 1 {
        1
    } else {
        usize::BITS - (n - 1).leading_zeros()
    }
}

impl OutlierWire {
    pub fn from_selection(n_total: usize, idx: Vec<usize>, codes: Vec<i32>, omax: f32, val_bits: u32) -> Self {
        debug_assert_eq!(idx.len(), codes.len());
        let mut entries: Vec<(u32, i32)> = idx.into_iter().map(|i| i as u32).zip(codes).collect();
        entries.sort_unstable_by_key(|&(i, _)| i);
        OutlierWire { omax_bits: omax.to_bits(), entries, idx_bits: idx_bits_for(n_total), val_bits: val_bits.clamp(2, 16) }
    }

    pub fn wire_bytes(&self) -> u64 {
        12 + (self.entries.len() as u64 * (self.idx_bits + self.val_bits) as u64).div_ceil(8)
    }

    pub fn dequant_vals(&self) -> impl Iterator<Item = (u32, f32)> + '_ {
        let omax = f32::from_bits(self.omax_bits);
        let levels = ((1i64 << (self.val_bits - 1)) - 1) as f32;
        self.entries.iter().map(move |&(i, c)| (i, (c as f32) / levels * omax))
    }
}

#[inline]
fn write_bits(out: &mut Vec<u8>, bit_cursor: &mut usize, value: u64, nbits: u32) {
    for i in 0..nbits as usize {
        let bit = ((value >> i) & 1) as u8;
        let byte_idx = (*bit_cursor + i) >> 3;
        let in_byte = (*bit_cursor + i) & 7;
        if byte_idx >= out.len() {
            out.push(0);
        }
        out[byte_idx] |= bit << in_byte;
    }
    *bit_cursor += nbits as usize;
}

#[inline]
fn read_bits_u64(bytes: &[u8], start_bit: usize, nbits: u32) -> u64 {
    let mut acc = 0u64;
    for i in 0..nbits as usize {
        let bit_idx = start_bit + i;
        let byte_idx = bit_idx >> 3;
        let in_byte = bit_idx & 7;
        let bit = if byte_idx < bytes.len() { ((bytes[byte_idx] >> in_byte) & 1) as u64 } else { 0 };
        acc |= bit << i;
    }
    acc
}

#[inline]
fn sign_extend(v: u64, nbits: u32) -> i32 {
    let shift = 64 - nbits;
    (((v << shift) as i64) >> shift) as i32
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutlSection {
    pub tensors: Vec<Option<OutlierWire>>,
}

impl OutlSection {
    pub fn n_with_channel(&self) -> usize {
        self.tensors.iter().filter(|t| t.is_some()).count()
    }

    pub fn total_entries(&self) -> usize {
        self.tensors.iter().filter_map(|t| t.as_ref().map(|w| w.entries.len())).sum()
    }
}

#[inline]
fn page_align(x: usize) -> usize {
    (x + PAGE - 1) & !(PAGE - 1)
}

fn outl_section_bytes(wires: &[Option<OutlierWire>], totals: &[usize], pos_rans: bool) -> Result<Vec<u8>, String> {
    debug_assert_eq!(wires.len(), totals.len());
    let mut o = Vec::new();
    o.extend_from_slice(OUTL_MAGIC);
    o.extend_from_slice(&OUTL_VERSION.to_le_bytes());
    o.extend_from_slice(&(wires.len() as u32).to_le_bytes());
    let flags = if pos_rans { OUTL_FLAG_POS_RANS } else { 0 };
    o.extend_from_slice(&flags.to_le_bytes());
    o.extend_from_slice(&[0u8; 16]);
    debug_assert_eq!(o.len(), OUTL_HEADER_BYTES);

    for (i, (w, &total)) in wires.iter().zip(totals.iter()).enumerate() {
        match w {
            None => {
                o.extend_from_slice(&0u64.to_le_bytes());
                o.extend_from_slice(&0u32.to_le_bytes());
                o.extend_from_slice(&0u32.to_le_bytes());
                o.extend_from_slice(&0u32.to_le_bytes());
                o.extend_from_slice(&0u32.to_le_bytes());
            }
            Some(w) => {
                if w.idx_bits == 0 || w.idx_bits > 32 {
                    return Err(format!("outl: tensor record {i}: idx_bits {} out of range", w.idx_bits));
                }
                if !(2..=16).contains(&w.val_bits) {
                    return Err(format!("outl: tensor record {i}: val_bits {} out of range", w.val_bits));
                }
                let levels = (1i64 << (w.val_bits - 1)) - 1;
                let mut prev: Option<u32> = None;
                for &(idx, code) in &w.entries {
                    if idx as usize >= total {
                        return Err(format!("outl: tensor record {i}: index {idx} out of range ({total} weights)"));
                    }
                    if let Some(p) = prev {
                        if idx <= p {
                            return Err(format!("outl: tensor record {i}: indices must be strictly ascending"));
                        }
                    }
                    if (code as i64) < -levels || (code as i64) > levels {
                        return Err(format!("outl: tensor record {i}: code {code} does not fit val_bits {}", w.val_bits));
                    }
                    prev = Some(idx);
                }
                o.extend_from_slice(&(w.entries.len() as u64).to_le_bytes());
                o.extend_from_slice(&w.omax_bits.to_le_bytes());
                // POS_RANS: store idx_bits=0 sentinel; positions live in the rANS
                // stream appended after the value-only packed codes.
                let stored_idx_bits = if pos_rans { 0 } else { w.idx_bits };
                o.extend_from_slice(&stored_idx_bits.to_le_bytes());
                o.extend_from_slice(&w.val_bits.to_le_bytes());
                o.extend_from_slice(&0u32.to_le_bytes());

                if pos_rans {
                    // value-only packed codes (no interleaved idx)
                    let mut packed: Vec<u8> = Vec::with_capacity((w.entries.len() * w.val_bits as usize).div_ceil(8));
                    let mut cursor = 0usize;
                    for &(_, code) in &w.entries {
                        write_bits(&mut packed, &mut cursor, (code as u32 as u64) & ((1u64 << w.val_bits) - 1), w.val_bits);
                    }
                    o.extend_from_slice(&packed);
                    // gap-coded position stream (entries are already ascending)
                    let positions: Vec<u32> = w.entries.iter().map(|&(idx, _)| idx).collect();
                    let pstream = crate::c2_final::encode_positions(&positions);
                    let plen: u32 = pstream.len().try_into().map_err(|_| format!("outl: tensor record {i}: position stream exceeds u32"))?;
                    o.extend_from_slice(&plen.to_le_bytes());
                    o.extend_from_slice(&pstream);
                } else {
                    let mut packed: Vec<u8> = Vec::with_capacity((w.entries.len() * (w.idx_bits + w.val_bits) as usize).div_ceil(8));
                    let mut cursor = 0usize;
                    for &(idx, code) in &w.entries {
                        write_bits(&mut packed, &mut cursor, idx as u64, w.idx_bits);
                        write_bits(&mut packed, &mut cursor, (code as u32 as u64) & ((1u64 << w.val_bits) - 1), w.val_bits);
                    }
                    o.extend_from_slice(&packed);
                }
            }
        }
    }
    Ok(o)
}

pub fn append_outl(path: impl AsRef<Path>, wires: &[Option<OutlierWire>]) -> Result<(), String> {
    append_outl_inner(path, wires, false)
}

/// Like [`append_outl`] but gap-codes each tensor's outlier positions with the
/// C2F entropy coder instead of bit-packing them inline (`OUTL_FLAG_POS_RANS`).
/// Container-only: `read_outl_bytes` reconstructs byte-identical `entries`, so the
/// decode/MAC/SPRV path is unchanged — only the on-disk position footprint shrinks.
pub fn append_outl_c2f(path: impl AsRef<Path>, wires: &[Option<OutlierWire>]) -> Result<(), String> {
    append_outl_inner(path, wires, true)
}

fn append_outl_inner(path: impl AsRef<Path>, wires: &[Option<OutlierWire>], pos_rans: bool) -> Result<(), String> {
    let path = path.as_ref();
    let buf = fs::read(path).map_err(|e| format!("outl: read {path:?}: {e}"))?;

    if buf.len() >= OUTL_TRAILER_BYTES && &buf[buf.len() - 4..] == &SPRV_MAGIC[..] {
        return Err("outl: file already has an SPRV trailer — OUTL must be appended BEFORE SPRV \
             (sections stack as OUTL then SPRV, SPRV outermost)"
            .into());
    }
    match read_outl_bytes(&buf, true) {
        Ok(Some(_)) => return Err("outl: file already has an OUTL section (double-append rejected)".into()),
        Err(e) => {
            return Err(format!(
                "outl: file ends in OUTL magic but the section is invalid — refusing to \
                 append a second trailer behind it: {e}"
            ))
        }
        Ok(None) => {}
    }

    let hdr = read_strand_v2_header(&buf)?;
    if wires.len() != hdr.tensors.len() {
        return Err(format!("outl: {} wire records, archive has {} tensors", wires.len(), hdr.tensors.len()));
    }
    let totals: Vec<usize> = hdr.tensors.iter().map(|t| t.total).collect();
    let section = outl_section_bytes(wires, &totals, pos_rans)?;
    let outl_bytes: u32 = section.len().try_into().map_err(|_| format!("outl: section is {} bytes — exceeds the u32 field", section.len()))?;

    let outl_offset = page_align(buf.len());
    let lead_pad = outl_offset - buf.len();

    let end = page_align(outl_offset + section.len() + OUTL_TRAILER_BYTES);
    let tail_pad = end - OUTL_TRAILER_BYTES - outl_offset - section.len();

    let mut tail = Vec::with_capacity(lead_pad + section.len() + tail_pad + OUTL_TRAILER_BYTES);
    tail.resize(lead_pad, 0u8);
    tail.extend_from_slice(&section);
    tail.resize(tail.len() + tail_pad, 0u8);
    tail.extend_from_slice(&(outl_offset as u64).to_le_bytes());
    tail.extend_from_slice(&outl_bytes.to_le_bytes());
    tail.extend_from_slice(OUTL_MAGIC);

    let mut f = fs::OpenOptions::new().append(true).open(path).map_err(|e| format!("outl: open {path:?} for append: {e}"))?;
    f.write_all(&tail).map_err(|e| format!("outl: append to {path:?}: {e}"))?;
    Ok(())
}

fn parse_outl_section(buf: &[u8], outl_offset: usize, outl_bytes: usize, trailer_end: usize) -> Result<OutlSection, String> {
    if outl_offset % PAGE != 0 {
        return Err(format!("outl: outl_offset {outl_offset} not page-aligned"));
    }

    let min_end = outl_offset.checked_add(outl_bytes).and_then(|x| x.checked_add(OUTL_TRAILER_BYTES)).ok_or("outl: outl_offset + outl_bytes overflows")?;
    if min_end > trailer_end || trailer_end % PAGE != 0 {
        return Err(format!(
            "outl: section [{outl_offset}, +{outl_bytes}] + trailer does not fit the \
             page-aligned region ending at {trailer_end}"
        ));
    }
    if outl_bytes < OUTL_HEADER_BYTES {
        return Err("outl: section shorter than the 32-byte header".into());
    }

    if buf[outl_offset + outl_bytes..trailer_end - OUTL_TRAILER_BYTES].iter().any(|&b| b != 0) {
        return Err("outl: nonzero bytes in section padding".into());
    }

    let v2 = read_strand_v2_header(buf)?;

    let s = &buf[outl_offset..outl_offset + outl_bytes];
    if &s[0..4] != &OUTL_MAGIC[..] {
        return Err("outl: bad section header magic".into());
    }
    let version = u32::from_le_bytes(s[4..8].try_into().unwrap());
    if version != OUTL_VERSION {
        return Err(format!("outl: version {version} != {OUTL_VERSION}"));
    }
    let n_tensors = u32::from_le_bytes(s[8..12].try_into().unwrap()) as usize;
    if n_tensors != v2.tensors.len() {
        return Err(format!("outl: section n_tensors {n_tensors} != archive's {}", v2.tensors.len()));
    }
    let flags = u32::from_le_bytes(s[12..16].try_into().unwrap());
    if flags & !OUTL_FLAG_POS_RANS != 0 {
        return Err(format!("outl: unknown flag bits set: {flags:#x}"));
    }
    let pos_rans = flags & OUTL_FLAG_POS_RANS != 0;
    if s[16..32].iter().any(|&b| b != 0) {
        return Err("outl: header reserved bytes not zero".into());
    }

    let mut p = OUTL_HEADER_BYTES;
    let take = |p: &mut usize, n: usize| -> Result<&[u8], String> {
        let end = p.checked_add(n).filter(|&e| e <= s.len()).ok_or("outl: section truncated")?;
        let sl = &s[*p..end];
        *p = end;
        Ok(sl)
    };

    let mut tensors = Vec::with_capacity(n_tensors);
    for (i, desc) in v2.tensors.iter().enumerate() {
        let count = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap()) as usize;
        let omax_bits = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap());
        let idx_bits = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap());
        let val_bits = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap());
        let reserved = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap());
        if reserved != 0 {
            return Err(format!("outl: tensor record {i}: reserved field not zero"));
        }
        if count == 0 {
            if omax_bits != 0 || idx_bits != 0 || val_bits != 0 {
                return Err(format!("outl: tensor record {i}: count 0 but nonzero channel fields"));
            }
            tensors.push(None);
            continue;
        }
        // POS_RANS records store idx_bits=0 (positions live in the rANS stream);
        // legacy records require a real idx_bits in [1,32].
        if pos_rans {
            if idx_bits != 0 {
                return Err(format!("outl: tensor record {i}: POS_RANS section but idx_bits {idx_bits} != 0 sentinel"));
            }
        } else if idx_bits == 0 || idx_bits > 32 {
            return Err(format!("outl: tensor record {i}: idx_bits {idx_bits} out of range"));
        }
        if !(2..=16).contains(&val_bits) {
            return Err(format!("outl: tensor record {i}: val_bits {val_bits} out of range"));
        }
        if count > desc.total {
            return Err(format!("outl: tensor record {i}: count {count} exceeds tensor total {}", desc.total));
        }

        let effective_idx_bits = if pos_rans { idx_bits_for(desc.total) } else { idx_bits };
        let entries = if pos_rans {
            // value-only packed codes, then a gap-coded position rANS stream
            let val_bytes = (count * val_bits as usize).div_ceil(8);
            let packed = take(&mut p, val_bytes)?;
            let mut codes = Vec::with_capacity(count);
            let mut cursor = 0usize;
            for _ in 0..count {
                let code = sign_extend(read_bits_u64(packed, cursor, val_bits), val_bits);
                cursor += val_bits as usize;
                codes.push(code);
            }
            if cursor % 8 != 0 {
                let pad = read_bits_u64(packed, cursor, (8 - (cursor % 8)) as u32);
                if pad != 0 {
                    return Err(format!("outl: tensor record {i}: nonzero packed pad bits"));
                }
            }
            let plen = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap()) as usize;
            let pstream = take(&mut p, plen)?;
            let mut ppos = 0usize;
            let positions = crate::c2_final::decode_positions(pstream, &mut ppos).map_err(|e| format!("outl: tensor record {i}: position stream: {e}"))?;
            if ppos != pstream.len() {
                return Err(format!("outl: tensor record {i}: {} trailing bytes in position stream", pstream.len() - ppos));
            }
            if positions.len() != count {
                return Err(format!("outl: tensor record {i}: decoded {} positions, count is {count}", positions.len()));
            }
            let mut entries = Vec::with_capacity(count);
            let mut prev: Option<u32> = None;
            for (idx, code) in positions.into_iter().zip(codes.into_iter()) {
                if idx as usize >= desc.total {
                    return Err(format!("outl: tensor record {i}: index {idx} out of range ({} weights)", desc.total));
                }
                if let Some(pv) = prev {
                    if idx <= pv {
                        return Err(format!("outl: tensor record {i}: positions not strictly ascending"));
                    }
                }
                prev = Some(idx);
                entries.push((idx, code));
            }
            entries
        } else {
            let packed_bytes = (count * (idx_bits + val_bits) as usize).div_ceil(8);
            let packed = take(&mut p, packed_bytes)?;
            let mut entries = Vec::with_capacity(count);
            let mut prev: Option<u32> = None;
            let mut cursor = 0usize;
            for _ in 0..count {
                let idx = read_bits_u64(packed, cursor, idx_bits) as u32;
                cursor += idx_bits as usize;
                let code = sign_extend(read_bits_u64(packed, cursor, val_bits), val_bits);
                cursor += val_bits as usize;
                if idx as usize >= desc.total {
                    return Err(format!("outl: tensor record {i}: index {idx} out of range ({} weights)", desc.total));
                }
                if let Some(pv) = prev {
                    if idx <= pv {
                        return Err(format!("outl: tensor record {i}: indices not strictly ascending"));
                    }
                }
                prev = Some(idx);
                entries.push((idx, code));
            }
            if cursor % 8 != 0 {
                let pad = read_bits_u64(packed, cursor, (8 - (cursor % 8)) as u32);
                if pad != 0 {
                    return Err(format!("outl: tensor record {i}: nonzero packed pad bits"));
                }
            }
            entries
        };
        // Reconstruct the in-memory OutlierWire with the *natural* idx_bits so a
        // POS_RANS archive's OutlierWire is byte-identical to the inline one (the
        // sentinel 0 on disk is an encoding detail, not part of the logical value).
        tensors.push(Some(OutlierWire { omax_bits, entries, idx_bits: effective_idx_bits, val_bits }));
    }
    if p != outl_bytes {
        return Err(format!("outl: {} trailing bytes after the last record", outl_bytes - p));
    }
    Ok(OutlSection { tensors })
}

pub fn read_outl_bytes(buf: &[u8], strict: bool) -> Result<Option<OutlSection>, String> {
    let mut end = buf.len();

    for _ in 0..4 {
        if end < OUTL_TRAILER_BYTES {
            return Ok(None);
        }
        let t = &buf[end - OUTL_TRAILER_BYTES..end];
        let magic = &t[12..16];
        if magic == &OUTL_MAGIC[..] {
            let parse = (|| -> Result<OutlSection, String> {
                let outl_offset = u64::from_le_bytes(t[0..8].try_into().unwrap());
                let outl_bytes = u32::from_le_bytes(t[8..12].try_into().unwrap());
                let outl_offset: usize = outl_offset.try_into().map_err(|_| "outl: outl_offset exceeds address space".to_string())?;
                parse_outl_section(buf, outl_offset, outl_bytes as usize, end)
            })();
            return match parse {
                Ok(s) => Ok(Some(s)),
                Err(e) if strict => Err(e),
                Err(_) => Ok(None),
            };
        } else if magic == &SPRV_MAGIC[..] || magic == &SDSQ_MAGIC[..] {
            // Both the SPRV seal and an SDSQ side-info section can sit above OUTL.
            // Each stores its own section start at trailer[0..8]; step back to it.
            let prov_offset = u64::from_le_bytes(t[0..8].try_into().unwrap());
            let Ok(prov_offset) = usize::try_from(prov_offset) else {
                return Ok(None);
            };
            if prov_offset >= end {
                return Ok(None);
            }
            end = prov_offset;
        } else {
            return Ok(None);
        }
    }
    Ok(None)
}

pub fn read_outl(path: impl AsRef<Path>) -> Result<Option<OutlSection>, String> {
    let path = path.as_ref();
    let buf = fs::read(path).map_err(|e| format!("outl: read {path:?}: {e}"))?;
    read_outl_bytes(&buf, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{encode_tensor_with, EncodeOpts};
    use crate::format::{read_strand_v2, write_strand_v2, PackedTensor, PackedTensorV2};
    use crate::provenance_io::{append_sprv_computed, read_sprv, verify_archive, VerifyDepth};
    use crate::trellis::TrellisConfig;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn tmp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("strand-outl-{tag}-{}-{}.strand", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed)))
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

    fn build_test_archive() -> (Vec<u8>, TrellisConfig) {
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc_a = encode_tensor_with(&test_weights(1024, 11), &cfg, &EncodeOpts::default());
        let enc_b = encode_tensor_with(&test_weights(900, 23), &cfg, &EncodeOpts::default());
        let shape_a = [4u64, 256u64];
        let shape_b = [900u64];
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
        let buf = write_strand_v2(&tensors, [9u8; 32], true).expect("write v2");
        (buf, cfg)
    }

    fn sample_wires() -> Vec<Option<OutlierWire>> {
        vec![Some(OutlierWire::from_selection(1024, vec![700, 3, 511], vec![-127, 5, 127], 0.3125f32, 8)), None]
    }

    #[test]
    fn outl_round_trip_and_v2_reader_compat() {
        let (buf, _cfg) = build_test_archive();
        let path = tmp_path("roundtrip");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();

        assert_eq!(read_outl(&path).unwrap(), None, "plain v2 must read as absent");

        let wires = sample_wires();
        append_outl(&path, &wires).expect("append outl");

        let back = read_outl(&path).unwrap().expect("section found");

        let w0 = back.tensors[0].as_ref().unwrap();
        assert_eq!(w0.entries, vec![(3, 5), (511, 127), (700, -127)]);
        assert_eq!(w0.idx_bits, 10);
        assert_eq!(w0.val_bits, 8);
        assert_eq!(w0.omax_bits, 0.3125f32.to_bits());
        assert!(back.tensors[1].is_none());
        assert_eq!(back.tensors, wires);

        let trailered = std::fs::read(&path).unwrap();
        assert_eq!(&trailered[..buf.len()], &buf[..], "append must not touch v2 bytes");
        assert_eq!(trailered.len() % PAGE, 0, "OUTL end must be page-aligned (stacking)");
        let h0 = crate::format::read_strand_v2_header(&buf).unwrap();
        let h1 = crate::format::read_strand_v2_header(&trailered).unwrap();
        assert_eq!(h0.tensors.len(), h1.tensors.len());
        let full = read_strand_v2(&trailered).expect("full v2 read under OUTL trailer");
        assert_eq!(full.len(), 2);

        let before = std::fs::read(&path).unwrap();
        assert!(append_outl(&path, &wires).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    #[test]
    fn outl_c2f_pos_rans_reconstruct_is_byte_identical_to_inline() {
        // The C2F outlier-position lever (OUTL_FLAG_POS_RANS) is container-only:
        // gap-coding positions must reconstruct an OutlSection byte-identical to the
        // inline (bit-packed idx) encoding. This is the make-or-break invariant —
        // the sparse-outlier MAC and SPRV hashes read `entries`, so they MUST match.
        let (buf, _cfg) = build_test_archive();

        let legacy_path = tmp_path("c2f-legacy");
        let c2f_path = tmp_path("c2f-packed");
        let _g1 = TmpFile(legacy_path.clone());
        let _g2 = TmpFile(c2f_path.clone());
        std::fs::write(&legacy_path, &buf).unwrap();
        std::fs::write(&c2f_path, &buf).unwrap();

        let wires = sample_wires();
        append_outl(&legacy_path, &wires).expect("append legacy outl");
        append_outl_c2f(&c2f_path, &wires).expect("append c2f outl");

        let legacy = read_outl(&legacy_path).unwrap().expect("legacy section");
        let c2f = read_outl(&c2f_path).unwrap().expect("c2f section");

        // The reconstructed OutlSection (entries, idx_bits, val_bits, omax) must be
        // byte-identical — i.e. equal to the source wires too.
        assert_eq!(c2f, legacy, "C2F-pos reconstruct != inline reconstruct");
        assert_eq!(c2f.tensors, wires, "C2F-pos reconstruct != source wires");

        // And the on-disk encoding must actually differ (flag set, smaller or
        // different bytes) — prove the lever is live, not a no-op alias.
        let legacy_bytes = std::fs::read(&legacy_path).unwrap();
        let c2f_bytes = std::fs::read(&c2f_path).unwrap();
        let lt = &legacy_bytes[legacy_bytes.len() - OUTL_TRAILER_BYTES..];
        let ct = &c2f_bytes[c2f_bytes.len() - OUTL_TRAILER_BYTES..];
        let loff = u64::from_le_bytes(lt[0..8].try_into().unwrap()) as usize;
        let coff = u64::from_le_bytes(ct[0..8].try_into().unwrap()) as usize;
        let lflags = u32::from_le_bytes(legacy_bytes[loff + 12..loff + 16].try_into().unwrap());
        let cflags = u32::from_le_bytes(c2f_bytes[coff + 12..coff + 16].try_into().unwrap());
        assert_eq!(lflags & OUTL_FLAG_POS_RANS, 0, "legacy must NOT set POS_RANS");
        assert_ne!(cflags & OUTL_FLAG_POS_RANS, 0, "c2f MUST set POS_RANS");

        // SPRV must still seal+verify over a POS_RANS archive (decode unchanged).
        append_sprv_computed(&c2f_path, false).expect("sprv on c2f-outl file");
        verify_archive(&c2f_path, VerifyDepth::Full).expect("verify c2f-outl archive");
    }

    #[test]
    fn outl_sprv_stacking_both_orders_of_read() {
        let (buf, _cfg) = build_test_archive();
        let path = tmp_path("stack");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();

        let wires = sample_wires();
        append_outl(&path, &wires).expect("append outl");
        let sprv = append_sprv_computed(&path, false).expect("append sprv on outl-trailered file");

        let outl_1 = read_outl(&path).unwrap().expect("outl under sprv");
        let sprv_1 = read_sprv(&path).unwrap().expect("sprv outermost");

        let sprv_2 = read_sprv(&path).unwrap().expect("sprv outermost");
        let outl_2 = read_outl(&path).unwrap().expect("outl under sprv");
        assert_eq!(outl_1, outl_2);
        assert_eq!(sprv_1, sprv_2);
        assert_eq!(outl_1.tensors, wires);
        assert_eq!(sprv_1, sprv);

        verify_archive(&path, VerifyDepth::Full).expect("full verify on stacked file");

        let err = append_outl(&path, &wires).unwrap_err();
        assert!(err.contains("BEFORE SPRV"), "err was: {err}");
    }

    #[test]
    fn outl_append_validates() {
        let (buf, _cfg) = build_test_archive();
        let path = tmp_path("validate");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();

        assert!(append_outl(&path, &sample_wires()[..1]).is_err());

        let mut w = sample_wires();
        w[0].as_mut().unwrap().entries.push((5000, 1));
        w[0].as_mut().unwrap().entries.sort_unstable_by_key(|&(i, _)| i);
        assert!(append_outl(&path, &w).is_err());

        let mut w = sample_wires();
        w[0].as_mut().unwrap().entries[0].1 = 200;
        assert!(append_outl(&path, &w).is_err());

        assert_eq!(std::fs::read(&path).unwrap(), buf);
    }

    #[test]
    fn outl_corrupt_trailer_is_error_not_crash() {
        let (buf, _cfg) = build_test_archive();
        let path = tmp_path("corrupt");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();
        append_outl(&path, &sample_wires()).expect("append");
        let clean = std::fs::read(&path).unwrap();

        let mut c1 = clean.clone();
        let pb_pos = c1.len() - 8;
        c1[pb_pos] ^= 0xFF;
        assert!(read_outl_bytes(&c1, true).is_err());
        assert_eq!(read_outl_bytes(&c1, false).unwrap(), None);

        let outl_off = {
            let t = &clean[clean.len() - OUTL_TRAILER_BYTES..];
            u64::from_le_bytes(t[0..8].try_into().unwrap()) as usize
        };
        let mut c2 = clean.clone();
        c2[outl_off + 4] ^= 0xFF;
        assert!(read_outl_bytes(&c2, true).is_err());

        assert_eq!(read_outl_bytes(b"", true).unwrap(), None);
        assert_eq!(read_outl_bytes(b"OUTL", true).unwrap(), None);
        let mut tiny = vec![0u8; OUTL_TRAILER_BYTES];
        tiny[12..].copy_from_slice(OUTL_MAGIC);
        assert!(read_outl_bytes(&tiny, true).is_err());
        assert_eq!(read_outl_bytes(&tiny, false).unwrap(), None);
    }

    #[test]
    fn dequant_matches_quantize_model_recon_math() {
        let omax = 0.7341f32;
        let ob = 8u32;
        let levels = ((1i64 << (ob - 1)) - 1) as f32;
        let gts = [0.7341f32, -0.5, 0.001, 0.25, -0.7, 0.123_456_7];
        let mut idx = Vec::new();
        let mut codes = Vec::new();
        let mut want = Vec::new();
        for (i, &g) in gts.iter().enumerate() {
            let v = (g / omax * levels).round() / levels * omax;
            let code = (g / omax * levels).round() as i32;
            idx.push(i);
            codes.push(code);
            want.push(v.to_bits());
        }
        let w = OutlierWire::from_selection(4096, idx, codes, omax, ob);
        let got: Vec<u32> = w.dequant_vals().map(|(_, v)| v.to_bits()).collect();
        assert_eq!(got, want, "dequant must be byte-identical to the recon path");
    }

    #[test]
    fn wire_bytes_matches_delta_billing() {
        let w = OutlierWire::from_selection(1024, vec![1, 2, 3], vec![1, -1, 7], 1.0, 8);

        assert_eq!(w.wire_bytes(), 19);
    }

    #[test]
    fn idx_bits_edges() {
        assert_eq!(idx_bits_for(0), 1);
        assert_eq!(idx_bits_for(1), 1);
        assert_eq!(idx_bits_for(2), 1);
        assert_eq!(idx_bits_for(3), 2);
        assert_eq!(idx_bits_for(1024), 10);
        assert_eq!(idx_bits_for(1025), 11);
        assert_eq!(idx_bits_for(896 * 4864), 23);
    }
}

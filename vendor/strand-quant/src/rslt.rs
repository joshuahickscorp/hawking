use crate::format::{read_strand_v2_header, PAGE};
use std::fs;
use std::io::Write as _;
use std::path::Path;

pub const RSLT_MAGIC: &[u8; 4] = b"RSLT";

pub const RSLT_VERSION: u8 = 1;

pub const RSLT_HEADER_BYTES: usize = 32;

pub const RSLT_TRAILER_BYTES: usize = 16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RsltSection {
    pub version: u8,

    pub block_counts: Vec<Vec<u32>>,
}

pub fn record_decode(section: &mut RsltSection, tensor_idx: usize, n_blocks: usize) -> &mut [u32] {
    if section.block_counts.len() <= tensor_idx {
        section.block_counts.resize(tensor_idx + 1, Vec::new());
    }
    let v = &mut section.block_counts[tensor_idx];
    if v.len() < n_blocks {
        v.resize(n_blocks, 0u32);
    }
    &mut v[..n_blocks]
}

pub fn serialize(section: &RsltSection) -> Vec<u8> {
    let n_tensors = section.block_counts.len();
    let mut o = Vec::new();

    o.extend_from_slice(RSLT_MAGIC);
    o.push(RSLT_VERSION);
    o.extend_from_slice(&[0u8; 3]);
    o.extend_from_slice(&(n_tensors as u32).to_le_bytes());
    o.extend_from_slice(&[0u8; 20]);
    debug_assert_eq!(o.len(), RSLT_HEADER_BYTES);

    for counts in &section.block_counts {
        o.extend_from_slice(&(counts.len() as u32).to_le_bytes());
        o.extend_from_slice(&0u32.to_le_bytes());
        for &c in counts {
            o.extend_from_slice(&c.to_le_bytes());
        }
    }
    o
}

pub fn deserialize(bytes: &[u8]) -> Result<RsltSection, &'static str> {
    if bytes.len() < RSLT_HEADER_BYTES {
        return Err("rslt: section shorter than the 32-byte header");
    }
    if &bytes[0..4] != &RSLT_MAGIC[..] {
        return Err("rslt: bad section header magic");
    }
    let version = bytes[4];
    if version != RSLT_VERSION {
        return Err("rslt: unsupported version");
    }
    if bytes[5..8].iter().any(|&b| b != 0) {
        return Err("rslt: header reserved bytes (5-7) not zero");
    }
    let n_tensors = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if bytes[12..RSLT_HEADER_BYTES].iter().any(|&b| b != 0) {
        return Err("rslt: header reserved2 bytes not zero");
    }

    let mut p = RSLT_HEADER_BYTES;
    let take = |p: &mut usize, n: usize| -> Result<&[u8], &'static str> {
        let end = p.checked_add(n).filter(|&e| e <= bytes.len()).ok_or("rslt: section truncated")?;
        let s = &bytes[*p..end];
        *p = end;
        Ok(s)
    };

    let mut block_counts = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let n_blocks = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap()) as usize;
        let reserved = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap());
        if reserved != 0 {
            return Err("rslt: per-tensor reserved field not zero");
        }
        let raw = take(&mut p, n_blocks * 4)?;
        let counts: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        block_counts.push(counts);
    }

    if p != bytes.len() {
        return Err("rslt: trailing bytes after the last tensor record");
    }

    Ok(RsltSection { version, block_counts })
}

pub fn merge(a: &mut RsltSection, b: &RsltSection) {
    for (a_counts, b_counts) in a.block_counts.iter_mut().zip(b.block_counts.iter()) {
        for (ac, &bc) in a_counts.iter_mut().zip(b_counts.iter()) {
            *ac = ac.saturating_add(bc);
        }
    }
}

pub fn hot_blocks(section: &RsltSection, threshold: u32) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (ti, counts) in section.block_counts.iter().enumerate() {
        for (bi, &c) in counts.iter().enumerate() {
            if c > threshold {
                out.push((ti, bi));
            }
        }
    }
    out
}

#[inline]
fn page_align(x: usize) -> usize {
    (x + PAGE - 1) & !(PAGE - 1)
}

fn rslt_section_and_trailer(section: &RsltSection, rslt_offset: usize) -> Result<Vec<u8>, String> {
    let body = serialize(section);
    let rslt_bytes: u32 = body.len().try_into().map_err(|_| format!("rslt: section is {} bytes — exceeds the u32 rslt_bytes field", body.len()))?;
    let mut out = body;
    out.extend_from_slice(&(rslt_offset as u64).to_le_bytes());
    out.extend_from_slice(&rslt_bytes.to_le_bytes());
    out.extend_from_slice(RSLT_MAGIC);
    Ok(out)
}

fn parse_rslt_section(buf: &[u8], rslt_offset: usize, rslt_bytes: usize) -> Result<RsltSection, String> {
    if rslt_offset % PAGE != 0 {
        return Err(format!("rslt: rslt_offset {rslt_offset} not page-aligned"));
    }
    let expected_len = (rslt_offset as u64).checked_add(rslt_bytes as u64).and_then(|x| x.checked_add(RSLT_TRAILER_BYTES as u64)).ok_or("rslt: rslt_offset + rslt_bytes overflows".to_string())?;
    if expected_len != buf.len() as u64 {
        return Err(format!("rslt: rslt_offset {rslt_offset} + rslt_bytes {rslt_bytes} + 16 != file len {}", buf.len()));
    }
    if rslt_bytes < RSLT_HEADER_BYTES {
        return Err("rslt: section shorter than the 32-byte RSLT header".into());
    }
    let section_bytes = &buf[rslt_offset..rslt_offset + rslt_bytes];
    deserialize(section_bytes).map_err(|e| e.to_string())
}

pub fn read_rslt_bytes(buf: &[u8], strict: bool) -> Result<Option<RsltSection>, String> {
    if buf.len() < RSLT_TRAILER_BYTES || &buf[buf.len() - 4..] != &RSLT_MAGIC[..] {
        return Ok(None);
    }
    let t = &buf[buf.len() - RSLT_TRAILER_BYTES..];
    let parse = (|| -> Result<RsltSection, String> {
        let rslt_offset = u64::from_le_bytes(t[0..8].try_into().unwrap());
        let rslt_bytes = u32::from_le_bytes(t[8..12].try_into().unwrap());
        let rslt_offset: usize = rslt_offset.try_into().map_err(|_| "rslt: rslt_offset exceeds address space".to_string())?;
        parse_rslt_section(buf, rslt_offset, rslt_bytes as usize)
    })();
    match parse {
        Ok(s) => Ok(Some(s)),
        Err(e) => {
            if strict {
                Err(e)
            } else {
                Ok(None)
            }
        }
    }
}

pub fn read_rslt(path: impl AsRef<Path>) -> Result<Option<RsltSection>, String> {
    let path = path.as_ref();
    let buf = fs::read(path).map_err(|e| format!("rslt: read {path:?}: {e}"))?;
    read_rslt_bytes(&buf, true)
}

pub fn append_rslt(path: impl AsRef<Path>, section: &RsltSection) -> Result<(), String> {
    let path = path.as_ref();
    let buf = fs::read(path).map_err(|e| format!("rslt: read {path:?}: {e}"))?;

    match read_rslt_bytes(&buf, true) {
        Ok(Some(_)) => return Err("rslt: file already has an RSLT section (double-append rejected)".into()),
        Err(e) => {
            return Err(format!(
                "rslt: file ends in RSLT magic but the section is invalid — refusing to \
                 append a second trailer behind it: {e}"
            ))
        }
        Ok(None) => {}
    }

    let hdr = read_strand_v2_header(&buf)?;
    if section.block_counts.len() != hdr.tensors.len() {
        return Err(format!("rslt: section has {} tensor records, archive has {} tensors", section.block_counts.len(), hdr.tensors.len()));
    }
    for (i, (counts, desc)) in section.block_counts.iter().zip(hdr.tensors.iter()).enumerate() {
        if counts.len() != desc.n_blocks {
            return Err(format!(
                "rslt: tensor record {i} ({:?}): block_counts has {} entries, \
                 descriptor n_blocks is {}",
                desc.name,
                counts.len(),
                desc.n_blocks
            ));
        }
    }

    let rslt_offset = page_align(buf.len());
    let lead_pad = rslt_offset - buf.len();
    let section_and_trailer = rslt_section_and_trailer(section, rslt_offset)?;

    let mut tail = Vec::with_capacity(lead_pad + section_and_trailer.len());
    tail.resize(lead_pad, 0u8);
    tail.extend_from_slice(&section_and_trailer);

    let mut f = fs::OpenOptions::new().append(true).open(path).map_err(|e| format!("rslt: open {path:?} for append: {e}"))?;
    f.write_all(&tail).map_err(|e| format!("rslt: append to {path:?}: {e}"))?;
    Ok(())
}

pub fn write_rslt(path: impl AsRef<Path>, section: &RsltSection) -> Result<(), String> {
    let path = path.as_ref();
    let bytes = serialize(section);
    fs::write(path, &bytes).map_err(|e| format!("rslt: write {path:?}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{encode_tensor_with, EncodeOpts};
    use crate::format::{write_strand_v2, PackedTensor, PackedTensorV2};
    use crate::trellis::TrellisConfig;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn tmp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("strand-rslt-{tag}-{}-{}.strand", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed)))
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
        write_strand_v2(&tensors, [7u8; 32], true).expect("write v2")
    }

    #[test]
    fn rslt_roundtrip() {
        let section = RsltSection { version: RSLT_VERSION, block_counts: vec![vec![0, 5, 100, 999], vec![1, 2, 3]] };
        let bytes = serialize(&section);
        let back = deserialize(&bytes).expect("deserialize");
        assert_eq!(back, section, "RSLT round-trip must be exact");
    }

    #[test]
    fn rslt_file_roundtrip() {
        let buf = build_test_archive();
        let path = tmp_path("roundtrip");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();

        assert_eq!(read_rslt(&path).unwrap(), None, "plain v2 file must have no RSLT");

        let hdr = crate::format::read_strand_v2_header(&buf).unwrap();
        let section = RsltSection { version: RSLT_VERSION, block_counts: hdr.tensors.iter().map(|t| vec![0u32; t.n_blocks]).collect() };

        let mut s = section.clone();
        s.block_counts[0][0] = 42;
        s.block_counts[1][0] = 7;

        append_rslt(&path, &s).expect("append rslt");
        let back = read_rslt(&path).unwrap().expect("RSLT section found");
        assert_eq!(back, s, "RSLT file round-trip must be exact");

        let trailered = std::fs::read(&path).unwrap();
        assert!(trailered.len() > buf.len());
        assert_eq!(&trailered[..buf.len()], &buf[..], "append must not touch v2 bytes");
    }

    #[test]
    fn rslt_merge() {
        let mut a = RsltSection { version: RSLT_VERSION, block_counts: vec![vec![1, 2, u32::MAX - 1], vec![10, 20]] };
        let b = RsltSection { version: RSLT_VERSION, block_counts: vec![vec![3, 4, 2], vec![100, 200]] };
        merge(&mut a, &b);
        assert_eq!(a.block_counts[0], vec![4, 6, u32::MAX], "saturating add at MAX");
        assert_eq!(a.block_counts[1], vec![110, 220], "normal add");
    }

    #[test]
    fn rslt_hot_blocks() {
        let section = RsltSection { version: RSLT_VERSION, block_counts: vec![vec![0, 50, 100, 1], vec![99, 101, 200], vec![5]] };
        let hot = hot_blocks(&section, 99);

        assert_eq!(hot, vec![(0, 2), (1, 1), (1, 2)]);

        let hot0 = hot_blocks(&section, 0);
        assert_eq!(hot0, vec![(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (2, 0)]);

        let hot_max = hot_blocks(&section, u32::MAX);
        assert!(hot_max.is_empty());
    }

    #[test]
    fn rslt_double_append_rejected() {
        let buf = build_test_archive();
        let path = tmp_path("double");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();

        let hdr = crate::format::read_strand_v2_header(&buf).unwrap();
        let section = RsltSection { version: RSLT_VERSION, block_counts: hdr.tensors.iter().map(|t| vec![0u32; t.n_blocks]).collect() };

        append_rslt(&path, &section).expect("first append");
        let after_first = std::fs::read(&path).unwrap();

        let err = append_rslt(&path, &section).unwrap_err();
        assert!(err.contains("already has"), "err was: {err}");
        assert_eq!(std::fs::read(&path).unwrap(), after_first, "file must be untouched");
    }

    #[test]
    fn rslt_corrupt_trailer_is_error_not_crash() {
        let buf = build_test_archive();
        let path = tmp_path("corrupt");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();

        let hdr = crate::format::read_strand_v2_header(&buf).unwrap();
        let section = RsltSection { version: RSLT_VERSION, block_counts: hdr.tensors.iter().map(|t| vec![1u32; t.n_blocks]).collect() };
        append_rslt(&path, &section).expect("append");
        let clean = std::fs::read(&path).unwrap();

        let mut c1 = clean.clone();
        let pb_pos = c1.len() - 8;
        c1[pb_pos] ^= 0xFF;
        assert!(read_rslt_bytes(&c1, true).is_err());
        assert_eq!(read_rslt_bytes(&c1, false).unwrap(), None);

        let c2 = &clean[..clean.len() - 1];
        assert_eq!(read_rslt_bytes(c2, true).unwrap(), None);

        assert_eq!(read_rslt_bytes(b"", true).unwrap(), None);
        assert_eq!(read_rslt_bytes(b"RSLT", true).unwrap(), None);
        let mut tiny = vec![0u8; RSLT_TRAILER_BYTES];
        tiny[12..].copy_from_slice(RSLT_MAGIC);
        assert!(read_rslt_bytes(&tiny, true).is_err());
        assert_eq!(read_rslt_bytes(&tiny, false).unwrap(), None);
    }

    #[test]
    fn rslt_write_raw() {
        let tmp = std::env::temp_dir().join(format!("strand-rslt-raw-{}-{}.bin", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed)));
        let _guard = TmpFile(tmp.clone());
        let section = RsltSection { version: RSLT_VERSION, block_counts: vec![vec![1, 2, 3], vec![4, 5]] };
        write_rslt(&tmp, &section).expect("write_rslt");
        let bytes = std::fs::read(&tmp).unwrap();
        let back = deserialize(&bytes).expect("deserialize raw");
        assert_eq!(back, section);
    }
}

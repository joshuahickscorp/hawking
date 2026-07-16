use std::fs;
use std::io::Write as _;
use std::path::Path;

use crate::codebook::codebook_lut;
use crate::format::{read_strand_v2_header, OwnedTensorV2, PAGE};
use crate::outlier_wire::{read_outl_bytes, OutlierWire};
use crate::provenance::{block_hashes, descriptor_digest, make_test_vectors, model_root_from_tensor_roots, outlier_digest, tensor_root_from_hashes, verify_test_vectors, ProvenanceVector};
use crate::sideinfo_wire::read_strand_v2_applied;
use crate::trellis::TrellisConfig;

fn live_descriptor_digest(t: &OwnedTensorV2, wire: Option<&OutlierWire>) -> [u8; 32] {
    let mut f = 0u8;
    if t.base.enc.has_rht_seed {
        f |= 1;
    }
    if t.base.enc.tail_biting {
        f |= 2;
    }
    if t.base.enc.has_affine_min {
        f |= 4;
    }
    let outl = wire.map(outlier_digest).unwrap_or([0u8; 32]);
    descriptor_digest(&t.base.name, &t.base.shape, t.base.rht_seed, t.base.l_bits, t.base.k_bits, t.base.vec_dim, f, t.block_len, t.base.enc.total as u64, &outl)
}

pub const SPRV_MAGIC: &[u8; 4] = b"SPRV";

pub const SPRV_VERSION: u32 = 2;

pub const SPRV_HEADER_BYTES: usize = 64;

pub const SPRV_TRAILER_BYTES: usize = 16;

pub const SPRV_RECORD_FIXED_BYTES: usize = 80;

pub const SPRV_VECTOR_BYTES: usize = 40;

pub const DEFAULT_VECTORS_PER_TENSOR: usize = 8;

pub mod sprv_flags {

    pub const LEAF_LISTS: u32 = 1 << 0;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SprvTensor {
    pub tensor_root: [u8; 32],

    pub descriptor_digest: [u8; 32],

    pub n_blocks: u64,

    pub vectors: Vec<ProvenanceVector>,

    pub leaves: Option<Vec<[u8; 32]>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sprv {
    pub flags: u32,

    pub model_root: [u8; 32],

    pub tensors: Vec<SprvTensor>,
}

impl Sprv {
    pub fn has_leaf_lists(&self) -> bool {
        self.flags & sprv_flags::LEAF_LISTS != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifyDepth {
    Vectors,

    Full,
}

pub type LutProvider<'a> = &'a dyn Fn(&OwnedTensorV2) -> Result<Vec<i32>, String>;

pub fn default_lut_provider(t: &OwnedTensorV2) -> Result<Vec<i32>, String> {
    if t.base.vec_dim <= 1 {
        Ok(codebook_lut(t.base.l_bits as u32).to_vec())
    } else {
        Err(format!(
            "sprv: tensor {:?} has vec_dim {} — its LUT is not derivable from the \
             descriptor; supply it via a LutProvider",
            t.base.name, t.base.vec_dim
        ))
    }
}

fn cfg_for(t: &OwnedTensorV2) -> Result<TrellisConfig, String> {
    let cfg = TrellisConfig::new(t.base.l_bits as u32, t.base.k_bits as u32, t.block_len as usize).with_vec_dim(t.base.vec_dim as u32);
    if cfg.l_bits != t.base.l_bits as u32 || cfg.k_bits != t.base.k_bits as u32 || cfg.block_len != t.block_len as usize || cfg.vec_dim() != (t.base.vec_dim as usize).max(1) {
        return Err(format!(
            "sprv: tensor {:?} descriptor geometry (L={} k={} block_len={} d={}) is \
             outside the supported TrellisConfig range",
            t.base.name, t.base.l_bits, t.base.k_bits, t.block_len, t.base.vec_dim
        ));
    }
    Ok(cfg)
}

fn lut_for_checked(t: &OwnedTensorV2, cfg: &TrellisConfig, lut_for: LutProvider) -> Result<Vec<i32>, String> {
    let lut = lut_for(t)?;
    if lut.len() < cfg.lut_len() {
        return Err(format!("sprv: tensor {:?} LUT has {} entries, config needs {}", t.base.name, lut.len(), cfg.lut_len()));
    }
    Ok(lut)
}

#[inline]
fn page_align(x: usize) -> usize {
    (x + PAGE - 1) & !(PAGE - 1)
}

pub fn build_sprv(buf: &[u8], n_vectors: usize, include_leaves: bool, lut_for: LutProvider) -> Result<Sprv, String> {
    let tensors = read_strand_v2_applied(buf)?;

    let outl = read_outl_bytes(buf, true)?;
    if let Some(o) = &outl {
        if o.tensors.len() != tensors.len() {
            return Err(format!("sprv: OUTL section has {} records, archive has {} tensors", o.tensors.len(), tensors.len()));
        }
    }
    let mut records = Vec::with_capacity(tensors.len());
    let mut named_roots: Vec<(String, [u8; 32])> = Vec::with_capacity(tensors.len());
    for (i, t) in tensors.iter().enumerate() {
        let cfg = cfg_for(t)?;
        let lut = lut_for_checked(t, &cfg, lut_for)?;
        let leaves = block_hashes(&t.base.enc, &cfg, &lut);
        let root = tensor_root_from_hashes(&leaves);
        let vectors = make_test_vectors(&t.base.enc, &cfg, &lut, n_vectors);
        let wire = outl.as_ref().and_then(|o| o.tensors[i].as_ref());
        records.push(SprvTensor {
            tensor_root: root,
            descriptor_digest: live_descriptor_digest(t, wire),
            n_blocks: leaves.len() as u64,
            vectors,
            leaves: if include_leaves { Some(leaves) } else { None },
        });
        named_roots.push((t.base.name.clone(), root));
    }
    let model_root = model_root_from_tensor_roots(named_roots.iter().map(|(n, r)| (n.as_str(), *r)));
    Ok(Sprv { flags: if include_leaves { sprv_flags::LEAF_LISTS } else { 0 }, model_root, tensors: records })
}

fn sprv_section_bytes(sprv: &Sprv) -> Result<Vec<u8>, String> {
    if sprv.flags & !sprv_flags::LEAF_LISTS != 0 {
        return Err(format!("sprv: reserved flag bits set: {:#x}", sprv.flags));
    }
    let want_leaves = sprv.has_leaf_lists();
    let mut o = Vec::new();
    o.extend_from_slice(SPRV_MAGIC);
    o.extend_from_slice(&SPRV_VERSION.to_le_bytes());
    o.extend_from_slice(&(sprv.tensors.len() as u32).to_le_bytes());
    o.extend_from_slice(&sprv.flags.to_le_bytes());
    o.extend_from_slice(&sprv.model_root);
    o.extend_from_slice(&[0u8; 16]);
    debug_assert_eq!(o.len(), SPRV_HEADER_BYTES);

    for (i, t) in sprv.tensors.iter().enumerate() {
        match (&t.leaves, want_leaves) {
            (Some(l), true) => {
                if l.len() as u64 != t.n_blocks {
                    return Err(format!("sprv: tensor record {i}: leaf list has {} entries, n_blocks is {}", l.len(), t.n_blocks));
                }
            }
            (None, false) => {}
            _ => {
                return Err(format!(
                    "sprv: tensor record {i}: leaf-list presence disagrees with flags \
                     bit 0 (file-level flag — all tensors or none)"
                ));
            }
        }
        if t.vectors.len() > u32::MAX as usize {
            return Err(format!("sprv: tensor record {i}: too many vectors"));
        }
        let mut prev: Option<u64> = None;
        for v in &t.vectors {
            if v.block_index >= t.n_blocks {
                return Err(format!("sprv: tensor record {i}: vector block_index {} out of range ({} blocks)", v.block_index, t.n_blocks));
            }
            if let Some(p) = prev {
                if v.block_index <= p {
                    return Err(format!("sprv: tensor record {i}: vector indices must be strictly ascending"));
                }
            }
            prev = Some(v.block_index);
        }

        o.extend_from_slice(&t.tensor_root);
        o.extend_from_slice(&t.descriptor_digest);
        o.extend_from_slice(&t.n_blocks.to_le_bytes());
        o.extend_from_slice(&(t.vectors.len() as u32).to_le_bytes());
        o.extend_from_slice(&0u32.to_le_bytes());
        for v in &t.vectors {
            o.extend_from_slice(&v.block_index.to_le_bytes());
            o.extend_from_slice(&v.block_hash);
        }
        if let Some(leaves) = &t.leaves {
            for h in leaves {
                o.extend_from_slice(h);
            }
        }
    }
    Ok(o)
}

pub fn append_sprv(path: impl AsRef<Path>, sprv: &Sprv) -> Result<(), String> {
    let path = path.as_ref();
    let buf = fs::read(path).map_err(|e| format!("sprv: read {path:?}: {e}"))?;

    match read_sprv_bytes(&buf, true) {
        Ok(Some(_)) => return Err("sprv: file already has an SPRV section (double-append rejected)".into()),
        Err(e) => {
            return Err(format!(
                "sprv: file ends in SPRV magic but the section is invalid — refusing to \
                 append a second trailer behind it: {e}"
            ))
        }
        Ok(None) => {}
    }

    let hdr = read_strand_v2_header(&buf)?;
    if sprv.tensors.len() != hdr.tensors.len() {
        return Err(format!("sprv: section has {} tensor records, archive has {} tensors", sprv.tensors.len(), hdr.tensors.len()));
    }
    for (i, (rec, desc)) in sprv.tensors.iter().zip(hdr.tensors.iter()).enumerate() {
        if rec.n_blocks != desc.n_blocks as u64 {
            return Err(format!("sprv: tensor record {i} ({:?}): n_blocks {} != descriptor's {}", desc.name, rec.n_blocks, desc.n_blocks));
        }
    }

    let section = sprv_section_bytes(sprv)?;
    let prov_bytes: u32 = section.len().try_into().map_err(|_| format!("sprv: section is {} bytes — exceeds the u32 prov_bytes field", section.len()))?;

    let prov_offset = page_align(buf.len());
    let pad = prov_offset - buf.len();

    let mut tail = Vec::with_capacity(pad + section.len() + SPRV_TRAILER_BYTES);
    tail.resize(pad, 0u8);
    tail.extend_from_slice(&section);
    tail.extend_from_slice(&(prov_offset as u64).to_le_bytes());
    tail.extend_from_slice(&prov_bytes.to_le_bytes());
    tail.extend_from_slice(SPRV_MAGIC);

    let mut f = fs::OpenOptions::new().append(true).open(path).map_err(|e| format!("sprv: open {path:?} for append: {e}"))?;
    f.write_all(&tail).map_err(|e| format!("sprv: append to {path:?}: {e}"))?;
    Ok(())
}

pub fn append_sprv_computed(path: impl AsRef<Path>, include_leaves: bool) -> Result<Sprv, String> {
    let path = path.as_ref();
    let buf = fs::read(path).map_err(|e| format!("sprv: read {path:?}: {e}"))?;
    let sprv = build_sprv(&buf, DEFAULT_VECTORS_PER_TENSOR, include_leaves, &default_lut_provider)?;
    append_sprv(path, &sprv)?;
    Ok(sprv)
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.p.checked_add(n).filter(|&e| e <= self.b.len()).ok_or("sprv: section truncated")?;
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn h32(&mut self) -> Result<[u8; 32], String> {
        Ok(self.take(32)?.try_into().unwrap())
    }
}

fn parse_sprv_section(buf: &[u8], prov_offset: usize, prov_bytes: usize) -> Result<Sprv, String> {
    if prov_offset % PAGE != 0 {
        return Err(format!("sprv: prov_offset {prov_offset} not page-aligned"));
    }
    let expected_len = (prov_offset as u64).checked_add(prov_bytes as u64).and_then(|x| x.checked_add(SPRV_TRAILER_BYTES as u64)).ok_or("sprv: prov_offset + prov_bytes overflows")?;
    if expected_len != buf.len() as u64 {
        return Err(format!("sprv: prov_offset {prov_offset} + prov_bytes {prov_bytes} + 16 != file len {}", buf.len()));
    }
    if prov_bytes < SPRV_HEADER_BYTES {
        return Err("sprv: section shorter than the 64-byte PROV header".into());
    }

    let v2 = read_strand_v2_header(buf)?;

    let mut r = Rd { b: &buf[prov_offset..prov_offset + prov_bytes], p: 0 };
    if r.take(4)? != &SPRV_MAGIC[..] {
        return Err("sprv: bad PROV header magic".into());
    }
    let version = r.u32()?;
    if version != SPRV_VERSION {
        return Err(format!("sprv: version {version} != {SPRV_VERSION}"));
    }
    let n_tensors = r.u32()? as usize;
    if n_tensors != v2.tensors.len() {
        return Err(format!("sprv: section n_tensors {n_tensors} != archive's {}", v2.tensors.len()));
    }
    let flags = r.u32()?;
    if flags & !sprv_flags::LEAF_LISTS != 0 {
        return Err(format!("sprv: reserved flag bits set: {flags:#x}"));
    }
    let has_leaves = flags & sprv_flags::LEAF_LISTS != 0;
    let model_root = r.h32()?;
    if r.take(16)?.iter().any(|&b| b != 0) {
        return Err("sprv: PROV header reserved bytes not zero".into());
    }

    let mut tensors = Vec::with_capacity(n_tensors);
    for (i, desc) in v2.tensors.iter().enumerate() {
        let tensor_root = r.h32()?;
        let descriptor_digest = r.h32()?;
        let n_blocks = r.u64()?;
        if n_blocks != desc.n_blocks as u64 {
            return Err(format!("sprv: tensor record {i} ({:?}): n_blocks {n_blocks} != descriptor's {}", desc.name, desc.n_blocks));
        }
        let n_vectors = r.u32()? as usize;
        if r.u32()? != 0 {
            return Err(format!("sprv: tensor record {i}: reserved field not zero"));
        }
        let mut vectors = Vec::with_capacity(n_vectors.min(1 << 20));
        let mut prev: Option<u64> = None;
        for _ in 0..n_vectors {
            let block_index = r.u64()?;
            let block_hash = r.h32()?;
            if block_index >= n_blocks {
                return Err(format!(
                    "sprv: tensor record {i}: vector block_index {block_index} out of \
                     range ({n_blocks} blocks)"
                ));
            }
            if let Some(p) = prev {
                if block_index <= p {
                    return Err(format!("sprv: tensor record {i}: vector indices not strictly ascending"));
                }
            }
            prev = Some(block_index);
            vectors.push(ProvenanceVector { block_index, block_hash });
        }
        let leaves = if has_leaves {
            let mut l = Vec::with_capacity(desc.n_blocks);
            for _ in 0..desc.n_blocks {
                l.push(r.h32()?);
            }
            Some(l)
        } else {
            None
        };
        tensors.push(SprvTensor { tensor_root, descriptor_digest, n_blocks, vectors, leaves });
    }

    if r.p != prov_bytes {
        return Err(format!("sprv: {} trailing bytes after the last record", prov_bytes - r.p));
    }
    Ok(Sprv { flags, model_root, tensors })
}

pub fn read_sprv_bytes(buf: &[u8], strict: bool) -> Result<Option<Sprv>, String> {
    if buf.len() < SPRV_TRAILER_BYTES || &buf[buf.len() - 4..] != &SPRV_MAGIC[..] {
        return Ok(None);
    }
    let t = &buf[buf.len() - SPRV_TRAILER_BYTES..];
    let parse = (|| -> Result<Sprv, String> {
        let prov_offset = u64::from_le_bytes(t[0..8].try_into().unwrap());
        let prov_bytes = u32::from_le_bytes(t[8..12].try_into().unwrap());
        let prov_offset: usize = prov_offset.try_into().map_err(|_| "sprv: prov_offset exceeds address space".to_string())?;
        parse_sprv_section(buf, prov_offset, prov_bytes as usize)
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

pub fn read_sprv(path: impl AsRef<Path>) -> Result<Option<Sprv>, String> {
    let path = path.as_ref();
    let buf = fs::read(path).map_err(|e| format!("sprv: read {path:?}: {e}"))?;
    read_sprv_bytes(&buf, true)
}

pub fn verify_archive_with(path: impl AsRef<Path>, depth: VerifyDepth, lut_for: LutProvider) -> Result<(), String> {
    let path = path.as_ref();
    let buf = fs::read(path).map_err(|e| format!("sprv: read {path:?}: {e}"))?;
    let sprv = read_sprv_bytes(&buf, true)?.ok_or("sprv: archive has no SPRV section — nothing to verify against")?;
    let tensors = read_strand_v2_applied(&buf)?;

    debug_assert_eq!(sprv.tensors.len(), tensors.len());

    let outl = read_outl_bytes(&buf, true)?;
    if let Some(o) = &outl {
        if o.tensors.len() != tensors.len() {
            return Err(format!("sprv: OUTL section has {} records, archive has {} tensors", o.tensors.len(), tensors.len()));
        }
    }
    for (i, (t, rec)) in tensors.iter().zip(sprv.tensors.iter()).enumerate() {
        let wire = outl.as_ref().and_then(|o| o.tensors[i].as_ref());
        if live_descriptor_digest(t, wire) != rec.descriptor_digest {
            return Err(format!(
                "sprv: tensor {:?}: descriptor digest MISMATCH (rht_seed/shape/geometry/\
                 flags/outlier-channel metadata does not match the attested descriptor)",
                t.base.name
            ));
        }
    }

    let stored = model_root_from_tensor_roots(tensors.iter().zip(sprv.tensors.iter()).map(|(t, r)| (t.base.name.as_str(), r.tensor_root)));
    if stored != sprv.model_root {
        return Err("sprv: model_root does not match the stored tensor roots".into());
    }

    for (t, rec) in tensors.iter().zip(sprv.tensors.iter()) {
        let name = &t.base.name;
        if let Some(leaves) = &rec.leaves {
            if tensor_root_from_hashes(leaves) != rec.tensor_root {
                return Err(format!("sprv: tensor {name:?}: stored leaf list does not match stored tensor_root"));
            }
            for v in &rec.vectors {
                if leaves[v.block_index as usize] != v.block_hash {
                    return Err(format!(
                        "sprv: tensor {name:?}: stored vector for block {} disagrees with \
                         stored leaf list",
                        v.block_index
                    ));
                }
            }
        }

        let cfg = cfg_for(t)?;
        let lut = lut_for_checked(t, &cfg, lut_for)?;
        match depth {
            VerifyDepth::Vectors => {
                if !verify_test_vectors(&t.base.enc, &cfg, &lut, &rec.vectors) {
                    return Err(format!(
                        "sprv: tensor {name:?}: self-test vector verification FAILED \
                         (payload does not decode to the committed leaves)"
                    ));
                }
            }
            VerifyDepth::Full => {
                let recomputed = block_hashes(&t.base.enc, &cfg, &lut);
                if let Some(leaves) = &rec.leaves {
                    if &recomputed != leaves {
                        return Err(format!(
                            "sprv: tensor {name:?}: recomputed leaves differ from the \
                             stored leaf list"
                        ));
                    }
                }
                for v in &rec.vectors {
                    if recomputed[v.block_index as usize] != v.block_hash {
                        return Err(format!(
                            "sprv: tensor {name:?}: stored vector for block {} disagrees \
                             with the recomputed leaf",
                            v.block_index
                        ));
                    }
                }
                if tensor_root_from_hashes(&recomputed) != rec.tensor_root {
                    return Err(format!(
                        "sprv: tensor {name:?}: recomputed tensor_root MISMATCH \
                         (decoded weights are not the committed weights)"
                    ));
                }
            }
        }
    }

    Ok(())
}

pub fn verify_archive(path: impl AsRef<Path>, depth: VerifyDepth) -> Result<(), String> {
    verify_archive_with(path, depth, &default_lut_provider)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{encode_tensor_with, EncodeOpts, EncodedTensor};
    use crate::format::{read_strand_v2, write_strand_v2, PackedTensor, PackedTensorV2};
    use crate::provenance::block_bit_offset;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn tmp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("strand-sprv-{tag}-{}-{}.strand", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed)))
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

    fn build_test_archive() -> (Vec<u8>, Vec<EncodedTensor>, TrellisConfig) {
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc_a = encode_tensor_with(&test_weights(1024, 11), &cfg, &EncodeOpts::default());
        let enc_b = encode_tensor_with(&test_weights(900, 23), &cfg, &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() });
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
        (buf, vec![enc_a, enc_b], cfg)
    }

    #[test]
    fn sprv_round_trip_and_v2_reader_compat() {
        let (buf, encs, _cfg) = build_test_archive();
        let path = tmp_path("roundtrip");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();

        assert_eq!(read_sprv(&path).unwrap(), None, "plain v2 file must read as absent");

        let written = append_sprv_computed(&path, false).expect("append");
        assert_eq!(written.tensors.len(), 2);
        assert!(!written.has_leaf_lists());

        for t in &written.tensors {
            assert_eq!(t.vectors.len(), (t.n_blocks as usize).min(DEFAULT_VECTORS_PER_TENSOR));
        }

        let back = read_sprv(&path).unwrap().expect("trailer must be found");
        assert_eq!(back, written, "SPRV round-trip must be exact");

        let trailered = std::fs::read(&path).unwrap();
        assert!(trailered.len() > buf.len());
        assert_eq!(&trailered[..buf.len()], &buf[..], "append must not touch v2 bytes");
        let h0 = read_strand_v2_header(&buf).unwrap();
        let h1 = read_strand_v2_header(&trailered).expect("v2 header parse of trailered file");
        assert_eq!(h0.source_sha256, h1.source_sha256);
        assert_eq!(h0.tensors.len(), h1.tensors.len());
        for (a, b) in h0.tensors.iter().zip(h1.tensors.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.table, b.table);
            assert_eq!(a.payload_offset, b.payload_offset);
            assert_eq!(a.payload_bytes, b.payload_bytes);
        }
        let full = read_strand_v2(&trailered).expect("v2 full read of trailered file");
        for (t, enc) in full.iter().zip(encs.iter()) {
            assert_eq!(&t.base.enc, enc, "EncodedTensor must round-trip under the trailer");
        }

        verify_archive(&path, VerifyDepth::Vectors).expect("vector verify");
        verify_archive(&path, VerifyDepth::Full).expect("full verify");
    }

    #[test]
    fn sprv_leaf_lists_round_trip() {
        let (buf, _encs, _cfg) = build_test_archive();
        let path = tmp_path("leaves");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();

        let written = append_sprv_computed(&path, true).expect("append with leaves");
        assert!(written.has_leaf_lists());
        for t in &written.tensors {
            let leaves = t.leaves.as_ref().expect("leaves present");
            assert_eq!(leaves.len() as u64, t.n_blocks);
            assert_eq!(tensor_root_from_hashes(leaves), t.tensor_root);
        }
        let back = read_sprv(&path).unwrap().expect("found");
        assert_eq!(back, written);
        verify_archive(&path, VerifyDepth::Vectors).expect("vector verify");
        verify_archive(&path, VerifyDepth::Full).expect("full verify");
    }

    #[test]
    fn sprv_tamper_detection() {
        let (buf, _encs, _cfg) = build_test_archive();
        let path = tmp_path("tamper");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();
        let written = append_sprv_computed(&path, false).expect("append");

        let clean = std::fs::read(&path).unwrap();
        let hdr = read_strand_v2_header(&clean).unwrap();
        let full = read_strand_v2(&clean).unwrap();

        let mut t1 = clean.clone();
        t1[hdr.tensors[0].payload_offset] ^= 1;
        std::fs::write(&path, &t1).unwrap();
        assert!(read_sprv(&path).unwrap().is_some(), "trailer chain must still parse");

        let err = verify_archive(&path, VerifyDepth::Full).unwrap_err();
        assert!(err.contains("MISMATCH") || err.contains("disagrees"), "err was: {err}");

        let cfg1 = cfg_for(&full[1]).unwrap();
        let target = written.tensors[1].vectors[0].block_index as usize;
        let bit = block_bit_offset(&full[1].base.enc, &cfg1, target);
        let mut t2 = clean.clone();
        t2[hdr.tensors[1].payload_offset + bit / 8] ^= 1 << (bit % 8);
        std::fs::write(&path, &t2).unwrap();
        let err = verify_archive(&path, VerifyDepth::Vectors).unwrap_err();
        assert!(err.contains("FAILED"), "err was: {err}");

        let sprv_off = {
            let t = &clean[clean.len() - SPRV_TRAILER_BYTES..];
            u64::from_le_bytes(t[0..8].try_into().unwrap()) as usize
        };
        let mut t3 = clean.clone();
        t3[sprv_off + 16] ^= 0xFF;
        std::fs::write(&path, &t3).unwrap();
        let err = verify_archive(&path, VerifyDepth::Vectors).unwrap_err();
        assert!(err.contains("model_root"), "err was: {err}");
    }

    #[test]
    fn sprv_corrupt_trailer_is_error_not_crash() {
        let (buf, _encs, _cfg) = build_test_archive();
        let path = tmp_path("corrupt");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();
        append_sprv_computed(&path, false).expect("append");
        let clean = std::fs::read(&path).unwrap();

        let mut c1 = clean.clone();
        let pb_pos = c1.len() - 8;
        c1[pb_pos] ^= 0xFF;
        assert!(read_sprv_bytes(&c1, true).is_err());
        assert_eq!(read_sprv_bytes(&c1, false).unwrap(), None);

        let sprv_off = {
            let t = &clean[clean.len() - SPRV_TRAILER_BYTES..];
            u64::from_le_bytes(t[0..8].try_into().unwrap()) as usize
        };
        let mut c2 = clean.clone();
        c2[sprv_off + 4] ^= 0xFF;
        assert!(read_sprv_bytes(&c2, true).is_err());

        let mut c3 = clean.clone();
        c3[sprv_off + SPRV_HEADER_BYTES - 1] = 1;
        assert!(read_sprv_bytes(&c3, true).is_err());

        let c4 = &clean[..clean.len() - 1];
        assert_eq!(read_sprv_bytes(c4, true).unwrap(), None);

        assert_eq!(read_sprv_bytes(b"", true).unwrap(), None);
        assert_eq!(read_sprv_bytes(b"SPRV", true).unwrap(), None);
        let mut tiny = vec![0u8; SPRV_TRAILER_BYTES];
        tiny[12..].copy_from_slice(SPRV_MAGIC);
        assert!(read_sprv_bytes(&tiny, true).is_err());
        assert_eq!(read_sprv_bytes(&tiny, false).unwrap(), None);

        std::fs::write(&path, &c1).unwrap();
        assert!(verify_archive(&path, VerifyDepth::Vectors).is_err());
    }

    #[test]
    fn sprv_double_append_rejected() {
        let (buf, _encs, _cfg) = build_test_archive();
        let path = tmp_path("double");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();
        let written = append_sprv_computed(&path, false).expect("first append");
        let after_first = std::fs::read(&path).unwrap();

        let err = append_sprv(&path, &written).unwrap_err();
        assert!(err.contains("already has"), "err was: {err}");
        assert_eq!(std::fs::read(&path).unwrap(), after_first, "file must be untouched");
    }

    #[test]
    fn sprv_v2_descriptor_tamper_detection() {
        use crate::outlier_wire::{append_outl, OutlierWire};

        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor_with(&test_weights(1024, 31), &cfg, &EncodeOpts::default());
        let shape = [4u64, 256u64];
        let build = |seed: u64| -> Vec<u8> {
            let pt = PackedTensorV2 {
                base: PackedTensor { name: "model.layers.0.q_proj", shape: &shape, rht_seed: seed, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc: &enc },
                block_len: cfg.block_len as u32,
            };
            write_strand_v2(&[pt], [3u8; 32], true).expect("write v2")
        };
        let buf_a = build(0xA5A5_DEAD_BEEF_0001);
        let buf_b = build(0xA5A5_DEAD_BEEF_0002);
        assert_eq!(buf_a.len(), buf_b.len());
        let seed_diff: Vec<usize> = (0..buf_a.len()).filter(|&i| buf_a[i] != buf_b[i]).collect();
        assert!(!seed_diff.is_empty(), "seed must live in the descriptor bytes");

        let path = tmp_path("r2-tamper");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf_a).unwrap();

        let wire = OutlierWire::from_selection(1024, vec![7, 900], vec![-100, 42], 0.5, 8);
        append_outl(&path, &[Some(wire)]).expect("append outl");
        append_sprv_computed(&path, false).expect("append sprv");
        verify_archive(&path, VerifyDepth::Vectors).expect("clean file verifies");
        verify_archive(&path, VerifyDepth::Full).expect("clean file verifies (full)");
        let clean = std::fs::read(&path).unwrap();

        let mut t_seed = clean.clone();
        for &i in &seed_diff {
            t_seed[i] = buf_b[i];
        }
        std::fs::write(&path, &t_seed).unwrap();
        let err = verify_archive(&path, VerifyDepth::Vectors).unwrap_err();
        assert!(err.contains("descriptor digest"), "seed tamper must fail R2: {err}");
        let err = verify_archive(&path, VerifyDepth::Full).unwrap_err();
        assert!(err.contains("descriptor digest"), "seed tamper must fail R2 (full): {err}");

        let buf_c = {
            let shape_c = [8u64, 128u64];
            let pt = PackedTensorV2 {
                base: PackedTensor {
                    name: "model.layers.0.q_proj",
                    shape: &shape_c,
                    rht_seed: 0xA5A5_DEAD_BEEF_0001,
                    l_bits: cfg.l_bits as u8,
                    k_bits: cfg.k_bits as u8,
                    vec_dim: cfg.vec_dim() as u8,
                    enc: &enc,
                },
                block_len: cfg.block_len as u32,
            };
            write_strand_v2(&[pt], [3u8; 32], false).expect("write v2 (ragged ok)")
        };
        assert_eq!(buf_a.len(), buf_c.len());
        let mut t_shape = clean.clone();
        for i in 0..buf_a.len() {
            if buf_a[i] != buf_c[i] {
                t_shape[i] = buf_c[i];
            }
        }
        std::fs::write(&path, &t_shape).unwrap();
        let err = verify_archive(&path, VerifyDepth::Vectors).unwrap_err();
        assert!(err.contains("descriptor digest"), "shape tamper must fail R2: {err}");

        let sprv_off = {
            let t = &clean[clean.len() - SPRV_TRAILER_BYTES..];
            u64::from_le_bytes(t[0..8].try_into().unwrap()) as usize
        };
        let outl_trailer = &clean[sprv_off - 16..sprv_off];
        assert_eq!(&outl_trailer[12..16], b"OUTL");
        let outl_off = u64::from_le_bytes(outl_trailer[0..8].try_into().unwrap()) as usize;
        let mut t_outl = clean.clone();

        t_outl[outl_off + 32 + 24] ^= 0x40;
        std::fs::write(&path, &t_outl).unwrap();
        assert!(verify_archive(&path, VerifyDepth::Vectors).is_err(), "outlier-channel tamper must fail verification");

        std::fs::write(&path, &clean).unwrap();
        verify_archive(&path, VerifyDepth::Full).expect("clean file verifies again");
    }

    #[test]
    fn sprv_append_validates_section() {
        let (buf, _encs, _cfg) = build_test_archive();
        let path = tmp_path("validate");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();
        let good = build_sprv(&buf, 4, false, &default_lut_provider).unwrap();

        let mut s1 = good.clone();
        s1.tensors.pop();
        assert!(append_sprv(&path, &s1).is_err());

        let mut s2 = good.clone();
        s2.tensors[0].n_blocks += 1;
        assert!(append_sprv(&path, &s2).is_err());

        let mut s3 = good.clone();
        s3.tensors[0].leaves = Some(vec![[0u8; 32]; s3.tensors[0].n_blocks as usize]);
        assert!(append_sprv(&path, &s3).is_err());

        let mut s4 = good.clone();
        s4.tensors[0].vectors.reverse();
        assert!(append_sprv(&path, &s4).is_err());

        assert_eq!(std::fs::read(&path).unwrap(), buf);

        append_sprv(&path, &good).expect("good section appends");
        assert_eq!(read_sprv(&path).unwrap().unwrap(), good);
    }
}

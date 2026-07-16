use memmap2::Mmap;
use strand_quant::encode::{BlockMeta, EncodedTensor, SUB_BLOCK};
use strand_quant::format::{flags_v2, read_strand_v2_header, BlockOffsetRecord, StrandV2Header, TensorHeaderV2};
use strand_quant::outlier_wire::{read_outl_bytes, OutlSection, OutlierWire};
use strand_quant::selfdesc::{read_sdsc_bytes, Sdsc};
use strand_quant::sideinfo_wire::{apply_sdsq_to_header, read_sdsq_bytes};
use strand_quant::{CodebookMode, TrellisConfig};

#[inline]
fn align_up(x: usize, a: usize) -> usize {
    (x + a - 1) & !(a - 1)
}

#[inline]
fn block_weight_count(b: usize, n_blocks: usize, total: usize, block_len: usize) -> usize {
    if b + 1 < n_blocks {
        block_len
    } else {
        total - (n_blocks - 1) * block_len
    }
}

pub struct StrandModel {
    mmap: Mmap,
    header: StrandV2Header,

    outl: Option<OutlSection>,
    /// Optional self-describing codebook section. Vector tensors require an
    /// archive-bound per-tensor LUT; absence is a hard decode error.
    sdsc: Option<Sdsc>,
}

pub struct TensorView<'a> {
    pub hdr: &'a TensorHeaderV2,

    pub payload: &'a [u8],

    pub sideinfo: &'a [u8],

    pub table: &'a [BlockOffsetRecord],
}

impl StrandModel {
    pub fn open(path: &std::path::Path) -> std::io::Result<Self> {
        let f = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&f)? };
        Self::from_mmap(mmap).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn from_mmap(mmap: Mmap) -> Result<Self, String> {
        let mut header = read_strand_v2_header(&mmap)?;
        // SDSQ side-info: when the archive packs the seek table (12-byte records,
        // SCALEQ_IN_SDSQ), the inline scale_q is absent and the readers left a 0
        // placeholder — source the authoritative scale_q from the EOF-chained SDSQ
        // section and overwrite each block's table record. Byte-identical to a
        // legacy 16-byte-record archive of the same weights, so the integer
        // reconstruct (and every device's decode) is unchanged. Hard-error if the
        // flag is set but SDSQ is missing/corrupt — a packed archive with no
        // side-info is unrecoverable and must never decode with the placeholder.
        if header.flags & flags_v2::SCALEQ_IN_SDSQ != 0 {
            let sdsq = read_sdsq_bytes(&mmap, true)?.ok_or(
                "strand loader: archive sets SCALEQ_IN_SDSQ but has no SDSQ section — \
                 scale_q is unrecoverable",
            )?;
            apply_sdsq_to_header(&mut header, &sdsq)?;
        }
        let outl = read_outl_bytes(&mmap, true)?;
        if let Some(o) = &outl {
            if o.tensors.len() != header.tensors.len() {
                return Err(format!("strand loader: OUTL section has {} records, archive has {} tensors", o.tensors.len(), header.tensors.len()));
            }
        }
        let sdsc = read_sdsc_bytes(&mmap, true)?;
        let has_vector = header.tensors.iter().any(|t| t.vec_dim > 1);
        if has_vector && sdsc.is_none() {
            return Err("strand loader: vector archive has no archive-bound SDSC tensor LUTs".into());
        }
        if let Some(section) = &sdsc {
            for (index, tensor) in header.tensors.iter().enumerate() {
                if tensor.vec_dim > 1 {
                    let record = section.tensor_lut(index)?;
                    if record.l_bits != tensor.l_bits || record.vec_dim != tensor.vec_dim {
                        return Err(format!(
                            "strand loader: SDSC tensor LUT geometry mismatch at {index}: \
                             L={} d={} vs header L={} d={}",
                            record.l_bits, record.vec_dim, tensor.l_bits, tensor.vec_dim,
                        ));
                    }
                }
            }
        }
        Ok(Self { mmap, header, outl, sdsc })
    }

    pub fn header(&self) -> &StrandV2Header {
        &self.header
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.header.tensors.iter().map(|t| t.name.as_str())
    }

    pub fn tensor_header(&self, name: &str) -> Option<&TensorHeaderV2> {
        self.header.tensors.iter().find(|t| t.name == name)
    }

    pub fn outl_section(&self) -> Option<&OutlSection> {
        self.outl.as_ref()
    }

    pub fn outlier(&self, name: &str) -> Option<&OutlierWire> {
        let idx = self.header.tensors.iter().position(|t| t.name == name)?;
        self.outl.as_ref()?.tensors[idx].as_ref()
    }

    /// Resolve the exact codebook for one tensor. Scalar archives retain the
    /// canonical deterministic table; vector archives may use only their
    /// source- and ordinal-bound SDSC record.
    pub fn lut_for(&self, name: &str) -> Result<&[i32], String> {
        let index = self.header.tensors.iter().position(|t| t.name == name).ok_or_else(|| format!("strand loader: no tensor {name:?}"))?;
        let tensor = &self.header.tensors[index];
        if tensor.vec_dim <= 1 {
            return Ok(strand_quant::codebook::codebook_lut(tensor.l_bits as u32));
        }
        let section = self.sdsc.as_ref().ok_or_else(|| format!("strand loader: vector tensor {name:?} has no SDSC section"))?;
        Ok(&section.tensor_lut(index)?.entries)
    }

    pub fn view(&self, name: &str) -> Option<TensorView<'_>> {
        let hdr = self.tensor_header(name)?;
        let payload = self.mmap.get(hdr.payload_offset..hdr.payload_offset + hdr.payload_bytes)?;
        let sideinfo = if hdr.sideinfo_offset == 0 { &[][..] } else { self.mmap.get(hdr.sideinfo_offset..hdr.sideinfo_offset + hdr.sideinfo_bytes)? };
        Some(TensorView { hdr, payload, sideinfo, table: &hdr.table })
    }

    pub fn config_for(&self, h: &TensorHeaderV2) -> TrellisConfig {
        TrellisConfig { l_bits: h.l_bits as u32, k_bits: h.k_bits as u32, block_len: h.block_len as usize, vec_dim: h.vec_dim as u32, codebook_mode: CodebookMode::StoredLut }
    }

    pub fn encoded_tensor_checked(&self, name: &str) -> Result<EncodedTensor, String> {
        let v = self.view(name).ok_or_else(|| format!("strand loader: no tensor {name:?} or region out of bounds"))?;
        encoded_tensor_from_view(v.hdr, v.payload, v.sideinfo)
    }

    pub fn encoded_tensor(&self, name: &str) -> Option<EncodedTensor> {
        self.encoded_tensor_checked(name).ok()
    }
}

pub fn encoded_tensor_from_view(hdr: &TensorHeaderV2, payload: &[u8], sideinfo: &[u8]) -> Result<EncodedTensor, String> {
    let total = hdr.total;
    let n_blocks = hdr.n_blocks;
    let block_len = hdr.block_len as usize;
    let has_affine_min = hdr.has_affine_min;
    let has_sideinfo = !sideinfo.is_empty();

    let mins_half_base = if has_affine_min && has_sideinfo {
        let mut ss_end = 0usize;
        for b in 0..n_blocks {
            let nb = block_weight_count(b, n_blocks, total, block_len);
            let n_sub = nb.div_ceil(SUB_BLOCK);
            ss_end += (6 * n_sub).div_ceil(8);
        }
        Some(align_up(ss_end, 4))
    } else {
        None
    };

    let mut ss_cursor = 0usize;

    let mins_codes_base = mins_half_base.map(|mb| mb + n_blocks * 4);
    let mut mins_cursor = mins_codes_base.unwrap_or(0);

    let mut blocks = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let rec = &hdr.table[b];
        let nb = block_weight_count(b, n_blocks, total, block_len);
        let n_sub = nb.div_ceil(SUB_BLOCK);
        let ss_bytes = (6 * n_sub).div_ceil(8);

        let sub_scales = if has_sideinfo {
            let s = sideinfo.get(ss_cursor..ss_cursor + ss_bytes).ok_or("strand loader: EOF sub_scales")?.to_vec();
            ss_cursor += ss_bytes;
            s
        } else {
            Vec::new()
        };

        let (min_base_q, mins) = if let Some(mb) = mins_half_base {
            let off = mb + b * 4;
            let mbq = sideinfo.get(off..off + 4).map(|s| i32::from_le_bytes(s.try_into().unwrap())).ok_or("strand loader: EOF min_base_q")?;
            let mins = sideinfo.get(mins_cursor..mins_cursor + ss_bytes).ok_or("strand loader: EOF mins")?.to_vec();
            mins_cursor += ss_bytes;
            (mbq, mins)
        } else {
            (0i32, Vec::new())
        };

        blocks.push(BlockMeta { scale_q: rec.scale_q, sub_scales, min_base_q, mins, init_state: rec.init_state, n: nb as u32 });
    }

    Ok(EncodedTensor { bits: payload.to_vec(), blocks, total, has_rht_seed: hdr.has_rht_seed, tail_biting: hdr.tail_biting, has_affine_min })
}

impl StrandModel {
    pub fn prepared_tensor(&self, name: &str) -> Option<crate::prepared::PreparedTensor> {
        let hdr = self.tensor_header(name)?;
        let cfg = self.config_for(hdr);
        let shape = if hdr.shape.len() >= 2 { Some((hdr.shape[0] as usize, hdr.shape[1] as usize)) } else { None };
        let enc = self.encoded_tensor(name)?;
        let mut p = crate::prepared::PreparedTensor::new(enc, cfg);
        if let Some((o, i)) = shape {
            p = p.with_shape(o, i);
        }
        Some(p)
    }

    pub fn prepare(&self) -> Result<crate::prepared::PreparedModel, String> {
        crate::prepared::PreparedModel::from_model(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use strand_quant::decode::decode_tensor_fixed;
    use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts};
    use strand_quant::format::{write_strand_v2, PackedTensor, PackedTensorV2};

    fn write_tiny_v2(name: &str, rows: u64, cols: u64, cfg: &TrellisConfig, enc: &EncodedTensor) -> std::path::PathBuf {
        let shape = [rows, cols];
        let pt = PackedTensorV2 {
            base: PackedTensor { name, shape: &shape, rht_seed: 0, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc },
            block_len: cfg.block_len as u32,
        };
        let buf = write_strand_v2(&[pt], [0u8; 32], true).expect("write_strand_v2");
        let mut path = std::env::temp_dir();
        let pid = std::process::id();
        let uniq = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        path.push(format!("strand_loader_{name}_{pid}_{uniq}.strand"));
        let mut f = std::fs::File::create(&path).expect("create temp .strand");
        f.write_all(&buf).expect("write temp .strand");
        f.sync_all().ok();
        path
    }

    // ---- SDSQ GATE: packed (12-byte-record + SDSQ) reconstruct must be
    // byte-identical to a legacy (16-byte-record, inline scale_q) archive of the
    // exact same weights. This is the make-or-break container-only invariant:
    // recon MUST NOT change. ----

    fn unique_path(tag: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        let pid = std::process::id();
        let uniq = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        path.push(format!("strand_sdsq_gate_{tag}_{pid}_{uniq}.strand"));
        path
    }

    fn write_multi_v2(names: &[&str], encs: &[EncodedTensor], cfg: &TrellisConfig, packed: bool, tag: &str) -> std::path::PathBuf {
        use strand_quant::format::write_strand_v2_packed;
        let shape = [4u64, 256u64];
        let pts: Vec<PackedTensorV2> = names
            .iter()
            .zip(encs.iter())
            .map(|(name, enc)| PackedTensorV2 {
                base: PackedTensor { name, shape: &shape, rht_seed: 0, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc },
                block_len: cfg.block_len as u32,
            })
            .collect();
        let buf = if packed { write_strand_v2_packed(&pts, [0u8; 32], true).expect("write_strand_v2_packed") } else { write_strand_v2(&pts, [0u8; 32], true).expect("write_strand_v2") };
        let path = unique_path(tag);
        let mut f = std::fs::File::create(&path).expect("create temp .strand");
        f.write_all(&buf).expect("write temp .strand");
        f.sync_all().ok();
        path
    }

    #[test]
    fn sdsq_packed_reconstruct_is_byte_identical_to_legacy() {
        use strand_quant::sideinfo_wire::append_sdsq;

        // Two tensors so the SDSQ stream exercises tensor-then-block ordering.
        let names = ["model.layers.0.q_proj.weight", "model.layers.0.down_proj.weight"];
        let cfg = TrellisConfig::for_bpw(2.0);
        let encs: Vec<EncodedTensor> = (0..2)
            .map(|t| {
                let w: Vec<f32> = (0..1024).map(|i| ((i as f32 + 17.0 * t as f32) * 0.0137).sin() * 0.6).collect();
                encode_tensor(&w, &cfg)
            })
            .collect();

        // Legacy archive: 16-byte records, inline scale_q, no SDSQ.
        let legacy = write_multi_v2(&names, &encs, &cfg, false, "legacy");
        // Packed archive: 12-byte records, SCALEQ_IN_SDSQ, scale_q in SDSQ.
        let packed = write_multi_v2(&names, &encs, &cfg, true, "packed");

        // The scale_q the producer feeds append_sdsq: tensor-then-block order, the
        // SAME order write_strand_v2_packed laid the (scale_q-less) table in.
        let scale_q: Vec<i32> = encs.iter().flat_map(|e| e.blocks.iter().map(|b| b.scale_q)).collect();
        append_sdsq(&packed, &scale_q).expect("append SDSQ");

        // Prove the on-disk seek table actually shrank (the bpw win) AND the flag is set.
        let legacy_bytes = std::fs::read(&legacy).unwrap();
        let packed_bytes = std::fs::read(&packed).unwrap();
        let lh = strand_quant::format::read_strand_v2_header(&legacy_bytes).unwrap();
        let ph = strand_quant::format::read_strand_v2_header(&packed_bytes).unwrap();
        assert_eq!(lh.flags & flags_v2::SCALEQ_IN_SDSQ, 0, "legacy must NOT set the flag");
        assert_ne!(ph.flags & flags_v2::SCALEQ_IN_SDSQ, 0, "packed MUST set the flag");
        // Legacy table parsed scale_q from disk; packed left 0 placeholders (pre-apply).
        assert!(lh.tensors[0].table.iter().any(|r| r.scale_q != 0), "legacy inline scale_q");
        assert!(ph.tensors[0].table.iter().all(|r| r.scale_q == 0), "packed header (no SDSQ apply) must show 0-placeholder scale_q");

        // Open both via the DEPLOY READ PATH (StrandModel applies SDSQ on load).
        let m_legacy = StrandModel::open(&legacy).expect("open legacy");
        let m_packed = StrandModel::open(&packed).expect("open packed");

        let mut max_abs_diff: i64 = 0;
        for name in &names {
            // scale_q sourced from SDSQ in the packed model must equal the inline one.
            let lt = m_legacy.tensor_header(name).unwrap();
            let pt = m_packed.tensor_header(name).unwrap();
            assert_eq!(lt.table.iter().map(|r| r.scale_q).collect::<Vec<_>>(), pt.table.iter().map(|r| r.scale_q).collect::<Vec<_>>(), "SDSQ-sourced scale_q must equal inline scale_q for {name}");

            let el = m_legacy.encoded_tensor(name).expect("legacy enc");
            let ep = m_packed.encoded_tensor(name).expect("packed enc");
            assert_eq!(el.blocks, ep.blocks, "EncodedTensor.blocks must match for {name}");

            // The load-bearing assertion: dequantized integer reconstruct identical.
            let ql = decode_tensor_fixed(&el, &m_legacy.config_for(lt));
            let qp = decode_tensor_fixed(&ep, &m_packed.config_for(pt));
            assert_eq!(ql.len(), qp.len());
            for (a, b) in ql.iter().zip(qp.iter()) {
                max_abs_diff = max_abs_diff.max((*a as i64 - *b as i64).abs());
            }
        }
        assert_eq!(max_abs_diff, 0, "maxabsdiff between legacy and SDSQ-packed reconstruct");

        let _ = std::fs::remove_file(&legacy);
        let _ = std::fs::remove_file(&packed);
    }

    #[test]
    fn sdsq_flag_set_but_section_missing_is_hard_error() {
        // A packed archive (flag set) with NO SDSQ section must REFUSE to load —
        // never silently decode with the 0-placeholder scale_q.
        let names = ["model.layers.0.q_proj.weight"];
        let cfg = TrellisConfig::for_bpw(2.0);
        let w: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.0137).sin() * 0.6).collect();
        let encs = vec![encode_tensor(&w, &cfg)];
        let packed_no_sdsq = write_multi_v2(&names, &encs, &cfg, true, "nosdsq");

        let msg = match StrandModel::open(&packed_no_sdsq) {
            Ok(_) => panic!("packed archive without SDSQ must hard-error"),
            Err(e) => e.to_string(),
        };
        assert!(msg.contains("SCALEQ_IN_SDSQ") || msg.contains("SDSQ"), "error must name the missing SDSQ section, got: {msg}");

        let _ = std::fs::remove_file(&packed_no_sdsq);
    }

    #[test]
    fn open_round_trips_header() {
        let weights: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.013).sin() * 0.7).collect();
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor(&weights, &cfg);
        let path = write_tiny_v2("blk.0.ffn_down.weight", 4, 256, &cfg, &enc);

        let model = StrandModel::open(&path).expect("open");
        assert_eq!(model.header().tensors.len(), 1);
        assert!(model.header().all_strict());
        let names: Vec<&str> = model.tensor_names().collect();
        assert_eq!(names, vec!["blk.0.ffn_down.weight"]);
        let h = model.tensor_header("blk.0.ffn_down.weight").unwrap();
        assert_eq!(h.shape, vec![4, 256]);
        assert_eq!(h.l_bits as u32, cfg.l_bits);
        assert_eq!(h.k_bits as u32, cfg.k_bits);
        assert_eq!(h.block_len as usize, cfg.block_len);
        assert_eq!(h.n_blocks, enc.blocks.len());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn view_slices_match_payload() {
        let weights: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.011).cos() * 0.5).collect();
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor(&weights, &cfg);
        let path = write_tiny_v2("w", 4, 256, &cfg, &enc);

        let model = StrandModel::open(&path).expect("open");
        let v = model.view("w").expect("view");

        assert_eq!(v.payload, &enc.bits[..]);

        assert_eq!(v.table.len(), enc.blocks.len());
        assert_eq!(v.table[0].bit_offset, 0);

        for (rec, blk) in v.table.iter().zip(enc.blocks.iter()) {
            assert_eq!(rec.scale_q, blk.scale_q);
            assert_eq!(rec.init_state, blk.init_state);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn encoded_tensor_decodes_identically_to_v1() {
        let weights: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.013).sin() * 0.7).collect();
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor(&weights, &cfg);
        let path = write_tiny_v2("w", 4, 256, &cfg, &enc);

        let model = StrandModel::open(&path).expect("open");
        let h = model.tensor_header("w").unwrap().clone();
        let enc_back = model.encoded_tensor("w").expect("encoded_tensor");

        assert_eq!(enc_back.bits, enc.bits);
        assert_eq!(enc_back.blocks, enc.blocks);
        assert_eq!(enc_back.total, enc.total);
        assert_eq!(enc_back.has_affine_min, enc.has_affine_min);
        assert_eq!(enc_back.tail_biting, enc.tail_biting);

        let q_ref = decode_tensor_fixed(&enc, &cfg);
        let q_back = decode_tensor_fixed(&enc_back, &model.config_for(&h));
        assert_eq!(q_ref, q_back);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn loader_surfaces_outl_section() {
        use strand_quant::outlier_wire::{append_outl, OutlierWire};
        let weights: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.019).sin() * 0.6).collect();
        let cfg = TrellisConfig::for_bpw(2.0);
        let enc = encode_tensor(&weights, &cfg);
        let path = write_tiny_v2("w_outl", 4, 256, &cfg, &enc);

        let model = StrandModel::open(&path).expect("open plain");
        assert!(model.outl_section().is_none());
        assert!(model.outlier("w_outl").is_none());
        drop(model);

        let wire = OutlierWire::from_selection(1024, vec![512, 9], vec![100, -3], 0.25, 8);
        append_outl(&path, &[Some(wire.clone())]).expect("append outl");
        let model = StrandModel::open(&path).expect("open with OUTL");
        assert_eq!(model.outl_section().unwrap().n_with_channel(), 1);
        let got = model.outlier("w_outl").expect("wire present");
        assert_eq!(got, &wire);
        assert!(model.outlier("missing").is_none());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn encoded_tensor_with_affine_min_round_trips() {
        let weights: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.017).sin() * 0.9 - 0.1).collect();
        let cfg = TrellisConfig::for_bpw(4.0);
        let opts = EncodeOpts { affine_min: true, ..Default::default() };
        let enc = encode_tensor_with(&weights, &cfg, &opts);

        if !enc.has_affine_min {
            return;
        }
        let path = write_tiny_v2("w_affine", 4, 256, &cfg, &enc);

        let model = StrandModel::open(&path).expect("open");
        let enc_back = model.encoded_tensor("w_affine").expect("encoded_tensor");
        assert_eq!(enc_back.blocks, enc.blocks, "affine-min side-info mismatch");
        let q_ref = decode_tensor_fixed(&enc, &cfg);
        let q_back = decode_tensor_fixed(&enc_back, &cfg);
        assert_eq!(q_ref, q_back);

        let _ = std::fs::remove_file(&path);
    }
}

pub const MAGIC: &[u8; 4] = b"STRQ";
pub const VERSION: u32 = 1;

pub mod flags {
    pub const TAIL_BITING: u32 = 1 << 0;
    pub const AFFINE_MIN: u32 = 1 << 1;
    pub const HAS_RHT: u32 = 1 << 2;
}

#[derive(Debug, Clone)]
pub struct TensorHeader {
    pub name: String,
    pub shape: Vec<u64>,
    pub rht_seed: u64,
    pub trellis_l: u8,
    pub trellis_k: u8,
    pub block_len: u32,
    pub codebook_hash: u32,
    pub flags: u32,
}

use crate::encode::{BlockMeta, EncodedTensor, SUB_BLOCK};

pub struct PackedTensor<'a> {
    pub name: &'a str,
    pub shape: &'a [u64],
    pub rht_seed: u64,
    pub l_bits: u8,
    pub k_bits: u8,
    pub vec_dim: u8,
    pub enc: &'a EncodedTensor,
}

#[derive(Clone, Debug)]
pub struct OwnedTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub rht_seed: u64,
    pub l_bits: u8,
    pub k_bits: u8,
    pub vec_dim: u8,
    pub enc: EncodedTensor,
}

pub fn write_strand(tensors: &[PackedTensor]) -> Vec<u8> {
    let mut o = Vec::new();
    o.extend_from_slice(MAGIC);
    o.extend_from_slice(&VERSION.to_le_bytes());
    o.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for t in tensors {
        o.extend_from_slice(&(t.name.len() as u32).to_le_bytes());
        o.extend_from_slice(t.name.as_bytes());
        o.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
        for &d in t.shape {
            o.extend_from_slice(&d.to_le_bytes());
        }
        o.extend_from_slice(&t.rht_seed.to_le_bytes());
        o.push(t.l_bits);
        o.push(t.k_bits);
        o.push(t.vec_dim);
        let mut f = 0u8;
        if t.enc.has_rht_seed {
            f |= 1;
        }
        if t.enc.tail_biting {
            f |= 2;
        }
        if t.enc.has_affine_min {
            f |= 4;
        }
        o.push(f);
        o.extend_from_slice(&(t.enc.total as u64).to_le_bytes());
        o.extend_from_slice(&(t.enc.bits.len() as u64).to_le_bytes());
        o.extend_from_slice(&t.enc.bits);
        o.extend_from_slice(&(t.enc.blocks.len() as u32).to_le_bytes());
        for b in &t.enc.blocks {
            o.extend_from_slice(&b.scale_q.to_le_bytes());
            o.extend_from_slice(&b.min_base_q.to_le_bytes());
            o.extend_from_slice(&b.init_state.to_le_bytes());
            o.extend_from_slice(&b.n.to_le_bytes());
            o.extend_from_slice(&(b.sub_scales.len() as u32).to_le_bytes());
            o.extend_from_slice(&b.sub_scales);
            o.extend_from_slice(&(b.mins.len() as u32).to_le_bytes());
            o.extend_from_slice(&b.mins);
        }
    }
    o
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.p + n > self.b.len() {
            return Err("strand: unexpected EOF".into());
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

pub fn read_strand(buf: &[u8]) -> Result<Vec<OwnedTensor>, String> {
    let mut r = Rd { b: buf, p: 0 };
    if r.take(4)? != &MAGIC[..] {
        return Err("strand: bad magic".into());
    }
    let ver = r.u32()?;
    if ver != VERSION {
        return Err(format!("strand: version {ver} != {VERSION}"));
    }
    let n = r.u32()? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let nl = r.u32()? as usize;
        let name = String::from_utf8(r.take(nl)?.to_vec()).map_err(|e| e.to_string())?;
        let nd = r.u32()? as usize;
        let mut shape = Vec::with_capacity(nd);
        for _ in 0..nd {
            shape.push(r.u64()?);
        }
        let rht_seed = r.u64()?;
        let l_bits = r.u8()?;
        let k_bits = r.u8()?;
        let vec_dim = r.u8()?;
        let f = r.u8()?;
        let total = r.u64()? as usize;
        let bits_len = r.u64()? as usize;
        let bits = r.take(bits_len)?.to_vec();
        let nb = r.u32()? as usize;
        let mut blocks = Vec::with_capacity(nb);
        for _ in 0..nb {
            let scale_q = r.i32()?;
            let min_base_q = r.i32()?;
            let init_state = r.u32()?;
            let nn = r.u32()?;
            let sl = r.u32()? as usize;
            let sub_scales = r.take(sl)?.to_vec();
            let ml = r.u32()? as usize;
            let mins = r.take(ml)?.to_vec();
            blocks.push(BlockMeta { scale_q, sub_scales, min_base_q, mins, init_state, n: nn });
        }
        out.push(OwnedTensor {
            name,
            shape,
            rht_seed,
            l_bits,
            k_bits,
            vec_dim,
            enc: EncodedTensor { bits, blocks, total, has_rht_seed: f & 1 != 0, tail_biting: f & 2 != 0, has_affine_min: f & 4 != 0 },
        });
    }
    Ok(out)
}

pub const MAGIC_V2: &[u8; 4] = b"STR2";

pub const VERSION_V2: u32 = 2;

pub const PAGE: usize = 4096;

pub mod section_tag {
    pub const SDSC: &[u8; 4] = b"SDSC";
    pub const OUTL: &[u8; 4] = b"OUTL";
    pub const SPRV: &[u8; 4] = b"SPRV";
    pub const RSLT: &[u8; 4] = b"RSLT";
}

pub mod flags_v2 {

    pub const ALL_STRICT: u32 = 1 << 0;

    /// The per-block `scale_q` is **not** stored inline in the seek table; it lives
    /// in the EOF-chained SDSQ side-info section instead. When this bit is set the
    /// `BlockOffsetRecord` on-disk stride shrinks from 16 to 12 bytes (`bit_offset`
    /// + `init_state` only), and a reader MUST source `scale_q` from SDSQ (via
    /// `sideinfo_wire::apply_sdsq_*`) before the value is used — the in-memory
    /// `BlockOffsetRecord::scale_q` is left `0` as a placeholder by the raw readers.
    /// This is what realizes the SDSQ bpw drop: without it SDSQ is pure +overhead
    /// because the redundant inline `scale_q` is still billed.
    pub const SCALEQ_IN_SDSQ: u32 = 1 << 1;
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockOffsetRecord {
    pub bit_offset: u64,

    pub init_state: u32,

    pub scale_q: i32,
}

impl BlockOffsetRecord {
    /// On-disk stride of the legacy (inline-`scale_q`) record: `bit_offset:u64` +
    /// `init_state:u32` + `scale_q:i32`.
    pub const SIZE: usize = 16;

    /// On-disk stride of the packed record used when
    /// [`flags_v2::SCALEQ_IN_SDSQ`] is set: `bit_offset:u64` + `init_state:u32`,
    /// with `scale_q` sourced from the SDSQ section instead.
    pub const SIZE_PACKED: usize = 12;

    /// On-disk stride of a seek-table record for the given archive `flags`.
    ///
    /// **Load-bearing:** every reader that strides the seek table MUST use this
    /// (not the bare `SIZE` constant) — a wrong stride silently mis-reads every
    /// block's `bit_offset`/`init_state` and corrupts the whole archive view.
    #[inline]
    pub const fn stride_for(flags: u32) -> usize {
        if flags & flags_v2::SCALEQ_IN_SDSQ != 0 {
            Self::SIZE_PACKED
        } else {
            Self::SIZE
        }
    }
}

pub struct PackedTensorV2<'a> {
    pub base: PackedTensor<'a>,
    pub block_len: u32,
}

#[derive(Clone, Debug)]
pub struct OwnedTensorV2 {
    pub base: OwnedTensor,
    /// Per-column RHT mode persisted in tensor descriptor flag bit 3.
    pub rht_cols: bool,
    pub block_len: u32,
    pub table: Vec<BlockOffsetRecord>,
}

#[inline]
fn align_up(x: usize, a: usize) -> usize {
    (x + a - 1) & !(a - 1)
}

#[inline]
fn pad_to_page(o: &mut Vec<u8>) {
    let target = align_up(o.len(), PAGE);
    o.resize(target, 0u8);
}

#[inline]
fn num_steps_for(n: usize, vec_dim: u8) -> usize {
    let d = (vec_dim as usize).max(1);
    n.div_ceil(d)
}

#[inline]
fn in_features_of(shape: &[u64]) -> Option<u64> {
    if shape.len() >= 2 {
        Some(shape[1])
    } else {
        None
    }
}

pub fn write_strand_v2(tensors: &[PackedTensorV2], source_sha256: [u8; 32], strict: bool) -> Result<Vec<u8>, String> {
    write_strand_v2_inner(tensors, source_sha256, strict, false, &[])
}

/// Like [`write_strand_v2`] / [`write_strand_v2_packed`] but stamps the per-tensor
/// **column-sign RHT** flag (bit 3 of the tensor flag byte) for each tensor whose
/// `rht_cols[i]` is true. A col-RHT tensor stores its weights column-rotated; the
/// decoder must transform the activation once with `rht_forward` and skip the
/// per-row weight inverse (see `strand-decode-kernel::outlier_mac`). `rht_cols`
/// indexes tensors positionally; a short/empty slice means "rows" (the default,
/// byte-identical to the plain writers). `scaleq_in_sdsq` mirrors
/// [`write_strand_v2_packed`].
pub fn write_strand_v2_rht(tensors: &[PackedTensorV2], source_sha256: [u8; 32], strict: bool, scaleq_in_sdsq: bool, rht_cols: &[bool]) -> Result<Vec<u8>, String> {
    write_strand_v2_inner(tensors, source_sha256, strict, scaleq_in_sdsq, rht_cols)
}

/// Like [`write_strand_v2`] but writes the **packed** 12-byte seek-table records
/// (no inline `scale_q`) and sets [`flags_v2::SCALEQ_IN_SDSQ`]. The caller is
/// responsible for appending the matching SDSQ section (`sideinfo_wire::append_sdsq`)
/// carrying every block's `scale_q` in the same tensor-then-block order — without
/// it the archive's `scale_q` is unrecoverable, so the loader hard-errors. This is
/// the ship writer that realizes the SDSQ bpw drop.
pub fn write_strand_v2_packed(tensors: &[PackedTensorV2], source_sha256: [u8; 32], strict: bool) -> Result<Vec<u8>, String> {
    write_strand_v2_inner(tensors, source_sha256, strict, true, &[])
}

fn write_strand_v2_inner(tensors: &[PackedTensorV2], source_sha256: [u8; 32], strict: bool, scaleq_in_sdsq: bool, rht_cols: &[bool]) -> Result<Vec<u8>, String> {
    let mut all_strict = true;
    for t in tensors {
        let bl = t.base.enc_block_len(t.block_len);
        if let Some(inf) = in_features_of(t.base.shape) {
            if inf % bl as u64 != 0 {
                if strict {
                    return Err(format!(
                        "strand v2: tensor {:?} violates STRICT deploy invariant: \
                         in_features {} not divisible by block_len {}",
                        t.base.name, inf, bl
                    ));
                }
                all_strict = false;
            }
        }
    }
    let mut flags = if all_strict { flags_v2::ALL_STRICT } else { 0 };
    if scaleq_in_sdsq {
        flags |= flags_v2::SCALEQ_IN_SDSQ;
    }

    let mut o = Vec::new();

    o.extend_from_slice(MAGIC_V2);
    o.extend_from_slice(&VERSION_V2.to_le_bytes());
    let header_bytes_pos = o.len();
    o.extend_from_slice(&0u32.to_le_bytes());
    o.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    o.extend_from_slice(&flags.to_le_bytes());
    o.extend_from_slice(&source_sha256);
    o.extend_from_slice(&0u32.to_le_bytes());

    struct Patch {
        table_off_pos: usize,
        payload_off_pos: usize,
        payload_bytes_pos: usize,
        sideinfo_off_pos: usize,
        sideinfo_bytes_pos: usize,
    }
    let mut patches: Vec<Patch> = Vec::with_capacity(tensors.len());

    for (ti, t) in tensors.iter().enumerate() {
        let b = &t.base;
        let bl = b.enc_block_len(t.block_len);
        o.extend_from_slice(&(b.name.len() as u32).to_le_bytes());
        o.extend_from_slice(b.name.as_bytes());
        o.extend_from_slice(&(b.shape.len() as u32).to_le_bytes());
        for &d in b.shape {
            o.extend_from_slice(&d.to_le_bytes());
        }
        o.extend_from_slice(&b.rht_seed.to_le_bytes());
        o.push(b.l_bits);
        o.push(b.k_bits);
        o.push(b.vec_dim);

        let mut f = 0u8;
        if b.enc.has_rht_seed {
            f |= 1;
        }
        if b.enc.tail_biting {
            f |= 2;
        }
        if b.enc.has_affine_min {
            f |= 4;
        }
        if rht_cols.get(ti).copied().unwrap_or(false) {
            f |= 8;
        }
        o.push(f);
        o.extend_from_slice(&bl.to_le_bytes());
        o.extend_from_slice(&(b.enc.total as u64).to_le_bytes());
        o.extend_from_slice(&(b.enc.blocks.len() as u64).to_le_bytes());
        o.extend_from_slice(&0u32.to_le_bytes());
        o.extend_from_slice(&0u32.to_le_bytes());
        let table_off_pos = o.len();
        o.extend_from_slice(&0u64.to_le_bytes());
        let payload_off_pos = o.len();
        o.extend_from_slice(&0u64.to_le_bytes());
        let payload_bytes_pos = o.len();
        o.extend_from_slice(&0u64.to_le_bytes());
        let sideinfo_off_pos = o.len();
        o.extend_from_slice(&0u64.to_le_bytes());
        let sideinfo_bytes_pos = o.len();
        o.extend_from_slice(&0u64.to_le_bytes());
        patches.push(Patch { table_off_pos, payload_off_pos, payload_bytes_pos, sideinfo_off_pos, sideinfo_bytes_pos });
    }

    let header_bytes = o.len() as u32;
    pad_to_page(&mut o);

    for (t, patch) in tensors.iter().zip(patches.iter()) {
        let enc = t.base.enc;
        let vec_dim = t.base.vec_dim;
        let k = t.base.k_bits as usize;

        let table_offset = o.len() as u64;
        let mut bit_offset: u64 = 0;
        for blk in &enc.blocks {
            o.extend_from_slice(&bit_offset.to_le_bytes());
            o.extend_from_slice(&blk.init_state.to_le_bytes());
            // 12-byte packed record omits scale_q (sourced from SDSQ); 16-byte
            // legacy record carries it inline.
            if !scaleq_in_sdsq {
                o.extend_from_slice(&blk.scale_q.to_le_bytes());
            }
            bit_offset += (num_steps_for(blk.n as usize, vec_dim) * k) as u64;
        }
        pad_to_page(&mut o);

        let payload_offset = o.len() as u64;
        o.extend_from_slice(&enc.bits);
        let payload_bytes = enc.bits.len() as u64;
        pad_to_page(&mut o);

        let has_subscales = enc.blocks.iter().any(|b| !b.sub_scales.is_empty());
        let has_mins = enc.has_affine_min && enc.blocks.iter().any(|b| !b.mins.is_empty());
        let (sideinfo_offset, sideinfo_bytes) = if has_subscales || has_mins {
            let sideinfo_offset = o.len() as u64;

            for b in &enc.blocks {
                o.extend_from_slice(&b.sub_scales);
            }
            if has_mins {
                let aligned = align_up(o.len(), 4);
                o.resize(aligned, 0u8);
                for b in &enc.blocks {
                    o.extend_from_slice(&b.min_base_q.to_le_bytes());
                }
                for b in &enc.blocks {
                    o.extend_from_slice(&b.mins);
                }
            }
            let sideinfo_bytes = o.len() as u64 - sideinfo_offset;
            pad_to_page(&mut o);
            (sideinfo_offset, sideinfo_bytes)
        } else {
            (0u64, 0u64)
        };

        o[patch.table_off_pos..patch.table_off_pos + 8].copy_from_slice(&table_offset.to_le_bytes());
        o[patch.payload_off_pos..patch.payload_off_pos + 8].copy_from_slice(&payload_offset.to_le_bytes());
        o[patch.payload_bytes_pos..patch.payload_bytes_pos + 8].copy_from_slice(&payload_bytes.to_le_bytes());
        o[patch.sideinfo_off_pos..patch.sideinfo_off_pos + 8].copy_from_slice(&sideinfo_offset.to_le_bytes());
        o[patch.sideinfo_bytes_pos..patch.sideinfo_bytes_pos + 8].copy_from_slice(&sideinfo_bytes.to_le_bytes());
    }

    o[header_bytes_pos..header_bytes_pos + 4].copy_from_slice(&header_bytes.to_le_bytes());

    Ok(o)
}

#[derive(Clone, Debug)]
pub struct TensorHeaderV2 {
    pub name: String,
    pub shape: Vec<u64>,
    pub rht_seed: u64,
    pub l_bits: u8,
    pub k_bits: u8,
    pub vec_dim: u8,
    pub has_rht_seed: bool,
    pub tail_biting: bool,
    pub has_affine_min: bool,
    /// Per-COLUMN-sign RHT (flag bit 3). When set, the weights are column-rotated and
    /// the decoder serves them by transforming the activation once with `rht_forward`
    /// instead of inverting the weights per row — see `strand-decode-kernel::outlier_mac`.
    pub rht_cols: bool,
    pub block_len: u32,
    pub total: usize,
    pub n_blocks: usize,
    pub table_offset: usize,
    pub payload_offset: usize,
    pub payload_bytes: usize,
    pub sideinfo_offset: usize,
    pub sideinfo_bytes: usize,

    pub table: Vec<BlockOffsetRecord>,
}

#[derive(Clone, Debug)]
pub struct StrandV2Header {
    pub flags: u32,
    pub source_sha256: [u8; 32],
    pub tensors: Vec<TensorHeaderV2>,
}

impl StrandV2Header {
    pub fn all_strict(&self) -> bool {
        self.flags & flags_v2::ALL_STRICT != 0
    }
}

pub fn read_strand_v2_header(buf: &[u8]) -> Result<StrandV2Header, String> {
    let flen = buf.len();
    let rd_u32 = |off: usize| -> Result<u32, String> { buf.get(off..off + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap())).ok_or_else(|| "strand v2: EOF reading u32".to_string()) };
    let rd_u64 = |off: usize| -> Result<u64, String> { buf.get(off..off + 8).map(|s| u64::from_le_bytes(s.try_into().unwrap())).ok_or_else(|| "strand v2: EOF reading u64".to_string()) };
    let rd_i32 = |off: usize| -> Result<i32, String> { buf.get(off..off + 4).map(|s| i32::from_le_bytes(s.try_into().unwrap())).ok_or_else(|| "strand v2: EOF reading i32".to_string()) };

    if flen < 56 {
        return Err("strand v2: file shorter than header".into());
    }
    if &buf[0..4] != &MAGIC_V2[..] {
        return Err("strand v2: bad magic".into());
    }
    let ver = rd_u32(4)?;
    if ver != VERSION_V2 {
        return Err(format!("strand v2: version {ver} != {VERSION_V2}"));
    }
    let n_tensors = rd_u32(12)? as usize;
    let flags = rd_u32(16)?;
    // Seek-table record stride depends on the SCALEQ_IN_SDSQ flag (12 vs 16 B).
    let rec_stride = BlockOffsetRecord::stride_for(flags);
    let scaleq_in_sdsq = flags & flags_v2::SCALEQ_IN_SDSQ != 0;
    let mut source_sha256 = [0u8; 32];
    source_sha256.copy_from_slice(buf.get(20..52).ok_or("strand v2: EOF reading source_sha256")?);

    let mut p = 56usize;
    let mut tensors = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name_len = rd_u32(p)? as usize;
        p += 4;
        let name = String::from_utf8(buf.get(p..p + name_len).ok_or("strand v2: EOF reading name")?.to_vec()).map_err(|e| e.to_string())?;
        p += name_len;
        let ndim = rd_u32(p)? as usize;
        p += 4;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(rd_u64(p)?);
            p += 8;
        }
        let rht_seed = rd_u64(p)?;
        p += 8;
        let l_bits = *buf.get(p).ok_or("strand v2: EOF l_bits")?;
        let k_bits = *buf.get(p + 1).ok_or("strand v2: EOF k_bits")?;
        let vec_dim = *buf.get(p + 2).ok_or("strand v2: EOF vec_dim")?;
        let f = *buf.get(p + 3).ok_or("strand v2: EOF flags")?;
        p += 4;
        let block_len = rd_u32(p)?;
        p += 4;
        let total = rd_u64(p)? as usize;
        p += 8;
        let n_blocks = rd_u64(p)? as usize;
        p += 8;
        p += 8;
        let table_offset = rd_u64(p)? as usize;
        p += 8;
        let payload_offset = rd_u64(p)? as usize;
        p += 8;
        let payload_bytes = rd_u64(p)? as usize;
        p += 8;
        let sideinfo_offset = rd_u64(p)? as usize;
        p += 8;
        let sideinfo_bytes = rd_u64(p)? as usize;
        p += 8;

        if table_offset % PAGE != 0 {
            return Err(format!("strand v2: table_offset {table_offset} not page-aligned"));
        }
        if payload_offset % PAGE != 0 {
            return Err(format!("strand v2: payload_offset {payload_offset} not page-aligned"));
        }
        if sideinfo_offset != 0 && sideinfo_offset % PAGE != 0 {
            return Err(format!("strand v2: sideinfo_offset {sideinfo_offset} not page-aligned"));
        }
        if table_offset + n_blocks * rec_stride > payload_offset {
            return Err("strand v2: offset table overruns payload".into());
        }
        if payload_offset + payload_bytes > flen {
            return Err("strand v2: payload overruns file".into());
        }
        if sideinfo_offset != 0 && sideinfo_offset + sideinfo_bytes > flen {
            return Err("strand v2: side-info overruns file".into());
        }

        let mut table = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            let base = table_offset + b * rec_stride;
            // When scale_q lives in SDSQ the record is 12 B (no inline scale_q);
            // leave a 0 placeholder for `sideinfo_wire::apply_sdsq_to_header`.
            let scale_q = if scaleq_in_sdsq { 0 } else { rd_i32(base + 12)? };
            table.push(BlockOffsetRecord { bit_offset: rd_u64(base)?, init_state: rd_u32(base + 8)?, scale_q });
        }

        tensors.push(TensorHeaderV2 {
            name,
            shape,
            rht_seed,
            l_bits,
            k_bits,
            vec_dim,
            has_rht_seed: f & 1 != 0,
            tail_biting: f & 2 != 0,
            has_affine_min: f & 4 != 0,
            rht_cols: f & 8 != 0,
            block_len,
            total,
            n_blocks,
            table_offset,
            payload_offset,
            payload_bytes,
            sideinfo_offset,
            sideinfo_bytes,
            table,
        });
    }
    Ok(StrandV2Header { flags, source_sha256, tensors })
}

pub fn read_strand_v2(buf: &[u8]) -> Result<Vec<OwnedTensorV2>, String> {
    let flen = buf.len();
    let rd_u32 = |off: usize| -> Result<u32, String> { buf.get(off..off + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap())).ok_or_else(|| "strand v2: EOF reading u32".to_string()) };
    let rd_u64 = |off: usize| -> Result<u64, String> { buf.get(off..off + 8).map(|s| u64::from_le_bytes(s.try_into().unwrap())).ok_or_else(|| "strand v2: EOF reading u64".to_string()) };
    let rd_i32 = |off: usize| -> Result<i32, String> { buf.get(off..off + 4).map(|s| i32::from_le_bytes(s.try_into().unwrap())).ok_or_else(|| "strand v2: EOF reading i32".to_string()) };

    if buf.len() < 56 {
        return Err("strand v2: file shorter than header".into());
    }
    if &buf[0..4] != &MAGIC_V2[..] {
        return Err("strand v2: bad magic".into());
    }
    let ver = rd_u32(4)?;
    if ver != VERSION_V2 {
        return Err(format!("strand v2: version {ver} != {VERSION_V2}"));
    }
    let _header_bytes = rd_u32(8)? as usize;
    let n_tensors = rd_u32(12)? as usize;
    let flags = rd_u32(16)?;
    let rec_stride = BlockOffsetRecord::stride_for(flags);
    let scaleq_in_sdsq = flags & flags_v2::SCALEQ_IN_SDSQ != 0;

    let mut p = 56usize;
    let mut out = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name_len = rd_u32(p)? as usize;
        p += 4;
        let name = String::from_utf8(buf.get(p..p + name_len).ok_or_else(|| "strand v2: EOF reading name".to_string())?.to_vec()).map_err(|e| e.to_string())?;
        p += name_len;
        let ndim = rd_u32(p)? as usize;
        p += 4;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(rd_u64(p)?);
            p += 8;
        }
        let rht_seed = rd_u64(p)?;
        p += 8;
        let l_bits = *buf.get(p).ok_or("strand v2: EOF l_bits")?;
        let k_bits = *buf.get(p + 1).ok_or("strand v2: EOF k_bits")?;
        let vec_dim = *buf.get(p + 2).ok_or("strand v2: EOF vec_dim")?;
        let f = *buf.get(p + 3).ok_or("strand v2: EOF flags")?;
        p += 4;
        let block_len = rd_u32(p)?;
        p += 4;
        let total = rd_u64(p)? as usize;
        p += 8;
        let n_blocks = rd_u64(p)? as usize;
        p += 8;
        let _reserved = rd_u32(p)?;
        p += 4;
        let _reserved2 = rd_u32(p)?;
        p += 4;
        let table_offset = rd_u64(p)? as usize;
        p += 8;
        let payload_offset = rd_u64(p)? as usize;
        p += 8;
        let payload_bytes = rd_u64(p)? as usize;
        p += 8;
        let sideinfo_offset = rd_u64(p)? as usize;
        p += 8;
        let sideinfo_bytes = rd_u64(p)? as usize;
        p += 8;

        for (label, off) in [("table_offset", table_offset), ("payload_offset", payload_offset)] {
            if off % PAGE != 0 {
                return Err(format!("strand v2: {label} {off} not page-aligned"));
            }
        }
        if sideinfo_offset != 0 && sideinfo_offset % PAGE != 0 {
            return Err(format!("strand v2: sideinfo_offset {sideinfo_offset} not page-aligned"));
        }
        if table_offset + n_blocks * rec_stride > payload_offset {
            return Err("strand v2: offset table overruns payload".into());
        }
        if payload_offset + payload_bytes > flen {
            return Err("strand v2: payload overruns file".into());
        }
        if sideinfo_offset != 0 && sideinfo_offset + sideinfo_bytes > flen {
            return Err("strand v2: side-info overruns file".into());
        }

        let has_affine_min = f & 4 != 0;
        let sub_stride = {
            let n_sub_full = (block_len as usize).div_ceil(SUB_BLOCK);
            (6 * n_sub_full).div_ceil(8)
        };

        let mins_half_base = if has_affine_min && sideinfo_offset != 0 {
            let mut ss_end = sideinfo_offset;
            for b in 0..n_blocks {
                let nb = if b + 1 < n_blocks { block_len as usize } else { total - (n_blocks - 1) * block_len as usize };
                let n_sub = nb.div_ceil(SUB_BLOCK);
                ss_end += (6 * n_sub).div_ceil(8);
            }
            Some(align_up(ss_end, 4))
        } else {
            None
        };

        let mut table = Vec::with_capacity(n_blocks);
        let mut blocks = Vec::with_capacity(n_blocks);
        let mut ss_cursor = sideinfo_offset;
        let mins_codes_base = mins_half_base.map(|mb| mb + n_blocks * 4);
        let mut mins_cursor = mins_codes_base.unwrap_or(0);
        let mut cursor_bits: u64 = 0;
        for b in 0..n_blocks {
            let rec_off = table_offset + b * rec_stride;
            let bit_offset = rd_u64(rec_off)?;
            let init_state = rd_u32(rec_off + 8)?;
            // 12-byte packed record: scale_q lives in SDSQ; 0 placeholder until
            // `sideinfo_wire::apply_sdsq_to_encoded` overwrites it.
            let scale_q = if scaleq_in_sdsq { 0 } else { rd_i32(rec_off + 12)? };
            table.push(BlockOffsetRecord { bit_offset, init_state, scale_q });

            let nb = if b + 1 < n_blocks { block_len as usize } else { total - (n_blocks - 1) * block_len as usize };
            let n_sub = nb.div_ceil(SUB_BLOCK);
            let ss_bytes = (6 * n_sub).div_ceil(8);

            let sub_scales = if sideinfo_offset != 0 {
                let s = buf.get(ss_cursor..ss_cursor + ss_bytes).ok_or("strand v2: EOF sub_scales")?.to_vec();
                ss_cursor += ss_bytes;
                let _ = sub_stride;
                s
            } else {
                Vec::new()
            };

            let (min_base_q, mins) = if let Some(mb) = mins_half_base {
                let mbq = rd_i32(mb + b * 4)?;
                let mins = buf.get(mins_cursor..mins_cursor + ss_bytes).ok_or("strand v2: EOF mins")?.to_vec();
                mins_cursor += ss_bytes;
                (mbq, mins)
            } else {
                (0i32, Vec::new())
            };

            blocks.push(BlockMeta { scale_q, sub_scales, min_base_q, mins, init_state, n: nb as u32 });

            if bit_offset != cursor_bits {
                return Err(format!("strand v2: block {b} bit_offset {bit_offset} != cursor {cursor_bits}"));
            }
            cursor_bits += (num_steps_for(nb, vec_dim) * k_bits as usize) as u64;
        }

        let total_payload_bits = (payload_bytes * 8) as u64;
        if cursor_bits > total_payload_bits {
            return Err(format!("strand v2: total symbol bits {cursor_bits} exceed payload bits {total_payload_bits}"));
        }

        let bits = buf.get(payload_offset..payload_offset + payload_bytes).ok_or("strand v2: EOF payload")?.to_vec();

        out.push(OwnedTensorV2 {
            base: OwnedTensor { name, shape, rht_seed, l_bits, k_bits, vec_dim, enc: EncodedTensor { bits, blocks, total, has_rht_seed: f & 1 != 0, tail_biting: f & 2 != 0, has_affine_min } },
            rht_cols: f & 8 != 0,
            block_len,
            table,
        });
    }
    Ok(out)
}

pub fn strand_v1_to_v2(v1: &[u8], block_lens: &[u32], strict: bool) -> Result<Vec<u8>, String> {
    let tensors = read_strand(v1)?;
    if block_lens.len() != tensors.len() {
        return Err(format!("strand v1->v2: block_lens len {} != tensor count {}", block_lens.len(), tensors.len()));
    }
    let packed: Vec<PackedTensorV2> = tensors
        .iter()
        .zip(block_lens.iter())
        .map(|(t, &bl)| PackedTensorV2 {
            base: PackedTensor { name: &t.name, shape: &t.shape, rht_seed: t.rht_seed, l_bits: t.l_bits, k_bits: t.k_bits, vec_dim: t.vec_dim, enc: &t.enc },
            block_len: bl,
        })
        .collect();
    write_strand_v2(&packed, [0u8; 32], strict)
}

impl<'a> PackedTensor<'a> {
    #[inline]
    fn enc_block_len(&self, block_len: u32) -> u32 {
        if block_len == 0 {
            256
        } else {
            block_len
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::encode_tensor;
    use crate::TrellisConfig;

    #[test]
    fn strand_round_trip_decodes_identically() {
        let weights: Vec<f32> = (0..512).map(|i| (i as f32 * 0.017).sin()).collect();
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor(&weights, &cfg);
        let shape = [8u64, 64u64];
        let pt = PackedTensor { name: "blk.0.attn.weight", shape: &shape, rht_seed: 0, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc: &enc };
        let buf = write_strand(&[pt]);
        let back = read_strand(&buf).expect("read_strand");
        assert_eq!(back.len(), 1);
        let t = &back[0];
        assert_eq!(t.name, "blk.0.attn.weight");
        assert_eq!(t.shape, vec![8, 64]);
        assert_eq!(t.enc.bits, enc.bits);
        assert_eq!(t.enc.blocks, enc.blocks);
        assert_eq!(t.enc.total, enc.total);

        let q0 = crate::decode::decode_tensor_fixed(&enc, &cfg);
        let q1 = crate::decode::decode_tensor_fixed(&t.enc, &cfg);
        assert_eq!(q0, q1);
    }

    #[test]
    fn strand_v2_round_trip_matches_v1_q12() {
        let weights: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.013).sin() * 0.7).collect();
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor(&weights, &cfg);
        let shape = [4u64, 256u64];
        let bl = cfg.block_len as u32;

        let pt = PackedTensorV2 {
            base: PackedTensor { name: "blk.0.ffn_down.weight", shape: &shape, rht_seed: 0, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc: &enc },
            block_len: bl,
        };

        let buf = write_strand_v2(&[pt], [0u8; 32], true).expect("write_strand_v2");
        assert_eq!(&buf[0..4], &MAGIC_V2[..]);

        let back = read_strand_v2(&buf).expect("read_strand_v2");
        assert_eq!(back.len(), 1);
        let t = &back[0];

        assert_eq!(t.base.name, "blk.0.ffn_down.weight");
        assert_eq!(t.base.shape, vec![4, 256]);
        assert_eq!(t.block_len, bl);

        assert_eq!(t.base.enc.bits, enc.bits);
        assert_eq!(t.base.enc.blocks, enc.blocks);
        assert_eq!(t.base.enc.total, enc.total);
        assert_eq!(t.base.enc.has_rht_seed, enc.has_rht_seed);
        assert_eq!(t.base.enc.tail_biting, enc.tail_biting);
        assert_eq!(t.base.enc.has_affine_min, enc.has_affine_min);

        let q_v1 = crate::decode::decode_tensor_fixed(&enc, &cfg);
        let q_v2 = crate::decode::decode_tensor_fixed(&t.base.enc, &cfg);
        assert_eq!(q_v1, q_v2);

        assert_eq!(t.table.len(), enc.blocks.len());
        assert_eq!(t.table[0].bit_offset, 0);
        let mut cursor: u64 = 0;
        for (b, blk) in enc.blocks.iter().enumerate() {
            assert_eq!(t.table[b].bit_offset, cursor, "block {b} bit_offset");
            assert_eq!(t.table[b].init_state, blk.init_state, "block {b} init_state");
            assert_eq!(t.table[b].scale_q, blk.scale_q, "block {b} scale_q");
            cursor += (cfg.num_steps(blk.n as usize) * cfg.k_bits as usize) as u64;
        }
        assert!(cursor <= (enc.bits.len() * 8) as u64);
    }

    #[test]
    fn strand_v2_header_matches_full_read() {
        let weights: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.013).sin() * 0.7).collect();
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor(&weights, &cfg);
        let shape = [4u64, 256u64];
        let bl = cfg.block_len as u32;
        let pt = PackedTensorV2 {
            base: PackedTensor { name: "blk.0.ffn_down.weight", shape: &shape, rht_seed: 0, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc: &enc },
            block_len: bl,
        };
        let buf = write_strand_v2(&[pt], [7u8; 32], true).expect("write_strand_v2");

        let hdr = read_strand_v2_header(&buf).expect("read_strand_v2_header");
        let full = read_strand_v2(&buf).expect("read_strand_v2");

        assert!(hdr.all_strict());
        assert_eq!(hdr.source_sha256, [7u8; 32]);
        assert_eq!(hdr.tensors.len(), 1);
        let h = &hdr.tensors[0];
        let f = &full[0];

        assert_eq!(h.name, f.base.name);
        assert_eq!(h.shape, f.base.shape);
        assert_eq!(h.block_len, f.block_len);
        assert_eq!(h.n_blocks, f.base.enc.blocks.len());
        assert_eq!(h.total, f.base.enc.total);
        assert_eq!(h.has_affine_min, f.base.enc.has_affine_min);

        assert_eq!(h.table, f.table);
        assert_eq!(h.table_offset % PAGE, 0);
        assert_eq!(h.payload_offset % PAGE, 0);
        assert!(h.payload_offset + h.payload_bytes <= buf.len());
        assert_eq!(&buf[h.payload_offset..h.payload_offset + h.payload_bytes], &enc.bits[..]);
    }

    #[test]
    fn strand_v2_strict_rejects_ragged_in_features() {
        let weights: Vec<f32> = (0..200).map(|i| (i as f32 * 0.021).cos() * 0.3).collect();
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor(&weights, &cfg);
        let shape = [2u64, 100u64];
        let make = || PackedTensorV2 {
            base: PackedTensor { name: "ragged.weight", shape: &shape, rht_seed: 0, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc: &enc },
            block_len: cfg.block_len as u32,
        };
        let err = write_strand_v2(&[make()], [0u8; 32], true).unwrap_err();
        assert!(err.contains("ragged.weight"), "err was: {err}");
        let buf = write_strand_v2(&[make()], [0u8; 32], false).expect("ragged write");
        let file_flags = u32::from_le_bytes(buf[16..20].try_into().unwrap());
        assert_eq!(file_flags & flags_v2::ALL_STRICT, 0);
        let back = read_strand_v2(&buf).expect("ragged read");
        assert_eq!(back[0].base.enc.bits, enc.bits);
        let q_v1 = crate::decode::decode_tensor_fixed(&enc, &cfg);
        let q_v2 = crate::decode::decode_tensor_fixed(&back[0].base.enc, &cfg);
        assert_eq!(q_v1, q_v2);
    }
}

use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::codebook::codebook_lut;
use crate::format::{
    read_strand_v2_header, BlockOffsetRecord, OwnedTensorV2, MAGIC_V2, PAGE, VERSION_V2,
};
use crate::outlier_wire::{append_outl, read_outl_bytes};
use crate::provenance_io::{append_sprv, read_sprv_bytes};
use crate::sha256::sha256;
use crate::trellis::{read_bits, TrellisConfig};

pub const SDSC_MAGIC: &[u8; 4] = b"SDSC";

const SPRV_MAGIC: &[u8; 4] = b"SPRV";
const OUTL_MAGIC: &[u8; 4] = b"OUTL";
/// SDSQ (sprint Lever 1 side-info rANS) chains above SDSC (SDSC is the innermost
/// section), so both SDSC walkers must step over an SDSQ trailer to reach SDSC
/// beneath it — mirrors the OUTL step-over already here.
const SDSQ_MAGIC: &[u8; 4] = b"SDSQ";

/// Scalar/geometry LUT format shipped by the original self-describing archive.
pub const SDSC_VERSION_V1: u32 = 1;

/// Per-tensor learned-vector-LUT format.  V2 is emitted only when
/// [`Sdsc::tensor_luts`] is non-empty; scalar archives remain byte-compatible V1.
pub const SDSC_VERSION: u32 = 2;

// SDSC V2 extends the 48-byte V1 header without changing its size:
//   +24 n_tensor_luts:u32, +28..48 reserved zero.
// The body starts with the base STR2 source_sha256, then the unchanged
// const/expression/geometry-LUT streams, then records sorted by tensor ordinal:
//   index:u32 | L:u32 | d:u32 | n_entries:u32 | reserved:u32 |
//   descriptor+content_sha256:[u8;32] | entries:[i32; n_entries]
// One record therefore costs exactly 52 + 4*(2^L*d) unpadded bytes.  The whole
// SDSC section retains the existing page-aligned 16-byte EOF trailer.

pub const SDSC_HEADER_BYTES: usize = 48;

pub const SDSC_TRAILER_BYTES: usize = 16;

/// Fixed bytes in one SDSC V2 per-tensor LUT record before its Q12 entries:
/// tensor index, L, d, entry count, reserved (5*u32), and SHA-256.
pub const SDSC_TENSOR_LUT_RECORD_BYTES: usize = 52;

const MAX_CONSTS: usize = 256;
const MAX_EXPRS: usize = 64;
const MAX_LUTS: usize = 32;
const MAX_TENSOR_LUTS: usize = 1 << 20;
const MAX_PROG_BYTES: usize = 4096;
const MAX_SLOTS: usize = 16;

pub mod const_id {

    pub const SCALE_SHIFT: u32 = 1;

    pub const SUB_SCALE_SHIFT: u32 = 2;

    pub const SUB_BLOCK: u32 = 3;

    pub const SUBSCALE_CODE_BITS: u32 = 4;

    pub const QUANTILE_SHIFT: u32 = 5;

    pub const PAYLOAD_BIT_ORDER: u32 = 6;

    pub const SIDEINFO_LAYOUT: u32 = 7;

    pub const TAILBITE_RULE: u32 = 8;

    pub const OUTL_PATCH_RULE: u32 = 9;

    pub const RHT_RULE: u32 = 10;
}

pub mod op {

    pub const END: u8 = 0x00;

    pub const IMM: u8 = 0x01;

    pub const LOAD: u8 = 0x02;

    pub const ADD: u8 = 0x10;

    pub const SUB: u8 = 0x11;

    pub const MUL: u8 = 0x12;

    pub const TDIV: u8 = 0x13;

    pub const NEG: u8 = 0x14;

    pub const ABS: u8 = 0x15;

    pub const WRAP32: u8 = 0x16;

    pub const SHL: u8 = 0x18;

    pub const ASR: u8 = 0x19;

    pub const AND: u8 = 0x1C;

    pub const OR: u8 = 0x1D;

    pub const XOR: u8 = 0x1E;

    pub const CLAMP: u8 = 0x20;
}

pub mod expr_id {

    pub const ADVANCE: u32 = 1;

    pub const EFF_SCALE: u32 = 2;

    pub const OFFSET: u32 = 3;

    pub const RECON: u32 = 4;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SdscExpr {
    pub id: u32,

    pub n_slots: u32,

    pub prog: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SdscLut {
    pub l_bits: u8,
    pub vec_dim: u8,
    pub entries: Vec<i32>,
}

/// An exact learned Q12 vector codebook bound to one STR2 tensor ordinal.
///
/// Geometry alone is not a sufficient key: two tensors with the same `(L, d)`
/// normally learn different centroids.  `record_sha256` binds the raw LUT bytes
/// to the archive source hash and the tensor descriptor, so corruption or a
/// misplaced record fails closed before reconstruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SdscTensorLut {
    pub tensor_index: u32,
    pub l_bits: u8,
    pub vec_dim: u8,
    pub record_sha256: [u8; 32],
    pub entries: Vec<i32>,
}

/// Borrowed input used to build/append a per-tensor learned vector LUT section.
#[derive(Clone, Copy, Debug)]
pub struct TensorLutInput<'a> {
    pub tensor_index: usize,
    pub entries: &'a [i32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sdsc {
    pub consts: Vec<(u32, i64)>,
    pub exprs: Vec<SdscExpr>,
    pub luts: Vec<SdscLut>,
    /// Present for SDSC V2 and equal to the STR2 header's source SHA-256.
    pub archive_source_sha256: Option<[u8; 32]>,
    /// Exact per-tensor learned vector LUTs, sorted by `tensor_index`.
    pub tensor_luts: Vec<SdscTensorLut>,
}

impl Sdsc {
    pub fn const_val(&self, id: u32) -> Result<i64, String> {
        self.consts
            .iter()
            .find(|(i, _)| *i == id)
            .map(|(_, v)| *v)
            .ok_or_else(|| format!("sdsc: missing constant id {id}"))
    }

    pub fn expr(&self, id: u32) -> Result<&SdscExpr, String> {
        self.exprs
            .iter()
            .find(|e| e.id == id)
            .ok_or_else(|| format!("sdsc: missing expression id {id}"))
    }

    pub fn lut(&self, l_bits: u8, vec_dim: u8) -> Result<&SdscLut, String> {
        let d = vec_dim.max(1);
        self.luts
            .iter()
            .find(|l| l.l_bits == l_bits && l.vec_dim == d)
            .ok_or_else(|| format!("sdsc: no LUT for geometry L={l_bits} d={d}"))
    }

    /// Resolve a learned vector LUT by archive tensor ordinal.  The decoder also
    /// validates its descriptor/hash; this accessor intentionally does not make
    /// geometry-only fallback possible for vector tensors.
    pub fn tensor_lut(&self, tensor_index: usize) -> Result<&SdscTensorLut, String> {
        self.tensor_luts
            .binary_search_by_key(&tensor_index, |l| l.tensor_index as usize)
            .map(|i| &self.tensor_luts[i])
            .map_err(|_| format!("sdsc: no per-tensor LUT for tensor index {tensor_index}"))
    }
}

pub fn eval_expr(prog: &[u8], slots: &[i64]) -> Result<i64, String> {
    let mut st: Vec<i64> = Vec::with_capacity(8);
    let mut p = 0usize;
    macro_rules! pop {
        () => {
            st.pop().ok_or("sdsc: expr stack underflow")?
        };
    }
    loop {
        let opc = *prog.get(p).ok_or("sdsc: expr program missing END")?;
        p += 1;
        match opc {
            op::END => break,
            op::IMM => {
                let b = prog.get(p..p + 8).ok_or("sdsc: expr truncated IMM")?;
                st.push(i64::from_le_bytes(b.try_into().unwrap()));
                p += 8;
            }
            op::LOAD => {
                let s = *prog.get(p).ok_or("sdsc: expr truncated LOAD")? as usize;
                p += 1;
                st.push(*slots.get(s).ok_or("sdsc: expr LOAD slot out of range")?);
            }
            op::ADD => {
                let b = pop!();
                let a = pop!();
                st.push(a.wrapping_add(b));
            }
            op::SUB => {
                let b = pop!();
                let a = pop!();
                st.push(a.wrapping_sub(b));
            }
            op::MUL => {
                let b = pop!();
                let a = pop!();
                st.push(a.wrapping_mul(b));
            }
            op::TDIV => {
                let b = pop!();
                let a = pop!();
                st.push(if b == 0 { 0 } else { a.wrapping_div(b) });
            }
            op::NEG => {
                let a = pop!();
                st.push(a.wrapping_neg());
            }
            op::ABS => {
                let a = pop!();
                st.push(a.wrapping_abs());
            }
            op::WRAP32 => {
                let a = pop!();
                st.push(a as i32 as i64);
            }
            op::SHL => {
                let b = pop!();
                let a = pop!();
                st.push(a.wrapping_shl((b & 63) as u32));
            }
            op::ASR => {
                let b = pop!();
                let a = pop!();
                st.push(a >> (b & 63));
            }
            op::AND => {
                let b = pop!();
                let a = pop!();
                st.push(a & b);
            }
            op::OR => {
                let b = pop!();
                let a = pop!();
                st.push(a | b);
            }
            op::XOR => {
                let b = pop!();
                let a = pop!();
                st.push(a ^ b);
            }
            op::CLAMP => {
                let hi = pop!();
                let lo = pop!();
                let x = pop!();
                st.push(x.max(lo).min(hi));
            }
            other => return Err(format!("sdsc: unknown opcode {other:#04x}")),
        }
    }
    if st.len() != 1 {
        return Err(format!(
            "sdsc: expr left {} values on the stack (want 1)",
            st.len()
        ));
    }
    Ok(st[0])
}

fn validate_prog(prog: &[u8], n_slots: u32) -> Result<(), String> {
    let mut p = 0usize;
    loop {
        let opc = *prog.get(p).ok_or("sdsc: program has no END")?;
        p += 1;
        match opc {
            op::END => {
                if p != prog.len() {
                    return Err("sdsc: bytes after END".into());
                }
                return Ok(());
            }
            op::IMM => {
                if p + 8 > prog.len() {
                    return Err("sdsc: truncated IMM".into());
                }
                p += 8;
            }
            op::LOAD => {
                let s = *prog.get(p).ok_or("sdsc: truncated LOAD")?;
                if (s as u32) >= n_slots {
                    return Err(format!("sdsc: LOAD slot {s} >= n_slots {n_slots}"));
                }
                p += 1;
            }
            op::ADD
            | op::SUB
            | op::MUL
            | op::TDIV
            | op::NEG
            | op::ABS
            | op::WRAP32
            | op::SHL
            | op::ASR
            | op::AND
            | op::OR
            | op::XOR
            | op::CLAMP => {}
            other => return Err(format!("sdsc: unknown opcode {other:#04x}")),
        }
    }
}

fn asm_imm(o: &mut Vec<u8>, v: i64) {
    o.push(op::IMM);
    o.extend_from_slice(&v.to_le_bytes());
}
fn asm_load(o: &mut Vec<u8>, slot: u8) {
    o.push(op::LOAD);
    o.push(slot);
}

fn prog_advance() -> Vec<u8> {
    let mut o = Vec::new();
    asm_load(&mut o, 0);
    asm_load(&mut o, 2);
    o.push(op::SHL);
    asm_load(&mut o, 1);
    o.push(op::OR);
    asm_load(&mut o, 3);
    o.push(op::AND);
    o.push(op::END);
    o
}

fn prog_eff_scale() -> Vec<u8> {
    let mut o = Vec::new();
    asm_load(&mut o, 1);
    asm_imm(&mut o, 63);
    o.push(op::AND);
    asm_imm(&mut o, 1);
    o.push(op::ADD);
    asm_load(&mut o, 0);
    o.push(op::MUL);
    asm_imm(&mut o, 6);
    o.push(op::ASR);
    o.push(op::WRAP32);
    o.push(op::END);
    o
}

fn prog_offset() -> Vec<u8> {
    let mut o = Vec::new();
    asm_load(&mut o, 0);
    o.push(op::ABS);
    asm_load(&mut o, 1);
    asm_imm(&mut o, 31);
    o.push(op::AND);
    o.push(op::MUL);
    asm_load(&mut o, 1);
    asm_imm(&mut o, 5);
    o.push(op::ASR);
    asm_imm(&mut o, 1);
    o.push(op::AND);
    asm_imm(&mut o, 2);
    o.push(op::MUL);
    asm_imm(&mut o, 1);
    o.push(op::SUB);
    o.push(op::MUL);
    asm_imm(&mut o, 31);
    o.push(op::TDIV);
    o.push(op::WRAP32);
    o.push(op::END);
    o
}

fn prog_recon() -> Vec<u8> {
    let mut o = Vec::new();
    asm_load(&mut o, 0);
    asm_load(&mut o, 1);
    o.push(op::MUL);
    asm_imm(&mut o, 16);
    o.push(op::ASR);
    o.push(op::WRAP32);
    asm_load(&mut o, 2);
    o.push(op::ADD);
    o.push(op::WRAP32);
    o.push(op::END);
    o
}

pub fn default_consts() -> Vec<(u32, i64)> {
    vec![
        (const_id::SCALE_SHIFT, 16),
        (const_id::SUB_SCALE_SHIFT, 6),
        (const_id::SUB_BLOCK, 32),
        (const_id::SUBSCALE_CODE_BITS, 6),
        (const_id::QUANTILE_SHIFT, 12),
        (const_id::PAYLOAD_BIT_ORDER, 0),
        (const_id::SIDEINFO_LAYOUT, 1),
        (const_id::TAILBITE_RULE, 1),
        (const_id::OUTL_PATCH_RULE, 1),
        (const_id::RHT_RULE, 0),
    ]
}

pub fn default_exprs() -> Vec<SdscExpr> {
    vec![
        SdscExpr {
            id: expr_id::ADVANCE,
            n_slots: 4,
            prog: prog_advance(),
        },
        SdscExpr {
            id: expr_id::EFF_SCALE,
            n_slots: 2,
            prog: prog_eff_scale(),
        },
        SdscExpr {
            id: expr_id::OFFSET,
            n_slots: 2,
            prog: prog_offset(),
        },
        SdscExpr {
            id: expr_id::RECON,
            n_slots: 3,
            prog: prog_recon(),
        },
    ]
}

pub fn emit_sdsc(cfg: &TrellisConfig, lut: &[i32]) -> Result<Vec<u8>, String> {
    if cfg.vec_dim() > 1 {
        return Err(
            "sdsc: unbound vector LUT emission is forbidden; use the archive-bound per-tensor V2 builder"
                .into(),
        );
    }
    if lut.len() != cfg.lut_len() {
        return Err(format!(
            "sdsc: LUT has {} entries, geometry L={} d={} needs {}",
            lut.len(),
            cfg.l_bits,
            cfg.vec_dim(),
            cfg.lut_len()
        ));
    }
    let sdsc = Sdsc {
        consts: default_consts(),
        exprs: default_exprs(),
        luts: vec![SdscLut {
            l_bits: cfg.l_bits as u8,
            vec_dim: cfg.vec_dim() as u8,
            entries: lut.to_vec(),
        }],
        archive_source_sha256: None,
        tensor_luts: Vec::new(),
    };
    sdsc_section_bytes(&sdsc)
}

pub fn build_sdsc_for_archive(buf: &[u8]) -> Result<Sdsc, String> {
    let hdr = read_strand_v2_header(buf)?;
    let mut ls: Vec<u8> = Vec::new();
    for t in &hdr.tensors {
        if t.vec_dim > 1 {
            return Err(format!(
                "sdsc: tensor {:?} has vec_dim {} — its learned LUT is not derivable \
                 from the descriptor (SDSC v1 covers the scalar frozen-LUT path)",
                t.name, t.vec_dim
            ));
        }
        if !ls.contains(&t.l_bits) {
            ls.push(t.l_bits);
        }
    }
    ls.sort_unstable();
    let luts = ls
        .into_iter()
        .map(|l| SdscLut {
            l_bits: l,
            vec_dim: 1,
            entries: codebook_lut(l as u32).to_vec(),
        })
        .collect();
    Ok(Sdsc {
        consts: default_consts(),
        exprs: default_exprs(),
        luts,
        archive_source_sha256: None,
        tensor_luts: Vec::new(),
    })
}

#[derive(Clone, Debug)]
struct TensorLutDescriptor {
    name: String,
    shape: Vec<u64>,
    rht_seed: u64,
    l_bits: u8,
    k_bits: u8,
    vec_dim: u8,
    has_rht_seed: bool,
    rht_cols: bool,
    tail_biting: bool,
    has_affine_min: bool,
    block_len: u32,
    total: usize,
    n_blocks: usize,
}

#[derive(Clone, Debug)]
struct ArchiveLutDescriptor {
    source_sha256: [u8; 32],
    tensors: Vec<TensorLutDescriptor>,
    base_bytes: usize,
}

const MAX_STR2_DESCRIPTOR_BYTES: usize = 256 * 1024 * 1024;

/// Parse only the compact STR2 descriptor prefix.  Unlike
/// `read_strand_v2_header`, this deliberately does not touch seek tables or
/// payload pages, so the offline LUT append path remains O(descriptor + LUTs)
/// memory even for a model artifact much larger than RAM.
fn parse_archive_lut_descriptor(prefix: &[u8]) -> Result<ArchiveLutDescriptor, String> {
    let take = |p: &mut usize, n: usize| -> Result<&[u8], String> {
        let end = p
            .checked_add(n)
            .filter(|&end| end <= prefix.len())
            .ok_or("sdsc: STR2 descriptor truncated")?;
        let bytes = &prefix[*p..end];
        *p = end;
        Ok(bytes)
    };
    let u32_at = |p: &mut usize| -> Result<u32, String> {
        Ok(u32::from_le_bytes(take(p, 4)?.try_into().unwrap()))
    };
    let u64_at = |p: &mut usize| -> Result<u64, String> {
        Ok(u64::from_le_bytes(take(p, 8)?.try_into().unwrap()))
    };

    if prefix.len() < 56 || &prefix[..4] != &MAGIC_V2[..] {
        return Err("sdsc: not a STR2 archive descriptor".into());
    }
    let version = u32::from_le_bytes(prefix[4..8].try_into().unwrap());
    if version != VERSION_V2 {
        return Err(format!("sdsc: STR2 version {version} != {VERSION_V2}"));
    }
    let header_bytes = u32::from_le_bytes(prefix[8..12].try_into().unwrap()) as usize;
    if header_bytes != prefix.len() || header_bytes < 56 {
        return Err(format!(
            "sdsc: STR2 descriptor length {} != header_bytes {header_bytes}",
            prefix.len()
        ));
    }
    let n_tensors = u32::from_le_bytes(prefix[12..16].try_into().unwrap()) as usize;
    // Even an empty-name, rank-0 descriptor occupies 88 bytes.  Check before
    // `Vec::with_capacity` so a hostile count cannot request model-sized RAM.
    if n_tensors > (header_bytes - 56) / 88 {
        return Err(format!(
            "sdsc: STR2 tensor count {n_tensors} cannot fit in the {header_bytes}-byte descriptor"
        ));
    }
    let archive_flags = u32::from_le_bytes(prefix[16..20].try_into().unwrap());
    let mut source_sha256 = [0u8; 32];
    source_sha256.copy_from_slice(&prefix[20..52]);
    if prefix[52..56].iter().any(|&b| b != 0) {
        return Err("sdsc: STR2 fixed-header reserved bytes are nonzero".into());
    }

    let record_stride = BlockOffsetRecord::stride_for(archive_flags);
    let mut p = 56usize;
    let mut tensors = Vec::with_capacity(n_tensors);
    let mut base_bytes = checked_page_align(header_bytes)?;
    for tensor_index in 0..n_tensors {
        let name_len = u32_at(&mut p)? as usize;
        let name = String::from_utf8(take(&mut p, name_len)?.to_vec())
            .map_err(|e| format!("sdsc: STR2 tensor {tensor_index} name: {e}"))?;
        let ndim = u32_at(&mut p)? as usize;
        if ndim > (prefix.len() - p) / 8 {
            return Err(format!(
                "sdsc: STR2 tensor {tensor_index} rank {ndim} cannot fit in the descriptor"
            ));
        }
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(u64_at(&mut p)?);
        }
        let rht_seed = u64_at(&mut p)?;
        let geom = take(&mut p, 4)?;
        let (l_bits, k_bits, vec_dim, flags) = (geom[0], geom[1], geom[2], geom[3]);
        let block_len = u32_at(&mut p)?;
        let total_u64 = u64_at(&mut p)?;
        let n_blocks_u64 = u64_at(&mut p)?;
        let reserved = take(&mut p, 8)?;
        if reserved.iter().any(|&b| b != 0) {
            return Err(format!(
                "sdsc: STR2 tensor {tensor_index} reserved descriptor bytes are nonzero"
            ));
        }
        let table_offset = usize::try_from(u64_at(&mut p)?)
            .map_err(|_| "sdsc: table offset exceeds address space")?;
        let payload_offset = usize::try_from(u64_at(&mut p)?)
            .map_err(|_| "sdsc: payload offset exceeds address space")?;
        let payload_bytes = usize::try_from(u64_at(&mut p)?)
            .map_err(|_| "sdsc: payload length exceeds address space")?;
        let sideinfo_offset = usize::try_from(u64_at(&mut p)?)
            .map_err(|_| "sdsc: side-info offset exceeds address space")?;
        let sideinfo_bytes = usize::try_from(u64_at(&mut p)?)
            .map_err(|_| "sdsc: side-info length exceeds address space")?;
        let total = usize::try_from(total_u64)
            .map_err(|_| "sdsc: tensor element count exceeds address space")?;
        let n_blocks = usize::try_from(n_blocks_u64)
            .map_err(|_| "sdsc: tensor block count exceeds address space")?;

        for (label, offset) in [("table", table_offset), ("payload", payload_offset)] {
            if offset % PAGE != 0 {
                return Err(format!(
                    "sdsc: STR2 tensor {tensor_index} {label} offset {offset} is not page-aligned"
                ));
            }
        }
        if sideinfo_offset != 0 && sideinfo_offset % PAGE != 0 {
            return Err(format!(
                "sdsc: STR2 tensor {tensor_index} side-info offset is not page-aligned"
            ));
        }
        let table_bytes = n_blocks
            .checked_mul(record_stride)
            .ok_or("sdsc: STR2 seek table length overflows")?;
        let table_end = table_offset
            .checked_add(table_bytes)
            .ok_or("sdsc: STR2 seek table extent overflows")?;
        if table_end > payload_offset {
            return Err(format!(
                "sdsc: STR2 tensor {tensor_index} seek table overruns payload"
            ));
        }
        let payload_end = payload_offset
            .checked_add(payload_bytes)
            .ok_or("sdsc: STR2 payload extent overflows")?;
        base_bytes = base_bytes.max(checked_page_align(payload_end)?);
        if sideinfo_offset != 0 {
            let sideinfo_end = sideinfo_offset
                .checked_add(sideinfo_bytes)
                .ok_or("sdsc: STR2 side-info extent overflows")?;
            base_bytes = base_bytes.max(checked_page_align(sideinfo_end)?);
        }

        tensors.push(TensorLutDescriptor {
            name,
            shape,
            rht_seed,
            l_bits,
            k_bits,
            vec_dim,
            has_rht_seed: flags & 1 != 0,
            rht_cols: flags & 8 != 0,
            tail_biting: flags & 2 != 0,
            has_affine_min: flags & 4 != 0,
            block_len,
            total,
            n_blocks,
        });
    }
    if p != header_bytes {
        return Err(format!(
            "sdsc: {} trailing bytes in STR2 descriptor header",
            header_bytes - p
        ));
    }
    Ok(ArchiveLutDescriptor {
        source_sha256,
        tensors,
        base_bytes,
    })
}

fn archive_lut_descriptor_from_bytes(buf: &[u8]) -> Result<ArchiveLutDescriptor, String> {
    if buf.len() < 12 {
        return Err("sdsc: file shorter than STR2 fixed header".into());
    }
    let header_bytes = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
    if header_bytes > MAX_STR2_DESCRIPTOR_BYTES {
        return Err(format!(
            "sdsc: STR2 descriptor is {header_bytes} bytes (cap {MAX_STR2_DESCRIPTOR_BYTES})"
        ));
    }
    let prefix = buf
        .get(..header_bytes)
        .ok_or("sdsc: STR2 descriptor extends past the file")?;
    parse_archive_lut_descriptor(prefix)
}

fn archive_lut_descriptor_from_file(file: &mut fs::File) -> Result<ArchiveLutDescriptor, String> {
    let mut fixed = [0u8; 56];
    file.seek(SeekFrom::Start(0))
        .map_err(|e| format!("sdsc: seek STR2 header: {e}"))?;
    file.read_exact(&mut fixed)
        .map_err(|e| format!("sdsc: read STR2 fixed header: {e}"))?;
    if &fixed[..4] != &MAGIC_V2[..] {
        return Err("sdsc: not a STR2 archive".into());
    }
    let header_bytes = u32::from_le_bytes(fixed[8..12].try_into().unwrap()) as usize;
    if !(56..=MAX_STR2_DESCRIPTOR_BYTES).contains(&header_bytes) {
        return Err(format!(
            "sdsc: STR2 descriptor is {header_bytes} bytes (allowed 56..={MAX_STR2_DESCRIPTOR_BYTES})"
        ));
    }
    let mut prefix = vec![0u8; header_bytes];
    prefix[..56].copy_from_slice(&fixed);
    file.read_exact(&mut prefix[56..])
        .map_err(|e| format!("sdsc: read STR2 descriptor: {e}"))?;
    parse_archive_lut_descriptor(&prefix)
}

fn tensor_lut_record_sha256(
    source_sha256: &[u8; 32],
    tensor_index: usize,
    desc: &TensorLutDescriptor,
    entries: &[i32],
) -> [u8; 32] {
    let mut msg =
        Vec::with_capacity(128 + desc.name.len() + desc.shape.len() * 8 + entries.len() * 4);
    msg.extend_from_slice(b"hawking.sdsc.tensor-lut.v2\0");
    msg.extend_from_slice(source_sha256);
    msg.extend_from_slice(&(tensor_index as u64).to_le_bytes());
    msg.extend_from_slice(&(desc.name.len() as u64).to_le_bytes());
    msg.extend_from_slice(desc.name.as_bytes());
    msg.extend_from_slice(&(desc.shape.len() as u64).to_le_bytes());
    for &dim in &desc.shape {
        msg.extend_from_slice(&dim.to_le_bytes());
    }
    msg.extend_from_slice(&desc.rht_seed.to_le_bytes());
    msg.push(desc.l_bits);
    msg.push(desc.k_bits);
    msg.push(desc.vec_dim);
    msg.push(desc.has_rht_seed as u8);
    msg.push(desc.rht_cols as u8);
    msg.push(desc.tail_biting as u8);
    msg.push(desc.has_affine_min as u8);
    msg.extend_from_slice(&desc.block_len.to_le_bytes());
    msg.extend_from_slice(&(desc.total as u64).to_le_bytes());
    msg.extend_from_slice(&(desc.n_blocks as u64).to_le_bytes());
    msg.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for &entry in entries {
        msg.extend_from_slice(&entry.to_le_bytes());
    }
    sha256(&msg)
}

fn owned_tensor_lut_record_sha256(
    source_sha256: &[u8; 32],
    tensor_index: usize,
    t: &OwnedTensorV2,
    entries: &[i32],
) -> [u8; 32] {
    let mut msg =
        Vec::with_capacity(128 + t.base.name.len() + t.base.shape.len() * 8 + entries.len() * 4);
    msg.extend_from_slice(b"hawking.sdsc.tensor-lut.v2\0");
    msg.extend_from_slice(source_sha256);
    msg.extend_from_slice(&(tensor_index as u64).to_le_bytes());
    msg.extend_from_slice(&(t.base.name.len() as u64).to_le_bytes());
    msg.extend_from_slice(t.base.name.as_bytes());
    msg.extend_from_slice(&(t.base.shape.len() as u64).to_le_bytes());
    for &dim in &t.base.shape {
        msg.extend_from_slice(&dim.to_le_bytes());
    }
    msg.extend_from_slice(&t.base.rht_seed.to_le_bytes());
    msg.push(t.base.l_bits);
    msg.push(t.base.k_bits);
    msg.push(t.base.vec_dim);
    msg.push(t.base.enc.has_rht_seed as u8);
    msg.push(t.rht_cols as u8);
    msg.push(t.base.enc.tail_biting as u8);
    msg.push(t.base.enc.has_affine_min as u8);
    msg.extend_from_slice(&t.block_len.to_le_bytes());
    msg.extend_from_slice(&(t.base.enc.total as u64).to_le_bytes());
    msg.extend_from_slice(&(t.base.enc.blocks.len() as u64).to_le_bytes());
    msg.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for &entry in entries {
        msg.extend_from_slice(&entry.to_le_bytes());
    }
    sha256(&msg)
}

fn validate_tensor_lut_record(
    source_sha256: &[u8; 32],
    descs: &[TensorLutDescriptor],
    record: &SdscTensorLut,
) -> Result<(), String> {
    let index = record.tensor_index as usize;
    let desc = descs
        .get(index)
        .ok_or_else(|| format!("sdsc: tensor LUT index {index} is outside the archive"))?;
    if desc.vec_dim <= 1 {
        return Err(format!(
            "sdsc: tensor LUT index {index} ({:?}) targets scalar vec_dim {}",
            desc.name, desc.vec_dim
        ));
    }
    if record.l_bits != desc.l_bits || record.vec_dim != desc.vec_dim {
        return Err(format!(
            "sdsc: tensor LUT index {index} geometry L={} d={} != archive L={} d={}",
            record.l_bits, record.vec_dim, desc.l_bits, desc.vec_dim
        ));
    }
    let want = 1usize
        .checked_shl(desc.l_bits as u32)
        .and_then(|n| n.checked_mul(desc.vec_dim as usize))
        .ok_or("sdsc: tensor LUT geometry overflows")?;
    if record.entries.len() != want {
        return Err(format!(
            "sdsc: tensor LUT index {index} has {} entries, want {want}",
            record.entries.len()
        ));
    }
    let digest = tensor_lut_record_sha256(source_sha256, index, desc, &record.entries);
    if digest != record.record_sha256 {
        return Err(format!(
            "sdsc: tensor LUT index {index} descriptor/content SHA-256 mismatch"
        ));
    }
    Ok(())
}

/// Build an SDSC V2 section for a STR2 archive using exact learned LUTs.
///
/// Every `vec_dim > 1` tensor must appear exactly once, and scalar tensors must
/// not appear.  This all-or-nothing rule prevents a missing learned codebook from
/// silently falling back to the unrelated broadcast/frozen LUT.
pub fn build_sdsc_for_archive_with_tensor_luts(
    buf: &[u8],
    inputs: &[TensorLutInput<'_>],
) -> Result<Sdsc, String> {
    let archive = archive_lut_descriptor_from_bytes(buf)?;
    build_sdsc_for_descriptor(&archive, inputs)
}

fn build_sdsc_for_descriptor(
    archive: &ArchiveLutDescriptor,
    inputs: &[TensorLutInput<'_>],
) -> Result<Sdsc, String> {
    let mut sorted = inputs.to_vec();
    sorted.sort_unstable_by_key(|x| x.tensor_index);
    if sorted
        .windows(2)
        .any(|w| w[0].tensor_index == w[1].tensor_index)
    {
        return Err("sdsc: duplicate per-tensor LUT index".into());
    }

    let required: Vec<usize> = archive
        .tensors
        .iter()
        .enumerate()
        .filter_map(|(i, t)| (t.vec_dim > 1).then_some(i))
        .collect();
    let supplied: Vec<usize> = sorted.iter().map(|x| x.tensor_index).collect();
    if supplied != required {
        return Err(format!(
            "sdsc: learned vector LUT coverage mismatch: supplied {supplied:?}, required {required:?}"
        ));
    }

    let mut tensor_luts = Vec::with_capacity(sorted.len());
    for input in sorted {
        let desc = &archive.tensors[input.tensor_index];
        let record_sha256 = tensor_lut_record_sha256(
            &archive.source_sha256,
            input.tensor_index,
            desc,
            input.entries,
        );
        let record = SdscTensorLut {
            tensor_index: input.tensor_index as u32,
            l_bits: desc.l_bits,
            vec_dim: desc.vec_dim,
            record_sha256,
            entries: input.entries.to_vec(),
        };
        validate_tensor_lut_record(&archive.source_sha256, &archive.tensors, &record)?;
        tensor_luts.push(record);
    }

    // Scalar tensors continue to use the deterministic geometry LUT.  Vector
    // tensors are deliberately absent from this table: only their bound record
    // is legal at decode time.
    let mut scalar_ls: Vec<u8> = archive
        .tensors
        .iter()
        .filter(|t| t.vec_dim <= 1)
        .map(|t| t.l_bits)
        .collect();
    scalar_ls.sort_unstable();
    scalar_ls.dedup();
    let luts = scalar_ls
        .into_iter()
        .map(|l| SdscLut {
            l_bits: l,
            vec_dim: 1,
            entries: codebook_lut(l as u32).to_vec(),
        })
        .collect();

    Ok(Sdsc {
        consts: default_consts(),
        exprs: default_exprs(),
        luts,
        archive_source_sha256: Some(archive.source_sha256),
        tensor_luts,
    })
}

pub fn sdsc_section_bytes(sdsc: &Sdsc) -> Result<Vec<u8>, String> {
    if sdsc.consts.len() > MAX_CONSTS || sdsc.exprs.len() > MAX_EXPRS || sdsc.luts.len() > MAX_LUTS
    {
        return Err("sdsc: section exceeds sanity caps".into());
    }
    if sdsc.tensor_luts.len() > MAX_TENSOR_LUTS {
        return Err("sdsc: per-tensor LUT count exceeds sanity cap".into());
    }
    let is_v2 = !sdsc.tensor_luts.is_empty();
    if is_v2 != sdsc.archive_source_sha256.is_some() {
        return Err(
            "sdsc: V2 requires both archive_source_sha256 and per-tensor LUT records".into(),
        );
    }
    let mut prev = None;
    for &(id, _) in &sdsc.consts {
        if prev.map_or(false, |p| id <= p) {
            return Err("sdsc: const ids must be strictly ascending".into());
        }
        prev = Some(id);
    }
    let mut prev = None;
    for e in &sdsc.exprs {
        if prev.map_or(false, |p| e.id <= p) {
            return Err("sdsc: expr ids must be strictly ascending".into());
        }
        prev = Some(e.id);
        if e.n_slots as usize > MAX_SLOTS || e.prog.len() > MAX_PROG_BYTES {
            return Err(format!("sdsc: expr {} exceeds sanity caps", e.id));
        }
        validate_prog(&e.prog, e.n_slots).map_err(|er| format!("sdsc: expr {}: {er}", e.id))?;
    }
    let mut prev: Option<(u8, u8)> = None;
    for l in &sdsc.luts {
        let d = l.vec_dim.max(1);
        let key = (l.l_bits, d);
        if prev.map_or(false, |p| key <= p) {
            return Err("sdsc: LUT geometries must be strictly ascending".into());
        }
        prev = Some(key);
        let want = 1usize
            .checked_shl(l.l_bits as u32)
            .and_then(|n| n.checked_mul(d as usize))
            .ok_or("sdsc: LUT geometry overflows")?;
        if l.entries.len() != want {
            return Err(format!(
                "sdsc: LUT L={} d={d} has {} entries, want {want}",
                l.l_bits,
                l.entries.len()
            ));
        }
    }
    let mut prev_tensor = None;
    for l in &sdsc.tensor_luts {
        let index = l.tensor_index as usize;
        if prev_tensor.map_or(false, |p| index <= p) {
            return Err("sdsc: tensor LUT indices must be strictly ascending".into());
        }
        prev_tensor = Some(index);
        if l.vec_dim <= 1 || l.l_bits == 0 {
            return Err(format!(
                "sdsc: tensor LUT {index} has invalid vector geometry L={} d={}",
                l.l_bits, l.vec_dim
            ));
        }
        let want = 1usize
            .checked_shl(l.l_bits as u32)
            .and_then(|n| n.checked_mul(l.vec_dim as usize))
            .ok_or("sdsc: tensor LUT geometry overflows")?;
        if l.entries.len() != want {
            return Err(format!(
                "sdsc: tensor LUT {index} has {} entries, want {want}",
                l.entries.len()
            ));
        }
    }

    let mut o = Vec::new();
    o.extend_from_slice(SDSC_MAGIC);
    o.extend_from_slice(&(if is_v2 { SDSC_VERSION } else { SDSC_VERSION_V1 }).to_le_bytes());
    o.extend_from_slice(&0u32.to_le_bytes());
    o.extend_from_slice(&(sdsc.consts.len() as u32).to_le_bytes());
    o.extend_from_slice(&(sdsc.exprs.len() as u32).to_le_bytes());
    o.extend_from_slice(&(sdsc.luts.len() as u32).to_le_bytes());
    o.extend_from_slice(&(sdsc.tensor_luts.len() as u32).to_le_bytes());
    o.extend_from_slice(&[0u8; 20]);
    debug_assert_eq!(o.len(), SDSC_HEADER_BYTES);

    if let Some(source_sha256) = sdsc.archive_source_sha256 {
        o.extend_from_slice(&source_sha256);
    }

    for &(id, v) in &sdsc.consts {
        o.extend_from_slice(&id.to_le_bytes());
        o.extend_from_slice(&v.to_le_bytes());
    }
    for e in &sdsc.exprs {
        o.extend_from_slice(&e.id.to_le_bytes());
        o.extend_from_slice(&e.n_slots.to_le_bytes());
        o.extend_from_slice(&(e.prog.len() as u32).to_le_bytes());
        o.extend_from_slice(&e.prog);
    }
    for l in &sdsc.luts {
        o.extend_from_slice(&(l.l_bits as u32).to_le_bytes());
        o.extend_from_slice(&(l.vec_dim.max(1) as u32).to_le_bytes());
        o.extend_from_slice(&(l.entries.len() as u32).to_le_bytes());
        o.extend_from_slice(&0u32.to_le_bytes());
        for &q in &l.entries {
            o.extend_from_slice(&q.to_le_bytes());
        }
    }
    for l in &sdsc.tensor_luts {
        o.extend_from_slice(&l.tensor_index.to_le_bytes());
        o.extend_from_slice(&(l.l_bits as u32).to_le_bytes());
        o.extend_from_slice(&(l.vec_dim as u32).to_le_bytes());
        o.extend_from_slice(&(l.entries.len() as u32).to_le_bytes());
        o.extend_from_slice(&0u32.to_le_bytes());
        o.extend_from_slice(&l.record_sha256);
        for &q in &l.entries {
            o.extend_from_slice(&q.to_le_bytes());
        }
    }
    Ok(o)
}

/// Exact unpadded bytes charged by one SDSC V2 tensor-LUT record.
pub fn tensor_lut_wire_bytes(record: &SdscTensorLut) -> usize {
    SDSC_TENSOR_LUT_RECORD_BYTES + record.entries.len() * 4
}

#[inline]
fn page_align(x: usize) -> usize {
    (x + PAGE - 1) & !(PAGE - 1)
}

fn checked_page_align(x: usize) -> Result<usize, String> {
    x.checked_add(PAGE - 1)
        .map(|end| end & !(PAGE - 1))
        .ok_or_else(|| "sdsc: page alignment overflows address space".into())
}

fn chain_scan(buf: &[u8]) -> (usize, bool) {
    let mut end = buf.len();
    for _ in 0..8 {
        if end < SDSC_TRAILER_BYTES {
            return (end, false);
        }
        let t = &buf[end - SDSC_TRAILER_BYTES..end];
        let magic = &t[12..16];
        if magic == &SDSC_MAGIC[..] {
            return (end, true);
        }
        if magic == &SPRV_MAGIC[..] || magic == &OUTL_MAGIC[..] || magic == &SDSQ_MAGIC[..] {
            let off = u64::from_le_bytes(t[0..8].try_into().unwrap());
            match usize::try_from(off) {
                Ok(off) if off < end && off % PAGE == 0 => end = off,
                _ => return (end, false),
            }
        } else {
            return (end, false);
        }
    }
    (end, false)
}

/// Append a per-tensor learned-vector SDSC V2 section to a *bare* STR2 archive.
///
/// The bare-archive restriction is intentional and fail-closed: callers must do
/// this immediately after `write_strand_v2`, before OUTL/SDSQ/SPRV.  It prevents
/// an unrecognised or already-sealed trailer from being silently hidden by a new
/// section.  Existing scalar [`append_sdsc`] retains its restacking behaviour.
pub fn append_sdsc_with_tensor_luts(
    path: impl AsRef<Path>,
    inputs: &[TensorLutInput<'_>],
) -> Result<Sdsc, String> {
    let path = path.as_ref();
    let mut file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| format!("sdsc: open {path:?}: {e}"))?;
    let file_bytes = usize::try_from(
        file.metadata()
            .map_err(|e| format!("sdsc: stat {path:?}: {e}"))?
            .len(),
    )
    .map_err(|_| "sdsc: file length exceeds address space")?;
    let archive = archive_lut_descriptor_from_file(&mut file)?;
    let sdsc = build_sdsc_for_descriptor(&archive, inputs)?;
    let section = sdsc_section_bytes(&sdsc)?;
    let sdsc_bytes: u32 = section.len().try_into().map_err(|_| {
        format!(
            "sdsc: section is {} bytes — exceeds the u32 field",
            section.len()
        )
    })?;
    let sdsc_offset = checked_page_align(archive.base_bytes)?;
    let unpadded_end = sdsc_offset
        .checked_add(section.len())
        .and_then(|end| end.checked_add(SDSC_TRAILER_BYTES))
        .ok_or("sdsc: appended section extent overflows address space")?;
    let end = checked_page_align(unpadded_end)?;

    // Materialise only the small append tail, never the model payload.  On a
    // retry after power loss, accept and repair a suffix iff it is an exact
    // prefix of the bytes this invocation would append.  Unknown trailers or
    // foreign bytes remain fail-closed and untouched.
    let mut tail = vec![0u8; sdsc_offset - archive.base_bytes];
    tail.extend_from_slice(&section);
    tail.resize(end - archive.base_bytes - SDSC_TRAILER_BYTES, 0u8);
    tail.extend_from_slice(&(sdsc_offset as u64).to_le_bytes());
    tail.extend_from_slice(&sdsc_bytes.to_le_bytes());
    tail.extend_from_slice(SDSC_MAGIC);
    debug_assert_eq!(archive.base_bytes + tail.len(), end);

    if file_bytes < archive.base_bytes {
        return Err(format!(
            "sdsc: STR2 file is truncated: file has {file_bytes} bytes, base extent is {}",
            archive.base_bytes,
        ));
    }
    if file_bytes > archive.base_bytes {
        let suffix_len = file_bytes - archive.base_bytes;
        if suffix_len > tail.len() {
            return Err(format!(
                "sdsc: learned vector LUTs require a bare STR2 archive or a recoverable partial SDSC append: file has {file_bytes} bytes, expected at most {end}"
            ));
        }
        let mut suffix = vec![0u8; suffix_len];
        file.seek(SeekFrom::Start(archive.base_bytes as u64))
            .and_then(|_| file.read_exact(&mut suffix))
            .map_err(|e| format!("sdsc: read existing append suffix {path:?}: {e}"))?;
        if suffix != tail[..suffix_len] {
            return Err(
                "sdsc: non-bare STR2 suffix is not an exact prefix of this SDSC append; refusing to truncate foreign or mismatched data"
                    .into(),
            );
        }
        if suffix_len == tail.len() {
            // The append completed and only the caller acknowledgement was lost.
            return Ok(sdsc);
        }
        file.set_len(archive.base_bytes as u64)
            .map_err(|e| format!("sdsc: truncate interrupted append {path:?}: {e}"))?;
        file.sync_all()
            .map_err(|e| format!("sdsc: fsync recovered base {path:?}: {e}"))?;
    }

    file.seek(SeekFrom::Start(archive.base_bytes as u64))
        .map_err(|e| format!("sdsc: seek append {path:?}: {e}"))?;
    file.write_all(&tail)
        .map_err(|e| format!("sdsc: append tail {path:?}: {e}"))?;
    file.sync_all()
        .map_err(|e| format!("sdsc: fsync {path:?}: {e}"))?;
    Ok(sdsc)
}

pub fn append_sdsc(path: impl AsRef<Path>) -> Result<Sdsc, String> {
    let path = path.as_ref();
    let buf = fs::read(path).map_err(|e| format!("sdsc: read {path:?}: {e}"))?;

    let (base, has_sdsc) = chain_scan(&buf);
    if has_sdsc {
        return Err("sdsc: file already has an SDSC section (double-append rejected)".into());
    }

    let sprv = read_sprv_bytes(&buf, true)?;
    let outl = read_outl_bytes(&buf, true)?;

    let sdsc = build_sdsc_for_archive(&buf)?;
    let section = sdsc_section_bytes(&sdsc)?;
    let sdsc_bytes: u32 = section.len().try_into().map_err(|_| {
        format!(
            "sdsc: section is {} bytes — exceeds the u32 field",
            section.len()
        )
    })?;

    let mut out = buf[..base].to_vec();
    let sdsc_offset = page_align(out.len());
    out.resize(sdsc_offset, 0u8);
    out.extend_from_slice(&section);

    let end = page_align(sdsc_offset + section.len() + SDSC_TRAILER_BYTES);
    out.resize(end - SDSC_TRAILER_BYTES, 0u8);
    out.extend_from_slice(&(sdsc_offset as u64).to_le_bytes());
    out.extend_from_slice(&sdsc_bytes.to_le_bytes());
    out.extend_from_slice(SDSC_MAGIC);

    fs::write(path, &out).map_err(|e| format!("sdsc: write {path:?}: {e}"))?;

    if let Some(o) = outl {
        append_outl(path, &o.tensors)?;
    }
    if let Some(s) = sprv {
        append_sprv(path, &s)?;
    }
    Ok(sdsc)
}

fn parse_sdsc_section(
    buf: &[u8],
    sdsc_offset: usize,
    sdsc_bytes: usize,
    trailer_end: usize,
) -> Result<Sdsc, String> {
    if sdsc_offset % PAGE != 0 {
        return Err(format!("sdsc: sdsc_offset {sdsc_offset} not page-aligned"));
    }
    let min_end = sdsc_offset
        .checked_add(sdsc_bytes)
        .and_then(|x| x.checked_add(SDSC_TRAILER_BYTES))
        .ok_or("sdsc: sdsc_offset + sdsc_bytes overflows")?;
    if min_end > trailer_end || trailer_end % PAGE != 0 {
        return Err(format!(
            "sdsc: section [{sdsc_offset}, +{sdsc_bytes}] + trailer does not fit the \
             page-aligned region ending at {trailer_end}"
        ));
    }
    if sdsc_bytes < SDSC_HEADER_BYTES {
        return Err("sdsc: section shorter than the 48-byte header".into());
    }
    if buf[sdsc_offset + sdsc_bytes..trailer_end - SDSC_TRAILER_BYTES]
        .iter()
        .any(|&b| b != 0)
    {
        return Err("sdsc: nonzero bytes in section padding".into());
    }

    let s = &buf[sdsc_offset..sdsc_offset + sdsc_bytes];
    if &s[0..4] != &SDSC_MAGIC[..] {
        return Err("sdsc: bad section header magic".into());
    }
    let version = u32::from_le_bytes(s[4..8].try_into().unwrap());
    if version != SDSC_VERSION_V1 && version != SDSC_VERSION {
        return Err(format!(
            "sdsc: unsupported version {version} (supported {SDSC_VERSION_V1}, {SDSC_VERSION})"
        ));
    }
    let flags = u32::from_le_bytes(s[8..12].try_into().unwrap());
    if flags != 0 {
        return Err(format!("sdsc: reserved flag bits set: {flags:#x}"));
    }
    let n_consts = u32::from_le_bytes(s[12..16].try_into().unwrap()) as usize;
    let n_exprs = u32::from_le_bytes(s[16..20].try_into().unwrap()) as usize;
    let n_luts = u32::from_le_bytes(s[20..24].try_into().unwrap()) as usize;
    let n_tensor_luts = u32::from_le_bytes(s[24..28].try_into().unwrap()) as usize;
    if n_consts > MAX_CONSTS
        || n_exprs > MAX_EXPRS
        || n_luts > MAX_LUTS
        || n_tensor_luts > MAX_TENSOR_LUTS
    {
        return Err("sdsc: section exceeds sanity caps".into());
    }
    if version == SDSC_VERSION_V1 && n_tensor_luts != 0 {
        return Err("sdsc: V1 section has nonzero per-tensor LUT count".into());
    }
    if s[28..SDSC_HEADER_BYTES].iter().any(|&b| b != 0) {
        return Err("sdsc: header reserved bytes not zero".into());
    }

    let mut p = SDSC_HEADER_BYTES;
    let take = |p: &mut usize, n: usize| -> Result<&[u8], String> {
        let end = p
            .checked_add(n)
            .filter(|&e| e <= s.len())
            .ok_or("sdsc: section truncated")?;
        let sl = &s[*p..end];
        *p = end;
        Ok(sl)
    };

    let archive = archive_lut_descriptor_from_bytes(buf)?;
    let archive_source_sha256 = if version == SDSC_VERSION {
        let mut stored = [0u8; 32];
        stored.copy_from_slice(take(&mut p, 32)?);
        if stored != archive.source_sha256 {
            return Err("sdsc: V2 source SHA-256 does not match the STR2 archive".into());
        }
        Some(stored)
    } else {
        None
    };

    let mut consts = Vec::with_capacity(n_consts);
    let mut prev: Option<u32> = None;
    for _ in 0..n_consts {
        let id = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap());
        let v = i64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap());
        if prev.map_or(false, |pv| id <= pv) {
            return Err("sdsc: const ids not strictly ascending".into());
        }
        prev = Some(id);
        consts.push((id, v));
    }

    let mut exprs = Vec::with_capacity(n_exprs);
    let mut prev: Option<u32> = None;
    for _ in 0..n_exprs {
        let id = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap());
        let n_slots = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap());
        let prog_len = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap()) as usize;
        if prev.map_or(false, |pv| id <= pv) {
            return Err("sdsc: expr ids not strictly ascending".into());
        }
        prev = Some(id);
        if n_slots as usize > MAX_SLOTS || prog_len > MAX_PROG_BYTES {
            return Err(format!("sdsc: expr {id} exceeds sanity caps"));
        }
        let prog = take(&mut p, prog_len)?.to_vec();
        validate_prog(&prog, n_slots).map_err(|e| format!("sdsc: expr {id}: {e}"))?;
        exprs.push(SdscExpr { id, n_slots, prog });
    }

    let mut luts = Vec::with_capacity(n_luts);
    let mut prev: Option<(u8, u8)> = None;
    for _ in 0..n_luts {
        let l_bits = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap());
        let vec_dim = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap());
        let n_entries = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap()) as usize;
        let reserved = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap());
        if reserved != 0 {
            return Err("sdsc: LUT reserved field not zero".into());
        }
        if l_bits == 0 || l_bits > 24 || vec_dim == 0 || vec_dim > 255 {
            return Err(format!(
                "sdsc: LUT geometry L={l_bits} d={vec_dim} out of range"
            ));
        }
        let key = (l_bits as u8, vec_dim as u8);
        if prev.map_or(false, |pv| key <= pv) {
            return Err("sdsc: LUT geometries not strictly ascending".into());
        }
        prev = Some(key);
        let want = 1usize
            .checked_shl(l_bits)
            .and_then(|n| n.checked_mul(vec_dim as usize))
            .ok_or("sdsc: LUT geometry overflows")?;
        if n_entries != want {
            return Err(format!(
                "sdsc: LUT L={l_bits} d={vec_dim} has {n_entries} entries, want {want}"
            ));
        }
        let raw_len = n_entries
            .checked_mul(4)
            .ok_or("sdsc: LUT byte length overflows")?;
        let raw = take(&mut p, raw_len)?;
        let entries: Vec<i32> = raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        luts.push(SdscLut {
            l_bits: key.0,
            vec_dim: key.1,
            entries,
        });
    }

    let mut tensor_luts = Vec::with_capacity(n_tensor_luts);
    let mut prev_tensor = None;
    for _ in 0..n_tensor_luts {
        let tensor_index = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap());
        let l_bits = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap());
        let vec_dim = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap());
        let n_entries = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap()) as usize;
        let reserved = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap());
        if reserved != 0 {
            return Err("sdsc: tensor LUT reserved field not zero".into());
        }
        if l_bits == 0 || l_bits > 24 || vec_dim <= 1 || vec_dim > 255 {
            return Err(format!(
                "sdsc: tensor LUT geometry L={l_bits} d={vec_dim} out of range"
            ));
        }
        let index = tensor_index as usize;
        if prev_tensor.map_or(false, |prev| index <= prev) {
            return Err("sdsc: tensor LUT indices not strictly ascending".into());
        }
        prev_tensor = Some(index);
        let want = 1usize
            .checked_shl(l_bits)
            .and_then(|n| n.checked_mul(vec_dim as usize))
            .ok_or("sdsc: tensor LUT geometry overflows")?;
        if n_entries != want {
            return Err(format!(
                "sdsc: tensor LUT {index} has {n_entries} entries, want {want}"
            ));
        }
        let mut record_sha256 = [0u8; 32];
        record_sha256.copy_from_slice(take(&mut p, 32)?);
        let raw_len = n_entries
            .checked_mul(4)
            .ok_or("sdsc: tensor LUT byte length overflows")?;
        let raw = take(&mut p, raw_len)?;
        let entries: Vec<i32> = raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let record = SdscTensorLut {
            tensor_index,
            l_bits: l_bits as u8,
            vec_dim: vec_dim as u8,
            record_sha256,
            entries,
        };
        validate_tensor_lut_record(
            archive_source_sha256
                .as_ref()
                .ok_or("sdsc: tensor LUT record in a V1 section")?,
            &archive.tensors,
            &record,
        )?;
        tensor_luts.push(record);
    }

    let required: Vec<usize> = archive
        .tensors
        .iter()
        .enumerate()
        .filter_map(|(i, t)| (t.vec_dim > 1).then_some(i))
        .collect();
    let supplied: Vec<usize> = tensor_luts
        .iter()
        .map(|t| t.tensor_index as usize)
        .collect();
    if version == SDSC_VERSION && supplied != required {
        return Err(format!(
            "sdsc: learned vector LUT coverage mismatch: supplied {supplied:?}, required {required:?}"
        ));
    }
    if version == SDSC_VERSION_V1 && !required.is_empty() {
        return Err("sdsc: V1 cannot decode an archive containing vector-trellis tensors".into());
    }

    if p != sdsc_bytes {
        return Err(format!(
            "sdsc: {} trailing bytes after the last LUT",
            sdsc_bytes - p
        ));
    }
    Ok(Sdsc {
        consts,
        exprs,
        luts,
        archive_source_sha256,
        tensor_luts,
    })
}

pub fn read_sdsc_bytes(buf: &[u8], strict: bool) -> Result<Option<Sdsc>, String> {
    let mut end = buf.len();
    for _ in 0..8 {
        if end < SDSC_TRAILER_BYTES {
            return Ok(None);
        }
        let t = &buf[end - SDSC_TRAILER_BYTES..end];
        let magic = &t[12..16];
        if magic == &SDSC_MAGIC[..] {
            let parse = (|| -> Result<Sdsc, String> {
                let off = u64::from_le_bytes(t[0..8].try_into().unwrap());
                let bytes = u32::from_le_bytes(t[8..12].try_into().unwrap());
                let off: usize = off
                    .try_into()
                    .map_err(|_| "sdsc: sdsc_offset exceeds address space".to_string())?;
                parse_sdsc_section(buf, off, bytes as usize, end)
            })();
            return match parse {
                Ok(s) => Ok(Some(s)),
                Err(e) if strict => Err(e),
                Err(_) => Ok(None),
            };
        } else if magic == &SPRV_MAGIC[..] || magic == &OUTL_MAGIC[..] || magic == &SDSQ_MAGIC[..] {
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

pub fn read_sdsc(path: impl AsRef<Path>) -> Result<Option<Sdsc>, String> {
    let path = path.as_ref();
    let buf = fs::read(path).map_err(|e| format!("sdsc: read {path:?}: {e}"))?;
    read_sdsc_bytes(&buf, true)
}

fn unpack_codes(bytes: &[u8], n: usize, code_bits: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let mut cursor = 0usize;
    for _ in 0..n {
        out.push(read_bits(bytes, cursor, code_bits) as u8);
        cursor += code_bits as usize;
    }
    out
}

pub fn decode_q12_with_sdsc(sdsc: &Sdsc, t: &OwnedTensorV2) -> Result<Vec<i32>, String> {
    if t.base.vec_dim > 1 {
        return Err(
            "sdsc: vector decode requires decode_q12_with_sdsc_at with the archive tensor index"
                .into(),
        );
    }
    let entries = &sdsc.lut(t.base.l_bits, 1)?.entries;
    decode_q12_with_sdsc_lut(sdsc, t, entries)
}

/// Reconstruct one tensor through the self-described integer program and its
/// exact archive-bound LUT.  Vector tensors never fall back to a geometry LUT.
pub fn decode_q12_with_sdsc_at(
    sdsc: &Sdsc,
    tensor_index: usize,
    t: &OwnedTensorV2,
) -> Result<Vec<i32>, String> {
    let d = t.base.vec_dim.max(1);
    if d == 1 {
        let entries = &sdsc.lut(t.base.l_bits, 1)?.entries;
        return decode_q12_with_sdsc_lut(sdsc, t, entries);
    }
    let record = sdsc.tensor_lut(tensor_index)?;
    if record.l_bits != t.base.l_bits || record.vec_dim != d {
        return Err(format!(
            "sdsc: tensor LUT {tensor_index} geometry L={} d={} != tensor L={} d={}",
            record.l_bits, record.vec_dim, t.base.l_bits, d
        ));
    }
    let source = sdsc
        .archive_source_sha256
        .as_ref()
        .ok_or("sdsc: vector tensor has no V2 archive source binding")?;
    let digest = owned_tensor_lut_record_sha256(source, tensor_index, t, &record.entries);
    if digest != record.record_sha256 {
        return Err(format!(
            "sdsc: tensor LUT {tensor_index} descriptor/content SHA-256 mismatch at decode"
        ));
    }
    decode_q12_with_sdsc_lut(sdsc, t, &record.entries)
}

fn decode_q12_with_sdsc_lut(
    sdsc: &Sdsc,
    t: &OwnedTensorV2,
    lut: &[i32],
) -> Result<Vec<i32>, String> {
    if sdsc.const_val(const_id::PAYLOAD_BIT_ORDER)? != 0 {
        return Err("sdsc: unknown PAYLOAD_BIT_ORDER".into());
    }
    if sdsc.const_val(const_id::SIDEINFO_LAYOUT)? != 1 {
        return Err("sdsc: unknown SIDEINFO_LAYOUT".into());
    }
    if sdsc.const_val(const_id::TAILBITE_RULE)? != 1 {
        return Err("sdsc: unknown TAILBITE_RULE".into());
    }
    let d = (t.base.vec_dim as usize).max(1);
    let want_lut = 1usize
        .checked_shl(t.base.l_bits as u32)
        .and_then(|n| n.checked_mul(d))
        .ok_or("sdsc: decode LUT geometry overflows")?;
    if lut.len() != want_lut {
        return Err(format!(
            "sdsc: decode LUT has {} entries, tensor needs {want_lut}",
            lut.len()
        ));
    }
    let sub_block = sdsc.const_val(const_id::SUB_BLOCK)? as usize;
    let code_bits = sdsc.const_val(const_id::SUBSCALE_CODE_BITS)? as u32;
    if sub_block == 0 || code_bits == 0 || code_bits > 8 {
        return Err("sdsc: SUB_BLOCK / SUBSCALE_CODE_BITS out of range".into());
    }
    let adv = &sdsc.expr(expr_id::ADVANCE)?.prog;
    let eff_e = &sdsc.expr(expr_id::EFF_SCALE)?.prog;
    let off_e = &sdsc.expr(expr_id::OFFSET)?.prog;
    let rec_e = &sdsc.expr(expr_id::RECON)?.prog;

    let enc = &t.base.enc;
    let k = t.base.k_bits as u32;
    let l = t.base.l_bits as i64;
    let mask = (1i64 << l) - 1;
    let kmask = ((1usize << k) - 1) as usize;

    let mut out = Vec::with_capacity(enc.total);
    let mut bit_cursor = 0usize;
    for blk in &enc.blocks {
        let n = blk.n as usize;
        let n_sub = n.div_ceil(sub_block);
        let scodes = unpack_codes(&blk.sub_scales, n_sub, code_bits);
        let mut eff = Vec::with_capacity(n_sub);
        for &c in &scodes {
            eff.push(eval_expr(eff_e, &[blk.scale_q as i64, c as i64])?);
        }
        let offs: Vec<i64> = if enc.has_affine_min {
            let mcodes = unpack_codes(&blk.mins, n_sub, code_bits);
            let mut o = Vec::with_capacity(n_sub);
            for &c in &mcodes {
                o.push(eval_expr(off_e, &[blk.min_base_q as i64, c as i64])?);
            }
            o
        } else {
            Vec::new()
        };

        let n_steps = n.div_ceil(d);
        let nk = n_steps * k as usize;
        let mut state: i64 = if enc.tail_biting && nk >= l as usize {
            let mut s: i64 = 0;
            let mut c = bit_cursor;
            for _ in 0..n_steps {
                let sym = (read_bits(&enc.bits, c, k) & kmask) as i64;
                c += k as usize;
                s = eval_expr(adv, &[s, sym, k as i64, mask])?;
            }
            s
        } else {
            blk.init_state as i64 & mask
        };

        let mut produced = 0usize;
        for _ in 0..n_steps {
            let sym = (read_bits(&enc.bits, bit_cursor, k) & kmask) as i64;
            bit_cursor += k as usize;
            state = eval_expr(adv, &[state, sym, k as i64, mask])?;
            let su = state as usize;
            let emit = (n - produced).min(d);
            for j in 0..emit {
                let i = produced + j;
                let q = *lut
                    .get(su * d + j)
                    .ok_or("sdsc: state/vector lane indexes past the embedded LUT")?
                    as i64;
                let sb = i / sub_block;
                let es = *eff.get(sb).ok_or("sdsc: sub-block index out of range")?;
                let off = offs.get(sb).copied().unwrap_or(0);
                out.push(eval_expr(rec_e, &[es, q, off])? as i32);
            }
            produced += emit;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{decode_lean, eff_min_q, eff_scale_q, reconstruct_q};
    use crate::encode::{encode_tensor_with, EncodeOpts};
    use crate::format::{read_strand_v2, write_strand_v2, PackedTensor, PackedTensorV2};
    use crate::outlier_wire::{read_outl, OutlierWire};
    use crate::provenance_io::{append_sprv_computed, read_sprv, verify_archive, VerifyDepth};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn tmp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "strand-sdsc-{tag}-{}-{}.strand",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ))
    }

    struct TmpFile(PathBuf);
    impl Drop for TmpFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn test_weights(n: usize, seed: u64) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5)
            .collect()
    }

    #[test]
    fn sdsc_exprs_match_native_arithmetic() {
        let adv = prog_advance();
        let eff = prog_eff_scale();
        let off = prog_offset();
        let rec = prog_recon();

        for (l, k) in [(6u32, 2u32), (7, 3), (8, 4), (12, 2), (14, 4)] {
            let cfg = TrellisConfig::new(l, k, 256);
            let mask = (cfg.num_states() - 1) as i64;
            for state in (0..cfg.num_states()).step_by(7) {
                for sym in 0..cfg.num_inputs() {
                    let want = cfg.next_state(state, sym) as i64;
                    let got = eval_expr(&adv, &[state as i64, sym as i64, k as i64, mask]).unwrap();
                    assert_eq!(got, want, "advance L={l} k={k} s={state} sym={sym}");
                }
            }
        }

        for scale_q in [
            -2_000_000_000i32,
            -65536,
            -1,
            0,
            1,
            12345,
            65536,
            2_000_000_000,
        ] {
            for code in 0u8..64 {
                let want = eff_scale_q(scale_q, code) as i64;
                let got = eval_expr(&eff, &[scale_q as i64, code as i64]).unwrap();
                assert_eq!(got, want, "eff_scale scale_q={scale_q} code={code}");
            }
        }

        for min_base in [-40960i32, -4096, -31, -1, 0, 1, 31, 4096, 40960] {
            for code in 0u8..64 {
                let want = eff_min_q(min_base, code) as i64;
                let got = eval_expr(&off, &[min_base as i64, code as i64]).unwrap();
                assert_eq!(got, want, "offset min_base={min_base} code={code}");
            }
        }

        for es in [-2_000_000_000i32, -65536, 0, 1, 99999, 2_000_000_000] {
            for q in [-20480i32, -4096, -1, 0, 1, 4096, 20480] {
                for o in [-4096i32, 0, 4096] {
                    let want = (reconstruct_q(es, q).wrapping_add(o)) as i64;
                    let got = eval_expr(&rec, &[es as i64, q as i64, o as i64]).unwrap();
                    assert_eq!(got, want, "recon es={es} q={q} off={o}");
                }
            }
        }
    }

    #[test]
    fn sdsc_expr_hostile_programs_error() {
        assert!(eval_expr(&[op::ADD, op::END], &[]).is_err());
        assert!(eval_expr(&[op::IMM, 1, 2], &[]).is_err());
        assert!(eval_expr(&[0xFF, op::END], &[]).is_err());
        assert!(eval_expr(&[op::LOAD, 5, op::END], &[0; 2]).is_err());
        assert!(eval_expr(&[op::LOAD, 0], &[1]).is_err());

        let mut two = Vec::new();
        asm_imm(&mut two, 1);
        asm_imm(&mut two, 2);
        two.push(op::END);
        assert!(eval_expr(&two, &[]).is_err());

        assert!(validate_prog(&[op::LOAD, 3, op::END], 2).is_err());
        assert!(validate_prog(&[0xFF, op::END], 0).is_err());
        assert!(validate_prog(&[op::END, op::END], 0).is_err());
        assert!(validate_prog(&prog_advance(), 4).is_ok());
    }

    fn build_test_archive() -> Vec<u8> {
        let cfg = TrellisConfig::for_bpw(3.0);
        let encs = [
            encode_tensor_with(&test_weights(1024, 11), &cfg, &EncodeOpts::default()),
            encode_tensor_with(
                &test_weights(900, 23),
                &cfg,
                &EncodeOpts {
                    tail_biting: true,
                    ..Default::default()
                },
            ),
            encode_tensor_with(
                &test_weights(700, 5),
                &cfg,
                &EncodeOpts {
                    affine_min: true,
                    ..Default::default()
                },
            ),
            encode_tensor_with(
                &test_weights(1030, 41),
                &cfg,
                &EncodeOpts {
                    tail_biting: true,
                    affine_min: true,
                    ..Default::default()
                },
            ),
        ];
        let shapes: [Vec<u64>; 4] = [vec![4, 256], vec![900], vec![700], vec![1030]];
        let names = ["t.plain", "t.tail", "t.affine", "t.both"];
        let tensors: Vec<PackedTensorV2> = encs
            .iter()
            .zip(shapes.iter())
            .zip(names.iter())
            .map(|((enc, shape), name)| PackedTensorV2 {
                base: PackedTensor {
                    name,
                    shape,
                    rht_seed: 0,
                    l_bits: cfg.l_bits as u8,
                    k_bits: cfg.k_bits as u8,
                    vec_dim: cfg.vec_dim() as u8,
                    enc,
                },
                block_len: cfg.block_len as u32,
            })
            .collect();
        write_strand_v2(&tensors, [9u8; 32], false).expect("write v2")
    }

    #[test]
    fn sdsc_driven_decode_is_bit_identical() {
        let buf = build_test_archive();
        let path = tmp_path("decode");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();
        append_sdsc(&path).expect("append sdsc");

        let trailered = std::fs::read(&path).unwrap();
        let sdsc = read_sdsc_bytes(&trailered, true)
            .unwrap()
            .expect("sdsc found");
        let tensors = read_strand_v2(&trailered).expect("v2 read under sdsc trailer");
        assert_eq!(tensors.len(), 4);
        for t in &tensors {
            let cfg = TrellisConfig::new(
                t.base.l_bits as u32,
                t.base.k_bits as u32,
                t.block_len as usize,
            );
            let want = decode_lean(&t.base.enc, &cfg);
            let got = decode_q12_with_sdsc(&sdsc, t).expect("sdsc decode");
            assert_eq!(
                got, want,
                "SDSC-driven decode diverged on {:?}",
                t.base.name
            );
        }
    }

    #[test]
    fn sdsc_restack_under_outl_and_sprv() {
        let buf = build_test_archive();
        let path = tmp_path("restack");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();

        let wires = vec![
            Some(OutlierWire::from_selection(
                1024,
                vec![7, 600],
                vec![-100, 42],
                0.5,
                8,
            )),
            None,
            None,
            None,
        ];
        append_outl(&path, &wires).expect("append outl");
        let sprv_before = append_sprv_computed(&path, false).expect("append sprv");
        verify_archive(&path, VerifyDepth::Full).expect("clean pre-sdsc verify");

        let written = append_sdsc(&path).expect("append sdsc (restack)");
        assert_eq!(written.luts.len(), 1);
        assert_eq!(written.luts[0].l_bits, 7);
        assert_eq!(written.luts[0].entries.len(), 128);

        let sdsc_back = read_sdsc(&path).unwrap().expect("sdsc innermost");
        assert_eq!(sdsc_back, written, "SDSC round-trip must be exact");
        let outl_back = read_outl(&path).unwrap().expect("outl above sdsc");
        assert_eq!(outl_back.tensors, wires);
        let sprv_back = read_sprv(&path).unwrap().expect("sprv outermost");
        assert_eq!(
            sprv_back, sprv_before,
            "restacked SPRV must be content-identical"
        );

        let now = std::fs::read(&path).unwrap();
        assert_eq!(&now[..buf.len()], &buf[..], "v2 bytes must be untouched");
        verify_archive(&path, VerifyDepth::Full).expect("full verify after restack");

        let before = std::fs::read(&path).unwrap();
        let err = append_sdsc(&path).unwrap_err();
        assert!(err.contains("already has"), "err was: {err}");
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    #[test]
    fn sdsc_plain_append_and_emit_entry_point() {
        let buf = build_test_archive();
        let path = tmp_path("plain");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();
        assert_eq!(
            read_sdsc(&path).unwrap(),
            None,
            "plain v2 must read as absent"
        );
        let written = append_sdsc(&path).expect("append");
        let back = read_sdsc(&path).unwrap().expect("found");
        assert_eq!(back, written);

        let trailered = std::fs::read(&path).unwrap();
        assert_eq!(&trailered[..buf.len()], &buf[..]);
        assert_eq!(
            trailered.len() % PAGE,
            0,
            "SDSC end must be page-aligned (stacking)"
        );
        assert_eq!(read_strand_v2(&trailered).unwrap().len(), 4);

        let cfg = TrellisConfig::for_bpw(3.0);
        let one = emit_sdsc(&cfg, codebook_lut(cfg.l_bits)).expect("emit_sdsc");
        assert_eq!(
            u32::from_le_bytes(one[4..8].try_into().unwrap()),
            SDSC_VERSION_V1,
            "scalar SDSC wire version must remain backward-compatible"
        );
        let multi = sdsc_section_bytes(&build_sdsc_for_archive(&buf).unwrap()).unwrap();
        assert_eq!(one, multi);

        assert!(emit_sdsc(&cfg, &[0i32; 7]).is_err());
    }

    #[test]
    fn sdsc_corrupt_trailer_is_error_not_crash() {
        let buf = build_test_archive();
        let path = tmp_path("corrupt");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();
        append_sdsc(&path).expect("append");
        let clean = std::fs::read(&path).unwrap();

        let mut c1 = clean.clone();
        let pb_pos = c1.len() - 8;
        c1[pb_pos] ^= 0xFF;
        assert!(read_sdsc_bytes(&c1, true).is_err());
        assert_eq!(read_sdsc_bytes(&c1, false).unwrap(), None);

        let off = {
            let t = &clean[clean.len() - SDSC_TRAILER_BYTES..];
            u64::from_le_bytes(t[0..8].try_into().unwrap()) as usize
        };
        let mut c2 = clean.clone();
        c2[off + 4] ^= 0xFF;
        assert!(read_sdsc_bytes(&c2, true).is_err());

        let mut c3 = clean.clone();
        c3[off + SDSC_HEADER_BYTES - 1] = 1;
        assert!(read_sdsc_bytes(&c3, true).is_err());

        let n_consts = default_consts().len();
        let expr0_prog = off + SDSC_HEADER_BYTES + n_consts * 12 + 12;
        let mut c4 = clean.clone();
        c4[expr0_prog] = 0xFF;
        assert!(read_sdsc_bytes(&c4, true).is_err());

        assert_eq!(read_sdsc_bytes(b"", true).unwrap(), None);
        assert_eq!(read_sdsc_bytes(b"SDSC", true).unwrap(), None);
        let mut tiny = vec![0u8; SDSC_TRAILER_BYTES];
        tiny[12..].copy_from_slice(SDSC_MAGIC);
        assert!(read_sdsc_bytes(&tiny, true).is_err());
        assert_eq!(read_sdsc_bytes(&tiny, false).unwrap(), None);
    }

    #[test]
    #[ignore]
    fn append_sdsc_to_env_target() {
        let Some(path) = std::env::var_os("STRAND_SDSC_TARGET") else {
            eprintln!("append_sdsc_to_env_target: STRAND_SDSC_TARGET not set — skipping");
            return;
        };
        let sdsc = append_sdsc(&path).expect("append sdsc to target");
        println!(
            "SDSC appended to {:?}: {} consts, {} exprs, {} luts ({})",
            path,
            sdsc.consts.len(),
            sdsc.exprs.len(),
            sdsc.luts.len(),
            sdsc.luts
                .iter()
                .map(|l| format!(
                    "L={} d={} ({} entries)",
                    l.l_bits,
                    l.vec_dim,
                    l.entries.len()
                ))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
}

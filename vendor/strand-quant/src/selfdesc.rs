
use std::fs;
use std::path::Path;

use crate::codebook::codebook_lut;
use crate::format::{read_strand_v2_header, OwnedTensorV2, PAGE};
use crate::outlier_wire::{append_outl, read_outl_bytes};
use crate::provenance_io::{append_sprv, read_sprv_bytes};
use crate::trellis::{read_bits, TrellisConfig};

pub const SDSC_MAGIC: &[u8; 4] = b"SDSC";

const SPRV_MAGIC: &[u8; 4] = b"SPRV";
const OUTL_MAGIC: &[u8; 4] = b"OUTL";
/// SDSQ (sprint Lever 1 side-info rANS) chains above SDSC (SDSC is the innermost
/// section), so both SDSC walkers must step over an SDSQ trailer to reach SDSC
/// beneath it — mirrors the OUTL step-over already here.
const SDSQ_MAGIC: &[u8; 4] = b"SDSQ";

pub const SDSC_VERSION: u32 = 1;

pub const SDSC_HEADER_BYTES: usize = 48;

pub const SDSC_TRAILER_BYTES: usize = 16;

const MAX_CONSTS: usize = 256;
const MAX_EXPRS: usize = 64;
const MAX_LUTS: usize = 32;
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sdsc {
    pub consts: Vec<(u32, i64)>,
    pub exprs: Vec<SdscExpr>,
    pub luts: Vec<SdscLut>,
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
        return Err(format!("sdsc: expr left {} values on the stack (want 1)", st.len()));
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
            op::ADD | op::SUB | op::MUL | op::TDIV | op::NEG | op::ABS | op::WRAP32
            | op::SHL | op::ASR | op::AND | op::OR | op::XOR | op::CLAMP => {}
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
        SdscExpr { id: expr_id::ADVANCE, n_slots: 4, prog: prog_advance() },
        SdscExpr { id: expr_id::EFF_SCALE, n_slots: 2, prog: prog_eff_scale() },
        SdscExpr { id: expr_id::OFFSET, n_slots: 2, prog: prog_offset() },
        SdscExpr { id: expr_id::RECON, n_slots: 3, prog: prog_recon() },
    ]
}

pub fn emit_sdsc(cfg: &TrellisConfig, lut: &[i32]) -> Result<Vec<u8>, String> {
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
        .map(|l| SdscLut { l_bits: l, vec_dim: 1, entries: codebook_lut(l as u32).to_vec() })
        .collect();
    Ok(Sdsc { consts: default_consts(), exprs: default_exprs(), luts })
}

pub fn sdsc_section_bytes(sdsc: &Sdsc) -> Result<Vec<u8>, String> {
    if sdsc.consts.len() > MAX_CONSTS || sdsc.exprs.len() > MAX_EXPRS || sdsc.luts.len() > MAX_LUTS
    {
        return Err("sdsc: section exceeds sanity caps".into());
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
        let want = (1usize << l.l_bits) * d as usize;
        if l.entries.len() != want {
            return Err(format!(
                "sdsc: LUT L={} d={d} has {} entries, want {want}",
                l.l_bits,
                l.entries.len()
            ));
        }
    }

    let mut o = Vec::new();
    o.extend_from_slice(SDSC_MAGIC);
    o.extend_from_slice(&SDSC_VERSION.to_le_bytes());
    o.extend_from_slice(&0u32.to_le_bytes()); 
    o.extend_from_slice(&(sdsc.consts.len() as u32).to_le_bytes());
    o.extend_from_slice(&(sdsc.exprs.len() as u32).to_le_bytes());
    o.extend_from_slice(&(sdsc.luts.len() as u32).to_le_bytes());
    o.extend_from_slice(&[0u8; 24]); 
    debug_assert_eq!(o.len(), SDSC_HEADER_BYTES);

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
    Ok(o)
}

#[inline]
fn page_align(x: usize) -> usize {
    (x + PAGE - 1) & !(PAGE - 1)
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
    let sdsc_bytes: u32 = section
        .len()
        .try_into()
        .map_err(|_| format!("sdsc: section is {} bytes — exceeds the u32 field", section.len()))?;

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
    if version != SDSC_VERSION {
        return Err(format!("sdsc: version {version} != {SDSC_VERSION}"));
    }
    let flags = u32::from_le_bytes(s[8..12].try_into().unwrap());
    if flags != 0 {
        return Err(format!("sdsc: reserved flag bits set: {flags:#x}"));
    }
    let n_consts = u32::from_le_bytes(s[12..16].try_into().unwrap()) as usize;
    let n_exprs = u32::from_le_bytes(s[16..20].try_into().unwrap()) as usize;
    let n_luts = u32::from_le_bytes(s[20..24].try_into().unwrap()) as usize;
    if n_consts > MAX_CONSTS || n_exprs > MAX_EXPRS || n_luts > MAX_LUTS {
        return Err("sdsc: section exceeds sanity caps".into());
    }
    if s[24..SDSC_HEADER_BYTES].iter().any(|&b| b != 0) {
        return Err("sdsc: header reserved bytes not zero".into());
    }

    let mut p = SDSC_HEADER_BYTES;
    let take = |p: &mut usize, n: usize| -> Result<&[u8], String> {
        let end = p.checked_add(n).filter(|&e| e <= s.len()).ok_or("sdsc: section truncated")?;
        let sl = &s[*p..end];
        *p = end;
        Ok(sl)
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
            return Err(format!("sdsc: LUT geometry L={l_bits} d={vec_dim} out of range"));
        }
        let key = (l_bits as u8, vec_dim as u8);
        if prev.map_or(false, |pv| key <= pv) {
            return Err("sdsc: LUT geometries not strictly ascending".into());
        }
        prev = Some(key);
        let want = (1usize << l_bits) * vec_dim as usize;
        if n_entries != want {
            return Err(format!(
                "sdsc: LUT L={l_bits} d={vec_dim} has {n_entries} entries, want {want}"
            ));
        }
        let raw = take(&mut p, n_entries * 4)?;
        let entries: Vec<i32> = raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        luts.push(SdscLut { l_bits: key.0, vec_dim: key.1, entries });
    }

    if p != sdsc_bytes {
        return Err(format!("sdsc: {} trailing bytes after the last LUT", sdsc_bytes - p));
    }
    Ok(Sdsc { consts, exprs, luts })
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
    if d != 1 {
        return Err("sdsc v1: vector-trellis (vec_dim > 1) decode is not covered".into());
    }
    let lut = sdsc.lut(t.base.l_bits, 1)?;
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

        let nk = n * k as usize;
        let mut state: i64 = if enc.tail_biting && nk >= l as usize {
            let mut s: i64 = 0;
            let mut c = bit_cursor;
            for _ in 0..n {
                let sym = (read_bits(&enc.bits, c, k) & kmask) as i64;
                c += k as usize;
                s = eval_expr(adv, &[s, sym, k as i64, mask])?;
            }
            s
        } else {
            blk.init_state as i64 & mask
        };

        for i in 0..n {
            let sym = (read_bits(&enc.bits, bit_cursor, k) & kmask) as i64;
            bit_cursor += k as usize;
            state = eval_expr(adv, &[state, sym, k as i64, mask])?;
            let su = state as usize;
            let q = *lut
                .entries
                .get(su)
                .ok_or("sdsc: state indexes past the embedded LUT")? as i64;
            let sb = i / sub_block;
            let es = *eff.get(sb).ok_or("sdsc: sub-block index out of range")?;
            let off = offs.get(sb).copied().unwrap_or(0);
            out.push(eval_expr(rec_e, &[es, q, off])? as i32);
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
    use crate::provenance_io::{
        append_sprv_computed, read_sprv, verify_archive, VerifyDepth,
    };
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
                    let got =
                        eval_expr(&adv, &[state as i64, sym as i64, k as i64, mask]).unwrap();
                    assert_eq!(got, want, "advance L={l} k={k} s={state} sym={sym}");
                }
            }
        }

        for scale_q in [-2_000_000_000i32, -65536, -1, 0, 1, 12345, 65536, 2_000_000_000] {
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
                &EncodeOpts { tail_biting: true, ..Default::default() },
            ),
            encode_tensor_with(
                &test_weights(700, 5),
                &cfg,
                &EncodeOpts { affine_min: true, ..Default::default() },
            ),
            encode_tensor_with(
                &test_weights(1030, 41),
                &cfg,
                &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() },
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
        let sdsc = read_sdsc_bytes(&trailered, true).unwrap().expect("sdsc found");
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
            assert_eq!(got, want, "SDSC-driven decode diverged on {:?}", t.base.name);
        }
    }

    #[test]
    fn sdsc_restack_under_outl_and_sprv() {
        let buf = build_test_archive();
        let path = tmp_path("restack");
        let _guard = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();

        let wires = vec![
            Some(OutlierWire::from_selection(1024, vec![7, 600], vec![-100, 42], 0.5, 8)),
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
        assert_eq!(sprv_back, sprv_before, "restacked SPRV must be content-identical");

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
        assert_eq!(read_sdsc(&path).unwrap(), None, "plain v2 must read as absent");
        let written = append_sdsc(&path).expect("append");
        let back = read_sdsc(&path).unwrap().expect("found");
        assert_eq!(back, written);
        
        let trailered = std::fs::read(&path).unwrap();
        assert_eq!(&trailered[..buf.len()], &buf[..]);
        assert_eq!(trailered.len() % PAGE, 0, "SDSC end must be page-aligned (stacking)");
        assert_eq!(read_strand_v2(&trailered).unwrap().len(), 4);

        let cfg = TrellisConfig::for_bpw(3.0);
        let one = emit_sdsc(&cfg, codebook_lut(cfg.l_bits)).expect("emit_sdsc");
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
                .map(|l| format!("L={} d={} ({} entries)", l.l_bits, l.vec_dim, l.entries.len()))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
}

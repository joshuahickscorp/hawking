//! **The one source-format registry.** Every weight representation an OFFICIAL model source actually
//! ships is declared here once, as a row in one table, with exact byte accounting and a bounded CPU
//! reference decoder. There is no second dtype table, no per-adapter dequantizer, no "overhead" bucket.
//!
//! This module EXTENDS [`super::source_decl`]: a [`FormatDecl`] declares itself into the one
//! [`SourceRecord`] (container -> `formats`, dtype -> `tensor_types`) and seals its accounting through the
//! Seed's one evidence engine. It does not define a second source authority.
//!
//! Element decode reuses the primitives already in this crate rather than re-implementing them:
//! [`crate::safetensors::bf16_to_f32`], [`crate::quant::f16_to_f32`], [`crate::mxfp4::dequant_row`]
//! (OCP E2M1 + E8M0), and [`crate::quant::dequant`] for GGUF blocks.
//!
//! ## Accounting honesty (the reason this file exists)
//!
//! [`FormatDecl::serialized_bits`] counts EVERY byte the format costs on disk: element payload,
//! per-block scales, shared exponents, zero points, per-tensor global scales, and the padding implied by
//! sub-byte element widths. `bits_per_weight` is that total divided by the element count. A format that
//! reported only `elem_bits` would understate MXFP4 by 0.25 bpw and NVFP4 by 0.5 bpw and would corrupt
//! every downstream BPW claim; the test `accounted_bits_equal_serialized_bits` asserts, for a synthetic
//! block of every declared format, that the accounted bits equal 8x the bytes actually serialized.
//!
//! ## References the semantics are bound to
//!
//! - OCP Microscaling Formats (MX) Specification v1.0: E2M1 element, block of 32, E8M0 shared scale.
//! - OCP 8-bit Floating Point Specification (OFP8) v1.0: E4M3 (bias 7, no infinities, S.1111.111 = NaN,
//!   max finite 448) and E5M2 (bias 15, IEEE-754-like, max finite 57344).
//! - NVIDIA NVFP4 (TensorRT Model Optimizer / `nvidia/*-FP4` sources): E2M1 element, block of 16, E4M3
//!   block scale, plus one FP32 per-tensor scale.
//! - DeepSeek-V3 official weights: FP8 E4M3 elements with FP32 `weight_scale_inv` per 128-wide block.
//! - `compressed-tensors` (`pack-quantized`) and GPTQ: INT4 nibbles packed 8-per-int32 little-endian,
//!   FP16 group scale, optional packed 4-bit group zero point.
//! - safetensors container: 8-byte little-endian header length, then a JSON header of
//!   `name -> {dtype, shape, data_offsets}`; GGUF container: `GGUF` magic, u32 version.
//!
//! Nothing here is implemented because a marketing page mentions it. Formats with no official source to
//! bind to, and GGUF block types with no CPU reference decoder in [`crate::quant`], are declared with
//! `decode: Decode::Unsupported` and a stated reason. Their accounting is still exact.

use super::source_decl::SourceRecord;
use crate::evidence::receipt;
use crate::gguf::GgmlType;
use crate::record::Record;
use crate::{Error, Result};
use serde::Serialize;

/// A bounded decode never expands more than this many elements at once. A 120B shard holds billions of
/// weights; a reference decoder that could be handed a whole shard is a memory bug waiting for a
/// campaign to hit it. Callers slice a row (see [`row_span`]) and decode that.
pub const MAX_DECODE_ELEMS: usize = 1 << 22;

/// The container the bytes arrive in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Container {
    Safetensors,
    Gguf,
}

impl Container {
    pub fn as_str(self) -> &'static str {
        match self {
            Container::Safetensors => "safetensors",
            Container::Gguf => "gguf",
        }
    }
}

/// How the element payload is turned into f32. One variant per distinct arithmetic, not per marketing
/// name; `Unsupported` carries the reason it is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decode {
    /// IEEE binary32, little-endian.
    F32,
    /// IEEE binary16.
    F16,
    /// bfloat16 (top 16 bits of binary32).
    Bf16,
    /// OFP8 E4M3, multiplied by the block/global scale (FP32 scale bytes).
    F8E4M3,
    /// OFP8 E5M2, multiplied by the block/global scale (FP32 scale bytes).
    F8E5M2,
    /// OCP MXFP4: E2M1 nibble * 2^(E8M0 byte - 127).
    Mxfp4,
    /// NVFP4: E2M1 nibble * E4M3 block scale * FP32 per-tensor scale.
    Nvfp4,
    /// INT4 nibble, symmetric: (q - 8) * scale, FP16 group scale.
    Int4Sym,
    /// INT4 nibble, asymmetric: (q - zp) * scale, FP16 group scale, packed 4-bit group zero point.
    Int4Asym,
    /// GGUF block type, decoded by [`crate::quant::dequant`].
    Ggml(GgmlType),
    /// Declared for accounting only. The `&'static str` is the reason decode is refused.
    Unsupported(&'static str),
}

/// One row of the one registry. This is the ENTIRE per-format surface.
#[derive(Debug, Clone, Copy)]
pub struct FormatDecl {
    /// Registry key, also the value written into `SourceRecord::tensor_types`.
    pub name: &'static str,
    pub container: Container,
    pub decode: Decode,
    /// Bits per stored element, before any scale/zero-point/padding.
    pub elem_bits: u32,
    /// Elements sharing one scale. 0 = unblocked (no per-block scale bytes exist).
    pub block_elems: u32,
    /// Bits of scale per block (E8M0 = 8, E4M3 = 8, FP16 = 16, FP32 = 32, GGUF = the whole non-payload
    /// part of the block: `d`, `dmin`, and packed sub-scales).
    pub scale_bits: u32,
    /// Bits of zero point per block.
    pub zero_point_bits: u32,
    /// Bits of per-tensor scale, paid once for the whole tensor.
    pub global_scale_bits: u32,
    /// The official specification or source this row's semantics are bound to.
    pub reference: &'static str,
}

const fn st(
    name: &'static str,
    decode: Decode,
    elem_bits: u32,
    block_elems: u32,
    scale_bits: u32,
    zero_point_bits: u32,
    global_scale_bits: u32,
    reference: &'static str,
) -> FormatDecl {
    FormatDecl {
        name,
        container: Container::Safetensors,
        decode,
        elem_bits,
        block_elems,
        scale_bits,
        zero_point_bits,
        global_scale_bits,
        reference,
    }
}

/// GGUF row. `scale_bits` is the exact non-payload remainder of the block, so
/// `elem_bits * block_elems + scale_bits == bytes_per_block * 8` for every entry (asserted by test).
const fn gg(name: &'static str, t: GgmlType, decode: Decode, elem_bits: u32, block_elems: u32, scale_bits: u32) -> FormatDecl {
    let _ = t;
    FormatDecl {
        name,
        container: Container::Gguf,
        decode,
        elem_bits,
        block_elems,
        scale_bits,
        zero_point_bits: 0,
        global_scale_bits: 0,
        reference: "GGML/GGUF block layout (llama.cpp ggml-common.h)",
    }
}

/// THE table. Adding a representation means adding a row here, not a module.
pub const FORMATS: &[FormatDecl] = &[
    // ---- unscaled floating point, as shipped in safetensors sources ----
    st("f32", Decode::F32, 32, 0, 0, 0, 0, "IEEE 754 binary32; safetensors dtype F32"),
    st("f16", Decode::F16, 16, 0, 0, 0, 0, "IEEE 754 binary16; safetensors dtype F16"),
    st("bf16", Decode::Bf16, 16, 0, 0, 0, 0, "bfloat16; safetensors dtype BF16"),
    // ---- FP8 (OFP8). Two scaling conventions both appear in official sources. ----
    st(
        "fp8_e4m3_block128",
        Decode::F8E4M3,
        8,
        128,
        32,
        0,
        0,
        "OCP OFP8 v1.0 E4M3; DeepSeek-V3 weight_scale_inv, FP32 scale per 128-wide block",
    ),
    st(
        "fp8_e4m3_tensor",
        Decode::F8E4M3,
        8,
        0,
        0,
        0,
        32,
        "OCP OFP8 v1.0 E4M3; compressed-tensors float-quantized, one FP32 weight_scale per tensor",
    ),
    st(
        "fp8_e5m2_tensor",
        Decode::F8E5M2,
        8,
        0,
        0,
        0,
        32,
        "OCP OFP8 v1.0 E5M2; compressed-tensors float-quantized, one FP32 weight_scale per tensor",
    ),
    // ---- FP4 ----
    st("mxfp4", Decode::Mxfp4, 4, 32, 8, 0, 0, "OCP MX v1.0: E2M1 element, block 32, E8M0 shared exponent (gpt-oss)"),
    st("nvfp4", Decode::Nvfp4, 4, 16, 8, 0, 32, "NVFP4: E2M1 element, block 16, E4M3 block scale, FP32 per-tensor scale"),
    // ---- INT4 ----
    st("int4_g128_sym", Decode::Int4Sym, 4, 128, 16, 0, 0, "compressed-tensors pack-quantized, symmetric, FP16 group scale"),
    st("int4_g128_asym", Decode::Int4Asym, 4, 128, 16, 4, 0, "GPTQ / compressed-tensors asymmetric, FP16 group scale, packed 4-bit group zero point"),
    // ---- GGUF, where a GGUF source is still required ----
    gg("gguf_f32", GgmlType::F32, Decode::Ggml(GgmlType::F32), 32, 1, 0),
    gg("gguf_f16", GgmlType::F16, Decode::Ggml(GgmlType::F16), 16, 1, 0),
    gg("gguf_q8_0", GgmlType::Q8_0, Decode::Ggml(GgmlType::Q8_0), 8, 32, 16),
    gg("gguf_q5_0", GgmlType::Q5_0, Decode::Ggml(GgmlType::Q5_0), 5, 32, 16),
    gg("gguf_q4_k", GgmlType::Q4_K, Decode::Ggml(GgmlType::Q4_K), 4, 256, 128),
    gg("gguf_q6_k", GgmlType::Q6_K, Decode::Ggml(GgmlType::Q6_K), 6, 256, 144),
    // Accounting is exact; decode is refused because this crate has no CPU reference decoder for them.
    gg("gguf_q4_0", GgmlType::Q4_0, Decode::Unsupported("no CPU reference decoder in crate::quant"), 4, 32, 16),
    gg("gguf_q4_1", GgmlType::Q4_1, Decode::Unsupported("no CPU reference decoder in crate::quant"), 4, 32, 32),
    gg("gguf_q5_1", GgmlType::Q5_1, Decode::Unsupported("no CPU reference decoder in crate::quant"), 5, 32, 32),
    gg("gguf_q8_1", GgmlType::Q8_1, Decode::Unsupported("no CPU reference decoder in crate::quant"), 8, 32, 32),
    gg("gguf_q2_k", GgmlType::Q2_K, Decode::Unsupported("no CPU reference decoder in crate::quant"), 2, 256, 160),
    gg("gguf_q3_k", GgmlType::Q3_K, Decode::Unsupported("no CPU reference decoder in crate::quant"), 3, 256, 112),
    gg("gguf_q5_k", GgmlType::Q5_K, Decode::Unsupported("no CPU reference decoder in crate::quant"), 5, 256, 128),
    gg("gguf_q8_k", GgmlType::Q8_K, Decode::Unsupported("no CPU reference decoder in crate::quant"), 8, 256, 288),
    // Declared unsupported on purpose: no official model source ships these as a WEIGHT element type.
    // MXFP6/MXFP8 exist in the OCP MX spec but there is no official checkpoint to bind semantics to, and
    // E8M0 appears only as the MX shared scale, never as a stored weight.
    st("mxfp8", Decode::Unsupported("OCP MX defines it; no official model source ships it as weights"), 8, 32, 8, 0, 0, "OCP MX v1.0"),
    st("mxfp6", Decode::Unsupported("OCP MX defines it; no official model source ships it as weights"), 6, 32, 8, 0, 0, "OCP MX v1.0"),
];

/// Look a format up by its registry key.
pub fn lookup(name: &str) -> Option<&'static FormatDecl> {
    FORMATS.iter().find(|f| f.name == name)
}

/// Map a safetensors header `dtype` string onto the registry. Ambiguous element types (FP8, INT4) carry
/// no scaling convention in the header, so the caller must pick the row from the source's own
/// quantization config; this maps only the self-describing dtypes.
pub fn from_safetensors_dtype(dtype: &str) -> Result<&'static FormatDecl> {
    let name = match dtype {
        "F32" => "f32",
        "F16" => "f16",
        "BF16" => "bf16",
        other => {
            return Err(Error::Model(format!(
                "safetensors dtype {other}: not a self-describing weight dtype in the source-format registry"
            )))
        }
    };
    lookup(name).ok_or_else(|| Error::Model(format!("registry missing {name}")))
}

#[inline]
const fn ceil_div(a: u64, b: u64) -> u64 {
    a.div_ceil(b)
}

#[inline]
const fn to_whole_bytes_bits(bits: u64) -> u64 {
    ceil_div(bits, 8) * 8
}

/// Exact byte accounting for one tensor of `elems` elements in this format. Every field is a real cost
/// on disk; `total_bits` is what the tensor occupies and `bits_per_weight` is that divided by `elems`.
#[derive(Debug, Clone, Serialize)]
pub struct Accounting {
    pub format: String,
    pub container: String,
    pub elems: u64,
    pub blocks: u64,
    /// Element payload including the padding a sub-byte element width forces at the end.
    pub payload_bits: u64,
    pub scale_bits: u64,
    pub zero_point_bits: u64,
    pub global_scale_bits: u64,
    pub total_bits: u64,
    pub bits_per_weight: f64,
}

impl Accounting {
    /// Stable identity over the accounting, through the Seed's one canonical-JSON + sha256 engine.
    pub fn identity(&self) -> Result<String> {
        Ok(Record::new("source_format_accounting", serde_json::to_value(self)?).identity)
    }
    /// Seal the accounting as a receipt through the ONE evidence engine.
    pub fn seal(&self) -> Result<Record> {
        Ok(receipt("source_format_accounting", serde_json::to_value(self)?))
    }
}

impl FormatDecl {
    pub fn decodable(&self) -> bool {
        !matches!(self.decode, Decode::Unsupported(_))
    }

    /// Blocks needed for `elems` elements (0 for unblocked formats).
    pub fn blocks(&self, elems: u64) -> u64 {
        if self.block_elems == 0 {
            0
        } else {
            ceil_div(elems, self.block_elems as u64)
        }
    }

    /// Exact accounting. Nothing is hidden: scales, shared exponents, zero points, per-tensor scales and
    /// sub-byte padding are all counted.
    pub fn accounting(&self, elems: u64) -> Result<Accounting> {
        if elems == 0 {
            return Err(Error::Model(format!("{}: accounting of 0 elements", self.name)));
        }
        let blocks = self.blocks(elems);
        let payload_bits = to_whole_bytes_bits(elems * self.elem_bits as u64);
        let scale_bits = to_whole_bytes_bits(blocks * self.scale_bits as u64);
        let zero_point_bits = to_whole_bytes_bits(blocks * self.zero_point_bits as u64);
        let global_scale_bits = self.global_scale_bits as u64;
        let total_bits = payload_bits + scale_bits + zero_point_bits + global_scale_bits;
        Ok(Accounting {
            format: self.name.into(),
            container: self.container.as_str().into(),
            elems,
            blocks,
            payload_bits,
            scale_bits,
            zero_point_bits,
            global_scale_bits,
            total_bits,
            bits_per_weight: total_bits as f64 / elems as f64,
        })
    }

    /// Total serialized bits for `elems` elements. `8 * (payload + scales + zeros + global) bytes`.
    pub fn serialized_bits(&self, elems: u64) -> Result<u64> {
        Ok(self.accounting(elems)?.total_bits)
    }

    /// GGUF interleaves a block's scale INSIDE the block, so those bytes travel in [`Block::payload`].
    /// Safetensors sources keep scales in a separate tensor. The accounting is identical either way; only
    /// which slice carries the bytes differs.
    fn interleaved_scales(&self) -> bool {
        matches!(self.container, Container::Gguf)
    }
    fn scale_bytes_raw(&self, elems: u64) -> u64 {
        ceil_div(self.blocks(elems) * self.scale_bits as u64, 8)
    }
    /// Bytes of element payload for `elems` elements (what [`Block::payload`] must be, exactly).
    pub fn payload_bytes(&self, elems: u64) -> u64 {
        ceil_div(elems * self.elem_bits as u64, 8) + if self.interleaved_scales() { self.scale_bytes_raw(elems) } else { 0 }
    }
    /// Bytes of separately stored per-block scale for `elems` elements (0 when interleaved).
    pub fn scale_bytes(&self, elems: u64) -> u64 {
        if self.interleaved_scales() {
            0
        } else {
            self.scale_bytes_raw(elems)
        }
    }
    /// Bytes of per-block zero point for `elems` elements.
    pub fn zero_point_bytes(&self, elems: u64) -> u64 {
        ceil_div(self.blocks(elems) * self.zero_point_bits as u64, 8)
    }

    /// Validate a tensor shape against this format. Returns the element count.
    pub fn validate_shape(&self, shape: &[usize]) -> Result<u64> {
        if shape.is_empty() {
            return Err(Error::Model(format!("{}: rank-0 tensor has no weights", self.name)));
        }
        let mut elems: u64 = 1;
        for (i, d) in shape.iter().enumerate() {
            if *d == 0 {
                return Err(Error::Model(format!("{}: shape {shape:?} has zero extent at dim {i}", self.name)));
            }
            elems = elems
                .checked_mul(*d as u64)
                .ok_or_else(|| Error::Model(format!("{}: shape {shape:?} overflows element count", self.name)))?;
        }
        // The fastest-varying dimension is the one blocks run along; a row that is not a whole number of
        // blocks cannot be decoded a row at a time, which is the only bounded access this crate allows.
        let last = *shape.last().unwrap() as u64;
        if self.block_elems > 0 && last % self.block_elems as u64 != 0 {
            return Err(Error::Model(format!(
                "{}: last dim {last} is not a multiple of block {} (bounded row decode impossible)",
                self.name, self.block_elems
            )));
        }
        Ok(elems)
    }

    /// Declare this format into the ONE source record: container into `formats`, dtype into
    /// `tensor_types`. This is the source identity hook; there is no second source authority.
    pub fn declare_into(&self, mut s: SourceRecord) -> SourceRecord {
        let c = self.container.as_str().to_string();
        if !s.formats.contains(&c) {
            s.formats.push(c);
        }
        let n = self.name.to_string();
        if !s.tensor_types.contains(&n) {
            s.tensor_types.push(n);
        }
        s
    }
}

/// The bounded byte view of ONE decode unit (typically a single row). Every slice is caller-supplied and
/// exactly sized; a wrong size is a corruption/truncation signal, not something to tolerate.
#[derive(Debug, Clone, Copy)]
pub struct Block<'a> {
    pub payload: &'a [u8],
    /// Per-block scale bytes; empty for unblocked formats.
    pub scales: &'a [u8],
    /// Per-block zero-point bytes; empty unless the format has one.
    pub zeros: &'a [u8],
    /// Per-tensor scale; must be 1.0 for formats that declare no global scale.
    pub global_scale: f32,
}

impl<'a> Block<'a> {
    pub fn new(payload: &'a [u8]) -> Self {
        Block { payload, scales: &[], zeros: &[], global_scale: 1.0 }
    }
    pub fn with_scales(mut self, scales: &'a [u8]) -> Self {
        self.scales = scales;
        self
    }
    pub fn with_zeros(mut self, zeros: &'a [u8]) -> Self {
        self.zeros = zeros;
        self
    }
    pub fn with_global(mut self, g: f32) -> Self {
        self.global_scale = g;
        self
    }
}

/// Byte ranges (payload, scales, zeros) for row `row` of a `cols`-wide row-major tensor. The caller
/// slices its mmap with these and decodes just that row; the shard is never materialized.
pub fn row_span(
    decl: &FormatDecl,
    row: usize,
    cols: usize,
) -> Result<(std::ops::Range<usize>, std::ops::Range<usize>, std::ops::Range<usize>)> {
    if cols == 0 {
        return Err(Error::Model(format!("{}: zero-width row", decl.name)));
    }
    if decl.block_elems > 0 && cols % decl.block_elems as usize != 0 {
        return Err(Error::Model(format!(
            "{}: cols {cols} is not a multiple of block {}",
            decl.name, decl.block_elems
        )));
    }
    let c = cols as u64;
    let (pb, sb, zb) = (decl.payload_bytes(c) as usize, decl.scale_bytes(c) as usize, decl.zero_point_bytes(c) as usize);
    Ok((row * pb..(row + 1) * pb, row * sb..(row + 1) * sb, row * zb..(row + 1) * zb))
}

#[inline]
fn rd_f32(b: &[u8], i: usize) -> f32 {
    f32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

/// OFP8 E4M3 (OCP OFP8 v1.0): sign, 4-bit exponent bias 7, 3-bit mantissa, no infinities, S.1111.111 is
/// NaN. Max finite magnitude 448.
#[inline]
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let man = (b & 0x07) as f32;
    if exp == 0 {
        sign * man * 2.0f32.powi(-9) // 2^(1-7) * man/8
    } else if exp == 0x0F && (b & 0x07) == 0x07 {
        f32::NAN
    } else {
        sign * (1.0 + man / 8.0) * 2.0f32.powi(exp - 7)
    }
}

/// OFP8 E5M2 (OCP OFP8 v1.0): sign, 5-bit exponent bias 15, 2-bit mantissa, IEEE-754-like infinities and
/// NaNs. Max finite magnitude 57344.
#[inline]
pub fn e5m2_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 2) & 0x1F) as i32;
    let man = (b & 0x03) as f32;
    if exp == 0 {
        sign * man * 2.0f32.powi(-16) // 2^(1-15) * man/4
    } else if exp == 0x1F {
        if man == 0.0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1.0 + man / 4.0) * 2.0f32.powi(exp - 15)
    }
}

#[inline]
fn nibble(payload: &[u8], i: usize) -> u8 {
    // Little-endian nibble order: element 2k in the low nibble of byte k, 2k+1 in the high nibble. This
    // is both the OCP MXFP4 packing and the GPTQ / compressed-tensors int32 packing seen bytewise.
    if i % 2 == 0 {
        payload[i / 2] & 0x0F
    } else {
        payload[i / 2] >> 4
    }
}

/// Unit-scaled E2M1 decode, borrowed from the crate's one OCP FP4 decoder (scale byte 127 = 2^0) so
/// there is exactly one E2M1 table in the crate.
fn e2m1_unit(payload: &[u8], n: usize, out: &mut [f32]) {
    let unit = vec![127u8; n.div_ceil(32)];
    crate::mxfp4::dequant_row(payload, &unit, n, out);
}

/// Reject a block whose slices are not exactly the size the format demands. Truncation, a short final
/// shard write, or a wrong-dtype pairing all land here.
fn check_sizes(decl: &FormatDecl, blk: &Block, n: usize) -> Result<()> {
    let c = n as u64;
    let want = (decl.payload_bytes(c), decl.scale_bytes(c), decl.zero_point_bytes(c));
    let got = (blk.payload.len() as u64, blk.scales.len() as u64, blk.zeros.len() as u64);
    if got != want {
        return Err(Error::Model(format!(
            "{}: {n} elements need (payload,scales,zeros) = {want:?} bytes, got {got:?} (truncated, corrupt, or wrong dtype)",
            decl.name
        )));
    }
    if decl.global_scale_bits == 0 && blk.global_scale != 1.0 {
        return Err(Error::Model(format!("{}: declares no per-tensor scale but global_scale = {}", decl.name, blk.global_scale)));
    }
    if !blk.global_scale.is_finite() {
        return Err(Error::Model(format!("{}: non-finite global scale {}", decl.name, blk.global_scale)));
    }
    Ok(())
}

/// CPU reference decode of `n` elements into `out`. Bounded by [`MAX_DECODE_ELEMS`]; sizes are validated
/// exactly before any read, so a truncated or wrong-dtype buffer is refused rather than misread.
pub fn decode(decl: &FormatDecl, blk: &Block, n: usize, out: &mut [f32]) -> Result<()> {
    if n == 0 {
        return Err(Error::Model(format!("{}: decode of 0 elements", decl.name)));
    }
    if n > MAX_DECODE_ELEMS {
        return Err(Error::Model(format!(
            "{}: decode of {n} elements exceeds the bounded limit {MAX_DECODE_ELEMS}; decode a row at a time (see row_span)",
            decl.name
        )));
    }
    if out.len() < n {
        return Err(Error::Model(format!("{}: out has {} slots for {n} elements", decl.name, out.len())));
    }
    if decl.block_elems > 0 && n % decl.block_elems as usize != 0 {
        return Err(Error::Model(format!("{}: {n} elements is not a whole number of {}-blocks", decl.name, decl.block_elems)));
    }
    check_sizes(decl, blk, n)?;
    let out = &mut out[..n];
    let bs = decl.block_elems as usize;
    match decl.decode {
        Decode::Unsupported(why) => return Err(Error::Model(format!("{}: unsupported source format ({why})", decl.name))),
        Decode::F32 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = rd_f32(blk.payload, i * 4);
            }
        }
        Decode::F16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = crate::quant::f16_to_f32(u16::from_le_bytes([blk.payload[i * 2], blk.payload[i * 2 + 1]]));
            }
        }
        Decode::Bf16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = crate::safetensors::bf16_to_f32(u16::from_le_bytes([blk.payload[i * 2], blk.payload[i * 2 + 1]]));
            }
        }
        Decode::F8E4M3 | Decode::F8E5M2 => {
            let elem: fn(u8) -> f32 = if decl.decode == Decode::F8E4M3 { e4m3_to_f32 } else { e5m2_to_f32 };
            for (i, o) in out.iter_mut().enumerate() {
                let s = if bs == 0 { blk.global_scale } else { rd_f32(blk.scales, (i / bs) * 4) };
                *o = elem(blk.payload[i]) * s;
            }
        }
        Decode::Mxfp4 => crate::mxfp4::dequant_row(blk.payload, blk.scales, n, out),
        Decode::Nvfp4 => {
            e2m1_unit(blk.payload, n, out);
            for (i, o) in out.iter_mut().enumerate() {
                *o *= e4m3_to_f32(blk.scales[i / bs]) * blk.global_scale;
            }
        }
        Decode::Int4Sym | Decode::Int4Asym => {
            let asym = decl.decode == Decode::Int4Asym;
            for (i, o) in out.iter_mut().enumerate() {
                let b = i / bs;
                let scale = crate::quant::f16_to_f32(u16::from_le_bytes([blk.scales[b * 2], blk.scales[b * 2 + 1]]));
                let zp = if asym { nibble(blk.zeros, b) as f32 } else { 8.0 };
                *o = (nibble(blk.payload, i) as f32 - zp) * scale;
            }
        }
        Decode::Ggml(t) => crate::quant::dequant(t, blk.payload, out)?,
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------
// Container header validation. Bounded: only the header prefix is read, never the tensor data.
// ---------------------------------------------------------------------------------------------------

/// A validated safetensors header entry, with absolute file offsets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaderTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub begin: usize,
    pub end: usize,
}

/// A safetensors header longer than this is rejected before allocation. Real shard headers are a few MB.
pub const MAX_SAFETENSORS_HEADER: usize = 64 << 20;

/// Validate a safetensors header from the file prefix alone. `prefix` must contain at least
/// `8 + header_len` bytes; `file_bytes` is the full file length. Checks the length field, the JSON, every
/// entry's dtype/shape/offset consistency, that spans lie inside the file, and that no two tensors
/// overlap. Returns the entries in ascending offset order.
pub fn validate_safetensors_header(prefix: &[u8], file_bytes: usize) -> Result<Vec<HeaderTensor>> {
    if prefix.len() < 8 {
        return Err(Error::Model("safetensors: file shorter than the 8-byte header length".into()));
    }
    let hlen = u64::from_le_bytes(prefix[0..8].try_into().unwrap()) as usize;
    if hlen == 0 || hlen > MAX_SAFETENSORS_HEADER {
        return Err(Error::Model(format!("safetensors: implausible header length {hlen}")));
    }
    let data_start = 8usize
        .checked_add(hlen)
        .ok_or_else(|| Error::Model("safetensors: header length overflows".into()))?;
    if data_start > file_bytes {
        return Err(Error::Model(format!("safetensors: header ends at {data_start}, past eof {file_bytes}")));
    }
    if prefix.len() < data_start {
        return Err(Error::Model(format!("safetensors: header truncated ({} of {data_start} bytes)", prefix.len())));
    }
    let hdr: serde_json::Value =
        serde_json::from_slice(&prefix[8..data_start]).map_err(|e| Error::Model(format!("safetensors header json: {e}")))?;
    let obj = hdr.as_object().ok_or_else(|| Error::Model("safetensors: header is not a JSON object".into()))?;
    let data_bytes = file_bytes - data_start;
    let mut out: Vec<HeaderTensor> = Vec::new();
    for (name, v) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = v["dtype"].as_str().ok_or_else(|| Error::Model(format!("safetensors {name}: missing dtype")))?.to_string();
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .ok_or_else(|| Error::Model(format!("safetensors {name}: missing shape")))?
            .iter()
            .map(|x| x.as_u64().unwrap_or(0) as usize)
            .collect();
        let off = v["data_offsets"].as_array().ok_or_else(|| Error::Model(format!("safetensors {name}: missing data_offsets")))?;
        if off.len() != 2 {
            return Err(Error::Model(format!("safetensors {name}: data_offsets must have 2 entries")));
        }
        let (begin, end) = (off[0].as_u64().unwrap_or(u64::MAX) as usize, off[1].as_u64().unwrap_or(u64::MAX) as usize);
        if begin > end || end > data_bytes {
            return Err(Error::Model(format!(
                "safetensors {name}: span {begin}..{end} outside the {data_bytes}-byte data section"
            )));
        }
        // Self-describing dtypes must have span == exactly the accounted bytes. Quantized dtypes carry
        // their scaling convention in the source config, not the header, so they are span-checked only.
        if let Ok(decl) = from_safetensors_dtype(&dtype) {
            let elems = decl.validate_shape(&shape)?;
            let want = (decl.serialized_bits(elems)? / 8) as usize;
            if end - begin != want {
                return Err(Error::Model(format!(
                    "safetensors {name}: dtype {dtype} shape {shape:?} needs {want} bytes, header spans {}",
                    end - begin
                )));
            }
        }
        out.push(HeaderTensor { name: name.clone(), dtype, shape, begin: data_start + begin, end: data_start + end });
    }
    out.sort_by_key(|t| t.begin);
    for w in out.windows(2) {
        if w[1].begin < w[0].end {
            return Err(Error::Model(format!("safetensors: {} and {} overlap", w[0].name, w[1].name)));
        }
    }
    Ok(out)
}

/// Validate a GGUF header prefix. Returns the container version.
pub fn validate_gguf_header(prefix: &[u8]) -> Result<u32> {
    if prefix.len() < 8 {
        return Err(Error::Gguf("gguf: prefix shorter than magic + version".into()));
    }
    if &prefix[0..4] != b"GGUF" {
        return Err(Error::Gguf(format!("gguf: bad magic {:?}", &prefix[0..4])));
    }
    let v = u32::from_le_bytes(prefix[4..8].try_into().unwrap());
    if !(2..=3).contains(&v) {
        return Err(Error::Gguf(format!("gguf: unsupported container version {v}")));
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(name: &str) -> &'static FormatDecl {
        lookup(name).unwrap()
    }

    /// Serialize a synthetic tensor of `n` elements: (payload, scales, zeros). Bytes are arbitrary but
    /// the LENGTHS are what the format really costs, which is what the accounting test compares against.
    fn synth(decl: &FormatDecl, n: u64) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        (
            vec![0u8; decl.payload_bytes(n) as usize],
            vec![0u8; decl.scale_bytes(n) as usize],
            vec![0u8; decl.zero_point_bytes(n) as usize],
        )
    }

    // ---- the honesty test ----

    #[test]
    fn accounted_bits_equal_serialized_bits() {
        for decl in FORMATS {
            let n = if decl.block_elems == 0 { 64 } else { decl.block_elems as u64 * 3 };
            let (p, s, z) = synth(decl, n);
            let serialized_bytes = p.len() + s.len() + z.len() + (decl.global_scale_bits as usize / 8);
            let acc = decl.accounting(n).unwrap();
            assert_eq!(
                acc.total_bits,
                serialized_bytes as u64 * 8,
                "{}: accounted {} bits vs {} bytes actually serialized",
                decl.name,
                acc.total_bits,
                serialized_bytes
            );
            assert_eq!(acc.total_bits, decl.serialized_bits(n).unwrap());
            // and the headline number nothing may hide behind
            assert!(acc.bits_per_weight >= decl.elem_bits as f64, "{}: bpw below element width", decl.name);
        }
    }

    #[test]
    fn gguf_rows_match_the_real_block_layout() {
        for decl in FORMATS.iter().filter(|f| f.container == Container::Gguf) {
            let t = match decl.decode {
                Decode::Ggml(t) => t,
                _ => match decl.name {
                    "gguf_q4_0" => GgmlType::Q4_0,
                    "gguf_q4_1" => GgmlType::Q4_1,
                    "gguf_q5_1" => GgmlType::Q5_1,
                    "gguf_q8_1" => GgmlType::Q8_1,
                    "gguf_q2_k" => GgmlType::Q2_K,
                    "gguf_q3_k" => GgmlType::Q3_K,
                    "gguf_q5_k" => GgmlType::Q5_K,
                    "gguf_q8_k" => GgmlType::Q8_K,
                    o => panic!("unmapped gguf row {o}"),
                },
            };
            let (bs, bb) = t.block_layout();
            assert_eq!(bs, decl.block_elems as u64, "{}: block size", decl.name);
            assert_eq!(decl.serialized_bits(bs).unwrap(), bb * 8, "{}: block bytes", decl.name);
        }
    }

    #[test]
    fn known_bits_per_weight_are_not_understated() {
        // MXFP4 is 4.25 bpw, not 4: the E8M0 shared exponent is a real byte per 32 weights.
        assert_eq!(d("mxfp4").accounting(32).unwrap().bits_per_weight, 4.25);
        // NVFP4 is 4.5 bpw plus the amortized per-tensor FP32 scale.
        let nv = d("nvfp4").accounting(1024).unwrap();
        assert_eq!(nv.bits_per_weight, 4.5 + 32.0 / 1024.0);
        // INT4 g128 symmetric = 4.125; asymmetric adds a packed 4-bit zero point per group.
        assert_eq!(d("int4_g128_sym").accounting(1024).unwrap().bits_per_weight, 4.125);
        assert!(d("int4_g128_asym").accounting(1024).unwrap().bits_per_weight > 4.125);
        // FP8 with a 128-wide FP32 block scale is 8.25, never 8.
        assert_eq!(d("fp8_e4m3_block128").accounting(128).unwrap().bits_per_weight, 8.25);
        assert_eq!(d("bf16").accounting(10).unwrap().bits_per_weight, 16.0);
    }

    // ---- round-trip decode against hand-computed values ----

    #[test]
    fn f32_f16_bf16_decode_exactly() {
        let mut out = [0f32; 2];
        let p: Vec<u8> = [1.0f32, -2.5].iter().flat_map(|x| x.to_le_bytes()).collect();
        decode(d("f32"), &Block::new(&p), 2, &mut out).unwrap();
        assert_eq!(out, [1.0, -2.5]);
        // f16: 0x3C00 = 1.0, 0xC100 = -2.5
        let p = [0x00u8, 0x3C, 0x00, 0xC1];
        decode(d("f16"), &Block::new(&p), 2, &mut out).unwrap();
        assert_eq!(out, [1.0, -2.5]);
        // bf16: 0x3F80 = 1.0, 0xC020 = -2.5
        let p = [0x80u8, 0x3F, 0x20, 0xC0];
        decode(d("bf16"), &Block::new(&p), 2, &mut out).unwrap();
        assert_eq!(out, [1.0, -2.5]);
    }

    #[test]
    fn fp8_matches_ofp8_hand_values() {
        // E4M3: 0x38 = exp 7 (bias 7) mantissa 0 -> 1.0; 0x3C -> 1.5; 0x7E = exp 15 man 6 -> 448 (max
        // finite); 0x7F is NaN; 0x01 is the smallest subnormal 2^-9.
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0x3C), 1.5);
        assert_eq!(e4m3_to_f32(0xB8), -1.0);
        assert_eq!(e4m3_to_f32(0x7E), 448.0);
        assert!(e4m3_to_f32(0x7F).is_nan());
        assert_eq!(e4m3_to_f32(0x01), 2.0f32.powi(-9));
        // E5M2: 0x3C = exp 15 man 0 -> 1.0; 0x3D -> 1.25; 0x7B = exp 30 man 3 -> 57344; 0x7C is +inf.
        assert_eq!(e5m2_to_f32(0x3C), 1.0);
        assert_eq!(e5m2_to_f32(0x3D), 1.25);
        assert_eq!(e5m2_to_f32(0x7B), 57344.0);
        assert!(e5m2_to_f32(0x7C).is_infinite());
        assert!(e5m2_to_f32(0x7D).is_nan());
        assert_eq!(e5m2_to_f32(0x01), 2.0f32.powi(-16));
    }

    #[test]
    fn fp8_block_and_tensor_scaling_round_trip() {
        // 128 E4M3 ones with an FP32 block scale of 3.0 -> 3.0 everywhere.
        let decl = d("fp8_e4m3_block128");
        let payload = vec![0x38u8; 128];
        let scales = 3.0f32.to_le_bytes().to_vec();
        let mut out = vec![0f32; 128];
        decode(decl, &Block::new(&payload).with_scales(&scales), 128, &mut out).unwrap();
        assert!(out.iter().all(|v| *v == 3.0));
        // per-tensor variant: same elements, scale carried on the block view.
        let decl = d("fp8_e5m2_tensor");
        let payload = vec![0x3Cu8; 4]; // 1.0
        let mut out = vec![0f32; 4];
        decode(decl, &Block::new(&payload).with_global(0.5), 4, &mut out).unwrap();
        assert_eq!(out, vec![0.5f32; 4]);
    }

    #[test]
    fn mxfp4_and_nvfp4_round_trip() {
        // MXFP4: codes 2 (+1.0) and 7 (+6.0) packed low/high, E8M0 128 = 2^1.
        let mut payload = vec![0u8; 16];
        payload[0] = 0x72;
        let scales = [128u8];
        let mut out = vec![0f32; 32];
        decode(d("mxfp4"), &Block::new(&payload).with_scales(&scales), 32, &mut out).unwrap();
        assert_eq!(out[0], 2.0);
        assert_eq!(out[1], 12.0);
        // NVFP4: same codes, E4M3 block scale 0x38 = 1.0, per-tensor scale 4.0, block of 16.
        let mut payload = vec![0u8; 8];
        payload[0] = 0x72;
        let scales = [0x38u8];
        let mut out = vec![0f32; 16];
        decode(d("nvfp4"), &Block::new(&payload).with_scales(&scales).with_global(4.0), 16, &mut out).unwrap();
        assert_eq!(out[0], 4.0);
        assert_eq!(out[1], 24.0);
    }

    #[test]
    fn int4_symmetric_and_asymmetric_round_trip() {
        // group of 128, scale f16 = 2.0 (0x4000). nibbles: 8 -> 0, 9 -> +2, 0 -> -16 under symmetric.
        let mut payload = vec![0x88u8; 64];
        payload[0] = 0x98; // element0 = 8, element1 = 9
        let scales = 0x4000u16.to_le_bytes().to_vec();
        let mut out = vec![0f32; 128];
        decode(d("int4_g128_sym"), &Block::new(&payload).with_scales(&scales), 128, &mut out).unwrap();
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 2.0);
        // asymmetric with zero point 9: element0 (q=8) -> -2, element1 (q=9) -> 0.
        let zeros = [0x09u8];
        decode(d("int4_g128_asym"), &Block::new(&payload).with_scales(&scales).with_zeros(&zeros), 128, &mut out).unwrap();
        assert_eq!(out[0], -2.0);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn gguf_decode_delegates_to_the_one_dequantizer() {
        // Q8_0 block: f16 d = 1.0, qs = 0..31.
        let mut payload = vec![0u8; 34];
        payload[0..2].copy_from_slice(&crate::quant::f32_to_f16_bits(1.0).to_le_bytes());
        for i in 0..32 {
            payload[2 + i] = i as u8;
        }
        let mut out = vec![0f32; 32];
        decode(d("gguf_q8_0"), &Block::new(&payload), 32, &mut out).unwrap();
        assert_eq!(out[5], 5.0);
        assert_eq!(out[31], 31.0);
    }

    // ---- rejection: truncation, corruption, wrong dtype, unbounded reads ----

    #[test]
    fn truncated_and_wrong_dtype_buffers_are_rejected() {
        let mut out = vec![0f32; 32];
        // MXFP4 row missing its last payload byte.
        let payload = vec![0u8; 15];
        let scales = [127u8];
        assert!(decode(d("mxfp4"), &Block::new(&payload).with_scales(&scales), 32, &mut out).is_err());
        // right payload, missing scales (the classic "scales are free" bug).
        let payload = vec![0u8; 16];
        assert!(decode(d("mxfp4"), &Block::new(&payload), 32, &mut out).is_err());
        // an MXFP4 row handed to the f16 decoder: sizes do not match, refused rather than misread.
        assert!(decode(d("f16"), &Block::new(&payload), 32, &mut out).is_err());
        // asymmetric int4 with no zero points.
        let p = vec![0u8; 64];
        let s = vec![0u8; 2];
        let mut o = vec![0f32; 128];
        assert!(decode(d("int4_g128_asym"), &Block::new(&p).with_scales(&s), 128, &mut o).is_err());
        // partial block.
        assert!(decode(d("mxfp4"), &Block::new(&vec![0u8; 8]).with_scales(&scales), 16, &mut out).is_err());
        // unsupported format refuses even with perfectly sized bytes.
        let decl = d("gguf_q2_k");
        let (p, s, z) = synth(decl, 256);
        let mut o = vec![0f32; 256];
        assert!(decode(decl, &Block::new(&p).with_scales(&s).with_zeros(&z), 256, &mut o).is_err());
        // a global scale on a format that declares none is a lie about the byte cost.
        assert!(decode(d("f32"), &Block::new(&vec![0u8; 8]).with_global(2.0), 2, &mut out).is_err());
    }

    #[test]
    fn decode_is_bounded() {
        let n = MAX_DECODE_ELEMS + 32;
        let mut out = vec![0f32; 1];
        assert!(decode(d("mxfp4"), &Block::new(&[]), n, &mut out).is_err());
    }

    #[test]
    fn row_span_reads_only_one_row() {
        let (p, s, z) = row_span(d("mxfp4"), 3, 64).unwrap();
        assert_eq!(p, 96..128); // 64 elems = 32 payload bytes per row
        assert_eq!(s, 6..8); // 2 E8M0 bytes per row
        assert_eq!(z, 0..0);
        assert!(row_span(d("mxfp4"), 0, 48).is_err()); // 48 is not a multiple of 32
    }

    #[test]
    fn shape_validation() {
        assert_eq!(d("bf16").validate_shape(&[4, 8]).unwrap(), 32);
        assert!(d("bf16").validate_shape(&[]).is_err());
        assert!(d("bf16").validate_shape(&[4, 0]).is_err());
        // last dim must be a whole number of blocks for bounded row decode
        assert!(d("mxfp4").validate_shape(&[8, 48]).is_err());
        assert_eq!(d("mxfp4").validate_shape(&[8, 64]).unwrap(), 512);
    }

    // ---- container headers ----

    fn st_file(header: &str, data: usize) -> (Vec<u8>, usize) {
        let mut v = (header.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(header.as_bytes());
        let total = v.len() + data;
        (v, total)
    }

    #[test]
    fn safetensors_header_validates_and_rejects_corruption() {
        let h = r#"{"a":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,16]},"b":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}}"#;
        let (prefix, total) = st_file(h, 24);
        let t = validate_safetensors_header(&prefix, total).unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].name, "a");
        assert_eq!(t[0].end - t[0].begin, 16);

        // shape/dtype disagree with the span: 2x4 BF16 is 16 bytes, not 12.
        let bad = r#"{"a":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,12]}}"#;
        let (p, tot) = st_file(bad, 12);
        assert!(validate_safetensors_header(&p, tot).is_err());

        // overlapping tensors
        let ov = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}"#;
        let (p, tot) = st_file(ov, 12);
        assert!(validate_safetensors_header(&p, tot).is_err());

        // span past the end of the file
        let past = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let (p, tot) = st_file(past, 4);
        assert!(validate_safetensors_header(&p, tot).is_err());

        // truncated header, absurd header length, non-object header, junk
        let (p, tot) = st_file(h, 24);
        assert!(validate_safetensors_header(&p[..p.len() - 5], tot).is_err());
        assert!(validate_safetensors_header(&[0xFFu8; 16], 1024).is_err());
        let (p, tot) = st_file("[]", 0);
        assert!(validate_safetensors_header(&p, tot).is_err());
        let (p, tot) = st_file("not json", 0);
        assert!(validate_safetensors_header(&p, tot).is_err());
        assert!(validate_safetensors_header(&[0u8; 4], 4).is_err());
    }

    #[test]
    fn unknown_safetensors_dtype_is_refused_not_guessed() {
        assert!(from_safetensors_dtype("BF16").is_ok());
        assert!(from_safetensors_dtype("F64").is_err());
        assert!(from_safetensors_dtype("F8_E4M3").is_err()); // scaling convention lives in the config
    }

    #[test]
    fn gguf_header_validation() {
        let mut v = b"GGUF".to_vec();
        v.extend_from_slice(&3u32.to_le_bytes());
        assert_eq!(validate_gguf_header(&v).unwrap(), 3);
        assert!(validate_gguf_header(b"GGUF\x63\x00\x00\x00").is_err()); // version 99
        assert!(validate_gguf_header(b"GGML\x03\x00\x00\x00").is_err());
        assert!(validate_gguf_header(b"GG").is_err());
    }

    // ---- source identity hook ----

    #[test]
    fn formats_declare_into_the_one_source_record() {
        let s = d("mxfp4").declare_into(d("bf16").declare_into(SourceRecord::hf("openai/gpt-oss-120b", "b5c939de")));
        assert_eq!(s.formats, vec!["safetensors".to_string()]); // container declared once
        assert_eq!(s.tensor_types, vec!["bf16".to_string(), "mxfp4".to_string()]);
        assert!(!s.identity().unwrap().is_empty());
        let acc = d("mxfp4").accounting(4096).unwrap();
        let rec = acc.seal().unwrap();
        assert!(rec.verify().is_ok() && rec.kind == "source_format_accounting");
        assert_eq!(acc.identity().unwrap().len(), 64);
    }

    #[test]
    fn registry_keys_are_unique() {
        for (i, a) in FORMATS.iter().enumerate() {
            assert!(FORMATS[i + 1..].iter().all(|b| b.name != a.name), "duplicate row {}", a.name);
        }
    }
}

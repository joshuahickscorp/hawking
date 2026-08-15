//! Reader for Ascension's direct Qwen complete-binary tensor artifacts.
//!
//! The physical Qwen30 and Qwen80 builders write every official source tensor
//! in one small fixed-group layout: an FP16 scale and packed sign bits for each
//! 128-element group.  This module is the strict native read side of that
//! format.  It validates every byte and can reconstruct a tensor for parity
//! work; it is not, by itself, a model loader, Metal executor, or a quality /
//! TPS qualification.

use crate::{Error, Result};
use half::f16;
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::thread;

mod activation_weighted_svd;
mod admission_warm_receipt;
mod qwen80_uniform_q4;
mod uniform_q4;
mod uniform_qn;
pub use activation_weighted_svd::*;
pub use admission_warm_receipt::{
    receipt_path_for_seal, warm_receipt_enabled, ADMISSION_WARM_RECEIPT_SCHEMA,
};
pub use qwen80_uniform_q4::*;
pub use uniform_q4::*;
pub use uniform_qn::*;

pub const COMPLETE_BINARY_MAGIC: [u8; 8] = *b"HQ30G1B1";
pub const COMPLETE_BINARY_VERSION: u32 = 1;
pub const COMPLETE_BINARY_HEADER_BYTES: usize = 32;
pub const QWEN30_COMPLETE_BINARY_SCHEMA: &str =
    "hawking.ascension.qwen30_complete_binary_gravity.v1";
pub const QWEN80_COMPLETE_BINARY_SCHEMA: &str =
    "hawking.ascension.qwen80_complete_binary_gravity.v1";
pub const COMPLETE_BINARY_REVALIDATION_SCHEMA: &str =
    "hawking.ascension.complete_binary_source_revalidation.v1";
pub const COMPLETE_BINARY_CANDIDATE_STATUS: &str =
    "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED";
/// Schema for the separately rooted Qwen30 gate/up residual experiment.  It
/// is deliberately *not* accepted by the ordinary direct-binary admission
/// interface: its two explicitly selected tensors have a different physical
/// layout and require this exact reader.
pub const QWEN30_QUALITY_REPACK_SCHEMA: &str =
    "hawking.ascension.qwen30_quality_repack_candidate.v1";
pub const QWEN30_QUALITY_REPACK_TERMINAL_SCHEMA: &str =
    "hawking.ascension.complete_binary_terminal_status.v1";
pub const QWEN30_QUALITY_REPACK_SELECTION_SCHEMA: &str =
    "hawking.ascension.qwen30_quality_repack_selection.v1";
pub const QWEN30_QUALITY_REPACK_SNAPSHOT_SCHEMA: &str =
    "hawking.ascension.qwen30_quality_repack_source_snapshot.v1";
pub const QWEN30_QUALITY_REPACK_MAGIC: [u8; 8] = *b"HQ30GR2\0";
pub const QWEN30_QUALITY_REPACK_VERSION: u32 = 1;
pub const QWEN30_QUALITY_REPACK_TENSOR_COUNT: usize = 18_867;
const QWEN30_QUALITY_REPACK_BRANCH_ID: &str = "qwen30-gate-up-sparse-fp16-residual-v1";
const QWEN30_QUALITY_REPACK_MODEL_ID: &str =
    "Qwen3-Coder-30B-A3B-Instruct-quality-gate-up-residual-v1";
const QWEN30_QUALITY_REPACK_RESIDUAL_MAGIC_TEXT: &str = "HQ30GR2\0";
const QWEN30_QUALITY_REPACK_SELECTED_ORGANS: [&str; 2] = [
    "model.layers.0.mlp.experts.0.gate_proj.weight",
    "model.layers.0.mlp.experts.0.up_proj.weight",
];
/// Full candidate admission is deliberately a bounded read-only scan.  Four
/// shard lanes keep independent payload verification useful on the live host
/// without turning an integrity check into an unbounded memory or I/O burst.
const QWEN30_QUALITY_REPACK_MAX_PAYLOAD_VERIFY_WORKERS: usize = 4;
pub const QWEN30_QUALITY_REPACK_PAYLOAD_VERIFY_MODE: &str =
    "bounded_parallel_source_shard_lanes_ordered_reconciliation_v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompleteBinaryHeader {
    pub version: u32,
    pub group_size: usize,
    pub shape: Vec<usize>,
    pub elements: usize,
    pub groups: usize,
    pub scale_offset: usize,
    pub sign_offset: usize,
    pub payload_bytes: usize,
}

/// Header geometry for an experimental sparse FP16 residual over an otherwise
/// ordinary direct Qwen30 payload.  The residual is intentionally limited to
/// the two organs sealed by the quality-selection receipt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen30QualityResidualHeader {
    pub shape: Vec<usize>,
    pub base: CompleteBinaryHeader,
    pub residual_count: usize,
    pub base_offset: usize,
    pub base_payload_bytes: usize,
    pub indices_offset: usize,
    pub values_offset: usize,
    pub payload_bytes: usize,
    pub indices_sha256: String,
    pub values_sha256: String,
}

fn read_u16(payload: &[u8], offset: usize) -> Result<u16> {
    let bytes = payload
        .get(offset..offset + 2)
        .ok_or_else(|| Error::Model("complete binary is truncated while reading u16".into()))?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32(payload: &[u8], offset: usize) -> Result<u32> {
    let bytes = payload
        .get(offset..offset + 4)
        .ok_or_else(|| Error::Model("complete binary is truncated while reading u32".into()))?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_u64(payload: &[u8], offset: usize) -> Result<u64> {
    let bytes = payload
        .get(offset..offset + 8)
        .ok_or_else(|| Error::Model("complete binary is truncated while reading u64".into()))?;
    Ok(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

/// Parse and validate one direct complete-binary Qwen tensor payload.
///
/// The final group retains all padded sign bits.  That is deliberate: device
/// kernels can address a uniform fixed group without a special tail format,
/// and the physical byte ledger must exactly bill those retained bits.
pub fn parse_complete_binary_header(payload: &[u8]) -> Result<CompleteBinaryHeader> {
    if payload.len() < COMPLETE_BINARY_HEADER_BYTES {
        return Err(Error::Model(format!(
            "complete binary payload is {} bytes; header requires {COMPLETE_BINARY_HEADER_BYTES}",
            payload.len()
        )));
    }
    if payload[..8] != COMPLETE_BINARY_MAGIC {
        return Err(Error::Model(
            "complete binary magic does not match Qwen direct layout".into(),
        ));
    }
    let version = read_u32(payload, 8)?;
    if version != COMPLETE_BINARY_VERSION {
        return Err(Error::Model(format!(
            "complete binary version {version} is unsupported (expected {COMPLETE_BINARY_VERSION})"
        )));
    }
    let group_size = read_u32(payload, 12)? as usize;
    let rank = read_u16(payload, 16)? as usize;
    let reserved = read_u16(payload, 18)?;
    let element_u64 = read_u64(payload, 20)?;
    let reserved_tail = read_u32(payload, 28)?;
    if reserved != 0 || reserved_tail != 0 {
        return Err(Error::Model(
            "complete binary reserved header fields must be zero".into(),
        ));
    }
    if group_size == 0 || group_size % 8 != 0 {
        return Err(Error::Model(format!(
            "complete binary group_size={group_size} must be a positive multiple of eight"
        )));
    }
    if rank == 0 {
        return Err(Error::Model(
            "complete binary tensor rank must be positive".into(),
        ));
    }
    let dimensions_offset = COMPLETE_BINARY_HEADER_BYTES;
    let dimensions_bytes = rank
        .checked_mul(4)
        .ok_or_else(|| Error::Model("complete binary rank byte count overflow".into()))?;
    let after_dimensions = dimensions_offset
        .checked_add(dimensions_bytes)
        .ok_or_else(|| Error::Model("complete binary dimension offset overflow".into()))?;
    if after_dimensions > payload.len() {
        return Err(Error::Model(
            "complete binary is truncated in tensor dimensions".into(),
        ));
    }
    let mut shape = Vec::with_capacity(rank);
    let mut derived_elements = 1usize;
    for dimension in 0..rank {
        let value = read_u32(payload, dimensions_offset + dimension * 4)? as usize;
        if value == 0 {
            return Err(Error::Model(
                "complete binary tensor dimensions must be positive".into(),
            ));
        }
        derived_elements = derived_elements
            .checked_mul(value)
            .ok_or_else(|| Error::Model("complete binary tensor element count overflow".into()))?;
        shape.push(value);
    }
    let elements = usize::try_from(element_u64)
        .map_err(|_| Error::Model("complete binary element count exceeds this platform".into()))?;
    if elements != derived_elements {
        return Err(Error::Model(format!(
            "complete binary element count {elements} disagrees with shape product {derived_elements}"
        )));
    }
    let groups = elements
        .checked_add(group_size - 1)
        .ok_or_else(|| Error::Model("complete binary group count overflow".into()))?
        / group_size;
    let scale_bytes = groups
        .checked_mul(2)
        .ok_or_else(|| Error::Model("complete binary scale byte count overflow".into()))?;
    let sign_bytes = groups
        .checked_mul(group_size / 8)
        .ok_or_else(|| Error::Model("complete binary sign byte count overflow".into()))?;
    let scale_offset = after_dimensions;
    let sign_offset = scale_offset
        .checked_add(scale_bytes)
        .ok_or_else(|| Error::Model("complete binary sign offset overflow".into()))?;
    let payload_bytes = sign_offset
        .checked_add(sign_bytes)
        .ok_or_else(|| Error::Model("complete binary payload byte count overflow".into()))?;
    if payload_bytes != payload.len() {
        return Err(Error::Model(format!(
            "complete binary payload size {} does not equal fixed-layout expectation {payload_bytes}",
            payload.len()
        )));
    }
    for group in 0..groups {
        let scale = f16::from_bits(read_u16(payload, scale_offset + group * 2)?);
        if !scale.is_finite() {
            return Err(Error::Model(format!(
                "complete binary scale for group {group} is not finite"
            )));
        }
    }
    Ok(CompleteBinaryHeader {
        version,
        group_size,
        shape,
        elements,
        groups,
        scale_offset,
        sign_offset,
        payload_bytes,
    })
}

/// Reconstruct the stored tensor into f32 for native parity and diagnostic
/// paths.  Production direct device decode may replace this materialization
/// only after retaining the exact header and fixed-tail validation above.
pub fn decode_complete_binary_f32(payload: &[u8]) -> Result<(CompleteBinaryHeader, Vec<f32>)> {
    let header = parse_complete_binary_header(payload)?;
    let mut scales = Vec::with_capacity(header.groups);
    for group in 0..header.groups {
        scales.push(f16::from_bits(read_u16(payload, header.scale_offset + group * 2)?).to_f32());
    }
    let signs = &payload[header.sign_offset..header.payload_bytes];
    let mut values = Vec::with_capacity(header.elements);
    for element in 0..header.elements {
        let group = element / header.group_size;
        let bit = element % header.group_size;
        let byte = signs[group * (header.group_size / 8) + bit / 8];
        values.push(if (byte >> (bit % 8)) & 1 == 1 {
            scales[group]
        } else {
            -scales[group]
        });
    }
    Ok((header, values))
}

/// Parse the exact ``HQ30GR2`` layout used by the isolated Qwen30 quality
/// candidate.  It validates the embedded direct payload rather than trusting
/// an outer JSON declaration, and rejects unordered/out-of-range residual
/// indices or non-finite residual values.
pub fn parse_qwen30_quality_residual_header(payload: &[u8]) -> Result<Qwen30QualityResidualHeader> {
    const HEADER_BYTES: usize = 32;
    if payload.len() < HEADER_BYTES {
        return Err(Error::Model(
            "quality residual payload is shorter than its fixed header".into(),
        ));
    }
    if payload[..8] != QWEN30_QUALITY_REPACK_MAGIC {
        return Err(Error::Model(
            "quality residual magic does not match HQ30GR2".into(),
        ));
    }
    let version = read_u32(payload, 8)?;
    let base_version = read_u32(payload, 12)?;
    let group_size = read_u32(payload, 16)? as usize;
    let rank = read_u32(payload, 20)? as usize;
    let base_payload_bytes = read_u32(payload, 24)? as usize;
    let residual_count = read_u32(payload, 28)? as usize;
    if version != QWEN30_QUALITY_REPACK_VERSION
        || base_version != COMPLETE_BINARY_VERSION
        || group_size != 128
        || rank == 0
    {
        return Err(Error::Model(
            "quality residual header has unsupported version/group/rank".into(),
        ));
    }
    let dimensions_bytes = rank
        .checked_mul(4)
        .ok_or_else(|| Error::Model("quality residual rank byte count overflows".into()))?;
    let base_offset = HEADER_BYTES
        .checked_add(dimensions_bytes)
        .ok_or_else(|| Error::Model("quality residual base offset overflows".into()))?;
    let base_end = base_offset
        .checked_add(base_payload_bytes)
        .ok_or_else(|| Error::Model("quality residual base end overflows".into()))?;
    let index_bytes = residual_count
        .checked_mul(4)
        .ok_or_else(|| Error::Model("quality residual index byte count overflows".into()))?;
    let indices_end = base_end
        .checked_add(index_bytes)
        .ok_or_else(|| Error::Model("quality residual index end overflows".into()))?;
    let value_bytes = residual_count
        .checked_mul(2)
        .ok_or_else(|| Error::Model("quality residual value byte count overflows".into()))?;
    let values_end = indices_end
        .checked_add(value_bytes)
        .ok_or_else(|| Error::Model("quality residual value end overflows".into()))?;
    if values_end != payload.len() {
        return Err(Error::Model(
            "quality residual payload byte geometry does not match its header".into(),
        ));
    }
    let mut shape = Vec::with_capacity(rank);
    for index in 0..rank {
        let value = read_u32(payload, HEADER_BYTES + index * 4)? as usize;
        if value == 0 {
            return Err(Error::Model(
                "quality residual tensor dimensions must be positive".into(),
            ));
        }
        shape.push(value);
    }
    let base = parse_complete_binary_header(&payload[base_offset..base_end])?;
    if base.version != COMPLETE_BINARY_VERSION
        || base.group_size != group_size
        || base.shape != shape
    {
        return Err(Error::Model(
            "quality residual embedded direct payload does not match wrapper geometry".into(),
        ));
    }
    let mut previous = None;
    for index in 0..residual_count {
        let value = read_u32(payload, base_end + index * 4)? as usize;
        if value >= base.elements || previous.is_some_and(|prior| value <= prior) {
            return Err(Error::Model(
                "quality residual indices must be sorted, unique, and in tensor bounds".into(),
            ));
        }
        previous = Some(value);
        if !f16::from_bits(read_u16(payload, indices_end + index * 2)?).is_finite() {
            return Err(Error::Model(
                "quality residual values must be finite FP16".into(),
            ));
        }
    }
    Ok(Qwen30QualityResidualHeader {
        shape,
        base,
        residual_count,
        base_offset,
        base_payload_bytes,
        indices_offset: base_end,
        values_offset: indices_end,
        payload_bytes: values_end,
        indices_sha256: sha256_hex(&payload[base_end..indices_end]),
        values_sha256: sha256_hex(&payload[indices_end..values_end]),
    })
}

/// Return the sorted, exact FP16 residual corrections carried by one
/// ``HQ30GR2`` payload.  This is deliberately separate from a direct-layout
/// decoder: callers must opt into the residual grammar, so an HQ30GR2 organ
/// can never silently degrade into its embedded HQ30G1B1 control body.
pub fn qwen30_quality_residual_entries(
    payload: &[u8],
) -> Result<(Qwen30QualityResidualHeader, Vec<(usize, f32)>)> {
    let header = parse_qwen30_quality_residual_header(payload)?;
    let mut entries = Vec::with_capacity(header.residual_count);
    for ordinal in 0..header.residual_count {
        let index = read_u32(payload, header.indices_offset + ordinal * 4)? as usize;
        let value = f16::from_bits(read_u16(payload, header.values_offset + ordinal * 2)?).to_f32();
        // The parser above has already proved sortedness, bounds, and
        // finiteness.  Repeat the local guard so this public extraction stays
        // fail-closed even if the parser is later refactored.
        if index >= header.base.elements || !value.is_finite() {
            return Err(Error::Model(
                "quality residual entry differs from its validated HQ30GR2 header".into(),
            ));
        }
        entries.push((index, value));
    }
    Ok((header, entries))
}

/// Reconstruct an HQ30GR2 tensor for CPU-only scalar compatibility/parity
/// work.  It is not a full model loader or a device decode path.  The embedded
/// HQ30G1B1 body is decoded first and every sealed FP16 correction is added at
/// its exact flat index.
pub fn decode_qwen30_quality_residual_f32(
    payload: &[u8],
) -> Result<(Qwen30QualityResidualHeader, Vec<f32>)> {
    let (header, entries) = qwen30_quality_residual_entries(payload)?;
    let base_end = header
        .base_offset
        .checked_add(header.base_payload_bytes)
        .ok_or_else(|| Error::Model("quality residual embedded base end overflows".into()))?;
    let (_, mut values) = decode_complete_binary_f32(&payload[header.base_offset..base_end])?;
    if values.len() != header.base.elements {
        return Err(Error::Model(
            "quality residual decoded base element count differs from its header".into(),
        ));
    }
    for (index, correction) in entries {
        values[index] += correction;
    }
    Ok((header, values))
}

/// Apply a rank-2 HQ30G1B1 tensor directly to one CPU vector without
/// materializing decoded weights.  This is a scalar/reference operator for
/// compatibility testing, not a full Qwen runtime or performance claim.
pub fn complete_binary_matvec_f64(
    payload: &[u8],
    input: &[f64],
) -> Result<(CompleteBinaryHeader, Vec<f64>)> {
    let header = parse_complete_binary_header(payload)?;
    if header.shape.len() != 2 {
        return Err(Error::Model(
            "direct packed matvec requires a rank-2 tensor".into(),
        ));
    }
    let rows = header.shape[0];
    let columns = header.shape[1];
    if rows == 0 || columns == 0 || rows.checked_mul(columns) != Some(header.elements) {
        return Err(Error::Model(
            "direct packed matvec tensor geometry is invalid".into(),
        ));
    }
    if input.len() != columns || input.iter().any(|value| !value.is_finite()) {
        return Err(Error::Model(
            "direct packed matvec input differs from the finite tensor column count".into(),
        ));
    }
    let signs = &payload[header.sign_offset..header.payload_bytes];
    let bytes_per_group = header.group_size / 8;
    let mut output = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut sum = 0.0f64;
        let row_offset = row
            .checked_mul(columns)
            .ok_or_else(|| Error::Model("direct packed matvec row offset overflows".into()))?;
        for column in 0..columns {
            let index = row_offset
                .checked_add(column)
                .ok_or_else(|| Error::Model("direct packed matvec index overflows".into()))?;
            let group = index / header.group_size;
            let bit = index % header.group_size;
            let scale = f64::from(
                f16::from_bits(read_u16(payload, header.scale_offset + group * 2)?).to_f32(),
            );
            let sign_byte = signs[group * bytes_per_group + bit / 8];
            let weight = if (sign_byte >> (bit % 8)) & 1 == 1 {
                scale
            } else {
                -scale
            };
            sum += weight * input[column];
        }
        output.push(sum);
    }
    Ok((header, output))
}

/// Apply a rank-2 HQ30GR2 candidate tensor without converting it to a dense
/// weight matrix: execute the embedded direct HQ30G1B1 base, then accumulate
/// only the sealed sparse FP16 corrections.  The direct function intentionally
/// rejects the wrapper and this function intentionally rejects bare HQ30G1B1,
/// preventing an accidental quality-dropping fallback at adapter integration.
pub fn qwen30_quality_residual_matvec_f64(
    payload: &[u8],
    input: &[f64],
) -> Result<(Qwen30QualityResidualHeader, Vec<f64>)> {
    let (header, entries) = qwen30_quality_residual_entries(payload)?;
    if header.shape.len() != 2 {
        return Err(Error::Model(
            "HQ30GR2 packed matvec requires a rank-2 tensor".into(),
        ));
    }
    let rows = header.shape[0];
    let columns = header.shape[1];
    if rows == 0 || columns == 0 || rows.checked_mul(columns) != Some(header.base.elements) {
        return Err(Error::Model(
            "HQ30GR2 packed matvec tensor geometry is invalid".into(),
        ));
    }
    let base_end = header
        .base_offset
        .checked_add(header.base_payload_bytes)
        .ok_or_else(|| Error::Model("HQ30GR2 packed matvec embedded base end overflows".into()))?;
    let (base_header, mut output) =
        complete_binary_matvec_f64(&payload[header.base_offset..base_end], input)?;
    if base_header.shape != header.shape || output.len() != rows {
        return Err(Error::Model(
            "HQ30GR2 packed matvec embedded base geometry differs from wrapper".into(),
        ));
    }
    for (flat_index, correction) in entries {
        let row = flat_index / columns;
        let column = flat_index % columns;
        let destination = output.get_mut(row).ok_or_else(|| {
            Error::Model("HQ30GR2 packed matvec residual row is out of bounds".into())
        })?;
        *destination += f64::from(correction) * input[column];
    }
    Ok((header, output))
}

/// The two source families whose complete direct-binary manifests are admitted
/// by this module.  This is an artifact identity, not a decoder selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenCompleteBinaryModel {
    Qwen30Coder,
    Qwen80CoderNext,
}

impl QwenCompleteBinaryModel {
    pub const fn manifest_schema(self) -> &'static str {
        match self {
            Self::Qwen30Coder => QWEN30_COMPLETE_BINARY_SCHEMA,
            Self::Qwen80CoderNext => QWEN80_COMPLETE_BINARY_SCHEMA,
        }
    }

    pub const fn source_repository(self) -> &'static str {
        match self {
            Self::Qwen30Coder => "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            Self::Qwen80CoderNext => "Qwen/Qwen3-Coder-Next",
        }
    }

    const fn source_audit_schema(self) -> &'static str {
        match self {
            Self::Qwen30Coder => "hawking.ascension.qwen30_source_body_audit_candidate.v1",
            Self::Qwen80CoderNext => "hawking.ascension.qwen80_source_body_audit_candidate.v1",
        }
    }

    const fn source_audit_status(self) -> &'static str {
        match self {
            Self::Qwen30Coder => "CANDIDATE_SOURCE_BODY_VERIFIED",
            Self::Qwen80CoderNext => "CANDIDATE_FULL_PINNED_SOURCE_BODY_VERIFIED",
        }
    }
}

/// Immutable external bindings required before a manifest is admitted.
///
/// Receipt seals are integrity checks rather than a signing authority.  The
/// caller must therefore supply the expected complete-artifact and source-audit
/// seals from its protected campaign authority.  A self-consistent, newly
/// resealed substitute does not pass this admission interface.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompleteBinaryAdmission {
    pub model: QwenCompleteBinaryModel,
    pub expected_manifest_seal_sha256: String,
    pub expected_source_audit_seal_sha256: String,
    pub expected_source_revision: String,
}

/// One manifest tensor bound to a validated direct binary payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompleteBinaryTensor {
    pub tensor_name: String,
    pub source_shard: String,
    pub source_shard_sha256: String,
    pub source_dtype: String,
    pub artifact_path: PathBuf,
    pub artifact_sha256: String,
    pub header: CompleteBinaryHeader,
}

/// A fully admitted all-tensor artifact catalog.
///
/// Admission retains an immutable in-process snapshot of every verified
/// direct payload. [`Self::verified_tensor_payload`] is therefore the only
/// appropriate production direct-runtime path: it returns the exact byte
/// snapshot scanned at process admission. [`Self::read_tensor_payload`]
/// deliberately re-reads and hashes a file for a targeted tamper diagnostic,
/// and must not sit on a native token path.
///
/// The catalog itself does not implement a Qwen layer, token loop, HCLI
/// endpoint, or TPS claim.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompleteBinaryArtifact {
    pub model: QwenCompleteBinaryModel,
    pub manifest_path: PathBuf,
    pub manifest_seal_sha256: String,
    pub source_audit_path: PathBuf,
    pub source_audit_seal_sha256: String,
    pub source_revision: String,
    pub source_index_path: PathBuf,
    pub source_weight_elements: u64,
    pub tensor_payload_bytes: u64,
    pub tensors: BTreeMap<String, CompleteBinaryTensor>,
    /// Admission-verified immutable direct-payload snapshots. This is
    /// crate-visible only so in-crate already-admitted fixtures can model
    /// their intentionally absent disk payloads; production consumers must
    /// use [`Self::verified_tensor_payload`].
    pub(crate) verified_payloads: BTreeMap<String, Arc<[u8]>>,
}

impl CompleteBinaryArtifact {
    /// Return a catalogued tensor or fail rather than guessing a tensor name.
    pub fn tensor(&self, tensor_name: &str) -> Result<&CompleteBinaryTensor> {
        self.tensors.get(tensor_name).ok_or_else(|| {
            Error::Model(format!(
                "complete binary artifact has no admitted tensor {tensor_name:?}"
            ))
        })
    }

    /// Return a cheap clone of the immutable payload snapshot that passed the
    /// exact payload SHA-256 and fixed-layout checks during this process's
    /// full artifact admission. It never reopens an artifact file, so a file
    /// replacement after admission cannot alter a native token graph. A new
    /// process must admit the artifact again and obtains fresh snapshots only
    /// after every payload is revalidated.
    pub fn verified_tensor_payload(&self, tensor_name: &str) -> Result<Arc<[u8]>> {
        self.tensor(tensor_name)?;
        self.verified_payloads
            .get(tensor_name)
            .cloned()
            .ok_or_else(|| {
                Error::Model(format!(
                    "complete binary artifact has no admission-verified immutable payload for {tensor_name:?}"
                ))
            })
    }

    /// Count exact immutable payload snapshots retained by this process.
    pub fn verified_payload_count(&self) -> usize {
        self.verified_payloads.len()
    }

    /// Whether the exact whole-artifact catalog has one immutable snapshot
    /// for every sealed tensor. This is integrity/residency evidence only,
    /// never a performance measurement.
    pub fn has_complete_verified_payload_cache(&self) -> bool {
        self.verified_payloads.len() == self.tensors.len()
            && self
                .tensors
                .keys()
                .all(|name| self.verified_payloads.contains_key(name))
    }

    /// Re-read one payload with the same hash and fixed-layout checks used at
    /// admission. This is a targeted tamper/restart diagnostic surface; a
    /// native token runtime must use [`Self::verified_tensor_payload`] rather
    /// than paying a SHA-256 scan after process admission.
    pub fn read_tensor_payload(&self, tensor_name: &str) -> Result<Vec<u8>> {
        let tensor = self.tensor(tensor_name)?;
        let payload = read_regular_file(&tensor.artifact_path, "complete binary tensor payload")?;
        let observed = sha256_hex(&payload);
        if observed != tensor.artifact_sha256 {
            return Err(Error::Model(format!(
                "complete binary tensor {tensor_name} SHA-256 mismatch: observed={observed} expected={}",
                tensor.artifact_sha256
            )));
        }
        let header = parse_complete_binary_header(&payload)?;
        if header != tensor.header {
            return Err(Error::Model(format!(
                "complete binary tensor {tensor_name} header changed after admission"
            )));
        }
        Ok(payload)
    }
}

/// Protected bindings for the separate Qwen30 gate/up residual experiment.
/// Every value is supplied by the candidate-local sealed handoff request; the
/// native reader compares it with the on-disk authority before scanning a
/// payload.  This prevents a self-consistent replacement selection/terminal
/// document from becoming an implicit admission authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen30QualityRepackAdmission {
    pub expected_manifest_seal_sha256: String,
    pub expected_source_audit_seal_sha256: String,
    pub expected_source_revision: String,
    pub expected_revalidation_path: PathBuf,
    pub expected_revalidation_seal_sha256: String,
    pub expected_selection_path: PathBuf,
    pub expected_selection_seal_sha256: String,
    pub expected_source_snapshot_path: PathBuf,
    pub expected_source_snapshot_seal_sha256: String,
    pub expected_terminal_path: PathBuf,
    pub expected_terminal_seal_sha256: String,
}

/// Native catalog entry for either the normal direct payload or one of the
/// two explicitly selected sparse-residual payloads.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Qwen30QualityRepackTensorLayout {
    Direct(CompleteBinaryHeader),
    SparseResidual(Qwen30QualityResidualHeader),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen30QualityRepackTensor {
    pub tensor_name: String,
    pub source_shard: String,
    pub source_shard_sha256: String,
    pub source_dtype: String,
    pub artifact_path: PathBuf,
    pub artifact_sha256: String,
    pub artifact_bytes: usize,
    pub elements: usize,
    pub layout: Qwen30QualityRepackTensorLayout,
}

/// One immutable candidate payload paired with its exact validated grammar.
///
/// This is a construction-time boundary for a future typed HQ30GR2 runtime.
/// It prevents selected sparse-residual organs from reaching a direct-only
/// Metal loader and prevents a direct control organ from entering a residual
/// decoder.  The enum itself does not execute a Qwen layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Qwen30QualityRepackVerifiedTensor {
    Direct {
        header: CompleteBinaryHeader,
        payload: Arc<[u8]>,
    },
    SparseResidual {
        header: Qwen30QualityResidualHeader,
        payload: Arc<[u8]>,
    },
}

/// Fully scanned catalog for the isolated candidate.  As with
/// [`CompleteBinaryArtifact`], this is admission evidence only; it deliberately
/// does not expose a Qwen execution graph, runtime, HCLI surface, or TPS claim.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen30QualityRepackArtifact {
    pub manifest_path: PathBuf,
    pub manifest_seal_sha256: String,
    pub source_audit_path: PathBuf,
    pub source_audit_seal_sha256: String,
    pub source_revision: String,
    pub source_index_path: PathBuf,
    pub source_weight_elements: u64,
    pub tensor_payload_bytes: u64,
    pub selected_residual_organs: Vec<String>,
    /// Number of independent read-only payload lanes used during this strict
    /// scan.  The coordinator still reconciles every result by manifest
    /// ordinal before constructing this catalog.
    pub payload_verification_workers: usize,
    pub tensors: BTreeMap<String, Qwen30QualityRepackTensor>,
    /// Admission-verified immutable snapshots for every candidate payload.
    ///
    /// This mirrors the direct-artifact safety boundary: a future typed
    /// HQ30GR2 runtime must consume this process-local snapshot rather than
    /// reopen a candidate file, hash it on a token path, or silently swap a
    /// selected sparse-residual organ for a direct payload.  The candidate
    /// catalog remains admission-only until a separately typed runtime opts
    /// into its `Direct`/`SparseResidual` layouts.
    pub(crate) verified_payloads: BTreeMap<String, Arc<[u8]>>,
}

impl Qwen30QualityRepackArtifact {
    /// Return one catalogued candidate tensor or fail rather than guessing a
    /// tensor name.  The layout enum is intentionally retained for a future
    /// typed reader; callers must not assume every candidate organ is direct.
    pub fn tensor(&self, tensor_name: &str) -> Result<&Qwen30QualityRepackTensor> {
        self.tensors.get(tensor_name).ok_or_else(|| {
            Error::Model(format!(
                "quality repack artifact has no admitted tensor {tensor_name:?}"
            ))
        })
    }

    /// Return the exact immutable payload snapshot scanned by the candidate's
    /// whole-artifact admission.  It never reopens a payload file.  A future
    /// all-layer candidate runtime must use this accessor and dispatch by the
    /// paired [`Qwen30QualityRepackTensorLayout`] rather than falling back to
    /// the ordinary direct reader for an HQ30GR2 organ.
    pub fn verified_tensor_payload(&self, tensor_name: &str) -> Result<Arc<[u8]>> {
        self.tensor(tensor_name)?;
        self.verified_payloads
            .get(tensor_name)
            .cloned()
            .ok_or_else(|| {
                Error::Model(format!(
                    "quality repack artifact has no admission-verified immutable payload for {tensor_name:?}"
                ))
            })
    }

    /// Return a verified snapshot together with the only legal reader grammar
    /// for that catalog entry.  A future runtime should resolve this exactly
    /// once while constructing device-resident tensors, then retain the typed
    /// result; it must not use this parsing/accessor as a per-token file path.
    pub fn verified_typed_tensor(
        &self,
        tensor_name: &str,
    ) -> Result<Qwen30QualityRepackVerifiedTensor> {
        let tensor = self.tensor(tensor_name)?;
        let payload = self.verified_tensor_payload(tensor_name)?;
        match &tensor.layout {
            Qwen30QualityRepackTensorLayout::Direct(expected) => {
                let observed = parse_complete_binary_header(payload.as_ref())?;
                if &observed != expected {
                    return Err(Error::Model(format!(
                        "quality repack direct tensor {tensor_name:?} immutable snapshot header differs from admission"
                    )));
                }
                Ok(Qwen30QualityRepackVerifiedTensor::Direct {
                    header: observed,
                    payload,
                })
            }
            Qwen30QualityRepackTensorLayout::SparseResidual(expected) => {
                let observed = parse_qwen30_quality_residual_header(payload.as_ref())?;
                if &observed != expected {
                    return Err(Error::Model(format!(
                        "quality repack sparse-residual tensor {tensor_name:?} immutable snapshot header differs from admission"
                    )));
                }
                Ok(Qwen30QualityRepackVerifiedTensor::SparseResidual {
                    header: observed,
                    payload,
                })
            }
        }
    }

    /// Count immutable payload snapshots retained by this candidate admission.
    pub fn verified_payload_count(&self) -> usize {
        self.verified_payloads.len()
    }

    /// Whether every candidate catalog row has exactly one immutable snapshot
    /// from the same admission scan.  This is an integrity/residency property,
    /// never a runtime or throughput claim.
    pub fn has_complete_verified_payload_cache(&self) -> bool {
        self.verified_payloads.len() == self.tensors.len()
            && self
                .tensors
                .keys()
                .all(|name| self.verified_payloads.contains_key(name))
    }
}

#[derive(Debug)]
struct NoDuplicateJson(Value);

impl<'de> Deserialize<'de> for NoDuplicateJson {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(NoDuplicateJsonVisitor)
    }
}

struct NoDuplicateJsonVisitor;

impl<'de> Visitor<'de> for NoDuplicateJsonVisitor {
    type Value = NoDuplicateJson;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(NoDuplicateJson(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(NoDuplicateJson(Value::Number(value.into())))
    }

    fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(NoDuplicateJson(Value::Number(value.into())))
    }

    fn visit_f64<E>(self, value: f64) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        serde_json::Number::from_f64(value)
            .map(|number| NoDuplicateJson(Value::Number(number)))
            .ok_or_else(|| E::custom("JSON number must be finite"))
    }

    fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(NoDuplicateJson(Value::String(value.to_owned())))
    }

    fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(NoDuplicateJson(Value::String(value)))
    }

    fn visit_none<E>(self) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(NoDuplicateJson(Value::Null))
    }

    fn visit_unit<E>(self) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(NoDuplicateJson(Value::Null))
    }

    fn visit_some<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        NoDuplicateJson::deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(NoDuplicateJson(value)) = sequence.next_element()? {
            values.push(value);
        }
        Ok(NoDuplicateJson(Value::Array(values)))
    }

    fn visit_map<A>(self, mut map_access: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        while let Some(key) = map_access.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(de::Error::custom(format!("duplicate JSON key {key:?}")));
            }
            let NoDuplicateJson(value) = map_access.next_value()?;
            values.insert(key, value);
        }
        Ok(NoDuplicateJson(Value::Object(values)))
    }
}

fn model_error(label: &str, detail: impl Into<String>) -> Error {
    Error::Model(format!("{label}: {}", detail.into()))
}

fn read_regular_file(path: &Path, label: &str) -> Result<Vec<u8>> {
    let before = fs::symlink_metadata(path)
        .map_err(|error| model_error(label, format!("cannot stat {}: {error}", path.display())))?;
    if before.file_type().is_symlink() || !before.is_file() {
        return Err(model_error(
            label,
            format!("{} must be a regular non-symlink file", path.display()),
        ));
    }
    let bytes = fs::read(path)
        .map_err(|error| model_error(label, format!("cannot read {}: {error}", path.display())))?;
    let after = fs::symlink_metadata(path).map_err(|error| {
        model_error(
            label,
            format!("cannot re-stat {} after read: {error}", path.display()),
        )
    })?;
    if after.file_type().is_symlink() || !after.is_file() || before.len() != after.len() {
        return Err(model_error(
            label,
            format!("{} changed while being read", path.display()),
        ));
    }
    if u64::try_from(bytes.len()).ok() != Some(before.len()) {
        return Err(model_error(
            label,
            format!("{} byte count changed while being read", path.display()),
        ));
    }
    Ok(bytes)
}

fn canonical_regular_path(path: &Path, label: &str) -> Result<PathBuf> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| model_error(label, format!("cannot stat {}: {error}", path.display())))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(model_error(
            label,
            format!("{} must be a regular non-symlink file", path.display()),
        ));
    }
    fs::canonicalize(path).map_err(|error| {
        model_error(
            label,
            format!("cannot canonicalize {}: {error}", path.display()),
        )
    })
}

fn canonical_directory(path: &Path, label: &str) -> Result<PathBuf> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| model_error(label, format!("cannot stat {}: {error}", path.display())))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(model_error(
            label,
            format!("{} must be a non-symlink directory", path.display()),
        ));
    }
    fs::canonicalize(path).map_err(|error| {
        model_error(
            label,
            format!("cannot canonicalize {}: {error}", path.display()),
        )
    })
}

fn parse_json_no_duplicate_keys(raw: &[u8], label: &str) -> Result<Value> {
    let mut deserializer = serde_json::Deserializer::from_slice(raw);
    let NoDuplicateJson(value) = NoDuplicateJson::deserialize(&mut deserializer)
        .map_err(|error| model_error(label, format!("invalid JSON: {error}")))?;
    deserializer
        .end()
        .map_err(|error| model_error(label, format!("trailing JSON data: {error}")))?;
    Ok(value)
}

/// Render a JSON floating number exactly as CPython's ``json.dumps`` does for
/// finite ``float`` values: shortest round-trip digits, fixed notation for
/// decimal exponents -4 through 15, and a signed, at-least-two-digit exponent
/// elsewhere.  The physical builders seal with that Python canonicalization,
/// so relying on serde_json's otherwise-valid (but differently styled) Ryu
/// exponent spelling would reject a legitimate sealed manifest such as 1e-06.
fn python_json_float(number: &serde_json::Number) -> Result<String> {
    let value = number
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| model_error("canonical JSON", "floating number must be finite"))?;
    if value == 0.0 {
        return Ok(if value.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        });
    }

    let raw = number.to_string();
    let (negative, unsigned) = match raw.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, raw.as_str()),
    };
    let (mantissa, exponent) = match unsigned.find('e').or_else(|| unsigned.find('E')) {
        Some(index) => {
            let exponent = unsigned[index + 1..].parse::<i32>().map_err(|error| {
                model_error(
                    "canonical JSON",
                    format!("invalid finite JSON exponent {raw:?}: {error}"),
                )
            })?;
            (&unsigned[..index], exponent)
        }
        None => (unsigned, 0),
    };
    let mut fractional_digits = 0i32;
    let mut after_decimal = false;
    let mut digits = String::new();
    for byte in mantissa.bytes() {
        match byte {
            b'.' if !after_decimal => after_decimal = true,
            b'0'..=b'9' => {
                if after_decimal {
                    fractional_digits = fractional_digits.checked_add(1).ok_or_else(|| {
                        model_error("canonical JSON", "fractional digit count overflows i32")
                    })?;
                }
                digits.push(char::from(byte));
            }
            _ => {
                return Err(model_error(
                    "canonical JSON",
                    format!("invalid finite JSON mantissa {raw:?}"),
                ))
            }
        }
    }
    let first_significant = digits
        .bytes()
        .position(|byte| byte != b'0')
        .ok_or_else(|| model_error("canonical JSON", "nonzero float had no significant digit"))?;
    let mut significant = digits[first_significant..].to_owned();
    let mut decimal_power = exponent
        .checked_sub(fractional_digits)
        .ok_or_else(|| model_error("canonical JSON", "floating decimal exponent overflows i32"))?;
    while significant.len() > 1 && significant.ends_with('0') {
        significant.pop();
        decimal_power = decimal_power.checked_add(1).ok_or_else(|| {
            model_error("canonical JSON", "floating decimal exponent overflows i32")
        })?;
    }
    let scientific_exponent = decimal_power
        .checked_add(i32::try_from(significant.len() - 1).unwrap_or(i32::MAX))
        .ok_or_else(|| model_error("canonical JSON", "floating exponent overflows i32"))?;
    let sign = if negative { "-" } else { "" };
    if !(-4..16).contains(&scientific_exponent) {
        let mut mantissa = significant[..1].to_owned();
        if significant.len() > 1 {
            mantissa.push('.');
            mantissa.push_str(&significant[1..]);
        }
        let exponent_sign = if scientific_exponent < 0 { '-' } else { '+' };
        return Ok(format!(
            "{sign}{mantissa}e{exponent_sign}{:02}",
            scientific_exponent.unsigned_abs()
        ));
    }

    let decimal_position = scientific_exponent + 1;
    let rendered = if decimal_position <= 0 {
        format!(
            "0.{}{}",
            "0".repeat(usize::try_from(-decimal_position).unwrap_or(usize::MAX)),
            significant
        )
    } else if usize::try_from(decimal_position).unwrap_or(usize::MAX) >= significant.len() {
        format!(
            "{}{}.0",
            significant,
            "0".repeat(usize::try_from(decimal_position).unwrap_or(usize::MAX) - significant.len())
        )
    } else {
        let position = usize::try_from(decimal_position).unwrap();
        format!("{}.{}", &significant[..position], &significant[position..])
    };
    Ok(format!("{sign}{rendered}"))
}

fn canonical_json_number(number: &serde_json::Number) -> Result<String> {
    if number.is_i64() || number.is_u64() {
        Ok(number.to_string())
    } else {
        python_json_float(number)
    }
}

fn canonical_json_into(value: &Value, output: &mut String) -> Result<()> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => output.push_str(&canonical_json_number(value)?),
        Value::String(value) => output.push_str(
            &serde_json::to_string(value)
                .map_err(|error| model_error("canonical JSON", error.to_string()))?,
        ),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                canonical_json_into(value, output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            let mut keys: Vec<&String> = values.keys().collect();
            keys.sort_unstable();
            output.push('{');
            for (index, key) in keys.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                output.push_str(
                    &serde_json::to_string(*key)
                        .map_err(|error| model_error("canonical JSON", error.to_string()))?,
                );
                output.push(':');
                canonical_json_into(&values[*key], output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn canonical_json(value: &Value) -> Result<Vec<u8>> {
    let mut output = String::new();
    canonical_json_into(value, &mut output)?;
    Ok(output.into_bytes())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn required_object<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<&'a Map<String, Value>> {
    object
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| model_error(label, format!("missing object field {key:?}")))
}

fn required_array<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<&'a [Value]> {
    object
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| model_error(label, format!("missing array field {key:?}")))
}

fn required_string<'a>(object: &'a Map<String, Value>, key: &str, label: &str) -> Result<&'a str> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| model_error(label, format!("missing non-empty string field {key:?}")))
}

fn required_sha256(object: &Map<String, Value>, key: &str, label: &str) -> Result<String> {
    let value = required_string(object, key, label)?;
    if !is_sha256(value) {
        return Err(model_error(
            label,
            format!("field {key:?} must be a lowercase 64-character SHA-256"),
        ));
    }
    Ok(value.to_owned())
}

fn required_u64(object: &Map<String, Value>, key: &str, label: &str) -> Result<u64> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| model_error(label, format!("missing unsigned integer field {key:?}")))
}

fn required_bool(object: &Map<String, Value>, key: &str, label: &str) -> Result<bool> {
    object
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| model_error(label, format!("missing boolean field {key:?}")))
}

fn required_f64(object: &Map<String, Value>, key: &str, label: &str) -> Result<f64> {
    object
        .get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| model_error(label, format!("missing finite numeric field {key:?}")))
}

fn verify_sealed_document(document: &Value, label: &str) -> Result<String> {
    let object = document
        .as_object()
        .ok_or_else(|| model_error(label, "root must be a JSON object"))?;
    let recorded = required_sha256(object, "seal_sha256", label)?;
    let mut unsigned = object.clone();
    unsigned.remove("seal_sha256");
    let expected = sha256_hex(&canonical_json(&Value::Object(unsigned))?);
    if recorded != expected {
        return Err(model_error(
            label,
            format!("seal_sha256 mismatch: recorded={recorded} expected={expected}"),
        ));
    }
    Ok(recorded)
}

fn require_exact_string(
    object: &Map<String, Value>,
    key: &str,
    expected: &str,
    label: &str,
) -> Result<()> {
    let observed = required_string(object, key, label)?;
    if observed != expected {
        return Err(model_error(
            label,
            format!("field {key:?} is {observed:?}; expected {expected:?}"),
        ));
    }
    Ok(())
}

fn require_safe_filename(value: &str, label: &str) -> Result<()> {
    if value.contains('\0') {
        return Err(model_error(label, "path contains a NUL byte"));
    }
    let mut components = Path::new(value).components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(_)), None) => Ok(()),
        _ => Err(model_error(
            label,
            format!("path {value:?} must be one ordinary filename"),
        )),
    }
}

fn absolute_path(value: &str, label: &str) -> Result<PathBuf> {
    if value.contains('\0') {
        return Err(model_error(label, "path contains a NUL byte"));
    }
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err(model_error(
            label,
            format!("path {value:?} must be absolute"),
        ));
    }
    Ok(path)
}

fn manifest_descendant_file(root: &Path, value: &str, label: &str) -> Result<PathBuf> {
    if value.contains('\0') {
        return Err(model_error(label, "path contains a NUL byte"));
    }
    let declared = PathBuf::from(value);
    let path = if declared.is_absolute() {
        declared
    } else {
        root.join(declared)
    };
    let metadata = fs::symlink_metadata(&path)
        .map_err(|error| model_error(label, format!("cannot stat {}: {error}", path.display())))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(model_error(
            label,
            format!("{} must be a regular non-symlink file", path.display()),
        ));
    }
    let canonical = fs::canonicalize(&path).map_err(|error| {
        model_error(
            label,
            format!("cannot canonicalize {}: {error}", path.display()),
        )
    })?;
    if !canonical.starts_with(root) {
        return Err(model_error(
            label,
            format!(
                "{} must remain under {}",
                canonical.display(),
                root.display()
            ),
        ));
    }
    Ok(canonical)
}

fn manifest_child_file(root: &Path, value: &str, label: &str) -> Result<PathBuf> {
    let canonical = manifest_descendant_file(root, value, label)?;
    if canonical.parent() != Some(root) {
        return Err(model_error(
            label,
            format!(
                "{} must be a direct child of {}",
                canonical.display(),
                root.display()
            ),
        ));
    }
    Ok(canonical)
}

fn current_file_identity_matches(
    path: &Path,
    expected: &Map<String, Value>,
    label: &str,
) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| model_error(label, format!("cannot stat {}: {error}", path.display())))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(model_error(
            label,
            format!("{} must be a regular non-symlink file", path.display()),
        ));
    }
    if required_u64(expected, "bytes", label)? != metadata.len() {
        return Err(model_error(
            label,
            format!(
                "{} byte identity no longer matches revalidation receipt",
                path.display()
            ),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        fn required_i128(object: &Map<String, Value>, key: &str, label: &str) -> Result<i128> {
            object
                .get(key)
                .and_then(|value| {
                    value
                        .as_i64()
                        .map(i128::from)
                        .or_else(|| value.as_u64().map(i128::from))
                })
                .ok_or_else(|| model_error(label, format!("missing signed integer field {key:?}")))
        }

        // st_dev is a mount-time artifact: a reboot/remount reassigns the APFS
        // volume device number without touching the file, so it is not file
        // identity. Content integrity is sealed independently in shard_hashes;
        // (bytes + inode) already pin the file on a single volume. Checking
        // device here made admission hard-refuse on a benign remount (observed
        // 16777234 -> 16777233, inode + bytes byte-identical). Dropped from the
        // hard-fail set; the receipt may still carry a "device" field, unread.
        let observed = [
            ("inode", i128::from(metadata.ino())),
            (
                "mtime_ns",
                i128::from(metadata.mtime()) * 1_000_000_000i128
                    + i128::from(metadata.mtime_nsec()),
            ),
            (
                "ctime_ns",
                i128::from(metadata.ctime()) * 1_000_000_000i128
                    + i128::from(metadata.ctime_nsec()),
            ),
        ];
        for (key, actual) in observed {
            if required_i128(expected, key, label)? != actual {
                return Err(model_error(
                    label,
                    format!(
                        "{} {key} identity no longer matches revalidation receipt",
                        path.display()
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn weight_map_from_index(index: &Value, label: &str) -> Result<BTreeMap<String, String>> {
    let object = index
        .as_object()
        .ok_or_else(|| model_error(label, "index root must be an object"))?;
    let weights = required_object(object, "weight_map", label)?;
    if weights.is_empty() {
        return Err(model_error(label, "weight_map must not be empty"));
    }
    let mut out = BTreeMap::new();
    for (name, shard) in weights {
        if name.is_empty() || name.contains('\0') {
            return Err(model_error(label, "weight_map has an invalid tensor name"));
        }
        let shard = shard.as_str().ok_or_else(|| {
            model_error(
                label,
                format!("weight_map tensor {name:?} does not name a shard"),
            )
        })?;
        require_safe_filename(shard, label)?;
        out.insert(name.clone(), shard.to_owned());
    }
    Ok(out)
}

fn canonical_string_map_sha256(values: &BTreeMap<String, String>) -> Result<String> {
    let object = values
        .iter()
        .map(|(key, value)| (key.clone(), Value::String(value.clone())))
        .collect();
    Ok(sha256_hex(&canonical_json(&Value::Object(object))?))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SourceChain {
    source_audit_path: PathBuf,
    source_audit_seal_sha256: String,
    source_revision: String,
    source_index_path: PathBuf,
    weight_map: BTreeMap<String, String>,
    shard_hashes: BTreeMap<String, String>,
}

fn validate_source_chain(
    receipt: &Value,
    receipt_path: &Path,
    root: &Path,
    admission: &CompleteBinaryAdmission,
    manifest_audit_seal: &str,
) -> Result<SourceChain> {
    validate_source_chain_at(receipt, receipt_path, root, admission, manifest_audit_seal)
}

/// Validate a current source receipt whose protected authority may live under
/// a different immutable control root.  Ordinary direct candidates pass their
/// own manifest root; the isolated quality-repack candidate is deliberately
/// constrained to the admitted baseline's revalidation parent instead.
fn validate_source_chain_at(
    receipt: &Value,
    receipt_path: &Path,
    expected_revalidation_parent: &Path,
    admission: &CompleteBinaryAdmission,
    manifest_audit_seal: &str,
) -> Result<SourceChain> {
    let label = "complete binary source revalidation receipt";
    let receipt_object = receipt
        .as_object()
        .ok_or_else(|| model_error(label, "root must be an object"))?;
    verify_sealed_document(receipt, label)?;
    require_exact_string(
        receipt_object,
        "schema",
        COMPLETE_BINARY_REVALIDATION_SCHEMA,
        label,
    )?;
    require_exact_string(
        receipt_object,
        "status",
        "EARNED_CURRENT_SOURCE_SHARDS_REVALIDATED",
        label,
    )?;
    require_exact_string(
        receipt_object,
        "source_repository",
        admission.model.source_repository(),
        label,
    )?;
    require_exact_string(
        receipt_object,
        "source_revision",
        &admission.expected_source_revision,
        label,
    )?;
    let receipt_audit_seal = required_sha256(receipt_object, "source_audit_seal_sha256", label)?;
    if receipt_audit_seal != manifest_audit_seal
        || receipt_audit_seal != admission.expected_source_audit_seal_sha256
    {
        return Err(model_error(
            label,
            "source audit seal does not match the manifest's protected admission binding",
        ));
    }

    let receipt_declared = required_string(receipt_object, "source_audit_path", label)?;
    let audit_path =
        canonical_regular_path(&absolute_path(receipt_declared, label)?, "source audit")?;
    let (audit_raw, audit) = {
        let raw = read_regular_file(&audit_path, "source audit")?;
        let document = parse_json_no_duplicate_keys(&raw, "source audit")?;
        (raw, document)
    };
    let observed_audit_document_sha = sha256_hex(&audit_raw);
    if observed_audit_document_sha
        != required_sha256(receipt_object, "source_audit_document_sha256", label)?
    {
        return Err(model_error(
            label,
            "source audit raw-document SHA-256 does not match its receipt binding",
        ));
    }
    let audit_seal = verify_sealed_document(&audit, "source audit")?;
    if audit_seal != receipt_audit_seal {
        return Err(model_error(
            label,
            "source audit seal does not match the revalidation receipt",
        ));
    }
    let audit_object = audit
        .as_object()
        .ok_or_else(|| model_error("source audit", "root must be an object"))?;
    require_exact_string(
        audit_object,
        "schema",
        admission.model.source_audit_schema(),
        "source audit",
    )?;
    require_exact_string(
        audit_object,
        "status",
        admission.model.source_audit_status(),
        "source audit",
    )?;
    let audit_source = required_object(audit_object, "source", "source audit")?;
    require_exact_string(
        audit_source,
        "repository",
        admission.model.source_repository(),
        "source audit",
    )?;
    require_exact_string(
        audit_source,
        "revision",
        &admission.expected_source_revision,
        "source audit",
    )?;

    let index_declared = required_string(receipt_object, "index_path", label)?;
    let source_index_path =
        canonical_regular_path(&absolute_path(index_declared, label)?, "source index")?;
    let index_raw = read_regular_file(&source_index_path, "source index")?;
    if sha256_hex(&index_raw) != required_sha256(receipt_object, "index_sha256", label)? {
        return Err(model_error(
            label,
            "current source index SHA-256 does not match its revalidation receipt",
        ));
    }
    let index = parse_json_no_duplicate_keys(&index_raw, "source index")?;
    let weight_map = weight_map_from_index(&index, "source index")?;
    if canonical_string_map_sha256(&weight_map)?
        != required_sha256(receipt_object, "weight_map_sha256", label)?
    {
        return Err(model_error(
            label,
            "current source weight map does not match its revalidation receipt",
        ));
    }

    let source_model_dir = source_index_path.parent().ok_or_else(|| {
        model_error(
            label,
            "source index has no parent directory for source shard binding",
        )
    })?;
    let declared_model_dir = canonical_directory(
        &absolute_path(
            required_string(receipt_object, "source_model_dir", label)?,
            label,
        )?,
        label,
    )?;
    if declared_model_dir != source_model_dir {
        return Err(model_error(
            label,
            "source_model_dir does not equal the current source-index directory",
        ));
    }

    let shards = required_object(receipt_object, "shards", label)?;
    let expected_shards: BTreeSet<String> = weight_map.values().cloned().collect();
    if shards.len() != expected_shards.len()
        || !expected_shards
            .iter()
            .all(|shard| shards.contains_key(shard))
    {
        return Err(model_error(
            label,
            "revalidation receipt shard set does not exactly match the current source index",
        ));
    }
    if required_u64(receipt_object, "sealed_shard_count", label)?
        != u64::try_from(expected_shards.len()).unwrap_or(u64::MAX)
    {
        return Err(model_error(
            label,
            "sealed_shard_count does not match the current source index",
        ));
    }
    let mut shard_hashes = BTreeMap::new();
    for shard in &expected_shards {
        let row = shards
            .get(shard)
            .and_then(Value::as_object)
            .ok_or_else(|| {
                model_error(
                    label,
                    format!("missing object receipt for source shard {shard}"),
                )
            })?;
        let expected_hash = required_sha256(row, "expected_sha256", label)?;
        if required_sha256(row, "observed_sha256", label)? != expected_hash {
            return Err(model_error(
                label,
                format!("source shard {shard} observed SHA-256 differs from expected SHA-256"),
            ));
        }
        let expected_bytes = required_u64(row, "expected_bytes", label)?;
        let identity = required_object(row, "file_identity", label)?;
        if required_u64(identity, "bytes", label)? != expected_bytes {
            return Err(model_error(
                label,
                format!("source shard {shard} byte evidence differs from its file identity"),
            ));
        }
        let source_shard_path = source_model_dir.join(shard);
        current_file_identity_matches(&source_shard_path, identity, label)?;
        shard_hashes.insert(shard.clone(), expected_hash);
    }
    if canonical_string_map_sha256(&shard_hashes)?
        != required_sha256(receipt_object, "sealed_shard_hashes_sha256", label)?
    {
        return Err(model_error(
            label,
            "sealed source-shard hash map does not match the revalidation receipt",
        ));
    }

    if receipt_path.parent() != Some(expected_revalidation_parent) {
        return Err(model_error(
            label,
            "revalidation receipt must be a direct child of its protected authority root",
        ));
    }
    Ok(SourceChain {
        source_audit_path: audit_path,
        source_audit_seal_sha256: audit_seal,
        source_revision: admission.expected_source_revision.clone(),
        source_index_path,
        weight_map,
        shard_hashes,
    })
}

fn validate_manifest_source(
    manifest: &Map<String, Value>,
    source: &SourceChain,
    model: QwenCompleteBinaryModel,
) -> Result<()> {
    let manifest_source = required_object(manifest, "source", "complete binary manifest")?;
    require_exact_string(
        manifest_source,
        "repository",
        model.source_repository(),
        "complete binary manifest source",
    )?;
    if required_u64(
        manifest_source,
        "tensor_count",
        "complete binary manifest source",
    )? != u64::try_from(source.weight_map.len()).unwrap_or(u64::MAX)
    {
        return Err(model_error(
            "complete binary manifest source",
            "tensor_count does not match the sealed current source index",
        ));
    }
    let declared_dir = canonical_directory(
        &absolute_path(
            required_string(
                manifest_source,
                "model_dir",
                "complete binary manifest source",
            )?,
            "complete binary manifest source",
        )?,
        "complete binary manifest source",
    )?;
    let actual_dir = source.source_index_path.parent().ok_or_else(|| {
        model_error(
            "complete binary manifest source",
            "source index has no parent model directory",
        )
    })?;
    if declared_dir != actual_dir {
        return Err(model_error(
            "complete binary manifest source",
            "model_dir does not match the revalidated source index directory",
        ));
    }
    let representation = required_object(manifest, "representation", "complete binary manifest")?;
    require_exact_string(
        representation,
        "family",
        "binary_sign_scale",
        "complete binary manifest representation",
    )?;
    if required_u64(
        representation,
        "group_size",
        "complete binary manifest representation",
    )? != u64::try_from(128usize).unwrap()
        || !required_bool(
            representation,
            "physical_direct_layout",
            "complete binary manifest representation",
        )?
    {
        return Err(model_error(
            "complete binary manifest representation",
            "requires direct binary sign+FP16 scale groups of exactly 128 values",
        ));
    }
    Ok(())
}

fn expected_tensor_path(root: &Path, tensor_name: &str) -> Result<PathBuf> {
    let tensors = root.join("tensors");
    let metadata = fs::symlink_metadata(&tensors).map_err(|error| {
        model_error(
            "complete binary artifact root",
            format!("cannot stat {}: {error}", tensors.display()),
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(model_error(
            "complete binary artifact root",
            format!(
                "{} must be a non-symlink tensors directory",
                tensors.display()
            ),
        ));
    }
    let filename = format!("{}.hq30g", sha256_hex(tensor_name.as_bytes()));
    Ok(tensors.join(filename))
}

fn declared_tensor_shape(row: &Map<String, Value>, label: &str) -> Result<Vec<usize>> {
    let values = required_array(row, "shape", label)?;
    if values.is_empty() {
        return Err(model_error(label, "shape must not be empty"));
    }
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let dimension = value.as_u64().ok_or_else(|| {
                model_error(label, format!("shape[{index}] must be an unsigned integer"))
            })?;
            if dimension == 0 {
                return Err(model_error(
                    label,
                    format!("shape[{index}] must be positive"),
                ));
            }
            usize::try_from(dimension).map_err(|_| {
                model_error(label, format!("shape[{index}] does not fit this platform"))
            })
        })
        .collect()
}

fn validate_tensor_row(
    row: &Map<String, Value>,
    root: &Path,
    source: &SourceChain,
) -> Result<(CompleteBinaryTensor, Arc<[u8]>)> {
    let label = "complete binary manifest tensor";
    let tensor_name = required_string(row, "tensor_name", label)?;
    if tensor_name.contains('\0') {
        return Err(model_error(label, "tensor_name contains a NUL byte"));
    }
    let source_shard = required_string(row, "source_shard", label)?;
    require_safe_filename(source_shard, label)?;
    let source_shard_sha256 = required_sha256(row, "source_shard_sha256", label)?;
    let expected_source_hash = source.shard_hashes.get(source_shard).ok_or_else(|| {
        model_error(
            label,
            format!("source shard {source_shard:?} is not in the sealed source receipt"),
        )
    })?;
    if &source_shard_sha256 != expected_source_hash {
        return Err(model_error(
            label,
            format!("source shard hash for {tensor_name:?} differs from the source receipt"),
        ));
    }
    if source.weight_map.get(tensor_name).map(String::as_str) != Some(source_shard) {
        return Err(model_error(
            label,
            format!("source index does not bind tensor {tensor_name:?} to shard {source_shard:?}"),
        ));
    }
    let source_dtype = required_string(row, "source_dtype", label)?;
    if !matches!(
        source_dtype,
        "BF16" | "BFLOAT16" | "F32" | "FLOAT32" | "F16" | "FLOAT16"
    ) {
        return Err(model_error(
            label,
            format!("tensor {tensor_name:?} has unsupported source_dtype {source_dtype:?}"),
        ));
    }
    let shape = declared_tensor_shape(row, label)?;
    let declared_elements = required_u64(row, "elements", label)?;
    let shape_elements = shape.iter().try_fold(1u64, |total, dimension| {
        total
            .checked_mul(u64::try_from(*dimension).unwrap_or(u64::MAX))
            .ok_or_else(|| model_error(label, "shape element product overflows u64"))
    })?;
    if declared_elements == 0 || declared_elements != shape_elements {
        return Err(model_error(
            label,
            format!("tensor {tensor_name:?} elements does not equal its shape product"),
        ));
    }

    let expected_path = expected_tensor_path(root, tensor_name)?;
    let expected_path = canonical_regular_path(&expected_path, label)?;
    let declared_path =
        manifest_descendant_file(root, required_string(row, "artifact_path", label)?, label)?;
    if declared_path != expected_path {
        return Err(model_error(
            label,
            format!(
                "tensor {tensor_name:?} artifact_path does not equal its deterministic tensor path"
            ),
        ));
    }
    let payload = read_regular_file(&expected_path, label)?;
    if required_u64(row, "artifact_bytes", label)?
        != u64::try_from(payload.len()).unwrap_or(u64::MAX)
    {
        return Err(model_error(
            label,
            format!("tensor {tensor_name:?} artifact_bytes does not equal physical payload bytes"),
        ));
    }
    let artifact_sha256 = required_sha256(row, "artifact_sha256", label)?;
    if sha256_hex(&payload) != artifact_sha256 {
        return Err(model_error(
            label,
            format!("tensor {tensor_name:?} payload SHA-256 does not match the manifest"),
        ));
    }
    let layout = required_object(row, "layout", label)?;
    require_exact_string(layout, "magic", "HQ30G1B1", label)?;
    if required_u64(layout, "version", label)? != u64::from(COMPLETE_BINARY_VERSION)
        || required_u64(layout, "group_size", label)? != 128
    {
        return Err(model_error(
            label,
            format!(
                "tensor {tensor_name:?} layout does not identify version-1 128-value direct groups"
            ),
        ));
    }
    require_exact_string(layout, "sign_bit_order", "little", label)?;
    require_exact_string(layout, "scale_dtype", "float16", label)?;

    let header = parse_complete_binary_header(&payload)?;
    if header.group_size != 128
        || header.version != COMPLETE_BINARY_VERSION
        || header.shape != shape
        || u64::try_from(header.elements).ok() != Some(declared_elements)
    {
        return Err(model_error(
            label,
            format!(
                "tensor {tensor_name:?} manifest geometry disagrees with its direct payload header"
            ),
        ));
    }
    Ok((
        CompleteBinaryTensor {
            tensor_name: tensor_name.to_owned(),
            source_shard: source_shard.to_owned(),
            source_shard_sha256,
            source_dtype: source_dtype.to_owned(),
            artifact_path: expected_path,
            artifact_sha256,
            header,
        },
        Arc::from(payload),
    ))
}

fn validate_ledger(
    manifest: &Map<String, Value>,
    tensors: &BTreeMap<String, CompleteBinaryTensor>,
    manifest_bytes: usize,
) -> Result<(u64, u64)> {
    let label = "complete binary manifest ledger";
    let ledger = required_object(manifest, "complete_physical_bpw_ledger", label)?;
    let payload_bytes = tensors.values().try_fold(0u64, |sum, tensor| {
        sum.checked_add(u64::try_from(tensor.header.payload_bytes).unwrap_or(u64::MAX))
            .ok_or_else(|| model_error(label, "tensor payload byte sum overflows u64"))
    })?;
    let elements = tensors.values().try_fold(0u64, |sum, tensor| {
        sum.checked_add(u64::try_from(tensor.header.elements).unwrap_or(u64::MAX))
            .ok_or_else(|| model_error(label, "source element sum overflows u64"))
    })?;
    if elements == 0
        || required_u64(ledger, "source_weight_elements", label)? != elements
        || required_u64(ledger, "tensor_payload_bytes", label)? != payload_bytes
    {
        return Err(model_error(
            label,
            "tensor payload or source-weight totals do not equal the validated manifest entries",
        ));
    }
    let billed_manifest_bytes = required_u64(ledger, "manifest_bytes_billed", label)?;
    if billed_manifest_bytes != u64::try_from(manifest_bytes).unwrap_or(u64::MAX) {
        return Err(model_error(
            label,
            "manifest_bytes_billed does not equal the exact manifest file length",
        ));
    }
    let total = payload_bytes
        .checked_add(billed_manifest_bytes)
        .ok_or_else(|| model_error(label, "all-required artifact byte sum overflows u64"))?;
    if required_u64(ledger, "all_required_weight_artifact_bytes", label)? != total {
        return Err(model_error(
            label,
            "all_required_weight_artifact_bytes does not equal payload plus manifest bytes",
        ));
    }
    let threshold = required_f64(ledger, "threshold_bpw", label)?;
    if threshold != 1.5 {
        return Err(model_error(label, "threshold_bpw must be exactly 1.5"));
    }
    let expected_bpw = (total as f64 * 8.0) / elements as f64;
    let reported_bpw = required_f64(ledger, "complete_physical_bpw", label)?;
    if (reported_bpw - expected_bpw).abs() > expected_bpw.abs().max(1.0) * 1e-12 {
        return Err(model_error(
            label,
            "complete_physical_bpw does not match the exact validated byte ledger",
        ));
    }
    if required_bool(ledger, "passes_storage_threshold", label)? != (expected_bpw <= threshold) {
        return Err(model_error(
            label,
            "passes_storage_threshold does not match the exact validated byte ledger",
        ));
    }
    Ok((elements, payload_bytes))
}

fn canonical_expected_regular_path(path: &Path, label: &str) -> Result<PathBuf> {
    if !path.is_absolute() {
        return Err(model_error(label, "protected path must be absolute"));
    }
    canonical_regular_path(path, label)
}

fn require_exact_regular_path(
    object: &Map<String, Value>,
    key: &str,
    expected: &Path,
    label: &str,
) -> Result<PathBuf> {
    let actual = canonical_regular_path(
        &absolute_path(required_string(object, key, label)?, label)?,
        label,
    )?;
    if actual != expected {
        return Err(model_error(
            label,
            format!("field {key:?} does not bind the protected path"),
        ));
    }
    Ok(actual)
}

fn validate_quality_file_binding(
    binding: &Map<String, Value>,
    expected_path: &Path,
    expected_raw_sha256: &str,
    expected_seal_sha256: &str,
    label: &str,
) -> Result<()> {
    require_exact_regular_path(binding, "path", expected_path, label)?;
    if required_sha256(binding, "document_sha256", label)? != expected_raw_sha256
        || required_sha256(binding, "seal_sha256", label)? != expected_seal_sha256
    {
        return Err(model_error(
            label,
            "file binding hash/seal differs from protected document",
        ));
    }
    Ok(())
}

fn quality_selected_organ_names(
    object: &Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<Vec<String>> {
    let names = required_array(object, key, label)?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_str()
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
                .ok_or_else(|| {
                    model_error(label, format!("{key}[{index}] must be a non-empty string"))
                })
        })
        .collect::<Result<Vec<_>>>()?;
    if names.as_slice() != QWEN30_QUALITY_REPACK_SELECTED_ORGANS {
        return Err(model_error(
            label,
            "selected organs differ from the sealed Qwen30 gate/up policy",
        ));
    }
    Ok(names)
}

fn validate_quality_manifest_source(
    manifest: &Map<String, Value>,
    source: &SourceChain,
) -> Result<()> {
    let label = "quality repack manifest";
    let manifest_source = required_object(manifest, "source", label)?;
    require_exact_string(
        manifest_source,
        "repository",
        QwenCompleteBinaryModel::Qwen30Coder.source_repository(),
        label,
    )?;
    if required_u64(manifest_source, "tensor_count", label)?
        != u64::try_from(source.weight_map.len()).unwrap_or(u64::MAX)
        || source.weight_map.len() != QWEN30_QUALITY_REPACK_TENSOR_COUNT
    {
        return Err(model_error(
            label,
            "source tensor count is not the full Qwen30 source index",
        ));
    }
    let declared_dir = canonical_directory(
        &absolute_path(required_string(manifest_source, "model_dir", label)?, label)?,
        label,
    )?;
    if source.source_index_path.parent() != Some(declared_dir.as_path()) {
        return Err(model_error(
            label,
            "source model directory differs from revalidated source index",
        ));
    }
    let representation = required_object(manifest, "representation", label)?;
    require_exact_string(
        representation,
        "family",
        "mixed_direct_binary_sign_scale_plus_selected_sparse_fp16_residual",
        label,
    )?;
    if !required_bool(representation, "physical_direct_layout", label)? {
        return Err(model_error(
            label,
            "quality representation must retain physical direct-layout accounting",
        ));
    }
    quality_selected_organ_names(representation, "selected_organs", label)?;
    Ok(())
}

fn selection_organs(selection: &Value) -> Result<BTreeMap<String, Map<String, Value>>> {
    let label = "quality repack selection receipt";
    let root = selection
        .as_object()
        .ok_or_else(|| model_error(label, "root must be an object"))?;
    require_exact_string(
        root,
        "schema",
        QWEN30_QUALITY_REPACK_SELECTION_SCHEMA,
        label,
    )?;
    require_exact_string(
        root,
        "status",
        "EARNED_SOURCE_BOUND_QUALITY_REPACK_SELECTION_UNQUALIFIED",
        label,
    )?;
    let selected = required_object(root, "selected_representation", label)?;
    require_exact_string(
        selected,
        "family",
        "binary_sign_scale_sparse_fp16_residual",
        label,
    )?;
    let organs = required_array(selected, "organs", label)?;
    if organs.len() != QWEN30_QUALITY_REPACK_SELECTED_ORGANS.len() {
        return Err(model_error(
            label,
            "selection must contain exactly the two approved gate/up organs",
        ));
    }
    let mut output = BTreeMap::new();
    for (position, value) in organs.iter().enumerate() {
        let organ = value
            .as_object()
            .ok_or_else(|| model_error(label, "selected organ must be an object"))?;
        let name = required_string(organ, "tensor_name", label)?;
        if name != QWEN30_QUALITY_REPACK_SELECTED_ORGANS[position]
            || output.insert(name.to_owned(), organ.clone()).is_some()
        {
            return Err(model_error(
                label,
                "selection organ ordering/content differs from policy",
            ));
        }
    }
    Ok(output)
}

fn validate_quality_selection_and_snapshot(
    manifest: &Map<String, Value>,
    root: &Path,
    admission: &Qwen30QualityRepackAdmission,
    expected_revalidation_path: &Path,
    expected_revalidation_raw_sha256: &str,
) -> Result<BTreeMap<String, Map<String, Value>>> {
    let label = "quality repack manifest";
    let expected_selection_path = canonical_expected_regular_path(
        &admission.expected_selection_path,
        "protected quality selection receipt",
    )?;
    let expected_snapshot_path = canonical_expected_regular_path(
        &admission.expected_source_snapshot_path,
        "protected quality source snapshot",
    )?;
    let selection_raw =
        read_regular_file(&expected_selection_path, "quality repack selection receipt")?;
    let selection =
        parse_json_no_duplicate_keys(&selection_raw, "quality repack selection receipt")?;
    let selection_seal = verify_sealed_document(&selection, "quality repack selection receipt")?;
    if selection_seal != admission.expected_selection_seal_sha256 {
        return Err(model_error(
            label,
            "selection receipt seal differs from protected handoff binding",
        ));
    }
    let snapshot_raw =
        read_regular_file(&expected_snapshot_path, "quality source binding snapshot")?;
    let snapshot = parse_json_no_duplicate_keys(&snapshot_raw, "quality source binding snapshot")?;
    let snapshot_seal = verify_sealed_document(&snapshot, "quality source binding snapshot")?;
    if snapshot_seal != admission.expected_source_snapshot_seal_sha256 {
        return Err(model_error(
            label,
            "source snapshot seal differs from protected handoff binding",
        ));
    }
    let branch = required_object(manifest, "quality_repack_branch", label)?;
    require_exact_string(branch, "branch_id", QWEN30_QUALITY_REPACK_BRANCH_ID, label)?;
    let branch_snapshot = required_object(branch, "source_binding_snapshot", label)?;
    validate_quality_file_binding(
        branch_snapshot,
        &expected_snapshot_path,
        &sha256_hex(&snapshot_raw),
        &snapshot_seal,
        "quality manifest source snapshot binding",
    )?;
    let branch_selection = required_object(branch, "selection_receipt", label)?;
    validate_quality_file_binding(
        branch_selection,
        &expected_selection_path,
        &sha256_hex(&selection_raw),
        &selection_seal,
        "quality manifest selection binding",
    )?;
    quality_selected_organ_names(branch, "changed_organs", label)?;

    let snapshot_root = snapshot
        .as_object()
        .ok_or_else(|| model_error("quality source binding snapshot", "root must be an object"))?;
    require_exact_string(
        snapshot_root,
        "schema",
        QWEN30_QUALITY_REPACK_SNAPSHOT_SCHEMA,
        "quality source binding snapshot",
    )?;
    require_exact_string(
        snapshot_root,
        "status",
        "EARNED_IMMUTABLE_SOURCE_AND_ROLLBACK_BINDING",
        "quality source binding snapshot",
    )?;
    let snapshot_binding =
        required_object(snapshot_root, "binding", "quality source binding snapshot")?;
    require_exact_string(
        snapshot_binding,
        "branch_id",
        QWEN30_QUALITY_REPACK_BRANCH_ID,
        "quality source binding snapshot",
    )?;
    quality_selected_organ_names(
        snapshot_binding,
        "selected_organs",
        "quality source binding snapshot",
    )?;
    let snapshot_revalidation = required_object(
        snapshot_binding,
        "immutable_source_revalidation",
        "quality source binding snapshot",
    )?;
    validate_quality_file_binding(
        snapshot_revalidation,
        expected_revalidation_path,
        expected_revalidation_raw_sha256,
        &admission.expected_revalidation_seal_sha256,
        "quality source snapshot immutable revalidation",
    )?;

    let selection_root = selection
        .as_object()
        .ok_or_else(|| model_error("quality repack selection receipt", "root must be an object"))?;
    let selection_snapshot = required_object(
        selection_root,
        "source_binding_snapshot",
        "quality repack selection receipt",
    )?;
    validate_quality_file_binding(
        selection_snapshot,
        &expected_snapshot_path,
        &sha256_hex(&snapshot_raw),
        &snapshot_seal,
        "quality selection source snapshot binding",
    )?;
    let selection_binding = required_object(
        selection_root,
        "binding",
        "quality repack selection receipt",
    )?;
    if selection_binding != snapshot_binding {
        return Err(model_error(
            label,
            "selection and source snapshot do not share one immutable binding",
        ));
    }
    let _ = root; // the branch paths above are explicit and independently canonicalized.
    selection_organs(&selection)
}

fn validate_quality_terminal(
    terminal: &Value,
    terminal_path: &Path,
    manifest_path: &Path,
    manifest_raw_sha256: &str,
    manifest_seal: &str,
    manifest_bytes: usize,
    admission: &Qwen30QualityRepackAdmission,
) -> Result<()> {
    let label = "quality repack terminal receipt";
    let object = terminal
        .as_object()
        .ok_or_else(|| model_error(label, "root must be an object"))?;
    verify_sealed_document(terminal, label)?;
    require_exact_string(
        object,
        "schema",
        QWEN30_QUALITY_REPACK_TERMINAL_SCHEMA,
        label,
    )?;
    require_exact_string(
        object,
        "status",
        "EARNED_COMPLETE_PHYSICAL_BINARY_CANDIDATE_UNQUALIFIED",
        label,
    )?;
    let binding = required_object(object, "binding", label)?;
    require_exact_string(binding, "model_id", QWEN30_QUALITY_REPACK_MODEL_ID, label)?;
    require_exact_string(
        binding,
        "artifact_prefix",
        "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1",
        label,
    )?;
    require_exact_string(
        binding,
        "manifest_schema",
        QWEN30_QUALITY_REPACK_SCHEMA,
        label,
    )?;
    let progress = required_object(binding, "progress", label)?;
    for key in ["planned_tensors", "completed_tensors", "next_cursor"] {
        if required_u64(progress, key, label)?
            != u64::try_from(QWEN30_QUALITY_REPACK_TENSOR_COUNT).unwrap()
        {
            return Err(model_error(
                label,
                "terminal cursor does not prove all Qwen30 tensors completed",
            ));
        }
    }
    if progress.get("next_source_shard") != Some(&Value::Null)
        || progress.get("next_tensor_name") != Some(&Value::Null)
    {
        return Err(model_error(
            label,
            "terminal cursor retains a next source tensor",
        ));
    }
    let candidate = required_object(object, "candidate", label)?;
    require_exact_regular_path(candidate, "manifest_path", manifest_path, label)?;
    if required_sha256(candidate, "manifest_document_sha256", label)? != manifest_raw_sha256
        || required_sha256(candidate, "manifest_seal_sha256", label)? != manifest_seal
    {
        return Err(model_error(
            label,
            "terminal candidate manifest binding differs from current manifest",
        ));
    }
    let identity = required_object(candidate, "manifest_file_identity", label)?;
    if required_u64(identity, "bytes", label)? != u64::try_from(manifest_bytes).unwrap_or(u64::MAX)
    {
        return Err(model_error(
            label,
            "terminal manifest identity byte count differs from current manifest",
        ));
    }
    let expected_terminal_path = canonical_expected_regular_path(
        &admission.expected_terminal_path,
        "protected quality terminal receipt",
    )?;
    if terminal_path != expected_terminal_path {
        return Err(model_error(
            label,
            "terminal path differs from protected handoff path",
        ));
    }
    Ok(())
}

fn validate_quality_tensor_row(
    row: &Map<String, Value>,
    root: &Path,
    source: &SourceChain,
    selected_organs: &BTreeMap<String, Map<String, Value>>,
    selection_path: &Path,
    selection_seal: &str,
) -> Result<(Qwen30QualityRepackTensor, Arc<[u8]>)> {
    let label = "quality repack manifest tensor";
    let tensor_name = required_string(row, "tensor_name", label)?;
    if tensor_name.contains('\0') {
        return Err(model_error(label, "tensor name contains a NUL byte"));
    }
    let source_shard = required_string(row, "source_shard", label)?;
    require_safe_filename(source_shard, label)?;
    let source_shard_sha256 = required_sha256(row, "source_shard_sha256", label)?;
    if source.shard_hashes.get(source_shard).map(String::as_str)
        != Some(source_shard_sha256.as_str())
        || source.weight_map.get(tensor_name).map(String::as_str) != Some(source_shard)
    {
        return Err(model_error(
            label,
            "tensor source shard does not match current revalidated source index",
        ));
    }
    let source_dtype = required_string(row, "source_dtype", label)?;
    if !matches!(
        source_dtype,
        "BF16" | "BFLOAT16" | "F32" | "FLOAT32" | "F16" | "FLOAT16"
    ) {
        return Err(model_error(label, "tensor source dtype is unsupported"));
    }
    let shape = declared_tensor_shape(row, label)?;
    let elements = required_u64(row, "elements", label)?;
    let shape_elements = shape.iter().try_fold(1u64, |total, dimension| {
        total
            .checked_mul(u64::try_from(*dimension).unwrap_or(u64::MAX))
            .ok_or_else(|| model_error(label, "tensor shape product overflows u64"))
    })?;
    if elements == 0 || elements != shape_elements {
        return Err(model_error(
            label,
            "tensor elements do not equal shape product",
        ));
    }
    let expected_path = canonical_regular_path(&expected_tensor_path(root, tensor_name)?, label)?;
    let declared_path =
        manifest_descendant_file(root, required_string(row, "artifact_path", label)?, label)?;
    if declared_path != expected_path {
        return Err(model_error(
            label,
            "tensor payload path is not the deterministic candidate path",
        ));
    }
    let payload = read_regular_file(&expected_path, label)?;
    let artifact_bytes = required_u64(row, "artifact_bytes", label)?;
    if artifact_bytes != u64::try_from(payload.len()).unwrap_or(u64::MAX) {
        return Err(model_error(
            label,
            "tensor artifact_bytes does not equal physical payload bytes",
        ));
    }
    let artifact_sha256 = required_sha256(row, "artifact_sha256", label)?;
    if sha256_hex(&payload) != artifact_sha256 {
        return Err(model_error(
            label,
            "tensor payload SHA-256 does not match the manifest",
        ));
    }
    let mutation = required_object(row, "candidate_mutation", label)?;
    let rollback = required_object(mutation, "baseline_rollback", label)?;
    require_exact_string(
        rollback,
        "rollback_action",
        "use the separately admitted baseline tensor; this candidate never overwrites it",
        label,
    )?;
    let layout = required_object(row, "layout", label)?;
    let is_selected = selected_organs.contains_key(tensor_name);
    let parsed_layout = if is_selected {
        if !required_bool(mutation, "changed_from_admitted_control", label)? {
            return Err(model_error(
                label,
                "selected organ was not marked as the explicit quality mutation",
            ));
        }
        require_exact_regular_path(mutation, "selection_receipt_path", selection_path, label)?;
        if required_sha256(mutation, "selection_receipt_seal_sha256", label)? != selection_seal {
            return Err(model_error(
                label,
                "selected organ selection seal differs from protected receipt",
            ));
        }
        let selected = selected_organs
            .get(tensor_name)
            .expect("checked contains_key");
        let representation = selected
            .get("representation")
            .and_then(Value::as_object)
            .ok_or_else(|| model_error(label, "selected organ has no sealed representation"))?;
        if layout != representation {
            return Err(model_error(
                label,
                "selected organ layout differs from sealed selection representation",
            ));
        }
        if required_sha256(selected, "physical_payload_sha256", label)? != artifact_sha256
            || required_u64(selected, "physical_payload_bytes", label)? != artifact_bytes
        {
            return Err(model_error(
                label,
                "selected organ physical payload differs from sealed selection",
            ));
        }
        require_exact_string(
            layout,
            "magic",
            QWEN30_QUALITY_REPACK_RESIDUAL_MAGIC_TEXT,
            label,
        )?;
        if required_u64(layout, "version", label)? != u64::from(QWEN30_QUALITY_REPACK_VERSION) {
            return Err(model_error(
                label,
                "selected organ residual layout version is unsupported",
            ));
        }
        let parsed = parse_qwen30_quality_residual_header(&payload)?;
        if parsed.shape != shape || u64::try_from(parsed.base.elements).ok() != Some(elements) {
            return Err(model_error(
                label,
                "selected organ residual payload geometry differs from manifest",
            ));
        }
        let residual = required_object(layout, "residual", label)?;
        if required_u64(residual, "selected_count", label)?
            != u64::try_from(parsed.residual_count).unwrap_or(u64::MAX)
            || required_u64(residual, "index_bytes", label)?
                != u64::try_from(parsed.residual_count.saturating_mul(4)).unwrap_or(u64::MAX)
            || required_u64(residual, "value_bytes", label)?
                != u64::try_from(parsed.residual_count.saturating_mul(2)).unwrap_or(u64::MAX)
            || required_sha256(residual, "indices_sha256", label)? != parsed.indices_sha256
            || required_sha256(residual, "values_sha256", label)? != parsed.values_sha256
        {
            return Err(model_error(
                label,
                "selected organ residual discriminator differs from exact payload",
            ));
        }
        let discriminator = required_object(mutation, "source_to_packed_discriminator", label)?;
        if required_sha256(discriminator, "payload_sha256", label)? != artifact_sha256 {
            return Err(model_error(
                label,
                "selected organ source-to-packed discriminator payload differs",
            ));
        }
        Qwen30QualityRepackTensorLayout::SparseResidual(parsed)
    } else {
        if required_bool(mutation, "changed_from_admitted_control", label)? {
            return Err(model_error(label, "non-selected organ was mutated"));
        }
        require_exact_string(layout, "magic", "HQ30G1B1", label)?;
        if required_u64(layout, "version", label)? != u64::from(COMPLETE_BINARY_VERSION)
            || required_u64(layout, "group_size", label)? != 128
        {
            return Err(model_error(
                label,
                "control tensor direct layout version/group differs",
            ));
        }
        require_exact_string(layout, "sign_bit_order", "little", label)?;
        require_exact_string(layout, "scale_dtype", "float16", label)?;
        let parsed = parse_complete_binary_header(&payload)?;
        if parsed.shape != shape || u64::try_from(parsed.elements).ok() != Some(elements) {
            return Err(model_error(
                label,
                "control tensor direct payload geometry differs from manifest",
            ));
        }
        Qwen30QualityRepackTensorLayout::Direct(parsed)
    };
    let artifact_bytes = payload.len();
    let snapshot: Arc<[u8]> = payload.into();
    Ok((
        Qwen30QualityRepackTensor {
            tensor_name: tensor_name.to_owned(),
            source_shard: source_shard.to_owned(),
            source_shard_sha256,
            source_dtype: source_dtype.to_owned(),
            artifact_path: expected_path,
            artifact_sha256,
            artifact_bytes,
            elements: usize::try_from(elements)
                .map_err(|_| model_error(label, "tensor elements do not fit platform usize"))?,
            layout: parsed_layout,
        },
        snapshot,
    ))
}

fn validate_quality_ledger(
    manifest: &Map<String, Value>,
    tensors: &BTreeMap<String, Qwen30QualityRepackTensor>,
    manifest_bytes: usize,
) -> Result<(u64, u64)> {
    let label = "quality repack manifest ledger";
    let ledger = required_object(manifest, "complete_physical_bpw_ledger", label)?;
    let payload_bytes = tensors.values().try_fold(0u64, |total, tensor| {
        total
            .checked_add(u64::try_from(tensor.artifact_bytes).unwrap_or(u64::MAX))
            .ok_or_else(|| model_error(label, "tensor payload byte total overflows"))
    })?;
    let elements = tensors.values().try_fold(0u64, |total, tensor| {
        total
            .checked_add(u64::try_from(tensor.elements).unwrap_or(u64::MAX))
            .ok_or_else(|| model_error(label, "source element total overflows"))
    })?;
    if elements == 0
        || required_u64(ledger, "source_weight_elements", label)? != elements
        || required_u64(ledger, "tensor_payload_bytes", label)? != payload_bytes
        || required_u64(ledger, "manifest_bytes_billed", label)?
            != u64::try_from(manifest_bytes).unwrap_or(u64::MAX)
    {
        return Err(model_error(
            label,
            "ledger physical totals differ from scanned candidate",
        ));
    }
    let total = payload_bytes
        .checked_add(u64::try_from(manifest_bytes).unwrap_or(u64::MAX))
        .ok_or_else(|| model_error(label, "total physical artifact bytes overflow"))?;
    if required_u64(ledger, "all_required_weight_artifact_bytes", label)? != total {
        return Err(model_error(
            label,
            "all required artifact bytes differ from payload plus manifest",
        ));
    }
    if required_f64(ledger, "threshold_bpw", label)? != 1.5 {
        return Err(model_error(label, "threshold_bpw must be exactly 1.5"));
    }
    let expected_bpw = total as f64 * 8.0 / elements as f64;
    if (required_f64(ledger, "complete_physical_bpw", label)? - expected_bpw).abs()
        > expected_bpw.abs().max(1.0) * 1e-12
        || !required_bool(ledger, "passes_storage_threshold", label)?
        || expected_bpw > 1.5
    {
        return Err(model_error(
            label,
            "candidate did not earn an exact <=1.5 complete BPW ledger",
        ));
    }
    Ok((elements, payload_bytes))
}

/// Assign manifest rows to a fixed number of source-shard lanes without
/// trusting the row for anything other than read-only scheduling.  A malformed
/// row receives its own deterministic synthetic key and is still rejected by
/// the normal row validator.  Keeping all rows from a named shard in the same
/// lane makes the admission I/O pattern reproducible, while the caller later
/// reconciles outcomes in manifest-ordinal order.
fn quality_payload_verification_lanes(rows: &[Value]) -> (Vec<Vec<usize>>, usize) {
    if rows.is_empty() {
        return (Vec::new(), 0);
    }
    let row_keys = rows
        .iter()
        .enumerate()
        .map(|(ordinal, value)| {
            value
                .as_object()
                .and_then(|row| row.get("source_shard"))
                .and_then(Value::as_str)
                .filter(|shard| !shard.is_empty())
                .map(|shard| format!("source-shard:{shard}"))
                // Do not collapse malformed rows into one lane.  They still
                // receive a deterministic ordinal and will fail strict row
                // validation after worker results are ordered below.
                .unwrap_or_else(|| format!("malformed-row:{ordinal:020}"))
        })
        .collect::<Vec<_>>();
    let distinct = row_keys.iter().cloned().collect::<BTreeSet<_>>();
    let workers = distinct
        .len()
        .min(rows.len())
        .min(QWEN30_QUALITY_REPACK_MAX_PAYLOAD_VERIFY_WORKERS)
        .max(1);
    let lane_by_key = distinct
        .into_iter()
        .enumerate()
        .map(|(index, key)| (key, index % workers))
        .collect::<BTreeMap<_, _>>();
    let mut lanes = (0..workers).map(|_| Vec::new()).collect::<Vec<_>>();
    for (ordinal, key) in row_keys.iter().enumerate() {
        // Every key was inserted into lane_by_key immediately above.
        let lane = *lane_by_key
            .get(key)
            .expect("quality payload verification lane key must be present");
        lanes[lane].push(ordinal);
    }
    (lanes, workers)
}

/// Scan independent candidate payload files in bounded source-shard lanes.
///
/// There is no concurrent mutation: each worker only reads a distinct set of
/// deterministic artifact paths, and retains no shared admission state.  The
/// coordinator joins every lane, sorts results by manifest ordinal, and only
/// then permits duplicate/set/ledger reconciliation.  Thus worker completion
/// order cannot affect either an error chosen for the same manifest or the
/// resulting immutable catalog.
fn validate_quality_tensor_rows_bounded_parallel(
    rows: &[Value],
    root: &Path,
    source: &SourceChain,
    selected_organs: &BTreeMap<String, Map<String, Value>>,
    selection_path: &Path,
    selection_seal: &str,
) -> Result<(Vec<(Qwen30QualityRepackTensor, Arc<[u8]>)>, usize)> {
    let (lanes, workers) = quality_payload_verification_lanes(rows);
    if workers == 0 {
        return Ok((Vec::new(), 0));
    }
    let outcomes = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(lanes.len());
        for lane in &lanes {
            handles.push(scope.spawn(move || {
                let mut lane_outcomes = Vec::with_capacity(lane.len());
                for &ordinal in lane {
                    let outcome = rows[ordinal]
                        .as_object()
                        .ok_or_else(|| {
                            Error::Model(
                                "quality repack manifest tensor entry must be an object".into(),
                            )
                        })
                        .and_then(|row| {
                            validate_quality_tensor_row(
                                row,
                                root,
                                source,
                                selected_organs,
                                selection_path,
                                selection_seal,
                            )
                        });
                    lane_outcomes.push((ordinal, outcome));
                }
                lane_outcomes
            }));
        }
        let mut joined = Vec::with_capacity(rows.len());
        for handle in handles {
            let lane = handle.join().map_err(|_| {
                Error::Model(
                    "quality payload verification worker panicked; refusing candidate admission"
                        .into(),
                )
            })?;
            joined.extend(lane);
        }
        Ok::<_, Error>(joined)
    })?;
    let mut outcomes = outcomes;
    outcomes.sort_by_key(|(ordinal, _)| *ordinal);
    let mut tensors = Vec::with_capacity(outcomes.len());
    for (ordinal, outcome) in outcomes {
        let tensor = outcome.map_err(|error| {
            Error::Model(format!(
                "quality payload verification failed at manifest ordinal {ordinal}: {error}"
            ))
        })?;
        tensors.push(tensor);
    }
    Ok((tensors, workers))
}

/// Admit an all-tensor Qwen30 or Qwen80 direct-binary candidate by verifying
/// the protected manifest seal, its source-audit chain, current source index
/// and shard identities, every tensor payload hash, and all fixed-layout
/// geometry.  It deliberately has no decoder, HCLI, or TPS side effect.
///
/// Warm path (`HAWKING_ADMISSION_WARM_RECEIPT`, default on): after a prior cold
/// full rehash wrote a sealed receipt, a later process may skip *recomputing*
/// content SHA-256 when every payload file's (size, mtime_ns, inode)
/// still matches. It never skips the unchanged-file check. Disable with
/// `HAWKING_ADMISSION_WARM_RECEIPT=0` to force a full cold rehash every start.
pub fn admit_complete_binary_artifact(
    manifest_path: impl AsRef<Path>,
    admission: &CompleteBinaryAdmission,
) -> Result<CompleteBinaryArtifact> {
    crate::startup_timing::time_ms_result("admit_complete_binary_total", || {
        admit_complete_binary_artifact_inner(manifest_path, admission)
    })
}

fn admit_complete_binary_artifact_inner(
    manifest_path: impl AsRef<Path>,
    admission: &CompleteBinaryAdmission,
) -> Result<CompleteBinaryArtifact> {
    if !is_sha256(&admission.expected_manifest_seal_sha256)
        || !is_sha256(&admission.expected_source_audit_seal_sha256)
        || admission.expected_source_revision.is_empty()
    {
        return Err(Error::Model(
            "complete binary admission requires protected lowercase SHA-256 seals and a source revision"
                .into(),
        ));
    }
    let manifest_path = crate::startup_timing::time_ms_result("admit_manifest_seal", || {
        canonical_regular_path(manifest_path.as_ref(), "complete binary manifest")
    })?;
    let root = manifest_path.parent().ok_or_else(|| {
        Error::Model("complete binary manifest has no parent artifact root".into())
    })?;
    let manifest_raw = read_regular_file(&manifest_path, "complete binary manifest")?;
    let manifest = parse_json_no_duplicate_keys(&manifest_raw, "complete binary manifest")?;
    let manifest_object = manifest
        .as_object()
        .ok_or_else(|| Error::Model("complete binary manifest: root must be an object".into()))?;
    let manifest_seal = verify_sealed_document(&manifest, "complete binary manifest")?;
    if manifest_seal != admission.expected_manifest_seal_sha256 {
        return Err(Error::Model(
            "complete binary manifest seal does not match the protected admission binding".into(),
        ));
    }
    require_exact_string(
        manifest_object,
        "schema",
        admission.model.manifest_schema(),
        "complete binary manifest",
    )?;
    require_exact_string(
        manifest_object,
        "status",
        COMPLETE_BINARY_CANDIDATE_STATUS,
        "complete binary manifest",
    )?;
    let manifest_audit_seal = required_sha256(
        manifest_object,
        "source_body_audit_seal_sha256",
        "complete binary manifest",
    )?;
    if manifest_audit_seal != admission.expected_source_audit_seal_sha256 {
        return Err(Error::Model(
            "complete binary manifest source audit seal does not match protected admission binding"
                .into(),
        ));
    }
    let receipt_path = manifest_child_file(
        root,
        required_string(
            manifest_object,
            "source_revalidation_receipt_path",
            "complete binary manifest",
        )?,
        "complete binary source revalidation receipt",
    )?;
    let source = crate::startup_timing::time_ms_result("admit_source_chain", || {
        let receipt_raw =
            read_regular_file(&receipt_path, "complete binary source revalidation receipt")?;
        let receipt = parse_json_no_duplicate_keys(
            &receipt_raw,
            "complete binary source revalidation receipt",
        )?;
        let receipt_seal =
            verify_sealed_document(&receipt, "complete binary source revalidation receipt")?;
        if receipt_seal
            != required_sha256(
                manifest_object,
                "source_revalidation_receipt_seal_sha256",
                "complete binary manifest",
            )?
        {
            return Err(Error::Model(
                "complete binary manifest revalidation receipt seal does not match its payload"
                    .into(),
            ));
        }
        let source = validate_source_chain(
            &receipt,
            &receipt_path,
            root,
            admission,
            &manifest_audit_seal,
        )?;
        validate_manifest_source(manifest_object, &source, admission.model)?;
        Ok(source)
    })?;

    let rows = required_array(manifest_object, "tensors", "complete binary manifest")?;
    if rows.len() != source.weight_map.len() {
        return Err(Error::Model(
            "complete binary manifest tensor count does not match the revalidated source index"
                .into(),
        ));
    }

    // ---- Warm path: skip content rehash when identity metadata still matches ----
    if admission_warm_receipt::warm_receipt_enabled() {
        match try_warm_payload_admission(
            &manifest_path,
            &manifest_seal,
            root,
            &source,
            rows,
            admission.model,
            manifest_object,
            manifest_raw.len(),
        ) {
            Ok(Some(artifact)) => {
                crate::startup_timing::record_ms("admit_payload_path", 0);
                crate::startup_timing::record_ms("admit_payload_mode_warm_skip_rehash", 1);
                return Ok(artifact);
            }
            Ok(None) => {
                // Fall through to cold full rehash.
            }
            Err(error) => {
                // Warm path must never weaken the gate: any hard failure during
                // a claimed warm hit aborts rather than silently accepting.
                return Err(error);
            }
        }
    }

    // ---- Cold path: full per-payload SHA-256 ----
    // Default: bounded parallel by source-shard lanes (same pattern as quality
    // repack). Force historical sequential scan with
    // `HAWKING_ADMISSION_PARALLEL=0` for baseline measurement.
    let (tensors, verified_payloads) =
        crate::startup_timing::time_ms_result("admit_payload_cold_rehash", || {
            let parallel = match std::env::var("HAWKING_ADMISSION_PARALLEL") {
                Ok(v)
                    if matches!(
                        v.as_str(),
                        "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
                    ) =>
                {
                    false
                }
                _ => true,
            };
            if parallel {
                validate_complete_binary_tensor_rows_bounded_parallel(rows, root, &source)
            } else {
                let mut tensors = BTreeMap::new();
                let mut verified_payloads = BTreeMap::new();
                for value in rows {
                    let row = value.as_object().ok_or_else(|| {
                        Error::Model(
                            "complete binary manifest tensor entry must be an object".into(),
                        )
                    })?;
                    let (tensor, verified_payload) = validate_tensor_row(row, root, &source)?;
                    let tensor_name = tensor.tensor_name.clone();
                    if tensors.insert(tensor_name.clone(), tensor).is_some() {
                        return Err(Error::Model(
                            "complete binary manifest contains a duplicate tensor_name".into(),
                        ));
                    }
                    if verified_payloads
                        .insert(tensor_name, verified_payload)
                        .is_some()
                    {
                        return Err(Error::Model(
                            "complete binary manifest duplicated an immutable payload entry".into(),
                        ));
                    }
                }
                Ok((tensors, verified_payloads))
            }
        })?;
    crate::startup_timing::record_ms("admit_payload_mode_cold_full_rehash", 1);

    if tensors.keys().ne(source.weight_map.keys()) {
        return Err(Error::Model(
            "complete binary manifest tensor set does not exactly match the revalidated source index"
                .into(),
        ));
    }
    if verified_payloads.len() != tensors.len() {
        return Err(Error::Model(
            "complete binary admission did not retain one verified immutable payload per tensor"
                .into(),
        ));
    }
    let (source_weight_elements, tensor_payload_bytes) =
        validate_ledger(manifest_object, &tensors, manifest_raw.len())?;
    let artifact = CompleteBinaryArtifact {
        model: admission.model,
        manifest_path: manifest_path.clone(),
        manifest_seal_sha256: manifest_seal.clone(),
        source_audit_path: source.source_audit_path,
        source_audit_seal_sha256: source.source_audit_seal_sha256,
        source_revision: source.source_revision,
        source_index_path: source.source_index_path,
        source_weight_elements,
        tensor_payload_bytes,
        tensors,
        verified_payloads,
    };

    if admission_warm_receipt::warm_receipt_enabled() {
        let _ = crate::startup_timing::time_ms_result("admit_warm_receipt_write", || {
            let receipt = admission_warm_receipt::build_receipt_from_admitted(
                &artifact.manifest_path,
                &artifact.manifest_seal_sha256,
                &artifact.tensors,
            )?;
            admission_warm_receipt::write_receipt(&receipt)
        });
    }

    Ok(artifact)
}

fn try_warm_payload_admission(
    manifest_path: &Path,
    manifest_seal: &str,
    root: &Path,
    source: &SourceChain,
    rows: &[Value],
    model: QwenCompleteBinaryModel,
    manifest_object: &Map<String, Value>,
    manifest_raw_len: usize,
) -> Result<Option<CompleteBinaryArtifact>> {
    let Some(receipt) = admission_warm_receipt::load_receipt(manifest_seal)? else {
        return Ok(None);
    };
    if receipt.manifest_path != manifest_path {
        // Same seal should imply same content, but require path identity too.
        return Ok(None);
    }
    if !admission_warm_receipt::receipt_covers_manifest_rows(&receipt, rows, root, source)? {
        return Ok(None);
    }
    let identity_ok =
        crate::startup_timing::time_ms_result("admit_warm_identity_recheck", || {
            admission_warm_receipt::receipt_identities_still_match(&receipt)
        })?;
    if !identity_ok {
        return Ok(None);
    }

    // Identity matches: load payloads without content rehash (still prove size
    // and header geometry). Parallel by source-shard lanes for I/O.
    let (tensors, verified_payloads) =
        crate::startup_timing::time_ms_result("admit_payload_warm_load_no_rehash", || {
            load_warm_payloads_bounded_parallel(&receipt)
        })?;

    if tensors.keys().ne(source.weight_map.keys()) {
        return Err(Error::Model(
            "warm admission tensor set does not exactly match the revalidated source index".into(),
        ));
    }
    if verified_payloads.len() != tensors.len() {
        return Err(Error::Model(
            "warm admission did not retain one immutable payload per tensor".into(),
        ));
    }
    let (source_weight_elements, tensor_payload_bytes) =
        validate_ledger(manifest_object, &tensors, manifest_raw_len)?;
    Ok(Some(CompleteBinaryArtifact {
        model,
        manifest_path: manifest_path.to_path_buf(),
        manifest_seal_sha256: manifest_seal.to_owned(),
        source_audit_path: source.source_audit_path.clone(),
        source_audit_seal_sha256: source.source_audit_seal_sha256.clone(),
        source_revision: source.source_revision.clone(),
        source_index_path: source.source_index_path.clone(),
        source_weight_elements,
        tensor_payload_bytes,
        tensors,
        verified_payloads,
    }))
}

fn validate_complete_binary_tensor_rows_bounded_parallel(
    rows: &[Value],
    root: &Path,
    source: &SourceChain,
) -> Result<(
    BTreeMap<String, CompleteBinaryTensor>,
    BTreeMap<String, Arc<[u8]>>,
)> {
    let (lanes, workers) = quality_payload_verification_lanes(rows);
    if workers == 0 {
        return Ok((BTreeMap::new(), BTreeMap::new()));
    }
    let outcomes = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(lanes.len());
        for lane in &lanes {
            handles.push(scope.spawn(move || {
                let mut lane_outcomes = Vec::with_capacity(lane.len());
                for &ordinal in lane {
                    let outcome = rows[ordinal]
                        .as_object()
                        .ok_or_else(|| {
                            Error::Model(
                                "complete binary manifest tensor entry must be an object".into(),
                            )
                        })
                        .and_then(|row| validate_tensor_row(row, root, source));
                    lane_outcomes.push((ordinal, outcome));
                }
                lane_outcomes
            }));
        }
        let mut joined = Vec::with_capacity(rows.len());
        for handle in handles {
            let lane = handle.join().map_err(|_| {
                Error::Model(
                    "complete binary payload verification worker panicked; refusing admission"
                        .into(),
                )
            })?;
            joined.extend(lane);
        }
        Ok::<_, Error>(joined)
    })?;
    let mut outcomes = outcomes;
    outcomes.sort_by_key(|(ordinal, _)| *ordinal);
    let mut tensors = BTreeMap::new();
    let mut verified_payloads = BTreeMap::new();
    for (ordinal, outcome) in outcomes {
        let (tensor, payload) = outcome.map_err(|error| {
            Error::Model(format!(
                "complete binary payload verification failed at manifest ordinal {ordinal}: {error}"
            ))
        })?;
        let name = tensor.tensor_name.clone();
        if tensors.insert(name.clone(), tensor).is_some() {
            return Err(Error::Model(
                "complete binary manifest contains a duplicate tensor_name".into(),
            ));
        }
        if verified_payloads.insert(name, payload).is_some() {
            return Err(Error::Model(
                "complete binary manifest duplicated an immutable payload entry".into(),
            ));
        }
    }
    Ok((tensors, verified_payloads))
}

fn load_warm_payloads_bounded_parallel(
    receipt: &admission_warm_receipt::WarmAdmissionReceipt,
) -> Result<(
    BTreeMap<String, CompleteBinaryTensor>,
    BTreeMap<String, Arc<[u8]>>,
)> {
    let entries: Vec<&admission_warm_receipt::ReceiptEntry> =
        receipt.entries.values().collect();
    if entries.is_empty() {
        return Ok((BTreeMap::new(), BTreeMap::new()));
    }
    let workers = entries
        .len()
        .min(QWEN30_QUALITY_REPACK_MAX_PAYLOAD_VERIFY_WORKERS)
        .max(1);
    let mut lanes: Vec<Vec<usize>> = (0..workers).map(|_| Vec::new()).collect();
    for (idx, _) in entries.iter().enumerate() {
        lanes[idx % workers].push(idx);
    }
    let outcomes = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(lanes.len());
        for lane in &lanes {
            let entries = &entries;
            handles.push(scope.spawn(move || {
                let mut lane_outcomes = Vec::with_capacity(lane.len());
                for &idx in lane {
                    let outcome =
                        admission_warm_receipt::load_payload_warm_skip_hash(entries[idx]);
                    lane_outcomes.push((idx, outcome));
                }
                lane_outcomes
            }));
        }
        let mut joined = Vec::with_capacity(entries.len());
        for handle in handles {
            let lane = handle.join().map_err(|_| {
                Error::Model(
                    "warm admission payload load worker panicked; refusing admission".into(),
                )
            })?;
            joined.extend(lane);
        }
        Ok::<_, Error>(joined)
    })?;
    let mut outcomes = outcomes;
    outcomes.sort_by_key(|(idx, _)| *idx);
    let mut tensors = BTreeMap::new();
    let mut verified_payloads = BTreeMap::new();
    for (idx, outcome) in outcomes {
        let (tensor, payload) = outcome.map_err(|error| {
            Error::Model(format!(
                "warm admission payload load failed at entry {idx}: {error}"
            ))
        })?;
        let name = tensor.tensor_name.clone();
        if tensors.insert(name.clone(), tensor).is_some() {
            return Err(Error::Model(
                "warm admission produced a duplicate tensor_name".into(),
            ));
        }
        if verified_payloads.insert(name, payload).is_some() {
            return Err(Error::Model(
                "warm admission duplicated an immutable payload entry".into(),
            ));
        }
    }
    Ok((tensors, verified_payloads))
}

/// Strict native admission for the separately rooted Qwen30 gate/up residual
/// candidate.  This has intentionally no relationship to the normal admitted
/// Qwen30 control's current pointer: it validates the candidate's own sealed
/// terminal handoff, then scans every tensor and its exact selected-residual
/// discriminators before returning an admission-only catalog.
pub fn admit_qwen30_quality_repack_artifact(
    manifest_path: impl AsRef<Path>,
    admission: &Qwen30QualityRepackAdmission,
) -> Result<Qwen30QualityRepackArtifact> {
    for (label, value) in [
        (
            "expected manifest seal",
            admission.expected_manifest_seal_sha256.as_str(),
        ),
        (
            "expected source audit seal",
            admission.expected_source_audit_seal_sha256.as_str(),
        ),
        (
            "expected revalidation seal",
            admission.expected_revalidation_seal_sha256.as_str(),
        ),
        (
            "expected selection seal",
            admission.expected_selection_seal_sha256.as_str(),
        ),
        (
            "expected source snapshot seal",
            admission.expected_source_snapshot_seal_sha256.as_str(),
        ),
        (
            "expected terminal seal",
            admission.expected_terminal_seal_sha256.as_str(),
        ),
    ] {
        if !is_sha256(value) {
            return Err(Error::Model(format!(
                "quality repack admission {label} must be lowercase SHA-256"
            )));
        }
    }
    if admission.expected_source_revision.is_empty() {
        return Err(Error::Model(
            "quality repack admission requires a protected source revision".into(),
        ));
    }
    let manifest_path = canonical_regular_path(manifest_path.as_ref(), "quality repack manifest")?;
    let root = manifest_path.parent().ok_or_else(|| {
        Error::Model("quality repack manifest has no candidate root parent".into())
    })?;
    let expected_revalidation_path = canonical_expected_regular_path(
        &admission.expected_revalidation_path,
        "protected quality source revalidation receipt",
    )?;
    let expected_terminal_path = canonical_expected_regular_path(
        &admission.expected_terminal_path,
        "protected quality terminal receipt",
    )?;
    let manifest_raw = read_regular_file(&manifest_path, "quality repack manifest")?;
    let manifest_raw_sha256 = sha256_hex(&manifest_raw);
    let manifest = parse_json_no_duplicate_keys(&manifest_raw, "quality repack manifest")?;
    let manifest_object = manifest
        .as_object()
        .ok_or_else(|| Error::Model("quality repack manifest root must be an object".into()))?;
    let manifest_seal = verify_sealed_document(&manifest, "quality repack manifest")?;
    if manifest_seal != admission.expected_manifest_seal_sha256 {
        return Err(Error::Model(
            "quality repack manifest seal differs from protected handoff binding".into(),
        ));
    }
    require_exact_string(
        manifest_object,
        "schema",
        QWEN30_QUALITY_REPACK_SCHEMA,
        "quality repack manifest",
    )?;
    require_exact_string(
        manifest_object,
        "status",
        COMPLETE_BINARY_CANDIDATE_STATUS,
        "quality repack manifest",
    )?;
    let manifest_audit_seal = required_sha256(
        manifest_object,
        "source_body_audit_seal_sha256",
        "quality repack manifest",
    )?;
    if manifest_audit_seal != admission.expected_source_audit_seal_sha256 {
        return Err(Error::Model(
            "quality repack manifest source audit seal differs from protected handoff binding"
                .into(),
        ));
    }
    require_exact_regular_path(
        manifest_object,
        "source_revalidation_receipt_path",
        &expected_revalidation_path,
        "quality repack manifest",
    )?;
    if required_sha256(
        manifest_object,
        "source_revalidation_receipt_seal_sha256",
        "quality repack manifest",
    )? != admission.expected_revalidation_seal_sha256
    {
        return Err(Error::Model(
            "quality repack manifest revalidation seal differs from protected handoff binding"
                .into(),
        ));
    }
    let receipt_raw = read_regular_file(
        &expected_revalidation_path,
        "quality source revalidation receipt",
    )?;
    let receipt =
        parse_json_no_duplicate_keys(&receipt_raw, "quality source revalidation receipt")?;
    let receipt_seal = verify_sealed_document(&receipt, "quality source revalidation receipt")?;
    if receipt_seal != admission.expected_revalidation_seal_sha256 {
        return Err(Error::Model(
            "quality source revalidation seal differs from protected handoff binding".into(),
        ));
    }
    let standard_admission = CompleteBinaryAdmission {
        model: QwenCompleteBinaryModel::Qwen30Coder,
        expected_manifest_seal_sha256: manifest_seal.clone(),
        expected_source_audit_seal_sha256: admission.expected_source_audit_seal_sha256.clone(),
        expected_source_revision: admission.expected_source_revision.clone(),
    };
    let revalidation_parent = expected_revalidation_path.parent().ok_or_else(|| {
        Error::Model("quality source revalidation receipt has no authority parent".into())
    })?;
    let source = validate_source_chain_at(
        &receipt,
        &expected_revalidation_path,
        revalidation_parent,
        &standard_admission,
        &manifest_audit_seal,
    )?;
    validate_quality_manifest_source(manifest_object, &source)?;
    let selected_organs = validate_quality_selection_and_snapshot(
        manifest_object,
        root,
        admission,
        &expected_revalidation_path,
        &sha256_hex(&receipt_raw),
    )?;

    let terminal_raw = read_regular_file(&expected_terminal_path, "quality terminal receipt")?;
    let terminal = parse_json_no_duplicate_keys(&terminal_raw, "quality terminal receipt")?;
    let terminal_seal = verify_sealed_document(&terminal, "quality terminal receipt")?;
    if terminal_seal != admission.expected_terminal_seal_sha256 {
        return Err(Error::Model(
            "quality terminal receipt seal differs from protected handoff binding".into(),
        ));
    }
    validate_quality_terminal(
        &terminal,
        &expected_terminal_path,
        &manifest_path,
        &manifest_raw_sha256,
        &manifest_seal,
        manifest_raw.len(),
        admission,
    )?;
    let terminal_object = terminal
        .as_object()
        .ok_or_else(|| Error::Model("quality terminal receipt root must be an object".into()))?;
    let terminal_binding = required_object(terminal_object, "binding", "quality terminal receipt")?;
    if required_sha256(
        terminal_binding,
        "source_body_audit_seal_sha256",
        "quality terminal receipt",
    )? != admission.expected_source_audit_seal_sha256
    {
        return Err(Error::Model(
            "quality terminal source audit seal differs from protected handoff".into(),
        ));
    }
    require_exact_regular_path(
        terminal_binding,
        "source_revalidation_receipt_path",
        &expected_revalidation_path,
        "quality terminal receipt",
    )?;
    if required_sha256(
        terminal_binding,
        "source_revalidation_receipt_seal_sha256",
        "quality terminal receipt",
    )? != receipt_seal
    {
        return Err(Error::Model(
            "quality terminal revalidation seal differs from current receipt".into(),
        ));
    }

    let rows = required_array(manifest_object, "tensors", "quality repack manifest")?;
    if rows.len() != source.weight_map.len() || rows.len() != QWEN30_QUALITY_REPACK_TENSOR_COUNT {
        return Err(Error::Model(
            "quality repack manifest does not contain every Qwen30 source tensor".into(),
        ));
    }
    let expected_selection_path = canonical_expected_regular_path(
        &admission.expected_selection_path,
        "protected quality selection receipt",
    )?;
    let (validated_rows, payload_verification_workers) =
        validate_quality_tensor_rows_bounded_parallel(
            rows,
            root,
            &source,
            &selected_organs,
            &expected_selection_path,
            &admission.expected_selection_seal_sha256,
        )?;
    let mut tensors = BTreeMap::new();
    let mut verified_payloads = BTreeMap::new();
    for (tensor, verified_payload) in validated_rows {
        let tensor_name = tensor.tensor_name.clone();
        if tensors.insert(tensor_name.clone(), tensor).is_some() {
            return Err(Error::Model(
                "quality repack manifest has a duplicate tensor name".into(),
            ));
        }
        if verified_payloads
            .insert(tensor_name, verified_payload)
            .is_some()
        {
            return Err(Error::Model(
                "quality repack manifest duplicated an immutable payload entry".into(),
            ));
        }
    }
    if tensors.keys().ne(source.weight_map.keys()) {
        return Err(Error::Model(
            "quality repack manifest tensor set differs from current source index".into(),
        ));
    }
    if verified_payloads.len() != tensors.len() {
        return Err(Error::Model(
            "quality repack admission did not retain one verified immutable payload per tensor"
                .into(),
        ));
    }
    let (source_weight_elements, tensor_payload_bytes) =
        validate_quality_ledger(manifest_object, &tensors, manifest_raw.len())?;
    let terminal_candidate =
        required_object(terminal_object, "candidate", "quality terminal receipt")?;
    if required_u64(
        terminal_candidate,
        "all_required_weight_artifact_bytes",
        "quality terminal receipt",
    )? != tensor_payload_bytes + u64::try_from(manifest_raw.len()).unwrap_or(u64::MAX)
    {
        return Err(Error::Model(
            "quality terminal byte ledger differs from scanned manifest".into(),
        ));
    }
    let exact_bpw =
        (tensor_payload_bytes + u64::try_from(manifest_raw.len()).unwrap_or(u64::MAX)) as f64 * 8.0
            / source_weight_elements as f64;
    if (required_f64(
        terminal_candidate,
        "complete_physical_bpw",
        "quality terminal receipt",
    )? - exact_bpw)
        .abs()
        > exact_bpw.abs().max(1.0) * 1e-12
        || !required_bool(
            terminal_candidate,
            "passes_storage_threshold",
            "quality terminal receipt",
        )?
    {
        return Err(Error::Model(
            "quality terminal BPW gate differs from scanned artifact".into(),
        ));
    }
    Ok(Qwen30QualityRepackArtifact {
        manifest_path,
        manifest_seal_sha256: manifest_seal,
        source_audit_path: source.source_audit_path,
        source_audit_seal_sha256: source.source_audit_seal_sha256,
        source_revision: source.source_revision,
        source_index_path: source.source_index_path,
        source_weight_elements,
        tensor_payload_bytes,
        selected_residual_organs: QWEN30_QUALITY_REPACK_SELECTED_ORGANS
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
        payload_verification_workers,
        tensors,
        verified_payloads,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    struct Fixture {
        _tempdir: TempDir,
        manifest_path: PathBuf,
        admission: CompleteBinaryAdmission,
        payload_path: PathBuf,
        source_shard_path: PathBuf,
        tensor_name: String,
    }

    fn fixture_100_values() -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&COMPLETE_BINARY_MAGIC);
        payload.extend_from_slice(&COMPLETE_BINARY_VERSION.to_le_bytes());
        payload.extend_from_slice(&(128u32).to_le_bytes());
        payload.extend_from_slice(&(1u16).to_le_bytes());
        payload.extend_from_slice(&(0u16).to_le_bytes());
        payload.extend_from_slice(&(100u64).to_le_bytes());
        payload.extend_from_slice(&(0u32).to_le_bytes());
        payload.extend_from_slice(&(100u32).to_le_bytes());
        payload.extend_from_slice(&f16::from_f32(1.5).to_bits().to_le_bytes());
        // All 128 retained bits are one, even though the logical tensor has
        // only 100 values.  The remaining tail is layout padding, not a loss
        // of accounting information.
        payload.extend_from_slice(&[0xff; 16]);
        payload
    }

    fn sealed(mut value: Value) -> Value {
        let object = value
            .as_object_mut()
            .expect("test fixtures always seal root objects");
        object.remove("seal_sha256");
        let digest = sha256_hex(&canonical_json(&Value::Object(object.clone())).unwrap());
        object.insert("seal_sha256".into(), Value::String(digest));
        value
    }

    fn seal_of(value: &Value) -> String {
        value
            .get("seal_sha256")
            .and_then(Value::as_str)
            .expect("fixture has a seal")
            .to_owned()
    }

    fn write_pretty(path: &Path, value: &Value) -> usize {
        let raw = format!("{}\n", serde_json::to_string_pretty(value).unwrap());
        fs::write(path, raw.as_bytes()).unwrap();
        raw.len()
    }

    fn fixture_identity(path: &Path) -> Value {
        let metadata = fs::symlink_metadata(path).unwrap();
        let mut identity = Map::new();
        identity.insert("bytes".into(), Value::from(metadata.len()));
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;

            identity.insert("device".into(), Value::from(metadata.dev()));
            identity.insert("inode".into(), Value::from(metadata.ino()));
            identity.insert(
                "mtime_ns".into(),
                Value::from(metadata.mtime() * 1_000_000_000 + metadata.mtime_nsec()),
            );
            identity.insert(
                "ctime_ns".into(),
                Value::from(metadata.ctime() * 1_000_000_000 + metadata.ctime_nsec()),
            );
        }
        Value::Object(identity)
    }

    fn write_manifest_with_exact_ledger(path: &Path, mut manifest: Value) -> Value {
        let mut billed = 0u64;
        for _ in 0..16 {
            let payload_bytes = manifest["tensors"][0]["artifact_bytes"].as_u64().unwrap();
            let elements = manifest["tensors"][0]["elements"].as_u64().unwrap();
            let total = payload_bytes + billed;
            manifest["complete_physical_bpw_ledger"] = json!({
                "source_weight_elements": elements,
                "tensor_payload_bytes": payload_bytes,
                "manifest_bytes_billed": billed,
                "all_required_weight_artifact_bytes": total,
                "complete_physical_bpw": (total as f64 * 8.0) / elements as f64,
                "threshold_bpw": 1.5,
                "passes_storage_threshold": (total as f64 * 8.0) / elements as f64 <= 1.5,
            });
            let sealed_manifest = sealed(manifest.clone());
            let actual = format!(
                "{}\n",
                serde_json::to_string_pretty(&sealed_manifest).unwrap()
            )
            .len();
            if u64::try_from(actual).unwrap() == billed {
                write_pretty(path, &sealed_manifest);
                return sealed_manifest;
            }
            billed = u64::try_from(actual).unwrap();
        }
        panic!("fixture manifest byte ledger did not converge");
    }

    fn complete_artifact_fixture(model: QwenCompleteBinaryModel) -> Fixture {
        let tempdir = TempDir::new().unwrap();
        let root = tempdir.path().join("complete-gravity");
        let tensor_directory = root.join("tensors");
        let source_dir = tempdir.path().join("source");
        fs::create_dir_all(&tensor_directory).unwrap();
        fs::create_dir_all(&source_dir).unwrap();

        let tensor_name = "model.layers.0.weight".to_owned();
        let shard_name = "model-00001-of-00001.safetensors";
        let source_shard_path = source_dir.join(shard_name);
        fs::write(
            &source_shard_path,
            b"small source body for identity testing",
        )
        .unwrap();
        let source_shard_sha = sha256_hex(&fs::read(&source_shard_path).unwrap());
        let index_path = source_dir.join("model.safetensors.index.json");
        let index = json!({"weight_map": {tensor_name.clone(): shard_name}});
        fs::write(&index_path, serde_json::to_vec(&index).unwrap()).unwrap();
        let weight_map = weight_map_from_index(&index, "test source index").unwrap();

        let revision = "pinned-test-revision".to_owned();
        let audit_path = tempdir.path().join("source-audit.json");
        let audit = sealed(json!({
            "schema": model.source_audit_schema(),
            "status": model.source_audit_status(),
            "source": {
                "repository": model.source_repository(),
                "revision": revision,
            },
        }));
        write_pretty(&audit_path, &audit);
        let audit_seal = seal_of(&audit);

        let mut shard_hashes = BTreeMap::new();
        shard_hashes.insert(shard_name.to_owned(), source_shard_sha.clone());
        let receipt_path = root.join("QWEN_CURRENT_SOURCE_SHARD_REVALIDATION.json");
        let receipt = sealed(json!({
            "schema": COMPLETE_BINARY_REVALIDATION_SCHEMA,
            "status": "EARNED_CURRENT_SOURCE_SHARDS_REVALIDATED",
            "source_audit_path": audit_path,
            "source_audit_document_sha256": sha256_hex(&fs::read(&audit_path).unwrap()),
            "source_audit_seal_sha256": audit_seal,
            "source_repository": model.source_repository(),
            "source_revision": revision,
            "source_model_dir": source_dir,
            "index_path": index_path,
            "index_sha256": sha256_hex(&fs::read(&index_path).unwrap()),
            "weight_map_sha256": canonical_string_map_sha256(&weight_map).unwrap(),
            "sealed_shard_hashes_sha256": canonical_string_map_sha256(&shard_hashes).unwrap(),
            "sealed_shard_count": 1,
            "shards": {
                shard_name: {
                    "expected_sha256": source_shard_sha,
                    "observed_sha256": source_shard_sha,
                    "expected_bytes": fs::metadata(&source_shard_path).unwrap().len(),
                    "file_identity": fixture_identity(&source_shard_path),
                },
            },
        }));
        write_pretty(&receipt_path, &receipt);

        let payload_path =
            tensor_directory.join(format!("{}.hq30g", sha256_hex(tensor_name.as_bytes())));
        let payload = fixture_100_values();
        fs::write(&payload_path, &payload).unwrap();
        let manifest_path = root.join("COMPLETE_BINARY_GRAVITY_CANDIDATE.json");
        let manifest = write_manifest_with_exact_ledger(
            &manifest_path,
            json!({
                "schema": model.manifest_schema(),
                "status": COMPLETE_BINARY_CANDIDATE_STATUS,
                "source_body_audit_seal_sha256": audit_seal,
                "source_revalidation_receipt_path": receipt_path,
                "source_revalidation_receipt_seal_sha256": seal_of(&receipt),
                "source": {
                    "repository": model.source_repository(),
                    "model_dir": source_dir,
                    "tensor_count": 1,
                },
                "representation": {
                    "family": "binary_sign_scale",
                    "group_size": 128,
                    "physical_direct_layout": true,
                },
                "tensors": [{
                    "tensor_name": tensor_name,
                    "source_shard": shard_name,
                    "source_shard_sha256": source_shard_sha,
                    "source_dtype": "F32",
                    "shape": [100],
                    "elements": 100,
                    "artifact_path": payload_path,
                    "artifact_bytes": payload.len(),
                    "artifact_sha256": sha256_hex(&payload),
                    "layout": {
                        "magic": "HQ30G1B1",
                        "version": COMPLETE_BINARY_VERSION,
                        "group_size": 128,
                        "sign_bit_order": "little",
                        "scale_dtype": "float16",
                    },
                }],
            }),
        );
        Fixture {
            _tempdir: tempdir,
            manifest_path,
            admission: CompleteBinaryAdmission {
                model,
                expected_manifest_seal_sha256: seal_of(&manifest),
                expected_source_audit_seal_sha256: audit_seal,
                expected_source_revision: revision,
            },
            payload_path,
            source_shard_path,
            tensor_name,
        }
    }

    #[test]
    fn parses_and_decodes_the_retained_non_aligned_tail() {
        let payload = fixture_100_values();
        let (header, decoded) = decode_complete_binary_f32(&payload).unwrap();
        assert_eq!(payload.len(), 54);
        assert_eq!(header.shape, vec![100]);
        assert_eq!(header.groups, 1);
        assert_eq!(header.payload_bytes, payload.len());
        assert_eq!(decoded.len(), 100);
        assert!(decoded.iter().all(|value| (*value - 1.5).abs() < 0.001));
    }

    #[test]
    fn rejects_theoretical_tail_size_that_does_not_match_the_fixed_layout() {
        let mut payload = fixture_100_values();
        payload.truncate(42); // Incorrectly bills only ceil(100 / 8) sign bytes.
        assert!(parse_complete_binary_header(&payload).is_err());
    }

    #[test]
    fn canonical_seal_matches_the_python_receipt_encoding() {
        let document =
            parse_json_no_duplicate_keys(br#"{"z":"\u00e9","a":[true,null,1.5]}"#, "fixture")
                .unwrap();
        assert_eq!(
            sha256_hex(&canonical_json(&document).unwrap()),
            "cb953db35eba5cc26cbbe6c14aa968dbc3f90f1b24a49761c7d25cfe76d3651b"
        );
    }

    #[test]
    fn canonical_seal_matches_python_float_exponent_spelling() {
        let document = parse_json_no_duplicate_keys(
            br#"{"a":[1e-6,1e-5,0.0001,1e20,1e21,1.0,-0.0]}"#,
            "float fixture",
        )
        .unwrap();
        assert_eq!(
            String::from_utf8(canonical_json(&document).unwrap()).unwrap(),
            r#"{"a":[1e-06,1e-05,0.0001,1e+20,1e+21,1.0,-0.0]}"#
        );
        assert_eq!(
            sha256_hex(&canonical_json(&document).unwrap()),
            "24bb183841c8123ef3a69407dd438dd4ab94f78dd0cba2751f53771c03758c43"
        );
    }

    #[test]
    fn canonical_seal_retains_python_float_roundtrip_bits() {
        // This decimal is a live-manifest-shaped value for which serde_json's
        // fast parser can land one ULP away from CPython's `json.loads`.
        // Sealing must not normalize that changed value: the manifest was
        // sealed over CPython's exact binary64 parse and `json.dumps` output.
        let document = parse_json_no_duplicate_keys(
            br#"{"metric":0.013685895904560725}"#,
            "float roundtrip fixture",
        )
        .unwrap();
        let number = document
            .get("metric")
            .and_then(Value::as_f64)
            .expect("finite metric");
        assert_eq!(number.to_bits(), 0x3f8c_0759_da9c_c58a);
        assert_eq!(
            String::from_utf8(canonical_json(&document).unwrap()).unwrap(),
            r#"{"metric":0.013685895904560725}"#
        );
        let sealed = parse_json_no_duplicate_keys(
            br#"{"metric":0.013685895904560725,"seal_sha256":"d44cba072088ee7806842dbf9fc22f38ab796d6a5229850059a353b449f9ec18"}"#,
            "Python-sealed float roundtrip fixture",
        )
        .unwrap();
        assert_eq!(
            verify_sealed_document(&sealed, "Python-sealed float roundtrip fixture").unwrap(),
            "d44cba072088ee7806842dbf9fc22f38ab796d6a5229850059a353b449f9ec18"
        );
    }

    #[test]
    fn admission_retains_immutable_payloads_and_rechecks_tamper_on_restart() {
        for model in [
            QwenCompleteBinaryModel::Qwen30Coder,
            QwenCompleteBinaryModel::Qwen80CoderNext,
        ] {
            let fixture = complete_artifact_fixture(model);
            let artifact =
                admit_complete_binary_artifact(&fixture.manifest_path, &fixture.admission)
                    .expect("sealed source-bound fixture must admit");
            assert_eq!(artifact.model, model);
            assert_eq!(artifact.tensors.len(), 1);
            assert_eq!(artifact.source_weight_elements, 100);
            assert_eq!(artifact.verified_payload_count(), 1);
            assert!(artifact.has_complete_verified_payload_cache());
            assert_eq!(
                artifact.read_tensor_payload(&fixture.tensor_name).unwrap(),
                fixture_100_values()
            );
            assert_eq!(
                artifact
                    .verified_tensor_payload(&fixture.tensor_name)
                    .unwrap()
                    .as_ref(),
                fixture_100_values().as_slice()
            );

            let mut replacement = fixture_100_values();
            replacement[40] ^= 1;
            fs::write(&fixture.payload_path, replacement).unwrap();
            // The admitted process retains the exact immutable payload it
            // scanned. A post-admission on-disk replacement cannot alter the
            // production cache; a targeted re-read remains available to
            // prove that the file is now unsafe.
            assert_eq!(
                artifact
                    .verified_tensor_payload(&fixture.tensor_name)
                    .unwrap()
                    .as_ref(),
                fixture_100_values().as_slice()
            );
            let admission_error =
                admit_complete_binary_artifact(&fixture.manifest_path, &fixture.admission)
                    .unwrap_err();
            assert!(
                admission_error.to_string().contains("SHA-256"),
                "{admission_error}"
            );
            let error = artifact
                .read_tensor_payload(&fixture.tensor_name)
                .unwrap_err();
            assert!(error.to_string().contains("SHA-256 mismatch"), "{error}");
        }
    }

    #[test]
    fn admission_rejects_a_resealed_manifest_when_geometry_disagrees_with_payload() {
        let mut fixture = complete_artifact_fixture(QwenCompleteBinaryModel::Qwen30Coder);
        let raw = fs::read(&fixture.manifest_path).unwrap();
        let mut document = parse_json_no_duplicate_keys(&raw, "fixture manifest").unwrap();
        document["tensors"][0]["shape"] = json!([99]);
        let resealed = sealed(document);
        write_pretty(&fixture.manifest_path, &resealed);
        fixture.admission.expected_manifest_seal_sha256 = seal_of(&resealed);

        let error =
            admit_complete_binary_artifact(&fixture.manifest_path, &fixture.admission).unwrap_err();
        assert!(error.to_string().contains("elements"), "{error}");
    }

    #[test]
    fn admission_rejects_current_source_identity_drift_and_duplicate_json_keys() {
        let fixture = complete_artifact_fixture(QwenCompleteBinaryModel::Qwen80CoderNext);
        fs::write(&fixture.source_shard_path, b"changed source body").unwrap();
        let error =
            admit_complete_binary_artifact(&fixture.manifest_path, &fixture.admission).unwrap_err();
        assert!(error.to_string().contains("identity"), "{error}");
        assert!(parse_json_no_duplicate_keys(br#"{"x":1,"x":2}"#, "duplicate").is_err());
    }

    #[test]
    fn admission_rejects_missing_payloads_and_unpinned_manifest_seals() {
        let fixture = complete_artifact_fixture(QwenCompleteBinaryModel::Qwen30Coder);
        fs::remove_file(&fixture.payload_path).unwrap();
        let error =
            admit_complete_binary_artifact(&fixture.manifest_path, &fixture.admission).unwrap_err();
        assert!(error.to_string().contains("cannot stat"), "{error}");

        let mut fixture = complete_artifact_fixture(QwenCompleteBinaryModel::Qwen80CoderNext);
        fixture.admission.expected_manifest_seal_sha256 = "0".repeat(64);
        let error =
            admit_complete_binary_artifact(&fixture.manifest_path, &fixture.admission).unwrap_err();
        assert!(
            error.to_string().contains("protected admission binding"),
            "{error}"
        );
    }

    fn quality_residual_fixture(indices: &[u32]) -> Vec<u8> {
        let base = fixture_100_values();
        let mut payload = Vec::new();
        payload.extend_from_slice(&QWEN30_QUALITY_REPACK_MAGIC);
        payload.extend_from_slice(&QWEN30_QUALITY_REPACK_VERSION.to_le_bytes());
        payload.extend_from_slice(&COMPLETE_BINARY_VERSION.to_le_bytes());
        payload.extend_from_slice(&(128u32).to_le_bytes());
        payload.extend_from_slice(&(1u32).to_le_bytes());
        payload.extend_from_slice(&(base.len() as u32).to_le_bytes());
        payload.extend_from_slice(&(indices.len() as u32).to_le_bytes());
        payload.extend_from_slice(&(100u32).to_le_bytes());
        payload.extend_from_slice(&base);
        for index in indices {
            payload.extend_from_slice(&index.to_le_bytes());
        }
        for _ in indices {
            payload.extend_from_slice(&f16::from_f32(0.25).to_bits().to_le_bytes());
        }
        payload
    }

    fn direct_matrix_fixture_2x4() -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&COMPLETE_BINARY_MAGIC);
        payload.extend_from_slice(&COMPLETE_BINARY_VERSION.to_le_bytes());
        payload.extend_from_slice(&(128u32).to_le_bytes());
        payload.extend_from_slice(&(2u16).to_le_bytes());
        payload.extend_from_slice(&(0u16).to_le_bytes());
        payload.extend_from_slice(&(8u64).to_le_bytes());
        payload.extend_from_slice(&(0u32).to_le_bytes());
        payload.extend_from_slice(&(2u32).to_le_bytes());
        payload.extend_from_slice(&(4u32).to_le_bytes());
        payload.extend_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        // + - + - | + - + - followed by unused padded signs.
        payload.push(0b0101_0101);
        payload.extend_from_slice(&[0; 15]);
        payload
    }

    fn quality_residual_matrix_fixture() -> Vec<u8> {
        let base = direct_matrix_fixture_2x4();
        let indices = [1u32, 6u32];
        let residuals = [0.25f32, -0.5f32];
        let mut payload = Vec::new();
        payload.extend_from_slice(&QWEN30_QUALITY_REPACK_MAGIC);
        payload.extend_from_slice(&QWEN30_QUALITY_REPACK_VERSION.to_le_bytes());
        payload.extend_from_slice(&COMPLETE_BINARY_VERSION.to_le_bytes());
        payload.extend_from_slice(&(128u32).to_le_bytes());
        payload.extend_from_slice(&(2u32).to_le_bytes());
        payload.extend_from_slice(&(base.len() as u32).to_le_bytes());
        payload.extend_from_slice(&(indices.len() as u32).to_le_bytes());
        payload.extend_from_slice(&(2u32).to_le_bytes());
        payload.extend_from_slice(&(4u32).to_le_bytes());
        payload.extend_from_slice(&base);
        for index in indices {
            payload.extend_from_slice(&index.to_le_bytes());
        }
        for residual in residuals {
            payload.extend_from_slice(&f16::from_f32(residual).to_bits().to_le_bytes());
        }
        payload
    }

    #[test]
    fn quality_residual_parser_validates_embedded_direct_geometry_and_index_order() {
        let payload = quality_residual_fixture(&[1, 99]);
        let header = parse_qwen30_quality_residual_header(&payload).unwrap();
        assert_eq!(header.shape, vec![100]);
        assert_eq!(header.base.elements, 100);
        assert_eq!(header.residual_count, 2);
        assert_eq!(header.payload_bytes, payload.len());

        let error =
            parse_qwen30_quality_residual_header(&quality_residual_fixture(&[9, 9])).unwrap_err();
        assert!(error.to_string().contains("sorted"), "{error}");
        let error =
            parse_qwen30_quality_residual_header(&quality_residual_fixture(&[100])).unwrap_err();
        assert!(error.to_string().contains("bounds"), "{error}");
    }

    #[test]
    fn quality_residual_cpu_decoder_requires_hq30gr2_and_applies_exact_entries() {
        let payload = quality_residual_fixture(&[1, 99]);
        let (header, entries) = qwen30_quality_residual_entries(&payload).unwrap();
        let base_start = header.base_offset;
        let base_end = base_start + header.base_payload_bytes;
        let (_, base) = decode_complete_binary_f32(&payload[base_start..base_end]).unwrap();
        let (_, decoded) = decode_qwen30_quality_residual_f32(&payload).unwrap();
        assert_eq!(entries, vec![(1, 0.25), (99, 0.25)]);
        assert_eq!(decoded.len(), base.len());
        assert_eq!(decoded[1], base[1] + 0.25);
        assert_eq!(decoded[99], base[99] + 0.25);
        assert_eq!(decoded[0], base[0]);

        // There is deliberately no implicit direct fallback for a selected
        // HQ30GR2 organ, and equally no residual path for the control body.
        assert!(decode_complete_binary_f32(&payload).is_err());
        assert!(decode_qwen30_quality_residual_f32(&payload[base_start..base_end]).is_err());
    }

    #[test]
    fn quality_residual_packed_matvec_is_base_plus_sparse_corrections_without_dense_fallback() {
        let base = direct_matrix_fixture_2x4();
        let candidate = quality_residual_matrix_fixture();
        let input = [2.0, 3.0, 5.0, 7.0];
        let (_, base_output) = complete_binary_matvec_f64(&base, &input).unwrap();
        let (_, candidate_output) = qwen30_quality_residual_matvec_f64(&candidate, &input).unwrap();
        assert_eq!(base_output, vec![-3.0, -3.0]);
        assert_eq!(candidate_output, vec![-2.25, -5.5]);
        assert!(complete_binary_matvec_f64(&candidate, &input).is_err());
        assert!(qwen30_quality_residual_matvec_f64(&base, &input).is_err());

        let (_, decoded) = decode_qwen30_quality_residual_f32(&candidate).unwrap();
        let dense = decoded
            .chunks_exact(4)
            .map(|row| {
                row.iter()
                    .zip(input)
                    .map(|(weight, value)| f64::from(*weight) * value)
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        assert_eq!(candidate_output, dense);
    }

    #[test]
    fn quality_repack_artifact_retains_only_admission_verified_payload_snapshots() {
        let payload: Arc<[u8]> = Arc::from(direct_matrix_fixture_2x4());
        let header = parse_complete_binary_header(&payload).unwrap();
        let name = "model.layers.0.mlp.experts.0.down_proj.weight".to_owned();
        let tensor = Qwen30QualityRepackTensor {
            tensor_name: name.clone(),
            source_shard: "fixture.safetensors".to_owned(),
            source_shard_sha256: "0".repeat(64),
            source_dtype: "BF16".to_owned(),
            artifact_path: PathBuf::from("fixture.hq30g"),
            artifact_sha256: "1".repeat(64),
            artifact_bytes: payload.len(),
            elements: header.elements,
            layout: Qwen30QualityRepackTensorLayout::Direct(header),
        };
        let mut tensors = BTreeMap::new();
        tensors.insert(name.clone(), tensor);
        let mut verified_payloads = BTreeMap::new();
        verified_payloads.insert(name.clone(), payload.clone());
        let artifact = Qwen30QualityRepackArtifact {
            manifest_path: PathBuf::from("candidate.json"),
            manifest_seal_sha256: "2".repeat(64),
            source_audit_path: PathBuf::from("source-audit.json"),
            source_audit_seal_sha256: "3".repeat(64),
            source_revision: "fixture-revision".to_owned(),
            source_index_path: PathBuf::from("model.safetensors.index.json"),
            source_weight_elements: 8,
            tensor_payload_bytes: u64::try_from(payload.len()).unwrap(),
            selected_residual_organs: vec!["selected.gate".to_owned(), "selected.up".to_owned()],
            payload_verification_workers: 1,
            tensors,
            verified_payloads,
        };
        assert_eq!(artifact.verified_payload_count(), 1);
        assert!(artifact.has_complete_verified_payload_cache());
        assert_eq!(
            &*artifact.verified_tensor_payload(&name).unwrap(),
            &*payload
        );
        assert!(matches!(
            artifact.verified_typed_tensor(&name).unwrap(),
            Qwen30QualityRepackVerifiedTensor::Direct { .. }
        ));
        assert!(artifact.verified_tensor_payload("missing.tensor").is_err());
    }

    #[test]
    fn quality_payload_verification_lanes_are_bounded_by_shard_and_deterministic() {
        let rows = vec![
            json!({"source_shard": "b.safetensors"}),
            json!({"source_shard": "a.safetensors"}),
            json!({"source_shard": "c.safetensors"}),
            json!({"source_shard": "b.safetensors"}),
            json!({"source_shard": "d.safetensors"}),
            json!({"source_shard": "e.safetensors"}),
        ];
        let (lanes, workers) = quality_payload_verification_lanes(&rows);
        assert_eq!(workers, QWEN30_QUALITY_REPACK_MAX_PAYLOAD_VERIFY_WORKERS);
        // Source shards are sorted before a fixed round-robin lane assignment:
        // a/e -> lane 0, b -> lane 1, c -> lane 2, d -> lane 3.
        assert_eq!(lanes, vec![vec![1, 5], vec![0, 3], vec![2], vec![4]]);
        let (repeat, repeat_workers) = quality_payload_verification_lanes(&rows);
        assert_eq!(repeat_workers, workers);
        assert_eq!(repeat, lanes);
        let mut covered = lanes.into_iter().flatten().collect::<Vec<_>>();
        covered.sort_unstable();
        assert_eq!(covered, (0..rows.len()).collect::<Vec<_>>());
    }
}

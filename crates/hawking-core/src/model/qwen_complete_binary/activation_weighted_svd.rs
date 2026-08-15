//! Mixed HQ30G1B1 + HGRAVS01 reader for the activation-weighted SVD Qwen30 candidate.
//!
//! Selected expert organs are physical `HGRAVS01` low-rank factors
//! (`W ≈ L @ R` with uniform group-quantized factors). Unchanged tensors remain
//! direct `HQ30G1B1`. Production execution must apply the factors as two matvecs
//! and must not expand `L @ R` into a dense weight matrix on a token path.

use super::{
    absolute_path, admission_warm_receipt, canonical_directory, canonical_expected_regular_path,
    canonical_regular_path, expected_tensor_path, is_sha256, model_error, parse_complete_binary_header,
    parse_json_no_duplicate_keys, quality_payload_verification_lanes, read_regular_file,
    require_exact_regular_path, require_exact_string, require_safe_filename, required_array,
    required_bool, required_f64, required_object, required_sha256, required_string, required_u64,
    sha256_hex, validate_source_chain_at, verify_sealed_document, CompleteBinaryAdmission,
    CompleteBinaryArtifact, CompleteBinaryHeader, CompleteBinaryTensor, QwenCompleteBinaryModel,
    COMPLETE_BINARY_CANDIDATE_STATUS, COMPLETE_BINARY_VERSION,
};
use crate::{Error, Result};
use half::f16;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

/// Process-local accumulators for fine-grained cold-payload buckets under
/// `HAWKING_STARTUP_TIMING=1`. Reset at the start of each HGRAVS admission.
static AW_PAYLOAD_READ_MS: AtomicU64 = AtomicU64::new(0);
static AW_PAYLOAD_SHA256_MS: AtomicU64 = AtomicU64::new(0);
static AW_PAYLOAD_LAYOUT_MS: AtomicU64 = AtomicU64::new(0);

fn aw_payload_timing_reset() {
    AW_PAYLOAD_READ_MS.store(0, Ordering::Relaxed);
    AW_PAYLOAD_SHA256_MS.store(0, Ordering::Relaxed);
    AW_PAYLOAD_LAYOUT_MS.store(0, Ordering::Relaxed);
}

fn aw_payload_timing_add(target: &AtomicU64, start: Instant) {
    let ms = crate::startup_timing::duration_ms(start.elapsed());
    target.fetch_add(ms, Ordering::Relaxed);
}

fn aw_payload_timing_flush() {
    crate::startup_timing::record_ms(
        "admit_payload_file_read",
        AW_PAYLOAD_READ_MS.load(Ordering::Relaxed),
    );
    crate::startup_timing::record_ms(
        "admit_payload_sha256",
        AW_PAYLOAD_SHA256_MS.load(Ordering::Relaxed),
    );
    crate::startup_timing::record_ms(
        "admit_payload_layout_geometry",
        AW_PAYLOAD_LAYOUT_MS.load(Ordering::Relaxed),
    );
}

pub const HGRAVS01_MAGIC: [u8; 8] = *b"HGRAVS01";
pub const HGRAVS01_MAGIC_TEXT: &str = "HGRAVS01";
pub const HGRAVS01_SCHEMA: &str = "hawking.gravity.activation_weighted_svd_low_rank.v1";
pub const HGRAVS01_REPRESENTATION: &str = "activation_weighted_svd_low_rank_q";
pub const HGRAVS01_UNIFORM_SCHEMA: &str = "hawking.gravity.uniform_group.v1";

pub const QWEN30_ACTIVATION_WEIGHTED_SVD_SCHEMA: &str =
    "hawking.ascension.qwen30_activation_weighted_svd_repack_candidate.v1";
pub const QWEN30_ACTIVATION_WEIGHTED_SVD_SELECTION_SCHEMA: &str =
    "hawking.ascension.qwen30_activation_weighted_svd_selection.v1";
pub const QWEN30_ACTIVATION_WEIGHTED_SVD_SNAPSHOT_SCHEMA: &str =
    "hawking.ascension.qwen30_activation_weighted_svd_source_snapshot.v1";
pub const QWEN30_ACTIVATION_WEIGHTED_SVD_TERMINAL_SCHEMA: &str =
    "hawking.ascension.complete_binary_terminal_status.v1";
pub const QWEN30_ACTIVATION_WEIGHTED_SVD_BRANCH_ID: &str = "qwen30-activation-weighted-svd-v1";
pub const QWEN30_ACTIVATION_WEIGHTED_SVD_MODEL_ID: &str =
    "Qwen3-Coder-30B-A3B-Instruct-activation-weighted-svd-v1";
pub const QWEN30_ACTIVATION_WEIGHTED_SVD_ARTIFACT_PREFIX: &str = "QWEN30_ACTIVATION_WEIGHTED_SVD_V1";
pub const QWEN30_ACTIVATION_WEIGHTED_SVD_TENSOR_COUNT: usize = 18_867;
const QWEN30_ACTIVATION_WEIGHTED_SVD_MAX_PAYLOAD_VERIFY_WORKERS: usize = 4;
pub const QWEN30_ACTIVATION_WEIGHTED_SVD_PAYLOAD_VERIFY_MODE: &str =
    "bounded_parallel_source_shard_lanes_ordered_reconciliation_v1";

/// Geometry for one uniform-quantized low-rank factor body (no outer magic).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Hgravs01UniformFactor {
    pub shape: Vec<usize>,
    pub elements: usize,
    pub bits: u8,
    pub group_size: usize,
    pub groups: usize,
    pub scale_bytes: usize,
    pub code_bytes: usize,
    pub retained_padding_elements: usize,
}

/// Validated HGRAVS01 header geometry. Factors remain separate; callers must
/// not treat this as permission to materialize dense `L @ R` on a token path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Hgravs01Header {
    pub shape: Vec<usize>,
    pub matrix_shape: [usize; 2],
    pub elements: usize,
    pub rank: usize,
    pub factor_bits: u8,
    pub factor_group_size: usize,
    pub left: Hgravs01UniformFactor,
    pub right: Hgravs01UniformFactor,
    pub left_body_offset: usize,
    pub right_body_offset: usize,
    pub left_body_bytes: usize,
    pub right_body_bytes: usize,
    pub payload_bytes: usize,
    pub activation_capture_sha256: String,
}

/// Protected bindings for the activation-weighted SVD candidate. Every seal is
/// supplied by the sealed handoff; a self-consistent replacement is refused.
/// The activation-capture SHA is operator-supplied so a forged manifest cannot
/// name its own capture and pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen30ActivationWeightedSvdAdmission {
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
    pub expected_activation_capture_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Qwen30ActivationWeightedTensorLayout {
    Direct(CompleteBinaryHeader),
    ActivationWeightedSvd(Hgravs01Header),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen30ActivationWeightedTensor {
    pub tensor_name: String,
    pub source_shard: String,
    pub source_shard_sha256: String,
    pub source_dtype: String,
    pub artifact_path: PathBuf,
    pub artifact_sha256: String,
    pub artifact_bytes: usize,
    pub elements: usize,
    pub layout: Qwen30ActivationWeightedTensorLayout,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Qwen30ActivationWeightedVerifiedTensor {
    Direct {
        header: CompleteBinaryHeader,
        payload: Arc<[u8]>,
    },
    ActivationWeightedSvd {
        header: Hgravs01Header,
        payload: Arc<[u8]>,
    },
}

/// Fully scanned mixed catalog. Admission only until a typed runtime opts in.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen30ActivationWeightedSvdArtifact {
    pub manifest_path: PathBuf,
    pub manifest_seal_sha256: String,
    pub source_audit_path: PathBuf,
    pub source_audit_seal_sha256: String,
    pub source_revision: String,
    pub source_index_path: PathBuf,
    pub source_weight_elements: u64,
    pub tensor_payload_bytes: u64,
    pub selected_hgravs_organs: Vec<String>,
    pub payload_verification_workers: usize,
    pub tensors: BTreeMap<String, Qwen30ActivationWeightedTensor>,
    pub(crate) verified_payloads: BTreeMap<String, Arc<[u8]>>,
}

impl Qwen30ActivationWeightedSvdArtifact {
    pub fn tensor(&self, tensor_name: &str) -> Result<&Qwen30ActivationWeightedTensor> {
        self.tensors.get(tensor_name).ok_or_else(|| {
            Error::Model(format!(
                "activation-weighted SVD artifact has no admitted tensor {tensor_name:?}"
            ))
        })
    }

    pub fn verified_tensor_payload(&self, tensor_name: &str) -> Result<Arc<[u8]>> {
        self.tensor(tensor_name)?;
        self.verified_payloads
            .get(tensor_name)
            .cloned()
            .ok_or_else(|| {
                Error::Model(format!(
                    "activation-weighted SVD artifact has no admission-verified immutable payload for {tensor_name:?}"
                ))
            })
    }

    pub fn verified_typed_tensor(
        &self,
        tensor_name: &str,
    ) -> Result<Qwen30ActivationWeightedVerifiedTensor> {
        let tensor = self.tensor(tensor_name)?;
        let payload = self.verified_tensor_payload(tensor_name)?;
        match &tensor.layout {
            Qwen30ActivationWeightedTensorLayout::Direct(expected) => {
                let observed = parse_complete_binary_header(payload.as_ref())?;
                if &observed != expected {
                    return Err(Error::Model(format!(
                        "activation-weighted direct tensor {tensor_name:?} snapshot header differs from admission"
                    )));
                }
                Ok(Qwen30ActivationWeightedVerifiedTensor::Direct {
                    header: observed,
                    payload,
                })
            }
            Qwen30ActivationWeightedTensorLayout::ActivationWeightedSvd(expected) => {
                let observed = parse_hgravs01_header(payload.as_ref())?;
                if &observed != expected {
                    return Err(Error::Model(format!(
                        "activation-weighted HGRAVS01 tensor {tensor_name:?} snapshot header differs from admission"
                    )));
                }
                Ok(Qwen30ActivationWeightedVerifiedTensor::ActivationWeightedSvd {
                    header: observed,
                    payload,
                })
            }
        }
    }

    pub fn verified_payload_count(&self) -> usize {
        self.verified_payloads.len()
    }

    pub fn has_complete_verified_payload_cache(&self) -> bool {
        self.verified_payloads.len() == self.tensors.len()
            && self
                .tensors
                .keys()
                .all(|name| self.verified_payloads.contains_key(name))
    }

    /// Direct-only view for the existing complete-native graph. HGRAVS01 organs
    /// are deliberately omitted so a mistaken direct packed loader cannot
    /// decode them as HQ30G1B1. The runtime must route those names through the
    /// native low-rank path.
    pub(crate) fn direct_base_view_for_runtime(&self) -> Result<CompleteBinaryArtifact> {
        let mut tensors = BTreeMap::new();
        let mut verified_payloads = BTreeMap::new();
        for (name, original) in &self.tensors {
            let Qwen30ActivationWeightedTensorLayout::Direct(header) = &original.layout else {
                continue;
            };
            let payload = self.verified_tensor_payload(name)?;
            let observed = parse_complete_binary_header(payload.as_ref())?;
            if &observed != header {
                return Err(Error::Model(format!(
                    "activation-weighted direct organ {name:?} drifted from admission"
                )));
            }
            tensors.insert(
                name.clone(),
                CompleteBinaryTensor {
                    tensor_name: name.clone(),
                    source_shard: original.source_shard.clone(),
                    source_shard_sha256: original.source_shard_sha256.clone(),
                    source_dtype: original.source_dtype.clone(),
                    artifact_path: original.artifact_path.clone(),
                    artifact_sha256: original.artifact_sha256.clone(),
                    header: observed,
                },
            );
            verified_payloads.insert(name.clone(), payload);
        }
        if tensors.len() + self.selected_hgravs_organs.len() != QWEN30_ACTIVATION_WEIGHTED_SVD_TENSOR_COUNT
            || verified_payloads.len() != tensors.len()
        {
            return Err(Error::Model(
                "activation-weighted direct-base view does not retain every non-HGRAVS organ".into(),
            ));
        }
        Ok(CompleteBinaryArtifact {
            model: QwenCompleteBinaryModel::Qwen30Coder,
            manifest_path: self.manifest_path.clone(),
            manifest_seal_sha256: self.manifest_seal_sha256.clone(),
            source_audit_path: self.source_audit_path.clone(),
            source_audit_seal_sha256: self.source_audit_seal_sha256.clone(),
            source_revision: self.source_revision.clone(),
            source_index_path: self.source_index_path.clone(),
            source_weight_elements: self.source_weight_elements,
            tensor_payload_bytes: self.tensor_payload_bytes,
            tensors,
            verified_payloads,
        })
    }
}

fn json_usize(value: &Value, label: &str) -> Result<usize> {
    let number = value
        .as_u64()
        .ok_or_else(|| Error::Model(format!("{label} must be an unsigned integer")))?;
    usize::try_from(number)
        .map_err(|_| Error::Model(format!("{label} does not fit this platform")))
}

fn json_shape(value: &Value, label: &str) -> Result<Vec<usize>> {
    let values = value
        .as_array()
        .ok_or_else(|| Error::Model(format!("{label} must be an array")))?;
    if values.is_empty() {
        return Err(Error::Model(format!("{label} must not be empty")));
    }
    values
        .iter()
        .enumerate()
        .map(|(index, item)| json_usize(item, &format!("{label}[{index}]")))
        .collect()
}

fn packed_byte_count(count: usize, bits: u8) -> Result<usize> {
    if bits == 0 || bits > 8 {
        return Err(Error::Model(
            "HGRAVS01 uniform factor bits must be in 1..=8".into(),
        ));
    }
    let bit_count = count
        .checked_mul(usize::from(bits))
        .ok_or_else(|| Error::Model("HGRAVS01 packed bit count overflows".into()))?;
    Ok(bit_count.div_ceil(8))
}

fn parse_uniform_factor(meta: &Map<String, Value>, label: &str) -> Result<Hgravs01UniformFactor> {
    require_exact_string(meta, "schema", HGRAVS01_UNIFORM_SCHEMA, label)?;
    let shape = json_shape(
        meta.get("shape")
            .ok_or_else(|| model_error(label, "uniform factor lacks shape"))?,
        &format!("{label} shape"),
    )?;
    let elements = required_u64(meta, "elements", label)? as usize;
    let shape_elements = shape.iter().try_fold(1usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or_else(|| model_error(label, "uniform factor shape product overflows"))
    })?;
    if elements == 0 || elements != shape_elements {
        return Err(model_error(
            label,
            "uniform factor elements disagree with shape product",
        ));
    }
    let bits = u8::try_from(required_u64(meta, "bits", label)?).map_err(|_| {
        model_error(label, "uniform factor bits do not fit u8")
    })?;
    if bits < 2 || bits > 8 {
        return Err(model_error(label, "uniform factor bits must be in 2..=8"));
    }
    let group_size = required_u64(meta, "group_size", label)? as usize;
    if group_size == 0 {
        return Err(model_error(label, "uniform factor group_size must be positive"));
    }
    let groups = required_u64(meta, "groups", label)? as usize;
    let expected_groups = elements.div_ceil(group_size);
    if groups != expected_groups {
        return Err(model_error(
            label,
            "uniform factor groups disagree with elements/group_size",
        ));
    }
    let scale_bytes = required_u64(meta, "scale_bytes", label)? as usize;
    let code_bytes = required_u64(meta, "code_bytes", label)? as usize;
    if scale_bytes != groups * 2 {
        return Err(model_error(
            label,
            "uniform factor scale_bytes disagree with groups",
        ));
    }
    let expected_code = packed_byte_count(groups * group_size, bits)?;
    if code_bytes != expected_code {
        return Err(model_error(
            label,
            "uniform factor code_bytes disagree with packed geometry",
        ));
    }
    let retained_padding_elements = required_u64(meta, "retained_padding_elements", label)? as usize;
    if retained_padding_elements != groups * group_size - elements {
        return Err(model_error(
            label,
            "uniform factor retained padding disagrees with group pad",
        ));
    }
    require_exact_string(meta, "scale_dtype", "float16", label)?;
    Ok(Hgravs01UniformFactor {
        shape,
        elements,
        bits,
        group_size,
        groups,
        scale_bytes,
        code_bytes,
        retained_padding_elements,
    })
}

fn unpack_unsigned(body: &[u8], count: usize, bits: u8) -> Result<Vec<u8>> {
    let expected = packed_byte_count(count, bits)?;
    if body.len() != expected {
        return Err(Error::Model(
            "HGRAVS01 uniform codes have the wrong physical byte length".into(),
        ));
    }
    let mut bits_out = Vec::with_capacity(count * usize::from(bits));
    for &byte in body {
        for bit in 0..8u8 {
            bits_out.push((byte >> bit) & 1);
        }
    }
    bits_out.truncate(count * usize::from(bits));
    let mut codes = Vec::with_capacity(count);
    for index in 0..count {
        let mut value = 0u16;
        for bit in 0..bits {
            let bit_value = bits_out[index * usize::from(bits) + usize::from(bit)];
            value |= u16::from(bit_value) << bit;
        }
        codes.push(u8::try_from(value).map_err(|_| {
            Error::Model("HGRAVS01 uniform code exceeds u8 after unpack".into())
        })?);
    }
    Ok(codes)
}

/// Decode one uniform factor body to f32 without constructing dense `W`.
pub fn decode_hgravs01_uniform_factor_f32(
    factor: &Hgravs01UniformFactor,
    body: &[u8],
) -> Result<Vec<f32>> {
    if body.len() != factor.scale_bytes + factor.code_bytes {
        return Err(Error::Model(
            "HGRAVS01 uniform factor body length disagrees with its ledger".into(),
        ));
    }
    let scales = &body[..factor.scale_bytes];
    let codes = unpack_unsigned(&body[factor.scale_bytes..], factor.groups * factor.group_size, factor.bits)?;
    let bound = (1u16 << (factor.bits - 1)) - 1;
    let mut values = Vec::with_capacity(factor.elements);
    for element in 0..factor.elements {
        let group = element / factor.group_size;
        let scale = f16::from_bits(u16::from_le_bytes([
            scales[group * 2],
            scales[group * 2 + 1],
        ]))
        .to_f32();
        if !scale.is_finite() {
            return Err(Error::Model(format!(
                "HGRAVS01 uniform factor scale for group {group} is not finite"
            )));
        }
        let unsigned = u16::from(codes[element]);
        let signed = i16::try_from(unsigned).unwrap_or(i16::MAX) - i16::try_from(bound).unwrap_or(0);
        values.push(f32::from(signed) * scale);
    }
    Ok(values)
}

/// Parse and validate one HGRAVS01 activation-weighted SVD payload.
pub fn parse_hgravs01_header(payload: &[u8]) -> Result<Hgravs01Header> {
    if payload.len() < 12 || payload[..8] != HGRAVS01_MAGIC {
        return Err(Error::Model(
            "activation-weighted SVD magic does not match HGRAVS01".into(),
        ));
    }
    let header_len = u32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]) as usize;
    let body_offset = 12usize
        .checked_add(header_len)
        .ok_or_else(|| Error::Model("HGRAVS01 header length overflows".into()))?;
    if body_offset > payload.len() {
        return Err(Error::Model(
            "HGRAVS01 header length exceeds physical payload".into(),
        ));
    }
    let header_bytes = &payload[12..body_offset];
    let header_json = parse_json_no_duplicate_keys(header_bytes, "HGRAVS01 header JSON")?;
    let header = header_json
        .as_object()
        .ok_or_else(|| Error::Model("HGRAVS01 header JSON root must be an object".into()))?;
    require_exact_string(header, "schema", HGRAVS01_SCHEMA, "HGRAVS01 header")?;
    require_exact_string(
        header,
        "representation",
        HGRAVS01_REPRESENTATION,
        "HGRAVS01 header",
    )?;
    let shape = json_shape(
        header
            .get("shape")
            .ok_or_else(|| Error::Model("HGRAVS01 header lacks shape".into()))?,
        "HGRAVS01 shape",
    )?;
    let matrix_shape = json_shape(
        header
            .get("matrix_shape")
            .ok_or_else(|| Error::Model("HGRAVS01 header lacks matrix_shape".into()))?,
        "HGRAVS01 matrix_shape",
    )?;
    if matrix_shape.len() != 2 {
        return Err(Error::Model(
            "HGRAVS01 matrix_shape must be rank-2".into(),
        ));
    }
    let matrix_shape = [matrix_shape[0], matrix_shape[1]];
    let elements = required_u64(header, "elements", "HGRAVS01 header")? as usize;
    let shape_elements = shape.iter().try_fold(1usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or_else(|| Error::Model("HGRAVS01 shape product overflows".into()))
    })?;
    if elements == 0 || elements != shape_elements || elements != matrix_shape[0] * matrix_shape[1] {
        return Err(Error::Model(
            "HGRAVS01 elements disagree with shape/matrix_shape".into(),
        ));
    }
    let rank = required_u64(header, "rank", "HGRAVS01 header")? as usize;
    if rank == 0 {
        return Err(Error::Model("HGRAVS01 rank must be positive".into()));
    }
    let factor_bits = u8::try_from(required_u64(header, "factor_bits", "HGRAVS01 header")?)
        .map_err(|_| Error::Model("HGRAVS01 factor_bits do not fit u8".into()))?;
    let factor_group_size =
        required_u64(header, "factor_group_size", "HGRAVS01 header")? as usize;
    let left = parse_uniform_factor(
        required_object(header, "left", "HGRAVS01 header")?,
        "HGRAVS01 left factor",
    )?;
    let right = parse_uniform_factor(
        required_object(header, "right", "HGRAVS01 header")?,
        "HGRAVS01 right factor",
    )?;
    if left.bits != factor_bits
        || right.bits != factor_bits
        || left.group_size != factor_group_size
        || right.group_size != factor_group_size
    {
        return Err(Error::Model(
            "HGRAVS01 factor bits/group_size disagree with envelope".into(),
        ));
    }
    if left.shape != [matrix_shape[0], rank] || right.shape != [rank, matrix_shape[1]] {
        return Err(Error::Model(
            "HGRAVS01 factor shapes disagree with matrix_shape/rank".into(),
        ));
    }
    let left_body_bytes = required_u64(header, "left_body_bytes", "HGRAVS01 header")? as usize;
    let right_body_bytes = required_u64(header, "right_body_bytes", "HGRAVS01 header")? as usize;
    if left_body_bytes != left.scale_bytes + left.code_bytes
        || right_body_bytes != right.scale_bytes + right.code_bytes
    {
        return Err(Error::Model(
            "HGRAVS01 factor body ledgers disagree with uniform geometry".into(),
        ));
    }
    let body = &payload[body_offset..];
    if left_body_bytes
        .checked_add(right_body_bytes)
        .ok_or_else(|| Error::Model("HGRAVS01 body ledger overflows".into()))?
        != body.len()
    {
        return Err(Error::Model(
            "HGRAVS01 physical body bytes disagree with factor ledgers".into(),
        ));
    }
    let capture = required_object(header, "activation_capture", "HGRAVS01 header")?;
    let activation_capture_sha256 =
        required_sha256(capture, "sha256", "HGRAVS01 activation_capture")?;
    if required_string(capture, "fit_kind", "HGRAVS01 activation_capture")?
        != "real_routed_activation_capture"
    {
        return Err(Error::Model(
            "HGRAVS01 refuses synthetic activation capture bindings".into(),
        ));
    }
    // Touch both bodies through the uniform decoder so a corrupt scale/code is
    // refused at parse time rather than later on a token path.
    let _ = decode_hgravs01_uniform_factor_f32(&left, &body[..left_body_bytes])?;
    let _ = decode_hgravs01_uniform_factor_f32(
        &right,
        &body[left_body_bytes..left_body_bytes + right_body_bytes],
    )?;
    Ok(Hgravs01Header {
        shape,
        matrix_shape,
        elements,
        rank,
        factor_bits,
        factor_group_size,
        left,
        right,
        left_body_offset: body_offset,
        right_body_offset: body_offset + left_body_bytes,
        left_body_bytes,
        right_body_bytes,
        payload_bytes: payload.len(),
        activation_capture_sha256,
    })
}

/// Decode HGRAVS01 factors to host f32 without forming dense `W`.
pub fn decode_hgravs01_factors_f32(
    payload: &[u8],
) -> Result<(Hgravs01Header, Vec<f32>, Vec<f32>)> {
    let header = parse_hgravs01_header(payload)?;
    let left = decode_hgravs01_uniform_factor_f32(
        &header.left,
        &payload[header.left_body_offset..header.right_body_offset],
    )?;
    let right = decode_hgravs01_uniform_factor_f32(
        &header.right,
        &payload[header.right_body_offset..header.payload_bytes],
    )?;
    Ok((header, left, right))
}

/// Native packed low-rank matvec: `y = L @ (R @ x)` without dense reconstruction.
pub fn hgravs01_matvec_f64(payload: &[u8], input: &[f64]) -> Result<(Hgravs01Header, Vec<f64>)> {
    let (header, left, right) = decode_hgravs01_factors_f32(payload)?;
    let rows = header.matrix_shape[0];
    let cols = header.matrix_shape[1];
    let rank = header.rank;
    if header.shape.len() != 2 {
        return Err(Error::Model(
            "HGRAVS01 packed matvec requires a rank-2 tensor".into(),
        ));
    }
    if input.len() != cols || input.iter().any(|value| !value.is_finite()) {
        return Err(Error::Model(
            "HGRAVS01 packed matvec input differs from the finite tensor column count".into(),
        ));
    }
    if left.len() != rows * rank || right.len() != rank * cols {
        return Err(Error::Model(
            "HGRAVS01 packed matvec factor element counts disagree with geometry".into(),
        ));
    }
    let mut mid = vec![0.0f64; rank];
    for r in 0..rank {
        let mut sum = 0.0f64;
        let row = &right[r * cols..(r + 1) * cols];
        for c in 0..cols {
            sum += f64::from(row[c]) * input[c];
        }
        mid[r] = sum;
    }
    let mut output = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut sum = 0.0f64;
        let left_row = &left[row * rank..(row + 1) * rank];
        for r in 0..rank {
            sum += f64::from(left_row[r]) * mid[r];
        }
        if !sum.is_finite() {
            return Err(Error::Model(format!(
                "HGRAVS01 packed matvec produced a non-finite output at row {row}"
            )));
        }
        output.push(sum);
    }
    Ok((header, output))
}

/// Dense reconstruction for CPU parity only. Never call this on a token path.
pub fn decode_hgravs01_dense_f32_for_parity(payload: &[u8]) -> Result<(Hgravs01Header, Vec<f32>)> {
    let (header, left, right) = decode_hgravs01_factors_f32(payload)?;
    let rows = header.matrix_shape[0];
    let cols = header.matrix_shape[1];
    let rank = header.rank;
    let mut dense = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0f32;
            for r in 0..rank {
                sum += left[row * rank + r] * right[r * cols + col];
            }
            dense[row * cols + col] = sum;
        }
    }
    Ok((header, dense))
}

fn path_basename_eq(path: &Path, expected_name: &str) -> bool {
    path.file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|name| name == expected_name)
}

fn resolve_candidate_tensor_path(
    root: &Path,
    tensor_name: &str,
    declared: &str,
    label: &str,
) -> Result<PathBuf> {
    let expected = canonical_regular_path(&expected_tensor_path(root, tensor_name)?, label)?;
    let declared_path = PathBuf::from(declared);
    if !path_basename_eq(&declared_path, expected.file_name().and_then(|v| v.to_str()).unwrap_or(""))
    {
        return Err(model_error(
            label,
            format!(
                "tensor {tensor_name:?} artifact basename is not the deterministic tensors/<sha>.hq30g name"
            ),
        ));
    }
    // Portable binding: candidates sealed under a cleaned packing worktree keep
    // absolute artifact_path strings, but the physical bytes live under the
    // admit root's tensors/ directory and are re-bound by basename + SHA-256.
    Ok(expected)
}

fn declared_tensor_shape_local(row: &Map<String, Value>, label: &str) -> Result<Vec<usize>> {
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

/// Validates manifest source/representation bindings and returns the
/// authority selected-organ count from `representation.selected_organs`.
fn validate_aw_manifest_source(
    manifest: &Map<String, Value>,
    source: &super::SourceChain,
) -> Result<usize> {
    let label = "activation-weighted SVD manifest";
    let manifest_source = required_object(manifest, "source", label)?;
    require_exact_string(
        manifest_source,
        "repository",
        QwenCompleteBinaryModel::Qwen30Coder.source_repository(),
        label,
    )?;
    if required_u64(manifest_source, "tensor_count", label)?
        != u64::try_from(source.weight_map.len()).unwrap_or(u64::MAX)
        || source.weight_map.len() != QWEN30_ACTIVATION_WEIGHTED_SVD_TENSOR_COUNT
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
        "mixed_direct_binary_sign_scale_plus_selected_activation_weighted_svd_low_rank",
        label,
    )?;
    require_exact_string(
        representation,
        "unchanged_tensor_layout",
        "HQ30G1B1 binary sign plus FP16 group scale (hard-linked from admitted baseline)",
        label,
    )?;
    require_exact_string(
        representation,
        "selected_organ_layout",
        "HGRAVS01 activation_weighted_svd_low_rank factors",
        label,
    )?;
    require_exact_string(
        representation,
        "selected_family",
        HGRAVS01_REPRESENTATION,
        label,
    )?;
    if !required_bool(representation, "physical_direct_layout", label)? {
        return Err(model_error(
            label,
            "activation-weighted representation must retain physical direct-layout accounting",
        ));
    }
    // Authority count is representation.selected_organs.len(). Callers must
    // require selection and branch changed_organs to match this same N.
    let selected = required_array(representation, "selected_organs", label)?;
    let authority_count = selected.len();
    if authority_count == 0 {
        return Err(model_error(
            label,
            "representation selected_organs count is 0; expected > 0",
        ));
    }
    // Coverage floor preserves the anti-partial-coverage property the old
    // fixed organ count stood in for. Fields live on the sealed selection
    // receipt and are mirrored on the manifest; validate the manifest copy
    // here and re-check the sealed selection receipt after it is bound.
    let coverage = required_object(representation, "coverage", label)?;
    validate_aw_coverage_floor(coverage, label)?;
    Ok(authority_count)
}

/// Anti-partial-coverage floor: selected set must be non-empty (checked by
/// caller), must not claim layer0-only coverage, and must cover every model
/// layer.
fn validate_aw_coverage_floor(coverage: &Map<String, Value>, label: &str) -> Result<()> {
    if required_bool(coverage, "layer0_only", label)? {
        return Err(model_error(
            label,
            "coverage.layer0_only is true; full-model activation-weighted coverage is required",
        ));
    }
    let n_layers_covered = required_u64(coverage, "n_layers_covered", label)?;
    let model_layers = required_u64(coverage, "model_layers", label)?;
    if model_layers == 0 {
        return Err(model_error(
            label,
            "coverage.model_layers is 0; expected a positive model layer count",
        ));
    }
    if n_layers_covered != model_layers {
        return Err(model_error(
            label,
            format!(
                "coverage.n_layers_covered is {n_layers_covered}, expected coverage.model_layers={model_layers}"
            ),
        ));
    }
    Ok(())
}

/// Expert gate/up/down organs on any decoder layer (not layer-0 only).
fn is_aw_expert_gate_up_down_organ(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("model.layers.") else {
        return false;
    };
    let Some((layer, rest)) = rest.split_once('.') else {
        return false;
    };
    if layer.is_empty() || !layer.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    let Some(rest) = rest.strip_prefix("mlp.experts.") else {
        return false;
    };
    let Some((expert, proj)) = rest.split_once('.') else {
        return false;
    };
    if expert.is_empty() || !expert.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    matches!(
        proj,
        "gate_proj.weight" | "up_proj.weight" | "down_proj.weight"
    )
}

fn aw_selection_organs(
    selection: &Value,
    authority_selected_count: usize,
) -> Result<BTreeMap<String, Map<String, Value>>> {
    let label = "activation-weighted SVD selection receipt";
    let root = selection
        .as_object()
        .ok_or_else(|| model_error(label, "root must be an object"))?;
    require_exact_string(
        root,
        "schema",
        QWEN30_ACTIVATION_WEIGHTED_SVD_SELECTION_SCHEMA,
        label,
    )?;
    require_exact_string(
        root,
        "status",
        "EARNED_SURPLUS_FIRST_ACTIVATION_WEIGHTED_SVD_SELECTION_UNQUALIFIED",
        label,
    )?;
    let selected = required_object(root, "selected_representation", label)?;
    require_exact_string(selected, "family", HGRAVS01_REPRESENTATION, label)?;
    let organs = required_array(selected, "organs", label)?;
    if organs.len() != authority_selected_count {
        return Err(model_error(
            label,
            format!(
                "selection selected_representation.organs count is {}, expected {} (representation.selected_organs authority)",
                organs.len(),
                authority_selected_count
            ),
        ));
    }
    let mut output = BTreeMap::new();
    for value in organs {
        let organ = value
            .as_object()
            .ok_or_else(|| model_error(label, "selected organ must be an object"))?;
        let name = required_string(organ, "tensor_name", label)?;
        if !is_aw_expert_gate_up_down_organ(name) {
            return Err(model_error(
                label,
                format!(
                    "selected organ {name:?} is outside the expert gate/up/down policy (model.layers.<N>.mlp.experts.<E>.{{gate,up,down}}_proj.weight)"
                ),
            ));
        }
        if required_string(organ, "family", label)? != HGRAVS01_REPRESENTATION {
            return Err(model_error(
                label,
                format!("selected organ {name:?} family is not activation-weighted SVD"),
            ));
        }
        if output.insert(name.to_owned(), organ.clone()).is_some() {
            return Err(model_error(
                label,
                format!("selection repeats organ {name:?}"),
            ));
        }
    }
    Ok(output)
}

fn validate_aw_file_binding(
    binding: &Map<String, Value>,
    expected_path: &Path,
    expected_raw_sha256: &str,
    expected_seal_sha256: &str,
    label: &str,
) -> Result<()> {
    // Bind by absolute path when the sealed packing tree still exists; otherwise
    // accept basename + content seal so a relocated complete candidate remains
    // admitable without resealing.
    let declared = absolute_path(required_string(binding, "path", label)?, label)?;
    if declared.exists() {
        require_exact_regular_path(binding, "path", expected_path, label)?;
    } else if !path_basename_eq(&declared, expected_path.file_name().and_then(|v| v.to_str()).unwrap_or(""))
    {
        return Err(model_error(
            label,
            "file binding basename differs from protected document",
        ));
    }
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

fn validate_aw_selection_and_snapshot(
    manifest: &Map<String, Value>,
    admission: &Qwen30ActivationWeightedSvdAdmission,
    expected_revalidation_path: &Path,
    expected_revalidation_raw_sha256: &str,
    authority_selected_count: usize,
) -> Result<BTreeMap<String, Map<String, Value>>> {
    let label = "activation-weighted SVD manifest";
    let expected_selection_path = canonical_expected_regular_path(
        &admission.expected_selection_path,
        "protected activation-weighted selection receipt",
    )?;
    let expected_snapshot_path = canonical_expected_regular_path(
        &admission.expected_source_snapshot_path,
        "protected activation-weighted source snapshot",
    )?;
    let selection_raw =
        read_regular_file(&expected_selection_path, "activation-weighted selection receipt")?;
    let selection =
        parse_json_no_duplicate_keys(&selection_raw, "activation-weighted selection receipt")?;
    let selection_seal =
        verify_sealed_document(&selection, "activation-weighted selection receipt")?;
    if selection_seal != admission.expected_selection_seal_sha256 {
        return Err(model_error(
            label,
            "selection receipt seal differs from protected handoff binding",
        ));
    }
    let snapshot_raw =
        read_regular_file(&expected_snapshot_path, "activation-weighted source snapshot")?;
    let snapshot =
        parse_json_no_duplicate_keys(&snapshot_raw, "activation-weighted source snapshot")?;
    let snapshot_seal = verify_sealed_document(&snapshot, "activation-weighted source snapshot")?;
    if snapshot_seal != admission.expected_source_snapshot_seal_sha256 {
        return Err(model_error(
            label,
            "source snapshot seal differs from protected handoff binding",
        ));
    }
    let branch = required_object(manifest, "activation_weighted_svd_branch", label)?;
    require_exact_string(
        branch,
        "branch_id",
        QWEN30_ACTIVATION_WEIGHTED_SVD_BRANCH_ID,
        label,
    )?;
    let branch_snapshot = required_object(branch, "source_binding_snapshot", label)?;
    validate_aw_file_binding(
        branch_snapshot,
        &expected_snapshot_path,
        &sha256_hex(&snapshot_raw),
        &snapshot_seal,
        "activation-weighted manifest source snapshot binding",
    )?;
    let branch_selection = required_object(branch, "selection_receipt", label)?;
    validate_aw_file_binding(
        branch_selection,
        &expected_selection_path,
        &sha256_hex(&selection_raw),
        &selection_seal,
        "activation-weighted manifest selection binding",
    )?;
    let changed = required_array(branch, "changed_organs", label)?;
    if changed.len() != authority_selected_count {
        return Err(model_error(
            label,
            format!(
                "branch changed_organs count is {}, expected {} (representation.selected_organs authority)",
                changed.len(),
                authority_selected_count
            ),
        ));
    }
    let capture = required_object(branch, "activation_capture", label)?;
    let observed_capture = required_sha256(capture, "sha256", label)?;
    if observed_capture != admission.expected_activation_capture_sha256 {
        return Err(model_error(
            label,
            format!(
                "branch activation_capture.sha256 is {observed_capture}, expected {} (operator --expected-activation-capture-sha256)",
                admission.expected_activation_capture_sha256
            ),
        ));
    }

    let snapshot_root = snapshot
        .as_object()
        .ok_or_else(|| model_error("activation-weighted source snapshot", "root must be an object"))?;
    require_exact_string(
        snapshot_root,
        "schema",
        QWEN30_ACTIVATION_WEIGHTED_SVD_SNAPSHOT_SCHEMA,
        "activation-weighted source snapshot",
    )?;
    require_exact_string(
        snapshot_root,
        "status",
        "EARNED_IMMUTABLE_SOURCE_AND_CAPTURE_BINDING",
        "activation-weighted source snapshot",
    )?;
    let snapshot_binding =
        required_object(snapshot_root, "binding", "activation-weighted source snapshot")?;
    require_exact_string(
        snapshot_binding,
        "branch_id",
        QWEN30_ACTIVATION_WEIGHTED_SVD_BRANCH_ID,
        "activation-weighted source snapshot",
    )?;
    let snapshot_revalidation = required_object(
        snapshot_binding,
        "immutable_source_revalidation",
        "activation-weighted source snapshot",
    )?;
    validate_aw_file_binding(
        snapshot_revalidation,
        expected_revalidation_path,
        expected_revalidation_raw_sha256,
        &admission.expected_revalidation_seal_sha256,
        "activation-weighted source snapshot immutable revalidation",
    )?;

    let selection_root = selection
        .as_object()
        .ok_or_else(|| model_error("activation-weighted selection receipt", "root must be an object"))?;
    // Sealed selection coverage is the binding authority for the anti-partial
    // floor (mirrors representation.coverage already checked on the manifest).
    let selection_coverage = required_object(
        selection_root,
        "coverage",
        "activation-weighted selection receipt",
    )?;
    validate_aw_coverage_floor(
        selection_coverage,
        "activation-weighted selection receipt",
    )?;
    let selection_snapshot = required_object(
        selection_root,
        "source_binding_snapshot",
        "activation-weighted selection receipt",
    )?;
    validate_aw_file_binding(
        selection_snapshot,
        &expected_snapshot_path,
        &sha256_hex(&snapshot_raw),
        &snapshot_seal,
        "activation-weighted selection source snapshot binding",
    )?;
    aw_selection_organs(&selection, authority_selected_count)
}

fn validate_aw_terminal(
    terminal: &Value,
    terminal_path: &Path,
    manifest_path: &Path,
    manifest_raw_sha256: &str,
    manifest_seal: &str,
    manifest_bytes: usize,
    admission: &Qwen30ActivationWeightedSvdAdmission,
) -> Result<()> {
    let label = "activation-weighted SVD terminal receipt";
    let object = terminal
        .as_object()
        .ok_or_else(|| model_error(label, "root must be an object"))?;
    verify_sealed_document(terminal, label)?;
    require_exact_string(
        object,
        "schema",
        QWEN30_ACTIVATION_WEIGHTED_SVD_TERMINAL_SCHEMA,
        label,
    )?;
    require_exact_string(
        object,
        "status",
        "EARNED_COMPLETE_PHYSICAL_BINARY_CANDIDATE_UNQUALIFIED",
        label,
    )?;
    let binding = required_object(object, "binding", label)?;
    require_exact_string(binding, "model_id", QWEN30_ACTIVATION_WEIGHTED_SVD_MODEL_ID, label)?;
    require_exact_string(
        binding,
        "artifact_prefix",
        QWEN30_ACTIVATION_WEIGHTED_SVD_ARTIFACT_PREFIX,
        label,
    )?;
    require_exact_string(
        binding,
        "manifest_schema",
        QWEN30_ACTIVATION_WEIGHTED_SVD_SCHEMA,
        label,
    )?;
    let progress = required_object(binding, "progress", label)?;
    for key in ["planned_tensors", "completed_tensors", "next_cursor"] {
        if required_u64(progress, key, label)?
            != u64::try_from(QWEN30_ACTIVATION_WEIGHTED_SVD_TENSOR_COUNT).unwrap()
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
    // Portable path binding: require basename + seal/document identity rather
    // than the cleaned packing worktree absolute path.
    let declared_manifest = absolute_path(required_string(candidate, "manifest_path", label)?, label)?;
    if declared_manifest.exists() {
        require_exact_regular_path(candidate, "manifest_path", manifest_path, label)?;
    } else if !path_basename_eq(
        &declared_manifest,
        manifest_path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or(""),
    ) {
        return Err(model_error(
            label,
            "terminal candidate manifest basename differs from admitted manifest",
        ));
    }
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
        "protected activation-weighted terminal receipt",
    )?;
    if terminal_path != expected_terminal_path {
        return Err(model_error(
            label,
            "terminal path differs from protected handoff path",
        ));
    }
    Ok(())
}

/// Validate one activation-weighted tensor row.
///
/// When `warm_payload` is `Some`, identity has already been proven via the
/// shared warm receipt and content SHA-256 is not recomputed. Size, layout,
/// selection binding, and geometry checks still run on every start.
fn validate_aw_tensor_row_with_payload(
    row: &Map<String, Value>,
    root: &Path,
    source: &super::SourceChain,
    selected_organs: &BTreeMap<String, Map<String, Value>>,
    selection_path: &Path,
    expected_activation_capture_sha256: &str,
    warm_payload: Option<Arc<[u8]>>,
) -> Result<(Qwen30ActivationWeightedTensor, Arc<[u8]>)> {
    let label = "activation-weighted SVD manifest tensor";
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
    let shape = declared_tensor_shape_local(row, label)?;
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
    let expected_path = resolve_candidate_tensor_path(
        root,
        tensor_name,
        required_string(row, "artifact_path", label)?,
        label,
    )?;
    let artifact_bytes = required_u64(row, "artifact_bytes", label)?;
    let artifact_sha256 = required_sha256(row, "artifact_sha256", label)?;
    let payload: Arc<[u8]> = if let Some(warm) = warm_payload {
        // Warm path: content hash already proven under matching identity.
        // Still prove size equals the sealed manifest row.
        if artifact_bytes != u64::try_from(warm.len()).unwrap_or(u64::MAX) {
            return Err(model_error(
                label,
                "warm tensor artifact_bytes does not equal physical payload bytes",
            ));
        }
        warm
    } else {
        let read_start = Instant::now();
        let bytes = read_regular_file(&expected_path, label)?;
        aw_payload_timing_add(&AW_PAYLOAD_READ_MS, read_start);
        if artifact_bytes != u64::try_from(bytes.len()).unwrap_or(u64::MAX) {
            return Err(model_error(
                label,
                "tensor artifact_bytes does not equal physical payload bytes",
            ));
        }
        let hash_start = Instant::now();
        let observed = sha256_hex(&bytes);
        aw_payload_timing_add(&AW_PAYLOAD_SHA256_MS, hash_start);
        if observed != artifact_sha256 {
            return Err(model_error(
                label,
                "tensor payload SHA-256 does not match the manifest",
            ));
        }
        Arc::from(bytes)
    };
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
    let layout_start = Instant::now();
    let parsed_layout = if is_selected {
        if !required_bool(mutation, "changed_from_admitted_control", label)? {
            return Err(model_error(
                label,
                "selected organ was not marked as the explicit activation-weighted mutation",
            ));
        }
        let selection_declared =
            absolute_path(required_string(mutation, "selection_receipt_path", label)?, label)?;
        if selection_declared.exists() {
            require_exact_regular_path(mutation, "selection_receipt_path", selection_path, label)?;
        } else if !path_basename_eq(
            &selection_declared,
            selection_path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or(""),
        ) {
            return Err(model_error(
                label,
                "selected organ selection receipt basename differs from protected receipt",
            ));
        }
        let selected = selected_organs
            .get(tensor_name)
            .expect("checked contains_key");
        if required_sha256(selected, "physical_payload_sha256", label)? != artifact_sha256
            || required_u64(selected, "physical_payload_bytes", label)? != artifact_bytes
        {
            return Err(model_error(
                label,
                "selected organ physical payload differs from sealed selection",
            ));
        }
        require_exact_string(layout, "magic", HGRAVS01_MAGIC_TEXT, label)?;
        require_exact_string(layout, "family", HGRAVS01_REPRESENTATION, label)?;
        require_exact_string(layout, "schema", HGRAVS01_SCHEMA, label)?;
        let parsed = parse_hgravs01_header(payload.as_ref())?;
        if parsed.shape != shape || u64::try_from(parsed.elements).ok() != Some(elements) {
            return Err(model_error(
                label,
                "selected organ HGRAVS01 payload geometry differs from manifest",
            ));
        }
        if required_u64(layout, "rank", label)? != u64::try_from(parsed.rank).unwrap_or(u64::MAX)
            || required_u64(layout, "factor_bits", label)? != u64::from(parsed.factor_bits)
        {
            return Err(model_error(
                label,
                "selected organ layout rank/bits differ from physical HGRAVS01 header",
            ));
        }
        let layout_capture = required_sha256(layout, "activation_capture_sha256", label)?;
        if layout_capture != parsed.activation_capture_sha256
            || parsed.activation_capture_sha256 != expected_activation_capture_sha256
        {
            return Err(model_error(
                label,
                format!(
                    "selected organ activation capture binding differs from operator expectation: layout={layout_capture} payload={} expected={expected_activation_capture_sha256}",
                    parsed.activation_capture_sha256
                ),
            ));
        }
        if required_u64(selected, "rank", label)? != u64::try_from(parsed.rank).unwrap_or(u64::MAX)
            || required_u64(selected, "bits", label)? != u64::from(parsed.factor_bits)
        {
            return Err(model_error(
                label,
                "selected organ selection rank/bits differ from physical payload",
            ));
        }
        Qwen30ActivationWeightedTensorLayout::ActivationWeightedSvd(parsed)
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
        let parsed = parse_complete_binary_header(payload.as_ref())?;
        if parsed.shape != shape || u64::try_from(parsed.elements).ok() != Some(elements) {
            return Err(model_error(
                label,
                "control tensor direct payload geometry differs from manifest",
            ));
        }
        Qwen30ActivationWeightedTensorLayout::Direct(parsed)
    };
    aw_payload_timing_add(&AW_PAYLOAD_LAYOUT_MS, layout_start);
    Ok((
        Qwen30ActivationWeightedTensor {
            tensor_name: tensor_name.to_owned(),
            source_shard: source_shard.to_owned(),
            source_shard_sha256,
            source_dtype: source_dtype.to_owned(),
            artifact_path: expected_path,
            artifact_sha256: artifact_sha256.to_owned(),
            artifact_bytes: payload.len(),
            elements: usize::try_from(elements)
                .map_err(|_| model_error(label, "tensor elements do not fit platform usize"))?,
            layout: parsed_layout,
        },
        payload,
    ))
}

fn validate_aw_ledger(
    manifest: &Map<String, Value>,
    tensors: &BTreeMap<String, Qwen30ActivationWeightedTensor>,
    manifest_bytes: usize,
) -> Result<(u64, u64)> {
    let label = "activation-weighted SVD manifest ledger";
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

fn validate_aw_tensor_rows_bounded_parallel(
    rows: &[Value],
    root: &Path,
    source: &super::SourceChain,
    selected_organs: &BTreeMap<String, Map<String, Value>>,
    selection_path: &Path,
    expected_activation_capture_sha256: &str,
) -> Result<(Vec<(Qwen30ActivationWeightedTensor, Arc<[u8]>)>, usize)> {
    validate_aw_tensor_rows_bounded_parallel_with_warm(
        rows,
        root,
        source,
        selected_organs,
        selection_path,
        expected_activation_capture_sha256,
        None,
    )
}

fn validate_aw_tensor_rows_bounded_parallel_with_warm(
    rows: &[Value],
    root: &Path,
    source: &super::SourceChain,
    selected_organs: &BTreeMap<String, Map<String, Value>>,
    selection_path: &Path,
    expected_activation_capture_sha256: &str,
    warm_payloads: Option<&BTreeMap<String, Arc<[u8]>>>,
) -> Result<(Vec<(Qwen30ActivationWeightedTensor, Arc<[u8]>)>, usize)> {
    let (lanes, workers) = quality_payload_verification_lanes(rows);
    let workers = workers.min(QWEN30_ACTIVATION_WEIGHTED_SVD_MAX_PAYLOAD_VERIFY_WORKERS);
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
                                "activation-weighted SVD manifest tensor entry must be an object"
                                    .into(),
                            )
                        })
                        .and_then(|row| {
                            let warm = if let Some(map) = warm_payloads {
                                let name = required_string(
                                    row,
                                    "tensor_name",
                                    "activation-weighted SVD manifest tensor",
                                )?;
                                Some(
                                    map.get(name)
                                        .cloned()
                                        .ok_or_else(|| {
                                            Error::Model(format!(
                                                "warm admission missing preloaded payload for {name:?}"
                                            ))
                                        })?,
                                )
                            } else {
                                None
                            };
                            validate_aw_tensor_row_with_payload(
                                row,
                                root,
                                source,
                                selected_organs,
                                selection_path,
                                expected_activation_capture_sha256,
                                warm,
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
                    "activation-weighted payload verification worker panicked; refusing candidate admission"
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
                "activation-weighted payload verification failed at manifest ordinal {ordinal}: {error}"
            ))
        })?;
        tensors.push(tensor);
    }
    Ok((tensors, workers))
}

fn load_aw_warm_payloads_bounded_parallel(
    receipt: &admission_warm_receipt::WarmAdmissionReceipt,
) -> Result<BTreeMap<String, Arc<[u8]>>> {
    let entries: Vec<&admission_warm_receipt::ReceiptEntry> = receipt.entries.values().collect();
    if entries.is_empty() {
        return Ok(BTreeMap::new());
    }
    let workers = entries
        .len()
        .min(QWEN30_ACTIVATION_WEIGHTED_SVD_MAX_PAYLOAD_VERIFY_WORKERS)
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
                        admission_warm_receipt::load_payload_bytes_warm_skip_hash(entries[idx]);
                    lane_outcomes.push((idx, outcome));
                }
                lane_outcomes
            }));
        }
        let mut joined = Vec::with_capacity(entries.len());
        for handle in handles {
            let lane = handle.join().map_err(|_| {
                Error::Model(
                    "activation-weighted warm payload load worker panicked; refusing admission"
                        .into(),
                )
            })?;
            joined.extend(lane);
        }
        Ok::<_, Error>(joined)
    })?;
    let mut outcomes = outcomes;
    outcomes.sort_by_key(|(idx, _)| *idx);
    let mut payloads = BTreeMap::new();
    for (idx, outcome) in outcomes {
        let payload = outcome.map_err(|error| {
            Error::Model(format!(
                "activation-weighted warm payload load failed at entry {idx}: {error}"
            ))
        })?;
        let name = entries[idx].tensor_name.clone();
        if payloads.insert(name, payload).is_some() {
            return Err(Error::Model(
                "activation-weighted warm load produced a duplicate tensor_name".into(),
            ));
        }
    }
    Ok(payloads)
}

/// Strict native admission for the mixed HQ30G1B1 + HGRAVS01 Qwen30 candidate.
///
/// Warm path (`HAWKING_ADMISSION_WARM_RECEIPT`, default on): after a prior cold
/// full rehash wrote a sealed receipt, a later process may skip *recomputing*
/// content SHA-256 when every payload file's (size, mtime_ns, inode) still
/// matches. device (st_dev) is deliberately excluded — it is a mount-time
/// artifact that a remount reassigns without changing the file, and including it
/// forced a full cold rehash on every post-reboot start. It never skips the
/// unchanged-file check. Selection, snapshot,
/// terminal, activation-capture, source-chain, and geometry seals still run
/// every start. Disable with `HAWKING_ADMISSION_WARM_RECEIPT=0` to force a full
/// cold rehash every start.
pub fn admit_qwen30_activation_weighted_svd_artifact(
    manifest_path: impl AsRef<Path>,
    admission: &Qwen30ActivationWeightedSvdAdmission,
) -> Result<Qwen30ActivationWeightedSvdArtifact> {
    crate::startup_timing::time_ms_result("admit_hgravs_total", || {
        admit_qwen30_activation_weighted_svd_artifact_inner(manifest_path, admission)
    })
}

fn admit_qwen30_activation_weighted_svd_artifact_inner(
    manifest_path: impl AsRef<Path>,
    admission: &Qwen30ActivationWeightedSvdAdmission,
) -> Result<Qwen30ActivationWeightedSvdArtifact> {
    aw_payload_timing_reset();
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
        (
            "expected activation capture",
            admission.expected_activation_capture_sha256.as_str(),
        ),
    ] {
        if !is_sha256(value) {
            return Err(Error::Model(format!(
                "activation-weighted SVD admission {label} must be lowercase SHA-256"
            )));
        }
    }
    if admission.expected_source_revision.is_empty() {
        return Err(Error::Model(
            "activation-weighted SVD admission requires a protected source revision".into(),
        ));
    }

    // ---- Manifest seal (always) ----
    let phase = Instant::now();
    let manifest_path =
        canonical_regular_path(manifest_path.as_ref(), "activation-weighted SVD manifest")?;
    let root = manifest_path.parent().ok_or_else(|| {
        Error::Model("activation-weighted SVD manifest has no candidate root parent".into())
    })?;
    let expected_revalidation_path = canonical_expected_regular_path(
        &admission.expected_revalidation_path,
        "protected activation-weighted source revalidation receipt",
    )?;
    let expected_terminal_path = canonical_expected_regular_path(
        &admission.expected_terminal_path,
        "protected activation-weighted terminal receipt",
    )?;
    let manifest_raw = read_regular_file(&manifest_path, "activation-weighted SVD manifest")?;
    let manifest_raw_sha256 = sha256_hex(&manifest_raw);
    let manifest = parse_json_no_duplicate_keys(&manifest_raw, "activation-weighted SVD manifest")?;
    let manifest_object = manifest.as_object().ok_or_else(|| {
        Error::Model("activation-weighted SVD manifest root must be an object".into())
    })?;
    let manifest_seal = verify_sealed_document(&manifest, "activation-weighted SVD manifest")?;
    if manifest_seal != admission.expected_manifest_seal_sha256 {
        return Err(Error::Model(
            "activation-weighted SVD manifest seal differs from protected handoff binding".into(),
        ));
    }
    require_exact_string(
        manifest_object,
        "schema",
        QWEN30_ACTIVATION_WEIGHTED_SVD_SCHEMA,
        "activation-weighted SVD manifest",
    )?;
    require_exact_string(
        manifest_object,
        "status",
        COMPLETE_BINARY_CANDIDATE_STATUS,
        "activation-weighted SVD manifest",
    )?;
    let manifest_audit_seal = required_sha256(
        manifest_object,
        "source_body_audit_seal_sha256",
        "activation-weighted SVD manifest",
    )?;
    if manifest_audit_seal != admission.expected_source_audit_seal_sha256 {
        return Err(Error::Model(
            "activation-weighted SVD manifest source audit seal differs from protected handoff binding"
                .into(),
        ));
    }
    crate::startup_timing::record_ms(
        "admit_manifest_seal",
        crate::startup_timing::duration_ms(phase.elapsed()),
    );

    // ---- Source chain (always) ----
    let phase = Instant::now();
    // Revalidation lives under the baseline complete-gravity authority, not
    // necessarily under the candidate root.
    if required_sha256(
        manifest_object,
        "source_revalidation_receipt_seal_sha256",
        "activation-weighted SVD manifest",
    )? != admission.expected_revalidation_seal_sha256
    {
        return Err(Error::Model(
            "activation-weighted SVD manifest revalidation seal differs from protected handoff binding"
                .into(),
        ));
    }
    let declared_revalidation = absolute_path(
        required_string(
            manifest_object,
            "source_revalidation_receipt_path",
            "activation-weighted SVD manifest",
        )?,
        "activation-weighted SVD manifest",
    )?;
    if declared_revalidation.exists() {
        require_exact_regular_path(
            manifest_object,
            "source_revalidation_receipt_path",
            &expected_revalidation_path,
            "activation-weighted SVD manifest",
        )?;
    } else if !path_basename_eq(
        &declared_revalidation,
        expected_revalidation_path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or(""),
    ) {
        return Err(Error::Model(
            "activation-weighted SVD revalidation basename differs from protected handoff".into(),
        ));
    }
    let receipt_raw = read_regular_file(
        &expected_revalidation_path,
        "activation-weighted source revalidation receipt",
    )?;
    let receipt = parse_json_no_duplicate_keys(
        &receipt_raw,
        "activation-weighted source revalidation receipt",
    )?;
    let receipt_seal =
        verify_sealed_document(&receipt, "activation-weighted source revalidation receipt")?;
    if receipt_seal != admission.expected_revalidation_seal_sha256 {
        return Err(Error::Model(
            "activation-weighted source revalidation seal differs from protected handoff binding"
                .into(),
        ));
    }
    let standard_admission = CompleteBinaryAdmission {
        model: QwenCompleteBinaryModel::Qwen30Coder,
        expected_manifest_seal_sha256: manifest_seal.clone(),
        expected_source_audit_seal_sha256: admission.expected_source_audit_seal_sha256.clone(),
        expected_source_revision: admission.expected_source_revision.clone(),
    };
    let revalidation_parent = expected_revalidation_path.parent().ok_or_else(|| {
        Error::Model(
            "activation-weighted source revalidation receipt has no authority parent".into(),
        )
    })?;
    let source = validate_source_chain_at(
        &receipt,
        &expected_revalidation_path,
        revalidation_parent,
        &standard_admission,
        &manifest_audit_seal,
    )?;
    let authority_selected_count = validate_aw_manifest_source(manifest_object, &source)?;
    crate::startup_timing::record_ms(
        "admit_source_chain",
        crate::startup_timing::duration_ms(phase.elapsed()),
    );

    // ---- Selection + source-binding snapshot seals (always) ----
    let phase = Instant::now();
    let selected_organs = validate_aw_selection_and_snapshot(
        manifest_object,
        admission,
        &expected_revalidation_path,
        &sha256_hex(&receipt_raw),
        authority_selected_count,
    )?;
    crate::startup_timing::record_ms(
        "admit_selection_snapshot_seals",
        crate::startup_timing::duration_ms(phase.elapsed()),
    );

    // ---- Terminal seal (always) ----
    let phase = Instant::now();
    let terminal_raw =
        read_regular_file(&expected_terminal_path, "activation-weighted terminal receipt")?;
    let terminal =
        parse_json_no_duplicate_keys(&terminal_raw, "activation-weighted terminal receipt")?;
    let terminal_seal =
        verify_sealed_document(&terminal, "activation-weighted terminal receipt")?;
    if terminal_seal != admission.expected_terminal_seal_sha256 {
        return Err(Error::Model(
            "activation-weighted terminal receipt seal differs from protected handoff binding"
                .into(),
        ));
    }
    validate_aw_terminal(
        &terminal,
        &expected_terminal_path,
        &manifest_path,
        &manifest_raw_sha256,
        &manifest_seal,
        manifest_raw.len(),
        admission,
    )?;
    let terminal_object = terminal.as_object().ok_or_else(|| {
        Error::Model("activation-weighted terminal receipt root must be an object".into())
    })?;
    let terminal_binding =
        required_object(terminal_object, "binding", "activation-weighted terminal receipt")?;
    if required_sha256(
        terminal_binding,
        "source_body_audit_seal_sha256",
        "activation-weighted terminal receipt",
    )? != admission.expected_source_audit_seal_sha256
    {
        return Err(Error::Model(
            "activation-weighted terminal source audit seal differs from protected handoff".into(),
        ));
    }
    crate::startup_timing::record_ms(
        "admit_terminal_seal",
        crate::startup_timing::duration_ms(phase.elapsed()),
    );

    let rows = required_array(
        manifest_object,
        "tensors",
        "activation-weighted SVD manifest",
    )?;
    if rows.len() != source.weight_map.len()
        || rows.len() != QWEN30_ACTIVATION_WEIGHTED_SVD_TENSOR_COUNT
    {
        return Err(Error::Model(
            "activation-weighted SVD manifest does not contain every Qwen30 source tensor".into(),
        ));
    }
    let expected_selection_path = canonical_expected_regular_path(
        &admission.expected_selection_path,
        "protected activation-weighted selection receipt",
    )?;

    // ---- Warm path: skip content rehash when identity metadata still matches ----
    if admission_warm_receipt::warm_receipt_enabled() {
        match try_warm_aw_payload_admission(
            &manifest_path,
            &manifest_seal,
            root,
            &source,
            rows,
            manifest_object,
            manifest_raw.len(),
            &selected_organs,
            &expected_selection_path,
            &admission.expected_activation_capture_sha256,
            authority_selected_count,
            terminal_object,
        ) {
            Ok(Some(artifact)) => {
                crate::startup_timing::record_ms("admit_payload_mode_warm_skip_rehash", 1);
                aw_payload_timing_flush();
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
    let (validated_rows, payload_verification_workers) =
        crate::startup_timing::time_ms_result("admit_payload_cold_rehash", || {
            validate_aw_tensor_rows_bounded_parallel(
                rows,
                root,
                &source,
                &selected_organs,
                &expected_selection_path,
                &admission.expected_activation_capture_sha256,
            )
        })?;
    crate::startup_timing::record_ms("admit_payload_mode_cold_full_rehash", 1);
    aw_payload_timing_flush();

    let artifact = assemble_aw_artifact(
        validated_rows,
        payload_verification_workers,
        &source,
        authority_selected_count,
        &selected_organs,
        manifest_object,
        terminal_object,
        manifest_raw.len(),
        manifest_path,
        manifest_seal,
    )?;

    if admission_warm_receipt::warm_receipt_enabled() {
        let _ = crate::startup_timing::time_ms_result("admit_warm_receipt_write", || {
            let specs: Vec<admission_warm_receipt::ReceiptEntrySpec> = artifact
                .tensors
                .iter()
                .map(|(name, tensor)| {
                    let layout_kind = match &tensor.layout {
                        Qwen30ActivationWeightedTensorLayout::Direct(_) => {
                            admission_warm_receipt::LAYOUT_KIND_HQ30G1B1
                        }
                        Qwen30ActivationWeightedTensorLayout::ActivationWeightedSvd(_) => {
                            admission_warm_receipt::LAYOUT_KIND_HGRAVS01
                        }
                    };
                    // Mixed catalog: layout is re-proven from bytes+manifest on
                    // every warm load. Identity receipt does not store headers.
                    admission_warm_receipt::ReceiptEntrySpec {
                        tensor_name: name.clone(),
                        artifact_path: tensor.artifact_path.clone(),
                        artifact_sha256: tensor.artifact_sha256.clone(),
                        source_shard: tensor.source_shard.clone(),
                        source_shard_sha256: tensor.source_shard_sha256.clone(),
                        source_dtype: tensor.source_dtype.clone(),
                        header: None,
                        layout_kind: layout_kind.to_owned(),
                    }
                })
                .collect();
            let receipt = admission_warm_receipt::build_receipt_from_specs(
                &artifact.manifest_path,
                &artifact.manifest_seal_sha256,
                &specs,
            )?;
            admission_warm_receipt::write_receipt(&receipt)
        });
    }

    Ok(artifact)
}

fn try_warm_aw_payload_admission(
    manifest_path: &Path,
    manifest_seal: &str,
    root: &Path,
    source: &super::SourceChain,
    rows: &[Value],
    manifest_object: &Map<String, Value>,
    manifest_raw_len: usize,
    selected_organs: &BTreeMap<String, Map<String, Value>>,
    selection_path: &Path,
    expected_activation_capture_sha256: &str,
    authority_selected_count: usize,
    terminal_object: &Map<String, Value>,
) -> Result<Option<Qwen30ActivationWeightedSvdArtifact>> {
    let Some(receipt) = admission_warm_receipt::load_receipt(manifest_seal)? else {
        return Ok(None);
    };
    if receipt.manifest_path != manifest_path {
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
        // Any single identity mismatch forces FULL cold rehash of the catalog.
        return Ok(None);
    }

    let warm_payloads =
        crate::startup_timing::time_ms_result("admit_payload_warm_load_no_rehash", || {
            load_aw_warm_payloads_bounded_parallel(&receipt)
        })?;

    // Layout / selection / geometry still re-proven on every warm start.
    let (validated_rows, payload_verification_workers) =
        crate::startup_timing::time_ms_result("admit_payload_warm_layout_revalidate", || {
            validate_aw_tensor_rows_bounded_parallel_with_warm(
                rows,
                root,
                source,
                selected_organs,
                selection_path,
                expected_activation_capture_sha256,
                Some(&warm_payloads),
            )
        })?;

    let artifact = assemble_aw_artifact(
        validated_rows,
        payload_verification_workers,
        source,
        authority_selected_count,
        selected_organs,
        manifest_object,
        terminal_object,
        manifest_raw_len,
        manifest_path.to_path_buf(),
        manifest_seal.to_owned(),
    )?;
    Ok(Some(artifact))
}

fn assemble_aw_artifact(
    validated_rows: Vec<(Qwen30ActivationWeightedTensor, Arc<[u8]>)>,
    payload_verification_workers: usize,
    source: &super::SourceChain,
    authority_selected_count: usize,
    selected_organs: &BTreeMap<String, Map<String, Value>>,
    manifest_object: &Map<String, Value>,
    terminal_object: &Map<String, Value>,
    manifest_raw_len: usize,
    manifest_path: PathBuf,
    manifest_seal: String,
) -> Result<Qwen30ActivationWeightedSvdArtifact> {
    let mut tensors = BTreeMap::new();
    let mut verified_payloads = BTreeMap::new();
    let mut selected_hgravs_organs = BTreeSet::new();
    for (tensor, verified_payload) in validated_rows {
        if matches!(
            tensor.layout,
            Qwen30ActivationWeightedTensorLayout::ActivationWeightedSvd(_)
        ) {
            selected_hgravs_organs.insert(tensor.tensor_name.clone());
        }
        let tensor_name = tensor.tensor_name.clone();
        if tensors.insert(tensor_name.clone(), tensor).is_some() {
            return Err(Error::Model(
                "activation-weighted SVD manifest has a duplicate tensor name".into(),
            ));
        }
        if verified_payloads
            .insert(tensor_name, verified_payload)
            .is_some()
        {
            return Err(Error::Model(
                "activation-weighted SVD manifest duplicated an immutable payload entry".into(),
            ));
        }
    }
    if selected_hgravs_organs.len() != authority_selected_count
        || selected_hgravs_organs.len() != selected_organs.len()
        || selected_hgravs_organs != selected_organs.keys().cloned().collect::<BTreeSet<_>>()
    {
        return Err(Error::Model(format!(
            "activation-weighted SVD selected HGRAVS01 set differs from sealed selection: observed_hgravs={} selection={} authority={}",
            selected_hgravs_organs.len(),
            selected_organs.len(),
            authority_selected_count
        )));
    }
    if tensors.keys().ne(source.weight_map.keys()) {
        return Err(Error::Model(
            "activation-weighted SVD manifest tensor set differs from current source index".into(),
        ));
    }
    if verified_payloads.len() != tensors.len() {
        return Err(Error::Model(
            "activation-weighted SVD admission did not retain one verified immutable payload per tensor"
                .into(),
        ));
    }
    let (source_weight_elements, tensor_payload_bytes) =
        validate_aw_ledger(manifest_object, &tensors, manifest_raw_len)?;
    let terminal_candidate =
        required_object(terminal_object, "candidate", "activation-weighted terminal receipt")?;
    if required_u64(
        terminal_candidate,
        "all_required_weight_artifact_bytes",
        "activation-weighted terminal receipt",
    )? != tensor_payload_bytes + u64::try_from(manifest_raw_len).unwrap_or(u64::MAX)
    {
        return Err(Error::Model(
            "activation-weighted terminal byte ledger differs from scanned manifest".into(),
        ));
    }
    let exact_bpw = (tensor_payload_bytes + u64::try_from(manifest_raw_len).unwrap_or(u64::MAX))
        as f64
        * 8.0
        / source_weight_elements as f64;
    if (required_f64(
        terminal_candidate,
        "complete_physical_bpw",
        "activation-weighted terminal receipt",
    )? - exact_bpw)
        .abs()
        > exact_bpw.abs().max(1.0) * 1e-12
        || !required_bool(
            terminal_candidate,
            "passes_storage_threshold",
            "activation-weighted terminal receipt",
        )?
    {
        return Err(Error::Model(
            "activation-weighted terminal BPW gate differs from scanned artifact".into(),
        ));
    }
    Ok(Qwen30ActivationWeightedSvdArtifact {
        manifest_path,
        manifest_seal_sha256: manifest_seal,
        source_audit_path: source.source_audit_path.clone(),
        source_audit_seal_sha256: source.source_audit_seal_sha256.clone(),
        source_revision: source.source_revision.clone(),
        source_index_path: source.source_index_path.clone(),
        source_weight_elements,
        tensor_payload_bytes,
        selected_hgravs_organs: selected_hgravs_organs.into_iter().collect(),
        payload_verification_workers,
        tensors,
        verified_payloads,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_unsigned(codes: &[u8], bits: u8) -> Vec<u8> {
        let mut bits_out = Vec::new();
        for &code in codes {
            for bit in 0..bits {
                bits_out.push((code >> bit) & 1);
            }
        }
        while bits_out.len() % 8 != 0 {
            bits_out.push(0);
        }
        bits_out
            .chunks(8)
            .map(|chunk| {
                let mut byte = 0u8;
                for (index, bit) in chunk.iter().enumerate() {
                    byte |= bit << index;
                }
                byte
            })
            .collect()
    }

    fn uniform_body(values: &[f32], bits: u8, group_size: usize) -> (Vec<u8>, Map<String, Value>) {
        let elements = values.len();
        let groups = elements.div_ceil(group_size);
        let bound = (1u16 << (bits - 1)) - 1;
        let mut scales = Vec::with_capacity(groups);
        let mut codes = Vec::with_capacity(groups * group_size);
        for group in 0..groups {
            let start = group * group_size;
            let end = (start + group_size).min(elements);
            let max_abs = values[start..end]
                .iter()
                .map(|value| value.abs())
                .fold(0.0f32, f32::max);
            let scale = max_abs / f32::from(bound.max(1));
            let scale_bits = f16::from_f32(scale).to_bits().to_le_bytes();
            scales.extend_from_slice(&scale_bits);
            let denom = if scale > 0.0 { scale } else { 1.0 };
            for index in 0..group_size {
                let value = values.get(start + index).copied().unwrap_or(0.0);
                let signed = (value / denom).round().clamp(-(bound as f32), bound as f32) as i16;
                codes.push((signed + bound as i16) as u8);
            }
        }
        let code_bytes = pack_unsigned(&codes, bits);
        let mut meta = Map::new();
        meta.insert("schema".into(), Value::String(HGRAVS01_UNIFORM_SCHEMA.into()));
        meta.insert(
            "representation".into(),
            Value::String(format!("uniform_q{bits}_group_scale")),
        );
        meta.insert(
            "shape".into(),
            Value::Array(vec![Value::from(elements as u64)]),
        );
        meta.insert("elements".into(), Value::from(elements as u64));
        meta.insert("bits".into(), Value::from(u64::from(bits)));
        meta.insert("group_size".into(), Value::from(group_size as u64));
        meta.insert("groups".into(), Value::from(groups as u64));
        meta.insert("scale_dtype".into(), Value::String("float16".into()));
        meta.insert("scale_bytes".into(), Value::from((groups * 2) as u64));
        meta.insert("code_bytes".into(), Value::from(code_bytes.len() as u64));
        meta.insert(
            "retained_padding_elements".into(),
            Value::from((groups * group_size - elements) as u64),
        );
        let mut body = scales;
        body.extend_from_slice(&code_bytes);
        (body, meta)
    }

    fn hgravs_fixture(rows: usize, cols: usize, rank: usize, bits: u8) -> Vec<u8> {
        let left_values: Vec<f32> = (0..rows * rank)
            .map(|index| ((index % 7) as f32 - 3.0) * 0.1)
            .collect();
        let right_values: Vec<f32> = (0..rank * cols)
            .map(|index| ((index % 5) as f32 - 2.0) * 0.05)
            .collect();
        let (left_body, mut left_meta) = uniform_body(&left_values, bits, 64);
        left_meta.insert(
            "shape".into(),
            Value::Array(vec![Value::from(rows as u64), Value::from(rank as u64)]),
        );
        let (right_body, mut right_meta) = uniform_body(&right_values, bits, 64);
        right_meta.insert(
            "shape".into(),
            Value::Array(vec![Value::from(rank as u64), Value::from(cols as u64)]),
        );
        let mut header = Map::new();
        header.insert("schema".into(), Value::String(HGRAVS01_SCHEMA.into()));
        header.insert(
            "representation".into(),
            Value::String(HGRAVS01_REPRESENTATION.into()),
        );
        header.insert(
            "shape".into(),
            Value::Array(vec![Value::from(rows as u64), Value::from(cols as u64)]),
        );
        header.insert(
            "matrix_shape".into(),
            Value::Array(vec![Value::from(rows as u64), Value::from(cols as u64)]),
        );
        header.insert("elements".into(), Value::from((rows * cols) as u64));
        header.insert("rank".into(), Value::from(rank as u64));
        header.insert("factor_bits".into(), Value::from(u64::from(bits)));
        header.insert("factor_group_size".into(), Value::from(64u64));
        header.insert("left".into(), Value::Object(left_meta));
        header.insert("right".into(), Value::Object(right_meta));
        header.insert("left_body_bytes".into(), Value::from(left_body.len() as u64));
        header.insert(
            "right_body_bytes".into(),
            Value::from(right_body.len() as u64),
        );
        // Synthetic fixture capture pin (unit tests only; production uses the
        // operator-supplied admission expectation).
        const FIXTURE_ACTIVATION_CAPTURE_SHA256: &str =
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let mut capture = Map::new();
        capture.insert(
            "sha256".into(),
            Value::String(FIXTURE_ACTIVATION_CAPTURE_SHA256.into()),
        );
        capture.insert(
            "fit_kind".into(),
            Value::String("real_routed_activation_capture".into()),
        );
        capture.insert("not_synthetic_unit_direction".into(), Value::Bool(true));
        header.insert("activation_capture".into(), Value::Object(capture));
        let encoded = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut payload = Vec::new();
        payload.extend_from_slice(&HGRAVS01_MAGIC);
        payload.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
        payload.extend_from_slice(&encoded);
        payload.extend_from_slice(&left_body);
        payload.extend_from_slice(&right_body);
        payload
    }

    #[test]
    fn hgravs01_parser_and_native_matvec_match_dense_product_without_token_path_reconstruction() {
        let payload = hgravs_fixture(4, 8, 2, 4);
        let header = parse_hgravs01_header(&payload).unwrap();
        assert_eq!(header.matrix_shape, [4, 8]);
        assert_eq!(header.rank, 2);
        let input = [0.5, -0.25, 0.125, 1.0, -1.0, 0.0, 0.75, -0.5];
        let input_f64: Vec<f64> = input.iter().map(|value| f64::from(*value)).collect();
        let (_, native) = hgravs01_matvec_f64(&payload, &input_f64).unwrap();
        let (_, dense) = decode_hgravs01_dense_f32_for_parity(&payload).unwrap();
        let mut reconstructed = Vec::with_capacity(4);
        for row in 0..4 {
            let mut sum = 0.0f64;
            for col in 0..8 {
                sum += f64::from(dense[row * 8 + col]) * input_f64[col];
            }
            reconstructed.push(sum);
        }
        for (native_value, dense_value) in native.iter().zip(reconstructed) {
            assert!((native_value - dense_value).abs() < 1e-5);
        }
        // Direct HQ30G1B1 path must refuse the payload.
        assert!(crate::model::qwen_complete_binary::complete_binary_matvec_f64(
            &payload,
            &input_f64
        )
        .is_err());
    }
}

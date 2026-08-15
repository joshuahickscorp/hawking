//! Qwen80 uniform-Q4 group-64 streaming packer and admission.
//!
//! Codec is identical to the proven Q30 HQ30UQ4 path (`uniform_q4.rs` +
//! `qwen_uniform_q4.metal`): 4-bit offset-binary codes and one FP16 scale
//! per 64 flat elements (nominal 4.25 BPW; campaign figure 4.256 includes
//! per-tensor header amortization on the 79.67B catalog). The runtime's
//! uniform-q4 kernels and device table apply unchanged.
//!
//! Q80 catalog identity (not codec) is different: 74,391 tensors, 512 experts
//! top-10 + shared expert, 36 DeltaNet + 12 GQA layers. This module does not
//! change the Q30 path, the codec numerics, or the Q80 runtime.
//!
//! # Streaming contract
//!
//! The packer never holds the ~160 GB BF16 source tree and the ~42 GB Q4
//! catalog at once. Peak RAM is bounded by:
//!
//!   `workers × (one source tensor + one f32 workspace + one Q4 payload)
//!    + shard-header metadata`
//!
//! Sequence per tensor: range-read shard slice → widen → quantize group-64 →
//! atomic-write payload → drop raw, f32, and payload. Shard file descriptors
//! are opened per shard and closed when that shard's tensors are done.
//!
//! Cosine versus the widened BF16 (or F16/F32) source is computed on the
//! live tensor *before* eviction and recorded on the catalog row.
//!
//! W3.T4 full pack (~42 GB out) is `pack_qwen80_uniform_q4` with
//! `require_full_qwen80_catalog = true`, invoked by the
//! `ascension_qwen80_uniform_q4_repack` example. This task must not run it.

use super::*;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
#[cfg(not(unix))]
use std::io::{Seek, SeekFrom};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::sync::Arc;

/// Campaign figure for the complete 79.67B physical catalog (codec 4.25 plus
/// header amortization). The on-the-wire group body is exactly 4.25 BPW when
/// `elements % 64 == 0`.
pub const QWEN80_UNIFORM_Q4_CAMPAIGN_BPW: f64 = 4.256;
/// 4-bit codes + one FP16 scale / 64 elements. Identical to Q30.
pub const UNIFORM_Q4_NOMINAL_BPW: f64 = 4.0 + 16.0 / UNIFORM_Q4_GROUP_SIZE as f64;
pub const QWEN80_UNIFORM_Q4_SCHEMA: &str =
    "hawking.ascension.qwen80_uniform_q4_group64_candidate.v1";
pub const QWEN80_UNIFORM_Q4_CANDIDATE_STATUS: &str =
    "CANDIDATE_UNIFORM_Q4_GROUP64_DIAGNOSTIC_UNQUALIFIED";
pub const QWEN80_UNIFORM_Q4_MODEL_ID: &str = "Qwen3-Coder-Next-uniform-q4-group64-v1";
pub const QWEN80_UNIFORM_Q4_BRANCH_ID: &str = "qwen80-uniform-q4-group64-v1";
pub const QWEN80_UNIFORM_Q4_ARTIFACT_PREFIX: &str = "QWEN80_UNIFORM_Q4_GROUP64_V1";
pub const QWEN80_UNIFORM_Q4_EXPECTED_TENSOR_COUNT: usize = 74_391;
pub const QWEN80_UNIFORM_Q4_TENSOR_EXT: &str = "hq80uq4";
const QWEN80_UNIFORM_Q4_TERMINAL_SCHEMA: &str =
    "hawking.ascension.complete_binary_terminal_status.v1";
const QWEN80_UNIFORM_Q4_COMPLETE_PHASE: &str =
    "EARNED_COMPLETE_PHYSICAL_UNIFORM_Q4_CANDIDATE_UNQUALIFIED";

/// Protected seals for `admit_qwen80_uniform_q4`. Same ceiling as Q30:
/// lowercase SHA-256 seals plus a non-empty source revision. A self-consistent
/// reseal does not pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen80UniformQ4Admission {
    pub expected_manifest_seal_sha256: String,
    pub expected_source_audit_seal_sha256: String,
    pub expected_source_revision: String,
    pub expected_revalidation_path: PathBuf,
    pub expected_revalidation_seal_sha256: String,
    pub expected_terminal_path: PathBuf,
    pub expected_terminal_seal_sha256: String,
}

/// Per-tensor pack quality versus the widened source, computed before eviction.
#[derive(Clone, Debug, PartialEq)]
pub struct UniformQ4PackQuality {
    pub cosine: f64,
    pub relative_l2: f64,
    pub rmse: f64,
    pub elements: usize,
    pub groups: usize,
    pub payload_bytes: usize,
    pub codec_bpw: f64,
}

/// One streamed catalog row. The Q4 payload is on disk; this row does not
/// retain source or packed bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen80UniformQ4PackedTensor {
    pub tensor_name: String,
    pub source_shard: String,
    pub source_shard_sha256: String,
    pub source_dtype: String,
    pub shape: Vec<usize>,
    pub elements: u64,
    pub artifact_path: PathBuf,
    pub artifact_bytes: u64,
    pub artifact_sha256: String,
    pub quality: UniformQ4PackQuality,
}

/// Inputs for the streaming catalog packer.
#[derive(Clone, Debug)]
pub struct Qwen80UniformQ4PackRequest {
    pub output_root: PathBuf,
    pub revalidation_path: PathBuf,
    /// When true (W3.T4), refuse unless the sealed source index has exactly
    /// [`QWEN80_UNIFORM_Q4_EXPECTED_TENSOR_COUNT`] tensors.
    pub require_full_qwen80_catalog: bool,
    /// Concurrent tensor jobs. Each worker holds at most one source tensor.
    /// `1` is the strict single-tensor residency bound.
    pub workers: usize,
}

/// Result of a streaming pack. Payloads live on disk under `output_root/tensors`.
#[derive(Clone, Debug)]
pub struct Qwen80UniformQ4PackReport {
    pub manifest_path: PathBuf,
    pub terminal_path: PathBuf,
    pub manifest_seal_sha256: String,
    pub terminal_seal_sha256: String,
    pub tensor_count: usize,
    pub source_weight_elements: u64,
    pub tensor_payload_bytes: u64,
    pub mean_component_cosine: f64,
    pub packed: Vec<Qwen80UniformQ4PackedTensor>,
}

fn q80_error(detail: impl Into<String>) -> Error {
    Error::Model(format!("qwen80 uniform Q4: {}", detail.into()))
}

/// Payload size of one HQ30UQ4 tensor: 32-byte header + 4×rank + 2×G + 32×G.
pub fn uniform_q4_group64_payload_bytes(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() {
        return Err(q80_error("shape must not be empty"));
    }
    let elements = shape.iter().try_fold(1usize, |total, dimension| {
        if *dimension == 0 {
            return Err(q80_error("shape dimensions must be positive"));
        }
        total
            .checked_mul(*dimension)
            .ok_or_else(|| q80_error("shape element product overflows usize"))
    })?;
    let groups = elements
        .checked_add(UNIFORM_Q4_GROUP_SIZE - 1)
        .ok_or_else(|| q80_error("group count overflow"))?
        / UNIFORM_Q4_GROUP_SIZE;
    let header = COMPLETE_BINARY_HEADER_BYTES
        .checked_add(
            shape
                .len()
                .checked_mul(4)
                .ok_or_else(|| q80_error("rank byte count overflow"))?,
        )
        .ok_or_else(|| q80_error("header size overflow"))?;
    let body = groups
        .checked_mul(2 + UNIFORM_Q4_CODE_BYTES_PER_GROUP)
        .ok_or_else(|| q80_error("body size overflow"))?;
    header
        .checked_add(body)
        .ok_or_else(|| q80_error("payload size overflow"))
}

/// IEEE-754 roundTiesToEven, matching NumPy `rint` used by the Q30 packer.
fn rint_ties_even(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    let truncated = value.trunc();
    let fraction = value - truncated;
    let abs_fraction = fraction.abs();
    if abs_fraction < 0.5 {
        return truncated;
    }
    if abs_fraction > 0.5 {
        return truncated + value.signum();
    }
    if (truncated as i64) % 2 == 0 {
        truncated
    } else {
        truncated + value.signum()
    }
}

/// Pack one tensor with the Q30 HQ30UQ4 group-64 codec.
///
/// Flat groups of 64; stored scale is `f16(max_abs/7)` (the FP16 value is
/// authority); `q ∈ [-8, 7]`; even local index in the low nibble, odd in the
/// high nibble. Does not retain the source after return.
pub fn pack_uniform_q4_group64(
    values: &[f32],
    shape: &[usize],
) -> Result<(Vec<u8>, UniformQ4PackQuality)> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(q80_error("source tensor contains a non-finite value"));
    }
    let elements = shape.iter().try_fold(1usize, |total, dimension| {
        if *dimension == 0 {
            return Err(q80_error("shape dimensions must be positive"));
        }
        total
            .checked_mul(*dimension)
            .ok_or_else(|| q80_error("shape element product overflows usize"))
    })?;
    if values.len() != elements {
        return Err(q80_error(format!(
            "value count {} disagrees with shape product {elements}",
            values.len()
        )));
    }
    if shape.len() > u16::MAX as usize {
        return Err(q80_error("rank exceeds u16"));
    }
    let groups = elements
        .checked_add(UNIFORM_Q4_GROUP_SIZE - 1)
        .ok_or_else(|| q80_error("group count overflow"))?
        / UNIFORM_Q4_GROUP_SIZE;
    let mut scales_bits = vec![0u16; groups];
    let mut codes = vec![0u8; groups * UNIFORM_Q4_CODE_BYTES_PER_GROUP];
    let mut reconstructed = vec![0.0f32; elements];
    for group in 0..groups {
        let start = group * UNIFORM_Q4_GROUP_SIZE;
        let end = (start + UNIFORM_Q4_GROUP_SIZE).min(elements);
        let mut max_abs = 0.0f32;
        for value in &values[start..end] {
            max_abs = max_abs.max(value.abs());
        }
        let scale = f16::from_f32(max_abs / 7.0);
        if !scale.is_finite() {
            return Err(q80_error(format!(
                "group {group} scale is not a finite FP16"
            )));
        }
        let reconstructed_scale = scale.to_f32();
        scales_bits[group] = scale.to_bits();
        let code_base = group * UNIFORM_Q4_CODE_BYTES_PER_GROUP;
        for (local, &value) in values[start..end].iter().enumerate() {
            let quantized = if reconstructed_scale == 0.0 {
                0i32
            } else {
                rint_ties_even(value / reconstructed_scale)
                    .clamp(-8.0, 7.0) as i32
            };
            let code = (quantized + 8) as u8;
            if local & 1 == 0 {
                codes[code_base + local / 2] |= code;
            } else {
                codes[code_base + local / 2] |= code << 4;
            }
            reconstructed[start + local] = quantized as f32 * reconstructed_scale;
        }
    }
    let rank = shape.len() as u16;
    let mut payload = Vec::with_capacity(uniform_q4_group64_payload_bytes(shape)?);
    payload.extend_from_slice(&UNIFORM_Q4_MAGIC);
    payload.extend_from_slice(&UNIFORM_Q4_VERSION.to_le_bytes());
    payload.extend_from_slice(&(UNIFORM_Q4_GROUP_SIZE as u32).to_le_bytes());
    payload.extend_from_slice(&rank.to_le_bytes());
    payload.extend_from_slice(&0u16.to_le_bytes());
    payload.extend_from_slice(&(elements as u64).to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    for dimension in shape {
        payload.extend_from_slice(&(*dimension as u32).to_le_bytes());
    }
    for bits in &scales_bits {
        payload.extend_from_slice(&bits.to_le_bytes());
    }
    payload.extend_from_slice(&codes);
    let expected = uniform_q4_group64_payload_bytes(shape)?;
    if payload.len() != expected {
        return Err(q80_error(format!(
            "payload size {} != expected {expected}",
            payload.len()
        )));
    }
    // Drop working code/scale buffers before the quality pass so peak RAM is
    // source + reconstructed + payload, not those plus the nibble workspace.
    drop(codes);
    drop(scales_bits);
    let quality = pack_quality(values, &reconstructed, elements, groups, payload.len());
    drop(reconstructed);
    Ok((payload, quality))
}

fn pack_quality(
    source: &[f32],
    reconstructed: &[f32],
    elements: usize,
    groups: usize,
    payload_bytes: usize,
) -> UniformQ4PackQuality {
    let mut dot = 0.0f64;
    let mut source_sq = 0.0f64;
    let mut recon_sq = 0.0f64;
    let mut err_sq = 0.0f64;
    for (left, right) in source.iter().zip(reconstructed) {
        let a = f64::from(*left);
        let b = f64::from(*right);
        let err = a - b;
        dot += a * b;
        source_sq += a * a;
        recon_sq += b * b;
        err_sq += err * err;
    }
    let source_norm = source_sq.sqrt().max(1e-12);
    let recon_norm = recon_sq.sqrt().max(1e-12);
    let codec_bits = groups * (UNIFORM_Q4_CODE_BYTES_PER_GROUP * 8 + 16);
    UniformQ4PackQuality {
        cosine: dot / (source_norm * recon_norm),
        relative_l2: err_sq.sqrt() / source_norm,
        rmse: (err_sq / elements.max(1) as f64).sqrt(),
        elements,
        groups,
        payload_bytes,
        codec_bpw: codec_bits as f64 / elements.max(1) as f64,
    }
}

/// Decode an HQ30UQ4 payload back to f32 for cosine / layout checks.
pub fn decode_uniform_q4_group64(payload: &[u8]) -> Result<Vec<f32>> {
    let header = parse_uniform_q4_header(payload)?;
    let mut values = vec![0.0f32; header.elements];
    for element in 0..header.elements {
        let group = element / UNIFORM_Q4_GROUP_SIZE;
        let local = element % UNIFORM_Q4_GROUP_SIZE;
        let scale = f16::from_bits(read_u16(
            payload,
            header.scale_offset + group * 2,
        )?)
        .to_f32();
        let packed = payload[header.sign_offset + group * UNIFORM_Q4_CODE_BYTES_PER_GROUP + local / 2];
        let nibble = if local & 1 == 0 {
            packed & 0x0f
        } else {
            packed >> 4
        };
        values[element] = (nibble as i32 - 8) as f32 * scale;
    }
    Ok(values)
}

fn expected_q80_q4_tensor_path(root: &Path, tensor_name: &str) -> Result<PathBuf> {
    let tensors = root.join("tensors");
    let metadata = fs::symlink_metadata(&tensors).map_err(|error| {
        q80_error(format!(
            "cannot stat {}: {error}",
            tensors.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(q80_error(format!(
            "{} must be a non-symlink tensors directory",
            tensors.display()
        )));
    }
    Ok(tensors.join(format!(
        "{}.{QWEN80_UNIFORM_Q4_TENSOR_EXT}",
        sha256_hex(tensor_name.as_bytes())
    )))
}

fn widen_source_to_f32(raw: &[u8], dtype: &str, elements: usize) -> Result<Vec<f32>> {
    match dtype {
        "BF16" | "BFLOAT16" => {
            let expect = elements
                .checked_mul(2)
                .ok_or_else(|| q80_error("BF16 byte count overflow"))?;
            if raw.len() != expect {
                return Err(q80_error(format!(
                    "BF16 payload {} bytes != expected {expect}",
                    raw.len()
                )));
            }
            let mut out = vec![0.0f32; elements];
            for (index, slot) in out.iter_mut().enumerate() {
                let bits = u16::from_le_bytes([raw[index * 2], raw[index * 2 + 1]]);
                *slot = f32::from_bits(u32::from(bits) << 16);
            }
            Ok(out)
        }
        "F16" | "FLOAT16" => {
            let expect = elements
                .checked_mul(2)
                .ok_or_else(|| q80_error("F16 byte count overflow"))?;
            if raw.len() != expect {
                return Err(q80_error(format!(
                    "F16 payload {} bytes != expected {expect}",
                    raw.len()
                )));
            }
            let mut out = vec![0.0f32; elements];
            for (index, slot) in out.iter_mut().enumerate() {
                let bits = u16::from_le_bytes([raw[index * 2], raw[index * 2 + 1]]);
                *slot = f16::from_bits(bits).to_f32();
            }
            Ok(out)
        }
        "F32" | "FLOAT32" => {
            let expect = elements
                .checked_mul(4)
                .ok_or_else(|| q80_error("F32 byte count overflow"))?;
            if raw.len() != expect {
                return Err(q80_error(format!(
                    "F32 payload {} bytes != expected {expect}",
                    raw.len()
                )));
            }
            let mut out = vec![0.0f32; elements];
            for (index, slot) in out.iter_mut().enumerate() {
                let base = index * 4;
                *slot = f32::from_le_bytes([raw[base], raw[base + 1], raw[base + 2], raw[base + 3]]);
            }
            Ok(out)
        }
        other => Err(q80_error(format!(
            "unsupported source dtype {other:?}"
        ))),
    }
}

fn atomic_write_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| q80_error("payload path has no parent"))?;
    fs::create_dir_all(parent).map_err(|error| {
        q80_error(format!("cannot create {}: {error}", parent.display()))
    })?;
    let tmp = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("payload"),
        std::process::id()
    ));
    let write_result = (|| -> Result<()> {
        let mut file = File::create(&tmp).map_err(|error| {
            q80_error(format!("cannot create {}: {error}", tmp.display()))
        })?;
        file.write_all(bytes).map_err(|error| {
            q80_error(format!("cannot write {}: {error}", tmp.display()))
        })?;
        file.sync_all().map_err(|error| {
            q80_error(format!("cannot fsync {}: {error}", tmp.display()))
        })?;
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_file(&tmp);
        return Err(error);
    }
    fs::rename(&tmp, path).map_err(|error| {
        let _ = fs::remove_file(&tmp);
        q80_error(format!(
            "cannot publish {} -> {}: {error}",
            tmp.display(),
            path.display()
        ))
    })?;
    Ok(())
}

fn write_pretty_json(path: &Path, value: &Value) -> Result<usize> {
    let raw = format!(
        "{}\n",
        serde_json::to_string_pretty(value).map_err(|error| {
            q80_error(format!("cannot serialize {}: {error}", path.display()))
        })?
    );
    atomic_write_bytes(path, raw.as_bytes())?;
    Ok(raw.len())
}

fn seal_value(mut value: Value) -> Result<Value> {
    let object = value
        .as_object_mut()
        .ok_or_else(|| q80_error("sealed document root must be an object"))?;
    object.remove("seal_sha256");
    let digest = sha256_hex(&canonical_json(&Value::Object(object.clone()))?);
    object.insert("seal_sha256".into(), Value::String(digest));
    Ok(value)
}

struct SafetensorsTensorMeta {
    dtype: String,
    shape: Vec<usize>,
    begin: u64,
    end: u64,
    header_nbytes: u64,
}

fn read_safetensors_meta(path: &Path) -> Result<HashMap<String, SafetensorsTensorMeta>> {
    let mut file = File::open(path).map_err(|error| {
        q80_error(format!("cannot open {}: {error}", path.display()))
    })?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf).map_err(|error| {
        q80_error(format!(
            "cannot read safetensors header length of {}: {error}",
            path.display()
        ))
    })?;
    let header_nbytes = u64::from_le_bytes(len_buf);
    if header_nbytes == 0 || header_nbytes > 64 * 1024 * 1024 {
        return Err(q80_error(format!(
            "implausible safetensors header length {header_nbytes} in {}",
            path.display()
        )));
    }
    let mut raw = vec![0u8; header_nbytes as usize];
    file.read_exact(&mut raw).map_err(|error| {
        q80_error(format!(
            "cannot read safetensors header of {}: {error}",
            path.display()
        ))
    })?;
    let value: Value = serde_json::from_slice(&raw).map_err(|error| {
        q80_error(format!(
            "safetensors header JSON invalid in {}: {error}",
            path.display()
        ))
    })?;
    let object = value.as_object().ok_or_else(|| {
        q80_error(format!(
            "safetensors header is not an object in {}",
            path.display()
        ))
    })?;
    let mut tensors = HashMap::new();
    for (name, info_value) in object {
        if name == "__metadata__" {
            continue;
        }
        let info = info_value.as_object().ok_or_else(|| {
            q80_error(format!("tensor {name} header is not an object"))
        })?;
        let dtype = info
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| q80_error(format!("tensor {name} lacks dtype")))?
            .to_owned();
        let shape = info
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| q80_error(format!("tensor {name} lacks shape")))?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|number| usize::try_from(number).ok())
                    .ok_or_else(|| q80_error(format!("tensor {name} has a non-integer shape")))
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = info
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| q80_error(format!("tensor {name} lacks data_offsets")))?;
        if offsets.len() != 2 {
            return Err(q80_error(format!(
                "tensor {name} data_offsets is not a pair"
            )));
        }
        let begin = offsets[0]
            .as_u64()
            .ok_or_else(|| q80_error(format!("tensor {name} data_offsets[0] is not u64")))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| q80_error(format!("tensor {name} data_offsets[1] is not u64")))?;
        if end < begin {
            return Err(q80_error(format!(
                "tensor {name} has inverted data_offsets"
            )));
        }
        tensors.insert(
            name.clone(),
            SafetensorsTensorMeta {
                dtype,
                shape,
                begin,
                end,
                header_nbytes,
            },
        );
    }
    Ok(tensors)
}

fn range_read_tensor(file: &File, meta: &SafetensorsTensorMeta, name: &str) -> Result<Vec<u8>> {
    let nbytes = usize::try_from(meta.end - meta.begin)
        .map_err(|_| q80_error(format!("tensor {name} byte count exceeds usize")))?;
    let offset = 8u64
        .checked_add(meta.header_nbytes)
        .and_then(|base| base.checked_add(meta.begin))
        .ok_or_else(|| q80_error(format!("tensor {name} range offset overflow")))?;
    let mut buf = vec![0u8; nbytes];
    #[cfg(unix)]
    {
        file.read_exact_at(&mut buf, offset).map_err(|error| {
            q80_error(format!(
                "range-read {name} ({nbytes} bytes) @ {offset}: {error}"
            ))
        })?;
    }
    #[cfg(not(unix))]
    {
        let mut file = file
            .try_clone()
            .map_err(|error| q80_error(format!("cannot clone shard handle: {error}")))?;
        file.seek(SeekFrom::Start(offset)).map_err(|error| {
            q80_error(format!("seek {name} @ {offset}: {error}"))
        })?;
        file.read_exact(&mut buf).map_err(|error| {
            q80_error(format!("range-read {name} ({nbytes} bytes): {error}"))
        })?;
    }
    Ok(buf)
}

struct PackJob {
    tensor_name: String,
    shard_name: String,
    shard_path: PathBuf,
    shard_sha256: String,
    meta: SafetensorsTensorMeta,
}

fn pack_one_job(job: &PackJob, tensor_dir: &Path) -> Result<Qwen80UniformQ4PackedTensor> {
    let file = File::open(&job.shard_path).map_err(|error| {
        q80_error(format!(
            "cannot open {}: {error}",
            job.shard_path.display()
        ))
    })?;
    let raw = range_read_tensor(&file, &job.meta, &job.tensor_name)?;
    drop(file);
    let elements = job.meta.shape.iter().try_fold(1usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or_else(|| q80_error("shape element product overflows usize"))
    })?;
    let values = widen_source_to_f32(&raw, &job.meta.dtype, elements)?;
    drop(raw);
    let (payload, quality) = pack_uniform_q4_group64(&values, &job.meta.shape)?;
    drop(values);
    let artifact_path = tensor_dir.join(format!(
        "{}.{QWEN80_UNIFORM_Q4_TENSOR_EXT}",
        sha256_hex(job.tensor_name.as_bytes())
    ));
    atomic_write_bytes(&artifact_path, &payload)?;
    let artifact_sha256 = sha256_hex(&payload);
    let artifact_bytes = u64::try_from(payload.len()).unwrap_or(u64::MAX);
    drop(payload);
    Ok(Qwen80UniformQ4PackedTensor {
        tensor_name: job.tensor_name.clone(),
        source_shard: job.shard_name.clone(),
        source_shard_sha256: job.shard_sha256.clone(),
        source_dtype: job.meta.dtype.clone(),
        shape: job.meta.shape.clone(),
        elements: u64::try_from(elements).unwrap_or(u64::MAX),
        artifact_path,
        artifact_bytes,
        artifact_sha256,
        quality,
    })
}

fn load_source_for_pack(revalidation_path: &Path) -> Result<SourceChain> {
    let revalidation_path =
        canonical_regular_path(revalidation_path, "qwen80 uniform Q4 revalidation")?;
    let revalidation_raw =
        read_regular_file(&revalidation_path, "qwen80 uniform Q4 revalidation")?;
    let revalidation =
        parse_json_no_duplicate_keys(&revalidation_raw, "qwen80 uniform Q4 revalidation")?;
    let revalidation_seal =
        verify_sealed_document(&revalidation, "qwen80 uniform Q4 revalidation")?;
    let revalidation_object = revalidation.as_object().ok_or_else(|| {
        q80_error("revalidation root must be an object")
    })?;
    let audit_seal = required_sha256(
        revalidation_object,
        "source_audit_seal_sha256",
        "qwen80 uniform Q4 revalidation",
    )?;
    let revision = required_string(
        revalidation_object,
        "source_revision",
        "qwen80 uniform Q4 revalidation",
    )?
    .to_owned();
    let admission = CompleteBinaryAdmission {
        model: QwenCompleteBinaryModel::Qwen80CoderNext,
        expected_manifest_seal_sha256: "0".repeat(64),
        expected_source_audit_seal_sha256: audit_seal.clone(),
        expected_source_revision: revision,
    };
    let parent = revalidation_path
        .parent()
        .ok_or_else(|| q80_error("revalidation has no parent"))?;
    let source = validate_source_chain(
        &revalidation,
        &revalidation_path,
        parent,
        &admission,
        &audit_seal,
    )?;
    let _ = revalidation_seal;
    Ok(source)
}

fn q4_layout_object() -> Value {
    json!({
        "magic": "HQ30UQ4\0",
        "version": UNIFORM_Q4_VERSION,
        "group_size": UNIFORM_Q4_GROUP_SIZE,
        "nibble_order": "even_low_odd_high",
        "q_range": "[-8,7]",
        "scale_dtype": "float16",
        "scale_rule": "max_abs_div_7_fp16_authority",
    })
}

fn catalog_row(row: &Qwen80UniformQ4PackedTensor) -> Value {
    json!({
        "tensor_name": row.tensor_name,
        "source_shard": row.source_shard,
        "source_shard_sha256": row.source_shard_sha256,
        "source_dtype": row.source_dtype,
        "shape": row.shape,
        "elements": row.elements,
        "artifact_path": row.artifact_path,
        "artifact_bytes": row.artifact_bytes,
        "artifact_sha256": row.artifact_sha256,
        "layout": q4_layout_object(),
        "component_quality": {
            "cosine": row.quality.cosine,
            "relative_l2": row.quality.relative_l2,
            "rmse": row.quality.rmse,
            "codec_bpw": row.quality.codec_bpw,
            "finite": true,
        },
    })
}

/// Stream-pack a Qwen80 (or Qwen80-shaped fixture) source index as HQ30UQ4.
///
/// Source shards are range-read one tensor at a time and evicted before the
/// next tensor is opened. The sealed catalog binds the same source-audit /
/// revalidation authority that admission will re-check.
pub fn pack_qwen80_uniform_q4(
    request: &Qwen80UniformQ4PackRequest,
) -> Result<Qwen80UniformQ4PackReport> {
    if request.workers == 0 {
        return Err(q80_error("workers must be at least 1"));
    }
    let source = load_source_for_pack(&request.revalidation_path)?;
    if request.require_full_qwen80_catalog
        && source.weight_map.len() != QWEN80_UNIFORM_Q4_EXPECTED_TENSOR_COUNT
    {
        return Err(q80_error(format!(
            "full Qwen80 catalog requires {QWEN80_UNIFORM_Q4_EXPECTED_TENSOR_COUNT} source tensors, found {}",
            source.weight_map.len()
        )));
    }
    let model_dir = source.source_index_path.parent().ok_or_else(|| {
        q80_error("source index has no parent model directory")
    })?;
    let output_root = request.output_root.clone();
    let tensor_dir = output_root.join("tensors");
    fs::create_dir_all(&tensor_dir).map_err(|error| {
        q80_error(format!(
            "cannot create {}: {error}",
            tensor_dir.display()
        ))
    })?;

    let mut shard_meta: HashMap<String, HashMap<String, SafetensorsTensorMeta>> = HashMap::new();
    for shard in source.weight_map.values() {
        if shard_meta.contains_key(shard) {
            continue;
        }
        shard_meta.insert(shard.clone(), read_safetensors_meta(&model_dir.join(shard))?);
    }

    let mut jobs = Vec::with_capacity(source.weight_map.len());
    for (name, shard) in &source.weight_map {
        let meta = shard_meta
            .get(shard)
            .and_then(|map| map.get(name))
            .ok_or_else(|| {
                q80_error(format!(
                    "source shard {shard} has no header for tensor {name}"
                ))
            })?;
        let shard_sha256 = source.shard_hashes.get(shard).cloned().ok_or_else(|| {
            q80_error(format!("revalidation has no hash for shard {shard}"))
        })?;
        jobs.push(PackJob {
            tensor_name: name.clone(),
            shard_name: shard.clone(),
            shard_path: model_dir.join(shard),
            shard_sha256,
            meta: SafetensorsTensorMeta {
                dtype: meta.dtype.clone(),
                shape: meta.shape.clone(),
                begin: meta.begin,
                end: meta.end,
                header_nbytes: meta.header_nbytes,
            },
        });
    }
    drop(shard_meta);

    let mut packed = if request.workers <= 1 {
        let mut rows = Vec::with_capacity(jobs.len());
        for job in &jobs {
            rows.push(pack_one_job(job, &tensor_dir)?);
        }
        rows
    } else {
        let job_queue = Arc::new(std::sync::Mutex::new(jobs.into_iter()));
        let workers = request.workers.min(source.weight_map.len()).max(1);
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let queue = Arc::clone(&job_queue);
            let tensor_dir = tensor_dir.clone();
            handles.push(thread::spawn(move || -> Result<Vec<Qwen80UniformQ4PackedTensor>> {
                let mut local = Vec::new();
                loop {
                    let job = {
                        let mut guard = queue.lock().map_err(|_| {
                            q80_error("packer worker queue poisoned")
                        })?;
                        guard.next()
                    };
                    match job {
                        Some(job) => local.push(pack_one_job(&job, &tensor_dir)?),
                        None => break,
                    }
                }
                Ok(local)
            }));
        }
        let mut rows = Vec::new();
        for handle in handles {
            let local = handle
                .join()
                .map_err(|_| q80_error("packer worker panicked"))??;
            rows.extend(local);
        }
        rows
    };
    packed.sort_by(|left, right| left.tensor_name.cmp(&right.tensor_name));
    if packed.len() != source.weight_map.len() {
        return Err(q80_error(format!(
            "packed {} tensors, source index has {}",
            packed.len(),
            source.weight_map.len()
        )));
    }

    let mean_cosine = if packed.is_empty() {
        0.0
    } else {
        packed.iter().map(|row| row.quality.cosine).sum::<f64>() / packed.len() as f64
    };
    let payload_bytes = packed.iter().try_fold(0u64, |sum, row| {
        sum.checked_add(row.artifact_bytes)
            .ok_or_else(|| q80_error("payload byte sum overflow"))
    })?;
    let elements = packed.iter().try_fold(0u64, |sum, row| {
        sum.checked_add(row.elements)
            .ok_or_else(|| q80_error("element sum overflow"))
    })?;

    let revalidation_path =
        canonical_regular_path(&request.revalidation_path, "qwen80 uniform Q4 revalidation")?;
    let revalidation_raw =
        read_regular_file(&revalidation_path, "qwen80 uniform Q4 revalidation")?;
    let revalidation =
        parse_json_no_duplicate_keys(&revalidation_raw, "qwen80 uniform Q4 revalidation")?;
    let revalidation_seal =
        verify_sealed_document(&revalidation, "qwen80 uniform Q4 revalidation")?;
    let revalidation_object = revalidation
        .as_object()
        .ok_or_else(|| q80_error("revalidation root must be an object"))?;
    let declared_model_dir = required_string(
        revalidation_object,
        "source_model_dir",
        "qwen80 uniform Q4 revalidation",
    )?;

    let manifest_path = output_root.join(format!(
        "{QWEN80_UNIFORM_Q4_ARTIFACT_PREFIX}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
    ));
    let terminal_path = output_root.join(format!(
        "{QWEN80_UNIFORM_Q4_ARTIFACT_PREFIX}_COMPLETE_GRAVITY_TERMINAL_RECEIPT.json"
    ));
    let draft_tensors: Vec<Value> = packed.iter().map(catalog_row).collect();

    let make_manifest = |billed: u64| -> Value {
        let total = payload_bytes + billed;
        let physical_bpw = if elements == 0 {
            0.0
        } else {
            (total as f64 * 8.0) / elements as f64
        };
        json!({
            "schema": QWEN80_UNIFORM_Q4_SCHEMA,
            "status": QWEN80_UNIFORM_Q4_CANDIDATE_STATUS,
            "branch_id": QWEN80_UNIFORM_Q4_BRANCH_ID,
            "model_id": QWEN80_UNIFORM_Q4_MODEL_ID,
            "artifact_prefix": QWEN80_UNIFORM_Q4_ARTIFACT_PREFIX,
            "source_body_audit_seal_sha256": source.source_audit_seal_sha256,
            "source_revalidation_receipt_path": revalidation_path,
            "source_revalidation_receipt_seal_sha256": revalidation_seal,
            "source": {
                "repository": QwenCompleteBinaryModel::Qwen80CoderNext.source_repository(),
                "model_dir": declared_model_dir,
                "tensor_count": packed.len(),
            },
            "representation": {
                "family": "uniform_q4_group64_fp16_scale",
                "group_size": UNIFORM_Q4_GROUP_SIZE,
                "bits_per_weight": 4,
                "nominal_bpw": UNIFORM_Q4_NOMINAL_BPW,
                "campaign_bpw": QWEN80_UNIFORM_Q4_CAMPAIGN_BPW,
                "physical_direct_layout": true,
            },
            "complete_physical_bpw_ledger": {
                "source_weight_elements": elements,
                "tensor_payload_bytes": payload_bytes,
                "manifest_bytes_billed": billed,
                "all_required_weight_artifact_bytes": total,
                "complete_physical_bpw": physical_bpw,
                "threshold_bpw": 1.5,
                "nominal_codec_bpw": UNIFORM_Q4_NOMINAL_BPW,
                "passes_storage_threshold": physical_bpw <= 1.5,
            },
            "quality_summary": {
                "mean_component_cosine": mean_cosine,
                "quality_rows_with_cosine": packed.len(),
                "verdict": "DIAGNOSTIC_UNIFORM_Q4_GROUP64_COHERENCE_PROBE",
            },
            "claim_boundary": {
                "complete_physical_tensor_coverage_is_true": request.require_full_qwen80_catalog,
                "diagnostic_uniform_q4_not_production_promotion": true,
                "weights_remain_packed_q4_at_token_time": true,
                "full_74391_pack_is_w3_t4_not_this_authoring_lane": true,
            },
            "tensors": draft_tensors,
        })
    };

    let mut billed = 0u64;
    let sealed_manifest = loop {
        let sealed = seal_value(make_manifest(billed))?;
        let actual = u64::try_from(
            format!(
                "{}\n",
                serde_json::to_string_pretty(&sealed).map_err(|error| {
                    q80_error(format!("cannot render catalog: {error}"))
                })?
            )
            .len(),
        )
        .unwrap_or(u64::MAX);
        if actual == billed {
            break sealed;
        }
        billed = actual;
    };
    write_pretty_json(&manifest_path, &sealed_manifest)?;
    let manifest_seal = sealed_manifest
        .get("seal_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| q80_error("sealed catalog missing seal"))?
        .to_owned();

    let terminal = seal_value(json!({
        "schema": QWEN80_UNIFORM_Q4_TERMINAL_SCHEMA,
        "status": QWEN80_UNIFORM_Q4_COMPLETE_PHASE,
        "binding": {
            "model_id": QWEN80_UNIFORM_Q4_MODEL_ID,
            "artifact_prefix": QWEN80_UNIFORM_Q4_ARTIFACT_PREFIX,
            "manifest_schema": QWEN80_UNIFORM_Q4_SCHEMA,
            "branch_id": QWEN80_UNIFORM_Q4_BRANCH_ID,
            "source_body_audit_seal_sha256": source.source_audit_seal_sha256,
            "source_revalidation_receipt_path": revalidation_path,
            "source_revalidation_receipt_seal_sha256": revalidation_seal,
            "source_revision": source.source_revision,
        },
        "candidate": {
            "manifest_path": manifest_path,
            "manifest_seal_sha256": manifest_seal,
            "all_required_weight_artifact_bytes": payload_bytes + billed,
            "complete_physical_bpw": if elements == 0 {
                0.0
            } else {
                ((payload_bytes + billed) as f64 * 8.0) / elements as f64
            },
            "tensor_count": packed.len(),
            "mean_component_cosine": mean_cosine,
        },
    }))?;
    write_pretty_json(&terminal_path, &terminal)?;
    let terminal_seal = terminal
        .get("seal_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| q80_error("sealed terminal missing seal"))?
        .to_owned();

    Ok(Qwen80UniformQ4PackReport {
        manifest_path,
        terminal_path,
        manifest_seal_sha256: manifest_seal,
        terminal_seal_sha256: terminal_seal,
        tensor_count: packed.len(),
        source_weight_elements: elements,
        tensor_payload_bytes: payload_bytes,
        mean_component_cosine: mean_cosine,
        packed,
    })
}

fn validate_q80_q4_tensor_row(
    row: &Map<String, Value>,
    root: &Path,
    source: &SourceChain,
) -> Result<(CompleteBinaryTensor, Arc<[u8]>)> {
    let label = "qwen80 uniform Q4 manifest tensor";
    let tensor_name = required_string(row, "tensor_name", label)?;
    if tensor_name.contains('\0') {
        return Err(Error::Model(format!(
            "{label}: tensor_name contains a NUL byte"
        )));
    }
    let source_shard = required_string(row, "source_shard", label)?;
    require_safe_filename(source_shard, label)?;
    let source_shard_sha256 = required_sha256(row, "source_shard_sha256", label)?;
    let expected_source_hash = source.shard_hashes.get(source_shard).ok_or_else(|| {
        Error::Model(format!(
            "{label}: source shard {source_shard:?} is not in the sealed source receipt"
        ))
    })?;
    if &source_shard_sha256 != expected_source_hash {
        return Err(Error::Model(format!(
            "{label}: source shard hash for {tensor_name:?} differs from the source receipt"
        )));
    }
    if source.weight_map.get(tensor_name).map(String::as_str) != Some(source_shard) {
        return Err(Error::Model(format!(
            "{label}: source index does not bind tensor {tensor_name:?} to shard {source_shard:?}"
        )));
    }
    let source_dtype = required_string(row, "source_dtype", label)?;
    if !matches!(
        source_dtype,
        "BF16" | "BFLOAT16" | "F32" | "FLOAT32" | "F16" | "FLOAT16"
    ) {
        return Err(Error::Model(format!(
            "{label}: tensor {tensor_name:?} has unsupported source_dtype {source_dtype:?}"
        )));
    }
    let shape = declared_tensor_shape(row, label)?;
    let declared_elements = required_u64(row, "elements", label)?;
    let shape_elements = shape.iter().try_fold(1u64, |total, dimension| {
        total
            .checked_mul(u64::try_from(*dimension).unwrap_or(u64::MAX))
            .ok_or_else(|| Error::Model(format!("{label}: shape element product overflows u64")))
    })?;
    if declared_elements == 0 || declared_elements != shape_elements {
        return Err(Error::Model(format!(
            "{label}: tensor {tensor_name:?} elements does not equal its shape product"
        )));
    }

    let expected_path = expected_q80_q4_tensor_path(root, tensor_name)?;
    let expected_path = canonical_regular_path(&expected_path, label)?;
    let declared_path =
        manifest_descendant_file(root, required_string(row, "artifact_path", label)?, label)?;
    if declared_path != expected_path {
        return Err(Error::Model(format!(
            "{label}: tensor {tensor_name:?} artifact_path does not equal its deterministic tensor path"
        )));
    }
    let payload = read_regular_file(&expected_path, label)?;
    if required_u64(row, "artifact_bytes", label)?
        != u64::try_from(payload.len()).unwrap_or(u64::MAX)
    {
        return Err(Error::Model(format!(
            "{label}: tensor {tensor_name:?} artifact_bytes does not equal physical payload bytes"
        )));
    }
    let artifact_sha256 = required_sha256(row, "artifact_sha256", label)?;
    if sha256_hex(&payload) != artifact_sha256 {
        return Err(Error::Model(format!(
            "{label}: tensor {tensor_name:?} payload SHA-256 does not match the manifest"
        )));
    }
    let layout = required_object(row, "layout", label)?;
    require_exact_string(layout, "magic", "HQ30UQ4\0", label)?;
    if required_u64(layout, "version", label)? != u64::from(UNIFORM_Q4_VERSION)
        || required_u64(layout, "group_size", label)?
            != u64::try_from(UNIFORM_Q4_GROUP_SIZE).unwrap()
    {
        return Err(Error::Model(format!(
            "{label}: tensor {tensor_name:?} layout does not identify version-1 64-value Q4 groups"
        )));
    }
    require_exact_string(layout, "scale_dtype", "float16", label)?;

    let header = parse_uniform_q4_header(&payload)?;
    if header.group_size != UNIFORM_Q4_GROUP_SIZE
        || header.version != UNIFORM_Q4_VERSION
        || header.shape != shape
        || u64::try_from(header.elements).ok() != Some(declared_elements)
    {
        return Err(Error::Model(format!(
            "{label}: tensor {tensor_name:?} manifest geometry disagrees with its Q4 payload header"
        )));
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

/// Admit a sealed Qwen80 uniform-Q4 catalog.
///
/// Binds source-audit seal, source revision, revalidation receipt, terminal
/// receipt, and every packed payload. Does not weaken the Q30 seal ceiling.
pub fn admit_qwen80_uniform_q4(
    manifest_path: impl AsRef<Path>,
    admission: &Qwen80UniformQ4Admission,
) -> Result<CompleteBinaryArtifact> {
    crate::startup_timing::time_ms_result("admit_qwen80_uniform_q4_total", || {
        admit_qwen80_uniform_q4_inner(manifest_path, admission)
    })
}

fn admit_qwen80_uniform_q4_inner(
    manifest_path: impl AsRef<Path>,
    admission: &Qwen80UniformQ4Admission,
) -> Result<CompleteBinaryArtifact> {
    if !is_sha256(&admission.expected_manifest_seal_sha256)
        || !is_sha256(&admission.expected_source_audit_seal_sha256)
        || !is_sha256(&admission.expected_revalidation_seal_sha256)
        || !is_sha256(&admission.expected_terminal_seal_sha256)
        || admission.expected_source_revision.is_empty()
    {
        return Err(Error::Model(
            "qwen80 uniform Q4 admission requires protected lowercase SHA-256 seals and a source revision"
                .into(),
        ));
    }
    let manifest_path =
        canonical_regular_path(manifest_path.as_ref(), "qwen80 uniform Q4 manifest")?;
    let root = manifest_path.parent().ok_or_else(|| {
        Error::Model("qwen80 uniform Q4 manifest has no parent artifact root".into())
    })?;
    let manifest_raw = read_regular_file(&manifest_path, "qwen80 uniform Q4 manifest")?;
    let manifest = parse_json_no_duplicate_keys(&manifest_raw, "qwen80 uniform Q4 manifest")?;
    let manifest_object = manifest.as_object().ok_or_else(|| {
        Error::Model("qwen80 uniform Q4 manifest: root must be an object".into())
    })?;
    let manifest_seal = verify_sealed_document(&manifest, "qwen80 uniform Q4 manifest")?;
    if manifest_seal != admission.expected_manifest_seal_sha256 {
        return Err(Error::Model(
            "qwen80 uniform Q4 manifest seal does not match the protected admission binding".into(),
        ));
    }
    require_exact_string(
        manifest_object,
        "schema",
        QWEN80_UNIFORM_Q4_SCHEMA,
        "qwen80 uniform Q4 manifest",
    )?;
    require_exact_string(
        manifest_object,
        "status",
        QWEN80_UNIFORM_Q4_CANDIDATE_STATUS,
        "qwen80 uniform Q4 manifest",
    )?;
    let manifest_audit_seal = required_sha256(
        manifest_object,
        "source_body_audit_seal_sha256",
        "qwen80 uniform Q4 manifest",
    )?;
    if manifest_audit_seal != admission.expected_source_audit_seal_sha256 {
        return Err(Error::Model(
            "qwen80 uniform Q4 manifest source audit seal does not match protected admission binding"
                .into(),
        ));
    }

    let terminal_path = canonical_regular_path(
        &admission.expected_terminal_path,
        "qwen80 uniform Q4 terminal receipt",
    )?;
    let terminal_raw = read_regular_file(&terminal_path, "qwen80 uniform Q4 terminal receipt")?;
    let terminal = parse_json_no_duplicate_keys(&terminal_raw, "qwen80 uniform Q4 terminal receipt")?;
    let terminal_seal = verify_sealed_document(&terminal, "qwen80 uniform Q4 terminal receipt")?;
    if terminal_seal != admission.expected_terminal_seal_sha256 {
        return Err(Error::Model(
            "qwen80 uniform Q4 terminal seal does not match protected admission binding".into(),
        ));
    }

    let revalidation_path = canonical_regular_path(
        &admission.expected_revalidation_path,
        "qwen80 uniform Q4 source revalidation receipt",
    )?;
    let revalidation_raw =
        read_regular_file(&revalidation_path, "qwen80 uniform Q4 source revalidation receipt")?;
    let revalidation = parse_json_no_duplicate_keys(
        &revalidation_raw,
        "qwen80 uniform Q4 source revalidation receipt",
    )?;
    let revalidation_seal = verify_sealed_document(
        &revalidation,
        "qwen80 uniform Q4 source revalidation receipt",
    )?;
    if revalidation_seal != admission.expected_revalidation_seal_sha256 {
        return Err(Error::Model(
            "qwen80 uniform Q4 revalidation seal does not match protected admission binding".into(),
        ));
    }
    let declared_revalidation = required_string(
        manifest_object,
        "source_revalidation_receipt_path",
        "qwen80 uniform Q4 manifest",
    )?;
    if Path::new(declared_revalidation) != revalidation_path.as_path()
        && canonical_regular_path(Path::new(declared_revalidation), "declared revalidation")
            .ok()
            .as_ref()
            != Some(&revalidation_path)
    {
        let declared_canon = absolute_path(declared_revalidation, "declared revalidation")?;
        if declared_canon != revalidation_path {
            return Err(Error::Model(
                "qwen80 uniform Q4 manifest revalidation path differs from protected handoff".into(),
            ));
        }
    }
    if required_sha256(
        manifest_object,
        "source_revalidation_receipt_seal_sha256",
        "qwen80 uniform Q4 manifest",
    )? != admission.expected_revalidation_seal_sha256
    {
        return Err(Error::Model(
            "qwen80 uniform Q4 manifest revalidation seal differs from protected handoff".into(),
        ));
    }

    let baseline_admission = CompleteBinaryAdmission {
        model: QwenCompleteBinaryModel::Qwen80CoderNext,
        expected_manifest_seal_sha256: admission.expected_manifest_seal_sha256.clone(),
        expected_source_audit_seal_sha256: admission.expected_source_audit_seal_sha256.clone(),
        expected_source_revision: admission.expected_source_revision.clone(),
    };
    let revalidation_parent = revalidation_path.parent().ok_or_else(|| {
        Error::Model("qwen80 uniform Q4 revalidation receipt has no parent".into())
    })?;
    let source = validate_source_chain(
        &revalidation,
        &revalidation_path,
        revalidation_parent,
        &baseline_admission,
        &manifest_audit_seal,
    )?;

    let representation =
        required_object(manifest_object, "representation", "qwen80 uniform Q4 manifest")?;
    require_exact_string(
        representation,
        "family",
        "uniform_q4_group64_fp16_scale",
        "qwen80 uniform Q4 representation",
    )?;
    if required_u64(representation, "group_size", "qwen80 uniform Q4 representation")?
        != u64::try_from(UNIFORM_Q4_GROUP_SIZE).unwrap()
        || !required_bool(
            representation,
            "physical_direct_layout",
            "qwen80 uniform Q4 representation",
        )?
    {
        return Err(Error::Model(
            "qwen80 uniform Q4 representation requires group_size=64 and physical_direct_layout"
                .into(),
        ));
    }

    let rows = required_array(manifest_object, "tensors", "qwen80 uniform Q4 manifest")?;
    if rows.len() != source.weight_map.len() {
        return Err(Error::Model(
            "qwen80 uniform Q4 manifest tensor count does not match the revalidated source index"
                .into(),
        ));
    }

    let mut tensors = BTreeMap::new();
    let mut verified_payloads = BTreeMap::new();
    let parallel = match std::env::var("HAWKING_ADMISSION_PARALLEL") {
        Ok(value) if matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF") => {
            false
        }
        _ => true,
    };
    if parallel {
        let (lanes, workers) = quality_payload_verification_lanes(rows);
        let mut handles = Vec::with_capacity(workers);
        for lane in lanes {
            let lane_rows: Vec<Value> = lane.iter().map(|&i| rows[i].clone()).collect();
            let root = root.to_path_buf();
            let source = source.clone();
            handles.push(thread::spawn(move || {
                let mut local_tensors = BTreeMap::new();
                let mut local_payloads = BTreeMap::new();
                for value in lane_rows {
                    let row = value.as_object().ok_or_else(|| {
                        Error::Model(
                            "qwen80 uniform Q4 manifest tensor entry must be an object".into(),
                        )
                    })?;
                    let (tensor, payload) = validate_q80_q4_tensor_row(row, &root, &source)?;
                    let name = tensor.tensor_name.clone();
                    if local_tensors.insert(name.clone(), tensor).is_some() {
                        return Err(Error::Model(
                            "qwen80 uniform Q4 manifest contains a duplicate tensor_name".into(),
                        ));
                    }
                    local_payloads.insert(name, payload);
                }
                Ok((local_tensors, local_payloads))
            }));
        }
        for handle in handles {
            let (local_tensors, local_payloads) = handle
                .join()
                .map_err(|_| Error::Model("qwen80 uniform Q4 admission worker panicked".into()))??;
            for (name, tensor) in local_tensors {
                let payload = local_payloads.get(&name).cloned().ok_or_else(|| {
                    Error::Model("qwen80 uniform Q4 admission lane missing payload".into())
                })?;
                if tensors.insert(name.clone(), tensor).is_some() {
                    return Err(Error::Model(
                        "qwen80 uniform Q4 manifest contains a duplicate tensor_name across lanes"
                            .into(),
                    ));
                }
                verified_payloads.insert(name, payload);
            }
        }
    } else {
        for value in rows {
            let row = value.as_object().ok_or_else(|| {
                Error::Model("qwen80 uniform Q4 manifest tensor entry must be an object".into())
            })?;
            let (tensor, payload) = validate_q80_q4_tensor_row(row, root, &source)?;
            let name = tensor.tensor_name.clone();
            if tensors.insert(name.clone(), tensor).is_some() {
                return Err(Error::Model(
                    "qwen80 uniform Q4 manifest contains a duplicate tensor_name".into(),
                ));
            }
            verified_payloads.insert(name, payload);
        }
    }

    if tensors.keys().ne(source.weight_map.keys()) {
        return Err(Error::Model(
            "qwen80 uniform Q4 manifest tensor set does not exactly match the revalidated source index"
                .into(),
        ));
    }
    if verified_payloads.len() != tensors.len() {
        return Err(Error::Model(
            "qwen80 uniform Q4 admission did not retain one verified immutable payload per tensor"
                .into(),
        ));
    }

    let (elements, payload_bytes) =
        validate_ledger(manifest_object, &tensors, manifest_raw.len())?;

    Ok(CompleteBinaryArtifact {
        model: QwenCompleteBinaryModel::Qwen80CoderNext,
        manifest_path,
        manifest_seal_sha256: manifest_seal,
        source_audit_path: source.source_audit_path,
        source_audit_seal_sha256: source.source_audit_seal_sha256,
        source_revision: source.source_revision,
        source_index_path: source.source_index_path,
        source_weight_elements: elements,
        tensor_payload_bytes: payload_bytes,
        tensors,
        verified_payloads,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

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

    fn f32_to_bf16_le(values: &[f32]) -> Vec<u8> {
        let mut raw = Vec::with_capacity(values.len() * 2);
        for value in values {
            let bits = (value.to_bits() >> 16) as u16;
            raw.extend_from_slice(&bits.to_le_bytes());
        }
        raw
    }

    fn deterministic_values(elements: usize, seed: u32) -> Vec<f32> {
        (0..elements)
            .map(|index| {
                let phase = (index as f32 + seed as f32) * 0.17;
                phase.sin() * (1.0 + (index % 7) as f32 * 0.05)
            })
            .collect()
    }

    fn write_two_tensor_safetensors(path: &Path) -> (Vec<f32>, Vec<f32>) {
        // Multiples of 64 so codec BPW is exactly 4.25.
        let a = deterministic_values(128, 3);
        let b = deterministic_values(64, 11);
        let a_raw = f32_to_bf16_le(&a);
        let b_raw = f32_to_bf16_le(&b);
        let header = json!({
            "model.layers.0.mlp.shared_expert.down_proj.weight": {
                "dtype": "BF16",
                "shape": [8, 16],
                "data_offsets": [0, a_raw.len()]
            },
            "model.layers.0.mlp.experts.0.gate_proj.weight": {
                "dtype": "BF16",
                "shape": [4, 16],
                "data_offsets": [a_raw.len(), a_raw.len() + b_raw.len()]
            }
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&a_raw).unwrap();
        file.write_all(&b_raw).unwrap();
        file.sync_all().unwrap();
        (a, b)
    }

    fn widen_bf16_bits(values: &[f32]) -> Vec<f32> {
        values
            .iter()
            .map(|value| f32::from_bits((value.to_bits() >> 16) << 16))
            .collect()
    }

    fn build_source_tree(temp: &TempDir) -> (PathBuf, PathBuf, String, String) {
        let root = temp.path().join("complete-gravity");
        let source_dir = temp.path().join("source");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&source_dir).unwrap();

        let shard_name = "model-00001-of-00001.safetensors";
        let shard_path = source_dir.join(shard_name);
        write_two_tensor_safetensors(&shard_path);
        let shard_sha = sha256_hex(&fs::read(&shard_path).unwrap());

        let index_path = source_dir.join("model.safetensors.index.json");
        let index = json!({
            "weight_map": {
                "model.layers.0.mlp.shared_expert.down_proj.weight": shard_name,
                "model.layers.0.mlp.experts.0.gate_proj.weight": shard_name,
            }
        });
        fs::write(&index_path, serde_json::to_vec(&index).unwrap()).unwrap();
        let weight_map = weight_map_from_index(&index, "test source index").unwrap();

        let revision = "unit-test-q80-uq4-revision".to_owned();
        let model = QwenCompleteBinaryModel::Qwen80CoderNext;
        let audit_path = temp.path().join("source-audit.json");
        let audit = seal_value(json!({
            "schema": model.source_audit_schema(),
            "status": model.source_audit_status(),
            "source": {
                "repository": model.source_repository(),
                "revision": revision,
            },
        }))
        .unwrap();
        write_pretty_json(&audit_path, &audit).unwrap();
        let audit_seal = audit
            .get("seal_sha256")
            .and_then(Value::as_str)
            .unwrap()
            .to_owned();

        let mut shard_hashes = BTreeMap::new();
        shard_hashes.insert(shard_name.to_owned(), shard_sha.clone());
        let receipt_path = root.join("QWEN80_CURRENT_SOURCE_SHARD_REVALIDATION.json");
        let receipt = seal_value(json!({
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
                    "expected_sha256": shard_sha,
                    "observed_sha256": shard_sha,
                    "expected_bytes": fs::metadata(&shard_path).unwrap().len(),
                    "file_identity": fixture_identity(&shard_path),
                },
            },
        }))
        .unwrap();
        write_pretty_json(&receipt_path, &receipt).unwrap();
        let receipt_seal = receipt
            .get("seal_sha256")
            .and_then(Value::as_str)
            .unwrap()
            .to_owned();
        (root, receipt_path, audit_seal, receipt_seal)
    }

    #[test]
    fn codec_layout_is_q30_group64_4_25_bpw() {
        assert_eq!(UNIFORM_Q4_GROUP_SIZE, 64);
        assert_eq!(UNIFORM_Q4_CODE_BYTES_PER_GROUP, 32);
        assert!((UNIFORM_Q4_NOMINAL_BPW - 4.25).abs() < 1e-12);
        assert_eq!(UNIFORM_Q4_MAGIC, *b"HQ30UQ4\0");
        let values = deterministic_values(128, 1);
        let (payload, quality) = pack_uniform_q4_group64(&values, &[8, 16]).unwrap();
        let header = parse_uniform_q4_header(&payload).unwrap();
        assert_eq!(header.group_size, 64);
        assert_eq!(header.groups, 2);
        assert_eq!(header.elements, 128);
        assert_eq!(
            header.payload_bytes,
            32 + 8 + 2 * 2 + 2 * UNIFORM_Q4_CODE_BYTES_PER_GROUP
        );
        assert!((quality.codec_bpw - 4.25).abs() < 1e-12);
        assert!((QWEN80_UNIFORM_Q4_CAMPAIGN_BPW - 4.256).abs() < 1e-12);
        let decoded = decode_uniform_q4_group64(&payload).unwrap();
        assert_eq!(decoded.len(), 128);
        assert!(quality.cosine > 0.99, "cosine={}", quality.cosine);
    }

    #[test]
    fn streaming_packer_admits_two_small_qwen80_tensors() {
        let temp = TempDir::new().unwrap();
        let (root, revalidation_path, audit_seal, _revalidation_seal) = build_source_tree(&temp);
        let report = pack_qwen80_uniform_q4(&Qwen80UniformQ4PackRequest {
            output_root: root,
            revalidation_path: revalidation_path.clone(),
            require_full_qwen80_catalog: false,
            workers: 1,
        })
        .expect("two-tensor fixture must stream-pack");
        assert_eq!(report.tensor_count, 2);
        assert_eq!(report.source_weight_elements, 192);
        assert_eq!(report.packed.len(), 2);
        for row in &report.packed {
            assert_eq!(row.quality.groups * 64, row.quality.elements.max(1).div_ceil(64) * 64);
            assert!(
                (row.quality.codec_bpw - 4.25).abs() < 1e-12,
                "{} codec_bpw={}",
                row.tensor_name,
                row.quality.codec_bpw
            );
            assert!(
                row.quality.cosine > 0.99,
                "{} cosine={}",
                row.tensor_name,
                row.quality.cosine
            );
            let payload = fs::read(&row.artifact_path).unwrap();
            let header = parse_uniform_q4_header(&payload).unwrap();
            assert_eq!(header.group_size, 64);
            assert_eq!(&payload[..8], &UNIFORM_Q4_MAGIC);
        }
        assert!(
            report.mean_component_cosine > 0.99,
            "mean cosine={}",
            report.mean_component_cosine
        );

        let admission = Qwen80UniformQ4Admission {
            expected_manifest_seal_sha256: report.manifest_seal_sha256.clone(),
            expected_source_audit_seal_sha256: audit_seal,
            expected_source_revision: "unit-test-q80-uq4-revision".into(),
            expected_revalidation_path: revalidation_path,
            expected_revalidation_seal_sha256: {
                let raw = fs::read(&report.manifest_path).unwrap();
                let manifest: Value = serde_json::from_slice(&raw).unwrap();
                manifest["source_revalidation_receipt_seal_sha256"]
                    .as_str()
                    .unwrap()
                    .to_owned()
            },
            expected_terminal_path: report.terminal_path.clone(),
            expected_terminal_seal_sha256: report.terminal_seal_sha256.clone(),
        };
        let artifact = admit_qwen80_uniform_q4(&report.manifest_path, &admission)
            .expect("packed two-tensor catalog must admit");
        assert_eq!(artifact.model, QwenCompleteBinaryModel::Qwen80CoderNext);
        assert_eq!(artifact.tensors.len(), 2);
        assert_eq!(artifact.verified_payload_count(), 2);
        assert_eq!(artifact.source_weight_elements, 192);
        assert!(artifact.has_complete_verified_payload_cache());

        let shard_path = artifact
            .source_index_path
            .parent()
            .unwrap()
            .join("model-00001-of-00001.safetensors");
        let (a_f32, b_f32) = {
            // Re-widen the same BF16 bits used as pack source.
            let a = widen_bf16_bits(&deterministic_values(128, 3));
            let b = widen_bf16_bits(&deterministic_values(64, 11));
            let _ = shard_path;
            (a, b)
        };
        let decoded_a = decode_uniform_q4_group64(
            artifact
                .verified_tensor_payload("model.layers.0.mlp.shared_expert.down_proj.weight")
                .unwrap()
                .as_ref(),
        )
        .unwrap();
        let decoded_b = decode_uniform_q4_group64(
            artifact
                .verified_tensor_payload("model.layers.0.mlp.experts.0.gate_proj.weight")
                .unwrap()
                .as_ref(),
        )
        .unwrap();
        let quality_a = pack_quality(&a_f32, &decoded_a, 128, 2, 0);
        let quality_b = pack_quality(&b_f32, &decoded_b, 64, 1, 0);
        assert!(quality_a.cosine > 0.99, "admitted A cosine={}", quality_a.cosine);
        assert!(quality_b.cosine > 0.99, "admitted B cosine={}", quality_b.cosine);
    }

    #[test]
    fn admission_rejects_wrong_manifest_seal() {
        let temp = TempDir::new().unwrap();
        let (root, revalidation_path, audit_seal, revalidation_seal) = build_source_tree(&temp);
        let report = pack_qwen80_uniform_q4(&Qwen80UniformQ4PackRequest {
            output_root: root,
            revalidation_path: revalidation_path.clone(),
            require_full_qwen80_catalog: false,
            workers: 1,
        })
        .unwrap();
        let mut admission = Qwen80UniformQ4Admission {
            expected_manifest_seal_sha256: report.manifest_seal_sha256.clone(),
            expected_source_audit_seal_sha256: audit_seal,
            expected_source_revision: "unit-test-q80-uq4-revision".into(),
            expected_revalidation_path: revalidation_path,
            expected_revalidation_seal_sha256: revalidation_seal,
            expected_terminal_path: report.terminal_path,
            expected_terminal_seal_sha256: report.terminal_seal_sha256,
        };
        admission.expected_manifest_seal_sha256 = "0".repeat(64);
        let error = admit_qwen80_uniform_q4(&report.manifest_path, &admission).unwrap_err();
        assert!(
            error.to_string().contains("protected admission binding"),
            "{error}"
        );
    }

    #[test]
    fn full_catalog_flag_refuses_partial_source_index() {
        let temp = TempDir::new().unwrap();
        let (root, revalidation_path, _, _) = build_source_tree(&temp);
        let error = pack_qwen80_uniform_q4(&Qwen80UniformQ4PackRequest {
            output_root: root,
            revalidation_path,
            require_full_qwen80_catalog: true,
            workers: 1,
        })
        .unwrap_err();
        assert!(
            error.to_string().contains("74391"),
            "{error}"
        );
    }
}

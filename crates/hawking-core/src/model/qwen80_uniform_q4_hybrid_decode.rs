//! Bind the sealed Qwen80 uniform-Q4 group-64 catalog to the existing hybrid
//! token-graph schedule and run a multi-token greedy decode.
//!
//! This is the velocity-track seam the composed hybrid graph does not itself
//! provide: the graph's dispatch sites consume complete-binary (group-128
//! sign/scale) payloads, while the admitted quality-candidate body is HQ30UQ4
//! (group-64).  The loop follows the same embed → 48-layer mixer/MoE →
//! terminal-head order as [`super::qwen80_hybrid_token_graph`], reads weights
//! only through the uniform-Q4 reader, and advances caller-owned DeltaNet +
//! GQA state across tokens.
//!
//! Weight-consuming GEMVs prefer the existing `qwen_uniform_q4_*` Metal
//! kernels when a device is available.  Activation math (residual RMSNorm,
//! DeltaNet recurrence, GQA, SwiGLU, top-10, greedy argmax) reuses the
//! source operators already used by the hybrid graph / BF16 layer-major
//! path.  Expert gather is a host fallback: the composed graph has no
//! 512-way device gather, so the live router ids are read back and the ten
//! bodies are streamed.  Every fallback is counted.  This is a VELOCITY
//! BASELINE, not BASE_TRUE_TPS.

use super::qwen80_complete_runtime::{
    qwen80_gqa_apply_sigmoid_gate, qwen80_gqa_causal_attention,
    qwen80_gqa_query_from_interleaved_q_projection, qwen80_gqa_source_norm_rope, qwen80_layer_kind,
    source_qwen80_ba_to_decay_beta, source_qwen80_causal_conv_step_dense,
    source_qwen80_gated_rms_norm, source_qwen80_l2_normalize, source_qwen80_recurrent_deltanet,
    source_qwen80_residual_rms_norm, source_qwen80_split_linear_qkvz, source_qwen80_topk_router,
    Qwen80CanonicalGqaLayout, Qwen80CanonicalLinearDeltaNetLayout, Qwen80LayerKind, QWEN80_EXPERTS,
    QWEN80_HIDDEN, QWEN80_LAYERS, QWEN80_MOE_INTERMEDIATE, QWEN80_TOKENIZER_VOCAB, QWEN80_VOCAB,
};
use super::qwen80_source_bf16_layer_major::{peak_rss_bytes, STREAMED_PEAK_RSS_HARD_CAP_BYTES};
use super::qwen_complete_binary::{
    decode_uniform_q4_group64, parse_uniform_q4_header, CompleteBinaryHeader,
    QWEN80_UNIFORM_Q4_SCHEMA, QWEN80_UNIFORM_Q4_TENSOR_EXT, UNIFORM_Q4_CODE_BYTES_PER_GROUP,
    UNIFORM_Q4_GROUP_SIZE,
};
#[cfg(test)]
use super::qwen_complete_binary::{
    pack_uniform_q4_group64, QWEN80_UNIFORM_Q4_EXPECTED_TENSOR_COUNT,
};
use crate::kernels::{add_inplace, silu_mul};
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

pub const QWEN80_UNIFORM_Q4_COMPLETE_PHYSICAL_BPW: f64 = 4.259241;
pub const QWEN80_UNIFORM_Q4_VELOCITY_NOT_BASE_TRUE_TPS: &str =
    "VELOCITY_BASELINE_NOT_BASE_TRUE_TPS";
pub const QWEN80_UNIFORM_Q4_DEFAULT_ARTIFACT_REL: &str =
    "workspace/campaign/records/ascension-sandbox/physical/qwen80/quality-candidates/uniform-q4-group64-v1";
pub const QWEN80_UNIFORM_Q4_MANIFEST_NAME: &str =
    "QWEN80_UNIFORM_Q4_GROUP64_V1_COMPLETE_BINARY_GRAVITY_CANDIDATE.json";
pub const QWEN80_UNIFORM_Q4_TERMINAL_NAME: &str =
    "QWEN80_UNIFORM_Q4_GROUP64_V1_COMPLETE_GRAVITY_TERMINAL_RECEIPT.json";
pub const QWEN80_UNIFORM_Q4_EXPECTED_MANIFEST_SEAL: &str =
    "d4a140ab353f2756f08365c59da4e7ae646f389b969e6aa87e2d7b8f053df55b";
pub const QWEN80_UNIFORM_Q4_EXPECTED_TERMINAL_SEAL: &str =
    "b84e2d53532fbed700a99390d7d32c9eff223963e80ee079ee45cde9edfa3b39";
pub const QWEN80_DEFAULT_TOKENIZER_REL: &str =
    "workspace/campaign/records/runs/qwen-80b/Qwen3-Coder-Next/tokenizer.json";
const QWEN80_DEFAULT_TOKENIZER_SHA256: &str =
    "19564a48c4f71a2a1b937cce34c737a1e662b171c5f5d7edf641a15cd896f07d";

fn q80q4_error(message: impl Into<String>) -> Error {
    Error::Model(format!(
        "qwen80 uniform-q4 hybrid decode: {}",
        message.into()
    ))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn require_rss_cap(label: &str) -> Result<()> {
    let peak = peak_rss_bytes();
    if peak > STREAMED_PEAK_RSS_HARD_CAP_BYTES {
        return Err(q80q4_error(format!(
            "{label}: peak RSS {peak} exceeds streamed 16 GiB cap {STREAMED_PEAK_RSS_HARD_CAP_BYTES}"
        )));
    }
    Ok(())
}

/// One catalogued Q4 tensor. Payload stays on disk until requested.
#[derive(Clone, Debug)]
pub struct Qwen80UniformQ4CatalogRow {
    pub tensor_name: String,
    pub shape: Vec<usize>,
    pub elements: u64,
    pub artifact_path: PathBuf,
    pub artifact_bytes: u64,
    pub artifact_sha256: String,
}

/// Streaming catalog: manifest index only. Never retains the 42 GiB body.
#[derive(Debug)]
pub struct Qwen80UniformQ4StreamingCatalog {
    pub root: PathBuf,
    pub manifest_path: PathBuf,
    pub manifest_seal_sha256: String,
    pub terminal_seal_sha256: Option<String>,
    pub complete_physical_bpw: f64,
    pub mean_component_cosine: Option<f64>,
    pub tensor_payload_bytes: u64,
    rows: HashMap<String, Qwen80UniformQ4CatalogRow>,
    verified_sha256: std::sync::Mutex<std::collections::HashSet<String>>,
}

impl Qwen80UniformQ4StreamingCatalog {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = fs::canonicalize(root.as_ref()).map_err(|error| {
            q80q4_error(format!(
                "cannot canonicalize artifact root {}: {error}",
                root.as_ref().display()
            ))
        })?;
        let manifest_path = root.join(QWEN80_UNIFORM_Q4_MANIFEST_NAME);
        Self::open_manifest(manifest_path)
    }

    pub fn open_manifest(manifest_path: impl AsRef<Path>) -> Result<Self> {
        let manifest_path = fs::canonicalize(manifest_path.as_ref()).map_err(|error| {
            q80q4_error(format!(
                "cannot canonicalize manifest {}: {error}",
                manifest_path.as_ref().display()
            ))
        })?;
        let root = manifest_path
            .parent()
            .ok_or_else(|| q80q4_error("manifest has no parent directory"))?
            .to_path_buf();
        let raw = fs::read(&manifest_path).map_err(|error| {
            q80q4_error(format!(
                "cannot read manifest {}: {error}",
                manifest_path.display()
            ))
        })?;
        let document: Value = serde_json::from_slice(&raw)
            .map_err(|error| q80q4_error(format!("manifest is not JSON: {error}")))?;
        let object = document
            .as_object()
            .ok_or_else(|| q80q4_error("manifest root must be an object"))?;
        let schema = object
            .get("schema")
            .and_then(Value::as_str)
            .ok_or_else(|| q80q4_error("manifest missing schema"))?;
        if schema != QWEN80_UNIFORM_Q4_SCHEMA {
            return Err(q80q4_error(format!(
                "manifest schema {schema:?} is not {QWEN80_UNIFORM_Q4_SCHEMA}"
            )));
        }
        let manifest_seal_sha256 = object
            .get("seal_sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| q80q4_error("manifest missing seal_sha256"))?
            .to_owned();
        let complete_physical_bpw = object
            .get("complete_physical_bpw_ledger")
            .and_then(|ledger| ledger.get("complete_physical_bpw"))
            .and_then(Value::as_f64)
            .ok_or_else(|| q80q4_error("manifest missing complete_physical_bpw"))?;
        let tensor_payload_bytes = object
            .get("complete_physical_bpw_ledger")
            .and_then(|ledger| ledger.get("tensor_payload_bytes"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let mean_component_cosine = object
            .get("quality_summary")
            .and_then(|quality| quality.get("mean_component_cosine"))
            .and_then(Value::as_f64);
        let tensors = object
            .get("tensors")
            .and_then(Value::as_array)
            .ok_or_else(|| q80q4_error("manifest missing tensors array"))?;
        let mut rows = HashMap::with_capacity(tensors.len());
        for entry in tensors {
            let row = parse_catalog_row(entry, &root)?;
            if rows.insert(row.tensor_name.clone(), row).is_some() {
                return Err(q80q4_error("manifest contains a duplicate tensor_name"));
            }
        }
        let terminal_seal_sha256 = read_optional_terminal_seal(&root);
        require_rss_cap("after catalog index")?;
        Ok(Self {
            root,
            manifest_path,
            manifest_seal_sha256,
            terminal_seal_sha256,
            complete_physical_bpw,
            mean_component_cosine,
            tensor_payload_bytes,
            rows,
            verified_sha256: std::sync::Mutex::new(std::collections::HashSet::new()),
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.rows.len()
    }

    pub fn require_row(&self, name: &str) -> Result<&Qwen80UniformQ4CatalogRow> {
        self.rows
            .get(name)
            .ok_or_else(|| q80q4_error(format!("missing tensor {name:?}")))
    }

    /// Read one payload. A missing or short file raises; the body is never
    /// silently zero-filled.
    pub fn read_payload(&self, name: &str) -> Result<Arc<[u8]>> {
        let row = self.require_row(name)?;
        let metadata = fs::metadata(&row.artifact_path).map_err(|error| {
            q80q4_error(format!(
                "missing tensor {name:?} at {}: {error}",
                row.artifact_path.display()
            ))
        })?;
        if metadata.len() != row.artifact_bytes {
            return Err(q80q4_error(format!(
                "tensor {name:?} is short or resized: on-disk {} bytes, catalog {}",
                metadata.len(),
                row.artifact_bytes
            )));
        }
        let payload = fs::read(&row.artifact_path).map_err(|error| {
            q80q4_error(format!(
                "cannot read tensor {name:?} {}: {error}",
                row.artifact_path.display()
            ))
        })?;
        if (payload.len() as u64) != row.artifact_bytes {
            return Err(q80q4_error(format!(
                "tensor {name:?} read {} bytes, catalog {}",
                payload.len(),
                row.artifact_bytes
            )));
        }
        if payload.len() < 32 {
            return Err(q80q4_error(format!(
                "tensor {name:?} payload is truncated ({} bytes)",
                payload.len()
            )));
        }
        let already_verified = self
            .verified_sha256
            .lock()
            .map(|set| set.contains(name))
            .unwrap_or(false);
        if !already_verified {
            let observed = sha256_hex(&payload);
            if observed != row.artifact_sha256 {
                return Err(q80q4_error(format!(
                    "tensor {name:?} sha256 {observed} != catalog {}",
                    row.artifact_sha256
                )));
            }
            if let Ok(mut set) = self.verified_sha256.lock() {
                set.insert(name.to_owned());
            }
        }
        let header = parse_uniform_q4_header(&payload)?;
        if header.shape != row.shape || header.group_size != UNIFORM_Q4_GROUP_SIZE {
            return Err(q80q4_error(format!(
                "tensor {name:?} header shape {:?}/group {} disagrees with catalog {:?}/64",
                header.shape, header.group_size, row.shape
            )));
        }
        Ok(Arc::from(payload))
    }

    pub fn load_packed(&self, name: &str) -> Result<Qwen80Q4PackedTensor> {
        let payload = self.read_payload(name)?;
        Qwen80Q4PackedTensor::from_payload(name.to_owned(), payload)
    }
}

fn read_optional_terminal_seal(root: &Path) -> Option<String> {
    let path = root.join(QWEN80_UNIFORM_Q4_TERMINAL_NAME);
    let raw = fs::read(path).ok()?;
    let document: Value = serde_json::from_slice(&raw).ok()?;
    document
        .get("seal_sha256")
        .and_then(Value::as_str)
        .map(str::to_owned)
}

fn parse_catalog_row(entry: &Value, root: &Path) -> Result<Qwen80UniformQ4CatalogRow> {
    let object = entry
        .as_object()
        .ok_or_else(|| q80q4_error("catalog tensor entry must be an object"))?;
    let tensor_name = object
        .get("tensor_name")
        .and_then(Value::as_str)
        .ok_or_else(|| q80q4_error("catalog row missing tensor_name"))?
        .to_owned();
    let shape = object
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| q80q4_error(format!("catalog row {tensor_name:?} missing shape")))?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|number| usize::try_from(number).ok())
                .ok_or_else(|| {
                    q80q4_error(format!(
                        "catalog row {tensor_name:?} has a non-unsigned shape dim"
                    ))
                })
        })
        .collect::<Result<Vec<_>>>()?;
    let elements = object
        .get("elements")
        .and_then(Value::as_u64)
        .ok_or_else(|| q80q4_error(format!("catalog row {tensor_name:?} missing elements")))?;
    let artifact_bytes = object
        .get("artifact_bytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            q80q4_error(format!(
                "catalog row {tensor_name:?} missing artifact_bytes"
            ))
        })?;
    let artifact_sha256 = object
        .get("artifact_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            q80q4_error(format!(
                "catalog row {tensor_name:?} missing artifact_sha256"
            ))
        })?
        .to_owned();
    let declared = object
        .get("artifact_path")
        .and_then(Value::as_str)
        .map(PathBuf::from);
    let hashed = root.join("tensors").join(format!(
        "{}.{QWEN80_UNIFORM_Q4_TENSOR_EXT}",
        sha256_hex(tensor_name.as_bytes())
    ));
    let artifact_path = if hashed.is_file() {
        hashed
    } else if let Some(declared) = declared.filter(|path| path.is_file()) {
        declared
    } else {
        return Err(q80q4_error(format!(
            "missing tensor {tensor_name:?}: neither {} nor the declared artifact_path exists",
            hashed.display()
        )));
    };
    Ok(Qwen80UniformQ4CatalogRow {
        tensor_name,
        shape,
        elements,
        artifact_path,
        artifact_bytes,
        artifact_sha256,
    })
}

/// One admitted HQ30UQ4 payload split into header / scales / codes.
#[derive(Clone, Debug)]
pub struct Qwen80Q4PackedTensor {
    pub name: String,
    pub header: CompleteBinaryHeader,
    payload: Arc<[u8]>,
}

impl Qwen80Q4PackedTensor {
    pub fn from_payload(name: String, payload: Arc<[u8]>) -> Result<Self> {
        let header = parse_uniform_q4_header(&payload)?;
        if header.group_size != UNIFORM_Q4_GROUP_SIZE {
            return Err(q80q4_error(format!(
                "tensor {name:?} group_size {} is not 64",
                header.group_size
            )));
        }
        if payload.len() < header.payload_bytes {
            return Err(q80q4_error(format!(
                "tensor {name:?} is short: {} < {}",
                payload.len(),
                header.payload_bytes
            )));
        }
        Ok(Self {
            name,
            header,
            payload,
        })
    }

    pub fn rows_cols(&self) -> Result<(usize, usize)> {
        match self.header.shape.as_slice() {
            [rows, cols] => Ok((*rows, *cols)),
            [elements] => Ok((*elements, 1)),
            other => Err(q80q4_error(format!(
                "tensor {:?} shape {other:?} is not a matrix or vector",
                self.name
            ))),
        }
    }

    pub fn decode_f32(&self) -> Result<Vec<f32>> {
        decode_uniform_q4_group64(&self.payload)
    }

    pub fn gather_row(&self, row: usize) -> Result<Vec<f32>> {
        let (rows, cols) = self.rows_cols()?;
        if row >= rows {
            return Err(q80q4_error(format!(
                "tensor {:?} row {row} is outside 0..{rows}",
                self.name
            )));
        }
        let mut values = vec![0.0f32; cols];
        self.decode_row_into(row, cols, &mut values)?;
        Ok(values)
    }

    fn decode_row_into(&self, row: usize, cols: usize, out: &mut [f32]) -> Result<()> {
        if out.len() != cols {
            return Err(q80q4_error(format!(
                "tensor {:?} row buffer {} != cols {cols}",
                self.name,
                out.len()
            )));
        }
        let groups_per_row = cols.div_ceil(UNIFORM_Q4_GROUP_SIZE);
        let scale_off = self.header.scale_offset;
        let code_off = self.header.sign_offset;
        let payload = &self.payload;
        for group in 0..groups_per_row {
            let col0 = group * UNIFORM_Q4_GROUP_SIZE;
            let glen = (cols - col0).min(UNIFORM_Q4_GROUP_SIZE);
            let group_index = row * groups_per_row + group;
            let scale_at = scale_off + group_index * 2;
            let scale_bytes = payload.get(scale_at..scale_at + 2).ok_or_else(|| {
                q80q4_error(format!(
                    "tensor {:?} scale group {group_index} truncated",
                    self.name
                ))
            })?;
            let scale =
                half::f16::from_bits(u16::from_le_bytes([scale_bytes[0], scale_bytes[1]])).to_f32();
            let code_at = code_off + group_index * UNIFORM_Q4_CODE_BYTES_PER_GROUP;
            for local in 0..glen {
                let packed = *payload.get(code_at + local / 2).ok_or_else(|| {
                    q80q4_error(format!(
                        "tensor {:?} code group {group_index} truncated",
                        self.name
                    ))
                })?;
                let nibble = if local & 1 == 0 {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                out[col0 + local] = (nibble as i32 - 8) as f32 * scale;
            }
        }
        Ok(())
    }

    pub fn matvec(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let (rows, cols) = self.rows_cols()?;
        if input.len() != cols {
            return Err(q80q4_error(format!(
                "tensor {:?} matvec input {} != cols {cols}",
                self.name,
                input.len()
            )));
        }
        if output.len() != rows {
            return Err(q80q4_error(format!(
                "tensor {:?} matvec output {} != rows {rows}",
                self.name,
                output.len()
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(q80q4_error(format!(
                "tensor {:?} matvec input is non-finite",
                self.name
            )));
        }
        let groups_per_row = cols.div_ceil(UNIFORM_Q4_GROUP_SIZE);
        let scale_off = self.header.scale_offset;
        let code_off = self.header.sign_offset;
        let payload = &self.payload;
        for row in 0..rows {
            let mut sum = 0.0f32;
            for group in 0..groups_per_row {
                let col0 = group * UNIFORM_Q4_GROUP_SIZE;
                let glen = (cols - col0).min(UNIFORM_Q4_GROUP_SIZE);
                let group_index = row * groups_per_row + group;
                let scale_at = scale_off + group_index * 2;
                let scale_bytes = payload.get(scale_at..scale_at + 2).ok_or_else(|| {
                    q80q4_error(format!(
                        "tensor {:?} scale group {group_index} truncated",
                        self.name
                    ))
                })?;
                let scale =
                    half::f16::from_bits(u16::from_le_bytes([scale_bytes[0], scale_bytes[1]]))
                        .to_f32();
                let code_at = code_off + group_index * UNIFORM_Q4_CODE_BYTES_PER_GROUP;
                let mut local = 0usize;
                while local + 1 < glen {
                    let packed = *payload.get(code_at + local / 2).ok_or_else(|| {
                        q80q4_error(format!(
                            "tensor {:?} code group {group_index} truncated",
                            self.name
                        ))
                    })?;
                    let q0 = (packed & 0x0f) as i32 - 8;
                    let q1 = (packed >> 4) as i32 - 8;
                    sum += q0 as f32 * scale * input[col0 + local];
                    sum += q1 as f32 * scale * input[col0 + local + 1];
                    local += 2;
                }
                if local < glen {
                    let packed = *payload.get(code_at + local / 2).ok_or_else(|| {
                        q80q4_error(format!(
                            "tensor {:?} code group {group_index} truncated",
                            self.name
                        ))
                    })?;
                    let q0 = (packed & 0x0f) as i32 - 8;
                    sum += q0 as f32 * scale * input[col0 + local];
                }
            }
            if !sum.is_finite() {
                return Err(q80q4_error(format!(
                    "tensor {:?} matvec produced a non-finite row {row}",
                    self.name
                )));
            }
            output[row] = sum;
        }
        Ok(())
    }
}

/// Caller-owned hybrid decode state. Resetting this between tokens is wrong.
#[derive(Clone, Debug)]
pub struct Qwen80HybridDecodeState {
    pub max_seq_len: usize,
    pub position: usize,
    pub linear_conv: Vec<Vec<f32>>,
    pub linear_recurrent: Vec<Vec<f32>>,
    pub gqa_key: Vec<Vec<f32>>,
    pub gqa_value: Vec<Vec<f32>>,
}

impl Qwen80HybridDecodeState {
    pub fn new(max_seq_len: usize) -> Result<Self> {
        if max_seq_len == 0 {
            return Err(q80q4_error("max_seq_len must be positive"));
        }
        let linear = Qwen80CanonicalLinearDeltaNetLayout::source_exact();
        linear.validate()?;
        let gqa = Qwen80CanonicalGqaLayout::source_exact();
        gqa.validate()?;
        let mut linear_conv = Vec::new();
        let mut linear_recurrent = Vec::new();
        let mut gqa_key = Vec::new();
        let mut gqa_value = Vec::new();
        for layer in 0..QWEN80_LAYERS {
            match qwen80_layer_kind(layer)? {
                Qwen80LayerKind::LinearAttention => {
                    linear_conv.push(vec![0.0; linear.conv_state_elements()?]);
                    linear_recurrent.push(vec![0.0; linear.recurrent_state_elements()?]);
                }
                Qwen80LayerKind::FullAttention => {
                    gqa_key.push(vec![0.0; max_seq_len * gqa.kv_dim]);
                    gqa_value.push(vec![0.0; max_seq_len * gqa.kv_dim]);
                }
            }
        }
        Ok(Self {
            max_seq_len,
            position: 0,
            linear_conv,
            linear_recurrent,
            gqa_key,
            gqa_value,
        })
    }

    pub fn reset(&mut self) {
        self.position = 0;
        for slot in &mut self.linear_conv {
            slot.fill(0.0);
        }
        for slot in &mut self.linear_recurrent {
            slot.fill(0.0);
        }
        for slot in &mut self.gqa_key {
            slot.fill(0.0);
        }
        for slot in &mut self.gqa_value {
            slot.fill(0.0);
        }
    }

    pub fn fingerprint_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update((self.position as u64).to_le_bytes());
        hasher.update((self.max_seq_len as u64).to_le_bytes());
        for slot in &self.linear_conv {
            for value in slot {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }
        for slot in &self.linear_recurrent {
            for value in slot {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }
        for slot in &self.gqa_key {
            for value in slot {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }
        for slot in &self.gqa_value {
            for value in slot {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }
        format!("{:x}", hasher.finalize())
    }

    fn linear_slot_for_layer(&self, layer: usize) -> Result<usize> {
        let mut slot = 0usize;
        for index in 0..=layer {
            if matches!(qwen80_layer_kind(index)?, Qwen80LayerKind::LinearAttention) {
                if index == layer {
                    return Ok(slot);
                }
                slot += 1;
            }
        }
        Err(q80q4_error(format!(
            "layer {layer} is not a DeltaNet layer"
        )))
    }

    fn gqa_slot_for_layer(&self, layer: usize) -> Result<usize> {
        let mut slot = 0usize;
        for index in 0..=layer {
            if matches!(qwen80_layer_kind(index)?, Qwen80LayerKind::FullAttention) {
                if index == layer {
                    return Ok(slot);
                }
                slot += 1;
            }
        }
        Err(q80q4_error(format!("layer {layer} is not a GQA layer")))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Qwen80DecodeFallbackCounts {
    pub host_q4_matvec: u64,
    pub host_q4_embedding_gather: u64,
    pub host_q4_vector_decode: u64,
    pub host_activation: u64,
    pub host_expert_payload_bind: u64,
    pub host_sample: u64,
}

impl Qwen80DecodeFallbackCounts {
    pub fn total(&self) -> u64 {
        self.host_q4_matvec
            .saturating_add(self.host_q4_embedding_gather)
            .saturating_add(self.host_q4_vector_decode)
            .saturating_add(self.host_activation)
            .saturating_add(self.host_expert_payload_bind)
            .saturating_add(self.host_sample)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Qwen80DecodeNativeCounts {
    pub q4_matvec_dispatches: u64,
    pub q4_embedding_dispatches: u64,
    pub q4_decode_vector_dispatches: u64,
}

/// Advance hybrid state with a cheap, deterministic mixer that still uses the
/// real DeltaNet recurrence and GQA KV append.  Used by the state-contract
/// tests so a silent reset between tokens fails without opening the 42 GiB
/// catalog.
pub fn qwen80_fixture_advance_hybrid_state(
    state: &mut Qwen80HybridDecodeState,
    token: u32,
    reset_before: bool,
) -> Result<String> {
    if reset_before {
        state.reset();
    }
    if state.position >= state.max_seq_len {
        return Err(q80q4_error("fixture position exceeds max_seq_len"));
    }
    let position = state.position;
    let linear = Qwen80CanonicalLinearDeltaNetLayout::source_exact();
    let gqa = Qwen80CanonicalGqaLayout::source_exact();
    let phase = (token as f32) * 0.017 + (position as f32) * 0.031;
    let mut query = vec![0.0f32; linear.value_heads * linear.key_head_dim];
    let mut key = vec![0.0f32; query.len()];
    let mut value = vec![0.0f32; linear.value_elements()?];
    for (index, slot) in query.iter_mut().enumerate() {
        *slot = ((index as f32 + 1.0) * phase).sin() * 0.05;
    }
    for (index, slot) in key.iter_mut().enumerate() {
        *slot = ((index as f32 + 3.0) * phase).cos() * 0.05;
    }
    for (index, slot) in value.iter_mut().enumerate() {
        *slot = ((index as f32 + 5.0) * phase).sin() * 0.04;
    }
    let decay = vec![0.92f32; linear.value_heads];
    let beta = vec![0.25f32; linear.value_heads];
    for slot in 0..state.linear_recurrent.len() {
        let _ = source_qwen80_recurrent_deltanet(
            &mut state.linear_recurrent[slot],
            &query,
            &key,
            &value,
            &decay,
            &beta,
            &linear,
        )?;
        let conv_len = state.linear_conv[slot].len();
        if conv_len % linear.conv_state_tokens != 0 {
            return Err(q80q4_error("fixture conv state geometry drifted"));
        }
        for channel in 0..(conv_len / linear.conv_state_tokens) {
            let base = channel * linear.conv_state_tokens;
            for tap in 0..linear.conv_state_tokens.saturating_sub(1) {
                state.linear_conv[slot][base + tap] = state.linear_conv[slot][base + tap + 1];
            }
            if linear.conv_state_tokens > 0 {
                state.linear_conv[slot][base + linear.conv_state_tokens - 1] =
                    phase.sin() * 0.01 + channel as f32 * 0.0001;
            }
        }
    }
    let mut key_row = vec![0.0f32; gqa.kv_dim];
    let mut value_row = vec![0.0f32; gqa.kv_dim];
    for (index, slot) in key_row.iter_mut().enumerate() {
        *slot = ((index as f32 + 7.0) * phase).sin() * 0.03;
    }
    for (index, slot) in value_row.iter_mut().enumerate() {
        *slot = ((index as f32 + 11.0) * phase).cos() * 0.03;
    }
    for slot in 0..state.gqa_key.len() {
        let start = position * gqa.kv_dim;
        let end = start + gqa.kv_dim;
        state.gqa_key[slot][start..end].copy_from_slice(&key_row);
        state.gqa_value[slot][start..end].copy_from_slice(&value_row);
    }
    state.position = state.position.saturating_add(1);
    Ok(state.fingerprint_sha256())
}

pub fn qwen80_fixture_greedy_token(state: &Qwen80HybridDecodeState, token: u32) -> u32 {
    let mut mix = token as u64;
    mix ^= (state.position as u64).wrapping_mul(0x9E37_79B9);
    if let Some(first) = state.linear_recurrent.first().and_then(|slot| slot.first()) {
        mix ^= u64::from(first.to_bits());
    }
    if let Some(first) = state.gqa_key.first().and_then(|slot| slot.first()) {
        mix ^= u64::from(first.to_bits()).rotate_left(13);
    }
    (mix % QWEN80_TOKENIZER_VOCAB as u64) as u32
}

struct PackedCache {
    tensors: HashMap<String, Qwen80Q4PackedTensor>,
    vectors: HashMap<String, Vec<f32>>,
}

impl PackedCache {
    fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            vectors: HashMap::new(),
        }
    }

    fn packed<'a>(
        &'a mut self,
        catalog: &Qwen80UniformQ4StreamingCatalog,
        name: &str,
    ) -> Result<&'a Qwen80Q4PackedTensor> {
        if !self.tensors.contains_key(name) {
            self.tensors
                .insert(name.to_owned(), catalog.load_packed(name)?);
        }
        Ok(self.tensors.get(name).expect("just inserted"))
    }

    fn vector(
        &mut self,
        catalog: &Qwen80UniformQ4StreamingCatalog,
        name: &str,
        fallbacks: &mut Qwen80DecodeFallbackCounts,
    ) -> Result<Vec<f32>> {
        if let Some(existing) = self.vectors.get(name) {
            return Ok(existing.clone());
        }
        let packed = catalog.load_packed(name)?;
        let values = packed.decode_f32()?;
        fallbacks.host_q4_vector_decode = fallbacks.host_q4_vector_decode.saturating_add(1);
        self.vectors.insert(name.to_owned(), values.clone());
        if !self.tensors.contains_key(name) {
            self.tensors.insert(name.to_owned(), packed);
        }
        Ok(values)
    }
}

#[cfg(target_os = "macos")]
struct MetalQ4Weight {
    codes: crate::metal::PinnedBuffer,
    scales: crate::metal::PinnedBuffer,
}

#[cfg(target_os = "macos")]
struct MetalQ4Accel {
    context: crate::metal::MetalContext,
    weights: HashMap<String, MetalQ4Weight>,
}

#[cfg(target_os = "macos")]
impl MetalQ4Accel {
    fn new() -> Result<Self> {
        Ok(Self {
            context: crate::metal::MetalContext::new()?,
            weights: HashMap::new(),
        })
    }

    fn upload_weight(&mut self, packed: &Qwen80Q4PackedTensor) -> Result<&MetalQ4Weight> {
        if !self.weights.contains_key(&packed.name) {
            let codes = packed
                .payload
                .get(packed.header.sign_offset..packed.header.payload_bytes)
                .ok_or_else(|| q80q4_error("metal q4 codes truncated"))?;
            let scales = packed
                .payload
                .get(packed.header.scale_offset..packed.header.sign_offset)
                .ok_or_else(|| q80q4_error("metal q4 scales truncated"))?;
            let uploaded = MetalQ4Weight {
                codes: self.context.new_buffer_with_bytes_checked(codes)?,
                scales: self.context.new_buffer_with_bytes_checked(scales)?,
            };
            self.weights.insert(packed.name.clone(), uploaded);
        }
        Ok(self.weights.get(&packed.name).expect("just inserted"))
    }

    fn evict(&mut self, name: &str) {
        self.weights.remove(name);
    }

    fn matvec(
        &mut self,
        packed: &Qwen80Q4PackedTensor,
        input: &[f32],
        output: &mut [f32],
        native: &mut Qwen80DecodeNativeCounts,
    ) -> Result<()> {
        use crate::metal::TokenCommandBuffer;
        let (rows, cols) = packed.rows_cols()?;
        if input.len() != cols || output.len() != rows {
            return Err(q80q4_error("metal q4 matvec geometry mismatch"));
        }
        self.upload_weight(packed)?;
        let codes_buf = self
            .weights
            .get(&packed.name)
            .expect("uploaded")
            .codes
            .clone();
        let scales_buf = self
            .weights
            .get(&packed.name)
            .expect("uploaded")
            .scales
            .clone();
        let input_bytes = bytemuck::cast_slice(input);
        let input_buf = self.context.new_buffer_with_bytes_checked(input_bytes)?;
        let output_buf = self
            .context
            .new_buffer_checked(rows * std::mem::size_of::<f32>())?;
        let mut tcb = TokenCommandBuffer::new(&self.context);
        crate::kernels::qwen_uniform_q4_group64_matvec_component_tcb(
            &mut tcb,
            &codes_buf,
            &scales_buf,
            &input_buf,
            &output_buf,
            rows,
            cols,
        )?;
        tcb.commit_and_wait()?;
        let observed =
            unsafe { std::slice::from_raw_parts(output_buf.contents() as *const f32, rows) };
        output.copy_from_slice(observed);
        native.q4_matvec_dispatches = native.q4_matvec_dispatches.saturating_add(1);
        Ok(())
    }

    fn embed_row(
        &mut self,
        packed: &Qwen80Q4PackedTensor,
        token: u32,
        output: &mut [f32],
        native: &mut Qwen80DecodeNativeCounts,
    ) -> Result<()> {
        use crate::metal::TokenCommandBuffer;
        if packed.header.shape != [QWEN80_VOCAB, QWEN80_HIDDEN] {
            return Err(q80q4_error("metal q4 embedding shape drifted"));
        }
        if output.len() != QWEN80_HIDDEN {
            return Err(q80q4_error("metal q4 embedding output width drifted"));
        }
        self.upload_weight(packed)?;
        let codes_buf = self
            .weights
            .get(&packed.name)
            .expect("uploaded")
            .codes
            .clone();
        let scales_buf = self
            .weights
            .get(&packed.name)
            .expect("uploaded")
            .scales
            .clone();
        let output_buf = self
            .context
            .new_buffer_checked(QWEN80_HIDDEN * std::mem::size_of::<f32>())?;
        let mut tcb = TokenCommandBuffer::new(&self.context);
        tcb.dispatch_threads(
            "qwen_uniform_q4_embedding_lookup",
            (QWEN80_HIDDEN as u32, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&codes_buf), 0);
                encoder.set_buffer(1, Some(&scales_buf), 0);
                encoder.set_buffer(2, Some(&output_buf), 0);
                encoder.set_bytes(3, 4, &token as *const u32 as *const _);
                let hidden = QWEN80_HIDDEN as u32;
                let vocab = QWEN80_VOCAB as u32;
                let group = UNIFORM_Q4_GROUP_SIZE as u32;
                encoder.set_bytes(4, 4, &hidden as *const u32 as *const _);
                encoder.set_bytes(5, 4, &vocab as *const u32 as *const _);
                encoder.set_bytes(6, 4, &group as *const u32 as *const _);
            },
        )?;
        tcb.commit_and_wait()?;
        let observed = unsafe {
            std::slice::from_raw_parts(output_buf.contents() as *const f32, QWEN80_HIDDEN)
        };
        output.copy_from_slice(observed);
        native.q4_embedding_dispatches = native.q4_embedding_dispatches.saturating_add(1);
        Ok(())
    }
}

/// Session that streams the q4 catalog through the hybrid token schedule.
pub struct Qwen80UniformQ4HybridDecodeSession {
    catalog: Qwen80UniformQ4StreamingCatalog,
    cache: PackedCache,
    pub state: Qwen80HybridDecodeState,
    pub fallbacks: Qwen80DecodeFallbackCounts,
    pub native: Qwen80DecodeNativeCounts,
    #[cfg(target_os = "macos")]
    metal: Option<MetalQ4Accel>,
    pub metal_error: Option<String>,
}

impl Qwen80UniformQ4HybridDecodeSession {
    pub fn new(catalog: Qwen80UniformQ4StreamingCatalog, max_seq_len: usize) -> Result<Self> {
        #[cfg(target_os = "macos")]
        let (metal, metal_error) = match MetalQ4Accel::new() {
            Ok(accel) => (Some(accel), None),
            Err(error) => (None, Some(error.to_string())),
        };
        #[cfg(not(target_os = "macos"))]
        let metal_error = Some("Metal q4 kernels require macOS".to_owned());
        Ok(Self {
            catalog,
            cache: PackedCache::new(),
            state: Qwen80HybridDecodeState::new(max_seq_len)?,
            fallbacks: Qwen80DecodeFallbackCounts::default(),
            native: Qwen80DecodeNativeCounts::default(),
            #[cfg(target_os = "macos")]
            metal,
            metal_error,
        })
    }

    pub fn catalog(&self) -> &Qwen80UniformQ4StreamingCatalog {
        &self.catalog
    }

    pub fn reset_state(&mut self) {
        self.state.reset();
    }

    fn matvec_named(&mut self, name: &str, input: &[f32], output: &mut [f32]) -> Result<()> {
        let packed = self.cache.packed(&self.catalog, name)?.clone();
        #[cfg(target_os = "macos")]
        if let Some(metal) = self.metal.as_mut() {
            match metal.matvec(&packed, input, output, &mut self.native) {
                Ok(()) => return Ok(()),
                Err(error) => {
                    if self.metal_error.is_none() {
                        self.metal_error = Some(error.to_string());
                    }
                }
            }
        }
        packed.matvec(input, output)?;
        self.fallbacks.host_q4_matvec = self.fallbacks.host_q4_matvec.saturating_add(1);
        Ok(())
    }

    fn embed(&mut self, token: u32) -> Result<Vec<f32>> {
        if token as usize >= QWEN80_VOCAB {
            return Err(q80q4_error(format!(
                "token {token} is outside the embedding vocab"
            )));
        }
        let packed = self
            .cache
            .packed(&self.catalog, "model.embed_tokens.weight")?
            .clone();
        let mut hidden = vec![0.0f32; QWEN80_HIDDEN];
        #[cfg(target_os = "macos")]
        if let Some(metal) = self.metal.as_mut() {
            if metal
                .embed_row(&packed, token, &mut hidden, &mut self.native)
                .is_ok()
            {
                return Ok(hidden);
            }
        }
        hidden = packed.gather_row(token as usize)?;
        self.fallbacks.host_q4_embedding_gather =
            self.fallbacks.host_q4_embedding_gather.saturating_add(1);
        Ok(hidden)
    }

    fn vector(&mut self, name: &str) -> Result<Vec<f32>> {
        self.cache.vector(&self.catalog, name, &mut self.fallbacks)
    }

    fn layer_name(layer: usize, suffix: &str) -> String {
        format!("model.layers.{layer}.{suffix}")
    }

    fn expert_name(layer: usize, expert: usize, proj: &str) -> String {
        format!("model.layers.{layer}.mlp.experts.{expert}.{proj}.weight")
    }

    fn mlp(
        &mut self,
        gate_name: &str,
        up_name: &str,
        down_name: &str,
        input: &[f32],
        intermediate: usize,
    ) -> Result<Vec<f32>> {
        let mut gate = vec![0.0f32; intermediate];
        let mut up = vec![0.0f32; intermediate];
        let mut act = vec![0.0f32; intermediate];
        let mut down = vec![0.0f32; QWEN80_HIDDEN];
        self.matvec_named(gate_name, input, &mut gate)?;
        self.matvec_named(up_name, input, &mut up)?;
        silu_mul(&gate, &up, &mut act);
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.matvec_named(down_name, &act, &mut down)?;
        Ok(down)
    }

    fn deltanet_mixer(&mut self, layer: usize, hidden: &[f32]) -> Result<Vec<f32>> {
        let layout = Qwen80CanonicalLinearDeltaNetLayout::source_exact();
        let slot = self.state.linear_slot_for_layer(layer)?;
        let input_w = self.vector(&Self::layer_name(layer, "input_layernorm.weight"))?;
        let rms = source_qwen80_residual_rms_norm(hidden, &input_w)?;
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        let qkvz_rows = layout.qkvz_projection_elements()?;
        let ba_rows = layout.ba_projection_elements()?;
        let mut projected_qkvz = vec![0.0f32; qkvz_rows];
        let mut projected_ba = vec![0.0f32; ba_rows];
        self.matvec_named(
            &Self::layer_name(layer, "linear_attn.in_proj_qkvz.weight"),
            &rms,
            &mut projected_qkvz,
        )?;
        self.matvec_named(
            &Self::layer_name(layer, "linear_attn.in_proj_ba.weight"),
            &rms,
            &mut projected_ba,
        )?;
        let (raw_query, raw_key, raw_value, z) =
            source_qwen80_split_linear_qkvz(&projected_qkvz, &layout)?;
        let mut mixed_qkv = Vec::with_capacity(layout.conv_channels);
        mixed_qkv.extend_from_slice(&raw_query);
        mixed_qkv.extend_from_slice(&raw_key);
        mixed_qkv.extend_from_slice(&raw_value);
        let conv_w = self.vector(&Self::layer_name(layer, "linear_attn.conv1d.weight"))?;
        let (convolved_qkv, next_conv) = source_qwen80_causal_conv_step_dense(
            &mixed_qkv,
            &self.state.linear_conv[slot],
            &conv_w,
            &layout,
        )?;
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        let raw_query_len = layout.key_elements()?;
        let raw_value_len = layout.value_elements()?;
        let convolved_query = &convolved_qkv[..raw_query_len];
        let convolved_key = &convolved_qkv[raw_query_len..raw_query_len + raw_query_len];
        let convolved_value = convolved_qkv[raw_query_len + raw_query_len..].to_vec();
        if convolved_value.len() != raw_value_len {
            return Err(q80q4_error("DeltaNet convolution value geometry drifted"));
        }
        let mut repeated_query = vec![0.0f32; raw_value_len];
        let mut repeated_key = vec![0.0f32; raw_value_len];
        for value_head in 0..layout.value_heads {
            let key_head = value_head / layout.value_heads_per_key_head;
            let mut query_head = convolved_query
                [key_head * layout.key_head_dim..(key_head + 1) * layout.key_head_dim]
                .to_vec();
            let mut key_head_values = convolved_key
                [key_head * layout.key_head_dim..(key_head + 1) * layout.key_head_dim]
                .to_vec();
            source_qwen80_l2_normalize(
                &mut query_head,
                (layout.key_head_dim as f32).sqrt().recip(),
            )?;
            source_qwen80_l2_normalize(&mut key_head_values, 1.0)?;
            let destination = value_head * layout.key_head_dim;
            repeated_query[destination..destination + layout.key_head_dim]
                .copy_from_slice(&query_head);
            repeated_key[destination..destination + layout.key_head_dim]
                .copy_from_slice(&key_head_values);
        }
        let a_log = self.vector(&Self::layer_name(layer, "linear_attn.A_log"))?;
        let dt_bias = self.vector(&Self::layer_name(layer, "linear_attn.dt_bias"))?;
        let (decay, beta) =
            source_qwen80_ba_to_decay_beta(&projected_ba, &a_log, &dt_bias, &layout)?;
        let recurrent_output = source_qwen80_recurrent_deltanet(
            &mut self.state.linear_recurrent[slot],
            &repeated_query,
            &repeated_key,
            &convolved_value,
            &decay,
            &beta,
            &layout,
        )?;
        self.state.linear_conv[slot] = next_conv;
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        let gated_norm = self.vector(&Self::layer_name(layer, "linear_attn.norm.weight"))?;
        let repeated_gated_norm = (0..layout.value_heads)
            .flat_map(|_| gated_norm.iter().copied())
            .collect::<Vec<_>>();
        let gated_output = source_qwen80_gated_rms_norm(
            &recurrent_output,
            &z,
            &repeated_gated_norm,
            layout.value_heads,
            layout.value_head_dim,
        )?;
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        let mut mixer_output = vec![0.0f32; QWEN80_HIDDEN];
        self.matvec_named(
            &Self::layer_name(layer, "linear_attn.out_proj.weight"),
            &gated_output,
            &mut mixer_output,
        )?;
        let mut residual = hidden.to_vec();
        add_inplace(&mut residual, &mixer_output);
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        if residual.iter().any(|value| !value.is_finite()) {
            return Err(q80q4_error(format!(
                "layer {layer} DeltaNet residual is non-finite"
            )));
        }
        Ok(residual)
    }

    fn gqa_mixer(&mut self, layer: usize, hidden: &[f32]) -> Result<Vec<f32>> {
        let layout = Qwen80CanonicalGqaLayout::source_exact();
        let slot = self.state.gqa_slot_for_layer(layer)?;
        let position = self.state.position;
        if position >= self.state.max_seq_len {
            return Err(q80q4_error(format!(
                "GQA position {position} exceeds max_seq_len {}",
                self.state.max_seq_len
            )));
        }
        let input_w = self.vector(&Self::layer_name(layer, "input_layernorm.weight"))?;
        let rms = source_qwen80_residual_rms_norm(hidden, &input_w)?;
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        let mut q_projection = vec![0.0f32; layout.q_proj_rows];
        let mut k_projection = vec![0.0f32; layout.kv_dim];
        let mut v_projection = vec![0.0f32; layout.kv_dim];
        self.matvec_named(
            &Self::layer_name(layer, "self_attn.q_proj.weight"),
            &rms,
            &mut q_projection,
        )?;
        self.matvec_named(
            &Self::layer_name(layer, "self_attn.k_proj.weight"),
            &rms,
            &mut k_projection,
        )?;
        self.matvec_named(
            &Self::layer_name(layer, "self_attn.v_proj.weight"),
            &rms,
            &mut v_projection,
        )?;
        let q_norm = self.vector(&Self::layer_name(layer, "self_attn.q_norm.weight"))?;
        let k_norm = self.vector(&Self::layer_name(layer, "self_attn.k_norm.weight"))?;
        let query_raw = qwen80_gqa_query_from_interleaved_q_projection(&q_projection, &layout)?;
        let query = qwen80_gqa_source_norm_rope(
            &query_raw,
            &q_norm,
            layout.query_heads,
            layout.head_dim,
            layout.rotary_dim,
            position,
            "GQA q_norm + partial RoPE",
        )?;
        let key_row = qwen80_gqa_source_norm_rope(
            &k_projection,
            &k_norm,
            layout.key_value_heads,
            layout.head_dim,
            layout.rotary_dim,
            position,
            "GQA k_norm + partial RoPE",
        )?;
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(2);
        let start = position * layout.kv_dim;
        let end = start + layout.kv_dim;
        self.state.gqa_key[slot][start..end].copy_from_slice(&key_row);
        self.state.gqa_value[slot][start..end].copy_from_slice(&v_projection);
        let attention = qwen80_gqa_causal_attention(
            &query,
            &self.state.gqa_key[slot],
            &self.state.gqa_value[slot],
            position + 1,
            &layout,
        )?;
        let gated = qwen80_gqa_apply_sigmoid_gate(&attention, &q_projection, &layout)?;
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(2);
        let mut mixer_output = vec![0.0f32; QWEN80_HIDDEN];
        self.matvec_named(
            &Self::layer_name(layer, "self_attn.o_proj.weight"),
            &gated,
            &mut mixer_output,
        )?;
        let mut residual = hidden.to_vec();
        add_inplace(&mut residual, &mixer_output);
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        if residual.iter().any(|value| !value.is_finite()) {
            return Err(q80q4_error(format!(
                "layer {layer} GQA residual is non-finite"
            )));
        }
        Ok(residual)
    }

    fn moe_suffix(&mut self, layer: usize, first_residual: &[f32]) -> Result<Vec<f32>> {
        let post_w = self.vector(&Self::layer_name(layer, "post_attention_layernorm.weight"))?;
        let router_input = source_qwen80_residual_rms_norm(first_residual, &post_w)?;
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        let shared = self.mlp(
            &Self::layer_name(layer, "mlp.shared_expert.gate_proj.weight"),
            &Self::layer_name(layer, "mlp.shared_expert.up_proj.weight"),
            &Self::layer_name(layer, "mlp.shared_expert.down_proj.weight"),
            &router_input,
            QWEN80_MOE_INTERMEDIATE,
        )?;
        let mut router_logits = vec![0.0f32; QWEN80_EXPERTS];
        self.matvec_named(
            &Self::layer_name(layer, "mlp.gate.weight"),
            &router_input,
            &mut router_logits,
        )?;
        let route = source_qwen80_topk_router(&router_logits)?;
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        let mut combined = vec![0.0f32; QWEN80_HIDDEN];
        for (&expert, &weight) in route.ids.iter().zip(route.weights.iter()) {
            let gate = Self::expert_name(layer, expert as usize, "gate_proj");
            let up = Self::expert_name(layer, expert as usize, "up_proj");
            let down = Self::expert_name(layer, expert as usize, "down_proj");
            // Touch the three payloads so a missing/short expert raises here.
            let _ = self.catalog.require_row(&gate)?;
            let _ = self.catalog.require_row(&up)?;
            let _ = self.catalog.require_row(&down)?;
            self.fallbacks.host_expert_payload_bind =
                self.fallbacks.host_expert_payload_bind.saturating_add(3);
            let expert_out = self.mlp(&gate, &up, &down, &router_input, QWEN80_MOE_INTERMEDIATE)?;
            for (dst, value) in combined.iter_mut().zip(expert_out) {
                *dst += value * weight;
            }
            self.cache.tensors.remove(&gate);
            self.cache.tensors.remove(&up);
            self.cache.tensors.remove(&down);
            #[cfg(target_os = "macos")]
            if let Some(metal) = self.metal.as_mut() {
                metal.evict(&gate);
                metal.evict(&up);
                metal.evict(&down);
            }
        }
        let mut gate_logit = [0.0f32; 1];
        self.matvec_named(
            &Self::layer_name(layer, "mlp.shared_expert_gate.weight"),
            &router_input,
            &mut gate_logit,
        )?;
        let gate_val = 1.0 / (1.0 + (-gate_logit[0]).exp());
        if !gate_val.is_finite() || !(0.0..=1.0).contains(&gate_val) {
            return Err(q80q4_error(format!(
                "layer {layer} shared-expert gate sigmoid is invalid"
            )));
        }
        for (dst, value) in combined.iter_mut().zip(shared) {
            *dst += value * gate_val;
        }
        let mut out = first_residual.to_vec();
        add_inplace(&mut out, &combined);
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        if out.iter().any(|value| !value.is_finite()) {
            return Err(q80q4_error(format!(
                "layer {layer} second residual is non-finite"
            )));
        }
        Ok(out)
    }

    fn terminal_greedy(&mut self, hidden: &[f32]) -> Result<u32> {
        let norm_w = self.vector("model.norm.weight")?;
        let normed = source_qwen80_residual_rms_norm(hidden, &norm_w)?;
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        let mut logits = vec![0.0f32; QWEN80_VOCAB];
        self.matvec_named("lm_head.weight", &normed, &mut logits)?;
        for logit in logits.iter_mut().skip(QWEN80_TOKENIZER_VOCAB) {
            *logit = f32::NEG_INFINITY;
        }
        // Lowest id on a tie, matching qwen80_terminal_head_greedy_sample_lowest_id.
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (index, &value) in logits.iter().take(QWEN80_TOKENIZER_VOCAB).enumerate() {
            if value > best_v || (value == best_v && index < best_i) {
                best_v = value;
                best_i = index;
            }
        }
        self.fallbacks.host_sample = self.fallbacks.host_sample.saturating_add(1);
        if !best_v.is_finite() {
            return Err(q80q4_error("greedy sample saw no finite logit"));
        }
        Ok(best_i as u32)
    }

    /// One hybrid-graph token: embed + 48 layers + terminal greedy.
    /// Advances DeltaNet + KV. Does not reset state.
    pub fn forward_token(&mut self, token: u32) -> Result<u32> {
        if self.state.position >= self.state.max_seq_len {
            return Err(q80q4_error(format!(
                "decode position {} exceeds max_seq_len {}",
                self.state.position, self.state.max_seq_len
            )));
        }
        let mut hidden = self.embed(token)?;
        for layer in 0..QWEN80_LAYERS {
            let first = match qwen80_layer_kind(layer)? {
                Qwen80LayerKind::LinearAttention => self.deltanet_mixer(layer, &hidden)?,
                Qwen80LayerKind::FullAttention => self.gqa_mixer(layer, &hidden)?,
            };
            hidden = self.moe_suffix(layer, &first)?;
        }
        let sampled = self.terminal_greedy(&hidden)?;
        self.state.position = self.state.position.saturating_add(1);
        require_rss_cap("after hybrid token")?;
        Ok(sampled)
    }
}

#[derive(Clone, Debug)]
pub struct Qwen80UniformQ4GreedyResult {
    pub prompt: String,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub generated_text: String,
    pub prefill_secs: f64,
    pub first_token_latency_secs: f64,
    pub decode_secs: f64,
    pub steady_state_decode_secs: f64,
    pub steady_state_tokens: usize,
    pub steady_state_tok_s: f64,
    pub peak_rss_bytes: u64,
    pub fallbacks: Qwen80DecodeFallbackCounts,
    pub native: Qwen80DecodeNativeCounts,
    pub complete_physical_bpw: f64,
    pub claim: &'static str,
    pub metal_q4_matvec_used: bool,
    pub metal_error: Option<String>,
}

pub fn render_qwen80_source_user_chat(user_text: &str) -> String {
    format!("<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n")
}

pub fn qwen80_default_artifact_root() -> PathBuf {
    PathBuf::from(QWEN80_UNIFORM_Q4_DEFAULT_ARTIFACT_REL)
}

pub fn qwen80_default_tokenizer_path() -> PathBuf {
    PathBuf::from(QWEN80_DEFAULT_TOKENIZER_REL)
}

pub fn load_qwen80_tokenizer(path: impl AsRef<Path>) -> Result<Tokenizer> {
    let path = path.as_ref();
    let raw = fs::read(path).map_err(|error| {
        q80q4_error(format!("cannot read tokenizer {}: {error}", path.display()))
    })?;
    let observed = sha256_hex(&raw);
    if observed != QWEN80_DEFAULT_TOKENIZER_SHA256 {
        return Err(q80q4_error(format!(
            "tokenizer sha256 {observed} != {QWEN80_DEFAULT_TOKENIZER_SHA256}"
        )));
    }
    let tokenizer = Tokenizer::from_file(path)?;
    if tokenizer.vocab_size() != QWEN80_TOKENIZER_VOCAB {
        return Err(q80q4_error(format!(
            "tokenizer vocab {} != {QWEN80_TOKENIZER_VOCAB}",
            tokenizer.vocab_size()
        )));
    }
    Ok(tokenizer)
}

pub fn generate_greedy(
    session: &mut Qwen80UniformQ4HybridDecodeSession,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
) -> Result<Qwen80UniformQ4GreedyResult> {
    if max_new_tokens == 0 {
        return Err(q80q4_error("max_new_tokens must be positive"));
    }
    let prompt_token_ids = tokenizer.encode(prompt, false)?;
    if prompt_token_ids.is_empty() {
        return Err(q80q4_error("prompt tokenization produced no tokens"));
    }
    if prompt_token_ids.len() + max_new_tokens > session.state.max_seq_len {
        return Err(q80q4_error(
            "prompt + max_new_tokens exceeds session max_seq_len",
        ));
    }
    session.reset_state();
    let prefill_started = Instant::now();
    let mut next = 0u32;
    for &token in &prompt_token_ids {
        next = session.forward_token(token)?;
    }
    let prefill_secs = prefill_started.elapsed().as_secs_f64();
    let first_token_latency_secs = prefill_secs;
    let mut generated = Vec::with_capacity(max_new_tokens);
    generated.push(next);
    let decode_started = Instant::now();
    let mut steady_started = None;
    for _ in 1..max_new_tokens {
        if tokenizer.is_eog(next) {
            break;
        }
        if steady_started.is_none() {
            steady_started = Some(Instant::now());
        }
        next = session.forward_token(next)?;
        generated.push(next);
    }
    let decode_secs = decode_started.elapsed().as_secs_f64();
    let steady_state_tokens = generated.len().saturating_sub(1);
    let steady_state_decode_secs = steady_started
        .map(|started| started.elapsed().as_secs_f64())
        .unwrap_or(0.0);
    let steady_state_tok_s = if steady_state_tokens == 0 || steady_state_decode_secs <= 0.0 {
        0.0
    } else {
        steady_state_tokens as f64 / steady_state_decode_secs
    };
    let generated_text = tokenizer.decode(&generated, true)?;
    #[cfg(target_os = "macos")]
    let metal_q4_matvec_used = session.metal.is_some() && session.native.q4_matvec_dispatches > 0;
    #[cfg(not(target_os = "macos"))]
    let metal_q4_matvec_used = false;
    Ok(Qwen80UniformQ4GreedyResult {
        prompt: prompt.to_owned(),
        prompt_token_ids,
        generated_token_ids: generated,
        generated_text,
        prefill_secs,
        first_token_latency_secs,
        decode_secs,
        steady_state_decode_secs,
        steady_state_tokens,
        steady_state_tok_s,
        peak_rss_bytes: peak_rss_bytes(),
        fallbacks: session.fallbacks.clone(),
        native: session.native.clone(),
        complete_physical_bpw: session.catalog.complete_physical_bpw,
        claim: QWEN80_UNIFORM_Q4_VELOCITY_NOT_BASE_TRUE_TPS,
        metal_q4_matvec_used,
        metal_error: session.metal_error.clone(),
    })
}

/// Resolve the contract-relative artifact root, then the main-repo copy.
pub fn discover_qwen80_uniform_q4_root() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from(QWEN80_UNIFORM_Q4_DEFAULT_ARTIFACT_REL),
        PathBuf::from(
            "/Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/physical/qwen80/quality-candidates/uniform-q4-group64-v1",
        ),
    ];
    candidates.into_iter().find(|path| {
        path.join(QWEN80_UNIFORM_Q4_MANIFEST_NAME).is_file() && path.join("tensors").is_dir()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    use tempfile::TempDir;

    fn write_q4_file(path: &Path, values: &[f32], shape: &[usize]) -> (u64, String) {
        let (payload, _) = pack_uniform_q4_group64(values, shape).unwrap();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, &payload).unwrap();
        (payload.len() as u64, sha256_hex(&payload))
    }

    fn fixture_catalog(temp: &TempDir) -> (PathBuf, String) {
        let root = temp.path().join("q4");
        let tensors = root.join("tensors");
        fs::create_dir_all(&tensors).unwrap();
        let values: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let name = "model.embed_tokens.weight";
        let hashed = format!(
            "{}.{QWEN80_UNIFORM_Q4_TENSOR_EXT}",
            sha256_hex(name.as_bytes())
        );
        let path = tensors.join(&hashed);
        let (bytes, sha) = write_q4_file(&path, &values, &[1, 64]);
        let manifest = json!({
            "schema": QWEN80_UNIFORM_Q4_SCHEMA,
            "status": "CANDIDATE_UNIFORM_Q4_GROUP64_DIAGNOSTIC_UNQUALIFIED",
            "seal_sha256": "aa".repeat(32),
            "complete_physical_bpw_ledger": {
                "complete_physical_bpw": QWEN80_UNIFORM_Q4_COMPLETE_PHYSICAL_BPW,
                "tensor_payload_bytes": bytes,
            },
            "quality_summary": { "mean_component_cosine": 0.994153 },
            "representation": {
                "family": "uniform_q4_group64_fp16_scale",
                "group_size": 64,
                "physical_direct_layout": true
            },
            "tensors": [{
                "tensor_name": name,
                "shape": [1, 64],
                "elements": 64,
                "artifact_path": path,
                "artifact_bytes": bytes,
                "artifact_sha256": sha,
            }]
        });
        let manifest_path = root.join(QWEN80_UNIFORM_Q4_MANIFEST_NAME);
        let mut file = fs::File::create(&manifest_path).unwrap();
        file.write_all(serde_json::to_vec(&manifest).unwrap().as_slice())
            .unwrap();
        (manifest_path, name.to_owned())
    }

    #[test]
    fn artifact_binding_missing_and_short_raise() {
        let temp = TempDir::new().unwrap();
        let (manifest, name) = fixture_catalog(&temp);
        let catalog = Qwen80UniformQ4StreamingCatalog::open_manifest(&manifest).unwrap();
        assert_eq!(catalog.tensor_count(), 1);
        assert!(catalog.require_row("no.such.tensor").is_err());
        let err = catalog.read_payload("no.such.tensor").unwrap_err();
        let message = format!("{err}");
        assert!(
            message.contains("missing tensor"),
            "missing tensor must raise, got {message}"
        );

        let row_path = catalog.require_row(&name).unwrap().artifact_path.clone();
        let original = fs::read(&row_path).unwrap();
        fs::write(&row_path, &original[..original.len() / 2]).unwrap();
        let short = catalog.read_payload(&name).unwrap_err();
        let short_msg = format!("{short}");
        assert!(
            short_msg.contains("short") || short_msg.contains("truncated"),
            "short tensor must raise, got {short_msg}"
        );
        // Restore so Drop is quiet.
        fs::write(&row_path, original).unwrap();
    }

    #[test]
    fn artifact_binding_real_manifest_is_74391() {
        let Some(root) = discover_qwen80_uniform_q4_root() else {
            // Fixture-only hosts still exercise the raise path above.
            return;
        };
        let catalog = Qwen80UniformQ4StreamingCatalog::open(&root).unwrap();
        assert_eq!(
            catalog.tensor_count(),
            QWEN80_UNIFORM_Q4_EXPECTED_TENSOR_COUNT
        );
        assert!(
            (catalog.complete_physical_bpw - QWEN80_UNIFORM_Q4_COMPLETE_PHYSICAL_BPW).abs() < 1e-6
        );
        assert!(catalog.require_row("lm_head.weight").is_ok());
        assert!(catalog.require_row("definitely.missing.tensor").is_err());
    }

    #[test]
    fn multi_token_state_advance_differs_from_reset_and_repeated_same_token() {
        let mut sequential = Qwen80HybridDecodeState::new(8).unwrap();
        let mut fingerprints = Vec::new();
        for _ in 0..3 {
            fingerprints
                .push(qwen80_fixture_advance_hybrid_state(&mut sequential, 7, false).unwrap());
        }
        assert_ne!(
            fingerprints[0], fingerprints[1],
            "state must change across tokens"
        );
        assert_ne!(fingerprints[1], fingerprints[2]);

        let mut reset_each = Qwen80HybridDecodeState::new(8).unwrap();
        let mut reset_prints = Vec::new();
        for _ in 0..3 {
            reset_prints
                .push(qwen80_fixture_advance_hybrid_state(&mut reset_each, 7, true).unwrap());
        }
        assert_eq!(
            reset_prints[0], reset_prints[1],
            "a reset between identical tokens must produce identical per-token state"
        );
        assert_ne!(
            fingerprints, reset_prints,
            "a state reset between tokens must fail this comparison"
        );

        let mut once = Qwen80HybridDecodeState::new(8).unwrap();
        let a = qwen80_fixture_advance_hybrid_state(&mut once, 3, false).unwrap();
        let b = qwen80_fixture_advance_hybrid_state(&mut once, 9, false).unwrap();
        let mut same = Qwen80HybridDecodeState::new(8).unwrap();
        let c = qwen80_fixture_advance_hybrid_state(&mut same, 3, false).unwrap();
        let d = qwen80_fixture_advance_hybrid_state(&mut same, 3, false).unwrap();
        assert_eq!(a, c);
        assert_ne!(
            b, d,
            "decoding N distinct tokens must differ from the same token N times"
        );
    }

    #[test]
    fn fixture_greedy_is_deterministic() {
        let mut left = Qwen80HybridDecodeState::new(8).unwrap();
        let mut right = Qwen80HybridDecodeState::new(8).unwrap();
        let mut left_ids = Vec::new();
        let mut right_ids = Vec::new();
        let mut token = 11u32;
        for _ in 0..4 {
            qwen80_fixture_advance_hybrid_state(&mut left, token, false).unwrap();
            left_ids.push(qwen80_fixture_greedy_token(&left, token));
            token = left_ids[left_ids.len() - 1];
        }
        token = 11;
        for _ in 0..4 {
            qwen80_fixture_advance_hybrid_state(&mut right, token, false).unwrap();
            right_ids.push(qwen80_fixture_greedy_token(&right, token));
            token = right_ids[right_ids.len() - 1];
        }
        assert_eq!(left_ids, right_ids);
        assert_eq!(left.fingerprint_sha256(), right.fingerprint_sha256());
    }
}

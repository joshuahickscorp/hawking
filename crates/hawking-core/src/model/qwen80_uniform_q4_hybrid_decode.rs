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

use super::qwen80_token_ns_ledger::Qwen80TokenNsSession;
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
use memmap2::{Advice, Mmap, MmapOptions};
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
const QWEN80_DEVICE_ACTIVATION_EPS: f32 = 1.0e-6;
const QWEN80_DEVICE_ROPE_THETA: f32 = 5_000_000.0;

/// Default **on**. Set `HAWKING_QWEN80_DEVICE_ACTIVATIONS=0` / `false` /
/// `off` / `no` to restore the host activation path for A/B.
pub fn qwen80_device_activations_enabled() -> bool {
    for key in [
        "HAWKING_QWEN80_DEVICE_ACTIVATIONS",
        "HAWKING_QWEN80_DEVICE_ACT",
    ] {
        if let Ok(raw) = std::env::var(key) {
            let trimmed = raw.trim();
            if trimmed.eq_ignore_ascii_case("0")
                || trimmed.eq_ignore_ascii_case("false")
                || trimmed.eq_ignore_ascii_case("off")
                || trimmed.eq_ignore_ascii_case("no")
            {
                return false;
            }
            if !trimmed.is_empty() {
                return true;
            }
        }
    }
    true
}

/// Default **off**. Measured 2026-08-16 (DIRTY_ENGINEERING):
/// `split` (3 CBs, no-wait suffix/mixer) is bit-identical but not a
/// robust tok/s win (extra submits tax prefill). `fuse` (suffix + next
/// mixer + moe_prefix in one CB) was a regression vs the 2-wait
/// topology on the same paired reps.
///
/// `HAWKING_QWEN80_OVERLAP_CBS=split` or `=fuse` to opt in.
pub fn qwen80_overlap_cbs_enabled() -> bool {
    match std::env::var("HAWKING_QWEN80_OVERLAP_CBS") {
        Ok(raw) => {
            let trimmed = raw.trim();
            !(trimmed.is_empty()
                || trimmed.eq_ignore_ascii_case("0")
                || trimmed.eq_ignore_ascii_case("false")
                || trimmed.eq_ignore_ascii_case("off")
                || trimmed.eq_ignore_ascii_case("no"))
        }
        Err(_) => false,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OverlapMode {
    Off,
    Split,
    Fuse,
}

fn qwen80_overlap_mode() -> OverlapMode {
    match std::env::var("HAWKING_QWEN80_OVERLAP_CBS") {
        Ok(raw) => {
            let trimmed = raw.trim().to_ascii_lowercase();
            if trimmed == "0"
                || trimmed == "false"
                || trimmed == "off"
                || trimmed == "no"
            {
                OverlapMode::Off
            } else if trimmed == "split" {
                OverlapMode::Split
            } else {
                OverlapMode::Fuse
            }
        }
        Err(_) => OverlapMode::Off,
    }
}

/// Default **off**. One serial compute encoder per mixer / prefix / suffix
/// CB. Paired 2026-08-16 DIRTY_ENGINEERING reps did not show a robust
/// complete-token win vs per-dispatch encoders. Opt in with
/// `HAWKING_QWEN80_SERIAL_MIXER=1`.
pub fn qwen80_serial_mixer_enabled() -> bool {
    match std::env::var("HAWKING_QWEN80_SERIAL_MIXER") {
        Ok(raw) => {
            let trimmed = raw.trim();
            !(trimmed.is_empty()
                || trimmed.eq_ignore_ascii_case("0")
                || trimmed.eq_ignore_ascii_case("false")
                || trimmed.eq_ignore_ascii_case("off")
                || trimmed.eq_ignore_ascii_case("no"))
        }
        Err(_) => false,
    }
}

/// Component (DeltaNet / GQA / shared / router / lm_head) matvec kernel.
/// Occupancy default: vectorized group decode at 64 threads/row.
/// Set `HAWKING_Q80_COMPONENT_Q4_OCCUPANCY=0` to restore the serial
/// one-thread-per-row kernel. Expert-table matvecs stay on their own
/// simdgroup/vecgroup selector.
pub const QWEN80_COMPONENT_MATVEC_KERNEL: &str = "qwen_uniform_q4_group64_matvec_vecgroup_x64";
pub const QWEN80_COMPONENT_MATVEC_ROWS_PER_TG: u32 = 4;

/// Default **on**. Set `HAWKING_Q80_EXPERT_NOCOPY=0` to restore
/// `new_buffer_with_bytes` packing for A/B.
pub fn qwen80_expert_nocopy_enabled() -> bool {
    crate::env_opt_out("HAWKING_Q80_EXPERT_NOCOPY")
        && crate::env_opt_out("HAWKING_QWEN80_EXPERT_NOCOPY")
}

/// Default **off**. Set `HAWKING_Q80_REHASH_PAYLOADS=1` to force the
/// per-payload SHA-256 on the token path even after session admission.
pub fn qwen80_payload_rehash_enabled() -> bool {
    match std::env::var("HAWKING_Q80_REHASH_PAYLOADS") {
        Ok(raw) => {
            let trimmed = raw.trim();
            !(trimmed.is_empty()
                || trimmed.eq_ignore_ascii_case("0")
                || trimmed.eq_ignore_ascii_case("false")
                || trimmed.eq_ignore_ascii_case("off")
                || trimmed.eq_ignore_ascii_case("no"))
        }
        Err(_) => false,
    }
}

/// Default **on**. Set `HAWKING_Q80_EXPERT_PREFETCH=0` to disable
/// previous-token next-layer expert prefault during suffix GPU.
pub fn qwen80_expert_prefetch_enabled() -> bool {
    crate::env_opt_out("HAWKING_Q80_EXPERT_PREFETCH")
}

/// Host-side first-touch split. `catalog_read_ns` is `fs::read` or mmap+advise
/// (page-in). `sha256_ns` is payload hashing. `metal_copy_ns` is
/// `new_buffer_with_bytes`. None of these is GPU time.
#[derive(Clone, Debug, Default)]
pub struct Qwen80FirstTouchSplit {
    pub catalog_read_ns: u64,
    pub sha256_ns: u64,
    pub metal_copy_ns: u64,
    pub metal_nocopy_ns: u64,
    pub header_parse_ns: u64,
    pub mmap_ns: u64,
    pub payloads_read: u64,
    pub payloads_hashed: u64,
    pub payloads_mmapped: u64,
    pub nocopy_binds: u64,
    pub copy_binds: u64,
    pub prefetch_uploads: u64,
    pub prefetch_ns: u64,
    pub prefetch_hits: u64,
    pub prefetch_misses: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Qwen80SessionAdmitReport {
    pub session_seal_sha256: String,
    pub tensors_stated: u64,
    pub sample_rehashed: u64,
    pub admit_ns: u64,
    pub production_seals_checked: bool,
}

/// Page-aligned mmap of one catalogued tensor. `file_bytes` is the sealed
/// payload; `mmap` may be rounded up to the Metal no-copy page size so the
/// window is 16 KiB aligned in both pointer and length.
pub struct Qwen80MappedTensor {
    pub name: String,
    pub header: CompleteBinaryHeader,
    pub file_bytes: usize,
    mmap: Arc<Mmap>,
}

impl Qwen80MappedTensor {
    pub fn mmap_arc(&self) -> Arc<Mmap> {
        Arc::clone(&self.mmap)
    }

    pub fn window(&self) -> &[u8] {
        &self.mmap[..]
    }

    pub fn payload(&self) -> &[u8] {
        &self.mmap[..self.file_bytes]
    }

    pub fn codes(&self) -> Result<&[u8]> {
        self.payload()
            .get(self.header.sign_offset..self.header.payload_bytes)
            .ok_or_else(|| q80q4_error(format!("tensor {:?} codes truncated", self.name)))
    }

    pub fn scales(&self) -> Result<&[u8]> {
        self.payload()
            .get(self.header.scale_offset..self.header.sign_offset)
            .ok_or_else(|| q80q4_error(format!("tensor {:?} scales truncated", self.name)))
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
}

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
    session_admitted: bool,
    pub session_seal_sha256: Option<String>,
    pub session_admit: Qwen80SessionAdmitReport,
    first_touch: std::sync::Mutex<Qwen80FirstTouchSplit>,
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
            session_admitted: false,
            session_seal_sha256: None,
            session_admit: Qwen80SessionAdmitReport::default(),
            first_touch: std::sync::Mutex::new(Qwen80FirstTouchSplit::default()),
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

    pub fn session_trusts_payloads(&self) -> bool {
        self.session_admitted && !qwen80_payload_rehash_enabled()
    }

    pub fn first_touch_split(&self) -> Qwen80FirstTouchSplit {
        self.first_touch
            .lock()
            .map(|split| split.clone())
            .unwrap_or_default()
    }

    pub(crate) fn add_first_touch(&self, update: impl FnOnce(&mut Qwen80FirstTouchSplit)) {
        if let Ok(mut split) = self.first_touch.lock() {
            update(&mut split);
        }
    }

    /// Cold artifact + session proof. Token-path SHA is skipped after this
    /// returns. Integrity is not deleted: the check moves here.
    ///
    /// Verified once:
    /// - every catalogued file exists as a regular non-symlink with exact
    ///   `artifact_bytes`
    /// - production catalog (74_391 tensors) matches the sealed manifest
    ///   and terminal constants
    /// - session seal binds `manifest_seal || terminal_seal || ordered
    ///   (name, bytes, declared_sha256)`
    /// - a small sample of payloads is rehashed against the catalog SHA
    ///
    /// Still guards every first-touch:
    /// - mmap / read size must equal `artifact_bytes`
    /// - header geometry must match the catalog row
    /// - no-copy bind is fail-closed (16 KiB window + pointer alias)
    /// - `read_payload_rehash` and `HAWKING_Q80_REHASH_PAYLOADS=1` remain
    pub fn admit_session(&mut self) -> Result<Qwen80SessionAdmitReport> {
        let started = Instant::now();
        let mut hasher = Sha256::new();
        hasher.update(self.manifest_seal_sha256.as_bytes());
        hasher.update(b"|");
        if let Some(terminal) = self.terminal_seal_sha256.as_deref() {
            hasher.update(terminal.as_bytes());
        }
        hasher.update(b"|");
        let mut names: Vec<&str> = self.rows.keys().map(String::as_str).collect();
        names.sort_unstable();
        let mut stated = 0u64;
        for name in &names {
            let row = self.require_row(name)?;
            let metadata = fs::symlink_metadata(&row.artifact_path).map_err(|error| {
                q80q4_error(format!(
                    "admit: cannot stat {name:?} at {}: {error}",
                    row.artifact_path.display()
                ))
            })?;
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(q80q4_error(format!(
                    "admit: {name:?} at {} is not a regular non-symlink file",
                    row.artifact_path.display()
                )));
            }
            if metadata.len() != row.artifact_bytes {
                return Err(q80q4_error(format!(
                    "admit: {name:?} is short or resized: on-disk {} bytes, catalog {}",
                    metadata.len(),
                    row.artifact_bytes
                )));
            }
            hasher.update(name.as_bytes());
            hasher.update(row.artifact_bytes.to_le_bytes());
            hasher.update(row.artifact_sha256.as_bytes());
            stated = stated.saturating_add(1);
        }
        let production = self.rows.len() == 74_391;
        if production {
            if self.manifest_seal_sha256 != QWEN80_UNIFORM_Q4_EXPECTED_MANIFEST_SEAL {
                return Err(q80q4_error(format!(
                    "admit: production manifest seal {} != {QWEN80_UNIFORM_Q4_EXPECTED_MANIFEST_SEAL}",
                    self.manifest_seal_sha256
                )));
            }
            if self.terminal_seal_sha256.as_deref() != Some(QWEN80_UNIFORM_Q4_EXPECTED_TERMINAL_SEAL)
            {
                return Err(q80q4_error(format!(
                    "admit: production terminal seal {:?} != {QWEN80_UNIFORM_Q4_EXPECTED_TERMINAL_SEAL}",
                    self.terminal_seal_sha256
                )));
            }
        }
        let session_seal = format!("{:x}", hasher.finalize());
        let mut sample_rehashed = 0u64;
        for sample in [
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.47.mlp.experts.511.down_proj.weight",
        ] {
            if self.rows.contains_key(sample) {
                let _ = self.hash_payload(sample)?;
                sample_rehashed = sample_rehashed.saturating_add(1);
            }
        }
        let report = Qwen80SessionAdmitReport {
            session_seal_sha256: session_seal.clone(),
            tensors_stated: stated,
            sample_rehashed,
            admit_ns: started.elapsed().as_nanos() as u64,
            production_seals_checked: production,
        };
        self.session_admitted = true;
        self.session_seal_sha256 = Some(session_seal);
        self.session_admit = report.clone();
        Ok(report)
    }

    fn hash_payload(&self, name: &str) -> Result<String> {
        let row = self.require_row(name)?;
        let read_started = Instant::now();
        let payload = fs::read(&row.artifact_path).map_err(|error| {
            q80q4_error(format!(
                "cannot read tensor {name:?} {}: {error}",
                row.artifact_path.display()
            ))
        })?;
        let read_ns = read_started.elapsed().as_nanos() as u64;
        let hash_started = Instant::now();
        let observed = sha256_hex(&payload);
        let hash_ns = hash_started.elapsed().as_nanos() as u64;
        self.add_first_touch(|split| {
            split.catalog_read_ns = split.catalog_read_ns.saturating_add(read_ns);
            split.sha256_ns = split.sha256_ns.saturating_add(hash_ns);
            split.payloads_read = split.payloads_read.saturating_add(1);
            split.payloads_hashed = split.payloads_hashed.saturating_add(1);
        });
        if observed != row.artifact_sha256 {
            return Err(q80q4_error(format!(
                "tensor {name:?} sha256 {observed} != catalog {}",
                row.artifact_sha256
            )));
        }
        if let Ok(mut set) = self.verified_sha256.lock() {
            set.insert(name.to_owned());
        }
        Ok(observed)
    }

    fn verify_payload_sha_if_needed(&self, name: &str, payload: &[u8]) -> Result<()> {
        if self.session_trusts_payloads() {
            return Ok(());
        }
        let already_verified = self
            .verified_sha256
            .lock()
            .map(|set| set.contains(name))
            .unwrap_or(false);
        if already_verified {
            return Ok(());
        }
        let row = self.require_row(name)?;
        let hash_started = Instant::now();
        let observed = sha256_hex(payload);
        let hash_ns = hash_started.elapsed().as_nanos() as u64;
        self.add_first_touch(|split| {
            split.sha256_ns = split.sha256_ns.saturating_add(hash_ns);
            split.payloads_hashed = split.payloads_hashed.saturating_add(1);
        });
        if observed != row.artifact_sha256 {
            return Err(q80q4_error(format!(
                "tensor {name:?} sha256 {observed} != catalog {}",
                row.artifact_sha256
            )));
        }
        if let Ok(mut set) = self.verified_sha256.lock() {
            set.insert(name.to_owned());
        }
        Ok(())
    }

    fn check_payload_geometry(&self, name: &str, payload: &[u8]) -> Result<CompleteBinaryHeader> {
        let row = self.require_row(name)?;
        if payload.len() < 32 {
            return Err(q80q4_error(format!(
                "tensor {name:?} payload is truncated ({} bytes)",
                payload.len()
            )));
        }
        let parse_started = Instant::now();
        let header = parse_uniform_q4_header(payload)?;
        let parse_ns = parse_started.elapsed().as_nanos() as u64;
        self.add_first_touch(|split| {
            split.header_parse_ns = split.header_parse_ns.saturating_add(parse_ns);
        });
        if header.shape != row.shape || header.group_size != UNIFORM_Q4_GROUP_SIZE {
            return Err(q80q4_error(format!(
                "tensor {name:?} header shape {:?}/group {} disagrees with catalog {:?}/64",
                header.shape, header.group_size, row.shape
            )));
        }
        Ok(header)
    }

    /// Read one payload. A missing or short file raises; the body is never
    /// silently zero-filled. SHA-256 runs only when the session is not
    /// admitted (or `HAWKING_Q80_REHASH_PAYLOADS=1`).
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
        let read_started = Instant::now();
        let payload = fs::read(&row.artifact_path).map_err(|error| {
            q80q4_error(format!(
                "cannot read tensor {name:?} {}: {error}",
                row.artifact_path.display()
            ))
        })?;
        let read_ns = read_started.elapsed().as_nanos() as u64;
        self.add_first_touch(|split| {
            split.catalog_read_ns = split.catalog_read_ns.saturating_add(read_ns);
            split.payloads_read = split.payloads_read.saturating_add(1);
        });
        if (payload.len() as u64) != row.artifact_bytes {
            return Err(q80q4_error(format!(
                "tensor {name:?} read {} bytes, catalog {}",
                payload.len(),
                row.artifact_bytes
            )));
        }
        self.verify_payload_sha_if_needed(name, &payload)?;
        let _ = self.check_payload_geometry(name, &payload)?;
        Ok(Arc::from(payload))
    }

    /// Always rehash. Used by the correctness gate that cached/mmap'd
    /// bytes must equal a freshly-hashed read.
    pub fn read_payload_rehash(&self, name: &str) -> Result<Arc<[u8]>> {
        let _ = self.hash_payload(name)?;
        let row = self.require_row(name)?;
        let payload = fs::read(&row.artifact_path).map_err(|error| {
            q80q4_error(format!(
                "cannot re-read tensor {name:?} {}: {error}",
                row.artifact_path.display()
            ))
        })?;
        let _ = self.check_payload_geometry(name, &payload)?;
        Ok(Arc::from(payload))
    }

    /// mmap the tensor file, rounded up to the Metal no-copy page size.
    /// Does not SHA when the session is admitted.
    pub fn map_payload(&self, name: &str) -> Result<Qwen80MappedTensor> {
        let row = self.require_row(name)?;
        let file = fs::File::open(&row.artifact_path).map_err(|error| {
            q80q4_error(format!(
                "cannot open tensor {name:?} {}: {error}",
                row.artifact_path.display()
            ))
        })?;
        let metadata = file.metadata().map_err(|error| {
            q80q4_error(format!(
                "cannot stat tensor {name:?} {}: {error}",
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
        let file_bytes = usize::try_from(row.artifact_bytes)
            .map_err(|_| q80q4_error(format!("tensor {name:?} byte count exceeds this platform")))?;
        let align = crate::metal::MetalContext::NO_COPY_PAGE_ALIGN;
        let map_len = file_bytes.div_ceil(align).saturating_mul(align).max(align);
        let map_started = Instant::now();
        // SAFETY: regular catalog file, read-only MAP. Length may exceed
        // the file so the Metal no-copy window is page-aligned; pages past
        // EOF are zero and the kernel never reads them.
        let mmap = unsafe {
            MmapOptions::new().len(map_len).map(&file).map_err(|error| {
                q80q4_error(format!(
                    "cannot mmap tensor {name:?} {}: {error}",
                    row.artifact_path.display()
                ))
            })?
        };
        let _ = mmap.advise(Advice::WillNeed);
        let map_ns = map_started.elapsed().as_nanos() as u64;
        self.add_first_touch(|split| {
            split.catalog_read_ns = split.catalog_read_ns.saturating_add(map_ns);
            split.mmap_ns = split.mmap_ns.saturating_add(map_ns);
            split.payloads_mmapped = split.payloads_mmapped.saturating_add(1);
        });
        if mmap.len() < file_bytes {
            return Err(q80q4_error(format!(
                "tensor {name:?} mmap {} is shorter than catalog {file_bytes}",
                mmap.len()
            )));
        }
        let payload = &mmap[..file_bytes];
        self.verify_payload_sha_if_needed(name, payload)?;
        let header = self.check_payload_geometry(name, payload)?;
        Ok(Qwen80MappedTensor {
            name: name.to_owned(),
            header,
            file_bytes,
            mmap: Arc::new(mmap),
        })
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

    /// Packed Q4 codes (nibble body starting at `sign_offset`).
    pub fn codes(&self) -> Result<&[u8]> {
        self.payload
            .get(self.header.sign_offset..self.header.payload_bytes)
            .ok_or_else(|| q80q4_error(format!("tensor {:?} codes truncated", self.name)))
    }

    /// Packed FP16 group scales.
    pub fn scales(&self) -> Result<&[u8]> {
        self.payload
            .get(self.header.scale_offset..self.header.sign_offset)
            .ok_or_else(|| q80q4_error(format!("tensor {:?} scales truncated", self.name)))
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
    pub expert_table_layer_builds: u64,
    pub expert_table_waves: u64,
    pub expert_table_matvec_dispatches: u64,
    pub device_activation_dispatches: u64,
    pub expert_upload_hits: u64,
    pub expert_upload_misses: u64,
    pub expert_residency_evictions: u64,
    pub expert_table_slot_patches: u64,
    pub expert_resident_slots: u64,
    pub expert_resident_bytes: u64,
    pub expert_nocopy_binds: u64,
    pub expert_copy_binds: u64,
    pub expert_prefetch_uploads: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Qwen80DecodeStageTimes {
    pub embed_secs: f64,
    pub deltanet_secs: f64,
    pub gqa_secs: f64,
    pub moe_norm_router_secs: f64,
    pub moe_shared_secs: f64,
    pub moe_table_build_secs: f64,
    pub moe_table_upload_miss_secs: f64,
    pub moe_table_entries_fill_secs: f64,
    pub moe_table_buffer_write_secs: f64,
    pub moe_table_resource_clone_secs: f64,
    pub moe_table_lease_secs: f64,
    pub first_touch_catalog_read_secs: f64,
    pub first_touch_sha256_secs: f64,
    pub first_touch_metal_copy_secs: f64,
    pub first_touch_metal_nocopy_secs: f64,
    pub prefetch_secs: f64,
    pub moe_routed_secs: f64,
    pub moe_combine_secs: f64,
    pub terminal_secs: f64,
    pub q4_matvec_secs: f64,
    pub host_expert_bind_secs: f64,
    pub embed_ns: u64,
    pub deltanet_ns: u64,
    pub gqa_ns: u64,
    pub moe_norm_router_ns: u64,
    pub moe_shared_ns: u64,
    pub moe_table_build_ns: u64,
    pub moe_routed_ns: u64,
    pub moe_combine_ns: u64,
    pub terminal_ns: u64,
    pub q4_matvec_ns: u64,
    pub host_expert_bind_ns: u64,
    pub activation: Qwen80ActivationClassTimes,
}

/// Per-token attribution of the five host-activation classes.
/// `*_secs` is wall time of that class only (CPU op, or the sandwich that
/// includes adjacent matvec round-trips when named `_sandwich`).
/// `*_ns` is the same interval in nanoseconds — a class that only exists
/// as encode-into-a-mixed-CB still reports encode ns, never a silent zero.
#[derive(Clone, Debug, Default)]
pub struct Qwen80ActivationClassTimes {
    pub shared_swiglu_secs: f64,
    pub shared_mlp_sandwich_secs: f64,
    pub deltanet_conv_secs: f64,
    pub deltanet_recurrent_secs: f64,
    pub gqa_input_layernorm_secs: f64,
    pub gqa_norm_rope_secs: f64,
    pub other_host_activation_secs: f64,
    pub metal_matvec_sync_secs: f64,
    pub shared_swiglu_ns: u64,
    pub shared_mlp_sandwich_ns: u64,
    pub deltanet_conv_ns: u64,
    pub deltanet_recurrent_ns: u64,
    pub gqa_input_layernorm_ns: u64,
    pub gqa_norm_rope_ns: u64,
    pub other_host_activation_ns: u64,
    pub metal_matvec_sync_ns: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Qwen80ActivationClassCounts {
    pub shared_swiglu: u64,
    pub deltanet_conv: u64,
    pub deltanet_recurrent: u64,
    pub gqa_input_layernorm: u64,
    pub gqa_norm_rope: u64,
    pub other_host_activation: u64,
}

/// Device-path GPU / host split for mixer families.
///
/// GPU nanoseconds are `MTLCommandBuffer.GPUEndTime − GPUStartTime` after
/// wait (or after a later same-queue wait for overlapped CBs). They are
/// never a CPU-wall proxy. On the historical fused-prefix path the mixer
/// CB also contains shared-expert + router work and is flagged mixed.
#[derive(Clone, Debug, Default)]
pub struct Qwen80FamilyGpuTimes {
    pub overlap_cbs: bool,
    pub overlap_mode: &'static str,
    pub serial_mixer: bool,
    pub component_matvec_kernel: &'static str,
    pub deltanet_mixer_cbs: u64,
    pub deltanet_mixer_dispatches: u64,
    pub deltanet_mixer_gpu_ns: u64,
    pub deltanet_mixer_wait_ns: u64,
    pub deltanet_mixer_encode_ns: u64,
    pub deltanet_mixer_mixed_with_moe_prefix: bool,
    pub gqa_mixer_cbs: u64,
    pub gqa_mixer_dispatches: u64,
    pub gqa_mixer_gpu_ns: u64,
    pub gqa_mixer_wait_ns: u64,
    pub gqa_mixer_encode_ns: u64,
    pub gqa_mixer_mixed_with_moe_prefix: bool,
    pub moe_prefix_cbs: u64,
    pub moe_prefix_dispatches: u64,
    pub moe_prefix_gpu_ns: u64,
    pub moe_prefix_wait_ns: u64,
    pub moe_prefix_encode_ns: u64,
    pub moe_suffix_cbs: u64,
    pub moe_suffix_dispatches: u64,
    pub moe_suffix_gpu_ns: u64,
    pub moe_suffix_wait_ns: u64,
    pub moe_suffix_encode_ns: u64,
    pub moe_suffix_combine_dispatches: u64,
    pub fused_layer_cbs: u64,
    pub fused_layer_dispatches: u64,
    pub fused_layer_gpu_ns: u64,
    pub fused_layer_wait_ns: u64,
    pub gpu_gap_ns: u64,
    pub gpu_gap_edges: u64,
    pub bytes_read_weight: u64,
    pub last_gpu_end_ns: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
enum FamilyGpuKind {
    DeltaNetMixer { mixed: bool },
    GqaMixer { mixed: bool },
    MoePrefix,
    MoeSuffix,
    FusedLayer,
    Other,
}

impl Qwen80FamilyGpuTimes {
    fn note_cb(&mut self, timing: &crate::metal::CommandBufferTiming) {
        if let (Some(start), Some(prev_end)) = (timing.gpu_start_ns, self.last_gpu_end_ns) {
            if start > prev_end {
                self.gpu_gap_ns = self.gpu_gap_ns.saturating_add(start - prev_end);
                self.gpu_gap_edges = self.gpu_gap_edges.saturating_add(1);
            }
        }
        if let Some(end) = timing.gpu_end_ns {
            self.last_gpu_end_ns = Some(end);
        }
    }
}

#[allow(dead_code)]
fn add_secs(slot: &mut f64, started: Instant) {
    *slot += started.elapsed().as_secs_f64();
}

fn add_elapsed(secs: &mut f64, ns: &mut u64, started: Instant) {
    let elapsed = started.elapsed();
    *secs += elapsed.as_secs_f64();
    *ns = ns.saturating_add(elapsed.as_nanos() as u64);
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
struct DeviceActivationWorkspace {
    hidden: crate::metal::PinnedBuffer,
    normalized: crate::metal::PinnedBuffer,
    mixer: crate::metal::PinnedBuffer,
    first_residual: crate::metal::PinnedBuffer,
    gate: crate::metal::PinnedBuffer,
    up: crate::metal::PinnedBuffer,
    act: crate::metal::PinnedBuffer,
    shared: crate::metal::PinnedBuffer,
    shared_logit: crate::metal::PinnedBuffer,
    gated_shared: crate::metal::PinnedBuffer,
    qkvz: crate::metal::PinnedBuffer,
    ba: crate::metal::PinnedBuffer,
    repeated_q: crate::metal::PinnedBuffer,
    repeated_k: crate::metal::PinnedBuffer,
    conv_v: crate::metal::PinnedBuffer,
    z: crate::metal::PinnedBuffer,
    decay: crate::metal::PinnedBuffer,
    beta: crate::metal::PinnedBuffer,
    rec_out: crate::metal::PinnedBuffer,
    gated: crate::metal::PinnedBuffer,
    q_proj: crate::metal::PinnedBuffer,
    k_proj: crate::metal::PinnedBuffer,
    v_proj: crate::metal::PinnedBuffer,
    query: crate::metal::PinnedBuffer,
    attn: crate::metal::PinnedBuffer,
    gated_attn: crate::metal::PinnedBuffer,
    router_logits: crate::metal::PinnedBuffer,
    logits: crate::metal::PinnedBuffer,
    linear_conv: crate::metal::PinnedBuffer,
    linear_recurrent: crate::metal::PinnedBuffer,
    gqa_key: crate::metal::PinnedBuffer,
    gqa_value: crate::metal::PinnedBuffer,
    vectors: HashMap<String, crate::metal::PinnedBuffer>,
    n_linear: usize,
    n_gqa: usize,
    max_seq_len: usize,
}

#[cfg(target_os = "macos")]
struct MetalQ4Accel {
    context: crate::metal::MetalContext,
    weights: HashMap<String, MetalQ4Weight>,
    expert_table: Option<super::qwen80_device_expert_table::Qwen80DeviceExpertTableLease>,
    expert_wave: Option<super::qwen80_device_expert_table::Qwen80DeviceExpertWaveWorkspace>,
    expert_kernel: super::qwen80_device_expert_table::Qwen80ExpertTableKernel,
    expert_cache: HashMap<(usize, u32), super::qwen80_device_expert_table::Qwen80ExpertGpuTriplet>,
    expert_slabs: Option<super::qwen80_device_expert_table::Qwen80CompactExpertSlabs>,
    residency: Option<
        super::device_residency::ResidencyPool<
            super::qwen80_device_expert_table::Qwen80ExpertGpuTriplet,
        >,
    >,
    activations: Option<DeviceActivationWorkspace>,
    last_routes: [[u32; 10]; QWEN80_LAYERS],
    have_last_routes: [bool; QWEN80_LAYERS],
    buffer_creations: u64,
    buffer_rebinds: u64,
}

#[cfg(target_os = "macos")]
fn bytes_f32(n: usize, label: &str) -> Result<usize> {
    n.checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| q80q4_error(format!("{label} byte count overflowed")))
}

#[cfg(target_os = "macos")]
impl DeviceActivationWorkspace {
    fn allocate(context: &crate::metal::MetalContext, max_seq_len: usize) -> Result<Self> {
        let linear = Qwen80CanonicalLinearDeltaNetLayout::source_exact();
        linear.validate()?;
        let gqa = Qwen80CanonicalGqaLayout::source_exact();
        gqa.validate()?;
        let mut n_linear = 0usize;
        let mut n_gqa = 0usize;
        for layer in 0..QWEN80_LAYERS {
            match qwen80_layer_kind(layer)? {
                Qwen80LayerKind::LinearAttention => n_linear += 1,
                Qwen80LayerKind::FullAttention => n_gqa += 1,
            }
        }
        let hidden = bytes_f32(QWEN80_HIDDEN, "act hidden")?;
        let mid = bytes_f32(QWEN80_MOE_INTERMEDIATE, "act moe mid")?;
        let qkvz = bytes_f32(linear.qkvz_projection_elements()?, "act qkvz")?;
        let ba = bytes_f32(linear.ba_projection_elements()?, "act ba")?;
        let value = bytes_f32(linear.value_elements()?, "act value")?;
        let q_proj = bytes_f32(gqa.q_proj_rows, "act q_proj")?;
        let kv = bytes_f32(gqa.kv_dim, "act kv")?;
        let query = bytes_f32(gqa.query_dim, "act query")?;
        let conv_bytes = bytes_f32(
            n_linear
                .checked_mul(linear.conv_state_elements()?)
                .ok_or_else(|| q80q4_error("act conv state overflowed"))?,
            "act conv state",
        )?;
        let rec_bytes = bytes_f32(
            n_linear
                .checked_mul(linear.recurrent_state_elements()?)
                .ok_or_else(|| q80q4_error("act recurrent state overflowed"))?,
            "act recurrent state",
        )?;
        let gqa_bytes = bytes_f32(
            n_gqa
                .checked_mul(max_seq_len)
                .and_then(|v| v.checked_mul(gqa.kv_dim))
                .ok_or_else(|| q80q4_error("act gqa cache overflowed"))?,
            "act gqa cache",
        )?;
        Ok(Self {
            hidden: context.new_buffer_checked(hidden)?,
            normalized: context.new_buffer_checked(hidden)?,
            mixer: context.new_buffer_checked(hidden)?,
            first_residual: context.new_buffer_checked(hidden)?,
            gate: context.new_buffer_checked(mid)?,
            up: context.new_buffer_checked(mid)?,
            act: context.new_buffer_checked(mid)?,
            shared: context.new_buffer_checked(hidden)?,
            shared_logit: context.new_buffer_checked(bytes_f32(1, "act shared logit")?)?,
            gated_shared: context.new_buffer_checked(hidden)?,
            qkvz: context.new_buffer_checked(qkvz)?,
            ba: context.new_buffer_checked(ba)?,
            repeated_q: context.new_buffer_checked(value)?,
            repeated_k: context.new_buffer_checked(value)?,
            conv_v: context.new_buffer_checked(value)?,
            z: context.new_buffer_checked(value)?,
            decay: context.new_buffer_checked(bytes_f32(linear.value_heads, "act decay")?)?,
            beta: context.new_buffer_checked(bytes_f32(linear.value_heads, "act beta")?)?,
            rec_out: context.new_buffer_checked(value)?,
            gated: context.new_buffer_checked(value)?,
            q_proj: context.new_buffer_checked(q_proj)?,
            k_proj: context.new_buffer_checked(kv)?,
            v_proj: context.new_buffer_checked(kv)?,
            query: context.new_buffer_checked(query)?,
            attn: context.new_buffer_checked(query)?,
            gated_attn: context.new_buffer_checked(query)?,
            router_logits: context.new_buffer_checked(bytes_f32(QWEN80_EXPERTS, "act router")?)?,
            logits: context.new_buffer_checked(bytes_f32(QWEN80_VOCAB, "act logits")?)?,
            linear_conv: context.new_buffer_checked(conv_bytes)?,
            linear_recurrent: context.new_buffer_checked(rec_bytes)?,
            gqa_key: context.new_buffer_checked(gqa_bytes)?,
            gqa_value: context.new_buffer_checked(gqa_bytes)?,
            vectors: HashMap::new(),
            n_linear,
            n_gqa,
            max_seq_len,
        })
    }

    fn zero_state(&mut self) {
        fn zero(buf: &crate::metal::PinnedBuffer) {
            let len = buf.length() as usize;
            if len == 0 {
                return;
            }
            unsafe {
                std::ptr::write_bytes(buf.contents() as *mut u8, 0, len);
            }
        }
        zero(&self.linear_conv);
        zero(&self.linear_recurrent);
        zero(&self.gqa_key);
        zero(&self.gqa_value);
        let _ = (self.n_linear, self.n_gqa, self.max_seq_len);
    }
}

#[cfg(target_os = "macos")]
impl MetalQ4Accel {
    fn new(max_seq_len: usize) -> Result<Self> {
        let context = crate::metal::MetalContext::new()?;
        let expert_wave =
            super::qwen80_device_expert_table::Qwen80DeviceExpertWaveWorkspace::allocate(&context)?;
        let expert_slabs = if super::qwen80_device_expert_table::qwen80_compact_expert_slabs_enabled()
        {
            Some(super::qwen80_device_expert_table::Qwen80CompactExpertSlabs::allocate(&context)?)
        } else {
            None
        };
        let residency = if super::device_residency::persistent_address_table_enabled()
            && expert_slabs.is_none()
        {
            let table = super::device_residency::PersistentAddressTable::allocate(
                &context,
                super::qwen80_device_expert_table::qwen80_q4_address_geometry(),
            )?;
            Some(super::device_residency::ResidencyPool::new(
                table,
                super::device_residency::residency_budget_bytes(),
            ))
        } else {
            None
        };
        let activations = if qwen80_device_activations_enabled() {
            Some(DeviceActivationWorkspace::allocate(&context, max_seq_len)?)
        } else {
            None
        };
        Ok(Self {
            context,
            weights: HashMap::new(),
            expert_table: None,
            expert_wave: Some(expert_wave),
            expert_kernel: super::qwen80_device_expert_table::qwen80_expert_table_kernel(),
            expert_cache: HashMap::new(),
            expert_slabs,
            residency,
            activations,
            last_routes: [[0u32; 10]; QWEN80_LAYERS],
            have_last_routes: [false; QWEN80_LAYERS],
            buffer_creations: 0,
            buffer_rebinds: 0,
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
            self.buffer_creations = self.buffer_creations.saturating_add(2);
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
        stages: &mut Qwen80DecodeStageTimes,
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
        let sync_started = Instant::now();
        tcb.commit_and_wait()?;
        let observed =
            unsafe { std::slice::from_raw_parts(output_buf.contents() as *const f32, rows) };
        output.copy_from_slice(observed);
        add_elapsed(
            &mut stages.activation.metal_matvec_sync_secs,
            &mut stages.activation.metal_matvec_sync_ns,
            sync_started,
        );
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

    fn embed_into(
        &mut self,
        packed: &Qwen80Q4PackedTensor,
        token: u32,
        output: &crate::metal::PinnedBuffer,
        native: &mut Qwen80DecodeNativeCounts,
    ) -> Result<crate::metal::CommandBufferTiming> {
        use crate::metal::TokenCommandBuffer;
        if packed.header.shape != [QWEN80_VOCAB, QWEN80_HIDDEN] {
            return Err(q80q4_error("metal q4 embedding shape drifted"));
        }
        if output.length() < (QWEN80_HIDDEN * std::mem::size_of::<f32>()) as u64 {
            return Err(q80q4_error("metal q4 embedding output buffer is short"));
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
        let output_buf = output.clone();
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
        let timing = tcb.commit_and_wait_timed()?;
        native.q4_embedding_dispatches = native.q4_embedding_dispatches.saturating_add(1);
        Ok(timing)
    }

    fn bind_real_route_ids(&self) -> bool {
        self.residency.is_some() || self.expert_slabs.is_some()
    }

    fn copy_residency_snapshot(&self, native: &mut Qwen80DecodeNativeCounts) {
        if let Some(pool) = self.residency.as_ref() {
            native.expert_upload_hits = pool.stats.upload_hits;
            native.expert_upload_misses = pool.stats.upload_misses;
            native.expert_residency_evictions = pool.stats.evictions;
            native.expert_table_slot_patches = pool.stats.table_slot_patches;
            native.expert_resident_slots = pool.stats.resident_slots;
            native.expert_resident_bytes = pool.stats.resident_bytes;
        }
    }

    fn copy_first_touch_snapshot(
        catalog: &Qwen80UniformQ4StreamingCatalog,
        native: &mut Qwen80DecodeNativeCounts,
        stages: &mut Qwen80DecodeStageTimes,
    ) {
        let split = catalog.first_touch_split();
        stages.first_touch_catalog_read_secs = split.catalog_read_ns as f64 / 1e9;
        stages.first_touch_sha256_secs = split.sha256_ns as f64 / 1e9;
        stages.first_touch_metal_copy_secs = split.metal_copy_ns as f64 / 1e9;
        stages.first_touch_metal_nocopy_secs = split.metal_nocopy_ns as f64 / 1e9;
        stages.prefetch_secs = split.prefetch_ns as f64 / 1e9;
        native.expert_nocopy_binds = split.nocopy_binds;
        native.expert_copy_binds = split.copy_binds;
        native.expert_prefetch_uploads = split.prefetch_uploads;
    }

    fn ensure_selected_expert_table(
        &mut self,
        catalog: &Qwen80UniformQ4StreamingCatalog,
        layer: usize,
        route_ids: &[u32],
        native: &mut Qwen80DecodeNativeCounts,
        stages: &mut Qwen80DecodeStageTimes,
    ) -> Result<()> {
        let started = Instant::now();
        if self.residency.is_some() {
            {
                let context = self.context.clone();
                let pool = self.residency.as_mut().expect("residency checked");
                pool.ensure_selected(layer, route_ids, |miss_layer, miss_expert| {
                    super::qwen80_device_expert_table::upload_qwen80_expert_triplet(
                        &context,
                        catalog,
                        miss_layer,
                        miss_expert as usize,
                    )
                })?;
                stages.moe_table_upload_miss_secs = pool.stats.upload_miss_secs;
                stages.moe_table_entries_fill_secs = pool.stats.entries_fill_secs;
                stages.moe_table_buffer_write_secs = pool.stats.buffer_write_secs;
                stages.moe_table_resource_clone_secs = pool.stats.resource_clone_secs;
                stages.moe_table_lease_secs = pool.stats.lease_secs;
            }
            let bind = self
                .residency
                .as_mut()
                .expect("residency checked")
                .layer_bind(layer, route_ids)?;
            self.copy_residency_snapshot(native);
            Self::copy_first_touch_snapshot(catalog, native, stages);
            self.expert_table =
                Some(super::qwen80_device_expert_table::Qwen80DeviceExpertTableLease::from_residency_bind(
                    bind,
                ));
            add_secs(&mut stages.moe_table_build_secs, started);
            native.expert_table_layer_builds = native.expert_table_layer_builds.saturating_add(1);
            return Ok(());
        }
        for &expert in route_ids {
            if self.expert_cache.contains_key(&(layer, expert)) {
                native.expert_upload_hits = native.expert_upload_hits.saturating_add(1);
                continue;
            }
            native.expert_upload_misses = native.expert_upload_misses.saturating_add(1);
            let upload_started = Instant::now();
            let trip = super::qwen80_device_expert_table::upload_qwen80_expert_triplet(
                &self.context,
                catalog,
                layer,
                expert as usize,
            )?;
            add_secs(&mut stages.moe_table_upload_miss_secs, upload_started);
            self.expert_cache.insert((layer, expert), trip);
        }
        let fill_started = Instant::now();
        let selected: Vec<(
            u32,
            &super::qwen80_device_expert_table::Qwen80ExpertGpuTriplet,
        )> = route_ids
            .iter()
            .map(|&expert| {
                let trip = self
                    .expert_cache
                    .get(&(layer, expert))
                    .expect("just inserted");
                (expert, trip)
            })
            .collect();
        add_secs(&mut stages.moe_table_entries_fill_secs, fill_started);
        let reused = self.expert_table.take().map(|lease| lease.table);
        let write_started = Instant::now();
        let lease = if let Some(slabs) = self.expert_slabs.as_ref() {
            super::qwen80_device_expert_table::write_compact_selected_table(
                &self.context,
                slabs,
                layer,
                &selected,
                reused,
            )?
        } else {
            // Historical A/B path: ten-entry rewrite + 0..10 remap.
            super::qwen80_device_expert_table::write_top10_address_table(
                &self.context,
                layer,
                &selected,
                reused,
            )?
        };
        add_secs(&mut stages.moe_table_buffer_write_secs, write_started);
        let lease_started = Instant::now();
        Self::copy_first_touch_snapshot(catalog, native, stages);
        self.expert_table = Some(lease);
        add_secs(&mut stages.moe_table_lease_secs, lease_started);
        native.expert_resident_slots = self.expert_cache.len() as u64;
        add_elapsed(
            &mut stages.moe_table_build_secs,
            &mut stages.moe_table_build_ns,
            started,
        );
        self.buffer_rebinds = self.buffer_rebinds.saturating_add(1);
        native.expert_table_layer_builds = native.expert_table_layer_builds.saturating_add(1);
        Ok(())
    }

    fn prefetch_predicted_layer(
        &mut self,
        catalog: &Qwen80UniformQ4StreamingCatalog,
        layer: usize,
    ) -> Result<u64> {
        if !qwen80_expert_prefetch_enabled() {
            return Ok(0);
        }
        if layer >= QWEN80_LAYERS || !self.have_last_routes[layer] {
            return Ok(0);
        }
        let predicted = self.last_routes[layer];
        let started = Instant::now();
        if self.residency.is_some() {
            let context = self.context.clone();
            let pool = self.residency.as_mut().expect("residency checked");
            let mut misses = 0u64;
            pool.ensure_selected(layer, &predicted, |miss_layer, miss_expert| {
                misses = misses.saturating_add(1);
                super::qwen80_device_expert_table::upload_qwen80_expert_triplet(
                    &context,
                    catalog,
                    miss_layer,
                    miss_expert as usize,
                )
            })?;
            let hits = predicted.len().saturating_sub(misses as usize) as u64;
            catalog.add_first_touch(|split| {
                split.prefetch_uploads = split.prefetch_uploads.saturating_add(misses);
                split.prefetch_misses = split.prefetch_misses.saturating_add(misses);
                split.prefetch_hits = split.prefetch_hits.saturating_add(hits);
            });
        }
        let ns = started.elapsed().as_nanos() as u64;
        catalog.add_first_touch(|split| {
            split.prefetch_ns = split.prefetch_ns.saturating_add(ns);
        });
        Ok(ns)
    }

    fn remember_routes(&mut self, layer: usize, route_ids: &[u32]) {
        if layer >= QWEN80_LAYERS {
            return;
        }
        let mut stored = [0u32; 10];
        let n = route_ids.len().min(10);
        stored[..n].copy_from_slice(&route_ids[..n]);
        self.last_routes[layer] = stored;
        self.have_last_routes[layer] = true;
    }

    fn routed_expert_table(
        &mut self,
        catalog: &Qwen80UniformQ4StreamingCatalog,
        layer: usize,
        route_ids: &[u32],
        route_weights: &[f32],
        input: &[f32],
        output: &mut [f32],
        native: &mut Qwen80DecodeNativeCounts,
        stages: &mut Qwen80DecodeStageTimes,
    ) -> Result<()> {
        self.ensure_selected_expert_table(catalog, layer, route_ids, native, stages)?;
        let lease = self
            .expert_table
            .as_ref()
            .ok_or_else(|| q80q4_error("expert table lease missing after ensure"))?;
        let wave = self
            .expert_wave
            .as_ref()
            .ok_or_else(|| q80q4_error("expert table workspace missing"))?;
        let started = Instant::now();
        let dispatches = super::qwen80_device_expert_table::run_qwen80_routed_expert_table(
            &self.context,
            lease,
            wave,
            route_ids,
            route_weights,
            input,
            output,
            self.expert_kernel,
        )?;
        add_elapsed(&mut stages.moe_routed_secs, &mut stages.moe_routed_ns, started);
        native.expert_table_waves = native.expert_table_waves.saturating_add(1);
        native.expert_table_matvec_dispatches = native
            .expert_table_matvec_dispatches
            .saturating_add(dispatches);
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
    pub stages: Qwen80DecodeStageTimes,
    pub activation_counts: Qwen80ActivationClassCounts,
    pub family_gpu: Qwen80FamilyGpuTimes,
    pub token_ns: Qwen80TokenNsSession,
    #[cfg(target_os = "macos")]
    metal: Option<MetalQ4Accel>,
    pub metal_error: Option<String>,
}

impl Qwen80UniformQ4HybridDecodeSession {
    pub fn reload_kernels(&mut self) -> Result<Qwen80KernelReloadReport> {
        let started = Instant::now();
        #[cfg(target_os = "macos")]
        {
            match MetalQ4Accel::new(self.state.max_seq_len) {
                Ok(accel) => {
                    self.metal = Some(accel);
                    self.metal_error = None;
                }
                Err(error) => {
                    self.metal = None;
                    self.metal_error = Some(error.to_string());
                }
            }
        }
        #[cfg(target_os = "macos")]
        let rebuilt_metal = self.metal.is_some();
        #[cfg(not(target_os = "macos"))]
        let rebuilt_metal = false;
        Ok(Qwen80KernelReloadReport {
            rebuilt_metal,
            catalog_reopened: false,
            elapsed_secs: started.elapsed().as_secs_f64(),
            metal_error: self.metal_error.clone(),
        })
    }

    pub fn new(mut catalog: Qwen80UniformQ4StreamingCatalog, max_seq_len: usize) -> Result<Self> {
        catalog.admit_session()?;
        #[cfg(target_os = "macos")]
        let (metal, metal_error) = match MetalQ4Accel::new(max_seq_len) {
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
            stages: Qwen80DecodeStageTimes::default(),
            activation_counts: Qwen80ActivationClassCounts::default(),
            family_gpu: Qwen80FamilyGpuTimes {
                overlap_cbs: qwen80_overlap_cbs_enabled(),
                overlap_mode: match qwen80_overlap_mode() {
                    OverlapMode::Off => "off",
                    OverlapMode::Split => "split",
                    OverlapMode::Fuse => "fuse",
                },
                serial_mixer: qwen80_serial_mixer_enabled(),
                component_matvec_kernel: QWEN80_COMPONENT_MATVEC_KERNEL,
                ..Qwen80FamilyGpuTimes::default()
            },
            token_ns: Qwen80TokenNsSession::from_env(),
            #[cfg(target_os = "macos")]
            metal,
            metal_error,
        })
    }

    pub fn catalog(&self) -> &Qwen80UniformQ4StreamingCatalog {
        &self.catalog
    }

    #[cfg(target_os = "macos")]
    pub fn device_memory_limits(&self) -> Option<crate::metal::DeviceMemoryLimits> {
        self.metal.as_ref().map(|metal| metal.context.device_memory_limits())
    }

    #[cfg(not(target_os = "macos"))]
    pub fn device_memory_limits(&self) -> Option<crate::metal::DeviceMemoryLimits> {
        None
    }

    pub fn residency_stats(&self) -> Option<super::device_residency::ResidencyStats> {
        #[cfg(target_os = "macos")]
        {
            return self
                .metal
                .as_ref()
                .and_then(|metal| metal.residency.as_ref())
                .map(|pool| pool.stats.clone());
        }
        #[cfg(not(target_os = "macos"))]
        None
    }

    pub fn residency_budget_bytes(&self) -> Option<u64> {
        #[cfg(target_os = "macos")]
        {
            return self
                .metal
                .as_ref()
                .and_then(|metal| metal.residency.as_ref())
                .map(|pool| pool.budget_bytes());
        }
        #[cfg(not(target_os = "macos"))]
        None
    }

    pub fn reset_state(&mut self) {
        self.state.reset();
        #[cfg(target_os = "macos")]
        if let Some(metal) = self.metal.as_mut() {
            if let Some(act) = metal.activations.as_mut() {
                act.zero_state();
            }
        }
    }

    fn matvec_named(&mut self, name: &str, input: &[f32], output: &mut [f32]) -> Result<()> {
        let started = Instant::now();
        let packed = self.cache.packed(&self.catalog, name)?.clone();
        #[cfg(target_os = "macos")]
        if let Some(metal) = self.metal.as_mut() {
            match metal.matvec(&packed, input, output, &mut self.native, &mut self.stages) {
                Ok(()) => {
                    add_elapsed(&mut self.stages.q4_matvec_secs, &mut self.stages.q4_matvec_ns, started);
                    return Ok(());
                }
                Err(error) => {
                    if self.metal_error.is_none() {
                        self.metal_error = Some(error.to_string());
                    }
                }
            }
        }
        packed.matvec(input, output)?;
        self.fallbacks.host_q4_matvec = self.fallbacks.host_q4_matvec.saturating_add(1);
        add_elapsed(&mut self.stages.q4_matvec_secs, &mut self.stages.q4_matvec_ns, started);
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
        let sandwich = Instant::now();
        self.matvec_named(gate_name, input, &mut gate)?;
        self.matvec_named(up_name, input, &mut up)?;
        let silu_started = Instant::now();
        silu_mul(&gate, &up, &mut act);
        add_elapsed(
            &mut self.stages.activation.shared_swiglu_secs,
            &mut self.stages.activation.shared_swiglu_ns,
            silu_started,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.shared_swiglu =
            self.activation_counts.shared_swiglu.saturating_add(1);
        self.matvec_named(down_name, &act, &mut down)?;
        add_elapsed(
            &mut self.stages.activation.shared_mlp_sandwich_secs,
            &mut self.stages.activation.shared_mlp_sandwich_ns,
            sandwich,
        );
        Ok(down)
    }

    fn deltanet_mixer(&mut self, layer: usize, hidden: &[f32]) -> Result<Vec<f32>> {
        let layout = Qwen80CanonicalLinearDeltaNetLayout::source_exact();
        let slot = self.state.linear_slot_for_layer(layer)?;
        let input_w = self.vector(&Self::layer_name(layer, "input_layernorm.weight"))?;
        let rms_started = Instant::now();
        let rms = source_qwen80_residual_rms_norm(hidden, &input_w)?;
        add_elapsed(
            &mut self.stages.activation.other_host_activation_secs,
            &mut self.stages.activation.other_host_activation_ns,
            rms_started,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(1);
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
        let conv_started = Instant::now();
        let (convolved_qkv, next_conv) = source_qwen80_causal_conv_step_dense(
            &mixed_qkv,
            &self.state.linear_conv[slot],
            &conv_w,
            &layout,
        )?;
        add_elapsed(
            &mut self.stages.activation.deltanet_conv_secs,
            &mut self.stages.activation.deltanet_conv_ns,
            conv_started,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.deltanet_conv =
            self.activation_counts.deltanet_conv.saturating_add(1);
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
        let recurrent_started = Instant::now();
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
        add_elapsed(
            &mut self.stages.activation.deltanet_recurrent_secs,
            &mut self.stages.activation.deltanet_recurrent_ns,
            recurrent_started,
        );
        self.state.linear_conv[slot] = next_conv;
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.deltanet_recurrent =
            self.activation_counts.deltanet_recurrent.saturating_add(1);
        let gated_norm = self.vector(&Self::layer_name(layer, "linear_attn.norm.weight"))?;
        let repeated_gated_norm = (0..layout.value_heads)
            .flat_map(|_| gated_norm.iter().copied())
            .collect::<Vec<_>>();
        let gated_started = Instant::now();
        let gated_output = source_qwen80_gated_rms_norm(
            &recurrent_output,
            &z,
            &repeated_gated_norm,
            layout.value_heads,
            layout.value_head_dim,
        )?;
        add_elapsed(
            &mut self.stages.activation.other_host_activation_secs,
            &mut self.stages.activation.other_host_activation_ns,
            gated_started,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(1);
        let mut mixer_output = vec![0.0f32; QWEN80_HIDDEN];
        self.matvec_named(
            &Self::layer_name(layer, "linear_attn.out_proj.weight"),
            &gated_output,
            &mut mixer_output,
        )?;
        let mut residual = hidden.to_vec();
        let add_started = Instant::now();
        add_inplace(&mut residual, &mixer_output);
        add_elapsed(
            &mut self.stages.activation.other_host_activation_secs,
            &mut self.stages.activation.other_host_activation_ns,
            add_started,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(1);
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
        let rms_started = Instant::now();
        let rms = source_qwen80_residual_rms_norm(hidden, &input_w)?;
        add_elapsed(
            &mut self.stages.activation.gqa_input_layernorm_secs,
            &mut self.stages.activation.gqa_input_layernorm_ns,
            rms_started,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.gqa_input_layernorm =
            self.activation_counts.gqa_input_layernorm.saturating_add(1);
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
        let rope_started = Instant::now();
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
        add_elapsed(
            &mut self.stages.activation.gqa_norm_rope_secs,
            &mut self.stages.activation.gqa_norm_rope_ns,
            rope_started,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(2);
        self.activation_counts.gqa_norm_rope =
            self.activation_counts.gqa_norm_rope.saturating_add(2);
        let start = position * layout.kv_dim;
        let end = start + layout.kv_dim;
        self.state.gqa_key[slot][start..end].copy_from_slice(&key_row);
        self.state.gqa_value[slot][start..end].copy_from_slice(&v_projection);
        let attn_started = Instant::now();
        let attention = qwen80_gqa_causal_attention(
            &query,
            &self.state.gqa_key[slot],
            &self.state.gqa_value[slot],
            position + 1,
            &layout,
        )?;
        let gated = qwen80_gqa_apply_sigmoid_gate(&attention, &q_projection, &layout)?;
        add_elapsed(
            &mut self.stages.activation.other_host_activation_secs,
            &mut self.stages.activation.other_host_activation_ns,
            attn_started,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(2);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(2);
        let mut mixer_output = vec![0.0f32; QWEN80_HIDDEN];
        self.matvec_named(
            &Self::layer_name(layer, "self_attn.o_proj.weight"),
            &gated,
            &mut mixer_output,
        )?;
        let mut residual = hidden.to_vec();
        let add_started = Instant::now();
        add_inplace(&mut residual, &mixer_output);
        add_elapsed(
            &mut self.stages.activation.other_host_activation_secs,
            &mut self.stages.activation.other_host_activation_ns,
            add_started,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(1);
        if residual.iter().any(|value| !value.is_finite()) {
            return Err(q80q4_error(format!(
                "layer {layer} GQA residual is non-finite"
            )));
        }
        Ok(residual)
    }

    fn moe_suffix(&mut self, layer: usize, first_residual: &[f32]) -> Result<Vec<f32>> {
        let norm_started = Instant::now();
        let post_w = self.vector(&Self::layer_name(layer, "post_attention_layernorm.weight"))?;
        let norm_op = Instant::now();
        let router_input = source_qwen80_residual_rms_norm(first_residual, &post_w)?;
        add_elapsed(
            &mut self.stages.activation.other_host_activation_secs,
            &mut self.stages.activation.other_host_activation_ns,
            norm_op,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(1);
        add_elapsed(
            &mut self.stages.moe_norm_router_secs,
            &mut self.stages.moe_norm_router_ns,
            norm_started,
        );

        let shared_started = Instant::now();
        let shared = self.mlp(
            &Self::layer_name(layer, "mlp.shared_expert.gate_proj.weight"),
            &Self::layer_name(layer, "mlp.shared_expert.up_proj.weight"),
            &Self::layer_name(layer, "mlp.shared_expert.down_proj.weight"),
            &router_input,
            QWEN80_MOE_INTERMEDIATE,
        )?;
        add_elapsed(
            &mut self.stages.moe_shared_secs,
            &mut self.stages.moe_shared_ns,
            shared_started,
        );

        let router_started = Instant::now();
        let mut router_logits = vec![0.0f32; QWEN80_EXPERTS];
        self.matvec_named(
            &Self::layer_name(layer, "mlp.gate.weight"),
            &router_input,
            &mut router_logits,
        )?;
        let route_op = Instant::now();
        let route = source_qwen80_topk_router(&router_logits)?;
        add_elapsed(
            &mut self.stages.activation.other_host_activation_secs,
            &mut self.stages.activation.other_host_activation_ns,
            route_op,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(1);
        add_elapsed(
            &mut self.stages.moe_norm_router_secs,
            &mut self.stages.moe_norm_router_ns,
            router_started,
        );

        let mut combined = vec![0.0f32; QWEN80_HIDDEN];
        let used_device_table =
            self.try_device_expert_table(layer, &route, &router_input, &mut combined)?;
        if !used_device_table {
            let bind_started = Instant::now();
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
                let expert_out =
                    self.mlp(&gate, &up, &down, &router_input, QWEN80_MOE_INTERMEDIATE)?;
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
            add_elapsed(
                &mut self.stages.host_expert_bind_secs,
                &mut self.stages.host_expert_bind_ns,
                bind_started,
            );
            add_elapsed(
                &mut self.stages.moe_routed_secs,
                &mut self.stages.moe_routed_ns,
                bind_started,
            );
        }

        let combine_started = Instant::now();
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
        let combine_op = Instant::now();
        for (dst, value) in combined.iter_mut().zip(shared) {
            *dst += value * gate_val;
        }
        let mut out = first_residual.to_vec();
        add_inplace(&mut out, &combined);
        add_elapsed(
            &mut self.stages.activation.other_host_activation_secs,
            &mut self.stages.activation.other_host_activation_ns,
            combine_op,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(1);
        add_elapsed(
            &mut self.stages.moe_combine_secs,
            &mut self.stages.moe_combine_ns,
            combine_started,
        );
        if out.iter().any(|value| !value.is_finite()) {
            return Err(q80q4_error(format!(
                "layer {layer} second residual is non-finite"
            )));
        }
        Ok(out)
    }

    fn try_device_expert_table(
        &mut self,
        layer: usize,
        route: &super::qwen80_complete_runtime::Qwen80RouteSelection,
        router_input: &[f32],
        combined: &mut [f32],
    ) -> Result<bool> {
        if !super::qwen80_device_expert_table::qwen80_device_expert_table_enabled() {
            return Ok(false);
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (layer, route, router_input, combined);
            return Ok(false);
        }
        #[cfg(target_os = "macos")]
        {
            if self.metal.is_none() {
                return Ok(false);
            }
            let mut route_ids = [0u32; 10];
            let mut route_weights = [0.0f32; 10];
            for (index, (&expert, &weight)) in
                route.ids.iter().zip(route.weights.iter()).enumerate()
            {
                route_ids[index] = expert as u32;
                route_weights[index] = weight;
            }
            let Some(metal) = self.metal.as_mut() else {
                return Ok(false);
            };
            match metal.routed_expert_table(
                &self.catalog,
                layer,
                &route_ids,
                &route_weights,
                router_input,
                combined,
                &mut self.native,
                &mut self.stages,
            ) {
                Ok(()) => Ok(true),
                Err(error) => {
                    if self.metal_error.is_none() {
                        self.metal_error = Some(error.to_string());
                    }
                    Ok(false)
                }
            }
        }
    }

    fn terminal_greedy(&mut self, hidden: &[f32]) -> Result<u32> {
        let norm_w = self.vector("model.norm.weight")?;
        let norm_op = Instant::now();
        let normed = source_qwen80_residual_rms_norm(hidden, &norm_w)?;
        add_elapsed(
            &mut self.stages.activation.other_host_activation_secs,
            &mut self.stages.activation.other_host_activation_ns,
            norm_op,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(1);
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

    #[cfg(target_os = "macos")]
    fn device_activations_live(&self) -> bool {
        self.metal
            .as_ref()
            .and_then(|metal| metal.activations.as_ref())
            .is_some()
    }

    #[cfg(target_os = "macos")]
    fn commit_profiled(
        &mut self,
        tcb: crate::metal::TokenCommandBuffer<'static>,
        name: &str,
        layer: Option<u32>,
        operator_classes: &[&str],
        force_sync_reason: &str,
        family: FamilyGpuKind,
    ) -> Result<crate::metal::CommandBufferTiming> {
        let timing = tcb.commit_and_wait_timed()?;
        self.record_family_timing(family, &timing);
        self.token_ns.record_cb(
            name,
            layer,
            operator_classes,
            timing,
            force_sync_reason,
        );
        Ok(timing)
    }

    #[cfg(target_os = "macos")]
    fn commit_submit_retain_profiled(
        &mut self,
        tcb: crate::metal::TokenCommandBuffer<'static>,
    ) -> Result<crate::metal::SubmittedTokenCommandBuffer> {
        tcb.commit_submit_retain()
    }

    #[cfg(target_os = "macos")]
    fn resolve_submitted(
        &mut self,
        submitted: crate::metal::SubmittedTokenCommandBuffer,
        name: &str,
        layer: Option<u32>,
        operator_classes: &[&str],
        force_sync_reason: &str,
        family: FamilyGpuKind,
    ) -> crate::metal::CommandBufferTiming {
        let timing = submitted.resolve();
        self.record_family_timing(family, &timing);
        if self.token_ns.enabled {
            self.token_ns.record_cb(
                name,
                layer,
                operator_classes,
                timing,
                force_sync_reason,
            );
        }
        timing
    }

    #[cfg(target_os = "macos")]
    fn record_family_timing(
        &mut self,
        family: FamilyGpuKind,
        timing: &crate::metal::CommandBufferTiming,
    ) {
        self.family_gpu.note_cb(timing);
        let gpu = timing.gpu_ns.unwrap_or(0);
        match family {
            FamilyGpuKind::DeltaNetMixer { mixed } => {
                self.family_gpu.deltanet_mixer_cbs =
                    self.family_gpu.deltanet_mixer_cbs.saturating_add(1);
                self.family_gpu.deltanet_mixer_dispatches = self
                    .family_gpu
                    .deltanet_mixer_dispatches
                    .saturating_add(timing.dispatches);
                self.family_gpu.deltanet_mixer_gpu_ns =
                    self.family_gpu.deltanet_mixer_gpu_ns.saturating_add(gpu);
                self.family_gpu.deltanet_mixer_wait_ns = self
                    .family_gpu
                    .deltanet_mixer_wait_ns
                    .saturating_add(timing.wait_ns);
                self.family_gpu.deltanet_mixer_encode_ns = self
                    .family_gpu
                    .deltanet_mixer_encode_ns
                    .saturating_add(timing.encode_ns);
                self.family_gpu.deltanet_mixer_mixed_with_moe_prefix = mixed;
            }
            FamilyGpuKind::GqaMixer { mixed } => {
                self.family_gpu.gqa_mixer_cbs = self.family_gpu.gqa_mixer_cbs.saturating_add(1);
                self.family_gpu.gqa_mixer_dispatches = self
                    .family_gpu
                    .gqa_mixer_dispatches
                    .saturating_add(timing.dispatches);
                self.family_gpu.gqa_mixer_gpu_ns =
                    self.family_gpu.gqa_mixer_gpu_ns.saturating_add(gpu);
                self.family_gpu.gqa_mixer_wait_ns = self
                    .family_gpu
                    .gqa_mixer_wait_ns
                    .saturating_add(timing.wait_ns);
                self.family_gpu.gqa_mixer_encode_ns = self
                    .family_gpu
                    .gqa_mixer_encode_ns
                    .saturating_add(timing.encode_ns);
                self.family_gpu.gqa_mixer_mixed_with_moe_prefix = mixed;
            }
            FamilyGpuKind::MoePrefix => {
                self.family_gpu.moe_prefix_cbs = self.family_gpu.moe_prefix_cbs.saturating_add(1);
                self.family_gpu.moe_prefix_dispatches = self
                    .family_gpu
                    .moe_prefix_dispatches
                    .saturating_add(timing.dispatches);
                self.family_gpu.moe_prefix_gpu_ns =
                    self.family_gpu.moe_prefix_gpu_ns.saturating_add(gpu);
                self.family_gpu.moe_prefix_wait_ns = self
                    .family_gpu
                    .moe_prefix_wait_ns
                    .saturating_add(timing.wait_ns);
                self.family_gpu.moe_prefix_encode_ns = self
                    .family_gpu
                    .moe_prefix_encode_ns
                    .saturating_add(timing.encode_ns);
            }
            FamilyGpuKind::MoeSuffix => {
                self.family_gpu.moe_suffix_cbs = self.family_gpu.moe_suffix_cbs.saturating_add(1);
                self.family_gpu.moe_suffix_dispatches = self
                    .family_gpu
                    .moe_suffix_dispatches
                    .saturating_add(timing.dispatches);
                self.family_gpu.moe_suffix_gpu_ns =
                    self.family_gpu.moe_suffix_gpu_ns.saturating_add(gpu);
                self.family_gpu.moe_suffix_wait_ns = self
                    .family_gpu
                    .moe_suffix_wait_ns
                    .saturating_add(timing.wait_ns);
                self.family_gpu.moe_suffix_encode_ns = self
                    .family_gpu
                    .moe_suffix_encode_ns
                    .saturating_add(timing.encode_ns);
            }
            FamilyGpuKind::FusedLayer => {
                self.family_gpu.fused_layer_cbs =
                    self.family_gpu.fused_layer_cbs.saturating_add(1);
                self.family_gpu.fused_layer_dispatches = self
                    .family_gpu
                    .fused_layer_dispatches
                    .saturating_add(timing.dispatches);
                self.family_gpu.fused_layer_gpu_ns =
                    self.family_gpu.fused_layer_gpu_ns.saturating_add(gpu);
                self.family_gpu.fused_layer_wait_ns = self
                    .family_gpu
                    .fused_layer_wait_ns
                    .saturating_add(timing.wait_ns);
            }
            FamilyGpuKind::Other => {}
        }
    }

    #[cfg(target_os = "macos")]
    fn flush_buffer_stats(&mut self) {
        if let Some(metal) = self.metal.as_mut() {
            let created = metal.buffer_creations;
            let rebinds = metal.buffer_rebinds;
            metal.buffer_creations = 0;
            metal.buffer_rebinds = 0;
            self.token_ns.add_buffer_creations(created);
            self.token_ns.add_buffer_rebinds(rebinds);
        }
    }

    #[cfg(target_os = "macos")]
    fn new_token_cb(&self) -> Result<crate::metal::TokenCommandBuffer<'static>> {
        let metal = self
            .metal
            .as_ref()
            .ok_or_else(|| q80q4_error("device token requires Metal"))?;
        // SAFETY: the Metal context is owned by the session and outlives every
        // command buffer created here. Callers drop the TCB before dropping
        // `metal`.
        let ctx: &'static crate::metal::MetalContext =
            unsafe { &*(&metal.context as *const crate::metal::MetalContext) };
        Ok(crate::metal::TokenCommandBuffer::new(ctx))
    }

    #[cfg(target_os = "macos")]
    fn device_vector_buf(&mut self, name: &str) -> Result<crate::metal::PinnedBuffer> {
        if let Some(existing) = self
            .metal
            .as_ref()
            .and_then(|metal| metal.activations.as_ref())
            .and_then(|act| act.vectors.get(name))
            .cloned()
        {
            return Ok(existing);
        }
        let host = self.vector(name)?;
        let metal = self
            .metal
            .as_mut()
            .ok_or_else(|| q80q4_error("device vector upload requires Metal"))?;
        let buf = metal
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&host))?;
        metal.buffer_creations = metal.buffer_creations.saturating_add(1);
        if let Some(act) = metal.activations.as_mut() {
            act.vectors.insert(name.to_owned(), buf.clone());
        }
        Ok(buf)
    }

    #[cfg(target_os = "macos")]
    fn encode_q4_matvec(
        &mut self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        name: &str,
        input: &crate::metal::PinnedBuffer,
        output: &crate::metal::PinnedBuffer,
    ) -> Result<()> {
        let encode_started = Instant::now();
        let packed = self.cache.packed(&self.catalog, name)?.clone();
        let (rows, cols) = packed.rows_cols()?;
        let metal = self
            .metal
            .as_mut()
            .ok_or_else(|| q80q4_error("device matvec requires Metal"))?;
        metal.upload_weight(&packed)?;
        let codes = metal
            .weights
            .get(&packed.name)
            .expect("uploaded")
            .codes
            .clone();
        let scales = metal
            .weights
            .get(&packed.name)
            .expect("uploaded")
            .scales
            .clone();
        crate::kernels::qwen_uniform_q4_group64_matvec_component_tcb(
            tcb, &codes, &scales, input, output, rows, cols,
        )?;
        let bytes = packed
            .codes()
            .map(|c| c.len() as u64)
            .unwrap_or(0)
            .saturating_add(packed.scales().map(|s| s.len() as u64).unwrap_or(0));
        self.family_gpu.bytes_read_weight =
            self.family_gpu.bytes_read_weight.saturating_add(bytes);
        if self.token_ns.enabled {
            self.token_ns.add_weight_bytes(bytes);
        }
        self.native.q4_matvec_dispatches = self.native.q4_matvec_dispatches.saturating_add(1);
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        // Encode wall only — GPU time stays on the parent mixed CB.
        let encode_ns = encode_started.elapsed().as_nanos() as u64;
        self.token_ns
            .record_substage("q4_matvec", "encode", encode_ns, 1, 0);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn encode_residual_rmsnorm(
        &mut self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        input: &crate::metal::PinnedBuffer,
        weight_name: &str,
        output: &crate::metal::PinnedBuffer,
    ) -> Result<()> {
        let weight = self.device_vector_buf(weight_name)?;
        tcb.dispatch_threads(
            "qwen80_residual_rmsnorm_f32",
            (256, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(&weight), 0);
                encoder.set_buffer(2, Some(output), 0);
                let hidden = QWEN80_HIDDEN as u32;
                encoder.set_bytes(3, 4, &hidden as *const u32 as *const _);
                encoder.set_bytes(
                    4,
                    4,
                    &QWEN80_DEVICE_ACTIVATION_EPS as *const f32 as *const _,
                );
                encoder.set_threadgroup_memory_length(0, 256 * 4);
            },
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn encode_silu_mul(
        &mut self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        gate: &crate::metal::PinnedBuffer,
        up: &crate::metal::PinnedBuffer,
        output: &crate::metal::PinnedBuffer,
        n: u32,
    ) -> Result<()> {
        let started = Instant::now();
        tcb.dispatch_threads(
            "qwen80_silu_mul_f32",
            (n, 1, 1),
            (n.min(256).max(1), 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(gate), 0);
                encoder.set_buffer(1, Some(up), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.set_bytes(3, 4, &n as *const u32 as *const _);
            },
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        add_elapsed(
            &mut self.stages.activation.shared_swiglu_secs,
            &mut self.stages.activation.shared_swiglu_ns,
            started,
        );
        self.token_ns.record_substage(
            "activation",
            "shared_swiglu_encode",
            started.elapsed().as_nanos() as u64,
            1,
            0,
        );
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn encode_shared_mlp(
        &mut self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        layer: usize,
        input: &crate::metal::PinnedBuffer,
    ) -> Result<()> {
        let sandwich = Instant::now();
        let (gate, up, act, shared) = {
            let actw = self
                .metal
                .as_ref()
                .and_then(|m| m.activations.as_ref())
                .ok_or_else(|| q80q4_error("device activations missing"))?;
            (
                actw.gate.clone(),
                actw.up.clone(),
                actw.act.clone(),
                actw.shared.clone(),
            )
        };
        self.encode_q4_matvec(
            tcb,
            &Self::layer_name(layer, "mlp.shared_expert.gate_proj.weight"),
            input,
            &gate,
        )?;
        self.encode_q4_matvec(
            tcb,
            &Self::layer_name(layer, "mlp.shared_expert.up_proj.weight"),
            input,
            &up,
        )?;
        self.encode_silu_mul(tcb, &gate, &up, &act, QWEN80_MOE_INTERMEDIATE as u32)?;
        self.encode_q4_matvec(
            tcb,
            &Self::layer_name(layer, "mlp.shared_expert.down_proj.weight"),
            &act,
            &shared,
        )?;
        add_elapsed(
            &mut self.stages.activation.shared_mlp_sandwich_secs,
            &mut self.stages.activation.shared_mlp_sandwich_ns,
            sandwich,
        );
        self.token_ns.record_substage(
            "activation",
            "shared_mlp_sandwich_encode",
            sandwich.elapsed().as_nanos() as u64,
            1,
            0,
        );
        self.token_ns.record_substage(
            "moe_shared",
            "encode",
            sandwich.elapsed().as_nanos() as u64,
            1,
            0,
        );
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn encode_deltanet_mixer(
        &mut self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        layer: usize,
        hidden: &crate::metal::PinnedBuffer,
    ) -> Result<()> {
        let layout = Qwen80CanonicalLinearDeltaNetLayout::source_exact();
        let slot = self.state.linear_slot_for_layer(layer)?;
        let conv_off = (slot * layout.conv_state_elements()? * std::mem::size_of::<f32>()) as u64;
        let rec_off_elems = slot * layout.recurrent_state_elements()?;
        let (
            normalized,
            qkvz,
            ba,
            repeated_q,
            repeated_k,
            conv_v,
            z,
            decay,
            beta,
            rec_out,
            gated,
            mixer,
            first_residual,
            conv_state,
            rec_state,
        ) = {
            let actw = self
                .metal
                .as_ref()
                .and_then(|m| m.activations.as_ref())
                .ok_or_else(|| q80q4_error("device activations missing"))?;
            (
                actw.normalized.clone(),
                actw.qkvz.clone(),
                actw.ba.clone(),
                actw.repeated_q.clone(),
                actw.repeated_k.clone(),
                actw.conv_v.clone(),
                actw.z.clone(),
                actw.decay.clone(),
                actw.beta.clone(),
                actw.rec_out.clone(),
                actw.gated.clone(),
                actw.mixer.clone(),
                actw.first_residual.clone(),
                actw.linear_conv.clone(),
                actw.linear_recurrent.clone(),
            )
        };
        if qwen80_serial_mixer_enabled() {
            tcb.begin_serial_group()?;
        }
        self.encode_residual_rmsnorm(
            tcb,
            hidden,
            &Self::layer_name(layer, "input_layernorm.weight"),
            &normalized,
        )?;
        self.encode_q4_matvec(
            tcb,
            &Self::layer_name(layer, "linear_attn.in_proj_qkvz.weight"),
            &normalized,
            &qkvz,
        )?;
        self.encode_q4_matvec(
            tcb,
            &Self::layer_name(layer, "linear_attn.in_proj_ba.weight"),
            &normalized,
            &ba,
        )?;
        let conv_w =
            self.device_vector_buf(&Self::layer_name(layer, "linear_attn.conv1d.weight"))?;
        let conv_started = Instant::now();
        tcb.dispatch_threads(
            "qwen80_qkvz_rearrange_conv_l2_f32",
            (256, layout.key_heads as u32, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&qkvz), 0);
                encoder.set_buffer(1, Some(&conv_w), 0);
                encoder.set_buffer(2, Some(&conv_state), conv_off);
                encoder.set_buffer(3, Some(&repeated_q), 0);
                encoder.set_buffer(4, Some(&repeated_k), 0);
                encoder.set_buffer(5, Some(&conv_v), 0);
                encoder.set_buffer(6, Some(&z), 0);
                let kh = layout.key_heads as u32;
                let vpk = layout.value_heads_per_key_head as u32;
                let kd = layout.key_head_dim as u32;
                let vd = layout.value_head_dim as u32;
                let ck = layout.conv_kernel as u32;
                encoder.set_bytes(7, 4, &kh as *const u32 as *const _);
                encoder.set_bytes(8, 4, &vpk as *const u32 as *const _);
                encoder.set_bytes(9, 4, &kd as *const u32 as *const _);
                encoder.set_bytes(10, 4, &vd as *const u32 as *const _);
                encoder.set_bytes(11, 4, &ck as *const u32 as *const _);
                encoder.set_bytes(
                    12,
                    4,
                    &QWEN80_DEVICE_ACTIVATION_EPS as *const f32 as *const _,
                );
                encoder.set_threadgroup_memory_length(0, 4 * 256 * 4);
            },
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        add_elapsed(
            &mut self.stages.activation.deltanet_conv_secs,
            &mut self.stages.activation.deltanet_conv_ns,
            conv_started,
        );
        self.token_ns.record_substage(
            "activation",
            "deltanet_conv_encode",
            conv_started.elapsed().as_nanos() as u64,
            1,
            0,
        );
        self.activation_counts.deltanet_conv =
            self.activation_counts.deltanet_conv.saturating_add(1);
        let a_log = self.device_vector_buf(&Self::layer_name(layer, "linear_attn.A_log"))?;
        let dt_bias = self.device_vector_buf(&Self::layer_name(layer, "linear_attn.dt_bias"))?;
        let recurrent_started = Instant::now();
        tcb.dispatch_threads(
            "qwen80_ba_to_decay_beta_f32",
            (layout.value_heads as u32, 1, 1),
            (layout.value_heads.min(32).max(1) as u32, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&ba), 0);
                encoder.set_buffer(1, Some(&a_log), 0);
                encoder.set_buffer(2, Some(&dt_bias), 0);
                encoder.set_buffer(3, Some(&decay), 0);
                encoder.set_buffer(4, Some(&beta), 0);
                let kh = layout.key_heads as u32;
                let vpk = layout.value_heads_per_key_head as u32;
                encoder.set_bytes(5, 4, &kh as *const u32 as *const _);
                encoder.set_bytes(6, 4, &vpk as *const u32 as *const _);
            },
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        let rec_off_bytes = (rec_off_elems * std::mem::size_of::<f32>()) as u64;
        tcb.dispatch_threads(
            "qwen80_gated_delta_decode_tg",
            (layout.key_head_dim as u32, layout.value_heads as u32, 1),
            (layout.key_head_dim as u32, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&rec_state), rec_off_bytes);
                encoder.set_buffer(1, Some(&repeated_q), 0);
                encoder.set_buffer(2, Some(&repeated_k), 0);
                encoder.set_buffer(3, Some(&conv_v), 0);
                encoder.set_buffer(4, Some(&decay), 0);
                encoder.set_buffer(5, Some(&beta), 0);
                encoder.set_buffer(6, Some(&rec_out), 0);
                let heads = layout.value_heads as u32;
                let kd = layout.key_head_dim as u32;
                let vd = layout.value_head_dim as u32;
                encoder.set_bytes(7, 4, &heads as *const u32 as *const _);
                encoder.set_bytes(8, 4, &kd as *const u32 as *const _);
                encoder.set_bytes(9, 4, &vd as *const u32 as *const _);
                encoder.set_threadgroup_memory_length(0, 128 * 4);
            },
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        add_elapsed(
            &mut self.stages.activation.deltanet_recurrent_secs,
            &mut self.stages.activation.deltanet_recurrent_ns,
            recurrent_started,
        );
        self.token_ns.record_substage(
            "activation",
            "deltanet_recurrent_encode",
            recurrent_started.elapsed().as_nanos() as u64,
            1,
            0,
        );
        self.activation_counts.deltanet_recurrent = self
            .activation_counts
            .deltanet_recurrent
            .saturating_add(1);
        let norm_w = self.device_vector_buf(&Self::layer_name(layer, "linear_attn.norm.weight"))?;
        tcb.dispatch_threads(
            "qwen80_deltanet_gated_rmsnorm_f32",
            (layout.value_heads as u32, 1, 1),
            (layout.value_heads.min(32).max(1) as u32, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&rec_out), 0);
                encoder.set_buffer(1, Some(&z), 0);
                encoder.set_buffer(2, Some(&norm_w), 0);
                encoder.set_buffer(3, Some(&gated), 0);
                let heads = layout.value_heads as u32;
                let dim = layout.value_head_dim as u32;
                encoder.set_bytes(4, 4, &heads as *const u32 as *const _);
                encoder.set_bytes(5, 4, &dim as *const u32 as *const _);
                encoder.set_bytes(
                    6,
                    4,
                    &QWEN80_DEVICE_ACTIVATION_EPS as *const f32 as *const _,
                );
            },
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        self.encode_q4_matvec(
            tcb,
            &Self::layer_name(layer, "linear_attn.out_proj.weight"),
            &gated,
            &mixer,
        )?;
        crate::kernels::qwen_next_add_residual_tcb(
            tcb,
            hidden,
            &mixer,
            &first_residual,
            QWEN80_HIDDEN,
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        if qwen80_serial_mixer_enabled() {
            tcb.end_concurrent_group()?;
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn encode_gqa_mixer(
        &mut self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        layer: usize,
        hidden: &crate::metal::PinnedBuffer,
    ) -> Result<()> {
        let layout = Qwen80CanonicalGqaLayout::source_exact();
        let slot = self.state.gqa_slot_for_layer(layer)?;
        let position = self.state.position;
        if position >= self.state.max_seq_len {
            return Err(q80q4_error(format!(
                "GQA position {position} exceeds max_seq_len {}",
                self.state.max_seq_len
            )));
        }
        let slot_elems = self.state.max_seq_len * layout.kv_dim;
        let cache_off = (slot * slot_elems * std::mem::size_of::<f32>()) as u64;
        let (
            normalized,
            q_proj,
            k_proj,
            v_proj,
            query,
            attn,
            gated_attn,
            mixer,
            first_residual,
            gqa_key,
            gqa_value,
        ) = {
            let actw = self
                .metal
                .as_ref()
                .and_then(|m| m.activations.as_ref())
                .ok_or_else(|| q80q4_error("device activations missing"))?;
            (
                actw.normalized.clone(),
                actw.q_proj.clone(),
                actw.k_proj.clone(),
                actw.v_proj.clone(),
                actw.query.clone(),
                actw.attn.clone(),
                actw.gated_attn.clone(),
                actw.mixer.clone(),
                actw.first_residual.clone(),
                actw.gqa_key.clone(),
                actw.gqa_value.clone(),
            )
        };
        if qwen80_serial_mixer_enabled() {
            tcb.begin_serial_group()?;
        }
        let ln_started = Instant::now();
        self.encode_residual_rmsnorm(
            tcb,
            hidden,
            &Self::layer_name(layer, "input_layernorm.weight"),
            &normalized,
        )?;
        add_elapsed(
            &mut self.stages.activation.gqa_input_layernorm_secs,
            &mut self.stages.activation.gqa_input_layernorm_ns,
            ln_started,
        );
        self.token_ns.record_substage(
            "activation",
            "gqa_input_layernorm_encode",
            ln_started.elapsed().as_nanos() as u64,
            1,
            0,
        );
        self.activation_counts.gqa_input_layernorm = self
            .activation_counts
            .gqa_input_layernorm
            .saturating_add(1);
        self.encode_q4_matvec(
            tcb,
            &Self::layer_name(layer, "self_attn.q_proj.weight"),
            &normalized,
            &q_proj,
        )?;
        self.encode_q4_matvec(
            tcb,
            &Self::layer_name(layer, "self_attn.k_proj.weight"),
            &normalized,
            &k_proj,
        )?;
        self.encode_q4_matvec(
            tcb,
            &Self::layer_name(layer, "self_attn.v_proj.weight"),
            &normalized,
            &v_proj,
        )?;
        let q_norm = self.device_vector_buf(&Self::layer_name(layer, "self_attn.q_norm.weight"))?;
        let k_norm = self.device_vector_buf(&Self::layer_name(layer, "self_attn.k_norm.weight"))?;
        let rope_started = Instant::now();
        tcb.dispatch_threads(
            "qwen80_gqa_qk_norm_rope_cache_f32",
            (layout.query_heads as u32, 1, 1),
            (layout.query_heads.min(16).max(1) as u32, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&q_proj), 0);
                encoder.set_buffer(1, Some(&k_proj), 0);
                encoder.set_buffer(2, Some(&v_proj), 0);
                encoder.set_buffer(3, Some(&q_norm), 0);
                encoder.set_buffer(4, Some(&k_norm), 0);
                encoder.set_buffer(5, Some(&query), 0);
                encoder.set_buffer(6, Some(&gqa_key), cache_off);
                encoder.set_buffer(7, Some(&gqa_value), cache_off);
                let pos = position as u32;
                let nh = layout.query_heads as u32;
                let nkv = layout.key_value_heads as u32;
                let hd = layout.head_dim as u32;
                let rd = layout.rotary_dim as u32;
                encoder.set_bytes(8, 4, &pos as *const u32 as *const _);
                encoder.set_bytes(9, 4, &nh as *const u32 as *const _);
                encoder.set_bytes(10, 4, &nkv as *const u32 as *const _);
                encoder.set_bytes(11, 4, &hd as *const u32 as *const _);
                encoder.set_bytes(12, 4, &rd as *const u32 as *const _);
                encoder.set_bytes(13, 4, &QWEN80_DEVICE_ROPE_THETA as *const f32 as *const _);
                encoder.set_bytes(
                    14,
                    4,
                    &QWEN80_DEVICE_ACTIVATION_EPS as *const f32 as *const _,
                );
            },
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        add_elapsed(
            &mut self.stages.activation.gqa_norm_rope_secs,
            &mut self.stages.activation.gqa_norm_rope_ns,
            rope_started,
        );
        self.token_ns.record_substage(
            "activation",
            "gqa_norm_rope_encode",
            rope_started.elapsed().as_nanos() as u64,
            1,
            0,
        );
        self.activation_counts.gqa_norm_rope =
            self.activation_counts.gqa_norm_rope.saturating_add(1);
        crate::kernels::mha_decode_f32_tcb(
            tcb,
            &query,
            &gqa_key,
            cache_off as usize,
            &gqa_value,
            cache_off as usize,
            &attn,
            position + 1,
            layout.head_dim,
            layout.query_heads,
            layout.key_value_heads,
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        let query_dim = layout.query_dim as u32;
        let head_dim = layout.head_dim as u32;
        tcb.dispatch_threads(
            "qwen80_attention_apply_sigmoid_gate",
            (query_dim, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&attn), 0);
                encoder.set_buffer(1, Some(&q_proj), 0);
                encoder.set_buffer(2, Some(&gated_attn), 0);
                encoder.set_bytes(3, 4, &query_dim as *const u32 as *const _);
                encoder.set_bytes(4, 4, &head_dim as *const u32 as *const _);
            },
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        self.encode_q4_matvec(
            tcb,
            &Self::layer_name(layer, "self_attn.o_proj.weight"),
            &gated_attn,
            &mixer,
        )?;
        crate::kernels::qwen_next_add_residual_tcb(
            tcb,
            hidden,
            &mixer,
            &first_residual,
            QWEN80_HIDDEN,
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        if qwen80_serial_mixer_enabled() {
            tcb.end_concurrent_group()?;
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn encode_moe_prefix(
        &mut self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        layer: usize,
        postnorm: &crate::metal::PinnedBuffer,
    ) -> Result<()> {
        let (first_residual, router_logits) = {
            let actw = self
                .metal
                .as_ref()
                .and_then(|m| m.activations.as_ref())
                .ok_or_else(|| q80q4_error("device activations missing"))?;
            (actw.first_residual.clone(), actw.router_logits.clone())
        };
        if qwen80_serial_mixer_enabled() {
            tcb.begin_serial_group()?;
        }
        let norm_started = Instant::now();
        self.encode_residual_rmsnorm(
            tcb,
            &first_residual,
            &Self::layer_name(layer, "post_attention_layernorm.weight"),
            postnorm,
        )?;
        let norm_ns = norm_started.elapsed().as_nanos() as u64;
        self.encode_shared_mlp(tcb, layer, postnorm)?;
        let router_started = Instant::now();
        self.encode_q4_matvec(
            tcb,
            &Self::layer_name(layer, "mlp.gate.weight"),
            postnorm,
            &router_logits,
        )?;
        if qwen80_serial_mixer_enabled() {
            tcb.end_concurrent_group()?;
        }
        let router_ns = router_started.elapsed().as_nanos() as u64;
        // Encode wall only. GPU stays on the mixed prefix CB.
        self.token_ns.record_substage(
            "moe_norm_router",
            "encode",
            norm_ns.saturating_add(router_ns),
            1,
            0,
        );
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn encode_moe_combine(
        &mut self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        layer: usize,
        hidden_out: &crate::metal::PinnedBuffer,
        routed: &crate::metal::PinnedBuffer,
        postnorm: &crate::metal::PinnedBuffer,
    ) -> Result<()> {
        let (shared, shared_logit, gated_shared, first_residual) = {
            let actw = self
                .metal
                .as_ref()
                .and_then(|m| m.activations.as_ref())
                .ok_or_else(|| q80q4_error("device activations missing"))?;
            (
                actw.shared.clone(),
                actw.shared_logit.clone(),
                actw.gated_shared.clone(),
                actw.first_residual.clone(),
            )
        };
        if qwen80_serial_mixer_enabled() {
            tcb.begin_serial_group()?;
        }
        self.encode_q4_matvec(
            tcb,
            &Self::layer_name(layer, "mlp.shared_expert_gate.weight"),
            postnorm,
            &shared_logit,
        )?;
        crate::kernels::qwen_next_shared_expert_sigmoid_gate_tcb(
            tcb,
            &shared,
            &shared_logit,
            &gated_shared,
            QWEN80_HIDDEN,
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        crate::kernels::qwen_next_add_residual_tcb(
            tcb,
            routed,
            &gated_shared,
            &gated_shared,
            QWEN80_HIDDEN,
        )?;
        crate::kernels::qwen_next_add_residual_tcb(
            tcb,
            &first_residual,
            &gated_shared,
            hidden_out,
            QWEN80_HIDDEN,
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(2);
        self.family_gpu.moe_suffix_combine_dispatches = self
            .family_gpu
            .moe_suffix_combine_dispatches
            .saturating_add(4);
        if qwen80_serial_mixer_enabled() {
            tcb.end_concurrent_group()?;
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn encode_mixer_into(
        &mut self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        layer: usize,
        hidden: &crate::metal::PinnedBuffer,
    ) -> Result<(&'static str, FamilyGpuKind)> {
        match qwen80_layer_kind(layer)? {
            Qwen80LayerKind::LinearAttention => {
                self.encode_deltanet_mixer(tcb, layer, hidden)?;
                Ok((
                    "deltanet",
                    FamilyGpuKind::DeltaNetMixer { mixed: false },
                ))
            }
            Qwen80LayerKind::FullAttention => {
                self.encode_gqa_mixer(tcb, layer, hidden)?;
                Ok(("gqa", FamilyGpuKind::GqaMixer { mixed: false }))
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn host_route_and_bind(&mut self, layer: usize) -> Result<()> {
        let snap_started = Instant::now();
        let router_logits = {
            let actw = self
                .metal
                .as_ref()
                .and_then(|m| m.activations.as_ref())
                .ok_or_else(|| q80q4_error("device activations missing"))?;
            Self::snapshot_f32(&actw.router_logits, QWEN80_EXPERTS)?
        };
        let snap_ns = snap_started.elapsed().as_nanos() as u64;
        self.token_ns.record_sync(
            "router_logits_readback",
            Some(layer as u32),
            "prefix CB wait then memcpy 512 f32 logits; forces CPU top-k",
            snap_ns,
            (QWEN80_EXPERTS * std::mem::size_of::<f32>()) as u64,
            "device_to_host",
        );
        self.token_ns.close_phase("router_readback");
        let route_started = Instant::now();
        let route = source_qwen80_topk_router(&router_logits)?;
        self.token_ns.record_host_work(
            "host_topk_router",
            Some(layer as u32),
            route_started.elapsed().as_nanos() as u64,
            0,
            "softmax + top-10 + renormalize on 512 host logits",
        );
        self.token_ns.close_phase("host_topk");
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(1);

        let mut route_ids = [0u32; 10];
        let mut route_weights = [0.0f32; 10];
        for (index, (&expert, &weight)) in route.ids.iter().zip(route.weights.iter()).enumerate() {
            route_ids[index] = expert as u32;
            route_weights[index] = weight;
        }
        {
            let pack_started = Instant::now();
            let split_before = self.catalog.first_touch_split();
            let metal = self
                .metal
                .as_mut()
                .ok_or_else(|| q80q4_error("device token requires Metal"))?;
            metal.ensure_selected_expert_table(
                &self.catalog,
                layer,
                &route_ids,
                &mut self.native,
                &mut self.stages,
            )?;
            metal.remember_routes(layer, &route_ids);
            let split_after = self.catalog.first_touch_split();
            let read_ns = split_after
                .catalog_read_ns
                .saturating_sub(split_before.catalog_read_ns);
            let sha_ns = split_after.sha256_ns.saturating_sub(split_before.sha256_ns);
            let copy_ns = split_after
                .metal_copy_ns
                .saturating_sub(split_before.metal_copy_ns);
            let nocopy_ns = split_after
                .metal_nocopy_ns
                .saturating_sub(split_before.metal_nocopy_ns);
            if read_ns > 0 {
                self.token_ns.record_host_work(
                    "first_touch_catalog_read",
                    Some(layer as u32),
                    read_ns,
                    0,
                    "fs::read or mmap+advise of newly selected expert files",
                );
            }
            if sha_ns > 0 {
                self.token_ns.record_host_work(
                    "first_touch_sha256",
                    Some(layer as u32),
                    sha_ns,
                    0,
                    "payload SHA-256 (only when session is not sealed or rehash is forced)",
                );
            }
            if copy_ns > 0 {
                self.token_ns.record_host_work(
                    "first_touch_metal_copy",
                    Some(layer as u32),
                    copy_ns,
                    0,
                    "new_buffer_with_bytes of Q4 codes/scales",
                );
            }
            if nocopy_ns > 0 {
                self.token_ns.record_host_work(
                    "first_touch_metal_nocopy",
                    Some(layer as u32),
                    nocopy_ns,
                    0,
                    "newBufferWithBytesNoCopy of the page-aligned mmap window",
                );
            }
            let wave = metal
                .expert_wave
                .as_ref()
                .ok_or_else(|| q80q4_error("expert wave workspace missing"))?;
            let bind_ids = if metal.bind_real_route_ids() {
                route_ids
            } else {
                let mut remapped = [0u32; 10];
                for (slot, id) in remapped.iter_mut().enumerate() {
                    *id = slot as u32;
                }
                remapped
            };
            crate::metal::MetalContext::write_buffer_bytes(
                &wave.route_ids,
                bytemuck::cast_slice(&bind_ids),
            );
            crate::metal::MetalContext::write_buffer_bytes(
                &wave.route_weights,
                bytemuck::cast_slice(&route_weights),
            );
            let pack_bytes = if metal.residency.is_some() {
                (super::qwen80_device_expert_table::QWEN80_EXPERT_TABLE_TOP_K
                    * std::mem::size_of::<u32>()) as u64
            } else {
                (QWEN80_EXPERTS
                    * std::mem::size_of::<
                        super::qwen80_device_expert_table::Qwen80DeviceExpertTriplet,
                    >()) as u64
            };
            let bind_note = if metal.residency.is_some() {
                "token-dynamic: write top-10 route ids; kernel indirects the persistent all-layer address table"
            } else {
                "rewrite 512-entry gpuAddress table for the live top-10; payloads stay in cached triplets"
            };
            self.token_ns.record_host_work(
                "expert_address_table_bind",
                Some(layer as u32),
                pack_started.elapsed().as_nanos() as u64,
                pack_bytes,
                bind_note,
            );
        }
        self.token_ns.close_phase("moe_table_build");
        self.flush_buffer_stats();
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn encode_suffix_into(
        &mut self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        layer: usize,
        hidden: &crate::metal::PinnedBuffer,
        expert_input: &crate::metal::PinnedBuffer,
    ) -> Result<()> {
        let (routed, dispatches) = {
            let metal = self
                .metal
                .as_ref()
                .ok_or_else(|| q80q4_error("device token requires Metal"))?;
            let lease = metal
                .expert_table
                .as_ref()
                .ok_or_else(|| q80q4_error("expert table lease missing after ensure"))?;
            let wave = metal
                .expert_wave
                .as_ref()
                .ok_or_else(|| q80q4_error("expert wave workspace missing"))?;
            let dispatches =
                super::qwen80_device_expert_table::dispatch_qwen80_device_expert_table_tcb(
                    tcb,
                    lease,
                    wave,
                    metal.expert_kernel,
                )?;
            (wave.combined.clone(), dispatches)
        };
        self.native.expert_table_waves = self.native.expert_table_waves.saturating_add(1);
        self.native.expert_table_matvec_dispatches = self
            .native
            .expert_table_matvec_dispatches
            .saturating_add(dispatches);
        let expert_bytes = (super::qwen80_device_expert_table::QWEN80_EXPERT_TABLE_TOP_K
            * 3
            * (super::qwen80_device_expert_table::QWEN80_EXPERT_CODE_BYTES
                + super::qwen80_device_expert_table::QWEN80_EXPERT_SCALE_BYTES))
            as u64;
        self.family_gpu.bytes_read_weight = self
            .family_gpu
            .bytes_read_weight
            .saturating_add(expert_bytes);
        if self.token_ns.enabled {
            self.token_ns.add_weight_bytes(expert_bytes);
        }
        self.encode_moe_combine(tcb, layer, hidden, &routed, expert_input)
    }

    #[cfg(target_os = "macos")]
    fn submit_mixer(
        &mut self,
        layer: usize,
        hidden: &crate::metal::PinnedBuffer,
    ) -> Result<(
        crate::metal::SubmittedTokenCommandBuffer,
        &'static str,
        FamilyGpuKind,
    )> {
        crate::metal::set_current_layer(Some(layer as u32));
        let mut tcb = self.new_token_cb()?;
        let started = Instant::now();
        let (mixer_class, family) = self.encode_mixer_into(&mut tcb, layer, hidden)?;
        match mixer_class {
            "deltanet" => add_secs(&mut self.stages.deltanet_secs, started),
            _ => add_secs(&mut self.stages.gqa_secs, started),
        }
        let submitted = self.commit_submit_retain_profiled(tcb)?;
        Ok((submitted, mixer_class, family))
    }

    #[cfg(target_os = "macos")]
    fn wait_moe_prefix(
        &mut self,
        layer: usize,
        expert_input: &crate::metal::PinnedBuffer,
    ) -> Result<()> {
        crate::metal::set_current_layer(Some(layer as u32));
        let mut tcb = self.new_token_cb()?;
        let started = Instant::now();
        self.encode_moe_prefix(&mut tcb, layer, expert_input)?;
        self.commit_profiled(
            tcb,
            &format!("L{layer}.moe_prefix"),
            Some(layer as u32),
            &["shared_expert", "router", "norm"],
            "host top-10 router needs 512 router logits on the CPU",
            FamilyGpuKind::MoePrefix,
        )?;
        add_secs(&mut self.stages.moe_shared_secs, started);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn submit_suffix(
        &mut self,
        layer: usize,
        hidden: &crate::metal::PinnedBuffer,
        expert_input: &crate::metal::PinnedBuffer,
    ) -> Result<crate::metal::SubmittedTokenCommandBuffer> {
        crate::metal::set_current_layer(Some(layer as u32));
        let mut tcb = self.new_token_cb()?;
        let started = Instant::now();
        self.encode_suffix_into(&mut tcb, layer, hidden, expert_input)?;
        add_secs(&mut self.stages.moe_combine_secs, started);
        self.commit_submit_retain_profiled(tcb)
    }

    #[cfg(target_os = "macos")]
    fn forward_token_device_overlapped(
        &mut self,
        hidden: &crate::metal::PinnedBuffer,
        expert_input: &crate::metal::PinnedBuffer,
    ) -> Result<Option<crate::metal::SubmittedTokenCommandBuffer>> {
        let (mixer0, class0, family0) = self.submit_mixer(0, hidden)?;
        self.wait_moe_prefix(0, expert_input)?;
        self.resolve_submitted(
            mixer0,
            &format!("L0.mixer.{class0}"),
            Some(0),
            &[class0],
            "overlapped mixer; resolved after moe_prefix wait",
            family0,
        );
        self.host_route_and_bind(0)?;

        for layer in 0..(QWEN80_LAYERS - 1) {
            let next = layer + 1;
            let suffix = self.submit_suffix(layer, hidden, expert_input)?;
            let (mixer, class, family) = self.submit_mixer(next, hidden)?;
            self.wait_moe_prefix(next, expert_input)?;
            self.resolve_submitted(
                suffix,
                &format!("L{layer}.suffix"),
                Some(layer as u32),
                &["routed_expert", "combine"],
                "overlapped suffix; resolved after next moe_prefix wait",
                FamilyGpuKind::MoeSuffix,
            );
            self.resolve_submitted(
                mixer,
                &format!("L{next}.mixer.{class}"),
                Some(next as u32),
                &[class],
                "overlapped mixer; resolved after moe_prefix wait",
                family,
            );
            self.host_route_and_bind(next)?;
        }
        let last = QWEN80_LAYERS - 1;
        let suffix = self.submit_suffix(last, hidden, expert_input)?;
        Ok(Some(suffix))
    }

    #[cfg(target_os = "macos")]
    fn forward_token_device_fused(
        &mut self,
        hidden: &crate::metal::PinnedBuffer,
        expert_input: &crate::metal::PinnedBuffer,
    ) -> Result<Option<crate::metal::SubmittedTokenCommandBuffer>> {
        // Layer 0: mixer + moe_prefix, wait for router logits.
        crate::metal::set_current_layer(Some(0));
        let mut first = self.new_token_cb()?;
        let started = Instant::now();
        let (class0, family0) = self.encode_mixer_into(&mut first, 0, hidden)?;
        match class0 {
            "deltanet" => add_secs(&mut self.stages.deltanet_secs, started),
            _ => add_secs(&mut self.stages.gqa_secs, started),
        }
        let prefix_started = Instant::now();
        self.encode_moe_prefix(&mut first, 0, expert_input)?;
        add_secs(&mut self.stages.moe_shared_secs, prefix_started);
        self.commit_profiled(
            first,
            "L0.mixer+moe_prefix",
            Some(0),
            &[class0, "shared_expert", "router", "norm"],
            "host top-10 router needs 512 router logits on the CPU",
            match family0 {
                FamilyGpuKind::DeltaNetMixer { .. } => {
                    FamilyGpuKind::DeltaNetMixer { mixed: true }
                }
                FamilyGpuKind::GqaMixer { .. } => FamilyGpuKind::GqaMixer { mixed: true },
                other => other,
            },
        )?;
        self.host_route_and_bind(0)?;

        for layer in 0..(QWEN80_LAYERS - 1) {
            let next = layer + 1;
            crate::metal::set_current_layer(Some(next as u32));
            let mut fused = self.new_token_cb()?;
            let suffix_started = Instant::now();
            self.encode_suffix_into(&mut fused, layer, hidden, expert_input)?;
            add_secs(&mut self.stages.moe_combine_secs, suffix_started);
            let mix_started = Instant::now();
            let (class, _family) = self.encode_mixer_into(&mut fused, next, hidden)?;
            match class {
                "deltanet" => add_secs(&mut self.stages.deltanet_secs, mix_started),
                _ => add_secs(&mut self.stages.gqa_secs, mix_started),
            }
            let prefix_started = Instant::now();
            self.encode_moe_prefix(&mut fused, next, expert_input)?;
            add_secs(&mut self.stages.moe_shared_secs, prefix_started);
            self.commit_profiled(
                fused,
                &format!("L{layer}.suffix+L{next}.mixer+prefix"),
                Some(next as u32),
                &["routed_expert", "combine", class, "shared_expert", "router"],
                "fused suffix + next mixer + moe_prefix; wait only for next router logits",
                FamilyGpuKind::FusedLayer,
            )?;
            self.host_route_and_bind(next)?;
        }

        let last = QWEN80_LAYERS - 1;
        let suffix = self.submit_suffix(last, hidden, expert_input)?;
        Ok(Some(suffix))
    }

    #[cfg(target_os = "macos")]
    fn forward_token_device_serial_waits(
        &mut self,
        hidden: &crate::metal::PinnedBuffer,
        expert_input: &crate::metal::PinnedBuffer,
    ) -> Result<Option<crate::metal::SubmittedTokenCommandBuffer>> {
        for layer in 0..QWEN80_LAYERS {
            crate::metal::set_current_layer(Some(layer as u32));
            let mut prefix = self.new_token_cb()?;
            let prefix_started = Instant::now();
            let (mixer_class, family) = match qwen80_layer_kind(layer)? {
                Qwen80LayerKind::LinearAttention => {
                    self.encode_deltanet_mixer(&mut prefix, layer, hidden)?;
                    (
                        "deltanet",
                        FamilyGpuKind::DeltaNetMixer { mixed: true },
                    )
                }
                Qwen80LayerKind::FullAttention => {
                    self.encode_gqa_mixer(&mut prefix, layer, hidden)?;
                    ("gqa", FamilyGpuKind::GqaMixer { mixed: true })
                }
            };
            self.encode_moe_prefix(&mut prefix, layer, expert_input)?;
            self.commit_profiled(
                prefix,
                &format!("L{layer}.prefix"),
                Some(layer as u32),
                &[mixer_class, "shared_expert", "router", "norm"],
                "host top-10 router needs 512 router logits on the CPU",
                family,
            )?;
            match mixer_class {
                "deltanet" => add_elapsed(
                    &mut self.stages.deltanet_secs,
                    &mut self.stages.deltanet_ns,
                    prefix_started,
                ),
                _ => add_elapsed(
                    &mut self.stages.gqa_secs,
                    &mut self.stages.gqa_ns,
                    prefix_started,
                ),
            }
            self.token_ns.close_phase(if mixer_class == "deltanet" {
                "prefix_deltanet"
            } else {
                "prefix_gqa"
            });
            self.host_route_and_bind(layer)?;
            let mut suffix = self.new_token_cb()?;
            let suffix_started = Instant::now();
            self.encode_suffix_into(&mut suffix, layer, hidden, expert_input)?;
            let next_layer = layer.saturating_add(1);
            let can_prefetch = qwen80_expert_prefetch_enabled()
                && next_layer < QWEN80_LAYERS
                && self
                    .metal
                    .as_ref()
                    .is_some_and(|metal| metal.have_last_routes[next_layer])
                && !self.token_ns.enabled;
            if can_prefetch {
                suffix.commit_no_wait()?;
                let prefetch_started = Instant::now();
                let metal = self
                    .metal
                    .as_mut()
                    .ok_or_else(|| q80q4_error("device token requires Metal"))?;
                metal.prefetch_predicted_layer(&self.catalog, next_layer)?;
                metal.context.wait_idle()?;
                add_secs(&mut self.stages.prefetch_secs, prefetch_started);
            } else {
                self.commit_profiled(
                    suffix,
                    &format!("L{layer}.suffix"),
                    Some(layer as u32),
                    &["routed_expert", "combine"],
                    "next layer mixer reads the updated residual hidden",
                    FamilyGpuKind::MoeSuffix,
                )?;
            }
            add_elapsed(
                &mut self.stages.moe_combine_secs,
                &mut self.stages.moe_combine_ns,
                suffix_started,
            );
            self.token_ns.close_phase("suffix");
        }
        Ok(None)
    }

    #[cfg(target_os = "macos")]
    fn snapshot_f32(buf: &crate::metal::PinnedBuffer, n: usize) -> Result<Vec<f32>> {
        if buf.length() < (n * std::mem::size_of::<f32>()) as u64 {
            return Err(q80q4_error("device snapshot buffer is short"));
        }
        let slice = unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n) };
        Ok(slice.to_vec())
    }

    #[cfg(target_os = "macos")]
    fn forward_token_device(&mut self, token: u32) -> Result<u32> {
        let embed_started = Instant::now();
        let packed = self
            .cache
            .packed(&self.catalog, "model.embed_tokens.weight")?
            .clone();
        let hidden = {
            let metal = self
                .metal
                .as_mut()
                .ok_or_else(|| q80q4_error("device token requires Metal"))?;
            let hidden = metal
                .activations
                .as_ref()
                .ok_or_else(|| q80q4_error("device activations missing"))?
                .hidden
                .clone();
            let timing = metal.embed_into(&packed, token, &hidden, &mut self.native)?;
            let embed_bytes = packed
                .codes()
                .map(|c| c.len() as u64 / QWEN80_VOCAB as u64)
                .unwrap_or(0)
                .saturating_add(
                    packed
                        .scales()
                        .map(|s| s.len() as u64 / QWEN80_VOCAB as u64)
                        .unwrap_or(0),
                );
            self.token_ns.add_weight_bytes(embed_bytes);
            self.token_ns.record_cb(
                "embed",
                None,
                &["embed"],
                timing,
                "embedding gather must finish before layer 0 reads hidden",
            );
            hidden
        };
        add_elapsed(
            &mut self.stages.embed_secs,
            &mut self.stages.embed_ns,
            embed_started,
        );
        self.token_ns.close_phase("embed");
        self.flush_buffer_stats();
        let expert_input = {
            let metal = self
                .metal
                .as_ref()
                .ok_or_else(|| q80q4_error("device token requires Metal"))?;
            metal
                .expert_wave
                .as_ref()
                .ok_or_else(|| q80q4_error("expert wave workspace missing"))?
                .input
                .clone()
        };
        let pending_suffix = match qwen80_overlap_mode() {
            OverlapMode::Fuse => {
                self.forward_token_device_fused(&hidden, &expert_input)?
            }
            OverlapMode::Split => {
                self.forward_token_device_overlapped(&hidden, &expert_input)?
            }
            OverlapMode::Off => {
                self.forward_token_device_serial_waits(&hidden, &expert_input)?
            }
        };
        crate::metal::set_current_layer(None);

        let terminal_started = Instant::now();
        let (hidden_buf, norm_name) = (hidden, "model.norm.weight");
        let mut terminal = self.new_token_cb()?;
        let (normalized, logits_buf) = {
            let actw = self
                .metal
                .as_ref()
                .and_then(|m| m.activations.as_ref())
                .ok_or_else(|| q80q4_error("device activations missing"))?;
            (actw.normalized.clone(), actw.logits.clone())
        };
        self.encode_residual_rmsnorm(&mut terminal, &hidden_buf, norm_name, &normalized)?;
        self.encode_q4_matvec(&mut terminal, "lm_head.weight", &normalized, &logits_buf)?;
        let _terminal_encode_ns = terminal_started.elapsed().as_nanos() as u64;
        self.commit_profiled(
            terminal,
            "terminal",
            None,
            &["norm", "lm_head"],
            "host greedy argmax needs the full vocab logits",
            FamilyGpuKind::Other,
        )?;
        add_elapsed(
            &mut self.stages.terminal_secs,
            &mut self.stages.terminal_ns,
            terminal_started,
        );
        self.token_ns.close_phase("terminal");
        if let Some(suffix) = pending_suffix {
            self.resolve_submitted(
                suffix,
                &format!("L{}.suffix", QWEN80_LAYERS - 1),
                Some((QWEN80_LAYERS - 1) as u32),
                &["routed_expert", "combine"],
                "last suffix resolved after terminal wait",
                FamilyGpuKind::MoeSuffix,
            );
        }
        let snap_started = Instant::now();
        let logits = Self::snapshot_f32(&logits_buf, QWEN80_VOCAB)?;
        self.token_ns.record_sync(
            "lm_head_logits_readback",
            None,
            "terminal CB wait then memcpy vocab logits for host argmax",
            snap_started.elapsed().as_nanos() as u64,
            (QWEN80_VOCAB * std::mem::size_of::<f32>()) as u64,
            "device_to_host",
        );
        self.token_ns.close_phase("logits_readback");
        let sample_started = Instant::now();
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (index, &value) in logits.iter().take(QWEN80_TOKENIZER_VOCAB).enumerate() {
            if value > best_v || (value == best_v && index < best_i) {
                best_v = value;
                best_i = index;
            }
        }
        self.token_ns.record_host_work(
            "host_greedy_argmax",
            None,
            sample_started.elapsed().as_nanos() as u64,
            0,
            "lowest-id-wins greedy over tokenizer vocab",
        );
        self.token_ns.close_phase("host_argmax");
        self.flush_buffer_stats();
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
        #[cfg(target_os = "macos")]
        if self.device_activations_live() {
            let sampled = self.forward_token_device(token)?;
            self.state.position = self.state.position.saturating_add(1);
            require_rss_cap("after hybrid token")?;
            return Ok(sampled);
        }
        let embed_started = Instant::now();
        let mut hidden = self.embed(token)?;
        add_elapsed(
            &mut self.stages.embed_secs,
            &mut self.stages.embed_ns,
            embed_started,
        );
        self.token_ns.close_phase("embed");
        for layer in 0..QWEN80_LAYERS {
            let first = match qwen80_layer_kind(layer)? {
                Qwen80LayerKind::LinearAttention => {
                    let started = Instant::now();
                    let value = self.deltanet_mixer(layer, &hidden)?;
                    add_elapsed(
                        &mut self.stages.deltanet_secs,
                        &mut self.stages.deltanet_ns,
                        started,
                    );
                    self.token_ns.close_phase("prefix_deltanet");
                    value
                }
                Qwen80LayerKind::FullAttention => {
                    let started = Instant::now();
                    let value = self.gqa_mixer(layer, &hidden)?;
                    add_elapsed(
                        &mut self.stages.gqa_secs,
                        &mut self.stages.gqa_ns,
                        started,
                    );
                    self.token_ns.close_phase("prefix_gqa");
                    value
                }
            };
            hidden = self.moe_suffix(layer, &first)?;
            self.token_ns.close_phase("suffix");
        }
        let terminal_started = Instant::now();
        let sampled = self.terminal_greedy(&hidden)?;
        add_elapsed(
            &mut self.stages.terminal_secs,
            &mut self.stages.terminal_ns,
            terminal_started,
        );
        self.token_ns.close_phase("terminal");
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
    pub stages: Qwen80DecodeStageTimes,
    pub activation_counts: Qwen80ActivationClassCounts,
    pub family_gpu: Qwen80FamilyGpuTimes,
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
    for (index, &token) in prompt_token_ids.iter().enumerate() {
        let kind = if index + 1 == prompt_token_ids.len() {
            "first_generated"
        } else {
            "prefill"
        };
        session.token_ns.begin(kind, session.state.position as u32);
        next = session.forward_token(token)?;
        session.token_ns.end();
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
        session.token_ns.begin("decode", session.state.position as u32);
        next = session.forward_token(next)?;
        session.token_ns.end();
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
        stages: session.stages.clone(),
        activation_counts: session.activation_counts.clone(),
        family_gpu: session.family_gpu.clone(),
        complete_physical_bpw: session.catalog.complete_physical_bpw,
        claim: QWEN80_UNIFORM_Q4_VELOCITY_NOT_BASE_TRUE_TPS,
        metal_q4_matvec_used,
        metal_error: session.metal_error.clone(),
    })
}

/// Attribute catalog-read / SHA-256 / mmap on real expert files without
/// running a token. Used by the first-touch probe. Does not touch Metal.
pub fn probe_qwen80_expert_first_touch_io(
    catalog: &Qwen80UniformQ4StreamingCatalog,
    n_experts: usize,
) -> Result<Qwen80FirstTouchIoProbe> {
    let n_experts = n_experts.max(1);
    let mut report = Qwen80FirstTouchIoProbe {
        n_experts,
        ..Qwen80FirstTouchIoProbe::default()
    };
    for expert in 0..n_experts {
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let name = format!("model.layers.0.mlp.experts.{expert}.{proj}.weight");
            let row = catalog.require_row(&name)?;
            let read_started = Instant::now();
            let fresh = fs::read(&row.artifact_path).map_err(|error| {
                q80q4_error(format!("probe cannot read {name}: {error}"))
            })?;
            report.catalog_read_ns = report
                .catalog_read_ns
                .saturating_add(read_started.elapsed().as_nanos() as u64);
            report.bytes_read = report.bytes_read.saturating_add(fresh.len() as u64);
            let hash_started = Instant::now();
            let observed = sha256_hex(&fresh);
            report.sha256_ns = report
                .sha256_ns
                .saturating_add(hash_started.elapsed().as_nanos() as u64);
            if observed != row.artifact_sha256 {
                return Err(q80q4_error(format!(
                    "probe {name} sha256 {observed} != catalog {}",
                    row.artifact_sha256
                )));
            }
            let map_started = Instant::now();
            let mapped = catalog.map_payload(&name)?;
            report.mmap_ns = report
                .mmap_ns
                .saturating_add(map_started.elapsed().as_nanos() as u64);
            if mapped.payload() != fresh.as_slice() {
                return Err(q80q4_error(format!(
                    "probe {name}: mmap window is not bit-identical to a freshly-read payload"
                )));
            }
            report.compared_payloads = report.compared_payloads.saturating_add(1);
        }
    }
    Ok(report)
}

#[derive(Clone, Debug, Default)]
pub struct Qwen80FirstTouchIoProbe {
    pub n_experts: usize,
    pub compared_payloads: u64,
    pub bytes_read: u64,
    pub catalog_read_ns: u64,
    pub sha256_ns: u64,
    pub mmap_ns: u64,
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

/// Resolve the contract-relative tokenizer, then the main-repo copy.
pub fn discover_qwen80_tokenizer() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from(QWEN80_DEFAULT_TOKENIZER_REL),
        PathBuf::from(
            "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen-80b/Qwen3-Coder-Next/tokenizer.json",
        ),
    ];
    candidates.into_iter().find(|path| path.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    use std::sync::Mutex;
    use tempfile::TempDir;

    static CATALOG_DECODE_TEST: Mutex<()> = Mutex::new(());

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
    fn admit_then_mmap_matches_rehash_and_wrong_sha_still_raises() {
        let temp = TempDir::new().unwrap();
        let (manifest, name) = fixture_catalog(&temp);
        let mut catalog = Qwen80UniformQ4StreamingCatalog::open_manifest(&manifest).unwrap();
        let report = catalog.admit_session().unwrap();
        assert_eq!(report.tensors_stated, 1);
        assert!(!report.session_seal_sha256.is_empty());
        assert!(catalog.session_trusts_payloads());
        let mapped = catalog.map_payload(&name).unwrap();
        let hashed = catalog.read_payload_rehash(&name).unwrap();
        assert_eq!(
            mapped.payload(),
            hashed.as_ref(),
            "sealed mmap window must equal a freshly-hashed read"
        );
        assert_eq!(mapped.header.scale_offset % 16_384, mapped.header.scale_offset);
        assert_ne!(
            mapped.header.scale_offset % 16_384,
            0,
            "Q4 scales are not a 16KiB-aligned slice; the whole file is the no-copy window"
        );

        let row_path = catalog.require_row(&name).unwrap().artifact_path.clone();
        let original = fs::read(&row_path).unwrap();
        let mut tampered = original.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 0xff;
        fs::write(&row_path, &tampered).unwrap();
        let rehash = catalog.read_payload_rehash(&name).unwrap_err();
        let message = format!("{rehash}");
        assert!(
            message.contains("sha256"),
            "rehash must still catch a same-size tamper, got {message}"
        );
        fs::write(&row_path, original).unwrap();
    }

    #[test]
    fn unadmitted_catalog_still_hashes_on_read() {
        let temp = TempDir::new().unwrap();
        let (manifest, name) = fixture_catalog(&temp);
        let catalog = Qwen80UniformQ4StreamingCatalog::open_manifest(&manifest).unwrap();
        assert!(!catalog.session_trusts_payloads());
        let row_path = catalog.require_row(&name).unwrap().artifact_path.clone();
        let original = fs::read(&row_path).unwrap();
        let mut tampered = original.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 0xff;
        fs::write(&row_path, &tampered).unwrap();
        let err = catalog.read_payload(&name).unwrap_err();
        let message = format!("{err}");
        assert!(
            message.contains("sha256"),
            "token-path SHA must still fire before admission, got {message}"
        );
        fs::write(&row_path, original).unwrap();
    }

    #[test]
    fn real_expert_mmap_matches_rehash_when_catalog_present() {
        let Some(root) = discover_qwen80_uniform_q4_root() else {
            return;
        };
        let mut catalog = Qwen80UniformQ4StreamingCatalog::open(&root).unwrap();
        catalog.admit_session().unwrap();
        let name = "model.layers.0.mlp.experts.0.gate_proj.weight";
        let mapped = catalog.map_payload(name).unwrap();
        let hashed = catalog.read_payload_rehash(name).unwrap();
        assert_eq!(mapped.payload(), hashed.as_ref());
        assert_eq!(mapped.header.scale_offset, 40);
        assert_eq!(mapped.header.sign_offset, 40 + 32_768);
        assert_eq!(mapped.file_bytes, 557_096);
        assert_eq!(mapped.window().len() % 16_384, 0);
        assert_eq!((mapped.window().as_ptr() as usize) % 16_384, 0);
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
    fn component_matvec_kernel_is_occupancy_not_the_expert_table() {
        assert_eq!(
            QWEN80_COMPONENT_MATVEC_KERNEL,
            "qwen_uniform_q4_group64_matvec_vecgroup_x64"
        );
        assert!(
            !QWEN80_COMPONENT_MATVEC_KERNEL.contains("expert_table"),
            "component path must not dispatch the expert-table kernel"
        );
        assert_eq!(QWEN80_COMPONENT_MATVEC_ROWS_PER_TG, 4);
    }

    #[test]
    fn overlap_and_serial_mixer_default_off() {
        // Defaults are opt-in; paired 2026-08-16 reps did not justify
        // changing the production command-buffer topology.
        assert!(!qwen80_overlap_cbs_enabled());
        assert!(!qwen80_serial_mixer_enabled());
        assert!(matches!(qwen80_overlap_mode(), OverlapMode::Off));
        // Expert first-touch win stays on; A/B escapes are explicit disable.
        assert!(qwen80_expert_nocopy_enabled());
        assert!(qwen80_expert_prefetch_enabled());
        assert!(!qwen80_payload_rehash_enabled());
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

    #[test]
    fn greedy_baseline_prompt_yields_twelve_hello_tokens() {
        let _guard = CATALOG_DECODE_TEST
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(root) = discover_qwen80_uniform_q4_root() else {
            return;
        };
        let Some(tokenizer_path) = discover_qwen80_tokenizer() else {
            return;
        };
        let catalog = Qwen80UniformQ4StreamingCatalog::open(&root).unwrap();
        let tokenizer = load_qwen80_tokenizer(&tokenizer_path).unwrap();
        let mut session = Qwen80UniformQ4HybridDecodeSession::new(catalog, 64).unwrap();
        let prompt = render_qwen80_source_user_chat("Hi");
        let result = generate_greedy(&mut session, &tokenizer, &prompt, 12).unwrap();
        assert_eq!(
            result.generated_token_ids,
            vec![9707, 0, 2585, 646, 358, 1492, 498, 3351, 30, 26525, 232, 151645]
        );
        assert!(
            result.fallbacks.host_activation < 2000,
            "device-resident activations should drop host_activation well below the 8900 host baseline, got {}",
            result.fallbacks.host_activation
        );
        assert!(
            result.fallbacks.host_activation > 0,
            "host_activation must remain a real counter"
        );
    }

    #[test]
    fn device_decode_state_advance_differs_from_reset_and_repeated_same_token() {
        let _guard = CATALOG_DECODE_TEST
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(root) = discover_qwen80_uniform_q4_root() else {
            return;
        };
        let catalog = Qwen80UniformQ4StreamingCatalog::open(&root).unwrap();
        let mut session = Qwen80UniformQ4HybridDecodeSession::new(catalog, 16).unwrap();
        let token = 13048u32;
        let first = session.forward_token(token).unwrap();
        let second = session.forward_token(token).unwrap();
        session.reset_state();
        let reset_first = session.forward_token(token).unwrap();
        session.reset_state();
        let reset_second = session.forward_token(token).unwrap();
        assert_eq!(first, reset_first, "first step after reset must match");
        assert_eq!(
            reset_first, reset_second,
            "a reset between identical tokens must produce identical per-token output"
        );
        assert_ne!(
            second, reset_second,
            "a state reset between tokens must fail this comparison"
        );

        session.reset_state();
        let a = session.forward_token(3).unwrap();
        let b = session.forward_token(9).unwrap();
        session.reset_state();
        let c = session.forward_token(3).unwrap();
        let d = session.forward_token(3).unwrap();
        assert_eq!(a, c);
        assert_ne!(
            b, d,
            "decoding N distinct tokens must differ from the same token N times"
        );
    }

    #[test]
    fn greedy_baseline_prompt_yields_hello_how_tokens() {
        let _guard = CATALOG_DECODE_TEST
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(root) = discover_qwen80_uniform_q4_root() else {
            return;
        };
        let Some(tokenizer_path) = discover_qwen80_tokenizer() else {
            return;
        };
        let catalog = Qwen80UniformQ4StreamingCatalog::open(&root).unwrap();
        let tokenizer = load_qwen80_tokenizer(&tokenizer_path).unwrap();
        let mut session = Qwen80UniformQ4HybridDecodeSession::new(catalog, 64).unwrap();
        let prompt = render_qwen80_source_user_chat("Hi");
        let result = generate_greedy(&mut session, &tokenizer, &prompt, 3).unwrap();
        assert_eq!(result.generated_token_ids, vec![9707, 0, 2585]);
    }
}

#[derive(Clone, Debug)]
pub struct Qwen80KernelReloadReport {
    pub rebuilt_metal: bool,
    pub catalog_reopened: bool,
    pub elapsed_secs: f64,
    pub metal_error: Option<String>,
}

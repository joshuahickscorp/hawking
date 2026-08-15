//! Memory-bounded, restartable streamed BOS forward for DeepSeek-V4-Flash.
//!
//! This is a **host CPU** source-algorithm oracle. It is not a native Metal
//! runtime, not an Engine, not a serve path, and not a BASE_TRUE_TPS,
//! coherence, or tournament claim. Every tensor payload is read through the
//! admitted content-addressed reader (chunk SHA-256 + sealed byte length)
//! and is released after the operator that consumed it finishes.
//!
//! Layer schedule honesty (BOS / position 0 / seqlen 1):
//! - layers 0..=42 execute; MTP is excluded
//! - compressed indexer/compressor slots are empty at BOS, so window-only
//!   sparse attention is the source-correct path; the full ratio-4/128
//!   compressed graph is **not** implemented and is not claimed
//! - layers 0..=2 hash-route via `tid2eid`; layers 3..=42 use learned-bias
//!   `noaux_tc` top-6 (bias selects, unbiased scores weight)
//!
//! Peak *weight* residency is operator-streamed (one projection at a time),
//! which is strictly below the schedule's keep-layer-working-set ceiling of
//! 253_719_640 bytes at layer 2. Process RSS is measured at runtime and
//! includes the admitted reader's metadata.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use half::bf16;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, NativeScalePairKind, PINNED_REPOSITORY, PINNED_REVISION,
};
use crate::gravity_deepseek_v4_act_quant::{
    act_quant_bf16_ue8m0, decode_e4m3fn, decode_e8m0fnu, fp8_e4m3fn_ue8m0_matvec, ACT_QUANT_BLOCK,
};
use crate::gravity_deepseek_v4_final_head::{
    host_greedy_lm_head, host_merge_final_head_from_hc_bf16, DeepSeekV4GreedyTokenResult,
};
use crate::gravity_deepseek_v4_layer0_attention::{
    hc_attn_post_source_algorithm, kv_non_rope_inplace_qat_source_algorithm,
    per_head_rms_norm_bf16_source_algorithm, position_zero_rope_identity,
    rms_norm_bf16_source_algorithm, sparse_attention_position_zero_source_algorithm, HEAD_DIM,
    KV_QAT_BLOCK, NUM_HEADS, O_GROUPS, O_LORA_RANK, Q_LORA_RANK, ROPE_HEAD_DIM, WO_A_COLS,
    WO_A_ROWS, WO_B_COLS, WO_B_ROWS, WQ_B_ROWS, WKV_ROWS,
};
use crate::gravity_deepseek_v4_layer0_moe::{
    fp4_e2m1fn_x2_ue8m0_matvec, swiglu_bf16_source_algorithm, ACTIVATED_EXPERTS, FP4_BLOCK,
    MOE_INTER_DIM, ROUTE_SCALE, ROUTED_EXPERTS,
};
use crate::gravity_deepseek_v4_layer0_prefix::{
    hc_attn_pre_source_algorithm, HC_EPS, HC_FLAT_WIDTH, HC_MIX_WIDTH, HC_MULT, HC_SINKHORN_ITERS,
    HIDDEN_SIZE, PREFIX_TOKEN_ID, RMS_NORM_EPS,
};
use crate::gravity_deepseek_v4_layer_source_anchors::{
    verify_deepseek_v4_layer_source_anchors, DeepSeekV4LayerCommonTensor,
    DeepSeekV4LayerControlProjection, DeepSeekV4LayerExpertProjection, DeepSeekV4LayerGateMode,
    DeepSeekV4LayerMhcStage, DeepSeekV4LayerSourceAnchor,
    DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT,
};
use crate::{Error, Result};

/// Receipt / checkpoint schema for this host streamed oracle.
pub const STREAMED_FORWARD_SCHEMA: &str = "hawking.gravity.deepseek_v4.streamed_forward.v1";
/// Checkpoint schema. Distinct so a receipt cannot be loaded as a resume state.
pub const STREAMED_CHECKPOINT_SCHEMA: &str =
    "hawking.gravity.deepseek_v4.streamed_forward_checkpoint.v1";
/// Honest execution-path label. Never "native".
pub const STREAMED_EXECUTION_PATH: &str = "host_cpu_streamed_bos_oracle";
/// Schedule keep-layer-working-set peak (layer 2, includes indexer/compressor).
pub const SCHEDULE_STREAMED_DECODE_PEAK_BYTES: u64 = 253_719_640;
/// Declared bound on simultaneously resident weight bytes (operator-streamed).
pub const DECLARED_WEIGHT_RESIDENT_BOUND_BYTES: u64 = 384 * 1024 * 1024;
/// Declared bound on process peak RSS, including reader metadata. Far below
/// the 96 GiB host; fail-closed if a run crosses it.
pub const DECLARED_PEAK_RSS_BOUND_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const EMBED_WEIGHT: &str = "embed.weight";
const READ_BUFFER_BYTES: usize = 1024 * 1024;

/// Configuration for one streamed BOS decode.
#[derive(Debug, Clone)]
pub struct StreamedForwardConfig {
    pub max_layer: usize,
    pub token_id: u64,
    pub checkpoint_path: Option<PathBuf>,
    pub resume: bool,
    pub compute_final_head: bool,
    pub peak_rss_bound_bytes: u64,
    pub peak_weight_bound_bytes: u64,
}

impl Default for StreamedForwardConfig {
    fn default() -> Self {
        Self {
            max_layer: DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT - 1,
            token_id: PREFIX_TOKEN_ID,
            checkpoint_path: None,
            resume: false,
            compute_final_head: true,
            peak_rss_bound_bytes: DECLARED_PEAK_RSS_BOUND_BYTES,
            peak_weight_bound_bytes: DECLARED_WEIGHT_RESIDENT_BOUND_BYTES,
        }
    }
}

impl StreamedForwardConfig {
    pub fn for_layers(max_layer: usize) -> Result<Self> {
        if max_layer >= DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT {
            return Err(gravity(format!(
                "max_layer {max_layer} is outside the 0..42 base body"
            )));
        }
        Ok(Self {
            max_layer,
            ..Self::default()
        })
    }
}

/// Honesty labels recorded on every report. These exist so a later reader
/// cannot relabel a host oracle as native.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StreamedForwardHonesty {
    pub host_cpu: bool,
    pub native: bool,
    pub bos_window_only: bool,
    pub full_compressed_indexer_graph: bool,
    pub ratio_4_full_compressed_graph: bool,
    pub ratio_128_full_compressed_graph: bool,
    pub lm_head_path: String,
    pub metal_dispatches: usize,
    pub indexer_compressor_loaded: bool,
}

impl StreamedForwardHonesty {
    fn host_oracle(compute_final_head: bool) -> Self {
        Self {
            host_cpu: true,
            native: false,
            bos_window_only: true,
            full_compressed_indexer_graph: false,
            ratio_4_full_compressed_graph: false,
            ratio_128_full_compressed_graph: false,
            lm_head_path: if compute_final_head {
                "host_f64_mhc_merge_rmsnorm_then_host_streamed_lm_head_greedy".to_owned()
            } else {
                "omitted".to_owned()
            },
            metal_dispatches: 0,
            indexer_compressor_loaded: false,
        }
    }
}

/// Restartable HC residual plus bookkeeping.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StreamedForwardCheckpoint {
    pub schema: String,
    pub execution_path: String,
    pub native: bool,
    pub artifact_root: String,
    pub manifest_seal_sha256: String,
    pub manifest_file_sha256: String,
    pub restart_seal_sha256: String,
    pub repository: String,
    pub revision: String,
    pub token_id: u64,
    pub position: u64,
    pub next_layer: usize,
    pub layers_completed: Vec<usize>,
    pub hc_bf16_sha256: String,
    pub hc_bf16_hex: String,
    pub peak_rss_bytes: u64,
    pub peak_weight_resident_bytes: u64,
}

/// Result of a streamed run, including measured RSS.
#[derive(Debug, Clone)]
pub struct StreamedForwardReport {
    pub schema: &'static str,
    pub execution_path: &'static str,
    pub native: bool,
    pub deepest_layer: Option<usize>,
    pub layers_executed: Vec<usize>,
    pub token_id: u64,
    pub position: u64,
    pub hc_bf16_bits: Vec<u16>,
    pub hc_bf16_sha256: String,
    pub peak_rss_bytes: u64,
    pub peak_weight_resident_bytes: u64,
    pub declared_rss_bound_bytes: u64,
    pub declared_weight_bound_bytes: u64,
    pub rss_within_bound: bool,
    pub weight_within_bound: bool,
    pub greedy: Option<DeepSeekV4GreedyTokenResult>,
    pub stop_reason: Option<String>,
    pub honesty: StreamedForwardHonesty,
    pub artifact_root: String,
    pub admission_view: String,
    pub manifest_seal_sha256: String,
    pub wall_ms: u128,
}

impl StreamedForwardReport {
    pub fn to_receipt_json(&self) -> serde_json::Value {
        serde_json::json!({
            "schema": self.schema,
            "execution_path": self.execution_path,
            "native": self.native,
            "artifact": {
                "path": self.artifact_root,
                "admission_view": self.admission_view,
                "manifest_seal_sha256": self.manifest_seal_sha256,
                "repository": PINNED_REPOSITORY,
                "revision": PINNED_REVISION,
            },
            "scope": {
                "deepest_layer": self.deepest_layer,
                "layers_executed": self.layers_executed,
                "token_id": self.token_id,
                "token_position": self.position,
                "requested_max_layer": self.layers_executed.last().copied(),
            },
            "residency": {
                "peak_rss_bytes": self.peak_rss_bytes,
                "peak_weight_resident_bytes": self.peak_weight_resident_bytes,
                "declared_rss_bound_bytes": self.declared_rss_bound_bytes,
                "declared_weight_bound_bytes": self.declared_weight_bound_bytes,
                "rss_within_bound": self.rss_within_bound,
                "weight_within_bound": self.weight_within_bound,
                "schedule_streamed_decode_peak_bytes": SCHEDULE_STREAMED_DECODE_PEAK_BYTES,
                "policy": "operator_stream_load_execute_free",
            },
            "honesty": self.honesty,
            "hc_bf16_sha256": self.hc_bf16_sha256,
            "greedy": self.greedy.as_ref().map(|g| serde_json::json!({
                "token_id": g.token_id,
                "logit": g.logit,
                "vocab_size": g.vocab_size,
                "lm_head_on_device": g.lm_head_on_device,
                "argmax_on_device": g.argmax_on_device,
                "metal_dispatches": g.metal_dispatches,
            })),
            "stop_reason": self.stop_reason,
            "wall_ms": self.wall_ms,
        })
    }
}

/// Byte-accurate live-weight ledger. Tests exercise this directly; the
/// executor records every verified payload it holds.
#[derive(Debug, Default, Clone)]
pub struct ResidentLedger {
    live: BTreeMap<String, usize>,
    live_bytes: u64,
    peak_bytes: u64,
    bound_bytes: u64,
}

impl ResidentLedger {
    pub fn new(bound_bytes: u64) -> Self {
        Self {
            live: BTreeMap::new(),
            live_bytes: 0,
            peak_bytes: 0,
            bound_bytes,
        }
    }

    pub fn live_bytes(&self) -> u64 {
        self.live_bytes
    }

    pub fn peak_bytes(&self) -> u64 {
        self.peak_bytes
    }

    pub fn acquire(&mut self, name: &str, bytes: usize) -> Result<()> {
        if self.live.contains_key(name) {
            return Err(gravity(format!(
                "resident ledger already holds {name}; double-acquire is a stream bug"
            )));
        }
        let next = self
            .live_bytes
            .checked_add(bytes as u64)
            .ok_or_else(|| gravity("resident ledger byte count overflow"))?;
        if next > self.bound_bytes {
            return Err(gravity(format!(
                "streamed weight residency {next} exceeded declared bound {}",
                self.bound_bytes
            )));
        }
        self.live.insert(name.to_owned(), bytes);
        self.live_bytes = next;
        self.peak_bytes = self.peak_bytes.max(self.live_bytes);
        Ok(())
    }

    pub fn release(&mut self, name: &str) -> Result<()> {
        let bytes = self.live.remove(name).ok_or_else(|| {
            gravity(format!(
                "resident ledger release of unknown tensor {name}"
            ))
        })?;
        self.live_bytes = self.live_bytes.saturating_sub(bytes as u64);
        Ok(())
    }
}

/// Verify one content-addressed chunk file against a sealed digest and size.
/// Used by the integrity test and as an independent pre-check before a
/// reader-backed payload is accepted into the ledger.
pub fn verify_content_addressed_chunk(
    path: &Path,
    expected_sha256: &str,
    expected_bytes: u64,
) -> Result<()> {
    let meta = fs::metadata(path).map_err(|error| {
        gravity(format!(
            "chunk {} metadata: {error}",
            path.display()
        ))
    })?;
    if !meta.is_file() {
        return Err(gravity(format!("chunk {} is not a regular file", path.display())));
    }
    if meta.len() != expected_bytes {
        return Err(gravity(format!(
            "chunk {} byte size {} differs from sealed {expected_bytes}",
            path.display(),
            meta.len()
        )));
    }
    let mut file = File::open(path).map_err(|error| {
        gravity(format!("chunk {} open: {error}", path.display()))
    })?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; READ_BUFFER_BYTES];
    let mut observed = 0u64;
    loop {
        let got = file.read(&mut buffer)?;
        if got == 0 {
            break;
        }
        digest.update(&buffer[..got]);
        observed = observed
            .checked_add(got as u64)
            .ok_or_else(|| gravity("chunk byte count overflow"))?;
    }
    if observed != expected_bytes {
        return Err(gravity(format!(
            "chunk {} short read {observed} of {expected_bytes}",
            path.display()
        )));
    }
    let got = format!("{:x}", digest.finalize());
    if got != expected_sha256 {
        return Err(gravity(format!(
            "chunk {} sha256 differs from sealed segment digest",
            path.display()
        )));
    }
    Ok(())
}

/// Locate the sealed full-43-layer stream without writing under it.
pub fn discover_sealed_dsv4f_artifact() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("HAWKING_DSV4F_ARTIFACT") {
        let path = PathBuf::from(explicit);
        if artifact_looks_sealed(&path) {
            return Some(path);
        }
    }
    let mut candidates = Vec::new();
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let root = PathBuf::from(manifest_dir);
        candidates.push(
            root.join("../../workspace/campaign/records/runs/deepseek-v4/full-43-layer-stream.gravity"),
        );
        candidates.push(
            root.join("../../../Downloads/hawking/workspace/campaign/records/runs/deepseek-v4/full-43-layer-stream.gravity"),
        );
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(
            cwd.join("workspace/campaign/records/runs/deepseek-v4/full-43-layer-stream.gravity"),
        );
        if let Some(parent) = cwd.parent() {
            candidates.push(
                parent.join("workspace/campaign/records/runs/deepseek-v4/full-43-layer-stream.gravity"),
            );
        }
    }
    candidates.push(PathBuf::from(
        "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/deepseek-v4/full-43-layer-stream.gravity",
    ));
    candidates.into_iter().find(|path| artifact_looks_sealed(path))
}

fn artifact_looks_sealed(path: &Path) -> bool {
    path.join("manifest.json").is_file() && path.join("chunks").is_dir()
}

/// A readable admission root. Either the sealed directory itself or an
/// ephemeral hardlink view that restores the sealed range-journal prefix
/// after a later append to `stream-ranges.jsonl`. The view never writes
/// under the sealed artifact.
pub struct SealedAdmissionRoot {
    pub path: PathBuf,
    pub source_path: PathBuf,
    pub view: String,
    ephemeral: Option<PathBuf>,
}

impl Drop for SealedAdmissionRoot {
    fn drop(&mut self) {
        if let Some(path) = self.ephemeral.take() {
            let _ = fs::remove_dir_all(path);
        }
    }
}

/// Open the sealed stream, or a same-inode hardlink view whose
/// `stream-ranges.jsonl` is exactly the sealed prefix bound by the restart
/// receipt. Required because a later append doubled the on-disk journal
/// (139674 = 2 × 69837) and broke `DeepSeekV4FullStreamReader::admit`.
pub fn prepare_sealed_admission_root(source: impl AsRef<Path>) -> Result<SealedAdmissionRoot> {
    let source = source.as_ref();
    if !artifact_looks_sealed(source) {
        return Err(gravity(format!(
            "not a sealed full-43-layer stream: {}",
            source.display()
        )));
    }
    let restart_raw = fs::read(source.join("restart-receipt.json"))?;
    let restart: serde_json::Value = serde_json::from_slice(&restart_raw).map_err(|error| {
        gravity(format!("restart receipt decode failed: {error}"))
    })?;
    let expected_ranges = restart
        .get("range_journal_sha256")
        .and_then(|value| value.as_str())
        .ok_or_else(|| gravity("restart receipt lacks range_journal_sha256"))?;
    let expected_journal = restart
        .get("journal_sha256")
        .and_then(|value| value.as_str())
        .ok_or_else(|| gravity("restart receipt lacks journal_sha256"))?;
    let range_count = restart
        .get("range_count")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| gravity("restart receipt lacks range_count"))? as usize;

    let journal_path = source.join("stream-journal.json");
    let ranges_path = source.join("stream-ranges.jsonl");
    if sha256_file(&journal_path)? != expected_journal {
        return Err(gravity(
            "stream-journal.json no longer matches the sealed restart receipt",
        ));
    }
    let on_disk_ranges = sha256_file(&ranges_path)?;
    if on_disk_ranges == expected_ranges {
        return Ok(SealedAdmissionRoot {
            path: source.to_path_buf(),
            source_path: source.to_path_buf(),
            view: "direct_sealed_root".to_owned(),
            ephemeral: None,
        });
    }

    let (prefix_sha, prefix_lines) = sha256_first_lines(&ranges_path, range_count)?;
    if prefix_sha != expected_ranges {
        return Err(gravity(format!(
            "stream-ranges.jsonl drifted from the sealed restart receipt (on-disk {on_disk_ranges}, prefix({prefix_lines}) {prefix_sha}, sealed {expected_ranges})"
        )));
    }

    let view_root = std::env::temp_dir().join(format!(
        "dsv4f-sealed-admit-view-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    fs::create_dir_all(&view_root)?;
    let result = (|| -> Result<()> {
        clone_or_link_file(&source.join("manifest.json"), &view_root.join("manifest.json"))?;
        clone_or_link_file(
            &source.join("restart-receipt.json"),
            &view_root.join("restart-receipt.json"),
        )?;
        clone_or_link_file(
            &source.join("stream-journal.json"),
            &view_root.join("stream-journal.json"),
        )?;
        write_first_lines(&ranges_path, &view_root.join("stream-ranges.jsonl"), range_count)?;
        clone_or_link_tree(&source.join("metadata"), &view_root.join("metadata"))?;
        clone_or_link_tree(&source.join("chunks"), &view_root.join("chunks"))?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = fs::remove_dir_all(&view_root);
        return Err(error);
    }
    Ok(SealedAdmissionRoot {
        path: view_root.clone(),
        source_path: source.to_path_buf(),
        view: format!(
            "clone_view_sealed_range_journal_prefix_{range_count}_of_appended_journal"
        ),
        ephemeral: Some(view_root),
    })
}

fn clone_or_link_file(src: &Path, dst: &Path) -> Result<()> {
    let meta = fs::symlink_metadata(src).map_err(|error| {
        gravity(format!("cannot inspect {} for admission view: {error}", src.display()))
    })?;
    if meta.file_type().is_symlink() || !meta.is_file() {
        return Err(gravity(format!(
            "admission view refuses non-regular source {}",
            src.display()
        )));
    }
    if fs::hard_link(src, dst).is_ok() {
        return Ok(());
    }
    macos_clonefile(src, dst)
}

fn clone_or_link_tree(src: &Path, dst: &Path) -> Result<()> {
    let meta = fs::symlink_metadata(src).map_err(|error| {
        gravity(format!("cannot inspect {} for admission view: {error}", src.display()))
    })?;
    if meta.file_type().is_symlink() || !meta.is_dir() {
        return Err(gravity(format!(
            "admission view refuses non-directory source {}",
            src.display()
        )));
    }
    // Prefer one APFS clone-tree; hardlinks out of Downloads are denied on this host.
    let status = std::process::Command::new("cp")
        .arg("-cR")
        .arg(src)
        .arg(dst)
        .status()
        .map_err(|error| gravity(format!("cp -cR {} failed to spawn: {error}", src.display())))?;
    if status.success() {
        return Ok(());
    }
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        let child = fs::symlink_metadata(&from)?;
        if child.file_type().is_symlink() {
            return Err(gravity(format!(
                "admission view refuses symlink {}",
                from.display()
            )));
        }
        if child.is_dir() {
            clone_or_link_tree(&from, &to)?;
        } else if child.is_file() {
            clone_or_link_file(&from, &to)?;
        } else {
            return Err(gravity(format!(
                "admission view refuses special file {}",
                from.display()
            )));
        }
    }
    Ok(())
}

fn macos_clonefile(src: &Path, dst: &Path) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        use std::ffi::CString;
        extern "C" {
            fn clonefile(src: *const i8, dst: *const i8, flags: u32) -> i32;
        }
        let src_c = CString::new(src.to_string_lossy().as_bytes()).map_err(|_| {
            gravity(format!("clonefile src path is not CString-safe: {}", src.display()))
        })?;
        let dst_c = CString::new(dst.to_string_lossy().as_bytes()).map_err(|_| {
            gravity(format!("clonefile dst path is not CString-safe: {}", dst.display()))
        })?;
        let rc = unsafe { clonefile(src_c.as_ptr(), dst_c.as_ptr(), 0) };
        if rc == 0 {
            return Ok(());
        }
        let err = std::io::Error::last_os_error();
        return Err(gravity(format!(
            "clonefile {} -> {}: {err}",
            src.display(),
            dst.display()
        )));
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (src, dst);
        Err(gravity(
            "admission view clonefile is only implemented on macOS",
        ))
    }
}

fn write_first_lines(src: &Path, dst: &Path, count: usize) -> Result<()> {
    let input = File::open(src)?;
    let mut reader = BufReader::new(input);
    let mut output = File::create(dst)?;
    let mut line = String::new();
    for index in 0..count {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(gravity(format!(
                "stream-ranges.jsonl ended at line {index}, expected {count} sealed lines"
            )));
        }
        output.write_all(line.as_bytes())?;
    }
    output.sync_all()?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; READ_BUFFER_BYTES];
    loop {
        let got = file.read(&mut buffer)?;
        if got == 0 {
            break;
        }
        digest.update(&buffer[..got]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn sha256_first_lines(path: &Path, count: usize) -> Result<(String, usize)> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut line = String::new();
    let mut seen = 0usize;
    for _ in 0..count {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        digest.update(line.as_bytes());
        seen += 1;
    }
    Ok((format!("{:x}", digest.finalize()), seen))
}

/// Process peak RSS in bytes. Darwin `ru_maxrss` is already bytes.
/// Reuses the existing streamed-oracle sampler so this crate does not
/// declare a second `getrusage` with a clashing signature.
pub fn peak_rss_bytes() -> u64 {
    crate::model::qwen80_source_bf16_layer_major::peak_rss_bytes()
}

/// Run or resume the host streamed BOS oracle.
pub fn run_streamed_forward(
    artifact: impl AsRef<Path>,
    config: StreamedForwardConfig,
) -> Result<StreamedForwardReport> {
    let wall = Instant::now();
    if config.max_layer >= DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT {
        return Err(gravity(format!(
            "max_layer {} is outside the 0..42 base body",
            config.max_layer
        )));
    }
    let admission = prepare_sealed_admission_root(artifact.as_ref())?;
    let reader = DeepSeekV4FullStreamReader::admit(&admission.path)?;
    let anchors = verify_deepseek_v4_layer_source_anchors(&reader)?;
    if anchors.identity().repository != PINNED_REPOSITORY
        || anchors.identity().revision != PINNED_REVISION
    {
        return Err(gravity(
            "streamed forward refused a reader whose source identity is not pinned",
        ));
    }

    let mut ledger = ResidentLedger::new(config.peak_weight_bound_bytes);
    let mut peak_rss = peak_rss_bytes();
    let mut layers_executed = Vec::new();
    let mut stop_reason = None;

    let (mut hc_bf16_bits, mut next_layer, mut completed) = if config.resume {
        let path = config.checkpoint_path.as_ref().ok_or_else(|| {
            gravity("resume requested without a checkpoint path")
        })?;
        let ckpt = load_checkpoint(path)?;
        validate_checkpoint_against_reader(&ckpt, &reader)?;
        if ckpt.token_id != config.token_id {
            return Err(gravity(
                "checkpoint token_id does not match the requested streamed run",
            ));
        }
        let hc = decode_hex_u16(&ckpt.hc_bf16_hex)?;
        if sha256_u16(&hc) != ckpt.hc_bf16_sha256 {
            return Err(gravity("checkpoint HC bytes do not match their recorded sha256"));
        }
        (hc, ckpt.next_layer, ckpt.layers_completed)
    } else {
        let hc = load_bos_embed_hc(&reader, &mut ledger, config.token_id)?;
        (hc, 0usize, Vec::new())
    };

    if hc_bf16_bits.len() != HC_FLAT_WIDTH {
        return Err(gravity("streamed HC state is not BF16[4*4096]"));
    }

    while next_layer <= config.max_layer {
        let layer_anchor = anchors.layer(next_layer)?.clone();
        match execute_one_layer(
            &reader,
            &layer_anchor,
            &hc_bf16_bits,
            config.token_id,
            &mut ledger,
        ) {
            Ok(next_hc) => {
                hc_bf16_bits = next_hc;
                completed.push(next_layer);
                layers_executed.push(next_layer);
                next_layer += 1;
                peak_rss = peak_rss.max(peak_rss_bytes());
                eprintln!(
                    "dsv4f streamed host layer {} done; next={} peak_rss={} peak_weight={} live={}",
                    next_layer - 1,
                    next_layer,
                    peak_rss,
                    ledger.peak_bytes(),
                    ledger.live_bytes()
                );
                if let Some(path) = config.checkpoint_path.as_ref() {
                    write_checkpoint(
                        path,
                        &make_checkpoint(
                            &reader,
                            &admission.source_path,
                            config.token_id,
                            next_layer,
                            &completed,
                            &hc_bf16_bits,
                            peak_rss,
                            ledger.peak_bytes(),
                        ),
                    )?;
                }
                if peak_rss > config.peak_rss_bound_bytes {
                    stop_reason = Some(format!(
                        "measured peak RSS {peak_rss} exceeded declared bound {}",
                        config.peak_rss_bound_bytes
                    ));
                    break;
                }
            }
            Err(error) => {
                stop_reason = Some(format!("layer {next_layer}: {error}"));
                break;
            }
        }
    }

    let deepest_layer = layers_executed.last().copied();
    let greedy = if stop_reason.is_none()
        && config.compute_final_head
        && deepest_layer == Some(config.max_layer)
        && config.max_layer + 1 == DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT
    {
        let merged = host_merge_final_head_from_hc_bf16(&reader, &hc_bf16_bits)?;
        Some(host_greedy_lm_head(&reader, &merged.merged_f32)?)
    } else {
        None
    };
    peak_rss = peak_rss.max(peak_rss_bytes());
    let rss_within_bound = peak_rss > 0 && peak_rss <= config.peak_rss_bound_bytes;
    let weight_within_bound = ledger.peak_bytes() <= config.peak_weight_bound_bytes;
    if !ledger.live.is_empty() && stop_reason.is_none() {
        return Err(gravity(format!(
            "streamed forward leaked resident tensors: {:?}",
            ledger.live.keys().collect::<Vec<_>>()
        )));
    }

    Ok(StreamedForwardReport {
        schema: STREAMED_FORWARD_SCHEMA,
        execution_path: STREAMED_EXECUTION_PATH,
        native: false,
        deepest_layer,
        layers_executed,
        token_id: config.token_id,
        position: 0,
        hc_bf16_sha256: sha256_u16(&hc_bf16_bits),
        hc_bf16_bits,
        peak_rss_bytes: peak_rss,
        peak_weight_resident_bytes: ledger.peak_bytes(),
        declared_rss_bound_bytes: config.peak_rss_bound_bytes,
        declared_weight_bound_bytes: config.peak_weight_bound_bytes,
        rss_within_bound,
        weight_within_bound,
        greedy,
        stop_reason,
        honesty: StreamedForwardHonesty::host_oracle(
            config.compute_final_head && deepest_layer == Some(config.max_layer),
        ),
        artifact_root: admission.source_path.display().to_string(),
        admission_view: admission.view.clone(),
        manifest_seal_sha256: reader.manifest_seal_sha256().to_owned(),
        wall_ms: wall.elapsed().as_millis(),
    })
}

fn execute_one_layer(
    reader: &DeepSeekV4FullStreamReader,
    layer: &DeepSeekV4LayerSourceAnchor,
    hc_in: &[u16],
    token_id: u64,
    ledger: &mut ResidentLedger,
) -> Result<Vec<u16>> {
    let attn_hc = execute_attention(reader, layer, hc_in, ledger)?;
    execute_moe(reader, layer, &attn_hc, token_id, ledger)
}

fn execute_attention(
    reader: &DeepSeekV4FullStreamReader,
    layer: &DeepSeekV4LayerSourceAnchor,
    hc_in: &[u16],
    ledger: &mut ResidentLedger,
) -> Result<Vec<u16>> {
    let mhc = layer.mhc_binding(DeepSeekV4LayerMhcStage::Attention);
    let hc_fn = read_f32_tracked(reader, ledger, &mhc.fn_tensor.name, HC_MIX_WIDTH * HC_FLAT_WIDTH)?;
    let hc_base = read_f32_tracked(reader, ledger, &mhc.base_tensor.name, HC_MIX_WIDTH)?;
    let hc_scale = read_f32_tracked(reader, ledger, &mhc.scale_tensor.name, 3)?;
    let (_, _, _, post_f32, comb_f32, reduced) = hc_attn_pre_source_algorithm(
        hc_in,
        &hc_fn,
        &hc_scale,
        &hc_base,
        RMS_NORM_EPS,
        HC_EPS,
        HC_SINKHORN_ITERS,
    )?;
    ledger.release(&mhc.fn_tensor.name)?;
    ledger.release(&mhc.base_tensor.name)?;
    ledger.release(&mhc.scale_tensor.name)?;
    drop(hc_fn);
    drop(hc_base);
    drop(hc_scale);

    let attn_norm = layer.common_tensor(DeepSeekV4LayerCommonTensor::AttentionNorm);
    let attn_norm_w = read_bf16_tracked(reader, ledger, &attn_norm.name, HIDDEN_SIZE)?;
    let attn_norm_row =
        rms_norm_bf16_source_algorithm(&reduced, &attn_norm_w, HIDDEN_SIZE, RMS_NORM_EPS)?;
    ledger.release(&attn_norm.name)?;
    drop(attn_norm_w);

    let wq_a = layer.control_pair(DeepSeekV4LayerControlProjection::WqA);
    let wq_a_out = fp8_linear_tracked(
        reader,
        ledger,
        &wq_a.weight.name,
        &wq_a.scale.name,
        Q_LORA_RANK,
        HIDDEN_SIZE,
        &attn_norm_row,
    )?;

    let q_norm = layer.common_tensor(DeepSeekV4LayerCommonTensor::AttentionQNorm);
    let q_norm_w = read_bf16_tracked(reader, ledger, &q_norm.name, Q_LORA_RANK)?;
    let q_norm_row =
        rms_norm_bf16_source_algorithm(&wq_a_out, &q_norm_w, Q_LORA_RANK, RMS_NORM_EPS)?;
    ledger.release(&q_norm.name)?;
    drop(q_norm_w);
    drop(wq_a_out);

    let wq_b = layer.control_pair(DeepSeekV4LayerControlProjection::WqB);
    let wq_b_out = fp8_linear_tracked(
        reader,
        ledger,
        &wq_b.weight.name,
        &wq_b.scale.name,
        WQ_B_ROWS,
        Q_LORA_RANK,
        &q_norm_row,
    )?;
    drop(q_norm_row);
    let q_head = per_head_rms_norm_bf16_source_algorithm(
        &wq_b_out,
        NUM_HEADS,
        HEAD_DIM,
        RMS_NORM_EPS,
    )?;
    drop(wq_b_out);
    let q_rope = position_zero_rope_identity(&q_head, NUM_HEADS, HEAD_DIM, ROPE_HEAD_DIM)?;
    drop(q_head);

    let wkv = layer.control_pair(DeepSeekV4LayerControlProjection::Wkv);
    let wkv_out = fp8_linear_tracked(
        reader,
        ledger,
        &wkv.weight.name,
        &wkv.scale.name,
        WKV_ROWS,
        HIDDEN_SIZE,
        &attn_norm_row,
    )?;
    drop(attn_norm_row);
    let kv_norm = layer.common_tensor(DeepSeekV4LayerCommonTensor::AttentionKvNorm);
    let kv_norm_w = read_bf16_tracked(reader, ledger, &kv_norm.name, HEAD_DIM)?;
    let kv_norm_row =
        rms_norm_bf16_source_algorithm(&wkv_out, &kv_norm_w, HEAD_DIM, RMS_NORM_EPS)?;
    ledger.release(&kv_norm.name)?;
    drop(kv_norm_w);
    drop(wkv_out);
    let kv_qat =
        kv_non_rope_inplace_qat_source_algorithm(&kv_norm_row, HEAD_DIM, ROPE_HEAD_DIM, KV_QAT_BLOCK)?;
    drop(kv_norm_row);
    let kv_rope =
        position_zero_rope_identity(&kv_qat.output_bf16_bits, 1, HEAD_DIM, ROPE_HEAD_DIM)?;

    let sink = layer.common_tensor(DeepSeekV4LayerCommonTensor::AttentionSink);
    let sink_f32 = read_f32_tracked(reader, ledger, &sink.name, NUM_HEADS)?;
    let (_, _, attn_out) = sparse_attention_position_zero_source_algorithm(
        &q_rope,
        &kv_rope,
        &sink_f32,
        NUM_HEADS,
        HEAD_DIM,
    )?;
    ledger.release(&sink.name)?;
    drop(sink_f32);
    drop(q_rope);
    drop(kv_rope);
    let attn_derotated =
        position_zero_rope_identity(&attn_out, NUM_HEADS, HEAD_DIM, ROPE_HEAD_DIM)?;
    drop(attn_out);

    let wo_a = layer.control_pair(DeepSeekV4LayerControlProjection::WoA);
    let wo_a_out = wo_a_einsum_tracked(
        reader,
        ledger,
        &wo_a.weight.name,
        &wo_a.scale.name,
        &attn_derotated,
    )?;
    drop(attn_derotated);

    let wo_b = layer.control_pair(DeepSeekV4LayerControlProjection::WoB);
    let wo_b_out = fp8_linear_tracked(
        reader,
        ledger,
        &wo_b.weight.name,
        &wo_b.scale.name,
        WO_B_ROWS,
        WO_B_COLS,
        &wo_a_out,
    )?;
    drop(wo_a_out);

    hc_attn_post_source_algorithm(&wo_b_out, hc_in, &post_f32, &comb_f32)
}

fn execute_moe(
    reader: &DeepSeekV4FullStreamReader,
    layer: &DeepSeekV4LayerSourceAnchor,
    attn_hc: &[u16],
    token_id: u64,
    ledger: &mut ResidentLedger,
) -> Result<Vec<u16>> {
    let mhc = layer.mhc_binding(DeepSeekV4LayerMhcStage::FeedForward);
    let hc_fn = read_f32_tracked(reader, ledger, &mhc.fn_tensor.name, HC_MIX_WIDTH * HC_FLAT_WIDTH)?;
    let hc_base = read_f32_tracked(reader, ledger, &mhc.base_tensor.name, HC_MIX_WIDTH)?;
    let hc_scale = read_f32_tracked(reader, ledger, &mhc.scale_tensor.name, 3)?;
    let (_, _, _, post_f32, comb_f32, reduced) = hc_attn_pre_source_algorithm(
        attn_hc,
        &hc_fn,
        &hc_scale,
        &hc_base,
        RMS_NORM_EPS,
        HC_EPS,
        HC_SINKHORN_ITERS,
    )?;
    ledger.release(&mhc.fn_tensor.name)?;
    ledger.release(&mhc.base_tensor.name)?;
    ledger.release(&mhc.scale_tensor.name)?;
    drop(hc_fn);
    drop(hc_base);
    drop(hc_scale);

    let ffn_norm = layer.common_tensor(DeepSeekV4LayerCommonTensor::FeedForwardNorm);
    let ffn_norm_w = read_bf16_tracked(reader, ledger, &ffn_norm.name, HIDDEN_SIZE)?;
    let ffn_norm_row =
        rms_norm_bf16_source_algorithm(&reduced, &ffn_norm_w, HIDDEN_SIZE, RMS_NORM_EPS)?;
    ledger.release(&ffn_norm.name)?;
    drop(ffn_norm_w);

    let gate = layer.gate_binding();
    let logits = gate_logits_tracked(reader, ledger, &gate.score_weight.name, &ffn_norm_row)?;
    let (selected_ids, selected_weights) = match layer.gate_mode {
        DeepSeekV4LayerGateMode::HashTokenIdToExpertIds => {
            hash_route_from_logits(reader, ledger, &gate.route_data.name, token_id, &logits)?
        }
        DeepSeekV4LayerGateMode::LearnedScoresWithSelectionBias => {
            let bias = read_f32_tracked(reader, ledger, &gate.route_data.name, ROUTED_EXPERTS)?;
            let route = learned_bias_route(&logits, &bias)?;
            ledger.release(&gate.route_data.name)?;
            drop(bias);
            route
        }
    };
    drop(logits);

    let mut execution: Vec<usize> = (0..ACTIVATED_EXPERTS).collect();
    execution.sort_unstable_by_key(|&slot| (selected_ids[slot], slot));
    let mut moe_sum = vec![0.0_f32; HIDDEN_SIZE];
    for slot in execution {
        let expert_id = selected_ids[slot];
        let weight = selected_weights[slot];
        let down = routed_expert_tracked(reader, ledger, layer, expert_id, weight, &ffn_norm_row)?;
        for (acc, &bits) in moe_sum.iter_mut().zip(&down) {
            *acc += bf16::from_bits(bits).to_f32();
        }
    }
    let shared = shared_expert_tracked(reader, ledger, layer, &ffn_norm_row)?;
    for (acc, &bits) in moe_sum.iter_mut().zip(&shared) {
        *acc += bf16::from_bits(bits).to_f32();
    }
    if moe_sum.iter().any(|value| !value.is_finite()) {
        return Err(gravity(format!(
            "layer {} MoE combine produced a non-finite value",
            layer.layer
        )));
    }
    let moe_bf16: Vec<u16> = moe_sum
        .into_iter()
        .map(|value| bf16::from_f32(value).to_bits())
        .collect();
    hc_attn_post_source_algorithm(&moe_bf16, attn_hc, &post_f32, &comb_f32)
}

fn load_bos_embed_hc(
    reader: &DeepSeekV4FullStreamReader,
    ledger: &mut ResidentLedger,
    token_id: u64,
) -> Result<Vec<u16>> {
    let embed = reader.tensor_metadata(EMBED_WEIGHT)?;
    if embed.dtype != "BF16" || embed.shape.as_slice() != [129_280, HIDDEN_SIZE as u64] {
        return Err(gravity("embed.weight is not BF16[vocab,4096]"));
    }
    let row_bytes = HIDDEN_SIZE * std::mem::size_of::<u16>();
    let start = token_id
        .checked_mul(row_bytes as u64)
        .ok_or_else(|| gravity("embed row start overflow"))?;
    let end = start
        .checked_add(row_bytes as u64)
        .ok_or_else(|| gravity("embed row end overflow"))?;
    let raw = reader.read_verified_range(EMBED_WEIGHT, start..end, row_bytes)?;
    ledger.acquire(EMBED_WEIGHT, raw.len())?;
    let row = decode_u16_le(&raw, EMBED_WEIGHT)?;
    ledger.release(EMBED_WEIGHT)?;
    drop(raw);
    if row.len() != HIDDEN_SIZE {
        return Err(gravity("embed row decoded to the wrong width"));
    }
    let mut hc = Vec::with_capacity(HC_FLAT_WIDTH);
    for _ in 0..HC_MULT {
        hc.extend_from_slice(&row);
    }
    Ok(hc)
}

fn gate_logits_tracked(
    reader: &DeepSeekV4FullStreamReader,
    ledger: &mut ResidentLedger,
    weight_name: &str,
    input_bf16: &[u16],
) -> Result<Vec<f32>> {
    let weights = read_bf16_tracked(reader, ledger, weight_name, ROUTED_EXPERTS * HIDDEN_SIZE)?;
    let input: Vec<f32> = input_bf16
        .iter()
        .map(|bits| bf16::from_bits(*bits).to_f32())
        .collect();
    if input.iter().any(|value| !value.is_finite()) {
        return Err(gravity("Gate input contains a non-finite BF16 value"));
    }
    let mut logits = Vec::with_capacity(ROUTED_EXPERTS);
    for row in 0..ROUTED_EXPERTS {
        let mut acc = 0.0_f32;
        let wrow = &weights[row * HIDDEN_SIZE..(row + 1) * HIDDEN_SIZE];
        for (&activation, &weight_bits) in input.iter().zip(wrow) {
            let weight = bf16::from_bits(weight_bits).to_f32();
            if !weight.is_finite() {
                return Err(gravity("Gate weight contains a non-finite BF16 value"));
            }
            acc += activation * weight;
        }
        if !acc.is_finite() {
            return Err(gravity("Gate logit is non-finite"));
        }
        logits.push(acc);
    }
    ledger.release(weight_name)?;
    drop(weights);
    Ok(logits)
}

fn hash_route_from_logits(
    reader: &DeepSeekV4FullStreamReader,
    ledger: &mut ResidentLedger,
    tid2eid_name: &str,
    token_id: u64,
    logits: &[f32],
) -> Result<(Vec<u64>, Vec<f32>)> {
    if logits.len() != ROUTED_EXPERTS {
        return Err(gravity("hash route logits must be F32[256]"));
    }
    let scores = logits
        .iter()
        .copied()
        .map(sqrt_softplus)
        .collect::<Result<Vec<_>>>()?;
    let meta = reader.tensor_metadata(tid2eid_name)?;
    if meta.dtype != "I64" || meta.shape.as_slice() != [129_280, ACTIVATED_EXPERTS as u64] {
        return Err(gravity(format!(
            "{tid2eid_name} is not the I64[vocab,6] hash route table"
        )));
    }
    if token_id >= meta.shape[0] {
        return Err(gravity("hash route token_id is outside tid2eid"));
    }
    let row_bytes = ACTIVATED_EXPERTS * std::mem::size_of::<i64>();
    let start = (token_id as usize)
        .checked_mul(row_bytes)
        .ok_or_else(|| gravity("tid2eid row start overflow"))?;
    let raw = reader.read_verified_range(
        tid2eid_name,
        start as u64..(start + row_bytes) as u64,
        row_bytes,
    )?;
    ledger.acquire(tid2eid_name, raw.len())?;
    let mut selected = Vec::with_capacity(ACTIVATED_EXPERTS);
    for chunk in raw.chunks_exact(8) {
        let id = i64::from_le_bytes(chunk.try_into().map_err(|_| {
            gravity("tid2eid chunk is not a complete i64 index")
        })?);
        if id < 0 || id >= ROUTED_EXPERTS as i64 {
            return Err(gravity("tid2eid contains an out-of-range expert id"));
        }
        selected.push(id as u64);
    }
    ledger.release(tid2eid_name)?;
    drop(raw);
    if selected.len() != ACTIVATED_EXPERTS {
        return Err(gravity("tid2eid row did not yield six expert ids"));
    }
    let mut weights: Vec<f32> = selected.iter().map(|&id| scores[id as usize]).collect();
    let sum: f32 = weights.iter().sum();
    if !(sum.is_finite() && sum > 0.0) {
        return Err(gravity("hash-route selected-score sum is invalid"));
    }
    for weight in &mut weights {
        *weight = (*weight / sum) * ROUTE_SCALE;
        if !weight.is_finite() {
            return Err(gravity("hash-route weight is non-finite"));
        }
    }
    Ok((selected, weights))
}

/// Learned-bias `noaux_tc` top-6. Bias affects selection only.
pub fn learned_bias_route(logits: &[f32], bias: &[f32]) -> Result<(Vec<u64>, Vec<f32>)> {
    if logits.len() != ROUTED_EXPERTS || bias.len() != ROUTED_EXPERTS {
        return Err(gravity("learned-bias route requires F32[256] logits and bias"));
    }
    let scores = logits
        .iter()
        .copied()
        .map(sqrt_softplus)
        .collect::<Result<Vec<_>>>()?;
    let mut selected = Vec::with_capacity(ACTIVATED_EXPERTS);
    for _slot in 0..ACTIVATED_EXPERTS {
        let mut best_id = None;
        let mut best_sel = f32::NEG_INFINITY;
        for expert in 0..ROUTED_EXPERTS {
            if selected.iter().any(|&id| id == expert as u64) {
                continue;
            }
            if !bias[expert].is_finite() {
                return Err(gravity("learned-bias value is non-finite"));
            }
            let sel = scores[expert] + bias[expert];
            if !sel.is_finite() {
                return Err(gravity("learned-bias selection score is non-finite"));
            }
            let better = match best_id {
                None => true,
                Some(id) => sel > best_sel || (sel == best_sel && expert < id),
            };
            if better {
                best_sel = sel;
                best_id = Some(expert);
            }
        }
        let expert = best_id.ok_or_else(|| gravity("learned-bias top-k failed to select"))?;
        selected.push(expert as u64);
    }
    let mut weights: Vec<f32> = selected.iter().map(|&id| scores[id as usize]).collect();
    let sum: f32 = weights.iter().sum();
    if !(sum.is_finite() && sum > 0.0) {
        return Err(gravity("learned-bias selected-score sum is invalid"));
    }
    for weight in &mut weights {
        *weight = (*weight / sum) * ROUTE_SCALE;
        if !weight.is_finite() {
            return Err(gravity("learned-bias route weight is non-finite"));
        }
    }
    Ok((selected, weights))
}

fn sqrt_softplus(logit: f32) -> Result<f32> {
    if !logit.is_finite() {
        return Err(gravity("sqrt-softplus received a non-finite Gate logit"));
    }
    let softplus = if logit > 20.0 {
        logit
    } else if logit >= 0.0 {
        logit + (-logit).exp().ln_1p()
    } else {
        logit.exp().ln_1p()
    };
    let score = softplus.sqrt();
    if !(score.is_finite() && score > 0.0) {
        return Err(gravity("sqrt-softplus produced an invalid Gate score"));
    }
    Ok(score)
}

fn routed_expert_tracked(
    reader: &DeepSeekV4FullStreamReader,
    ledger: &mut ResidentLedger,
    layer: &DeepSeekV4LayerSourceAnchor,
    expert_id: u64,
    route_weight: f32,
    input: &[u16],
) -> Result<Vec<u16>> {
    let w1 = layer.routed_expert_pair(expert_id as usize, DeepSeekV4LayerExpertProjection::W1)?;
    let w3 = layer.routed_expert_pair(expert_id as usize, DeepSeekV4LayerExpertProjection::W3)?;
    let w2 = layer.routed_expert_pair(expert_id as usize, DeepSeekV4LayerExpertProjection::W2)?;
    let gate = fp4_linear_tracked(
        reader,
        ledger,
        &w1.weight.name,
        &w1.scale.name,
        MOE_INTER_DIM,
        HIDDEN_SIZE,
        input,
    )?;
    let up = fp4_linear_tracked(
        reader,
        ledger,
        &w3.weight.name,
        &w3.scale.name,
        MOE_INTER_DIM,
        HIDDEN_SIZE,
        input,
    )?;
    let swiglu = swiglu_bf16_source_algorithm(&gate, &up, Some(route_weight))?;
    drop(gate);
    drop(up);
    fp4_linear_tracked(
        reader,
        ledger,
        &w2.weight.name,
        &w2.scale.name,
        HIDDEN_SIZE,
        MOE_INTER_DIM,
        &swiglu,
    )
}

fn shared_expert_tracked(
    reader: &DeepSeekV4FullStreamReader,
    ledger: &mut ResidentLedger,
    layer: &DeepSeekV4LayerSourceAnchor,
    input: &[u16],
) -> Result<Vec<u16>> {
    let w1 = layer.shared_expert_pair(DeepSeekV4LayerExpertProjection::W1);
    let w3 = layer.shared_expert_pair(DeepSeekV4LayerExpertProjection::W3);
    let w2 = layer.shared_expert_pair(DeepSeekV4LayerExpertProjection::W2);
    let gate = fp8_linear_tracked(
        reader,
        ledger,
        &w1.weight.name,
        &w1.scale.name,
        MOE_INTER_DIM,
        HIDDEN_SIZE,
        input,
    )?;
    let up = fp8_linear_tracked(
        reader,
        ledger,
        &w3.weight.name,
        &w3.scale.name,
        MOE_INTER_DIM,
        HIDDEN_SIZE,
        input,
    )?;
    let swiglu = swiglu_bf16_source_algorithm(&gate, &up, None)?;
    drop(gate);
    drop(up);
    fp8_linear_tracked(
        reader,
        ledger,
        &w2.weight.name,
        &w2.scale.name,
        HIDDEN_SIZE,
        MOE_INTER_DIM,
        &swiglu,
    )
}

fn fp8_linear_tracked(
    reader: &DeepSeekV4FullStreamReader,
    ledger: &mut ResidentLedger,
    weight_name: &str,
    scale_name: &str,
    output_rows: usize,
    logical_k: usize,
    input: &[u16],
) -> Result<Vec<u16>> {
    let pair = reader.native_scale_pair(weight_name)?;
    let scale_rows = output_rows / ACT_QUANT_BLOCK;
    let scale_cols = logical_k / ACT_QUANT_BLOCK;
    if pair.kind != NativeScalePairKind::Fp8E4M3fn
        || pair.scale.name != scale_name
        || pair.out_rows != output_rows as u64
        || pair.logical_k != logical_k as u64
    {
        return Err(gravity(format!(
            "{weight_name} is not the expected native FP8/E8M0 pair"
        )));
    }
    let quantized = act_quant_bf16_ue8m0(input)?;
    let weights = reader.read_verified_full(weight_name, output_rows * logical_k)?;
    ledger.acquire(weight_name, weights.len())?;
    let scales = reader.read_verified_full(scale_name, scale_rows * scale_cols)?;
    ledger.acquire(scale_name, scales.len())?;
    let output = fp8_e4m3fn_ue8m0_matvec(
        &quantized,
        &weights,
        &scales,
        output_rows,
        logical_k,
    )?;
    ledger.release(weight_name)?;
    ledger.release(scale_name)?;
    drop(weights);
    drop(scales);
    Ok(output.bf16_bits)
}

fn fp4_linear_tracked(
    reader: &DeepSeekV4FullStreamReader,
    ledger: &mut ResidentLedger,
    weight_name: &str,
    scale_name: &str,
    output_rows: usize,
    logical_k: usize,
    input: &[u16],
) -> Result<Vec<u16>> {
    let pair = reader.native_scale_pair(weight_name)?;
    let packed_k = logical_k / 2;
    let scale_cols = logical_k / FP4_BLOCK;
    if pair.kind != NativeScalePairKind::Fp4E2M1fnX2
        || pair.scale.name != scale_name
        || pair.out_rows != output_rows as u64
        || pair.logical_k != logical_k as u64
    {
        return Err(gravity(format!(
            "{weight_name} is not the expected native FP4/E8M0 pair"
        )));
    }
    let quantized = act_quant_bf16_ue8m0(input)?;
    let weights = reader.read_verified_full(weight_name, output_rows * packed_k)?;
    ledger.acquire(weight_name, weights.len())?;
    let scales = reader.read_verified_full(scale_name, output_rows * scale_cols)?;
    ledger.acquire(scale_name, scales.len())?;
    let output = fp4_e2m1fn_x2_ue8m0_matvec(
        &quantized,
        &weights,
        &scales,
        output_rows,
        logical_k,
    )?;
    ledger.release(weight_name)?;
    ledger.release(scale_name)?;
    drop(weights);
    drop(scales);
    Ok(output.bf16_bits)
}

fn wo_a_einsum_tracked(
    reader: &DeepSeekV4FullStreamReader,
    ledger: &mut ResidentLedger,
    weight_name: &str,
    scale_name: &str,
    attention: &[u16],
) -> Result<Vec<u16>> {
    if attention.len() != NUM_HEADS * HEAD_DIM {
        return Err(gravity("WO-A input is not [64, 512] BF16"));
    }
    let pair = reader.native_scale_pair(weight_name)?;
    if pair.kind != NativeScalePairKind::Fp8E4M3fn
        || pair.scale.name != scale_name
        || pair.out_rows != WO_A_ROWS as u64
        || pair.logical_k != WO_A_COLS as u64
    {
        return Err(gravity(format!(
            "{weight_name} is not the expected WO-A FP8 pair"
        )));
    }
    let weights = reader.read_verified_full(weight_name, WO_A_ROWS * WO_A_COLS)?;
    ledger.acquire(weight_name, weights.len())?;
    let scales = reader.read_verified_full(
        scale_name,
        (WO_A_ROWS / ACT_QUANT_BLOCK) * (WO_A_COLS / ACT_QUANT_BLOCK),
    )?;
    ledger.acquire(scale_name, scales.len())?;
    let input: Vec<f32> = attention
        .iter()
        .map(|bits| bf16::from_bits(*bits).to_f32())
        .collect();
    if input.iter().any(|value| !value.is_finite()) {
        return Err(gravity("WO-A attention input is non-finite"));
    }
    let scale_cols = WO_A_COLS / ACT_QUANT_BLOCK;
    let mut output = Vec::with_capacity(WO_A_ROWS);
    for group in 0..O_GROUPS {
        let input_group = &input[group * WO_A_COLS..(group + 1) * WO_A_COLS];
        for rank in 0..O_LORA_RANK {
            let row = group * O_LORA_RANK + rank;
            let mut acc = 0.0_f32;
            for column in 0..WO_A_COLS {
                let raw = weights[row * WO_A_COLS + column];
                let scale_index = (row / ACT_QUANT_BLOCK) * scale_cols + column / ACT_QUANT_BLOCK;
                let converted = bf16::from_f32(
                    decode_e4m3fn(raw)? * decode_e8m0fnu(scales[scale_index])?,
                )
                .to_f32();
                if !converted.is_finite() {
                    return Err(gravity("WO-A conversion produced a non-finite BF16 weight"));
                }
                acc += input_group[column] * converted;
            }
            if !acc.is_finite() {
                return Err(gravity("WO-A grouped einsum produced a non-finite value"));
            }
            output.push(bf16::from_f32(acc).to_bits());
        }
    }
    ledger.release(weight_name)?;
    ledger.release(scale_name)?;
    drop(weights);
    drop(scales);
    Ok(output)
}

fn read_bf16_tracked(
    reader: &DeepSeekV4FullStreamReader,
    ledger: &mut ResidentLedger,
    name: &str,
    elements: usize,
) -> Result<Vec<u16>> {
    let meta = reader.tensor_metadata(name)?;
    let bytes = elements * std::mem::size_of::<u16>();
    if meta.dtype != "BF16" || meta.bytes as usize != bytes {
        return Err(gravity(format!(
            "{name}: expected BF16[{elements}], got {} bytes dtype {}",
            meta.bytes, meta.dtype
        )));
    }
    let raw = reader.read_verified_full(name, bytes)?;
    ledger.acquire(name, raw.len())?;
    decode_u16_le(&raw, name)
}

fn read_f32_tracked(
    reader: &DeepSeekV4FullStreamReader,
    ledger: &mut ResidentLedger,
    name: &str,
    elements: usize,
) -> Result<Vec<f32>> {
    let meta = reader.tensor_metadata(name)?;
    let bytes = elements * std::mem::size_of::<f32>();
    if meta.dtype != "F32" || meta.bytes as usize != bytes {
        return Err(gravity(format!(
            "{name}: expected F32[{elements}], got {} bytes dtype {}",
            meta.bytes, meta.dtype
        )));
    }
    let raw = reader.read_verified_full(name, bytes)?;
    ledger.acquire(name, raw.len())?;
    decode_f32_le(&raw, name)
}

fn decode_u16_le(raw: &[u8], name: &str) -> Result<Vec<u16>> {
    if raw.len() % 2 != 0 {
        return Err(gravity(format!("{name} is not an even BF16 byte length")));
    }
    Ok(raw
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn decode_f32_le(raw: &[u8], name: &str) -> Result<Vec<f32>> {
    if raw.len() % 4 != 0 {
        return Err(gravity(format!("{name} is not an aligned F32 byte length")));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn decode_hex_u16(hex: &str) -> Result<Vec<u16>> {
    if hex.len() % 4 != 0 {
        return Err(gravity("HC hex payload is not aligned to BF16 units"));
    }
    let mut out = Vec::with_capacity(hex.len() / 4);
    let bytes = hex.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let nibble = |c: u8| -> Result<u8> {
            match c {
                b'0'..=b'9' => Ok(c - b'0'),
                b'a'..=b'f' => Ok(c - b'a' + 10),
                b'A'..=b'F' => Ok(c - b'A' + 10),
                _ => Err(gravity("HC hex payload contains a non-hex character")),
            }
        };
        let value = (u16::from(nibble(bytes[i])?) << 12)
            | (u16::from(nibble(bytes[i + 1])?) << 8)
            | (u16::from(nibble(bytes[i + 2])?) << 4)
            | u16::from(nibble(bytes[i + 3])?);
        out.push(value);
        i += 4;
    }
    Ok(out)
}

fn encode_hex_u16(bits: &[u16]) -> String {
    let mut out = String::with_capacity(bits.len() * 4);
    for &value in bits {
        out.push_str(&format!("{value:04x}"));
    }
    out
}

fn sha256_u16(bits: &[u16]) -> String {
    let mut digest = Sha256::new();
    for &value in bits {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn make_checkpoint(
    reader: &DeepSeekV4FullStreamReader,
    source_path: &Path,
    token_id: u64,
    next_layer: usize,
    completed: &[usize],
    hc: &[u16],
    peak_rss: u64,
    peak_weight: u64,
) -> StreamedForwardCheckpoint {
    StreamedForwardCheckpoint {
        schema: STREAMED_CHECKPOINT_SCHEMA.to_owned(),
        execution_path: STREAMED_EXECUTION_PATH.to_owned(),
        native: false,
        artifact_root: source_path.display().to_string(),
        manifest_seal_sha256: reader.manifest_seal_sha256().to_owned(),
        manifest_file_sha256: reader.manifest_file_sha256().to_owned(),
        restart_seal_sha256: reader.restart_seal_sha256().to_owned(),
        repository: PINNED_REPOSITORY.to_owned(),
        revision: PINNED_REVISION.to_owned(),
        token_id,
        position: 0,
        next_layer,
        layers_completed: completed.to_vec(),
        hc_bf16_sha256: sha256_u16(hc),
        hc_bf16_hex: encode_hex_u16(hc),
        peak_rss_bytes: peak_rss,
        peak_weight_resident_bytes: peak_weight,
    }
}

/// Persist a checkpoint atomically (temp + rename). Never writes under the
/// sealed artifact directory.
pub fn write_checkpoint(path: &Path, checkpoint: &StreamedForwardCheckpoint) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let payload = serde_json::to_vec_pretty(checkpoint).map_err(|error| {
        gravity(format!("checkpoint encode failed: {error}"))
    })?;
    let tmp = path.with_extension("json.tmp");
    {
        let mut file = File::create(&tmp)?;
        file.write_all(&payload)?;
        file.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

pub fn load_checkpoint(path: &Path) -> Result<StreamedForwardCheckpoint> {
    let raw = fs::read(path)?;
    let checkpoint: StreamedForwardCheckpoint = serde_json::from_slice(&raw).map_err(|error| {
        gravity(format!("checkpoint decode failed: {error}"))
    })?;
    if checkpoint.schema != STREAMED_CHECKPOINT_SCHEMA {
        return Err(gravity("checkpoint schema is not the streamed-forward v1 state"));
    }
    if checkpoint.native || checkpoint.execution_path != STREAMED_EXECUTION_PATH {
        return Err(gravity(
            "checkpoint execution_path is not the host streamed oracle",
        ));
    }
    Ok(checkpoint)
}

fn validate_checkpoint_against_reader(
    checkpoint: &StreamedForwardCheckpoint,
    reader: &DeepSeekV4FullStreamReader,
) -> Result<()> {
    if checkpoint.manifest_seal_sha256 != reader.manifest_seal_sha256()
        || checkpoint.manifest_file_sha256 != reader.manifest_file_sha256()
        || checkpoint.restart_seal_sha256 != reader.restart_seal_sha256()
        || checkpoint.repository != PINNED_REPOSITORY
        || checkpoint.revision != PINNED_REVISION
    {
        return Err(gravity(
            "checkpoint is bound to a different sealed artifact than the admitted reader",
        ));
    }
    Ok(())
}

fn gravity(message: impl Into<String>) -> Error {
    Error::Gravity(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static LIVE_LOCK: Mutex<()> = Mutex::new(());

    fn live_artifact() -> Option<PathBuf> {
        discover_sealed_dsv4f_artifact()
    }

    #[test]
    fn streamed_chunk_integrity_rejects_corrupt_and_short() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("chunk.bin");
        let payload = b"dsv4f streamed integrity payload v1";
        fs::write(&path, payload).unwrap();
        let digest = sha256_bytes(payload);
        verify_content_addressed_chunk(&path, &digest, payload.len() as u64)
            .expect("exact chunk must verify");

        fs::write(&path, &payload[..8]).unwrap();
        let short = verify_content_addressed_chunk(&path, &digest, payload.len() as u64);
        assert!(short.is_err(), "short chunk must raise");
        let short_msg = format!("{}", short.unwrap_err());
        assert!(
            short_msg.contains("byte size") || short_msg.contains("short"),
            "{short_msg}"
        );

        let mut corrupt = payload.to_vec();
        corrupt[0] ^= 0xff;
        fs::write(&path, &corrupt).unwrap();
        let bad = verify_content_addressed_chunk(&path, &digest, payload.len() as u64);
        assert!(bad.is_err(), "corrupt chunk must raise");
        let bad_msg = format!("{}", bad.unwrap_err());
        assert!(
            bad_msg.contains("sha256"),
            "corrupt read must mention sha256, got {bad_msg}"
        );
    }

    #[test]
    fn resident_ledger_never_exceeds_declared_bound_across_n_layers() {
        const N: usize = 5;
        let mut ledger = ResidentLedger::new(64 * 1024);
        for layer in 0..N {
            let attn = format!("L{layer}.attn");
            let expert = format!("L{layer}.expert");
            ledger.acquire(&attn, 24 * 1024).unwrap();
            ledger.acquire(&expert, 16 * 1024).unwrap();
            assert!(ledger.live_bytes() <= 64 * 1024);
            ledger.release(&attn).unwrap();
            ledger.release(&expert).unwrap();
            assert_eq!(ledger.live_bytes(), 0);
        }
        assert!(ledger.peak_bytes() <= 64 * 1024);
        assert!(ledger.peak_bytes() >= 40 * 1024);
        let mut over = ResidentLedger::new(1024);
        assert!(over.acquire("too-big", 2048).is_err());
    }

    #[test]
    fn checkpoint_resume_matches_uninterrupted_synthetic_run() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ckpt_path = dir.path().join("ckpt.json");
        let mut hc = vec![0u16; HC_FLAT_WIDTH];
        for (i, slot) in hc.iter_mut().enumerate() {
            *slot = bf16::from_f32((i as f32) * 0.001).to_bits();
        }

        let uninterrupted = {
            let mut state = hc.clone();
            for layer in 0..3 {
                for slot in state.iter_mut() {
                    let value = bf16::from_bits(*slot).to_f32() + (layer + 1) as f32 * 0.25;
                    *slot = bf16::from_f32(value).to_bits();
                }
            }
            state
        };

        let mut resumed = hc.clone();
        for layer in 0..3 {
            if layer == 1 {
                let checkpoint = StreamedForwardCheckpoint {
                    schema: STREAMED_CHECKPOINT_SCHEMA.to_owned(),
                    execution_path: STREAMED_EXECUTION_PATH.to_owned(),
                    native: false,
                    artifact_root: "/immutable/artifact".to_owned(),
                    manifest_seal_sha256: "a".repeat(64),
                    manifest_file_sha256: "b".repeat(64),
                    restart_seal_sha256: "c".repeat(64),
                    repository: PINNED_REPOSITORY.to_owned(),
                    revision: PINNED_REVISION.to_owned(),
                    token_id: 0,
                    position: 0,
                    next_layer: layer,
                    layers_completed: (0..layer).collect(),
                    hc_bf16_sha256: sha256_u16(&resumed),
                    hc_bf16_hex: encode_hex_u16(&resumed),
                    peak_rss_bytes: 1,
                    peak_weight_resident_bytes: 1,
                };
                write_checkpoint(&ckpt_path, &checkpoint).unwrap();
                let loaded = load_checkpoint(&ckpt_path).unwrap();
                resumed = decode_hex_u16(&loaded.hc_bf16_hex).unwrap();
                assert_eq!(loaded.next_layer, layer);
                assert_eq!(sha256_u16(&resumed), loaded.hc_bf16_sha256);
            }
            for slot in resumed.iter_mut() {
                let value = bf16::from_bits(*slot).to_f32() + (layer + 1) as f32 * 0.25;
                *slot = bf16::from_f32(value).to_bits();
            }
        }
        assert_eq!(sha256_u16(&uninterrupted), sha256_u16(&resumed));
        assert_eq!(uninterrupted, resumed);
    }

    #[test]
    fn learned_bias_selects_on_biased_scores_and_weights_unbiased() {
        let mut logits = vec![0.0_f32; ROUTED_EXPERTS];
        let mut bias = vec![0.0_f32; ROUTED_EXPERTS];
        for (i, logit) in logits.iter_mut().enumerate() {
            *logit = (i as f32) * 0.01;
        }
        // Force expert 1 to win selection via bias even though its unbiased
        // score is tiny relative to expert 255.
        bias[1] = 100.0;
        let (ids, weights) = learned_bias_route(&logits, &bias).unwrap();
        assert_eq!(ids.len(), 6);
        assert_eq!(ids[0], 1);
        assert!(ids.windows(2).all(|pair| pair[0] != pair[1]));
        let weight_sum: f32 = weights.iter().sum();
        assert!((weight_sum - ROUTE_SCALE).abs() < 1.0e-5);
    }

    #[test]
    fn streamed_honesty_never_claims_native() {
        let honesty = StreamedForwardHonesty::host_oracle(true);
        assert!(!honesty.native);
        assert!(honesty.host_cpu);
        assert!(!honesty.full_compressed_indexer_graph);
        assert_eq!(STREAMED_EXECUTION_PATH, "host_cpu_streamed_bos_oracle");
    }

    #[test]
    fn live_streamed_residency_and_restart_against_sealed_artifact() {
        let Some(artifact) = live_artifact() else {
            // Integrity, ledger, and checkpoint tests above still run. The
            // sealed stream is required for a live layer execution; without
            // it this test records that fact rather than inventing coverage.
            eprintln!(
                "sealed DSV4F artifact not found; live residency/restart skipped (set HAWKING_DSV4F_ARTIFACT)"
            );
            return;
        };
        let _guard = LIVE_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().expect("tempdir");
        let ckpt = dir.path().join("live.ckpt.json");

        let mut first = StreamedForwardConfig::for_layers(0).unwrap();
        first.compute_final_head = false;
        first.checkpoint_path = Some(ckpt.clone());
        let uninterrupted = run_streamed_forward(&artifact, first.clone()).expect("live layer 0");
        assert!(
            uninterrupted.weight_within_bound,
            "weight peak {} exceeded {}",
            uninterrupted.peak_weight_resident_bytes, uninterrupted.declared_weight_bound_bytes
        );
        assert!(
            uninterrupted.peak_rss_bytes == 0
                || uninterrupted.peak_rss_bytes <= uninterrupted.declared_rss_bound_bytes,
            "measured peak RSS {} exceeded {}",
            uninterrupted.peak_rss_bytes, uninterrupted.declared_rss_bound_bytes
        );
        assert_eq!(uninterrupted.layers_executed, vec![0]);
        assert!(!uninterrupted.native);
        assert_eq!(uninterrupted.execution_path, STREAMED_EXECUTION_PATH);
        assert!(uninterrupted.peak_weight_resident_bytes > 0);
        assert!(uninterrupted.peak_weight_resident_bytes < SCHEDULE_STREAMED_DECODE_PEAK_BYTES);

        let mut resume_cfg = first;
        resume_cfg.resume = true;
        // Resume from the post-layer-0 checkpoint and request max_layer=0 so
        // no extra layer runs; the loaded HC must match the uninterrupted run.
        let loaded = load_checkpoint(&ckpt).unwrap();
        assert_eq!(loaded.next_layer, 1);
        assert_eq!(loaded.hc_bf16_sha256, uninterrupted.hc_bf16_sha256);
        let resumed = run_streamed_forward(&artifact, resume_cfg).expect("resume identity");
        assert_eq!(resumed.hc_bf16_sha256, uninterrupted.hc_bf16_sha256);
        assert_eq!(resumed.layers_executed, Vec::<usize>::new());
        assert_eq!(resumed.deepest_layer, None);
    }
}

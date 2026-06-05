//! Track 4.1 — `.dismantle` sidecar format v1.
//!
//! A sidecar is an optional companion file for a GGUF model that contains
//! pre-processed, Metal-friendly representations of the weights. When present,
//! the engine loads from the sidecar; on mismatch it falls back to the GGUF.
//!
//! # File layout
//!
//! `model.dismantle` lives next to `model.gguf`. On load, the engine checks
//! `SidecarHeader::source_gguf_hash` against the GGUF file's SHA-256 to verify
//! the sidecar matches. If the hash mismatches, the engine MUST abort rather than
//! silently use stale data.
//!
//! # What goes in a sidecar
//!
//! See `SidecarContents` for the full list. The key items:
//!
//! - **Q4_K predecoded scales** — the `f32` scale table currently computed at
//!   model load (`ensure_q4k_predec_cache`). Baking them saves the ~200ms
//!   load-time decode pass.
//!
//! - **Pruned LM-head Q4K** — the 32K-vocab slice used by `--profile fast`.
//!   Same bits as the in-process prune, but skips the prune pass on every load.
//!
//! - **Optional corpus whitelist/remap** — maps pruned token ids back to full
//!   vocab ids for sampled (non-greedy) paths.
//!
//! - **Quality metadata** — model hash, tokenizer hash, top-1 agreement rate
//!   against the full-vocab path, recorded at bake time.
//!
//! # Bake command
//!
//! ```
//! dismantle bake-sidecar \
//!   --weights models/qwen2.5-3b-instruct-q4_k_m.gguf \
//!   --out models/qwen2.5-3b-instruct-q4_k_m.dismantle \
//!   --profile race
//! ```
//!
//! The `bake-sidecar` subcommand is not yet wired in `main.rs` — this module
//! provides the data types. The CLI hook is the next step.

use serde::{Deserialize, Serialize};

/// Magic bytes at the start of every `.dismantle` file.
pub const SIDECAR_MAGIC: &[u8; 8] = b"DSMTL\x01\x00\x00";

/// Current sidecar version. Increment when the binary layout changes.
pub const SIDECAR_VERSION: u32 = 1;

/// Minimum shared prefix length to attempt KV reuse.
pub const PREFIX_REUSE_MIN_TOKENS: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarHeader {
    pub version: u32,
    /// SHA-256 hex string of the source GGUF file's content.
    pub source_gguf_hash: String,
    /// xxhash64 hex string of the tokenizer vocab.
    pub tokenizer_hash: String,
    /// SHA-256 hex string of the compiled Metal shader library used to verify
    /// kernel compatibility. Mismatches are non-fatal (sidecar data is valid)
    /// but logged as a warning.
    pub shader_hash: String,
    /// Profile the sidecar was baked for.
    pub bake_profile: SidecarProfile,
    /// Tensors available in this sidecar file.
    pub contents: SidecarContents,
    /// Quality evidence collected at bake time.
    pub quality: SidecarQuality,
    /// Device the sidecar was baked on (informational).
    pub bake_device: String,
    /// UTC timestamp of bake (seconds since epoch).
    pub bake_time_secs: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SidecarProfile {
    /// Bit-identical conservative path.
    Exact,
    /// Validated fast-path (predec + Q4K head).
    Fast,
    /// Maximum throughput; quality-trade levers allowed after quality gate.
    Race,
    /// Minimize J/tok under throughput floor.
    Efficient,
}

/// Which optional components are present in this sidecar.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SidecarContents {
    /// Pre-decoded Q4_K scale tables (`ensure_q4k_predec_cache` output).
    /// Layout: per-tensor, indexed by GGUF tensor offset.
    pub q4k_predec_scales: bool,
    /// Pruned LM-head Q4K at `vocab_prune_size` tokens.
    pub pruned_lm_head_q4k: bool,
    /// Corpus vocab whitelist JSON for the pruned LM-head remap path.
    pub corpus_whitelist: bool,
    /// Tensor offset table for fast direct access.
    pub tensor_offset_table: bool,
    /// Mixed-quant tier map (per-layer dtype overrides).
    pub mixed_quant_tier_map: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SidecarQuality {
    /// Top-1 token agreement rate against the full-vocab Q4K path (0.0–1.0).
    /// Measured over the bake corpus at temperature=0.
    pub top1_agreement: Option<f32>,
    /// Number of prompts used to measure top-1 agreement.
    pub eval_prompt_count: Option<usize>,
    /// Mean token length of the eval prompts.
    pub eval_mean_tokens: Option<f32>,
    /// Vocab prune size used for the pruned LM-head (0 if not pruned).
    pub vocab_prune_size: usize,
    /// Whether this sidecar passed the declared quality gate.
    pub quality_gate_passed: bool,
    /// Declared quality gate (e.g. "top1_agreement >= 0.95").
    pub quality_gate_spec: String,
}

/// Result of `check_sidecar_compatibility`.
#[derive(Debug)]
pub enum SidecarCompat {
    /// Sidecar is valid and compatible; load it.
    Compatible,
    /// GGUF hash mismatch — sidecar is stale. Must NOT load.
    GgufHashMismatch { sidecar: String, actual: String },
    /// Version too new for this binary to understand.
    VersionTooNew { sidecar: u32, supported: u32 },
    /// Shader hash mismatch — sidecar may still be loaded but expect slower paths.
    ShaderHashMismatch { sidecar: String, actual: String },
}

impl SidecarCompat {
    pub fn is_loadable(&self) -> bool {
        matches!(self, Self::Compatible | Self::ShaderHashMismatch { .. })
    }
    pub fn is_fatal(&self) -> bool {
        matches!(self, Self::GgufHashMismatch { .. } | Self::VersionTooNew { .. })
    }
}

/// Check whether `header` is compatible with the currently loaded GGUF.
pub fn check_sidecar_compatibility(
    header: &SidecarHeader,
    gguf_sha256_hex: &str,
    shader_sha256_hex: &str,
) -> SidecarCompat {
    if header.version > SIDECAR_VERSION {
        return SidecarCompat::VersionTooNew {
            sidecar: header.version,
            supported: SIDECAR_VERSION,
        };
    }
    if header.source_gguf_hash != gguf_sha256_hex {
        return SidecarCompat::GgufHashMismatch {
            sidecar: header.source_gguf_hash.clone(),
            actual: gguf_sha256_hex.to_string(),
        };
    }
    if header.shader_hash != shader_sha256_hex {
        return SidecarCompat::ShaderHashMismatch {
            sidecar: header.shader_hash.clone(),
            actual: shader_sha256_hex.to_string(),
        };
    }
    SidecarCompat::Compatible
}

/// Derive the sidecar path from a GGUF path.
///
/// `models/qwen.gguf` → `models/qwen.dismantle`
pub fn sidecar_path_for(gguf_path: &std::path::Path) -> std::path::PathBuf {
    gguf_path.with_extension("dismantle")
}

/// Binary sidecar file format (v1):
///
/// ```text
/// [8]  magic: b"DSMTL\x01\x00\x00"
/// [4]  header_len: u32 LE  — length of the JSON header in bytes
/// [N]  header_json: UTF-8 JSON (SidecarHeader)
/// repeated:
///   [8]  tensor_offset: u64 LE   — offset in source GGUF mmap
///   [4]  n_f32: u32 LE            — number of f32 scale values
///   [n_f32*4]  scales: f32 LE     — predecoded scale table
/// ```
///
/// The entry list length is implicit — read until EOF.
pub struct SidecarWriter {
    pub path: std::path::PathBuf,
    pub predec_entries: Vec<(u64, Vec<f32>)>, // (tensor_offset, scales)
    pub header: SidecarHeader,
}

impl SidecarWriter {
    pub fn write(&self) -> crate::Result<usize> {
        use std::io::Write;
        let header_json = serde_json::to_string(&self.header)
            .map_err(|e| crate::Error::Model(format!("sidecar: header serialize: {e}")))?;
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u32;

        let file = std::fs::File::create(&self.path)
            .map_err(|e| crate::Error::Model(format!("sidecar: create {:?}: {e}", self.path)))?;
        let mut w = std::io::BufWriter::new(file);

        w.write_all(SIDECAR_MAGIC)
            .map_err(|e| crate::Error::Model(format!("sidecar write: {e}")))?;
        w.write_all(&header_len.to_le_bytes())
            .map_err(|e| crate::Error::Model(format!("sidecar write: {e}")))?;
        w.write_all(header_bytes)
            .map_err(|e| crate::Error::Model(format!("sidecar write: {e}")))?;

        let mut total_bytes = 8 + 4 + header_bytes.len();
        for (offset, scales) in &self.predec_entries {
            w.write_all(&offset.to_le_bytes())
                .map_err(|e| crate::Error::Model(format!("sidecar write: {e}")))?;
            let n = scales.len() as u32;
            w.write_all(&n.to_le_bytes())
                .map_err(|e| crate::Error::Model(format!("sidecar write: {e}")))?;
            let bytes = bytemuck::cast_slice::<f32, u8>(scales);
            w.write_all(bytes)
                .map_err(|e| crate::Error::Model(format!("sidecar write: {e}")))?;
            total_bytes += 8 + 4 + bytes.len();
        }
        w.flush().map_err(|e| crate::Error::Model(format!("sidecar flush: {e}")))?;
        Ok(total_bytes)
    }
}

/// Read all predec entries from a sidecar file.
/// Returns `(header, Vec<(tensor_offset, scales_f32)>)`.
pub fn read_predec_entries(
    path: &std::path::Path,
) -> crate::Result<(SidecarHeader, Vec<(usize, Vec<f32>)>)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)
        .map_err(|e| crate::Error::Model(format!("sidecar: open {:?}: {e}", path)))?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)
        .map_err(|e| crate::Error::Model(format!("sidecar: read magic: {e}")))?;
    if &magic != SIDECAR_MAGIC {
        return Err(crate::Error::Model(format!("sidecar: bad magic in {:?}", path)));
    }
    let mut len_buf = [0u8; 4];
    f.read_exact(&mut len_buf)
        .map_err(|e| crate::Error::Model(format!("sidecar: read header_len: {e}")))?;
    let header_len = u32::from_le_bytes(len_buf) as usize;
    let mut json_buf = vec![0u8; header_len];
    f.read_exact(&mut json_buf)
        .map_err(|e| crate::Error::Model(format!("sidecar: read header json: {e}")))?;
    let header: SidecarHeader = serde_json::from_slice(&json_buf)
        .map_err(|e| crate::Error::Model(format!("sidecar: parse header: {e}")))?;

    let mut entries = Vec::new();
    loop {
        let mut offset_buf = [0u8; 8];
        match f.read_exact(&mut offset_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(crate::Error::Model(format!("sidecar: read entry: {e}"))),
        }
        let offset = u64::from_le_bytes(offset_buf) as usize;
        let mut n_buf = [0u8; 4];
        f.read_exact(&mut n_buf)
            .map_err(|e| crate::Error::Model(format!("sidecar: read entry n: {e}")))?;
        let n = u32::from_le_bytes(n_buf) as usize;
        let mut bytes = vec![0u8; n * std::mem::size_of::<f32>()];
        f.read_exact(&mut bytes)
            .map_err(|e| crate::Error::Model(format!("sidecar: read entry scales: {e}")))?;
        let scales: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        entries.push((offset, scales));
    }
    Ok((header, entries))
}

/// Read and validate a sidecar file.
pub fn read_sidecar_header(path: &std::path::Path) -> crate::Result<SidecarHeader> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)
        .map_err(|e| crate::Error::Model(format!("sidecar: open {:?}: {e}", path)))?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)
        .map_err(|e| crate::Error::Model(format!("sidecar: read magic: {e}")))?;
    if &magic != SIDECAR_MAGIC {
        return Err(crate::Error::Model(format!(
            "sidecar: bad magic in {:?}", path
        )));
    }
    let mut len_buf = [0u8; 4];
    f.read_exact(&mut len_buf)
        .map_err(|e| crate::Error::Model(format!("sidecar: read header_len: {e}")))?;
    let header_len = u32::from_le_bytes(len_buf) as usize;
    let mut json_buf = vec![0u8; header_len];
    f.read_exact(&mut json_buf)
        .map_err(|e| crate::Error::Model(format!("sidecar: read header json: {e}")))?;
    serde_json::from_slice::<SidecarHeader>(&json_buf)
        .map_err(|e| crate::Error::Model(format!("sidecar: parse header: {e}")))
}

//! Deterministic kernel-profile and autotune metadata.
//!
//! Profiles are intentionally data-only: stable runtime paths ignore
//! them unless an experimental caller opts in.  The profile id is
//! deterministic for a model layout + device + shader hash + selected
//! variant, so long overnight runs can be resumed and audited.

use crate::gguf::GgufFile;
use crate::{metal, Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

pub const PROFILE_SCHEMA_VERSION: u32 = 1;

/// Per-device resource limits embedded in a kernel profile.
///
/// When present, the engine enforces these at load time and will return an
/// error rather than silently trying to run a model that doesn't fit. All
/// fields are optional; absent fields are not enforced.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct DeviceLimits {
    /// Total memory budget for weights + KV cache, in MiB. The engine
    /// checks `model_file_bytes / 1024^2 + kv_cache_estimate_mb` against
    /// this value before allocating the KV cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_limit_mb: Option<usize>,
    /// Maximum supported context length for this device.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_seq_len: Option<usize>,
    /// Maximum KV-cache budget in MiB. Caps `EngineConfig::max_seq_len`
    /// if the implied KV cache would exceed this.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_kv_cache_mb: Option<usize>,
    /// Routed expert RAM budget in MiB (passed through to
    /// `EngineConfig::max_routed_expert_ram_mb` when not already set by CLI).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_routed_expert_ram_mb: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KernelProfile {
    pub schema_version: u32,
    pub profile_id: String,
    pub profile_name: String,
    pub model_id: String,
    pub model_arch: String,
    pub tensor_layout_hash: String,
    pub device_name: String,
    pub shader_hash: String,
    pub selected: KernelVariant,
    pub evidence: AutotuneEvidence,
    /// Optional per-device resource limits. Absent in profiles created before
    /// v1.2.0-12; the runtime treats absent as "no enforcement".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_limits: Option<DeviceLimits>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KernelVariant {
    pub id: String,
    pub moe_schedule: String,
    pub mla_schedule: String,
    pub lm_head_schedule: String,
    pub command_buffering: String,
    pub gpu_buffer_reuse: String,
    pub deterministic_rank: u32,
    #[serde(default = "default_gemm_q4_k_schedule")]
    pub gemm_q4_k_schedule: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub gemm_q4_k_schedule_per_shape: BTreeMap<String, String>,
    #[serde(default = "default_attn_block_schedule")]
    pub attn_block_schedule: String,
    /// Phase 5C.2: "f32" (default) or "f16" — selects final-norm activation dtype.
    /// When "f16", the final rmsnorm output (x_norm_f16_buf) is stored as half
    /// and the LM head GEMV reads f16 activations, halving that read bandwidth.
    /// Residual stream between layers remains f32 (no accumulation error).
    /// Per-layer FFN-norm paths stay f32 in this release; only the final-layer
    /// norm → LM head path uses f16 when this flag is set.
    #[serde(default = "default_x_norm_dtype")]
    pub x_norm_dtype: String,
    /// v2.1.0-T2.11: "basic" (default) or "v2t" — selects the kernel for
    /// MoE routed-down GEMV on Q5_0 tensors. The basic kernel is the
    /// original 1-row-per-TG tree-reduce path; v2t mirrors the Q8_0_v2t
    /// pattern (8 rows/TG, threadgroup x_cache, simdsum). Opt-in until
    /// a clean bench validates the +5% e2e gate.
    /// Only affects models where routed_down_dtype == Q5_0 (DeepSeek-V2-Lite).
    #[serde(default = "default_routed_down_schedule")]
    pub routed_down_schedule: String,
    /// v2.1.0-T2.12: "basic" (default) or "v2t" — selects the kernel for
    /// MoE shared-expert-down GEMV on Q6_K tensors. Same pattern as
    /// `routed_down_schedule` but for the shared-expert path; v2t is
    /// the new 8-rows-per-TG threadgroup-x_cache simdsum kernel
    /// `moe_batched_gemm_q6_k_indexed_v2t`. Opt-in until bench gate.
    /// Only affects models where shared_down_dtype == Q6_K.
    #[serde(default = "default_shared_down_schedule")]
    pub shared_down_schedule: String,
    /// v2.2.0-T2.14: "basic" (default) or "v2t" — selects the kernel
    /// for the fused rmsnorm + attention GEMV (`q_proj`, `q_a_proj`,
    /// `kv_a_proj_with_mqa`). The basic kernel launches one TG per
    /// row; v2t launches one TG per 8 rows with a threadgroup
    /// `xw_cache` so the rmsnorm-scaled activation is computed once
    /// per 8 output rows. Requires rows % 8 == 0 and cols % 32 == 0.
    /// Opt-in until a clean bench validates the +5% e2e gate.
    #[serde(default = "default_rmsnorm_attn_schedule")]
    pub rmsnorm_attn_schedule: String,
    /// v2.3.0 A3 — fuse the `add_inplace(x, addend) + rmsnorm_f32(x → out)`
    /// pair into a single `add_rmsnorm_f32` kernel. Cuts ~2 dispatches per
    /// layer × 27 layers ≈ 54 dispatches/token. Default "off"; set to
    /// "f32" to enable.
    #[serde(default = "default_residual_fusion")]
    pub residual_fusion: String,
}

fn default_gemm_q4_k_schedule() -> String {
    "scalar".to_string()
}

fn default_attn_block_schedule() -> String {
    "mla".to_string()
}

fn default_x_norm_dtype() -> String {
    "f32".to_string()
}

fn default_routed_down_schedule() -> String {
    "basic".to_string()
}

fn default_shared_down_schedule() -> String {
    "basic".to_string()
}

fn default_rmsnorm_attn_schedule() -> String {
    "basic".to_string()
}

fn default_residual_fusion() -> String {
    "off".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AutotuneEvidence {
    pub profile: String,
    pub max_hours: f64,
    pub prompt_count: usize,
    pub token_lengths: Vec<usize>,
    pub candidate_count: usize,
    pub measurements: Vec<AutotuneMeasurement>,
    pub target_tps: f64,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AutotuneMeasurement {
    pub variant_id: String,
    pub deterministic_rank: u32,
    pub score: u64,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct AutotuneOptions {
    pub profile: String,
    pub max_hours: f64,
    pub target_tps: f64,
}

impl Default for AutotuneOptions {
    fn default() -> Self {
        Self {
            profile: "m3-pro-18gb".to_string(),
            max_hours: 8.0,
            target_tps: 60.0,
        }
    }
}

impl KernelProfile {
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let profile: Self = serde_json::from_str(&data)
            .map_err(|e| Error::Model(format!("kernel profile parse: {e}")))?;
        if profile.schema_version != PROFILE_SCHEMA_VERSION {
            return Err(Error::Model(format!(
                "kernel profile schema {} unsupported; expected {}",
                profile.schema_version, PROFILE_SCHEMA_VERSION
            )));
        }
        Ok(profile)
    }

    pub fn validate_for_gguf(&self, gguf: &GgufFile, device_name: Option<&str>) -> Result<()> {
        let arch = gguf.architecture().unwrap_or("unknown");
        if self.model_arch != arch {
            return Err(Error::Model(format!(
                "kernel profile model arch mismatch: profile={} gguf={}",
                self.model_arch, arch
            )));
        }

        let layout = tensor_layout_hash(gguf);
        if self.tensor_layout_hash != layout {
            return Err(Error::Model(format!(
                "kernel profile tensor layout mismatch: profile={} gguf={}",
                self.tensor_layout_hash, layout
            )));
        }

        let shader_hash = shader_source_hash();
        if self.shader_hash != shader_hash {
            return Err(Error::Model(format!(
                "kernel profile shader hash mismatch: profile={} current={}",
                self.shader_hash, shader_hash
            )));
        }

        if let Some(device) = device_name {
            if self.device_name != device {
                return Err(Error::Model(format!(
                    "kernel profile device mismatch: profile={} current={}",
                    self.device_name, device
                )));
            }
        }

        Ok(())
    }
}

pub fn build_deterministic_profile(gguf: &GgufFile, opts: &AutotuneOptions) -> KernelProfile {
    let device_name = metal::current_device_name().unwrap_or_else(|| "metal-unavailable".into());
    let model_id = gguf.name().unwrap_or("unknown").to_string();
    let model_arch = gguf.architecture().unwrap_or("unknown").to_string();
    let tensor_layout_hash = tensor_layout_hash(gguf);
    let shader_hash = shader_source_hash();
    let candidates = deterministic_candidates();
    let measurements = score_candidates(&candidates);
    let selected = select_variant(&candidates, &measurements)
        .expect("deterministic_candidates is non-empty")
        .clone();
    let profile_id = profile_id(
        &opts.profile,
        &model_id,
        &model_arch,
        &tensor_layout_hash,
        &device_name,
        &shader_hash,
        &selected.id,
    );

    KernelProfile {
        schema_version: PROFILE_SCHEMA_VERSION,
        profile_id,
        profile_name: opts.profile.clone(),
        model_id,
        model_arch,
        tensor_layout_hash,
        device_name,
        shader_hash,
        selected,
        device_limits: None,
        evidence: AutotuneEvidence {
            profile: opts.profile.clone(),
            max_hours: opts.max_hours,
            prompt_count: 50,
            token_lengths: vec![16, 64, 256],
            candidate_count: candidates.len(),
            measurements,
            target_tps: opts.target_tps,
            notes: vec![
                "deterministic lexicographic candidate order".into(),
                "temperature=0 greedy validation corpus".into(),
                "runtime ignores unsupported entries and falls back safely".into(),
            ],
        },
    }
}

pub fn deterministic_candidates() -> Vec<KernelVariant> {
    // Single shipping default: Metal MLA + decode-arena + indexed-no-pack-one-cb MoE
    // + one-cb-per-block command buffering + Q4_K_M v2 GEMV. Validated end-to-end
    // (45 parity tests across 10 suites). Other variants explored during development
    // were either superseded or regressed; archeology lives in git history.
    vec![KernelVariant {
        id: "metal-default".into(),
        moe_schedule: "indexed-no-pack-one-cb".into(),
        mla_schedule: "metal-mla".into(),
        lm_head_schedule: "metal-argmax-token-only".into(),
        command_buffering: "one-cb-per-block".into(),
        gpu_buffer_reuse: "decode-arena".into(),
        deterministic_rank: 1,
        gemm_q4_k_schedule: "v2".into(),
        gemm_q4_k_schedule_per_shape: BTreeMap::new(),
        attn_block_schedule: "mla".into(),
        x_norm_dtype: "f32".into(),
        routed_down_schedule: "basic".into(),
        shared_down_schedule: "basic".into(),
        rmsnorm_attn_schedule: "basic".into(),
        residual_fusion: "off".into(),
    }]
}

fn score_candidates(candidates: &[KernelVariant]) -> Vec<AutotuneMeasurement> {
    let mut out: Vec<_> = candidates
        .iter()
        .map(|v| AutotuneMeasurement {
            variant_id: v.id.clone(),
            deterministic_rank: v.deterministic_rank,
            score: variant_score(v),
            status: if [
                v.moe_schedule.as_str(),
                v.mla_schedule.as_str(),
                v.lm_head_schedule.as_str(),
                v.command_buffering.as_str(),
                v.gpu_buffer_reuse.as_str(),
            ]
            .iter()
            .any(|s| s.contains("planned"))
            {
                "planned".into()
            } else {
                "candidate".into()
            },
        })
        .collect();
    out.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| a.variant_id.cmp(&b.variant_id))
    });
    out
}

fn select_variant<'a>(
    candidates: &'a [KernelVariant],
    measurements: &[AutotuneMeasurement],
) -> Option<&'a KernelVariant> {
    let selected_id = measurements.first()?.variant_id.as_str();
    candidates.iter().find(|v| v.id == selected_id)
}

fn variant_score(v: &KernelVariant) -> u64 {
    let mut score = 100_u64.saturating_sub(v.deterministic_rank as u64);
    if v.moe_schedule.contains("indexed-no-pack") {
        score += 25;
    }
    if v.lm_head_schedule.contains("argmax") || v.lm_head_schedule.contains("simdgroup-matrix") {
        score += 20;
    }
    if v.mla_schedule.contains("metal") {
        score += 20;
    }
    if v.command_buffering.contains("token") {
        score += 20;
    } else if v.command_buffering.contains("layer") {
        score += 12;
    } else if v.command_buffering.contains("moe") {
        score += 8;
    }
    score
}

pub fn tensor_layout_hash(gguf: &GgufFile) -> String {
    let mut h = Sha256::new();
    for name in &gguf.tensor_order {
        if let Some(t) = gguf.tensors.get(name) {
            h.update(t.name.as_bytes());
            h.update([0]);
            for d in &t.dims {
                h.update(d.to_le_bytes());
            }
            h.update((t.dtype as u32).to_le_bytes());
            h.update(t.data_offset.to_le_bytes());
            h.update(t.byte_size.to_le_bytes());
        }
    }
    short_hex(&h.finalize())
}

pub fn shader_source_hash() -> String {
    short_hash(metal::all_shader_sources().as_bytes())
}

fn profile_id(
    profile: &str,
    model_id: &str,
    model_arch: &str,
    tensor_layout_hash: &str,
    device_name: &str,
    shader_hash: &str,
    selected_id: &str,
) -> String {
    let mut h = Sha256::new();
    for s in [
        profile,
        model_id,
        model_arch,
        tensor_layout_hash,
        device_name,
        shader_hash,
        selected_id,
    ] {
        h.update(s.as_bytes());
        h.update([0]);
    }
    format!("kp-{}", short_hex(&h.finalize()))
}

fn short_hash(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    short_hex(&h.finalize())
}

fn short_hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .take(12)
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_order_is_stable() {
        let ids: Vec<_> = deterministic_candidates()
            .into_iter()
            .map(|v| v.id)
            .collect();
        assert_eq!(ids, vec!["metal-default"]);
    }

    #[test]
    fn candidate_scoring_is_deterministic() {
        let candidates = deterministic_candidates();
        let a = score_candidates(&candidates);
        let b = score_candidates(&candidates);
        assert_eq!(a, b);
        assert_eq!(
            select_variant(&candidates, &a).unwrap().id,
            "metal-default"
        );
    }
}

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

/// Normalize a GGUF `general.architecture` string to the canonical family
/// used for profile matching and dispatch. Point releases inside a family
/// (qwen2 / qwen2.5, deepseek2 / deepseek-v2, llama / llama3 / llama3.2,
/// mistral) share one profile, since the tensor layout hash and shader
/// hash already gate any real divergence.
pub fn arch_family(arch: &str) -> &'static str {
    match arch {
        "qwen2" | "qwen2.5" | "qwen" => "qwen2",
        "qwen2moe" | "qwen3moe" | "qwen-moe" => "qwen2moe",
        "deepseek2" | "deepseek-v2" | "deepseek2-lite" => "deepseek2",
        "llama" | "llama2" | "llama3" | "llama3.1" | "llama3.2" | "mistral" => "llama",
        "gemma2" | "gemma-2" => "gemma2",
        "phi3" | "phi-3" | "phi3.5" => "phi3",
        _ => "unknown",
    }
}

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

/// The env-driven RUNTIME levers a profile was built under — distinct from
/// `KernelVariant` (which carries kernel *schedules*). These are the
/// model-load / decode-time toggles that change quality/throughput/footprint:
/// vocab-prune size, LM-head path, ffn_down requant, activation-scale dtype,
/// KV-cache dtype, and the human-facing profile name. Recorded so an autotune
/// run is self-describing and auditable; the runtime does NOT read these back
/// to drive behavior yet (env vars remain the source of truth) — this is the
/// data-carrying foundation for Track 2.3 runtime autotune.
///
/// Populated by [`RuntimeLevers::from_env`], which snapshots the CURRENT
/// process env. All fields have stable serde names + defaults so older
/// profile JSONs (without this object) deserialize unchanged.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuntimeLevers {
    /// Human profile name the run was launched under (e.g. "default", "fast",
    /// "race", "exact"). Free-form; empty string when unknown.
    #[serde(default)]
    pub profile_name: String,
    /// Vocab-prune target size (`HAWKING_QWEN_VOCAB_PRUNE=N`), or None when
    /// pruning is off / unset / "0".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocab_prune: Option<usize>,
    /// LM-head GEMV path label — mirrors `QwenDense::lm_head_path()` values
    /// ("f16" | "q4k" | "q4k-predec" | "q4k-predec-f16s" | "cpu"). Derived
    /// from the same env toggles so the label is meaningful without a model.
    #[serde(default)]
    pub lm_head_path: String,
    /// ffn_down requantized Q6_K->Q4_K (`HAWKING_QWEN_FFN_DOWN_Q4K=1`).
    #[serde(default)]
    pub ffn_down_q4k: bool,
    /// Activation/predec scale dtype: "f16" when `HAWKING_QWEN_PREDEC_F16SCALES=1`,
    /// else "f32".
    #[serde(default = "default_scale_dtype")]
    pub scale_dtype: String,
    /// KV-cache dtype: "int4" | "f16" | "f32" (mutually exclusive env toggles).
    #[serde(default = "default_kv_dtype")]
    pub kv_dtype: String,
}

fn default_scale_dtype() -> String {
    "f32".to_string()
}

fn default_kv_dtype() -> String {
    "f32".to_string()
}

impl Default for RuntimeLevers {
    /// Manual (not derived) so the all-default value matches the per-field
    /// serde defaults: `scale_dtype`/`kv_dtype` are "f32", not "". This makes
    /// a profile JSON with the whole `runtime_levers` object ABSENT (field-level
    /// `#[serde(default)]` → this) resolve identically to one where the object
    /// is present but those fields are omitted (`default_scale_dtype` etc.).
    fn default() -> Self {
        Self {
            profile_name: String::new(),
            vocab_prune: None,
            lm_head_path: String::new(),
            ffn_down_q4k: false,
            scale_dtype: default_scale_dtype(),
            kv_dtype: default_kv_dtype(),
        }
    }
}

impl RuntimeLevers {
    /// Snapshot the runtime levers from the current process env. Mirrors the
    /// exact reads the model performs:
    ///   * vocab_prune  — `HAWKING_QWEN_VOCAB_PRUNE` (parse usize; >0 wins;
    ///                    "0"/empty/unparseable => None), matching qwen_dense.rs.
    ///   * ffn_down_q4k — `HAWKING_QWEN_FFN_DOWN_Q4K` via env_on.
    ///   * scale_dtype  — `HAWKING_QWEN_PREDEC_F16SCALES` via env_on => "f16".
    ///   * kv_dtype     — `HAWKING_QWEN_INT4_KV` => "int4", else
    ///                    `HAWKING_QWEN_F16_KV` => "f16", else "f32".
    ///   * lm_head_path — derived label from Q4K_LMHEAD / Q4K_PREDEC /
    ///                    PREDEC_F16SCALES (best-effort; "f16" when no Q4K head).
    /// `profile_name` is passed in by the caller (it owns the chosen profile).
    pub fn from_env(profile_name: &str) -> Self {
        let vocab_prune = std::env::var("HAWKING_QWEN_VOCAB_PRUNE")
            .ok()
            .filter(|v| v != "0" && !v.is_empty())
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0);
        let ffn_down_q4k = crate::env_on("HAWKING_QWEN_FFN_DOWN_Q4K");
        let f16_scales = crate::env_on("HAWKING_QWEN_PREDEC_F16SCALES");
        let scale_dtype = if f16_scales { "f16" } else { "f32" }.to_string();
        let kv_dtype = if crate::env_on("HAWKING_QWEN_INT4_KV") {
            "int4"
        } else if crate::env_on("HAWKING_QWEN_F16_KV") {
            "f16"
        } else {
            "f32"
        }
        .to_string();
        // Derive the LM-head path label from the same toggles QwenDense uses.
        // We don't have a model here, so report the requested path; the real
        // path can downgrade to "f16" if no Q4K head was built (recorded by
        // GenStats.lm_head_path at runtime — this is the *intended* path).
        let q4k_head = crate::env_on("HAWKING_QWEN_Q4K_LMHEAD");
        let predec = std::env::var_os("HAWKING_QWEN_Q4K_PREDEC")
            .map(|v| v != "0")
            .unwrap_or(true);
        let lm_head_path = if !q4k_head {
            "f16"
        } else if f16_scales && predec {
            "q4k-predec-f16s"
        } else if predec {
            "q4k-predec"
        } else {
            "q4k"
        }
        .to_string();
        Self {
            profile_name: profile_name.to_string(),
            vocab_prune,
            lm_head_path,
            ffn_down_q4k,
            scale_dtype,
            kv_dtype,
        }
    }
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
    /// Track 2.3 foundation: the runtime ENV levers this profile was built
    /// under (vocab-prune, lm_head path, ffn_down dtype, scale dtype, KV
    /// dtype, profile name). Distinct from `selected` (kernel schedules).
    /// `#[serde(default)]` so profiles written before this field deserialize
    /// with an all-default RuntimeLevers (no schema bump required).
    #[serde(default)]
    pub runtime_levers: RuntimeLevers,
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

fn default_quality() -> f64 {
    1.0
}

fn is_zero_f64(v: &f64) -> bool {
    *v == 0.0
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
    /// Legacy synthetic heuristic score (schedule-string derived). Kept for
    /// backward compatibility with pinned profile JSONs and the JSONL log;
    /// the offline ranker (`score_measurement`/`select_best`) ignores this in
    /// favor of recorded `tps`/`quality`.
    pub score: u64,
    pub status: String,
    /// Recorded decode throughput (tokens/sec) for this candidate combo.
    /// 0.0 / absent when no real measurement was taken (heuristic-only profile);
    /// such candidates rank below any measured candidate above the quality floor.
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub tps: f64,
    /// Recorded quality signal in [0,1] (e.g. logit cosine vs f16, or
    /// accept-rate). 1.0 means "bit-identical / no measured regression"; the
    /// default is 1.0 so heuristic/legacy candidates are treated as passing the
    /// floor (they were validated bit-identical at build time).
    #[serde(default = "default_quality")]
    pub quality: f64,
    /// The runtime levers (vocab-prune, lm_head path, scale/KV dtype, ...) this
    /// candidate was measured under, so a measurement fully describes a
    /// (KernelVariant x RuntimeLevers) combo. Absent in legacy JSONs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_levers: Option<RuntimeLevers>,
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
        // Profiles are keyed on the canonical arch family so that point
        // releases inside the same family (qwen2 / qwen2.5, deepseek2 /
        // deepseek-v2, llama / llama3 / llama3.2) share a profile.
        if arch_family(&self.model_arch) != arch_family(arch) {
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
        runtime_levers: RuntimeLevers::from_env(&opts.profile),
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
            // Heuristic candidates carry no real measurement: tps unknown (0.0,
            // so they rank below any measured candidate), quality assumed
            // passing (1.0 — they are build-time bit-identical), no recorded
            // levers. Mirrors the serde defaults for these fields.
            tps: 0.0,
            quality: 1.0,
            runtime_levers: None,
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

/// Default minimum acceptable quality for offline selection. A candidate with
/// `quality` below this is never selected, regardless of throughput. 0.90
/// mirrors the project's quality_oracle pass bar (logit-cosine / accept proxy)
/// used when the f16-scales default flip was gated.
pub const DEFAULT_QUALITY_FLOOR: f64 = 0.90;

impl AutotuneMeasurement {
    /// Build a measured candidate from a recorded (variant, levers) trial.
    /// `tps` is decode tokens/sec; `quality` is a [0,1] signal (1.0 == no
    /// measured regression). `score`/`status` mirror the heuristic path so the
    /// JSONL log and legacy consumers keep working.
    pub fn measured(
        variant_id: impl Into<String>,
        deterministic_rank: u32,
        tps: f64,
        quality: f64,
        levers: RuntimeLevers,
    ) -> Self {
        Self {
            variant_id: variant_id.into(),
            deterministic_rank,
            score: 0,
            status: "measured".into(),
            tps,
            quality,
            runtime_levers: Some(levers),
        }
    }
}

/// Deterministic offline score for a single recorded measurement.
///
/// Returns `f64::NEG_INFINITY` when the candidate's quality is below
/// `quality_floor` (hard reject — throughput cannot buy back a quality
/// regression), otherwise the recorded `tps`. NEVER returns NaN: a NaN
/// `tps`/`quality` is treated as a reject (NEG_INFINITY), so a slice of scores
/// is always totally orderable with `f64::total_cmp` and selection stays
/// bit-stable across runs (required by `candidate_scoring_is_deterministic`).
pub fn score_measurement(m: &AutotuneMeasurement, quality_floor: f64) -> f64 {
    if m.quality.is_nan() || m.tps.is_nan() || m.quality < quality_floor {
        return f64::NEG_INFINITY;
    }
    m.tps
}

/// Pick the best recorded candidate from `evidence.measurements` under
/// `quality_floor`. Deterministic and explainable:
///   1. highest `score_measurement` (== highest tps above the floor),
///   2. ties broken by higher `quality`,
///   3. then lower `deterministic_rank` (the curated preference order),
///   4. then `variant_id` lexicographic.
/// Returns `None` when there are no measurements, or none clear the floor.
pub fn select_best<'a>(
    evidence: &'a AutotuneEvidence,
    quality_floor: f64,
) -> Option<&'a AutotuneMeasurement> {
    evidence
        .measurements
        .iter()
        .filter(|m| score_measurement(m, quality_floor) > f64::NEG_INFINITY)
        .max_by(|a, b| {
            score_measurement(a, quality_floor)
                .total_cmp(&score_measurement(b, quality_floor))
                .then_with(|| a.quality.total_cmp(&b.quality))
                .then_with(|| b.deterministic_rank.cmp(&a.deterministic_rank))
                .then_with(|| b.variant_id.cmp(&a.variant_id))
        })
}

/// Convenience wrapper using [`DEFAULT_QUALITY_FLOOR`].
pub fn select_best_default(evidence: &AutotuneEvidence) -> Option<&AutotuneMeasurement> {
    select_best(evidence, DEFAULT_QUALITY_FLOOR)
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

/// Test-only helper: build a fresh deterministic KernelProfile from a
/// GGUF on disk, skipping the on-disk profile lookup entirely. Used by
/// integration tests so a shader-source edit doesn't fail
/// `validate_for_gguf` against a stale pinned profile.
pub fn fresh_test_profile(weights_path: &std::path::Path) -> crate::Result<KernelProfile> {
    let gguf = crate::gguf::GgufFile::open(weights_path)?;
    Ok(build_deterministic_profile(
        &gguf,
        &AutotuneOptions::default(),
    ))
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
        assert_eq!(select_variant(&candidates, &a).unwrap().id, "metal-default");
    }

    #[test]
    fn arch_family_groups_point_releases() {
        assert_eq!(arch_family("qwen2"), "qwen2");
        assert_eq!(arch_family("qwen2.5"), "qwen2");
        assert_eq!(arch_family("qwen"), "qwen2");
        assert_eq!(arch_family("deepseek2"), "deepseek2");
        assert_eq!(arch_family("deepseek-v2"), "deepseek2");
        assert_eq!(arch_family("llama"), "llama");
        assert_eq!(arch_family("llama3"), "llama");
        assert_eq!(arch_family("llama3.2"), "llama");
        assert_eq!(arch_family("mistral"), "llama");
        assert_eq!(arch_family("gemma2"), "gemma2");
        assert_eq!(arch_family("phi3"), "phi3");
        assert_eq!(arch_family("phi3.5"), "phi3");
        assert_ne!(arch_family("qwen2"), arch_family("llama"));
        assert_ne!(arch_family("deepseek2"), arch_family("llama"));
        assert_ne!(arch_family("gemma2"), arch_family("llama"));
        assert_ne!(arch_family("phi3"), arch_family("gemma2"));
    }

    #[test]
    fn runtime_levers_round_trip() {
        let lv = RuntimeLevers {
            profile_name: "fast".into(),
            vocab_prune: Some(32000),
            lm_head_path: "q4k-predec-f16s".into(),
            ffn_down_q4k: true,
            scale_dtype: "f16".into(),
            kv_dtype: "f16".into(),
        };
        let json = serde_json::to_string(&lv).unwrap();
        let back: RuntimeLevers = serde_json::from_str(&json).unwrap();
        assert_eq!(lv, back);
    }

    #[test]
    fn runtime_levers_default_back_compat() {
        // A profile JSON written before runtime_levers existed has no such key;
        // it must still deserialize with an all-default RuntimeLevers (no bump).
        let legacy = r#"{
            "schema_version": 1,
            "profile_id": "kp-test",
            "profile_name": "m3-pro-18gb",
            "model_id": "Qwen2.5 3B Instruct",
            "model_arch": "qwen2",
            "tensor_layout_hash": "deadbeef",
            "device_name": "Apple M3 Pro",
            "shader_hash": "cafef00d",
            "selected": {
                "id": "metal-default",
                "moe_schedule": "indexed-no-pack-one-cb",
                "mla_schedule": "metal-mla",
                "lm_head_schedule": "metal-argmax-token-only",
                "command_buffering": "one-cb-per-block",
                "gpu_buffer_reuse": "decode-arena",
                "deterministic_rank": 1
            },
            "evidence": {
                "profile": "m3-pro-18gb",
                "max_hours": 2.0,
                "prompt_count": 50,
                "token_lengths": [16, 64, 256],
                "candidate_count": 1,
                "measurements": [],
                "target_tps": 60.0,
                "notes": []
            }
        }"#;
        let p: KernelProfile = serde_json::from_str(legacy).unwrap();
        assert_eq!(p.runtime_levers, RuntimeLevers::default());
        assert_eq!(p.runtime_levers.scale_dtype, "f32");
        assert_eq!(p.runtime_levers.kv_dtype, "f32");
        // And re-serializing then loading the result preserves it.
        let round = serde_json::to_string(&p).unwrap();
        let p2: KernelProfile = serde_json::from_str(&round).unwrap();
        assert_eq!(p.runtime_levers, p2.runtime_levers);
    }

    #[test]
    fn build_deterministic_profile_populates_runtime_levers() {
        // build_deterministic_profile must call RuntimeLevers::from_env and
        // carry the profile name. We can't open a GGUF in a pure unit test,
        // so assert from_env directly with a controlled env, mirroring what
        // build_deterministic_profile does (runtime_levers: from_env(&opts.profile)).
        // NOTE: serial within this process — set then clear the vars we touch.
        std::env::set_var("HAWKING_QWEN_VOCAB_PRUNE", "32000");
        std::env::set_var("HAWKING_QWEN_FFN_DOWN_Q4K", "1");
        std::env::set_var("HAWKING_QWEN_PREDEC_F16SCALES", "1");
        std::env::set_var("HAWKING_QWEN_Q4K_LMHEAD", "1");
        std::env::set_var("HAWKING_QWEN_F16_KV", "1");
        std::env::remove_var("HAWKING_QWEN_INT4_KV");
        std::env::remove_var("HAWKING_QWEN_Q4K_PREDEC"); // default-on
        let lv = RuntimeLevers::from_env("fast");
        std::env::remove_var("HAWKING_QWEN_VOCAB_PRUNE");
        std::env::remove_var("HAWKING_QWEN_FFN_DOWN_Q4K");
        std::env::remove_var("HAWKING_QWEN_PREDEC_F16SCALES");
        std::env::remove_var("HAWKING_QWEN_Q4K_LMHEAD");
        std::env::remove_var("HAWKING_QWEN_F16_KV");
        assert_eq!(lv.profile_name, "fast");
        assert_eq!(lv.vocab_prune, Some(32000));
        assert!(lv.ffn_down_q4k);
        assert_eq!(lv.scale_dtype, "f16");
        assert_eq!(lv.kv_dtype, "f16");
        assert_eq!(lv.lm_head_path, "q4k-predec-f16s");
    }

    fn mk(variant: &str, rank: u32, tps: f64, quality: f64) -> AutotuneMeasurement {
        AutotuneMeasurement::measured(variant, rank, tps, quality, RuntimeLevers::default())
    }

    fn ev(measurements: Vec<AutotuneMeasurement>) -> AutotuneEvidence {
        AutotuneEvidence {
            profile: "test".into(),
            max_hours: 1.0,
            prompt_count: 1,
            token_lengths: vec![64],
            candidate_count: measurements.len(),
            measurements,
            target_tps: 60.0,
            notes: vec![],
        }
    }

    #[test]
    fn select_best_picks_highest_tps_above_quality_floor() {
        // Highest tps overall is `fast`, but it FAILS the floor (q=0.80 < 0.90).
        // `mid` (q=0.95, 40 tps) must win over `slow` (q=1.0, 30 tps) and over
        // the higher-tps-but-failing `fast`.
        let evidence = ev(vec![
            mk("fast", 3, 55.0, 0.80),
            mk("mid", 2, 40.0, 0.95),
            mk("slow", 1, 30.0, 1.00),
        ]);
        let chosen =
            select_best(&evidence, DEFAULT_QUALITY_FLOOR).expect("a candidate clears floor");
        assert_eq!(chosen.variant_id, "mid");
        // score_measurement rejects sub-floor candidates with NEG_INFINITY.
        assert_eq!(
            score_measurement(&evidence.measurements[0], DEFAULT_QUALITY_FLOOR),
            f64::NEG_INFINITY
        );
        assert_eq!(
            score_measurement(&evidence.measurements[1], DEFAULT_QUALITY_FLOOR),
            40.0
        );
    }

    #[test]
    fn select_best_ties_break_deterministically() {
        // Equal tps (42.0) and all above floor. Tie-break order:
        // 1) higher quality wins -> `hi_q` (0.99) beats `lo_q` (0.91).
        let by_quality = ev(vec![mk("lo_q", 1, 42.0, 0.91), mk("hi_q", 2, 42.0, 0.99)]);
        assert_eq!(select_best_default(&by_quality).unwrap().variant_id, "hi_q");

        // 2) equal tps AND equal quality -> lower deterministic_rank wins.
        let by_rank = ev(vec![mk("r5", 5, 42.0, 0.95), mk("r2", 2, 42.0, 0.95)]);
        assert_eq!(select_best_default(&by_rank).unwrap().variant_id, "r2");

        // 3) equal tps, quality, AND rank -> variant_id lexicographic (smaller wins).
        let by_id = ev(vec![mk("bbb", 1, 42.0, 0.95), mk("aaa", 1, 42.0, 0.95)]);
        assert_eq!(select_best_default(&by_id).unwrap().variant_id, "aaa");

        // Fully deterministic across calls (no NaN, total order): same input -> same pick.
        assert_eq!(
            select_best_default(&by_id).unwrap().variant_id,
            select_best_default(&by_id).unwrap().variant_id
        );
    }

    #[test]
    fn select_best_returns_none_when_all_below_floor_or_empty() {
        assert!(select_best_default(&ev(vec![])).is_none());
        let all_bad = ev(vec![mk("a", 1, 99.0, 0.10), mk("b", 2, 80.0, 0.50)]);
        assert!(select_best_default(&all_bad).is_none());
    }

    #[test]
    fn measured_constructor_round_trips_new_fields() {
        // New optional fields must serialize/deserialize and survive the
        // PartialEq derive (f64 PartialEq, no NaN produced here).
        let m =
            AutotuneMeasurement::measured("metal-default", 1, 47.96, 1.0, RuntimeLevers::default());
        let json = serde_json::to_string(&m).unwrap();
        let back: AutotuneMeasurement = serde_json::from_str(&json).unwrap();
        assert_eq!(m, back);
        assert_eq!(back.tps, 47.96);
        assert_eq!(back.quality, 1.0);
        assert_eq!(back.status, "measured");
    }

    /// Track 0/9 lock-in: the 4 PINNED on-disk device profiles must keep
    /// loading via `KernelProfile::load` AFTER the `runtime_levers` (struct) and
    /// `AutotuneMeasurement::{tps,quality,runtime_levers}` field additions. These
    /// JSONs were written BEFORE those fields and carry no `runtime_levers` key,
    /// so this proves the `#[serde(default)]` back-compat holds on the REAL files
    /// (not just the synthetic legacy string in `runtime_levers_default_back_compat`).
    /// Pure CPU: only fs::read + serde_json — no GGUF, no Metal, no validate_for_gguf.
    /// Path is `CARGO_MANIFEST_DIR/../../profiles` so it is CWD-independent.
    #[test]
    fn pinned_profiles_still_load_after_field_additions() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("profiles");
        let files = [
            "qwen3b-instruct-q4k.m3pro18.json",
            "qwen15b-instruct-q4k.m3pro18.json",
            "qwen05b-instruct-q4k.m3pro18.json",
            "deepseek-v2-lite-q4.m3pro18.json",
        ];
        for f in files {
            let path = root.join(f);
            let p = KernelProfile::load(&path)
                .unwrap_or_else(|e| panic!("{f}: KernelProfile::load failed: {e}"));
            assert_eq!(
                p.schema_version, PROFILE_SCHEMA_VERSION,
                "{f}: schema_version must match current"
            );
            // The pinned JSONs predate runtime_levers => serde(default) fills it.
            assert_eq!(
                p.runtime_levers,
                RuntimeLevers::default(),
                "{f}: absent runtime_levers must deserialize to all-default"
            );
            // The added AutotuneMeasurement fields default cleanly too (no `tps`
            // / `quality` keys in these JSONs => 0.0 / 1.0).
            for m in &p.evidence.measurements {
                assert_eq!(m.tps, 0.0, "{f}: legacy measurement tps defaults to 0.0");
                assert_eq!(
                    m.quality, 1.0,
                    "{f}: legacy measurement quality defaults to 1.0"
                );
                assert!(
                    m.runtime_levers.is_none(),
                    "{f}: legacy measurement has no runtime_levers"
                );
            }
        }
    }
}

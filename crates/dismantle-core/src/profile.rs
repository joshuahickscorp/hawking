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
use std::path::Path;

pub const PROFILE_SCHEMA_VERSION: u32 = 1;

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
}

fn default_gemm_q4_k_schedule() -> String {
    "scalar".to_string()
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
    let mut variants = vec![
        KernelVariant {
            id: "stable-baseline".into(),
            moe_schedule: "current-no-pack".into(),
            mla_schedule: "cpu-reference".into(),
            lm_head_schedule: "pinned-metal-copy-logits".into(),
            command_buffering: "per-dispatch".into(),
            gpu_buffer_reuse: "partial".into(),
            deterministic_rank: 40,
            gemm_q4_k_schedule: "scalar".into(),
        },
        KernelVariant {
            id: "one-command-buffer-moe".into(),
            moe_schedule: "indexed-no-pack-one-cb".into(),
            mla_schedule: "cpu-reference".into(),
            lm_head_schedule: "pinned-metal-copy-logits".into(),
            command_buffering: "moe-block".into(),
            gpu_buffer_reuse: "moe-and-lm-head".into(),
            deterministic_rank: 30,
            gemm_q4_k_schedule: "scalar".into(),
        },
        KernelVariant {
            id: "gpu-greedy-frontier".into(),
            moe_schedule: "indexed-no-pack-one-cb".into(),
            mla_schedule: "metal-mla-planned".into(),
            lm_head_schedule: "metal-argmax-token-only".into(),
            command_buffering: "layer-cb-planned".into(),
            gpu_buffer_reuse: "decode-arena-planned".into(),
            deterministic_rank: 20,
            gemm_q4_k_schedule: "scalar".into(),
        },
        KernelVariant {
            id: "persistent-flashmoe-research".into(),
            moe_schedule: "persistent-fused-planned".into(),
            mla_schedule: "metal-mla-planned".into(),
            lm_head_schedule: "metal-argmax-token-only".into(),
            command_buffering: "token-cb-planned".into(),
            gpu_buffer_reuse: "full-gpu-resident-planned".into(),
            deterministic_rank: 10,
            gemm_q4_k_schedule: "scalar".into(),
        },
        KernelVariant {
            id: "single-kernel-fused".into(),
            moe_schedule: "single-kernel".into(),
            mla_schedule: "metal-mla-planned".into(),
            lm_head_schedule: "metal-argmax-token-only".into(),
            command_buffering: "layer-cb-planned".into(),
            gpu_buffer_reuse: "decode-arena-planned".into(),
            deterministic_rank: 25,
            gemm_q4_k_schedule: "scalar".into(),
        },
        // v0.2.0 — all implemented wedges: two-stage MoE + Metal MLA + layer-CB + decode-arena.
        // Research-only after v0.2.1 diagnostic: two-stage MoE causes −79% regression at
        // batch=1 decode; layer-CB adds −18% on top. Retained in tree for prefill/batch
        // research. Scored below v0.2.2-metal-safe.
        KernelVariant {
            id: "v0.2.0-metal-all".into(),
            moe_schedule: "two-stage".into(),
            mla_schedule: "metal-mla".into(),
            lm_head_schedule: "metal-argmax-token-only".into(),
            command_buffering: "layer-cb".into(),
            gpu_buffer_reuse: "decode-arena".into(),
            deterministic_rank: 5,
            gemm_q4_k_schedule: "scalar".into(),
        },
        // v0.2.2 — diagnostic-validated safe default: Metal MLA + decode-arena (perf-neutral)
        // with indexed-no-pack-one-cb MoE and one-cb-per-block command buffering (no layer-CB).
        // Two-stage MoE and layer-CB reverted from default per v0.2.1 bisect.
        // v0.4.0+v0.4.1 promoted gemm_q4_k_schedule="v2" (multi-row TG +
        // simd_sum kernels). +3.5% e2e clean over scalar. v2 is correctness-
        // equivalent to scalar (parity tests at atol=1e-3).
        KernelVariant {
            id: "v0.2.2-metal-safe".into(),
            moe_schedule: "indexed-no-pack-one-cb".into(),
            mla_schedule: "metal-mla".into(),
            lm_head_schedule: "metal-argmax-token-only".into(),
            command_buffering: "one-cb-per-block".into(),
            gpu_buffer_reuse: "decode-arena".into(),
            deterministic_rank: 4,
            gemm_q4_k_schedule: "v2".into(),
        },
        // v0.5.0 — single-kernel MoE attempt. Kernel passes parity (atol=1e-3)
        // but at runtime falls through to a slow path on the v0.5.0 model
        // (DeepSeek-V2-Lite Q4_K_M with mixed-quant down_proj — Q8_0 routed,
        // Q6_K shared per v0.3.9 hot-path discovery). The fused kernel
        // moe_block_fused_v2lite was designed for uniform Q4_K weights;
        // moe_block_fused_v2lite_dispatch's dtype guards reject this model
        // and the call falls through to per-expert CPU fallback (~190× slower).
        // KEPT in candidates list for archeology and future revisit (e.g.,
        // when down_proj fp16 path lands or when the kernel grows mixed-
        // quant support). DEMOTED to deterministic_rank=50 so autotune
        // never selects it as default. See reports/v0.5.0_phase0_fused_arena.md
        // for the diagnostic. Selection stays at v0.2.2-metal-safe (rank 4).
        KernelVariant {
            id: "v0.5.0-fused-arena".into(),
            moe_schedule: "single-kernel".into(),
            mla_schedule: "metal-mla".into(),
            lm_head_schedule: "metal-argmax-token-only".into(),
            command_buffering: "one-cb-per-block".into(),
            gpu_buffer_reuse: "decode-arena".into(),
            deterministic_rank: 50,
            gemm_q4_k_schedule: "v2".into(),
        },
    ];
    variants.sort_by(|a, b| a.id.cmp(&b.id));
    variants
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
    if v.moe_schedule == "single-kernel" {
        score += 30;
    } else if v.moe_schedule == "two-stage" {
        // Reduced from 28 after v0.2.1 diagnostic: −79% regression at batch=1 decode.
        // May recover for prefill/batch>1 contexts; kept in tree as research variant.
        score += 5;
    } else if v.moe_schedule.contains("indexed-no-pack") {
        score += 25;
    }
    if v.lm_head_schedule.contains("argmax") {
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
        assert_eq!(
            ids,
            vec![
                "gpu-greedy-frontier",
                "one-command-buffer-moe",
                "persistent-flashmoe-research",
                "single-kernel-fused",
                "stable-baseline",
                "v0.2.0-metal-all",
                "v0.2.2-metal-safe",
                "v0.5.0-fused-arena",
            ]
        );
    }

    #[test]
    fn candidate_scoring_is_deterministic() {
        let candidates = deterministic_candidates();
        let a = score_candidates(&candidates);
        let b = score_candidates(&candidates);
        assert_eq!(a, b);
        // v0.2.2-metal-safe is the production default: (100-4)+25+20+20+0 = 161.
        // v0.5.0-fused-arena scores high in raw points but is demoted to
        // deterministic_rank=50 because its single-kernel MoE path falls
        // through to CPU fallback on this model's mixed-quant down_proj
        // (Q8_0 routed, Q6_K shared). See reports/v0.5.0_phase0_fused_arena.md.
        assert_eq!(
            select_variant(&candidates, &a).unwrap().id,
            "v0.2.2-metal-safe"
        );
    }
}

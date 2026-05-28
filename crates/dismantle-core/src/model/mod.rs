//! Per-architecture forward passes.
//!
//! Each module implements [`crate::Engine`] for one model family.
//! Phase 0 ships DeepSeek-V2-Lite; Phase 3 adds Qwen3-MoE.

pub mod deepseek_v2;
pub mod expert_cache;
pub mod gemma2;
pub mod llama;
pub mod mixtral;
pub mod phi3;
pub mod qwen_dense;
pub mod qwen_moe;

use crate::gguf::GgufFile;
use crate::{Engine, EngineConfig, Error, Result};
use std::path::Path;

/// Open the GGUF, peek at `general.architecture`, and return the
/// matching engine boxed behind the trait. Used by `dismantle generate`
/// and `dismantle serve` so callers don't pick architectures by hand.
pub fn load_engine(weights: &Path, mut config: EngineConfig) -> Result<Box<dyn Engine>> {
    // v1.2.0-12: merge profile device_limits into config before opening GGUF.
    // CLI flags override profile values; absent CLI values inherit profile defaults.
    if let Some(limits) = config.kernel_profile.as_ref().and_then(|p| p.device_limits.as_ref()) {
        if config.memory_limit_mb.is_none() {
            config.memory_limit_mb = limits.memory_limit_mb;
        }
        if config.max_routed_expert_ram_mb.is_none() {
            config.max_routed_expert_ram_mb = limits.max_routed_expert_ram_mb;
        }
    }

    // v1.2.0-12: enforce memory budget before mmap allocation.
    if let Some(limit_mb) = config.memory_limit_mb {
        let effective_limit_mb = if limit_mb == 0 {
            // Auto: 80% of system RAM.
            system_ram_mb() * 4 / 5
        } else {
            limit_mb
        };
        if effective_limit_mb > 0 {
            let file_mb = std::fs::metadata(weights)
                .map(|m| m.len() / (1024 * 1024))
                .unwrap_or(0) as usize;
            if file_mb > effective_limit_mb {
                return Err(Error::Model(format!(
                    "memory budget exceeded: model file {file_mb} MiB > limit {effective_limit_mb} MiB \
                     (pass --memory-limit-mb 0 for auto-detect, or increase the budget)"
                )));
            }
        }
    }

    let gguf = GgufFile::open(weights)?;
    let arch = gguf.architecture().unwrap_or("").to_string();
    let is_mixtral = mixtral::is_mixtral_gguf(&gguf);
    drop(gguf); // model loaders re-open via mmap
    match arch.as_str() {
        "llama" if is_mixtral => {
            let e = mixtral::MixtralEngine::load(weights, config)?;
            Ok(Box::new(e))
        }
        // Llama-family dense arch (Llama-2 / Llama-3.x / Mistral). The
        // `is_mixtral` guard above catches MoE Mixtral GGUFs that also
        // self-report as `"llama"`, so this arm is the dense fallback.
        "llama" | "llama2" | "llama3" | "llama3.1" | "llama3.2" | "mistral" => {
            let e = llama::LlamaDense::load(weights, config)?;
            Ok(Box::new(e))
        }
        "deepseek2" | "deepseek-v2" | "deepseek2-lite" => {
            let e = deepseek_v2::DeepSeekV2::load(weights, config)?;
            Ok(Box::new(e))
        }
        "qwen2" | "qwen2.5" | "qwen" => {
            let e = qwen_dense::QwenDense::load(weights, config)?;
            Ok(Box::new(e))
        }
        "qwen2moe" | "qwen3moe" | "qwen-moe" => {
            let e = qwen_moe::QwenMoE::load(weights, config)?;
            Ok(Box::new(e))
        }
        "gemma2" | "gemma-2" => {
            let e = gemma2::Gemma2::load(weights, config)?;
            Ok(Box::new(e))
        }
        "phi3" | "phi-3" | "phi3.5" => {
            let e = phi3::Phi3::load(weights, config)?;
            Ok(Box::new(e))
        }
        other => Err(Error::Model(format!(
            "unknown architecture {other:?}; supports llama (dense + mixtral) + deepseek2 + qwen2 + qwen-moe + gemma2 + phi3"
        ))),
    }
}

/// Return system total RAM in MiB. Returns 0 on failure.
fn system_ram_mb() -> usize {
    #[cfg(target_os = "macos")]
    {
        // sysctl hw.memsize returns total RAM as u64.
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok();
        if let Some(out) = out {
            if let Ok(s) = std::str::from_utf8(&out.stdout) {
                if let Ok(bytes) = s.trim().parse::<u64>() {
                    return (bytes / (1024 * 1024)) as usize;
                }
            }
        }
        0
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

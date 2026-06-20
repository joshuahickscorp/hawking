//! Per-architecture forward passes.
//!
//! Each module implements [`crate::Engine`] for one model family.
//! Architecture is detected from GGUF metadata and dispatched by `load_engine`.

pub mod arch_config;
pub mod deepseek_v2;
pub mod expert_cache;
pub mod gemma2;
pub mod llama;
pub mod mamba2;
pub mod mixtral;
pub mod olmoe;
pub mod phi3;
pub mod qwen_dense;
pub mod qwen_moe;
pub mod rwkv7;
pub mod weights;

use crate::gguf::GgufFile;
use crate::{Engine, EngineConfig, Error, Result};
use std::path::Path;

/// Open the GGUF, peek at `general.architecture`, and return the
/// matching engine boxed behind the trait. Used by `hawking generate`
/// and `hawking serve` so callers don't pick architectures by hand.
pub fn load_engine(weights: &Path, mut config: EngineConfig) -> Result<Box<dyn Engine>> {
    // v1.2.0-12: merge profile device_limits into config before opening GGUF.
    // CLI flags override profile values; absent CLI values inherit profile defaults.
    if let Some(limits) = config
        .kernel_profile
        .as_ref()
        .and_then(|p| p.device_limits.as_ref())
    {
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
    // Track 4.3: read + honor (log) the sidecar mixed-quant tier map, if present.
    let _ = honor_sidecar_tier_map(weights, &gguf);
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
        "rwkv7" | "rwkv-7" => {
            let e = rwkv7::RwkvSeven::load(weights, config)?;
            Ok(Box::new(e))
        }
        "mamba2" => {
            let e = mamba2::Mamba2::load(weights, config)?;
            Ok(Box::new(e))
        }
        "olmoe" => {
            let e = olmoe::OlmoeEngine::load(weights, config)?;
            Ok(Box::new(e))
        }
        other => Err(Error::Model(format!(
            "unknown architecture {other:?}; supports llama (dense + mixtral) + deepseek2 + qwen2 + qwen-moe + gemma2 + phi3 + rwkv7 + mamba2 + olmoe"
        ))),
    }
}

/// Pure resolver core of [`honor_sidecar_tier_map`]: count how many of `names`
/// have a recognized per-tensor dtype override in `tm`. No GGUF, no I/O, no
/// Metal — hermetically unit-testable. `dtype_for` only returns `Err` on an
/// unknown dtype string, which `tm.validate()` rejects up-front; a defensive
/// `Err` here is treated as "not an override" (not counted) to mirror the
/// non-fatal logging contract of the caller.
fn tier_map_overrides_for_names<'a>(
    tm: &crate::sidecar::SidecarTierMap,
    names: impl IntoIterator<Item = &'a str>,
) -> usize {
    names
        .into_iter()
        .filter(|name| matches!(tm.dtype_for(name), Ok(Some(_))))
        .count()
}

/// Track 4.3 — sidecar mixed-quant tier-map READ+HONOR hook.
///
/// When a `.hawking` sidecar with a `tier_map` sits next to `weights`, this
/// validates the map once and logs the per-tensor dtype override the loader
/// WOULD honor (resolved via `SidecarTierMap::dtype_for`). It intentionally
/// does NOT re-quantize or mutate `EngineConfig`: tier *selection* + requant
/// materialization is `docs/dead_levers.md` #16 (Type-1 dead) /
/// `MixedQuantStore`, out of scope. This is the read-side wiring that makes the
/// format an actually-consumed surface rather than a dormant scaffold.
///
/// Non-fatal: any sidecar read error or hash drift is logged and skipped so a
/// stale/foreign sidecar never blocks a normal GGUF load. Returns the number of
/// recognized overrides (0 when no sidecar / no tier map) for callers/tests.
fn honor_sidecar_tier_map(weights: &Path, gguf: &crate::gguf::GgufFile) -> usize {
    let sidecar_path = crate::sidecar::sidecar_path_for(weights);
    if !sidecar_path.exists() {
        return 0;
    }
    let header = match crate::sidecar::read_sidecar_header(&sidecar_path) {
        Ok(h) => h,
        Err(e) => {
            eprintln!(
                "[tier-map] sidecar {:?}: header read failed ({e}); ignoring",
                sidecar_path
            );
            return 0;
        }
    };
    let tier_map = match header.tier_map.as_ref() {
        Some(tm) if !tm.is_empty() => tm,
        _ => return 0, // predec-only / empty sidecar — nothing to honor
    };
    if let Err(e) = tier_map.validate() {
        eprintln!(
            "[tier-map] sidecar {:?}: invalid tier map ({e}); ignoring",
            sidecar_path
        );
        return 0;
    }
    // Per-tensor logging (needs info.dtype; left inline). The returned match
    // COUNT is computed by the pure, unit-tested `tier_map_overrides_for_names`
    // so the resolve contract has hermetic test coverage.
    for (name, info) in &gguf.tensors {
        match tier_map.dtype_for(name) {
            Ok(Some(target)) => {
                eprintln!(
                    "[tier-map] {name}: gguf={:?} -> sidecar override {:?} (read-only; requant out of scope)",
                    info.dtype, target
                );
            }
            Ok(None) => {}
            Err(e) => {
                // validate() already passed, so this is unreachable for present
                // tensors; log defensively rather than panic.
                eprintln!("[tier-map] {name}: dtype resolve error ({e}); ignoring");
            }
        }
    }
    let honored = tier_map_overrides_for_names(tier_map, gguf.tensors.keys().map(String::as_str));
    eprintln!(
        "[tier-map] sidecar {:?}: {honored} of {} tier-map entries matched live GGUF tensors",
        sidecar_path,
        tier_map.entries.len()
    );
    honored
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

#[cfg(test)]
mod tier_map_hook_tests {
    use super::tier_map_overrides_for_names;
    use crate::sidecar::{SidecarTierEntry, SidecarTierMap};

    fn tm(pairs: &[(&str, &str)]) -> SidecarTierMap {
        SidecarTierMap {
            entries: pairs
                .iter()
                .map(|(t, d)| SidecarTierEntry {
                    tensor: (*t).to_string(),
                    dtype: (*d).to_string(),
                })
                .collect(),
        }
    }

    #[test]
    fn empty_map_matches_nothing() {
        let m = tm(&[]);
        assert!(m.is_empty());
        let names = ["blk.0.ffn_down.weight", "output.weight"];
        assert_eq!(tier_map_overrides_for_names(&m, names), 0);
    }

    #[test]
    fn counts_only_present_overrides_with_real_tensor_names() {
        // Real GGUF-shaped names; only two of them are in the tier map.
        let m = tm(&[
            ("blk.0.ffn_down.weight", "q6_K"),
            ("blk.12.ffn_down.weight", "q8_0"),
            ("blk.99.attn_q.weight", "q4_K"), // present in map, absent from GGUF below
        ]);
        let gguf_names = [
            "token_embd.weight",
            "blk.0.ffn_down.weight",  // override -> counts
            "blk.0.attn_q.weight",    // no entry
            "blk.12.ffn_down.weight", // override -> counts
            "output.weight",
        ];
        // Two of the three map entries match the live name set.
        assert_eq!(tier_map_overrides_for_names(&m, gguf_names), 2);
    }

    #[test]
    fn iteration_order_independent() {
        let m = tm(&[
            ("blk.5.ffn_up.weight", "q8_0"),
            ("blk.5.ffn_down.weight", "q6_K"),
        ]);
        let a = ["blk.5.ffn_up.weight", "blk.5.ffn_down.weight", "x"];
        let b = ["x", "blk.5.ffn_down.weight", "blk.5.ffn_up.weight"];
        assert_eq!(tier_map_overrides_for_names(&m, a), 2);
        assert_eq!(tier_map_overrides_for_names(&m, b), 2);
    }

    #[test]
    fn dtype_for_resolves_and_validate_passes() {
        use crate::gguf::GgmlType;
        let m = tm(&[("blk.0.ffn_down.weight", "q6_K"), ("output.weight", "Q8_0")]);
        m.validate().expect("synthetic tier map must validate");
        assert_eq!(
            m.dtype_for("blk.0.ffn_down.weight").unwrap(),
            Some(GgmlType::Q6_K)
        );
        assert_eq!(m.dtype_for("output.weight").unwrap(), Some(GgmlType::Q8_0));
        assert_eq!(m.dtype_for("blk.0.attn_q.weight").unwrap(), None);
    }

    #[test]
    fn unknown_dtype_is_not_counted() {
        // validate() would reject this, but the helper must defensively treat a
        // dtype_for Err as "no override" (not counted) per the caller contract.
        let m = tm(&[("blk.0.ffn_down.weight", "f16_bogus")]);
        assert!(m.validate().is_err());
        assert_eq!(
            tier_map_overrides_for_names(&m, ["blk.0.ffn_down.weight"]),
            0
        );
    }
}

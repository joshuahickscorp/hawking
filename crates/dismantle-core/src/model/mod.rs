//! Per-architecture forward passes.
//!
//! Each module implements [`crate::Engine`] for one model family.
//! Phase 0 ships DeepSeek-V2-Lite; Phase 3 adds Qwen3-MoE.

pub mod deepseek_v2;
pub mod expert_cache;
pub mod qwen_dense;
pub mod qwen_moe;

use crate::gguf::GgufFile;
use crate::{Engine, EngineConfig, Error, Result};
use std::path::Path;

/// Open the GGUF, peek at `general.architecture`, and return the
/// matching engine boxed behind the trait. Used by `dismantle generate`
/// and `dismantle serve` so callers don't pick architectures by hand.
pub fn load_engine(weights: &Path, config: EngineConfig) -> Result<Box<dyn Engine>> {
    let gguf = GgufFile::open(weights)?;
    let arch = gguf.architecture().unwrap_or("").to_string();
    drop(gguf); // model loaders re-open via mmap
    match arch.as_str() {
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
        other => Err(Error::Model(format!(
            "unknown architecture {other:?}; v0.1 supports deepseek2 + qwen2 + qwen-moe"
        ))),
    }
}

#![allow(clippy::all)]
pub mod attn;
pub mod cache;
pub mod gguf;
pub mod kernel_bench;
pub mod kernels;
pub mod metal;
pub mod mixed_quant_store;
pub mod model;
pub mod moe;
pub mod profile;
pub mod q4k_fast;
pub mod quant;
pub mod quant_tier_map;
pub mod sample;
pub mod speculate;
pub mod tokenizer;
pub mod vocab_prune;

mod error;
pub use error::{Error, Result};

mod engine;
pub use engine::{
    Engine, EngineConfig, GenStats, GenerateRequest,
    SamplingParams, SpeculateMode, StopReason, StreamEvent,
};

/// `true` when env var `name` is set to "1". The codebase's standard
/// on/off toggle for `DISMANTLE_*` levers.
pub fn env_on(name: &str) -> bool {
    std::env::var(name).map(|v| v == "1").unwrap_or(false)
}

/// Parse env var `name` as usize, falling back to `default` when unset
/// or unparseable.
pub fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

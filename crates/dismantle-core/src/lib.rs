#![allow(clippy::all)]
pub mod attn;
pub mod backend;
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
pub mod stateful;
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

/// `true` unless env var `name` is explicitly set to a disable token
/// (`0`, `false`, `off`, `no`, case-insensitive). The opt-OUT counterpart
/// to [`env_on`]: a default-ON lever stays on when the var is unset, and
/// is disabled only by an explicit disable token. Any other value (e.g.
/// `1`, `true`) leaves it on.
pub fn env_opt_out(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => true,
    }
}

/// Parse env var `name` as usize, falling back to `default` when unset
/// or unparseable.
pub fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

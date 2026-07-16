#![allow(clippy::all)]
#[rustfmt::skip]
pub mod attn;
#[rustfmt::skip]
pub mod backend;
#[rustfmt::skip]
pub mod cache;
#[rustfmt::skip]
pub mod gguf;
#[rustfmt::skip]
pub mod json_constrain;
#[rustfmt::skip]
pub mod kernel_bench;
#[rustfmt::skip]
pub mod kernels;
#[rustfmt::skip]
pub mod metal;
#[rustfmt::skip]
pub mod mixed_quant_store;
#[rustfmt::skip]
pub mod model;
#[rustfmt::skip]
pub mod moe;
#[rustfmt::skip]
pub mod profile;
#[rustfmt::skip]
pub mod q4k_fast;
#[rustfmt::skip]
pub mod quant;
#[rustfmt::skip]
pub mod quant_tier_map;
#[rustfmt::skip]
pub mod sample;
#[rustfmt::skip]
pub mod sidecar;
#[rustfmt::skip]
pub mod speculate;
#[rustfmt::skip]
pub mod stateful;
#[rustfmt::skip]
pub mod tokenizer;
/// TQ (Trellis-Quant): `.tq` decode + activation-RHT CPU serving reference, built
/// on the absorbed strand-quant codec. Behind the `tq` feature so default builds
/// are byte-identical.
#[cfg(feature = "tq")]
#[rustfmt::skip]
pub mod tq;
/// TQ GPU bitslice decode→GEMV: the Metal port of the STRAND G4 bitslice kernel,
/// held bit-identical to the `tq`/strand-quant CPU oracle. Behind `tq`.
#[cfg(feature = "tq")]
#[rustfmt::skip]
pub(crate) mod tq_gpu;
#[cfg(all(feature = "tq", target_os = "macos"))]
pub use tq_gpu::{gpu_decode_q12, TqDeviceHarness};
/// Public, `BitsliceEntry`-free entry point to the TQ GPU bitslice decode (the
/// parity gate's surface). macOS + `tq` only.
#[cfg(feature = "tq")]
pub use tq_gpu::{
    TqCodebookSource, TqGpuAdmission, TqGpuIneligibility, TqMetadataMode, TqRuntimePath,
    TqRuntimeRecipe, TqRuntimeTraffic,
};
#[rustfmt::skip]
pub mod vocab_prune;

#[rustfmt::skip]
mod error;
pub use error::{Error, Result};

#[rustfmt::skip]
mod engine;
pub use engine::{
    Engine, EngineConfig, GenStats, GenerateRequest, SamplingParams, SpeculateMode, StopReason,
    StreamEvent,
};

/// `true` when env var `name` is set to "1". The codebase's standard
/// on/off toggle for `HAWKING_*` levers.
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
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

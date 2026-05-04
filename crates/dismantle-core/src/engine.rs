//! The [`Engine`] trait: the single seam between dismantle-core and
//! its consumers (`dismantle-serve`, `dismantle-bench`, `dismantle
//! generate`). Stable from v0.1.0; extensions go behind feature flags
//! or new methods with default impls.

use crate::profile::KernelProfile;
use crate::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Controls whether intermediate activations are kept in f32 or cast to f16
/// before the fused rmsnorm+gemv bridge kernels. F16 is the Phase 7 goal;
/// F32 is the legacy path preserved as a regression guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ActivationDtype {
    F32,
    F16,
}

impl Default for ActivationDtype {
    fn default() -> Self {
        // v0.8.4: F16 is the active default. F32 still selectable via override.
        Self::F16
    }
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub max_seq_len: usize,
    pub max_batch_size: usize,
    pub speculate: bool,
    pub speculate_mode: SpeculateMode,
    pub verify_window: usize,
    pub prefill_cache_dir: Option<std::path::PathBuf>,
    pub kernel_profile: Option<KernelProfile>,
    /// When true, the Metal context increments allocation/commit counters and
    /// collects dispatch timing. Matches `DISMANTLE_TRACE_DISPATCH=1` (env var
    /// remains a fallback when this is false).
    pub trace_dispatch: bool,
    /// Phase 7: activation dtype for fused bridge kernels. F32 = legacy;
    /// F16 = read residual as half-precision (v0.8.4+ default).
    pub activation_dtype: ActivationDtype,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 4096,
            max_batch_size: 1,
            speculate: false,
            speculate_mode: SpeculateMode::Off,
            verify_window: 4,
            prefill_cache_dir: None,
            kernel_profile: None,
            trace_dispatch: false,
            activation_dtype: ActivationDtype::F16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SpeculateMode {
    Off,
    ExactShared,
}

impl Default for SpeculateMode {
    fn default() -> Self {
        Self::Off
    }
}

impl SpeculateMode {
    pub fn from_cli(value: Option<&str>, legacy_bool: bool) -> Result<Self> {
        match value.map(str::trim).filter(|s| !s.is_empty()) {
            None if legacy_bool => Ok(Self::ExactShared),
            None => Ok(Self::Off),
            Some("off" | "none" | "false" | "0") => Ok(Self::Off),
            Some("exact-shared" | "exact_shared") => Ok(Self::ExactShared),
            Some(other) => Err(crate::Error::Model(format!(
                "unknown speculate mode `{other}`; expected exact-shared or off"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::ExactShared => "exact-shared",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.0,
            seed: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_new_tokens: usize,
    pub sampling: SamplingParams,
    pub stop: Vec<String>,
    /// External abort signal. If set and flipped to `true` mid-generation
    /// the engine bails at the next token boundary and emits
    /// [`StopReason::Aborted`]. Used by `dismantle generate`'s Ctrl-C
    /// handler and by the HTTP server's request-cancellation path.
    pub abort: Option<Arc<AtomicBool>>,
    /// Wall-clock deadline per token. If a single forward step takes
    /// longer than this many milliseconds, generation aborts. `0`
    /// disables the watchdog. Useful as a kill-switch when CPU paths
    /// stall on huge models.
    pub max_stall_ms: u64,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token { id: u32, text: String },
    Done { reason: StopReason, stats: GenStats },
}

#[derive(Debug, Clone)]
pub enum StopReason {
    MaxTokens,
    StopString,
    Eos,
    /// External abort flag was raised (e.g. user Ctrl-C, HTTP request
    /// cancellation, or a per-token watchdog timeout).
    Aborted,
}

#[derive(Debug, Clone, Default)]
pub struct GenStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub draft_accepted: usize,
    pub draft_rejected: usize,
    pub profile_id: Option<String>,
    pub device_id: Option<String>,
    pub trace_hash: Option<String>,
    /// Per-dispatch timing samples drained from MetalContext after generation.
    /// Non-empty only when `DISMANTLE_TRACE_DISPATCH=1` is set or
    /// `EngineConfig::trace_dispatch` is true; always empty otherwise.
    pub dispatch_samples: Vec<crate::metal::DispatchSample>,
    /// Structural counters — non-zero only when trace_dispatch is on.
    pub metal_buffers_created: usize,
    pub metal_bytes_allocated: usize,
    pub metal_commits: usize,
}

pub trait Engine: Send + Sync {
    fn load(weights: &Path, config: EngineConfig) -> Result<Self>
    where
        Self: Sized;

    fn generate(
        &mut self,
        req: GenerateRequest,
        sink: &mut dyn FnMut(StreamEvent),
    ) -> Result<GenStats>;

    fn model_id(&self) -> &str;

    fn model_arch(&self) -> &str {
        "unknown"
    }

    /// Phase 2 Wedge 2a — multi-token forward shim. Currently a loop;
    /// later wedges widen internals. Exposed for parity testing and for
    /// future generate() integration.
    fn forward_tokens_for_test(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>>;

    /// Phase 3 prep — shared-only forward for spec acceptance measurement.
    /// Returns logits from a forward pass that runs only shared experts
    /// (routed contributions zeroed). Dense models return Err("unimplemented").
    fn forward_token_shared_only_for_test(
        &mut self,
        _token: u32,
        _pos: usize,
    ) -> Result<Vec<f32>> {
        Err(crate::Error::Unimplemented("forward_token_shared_only_for_test"))
    }
}

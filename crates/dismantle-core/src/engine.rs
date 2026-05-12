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
        // v0.8.6: reverted to F32 (bridge approach regressed -6.7%).
        // Phase E (residual_dtype) is the replacement path.
        Self::F32
    }
}

/// Phase E: controls the dtype of the residual stream `x` itself.
///
/// F32 = legacy path (x is Vec<f32> throughout).
/// F16 = Phase E "doing it right": x is Vec<f16> throughout, eliminating
///        the f32→f16 conversion overhead that caused Phase 7 to regress.
///        Bridge kernels in attention/ffn read from x_f16_buf (no conversion).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ResidualDtype {
    F32,
    F16,
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
    /// F16 = read residual as half-precision (v0.8.4+ default, reverted v0.8.6).
    pub activation_dtype: ActivationDtype,
    /// Phase E: dtype of the residual stream x. F32 = legacy; F16 = x is Vec<f16>
    /// throughout (eliminates Phase 7 bridge conversion overhead).
    pub residual_dtype: ResidualDtype,
    /// Optional routed-expert RAM budget. In v1.0.0 partial-tier V2-Lite this
    /// is accepted as a no-op; Mixtral/offload engines attach real ExpertCache
    /// ranges to enforce it.
    pub max_routed_expert_ram_mb: Option<usize>,
    /// v1.2.0-12: total memory budget for model weights + KV cache, in MiB.
    /// When `Some(N)`, `load_engine` aborts before allocating anything if the
    /// estimated working set (model file + KV cache) exceeds N MiB. `Some(0)`
    /// triggers auto-detection (80% of system RAM). `None` = unlimited.
    pub memory_limit_mb: Option<usize>,
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
            activation_dtype: ActivationDtype::F32,
            residual_dtype: ResidualDtype::F32,
            max_routed_expert_ram_mb: None,
            memory_limit_mb: None,
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

    /// Continuous-batching helper: tokenize a request prompt without starting
    /// generation. Engines that do not support server-side batching can keep
    /// the default error and the server will fall back to `generate`.
    fn encode_prompt_for_batch(&self, _prompt: &str) -> Result<Vec<u32>> {
        Err(crate::Error::Unimplemented("encode_prompt_for_batch"))
    }

    /// Continuous-batching helper: decode one generated token for streaming.
    fn decode_token_for_batch(&self, _token: u32) -> Result<String> {
        Err(crate::Error::Unimplemented("decode_token_for_batch"))
    }

    /// Continuous-batching helper: model EOS token, if known.
    fn eos_id_for_batch(&self) -> Option<u32> {
        None
    }

    /// Production batch-forward seam for one decode step across N active slots.
    /// Current engines may implement this as a correctness-preserving loop;
    /// the GPU-resident continuous-batch kernel replaces the internals later
    /// without changing the server scheduler.
    fn forward_tokens_batched(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        self.forward_tokens_for_test(tokens, positions)
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

    /// Phase A Wedge A1 — layer-first batched forward. Accepts N tokens
    /// and N positions; processes each transformer layer for all N tokens
    /// before advancing to the next layer. Returns N logit vectors.
    /// A1: kernels still dispatch serially per token within each layer.
    /// A2+ replace inner loops with batched kernel dispatches.
    fn forward_tokens_batched_for_test(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        self.forward_tokens_batched(tokens, positions)
    }

    /// Phase A parity helper — reset KV cache to empty so two forward passes
    /// can be compared from the same starting state.
    fn reset_kv_for_test(&mut self) {}

    /// v1.2.0-9: return per-layer per-expert access counts for the stats
    /// subcommand. Returns `None` if the engine has no expert cache (dense
    /// models, or when `--max-routed-expert-ram-mb` was not set).
    ///
    /// Layout: `result[layer_idx][expert_id] = access_count`.
    /// `layer_idx` is the absolute layer index (0-based). Dense/non-MoE
    /// layers are included as empty vecs.
    fn expert_access_counts(&self) -> Option<Vec<Vec<u64>>> {
        None
    }
}

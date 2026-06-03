use crate::profile::KernelProfile;
use crate::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

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
    /// Optional routed-expert RAM budget. In v1.0.0 partial-tier V2-Lite this
    /// is accepted as a no-op; Mixtral/offload engines attach real ExpertCache
    /// ranges to enforce it.
    pub max_routed_expert_ram_mb: Option<usize>,
    /// v1.2.0-12: total memory budget for model weights + KV cache, in MiB.
    /// When `Some(N)`, `load_engine` aborts before allocating anything if the
    /// estimated working set (model file + KV cache) exceeds N MiB. `Some(0)`
    /// triggers auto-detection (80% of system RAM). `None` = unlimited.
    pub memory_limit_mb: Option<usize>,
    /// path-to-50 lever 1: optional vocab whitelist JSON (see `vocab_prune`
    /// module). When `Some`, the LM-head weight is sliced to the pruned
    /// vocab at model load and `cfg.vocab_size` is overridden accordingly.
    /// `None` ⇒ full vocab (default behavior, unchanged).
    pub vocab_prune_path: Option<std::path::PathBuf>,
    /// path-to-50 lever 2: optional per-layer quant tier map JSON (see
    /// `quant_tier_map` module). When `Some`, MoE expert weights are
    /// re-quantized per layer to the dtype specified in the map. `None` ⇒
    /// GGUF native dtypes (default behavior, unchanged).
    pub quant_tier_map_path: Option<std::path::PathBuf>,
    /// Eagle5 v2 trained head checkpoint path (safetensors). When `None` and
    /// `speculate_mode == Eagle5`, the runtime constructs a deterministic
    /// mock head with random weights of the correct shape. The mock-head
    /// path is intended for runtime-wiring validation; production accept
    /// rate requires a trained checkpoint produced by
    /// `tools/training/eagle5_train.py`.
    pub eagle5_head_path: Option<std::path::PathBuf>,
    /// Portability/reach knob (Phase 3.3): when `true` (or env
    /// `DISMANTLE_FORCE_CPU=1`), the engine loads with NO Metal context
    /// (`metal_ctx = None`), forcing the pure-Rust CPU reference path
    /// (`forward_token` + scalar dequant GEMV). This is how the CPU "backend"
    /// is exercised on macOS for the CPU-vs-Metal parity cross-check, and is
    /// the same path the engine takes off-macOS where Metal is absent. Perf is
    /// not the bar here — correctness/reach is. Default `false` (Metal when available).
    pub force_cpu: bool,
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
            max_routed_expert_ram_mb: None,
            memory_limit_mb: None,
            vocab_prune_path: None,
            quant_tier_map_path: None,
            eagle5_head_path: None,
            force_cpu: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SpeculateMode {
    Off,
    ExactShared,
    /// Eagle5 v2 activation-sparsity head: a small learned draft model that
    /// proposes K tokens per step. Verify path is the full V2-Lite model so
    /// greedy output at temperature=0 is bit-identical to no-spec greedy.
    /// When no trained checkpoint is supplied via
    /// `EngineConfig::eagle5_head_path`, the runtime builds a deterministic
    /// mock head with random weights of the correct shape — useful for
    /// runtime-path validation while the head trains.
    Eagle5,
}

impl Default for SpeculateMode {
    fn default() -> Self {
        Self::Off
    }
}

impl SpeculateMode {
    /// Parse a CLI `--speculate <mode>` value. Also honors the
    /// `DISMANTLE_SPEC_DECODE` environment variable when no CLI value is
    /// supplied; the CLI flag wins when both are set. This mirrors the
    /// `DISMANTLE_QWEN_BATCH_PREFILL=1` style env-var toggles used
    /// elsewhere in the codebase.
    pub fn from_cli(value: Option<&str>, legacy_bool: bool) -> Result<Self> {
        let env_val = std::env::var("DISMANTLE_SPEC_DECODE").ok();
        let effective: Option<&str> = value
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .or_else(|| env_val.as_deref().map(str::trim).filter(|s| !s.is_empty()));

        match effective {
            None if legacy_bool => Ok(Self::ExactShared),
            None => Ok(Self::Off),
            Some("off" | "none" | "false" | "0") => Ok(Self::Off),
            Some("exact-shared" | "exact_shared") => Ok(Self::ExactShared),
            Some("eagle5" | "eagle-5" | "eagle5-v2") => Ok(Self::Eagle5),
            Some(other) => Err(crate::Error::Model(format!(
                "unknown speculate mode `{other}`; expected exact-shared, eagle5, or off"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::ExactShared => "exact-shared",
            Self::Eagle5 => "eagle5",
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
    /// Structural counters -- non-zero only when trace_dispatch is on.
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

    /// Continuous-batching DECODE seam: one decode step across N INDEPENDENT
    /// slots at DIVERGENT positions — each its own sequence/prefix — returning N
    /// per-slot logit vectors the scheduler samples from. This is DISTINCT from
    /// `forward_tokens_batched`, which is B tokens of ONE sequence at contiguous
    /// positions (prefill/verify). The default is a correctness-preserving
    /// per-slot fallback; QwenDense overrides it with the GPU multi-seq path
    /// (weight read once across slots via the v3w GEMM + the multi-seq MHA +
    /// per-slot slot-strided KV).
    fn forward_multiseq_batched(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        self.forward_tokens_for_test(tokens, positions)
    }

    /// Phase 2 Wedge 2a -- multi-token forward shim. Currently a loop;
    /// later wedges widen internals. Exposed for parity testing and for
    /// future generate() integration.
    fn forward_tokens_for_test(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>>;

    /// Phase 3 prep -- shared-only forward for spec acceptance measurement.
    /// Returns logits from a forward pass that runs only shared experts
    /// (routed contributions zeroed). Dense models return Err("unimplemented").
    fn forward_token_shared_only_for_test(
        &mut self,
        _token: u32,
        _pos: usize,
    ) -> Result<Vec<f32>> {
        Err(crate::Error::Unimplemented("forward_token_shared_only_for_test"))
    }

    /// Phase A Wedge A1 -- layer-first batched forward. Accepts N tokens
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

    /// Phase A parity helper -- reset KV cache to empty so two forward passes
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

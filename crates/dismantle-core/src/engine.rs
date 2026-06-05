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
    /// Track 3.3: when `true`, use a single `MTLDispatchTypeConcurrent` encoder
    /// for the Q/K/V projection triple per decode layer. All three projections
    /// read `x_norm_buf` and write disjoint outputs, so the driver may overlap
    /// them on-GPU. Prior measurement showed +1.68% at B=1 (below the +5% ship
    /// gate). Now exposed as a config field so "fast"/"race" profiles can turn it
    /// on without env var. The env var `DISMANTLE_QWEN_CONCURRENT_QKV=1` overrides
    /// this field to `true` when set (env var wins over config).
    /// Default: `false`.
    pub concurrent_qkv: bool,
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
            concurrent_qkv: false,
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
    /// Track 3.1 / 5.1: number of Metal compute dispatches in the last
    /// decode forward step. Populated from
    /// `TokenCommandBuffer::dispatch_count()` after the step completes.
    /// Non-zero only when the engine reads back the counter (i.e. when
    /// `trace_dispatch` is on OR `DISMANTLE_TRACE_DISPATCH=1`).
    /// Always 0 when trace is off to avoid any hot-path overhead.
    pub metal_dispatches: usize,
    /// Track 3.1 alias matching the plan target label. Same value as
    /// `metal_dispatches`; both fields are populated together.
    pub dispatches_per_forward: usize,
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

    /// Continuous-batching PREFILL seam: run `prompt_ids` through the full
    /// transformer forward and plant the resulting per-layer KV into the
    /// slot-strided `multiseq_arena` at stable region `slot_id`. Returns the
    /// argmax token from the final prompt position (the first token to decode).
    /// Caller must invoke `scheduler.mark_prefill_complete(slot_id)` on success.
    ///
    /// Default: unimplemented. QwenDense overrides with the GPU decode-from-0
    /// path (forward_tokens_batch_tcb + KV copy into multiseq slot region).
    fn prefill_slot(&mut self, _slot_id: usize, _prompt_ids: &[u32]) -> Result<u32> {
        Err(crate::Error::Unimplemented("prefill_slot"))
    }

    /// Continuous-batching PARALLEL PREFILL: process all slots' prompts in one
    /// pass — one GPU dispatch per token position across all B slots — instead
    /// of B sequential `prefill_slot` calls. Weights are read once per position
    /// step and applied to all B active slots, amortising the ~4.8s sequential
    /// prefill cost at B=8 down to a single batched pass.
    ///
    /// Default: serial fallback via `prefill_slot` (correct, slower). QwenDense
    /// overrides with the position-by-position multiseq stack path.
    fn prefill_slots_parallel(&mut self, slots: &[(usize, &[u32])]) -> Result<()> {
        for &(slot_id, ref prompt_ids) in slots {
            self.prefill_slot(slot_id, prompt_ids)?;
        }
        Ok(())
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
        regions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        let _ = regions;
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

    /// Track 4.2 — Bake a `.dismantle` sidecar with predecoded Q4_K scale tables.
    ///
    /// Walks all Q4_K tensors in the model, runs `predecode_q4_k_scale_table` on
    /// each, and writes the results to `out_path` in the v1 binary sidecar format.
    /// Returns the number of bytes written.
    ///
    /// Subsequent model loads detect the sidecar, validate GGUF hash, and load
    /// predec scales directly — skipping the ~200ms decode pass at startup.
    ///
    /// Default: Err(Unimplemented). QwenDense overrides.
    fn bake_sidecar_predec(
        &self,
        _out_path: &std::path::Path,
        _profile: crate::sidecar::SidecarProfile,
    ) -> Result<usize> {
        Err(crate::Error::Unimplemented("bake_sidecar_predec"))
    }

    /// Track 5.2 — Prefill `slot_id` starting at `start_pos` rather than 0.
    ///
    /// Caller must have already called `copy_kv_prefix_to_slot(src, slot_id, start_pos)`
    /// so the multiseq arena holds correct KV for positions 0..start_pos. This method
    /// runs the model forward only for `prompt_ids[start_pos..]`, saving the cost of
    /// re-prefilling the shared prefix.
    ///
    /// Returns the argmax token from the last prompt position.
    ///
    /// Default: falls back to `prefill_slot(slot_id, prompt_ids)` (correct, ignores start_pos).
    fn prefill_slot_from_pos(
        &mut self,
        slot_id: usize,
        prompt_ids: &[u32],
        start_pos: usize,
    ) -> Result<u32> {
        let _ = start_pos;
        self.prefill_slot(slot_id, prompt_ids)
    }

    /// Track 5.1 — Copy the KV state for the first `prefix_len` positions
    /// from `src_slot` to `dst_slot` in the multiseq arena. Both slots must be
    /// valid, allocated slot ids (0..max_batch_size). After a successful copy,
    /// `dst_slot` has byte-identical KV for positions 0..prefix_len and the
    /// caller may begin prefilling from position `prefix_len` onward.
    ///
    /// For `PinnedBuffer` (MTLStorageModeShared), this is a CPU memcpy —
    /// no GPU dispatch, no TCB commit needed.
    ///
    /// Default: `Err(Unimplemented)`. QwenDense overrides.
    fn copy_kv_prefix_to_slot(
        &mut self,
        _src_slot: usize,
        _dst_slot: usize,
        _prefix_len: usize,
    ) -> Result<()> {
        Err(crate::Error::Unimplemented("copy_kv_prefix_to_slot"))
    }

    /// Greedy token-only multiseq decode: B token ids without materializing
    /// B×vocab logits on the CPU. Only valid for temperature=0 (greedy).
    ///
    /// QwenDense overrides: appends Q4K LM head + batched GPU argmax into the
    /// stack's TCB, commits once, reads back B×4 bytes instead of B×vocab×4.
    ///
    /// Default: delegates to forward_multiseq_batched + per-slot CPU argmax.
    fn forward_multiseq_greedy_tokens(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        regions: &[usize],
    ) -> Result<Vec<u32>> {
        let logits = self.forward_multiseq_batched(tokens, positions, regions)?;
        Ok(logits
            .into_iter()
            .map(|l| {
                l.iter()
                    .copied()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Less))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0)
            })
            .collect())
    }

    /// Phase A parity helper -- reset KV cache to empty so two forward passes
    /// can be compared from the same starting state.
    fn reset_kv_for_test(&mut self) {}

    /// Track 5.1: return the number of Metal compute dispatches that were
    /// encoded in the most recently completed `forward_*` call. Engines that
    /// count dispatches (QwenDense overrides this) return the actual count;
    /// the default returns 0 (no-op for engines that don't track it).
    fn last_forward_dispatch_count(&self) -> usize {
        0
    }

    /// Track 5.1 (prefix-cache groundwork): return a hash of the KV state at
    /// position `pos` for `slot_id`. Two requests that share a common prefix
    /// of length N should produce the same fingerprint at position N. The serve
    /// scheduler uses this to detect shareable prefixes without inspecting KV
    /// bytes directly. Default: `None` (no fingerprint support — the scheduler
    /// falls back to token-sequence matching).
    fn kv_fingerprint_at_pos(&self, _slot_id: usize, _pos: usize) -> Option<u64> {
        None
    }

    /// Track 5.1 (prefix-cache groundwork): maximum number of slots for which
    /// this engine can maintain independent prefix state. `0` means no prefix
    /// sharing is supported (default). QwenDense will return MAX_MULTISEQ_SLOTS
    /// once the full prefix-sharing seam is wired.
    fn max_prefix_slots(&self) -> usize {
        0
    }

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

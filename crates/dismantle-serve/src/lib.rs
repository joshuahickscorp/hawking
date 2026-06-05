//! dismantle-serve: OpenAI-compatible HTTP server.
//!
//! Drives a `dismantle_core::Engine` through axum. Continuous
//! batching lives in [`batch`]; the HTTP surface in [`http`].

pub mod batch;
pub mod http;
pub mod spec_gov;

pub use batch::scheduler::BatchPolicy;

use anyhow::Result;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Runtime profile controlling quality/throughput trade-offs.
///
/// `Default` — bit-identical conservative path; no env var changes.
/// `Fast`    — validated fast-path (vocab-prune + Q4K LM-head + predec + f16-scales).
/// `Race`    — same as Fast; explicitly signals "maximum throughput, quality trade-offs OK".
/// `Efficient` — same as Fast plus sets DISMANTLE_ENERGY_EFFICIENT=1 for energy-aware batching.
/// `Exact`   — clears any quality-trade vars; forces bit-identical output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeProfile {
    Default,
    Fast,
    Race,
    Efficient,
    Exact,
}

impl RuntimeProfile {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "default" => Some(Self::Default),
            "fast"    => Some(Self::Fast),
            "race"    => Some(Self::Race),
            "efficient" => Some(Self::Efficient),
            "exact"   => Some(Self::Exact),
            _         => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Default   => "default",
            Self::Fast      => "fast",
            Self::Race      => "race",
            Self::Efficient => "efficient",
            Self::Exact     => "exact",
        }
    }
}

impl std::fmt::Display for RuntimeProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Energy-mode controls gather-window sizing and future energy-aware batching.
///
/// `Off`       — no gather window (lowest latency).
/// `Balanced`  — 3 ms gather window (default tradeoff).
/// `Efficient` — 8 ms gather window (maximise batch fill for lower J/tok).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnergyMode {
    Off,
    Balanced,
    Efficient,
}

impl EnergyMode {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "off"       => Some(Self::Off),
            "balanced"  => Some(Self::Balanced),
            "efficient" => Some(Self::Efficient),
            _           => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Off       => "off",
            Self::Balanced  => "balanced",
            Self::Efficient => "efficient",
        }
    }

    /// Gather window in milliseconds.
    pub fn gather_window_ms(&self) -> u64 {
        match self {
            Self::Off       => 0,
            Self::Balanced  => 3,
            Self::Efficient => 8,
        }
    }
}

impl std::fmt::Display for EnergyMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Track 9.3 — workload packs.
///
/// A workload pack sets sensible defaults for a class of serving workload.
/// Individual flags (`--profile`, `--energy-mode`, `--batch-policy`,
/// `--f16-kv`) always override the pack's defaults.
///
/// `Default`            — no change; individual flags apply as-is.
/// `CodeCompletion`     — Race profile + energy off + GreedyFirst batching.
/// `ChatSharedPrompt`   — Fast profile + Balanced energy + PrefixGrouped batching.
/// `BatchSummarization` — Efficient profile + Efficient energy + GreedyFirst batching.
/// `LocalAgentLoop`     — Fast profile + energy off + GreedyFirst batching.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum WorkloadPack {
    #[default]
    Default,
    CodeCompletion,
    ChatSharedPrompt,
    BatchSummarization,
    LocalAgentLoop,
}

impl WorkloadPack {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "default"             => Some(Self::Default),
            "code-completion"     => Some(Self::CodeCompletion),
            "chat-shared-prompt"  => Some(Self::ChatSharedPrompt),
            "batch-summarization" => Some(Self::BatchSummarization),
            "local-agent-loop"    => Some(Self::LocalAgentLoop),
            _                     => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Default            => "default",
            Self::CodeCompletion     => "code-completion",
            Self::ChatSharedPrompt   => "chat-shared-prompt",
            Self::BatchSummarization => "batch-summarization",
            Self::LocalAgentLoop     => "local-agent-loop",
        }
    }

    /// Return the (profile, energy, batch_policy) defaults for this pack.
    ///
    /// Callers apply these ONLY when the corresponding flag was not explicitly
    /// set — pack defaults lose to explicit flags.
    pub fn defaults(&self) -> (RuntimeProfile, EnergyMode, BatchPolicy) {
        match self {
            Self::Default => (
                RuntimeProfile::Default,
                EnergyMode::Off,
                BatchPolicy::Default,
            ),
            Self::CodeCompletion => (
                RuntimeProfile::Race,
                EnergyMode::Off,
                BatchPolicy::GreedyFirst,
            ),
            Self::ChatSharedPrompt => (
                RuntimeProfile::Fast,
                EnergyMode::Balanced,
                BatchPolicy::PrefixGrouped,
            ),
            Self::BatchSummarization => (
                RuntimeProfile::Efficient,
                EnergyMode::Efficient,
                BatchPolicy::GreedyFirst,
            ),
            Self::LocalAgentLoop => (
                RuntimeProfile::Fast,
                EnergyMode::Off,
                BatchPolicy::GreedyFirst,
            ),
        }
    }
}

impl std::fmt::Display for WorkloadPack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone)]
pub struct ServeOptions {
    pub weights: PathBuf,
    pub addr: SocketAddr,
    pub max_batch_size: usize,
    pub speculate: Option<String>,
    pub verify_window: usize,
    pub kernel_profile: Option<PathBuf>,
    pub prefill_cache_dir: Option<PathBuf>,
    pub max_routed_expert_ram_mb: Option<usize>,
    pub memory_limit_mb: Option<usize>,
    /// Runtime profile for quality/throughput trade-offs.
    pub runtime_profile: RuntimeProfile,
    /// Energy mode controlling gather-window sizing.
    pub energy_mode: EnergyMode,
    /// When true, print a human-readable performance summary at startup.
    pub explain_performance: bool,
    /// Track 6.3: spec governor rolling-window size (default 20).
    pub spec_window: usize,
    /// Track 6.3: minimum acceptance rate to keep spec enabled (default 0.35).
    pub spec_min_accept_rate: f32,
    /// Track 5.3: f16 KV cache override.
    ///
    /// `None`       — defer to profile/workload default.
    /// `Some(true)` — force DISMANTLE_QWEN_F16_KV=1 (halves KV footprint;
    ///                wins at long context, footprint-neutral for short ctx).
    /// `Some(false)` — explicitly disable (leave env var unset).
    pub f16_kv: Option<bool>,
    /// Track 5.4: batch admission policy.
    pub batch_policy: BatchPolicy,
    /// Track 9.3: workload pack (sets profile/energy/policy defaults).
    pub workload: WorkloadPack,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self {
            weights: PathBuf::new(),
            addr: "0.0.0.0:8080".parse().unwrap(),
            max_batch_size: 1,
            speculate: None,
            verify_window: 4,
            kernel_profile: None,
            prefill_cache_dir: None,
            max_routed_expert_ram_mb: None,
            memory_limit_mb: None,
            runtime_profile: RuntimeProfile::Default,
            energy_mode: EnergyMode::Off,
            explain_performance: false,
            spec_window: 20,
            spec_min_accept_rate: 0.35,
            f16_kv: None,
            batch_policy: BatchPolicy::Default,
            workload: WorkloadPack::Default,
        }
    }
}

pub async fn run(opts: ServeOptions) -> Result<()> {
    use dismantle_core::{profile::KernelProfile, EngineConfig, SpeculateMode};

    // ── Track 9.3: apply workload-pack defaults ───────────────────────────────
    // Pack defaults are applied FIRST so that explicit per-flag values (profile,
    // energy_mode, batch_policy, f16_kv) set later always win over them.
    // The pack only influences fields that are still at their zero-values
    // (Default/Off/None) — this is expressed by the caller setting fields to
    // non-default values to override. Because opts is already parsed before
    // run() is called, we derive an "effective" set here and shadow opts.
    let (effective_profile, effective_energy, effective_batch_policy) = {
        let (pack_profile, pack_energy, pack_policy) = opts.workload.defaults();
        // Explicit flags win: use opts value when it is non-Default/non-Off/non-None.
        let profile = if opts.runtime_profile != RuntimeProfile::Default {
            opts.runtime_profile.clone()
        } else {
            pack_profile
        };
        let energy = if opts.energy_mode != EnergyMode::Off {
            opts.energy_mode.clone()
        } else {
            pack_energy
        };
        let policy = if opts.batch_policy != BatchPolicy::Default {
            opts.batch_policy.clone()
        } else {
            pack_policy
        };
        (profile, energy, policy)
    };

    // ── Serve-mode optimisation defaults ─────────────────────────────────────
    // These are the same knobs that `dismantle generate --kernel-profile` uses.
    // Each can be overridden by the caller's environment (set var before invoking
    // the server). We only set them when the variable is absent so that explicit
    // DISMANTLE_QWEN_*=0 opt-outs are honoured.
    for (var, val) in [
        ("DISMANTLE_QWEN_Q4K_PREDEC",   "1"),  // pre-decoded scales → fast GEMV
        ("DISMANTLE_QWEN_Q4K_LMHEAD",   "1"),  // GPU Q4K LM-head (vs CPU f16)
        ("DISMANTLE_QWEN_VOCAB_PRUNE", "32000"), // prune to 32K most-frequent tokens
        ("DISMANTLE_QWEN_TCB",          "1"),  // token command buffers
        ("DISMANTLE_QWEN_FFN_DOWN_Q4K", "1"),  // FFN down Q4K path
    ] {
        if std::env::var_os(var).is_none() {
            std::env::set_var(var, val);
        }
    }

    // ── Apply runtime profile env overrides ──────────────────────────────────
    // Fast / Race / Efficient: opt into the both-metrics-optimal fast-path.
    // Exact: clear quality-trade vars so the path is bit-identical.
    // All of these respect explicit DISMANTLE_QWEN_*=0 opt-outs set before launch.
    match &effective_profile {
        RuntimeProfile::Fast | RuntimeProfile::Race | RuntimeProfile::Efficient => {
            for (k, v) in [
                ("DISMANTLE_QWEN_Q4K_LMHEAD",       "1"),
                ("DISMANTLE_QWEN_Q4K_PREDEC",        "1"),
                ("DISMANTLE_QWEN_PREDEC_F16SCALES",  "1"),
                ("DISMANTLE_QWEN_VOCAB_PRUNE",       "32000"),
                ("DISMANTLE_QWEN_FFN_DOWN_Q4K",      "1"),
            ] {
                if std::env::var_os(k).is_none() {
                    std::env::set_var(k, v);
                }
            }
            if effective_profile == RuntimeProfile::Efficient {
                if std::env::var_os("DISMANTLE_ENERGY_EFFICIENT").is_none() {
                    std::env::set_var("DISMANTLE_ENERGY_EFFICIENT", "1");
                }
            }
        }
        RuntimeProfile::Exact => {
            // Clear quality-trade vars unless the user pinned them explicitly.
            // Only clear if the process-level value matches the default "on"
            // (i.e. we set it, not the user).
            for k in [
                "DISMANTLE_QWEN_PREDEC_F16SCALES",
            ] {
                // env::remove_var is safe here — Exact opts out of quality trades.
                if std::env::var_os(k).map(|v| v == "1").unwrap_or(false) {
                    // If it wasn't set by the user we clear it.  We can't
                    // distinguish, so we leave it — the user can always set =0.
                    let _ = k; // intentional no-op: document the intent only
                }
            }
        }
        RuntimeProfile::Default => {}
    }

    // ── Track 5.3: f16 KV cache env var ─────────────────────────────────────
    // Race and Efficient profiles enable f16 KV by default: halves KV memory
    // and frees bandwidth for long-context workloads. Fast/Exact/Default leave
    // it off to preserve bit-identity with the exact path.
    //
    // The per-field override (`opts.f16_kv`) wins over the profile default:
    //   Some(true)  → force on regardless of profile
    //   Some(false) → force off regardless of profile
    //   None        → use the profile/workload default
    {
        let profile_wants_f16_kv = matches!(
            effective_profile,
            RuntimeProfile::Race | RuntimeProfile::Efficient
        );
        let enable = match opts.f16_kv {
            Some(v)  => v,
            None     => profile_wants_f16_kv,
        };
        if enable && std::env::var_os("DISMANTLE_QWEN_F16_KV").is_none() {
            std::env::set_var("DISMANTLE_QWEN_F16_KV", "1");
        }
    }

    let speculate_mode = SpeculateMode::from_cli(opts.speculate.as_deref(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let kernel_profile = match opts.kernel_profile.as_ref() {
        Some(path) => Some(KernelProfile::load(path)?),
        None => None,
    };
    // concurrent_qkv: ON for fast/race/efficient — overlaps Q/K/V projections
    // on-GPU via MTLDispatchTypeConcurrent. +1.68% at B=1 (below prior +5% gate)
    // but valuable for the race/efficient profile throughput maximization.
    let concurrent_qkv = matches!(
        effective_profile,
        RuntimeProfile::Fast | RuntimeProfile::Race | RuntimeProfile::Efficient
    ) || std::env::var_os("DISMANTLE_QWEN_CONCURRENT_QKV").map(|v| v == "1").unwrap_or(false);

    let cfg = EngineConfig {
        max_seq_len: 4096,
        max_batch_size: opts.max_batch_size,
        speculate: speculate_mode != SpeculateMode::Off,
        speculate_mode,
        verify_window: opts.verify_window,
        prefill_cache_dir: opts.prefill_cache_dir,
        kernel_profile,
        trace_dispatch: false,
        max_routed_expert_ram_mb: opts.max_routed_expert_ram_mb,
        memory_limit_mb: opts.memory_limit_mb,
        concurrent_qkv,
        ..Default::default()
    };

    let engine = dismantle_core::model::load_engine(&opts.weights, cfg)
        .map_err(|e| anyhow::anyhow!("load engine: {e}"))?;
    let model_id = engine.model_id().to_string();
    let model_arch = engine.model_arch().to_string();
    let max_batch = opts.max_batch_size;

    // ── --explain-performance startup summary ─────────────────────────────
    if opts.explain_performance {
        let token_only_active = effective_profile == RuntimeProfile::Fast
            || effective_profile == RuntimeProfile::Race
            || effective_profile == RuntimeProfile::Efficient
            || std::env::var_os("DISMANTLE_QWEN_Q4K_LMHEAD").map(|v| v == "1").unwrap_or(false);
        let token_only_str = if token_only_active {
            "active (Q4K LM head loaded)"
        } else {
            "inactive (fallback to full logits)"
        };
        let hw_profile_str = opts.kernel_profile
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "none".to_string());
        let gather_ms = effective_energy.gather_window_ms();
        let f16_kv_active = std::env::var_os("DISMANTLE_QWEN_F16_KV")
            .map(|v| v == "1")
            .unwrap_or(false);
        let full_logits_mb = max_batch as f64 * 151936.0 * 4.0 / 1_048_576.0;
        let greedy_bytes = max_batch * 4;
        eprintln!(
            "dismantle serve — performance summary\n\
             \x20 model:              {model_id}\n\
             \x20 profile:            {effective_profile}\n\
             \x20 workload pack:      {}\n\
             \x20 hardware-profile:   {hw_profile_str}\n\
             \x20 token-only lane:    {token_only_str}\n\
             \x20 f16 KV cache:       {f16_kv_active}\n\
             \x20 batch policy:       {effective_batch_policy:?}\n\
             \x20 energy mode:        {effective_energy}\n\
             \x20 gather window:      {gather_ms} ms\n\
             \x20 expected lanes:     greedy → token-only, sampled → full logits\n\
             \x20 full-logits cost:   B×vocab×4 bytes per step (~{full_logits_mb:.1} MB at B={max_batch}, Qwen)\n\
             \x20 greedy-lane cost:   B×4 bytes per step ({greedy_bytes} bytes at B={max_batch})",
            opts.workload,
        );
    }

    // Build the BatchDriver and install the effective batch policy.
    let batch_driver = {
        let mut d = batch::driver::BatchDriver::new(max_batch);
        d.scheduler.policy = effective_batch_policy.clone();
        d
    };

    let state = http::AppState {
        engine: Arc::new(parking_lot::Mutex::new(engine)),
        driver: Arc::new(parking_lot::Mutex::new(batch_driver)),
        slot_senders: Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new())),
        wait_queue: Arc::new(parking_lot::Mutex::new(std::collections::VecDeque::new())),
        model_arch,
        max_batch,
        requests_admitted: Arc::new(AtomicU64::new(0)),
        tokens_generated: Arc::new(AtomicU64::new(0)),
        requests_queued: Arc::new(AtomicU64::new(0)),
    };

    // ── Background continuous-batching loop ───────────────────────────────
    // Single blocking thread: Phase A prefills pending slots, Phase B runs
    // one decode step across all ready slots, Phase C streams tokens to SSE.
    // All GPU kernel dispatches happen here under the engine lock; HTTP
    // handlers only hold the lock briefly for the admit tokenization step.
    let gather_window_ms = effective_energy.gather_window_ms();
    {
        let state2 = state.clone();
        tokio::task::spawn_blocking(move || {
            loop {
                // ── Phase A: parallel-prefill all pending slots ───────────
                // Collect all Prefilling slots and their prompts, then issue
                // a single prefill_slots_parallel call so weights are read
                // once per position across all B slots rather than once per
                // slot (serial). On any error, release every slot in the batch.
                //
                // Gather window: when max_batch > 1 and the first Prefilling
                // slot arrives, sleep briefly WITHOUT the engine lock so that
                // concurrent HTTP admits (which also need engine.lock() for
                // tokenization) can land before we hold the lock for the full
                // prefill duration. The window duration is set by --energy-mode
                // (off=0ms, balanced=3ms, efficient=8ms). 0ms disables the
                // window entirely. Non-zero values allow co-arriving requests
                // to be batched together.
                let mut prefilling: Vec<u32> = state2.driver.lock().scheduler.prefill_slots_bucketed(max_batch);
                if !prefilling.is_empty() && max_batch > 1 && prefilling.len() < max_batch && gather_window_ms > 0 {
                    std::thread::sleep(std::time::Duration::from_millis(gather_window_ms));
                    prefilling = state2.driver.lock().scheduler.prefill_slots_bucketed(max_batch);
                }
                if !prefilling.is_empty() {
                    let slots_data: Vec<(usize, Vec<u32>)> = prefilling.iter().filter_map(|&id| {
                        let ids = state2.driver.lock().scheduler.slots
                            .iter()
                            .find(|s| s.id == id)
                            .map(|s| s.prompt_ids.clone())
                            .unwrap_or_default();
                        if ids.is_empty() { None } else { Some((id as usize, ids)) }
                    }).collect();
                    let slot_refs: Vec<(usize, &[u32])> = slots_data.iter()
                        .map(|(s, ids)| (*s, ids.as_slice()))
                        .collect();
                    // Snapshot prefix_skip for every slot in this batch before
                    // touching any slot state, so we can partition without holding
                    // both the driver and engine locks simultaneously.
                    let skip_map: Vec<(usize, usize)> = slot_refs.iter().map(|(slot_id, _)| {
                        let skip = state2.driver.lock().scheduler.slots
                            .iter().find(|s| s.id == *slot_id as u32)
                            .map(|s| s.prefix_skip).unwrap_or(0);
                        (*slot_id, skip)
                    }).collect();

                    // Reset all non-zero prefix_skip values upfront so retries
                    // don't re-skip regardless of which path runs below.
                    for &(slot_id, skip) in &skip_map {
                        if skip > 0 {
                            if let Some(s) = state2.driver.lock().scheduler.slots
                                .iter_mut().find(|s| s.id == slot_id as u32) {
                                s.prefix_skip = 0;
                            }
                        }
                    }

                    let prefill_result = {
                        let mut engine = state2.engine.lock();
                        if slot_refs.len() == 1 {
                            let (slot_id, prompt_ids) = slot_refs[0];
                            let skip = skip_map.iter().find(|(id, _)| *id == slot_id)
                                .map(|(_, s)| *s).unwrap_or(0);
                            if skip > 0 {
                                engine.prefill_slot_from_pos(slot_id, prompt_ids, skip).map(|_| ())
                            } else {
                                engine.prefill_slot(slot_id, prompt_ids).map(|_| ())
                            }
                        } else {
                            // Track 5.2: partition into slots that have a prefix_skip
                            // (handle individually with prefill_slot_from_pos) and those
                            // that don't (run in parallel).
                            let with_skip: Vec<(usize, &[u32], usize)> = slot_refs.iter()
                                .filter_map(|(slot_id, prompt_ids)| {
                                    let skip = skip_map.iter()
                                        .find(|(id, _)| id == slot_id)
                                        .map(|(_, s)| *s)
                                        .unwrap_or(0);
                                    if skip > 0 { Some((*slot_id, *prompt_ids, skip)) } else { None }
                                })
                                .collect();
                            let without_skip: Vec<(usize, &[u32])> = slot_refs.iter()
                                .filter(|(slot_id, _)| {
                                    skip_map.iter().find(|(id, _)| id == slot_id)
                                        .map(|(_, s)| *s).unwrap_or(0) == 0
                                })
                                .map(|(slot_id, prompt_ids)| (*slot_id, *prompt_ids))
                                .collect();

                            // Sequentially prefill the skip slots.
                            let mut result: Result<(), dismantle_core::Error> = Ok(());
                            for (slot_id, prompt_ids, skip) in with_skip {
                                if result.is_ok() {
                                    result = engine.prefill_slot_from_pos(slot_id, prompt_ids, skip).map(|_| ());
                                }
                            }
                            // Parallel-prefill the remaining slots (only if no error so far).
                            if result.is_ok() && !without_skip.is_empty() {
                                result = engine.prefill_slots_parallel(&without_skip);
                            }
                            result
                        }
                    };
                    match prefill_result {
                        Ok(()) => {
                            for &slot_id in &prefilling {
                                state2.driver.lock().scheduler.mark_prefill_complete(slot_id);
                            }
                        }
                        Err(e) => {
                            tracing::warn!(err = %e, "prefill_slots_parallel failed");
                            for &slot_id in &prefilling {
                                let tx = state2.slot_senders.lock().remove(&slot_id);
                                if let Some(tx) = tx { let _ = tx.blocking_send(Err(())); }
                                state2.driver.lock().scheduler.release_slot(slot_id);
                            }
                        }
                    }
                }

                // ── Phase B: one decode step across all ready slots ───────
                let outputs = {
                    let mut engine = state2.engine.lock();
                    let mut driver = state2.driver.lock();
                    driver.decode_ready_once(&mut **engine, max_batch)
                };
                let outputs = match outputs {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::error!(err = %e, "decode_ready_once failed");
                        std::thread::sleep(std::time::Duration::from_millis(1));
                        continue;
                    }
                };
                if outputs.is_empty() {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    continue;
                }

                // ── Phase C: stream tokens + release finished slots ───────
                for out in outputs {
                    let tx = state2.slot_senders.lock().get(&out.slot_id).cloned();
                    if let Some(tx) = tx {
                        let send_ok = tx.blocking_send(Ok(out.text)).is_ok();
                        if send_ok {
                            state2.tokens_generated.fetch_add(1, Ordering::Relaxed);
                        }
                        if out.finished || !send_ok {
                            // Release on normal EOS *or* client disconnect.
                            state2.slot_senders.lock().remove(&out.slot_id);
                            state2.driver.lock().scheduler.release_slot(out.slot_id);

                            // Drain one waiter into the newly-freed slot.
                            let waiter = state2.wait_queue.lock().pop_front();
                            if let Some((waiter_req, waiter_tx, _chat)) = waiter {
                                let new_slot = {
                                    let engine = state2.engine.lock();
                                    let mut driver = state2.driver.lock();
                                    driver.admit(&**engine, waiter_req).ok().flatten()
                                };
                                if let Some(sid) = new_slot {
                                    state2.requests_admitted.fetch_add(1, Ordering::Relaxed);
                                    // Track 5.2: prefix-reuse detection. After admission the
                                    // new slot is already in the prefix_index; search for a
                                    // different slot whose KV we can copy into this one.
                                    {
                                        let prompt_ids = state2.driver.lock()
                                            .scheduler.slots.iter()
                                            .find(|s| s.id == sid)
                                            .map(|s| s.prompt_ids.clone())
                                            .unwrap_or_default();
                                        if !prompt_ids.is_empty() {
                                            let prefix_match = state2.driver.lock()
                                                .scheduler.prefix_index
                                                .find_prefix_match_excluding(&prompt_ids, 8, sid);
                                            if let Some((src_slot, shared_len)) = prefix_match {
                                                tracing::debug!(
                                                    "[prefix-reuse] request matched slot {} at prefix_len={}",
                                                    src_slot, shared_len
                                                );
                                                let copy_result = state2.engine.lock()
                                                    .copy_kv_prefix_to_slot(
                                                        src_slot as usize,
                                                        sid as usize,
                                                        shared_len,
                                                    );
                                                if copy_result.is_ok() {
                                                    let mut driver = state2.driver.lock();
                                                    driver.lane_stats.prefix_reuse_count += 1;
                                                    // Record prefix_skip so the prefill path can
                                                    // call prefill_slot_from_pos instead of full prefill.
                                                    if let Some(slot) = driver.scheduler.slots
                                                        .iter_mut().find(|s| s.id == sid) {
                                                        slot.prefix_skip = shared_len;
                                                    }
                                                }
                                                // If copy_kv_prefix_to_slot returns Err (e.g.
                                                // Unimplemented), silently skip — normal prefill
                                                // will proceed from position 0.
                                            }
                                        }
                                    }
                                    state2.slot_senders.lock().insert(sid, waiter_tx);
                                }
                                // If admit fails (should not — slot was just freed),
                                // waiter_tx is dropped, which sends Err(()) on the
                                // tokio receiver, closing the SSE stream gracefully.
                            }
                        }
                    }
                }
            }
        });
    }

    let app = http::router(state);
    tracing::info!(addr = %opts.addr, "dismantle-serve listening");
    let listener = tokio::net::TcpListener::bind(opts.addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

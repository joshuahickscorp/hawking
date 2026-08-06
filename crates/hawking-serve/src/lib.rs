//! hawking-serve: OpenAI-compatible HTTP server.
//!
//! Drives a `hawking_core::Engine` through axum. Continuous
//! batching lives in [`batch`]; the HTTP surface in [`http`].

pub mod batch;
pub mod glm_chat;
pub mod http;
pub mod spec_gov;
pub mod system_kv_bank;
pub mod tool_calls;

pub use batch::scheduler::BatchPolicy;
pub use hawking_adapters::{bridge_surface_document, bridge_surface_json, EndpointStatus};
pub use system_kv_bank::{BankConfig, BankEntry, SystemPromptKvBank};

use anyhow::Result;
use serde_json::Value;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Runtime profile controlling quality/throughput trade-offs.
///
/// `Default` — conservative path; no env var changes. Source parity still
///              requires a current model-specific independent-oracle receipt.
/// `Fast`    — validated fast-path (vocab-prune + Q4K LM-head + predec + f16-scales).
/// `Race`    — same as Fast; explicitly signals "maximum throughput, quality trade-offs OK".
/// `Efficient` — same as Fast plus sets HAWKING_ENERGY_EFFICIENT=1 for energy-aware batching.
/// `Exact`   — clears profile-level quality-trade vars. Source parity still
///              requires a current model-specific independent-oracle receipt.
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
            "fast" => Some(Self::Fast),
            "race" => Some(Self::Race),
            "efficient" => Some(Self::Efficient),
            "exact" => Some(Self::Exact),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Fast => "fast",
            Self::Race => "race",
            Self::Efficient => "efficient",
            Self::Exact => "exact",
        }
    }
}

impl std::fmt::Display for RuntimeProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Data-only description of the env-var levers a [`RuntimeProfile`] activates.
///
/// Pure: building it touches no process state. Both the CLI generate path
/// (`apply_profile` in the `hawking` bin) and `serve::run` consume it, so
/// there is exactly ONE source of truth for the profile → lever mapping.
///
/// Caller contract:
///   * `set_if_unset` — set each (key,val) ONLY when the var is currently absent
///     (explicit `HAWKING_QWEN_*` env always wins → opt-out honoured).
///   * `force_off`    — set each var to "0" UNCONDITIONALLY (`Exact` uses this to
///     guarantee bit-identity even if a quality-trade var was set upstream).
///   * `f16_kv`       — profile default for the f16 KV cache (None = leave to a
///     more specific override such as `--f16-kv`).
///   * `concurrent_qkv` — whether the profile wants concurrent Q/K/V encode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeverPlan {
    pub set_if_unset: Vec<(&'static str, &'static str)>,
    pub force_off: Vec<&'static str>,
    pub f16_kv: Option<bool>,
    pub concurrent_qkv: bool,
}

impl RuntimeProfile {
    /// The validated fast-path lever bundle shared by Fast / Race / Efficient.
    /// Bit-identical EXCEPT PREDEC_F16SCALES (f16 scale rounding) and VOCAB_PRUNE
    /// (drops rare tokens) — mild quality trades; FFN_DOWN_Q4K requants Q6_K→Q4_K.
    fn fast_bundle() -> Vec<(&'static str, &'static str)> {
        vec![
            ("HAWKING_QWEN_Q4K_LMHEAD", "1"),
            ("HAWKING_QWEN_Q4K_PREDEC", "1"),
            ("HAWKING_QWEN_PREDEC_F16SCALES", "1"),
            ("HAWKING_QWEN_VOCAB_PRUNE", "32000"),
            ("HAWKING_QWEN_FFN_DOWN_Q4K", "1"),
        ]
    }

    /// Policy: which profile an UNSET `--profile` resolves to on the CLI front
    /// door. The ONE place the "fast is the default" decision lives. The library
    /// default (`RuntimeProfile::Default`) is deliberately NOT changed — embedders
    /// and serve integration tests keep the conservative default;
    /// only the CLI `generate`/`bench` front door flips.
    pub fn default_when_unset() -> Self {
        Self::Fast
    }

    /// Levers to force OFF when resolving an UNSET `--profile` (the MIDDLE
    /// variant): keep every fast lever EXCEPT `PREDEC_F16SCALES`, which failed
    /// quality_oracle at 0.792/11.46% (e613dde). Net ≈ 38–39 t/s at low quality
    /// risk. To ship FULL fast (~42) after the oracle re-passes f16-scales,
    /// return `&[]` here.
    pub fn default_unset_force_off() -> &'static [&'static str] {
        &["HAWKING_QWEN_PREDEC_F16SCALES"]
    }

    /// Pure profile → lever mapping. Touches no env state.
    pub fn lever_plan(&self) -> LeverPlan {
        match self {
            Self::Default => LeverPlan {
                set_if_unset: Vec::new(),
                force_off: Vec::new(),
                f16_kv: None,
                concurrent_qkv: false,
            },
            Self::Fast => LeverPlan {
                set_if_unset: Self::fast_bundle(),
                force_off: Vec::new(),
                f16_kv: Some(false),
                concurrent_qkv: true,
            },
            // Max t/s: fast bundle + f16 KV (frees bandwidth) + concurrent QKV.
            Self::Race => LeverPlan {
                set_if_unset: Self::fast_bundle(),
                force_off: Vec::new(),
                f16_kv: Some(true),
                concurrent_qkv: true,
            },
            // Min J/tok under a t/s floor: fast bundle + energy mode + f16 KV.
            Self::Efficient => {
                let mut s = Self::fast_bundle();
                s.push(("HAWKING_ENERGY_EFFICIENT", "1"));
                LeverPlan {
                    set_if_unset: s,
                    force_off: Vec::new(),
                    f16_kv: Some(true),
                    concurrent_qkv: true,
                }
            }
            // Bit-identical conservative path. Bit-identical default-ON levers
            // (predec/pair/gate-up-fuse) stay on; force OFF every quality-trade
            // var so output matches the golden default even if one was set upstream.
            Self::Exact => LeverPlan {
                set_if_unset: Vec::new(),
                force_off: vec![
                    "HAWKING_QWEN_PREDEC_F16SCALES", // f16 scale rounding
                    "HAWKING_QWEN_FFN_DOWN_Q4K",     // Q6_K→Q4_K requant
                    "HAWKING_QWEN_VOCAB_PRUNE",      // drops rare tokens
                ],
                f16_kv: Some(false),
                concurrent_qkv: false,
            },
        }
    }

    /// One-line human contract: lever set + quality + J/tok statement. Printed
    /// at startup so every profile "prints its active levers" (Track 2.2 gate).
    pub fn contract(&self) -> String {
        match self {
            Self::Default => "profile=default: conservative decode with no profile-level quality \
                trade. source parity: requires a current model-specific oracle receipt. J/tok: baseline."
                .to_string(),
            Self::Fast => "profile=fast: vocab-prune-32k + Q4K LM-head + Q4K FFN-down + predec \
                + f16-scales. quality: mild trade (f16 scale rounding, rare-token prune). \
                J/tok: lower than default (fewer bytes/token)."
                .to_string(),
            Self::Race => "profile=race: fast bundle + f16 KV + concurrent Q/K/V. \
                quality: same mild trade as fast. goal: MAX tokens/sec."
                .to_string(),
            Self::Efficient => "profile=efficient: fast bundle + f16 KV + energy-efficient gather \
                window. quality: same mild trade as fast. goal: MIN J/tok under a t/s floor."
                .to_string(),
            Self::Exact => "profile=exact: conservative path. Forces OFF f16-scales / Q4K-FFN-down \
                / vocab-prune. profile-level quality trades: off. source parity: requires a current \
                model-specific oracle receipt. J/tok: baseline."
                .to_string(),
        }
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
            "off" => Some(Self::Off),
            "balanced" => Some(Self::Balanced),
            "efficient" => Some(Self::Efficient),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Balanced => "balanced",
            Self::Efficient => "efficient",
        }
    }

    /// Gather window in milliseconds.
    pub fn gather_window_ms(&self) -> u64 {
        match self {
            Self::Off => 0,
            Self::Balanced => 3,
            Self::Efficient => 8,
        }
    }

    /// Pure gather/admission decision — the predicate the continuous-batch
    /// loop uses to decide whether to wait (sleep up to `gather_window_ms()`)
    /// for more requests before committing a prefill batch.
    ///
    /// Returns `true` ONLY when waiting can help AND is safe:
    ///   * `ready > 0`              — at least one slot is queued (never wait on empty),
    ///   * `max_batch > 1`          — single-slot servers can't batch → never wait
    ///     (a latency-sensitive single is NEVER delayed),
    ///   * `ready < max_batch`      — batch already full → commit now, don't wait,
    ///   * `gather_window_ms() > 0` — `Off` disables the window entirely.
    ///
    /// This is the extracted, unit-testable form of the inline predicate in
    /// `serve::run()` (the `prefilling.len() < max_batch && gather_window_ms > 0`
    /// guard). Keep the two in sync: the loop should call this helper.
    pub fn should_gather(&self, ready: usize, max_batch: usize) -> bool {
        ready > 0 && max_batch > 1 && ready < max_batch && self.gather_window_ms() > 0
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
            "default" => Some(Self::Default),
            "code-completion" => Some(Self::CodeCompletion),
            "chat-shared-prompt" => Some(Self::ChatSharedPrompt),
            "batch-summarization" => Some(Self::BatchSummarization),
            "local-agent-loop" => Some(Self::LocalAgentLoop),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::CodeCompletion => "code-completion",
            Self::ChatSharedPrompt => "chat-shared-prompt",
            Self::BatchSummarization => "batch-summarization",
            Self::LocalAgentLoop => "local-agent-loop",
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

/// Default wall-clock budget for a single completion when serving a `.gravity`
/// base runtime. Measured BASE_TRUE_TPS is ~0.4 tok/s warm; a 200-token reply
/// is ~8 minutes. Callers may override via `ServeOptions::request_timeout_secs`
/// or the env `HAWKING_GRAVITY_REQUEST_TIMEOUT_SECS`.
pub const GRAVITY_DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 3600;

/// SSE comment interval while a slow `.gravity` generation is in flight. Must
/// be short enough that proxies do not close an idle socket during multi-second
/// prefill, and explicit rather than a silent framework default.
pub const GRAVITY_DEFAULT_SSE_KEEP_ALIVE_SECS: u64 = 15;

#[derive(Debug, Clone)]
pub struct ServeOptions {
    pub weights: PathBuf,
    pub addr: SocketAddr,
    pub max_batch_size: usize,
    /// Upper bound on *new* prompt tokens scheduled together in one prefill
    /// wave. `None` retains the former unbounded-by-token behavior. A request
    /// larger than the cap advances in exact KV-preserving chunks; chunk turns
    /// rotate among eligible slots so progress does not starve shorter peers.
    pub max_prefill_tokens: Option<usize>,
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
    /// `Some(true)` — force HAWKING_QWEN_F16_KV=1 (halves KV footprint;
    ///                wins at long context, footprint-neutral for short ctx).
    /// `Some(false)` — explicitly disable (leave env var unset).
    pub f16_kv: Option<bool>,
    /// Track 5.4: batch admission policy.
    pub batch_policy: BatchPolicy,
    /// Track 9.3: workload pack (sets profile/energy/policy defaults).
    pub workload: WorkloadPack,
    /// Explicit per-request wall-clock budget in seconds. `None` leaves the
    /// non-gravity default (no server-side cutoff). Gravity serve always sets
    /// this to a large explicit value so a silent 30s cutoff cannot abort a
    /// real base-runtime completion.
    pub request_timeout_secs: Option<u64>,
    /// SSE keep-alive interval in seconds. `None` uses axum's default for
    /// non-gravity; gravity serve sets [`GRAVITY_DEFAULT_SSE_KEEP_ALIVE_SECS`].
    pub sse_keep_alive_secs: Option<u64>,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self {
            weights: PathBuf::new(),
            // Loopback by default: the local inference server must not be reachable from the LAN
            // unless the operator explicitly asks for it. The supervisor always passes an explicit
            // addr, but the binary's own default must be safe too.
            addr: "127.0.0.1:8080".parse().unwrap(),
            max_batch_size: 1,
            max_prefill_tokens: None,
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
            request_timeout_secs: None,
            sse_keep_alive_secs: None,
        }
    }
}

fn hawking_serve_system_kv_bank_default() -> system_kv_bank::SystemPromptKvBank {
    system_kv_bank::SystemPromptKvBank::new()
}

/// Receipt emitted by `tools/glm_fast_intake.py` after a fast custom-format
/// artifact has passed source-bound parity and its real decode contract.  GLM
/// is exceptionally easy to *load* in a slow host-state configuration; that
/// is not sufficient to expose it to Hide just because it has an HTTP port.
const GLM_FAST_INTAKE_RECEIPT_ENV: &str = "HAWKING_GLM_FAST_INTAKE_RECEIPT";

fn require_gate_pass(gates: &serde_json::Map<String, Value>, name: &str) -> Result<()> {
    let status = gates
        .get(name)
        .and_then(|gate| gate.get("status"))
        .and_then(Value::as_str);
    if status == Some("PASS") {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "GLM fast intake receipt has no PASS {name} gate (got {status:?})"
        ))
    }
}

fn validate_glm_fast_intake_doc(receipt: &Value, expected_index_sha256: &str) -> Result<()> {
    if receipt.get("schema").and_then(Value::as_str) != Some("hawking.glm52.fast_intake.v1") {
        return Err(anyhow::anyhow!(
            "GLM fast intake receipt has an unexpected schema"
        ));
    }
    if receipt.get("status").and_then(Value::as_str) != Some("PASS") {
        return Err(anyhow::anyhow!(
            "GLM fast intake receipt is not PASS; a slow or unbound artifact cannot be served"
        ));
    }
    let gates = receipt
        .get("gates")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("GLM fast intake receipt has no gates object"))?;
    // Require the independently meaningful leaves as well as the aggregate.
    // This makes a future bug in the aggregate calculation fail closed.
    for gate in [
        "TARGET_CONTRACT",
        "ARTIFACT_ASSEMBLY",
        "ORACLE_PARITY",
        "GPU_FAST_DECODE",
        "DECODE_PERFORMANCE",
        "HIDE_HANDOFF",
    ] {
        require_gate_pass(gates, gate)?;
    }
    let actual_index = gates
        .get("ARTIFACT_ASSEMBLY")
        .and_then(|gate| gate.get("index_sha256"))
        .and_then(Value::as_str);
    if actual_index != Some(expected_index_sha256) {
        return Err(anyhow::anyhow!(
            "GLM fast intake receipt index does not match the loaded artifact: receipt={actual_index:?} loaded={expected_index_sha256}"
        ));
    }
    Ok(())
}

fn require_glm_fast_intake(expected_index_sha256: Option<&str>) -> Result<()> {
    let expected_index_sha256 = expected_index_sha256.ok_or_else(|| {
        anyhow::anyhow!(
            "glm_moe_dsa fast serve requires an indexed custom artifact; refusing an unbound single-shard path"
        )
    })?;
    let path = std::env::var_os(GLM_FAST_INTAKE_RECEIPT_ENV).ok_or_else(|| {
        anyhow::anyhow!(
            "glm_moe_dsa serve is speed-gated; set {GLM_FAST_INTAKE_RECEIPT_ENV} to a PASS receipt from tools/glm_fast_intake.py"
        )
    })?;
    let path = PathBuf::from(path);
    let bytes = std::fs::read(&path).map_err(|error| {
        anyhow::anyhow!(
            "cannot read GLM fast intake receipt {}: {error}",
            path.display()
        )
    })?;
    let receipt: Value = serde_json::from_slice(&bytes).map_err(|error| {
        anyhow::anyhow!(
            "invalid GLM fast intake receipt {}: {error}",
            path.display()
        )
    })?;
    validate_glm_fast_intake_doc(&receipt, expected_index_sha256)
}

pub async fn run(opts: ServeOptions) -> Result<()> {
    use hawking_core::{profile::KernelProfile, EngineConfig, SpeculateMode};

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
    let max_prefill_tokens = opts.max_prefill_tokens.unwrap_or(usize::MAX);

    // ── Serve-mode optimisation defaults ─────────────────────────────────────
    // These are the same knobs that `hawking generate --kernel-profile` uses.
    // Each can be overridden by the caller's environment (set var before invoking
    // the server). We only set them when the variable is absent so that explicit
    // HAWKING_QWEN_*=0 opt-outs are honoured.
    for (var, val) in [
        ("HAWKING_QWEN_Q4K_PREDEC", "1"), // pre-decoded scales → fast GEMV
        ("HAWKING_QWEN_Q4K_LMHEAD", "1"), // GPU Q4K LM-head (vs CPU f16)
        ("HAWKING_QWEN_VOCAB_PRUNE", "32000"), // prune to 32K most-frequent tokens
        ("HAWKING_QWEN_TCB", "1"),        // token command buffers
        ("HAWKING_QWEN_FFN_DOWN_Q4K", "1"), // FFN down Q4K path
    ] {
        if std::env::var_os(var).is_none() {
            std::env::set_var(var, val);
        }
    }

    // ── Apply runtime profile env overrides ──────────────────────────────────
    // Fast / Race / Efficient: opt into the both-metrics-optimal fast-path.
    // Exact: clear quality-trade vars so the path is bit-identical.
    // All of these respect explicit HAWKING_QWEN_*=0 opt-outs set before launch.
    // Single source of truth = RuntimeProfile::lever_plan() (shared with the CLI
    // generate path). set_if_unset respects explicit HAWKING_QWEN_*=0 opt-outs;
    // force_off enforces Exact's bit-identity even if a quality-trade var was set.
    let plan = effective_profile.lever_plan();
    for (k, v) in &plan.set_if_unset {
        if std::env::var_os(k).is_none() {
            std::env::set_var(k, v);
        }
    }
    for k in &plan.force_off {
        std::env::set_var(k, "0");
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
        let profile_wants_f16_kv = plan.f16_kv.unwrap_or(false);
        let enable = match opts.f16_kv {
            Some(v) => v,
            None => profile_wants_f16_kv,
        };
        if enable && std::env::var_os("HAWKING_QWEN_F16_KV").is_none() {
            std::env::set_var("HAWKING_QWEN_F16_KV", "1");
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
    let concurrent_qkv = plan.concurrent_qkv
        || std::env::var_os("HAWKING_QWEN_CONCURRENT_QKV")
            .map(|v| v == "1")
            .unwrap_or(false);

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

    let engine = hawking_core::model::load_engine(&opts.weights, cfg)
        .map_err(|e| anyhow::anyhow!("load engine: {e}"))?;
    let model_id = engine.model_id().to_string();
    let model_arch = engine.model_arch().to_string();
    let max_batch = opts.max_batch_size;

    // Hide reaches GLM through this server.  Refuse the historic host-state
    // lane here, at the process boundary, before an HTTP client can mistake a
    // 0.x TPS reply for a runnable Ramanujan model.
    if model_arch == "glm_moe_dsa" {
        require_glm_fast_intake(engine.artifact_index_sha256())?;
    }

    // Gravity base-runtime capability surface. Detected from the loaded engine
    // (index sha256 + chat template), never guessed. When present we raise the
    // explicit request timeout / SSE keep-alive so a ~0.4 tok/s base path is
    // not cut off by a silent 30s default.
    let gravity_meta = if engine.is_base_runtime()
        || engine.artifact_index_sha256().is_some()
        || model_arch == "glm_moe_dsa"
    {
        // GLM without its real template is a hard fail (load already enforces
        // this for glm_moe_dsa; re-check so a future loader regression surfaces
        // at serve start rather than as fluent garbage).
        if model_arch == "glm_moe_dsa" && engine.chat_template().is_none() {
            return Err(anyhow::anyhow!(
                "glm_moe_dsa gravity serve: artifact chat template is missing; \
                 refusing to serve with a guessed template"
            ));
        }
        let timeout = opts
            .request_timeout_secs
            .or_else(|| {
                std::env::var("HAWKING_GRAVITY_REQUEST_TIMEOUT_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
            })
            .unwrap_or(GRAVITY_DEFAULT_REQUEST_TIMEOUT_SECS);
        let keep_alive = opts
            .sse_keep_alive_secs
            .unwrap_or(GRAVITY_DEFAULT_SSE_KEEP_ALIVE_SECS);
        eprintln!(
            "[gravity serve] model_id={model_id} arch={model_arch} base_runtime=true \
             fallback_present=false"
        );
        if let Some(sha) = engine.artifact_index_sha256() {
            eprintln!("[gravity serve] artifact_index_sha256={sha}");
        } else {
            eprintln!("[gravity serve] artifact_index_sha256=<none — single-shard or no index>");
        }
        if let Some(p) = engine.chat_template_path() {
            eprintln!("[gravity serve] chat_template={p}");
        }
        eprintln!(
            "[gravity serve] request_timeout_secs={timeout} (explicit; measured base decode is \
             ~0.4 tok/s warm — do not treat a slow reply as a hang)"
        );
        eprintln!("[gravity serve] sse_keep_alive_secs={keep_alive}");
        Some(http::GravityServeMeta {
            index_sha256: engine.artifact_index_sha256().map(str::to_string),
            architecture: model_arch.clone(),
            model_id: model_id.clone(),
            chat_template: engine.chat_template().map(str::to_string),
            chat_template_path: engine.chat_template_path().map(str::to_string),
            base_runtime: true,
            request_timeout_secs: timeout,
            sse_keep_alive_secs: keep_alive,
        })
    } else {
        None
    };

    // ── --explain-performance startup summary ─────────────────────────────
    if opts.explain_performance {
        let token_only_active = effective_profile == RuntimeProfile::Fast
            || effective_profile == RuntimeProfile::Race
            || effective_profile == RuntimeProfile::Efficient
            || std::env::var_os("HAWKING_QWEN_Q4K_LMHEAD")
                .map(|v| v == "1")
                .unwrap_or(false);
        let token_only_str = if token_only_active {
            "active (Q4K LM head loaded)"
        } else {
            "inactive (fallback to full logits)"
        };
        let hw_profile_str = opts
            .kernel_profile
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "none".to_string());
        let gather_ms = effective_energy.gather_window_ms();
        let f16_kv_active = std::env::var_os("HAWKING_QWEN_F16_KV")
            .map(|v| v == "1")
            .unwrap_or(false);
        let full_logits_mb = max_batch as f64 * 151936.0 * 4.0 / 1_048_576.0;
        let greedy_bytes = max_batch * 4;
        eprintln!(
            "hawking serve — performance summary\n\
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
        system_kv_bank: Arc::new(parking_lot::Mutex::new(
            hawking_serve_system_kv_bank_default(),
        )),
        gravity: gravity_meta,
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
                // Track 5: dispatch on scheduler.policy. prefill_slots_prefix_grouped
                // returns the same-prefix cohort (group_by_prefix, min_shared=8) when
                // policy == PrefixGrouped, else delegates to prefill_slots_bucketed —
                // byte-for-byte identical for Default/GreedyFirst. The policy was
                // installed at startup (`d.scheduler.policy = effective_batch_policy`),
                // so no extra binding is captured here.
                let mut prefill_chunks = state2
                    .driver
                    .lock()
                    .scheduler
                    .prefill_chunks_token_budgeted(max_batch, max_prefill_tokens);
                if effective_energy.should_gather(prefill_chunks.len(), max_batch) {
                    std::thread::sleep(std::time::Duration::from_millis(gather_window_ms));
                    prefill_chunks = state2
                        .driver
                        .lock()
                        .scheduler
                        .prefill_chunks_token_budgeted(max_batch, max_prefill_tokens);
                }
                if !prefill_chunks.is_empty() {
                    // Chunking is exact prompt continuation, not speculative
                    // generation: each engine call receives the original token
                    // prefix through `end`, starts at the durable slot cursor,
                    // and produces no externally-visible token until `complete`.
                    let slots_data: Vec<(usize, Vec<u32>, usize, usize, bool)> = prefill_chunks
                        .iter()
                        .filter_map(|chunk| {
                            let ids = state2
                                .driver
                                .lock()
                                .scheduler
                                .slots
                                .iter()
                                .find(|s| s.id == chunk.slot_id)
                                .map(|s| s.prompt_ids.clone())
                                .unwrap_or_default();
                            if ids.is_empty() || chunk.start >= chunk.end || chunk.end > ids.len() {
                                None
                            } else {
                                Some((
                                    chunk.slot_id as usize,
                                    ids,
                                    chunk.start,
                                    chunk.end,
                                    chunk.complete,
                                ))
                            }
                        })
                        .collect();
                    let slot_refs: Vec<(usize, &[u32], usize)> = slots_data
                        .iter()
                        .map(|(s, ids, start, end, _)| (*s, &ids[..*end], *start))
                        .collect();

                    // A bank entry is valid only while its source slot has
                    // not begun a cold overwrite. Device-buffer copies do
                    // not carry provenance, so invalidate before writing KV,
                    // then re-record only after a successful prefill.
                    for &(slot_id, _, start) in &slot_refs {
                        if start == 0 {
                            state2.system_kv_bank.lock().forget_slot(slot_id as u32);
                        }
                    }

                    let prefill_result = {
                        let mut engine = state2.engine.lock();
                        if slot_refs.len() == 1 {
                            let (slot_id, prompt_ids, start) = slot_refs[0];
                            if start > 0 {
                                engine
                                    .prefill_slot_from_pos(slot_id, prompt_ids, start)
                                    .map(|ft| vec![(slot_id, ft)])
                            } else {
                                engine
                                    .prefill_slot(slot_id, prompt_ids)
                                    .map(|ft| vec![(slot_id, ft)])
                            }
                        } else {
                            // Resumed chunks must preserve their absolute positions
                            // in their slot KV region, so run those individually.
                            // Fresh chunks can retain the parallel prefill path.
                            let with_skip: Vec<(usize, &[u32], usize)> = slot_refs
                                .iter()
                                .filter_map(|(slot_id, prompt_ids, start)| {
                                    if *start > 0 {
                                        Some((*slot_id, *prompt_ids, *start))
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            let without_skip: Vec<(usize, &[u32])> = slot_refs
                                .iter()
                                .filter(|(_, _, start)| *start == 0)
                                .map(|(slot_id, prompt_ids, _)| (*slot_id, *prompt_ids))
                                .collect();

                            // Sequentially prefill the skip slots, collecting each
                            // slot's first generated token to seed decode with.
                            let mut firsts: Vec<(usize, u32)> = Vec::new();
                            let mut result: Result<(), hawking_core::Error> = Ok(());
                            for (slot_id, prompt_ids, skip) in with_skip {
                                if result.is_ok() {
                                    match engine.prefill_slot_from_pos(slot_id, prompt_ids, skip) {
                                        Ok(ft) => firsts.push((slot_id, ft)),
                                        Err(e) => result = Err(e),
                                    }
                                }
                            }
                            // Parallel-prefill the remaining slots (only if no error so far).
                            if result.is_ok() && !without_skip.is_empty() {
                                match engine.prefill_slots_parallel(&without_skip) {
                                    Ok(fts) => {
                                        for ((sid, _), ft) in without_skip.iter().zip(fts) {
                                            firsts.push((*sid, ft));
                                        }
                                    }
                                    Err(e) => result = Err(e),
                                }
                            }
                            result.map(|()| firsts)
                        }
                    };
                    match prefill_result {
                        Ok(firsts) => {
                            // A cursor is durable only after the engine call
                            // succeeds. Incomplete chunks stay Prefilling and
                            // cannot seed decode, enter the prefix bank, or
                            // emit an unverified token.
                            let mut committed_final_slots = Vec::new();
                            for &(slot_id, _, _, end, complete) in &slots_data {
                                let committed = state2
                                    .driver
                                    .lock()
                                    .scheduler
                                    .commit_prefill_chunk(slot_id as u32, end);
                                if !committed {
                                    tracing::error!(slot_id, end, "prefill chunk commit rejected");
                                } else if complete {
                                    committed_final_slots.push(slot_id);
                                }
                            }

                            // Only the final prompt chunk becomes a reusable
                            // prefix source. Keeping partial KV out of the bank
                            // prevents a later request from treating a prefix
                            // in flight as complete state.
                            for &(slot_id, ref prompt_ids, _, _, _) in &slots_data {
                                if !committed_final_slots.contains(&slot_id) {
                                    continue;
                                }
                                let mut bank = state2.system_kv_bank.lock();
                                for prefix_len in http::bank_prefix_anchors(prompt_ids) {
                                    bank.record(prompt_ids, prefix_len, slot_id as u32);
                                }
                            }
                            // Mark each prefilled slot ready, then SEED it with the
                            // first generated token (from the prefill's last-position
                            // logits) and stream that token immediately. The decode
                            // loop then continues from the SECOND token. This avoids
                            // re-feeding the last prompt token through the decode
                            // path, which produced a spurious leading word.
                            let eos = { state2.engine.lock().eos_id_for_batch() };
                            for (slot_id, first_token) in firsts {
                                let complete = slots_data.iter().any(|(id, _, _, _, _)| {
                                    *id == slot_id && committed_final_slots.contains(id)
                                });
                                if !complete {
                                    continue;
                                }
                                let sid = slot_id as u32;
                                let decoded = {
                                    let mut driver = state2.driver.lock();
                                    driver.scheduler.mark_prefill_complete(sid);
                                    driver.scheduler.seed_first_token(sid, first_token, eos)
                                };
                                let Some(decoded) = decoded else { continue };
                                let text = {
                                    state2
                                        .engine
                                        .lock()
                                        .decode_token_for_batch(first_token)
                                        .unwrap_or_default()
                                };
                                let tx = state2.slot_senders.lock().get(&sid).cloned();
                                if let Some(tx) = tx {
                                    let _ = tx.blocking_send(Ok(text));
                                    state2.tokens_generated.fetch_add(1, Ordering::Relaxed);
                                    if decoded.finished {
                                        state2.slot_senders.lock().remove(&sid);
                                        state2.driver.lock().scheduler.release_slot(sid);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            // Batch prefill unavailable for this engine/device
                            // (e.g. Llama still lacks `prefill_slot`, or Metal is
                            // absent so Qwen's GPU prefill cannot run). Fall back
                            // to single-stream `Engine::generate` — the same path
                            // `hawking generate` uses — so real tokens still flow
                            // over `/v1/hawking/generate` instead of empty
                            // completions. Serial under the engine lock; correct
                            // for the HIDE single-turn path.
                            tracing::warn!(
                                err = %e,
                                "prefill_slots_parallel failed; falling back to single-stream generate"
                            );
                            for chunk in &prefill_chunks {
                                let slot_id = chunk.slot_id;
                                let req = state2
                                    .driver
                                    .lock()
                                    .scheduler
                                    .slots
                                    .iter()
                                    .find(|s| s.id == slot_id)
                                    .and_then(|s| s.req.clone());
                                let tx = state2.slot_senders.lock().remove(&slot_id);
                                match (req, tx) {
                                    (Some(req), Some(tx)) => {
                                        let gen_result = {
                                            let mut engine = state2.engine.lock();
                                            let mut send_err = false;
                                            let r = engine.generate(req, &mut |ev| match ev {
                                                hawking_core::StreamEvent::Token {
                                                    text, ..
                                                } => {
                                                    if tx.blocking_send(Ok(text)).is_err() {
                                                        send_err = true;
                                                    } else {
                                                        state2
                                                            .tokens_generated
                                                            .fetch_add(1, Ordering::Relaxed);
                                                    }
                                                }
                                                hawking_core::StreamEvent::Done { .. } => {}
                                            });
                                            (r, send_err)
                                        };
                                        if let Err(gen_err) = gen_result.0 {
                                            tracing::warn!(
                                                err = %gen_err,
                                                slot_id,
                                                "single-stream generate fallback failed"
                                            );
                                            let _ = tx.blocking_send(Err(()));
                                        } else if gen_result.1 {
                                            // Client disconnected mid-stream.
                                        }
                                    }
                                    (_, Some(tx)) => {
                                        let _ = tx.blocking_send(Err(()));
                                    }
                                    _ => {}
                                }
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
                                let new_slot = http::admit_with_prefix_reuse(&state2, waiter_req)
                                    .ok()
                                    .flatten();
                                if let Some(sid) = new_slot {
                                    state2.requests_admitted.fetch_add(1, Ordering::Relaxed);
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
    tracing::info!(addr = %opts.addr, "hawking-serve listening");
    let listener = tokio::net::TcpListener::bind(opts.addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod profile_lever_tests {
    use super::RuntimeProfile as RP;
    fn has(plan_keys: &[(&'static str, &'static str)], k: &str) -> bool {
        plan_keys.iter().any(|(kk, _)| *kk == k)
    }
    #[test]
    fn default_touches_nothing() {
        let p = RP::Default.lever_plan();
        assert!(p.set_if_unset.is_empty());
        assert!(p.force_off.is_empty());
        assert_eq!(p.f16_kv, None);
        assert!(!p.concurrent_qkv);
    }
    #[test]
    fn fast_sets_full_bundle_no_f16kv() {
        let p = RP::Fast.lever_plan();
        for k in [
            "HAWKING_QWEN_Q4K_LMHEAD",
            "HAWKING_QWEN_Q4K_PREDEC",
            "HAWKING_QWEN_PREDEC_F16SCALES",
            "HAWKING_QWEN_VOCAB_PRUNE",
            "HAWKING_QWEN_FFN_DOWN_Q4K",
        ] {
            assert!(has(&p.set_if_unset, k), "fast must set {k}");
        }
        assert_eq!(p.f16_kv, Some(false), "fast leaves f16-KV off");
        assert!(p.concurrent_qkv);
        assert!(p.force_off.is_empty());
    }
    #[test]
    fn race_is_fast_plus_f16kv() {
        let p = RP::Race.lever_plan();
        assert!(has(&p.set_if_unset, "HAWKING_QWEN_VOCAB_PRUNE"));
        assert_eq!(p.f16_kv, Some(true), "race enables f16-KV");
        assert!(p.concurrent_qkv);
        assert!(!has(&p.set_if_unset, "HAWKING_ENERGY_EFFICIENT"));
    }
    #[test]
    fn efficient_adds_energy_and_f16kv() {
        let p = RP::Efficient.lever_plan();
        assert!(
            has(&p.set_if_unset, "HAWKING_ENERGY_EFFICIENT"),
            "efficient sets energy mode"
        );
        assert_eq!(p.f16_kv, Some(true), "efficient enables f16-KV");
        assert!(has(&p.set_if_unset, "HAWKING_QWEN_Q4K_PREDEC"));
    }
    #[test]
    fn exact_force_offs_every_quality_trade() {
        let p = RP::Exact.lever_plan();
        for k in [
            "HAWKING_QWEN_PREDEC_F16SCALES",
            "HAWKING_QWEN_FFN_DOWN_Q4K",
            "HAWKING_QWEN_VOCAB_PRUNE",
        ] {
            assert!(p.force_off.contains(&k), "exact must force-off {k}");
        }
        assert!(p.set_if_unset.is_empty(), "exact sets no quality-trade var");
        assert_eq!(
            p.f16_kv,
            Some(false),
            "exact leaves f16-KV off (bit-identity)"
        );
        assert!(!p.concurrent_qkv);
    }
    #[test]
    fn contracts_are_nonempty_and_self_label() {
        for rp in [RP::Default, RP::Fast, RP::Race, RP::Efficient, RP::Exact] {
            let c = rp.contract();
            assert!(
                c.contains(rp.as_str()),
                "contract for {rp} must name itself"
            );
            assert!(c.len() > 20);
        }
    }
    #[test]
    fn from_str_roundtrips_all_known() {
        for s in ["default", "fast", "race", "efficient", "exact"] {
            assert_eq!(RP::from_str(s).unwrap().as_str(), s);
        }
        assert!(
            RP::from_str("m3-pro-18gb").is_none(),
            "hardware string is not a runtime profile"
        );
    }
    #[test]
    fn default_when_unset_is_fast() {
        assert_eq!(RP::default_when_unset(), RP::Fast);
    }
    #[test]
    fn unset_default_is_fast_minus_f16scales() {
        let bundle = RP::Fast.lever_plan().set_if_unset; // == fast_bundle()
        let force_off = RP::default_unset_force_off();
        assert_eq!(force_off, &["HAWKING_QWEN_PREDEC_F16SCALES"]);
        for k in [
            "HAWKING_QWEN_Q4K_LMHEAD",
            "HAWKING_QWEN_Q4K_PREDEC",
            "HAWKING_QWEN_VOCAB_PRUNE",
            "HAWKING_QWEN_FFN_DOWN_Q4K",
        ] {
            assert!(
                has(&bundle, k),
                "fast bundle must keep {k} (a kept lever under the unset default)"
            );
        }
        assert!(has(&bundle, force_off[0]));
    }
}

#[cfg(test)]
mod glm_fast_intake_tests {
    use super::validate_glm_fast_intake_doc;
    use serde_json::json;

    fn receipt(index: &str) -> serde_json::Value {
        let mut gates = serde_json::Map::new();
        for name in [
            "TARGET_CONTRACT",
            "ORACLE_PARITY",
            "GPU_FAST_DECODE",
            "DECODE_PERFORMANCE",
            "HIDE_HANDOFF",
        ] {
            gates.insert(name.into(), json!({"status": "PASS"}));
        }
        gates.insert(
            "ARTIFACT_ASSEMBLY".into(),
            json!({"status": "PASS", "index_sha256": index}),
        );
        json!({
            "schema": "hawking.glm52.fast_intake.v1",
            "status": "PASS",
            "gates": gates,
        })
    }

    #[test]
    fn exact_pass_receipt_is_bound_to_the_loaded_index() {
        assert!(validate_glm_fast_intake_doc(&receipt("abc"), "abc").is_ok());
    }

    #[test]
    fn index_mismatch_cannot_serve_a_repacked_glm() {
        let error = validate_glm_fast_intake_doc(&receipt("old"), "new")
            .expect_err("a receipt for another artifact must be refused");
        assert!(error.to_string().contains("does not match"));
    }

    #[test]
    fn aggregate_pass_cannot_hide_a_missing_speed_leaf() {
        let mut value = receipt("abc");
        value["gates"]["GPU_FAST_DECODE"]["status"] = json!("BLOCKED");
        let error = validate_glm_fast_intake_doc(&value, "abc")
            .expect_err("missing fast proof must be refused");
        assert!(error.to_string().contains("GPU_FAST_DECODE"));
    }
}

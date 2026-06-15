mod bench_kernel;
mod bench_server;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "dismantle", about = "Apple Silicon MoE inference", version)]
struct Cli {
    /// Apply a named lever bundle (global; works after the subcommand). Currently
    /// only "fast" = the validated both-metrics fast-path: vocab-prune-32k + Q4K
    /// LM-head + Q4K FFN-down + predec + f16-scales. Opt-in; the default decode
    /// stays bit-identical. f16-scales / vocab-prune are mild quality trades.
    /// Explicitly-set DISMANTLE_QWEN_* env vars always take precedence.
    #[arg(long, global = true)]
    profile: Option<String>,
    #[command(subcommand)]
    cmd: Cmd,
}

/// Apply a named lever bundle by setting the corresponding DISMANTLE_QWEN_* env
/// vars *only if the user has not already set them* (explicit env always wins).
/// No `--profile` ⇒ no change ⇒ the default decode stays bit-identical.
///
/// Known profiles:
///   `fast`      — validated fast-path: vocab-prune-32k + Q4K LM-head + Q4K
///                 FFN-down + predec + f16-scales (mild quality trade).
///   `race`      — same as fast; explicitly signals max throughput, quality
///                 trade-offs OK.
///   `efficient` — same as fast plus DISMANTLE_ENERGY_EFFICIENT=1.
///   `exact`     — bit-identical conservative path (no quality trade-offs).
///   `default`   — no change from the locked bit-identical default.
///
/// Explicitly-set DISMANTLE_QWEN_* env vars always take precedence.
fn apply_profile(profile: &Option<String>, announce: bool) {
    let Some(name) = profile.as_deref() else {
        // Unset --profile → policy default = `fast` MINUS the f16-scales lever
        // that failed quality_oracle (e613dde): vocab-prune + Q4K LM-head +
        // Q4K FFN-down + predec, f16-scales OFF (~38–39 t/s, low quality risk).
        // Explicit DISMANTLE_QWEN_*=0 still wins (set_if_unset); the force-off
        // is unconditional. Pass --profile exact for the bit-identical path.
        let rp = dismantle_serve::RuntimeProfile::default_when_unset();
        let plan = rp.lever_plan();
        for (k, v) in &plan.set_if_unset {
            if std::env::var_os(k).is_none() {
                std::env::set_var(k, v);
            }
        }
        for k in &plan.force_off {
            std::env::set_var(k, "0");
        }
        for k in dismantle_serve::RuntimeProfile::default_unset_force_off() {
            std::env::set_var(k, "0");
        }
        if let Some(true) = plan.f16_kv {
            if std::env::var_os("DISMANTLE_QWEN_F16_KV").is_none() {
                std::env::set_var("DISMANTLE_QWEN_F16_KV", "1");
            }
        }
        if plan.concurrent_qkv && std::env::var_os("DISMANTLE_QWEN_CONCURRENT_QKV").is_none() {
            std::env::set_var("DISMANTLE_QWEN_CONCURRENT_QKV", "1");
        }
        if announce {
            eprintln!(
                "[dismantle] no --profile → default=fast (minus f16-scales, ~38-39 t/s); \
                 pass --profile exact for bit-identical, --profile fast for full ~42 t/s"
            );
        }
        return;
    };
    // autotune's hardware string ("m3-pro-18gb") is a different concept (a
    // subcommand arg), not this global runtime lever — only known runtime
    // profiles apply. The mapping is the SAME LeverPlan that serve::run uses,
    // so generate/bench and serve never drift (fixes: race/efficient were silent
    // aliases of fast here, and `exact` did not actually force-off the f16-scales
    // quality lever → non-bit-identical despite its contract).
    let Some(rp) = dismantle_serve::RuntimeProfile::from_str(name) else {
        eprintln!(
            "[dismantle] warning: unknown --profile '{name}' \
             (known: default, fast, race, efficient, exact); ignoring"
        );
        return;
    };
    let plan = rp.lever_plan();
    for (k, v) in &plan.set_if_unset {
        if std::env::var_os(k).is_none() {
            std::env::set_var(k, v);
        }
    }
    for k in &plan.force_off {
        // Unconditional: exact opts out of quality trades even if set upstream.
        std::env::set_var(k, "0");
    }
    if let Some(true) = plan.f16_kv {
        if std::env::var_os("DISMANTLE_QWEN_F16_KV").is_none() {
            std::env::set_var("DISMANTLE_QWEN_F16_KV", "1");
        }
    }
    if plan.concurrent_qkv && std::env::var_os("DISMANTLE_QWEN_CONCURRENT_QKV").is_none() {
        std::env::set_var("DISMANTLE_QWEN_CONCURRENT_QKV", "1");
    }
    if announce {
        eprintln!("[dismantle] {}", rp.contract());
    }
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Start the OpenAI-compatible HTTP server.
    Serve {
        #[arg(long)]
        weights: PathBuf,
        #[arg(long, default_value = "0.0.0.0:8080")]
        addr: std::net::SocketAddr,
        #[arg(long, default_value_t = 1)]
        max_batch_size: usize,
        #[arg(long, num_args = 0..=1, default_missing_value = "exact-shared", value_name = "MODE")]
        speculate: Option<String>,
        #[arg(long, default_value_t = 4)]
        verify_window: usize,
        /// Load a hardware kernel-profile JSON produced by `dismantle autotune`.
        /// Controls which Metal kernel variant is selected per tensor shape.
        /// This flag is about hardware tuning, not runtime behavior — see
        /// --profile for the runtime quality/throughput lever.
        #[arg(long)]
        kernel_profile: Option<PathBuf>,
        /// Alias for --kernel-profile. Both names are accepted; --hardware-profile
        /// is the preferred spelling because it makes clear this is a JSON path
        /// from `dismantle autotune`, not a runtime mode selector.
        #[arg(long, conflicts_with = "kernel_profile")]
        hardware_profile: Option<PathBuf>,
        #[arg(long)]
        prefill_cache_dir: Option<PathBuf>,
        #[arg(long)]
        max_routed_expert_ram_mb: Option<usize>,
        /// Total memory budget for weights + KV cache in MiB. Engine errors at
        /// load time if the model file exceeds this limit. Pass 0 for auto-
        /// detection (80% of system RAM). Default: unlimited.
        #[arg(long)]
        memory_limit_mb: Option<usize>,
        /// Energy mode for gather-window sizing.
        ///   off       — no gather window (lowest latency, default)
        ///   balanced  — 3 ms gather window (good batch-fill vs latency tradeoff)
        ///   efficient — 8 ms gather window (maximise batch fill for lower J/tok)
        #[arg(long, default_value = "off", value_name = "MODE")]
        energy_mode: Option<String>,
        /// Print a human-readable performance summary at startup before
        /// accepting connections, then continue serving normally.
        #[arg(long, default_value_t = false)]
        explain_performance: bool,
        /// Track 5.3: force f16 KV cache on (overrides profile default).
        /// Halves KV memory; wins at long context, neutral for short ctx.
        /// Mutually exclusive with --no-f16-kv.
        #[arg(long, conflicts_with = "no_f16_kv")]
        f16_kv: bool,
        /// Track 5.3: force f16 KV cache off (overrides profile default).
        /// Mutually exclusive with --f16-kv.
        #[arg(long, conflicts_with = "f16_kv")]
        no_f16_kv: bool,
        /// Track 5.4: batch admission policy.
        ///   default         — FIFO (current behavior)
        ///   greedy-first    — greedy (temp=0) slots first; maximises token-only lane hits
        ///   prefix-grouped  — prefer slots sharing a common prefix
        #[arg(long, default_value = "default", value_name = "POLICY")]
        batch_policy: Option<String>,
        /// Track 9.3: workload pack — sets profile/energy/batch-policy defaults.
        ///   default             — no change (individual flags apply as-is)
        ///   code-completion     — race profile + energy off + greedy-first batching
        ///   chat-shared-prompt  — fast profile + balanced energy + prefix-grouped batching
        ///   batch-summarization — efficient profile + efficient energy + greedy-first batching
        ///   local-agent-loop    — fast profile + energy off + greedy-first batching
        /// Individual flags always override the workload pack's defaults.
        #[arg(long, default_value = "default", value_name = "PACK")]
        workload: Option<String>,
    },
    /// One-shot generation to stdout.
    Generate {
        #[arg(long)]
        weights: PathBuf,
        /// Single prompt. Optional when --prompts-file is given (the file
        /// then supplies every prompt).
        #[arg(long, default_value = "")]
        prompt: String,
        #[arg(long, default_value_t = 256)]
        max_new_tokens: usize,
        /// KV-cache capacity in tokens. The cache is ALLOCATED at this size; a
        /// prompt+generation exceeding it errors "kv cache full at N". Default
        /// 4096. Raise for long context — pair with --profile race / f16-KV /
        /// int4-KV so the larger cache still fits in memory.
        #[arg(long, default_value_t = 4096)]
        max_seq_len: usize,
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        #[arg(long, default_value_t = 40)]
        top_k: u32,
        #[arg(long, default_value_t = 0.95)]
        top_p: f32,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        kernel_profile: Option<PathBuf>,
        #[arg(long, num_args = 0..=1, default_missing_value = "exact-shared", value_name = "MODE")]
        speculate: Option<String>,
        #[arg(long, default_value_t = 4)]
        verify_window: usize,
        /// Abort if any single forward step takes longer than this many
        /// milliseconds. `0` disables the watchdog. Default 0 (off);
        /// set to e.g. 30000 to bail on a stuck CPU step.
        #[arg(long, default_value_t = 0)]
        max_stall_ms: u64,
        /// Enable Metal dispatch tracing and structural allocation/commit
        /// counters. Equivalent to setting DISMANTLE_TRACE_DISPATCH=1.
        #[arg(long, default_value_t = false)]
        trace_dispatch: bool,
        #[arg(long)]
        max_routed_expert_ram_mb: Option<usize>,
        /// Total memory budget for weights + KV cache in MiB. Engine errors at
        /// load time if the model file exceeds this limit. Pass 0 for auto-
        /// detection (80% of system RAM). Default: unlimited.
        #[arg(long)]
        memory_limit_mb: Option<usize>,
        /// path-to-50 lever 1: path to a vocab whitelist JSON (built by
        /// `tools/training/analyze_corpus.py`). When set, the LM head is
        /// sliced to the pruned vocab at load time. DeepSeek-V2-Lite only.
        #[arg(long)]
        vocab_prune_path: Option<PathBuf>,
        /// path-to-50 lever 2: path to a per-layer quant tier-map JSON
        /// (see `crates/dismantle-core/src/quant_tier_map.rs`). When set,
        /// MoE expert weights are re-quantized per-layer at load time.
        #[arg(long)]
        quant_tier_map_path: Option<PathBuf>,
        /// Path to a trained Eagle5 v2 head checkpoint (safetensors).
        /// Only meaningful when `--speculate eagle5` (or
        /// `DISMANTLE_SPEC_DECODE=eagle5`) is set. When omitted, a
        /// deterministic mock head is constructed — useful for
        /// validating the spec-decode runtime path while the trained
        /// checkpoint is being produced.
        #[arg(long)]
        eagle5_head: Option<PathBuf>,
        /// Write per-cycle Eagle5 accept/reject records as JSONL. Also
        /// available through DISMANTLE_QWEN_EAGLE5_ACCEPT_TRACE.
        #[arg(long)]
        eagle5_accept_trace: Option<PathBuf>,
        /// Capture corpus mode: path to a newline-delimited prompts file.
        /// When set, the model is loaded ONCE and every prompt is decoded
        /// in sequence into the same process — the efficient path for
        /// building a quantized-residual capture corpus (set
        /// DISMANTLE_QWEN_CAPTURE_CORPUS_PATH + DISMANTLE_QWEN_EAGLE5_CAPTURE=1).
        /// Overrides --prompt when present.
        #[arg(long)]
        prompts_file: Option<PathBuf>,
        /// L3.1 §2.1b — enable the per-user n-gram speculative draft (the
        /// propose→batched-verify→accept loop). Equivalent to setting
        /// DISMANTLE_QWEN_USER_DRAFT=1. Greedy (temp=0) + TCB only; lossless
        /// by construction (output is bit-identical to plain greedy). Without
        /// this flag the draft path is never entered (the failing CLI run in
        /// reports/move2_user_draft_diagnosis.md decoded with draft_accepted=0
        /// because the flag was unset).
        #[arg(long, default_value_t = false)]
        user_draft: bool,
        /// Select the PROPOSE-FIRST user-draft loop (1 verify forward/cycle)
        /// instead of the default bonus-first loop (2 forwards/cycle).
        /// Equivalent to also setting DISMANTLE_QWEN_USER_DRAFT_PROPOSE_FIRST=1.
        /// Only meaningful together with --user-draft; bit-identical to the
        /// bonus-first loop (changes only forward-count scheduling).
        #[arg(long, default_value_t = false)]
        user_draft_propose_first: bool,
        /// Print a one-shot performance/configuration banner to stderr at
        /// startup (model, active profile incl. the unset→fast-minus-f16scales
        /// default, fast levers active, lm_head path, sidecar status, token-only
        /// availability, full-logits cost), then generate normally. Mirrors
        /// `serve --explain-performance`.
        #[arg(long, default_value_t = false)]
        explain_performance: bool,
    },
    /// Run a benchmark suite.
    Bench {
        #[arg(long)]
        weights: PathBuf,
        #[arg(long, default_value = "deepseek-v2-lite-q4")]
        model: String,
        #[arg(long, default_value = "all")]
        suite: String,
        #[arg(long)]
        json: Option<PathBuf>,
        #[arg(long)]
        trace_json: Option<PathBuf>,
        #[arg(long)]
        kernel_profile: Option<PathBuf>,
        #[arg(long, num_args = 0..=1, default_missing_value = "exact-shared", value_name = "MODE")]
        speculate: Option<String>,
        #[arg(long, default_value_t = 4)]
        verify_window: usize,
        #[arg(long, default_value_t = 3)]
        trials: usize,
        /// Completion length for decode/competitive benchmark runs.
        /// Use 16/32/64 for fast public smoke numbers; default remains
        /// the historical 256-token decode suite.
        #[arg(long, default_value_t = 256)]
        max_new_tokens: usize,
        /// Backend selector for single-suite runs (decode/prefill).
        /// `"dismantle"` (default) drives the in-process engine;
        /// `"llamacpp"` and `"mlx"` shell out to competitor binaries.
        #[arg(long, default_value = "dismantle")]
        backend: String,
        /// Enable Metal dispatch tracing and structural allocation/commit
        /// counters. Equivalent to setting DISMANTLE_TRACE_DISPATCH=1.
        #[arg(long, default_value_t = false)]
        trace_dispatch: bool,
    },
    /// Deterministically select an experimental kernel/runtime profile.
    Autotune {
        #[arg(long)]
        weights: PathBuf,
        #[arg(long, default_value = "m3-pro-18gb")]
        profile: String,
        #[arg(long, default_value_t = 8.0)]
        max_hours: f64,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        log: Option<PathBuf>,
        /// Track 2.3: run runtime autotune phase after kernel selection.
        ///
        /// Tests B=1 decode with --profile default vs --profile fast (paired,
        /// contamination-robust). If fast beats default by >3%, records
        /// `runtime_profile: "fast"` in the output profile JSON.
        #[arg(long, default_value_t = false)]
        runtime_autotune: bool,
    },
    /// Benchmark Q4_K GEMV kernels at production shapes and emit JSON.
    BenchQ4kShapes {
        #[arg(long, default_value_t = 100)]
        iters: usize,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Inspect model size, KV-cache budget, current RSS, and M3-Pro fit.
    Doctor {
        #[arg(long)]
        weights: PathBuf,
        #[arg(long, default_value_t = 4096)]
        max_seq_len: usize,
    },
    /// Rank a kernel-profile's recorded autotune evidence using the shipped
    /// offline scorer (profile::select_best / score_measurement). Loads the
    /// profile JSON, prints the chosen variant (highest tps above the quality
    /// floor) with its runtime levers, and a score-ordered table of every
    /// recorded measurement. Pure CPU: JSON parse + scoring only, no model,
    /// no Metal. Use after `dismantle autotune` to audit which (kernel x
    /// runtime-levers) combo wins and why.
    ProfileRank {
        /// Path to a kernel-profile JSON produced by `dismantle autotune`.
        /// (Named `--profile-json` to avoid colliding with the global
        /// `--profile` lever-bundle flag.)
        #[arg(long = "profile-json")]
        profile: PathBuf,
        /// Minimum acceptable quality in [0,1]; candidates below this are
        /// rejected regardless of tps. Defaults to the project quality bar.
        #[arg(long, default_value_t = dismantle_core::profile::DEFAULT_QUALITY_FLOOR)]
        quality_floor: f64,
        /// Emit a machine-readable JSON report instead of the text table.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Run a short diagnostic generation and print routed-expert access status.
    Stats {
        #[arg(long)]
        weights: PathBuf,
        #[arg(long, default_value = "Once upon a time")]
        prompt: String,
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,
        #[arg(long)]
        kernel_profile: Option<PathBuf>,
        #[arg(long)]
        max_routed_expert_ram_mb: Option<usize>,
    },
    /// Print version and the model id, if a weights path is given.
    Version {
        #[arg(long)]
        weights: Option<PathBuf>,
    },
    /// Run a list of prompts through one in-process engine, emitting
    /// per-prompt b3sum hashes of the decoded text. Replaces the
    /// 50-launch shell loop in capture-baseline-50 / token-regression
    /// -- the one model load amortizes across all prompts.
    BatchHash {
        #[arg(long)]
        weights: PathBuf,
        /// Prompts file. One `pNNN:<text>` line per prompt; lines
        /// starting with `#` and blank lines are skipped.
        #[arg(long)]
        prompts: PathBuf,
        #[arg(long, default_value_t = 3)]
        tokens: usize,
        /// Output file for `<id> <N> <hash> <prompt>` lines. If absent
        /// (default), writes to stdout.
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        kernel_profile: Option<PathBuf>,
        #[arg(long, num_args = 0..=1, default_missing_value = "exact-shared", value_name = "MODE")]
        speculate: Option<String>,
        #[arg(long, default_value_t = 4)]
        verify_window: usize,
        #[arg(long, default_value_t = 240000)]
        max_stall_ms: u64,
    },
    /// Print the SHA-256 prefix of all compiled Metal shader sources.
    /// Used to update kernel-profile JSON after shader changes.
    ShaderHash,
    /// Load a model once and serve repeated inference requests over stdin/stdout
    /// (JSON-line protocol). Eliminates the 5-15s model-load cost for each
    /// smoke iteration during development. Use bench_server_driver.sh for
    /// automated multi-request runs.
    BenchServer {
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        kernel_profile: Option<PathBuf>,
        #[arg(long, num_args = 0..=1, default_missing_value = "exact-shared", value_name = "MODE")]
        speculate: Option<String>,
        #[arg(long, default_value_t = 4)]
        verify_window: usize,
        /// Enable Metal dispatch tracing for per-request structural metrics.
        #[arg(long, default_value_t = false)]
        trace_dispatch: bool,
        /// Read requests from stdin (JSON-line). Currently the only supported
        /// transport; HTTP bind will be added in a future sub-phase.
        #[arg(long, default_value_t = true)]
        stdin: bool,
    },
    /// Micro-benchmark an individual Metal GEMV kernel at a production tensor
    /// shape without loading a model. Allocates synthetic buffers, dispatches
    /// the kernel N times, and reports mean/p50/p99/min/max latency in μs.
    /// Use --all to bench every supported kernel at a given shape.
    BenchKernel {
        /// Kernel name, e.g. gemv_q4_k_m_v2_pinned_tcb. Use --all to bench
        /// all kernels that support the given shape.
        #[arg(long, conflicts_with = "all")]
        kernel: Option<String>,
        /// Bench all kernels that support the given shape.
        #[arg(long, default_value_t = false)]
        all: bool,
        /// Matrix shape as ROWSxCOLS, e.g. 1408x2048 (rows=output, cols=input).
        /// Kernel constraints (e.g. cols%256==0) are checked at runtime.
        #[arg(long)]
        shape: String,
        /// Number of dispatches to time. Default 1000.
        #[arg(long, default_value_t = 1000)]
        iterations: usize,
        /// Suppress appending to bench_results/kernel_perf_history.jsonl.
        #[arg(long, default_value_t = false)]
        no_history: bool,
    },
    /// Verify model file integrity: compute SHA-256, print size, check sidecar.
    ///
    /// Usage:
    ///   dismantle verify --weights <path> [--expected-sha256 <hex>]
    ///
    /// Prints the file path, byte size, and SHA-256 hash. When --expected-sha256
    /// is supplied, compares and prints PASS or FAIL. Also reports whether the
    /// matching `.dismantle` sidecar file is present.
    Verify {
        #[arg(long)]
        weights: PathBuf,
        /// Expected SHA-256 hex string (64 lower-case hex chars). When supplied,
        /// the computed hash is compared and PASS / FAIL is printed.
        #[arg(long)]
        expected_sha256: Option<String>,
    },
    /// Bake a `.dismantle` sidecar file from a GGUF model.
    ///
    /// The sidecar encodes pre-processed, Metal-friendly weight representations
    /// (Q4_K predecoded scale tables, pruned LM-head, etc.) so that subsequent
    /// `generate` / `serve` invocations load them directly rather than recomputing
    /// them at startup.
    ///
    /// NOTE: the full bake implementation is a follow-on task. This subcommand
    /// prints a plan and exits cleanly — no output file is written yet.
    BakeSidecar {
        /// Source GGUF file to bake from (required).
        #[arg(long)]
        weights: PathBuf,
        /// Output `.dismantle` sidecar path. Defaults to the same directory as
        /// --weights with the `.dismantle` extension.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Runtime profile to bake for. Determines which levers are active.
        ///   fast      — predec scales + Q4K LM-head (default)
        ///   race      — same as fast; signals max-throughput, quality trades OK
        ///   efficient — same as fast + energy-efficient mode
        ///   exact     — bit-identical conservative path (no quality trades)
        #[arg(long, default_value = "fast", value_name = "PROFILE")]
        profile: String,
        /// Optional hardware kernel-autotune JSON produced by `dismantle autotune`.
        /// When provided, its kernel-routing table is embedded in the sidecar.
        #[arg(long, value_name = "PATH")]
        kernel_profile: Option<PathBuf>,
        /// Optional corpus vocab-whitelist JSON for the pruned LM-head path
        /// (produced by `tools/training/analyze_corpus.py`). When provided,
        /// a pruned-LM-head Q4K blob is included in the sidecar.
        #[arg(long, value_name = "PATH")]
        vocab_prune: Option<PathBuf>,
        /// Number of prompts to use for the top-1 token-agreement quality check
        /// run after baking. 0 skips the quality eval.
        #[arg(long, default_value_t = 50, value_name = "N")]
        quality_eval_count: usize,
        /// Optional mixed-quant tier-map JSON: { "entries": [ {"tensor":"blk.0.ffn_down.weight","dtype":"q6_K"}, ... ] }.
        /// When provided, the baked sidecar carries a per-tensor dtype override
        /// map that the loader reads + logs (Track 4.3). Selection/requant is
        /// out of scope (dead_levers #16) — this only embeds + honors the map.
        #[arg(long, value_name = "PATH")]
        tier_map_json: Option<PathBuf>,
    },
    /// Track 6: offline spec replay-oracle (pure CPU — no Metal, no model
    /// forward). Tokenizes a text corpus with the model's OWN tokenizer
    /// (GGUF-embedded vocab, or a sibling tokenizer.json), then replays the
    /// ids through the shipped n-gram user-draft to measure acceptance.
    /// Prints the GO/MARGINAL/NO-GO tau verdict + per-k tau /
    /// mean_accepted_len / hit_rate / accept_hist / governor_propose_frac.
    ///
    ///   dismantle spec-oracle --corpus prompts.txt \
    ///       --tokenizer-from model.gguf --k 4,7 --warm-frac 0.5 [--json]
    SpecOracle {
        /// UTF-8 text file to score (the whole file is one corpus).
        #[arg(long)]
        corpus: PathBuf,
        /// GGUF whose embedded tokenizer (or sibling tokenizer.json) encodes
        /// the corpus — the SAME tokenizer `generate` uses. CPU-only load
        /// (mmap + metadata parse); the Metal engine is never constructed.
        #[arg(long)]
        tokenizer_from: PathBuf,
        /// Comma-separated lookahead caps to sweep, e.g. `4,7`. Default `4,7`.
        #[arg(long, default_value = "4,7", value_name = "K_LIST")]
        k: String,
        /// Fraction of the corpus (leading prefix) used to warm-start the
        /// n-gram index before scoring begins; the remainder is scored.
        #[arg(long, default_value_t = 0.5, value_name = "FRAC")]
        warm_frac: f64,
        /// Emit the report as JSON instead of the human-readable table.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    // Only the decode-class subcommands (generate/serve/bench) announce the
    // resolved profile; utility subcommands (shader-hash/doctor/version/verify/
    // stats/autotune/bake-sidecar/bench-*) keep clean single-line stdout/stderr.
    let announce_profile = matches!(
        cli.cmd,
        Cmd::Generate { .. } | Cmd::Serve { .. } | Cmd::Bench { .. }
    );
    apply_profile(&cli.profile, announce_profile);
    match cli.cmd {
        Cmd::Serve {
            weights,
            addr,
            max_batch_size,
            speculate,
            verify_window,
            kernel_profile,
            hardware_profile,
            prefill_cache_dir,
            max_routed_expert_ram_mb,
            memory_limit_mb,
            energy_mode,
            explain_performance,
            f16_kv,
            no_f16_kv,
            batch_policy,
            workload,
        } => {
            // --hardware-profile is the preferred alias for --kernel-profile.
            let resolved_kernel_profile = hardware_profile.or(kernel_profile);

            // Parse --profile (global flag) into RuntimeProfile.
            let runtime_profile = cli
                .profile
                .as_deref()
                .and_then(dismantle_serve::RuntimeProfile::from_str)
                .unwrap_or(dismantle_serve::RuntimeProfile::Default);

            // Parse --energy-mode.
            let resolved_energy_mode = energy_mode
                .as_deref()
                .and_then(dismantle_serve::EnergyMode::from_str)
                .unwrap_or(dismantle_serve::EnergyMode::Off);

            // Parse --batch-policy.
            let resolved_batch_policy = batch_policy
                .as_deref()
                .and_then(|s| match s {
                    "default" => Some(dismantle_serve::BatchPolicy::Default),
                    "greedy-first" => Some(dismantle_serve::BatchPolicy::GreedyFirst),
                    "prefix-grouped" => Some(dismantle_serve::BatchPolicy::PrefixGrouped),
                    other => {
                        eprintln!(
                            "[dismantle] warning: unknown --batch-policy {other:?} \
                             (known: default, greedy-first, prefix-grouped); using default"
                        );
                        None
                    }
                })
                .unwrap_or(dismantle_serve::BatchPolicy::Default);

            // Parse --workload.
            let resolved_workload = workload
                .as_deref()
                .and_then(dismantle_serve::WorkloadPack::from_str)
                .unwrap_or(dismantle_serve::WorkloadPack::Default);

            // Resolve --f16-kv / --no-f16-kv into Option<bool>.
            let resolved_f16_kv = if f16_kv {
                Some(true)
            } else if no_f16_kv {
                Some(false)
            } else {
                None
            };

            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(dismantle_serve::run(dismantle_serve::ServeOptions {
                weights,
                addr,
                max_batch_size,
                speculate,
                verify_window,
                kernel_profile: resolved_kernel_profile,
                prefill_cache_dir,
                max_routed_expert_ram_mb,
                memory_limit_mb,
                runtime_profile,
                energy_mode: resolved_energy_mode,
                explain_performance,
                f16_kv: resolved_f16_kv,
                batch_policy: resolved_batch_policy,
                workload: resolved_workload,
                ..Default::default()
            }))
        }
        Cmd::Generate {
            weights,
            prompt,
            max_new_tokens,
            max_seq_len,
            temperature,
            top_k,
            top_p,
            seed,
            kernel_profile,
            speculate,
            verify_window,
            max_stall_ms,
            trace_dispatch,
            max_routed_expert_ram_mb,
            memory_limit_mb,
            vocab_prune_path,
            quant_tier_map_path,
            eagle5_head,
            eagle5_accept_trace,
            prompts_file,
            user_draft,
            user_draft_propose_first,
            explain_performance,
        } => generate_main(
            weights,
            prompt,
            max_new_tokens,
            max_seq_len,
            temperature,
            top_k,
            top_p,
            seed,
            kernel_profile,
            speculate,
            verify_window,
            max_stall_ms,
            trace_dispatch,
            max_routed_expert_ram_mb,
            memory_limit_mb,
            vocab_prune_path,
            quant_tier_map_path,
            eagle5_head,
            eagle5_accept_trace,
            prompts_file,
            user_draft,
            user_draft_propose_first,
            explain_performance,
        ),
        Cmd::Bench {
            weights,
            model,
            suite,
            json,
            trace_json,
            kernel_profile,
            speculate,
            verify_window,
            trials,
            max_new_tokens,
            backend,
            trace_dispatch,
        } => dismantle_bench::run(dismantle_bench::BenchOptions {
            weights,
            model_id: model,
            suite,
            json_out: json,
            trials,
            max_new_tokens,
            trace_json,
            kernel_profile,
            speculate_mode: speculate.unwrap_or_else(|| "off".into()),
            verify_window,
            backend,
            trace_dispatch,
        }),
        Cmd::Autotune {
            weights,
            profile,
            max_hours,
            out,
            log,
            runtime_autotune,
        } => autotune_main(weights, profile, max_hours, out, log, runtime_autotune),
        Cmd::BenchQ4kShapes { iters, out } => bench_q4k_shapes_main(iters, out),
        Cmd::Doctor {
            weights,
            max_seq_len,
        } => doctor_main(weights, max_seq_len),
        Cmd::ProfileRank {
            profile,
            quality_floor,
            json,
        } => profile_rank_main(profile, quality_floor, json),
        Cmd::Stats {
            weights,
            prompt,
            max_new_tokens,
            kernel_profile,
            max_routed_expert_ram_mb,
        } => stats_main(
            weights,
            prompt,
            max_new_tokens,
            kernel_profile,
            max_routed_expert_ram_mb,
        ),
        Cmd::Version { weights } => version_main(weights),
        Cmd::BatchHash {
            weights,
            prompts,
            tokens,
            out,
            kernel_profile,
            speculate,
            verify_window,
            max_stall_ms,
        } => batch_hash_main(
            weights,
            prompts,
            tokens,
            out,
            kernel_profile,
            speculate,
            verify_window,
            max_stall_ms,
        ),
        Cmd::ShaderHash => {
            println!("{}", dismantle_core::profile::shader_source_hash());
            Ok(())
        }
        Cmd::BenchServer {
            weights,
            kernel_profile,
            speculate,
            verify_window,
            trace_dispatch,
            stdin: _,
        } => bench_server::run(bench_server::BenchServerOptions {
            weights,
            kernel_profile,
            speculate,
            verify_window,
            trace_dispatch,
        }),
        Cmd::BenchKernel {
            kernel,
            all,
            shape,
            iterations,
            no_history,
        } => bench_kernel::run(bench_kernel::BenchKernelOptions {
            kernel,
            all,
            shape,
            iterations,
            no_history,
        }),
        Cmd::BakeSidecar {
            weights,
            out,
            profile,
            kernel_profile,
            vocab_prune,
            quality_eval_count,
            tier_map_json,
        } => bake_sidecar_main(
            weights,
            out,
            profile,
            kernel_profile,
            vocab_prune,
            quality_eval_count,
            tier_map_json,
        ),
        Cmd::Verify {
            weights,
            expected_sha256,
        } => verify_main(weights, expected_sha256),
        Cmd::SpecOracle {
            corpus,
            tokenizer_from,
            k,
            warm_frac,
            json,
        } => spec_oracle_main(corpus, tokenizer_from, k, warm_frac, json),
    }
}

/// CLI glue for `dismantle profile-rank`: load the profile JSON (pure CPU),
/// build the report string with [`rank_profile_report`], print it. JSON mode
/// emits the same data as a serde_json object.
fn profile_rank_main(profile: PathBuf, quality_floor: f64, json: bool) -> Result<()> {
    use dismantle_core::profile::KernelProfile;
    let p = KernelProfile::load(&profile)
        .map_err(|e| anyhow::anyhow!("load kernel profile {}: {e}", profile.display()))?;
    if json {
        println!("{}", rank_profile_json(&p, quality_floor)?);
    } else {
        print!("{}", rank_profile_report(&p, quality_floor));
    }
    Ok(())
}

/// Pure, deterministic, GPU-free renderer for the profile-rank text report.
/// Takes the loaded profile + a quality floor, returns the full multi-line
/// report. Selection reuses the SHIPPED scorer (`profile::select_best`); the
/// table is the measurements re-sorted by `profile::score_measurement`
/// descending (NEG_INFINITY-rejected rows sink to the bottom, tagged REJECT).
/// No I/O, no model, no Metal — this is the unit covered by the gate test.
fn rank_profile_report(p: &dismantle_core::profile::KernelProfile, quality_floor: f64) -> String {
    use dismantle_core::profile::{score_measurement, select_best};
    use std::fmt::Write as _;

    let mut s = String::new();
    let _ = writeln!(s, "dismantle profile-rank");
    let _ = writeln!(s, "profile_id: {}", p.profile_id);
    let _ = writeln!(s, "profile_name: {}", p.profile_name);
    let _ = writeln!(s, "model: {} (arch={})", p.model_id, p.model_arch);
    let _ = writeln!(s, "device: {}", p.device_name);
    let _ = writeln!(s, "quality_floor: {quality_floor:.4}");
    let _ = writeln!(s, "measurements: {}", p.evidence.measurements.len());

    match select_best(&p.evidence, quality_floor) {
        Some(best) => {
            let _ = writeln!(s, "selected: {}", best.variant_id);
            let _ = writeln!(s, "  tps: {:.3}", best.tps);
            let _ = writeln!(s, "  quality: {:.4}", best.quality);
            let _ = writeln!(s, "  deterministic_rank: {}", best.deterministic_rank);
            let _ = writeln!(s, "  status: {}", best.status);
            match &best.runtime_levers {
                Some(lv) => {
                    let _ = writeln!(
                        s,
                        "  runtime_levers: profile_name={:?} vocab_prune={:?} \
                         lm_head_path={:?} ffn_down_q4k={} scale_dtype={:?} kv_dtype={:?}",
                        lv.profile_name,
                        lv.vocab_prune,
                        lv.lm_head_path,
                        lv.ffn_down_q4k,
                        lv.scale_dtype,
                        lv.kv_dtype
                    );
                }
                None => {
                    let _ = writeln!(s, "  runtime_levers: (none recorded)");
                }
            }
        }
        None => {
            let _ = writeln!(
                s,
                "selected: NONE (no measurement at or above quality_floor {quality_floor:.4})"
            );
        }
    }

    // Score-ordered table. Sort a (index, score) view by score descending with
    // total_cmp (NEG_INFINITY sinks), tie-break by lower deterministic_rank
    // then variant_id — mirrors select_best's ordering for a stable display.
    let _ = writeln!(s, "rank\tscore\tvariant_id\ttps\tquality\tstatus");
    let mut order: Vec<usize> = (0..p.evidence.measurements.len()).collect();
    order.sort_by(|&ia, &ib| {
        let a = &p.evidence.measurements[ia];
        let b = &p.evidence.measurements[ib];
        score_measurement(b, quality_floor)
            .total_cmp(&score_measurement(a, quality_floor))
            .then_with(|| a.deterministic_rank.cmp(&b.deterministic_rank))
            .then_with(|| a.variant_id.cmp(&b.variant_id))
    });
    for (display_rank, &i) in order.iter().enumerate() {
        let m = &p.evidence.measurements[i];
        let sc = score_measurement(m, quality_floor);
        let score_str = if sc == f64::NEG_INFINITY {
            "REJECT".to_string()
        } else {
            format!("{sc:.3}")
        };
        let _ = writeln!(
            s,
            "{}\t{}\t{}\t{:.3}\t{:.4}\t{}",
            display_rank + 1,
            score_str,
            m.variant_id,
            m.tps,
            m.quality,
            m.status
        );
    }
    s
}

/// JSON sibling of [`rank_profile_report`] for `--json`. Same selection +
/// score-ordering, emitted as a serde_json object. Pure CPU.
fn rank_profile_json(
    p: &dismantle_core::profile::KernelProfile,
    quality_floor: f64,
) -> Result<String> {
    use dismantle_core::profile::{score_measurement, select_best};

    let mut order: Vec<usize> = (0..p.evidence.measurements.len()).collect();
    order.sort_by(|&ia, &ib| {
        let a = &p.evidence.measurements[ia];
        let b = &p.evidence.measurements[ib];
        score_measurement(b, quality_floor)
            .total_cmp(&score_measurement(a, quality_floor))
            .then_with(|| a.deterministic_rank.cmp(&b.deterministic_rank))
            .then_with(|| a.variant_id.cmp(&b.variant_id))
    });
    let ranked: Vec<serde_json::Value> = order
        .iter()
        .map(|&i| {
            let m = &p.evidence.measurements[i];
            let sc = score_measurement(m, quality_floor);
            serde_json::json!({
                "variant_id": m.variant_id,
                "tps": m.tps,
                "quality": m.quality,
                "status": m.status,
                "deterministic_rank": m.deterministic_rank,
                "score": if sc == f64::NEG_INFINITY { serde_json::Value::Null } else { serde_json::json!(sc) },
                "rejected": sc == f64::NEG_INFINITY,
            })
        })
        .collect();
    let selected = select_best(&p.evidence, quality_floor).map(
        |b| serde_json::json!({ "variant_id": b.variant_id, "tps": b.tps, "quality": b.quality }),
    );
    let obj = serde_json::json!({
        "profile_id": p.profile_id,
        "profile_name": p.profile_name,
        "quality_floor": quality_floor,
        "selected": selected,
        "ranked": ranked,
    });
    Ok(serde_json::to_string_pretty(&obj)?)
}

fn doctor_main(weights: PathBuf, max_seq_len: usize) -> Result<()> {
    use dismantle_core::gguf::GgufFile;

    let rss_before = current_rss_mb();
    let file_bytes = std::fs::metadata(&weights)?.len();
    let gguf = GgufFile::open(&weights)?;
    let rss_after = current_rss_mb();
    let arch = gguf.architecture().unwrap_or("unknown");
    let name = gguf.name().unwrap_or("unknown");
    let get_u32 = |keys: &[&str]| {
        keys.iter()
            .find_map(|k| gguf.metadata.get(*k).and_then(|v| v.as_u32()))
            .map(|v| v as usize)
    };

    let layers = get_u32(&[
        &format!("{arch}.block_count"),
        "deepseek2.block_count",
        "qwen2.block_count",
        "llama.block_count",
    ])
    .unwrap_or(0);
    let hidden = get_u32(&[
        &format!("{arch}.embedding_length"),
        "deepseek2.embedding_length",
        "qwen2.embedding_length",
        "llama.embedding_length",
    ])
    .unwrap_or(0);
    let heads = get_u32(&[
        &format!("{arch}.attention.head_count"),
        "deepseek2.attention.head_count",
        "qwen2.attention.head_count",
        "llama.attention.head_count",
    ])
    .unwrap_or(0);
    let kv_heads = get_u32(&[
        &format!("{arch}.attention.head_count_kv"),
        "deepseek2.attention.head_count_kv",
        "qwen2.attention.head_count_kv",
        "llama.attention.head_count_kv",
    ])
    .unwrap_or(heads);
    let context = get_u32(&[
        &format!("{arch}.context_length"),
        "deepseek2.context_length",
        "qwen2.context_length",
        "llama.context_length",
    ])
    .unwrap_or(max_seq_len)
    .min(max_seq_len);

    let deepseek_head_dim = get_u32(&["deepseek2.attention.qk_nope_head_dim"]).unwrap_or(0)
        + get_u32(&["deepseek2.attention.qk_rope_head_dim"]).unwrap_or(0);
    let head_dim = if deepseek_head_dim > 0 {
        deepseek_head_dim
    } else {
        hidden.checked_div(heads).unwrap_or(0)
    };
    let kv_cache_bytes = layers
        .saturating_mul(context)
        .saturating_mul(kv_heads)
        .saturating_mul(head_dim)
        .saturating_mul(2)
        .saturating_mul(std::mem::size_of::<f32>());
    let total_working_bytes = file_bytes.saturating_add(kv_cache_bytes as u64);
    let swap_risk = if total_working_bytes > 16_u64 * 1024 * 1024 * 1024 {
        "high"
    } else if total_working_bytes > 14_u64 * 1024 * 1024 * 1024 {
        "watch"
    } else {
        "low"
    };

    println!("dismantle doctor");
    println!("model: {name}");
    println!("architecture: {arch}");
    println!("weights: {}", weights.display());
    println!("weights_bytes: {} ({:.2} GiB)", file_bytes, gib(file_bytes));
    println!(
        "mmap_bytes: {} ({:.2} GiB)",
        gguf.mmap.len(),
        gib(gguf.mmap.len() as u64)
    );
    println!("tensors: {}", gguf.tensor_count);
    println!("layers: {layers}");
    println!("hidden: {hidden}");
    println!("kv_heads: {kv_heads}");
    println!("head_dim_estimate: {head_dim}");
    println!("context_estimate: {context}");
    println!(
        "kv_cache_estimate: {} ({:.2} GiB)",
        kv_cache_bytes,
        gib(kv_cache_bytes as u64)
    );
    println!(
        "weights_plus_kv_estimate: {} ({:.2} GiB)",
        total_working_bytes,
        gib(total_working_bytes)
    );
    if let Some(v) = rss_before {
        println!("rss_before_mmap_mb: {v:.1}");
    }
    if let Some(v) = rss_after {
        println!("rss_after_mmap_mb: {v:.1}");
    }
    println!("m3_pro_18gb_swap_risk: {swap_risk}");
    Ok(())
}

fn stats_main(
    weights: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    kernel_profile: Option<PathBuf>,
    max_routed_expert_ram_mb: Option<usize>,
) -> Result<()> {
    use anyhow::Context;
    use dismantle_core::{
        gguf::GgufFile, profile::KernelProfile, EngineConfig, GenerateRequest, SamplingParams,
        StreamEvent,
    };

    let gguf = GgufFile::open(&weights)?;
    let arch = gguf.architecture().unwrap_or("unknown");
    let name = gguf.name().unwrap_or("unknown");
    let get_u32 = |keys: &[&str]| {
        keys.iter()
            .find_map(|k| gguf.metadata.get(*k).and_then(|v| v.as_u32()))
            .map(|v| v as usize)
    };
    let block_key = format!("{arch}.block_count");
    let expert_key = format!("{arch}.expert_count");
    let expert_used_key = format!("{arch}.expert_used_count");
    let layers = get_u32(&[
        block_key.as_str(),
        "deepseek2.block_count",
        "llama.block_count",
        "qwen2moe.block_count",
    ])
    .unwrap_or(0);
    let experts = get_u32(&[
        expert_key.as_str(),
        "deepseek2.expert_count",
        "llama.expert_count",
        "qwen2moe.expert_count",
    ])
    .unwrap_or(0);
    let top_k = get_u32(&[
        expert_used_key.as_str(),
        "deepseek2.expert_used_count",
        "llama.expert_used_count",
        "qwen2moe.expert_used_count",
    ])
    .unwrap_or(0);

    let profile = match kernel_profile.as_ref() {
        Some(path) => Some(KernelProfile::load(path)?),
        None => None,
    };
    let cfg = EngineConfig {
        kernel_profile: profile,
        max_routed_expert_ram_mb,
        ..Default::default()
    };
    let mut engine = dismantle_core::model::load_engine(&weights, cfg)
        .with_context(|| format!("load engine from {}", weights.display()))?;
    let req = GenerateRequest {
        prompt: prompt.clone(),
        max_new_tokens,
        sampling: SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(42),
        },
        stop: Vec::new(),
        abort: None,
        max_stall_ms: 60_000,
    };
    let mut decoded = String::new();
    let mut final_done = None;
    engine.generate(req, &mut |ev| match ev {
        StreamEvent::Token { text, .. } => decoded.push_str(&text),
        StreamEvent::Done { stats, reason } => final_done = Some((stats, reason)),
    })?;
    let (stats, reason) = final_done.context("generation completed without Done event")?;

    println!("dismantle stats");
    println!("model: {name}");
    println!("architecture: {arch}");
    println!("weights: {}", weights.display());
    println!("prompt: {prompt:?}");
    println!("decoded: {:?}", decoded.trim());
    println!("finish_reason: {:?}", reason);
    println!("prompt_tokens: {}", stats.prompt_tokens);
    println!("completion_tokens: {}", stats.completion_tokens);
    println!("decode_ms: {:.1}", stats.decode_ms);
    println!(
        "offload_budget_mb: {}",
        max_routed_expert_ram_mb
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unlimited".into())
    );
    println!("layers: {layers}");
    println!("routed_experts_per_layer: {experts}");
    println!("top_k_routed: {top_k}");

    // Print per-layer per-expert access counts if the cache is active.
    if let Some(counts) = engine.expert_access_counts() {
        let total_accesses: u64 = counts.iter().flat_map(|l| l.iter()).sum();
        println!("expert_tracking: active  total_accesses={total_accesses}");
        println!("layer\texpert\taccess_count\t%_of_total");
        for (li, layer) in counts.iter().enumerate() {
            if layer.is_empty() {
                continue;
            }
            for (eid, &count) in layer.iter().enumerate() {
                let pct = if total_accesses > 0 {
                    count as f64 / total_accesses as f64 * 100.0
                } else {
                    0.0
                };
                println!("{li}\t{eid}\t{count}\t{pct:.2}");
            }
        }
    } else {
        println!("expert_tracking: disabled (pass --max-routed-expert-ram-mb to enable)");
    }
    Ok(())
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0 / 1024.0
}

fn current_rss_mb() -> Option<f64> {
    let pid = std::process::id().to_string();
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok()?;
    let kb = String::from_utf8(out.stdout)
        .ok()?
        .trim()
        .parse::<f64>()
        .ok()?;
    Some(kb / 1024.0)
}

struct Q4ShapeBenchSummary {
    json: serde_json::Value,
    winners: BTreeMap<String, String>,
}

fn bench_q4k_shapes_main(iters: usize, out: Option<PathBuf>) -> Result<()> {
    let summary = bench_q4k_shapes(iters)?;
    let text = serde_json::to_string_pretty(&summary.json)?;
    if let Some(path) = out {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, &text)?;
        println!("wrote Q4_K shape bench: {}", path.display());
    } else {
        println!("{text}");
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn bench_q4k_shapes(iters: usize) -> Result<Q4ShapeBenchSummary> {
    use dismantle_core::metal::MetalContext;

    if iters == 0 {
        anyhow::bail!("--iters must be positive");
    }

    let ctx = MetalContext::new()?;
    let shapes = [
        ("gate_up_1024x4096", 1024usize, 4096usize),
        ("down_4096x1024", 4096usize, 1024usize),
        ("dense_4096x4096", 4096usize, 4096usize),
    ];
    let kernels = ["v2", "simdmat", "v3_dual", "llama_port"];
    let mut shape_json = Vec::with_capacity(shapes.len());
    let mut winners = BTreeMap::new();

    for (label, rows, cols) in shapes {
        let w_bytes = synthetic_q4_k_bytes(rows * (cols / 256));
        let model_buf = ctx.new_buffer_with_bytes(&w_bytes);
        let x = synthetic_input(cols);
        let mut results = serde_json::Map::new();
        let mut best_name = "";
        let mut best_us = f64::INFINITY;

        for kernel in kernels {
            let mut out = vec![0.0f32; rows];
            for _ in 0..5 {
                run_q4k_shape_kernel(&ctx, &model_buf, &w_bytes, rows, cols, &x, &mut out, kernel)?;
            }
            let start = Instant::now();
            for _ in 0..iters {
                run_q4k_shape_kernel(&ctx, &model_buf, &w_bytes, rows, cols, &x, &mut out, kernel)?;
            }
            let mean_us = start.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;
            if mean_us < best_us {
                best_us = mean_us;
                best_name = kernel;
            }
            results.insert(
                kernel.to_string(),
                serde_json::json!({
                    "mean_us": mean_us,
                }),
            );
        }

        let key = format!("{rows}x{cols}");
        winners.insert(key.clone(), best_name.to_string());
        shape_json.push(serde_json::json!({
            "label": label,
            "key": key,
            "rows": rows,
            "cols": cols,
            "winner": best_name,
            "kernels": results,
        }));
    }

    Ok(Q4ShapeBenchSummary {
        winners,
        json: serde_json::json!({
            "iters": iters,
            "shapes": shape_json,
        }),
    })
}

#[cfg(not(target_os = "macos"))]
fn bench_q4k_shapes(_iters: usize) -> Result<Q4ShapeBenchSummary> {
    anyhow::bail!("bench-q4k-shapes requires macOS Metal")
}

#[cfg(target_os = "macos")]
fn run_q4k_shape_kernel(
    ctx: &dismantle_core::metal::MetalContext,
    model_buf: &dismantle_core::metal::PinnedBuffer,
    w_bytes: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
    kernel: &str,
) -> Result<()> {
    match kernel {
        "v2" => dismantle_core::kernels::gemv_q4_k_m_v2_pinned(
            ctx,
            model_buf,
            0,
            w_bytes.len(),
            rows,
            cols,
            x,
            out,
        )?,
        "simdmat" => dismantle_core::kernels::gemv_q4_k_m_simdmat_pinned(
            ctx,
            model_buf,
            0,
            w_bytes.len(),
            rows,
            cols,
            x,
            out,
        )?,
        "v3_dual" => dismantle_core::kernels::gemv_q4_k_m_v3_dual_pinned(
            ctx,
            model_buf,
            0,
            w_bytes.len(),
            rows,
            cols,
            x,
            out,
        )?,
        "llama_port" => dismantle_core::kernels::gemv_q4_k_m_llama_port_pinned(
            ctx,
            model_buf,
            0,
            w_bytes.len(),
            rows,
            cols,
            x,
            out,
        )?,
        other => anyhow::bail!("unknown Q4_K shape kernel {other:?}"),
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn synthetic_q4_k_bytes(n_blocks: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; n_blocks * 144];
    for b in 0..n_blocks {
        let off = b * 144;
        bytes[off] = 0x00;
        bytes[off + 1] = 0x3c; // f16 1.0
        bytes[off + 2] = 0x00;
        bytes[off + 3] = 0x00; // f16 0.0 dmin
        for i in 4..144 {
            bytes[off + i] = ((b * 13 + i * 37) & 0xff) as u8;
        }
    }
    bytes
}

#[cfg(target_os = "macos")]
fn synthetic_input(cols: usize) -> Vec<f32> {
    (0..cols).map(|i| ((i % 97) as f32 - 48.0) / 97.0).collect()
}

fn autotune_main(
    weights: PathBuf,
    profile: String,
    max_hours: f64,
    out: PathBuf,
    log: Option<PathBuf>,
    runtime_autotune: bool,
) -> Result<()> {
    use dismantle_core::gguf::GgufFile;
    use dismantle_core::profile::{build_deterministic_profile, AutotuneOptions};

    if max_hours <= 0.0 {
        anyhow::bail!("--max-hours must be positive");
    }
    let gguf = GgufFile::open(&weights)?;
    let opts = AutotuneOptions {
        profile,
        max_hours,
        target_tps: 60.0,
    };
    let mut selected = build_deterministic_profile(&gguf, &opts);
    let q4_shape_bench = bench_q4k_shapes(100).ok();
    if let Some(summary) = q4_shape_bench.as_ref() {
        selected.selected.gemm_q4_k_schedule = "per_shape".into();
        selected.selected.gemm_q4_k_schedule_per_shape = summary.winners.clone();
    }
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let log_path = log.unwrap_or_else(|| out.with_extension("jsonl"));
    if let Some(parent) = log_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut log_lines = Vec::with_capacity(selected.evidence.measurements.len() + 4);
    log_lines.push(
        serde_json::json!({
            "event": "autotune-start",
            "profile": selected.profile_name,
            "profile_id": selected.profile_id,
            "model_id": selected.model_id,
            "device": selected.device_name,
            "shader_hash": selected.shader_hash,
            "tensor_layout_hash": selected.tensor_layout_hash,
            "max_hours": max_hours,
        })
        .to_string(),
    );
    for m in &selected.evidence.measurements {
        log_lines.push(
            serde_json::json!({
                "event": "candidate",
                "profile_id": selected.profile_id,
                "variant_id": m.variant_id,
                "deterministic_rank": m.deterministic_rank,
                "score": m.score,
                "status": m.status,
            })
            .to_string(),
        );
    }
    if let Some(summary) = q4_shape_bench.as_ref() {
        log_lines.push(
            serde_json::json!({
                "event": "bench-q4k-shapes",
                "profile_id": selected.profile_id,
                "summary": summary.json,
            })
            .to_string(),
        );
    }
    log_lines.push(
        serde_json::json!({
            "event": "selected",
            "profile_id": selected.profile_id,
            "variant_id": selected.selected.id,
            "target_tps": selected.evidence.target_tps,
        })
        .to_string(),
    );

    // ── Track 2.3: runtime autotune phase ──────────────────────────────────
    // When --runtime-autotune is set, run a paired B=1 decode comparison:
    // default profile vs fast profile. Both share the same process, so
    // contamination cancels in the relative delta (paired bench rule).
    //
    // If fast beats default by >3%, record `runtime_profile: "fast"` in
    // the output profile JSON (injected into the serialized JSON because
    // KernelVariant doesn't carry a runtime_profile field and we do not
    // own dismantle-core).
    let mut runtime_profile_label = "default".to_string();
    if runtime_autotune {
        eprintln!("[autotune] phase 2: runtime autotune (paired default vs fast, B=1)");
        let runtime_profile_choice =
            run_runtime_autotune_phase(&weights, selected.profile_id.as_str());
        runtime_profile_label = runtime_profile_choice
            .clone()
            .unwrap_or_else(|| "default".to_string());
        match &runtime_profile_choice {
            Some(chosen) => {
                eprintln!("[autotune] runtime autotune result: fast wins by >3% — selecting runtime_profile={chosen}");
            }
            None => {
                eprintln!("[autotune] runtime autotune result: delta <3% or fast did not win — keeping default");
            }
        }
        log_lines.push(
            serde_json::json!({
                "event": "runtime-autotune",
                "profile_id": selected.profile_id,
                "runtime_profile": &runtime_profile_label,
                "fast_won": runtime_profile_choice.is_some(),
            })
            .to_string(),
        );
    }

    // Serialize to JSON value so we can inject `runtime_profile` at the
    // top level when runtime_autotune ran.
    let mut profile_json: serde_json::Value = serde_json::to_value(&selected)?;
    if runtime_autotune {
        if let serde_json::Value::Object(ref mut map) = profile_json {
            map.insert(
                "runtime_profile".to_string(),
                serde_json::Value::String(runtime_profile_label.clone()),
            );
        }
    }
    std::fs::write(&out, serde_json::to_string_pretty(&profile_json)?)?;
    std::fs::write(&log_path, log_lines.join("\n") + "\n")?;
    println!("wrote kernel profile: {}", out.display());
    println!("wrote autotune log: {}", log_path.display());
    println!("profile_id: {}", selected.profile_id);
    println!("selected_variant: {}", selected.selected.id);
    println!("device: {}", selected.device_name);
    println!("target_tps: {:.1}", selected.evidence.target_tps);
    if runtime_autotune {
        println!("runtime_profile: {runtime_profile_label}");
    }
    Ok(())
}

/// Run a paired B=1 decode comparison between `--profile default` and
/// `--profile fast`. Returns `Some("fast")` if the fast profile beats
/// default by more than 3%, `None` otherwise.
///
/// Both runs share this process so contamination cancels in the delta.
/// The measurement uses wall-clock time for a fixed number of decode steps
/// via the bench harness (no model I/O; just timing the forward pass).
fn run_runtime_autotune_phase(weights: &std::path::Path, profile_id: &str) -> Option<String> {
    use dismantle_core::{EngineConfig, GenerateRequest, SamplingParams, StreamEvent};

    let n_tokens: usize = 32; // decode budget: enough for a stable mean
    let prompt = "The quick brown fox jumps over the lazy dog.";

    // Helper: run one decode trial and return tokens/second.
    let measure_tps = |extra_vars: &[(&str, &str)]| -> Option<f64> {
        // Set profile env vars (only if not already set by the user).
        let mut set_vars: Vec<&str> = Vec::new();
        for &(k, v) in extra_vars {
            if std::env::var_os(k).is_none() {
                std::env::set_var(k, v);
                set_vars.push(k);
            }
        }

        let cfg = EngineConfig {
            max_seq_len: 512,
            max_batch_size: 1,
            ..Default::default()
        };
        let result = (|| -> Option<f64> {
            let mut engine = dismantle_core::model::load_engine(weights, cfg).ok()?;
            let req = GenerateRequest {
                prompt: prompt.into(),
                max_new_tokens: n_tokens,
                sampling: SamplingParams {
                    temperature: 0.0,
                    top_k: 0,
                    top_p: 1.0,
                    repetition_penalty: 1.0,
                    seed: Some(42),
                },
                stop: Vec::new(),
                abort: None,
                max_stall_ms: 30_000,
            };
            let mut decode_ms = 0.0f64;
            let mut completion_tokens = 0usize;
            engine
                .generate(req, &mut |ev| {
                    if let StreamEvent::Done { stats, .. } = ev {
                        decode_ms = stats.decode_ms;
                        completion_tokens = stats.completion_tokens;
                    }
                })
                .ok()?;
            if decode_ms > 0.0 && completion_tokens > 0 {
                Some(completion_tokens as f64 / (decode_ms / 1000.0))
            } else {
                None
            }
        })();

        // Clean up vars we set so the next run starts fresh.
        for k in &set_vars {
            std::env::remove_var(k);
        }
        result
    };

    eprintln!("[autotune/runtime] measuring default profile (profile_id={profile_id})");
    let tps_default = match measure_tps(&[]) {
        Some(v) => v,
        None => {
            eprintln!("[autotune/runtime] default profile measurement failed; skipping");
            return None;
        }
    };

    eprintln!("[autotune/runtime] measuring fast profile (profile_id={profile_id})");
    let fast_vars = [
        ("DISMANTLE_QWEN_VOCAB_PRUNE", "32000"),
        ("DISMANTLE_QWEN_Q4K_LMHEAD", "1"),
        ("DISMANTLE_QWEN_FFN_DOWN_Q4K", "1"),
        ("DISMANTLE_QWEN_Q4K_PREDEC", "1"),
        ("DISMANTLE_QWEN_PREDEC_F16SCALES", "1"),
    ];
    let tps_fast = match measure_tps(&fast_vars) {
        Some(v) => v,
        None => {
            eprintln!("[autotune/runtime] fast profile measurement failed; skipping");
            return None;
        }
    };

    let delta_pct = (tps_fast - tps_default) / tps_default.max(1e-6) * 100.0;
    eprintln!(
        "[autotune/runtime] default={tps_default:.1} tps  fast={tps_fast:.1} tps  \
         delta={delta_pct:+.1}%  threshold=+3.0%"
    );

    if delta_pct > 3.0 {
        Some("fast".into())
    } else {
        None
    }
}

/// One-shot startup banner for `dismantle generate --explain-performance`.
/// Mirrors `serve --explain-performance` (dismantle-serve/src/lib.rs:497).
/// Reads model identity cheaply via GGUF metadata and DERIVES the levers
/// from the env vars that `apply_profile` already resolved (the binary cannot
/// call QwenDense::lm_head_path — it is pub(crate); the authoritative value is
/// printed on the `[stats-json]` line after the first generated token).
fn explain_performance_banner(weights: &std::path::Path, max_seq_len: usize) {
    use dismantle_core::env_on;
    // env_opt_out: true unless explicitly 0/false/off/no (default-ON lever).
    let env_opt_out = |k: &str| -> bool {
        match std::env::var(k) {
            Ok(v) => !matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            ),
            Err(_) => true,
        }
    };

    // Model identity (cheap: header-only GGUF open, same as doctor_main).
    let (model_name, arch) = match dismantle_core::gguf::GgufFile::open(weights) {
        Ok(g) => (
            g.name().unwrap_or("unknown").to_string(),
            g.architecture().unwrap_or("unknown").to_string(),
        ),
        Err(_) => ("unknown".to_string(), "unknown".to_string()),
    };

    // Resolve the active profile label from env (apply_profile already ran).
    // The unset→fast-minus-f16scales default sets these four ON and leaves
    // PREDEC_F16SCALES OFF; --profile exact leaves the bundle unset.
    let vocab_prune = std::env::var("DISMANTLE_QWEN_VOCAB_PRUNE").ok();
    let q4k_lmhead = env_on("DISMANTLE_QWEN_Q4K_LMHEAD");
    let ffn_down_q4k = env_on("DISMANTLE_QWEN_FFN_DOWN_Q4K");
    let predec = env_opt_out("DISMANTLE_QWEN_Q4K_PREDEC"); // default-ON
    let f16_scales = env_on("DISMANTLE_QWEN_PREDEC_F16SCALES");
    let f16_kv = env_on("DISMANTLE_QWEN_F16_KV");
    let w4a8 = env_on("DISMANTLE_QWEN_W4A8");

    let profile_label = if vocab_prune.is_some() && q4k_lmhead && ffn_down_q4k && f16_scales {
        "fast (full)"
    } else if vocab_prune.is_some() && q4k_lmhead && ffn_down_q4k {
        "default (unset→fast minus f16-scales)"
    } else if !q4k_lmhead && !ffn_down_q4k && vocab_prune.is_none() {
        "exact / bit-identical"
    } else {
        "custom (env overrides)"
    };

    // DERIVED lm_head path — mirrors QwenDense::lm_head_path (qwen_dense.rs:3392).
    // The model may not actually be loaded on Metal yet, so this is a best-
    // effort prediction; the real value lands on the [stats-json] line.
    let any_q4k = q4k_lmhead || w4a8;
    let lmhead_predec = predec && env_opt_out("DISMANTLE_QWEN_LMHEAD_PREDEC");
    let predec_f16s = predec && f16_scales;
    let lm_head_pred = if !any_q4k {
        "f16 (no Q4K LM head)"
    } else if w4a8 {
        "q4k (w4a8)"
    } else if predec_f16s && lmhead_predec {
        "q4k-predec-f16s"
    } else if lmhead_predec {
        "q4k-predec"
    } else {
        "q4k"
    };

    // Sidecar presence (predec/Q4K_FAST .dismantle next to the weights).
    let sidecar_status = match dismantle_core::sidecar::sidecar_path_for(weights) {
        p if p.exists() => format!("present ({})", p.display()),
        p => format!("absent ({})", p.display()),
    };

    let token_only = if any_q4k {
        "available (greedy/temp=0 → token-only lane via Q4K LM head)"
    } else {
        "unavailable (full-logits LM head; greedy still works but reads B×vocab)"
    };

    // Full-logits cost at B=1 (generate is single-stream). Qwen vocab=151936.
    let full_logits_mb = 151936.0_f64 * 4.0 / 1_048_576.0;

    eprintln!(
        "dismantle generate — performance summary\n\
         \x20 model:              {model_name} ({arch})\n\
         \x20 active profile:     {profile_label}\n\
         \x20 fast levers:        vocab_prune={} q4k_lmhead={q4k_lmhead} \
ffn_down_q4k={ffn_down_q4k} predec={predec} f16_scales={f16_scales}\n\
         \x20 f16 KV cache:       {f16_kv}   (max_seq_len={max_seq_len})\n\
         \x20 lm_head path (pred):{lm_head_pred}  (authoritative value on [stats-json] after token 1)\n\
         \x20 sidecar:            {sidecar_status}\n\
         \x20 token-only lane:    {token_only}\n\
         \x20 full-logits cost:   vocab×4 ≈ {full_logits_mb:.1} MB/step at B=1 (expensive; avoided when token-only is active)",
        vocab_prune.as_deref().unwrap_or("off"),
    );
}

#[allow(clippy::too_many_arguments)]
fn generate_main(
    weights: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    max_seq_len: usize,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    seed: Option<u64>,
    kernel_profile: Option<PathBuf>,
    speculate: Option<String>,
    verify_window: usize,
    max_stall_ms: u64,
    trace_dispatch: bool,
    max_routed_expert_ram_mb: Option<usize>,
    memory_limit_mb: Option<usize>,
    vocab_prune_path: Option<PathBuf>,
    quant_tier_map_path: Option<PathBuf>,
    eagle5_head: Option<PathBuf>,
    eagle5_accept_trace: Option<PathBuf>,
    prompts_file: Option<PathBuf>,
    user_draft: bool,
    user_draft_propose_first: bool,
    explain_performance: bool,
) -> Result<()> {
    use dismantle_core::{
        profile::KernelProfile, EngineConfig, GenerateRequest, SamplingParams, SpeculateMode,
        StopReason, StreamEvent,
    };
    use std::io::Write;
    use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
    use std::sync::Arc;

    // Two-stage Ctrl-C: first press flips the abort flag (engine bails
    // at the next token boundary, prints partial stats); second press
    // exits hard with status 130. Without this, CPU-bound prefill on a
    // 9 GB model is unkillable from the terminal.
    let abort = Arc::new(AtomicBool::new(false));
    let press_count = Arc::new(AtomicU8::new(0));
    {
        let abort = Arc::clone(&abort);
        let press_count = Arc::clone(&press_count);
        ctrlc::set_handler(move || {
            let n = press_count.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                eprintln!("\n[dismantle] Ctrl-C -- aborting at next token boundary; press again to force-exit");
                abort.store(true, Ordering::SeqCst);
            } else {
                eprintln!("\n[dismantle] second Ctrl-C -- force-exit");
                std::process::exit(130);
            }
        })
        .map_err(|e| anyhow::anyhow!("install Ctrl-C handler: {e}"))?;
    }

    let speculate_mode = SpeculateMode::from_cli(speculate.as_deref(), false)?;
    if let Some(path) = eagle5_accept_trace.as_ref() {
        std::env::set_var("DISMANTLE_QWEN_EAGLE5_ACCEPT_TRACE", path);
    }
    // L3.1 §2.1b — expose the user-ngram draft (and its propose-first variant)
    // on the CLI by setting the env the core reads via `env_on`. Without this
    // wiring the draft is unreachable from `dismantle generate` (the gap
    // diagnosed in reports/move2_user_draft_diagnosis.md). propose-first
    // implies the draft is on.
    if user_draft || user_draft_propose_first {
        std::env::set_var("DISMANTLE_QWEN_USER_DRAFT", "1");
    }
    if user_draft_propose_first {
        std::env::set_var("DISMANTLE_QWEN_USER_DRAFT_PROPOSE_FIRST", "1");
    }
    let profile = match kernel_profile.as_ref() {
        Some(path) => Some(KernelProfile::load(path)?),
        None => None,
    };
    let cfg = EngineConfig {
        max_seq_len,
        max_batch_size: 1,
        speculate: speculate_mode != SpeculateMode::Off,
        speculate_mode,
        verify_window,
        prefill_cache_dir: None,
        kernel_profile: profile,
        trace_dispatch,
        max_routed_expert_ram_mb,
        memory_limit_mb,
        vocab_prune_path,
        quant_tier_map_path,
        eagle5_head_path: eagle5_head,
        // CLI force-cpu is via the DISMANTLE_FORCE_CPU env var (checked at load);
        // the config field is the programmatic knob (tests / embedders).
        force_cpu: false,
        concurrent_qkv: false,
    };
    if explain_performance {
        explain_performance_banner(&weights, max_seq_len);
    }
    let mut engine = dismantle_core::model::load_engine(&weights, cfg)?;

    // Build the prompt list: either every line of --prompts-file (capture
    // corpus mode — model loaded once, all prompts decoded in sequence) or
    // the single --prompt. Blank lines and leading/trailing whitespace are
    // dropped so a hand-edited prompts file is forgiving.
    let prompts: Vec<String> = match prompts_file.as_ref() {
        Some(path) => {
            let raw = std::fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("read prompts file {}: {e}", path.display()))?;
            let v: Vec<String> = raw
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
                .map(|l| l.to_string())
                .collect();
            if v.is_empty() {
                return Err(anyhow::anyhow!(
                    "prompts file {} has no prompts",
                    path.display()
                ));
            }
            eprintln!("[capture] {} prompts from {}", v.len(), path.display());
            v
        }
        None => {
            if prompt.is_empty() {
                return Err(anyhow::anyhow!("provide --prompt or --prompts-file"));
            }
            vec![prompt]
        }
    };

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let n_prompts = prompts.len();
    for (idx, p) in prompts.into_iter().enumerate() {
        if abort.load(Ordering::SeqCst) {
            break;
        }
        if n_prompts > 1 {
            eprintln!("[capture] prompt {}/{}", idx + 1, n_prompts);
        }
        let req = GenerateRequest {
            prompt: p,
            max_new_tokens,
            sampling: SamplingParams {
                temperature,
                top_k,
                top_p,
                repetition_penalty: 1.0,
                seed,
            },
            stop: Vec::new(),
            abort: Some(Arc::clone(&abort)),
            max_stall_ms,
        };
        let mut sink = |ev: StreamEvent| match ev {
            StreamEvent::Token { text, .. } => {
                let _ = out.write_all(text.as_bytes());
                let _ = out.flush();
            }
            StreamEvent::Done { stats, reason } => {
                let _ = out.write_all(b"\n");
                let _ = out.flush();
                let dec = (stats.completion_tokens as f64) / (stats.decode_ms / 1000.0).max(1e-6);
                let reason_s = match reason {
                    StopReason::MaxTokens => "max_tokens",
                    StopReason::StopString => "stop_string",
                    StopReason::Eos => "eos",
                    StopReason::Aborted => "aborted",
                };
                eprintln!(
                    "\n[stats] reason={} prompt={} completion={} prefill_ms={:.1} decode_ms={:.1} dec_tps={:.2} dispatches_per_fwd={} draft_accepted={} draft_rejected={} profile={}",
                    reason_s,
                    stats.prompt_tokens,
                    stats.completion_tokens,
                    stats.prefill_ms,
                    stats.decode_ms,
                    dec,
                    stats.dispatches_per_forward,
                    stats.draft_accepted,
                    stats.draft_rejected,
                    stats.profile_id.as_deref().unwrap_or("none")
                );
                // Track 0.2/8.3: machine-readable per-request stats (parseable
                // by report-card / harnesses; carries lm_head_path + the
                // observability counters alongside dec_tps).
                eprintln!("[stats-json] {}", stats.stats_json());
            }
        };
        engine.generate(req, &mut sink)?;
    }
    // L1.1 attention-mass oracle (default-off): dump the per-layer
    // concentration curve accumulated during prefill. No-op unless
    // DISMANTLE_QWEN_ATTN_CAPTURE=1.
    dismantle_core::stateful::attn_capture::flush();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn batch_hash_main(
    weights: PathBuf,
    prompts_path: PathBuf,
    tokens: usize,
    out_path: Option<PathBuf>,
    kernel_profile: Option<PathBuf>,
    speculate: Option<String>,
    verify_window: usize,
    max_stall_ms: u64,
) -> Result<()> {
    use dismantle_core::{
        profile::KernelProfile, EngineConfig, GenerateRequest, SamplingParams, SpeculateMode,
        StreamEvent,
    };
    use std::io::Write;
    use std::process::{Command, Stdio};

    let prompts_text = std::fs::read_to_string(&prompts_path)?;
    let prompts: Vec<(String, String)> = prompts_text
        .lines()
        .filter(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with('#')
        })
        .filter_map(|l| {
            let l = l.trim();
            let (id, prompt) = l.split_once(':')?;
            Some((id.trim().to_string(), prompt.to_string()))
        })
        .collect();

    if prompts.is_empty() {
        anyhow::bail!("no prompts parsed from {}", prompts_path.display());
    }
    eprintln!(
        "[batch-hash] loaded {} prompt(s) from {}",
        prompts.len(),
        prompts_path.display()
    );

    let speculate_mode = SpeculateMode::from_cli(speculate.as_deref(), false)?;
    let profile = match kernel_profile.as_ref() {
        Some(path) => Some(KernelProfile::load(path)?),
        None => None,
    };
    let cfg = EngineConfig {
        max_seq_len: 4096,
        max_batch_size: 1,
        speculate: speculate_mode != SpeculateMode::Off,
        speculate_mode,
        verify_window,
        prefill_cache_dir: None,
        kernel_profile: profile,
        trace_dispatch: false,
        ..Default::default()
    };
    let load_start = std::time::Instant::now();
    let mut engine = dismantle_core::model::load_engine(&weights, cfg)?;
    eprintln!(
        "[batch-hash] engine loaded in {:.1}s",
        load_start.elapsed().as_secs_f64()
    );

    let mut output_lines: Vec<String> = Vec::with_capacity(prompts.len());
    let total = prompts.len();
    for (i, (id, prompt)) in prompts.iter().enumerate() {
        let prompt_for_req = prompt.clone();
        let req = GenerateRequest {
            prompt: prompt_for_req,
            max_new_tokens: tokens,
            sampling: SamplingParams {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                repetition_penalty: 1.0,
                seed: Some(42),
            },
            stop: Vec::new(),
            abort: None,
            max_stall_ms,
        };

        let mut decoded = String::new();
        let mut sink = |ev: StreamEvent| {
            if let StreamEvent::Token { text, .. } = ev {
                decoded.push_str(&text);
            }
        };
        let t0 = std::time::Instant::now();
        engine
            .generate(req, &mut sink)
            .map_err(|e| anyhow::anyhow!("{id}: {e}"))?;
        let dt = t0.elapsed().as_secs_f64();

        // Hash via b3sum (already on PATH via Homebrew).
        let mut child = Command::new("b3sum")
            .arg("--no-names")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| anyhow::anyhow!("spawn b3sum: {e}"))?;
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(decoded.as_bytes())?;
        let out = child.wait_with_output()?;
        let hash = String::from_utf8(out.stdout)?
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .to_string();

        // Escape \n in prompt -- same convention as expand-baseline.sh.
        let prompt_escaped = prompt.replace('\n', "\\n");
        output_lines.push(format!("{id} {tokens} {hash} {prompt_escaped}"));
        eprintln!(
            "[batch-hash] {}/{} {} hash={} ({:.1}s, {}B)",
            i + 1,
            total,
            id,
            &hash[..hash.len().min(16)],
            dt,
            decoded.len()
        );
    }

    // Header + lines, matching expand-baseline.sh's output format.
    let model_name = weights.file_name().and_then(|n| n.to_str()).unwrap_or("?");
    let header = format!(
        "# Phase 1 token-output baseline -- captured by `dismantle batch-hash`\n\
         # Format: <prompt-id> <max-new-tokens> <hash-hex> <prompt-text>\n\
         # algo: blake3\n\
         # Generation: temp=0 greedy, max_new_tokens={}, model={}\n",
        tokens, model_name
    );
    let body = output_lines.join("\n") + "\n";
    let blob = format!("{header}{body}");
    match out_path {
        Some(p) => std::fs::write(&p, &blob)?,
        None => print!("{blob}"),
    }
    Ok(())
}

fn bake_sidecar_main(
    weights: PathBuf,
    out: Option<PathBuf>,
    profile: String,
    kernel_profile: Option<PathBuf>,
    vocab_prune: Option<PathBuf>,
    quality_eval_count: usize,
    tier_map_json: Option<PathBuf>,
) -> Result<()> {
    use dismantle_core::sidecar::{sidecar_path_for, SidecarProfile};
    use dismantle_core::{profile::KernelProfile, EngineConfig};

    // Resolve the output path: default to same dir as weights, .dismantle ext.
    let out_path = out.unwrap_or_else(|| sidecar_path_for(&weights));

    // Parse and validate the profile name.
    let sidecar_profile = match profile.as_str() {
        "fast" => SidecarProfile::Fast,
        "race" => SidecarProfile::Race,
        "efficient" => SidecarProfile::Efficient,
        "exact" => SidecarProfile::Exact,
        other => anyhow::bail!("unknown --profile {other:?} (known: fast, race, efficient, exact)"),
    };

    // --- Step 1: print the planned bake steps ---
    eprintln!("[bake-sidecar] weights:        {}", weights.display());
    eprintln!("[bake-sidecar] out:            {}", out_path.display());
    eprintln!("[bake-sidecar] profile:        {profile}");
    eprintln!("[bake-sidecar] planned steps:");
    eprintln!("[bake-sidecar]   q4k_predec_scales = true");
    if let Some(vp) = vocab_prune.as_ref() {
        eprintln!(
            "[bake-sidecar]   pruned_lm_head_q4k = true  (vocab-prune: {})",
            vp.display()
        );
    } else {
        eprintln!("[bake-sidecar]   pruned_lm_head_q4k = false  (no --vocab-prune given)");
    }
    if let Some(kp) = kernel_profile.as_ref() {
        eprintln!("[bake-sidecar]   kernel_profile = {}", kp.display());
    }
    if quality_eval_count > 0 {
        eprintln!("[bake-sidecar]   quality_eval: {quality_eval_count} prompts (top-1 agreement)");
    } else {
        eprintln!("[bake-sidecar]   quality_eval: skipped (--quality-eval-count 0)");
    }
    eprintln!("[bake-sidecar] bake_profile:  {sidecar_profile:?}");

    // --- Step 2: load the engine (same as `generate` does) ---
    let kprofile = match kernel_profile.as_ref() {
        Some(path) => Some(KernelProfile::load(path)?),
        None => None,
    };
    let cfg = EngineConfig {
        kernel_profile: kprofile,
        ..Default::default()
    };
    eprintln!(
        "[bake-sidecar] loading engine from {} ...",
        weights.display()
    );
    let _engine = dismantle_core::model::load_engine(&weights, cfg)?;
    eprintln!("[bake-sidecar] engine loaded");

    // --- Step 3: bake predec scales ---
    eprintln!("[bake-sidecar] baking predec scales...");
    let bytes = match _engine.bake_sidecar_predec(&out_path, sidecar_profile) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[bake-sidecar] WARNING: bake_sidecar_predec failed: {e}");
            eprintln!("[bake-sidecar] This engine may not support sidecar baking yet.");
            return Ok(());
        }
    };

    // --- Step 4: summary ---
    // Track 4.3: fold an optional mixed-quant tier map into the baked sidecar.
    if let Some(tm_path) = tier_map_json.as_ref() {
        eprintln!(
            "[bake-sidecar] attaching tier map from {} ...",
            tm_path.display()
        );
        let tm = dismantle_core::sidecar::load_sidecar_tier_map_json(tm_path)?;
        let n_entries = tm.entries.len();
        let new_bytes = dismantle_core::sidecar::attach_tier_map_to_sidecar(&out_path, tm)?;
        eprintln!(
            "[bake-sidecar]   tier_map: {n_entries} entries embedded ({new_bytes} bytes total)"
        );
    }

    eprintln!("[bake-sidecar] summary:");
    eprintln!(
        "[bake-sidecar]   wrote {bytes} bytes → {}",
        out_path.display()
    );
    eprintln!("[bake-sidecar]   q4k_predec_scales: baked");
    if vocab_prune.is_some() {
        eprintln!("[bake-sidecar]   pruned_lm_head_q4k: not yet implemented (follow-on)");
    }
    eprintln!(
        "[bake-sidecar] done. Next load from {} will detect this sidecar and skip \
               the ~200ms predec decode pass.",
        weights.display()
    );
    Ok(())
}

fn version_main(weights: Option<PathBuf>) -> Result<()> {
    println!("dismantle {}", env!("CARGO_PKG_VERSION"));
    if let Some(p) = weights {
        match dismantle_core::gguf::GgufFile::open(&p) {
            Ok(g) => {
                println!(
                    "model: {} (arch={})",
                    g.name().unwrap_or("?"),
                    g.architecture().unwrap_or("?")
                );
            }
            Err(e) => eprintln!("could not read weights: {e}"),
        }
    }
    Ok(())
}

fn verify_main(weights: PathBuf, expected_sha256: Option<String>) -> Result<()> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    // Open and read the file in chunks to avoid a single large allocation.
    let mut f = std::fs::File::open(&weights)
        .map_err(|e| anyhow::anyhow!("open {}: {e}", weights.display()))?;
    let file_size = f.metadata()?.len();

    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 65536];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let hash_bytes = hasher.finalize();
    // Format as 64 lower-case hex chars.
    let hash_hex: String = hash_bytes.iter().map(|b| format!("{b:02x}")).collect();

    println!("file: {}", weights.display());
    println!("size: {file_size} bytes");
    println!("SHA-256: {hash_hex}");

    if let Some(expected) = expected_sha256.as_deref() {
        let expected_lower = expected.to_lowercase();
        if hash_hex == expected_lower {
            println!("hash check: PASS");
        } else {
            println!("hash check: FAIL");
            println!("  expected: {expected_lower}");
            println!("  actual:   {hash_hex}");
        }
    }

    // Check for sidecar.
    let sidecar = dismantle_core::sidecar::sidecar_path_for(&weights);
    if sidecar.exists() {
        println!("sidecar: {} (present)", sidecar.display());
    } else {
        println!("sidecar: not found");
    }

    Ok(())
}

#[cfg(test)]
mod profile_rank_tests {
    use super::{rank_profile_json, rank_profile_report};
    use dismantle_core::profile::{
        AutotuneEvidence, AutotuneMeasurement, KernelProfile, KernelVariant, RuntimeLevers,
        DEFAULT_QUALITY_FLOOR, PROFILE_SCHEMA_VERSION,
    };
    use std::collections::BTreeMap;

    fn mk(variant: &str, rank: u32, tps: f64, quality: f64) -> AutotuneMeasurement {
        AutotuneMeasurement::measured(variant, rank, tps, quality, RuntimeLevers::default())
    }

    fn profile_with(measurements: Vec<AutotuneMeasurement>) -> KernelProfile {
        KernelProfile {
            schema_version: PROFILE_SCHEMA_VERSION,
            profile_id: "kp-test".into(),
            profile_name: "test".into(),
            model_id: "Qwen2.5 3B Instruct".into(),
            model_arch: "qwen2".into(),
            tensor_layout_hash: "deadbeef".into(),
            device_name: "Apple M3 Pro".into(),
            shader_hash: "cafef00d".into(),
            selected: KernelVariant {
                id: "metal-default".into(),
                moe_schedule: "indexed-no-pack-one-cb".into(),
                mla_schedule: "metal-mla".into(),
                lm_head_schedule: "metal-argmax-token-only".into(),
                command_buffering: "one-cb-per-block".into(),
                gpu_buffer_reuse: "decode-arena".into(),
                deterministic_rank: 1,
                gemm_q4_k_schedule: "v2".into(),
                gemm_q4_k_schedule_per_shape: BTreeMap::new(),
                attn_block_schedule: "mla".into(),
                x_norm_dtype: "f32".into(),
                routed_down_schedule: "basic".into(),
                shared_down_schedule: "basic".into(),
                rmsnorm_attn_schedule: "basic".into(),
            },
            device_limits: None,
            runtime_levers: RuntimeLevers::default(),
            evidence: AutotuneEvidence {
                profile: "test".into(),
                max_hours: 1.0,
                prompt_count: 1,
                token_lengths: vec![64],
                candidate_count: measurements.len(),
                measurements,
                target_tps: 60.0,
                notes: vec![],
            },
        }
    }

    #[test]
    fn report_selects_highest_tps_above_floor_and_orders_table() {
        // `fast` has the highest tps (55) but FAILS the floor (q=0.80 < 0.90);
        // `mid` (40 tps, q=0.95) must be the selected line. Table must be in
        // descending score order: above-floor rows by tps desc, then the
        // rejected `fast` last tagged REJECT.
        let p = profile_with(vec![
            mk("fast", 3, 55.0, 0.80),
            mk("mid", 2, 40.0, 0.95),
            mk("slow", 1, 30.0, 1.00),
        ]);
        let report = rank_profile_report(&p, DEFAULT_QUALITY_FLOOR);

        // Chosen line names `mid`, NOT `fast`.
        assert!(
            report.contains("selected: mid"),
            "expected `mid` selected, got:\n{report}"
        );
        assert!(!report.contains("selected: fast"));

        // Table order: rank 1 = mid (40 tps), rank 2 = slow (30 tps),
        // rank 3 = fast (REJECT). Check by relative byte position.
        let line_idx = |needle: &str| report.find(needle).expect(needle);
        assert!(line_idx("1\t40.000\tmid") < line_idx("2\t30.000\tslow"));
        assert!(line_idx("2\t30.000\tslow") < line_idx("3\tREJECT\tfast"));
        // The failing candidate is rendered as REJECT, not a numeric score.
        assert!(report.contains("3\tREJECT\tfast"));
        // Selected block echoes the chosen tps/quality.
        assert!(report.contains("tps: 40.000"));
        assert!(report.contains("quality: 0.9500"));
    }

    #[test]
    fn report_handles_none_above_floor() {
        let p = profile_with(vec![mk("a", 1, 99.0, 0.10), mk("b", 2, 80.0, 0.50)]);
        let report = rank_profile_report(&p, DEFAULT_QUALITY_FLOOR);
        assert!(report.contains("selected: NONE"), "got:\n{report}");
        // Both rows still listed, both REJECT (sub-floor), highest-tps first.
        assert!(report.contains("REJECT\ta"));
        assert!(report.contains("REJECT\tb"));
    }

    #[test]
    fn custom_floor_changes_selection() {
        // With a 0.96 floor, `mid` (0.95) now fails and `slow` (1.00) wins.
        let p = profile_with(vec![mk("mid", 2, 40.0, 0.95), mk("slow", 1, 30.0, 1.00)]);
        let report = rank_profile_report(&p, 0.96);
        assert!(report.contains("selected: slow"), "got:\n{report}");
        assert!(report.contains("quality_floor: 0.9600"));
    }

    #[test]
    fn json_report_is_valid_and_marks_rejects() {
        let p = profile_with(vec![mk("fast", 3, 55.0, 0.80), mk("mid", 2, 40.0, 0.95)]);
        let out = rank_profile_json(&p, DEFAULT_QUALITY_FLOOR).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["selected"]["variant_id"], "mid");
        let ranked = v["ranked"].as_array().unwrap();
        // First ranked entry is the winner `mid` with a numeric score.
        assert_eq!(ranked[0]["variant_id"], "mid");
        assert_eq!(ranked[0]["rejected"], false);
        // The sub-floor `fast` is present and flagged rejected with null score.
        let fast = ranked.iter().find(|e| e["variant_id"] == "fast").unwrap();
        assert_eq!(fast["rejected"], true);
        assert!(fast["score"].is_null());
    }
}

fn spec_oracle_parse_k_list(s: &str) -> Result<Vec<usize>> {
    let mut ks = Vec::new();
    for part in s.split(',') {
        let t = part.trim();
        if t.is_empty() {
            continue;
        }
        let v: usize = t
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid --k entry {t:?} (want a comma list like 4,7)"))?;
        ks.push(v);
    }
    if ks.is_empty() {
        return Err(anyhow::anyhow!("--k produced no values (got {s:?})"));
    }
    Ok(ks)
}

/// CPU-only spec replay-oracle handler. Mirrors the `generate` tokenizer path
/// (sibling tokenizer.json preferred, else GGUF-embedded vocab) WITHOUT
/// constructing the Metal engine, encodes the corpus, and replays through the
/// shipped `replay_grid`.
fn spec_oracle_main(
    corpus: PathBuf,
    tokenizer_from: PathBuf,
    k: String,
    warm_frac: f64,
    json: bool,
) -> Result<()> {
    use dismantle_core::gguf::GgufFile;
    use dismantle_core::speculate::replay_oracle::replay_grid;
    use dismantle_core::tokenizer::Tokenizer;

    let text = std::fs::read_to_string(&corpus)
        .map_err(|e| anyhow::anyhow!("read corpus {}: {e}", corpus.display()))?;

    // Same tokenizer resolution as the live generate path (qwen_dense.rs):
    // prefer a sibling tokenizer.json, else the GGUF-embedded vocab. Both CPU.
    let sidecar_json = tokenizer_from.parent().map(|d| d.join("tokenizer.json"));
    let tokenizer = match sidecar_json {
        Some(p) if p.exists() => {
            Tokenizer::from_file(&p).map_err(|e| anyhow::anyhow!("tokenizer.json load: {e}"))?
        }
        _ => {
            let gguf = GgufFile::open(&tokenizer_from)
                .map_err(|e| anyhow::anyhow!("open {}: {e}", tokenizer_from.display()))?;
            Tokenizer::from_gguf(&gguf).map_err(|e| anyhow::anyhow!("tokenizer from gguf: {e}"))?
        }
    };

    let ids: Vec<u32> = tokenizer
        .encode(&text, true)
        .map_err(|e| anyhow::anyhow!("encode corpus: {e}"))?;
    let ks = spec_oracle_parse_k_list(&k)?;
    let warm = ((ids.len() as f64) * warm_frac.clamp(0.0, 1.0)).floor() as usize;
    let report = replay_grid(&ids, &ks, warm);

    if json {
        let mut s = String::new();
        s.push_str("{\n");
        s.push_str(&format!("  \"verdict\": \"{}\",\n", report.verdict()));
        s.push_str(&format!("  \"corpus_tokens\": {},\n", ids.len()));
        s.push_str(&format!("  \"scored_tokens\": {},\n", report.scored_tokens));
        s.push_str(&format!(
            "  \"warm_start_tokens\": {},\n",
            report.warm_start_tokens
        ));
        s.push_str(&format!(
            "  \"best_k\": {},\n",
            report.best().map(|b| b.k as i64).unwrap_or(-1)
        ));
        s.push_str("  \"per_k\": [\n");
        for (i, r) in report.per_k.iter().enumerate() {
            let comma = if i + 1 < report.per_k.len() { "," } else { "" };
            s.push_str(&format!(
                "    {{\"k\": {}, \"forward_cycles\": {}, \"tokens_emitted\": {}, \
                 \"tau\": {:.6}, \"mean_accepted_len\": {:.6}, \"hit_rate\": {:.6}, \
                 \"proposal_coverage\": {:.6}, \"draft_accept_frac\": {:.6}, \
                 \"governor_propose_frac\": {:.6}, \"accept_hist\": {:?}}}{}\n",
                r.k,
                r.forward_cycles,
                r.tokens_emitted,
                r.tau,
                r.mean_accepted_len,
                r.hit_rate,
                r.proposal_coverage,
                r.draft_accept_frac,
                r.governor_propose_frac,
                r.accept_hist,
                comma
            ));
        }
        s.push_str("  ]\n}");
        println!("{s}");
    } else {
        println!(
            "spec-oracle: verdict={} corpus_tokens={} scored={} warm_start={}",
            report.verdict(),
            ids.len(),
            report.scored_tokens,
            report.warm_start_tokens
        );
        println!(
            "  {:>3}  {:>7}  {:>7}  {:>6}  {:>6}  {:>6}  accept_hist",
            "k", "tau", "mal", "hit", "cov", "gov"
        );
        for r in &report.per_k {
            println!(
                "  {:>3}  {:>7.3}  {:>7.3}  {:>6.3}  {:>6.3}  {:>6.3}  {:?}",
                r.k,
                r.tau,
                r.mean_accepted_len,
                r.hit_rate,
                r.proposal_coverage,
                r.governor_propose_frac,
                r.accept_hist
            );
        }
        if let Some(b) = report.best() {
            println!(
                "  best k={} tau={:.3} ({}) — bands GO>=2.5 MARGINAL>=1.6",
                b.k,
                b.tau,
                report.verdict()
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod spec_oracle_tests {
    use super::*;

    #[test]
    fn parse_k_list_handles_comma_and_whitespace() {
        assert_eq!(spec_oracle_parse_k_list("4,7").unwrap(), vec![4, 7]);
        assert_eq!(
            spec_oracle_parse_k_list(" 4 , 7 ,8").unwrap(),
            vec![4, 7, 8]
        );
        assert!(spec_oracle_parse_k_list("").is_err());
        assert!(spec_oracle_parse_k_list("4,x").is_err());
    }

    #[test]
    fn synthetic_ids_flow_through_replay_grid() {
        // The text->ids->report GLUE: feed a synthetic, highly-repetitive id
        // stream (what encode() would yield on a repetitive corpus) straight
        // into the shipped replay_grid and assert the report the handler prints.
        use dismantle_core::speculate::replay_oracle::replay_grid;
        let mut ids: Vec<u32> = Vec::new();
        for _ in 0..200 {
            ids.extend_from_slice(&[10, 11, 12, 13, 14, 15, 16, 17]);
        }
        let ks = spec_oracle_parse_k_list("4,7").unwrap();
        let warm = ((ids.len() as f64) * 0.5_f64).floor() as usize;
        let report = replay_grid(&ids, &ks, warm);
        assert_eq!(report.per_k.len(), 2);
        assert_eq!(report.warm_start_tokens, warm);
        assert_eq!(report.scored_tokens, ids.len() - warm);
        let best = report.best().expect("non-empty grid");
        assert!(
            best.tau > 1.05,
            "repetitive corpus must beat plain decode (tau {})",
            best.tau
        );
        assert_eq!(
            report.verdict(),
            "GO",
            "repetitive stream should clear the GO band"
        );
        // accounting closes for every row (the property the handler relies on).
        for r in &report.per_k {
            assert_eq!(r.tokens_emitted as usize, report.scored_tokens);
        }
    }
}

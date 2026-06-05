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
fn apply_profile(profile: &Option<String>) {
    let Some(name) = profile.as_deref() else {
        return;
    };
    match name {
        "fast" | "race" | "efficient" => {
            for (k, v) in [
                ("DISMANTLE_QWEN_VOCAB_PRUNE",       "32000"),
                ("DISMANTLE_QWEN_Q4K_LMHEAD",        "1"),
                ("DISMANTLE_QWEN_FFN_DOWN_Q4K",       "1"),
                ("DISMANTLE_QWEN_Q4K_PREDEC",         "1"),
                ("DISMANTLE_QWEN_PREDEC_F16SCALES",   "1"),
            ] {
                if std::env::var_os(k).is_none() {
                    std::env::set_var(k, v);
                }
            }
            if name == "efficient" && std::env::var_os("DISMANTLE_ENERGY_EFFICIENT").is_none() {
                std::env::set_var("DISMANTLE_ENERGY_EFFICIENT", "1");
            }
            let extra = if name == "efficient" { " + energy-efficient mode" } else { "" };
            eprintln!(
                "[dismantle] --profile {name}: vocab-prune-32k + Q4K LM-head + Q4K \
                 FFN-down + predec + f16-scales{extra} (mild quality trade; omit \
                 --profile for the bit-identical default)"
            );
        }
        "exact" => {
            eprintln!(
                "[dismantle] --profile exact: bit-identical conservative path \
                 (no quality trade-offs; all fast-path env vars left at their \
                 current values — set DISMANTLE_QWEN_PREDEC_F16SCALES=0 etc. \
                 explicitly to opt out of individual levers)"
            );
        }
        "default" => {
            // Explicit no-op: same as not passing --profile at all.
        }
        other => eprintln!(
            "[dismantle] warning: unknown --profile '{other}' \
             (known: default, fast, race, efficient, exact); ignoring"
        ),
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
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    apply_profile(&cli.profile);
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
        } => {
            // --hardware-profile is the preferred alias for --kernel-profile.
            let resolved_kernel_profile = hardware_profile.or(kernel_profile);

            // Parse --profile (global flag) into RuntimeProfile.
            let runtime_profile = cli.profile.as_deref()
                .and_then(dismantle_serve::RuntimeProfile::from_str)
                .unwrap_or(dismantle_serve::RuntimeProfile::Default);

            // Parse --energy-mode.
            let resolved_energy_mode = energy_mode.as_deref()
                .and_then(dismantle_serve::EnergyMode::from_str)
                .unwrap_or(dismantle_serve::EnergyMode::Off);

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
            }))
        }
        Cmd::Generate {
            weights,
            prompt,
            max_new_tokens,
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
        } => generate_main(
            weights,
            prompt,
            max_new_tokens,
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
        } => autotune_main(weights, profile, max_hours, out, log),
        Cmd::BenchQ4kShapes { iters, out } => bench_q4k_shapes_main(iters, out),
        Cmd::Doctor {
            weights,
            max_seq_len,
        } => doctor_main(weights, max_seq_len),
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
    }
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
    } else if heads > 0 {
        hidden / heads
    } else {
        0
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
        println!(
            "expert_tracking: disabled (pass --max-routed-expert-ram-mb to enable)"
        );
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
    (0..cols)
        .map(|i| ((i % 97) as f32 - 48.0) / 97.0)
        .collect()
}

fn autotune_main(
    weights: PathBuf,
    profile: String,
    max_hours: f64,
    out: PathBuf,
    log: Option<PathBuf>,
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
    let mut log_lines = Vec::with_capacity(selected.evidence.measurements.len() + 2);
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
    std::fs::write(&out, serde_json::to_string_pretty(&selected)?)?;
    std::fs::write(&log_path, log_lines.join("\n") + "\n")?;
    println!("wrote kernel profile: {}", out.display());
    println!("wrote autotune log: {}", log_path.display());
    println!("profile_id: {}", selected.profile_id);
    println!("selected_variant: {}", selected.selected.id);
    println!("device: {}", selected.device_name);
    println!("target_tps: {:.1}", selected.evidence.target_tps);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn generate_main(
    weights: PathBuf,
    prompt: String,
    max_new_tokens: usize,
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
        max_seq_len: 4096,
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
                return Err(anyhow::anyhow!("prompts file {} has no prompts", path.display()));
            }
            eprintln!("[capture] {} prompts from {}", v.len(), path.display());
            v
        }
        None => {
            if prompt.is_empty() {
                return Err(anyhow::anyhow!(
                    "provide --prompt or --prompts-file"
                ));
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
                    "\n[stats] reason={} prompt={} completion={} prefill_ms={:.1} decode_ms={:.1} dec_tps={:.2} draft_accepted={} draft_rejected={} profile={}",
                    reason_s,
                    stats.prompt_tokens,
                    stats.completion_tokens,
                    stats.prefill_ms,
                    stats.decode_ms,
                    dec,
                    stats.draft_accepted,
                    stats.draft_rejected,
                    stats.profile_id.as_deref().unwrap_or("none")
                );
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
    let model_name = weights
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("?");
    let header = format!(
        "# Phase 1 token-output baseline -- captured by `dismantle batch-hash`\n\
         # Format: <prompt-id> <max-new-tokens> <hash-hex> <prompt-text>\n\
         # algo: blake3\n\
         # Generation: temp=0 greedy, max_new_tokens={}, model={}\n",
        tokens,
        model_name
    );
    let body = output_lines.join("\n") + "\n";
    let blob = format!("{header}{body}");
    match out_path {
        Some(p) => std::fs::write(&p, &blob)?,
        None => print!("{blob}"),
    }
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

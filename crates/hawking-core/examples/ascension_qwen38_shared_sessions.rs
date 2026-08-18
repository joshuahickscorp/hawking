//! One resident Qwen3.8 weight set, N decode sessions.
//!
//! Process-pool children do not share artifact pages (measured 2026-08-16:
//! 8.77 GB RSS / child vs 8.5 GB on disk). This binary is the redesigned
//! primary: load weights once, attach independent KV/state sessions, admit
//! or refuse before swap.
//!
//! ```text
//! ./tools/gpu_lane_lock.sh qwen38-shared-sessions \
//!   target/release/examples/ascension_qwen38_shared_sessions \
//!   --artifact-root .../uniform-q4-v1 \
//!   --tokenizer .../bf16/tokenizer.json \
//!   --mode probe --sessions 4 --max-seq-len 2048 --max-new-tokens 8 \
//!   --out receipts/ascent-2026-08-16/QWEN38_SHARED_SESSIONS.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;
use std::time::Instant;

fn usage() -> &'static str {
    "usage: ascension_qwen38_shared_sessions --mode probe|gravity-recipe|kernel-floor|refuse \
        [--artifact-root DIR] [--tokenizer PATH] [--sessions N] [--max-seq-len N] \
        [--max-new-tokens N] [--child-id ID] [--lock-held] [--reserve-bytes N] \
        [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("ascension_qwen38_shared_sessions: {message}");
    process::exit(2);
}

struct Args {
    mode: String,
    artifact_root: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    sessions: usize,
    max_seq_len: usize,
    max_new_tokens: usize,
    child_id: String,
    lock_held: bool,
    reserve_bytes: u64,
    out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut mode = "probe".to_owned();
    let mut artifact_root = None;
    let mut tokenizer = None;
    let mut sessions = 2usize;
    let mut max_seq_len = 2048usize;
    let mut max_new_tokens = 8usize;
    let mut child_id = "child-0".to_owned();
    let mut lock_held = false;
    let mut reserve_bytes = hawking_core::model::qwen38_host_admission::DEFAULT_RESERVE_BYTES;
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--mode" => mode = args.next().unwrap_or_else(|| fail(usage())),
            "--artifact-root" => {
                artifact_root = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--tokenizer" => {
                tokenizer = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--sessions" => {
                sessions = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--sessions"));
            }
            "--max-seq-len" => {
                max_seq_len = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--max-seq-len"));
            }
            "--max-new-tokens" => {
                max_new_tokens = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--max-new-tokens"));
            }
            "--child-id" => child_id = args.next().unwrap_or_else(|| fail(usage())),
            "--lock-held" => lock_held = true,
            "--reserve-bytes" => {
                reserve_bytes = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--reserve-bytes"));
            }
            "--out" => out = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage())))),
            other => fail(format!("unknown {other}; {}", usage())),
        }
    }
    Args {
        mode,
        artifact_root,
        tokenizer,
        sessions,
        max_seq_len,
        max_new_tokens,
        child_id,
        lock_held,
        reserve_bytes,
        out,
    }
}

fn write_json(path: &Path, body: &Value) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(path, serde_json::to_vec_pretty(body).expect("json")).unwrap_or_else(|e| fail(e));
    eprintln!("wrote {}", path.display());
}

fn emit(out: &Option<PathBuf>, body: &Value) {
    println!("{}", serde_json::to_string_pretty(body).expect("json"));
    if let Some(path) = out {
        write_json(path, body);
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    fail("qwen38 shared sessions are Metal-only");
}

#[cfg(target_os = "macos")]
fn main() {
    let args = parse_args();
    match args.mode.as_str() {
        "refuse" => run_refuse(&args),
        "probe" => run_probe(&args),
        "gravity-recipe" => run_gravity_recipe(&args),
        "kernel-floor" => run_kernel_floor(&args),
        other => fail(format!("unknown --mode {other}")),
    }
}

#[cfg(target_os = "macos")]
fn run_refuse(args: &Args) {
    use hawking_core::model::qwen38_host_admission::{
        decide_admission, host_memory_snapshot, process_pool_child_cost_bytes, AdmissionRequest,
        AdmissionVerdict,
    };
    use hawking_core::model::qwen38_hybrid_decode::qwen38_workspace_bytes;

    let memory = host_memory_snapshot().unwrap_or_else(|e| fail(e));
    let workspace = qwen38_workspace_bytes(args.max_seq_len)
        .map(|b| b.total_bytes as u64)
        .unwrap_or(u64::MAX);
    // Over-subscribe: pretend we need every remaining byte plus one process child.
    let cost = memory
        .free_bytes
        .saturating_sub(args.reserve_bytes.saturating_sub(1))
        .saturating_add(process_pool_child_cost_bytes(args.max_seq_len));
    let decision = decide_admission(
        &memory,
        AdmissionRequest {
            label: args.child_id.clone(),
            cost_bytes: cost.max(memory.free_bytes.saturating_add(1)),
            kind: "oversub_demo".into(),
        },
        args.reserve_bytes,
    );
    if decision.verdict != AdmissionVerdict::Refuse {
        fail("admission gate failed to refuse an over-subscription");
    }
    let body = json!({
        "schema": "hawking.qwen38.shared_sessions.v1",
        "mode": "refuse",
        "child_id": args.child_id,
        "lock_held": args.lock_held,
        "workspace_bytes": workspace,
        "host": memory,
        "admission": decision,
        "demonstrated": "REFUSE",
    });
    emit(&args.out, &body);
}

#[cfg(target_os = "macos")]
fn require_paths(args: &Args) -> (PathBuf, PathBuf) {
    let artifact = args
        .artifact_root
        .clone()
        .unwrap_or_else(|| fail("--artifact-root required"));
    let tokenizer = args
        .tokenizer
        .clone()
        .unwrap_or_else(|| fail("--tokenizer required"));
    (artifact, tokenizer)
}

#[cfg(target_os = "macos")]
fn run_probe(args: &Args) {
    use hawking_core::model::qwen38_host_admission::{
        decide_admission, first_shared_process_cost_bytes, host_memory_snapshot, process_rss_bytes,
        shared_session_attach_cost_bytes, AdmissionRequest, AdmissionVerdict,
    };
    use hawking_core::model::qwen38_hybrid_decode::{
        generate_greedy, generate_greedy_parallel, load_qwen38_tokenizer, measure_shared_weight_fanout,
        qwen38_workspace_bytes, render_qwen38_user_chat, Qwen38HybridDecodeSession,
        Qwen38HybridWeights,
    };

    if args.sessions == 0 {
        fail("--sessions must be >= 1");
    }
    let (artifact, tokenizer_path) = require_paths(args);
    let memory_before = host_memory_snapshot().unwrap_or_else(|e| fail(e));
    let first = decide_admission(
        &memory_before,
        AdmissionRequest {
            label: format!("{}-load", args.child_id),
            cost_bytes: first_shared_process_cost_bytes(args.max_seq_len),
            kind: "shared_process_load".into(),
        },
        args.reserve_bytes,
    );
    if first.verdict == AdmissionVerdict::Refuse {
        emit(
            &args.out,
            &json!({
                "schema": "hawking.qwen38.shared_sessions.v1",
                "mode": "probe",
                "child_id": args.child_id,
                "lock_held": args.lock_held,
                "admission": first,
                "status": "REFUSED_LOAD",
            }),
        );
        process::exit(3);
    }

    let pid = process::id();
    let rss_before = process_rss_bytes(pid).ok();
    let load_started = Instant::now();
    let weights = Arc::new(Qwen38HybridWeights::load(&artifact).unwrap_or_else(|e| fail(e)));
    let load_ns = load_started.elapsed().as_nanos() as u64;
    let rss_after_load = process_rss_bytes(pid).ok();
    let workspace = qwen38_workspace_bytes(args.max_seq_len).unwrap_or_else(|e| fail(e));

    let mut sessions = Vec::new();
    let mut attach_decisions = Vec::new();
    let mut rss_after_attach = Vec::new();
    let mut refused = None;
    for index in 0..args.sessions {
        let memory = host_memory_snapshot().unwrap_or_else(|e| fail(e));
        let decision = decide_admission(
            &memory,
            AdmissionRequest {
                label: format!("{}-session-{index}", args.child_id),
                cost_bytes: shared_session_attach_cost_bytes(workspace.total_bytes as u64),
                kind: "shared_session".into(),
            },
            args.reserve_bytes,
        );
        if decision.verdict == AdmissionVerdict::Refuse {
            refused = Some(decision);
            break;
        }
        attach_decisions.push(decision);
        let session = Qwen38HybridDecodeSession::attach(Arc::clone(&weights), args.max_seq_len)
            .unwrap_or_else(|e| fail(e));
        if index > 0 && !session.shares_weights_with(&sessions[0]) {
            fail("attached session does not share the resident weight Arc");
        }
        sessions.push(session);
        rss_after_attach.push(process_rss_bytes(pid).ok());
    }

    let tokenizer = load_qwen38_tokenizer(&tokenizer_path).unwrap_or_else(|e| fail(e));
    let prompts: Vec<String> = (0..sessions.len())
        .map(|i| {
            render_qwen38_user_chat(&format!(
                "Session {i}: reply with the single word ready{i}."
            ))
        })
        .collect();
    let prompt_ids: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| tokenizer.encode(p, false).unwrap_or_else(|e| fail(e)))
        .collect();

    let mut baseline = None;
    if !sessions.is_empty() {
        let started = Instant::now();
        let one = generate_greedy(&mut sessions[0], &prompt_ids[0], args.max_new_tokens)
            .unwrap_or_else(|e| fail(e));
        baseline = Some(json!({
            "sessions": 1,
            "wall_ns": one.wall_ns,
            "decode_wall_ns": one.decode_wall_ns,
            "decode_steps": one.decode_steps,
            "steady_decode_wall_ns_per_token": one.steady_decode_wall_ns_per_token(),
            "median_gpu_ns_per_token": one.median_gpu_ns_per_token(),
            "tokens_per_s": one.steady_decode_wall_ns_per_token().map(|ns| {
                if ns == 0 { 0.0 } else { 1e9 / ns as f64 }
            }),
            "new_token_ids": one.new_tokens(),
            "elapsed_ns": started.elapsed().as_nanos() as u64,
        }));
    }

    let mut parallel = None;
    if sessions.len() > 1 {
        let started = Instant::now();
        let results = generate_greedy_parallel(&mut sessions, &prompt_ids, args.max_new_tokens)
            .unwrap_or_else(|e| fail(e));
        let elapsed = started.elapsed().as_nanos() as u64;
        let decode_steps: usize = results.iter().map(|r| r.decode_steps).sum();
        let tokens_per_s = if elapsed == 0 {
            0.0
        } else {
            decode_steps as f64 * 1e9 / elapsed as f64
        };
        parallel = Some(json!({
            "sessions": sessions.len(),
            "wall_ns": elapsed,
            "decode_steps_sum": decode_steps,
            "aggregate_tokens_per_s": tokens_per_s,
            "per_session": results.iter().map(|r| json!({
                "decode_steps": r.decode_steps,
                "decode_wall_ns": r.decode_wall_ns,
                "steady_decode_wall_ns_per_token": r.steady_decode_wall_ns_per_token(),
                "median_gpu_ns_per_token": r.median_gpu_ns_per_token(),
                "new_token_ids": r.new_tokens(),
            })).collect::<Vec<_>>(),
        }));
    }

    let refs: Vec<&Qwen38HybridDecodeSession> = sessions.iter().collect();
    let fanout_serial = if refs.is_empty() {
        None
    } else {
        Some(
            measure_shared_weight_fanout(&refs, "language_model.lm_head.weight", false)
                .unwrap_or_else(|e| fail(e)),
        )
    };
    let fanout_concurrent = if refs.len() > 1 {
        Some(
            measure_shared_weight_fanout(&refs, "language_model.lm_head.weight", true)
                .unwrap_or_else(|e| fail(e)),
        )
    } else {
        None
    };
    let fanout_one = if refs.is_empty() {
        None
    } else {
        Some(
            measure_shared_weight_fanout(&refs[..1], "language_model.lm_head.weight", false)
                .unwrap_or_else(|e| fail(e)),
        )
    };

    let rss_final = process_rss_bytes(pid).ok();
    let memory_after = host_memory_snapshot().ok();
    let marginals: Vec<Value> = rss_after_attach
        .windows(2)
        .map(|w| json!({"from": w[0], "to": w[1], "delta": w[1].and_then(|b| w[0].map(|a| b as i64 - a as i64))}))
        .collect();

    let body = json!({
        "schema": "hawking.qwen38.shared_sessions.v1",
        "mode": "probe",
        "status": if refused.is_some() { "PARTIAL_REFUSED_ATTACH" } else { "MEASURED" },
        "child_id": args.child_id,
        "pid": pid,
        "lock_held": args.lock_held,
        "artifact_root": artifact,
        "max_seq_len": args.max_seq_len,
        "requested_sessions": args.sessions,
        "attached_sessions": sessions.len(),
        "weights_ptr_shared": sessions.len() <= 1 || sessions.windows(2).all(|w| w[1].shares_weights_with(&w[0])),
        "resident_weight_bytes": weights.resident_bytes(),
        "weight_tensors": {
            "q4": weights.q4_tensor_count(),
            "f32": weights.f32_tensor_count(),
            "mixed": weights.mixed_tensor_count(),
        },
        "workspace": workspace,
        "workspace_resident_bytes": sessions.first().map(|s| s.workspace_resident_bytes()),
        "load_ns": load_ns,
        "rss_before_bytes": rss_before,
        "rss_after_load_bytes": rss_after_load,
        "rss_after_each_attach_bytes": rss_after_attach,
        "rss_final_bytes": rss_final,
        "marginal_rss_per_session_bytes": marginals,
        "host_before": memory_before,
        "host_after": memory_after,
        "load_admission": first,
        "attach_admissions": attach_decisions,
        "refused_attach": refused,
        "baseline_1_session": baseline,
        "parallel_n_session": parallel,
        "fanout_1": fanout_one,
        "fanout_n_serial": fanout_serial,
        "fanout_n_concurrent": fanout_concurrent,
        "parallelizes": [
            "text generate across N sessions sharing one weight Arc",
            "concurrent GEMV fanout against one resident weight"
        ],
        "lock_bound": [
            "kernel-floor GPU timestamps are only honest with --lock-held",
            "Metal command-queue work still contends for the same GPU"
        ],
    });
    emit(&args.out, &body);
}

#[cfg(target_os = "macos")]
fn run_gravity_recipe(args: &Args) {
    use hawking_core::model::qwen38_hybrid_decode::{
        generate_greedy, load_qwen38_tokenizer, render_qwen38_user_chat, Qwen38HybridDecodeSession,
        Qwen38HybridWeights,
    };
    let (artifact, tokenizer_path) = require_paths(args);
    let weights = Arc::new(Qwen38HybridWeights::load(&artifact).unwrap_or_else(|e| fail(e)));
    let mut session = Qwen38HybridDecodeSession::attach(weights, args.max_seq_len)
        .unwrap_or_else(|e| fail(e));
    let tokenizer = load_qwen38_tokenizer(&tokenizer_path).unwrap_or_else(|e| fail(e));
    let prompt = render_qwen38_user_chat(
        "Propose one gravity pack recipe for Qwen3.8-27B that attacks \
         weight_addressing (60.44% of the token, DRAM traffic), not unused experts. \
         Give: codec, bits, group, which tensors, and the measurement that would kill it. \
         Be concrete. No projections.",
    );
    let ids = tokenizer.encode(&prompt, false).unwrap_or_else(|e| fail(e));
    let result = generate_greedy(&mut session, &ids, args.max_new_tokens).unwrap_or_else(|e| fail(e));
    let text = result.decode_new(&tokenizer).unwrap_or_else(|e| fail(e));
    println!("GENERATED_TEXT_VERBATIM: {text}");
    let body = json!({
        "schema": "hawking.qwen38.shared_sessions.v1",
        "mode": "gravity-recipe",
        "child_id": args.child_id,
        "lock_held": args.lock_held,
        "timing_honest": args.lock_held,
        "workload": "text_search",
        "parallelizes": true,
        "lock_bound": false,
        "generated_text": text,
        "new_token_ids": result.new_tokens(),
        "decode_steps": result.decode_steps,
        "decode_wall_ns": result.decode_wall_ns,
        "median_gpu_ns_per_token": result.median_gpu_ns_per_token(),
    });
    emit(&args.out, &body);
}

#[cfg(target_os = "macos")]
fn run_kernel_floor(args: &Args) {
    use hawking_core::model::qwen38_hybrid_decode::{
        Qwen38HybridDecodeSession, Qwen38HybridWeights, Qwen38MatvecKernel,
    };
    if !args.lock_held {
        eprintln!("kernel-floor: GPU lock not held; timestamps labeled CONTENDED");
    }
    let (artifact, _) = require_paths(args);
    let weights = Arc::new(Qwen38HybridWeights::load(&artifact).unwrap_or_else(|e| fail(e)));
    let mut session = Qwen38HybridDecodeSession::attach(weights, args.max_seq_len.min(128))
        .unwrap_or_else(|e| fail(e));
    let mut rows = Vec::new();
    for &kernel in Qwen38MatvecKernel::all() {
        session.matvec_kernel = kernel;
        let mut gpu = Vec::new();
        let mut wait = Vec::new();
        for _ in 0..3 {
            let timing = session
                .measure_isolated_lm_head()
                .unwrap_or_else(|e| fail(e));
            gpu.push(timing.gpu_ns);
            wait.push(timing.wait_ns);
        }
        let mut present: Vec<u64> = gpu.iter().copied().flatten().collect();
        present.sort_unstable();
        rows.push(json!({
            "kernel": kernel.as_str(),
            "gpu_ns_reps": gpu,
            "wait_ns_reps": wait,
            "median_gpu_ns": present.get(present.len() / 2).copied(),
        }));
    }
    session.matvec_kernel = Qwen38MatvecKernel::GeoTpr64Tg128;
    let body = json!({
        "schema": "hawking.qwen38.shared_sessions.v1",
        "mode": "kernel-floor",
        "child_id": args.child_id,
        "lock_held": args.lock_held,
        "timing_label": if args.lock_held { "LOCK_HELD" } else { "CONTENDED" },
        "workload": "timing_search",
        "parallelizes": false,
        "lock_bound": true,
        "kernels": rows,
    });
    emit(&args.out, &body);
}

//! G126: does the SEALED DEFAULT actually dispatch the measured kernels?
//!
//! Every existing A/B pins its arms with `set_dn_state_kernel`, so none of
//! them can answer this. This one NEVER pins the state kernel: it opens a
//! session with the levers unset and reports what the dispatcher actually
//! launched. That is the whole point - a default that silently fails to
//! select its kernel would report the old number under a new label.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core \
//!   --example ascension_qwen38_sealed_default
//! ./tools/gpu_lane_lock.sh g126 \
//!   workspace/ops/build/rust/release-fast/examples/ascension_qwen38_sealed_default \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A \
//!   --tokenizer ~/noetic/NOETIC_PARENT_A/tokenizer.json \
//!   --reps 7 --max-new-tokens 32 \
//!   --out receipts/future/_G126_SEALED_DEFAULT_raw.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

#[cfg(target_os = "macos")]
use hawking_core::model::qwen38_hybrid_decode::{
    generate_greedy, load_qwen38_tokenizer, render_qwen38_user_chat, Qwen38DeltaNetStateKernel,
    Qwen38GenerateResult, Qwen38HybridDecodeSession, Qwen38MlpFusion, QWEN38_DN_STATE_F4_KERNEL,
};

fn usage() -> &'static str {
    "usage: ascension_qwen38_sealed_default --artifact-root DIR --tokenizer PATH \
        [--prompt TEXT] [--raw-prompt] [--max-new-tokens N] [--max-seq-len N] \
        [--reps N] [--warmup N] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("ascension_qwen38_sealed_default: {message}");
    process::exit(2);
}

struct Args {
    artifact_root: PathBuf,
    tokenizer: PathBuf,
    prompt: String,
    raw_prompt: bool,
    max_new_tokens: usize,
    max_seq_len: usize,
    reps: usize,
    warmup: usize,
    out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut artifact_root = None;
    let mut tokenizer = None;
    let mut prompt = concat!(
        "Explain, in ordinary prose and at length, how a compiler turns a ",
        "for-loop into basic blocks and then into machine code."
    )
    .to_owned();
    let mut raw_prompt = false;
    let mut max_new_tokens = 32usize;
    let mut max_seq_len = 128usize;
    let mut reps = 7usize;
    let mut warmup = 1usize;
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--artifact-root" => {
                artifact_root = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--tokenizer" => {
                tokenizer = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--prompt" => prompt = args.next().unwrap_or_else(|| fail(usage())),
            "--raw-prompt" => raw_prompt = true,
            "--max-new-tokens" => {
                max_new_tokens = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--max-new-tokens"));
            }
            "--max-seq-len" => {
                max_seq_len = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--max-seq-len"));
            }
            "--reps" => {
                reps = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--reps"));
            }
            "--warmup" => {
                warmup = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--warmup"));
            }
            "--out" => out = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage())))),
            other => fail(format!("unknown {other}; {}", usage())),
        }
    }
    if reps < 7 {
        fail("--reps must be >= 7 (S020 §37)");
    }
    Args {
        artifact_root: artifact_root.unwrap_or_else(|| fail(usage())),
        tokenizer: tokenizer.unwrap_or_else(|| fail(usage())),
        prompt,
        raw_prompt,
        max_new_tokens,
        max_seq_len,
        reps,
        warmup,
        out,
    }
}

fn git_head() -> String {
    std::process::Command::new("git")
        .args(["--no-optional-locks", "rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn cmd_stdout(args: &[&str]) -> String {
    std::process::Command::new(args[0])
        .args(&args[1..])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default()
}

fn concurrent_load() -> Value {
    let loadavg = cmd_stdout(&["sysctl", "-n", "vm.loadavg"])
        .trim()
        .to_string();
    let uptime = cmd_stdout(&["uptime"]).trim().to_string();
    json!({
        "loadavg": loadavg,
        "uptime": uptime,
        "note": "absolute ms are measured-under-load; the A/B ratio is back-to-back in this process",
    })
}

fn median_u64(mut v: Vec<u64>) -> Option<u64> {
    if v.is_empty() {
        return None;
    }
    v.sort_unstable();
    Some(v[v.len() / 2])
}

fn timing_json(name: &str, gpu: &[u64], wait: &[u64], dispatches: u64, reps: usize) -> Value {
    json!({
        "name": name,
        "gpu_ns_reps": gpu,
        "gpu_ns_min": gpu.iter().copied().min(),
        "gpu_ns_median": median_u64(gpu.to_vec()),
        "gpu_ns_max": gpu.iter().copied().max(),
        "wait_ns_median": median_u64(wait.to_vec()),
        "dispatches": dispatches,
        "n_reps": reps,
        "gpu_timestamp_authority":
            "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy",
        "dense_w_materialized": 0,
    })
}

#[cfg(target_os = "macos")]
fn reps_cb(
    label: &str,
    reps: usize,
    mut once: impl FnMut() -> Result<hawking_core::metal::CommandBufferTiming, String>,
) -> Result<Value, String> {
    let mut gpu = Vec::new();
    let mut wait = Vec::new();
    let mut disp = 0u64;
    eprintln!("  {label} reps={reps}");
    for i in 0..reps {
        let t = once()?;
        let g = t
            .gpu_ns
            .ok_or_else(|| format!("{label}: driver did not expose GPUEndTime-GPUStartTime"))?;
        eprintln!("    rep{i} gpu={g} wait={} disp={}", t.wait_ns, t.dispatches);
        gpu.push(g);
        wait.push(t.wait_ns);
        disp = t.dispatches;
    }
    Ok(timing_json(label, &gpu, &wait, disp, reps))
}

/// Complete-token GPU samples: last prefill (emits first new token) plus
/// every decode step. Each is one full sealed graph.
#[cfg(target_os = "macos")]
fn complete_token_gpu_ns(result: &Qwen38GenerateResult) -> Vec<u64> {
    let start = result.prompt_len.saturating_sub(1);
    result
        .gpu_ns
        .get(start.min(result.gpu_ns.len())..)
        .unwrap_or(&[])
        .iter()
        .copied()
        .flatten()
        .collect()
}

#[cfg(target_os = "macos")]
fn complete_token_dispatches(result: &Qwen38GenerateResult) -> Vec<u64> {
    let start = result.prompt_len.saturating_sub(1);
    result
        .dispatches
        .get(start.min(result.dispatches.len())..)
        .unwrap_or(&[])
        .to_vec()
}

#[cfg(target_os = "macos")]
fn generate_once(
    session: &mut Qwen38HybridDecodeSession,
    tokenizer: &hawking_core::tokenizer::Tokenizer,
    prompt_ids: &[u32],
    max_new: usize,
) -> Result<Value, String> {
    session.reset();
    let _ = session.drain_dispatched_kernel_names();
    let result = generate_greedy(session, prompt_ids, max_new).map_err(|e| e.to_string())?;
    let histogram = session.dispatched_kernel_histogram();
    let names = session.drain_dispatched_kernel_names();
    let text = result.decode_new(tokenizer).map_err(|e| e.to_string())?;
    let new_ids = result.new_tokens().to_vec();
    let complete = complete_token_gpu_ns(&result);
    let disp = complete_token_dispatches(&result);
    let launched = session.launched_gated_delta_kernel();
    Ok(json!({
        "generated_text": text,
        "new_token_ids": new_ids,
        "prompt_len": result.prompt_len,
        "decode_steps": result.decode_steps,
        "fallbacks": result.fallbacks,
        "dense_w_materialized": result.dense_w_materialized,
        "decode_wall_ns": result.decode_wall_ns,
        "prefill_wall_ns": result.prefill_wall_ns,
        "complete_token_gpu_ns": complete,
        "complete_token_gpu_ns_median": median_u64(complete.clone()),
        "complete_token_dispatches": disp,
        "complete_token_dispatches_last": disp.last().copied(),
        "theoretical_dispatches": session.theoretical_dispatches(),
        "launched_gated_delta_kernel": launched,
        "fuse_ba_delta": session.fuse_ba_delta,
        "dn_state_kernel": session.dn_state_kernel.as_str(),
        "kernel_histogram": histogram.iter().map(|(k, n)| json!({"kernel": k, "count": n})).collect::<Vec<_>>(),
        "kernel_names": names,
        "gpu_timestamp_authority":
            "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy",
    }))
}

/// The sealed graph MINUS the state-kernel pin. Everything the resident sets
/// stays set; the one thing under test is left to `from_env`, which with the
/// levers unset is the promoted default.
#[cfg(target_os = "macos")]
fn apply_sealed_but_never_pin_the_state_kernel(session: &mut Qwen38HybridDecodeSession) {
    session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);
    session.set_fuse_add_rmsnorm(true, false);
    session.set_fuse_ba_delta(false, false);
    // DELIBERATELY ABSENT: session.set_dn_state_kernel(..). Calling it here
    // would overwrite the value `open` read from the environment and this
    // example would verify nothing.
}

#[cfg(target_os = "macos")]
fn run(args: Args) {
    std::env::set_var("HAWKING_TRACE_DISPATCH", "1");
    std::env::set_var("HAWKING_QWEN_RESIDENCY", "1");
    std::env::set_var("HAWKING_QWEN38_IGNORE_EOS", "1");
    // Unset every lever under test. An inherited value would make a promoted
    // default indistinguishable from an env var someone left exported.
    for k in [
        "HAWKING_QWEN38_DN_STATE",
        "HAWKING_AFFINE2_GEO",
        "HAWKING_Q4_UNPACK",
        "HAWKING_QWEN38_FAST",
        "HAWKING_QWEN38_FUSE_MLP",
        "HAWKING_QWEN38_FUSE_GQA_QKV",
        "HAWKING_QWEN38_FUSE_DN_INPROJ",
        "HAWKING_QWEN38_FUSE_ADD_RMSNORM",
        "HAWKING_QWEN38_FUSE_BA_DELTA",
    ] {
        std::env::remove_var(k);
    }

    let tokenizer = load_qwen38_tokenizer(&args.tokenizer).unwrap_or_else(|e| fail(e));
    let rendered = if args.raw_prompt {
        args.prompt.clone()
    } else {
        render_qwen38_user_chat(&args.prompt)
    };
    let prompt_ids = tokenizer
        .encode(&rendered, false)
        .unwrap_or_else(|e| fail(e));
    if prompt_ids.is_empty() {
        fail("prompt encoded to zero tokens");
    }

    let load_start = concurrent_load();
    let open_started = Instant::now();
    let mut session = Qwen38HybridDecodeSession::open(&args.artifact_root, args.max_seq_len)
        .unwrap_or_else(|e| fail(e));
    let session_open_s = open_started.elapsed().as_secs_f64();

    // What `open` read from the environment, BEFORE any example touches it.
    let state_kernel_at_open = session.dn_state_kernel.as_str().to_owned();
    apply_sealed_but_never_pin_the_state_kernel(&mut session);
    let state_kernel_after_sealed = session.dn_state_kernel.as_str().to_owned();

    let mut runs: Vec<Value> = Vec::new();
    for rep in 0..(args.warmup + args.reps) {
        let v = generate_once(&mut session, &tokenizer, &prompt_ids, args.max_new_tokens)
            .unwrap_or_else(|e| fail(e));
        if rep >= args.warmup {
            eprintln!(
                "sealed default rep {}/{} launched={}",
                rep + 1,
                args.warmup + args.reps,
                v["launched_gated_delta_kernel"]
            );
            runs.push(v);
        }
    }
    let last = runs.last().cloned().unwrap_or(Value::Null);
    let gpu: Vec<u64> = runs
        .iter()
        .filter_map(|r| r["complete_token_gpu_ns_median"].as_u64())
        .collect();

    let doc = json!({
        "schema": "hawking.future.sealed_default.raw.v1",
        "obligation": "G126",
        "question": "with every lever unset, what does the dispatcher actually launch?",
        "artifact_root": args.artifact_root,
        "tokenizer": args.tokenizer,
        "max_new_tokens": args.max_new_tokens,
        "reps": args.reps,
        "warmup": args.warmup,
        "session_open_s": session_open_s,
        "levers_unset": true,
        "dn_state_kernel_at_open": state_kernel_at_open,
        "dn_state_kernel_after_sealed_config": state_kernel_after_sealed,
        "the_state_kernel_was_never_pinned": true,
        "runs": runs,
        "last": last,
        "complete_token_gpu_ns_median_of_medians": median_u64(gpu.clone()),
        "complete_token_gpu_ns_medians": gpu,
        "concurrent_load_start": load_start,
        "concurrent_load": concurrent_load(),
        "absolute_ms_are_measured_under_load": true,
        "claim_boundary":
            "this example verifies KERNEL IDENTITY and DISPATCH COUNT under the \
             sealed default. Timings are recorded but are not a protected \
             absolute and must not be promoted as one.",
        "gpu_timestamp_authority":
            "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy",
        "git_head": std::process::Command::new("git")
            .args(["--no-optional-locks", "rev-parse", "HEAD"])
            .output().ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_owned()),
    });
    let text = serde_json::to_string_pretty(&doc).unwrap();
    if let Some(p) = &args.out {
        if let Some(d) = p.parent() {
            let _ = fs::create_dir_all(d);
        }
        fs::write(p, format!("{text}\n")).unwrap_or_else(|e| fail(e));
        eprintln!("wrote {}", p.display());
    } else {
        println!("{text}");
    }
}

#[cfg(not(target_os = "macos"))]
fn run(_args: Args) {
    fail("this example requires macOS/Metal");
}

fn main() {
    run(parse_args());
}

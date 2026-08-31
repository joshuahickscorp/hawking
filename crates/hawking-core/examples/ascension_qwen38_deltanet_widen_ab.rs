//! 628-graph A/B: production unfused vi-SIMD vs widen_f4 on the real decode path.
//!
//! One process, one catalog, both arms back-to-back. Isolated family CBs
//! sit next to complete-token generate so a probe that does not survive
//! integration is named, not absorbed. Token identity is token-id equality
//! with fallbacks 0; argmax is not recorded as parity.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core \
//!   --example ascension_qwen38_deltanet_widen_ab
//! ./tools/gpu_lane_lock.sh y3widen \
//!   workspace/ops/build/rust/release-fast/examples/ascension_qwen38_deltanet_widen_ab \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A \
//!   --tokenizer ~/noetic/NOETIC_PARENT_A/tokenizer.json \
//!   --reps 7 --max-new-tokens 32 \
//!   --out receipts/future/_DELTANET_WIDEN_AB_raw.json
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
    "usage: ascension_qwen38_deltanet_widen_ab --artifact-root DIR --tokenizer PATH \
        [--prompt TEXT] [--raw-prompt] [--max-new-tokens N] [--max-seq-len N] \
        [--reps N] [--warmup N] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("ascension_qwen38_deltanet_widen_ab: {message}");
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
fn apply_sealed_628(session: &mut Qwen38HybridDecodeSession) {
    session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);
    session.set_fuse_add_rmsnorm(true, false);
    session.set_fuse_ba_delta(false, false);
    session.set_dn_state_kernel(Qwen38DeltaNetStateKernel::Baseline);
}

#[cfg(target_os = "macos")]
fn apply_widen_f4(session: &mut Qwen38HybridDecodeSession) {
    session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);
    session.set_fuse_add_rmsnorm(true, false);
    // Flag stays off: the kernel itself folds ba_to_decay. That is the
    // production-path wiring this A/B is here to exercise.
    session.set_fuse_ba_delta(false, false);
    session.set_dn_state_kernel(Qwen38DeltaNetStateKernel::WidenF4);
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

#[cfg(target_os = "macos")]
fn run(args: Args) {
    std::env::set_var("HAWKING_TRACE_DISPATCH", "1");
    std::env::set_var("HAWKING_QWEN_RESIDENCY", "1");
    std::env::set_var("HAWKING_QWEN38_IGNORE_EOS", "1");
    std::env::remove_var("HAWKING_QWEN38_FUSE_MLP");
    std::env::remove_var("HAWKING_QWEN38_FUSE_GQA_QKV");
    std::env::remove_var("HAWKING_QWEN38_FUSE_DN_INPROJ");
    std::env::remove_var("HAWKING_QWEN38_FUSE_ADD_RMSNORM");
    std::env::remove_var("HAWKING_QWEN38_FUSE_BA_DELTA");
    std::env::remove_var("HAWKING_QWEN38_DN_STATE");
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
    eprintln!(
        "qwen38 deltanet widen_ab: open {} max_seq={} max_new={} reps={} warmup={}",
        args.artifact_root.display(),
        args.max_seq_len,
        args.max_new_tokens,
        args.reps,
        args.warmup
    );
    let open_started = Instant::now();
    let mut session = Qwen38HybridDecodeSession::open(&args.artifact_root, args.max_seq_len)
        .unwrap_or_else(|e| fail(e));
    let session_open_s = open_started.elapsed().as_secs_f64();
    apply_sealed_628(&mut session);
    eprintln!(
        "session open {session_open_s:.3}s dense_w={} theoretical_628={} launched={}",
        session.dense_w_materialized,
        session.theoretical_dispatches(),
        session.launched_gated_delta_kernel()
    );

    // Isolated family CBs: same work the organ decompose timed, so we can
    // say whether the 0.702 ms still exists next to the complete token.
    apply_sealed_628(&mut session);
    let iso_unfused = reps_cb("gated_delta_unfused", args.reps, || {
        session
            .measure_isolated_gated_delta()
            .map_err(|e| e.to_string())
    })
    .unwrap_or_else(|e| fail(e));

    session.set_fuse_ba_delta(true, false);
    session.set_dn_state_kernel(Qwen38DeltaNetStateKernel::Baseline);
    let iso_fused_ba = reps_cb("gated_delta_fused_ba", args.reps, || {
        session
            .measure_isolated_dn_state_update()
            .map_err(|e| e.to_string())
    })
    .unwrap_or_else(|e| fail(e));

    apply_widen_f4(&mut session);
    let iso_f4 = reps_cb("gated_delta_widen_f4", args.reps, || {
        session
            .measure_isolated_dn_state_update()
            .map_err(|e| e.to_string())
    })
    .unwrap_or_else(|e| fail(e));

    apply_sealed_628(&mut session);
    let organ_incumbent = reps_cb("organ_incumbent_628", args.reps, || {
        session
            .measure_isolated_organ("deltanet")
            .map_err(|e| e.to_string())
    })
    .unwrap_or_else(|e| fail(e));
    apply_widen_f4(&mut session);
    let organ_f4 = reps_cb("organ_widen_f4", args.reps, || {
        session
            .measure_isolated_organ("deltanet")
            .map_err(|e| e.to_string())
    })
    .unwrap_or_else(|e| fail(e));

    let load_after_iso = concurrent_load();

    // Complete-token A/B, interleaved so load drift hits both arms.
    let mut incumbent_runs: Vec<Value> = Vec::new();
    let mut widen_runs: Vec<Value> = Vec::new();
    let total = args.warmup + args.reps;
    for i in 0..total {
        let warm = i < args.warmup;
        apply_sealed_628(&mut session);
        eprintln!(
            "decode incumbent 628 {} {i}/{total} launched={}",
            if warm { "warmup" } else { "rep" },
            session.launched_gated_delta_kernel()
        );
        let inc = generate_once(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
        )
        .unwrap_or_else(|e| fail(e));
        apply_widen_f4(&mut session);
        eprintln!(
            "decode widen_f4 {} {i}/{total} launched={}",
            if warm { "warmup" } else { "rep" },
            session.launched_gated_delta_kernel()
        );
        let f4 = generate_once(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
        )
        .unwrap_or_else(|e| fail(e));
        if !warm {
            incumbent_runs.push(inc);
            widen_runs.push(f4);
        }
    }

    let load_end = concurrent_load();
    let inc_medians: Vec<u64> = incumbent_runs
        .iter()
        .filter_map(|r| r.get("complete_token_gpu_ns_median").and_then(Value::as_u64))
        .collect();
    let f4_medians: Vec<u64> = widen_runs
        .iter()
        .filter_map(|r| r.get("complete_token_gpu_ns_median").and_then(Value::as_u64))
        .collect();

    let body = json!({
        "schema": "hawking.future.deltanet_widen_ab.raw.v1",
        "git_head": git_head(),
        "artifact_root": args.artifact_root,
        "tokenizer": args.tokenizer,
        "prompt": args.prompt,
        "rendered_prompt": rendered,
        "prompt_ids": prompt_ids,
        "max_new_tokens": args.max_new_tokens,
        "max_seq_len": args.max_seq_len,
        "reps": args.reps,
        "warmup": args.warmup,
        "session_open_s": session_open_s,
        "dense_w_materialized": session.dense_w_materialized,
        "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
        "absolute_ms_are_measured_under_load": true,
        "production_fusions": {
            "mlp": "GateUpSwiglu",
            "fuse_gqa_qkv": true,
            "fuse_dn_inproj": true,
            "fuse_add_rmsnorm": true,
            "fuse_ba_delta": false,
            "dn_state_kernel_incumbent": "baseline",
            "dn_state_kernel_candidate": "widen_f4",
        },
        "expected_kernels": {
            "incumbent": "qwen38_gated_delta_decode_vi_simd",
            "widen_f4": QWEN38_DN_STATE_F4_KERNEL,
        },
        "concurrent_load_start": load_start,
        "concurrent_load_after_isolated": load_after_iso,
        "concurrent_load": load_end,
        "isolated_gated_delta": {
            "unfused": iso_unfused,
            "fused_ba": iso_fused_ba,
            "widen_f4": iso_f4,
        },
        "isolated_organ": {
            "incumbent": organ_incumbent,
            "widen_f4": organ_f4,
        },
        "decode": {
            "interleaved": true,
            "incumbent": incumbent_runs,
            "widen_f4": widen_runs,
            "incumbent_complete_token_gpu_ns_median_reps": inc_medians,
            "widen_f4_complete_token_gpu_ns_median_reps": f4_medians,
            "incumbent_complete_token_gpu_ns_median": median_u64(inc_medians.clone()),
            "widen_f4_complete_token_gpu_ns_median": median_u64(f4_medians.clone()),
        },
        "gpu_timestamp_authority":
            "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy",
    });
    let text = serde_json::to_string_pretty(&body).unwrap_or_else(|e| fail(e));
    println!("{text}");
    if let Some(path) = &args.out {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(path, format!("{text}\n")).unwrap_or_else(|e| fail(e));
        eprintln!("wrote {}", path.display());
    }
}

#[cfg(not(target_os = "macos"))]
fn run(_args: Args) {
    fail("requires macOS Metal");
}

fn main() {
    run(parse_args());
}

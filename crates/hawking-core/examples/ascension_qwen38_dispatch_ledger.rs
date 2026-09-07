//! Dispatch ledger A/B: sealed 756-dispatch parent vs residual+RMSNorm fusion.
//!
//! Opens one catalog (does not load a second 27B). Does not mutate
//! ~/noetic/NOETIC_PARENT_A — fusion is an encode-path child of the same bytes.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core \
//!   --example ascension_qwen38_dispatch_ledger
//! ./tools/gpu_lane_lock.sh qwen38-dispatch-ledger \
//!   workspace/ops/build/rust/release-fast/examples/ascension_qwen38_dispatch_ledger \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A \
//!   --tokenizer ~/models/qwen3.8-27b-abliterated-bf16/tokenizer.json \
//!   --out receipts/headless/_DISPATCH_LEDGER_raw.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

#[cfg(target_os = "macos")]
use hawking_core::model::qwen38_hybrid_decode::{
    generate_greedy, load_qwen38_tokenizer, qwen38_fused_dispatches_per_token_ex,
    render_qwen38_user_chat, Qwen38FusionParity, Qwen38GenerateResult, Qwen38HybridDecodeSession,
    Qwen38MlpFusion, QWEN38_ADD_RMSNORM_BAD_KERNEL, QWEN38_ADD_RMSNORM_KERNEL,
    QWEN38_ADD_RMSNORM_SAVED_PER_TOKEN,
};

fn usage() -> &'static str {
    "usage: ascension_qwen38_dispatch_ledger --artifact-root DIR --tokenizer PATH \
        [--prompt TEXT] [--raw-prompt] [--max-new-tokens N] [--max-seq-len N] \
        [--reps N] [--skip-decode] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("ascension_qwen38_dispatch_ledger: {message}");
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
    skip_decode: bool,
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
    let mut max_new_tokens = 16usize;
    let mut max_seq_len = 128usize;
    let mut reps = 2usize;
    let mut skip_decode = false;
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
            "--skip-decode" => skip_decode = true,
            "--out" => out = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage())))),
            other => fail(format!("unknown {other}; {}", usage())),
        }
    }
    Args {
        artifact_root: artifact_root.unwrap_or_else(|| fail(usage())),
        tokenizer: tokenizer.unwrap_or_else(|| fail(usage())),
        prompt,
        raw_prompt,
        max_new_tokens,
        max_seq_len,
        reps: reps.max(1),
        skip_decode,
        out,
    }
}

fn git_head() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn now_iso() -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("unix:{ts}")
}

fn max_abs(a: f32, b: f32, c: f32) -> f32 {
    a.max(b).max(c)
}

#[cfg(target_os = "macos")]
fn parity_json(p: &Qwen38FusionParity) -> Value {
    json!({
        "fusion": p.fusion,
        "layer": p.layer,
        "unfused_dispatches": p.unfused_dispatches,
        "fused_dispatches": p.fused_pair_dispatches,
        "unfused_gpu_ns": p.unfused_gpu_ns,
        "fused_gpu_ns": p.fused_pair_gpu_ns,
        "max_abs_diff_residual": p.max_abs_diff_gate,
        "max_abs_diff_norm": p.max_abs_diff_up,
        "max_abs_diff": max_abs(p.max_abs_diff_gate, p.max_abs_diff_up, p.max_abs_diff_act),
        "dense_w_materialized": p.dense_w_materialized,
        "kernel": if p.fusion.contains("plainweight") {
            QWEN38_ADD_RMSNORM_BAD_KERNEL
        } else {
            QWEN38_ADD_RMSNORM_KERNEL
        },
    })
}

#[cfg(target_os = "macos")]
fn tok_s(result: &Qwen38GenerateResult) -> Option<f64> {
    if result.decode_steps == 0 || result.decode_wall_ns == 0 {
        return None;
    }
    Some(result.decode_steps as f64 / (result.decode_wall_ns as f64 / 1e9))
}

#[cfg(target_os = "macos")]
fn generate_arm(
    session: &mut Qwen38HybridDecodeSession,
    tokenizer: &hawking_core::tokenizer::Tokenizer,
    prompt_ids: &[u32],
    max_new: usize,
    reps: usize,
) -> Result<Value, String> {
    let mut texts = Vec::new();
    let mut ids = Vec::new();
    let mut tok_s_reps = Vec::new();
    let mut dispatch_reps = Vec::new();
    let mut decode_wall_ns = Vec::new();
    let mut gpu_median = Vec::new();
    let mut fallbacks = Vec::new();
    let mut kernels = Vec::new();
    for i in 0..reps {
        session.reset();
        let result = generate_greedy(session, prompt_ids, max_new).map_err(|e| e.to_string())?;
        let text = result.decode_new(tokenizer).map_err(|e| e.to_string())?;
        let new_ids = result.new_tokens().to_vec();
        let ts = tok_s(&result);
        let disp = result.dispatches.last().copied();
        let names = session.drain_dispatched_kernel_names();
        eprintln!(
            "  rep {i}: tok/s={ts:?} dispatches={disp:?} new={} kernels={}",
            new_ids.len(),
            names.len()
        );
        texts.push(text);
        ids.push(new_ids);
        tok_s_reps.push(ts);
        dispatch_reps.push(disp);
        decode_wall_ns.push(result.decode_wall_ns);
        gpu_median.push(result.median_gpu_ns_per_token());
        fallbacks.push(result.fallbacks);
        kernels.push(names);
    }
    let finite: Vec<f64> = tok_s_reps.iter().copied().flatten().collect();
    let mean = if finite.is_empty() {
        None
    } else {
        Some(finite.iter().sum::<f64>() / finite.len() as f64)
    };
    Ok(json!({
        "reps": reps,
        "generated_text_verbatim": texts.first(),
        "generated_text_all_reps": texts,
        "new_token_ids": ids.first(),
        "new_token_ids_all_reps": ids,
        "tok_s_reps": tok_s_reps,
        "tok_s_mean": mean,
        "tok_s_min": finite.iter().copied().reduce(f64::min),
        "tok_s_max": finite.iter().copied().reduce(f64::max),
        "dispatches_last_step_reps": dispatch_reps,
        "decode_wall_ns_reps": decode_wall_ns,
        "median_gpu_ns_per_token_reps": gpu_median,
        "fallbacks_reps": fallbacks,
        "dispatched_kernels_rep0": kernels.first(),
        "dense_w_materialized": 0,
    }))
}

#[cfg(target_os = "macos")]
fn probe_dispatches(session: &mut Qwen38HybridDecodeSession, token: u32) -> Result<Value, String> {
    session.reset();
    let theoretical = session.theoretical_dispatches();
    let (sampled, dispatches, timing) = session
        .measure_token_dispatches(token)
        .map_err(|e| e.to_string())?;
    let names = session.drain_dispatched_kernel_names();
    Ok(json!({
        "theoretical": theoretical,
        "measured": dispatches,
        "matches_theoretical": dispatches == theoretical,
        "sampled": sampled,
        "gpu_ns": timing.gpu_ns,
        "wait_ns": timing.wait_ns,
        "mlp_fusion": session.mlp_fusion.as_str(),
        "fuse_gqa_qkv": session.fuse_gqa_qkv,
        "fuse_dn_inproj": session.fuse_dn_inproj,
        "fuse_add_rmsnorm": session.fuse_add_rmsnorm,
        "fuse_add_rmsnorm_bad": session.fuse_add_rmsnorm_bad,
        "kernel_names": names,
        "sentinel_kernel_present": names.iter().any(|n| n == QWEN38_ADD_RMSNORM_KERNEL),
        "bad_kernel_present": names.iter().any(|n| n == QWEN38_ADD_RMSNORM_BAD_KERNEL),
    }))
}

#[cfg(target_os = "macos")]
fn run(args: Args) {
    // Structural kernel-name harvest is the candidate sentinel. Fusion env
    // must not leak in from the parent process: this example sets the graph.
    std::env::set_var("HAWKING_TRACE_DISPATCH", "1");
    std::env::remove_var("HAWKING_QWEN38_FUSE_MLP");
    std::env::remove_var("HAWKING_QWEN38_FUSE_GQA_QKV");
    std::env::remove_var("HAWKING_QWEN38_FUSE_DN_INPROJ");
    std::env::remove_var("HAWKING_QWEN38_FUSE_ADD_RMSNORM");
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

    eprintln!(
        "qwen38 dispatch ledger: open {} max_seq={}",
        args.artifact_root.display(),
        args.max_seq_len
    );
    let open_started = Instant::now();
    let mut session = Qwen38HybridDecodeSession::open(&args.artifact_root, args.max_seq_len)
        .unwrap_or_else(|e| fail(e));
    let session_open_s = open_started.elapsed().as_secs_f64();
    eprintln!("session open {session_open_s:.3}s");

    session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);
    session.set_fuse_add_rmsnorm(false, false);

    eprintln!("parity: add_residual_rmsnorm layer 0 (good kernel)");
    let good_parity = session
        .measure_add_rmsnorm_fusion_parity(0, false)
        .unwrap_or_else(|e| fail(e));
    eprintln!(
        "  unfused={} fused={} max_abs residual={} norm={}",
        good_parity.unfused_dispatches,
        good_parity.fused_pair_dispatches,
        good_parity.max_abs_diff_gate,
        good_parity.max_abs_diff_up
    );

    eprintln!("parity: BAD control plainweight");
    let bad_parity = session
        .measure_add_rmsnorm_fusion_parity(0, true)
        .unwrap_or_else(|e| fail(e));
    eprintln!(
        "  bad max_abs residual={} norm={}",
        bad_parity.max_abs_diff_gate, bad_parity.max_abs_diff_up
    );

    let probe_token = prompt_ids[0];
    let mut dispatch_probes = Vec::new();

    session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);
    session.set_fuse_add_rmsnorm(false, false);
    eprintln!("dispatch probe: parent 756 (noop control)");
    dispatch_probes.push(json!({
        "id": "parent_756",
        "role": "noop_control",
        "probe": probe_dispatches(&mut session, probe_token).unwrap_or_else(|e| fail(e)),
    }));

    session.set_fuse_add_rmsnorm(true, false);
    eprintln!("dispatch probe: add_rmsnorm 628");
    dispatch_probes.push(json!({
        "id": "add_rmsnorm_628",
        "role": "candidate",
        "probe": probe_dispatches(&mut session, probe_token).unwrap_or_else(|e| fail(e)),
    }));

    session.set_fuse_add_rmsnorm(true, true);
    eprintln!("dispatch probe: BAD plainweight");
    dispatch_probes.push(json!({
        "id": "add_rmsnorm_bad",
        "role": "bad_control",
        "probe": probe_dispatches(&mut session, probe_token).unwrap_or_else(|e| fail(e)),
    }));

    let mut decode_arms = json!({});
    if !args.skip_decode {
        session.set_fuse_add_rmsnorm(false, false);
        eprintln!("decode NOOP parent-756 reps={}", args.reps);
        decode_arms["parent_756"] = generate_arm(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
            args.reps,
        )
        .unwrap_or_else(|e| fail(e));

        session.set_fuse_add_rmsnorm(true, false);
        eprintln!("decode CANDIDATE add_rmsnorm-628 reps={}", args.reps);
        decode_arms["add_rmsnorm_628"] = generate_arm(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
            args.reps,
        )
        .unwrap_or_else(|e| fail(e));

        session.set_fuse_add_rmsnorm(true, true);
        eprintln!("decode BAD plainweight reps=1");
        decode_arms["add_rmsnorm_bad"] = generate_arm(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
            1,
        )
        .unwrap_or_else(|e| fail(e));
    }

    let parent_n =
        qwen38_fused_dispatches_per_token_ex(Qwen38MlpFusion::GateUpSwiglu, true, true, false);
    let candidate_n =
        qwen38_fused_dispatches_per_token_ex(Qwen38MlpFusion::GateUpSwiglu, true, true, true);
    let body = json!({
        "schema": "hawking.headless.dispatch_ledger_raw.v1",
        "generated_at": now_iso(),
        "git_head": git_head(),
        "did_not_load_second_27b": true,
        "did_not_write_under_models": true,
        "did_not_mutate_parent": true,
        "artifact_root": args.artifact_root,
        "tokenizer": args.tokenizer,
        "prompt": args.prompt,
        "rendered_prompt": rendered,
        "prompt_ids": prompt_ids,
        "max_new_tokens": args.max_new_tokens,
        "session_open_s": session_open_s,
        "dense_w_materialized": 0,
        "expanded_to_q4": 0,
        "expanded_to_float_gemv": 0,
        "counting": {
            "method": "TokenCommandBuffer.dispatch_count, one kernel launch = one dispatch",
            "parent_756": parent_n,
            "candidate": candidate_n,
            "saved": QWEN38_ADD_RMSNORM_SAVED_PER_TOKEN,
            "command_buffers": 1,
        },
        "kernels": [QWEN38_ADD_RMSNORM_KERNEL, QWEN38_ADD_RMSNORM_BAD_KERNEL],
        "parity": {
            "add_residual_rmsnorm": parity_json(&good_parity),
            "bad_plainweight": parity_json(&bad_parity),
        },
        "dispatch_probes": dispatch_probes,
        "decode": decode_arms,
        "skip_decode": args.skip_decode,
    });

    println!("{}", serde_json::to_string_pretty(&body).unwrap());
    if let Some(path) = &args.out {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(path, serde_json::to_string_pretty(&body).unwrap()).unwrap_or_else(|e| fail(e));
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

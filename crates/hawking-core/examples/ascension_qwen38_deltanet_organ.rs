//! N026 DeltaNet organ: kernel autopsy + two state-update changes.
//!
//! Opens one catalog (does not load a second 27B). Does not mutate
//! ~/noetic/NOETIC_PARENT_A. Isolated organ CBs partition production GPU
//! time; production stays one command buffer.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core \
//!   --example ascension_qwen38_deltanet_organ
//! ./tools/gpu_lane_lock.sh n026-deltanet \
//!   workspace/ops/build/rust/release-fast/examples/ascension_qwen38_deltanet_organ \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A \
//!   --tokenizer ~/models/qwen3.8-27b-abliterated-bf16/tokenizer.json \
//!   --reps 7 --out receipts/headless/_DELTANET_ORGAN_raw.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

#[cfg(target_os = "macos")]
use hawking_core::model::qwen38_hybrid_decode::{
    generate_greedy, load_qwen38_tokenizer, qwen38_fused_dispatches_per_token_full,
    render_qwen38_user_chat, Qwen38DeltaNetStateKernel, Qwen38DeltaNetStateParity,
    Qwen38GenerateResult, Qwen38HybridDecodeSession, Qwen38MlpFusion, QWEN38_BA_DELTA_BAD_KERNEL,
    QWEN38_BA_DELTA_KERNEL, QWEN38_DN_STATE_F4_KERNEL, QWEN38_DN_STATE_TG32_KERNEL,
};

fn usage() -> &'static str {
    "usage: ascension_qwen38_deltanet_organ --artifact-root DIR --tokenizer PATH \
        [--prompt TEXT] [--raw-prompt] [--max-new-tokens N] [--max-seq-len N] \
        [--reps N] [--skip-decode] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("ascension_qwen38_deltanet_organ: {message}");
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
    let mut reps = 7usize;
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

fn median_u64(mut v: Vec<u64>) -> Option<u64> {
    if v.is_empty() {
        return None;
    }
    v.sort_unstable();
    Some(v[v.len() / 2])
}

fn median_f64(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(s[s.len() / 2])
}

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
fn parity_json(p: &Qwen38DeltaNetStateParity) -> Value {
    json!({
        "kernel": p.kernel,
        "layer": p.layer,
        "max_abs_diff_rec_out": p.max_abs_diff_rec_out,
        "max_abs_diff_rec_state": p.max_abs_diff_rec_state,
        "baseline_gpu_ns": p.baseline_gpu_ns,
        "candidate_gpu_ns": p.candidate_gpu_ns,
        "baseline_dispatches": p.baseline_dispatches,
        "candidate_dispatches": p.candidate_dispatches,
        "dense_w_materialized": p.dense_w_materialized,
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
    let gpu_vals: Vec<u64> = gpu_median.iter().copied().flatten().collect();
    Ok(json!({
        "reps": reps,
        "generated_text_verbatim": texts.first(),
        "generated_text_all_reps": texts,
        "new_token_ids": ids.first(),
        "new_token_ids_all_reps": ids,
        "tok_s_reps": tok_s_reps,
        "tok_s_min": finite.iter().copied().reduce(f64::min),
        "tok_s_median": median_f64(&finite),
        "tok_s_max": finite.iter().copied().reduce(f64::max),
        "dispatches_last_step_reps": dispatch_reps,
        "decode_wall_ns_reps": decode_wall_ns,
        "median_gpu_ns_per_token_reps": gpu_median,
        "gpu_ns_min": gpu_vals.iter().copied().min(),
        "gpu_ns_median": median_u64(gpu_vals.clone()),
        "gpu_ns_max": gpu_vals.iter().copied().max(),
        "fallbacks_reps": fallbacks,
        "dispatched_kernels_rep0": kernels.first(),
        "dense_w_materialized": 0,
        "mlp_fusion": session.mlp_fusion.as_str(),
        "fuse_gqa_qkv": session.fuse_gqa_qkv,
        "fuse_dn_inproj": session.fuse_dn_inproj,
        "fuse_add_rmsnorm": session.fuse_add_rmsnorm,
        "fuse_ba_delta": session.fuse_ba_delta,
        "fuse_ba_delta_bad": session.fuse_ba_delta_bad,
        "dn_state_kernel": session.dn_state_kernel.as_str(),
        "theoretical_dispatches": session.theoretical_dispatches(),
    }))
}

#[cfg(target_os = "macos")]
fn run(args: Args) {
    std::env::set_var("HAWKING_TRACE_DISPATCH", "1");
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

    eprintln!(
        "qwen38 deltanet organ: open {} max_seq={}",
        args.artifact_root.display(),
        args.max_seq_len
    );
    let open_started = Instant::now();
    let mut session = Qwen38HybridDecodeSession::open(&args.artifact_root, args.max_seq_len)
        .unwrap_or_else(|e| fail(e));
    let session_open_s = open_started.elapsed().as_secs_f64();
    eprintln!("session open {session_open_s:.3}s dense_w={}", session.dense_w_materialized);

    session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);
    session.set_fuse_add_rmsnorm(true, false);
    session.set_fuse_ba_delta(true, false);
    session.set_dn_state_kernel(Qwen38DeltaNetStateKernel::Baseline);

    eprintln!("parity: widen_f4 vs baseline rec_out+rec_state");
    let f4_parity = session
        .measure_dn_state_kernel_parity(0, Qwen38DeltaNetStateKernel::WidenF4, false)
        .unwrap_or_else(|e| fail(e));
    eprintln!(
        "  f4 rec_out={} rec_state={}",
        f4_parity.max_abs_diff_rec_out, f4_parity.max_abs_diff_rec_state
    );

    eprintln!("parity: coalesce_tg32 vs baseline rec_out+rec_state");
    let tg_parity = session
        .measure_dn_state_kernel_parity(0, Qwen38DeltaNetStateKernel::CoalesceTg32, false)
        .unwrap_or_else(|e| fail(e));
    eprintln!(
        "  tg32 rec_out={} rec_state={}",
        tg_parity.max_abs_diff_rec_out, tg_parity.max_abs_diff_rec_state
    );

    eprintln!("parity: BAD identity decay/beta");
    let bad_parity = session
        .measure_dn_state_kernel_parity(0, Qwen38DeltaNetStateKernel::Baseline, true)
        .unwrap_or_else(|e| fail(e));
    eprintln!(
        "  bad rec_out={} rec_state={}",
        bad_parity.max_abs_diff_rec_out, bad_parity.max_abs_diff_rec_state
    );

    let noop = {
        let mut gpu = Vec::new();
        let mut wait = Vec::new();
        let mut disp = 0u64;
        eprintln!("  noop_empty reps={}", args.reps);
        for i in 0..args.reps {
            let t = session
                .measure_isolated_organ("noop_empty")
                .unwrap_or_else(|e| fail(e));
            // Empty CB: the driver sometimes omits GPUStart/End. That is 0 ns
            // of GPU work, not a failed measurement.
            let g = t.gpu_ns.unwrap_or(0);
            eprintln!("    rep{i} gpu={g} wait={} disp={}", t.wait_ns, t.dispatches);
            gpu.push(g);
            wait.push(t.wait_ns);
            disp = t.dispatches;
        }
        timing_json("noop_empty", &gpu, &wait, disp, args.reps)
    };

    eprintln!("isolated components (autopsy: why 325.5 GB/s)");
    session.set_dn_state_kernel(Qwen38DeltaNetStateKernel::Baseline);
    session.set_fuse_ba_delta(true, false);
    let iso_inproj = reps_cb("dn_inproj", args.reps, || {
        session
            .measure_isolated_dn_inproj()
            .map_err(|e| e.to_string())
    })
    .unwrap_or_else(|e| fail(e));
    let iso_rearrange = reps_cb("rearrange_48", args.reps, || {
        session
            .measure_isolated_family("rearrange_48")
            .map_err(|e| e.to_string())
    })
    .unwrap_or_else(|e| fail(e));
    let iso_gated_n = reps_cb("gated_rmsnorm_48", args.reps, || {
        session
            .measure_isolated_family("gated_rmsnorm_48")
            .map_err(|e| e.to_string())
    })
    .unwrap_or_else(|e| fail(e));

    eprintln!("isolated gated-delta state update (fused ba)");
    session.set_dn_state_kernel(Qwen38DeltaNetStateKernel::Baseline);
    let iso_delta_base = reps_cb("gated_delta_baseline", args.reps, || {
        session
            .measure_isolated_dn_state_update()
            .map_err(|e| e.to_string())
    })
    .unwrap_or_else(|e| fail(e));
    session.set_dn_state_kernel(Qwen38DeltaNetStateKernel::WidenF4);
    let iso_delta_f4 = reps_cb("gated_delta_widen_f4", args.reps, || {
        session
            .measure_isolated_dn_state_update()
            .map_err(|e| e.to_string())
    })
    .unwrap_or_else(|e| fail(e));
    session.set_dn_state_kernel(Qwen38DeltaNetStateKernel::CoalesceTg32);
    let iso_delta_tg = reps_cb("gated_delta_coalesce_tg32", args.reps, || {
        session
            .measure_isolated_dn_state_update()
            .map_err(|e| e.to_string())
    })
    .unwrap_or_else(|e| fail(e));

    eprintln!("isolated DeltaNet organ (48 layers, fused ba, 580 graph)");
    session.set_dn_state_kernel(Qwen38DeltaNetStateKernel::Baseline);
    let organ_base = reps_cb("organ_baseline", args.reps, || {
        session
            .measure_isolated_organ("deltanet")
            .map_err(|e| e.to_string())
    })
    .unwrap_or_else(|e| fail(e));
    session.set_dn_state_kernel(Qwen38DeltaNetStateKernel::WidenF4);
    let organ_f4 = reps_cb("organ_widen_f4", args.reps, || {
        session
            .measure_isolated_organ("deltanet")
            .map_err(|e| e.to_string())
    })
    .unwrap_or_else(|e| fail(e));
    session.set_dn_state_kernel(Qwen38DeltaNetStateKernel::CoalesceTg32);
    let organ_tg = reps_cb("organ_coalesce_tg32", args.reps, || {
        session
            .measure_isolated_organ("deltanet")
            .map_err(|e| e.to_string())
    })
    .unwrap_or_else(|e| fail(e));

    let mut decode_arms = json!({});
    if !args.skip_decode {
        session.set_fuse_ba_delta(true, false);
        session.set_dn_state_kernel(Qwen38DeltaNetStateKernel::Baseline);
        eprintln!("decode BASELINE vi_simd_ba reps={}", args.reps);
        decode_arms["baseline"] = generate_arm(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
            args.reps,
        )
        .unwrap_or_else(|e| fail(e));

        session.set_dn_state_kernel(Qwen38DeltaNetStateKernel::WidenF4);
        eprintln!("decode CHANGE1 widen_f4 reps={}", args.reps);
        decode_arms["widen_f4"] = generate_arm(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
            args.reps,
        )
        .unwrap_or_else(|e| fail(e));

        session.set_dn_state_kernel(Qwen38DeltaNetStateKernel::CoalesceTg32);
        eprintln!("decode CHANGE2 coalesce_tg32 reps={}", args.reps);
        decode_arms["coalesce_tg32"] = generate_arm(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
            args.reps,
        )
        .unwrap_or_else(|e| fail(e));

        session.set_dn_state_kernel(Qwen38DeltaNetStateKernel::Baseline);
        session.set_fuse_ba_delta(true, true);
        eprintln!("decode BAD identity reps=1");
        decode_arms["bad_identity"] = generate_arm(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
            1,
        )
        .unwrap_or_else(|e| fail(e));
    }

    let candidate_n = qwen38_fused_dispatches_per_token_full(
        Qwen38MlpFusion::GateUpSwiglu,
        true,
        true,
        true,
        true,
    );
    let body = json!({
        "schema": "hawking.headless.deltanet_organ.raw.v1",
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
        "dense_w_materialized": session.dense_w_materialized,
        "expanded_to_q4": 0,
        "expanded_to_float_gemv": 0,
        "counting": {
            "method": "TokenCommandBuffer.dispatch_count, one kernel launch = one dispatch",
            "candidate_580": candidate_n,
            "command_buffers": 1,
        },
        "kernels": [
            QWEN38_BA_DELTA_KERNEL,
            QWEN38_BA_DELTA_BAD_KERNEL,
            QWEN38_DN_STATE_F4_KERNEL,
            QWEN38_DN_STATE_TG32_KERNEL,
        ],
        "parity": {
            "widen_f4": parity_json(&f4_parity),
            "coalesce_tg32": parity_json(&tg_parity),
            "bad_identity": parity_json(&bad_parity),
        },
        "noop_empty": noop,
        "isolated_components": {
            "dn_inproj": iso_inproj,
            "rearrange_48": iso_rearrange,
            "gated_rmsnorm_48": iso_gated_n,
        },
        "isolated_gated_delta": {
            "baseline": iso_delta_base,
            "widen_f4": iso_delta_f4,
            "coalesce_tg32": iso_delta_tg,
        },
        "isolated_organ": {
            "baseline": organ_base,
            "widen_f4": organ_f4,
            "coalesce_tg32": organ_tg,
        },
        "decode": decode_arms,
        "skip_decode": args.skip_decode,
        "gpu_timestamp_authority":
            "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy",
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

//! N025 organ bandwidth: per-organ GPU ns + GB/s, and a dispatch cut below 628.
//!
//! Opens one catalog (does not load a second 27B). Does not mutate
//! ~/noetic/NOETIC_PARENT_A. Isolated organ CBs partition production GPU
//! time; production stays one command buffer.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core \
//!   --example ascension_qwen38_organ_bandwidth
//! ./tools/gpu_lane_lock.sh n025-organ \
//!   workspace/ops/build/rust/release-fast/examples/ascension_qwen38_organ_bandwidth \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A \
//!   --tokenizer ~/models/qwen3.8-27b-abliterated-bf16/tokenizer.json \
//!   --reps 7 --out receipts/headless/_ORGAN_BANDWIDTH_raw.json
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
    qwen38_fused_dispatches_per_token_full, render_qwen38_user_chat, Qwen38FusionParity,
    Qwen38GenerateResult, Qwen38HybridDecodeSession, Qwen38MlpFusion, QWEN38_BA_DELTA_BAD_KERNEL,
    QWEN38_BA_DELTA_KERNEL, QWEN38_BA_DELTA_SAVED_PER_TOKEN,
};

const ORGANS: &[&str] = &[
    "embedding",
    "gqa_attention",
    "deltanet",
    "mlp_gate_up",
    "mlp_down",
    "q4_remainder",
    "lm_head",
    "sampling",
];

fn usage() -> &'static str {
    "usage: ascension_qwen38_organ_bandwidth --artifact-root DIR --tokenizer PATH \
        [--prompt TEXT] [--raw-prompt] [--max-new-tokens N] [--max-seq-len N] \
        [--reps N] [--skip-decode] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("ascension_qwen38_organ_bandwidth: {message}");
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

#[cfg(target_os = "macos")]
fn parity_json(p: &Qwen38FusionParity) -> Value {
    json!({
        "fusion": p.fusion,
        "layer": p.layer,
        "unfused_dispatches": p.unfused_dispatches,
        "fused_dispatches": p.fused_pair_dispatches,
        "unfused_gpu_ns": p.unfused_gpu_ns,
        "fused_gpu_ns": p.fused_pair_gpu_ns,
        "max_abs_diff": p.max_abs_diff_act
            .max(p.max_abs_diff_gate)
            .max(p.max_abs_diff_up),
        "max_abs_diff_rec_out": p.max_abs_diff_act,
        "dense_w_materialized": p.dense_w_materialized,
        "kernel": if p.fusion.contains("identity") {
            QWEN38_BA_DELTA_BAD_KERNEL
        } else {
            QWEN38_BA_DELTA_KERNEL
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
        "theoretical_dispatches": session.theoretical_dispatches(),
    }))
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
        "fuse_ba_delta": session.fuse_ba_delta,
        "fuse_ba_delta_bad": session.fuse_ba_delta_bad,
        "kernel_names": names,
        "sentinel_kernel_present": names.iter().any(|n| n == QWEN38_BA_DELTA_KERNEL),
        "bad_kernel_present": names.iter().any(|n| n == QWEN38_BA_DELTA_BAD_KERNEL),
    }))
}

#[cfg(target_os = "macos")]
fn measure_organs(session: &Qwen38HybridDecodeSession, reps: usize) -> Result<Value, String> {
    let mut rows = Vec::new();
    let noop = {
        let mut gpu = Vec::new();
        let mut wait = Vec::new();
        let mut disp = 0u64;
        let mut no_timestamp = 0usize;
        // An EMPTY command buffer often returns no GPUEndTime-GPUStartTime on this
        // device: the observed reps are 41-458 ns, at or below the driver's
        // timestamp resolution. Aborting the whole organ census because a
        // ZERO-WORK control sometimes has no timestamp is the wrong failure mode -
        // this control exists to establish the floor, and "below resolution" is
        // information about that floor rather than a reason to discard every real
        // organ measurement behind it. Two protected windows were spent this way,
        // failing at rep 4 of 7 and then rep 2 of 7.
        // Refuse only if NO rep produced a timestamp, and record the misses so the
        // control's own reliability is on the receipt rather than implied.
        for i in 0..reps {
            let t = session
                .measure_isolated_organ("noop_empty")
                .map_err(|e| e.to_string())?;
            match t.gpu_ns {
                Some(g) => {
                    eprintln!(
                        "  noop_empty rep{i} gpu={g} wait={} disp={}",
                        t.wait_ns, t.dispatches
                    );
                    gpu.push(g);
                    wait.push(t.wait_ns);
                }
                None => {
                    no_timestamp += 1;
                    eprintln!(
                        "  noop_empty rep{i} gpu=<below timestamp resolution> wait={}",
                        t.wait_ns
                    );
                    wait.push(t.wait_ns);
                }
            }
            disp = t.dispatches;
        }
        if gpu.is_empty() {
            return Err(
                "noop_empty: no rep produced a GPU timestamp; the floor control \
                 measured nothing at all"
                    .to_string(),
            );
        }
        json!({
            "organ": "noop_empty",
            "role": "no_op_control",
            "gpu_ns_reps": gpu,
            "gpu_ns_min": gpu.iter().copied().min(),
            "gpu_ns_median": median_u64(gpu.clone()),
            "gpu_ns_max": gpu.iter().copied().max(),
            "wait_ns_median": median_u64(wait),
            "dispatches": disp,
            "n_reps": reps,
            "n_reps_with_timestamp": gpu.len(),
            "n_reps_below_timestamp_resolution": no_timestamp,
            "dense_w_materialized": 0,
        })
    };
    for organ in ORGANS {
        let mut gpu = Vec::new();
        let mut wait = Vec::new();
        let mut disp = 0u64;
        eprintln!("isolated organ {organ} reps={reps}");
        for i in 0..reps {
            let t = session
                .measure_isolated_organ(organ)
                .map_err(|e| e.to_string())?;
            let g = t
                .gpu_ns
                .ok_or_else(|| format!("{organ}: driver did not expose GPUEndTime-GPUStartTime"))?;
            eprintln!(
                "  {organ} rep{i} gpu={g} wait={} disp={}",
                t.wait_ns, t.dispatches
            );
            gpu.push(g);
            wait.push(t.wait_ns);
            disp = t.dispatches;
        }
        rows.push(json!({
            "organ": organ,
            "role": "organ",
            "gpu_ns_reps": gpu,
            "gpu_ns_min": gpu.iter().copied().min(),
            "gpu_ns_median": median_u64(gpu.clone()),
            "gpu_ns_max": gpu.iter().copied().max(),
            "wait_ns_median": median_u64(wait),
            "dispatches": disp,
            "n_reps": reps,
            "gpu_timestamp_authority":
                "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy",
            "dense_w_materialized": 0,
        }));
    }
    Ok(json!({
        "kind": "MEASURED",
        "reps": reps,
        "organs": rows,
        "noop_empty": noop,
        "note": "One CB per organ, all layers of that organ. Production is 1 CB; scale isolated medians onto production GPU ns.",
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
        "qwen38 organ bandwidth: open {} max_seq={}",
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
    session.set_fuse_ba_delta(false, false);

    eprintln!("parity: ba_delta layer 0 (good kernel)");
    let good_parity = session
        .measure_ba_delta_fusion_parity(0, false)
        .unwrap_or_else(|e| fail(e));
    eprintln!(
        "  unfused={} fused={} max_abs rec_out={}",
        good_parity.unfused_dispatches,
        good_parity.fused_pair_dispatches,
        good_parity.max_abs_diff_act
    );

    eprintln!("parity: BAD identity decay/beta");
    let bad_parity = session
        .measure_ba_delta_fusion_parity(0, true)
        .unwrap_or_else(|e| fail(e));
    eprintln!("  bad max_abs rec_out={}", bad_parity.max_abs_diff_act);

    eprintln!("isolated organs (parent 756 graph)");
    let isolated = measure_organs(&session, args.reps).unwrap_or_else(|e| fail(e));

    let probe_token = prompt_ids[0];
    let mut dispatch_probes = Vec::new();

    session.set_fuse_add_rmsnorm(false, false);
    session.set_fuse_ba_delta(false, false);
    eprintln!("dispatch probe: parent 756");
    dispatch_probes.push(json!({
        "id": "parent_756",
        "role": "production_graph",
        "probe": probe_dispatches(&mut session, probe_token).unwrap_or_else(|e| fail(e)),
    }));

    session.set_fuse_add_rmsnorm(true, false);
    session.set_fuse_ba_delta(false, false);
    eprintln!("dispatch probe: add_rmsnorm 628 (noop control for <628 cut)");
    dispatch_probes.push(json!({
        "id": "add_rmsnorm_628",
        "role": "noop_control",
        "probe": probe_dispatches(&mut session, probe_token).unwrap_or_else(|e| fail(e)),
    }));

    session.set_fuse_add_rmsnorm(true, false);
    session.set_fuse_ba_delta(true, false);
    eprintln!("dispatch probe: ba_delta 580");
    dispatch_probes.push(json!({
        "id": "ba_delta_580",
        "role": "candidate",
        "probe": probe_dispatches(&mut session, probe_token).unwrap_or_else(|e| fail(e)),
    }));

    session.set_fuse_add_rmsnorm(true, false);
    session.set_fuse_ba_delta(true, true);
    eprintln!("dispatch probe: BAD identity");
    dispatch_probes.push(json!({
        "id": "ba_delta_bad",
        "role": "bad_control",
        "probe": probe_dispatches(&mut session, probe_token).unwrap_or_else(|e| fail(e)),
    }));

    let mut decode_arms = json!({});
    if !args.skip_decode {
        session.set_fuse_add_rmsnorm(false, false);
        session.set_fuse_ba_delta(false, false);
        eprintln!("decode PRODUCTION parent-756 reps={}", args.reps);
        decode_arms["parent_756"] = generate_arm(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
            args.reps,
        )
        .unwrap_or_else(|e| fail(e));

        session.set_fuse_add_rmsnorm(true, false);
        session.set_fuse_ba_delta(false, false);
        eprintln!("decode NOOP add_rmsnorm-628 reps={}", args.reps);
        decode_arms["add_rmsnorm_628"] = generate_arm(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
            args.reps,
        )
        .unwrap_or_else(|e| fail(e));

        session.set_fuse_add_rmsnorm(true, false);
        session.set_fuse_ba_delta(true, false);
        eprintln!("decode CANDIDATE ba_delta-580 reps={}", args.reps);
        decode_arms["ba_delta_580"] = generate_arm(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
            args.reps,
        )
        .unwrap_or_else(|e| fail(e));

        session.set_fuse_add_rmsnorm(true, false);
        session.set_fuse_ba_delta(true, true);
        eprintln!("decode BAD identity reps=1");
        decode_arms["ba_delta_bad"] = generate_arm(
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
    let baseline_n =
        qwen38_fused_dispatches_per_token_ex(Qwen38MlpFusion::GateUpSwiglu, true, true, true);
    let candidate_n = qwen38_fused_dispatches_per_token_full(
        Qwen38MlpFusion::GateUpSwiglu,
        true,
        true,
        true,
        true,
    );
    let body = json!({
        "schema": "hawking.headless.organ_bandwidth.raw.v1",
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
            "parent_756": parent_n,
            "baseline_628": baseline_n,
            "candidate": candidate_n,
            "saved_below_628": QWEN38_BA_DELTA_SAVED_PER_TOKEN,
            "command_buffers": 1,
        },
        "kernels": [QWEN38_BA_DELTA_KERNEL, QWEN38_BA_DELTA_BAD_KERNEL],
        "parity": {
            "ba_delta": parity_json(&good_parity),
            "ba_delta_identity": parity_json(&bad_parity),
        },
        "isolated_organs": isolated,
        "dispatch_probes": dispatch_probes,
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

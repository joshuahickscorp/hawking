//! N029 — instrument the production token loop's GPU idle intervals.
//!
//! Production stays one command buffer. GPUStart/GPUEnd of that CB name the
//! inter-token GPU gap. Host Instants classify the work that occupies the
//! GPU-idle window. Intra-CB bubbles stay unstamped: atDispatchBoundary is
//! false and ComputePassDescriptor samples changed greedy token ids.
//!
//! Arms:
//!   noop   — one encoder per dispatch (production)
//!   serial — one serial encoder for the token (attack on command construction)
//!   split  — HAWKING_TCB_TRACE=gpu, one CB per dispatch (deliberately-bad sync)
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core --example ascension_qwen38_idle_gap
//! ./tools/gpu_lane_lock.sh n029-idlegap \
//!   workspace/ops/build/rust/release-fast/examples/ascension_qwen38_idle_gap \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A \
//!   --tokenizer ~/models/qwen3.8-27b-abliterated-bf16/tokenizer.json \
//!   --reps 7 --out receipts/headless/_GPU_IDLE_GAP_raw.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

fn usage() -> &'static str {
    "usage: ascension_qwen38_idle_gap --artifact-root DIR --tokenizer PATH \
        [--prompt TEXT] [--raw-prompt] [--max-new-tokens N] [--max-seq-len N] \
        [--reps N] [--bad-reps N] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("ascension_qwen38_idle_gap: {message}");
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
    bad_reps: usize,
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
    let mut bad_reps = 1usize;
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
            "--bad-reps" => {
                bad_reps = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--bad-reps"));
            }
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
        bad_reps: bad_reps.max(1),
        out,
    }
}

fn median_u64(values: &[u64]) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut s = values.to_vec();
    s.sort_unstable();
    Some(s[s.len() / 2])
}

#[cfg(not(target_os = "macos"))]
fn main() {
    fail("qwen38 idle-gap is Metal-only");
}

#[cfg(target_os = "macos")]
fn main() {
    use hawking_core::model::qwen38_hybrid_decode::{
        generate_greedy, load_qwen38_tokenizer, qwen38_fused_dispatches_per_token_full,
        render_qwen38_user_chat, Qwen38HybridDecodeSession, Qwen38MlpFusion,
    };
    use std::time::Instant;

    let args = parse_args();
    std::env::remove_var("HAWKING_TCB_TRACE");
    std::env::remove_var("HAWKING_TRACE_DISPATCH");
    std::env::remove_var("HAWKING_COST_LEDGER");
    std::env::remove_var("HAWKING_QWEN38_FUSE_MLP");
    std::env::remove_var("HAWKING_QWEN38_FUSE_GQA_QKV");
    std::env::remove_var("HAWKING_QWEN38_FUSE_DN_INPROJ");
    std::env::remove_var("HAWKING_QWEN38_FUSE_ADD_RMSNORM");
    std::env::remove_var("HAWKING_QWEN38_FUSE_BA_DELTA");
    std::env::remove_var("HAWKING_QWEN38_CONCURRENT");

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
        "qwen38 idle-gap: open {} max_seq={} prompt_tokens={}",
        args.artifact_root.display(),
        args.max_seq_len,
        prompt_ids.len()
    );
    let open_started = Instant::now();
    let mut session = Qwen38HybridDecodeSession::open(&args.artifact_root, args.max_seq_len)
        .unwrap_or_else(|e| fail(e));
    let session_open_s = open_started.elapsed().as_secs_f64();
    eprintln!("session open {session_open_s:.3}s");

    session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);
    session.set_fuse_add_rmsnorm(true, false);
    session.set_fuse_ba_delta(true, false);
    session.set_serial_token_encoder(false);
    session.concurrent_independent = false;
    let theoretical = qwen38_fused_dispatches_per_token_full(
        Qwen38MlpFusion::GateUpSwiglu,
        true,
        true,
        true,
        true,
    );
    eprintln!(
        "580-graph theoretical_dispatches={theoretical} (session={})",
        session.theoretical_dispatches()
    );

    eprintln!("warmup generate (discarded)");
    let warmup = generate_greedy(&mut session, &prompt_ids, 4).unwrap_or_else(|e| fail(e));
    if warmup.fallbacks != 0 {
        fail("warmup fallback");
    }

    let noop = run_arm(
        &mut session,
        &tokenizer,
        &prompt_ids,
        args.max_new_tokens,
        args.reps,
        "noop",
        "noop_control",
        false,
        None,
    );
    let serial = run_arm(
        &mut session,
        &tokenizer,
        &prompt_ids,
        args.max_new_tokens,
        args.reps,
        "serial",
        "attack",
        true,
        None,
    );
    let split = run_arm(
        &mut session,
        &tokenizer,
        &prompt_ids,
        4,
        args.bad_reps,
        "split",
        "bad_control",
        false,
        Some("gpu"),
    );

    let body = json!({
        "schema": "hawking.headless.gpu_idle_gap.raw.v1",
        "gpu_timestamp_authority":
            "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy",
        "artifact_root": args.artifact_root.display().to_string(),
        "session_open_s": session_open_s,
        "theoretical_dispatches": theoretical,
        "fusion": {
            "mlp": "swiglu",
            "gqa_qkv": true,
            "dn_inproj": true,
            "add_rmsnorm": true,
            "ba_delta": true,
        },
        "dense_w_materialized": session.dense_w_materialized,
        "fallbacks": session.fallbacks,
        "prompt_len": prompt_ids.len(),
        "max_new_tokens": args.max_new_tokens,
        "reps": args.reps,
        "bad_reps": args.bad_reps,
        "warmup_new_token_ids": warmup.new_tokens(),
        "arms": {
            "noop": noop,
            "serial": serial,
            "split": split,
        },
    });

    if let Some(path) = &args.out {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(path, serde_json::to_vec_pretty(&body).expect("json"))
            .unwrap_or_else(|e| fail(e));
        eprintln!("wrote {}", path.display());
    } else {
        println!("{}", serde_json::to_string_pretty(&body).expect("json"));
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn run_arm(
    session: &mut hawking_core::model::qwen38_hybrid_decode::Qwen38HybridDecodeSession,
    tokenizer: &hawking_core::tokenizer::Tokenizer,
    prompt_ids: &[u32],
    max_new: usize,
    reps: usize,
    id: &str,
    role: &str,
    serial: bool,
    tcb_trace: Option<&str>,
) -> Value {
    use hawking_core::model::qwen38_hybrid_decode::generate_greedy_complete_wall;

    session.set_serial_token_encoder(serial);
    match tcb_trace {
        Some(mode) => std::env::set_var("HAWKING_TCB_TRACE", mode),
        None => std::env::remove_var("HAWKING_TCB_TRACE"),
    }
    eprintln!("arm {id} role={role} serial={serial} tcb_trace={tcb_trace:?} reps={reps}");

    let mut rep_docs = Vec::new();
    let mut complete_medians = Vec::new();
    let mut gpu_medians = Vec::new();
    let mut ids = Vec::new();
    for i in 0..reps {
        let result = generate_greedy_complete_wall(session, tokenizer, prompt_ids, max_new)
            .unwrap_or_else(|e| fail(e));
        if result.fallbacks != 0 {
            fail(format!("{id} rep {i} fallback"));
        }
        let new_ids = result.new_tokens().to_vec();
        let steady: Vec<&hawking_core::model::qwen38_hybrid_decode::Qwen38CompleteToken> =
            result.steady_decode_steps().collect();
        let complete: Vec<u64> = steady.iter().map(|s| s.complete_wall_ns).collect();
        let gpu: Vec<u64> = steady.iter().filter_map(|s| s.step.gpu_ns).collect();
        let med_c = median_u64(&complete);
        let med_g = median_u64(&gpu);
        eprintln!(
            "  {id} rep{} new={} complete_med={:?} gpu_med={:?} disp={:?} enc={:?}",
            i + 1,
            new_ids.len(),
            med_c,
            med_g,
            steady.first().map(|s| s.step.dispatches),
            steady.first().map(|s| s.step.encoder_count),
        );
        if let Some(v) = med_c {
            complete_medians.push(v);
        }
        if let Some(v) = med_g {
            gpu_medians.push(v);
        }
        ids.push(new_ids);
        rep_docs.push(arm_rep_json(&result, &steady));
    }
    std::env::remove_var("HAWKING_TCB_TRACE");
    session.set_serial_token_encoder(false);
    json!({
        "id": id,
        "role": role,
        "serial_token_encoder": serial,
        "tcb_trace": tcb_trace,
        "reps": reps,
        "new_token_ids": ids.first(),
        "new_token_ids_all_reps": ids,
        "token_ids_stable_across_reps": ids.windows(2).all(|w| w[0] == w[1]),
        "complete_wall_ns_rep_medians": complete_medians,
        "complete_wall_ns_min": complete_medians.iter().copied().min(),
        "complete_wall_ns_median": median_u64(&complete_medians),
        "complete_wall_ns_max": complete_medians.iter().copied().max(),
        "gpu_ns_rep_medians": gpu_medians,
        "gpu_ns_min": gpu_medians.iter().copied().min(),
        "gpu_ns_median": median_u64(&gpu_medians),
        "gpu_ns_max": gpu_medians.iter().copied().max(),
        "dense_w_materialized": session.dense_w_materialized,
        "rep_docs": rep_docs,
    })
}

#[cfg(target_os = "macos")]
fn arm_rep_json(
    result: &hawking_core::model::qwen38_hybrid_decode::Qwen38CompleteWallResult,
    steady: &[&hawking_core::model::qwen38_hybrid_decode::Qwen38CompleteToken],
) -> Value {
    let mut prev_end_s: Option<f64> = None;
    let mut tokens = Vec::new();
    for step in result.steps.iter() {
        let inter = match (prev_end_s, step.step.gpu_start_s) {
            (Some(end), Some(start)) if start > end => {
                Some(((start - end) * 1_000_000_000.0) as u64)
            }
            _ => None,
        };
        if let Some(end) = step.step.gpu_end_s {
            prev_end_s = Some(end);
        }
        tokens.push(token_json(step, inter));
    }
    json!({
        "fallbacks": result.fallbacks,
        "dense_w_materialized": result.dense_w_materialized,
        "new_token_ids": result.new_tokens(),
        "n_steps": result.steps.len(),
        "n_steady": steady.len(),
        "reset_ns": result.reset_ns,
        "tokens": tokens,
    })
}

#[cfg(target_os = "macos")]
fn token_json(
    tok: &hawking_core::model::qwen38_hybrid_decode::Qwen38CompleteToken,
    inter_token_gpu_idle_ns: Option<u64>,
) -> Value {
    let wait_minus_gpu = tok
        .step
        .wait_minus_gpu_ns()
        .map(|v| v.max(0) as u64)
        .unwrap_or(0);
    let residual = tok.residual_ns().max(0) as u64;
    let intervals = json!([
        {
            "cause": "allocation",
            "ns": tok.step.allocation_ns,
            "where": "TokenCommandBuffer::new",
            "kind": "MEASURED",
        },
        {
            "cause": "command construction",
            "ns": tok.step.encode_ns.saturating_add(tok.step.submit_ns),
            "where": "encode_embed+layers+terminal + MTLCommandBuffer.commit",
            "kind": "MEASURED",
            "encode_ns": tok.step.encode_ns,
            "submit_ns": tok.step.submit_ns,
        },
        {
            "cause": "sync",
            "ns": wait_minus_gpu.saturating_add(tok.step.commit_epilogue_ns),
            "where": "wait_until_completed − GPUEnd+GPUStart, plus timestamp/status epilogue",
            "kind": "MEASURED",
            "wait_minus_gpu_ns": wait_minus_gpu,
            "commit_epilogue_ns": tok.step.commit_epilogue_ns,
        },
        {
            "cause": "sampler",
            "ns": tok.step.sample_readback_ns,
            "where": "unified-memory load of device argmax u32",
            "kind": "MEASURED",
        },
        {
            "cause": "state bookkeeping",
            "ns": tok.step.state_update_ns
                .saturating_add(tok.tokenizer_decode_ns)
                .saturating_add(tok.bookkeeping_ns),
            "where": "position += 1, tokenizer.decode of sampled id, tokens.push",
            "kind": "MEASURED",
            "state_update_ns": tok.step.state_update_ns,
            "tokenizer_decode_ns": tok.tokenizer_decode_ns,
            "bookkeeping_ns": tok.bookkeeping_ns,
        },
        {
            "cause": "CPU sched",
            "ns": residual,
            "where": "unnamed Instant inter-phase gap on the host",
            "kind": "MEASURED",
        },
        {
            "cause": "Python",
            "ns": 0,
            "where": "native Rust token loop; harness is a subprocess around the binary",
            "kind": "MEASURED",
        },
        {
            "cause": "runtime lock",
            "ns": 0,
            "where": "no in-process mutex on the token; gpu_lane_lock.sh is process-level",
            "kind": "MEASURED",
        },
        {
            "cause": "serialization",
            "ns": 0,
            "where": "no host-device activation memcpy; unified memory. Metal set_bytes of scalars is folded into command construction, not double-counted",
            "kind": "MEASURED",
        },
        {
            "cause": "dependency",
            "ns": null,
            "where": "intra-CB producer→consumer bubbles inside GPUEnd−GPUStart",
            "kind": "ABSENT",
            "absent_reason": "Production is one mixed command buffer. Intra-CB idle sits inside GPUEndTime−GPUStartTime and cannot be split: supportsCounterSampling.atDispatchBoundary=false. TCB production identity refuses ComputePassDescriptor boundary samples because they changed greedy token ids.",
        }
    ]);
    json!({
        "role": tok.role,
        "step_index": tok.step_index,
        "token_in": tok.token_in,
        "token_out": tok.token_out,
        "complete_wall_ns": tok.complete_wall_ns,
        "gpu_ns": tok.step.gpu_ns,
        "gpu_start_s": tok.step.gpu_start_s,
        "gpu_end_s": tok.step.gpu_end_s,
        "gpu_start_ns": tok.step.gpu_start_ns,
        "gpu_end_ns": tok.step.gpu_end_ns,
        "inter_token_gpu_idle_ns": inter_token_gpu_idle_ns,
        "dispatches": tok.step.dispatches,
        "encoder_count": tok.step.encoder_count,
        "command_buffers": tok.step.command_buffers,
        "intervals": intervals,
    })
}

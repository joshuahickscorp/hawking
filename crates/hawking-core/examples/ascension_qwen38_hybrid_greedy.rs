//! Native Qwen3.8 greedy generate on the language-only Q4 catalog.
//!
//! ```text
//! ./tools/gpu_lane_lock.sh qwen38-native-bringup \
//!   workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy \
//!   --artifact-root .../uniform-q4-v1 \
//!   --tokenizer .../bf16/tokenizer.json \
//!   --prompt "Say hi." --max-new-tokens 16 \
//!   --out receipts/ascent-2026-08-16/qwen38-native-generate.json
//!
//! ./tools/gpu_lane_lock.sh qwen38-complete-wall \
//!   .../ascension_qwen38_hybrid_greedy \
//!   --artifact-root .../uniform-q4-v1 \
//!   --tokenizer .../bf16/tokenizer.json \
//!   --complete-wall --pairs 3 --max-new-tokens 32 \
//!   --out receipts/ascent-2026-08-16/QWEN38_COMPLETE_TOKEN_WALL.json
//! ```

use hawking_core::model::qwen38_hybrid_decode::{
    load_qwen38_tokenizer, render_qwen38_user_chat, Qwen38CompleteToken, Qwen38CompleteWallResult,
    Qwen38GenerateResult,
};
use hawking_core::tokenizer::Tokenizer;
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

#[cfg(target_os = "macos")]
use hawking_core::model::qwen38_hybrid_decode::{
    generate_greedy, generate_greedy_complete_wall, Qwen38HybridDecodeSession,
};

fn usage() -> &'static str {
    "usage: ascension_qwen38_hybrid_greedy --artifact-root DIR --tokenizer PATH \
        [--prompt TEXT] [--prompts-file PATH] [--raw-prompt] [--max-new-tokens N] \
        [--max-seq-len N] [--complete-wall] [--pairs N] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("ascension_qwen38_hybrid_greedy: {message}");
    process::exit(2);
}

struct Args {
    artifact_root: PathBuf,
    tokenizer: PathBuf,
    prompt: String,
    prompts_file: Option<PathBuf>,
    raw_prompt: bool,
    max_new_tokens: usize,
    max_seq_len: usize,
    complete_wall: bool,
    pairs: usize,
    out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut artifact_root = None;
    let mut tokenizer = None;
    let mut prompt = "Say hi.".to_owned();
    let mut prompts_file = None;
    let mut raw_prompt = false;
    let mut max_new_tokens = 16usize;
    let mut max_new_tokens_set = false;
    let mut max_seq_len = 128usize;
    let mut complete_wall = false;
    let mut pairs = 3usize;
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
            "--prompts-file" => {
                prompts_file = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--raw-prompt" => raw_prompt = true,
            "--complete-wall" => complete_wall = true,
            "--pairs" => {
                pairs = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--pairs"));
            }
            "--max-new-tokens" => {
                max_new_tokens = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--max-new-tokens"));
                max_new_tokens_set = true;
            }
            "--max-seq-len" => {
                max_seq_len = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--max-seq-len"));
            }
            "--out" => out = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage())))),
            other => fail(format!("unknown {other}; {}", usage())),
        }
    }
    if complete_wall && !max_new_tokens_set {
        max_new_tokens = 32;
    }
    Args {
        artifact_root: artifact_root.unwrap_or_else(|| fail(usage())),
        tokenizer: tokenizer.unwrap_or_else(|| fail(usage())),
        prompt,
        prompts_file,
        raw_prompt,
        max_new_tokens,
        max_seq_len,
        complete_wall,
        pairs,
        out,
    }
}

fn median_u64(values: &[u64]) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    Some(sorted[sorted.len() / 2])
}

fn mean_u64(values: &[u64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64)
}

fn spread_u64(values: &[u64]) -> Value {
    json!({
        "n": values.len(),
        "min": values.iter().copied().min(),
        "median": median_u64(values),
        "mean": mean_u64(values),
        "max": values.iter().copied().max(),
        "all": values,
    })
}

fn collect_field(steps: &[&Qwen38CompleteToken], f: impl Fn(&Qwen38CompleteToken) -> u64) -> Vec<u64> {
    steps.iter().map(|s| f(s)).collect()
}

fn summarize_complete(result: &Qwen38CompleteWallResult, tokenizer: &Tokenizer) -> Value {
    let first = result.first_step();
    let steady: Vec<&Qwen38CompleteToken> = result.steady_decode_steps().collect();
    let prefill: Vec<&Qwen38CompleteToken> = result.prefill_steps().collect();
    let complete = collect_field(&steady, |s| s.complete_wall_ns);
    let gpu: Vec<u64> = steady.iter().filter_map(|s| s.step.gpu_ns).collect();
    let wait = collect_field(&steady, |s| s.step.wait_ns);
    let encode = collect_field(&steady, |s| s.step.encode_ns);
    let submit = collect_field(&steady, |s| s.step.submit_ns);
    let epilogue = collect_field(&steady, |s| s.step.commit_epilogue_ns);
    let readback = collect_field(&steady, |s| s.step.sample_readback_ns);
    let state = collect_field(&steady, |s| s.step.state_update_ns);
    let tokenizer_decode = collect_field(&steady, |s| s.tokenizer_decode_ns);
    let bookkeeping = collect_field(&steady, |s| s.bookkeeping_ns);
    let wait_minus_gpu: Vec<u64> = steady
        .iter()
        .filter_map(|s| s.step.wait_minus_gpu_ns().map(|v| v.max(0) as u64))
        .collect();
    let wall_minus_gpu: Vec<u64> = steady
        .iter()
        .filter_map(|s| s.wall_minus_gpu_ns().map(|v| v.max(0) as u64))
        .collect();
    let residuals: Vec<i64> = steady.iter().map(|s| s.residual_ns()).collect();
    let step_residuals: Vec<i64> = steady.iter().map(|s| s.step.residual_ns()).collect();
    let mean_complete = mean_u64(&complete).unwrap_or(0.0);
    let mean_gpu = mean_u64(&gpu).unwrap_or(0.0);
    let named_means = json!({
        "encode_host_prepare": mean_u64(&encode),
        "submit": mean_u64(&submit),
        "wait": mean_u64(&wait),
        "gpu": mean_u64(&gpu),
        "wait_minus_gpu": mean_u64(&wait_minus_gpu),
        "commit_epilogue_gpu_timestamp_and_status": mean_u64(&epilogue),
        "sample_readback": mean_u64(&readback),
        "state_update": mean_u64(&state),
        "tokenizer_decode_new_token": mean_u64(&tokenizer_decode),
        "bookkeeping": mean_u64(&bookkeeping),
    });
    let mean_named_minus_wait_plus_wait = mean_u64(&encode).unwrap_or(0.0)
        + mean_u64(&submit).unwrap_or(0.0)
        + mean_u64(&wait).unwrap_or(0.0)
        + mean_u64(&epilogue).unwrap_or(0.0)
        + mean_u64(&readback).unwrap_or(0.0)
        + mean_u64(&state).unwrap_or(0.0)
        + mean_u64(&tokenizer_decode).unwrap_or(0.0)
        + mean_u64(&bookkeeping).unwrap_or(0.0);
    let mean_residual = mean_complete - mean_named_minus_wait_plus_wait;
    let text = result.decode_new(tokenizer).unwrap_or_else(|e| fail(e));
    json!({
        "generated_text": text,
        "new_token_ids": result.new_tokens(),
        "fallbacks": result.fallbacks,
        "prompt_len": result.prompt_len,
        "n_steps": result.steps.len(),
        "n_prefill_steps": prefill.len(),
        "n_steady_decode_steps": steady.len(),
        "reset_ns": result.reset_ns,
        "prefill_wall_ns": result.prefill_wall_ns,
        "decode_wall_ns": result.decode_wall_ns,
        "all_steps_wall_ns": result.wall_ns,
        "cold_or_first_step": first.map(|s| json!({
            "role": s.role,
            "complete_wall_ns": s.complete_wall_ns,
            "gpu_ns": s.step.gpu_ns,
            "wait_ns": s.step.wait_ns,
            "encode_ns": s.step.encode_ns,
            "dispatches": s.step.dispatches,
        })),
        "steady_decode": {
            "complete_wall_ns": spread_u64(&complete),
            "gpu_ns": spread_u64(&gpu),
            "wait_ns": spread_u64(&wait),
            "wait_minus_gpu_ns": spread_u64(&wait_minus_gpu),
            "wall_minus_gpu_ns": spread_u64(&wall_minus_gpu),
            "encode_host_prepare_ns": spread_u64(&encode),
            "submit_ns": spread_u64(&submit),
            "commit_epilogue_ns": spread_u64(&epilogue),
            "sample_readback_ns": spread_u64(&readback),
            "state_update_ns": spread_u64(&state),
            "tokenizer_decode_ns": spread_u64(&tokenizer_decode),
            "bookkeeping_ns": spread_u64(&bookkeeping),
            "dispatches": steady.first().map(|s| s.step.dispatches),
            "command_buffers": steady.first().map(|s| s.step.command_buffers),
        },
        "closure": {
            "identity": "complete_wall = encode + submit + wait + commit_epilogue + sample_readback + state_update + tokenizer_decode + bookkeeping + residual",
            "residual_named_as": "instant_inter_phase_gap",
            "per_step_complete_residual_ns": residuals,
            "per_step_inner_step_residual_ns": step_residuals,
            "max_abs_complete_residual_ns": residuals.iter().map(|v| v.unsigned_abs()).max(),
            "mean_complete_wall_ns": mean_complete,
            "mean_named_sum_ns": mean_named_minus_wait_plus_wait,
            "mean_residual_ns": mean_residual,
            "mean_gpu_ns": mean_gpu,
            "mean_wall_minus_gpu_ns": mean_complete - mean_gpu,
            "named_component_means_ns": named_means,
        },
    })
}

#[cfg(not(target_os = "macos"))]
fn main() {
    fail("qwen38 native generate is Metal-only");
}

#[cfg(target_os = "macos")]
fn write_json(path: &PathBuf, body: &Value) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(path, serde_json::to_vec_pretty(body).expect("json")).unwrap_or_else(|e| fail(e));
    eprintln!("wrote {}", path.display());
}

#[cfg(target_os = "macos")]
fn run_default(args: &Args, tokenizer: &Tokenizer, rendered: &str, prompt_ids: &[u32]) {
    let mut session = Qwen38HybridDecodeSession::open(&args.artifact_root, args.max_seq_len)
        .unwrap_or_else(|e| fail(e));
    let result: Qwen38GenerateResult =
        generate_greedy(&mut session, prompt_ids, args.max_new_tokens).unwrap_or_else(|e| fail(e));
    let text = result.decode_new(tokenizer).unwrap_or_else(|e| fail(e));
    let full = tokenizer
        .decode(&result.tokens, false)
        .unwrap_or_else(|e| fail(e));
    let gpu: Vec<u64> = result.gpu_ns.iter().copied().flatten().collect();
    let median = result.median_gpu_ns_per_token();
    println!("GENERATED_TEXT_VERBATIM: {text}");
    println!("FALLBACKS: {}", result.fallbacks);
    println!("DENSE_W_MATERIALIZED: 0");
    println!("PROMPT_LEN: {}", result.prompt_len);
    println!("NEW_TOKENS: {:?}", result.new_tokens());
    println!("generated_token_ids={:?}", result.new_tokens());
    println!("GPU_NS_PER_STEP: {gpu:?}");
    println!("WAIT_NS_PER_STEP: {:?}", result.wait_ns);
    println!("MEDIAN_GPU_NS_PER_TOKEN: {median:?}");
    println!("WALL_NS: {}", result.wall_ns);
    println!("PREFILL_WALL_NS: {}", result.prefill_wall_ns);
    println!("DECODE_WALL_NS: {}", result.decode_wall_ns);
    println!("FIRST_STEP_WALL_NS: {}", result.first_step_wall_ns);
    println!(
        "STEADY_DECODE_WALL_NS_PER_TOKEN: {:?}",
        result.steady_decode_wall_ns_per_token()
    );
    if let Some(path) = &args.out {
        write_json(
            path,
            &json!({
                "lane": "qwen38-native-bringup",
                "generated_text": text,
                "full_decode": full,
                "prompt": rendered,
                "prompt_ids": prompt_ids,
                "new_token_ids": result.new_tokens(),
                "fallbacks": result.fallbacks,
                "dense_w_materialized": 0,
                "gpu_ns_per_step": result.gpu_ns,
                "wait_ns_per_step": result.wait_ns,
                "median_gpu_ns_per_token": median,
                "wall_ns": result.wall_ns,
                "first_step_wall_ns": result.first_step_wall_ns,
                "prefill_wall_ns": result.prefill_wall_ns,
                "decode_wall_ns": result.decode_wall_ns,
                "decode_steps": result.decode_steps,
                "wall_ns_per_step": result.wall_ns_per_step,
                "timing_label": "DIRTY_ENGINEERING",
            }),
        );
    }
}

#[cfg(target_os = "macos")]
fn run_complete_wall(args: &Args, tokenizer: &Tokenizer, rendered: &str, prompt_ids: &[u32]) {
    if args.pairs == 0 {
        fail("--pairs must be >= 1");
    }
    let encode_started = Instant::now();
    // prompt_ids already encoded by caller; re-time encode for the receipt.
    let _ = tokenizer
        .encode(rendered, false)
        .unwrap_or_else(|e| fail(e));
    let prompt_encode_ns = encode_started.elapsed().as_nanos() as u64;

    let open_started = Instant::now();
    let mut session = Qwen38HybridDecodeSession::open(&args.artifact_root, args.max_seq_len)
        .unwrap_or_else(|e| fail(e));
    let session_open_ns = open_started.elapsed().as_nanos() as u64;
    eprintln!("qwen38 session open {:.3}s", session_open_ns as f64 / 1e9);

    eprintln!("qwen38 complete-wall COLD generate (discard from authority except first-step)");
    let cold = generate_greedy_complete_wall(
        &mut session,
        tokenizer,
        prompt_ids,
        args.max_new_tokens,
    )
    .unwrap_or_else(|e| fail(e));
    if cold.fallbacks != 0 {
        fail(format!("cold generate fallbacks={}", cold.fallbacks));
    }
    let cold_summary = summarize_complete(&cold, tokenizer);
    let cold_first_gpu = cold
        .first_step()
        .and_then(|s| s.step.gpu_ns)
        .unwrap_or(0);
    eprintln!(
        "qwen38 COLD first step wall={} ns gpu={} ns prefill={} ns",
        cold.first_step().map(|s| s.complete_wall_ns).unwrap_or(0),
        cold_first_gpu,
        cold.prefill_wall_ns
    );

    let mut pair_reps = Vec::new();
    let mut authority_rep_medians = Vec::new();
    let mut authority_rep_gpu_medians = Vec::new();
    let mut all_steady_walls = Vec::new();
    let mut all_steady_gpus = Vec::new();
    let mut last_ids: Option<Vec<u32>> = None;
    for pair in 0..args.pairs {
        for arm in ["A", "B"] {
            let label = format!("{arm}{}", pair + 1);
            eprintln!("qwen38 complete-wall WARM {label}");
            let result = generate_greedy_complete_wall(
                &mut session,
                tokenizer,
                prompt_ids,
                args.max_new_tokens,
            )
            .unwrap_or_else(|e| fail(e));
            if result.fallbacks != 0 {
                fail(format!("{label} fallbacks={}", result.fallbacks));
            }
            let ids = result.new_tokens().to_vec();
            if let Some(prev) = &last_ids {
                if prev != &ids {
                    fail(format!(
                        "{label} greedy ids drifted vs previous warm rep"
                    ));
                }
            }
            last_ids = Some(ids);
            let summary = summarize_complete(&result, tokenizer);
            let steady_median = summary["steady_decode"]["complete_wall_ns"]["median"]
                .as_u64()
                .unwrap_or(0);
            let gpu_median = summary["steady_decode"]["gpu_ns"]["median"].as_u64().unwrap_or(0);
            eprintln!(
                "qwen38 WARM {label} complete-wall median={} ns ({:.3} ms) gpu median={} ns decode_steps={}",
                steady_median,
                steady_median as f64 / 1e6,
                gpu_median,
                result.steady_decode_steps().count()
            );
            authority_rep_medians.push(steady_median);
            authority_rep_gpu_medians.push(gpu_median);
            for step in result.steady_decode_steps() {
                all_steady_walls.push(step.complete_wall_ns);
                if let Some(gpu) = step.step.gpu_ns {
                    all_steady_gpus.push(gpu);
                }
            }
            pair_reps.push(json!({
                "label": label,
                "pair": pair + 1,
                "arm": arm,
                "regime": "warm_in_process_after_cold_generate",
                "summary": summary,
            }));
        }
    }

    eprintln!("qwen38 complete-wall CONTROL uninstrumented generate_greedy");
    let control = generate_greedy(&mut session, prompt_ids, args.max_new_tokens)
        .unwrap_or_else(|e| fail(e));
    if control.fallbacks != 0 {
        fail(format!("control fallbacks={}", control.fallbacks));
    }
    if let Some(prev) = &last_ids {
        if prev != control.new_tokens() {
            fail("control greedy ids drifted vs complete-wall reps");
        }
    }

    let authority_median = median_u64(&authority_rep_medians).unwrap_or(0);
    let authority_gpu = median_u64(&authority_rep_gpu_medians).unwrap_or(0);
    let pooled_wall = median_u64(&all_steady_walls).unwrap_or(0);
    let pooled_gpu = median_u64(&all_steady_gpus).unwrap_or(0);
    let control_decode_ns_per_token = control.steady_decode_wall_ns_per_token().unwrap_or(0);

    let first_warm = pair_reps
        .first()
        .cloned()
        .unwrap_or_else(|| json!({}));
    let warm_components = first_warm
        .get("summary")
        .and_then(|s| s.get("closure"))
        .cloned()
        .unwrap_or_else(|| json!({}));

    let wall_minus_gpu = authority_median as i64 - authority_gpu as i64;
    let gpu_fraction = if authority_median > 0 {
        authority_gpu as f64 / authority_median as f64
    } else {
        0.0
    };
    let recorded_gpu_ms = 33.536999;
    let current_bpw = 4.252735126866492;
    let target_bpw = 2.0;
    let projected_from_recorded_gpu_ms = recorded_gpu_ms * (target_bpw / current_bpw);
    let authority_wall_ms = authority_median as f64 / 1e6;
    let projected_from_measured_wall_ms = authority_wall_ms * (target_bpw / current_bpw);
    let projected_from_measured_gpu_ms = (authority_gpu as f64 / 1e6) * (target_bpw / current_bpw);

    let body = json!({
        "schema": "hawking.ascent.qwen38_complete_token_wall.v1",
        "date": "2026-08-16",
        "lane": "qwen38-complete-wall",
        "timing_label": "DIRTY_ENGINEERING",
        "timing_label_reason": "GPU lock held for the series; other CPU/memory lanes may still be live. Not offered as CLEAN_CANDIDATE or BASE_TRUE_TPS.",
        "vehicle": {
            "artifact_root": args.artifact_root,
            "role": "PROFILING_ORACLE_ONLY",
            "complete_physical_bpw": current_bpw,
            "not_optimized": true,
        },
        "identity": {
            "model": "PocketAiHub/Qwen3.8-27B-Abliterated-MLX",
            "base_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "prompt": rendered,
            "prompt_ids": prompt_ids,
            "prompt_len": prompt_ids.len(),
            "max_new_tokens": args.max_new_tokens,
            "greedy_new_token_ids": last_ids,
            "fallbacks": 0,
            "kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
            "deltanet_vi_parallel": true,
            "concurrent_independent": false,
        },
        "definition": {
            "complete_token": "one native decode step plus the per-token host work that recurs: host encode/prepare, submit, wait, GPU timestamp/status epilogue, sample readback, position/state update, tokenizer decode of the new id, bookkeeping. Sampling is device argmax inside the same CB.",
            "prefill_exclusion": "Q80 mixed generate_mixed_greedy definition, applied verbatim. Prefill = teacher-forced walk of every prompt token. The last prefill step emits new-token[0] and is first_token_latency. Steady-state denominator is new-tokens[1..] only (role==decode). Prompt encode, chat render, session.open, and session.reset are request/process-level and are not divided into the per-token wall.",
            "cold_first_step": "The first step of the first generate in a fresh process is graph-cold (shader/pipeline first-touch). Historically 107.97 ms GPU. It is a prefill step. It is reported and is never averaged into the steady median.",
            "authority_set": "3 alternating A/B pairs = 6 warm in-process generates after one discarded cold generate. Headline is the median of the 6 per-rep medians of per-step complete_wall_ns on decode steps.",
            "gpu_definition": "MTLCommandBuffer.GPUEndTime - GPUStartTime after wait; never a CPU-wait proxy.",
            "not_base_true_tps": true,
        },
        "request_level_excluded_from_per_token": {
            "prompt_encode_ns": prompt_encode_ns,
            "session_open_ns": session_open_ns,
            "chat_template": "render_qwen38_user_chat, once per request",
        },
        "cold_generate": cold_summary,
        "warm_reps": pair_reps,
        "control_uninstrumented_generate_greedy": {
            "purpose": "show Instant decomposition does not dominate: compare decode-only wall (no per-token tokenizer) to complete-wall authority",
            "first_step_wall_ns": control.first_step_wall_ns,
            "prefill_wall_ns": control.prefill_wall_ns,
            "decode_wall_ns": control.decode_wall_ns,
            "decode_steps": control.decode_steps,
            "decode_wall_ns_per_token": control_decode_ns_per_token,
            "median_gpu_ns_including_prefill_steps": control.median_gpu_ns_per_token(),
            "new_token_ids": control.new_tokens(),
        },
        "authority": {
            "set": "warm A1 B1 A2 B2 A3 B3",
            "discarded": "cold first generate in this process (graph-cold first step + page-warm open already paid)",
            "rep_median_complete_wall_ns": authority_rep_medians,
            "rep_median_gpu_ns": authority_rep_gpu_medians,
            "spread_rep_median_complete_wall_ns": spread_u64(&authority_rep_medians),
            "spread_rep_median_gpu_ns": spread_u64(&authority_rep_gpu_medians),
            "headline_complete_wall_ns_per_token": authority_median,
            "headline_complete_wall_ms_per_token": authority_wall_ms,
            "headline_complete_tps": if authority_median > 0 { 1.0e9 / authority_median as f64 } else { 0.0 },
            "headline_gpu_ns_per_token": authority_gpu,
            "headline_gpu_ms_per_token": authority_gpu as f64 / 1e6,
            "pooled_steady_complete_wall_ns": spread_u64(&all_steady_walls),
            "pooled_steady_gpu_ns": spread_u64(&all_steady_gpus),
            "pooled_median_complete_wall_ns": pooled_wall,
            "pooled_median_gpu_ns": pooled_gpu,
        },
        "wall_minus_gpu": {
            "headline_ns": wall_minus_gpu,
            "headline_ms": wall_minus_gpu as f64 / 1e6,
            "gpu_as_fraction_of_wall": gpu_fraction,
            "wait_is_not_wall": true,
            "components_from_first_warm_rep_means": warm_components,
        },
        "gpu_proxy_verdict": {
            "recorded_gpu_ms_g015": recorded_gpu_ms,
            "recorded_source": "G015_NATIVE_LEG_VERIFY_ON_MAIN.json median of 25 steps after dropping only the first step (those 25 still include leftover prefill steps 2..prompt_len)",
            "measured_steady_decode_gpu_ms": authority_gpu as f64 / 1e6,
            "measured_steady_decode_wall_ms": authority_wall_ms,
            "gpu_minus_wall_ms": (authority_gpu as f64 - authority_median as f64) / 1e6,
            "is_honest_proxy": gpu_fraction >= 0.97 && (wall_minus_gpu as f64).abs() < 1.5e6,
            "arithmetic": {
                "projection_uses": "ms_at_target = measured_ms * (2.0 / 4.252735126866492)",
                "from_recorded_33_537_gpu_ms": projected_from_recorded_gpu_ms,
                "from_recorded_33_537_gpu_tps": 1000.0 / projected_from_recorded_gpu_ms,
                "from_measured_gpu_ms": projected_from_measured_gpu_ms,
                "from_measured_gpu_tps": if projected_from_measured_gpu_ms > 0.0 { 1000.0 / projected_from_measured_gpu_ms } else { 0.0 },
                "from_measured_complete_wall_ms": projected_from_measured_wall_ms,
                "from_measured_complete_wall_tps": if projected_from_measured_wall_ms > 0.0 { 1000.0 / projected_from_measured_wall_ms } else { 0.0 },
                "optimism_of_gpu_projection_ms": projected_from_recorded_gpu_ms - projected_from_measured_wall_ms,
                "optimism_of_gpu_projection_tps": if projected_from_measured_wall_ms > 0.0 {
                    (1000.0 / projected_from_recorded_gpu_ms) - (1000.0 / projected_from_measured_wall_ms)
                } else { 0.0 },
            },
        },
    });

    println!(
        "COMPLETE_WALL_NS_PER_TOKEN: {}",
        authority_median
    );
    println!(
        "COMPLETE_WALL_MS_PER_TOKEN: {:.6}",
        authority_wall_ms
    );
    println!(
        "COMPLETE_WALL_TPS: {:.4}",
        if authority_median > 0 {
            1.0e9 / authority_median as f64
        } else {
            0.0
        }
    );
    println!("GPU_NS_PER_TOKEN: {authority_gpu}");
    println!(
        "WALL_MINUS_GPU_NS: {wall_minus_gpu}"
    );
    println!("REP_MEDIANS_NS: {authority_rep_medians:?}");
    println!(
        "CONTROL_DECODE_WALL_NS_PER_TOKEN: {control_decode_ns_per_token}"
    );
    println!(
        "GENERATED_TEXT_VERBATIM: {}",
        control.decode_new(tokenizer).unwrap_or_else(|e| fail(e))
    );
    println!("FALLBACKS: 0");
    if let Some(path) = &args.out {
        write_json(path, &body);
    }
}

#[cfg(target_os = "macos")]
fn load_prompt_lines(path: &PathBuf) -> Vec<String> {
    let raw = fs::read_to_string(path).unwrap_or_else(|e| fail(e));
    let lines: Vec<String> = raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect();
    if lines.len() < 2 {
        fail("prompts-file needs at least two non-empty lines");
    }
    lines
}

#[cfg(target_os = "macos")]
fn run_prompts_file(args: &Args, tokenizer: &Tokenizer, prompts: &[String]) {
    let open_started = Instant::now();
    let mut session = Qwen38HybridDecodeSession::open(&args.artifact_root, args.max_seq_len)
        .unwrap_or_else(|e| fail(e));
    eprintln!(
        "qwen38 session open {:.3}s for {} prompts",
        open_started.elapsed().as_secs_f64(),
        prompts.len()
    );
    let mut results = Vec::new();
    for (i, prompt) in prompts.iter().enumerate() {
        let rendered = if args.raw_prompt {
            prompt.clone()
        } else {
            render_qwen38_user_chat(prompt)
        };
        let prompt_ids = tokenizer
            .encode(&rendered, false)
            .unwrap_or_else(|e| fail(e));
        eprintln!(
            "qwen38 prompt {}/{} tokens={} text={prompt:?}",
            i + 1,
            prompts.len(),
            prompt_ids.len()
        );
        let result: Qwen38GenerateResult =
            generate_greedy(&mut session, &prompt_ids, args.max_new_tokens)
                .unwrap_or_else(|e| fail(e));
        let text = result.decode_new(tokenizer).unwrap_or_else(|e| fail(e));
        let new_tokens = result.new_tokens().to_vec();
        println!("PROMPT: {prompt}");
        println!("GENERATED_TEXT_VERBATIM: {text}");
        println!("FALLBACKS: {}", result.fallbacks);
        println!("DENSE_W_MATERIALIZED: 0");
        println!("PROMPT_LEN: {}", result.prompt_len);
        println!("NEW_TOKENS: {new_tokens:?}");
        println!("generated_token_ids={new_tokens:?}");
        println!("WALL_NS: {}", result.wall_ns);
        results.push(json!({
            "prompt": prompt,
            "rendered": rendered,
            "prompt_ids": prompt_ids,
            "generated_text": text,
            "new_token_ids": new_tokens,
            "fallbacks": result.fallbacks,
            "dense_w_materialized": 0,
            "prompt_len": result.prompt_len,
            "wall_ns": result.wall_ns,
            "prefill_wall_ns": result.prefill_wall_ns,
            "decode_wall_ns": result.decode_wall_ns,
        }));
    }
    if let Some(path) = &args.out {
        write_json(
            path,
            &json!({
                "lane": "qwen38-coherence-generate",
                "artifact_root": args.artifact_root,
                "max_new_tokens": args.max_new_tokens,
                "fallbacks_total": results.iter().map(|r| r["fallbacks"].as_u64().unwrap_or(0)).sum::<u64>(),
                "dense_w_materialized_total": 0,
                "prompts": results,
            }),
        );
    }
}

#[cfg(target_os = "macos")]
fn main() {
    let args = parse_args();
    let tokenizer = load_qwen38_tokenizer(&args.tokenizer).unwrap_or_else(|e| fail(e));
    if let Some(path) = &args.prompts_file {
        if args.complete_wall {
            fail("--prompts-file cannot be combined with --complete-wall");
        }
        let prompts = load_prompt_lines(path);
        run_prompts_file(&args, &tokenizer, &prompts);
        return;
    }
    let rendered = if args.raw_prompt {
        args.prompt.clone()
    } else {
        render_qwen38_user_chat(&args.prompt)
    };
    let prompt_ids = tokenizer
        .encode(&rendered, false)
        .unwrap_or_else(|e| fail(e));
    eprintln!(
        "qwen38 prompt tokens={} text={rendered:?}",
        prompt_ids.len()
    );
    if args.complete_wall {
        run_complete_wall(&args, &tokenizer, &rendered, &prompt_ids);
    } else {
        run_default(&args, &tokenizer, &rendered, &prompt_ids);
    }
}

//! FULL-MLP native 2-bit (q2f g64) path: packed codes, fused SwiGLU, zero dense W.
//!
//! Opens one mixed catalog (does not load a second 27B). Measures a complete
//! token (>=7 reps, min/median/max) on three arms:
//!   production  fused SwiGLU + specialized geo_tpr64
//!   no-op       unfused + specialized geo_tpr64 (same kernel family)
//!   bad         unfused + serial one-thread-per-row (deliberately worse)
//! then greedy-generates with fused SwiGLU. Packed codes stay packed.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core --example native_2bit_mlp
//! ./tools/gpu_lane_lock.sh n021-native2bit \
//!   workspace/ops/build/rust/release-fast/examples/native_2bit_mlp \
//!   --artifact-root artifacts/qwen38-q2f-g64/mix_all_mlp_q2f_g64 \
//!   --tokenizer ~/models/qwen3.8-27b-abliterated-bf16/tokenizer.json \
//!   --out receipts/headless/_NATIVE_2BIT_MLP_raw.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

#[cfg(target_os = "macos")]
use hawking_core::model::qwen38_hybrid_decode::{
    generate_greedy, load_qwen38_tokenizer, qwen38_fused_dispatches_per_token,
    render_qwen38_user_chat, Qwen38FusionParity, Qwen38GenerateResult,
    Qwen38HybridDecodeSession, Qwen38MlpFusion,
};
use hawking_core::model::qwen38_token_ns_ledger::production_dispatches_per_token;

fn usage() -> &'static str {
    "usage: native_2bit_mlp --artifact-root DIR --tokenizer PATH \
        [--prompt TEXT] [--raw-prompt] [--max-new-tokens N] [--max-seq-len N] \
        [--reps N] [--warmup N] [--skip-generate] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("native_2bit_mlp: {message}");
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
    skip_generate: bool,
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
    let mut warmup = 2usize;
    let mut skip_generate = false;
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
            "--skip-generate" => skip_generate = true,
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
        reps: reps.max(7),
        warmup,
        skip_generate,
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

fn spread_u64(values: &[u64]) -> Value {
    if values.is_empty() {
        return json!({ "n": 0, "min": null, "median": null, "max": null, "all": [] });
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    json!({
        "n": sorted.len(),
        "min": sorted.first().copied(),
        "median": sorted[sorted.len() / 2],
        "max": sorted.last().copied(),
        "all": values,
    })
}

fn ranges_overlap(a: &[u64], b: &[u64]) -> bool {
    let (Some(&amin), Some(&amax), Some(&bmin), Some(&bmax)) = (
        a.iter().min(),
        a.iter().max(),
        b.iter().min(),
        b.iter().max(),
    ) else {
        return false;
    };
    amin <= bmax && bmin <= amax
}

fn max_abs(a: f32, b: f32, c: f32) -> f32 {
    a.max(b).max(c)
}

#[cfg(not(target_os = "macos"))]
fn main() {
    fail("native 2-bit MLP is Metal-only");
}

#[cfg(target_os = "macos")]
fn parity_json(p: &Qwen38FusionParity) -> Value {
    json!({
        "fusion": p.fusion,
        "layer": p.layer,
        "unfused_dispatches": p.unfused_dispatches,
        "fused_pair_dispatches": p.fused_pair_dispatches,
        "fused_swiglu_dispatches": p.fused_swiglu_dispatches,
        "unfused_gpu_ns": p.unfused_gpu_ns,
        "fused_pair_gpu_ns": p.fused_pair_gpu_ns,
        "fused_swiglu_gpu_ns": p.fused_swiglu_gpu_ns,
        "max_abs_diff_gate": p.max_abs_diff_gate,
        "max_abs_diff_up": p.max_abs_diff_up,
        "max_abs_diff_act": p.max_abs_diff_act,
        "max_abs_diff": max_abs(p.max_abs_diff_gate, p.max_abs_diff_up, p.max_abs_diff_act),
        "dense_w_materialized": p.dense_w_materialized,
        "tolerance": 1e-2,
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
fn generate_once(
    session: &mut Qwen38HybridDecodeSession,
    tokenizer: &hawking_core::tokenizer::Tokenizer,
    prompt_ids: &[u32],
    max_new: usize,
) -> Result<Value, String> {
    session.reset();
    let result = generate_greedy(session, prompt_ids, max_new).map_err(|e| e.to_string())?;
    let text = result.decode_new(tokenizer).map_err(|e| e.to_string())?;
    let new_ids = result.new_tokens().to_vec();
    let gpu: Vec<u64> = result.gpu_ns.iter().copied().flatten().collect();
    Ok(json!({
        "generated_text_verbatim": text,
        "new_token_ids": new_ids,
        "n_new_tokens": new_ids.len(),
        "tok_s": tok_s(&result),
        "dispatches_last_step": result.dispatches.last().copied(),
        "dispatches_per_step": result.dispatches,
        "gpu_ns_per_step": result.gpu_ns,
        "gpu_ns_spread": spread_u64(&gpu),
        "decode_wall_ns": result.decode_wall_ns,
        "decode_steps": result.decode_steps,
        "fallbacks": result.fallbacks,
        "dense_w_materialized": result.dense_w_materialized,
        "median_gpu_ns_per_token": result.median_gpu_ns_per_token(),
    }))
}

#[cfg(target_os = "macos")]
fn bench_complete_token(
    session: &mut Qwen38HybridDecodeSession,
    token: u32,
    warmup: usize,
    reps: usize,
) -> Result<Value, String> {
    for _ in 0..warmup {
        session.reset();
        let _ = session
            .measure_token_dispatches(token)
            .map_err(|e| e.to_string())?;
    }
    let mut gpu = Vec::new();
    let mut wall = Vec::new();
    let mut disp = Vec::new();
    for _ in 0..reps {
        session.reset();
        let t0 = Instant::now();
        let (_sampled, dispatches, timing) = session
            .measure_token_dispatches(token)
            .map_err(|e| e.to_string())?;
        wall.push(t0.elapsed().as_nanos() as u64);
        if let Some(ns) = timing.gpu_ns {
            gpu.push(ns);
        }
        disp.push(dispatches);
    }
    Ok(json!({
        "warmup": warmup,
        "reps": reps,
        "gpu_ns": spread_u64(&gpu),
        "wall_ns": spread_u64(&wall),
        "dispatches": disp.last().copied(),
        "dispatches_all": disp,
        "dense_w_materialized": session.dense_w_materialized,
        "mlp_fusion": session.mlp_fusion.as_str(),
        "q2f_force_serial": session.q2f_force_serial,
        "timing_label": "DIRTY_ENGINEERING",
    }))
}

#[cfg(target_os = "macos")]
fn main() {
    let args = parse_args();
    if !args.artifact_root.join("catalog.hq38m20").is_file() {
        fail(format!(
            "missing {} — pack the q2f mix first",
            args.artifact_root.join("catalog.hq38m20").display()
        ));
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

    eprintln!(
        "native_2bit_mlp open {} max_seq={}",
        args.artifact_root.display(),
        args.max_seq_len
    );
    let open_started = Instant::now();
    let mut session = Qwen38HybridDecodeSession::open(&args.artifact_root, args.max_seq_len)
        .unwrap_or_else(|e| fail(e));
    let session_open_s = open_started.elapsed().as_secs_f64();
    eprintln!(
        "session open {session_open_s:.3}s dense_w={}",
        session.dense_w_materialized
    );

    eprintln!("parity: MLP gate+up+swiglu layer 0 (q2f packed vs unfused packed)");
    let mlp_parity = session
        .measure_mlp_fusion_parity(0)
        .unwrap_or_else(|e| fail(e));
    eprintln!(
        "  unfused={} pair={} swiglu={} max_abs gate={} up={} act={} dense_w={}",
        mlp_parity.unfused_dispatches,
        mlp_parity.fused_pair_dispatches,
        mlp_parity.fused_swiglu_dispatches,
        mlp_parity.max_abs_diff_gate,
        mlp_parity.max_abs_diff_up,
        mlp_parity.max_abs_diff_act,
        mlp_parity.dense_w_materialized
    );

    let probe_token = prompt_ids[0];
    let mut arms = Vec::new();

    session.set_q2f_force_serial(false);
    session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, false, false);
    eprintln!(
        "complete-token PRODUCTION fused-swiglu geo warmup={} reps={}",
        args.warmup, args.reps
    );
    let production = bench_complete_token(&mut session, probe_token, args.warmup, args.reps)
        .unwrap_or_else(|e| fail(e));
    arms.push(json!({
        "id": "production_fused_swiglu_geo",
        "role": "production",
        "kernel": "qwen_q2f_group64_matvec_gate_up_swiglu_geo_tpr64_tg128 + qwen_q2f_group64_matvec_geo_tpr64_tg128 (down)",
        "theoretical_dispatches": qwen38_fused_dispatches_per_token(Qwen38MlpFusion::GateUpSwiglu, false, false),
        "bench": production,
    }));

    session.set_q2f_force_serial(false);
    session.apply_fusion(Qwen38MlpFusion::Off, false, false);
    eprintln!("complete-token NO-OP unfused geo");
    let noop = bench_complete_token(&mut session, probe_token, args.warmup, args.reps)
        .unwrap_or_else(|e| fail(e));
    arms.push(json!({
        "id": "noop_unfused_geo",
        "role": "noop_control",
        "kernel": "qwen_q2f_group64_matvec_geo_tpr64_tg128",
        "theoretical_dispatches": production_dispatches_per_token(),
        "bench": noop,
        "why": "same specialized geo kernel, fusion off; a no-op fusion must not look like a win on dispatch count",
    }));

    session.set_q2f_force_serial(true);
    session.apply_fusion(Qwen38MlpFusion::Off, false, false);
    eprintln!("complete-token BAD serial unfused");
    let bad = bench_complete_token(&mut session, probe_token, args.warmup, args.reps)
        .unwrap_or_else(|e| fail(e));
    arms.push(json!({
        "id": "bad_serial_unfused",
        "role": "bad_control",
        "kernel": "qwen_q2f_group64_matvec (serial, one thread per row)",
        "theoretical_dispatches": production_dispatches_per_token(),
        "bench": bad,
        "why": "deliberately worse occupancy; the specialized geo kernel must beat this or the speed claim is invalid",
    }));
    session.set_q2f_force_serial(false);

    let prod_gpu: Vec<u64> = arms[0]["bench"]["gpu_ns"]["all"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
        .unwrap_or_default();
    let noop_gpu: Vec<u64> = arms[1]["bench"]["gpu_ns"]["all"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
        .unwrap_or_default();
    let bad_gpu: Vec<u64> = arms[2]["bench"]["gpu_ns"]["all"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
        .unwrap_or_default();
    let prod_noop_overlap = ranges_overlap(&prod_gpu, &noop_gpu);
    let prod_bad_overlap = ranges_overlap(&prod_gpu, &bad_gpu);
    let mut comparison = json!({
        "production_vs_noop_overlap": prod_noop_overlap,
        "production_vs_bad_overlap": prod_bad_overlap,
        "timing_label": "DIRTY_ENGINEERING",
    });
    if prod_noop_overlap {
        comparison["production_vs_noop"] = json!("NOT SEPARATED");
        comparison["mean_delta_production_vs_noop"] = Value::Null;
        comparison["why_no_mean_delta"] = json!("N016 §16: overlapping arms are not separated");
    }
    if prod_bad_overlap {
        comparison["production_vs_bad"] = json!("NOT SEPARATED");
        comparison["mean_delta_production_vs_bad"] = Value::Null;
    } else if !prod_gpu.is_empty() && !bad_gpu.is_empty() {
        comparison["production_vs_bad"] = json!("SEPARATED");
        comparison["bad_control_rejected"] = json!(true);
    }

    let mut generate = json!({"skipped": args.skip_generate});
    if !args.skip_generate {
        std::env::set_var("HAWKING_QWEN38_IGNORE_EOS", "1");
        session.set_q2f_force_serial(false);
        session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, false, false);
        eprintln!("generate PRODUCTION fused-swiglu {} tokens", args.max_new_tokens);
        let fused = generate_once(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
        )
        .unwrap_or_else(|e| fail(e));
        session.apply_fusion(Qwen38MlpFusion::Off, false, false);
        eprintln!("generate NO-OP unfused {} tokens (token-id oracle)", args.max_new_tokens);
        let unfused = generate_once(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
        )
        .unwrap_or_else(|e| fail(e));
        let fused_ids = fused["new_token_ids"].clone();
        let unfused_ids = unfused["new_token_ids"].clone();
        generate = json!({
            "ignore_eos": true,
            "fused_swiglu": fused,
            "unfused_oracle": unfused,
            "token_ids_match_unfused_oracle": fused_ids == unfused_ids,
            "dense_w_materialized": session.dense_w_materialized,
        });
    }

    let body = json!({
        "schema": "hawking.headless.native_2bit_mlp.v1",
        "generated_at": now_iso(),
        "git_head": git_head(),
        "question": "FULL-MLP native 2-bit: packed q2f g64 codes, fused SwiGLU, dense_w=0, complete-token bench, coherent generate",
        "did_not_load_second_27b": true,
        "did_not_write_under_models": true,
        "packed_2bit_codes_consumed_directly": true,
        "dense_w_materialized": session.dense_w_materialized,
        "session_open_s": session_open_s,
        "artifact_root": args.artifact_root,
        "tokenizer": args.tokenizer,
        "prompt": args.prompt,
        "rendered_prompt": rendered,
        "prompt_ids": prompt_ids,
        "max_new_tokens": args.max_new_tokens,
        "kernels": {
            "production_geo": "qwen_q2f_group64_matvec_geo_tpr64_tg128",
            "production_fused_swiglu": "qwen_q2f_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
            "bad_serial": "qwen_q2f_group64_matvec",
            "specialized_on_group": true,
            "group_size_bind_time_divide": false,
            "group": 64,
            "reconstruction": "w = (float(q) - 1.5) * delta, q in {0,1,2,3}",
        },
        "parity": {
            "mlp_gate_up_swiglu": parity_json(&mlp_parity),
        },
        "complete_token_bench": {
            "reps": args.reps,
            "warmup": args.warmup,
            "arms": arms,
            "comparison": comparison,
            "timing_label": "DIRTY_ENGINEERING",
        },
        "generate": generate,
        "counting": {
            "method": "TokenCommandBuffer.dispatch_count, one kernel launch = one dispatch",
            "unfused": production_dispatches_per_token(),
            "fused_swiglu": qwen38_fused_dispatches_per_token(Qwen38MlpFusion::GateUpSwiglu, false, false),
        },
    });

    if let Some(path) = &args.out {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(path, serde_json::to_vec_pretty(&body).expect("json")).unwrap_or_else(|e| fail(e));
        eprintln!("wrote {}", path.display());
    }
    println!("{}", serde_json::to_string_pretty(&body).expect("json"));
}

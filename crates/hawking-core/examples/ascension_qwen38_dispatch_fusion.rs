//! Dispatch fusion A/B on the Qwen3.8 uniform-q4 production path.
//!
//! Opens one catalog (does not load a second 27B). Measures unfused 964
//! vs fused gate+up(+SwiGLU) / GQA QKV / DeltaNet qkvz+ba. Reports
//! dispatches, tok/s spread, 16 verbatim tokens, and fused-vs-unfused
//! max_abs_diff. Fusion that is slower is still the result.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core \
//!   --example ascension_qwen38_dispatch_fusion
//! ./tools/gpu_lane_lock.sh qwen38-dispatch-fusion \
//!   workspace/ops/build/rust/release-fast/examples/ascension_qwen38_dispatch_fusion \
//!   --artifact-root ~/models/qwen38-gravity-uniform-q4-v1 \
//!   --tokenizer ~/models/qwen3.8-27b-abliterated-bf16/tokenizer.json \
//!   --out receipts/headless/NOETIC_DISPATCH_FUSION.json
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
    render_qwen38_user_chat, Qwen38FusionParity, Qwen38GenerateResult, Qwen38HybridDecodeSession,
    Qwen38MlpFusion,
};
use hawking_core::model::qwen38_token_ns_ledger::production_dispatches_per_token;

fn usage() -> &'static str {
    "usage: ascension_qwen38_dispatch_fusion --artifact-root DIR --tokenizer PATH \
        [--prompt TEXT] [--raw-prompt] [--max-new-tokens N] [--max-seq-len N] \
        [--reps N] [--skip-decode] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("ascension_qwen38_dispatch_fusion: {message}");
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
    let mut reps = 3usize;
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
        "fused_dispatches": p.fused_pair_dispatches.min(p.fused_swiglu_dispatches),
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
    for i in 0..reps {
        session.reset();
        let result = generate_greedy(session, prompt_ids, max_new).map_err(|e| e.to_string())?;
        let text = result.decode_new(tokenizer).map_err(|e| e.to_string())?;
        let new_ids = result.new_tokens().to_vec();
        let ts = tok_s(&result);
        let disp = result.dispatches.last().copied();
        eprintln!(
            "  rep {i}: tok/s={ts:?} dispatches={disp:?} new={} text={text:?}",
            new_ids.len()
        );
        texts.push(text);
        ids.push(new_ids);
        tok_s_reps.push(ts);
        dispatch_reps.push(disp);
        decode_wall_ns.push(result.decode_wall_ns);
        gpu_median.push(result.median_gpu_ns_per_token());
        fallbacks.push(result.fallbacks);
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
        "dense_w_materialized": 0,
    }))
}

#[cfg(target_os = "macos")]
fn probe_dispatches(
    session: &mut Qwen38HybridDecodeSession,
    token: u32,
) -> Result<Value, String> {
    session.reset();
    let theoretical = session.theoretical_dispatches();
    let (sampled, dispatches, timing) = session
        .measure_token_dispatches(token)
        .map_err(|e| e.to_string())?;
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
    }))
}

#[cfg(target_os = "macos")]
fn run(args: Args) {
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
        "qwen38 dispatch fusion: open {} max_seq={}",
        args.artifact_root.display(),
        args.max_seq_len
    );
    let open_started = Instant::now();
    let mut session = Qwen38HybridDecodeSession::open(&args.artifact_root, args.max_seq_len)
        .unwrap_or_else(|e| fail(e));
    let session_open_s = open_started.elapsed().as_secs_f64();
    eprintln!("session open {session_open_s:.3}s");

    eprintln!("parity: MLP gate+up+swiglu layer 0");
    let mlp_parity = session
        .measure_mlp_fusion_parity(0)
        .unwrap_or_else(|e| fail(e));
    eprintln!(
        "  unfused={} pair={} swiglu={} max_abs gate={} up={} act={}",
        mlp_parity.unfused_dispatches,
        mlp_parity.fused_pair_dispatches,
        mlp_parity.fused_swiglu_dispatches,
        mlp_parity.max_abs_diff_gate,
        mlp_parity.max_abs_diff_up,
        mlp_parity.max_abs_diff_act
    );

    eprintln!("parity: GQA QKV layer 3");
    let qkv_parity = session
        .measure_qkv_fusion_parity(3)
        .unwrap_or_else(|e| fail(e));
    eprintln!(
        "  unfused={} fused={} max_abs q={} k={} v={}",
        qkv_parity.unfused_dispatches,
        qkv_parity.fused_pair_dispatches,
        qkv_parity.max_abs_diff_gate,
        qkv_parity.max_abs_diff_up,
        qkv_parity.max_abs_diff_act
    );

    eprintln!("parity: DeltaNet qkvz+ba layer 0");
    let dn_parity = session
        .measure_dn_inproj_fusion_parity(0)
        .unwrap_or_else(|e| fail(e));
    eprintln!(
        "  unfused={} fused={} max_abs qkvz={} ba={}",
        dn_parity.unfused_dispatches,
        dn_parity.fused_pair_dispatches,
        dn_parity.max_abs_diff_gate,
        dn_parity.max_abs_diff_up
    );

    let probe_token = prompt_ids[0];
    let mut dispatch_probes = Vec::new();

    session.apply_fusion(Qwen38MlpFusion::Off, false, false);
    eprintln!("dispatch probe: unfused");
    dispatch_probes.push(json!({
        "id": "unfused",
        "probe": probe_dispatches(&mut session, probe_token).unwrap_or_else(|e| fail(e)),
    }));

    session.apply_fusion(Qwen38MlpFusion::GateUpPair, false, false);
    eprintln!("dispatch probe: mlp pair");
    dispatch_probes.push(json!({
        "id": "mlp_pair",
        "probe": probe_dispatches(&mut session, probe_token).unwrap_or_else(|e| fail(e)),
    }));

    session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, false, false);
    eprintln!("dispatch probe: mlp swiglu");
    dispatch_probes.push(json!({
        "id": "mlp_swiglu",
        "probe": probe_dispatches(&mut session, probe_token).unwrap_or_else(|e| fail(e)),
    }));

    session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);
    eprintln!("dispatch probe: mlp swiglu + qkv + dn");
    dispatch_probes.push(json!({
        "id": "mlp_swiglu_qkv_dn",
        "probe": probe_dispatches(&mut session, probe_token).unwrap_or_else(|e| fail(e)),
    }));

    let mut decode_arms = json!({});
    if !args.skip_decode {
        session.apply_fusion(Qwen38MlpFusion::Off, false, false);
        eprintln!("decode BEFORE (unfused) reps={}", args.reps);
        decode_arms["unfused"] = generate_arm(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
            args.reps,
        )
        .unwrap_or_else(|e| fail(e));

        session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, false, false);
        eprintln!("decode AFTER mlp swiglu reps={}", args.reps);
        decode_arms["mlp_swiglu"] = generate_arm(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
            args.reps,
        )
        .unwrap_or_else(|e| fail(e));

        session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);
        eprintln!("decode AFTER mlp swiglu + qkv + dn reps={}", args.reps);
        decode_arms["mlp_swiglu_qkv_dn"] = generate_arm(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
            args.reps,
        )
        .unwrap_or_else(|e| fail(e));
    }

    let before = production_dispatches_per_token();
    let after_swiglu = qwen38_fused_dispatches_per_token(Qwen38MlpFusion::GateUpSwiglu, false, false);
    let after_all = qwen38_fused_dispatches_per_token(Qwen38MlpFusion::GateUpSwiglu, true, true);
    let body = json!({
        "schema": "hawking.headless.noetic_dispatch_fusion.v1",
        "generated_at": now_iso(),
        "git_head": git_head(),
        "question": "Does fusing operators cut dispatches per token below 964 while staying coherent?",
        "production_path": "qwen38_uniform_q4_fused",
        "did_not_load_second_27b": true,
        "did_not_write_under_models": true,
        "dense_w_materialized": 0,
        "expanded_to_q4": 0,
        "expanded_to_float_gemv": 0,
        "session_open_s": session_open_s,
        "artifact_root": args.artifact_root,
        "tokenizer": args.tokenizer,
        "prompt": args.prompt,
        "rendered_prompt": rendered,
        "prompt_ids": prompt_ids,
        "max_new_tokens": args.max_new_tokens,
        "counting": {
            "method": "TokenCommandBuffer.dispatch_count, one kernel launch = one dispatch, same as production_dispatches_per_token",
            "before": before,
            "after_mlp_pair": qwen38_fused_dispatches_per_token(Qwen38MlpFusion::GateUpPair, false, false),
            "after_mlp_swiglu": after_swiglu,
            "after_mlp_swiglu_qkv_dn": after_all,
            "command_buffers": 1,
        },
        "prior_art": {
            "q80_command_buffers_before": 337,
            "q80_command_buffers_after": 49,
            "megakernel_8layer_f16": "measured 4.4x SLOWER; fusion is not automatically a win",
            "note": "those numbers are anchors, not re-derived in this run",
        },
        "fusions_attempted": [
            "gate_up_pair (same-row dual geo_tpr64, still a SwiGLU dispatch)",
            "gate_up_swiglu (same-row dual + silu(g)*up in-register)",
            "gqa_qkv concat geo_tpr64 (3 matvecs -> 1, 16 GQA layers)",
            "dn_qkvz_ba concat geo_tpr64 (2 matvecs -> 1, 48 DeltaNet layers)",
        ],
        "kernels": [
            "qwen_uniform_q4_group64_matvec_gate_up_geo_tpr64_tg128",
            "qwen_uniform_q4_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
            "qwen_uniform_q4_group64_matvec_pair_concat_geo_tpr64_tg128",
            "qwen_uniform_q4_group64_matvec_qkv_geo_tpr64_tg128",
        ],
        "parity": {
            "mlp_gate_up_swiglu": parity_json(&mlp_parity),
            "gqa_qkv": parity_json(&qkv_parity),
            "dn_qkvz_ba": parity_json(&dn_parity),
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

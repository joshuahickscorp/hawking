//! Complete Qwen3.8 TOKEN_NS decomposition down to physical work.
//!
//! ```text
//! ./tools/gpu_lane_lock.sh qwen38-token-ns \
//!   workspace/ops/build/rust/release-fast/examples/ascension_qwen38_token_ns \
//!   --artifact-root .../uniform-q4-v1 \
//!   --tokenizer .../bf16/tokenizer.json \
//!   --out receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json
//! ```

use hawking_core::model::qwen38_geometry::{QWEN38_INTERMEDIATE, QWEN38_VOCAB};
use hawking_core::model::qwen38_hybrid_decode::{
    generate_greedy, load_qwen38_tokenizer, render_qwen38_user_chat, Qwen38GenerateResult,
    Qwen38HybridDecodeSession, QWEN38_Q4_ADDR_PROBE_KERNEL, QWEN38_Q4_DECODE_PROBE_KERNEL,
    QWEN38_Q4_MATVEC_KERNEL,
};
use hawking_core::model::qwen38_token_ns_ledger::{
    geo_tpr64_occupancy, median_u64, production_dispatches_per_token, seal_components,
    theoretical_dispatches, theoretical_state_bytes, theoretical_weight_bytes, IsolatedFamily,
    ProbeSplit, ProductionStep, Qwen38TokenNsLedger, GPU_TIMESTAMP_AUTHORITY,
    HONEST_DECODE_CEILING_GB_S, QWEN38_TOKEN_NS_LEDGER_SCHEMA, UNIFORM_Q4_V1_BPW,
};
use serde_json::json;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

fn usage() -> &'static str {
    "usage: ascension_qwen38_token_ns --artifact-root DIR --tokenizer PATH \
        [--prompt TEXT] [--max-new-tokens N] [--reps N] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("ascension_qwen38_token_ns: {message}");
    process::exit(2);
}

fn require_gpu(label: &str, gpu: Option<u64>) -> u64 {
    gpu.unwrap_or_else(|| {
        fail(format!(
            "{label}: driver did not expose GPUEndTime-GPUStartTime"
        ))
    })
}

fn measure_family(
    _session: &Qwen38HybridDecodeSession,
    name: &str,
    reps: usize,
    mut run: impl FnMut() -> hawking_core::metal::CommandBufferTiming,
) -> IsolatedFamily {
    let mut gpu = Vec::new();
    let mut wait = Vec::new();
    let mut disp = 0u64;
    for i in 0..reps {
        let t = run();
        let g = require_gpu(name, t.gpu_ns);
        gpu.push(g);
        wait.push(t.wait_ns);
        disp = t.dispatches;
        eprintln!("  {name} rep{} gpu={g} wait={} disp={disp}", i + 1, t.wait_ns);
    }
    IsolatedFamily {
        name: name.to_owned(),
        median_gpu_ns: median_u64(&gpu),
        wait_ns_median: median_u64(&wait),
        gpu_ns_reps: gpu,
        dispatches: disp,
        command_buffers: 1,
    }
}

struct Args {
    artifact_root: PathBuf,
    tokenizer: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    reps: usize,
    out: PathBuf,
}

fn parse_args() -> Args {
    let mut artifact_root = None;
    let mut tokenizer = None;
    let mut prompt = "Say hi.".to_owned();
    let mut max_new_tokens = 16usize;
    let mut reps = 3usize;
    let mut out = PathBuf::from("receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json");
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
            "--max-new-tokens" => {
                max_new_tokens = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--max-new-tokens"));
            }
            "--reps" => {
                reps = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--reps"));
            }
            "--out" => out = PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))),
            other => fail(format!("unknown {other}; {}", usage())),
        }
    }
    Args {
        artifact_root: artifact_root.unwrap_or_else(|| fail(usage())),
        tokenizer: tokenizer.unwrap_or_else(|| fail(usage())),
        prompt,
        max_new_tokens,
        reps: reps.max(1),
        out,
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    fail("qwen38 token_ns is Metal-only");
}

#[cfg(target_os = "macos")]
fn main() {
    let args = parse_args();
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default()
        .trim()
        .to_owned();
    let tokenizer = load_qwen38_tokenizer(&args.tokenizer).unwrap_or_else(|e| fail(e));
    let rendered = render_qwen38_user_chat(&args.prompt);
    let prompt_ids = tokenizer
        .encode(&rendered, false)
        .unwrap_or_else(|e| fail(e));
    if prompt_ids.is_empty() {
        fail("empty prompt");
    }
    eprintln!(
        "qwen38-token-ns opening catalog; prompt tokens={}",
        prompt_ids.len()
    );
    let mut session = Qwen38HybridDecodeSession::open(&args.artifact_root, 128)
        .unwrap_or_else(|e| fail(e));

    const EXPECTED_16: [u32; 16] = [
        248068, 198, 760, 1156, 4777, 6587, 728, 310, 1910, 328, 5834, 1149, 1061, 369, 264, 1546,
    ];

    eprintln!("warmup generate (4 new tokens, discarded from claims)");
    let warmup = generate_greedy(&mut session, &prompt_ids, 4).unwrap_or_else(|e| fail(e));
    if warmup.fallbacks != 0 {
        fail("warmup fallback");
    }

    let mut production: Vec<Qwen38GenerateResult> = Vec::new();
    let mut isolated: Vec<IsolatedFamily> = Vec::new();
    let mut probes: Vec<ProbeSplit> = Vec::new();

    let rec_n = session.rec_state_f32_count();
    let conv_n = session.conv_state_f32_count();
    let gqa_n = (session.gqa_cache_f32_count() + 1) / 2;
    let rec_dest = session
        .alloc_profile_buffer(rec_n.saturating_mul(4))
        .unwrap_or_else(|e| fail(e));
    let conv_dest = session
        .alloc_profile_buffer(conv_n.saturating_mul(4))
        .unwrap_or_else(|e| fail(e));
    let gqa_dest = session
        .alloc_profile_buffer(gqa_n.saturating_mul(4))
        .unwrap_or_else(|e| fail(e));

    for rep in 0..args.reps {
        eprintln!("production generate rep {}", rep + 1);
        let g = generate_greedy(&mut session, &prompt_ids, args.max_new_tokens)
            .unwrap_or_else(|e| fail(e));
        if g.fallbacks != 0 {
            fail("production fallback");
        }
        eprintln!(
            "  median_gpu={:?} wall={} tokens={:?}",
            g.median_gpu_ns_per_token(),
            g.wall_ns,
            g.new_tokens()
        );
        production.push(g);

        if rep == 0 {
            eprintln!("isolated families ({} reps each)", args.reps);
            let families = [
                "input_norms",
                "post_norms",
                "final_norm",
                "silu_64",
                "mlp_residual_64",
                "mixer_residual_64",
                "rearrange_48",
                "ba_to_decay_48",
                "gated_rmsnorm_48",
                "rope_cache_16",
                "mha_16",
                "sigmoid_16",
                "argmax",
                "dn_gemvs",
                "gqa_gemvs",
            ];
            for name in families {
                isolated.push(measure_family(&session, name, args.reps, || {
                    session
                        .measure_isolated_family(name)
                        .unwrap_or_else(|e| fail(e))
                }));
            }
            isolated.push(measure_family(&session, "mlp_matvecs_64", args.reps, || {
                session
                    .measure_isolated_mlp_matvecs()
                    .unwrap_or_else(|e| fail(e))
            }));
            isolated.push(measure_family(&session, "mlp_full_64", args.reps, || {
                session
                    .measure_isolated_mlp_full()
                    .unwrap_or_else(|e| fail(e))
            }));
            isolated.push(measure_family(&session, "lm_head", args.reps, || {
                session.measure_isolated_lm_head().unwrap_or_else(|e| fail(e))
            }));
            isolated.push(measure_family(&session, "embed", args.reps, || {
                session
                    .measure_isolated_embed(prompt_ids[0])
                    .unwrap_or_else(|e| fail(e))
            }));
            isolated.push(measure_family(&session, "gated_delta_48", args.reps, || {
                session
                    .measure_isolated_gated_delta()
                    .unwrap_or_else(|e| fail(e))
            }));
            isolated.push(measure_family(&session, "stream_rec_state", args.reps, || {
                session
                    .measure_f32_stream("rec_state", &rec_dest)
                    .unwrap_or_else(|e| fail(e))
            }));
            isolated.push(measure_family(&session, "stream_conv_state", args.reps, || {
                session
                    .measure_f32_stream("conv_state", &conv_dest)
                    .unwrap_or_else(|e| fail(e))
            }));
            isolated.push(measure_family(&session, "stream_gqa_key", args.reps, || {
                session
                    .measure_f32_stream("gqa_key", &gqa_dest)
                    .unwrap_or_else(|e| fail(e))
            }));
            isolated.push(measure_family(&session, "stream_gqa_value", args.reps, || {
                session
                    .measure_f32_stream("gqa_value", &gqa_dest)
                    .unwrap_or_else(|e| fail(e))
            }));

            eprintln!("Q4 addr/decode probes");
            for class in ["mlp", "dn", "gqa", "lm_head"] {
                let full = measure_family(
                    &session,
                    &format!("{class}_full_probe"),
                    args.reps,
                    || {
                        session
                            .measure_isolated_class_gemvs_kernel(class, QWEN38_Q4_MATVEC_KERNEL)
                            .unwrap_or_else(|e| fail(e))
                    },
                );
                let addr = measure_family(
                    &session,
                    &format!("{class}_addr_probe"),
                    args.reps,
                    || {
                        session
                            .measure_isolated_class_gemvs_kernel(class, QWEN38_Q4_ADDR_PROBE_KERNEL)
                            .unwrap_or_else(|e| fail(e))
                    },
                );
                let dec = measure_family(
                    &session,
                    &format!("{class}_decode_probe"),
                    args.reps,
                    || {
                        session
                            .measure_isolated_class_gemvs_kernel(class, QWEN38_Q4_DECODE_PROBE_KERNEL)
                            .unwrap_or_else(|e| fail(e))
                    },
                );
                let full_n = full.median_gpu_ns.max(1) as f64;
                let addr_f = (addr.median_gpu_ns as f64 / full_n).clamp(0.0, 1.0);
                let dec_f = ((dec.median_gpu_ns as i64 - addr.median_gpu_ns as i64).max(0) as f64
                    / full_n)
                    .clamp(0.0, 1.0);
                let fma_f = (1.0 - addr_f - dec_f).max(0.0);
                eprintln!(
                    "  {class} addr_frac={addr_f:.4} decode_frac={dec_f:.4} fma_frac={fma_f:.4}"
                );
                probes.push(ProbeSplit {
                    class: class.to_owned(),
                    full_median_gpu_ns: full.median_gpu_ns,
                    addr_median_gpu_ns: addr.median_gpu_ns,
                    decode_median_gpu_ns: dec.median_gpu_ns,
                    addr_frac_of_full: addr_f,
                    decode_minus_addr_frac: dec_f,
                    fma_remainder_frac: fma_f,
                });
                isolated.push(full);
                isolated.push(addr);
                isolated.push(dec);
            }
        }
    }

    let mut steady_gpu = Vec::new();
    let mut steady_wait = Vec::new();
    let mut steady_encode = Vec::new();
    let mut steady_submit = Vec::new();
    let mut production_steps = Vec::new();
    for (rep_i, run) in production.iter().enumerate() {
        for (i, gpu) in run.gpu_ns.iter().enumerate() {
            let g = require_gpu("production", *gpu);
            let w = run.wait_ns[i];
            let e = run.encode_ns.get(i).copied().unwrap_or(0);
            let s = run.submit_ns.get(i).copied().unwrap_or(0);
            let d = run.dispatches.get(i).copied().unwrap_or(0);
            let kind = if i < run.prompt_len {
                "prefill"
            } else {
                "decode"
            };
            production_steps.push(ProductionStep {
                position: i as u32,
                kind,
                gpu_ns: g,
                wait_ns: w,
                encode_ns: e,
                submit_ns: s,
                dispatches: d,
                wall_ns: e.saturating_add(s).saturating_add(w),
            });
            if i >= run.prompt_len {
                steady_gpu.push(g);
                steady_wait.push(w);
                steady_encode.push(e);
                steady_submit.push(s);
            }
        }
        let _ = rep_i;
    }

    let median_gpu = median_u64(&steady_gpu);
    let median_wait = median_u64(&steady_wait);
    let median_encode = median_u64(&steady_encode);
    let median_submit = median_u64(&steady_submit);
    let median_wall = median_encode
        .saturating_add(median_submit)
        .saturating_add(median_wait);
    let gpu_min = steady_gpu.iter().copied().min().unwrap_or(0);
    let gpu_max = steady_gpu.iter().copied().max().unwrap_or(0);

    let tokens = production[0].new_tokens().to_vec();
    let bit_identical = production.iter().all(|p| p.new_tokens() == tokens);
    let matches_oracle = tokens.len() >= 16 && tokens[..16] == EXPECTED_16;
    if !bit_identical {
        fail("greedy tokens diverged across production reps");
    }
    if production.iter().any(|p| p.fallbacks != 0) {
        fail("fallback");
    }

    let weights = theoretical_weight_bytes();
    let seq = (prompt_ids.len() + args.max_new_tokens / 2) as u64;
    let state = theoretical_state_bytes(seq);
    let (components, closure) = seal_components(
        median_wall,
        median_gpu,
        median_encode,
        median_submit,
        median_wait,
        &isolated,
        &probes,
        &state,
        &weights,
        seq,
    );

    let genome = format!(
        "Qwen38HybridDecodeSession + {QWEN38_Q4_MATVEC_KERNEL} + qwen38_gated_delta_decode_vi \
         + qwen38_qkvz_rearrange_conv_l2_f32 + qwen38_gqa_qk_norm_rope_cache_f32; \
         deltanet_vi_parallel=true concurrent_independent=false; \
         1 production CB / {} dispatches; uniform-q4-v1 BPW={UNIFORM_Q4_V1_BPW}; \
         PocketAiHub/Qwen3.8-27B-Abliterated-MLX rev 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        production_dispatches_per_token()
    );

    let inst_note = format!(
        "Production step() adds two Instant::now calls around encode (~20 ns). \
         Isolated families and Q4 probes run in separate CBs AFTER a production generate; \
         they do not sit inside the timed token. Isolated GPU sum={} ns vs production GPU {} ns; \
         scale={:.6}. Reference G015 median 33.537 ms; this run median {:.3} ms.",
        closure.isolated_family_sum_gpu_ns,
        closure.production_gpu_ns,
        closure.gpu_scale_applied,
        median_gpu as f64 / 1e6
    );

    let ledger = Qwen38TokenNsLedger {
        schema: QWEN38_TOKEN_NS_LEDGER_SCHEMA,
        model: "qwen38",
        vehicle: "qwen38-27b/uniform-q4-v1",
        bpw: UNIFORM_Q4_V1_BPW,
        kernel_runtime_genome: genome,
        measurement_label: "DIRTY_ENGINEERING",
        gpu_timestamp_authority: GPU_TIMESTAMP_AUTHORITY,
        commit,
        regime: format!(
            "warm after 4-token discarded generate; paired {} production generates interleaved with one isolated/probe suite after rep 1; on-main worktree",
            args.reps
        ),
        production_cb_shape: true,
        weight_bytes: weights,
        state_bytes: state,
        dispatches: theoretical_dispatches(),
        occupancy: vec![
            geo_tpr64_occupancy(QWEN38_INTERMEDIATE as u64),
            geo_tpr64_occupancy(QWEN38_VOCAB as u64),
        ],
        production_steps,
        steady_gpu_ns: steady_gpu,
        steady_wait_ns: steady_wait,
        steady_encode_ns: steady_encode,
        steady_submit_ns: steady_submit,
        median_gpu_ns: median_gpu,
        median_wait_ns: median_wait,
        median_encode_ns: median_encode,
        median_submit_ns: median_submit,
        median_wall_ns: median_wall,
        gpu_spread_ns: [gpu_min, gpu_max],
        wait_minus_gpu_ns: median_wait as i64 - median_gpu as i64,
        isolated,
        probes,
        components,
        closure,
        greedy_token_ids: tokens,
        greedy_matches_oracle: matches_oracle,
        fallbacks: 0,
        instrumentation_overhead: inst_note,
        notes: vec![
            format!("honest decode ceiling {HONEST_DECODE_CEILING_GB_S} GB/s; peak 819 GB/s"),
            "uniform-q4-v1 is an ORACLE for profiling ONLY, never an endpoint".into(),
            "GPU time is GPUEndTime-GPUStartTime on completed CBs only".into(),
            format!(
                "decode_family_enabled={} swiglu={}",
                hawking_core::decode_family::family_dispatch_enabled(),
                hawking_core::decode_family::swiglu_f32()
            ),
        ],
    };

    if let Some(parent) = args.out.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let body = serde_json::to_vec_pretty(&ledger).unwrap_or_else(|e| fail(e));
    fs::write(&args.out, &body).unwrap_or_else(|e| fail(e));
    eprintln!("wrote {}", args.out.display());

    let unified = args.out.with_file_name("TOKEN_NS_QWEN38.json");
    let unified_body = json!({
        "schema": "hawking.ascent.token_ns.v1",
        "model": "qwen38",
        "vehicle": ledger.vehicle,
        "source_schema": ledger.schema,
        "source_path": args.out.display().to_string(),
        "measurement_label": ledger.measurement_label,
        "gpu_timestamp_authority": ledger.gpu_timestamp_authority,
        "TOTAL_TOKEN_NS": ledger.median_wall_ns,
        "TOTAL_GPU_BUSY_NS": ledger.median_gpu_ns,
        "components": ledger.components,
        "closure": ledger.closure,
        "greedy_token_ids": ledger.greedy_token_ids,
        "fallbacks": 0,
    });
    fs::write(
        &unified,
        serde_json::to_vec_pretty(&unified_body).unwrap_or_else(|e| fail(e)),
    )
    .unwrap_or_else(|e| fail(e));
    eprintln!("wrote {}", unified.display());

    println!("MEDIAN_GPU_NS={median_gpu}");
    println!("MEDIAN_WAIT_NS={median_wait}");
    println!("MEDIAN_WALL_NS={median_wall}");
    println!("WAIT_MINUS_GPU_NS={}", ledger.wait_minus_gpu_ns);
    println!("CLOSURE_HOLDS={}", ledger.closure.identity_holds);
    println!("GREEDY_ORACLE={matches_oracle}");
    println!("FALLBACKS=0");
    println!("GPU_SPREAD=[{gpu_min},{gpu_max}]");
}
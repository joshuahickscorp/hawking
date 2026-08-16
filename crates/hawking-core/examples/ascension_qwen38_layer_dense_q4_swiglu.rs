//! Reproduce, decompose, and attack the Qwen3.8 64-layer dense Q4 SwiGLU wall.
//!
//! ```text
//! ./tools/gpu_lane_lock.sh auto-qwen38-layer-dense-q4-swiglu \
//!   workspace/ops/build/rust/release-fast/examples/ascension_qwen38_layer_dense_q4_swiglu \
//!   --artifact-root .../uniform-q4-v1 \
//!   --tokenizer .../bf16/tokenizer.json \
//!   --out receipts/ascent-2026-08-16/qwen38-layer-dense-q4-swiglu.json
//! ```

use hawking_core::model::qwen38_geometry::{
    qwen38_layer_name, QWEN38_HIDDEN, QWEN38_INTERMEDIATE,
};
use hawking_core::model::qwen38_hybrid_decode::{
    generate_greedy, load_qwen38_tokenizer, render_qwen38_user_chat, Qwen38GenerateResult,
    Qwen38HybridDecodeSession, Qwen38MatvecKernel,
};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

fn usage() -> &'static str {
    "usage: ascension_qwen38_layer_dense_q4_swiglu --artifact-root DIR --tokenizer PATH \
        [--prompt TEXT] [--max-new-tokens N] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("ascension_qwen38_layer_dense_q4_swiglu: {message}");
    process::exit(2);
}

fn require_gpu(label: &str, gpu: Option<u64>) -> u64 {
    gpu.unwrap_or_else(|| fail(format!("{label}: driver did not expose GPUEndTime-GPUStartTime")))
}

fn median_u64(values: &[u64]) -> u64 {
    if values.is_empty() {
        fail("median of empty");
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
}

fn generate_summary(result: &Qwen38GenerateResult) -> Value {
    let gpu: Vec<u64> = result
        .gpu_ns
        .iter()
        .copied()
        .map(|v| require_gpu("generate step", v))
        .collect();
    json!({
        "new_token_ids": result.new_tokens(),
        "fallbacks": result.fallbacks,
        "gpu_ns_per_step": gpu,
        "wait_ns_per_step": result.wait_ns,
        "median_gpu_ns_per_token": result.median_gpu_ns_per_token(),
        "wall_ns": result.wall_ns,
        "steady_gpu_ns": if gpu.len() > 1 { json!(gpu[1..].to_vec()) } else { Value::Null },
        "steady_median_gpu_ns": if gpu.len() > 1 { json!(median_u64(&gpu[1..])) } else { Value::Null },
    })
}

struct Args {
    artifact_root: PathBuf,
    tokenizer: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    out: PathBuf,
}

fn parse_args() -> Args {
    let mut artifact_root = None;
    let mut tokenizer = None;
    let mut prompt = "Say hi.".to_owned();
    let mut max_new_tokens = 16usize;
    let mut out = PathBuf::from("receipts/ascent-2026-08-16/qwen38-layer-dense-q4-swiglu.json");
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
            "--out" => out = PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))),
            other => fail(format!("unknown {other}; {}", usage())),
        }
    }
    Args {
        artifact_root: artifact_root.unwrap_or_else(|| fail(usage())),
        tokenizer: tokenizer.unwrap_or_else(|| fail(usage())),
        prompt,
        max_new_tokens,
        out,
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    fail("qwen38 layer-dense-q4-swiglu is Metal-only");
}

#[cfg(target_os = "macos")]
fn main() {
    let args = parse_args();
    let tokenizer = load_qwen38_tokenizer(&args.tokenizer).unwrap_or_else(|e| fail(e));
    let rendered = render_qwen38_user_chat(&args.prompt);
    let prompt_ids = tokenizer
        .encode(&rendered, false)
        .unwrap_or_else(|e| fail(e));
    if prompt_ids.is_empty() {
        fail("empty prompt");
    }
    eprintln!(
        "qwen38-swiglu opening catalog; prompt tokens={}",
        prompt_ids.len()
    );
    let mut session = Qwen38HybridDecodeSession::open(&args.artifact_root, 128)
        .unwrap_or_else(|e| fail(e));

    const EXPECTED_16: [u32; 16] = [
        248068, 198, 760, 1156, 4777, 6587, 728, 310, 1910, 328, 5834, 1149, 1061, 369, 264, 1546,
    ];
    const NUMERIC_GATE: f32 = 2.0e-5;

    eprintln!("warmup generate (4 new tokens, discarded from claims)");
    let warmup = generate_greedy(&mut session, &prompt_ids, 4).unwrap_or_else(|e| fail(e));
    if warmup.fallbacks != 0 {
        fail("warmup fallback");
    }

    // Residual-correct class split on a warm token, then isolated classes.
    session.reset();
    let (first, first_t) = session
        .step(prompt_ids[0])
        .unwrap_or_else(|e| fail(e));
    let _ = first;
    let first_gpu = require_gpu("seed step", first_t.gpu_ns);
    let second_token = if prompt_ids.len() > 1 {
        prompt_ids[1]
    } else {
        first
    };
    let (decomp_sampled, decomp) = session
        .step_decomposed(second_token)
        .unwrap_or_else(|e| fail(e));
    eprintln!(
        "decomposed token gpu embed={:?} mixer={} mlp={} terminal={:?} sampled={decomp_sampled}",
        decomp.embed_gpu_ns, decomp.mixer_gpu_ns, decomp.mlp_gpu_ns, decomp.terminal_gpu_ns
    );

    let mut isolated = serde_json::Map::new();
    for (name, reps) in [
        (
            "mlp_full_64",
            (0..3)
                .map(|_| {
                    require_gpu(
                        "mlp_full_64",
                        session
                            .measure_isolated_mlp_full()
                            .unwrap_or_else(|e| fail(e))
                            .gpu_ns,
                    )
                })
                .collect::<Vec<_>>(),
        ),
        (
            "mlp_matvecs_64",
            (0..3)
                .map(|_| {
                    require_gpu(
                        "mlp_matvecs_64",
                        session
                            .measure_isolated_mlp_matvecs()
                            .unwrap_or_else(|e| fail(e))
                            .gpu_ns,
                    )
                })
                .collect::<Vec<_>>(),
        ),
        (
            "mlp_gate_64",
            (0..3)
                .map(|_| {
                    require_gpu(
                        "mlp_gate_64",
                        session
                            .measure_isolated_mlp_one_proj("gate")
                            .unwrap_or_else(|e| fail(e))
                            .gpu_ns,
                    )
                })
                .collect::<Vec<_>>(),
        ),
        (
            "mlp_up_64",
            (0..3)
                .map(|_| {
                    require_gpu(
                        "mlp_up_64",
                        session
                            .measure_isolated_mlp_one_proj("up")
                            .unwrap_or_else(|e| fail(e))
                            .gpu_ns,
                    )
                })
                .collect::<Vec<_>>(),
        ),
        (
            "mlp_down_64",
            (0..3)
                .map(|_| {
                    require_gpu(
                        "mlp_down_64",
                        session
                            .measure_isolated_mlp_one_proj("down")
                            .unwrap_or_else(|e| fail(e))
                            .gpu_ns,
                    )
                })
                .collect::<Vec<_>>(),
        ),
        (
            "mixer_gemvs_64",
            (0..3)
                .map(|_| {
                    require_gpu(
                        "mixer_gemvs_64",
                        session
                            .measure_isolated_mixer_gemvs()
                            .unwrap_or_else(|e| fail(e))
                            .gpu_ns,
                    )
                })
                .collect::<Vec<_>>(),
        ),
        (
            "lm_head",
            (0..3)
                .map(|_| {
                    require_gpu(
                        "lm_head",
                        session
                            .measure_isolated_lm_head()
                            .unwrap_or_else(|e| fail(e))
                            .gpu_ns,
                    )
                })
                .collect::<Vec<_>>(),
        ),
        (
            "embed",
            (0..3)
                .map(|_| {
                    require_gpu(
                        "embed",
                        session
                            .measure_isolated_embed(prompt_ids[0])
                            .unwrap_or_else(|e| fail(e))
                            .gpu_ns,
                    )
                })
                .collect::<Vec<_>>(),
        ),
        (
            "gated_delta_48_serial_vi",
            (0..3)
                .map(|_| {
                    require_gpu(
                        "gated_delta_48",
                        session
                            .measure_isolated_gated_delta()
                            .unwrap_or_else(|e| fail(e))
                            .gpu_ns,
                    )
                })
                .collect::<Vec<_>>(),
        ),
    ] {
        let med = median_u64(&reps);
        eprintln!("isolated {name} reps={reps:?} median={med}");
        isolated.insert(
            name.to_string(),
            json!({ "gpu_ns": reps, "median_gpu_ns": med }),
        );
    }

    session
        .measure_named_matvec(&qwen38_layer_name(0, "mlp.gate_proj.weight"), "gate")
        .unwrap_or_else(|e| fail(e));
    let geo_gate = session
        .read_f32_workspace("gate", QWEN38_INTERMEDIATE)
        .unwrap_or_else(|e| fail(e));

    let mut kernel_pairs = Vec::new();
    let mut _best_kernel = Qwen38MatvecKernel::GeoTpr64Tg128;
    let mut best_mlp_median = isolated
        .get("mlp_matvecs_64")
        .and_then(|v| v.get("median_gpu_ns"))
        .and_then(Value::as_u64)
        .unwrap_or(u64::MAX);
    let mut best_max_abs = 0.0_f32;

    for &candidate in Qwen38MatvecKernel::all() {
        if candidate == Qwen38MatvecKernel::GeoTpr64Tg128 {
            continue;
        }
        session.matvec_kernel = candidate;
        let _ = session
            .measure_named_matvec(&qwen38_layer_name(0, "mlp.gate_proj.weight"), "gate")
            .unwrap_or_else(|e| fail(e));
        let cand_gate = session
            .read_f32_workspace("gate", QWEN38_INTERMEDIATE)
            .unwrap_or_else(|e| fail(e));
        let max_abs = max_abs_diff(&geo_gate, &cand_gate);
        let mut a_reps = Vec::new();
        let mut b_reps = Vec::new();
        for _ in 0..3 {
            session.matvec_kernel = Qwen38MatvecKernel::GeoTpr64Tg128;
            a_reps.push(require_gpu(
                "pair A mlp",
                session
                    .measure_isolated_mlp_matvecs()
                    .unwrap_or_else(|e| fail(e))
                    .gpu_ns,
            ));
            session.matvec_kernel = candidate;
            b_reps.push(require_gpu(
                "pair B mlp",
                session
                    .measure_isolated_mlp_matvecs()
                    .unwrap_or_else(|e| fail(e))
                    .gpu_ns,
            ));
        }
        let a_med = median_u64(&a_reps);
        let b_med = median_u64(&b_reps);
        eprintln!(
            "kernel {} vs geo: A={a_reps:?} ({a_med}) B={b_reps:?} ({b_med}) max_abs={max_abs}",
            candidate.as_str()
        );
        kernel_pairs.push(json!({
            "A": "geo_tpr64_tg128",
            "B": candidate.as_str(),
            "A_gpu_ns": a_reps,
            "B_gpu_ns": b_reps,
            "A_median_gpu_ns": a_med,
            "B_median_gpu_ns": b_med,
            "gate0_max_abs": max_abs,
            "numeric_gate": NUMERIC_GATE,
            "numeric_pass": max_abs <= NUMERIC_GATE,
        }));
        if max_abs <= NUMERIC_GATE && b_med < best_mlp_median {
            best_mlp_median = b_med;
            _best_kernel = candidate;
            best_max_abs = max_abs;
        }
        session.matvec_kernel = Qwen38MatvecKernel::GeoTpr64Tg128;
    }

    session.matvec_kernel = Qwen38MatvecKernel::GeoTpr64Tg128;
    session.concurrent_independent = false;
    let mut conc_a = Vec::new();
    let mut conc_b = Vec::new();
    for _ in 0..3 {
        session.concurrent_independent = false;
        conc_a.push(require_gpu(
            "conc A",
            session
                .measure_isolated_mlp_matvecs()
                .unwrap_or_else(|e| fail(e))
                .gpu_ns,
        ));
        session.concurrent_independent = true;
        conc_b.push(require_gpu(
            "conc B",
            session
                .measure_isolated_mlp_matvecs()
                .unwrap_or_else(|e| fail(e))
                .gpu_ns,
        ));
    }
    session.concurrent_independent = false;
    let conc_a_med = median_u64(&conc_a);
    let conc_b_med = median_u64(&conc_b);
    eprintln!("concurrent mlp A={conc_a:?} ({conc_a_med}) B={conc_b:?} ({conc_b_med})");

    let mut delta_a = Vec::new();
    let mut delta_b = Vec::new();
    for _ in 0..3 {
        session.deltanet_vi_parallel = false;
        delta_a.push(require_gpu(
            "delta A",
            session
                .measure_isolated_gated_delta()
                .unwrap_or_else(|e| fail(e))
                .gpu_ns,
        ));
        session.deltanet_vi_parallel = true;
        delta_b.push(require_gpu(
            "delta B",
            session
                .measure_isolated_gated_delta()
                .unwrap_or_else(|e| fail(e))
                .gpu_ns,
        ));
    }
    session.deltanet_vi_parallel = false;
    let delta_a_med = median_u64(&delta_a);
    let delta_b_med = median_u64(&delta_b);
    eprintln!("gated_delta vi-parallel A={delta_a:?} ({delta_a_med}) B={delta_b:?} ({delta_b_med})");

    // Complete-token attack is the largest measured class: DeltaNet mixer,
    // specifically the 48-head gated-delta recurrence. Concurrent MLP and
    // shipped-kernel retargets are recorded as isolated negatives only —
    // concurrent already lost the token in the first protocol.
    let attack = "deltanet_vi_parallel";
    let attack_kernel = Qwen38MatvecKernel::GeoTpr64Tg128;
    let attack_concurrent = false;
    let attack_deltanet_vi = true;
    eprintln!(
        "attack={attack} kernel={} concurrent={attack_concurrent} deltanet_vi={attack_deltanet_vi}",
        attack_kernel.as_str()
    );

    let mut paired = Vec::new();
    let mut a_medians = Vec::new();
    let mut b_medians = Vec::new();
    let mut a_tokens = Vec::new();
    let mut b_tokens = Vec::new();
    for rep in 0..3 {
        session.matvec_kernel = Qwen38MatvecKernel::GeoTpr64Tg128;
        session.concurrent_independent = false;
        session.deltanet_vi_parallel = false;
        let a = generate_greedy(&mut session, &prompt_ids, args.max_new_tokens)
            .unwrap_or_else(|e| fail(e));
        if a.fallbacks != 0 {
            fail("A generate fallback");
        }
        session.matvec_kernel = attack_kernel;
        session.concurrent_independent = attack_concurrent;
        session.deltanet_vi_parallel = attack_deltanet_vi;
        let b = generate_greedy(&mut session, &prompt_ids, args.max_new_tokens)
            .unwrap_or_else(|e| fail(e));
        if b.fallbacks != 0 {
            fail("B generate fallback");
        }
        let a_med = a.median_gpu_ns_per_token().unwrap_or(0);
        let b_med = b.median_gpu_ns_per_token().unwrap_or(0);
        eprintln!(
            "pair{} A median_gpu={} tokens={:?} B median_gpu={} tokens={:?}",
            rep + 1,
            a_med,
            a.new_tokens(),
            b_med,
            b.new_tokens()
        );
        a_medians.push(a_med);
        b_medians.push(b_med);
        a_tokens.push(a.new_tokens().to_vec());
        b_tokens.push(b.new_tokens().to_vec());
        paired.push(json!({
            "rep": rep + 1,
            "A": generate_summary(&a),
            "B": generate_summary(&b),
        }));
    }

    let a_ids = a_tokens[0].clone();
    let a_bit_identical = a_tokens.iter().all(|t| t == &a_ids);
    let b_matches_a = b_tokens.iter().all(|t| t == &a_ids);
    let a_matches_oracle = args.max_new_tokens >= 16
        && a_ids.len() >= 16
        && a_ids[..16] == EXPECTED_16;
    let result_median = if attack == "none" {
        median_u64(&a_medians)
    } else {
        median_u64(&b_medians)
    };

    let mlp_med = isolated
        .get("mlp_matvecs_64")
        .and_then(|v| v.get("median_gpu_ns"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mixer_med = isolated
        .get("mixer_gemvs_64")
        .and_then(|v| v.get("median_gpu_ns"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let head_med = isolated
        .get("lm_head")
        .and_then(|v| v.get("median_gpu_ns"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let (top_class, top_ns) = [
        ("mlp_q4_matvecs_64", mlp_med),
        ("mixer_q4_gemvs_64", mixer_med),
        ("lm_head", head_med),
    ]
    .into_iter()
    .max_by_key(|(_, n)| *n)
    .unwrap();

    let body = json!({
        "schema": "hawking.special_unit.qwen38_layer_dense_q4_swiglu.v1",
        "lane": "auto-qwen38-layer-dense-q4-swiglu",
        "date": "2026-08-16",
        "timing_label": "DIRTY_ENGINEERING",
        "gpu_definition": "MTLCommandBuffer.GPUEndTime - GPUStartTime after wait",
        "wait_is_not_gpu": true,
        "dense_weight_materialized": false,
        "artifact": args.artifact_root.display().to_string(),
        "prompt": rendered,
        "prompt_ids": prompt_ids,
        "expected_greedy_16": EXPECTED_16,
        "warmup": generate_summary(&warmup),
        "seed_step_gpu_ns": first_gpu,
        "decomposed": {
            "embed_gpu_ns": decomp.embed_gpu_ns,
            "embed_wait_ns": decomp.embed_wait_ns,
            "mixer_gpu_ns": decomp.mixer_gpu_ns,
            "mixer_wait_ns": decomp.mixer_wait_ns,
            "mlp_gpu_ns": decomp.mlp_gpu_ns,
            "mlp_wait_ns": decomp.mlp_wait_ns,
            "terminal_gpu_ns": decomp.terminal_gpu_ns,
            "terminal_wait_ns": decomp.terminal_wait_ns,
            "deltanet_gpu_ns": decomp.deltanet_gpu_ns,
            "gqa_gpu_ns": decomp.gqa_gpu_ns,
            "sampled": decomp.sampled,
            "layer_mlp_gpu_ns": decomp.layer_mlp_gpu_ns,
            "layer_mixer_gpu_ns": decomp.layer_mixer_gpu_ns,
            "note": "sum of per-class CBs; extra CB gaps are in wait, not added into gpu sums"
        },
        "isolated": isolated,
        "kernel_pairs_isolated_mlp": kernel_pairs,
        "concurrent_isolated_mlp": {
            "A_serial_gpu_ns": conc_a,
            "B_concurrent_gpu_ns": conc_b,
            "A_median_gpu_ns": conc_a_med,
            "B_median_gpu_ns": conc_b_med,
            "complete_token": "LOST on prior paired generate (44.3 vs 42.7 ms); not shipped"
        },
        "gated_delta_vi_parallel": {
            "A_serial_gpu_ns": delta_a,
            "B_vi_parallel_gpu_ns": delta_b,
            "A_median_gpu_ns": delta_a_med,
            "B_median_gpu_ns": delta_b_med
        },
        "attack": {
            "kind": attack,
            "kernel": attack_kernel.as_str(),
            "concurrent_independent": attack_concurrent,
            "deltanet_vi_parallel": attack_deltanet_vi,
            "isolated_mlp_max_abs": best_max_abs,
            "numeric_gate": NUMERIC_GATE
        },
        "paired_generate": paired,
        "A_median_gpu_ns_per_token": a_medians,
        "B_median_gpu_ns_per_token": b_medians,
        "baseline_ns_per_token": median_u64(&a_medians),
        "result_ns_per_token": result_median,
        "correctness": {
            "A_greedy_bit_identical_across_reps": a_bit_identical,
            "B_greedy_matches_A": b_matches_a,
            "A_matches_bringup_oracle_16": a_matches_oracle,
            "A_tokens": a_ids,
            "B_tokens": b_tokens.get(0).cloned(),
            "fallbacks": 0,
            "oracle": "bring-up greedy ids + packed-artifact numeric gate 2e-5 vs geo kernel",
            "hidden": QWEN38_HIDDEN,
            "intermediate": QWEN38_INTERMEDIATE
        },
        "next_bottleneck": {
            "what": top_class,
            "measured_ns": top_ns
        }
    });

    if let Some(parent) = args.out.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(&args.out, serde_json::to_vec_pretty(&body).expect("json"))
        .unwrap_or_else(|e| fail(e));
    eprintln!("wrote {}", args.out.display());
    println!("ATTACK={attack}");
    println!("A_MEDIANS={a_medians:?}");
    println!("B_MEDIANS={b_medians:?}");
    println!("TOP_CLASS={top_class} {top_ns}");
    println!("A_ORACLE={a_matches_oracle} A_IDENT={a_bit_identical} B_MATCHES_A={b_matches_a}");
}

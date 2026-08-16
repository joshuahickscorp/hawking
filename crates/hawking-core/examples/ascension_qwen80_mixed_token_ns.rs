//! Closed Q80 mixed-1p5-v1 TOKEN_NS decomposition after recon-fuse.
//!
//! ```text
//! ./tools/gpu_lane_lock.sh q80-token-ns \
//!   workspace/ops/build/rust/release-fast/examples/ascension_qwen80_mixed_token_ns \
//!   --out receipts/ascent-2026-08-16/QWEN80_MIXED_TOKEN_NS_LEDGER.json
//! ```

use hawking_core::model::qwen80_complete_runtime::qwen80_assert_native_operator_composition_complete;
use hawking_core::model::qwen80_mixed_catalog::Qwen80MixedStreamingCatalog;
use hawking_core::model::qwen80_mixed_hybrid_decode::{
    discover_qwen80_mixed_root, generate_mixed_greedy, load_mixed_tokenizer, MixedProbeMode,
    Qwen80MixedHybridDecodeSession, QWEN80_MIXED_CLAIM,
};
use hawking_core::model::qwen80_mixed_token_ns_ledger::{
    median_of_decode_samples, median_u64, occupancy_note, seal_components, IsolatedFamily,
    MixedByteBudget, ProbeSplit, Qwen80MixedTokenNsLedger, GPU_TIMESTAMP_AUTHORITY,
    HONEST_DECODE_CEILING_GB_S, MIXED_1P5_V1_BPW, QWEN80_MIXED_TOKEN_NS_LEDGER_SCHEMA,
};
use hawking_core::model::qwen80_uniform_q4_hybrid_decode::{
    discover_qwen80_tokenizer, qwen80_default_tokenizer_path, render_qwen80_source_user_chat,
};
use serde_json::json;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

const EXPECTED_12: [u32; 12] = [8420, 748, 264, 729, 429, 17431, 288, 264, 914, 320, 72, 1734];

fn usage() -> &'static str {
    "usage: ascension_qwen80_mixed_token_ns [--artifact-root DIR] [--tokenizer PATH] \
        [--prompt TEXT] [--max-new-tokens N] [--reps N] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("ascension_qwen80_mixed_token_ns: {message}");
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
        eprintln!(
            "  {name} rep{} gpu={g} wait={} disp={disp}",
            i + 1,
            t.wait_ns
        );
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

fn probe_split(class: &str, full: &IsolatedFamily, addr: &IsolatedFamily, dec: &IsolatedFamily) -> ProbeSplit {
    let full_n = full.median_gpu_ns.max(1) as f64;
    let addr_f = (addr.median_gpu_ns as f64 / full_n).clamp(0.0, 1.0);
    let dec_f = ((dec.median_gpu_ns as i64 - addr.median_gpu_ns as i64).max(0) as f64 / full_n)
        .clamp(0.0, 1.0);
    let fma_f = (1.0 - addr_f - dec_f).max(0.0);
    eprintln!("  {class} addr_frac={addr_f:.4} decode_frac={dec_f:.4} fma_frac={fma_f:.4}");
    ProbeSplit {
        class: class.to_owned(),
        full_median_gpu_ns: full.median_gpu_ns,
        addr_median_gpu_ns: addr.median_gpu_ns,
        decode_median_gpu_ns: dec.median_gpu_ns,
        addr_frac_of_full: addr_f,
        decode_minus_addr_frac: dec_f,
        fma_remainder_frac: fma_f,
    }
}

fn byte_budget(catalog: &Qwen80MixedStreamingCatalog) -> MixedByteBudget {
    let mut b = MixedByteBudget {
        note: "catalog.nbytes of tensors read on a decode token; routed uses mean expert * 10 * 48"
            .into(),
        ..MixedByteBudget::default()
    };
    let mut routed_gate = Vec::new();
    let mut routed_up = Vec::new();
    let mut routed_down = Vec::new();
    for (name, row) in catalog.rows() {
        if name == "model.embed_tokens.weight" {
            // One gathered row, not the table.
            if row.shape.len() == 2 && row.shape[0] > 0 {
                b.embed_row_bytes = row.nbytes / row.shape[0] as u64;
            } else {
                b.embed_row_bytes = row.nbytes;
            }
            continue;
        }
        if name == "lm_head.weight" {
            b.lm_head_bytes = row.nbytes;
            continue;
        }
        if name.contains("linear_attn") && name.ends_with(".weight") {
            b.deltanet_bytes = b.deltanet_bytes.saturating_add(row.nbytes);
            continue;
        }
        if name.contains("self_attn") && name.ends_with(".weight") {
            b.gqa_bytes = b.gqa_bytes.saturating_add(row.nbytes);
            continue;
        }
        if name.contains("shared_expert_gate") {
            b.combine_gate_bytes = b.combine_gate_bytes.saturating_add(row.nbytes);
            continue;
        }
        if name.contains("shared_expert") && name.ends_with(".weight") {
            b.shared_expert_bytes = b.shared_expert_bytes.saturating_add(row.nbytes);
            continue;
        }
        if name.contains("mlp.gate.weight") {
            b.router_bytes = b.router_bytes.saturating_add(row.nbytes);
            continue;
        }
        if name.contains("mlp.experts.") && name.ends_with("gate_proj.weight") {
            routed_gate.push(row.nbytes);
        } else if name.contains("mlp.experts.") && name.ends_with("up_proj.weight") {
            routed_up.push(row.nbytes);
        } else if name.contains("mlp.experts.") && name.ends_with("down_proj.weight") {
            routed_down.push(row.nbytes);
        } else if name.contains("layernorm")
            || name.ends_with(".norm.weight")
            || name.ends_with("A_log")
            || name.ends_with("dt_bias")
            || name.contains("conv1d")
        {
            b.norms_bytes = b.norms_bytes.saturating_add(row.nbytes);
        }
    }
    let mean = |v: &[u64]| -> u64 {
        if v.is_empty() {
            0
        } else {
            v.iter().sum::<u64>() / v.len() as u64
        }
    };
    b.routed_expert_bytes = (mean(&routed_gate) + mean(&routed_up) + mean(&routed_down)) * 10 * 48;
    // conv state 36 layers + GQA cache slot 12 layers, f32, approximate.
    b.state_rw_bytes = 36 * 2 * 4 * (16 * 128 + 32 * 128) + 12 * 2 * 2 * 256 * 4;
    b.total_weight_bytes = b.deltanet_bytes
        + b.gqa_bytes
        + b.router_bytes
        + b.shared_expert_bytes
        + b.routed_expert_bytes
        + b.combine_gate_bytes
        + b.lm_head_bytes
        + b.embed_row_bytes
        + b.norms_bytes;
    b
}

struct Args {
    artifact_root: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    prompt: String,
    max_new_tokens: usize,
    reps: usize,
    out: PathBuf,
}

fn parse_args() -> Args {
    let mut artifact_root = None;
    let mut tokenizer = None;
    let mut prompt = "Write a function that reverses a string.".to_owned();
    let mut max_new_tokens = 12usize;
    let mut reps = 3usize;
    let mut out = PathBuf::from("receipts/ascent-2026-08-16/QWEN80_MIXED_TOKEN_NS_LEDGER.json");
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
        artifact_root,
        tokenizer,
        prompt,
        max_new_tokens,
        reps: reps.max(1),
        out,
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    fail("q80 mixed token_ns is Metal-only");
}

#[cfg(target_os = "macos")]
fn main() {
    qwen80_assert_native_operator_composition_complete()
        .unwrap_or_else(|e| fail(e));
    let args = parse_args();
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default()
        .trim()
        .to_owned();
    let root = args
        .artifact_root
        .clone()
        .or_else(discover_qwen80_mixed_root)
        .unwrap_or_else(|| fail("mixed artifact root not found"));
    let tokenizer_path = args
        .tokenizer
        .clone()
        .or_else(discover_qwen80_tokenizer)
        .unwrap_or_else(qwen80_default_tokenizer_path);
    let rendered = render_qwen80_source_user_chat(&args.prompt);
    eprintln!("opening mixed catalog at {}", root.display());
    let catalog = Qwen80MixedStreamingCatalog::open(&root).unwrap_or_else(|e| fail(e));
    let bytes = byte_budget(&catalog);
    eprintln!(
        "bytes deltanet={} gqa={} routed={} shared={} router={} lm_head={} total={}",
        bytes.deltanet_bytes,
        bytes.gqa_bytes,
        bytes.routed_expert_bytes,
        bytes.shared_expert_bytes,
        bytes.router_bytes,
        bytes.lm_head_bytes,
        bytes.total_weight_bytes
    );
    let tokenizer = load_mixed_tokenizer(&tokenizer_path).unwrap_or_else(|e| fail(e));
    let mut session =
        Qwen80MixedHybridDecodeSession::new(catalog, 64).unwrap_or_else(|e| fail(e));

    eprintln!("warmup generate (4 new tokens, discarded from claims)");
    let warmup = generate_mixed_greedy(&mut session, &tokenizer, &rendered, 4)
        .unwrap_or_else(|e| fail(e));
    if warmup.fallbacks.silent_or_invalid() != 0 {
        fail("warmup fallback");
    }
    eprintln!(
        "  warmup wall={:.0} gpu={:.0} ids={:?}",
        warmup.wall_ns_per_token, warmup.gpu_matvec_ns_per_token, warmup.generated_token_ids
    );

    let mut production = Vec::new();
    let mut isolated = Vec::new();
    let mut probes = Vec::new();
    let mut simd_checks = Vec::new();
    let probe_reps = args.reps.max(3);

    for rep in 0..args.reps {
        eprintln!("production generate rep {}", rep + 1);
        let g = generate_mixed_greedy(&mut session, &tokenizer, &rendered, args.max_new_tokens)
            .unwrap_or_else(|e| fail(e));
        if g.fallbacks.silent_or_invalid() != 0 {
            fail("production fallback");
        }
        if g.stages.gpu_matvec_timestamps_missing != 0 {
            fail(format!(
                "missing GPU timestamps: {}",
                g.stages.gpu_matvec_timestamps_missing
            ));
        }
        eprintln!(
            "  wall={:.0} gpu={:.0} encode={} submit={} wait_minus={} cbs={:.1} disp={:.1} ids={:?}",
            g.wall_ns_per_token,
            g.gpu_matvec_ns_per_token,
            g.stages.cb_encode_ns,
            g.stages.cb_submit_ns,
            g.wait_minus_gpu_ns_per_token,
            g.command_buffers_per_token,
            g.dispatches_per_token,
            g.generated_token_ids
        );
        production.push(g);

        if rep == 0 {
            eprintln!("isolated probes ({} reps each)", probe_reps);
            let singles = [
                (
                    "q8",
                    "model.layers.0.linear_attn.in_proj_qkvz.weight",
                    MixedProbeMode::Full,
                    MixedProbeMode::Addr,
                    MixedProbeMode::Decode,
                ),
                (
                    "q8_ba",
                    "model.layers.0.linear_attn.in_proj_ba.weight",
                    MixedProbeMode::Full,
                    MixedProbeMode::Addr,
                    MixedProbeMode::Decode,
                ),
                (
                    "q8_out",
                    "model.layers.0.linear_attn.out_proj.weight",
                    MixedProbeMode::Full,
                    MixedProbeMode::Addr,
                    MixedProbeMode::Decode,
                ),
                (
                    "q8_lm_head",
                    "lm_head.weight",
                    MixedProbeMode::Full,
                    MixedProbeMode::Addr,
                    MixedProbeMode::Decode,
                ),
                (
                    "q8_router",
                    "model.layers.0.mlp.gate.weight",
                    MixedProbeMode::Full,
                    MixedProbeMode::Addr,
                    MixedProbeMode::Decode,
                ),
            ];
            for (class, name, full_m, addr_m, dec_m) in singles {
                let full = measure_family(&format!("{class}_full"), probe_reps, || {
                    session
                        .measure_isolated_named(name, full_m)
                        .unwrap_or_else(|e| fail(e))
                });
                let addr = measure_family(&format!("{class}_addr"), probe_reps, || {
                    session
                        .measure_isolated_named(name, addr_m)
                        .unwrap_or_else(|e| fail(e))
                });
                let dec = measure_family(&format!("{class}_decode"), probe_reps, || {
                    session
                        .measure_isolated_named(name, dec_m)
                        .unwrap_or_else(|e| fail(e))
                });
                probes.push(probe_split(class, &full, &addr, &dec));
                isolated.push(full);
                isolated.push(addr);
                isolated.push(dec);
            }

            // Representative binary / residual / hgravs from a bound expert.
            let uploaded = session.uploaded_weight_names();
            let gate = uploaded
                .iter()
                .find(|n| n.contains("experts.") && n.ends_with("gate_proj.weight"))
                .cloned();
            let up = uploaded
                .iter()
                .find(|n| n.contains("experts.") && n.ends_with("up_proj.weight"))
                .cloned();
            let down = uploaded
                .iter()
                .find(|n| n.contains("experts.") && n.ends_with("down_proj.weight"))
                .cloned();
            if let Some(name) = gate {
                let full = measure_family("binary_full", probe_reps, || {
                    session
                        .measure_isolated_named(&name, MixedProbeMode::Full)
                        .unwrap_or_else(|e| fail(e))
                });
                let addr = measure_family("binary_addr", probe_reps, || {
                    session
                        .measure_isolated_named(&name, MixedProbeMode::Addr)
                        .unwrap_or_else(|e| fail(e))
                });
                let dec = measure_family("binary_decode", probe_reps, || {
                    session
                        .measure_isolated_named(&name, MixedProbeMode::Decode)
                        .unwrap_or_else(|e| fail(e))
                });
                probes.push(probe_split("binary", &full, &addr, &dec));
                // simd A/B on the same binary organ
                let simd = measure_family("binary_q80_simd", probe_reps, || {
                    session
                        .measure_isolated_named(&name, MixedProbeMode::BinarySimd)
                        .unwrap_or_else(|e| fail(e))
                });
                let gk = measure_family("binary_gk_simd", probe_reps, || {
                    session
                        .measure_isolated_named(&name, MixedProbeMode::GkBinarySimd)
                        .unwrap_or_else(|e| fail(e))
                });
                eprintln!(
                    "  simd check binary tg256={} q80_simd={} gk_simd={} ratio_q80={:.3} ratio_gk={:.3}",
                    full.median_gpu_ns,
                    simd.median_gpu_ns,
                    gk.median_gpu_ns,
                    full.median_gpu_ns as f64 / simd.median_gpu_ns.max(1) as f64,
                    full.median_gpu_ns as f64 / gk.median_gpu_ns.max(1) as f64
                );
                isolated.push(full.clone());
                isolated.push(addr);
                isolated.push(dec);
                simd_checks.push(full);
                simd_checks.push(simd);
                simd_checks.push(gk);
            }
            if let Some(name) = up {
                let full = measure_family("residual_full", probe_reps, || {
                    session
                        .measure_isolated_named(&name, MixedProbeMode::Full)
                        .unwrap_or_else(|e| fail(e))
                });
                let addr = measure_family("residual_addr", probe_reps, || {
                    session
                        .measure_isolated_named(&name, MixedProbeMode::Addr)
                        .unwrap_or_else(|e| fail(e))
                });
                let dec = measure_family("residual_decode", probe_reps, || {
                    session
                        .measure_isolated_named(&name, MixedProbeMode::Decode)
                        .unwrap_or_else(|e| fail(e))
                });
                probes.push(probe_split("residual", &full, &addr, &dec));
                isolated.push(full);
                isolated.push(addr);
                isolated.push(dec);
            }
            if let Some(name) = down {
                let full = measure_family("hgravs_full", probe_reps, || {
                    session
                        .measure_isolated_named(&name, MixedProbeMode::Full)
                        .unwrap_or_else(|e| fail(e))
                });
                let addr = measure_family("hgravs_addr", probe_reps, || {
                    session
                        .measure_isolated_named(&name, MixedProbeMode::Addr)
                        .unwrap_or_else(|e| fail(e))
                });
                let dec = measure_family("hgravs_decode", probe_reps, || {
                    session
                        .measure_isolated_named(&name, MixedProbeMode::Decode)
                        .unwrap_or_else(|e| fail(e))
                });
                probes.push(probe_split("hgravs", &full, &addr, &dec));
                let gk = measure_family("hgravs_gk_simd", probe_reps, || {
                    session
                        .measure_isolated_named(&name, MixedProbeMode::GkHgravsSimd)
                        .unwrap_or_else(|e| fail(e))
                });
                eprintln!(
                    "  simd check hgravs shipped={} gk_simd={} ratio={:.3}",
                    full.median_gpu_ns,
                    gk.median_gpu_ns,
                    full.median_gpu_ns as f64 / gk.median_gpu_ns.max(1) as f64
                );
                isolated.push(full.clone());
                isolated.push(addr);
                isolated.push(dec);
                simd_checks.push(full);
                simd_checks.push(gk);
            }

            // Q8 large organ: current tg256 vs simd_bytes vs gk (n/a for Q8)
            let qkvz = "model.layers.0.linear_attn.in_proj_qkvz.weight";
            let q8_full = measure_family("qkvz_tg256", probe_reps, || {
                session
                    .measure_isolated_named(qkvz, MixedProbeMode::Full)
                    .unwrap_or_else(|e| fail(e))
            });
            let q8_simd = measure_family("qkvz_simd_bytes", probe_reps, || {
                session
                    .measure_isolated_named(qkvz, MixedProbeMode::Q8SimdBytes)
                    .unwrap_or_else(|e| fail(e))
            });
            eprintln!(
                "  simd check qkvz tg256={} simd_bytes={} ratio={:.3}",
                q8_full.median_gpu_ns,
                q8_simd.median_gpu_ns,
                q8_full.median_gpu_ns as f64 / q8_simd.median_gpu_ns.max(1) as f64
            );
            simd_checks.push(q8_full);
            simd_checks.push(q8_simd);

            if session.bound_expert_count() > 0 {
                let wave = measure_family("routed_wave_full", probe_reps, || {
                    session
                        .measure_isolated_routed_wave(MixedProbeMode::Full)
                        .unwrap_or_else(|e| fail(e))
                });
                let wave_addr = measure_family("routed_wave_addr", probe_reps, || {
                    session
                        .measure_isolated_routed_wave(MixedProbeMode::Addr)
                        .unwrap_or_else(|e| fail(e))
                });
                let wave_dec = measure_family("routed_wave_decode", probe_reps, || {
                    session
                        .measure_isolated_routed_wave(MixedProbeMode::Decode)
                        .unwrap_or_else(|e| fail(e))
                });
                probes.push(probe_split("routed", &wave, &wave_addr, &wave_dec));
                isolated.push(wave);
                isolated.push(wave_addr);
                isolated.push(wave_dec);
            }
        }
    }

    let tokens = production[0].generated_token_ids.clone();
    let bit_identical = production.iter().all(|p| p.generated_token_ids == tokens);
    if !bit_identical {
        fail("greedy tokens diverged across production reps");
    }
    let matches_oracle = tokens == EXPECTED_12;
    if !matches_oracle {
        eprintln!("WARNING greedy ids {:?} != oracle {:?}", tokens, EXPECTED_12);
    }
    if production.iter().any(|p| p.fallbacks.silent_or_invalid() != 0) {
        fail("fallback");
    }

    let mut decode_samples = Vec::new();
    let mut steady_wall = Vec::new();
    let mut steady_gpu = Vec::new();
    for run in &production {
        for s in &run.token_samples {
            if s.kind == "decode" {
                decode_samples.push(s.clone());
                steady_wall.push(s.wall_ns);
                steady_gpu.push(s.snap.gpu_ns);
            }
        }
    }
    let (median_wall, median_snap) = median_of_decode_samples(&decode_samples);
    let median_gpu = median_snap.gpu_ns;
    let gpu_min = steady_gpu.iter().copied().min().unwrap_or(0);
    let gpu_max = steady_gpu.iter().copied().max().unwrap_or(0);
    let (components, closure) = seal_components(median_wall, &median_snap, &probes, &bytes);

    let genome = format!(
        "Qwen80MixedHybridDecodeSession recon_fuse={} facet1={} facet2={}; \
         kernels q80_binary_group_matvec_tg256 + q80_binary_group_csr_matvec_tg256 + \
         q80_uniform8_matvec_tg256/simd_bytes + q80_hgravs01_factor_matvec_simd3; \
         mixed-1p5-v1 BPW={MIXED_1P5_V1_BPW}; host DeltaNet/GQA/RMS/topk; \
         {} CBs / {} dispatches median decode token",
        hawking_core::model::qwen80_mixed_hybrid_decode::qwen80_recon_fuse_enabled(),
        hawking_core::model::qwen80_mixed_hybrid_decode::qwen80_host_facet1_enabled(),
        hawking_core::model::qwen80_mixed_hybrid_decode::qwen80_host_facet2_enabled(),
        median_snap.cbs,
        median_snap.dispatches
    );

    let inst_note = format!(
        "Production forward_token adds Instant wraps around exclusive host slices (~20 ns each, \
         ~25 extra Instants/layer * 48 ≈ 24 µs/token, 0.008% of a 301 ms token). \
         Isolated probes run AFTER production generate 1 and are not inside the timed token. \
         Organ GPU sum={} vs production GPU {}. \
         Occupancy notes: {} ; {}",
        closure.organ_gpu_sum_ns,
        closure.production_gpu_ns,
        occupancy_note(12_288, "q80_uniform8_matvec_tg256"),
        occupancy_note(64, "q80_uniform8_matvec_tg256")
    );

    let ledger = Qwen80MixedTokenNsLedger {
        schema: QWEN80_MIXED_TOKEN_NS_LEDGER_SCHEMA,
        model: "qwen80",
        vehicle: "qwen80/mixed-1p5-v1",
        bpw: MIXED_1P5_V1_BPW,
        kernel_runtime_genome: genome,
        measurement_label: "DIRTY_ENGINEERING",
        gpu_timestamp_authority: GPU_TIMESTAMP_AUTHORITY,
        commit,
        regime: format!(
            "warm after 4-token discarded generate; {} production 12-token generates; \
             isolated/probe/simd suite after production rep 1; first generate after process \
             start is the warmup (catalog already admitted)",
            args.reps
        ),
        production_cb_shape: true,
        bytes,
        token_samples: decode_samples,
        steady_wall_ns: steady_wall,
        steady_gpu_ns: steady_gpu,
        median_wall_ns: median_wall,
        median_gpu_ns: median_gpu,
        median_encode_ns: median_snap.encode_ns,
        median_submit_ns: median_snap.submit_ns,
        median_wait_ns: median_snap.wait_ns,
        gpu_spread_ns: [gpu_min, gpu_max],
        wait_minus_gpu_ns: median_snap.wait_ns as i64 - median_gpu as i64,
        isolated,
        probes,
        simd_checks,
        components,
        closure,
        greedy_token_ids: tokens,
        greedy_matches_oracle: matches_oracle,
        fallbacks: 0,
        instrumentation_overhead: inst_note,
        notes: vec![
            format!("honest decode ceiling {HONEST_DECODE_CEILING_GB_S} GB/s; peak 819 GB/s"),
            "Q4 is de-authorised; vehicle is mixed-1p5-v1".into(),
            "GPU time is GPUEndTime-GPUStartTime on completed CBs only".into(),
            format!("claim={QWEN80_MIXED_CLAIM}"),
        ],
    };

    if let Some(parent) = args.out.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let body = serde_json::to_vec_pretty(&ledger).unwrap_or_else(|e| fail(e));
    fs::write(&args.out, &body).unwrap_or_else(|e| fail(e));
    eprintln!("wrote {}", args.out.display());

    let unified = args.out.with_file_name("TOKEN_NS_Q80_MIXED.json");
    let ranked: Vec<_> = {
        let mut c = ledger.components.clone();
        c.sort_by(|a, b| {
            b.ns_per_token
                .partial_cmp(&a.ns_per_token)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        c
    };
    let unified_body = json!({
        "schema": "hawking.ascent.token_ns.v1",
        "model": "q80",
        "vehicle": ledger.vehicle,
        "source_schema": ledger.schema,
        "source_path": args.out.display().to_string(),
        "measurement_label": ledger.measurement_label,
        "gpu_timestamp_authority": ledger.gpu_timestamp_authority,
        "TOTAL_TOKEN_NS": ledger.median_wall_ns,
        "TOTAL_GPU_BUSY_NS": ledger.median_gpu_ns,
        "components": ledger.components,
        "ranked": ranked,
        "closure": ledger.closure,
        "greedy_token_ids": ledger.greedy_token_ids,
        "greedy_matches_oracle": ledger.greedy_matches_oracle,
        "fallbacks": 0,
        "probes": ledger.probes,
        "simd_checks": ledger.simd_checks.iter().map(|f| json!({
            "name": f.name,
            "median_gpu_ns": f.median_gpu_ns,
        })).collect::<Vec<_>>(),
    });
    fs::write(
        &unified,
        serde_json::to_vec_pretty(&unified_body).unwrap_or_else(|e| fail(e)),
    )
    .unwrap_or_else(|e| fail(e));
    eprintln!("wrote {}", unified.display());

    println!("MEDIAN_GPU_NS={median_gpu}");
    println!("MEDIAN_WAIT_NS={}", median_snap.wait_ns);
    println!("MEDIAN_WALL_NS={median_wall}");
    println!("WAIT_MINUS_GPU_NS={}", ledger.wait_minus_gpu_ns);
    println!("CLOSURE_HOLDS={}", ledger.closure.identity_holds);
    println!("GREEDY_ORACLE={matches_oracle}");
    println!("FALLBACKS=0");
    println!("GPU_SPREAD=[{gpu_min},{gpu_max}]");
    for c in &ranked {
        println!(
            "COMPONENT {} ns={:.0} pct={:.2} ms={:.3}",
            c.component,
            c.ns_per_token,
            c.pct_of_token_wall,
            c.ns_per_token / 1e6
        );
    }
}

//! Multi-token greedy decode of Qwen3-Coder-Next through the hybrid token
//! graph, bound to the sealed uniform-Q4 group-64 catalog.
//!
//! Reports prefill, first-token latency, and steady-state tok/s separately.
//! This is a VELOCITY BASELINE at complete_physical_bpw 4.259241 — it is not
//! BASE_TRUE_TPS, HCLI, coherence, or tournament evidence.
//!
//! ```text
//! cargo run --release -p hawking-core --example ascension_qwen80_uniform_q4_hybrid_greedy -- \
//!   --prompt "Say hi." --max-new-tokens 4 \
//!   --out receipts/QWEN80_UNIFORM_Q4_VELOCITY_BASELINE.json
//! ```

use hawking_core::model::qwen80_complete_runtime::qwen80_assert_native_operator_composition_complete;
use hawking_core::model::qwen80_token_ns_ledger::format_stage_table;
use hawking_core::model::qwen80_uniform_q4_hybrid_decode::{
    discover_qwen80_tokenizer, discover_qwen80_uniform_q4_root, generate_greedy,
    load_qwen80_tokenizer, qwen80_default_tokenizer_path, render_qwen80_source_user_chat,
    Qwen80UniformQ4HybridDecodeSession, Qwen80UniformQ4StreamingCatalog,
    QWEN80_UNIFORM_Q4_COMPLETE_PHYSICAL_BPW, QWEN80_UNIFORM_Q4_EXPECTED_MANIFEST_SEAL,
    QWEN80_UNIFORM_Q4_EXPECTED_TERMINAL_SEAL, QWEN80_UNIFORM_Q4_VELOCITY_NOT_BASE_TRUE_TPS,
};
use serde_json::json;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

struct Arguments {
    artifact_root: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    prompt: String,
    raw_prompt: bool,
    max_new_tokens: usize,
    max_seq_len: usize,
    out: Option<PathBuf>,
    ledger: Option<PathBuf>,
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_uniform_q4_hybrid_greedy \
        [--artifact-root DIR] [--tokenizer PATH] \
        [--prompt TEXT] [--raw-prompt] \
        [--max-new-tokens N] [--max-seq-len N] \
        [--out RECEIPT.json] [--ledger LEDGER.json]"
}

fn parse_args() -> Result<Arguments, String> {
    let mut artifact_root = None;
    let mut tokenizer = None;
    let mut prompt = "Say hi.".to_owned();
    let mut raw_prompt = false;
    let mut max_new_tokens = 4usize;
    let mut max_seq_len = 64usize;
    let mut out = None;
    let mut ledger = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--artifact-root" => {
                artifact_root = Some(PathBuf::from(
                    args.next().ok_or_else(|| usage().to_owned())?,
                ));
            }
            "--tokenizer" => {
                tokenizer = Some(PathBuf::from(
                    args.next().ok_or_else(|| usage().to_owned())?,
                ));
            }
            "--prompt" => {
                prompt = args.next().ok_or_else(|| usage().to_owned())?;
            }
            "--raw-prompt" => raw_prompt = true,
            "--max-new-tokens" => {
                max_new_tokens = args
                    .next()
                    .ok_or_else(|| usage().to_owned())?
                    .parse()
                    .map_err(|_| usage().to_owned())?;
            }
            "--max-seq-len" => {
                max_seq_len = args
                    .next()
                    .ok_or_else(|| usage().to_owned())?
                    .parse()
                    .map_err(|_| usage().to_owned())?;
            }
            "--out" => {
                out = Some(PathBuf::from(
                    args.next().ok_or_else(|| usage().to_owned())?,
                ));
            }
            "--ledger" => {
                ledger = Some(PathBuf::from(
                    args.next().ok_or_else(|| usage().to_owned())?,
                ));
            }
            "--help" | "-h" => return Err(usage().to_owned()),
            other => return Err(format!("unknown flag {other}; {}", usage())),
        }
    }
    Ok(Arguments {
        artifact_root,
        tokenizer,
        prompt,
        raw_prompt,
        max_new_tokens,
        max_seq_len,
        out,
        ledger,
    })
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    qwen80_assert_native_operator_composition_complete()
        .map_err(|error| format!("hybrid operator composition is incomplete: {error}"))?;
    let args = parse_args()?;
    let root = args
        .artifact_root
        .or_else(discover_qwen80_uniform_q4_root)
        .ok_or_else(|| {
            "qwen80 uniform-q4 artifact root not found; pass --artifact-root".to_owned()
        })?;
    let tokenizer_path = args
        .tokenizer
        .or_else(discover_qwen80_tokenizer)
        .unwrap_or_else(qwen80_default_tokenizer_path);
    let rendered = if args.raw_prompt {
        args.prompt.clone()
    } else {
        render_qwen80_source_user_chat(&args.prompt)
    };

    eprintln!("opening streaming catalog at {}", root.display());
    let catalog = Qwen80UniformQ4StreamingCatalog::open(&root).map_err(|e| e.to_string())?;
    if catalog.tensor_count() != 74_391 {
        return Err(format!(
            "catalog tensor count {} != 74391",
            catalog.tensor_count()
        ));
    }
    if catalog.manifest_seal_sha256 != QWEN80_UNIFORM_Q4_EXPECTED_MANIFEST_SEAL {
        eprintln!(
            "warning: manifest seal {} differs from sealed d4a140ab…",
            catalog.manifest_seal_sha256
        );
    }
    if catalog.terminal_seal_sha256.as_deref() != Some(QWEN80_UNIFORM_Q4_EXPECTED_TERMINAL_SEAL) {
        eprintln!(
            "warning: terminal seal {:?} differs from sealed b84e2d53…",
            catalog.terminal_seal_sha256
        );
    }
    eprintln!(
        "catalog tensors={} complete_physical_bpw={:.6} claim={}",
        catalog.tensor_count(),
        catalog.complete_physical_bpw,
        QWEN80_UNIFORM_Q4_VELOCITY_NOT_BASE_TRUE_TPS
    );

    let tokenizer = load_qwen80_tokenizer(&tokenizer_path).map_err(|e| e.to_string())?;
    let mut session = Qwen80UniformQ4HybridDecodeSession::new(catalog, args.max_seq_len)
        .map_err(|e| e.to_string())?;
    if args.ledger.is_some() {
        session.token_ns.enable();
    }
    eprintln!(
        "running greedy decode prompt_chars={} max_new_tokens={}",
        rendered.len(),
        args.max_new_tokens
    );
    let result = generate_greedy(&mut session, &tokenizer, &rendered, args.max_new_tokens)
        .map_err(|e| e.to_string())?;

    println!("prompt_token_ids={:?}", result.prompt_token_ids);
    println!("generated_token_ids={:?}", result.generated_token_ids);
    println!("generated_text={:?}", result.generated_text);
    println!("prefill_secs={:.6}", result.prefill_secs);
    println!(
        "first_token_latency_secs={:.6}",
        result.first_token_latency_secs
    );
    println!("decode_secs={:.6}", result.decode_secs);
    println!(
        "steady_state_tokens={} steady_state_decode_secs={:.6} steady_state_tok_s={:.6}",
        result.steady_state_tokens, result.steady_state_decode_secs, result.steady_state_tok_s
    );
    println!("peak_rss_bytes={}", result.peak_rss_bytes);
    println!(
        "fallback_count={} (matvec={} embed={} vec={} act={} expert_bind={} sample={})",
        result.fallbacks.total(),
        result.fallbacks.host_q4_matvec,
        result.fallbacks.host_q4_embedding_gather,
        result.fallbacks.host_q4_vector_decode,
        result.fallbacks.host_activation,
        result.fallbacks.host_expert_payload_bind,
        result.fallbacks.host_sample
    );
    println!(
        "native_q4_dispatches matvec={} embed={} decode_vector={} table_builds={} table_waves={} table_dispatches={}",
        result.native.q4_matvec_dispatches,
        result.native.q4_embedding_dispatches,
        result.native.q4_decode_vector_dispatches,
        result.native.expert_table_layer_builds,
        result.native.expert_table_waves,
        result.native.expert_table_matvec_dispatches
    );
    println!(
        "stage_secs embed={:.4} deltanet={:.4} gqa={:.4} moe_norm_router={:.4} moe_shared={:.4} moe_table_build={:.4} moe_routed={:.4} moe_combine={:.4} terminal={:.4} q4_matvec={:.4} host_expert_bind={:.4}",
        result.stages.embed_secs,
        result.stages.deltanet_secs,
        result.stages.gqa_secs,
        result.stages.moe_norm_router_secs,
        result.stages.moe_shared_secs,
        result.stages.moe_table_build_secs,
        result.stages.moe_routed_secs,
        result.stages.moe_combine_secs,
        result.stages.terminal_secs,
        result.stages.q4_matvec_secs,
        result.stages.host_expert_bind_secs
    );
    println!(
        "stage_ns embed={} deltanet={} gqa={} moe_norm_router={} moe_shared={} moe_table_build={} moe_routed={} moe_combine={} terminal={} q4_matvec={} host_expert_bind={}",
        result.stages.embed_ns,
        result.stages.deltanet_ns,
        result.stages.gqa_ns,
        result.stages.moe_norm_router_ns,
        result.stages.moe_shared_ns,
        result.stages.moe_table_build_ns,
        result.stages.moe_routed_ns,
        result.stages.moe_combine_ns,
        result.stages.terminal_ns,
        result.stages.q4_matvec_ns,
        result.stages.host_expert_bind_ns
    );
    let act = &result.stages.activation;
    println!(
        "activation_ns shared_swiglu={} shared_mlp_sandwich={} deltanet_conv={} deltanet_recurrent={} gqa_input_layernorm={} gqa_norm_rope={} other_host_activation={} metal_matvec_sync={}",
        act.shared_swiglu_ns,
        act.shared_mlp_sandwich_ns,
        act.deltanet_conv_ns,
        act.deltanet_recurrent_ns,
        act.gqa_input_layernorm_ns,
        act.gqa_norm_rope_ns,
        act.other_host_activation_ns,
        act.metal_matvec_sync_ns
    );
    println!(
        "complete_physical_bpw={:.6} claim={} metal_q4_matvec_used={}",
        result.complete_physical_bpw, result.claim, result.metal_q4_matvec_used
    );
    if let Some(error) = &result.metal_error {
        println!("metal_error={error}");
    }

    if let Some(out) = args.out {
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let receipt = json!({
            "schema": "hawking.ascension.qwen80_uniform_q4_velocity_baseline.v1",
            "status": QWEN80_UNIFORM_Q4_VELOCITY_NOT_BASE_TRUE_TPS,
            "claim_boundary": {
                "base_true_tps": false,
                "coherence": false,
                "capability": false,
                "hcli": false,
                "restart": false,
                "tournament": false,
                "complete_physical_bpw": result.complete_physical_bpw,
                "complete_physical_bpw_reported": QWEN80_UNIFORM_Q4_COMPLETE_PHYSICAL_BPW,
                "density_gate_bpw_max": 1.5,
                "artifact_cannot_satisfy_1_5_bpw_gate": true,
            },
            "artifact": {
                "root": root,
                "tensor_count": 74391,
                "manifest_seal_sha256": session.catalog().manifest_seal_sha256,
                "terminal_seal_sha256": session.catalog().terminal_seal_sha256,
                "complete_physical_bpw": result.complete_physical_bpw,
                "mean_component_cosine": session.catalog().mean_component_cosine,
            },
            "prompt": result.prompt,
            "prompt_token_ids": result.prompt_token_ids,
            "generated_token_ids": result.generated_token_ids,
            "generated_text": result.generated_text,
            "timing": {
                "prefill_secs": result.prefill_secs,
                "first_token_latency_secs": result.first_token_latency_secs,
                "decode_secs": result.decode_secs,
                "steady_state_tokens": result.steady_state_tokens,
                "steady_state_decode_secs": result.steady_state_decode_secs,
                "steady_state_tok_s": result.steady_state_tok_s,
            },
            "resources": {
                "peak_rss_bytes": result.peak_rss_bytes,
                "streamed_rss_cap_bytes": 16u64 * 1024 * 1024 * 1024,
            },
            "execution": {
                "hybrid_token_graph": "embed + 48-layer mixer/MoE + terminal greedy",
                "weight_codec": "uniform_q4_group64",
                "metal_q4_matvec_used": result.metal_q4_matvec_used,
                "metal_error": result.metal_error,
                "native": {
                    "q4_matvec_dispatches": result.native.q4_matvec_dispatches,
                    "q4_embedding_dispatches": result.native.q4_embedding_dispatches,
                    "q4_decode_vector_dispatches": result.native.q4_decode_vector_dispatches,
                    "expert_table_layer_builds": result.native.expert_table_layer_builds,
                    "expert_table_waves": result.native.expert_table_waves,
                    "expert_table_matvec_dispatches": result.native.expert_table_matvec_dispatches,
                },
                "stages": {
                    "embed_secs": result.stages.embed_secs,
                    "deltanet_secs": result.stages.deltanet_secs,
                    "gqa_secs": result.stages.gqa_secs,
                    "moe_norm_router_secs": result.stages.moe_norm_router_secs,
                    "moe_shared_secs": result.stages.moe_shared_secs,
                    "moe_table_build_secs": result.stages.moe_table_build_secs,
                    "moe_routed_secs": result.stages.moe_routed_secs,
                    "moe_combine_secs": result.stages.moe_combine_secs,
                    "terminal_secs": result.stages.terminal_secs,
                    "q4_matvec_secs": result.stages.q4_matvec_secs,
                    "host_expert_bind_secs": result.stages.host_expert_bind_secs,
                    "embed_ns": result.stages.embed_ns,
                    "deltanet_ns": result.stages.deltanet_ns,
                    "gqa_ns": result.stages.gqa_ns,
                    "moe_norm_router_ns": result.stages.moe_norm_router_ns,
                    "moe_shared_ns": result.stages.moe_shared_ns,
                    "moe_table_build_ns": result.stages.moe_table_build_ns,
                    "moe_routed_ns": result.stages.moe_routed_ns,
                    "moe_combine_ns": result.stages.moe_combine_ns,
                    "terminal_ns": result.stages.terminal_ns,
                    "q4_matvec_ns": result.stages.q4_matvec_ns,
                    "host_expert_bind_ns": result.stages.host_expert_bind_ns,
                },
                "fallbacks": {
                    "total": result.fallbacks.total(),
                    "host_q4_matvec": result.fallbacks.host_q4_matvec,
                    "host_q4_embedding_gather": result.fallbacks.host_q4_embedding_gather,
                    "host_q4_vector_decode": result.fallbacks.host_q4_vector_decode,
                    "host_activation": result.fallbacks.host_activation,
                    "host_expert_payload_bind": result.fallbacks.host_expert_payload_bind,
                    "host_sample": result.fallbacks.host_sample,
                    "note": if result.native.expert_table_waves > 0 {
                        "512-way device expert table is live; host_expert_payload_bind counts remaining host binds".to_string()
                    } else {
                        "expert gather is a host fallback; the composed graph has no 512-way device gather".to_string()
                    },
                },
            },
        });
        let pretty = serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?;
        fs::write(&out, pretty).map_err(|e| e.to_string())?;
        eprintln!("wrote {}", out.display());
    }
    if let Some(ledger_path) = args.ledger {
        if let Some(parent) = ledger_path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        if session.token_ns.measured_commit.is_empty() {
            if let Ok(out) = std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .output()
            {
                session.token_ns.measured_commit =
                    String::from_utf8_lossy(&out.stdout).trim().to_owned();
            }
        }
        if session.token_ns.measurement_label.is_empty() {
            session.token_ns.measurement_label = "DIRTY_ENGINEERING".to_owned();
        }
        // Compact receipt only. The v1 full CB dump was 1.2 MiB and is not
        // needed to close the token identity.
        let compact = session.token_ns.compact_receipt();
        let pretty = serde_json::to_string_pretty(&compact).map_err(|e| e.to_string())?;
        fs::write(&ledger_path, pretty).map_err(|e| e.to_string())?;
        eprintln!("wrote compact ledger {} ({} bytes)", ledger_path.display(), fs::metadata(&ledger_path).map(|m| m.len()).unwrap_or(0));
        let wall = compact
            .identity
            .as_ref()
            .map(|id| id.mean_wall_ns)
            .unwrap_or(0.0);
        eprint!("{}", format_stage_table(&compact.stage_table, wall));
        if let Some(id) = &compact.identity {
            eprintln!(
                "identity n={} mean_wall_ns={:.0} mean_sum_ns={:.0} mean_residual_ns={:.0} holds_all={} residual={}",
                id.n,
                id.mean_wall_ns,
                id.mean_sum_identity_ns,
                id.mean_residual_ns,
                id.identity_holds_all,
                id.residual_name
            );
        }
        if let Some(tot) = &compact.totals_mean_decode {
            eprintln!(
                "totals token_ns={} gpu_busy_ns={} gpu_idle_ns={} gpu_gap_ns={} cpu_critical_ns={} cbs={} disp={} syncs={} readbacks={} buf_create={} buf_rebind={} dram={} temp={}",
                tot.total_token_ns,
                tot.total_gpu_busy_ns,
                tot.total_gpu_idle_ns,
                tot.total_gpu_gap_ns,
                tot.total_cpu_critical_ns,
                tot.total_command_buffers,
                tot.total_dispatches,
                tot.total_sync_points,
                tot.total_readbacks,
                tot.total_buffer_creations,
                tot.total_buffer_rebinds,
                tot.dram_bytes_per_token,
                tot.temp_bytes_per_token
            );
        }
        eprintln!(
            "catalog_complete={} silent_zero_stages={:?}",
            compact.catalog_complete, compact.silent_zero_stages
        );
        if let Some(diag) = &compact.diagnosis {
            eprintln!(
                "token_ns_ledger verdict={} wall_ms={:.1} gpu_ms={:.1} wait_ms={:.1} cbs={:.0} disp={:.0} weight_gib={:.3} implied_gb_s_gpu={:?}",
                diag.verdict,
                diag.wall_ns as f64 / 1e6,
                diag.gpu_execution_ns as f64 / 1e6,
                diag.cpu_wait_ns as f64 / 1e6,
                diag.command_buffers_per_token,
                diag.dispatches_per_token,
                diag.weight_bytes_per_token as f64 / (1024.0 * 1024.0 * 1024.0),
                diag.implied_gb_s_from_gpu
            );
            eprintln!("rationale={}", diag.rationale);
        }
    }
    Ok(())
}

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
use hawking_core::model::qwen80_uniform_q4_hybrid_decode::{
    discover_qwen80_uniform_q4_root, generate_greedy, load_qwen80_tokenizer,
    qwen80_default_tokenizer_path, render_qwen80_source_user_chat,
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
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_uniform_q4_hybrid_greedy \
        [--artifact-root DIR] [--tokenizer PATH] \
        [--prompt TEXT] [--raw-prompt] \
        [--max-new-tokens N] [--max-seq-len N] \
        [--out RECEIPT.json]"
}

fn parse_args() -> Result<Arguments, String> {
    let mut artifact_root = None;
    let mut tokenizer = None;
    let mut prompt = "Say hi.".to_owned();
    let mut raw_prompt = false;
    let mut max_new_tokens = 4usize;
    let mut max_seq_len = 64usize;
    let mut out = None;
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
    let tokenizer_path = args.tokenizer.unwrap_or_else(qwen80_default_tokenizer_path);
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
        "native_q4_dispatches matvec={} embed={} decode_vector={}",
        result.native.q4_matvec_dispatches,
        result.native.q4_embedding_dispatches,
        result.native.q4_decode_vector_dispatches
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
                },
                "fallbacks": {
                    "total": result.fallbacks.total(),
                    "host_q4_matvec": result.fallbacks.host_q4_matvec,
                    "host_q4_embedding_gather": result.fallbacks.host_q4_embedding_gather,
                    "host_q4_vector_decode": result.fallbacks.host_q4_vector_decode,
                    "host_activation": result.fallbacks.host_activation,
                    "host_expert_payload_bind": result.fallbacks.host_expert_payload_bind,
                    "host_sample": result.fallbacks.host_sample,
                    "note": "expert gather is a host fallback; the composed graph has no 512-way device gather",
                },
            },
        });
        let pretty = serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?;
        fs::write(&out, pretty).map_err(|e| e.to_string())?;
        eprintln!("wrote {}", out.display());
    }
    Ok(())
}

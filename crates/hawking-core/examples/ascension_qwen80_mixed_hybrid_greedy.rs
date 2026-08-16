//! Generate from the packed Q80 mixed ≤1.5 catalog through the hybrid graph.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core --example ascension_qwen80_mixed_hybrid_greedy
//! ./tools/gpu_lane_lock.sh q80-mixed-generate \
//!   workspace/ops/build/rust/release-fast/examples/ascension_qwen80_mixed_hybrid_greedy \
//!   --prompt "Write a function that reverses a string." --max-new-tokens 12 \
//!   --reps 6 --out receipts/ascent-2026-08-16/Q80_MIXED_GENERATE.json
//! ```

use hawking_core::model::qwen80_complete_runtime::qwen80_assert_native_operator_composition_complete;
use hawking_core::model::qwen80_mixed_catalog::Qwen80MixedStreamingCatalog;
use hawking_core::model::qwen80_mixed_hybrid_decode::{
    discover_qwen80_mixed_root, generate_mixed_greedy, load_mixed_tokenizer,
    Qwen80MixedHybridDecodeSession, QWEN80_MIXED_CLAIM, QWEN80_MIXED_EXPECTED_MANIFEST_SEAL,
};
use hawking_core::model::qwen80_uniform_q4_hybrid_decode::{
    discover_qwen80_tokenizer, qwen80_default_tokenizer_path, render_qwen80_source_user_chat,
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
    reps: usize,
    out: Option<PathBuf>,
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_mixed_hybrid_greedy \
        [--artifact-root DIR] [--tokenizer PATH] \
        [--prompt TEXT] [--raw-prompt] \
        [--max-new-tokens N] [--max-seq-len N] [--reps N] \
        [--out RECEIPT.json]"
}

fn parse_args() -> Result<Arguments, String> {
    let mut artifact_root = None;
    let mut tokenizer = None;
    let mut prompt = "Write a function that reverses a string.".to_owned();
    let mut raw_prompt = false;
    let mut max_new_tokens = 12usize;
    let mut max_seq_len = 64usize;
    let mut reps = 1usize;
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
            "--reps" => {
                reps = args
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
    if reps == 0 {
        return Err("--reps must be positive".to_owned());
    }
    Ok(Arguments {
        artifact_root,
        tokenizer,
        prompt,
        raw_prompt,
        max_new_tokens,
        max_seq_len,
        reps,
        out,
    })
}

fn classify_text(text: &str) -> &'static str {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return "INCOHERENT";
    }
    let alpha = trimmed.chars().filter(|c| c.is_ascii_alphabetic()).count();
    let printable = trimmed
        .chars()
        .filter(|c| c.is_ascii_graphic() || c.is_ascii_whitespace())
        .count();
    if printable < trimmed.chars().count().saturating_mul(3) / 4 || alpha < 8 {
        return "INCOHERENT";
    }
    let lower = trimmed.to_ascii_lowercase();
    let has_answer = ["def ", "function", "reverse", "string", "python", "here's", "here is"]
        .iter()
        .any(|needle| lower.contains(needle));
    let repeated = {
        let words: Vec<&str> = lower.split_whitespace().collect();
        words.len() >= 6 && words.windows(3).any(|w| words.iter().filter(|x| **x == w[0]).count() >= 4)
    };
    if has_answer && !repeated {
        "COHERENT"
    } else {
        "DEGRADED"
    }
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
        .or_else(discover_qwen80_mixed_root)
        .ok_or_else(|| "qwen80 mixed artifact root not found; pass --artifact-root".to_owned())?;
    let tokenizer_path = args
        .tokenizer
        .or_else(discover_qwen80_tokenizer)
        .unwrap_or_else(qwen80_default_tokenizer_path);
    let rendered = if args.raw_prompt {
        args.prompt.clone()
    } else {
        render_qwen80_source_user_chat(&args.prompt)
    };

    eprintln!("opening mixed catalog at {}", root.display());
    let catalog = Qwen80MixedStreamingCatalog::open(&root).map_err(|e| e.to_string())?;
    if catalog.manifest_seal_sha256 != QWEN80_MIXED_EXPECTED_MANIFEST_SEAL {
        eprintln!(
            "warning: manifest seal {} differs from packed 6a09fa74…",
            catalog.manifest_seal_sha256
        );
    }
    eprintln!(
        "catalog tensors={} complete_physical_bpw={:.6} claim={}",
        catalog.tensor_count(),
        catalog.complete_physical_bpw,
        QWEN80_MIXED_CLAIM
    );

    let tokenizer = load_mixed_tokenizer(&tokenizer_path).map_err(|e| e.to_string())?;
    let mut session = Qwen80MixedHybridDecodeSession::new(catalog, args.max_seq_len)
        .map_err(|e| e.to_string())?;
    eprintln!(
        "sample parity passed={} dense_w={} silent_fallbacks={}",
        session.parity.passed,
        session.parity.dense_w_materialized,
        session.fallbacks.silent_or_invalid()
    );

    let mut reps = Vec::new();
    let mut first_text = String::new();
    let mut first_ids = Vec::new();
    for rep in 0..args.reps {
        eprintln!(
            "greedy rep {}/{} prompt_chars={} max_new_tokens={}",
            rep + 1,
            args.reps,
            rendered.len(),
            args.max_new_tokens
        );
        let result =
            generate_mixed_greedy(&mut session, &tokenizer, &rendered, args.max_new_tokens)
                .map_err(|e| e.to_string())?;
        if rep == 0 {
            first_text = result.generated_text.clone();
            first_ids = result.generated_token_ids.clone();
            println!("prompt_token_ids={:?}", result.prompt_token_ids);
            println!("generated_token_ids={:?}", result.generated_token_ids);
            println!("generated_text={:?}", result.generated_text);
            println!("coherence_class={}", classify_text(&result.generated_text));
            println!("prefill_secs={:.6}", result.prefill_secs);
            println!(
                "first_token_latency_secs={:.6}",
                result.first_token_latency_secs
            );
            println!("decode_secs={:.6}", result.decode_secs);
            println!(
                "steady_state_tokens={} steady_state_decode_secs={:.6} steady_state_tok_s={:.6}",
                result.steady_state_tokens,
                result.steady_state_decode_secs,
                result.steady_state_tok_s
            );
            println!("wall_ns_per_token={:.0}", result.wall_ns_per_token);
            println!(
                "gpu_matvec_ns_per_token={:.0} (MTLCommandBuffer GPUEnd-GPUStart)",
                result.gpu_matvec_ns_per_token
            );
            println!("peak_rss_bytes={}", result.peak_rss_bytes);
            println!(
                "fallback silent={} designed_host={} (vec={} embed={} act={} sample={})",
                result.fallbacks.silent_or_invalid(),
                result.fallbacks.designed_host_ops(),
                result.fallbacks.host_q8_vector_decode,
                result.fallbacks.host_q8_embed_gather,
                result.fallbacks.host_activation,
                result.fallbacks.host_sample
            );
            println!(
                "native binary={} residual={} hgravs={} q8={} waves={} cbs={}",
                result.native.binary_dispatches,
                result.native.residual_dispatches,
                result.native.hgravs_factor_dispatches,
                result.native.uniform8_dispatches,
                result.native.routed_expert_waves,
                result.native.command_buffers
            );
            println!(
                "stages embed={:.4} deltanet={:.4} gqa={:.4} moe_shared={:.4} moe_routed={:.4} moe_combine={:.4} terminal={:.4} mixed_matvec={:.4} gpu_matvec_ns={}",
                result.stages.embed_secs,
                result.stages.deltanet_secs,
                result.stages.gqa_secs,
                result.stages.moe_shared_secs,
                result.stages.moe_routed_secs,
                result.stages.moe_combine_secs,
                result.stages.terminal_secs,
                result.stages.mixed_matvec_secs,
                result.stages.gpu_matvec_ns
            );
            println!("dense_w_materialized={}", result.dense_w_materialized);
        } else {
            println!(
                "rep{} generated_text={:?} wall_ns_per_token={:.0} gpu_matvec_ns_per_token={:.0} ids_match={}",
                rep + 1,
                result.generated_text,
                result.wall_ns_per_token,
                result.gpu_matvec_ns_per_token,
                result.generated_token_ids == first_ids
            );
        }
        reps.push(result);
    }

    if let Some(path) = args.out {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let first = &reps[0];
        let wall: Vec<f64> = reps.iter().map(|r| r.wall_ns_per_token).collect();
        let gpu: Vec<f64> = reps.iter().map(|r| r.gpu_matvec_ns_per_token).collect();
        let receipt = json!({
            "schema": "hawking.ascent.q80_mixed_generate.v1",
            "lane": "q80-mixed-generate",
            "status": classify_text(&first.generated_text),
            "claim": first.claim,
            "claim_boundary": {
                "generation_is_the_gate": true,
                "packing_is_not_a_coherence_claim": true,
                "not_bit_exact_vs_q4": true,
                "dense_w_materialized": false,
                "graded_against": "artifact CPU/numpy oracle, never BF16 parent",
            },
            "artifact": {
                "root": root,
                "complete_physical_bpw": first.complete_physical_bpw,
                "manifest_seal_sha256": session.catalog().manifest_seal_sha256,
                "tensor_count": session.catalog().tensor_count(),
            },
            "prompt": first.prompt,
            "generated_text": first.generated_text,
            "generated_token_ids": first.generated_token_ids,
            "prompt_token_ids": first.prompt_token_ids,
            "coherence_class": classify_text(&first.generated_text),
            "correctness": {
                "gate_kind": "numeric_equivalence_vs_artifact_oracle",
                "tolerance": 2e-5,
                "parity": first.parity,
                "dense_w_materialized": false,
                "silent_fallback_count": first.fallbacks.silent_or_invalid(),
                "designed_host_ops": first.fallbacks.designed_host_ops(),
                "fallbacks": first.fallbacks,
            },
            "execution": {
                "native": first.native,
                "stages": first.stages,
                "metal_error": first.metal_error,
                "peak_rss_bytes": first.peak_rss_bytes,
                "weight_codec": "mixed_gate_binary_up_rice_q1_down_hgravs01_r160_b3_nonexpert_q8",
            },
            "timing": {
                "label": "DIRTY_ENGINEERING",
                "authority": "wall decode_secs/generated_tokens; GPU is MTLCommandBuffer.GPUEndTime-GPUStartTime of weight GEMVs only",
                "prefill_secs": first.prefill_secs,
                "decode_secs": first.decode_secs,
                "steady_state_tok_s": first.steady_state_tok_s,
                "wall_ns_per_token_reps": wall,
                "gpu_matvec_ns_per_token_reps": gpu,
                "gpu_timestamps_missing": first.stages.gpu_matvec_timestamps_missing,
            },
            "reps_generated_text": reps.iter().map(|r| r.generated_text.clone()).collect::<Vec<_>>(),
            "reps_ids_match_first": reps.iter().map(|r| r.generated_token_ids == first_ids).collect::<Vec<_>>(),
        });
        fs::write(&path, serde_json::to_string_pretty(&receipt).unwrap())
            .map_err(|e| e.to_string())?;
        eprintln!("wrote {}", path.display());
        let _ = first_text;
    }
    Ok(())
}

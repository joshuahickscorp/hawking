//! Source-token probe for a resident Gravity Llama artifact.
//!
//! This is deliberately separate from the throughput harness: a pseudo-token
//! stream can prove that the kernels execute, but it cannot prove that a
//! source-preserving artifact produces the source model's next token.  The
//! probe feeds an exact token-id prefix, records the greedy next-token ids,
//! and then continues through the same stateful KV cache.

use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .expect("non-empty logits")
}

fn percentile_ms(values: &[f64], percentile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    // Nearest-rank percentile: it is deliberately conservative for the short
    // complete-token sweeps used by the TG gate (p99 is the slowest sample
    // until at least 100 decode tokens have been collected).
    let rank = ((percentile * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Some(sorted[rank])
}

fn token_hash(tokens: &[u32]) -> String {
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hawking_core::gravity_llama::gpu::GravityLlamaGpu;
    use hawking_core::metal::MetalContext;

    let mut artifact: Option<PathBuf> = None;
    let mut tokens_file: Option<PathBuf> = None;
    let mut inline_tokens: Option<Vec<u32>> = None;
    let mut generate = 16usize;
    let mut out: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--artifact" => artifact = args.next().map(PathBuf::from),
            "--tokens-file" => tokens_file = args.next().map(PathBuf::from),
            "--tokens" => {
                let value = args
                    .next()
                    .ok_or("--tokens needs whitespace-separated token ids")?;
                inline_tokens = Some(
                    value
                        .split_whitespace()
                        .map(str::parse)
                        .collect::<Result<_, _>>()?,
                );
            }
            "--generate" => generate = args.next().ok_or("--generate needs a value")?.parse()?,
            "--out" => out = args.next().map(PathBuf::from),
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    let artifact = artifact.ok_or("--artifact is required")?;
    if tokens_file.is_some() && inline_tokens.is_some() {
        return Err("use exactly one of --tokens-file or --tokens".into());
    }
    let prefix: Vec<u32> = match (tokens_file.as_ref(), inline_tokens) {
        (Some(path), None) => std::fs::read_to_string(path)?
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()?,
        (None, Some(tokens)) => tokens,
        (None, None) => return Err("one of --tokens-file or --tokens is required".into()),
        (Some(_), Some(_)) => unreachable!("validated above"),
    };
    if prefix.is_empty() {
        return Err("token prefix is empty".into());
    }

    let model = GravityLlamaGpu::open_with(MetalContext::new()?, &artifact, true)?;
    let (logits, prefill_stats) = model.forward(&prefix)?;
    let first = argmax(&logits);
    let mut generated = Vec::with_capacity(generate);
    let mut decode_ms = Vec::with_capacity(generate);
    let mut dispatches = 0usize;
    let mut fused_qkv_dispatches = 0usize;
    let mut fused_gate_up_dispatches = 0usize;
    let mut command_buffers = 0usize;
    let mut next = first;
    for i in 0..generate {
        generated.push(next);
        let t0 = std::time::Instant::now();
        let (next_logits, stats) = model.forward_at(&[next], prefix.len() + i)?;
        decode_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        dispatches += stats.dispatches;
        fused_qkv_dispatches += stats.fused_qkv_dispatches;
        fused_gate_up_dispatches += stats.fused_gate_up_dispatches;
        command_buffers += stats.command_buffers;
        next = argmax(&next_logits);
    }

    let mut sorted: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    sorted.sort_by(|(_, a), (_, b)| b.total_cmp(a));
    let top5: Vec<serde_json::Value> = sorted
        .into_iter()
        .take(5)
        .map(|(id, value)| serde_json::json!({"id": id, "logit": value}))
        .collect();
    let receipt = serde_json::json!({
        "schema": "hawking.gravity.llama_source_probe.v2",
        "measurement_classification": {
            "complete_token": true,
            "same_model_candidate_only": true,
            "capability_gate": "NOT_RUN",
            "clean_lease": "NOT_ATTESTED_BY_THIS_PROBE",
            "tg_promotion_allowed": false,
            "reason": "The probe records physical resident decode facts; a TG promotion additionally requires a capability-matched, CLEAN same-model before/after campaign."
        },
        "artifact": artifact.to_string_lossy(),
        "execution_representation": {
            "artifact_grammar": "source-preserving packed Gravity",
            "q4_kernel": if std::env::var_os("HAWKING_GRAVITY_Q4_V3").is_some() {
                "q4_k_v3_8row_candidate"
            } else {
                "q4_k_b9430_parity_baseline"
            },
            "kv_precision": if std::env::var_os("HAWKING_GRAVITY_F16_KV").is_some() {
                "explicit_f16_candidate"
            } else {
                "source_parity_f32"
            },
            "qkv_rope_kv_append": if std::env::var_os("HAWKING_GRAVITY_FUSED_QKV").is_some() {
                "fused_source_candidate"
            } else {
                "decomposed_parity_baseline"
            },
        },
        "tokens_file": tokens_file.map(|path| path.to_string_lossy().into_owned()),
        "inline_token_prefix": prefix,
        "device": model.device_name(),
        "load_ms": model.load_ms,
        "resident_weight_bytes": model.device_bytes,
        "kv_bytes_at_final_position": model.kv_bytes_for(prefix.len() + generate),
        "rope_interleaved": model.arch.rope_interleaved,
        "rope_freq_factors": model.arch.rope_freq_factors.as_ref().map(|v| v.len()),
        "prefix_tokens": prefix.len(),
        "generated_tokens": generate,
        "greedy_first_token": first,
        "greedy_generated_token_ids": generated,
        "continuation_sha256": token_hash(&generated),
        "top5_after_prefix": top5,
        "prefill_ms": prefill_stats.total_ms,
        "prefill_dispatches": prefill_stats.dispatches,
        "prefill_fused_qkv_dispatches": prefill_stats.fused_qkv_dispatches,
        "prefill_fused_gate_up_dispatches": prefill_stats.fused_gate_up_dispatches,
        "prefill_command_buffers": prefill_stats.command_buffers,
        "decode_ms": decode_ms,
        "decode_dispatches": dispatches,
        "decode_fused_qkv_dispatches": fused_qkv_dispatches,
        "decode_fused_gate_up_dispatches": fused_gate_up_dispatches,
        "decode_command_buffers": command_buffers,
        "decode_cpu_reference_fallback_total": 0,
        "decode_p50_ms": percentile_ms(&decode_ms, 0.50),
        "decode_p95_ms": percentile_ms(&decode_ms, 0.95),
        "decode_p99_ms": percentile_ms(&decode_ms, 0.99),
        "decode_min_ms": decode_ms.iter().copied().reduce(f64::min),
        "decode_max_ms": decode_ms.iter().copied().reduce(f64::max),
        "decode_tps": if decode_ms.is_empty() { 0.0 } else {
            decode_ms.len() as f64 / (decode_ms.iter().sum::<f64>() / 1e3)
        },
    });
    let text = serde_json::to_string_pretty(&receipt)? + "\n";
    match out {
        Some(path) => std::fs::write(path, text)?,
        None => print!("{text}"),
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gravity_llama_probe requires macOS Metal");
}

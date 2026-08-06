//! BASE_TRUE_TPS for the GLM-5.2 GPU-resident Gravity or activation-aware path.
//!
//! Companion to `gravity_tps.rs` (Llama). Two things do not carry over from
//! the dense case: device-resident bytes grow with the run instead of being
//! fixed at load (only touched experts are ever uploaded), and content
//! affects cost, since different tokens can route to different experts.
//! Both are called out in the receipt rather than assumed away.
//!
//! Defaults are deliberately small -- this is a first honest measurement on
//! an 83 GB, 78-layer MoE artifact, not a scoreboard run. Scale up with
//! `--context` once the small numbers are in.
//!
//!     cargo run --release -p hawking-core --example gravity_glm_tps -- \
//!         --context 16 --decode 8 --out receipt.json
//!
//! Long warm / residency-curve run (additive flag, off by default):
//!
//!     cargo run --release -p hawking-core --example gravity_glm_tps -- \
//!         --context 4 --decode 80 --token-curve --out warm.json

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[cfg(target_os = "macos")]
use sha2::{Digest, Sha256};

const DEFAULT_MODEL_DIR: &str = "Library/Application Support/Hawking/Models/GLM-5.2/\
    b4734de4facf877f85769a911abafc5283eab3d9/General-R0";

#[cfg(target_os = "macos")]
fn artifact_index(
    dir: &std::path::Path,
) -> Result<(String, String, &'static str), Box<dyn std::error::Error>> {
    const CANDIDATES: [(&str, &str); 2] = [
        ("model.gravity.index.json", "gravity"),
        ("model.activation_aware.index.json", "activation_aware"),
    ];
    let present: Vec<_> = CANDIDATES
        .iter()
        .filter_map(|&(name, representation)| {
            let path = dir.join(name);
            path.is_file().then_some((path, name, representation))
        })
        .collect();
    if present.len() != 1 {
        return Err(format!(
            "{}: expected exactly one supported model index, found {}",
            dir.display(),
            present.len()
        )
        .into());
    }
    let (path, name, representation) = &present[0];
    let digest = Sha256::digest(std::fs::read(path)?);
    Ok(((*name).to_string(), format!("{digest:x}"), representation))
}

#[cfg(target_os = "macos")]
fn decode_output(
    logits: &[f32],
    trace: &hawking_core::gravity_glm::GlmTrace,
    vocab: usize,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    if logits.is_empty() {
        let token = trace
            .sample_token
            .ok_or("token-only head returned neither logits nor trace.sample_token")?;
        return Ok(serde_json::json!({
            "mode": "token_plus_topk_diagnostics",
            "token": token,
            "full_logits_readback": false,
            "topk_indices": trace.head_topk_idx,
            "topk_values": trace.head_topk_val,
        }));
    }
    if logits.len() != vocab {
        return Err(format!(
            "decode output has {} logits; expected 0 (token-only) or {vocab}",
            logits.len()
        )
        .into());
    }
    let token = trace.sample_token.unwrap_or_else(|| {
        logits
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    });
    Ok(serde_json::json!({
        "mode": "full_vocab_logits",
        "token": token,
        "full_logits_readback": true,
        "logit_count": logits.len(),
        "topk_indices": trace.head_topk_idx,
        "topk_values": trace.head_topk_val,
    }))
}

#[cfg(target_os = "macos")]
fn env_snapshot() -> serde_json::Value {
    const KEYS: [&str; 15] = [
        "HAWKING_GLM_GPU_RESIDENT_STATE",
        "HAWKING_GLM_GPU_COMPACT_MLA",
        "HAWKING_GLM_GPU_DEVICE_DSA",
        "HAWKING_GLM_GPU_COMPACT_ATTENTION_ICB",
        "HAWKING_GLM_GPU_DEVICE_ROUTER",
        "HAWKING_GLM_GPU_LM_HEAD",
        "HAWKING_GLM_GPU_LM_HEAD_ICB",
        "HAWKING_GLM_GPU_LM_HEAD_FULL_LOGITS",
        "HAWKING_GLM_GPU_EXPERT_WAVE",
        "HAWKING_GLM_GPU_EXPERT_WAVE_CONCURRENT",
        "HAWKING_GLM_GPU_EXPERT_TABLE_HIT",
        "HAWKING_GLM_GPU_EXPERT_TABLE_ICB",
        "HAWKING_GRAVITY_GPU_CACHE_BUDGET_BYTES",
        "HAWKING_TCB_TRACE",
        "HAWKING_COST_LEDGER",
    ];
    let mut out = serde_json::Map::new();
    for key in KEYS {
        out.insert(
            key.to_string(),
            std::env::var_os(key)
                .map(|v| serde_json::Value::String(v.to_string_lossy().into_owned()))
                .unwrap_or(serde_json::Value::Null),
        );
    }
    serde_json::Value::Object(out)
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hawking_core::gravity_glm::gpu::GravityGlmGpu;

    let mut dir: Option<PathBuf> = None;
    let mut contexts: Vec<usize> = Vec::new();
    let mut decode = 4usize;
    let mut out: Option<PathBuf> = None;
    let mut verify_hash = true;
    // Additive: when set, record residency/eviction after every decode token
    // and emit a per-token curve in the receipt. Default off so short
    // scoreboard runs stay byte-identical in shape.
    let mut token_curve = false;
    // Additive: path for progressive JSONL (one line per decode token).
    // Lets a detached long run be polled without waiting for the final receipt.
    let mut progress: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dir" => dir = args.next().map(PathBuf::from),
            "--context" => contexts.push(args.next().ok_or("--context needs a value")?.parse()?),
            "--decode" => decode = args.next().ok_or("--decode needs a value")?.parse()?,
            "--out" => out = args.next().map(PathBuf::from),
            "--no-verify-hash" => verify_hash = false,
            "--token-curve" => token_curve = true,
            "--progress" => progress = args.next().map(PathBuf::from),
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    if contexts.is_empty() {
        contexts = vec![8, 16];
    }
    let dir = dir.unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(DEFAULT_MODEL_DIR)
    });
    if !dir.is_dir() {
        return Err(format!("no model directory at {dir:?}").into());
    }
    let (artifact_index_name, artifact_index_sha256, representation) = artifact_index(&dir)?;

    let mut progress_file = match &progress {
        Some(p) => {
            if let Some(parent) = p.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            Some(std::fs::File::create(p)?)
        }
        None => None,
    };

    eprintln!("opening (indexing shard headers, decoding nothing)...");
    eprintln!("verify_hash={verify_hash} token_curve={token_curve}");
    let t_open = Instant::now();
    let model = GravityGlmGpu::open_dir(&dir, verify_hash)?;
    let open_ms = t_open.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "opened in {open_ms:.0} ms on {} | layers={} hidden={} experts={} vocab={}",
        model.device_name(),
        model.arch.n_layers,
        model.arch.hidden,
        model.arch.n_routed_experts,
        model.arch.vocab_size
    );
    {
        let c = model.cache_stats();
        eprintln!(
            "cache budget {} bytes ({:.1} GiB)",
            c.budget_bytes,
            c.budget_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    // Deterministic pseudo-token stream inside the vocabulary, continued
    // (not restarted) from prefill into decode so a run is reproducible.
    let vocab = model.arch.vocab_size as u64;
    let stream = |n: usize| -> Vec<u32> {
        (0..n)
            .map(|i| ((i as u64 * 2_654_435_761) % vocab) as u32)
            .collect()
    };

    let mut rows = Vec::new();
    for &context in &contexts {
        let tokens = stream(context + decode);

        let cache_before = model.cache_stats();
        let t_prefill = Instant::now();
        model.forward(&tokens[..context])?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
        let cache_after_prefill = model.cache_stats();
        eprintln!(
            "ctx {context} prefill done in {prefill_ms:.0} ms | resident {:.2} GB entries {} evictions {}",
            cache_after_prefill.resident_bytes as f64 / 1e9,
            cache_after_prefill.entries,
            cache_after_prefill.evictions,
        );

        let mut decode_ms_each = Vec::with_capacity(decode);
        // A resident GLM step emits several physical command buffers.  Count
        // the completed waits directly from the resident session, rather than
        // reporting an invented fixed dispatch count.  This remains enabled
        // in the untraced benchmark path, so it does not turn the speed run
        // into a per-dispatch tracing run.
        let mut command_buffer_waits_each = Vec::with_capacity(decode);
        let mut curve = Vec::with_capacity(decode);
        let mut output_modes = std::collections::BTreeSet::new();
        for (i, &t) in tokens[context..].iter().enumerate() {
            let waits_before = model.last_resident_waits();
            let t0 = Instant::now();
            let (logits, trace) = model.forward_at(&[t], context + i)?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            decode_ms_each.push(ms);
            let command_buffer_waits = match (waits_before, model.last_resident_waits()) {
                (Some(before), Some(after)) => Some(after.saturating_sub(before)),
                _ => None,
            };
            command_buffer_waits_each.push(command_buffer_waits);
            let output = decode_output(&logits, &trace, model.arch.vocab_size)?;
            output_modes.insert(output["mode"].as_str().unwrap_or("unknown").to_string());

            let cache = model.cache_stats();
            let tok_index = i + 1; // 1-based decode token index within this run
            let sample = serde_json::json!({
                "decode_token_index": tok_index,
                "absolute_position": context + i,
                "ms": ms,
                "tps": 1000.0 / ms,
                "resident_bytes": cache.resident_bytes,
                "high_water_bytes": cache.high_water_bytes,
                "entries": cache.entries,
                "evictions": cache.evictions,
                "command_buffer_waits": command_buffer_waits,
                "output": output,
            });
            if token_curve {
                curve.push(sample.clone());
            }
            if let Some(f) = progress_file.as_mut() {
                let mut line = sample;
                line.as_object_mut()
                    .unwrap()
                    .insert("context_tokens".into(), serde_json::json!(context));
                line.as_object_mut()
                    .unwrap()
                    .insert("verify_hash".into(), serde_json::json!(verify_hash));
                writeln!(f, "{}", serde_json::to_string(&line)?)?;
                f.flush()?;
            }
            eprintln!(
                "  tok {tok_index:>3}/{decode}  {ms:>8.1} ms  {:.4} tok/s  resident {:.2} GB  entries {}  evictions {}",
                1000.0 / ms,
                cache.resident_bytes as f64 / 1e9,
                cache.entries,
                cache.evictions,
            );
        }
        let decode_ms: f64 = decode_ms_each.iter().sum();
        let mut sorted = decode_ms_each.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut row = serde_json::json!({
            "context_tokens": context,
            "decode_tokens": decode,
            "prefill_ms": prefill_ms,
            "prefill_tps": context as f64 / (prefill_ms / 1e3),
            "cache_before_prefill": {
                "resident_bytes": cache_before.resident_bytes,
                "entries": cache_before.entries,
                "evictions": cache_before.evictions,
            },
            "cache_after_prefill": {
                "resident_bytes": cache_after_prefill.resident_bytes,
                "entries": cache_after_prefill.entries,
                "evictions": cache_after_prefill.evictions,
            },
            "decode_ms": decode_ms,
            "base_true_decode_tps": decode as f64 / (decode_ms / 1e3),
            "decode_ms_per_token_median": sorted.get(sorted.len() / 2).copied().unwrap_or(0.0),
            "decode_ms_per_token_min": sorted.first().copied().unwrap_or(0.0),
            "decode_ms_per_token_max": sorted.last().copied().unwrap_or(0.0),
            "decode_ms_per_token_all": decode_ms_each,
            "device_execution": {
                "backend": "metal",
                "device_name": model.device_name(),
                "resident_state": model.resident_state_enabled(),
                "command_buffer_waits_per_token_all": command_buffer_waits_each,
                "command_buffer_waits_metric": "completed commit_and_wait calls during each decode token; null means the resident path was not active",
            },
            "output_modes": output_modes,
        });
        if token_curve {
            row.as_object_mut()
                .unwrap()
                .insert("token_curve".into(), serde_json::json!(curve));
        }
        rows.push(row);
        let last = rows.last().unwrap();
        eprintln!(
            "ctx {context:>4}  prefill {:>7.2} tok/s ({prefill_ms:>8.0} ms)  decode {:>7.2} tok/s  \
             ({:.1} ms/tok median)",
            last["prefill_tps"].as_f64().unwrap(),
            last["base_true_decode_tps"].as_f64().unwrap(),
            last["decode_ms_per_token_median"].as_f64().unwrap(),
        );
    }

    let cache = model.cache_stats();
    let receipt = serde_json::json!({
        "schema": "hawking.gravity.glm_base_tps.v1",
        "scoreboard": "BASE_TRUE_TPS",
        "artifact": {
            "representation": representation,
            "index": artifact_index_name,
            "index_sha256": artifact_index_sha256,
        },
        "verify_hash": verify_hash,
        "token_curve": token_curve,
        "device": {
            "backend": "metal",
            "name": model.device_name(),
            "resident_state": model.resident_state_enabled(),
        },
        "run_configuration": {
            "raw_environment": env_snapshot(),
            "resolved": {
                "resident_state": model.resident_state_enabled(),
                "compact_mla": hawking_core::gravity_glm::gpu_compact_mla_enabled(),
                "device_dsa": hawking_core::gravity_glm::gpu_device_dsa_enabled(),
                "compact_attention_icb": hawking_core::gravity_glm::gpu_compact_attention_icb_enabled(),
                "device_router": hawking_core::gravity_glm::gpu_device_router_enabled(),
                "gpu_native_bf16_head": hawking_core::gravity_glm::gpu_lm_head_enabled(),
                "gpu_lm_head_icb": hawking_core::gravity_glm::gpu_lm_head_icb_enabled(),
                "full_logits_readback": hawking_core::gravity_glm::gpu_lm_head_full_logits_enabled(),
                "expert_wave": hawking_core::gravity_glm::gpu_expert_wave_enabled(),
                "expert_wave_concurrent": hawking_core::gravity_glm::gpu_expert_wave_concurrent_enabled(),
                "expert_table_hit": hawking_core::gravity_glm::gpu_expert_table_hit_enabled(),
                "expert_table_icb": hawking_core::gravity_glm::gpu_expert_table_icb_enabled(),
                "cost_ledger": false,
            },
            "output_contract": "each token accepts either full vocab logits or promoted token + top-k diagnostics; actual mode is recorded per token",
        },
        "note": "measured, not modelled; true batch-1 and non-speculative. Every admitted \
                 runtime flag is recorded above rather than assumed. GLM's routed MoE means \
                 device-resident bytes grow with the run instead of being fixed at load, and \
                 cost is mildly content-dependent. Resident set is byte-budgeted LRU \
                 (HAWKING_GRAVITY_GPU_CACHE_BUDGET_BYTES); high_water is the peak observed \
                 this process, not modelled capacity.",
        "model_dir": dir.to_string_lossy(),
        "architecture": {
            "layers": model.arch.n_layers,
            "hidden": model.arch.hidden,
            "routed_experts": model.arch.n_routed_experts,
            "experts_per_tok": model.arch.num_experts_per_tok,
            "vocab": model.arch.vocab_size,
        },
        "gpu_weight_cache": {
            "budget_bytes": cache.budget_bytes,
            "resident_bytes": cache.resident_bytes,
            "high_water_bytes": cache.high_water_bytes,
            "entries": cache.entries,
            "evictions": cache.evictions,
            "budget_env": "HAWKING_GRAVITY_GPU_CACHE_BUDGET_BYTES",
        },
        "open_ms": open_ms,
        "measurements": rows,
    });
    let text = serde_json::to_string_pretty(&receipt)? + "\n";
    match out {
        Some(p) => {
            if let Some(parent) = p.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            std::fs::write(&p, &text)?;
            eprintln!("wrote {}", p.display());
        }
        None => print!("{text}"),
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gravity_glm_tps measures the Metal runtime; it only runs on macOS");
}

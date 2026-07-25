//! BASE_TRUE_TPS: measured throughput of the resident-`.gravity` runtime.
//!
//! This is the base scoreboard the campaign requires to be kept separate
//! from any accelerated number. Nothing here is modelled or extrapolated:
//! every figure is a wall-clock measurement of the same code path the
//! parity test grades against the frozen oracle, on this machine, with the
//! artifact resident on the device.
//!
//! Prefill and decode are split from a single run's per-token series rather
//! than measured in two passes, because both phases execute the same code
//! against a growing cache -- running decode "fresh" would time a cache
//! state that never occurs in service.
//!
//!     cargo run --release -p hawking-core --example gravity_tps -- \
//!         --context 512 --decode 16 --out receipt.json

use std::path::PathBuf;
use std::time::Instant;

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hawking_core::gravity_llama::gpu::GravityLlamaGpu;
    use hawking_core::metal::MetalContext;

    let mut artifact: Option<PathBuf> = None;
    let mut contexts: Vec<usize> = Vec::new();
    let mut decode = 16usize;
    let mut out: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--artifact" => artifact = args.next().map(PathBuf::from),
            "--context" => contexts.push(args.next().ok_or("--context needs a value")?.parse()?),
            "--decode" => decode = args.next().ok_or("--decode needs a value")?.parse()?,
            "--out" => out = args.next().map(PathBuf::from),
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    if contexts.is_empty() {
        contexts = vec![128, 512, 2048];
    }
    let artifact = artifact.unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("HOME").expect("HOME"))
            .join("Library/Application Support/Hawking/CampaignS08/llama32-1b-R0.v2.gravity")
    });
    if !artifact.is_file() {
        return Err(format!("no artifact at {artifact:?}").into());
    }
    let artifact_bytes = std::fs::metadata(&artifact)?.len();

    // Cold load includes SHA-256 verification of every tensor -- that is
    // what a real first load does, so reporting it without would be
    // reporting a load nobody performs. Warm load is the same work with the
    // file already in the page cache, which is the honest repeat-open cost.
    // Each open builds its own Metal context, exactly as a real first load
    // does; sharing one would measure a warm load twice.
    let cold = GravityLlamaGpu::open_with(MetalContext::new()?, &artifact, true)?;
    let cold_load_ms = cold.load_ms;
    let device_bytes = cold.device_bytes;
    drop(cold);
    let model = GravityLlamaGpu::open_with(MetalContext::new()?, &artifact, true)?;
    let warm_load_ms = model.load_ms;

    // Deterministic pseudo-token stream inside the vocabulary. Content does
    // not affect cost on a dense model -- every token touches every weight --
    // so a fixed stream keeps the measurement reproducible.
    let vocab = model.arch.vocab_size as u64;
    let stream = |n: usize| -> Vec<u32> {
        (0..n)
            .map(|i| ((i as u64 * 2_654_435_761) % vocab) as u32)
            .collect()
    };

    let mut rows = Vec::new();
    for &context in &contexts {
        let tokens = stream(context + decode);
        let t0 = Instant::now();
        let (logits, stats) = model.forward(&tokens)?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        assert_eq!(logits.len(), model.arch.vocab_size);

        let prefill_ms: f64 = stats.per_token_ms[..context].iter().sum();
        let decode_ms: f64 = stats.per_token_ms[context..].iter().sum();
        let mut tail = stats.per_token_ms[context..].to_vec();
        tail.sort_by(|a, b| a.partial_cmp(b).unwrap());

        rows.push(serde_json::json!({
            "context_tokens": context,
            "decode_tokens": decode,
            "time_to_first_token_ms": stats.first_token_ms,
            "prefill_ms": prefill_ms,
            "prefill_tps": context as f64 / (prefill_ms / 1e3),
            "decode_ms": decode_ms,
            "base_true_decode_tps": decode as f64 / (decode_ms / 1e3),
            "decode_ms_per_token_median": tail[tail.len() / 2],
            "decode_ms_per_token_min": tail[0],
            "decode_ms_per_token_max": tail[tail.len() - 1],
            "command_buffers_per_token": stats.command_buffers as f64 / stats.tokens as f64,
            "dispatches_per_token": stats.dispatches as f64 / stats.tokens as f64,
            "wall_ms": wall_ms,
        }));
        let last = rows.last().unwrap();
        eprintln!(
            "ctx {context:>5}  ttft {:>8.1} ms  prefill {:>7.2} tok/s  decode {:>7.2} tok/s  \
             ({:.1} ms/tok)",
            stats.first_token_ms,
            last["prefill_tps"].as_f64().unwrap(),
            last["base_true_decode_tps"].as_f64().unwrap(),
            last["decode_ms_per_token_median"].as_f64().unwrap(),
        );
    }

    // KV bytes are exact, not estimated: two f32 caches of
    // n_kv_heads * head_dim per layer per position.
    let a = &model.arch;
    let kv_bytes_per_token = 2 * a.n_layers * a.n_kv_heads * a.head_dim * 4;
    let receipt = serde_json::json!({
        "schema": "hawking.gravity.base_tps.v1",
        "scoreboard": "BASE_TRUE_TPS",
        "note": "measured, not modelled; no acceleration of any kind is enabled on this path",
        "artifact": artifact.to_string_lossy(),
        "artifact_bytes": artifact_bytes,
        "device_resident_bytes": device_bytes,
        "active_bytes_per_token": device_bytes,
        "architecture": {
            "layers": a.n_layers, "hidden": a.hidden, "heads": a.n_heads,
            "kv_heads": a.n_kv_heads, "head_dim": a.head_dim, "vocab": a.vocab_size,
        },
        "kv_bytes_per_token": kv_bytes_per_token,
        "cold_load_ms": cold_load_ms,
        "warm_load_ms": warm_load_ms,
        "measurements": rows,
    });
    let text = serde_json::to_string_pretty(&receipt)? + "\n";
    match out {
        Some(p) => std::fs::write(&p, &text)?,
        None => print!("{text}"),
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gravity_tps measures the Metal runtime; it only runs on macOS");
}

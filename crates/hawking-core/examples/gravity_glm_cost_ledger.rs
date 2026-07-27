//! Per-token Temporal Gravity cost ledger for the sealed GLM-5.2 Math-Preserve
//! GPU path.
//!
//! Attributes every microsecond of one or more decode tokens so exclusive CPU
//! buckets + an explicit **unattributed** line sum to wall time, and a separate
//! **device timeline** reports GPU execution vs queue wait from Metal
//! `GPUStartTime`/`GPUEndTime`. Reports p50/p95/p99 across ledger tokens and
//! discloses profiler overhead.
//!
//! Default-off instrumentation (`HAWKING_COST_LEDGER=1`). Does not change
//! decode output. This binary only runs on macOS with Metal; the executor
//! sandbox cannot create a Metal device, so hand the controller this command.
//!
//! Full multi-token ledger (warm, distributions):
//!
//! ```text
//! HAWKING_COST_LEDGER=1 cargo run --release -p hawking-core \
//!   --example gravity_glm_cost_ledger -- \
//!   --context 4 --warmup 8 --ledger-tokens 16 \
//!   --out reports/base_runtime/tg_cost_ledger_warm.json
//! ```
//!
//! Single-token warm ledger:
//!
//! ```text
//! HAWKING_COST_LEDGER=1 cargo run --release -p hawking-core \
//!   --example gravity_glm_cost_ledger -- \
//!   --context 4 --warmup 8 --ledger-tokens 1 \
//!   --out reports/base_runtime/cost_ledger_warm.json
//! ```
//!
//! Cold first-token ledger (no warm-up; first-touch loads dominate):
//!
//! ```text
//! HAWKING_COST_LEDGER=1 cargo run --release -p hawking-core \
//!   --example gravity_glm_cost_ledger -- \
//!   --context 4 --warmup 0 --ledger-tokens 1 --no-verify-hash \
//!   --out reports/base_runtime/cost_ledger_cold_nohash.json
//! ```

use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_MODEL_DIR: &str = "Library/Application Support/Hawking/Models/GLM-5.2/\
    b4734de4facf877f85769a911abafc5283eab3d9/General-R0";

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hawking_core::cost_ledger::{
        self, aggregate_reports, bucket_source_catalogue, sealed_glm_active_byte_schedule,
        SEALED_ARTIFACT_ACTIVE_ROUTED_BYTES,
    };
    use hawking_core::gravity_glm::gpu::GravityGlmGpu;

    // Force the ledger on for this process even if the env var was omitted
    // (the example's whole point is the ledger).
    cost_ledger::set_enabled(true);

    let mut dir: Option<PathBuf> = None;
    let mut context = 4usize;
    let mut warmup = 8usize;
    let mut ledger_tokens = 1usize;
    let mut out: Option<PathBuf> = None;
    let mut verify_hash = true;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dir" => dir = args.next().map(PathBuf::from),
            "--context" => context = args.next().ok_or("--context needs a value")?.parse()?,
            "--warmup" => warmup = args.next().ok_or("--warmup needs a value")?.parse()?,
            "--ledger-tokens" => {
                ledger_tokens = args.next().ok_or("--ledger-tokens needs a value")?.parse()?
            }
            "--out" => out = args.next().map(PathBuf::from),
            "--no-verify-hash" => verify_hash = false,
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    let dir = dir.unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(DEFAULT_MODEL_DIR)
    });
    if !dir.is_dir() {
        return Err(format!("no model directory at {dir:?}").into());
    }

    eprintln!(
        "TG cost ledger | verify_hash={verify_hash} context={context} warmup={warmup} \
         ledger_tokens={ledger_tokens}"
    );
    eprintln!("model dir: {}", dir.display());

    let t_open = Instant::now();
    let model = GravityGlmGpu::open_dir(&dir, verify_hash)?;
    let open_ms = t_open.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "opened in {open_ms:.0} ms | layers={} hidden={} experts_per_tok={} vocab={}",
        model.arch.n_layers,
        model.arch.hidden,
        model.arch.num_experts_per_tok,
        model.arch.vocab_size
    );

    let vocab = model.arch.vocab_size as u64;
    let stream = |n: usize| -> Vec<u32> {
        (0..n)
            .map(|i| ((i as u64 * 2_654_435_761) % vocab) as u32)
            .collect()
    };
    let total = context + warmup + ledger_tokens;
    let tokens = stream(total);

    // Prefill: not ledgered (the gate asks for decode-token attribution).
    let t_prefill = Instant::now();
    model.forward(&tokens[..context])?;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
    eprintln!("prefill {context} tokens in {prefill_ms:.0} ms");

    // Warm-up decode tokens without the ledger (residency curve, not cost).
    for (i, &t) in tokens[context..context + warmup].iter().enumerate() {
        let t0 = Instant::now();
        let _ = model.forward_at(&[t], context + i)?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "  warmup {:>3}/{}  {ms:>8.1} ms  {:.4} tok/s",
            i + 1,
            warmup,
            1000.0 / ms
        );
    }

    let mut token_reports = Vec::with_capacity(ledger_tokens);
    let mut raw_reports = Vec::with_capacity(ledger_tokens);
    for (i, &t) in tokens[context + warmup..].iter().enumerate() {
        let abs_pos = context + warmup + i;
        let wall = Instant::now();
        let (logits, _trace, report) = model.forward_at_with_ledger(&[t], abs_pos)?;
        let wall_ms = wall.elapsed().as_secs_f64() * 1e3;
        assert_eq!(logits.len(), model.arch.vocab_size);

        let mut report = report.ok_or("ledger enabled but no report returned")?;

        // Residency snapshot from the live GPU weight cache (not a proxy).
        let cache = model.cache_stats();
        // Re-open a micro begin/end is wrong; stamp residency via a side
        // channel: the report is already finished. Attach via mutation of
        // the counters we own in the JSON path below, and also patch the
        // struct for aggregation.
        report.counters.residency_bytes = Some(cache.resident_bytes);
        report.counters.residency_entries = Some(cache.entries as u64);
        report.counters.residency_evictions = Some(cache.evictions);

        eprintln!(
            "  ledger {:>3}/{}  wall {:>8.1} ms  attributed {:.1}%  unattributed_us={}  \
             cbs={} disp={} syncs={} gpu_exec_us={} gpu_q_us={:?} active_bytes={:.3} GB  \
             profiler_oh_us={} ({:.2}%)",
            i + 1,
            ledger_tokens,
            wall_ms,
            report.attributed_fraction * 100.0,
            report.unattributed_us,
            report.counters.command_buffers_submitted,
            report.counters.dispatches_encoded,
            report.counters.synchronization_points,
            report.device.gpu_execution_us,
            report.device.gpu_queue_wait_us,
            report.counters.active_bytes_read as f64 / 1e9,
            report.profiler_overhead_us,
            report.profiler_overhead_fraction * 100.0,
        );
        // Print bucket table for the operator.
        eprintln!("  buckets (us, exclusive CPU):");
        let mut keys: Vec<_> = report.buckets_us.keys().cloned().collect();
        keys.sort();
        for k in keys {
            let us = report.buckets_us[&k].as_u64().unwrap_or(0);
            if us == 0 {
                continue;
            }
            let pct = 100.0 * us as f64 / report.wall_us.max(1) as f64;
            eprintln!("    {k:<36} {us:>12}  ({pct:5.1}%)");
        }
        eprintln!(
            "    {:<36} {:>12}  ({:5.1}%)",
            report.unattributed_name,
            report.unattributed_us,
            100.0 * report.unattributed_us as f64 / report.wall_us.max(1) as f64
        );
        eprintln!(
            "  device: gpu_execution_us={}  gpu_queue_wait_us={:?}  \
             ts_observed={} ts_missing={} counter_supported={:?} counter_samples={}",
            report.device.gpu_execution_us,
            report.device.gpu_queue_wait_us,
            report.device.gpu_timestamps_observed,
            report.device.gpu_timestamps_missing,
            report.device.counter_sample_supported,
            report.device.counter_samples_recorded,
        );
        for n in &report.device.notes {
            eprintln!("    note: {n}");
        }
        if let Some(pf) = report.counters.page_faults_minor {
            eprintln!(
                "  page_faults: minor={} major={:?} (getrusage)",
                pf, report.counters.page_faults_major
            );
        } else {
            eprintln!("  page_faults: unavailable on this path");
        }

        // Active-byte category partition (must sum to active_bytes_read).
        let cat = &report.counters.active_bytes_by_category;
        if !cat.is_empty() {
            eprintln!("  active_bytes_by_category (GB):");
            let mut keys: Vec<_> = cat.keys().cloned().collect();
            keys.sort();
            for k in keys {
                let b = cat[&k].as_u64().unwrap_or(0);
                if b == 0 {
                    continue;
                }
                eprintln!("    {k:<20} {:>8.3}", b as f64 / 1e9);
            }
        }

        token_reports.push(serde_json::json!({
            "decode_token_index": i + 1,
            "absolute_position": abs_pos,
            "external_wall_ms": wall_ms,
            "ledger": report.to_json_value(),
        }));
        raw_reports.push(report);
    }

    let aggregate = if raw_reports.is_empty() {
        None
    } else {
        Some(aggregate_reports(&raw_reports))
    };

    let cache = model.cache_stats();
    let receipt = serde_json::json!({
        "schema": "hawking.gravity.per_token_cost_ledger_run.v2",
        "gate": "TEMPORAL_GRAVITY / BASE_RUNTIME_MAXIMIZED",
        "deliverable": "per-token cost ledger with device timeline + p50/p95/p99",
        "verify_hash": verify_hash,
        "model_dir": dir.to_string_lossy(),
        "architecture": {
            "layers": model.arch.n_layers,
            "hidden": model.arch.hidden,
            "routed_experts": model.arch.n_routed_experts,
            "experts_per_tok": model.arch.num_experts_per_tok,
            "vocab": model.arch.vocab_size,
        },
        "geometry_active_routed_bytes_sealed": SEALED_ARTIFACT_ACTIVE_ROUTED_BYTES,
        "geometry_gb": SEALED_ARTIFACT_ACTIVE_ROUTED_BYTES as f64 / 1e9,
        "static_active_byte_schedule": sealed_glm_active_byte_schedule(),
        "static_schedule_note": "Header-derived static expectation for active_bytes_read. Compare aggregate counters_mean.active_bytes_read and per-token active_bytes_by_category against it. A gap is a finding.",
        "open_ms": open_ms,
        "prefill_ms": prefill_ms,
        "context_tokens": context,
        "warmup_decode_tokens": warmup,
        "ledger_tokens": ledger_tokens,
        "bucket_sources": bucket_source_catalogue(),
        "gpu_weight_cache": {
            "budget_bytes": cache.budget_bytes,
            "resident_bytes": cache.resident_bytes,
            "high_water_bytes": cache.high_water_bytes,
            "entries": cache.entries,
            "evictions": cache.evictions,
        },
        "aggregate": aggregate,
        "tokens": token_reports,
        "notes": [
            "Exclusive stack timing: nested metal/verify/transfer steal time from parent semantic scopes so sum(buckets)+unattributed ≈ wall.",
            "unattributed is a first-class line with its own magnitude — never absorbed into cpu_residual_scoped or any neighbour.",
            "cpu_residual_scoped is only what callers explicitly Scope; it is not the wall remainder.",
            "Device timeline (gpu_execution_us, gpu_queue_wait_us) is independent of exclusive CPU buckets; GPU work overlaps metal_synchronize_cpu_wait.",
            "gpu_queue_wait_us = sum max(0, host_wait_us - gpu_execution_us) per CB when GPUStartTime/GPUEndTime are readable; otherwise None (no CPU proxy).",
            "Metal timestamp counter samples are opt-in (markers must be encoded). This path records CB-level GPUStartTime/GPUEndTime only unless a counter probe is wired.",
            "page_faults from getrusage(RUSAGE_SELF) minflt/majflt deltas when available.",
            "profiler_overhead_us is ledger bookkeeping cost disclosed on every token.",
            "p50/p95/p99 live under aggregate.* when ledger-tokens > 1.",
            "Routed experts and the shared expert are co-batched into command buffers; metal_* owns that GPU host-wait, shared_experts owns only the CPU residual add.",
            "Attention (including DSA IndexShare) runs on the CPU today; its matvecs still go through the GPU PQ path and land in metal_*.",
            "Bucket::Norm is defined for RMSNorm/LayerNorm hooks; wire Scope::new(Bucket::Norm) from gravity_glm when that lane is free.",
            "This binary does not claim timings it did not measure; empty buckets mean that work was not observed on the instrumented path.",
            "No Metal device in the agent sandbox — live numbers require running this command on a Mac with the model present.",
            "active_bytes_by_category partitions active_bytes_read by tensor class (routed/shared/dense_mlp/attention/indexer/router/lm_head/other).",
            "Geometry (2.58 GB) is routed-experts only under an 8×3×78 idealisation; the static schedule is the full forward touch set (~9.34 GB).",
        ],
        "hooks_not_yet_in_gravity_glm_decode": [
            "Scope::new(Bucket::Norm) around RMSNorm/LayerNorm",
            "record_operations(n) for attention cells / FMA estimates",
            "record_counter_sample_capability + encode timestamp counter markers if device supports them",
            "record_residency inside the token (example stamps post-hoc from cache_stats)",
        ],
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

    if let Some(agg) = receipt.get("aggregate") {
        eprintln!("\n=== aggregate (p50 / p95 / p99) ===");
        if let Some(w) = agg.get("wall_us") {
            eprintln!(
                "wall_us:     p50={:.0}  p95={:.0}  p99={:.0}  mean={:.0}",
                w["p50"].as_f64().unwrap_or(0.0),
                w["p95"].as_f64().unwrap_or(0.0),
                w["p99"].as_f64().unwrap_or(0.0),
                w["mean"].as_f64().unwrap_or(0.0),
            );
        }
        if let Some(u) = agg.get("unattributed_us") {
            eprintln!(
                "unattributed p50={:.0}  p95={:.0}  p99={:.0}",
                u["p50"].as_f64().unwrap_or(0.0),
                u["p95"].as_f64().unwrap_or(0.0),
                u["p99"].as_f64().unwrap_or(0.0),
            );
        }
        if let Some(g) = agg.get("device_gpu_execution_us") {
            eprintln!(
                "gpu_exec_us  p50={:.0}  p95={:.0}  p99={:.0}",
                g["p50"].as_f64().unwrap_or(0.0),
                g["p95"].as_f64().unwrap_or(0.0),
                g["p99"].as_f64().unwrap_or(0.0),
            );
        }
        if let Some(oh) = agg.get("profiler_overhead_us") {
            eprintln!(
                "profiler_oh  p50={:.0}  mean={:.0}",
                oh["p50"].as_f64().unwrap_or(0.0),
                oh["mean"].as_f64().unwrap_or(0.0),
            );
        }
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gravity_glm_cost_ledger measures the Metal runtime; it only runs on macOS");
}

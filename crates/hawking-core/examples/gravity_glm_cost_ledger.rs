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
//!   --dir /absolute/path/GLM-5.2-H0.98-Math-Preserve.gravity \
//!   --context 4 --warmup 8 --ledger-tokens 16 \
//!   --out reports/base_runtime/tg_cost_ledger_warm.json
//! ```
//!
//! Single-token warm ledger:
//!
//! ```text
//! HAWKING_COST_LEDGER=1 cargo run --release -p hawking-core \
//!   --example gravity_glm_cost_ledger -- \
//!   --dir /absolute/path/GLM-5.2-H0.98-Math-Preserve.gravity \
//!   --context 4 --warmup 8 --ledger-tokens 1 \
//!   --out reports/base_runtime/cost_ledger_warm.json
//! ```
//!
//! Cold first-token ledger (no warm-up; first-touch loads dominate):
//!
//! ```text
//! HAWKING_COST_LEDGER=1 cargo run --release -p hawking-core \
//!   --example gravity_glm_cost_ledger -- \
//!   --dir /absolute/path/GLM-5.2-H0.98-Math-Preserve.gravity \
//!   --context 4 --warmup 0 --ledger-tokens 1 --no-verify-hash \
//!   --out reports/base_runtime/cost_ledger_cold_nohash.json
//! ```

use std::path::PathBuf;
use std::time::Instant;

const EXPECTED_ARTIFACT_DIR_NAME: &str = "GLM-5.2-H0.98-Math-Preserve.gravity";

#[cfg(target_os = "macos")]
#[derive(Debug, PartialEq, Eq, serde::Serialize)]
struct CacheDelta {
    resident_bytes: i128,
    entries: i128,
    evictions: i128,
    high_water_bytes: i128,
}

#[cfg(target_os = "macos")]
fn cache_delta(
    before: hawking_core::gravity_glm::GpuWeightCacheStats,
    after: hawking_core::gravity_glm::GpuWeightCacheStats,
) -> CacheDelta {
    CacheDelta {
        resident_bytes: after.resident_bytes as i128 - before.resident_bytes as i128,
        entries: after.entries as i128 - before.entries as i128,
        evictions: after.evictions as i128 - before.evictions as i128,
        high_water_bytes: after.high_water_bytes as i128 - before.high_water_bytes as i128,
    }
}

#[cfg(target_os = "macos")]
fn cache_snapshot(stats: hawking_core::gravity_glm::GpuWeightCacheStats) -> serde_json::Value {
    serde_json::json!({
        "budget_bytes": stats.budget_bytes,
        "resident_bytes": stats.resident_bytes,
        "high_water_bytes": stats.high_water_bytes,
        "entries": stats.entries,
        "evictions": stats.evictions,
    })
}

#[cfg(target_os = "macos")]
fn physical_identity(
    interval_hex: char,
    role: &str,
) -> hawking_core::Result<hawking_core::metal::PhysicalTraceIdentity> {
    hawking_core::metal::PhysicalTraceIdentity::new(
        interval_hex.to_string().repeat(64),
        "c".repeat(64),
        "cost_ledger_ab".into(),
        role.into(),
        None,
        0,
    )
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
    const KEYS: [&str; 7] = [
        "HAWKING_GLM_GPU_RESIDENT_STATE",
        "HAWKING_GLM_GPU_LM_HEAD",
        "HAWKING_GLM_GPU_LM_HEAD_FULL_LOGITS",
        "HAWKING_GLM_GPU_EXPERT_WAVE",
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
    use hawking_core::cost_ledger::{
        self, aggregate_reports, bucket_source_catalogue, math_preserve_active_byte_contract,
    };
    use hawking_core::gravity_glm::gpu::GravityGlmGpu;

    // Force the ledger on for this process even if the env var was omitted
    // (the example's whole point is the ledger).
    cost_ledger::set_enabled(true);

    let mut dir: Option<PathBuf> = None;
    let mut context = 4usize;
    let mut warmup = 8usize;
    let mut ledger_tokens = 1usize;
    let mut ab_context = 1usize;
    let mut out: Option<PathBuf> = None;
    let mut verify_hash = true;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dir" => dir = args.next().map(PathBuf::from),
            "--context" => context = args.next().ok_or("--context needs a value")?.parse()?,
            "--warmup" => warmup = args.next().ok_or("--warmup needs a value")?.parse()?,
            "--ledger-tokens" => {
                ledger_tokens = args
                    .next()
                    .ok_or("--ledger-tokens needs a value")?
                    .parse()?
            }
            "--ab-context" => {
                ab_context = args.next().ok_or("--ab-context needs a value")?.parse()?
            }
            "--out" => out = args.next().map(PathBuf::from),
            "--no-verify-hash" => verify_hash = false,
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    let dir = dir.ok_or(
        "--dir is required and must name the sealed GLM-5.2-H0.98-Math-Preserve.gravity artifact",
    )?;
    if !dir.is_dir() {
        return Err(format!("no model directory at {dir:?}").into());
    }
    if dir.file_name().and_then(|name| name.to_str()) != Some(EXPECTED_ARTIFACT_DIR_NAME) {
        return Err(format!(
            "profiler recovery is Math-Preserve-specific: --dir must end in \
             {EXPECTED_ARTIFACT_DIR_NAME:?}, got {dir:?}"
        )
        .into());
    }
    if ab_context == 0 {
        return Err("--ab-context must be at least 1".into());
    }
    if ledger_tokens == 0 {
        return Err("--ledger-tokens must be at least 1".into());
    }
    if let Some(mode) = std::env::var_os("HAWKING_TCB_TRACE") {
        let mode = mode.to_string_lossy();
        if !mode.is_empty() && mode != "off" && mode != "0" {
            return Err(format!(
                "HAWKING_TCB_TRACE={mode:?} changes command-buffer structure; unset it for a comparable cost-ledger A/B"
            )
            .into());
        }
    }

    eprintln!(
        "TG cost ledger | verify_hash={verify_hash} context={context} warmup={warmup} \
         ledger_tokens={ledger_tokens}"
    );
    eprintln!("model dir: {}", dir.display());

    let t_open = Instant::now();
    let model = GravityGlmGpu::open_dir(&dir, verify_hash)?;
    if !model.resident_state_enabled()
        || !hawking_core::gravity_glm::gpu_lm_head_enabled()
        || hawking_core::gravity_glm::gpu_lm_head_full_logits_enabled()
        || hawking_core::gravity_glm::gpu_expert_wave_enabled()
    {
        return Err(
            "this recovery harness requires the promoted token-only resident path: \
             HAWKING_GLM_GPU_RESIDENT_STATE=1 HAWKING_GLM_GPU_LM_HEAD=1 \
             HAWKING_GLM_GPU_LM_HEAD_FULL_LOGITS=0 HAWKING_GLM_GPU_EXPERT_WAVE=0"
                .into(),
        );
    }
    cost_ledger::set_expected_fixed_active_bytes(Some(
        cost_ledger::MATH_PRESERVE_FIXED_ACTIVE_BYTES,
    ));
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
    let total = (context + warmup + ledger_tokens).max(ab_context + 1);
    let tokens = stream(total);

    // Exact-input same-process A/B. A throw-away pass warms the shared weight
    // cache, then the same prefix + decode token runs once with ledger fully
    // disabled and once with it recording. Session reset keeps KV/state input
    // identical; weight residency remains in the same model instance.
    let ab_prefix = &tokens[..ab_context];
    let ab_token = tokens[ab_context];
    cost_ledger::set_enabled(false);
    model.forward(ab_prefix)?;
    model.forward_at(&[ab_token], ab_context)?;

    model.forward(ab_prefix)?;
    let waits_a0 = model.last_resident_waits().unwrap_or(0);
    let cache_a_before = model.cache_stats();
    let physical_a =
        hawking_core::metal::PhysicalTraceGuard::begin(physical_identity('a', "ledger_off")?)?;
    let a_wall = Instant::now();
    let (a_logits, a_trace) = model.forward_at(&[ab_token], ab_context)?;
    let a_ms = a_wall.elapsed().as_secs_f64() * 1e3;
    let cache_a_after = model.cache_stats();
    let physical_a_counts = physical_a.counts();
    drop(physical_a);
    let waits_a = model
        .last_resident_waits()
        .unwrap_or(waits_a0)
        .saturating_sub(waits_a0);
    let a_output = decode_output(&a_logits, &a_trace, model.arch.vocab_size)?;
    let cache_a_delta = cache_delta(cache_a_before, cache_a_after);

    cost_ledger::set_enabled(false);
    model.forward(ab_prefix)?;
    let waits_b0 = model.last_resident_waits().unwrap_or(0);
    let cache_b_before = model.cache_stats();
    cost_ledger::set_enabled(true);
    let physical_b =
        hawking_core::metal::PhysicalTraceGuard::begin(physical_identity('b', "ledger_on")?)?;
    let b_wall = Instant::now();
    let (b_logits, b_trace, b_report) = model.forward_at_with_ledger(&[ab_token], ab_context)?;
    let b_ms = b_wall.elapsed().as_secs_f64() * 1e3;
    let cache_b_after = model.cache_stats();
    let physical_b_counts = physical_b.counts();
    drop(physical_b);
    let waits_b = model
        .last_resident_waits()
        .unwrap_or(waits_b0)
        .saturating_sub(waits_b0);
    let b_output = decode_output(&b_logits, &b_trace, model.arch.vocab_size)?;
    let b_report = b_report.ok_or("A/B profiled arm returned no ledger report")?;
    let cache_b_delta = cache_delta(cache_b_before, cache_b_after);

    let same_token = a_output["token"] == b_output["token"];
    let same_mode = a_output["mode"] == b_output["mode"];
    let same_logits = a_logits == b_logits;
    let same_topk = a_output["topk_indices"] == b_output["topk_indices"]
        && a_output["topk_values"] == b_output["topk_values"];
    let same_router_and_indexer = a_trace.expert_choices == b_trace.expert_choices
        && a_trace.final_topk == b_trace.final_topk;
    let same_waits = waits_a == waits_b;
    let same_physical_graph = physical_a_counts == physical_b_counts;
    let same_cache_delta = cache_a_delta == cache_b_delta;
    let ledger_commands_match_physical =
        b_report.counters.command_buffers_submitted == physical_b_counts.command_count;
    let equivalence_passed = same_token
        && same_mode
        && same_logits
        && same_topk
        && same_router_and_indexer
        && same_waits
        && same_physical_graph
        && same_cache_delta
        && ledger_commands_match_physical;
    if !equivalence_passed {
        return Err(format!(
            "ledger A/B changed the path/output: token={same_token} mode={same_mode} \
             logits={same_logits} topk={same_topk} route/index={same_router_and_indexer} \
             waits={same_waits} physical_graph={same_physical_graph} \
             cache_delta={same_cache_delta} ledger_commands={ledger_commands_match_physical}"
        )
        .into());
    }
    let ab_equivalence = serde_json::json!({
        "method": "same process, warmed shared weight cache, session reset, exact same prefix and decode token",
        "context_tokens": ab_context,
        "decode_token": ab_token,
        "unprofiled": {
            "external_wall_ms": a_ms,
            "output": a_output,
            "resident_waits": waits_a,
            "physical_trace_counts": physical_a_counts,
            "cache_before": cache_snapshot(cache_a_before),
            "cache_after": cache_snapshot(cache_a_after),
            "cache_delta": cache_a_delta,
        },
        "profiled": {
            "external_wall_ms": b_ms,
            "output": b_output,
            "resident_waits": waits_b,
            "internal_bookkeeping_overhead_us": b_report.profiler_overhead_us,
            "command_buffers": b_report.counters.command_buffers_submitted,
            "dispatches": b_report.counters.dispatches_encoded,
            "synchronizations": b_report.counters.synchronization_points,
            "physical_trace_counts": physical_b_counts,
            "cache_before": cache_snapshot(cache_b_before),
            "cache_after": cache_snapshot(cache_b_after),
            "cache_delta": cache_b_delta,
        },
        "equivalent": {
            "all_required_checks_passed": equivalence_passed,
            "same_token": same_token,
            "same_output_mode": same_mode,
            "same_logits": same_logits,
            "same_topk_diagnostics": same_topk,
            "same_router_and_indexer_choices": same_router_and_indexer,
            "same_resident_wait_count": same_waits,
            "same_physical_command_and_encoder_counts": same_physical_graph,
            "same_per_token_cache_delta": same_cache_delta,
            "ledger_command_count_matches_physical_trace": ledger_commands_match_physical,
        },
        "empirical_observer_delta_ms_single_pair": b_ms - a_ms,
        "observer_delta_caveat": "one exact-input pair is a path-equivalence/calibration diagnostic, not a stable latency estimate; use aggregate tokens for tails",
        "cache_state_caveat": "absolute before/after cache snapshots are disclosed but not required to match because the shared LRU advances between arms; exact per-token resident/entry/eviction/high-water deltas are required after an identical throw-away warm pass",
    });
    eprintln!(
        "A/B exact token | off={a_ms:.1} ms on={b_ms:.1} ms waits={waits_a} output={}",
        ab_equivalence["unprofiled"]["output"]["mode"]
    );

    // Prefill: not ledgered (the gate asks for decode-token attribution).
    cost_ledger::set_enabled(false);
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
    cost_ledger::set_enabled(true);
    for (i, &t) in tokens[context + warmup..]
        .iter()
        .take(ledger_tokens)
        .enumerate()
    {
        let abs_pos = context + warmup + i;
        let wall = Instant::now();
        let (logits, trace, report) = model.forward_at_with_ledger(&[t], abs_pos)?;
        let wall_ms = wall.elapsed().as_secs_f64() * 1e3;
        let output = decode_output(&logits, &trace, model.arch.vocab_size)?;

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
            "output": output,
            "ledger": report.to_json_value(),
        }));
        raw_reports.push(report);
    }

    let min_attributed_fraction = raw_reports
        .iter()
        .map(|r| r.attributed_fraction)
        .reduce(f64::min)
        .unwrap_or(0.0);
    let untagged_gpu_dispatches: u64 = raw_reports
        .iter()
        .flat_map(|r| r.device.command_buffers.iter())
        .flat_map(|cb| cb.stage_composition.iter())
        .filter(|stage| stage.stage == "untagged")
        .map(|stage| stage.dispatches)
        .sum();
    let stage_dispatch_count_mismatches = raw_reports
        .iter()
        .flat_map(|r| r.device.command_buffers.iter())
        .filter(|cb| !cb.stage_dispatches_match_buffer)
        .count();
    let generic_orchestration_lines: Vec<_> = raw_reports
        .first()
        .map(|r| {
            r.buckets_us
                .keys()
                .filter(|key| key.contains("orchestration") || key.contains("residual_scoped"))
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    let operations_nonzero = raw_reports
        .iter()
        .all(|r| r.counters.operations > 0 && r.counters.source_modelled_fp_operations > 0);
    let coverage_gate = serde_json::json!({
        "required_attributed_fraction": 0.95,
        "minimum_token_attributed_fraction": min_attributed_fraction,
        "all_tokens_at_least_95_percent": !raw_reports.is_empty() && min_attributed_fraction >= 0.95,
        "untagged_gpu_dispatches": untagged_gpu_dispatches,
        "no_untagged_gpu_dispatches": untagged_gpu_dispatches == 0,
        "stage_dispatch_count_mismatches": stage_dispatch_count_mismatches,
        "all_command_buffer_stage_counts_exact": stage_dispatch_count_mismatches == 0,
        "generic_orchestration_lines": generic_orchestration_lines,
        "no_generic_orchestration_bucket": generic_orchestration_lines.is_empty(),
        "operations_nonzero_every_token": operations_nonzero,
    });

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
        "run_configuration": {
            "raw_environment": env_snapshot(),
            "resolved": {
                "resident_state": model.resident_state_enabled(),
                "gpu_native_bf16_head": hawking_core::gravity_glm::gpu_lm_head_enabled(),
                "full_logits_readback": hawking_core::gravity_glm::gpu_lm_head_full_logits_enabled(),
                "expert_wave": hawking_core::gravity_glm::gpu_expert_wave_enabled(),
                "cost_ledger": true,
                "tcb_trace": "off",
            },
            "command_buffer_equivalence": "promoted resident path encodes the same TCB dispatch graph in both arms; ledger-on adds host timestamps and post-completion GPUStartTime/GPUEndTime reads, but does not split dispatches",
        },
        "same_process_exact_input_ab": ab_equivalence,
        "coverage_gate": coverage_gate,
        "architecture": {
            "layers": model.arch.n_layers,
            "hidden": model.arch.hidden,
            "routed_experts": model.arch.n_routed_experts,
            "experts_per_tok": model.arch.num_experts_per_tok,
            "vocab": model.arch.vocab_size,
        },
        "math_preserve_active_byte_contract": math_preserve_active_byte_contract(),
        "active_byte_accounting_note": "Each token seeds the exact fixed Math-Preserve resident-source extent, then extends it from live R4/R0/native-bf16 routed projection evidence. active_bytes_read and its category partition must equal that route-conditioned expectation. This is not a physical-DRAM claim.",
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
            "unattributed is a first-class line with its own magnitude; there is no generic orchestration bucket.",
            "Device timeline (gpu_execution_us, gpu_queue_wait_us) is independent of exclusive CPU buckets; GPU work overlaps metal_synchronize_cpu_wait.",
            "gpu_execution_by_stage_us uses real whole-command-buffer Metal timestamps. A CB with multiple semantic dispatch tags remains under an exact mixed:* key with stage_composition; no proportional split is invented.",
            "gpu_queue_wait_us = sum max(0, host_wait_us - gpu_execution_us) per CB when GPUStartTime/GPUEndTime are readable; otherwise None (no CPU proxy).",
            "Metal timestamp counter samples are opt-in (markers must be encoded). This path records CB-level GPUStartTime/GPUEndTime only unless a counter probe is wired.",
            "page_faults from getrusage(RUSAGE_SELF) minflt/majflt deltas when available.",
            "profiler_overhead_us is ledger bookkeeping cost disclosed on every token.",
            "same_process_exact_input_ab reports the full observer delta separately from internal bookkeeping and proves output mode/token/top-k/wait-count equivalence.",
            "p50/p95/p99 live under aggregate.* when ledger-tokens > 1.",
            "Routed experts and the shared expert are co-batched; their mixed command buffers disclose exact routed_experts/shared_experts dispatch counts.",
            "Operation counters are source-modelled rather than hardware counters: floating point, packed integer/bitwise lower bound, comparisons, transcendentals, and dense-equivalent FP are separate.",
            "This binary does not claim timings it did not measure; empty buckets mean that work was not observed on the instrumented path.",
            "active_bytes_by_category partitions active_bytes_read by tensor class (routed/shared/dense_mlp/attention/indexer/router/lm_head/other).",
            "Historical General-R0 2.58/9.34 GB geometry is not current Math-Preserve evidence. The current fixed set is 3,054,873,024 B plus the exact live routed representation mix.",
        ],
        "explicit_limitations": [
            "Metal counter-sample markers are not encoded; CB GPUStartTime/GPUEndTime is the non-perturbing device source.",
            "Residency is sampled immediately after each token and attached before aggregation because the model cache snapshot API is outside the token scope.",
            "Packed integer/bitwise operation accounting is a source-level lower bound; address arithmetic and compiler transformations are excluded.",
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

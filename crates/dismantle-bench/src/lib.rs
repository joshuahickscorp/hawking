pub mod competitors;
pub mod suites;

use anyhow::Result;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct BenchOptions {
    pub weights: PathBuf,
    pub model_id: String,
    pub suite: String,
    pub json_out: Option<PathBuf>,
    pub trials: usize,
    pub max_new_tokens: usize,
    pub trace_json: Option<PathBuf>,
    pub kernel_profile: Option<PathBuf>,
    pub speculate_mode: String,
    pub verify_window: usize,
    /// Which backend to run a single-suite bench against. `"dismantle"`
    /// (default) drives the in-process Engine; `"llamacpp"` and
    /// `"mlx"` shell out to the corresponding competitor binaries.
    /// Only the decode and prefill suites honor this -- the
    /// `competitive` suite explicitly runs all three.
    pub backend: String,
    /// When true, enables Metal dispatch tracing and structural counters
    /// (buffers/commits per token). Mirrors `--trace-dispatch` CLI flag.
    pub trace_dispatch: bool,
}

#[derive(Debug, Serialize)]
pub struct BenchReport {
    pub model_id: String,
    pub suite: String,
    pub trials: usize,
    pub max_new_tokens: usize,
    pub kernel_profile: Option<String>,
    pub speculate_mode: String,
    pub verify_window: usize,
    pub results: serde_json::Value,
}

pub fn run(opts: BenchOptions) -> Result<()> {
    use suites::*;

    let res: serde_json::Value = match opts.suite.as_str() {
        "decode-tps" | "decode" => decode::run(&opts)?,
        "prefill-tps" | "prefill" => prefill::run(&opts)?,
        "throughput-vs-batch" | "throughput" => throughput::run(&opts)?,
        "bandwidth-utilization" | "bandwidth" => bandwidth::run(&opts)?,
        "competitive" => competitive::run(&opts)?,
        // Deprecated alias from the pre-2026-04-audit naming. Kept for
        // one release; will be removed in v0.2.
        "wax-vs-llama-cpp" | "wax" => {
            eprintln!(
                "[dismantle-bench] suite name `wax` is deprecated; use `competitive`. \
                       (Renamed after the 2026-04 competitive audit; see ROADMAP.md.)"
            );
            competitive::run(&opts)?
        }
        "all" => serde_json::json!({
            "decode":      decode::run(&opts).unwrap_or_default(),
            "prefill":     prefill::run(&opts).unwrap_or_default(),
            "throughput":  throughput::run(&opts).unwrap_or_default(),
            "bandwidth":   bandwidth::run(&opts).unwrap_or_default(),
            "competitive": competitive::run(&opts).unwrap_or_default(),
        }),
        other => anyhow::bail!("unknown suite `{other}`"),
    };

    let report = BenchReport {
        model_id: opts.model_id.clone(),
        suite: opts.suite.clone(),
        trials: opts.trials,
        max_new_tokens: opts.max_new_tokens,
        kernel_profile: opts
            .kernel_profile
            .as_ref()
            .map(|p| p.display().to_string()),
        speculate_mode: opts.speculate_mode.clone(),
        verify_window: opts.verify_window,
        results: res,
    };
    let s = serde_json::to_string_pretty(&report)?;
    if let Some(trace_path) = &opts.trace_json {
        let report_hash = short_hash(s.as_bytes());
        let decode_tps = report
            .results
            .get("decode_tps")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Extract dispatch samples from the decode suite result (only present
        // when DISMANTLE_TRACE_DISPATCH=1).
        let raw_samples = report.results.get("dispatch_samples");
        let dispatch_total_us = report
            .results
            .get("dispatch_total_us")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let dispatch_total_decode_us = report
            .results
            .get("dispatch_total_decode_us")
            .and_then(|v| v.as_u64())
            .unwrap_or(1);
        let dispatch_pct = if dispatch_total_decode_us > 0 {
            (dispatch_total_us as f64 / dispatch_total_decode_us as f64) * 100.0
        } else {
            0.0
        };

        let kernel_summary = if let Some(serde_json::Value::Array(samples)) = raw_samples {
            build_kernel_summary(samples)
        } else {
            serde_json::Value::Array(vec![])
        };
        let layer_summary = if let Some(serde_json::Value::Array(samples)) = raw_samples {
            build_layer_summary(samples)
        } else {
            serde_json::Value::Array(vec![])
        };
        // Only embed raw samples when present (gated by env var on the engine side).
        let dispatch_samples_value = raw_samples.cloned().unwrap_or(serde_json::Value::Null);

        let trace = serde_json::json!({
            "schema_version": 2,
            "kind": "dismantle-bench-trace",
            "metric": "decode_only_tps",
            "report_hash": report_hash,
            "target_tps": 60.0,
            "target_gap_x": if decode_tps > 0.0 { 60.0 / decode_tps } else { 0.0 },
            "rss_mb_after_run": current_rss_mb(),
            "profile_path": opts.kernel_profile.as_ref().map(|p| p.display().to_string()),
            "speculate_mode": opts.speculate_mode,
            "verify_window": opts.verify_window,
            "dispatch_wall_pct_of_decode": dispatch_pct,
            "dispatch_total_us": dispatch_total_us,
            "kernel_summary": kernel_summary,
            "layer_summary": layer_summary,
            "dispatch_samples": dispatch_samples_value,
            "report": report,
        });
        std::fs::write(trace_path, serde_json::to_string_pretty(&trace)?)?;
    }
    match opts.json_out {
        Some(p) => std::fs::write(&p, s)?,
        None => println!("{s}"),
    }
    Ok(())
}

/// Aggregate per-dispatch samples into per-kernel statistics.
fn build_kernel_summary(samples: &[serde_json::Value]) -> serde_json::Value {
    use std::collections::HashMap;
    // kernel_name → Vec<wall_us>
    let mut by_kernel: HashMap<&str, Vec<u64>> = HashMap::new();
    for s in samples {
        let name = s
            .get("kernel_name")
            .and_then(|v| v.as_str())
            .unwrap_or("other");
        let us = s.get("wall_us").and_then(|v| v.as_u64()).unwrap_or(0);
        by_kernel.entry(name).or_default().push(us);
    }
    let mut rows: Vec<serde_json::Value> = by_kernel
        .iter()
        .map(|(name, times)| {
            let count = times.len() as u64;
            let total: u64 = times.iter().sum();
            let mean = total.checked_div(count).unwrap_or(0);
            let mut sorted = times.clone();
            sorted.sort_unstable();
            let p50 = sorted[sorted.len() / 2];
            let p99 = sorted[(sorted.len() * 99 / 100).min(sorted.len() - 1)];
            serde_json::json!({
                "kernel": name,
                "count": count,
                "total_us": total,
                "mean_us": mean,
                "p50_us": p50,
                "p99_us": p99,
            })
        })
        .collect();
    // Sort descending by total_us.
    rows.sort_by(|a, b| {
        let ta = a.get("total_us").and_then(|v| v.as_u64()).unwrap_or(0);
        let tb = b.get("total_us").and_then(|v| v.as_u64()).unwrap_or(0);
        tb.cmp(&ta)
    });
    serde_json::Value::Array(rows)
}

/// Aggregate per-dispatch samples into per-layer statistics.
fn build_layer_summary(samples: &[serde_json::Value]) -> serde_json::Value {
    use std::collections::HashMap;
    // layer → (total_us, HashMap<kernel, total_us>)
    let mut by_layer: HashMap<u32, (u64, HashMap<String, u64>)> = HashMap::new();
    for s in samples {
        let layer = match s.get("layer_hint").and_then(|v| v.as_u64()) {
            Some(l) => l as u32,
            None => continue, // skip non-layer dispatches (final norm, LM head)
        };
        let name = s
            .get("kernel_name")
            .and_then(|v| v.as_str())
            .unwrap_or("other");
        let us = s.get("wall_us").and_then(|v| v.as_u64()).unwrap_or(0);
        let entry = by_layer.entry(layer).or_default();
        entry.0 += us;
        *entry.1.entry(name.to_string()).or_default() += us;
    }
    let mut rows: Vec<serde_json::Value> = by_layer
        .iter()
        .map(|(layer, (total, kernels))| {
            serde_json::json!({
                "layer": layer,
                "total_us": total,
                "kernels": kernels,
            })
        })
        .collect();
    rows.sort_by_key(|r| r.get("layer").and_then(|v| v.as_u64()).unwrap_or(u64::MAX));
    serde_json::Value::Array(rows)
}

fn short_hash(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize()
        .iter()
        .take(12)
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn current_rss_mb() -> Option<f64> {
    let pid = std::process::id().to_string();
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok()?;
    let kb = String::from_utf8(out.stdout)
        .ok()?
        .trim()
        .parse::<f64>()
        .ok()?;
    Some(kb / 1024.0)
}

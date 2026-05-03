//! dismantle-bench: produces every published benchmark number.
//!
//! Each suite is a function in [`suites`]. The binary entrypoint
//! lives in `dismantle bench`; this crate is invoked from that
//! umbrella binary.
//!
//! Output is a single JSON document per run, written to stdout or a
//! `--json <path>` target. `docs/benchmarks.md` is auto-generated
//! from these JSON outputs by a script in `docs/`.

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
    /// Only the decode and prefill suites honor this — the
    /// `competitive` suite explicitly runs all three.
    pub backend: String,
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
        let trace = serde_json::json!({
            "schema_version": 1,
            "kind": "dismantle-bench-trace",
            "metric": "decode_only_tps",
            "report_hash": report_hash,
            "target_tps": 60.0,
            "target_gap_x": if decode_tps > 0.0 { 60.0 / decode_tps } else { 0.0 },
            "rss_mb_after_run": current_rss_mb(),
            "profile_path": opts.kernel_profile.as_ref().map(|p| p.display().to_string()),
            "speculate_mode": opts.speculate_mode,
            "verify_window": opts.verify_window,
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

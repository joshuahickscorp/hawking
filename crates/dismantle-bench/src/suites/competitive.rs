//! competitive: head-to-head against llama.cpp.
//!
//! Replaces the old `wax-vs-llama-cpp` suite. The audit doc
//! (`docs/competitive_audit.md`) frames the broader competitive
//! landscape including MLX; this suite measures what we can
//! reproduce locally without crossing into a foreign tensor format.
//!
//! Spawns each competitor as a sibling process on the same prompt
//! suite + same hardware + same model. Emits a single JSON document
//! with per-(backend, prompt) measurements; the audit doc is
//! generated from this output.
//!
//! Output shape matches `tools/competitors/results.json` so the
//! in-binary harness and the offline shell harness are interchangeable.

use crate::competitors::{Competitor, DismantleBackend, LlamaCppBackend};
use crate::BenchOptions;
use anyhow::Result;
use serde_json::json;

pub fn run(opts: &BenchOptions) -> Result<serde_json::Value> {
    let prompts = load_prompts();
    if prompts.is_empty() {
        return Ok(json!({
            "error": "no prompts found in tools/competitors/prompts.txt; \
                      run from the workspace root or copy prompts.txt next \
                      to the binary",
        }));
    }

    let mut backends: Vec<Box<dyn Competitor>> = vec![
        Box::new(LlamaCppBackend::new()),
        Box::new(DismantleBackend::new(&opts.weights)),
    ];

    let backend_meta: Vec<serde_json::Value> = backends
        .iter()
        .map(|b| {
            json!({
                "name": b.name(),
                "version": b.version(),
                "phase": b.phase_tag(),
            })
        })
        .collect();

    let mut rows = Vec::new();
    for (prompt_idx, (tier, prompt)) in prompts.iter().enumerate() {
        for backend in backends.iter_mut() {
            let mut trials = Vec::new();
            for _ in 0..opts.trials {
                match backend.run(&opts.weights, prompt, opts.max_new_tokens, 0.0) {
                    Ok(m) => trials.push(json!({
                        "decode_tps":  m.decode_tps,
                        "prefill_tps": m.prefill_tps,
                        "ttft_ms":     m.ttft_ms,
                        "peak_rss_mb": m.peak_rss_mb,
                        "output_chars": m.output.len(),
                    })),
                    Err(e) => trials.push(json!({"error": e.to_string()})),
                }
            }
            rows.push(json!({
                "prompt_idx": prompt_idx + 1,
                "tier": tier,
                "backend": backend.name(),
                "trials": trials,
            }));
        }
    }

    Ok(json!({
        "model": opts.model_id,
        "hw": detect_hw(),
        "backends": backend_meta,
        "rows": rows,
        "audit_doc": "docs/competitive_audit.md",
    }))
}

/// Read `tools/competitors/prompts.txt` from the workspace root.
/// Returns `Vec<(tier, prompt)>`. Falls back to a tiny built-in list
/// if the file is missing (so the suite still produces *something*
/// for smoke-testing without the workspace layout in place).
fn load_prompts() -> Vec<(String, String)> {
    let candidates = [
        std::path::PathBuf::from("tools/competitors/prompts.txt"),
        std::path::PathBuf::from("../tools/competitors/prompts.txt"),
        std::path::PathBuf::from("../../tools/competitors/prompts.txt"),
    ];
    for p in &candidates {
        if let Ok(s) = std::fs::read_to_string(p) {
            return parse_prompts(&s);
        }
    }
    vec![
        ("SHORT".into(), "Once upon a time".into()),
        (
            "MED".into(),
            "Explain how Mixture of Experts routing works in three sentences.".into(),
        ),
    ]
}

fn parse_prompts(s: &str) -> Vec<(String, String)> {
    s.lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .filter_map(|l| {
            let mut it = l.splitn(2, '|');
            let tier = it.next()?.trim().to_string();
            let prompt = it.next()?.trim().to_string();
            Some((tier, prompt))
        })
        .collect()
}

fn detect_hw() -> String {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".into())
    }
    #[cfg(not(target_os = "macos"))]
    {
        "non-macos".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_prompts_strips_comments_and_blanks() {
        let s = "# header\n\nSHORT|hello\nMED|world\n\n# trailer\n";
        let p = parse_prompts(s);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0], ("SHORT".to_string(), "hello".to_string()));
        assert_eq!(p[1], ("MED".to_string(), "world".to_string()));
    }
}

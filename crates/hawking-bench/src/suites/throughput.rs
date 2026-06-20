use crate::BenchOptions;
use anyhow::Result;

pub fn run(_opts: &BenchOptions) -> Result<serde_json::Value> {
    Ok(serde_json::json!({
        "phase4_pending": true,
        "note": "batch>1 lands with the continuous-batching slot manager (Phase 4).",
    }))
}

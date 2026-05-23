use crate::BenchOptions;
use anyhow::Result;

pub fn run(_opts: &BenchOptions) -> Result<serde_json::Value> {
    Ok(serde_json::json!({
        "phase1_pending": true,
        "note": "real numbers land with the wedge-2 fused-dequant kernel (Phase 1).",
    }))
}

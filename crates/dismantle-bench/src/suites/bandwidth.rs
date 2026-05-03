//! bandwidth-utilization: measured GB/s of weight reads during
//! decode, divided by the platform's theoretical peak (M3 Pro: 150
//! GB/s, M3 Max: 400 GB/s, M3 Ultra: 800 GB/s). The honesty number —
//! a 90% figure means there is no further perf to extract without
//! changing the math, only the algorithm.
//!
//! Method: time the decode loop on a known-shape model, compute
//! bytes-per-token from the model config (one full pass through every
//! weight per token, minus what KV cache covers). Divide by elapsed
//! time, and divide that by the platform peak.

use crate::BenchOptions;
use anyhow::Result;

pub fn run(_opts: &BenchOptions) -> Result<serde_json::Value> {
    Ok(serde_json::json!({
        "phase1_pending": true,
        "note": "real numbers land with the wedge-2 fused-dequant kernel (Phase 1).",
    }))
}

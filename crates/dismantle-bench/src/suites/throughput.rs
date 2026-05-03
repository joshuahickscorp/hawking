//! throughput-vs-batch: aggregate tokens/sec at batch sizes
//! 1, 2, 4, 8, 16. Validates the Phase 4 batching gate and the
//! continuous-batching speed-up.
//!
//! Until continuous batching lands (Phase 4) this suite reports
//! batch=1 only and emits a "phase4_pending" sentinel. Wiring it up
//! at this layer means the JSON shape doesn't change when the
//! batching path goes live.

use crate::BenchOptions;
use anyhow::Result;

pub fn run(_opts: &BenchOptions) -> Result<serde_json::Value> {
    Ok(serde_json::json!({
        "phase4_pending": true,
        "note": "batch>1 lands with the continuous-batching slot manager (Phase 4).",
    }))
}

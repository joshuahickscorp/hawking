use crate::BenchOptions;
use anyhow::Result;

/// Measure sustained device-side buffer-copy throughput.  This is a hardware
/// transfer roof only: model Q4/Q6 kernels have different access patterns and
/// must not be reported as achieving this number.
#[cfg(target_os = "macos")]
fn run_macos() -> Result<serde_json::Value> {
    use hawking_core::metal::{MetalContext, TokenCommandBuffer};
    use std::time::Instant;

    const COPY_BYTES: usize = 256 * 1024 * 1024;
    const WARMUPS: usize = 2;
    const SAMPLES: usize = 5;

    let ctx = MetalContext::new()?;
    let src = ctx.new_buffer(COPY_BYTES);
    let dst = ctx.new_buffer(COPY_BYTES);
    for _ in 0..WARMUPS {
        let mut tcb = TokenCommandBuffer::new(&ctx);
        tcb.copy_buffer_bytes(&src, 0, &dst, 0, COPY_BYTES as u64)?;
        tcb.commit_and_wait()?;
    }

    let mut gib_per_s = Vec::with_capacity(SAMPLES);
    let mut sample_ms = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let t0 = Instant::now();
        let mut tcb = TokenCommandBuffer::new(&ctx);
        tcb.copy_buffer_bytes(&src, 0, &dst, 0, COPY_BYTES as u64)?;
        tcb.commit_and_wait()?;
        let elapsed = t0.elapsed().as_secs_f64();
        sample_ms.push(elapsed * 1_000.0);
        gib_per_s.push((COPY_BYTES as f64 / (1024.0 * 1024.0 * 1024.0)) / elapsed);
    }
    gib_per_s.sort_by(|a, b| a.total_cmp(b));
    sample_ms.sort_by(|a, b| a.total_cmp(b));
    let mid = SAMPLES / 2;
    Ok(serde_json::json!({
        "schema": "hawking.bandwidth.blit.v1",
        "device": ctx.device_name(),
        "measurement": "one device-side Metal blit per command buffer, commit_and_wait included",
        "metric_source": "host wall time around encoded Metal blit and completion",
        "copy_bytes": COPY_BYTES,
        "warmups": WARMUPS,
        "samples": SAMPLES,
        "gib_per_s_samples": gib_per_s,
        "gib_per_s_median": gib_per_s[mid],
        "copy_ms_samples": sample_ms,
        "copy_ms_median": sample_ms[mid],
        "scope_limit": "This is a copy-bandwidth roof, not effective Q4/Q6 model bandwidth or a decode TPS result."
    }))
}

#[cfg(not(target_os = "macos"))]
fn run_macos() -> Result<serde_json::Value> {
    Ok(serde_json::json!({
        "schema": "hawking.bandwidth.blit.v1",
        "available": false,
        "reason": "Metal buffer-copy benchmark requires macOS"
    }))
}

pub fn run(_opts: &BenchOptions) -> Result<serde_json::Value> {
    run_macos()
}

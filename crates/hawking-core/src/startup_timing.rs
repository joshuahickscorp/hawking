//! Process-local startup phase timers for experiment-loop iteration cost.
//!
//! Enabled only when `HAWKING_STARTUP_TIMING=1`. Timers are process-wide and
//! intentionally lightweight: they never affect admission correctness.

use std::sync::Mutex;
use std::time::{Duration, Instant};

static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
// Durations are stored in integer nanoseconds. Keep the value type small and
// derive human-readable milliseconds only at the JSON/presentation edge.
static PHASES: Mutex<Vec<(String, u64)>> = Mutex::new(Vec::new());
static PROCESS_START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

fn enabled() -> bool {
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("HAWKING_STARTUP_TIMING").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    })
}

/// Mark process start (idempotent). Call as early as possible in main.
pub fn mark_process_start() {
    let _ = PROCESS_START.get_or_init(Instant::now);
    let _ = enabled();
}

/// Record one named phase duration in integer nanoseconds.
pub fn record_ns(phase: impl Into<String>, ns: u64) {
    if !enabled() {
        return;
    }
    if let Ok(mut phases) = PHASES.lock() {
        phases.push((phase.into(), ns));
    }
}

/// Compatibility writer for callers that already have milliseconds.
///
/// New callers should use [`record_ns`]. The public name remains available so
/// old model-specific probes keep compiling while the stored and emitted
/// representation remains ns-first.
pub fn record_ms(phase: impl Into<String>, ms: u64) {
    record_ns(phase, ms.saturating_mul(1_000_000));
}

/// Time a closure and record its wall duration under `phase` in nanoseconds.
pub fn time_ns<T>(phase: impl Into<String>, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let phase = phase.into();
    let start = Instant::now();
    let out = f();
    let ns = duration_ns(start.elapsed());
    record_ns(phase, ns);
    out
}

/// Compatibility name for the old millisecond timer API. The measurement is
/// now captured at ns precision before any presentation conversion.
pub fn time_ms<T>(phase: impl Into<String>, f: impl FnOnce() -> T) -> T {
    time_ns(phase, f)
}

/// Time a fallible closure in nanoseconds.
pub fn time_ns_result<T, E>(
    phase: impl Into<String>,
    f: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    if !enabled() {
        return f();
    }
    let phase = phase.into();
    let start = Instant::now();
    let out = f();
    let ns = duration_ns(start.elapsed());
    record_ns(phase, ns);
    out
}

/// Compatibility name for the old millisecond timer API.
pub fn time_ms_result<T, E>(
    phase: impl Into<String>,
    f: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    time_ns_result(phase, f)
}

/// Convert a duration to a saturating integer nanosecond count.
pub fn duration_ns(d: Duration) -> u64 {
    d.as_nanos().min(u64::MAX as u128) as u64
}

/// Compatibility presentation helper for callers that need milliseconds.
pub fn duration_ms(d: Duration) -> u64 {
    duration_ns(d).saturating_add(500_000) / 1_000_000
}

/// Snapshot recorded phases and process wall so far.
pub fn snapshot() -> StartupTimingSnapshot {
    let process_wall_ns = PROCESS_START
        .get()
        .map(|t| duration_ns(t.elapsed()))
        .unwrap_or(0);
    let phases = PHASES.lock().map(|g| g.clone()).unwrap_or_default();
    StartupTimingSnapshot {
        enabled: enabled(),
        process_wall_ns,
        phases,
    }
}

/// Emit a single JSON line to stderr (never stdout — stdout is the receipt).
pub fn emit_stderr_json() {
    if !enabled() {
        return;
    }
    let snap = snapshot();
    let phases: Vec<serde_json::Value> = snap
        .phases
        .iter()
        .map(|(name, ns)| {
            serde_json::json!({
                "phase": name,
                "elapsed_ns": ns,
                "ms": (*ns as f64) / 1_000_000.0,
            })
        })
        .collect();
    let doc = serde_json::json!({
        "schema": "hawking.startup_timing.v2",
        "timing_unit": "ns",
        "process_wall_ns": snap.process_wall_ns,
        "process_wall_ms": (snap.process_wall_ns as f64) / 1_000_000.0,
        "phases": phases,
    });
    eprintln!(
        "HAWKING_STARTUP_TIMING {}",
        serde_json::to_string(&doc).unwrap_or_else(|_| "{}".into())
    );
}

#[derive(Clone, Debug, Default)]
pub struct StartupTimingSnapshot {
    pub enabled: bool,
    pub process_wall_ns: u64,
    pub phases: Vec<(String, u64)>,
}

impl StartupTimingSnapshot {
    pub fn to_json(&self) -> serde_json::Value {
        let phases: Vec<serde_json::Value> = self
            .phases
            .iter()
            .map(|(name, ns)| {
                serde_json::json!({
                    "phase": name,
                    "elapsed_ns": ns,
                    "ms": (*ns as f64) / 1_000_000.0,
                })
            })
            .collect();
        serde_json::json!({
            "schema": "hawking.startup_timing.v2",
            "enabled": self.enabled,
            "timing_unit": "ns",
            "process_wall_ns": self.process_wall_ns,
            "process_wall_ms": (self.process_wall_ns as f64) / 1_000_000.0,
            "phases": phases,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{duration_ms, duration_ns};
    use std::time::Duration;

    #[test]
    fn duration_ns_preserves_sub_millisecond_precision() {
        assert_eq!(duration_ns(Duration::new(0, 1_234_567)), 1_234_567);
        assert_eq!(duration_ms(Duration::new(0, 1_234_567)), 1);
    }

    #[test]
    fn duration_ns_saturates_without_float_rounding() {
        assert_eq!(duration_ns(Duration::MAX), u64::MAX);
    }
}

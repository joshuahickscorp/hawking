//! Process-local startup phase timers for experiment-loop iteration cost.
//!
//! Enabled only when `HAWKING_STARTUP_TIMING=1`. Timers are process-wide and
//! intentionally lightweight: they never affect admission correctness.

use std::sync::Mutex;
use std::time::{Duration, Instant};

static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
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

/// Record one named phase duration in milliseconds.
pub fn record_ms(phase: impl Into<String>, ms: u64) {
    if !enabled() {
        return;
    }
    if let Ok(mut phases) = PHASES.lock() {
        phases.push((phase.into(), ms));
    }
}

/// Time a closure and record its wall duration under `phase`.
pub fn time_ms<T>(phase: impl Into<String>, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let phase = phase.into();
    let start = Instant::now();
    let out = f();
    let ms = duration_ms(start.elapsed());
    record_ms(phase, ms);
    out
}

/// Time a fallible closure.
pub fn time_ms_result<T, E>(
    phase: impl Into<String>,
    f: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    if !enabled() {
        return f();
    }
    let phase = phase.into();
    let start = Instant::now();
    let out = f();
    let ms = duration_ms(start.elapsed());
    record_ms(phase, ms);
    out
}

pub fn duration_ms(d: Duration) -> u64 {
    let ms = d.as_secs_f64() * 1000.0;
    if ms <= 0.0 {
        0
    } else if ms >= u64::MAX as f64 {
        u64::MAX
    } else {
        ms.round() as u64
    }
}

/// Snapshot recorded phases and process wall so far.
pub fn snapshot() -> StartupTimingSnapshot {
    let process_wall_ms = PROCESS_START
        .get()
        .map(|t| duration_ms(t.elapsed()))
        .unwrap_or(0);
    let phases = PHASES
        .lock()
        .map(|g| g.clone())
        .unwrap_or_default();
    StartupTimingSnapshot {
        enabled: enabled(),
        process_wall_ms,
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
        .map(|(name, ms)| {
            serde_json::json!({
                "phase": name,
                "ms": ms,
            })
        })
        .collect();
    let doc = serde_json::json!({
        "schema": "hawking.startup_timing.v1",
        "process_wall_ms": snap.process_wall_ms,
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
    pub process_wall_ms: u64,
    pub phases: Vec<(String, u64)>,
}

impl StartupTimingSnapshot {
    pub fn to_json(&self) -> serde_json::Value {
        let phases: Vec<serde_json::Value> = self
            .phases
            .iter()
            .map(|(name, ms)| {
                serde_json::json!({
                    "phase": name,
                    "ms": ms,
                })
            })
            .collect();
        serde_json::json!({
            "schema": "hawking.startup_timing.v1",
            "enabled": self.enabled,
            "process_wall_ms": self.process_wall_ms,
            "phases": phases,
        })
    }
}

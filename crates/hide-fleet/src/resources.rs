//! The resource probe (bible ch.09 §4.6.1 `resources.rs`).
//!
//! The Governor admits on *physical* signals (P1). This module reads them:
//! - **Free RAM** via a light OS read (`vm_stat` on macOS, `/proc/meminfo` on
//!   Linux) — no privilege, no heavy `sysinfo` dep.
//! - **The thermal proxy**: macOS hands no clean "you are throttling" bit, so the
//!   Governor derives it from the runtime's own throughput — a sustained
//!   `dec_tps` drop vs a per-model baseline is read as throttle (§4.6.1). The
//!   runtime's throughput number is the thermometer.
//! - **`max_batch_size`/active slots**: supplied by the runtime (the host reads
//!   `/metrics`); the probe takes them as inputs and returns a full snapshot.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// A coarse thermal classification derived from the throughput proxy + OS hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThermalState {
    Nominal,
    Fair,
    Serious,
    Critical,
}

/// Live machine state sampled ~1 Hz (A.2 `GovernorState`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    pub free_memory_mb: u64,
    /// The runtime's `max_batch_size` (the hard model-concurrency ceiling).
    pub max_generation_slots: u32,
    pub active_generation_slots: u32,
    pub thermal: ThermalState,
    /// Current decode throughput (tok/s) from `/metrics`.
    pub dec_tps_now: f32,
    /// Per-model baseline throughput (cool-machine reference).
    pub dec_tps_baseline: f32,
    pub battery_percent: Option<u8>,
    pub on_ac_power: bool,
    pub idle: bool,
}

impl ResourceSnapshot {
    /// A conservative idle default (used before the first probe).
    pub fn idle() -> Self {
        Self {
            free_memory_mb: 0,
            max_generation_slots: 1,
            active_generation_slots: 0,
            thermal: ThermalState::Nominal,
            dec_tps_now: 0.0,
            dec_tps_baseline: 0.0,
            battery_percent: None,
            on_ac_power: true,
            idle: true,
        }
    }

    /// The thermal proxy: the fractional drop of current dec_tps vs baseline. A
    /// drop ≥ the envelope's throttle threshold signals throttling/contention.
    pub fn thermal_drop_pct(&self) -> f32 {
        if self.dec_tps_baseline <= 0.0 || self.dec_tps_now <= 0.0 {
            return 0.0;
        }
        let drop = (self.dec_tps_baseline - self.dec_tps_now) / self.dec_tps_baseline;
        drop.clamp(0.0, 1.0)
    }
}

/// Reads physical machine state. The default [`OsResourceProbe`] reads the real
/// OS; tests use [`FixedResourceProbe`].
#[async_trait]
pub trait ResourceProbe: Send + Sync {
    /// Sample free RAM (+ derive thermal from the supplied throughput) and fold
    /// in the runtime-supplied slot counts. `max_slots`/`active` come from the
    /// runtime's `/metrics`; `dec_tps_now`/`baseline` likewise — the probe owns
    /// RAM + power, the host owns runtime telemetry.
    async fn snapshot(&self, max_slots: u32, active: u32) -> ResourceSnapshot;
}

/// The real probe: reads free RAM from the OS and power state where available.
/// Throughput/slots are passed through (the host reads `/metrics`).
#[derive(Debug, Clone, Default)]
pub struct OsResourceProbe {
    /// Per-model baseline dec_tps for the thermal proxy (set by the host once a
    /// cool-machine baseline is known).
    pub dec_tps_baseline: f32,
    /// Latest observed dec_tps (the host updates this from `/metrics`).
    pub dec_tps_now: f32,
}

#[async_trait]
impl ResourceProbe for OsResourceProbe {
    async fn snapshot(&self, max_slots: u32, active: u32) -> ResourceSnapshot {
        let free_memory_mb = read_free_memory_mb().unwrap_or(0);
        let mut snap = ResourceSnapshot {
            free_memory_mb,
            max_generation_slots: max_slots,
            active_generation_slots: active,
            thermal: ThermalState::Nominal,
            dec_tps_now: self.dec_tps_now,
            dec_tps_baseline: self.dec_tps_baseline,
            battery_percent: None,
            on_ac_power: true,
            idle: active == 0,
        };
        // Classify thermal from the throughput-derived proxy.
        let drop = snap.thermal_drop_pct();
        snap.thermal = if drop >= 0.40 {
            ThermalState::Critical
        } else if drop >= 0.25 {
            ThermalState::Serious
        } else if drop >= 0.15 {
            ThermalState::Fair
        } else {
            ThermalState::Nominal
        };
        snap
    }
}

/// A fixed probe for tests / deterministic scheduling.
#[derive(Debug, Clone)]
pub struct FixedResourceProbe {
    pub snapshot: ResourceSnapshot,
}

#[async_trait]
impl ResourceProbe for FixedResourceProbe {
    async fn snapshot(&self, max_slots: u32, active: u32) -> ResourceSnapshot {
        let mut s = self.snapshot.clone();
        s.max_generation_slots = max_slots;
        s.active_generation_slots = active;
        s
    }
}

/// Read free physical memory in MB without a heavy dependency. Returns `None` on
/// an unsupported OS or a parse failure (the Governor then treats RAM as
/// unknown-but-present; the host can supply a fixed probe instead).
pub fn read_free_memory_mb() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        read_free_memory_mb_macos()
    }
    #[cfg(target_os = "linux")]
    {
        read_free_memory_mb_linux()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

#[cfg(target_os = "macos")]
fn read_free_memory_mb_macos() -> Option<u64> {
    // `vm_stat` reports page counts; multiply free+inactive+speculative by page
    // size. Inactive pages are reclaimable, so they count as effectively free for
    // admission headroom (matching Activity Monitor's "available" notion).
    use std::process::Command;
    let out = Command::new("vm_stat").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut page_size: u64 = 4096;
    let mut free_pages: u64 = 0;
    let mut inactive_pages: u64 = 0;
    let mut speculative_pages: u64 = 0;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Mach Virtual Memory Statistics:") {
            // The header sometimes carries "(page size of N bytes)".
            if let Some(idx) = rest.find("page size of ") {
                let tail = &rest[idx + "page size of ".len()..];
                if let Some(num) = tail.split_whitespace().next() {
                    if let Ok(n) = num.parse::<u64>() {
                        page_size = n;
                    }
                }
            }
        } else if let Some(v) = parse_vm_stat_line(line, "Pages free:") {
            free_pages = v;
        } else if let Some(v) = parse_vm_stat_line(line, "Pages inactive:") {
            inactive_pages = v;
        } else if let Some(v) = parse_vm_stat_line(line, "Pages speculative:") {
            speculative_pages = v;
        }
    }
    let total_free_pages = free_pages + inactive_pages + speculative_pages;
    Some(total_free_pages.saturating_mul(page_size) / (1024 * 1024))
}

#[cfg(target_os = "macos")]
fn parse_vm_stat_line(line: &str, prefix: &str) -> Option<u64> {
    let rest = line.trim().strip_prefix(prefix)?;
    let digits: String = rest.trim().chars().filter(|c| c.is_ascii_digit()).collect();
    digits.parse::<u64>().ok()
}

#[cfg(target_os = "linux")]
fn read_free_memory_mb_linux() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    // Prefer MemAvailable (kernel's reclaimable estimate); fall back to MemFree.
    let mut available_kb: Option<u64> = None;
    let mut free_kb: Option<u64> = None;
    for line in text.lines() {
        if let Some(v) = line.strip_prefix("MemAvailable:") {
            available_kb = v.split_whitespace().next().and_then(|n| n.parse().ok());
        } else if let Some(v) = line.strip_prefix("MemFree:") {
            free_kb = v.split_whitespace().next().and_then(|n| n.parse().ok());
        }
    }
    available_kb.or(free_kb).map(|kb| kb / 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thermal_drop_is_fractional_and_clamped() {
        let mut s = ResourceSnapshot::idle();
        s.dec_tps_baseline = 100.0;
        s.dec_tps_now = 75.0;
        assert!((s.thermal_drop_pct() - 0.25).abs() < 1e-6);
        // No baseline → 0 (don't false-throttle on cold start).
        s.dec_tps_baseline = 0.0;
        assert_eq!(s.thermal_drop_pct(), 0.0);
    }

    #[tokio::test]
    async fn os_probe_classifies_thermal_from_proxy() {
        let probe = OsResourceProbe {
            dec_tps_baseline: 40.0,
            dec_tps_now: 22.0, // 45% drop → Critical.
        };
        let snap = probe.snapshot(4, 1).await;
        assert_eq!(snap.thermal, ThermalState::Critical);
        assert_eq!(snap.max_generation_slots, 4);
    }

    #[tokio::test]
    async fn os_probe_reads_some_memory_on_this_platform() {
        // On macOS/Linux this returns a real number; on others it's 0 (None path).
        let probe = OsResourceProbe::default();
        let snap = probe.snapshot(1, 0).await;
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        assert!(snap.free_memory_mb > 0, "expected a real free-RAM read");
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        let _ = snap;
    }
}

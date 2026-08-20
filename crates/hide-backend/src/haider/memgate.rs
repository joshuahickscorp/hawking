//! MemGate adapter + per-episode feedback.
//!
//! The authoritative memory/resource gate already exists in Hawking; this module
//! does NOT duplicate it. It defines the narrow [`MemGate`] interface the parallel
//! parent needs, provides [`SystemMemGate`] (which reads *real* machine pressure —
//! wired, compressed, swap, available — plus app-level allocations) as the default
//! binding, and records [`EpisodeFeedback`] so the real optimal concurrency can be
//! learned. Once the existing gate's exact interface is confirmed, it is bound
//! behind this trait (delegating to it) rather than re-implemented here.

use std::sync::atomic::{AtomicU64, Ordering};

/// A snapshot of real machine + app-level memory pressure.
///
/// Admission must consider all of these, not just nominal RSS.
#[derive(Clone, Debug, Default)]
pub struct MemoryPressure {
    /// Total physical RAM on the machine.
    pub total_physical_bytes: u64,
    /// Wired (non-purgeable, non-swappable) memory.
    pub wired_bytes: u64,
    /// Compressed memory.
    pub compressed_bytes: u64,
    /// Swap in use.
    pub swap_bytes: u64,
    /// Free / available headroom.
    pub available_bytes: u64,
    /// Resident local-model allocation (shared model body + KV).
    pub resident_model_bytes: u64,
    /// Allocations held by active worker lanes.
    pub active_worker_bytes: u64,
    /// Context / KV allocation across lanes.
    pub context_kv_bytes: u64,
}

/// The result of an admission query.
#[derive(Clone, Debug)]
pub struct AdmissionDecision {
    /// How many lanes the gate admits right now (0..=ceiling).
    pub admitted_lanes: usize,
    /// Human-readable reason (for the UI + feedback).
    pub reason: String,
}

/// The narrow interface the parallel parent needs from the authoritative gate.
pub trait MemGate: Send + Sync {
    /// The bootstrap concurrency ceiling (3).
    fn ceiling(&self) -> usize;
    /// Current pressure snapshot.
    fn pressure(&self) -> MemoryPressure;
    /// Admit up to `requested_lanes` given current pressure.
    fn admit(&self, requested_lanes: usize) -> AdmissionDecision;
}

/// Default binding: reads real OS pressure + app-level allocations, and admits
/// based on measured headroom. `ceiling` is the bootstrap maximum (3); the gate
/// may admit fewer (or zero) under pressure.
pub struct SystemMemGate {
    ceiling: usize,
    /// Estimated per-lane memory need (context + KV growth). Set by the caller.
    per_lane_bytes: AtomicU64,
    resident_model_bytes: AtomicU64,
    active_worker_bytes: AtomicU64,
    context_kv_bytes: AtomicU64,
}

impl SystemMemGate {
    pub fn new(ceiling: usize) -> Self {
        Self {
            ceiling: ceiling.max(1),
            per_lane_bytes: AtomicU64::new(1 << 30), // 1 GiB default estimate
            resident_model_bytes: AtomicU64::new(0),
            active_worker_bytes: AtomicU64::new(0),
            context_kv_bytes: AtomicU64::new(0),
        }
    }

    pub fn set_per_lane_bytes(&self, b: u64) {
        self.per_lane_bytes.store(b, Ordering::Relaxed);
    }
    pub fn set_resident_model_bytes(&self, b: u64) {
        self.resident_model_bytes.store(b, Ordering::Relaxed);
    }
    pub fn set_active_worker_bytes(&self, b: u64) {
        self.active_worker_bytes.store(b, Ordering::Relaxed);
    }
    pub fn set_context_kv_bytes(&self, b: u64) {
        self.context_kv_bytes.store(b, Ordering::Relaxed);
    }
}

impl MemGate for SystemMemGate {
    fn ceiling(&self) -> usize {
        self.ceiling
    }

    fn pressure(&self) -> MemoryPressure {
        let os = read_os_pressure();
        MemoryPressure {
            total_physical_bytes: os.total_physical,
            wired_bytes: os.wired,
            compressed_bytes: os.compressed,
            swap_bytes: os.swap,
            available_bytes: os.available,
            resident_model_bytes: self.resident_model_bytes.load(Ordering::Relaxed),
            active_worker_bytes: self.active_worker_bytes.load(Ordering::Relaxed),
            context_kv_bytes: self.context_kv_bytes.load(Ordering::Relaxed),
        }
    }

    fn admit(&self, requested: usize) -> AdmissionDecision {
        let p = self.pressure();
        let want = requested.min(self.ceiling);
        if want == 0 {
            return AdmissionDecision {
                admitted_lanes: 0,
                reason: "none requested".into(),
            };
        }
        // Headroom after what workers + context already hold.
        let held = p.active_worker_bytes.saturating_add(p.context_kv_bytes);
        let headroom = p.available_bytes.saturating_sub(held);
        let per_lane = self.per_lane_bytes.load(Ordering::Relaxed).max(1);
        let affordable = (headroom / per_lane) as usize;
        let mut admitted = want.min(affordable);

        // Hard pressure signals: swap growing or very little free -> shrink.
        let total = p.total_physical_bytes.max(1);
        let used_frac =
            (p.wired_bytes.saturating_add(p.compressed_bytes)) as f64 / total as f64;
        if p.swap_bytes > 0 && used_frac > 0.85 {
            admitted = admitted.min(1);
        }
        if used_frac > 0.95 {
            admitted = admitted.min(1);
        }
        if headroom < per_lane {
            admitted = 0;
        }

        let reason = format!(
            "headroom={}B per_lane={}B used_frac={:.2} swap={}B",
            headroom, per_lane, used_frac, p.swap_bytes
        );
        AdmissionDecision {
            admitted_lanes: admitted,
            reason,
        }
    }
}

/// OS-level pressure (wired, compressed, swap, available, total).
struct OsPressure {
    total_physical: u64,
    wired: u64,
    compressed: u64,
    swap: u64,
    available: u64,
}

impl Default for OsPressure {
    fn default() -> Self {
        Self {
            total_physical: 0,
            wired: 0,
            compressed: 0,
            swap: 0,
            available: 0,
        }
    }
}

/// Read real machine pressure. Best-effort: returns zeros on failure so admission
/// degrades to "affordable by headroom" rather than panicking.
fn read_os_pressure() -> OsPressure {
    #[cfg(target_os = "macos")]
    {
        unsafe {
            let page = libc::sysconf(libc::_SC_PAGESIZE).max(1) as u64;
            let total = libc::sysconf(libc::_SC_PHYS_PAGES).max(0) as u64 * page;

            let mut stats: libc::vm_statistics64_data_t = std::mem::zeroed();
            let mut count =
                (std::mem::size_of::<libc::vm_statistics64_data_t>() / std::mem::size_of::<u32>())
                    as u32;
            let host = libc::mach_host_self();
            let kr = libc::host_statistics64(
                host,
                libc::HOST_VM_INFO64,
                &mut stats as *mut _ as *mut libc::vm_statistics64_data_t,
                &mut count,
            );
            if kr != libc::KERN_SUCCESS {
                return OsPressure {
                    total_physical: total,
                    ..Default::default()
                };
            }
            let wired = stats.wire_count as u64 * page;
            let compressed = stats.compressor_page_count as u64 * page;
            let available = stats.free_count as u64 * page;

            let mut xsw: libc::xsw_usage = std::mem::zeroed();
            let mut xsw_count =
                (std::mem::size_of::<libc::xsw_usage>() / std::mem::size_of::<u32>()) as u32;
            let _ = libc::host_statistics64(
                host,
                libc::HOST_XSW_USAGE_INFO,
                &mut xsw as *mut _ as *mut libc::vm_statistics64_data_t,
                &mut xsw_count,
            );
            let swap = xsw.xsu_used as u64;

            OsPressure {
                total_physical: total,
                wired,
                compressed,
                swap,
                available,
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        // Linux: /proc/meminfo. Best-effort.
        let mut out = OsPressure::default();
        let mut swap_total = 0u64;
        let mut swap_free = 0u64;
        if let Ok(mi) = std::fs::read_to_string("/proc/meminfo") {
            for line in mi.lines() {
                let mut it = line.splitn(2, ':');
                let key = it.next().unwrap_or("").trim();
                let val = it.next().and_then(|v| v.trim().split_whitespace().next());
                let kb: u64 = val.and_then(|v| v.parse().ok()).unwrap_or(0);
                match key {
                    "MemTotal" => out.total_physical = kb * 1024,
                    "MemAvailable" => out.available = kb * 1024,
                    "SwapTotal" => swap_total = kb * 1024,
                    "SwapFree" => swap_free = kb * 1024,
                    _ => {}
                }
            }
        }
        out.swap = swap_total.saturating_sub(swap_free);
        out
    }
}

/// Per-episode feedback (section 10). Recorded for every parallel episode so the
/// real optimal concurrency can be learned.
#[derive(Clone, Debug, Default)]
pub struct EpisodeFeedback {
    pub admitted_lane_count: usize,
    pub peak_wired_bytes: u64,
    pub peak_compressed_bytes: u64,
    pub peak_swap_bytes: u64,
    pub model_memory_bytes: u64,
    pub context_sizes: Vec<usize>,
    pub wall_time_ms: u64,
    pub aggregate_token_throughput: f64,
    pub per_lane_latency_ms: Vec<u64>,
    pub successful_work: usize,
}

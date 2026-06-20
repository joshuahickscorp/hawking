//! MoE expert access accounting and advisory page-cache eviction.
//!
//! This module is intentionally independent from a concrete model engine. The
//! first v1.0.0 launch tier wires the CLI and diagnostics while keeping
//! DeepSeek-V2-Lite behavior unchanged; Mixtral can attach real expert ranges
//! and route IDs on top of the same primitives.

use std::sync::atomic::{AtomicU64, Ordering};

/// Access counters for one routed expert.
#[derive(Debug)]
pub struct ExpertAccessStats {
    active_count: AtomicU64,
    last_active_pos: AtomicU64,
}

impl ExpertAccessStats {
    pub fn new() -> Self {
        Self {
            active_count: AtomicU64::new(0),
            last_active_pos: AtomicU64::new(0),
        }
    }

    pub fn active_count(&self) -> u64 {
        self.active_count.load(Ordering::Relaxed)
    }

    pub fn last_active_pos(&self) -> u64 {
        self.last_active_pos.load(Ordering::Relaxed)
    }
}

impl Default for ExpertAccessStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-layer access table.
#[derive(Debug)]
pub struct LayerExpertStats {
    pub experts: Vec<ExpertAccessStats>,
    pub window_size: u64,
}

/// Byte range for one expert slab within an mmap'd model file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertRange {
    pub layer: usize,
    pub expert_id: u32,
    pub byte_offset: u64,
    pub byte_size: usize,
}

/// Route accounting plus optional POSIX advisory eviction.
#[derive(Debug)]
pub struct ExpertCache {
    pub stats: Vec<LayerExpertStats>,
    pub eviction_threshold_tokens: u64,
    model_base_addr: Option<usize>,
    current_pos: AtomicU64,
}

impl ExpertCache {
    pub fn new(n_layers: usize, n_experts: usize, eviction_threshold_tokens: u64) -> Self {
        Self {
            stats: (0..n_layers)
                .map(|_| LayerExpertStats {
                    experts: (0..n_experts).map(|_| ExpertAccessStats::new()).collect(),
                    window_size: eviction_threshold_tokens,
                })
                .collect(),
            eviction_threshold_tokens,
            model_base_addr: None,
            current_pos: AtomicU64::new(0),
        }
    }

    /// Attach an mmap base pointer. `ExpertRange::byte_offset` is interpreted
    /// relative to this pointer when issuing `posix_madvise` hints.
    pub fn with_model_base_addr(mut self, model_base_addr: usize) -> Self {
        self.model_base_addr = Some(model_base_addr);
        self
    }

    /// Increment access count for a routed expert at the given layer.
    pub fn note_access(&self, layer: usize, expert_id: u32, pos: u64) {
        self.current_pos.fetch_max(pos, Ordering::Relaxed);
        let Some(layer_stats) = self.stats.get(layer) else {
            return;
        };
        let Some(expert) = layer_stats.experts.get(expert_id as usize) else {
            return;
        };
        expert.active_count.fetch_add(1, Ordering::Relaxed);
        expert.last_active_pos.store(pos, Ordering::Relaxed);
    }

    /// Run `posix_madvise(POSIX_MADV_DONTNEED)` on cold experts.
    ///
    /// If no mmap base pointer has been attached, this is a safe no-op and
    /// returns `(0, 0)`. That is the v1.0.0 partial-tier behavior for V2-Lite.
    pub fn evict_cold(&self, model_buf_offsets: &[ExpertRange]) -> (usize, usize) {
        let Some(base) = self.model_base_addr else {
            return (0, 0);
        };
        let current = self.current_pos.load(Ordering::Relaxed);
        let mut evicted = 0usize;
        let mut freed = 0usize;
        for range in model_buf_offsets {
            if !self.is_cold(*range, current) {
                continue;
            }
            if advise_range(base, range.byte_offset, range.byte_size, Advice::DontNeed) {
                evicted += 1;
                freed = freed.saturating_add(range.byte_size);
            }
        }
        (evicted, freed / (1024 * 1024))
    }

    /// Run `POSIX_MADV_WILLNEED` on predicted-hot experts.
    pub fn mark_warm(&self, model_buf_offsets: &[ExpertRange], experts: &[u32]) {
        let Some(base) = self.model_base_addr else {
            return;
        };
        for range in model_buf_offsets {
            if experts.contains(&range.expert_id) {
                let _ = advise_range(base, range.byte_offset, range.byte_size, Advice::WillNeed);
            }
        }
    }

    fn is_cold(&self, range: ExpertRange, current: u64) -> bool {
        let Some(layer_stats) = self.stats.get(range.layer) else {
            return false;
        };
        let Some(expert) = layer_stats.experts.get(range.expert_id as usize) else {
            return false;
        };
        let last = expert.last_active_pos();
        expert.active_count() > 0 && current.saturating_sub(last) >= self.eviction_threshold_tokens
    }
}

#[derive(Debug, Clone, Copy)]
enum Advice {
    DontNeed,
    WillNeed,
}

fn advise_range(base: usize, byte_offset: u64, byte_size: usize, advice: Advice) -> bool {
    if byte_size == 0 {
        return false;
    }
    let Some(addr) = base.checked_add(byte_offset as usize) else {
        return false;
    };
    platform_posix_madvise(addr, byte_size, advice)
}

#[cfg(unix)]
fn platform_posix_madvise(addr: usize, len: usize, advice: Advice) -> bool {
    use core::ffi::{c_int, c_void};

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    const POSIX_MADV_WILLNEED: c_int = 3;
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    const POSIX_MADV_DONTNEED: c_int = 4;

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    const POSIX_MADV_WILLNEED: c_int = 3;
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    const POSIX_MADV_DONTNEED: c_int = 4;

    unsafe extern "C" {
        fn posix_madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
    }

    let advice = match advice {
        Advice::DontNeed => POSIX_MADV_DONTNEED,
        Advice::WillNeed => POSIX_MADV_WILLNEED,
    };
    // Safety: this only provides an advisory hint to the OS for a caller-owned
    // mmap range. Failure is non-fatal and reported as `false`.
    unsafe { posix_madvise(addr as *mut c_void, len, advice) == 0 }
}

#[cfg(not(unix))]
fn platform_posix_madvise(_addr: usize, _len: usize, _advice: Advice) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_access_updates_counts_and_positions() {
        let cache = ExpertCache::new(2, 4, 8);
        cache.note_access(1, 3, 42);
        cache.note_access(1, 3, 43);

        let stats = &cache.stats[1].experts[3];
        assert_eq!(stats.active_count(), 2);
        assert_eq!(stats.last_active_pos(), 43);
    }

    #[test]
    fn evict_without_model_base_is_noop() {
        let cache = ExpertCache::new(1, 2, 1);
        cache.note_access(0, 1, 1);
        cache.note_access(0, 0, 8);

        let ranges = [
            ExpertRange {
                layer: 0,
                expert_id: 0,
                byte_offset: 0,
                byte_size: 4096,
            },
            ExpertRange {
                layer: 0,
                expert_id: 1,
                byte_offset: 4096,
                byte_size: 4096,
            },
        ];
        assert_eq!(cache.evict_cold(&ranges), (0, 0));
    }
}

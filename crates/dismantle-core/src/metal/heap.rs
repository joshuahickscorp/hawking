//! `MTLHeap`-backed weight residency for the QwenDense forward path.
//!
//! Scope: POC. Today every pinned weight is its own `MTLBuffer`, so the
//! Metal driver re-validates residency for each one on every dispatch.
//! This module allocates all of them from a single `MTLHeap`, so a single
//! `useHeap:` declaration on the encoder covers the entire weight set in
//! one go.
//!
//! References: llama.cpp PRs #11427 (MTLHeap weight residency) and
//! #17766 (residency-set keep-alive). Memo:
//! `memory/build_heap_residency_2026_05_25.md`.
//!
//! Not yet wired into any production forward path. The parallel entry
//! point `QwenDense::load_heap_resident` consumes this; the existing
//! `Engine::load` is untouched.
//!
//! What's heap-resident in this POC
//! --------------------------------
//! * the full GGUF mmap (Q4_K + Q6_K weight bytes — bandwidth-dominant)
//! * the embed table (f16)
//! * the final norm (f32)
//! * the LM-head matrix (f16, either tied-to-embed or explicit)
//! * every per-layer norm + bias + f16-fallback projection buffer
//!
//! What's NOT
//! ----------
//! * Optional Q4_K LM-head / Q4_K ffn_down / vocab-pruned LM-head — these
//!   require setting env vars at load and add no incremental signal to
//!   the POC parity test (which runs with all opt-ins off).
//! * The decode arena (KV cache + activation scratch) — that's allocated
//!   lazily on the first `forward_token_greedy_tcb` call and lives in
//!   `DenseDecodeArena`. Wiring it to the heap is a follow-up.
//! * `MTLResidencySet` keep-alive — metal-rs 0.29 doesn't expose it.
//!   Documented as a follow-up in the memo. `useHeap:` on the encoder
//!   already removes the per-buffer residency churn this POC targets.
//!
//! Visibility: `pub(crate)`. The struct is reachable from
//! `model::qwen_dense::load_heap_resident` but not part of the crate's
//! public surface.

#![cfg(target_os = "macos")]

use crate::{Error, Result};
use metal::{
    Buffer, Heap, HeapDescriptor, MTLCPUCacheMode, MTLHazardTrackingMode, MTLResourceOptions,
    MTLStorageMode,
};
use std::time::Instant;

use super::MetalContext;

/// Round `n` up to a multiple of `align` (which must be a power of two
/// for `MTLHeapBufferSizeAndAlign`, which it always is in practice).
fn align_up(n: u64, align: u64) -> u64 {
    if align == 0 {
        return n;
    }
    (n + align - 1) & !(align - 1)
}

/// A weight heap holding the GGUF mmap and all per-layer pinned buffers
/// for one model load. Buffers carved out of the heap have the same
/// `Buffer` type as the per-buffer allocations they replace — call sites
/// don't need to care which path produced them.
pub(crate) struct WeightHeap {
    /// The backing `MTLHeap`. Kept alive for the lifetime of the model.
    /// All sub-allocations reference this; dropping the heap before its
    /// sub-buffers is undefined.
    heap: Heap,
    /// Total heap size in bytes (used for the memo + diagnostics).
    pub total_bytes: u64,
    /// Sum of `byte_size` (un-aligned) across all allocations. Distinct
    /// from `total_bytes` because per-allocation alignment padding is not
    /// counted here.
    pub allocated_bytes: u64,
    /// Number of distinct buffer allocations carved out of the heap.
    pub n_allocations: usize,
    /// Created at construction; used by `touch()` to log the heartbeat.
    /// Real `MTLResidencySet` keep-alive is a follow-up (see memo).
    created_at: Instant,
    last_touched: Instant,
}

impl WeightHeap {
    /// Build a `WeightHeap` sized for `requested_bytes`. The heap is
    /// `StorageModeShared` (unified-memory Apple Silicon — same options
    /// as `MetalContext::new_buffer`) and untracked (`HazardTrackingMode
    /// ::Untracked`), since the caller already linearizes weight reads
    /// via Metal's command-queue ordering.
    ///
    /// `requested_bytes` should be a conservative upper bound; the heap
    /// is sized exactly to this value (no slack) because every byte
    /// counts on a 36 GB-RAM M3 Pro running concurrent workloads.
    pub fn new(ctx: &MetalContext, requested_bytes: u64) -> Result<Self> {
        let desc = HeapDescriptor::new();
        desc.set_storage_mode(MTLStorageMode::Shared);
        desc.set_cpu_cache_mode(MTLCPUCacheMode::DefaultCache);
        // Untracked — caller relies on command-queue ordering, no need
        // for the driver to insert implicit hazards.
        desc.set_hazard_tracking_mode(MTLHazardTrackingMode::Untracked);
        desc.set_size(requested_bytes);

        let heap = ctx.device().new_heap(&desc);
        if heap.size() == 0 {
            return Err(Error::Metal(format!(
                "MTLHeap allocation returned 0 size for requested {} bytes",
                requested_bytes
            )));
        }
        let now = Instant::now();
        Ok(Self {
            heap,
            total_bytes: requested_bytes,
            allocated_bytes: 0,
            n_allocations: 0,
            created_at: now,
            last_touched: now,
        })
    }

    /// Query the additional bytes required to fit `byte_size` into this
    /// heap given Metal's alignment requirements. Useful when sizing the
    /// heap up front.
    pub fn aligned_buffer_size(ctx: &MetalContext, byte_size: u64) -> u64 {
        let sa = ctx
            .device()
            .heap_buffer_size_and_align(byte_size, MTLResourceOptions::StorageModeShared);
        align_up(sa.size, sa.align.max(1))
    }

    /// Allocate `byte_size` bytes from the heap and `memcpy` `bytes` into
    /// it. Returns the resulting `Buffer` (same type as the per-buffer
    /// path; callers store these in `Option<PinnedBuffer>` slots).
    pub fn new_buffer_with_bytes(&mut self, bytes: &[u8]) -> Result<Buffer> {
        let len = bytes.len() as u64;
        let buf = self
            .heap
            .new_buffer(len, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                Error::Metal(format!(
                    "MTLHeap::new_buffer({} bytes) returned nil (heap full? total={}, allocated={})",
                    len, self.total_bytes, self.allocated_bytes
                ))
            })?;
        // SAFETY: shared-storage buffers expose `contents()` as a host
        // pointer on Apple Silicon. memcpy is the canonical write path.
        unsafe {
            let dst = buf.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        }
        self.allocated_bytes += len;
        self.n_allocations += 1;
        Ok(buf)
    }

    /// Borrow the underlying `Heap` so a compute encoder can call
    /// `useHeap:` on it. One call covers every buffer carved from the
    /// heap, replacing the per-buffer `useResource:` storm.
    pub fn heap(&self) -> &Heap {
        &self.heap
    }

    /// Heartbeat marker. With a real `MTLResidencySet` we'd re-`request`
    /// the set every 180s to keep the OS from purging the residency. For
    /// the POC we just stamp the timer and leave a hook for the real
    /// keep-alive to land later. Calling this is a no-op unless someone
    /// reads `last_touched`.
    #[allow(dead_code)]
    pub fn touch(&mut self) {
        self.last_touched = Instant::now();
    }

    #[allow(dead_code)]
    pub fn age_seconds(&self) -> f64 {
        self.created_at.elapsed().as_secs_f64()
    }
}

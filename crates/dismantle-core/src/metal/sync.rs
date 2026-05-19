//! path-to-125 L5w — `MTLSharedEvent` helper for cross-queue ordering.
//!
//! When dismantle runs Eagle4 chain decode under `multi_queue=true` AND
//! `EAGLE4_BACKEND=metal`, the head's propose dispatches run on the
//! secondary command queue while the verifier's first-layer work runs
//! on the primary queue. `SharedEventBarrier` lets a buffer on one
//! queue signal a monotonic counter and another buffer on the other
//! queue wait for it before launching its dependent kernels — the
//! standard Metal pattern for cross-queue ordering. Producing the
//! signal and consuming the wait are GPU-side operations (no CPU
//! ticks), so the overhead is a few hundred nanoseconds per barrier.
//!
//! Under the default `EAGLE4_BACKEND` (AMX via `cblas_sgemv`), the head
//! runs on CPU and never touches the GPU, so it's already overlapping
//! with verifier GPU work without needing a SharedEvent. This helper
//! is the foundation for the EAGLE4_BACKEND=metal multi-queue case
//! and for any future cross-queue scheduling outside Eagle4.

#![cfg(target_os = "macos")]

use metal::{CommandBufferRef, SharedEvent};

use crate::metal::MetalContext;

/// Single-event barrier across two command queues. Holds the
/// `SharedEvent` and the next value to wait/signal on. Both signal and
/// wait sides increment a monotonic counter; the wait side is encoded
/// before the dependent kernel and unblocks when the producer's signal
/// reaches that value.
pub struct SharedEventBarrier {
    event: SharedEvent,
    /// Monotonically-increasing counter. Each `signal` bumps this and
    /// returns the value it just signaled; `wait_for` consumes a value
    /// to block on. Wrapping around is impossible at u64 over any
    /// realistic decode session.
    counter: u64,
}

impl SharedEventBarrier {
    /// Allocate a fresh `MTLSharedEvent` against the context's device.
    /// The counter starts at 0; the first `signal()` will encode 1.
    pub fn new(ctx: &MetalContext) -> Self {
        let event = ctx.device().new_shared_event();
        event.set_signaled_value(0);
        Self { event, counter: 0 }
    }

    /// Encode a GPU-side signal on the given command buffer. Returns
    /// the value that will be signaled when this buffer's preceding
    /// kernels complete — pass that value into `encode_wait` on a
    /// buffer that is going to use the producer's output.
    pub fn encode_signal(&mut self, cb: &CommandBufferRef) -> u64 {
        self.counter += 1;
        cb.encode_signal_event(&self.event, self.counter);
        self.counter
    }

    /// Encode a GPU-side wait on the given command buffer for the
    /// specified counter value. Subsequent kernels encoded on `cb`
    /// won't dispatch until the producer's signal reaches `value`.
    pub fn encode_wait(&self, cb: &CommandBufferRef, value: u64) {
        cb.encode_wait_for_event(&self.event, value);
    }

    /// Current monotonic counter. Useful for tests that want to assert
    /// the barrier was hit a specific number of times in a decode step.
    pub fn counter(&self) -> u64 {
        self.counter
    }
}

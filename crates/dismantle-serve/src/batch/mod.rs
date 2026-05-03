//! Continuous batching: prefill/decode interleaving so concurrent
//! requests share MoE kernel launches. The MoE-specific win at batch ≥ 4.
//!
//! Phase 4 fills in the slot manager. The data structures that the
//! HTTP layer reads from are already shaped here so the seams don't
//! move when the implementation lands.

pub mod scheduler;

use dismantle_core::GenerateRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    Idle,
    Prefilling,
    Decoding,
    Finishing,
}

#[derive(Debug)]
pub struct Slot {
    pub id: u32,
    pub state: SlotState,
    pub req: Option<GenerateRequest>,
}

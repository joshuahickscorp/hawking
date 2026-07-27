//! Hawking speculative decoding pack (extracted from hawking-core / NUCLEAR PASTA).
//! Leaf crate: own Error, inlined argmax; hawking-core depends on it (no cycle).

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("model: {0}")]
    Model(String),
    #[error("not yet implemented: {0}")]
    Unimplemented(&'static str),
}
pub type Result<T> = std::result::Result<T, Error>;

/// argmax over f32 (inlined from hawking-core kernels).
pub fn argmax_f32(xs: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v { best = i; best_v = v; }
    }
    best as u32
}

// Speculative decoding: n-gram draft, EAGLE5 trained head, and shared-expert draft paths.
// The shared-expert path uses DeepSeek-V2/V3's 2 always-active experts as a free draft,
// then runs routed experts as verification. Agreed tokens accepted; mismatches roll back.

pub mod cross_tokenizer;
/// Durable sinks that accept only target-verified tokens (speculation safety).
pub mod durable;
pub mod eagle5;
pub mod eagle5_forward;
pub mod eagle_proposer;
pub mod governor;
/// Dual committed/provisional KV with rollback + rebase (speculation safety).
pub mod kv_dual;
/// Separated BASE_TRUE_TPS / ACCELERATED_ACCEPTED_TPS scoreboards.
pub mod metrics_sep;
pub mod parallel_draft;
pub mod policy;
pub mod proposal;
pub mod replay_oracle;
pub mod retrieval;
pub mod router;
pub mod safetensors_io;
pub mod shared;
/// Enumerable speculation suspension policy (one list, one place).
pub mod suspension;
pub mod suffix_array;
pub mod suffix_automaton;
/// Type-level Draft / Verified boundary (speculation safety).
pub mod token_boundary;
pub mod tree;
pub mod user_ngram;
pub mod verifier;

// Re-export the safety surface so host/engine code can depend on a short path.
pub use durable::{DurableRecord, DurableTokenSink, InMemoryDurableSink};
pub use kv_dual::{CommittedKv, DualKv, KvFingerprint, ProvisionalKv};
pub use metrics_sep::{
    AccelCostLedger, AcceleratedAcceptedTps, BaseTrueTps, SeparatedTpsScoreboard,
};
pub use suspension::{
    action_for, evaluate, policy_for, SpeculationPermit, SuspensionAction, SuspensionBoundary,
    SuspensionPoint, SUSPENSION_POLICY,
};
pub use token_boundary::{
    draft_ids, Draft, DraftTokenId, PromoteResult, TargetVerification, Verified, VerifiedTokenId,
};

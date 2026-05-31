//! Stateful core — the runtime tier that refuses the incumbents' three
//! requirements (general / stateless / static).
//!
//! Design-ahead scaffold for the two Layer-1 levers of the Throughput
//! Bible §8 ("The System-Level Shift"). See the companion design doc
//! `plans/stateful_core_design_2026_05_30.md` for the full rationale,
//! the exact decode-loop hook points, and the per-lever oracle gates.
//!
//! # Status: INTERFACES ONLY
//!
//! Every function body in this module and its submodules is `todo!()` /
//! `unimplemented!()`. This is **deliberate** and slightly ahead of the
//! project's oracle-first discipline: the interfaces are committed for
//! review *before* the oracle greenlights writing the bodies. Nothing
//! here is wired into the forward pass or the decode loop. The two
//! levers, and the oracle each waits on:
//!
//! - [`prefix_cache`] — **L1.2** cross-prompt computation reuse. A
//!   session-scoped, in-RAM KV-block store that sits *in front of* the
//!   already-shipped on-disk [`crate::cache::prefill_disk::PrefillDiskCache`].
//!   A matched prefix is **bit-identical reuse** (greedy-lossless).
//!   *Oracle:* the prefix-cache hit-rate oracle (measured separately on
//!   real coding transcripts). Build the bodies only when it shows a high
//!   shared-prefix hit rate on code.
//!
//! - [`working_set`] — **L1.1** KV cache as a living working set. An
//!   eviction-policy trait abstracting StreamingLLM / H2O / SnapKV, a
//!   bounded working-set size, and a lossless-mode escape hatch.
//!   *Oracle:* attention-mass concentration on a long-context capture.
//!   Build the bodies only when a small bounded position set captures
//!   ≥99% of attention mass per layer on Qwen2.5-3B.
//!
//! # Relationship to existing caches
//!
//! `crate::cache::prefill_disk` is the **persistence** tier (survives
//! process restarts, disk-backed, mmap'd hits). The [`prefix_cache`]
//! tier here is the **hot-session** tier (bounded RAM, zero disk I/O on
//! a hit, eventually zero-copy). They share key derivation (the rolling
//! prefix hash) so a KV block is addressable identically in both.

pub mod attn_capture;
pub mod prefix_cache;
pub mod usage_capture;
pub mod working_set;

// Re-export the primary entry points so callers can `use
// crate::stateful::{PrefixCache, KvWorkingSet, KvEvictionPolicy}` without
// reaching into submodules. Kept minimal on purpose — this is an
// interface surface, not a façade with logic.
pub use prefix_cache::{
    InMemoryPrefixCache, InsertOutcome, PrefixCache, PrefixCacheBudget, PrefixCacheStats,
    PrefixKey, PrefixMatch,
};
pub use working_set::{
    AttentionScores, EvictionAction, EvictionPlan, H2OPolicy, KvEvictionPolicy, KvWorkingSet,
    LosslessPolicy, SnapKvPolicy, StreamingLlmPolicy, WorkingSetBudget, WorkingSetCtx,
    WorkingSetMode,
};

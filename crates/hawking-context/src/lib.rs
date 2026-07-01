//! HIDE context and memory substrate (bible ch.04).
//!
//! This is the shell-side compiler described in `docs/hide-bible/04-*`: it
//! ranks sources, packs a token budget with a real reservation-aware knapsack,
//! and emits a replayable manifest. It also owns the hierarchical memory store
//! (SQLite/FTS5 + cosine vectors), the per-task context profiles, and the KV
//! reuse-banking seam to `hawking-serve`.

pub mod budget;
pub mod compiler;
pub mod embed;
pub mod fidelity;
pub mod kv;
pub mod manifest;
pub mod memory;
pub mod profiles;
pub mod recall;
pub mod sources;

pub use budget::{estimate_tokens, RegionBudget, Reservations, TokenBudget, TokenCounter};
pub use compiler::{
    CompileInput, CompiledContext, ContextCandidate, ContextCompiler, ContextSource, RealizedSpan,
};
pub use embed::{cosine, EmbeddingClient, HashingEmbeddingClient, HttpEmbeddingClient};
pub use kv::{
    CheckpointId, CheckpointMeta, EvictionChoice as KvEvictionChoice, HttpKvStore, KvCheckpoint,
    KvHandle, KvStore, KvStoreClient, KvStoreStats, KvTier, PrefixHandle, PrefixKey,
    RestoredSession, SlotId, StubKvStore, WorkingSetBudget,
};
pub use manifest::{
    ContextManifest, ContextSourceKind, ContextSpan, DropReason, DroppedContextSpan, PinState,
    SpanSignals,
};
pub use memory::{
    InMemoryMemoryStore, MemoryKind, MemoryQuery, MemoryRecord, MemoryStore, RankedMemory,
    ScoredMemory, SqliteMemoryStore,
};
pub use profiles::{
    ContextProfile, EvictionChoice, KvPrecision, OrderingPolicy, PositionPolicy, SourceWeights,
    WorkingSetMode,
};

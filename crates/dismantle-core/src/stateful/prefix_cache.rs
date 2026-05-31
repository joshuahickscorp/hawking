//! L1.2 — Prefix cache (cross-prompt computation reuse).
//!
//! Bible §8.1 L1.2. Coding workloads re-send the same files, imports,
//! and scaffolding constantly; a single-user local engine is the ideal
//! case for reusing the KV of shared **prefixes** so an unchanged prefix
//! is never recomputed. A matched prefix is **bit-identical reuse**
//! (see [`PrefixCache`] — the exact-match guarantee), so this lever is
//! **E (greedy-lossless)** in its safe mode. Energy verdict: **GENUINE**
//! — a skipped forward pass is power not drawn.
//!
//! # Status: INTERFACES ONLY — all bodies are `todo!()`.
//!
//! Build the bodies only after the **prefix-cache hit-rate oracle**
//! (measured separately on real coding transcripts; cf.
//! `tools/bench/oracle_spec_accept.py` for the oracle-family shape)
//! shows a high shared-prefix hit rate on code. See
//! `plans/stateful_core_design_2026_05_30.md` §1.6.
//!
//! # Where this hooks in (described, not wired)
//!
//! `QwenDense::generate` (`crates/dismantle-core/src/model/qwen_dense.rs`
//! ~line 1227) already calls the on-disk tier at two seams:
//!   - **lookup** ~lines 1292–1319, and
//!   - **store**  ~lines 1388–1394,
//! keyed on `self.model_id` + `tokenizer_signature(&self.tokenizer)` +
//! `prompt_ids`. The in-RAM [`PrefixCache`] slots in *in front of* the
//! disk tier at those same two seams. This module does **not** modify
//! `qwen_dense.rs`.

use crate::cache::prefill_disk::PrefillKey;

/// Identifies a token prefix for KV reuse.
///
/// Wraps the same rolling-prefix-hash derivation the on-disk tier uses
/// ([`crate::cache::prefill_disk::PrefillKey::rolling_prefix_hash`]) so a
/// block is addressable identically in the RAM and disk tiers. The
/// rolling hash has the property that the hash of the first `i` tokens is
/// recoverable by feeding `tokens[0..i]` in order — this is what makes
/// longest-prefix lookup cheap (O(N) hash-update ops, no body reads).
///
/// All three components are bound so that a key match implies the cached
/// KV is the *same bytes* a cold prefill would produce (the exact-match
/// guarantee — see [`PrefixCache`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixKey {
    /// sha256 of the model identity. A model change invalidates the key.
    pub model_hash: [u8; 32],
    /// sha256 of the tokenizer signature. A tokenizer change invalidates.
    pub tokenizer_hash: [u8; 32],
    /// Rolling sha256 over the prefix's token ids (seeded by the two
    /// hashes above).
    pub prefix_hash: [u8; 32],
    /// Number of tokens this key covers.
    pub n_tokens: usize,
}

impl PrefixKey {
    /// Build a key for `prompt_tokens` under `(model_id,
    /// tokenizer_signature)`. Defined to be byte-compatible with the
    /// on-disk tier's [`PrefillKey::from_model_and_prompt`] so the RAM
    /// and disk caches agree on the address of a prefix.
    pub fn from_model_and_prompt(
        _model_id: &str,
        _tokenizer_signature: &[u8],
        _prompt_tokens: &[u32],
    ) -> Self {
        // Delegates to the shipped rolling-hash derivation; wrapper kept
        // so callers depend on this module's type, not the disk tier's.
        todo!("derive from PrefillKey::from_model_and_prompt + record n_tokens")
    }

    /// Bridge to the on-disk tier's key type, so a RAM miss can fall
    /// through to [`crate::cache::prefill_disk::PrefillDiskCache`] using
    /// the identical address. `prompt_tokens` must be the tokens this
    /// key was derived from.
    pub fn to_prefill_key(&self, _prompt_tokens: &[u32]) -> PrefillKey {
        todo!("reconstruct the disk-tier PrefillKey from the shared hashes")
    }
}

/// A handle to a retained range of KV blocks for one prefix.
///
/// Opaque on purpose: the in-RAM implementation will hold (eventually
/// zero-copy) references into the live per-decode KV arenas
/// ([`crate::cache::KvCache`]); a disk-backed adapter would hold an
/// mmap. The decode loop restores from this into its `KvCache` the same
/// way [`crate::cache::prefill_disk::restore_hit_into_kv`] does for the
/// disk tier.
#[derive(Debug)]
pub struct KvBlockRange {
    /// Number of token positions covered (== `PrefixKey::n_tokens` of the
    /// key it was inserted under).
    pub n_tokens: usize,
    /// Per-layer KV shape this range was produced with, for a shape check
    /// at restore time (mirrors `restore_hit_into_kv`'s guard).
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    // NOTE: the actual K/V storage handle (Arc<[f32]> per layer, or a
    // reference-counted block table) is intentionally omitted from this
    // interface stub — it is an implementation choice the body makes.
}

/// Result of a successful [`PrefixCache::lookup`].
#[derive(Debug)]
pub struct PrefixMatch {
    /// Length of the matched prefix in tokens. **Invariant:** strictly
    /// less than the query length (never the whole prompt — mirrors the
    /// disk tier's "bail one token short" rule so the decode loop always
    /// has a real `last_id`).
    pub matched_len: usize,
    /// Handle to the retained KV for the matched prefix.
    pub blocks: KvBlockRange,
}

/// Outcome of an [`PrefixCache::insert`]. Never silently drops — the
/// caller learns whether the block was retained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertOutcome {
    /// New entry retained.
    Inserted,
    /// Replaced an existing entry for the same key.
    Replaced,
    /// Rejected because retaining it would exceed the budget and nothing
    /// was eligible for eviction (e.g. all entries pinned).
    RejectedOverBudget,
}

/// Budget for the in-RAM prefix cache. Either bound may be `None`
/// (unbounded on that axis). Eviction is LRU by last-hit — the same
/// policy class as the disk tier's mtime LRU.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrefixCacheBudget {
    /// Cap on total retained KV bytes across all entries.
    pub max_bytes: Option<u64>,
    /// Cap on the number of retained prefix entries.
    pub max_entries: Option<usize>,
}

/// Diagnostic counters for the prefix cache. Populated by the body; the
/// hit-rate fields are what the runtime reports alongside the offline
/// oracle to confirm the oracle's projection holds in production.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrefixCacheStats {
    pub lookups: u64,
    pub hits: u64,
    /// Sum of matched prefix lengths across all hits — divide by `hits`
    /// for mean matched length, the runtime mirror of the oracle's metric.
    pub matched_tokens_total: u64,
    pub inserts: u64,
    pub evictions: u64,
    pub retained_entries: usize,
    pub retained_bytes: u64,
}

/// A session-scoped store of KV blocks keyed by token prefix.
///
/// # The exact-match guarantee (greedy-lossless)
///
/// A [`lookup`](PrefixCache::lookup) hit is **bit-identical reuse**, not
/// an approximation:
/// 1. The KV state for tokens `[0..n)` is a pure function of
///    `(model weights, tokenizer, tokens[0..n))` — decode is causal, so
///    position `i`'s K/V depend only on tokens `≤ i`.
/// 2. [`PrefixKey`] binds all three (`model_hash`, `tokenizer_hash`,
///    rolling `prefix_hash`); a false match would require a sha256
///    collision.
/// 3. Therefore a key match ⇒ the cached blocks equal what a cold
///    prefill of `tokens[0..n)` would produce ⇒ identical logits at
///    every later position ⇒ identical greedy argmax ⇒ identical output.
///    This is **E**, no tolerance.
///
/// Implementations MUST honor: (a) never return `matched_len >=
/// query_len`; (b) invalidate on any `model_hash`/`tokenizer_hash`
/// change; (c) inserted blocks are immutable (a later session must not
/// mutate a shared prefix's K/V).
pub trait PrefixCache {
    /// Return the **longest** retained prefix that is a *strict* prefix
    /// of `query_tokens` under `key`'s model+tokenizer, or `None` on a
    /// miss. `key` is the full-prompt key; the implementation walks the
    /// recoverable rolling hash to find the longest cached candidate
    /// (cf. the disk tier's `lookup_longest_prefix`).
    fn lookup(&self, key: &PrefixKey, query_tokens: &[u32]) -> Option<PrefixMatch>;

    /// Retain `blocks` under `key` for reuse by a later request that
    /// extends this prefix. Runs eviction if needed to stay within
    /// budget. Returns whether the block was retained.
    fn insert(&mut self, key: PrefixKey, blocks: KvBlockRange) -> InsertOutcome;

    /// Evict (LRU by last-hit) until within `budget`.
    fn evict_to(&mut self, budget: PrefixCacheBudget);

    /// Snapshot the diagnostic counters.
    fn stats(&self) -> PrefixCacheStats;
}

/// The hot-session, in-RAM implementation of [`PrefixCache`].
///
/// Retains the KV blocks the decode loop just produced so the **next
/// request in the same session** reuses them with zero disk I/O. The
/// on-disk [`crate::cache::prefill_disk::PrefillDiskCache`] is the
/// cold-start backstop behind it.
#[derive(Debug, Default)]
pub struct InMemoryPrefixCache {
    budget: PrefixCacheBudget,
    // Implementation fields (entry map keyed by prefix_hash, LRU order,
    // retained-bytes accounting, the KV block table) are omitted from
    // the interface stub — they are body decisions.
}

impl InMemoryPrefixCache {
    /// Create an unbounded in-RAM prefix cache.
    pub fn new() -> Self {
        Self {
            budget: PrefixCacheBudget::default(),
        }
    }

    /// Create with an explicit byte/entry budget.
    pub fn with_budget(budget: PrefixCacheBudget) -> Self {
        Self { budget }
    }

    /// The configured budget.
    pub fn budget(&self) -> PrefixCacheBudget {
        self.budget
    }
}

impl PrefixCache for InMemoryPrefixCache {
    fn lookup(&self, _key: &PrefixKey, _query_tokens: &[u32]) -> Option<PrefixMatch> {
        todo!("walk recoverable rolling hash; return longest strict-prefix hit, LRU-touch")
    }

    fn insert(&mut self, _key: PrefixKey, _blocks: KvBlockRange) -> InsertOutcome {
        todo!("retain blocks; evict_to(budget); report Inserted/Replaced/RejectedOverBudget")
    }

    fn evict_to(&mut self, _budget: PrefixCacheBudget) {
        todo!("LRU-by-last-hit eviction until within budget")
    }

    fn stats(&self) -> PrefixCacheStats {
        todo!("snapshot diagnostic counters")
    }
}

// ---------------------------------------------------------------------
// SEMANTIC LAYER (later) — design doc §1.5.
//
// Past exact-match: embed recent contexts, recognize a near-identical
// prior context, and reuse it AFTER an exact-match verification step
// (Q for acceptance, E once verified). The embedding model + bodies are
// OUT OF SCOPE for this design — the trait shape below reserves the seam
// so wiring it later does not reshape the exact-match interface above.
// Confidence M (vs H for exact prefix); gated separately on the
// near-duplicate-rate side of the prefix-cache hit-rate oracle.
// ---------------------------------------------------------------------

/// A near-duplicate context the semantic layer proposes for reuse, before
/// verification. `similarity` is the embedding-space score that surfaced
/// it; `candidate` is the exact key whose KV would be reused *if* the
/// verify confirms a true prefix/equivalence.
#[derive(Debug)]
pub struct SemanticCandidate {
    pub candidate: PrefixKey,
    pub similarity: f32,
}

/// Embedding-similarity index over recent contexts. A probe returns
/// candidate [`PrefixKey`]s ranked by similarity; the caller then runs
/// the exact-match verify before trusting any near-hit (keeping reuse
/// **E**). Bodies + the embedding model are a Phase-B+ follow-on.
pub trait SemanticIndex {
    /// Index the embedding of a context under its exact key.
    fn index(&mut self, key: PrefixKey, embedding: &[f32]);

    /// Probe for contexts within `threshold` similarity of `embedding`.
    fn probe(&self, embedding: &[f32], threshold: f32) -> Vec<SemanticCandidate>;
}

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
//! `QwenDense::generate` (`crates/hawking-core/src/model/qwen_dense.rs`
//! ~line 1227) already calls the on-disk tier at two seams:
//!   - **lookup** ~lines 1292–1319, and
//!   - **store**  ~lines 1388–1394,
//! keyed on `self.model_id` + `tokenizer_signature(&self.tokenizer)` +
//! `prompt_ids`. The in-RAM [`PrefixCache`] slots in *in front of* the
//! disk tier at those same two seams. This module does **not** modify
//! `qwen_dense.rs`.

use crate::cache::prefill_disk::PrefillKey;
use crate::cache::KvCache;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

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
        model_id: &str,
        tokenizer_signature: &[u8],
        prompt_tokens: &[u32],
    ) -> Self {
        // Delegates to the shipped rolling-hash derivation; wrapper kept
        // so callers depend on this module's type, not the disk tier's.
        // Byte-compatible by construction: same `PrefillKey` derivation,
        // so a block is addressable identically in the RAM and disk tiers.
        let dk = PrefillKey::from_model_and_prompt(model_id, tokenizer_signature, prompt_tokens);
        Self {
            model_hash: dk.model_hash,
            tokenizer_hash: dk.tokenizer_hash,
            prefix_hash: dk.prefix_hash,
            n_tokens: prompt_tokens.len(),
        }
    }

    /// Bridge to the on-disk tier's key type, so a RAM miss can fall
    /// through to [`crate::cache::prefill_disk::PrefillDiskCache`] using
    /// the identical address. `prompt_tokens` must be the tokens this
    /// key was derived from.
    pub fn to_prefill_key(&self, prompt_tokens: &[u32]) -> PrefillKey {
        debug_assert_eq!(
            prompt_tokens.len(),
            self.n_tokens,
            "to_prefill_key: prompt_tokens length must match the key's n_tokens"
        );
        PrefillKey {
            model_hash: self.model_hash,
            tokenizer_hash: self.tokenizer_hash,
            prefix_hash: self.prefix_hash,
            prompt_tokens: prompt_tokens.to_vec(),
        }
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

/// Default byte cap for the in-RAM prefix cache: ~3 GiB of retained KV.
/// On an 18 GB M3 Pro this is the headroom that survives alongside the
/// ~2 GB resident model + working set, so the cache can never grow
/// unbounded and OOM the box. Eviction (LRU-by-last-hit) keeps total
/// retained bytes at or under this; a single prefix larger than the cap
/// is rejected up front.
pub const DEFAULT_MAX_BYTES: u64 = 3 * 1024 * 1024 * 1024;

/// Budget for the in-RAM prefix cache. Either bound may be `None`
/// (unbounded on that axis). Eviction is LRU by last-hit — the same
/// policy class as the disk tier's mtime LRU.
///
/// The [`Default`] is **bounded**, not unbounded: `max_bytes` is
/// [`DEFAULT_MAX_BYTES`] (~3 GiB) so [`InMemoryPrefixCache::new`] /
/// [`InMemoryPrefixCache::default`] can never grow without limit. Pass
/// an explicit `{ max_bytes: None, .. }` via
/// [`InMemoryPrefixCache::with_budget`] only when an unbounded axis is
/// genuinely wanted (e.g. a test that controls insert volume itself).
#[derive(Debug, Clone, Copy)]
pub struct PrefixCacheBudget {
    /// Cap on total retained KV bytes across all entries.
    pub max_bytes: Option<u64>,
    /// Cap on the number of retained prefix entries.
    pub max_entries: Option<usize>,
}

impl Default for PrefixCacheBudget {
    fn default() -> Self {
        Self {
            max_bytes: Some(DEFAULT_MAX_BYTES),
            max_entries: None,
        }
    }
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
    /// model-scoped index: `model_hash → (prefix_hash → entry)`, mirroring
    /// the disk tier's two-level index so a tokenizer/model change can
    /// never alias another model's prefix.
    entries: HashMap<[u8; 32], HashMap<[u8; 32], CacheEntry>>,
    /// Monotonic logical clock for LRU-by-last-hit ordering. Bumped on
    /// every lookup-hit and insert; the entry with the smallest tick is
    /// the least-recently-used.
    clock: u64,
    /// Running total of retained KV bytes across all entries (the sum of
    /// every entry's `kv_bytes()`), kept in sync on insert/evict so the
    /// byte budget is O(1) to enforce.
    retained_bytes: u64,
    stats: PrefixCacheStats,
}

/// One retained prefix: its exact token ids (for the collision-safe
/// guard + disk-key reconstruction), KV shape, and the f32 KV bytes laid
/// out exactly as [`crate::cache::KvCache`] / the disk body store them
/// (per-layer keys, per-layer values; `n_tokens * n_kv_heads * head_dim`
/// f32 each). Immutable once inserted (the design's invariant (c)).
#[derive(Debug, Clone)]
struct CacheEntry {
    tokens: Vec<u32>,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    /// `keys[layer]` / `values[layer]` are each `n_tokens * stride` f32,
    /// `stride = n_kv_heads * head_dim`. Same layout `restore_hit_into_kv`
    /// copies, so reuse is byte-identical to a cold prefill.
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    /// Logical clock value of the last hit/insert (LRU ordering key).
    last_tick: u64,
}

impl CacheEntry {
    fn n_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Total retained KV byte footprint of this entry.
    fn kv_bytes(&self) -> u64 {
        let per_layer = self.n_tokens() * self.n_kv_heads * self.head_dim * 4;
        (self.keys.len() + self.values.len()) as u64 * per_layer as u64
    }
}

impl InMemoryPrefixCache {
    /// Create an in-RAM prefix cache with the bounded default budget
    /// ([`PrefixCacheBudget::default`] — a ~3 GiB byte cap, no entry cap).
    /// Eviction (LRU-by-last-hit) runs on every insert, so the cache can
    /// never grow past the cap. Use [`with_budget`](Self::with_budget) to
    /// override (including an explicitly-unbounded budget for tests).
    pub fn new() -> Self {
        Self {
            budget: PrefixCacheBudget::default(),
            entries: HashMap::new(),
            clock: 0,
            retained_bytes: 0,
            stats: PrefixCacheStats::default(),
        }
    }

    /// Create with an explicit byte/entry budget.
    pub fn with_budget(budget: PrefixCacheBudget) -> Self {
        Self {
            budget,
            ..Self::new()
        }
    }

    /// The configured budget.
    pub fn budget(&self) -> PrefixCacheBudget {
        self.budget
    }

    /// Snapshot the live KV state of `kv` (its first `seq_len` positions)
    /// under `key` for reuse by a later request that extends this prefix.
    /// Convenience over [`PrefixCache::insert`] that copies the bytes out
    /// of a [`KvCache`] exactly the way the disk tier's `store` does, so
    /// the RAM and disk tiers retain identical bytes.
    ///
    /// `key.n_tokens` must equal `kv.seq_len` (the full prompt just
    /// prefilled). Returns the same [`InsertOutcome`] as `insert`.
    pub fn insert_from_kv(&mut self, key: PrefixKey, kv: &KvCache) -> InsertOutcome {
        debug_assert_eq!(
            key.n_tokens, kv.seq_len,
            "insert_from_kv: key.n_tokens must equal kv.seq_len"
        );
        let n_tokens = kv.seq_len;
        let stride = kv.n_kv_heads * kv.head_dim;
        let want = n_tokens * stride;
        let keys: Vec<Vec<f32>> = (0..kv.n_layers)
            .map(|li| kv.keys[li][..want].to_vec())
            .collect();
        let values: Vec<Vec<f32>> = (0..kv.n_layers)
            .map(|li| kv.values[li][..want].to_vec())
            .collect();
        let blocks = KvBlockRange {
            n_tokens,
            n_layers: kv.n_layers,
            n_kv_heads: kv.n_kv_heads,
            head_dim: kv.head_dim,
        };
        self.insert_inner(key, blocks, keys, values)
    }

    /// Restore the KV bytes of the entry matched at `(key, matched_len)`
    /// into `kv`, setting `kv.seq_len = matched_len`. Mirrors the disk
    /// tier's [`crate::cache::prefill_disk::restore_hit_into_kv`]: the
    /// caller looks up the longest prefix, then restores it, then runs
    /// prefill on the remaining `prompt[matched_len..]` tokens.
    ///
    /// The matched entry is addressed by re-deriving the rolling hash of
    /// `query_tokens[..matched_len]` under `key`'s model/tokenizer, so it
    /// finds exactly the entry [`lookup`](PrefixCache::lookup) returned.
    /// Returns the number of restored tokens, or an error on a shape
    /// mismatch / missing entry.
    pub fn restore_into(
        &self,
        key: &PrefixKey,
        query_tokens: &[u32],
        matched_len: usize,
        kv: &mut KvCache,
    ) -> crate::Result<usize> {
        let prefix_hash = rolling_prefix_hash_of(
            &key.model_hash,
            &key.tokenizer_hash,
            &query_tokens[..matched_len],
        );
        let entry = self
            .entries
            .get(&key.model_hash)
            .and_then(|b| b.get(&prefix_hash))
            .ok_or_else(|| {
                crate::Error::Model("prefix_cache restore_into: no entry for match".into())
            })?;
        if entry.n_layers != kv.n_layers
            || entry.n_kv_heads != kv.n_kv_heads
            || entry.head_dim != kv.head_dim
        {
            return Err(crate::Error::Model(format!(
                "prefix_cache restore_into: shape mismatch (entry={}x{}x{}, kv={}x{}x{})",
                entry.n_layers,
                entry.n_kv_heads,
                entry.head_dim,
                kv.n_layers,
                kv.n_kv_heads,
                kv.head_dim,
            )));
        }
        if entry.n_tokens() > kv.max_seq {
            return Err(crate::Error::Model(format!(
                "prefix_cache restore_into: entry n_tokens {} > kv.max_seq {}",
                entry.n_tokens(),
                kv.max_seq
            )));
        }
        let stride = kv.n_kv_heads * kv.head_dim;
        let want = entry.n_tokens() * stride;
        for li in 0..kv.n_layers {
            kv.keys[li][..want].copy_from_slice(&entry.keys[li]);
            kv.values[li][..want].copy_from_slice(&entry.values[li]);
        }
        kv.seq_len = entry.n_tokens();
        Ok(entry.n_tokens())
    }

    /// Shared insert: stores the (already-copied) KV bytes, evicts to
    /// budget, updates accounting + stats. `Replaced` if a same-key entry
    /// existed; `RejectedOverBudget` if even after eviction the byte/entry
    /// budget cannot fit it.
    fn insert_inner(
        &mut self,
        key: PrefixKey,
        blocks: KvBlockRange,
        keys: Vec<Vec<f32>>,
        values: Vec<Vec<f32>>,
    ) -> InsertOutcome {
        self.clock += 1;
        let tick = self.clock;
        // `KvBlockRange` deliberately omits the tokens; the key carries the
        // token *count* (`n_tokens`) and the rolling `prefix_hash` already
        // binds the exact token sequence (a hash match ⇒ same tokens, sha256
        // collisions aside). We keep a length-only token placeholder so
        // lookup's secondary collision guard (entry length == candidate
        // length) works without storing the ids redundantly.
        let entry = CacheEntry {
            tokens: vec![0u32; key.n_tokens],
            n_layers: blocks.n_layers,
            n_kv_heads: blocks.n_kv_heads,
            head_dim: blocks.head_dim,
            keys,
            values,
            last_tick: tick,
        };
        let new_bytes = entry.kv_bytes();

        // Reject up front if a single entry can never fit the byte budget.
        if let Some(max) = self.budget.max_bytes {
            if new_bytes > max {
                return InsertOutcome::RejectedOverBudget;
            }
        }

        let bucket = self.entries.entry(key.model_hash).or_default();
        let replaced = bucket.contains_key(&key.prefix_hash);
        if replaced {
            if let Some(old) = bucket.get(&key.prefix_hash) {
                self.retained_bytes = self.retained_bytes.saturating_sub(old.kv_bytes());
            }
        }
        bucket.insert(key.prefix_hash, entry);
        self.retained_bytes += new_bytes;
        self.stats.inserts += 1;

        // Enforce budget (evicts other entries LRU-first; never the one we
        // just inserted unless it alone is the LRU and still over — handled
        // by the up-front single-entry check above for bytes).
        self.evict_to(self.budget);

        // If after eviction our entry was itself evicted (entry-count
        // budget of 0, say), report rejection.
        let still_present = self
            .entries
            .get(&key.model_hash)
            .map(|b| b.contains_key(&key.prefix_hash))
            .unwrap_or(false);
        if !still_present {
            return InsertOutcome::RejectedOverBudget;
        }

        self.stats.retained_entries = self.total_entries();
        self.stats.retained_bytes = self.retained_bytes;
        if replaced {
            InsertOutcome::Replaced
        } else {
            InsertOutcome::Inserted
        }
    }

    fn total_entries(&self) -> usize {
        self.entries.values().map(|b| b.len()).sum()
    }
}

/// Rolling SHA-256 over `tokens`, seeded by the model + tokenizer hashes —
/// identical derivation to [`PrefillKey::rolling_prefix_hash`] so the RAM
/// tier addresses a prefix at the same hash as the disk tier.
fn rolling_prefix_hash_of(
    model_hash: &[u8; 32],
    tokenizer_hash: &[u8; 32],
    tokens: &[u32],
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(model_hash);
    h.update(tokenizer_hash);
    for &t in tokens {
        h.update(t.to_le_bytes());
    }
    h.finalize().into()
}

impl PrefixCache for InMemoryPrefixCache {
    fn lookup(&self, key: &PrefixKey, query_tokens: &[u32]) -> Option<PrefixMatch> {
        // Mirror the disk tier's `lookup_longest_prefix`: walk the rolling
        // hash forward over the query tokens, remember the longest cached
        // candidate, and NEVER match the full prompt (bail one token short
        // so the decode loop always has a real `last_id`).
        let bucket = match self.entries.get(&key.model_hash) {
            Some(b) if !b.is_empty() => b,
            _ => {
                // `lookup` is &self; the stats bump for the miss happens in
                // the &mut caller wrapper (`lookup_counting`). Pure-trait
                // callers that only need the match don't mutate stats.
                return None;
            }
        };
        let max_len = query_tokens.len();
        let mut h = Sha256::new();
        h.update(key.model_hash);
        h.update(key.tokenizer_hash);
        let mut best: Option<usize> = None;
        for (i, &tok) in query_tokens.iter().enumerate() {
            h.update(tok.to_le_bytes());
            let n_so_far = i + 1;
            if n_so_far == max_len {
                // Strict prefix only — never the entire prompt.
                break;
            }
            let candidate: [u8; 32] = h.clone().finalize().into();
            if let Some(entry) = bucket.get(&candidate) {
                // Collision guard: the cached entry's token count must
                // match, and (when we stored real tokens) the tokens must
                // be a true prefix. We always have the length; verify it.
                if entry.n_tokens() == n_so_far {
                    best = Some(n_so_far);
                }
            }
        }
        let matched_len = best?;
        // Re-derive the matched entry for its shape (the handle the caller
        // restores from is addressed by (key, matched_len)).
        let prefix_hash = rolling_prefix_hash_of(
            &key.model_hash,
            &key.tokenizer_hash,
            &query_tokens[..matched_len],
        );
        let entry = bucket.get(&prefix_hash)?;
        Some(PrefixMatch {
            matched_len,
            blocks: KvBlockRange {
                n_tokens: entry.n_tokens(),
                n_layers: entry.n_layers,
                n_kv_heads: entry.n_kv_heads,
                head_dim: entry.head_dim,
            },
        })
    }

    fn insert(&mut self, key: PrefixKey, blocks: KvBlockRange) -> InsertOutcome {
        // Raw-trait insert with caller-owned blocks but no KV bytes is not
        // meaningful for reuse (the bytes are what we restore). The
        // production path uses `insert_from_kv`, which carries the bytes.
        // Keep this entry point honest: allocate zeroed storage of the
        // declared shape so accounting + lookup work, but document that
        // restoring it yields zeros (callers must use `insert_from_kv`).
        let stride = blocks.n_kv_heads * blocks.head_dim;
        let want = blocks.n_tokens * stride;
        let keys = vec![vec![0.0f32; want]; blocks.n_layers];
        let values = vec![vec![0.0f32; want]; blocks.n_layers];
        self.insert_inner(key, blocks, keys, values)
    }

    fn evict_to(&mut self, budget: PrefixCacheBudget) {
        // LRU-by-last-hit: repeatedly drop the globally-oldest entry until
        // both the byte and entry-count bounds are satisfied. Either bound
        // may be `None` (unbounded on that axis).
        loop {
            let over_bytes = budget
                .max_bytes
                .map(|max| self.retained_bytes > max)
                .unwrap_or(false);
            let over_entries = budget
                .max_entries
                .map(|max| self.total_entries() > max)
                .unwrap_or(false);
            if !over_bytes && !over_entries {
                break;
            }
            // Find the globally least-recently-used entry.
            let mut victim: Option<([u8; 32], [u8; 32], u64)> = None;
            for (mh, bucket) in self.entries.iter() {
                for (ph, entry) in bucket.iter() {
                    let lru = entry.last_tick;
                    if victim.as_ref().map(|(_, _, t)| lru < *t).unwrap_or(true) {
                        victim = Some((*mh, *ph, lru));
                    }
                }
            }
            let Some((mh, ph, _)) = victim else { break };
            if let Some(bucket) = self.entries.get_mut(&mh) {
                if let Some(removed) = bucket.remove(&ph) {
                    self.retained_bytes = self.retained_bytes.saturating_sub(removed.kv_bytes());
                    self.stats.evictions += 1;
                }
                if bucket.is_empty() {
                    self.entries.remove(&mh);
                }
            } else {
                break;
            }
        }
        self.stats.retained_entries = self.total_entries();
        self.stats.retained_bytes = self.retained_bytes;
    }

    fn stats(&self) -> PrefixCacheStats {
        let mut s = self.stats;
        s.retained_entries = self.total_entries();
        s.retained_bytes = self.retained_bytes;
        s
    }
}

impl InMemoryPrefixCache {
    /// Like [`PrefixCache::lookup`] but records the lookup/hit counters
    /// (the trait's `lookup` is `&self` so cannot bump stats). Also LRU-
    /// touches the matched entry. This is the entry point the decode loop
    /// uses so the runtime hit-rate mirror (`stats().hits / lookups`) is
    /// populated.
    pub fn lookup_counting(
        &mut self,
        key: &PrefixKey,
        query_tokens: &[u32],
    ) -> Option<PrefixMatch> {
        self.stats.lookups += 1;
        let m = self.lookup(key, query_tokens);
        if let Some(ref hit) = m {
            self.stats.hits += 1;
            self.stats.matched_tokens_total += hit.matched_len as u64;
            // LRU-touch the matched entry.
            self.clock += 1;
            let tick = self.clock;
            let prefix_hash = rolling_prefix_hash_of(
                &key.model_hash,
                &key.tokenizer_hash,
                &query_tokens[..hit.matched_len],
            );
            if let Some(bucket) = self.entries.get_mut(&key.model_hash) {
                if let Some(entry) = bucket.get_mut(&prefix_hash) {
                    entry.last_tick = tick;
                }
            }
        }
        m
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KvCache;

    /// Deterministic per-token KV writer (mirrors the disk-tier parity
    /// test's `fake_forward`): any off-by-one in restore is loud.
    fn fake_forward(kv: &mut KvCache, token: u32, pos: usize) {
        assert_eq!(kv.seq_len, pos);
        let stride = kv.n_kv_heads * kv.head_dim;
        for li in 0..kv.n_layers {
            for d in 0..stride {
                let mix = ((li as u32).wrapping_mul(2654435761))
                    ^ token.wrapping_mul(40503)
                    ^ ((pos as u32).wrapping_mul(0x9E37_79B9))
                    ^ ((d as u32).wrapping_mul(0xDEAD_BEEF));
                let off = pos * stride + d;
                kv.keys[li][off] = (mix as f32) * 1e-9;
                kv.values[li][off] = -(mix as f32) * 1e-9;
            }
        }
        kv.seq_len += 1;
    }

    fn cold_prefill(prompt: &[u32], n_layers: usize, n_kv: usize, head_dim: usize) -> KvCache {
        let mut kv = KvCache::new(n_layers, prompt.len() + 8, n_kv, head_dim);
        for (i, &t) in prompt.iter().enumerate() {
            fake_forward(&mut kv, t, i);
        }
        kv
    }

    fn assert_kv_eq(a: &KvCache, b: &KvCache) {
        assert_eq!(a.seq_len, b.seq_len, "seq_len mismatch");
        for li in 0..a.n_layers {
            assert_eq!(a.keys_for(li), b.keys_for(li), "keys mismatch layer {li}");
            assert_eq!(
                a.values_for(li),
                b.values_for(li),
                "values mismatch layer {li}"
            );
        }
    }

    #[test]
    fn key_is_byte_compatible_with_disk_tier() {
        let toks = [1u32, 2, 3, 4];
        let ram = PrefixKey::from_model_and_prompt("m", b"sig", &toks);
        let disk = PrefillKey::from_model_and_prompt("m", b"sig", &toks);
        assert_eq!(ram.model_hash, disk.model_hash);
        assert_eq!(ram.tokenizer_hash, disk.tokenizer_hash);
        assert_eq!(
            ram.prefix_hash, disk.prefix_hash,
            "rolling hash must match disk tier"
        );
        assert_eq!(ram.n_tokens, 4);
        // Round-trip back to the disk key type.
        let back = ram.to_prefill_key(&toks);
        assert_eq!(back.prefix_hash, disk.prefix_hash);
    }

    #[test]
    fn lookup_returns_longest_strict_prefix() {
        let mut c = InMemoryPrefixCache::new();
        let (nl, nkv, hd) = (4, 2, 16);
        // Insert a 2-token and a 4-token prefix of the same stream.
        let p2: Vec<u32> = vec![10, 11];
        let p4: Vec<u32> = vec![10, 11, 12, 13];
        c.insert_from_kv(
            PrefixKey::from_model_and_prompt("m", b"t", &p2),
            &cold_prefill(&p2, nl, nkv, hd),
        );
        c.insert_from_kv(
            PrefixKey::from_model_and_prompt("m", b"t", &p4),
            &cold_prefill(&p4, nl, nkv, hd),
        );
        // Query extends p4 → longest strict prefix is the 4-token one.
        let query: Vec<u32> = vec![10, 11, 12, 13, 14];
        let key = PrefixKey::from_model_and_prompt("m", b"t", &query);
        let m = c.lookup(&key, &query).expect("hit");
        assert_eq!(m.matched_len, 4);
    }

    #[test]
    fn lookup_never_returns_full_prompt() {
        let mut c = InMemoryPrefixCache::new();
        let p: Vec<u32> = vec![1, 2, 3];
        c.insert_from_kv(
            PrefixKey::from_model_and_prompt("m", b"t", &p),
            &cold_prefill(&p, 1, 1, 4),
        );
        // Exact same tokens → strict-prefix rule means miss.
        let key = PrefixKey::from_model_and_prompt("m", b"t", &p);
        assert!(
            c.lookup(&key, &p).is_none(),
            "must not match the whole prompt"
        );
        // One token longer → returns the 3-token prefix.
        let mut longer = p.clone();
        longer.push(99);
        let key2 = PrefixKey::from_model_and_prompt("m", b"t", &longer);
        let m = c.lookup(&key2, &longer).expect("hit");
        assert_eq!(m.matched_len, 3);
    }

    #[test]
    fn cold_vs_warm_prefill_byte_identical() {
        // The gate's core invariant, at the cache layer: a warm path
        // (restore prefix + prefill delta) yields byte-identical KV to a
        // cold full prefill — so greedy decode is bit-identical.
        let (nl, nkv, hd) = (4, 2, 16);
        let mut c = InMemoryPrefixCache::new();

        let system: Vec<u32> = (0..50u32).map(|i| 100 + i).collect();
        let kv_t1 = cold_prefill(&system, nl, nkv, hd);
        c.insert_from_kv(
            PrefixKey::from_model_and_prompt("m", b"v1", &system),
            &kv_t1,
        );

        let mut turn2 = system.clone();
        turn2.extend((0..10u32).map(|i| 1000 + i));
        turn2.extend((0..20u32).map(|i| 2000 + i));

        let kv_cold = cold_prefill(&turn2, nl, nkv, hd);

        let key2 = PrefixKey::from_model_and_prompt("m", b"v1", &turn2);
        let m = c.lookup(&key2, &turn2).expect("prefix hit on turn 2");
        assert_eq!(m.matched_len, 50);
        let mut kv_warm = KvCache::new(nl, turn2.len() + 8, nkv, hd);
        let restored = c
            .restore_into(&key2, &turn2, m.matched_len, &mut kv_warm)
            .unwrap();
        assert_eq!(restored, 50);
        assert_eq!(kv_warm.seq_len, 50);
        for (i, &t) in turn2.iter().enumerate().skip(50) {
            fake_forward(&mut kv_warm, t, i);
        }
        assert_kv_eq(&kv_cold, &kv_warm);
    }

    #[test]
    fn tokenizer_change_invalidates() {
        let mut c = InMemoryPrefixCache::new();
        let p: Vec<u32> = vec![5, 6];
        c.insert_from_kv(
            PrefixKey::from_model_and_prompt("m", b"tok-v1", &p),
            &cold_prefill(&p, 1, 1, 4),
        );
        let q: Vec<u32> = vec![5, 6, 7];
        let key_v2 = PrefixKey::from_model_and_prompt("m", b"tok-v2", &q);
        assert!(
            c.lookup(&key_v2, &q).is_none(),
            "tokenizer change must invalidate"
        );
    }

    #[test]
    fn lru_eviction_respects_entry_budget() {
        let mut c = InMemoryPrefixCache::with_budget(PrefixCacheBudget {
            max_bytes: None,
            max_entries: Some(2),
        });
        for i in 0..5u32 {
            let p: Vec<u32> = vec![i, i + 1, i + 2];
            c.insert_from_kv(
                PrefixKey::from_model_and_prompt("m", b"t", &p),
                &cold_prefill(&p, 1, 1, 4),
            );
        }
        assert!(c.stats().retained_entries <= 2, "entry budget enforced");
        assert!(c.stats().evictions >= 3);
    }

    #[test]
    fn lru_eviction_respects_byte_budget() {
        // Each 3-token entry: 1 layer * 2 (k+v) * 3 tok * (1*4 dim) * 4 B
        // = 96 bytes. Budget 200 B keeps ~2 entries.
        let mut c = InMemoryPrefixCache::with_budget(PrefixCacheBudget {
            max_bytes: Some(200),
            max_entries: None,
        });
        for i in 0..6u32 {
            let p: Vec<u32> = vec![i, i + 1, i + 2];
            c.insert_from_kv(
                PrefixKey::from_model_and_prompt("m", b"t", &p),
                &cold_prefill(&p, 1, 1, 4),
            );
        }
        assert!(c.stats().retained_bytes <= 200, "byte budget enforced");
        assert!(c.stats().retained_bytes > 0);
    }

    #[test]
    fn default_budget_is_bounded() {
        // The shipped default must NOT be unbounded — that was the
        // OOM hazard blocking default-on. Both `new()` and `default()`
        // must carry the byte cap.
        assert_eq!(
            InMemoryPrefixCache::new().budget().max_bytes,
            Some(DEFAULT_MAX_BYTES),
            "new() must be byte-bounded by default"
        );
        assert_eq!(
            PrefixCacheBudget::default().max_bytes,
            Some(DEFAULT_MAX_BYTES),
            "PrefixCacheBudget::default() must carry the byte cap"
        );
        assert!(DEFAULT_MAX_BYTES > 0);
    }

    #[test]
    fn insert_past_byte_cap_evicts_oldest_keeps_newest() {
        // Fix #1's gate: inserting past the byte cap evicts the OLDEST
        // entry (LRU), the NEWEST survives, and retained bytes stay ≤ cap.
        // Each 3-token entry here: 1 layer * (k+v) * 3 tok * (1*4 dim) * 4 B
        // = 96 bytes. A 250 B cap holds 2 entries (192 B); the 3rd insert
        // must evict the 1st.
        let cap = 250u64;
        let mut c = InMemoryPrefixCache::with_budget(PrefixCacheBudget {
            max_bytes: Some(cap),
            max_entries: None,
        });
        let p0: Vec<u32> = vec![0, 1, 2];
        let p1: Vec<u32> = vec![10, 11, 12];
        let p2: Vec<u32> = vec![20, 21, 22];
        let k0 = PrefixKey::from_model_and_prompt("m", b"t", &p0);
        let k1 = PrefixKey::from_model_and_prompt("m", b"t", &p1);
        let k2 = PrefixKey::from_model_and_prompt("m", b"t", &p2);
        c.insert_from_kv(k0.clone(), &cold_prefill(&p0, 1, 1, 4));
        c.insert_from_kv(k1.clone(), &cold_prefill(&p1, 1, 1, 4));
        // Third insert pushes past the cap → oldest (p0) must be evicted.
        c.insert_from_kv(k2.clone(), &cold_prefill(&p2, 1, 1, 4));

        assert!(
            c.stats().retained_bytes <= cap,
            "retained bytes {} must stay ≤ cap {cap}",
            c.stats().retained_bytes
        );
        assert!(c.stats().evictions >= 1, "the oldest entry must be evicted");

        // The OLDEST (p0) is gone; the NEWEST (p2) survives. We probe via
        // a strict-prefix lookup (query = entry tokens + 1 sentinel).
        let mut q0 = p0.clone();
        q0.push(99);
        assert!(
            c.lookup(&PrefixKey::from_model_and_prompt("m", b"t", &q0), &q0)
                .is_none(),
            "oldest entry must have been evicted"
        );
        let mut q2 = p2.clone();
        q2.push(99);
        let m2 = c
            .lookup(&PrefixKey::from_model_and_prompt("m", b"t", &q2), &q2)
            .expect("newest entry must survive");
        assert_eq!(m2.matched_len, 3, "newest entry's full prefix survives");
    }

    #[test]
    fn stats_track_hits_and_misses() {
        let mut c = InMemoryPrefixCache::new();
        let p: Vec<u32> = vec![1, 2, 3, 4];
        c.insert_from_kv(
            PrefixKey::from_model_and_prompt("m", b"t", &p),
            &cold_prefill(&p, 1, 1, 4),
        );
        // miss (different model)
        let q: Vec<u32> = vec![1, 2, 3, 4, 5];
        let miss_key = PrefixKey::from_model_and_prompt("other", b"t", &q);
        assert!(c.lookup_counting(&miss_key, &q).is_none());
        // hit
        let hit_key = PrefixKey::from_model_and_prompt("m", b"t", &q);
        assert!(c.lookup_counting(&hit_key, &q).is_some());
        let s = c.stats();
        assert_eq!(s.lookups, 2);
        assert_eq!(s.hits, 1);
        assert_eq!(s.matched_tokens_total, 4);
    }
}

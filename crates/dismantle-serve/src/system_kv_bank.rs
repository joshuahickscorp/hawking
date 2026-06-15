//! Track 5.2 — System-prompt KV bank.
//!
//! A serve-lifetime registry that remembers, for each FIXED leading prefix
//! span (the system / instruction prompt that many requests share), which
//! decode SLOT most recently held copyable KV for that span. Unlike
//! `scheduler::PrefixIndex` (Track 5.1), which only matches against slots
//! that are CURRENTLY active, this bank survives a source request finishing
//! — so a serial chat workload (one request at a time, identical system
//! prompt) still gets shared-prefix reuse instead of re-prefilling the
//! system block every turn.
//!
//! # What this is NOT
//! It stores ZERO KV bytes. It is a `hash(prefix) -> source_slot` routing
//! hint. The hit is ALWAYS re-verified downstream by the bit-identical
//! `Engine::copy_kv_prefix_to_slot` + `prefill_slot_from_pos` path, so a
//! (vanishingly unlikely) hash false-positive cannot corrupt output: a stale
//! `source_slot` simply fails the copy and the serve loop falls back to a
//! cold prefill from position 0. This keeps the lever **greedy-lossless (E)**.
//!
//! The detached, slot-independent KV-block store (so reuse survives even when
//! NO slot currently holds the bytes) is the deferred half of 5.2 — it lands
//! in the model/arena layer (`qwen_dense.rs` / `dense_decode_arena.rs`) and is
//! out of scope here. This bank is the routing index that store plugs into.

use std::collections::HashMap;

/// Default minimum leading-prefix length (tokens) to bank. Mirrors
/// `dismantle_core::sidecar::PREFIX_REUSE_MIN_TOKENS` (= 8): a span shorter
/// than this is not worth a copy + from-pos prefill.
pub const DEFAULT_MIN_PREFIX_TOKENS: usize = 8;

/// Default cap on distinct banked prefixes. A handful of system prompts is
/// the norm; the LRU keeps the bank from growing without bound across a long
/// server lifetime.
pub const DEFAULT_MAX_ENTRIES: usize = 64;

/// Tuning for the bank.
#[derive(Debug, Clone, Copy)]
pub struct BankConfig {
    /// Reject (do not bank, do not match) prefixes shorter than this.
    pub min_prefix_tokens: usize,
    /// LRU-evict down to this many distinct prefixes after each insert.
    pub max_entries: usize,
}

impl Default for BankConfig {
    fn default() -> Self {
        Self {
            min_prefix_tokens: DEFAULT_MIN_PREFIX_TOKENS,
            max_entries: DEFAULT_MAX_ENTRIES,
        }
    }
}

/// One banked prefix: which slot last held it + bookkeeping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BankEntry {
    /// Number of leading tokens this entry covers (== the span hashed).
    pub prefix_len: usize,
    /// Slot id that most recently held copyable KV for this prefix. The
    /// serve loop passes this as `src_slot` to `copy_kv_prefix_to_slot`.
    pub source_slot: u32,
    /// LRU clock value of the last record/hit (smallest == least recent).
    pub last_tick: u64,
    /// Lifetime lookup-hits for this prefix (diagnostics / /metrics).
    pub hits: u64,
}

/// Outcome of a [`SystemPromptKvBank::record`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordOutcome {
    /// A new prefix was banked.
    Inserted,
    /// An existing prefix's `source_slot` was refreshed.
    Updated,
    /// Prefix shorter than `min_prefix_tokens`; nothing banked.
    TooShort,
}

/// Aggregate counters (mirrors `LaneStats` style; surfaced via /metrics).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BankStats {
    pub lookups: u64,
    pub hits: u64,
    pub records: u64,
    pub evictions: u64,
    pub entries: usize,
}

/// A serve-lifetime hash(prefix) -> source-slot registry. Pure data; no model.
#[derive(Debug, Default)]
pub struct SystemPromptKvBank {
    cfg: BankConfig,
    /// prefix-hash -> entry. Hash is the 128-bit fold of the leading span.
    entries: HashMap<u128, BankEntry>,
    clock: u64,
    stats: BankStats,
}

impl SystemPromptKvBank {
    pub fn new() -> Self {
        Self::with_config(BankConfig::default())
    }

    pub fn with_config(cfg: BankConfig) -> Self {
        Self {
            cfg,
            entries: HashMap::new(),
            clock: 0,
            stats: BankStats::default(),
        }
    }

    pub fn config(&self) -> BankConfig {
        self.cfg
    }

    /// Stable 128-bit content hash of the FIRST `prefix_len` tokens — the
    /// fixed leading span. Two FNV-1a streams with distinct seeds folded into
    /// a u128 (collision-resistant enough for a re-verified routing hint).
    /// `prefix_len` is folded in so the same tokens at a different banked
    /// length address a different entry.
    pub fn hash_prefix(tokens: &[u32], prefix_len: usize) -> u128 {
        let n = prefix_len.min(tokens.len());
        let mut a: u64 = 0xcbf29ce484222325; // FNV offset basis
        let mut b: u64 = 0x100000001b3 ^ 0x9e3779b97f4a7c15; // distinct seed
        let mix = |h: &mut u64, x: u64, prime: u64| {
            *h ^= x;
            *h = h.wrapping_mul(prime);
        };
        mix(&mut a, n as u64, 0x100000001b3);
        mix(&mut b, (n as u64).rotate_left(32), 0x9e3779b97f4a7c15);
        for &t in &tokens[..n] {
            mix(&mut a, t as u64, 0x100000001b3);
            mix(
                &mut b,
                (t as u64).wrapping_add(0x632be59bd9b4e019),
                0x9e3779b97f4a7c15,
            );
        }
        ((a as u128) << 64) | (b as u128)
    }

    /// Look up a banked source-slot for the leading span of `tokens`.
    ///
    /// `banked_len` is the prefix length to probe — typically the caller's
    /// notion of where the fixed system span ends (e.g. min(tokens.len()-1,
    /// a configured system-span length), always at least one token short of
    /// the full prompt so the decode loop keeps a real last_id, mirroring the
    /// disk/RAM tiers' "bail one token short" rule). Returns the entry on a
    /// hit (and LRU-touches it + bumps hit counters); `None` on miss/too-short.
    pub fn lookup(&mut self, tokens: &[u32], banked_len: usize) -> Option<BankEntry> {
        self.stats.lookups += 1;
        if banked_len < self.cfg.min_prefix_tokens || banked_len >= tokens.len().max(1) {
            // Strict prefix only; never match the whole prompt.
            return None;
        }
        let key = Self::hash_prefix(tokens, banked_len);
        let entry = self.entries.get_mut(&key)?;
        if entry.prefix_len != banked_len {
            // Length collision guard (cannot reuse a different-length block).
            return None;
        }
        self.clock += 1;
        entry.last_tick = self.clock;
        entry.hits += 1;
        self.stats.hits += 1;
        Some(*entry)
    }

    /// Bank (or refresh) that `source_slot` holds copyable KV for the leading
    /// `prefix_len` tokens of `tokens`. Runs LRU eviction to `max_entries`.
    pub fn record(&mut self, tokens: &[u32], prefix_len: usize, source_slot: u32) -> RecordOutcome {
        if prefix_len < self.cfg.min_prefix_tokens || prefix_len > tokens.len() {
            return RecordOutcome::TooShort;
        }
        self.clock += 1;
        let tick = self.clock;
        let key = Self::hash_prefix(tokens, prefix_len);
        let outcome = match self.entries.get_mut(&key) {
            Some(e) => {
                e.prefix_len = prefix_len;
                e.source_slot = source_slot;
                e.last_tick = tick;
                RecordOutcome::Updated
            }
            None => {
                self.entries.insert(
                    key,
                    BankEntry {
                        prefix_len,
                        source_slot,
                        last_tick: tick,
                        hits: 0,
                    },
                );
                RecordOutcome::Inserted
            }
        };
        self.stats.records += 1;
        self.evict_to_cap();
        self.stats.entries = self.entries.len();
        outcome
    }

    /// Drop the banked mapping for a slot that is being torn down with a
    /// changed prefix, or invalidate a known-stale source. Returns how many
    /// entries were removed (entries whose `source_slot == slot`).
    pub fn forget_slot(&mut self, slot: u32) -> usize {
        let before = self.entries.len();
        self.entries.retain(|_, e| e.source_slot != slot);
        let removed = before - self.entries.len();
        self.stats.entries = self.entries.len();
        removed
    }

    /// LRU-evict (smallest `last_tick` first) until within `max_entries`.
    pub fn evict_to_cap(&mut self) {
        while self.entries.len() > self.cfg.max_entries {
            let victim = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_tick)
                .map(|(k, _)| *k);
            match victim {
                Some(k) => {
                    self.entries.remove(&k);
                    self.stats.evictions += 1;
                }
                None => break,
            }
        }
        self.stats.entries = self.entries.len();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn stats(&self) -> BankStats {
        let mut s = self.stats;
        s.entries = self.entries.len();
        s
    }
}

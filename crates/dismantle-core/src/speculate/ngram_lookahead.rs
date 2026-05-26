//! Lookahead-decoding n-gram hashmap (Cai et al., 2024 style).
//!
//! Unlike `speculate::ngram::NGramDraft` (which scans history linearly
//! for the most-recent occurrence of the last N-1 tokens), this module
//! maintains an O(1) hashmap keyed by every (N-1)-token suffix seen,
//! mapping to a small frequency-counted distribution over next tokens.
//! At decode time, the proposer chains lookups together to build a
//! K-token draft sequence at zero compute cost: take the top-1 next
//! token for the current suffix, slide the suffix window forward, and
//! repeat K times.
//!
//! ## Why a hashmap instead of a scan
//!
//! The scan-based proposer in `ngram.rs` finds at most one prior match
//! per call and proposes the same continuation regardless of how often
//! it has occurred. The hashmap variant proposes the most-frequent
//! continuation across all prior occurrences (better recall on
//! repetitive / structured / code prompts where the same context
//! recurs many times with the same continuation), and is also much
//! cheaper for long histories (constant-time lookup vs O(history) per
//! propose).
//!
//! ## Wire-up
//!
//! - Seed: call `observe` for every prompt token before decode starts.
//! - Each accepted output token: call `observe(token)` to extend the
//!   hashmap.
//! - Each decode step: call `propose(k)` to get a draft of up to K
//!   tokens, then verify against the model. Drafts may be shorter than
//!   K if the hashmap runs out of known next tokens partway through.
//!
//! ## Tunables (via `LookaheadConfig`)
//!
//! - `n`: n-gram order. The key is the last (n-1) tokens; n=4 means
//!   trigram keys. Practical range 3..=5 (larger keys have lower hit
//!   rate; smaller keys have lower precision).
//! - `max_branches_per_key`: distribution width per key. We keep up to
//!   this many distinct next-token observations per key, sorted by
//!   frequency. 1 is greedy (top-1 only); larger values cost memory
//!   but currently aren't used by the proposer (kept for future
//!   tree-of-drafts work).
//! - `cap`: total observation cap, evicting oldest writes when
//!   exceeded. Bounded memory regardless of generation length.

use std::collections::HashMap;
use std::collections::VecDeque;

/// Configuration for [`LookaheadCache`].
#[derive(Debug, Clone, Copy)]
pub struct LookaheadConfig {
    /// N-gram order. Key is the last `n - 1` tokens. Must be `>= 2`.
    pub n: usize,
    /// Max distinct next-token entries kept per key (sorted by
    /// frequency). 1 = greedy, 4 = small fan-out for future tree
    /// drafts.
    pub max_branches_per_key: usize,
    /// Total observation cap. Oldest writes are evicted past this.
    pub cap: usize,
}

impl Default for LookaheadConfig {
    fn default() -> Self {
        Self {
            n: 4,
            max_branches_per_key: 4,
            cap: 16_384,
        }
    }
}

/// Token-id type. Matches `u32` IDs used across the engine.
pub type TokenId = u32;

/// A key into the n-gram map: the last `n - 1` tokens.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct Key(Vec<TokenId>);

/// Frequency-counted next-token distribution for a single key.
#[derive(Debug, Clone, Default)]
struct Entry {
    /// (token, count), sorted descending by count.
    branches: Vec<(TokenId, u32)>,
}

impl Entry {
    fn observe(&mut self, tok: TokenId, max_branches: usize) {
        // Linear scan -- max_branches is tiny in practice (1..=8).
        if let Some(slot) = self.branches.iter_mut().find(|(t, _)| *t == tok) {
            slot.1 = slot.1.saturating_add(1);
        } else {
            self.branches.push((tok, 1));
            if self.branches.len() > max_branches {
                // Drop the least-frequent entry. The vector stays
                // small enough that a full sort each time is fine.
                self.branches.sort_by(|a, b| b.1.cmp(&a.1));
                self.branches.truncate(max_branches);
                return;
            }
        }
        // Keep sorted-by-count-desc invariant.
        self.branches.sort_by(|a, b| b.1.cmp(&a.1));
    }

    fn top(&self) -> Option<TokenId> {
        self.branches.first().map(|(t, _)| *t)
    }
}

/// Lookahead n-gram cache. Maintains a hashmap keyed by `(n - 1)`-token
/// suffixes; provides constant-time draft proposal via chained top-1
/// lookups. Bounded memory via eviction queue.
pub struct LookaheadCache {
    cfg: LookaheadConfig,
    /// Most recent tokens. We need the last `n` to (a) form the next
    /// key on observation and (b) form the seed key when proposing.
    history: VecDeque<TokenId>,
    /// (key, next_token) write log, in observation order. When the
    /// log size exceeds `cfg.cap`, we drop the oldest entry from the
    /// log and decrement (or remove) the matching entry in `map`.
    write_log: VecDeque<(Key, TokenId)>,
    map: HashMap<Key, Entry>,
    /// Total accepted-draft and rejected-draft counters for the host
    /// to fold into engine stats.
    accepted_drafts: u64,
    rejected_drafts: u64,
    /// Total drafts proposed (regardless of acceptance). Useful for
    /// hit-rate reporting.
    proposals: u64,
    /// Sum of draft lengths over all `propose` calls. Combined with
    /// `proposals` gives mean draft length.
    proposed_total_len: u64,
}

impl LookaheadCache {
    pub fn new(cfg: LookaheadConfig) -> Self {
        let n = cfg.n.max(2);
        let cfg = LookaheadConfig { n, ..cfg };
        Self {
            cfg,
            history: VecDeque::with_capacity(64),
            write_log: VecDeque::with_capacity(cfg.cap),
            map: HashMap::with_capacity(cfg.cap),
            accepted_drafts: 0,
            rejected_drafts: 0,
            proposals: 0,
            proposed_total_len: 0,
        }
    }

    /// Record `tok` as the next observed token, updating the n-gram
    /// map for the (n-1)-token suffix immediately before `tok`. Safe
    /// to call before there are enough tokens to form a key -- the
    /// first `n-1` calls are pure history-bookkeeping.
    pub fn observe(&mut self, tok: TokenId) {
        let key_len = self.cfg.n - 1;
        if self.history.len() >= key_len {
            // The key is the last `key_len` tokens BEFORE we push tok.
            let start = self.history.len() - key_len;
            let key: Vec<TokenId> = self.history.iter().skip(start).copied().collect();
            let k = Key(key);
            self.write_log.push_back((k.clone(), tok));
            self.map
                .entry(k)
                .or_default()
                .observe(tok, self.cfg.max_branches_per_key);
            // Evict oldest if over cap.
            if self.write_log.len() > self.cfg.cap {
                if let Some((old_k, old_t)) = self.write_log.pop_front() {
                    if let Some(e) = self.map.get_mut(&old_k) {
                        // Decrement (or remove) the old observation.
                        if let Some(pos) = e.branches.iter().position(|(t, _)| *t == old_t) {
                            let c = &mut e.branches[pos].1;
                            *c = c.saturating_sub(1);
                            if *c == 0 {
                                e.branches.swap_remove(pos);
                                e.branches.sort_by(|a, b| b.1.cmp(&a.1));
                            }
                        }
                        if e.branches.is_empty() {
                            self.map.remove(&old_k);
                        }
                    }
                }
            }
        }
        self.history.push_back(tok);
        // History only needs to be `n` long to form keys; keep a bit
        // extra for safety but avoid unbounded growth.
        let max_history = (self.cfg.n * 4).max(16);
        while self.history.len() > max_history {
            self.history.pop_front();
        }
    }

    /// Propose up to `k` draft tokens by chained top-1 lookups. Walks
    /// the n-gram graph: starting from the current `(n-1)` suffix,
    /// pick the most-frequent next token, slide the window forward,
    /// repeat until either `k` tokens are produced or no key is found.
    ///
    /// Returns an empty vec if there is no seed key (history shorter
    /// than `n-1`) or no observed continuation for the seed.
    pub fn propose(&mut self, k: usize) -> Vec<TokenId> {
        if k == 0 {
            return vec![];
        }
        let key_len = self.cfg.n - 1;
        if self.history.len() < key_len {
            return vec![];
        }
        // Seed key = last `key_len` tokens of history.
        let mut window: Vec<TokenId> = self
            .history
            .iter()
            .skip(self.history.len() - key_len)
            .copied()
            .collect();
        let mut out = Vec::with_capacity(k);
        for _ in 0..k {
            let key = Key(window.clone());
            let next = match self.map.get(&key).and_then(|e| e.top()) {
                Some(t) => t,
                None => break,
            };
            out.push(next);
            // Slide window forward.
            window.remove(0);
            window.push(next);
        }
        self.proposals = self.proposals.saturating_add(1);
        self.proposed_total_len = self.proposed_total_len.saturating_add(out.len() as u64);
        out
    }

    /// Update acceptance stats. Call after each verify step with the
    /// number of drafts accepted and the total drafted.
    pub fn record_outcome(&mut self, accepted: usize, drafted: usize) {
        let rejected = drafted.saturating_sub(accepted);
        self.accepted_drafts = self.accepted_drafts.saturating_add(accepted as u64);
        self.rejected_drafts = self.rejected_drafts.saturating_add(rejected as u64);
    }

    /// Snapshot stats: (accepted, rejected, proposals, mean draft len).
    pub fn stats(&self) -> (u64, u64, u64, f64) {
        let mean_len = if self.proposals == 0 {
            0.0
        } else {
            self.proposed_total_len as f64 / self.proposals as f64
        };
        (
            self.accepted_drafts,
            self.rejected_drafts,
            self.proposals,
            mean_len,
        )
    }

    /// Acceptance rate = accepted / (accepted + rejected). 0 when no
    /// drafts have been verified yet.
    pub fn acceptance_rate(&self) -> f64 {
        let denom = self.accepted_drafts + self.rejected_drafts;
        if denom == 0 {
            0.0
        } else {
            self.accepted_drafts as f64 / denom as f64
        }
    }

    pub fn config(&self) -> LookaheadConfig {
        self.cfg
    }

    /// Number of distinct keys in the map. For diagnostics.
    pub fn distinct_keys(&self) -> usize {
        self.map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propose_empty_history() {
        let mut c = LookaheadCache::new(LookaheadConfig {
            n: 3,
            ..Default::default()
        });
        assert!(c.propose(4).is_empty());
    }

    #[test]
    fn propose_walks_top1_chain() {
        // n=3: keys are bigrams. Seed: 1,2,3,4,5 then repeat 4,5 to
        // make (4,5)->? deterministic.
        let mut c = LookaheadCache::new(LookaheadConfig {
            n: 3,
            max_branches_per_key: 4,
            cap: 1024,
        });
        // Build deterministic chain: 1 2 3 4 5 ; (1,2)->3 (2,3)->4
        // (3,4)->5. Then seed (4,5) by observing one more token (e.g.
        // 6) so the seed key after observation is (5,6) — instead, we
        // want to *query* with the trailing (4,5). Easiest: observe a
        // tail that ends in 4,5, then continue with 6,7. So:
        for &t in &[1u32, 2, 3, 4, 5, 6, 7] {
            c.observe(t);
        }
        // Now observe another loop that gives (4,5)->6 and (5,6)->7.
        for &t in &[4u32, 5, 6, 7, 4, 5] {
            c.observe(t);
        }
        // Current history ends in [...,4,5]. propose(3) should give
        // [6, 7, ?] -- (4,5)->6, then (5,6)->7, then (6,7)->? -- (6,7)
        // was followed by 4 in the second loop, so [6, 7, 4].
        let p = c.propose(3);
        assert_eq!(p, vec![6, 7, 4]);
    }

    #[test]
    fn propose_stops_at_unknown_key() {
        let mut c = LookaheadCache::new(LookaheadConfig {
            n: 3,
            ..Default::default()
        });
        for &t in &[10u32, 20, 30] {
            c.observe(t);
        }
        // History (20, 30) has no successor recorded for (20, 30).
        assert!(c.propose(5).is_empty());
    }

    #[test]
    fn observe_updates_top1_to_most_frequent() {
        let mut c = LookaheadCache::new(LookaheadConfig {
            n: 3,
            max_branches_per_key: 4,
            cap: 1024,
        });
        // (1, 2) -> 9 twice, -> 7 once. Top should be 9.
        for &t in &[1u32, 2, 9, 1, 2, 9, 1, 2, 7, 1, 2] {
            c.observe(t);
        }
        // History tail = [..., 1, 2].
        let p = c.propose(1);
        assert_eq!(p, vec![9]);
    }

    #[test]
    fn record_outcome_tracks_acceptance() {
        let mut c = LookaheadCache::new(LookaheadConfig::default());
        c.record_outcome(2, 4);
        c.record_outcome(1, 4);
        let (acc, rej, _, _) = c.stats();
        assert_eq!(acc, 3);
        assert_eq!(rej, 5);
        assert!((c.acceptance_rate() - 3.0 / 8.0).abs() < 1e-9);
    }

    #[test]
    fn eviction_keeps_map_bounded() {
        let mut c = LookaheadCache::new(LookaheadConfig {
            n: 2,
            max_branches_per_key: 1,
            cap: 8,
        });
        // n=2 means singleton keys. Push 100 distinct tokens; cap is
        // 8 so we should never hold more than 8 keys.
        for t in 0u32..100 {
            c.observe(t);
        }
        assert!(c.distinct_keys() <= 8, "map size {} exceeded cap 8", c.distinct_keys());
    }
}

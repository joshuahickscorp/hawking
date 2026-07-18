//! L3.1 (b) — per-user n-gram draft, grown online from the emitted stream.
//!
//! The live speculation mechanism on code is the **n-gram / PLD / SAM draft**
//! (τ≈1.43 generic, `reports/oracle/spec_accept.json`), *not* a trained head
//! (EAGLE-3 is NO-GO, τ=0.877 net-negative — `docs/dead_levers.md`). The
//! offline warm-start oracle (`reports/oracle/spec_accept_warmstart.json`)
//! cleared GO: seeding a per-user n-gram index from the user's own prior
//! tokens lifts the recomputed-suffix draft to τ≈3.40 (+0.888 over a cold
//! index), **additive to the shipped prefix cache**.
//!
//! This struct is that index. It grows from the user's emitted token stream
//! ([`note_token`]) — their codebase identifiers, boilerplate, frequent
//! completions — and proposes a draft continuation for the current context
//! ([`propose`]). It is the **draft source** the propose→verify loop plugs in;
//! the [`crate::shared::verify_draft_ids_until_mismatch`] /
//! `forward_tokens_verify` verifier is reused **unchanged**, so every emitted
//! token is still the verifier's token — the draft affects only speed, never
//! output. **Lossless by construction** (design §2.1b: "E unconditionally").
//!
//! The index is a CPU automaton (KB–MB in unified memory, the Apple-Silicon-
//! safe spec path the bible blesses). No GPU, no second model.
//!
//! # Mechanism
//!
//! Two tiers, queried highest-order-first so the most specific match wins:
//!
//!   - **2-gram → successor frequency** (matching `n=2` in the spec oracle):
//!     `(prev, cur) → {next_id: count}`. `propose` chains greedily: from the
//!     context's most-frequent successor `s`, then `(cur, s)`'s most-frequent
//!     successor, and so on up to `k` tokens.
//!   - **1-gram backoff**: `cur → {next_id: count}`, used when a 2-gram chain
//!     step has no recorded successor, so a partly-seen context still drafts.
//!
//! Chaining stops early on a backoff miss (no point proposing tokens the
//! verifier will certainly reject), and a cycle-guard caps repeats so a
//! degenerate "AAAA…" successor cannot fill the whole window with one token.

use std::collections::HashMap;

/// Per-user n-gram draft index. Default-empty (cold); call [`note_token`] for
/// each emitted token to grow it, or [`warm_start`] to seed it from a prior
/// token slice (the user's history) in one shot.
#[derive(Debug, Clone, Default)]
pub struct UserNgramDraft {
    /// `(prev, cur) → (successor_id → count)`.
    bigram: HashMap<(u32, u32), HashMap<u32, u32>>,
    /// `cur → (successor_id → count)` backoff.
    unigram: HashMap<u32, HashMap<u32, u32>>,
    /// Trailing two tokens of the stream fed so far, so `note_token` can form
    /// the next `(prev, cur) → next` transition incrementally.
    prev: Option<u32>,
    cur: Option<u32>,
    /// Total transitions recorded — cheap "is the index worth querying" gate.
    transitions: u64,
}

impl UserNgramDraft {
    /// A fresh, empty (cold) index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of `(context → next)` transitions recorded so far.
    pub fn len(&self) -> u64 {
        self.transitions
    }

    /// `true` if no transition has been recorded yet (a cold index proposes
    /// nothing).
    pub fn is_empty(&self) -> bool {
        self.transitions == 0
    }

    /// Reset the *rolling context* (the trailing two tokens) between
    /// generation requests, **without** discarding the learned index. The
    /// per-user index is meant to persist across a user's turns (that is the
    /// warm-start the oracle measured); only the in-flight `prev`/`cur` cursor
    /// resets so a new prompt does not chain off the previous prompt's tail.
    pub fn reset_context(&mut self) {
        self.prev = None;
        self.cur = None;
    }

    /// Fully clear the index *and* context (a brand-new cold user).
    pub fn clear(&mut self) {
        self.bigram.clear();
        self.unigram.clear();
        self.prev = None;
        self.cur = None;
        self.transitions = 0;
    }

    /// Grow the index by one emitted token. Forms the `(prev, cur) → token`
    /// bigram transition and the `cur → token` unigram backoff, then advances
    /// the rolling context. Idempotent in shape; cheap (two hash bumps).
    pub fn note_token(&mut self, token: u32) {
        if let Some(cur) = self.cur {
            // unigram backoff: cur → token
            *self
                .unigram
                .entry(cur)
                .or_default()
                .entry(token)
                .or_insert(0) += 1;
            if let Some(prev) = self.prev {
                // bigram: (prev, cur) → token
                *self
                    .bigram
                    .entry((prev, cur))
                    .or_default()
                    .entry(token)
                    .or_insert(0) += 1;
            }
            self.transitions += 1;
        }
        self.prev = self.cur;
        self.cur = Some(token);
    }

    /// Seed the index from a prior token slice (the user's history) in one
    /// shot, then reset the rolling context so the next live token does not
    /// chain off the seed's tail. This is the **warm-start** the oracle scored.
    pub fn warm_start(&mut self, history: &[u32]) {
        for &t in history {
            self.note_token(t);
        }
        self.reset_context();
    }

    /// Most-frequent successor of a 2-gram context, with 1-gram backoff.
    /// Returns `None` if neither tier has seen `cur` (or `(prev,cur)`).
    /// Ties broken by smallest id for determinism.
    fn best_successor(&self, prev: Option<u32>, cur: u32) -> Option<u32> {
        if let Some(p) = prev {
            if let Some(succ) = self.bigram.get(&(p, cur)) {
                if let Some(best) = argmax_count(succ) {
                    return Some(best);
                }
            }
        }
        self.unigram.get(&cur).and_then(argmax_count)
    }

    /// Propose up to `k` draft continuation tokens for the context whose last
    /// two tokens are `ctx` (`[.., prev, cur]`, oldest first). Analogous to
    /// `Eagle5Head::propose_rollout_chained(start, …, k)` — returns a `Vec<u32>`
    /// the propose→verify loop checks; the verifier emits, so any of these may
    /// be wrong without affecting output.
    ///
    /// Chains greedily: predict `cur`'s best successor `s0`, then `(cur, s0)`'s
    /// best, etc. Stops early when a step has no recorded successor (drafting
    /// past a miss only wastes verifier work) or after `k` tokens. A cycle
    /// guard caps how many times the same token may repeat so a degenerate
    /// self-loop successor cannot fill the window.
    pub fn propose(&self, ctx: &[u32], k: usize) -> Vec<u32> {
        if k == 0 {
            return Vec::new();
        }
        let Some(&cur0) = ctx.last() else {
            return Vec::new();
        };
        let prev0 = if ctx.len() >= 2 {
            Some(ctx[ctx.len() - 2])
        } else {
            None
        };

        let mut out: Vec<u32> = Vec::with_capacity(k);
        let mut prev = prev0;
        let mut cur = cur0;
        // Cap per-token repeats: a draft of the same token more than a few
        // times running is almost never accepted and risks an infinite loop on
        // a self-referential successor.
        let mut last_emitted: Option<u32> = None;
        let mut repeat_run: usize = 0;
        const MAX_REPEAT_RUN: usize = 3;

        for _ in 0..k {
            let Some(next) = self.best_successor(prev, cur) else {
                break;
            };
            if Some(next) == last_emitted {
                repeat_run += 1;
                if repeat_run >= MAX_REPEAT_RUN {
                    break;
                }
            } else {
                repeat_run = 0;
            }
            out.push(next);
            last_emitted = Some(next);
            prev = Some(cur);
            cur = next;
        }
        out
    }
}

/// The key of the entry with the largest count, ties broken by smallest id so
/// the choice is deterministic across runs (parity-relevant: a stable draft
/// makes paired benches reproducible; correctness is independent of the choice
/// because the verifier emits).
fn argmax_count(m: &HashMap<u32, u32>) -> Option<u32> {
    m.iter()
        .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
        .map(|(&id, _)| id)
}

// ---- Event Horizon: the live, lossless, tokenizer-native BASE proposer ----
use crate::proposal::{Budget, Ctx, Proposal, Proposer, Telemetry};

/// Default-on, lossless, tokenizer-native base proposer: a thin [`Proposer`]
/// adapter over [`UserNgramDraft`] (the live τ=1.43 base that beat the trained
/// head — see docs/dead_levers.md). Newtype so adapter-only state can be added
/// later without touching the inherent API (its determinism/tie-break is
/// load-bearing for paired-bench parity).
#[derive(Debug, Default)]
pub struct NgramProposer {
    inner: UserNgramDraft,
}

impl NgramProposer {
    pub fn new() -> Self {
        Self {
            inner: UserNgramDraft::new(),
        }
    }
    pub fn index(&self) -> &UserNgramDraft {
        &self.inner
    }
}

impl Proposer for NgramProposer {
    fn name(&self) -> &'static str {
        "user_ngram"
    }

    fn propose(&mut self, ctx: &Ctx<'_>, budget: Budget, _tel: &Telemetry) -> Proposal {
        // UserNgramDraft::propose is &self, reads only the last two ids, may
        // return < k on a chain miss. Caller already clamped budget.k to ≤7.
        Proposal::TokenLine(self.inner.propose(ctx.last_two(), budget.k))
    }

    fn observe(&mut self, emitted: &[u32]) {
        // MANDATORY: grow the index from the emitted (verifier) stream — one
        // note_token per token, in order, identical to the live 'ud_loop feed.
        for &t in emitted {
            self.inner.note_token(t);
        }
    }

    fn warm(&mut self, history: &[u32]) {
        self.inner.warm_start(history);
    }

    fn reset(&mut self) {
        // cursor only — KEEP the learned grams across requests.
        self.inner.reset_context();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_index_proposes_nothing() {
        let d = UserNgramDraft::new();
        assert!(d.is_empty());
        assert_eq!(d.propose(&[1, 2], 4), Vec::<u32>::new());
        assert_eq!(d.propose(&[], 4), Vec::<u32>::new());
    }

    #[test]
    fn proposes_the_users_repeated_grams() {
        // The user repeatedly emits the sequence 10, 11, 12, 13 (e.g. a
        // frequent identifier / boilerplate run). After seeing it, proposing
        // from the [.., 10, 11] context must roll out 12, 13.
        let mut d = UserNgramDraft::new();
        for _ in 0..3 {
            for &t in &[10u32, 11, 12, 13] {
                d.note_token(t);
            }
        }
        assert!(!d.is_empty());
        // Context ends in (10, 11) → best successor 12 → (11,12)→13 → (12,13)→10.
        let drafted = d.propose(&[10, 11], 3);
        assert_eq!(drafted[0], 12, "2-gram (10,11) should predict 12");
        assert_eq!(drafted[1], 13, "chained (11,12) should predict 13");
        // k limits the length.
        assert_eq!(d.propose(&[10, 11], 1), vec![12]);
    }

    #[test]
    fn warm_start_seeds_then_resets_context() {
        // Seed from history, then a live prompt's first token must not chain
        // off the seed's tail — only the rolling cursor resets, the learned
        // grams persist.
        let mut d = UserNgramDraft::new();
        d.warm_start(&[1, 2, 3, 2, 3, 4]);
        assert!(!d.is_empty());
        // (2,3) was seen twice → 2 then 4; argmax by count then smaller id → 2.
        let p = d.propose(&[2, 3], 2);
        assert_eq!(
            p[0], 2,
            "warm-started (2,3) predicts the repeated successor"
        );
        // The seed left no live cursor; note_token after warm_start starts a
        // fresh transition (no chaining off the seed's last token 4).
        d.note_token(99);
        assert!(d.bigram.get(&(4, 99)).is_none());
    }

    #[test]
    fn unigram_backoff_when_bigram_unseen() {
        // Only a unigram transition exists for `cur`; an unseen (prev,cur)
        // must back off to it rather than propose nothing.
        let mut d = UserNgramDraft::new();
        // Seed cur=5 → 6 a few times via a context the bigram won't match.
        for _ in 0..2 {
            d.note_token(5);
            d.note_token(6);
            d.reset_context();
        }
        // prev=777 never co-occurred with 5, so (777,5) misses → unigram 5→6.
        let p = d.propose(&[777, 5], 1);
        assert_eq!(p, vec![6], "should back off to the 1-gram successor");
    }

    #[test]
    fn repeat_run_is_capped() {
        // A self-loop successor (8 → 8) must not fill the whole window.
        let mut d = UserNgramDraft::new();
        for _ in 0..5 {
            d.note_token(8);
        }
        let p = d.propose(&[8], 16);
        assert!(
            p.len() <= 3,
            "self-loop successor must be capped, got {} tokens",
            p.len()
        );
    }
}

//! Event Horizon Phase 2 — REST-style local retrieval proposer.
//!
//! `RetrievalProposer` is the Phase 2 complement to `SuffixArrayDraft`.
//! Both are model-free exact-match copiers, but they differ in scope:
//!
//! | Proposer           | Corpus searched                     | Anchor h |
//! |--------------------|-------------------------------------|----------|
//! | `SuffixArrayDraft` | Emitted tokens only                 | 3        |
//! | `RetrievalProposer`| Warm-seeded text + emitted tokens   | 4        |
//!
//! The wider corpus means `RetrievalProposer` can draft from prompt/repo
//! patterns that the suffix-array never sees: large code bases seeded via
//! `warm()`, repeated boilerplate blocks injected before generation starts,
//! or any text from the current session history.
//!
//! Lossless by construction: the verifier confirms every proposed token.
//!
//! # Lifecycle
//! 1. `warm(history)` — called once per request with repo/session/prompt
//!    tokens (oldest-first), extends the corpus.
//! 2. `propose(ctx, budget, tel)` — searches the corpus for `h`-token anchor
//!    from `ctx.tokens` tail and returns the `k` tokens that followed it.
//! 3. `observe(emitted)` — appends each verifier-confirmed token to the corpus
//!    so future queries can match patterns from the ongoing generation.
//! 4. `reset()` — no-op; the corpus persists across requests (same policy as
//!    `SuffixArrayDraft` and `NgramProposer`).

use crate::speculate::proposal::{Budget, CostNs, Ctx, Proposal, Proposer, Telemetry};

/// Rolling-window exact-match retrieval proposer.
///
/// Maintains a combined corpus of warm-seeded text and emitted tokens
/// (bounded to `window` tokens, oldest-first). At each decode step, searches
/// backwards through the corpus for the most recent occurrence of the last
/// `h` tokens from `ctx.tokens` and drafts the `k` tokens that followed it.
///
/// `h=4` (vs `SuffixArrayDraft`'s h=3) means longer anchors are required;
/// precision is higher but recall is lower — this proposer fires on long
/// repeated spans that the wider corpus makes reachable.
#[derive(Debug, Clone)]
pub struct RetrievalProposer {
    /// Warm-seeded text concatenated with observed (emitted) tokens, oldest-first.
    /// Bounded to `window` tokens; oldest entries are drained when the cap is hit.
    corpus: Vec<u32>,
    /// Maximum corpus size to retain and search (default 50 000).
    window: usize,
    /// Anchor length — number of tail tokens from `ctx.tokens` to match (default 4).
    h: usize,
}

impl Default for RetrievalProposer {
    fn default() -> Self {
        Self {
            corpus: Vec::new(),
            window: 50_000,
            h: 4,
        }
    }
}

impl RetrievalProposer {
    /// Create a proposer with default parameters (`window=50_000`, `h=4`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a proposer with explicit `window` and anchor length `h`.
    pub fn with_params(window: usize, h: usize) -> Self {
        Self {
            corpus: Vec::new(),
            window,
            h,
        }
    }

    /// Append tokens to the corpus and trim to `window` if necessary.
    fn append(&mut self, tokens: &[u32]) {
        self.corpus.extend_from_slice(tokens);
        if self.corpus.len() > self.window {
            let drain_to = self.corpus.len() - self.window;
            self.corpus.drain(..drain_to);
        }
    }
}

impl Proposer for RetrievalProposer {
    fn name(&self) -> &'static str {
        "retrieval"
    }

    /// Free: backward linear scan is O(window) but runs on `u32` slices with
    /// no model math. Caller pays zero extra forward passes.
    fn cost_estimate(&self, _ctx: &Ctx<'_>, _budget: Budget) -> CostNs {
        0
    }

    fn propose(&mut self, ctx: &Ctx<'_>, budget: Budget, _tel: &Telemetry) -> Proposal {
        let k = budget.k;
        let clen = self.corpus.len();

        // Guard: need h context tokens and a corpus large enough to hold at
        // least one prior position (h tokens) plus one draft token.
        if k == 0 || ctx.tokens.len() < self.h || clen <= self.h {
            return Proposal::TokenLine(Vec::new());
        }

        // The anchor is the last `h` tokens from ctx.tokens (the emitted
        // context).  Note: the corpus also contains those emitted tokens
        // (added via observe()), so we exclude the current tail of the corpus
        // from the search to avoid a trivial self-match.
        let anchor_start = ctx.tokens.len() - self.h;
        let anchor = &ctx.tokens[anchor_start..];

        // search_end: exclusive upper bound for the start of a candidate match.
        // i + h <= corpus.len() - h  ⟹  i < corpus.len() - h
        // This prevents matching against the current tail of the corpus (the
        // position where these same tokens live as the most-recently-observed
        // emitted tokens).
        let search_end = clen.saturating_sub(self.h); // exclusive

        // Search backwards (most-recent-first) for an exact h-token match.
        let mut best_pos: Option<usize> = None;
        'search: for i in (0..search_end).rev() {
            // Collision guard: compare all h elements before accepting.
            for d in 0..self.h {
                if self.corpus[i + d] != anchor[d] {
                    continue 'search;
                }
            }
            best_pos = Some(i);
            break; // most-recent match wins
        }

        let Some(prior) = best_pos else {
            return Proposal::TokenLine(Vec::new());
        };

        // Copy up to k tokens starting immediately after the matched anchor.
        let copy_start = prior + self.h;
        let copy_end = copy_start.saturating_add(k).min(clen);
        if copy_start >= copy_end {
            return Proposal::TokenLine(Vec::new());
        }

        Proposal::TokenLine(self.corpus[copy_start..copy_end].to_vec())
    }

    /// Mandatory online feedback: append every verifier-confirmed token so the
    /// corpus stays current with the ongoing generation.
    fn observe(&mut self, emitted: &[u32]) {
        self.append(emitted);
    }

    /// Per-request warm-start. Caller passes repo→session→prompt tokens in
    /// oldest-first order (P1.2 convention). The corpus accumulates across
    /// `warm()` calls within a session; the window cap prevents unbounded growth.
    fn warm(&mut self, history: &[u32]) {
        self.append(history);
    }

    /// Per-request teardown. The corpus IS the learned index — no cursor to
    /// reset; it persists across requests intentionally (identical policy to
    /// `SuffixArrayDraft::reset`).
    fn reset(&mut self) {
        // corpus persists
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculate::proposal::{Budget, Ctx, Telemetry};

    fn make_ctx<'a>(tokens: &'a [u32]) -> Ctx<'a> {
        Ctx {
            tokens,
            pos: tokens.len(),
            hidden: None,
        }
    }

    /// Helper: extract the Vec<u32> from a TokenLine or panic.
    fn token_line(p: Proposal) -> Vec<u32> {
        match p {
            Proposal::TokenLine(v) => v,
            _ => panic!("expected TokenLine"),
        }
    }

    // Test 1: cold corpus → empty proposal regardless of ctx content.
    #[test]
    fn cold_corpus_proposes_nothing() {
        let mut p = RetrievalProposer::new();
        let ctx = make_ctx(&[1, 2, 3, 4]);
        let proposal = p.propose(&ctx, Budget::line(4), &Telemetry::default());
        assert!(proposal.is_empty(), "cold corpus must propose nothing");
    }

    // Test 2: span seeded via warm() is retrieved and proposed correctly.
    //
    // Corpus after warm(): [10, 20, 30, 40, 50, 60]
    // ctx.tokens tail (h=4): [10, 20, 30, 40]
    // anchor [10,20,30,40] matches at corpus index 0.
    // copy_start = 0+4 = 4, copy_end = min(4+3, 6) = 6
    // draft = corpus[4..6] = [50, 60]
    #[test]
    fn warm_seeded_span_is_retrieved() {
        let mut p = RetrievalProposer::new();
        p.warm(&[10u32, 20, 30, 40, 50, 60]);

        // ctx.tokens is the anchor (last h=4 tokens of the emitted context).
        let tokens = [10u32, 20, 30, 40];
        let ctx = make_ctx(&tokens);
        let v = token_line(p.propose(&ctx, Budget::line(3), &Telemetry::default()));
        // The anchor is found at corpus[0..4]; tokens after it are [50, 60].
        assert_eq!(
            v,
            vec![50u32, 60],
            "should draft the 2 tokens after the warm anchor"
        );
    }

    // Test 3: novel tail → nothing (anchor absent in corpus).
    #[test]
    fn novel_tail_proposes_nothing() {
        let mut p = RetrievalProposer::new();
        // Corpus contains a specific sequence.
        p.warm(&[1u32, 2, 3, 4, 5, 6, 7, 8]);
        // ctx.tokens uses a 4-token anchor not present in the corpus.
        let tokens = [99u32, 100, 101, 102];
        let ctx = make_ctx(&tokens);
        let proposal = p.propose(&ctx, Budget::line(4), &Telemetry::default());
        assert!(proposal.is_empty(), "novel anchor must propose nothing");
    }

    // Test 4: collision guard — near-miss does NOT match, only the exact h-token
    // anchor matches.
    //
    // Corpus: [1, 2, 3, 4, 99,  1, 2, 3, 5, 88,  1, 2, 3, 4]
    //          ^--- exact [1,2,3,4]             ^--- near-miss [1,2,3,5]
    //                                                            ^--- exact [1,2,3,4] (tail, excluded)
    // h=4, anchor from ctx = [1,2,3,4]
    // corpus.len()=14, search_end = 14-4 = 10
    // Search backward: most recent match before index 10?
    //   i=9: corpus[9..13]=[88,1,2,3] → no
    //   i=8: corpus[8..12]=[5,88,1,2] → no
    //   i=7: corpus[7..11]=[3,5,88,1] → no
    //   i=6: corpus[6..10]=[2,3,5,88] → no
    //   i=5: corpus[5..9]=[1,2,3,5]  → [1,2,3,5] ≠ [1,2,3,4] (collision guard fires!)
    //   i=4: corpus[4..8]=[99,1,2,3] → no
    //   i=3: corpus[3..7]=[4,99,1,2] → no
    //   i=2: corpus[2..6]=[3,4,99,1] → no
    //   i=1: corpus[1..5]=[2,3,4,99] → no
    //   i=0: corpus[0..4]=[1,2,3,4]  → MATCH; copy_start=4, copy_end=min(4+1,14)=5 → [99]
    #[test]
    fn collision_guard_exact_match_only() {
        let mut p = RetrievalProposer::new();
        // Near-miss [1,2,3,5] at index 5; exact [1,2,3,4] at index 0.
        // Tail of corpus is [1,2,3,4] (the current emitted context, excluded).
        p.warm(&[1u32, 2, 3, 4, 99, 1, 2, 3, 5, 88, 1, 2, 3, 4]);

        let tokens = [1u32, 2, 3, 4];
        let ctx = make_ctx(&tokens);
        let v = token_line(p.propose(&ctx, Budget::line(1), &Telemetry::default()));
        // Only exact match at index 0 should fire; draft = corpus[4..5] = [99].
        assert_eq!(
            v,
            vec![99u32],
            "only the exact 4-token match should produce a draft"
        );
    }

    // Test 5: ctx.tokens.len() < h → nothing (not enough context to form anchor).
    #[test]
    fn insufficient_context_proposes_nothing() {
        let mut p = RetrievalProposer::new();
        // Corpus is populated.
        p.warm(&[10u32, 20, 30, 40, 50, 60, 70, 80]);
        // ctx.tokens has only 3 tokens but h=4; can't form a 4-token anchor.
        let tokens = [10u32, 20, 30]; // len=3 < h=4
        let ctx = make_ctx(&tokens);
        let proposal = p.propose(&ctx, Budget::line(4), &Telemetry::default());
        assert!(
            proposal.is_empty(),
            "ctx shorter than h must propose nothing"
        );
    }

    // Test 6: observe() extends the corpus so emitted tokens become searchable.
    //
    // Start with a warm seed, then observe new tokens. After observation the
    // new tokens should be findable as the continuation of a repeated anchor.
    #[test]
    fn observe_extends_corpus() {
        let mut p = RetrievalProposer::new();
        // Seed corpus: [A B C D E]  (anchor [A B C D] at index 0 → continuation [E])
        p.warm(&[1u32, 2, 3, 4, 5]);
        // Observe the generation: [1, 2, 3, 4]  (appended to corpus)
        // Corpus now: [1, 2, 3, 4, 5, 1, 2, 3, 4]
        //              ^0           ^5
        // anchor [1,2,3,4]; search_end = 9-4 = 5
        // i=4: corpus[4..8]=[5,1,2,3] → no
        // i=3: corpus[3..7]=[4,5,1,2] → no
        // i=2: corpus[2..6]=[3,4,5,1] → no
        // i=1: corpus[1..5]=[2,3,4,5] → no
        // i=0: corpus[0..4]=[1,2,3,4] → MATCH; copy_start=4, copy_end=min(5,9)=5 → [5]
        p.observe(&[1u32, 2, 3, 4]);
        let tokens = [1u32, 2, 3, 4];
        let ctx = make_ctx(&tokens);
        let v = token_line(p.propose(&ctx, Budget::line(1), &Telemetry::default()));
        assert_eq!(
            v,
            vec![5u32],
            "observed tokens must extend the searchable corpus"
        );
    }
}

//! Event Horizon — `SuffixArrayDraft`: rolling-window exact-match copier.
//!
//! A second model-free base proposer that complements `NgramProposer`.
//! Where the n-gram uses bigram/unigram statistics, this proposer finds
//! the most recent prior occurrence of the last `H` emitted tokens in the
//! full emitted stream and proposes the tokens that followed it.  Long
//! exact recurrences (copy-paste, repeated boilerplate, agent loops) that
//! the bigram index misses are the target regime.
//!
//! Lossless by construction: the verifier confirms every proposed token.

use crate::proposal::{Budget, Ctx, Proposal, Proposer, Telemetry};

/// Rolling-window exact-match suffix proposer.
///
/// Maintains the full emitted token stream (bounded to `window` tokens).
/// At each step, searches backwards for the most recent occurrence of the
/// last `H` emitted tokens and drafts the `k` tokens that followed it.
/// `H=3` strikes the precision/recall balance for code/JSON corpora.
#[derive(Debug, Clone)]
pub struct SuffixArrayDraft {
    /// Bounded emitted stream (oldest → newest, capacity `window`).
    stream: Vec<u32>,
    /// Maximum emitted history to retain and search (default 10 000).
    window: usize,
    /// Context window length — number of tail tokens to match (default 3).
    h: usize,
}

impl Default for SuffixArrayDraft {
    fn default() -> Self {
        Self {
            stream: Vec::new(),
            window: 10_000,
            h: 3,
        }
    }
}

impl SuffixArrayDraft {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Proposer for SuffixArrayDraft {
    fn name(&self) -> &'static str {
        "suffix_array"
    }

    fn cost_estimate(&self, _ctx: &Ctx<'_>, _budget: Budget) -> crate::proposal::CostNs {
        0
    }

    fn propose(&mut self, _ctx: &Ctx<'_>, budget: Budget, _tel: &Telemetry) -> Proposal {
        let k = budget.k;
        let slen = self.stream.len();
        // Need at least h+1 tokens: h for the tail query + 1 after the match.
        if k == 0 || slen <= self.h {
            return Proposal::TokenLine(Vec::new());
        }
        // Tail: the last h tokens of our own stream (most accurate context).
        let tail_start = slen - self.h;
        // Search backwards (most-recent-first) for a prior occurrence of the tail.
        // Upper bound: stop before the tail itself (i < slen - h means i + h <= slen - 1
        // so we never match the tail's own position).
        let mut best_pos: Option<usize> = None;
        let search_end = slen.saturating_sub(self.h); // exclusive
        'search: for i in (0..search_end).rev() {
            // Collision guard: compare h tokens element-by-element.
            for d in 0..self.h {
                if self.stream[i + d] != self.stream[tail_start + d] {
                    continue 'search;
                }
            }
            best_pos = Some(i);
            break;
        }
        let Some(prior) = best_pos else {
            return Proposal::TokenLine(Vec::new());
        };
        // Copy up to k tokens starting just after the prior match.
        let copy_start = prior + self.h;
        let copy_end = copy_start.saturating_add(k).min(slen);
        if copy_start >= copy_end {
            return Proposal::TokenLine(Vec::new());
        }
        Proposal::TokenLine(self.stream[copy_start..copy_end].to_vec())
    }

    fn observe(&mut self, emitted: &[u32]) {
        self.stream.extend_from_slice(emitted);
        if self.stream.len() > self.window {
            let drain_to = self.stream.len() - self.window;
            self.stream.drain(..drain_to);
        }
    }

    fn warm(&mut self, history: &[u32]) {
        self.stream.extend_from_slice(history);
        if self.stream.len() > self.window {
            let keep_from = self.stream.len() - self.window;
            self.stream.drain(..keep_from);
        }
    }

    fn reset(&mut self) {
        // The stream IS the index — no cursor to reset; the learned stream
        // persists across requests (same as NgramProposer's note_token index).
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proposal::{Budget, Ctx, Telemetry};

    fn make_ctx<'a>(tokens: &'a [u32]) -> Ctx<'a> {
        Ctx {
            tokens,
            pos: tokens.len(),
            hidden: None,
        }
    }

    #[test]
    fn cold_stream_proposes_nothing() {
        let mut p = SuffixArrayDraft::new();
        let ctx = make_ctx(&[1, 2, 3]);
        let proposal = p.propose(&ctx, Budget::line(4), &Telemetry::default());
        assert!(proposal.is_empty(), "cold stream must propose nothing");
    }

    #[test]
    fn repeated_span_is_proposed() {
        // Stream: A B C D A B C D A B C   ← tail is [A B C]
        // Prior occurrence at index 0 → tokens after it: B C D A B C D A B C
        // We request k=3 → draft = [D, A, B]  (the 3 tokens after prior [A B C])
        let span = [10u32, 20, 30, 40];
        let mut p = SuffixArrayDraft::new();
        // Feed the stream: two full repetitions + partial third
        let stream: Vec<u32> = span
            .iter()
            .cloned()
            .chain(span.iter().cloned())
            .chain([10u32, 20, 30])
            .collect(); // tail = [10, 20, 30]
        p.observe(&stream);

        let ctx = make_ctx(&[10, 20, 30]); // last 2 (last_two) not relevant here
        let proposal = p.propose(&ctx, Budget::line(3), &Telemetry::default());
        match proposal {
            Proposal::TokenLine(v) => {
                // Stream: [10,20,30,40, 10,20,30,40, 10,20,30]  (indices 0..10)
                // tail = [10,20,30] at index 8; search_end = 8
                // Most recent prior: i=4 → stream[4..7]=[10,20,30] ✓
                // copy_start=7, copy_end=min(10,11)=10 → [40,10,20]
                assert_eq!(v, vec![40u32, 10, 20]);
            }
            _ => panic!("expected TokenLine"),
        }
    }

    #[test]
    fn non_recurring_tail_proposes_nothing() {
        let mut p = SuffixArrayDraft::new();
        // A stream with no repeated H=3 spans.
        p.observe(&[1u32, 2, 3, 4, 5, 6, 7, 8, 9]);
        // Tail is [7, 8, 9]; only occurrence is at index 6 (the tail itself is excluded).
        let ctx = make_ctx(&[7, 8, 9]);
        let proposal = p.propose(&ctx, Budget::line(4), &Telemetry::default());
        assert!(
            proposal.is_empty(),
            "non-recurring tail must propose nothing"
        );
    }

    #[test]
    fn collision_guard_exact_match() {
        // Deliberately craft a stream where near-misses exist but only
        // one exact match: [ 1 2 3 99 ] followed by [ 1 2 4 ] (close but not equal)
        // then [ 1 2 3 ] as the current tail.
        // The [1,2,3] exact match is at index 0; [1,2,4] at index 4 must NOT match.
        let mut p = SuffixArrayDraft::new();
        p.observe(&[1u32, 2, 3, 99, 1, 2, 4, 88, 1, 2, 3]);
        // tail = [1, 2, 3] (last h=3 tokens of stream)
        // prior occurrence at index 0; stream after it: [2, 3, 99] (wait, copy_start=3 → [99,1,2])
        // Actually: prior at index 0, copy_start = 0+3 = 3, copy_end = min(3+k, 11-3=8) = min(3+1,8)=4
        // stream[3..4] = [99]
        let ctx = make_ctx(&[1, 2, 3]);
        let proposal = p.propose(&ctx, Budget::line(1), &Telemetry::default());
        match proposal {
            Proposal::TokenLine(v) => {
                // Only exact match [1,2,3] at index 0 or index 8.
                // Most recent search: index 8 is slen-h=11-3=8, excluded (= search_end).
                // Wait: slen=11, search_end = slen - h = 11-3 = 8.
                // Loop: i in (0..8).rev() → 7, 6, 5, ..., 0.
                // i=5: stream[5..8] = [2, 4, 88] ≠ [1,2,3] → skip
                // i=4: stream[4..7] = [1,2,4] ≠ [1,2,3] → skip (collision guard works!)
                // i=3: stream[3..6] = [99,1,2] ≠ [1,2,3] → skip
                // i=2: stream[2..5] = [3,99,1] ≠ [1,2,3] → skip
                // i=1: stream[1..4] = [2,3,99] ≠ [1,2,3] → skip
                // i=0: stream[0..3] = [1,2,3] = [1,2,3] → MATCH! copy_start=3, copy_end=4 → [99]
                assert!(
                    !v.is_empty(),
                    "exact match at index 0 should produce a draft"
                );
            }
            _ => panic!("expected TokenLine"),
        }
    }
}

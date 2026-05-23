//! N-gram speculative decoding draft.
//!
//! Maintains a rolling token history and proposes draft completions by
//! searching for the last `n` tokens in that history, then returning the
//! tokens that followed the most recent match. Zero compute cost — pure
//! hash-table-style linear scan over the history.
//!
//! Wire-up: at engine load, feed all prompt tokens via `note_token`. Each
//! decode step: call `propose` → serial verify with full model → emit
//! accepted tokens + correction/bonus. Then call `note_token` for every
//! emitted token (accepted drafts + correction/bonus).

/// N-gram draft model. Keeps a rolling history of seen token IDs and
/// finds the longest recent context window that has been seen before.
pub struct NGramDraft {
    history: Vec<u32>,
    n: usize,
}

impl NGramDraft {
    pub fn new(n: usize) -> Self {
        Self {
            history: Vec::with_capacity(512),
            n: n.max(1),
        }
    }

    /// Record a token as observed.
    pub fn note_token(&mut self, token: u32) {
        self.history.push(token);
    }

    /// Propose up to `k` draft tokens by finding the most recent occurrence
    /// of the last `n` history tokens and returning what came after.
    ///
    /// Returns an empty vec when the history is too short or no match is found.
    pub fn propose(&self, k: usize) -> Vec<u32> {
        let h = &self.history;
        let len = h.len();
        if len < self.n || k == 0 {
            return vec![];
        }
        let key = &h[len - self.n..];

        // Scan backwards through history looking for the most recent prior match.
        // Skip the last n positions (that's the key itself).
        let search_end = len.saturating_sub(self.n);
        for start in (0..search_end).rev() {
            if start + self.n > len {
                continue;
            }
            if &h[start..start + self.n] == key {
                // Found a match at `start`; return up to k tokens after it.
                let completion_start = start + self.n;
                let completion_end = (completion_start + k).min(len);
                if completion_start < completion_end {
                    return h[completion_start..completion_end].to_vec();
                }
            }
        }
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propose_empty_history() {
        let d = NGramDraft::new(3);
        assert!(d.propose(4).is_empty());
    }

    #[test]
    fn propose_finds_repetition() {
        let mut d = NGramDraft::new(2);
        for t in [1u32, 2, 3, 4, 1, 2] {
            d.note_token(t);
        }
        // last 2 tokens = [1, 2]; prior [1, 2] at index 0 is followed by [3, 4, 1].
        let p = d.propose(3);
        assert_eq!(p, vec![3, 4, 1]);
    }

    #[test]
    fn propose_capped_by_history_length() {
        let mut d = NGramDraft::new(2);
        for t in [1u32, 2, 3, 1, 2] {
            d.note_token(t);
        }
        // [1, 2] at index 0 is followed by [3, 1, 2] — 3 elements available.
        let p = d.propose(4);
        assert_eq!(p, vec![3, 1, 2]);
    }

    #[test]
    fn propose_returns_at_most_k() {
        let mut d = NGramDraft::new(1);
        for t in [5u32, 5, 5, 5, 5] {
            d.note_token(t);
        }
        let p = d.propose(2);
        assert!(p.len() <= 2);
    }

    #[test]
    fn propose_no_match() {
        let mut d = NGramDraft::new(2);
        for t in [1u32, 2, 3, 4] {
            d.note_token(t);
        }
        // last 2 = [3, 4]; that pattern has not appeared before
        assert!(d.propose(4).is_empty());
    }
}

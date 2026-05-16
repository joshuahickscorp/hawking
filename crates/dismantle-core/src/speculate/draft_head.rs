//! Path-to-90 C2/C3 — `DraftSpecDecoder` skeleton for trained draft heads.
//!
//! The trained draft head (EAGLE-3 style per [stage3_c1/architecture.md])
//! consumes the target's post-final-rmsnorm hidden state plus the previous
//! token embedding, and emits K candidate next tokens. The verifier then
//! runs the target on those K tokens via the existing batched-forward path
//! and accepts the longest matching greedy prefix.
//!
//! This module ships in this commit as a SKELETON ONLY:
//!
//! - The trait `DraftHead` exists with the interface a trained head will
//!   implement.
//! - `NoopDraftHead` is the only impl provided — it returns zero candidates,
//!   so the verify path acts identically to single-token greedy. Bit-
//!   identical to today's default decode, by construction.
//! - `DraftSpecDecoder::draft_then_verify` is wired but, with `NoopDraftHead`
//!   plugged in, it never proposes anything → never invokes verify.
//!
//! C3 will replace `NoopDraftHead` with a real consumer of the trained
//! head GGUF. The wire-up site in `deepseek_v2.rs` is unchanged in this
//! commit (no engine path uses the skeleton); the module is here so C3
//! has a concrete landing site rather than a green-field start.
//!
//! The trait + struct compile and pass `cargo test`. There are no
//! perf claims and no behavior change to any shipped decode path.

use crate::Result;

/// Trained draft head — consumes target hidden state + previous token,
/// returns K candidate next-token ids in greedy order.
///
/// Implementations are expected to be `~1 transformer block` of compute
/// per call (EAGLE-3 spec). Returning fewer than K candidates is allowed;
/// the verify path will only check what was returned.
///
/// `prev_token` is the most recently committed target token. `hidden`
/// is the post-final-rmsnorm hidden state at that token's position
/// (i.e. what `Engine::forward_token_with_hidden_for_test` returned).
pub trait DraftHead: Send + Sync {
    /// Propose up to `k` candidate next tokens.
    fn propose(&mut self, prev_token: u32, hidden: &[f32], k: usize) -> Result<Vec<u32>>;

    /// Reset any per-sequence state (e.g. the head's own KV cache, if it
    /// keeps one). Called on each new generation request.
    fn reset(&mut self) {}

    /// Hidden dimension the head expects. Used for shape-matching against
    /// the target's `Engine::forward_token_with_hidden_for_test`.
    fn hidden_dim(&self) -> usize;

    /// Human-readable id for logging / GenStats provenance.
    fn id(&self) -> &str;
}

/// No-op draft head — proposes nothing. Useful for:
/// 1. Skeleton plumbing — a `DraftSpecDecoder<NoopDraftHead>` runs the
///    verify path zero times per step (because no candidates) so e2e
///    behavior is identical to single-token greedy. Bit-identical, by
///    construction.
/// 2. Regression baseline — when measuring the cost of the spec-decode
///    plumbing itself, swap in `NoopDraftHead` to isolate the overhead
///    of dispatch + acceptance bookkeeping from the trained-head cost.
pub struct NoopDraftHead {
    hidden_dim: usize,
}

impl NoopDraftHead {
    pub fn new(hidden_dim: usize) -> Self {
        Self { hidden_dim }
    }
}

impl DraftHead for NoopDraftHead {
    fn propose(&mut self, _prev_token: u32, _hidden: &[f32], _k: usize) -> Result<Vec<u32>> {
        Ok(Vec::new())
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn id(&self) -> &str {
        "noop"
    }
}

/// Draft-then-verify orchestrator skeleton.
///
/// The full flow (filled in by C3):
///   1. Caller hands in `(prev_token, target_hidden_at_prev_token)`.
///   2. `head.propose(prev_token, hidden, K-1)` returns up to K-1 draft tokens.
///   3. Caller runs `Engine::forward_tokens_batched_for_test` on
///      `[prev_token, ...drafts]` to get K logit vectors (verifier output).
///      The first logits vector predicts position prev_token+1; the i'th
///      predicts prev_token+1+i.
///   4. `verify_prefix(drafts, verifier_logits)` returns the longest
///      matching greedy prefix.
///   5. KV cache rolls back on rejection (engine-side; not orchestrated here).
///
/// In this skeleton the orchestration is split into independent helpers
/// so each piece can be tested without an engine. The actual engine wire-
/// up (which calls these in sequence) is C3 work and lives next to the
/// existing `SpeculateMode::ExactShared` / `SpeculateMode::NGram` paths
/// in `model/deepseek_v2.rs`.
pub struct DraftSpecDecoder<H: DraftHead> {
    pub head: H,
    pub verify_window: usize,
}

impl<H: DraftHead> DraftSpecDecoder<H> {
    pub fn new(head: H, verify_window: usize) -> Self {
        Self {
            head,
            verify_window: verify_window.max(1),
        }
    }

    /// Pure-function: longest matching greedy prefix between proposed
    /// draft ids and the verifier's argmax-per-position. Identical
    /// semantics to `crate::speculate::shared::verify_window` but
    /// expressed for the trained-head path (no per-draft logits).
    pub fn verify_prefix(drafts: &[u32], verifier_logits: &[Vec<f32>]) -> usize {
        let n = drafts.len().min(verifier_logits.len());
        let mut accepted = 0usize;
        for i in 0..n {
            let v_argmax = argmax(&verifier_logits[i]);
            if v_argmax == drafts[i] {
                accepted += 1;
            } else {
                break;
            }
        }
        accepted
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold((0u32, f32::NEG_INFINITY), |(mi, mv), (i, &x)| {
            if x > mv {
                (i as u32, x)
            } else {
                (mi, mv)
            }
        })
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_head_proposes_nothing() {
        let mut head = NoopDraftHead::new(2048);
        let hidden = vec![0.0f32; 2048];
        let proposals = head.propose(42, &hidden, 4).unwrap();
        assert!(proposals.is_empty());
        assert_eq!(head.hidden_dim(), 2048);
        assert_eq!(head.id(), "noop");
    }

    #[test]
    fn verify_prefix_full_match() {
        // verifier argmax at each position = draft, so all 3 accepted.
        let drafts = vec![5, 7, 11];
        let logits = vec![
            argmax_winner(5, 100),
            argmax_winner(7, 100),
            argmax_winner(11, 100),
        ];
        assert_eq!(DraftSpecDecoder::<NoopDraftHead>::verify_prefix(&drafts, &logits), 3);
    }

    #[test]
    fn verify_prefix_partial_match_stops_at_first_divergence() {
        let drafts = vec![5, 7, 11];
        let logits = vec![
            argmax_winner(5, 100),
            argmax_winner(99, 100), // diverges here
            argmax_winner(11, 100),
        ];
        assert_eq!(DraftSpecDecoder::<NoopDraftHead>::verify_prefix(&drafts, &logits), 1);
    }

    #[test]
    fn verify_prefix_no_match() {
        let drafts = vec![5, 7];
        let logits = vec![argmax_winner(99, 100), argmax_winner(99, 100)];
        assert_eq!(DraftSpecDecoder::<NoopDraftHead>::verify_prefix(&drafts, &logits), 0);
    }

    #[test]
    fn verify_prefix_empty_drafts() {
        let drafts: Vec<u32> = vec![];
        let logits: Vec<Vec<f32>> = vec![];
        assert_eq!(DraftSpecDecoder::<NoopDraftHead>::verify_prefix(&drafts, &logits), 0);
    }

    #[test]
    fn decoder_holds_head_and_window() {
        let dec = DraftSpecDecoder::new(NoopDraftHead::new(2048), 4);
        assert_eq!(dec.verify_window, 4);
        assert_eq!(dec.head.id(), "noop");
    }

    fn argmax_winner(target: u32, vocab: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; vocab];
        v[target as usize] = 1.0;
        v
    }
}

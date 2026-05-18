//! Path-to-90 C2/C3 — `DraftSpecDecoder` skeleton for trained draft heads.
//!
//! Supports two head families with one trait:
//!
//! - **EAGLE-3 style** — single post-final-rmsnorm hidden state input.
//!   `inputs.hiddens.len() == 1`, no `routing_mask` or `calib` output.
//! - **EAGLE-4 style** — four hidden states (low/mid/high/shared) input,
//!   plus a 26×64 per-MoE-layer routing-mask prediction and a P(accept)
//!   calibration scalar in the output. `inputs.hiddens.len() == 4`.
//!
//! The verify path consumes only `DraftOutputs::tokens` for the bit-
//! identical-greedy regression — `routing_mask` and `calib` are advisory
//! signals consumed by the masked-verify kernel + cascade utility guard
//! (Path B work).
//!
//! Concrete impls:
//!
//! - `NoopDraftHead` — proposes nothing. e2e identical to single-token
//!   greedy. Used as the regression baseline.
//! - `Eagle4Head` — loads an NPZ checkpoint produced by `eagle4.py train`,
//!   runs the 5-input fusion + 1-transformer-block forward, returns
//!   tokens + mask + calib. **Skeleton in this commit** — `propose()`
//!   returns `Err(Unimplemented)`. See `eagle4_head.rs` and
//!   `reports/path_to_90/eagle4_convergence.md`.
//!
//! No engine path consumes this module yet; the wire-up site in
//! `model/deepseek_v2.rs` next to `SpeculateMode::ExactShared` /
//! `SpeculateMode::NGram` lands in C3.

use crate::Result;

/// Bundle of per-token inputs the head sees. The head decides how many
/// hidden vectors it expects via `n_hiddens()` — implementations are
/// responsible for validating `inputs.hiddens.len() == self.n_hiddens()`.
pub struct DraftInputs<'a> {
    /// The most recently committed target token.
    pub prev_token: u32,
    /// One or more hidden vectors, in head-specific order. For EAGLE-4:
    /// `[h_low, h_mid, h_high, h_shared]`. For EAGLE-3: `[post_norm_hidden]`.
    pub hiddens: &'a [&'a [f32]],
}

/// Bundle of per-token outputs the head returns.
#[derive(Debug, Clone)]
pub struct DraftOutputs {
    /// Up to K candidate next-token ids, greedy-ordered. May be shorter
    /// than K if the head ran out of confident proposals or returned an
    /// empty set (e.g. `NoopDraftHead`).
    pub tokens: Vec<u32>,
    /// EAGLE-4 only: predicted top-8 routed-expert mask per MoE layer,
    /// packed as `[N_MOE_LAYERS * N_ROUTED]` bytes (1 = predicted-active,
    /// 0 = predicted-inactive). `None` for EAGLE-3.
    pub routing_mask: Option<Vec<u8>>,
    /// EAGLE-4 only: predicted probability that the verifier will accept
    /// the head's argmax for the next position. Used by the cascade
    /// utility guard to fall back to autoregressive when low. `None` for
    /// EAGLE-3.
    pub calib: Option<f32>,
}

impl DraftOutputs {
    /// Empty output convenience constructor — used by `NoopDraftHead`
    /// and as the "no proposals" return when calib < threshold.
    pub fn empty() -> Self {
        Self {
            tokens: Vec::new(),
            routing_mask: None,
            calib: None,
        }
    }
}

/// Trained draft head — consumes target hidden state(s) + previous token,
/// returns up to K candidate next-token ids plus (optionally) a routing
/// mask + calibration scalar.
///
/// Implementations are expected to be `~1 transformer block` of compute
/// per call (EAGLE-3 / EAGLE-4 spec). Returning fewer than `k` candidates
/// is allowed; the verify path will only check what was returned.
pub trait DraftHead: Send + Sync {
    /// Propose up to `k` candidate next tokens.
    fn propose(&mut self, inputs: &DraftInputs<'_>, k: usize) -> Result<DraftOutputs>;

    /// Reset any per-sequence state (e.g. the head's own KV cache, if it
    /// keeps one). Called on each new generation request.
    fn reset(&mut self) {}

    /// Hidden dimension the head expects. Used for shape-matching against
    /// the target's `Engine::forward_token_with_hidden_for_test` /
    /// `forward_token_eagle4_for_test`.
    fn hidden_dim(&self) -> usize;

    /// Number of hidden vectors the head expects in `inputs.hiddens`.
    /// 1 for EAGLE-3-style heads, 4 for EAGLE-4.
    fn n_hiddens(&self) -> usize;

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
    fn propose(&mut self, _inputs: &DraftInputs<'_>, _k: usize) -> Result<DraftOutputs> {
        Ok(DraftOutputs::empty())
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn n_hiddens(&self) -> usize {
        1
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
        let hiddens: [&[f32]; 1] = [&hidden];
        let inputs = DraftInputs {
            prev_token: 42,
            hiddens: &hiddens,
        };
        let out = head.propose(&inputs, 4).unwrap();
        assert!(out.tokens.is_empty());
        assert!(out.routing_mask.is_none());
        assert!(out.calib.is_none());
        assert_eq!(head.hidden_dim(), 2048);
        assert_eq!(head.n_hiddens(), 1);
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

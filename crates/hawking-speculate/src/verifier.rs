//! Exact target verifier for user-draft / ExactShared. Every token returned is
//! an argmax of the target model; worst case a proposer is wrong and we fall
//! back to one greedy token. Output is bit-identical to plain greedy (temp==0).
//! Adds NO model math — wraps forward_tokens_verify + forward_token_greedy_tcb.
//! KV bookkeeping stays with the caller (returns the position math via next_seq_len).

use crate::shared::{verify_draft_ids_until_mismatch, VerifyResult};
use crate::token_boundary::{draft_ids, TargetVerification, VerifiedTokenId};
use crate::Result;

/// The only thing the verifier needs from a model. QwenDense is the Phase-0
/// implementor; DeepSeek-V2 can impl the same over its rollback KV later.
/// The callback boundary uses a boxed error so a model in another crate (hawking-core) can
/// implement it without a dependency cycle; the verifier maps it into its own Error.
pub type TargetResult<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub trait ExactTarget {
    /// Batched linear verify: `tokens` at contiguous `positions` in one TCB →
    /// (argmax_per_pos, residual_per_pos). b == tokens.len() must be 1..=8 for the
    /// fast path. Wraps QwenDense::forward_tokens_verify (qwen_dense.rs:9004).
    fn forward_tokens_verify(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> TargetResult<(Vec<u32>, Vec<Vec<f32>>)>;

    /// Single greedy/bonus token; writes KV[pos], returns argmax. Wraps
    /// QwenDense::forward_token_greedy_tcb (qwen_dense.rs:4456).
    fn forward_token_greedy(&mut self, token: u32, pos: usize) -> TargetResult<u32>;

    /// Phase-6: ancestor-mask tree verify? False until the Metal build lands.
    fn supports_tree_verify(&self) -> bool {
        false
    }
}

/// Result of one exact verify pass. accepted ids are argmax-confirmed;
/// correction is the target's argmax at the first divergence (None ⇒ full accept).
///
/// # Speculation safety
///
/// [`Self::accepted_verified`] / [`Self::correction_verified`] are the only
/// forms durable sinks may consume. [`Self::accepted`] remains a raw `u32`
/// mirror for existing engine loops (qwen_dense / tree) and is **not** a
/// durable-sink input — use the verified wrappers.
#[derive(Debug, Clone, Default)]
pub struct VerifyOutcome {
    /// Raw accepted draft ids (engine mirror of [`Self::accepted_verified`]).
    pub accepted: Vec<u32>,
    pub correction: Option<u32>,
    /// Target-verified accepted prefix. Only these may enter durable sinks.
    pub accepted_verified: Vec<VerifiedTokenId>,
    /// Target's own token at the first divergence, already verified.
    pub correction_verified: Option<VerifiedTokenId>,
    /// KV length the caller sets seq_len to before the next cycle:
    /// reject  ⇒ bonus_pos + accepted.len() + 1 (correction slot)
    /// accept  ⇒ bonus_pos + draft.len()
    pub next_seq_len: usize,
    /// Per-position residuals (hidden tap). Empty unless want_residuals,
    /// so the n-gram base pays zero copy cost.
    pub residuals: Vec<Vec<f32>>,
}

/// Stateless-per-call verifier. Configured once per request.
#[derive(Debug, Clone)]
pub struct Verifier {
    pub max_batch: usize,     // forward_tokens_verify fast-path cap (8)
    pub want_residuals: bool, // fill VerifyOutcome::residuals (hidden tap); off for n-gram
}
impl Default for Verifier {
    fn default() -> Self {
        Self {
            max_batch: 8,
            want_residuals: false,
        }
    }
}

impl Verifier {
    pub fn new(max_batch: usize, want_residuals: bool) -> Self {
        Self {
            max_batch: max_batch.clamp(1, 8),
            want_residuals,
        }
    }

    /// THE single home for the accept rule (retires the inline copy at
    /// qwen_dense.rs:2632). Bit-identical to the inline loop by construction:
    /// same vtoks = [bonus, draft[0..k-1]], same preds[i]==draft[i] test.
    ///
    /// Accepted tokens are promoted through [`TargetVerification`] so only
    /// target-confirmed ids appear in [`VerifyOutcome::accepted_verified`].
    pub fn verify_line<T: ExactTarget>(
        &self,
        target: &mut T,
        bonus: u32,
        bonus_pos: usize,
        draft: &[u32],
    ) -> Result<VerifyOutcome> {
        let gate = TargetVerification::gate();
        // Degenerate: empty draft → one plain greedy bonus step (still lossless).
        if draft.is_empty() {
            let corr = target
                .forward_token_greedy(bonus, bonus_pos)
                .map_err(|e| crate::Error::Model(e.to_string()))?;
            let corr_v = gate.emit_target(corr);
            return Ok(VerifyOutcome {
                accepted: Vec::new(),
                correction: Some(corr),
                accepted_verified: Vec::new(),
                correction_verified: Some(corr_v),
                next_seq_len: bonus_pos + 1,
                residuals: Vec::new(),
            });
        }
        // Clamp bonus + draft ≤ max_batch.
        let k = draft.len().min(self.max_batch.saturating_sub(1));
        let draft = &draft[..k];

        let mut vtoks = Vec::with_capacity(k);
        vtoks.push(bonus);
        if k > 1 {
            vtoks.extend_from_slice(&draft[..k - 1]);
        }
        let vpos: Vec<usize> = (0..k).map(|j| bonus_pos + j).collect();

        let (preds, residuals) = target
            .forward_tokens_verify(&vtoks, &vpos)
            .map_err(|e| crate::Error::Model(e.to_string()))?;
        debug_assert_eq!(preds.len(), k);

        let VerifyResult {
            accepted_count,
            first_divergent_token,
        } = verify_draft_ids_until_mismatch(draft, |i| Ok(preds[i]))?;

        let accepted = draft[..accepted_count].to_vec();
        // Type-level promotion: draft ids become Verified only via the gate, and
        // only for the longest prefix where draft[i] == preds[i].
        let drafts = draft_ids(draft);
        let promote = gate.promote_matching_prefix(&drafts, &preds);
        debug_assert_eq!(
            promote.accepted.len(),
            accepted_count,
            "promote prefix must match verify_draft_ids_until_mismatch"
        );
        let accepted_verified = promote.accepted;
        let correction_verified = first_divergent_token
            .map(|c| gate.emit_target(c))
            .or(promote.correction);
        let next_seq_len = if first_divergent_token.is_some() {
            bonus_pos + accepted_count + 1
        } else {
            bonus_pos + k
        };
        Ok(VerifyOutcome {
            accepted,
            correction: first_divergent_token,
            accepted_verified,
            correction_verified,
            next_seq_len,
            residuals: if self.want_residuals {
                residuals
            } else {
                Vec::new()
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct MockTarget {
        preds: Vec<u32>,
    }
    impl ExactTarget for MockTarget {
        fn forward_tokens_verify(
            &mut self,
            tokens: &[u32],
            _positions: &[usize],
        ) -> TargetResult<(Vec<u32>, Vec<Vec<f32>>)> {
            let n = tokens.len();
            Ok((self.preds[..n].to_vec(), vec![Vec::new(); n]))
        }
        fn forward_token_greedy(&mut self, _token: u32, _pos: usize) -> TargetResult<u32> {
            Ok(self.preds[0])
        }
    }
    #[test]
    fn full_accept() {
        let mut t = MockTarget {
            preds: vec![10, 20, 30],
        };
        let v = Verifier::default();
        let o = v.verify_line(&mut t, 1, 5, &[10, 20, 30]).unwrap();
        assert_eq!(o.accepted, vec![10, 20, 30]);
        assert_eq!(
            o.accepted_verified
                .iter()
                .map(|x| x.get())
                .collect::<Vec<_>>(),
            vec![10, 20, 30]
        );
        assert_eq!(o.correction, None);
        assert!(o.correction_verified.is_none());
        assert_eq!(o.next_seq_len, 5 + 3);
    }
    #[test]
    fn mid_reject() {
        let mut t = MockTarget {
            preds: vec![10, 99, 30],
        };
        let v = Verifier::default();
        let o = v.verify_line(&mut t, 1, 5, &[10, 20, 30]).unwrap();
        assert_eq!(o.accepted, vec![10]);
        assert_eq!(o.accepted_verified.len(), 1);
        assert_eq!(o.accepted_verified[0].get(), 10);
        assert_eq!(o.correction, Some(99));
        assert_eq!(o.correction_verified.map(|c| c.get()), Some(99));
        assert_eq!(o.next_seq_len, 5 + 1 + 1);
    }
    #[test]
    fn empty_draft_degenerates() {
        let mut t = MockTarget { preds: vec![42] };
        let v = Verifier::default();
        let o = v.verify_line(&mut t, 7, 5, &[]).unwrap();
        assert!(o.accepted.is_empty());
        assert!(o.accepted_verified.is_empty());
        assert_eq!(o.correction, Some(42));
        assert_eq!(o.correction_verified.map(|c| c.get()), Some(42));
        assert_eq!(o.next_seq_len, 6);
    }
    #[test]
    fn verified_accepted_may_enter_durable_sink_drafts_may_not() {
        use crate::durable::{DurableTokenSink, InMemoryDurableSink};
        let mut t = MockTarget {
            preds: vec![10, 99],
        };
        let v = Verifier::default();
        let o = v.verify_line(&mut t, 1, 0, &[10, 20]).unwrap();
        let mut sink = InMemoryDurableSink::default();
        for tok in &o.accepted_verified {
            sink.emit_canonical_event(*tok).unwrap();
        }
        if let Some(c) = o.correction_verified {
            sink.emit_canonical_event(c).unwrap();
        }
        assert_eq!(sink.token_ids(), vec![10, 99]);
    }
}

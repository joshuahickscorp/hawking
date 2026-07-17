//! Event Horizon — the exact target verifier. Every token returned is an argmax
//! of the target model; worst case a proposer is wrong and we fall back to one
//! greedy token. Output is bit-identical to plain greedy (Phase 0, temp==0).
//! Adds NO model math — wraps forward_tokens_verify + forward_token_greedy_tcb.
//! KV bookkeeping stays with the caller (returns the position math via next_seq_len).

use crate::speculate::shared::{verify_draft_ids_until_mismatch, VerifyResult};
use crate::Result;

/// The only thing the verifier needs from a model. QwenDense is the Phase-0
/// implementor; DeepSeek-V2 can impl the same over its rollback KV later.
pub trait ExactTarget {
    /// Batched linear verify: `tokens` at contiguous `positions` in one TCB →
    /// (argmax_per_pos, residual_per_pos). b == tokens.len() must be 1..=8 for the
    /// fast path. Wraps QwenDense::forward_tokens_verify (qwen_dense.rs:9004).
    fn forward_tokens_verify(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<(Vec<u32>, Vec<Vec<f32>>)>;

    /// Single greedy/bonus token; writes KV[pos], returns argmax. Wraps
    /// QwenDense::forward_token_greedy_tcb (qwen_dense.rs:4456).
    fn forward_token_greedy(&mut self, token: u32, pos: usize) -> Result<u32>;

    /// Phase-6: ancestor-mask tree verify? False until the Metal build lands.
    fn supports_tree_verify(&self) -> bool {
        false
    }
}

/// Result of one exact verify pass. accepted ids are argmax-confirmed;
/// correction is the target's argmax at the first divergence (None ⇒ full accept).
#[derive(Debug, Clone, Default)]
pub struct VerifyOutcome {
    pub accepted: Vec<u32>,
    pub correction: Option<u32>,
    /// KV length the caller sets seq_len to before the next cycle:
    /// reject  ⇒ bonus_pos + accepted.len() + 1 (correction slot)
    /// accept  ⇒ bonus_pos + draft.len()
    pub next_seq_len: usize,
    /// Per-position residuals (EAGLE hidden tap). Empty unless want_residuals,
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
    pub fn verify_line<T: ExactTarget>(
        &self,
        target: &mut T,
        bonus: u32,
        bonus_pos: usize,
        draft: &[u32],
    ) -> Result<VerifyOutcome> {
        // Degenerate: empty draft → one plain greedy bonus step (still lossless).
        if draft.is_empty() {
            let corr = target.forward_token_greedy(bonus, bonus_pos)?;
            return Ok(VerifyOutcome {
                accepted: Vec::new(),
                correction: Some(corr),
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

        let (preds, residuals) = target.forward_tokens_verify(&vtoks, &vpos)?;
        debug_assert_eq!(preds.len(), k);

        let VerifyResult {
            accepted_count,
            first_divergent_token,
        } = verify_draft_ids_until_mismatch(draft, |i| Ok(preds[i]))?;

        let accepted = draft[..accepted_count].to_vec();
        let next_seq_len = if first_divergent_token.is_some() {
            bonus_pos + accepted_count + 1
        } else {
            bonus_pos + k
        };
        Ok(VerifyOutcome {
            accepted,
            correction: first_divergent_token,
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

    /// Mock target driven by canned argmax preds; no Metal, no model.
    struct MockTarget {
        preds: Vec<u32>,
    }
    impl ExactTarget for MockTarget {
        fn forward_tokens_verify(
            &mut self,
            tokens: &[u32],
            _positions: &[usize],
        ) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
            // Return the first `tokens.len()` canned preds.
            let n = tokens.len();
            Ok((self.preds[..n].to_vec(), vec![Vec::new(); n]))
        }
        fn forward_token_greedy(&mut self, _token: u32, _pos: usize) -> Result<u32> {
            Ok(self.preds[0])
        }
    }

    #[test]
    fn full_accept() {
        // preds confirm every draft token (model argmax == draft[i] at each step)
        // → full accept, no correction, next_seq_len = bonus_pos + k.
        let mut t = MockTarget {
            preds: vec![10, 20, 30],
        };
        let v = Verifier::default();
        let o = v.verify_line(&mut t, 1, 5, &[10, 20, 30]).unwrap();
        assert_eq!(o.accepted, vec![10, 20, 30]);
        assert_eq!(o.correction, None);
        assert_eq!(o.next_seq_len, 5 + 3);
        // Real losslessness is the engine bit-identity gate (P0.6); this pins the contract.
    }

    #[test]
    fn mid_reject() {
        let mut t = MockTarget {
            preds: vec![10, 99, 30],
        };
        let v = Verifier::default();
        let o = v.verify_line(&mut t, 1, 5, &[10, 20, 30]).unwrap();
        assert_eq!(o.accepted, vec![10]);
        assert_eq!(o.correction, Some(99));
        assert_eq!(o.next_seq_len, 5 + 1 + 1);
    }

    #[test]
    fn empty_draft_degenerates() {
        let mut t = MockTarget { preds: vec![42] };
        let v = Verifier::default();
        let o = v.verify_line(&mut t, 7, 5, &[]).unwrap();
        assert!(o.accepted.is_empty());
        assert_eq!(o.correction, Some(42));
        assert_eq!(o.next_seq_len, 6);
    }
}

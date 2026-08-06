//! Dual KV state for speculative decoding.
//!
//! # Invariant (`COMMITTED_IS_SOURCE_OF_TRUTH`)
//!
//! There are exactly two KV views:
//!
//! 1. **Committed target KV** — positions the target has verified; durable and
//!    the only state that may back sinks.
//! 2. **Provisional draft KV** — a pure extension of the committed base used
//!    while drafting. On reject, **rollback** restores the committed state
//!    bit-identically. On accept, **rebase** advances the committed cursor by
//!    the verified prefix only.
//!
//! After every operation:
//!
//! ```text
//! provisional.base_fingerprint == committed.fingerprint
//! provisional.seq_len >= committed.seq_len
//! provisional.seq_len - committed.seq_len == provisional.draft_len
//! ```
//!
//! Named: **`COMMITTED_IS_SOURCE_OF_TRUTH`**.

use crate::token_boundary::VerifiedTokenId;
use sha2::{Digest, Sha256};

/// Fingerprint of committed KV contents (token ids + seq_len). Used for
/// bit-identity checks without holding GPU tensors in this leaf crate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KvFingerprint([u8; 32]);

impl KvFingerprint {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    fn of(seq_len: usize, token_ids: &[u32]) -> Self {
        let mut h = Sha256::new();
        h.update(seq_len.to_le_bytes());
        h.update((token_ids.len() as u64).to_le_bytes());
        for &t in token_ids {
            h.update(t.to_le_bytes());
        }
        let dig = h.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&dig);
        KvFingerprint(out)
    }
}

/// Committed target KV — only target-verified tokens live here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedKv {
    /// Logical sequence length (positions written).
    pub seq_len: usize,
    /// Token ids occupying positions `0..seq_len` (CPU mirror for the invariant).
    token_ids: Vec<u32>,
    fingerprint: KvFingerprint,
}

impl CommittedKv {
    pub fn empty() -> Self {
        Self::from_tokens(vec![])
    }

    pub fn from_tokens(token_ids: Vec<u32>) -> Self {
        let seq_len = token_ids.len();
        let fingerprint = KvFingerprint::of(seq_len, &token_ids);
        Self {
            seq_len,
            token_ids,
            fingerprint,
        }
    }

    pub fn fingerprint(&self) -> &KvFingerprint {
        &self.fingerprint
    }

    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    fn recompute_fp(&mut self) {
        self.seq_len = self.token_ids.len();
        self.fingerprint = KvFingerprint::of(self.seq_len, &self.token_ids);
    }
}

/// Provisional draft KV: committed base + uncommitted draft extension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvisionalKv {
    base: CommittedKv,
    /// Draft token ids stacked on top of `base` (not yet verified).
    draft_ids: Vec<u32>,
}

impl ProvisionalKv {
    pub fn from_committed(committed: &CommittedKv) -> Self {
        Self {
            base: committed.clone(),
            draft_ids: Vec::new(),
        }
    }

    pub fn draft_len(&self) -> usize {
        self.draft_ids.len()
    }

    pub fn seq_len(&self) -> usize {
        self.base.seq_len + self.draft_ids.len()
    }

    pub fn base_fingerprint(&self) -> &KvFingerprint {
        self.base.fingerprint()
    }

    pub fn draft_ids(&self) -> &[u32] {
        &self.draft_ids
    }

    /// Append a raw draft proposal onto the provisional extension.
    pub fn push_draft(&mut self, id: u32) {
        self.draft_ids.push(id);
    }

    pub fn push_drafts(&mut self, ids: &[u32]) {
        self.draft_ids.extend_from_slice(ids);
    }
}

/// Paired dual state. Enforces `COMMITTED_IS_SOURCE_OF_TRUTH`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DualKv {
    committed: CommittedKv,
    provisional: ProvisionalKv,
}

impl DualKv {
    pub fn new(committed: CommittedKv) -> Self {
        let provisional = ProvisionalKv::from_committed(&committed);
        let dual = Self {
            committed,
            provisional,
        };
        dual.assert_invariant();
        dual
    }

    pub fn empty() -> Self {
        Self::new(CommittedKv::empty())
    }

    pub fn committed(&self) -> &CommittedKv {
        &self.committed
    }

    pub fn provisional(&self) -> &ProvisionalKv {
        &self.provisional
    }

    pub fn provisional_mut(&mut self) -> &mut ProvisionalKv {
        &mut self.provisional
    }

    /// INVARIANT `COMMITTED_IS_SOURCE_OF_TRUTH`.
    pub fn check_invariant(&self) -> Result<(), &'static str> {
        if self.provisional.base.fingerprint != self.committed.fingerprint {
            return Err("provisional.base fingerprint diverged from committed");
        }
        if self.provisional.base.seq_len != self.committed.seq_len {
            return Err("provisional.base seq_len diverged from committed");
        }
        if self.provisional.base.token_ids != self.committed.token_ids {
            return Err("provisional.base token_ids diverged from committed");
        }
        if self.provisional.seq_len() != self.committed.seq_len + self.provisional.draft_len() {
            return Err("provisional seq_len != committed + draft_len");
        }
        Ok(())
    }

    pub fn assert_invariant(&self) {
        if let Err(e) = self.check_invariant() {
            panic!("COMMITTED_IS_SOURCE_OF_TRUTH violated: {e}");
        }
    }

    /// Speculate: stack draft ids on the provisional KV. Does not touch committed.
    pub fn speculate(&mut self, draft_ids: &[u32]) {
        self.assert_invariant();
        self.provisional.push_drafts(draft_ids);
        self.assert_invariant();
    }

    /// Reject path: discard the provisional extension and restore provisional to
    /// a pure clone of committed. Committed is **bit-identical** to before the
    /// failed speculation (same fingerprint, same token ids).
    pub fn rollback(&mut self) {
        let before = self.committed.fingerprint().clone();
        let before_ids = self.committed.token_ids().to_vec();
        self.provisional = ProvisionalKv::from_committed(&self.committed);
        assert_eq!(
            self.committed.fingerprint(),
            &before,
            "rollback must not mutate committed fingerprint"
        );
        assert_eq!(
            self.committed.token_ids(),
            &before_ids[..],
            "rollback must not mutate committed token ids"
        );
        self.assert_invariant();
    }

    /// Accept path: advance committed by the verified prefix only, then rebase
    /// provisional on the new committed base (dropping any leftover draft tail).
    pub fn rebase(&mut self, accepted: &[VerifiedTokenId]) {
        self.assert_invariant();
        for v in accepted {
            self.committed.token_ids.push(v.get());
        }
        self.committed.recompute_fp();
        self.provisional = ProvisionalKv::from_committed(&self.committed);
        self.assert_invariant();
    }

    /// Target-direct append (non-spec greedy step): write one verified token to
    /// committed and rebase provisional.
    pub fn append_verified(&mut self, token: VerifiedTokenId) {
        self.rebase(&[token]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token_boundary::{DraftTokenId, TargetVerification};
    #[test]
    fn rollback_restores_committed_bit_identical() {
        let gate = TargetVerification::gate();
        let mut dual = DualKv::new(CommittedKv::from_tokens(vec![10, 20, 30]));
        let fp_before = dual.committed().fingerprint().clone();
        let ids_before = dual.committed().token_ids().to_vec();
        dual.speculate(&[99, 100, 101]);
        assert_eq!(dual.provisional().draft_len(), 3);
        assert_eq!(dual.provisional().seq_len(), 6);
        dual.rollback();
        assert_eq!(dual.committed().fingerprint(), &fp_before);
        assert_eq!(dual.committed().token_ids(), &ids_before[..]);
        assert_eq!(dual.provisional().draft_len(), 0);
        assert_eq!(dual.provisional().seq_len(), dual.committed().seq_len);
        dual.assert_invariant();
        assert!(gate.try_promote(DraftTokenId::id(99), 11).is_err());
    }
    #[test]
    fn rebase_advances_committed_by_verified_prefix_only() {
        let gate = TargetVerification::gate();
        let mut dual = DualKv::new(CommittedKv::from_tokens(vec![1]));
        dual.speculate(&[2, 3, 4]);
        let accepted = vec![gate.emit_target(2u32), gate.emit_target(3u32)];
        dual.rebase(&accepted);
        assert_eq!(dual.committed().token_ids(), &[1, 2, 3]);
        assert_eq!(dual.provisional().draft_len(), 0);
        dual.assert_invariant();
    }
    #[test]
    fn invariant_holds_across_speculate_rollback_rebase() {
        let gate = TargetVerification::gate();
        let mut dual = DualKv::empty();
        dual.append_verified(gate.emit_target(5u32));
        dual.speculate(&[6, 7]);
        dual.assert_invariant();
        dual.rollback();
        dual.assert_invariant();
        dual.speculate(&[6]);
        dual.rebase(&[gate.emit_target(6u32)]);
        dual.assert_invariant();
        assert_eq!(dual.committed().token_ids(), &[5, 6]);
    }
}

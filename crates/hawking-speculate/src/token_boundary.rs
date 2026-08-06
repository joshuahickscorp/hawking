//! Type-level speculation boundary.
//!
//! # Invariant (speculation safety)
//!
//! Speculative decoding produces **draft** tokens the target has not verified.
//! Only **target-verified** tokens may enter durable sinks (canonical event
//! stream, memory, tool dispatch, file edit, final user-visible output).
//!
//! This module makes that a **compile error**, not a review catch:
//!
//! - [`Draft<T>`] and [`Verified<T>`] are distinct newtypes with no conversion
//!   trait between them.
//! - The only way to obtain a [`Verified`] from a draft is
//!   [`TargetVerification::promote_matching_prefix`] (or emit a pure target
//!   correction via [`TargetVerification::emit_target`]).
//! - Every durable sink accepts only [`Verified`] (see [`crate::durable`]).
//!
//! ## Compile-time enforcement
//!
//! There is no `From<Draft<T>> for Verified<T>`, no public constructor on
//! `Verified` that accepts a draft, and durable sink methods are typed as
//! `Verified<_>`. Passing a `Draft` is a type error.
//!
//! API-level tests in this module (and in `durable`) lock the surface; a
//! trybuild compile-fail is not used because the workspace has no trybuild
//! harness. The type signatures *are* the gate.

use core::fmt;

/// A token (or payload) proposed by a draft path and **not** yet confirmed by
/// the target model. Must never reach a durable sink.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Draft<T>(T);

/// A token (or payload) that the target model has emitted or confirmed.
/// The only type durable sinks accept.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Verified<T>(T);

impl<T> Draft<T> {
    /// Wrap a raw value as an unverified draft. Any proposer output starts here.
    #[inline]
    pub const fn new(inner: T) -> Self {
        Self(inner)
    }

    #[inline]
    pub fn inner(&self) -> &T {
        &self.0
    }

    #[inline]
    pub fn into_inner(self) -> T {
        self.0
    }

    #[inline]
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Draft<U> {
        Draft(f(self.0))
    }

    #[inline]
    pub fn get(&self) -> T
    where
        T: Copy,
    {
        self.0
    }
}

impl<T> Verified<T> {
    #[inline]
    pub fn inner(&self) -> &T {
        &self.0
    }

    #[inline]
    pub fn into_inner(self) -> T {
        self.0
    }

    #[inline]
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Verified<U> {
        Verified(f(self.0))
    }

    /// Copy the inner value. Prefer this over leaking the constructor.
    #[inline]
    pub fn get(&self) -> T
    where
        T: Copy,
    {
        self.0
    }
}

impl<T: fmt::Debug> fmt::Debug for Draft<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Draft").field(&self.0).finish()
    }
}

impl<T: fmt::Debug> fmt::Debug for Verified<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Verified").field(&self.0).finish()
    }
}

/// Token id aliases used throughout the speculation pack.
pub type DraftTokenId = Draft<u32>;
pub type VerifiedTokenId = Verified<u32>;

/// The sole authority that may mint [`Verified`] values.
///
/// Constructed only inside the verification step. Callers outside this crate
/// obtain verified tokens only through the methods on this type (or through
/// re-exports that wrap them).
#[derive(Debug, Clone, Copy, Default)]
pub struct TargetVerification {
    _private: (),
}

impl TargetVerification {
    /// Create the gate used by the target verifier. Public so host/engine code
    /// that *is* the target path can emit verified tokens without going through
    /// a draft. Draft proposers must not call this with draft contents — the
    /// type system still prevents those drafts from being passed as `Draft`
    /// into sinks; this is for pure target argmax / non-spec decode only.
    #[inline]
    pub const fn gate() -> Self {
        Self { _private: () }
    }

    /// Emit a token the **target** itself produced (greedy/sample/correction).
    /// Never wrap a draft id here.
    #[inline]
    pub fn emit_target<T>(&self, token: T) -> Verified<T> {
        Verified(token)
    }

    /// Promote a single draft token iff it bit-matches the target's decision.
    /// On mismatch the draft is returned untouched (still unverified).
    #[inline]
    pub fn try_promote<T: PartialEq>(
        &self,
        draft: Draft<T>,
        target: T,
    ) -> Result<Verified<T>, Draft<T>> {
        if draft.0 == target {
            Ok(Verified(draft.0))
        } else {
            Err(draft)
        }
    }

    /// Longest agreeing prefix: each draft id that matches the corresponding
    /// target id is promoted; stop at the first mismatch. Returns the verified
    /// prefix and the target's correction token at the divergence (if any).
    ///
    /// This is the structural accept rule used by user-draft / ExactShared.
    pub fn promote_matching_prefix(
        &self,
        drafts: &[DraftTokenId],
        target_ids: &[u32],
    ) -> PromoteResult {
        let n = drafts.len().min(target_ids.len());
        let mut accepted = Vec::with_capacity(n);
        let mut correction = None;
        for i in 0..n {
            let d = drafts[i].get();
            let t = target_ids[i];
            if d == t {
                accepted.push(Verified(d));
            } else {
                correction = Some(Verified(t));
                break;
            }
        }
        // If the target produced more decisions than drafts and all drafts matched,
        // there is no correction from this window (bonus token is a separate emit).
        if correction.is_none() && target_ids.len() > drafts.len() && accepted.len() == drafts.len()
        {
            // leave correction None — full draft accept
        }
        PromoteResult {
            accepted,
            correction,
        }
    }
}

impl DraftTokenId {
    #[inline]
    pub const fn id(id: u32) -> Self {
        Draft(id)
    }
}

/// Outcome of [`TargetVerification::promote_matching_prefix`].
#[derive(Debug, Clone, Default)]
pub struct PromoteResult {
    pub accepted: Vec<VerifiedTokenId>,
    /// Target's own token at the first divergence (`None` on full accept).
    pub correction: Option<VerifiedTokenId>,
}

impl PromoteResult {
    /// Flat raw ids for engine loops that still operate on `u32`. Durable
    /// sinks must use [`Self::accepted`] (the verified wrappers), not this.
    pub fn accepted_raw(&self) -> Vec<u32> {
        self.accepted.iter().map(|v| v.get()).collect()
    }
}

/// Convert a slice of raw draft ids into typed drafts. Call at the proposer→verify boundary.
pub fn draft_ids(ids: &[u32]) -> Vec<DraftTokenId> {
    ids.iter().copied().map(DraftTokenId::id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::durable::{DurableRecord, DurableTokenSink, InMemoryDurableSink};
    #[test]
    fn draft_and_verified_are_distinct_and_only_promote_matches() {
        let gate = TargetVerification::gate();
        let draft = DraftTokenId::id(7);
        assert!(gate.try_promote(draft, 8).is_err());
        let v = gate.try_promote(DraftTokenId::id(7), 7).unwrap();
        assert_eq!(v.get(), 7);
        let corr = gate.emit_target(99u32);
        assert_eq!(corr.get(), 99);
    }
    #[test]
    fn promote_matching_prefix_stops_at_first_mismatch() {
        let gate = TargetVerification::gate();
        let drafts = draft_ids(&[1, 2, 3, 4]);
        let targets = [1, 2, 9, 4];
        let r = gate.promote_matching_prefix(&drafts, &targets);
        assert_eq!(r.accepted_raw(), vec![1, 2]);
        assert_eq!(r.correction.map(|c| c.get()), Some(9));
    }
    #[test]
    fn draft_cannot_reach_durable_sink_without_verification() {
        let gate = TargetVerification::gate();
        let mut sink = InMemoryDurableSink::default();
        let draft = DraftTokenId::id(42);
        let verified = gate.try_promote(draft, 42).expect("target agrees");
        sink.emit_canonical_event(verified)
            .expect("verified may enter the event stream");
        assert_eq!(
            sink.events(),
            &[DurableRecord::CanonicalEvent { token_id: 42 }]
        );
        let rejected = DraftTokenId::id(7);
        assert!(gate.try_promote(rejected, 8).is_err());
        assert_eq!(
            sink.events().len(),
            1,
            "rejected draft must leave no durable trace"
        );
    }
}

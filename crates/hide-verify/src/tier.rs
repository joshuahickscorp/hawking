//! Verification tiers (Bible Book IX, sec 28).
//!
//! HIDE verifies a change through a ladder of tiers ordered by AUTHORITY, not by
//! convenience. The lower tiers are deterministic: they observe a fact (the
//! patch applied, the file parsed, the build succeeded, the test passed) that is
//! reproducible and not open to interpretation. The top tier is a probabilistic
//! model review that reasons about correctness, security, performance, and
//! scope.
//!
//! THE AUTHORITY RULE (sec 28-29), which every consumer of this crate must
//! honor: a probabilistic review may NEVER overrule a failing deterministic
//! gate. A reviewer that "thinks the code is fine" cannot rescue a red build or
//! a failing test; at most it ranks candidates that have ALREADY passed every
//! deterministic gate. This rule is encoded, not merely documented, in
//! [`crate::gate::apply_gate`].

use serde::{Deserialize, Serialize};

/// The tiers of the verification plane, lowest (most authoritative, cheapest,
/// deterministic) to highest (probabilistic review).
///
/// * [`VerificationTier::Tier0Structural`] - the patch applies, files parse,
///   formatting holds. Structural facts about the candidate.
/// * [`VerificationTier::Tier1Deterministic`] - build, typecheck, unit and
///   integration tests, lint, static analysis. The deterministic core.
/// * [`VerificationTier::Tier2Reproduction`] - a bug reproduction or acceptance
///   test that demonstrates the change actually does the thing.
/// * [`VerificationTier::Tier3Environment`] - browser, service, and database
///   checks against a live environment.
/// * [`VerificationTier::Tier4Review`] - correctness, security, performance, and
///   scope reviewers. Probabilistic. DEFERRED_MODEL_REQUIRED: executing a
///   reviewer needs a model and is out of scope for this crate, which carries
///   only the review-role profiles (see [`crate::review`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationTier {
    Tier0Structural,
    Tier1Deterministic,
    Tier2Reproduction,
    Tier3Environment,
    Tier4Review,
}

impl VerificationTier {
    /// True for the deterministic tiers (Tier0 through Tier3): every verdict they
    /// produce is a reproducible fact and is authoritative over any review.
    pub fn is_deterministic(self) -> bool {
        matches!(
            self,
            VerificationTier::Tier0Structural
                | VerificationTier::Tier1Deterministic
                | VerificationTier::Tier2Reproduction
                | VerificationTier::Tier3Environment
        )
    }

    /// True for the probabilistic tier (Tier4 review). A verdict from this tier
    /// may confirm or block, but per the authority rule it can never override a
    /// deterministic failure.
    pub fn is_probabilistic(self) -> bool {
        matches!(self, VerificationTier::Tier4Review)
    }
}

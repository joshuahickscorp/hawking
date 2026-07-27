//! Tier4 review-role profiles (Bible Book IX, sec 28).
//!
//! The top tier of the verification plane is a set of probabilistic reviewers,
//! each with a narrow charter: correctness, security, performance, API
//! compatibility, tests, documentation, simplicity, and scope. This module
//! carries those charters as DATA: for each role, what it focuses on, what
//! context it needs, which output schema it fills, and what it accepts.
//!
//! DEFERRED_MODEL_REQUIRED: executing a review role requires a model. This
//! module deliberately provides ONLY the profiles and a selector. The selector
//! returns a [`ReviewRoleProfile`] (data), never a [`crate::oracle::Verdict`],
//! and it performs NO model call. Wiring a model to a profile to produce a
//! verdict is out of scope for this crate.

use serde::{Deserialize, Serialize};

/// The Tier4 review roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewRole {
    Correctness,
    Security,
    Performance,
    ApiCompatibility,
    Tests,
    Documentation,
    Simplicity,
    Scope,
}

impl ReviewRole {
    /// Every review role, in a stable order.
    pub const ALL: [ReviewRole; 8] = [
        ReviewRole::Correctness,
        ReviewRole::Security,
        ReviewRole::Performance,
        ReviewRole::ApiCompatibility,
        ReviewRole::Tests,
        ReviewRole::Documentation,
        ReviewRole::Simplicity,
        ReviewRole::Scope,
    ];
}

/// A review role's charter, as pure data. Contains no executable behavior and no
/// model handle: it describes what a reviewer of this role would do, so a model
/// harness (elsewhere) can be pointed at it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReviewRoleProfile {
    pub role: ReviewRole,
    /// What this reviewer looks for, in one sentence.
    pub focus: String,
    /// The kinds of context a reviewer of this role needs (diff, tests, deps).
    pub context_kinds: Vec<String>,
    /// A reference to the output schema the reviewer must fill (schema id).
    pub output_schema_ref: String,
    /// The acceptance condition for a passing review of this role.
    pub acceptance: String,
}

/// Return the profile for a review role. This is a pure DATA selector: it builds
/// and returns a [`ReviewRoleProfile`], performs no model call, and never
/// produces a verdict.
pub fn profile_for(role: ReviewRole) -> ReviewRoleProfile {
    let (focus, context_kinds, acceptance): (&str, &[&str], &str) = match role {
        ReviewRole::Correctness => (
            "whether the change does what it claims and handles edge and error cases",
            &["diff", "requirement", "tests", "call_sites"],
            "no correctness defect that a deterministic test could not have caught",
        ),
        ReviewRole::Security => (
            "injection, auth, secret handling, unsafe input, and privilege boundaries",
            &["diff", "threat_model", "dependencies", "config"],
            "no introduced vulnerability and no weakened boundary",
        ),
        ReviewRole::Performance => (
            "algorithmic complexity, allocations, and hot-path regressions",
            &["diff", "benchmarks", "hot_paths"],
            "no unjustified regression on a measured path",
        ),
        ReviewRole::ApiCompatibility => (
            "public surface changes and their effect on existing callers",
            &["diff", "public_api", "call_sites", "semver"],
            "no breaking change without an intentional, documented bump",
        ),
        ReviewRole::Tests => (
            "whether tests cover the change and actually assert its behavior",
            &["diff", "tests", "coverage"],
            "the change is exercised by an assertion, not merely compiled",
        ),
        ReviewRole::Documentation => (
            "whether public items and behavior changes are documented accurately",
            &["diff", "doc_comments", "changelog"],
            "no undocumented public item and no stale doc",
        ),
        ReviewRole::Simplicity => (
            "unnecessary complexity, duplication, and reinvention of existing code",
            &["diff", "surrounding_code", "existing_utilities"],
            "no simpler equivalent was passed over without reason",
        ),
        ReviewRole::Scope => (
            "whether the change stays within its stated intent and touches nothing else",
            &["diff", "intent", "task"],
            "no change outside the declared scope",
        ),
    };

    ReviewRoleProfile {
        role,
        focus: focus.to_string(),
        context_kinds: context_kinds.iter().map(|s| s.to_string()).collect(),
        output_schema_ref: format!("hide.review.{}.v1", schema_slug(role)),
        acceptance: acceptance.to_string(),
    }
}

/// All review-role profiles, in [`ReviewRole::ALL`] order.
pub fn all_profiles() -> Vec<ReviewRoleProfile> {
    ReviewRole::ALL.iter().copied().map(profile_for).collect()
}

fn schema_slug(role: ReviewRole) -> &'static str {
    match role {
        ReviewRole::Correctness => "correctness",
        ReviewRole::Security => "security",
        ReviewRole::Performance => "performance",
        ReviewRole::ApiCompatibility => "api_compatibility",
        ReviewRole::Tests => "tests",
        ReviewRole::Documentation => "documentation",
        ReviewRole::Simplicity => "simplicity",
        ReviewRole::Scope => "scope",
    }
}

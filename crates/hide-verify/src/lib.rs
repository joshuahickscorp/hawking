//! hide-verify: the verification plane (Bible Book IX, sec 28-29).
//!
//! HIDE never accepts a change on faith. Every candidate passes through a ladder
//! of verification tiers ordered by AUTHORITY: structural facts, then the
//! deterministic core (build, typecheck, tests, lint, static analysis), then
//! reproduction, then live-environment checks, and only at the top a set of
//! probabilistic model reviewers. The one rule that holds the whole plane
//! together is this: a probabilistic review may NEVER overrule a failing
//! deterministic gate.
//!
//! This crate implements the DETERMINISTIC part in full and model-free:
//!
//! - [`tier::VerificationTier`]: the tiers and which are deterministic.
//! - [`oracle`]: the [`oracle::Oracle`] trait, the [`oracle::Verdict`]
//!   (`Pass` / `Fail` / `Skipped`) with [`oracle::Evidence`].
//! - [`receipt::VerificationReceipt`]: the stable, serde-serializable evidence
//!   record (sec 29).
//! - [`static_analysis::StaticAnalysisOracle`]: a REAL deterministic lint over
//!   Rust source text (unwrap/expect outside tests, marker macros, the house-rule
//!   dash lint, long functions, TODO/FIXME), running over source strings or a
//!   walked directory. No model, no subprocess.
//! - [`rereview`]: the re-review dependency model. Given prior receipts and a set
//!   of changed paths, it returns exactly the receipts whose scope intersects the
//!   change and must be re-run.
//! - [`gate::apply_gate`]: the authority rule, encoded so a review can never
//!   override a deterministic failure.
//!
//! The PROBABILISTIC part is data only. [`review`] carries the Tier4 review-role
//! profiles and a selector, but executing a review role requires a model and is
//! DEFERRED_MODEL_REQUIRED: no model is called anywhere in this crate.

pub mod error;
pub mod finding;
pub mod gate;
pub mod oracle;
pub mod receipt;
pub mod rereview;
pub mod review;
pub mod static_analysis;
pub mod tier;

pub use error::{Result, VerifyError};
pub use finding::{CheckKind, Finding, Severity};
pub use gate::{apply_gate, probabilistic_can_override_deterministic, GateDecision, TieredVerdict};
pub use oracle::{
    Evidence, Oracle, OracleClass, OracleOutcome, SourceFile, VerificationInput, Verdict,
};
pub use receipt::{source_hash, source_hash_of, VerificationReceipt};
pub use rereview::{invalidated_ids, invalidated_receipts, paths_intersect};
pub use review::{all_profiles, profile_for, ReviewRole, ReviewRoleProfile};
pub use static_analysis::{StaticAnalysisOracle, DEFAULT_LONG_FUNCTION_THRESHOLD};
pub use tier::VerificationTier;

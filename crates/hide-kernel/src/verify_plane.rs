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

pub use error::{Result, VerifyError};
pub use finding::{CheckKind, Finding, Severity};
pub use gate::{apply_gate, probabilistic_can_override_deterministic, GateDecision, TieredVerdict};
pub use oracle::{
    Evidence, Oracle, OracleClass, OracleOutcome, SourceFile, Verdict, VerificationInput,
};
pub use receipt::{source_hash, source_hash_of, VerificationReceipt};
pub use rereview::{invalidated_ids, invalidated_receipts, paths_intersect};
pub use review::{all_profiles, profile_for, ReviewRole, ReviewRoleProfile};
pub use static_analysis::{StaticAnalysisOracle, DEFAULT_LONG_FUNCTION_THRESHOLD};
pub use tier::VerificationTier;

pub mod error {
    //! Errors for the verification plane. Only the filesystem-facing paths (a walked
    //! directory scan) can fail; the in-memory analysis over source strings is
    //! infallible and returns findings directly.

    use thiserror::Error;

    /// A failure while walking or reading a directory of Rust source.
    #[derive(Debug, Error)]
    pub enum VerifyError {
        /// The directory walker itself failed (permissions, a vanished entry, a
        /// symlink loop). Carries a rendered message so the public API never leaks
        /// the underlying `walkdir` error type.
        #[error("directory walk failed under {root}: {message}")]
        Walk { root: String, message: String },

        /// A source file could not be read as UTF-8 text.
        #[error("failed to read {path}: {message}")]
        Read { path: String, message: String },
    }

    /// Result alias for the fallible directory-facing surface of this crate.
    pub type Result<T> = std::result::Result<T, VerifyError>;
}

pub mod finding {
    //! Typed findings produced by deterministic checks.
    //!
    //! A [`Finding`] is the atomic output of the static-analysis oracle: which check
    //! fired, in which file, on which line, at what severity, and a human-readable
    //! message. Findings are pure data with a stable serde shape so a repair stage
    //! can feed the exact (file, line, message) back to the author.

    use serde::{Deserialize, Serialize};

    /// Severity of a finding. Ordered `Info < Warning < Error`, so a gate can ask
    /// whether any finding is at or above [`Severity::Warning`] with a comparison.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum Severity {
        Info,
        Warning,
        Error,
    }

    /// Which deterministic check produced a finding. Stable identifiers so downstream
    /// tooling can filter or de-duplicate by kind.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum CheckKind {
        /// `.unwrap()` or `.expect(...)` used outside `#[cfg(test)]` / `#[test]` code.
        UnwrapOutsideTest,
        /// A `panic!` / `todo!` / `unimplemented!` / `unreachable!` marker macro.
        PanicMarker,
        /// An en dash (U+2013) or em dash (U+2014). The house-rule lint.
        EmDash,
        /// A function whose body exceeds the configured line-count threshold.
        LongFunction,
        /// A `TODO` / `FIXME` / `XXX` marker in the source text.
        TodoMarker,
    }

    /// A single deterministic observation about a source file.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Finding {
        pub check: CheckKind,
        pub file: String,
        /// 1-based line number.
        pub line: u32,
        pub severity: Severity,
        pub message: String,
    }

    impl Finding {
        pub fn new(
            check: CheckKind,
            file: impl Into<String>,
            line: u32,
            severity: Severity,
            message: impl Into<String>,
        ) -> Self {
            Self {
                check,
                file: file.into(),
                line,
                severity,
                message: message.into(),
            }
        }
    }
}

pub mod gate {
    //! The verification gate: the authority rule, encoded (Bible Book IX, sec 28-29).
    //!
    //! The gate reconciles the verdicts from every tier into one decision. Its
    //! defining rule, from which everything else follows: a probabilistic review
    //! (Tier4) may NEVER overrule a failing deterministic gate. This is not advice;
    //! it is enforced by [`apply_gate`], which returns [`GateDecision::Reject`] on
    //! any deterministic failure regardless of what the review says.

    use serde::{Deserialize, Serialize};

    use crate::verify_plane::oracle::Verdict;
    use crate::verify_plane::tier::VerificationTier;

    /// A verdict tagged with the tier and oracle that produced it. The gate consumes
    /// a slice of these.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct TieredVerdict {
        pub tier: VerificationTier,
        pub oracle: String,
        pub verdict: Verdict,
    }

    impl TieredVerdict {
        pub fn new(tier: VerificationTier, oracle: impl Into<String>, verdict: Verdict) -> Self {
            Self {
                tier,
                oracle: oracle.into(),
                verdict,
            }
        }
    }

    /// The gate's decision.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(tag = "decision", rename_all = "snake_case")]
    pub enum GateDecision {
        Accept,
        Reject {
            reasons: Vec<String>,
        },
        /// Nothing decisive ran (no deterministic pass and no blocking failure), so
        /// the gate cannot accept on faith.
        Inconclusive,
    }

    /// Reconcile tier verdicts into one decision, honoring the authority rule.
    ///
    /// Order of authority:
    /// 1. Any DETERMINISTIC failure forces [`GateDecision::Reject`], unconditionally.
    ///    A probabilistic review Pass cannot rescue it.
    /// 2. If no deterministic verdict passed, the gate is [`GateDecision::Inconclusive`]:
    ///    a review alone can never carry a change to Accept.
    /// 3. With the deterministic gate passed, a review failure still blocks (Reject),
    ///    and only then does a clean review yield [`GateDecision::Accept`].
    pub fn apply_gate(verdicts: &[TieredVerdict]) -> GateDecision {
        let det_fail: Vec<String> = verdicts
            .iter()
            .filter(|v| v.tier.is_deterministic())
            .filter_map(reason_if_fail)
            .collect();
        if !det_fail.is_empty() {
            return GateDecision::Reject { reasons: det_fail };
        }

        let any_det_pass = verdicts
            .iter()
            .any(|v| v.tier.is_deterministic() && v.verdict.is_pass());
        if !any_det_pass {
            return GateDecision::Inconclusive;
        }

        let review_fail: Vec<String> = verdicts
            .iter()
            .filter(|v| v.tier.is_probabilistic())
            .filter_map(reason_if_fail)
            .collect();
        if !review_fail.is_empty() {
            return GateDecision::Reject {
                reasons: review_fail,
            };
        }

        GateDecision::Accept
    }

    fn reason_if_fail(v: &TieredVerdict) -> Option<String> {
        match &v.verdict {
            Verdict::Fail { reasons } => Some(format!("{}: {}", v.oracle, reasons.join("; "))),
            _ => None,
        }
    }

    /// The authority invariant, encoded as a value so it can be asserted directly: a
    /// probabilistic review can never override a deterministic verdict. Always
    /// `false`. See [`apply_gate`], which enforces it.
    pub const fn probabilistic_can_override_deterministic() -> bool {
        false
    }
}

pub mod oracle {
    //! The oracle interface (Bible Book IX, sec 28).
    //!
    //! An [`Oracle`] checks a candidate and returns a [`Verdict`] plus [`Evidence`].
    //! The verdict is one of `Pass`, `Fail { reasons }`, or `Skipped { why }`. Every
    //! oracle declares its [`VerificationTier`] and its [`OracleClass`]
    //! (Deterministic vs Probabilistic) so the gate can honor the authority rule:
    //! deterministic verdicts outrank probabilistic ones and are never overridden by
    //! them.

    use std::path::PathBuf;

    use serde::{Deserialize, Serialize};

    use crate::verify_plane::finding::Finding;
    use crate::verify_plane::tier::VerificationTier;

    /// The outcome of a single oracle check.
    ///
    /// The three shapes carry exactly what a repair loop needs: nothing on `Pass`,
    /// the specific `reasons` on `Fail`, and the `why` on `Skipped` (so a skipped
    /// gate is auditable and never silently treated as a pass).
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(tag = "status", rename_all = "snake_case")]
    pub enum Verdict {
        Pass,
        Fail { reasons: Vec<String> },
        Skipped { why: String },
    }

    impl Verdict {
        pub fn is_pass(&self) -> bool {
            matches!(self, Verdict::Pass)
        }

        pub fn is_fail(&self) -> bool {
            matches!(self, Verdict::Fail { .. })
        }

        pub fn is_skipped(&self) -> bool {
            matches!(self, Verdict::Skipped { .. })
        }

        /// The failure reasons, or an empty slice for non-failing verdicts.
        pub fn reasons(&self) -> &[String] {
            match self {
                Verdict::Fail { reasons } => reasons,
                _ => &[],
            }
        }
    }

    /// Whether an oracle's verdicts are reproducible facts or probabilistic
    /// judgments. The gate ranks [`OracleClass::Deterministic`] strictly above
    /// [`OracleClass::Probabilistic`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum OracleClass {
        Deterministic,
        Probabilistic,
    }

    /// Structured evidence attached to a verdict. Findings are the machine-readable
    /// core; notes carry free-form context (for example, a directory that could not
    /// be read during a scan).
    #[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Evidence {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub findings: Vec<Finding>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub notes: Vec<String>,
    }

    /// A verdict together with the evidence that produced it.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct OracleOutcome {
        pub verdict: Verdict,
        pub evidence: Evidence,
    }

    /// A single source file to analyze, given directly as text (no filesystem read).
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct SourceFile {
        pub path: String,
        pub text: String,
    }

    impl SourceFile {
        pub fn new(path: impl Into<String>, text: impl Into<String>) -> Self {
            Self {
                path: path.into(),
                text: text.into(),
            }
        }
    }

    /// What an oracle checks against: in-memory source files, an optional directory
    /// root to walk, and the set of files the candidate changed (so scope-aware
    /// oracles can narrow themselves).
    #[derive(Debug, Clone, Default)]
    pub struct VerificationInput {
        pub sources: Vec<SourceFile>,
        pub root: Option<PathBuf>,
        pub changed_files: Vec<String>,
    }

    impl VerificationInput {
        /// An input over a set of in-memory source files.
        pub fn from_sources(sources: Vec<SourceFile>) -> Self {
            Self {
                sources,
                root: None,
                changed_files: Vec::new(),
            }
        }

        /// An input that walks a directory root.
        pub fn from_root(root: impl Into<PathBuf>) -> Self {
            Self {
                sources: Vec::new(),
                root: Some(root.into()),
                changed_files: Vec::new(),
            }
        }
    }

    /// The verifier interface. `name` identifies the oracle; `tier` and `class`
    /// describe where it sits in the plane; `evaluate` runs it (deterministic, pure
    /// with respect to its input) and returns an [`OracleOutcome`].
    ///
    /// This trait is synchronous and model-free by construction. A probabilistic
    /// (Tier4) reviewer would need a model to implement `evaluate`, which is
    /// DEFERRED_MODEL_REQUIRED; this crate provides no such implementation.
    pub trait Oracle {
        fn name(&self) -> &str;

        fn tier(&self) -> VerificationTier;

        fn class(&self) -> OracleClass {
            OracleClass::Deterministic
        }

        fn evaluate(&self, input: &VerificationInput) -> OracleOutcome;
    }
}

pub mod receipt {
    //! The verification receipt (Bible Book IX, sec 29).
    //!
    //! Every gate run emits a [`VerificationReceipt`]: a stable, serde-serializable
    //! record of what was checked, over what scope, against what source, and what
    //! the verdict was. Receipts are the durable evidence trail and the input to the
    //! re-review dependency model (see [`crate::verify_plane::rereview`]).

    use serde::{Deserialize, Serialize};

    use crate::verify_plane::oracle::Verdict;
    use crate::verify_plane::tier::VerificationTier;

    /// A durable record of one oracle run.
    ///
    /// The serde shape is intentionally fixed and every field is always present
    /// (including `command: null` when there was no command), so a stored receipt
    /// parses identically across versions.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct VerificationReceipt {
        pub verification_id: String,
        pub tier: VerificationTier,
        pub oracle: String,
        /// The command that was run, if this oracle ran one (build, test). `None`
        /// for in-process oracles such as static analysis.
        #[serde(default)]
        pub command: Option<String>,
        /// The file paths this receipt covers. Drives re-review invalidation: a
        /// change intersecting this scope invalidates the receipt.
        pub scope: Vec<String>,
        /// Content hash of the source the verdict was computed against, so a receipt
        /// can be tied to an exact snapshot.
        pub source_hash: String,
        pub verdict: Verdict,
        pub started_ms: u64,
        pub duration_ms: u64,
    }

    impl VerificationReceipt {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            verification_id: impl Into<String>,
            tier: VerificationTier,
            oracle: impl Into<String>,
            command: Option<String>,
            scope: Vec<String>,
            source_hash: impl Into<String>,
            verdict: Verdict,
            started_ms: u64,
            duration_ms: u64,
        ) -> Self {
            Self {
                verification_id: verification_id.into(),
                tier,
                oracle: oracle.into(),
                command,
                scope,
                source_hash: source_hash.into(),
                verdict,
                started_ms,
                duration_ms,
            }
        }

        /// Serialize to canonical JSON.
        pub fn to_json(&self) -> serde_json::Result<String> {
            serde_json::to_string(self)
        }

        /// Parse from JSON.
        pub fn from_json(s: &str) -> serde_json::Result<Self> {
            serde_json::from_str(s)
        }
    }

    /// A stable content hash for source bytes (blake3, hex-encoded). Used to fill a
    /// receipt's `source_hash` and to tie a verdict to an exact snapshot.
    pub fn source_hash(bytes: &[u8]) -> String {
        blake3::hash(bytes).to_hex().to_string()
    }

    /// A deterministic hash over a set of `(path, text)` sources: each entry is
    /// folded in path-then-text order after sorting by path, so the same set of
    /// sources always yields the same hash regardless of input ordering.
    pub fn source_hash_of<I, P, T>(sources: I) -> String
    where
        I: IntoIterator<Item = (P, T)>,
        P: AsRef<str>,
        T: AsRef<str>,
    {
        let mut entries: Vec<(String, String)> = sources
            .into_iter()
            .map(|(p, t)| (p.as_ref().to_string(), t.as_ref().to_string()))
            .collect();
        entries.sort();
        let mut hasher = blake3::Hasher::new();
        for (path, text) in entries {
            hasher.update(path.as_bytes());
            hasher.update(&[0u8]);
            hasher.update(text.as_bytes());
            hasher.update(&[0u8]);
        }
        hasher.finalize().to_hex().to_string()
    }
}

pub mod rereview {
    //! The re-review dependency model (Bible Book IX, sec 29).
    //!
    //! A verification receipt is only valid as long as the source it covers has not
    //! changed. Given a set of prior receipts and the set of file paths a new change
    //! touched, this module returns exactly the receipts whose scope INTERSECTS the
    //! change: those are invalidated and must be re-run. Receipts whose scope is
    //! disjoint from the change stay valid and are not re-run.

    use crate::verify_plane::receipt::VerificationReceipt;

    /// Normalize a path for comparison: strip a leading `./` and drop a trailing `/`
    /// so directory and file spellings compare cleanly.
    fn norm(p: &str) -> &str {
        let p = p.strip_prefix("./").unwrap_or(p);
        p.strip_suffix('/').unwrap_or(p)
    }

    /// True if two paths refer to the same file, or one is a directory that contains
    /// the other. So a receipt scoping `crates/a/src` is invalidated by a change to
    /// `crates/a/src/lib.rs`, and vice versa, but not by a change to `crates/ab`.
    pub fn paths_intersect(a: &str, b: &str) -> bool {
        let a = norm(a);
        let b = norm(b);
        if a == b {
            return true;
        }
        let under = |child: &str, parent: &str| {
            child.starts_with(parent) && child.as_bytes().get(parent.len()) == Some(&b'/')
        };
        under(a, b) || under(b, a)
    }

    /// The receipts whose scope intersects ANY changed path (they must be re-run),
    /// in the order they appear in `receipts`. Receipts with no intersecting scope
    /// are omitted.
    pub fn invalidated_receipts<'a>(
        receipts: &'a [VerificationReceipt],
        changed: &[String],
    ) -> Vec<&'a VerificationReceipt> {
        receipts
            .iter()
            .filter(|r| {
                r.scope
                    .iter()
                    .any(|s| changed.iter().any(|c| paths_intersect(s, c)))
            })
            .collect()
    }

    /// The `verification_id`s of the invalidated receipts. A convenience over
    /// [`invalidated_receipts`].
    pub fn invalidated_ids(receipts: &[VerificationReceipt], changed: &[String]) -> Vec<String> {
        invalidated_receipts(receipts, changed)
            .into_iter()
            .map(|r| r.verification_id.clone())
            .collect()
    }
}

pub mod review {
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
    //! returns a [`ReviewRoleProfile`] (data), never a [`crate::verify_plane::oracle::Verdict`],
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
}

pub mod static_analysis {
    //! A real, deterministic static-analysis oracle over Rust source text.
    //!
    //! This is a genuine Tier1 deterministic check: it runs entirely in-process over
    //! source strings (or a walked directory), with NO model and NO subprocess, and
    //! the same input always yields the same findings. It is a lint, not a compiler:
    //! it works over a light lexical model of the source (comments and string
    //! literals are masked so braces and identifiers inside them do not count).
    //!
    //! Checks:
    //! * `unwrap()` / `expect()` used outside `#[cfg(test)]` / `#[test]` code.
    //! * `panic!` / `todo!` / `unimplemented!` / `unreachable!` marker macros.
    //! * en dash (U+2013) / em dash (U+2014) presence: the house-rule lint.
    //! * a very-long-function heuristic (body line count over a threshold).
    //! * `TODO` / `FIXME` / `XXX` markers.

    use std::path::Path;

    use regex::Regex;
    use walkdir::WalkDir;

    use crate::verify_plane::error::{Result, VerifyError};
    use crate::verify_plane::finding::{CheckKind, Finding, Severity};
    use crate::verify_plane::oracle::{
        Evidence, Oracle, OracleClass, OracleOutcome, SourceFile, Verdict, VerificationInput,
    };
    use crate::verify_plane::tier::VerificationTier;

    /// En dash. Referenced by codepoint so this source file never contains the
    /// banned character itself.
    const EN_DASH: char = '\u{2013}';
    /// Em dash. Referenced by codepoint for the same reason.
    const EM_DASH: char = '\u{2014}';

    /// Default line-count threshold above which a function body is flagged as long.
    pub const DEFAULT_LONG_FUNCTION_THRESHOLD: usize = 80;

    /// The deterministic static-analysis oracle (Tier1, Deterministic class).
    pub struct StaticAnalysisOracle {
        long_function_threshold: usize,
        unwrap_re: Regex,
        macro_re: Regex,
        todo_re: Regex,
    }

    impl Default for StaticAnalysisOracle {
        fn default() -> Self {
            Self {
                long_function_threshold: DEFAULT_LONG_FUNCTION_THRESHOLD,
                // A method call: a dot, optional space, then `unwrap`/`expect`, then
                // an open paren. `\b` keeps `unwrap_or`, `expect_err`, etc. clear.
                unwrap_re: Regex::new(r"\.\s*(unwrap|expect)\b\s*\(").expect("static unwrap regex"),
                // A marker macro invocation: name immediately followed by `!`.
                macro_re: Regex::new(r"\b(panic|todo|unimplemented|unreachable)\s*!")
                    .expect("static macro regex"),
                // Uppercase-only so the lowercase `todo!` macro is not double-counted.
                todo_re: Regex::new(r"\b(TODO|FIXME|XXX)\b").expect("static todo regex"),
            }
        }
    }

    impl StaticAnalysisOracle {
        pub fn new() -> Self {
            Self::default()
        }

        /// An oracle with a custom long-function threshold.
        pub fn with_long_function_threshold(threshold: usize) -> Self {
            Self {
                long_function_threshold: threshold,
                ..Self::default()
            }
        }

        pub fn long_function_threshold(&self) -> usize {
            self.long_function_threshold
        }

        /// Analyze a single source file, returning findings sorted by (line, check).
        pub fn analyze_source(&self, file: &str, source: &str) -> Vec<Finding> {
            let mut out = Vec::new();

            // 1. En/em dash over the RAW text: the house rule bans them everywhere,
            //    including inside comments and strings.
            self.scan_dashes(file, source, &mut out);

            // 2. TODO/FIXME markers over the RAW text (they usually live in comments).
            self.scan_todo(file, source, &mut out);

            // 3. Structural checks over a comment- and string-masked view: unwrap /
            //    expect outside test code, marker macros, and long functions. The
            //    mask also yields, per line, whether we are inside test code.
            let masked = mask_comments_and_strings(source);
            let masked_lines: Vec<&str> = masked.lines().collect();
            let in_test = self.scan_structure(file, &masked_lines, &mut out);
            self.scan_unwrap(file, &masked_lines, &in_test, &mut out);
            self.scan_macros(file, &masked_lines, &mut out);

            out.sort_by(|a, b| {
                a.line
                    .cmp(&b.line)
                    .then(a.check.cmp(&b.check))
                    .then(a.message.cmp(&b.message))
            });
            out
        }

        /// Analyze a set of in-memory source files.
        pub fn analyze_sources(&self, sources: &[SourceFile]) -> Vec<Finding> {
            sources
                .iter()
                .flat_map(|s| self.analyze_source(&s.path, &s.text))
                .collect()
        }

        /// Walk `root`, analyze every `*.rs` file, and return the combined findings.
        /// Deterministic: entries are visited in sorted order.
        pub fn analyze_dir(&self, root: &Path) -> Result<Vec<Finding>> {
            let mut out = Vec::new();
            for entry in WalkDir::new(root).sort_by_file_name() {
                let entry = entry.map_err(|e| VerifyError::Walk {
                    root: root.display().to_string(),
                    message: e.to_string(),
                })?;
                if !entry.file_type().is_file() {
                    continue;
                }
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                    continue;
                }
                let text = std::fs::read_to_string(path).map_err(|e| VerifyError::Read {
                    path: path.display().to_string(),
                    message: e.to_string(),
                })?;
                out.extend(self.analyze_source(&path.display().to_string(), &text));
            }
            Ok(out)
        }

        fn scan_dashes(&self, file: &str, source: &str, out: &mut Vec<Finding>) {
            for (idx, line) in source.lines().enumerate() {
                for ch in line.chars() {
                    if ch == EN_DASH || ch == EM_DASH {
                        let which = if ch == EN_DASH { "en" } else { "em" };
                        out.push(Finding::new(
                            CheckKind::EmDash,
                            file,
                            (idx + 1) as u32,
                            Severity::Error,
                            format!(
                            "house-rule violation: {which} dash (U+{:04X}) is banned; use a hyphen",
                            ch as u32
                        ),
                        ));
                    }
                }
            }
        }

        fn scan_todo(&self, file: &str, source: &str, out: &mut Vec<Finding>) {
            for (idx, line) in source.lines().enumerate() {
                for m in self.todo_re.find_iter(line) {
                    out.push(Finding::new(
                        CheckKind::TodoMarker,
                        file,
                        (idx + 1) as u32,
                        Severity::Info,
                        format!("`{}` marker found", m.as_str()),
                    ));
                }
            }
        }

        fn scan_unwrap(
            &self,
            file: &str,
            lines: &[&str],
            in_test: &[bool],
            out: &mut Vec<Finding>,
        ) {
            for (idx, line) in lines.iter().enumerate() {
                if in_test.get(idx).copied().unwrap_or(false) {
                    continue;
                }
                for caps in self.unwrap_re.captures_iter(line) {
                    let which = &caps[1];
                    out.push(Finding::new(
                        CheckKind::UnwrapOutsideTest,
                        file,
                        (idx + 1) as u32,
                        Severity::Warning,
                        format!("`{which}()` used outside test code; handle the error explicitly"),
                    ));
                }
            }
        }

        fn scan_macros(&self, file: &str, lines: &[&str], out: &mut Vec<Finding>) {
            for (idx, line) in lines.iter().enumerate() {
                for caps in self.macro_re.captures_iter(line) {
                    let name = &caps[1];
                    let severity = match name {
                        "todo" | "unimplemented" => Severity::Error,
                        _ => Severity::Warning,
                    };
                    out.push(Finding::new(
                        CheckKind::PanicMarker,
                        file,
                        (idx + 1) as u32,
                        severity,
                        format!("`{name}!` marker macro"),
                    ));
                }
            }
        }

        /// Single structural pass over masked lines. Tracks brace depth to (a) decide
        /// which lines are inside `#[cfg(test)]` / `#[test]` code and (b) measure
        /// function body lengths. Long-function findings are pushed to `out`; the
        /// per-line "in test" vector is returned for the unwrap check.
        fn scan_structure(&self, file: &str, lines: &[&str], out: &mut Vec<Finding>) -> Vec<bool> {
            let mut in_test = vec![false; lines.len()];

            let mut depth: i32 = 0;
            // Brace levels at which an active test region was opened.
            let mut test_stack: Vec<i32> = Vec::new();
            // A test attribute was seen and is waiting for the block it guards.
            let mut armed_test = false;
            // A `fn` token was seen and is waiting for its body's opening brace.
            let mut armed_fn_line: Option<u32> = None;
            // Open function bodies: (signature line, brace level of the enclosing block).
            let mut fn_stack: Vec<(u32, i32)> = Vec::new();

            for (idx, line) in lines.iter().enumerate() {
                in_test[idx] = !test_stack.is_empty();

                if line.contains("#[test]")
                    || line.contains("#[tokio::test]")
                    || line.contains("#[cfg(test)]")
                {
                    armed_test = true;
                }

                let mut ident = String::new();
                for c in line.chars() {
                    if c.is_alphanumeric() || c == '_' {
                        ident.push(c);
                        continue;
                    }
                    if ident == "fn" && armed_fn_line.is_none() {
                        armed_fn_line = Some((idx + 1) as u32);
                    }
                    ident.clear();

                    match c {
                        '{' => {
                            let level = depth;
                            depth += 1;
                            if armed_test {
                                test_stack.push(level);
                                armed_test = false;
                            }
                            if let Some(sig) = armed_fn_line.take() {
                                fn_stack.push((sig, level));
                            }
                        }
                        '}' => {
                            depth -= 1;
                            if let Some(&(sig, level)) = fn_stack.last() {
                                if depth == level {
                                    fn_stack.pop();
                                    let body_len = (idx as i64 + 1) - sig as i64;
                                    if body_len > self.long_function_threshold as i64 {
                                        out.push(Finding::new(
                                            CheckKind::LongFunction,
                                            file,
                                            sig,
                                            Severity::Warning,
                                            format!(
                                            "function body spans {body_len} lines (threshold {}); \
                                             consider splitting it",
                                            self.long_function_threshold
                                        ),
                                        ));
                                    }
                                }
                            }
                            if let Some(&level) = test_stack.last() {
                                if depth == level {
                                    test_stack.pop();
                                }
                            }
                        }
                        ';' => {
                            // A `fn foo();` declaration or `#[cfg(test)] mod m;`
                            // ended without opening a block: disarm.
                            armed_fn_line = None;
                            armed_test = false;
                        }
                        _ => {}
                    }
                }
                // A trailing `fn` at end of line (no delimiter after it).
                if ident == "fn" && armed_fn_line.is_none() {
                    armed_fn_line = Some((idx + 1) as u32);
                }
            }

            in_test
        }
    }

    impl Oracle for StaticAnalysisOracle {
        fn name(&self) -> &str {
            "static_analysis"
        }

        fn tier(&self) -> VerificationTier {
            VerificationTier::Tier1Deterministic
        }

        fn class(&self) -> OracleClass {
            OracleClass::Deterministic
        }

        fn evaluate(&self, input: &VerificationInput) -> OracleOutcome {
            let mut findings = self.analyze_sources(&input.sources);
            let mut notes = Vec::new();

            if let Some(root) = &input.root {
                match self.analyze_dir(root) {
                    Ok(dir_findings) => findings.extend(dir_findings),
                    Err(e) => notes.push(format!("directory scan skipped: {e}")),
                }
            }

            findings.sort_by(|a, b| {
                a.file
                    .cmp(&b.file)
                    .then(a.line.cmp(&b.line))
                    .then(a.check.cmp(&b.check))
                    .then(a.message.cmp(&b.message))
            });

            // A finding at or above Warning severity fails the gate. Info-only
            // findings (bare TODO markers) do not, but are still reported.
            let blocking: Vec<String> = findings
                .iter()
                .filter(|f| f.severity >= Severity::Warning)
                .map(|f| format!("{}:{} {}", f.file, f.line, f.message))
                .collect();

            let verdict = if blocking.is_empty() {
                Verdict::Pass
            } else {
                Verdict::Fail { reasons: blocking }
            };

            OracleOutcome {
                verdict,
                evidence: Evidence { findings, notes },
            }
        }
    }

    /// Return a copy of `source` with line comments, block comments, string
    /// literals, and char literals replaced by spaces, while preserving every
    /// newline so line numbers are unchanged. Lifetimes (`'a`) are left intact so
    /// they are not mistaken for char literals.
    ///
    /// This is a lexical approximation, not a Rust parser: raw strings with embedded
    /// quotes are not handled specially. That is acceptable for a lint whose job is
    /// to keep braces and identifiers inside strings and comments from skewing the
    /// structural pass.
    fn mask_comments_and_strings(source: &str) -> String {
        let chars: Vec<char> = source.chars().collect();
        let n = chars.len();
        let mut out = String::with_capacity(source.len());
        let mut i = 0;

        while i < n {
            let c = chars[i];

            // Line comment: to end of line.
            if c == '/' && i + 1 < n && chars[i + 1] == '/' {
                while i < n && chars[i] != '\n' {
                    out.push(' ');
                    i += 1;
                }
                continue;
            }

            // Block comment: to the closing `*/` (may span lines).
            if c == '/' && i + 1 < n && chars[i + 1] == '*' {
                out.push(' ');
                out.push(' ');
                i += 2;
                while i < n && !(chars[i] == '*' && i + 1 < n && chars[i + 1] == '/') {
                    out.push(if chars[i] == '\n' { '\n' } else { ' ' });
                    i += 1;
                }
                if i < n {
                    out.push(' ');
                    out.push(' ');
                    i += 2;
                }
                continue;
            }

            // String literal.
            if c == '"' {
                out.push(' ');
                i += 1;
                while i < n {
                    if chars[i] == '\\' {
                        out.push(' ');
                        i += 1;
                        if i < n {
                            out.push(if chars[i] == '\n' { '\n' } else { ' ' });
                            i += 1;
                        }
                        continue;
                    }
                    if chars[i] == '"' {
                        out.push(' ');
                        i += 1;
                        break;
                    }
                    out.push(if chars[i] == '\n' { '\n' } else { ' ' });
                    i += 1;
                }
                continue;
            }

            // Char literal vs lifetime. `'\...` or `'x'` is a char literal; `'a`
            // followed by an identifier that does not immediately close is a lifetime.
            if c == '\'' {
                let is_char_lit =
                    (i + 1 < n && chars[i + 1] == '\\') || (i + 2 < n && chars[i + 2] == '\'');
                if is_char_lit {
                    out.push(' ');
                    i += 1;
                    while i < n {
                        if chars[i] == '\\' {
                            out.push(' ');
                            i += 1;
                            if i < n {
                                out.push(' ');
                                i += 1;
                            }
                            continue;
                        }
                        if chars[i] == '\'' {
                            out.push(' ');
                            i += 1;
                            break;
                        }
                        out.push(' ');
                        i += 1;
                    }
                    continue;
                }
                // Lifetime tick: keep it as ordinary punctuation.
                out.push(c);
                i += 1;
                continue;
            }

            out.push(c);
            i += 1;
        }

        out
    }
}

pub mod tier {
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
    //! [`crate::verify_plane::gate::apply_gate`].

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
    ///   only the review-role profiles (see [`crate::verify_plane::review`]).
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
}

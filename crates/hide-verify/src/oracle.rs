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

use crate::finding::Finding;
use crate::tier::VerificationTier;

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

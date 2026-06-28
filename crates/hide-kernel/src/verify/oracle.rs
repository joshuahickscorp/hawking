//! The verifier interface (bible ch.02 Appendix A.2).
//!
//! An [`Oracle`] checks a candidate against a step's acceptance contract and
//! returns a [`Verdict`]. The defining rule: a **Deterministic** verdict is
//! authoritative; a **Probabilistic** score only ranks *within* the
//! deterministic-pass set and never overrides `build`/`test` (§3.2 / §4.8.4).

use futures::future::BoxFuture;
use hide_core::Result;
use serde::{Deserialize, Serialize};

/// The execution environment an oracle checks against: a workspace root and the
/// set of files the step changed (so an oracle can scope itself).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationInput {
    pub step_id: Option<String>,
    pub workspace_root: String,
    pub changed_files: Vec<String>,
    /// Optional test selectors propagated from the step's `acceptance.tests`.
    #[serde(default)]
    pub tests: Vec<String>,
    /// The candidate's raw output (model text / diff), for probabilistic oracles.
    #[serde(default)]
    pub candidate_output: String,
}

impl VerificationInput {
    pub fn new(workspace_root: impl Into<String>) -> Self {
        Self {
            step_id: None,
            workspace_root: workspace_root.into(),
            changed_files: Vec::new(),
            tests: Vec::new(),
            candidate_output: String::new(),
        }
    }
}

/// Oracle class (A.2). The gate ranks Deterministic strictly over Probabilistic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OracleClass {
    Deterministic,
    Probabilistic,
}

/// Relative cost hint (A.2 `cost_hint`) so the gate can run cheap oracles first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cost {
    Cheap,
    Medium,
    Expensive,
}

/// A structured oracle failure (A.2 `Failure`) — the minimal-repair context. The
/// repair stage feeds these (file/line/code/message) back verbatim so the model
/// fixes the *specific* error, not the whole history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Failure {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// e.g. `"type"`, `"test"`, `"lint"`, `"patch"`.
    pub category: String,
    pub message: String,
}

impl Failure {
    pub fn new(category: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            file: None,
            line: None,
            code: None,
            category: category.into(),
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    pub status: VerdictStatus,
    /// Probabilistic only ∈ [0,1]; for deterministic oracles, 1.0 on Pass / 0.0
    /// on Fail. Never overrides a Deterministic verdict.
    pub score: f32,
    pub oracle: String,
    /// Which class produced this verdict (drives gate ranking).
    #[serde(default = "default_class")]
    pub class: OracleClass,
    pub detail: String,
    /// Structured failures (empty on Pass). Minimal-repair context (§4.7).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failures: Vec<Failure>,
    /// Content-addressed artifact refs (logs/diffs) — blob hashes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<String>,
    #[serde(default)]
    pub duration_ms: u64,
}

fn default_class() -> OracleClass {
    OracleClass::Deterministic
}

impl Verdict {
    pub fn pass(oracle: impl Into<String>, class: OracleClass, detail: impl Into<String>) -> Self {
        Self {
            status: VerdictStatus::Pass,
            score: 1.0,
            oracle: oracle.into(),
            class,
            detail: detail.into(),
            failures: Vec::new(),
            artifacts: Vec::new(),
            duration_ms: 0,
        }
    }

    pub fn fail(
        oracle: impl Into<String>,
        class: OracleClass,
        detail: impl Into<String>,
        failures: Vec<Failure>,
    ) -> Self {
        Self {
            status: VerdictStatus::Fail,
            score: 0.0,
            oracle: oracle.into(),
            class,
            detail: detail.into(),
            failures,
            artifacts: Vec::new(),
            duration_ms: 0,
        }
    }

    pub fn is_deterministic(&self) -> bool {
        self.class == OracleClass::Deterministic
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerdictStatus {
    Pass,
    Fail,
    Inconclusive,
    Skipped,
}

/// The verifier interface (A.2). `id`/`class`/`cost_hint` describe the oracle;
/// `verify` runs it (sandboxed, pure w.r.t. the snapshot) and returns a verdict.
pub trait Oracle: Send + Sync {
    fn name(&self) -> &str;

    /// Deterministic vs Probabilistic — drives the gate's authority ranking.
    fn class(&self) -> OracleClass {
        OracleClass::Deterministic
    }

    /// Relative cost so the gate can order cheap-before-expensive.
    fn cost_hint(&self) -> Cost {
        Cost::Medium
    }

    fn verify<'a>(&'a self, input: &'a VerificationInput) -> BoxFuture<'a, Result<Verdict>>;
}

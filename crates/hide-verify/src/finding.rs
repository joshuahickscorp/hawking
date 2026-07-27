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

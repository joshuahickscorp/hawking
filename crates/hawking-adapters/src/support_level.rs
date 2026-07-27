//! Honest support grades. Never inflate.
//!
//! Exactly these seven grades — no others. Promotion requires the evidence
//! the grade names. No family becomes PRODUCTION because shapes look familiar.

use serde::{Deserialize, Serialize};

/// Ladder of evidence for a model family. Literal meanings; do not promote
/// from a code reading alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportLevel {
    /// Family described; nothing parsed, nothing executes.
    Declared,
    /// Real official config / tokenizer / safetensors header parsed and mapped.
    SourceHeaderValidated,
    /// Matches a deterministic reference on a synthetic twin.
    SyntheticParity,
    /// At least one real tensor decoded from a real checkpoint.
    RealTensorDecode,
    /// A real small checkpoint of the family runs end to end.
    SmallRealCheckpoint,
    /// A real full-size parent validated.
    FullParentValidated,
    /// Served, under test, with a standing parity receipt.
    /// **No family is at this level today.**
    Production,
}

impl SupportLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            SupportLevel::Declared => "DECLARED",
            SupportLevel::SourceHeaderValidated => "SOURCE_HEADER_VALIDATED",
            SupportLevel::SyntheticParity => "SYNTHETIC_PARITY",
            SupportLevel::RealTensorDecode => "REAL_TENSOR_DECODE",
            SupportLevel::SmallRealCheckpoint => "SMALL_REAL_CHECKPOINT",
            SupportLevel::FullParentValidated => "FULL_PARENT_VALIDATED",
            SupportLevel::Production => "PRODUCTION",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "DECLARED" => Some(SupportLevel::Declared),
            "SOURCE_HEADER_VALIDATED" => Some(SupportLevel::SourceHeaderValidated),
            "SYNTHETIC_PARITY" => Some(SupportLevel::SyntheticParity),
            "REAL_TENSOR_DECODE" => Some(SupportLevel::RealTensorDecode),
            "SMALL_REAL_CHECKPOINT" => Some(SupportLevel::SmallRealCheckpoint),
            "FULL_PARENT_VALIDATED" => Some(SupportLevel::FullParentValidated),
            "PRODUCTION" => Some(SupportLevel::Production),
            _ => None,
        }
    }

    /// Stable ladder order (lowest → highest).
    pub fn all() -> &'static [SupportLevel] {
        &[
            SupportLevel::Declared,
            SupportLevel::SourceHeaderValidated,
            SupportLevel::SyntheticParity,
            SupportLevel::RealTensorDecode,
            SupportLevel::SmallRealCheckpoint,
            SupportLevel::FullParentValidated,
            SupportLevel::Production,
        ]
    }
}

//! The verification receipt (Bible Book IX, sec 29).
//!
//! Every gate run emits a [`VerificationReceipt`]: a stable, serde-serializable
//! record of what was checked, over what scope, against what source, and what
//! the verdict was. Receipts are the durable evidence trail and the input to the
//! re-review dependency model (see [`crate::rereview`]).

use serde::{Deserialize, Serialize};

use crate::oracle::Verdict;
use crate::tier::VerificationTier;

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

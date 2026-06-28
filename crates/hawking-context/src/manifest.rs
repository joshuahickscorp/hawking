//! The context manifest — the single source of truth for what the model saw,
//! why, and what was dropped (bible §4.3, Appendix A.1).
//!
//! It is the UI's "context stack", the agent loop's replay substrate, and a
//! versioned public contract. Span ids are **blake3 content addresses** so two
//! turns that include the same span share an id (dedup + replay key).

use hide_core::ids::{now_ms, EventId, TimestampMs};
use hide_core::types::{BlobRef, Provenance};
use serde::{Deserialize, Serialize};

/// Schema version of the manifest contract (A.1). Additive = minor bump.
pub const MANIFEST_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextManifest {
    pub schema_version: u16,
    pub created_at_ms: TimestampMs,
    pub source_event: Option<EventId>,
    /// Turn / session identifiers (A.1). Optional so callers that don't track
    /// a session still get a valid manifest.
    #[serde(default)]
    pub turn_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    /// The profile block (name + effective window + policy summary).
    #[serde(default)]
    pub profile: Option<ManifestProfile>,
    /// The model block (id, arch, native/effective ctx, tokenizer signature).
    #[serde(default)]
    pub model: Option<ManifestModel>,
    /// Budget accounting (total / used / free / reservations).
    #[serde(default)]
    pub budget: Option<ManifestBudget>,
    pub model_context_tokens: usize,
    pub used_tokens: usize,
    /// Retained spans, in final window order (head→tail).
    pub retained: Vec<ContextSpan>,
    /// Candidates that did not make it in, with a reason and would-be cost.
    pub dropped: Vec<DroppedContextSpan>,
    /// Surfaced contradictions for user resolution (F12/F4).
    #[serde(default)]
    pub conflicts: Vec<ManifestConflict>,
    /// KV-reuse accounting (§4.5).
    #[serde(default)]
    pub kv: ManifestKv,
    /// Compaction events: a span was replaced by a shorter rendering.
    #[serde(default)]
    pub compaction_events: Vec<CompactionEvent>,
}

impl ContextManifest {
    pub fn new(model_context_tokens: usize) -> Self {
        Self {
            schema_version: MANIFEST_SCHEMA_VERSION,
            created_at_ms: now_ms(),
            source_event: None,
            turn_id: None,
            session_id: None,
            profile: None,
            model: None,
            budget: None,
            model_context_tokens,
            used_tokens: 0,
            retained: Vec::new(),
            dropped: Vec::new(),
            conflicts: Vec::new(),
            kv: ManifestKv::default(),
            compaction_events: Vec::new(),
        }
    }
}

/// The profile summary recorded on a manifest (A.1 `profile` block).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestProfile {
    pub name: String,
    pub target_ctx_tokens: usize,
    pub position_policy: String,
    pub working_set_mode: String,
    pub kv_precision: String,
}

/// The model summary recorded on a manifest (A.1 `model` block).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestModel {
    pub id: String,
    pub arch: String,
    pub ctx_len_native: usize,
    pub ctx_len_effective: usize,
    pub tokenizer_sig: String,
}

/// Budget accounting (A.1 `budget` block).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestBudget {
    pub total: usize,
    pub used: usize,
    pub free: usize,
    pub reservation_system: usize,
    pub reservation_response: usize,
    pub reservation_scratchpad: usize,
}

/// Where a span sits in the pin lattice (A.1 `pin`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PinState {
    /// Always present, never evicted (system rails, safety rules).
    NeverEvict,
    /// Floated to the top by the user.
    UserPinned,
    /// Competes on value like everything else.
    Normal,
}

impl Default for PinState {
    fn default() -> Self {
        PinState::Normal
    }
}

/// The four scoring signals attached to every retained span (A.1 `signals`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct SpanSignals {
    pub recency: f32,
    pub importance: f32,
    pub relevance: f32,
    pub redundancy: f32,
}

/// Record that a span was produced by compacting a larger original.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactedFrom {
    pub original_id: String,
    pub method: String,
    pub ratio: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextSpan {
    /// blake3 content address of the canonical span (dedup + replay key).
    pub id: String,
    pub source: ContextSourceKind,
    pub title: String,
    pub text: String,
    /// Position in the final window (head→tail), assigned by the packer.
    #[serde(default)]
    pub order_index: usize,
    pub token_count: usize,
    /// Blended value the packer scored this span at.
    pub score: f32,
    #[serde(default)]
    pub signals: SpanSignals,
    #[serde(default)]
    pub pin: PinState,
    /// True when the span's KV was reused from the prefix bank/cache (not
    /// re-prefilled) — bible §4.5.
    #[serde(default)]
    pub banked: bool,
    #[serde(default)]
    pub compacted_from: Option<CompactedFrom>,
    pub provenance: Provenance,
    pub blob_ref: Option<BlobRef>,
}

/// Compute the blake3 content address of a span's canonical form. Stable across
/// turns: the same (kind, title, text) always hashes identically.
pub fn span_content_id(kind: &ContextSourceKind, title: &str, text: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"hide-span-v1\0");
    hasher.update(format!("{kind:?}").as_bytes());
    hasher.update(b"\0");
    hasher.update(title.as_bytes());
    hasher.update(b"\0");
    hasher.update(text.as_bytes());
    format!("blake3:{}", hasher.finalize().to_hex())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextSourceKind {
    System,
    UserTurn,
    Plan,
    Code,
    Symbol,
    ToolOutput,
    Memory,
    Scratchpad,
    Diagnostics,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DroppedContextSpan {
    pub id: String,
    pub source: ContextSourceKind,
    pub token_count: usize,
    pub score: f32,
    pub reason: DropReason,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DropReason {
    Budget,
    NoFit,
    Duplicate,
    Redundant,
    Stale,
    LowScore,
    LowValue,
    Unsafe,
    SourceUnavailable,
}

/// A surfaced contradiction between two spans (F12/F4): the compiler does not
/// silently pick one; it records the conflict for the user.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestConflict {
    pub between: Vec<String>,
    pub note: String,
    pub resolved: bool,
}

/// KV-reuse accounting for the manifest (A.1 `kv`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ManifestKv {
    pub prefix_reuse_tokens: usize,
    pub bank_hit: bool,
    pub tiers_touched: Vec<String>,
    pub checkpoint_id: Option<String>,
}

/// A compaction event (A.1 `compaction_events`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionEvent {
    pub original_id: String,
    pub result_id: String,
    pub method: String,
    pub model: Option<String>,
    pub ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_id_is_content_addressed_and_stable() {
        let a = span_content_id(&ContextSourceKind::Code, "t", "body");
        let b = span_content_id(&ContextSourceKind::Code, "t", "body");
        let c = span_content_id(&ContextSourceKind::Code, "t", "other");
        assert_eq!(a, b, "same content => same id");
        assert_ne!(a, c, "different body => different id");
        assert!(a.starts_with("blake3:"));
    }
}

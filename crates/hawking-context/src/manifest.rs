use hide_core::ids::{now_ms, EventId, TimestampMs};
use hide_core::types::{BlobRef, Provenance};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextManifest {
    pub schema_version: u16,
    pub created_at_ms: TimestampMs,
    pub source_event: Option<EventId>,
    pub model_context_tokens: usize,
    pub used_tokens: usize,
    pub retained: Vec<ContextSpan>,
    pub dropped: Vec<DroppedContextSpan>,
}

impl ContextManifest {
    pub fn new(model_context_tokens: usize) -> Self {
        Self {
            schema_version: 1,
            created_at_ms: now_ms(),
            source_event: None,
            model_context_tokens,
            used_tokens: 0,
            retained: Vec::new(),
            dropped: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextSpan {
    pub id: String,
    pub source: ContextSourceKind,
    pub title: String,
    pub text: String,
    pub token_count: usize,
    pub score: f32,
    pub provenance: Provenance,
    pub blob_ref: Option<BlobRef>,
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
    Duplicate,
    LowScore,
    Unsafe,
    SourceUnavailable,
}

use crate::ids::BlobId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustLevel {
    Trusted,
    UserAuthored,
    Workspace,
    ToolOutput,
    Network,
    Untrusted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Trivial,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    Allow,
    Ask,
    Deny,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextRange {
    pub start_line: u32,
    pub start_col: u32,
    pub end_line: u32,
    pub end_col: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ByteRange {
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobRef {
    pub id: BlobId,
    pub hash: String,
    pub size_bytes: u64,
    pub media_type: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub source: String,
    pub trust: TrustLevel,
    /// Confidence in this provenance's trust claim, 0.0..=1.0 (bible A.2/F12).
    /// Defaults to 1.0 for trusted/builtin sources.
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    pub labels: Vec<String>,
    pub derived_from: Vec<String>,
}

fn default_confidence() -> f32 {
    1.0
}

impl Provenance {
    pub fn trusted(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            trust: TrustLevel::Trusted,
            confidence: 1.0,
            labels: Vec::new(),
            derived_from: Vec::new(),
        }
    }

    /// Set the confidence in this provenance (builder).
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceScope {
    pub kind: String,
    pub pattern: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectKind {
    Read,
    Write,
    Delete,
    Execute,
    Network,
    Model,
    Plugin,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Effect {
    pub kind: EffectKind,
    pub target: String,
    pub bytes_hash: Option<String>,
    pub risk: RiskLevel,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct EffectSet {
    pub effects: Vec<Effect>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileSpan {
    pub path: PathBuf,
    pub range: Option<TextRange>,
    pub content_hash: Option<String>,
}

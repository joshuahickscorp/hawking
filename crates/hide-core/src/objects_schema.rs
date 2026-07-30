//! Object record schema: the durable shape of a YOU-store object.

use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::objects::hash::ContentHash;
use crate::objects::kinds::ObjectKind;
use crate::objects::permissions::ObjectPermissions;
use crate::objects::retention::RetentionPolicy;

/// Stable reference id (not the content identity). Many refs may share one hash.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RefId(pub String);

impl RefId {
    pub fn new() -> Self {
        Self(format!("oref_{}", Ulid::new()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for RefId {
    fn default() -> Self {
        Self::new()
    }
}

/// Where the body lives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "tier", rename_all = "snake_case")]
pub enum ObjectLocation {
    /// Content-addressed path under the local store root.
    Local { path: String },
    /// Cloud URI (accounting / future wire; not fetched by this crate).
    Cloud { uri: String },
    /// Not yet persisted.
    Pending,
}

/// Provenance of how the bytes entered the store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ObjectSource {
    UserUpload {
        filename: Option<String>,
        session_id: Option<String>,
    },
    Clipboard {
        session_id: Option<String>,
    },
    ToolOutput {
        tool: String,
        call_id: Option<String>,
    },
    WebCapture {
        url: String,
    },
    Email {
        message_id: Option<String>,
        attachment_name: Option<String>,
    },
    Connector {
        connector_id: String,
        remote_id: String,
    },
    Synthetic {
        /// Test / pipeline fixtures.
        label: String,
    },
}

/// Kind of derived representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivativeKind {
    /// Plain text extracted without OCR/ASR.
    TextExtract,
    /// OCR text (images/PDFs).
    Ocr,
    /// Audio/video transcript.
    Transcript,
    /// Small image proxy / thumbnail.
    Thumbnail,
    /// Lower-bitrate / shorter proxy of media.
    Proxy,
    /// Short model-facing summary of the object (metadata-only here).
    Summary,
}

impl DerivativeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TextExtract => "text_extract",
            Self::Ocr => "ocr",
            Self::Transcript => "transcript",
            Self::Thumbnail => "thumbnail",
            Self::Proxy => "proxy",
            Self::Summary => "summary",
        }
    }
}

/// A derived representation. Bytes live content-addressed; this is metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Derivative {
    pub kind: DerivativeKind,
    /// Content hash of the derivative body (when materialised).
    pub content_hash: Option<ContentHash>,
    pub mime: String,
    pub size_bytes: u64,
    /// Inline text for small text derivatives (OCR/transcript/extract).
    /// Large bodies use content_hash only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inline_text: Option<String>,
    /// Producer label (e.g. "FakeOcrEngine", "utf8_text_extract").
    pub produced_by: String,
    pub produced_at_ms: u64,
}

/// Pipeline stage names. Order is the default processing sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageName {
    Receive,
    Persist,
    Classify,
    ExtractText,
    OcrOrTranscript,
    Thumbnail,
    Finalize,
}

impl StageName {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Receive => "receive",
            Self::Persist => "persist",
            Self::Classify => "classify",
            Self::ExtractText => "extract_text",
            Self::OcrOrTranscript => "ocr_or_transcript",
            Self::Thumbnail => "thumbnail",
            Self::Finalize => "finalize",
        }
    }

    /// Canonical ordered pipeline.
    pub fn pipeline() -> &'static [StageName] {
        &[
            Self::Receive,
            Self::Persist,
            Self::Classify,
            Self::ExtractText,
            Self::OcrOrTranscript,
            Self::Thumbnail,
            Self::Finalize,
        ]
    }
}

/// Per-stage status — independently resumable and recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    Pending,
    Running,
    /// Partial progress; resume from `bytes_processed`.
    Partial,
    Complete,
    Failed,
    Skipped,
}

/// Recorded progress for one stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageRecord {
    pub stage: StageName,
    pub status: StageStatus,
    /// Bytes consumed so far (for streaming stages).
    pub bytes_processed: u64,
    /// Total size if known.
    pub bytes_total: Option<u64>,
    /// Peak working buffer used during this stage (proves streaming bound).
    pub peak_buffer_bytes: usize,
    pub attempts: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    pub updated_at_ms: u64,
}

impl StageRecord {
    pub fn pending(stage: StageName, now_ms: u64) -> Self {
        Self {
            stage,
            status: StageStatus::Pending,
            bytes_processed: 0,
            bytes_total: None,
            peak_buffer_bytes: 0,
            attempts: 0,
            last_error: None,
            updated_at_ms: now_ms,
        }
    }
}

/// Lifecycle of the object body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectStatus {
    Queued,
    Processing,
    Ready,
    Failed,
}

/// One content-addressed object. Identity is [`ObjectRecord::content_hash`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectRecord {
    /// blake3 of body bytes — the identity.
    pub content_hash: ContentHash,
    pub mime: String,
    pub kind: ObjectKind,
    pub size_bytes: u64,
    pub source: ObjectSource,
    pub location: ObjectLocation,
    pub status: ObjectStatus,
    pub stages: Vec<StageRecord>,
    pub derivatives: Vec<Derivative>,
    pub permissions: ObjectPermissions,
    pub retention: RetentionPolicy,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

impl ObjectRecord {
    pub fn stage_mut(&mut self, name: StageName) -> Option<&mut StageRecord> {
        self.stages.iter_mut().find(|s| s.stage == name)
    }

    pub fn stage(&self, name: StageName) -> Option<&StageRecord> {
        self.stages.iter().find(|s| s.stage == name)
    }

    pub fn is_ready(&self) -> bool {
        self.status == ObjectStatus::Ready
    }

    pub fn derivative(&self, kind: DerivativeKind) -> Option<&Derivative> {
        self.derivatives.iter().find(|d| d.kind == kind)
    }
}

/// A named reference to an object. Dedup is "one object, many refs".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectRef {
    pub id: RefId,
    pub content_hash: ContentHash,
    /// Display name (filename etc.) — not part of identity.
    pub label: Option<String>,
    pub created_at_ms: u64,
    /// Principal that created this reference.
    pub created_by: String,
}

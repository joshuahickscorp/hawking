//! Derivatives and the model-facing type boundary.
//!
//! The context-compile path may only receive [`ModelFacingDerivative`] /
//! [`CompileObjectView`]. It has **no** field and **no** method that yields
//! raw object bytes. Raw access requires an explicit [`RawBytesCap`] held only
//! by privileged host paths (export, local open), never by the compile path.

use serde::{Deserialize, Serialize};

use crate::objects::error::{ObjectError, Result};
use crate::objects::hash::ContentHash;
use crate::objects::kinds::ObjectKind;
use crate::objects::schema::{Derivative, DerivativeKind, ObjectRecord};

/// Capability token required to read raw object body bytes.
///
/// Construct only at privileged host entry points (export, download, open-in-
/// place). The context compiler must never hold this type.
#[derive(Debug, Clone, Copy)]
pub struct RawBytesCap {
    _private: (),
}

impl RawBytesCap {
    /// Mint only at privileged host paths — not the context-compile path.
    pub fn mint() -> Self {
        Self { _private: () }
    }
}

/// A single derivative selected for model context.
///
/// Deliberately cannot carry raw body bytes: only derivative text or a
/// content-hash reference to a derivative blob.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFacingDerivative {
    pub kind: DerivativeKind,
    pub mime: String,
    /// Inline text when the derivative is small text (OCR, transcript, extract).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Content hash of a non-text derivative (thumbnail/proxy) — never the
    /// original object hash unless they happen to collide (they must not).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derivative_hash: Option<ContentHash>,
    pub size_bytes: u64,
    pub produced_by: String,
}

impl ModelFacingDerivative {
    pub fn from_derivative(d: &Derivative) -> Self {
        Self {
            kind: d.kind,
            mime: d.mime.clone(),
            text: d.inline_text.clone(),
            derivative_hash: d.content_hash.clone(),
            size_bytes: d.size_bytes,
            produced_by: d.produced_by.clone(),
        }
    }
}

/// What the context-compile path is allowed to see for one object.
///
/// No raw bytes. No filesystem path to the body. Only selected derivatives
/// plus safe metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompileObjectView {
    pub content_hash: ContentHash,
    pub kind: ObjectKind,
    pub mime: String,
    pub size_bytes: u64,
    pub label: Option<String>,
    pub derivatives: Vec<ModelFacingDerivative>,
}

impl CompileObjectView {
    /// Build a compile view from a ready record and a selection of derivative kinds.
    ///
    /// Missing requested kinds are omitted (not an error); empty selection
    /// yields metadata-only.
    pub fn from_record(
        record: &ObjectRecord,
        select: &[DerivativeKind],
        label: Option<String>,
    ) -> Self {
        let derivatives = select
            .iter()
            .filter_map(|k| {
                record
                    .derivative(*k)
                    .map(ModelFacingDerivative::from_derivative)
            })
            .collect();
        Self {
            content_hash: record.content_hash.clone(),
            kind: record.kind,
            mime: record.mime.clone(),
            size_bytes: record.size_bytes,
            label,
            derivatives,
        }
    }

    /// There is intentionally no `raw_bytes` / `body` method on this type.
    /// This helper documents the boundary for tests and the contract.
    pub fn exposes_raw_bytes() -> bool {
        false
    }

    /// Attempting to "upgrade" a compile view to raw bytes always fails.
    pub fn try_raw_bytes(&self) -> Result<Vec<u8>> {
        let _ = self;
        Err(ObjectError::RawBytesForbidden)
    }
}

/// Selection request from the context compiler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivativeSelection {
    pub kinds: Vec<DerivativeKind>,
}

impl Default for DerivativeSelection {
    fn default() -> Self {
        Self {
            kinds: vec![
                DerivativeKind::TextExtract,
                DerivativeKind::Ocr,
                DerivativeKind::Transcript,
                DerivativeKind::Summary,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::permissions::{ObjectPermissions, Surface};
    use crate::objects::retention::RetentionPolicy;
    use crate::objects::schema::*;
    fn sample_record() -> ObjectRecord {
        ObjectRecord {
            content_hash: ContentHash::of_bytes(b"body"),
            mime: "text/plain".into(),
            kind: ObjectKind::Document,
            size_bytes: 4,
            source: ObjectSource::Synthetic { label: "t".into() },
            location: ObjectLocation::Pending,
            status: ObjectStatus::Ready,
            stages: vec![],
            derivatives: vec![Derivative {
                kind: DerivativeKind::TextExtract,
                content_hash: None,
                mime: "text/plain".into(),
                size_bytes: 4,
                inline_text: Some("body".into()),
                produced_by: "utf8_text_extract".into(),
                produced_at_ms: 0,
            }],
            permissions: ObjectPermissions::owner_only("u", vec![Surface::You]),
            retention: RetentionPolicy::durable(),
            created_at_ms: 0,
            updated_at_ms: 0,
        }
    }
    #[test]
    fn compile_view_has_derivatives_not_raw() {
        let rec = sample_record();
        let view = CompileObjectView::from_record(
            &rec,
            &[DerivativeKind::TextExtract],
            Some("note.txt".into()),
        );
        assert_eq!(view.derivatives.len(), 1);
        assert_eq!(view.derivatives[0].text.as_deref(), Some("body"));
        assert!(!CompileObjectView::exposes_raw_bytes());
        assert!(matches!(
            view.try_raw_bytes(),
            Err(ObjectError::RawBytesForbidden)
        ));
    }
}

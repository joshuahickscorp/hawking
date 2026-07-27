//! ACP content blocks (spec-derived).
//!
//! Spec-derived (ACP): a content block is a tagged union on a `type` field.
//! Prompts and agent-output chunks both carry content blocks. Only the public
//! wire shape is mirrored; no proprietary source is copied. HIDE uses the text,
//! resource-link, embedded-resource, and image forms; audio is out of scope for
//! the model-free boundary.

use serde::{Deserialize, Serialize};

/// The contents of an embedded resource block (the text-resource form).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceContents {
    pub uri: String,
    #[serde(rename = "mimeType", default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// One block of prompt input or agent output.
///
/// Spec-derived (ACP): tagged on `type` with the values `text`, `resource_link`,
/// `resource`, and `image`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text.
    Text { text: String },
    /// A link to a resource the peer can fetch (a file path or URI). HIDE maps
    /// its message attachments to these.
    ResourceLink {
        uri: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(rename = "mimeType", default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
    },
    /// An inline embedded resource (its text carried directly).
    Resource { resource: ResourceContents },
    /// A base64 image. The bytes are opaque to the model-free boundary.
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
}

impl ContentBlock {
    /// A plain-text block.
    pub fn text(text: impl Into<String>) -> Self {
        ContentBlock::Text { text: text.into() }
    }

    /// If this block is text, its string.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentBlock::Text { text } => Some(text),
            _ => None,
        }
    }

    /// The wire `type` tag for this block.
    pub fn type_tag(&self) -> &'static str {
        match self {
            ContentBlock::Text { .. } => "text",
            ContentBlock::ResourceLink { .. } => "resource_link",
            ContentBlock::Resource { .. } => "resource",
            ContentBlock::Image { .. } => "image",
        }
    }
}

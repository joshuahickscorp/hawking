//! Typed identifiers and artifact references for the browser evidence model.
//!
//! Ids are transparent string newtypes: they serialize as bare strings and
//! generate a plain-string JSON Schema. This crate never mints an id or a live
//! artifact; a caller (a recorder, a fixture, or a real driver built later)
//! owns the values. That keeps the crate deterministic and model-free.
//!
//! [`ArtifactRef`] is the one non-newtype here: it is how heavy evidence
//! (screenshots, raw DOM HTML, network bodies) is carried by reference instead
//! of inlined. Bytes never live in the evidence records; only a content
//! addressed reference to a stored blob does.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

macro_rules! id_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(
            Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash,
            Serialize, Deserialize, JsonSchema,
        )]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            /// Wrap an existing id value. This crate never generates ids.
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_string())
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

id_newtype!(
    /// Identifies one recorded browser session (a `Vec<BrowserStep>` with a
    /// header).
    BrowserSessionId
);
id_newtype!(
    /// A stable id for a node inside a captured DOM snapshot. The recorder
    /// assigns it; a selection and a Design Mode annotation both resolve to it.
    DomNodeId
);
id_newtype!(
    /// A stable id for a node inside a captured accessibility tree.
    AccessibilityNodeId
);

/// A reference to a stored binary artifact: a screenshot, the raw DOM HTML, a
/// network request or response body. The evidence model carries these instead
/// of the bytes so a step record stays small and the heavy payload lives in a
/// blob store addressed by [`ArtifactRef::id`].
///
/// The id is content addressed when built with [`ArtifactRef::content_addressed`]
/// (a `blake3:` digest of the bytes), so the same bytes always produce the same
/// reference -- deterministic and de-duplicating. There is deliberately no field
/// on this type that can hold the bytes: inlining is impossible by construction.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct ArtifactRef {
    /// The blob address. `blake3:<hex>` when content addressed.
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
}

impl ArtifactRef {
    /// Reference a blob by an already-known id (for example one a recorder
    /// assigned). No bytes are involved.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            media_type: None,
            size_bytes: None,
        }
    }

    /// Build a content-addressed reference from the artifact bytes. The bytes
    /// are hashed and then dropped; only the digest, size, and media type are
    /// retained. Deterministic: identical bytes yield an identical reference.
    pub fn content_addressed(bytes: &[u8], media_type: Option<&str>) -> Self {
        let hex = blake3::hash(bytes).to_hex();
        Self {
            id: format!("blake3:{hex}"),
            media_type: media_type.map(|s| s.to_string()),
            size_bytes: Some(bytes.len() as u64),
        }
    }

    /// Whether this reference is a content-addressed `blake3:` digest.
    pub fn is_content_addressed(&self) -> bool {
        self.id.starts_with("blake3:")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_serialize_transparently_as_bare_strings() {
        let id = DomNodeId::from("n-42");
        assert_eq!(serde_json::to_string(&id).unwrap(), "\"n-42\"");
        let back: DomNodeId = serde_json::from_str("\"n-42\"").unwrap();
        assert_eq!(back, id);
    }

    #[test]
    fn content_addressed_ref_is_deterministic_and_holds_no_bytes() {
        let bytes = b"fake-png-bytes";
        let a = ArtifactRef::content_addressed(bytes, Some("image/png"));
        let b = ArtifactRef::content_addressed(bytes, Some("image/png"));
        assert_eq!(a, b, "same bytes -> same reference");
        assert!(a.is_content_addressed());
        assert_eq!(a.size_bytes, Some(bytes.len() as u64));
        // The serialized reference names the blob but never carries its bytes.
        let json = serde_json::to_string(&a).unwrap();
        assert!(json.contains("blake3:"));
        assert!(!json.contains("fake-png-bytes"));
    }
}

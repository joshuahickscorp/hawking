//! The compact metadata index.
//!
//! Progressive disclosure means a planner first sees only enough to decide
//! whether a capability is worth loading. That view is a [`CompactEntry`]: id,
//! kind, description, and scopes, and nothing heavier. In particular it carries
//! no schema and no raw schema text, so building or returning the index never
//! materializes a full schema. The full schema is reached only through an
//! explicit [`crate::Registry::load_full_schema`] call.

use crate::manifest::{CapabilityKind, CapabilityManifest, Scope};

/// The disclosed-up-front view of one capability. Deliberately excludes the
/// input/output schemas, effects, sandbox, network and secret policies, and
/// provenance. Those come from the narrow enforcement accessors or from a full
/// schema load, so the cheap index stays cheap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactEntry {
    pub id: String,
    pub kind: CapabilityKind,
    pub description: String,
    pub scopes: Vec<Scope>,
}

impl CompactEntry {
    /// Project a manifest down to its compact view. This copies only the four
    /// disclosed fields; the schema refs are not touched.
    pub fn from_manifest(m: &CapabilityManifest) -> Self {
        CompactEntry {
            id: m.id.clone(),
            kind: m.kind,
            description: m.description.clone(),
            scopes: m.scopes.clone(),
        }
    }
}

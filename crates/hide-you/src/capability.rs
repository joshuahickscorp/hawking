//! Surface capability derivation — structurally non-widening.
//!
//! Mirrors `hide_core::automation::{PermissionSet, JobCapability}`: a
//! [`SurfaceCapability`] can only be obtained by deriving from a
//! [`SurfacePermissionSet`]. There is no public constructor that invents
//! tools or connectors, and no method that adds them after the fact.
//!
//! This is the enforcement spine for the handoff invariant: a capsule may
//! *describe* permissions held at creation, but cannot mint a capability.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::error::{Result, YouError};

/// Closed set of tools and connectors a surface (or agent) is allowed to use.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SurfacePermissionSet {
    tools: BTreeSet<String>,
    connectors: BTreeSet<String>,
}

impl SurfacePermissionSet {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn new(
        tools: impl IntoIterator<Item = impl Into<String>>,
        connectors: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            tools: tools.into_iter().map(Into::into).collect(),
            connectors: connectors.into_iter().map(Into::into).collect(),
        }
    }

    pub fn tools(&self) -> &BTreeSet<String> {
        &self.tools
    }

    pub fn connectors(&self) -> &BTreeSet<String> {
        &self.connectors
    }

    pub fn grants_tool(&self, name: &str) -> bool {
        self.tools.contains(name)
    }

    pub fn grants_connector(&self, name: &str) -> bool {
        self.connectors.contains(name)
    }

    /// Full capability granted by this set. Receiver cannot widen it.
    pub fn derive_capability(&self) -> SurfaceCapability {
        SurfaceCapability {
            tools: self.tools.clone(),
            connectors: self.connectors.clone(),
        }
    }

    /// Subset derivation; requesting anything outside the set fails closed.
    pub fn derive_capability_subset(
        &self,
        tools: impl IntoIterator<Item = impl AsRef<str>>,
        connectors: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<SurfaceCapability> {
        let mut out_tools = BTreeSet::new();
        for t in tools {
            let name = t.as_ref();
            if !self.tools.contains(name) {
                return Err(YouError::CapabilityMissing(format!(
                    "cannot derive capability for tool '{name}': not in permission set"
                )));
            }
            out_tools.insert(name.to_string());
        }
        let mut out_connectors = BTreeSet::new();
        for c in connectors {
            let name = c.as_ref();
            if !self.connectors.contains(name) {
                return Err(YouError::CapabilityMissing(format!(
                    "cannot derive capability for connector '{name}': not in permission set"
                )));
            }
            out_connectors.insert(name.to_string());
        }
        Ok(SurfaceCapability {
            tools: out_tools,
            connectors: out_connectors,
        })
    }

    /// Intersection with another set (for session-scoped narrowing only).
    pub fn intersect(&self, other: &SurfacePermissionSet) -> SurfacePermissionSet {
        SurfacePermissionSet {
            tools: self
                .tools
                .intersection(&other.tools)
                .cloned()
                .collect(),
            connectors: self
                .connectors
                .intersection(&other.connectors)
                .cloned()
                .collect(),
        }
    }
}

/// Capability handed to a surface session or agent. **Structurally
/// non-widening**: fields private; only construction path is
/// [`SurfacePermissionSet::derive_capability`] /
/// [`SurfacePermissionSet::derive_capability_subset`].
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SurfaceCapability {
    tools: BTreeSet<String>,
    connectors: BTreeSet<String>,
}

impl SurfaceCapability {
    pub fn tools(&self) -> &BTreeSet<String> {
        &self.tools
    }

    pub fn connectors(&self) -> &BTreeSet<String> {
        &self.connectors
    }

    pub fn allows_tool(&self, name: &str) -> bool {
        self.tools.contains(name)
    }

    pub fn allows_connector(&self, name: &str) -> bool {
        self.connectors.contains(name)
    }

    pub fn require_tool(&self, name: &str) -> Result<()> {
        if self.allows_tool(name) {
            Ok(())
        } else {
            Err(YouError::PolicyDenied(format!(
                "surface capability does not grant tool '{name}'"
            )))
        }
    }

    pub fn require_connector(&self, name: &str) -> Result<()> {
        if self.allows_connector(name) {
            Ok(())
        } else {
            Err(YouError::PolicyDenied(format!(
                "surface capability does not grant connector '{name}'"
            )))
        }
    }

    /// True iff every tool/connector in `self` is also in `parent`.
    pub fn is_within(&self, parent: &SurfacePermissionSet) -> bool {
        self.tools.is_subset(parent.tools())
            && self.connectors.is_subset(parent.connectors())
    }

    /// Snapshot for audit (claims about authority, not a grant handle).
    pub fn snapshot(&self) -> CapabilitySnapshot {
        CapabilitySnapshot {
            tools: self.tools.iter().cloned().collect(),
            connectors: self.connectors.iter().cloned().collect(),
        }
    }
}

/// Serializable description of tools/connectors held at a moment in time.
/// This is a CLAIM about authority for provenance — not a live capability.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CapabilitySnapshot {
    pub tools: Vec<String>,
    pub connectors: Vec<String>,
}

impl CapabilitySnapshot {
    pub fn from_set(set: &SurfacePermissionSet) -> Self {
        Self {
            tools: set.tools().iter().cloned().collect(),
            connectors: set.connectors().iter().cloned().collect(),
        }
    }

    pub fn from_capability(cap: &SurfaceCapability) -> Self {
        cap.snapshot()
    }
}

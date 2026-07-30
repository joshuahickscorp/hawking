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

use crate::lenses::error::{Result, YouError};

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
            live: true,
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
            live: true,
        })
    }

    /// Intersection with another set (for session-scoped narrowing only).
    pub fn intersect(&self, other: &SurfacePermissionSet) -> SurfacePermissionSet {
        SurfacePermissionSet {
            tools: self.tools.intersection(&other.tools).cloned().collect(),
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
///
/// `live` is never serialized. A capability forged via `serde` (or any other
/// path that does not go through derive) deserializes with `live = false` and
/// fails closed on every gate. That closes the export / handoff payload attack
/// of smuggling a capability-shaped JSON object.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SurfaceCapability {
    tools: BTreeSet<String>,
    connectors: BTreeSet<String>,
    /// True only when constructed by [`SurfacePermissionSet::derive_capability`]
    /// or [`SurfacePermissionSet::derive_capability_subset`].
    #[serde(skip)]
    live: bool,
}

impl SurfaceCapability {
    pub fn tools(&self) -> &BTreeSet<String> {
        &self.tools
    }

    pub fn connectors(&self) -> &BTreeSet<String> {
        &self.connectors
    }

    /// Whether this handle was minted by derive (not forged via serde/export).
    pub fn is_live(&self) -> bool {
        self.live
    }

    pub fn allows_tool(&self, name: &str) -> bool {
        self.live && self.tools.contains(name)
    }

    pub fn allows_connector(&self, name: &str) -> bool {
        self.live && self.connectors.contains(name)
    }

    pub fn require_tool(&self, name: &str) -> Result<()> {
        if !self.live {
            return Err(YouError::PolicyDenied(
                "surface capability is not live (forged or deserialized; derive only)".into(),
            ));
        }
        if self.tools.contains(name) {
            Ok(())
        } else {
            Err(YouError::PolicyDenied(format!(
                "surface capability does not grant tool '{name}'"
            )))
        }
    }

    pub fn require_connector(&self, name: &str) -> Result<()> {
        if !self.live {
            return Err(YouError::PolicyDenied(
                "surface capability is not live (forged or deserialized; derive only)".into(),
            ));
        }
        if self.connectors.contains(name) {
            Ok(())
        } else {
            Err(YouError::PolicyDenied(format!(
                "surface capability does not grant connector '{name}'"
            )))
        }
    }

    /// True iff every tool/connector in `self` is also in `parent`.
    pub fn is_within(&self, parent: &SurfacePermissionSet) -> bool {
        self.live
            && self.tools.is_subset(parent.tools())
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn derive_is_live_default_is_not() {
        let empty = SurfaceCapability::default();
        assert!(!empty.is_live());
        assert!(!empty.allows_tool("x"));
        let set = SurfacePermissionSet::new(["t1"], ["c1"]);
        let cap = set.derive_capability();
        assert!(cap.is_live());
        assert!(cap.allows_tool("t1"));
        assert!(cap.allows_connector("c1"));
    }
    #[test]
    fn adversarial_serde_forge_is_not_live() {
        let forged: SurfaceCapability = serde_json::from_value(json!({
            "tools": ["shell.exec"],
            "connectors": ["gmail"],
            "live": true
        }))
        .unwrap();
        assert!(!forged.is_live());
        assert!(!forged.allows_tool("shell.exec"));
        assert!(forged.require_tool("shell.exec").is_err());
        assert!(forged.require_connector("gmail").is_err());
    }
    #[test]
    fn subset_cannot_widen() {
        let set = SurfacePermissionSet::new(["a"], ["x"]);
        assert!(set
            .derive_capability_subset(["a", "b"], None::<&str>)
            .is_err());
        assert!(set.derive_capability_subset(["a"], ["y"]).is_err());
        let ok = set.derive_capability_subset(["a"], ["x"]).unwrap();
        assert!(ok.is_live());
        assert!(ok.allows_tool("a"));
    }
}

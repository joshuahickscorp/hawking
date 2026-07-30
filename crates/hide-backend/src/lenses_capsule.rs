//! Typed handoff capsules: CLAIM, never CAPABILITY.
//!
//! Every transfer between surfaces creates a typed capsule, never a flattened
//! transcript. The single most important invariant:
//!
//! > A capsule never widens permission. Receiving a capsule in CHAT does not
//! > grant CHAT the connector access YOU held when it was created. The capsule
//! > carries the CLAIM, never the CAPABILITY.
//!
//! Enforced as a type boundary: there is no API that turns a capsule into a
//! [`SurfaceCapability`]. Attempts to extract capability fail closed and are
//! testable.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::lenses::capability::{CapabilitySnapshot, SurfaceCapability};
use crate::lenses::error::{Result, YouError};
use crate::lenses::evidence::EvidenceTier;
use crate::lenses::surface::Surface;

/// Direction of a typed surface handoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandoffKind {
    /// YOU → CHAT: implementation campaign.
    YouToChat,
    /// CHAT → IDE: repository/worktree and verification plan.
    ChatToIde,
    /// IDE → YOU: release summary.
    IdeToYou,
}

impl HandoffKind {
    pub fn from_surface(self) -> Surface {
        match self {
            Self::YouToChat => Surface::You,
            Self::ChatToIde => Surface::Chat,
            Self::IdeToYou => Surface::Ide,
        }
    }

    pub fn to_surface(self) -> Surface {
        match self {
            Self::YouToChat => Surface::Chat,
            Self::ChatToIde => Surface::Ide,
            Self::IdeToYou => Surface::You,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::YouToChat => "you_to_chat",
            Self::ChatToIde => "chat_to_ide",
            Self::IdeToYou => "ide_to_you",
        }
    }
}

/// One link in the provenance chain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceEntry {
    pub actor: String,
    pub surface: Surface,
    pub at_ms: u64,
    pub action: String,
}

/// A claim the capsule asserts, with its evidence tier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Claim {
    pub id: String,
    pub text: String,
    pub evidence_tier: EvidenceTier,
    /// Optional structured payload (goals, plans, summaries — never capabilities).
    #[serde(default)]
    pub payload: Value,
}

/// What the capsule deliberately leaves out, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeliberateExclusion {
    pub item: String,
    pub reason: String,
}

/// Snapshot of permissions under which the capsule was created.
/// This is an audit claim, not a grant handle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionSnapshot {
    pub surface: Surface,
    pub tools: Vec<String>,
    pub connectors: Vec<String>,
}

impl PermissionSnapshot {
    pub fn from_capability(surface: Surface, cap: &SurfaceCapability) -> Self {
        let snap = cap.snapshot();
        Self {
            surface,
            tools: snap.tools,
            connectors: snap.connectors,
        }
    }

    pub fn from_snapshot(surface: Surface, snap: &CapabilitySnapshot) -> Self {
        Self {
            surface,
            tools: snap.tools.clone(),
            connectors: snap.connectors.clone(),
        }
    }
}

/// Typed handoff capsule. Carries claims + audit metadata. **Does not grant
/// capability.** Fields that describe permissions are snapshots for provenance
/// only; the only way to obtain a live [`SurfaceCapability`] is surface-default
/// derivation (or an explicit session grant elsewhere), never from this type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HandoffCapsule {
    pub id: String,
    pub kind: HandoffKind,
    pub origin_surface: Surface,
    pub origin_session: String,
    pub target_surface: Surface,
    pub created_ms: u64,
    pub provenance: Vec<ProvenanceEntry>,
    pub claims: Vec<Claim>,
    /// Permissions under which this capsule was created (audit only).
    pub permissions_at_creation: PermissionSnapshot,
    pub deliberately_excludes: Vec<DeliberateExclusion>,
    /// blake3 hex of canonical claim+exclusion content.
    pub content_hash: String,
    /// Kind-specific body (campaign / verification plan / release summary).
    pub body: Value,
}

impl HandoffCapsule {
    /// Build a capsule. Computes content hash over claims + exclusions + body.
    pub fn seal(
        kind: HandoffKind,
        origin_session: impl Into<String>,
        created_ms: u64,
        provenance: Vec<ProvenanceEntry>,
        claims: Vec<Claim>,
        permissions_at_creation: PermissionSnapshot,
        deliberately_excludes: Vec<DeliberateExclusion>,
        body: Value,
    ) -> Result<Self> {
        let origin = kind.from_surface();
        let target = kind.to_surface();
        if permissions_at_creation.surface != origin {
            return Err(YouError::InvalidHandoff(format!(
                "permissions snapshot surface {:?} does not match origin {:?}",
                permissions_at_creation.surface, origin
            )));
        }
        let content_hash = hash_capsule_content(&claims, &deliberately_excludes, &body);
        Ok(Self {
            id: format!("hcap_{}", mint_ulid()),
            kind,
            origin_surface: origin,
            origin_session: origin_session.into(),
            target_surface: target,
            created_ms,
            provenance,
            claims,
            permissions_at_creation,
            deliberately_excludes,
            content_hash,
            body,
        })
    }

    /// Open for a receiving surface: returns claims and body only.
    /// Does **not** transfer or grant any capability from
    /// `permissions_at_creation`.
    pub fn open_for(&self, receiver: Surface) -> Result<OpenedCapsule> {
        if receiver != self.target_surface {
            return Err(YouError::InvalidHandoff(format!(
                "capsule targets {:?}, cannot open as {:?}",
                self.target_surface, receiver
            )));
        }
        Ok(OpenedCapsule {
            capsule_id: self.id.clone(),
            kind: self.kind,
            origin_surface: self.origin_surface,
            origin_session: self.origin_session.clone(),
            claims: self.claims.clone(),
            deliberately_excludes: self.deliberately_excludes.clone(),
            body: self.body.clone(),
            content_hash: self.content_hash.clone(),
            permissions_described: self.permissions_at_creation.clone(),
        })
    }

    /// Type-boundary enforcement: a capsule cannot be turned into a live
    /// capability. Always fails closed. Tests assert this path.
    pub fn try_extract_capability(&self) -> Result<SurfaceCapability> {
        Err(YouError::PolicyDenied(format!(
            "capsule {} carries claims only; cannot extract capability \
             (connectors at creation: {:?})",
            self.id, self.permissions_at_creation.connectors
        )))
    }

    /// Attempt to use a connector named in the creation snapshot as if the
    /// capsule granted it. Always fails — documents the attack surface.
    pub fn try_use_creator_connector(&self, connector: &str) -> Result<()> {
        if self
            .permissions_at_creation
            .connectors
            .iter()
            .any(|c| c == connector)
        {
            return Err(YouError::PolicyDenied(format!(
                "capsule describes creator connector '{connector}' but does not grant it; \
                 receiving surface must derive its own capability"
            )));
        }
        Err(YouError::CapabilityMissing(format!(
            "connector '{connector}' not even described on capsule"
        )))
    }

    /// Verify content hash integrity.
    pub fn verify_hash(&self) -> bool {
        let expected = hash_capsule_content(&self.claims, &self.deliberately_excludes, &self.body);
        expected == self.content_hash
    }
}

/// What a receiving surface gets after opening a capsule: claims + body +
/// audit description of creator permissions. No live capability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenedCapsule {
    pub capsule_id: String,
    pub kind: HandoffKind,
    pub origin_surface: Surface,
    pub origin_session: String,
    pub claims: Vec<Claim>,
    pub deliberately_excludes: Vec<DeliberateExclusion>,
    pub body: Value,
    pub content_hash: String,
    /// Audit-only description of what the origin held. Not a grant.
    pub permissions_described: PermissionSnapshot,
}

impl OpenedCapsule {
    /// There is no method on OpenedCapsule that yields SurfaceCapability.
    /// This helper makes the invariant explicit for callers and tests.
    pub fn grants_capability(&self) -> bool {
        false
    }
}

/// Result of a surface receiving a capsule into its session: claims are
/// ingested; the session's live capability is **unchanged**.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReceivedHandoff {
    pub opened: OpenedCapsule,
    /// Capability the receiver held before and after receive (must be equal).
    pub receiver_capability_before: CapabilitySnapshot,
    pub receiver_capability_after: CapabilitySnapshot,
}

impl ReceivedHandoff {
    pub fn capability_unchanged(&self) -> bool {
        self.receiver_capability_before == self.receiver_capability_after
    }
}

/// A live surface session: holds its own derived capability. Receiving a
/// capsule never mutates that capability.
#[derive(Debug, Clone)]
pub struct SurfaceSession {
    pub surface: Surface,
    pub session_id: String,
    capability: SurfaceCapability,
}

impl SurfaceSession {
    pub fn open(surface: Surface, session_id: impl Into<String>) -> Self {
        let defaults = crate::lenses::surface::SurfaceDefaults::for_surface(surface);
        Self {
            surface,
            session_id: session_id.into(),
            capability: defaults.permissions.derive_capability(),
        }
    }

    pub fn with_capability(
        surface: Surface,
        session_id: impl Into<String>,
        capability: SurfaceCapability,
    ) -> Self {
        Self {
            surface,
            session_id: session_id.into(),
            capability,
        }
    }

    pub fn capability(&self) -> &SurfaceCapability {
        &self.capability
    }

    /// Receive a typed capsule. Ingests claims; **does not** widen capability
    /// to include creator connectors/tools.
    pub fn receive(&self, capsule: &HandoffCapsule) -> Result<ReceivedHandoff> {
        let before = self.capability.snapshot();
        let opened = capsule.open_for(self.surface)?;
        // Deliberately do nothing to self.capability — receive is &self.
        let after = self.capability.snapshot();
        Ok(ReceivedHandoff {
            opened,
            receiver_capability_before: before,
            receiver_capability_after: after,
        })
    }

    /// Gate a tool under this session's capability (not the capsule's snapshot).
    pub fn require_tool(&self, tool: &str) -> Result<()> {
        self.capability.require_tool(tool)
    }

    pub fn require_connector(&self, connector: &str) -> Result<()> {
        self.capability.require_connector(connector)
    }
}

fn hash_capsule_content(
    claims: &[Claim],
    exclusions: &[DeliberateExclusion],
    body: &Value,
) -> String {
    let payload = serde_json::json!({
        "claims": claims,
        "exclusions": exclusions,
        "body": body,
    });
    let bytes = serde_json::to_vec(&payload).unwrap_or_default();
    format!("blake3:{}", blake3::hash(&bytes).to_hex())
}

fn mint_ulid() -> String {
    ulid::Ulid::new().to_string().to_ascii_lowercase()
}

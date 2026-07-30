//! One HIDE session, three surface lenses.
//!
//! Doctrine (`evidence/hide/HIDE_YOU_SURFACE_AUTHORITY.json`): YOU, CHAT and IDE are three
//! **lenses** over one session. They share session identity and must not each
//! own a copy of memory, objects, connectors, or the event stream. What differs
//! is default context and default **capability** (non-widening, derived once).
//!
//! A handoff capsule still carries a CLAIM only. Switching the active lens, or
//! receiving a capsule into a lens, never transports authority.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::lenses::capsule::{
    Claim, DeliberateExclusion, HandoffCapsule, HandoffKind, OpenedCapsule, PermissionSnapshot,
    ProvenanceEntry, ReceivedHandoff, SurfaceSession,
};
use crate::lenses::error::{Result, YouError};
use crate::lenses::surface::Surface;

/// Shared product session: one identity, three capability lenses, typed handoffs.
///
/// Surfaces do not construct independent sessions. They call [`SurfaceGraph::lens`]
/// and [`SurfaceGraph::switch`]. The host owns the single durable event log /
/// memory / object store; this graph only holds the surface authority view.
#[derive(Debug, Clone)]
pub struct SurfaceGraph {
    session_id: String,
    active: Surface,
    lenses: BTreeMap<Surface, SurfaceSession>,
    /// Sealed outbound capsules, keyed by capsule id.
    capsules: BTreeMap<String, HandoffCapsule>,
    /// Claims received per surface (capability never stored here).
    inbox: BTreeMap<Surface, Vec<OpenedCapsule>>,
}

/// Read-only snapshot a FE / projection can render without holding live capability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceGraphView {
    pub session_id: String,
    pub active_surface: String,
    /// Per-surface audit capability description (tools + connectors). Not a grant handle.
    pub lenses: BTreeMap<String, LensView>,
    pub unread_handoffs: usize,
    pub capsules: Vec<CapsuleView>,
    pub inbox: BTreeMap<String, Vec<OpenedCapsuleView>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LensView {
    pub surface: String,
    pub tools: Vec<String>,
    pub connectors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapsuleView {
    pub id: String,
    pub kind: String,
    pub origin_surface: String,
    pub target_surface: String,
    pub content_hash: String,
    pub claim_count: usize,
    pub exclusion_count: usize,
    pub exclusions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenedCapsuleView {
    pub capsule_id: String,
    pub kind: String,
    pub origin_surface: String,
    pub claim_count: usize,
    pub content_hash: String,
    /// Audit-only description of creator permissions. Not a grant.
    pub permissions_described_tools: Vec<String>,
    pub permissions_described_connectors: Vec<String>,
}

impl SurfaceGraph {
    /// Open a graph on one session id. All three lenses share that id and hold
    /// surface-default capabilities only.
    pub fn open(session_id: impl Into<String>) -> Self {
        let session_id = session_id.into();
        let mut lenses = BTreeMap::new();
        for surface in Surface::all() {
            lenses.insert(
                surface,
                SurfaceSession::open(surface, session_id.clone()),
            );
        }
        let mut inbox = BTreeMap::new();
        for surface in Surface::all() {
            inbox.insert(surface, Vec::new());
        }
        Self {
            session_id,
            // Doctrine: Workstation / Chat is the front door; YOU is a lens, not a silo.
            active: Surface::Chat,
            lenses,
            capsules: BTreeMap::new(),
            inbox,
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn active(&self) -> Surface {
        self.active
    }

    /// Switch the active lens. Does not mint a new session. Does not change any
    /// surface's capability.
    pub fn switch(&mut self, surface: Surface) -> Surface {
        self.active = surface;
        self.active
    }

    /// Borrow the live lens for a surface. All lenses share [`Self::session_id`].
    pub fn lens(&self, surface: Surface) -> Result<&SurfaceSession> {
        self.lenses
            .get(&surface)
            .ok_or_else(|| YouError::InvalidState(format!("missing lens for {surface}")))
    }

    pub fn active_lens(&self) -> Result<&SurfaceSession> {
        self.lens(self.active)
    }

    /// Seal a typed handoff from the active surface (or an explicit origin).
    ///
    /// The capsule records a permission **snapshot** for audit only. Live
    /// capability stays on the origin lens; the capsule cannot reconstitute it.
    pub fn create_handoff(
        &mut self,
        kind: HandoffKind,
        created_ms: u64,
        claims: Vec<Claim>,
        deliberately_excludes: Vec<DeliberateExclusion>,
        body: Value,
        actor: impl Into<String>,
    ) -> Result<HandoffCapsule> {
        let origin = kind.from_surface();
        let origin_lens = self.lens(origin)?;
        // Caller may only create handoffs from a surface that is part of this graph
        // and that matches the kind's origin. Active surface should be the origin
        // (prevents CHAT sealing a YOU→CHAT capsule while pretending to be YOU).
        if self.active != origin {
            return Err(YouError::PolicyDenied(format!(
                "active surface is {}; handoff kind {} requires origin {}",
                self.active,
                kind.as_str(),
                origin
            )));
        }
        let permissions =
            PermissionSnapshot::from_capability(origin, origin_lens.capability());
        let provenance = vec![ProvenanceEntry {
            actor: actor.into(),
            surface: origin,
            at_ms: created_ms,
            action: format!("handoff_{}", kind.as_str()),
        }];
        // Shared session identity on every capsule (one session, three lenses).
        let capsule = HandoffCapsule::seal(
            kind,
            self.session_id.clone(),
            created_ms,
            provenance,
            claims,
            permissions,
            deliberately_excludes,
            body,
        )?;
        self.capsules
            .insert(capsule.id.clone(), capsule.clone());
        Ok(capsule)
    }

    /// Receive a sealed capsule into its target lens on **this same session**.
    ///
    /// Capability of the target lens is unchanged. Creator connectors remain
    /// unusable on the receiver. Claims land in the target inbox.
    pub fn receive_handoff(&mut self, capsule_id: &str) -> Result<ReceivedHandoff> {
        let capsule = self
            .capsules
            .get(capsule_id)
            .cloned()
            .ok_or_else(|| {
                YouError::InvalidHandoff(format!("unknown capsule id {capsule_id}"))
            })?;
        // Capsules created elsewhere for a different session are refused: lenses
        // share one session, not a free-floating claim bus across sessions.
        if capsule.origin_session != self.session_id {
            return Err(YouError::InvalidHandoff(format!(
                "capsule session {} does not match graph session {}",
                capsule.origin_session, self.session_id
            )));
        }
        let target = capsule.target_surface;
        let lens = self.lens(target)?;
        let received = lens.receive(&capsule)?;
        // Type boundary re-check at the graph boundary (defense in depth).
        if !received.capability_unchanged() {
            return Err(YouError::PolicyDenied(
                "receive mutated receiver capability; refused".into(),
            ));
        }
        if let Err(err) = capsule.try_extract_capability() {
            // Expected always. Keep the error path live so a regression that
            // starts succeeding is not silently ignored.
            let _ = err;
        } else {
            return Err(YouError::PolicyDenied(
                "capsule yielded a capability; refused".into(),
            ));
        }
        self.inbox
            .entry(target)
            .or_default()
            .push(received.opened.clone());
        Ok(received)
    }

    /// Import a capsule sealed outside this graph (e.g. restored from the event
    /// log). Still refuses if its origin_session disagrees with this session.
    pub fn admit_capsule(&mut self, capsule: HandoffCapsule) -> Result<()> {
        if capsule.origin_session != self.session_id {
            return Err(YouError::InvalidHandoff(format!(
                "admit refused: capsule session {} != graph {}",
                capsule.origin_session, self.session_id
            )));
        }
        // Never trust a capsule that could somehow extract capability.
        if capsule.try_extract_capability().is_ok() {
            return Err(YouError::PolicyDenied(
                "admit refused: capsule carries capability".into(),
            ));
        }
        self.capsules.insert(capsule.id.clone(), capsule);
        Ok(())
    }

    pub fn capsule(&self, id: &str) -> Option<&HandoffCapsule> {
        self.capsules.get(id)
    }

    pub fn inbox_for(&self, surface: Surface) -> &[OpenedCapsule] {
        self.inbox
            .get(&surface)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn unread_handoff_count(&self) -> usize {
        self.inbox.values().map(|v| v.len()).sum()
    }

    /// Projection for Wire-B / FE: state only, no live capability handles.
    pub fn view(&self) -> SurfaceGraphView {
        let mut lenses = BTreeMap::new();
        for surface in Surface::all() {
            if let Ok(lens) = self.lens(surface) {
                let snap = lens.capability().snapshot();
                lenses.insert(
                    surface.as_str().to_string(),
                    LensView {
                        surface: surface.as_str().to_string(),
                        tools: snap.tools,
                        connectors: snap.connectors,
                    },
                );
            }
        }
        let capsules: Vec<CapsuleView> = self
            .capsules
            .values()
            .map(|c| CapsuleView {
                id: c.id.clone(),
                kind: c.kind.as_str().to_string(),
                origin_surface: c.origin_surface.as_str().to_string(),
                target_surface: c.target_surface.as_str().to_string(),
                content_hash: c.content_hash.clone(),
                claim_count: c.claims.len(),
                exclusion_count: c.deliberately_excludes.len(),
                exclusions: c
                    .deliberately_excludes
                    .iter()
                    .map(|e| format!("{} ({})", e.item, e.reason))
                    .collect(),
            })
            .collect();
        let mut inbox = BTreeMap::new();
        for surface in Surface::all() {
            let items: Vec<OpenedCapsuleView> = self
                .inbox_for(surface)
                .iter()
                .map(|o| OpenedCapsuleView {
                    capsule_id: o.capsule_id.clone(),
                    kind: o.kind.as_str().to_string(),
                    origin_surface: o.origin_surface.as_str().to_string(),
                    claim_count: o.claims.len(),
                    content_hash: o.content_hash.clone(),
                    permissions_described_tools: o.permissions_described.tools.clone(),
                    permissions_described_connectors: o.permissions_described.connectors.clone(),
                })
                .collect();
            inbox.insert(surface.as_str().to_string(), items);
        }
        SurfaceGraphView {
            session_id: self.session_id.clone(),
            active_surface: self.active.as_str().to_string(),
            lenses,
            unread_handoffs: self.unread_handoff_count(),
            capsules,
            inbox,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lenses::evidence::EvidenceTier;
    use serde_json::json;
    #[test]
    fn three_lenses_share_one_session_id() {
        let g = SurfaceGraph::open("ses_shared");
        assert_eq!(g.session_id(), "ses_shared");
        for s in Surface::all() {
            assert_eq!(g.lens(s).unwrap().session_id, "ses_shared");
        }
 assert!(g .lens(Surface::You) .unwrap() .capability() .allows_connector("gmail"));
 assert!(!g .lens(Surface::Chat) .unwrap() .capability() .allows_connector("gmail"));
 assert!(!g .lens(Surface::Ide) .unwrap() .capability() .allows_connector("gmail"));
    }
    #[test]
    fn switch_does_not_change_session_or_capability() {
        let mut g = SurfaceGraph::open("ses_1");
        let before_chat = g.lens(Surface::Chat).unwrap().capability().snapshot();
        let before_you = g.lens(Surface::You).unwrap().capability().snapshot();
        g.switch(Surface::You);
        assert_eq!(g.active(), Surface::You);
        assert_eq!(g.session_id(), "ses_1");
 assert_eq!( g.lens(Surface::Chat).unwrap().capability().snapshot(), before_chat );
 assert_eq!( g.lens(Surface::You).unwrap().capability().snapshot(), before_you );
    }
    #[test]
    fn handoff_claim_never_grants_creator_capability_on_shared_session() {
        let mut g = SurfaceGraph::open("ses_shared");
        g.switch(Surface::You);
        let capsule = g
            .create_handoff(
                HandoffKind::YouToChat,
                1_000,
                vec![Claim {
                    id: "c1".into(),
                    text: "build triage worker".into(),
                    evidence_tier: EvidenceTier::Cited,
                    payload: json!({}),
                }],
                vec![DeliberateExclusion {
                    item: "gmail credentials".into(),
                    reason: "claim only".into(),
                }],
                json!({"kind": "implementation_campaign", "goal": "triage"}),
                "user",
            )
            .expect("create");
        assert_eq!(capsule.origin_session, "ses_shared");
        assert!(capsule.try_extract_capability().is_err());
        let received = g.receive_handoff(&capsule.id).expect("receive");
        assert!(received.capability_unchanged());
        assert!(!received.opened.grants_capability());
 assert!(g .lens(Surface::Chat) .unwrap() .require_connector("gmail") .is_err());
 assert!(g .lens(Surface::You) .unwrap() .require_connector("gmail") .is_ok());
        assert_eq!(g.inbox_for(Surface::Chat).len(), 1);
    }
    #[test]
    fn cannot_seal_handoff_from_wrong_active_surface() {
        let mut g = SurfaceGraph::open("ses_x");
        let err = g
            .create_handoff(
                HandoffKind::YouToChat,
                1,
                vec![],
                vec![],
                json!({}),
                "user",
            )
            .unwrap_err();
 assert!( err.to_string().contains("active surface"), "wrong origin refused: {err}" );
    }
}

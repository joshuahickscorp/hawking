//! Shared surface graph for YOU / CHAT / IDE lenses on one session.
//!
//! Doctrine: the three surfaces are lenses over one HIDE session. They share
//! session identity, the durable event log, memory, objects, and connectors.
//! They do not each own a copy. A handoff capsule carries a CLAIM, never a
//! CAPABILITY.
//!
//! This module is the host-side holder of [`crate::lenses::SurfaceGraph`]. It is
//! model-free: seal/receive/switch only. No connector credentials, no inference.

use crate::lenses::{
    Claim, DeliberateExclusion, EvidenceTier, HandoffCapsule, HandoffKind, Surface, SurfaceGraph,
    SurfaceGraphView,
};
use hide_core::api::{UiEvent, UiEventKind};
use hide_core::event::NewEvent;
use hide_core::ids::SessionId;
use hide_core::persistence::DynEventLog;
use hide_core::Result;
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::sync::Arc;

use crate::ui_bus::UiEventBus;

/// Projection name the FE folds for the three-surface navigation state.
pub const SURFACE_GRAPH_PROJECTION: &str = "surface_graph";

/// Host-owned graph for the primary session. Surfaces call through intents;
/// they never construct a second graph.
pub struct SurfaceGraphService {
    graph: Mutex<SurfaceGraph>,
    events: DynEventLog,
    ui_bus: Arc<UiEventBus>,
}

impl SurfaceGraphService {
    /// Bind the graph to the host's primary session id (one identity for all lenses).
    pub fn for_session(
        session_id: &SessionId,
        events: DynEventLog,
        ui_bus: Arc<UiEventBus>,
    ) -> Self {
        Self {
            graph: Mutex::new(SurfaceGraph::open(session_id.as_str())),
            events,
            ui_bus,
        }
    }

    pub fn session_id(&self) -> String {
        self.graph.lock().session_id().to_string()
    }

    pub fn view(&self) -> SurfaceGraphView {
        self.graph.lock().view()
    }

    pub fn active(&self) -> Surface {
        self.graph.lock().active()
    }

    /// Publish the surface_graph projection so every FE surface reads the same state.
    pub fn publish_view(&self) {
        let view = self.view();
        let session = SessionId::from(view.session_id.as_str());
        self.ui_bus.publish(UiEvent {
            seq: 0,
            session_id: Some(session),
            kind: UiEventKind::ProjectionPatch {
                projection: SURFACE_GRAPH_PROJECTION.into(),
                patch: serde_json::to_value(&view).unwrap_or_else(|_| json!({})),
            },
        });
    }

    /// Switch the active lens. Same session id. No capability change.
    pub fn switch_surface(&self, surface: Surface) -> Result<SurfaceGraphView> {
        {
            let mut g = self.graph.lock();
            g.switch(surface);
        }
        self.publish_view();
        Ok(self.view())
    }

    /// Parse a surface name from wire payload (`you` / `chat` / `ide`).
    pub fn parse_surface(name: &str) -> std::result::Result<Surface, String> {
        match name {
            "you" => Ok(Surface::You),
            "chat" => Ok(Surface::Chat),
            "ide" | "code" => Ok(Surface::Ide),
            other => Err(format!("unknown surface '{other}'; expected you|chat|ide")),
        }
    }

    pub fn parse_kind(name: &str) -> std::result::Result<HandoffKind, String> {
        match name {
            "you_to_chat" => Ok(HandoffKind::YouToChat),
            "chat_to_ide" => Ok(HandoffKind::ChatToIde),
            "ide_to_you" => Ok(HandoffKind::IdeToYou),
            other => Err(format!(
                "unknown handoff kind '{other}'; expected you_to_chat|chat_to_ide|ide_to_you"
            )),
        }
    }

    /// Seal a typed handoff from the active surface. Emits `you.handoff.created`
    /// on the existing durable event log (not a second bus). Capsule carries
    /// claims only; host refuses any attempt path that would widen capability.
    pub async fn handoff_create(
        &self,
        kind: HandoffKind,
        claims: Vec<Claim>,
        exclusions: Vec<DeliberateExclusion>,
        body: Value,
        actor: &str,
        now_ms: u64,
    ) -> Result<HandoffCapsule> {
        let capsule = {
            let mut g = self.graph.lock();
            g.create_handoff(kind, now_ms, claims, exclusions, body, actor)
                .map_err(|e| hide_core::error::HideError::PolicyDenied(e.to_string()))?
        };

        // Fail-closed type boundary before anything hits the log.
        if capsule.try_extract_capability().is_ok() {
            return Err(hide_core::error::HideError::PolicyDenied(
                "capsule carried capability; refused before log append".into(),
            ));
        }

        let session = SessionId::from(capsule.origin_session.as_str());
        let payload = json!({
            "from_surface": capsule.origin_surface.as_str(),
            "to_surface": capsule.target_surface.as_str(),
            "capsule_hash": capsule.content_hash,
            "capsule_id": capsule.id,
            "what_it_excludes": capsule.deliberately_excludes.iter().map(|e| {
                json!({"item": e.item, "reason": e.reason})
            }).collect::<Vec<_>>(),
            "claim_count": capsule.claims.len(),
            "kind": capsule.kind.as_str(),
            // Audit snapshot only. Not a grant.
            "permissions_at_creation": {
                "surface": capsule.permissions_at_creation.surface.as_str(),
                "tools": capsule.permissions_at_creation.tools,
                "connectors": capsule.permissions_at_creation.connectors,
            },
            "_canonical": {
                "schema": "hawking.events.canonical.v1",
                "surface": capsule.origin_surface.as_str(),
                "subsystem": "hide_you",
                "verification": "target_verified",
                "category": "you_handoff",
            }
        });
        let mut new = NewEvent::system(session.clone(), "you.handoff.created", payload);
        new.class = hide_core::event::EventClass::Action;
        new.actor = Some("hide_you".into());
        let _event = self.events.append(new).await?;

        self.publish_view();
        // Also a custom UiEvent so surfaces that only listen for handoff badges update.
        self.ui_bus.publish(UiEvent {
            seq: 0,
            session_id: Some(session),
            kind: UiEventKind::Custom(json!({
                "kind": "handoff_created",
                "capsule_id": capsule.id,
                "from_surface": capsule.origin_surface.as_str(),
                "to_surface": capsule.target_surface.as_str(),
                "capsule_hash": capsule.content_hash,
                "what_it_excludes": capsule.deliberately_excludes.iter().map(|e| {
                    json!({"item": e.item, "reason": e.reason})
                }).collect::<Vec<_>>(),
            })),
        });
        Ok(capsule)
    }

    /// Receive a sealed capsule into its target lens on the same session.
    /// Capability of the receiver is unchanged.
    pub async fn handoff_receive(&self, capsule_id: &str) -> Result<SurfaceGraphView> {
        let received = {
            let mut g = self.graph.lock();
            g.receive_handoff(capsule_id)
                .map_err(|e| hide_core::error::HideError::PolicyDenied(e.to_string()))?
        };
        if !received.capability_unchanged() {
            return Err(hide_core::error::HideError::PolicyDenied(
                "receive widened capability; refused".into(),
            ));
        }
        // Creator connectors described on the capsule must still be unusable
        // via the capsule itself (claim never capability).
        if let Some(conn) = received.opened.permissions_described.connectors.first() {
            let g = self.graph.lock();
            if let Some(cap) = g.capsule(capsule_id) {
                let _ = cap.try_use_creator_connector(conn);
            }
        }
        self.publish_view();
        let session = SessionId::from(self.session_id().as_str());
        self.ui_bus.publish(UiEvent {
            seq: 0,
            session_id: Some(session),
            kind: UiEventKind::Custom(json!({
                "kind": "handoff_received",
                "capsule_id": received.opened.capsule_id,
                "target_surface": received.opened.kind.to_surface().as_str(),
                "claim_count": received.opened.claims.len(),
                "capability_unchanged": true,
            })),
        });
        Ok(self.view())
    }
}

/// Build claims from a wire payload array. Default evidence tier is `asserted`
/// (lowest): an inference must not silently become a stronger tier.
pub fn claims_from_payload(raw: &Value) -> std::result::Result<Vec<Claim>, String> {
    let arr = raw
        .as_array()
        .ok_or_else(|| "claims must be an array".to_string())?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        let text = item
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("claims[{i}].text required"))?
            .to_string();
        let id = item
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("clm_{i}"));
        let tier = item
            .get("evidence_tier")
            .and_then(|v| v.as_str())
            .map(parse_evidence_tier)
            .transpose()?
            .unwrap_or(EvidenceTier::Asserted);
        let payload = item.get("payload").cloned().unwrap_or_else(|| json!({}));
        out.push(Claim {
            id,
            text,
            evidence_tier: tier,
            payload,
        });
    }
    Ok(out)
}

pub fn exclusions_from_payload(
    raw: &Value,
) -> std::result::Result<Vec<DeliberateExclusion>, String> {
    let arr = match raw.as_array() {
        Some(a) => a,
        None if raw.is_null() => return Ok(Vec::new()),
        None => return Err("deliberately_excludes must be an array".into()),
    };
    let mut out = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        let item_s = item
            .get("item")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("deliberately_excludes[{i}].item required"))?
            .to_string();
        let reason = item
            .get("reason")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("deliberately_excludes[{i}].reason required"))?
            .to_string();
        out.push(DeliberateExclusion {
            item: item_s,
            reason,
        });
    }
    Ok(out)
}

fn parse_evidence_tier(s: &str) -> std::result::Result<EvidenceTier, String> {
    match s {
        "asserted" => Ok(EvidenceTier::Asserted),
        "cited" => Ok(EvidenceTier::Cited),
        "independently_verified" => Ok(EvidenceTier::IndependentlyVerified),
        "reproduced" => Ok(EvidenceTier::Reproduced),
        other => Err(format!("unknown evidence_tier '{other}'")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::event::InMemoryEventLog;
    use hide_core::ids::with_deterministic_ids;
    use std::sync::Arc;
    fn service() -> SurfaceGraphService {
        let events: DynEventLog = Arc::new(InMemoryEventLog::new());
        let ui = Arc::new(UiEventBus::default());
        SurfaceGraphService::for_session(&SessionId::from("ses_test"), events, ui)
    }
    #[test]
    fn lenses_share_primary_session() {
        with_deterministic_ids(1, || {
            let s = service();
            assert_eq!(s.session_id(), "ses_test");
            let v = s.view();
            assert_eq!(v.session_id, "ses_test");
            assert!(v.lenses.contains_key("you"));
            assert!(v.lenses.contains_key("chat"));
            assert!(v.lenses.contains_key("ide"));
            assert!(v.lenses["you"].connectors.iter().any(|c| c == "gmail"));
            assert!(!v.lenses["chat"].connectors.iter().any(|c| c == "gmail"));
        });
    }
    #[tokio::test]
    async fn handoff_does_not_widen_chat_capability() {
        let s = with_deterministic_ids(2, service);
        s.switch_surface(Surface::You).unwrap();
        let capsule = s
            .handoff_create(
                HandoffKind::YouToChat,
                vec![Claim {
                    id: "c1".into(),
                    text: "implement feature".into(),
                    evidence_tier: EvidenceTier::Cited,
                    payload: json!({}),
                }],
                vec![DeliberateExclusion {
                    item: "vault".into(),
                    reason: "claim only".into(),
                }],
                json!({"kind": "implementation_campaign"}),
                "test",
                1_000,
            )
            .await
            .expect("create");
        assert!(capsule.try_extract_capability().is_err());
        let view = s.handoff_receive(&capsule.id).await.expect("receive");
        assert!(!view.lenses["chat"].connectors.iter().any(|c| c == "gmail"));
        assert!(view.lenses["you"].connectors.iter().any(|c| c == "gmail"));
        assert_eq!(view.inbox["chat"].len(), 1);
        assert_eq!(view.session_id, "ses_test");
    }
}

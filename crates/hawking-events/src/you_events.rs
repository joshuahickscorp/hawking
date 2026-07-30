//! Seventeen YOU surface events that join the **same** canonical bus.
//!
//! Authority: [`evidence/hide/HIDE_YOU_EVENTS_AND_FRONTEND_CONTRACT.json`]. These are not a
//! second event bus — they are named open kinds on [`hide_core::event::Event`]
//! with the full envelope (id, seq, session, surface, subsystem, verification).
//!
//! The provisional flag is load-bearing: `hawking-speculate` has a Draft/Verified
//! type wall, and an event that loses that distinction can put a draft token in
//! a UI as though it were final.

use hide_core::event::EventClass;
use hide_core::ids::SessionId;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::categories::Category;
use crate::envelope::{
    CanonicalEvent, ContentVerification, NewCanonical, ProducingSurface, Subsystem,
};

/// One of the seventeen YOU product events from the sealed contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum YouEvent {
    ObjectAdded,
    ObjectProcessed,
    MemoryProposed,
    MemoryCommitted,
    MemoryCorrected,
    ConnectorRead,
    ConnectorWriteProposed,
    ResearchStarted,
    SourceCaptured,
    ClaimVerified,
    AutomationCreated,
    AutomationRan,
    SwarmCreated,
    AgentDelegated,
    AgentResult,
    ProjectUpdated,
    HandoffCreated,
}

/// Static description of a YOU event (kind, category, payload contract, default verification).
#[derive(Debug, Clone, Copy)]
pub struct YouEventSpec {
    pub event: YouEvent,
    /// Open dotted kind on the durable Event (e.g. `you.object.added`).
    pub kind: &'static str,
    pub category: Category,
    /// Fields the payload must carry (contract language).
    pub carries: &'static [&'static str],
    /// Contract note (may be empty).
    pub note: &'static str,
    /// Default verification when the emitter does not override.
    /// Proposed/draft-style events default to Provisional; commits default to TargetVerified.
    pub default_verification: ContentVerification,
    pub default_class: EventClass,
}

/// All seventeen YOU events, in contract order.
pub const YOU_EVENTS: &[YouEventSpec] = &[
    YouEventSpec {
        event: YouEvent::ObjectAdded,
        kind: "you.object.added",
        category: Category::YouObjects,
        carries: &["content_hash", "kind", "size", "source", "scope"],
        note: "the hash is the identity; the same bytes twice is one object with two references",
        default_verification: ContentVerification::TargetVerified,
        default_class: EventClass::Observation,
    },
    YouEventSpec {
        event: YouEvent::ObjectProcessed,
        kind: "you.object.processed",
        category: Category::YouObjects,
        carries: &[
            "object_hash",
            "stage",
            "derived_representations",
            "what_remains",
        ],
        note: "",
        default_verification: ContentVerification::TargetVerified,
        default_class: EventClass::Observation,
    },
    YouEventSpec {
        event: YouEvent::MemoryProposed,
        kind: "you.memory.proposed",
        category: Category::YouMemory,
        carries: &["candidate_record", "class", "scope", "provenance"],
        note: "proposed is not committed; this is what makes memory inspectable BEFORE it lands",
        default_verification: ContentVerification::Provisional,
        default_class: EventClass::Action,
    },
    YouEventSpec {
        event: YouEvent::MemoryCommitted,
        kind: "you.memory.committed",
        category: Category::YouMemory,
        carries: &["record_id", "class", "scope", "approver"],
        note: "",
        default_verification: ContentVerification::TargetVerified,
        default_class: EventClass::Observation,
    },
    YouEventSpec {
        event: YouEvent::MemoryCorrected,
        kind: "you.memory.corrected",
        category: Category::YouMemory,
        carries: &["old_id", "new_id", "reason"],
        note: "correction is supersession; both remain until forgotten",
        default_verification: ContentVerification::TargetVerified,
        default_class: EventClass::Observation,
    },
    YouEventSpec {
        event: YouEvent::ConnectorRead,
        kind: "you.connector.read",
        category: Category::YouConnectors,
        carries: &["connector_id", "account_handle", "scope_read", "object_count"],
        note: "",
        default_verification: ContentVerification::TargetVerified,
        default_class: EventClass::Observation,
    },
    YouEventSpec {
        event: YouEvent::ConnectorWriteProposed,
        kind: "you.connector.write_proposed",
        category: Category::YouConnectors,
        carries: &["connector_id", "intended_effect", "permission_required"],
        note: "proposed, never executed by the event",
        default_verification: ContentVerification::Provisional,
        default_class: EventClass::Action,
    },
    YouEventSpec {
        event: YouEvent::ResearchStarted,
        kind: "you.research.started",
        category: Category::YouResearch,
        carries: &["question", "mode", "budget"],
        note: "",
        default_verification: ContentVerification::Provisional,
        default_class: EventClass::Action,
    },
    YouEventSpec {
        event: YouEvent::SourceCaptured,
        kind: "you.research.source_captured",
        category: Category::YouResearch,
        carries: &["url_or_fixture_id", "captured_at", "quality_grade", "content_hash"],
        note: "",
        default_verification: ContentVerification::TargetVerified,
        default_class: EventClass::Observation,
    },
    YouEventSpec {
        event: YouEvent::ClaimVerified,
        kind: "you.research.claim_verified",
        category: Category::YouResearch,
        carries: &["claim_id", "category", "evidence_ids", "verifier"],
        note: "",
        default_verification: ContentVerification::TargetVerified,
        default_class: EventClass::Observation,
    },
    YouEventSpec {
        event: YouEvent::AutomationCreated,
        kind: "you.automation.created",
        category: Category::YouAutomation,
        carries: &["trigger", "goal", "permission_set", "budget", "stop_condition"],
        note: "",
        default_verification: ContentVerification::TargetVerified,
        default_class: EventClass::Action,
    },
    YouEventSpec {
        event: YouEvent::AutomationRan,
        kind: "you.automation.ran",
        category: Category::YouAutomation,
        carries: &["automation_id", "outcome", "resources_consumed", "next_run"],
        note: "",
        default_verification: ContentVerification::TargetVerified,
        default_class: EventClass::Observation,
    },
    YouEventSpec {
        event: YouEvent::SwarmCreated,
        kind: "you.swarm.created",
        category: Category::YouSwarm,
        carries: &["goal", "roles", "budget", "stop_rules"],
        note: "",
        default_verification: ContentVerification::TargetVerified,
        default_class: EventClass::Action,
    },
    YouEventSpec {
        event: YouEvent::AgentDelegated,
        kind: "you.agent.delegated",
        category: Category::YouSwarm,
        carries: &[
            "agent_id",
            "role",
            "context_capsule_hash",
            "permissions",
            "deadline",
        ],
        note: "",
        default_verification: ContentVerification::Provisional,
        default_class: EventClass::Action,
    },
    YouEventSpec {
        event: YouEvent::AgentResult,
        kind: "you.agent.result",
        category: Category::YouSwarm,
        carries: &[
            "agent_id",
            "output_schema_conformance",
            "verification_state",
        ],
        note: "",
        // Result may still be provisional until promotion; default provisional
        // forces UIs to treat it as non-final unless explicitly verified.
        default_verification: ContentVerification::Provisional,
        default_class: EventClass::Observation,
    },
    YouEventSpec {
        event: YouEvent::ProjectUpdated,
        kind: "you.project.updated",
        category: Category::YouProjects,
        carries: &["project_id", "state_transition", "what_changed"],
        note: "",
        default_verification: ContentVerification::TargetVerified,
        default_class: EventClass::Observation,
    },
    YouEventSpec {
        event: YouEvent::HandoffCreated,
        kind: "you.handoff.created",
        category: Category::YouHandoff,
        carries: &["from_surface", "to_surface", "capsule_hash", "what_it_excludes"],
        note: "",
        default_verification: ContentVerification::TargetVerified,
        default_class: EventClass::Action,
    },
];

impl YouEvent {
    pub fn as_pascal(self) -> &'static str {
        match self {
            YouEvent::ObjectAdded => "ObjectAdded",
            YouEvent::ObjectProcessed => "ObjectProcessed",
            YouEvent::MemoryProposed => "MemoryProposed",
            YouEvent::MemoryCommitted => "MemoryCommitted",
            YouEvent::MemoryCorrected => "MemoryCorrected",
            YouEvent::ConnectorRead => "ConnectorRead",
            YouEvent::ConnectorWriteProposed => "ConnectorWriteProposed",
            YouEvent::ResearchStarted => "ResearchStarted",
            YouEvent::SourceCaptured => "SourceCaptured",
            YouEvent::ClaimVerified => "ClaimVerified",
            YouEvent::AutomationCreated => "AutomationCreated",
            YouEvent::AutomationRan => "AutomationRan",
            YouEvent::SwarmCreated => "SwarmCreated",
            YouEvent::AgentDelegated => "AgentDelegated",
            YouEvent::AgentResult => "AgentResult",
            YouEvent::ProjectUpdated => "ProjectUpdated",
            YouEvent::HandoffCreated => "HandoffCreated",
        }
    }

    pub fn spec(self) -> &'static YouEventSpec {
        YOU_EVENTS
            .iter()
            .find(|s| s.event == self)
            .expect("every YouEvent has a YOU_EVENTS entry")
    }

    pub fn kind(self) -> &'static str {
        self.spec().kind
    }

    pub fn parse(name: &str) -> Option<Self> {
        YOU_EVENTS
            .iter()
            .find(|s| s.event.as_pascal() == name || s.kind == name)
            .map(|s| s.event)
    }
}

impl YouEventSpec {
    /// Build a not-yet-sequenced canonical event for this YOU kind.
    ///
    /// Always stamps producing surface = YOU (unless overridden) and the
    /// HideYou subsystem. Pass an explicit [`ContentVerification`] when the
    /// emitter knows more than the default (e.g. promoting AgentResult).
    pub fn to_new_canonical(
        &self,
        session_id: SessionId,
        payload: Value,
        verification: Option<ContentVerification>,
    ) -> NewCanonical {
        NewCanonical::new(
            session_id,
            Subsystem::HideYou,
            verification.unwrap_or(self.default_verification),
            self.category,
            payload,
        )
        .with_surface(ProducingSurface::You)
        .with_class(self.default_class)
        .with_kind(self.kind)
    }

    /// Sequence a YOU event for tests / pure construction.
    pub fn sequence(
        &self,
        seq: u64,
        session_id: SessionId,
        payload: Value,
        verification: Option<ContentVerification>,
    ) -> CanonicalEvent {
        CanonicalEvent::sequence(seq, self.to_new_canonical(session_id, payload, verification))
    }
}

/// JSON export fragment for HAWKING_CANONICAL_EVENTS.json.
pub fn you_events_export() -> Value {
    let events: Vec<Value> = YOU_EVENTS
        .iter()
        .map(|s| {
            json!({
                "event": s.event.as_pascal(),
                "kind": s.kind,
                "category": s.category.as_str(),
                "carries": s.carries,
                "note": if s.note.is_empty() { Value::Null } else { Value::String(s.note.into()) },
                "default_verification": s.default_verification.as_str(),
                "default_class": match s.default_class {
                    EventClass::Action => "action",
                    EventClass::Observation => "observation",
                    EventClass::Neither => "neither",
                },
            })
        })
        .collect();
    json!({
        "law": "YOU events join the EXISTING canonical model. They are not a second bus.",
        "count": YOU_EVENTS.len(),
        "every_event_carries": [
            "stable id",
            "monotone sequence",
            "session identity",
            "producing surface",
            "producing subsystem",
            "whether its content is target-verified or provisional"
        ],
        "provisional_flag_matters_because": "speculation is default-off but the Draft/Verified type wall already exists in hawking-speculate. An event carrying provisional content must be visibly provisional, or a draft token reaches a UI as though it were final.",
        "events": events,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::ids::with_deterministic_ids;
    #[test]
    fn seventeen_you_events() {
        assert_eq!(YOU_EVENTS.len(), 17);
        let mut kinds = std::collections::BTreeSet::new();
        for s in YOU_EVENTS {
            assert!(kinds.insert(s.kind), "duplicate kind {}", s.kind);
            assert!(!s.carries.is_empty());
        }
    }
    #[test]
    fn proposed_defaults_provisional_committed_verified() {
        assert_eq!(YouEvent::MemoryProposed.spec().default_verification, ContentVerification::Provisional);
        assert_eq!(YouEvent::MemoryCommitted.spec().default_verification, ContentVerification::TargetVerified);
        assert_eq!(YouEvent::ConnectorWriteProposed.spec().default_verification, ContentVerification::Provisional);
 assert_eq!( YouEvent::AgentResult.spec().default_verification, ContentVerification::Provisional );
    }
    #[test]
    fn envelope_carries_surface_and_verification() {
        with_deterministic_ids(42, || {
            let c = YouEvent::ObjectAdded.spec().sequence(
                1,
                SessionId::from("ses_you"),
                json!({
                    "content_hash": "abc",
                    "kind": "image",
                    "size": 12,
                    "source": "upload",
                    "scope": "session",
                }),
                None,
            );
            assert_eq!(c.surface, ProducingSurface::You);
            assert_eq!(c.subsystem, Subsystem::HideYou);
            assert_eq!(c.verification, ContentVerification::TargetVerified);
            assert_eq!(c.kind(), "you.object.added");
            assert!(!c.id().as_str().is_empty());
            assert_eq!(c.seq(), 1);
            assert_eq!(c.session_id().as_str(), "ses_you");
        });
    }
}

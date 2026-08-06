//! Canonical envelope around [`hide_core::event::Event`].
//!
//! The durable bytes are always a hide-core `Event`. This module is the only
//! place that *requires* the extra fields the contract demands (surface +
//! subsystem + verification) so emitters cannot forget them.

use hide_core::event::{Event, EventClass, EventSource, NewEvent};
use hide_core::ids::{EventId, RunId, SessionId};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

use crate::categories::{category_for_kind, kind_for_category, Category};

/// Schema marker embedded on every canonical event under `ext.canonical_schema`.
pub const CANONICAL_SCHEMA: &str = "hawking.events.canonical.v1";

/// Whether the content is target-verified or still provisional.
///
/// Load-bearing for the Draft/Verified type wall in `hawking-speculate`: an
/// event that loses this distinction can put a draft token in a UI as final.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentVerification {
    /// Agrees with a named target / oracle / parent reference.
    TargetVerified,
    /// Best-effort or speculative; must not be treated as sealed truth.
    Provisional,
}

impl ContentVerification {
    pub fn as_str(self) -> &'static str {
        match self {
            ContentVerification::TargetVerified => "target_verified",
            ContentVerification::Provisional => "provisional",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "target_verified" => Some(ContentVerification::TargetVerified),
            "provisional" => Some(ContentVerification::Provisional),
            _ => None,
        }
    }
}

/// Product surface that produced the event (YOU / CHAT / IDE / bridge / …).
///
/// Distinct from [`Subsystem`]: surface is the user-facing lens; subsystem is
/// the code path. Both are required on every canonical event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProducingSurface {
    You,
    Chat,
    Ide,
    Bridge,
    Terminal,
    Mcp,
    Sdk,
    Serve,
    System,
    Other(String),
}

impl ProducingSurface {
    pub fn as_str(&self) -> String {
        match self {
            ProducingSurface::You => "you".into(),
            ProducingSurface::Chat => "chat".into(),
            ProducingSurface::Ide => "ide".into(),
            ProducingSurface::Bridge => "bridge".into(),
            ProducingSurface::Terminal => "terminal".into(),
            ProducingSurface::Mcp => "mcp".into(),
            ProducingSurface::Sdk => "sdk".into(),
            ProducingSurface::Serve => "serve".into(),
            ProducingSurface::System => "system".into(),
            ProducingSurface::Other(s) => s.clone(),
        }
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "you" => ProducingSurface::You,
            "chat" => ProducingSurface::Chat,
            "ide" => ProducingSurface::Ide,
            "bridge" => ProducingSurface::Bridge,
            "terminal" => ProducingSurface::Terminal,
            "mcp" => ProducingSurface::Mcp,
            "sdk" => ProducingSurface::Sdk,
            "serve" => ProducingSurface::Serve,
            "system" => ProducingSurface::System,
            other => ProducingSurface::Other(other.to_string()),
        }
    }
}

/// Producing subsystem (finer than [`EventSource`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Subsystem {
    Serve,
    CoreEngine,
    Gravity,
    HideBackend,
    HideKernel,
    HideFleet,
    HideYou,
    SeedC,
    Speculate,
    Orch,
    Bridge,
    Fabric,
    Adapter(String),
    Other(String),
}

impl Subsystem {
    pub fn as_str(&self) -> String {
        match self {
            Subsystem::Serve => "serve".into(),
            Subsystem::CoreEngine => "core_engine".into(),
            Subsystem::Gravity => "gravity".into(),
            Subsystem::HideBackend => "hide_backend".into(),
            Subsystem::HideKernel => "hide_kernel".into(),
            Subsystem::HideFleet => "hide_fleet".into(),
            Subsystem::HideYou => "hide_you".into(),
            Subsystem::SeedC => "seed_c".into(),
            Subsystem::Speculate => "speculate".into(),
            Subsystem::Orch => "orch".into(),
            Subsystem::Bridge => "bridge".into(),
            Subsystem::Fabric => "fabric".into(),
            Subsystem::Adapter(s) => format!("adapter:{s}"),
            Subsystem::Other(s) => s.clone(),
        }
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "serve" => Subsystem::Serve,
            "core_engine" => Subsystem::CoreEngine,
            "gravity" => Subsystem::Gravity,
            "hide_backend" => Subsystem::HideBackend,
            "hide_kernel" => Subsystem::HideKernel,
            "hide_fleet" => Subsystem::HideFleet,
            "hide_you" => Subsystem::HideYou,
            "seed_c" => Subsystem::SeedC,
            "speculate" => Subsystem::Speculate,
            "orch" => Subsystem::Orch,
            "bridge" => Subsystem::Bridge,
            "fabric" => Subsystem::Fabric,
            other if other.starts_with("adapter:") => {
                Subsystem::Adapter(other.trim_start_matches("adapter:").to_string())
            }
            other => Subsystem::Other(other.to_string()),
        }
    }

    fn to_event_source(&self) -> EventSource {
        match self {
            Subsystem::Serve
            | Subsystem::CoreEngine
            | Subsystem::Gravity
            | Subsystem::Speculate => EventSource::Runtime,
            Subsystem::HideBackend
            | Subsystem::HideKernel
            | Subsystem::HideFleet
            | Subsystem::HideYou => EventSource::System,
            Subsystem::SeedC | Subsystem::Orch | Subsystem::Bridge | Subsystem::Fabric => {
                EventSource::System
            }
            Subsystem::Adapter(_) => EventSource::Runtime,
            Subsystem::Other(_) => EventSource::System,
        }
    }

    /// Default product surface for events from this subsystem when the emitter
    /// does not set one explicitly.
    pub fn default_surface(&self) -> ProducingSurface {
        match self {
            Subsystem::HideYou => ProducingSurface::You,
            Subsystem::Bridge => ProducingSurface::Bridge,
            Subsystem::Serve | Subsystem::CoreEngine | Subsystem::Gravity => {
                ProducingSurface::Serve
            }
            Subsystem::HideBackend | Subsystem::HideKernel | Subsystem::HideFleet => {
                ProducingSurface::System
            }
            _ => ProducingSurface::System,
        }
    }
}

/// Input for building a not-yet-sequenced canonical event.
#[derive(Debug, Clone)]
pub struct NewCanonical {
    pub session_id: SessionId,
    pub run_id: Option<RunId>,
    pub parent: Option<EventId>,
    pub cause: Option<EventId>,
    pub surface: ProducingSurface,
    pub subsystem: Subsystem,
    pub verification: ContentVerification,
    pub category: Category,
    pub class: EventClass,
    pub payload: Value,
    /// Optional override of the primary kind for the category.
    pub kind_override: Option<String>,
}

impl NewCanonical {
    pub fn new(
        session_id: SessionId,
        subsystem: Subsystem,
        verification: ContentVerification,
        category: Category,
        payload: Value,
    ) -> Self {
        let surface = subsystem.default_surface();
        Self {
            session_id,
            run_id: None,
            parent: None,
            cause: None,
            surface,
            subsystem,
            verification,
            category,
            class: EventClass::Neither,
            payload,
            kind_override: None,
        }
    }

    pub fn with_run(mut self, run_id: RunId) -> Self {
        self.run_id = Some(run_id);
        self
    }

    pub fn with_class(mut self, class: EventClass) -> Self {
        self.class = class;
        self
    }

    pub fn with_kind(mut self, kind: impl Into<String>) -> Self {
        self.kind_override = Some(kind.into());
        self
    }

    pub fn with_surface(mut self, surface: ProducingSurface) -> Self {
        self.surface = surface;
        self
    }

    /// Build the open [`NewEvent`] that the durable log will sequence.
    pub fn into_new_event(self) -> NewEvent {
        let kind = self
            .kind_override
            .unwrap_or_else(|| kind_for_category(self.category).to_string());
        let subsystem = self.subsystem.as_str();
        let surface = self.surface.as_str();
        let source = self.subsystem.to_event_source();
        let mut new = NewEvent {
            session_id: self.session_id,
            run_id: self.run_id,
            parent: self.parent,
            cause: self.cause,
            source,
            actor: Some(subsystem.clone()),
            class: self.class,
            kind,
            payload: self.payload,
            redactions: Vec::new(),
        };
        // Attach envelope markers into payload so they survive open-payload storage.
        let meta = json!({
            "schema": CANONICAL_SCHEMA,
            "surface": surface,
            "subsystem": subsystem,
            "verification": self.verification.as_str(),
            "category": self.category.as_str(),
        });
        if let Value::Object(ref mut map) = new.payload {
            map.insert("_canonical".into(), meta);
        } else {
            new.payload = json!({
                "body": new.payload,
                "_canonical": meta,
            });
        }
        new
    }
}

/// A sequenced canonical event: hide-core `Event` + required envelope reads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalEvent {
    #[serde(flatten)]
    pub event: Event,
    pub surface: ProducingSurface,
    pub subsystem: Subsystem,
    pub verification: ContentVerification,
    pub category: Category,
}

impl CanonicalEvent {
    /// Wrap a sequenced Event, reading envelope fields from payload._canonical
    /// or from actor + ext fallbacks.
    pub fn from_sequenced(event: Event) -> Result<Self, String> {
        let (surface, subsystem, verification, category) = read_envelope(&event)?;
        Ok(Self {
            event,
            surface,
            subsystem,
            verification,
            category,
        })
    }

    /// Sequence via an in-process counter (for tests / pure construction).
    pub fn sequence(seq: u64, input: NewCanonical) -> Self {
        let new = input.into_new_event();
        let event = Event::new(seq, new);
        Self::from_sequenced(event).expect("NewCanonical always stamps a valid envelope")
    }

    pub fn id(&self) -> &EventId {
        &self.event.id
    }

    pub fn seq(&self) -> u64 {
        self.event.seq
    }

    pub fn session_id(&self) -> &SessionId {
        &self.event.session_id
    }

    pub fn kind(&self) -> &str {
        &self.event.kind
    }

    /// JSON value suitable for fixtures / export.
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("CanonicalEvent serializes")
    }

    /// Round-trip through JSON (serde).
    pub fn round_trip_json(&self) -> Result<Self, String> {
        let v = self.to_value();
        serde_json::from_value(v).map_err(|e| e.to_string())
    }
}

fn read_envelope(
    event: &Event,
) -> Result<(ProducingSurface, Subsystem, ContentVerification, Category), String> {
    if let Some(meta) = event.payload.get("_canonical") {
        let subsystem = meta
            .get("subsystem")
            .and_then(|v| v.as_str())
            .map(Subsystem::parse)
            .ok_or_else(|| "missing _canonical.subsystem".to_string())?;
        let surface = meta
            .get("surface")
            .and_then(|v| v.as_str())
            .map(ProducingSurface::parse)
            .unwrap_or_else(|| subsystem.default_surface());
        let verification = meta
            .get("verification")
            .and_then(|v| v.as_str())
            .and_then(ContentVerification::parse)
            .ok_or_else(|| "missing/invalid _canonical.verification".to_string())?;
        let category = meta
            .get("category")
            .and_then(|v| v.as_str())
            .and_then(|s| Category::all().iter().find(|c| c.as_str() == s).copied())
            .or_else(|| category_for_kind(&event.kind))
            .ok_or_else(|| format!("unknown category for kind {}", event.kind))?;
        return Ok((surface, subsystem, verification, category));
    }

    // Fallback for adapted legacy events that only set actor + kind.
    let subsystem = event
        .actor
        .as_deref()
        .map(Subsystem::parse)
        .unwrap_or(Subsystem::Other("unknown".into()));
    let surface = event
        .ext
        .get("surface")
        .and_then(|v| v.as_str())
        .map(ProducingSurface::parse)
        .unwrap_or_else(|| subsystem.default_surface());
    let verification = event
        .ext
        .get("verification")
        .and_then(|v| v.as_str())
        .and_then(ContentVerification::parse)
        .unwrap_or(ContentVerification::Provisional);
    let category = category_for_kind(&event.kind)
        .ok_or_else(|| format!("kind {} is not a known canonical category", event.kind))?;
    Ok((surface, subsystem, verification, category))
}

/// Stamp envelope ext markers onto an already-built Event (adapter path).
pub fn stamp_legacy(
    event: Event,
    subsystem: Subsystem,
    verification: ContentVerification,
    category: Category,
) -> CanonicalEvent {
    stamp_legacy_with_surface(
        event,
        subsystem.default_surface(),
        subsystem,
        verification,
        category,
    )
}

/// Stamp envelope with an explicit producing surface.
pub fn stamp_legacy_with_surface(
    mut event: Event,
    surface: ProducingSurface,
    subsystem: Subsystem,
    verification: ContentVerification,
    category: Category,
) -> CanonicalEvent {
    let subsystem_s = subsystem.as_str();
    let surface_s = surface.as_str();
    if event.actor.is_none() {
        event.actor = Some(subsystem_s.clone());
    }
    event.ext.insert(
        "verification".into(),
        Value::String(verification.as_str().into()),
    );
    event
        .ext
        .insert("surface".into(), Value::String(surface_s.clone()));
    event.ext.insert(
        "canonical_schema".into(),
        Value::String(CANONICAL_SCHEMA.into()),
    );
    let meta = json!({
        "schema": CANONICAL_SCHEMA,
        "surface": surface_s,
        "subsystem": subsystem_s,
        "verification": verification.as_str(),
        "category": category.as_str(),
    });
    match &mut event.payload {
        Value::Object(map) => {
            map.insert("_canonical".into(), meta);
        }
        other => {
            event.payload = json!({ "body": other.clone(), "_canonical": meta });
        }
    }
    CanonicalEvent {
        event,
        surface,
        subsystem,
        verification,
        category,
    }
}

/// Helper: empty BTreeMap for ext in tests.
pub fn empty_ext() -> BTreeMap<String, Value> {
    BTreeMap::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::ids::with_deterministic_ids;
    #[test]
    fn envelope_round_trips_every_required_field() {
        with_deterministic_ids(1, || {
            let session = SessionId::from("ses_test");
            let c = CanonicalEvent::sequence(
                7,
                NewCanonical::new(
                    session.clone(),
                    Subsystem::Serve,
                    ContentVerification::TargetVerified,
                    Category::Text,
                    json!({ "text": "hi" }),
                )
                .with_surface(ProducingSurface::Serve),
            );
            assert_eq!(c.seq(), 7);
            assert_eq!(c.session_id(), &session);
            assert!(!c.id().as_str().is_empty());
            assert_eq!(c.subsystem, Subsystem::Serve);
            assert_eq!(c.surface, ProducingSurface::Serve);
            assert_eq!(c.verification, ContentVerification::TargetVerified);
            assert_eq!(c.category, Category::Text);
            assert_eq!(c.kind(), "model.token");
            let again = c.round_trip_json().unwrap();
            assert_eq!(again.seq(), 7);
            assert_eq!(again.verification, ContentVerification::TargetVerified);
            assert_eq!(again.subsystem.as_str(), "serve");
            assert_eq!(again.surface.as_str(), "serve");
        });
    }
}

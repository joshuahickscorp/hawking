//! Typed transport contracts for the Hawking Web boundary.
//!
//! This crate intentionally contains DTOs and bounded event projection only.
//! The live Python `hawkingd`/Goal/DAG/event owners remain authoritative during
//! the migration. There is no database, scheduler, provider client, writable
//! mirror, or second HTTP listener here. A future adapter can serialize these
//! views from the existing authorities and can roll back to the current
//! Python surface without migrating domain state.

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const WEB_SCHEMA: &str = "hawking.web.contract.v1";
pub const API_VERSION: u16 = 1;
pub const MAX_REPLAY_EVENTS: usize = 256;

/// A request-scoped replay cursor. It is not a domain revision and cannot be
/// used to authorize a mutation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateRevision {
    pub value: String,
}

impl StateRevision {
    pub fn new(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
        }
    }
}

/// Client retry identity. The Web adapter may use this to deduplicate a
/// request at the existing authority; the type itself performs no mutation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdempotencyKey(pub String);

impl IdempotencyKey {
    pub fn new(value: impl Into<String>) -> Result<Self, &'static str> {
        let value = value.into();
        if value.is_empty()
            || value.len() > 160
            || !value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b':' | b'-'))
        {
            return Err("idempotency key must be 1..=160 ASCII identifier characters");
        }
        Ok(Self(value))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionView {
    pub schema: String,
    pub session_id: String,
    pub conversation_id: String,
    pub source_scope: String,
    pub selection: String,
    pub auto_enabled: bool,
    pub active_goal_ids: Vec<String>,
    #[serde(default)]
    pub turn_count: u64,
    #[serde(default)]
    pub latest_message_seq: u64,
    pub state_revision: StateRevision,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoalView {
    pub schema: String,
    pub goal_id: String,
    pub objective: String,
    pub status: String,
    pub phase: Option<String>,
    pub workunit_ids: Vec<String>,
    pub worker_ids: Vec<String>,
    pub budget: BudgetView,
    pub recent_event_ids: Vec<String>,
    pub result_artifact_ids: Vec<String>,
    pub state_revision: StateRevision,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkUnitView {
    pub schema: String,
    pub workunit_id: String,
    pub goal_id: String,
    pub objective: String,
    pub role: String,
    pub status: String,
    pub worker_id: Option<String>,
    pub worker_attempt_id: Option<String>,
    pub evidence_ids: Vec<String>,
    pub state_revision: StateRevision,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerView {
    pub schema: String,
    pub worker_id: String,
    pub worker_attempt_id: String,
    pub goal_id: String,
    pub workunit_id: String,
    pub model: String,
    pub provider: Option<String>,
    pub status: String,
    pub capability_ids: Vec<String>,
    pub cost_usd: Option<f64>,
    pub state_revision: StateRevision,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityView {
    pub schema: String,
    pub capability_id: String,
    pub schema_version: String,
    pub effects: Vec<String>,
    pub available: bool,
    pub health: String,
    pub expansion_route: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactView {
    pub schema: String,
    pub artifact_id: String,
    pub kind: String,
    pub content_sha256: Option<String>,
    pub path: Option<String>,
    pub redacted: bool,
    pub provenance_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteDecisionView {
    pub schema: String,
    pub decision_id: String,
    pub source_scope: String,
    pub selection: String,
    pub resolved_model: Option<String>,
    pub worker_id: Option<String>,
    pub rationale: String,
    pub policy_revision: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetView {
    pub cap_usd: Option<f64>,
    pub spent_usd: f64,
    pub reserved_usd: f64,
    pub remaining_usd: Option<f64>,
    pub authority: String,
}

/// A compact, replayable projection. `payload` must contain evidence or state
/// references, not raw model reasoning, secrets, or unbounded logs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WebEvent {
    pub schema: String,
    pub sequence: u64,
    pub event_id: String,
    /// Canonical object identity.  The browser may use it to route a view
    /// update but never to manufacture a domain event.
    #[serde(default)]
    pub identity: Value,
    #[serde(rename = "type", alias = "kind")]
    pub event_type: String,
    pub session_id: String,
    pub goal_id: Option<String>,
    pub workunit_id: Option<String>,
    pub worker_id: Option<String>,
    #[serde(rename = "source_revision", alias = "state_revision")]
    pub source_revision: StateRevision,
    #[serde(rename = "timestamp", alias = "emitted_at")]
    pub timestamp: String,
    #[serde(default = "default_event_payload_version")]
    pub payload_version: u16,
    pub terminal: bool,
    pub payload: Value,
}

fn default_event_payload_version() -> u16 {
    1
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayView {
    pub schema: String,
    pub from_sequence: u64,
    pub through_sequence: u64,
    pub events: Vec<WebEvent>,
    pub gap: bool,
    pub snapshot_required: bool,
    pub max_events: usize,
}

/// Replay at most `max_events`, retaining terminal/error/budget events when a
/// stream is too noisy. The caller supplies already-authorized events from the
/// canonical event owner; this function never reads or writes a store.
pub fn replay_since(events: &[WebEvent], last_sequence: u64, max_events: usize) -> ReplayView {
    let cap = max_events.clamp(1, MAX_REPLAY_EVENTS);
    let mut eligible: Vec<WebEvent> = events
        .iter()
        .filter(|event| event.sequence > last_sequence)
        .cloned()
        .collect();
    eligible.sort_by_key(|event| event.sequence);
    let gap = eligible
        .first()
        .map(|event| event.sequence > last_sequence.saturating_add(1))
        .unwrap_or(false);
    let through_sequence = eligible
        .last()
        .map(|event| event.sequence)
        .unwrap_or(last_sequence);
    let snapshot_required = gap || eligible.len() > cap;
    if eligible.len() > cap {
        let mut retained: Vec<WebEvent> = eligible
            .iter()
            .filter(|event| {
                event.terminal
                    || event.event_type.ends_with("error")
                    || event.event_type == "budget.rejected"
            })
            .cloned()
            .collect();
        let reserve = retained.len().min(cap);
        retained.truncate(reserve);
        let mut seen: std::collections::BTreeSet<u64> =
            retained.iter().map(|event| event.sequence).collect();
        for event in eligible.into_iter().rev() {
            if seen.len() >= cap {
                break;
            }
            if seen.insert(event.sequence) {
                retained.push(event);
            }
        }
        retained.sort_by_key(|event| event.sequence);
        eligible = retained;
    }
    ReplayView {
        schema: WEB_SCHEMA.into(),
        from_sequence: last_sequence,
        through_sequence,
        events: eligible,
        gap,
        snapshot_required,
        max_events: cap,
    }
}

/// Negotiate the highest supported version without silently accepting an
/// incompatible wire contract.
pub fn negotiate_api_version(requested: &[u16]) -> Option<u16> {
    requested
        .iter()
        .copied()
        .filter(|version| *version == API_VERSION)
        .max()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(sequence: u64, event_type: &str, terminal: bool) -> WebEvent {
        WebEvent {
            schema: WEB_SCHEMA.into(),
            sequence,
            event_id: format!("event-{sequence}"),
            identity: serde_json::json!({"session_id": "session-1"}),
            event_type: event_type.into(),
            session_id: "session-1".into(),
            goal_id: Some("goal-1".into()),
            workunit_id: None,
            worker_id: None,
            source_revision: StateRevision::new(format!("r-{sequence}")),
            timestamp: "2026-09-15T00:00:00Z".into(),
            payload_version: 1,
            terminal,
            payload: serde_json::json!({"evidence_id": format!("receipt-{sequence}")}),
        }
    }

    #[test]
    fn replay_reports_gaps_and_bounds_noisy_streams() {
        let events = vec![event(2, "progress", false), event(3, "goal.complete", true)];
        let replay = replay_since(&events, 0, 1);
        assert!(replay.gap);
        assert!(replay.snapshot_required);
        assert!(replay.events.iter().any(|item| item.terminal));
        assert!(replay.events.len() <= 1);
    }

    #[test]
    fn idempotency_and_version_negotiation_are_fail_closed() {
        assert!(IdempotencyKey::new("request-1").is_ok());
        assert!(IdempotencyKey::new("bad key").is_err());
        assert_eq!(negotiate_api_version(&[0, 1, 2]), Some(1));
        assert_eq!(negotiate_api_version(&[0, 2]), None);
    }
}

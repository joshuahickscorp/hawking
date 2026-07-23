//! Canonical event fixtures.
//!
//! A small, deterministic set of [`Notification`] and [`Item`] values, built
//! from `hide-protocol` types, that the compatibility tests round-trip through
//! serde. Because the fixtures are typed hide-protocol values, they cannot
//! encode a shape the protocol does not accept, and [`events_json`] renders them
//! into the golden artifact the frontend and external clients can pin against.
//!
//! One source: the typed fixtures below are the origin; the JSON bundle and its
//! golden are serialized from them, never maintained by hand.

use serde_json::{Map, Value};

use hide_protocol::ids::{
    ApprovalId, ItemId, PlanId, StepId, ToolCallId, ToolId, TurnId,
};
use hide_protocol::item::{
    AgentMessage, ApprovalRequest, Completion, Item, ItemKind, ToolCall, ToolResult, UserMessage,
};
use hide_protocol::model::{CompletionStatus, Risk};
use hide_protocol::plan::{Cost, Effect, Plan, PlanStep, RollbackBoundary, Scope};
use hide_protocol::protocol::Notification;

fn item(id: &str, seq: u64, kind: ItemKind) -> Item {
    Item {
        id: ItemId::from(id),
        turn: Some(TurnId::from("trn_1")),
        seq,
        kind,
        created_ms: 1_000 + seq,
    }
}

fn sample_plan() -> Plan {
    Plan {
        id: PlanId::from("pln_1"),
        goal: None,
        steps: vec![PlanStep {
            id: StepId::from("stp_1"),
            objective: "reproduce the flake".into(),
            dependencies: vec![],
            scope: Scope {
                paths: vec!["src/retry.ts".into()],
                network: false,
                description: Some("the retry module only".into()),
            },
            effects: vec![Effect::ReadFs, Effect::Shell],
            expected_artifacts: vec!["repro.log".into()],
            acceptance_oracle: None,
            rollback_boundary: RollbackBoundary::default(),
            cost: Cost {
                tokens: Some(1200),
                wall_ms: Some(30_000),
                usd_micros: None,
            },
            parallelizable: false,
        }],
        created_ms: 900,
    }
}

/// The canonical [`Item`] fixtures, each with a stable name.
pub fn item_fixtures() -> Vec<(&'static str, Item)> {
    vec![
        (
            "user_message",
            item(
                "itm_user",
                0,
                ItemKind::UserMessage(UserMessage {
                    text: "the retry test flakes on CI".into(),
                    attachments: vec![],
                }),
            ),
        ),
        (
            "agent_message",
            item(
                "itm_agent",
                1,
                ItemKind::AgentMessage(AgentMessage {
                    text: "reproducing it now".into(),
                }),
            ),
        ),
        (
            "plan",
            item("itm_plan", 2, ItemKind::Plan(sample_plan())),
        ),
        (
            "tool_call",
            item(
                "itm_call",
                3,
                ItemKind::ToolCall(ToolCall {
                    call_id: ToolCallId::from("tcl_1"),
                    tool: ToolId::from("tool_bash"),
                    arguments: serde_json::json!({ "cmd": "cargo test -p retry" }),
                }),
            ),
        ),
        (
            "tool_result",
            item(
                "itm_result",
                4,
                ItemKind::ToolResult(ToolResult {
                    call_id: ToolCallId::from("tcl_1"),
                    ok: true,
                    output: serde_json::json!({ "code": 0, "passed": 12 }),
                    error: None,
                }),
            ),
        ),
        (
            "completion",
            item(
                "itm_done",
                5,
                ItemKind::Completion(Completion {
                    status: CompletionStatus::Success,
                    summary: Some("retry test is green".into()),
                }),
            ),
        ),
    ]
}

/// The canonical [`Notification`] fixtures, each with a stable name.
pub fn notification_fixtures() -> Vec<(&'static str, Notification)> {
    vec![
        (
            "turn_started",
            Notification::TurnStarted {
                turn: TurnId::from("trn_1"),
            },
        ),
        (
            "item_added",
            Notification::ItemAdded {
                item: item(
                    "itm_agent",
                    1,
                    ItemKind::AgentMessage(AgentMessage {
                        text: "reproducing it now".into(),
                    }),
                ),
            },
        ),
        (
            "approval_requested",
            Notification::ApprovalRequested {
                request: ApprovalRequest {
                    request_id: ApprovalId::from("apr_1"),
                    action: "write src/retry.ts".into(),
                    risk: Risk::Low,
                    effects: vec![Effect::WriteFs],
                    detail: Some("apply the one-line fix".into()),
                },
            },
        ),
        (
            "runtime_status",
            Notification::RuntimeStatus {
                status: "ready".into(),
                detail: None,
            },
        ),
    ]
}

/// The full event bundle as a [`serde_json::Value`]:
///
/// ```json
/// { "items": { "<name>": { ... } }, "notifications": { "<name>": { ... } } }
/// ```
///
/// Serialized from the typed fixtures, so it is a faithful projection of the
/// protocol shapes, not a hand-written mirror.
pub fn events_bundle() -> Value {
    let mut items = Map::new();
    for (name, value) in item_fixtures() {
        items.insert(
            name.to_string(),
            serde_json::to_value(&value).expect("an Item always serializes"),
        );
    }
    let mut notifications = Map::new();
    for (name, value) in notification_fixtures() {
        notifications.insert(
            name.to_string(),
            serde_json::to_value(&value).expect("a Notification always serializes"),
        );
    }

    let mut bundle = Map::new();
    bundle.insert("items".to_string(), Value::Object(items));
    bundle.insert("notifications".to_string(), Value::Object(notifications));
    Value::Object(bundle)
}

/// The event bundle as a stable, pretty-printed string: the golden fixtures
/// artifact.
pub fn events_json() -> String {
    let mut s = serde_json::to_string_pretty(&events_bundle())
        .expect("the bundle is plain JSON and always serializes");
    s.push('\n');
    s
}

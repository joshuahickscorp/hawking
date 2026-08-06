//! Localhost fabric agent wire protocol (JSON lines over TCP).
//!
//! ABI does **not** assume co-location: messages are explicit, node ids are
//! opaque, and only loopback is used in this session (no network discovery).

use serde::{Deserialize, Serialize};

use super::failure::FailureReplayReceipt;
use super::node::{NodeCapabilities, NodeId};
use super::placement::PlacementPlan;

/// Assignment of a placement plan slice to a node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlacementAssignment {
    pub plan_id: String,
    pub plan: PlacementPlan,
    pub assigned_node: NodeId,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum AgentRequest {
    /// Agent → coordinator: register with capabilities.
    Register { capabilities: NodeCapabilities },
    /// Agent → coordinator: heartbeat with monotonic seq.
    Heartbeat { node_id: NodeId, seq: u64 },
    /// Coordinator → agent: accept a placement assignment.
    Assign { assignment: PlacementAssignment },
    /// Coordinator → agent: run a logical request id through local stages.
    RunRequest { request_id: String, plan_id: String },
    /// Coordinator → agent: inject a synthetic failure (fixture only).
    InjectFailure { node_id: NodeId },
    /// Coordinator → agent: query status.
    GetStatus { node_id: NodeId },
    /// Coordinator → agent: shutdown.
    Shutdown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum AgentResponse {
    Ok {
        node_id: NodeId,
        detail: String,
    },
    Registered {
        capabilities: NodeCapabilities,
    },
    HeartbeatAck {
        node_id: NodeId,
        seq: u64,
    },
    AssignmentAccepted {
        node_id: NodeId,
        plan_id: String,
        held_section_hashes: Vec<String>,
    },
    RequestProgress {
        node_id: NodeId,
        request_id: String,
        completed_local_stages: u32,
    },
    Failed {
        node_id: NodeId,
        reason: String,
    },
    Status {
        node_id: NodeId,
        alive: bool,
        assignment_plan_id: Option<String>,
        held_section_hashes: Vec<String>,
    },
    /// Placeholder event emission (Bridge lane owns the real event model).
    PlaceholderEvent {
        kind: String,
        payload: serde_json::Value,
    },
    Receipt {
        receipt: FailureReplayReceipt,
    },
    Error {
        message: String,
    },
}

/// Fabric placeholder event kind — Bridge lane owns the real event model.
pub const FABRIC_PLACEHOLDER_EVENT_KIND: &str = "fabric.placeholder.event.v1";

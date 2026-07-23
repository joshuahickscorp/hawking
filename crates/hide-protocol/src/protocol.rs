//! The wire protocol (Bible sec 15).
//!
//! Three surfaces:
//!
//! - [`Method`]: the request namespace. Every method is a slash-namespaced
//!   string (`workspace/create`, `thread/fork`, `state/save`, ...). The enum is
//!   the closed set the server answers.
//! - [`Notification`]: server-to-client pushes (sec 15.5) that stream a turn's
//!   items and lifecycle without a request.
//! - [`InitializeRequest`] / [`InitializeResult`]: the opening handshake
//!   (sec 15.3), including protocol-version and capability negotiation.
//!
//! Wire shape note (spec-derived): the request/response envelope and the
//! initialize handshake follow the JSON-RPC 2.0 and Agent Client Protocol
//! (ACP) conventions -- a correlation `id`, a slash-namespaced `method`, a
//! `params` object, and an `initialize` exchange that negotiates a single
//! protocol version and a capability set. Only the public wire conventions are
//! mirrored; no proprietary source is copied. The concrete shapes below are
//! HIDE-native.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

use crate::ids::{
    AgentId, CheckpointId, ItemId, RequestId, SessionId, StateCapsuleId, ThreadId, ToolCallId,
    TurnId,
};
use crate::item::{ApprovalRequest, Item};
use crate::model::{CompletionStatus, SessionStatus, TurnStatus};
use crate::plan::Plan;

/// The protocol version this crate defines. The handshake negotiates a single
/// version string; this is the one HIDE speaks.
pub const PROTOCOL_VERSION: &str = "hide.agent.v1";

macro_rules! methods {
    ($( $(#[$vmeta:meta])* $variant:ident => $name:literal ),* $(,)?) => {
        /// The closed set of request methods (Bible sec 15), grouped by
        /// namespace. Serializes as its slash-namespaced string.
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, Hash,
            Serialize, Deserialize, JsonSchema,
        )]
        pub enum Method {
            $( $(#[$vmeta])* #[serde(rename = $name)] $variant ),*
        }

        impl Method {
            /// Every method, in declaration order. Used by capability
            /// advertisement and coverage tests.
            pub const ALL: &'static [Method] = &[ $( Method::$variant ),* ];

            /// The slash-namespaced wire string for this method.
            pub fn as_str(&self) -> &'static str {
                match self { $( Method::$variant => $name ),* }
            }
        }
    };
}

methods! {
    // workspace/*
    WorkspaceCreate => "workspace/create",
    WorkspaceOpen => "workspace/open",
    WorkspaceClose => "workspace/close",
    WorkspaceGet => "workspace/get",
    WorkspaceList => "workspace/list",
    // environment/*
    EnvironmentCreate => "environment/create",
    EnvironmentGet => "environment/get",
    EnvironmentList => "environment/list",
    EnvironmentDispose => "environment/dispose",
    // session/*
    SessionNew => "session/new",
    SessionGet => "session/get",
    SessionList => "session/list",
    SessionClose => "session/close",
    // thread/*
    ThreadNew => "thread/new",
    ThreadGet => "thread/get",
    ThreadList => "thread/list",
    ThreadFork => "thread/fork",
    ThreadForkEphemeral => "thread/fork_ephemeral",
    ThreadMergeSummary => "thread/merge_summary",
    // goal/*
    GoalSet => "goal/set",
    GoalGet => "goal/get",
    GoalList => "goal/list",
    // turn/*
    TurnCreate => "turn/create",
    TurnGet => "turn/get",
    TurnSteer => "turn/steer",
    TurnInterrupt => "turn/interrupt",
    TurnPause => "turn/pause",
    TurnResume => "turn/resume",
    // item/*
    ItemGet => "item/get",
    ItemList => "item/list",
    ItemSubscribe => "item/subscribe",
    // agent/*
    AgentSpawn => "agent/spawn",
    AgentGet => "agent/get",
    AgentList => "agent/list",
    AgentResult => "agent/result",
    // checkpoint/*
    CheckpointCreate => "checkpoint/create",
    CheckpointList => "checkpoint/list",
    CheckpointRestore => "checkpoint/restore",
    // state/*
    StateSave => "state/save",
    StateLoad => "state/load",
    StateFork => "state/fork",
    StateRelease => "state/release",
    StateInspect => "state/inspect",
    // approval/*
    ApprovalRequestMethod => "approval/request",
    ApprovalRespond => "approval/respond",
    // artifact/*
    ArtifactGet => "artifact/get",
    ArtifactList => "artifact/list",
    ArtifactPut => "artifact/put",
}

impl Method {
    /// The namespace segment before the slash (`"thread"` for
    /// `thread/fork`). Every method has one.
    pub fn namespace(&self) -> &'static str {
        self.as_str()
            .split_once('/')
            .map(|(ns, _)| ns)
            .unwrap_or(self.as_str())
    }
}

/// A protocol request: correlation id, method, and opaque params. Per-method
/// param types are the model objects in [`crate::model`], [`crate::plan`], and
/// [`crate::item`]; the envelope carries them as `params` (JSON-RPC-derived).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Request {
    pub id: RequestId,
    pub method: Method,
    #[serde(default)]
    pub params: Value,
}

/// A protocol response: the matching correlation id, and exactly one of a
/// result or an error.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Response {
    pub id: RequestId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

/// A protocol error (JSON-RPC-derived shape).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// A streaming delta applied to an already-added item (for token-by-token
/// agent messages and live shell output).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ItemDelta {
    #[serde(default)]
    pub append_text: Option<String>,
    #[serde(default)]
    pub shell_chunk: Option<String>,
}

/// Server-to-client notifications (Bible sec 15.5). Adjacently tagged with a
/// slash-namespaced `method`, matching the request namespaces so a client
/// routes both by the same key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "method", content = "params")]
pub enum Notification {
    #[serde(rename = "session/updated")]
    SessionUpdated {
        session: SessionId,
        status: SessionStatus,
    },
    #[serde(rename = "thread/updated")]
    ThreadUpdated { thread: ThreadId },
    #[serde(rename = "turn/started")]
    TurnStarted { turn: TurnId },
    #[serde(rename = "turn/updated")]
    TurnUpdated { turn: TurnId, status: TurnStatus },
    #[serde(rename = "turn/completed")]
    TurnCompleted {
        turn: TurnId,
        status: CompletionStatus,
    },
    #[serde(rename = "item/added")]
    ItemAdded { item: Item },
    #[serde(rename = "item/updated")]
    ItemUpdated { item: Item },
    #[serde(rename = "item/delta")]
    ItemDeltaNotification { item: ItemId, delta: ItemDelta },
    #[serde(rename = "plan/updated")]
    PlanUpdated { plan: Plan },
    #[serde(rename = "tool/progress")]
    ToolProgress {
        call_id: ToolCallId,
        message: String,
    },
    #[serde(rename = "approval/requested")]
    ApprovalRequested { request: ApprovalRequest },
    #[serde(rename = "checkpoint/created")]
    CheckpointCreated { checkpoint: CheckpointId },
    #[serde(rename = "agent/spawned")]
    AgentSpawned { agent: AgentId },
    #[serde(rename = "agent/updated")]
    AgentUpdated { agent: AgentId, status: TurnStatus },
    #[serde(rename = "state/saved")]
    StateSaved { capsule: StateCapsuleId },
    #[serde(rename = "runtime/status")]
    RuntimeStatus {
        status: String,
        #[serde(default)]
        detail: Option<String>,
    },
    #[serde(rename = "error")]
    Error { code: String, message: String },
    #[serde(rename = "custom")]
    Custom { name: String, payload: Value },
}

impl Notification {
    /// The wire `method` tag for this notification.
    pub fn method(&self) -> &'static str {
        match self {
            Notification::SessionUpdated { .. } => "session/updated",
            Notification::ThreadUpdated { .. } => "thread/updated",
            Notification::TurnStarted { .. } => "turn/started",
            Notification::TurnUpdated { .. } => "turn/updated",
            Notification::TurnCompleted { .. } => "turn/completed",
            Notification::ItemAdded { .. } => "item/added",
            Notification::ItemUpdated { .. } => "item/updated",
            Notification::ItemDeltaNotification { .. } => "item/delta",
            Notification::PlanUpdated { .. } => "plan/updated",
            Notification::ToolProgress { .. } => "tool/progress",
            Notification::ApprovalRequested { .. } => "approval/requested",
            Notification::CheckpointCreated { .. } => "checkpoint/created",
            Notification::AgentSpawned { .. } => "agent/spawned",
            Notification::AgentUpdated { .. } => "agent/updated",
            Notification::StateSaved { .. } => "state/saved",
            Notification::RuntimeStatus { .. } => "runtime/status",
            Notification::Error { .. } => "error",
            Notification::Custom { .. } => "custom",
        }
    }
}

// -- initialize handshake (sec 15.3) ---------------------------------------

/// Identifies the peer at either end of the handshake.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PeerInfo {
    pub name: String,
    pub version: String,
}

/// What the client can do. The server ANDs these with its own to reach the
/// effective capability set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ClientCapabilities {
    #[serde(default)]
    pub streaming: bool,
    #[serde(default)]
    pub approvals: bool,
    #[serde(default)]
    pub fs: bool,
    #[serde(default)]
    pub terminal: bool,
    #[serde(default)]
    pub subscriptions: bool,
    #[serde(default)]
    pub experimental: BTreeMap<String, Value>,
}

/// What the server offers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ServerCapabilities {
    #[serde(default)]
    pub streaming: bool,
    #[serde(default)]
    pub subscriptions: bool,
    #[serde(default)]
    pub state: bool,
    #[serde(default)]
    pub agents: bool,
    #[serde(default)]
    pub checkpoints: bool,
    #[serde(default)]
    pub remote: bool,
    #[serde(default)]
    pub methods: Vec<Method>,
    #[serde(default)]
    pub experimental: BTreeMap<String, Value>,
}

/// The client's opening message (sec 15.3): who it is, which protocol versions
/// it speaks (highest preference first), and what it can do.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct InitializeRequest {
    pub client: PeerInfo,
    pub protocol_versions: Vec<String>,
    pub capabilities: ClientCapabilities,
}

/// The server's reply: who it is, the ONE negotiated protocol version, and its
/// capabilities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    pub server: PeerInfo,
    pub protocol_version: String,
    pub capabilities: ServerCapabilities,
}

/// The effective capability set after negotiation: shared booleans ANDed,
/// server-only features passed through.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NegotiatedCapabilities {
    pub streaming: bool,
    pub subscriptions: bool,
    pub approvals: bool,
    pub state: bool,
    pub agents: bool,
    pub checkpoints: bool,
    pub remote: bool,
}

/// Pick the highest-preference protocol version both sides support. The server
/// list is in the server's preference order; the first server version the
/// client also offers wins. Returns `None` if there is no overlap.
pub fn negotiate_version(client_versions: &[String], server_versions: &[String]) -> Option<String> {
    server_versions
        .iter()
        .find(|v| client_versions.iter().any(|c| c == *v))
        .cloned()
}

/// Compute the effective capabilities: shared booleans are ANDed (both sides
/// must want them), server-only capabilities are advertised as-is.
pub fn negotiate_capabilities(
    client: &ClientCapabilities,
    server: &ServerCapabilities,
) -> NegotiatedCapabilities {
    NegotiatedCapabilities {
        streaming: client.streaming && server.streaming,
        subscriptions: client.subscriptions && server.subscriptions,
        approvals: client.approvals,
        state: server.state,
        agents: server.agents,
        checkpoints: server.checkpoints,
        remote: server.remote,
    }
}

impl ServerCapabilities {
    /// A server that advertises every method and the full feature set. Handy
    /// for tests and for a default local host.
    pub fn full() -> Self {
        Self {
            streaming: true,
            subscriptions: true,
            state: true,
            agents: true,
            checkpoints: true,
            remote: false,
            methods: Method::ALL.to_vec(),
            experimental: BTreeMap::new(),
        }
    }
}

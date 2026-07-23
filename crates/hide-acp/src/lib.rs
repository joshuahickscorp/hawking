//! hide-acp: the Agent Client Protocol (ACP) server boundary for HIDE.
//!
//! ACP is an open, Apache-2.0-licensed protocol that lets a code editor
//! (Zed, JetBrains, ...) talk to an external coding agent over JSON-RPC. This
//! crate lets HIDE appear as such an agent (Bible sec 10, sec 69) WITHOUT the
//! rest of HIDE having to know ACP: it defines the ACP message and session
//! types, negotiates capabilities, and projects between the ONE HIDE schema
//! authority (`hide-protocol`) and the ACP wire.
//!
//! # Spec-derived, not copied
//!
//! Every ACP wire shape here is derived from the PUBLIC ACP specification: the
//! `initialize` handshake, the `session/new` / `session/prompt` methods, the
//! `session/update` notification and its `sessionUpdate` union, the tool-call
//! and `session/request_permission` surfaces, and the content-block, diff, and
//! terminal shapes. Each carrying module notes its spec-derived shapes in a doc
//! comment. Only the public wire conventions are mirrored; no proprietary Zed
//! source is copied, and the internal semantics are HIDE-native.
//!
//! # Two directions
//!
//! - [`ProjectHideToAcp`] projects a stream of `hide_protocol::Item`s (and
//!   server `Notification`s) onto ordered ACP outbounds: `session/update`
//!   notifications and `session/request_permission` requests.
//! - [`AcpToHide`] maps an inbound ACP `session/prompt` back into a
//!   [`HideTurnIntent`] the runtime turns into a `turn/create`.
//!
//! [`SessionThreadMap`] binds the flat ACP `sessionId` to HIDE's
//! Session -> Thread spine.
//!
//! # Capability negotiation and honest degradation
//!
//! [`negotiate`] ANDs [`HideExposure`] with the client's declared
//! [`AcpClientCapabilities`] to produce the [`EffectiveCapabilities`] the
//! projector runs under, and RECORDS every surface the client cannot receive as
//! a [`Degradation`] rather than pretending it is available (for example, shell
//! output falls back from a terminal projection to plain text content).
//!
//! # Model-free
//!
//! This crate is a schema-and-mapping boundary and is entirely model-free
//! (RIP doctrine). It never runs a model, opens a socket, or drives an editor;
//! it is proven over deterministic fixtures. The legs that inherently need a
//! live runtime -- streaming real terminal bytes over the ACP terminal channel,
//! and hashing/sizing fetched attachment bytes -- are marked
//! `DEFERRED_MODEL_REQUIRED` at their definitions and are not implemented or
//! claimed here.
//!
//! ```
//! use hide_acp::{negotiate, HideExposure, AcpInitializeRequest};
//!
//! let req = AcpInitializeRequest { protocol_version: 1, client_capabilities: Default::default() };
//! let out = negotiate(&req, &HideExposure::full_local()).unwrap();
//! assert_eq!(out.response.protocol_version, 1);
//! // No terminal advertised by the client -> honest downgrade recorded.
//! assert!(out.degradations.iter().any(|d| d.capability == "terminal"));
//! ```

pub mod capability;
pub mod content;
pub mod error;
pub mod handshake;
pub mod ids;
pub mod ingest;
pub mod map;
pub mod permission;
pub mod project;
pub mod server;
pub mod session;
pub mod tool_call;
pub mod transport;
pub mod unified_diff;

pub use capability::{negotiate, Degradation, EffectiveCapabilities, HideExposure, Negotiation};
pub use content::{ContentBlock, ResourceContents};
pub use error::{AcpError, Result};
pub use handshake::{
    negotiate_protocol_version, AcpAgentCapabilities, AcpClientCapabilities, AcpInitializeRequest,
    AcpInitializeResponse, AcpPromptCapabilities, AuthMethod, FsCapabilities, ACP_PROTOCOL_VERSION,
};
pub use ids::{AcpSessionId, AcpTerminalId, AcpToolCallId};
pub use ingest::{AcpToHide, HideTurnIntent};
pub use map::{HideBinding, SessionThreadMap};
pub use permission::{
    standard_options, PermissionOption, PermissionOptionKind, PermissionOutcome,
    RequestPermissionRequest, RequestPermissionResponse,
};
pub use project::{AcpOutbound, ProjectHideToAcp};
pub use server::{
    AcpServer, CountingBinder, DeferredTurnHandler, MintedSession, ScriptedTurnHandler,
    SessionBinder, TurnEvent, TurnHandler,
};
pub use session::{
    AcpLoadSessionRequest, AcpNewSessionRequest, AcpNewSessionResponse, AcpPlan, AcpPromptRequest,
    AcpPromptResponse, McpServer, PlanEntry, PlanEntryPriority, PlanEntryStatus, SessionNotification,
    SessionUpdate, StopReason,
};
pub use tool_call::{
    ToolCall, ToolCallContent, ToolCallLocation, ToolCallStatus, ToolCallUpdate, ToolKind,
};
pub use transport::{
    memory_duplex, AcpClientMessage, AcpServerMessage, CancelParams, LineTransport, MemoryClient,
    MemoryTransport, ServerError, Transport,
};

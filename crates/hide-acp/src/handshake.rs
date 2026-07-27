//! The ACP `initialize` handshake (spec-derived).
//!
//! Spec-derived (ACP): the client opens with `initialize`, sending an integer
//! `protocolVersion` and its `clientCapabilities`. The agent replies with the
//! ONE negotiated `protocolVersion`, its `agentCapabilities`, and any
//! `authMethods`. Version negotiation is: if the agent supports the client's
//! requested version it echoes it, otherwise it returns the latest version it
//! supports and the client decides whether to proceed. Only the public wire
//! shape is mirrored; no proprietary source is copied.

use serde::{Deserialize, Serialize};

/// The highest ACP major protocol version this boundary speaks. ACP versions
/// are integers; the agent supports the inclusive range `1..=ACP_PROTOCOL_VERSION`.
pub const ACP_PROTOCOL_VERSION: u16 = 1;

/// The client's filesystem capabilities (whether it can service the agent's
/// `fs/read_text_file` / `fs/write_text_file` calls).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FsCapabilities {
    #[serde(default)]
    pub read_text_file: bool,
    #[serde(default)]
    pub write_text_file: bool,
}

/// What the ACP client can do. The agent ANDs the relevant flags with its own
/// exposure to reach the effective set (see [`crate::capability`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcpClientCapabilities {
    #[serde(default)]
    pub fs: FsCapabilities,
    #[serde(default)]
    pub terminal: bool,
}

/// What kinds of content the agent accepts in a prompt.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcpPromptCapabilities {
    #[serde(default)]
    pub image: bool,
    #[serde(default)]
    pub audio: bool,
    #[serde(default)]
    pub embedded_context: bool,
}

/// What the agent (HIDE) offers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcpAgentCapabilities {
    #[serde(default)]
    pub load_session: bool,
    #[serde(default)]
    pub prompt_capabilities: AcpPromptCapabilities,
}

/// An authentication method the agent advertises. The local HIDE boundary needs
/// none, so this list is normally empty.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuthMethod {
    pub id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// The client's opening `initialize` params.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcpInitializeRequest {
    pub protocol_version: u16,
    #[serde(default)]
    pub client_capabilities: AcpClientCapabilities,
}

/// The agent's `initialize` result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcpInitializeResponse {
    pub protocol_version: u16,
    pub agent_capabilities: AcpAgentCapabilities,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub auth_methods: Vec<AuthMethod>,
}

/// Negotiate the single protocol version. Spec-derived (ACP): echo the client's
/// requested version when the agent supports it, otherwise return the agent's
/// latest. Returns `None` only when the requested version is below the agent's
/// minimum (0), which the caller treats as an unsupported-version error.
pub fn negotiate_protocol_version(client: u16, agent_max: u16) -> Option<u16> {
    if client == 0 {
        return None;
    }
    if client <= agent_max {
        Some(client)
    } else {
        Some(agent_max)
    }
}

//! Initialization and capability negotiation (Stage 4, Group B; Codex
//! adapted-port of `app-server-protocol/src/protocol/v1.rs` Initialize).
//!
//! `hide-protocol` already ships the version-negotiation handshake
//! (`InitializeRequest`/`InitializeResult`). What the depth map singles out as
//! the two cheap, high-value negotiation levers is NOT in that handshake:
//!
//! * `experimental_api` - a single per-connection gate that unlocks an
//!   experimental method surface.
//! * `opt_out_notification_methods` - per-connection notification suppression by
//!   exact wire method name (for example `runtime/status`), so a connection that
//!   does not want a class of pushes simply never receives them.
//!
//! This module holds those two levers plus the per-connection store the server
//! consults in its notification emit path. Trimmed to HIDE's needs, model-free.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Who the client is (Codex `ClientInfo`). `title` is an optional display name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientInfo {
    pub name: String,
    #[serde(default)]
    pub title: Option<String>,
    pub version: String,
}

/// The two per-connection capability levers a client negotiates at Initialize.
/// Both fields default (serde) so an older client that sends neither gets the
/// safe baseline: experimental methods off, nothing suppressed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientCapabilities {
    /// Opt into the experimental method/field surface. Off by default.
    #[serde(default)]
    pub experimental_api: bool,
    /// Exact wire method names this connection does NOT want pushed to it (for
    /// example `["runtime/status", "tool/progress"]`).
    #[serde(default)]
    pub opt_out_notification_methods: Vec<String>,
}

/// The server's reply to Initialize (Codex `InitializeResponse`, HIDE fields):
/// who the server is and the platform it runs on. Purely informational.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InitializeResponse {
    pub user_agent: String,
    pub workspace_root: String,
    pub platform_family: String,
    pub platform_os: String,
}

/// The per-connection capability store the server consults when emitting
/// notifications. Keyed by an opaque connection id (the transport supplies it).
/// A connection that never sent Initialize is treated as the safe baseline
/// (nothing suppressed, experimental off).
#[derive(Default)]
pub struct ConnectionRegistry {
    connections: Mutex<HashMap<String, ClientCapabilities>>,
}

impl ConnectionRegistry {
    /// Record (or replace) the negotiated capabilities for a connection. Called
    /// by the Initialize handler.
    pub fn initialize(&self, connection_id: impl Into<String>, capabilities: ClientCapabilities) {
        self.connections
            .lock()
            .insert(connection_id.into(), capabilities);
    }

    /// The negotiated capabilities for a connection, if it initialized.
    pub fn capabilities(&self, connection_id: &str) -> Option<ClientCapabilities> {
        self.connections.lock().get(connection_id).cloned()
    }

    /// Whether the experimental method surface is unlocked for a connection.
    /// A connection that never initialized has it OFF (the safe default).
    pub fn experimental_api(&self, connection_id: &str) -> bool {
        self.connections
            .lock()
            .get(connection_id)
            .map(|c| c.experimental_api)
            .unwrap_or(false)
    }

    /// Whether a notification with wire method `method` is SUPPRESSED for a
    /// connection (it opted out). A connection that never initialized suppresses
    /// nothing.
    pub fn is_notification_suppressed(&self, connection_id: &str, method: &str) -> bool {
        self.connections
            .lock()
            .get(connection_id)
            .map(|c| c.opt_out_notification_methods.iter().any(|m| m == method))
            .unwrap_or(false)
    }

    /// Forget a connection's capabilities (the connection closed).
    pub fn forget(&self, connection_id: &str) {
        self.connections.lock().remove(connection_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uninitialized_connection_is_the_safe_baseline() {
        let reg = ConnectionRegistry::default();
        assert!(!reg.experimental_api("conn-unknown"));
        assert!(!reg.is_notification_suppressed("conn-unknown", "runtime/status"));
        assert!(reg.capabilities("conn-unknown").is_none());
    }

    #[test]
    fn opt_out_suppresses_only_the_named_methods() {
        let reg = ConnectionRegistry::default();
        reg.initialize(
            "conn-1",
            ClientCapabilities {
                experimental_api: true,
                opt_out_notification_methods: vec!["runtime/status".to_string()],
            },
        );
        assert!(reg.experimental_api("conn-1"));
        assert!(reg.is_notification_suppressed("conn-1", "runtime/status"));
        assert!(
            !reg.is_notification_suppressed("conn-1", "tool/progress"),
            "a method NOT opted out is still delivered"
        );
    }

    #[test]
    fn forget_drops_the_connection() {
        let reg = ConnectionRegistry::default();
        reg.initialize("conn-2", ClientCapabilities::default());
        assert!(reg.capabilities("conn-2").is_some());
        reg.forget("conn-2");
        assert!(reg.capabilities("conn-2").is_none());
    }

    #[test]
    fn capabilities_default_off() {
        let caps = ClientCapabilities::default();
        assert!(!caps.experimental_api);
        assert!(caps.opt_out_notification_methods.is_empty());
    }
}

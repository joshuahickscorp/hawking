use hide_core::api::Intent;
use hide_core::event::Event;
use hide_core::ids::SessionId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", content = "params", rename_all = "snake_case")]
pub enum RemoteRequest {
    SessionNew {
        workspace: String,
    },
    SessionLoad {
        session_id: SessionId,
        after_seq: Option<u64>,
    },
    Intent {
        intent: Intent,
    },
    Subscribe {
        session_id: SessionId,
        after_seq: Option<u64>,
    },
    Ping,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum RemoteUpdate {
    SessionReady {
        session_id: SessionId,
        head_seq: u64,
    },
    Events {
        events: Vec<Event>,
    },
    Ack {
        request_id: String,
    },
    Error {
        code: String,
        message: String,
    },
    Pong,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteAuthPolicy {
    pub loopback_only: bool,
    pub allow_ssh_tunnel: bool,
    pub token_required: bool,
}

impl Default for RemoteAuthPolicy {
    fn default() -> Self {
        Self {
            loopback_only: true,
            allow_ssh_tunnel: true,
            token_required: true,
        }
    }
}

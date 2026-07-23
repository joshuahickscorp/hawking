//! Transparent string id newtypes for the ACP surface.
//!
//! Spec-derived (ACP): ACP addresses sessions, tool calls, and terminals by
//! opaque string ids (`sessionId`, `toolCallId`, `terminalId`). These serialize
//! as bare strings so the wire carries `"sess_..."` rather than a wrapper
//! object. This crate never mints ids; a caller (the HIDE runtime or a test
//! fixture) owns the value.

use serde::{Deserialize, Serialize};

macro_rules! acp_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }
        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }
        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_string())
            }
        }
        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

acp_id!(
    /// An ACP session id. One ACP session maps onto a HIDE (session, thread)
    /// pair; see [`crate::map::SessionThreadMap`].
    AcpSessionId
);
acp_id!(
    /// An ACP tool-call id. Projected 1:1 from a HIDE `ToolCallId`.
    AcpToolCallId
);
acp_id!(
    /// An ACP terminal id. The terminal projection references one of these; the
    /// live byte stream over the ACP terminal channel is DEFERRED_MODEL_REQUIRED.
    AcpTerminalId
);

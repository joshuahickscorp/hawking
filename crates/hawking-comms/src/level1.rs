//! LEVEL 1 — TEXT: portable and inspectable.

use crate::error::{CommsError, Result};
use serde::{Deserialize, Serialize};

/// Portable text message between sessions / agents / humans.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextMessage {
    pub schema: String,
    /// Session id (HCLI / hide-protocol session style).
    pub session_id: String,
    /// Sender identity (agent id, user id, or service name).
    pub sender: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipient: Option<String>,
    pub role: TextRole,
    pub text: String,
    /// Unix epoch ms when produced.
    pub created_unix_ms: u64,
    #[serde(default)]
    pub content_hash: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextRole {
    User,
    Agent,
    System,
    Tool,
    Peer,
}

impl TextMessage {
    pub fn user(session_id: impl Into<String>, text: impl Into<String>) -> Self {
        let text = text.into();
        let content_hash = format!("blake3:{}", blake3::hash(text.as_bytes()).to_hex());
        Self {
            schema: crate::TEXT_MESSAGE_SCHEMA.to_string(),
            session_id: session_id.into(),
            sender: "user".into(),
            recipient: None,
            role: TextRole::User,
            text,
            created_unix_ms: 0,
            content_hash,
        }
    }

    pub fn agent(
        session_id: impl Into<String>,
        sender: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        let text = text.into();
        let content_hash = format!("blake3:{}", blake3::hash(text.as_bytes()).to_hex());
        Self {
            schema: crate::TEXT_MESSAGE_SCHEMA.to_string(),
            session_id: session_id.into(),
            sender: sender.into(),
            recipient: None,
            role: TextRole::Agent,
            text,
            created_unix_ms: 0,
            content_hash,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != crate::TEXT_MESSAGE_SCHEMA {
            return Err(CommsError::Invalid(format!(
                "text schema {}",
                self.schema
            )));
        }
        if self.session_id.trim().is_empty() {
            return Err(CommsError::Invalid("text message missing session_id".into()));
        }
        if self.sender.trim().is_empty() {
            return Err(CommsError::Invalid("text message missing sender".into()));
        }
        if self.text.is_empty() {
            return Err(CommsError::Invalid("text message body empty".into()));
        }
        let expected = format!("blake3:{}", blake3::hash(self.text.as_bytes()).to_hex());
        if !self.content_hash.is_empty() && self.content_hash != expected {
            return Err(CommsError::HashMismatch {
                expected,
                got: self.content_hash.clone(),
            });
        }
        Ok(())
    }
}

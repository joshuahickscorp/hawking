//! Outer bus envelope shared by all levels.

use crate::error::{CommsError, Result};
use crate::level1::TextMessage;
use crate::level2::StructuredState;
use crate::level3::LatentPacket;
use serde::{Deserialize, Serialize};

/// Communication level (bible §20).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommLevel {
    /// LEVEL 1 — portable and inspectable text.
    Text = 1,
    /// LEVEL 2 — plans, evidence graphs, beliefs, tool results.
    StructuredState = 2,
    /// LEVEL 3 — latent / KV / embedding — experimental only.
    Latent = 3,
}

impl CommLevel {
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::StructuredState => "structured_state",
            Self::Latent => "latent",
        }
    }

    pub fn is_experimental(self) -> bool {
        matches!(self, Self::Latent)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PacketId(pub String);

impl PacketId {
    pub fn new(level: CommLevel, body_hash: &str) -> Self {
        let raw = format!("{}:{}:{}", crate::COMMS_SCHEMA, level.as_str(), body_hash);
        Self(format!("pkt:{}", blake3::hash(raw.as_bytes()).to_hex()))
    }
}

/// Top-level envelope a bus hop carries. LEVEL 3 still rides inside this so
/// routers can refuse experimental traffic without parsing the latent body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "level", rename_all = "snake_case")]
pub enum BusEnvelope {
    Text {
        id: PacketId,
        schema: String,
        message: TextMessage,
    },
    StructuredState {
        id: PacketId,
        schema: String,
        state: StructuredState,
    },
    /// Experimental. Must pass [`LatentPacket::validate_sealed`] before any
    /// attempt to bind payload bytes into a runtime (binding itself is deferred).
    Latent {
        id: PacketId,
        schema: String,
        packet: LatentPacket,
    },
}

impl BusEnvelope {
    pub fn level(&self) -> CommLevel {
        match self {
            Self::Text { .. } => CommLevel::Text,
            Self::StructuredState { .. } => CommLevel::StructuredState,
            Self::Latent { .. } => CommLevel::Latent,
        }
    }

    pub fn id(&self) -> &PacketId {
        match self {
            Self::Text { id, .. } | Self::StructuredState { id, .. } | Self::Latent { id, .. } => {
                id
            }
        }
    }

    pub fn from_text(message: TextMessage) -> Self {
        let body = serde_json::to_vec(&message).unwrap_or_default();
        let hash = format!("blake3:{}", blake3::hash(&body).to_hex());
        Self::Text {
            id: PacketId::new(CommLevel::Text, &hash),
            schema: crate::TEXT_MESSAGE_SCHEMA.to_string(),
            message,
        }
    }

    pub fn from_structured(state: StructuredState) -> Self {
        let body = serde_json::to_vec(&state).unwrap_or_default();
        let hash = format!("blake3:{}", blake3::hash(&body).to_hex());
        Self::StructuredState {
            id: PacketId::new(CommLevel::StructuredState, &hash),
            schema: crate::STRUCTURED_STATE_SCHEMA.to_string(),
            state,
        }
    }

    pub fn from_latent(packet: LatentPacket) -> Result<Self> {
        packet.validate_sealed()?;
        let body = serde_json::to_vec(&packet)
            .map_err(|e| CommsError::Invalid(format!("serialize latent packet: {e}")))?;
        let hash = format!("blake3:{}", blake3::hash(&body).to_hex());
        Ok(Self::Latent {
            id: PacketId::new(CommLevel::Latent, &hash),
            schema: crate::LATENT_PACKET_SCHEMA.to_string(),
            packet,
        })
    }

    /// Validate envelope invariants. Latent always requires a complete seal.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Text {
                message, schema, ..
            } => {
                if schema != crate::TEXT_MESSAGE_SCHEMA {
                    return Err(CommsError::Invalid(format!(
                        "unexpected text schema {schema}"
                    )));
                }
                message.validate()
            }
            Self::StructuredState { state, schema, .. } => {
                if schema != crate::STRUCTURED_STATE_SCHEMA {
                    return Err(CommsError::Invalid(format!(
                        "unexpected structured schema {schema}"
                    )));
                }
                state.validate()
            }
            Self::Latent { packet, schema, .. } => {
                if schema != crate::LATENT_PACKET_SCHEMA {
                    return Err(CommsError::Invalid(format!(
                        "unexpected latent schema {schema}"
                    )));
                }
                packet.validate_sealed()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::level1::TextMessage;

    #[test]
    fn text_envelope_round_trips() {
        let msg = TextMessage::user("ses_demo", "hello bus");
        let env = BusEnvelope::from_text(msg);
        assert_eq!(env.level(), CommLevel::Text);
        env.validate().unwrap();
        let json = serde_json::to_string(&env).unwrap();
        let back: BusEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(back.level(), CommLevel::Text);
    }
}

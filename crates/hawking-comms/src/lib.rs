//! HCLI Communication Bus (Ascension Bible §20).
//!
//! Three communication levels:
//!
//! | Level | Name | Role |
//! |-------|------|------|
//! | 1 | TEXT | Portable and inspectable natural language / transcripts |
//! | 2 | STRUCTURED STATE | Plans, evidence graphs, typed beliefs, tool results |
//! | 3 | LATENT | Hidden-state / embedding / KV transfer — **experimental only** |
//!
//! ## Non-goals of this crate
//!
//! - No live cross-session latent or KV transfer.
//! - No model execution.
//! - No unsealed latent packets (validation refuses them).
//!
//! LEVEL 3 is a **sealed-packet format** ready for same-model Qwen session
//! pairs later. Cross-model latent transfer remains forbidden until trained
//! alignment has independent evidence (bible §20).
//!
//! ## Existing patterns reused
//!
//! - Session identity style aligns with `hide-backend::hcli_bridge` session
//!   ids and `hide_protocol::model::Session` / `StateCapsuleRef` (digest pins).
//! - Evidence graph nodes intentionally mirror research claim/evidence shapes
//!   (`hawking-research` CAS / KG) without pulling those crates.
//! - Plan payloads are JSON-shaped so they can carry `hide_protocol::Plan`
//!   without a hard dependency on hide-protocol.

pub mod error;
pub mod level1;
pub mod level2;
pub mod level3;
pub mod packet;
pub mod seal;

pub use error::{CommsError, Result};
pub use level1::TextMessage;
pub use level2::{
    Belief, BeliefPolarity, EvidenceEdge, EvidenceGraph, EvidenceNode, StructuredKind,
    StructuredState, ToolResultPayload,
};
pub use level3::{
    LatentDType, LatentKind, LatentPacket, LatentPayloadRef, LayerRange, ModelIdentity,
};
pub use packet::{BusEnvelope, CommLevel, PacketId};
pub use seal::{
    CapabilityScope, SealHeader, SealStatus, VisibleCommitment, LATENT_EXPERIMENTAL_GATE,
};

/// Schema markers for capability surfaces and receipts.
pub const COMMS_SCHEMA: &str = "hcli.comms.v0";
pub const LATENT_PACKET_SCHEMA: &str = "hcli.comms.latent.v0";
pub const STRUCTURED_STATE_SCHEMA: &str = "hcli.comms.structured.v0";
pub const TEXT_MESSAGE_SCHEMA: &str = "hcli.comms.text.v0";

//! Capability negotiation: what HIDE exposes over ACP, and how it degrades
//! honestly when the client cannot receive it.
//!
//! HIDE has richer surfaces than a bare ACP client is guaranteed to render
//! (edit review, terminals, artifacts, reasoning). The boundary advertises what
//! it exposes, ANDs it with the client's declared capabilities, and RECORDS
//! every downgrade as a [`Degradation`] rather than pretending the surface is
//! available. Nothing here runs a model.

use serde::{Deserialize, Serialize};

use crate::handshake::{
    negotiate_protocol_version, AcpAgentCapabilities, AcpInitializeRequest, AcpInitializeResponse,
    AcpPromptCapabilities, ACP_PROTOCOL_VERSION,
};

/// What this HIDE boundary is willing to expose over ACP. This is HIDE's side
/// of the negotiation, independent of any client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HideExposure {
    /// Stream agent output token-by-token as `agent_message_chunk` updates.
    pub streaming: bool,
    /// Project HIDE reasoning summaries as `agent_thought_chunk` updates.
    pub thoughts: bool,
    /// Present patches/diffs as reviewable ACP `diff` tool-call content.
    pub edit_review: bool,
    /// Project shell output onto the ACP terminal surface.
    pub terminal: bool,
    /// Surface HIDE artifacts as `resource_link` content.
    pub artifacts: bool,
    /// Advertise `session/load` so a client can resume a prior thread.
    pub load_session: bool,
}

impl Default for HideExposure {
    fn default() -> Self {
        Self::full_local()
    }
}

impl HideExposure {
    /// The default local HIDE host exposes every surface.
    pub fn full_local() -> Self {
        Self {
            streaming: true,
            thoughts: true,
            edit_review: true,
            terminal: true,
            artifacts: true,
            load_session: true,
        }
    }

    /// The ACP `agentCapabilities` HIDE advertises. Terminal is a CLIENT
    /// capability in ACP (the agent drives the client's terminal), so it is not
    /// echoed here; it surfaces in the effective set instead.
    pub fn agent_capabilities(&self) -> AcpAgentCapabilities {
        AcpAgentCapabilities {
            load_session: self.load_session,
            prompt_capabilities: AcpPromptCapabilities {
                image: false,
                audio: false,
                // HIDE accepts embedded resource context blocks in a prompt.
                embedded_context: true,
            },
        }
    }
}

/// The capability set actually in force after ANDing HIDE's exposure with the
/// client's declared capabilities. The projector reads this to decide each
/// item's honest ACP shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EffectiveCapabilities {
    pub streaming: bool,
    pub thoughts: bool,
    /// Diffs can be shown for review.
    pub edit_review: bool,
    /// Edits can actually be applied (requires the client to be able to write
    /// files). When false, edits are review-only.
    pub edit_apply: bool,
    /// Shell output can use the ACP terminal surface.
    pub terminal: bool,
    pub artifacts: bool,
    pub load_session: bool,
}

/// One honest downgrade: a surface HIDE wanted to expose that the client cannot
/// receive, plus the reason and the fallback that was used instead.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Degradation {
    /// The capability that was downgraded (stable slug).
    pub capability: &'static str,
    /// Why it was downgraded.
    pub reason: String,
    /// The honest fallback the projector uses instead.
    pub fallback: &'static str,
}

/// The full outcome of an `initialize` exchange: the response to send, the
/// effective capabilities to project under, and the honest downgrade log.
#[derive(Debug, Clone, PartialEq)]
pub struct Negotiation {
    pub response: AcpInitializeResponse,
    pub effective: EffectiveCapabilities,
    pub degradations: Vec<Degradation>,
}

/// Run the whole `initialize` negotiation for a given HIDE exposure.
///
/// Version is negotiated per the ACP rule; capabilities are ANDed with the
/// client's; every surface the client cannot receive is recorded in
/// `degradations`. Returns `Err(UnsupportedVersion)` only when the client's
/// requested version is below the agent minimum.
pub fn negotiate(
    req: &AcpInitializeRequest,
    exposure: &HideExposure,
) -> crate::error::Result<Negotiation> {
    let version = negotiate_protocol_version(req.protocol_version, ACP_PROTOCOL_VERSION).ok_or(
        crate::error::AcpError::UnsupportedVersion {
            offered: req.protocol_version,
            agent_max: ACP_PROTOCOL_VERSION,
        },
    )?;

    let client = &req.client_capabilities;
    let mut degradations = Vec::new();

    let terminal = exposure.terminal && client.terminal;
    if exposure.terminal && !client.terminal {
        degradations.push(Degradation {
            capability: "terminal",
            reason: "client did not advertise terminal support".to_string(),
            fallback: "shell output projected as plain text tool-call content",
        });
    }

    let edit_apply = exposure.edit_review && client.fs.write_text_file;
    if exposure.edit_review && !client.fs.write_text_file {
        degradations.push(Degradation {
            capability: "edit_apply",
            reason: "client cannot write files (fs.writeTextFile is false)".to_string(),
            fallback: "edits presented as review-only diffs",
        });
    }

    let effective = EffectiveCapabilities {
        streaming: exposure.streaming,
        thoughts: exposure.thoughts,
        edit_review: exposure.edit_review,
        edit_apply,
        terminal,
        artifacts: exposure.artifacts,
        load_session: exposure.load_session,
    };

    let response = AcpInitializeResponse {
        protocol_version: version,
        agent_capabilities: exposure.agent_capabilities(),
        auth_methods: Vec::new(),
    };

    Ok(Negotiation {
        response,
        effective,
        degradations,
    })
}

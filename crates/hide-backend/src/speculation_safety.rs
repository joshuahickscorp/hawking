//! Host-side speculation safety boundary.
//!
//! Draft tokens must never reach durable host sinks. Every sink here accepts
//! only [`hawking_speculate::VerifiedTokenId`] (or a verified text payload).
//! The speculation engine promotes drafts through target verification; the host
//! only sees verified tokens on this surface.
//!
//! # Invariant
//!
//! Only target-verified tokens may:
//! - enter the canonical event stream
//! - enter durable context or any memory class
//! - trigger a tool call
//! - change a file
//! - appear as final user-visible output
//!
//! Compile-time enforcement: sink methods take `VerifiedTokenId` only. There is
//! no overload for `DraftTokenId`. See `hawking_speculate::token_boundary`.

use hawking_speculate::{
    action_for, evaluate, DurableTokenSink, SpeculationPermit, SuspensionAction,
    SuspensionBoundary, TargetVerification, VerifiedTokenId, SUSPENSION_POLICY,
};
use hide_core::error::{HideError, Result};
use hide_core::ids::{EventId, RunId, SessionId};
use hide_core::tool::ToolCall;
use std::sync::Arc;

use crate::classed_writers::VerifiedTokenEventLog;

/// Host durable sinks: each method requires a [`VerifiedTokenId`].
///
/// This is the type-level gate between speculation and the hide durable plane.
/// Production wiring (event log append, memory, tools, file edit, UI final text)
/// must go through these methods when the token origin is the speculative path.
pub struct HostDurableSinks {
    /// Session the sinks write into.
    pub session_id: SessionId,
    /// Optional live event log. When `None`, records stay in `recorded` only
    /// (fixture / unit-test mode — not production I/O).
    event_log: Option<Arc<VerifiedTokenEventLog>>,
    /// In-process ledger of durable actions (always populated; tests assert on it).
    recorded: Vec<HostDurableRecord>,
    /// Accumulated final output text from verified tokens (detokenized by caller).
    final_text: String,
}

/// One host durable action. Mirrors the five sinks of the safety invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostDurableRecord {
    CanonicalEvent { token_id: u32 },
    MemoryWrite { token_id: u32 },
    ToolDispatch { token_id: u32 },
    FileEdit { token_id: u32 },
    FinalOutput { token_id: u32 },
}

/// One-use causal authority for an effect proposed by a model token.
///
/// The constructor is private: production code can obtain this only after the
/// canonical target-verified event was appended to the product event log. It
/// is intentionally consumed by the dispatch-context helper, making a tool or
/// edit event addressable from the exact verified token that caused it.
pub struct VerifiedEffectPermit {
    session_id: SessionId,
    run_id: Option<RunId>,
    token_id: u32,
    canonical_event_id: EventId,
    tool_call_digest: blake3::Hash,
}

impl VerifiedEffectPermit {
    /// Consume this permit only for the exact session, run, and normalized
    /// tool call it was minted for.  A target-verified token is evidence for
    /// its own parsed action, not a reusable delegation capability.
    pub(crate) fn consume_for(
        self,
        session_id: &SessionId,
        run_id: &Option<RunId>,
        call: &ToolCall,
    ) -> Result<(u32, EventId)> {
        if &self.session_id != session_id {
            return Err(HideError::PolicyDenied(
                "verified model-effect permit belongs to a different session".into(),
            ));
        }
        if &self.run_id != run_id {
            return Err(HideError::PolicyDenied(
                "verified model-effect permit belongs to a different run".into(),
            ));
        }
        if self.tool_call_digest != tool_call_digest(call) {
            return Err(HideError::PolicyDenied(
                "verified model-effect permit does not match this exact tool call".into(),
            ));
        }
        Ok((self.token_id, self.canonical_event_id))
    }
}

/// Digest the complete canonical tool-call envelope before it is dispatched.
/// A serde failure is impossible for [`ToolCall`] today, but a refusal is
/// safer than turning a future non-serializable field into an unbound effect.
fn tool_call_digest(call: &ToolCall) -> blake3::Hash {
    let bytes = serde_json::to_vec(call)
        .expect("ToolCall is a durable serde contract and must serialize before dispatch");
    blake3::hash(&bytes)
}

impl HostDurableSinks {
    /// Fixture mode: no event log, pure in-memory ledger.
    pub fn fixture(session_id: SessionId) -> Self {
        Self {
            session_id,
            event_log: None,
            recorded: Vec::new(),
            final_text: String::new(),
        }
    }

    /// Live mode: appends a `token` system event for each canonical emission.
    pub fn with_event_log(
        session_id: SessionId,
        event_log: hide_core::persistence::DynEventLog,
    ) -> Self {
        Self {
            session_id,
            event_log: Some(VerifiedTokenEventLog::authority(event_log)),
            recorded: Vec::new(),
            final_text: String::new(),
        }
    }

    pub(crate) fn with_token_authority(
        session_id: SessionId,
        event_log: Arc<VerifiedTokenEventLog>,
    ) -> Self {
        Self {
            session_id,
            event_log: Some(event_log),
            recorded: Vec::new(),
            final_text: String::new(),
        }
    }

    pub fn recorded(&self) -> &[HostDurableRecord] {
        &self.recorded
    }

    pub fn final_text(&self) -> &str {
        &self.final_text
    }

    /// Target-direct gate for non-speculative decode (serve already emits target
    /// tokens). Never wrap a draft id.
    pub fn target_gate() -> TargetVerification {
        TargetVerification::gate()
    }

    /// Emit a verified token into the canonical event stream.
    /// Signature: **Verified only**.
    pub async fn emit_canonical_event(&mut self, token: VerifiedTokenId) -> Result<()> {
        let token_id = token.get();
        if let Some(log) = &self.event_log {
            log.append_target_verified(self.session_id.clone(), token, "")
                .await?;
        }
        self.recorded
            .push(HostDurableRecord::CanonicalEvent { token_id });
        Ok(())
    }

    /// Durably publish a verified token and mint the causal authority needed
    /// for one model-origin effect. Fixture sinks cannot mint an effect permit:
    /// they have no durable event id to bind as the tool/edit cause.
    pub async fn authorize_verified_tool_effect(
        &mut self,
        token: VerifiedTokenId,
        run_id: Option<RunId>,
        call: &ToolCall,
    ) -> Result<VerifiedEffectPermit> {
        let token_id = token.get();
        let log = self.event_log.as_ref().ok_or_else(|| {
            HideError::PolicyDenied(
                "fixture token sink cannot authorize a live model-origin effect".into(),
            )
        })?;
        let event = log
            .append_target_verified(self.session_id.clone(), token, "")
            .await?;
        self.recorded
            .push(HostDurableRecord::CanonicalEvent { token_id });
        Ok(VerifiedEffectPermit {
            session_id: self.session_id.clone(),
            run_id,
            token_id,
            canonical_event_id: event.id,
            tool_call_digest: tool_call_digest(call),
        })
    }

    /// Durable memory write — Verified only.
    pub fn write_memory(&mut self, token: VerifiedTokenId) -> Result<()> {
        self.recorded.push(HostDurableRecord::MemoryWrite {
            token_id: token.get(),
        });
        Ok(())
    }

    /// Tool dispatch — Verified only.
    pub fn dispatch_tool(&mut self, token: VerifiedTokenId) -> Result<()> {
        self.recorded.push(HostDurableRecord::ToolDispatch {
            token_id: token.get(),
        });
        Ok(())
    }

    /// File edit — Verified only.
    pub fn edit_file(&mut self, token: VerifiedTokenId) -> Result<()> {
        self.recorded.push(HostDurableRecord::FileEdit {
            token_id: token.get(),
        });
        Ok(())
    }

    /// Final user-visible output — Verified only.
    pub fn final_output(&mut self, token: VerifiedTokenId, text: &str) -> Result<()> {
        self.recorded.push(HostDurableRecord::FinalOutput {
            token_id: token.get(),
        });
        self.final_text.push_str(text);
        Ok(())
    }

    /// Apply the suspension policy at a detected boundary. Returns whether the
    /// host may continue proposing drafts.
    pub fn on_boundary(boundary: SuspensionBoundary) -> SpeculationPermit {
        evaluate(Some(boundary))
    }

    /// Static policy table (one list, one place — re-exported for host callers).
    pub fn policy() -> &'static [hawking_speculate::SuspensionPoint] {
        SUSPENSION_POLICY
    }

    pub fn action(boundary: SuspensionBoundary) -> SuspensionAction {
        action_for(boundary)
    }
}

/// Sync adapter implementing the leaf-crate [`DurableTokenSink`] trait so the
/// host sinks satisfy the same interface as the in-memory fixture.
impl DurableTokenSink for HostDurableSinks {
    fn emit_canonical_event(
        &mut self,
        token: VerifiedTokenId,
    ) -> hawking_speculate::durable::DurableResult<()> {
        let token_id = token.get();
        if self.event_log.is_some() {
            return Err(hawking_speculate::durable::DurableSinkError::Refused(
                "live canonical token append is async; sync adapter cannot acknowledge durability"
                    .into(),
            ));
        }
        self.recorded
            .push(HostDurableRecord::CanonicalEvent { token_id });
        // Async event-log append is not available on the sync trait; live
        // callers use `HostDurableSinks::emit_canonical_event` (async).
        Ok(())
    }

    fn write_memory(
        &mut self,
        token: VerifiedTokenId,
    ) -> hawking_speculate::durable::DurableResult<()> {
        self.recorded.push(HostDurableRecord::MemoryWrite {
            token_id: token.get(),
        });
        Ok(())
    }

    fn dispatch_tool(
        &mut self,
        token: VerifiedTokenId,
    ) -> hawking_speculate::durable::DurableResult<()> {
        self.recorded.push(HostDurableRecord::ToolDispatch {
            token_id: token.get(),
        });
        Ok(())
    }

    fn edit_file(
        &mut self,
        token: VerifiedTokenId,
    ) -> hawking_speculate::durable::DurableResult<()> {
        self.recorded.push(HostDurableRecord::FileEdit {
            token_id: token.get(),
        });
        Ok(())
    }

    fn final_output(
        &mut self,
        token: VerifiedTokenId,
    ) -> hawking_speculate::durable::DurableResult<()> {
        self.recorded.push(HostDurableRecord::FinalOutput {
            token_id: token.get(),
        });
        Ok(())
    }
}

/// Map a trait-level refuse into a host error (for async wrappers).
#[allow(dead_code)]
fn map_refuse(e: hawking_speculate::durable::DurableSinkError) -> HideError {
    HideError::msg(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hawking_speculate::{DraftTokenId, DualKv, TargetVerification};
    #[test]
    fn host_sinks_accept_verified_only_path() {
        let gate = TargetVerification::gate();
        let mut sinks = HostDurableSinks::fixture(SessionId::from("sess-spec-safe"));
        let draft = DraftTokenId::id(5);
        let verified = gate.try_promote(draft, 5).expect("match");
        DurableTokenSink::emit_canonical_event(&mut sinks, verified).unwrap();
        DurableTokenSink::write_memory(&mut sinks, verified).unwrap();
        DurableTokenSink::dispatch_tool(&mut sinks, verified).unwrap();
        DurableTokenSink::edit_file(&mut sinks, verified).unwrap();
        DurableTokenSink::final_output(&mut sinks, verified).unwrap();
        assert_eq!(sinks.recorded().len(), 5);
        assert!(gate.try_promote(DraftTokenId::id(1), 2).is_err());
        assert_eq!(sinks.recorded().len(), 5);
    }

    #[tokio::test]
    async fn fixture_sink_cannot_mint_a_live_effect_permit() {
        let gate = TargetVerification::gate();
        let mut sinks = HostDurableSinks::fixture(SessionId::from("fixture-effect"));
        let verified = gate.emit_target(7);
        assert!(sinks
            .authorize_verified_tool_effect(
                verified,
                None,
                &hide_core::tool::ToolCall::new("fs.read", serde_json::json!({"path": "x"})),
            )
            .await
            .is_err());
    }
    #[test]
    fn host_honours_every_suspension_boundary() {
        for point in HostDurableSinks::policy() {
            let permit = HostDurableSinks::on_boundary(point.boundary);
            match point.action {
                SuspensionAction::Suspend | SuspensionAction::FlushThenContinue => {
                    assert!(
                        !permit.may_propose(),
                        "{:?} must suspend proposal",
                        point.boundary
                    );
                }
                SuspensionAction::Constrain => {
                    assert!(
                        matches!( permit, SpeculationPermit::Constrained { boundary } if boundary == point.boundary )
                    );
                }
            }
        }
    }
    #[test]
    fn dual_kv_rollback_via_host_gate_keeps_committed() {
        let gate = HostDurableSinks::target_gate();
        let mut dual = DualKv::new(hawking_speculate::CommittedKv::from_tokens(vec![1, 2]));
        let fp = dual.committed().fingerprint().clone();
        dual.speculate(&[9, 8]);
        dual.rollback();
        assert_eq!(dual.committed().fingerprint(), &fp);
        let mut sinks = HostDurableSinks::fixture(SessionId::from("s"));
        dual.rebase(&[gate.emit_target(3u32)]);
        for &id in dual.committed().token_ids() {
            let v = gate.emit_target(id);
            DurableTokenSink::emit_canonical_event(&mut sinks, v).unwrap();
        }
        assert_eq!(
            sinks
                .recorded()
                .iter()
                .filter_map(|r| match r {
                    HostDurableRecord::CanonicalEvent { token_id } => Some(*token_id),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }
}

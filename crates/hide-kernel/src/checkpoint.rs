//! Checkpointing (bible ch.02 §4.13) — three operations, one mechanism.
//!
//! Snapshot / restore / resume / fork are all the *same* deterministic fold over
//! the event log (K2). A checkpoint is a serialized [`AgentState`] tagged with
//! the log `seq` it was taken at; restoring re-establishes that state, resuming
//! continues a live run from it, and forking clones it under a fresh `run_id` so
//! an alternate continuation can be explored without disturbing the original.

use crate::machine::state::AgentState;
use crate::projection::{empty_projection, BasicProjectionEngine, ProjectionEngine};
use crate::session::SessionProjection;
use hide_core::event::Event;
use hide_core::ids::{RunId, SessionId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentCheckpoint {
    pub session_id: SessionId,
    /// The log `seq` this snapshot was taken at (the fold boundary).
    pub seq: u64,
    pub state: AgentState,
    pub source_event: Option<Event>,
}

impl AgentCheckpoint {
    /// Snapshot the current state at a given log `seq`.
    pub fn snapshot(state: &AgentState, seq: u64) -> Self {
        Self {
            session_id: state.session_id.clone(),
            seq,
            state: state.clone(),
            source_event: None,
        }
    }

    /// Restore the exact state captured (the fold's output is the state itself).
    pub fn restore(&self) -> AgentState {
        self.state.clone()
    }

    /// Resume: restore and continue the *same* run (live).
    pub fn resume(&self) -> AgentState {
        self.restore()
    }

    /// Fork: restore under a *new* run id so an alternate continuation can be
    /// explored without disturbing the original (scrub-to-event + branch).
    pub fn fork(&self) -> AgentState {
        let mut forked = self.restore();
        forked.run_id = RunId::new();
        forked.pending_approval = None;
        forked
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayRequest {
    pub session_id: SessionId,
    pub target_seq: u64,
    pub live_resume: bool,
}

/// Rebuild a [`SessionProjection`] by folding the event log up to (and including)
/// `target_seq` — the deterministic replay fold (K2). Effects are never re-fired
/// during this fold; only recorded outcomes are applied.
pub fn fold_to_seq(
    session_id: SessionId,
    events: &[Event],
    target_seq: u64,
) -> hide_core::Result<SessionProjection> {
    let engine = BasicProjectionEngine;
    let upto: Vec<Event> = events
        .iter()
        .filter(|e| e.seq <= target_seq)
        .cloned()
        .collect();
    engine.fold(empty_projection(session_id), &upto)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::ids::SessionId;

    fn state() -> AgentState {
        AgentState::new(SessionId::new(), RunId::new(), "obj".to_string())
    }

    #[test]
    fn snapshot_restore_round_trips() {
        let mut s = state();
        s.ledger.steps = 7;
        let cp = AgentCheckpoint::snapshot(&s, 42);
        let restored = cp.restore();
        assert_eq!(restored.ledger.steps, 7);
        assert_eq!(cp.seq, 42);
    }

    #[test]
    fn fork_changes_run_id_only() {
        let s = state();
        let cp = AgentCheckpoint::snapshot(&s, 1);
        let forked = cp.fork();
        assert_ne!(forked.run_id, s.run_id);
        assert_eq!(forked.session_id, s.session_id);
        assert_eq!(forked.objective, s.objective);
    }
}

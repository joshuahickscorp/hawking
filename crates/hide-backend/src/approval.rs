//! The approval hub: the seam an approval decision uses to release a *paused*
//! effectful kernel turn (bible ch.02 §4.3 autonomy / §78.1 #7 effect+approval).
//!
//! Under the bounded `Autonomy::SuggestOnly` a live turn runs under, the kernel
//! driver pauses on an effectful step (`Phase::Paused` + `state.pending_approval`)
//! and resumes only once that approval is "cleared by the host out-of-band"
//! (`hide_kernel::machine::driver::AgentDriver::do_paused`). Nothing in the host
//! delivered that decision, so an effectful turn spun `Paused` until the Governor
//! aborted at the step cap. [`ApprovalHub`] is the missing mailbox: it mirrors
//! [`crate::interrupt::InterruptHub`] exactly: a per-`run_id` slot the host
//! deposits a decision into and the running turn drains between transitions.
//!
//! Flow: the [`crate::host::BackendHost`] intent router calls [`ApprovalHub::decide`]
//! when it handles an `approve_effect`/`deny_effect` intent; the kernel-turn loop
//! calls [`ApprovalHub::take`] while paused and, on a hit, clears the approval via
//! the driver's sanctioned `AgentState::approve_pending_effect` /
//! `deny_pending_effect`. Decisions are *buffered*: one deposited before the turn
//! reaches its pause is still consumed once it pauses (mirroring how a `Cancel`
//! that arrives early still aborts the run once it starts polling).

use hide_core::ids::{RunId, StepId};
use parking_lot::Mutex;
use std::collections::HashMap;

/// A human decision on a paused effectful step (§4.3). Never auto-derived: the
/// host only ever deposits one from an explicit `approve_effect`/`deny_effect`
/// intent, so a turn can never self-approve its own effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalDecision {
    /// Let the effect run: the driver resumes `Paused -> Act`.
    Approve,
    /// Skip the effect: the driver marks the step skipped and reselects.
    Deny,
}

/// A decision plus the step it targets, so a stale decision for a step that is no
/// longer the pending one is ignored rather than mis-applied.
#[derive(Debug, Clone)]
struct ApprovalMessage {
    /// The step the decision names, or `None` to resolve whatever is pending.
    step_id: Option<StepId>,
    decision: ApprovalDecision,
}

/// A per-`run_id` approval mailbox. Last-write-wins per run (only one effectful
/// step is ever pending at a time, since the FSM pauses the whole run).
#[derive(Default)]
pub struct ApprovalHub {
    pending: Mutex<HashMap<RunId, ApprovalMessage>>,
}

impl ApprovalHub {
    /// Deposit a decision for `run_id`, optionally scoped to the `step_id` the
    /// `approval.requested` event carried. Last-write-wins.
    pub fn decide(&self, run_id: RunId, step_id: Option<StepId>, decision: ApprovalDecision) {
        self.pending
            .lock()
            .insert(run_id, ApprovalMessage { step_id, decision });
    }

    /// Take (and clear) any decision buffered for `run_id`. Returns the decision
    /// and the step it named (`None` step = "resolve whatever is pending").
    /// Called by the turn loop while paused.
    pub fn take(&self, run_id: &RunId) -> Option<(Option<StepId>, ApprovalDecision)> {
        self.pending
            .lock()
            .remove(run_id)
            .map(|m| (m.step_id, m.decision))
    }

    /// Whether a decision is buffered for `run_id` without consuming it.
    pub fn is_pending(&self, run_id: &RunId) -> bool {
        self.pending.lock().contains_key(run_id)
    }

    /// Drop any buffered decision for `run_id` (e.g. the run ended).
    pub fn clear(&self, run_id: &RunId) {
        self.pending.lock().remove(run_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decide_then_take_returns_the_decision_and_clears() {
        let hub = ApprovalHub::default();
        let run = RunId::new();
        let step = StepId::new();
        hub.decide(run.clone(), Some(step.clone()), ApprovalDecision::Approve);
        assert!(hub.is_pending(&run));
        assert_eq!(
            hub.take(&run),
            Some((Some(step), ApprovalDecision::Approve))
        );
        // Mailbox is now empty (single-use).
        assert!(!hub.is_pending(&run));
        assert!(hub.take(&run).is_none());
    }

    #[test]
    fn last_write_wins_per_run() {
        let hub = ApprovalHub::default();
        let run = RunId::new();
        hub.decide(run.clone(), None, ApprovalDecision::Approve);
        hub.decide(run.clone(), None, ApprovalDecision::Deny);
        assert_eq!(hub.take(&run), Some((None, ApprovalDecision::Deny)));
    }

    #[test]
    fn decisions_are_per_run_isolated() {
        let hub = ApprovalHub::default();
        let a = RunId::new();
        let b = RunId::new();
        hub.decide(a.clone(), None, ApprovalDecision::Approve);
        assert!(!hub.is_pending(&b));
        assert!(hub.take(&b).is_none());
        assert_eq!(hub.take(&a), Some((None, ApprovalDecision::Approve)));
    }

    #[test]
    fn clear_drops_a_buffered_decision() {
        let hub = ApprovalHub::default();
        let run = RunId::new();
        hub.decide(run.clone(), None, ApprovalDecision::Deny);
        hub.clear(&run);
        assert!(hub.take(&run).is_none());
    }
}

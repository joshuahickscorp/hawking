//! The interrupt hub — the seam control intents use to signal a running kernel
//! (bible ch.02 §4.3.2 / ch.07 §4.4).
//!
//! `Cancel`/`Pause`/`Resume` intents must do more than append a log line: they
//! must reach the *running* run and actually abort/pause it. The kernel already
//! exposes the receiving end — `hide_kernel::govern::Interrupt` plus
//! `AgentKernel::interrupt(..)` — which the driver polls between transitions
//! (K8). What was missing was a *per-run* mailbox the host could deposit signals
//! into and a running run could drain. [`InterruptHub`] is that mailbox.
//!
//! Flow: the [`crate::commands::CommandRouter`] calls [`InterruptHub::signal`]
//! when it handles a control intent; a run loop calls [`InterruptHub::take`]
//! between transitions and, on a hit, calls `kernel.interrupt(..)` (or, for a
//! fleet run, flips the cooperative-cancel flag). Signals are *buffered*: a
//! `Cancel` that arrives before the run attaches still aborts it once it starts
//! polling.

use hide_core::ids::RunId;
use hide_kernel::govern::Interrupt;
use parking_lot::Mutex;
use std::collections::HashMap;

/// A per-`run_id` interrupt mailbox. Last-write-wins per run (an `Abort`
/// supersedes a pending `Pause`).
#[derive(Default)]
pub struct InterruptHub {
    pending: Mutex<HashMap<RunId, Interrupt>>,
}

impl InterruptHub {
    /// Deposit an interrupt for `run_id`. An `Abort` always wins over a buffered
    /// `Pause`/`Steer`; otherwise last-write-wins.
    pub fn signal(&self, run_id: RunId, interrupt: Interrupt) {
        let mut map = self.pending.lock();
        match map.get(&run_id) {
            // Never downgrade an Abort.
            Some(Interrupt::Abort) if !matches!(interrupt, Interrupt::Abort) => {}
            _ => {
                map.insert(run_id, interrupt);
            }
        }
    }

    /// Take (and clear) any pending interrupt for `run_id`. Called by the run
    /// loop between transitions.
    pub fn take(&self, run_id: &RunId) -> Option<Interrupt> {
        self.pending.lock().remove(run_id)
    }

    /// Peek whether an interrupt is pending without consuming it.
    pub fn is_pending(&self, run_id: &RunId) -> bool {
        self.pending.lock().contains_key(run_id)
    }

    /// Clear any pending interrupt (a `Resume` cancels a buffered `Pause`).
    pub fn clear(&self, run_id: &RunId) {
        self.pending.lock().remove(run_id);
    }

    /// Drain a run's interrupt into a live kernel: if one is pending, inject it
    /// via `AgentKernel::interrupt` so it's consumed on the next transition.
    /// Returns the interrupt that was forwarded (for observability).
    pub fn drain_into_kernel(
        &self,
        run_id: &RunId,
        kernel: &hide_kernel::AgentKernel,
    ) -> Option<Interrupt> {
        let interrupt = self.take(run_id)?;
        kernel.interrupt(interrupt.clone());
        Some(interrupt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abort_supersedes_pending_pause() {
        let hub = InterruptHub::default();
        let run = RunId::new();
        hub.signal(run.clone(), Interrupt::Pause);
        hub.signal(run.clone(), Interrupt::Abort);
        assert!(matches!(hub.take(&run), Some(Interrupt::Abort)));
    }

    #[test]
    fn pause_does_not_downgrade_an_abort() {
        let hub = InterruptHub::default();
        let run = RunId::new();
        hub.signal(run.clone(), Interrupt::Abort);
        hub.signal(run.clone(), Interrupt::Pause);
        assert!(matches!(hub.take(&run), Some(Interrupt::Abort)));
    }

    #[test]
    fn resume_clears_a_buffered_pause() {
        let hub = InterruptHub::default();
        let run = RunId::new();
        hub.signal(run.clone(), Interrupt::Pause);
        hub.clear(&run);
        assert!(hub.take(&run).is_none());
    }

    #[tokio::test]
    async fn drain_into_kernel_forwards_the_signal() {
        use hide_core::event::InMemoryEventLog;
        use std::sync::Arc;
        let log = Arc::new(InMemoryEventLog::new());
        let kernel = hide_kernel::AgentKernel::new(log);
        let hub = InterruptHub::default();
        let run = RunId::new();
        hub.signal(run.clone(), Interrupt::Abort);
        let forwarded = hub.drain_into_kernel(&run, &kernel);
        assert!(matches!(forwarded, Some(Interrupt::Abort)));
        // Mailbox is now empty.
        assert!(!hub.is_pending(&run));
    }
}

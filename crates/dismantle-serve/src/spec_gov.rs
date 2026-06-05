//! Track 6.3 — Speculative-decode acceptance governor.
//!
//! [`SpecGovernor`] tracks a per-session rolling acceptance rate and decides
//! whether spec-decode is currently helping. The actual enable/disable loop
//! in the serve path is a follow-on; for now the governor just accumulates
//! state so the infrastructure is wired and ready.

use std::collections::VecDeque;

/// Rolling acceptance tracker for spec-decode auto-enable/disable.
///
/// Per-session: tracks rolling accept rate and decides whether spec is helping.
pub struct SpecGovernor {
    /// Rolling window size for acceptance rate calculation.
    pub window: usize,
    /// Minimum acceptance rate to keep spec enabled (default 0.35).
    pub min_accept_rate: f32,
    /// Maximum consecutive zero-acceptance steps before disabling (default 5).
    pub max_consecutive_rejections: usize,
    // private rolling state
    accepted: VecDeque<bool>,
    consecutive_rejections: usize,
    pub enabled: bool,
}

impl SpecGovernor {
    /// Create a new governor with the given window and minimum acceptance rate.
    ///
    /// `window` — number of most-recent verify steps to average over.
    /// `min_accept_rate` — acceptance rate below which spec is considered unhelpful.
    pub fn new(window: usize, min_accept_rate: f32) -> Self {
        Self {
            window,
            min_accept_rate,
            max_consecutive_rejections: 5,
            accepted: VecDeque::with_capacity(window),
            consecutive_rejections: 0,
            enabled: true,
        }
    }

    /// Record the outcome of one verify step.
    ///
    /// Call after each spec-decode verify cycle. `accepted` is `true` when at
    /// least one draft token was accepted by the verifier.
    pub fn record(&mut self, accepted: bool) {
        if self.accepted.len() >= self.window {
            self.accepted.pop_front();
        }
        self.accepted.push_back(accepted);

        if accepted {
            self.consecutive_rejections = 0;
        } else {
            self.consecutive_rejections += 1;
        }

        // Auto-disable if we have exceeded the consecutive-rejection ceiling.
        if self.consecutive_rejections >= self.max_consecutive_rejections {
            self.enabled = false;
        }
        // Re-enable when rolling acceptance rate recovers above the threshold.
        if !self.enabled && self.accept_rate() >= self.min_accept_rate {
            self.enabled = true;
            self.consecutive_rejections = 0;
        }
    }

    /// Rolling acceptance rate over the last `window` steps.
    ///
    /// Returns 1.0 when no steps have been recorded yet (optimistic prior,
    /// so spec starts enabled).
    pub fn accept_rate(&self) -> f32 {
        if self.accepted.is_empty() {
            return 1.0;
        }
        let accepted_count = self.accepted.iter().filter(|&&v| v).count();
        accepted_count as f32 / self.accepted.len() as f32
    }

    /// Whether spec-decode should be used for the next draft proposal.
    ///
    /// Returns `false` once acceptance rate has fallen below `min_accept_rate`
    /// AND consecutive rejections have reached `max_consecutive_rejections`.
    pub fn should_enable(&self) -> bool {
        self.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_governor_starts_enabled_with_optimistic_rate() {
        let gov = SpecGovernor::new(20, 0.35);
        assert!(gov.enabled);
        assert!(gov.should_enable());
        assert_eq!(gov.accept_rate(), 1.0);
    }

    #[test]
    fn spec_governor_tracks_rolling_window() {
        let mut gov = SpecGovernor::new(4, 0.35);
        // 4 accepts then 4 rejections: window sees the 4 rejections.
        for _ in 0..4 {
            gov.record(true);
        }
        assert!((gov.accept_rate() - 1.0).abs() < 1e-6);
        for _ in 0..4 {
            gov.record(false);
        }
        assert!((gov.accept_rate() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn spec_governor_disables_after_max_consecutive_rejections() {
        let mut gov = SpecGovernor::new(20, 0.35);
        // 5 consecutive rejections → disabled.
        for _ in 0..5 {
            gov.record(false);
        }
        assert!(!gov.enabled);
        assert!(!gov.should_enable());
    }

    #[test]
    fn spec_governor_reenables_when_rate_recovers() {
        let mut gov = SpecGovernor::new(4, 0.35);
        // Disable first.
        for _ in 0..5 {
            gov.record(false);
        }
        assert!(!gov.enabled);
        // 3 consecutive accepts push the rolling window above 0.35.
        gov.record(true);
        gov.record(true);
        // Window has 4 entries: [false, false, true, true] → rate = 0.5 > 0.35.
        // The last record() call will re-enable.
        // But first two records after disable still hit the re-enable check.
        // Just verify the final state.
        assert!((gov.accept_rate() - 0.5).abs() < 1e-6);
        assert!(gov.enabled);
    }
}

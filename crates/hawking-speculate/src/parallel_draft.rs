//! Event Horizon Phase 5 — parallel-head draft scaffold.
//! SCAFFOLD, NEVER ENABLED without τ≥2.5 oracle GO (docs/dead_levers.md).
//!
//! Kill-ledger rule (structural in code):
//!   This proposer requires_hidden=true, so `enable_neural_slot` will refuse it
//!   unless verdict=="GO". No oracle has produced τ≥2.5 for a parallel head on
//!   any target in this codebase. `is_enabled()` also returns false unless
//!   HAWKING_EH_PARALLEL_DRAFT is explicitly set in the environment.
//!
//! Design: a P-EAGLE-H / DFlash-H style head that emits ALL k draft tokens in
//! ONE forward pass (1 model forward, not k autoregressive forwards). The CPU
//! stub always emits k zero-tokens. The real implementation would run the
//! parallel head once and decode all k positions simultaneously.
//!
//! Sub-flag: HAWKING_EH_PARALLEL_DRAFT (unset = disabled).

use crate::proposal::{Budget, CostNs, Ctx, Proposal, Proposer, Telemetry};

/// Maximum number of draft tokens this proposer will emit in one shot.
/// Mirrors `Budget::MAX_DRAFT_LEN` (7) — the verifier's B≤8 fast path.
const MAX_DRAFT_LEN: usize = Budget::MAX_DRAFT_LEN;

/// CPU stub for a parallel-head proposer.
///
/// SCAFFOLD ONLY — never enabled at runtime. Registration is always refused:
/// `enable_neural_slot(ProposerId::ParallelDraft, .., true, false, "NO-GO")` → `Err`.
///
/// A real parallel head emits ALL k tokens in ONE forward pass (not k AR
/// forwards), costing ~1× the verifier's per-token budget regardless of k.
/// This makes it fundamentally cheaper than autoregressive EAGLE for large k.
pub struct ParallelDraftProposer;

impl ParallelDraftProposer {
    /// Create the stub.
    pub fn new() -> Self {
        Self
    }

    /// Returns true only if `HAWKING_EH_PARALLEL_DRAFT` is set in the environment.
    ///
    /// This is a secondary gate on top of the router's oracle guard. Both must
    /// be satisfied before this proposer participates in any inference loop.
    pub fn is_enabled() -> bool {
        std::env::var("HAWKING_EH_PARALLEL_DRAFT").is_ok()
    }
}

impl Default for ParallelDraftProposer {
    fn default() -> Self {
        Self::new()
    }
}

impl Proposer for ParallelDraftProposer {
    fn name(&self) -> &'static str {
        "parallel_draft"
    }

    /// Always true — this proposer requires the target's hidden-state tap.
    /// The router uses this flag to gate registration behind an oracle GO.
    fn requires_hidden(&self) -> bool {
        true
    }

    /// Draft cost estimate.
    ///
    /// A real parallel head costs ~1 forward pass (not k), making it O(1) in k.
    /// The stub reports 0 so the router's expected_gain arithmetic is honest
    /// about a scaffold that never runs.
    fn cost_estimate(&self, _ctx: &Ctx<'_>, _budget: Budget) -> CostNs {
        0
    }

    /// Produce up to `budget.k` draft tokens in one shot.
    ///
    /// PARALLEL: emits k tokens in one shot (1 forward), not AR (k forwards).
    ///
    /// Stub behaviour: emits exactly `budget.k.min(MAX_DRAFT_LEN)` zero-tokens
    /// regardless of whether a hidden tap is present. A real implementation
    /// would run the parallel head's single forward pass here, using
    /// `ctx.hidden` to condition the parallel decode positions.
    ///
    /// NOTE: this code path is exercised by tests only. The router refuses to
    /// schedule ParallelDraft at runtime until `enable_neural_slot` is called
    /// with verdict="GO" (τ≥2.5), which no oracle has produced for any
    /// parallel head on current workloads.
    fn propose(&mut self, _ctx: &Ctx<'_>, budget: Budget, _tel: &Telemetry) -> Proposal {
        // PARALLEL: emits k tokens in one shot (1 forward), not AR (k forwards)
        let k = budget.k.min(MAX_DRAFT_LEN);
        // Stub: return k zero-tokens. Real impl would run the parallel head forward once.
        Proposal::TokenLine(vec![0u32; k])
    }

    /// No-op for the stub. A real parallel head would update its positional
    /// context here so the next propose() call is correctly conditioned.
    fn observe(&mut self, _emitted: &[u32]) {}

    /// No-op for the stub. A real parallel head might reset positional state here.
    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proposal::{Budget, Ctx, Telemetry};
    use crate::router::{ProposalRouter, ProposerId};

    fn ctx_no_hidden<'a>(tokens: &'a [u32]) -> Ctx<'a> {
        Ctx {
            tokens,
            pos: tokens.len(),
            hidden: None,
        }
    }

    /// The scaffold is disabled by default (HAWKING_EH_PARALLEL_DRAFT not set).
    #[test]
    fn parallel_draft_scaffold_is_disabled() {
        // This test must NOT set the env var — it verifies the default-off state.
        // If the env var happens to be set in the test environment, skip gracefully.
        if std::env::var("HAWKING_EH_PARALLEL_DRAFT").is_ok() {
            // Env already set externally; can't assert false without side-effects.
            // The kill-ledger gate is still enforced by the router (see test below).
            return;
        }
        assert!(
            !ParallelDraftProposer::is_enabled(),
            "ParallelDraftProposer must be disabled when HAWKING_EH_PARALLEL_DRAFT is not set"
        );
    }

    /// propose() with budget.k=5 must return a TokenLine of exactly length 5.
    #[test]
    fn parallel_draft_emits_correct_budget() {
        let mut proposer = ParallelDraftProposer::new();
        let tokens = [1u32, 2, 3];
        let ctx = ctx_no_hidden(&tokens);
        let tel = Telemetry::default();
        let proposal = proposer.propose(&ctx, Budget::line(5), &tel);
        match proposal {
            Proposal::TokenLine(v) => {
                assert_eq!(v.len(), 5, "expected k=5 drafts from parallel head stub")
            }
            other => panic!("expected TokenLine, got len={}", other.draft_len()),
        }
    }

    /// requires_hidden() must return true — the router uses this to gate scheduling.
    #[test]
    fn parallel_draft_requires_hidden() {
        let proposer = ParallelDraftProposer::new();
        assert!(
            proposer.requires_hidden(),
            "ParallelDraftProposer must advertise requires_hidden=true"
        );
    }

    /// The kill-ledger rule encoded structurally: enable_neural_slot refuses
    /// any hidden slot whose oracle verdict is not "GO".
    #[test]
    fn parallel_draft_enable_refused_without_go() {
        let mut router = ProposalRouter::new(16, 0.35, 1.0);
        // Attempting to register ParallelDraft with "NO-GO" must return Err.
        let result = router.enable_neural_slot(
            ProposerId::ParallelDraft,
            16,
            0.35,
            true,  // requires_hidden
            false, // requires_text_bridge
            "NO-GO",
        );
        assert!(
            result.is_err(),
            "enable_neural_slot must refuse hidden slot with verdict=NO-GO"
        );
    }

    /// propose() caps at MAX_DRAFT_LEN even if budget.k exceeds it.
    /// Budget::line() already enforces this, but verify the full chain.
    #[test]
    fn parallel_draft_shape_and_budget_respected() {
        let mut proposer = ParallelDraftProposer::new();
        let tokens = [0u32; 10];
        let ctx = ctx_no_hidden(&tokens);
        let tel = Telemetry::default();

        // Budget::line caps at MAX_DRAFT_LEN=7; request 10 to exercise the cap.
        let budget = Budget::line(10);
        assert!(
            budget.k <= MAX_DRAFT_LEN,
            "Budget::line must cap k at MAX_DRAFT_LEN={}",
            MAX_DRAFT_LEN
        );

        let proposal = proposer.propose(&ctx, budget, &tel);
        match proposal {
            Proposal::TokenLine(v) => {
                assert!(
                    v.len() <= MAX_DRAFT_LEN,
                    "propose must never emit more than MAX_DRAFT_LEN={} tokens, got {}",
                    MAX_DRAFT_LEN,
                    v.len()
                );
            }
            other => panic!("expected TokenLine, got len={}", other.draft_len()),
        }
    }
}

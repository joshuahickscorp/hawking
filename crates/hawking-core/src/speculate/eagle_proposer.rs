//! Event Horizon Phase 4 — EAGLE hidden-state draft adapter.
//! SCAFFOLD, NEVER ENABLED without τ≥2.5 oracle GO (docs/dead_levers.md).
//!
//! Kill-ledger entry (docs/dead_levers.md EAGLE-3 / EAGLE-5 head):
//!   τ=0.877 < 2.5 gate — NO-GO.
//!   Net-negative on Qwen-3B: K=2/4/8 → 0.40×/0.30×/0.21× baseline tps.
//!   Free n-gram (τ=1.43) beats the trained head on code.
//!
//! This adapter exists solely so the runtime dispatch path and wiring tests
//! can be exercised without a trained checkpoint. It MUST NOT be registered
//! via `enable_neural_slot` with verdict="GO" unless a replay oracle produces
//! τ≥2.5 on the target workload. The `enable_neural_slot` guard enforces
//! this structurally (returns Err for any hidden slot with verdict≠"GO").

use crate::speculate::eagle5::Eagle5Head;
use crate::speculate::proposal::{Budget, CostNs, Ctx, Proposal, Proposer, Telemetry};

/// Adapter that wraps `Eagle5Head` behind the `Proposer` trait.
///
/// SCAFFOLD ONLY — never enabled at runtime. Registration is always refused:
/// `enable_neural_slot(ProposerId::Eagle5, .., true, false, "NO-GO")` → `Err`.
/// See kill ledger: τ=0.877 < 2.5 gate, net-negative tps on Qwen-3B.
pub struct EagleProposer {
    head: Eagle5Head,
}

impl EagleProposer {
    /// Wrap an existing `Eagle5Head` (mock or trained).
    pub fn new(head: Eagle5Head) -> Self {
        Self { head }
    }
}

impl Proposer for EagleProposer {
    fn name(&self) -> &'static str {
        "eagle5"
    }

    /// Always true — this proposer requires the target's hidden-state tap.
    /// The router uses this flag to gate registration behind an oracle GO.
    fn requires_hidden(&self) -> bool {
        true
    }

    /// Draft cost estimate. Mock head is CPU-only and negligible; report 0
    /// so the router's expected_gain arithmetic is honest about the scaffold.
    fn cost_estimate(&self, _ctx: &Ctx<'_>, _budget: Budget) -> CostNs {
        0
    }

    /// Produce up to `budget.k` draft tokens.
    ///
    /// When `ctx.hidden` is `Some`, uses `propose_rollout_chained` which feeds
    /// the captured residual+intermediate into the head's in_proj (Phase 4
    /// chained-hidden rollout). When `ctx.hidden` is `None` (base path /
    /// capture-off / first cycle), falls back to simple AR `propose`.
    ///
    /// NOTE: this code path is exercised by tests only. The router refuses
    /// to schedule Eagle5 at runtime until a replay oracle returns "GO"
    /// (τ≥2.5), which the kill-ledger records as permanently not the case
    /// for current heads on Qwen-3B.
    fn propose(&mut self, ctx: &Ctx<'_>, budget: Budget, _tel: &Telemetry) -> Proposal {
        let k = budget.k;
        if k == 0 {
            return Proposal::TokenLine(Vec::new());
        }
        match ctx.hidden.as_ref() {
            Some(tap) => {
                // Phase 4 path: chained-hidden rollout from the verifier's
                // capture. propose_rollout_chained takes &self (not &mut self)
                // because the chained state is local to the call.
                let drafts = self.head.propose_rollout_chained(
                    tap.start_token,
                    tap.residual,
                    tap.intermediate,
                    k,
                );
                Proposal::TokenLine(drafts)
            }
            None => {
                // Fallback: simple AR propose (for tests without capture).
                // Uses last_token via note_token state if available, else
                // seeds from the trailing context token.
                let start = ctx.tokens.last().copied().unwrap_or(0);
                let drafts = self.head.propose(start, k);
                Proposal::TokenLine(drafts)
            }
        }
    }

    /// Feed every emitted token back to the head so it can seed the next
    /// draft window via `last_token`. Called for EVERY verifier-emitted token,
    /// even while the router has this proposer disabled.
    fn observe(&mut self, emitted: &[u32]) {
        for &t in emitted {
            self.head.note_token(t);
        }
    }

    /// Clear per-sequence state between generation requests. Delegates to
    /// `Eagle5Head::reset()` which zeroes `last_token`.
    fn reset(&mut self) {
        self.head.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculate::proposal::{Budget, Ctx, HiddenTap, Telemetry};
    use crate::speculate::router::{ProposalRouter, ProposerId};

    fn ctx_no_hidden<'a>(tokens: &'a [u32]) -> Ctx<'a> {
        Ctx {
            tokens,
            pos: tokens.len(),
            hidden: None,
        }
    }

    /// Basic smoke test: mock head wraps, propose returns the expected length.
    #[test]
    fn adapter_wraps_mock_head() {
        let head = Eagle5Head::mock(42, 64, 256);
        let mut proposer = EagleProposer::new(head);
        let tokens = [1u32, 2, 3];
        let ctx = ctx_no_hidden(&tokens);
        let tel = Telemetry::default();
        let proposal = proposer.propose(&ctx, Budget::line(3), &tel);
        match proposal {
            Proposal::TokenLine(v) => assert_eq!(v.len(), 3, "expected k=3 drafts"),
            other => panic!("expected TokenLine, got {:?}", other.draft_len()),
        }
    }

    /// requires_hidden must be true — the router uses this to gate scheduling.
    #[test]
    fn requires_hidden_is_true() {
        let head = Eagle5Head::mock(1, 32, 64);
        let proposer = EagleProposer::new(head);
        assert!(
            proposer.requires_hidden(),
            "EagleProposer must advertise requires_hidden=true"
        );
    }

    /// The kill-ledger rule encoded structurally: enable_neural_slot refuses
    /// any hidden slot whose oracle verdict is not "GO".
    ///
    /// τ=0.877 < 2.5 gate → verdict is always "NO-GO" for this head.
    #[test]
    fn enable_neural_slot_refuses_without_go() {
        let mut router = ProposalRouter::new(16, 0.35, 1.0);
        // Attempting to register Eagle5 with "NO-GO" must return Err.
        let result = router.enable_neural_slot(
            ProposerId::Eagle5,
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

    /// When a HiddenTap is present, propose delegates to propose_rollout_chained.
    /// Mock head ignores residual/intermediate contents, so zero-length slices work.
    #[test]
    fn propose_with_hidden_tap_uses_rollout() {
        let head = Eagle5Head::mock(7, 32, 128);
        let mut proposer = EagleProposer::new(head);
        let tokens = [10u32, 20, 30, 40, 5];
        let residual: &[f32] = &[];
        let intermediate: &[f32] = &[];
        let tap = HiddenTap {
            residual,
            intermediate,
            start_token: 5,
        };
        let ctx = Ctx {
            tokens: &tokens,
            pos: tokens.len(),
            hidden: Some(tap),
        };
        let tel = Telemetry::default();
        let proposal = proposer.propose(&ctx, Budget::line(4), &tel);
        match proposal {
            Proposal::TokenLine(v) => assert_eq!(v.len(), 4, "expected k=4 drafts from rollout"),
            other => panic!("expected TokenLine, got {:?}", other.draft_len()),
        }
    }
}

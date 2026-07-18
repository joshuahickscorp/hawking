//! Event Horizon Phase 7 — cross-tokenizer proposer scaffold (GATED, DISABLED).
//! requires_text_bridge()=true → enable_neural_slot refuses without GO.
//! Default: NO-GO (0.58-0.70× slowdown on Apple Silicon, ref 2604.16368).

use crate::proposal::{Budget, CostNs, Ctx, Proposal, Proposer, Telemetry};
use std::collections::HashMap;

/// Scaffold cross-tokenizer proposer. DISABLED by default.
///
/// Translation between source and target token vocabularies is mediated by a
/// detok→retok text bridge. On Apple Silicon this round-trip costs 0.58–0.70×
/// throughput relative to a token-native proposer (ref 2604.16368), so this slot
/// is permanently gated: `enable_neural_slot` refuses `requires_text_bridge=true`
/// unless the offline oracle verdict is "GO".
///
/// The span map is populated via `learn_span` or `warm`; `propose` looks up the
/// last context token and returns up to `budget.k` mapped continuations.
#[derive(Debug, Clone)]
pub struct CrossTokenizerProposer {
    /// src_token → [dst_tokens]: a many-to-many span table built offline or online.
    span_map: HashMap<u32, Vec<u32>>,
    /// Monotonically increasing count of `learn_span` calls (for observability).
    spans_learned: usize,
}

impl Default for CrossTokenizerProposer {
    fn default() -> Self {
        Self {
            span_map: HashMap::new(),
            spans_learned: 0,
        }
    }
}

impl CrossTokenizerProposer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an explicit src→dst mapping (e.g. from an offline alignment pass).
    pub fn learn_span(&mut self, src: u32, dst_tokens: Vec<u32>) {
        self.span_map.insert(src, dst_tokens);
        self.spans_learned += 1;
    }

    /// Number of `learn_span` registrations seen so far.
    pub fn spans_learned(&self) -> usize {
        self.spans_learned
    }
}

impl Proposer for CrossTokenizerProposer {
    fn name(&self) -> &'static str {
        "cross_tokenizer"
    }

    /// This proposer requires a detok→retok text bridge. `enable_neural_slot` will
    /// refuse to activate this slot unless the offline oracle verdict is "GO".
    fn requires_text_bridge(&self) -> bool {
        true
    }

    /// Cost estimate: 1 ms. Structurally disfavors this slot even if the gate opens,
    /// because the text bridge overhead is large relative to the token-native base.
    fn cost_estimate(&self, _ctx: &Ctx<'_>, _budget: Budget) -> CostNs {
        1_000_000 // 1 ms — disfavors this slot even if somehow enabled
    }

    /// SCAFFOLD: gated by enable_neural_slot. Without GO, never reached.
    ///
    /// Looks up `ctx.tokens.last()` in the span map and returns up to `budget.k`
    /// mapped dst tokens as a `Proposal::TokenLine`.
    fn propose(&mut self, ctx: &Ctx<'_>, budget: Budget, _tel: &Telemetry) -> Proposal {
        let src = match ctx.tokens.last() {
            Some(&t) => t,
            None => return Proposal::TokenLine(vec![]),
        };
        let draft = self
            .span_map
            .get(&src)
            .map(|v| v.iter().copied().take(budget.k).collect())
            .unwrap_or_default();
        Proposal::TokenLine(draft)
    }

    /// Build an online src→dst map from consecutive token pairs in `history`.
    ///
    /// For each window `[a, b]` in `history`, records `a → [b]` (appending if
    /// `a` already maps to something). This mirrors the n-gram warm path and gives
    /// test coverage without a real alignment model.
    fn warm(&mut self, history: &[u32]) {
        for window in history.windows(2) {
            let (src, dst) = (window[0], window[1]);
            self.span_map.entry(src).or_default().push(dst);
        }
    }

    /// No per-step feedback needed at scaffold stage.
    fn observe(&mut self, _emitted: &[u32]) {}

    /// Map persists across turns (matches n-gram "reset context, keep index" pattern).
    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proposal::Telemetry;
    use crate::router::{ProposalRouter, ProposerId};

    #[test]
    fn map_learns_a_span() {
        let mut p = CrossTokenizerProposer::new();
        p.learn_span(42, vec![100, 101, 102]);

        let tokens = vec![42u32];
        let ctx = Ctx {
            tokens: &tokens,
            pos: 1,
            hidden: None,
        };
        let budget = Budget::line(2);
        let tel = Telemetry::default();

        match p.propose(&ctx, budget, &tel) {
            Proposal::TokenLine(v) => {
                assert_eq!(v, vec![100, 101], "should return first k=2 mapped tokens");
            }
            _ => panic!("expected TokenLine"),
        }
    }

    #[test]
    fn warm_builds_span_map_from_consecutive_pairs() {
        let mut p = CrossTokenizerProposer::new();
        p.warm(&[1, 2, 3, 4]);

        // warm([1,2,3,4]) → pairs (1→2), (2→3), (3→4)
        // propose ctx=[2], k=1 → first dst of 2 = 3
        let tokens = vec![2u32];
        let ctx = Ctx {
            tokens: &tokens,
            pos: 1,
            hidden: None,
        };
        let budget = Budget::line(1);
        let tel = Telemetry::default();

        match p.propose(&ctx, budget, &tel) {
            Proposal::TokenLine(v) => {
                assert_eq!(v, vec![3], "warm pair 2→3 should be first dst");
            }
            _ => panic!("expected TokenLine"),
        }
    }

    #[test]
    fn unknown_src_proposes_nothing() {
        let mut p = CrossTokenizerProposer::new();
        p.learn_span(1, vec![10, 11]);

        let tokens = vec![99u32]; // not in map
        let ctx = Ctx {
            tokens: &tokens,
            pos: 1,
            hidden: None,
        };
        let budget = Budget::line(4);
        let tel = Telemetry::default();

        match p.propose(&ctx, budget, &tel) {
            Proposal::TokenLine(v) => {
                assert!(v.is_empty(), "unknown src must produce empty draft");
            }
            _ => panic!("expected TokenLine"),
        }
    }

    #[test]
    fn enable_neural_slot_refuses_text_bridge_without_go() {
        let mut router = ProposalRouter::new(16, 0.35, 1.0);
        let result = router.enable_neural_slot(
            ProposerId::CrossTokenizer,
            16,
            0.35,
            false, // requires_hidden
            true,  // requires_text_bridge
            "NO-GO",
        );
        assert!(
            result.is_err(),
            "text-bridge slot must be refused when oracle verdict is not GO"
        );
    }
}

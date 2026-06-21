//! Wall-clock-optimizing proposal router. Generalizes SpecGovernor from one
//! optional accept-rate gate to N per-proposer hysteresis machines under a
//! wall-clock expected_gain arbiter. Pure CPU logic — the loop feeds it measured
//! ns and it returns a plan. Losslessness is independent of the router; it only
//! chooses whether/how much to propose.

use super::governor::SpecGovernor;

pub const MAX_VERIFY_BATCH: usize = 8;
pub const MAX_DRAFT_LEN: usize = MAX_VERIFY_BATCH - 1; // 7, matches k_la cap

/// Cost of a B-token batched verify in canonical-greedy-forward units, from the
/// `verify_cost_vs_k` microbench (qwen_dense.rs). B=1 routes to the greedy kernel
/// (=1.0); B>=2 amortizes sub-linearly (B=8 ≈ 4.15, the 1.93x ideal-speedup
/// point). A spec cycle emits ~accepted_prefix tokens for ~verify_cost(B) extra
/// forwards, so it only pays when the accepted prefix exceeds this. THIS is why
/// the free market was net-negative: short drafts (low B / low accept) never
/// clear the curve, and the old confidence-based plan_shape ignored it.
fn verify_cost_forwards(b: usize) -> f32 {
    match b {
        0 => 0.0,
        1 => 1.0,
        2 => 2.20,
        3 => 2.70,
        4 => 3.25,
        5 => 3.62,
        6 => 3.77,
        7 => 4.00,
        _ => 4.15,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProposerId {
    UserNgram,
    SuffixArray,
    Eagle5,
    Rest,
    CrossTokenizer,
    Retrieval,      // Phase 2 — REST-style retrieval proposer
    Tree,           // Phase 6 — token-tree CPU fallback
    ParallelDraft,  // Phase 5 — parallel-head scaffold (HAWKING_EH_PARALLEL_DRAFT; kill-ledger: τ≥2.5 required)
}
impl ProposerId {
    pub fn as_str(self) -> &'static str {
        match self {
            ProposerId::UserNgram => "user_ngram",
            ProposerId::SuffixArray => "suffix_array",
            ProposerId::Eagle5 => "eagle5",
            ProposerId::Rest => "rest",
            ProposerId::CrossTokenizer => "cross_tokenizer",
            ProposerId::Retrieval => "retrieval",
            ProposerId::Tree => "tree",
            ProposerId::ParallelDraft => "parallel_draft",
        }
    }
}

/// Per-cycle measurements the loop hands back after each verify.
#[derive(Debug, Clone, Copy, Default)]
pub struct StepObservation {
    pub accepted: usize,         // na / first_reject (qwen_dense.rs:2632)
    pub drafted: usize,          // draft_len
    pub draft_ns: u64,           // wrap propose()
    pub verify_extra_ns: u64,    // B-token verify minus the 1 fwd you'd run anyway
    pub retokenize_ns: u64,      // 0 for token-native proposers
    pub sync_ns: u64,            // GPU submit/commit/wait
}

/// Target/context signals the loop fills each step.
#[derive(Debug, Clone, Copy)]
pub struct RouterCtx {
    pub target_ns_per_token: f32, // value an accepted draft token SAVES (small on
                                  // fast Qwen-3B → auto-kills neural spec there)
    pub context_confidence: f32,  // [0,1]; higher ⇒ longer draft (EAGLE-2 length)
    pub hidden_available: bool,   // gates any requires_hidden proposer this step
}

#[derive(Debug, Clone, PartialEq)]
pub enum RouterPlan {
    NoSpec,                                          // plain single-token greedy
    Spec { id: ProposerId, draft_len: usize, tree_width: usize },
}

#[derive(Debug, Clone, Copy)]
struct CostModel {
    ewma_accept_len: f32, ewma_draft_ns: f32, ewma_verify_extra_ns: f32,
    ewma_retok_ns: f32, ewma_sync_ns: f32, ewma_hit_frac: f32, seen: u64,
}
impl CostModel {
    fn new() -> Self {
        // ewma_accept_len seeded optimistically (4.0, mid-curve) so a fresh slot
        // explores (specs at B≈4) and converges to its real accept; a cold 1.0
        // seed would clear no payoff and the slot would never spec → never learn.
        Self { ewma_accept_len: 4.0, ewma_draft_ns: 0.0, ewma_verify_extra_ns: 0.0,
               ewma_retok_ns: 0.0, ewma_sync_ns: 0.0, ewma_hit_frac: 1.0, seen: 0 }
    }
    fn update(&mut self, o: &StepObservation, alpha: f32) {
        let mix = |old: f32, new: f32| old + alpha * (new - old);
        self.ewma_accept_len = mix(self.ewma_accept_len, o.accepted as f32);
        self.ewma_draft_ns = mix(self.ewma_draft_ns, o.draft_ns as f32);
        self.ewma_verify_extra_ns = mix(self.ewma_verify_extra_ns, o.verify_extra_ns as f32);
        self.ewma_retok_ns = mix(self.ewma_retok_ns, o.retokenize_ns as f32);
        self.ewma_sync_ns = mix(self.ewma_sync_ns, o.sync_ns as f32);
        let hit = if o.drafted > 0 { o.accepted as f32 / o.drafted as f32 } else { 0.0 };
        self.ewma_hit_frac = mix(self.ewma_hit_frac, hit);
        self.seen += 1;
    }
}

struct Slot {
    id: ProposerId,
    gov: SpecGovernor,
    cost: CostModel,
    requires_hidden: bool,
    #[allow(dead_code)]
    requires_text_bridge: bool,
    /// true only after replay_oracle verdict == "GO" (τ≥2.5) on the target
    /// workload. n-gram base = true unconditionally; any requires_hidden slot
    /// stays false until gated. THE KILL-LEDGER RULE IN CODE.
    oracle_cleared: bool,
}

pub struct ProposalRouter {
    slots: Vec<Slot>,
    alpha: f32,
    #[allow(dead_code)]
    margin_ns: f32,
    /// Payoff floor in greedy-forward units: a slot specs only when its best B
    /// saves more than this per cycle (avoids marginal specs the per-cycle
    /// overhead/variance would eat). Tunable; 0.5 ≈ "must clearly pay".
    margin_forwards: f32,
    #[allow(dead_code)]
    bandit: crate::speculate::policy::BanditPolicy,
}

impl ProposalRouter {
    /// Build with the always-on n-gram base. Neural/cross slots via enable_neural_slot.
    pub fn new(window: usize, min_accept_rate: f32, margin_ns: f32) -> Self {
        let base = Slot {
            id: ProposerId::UserNgram, gov: SpecGovernor::new(window, min_accept_rate),
            cost: CostModel::new(), requires_hidden: false, requires_text_bridge: false,
            oracle_cleared: true,
        };
        let mut bandit = crate::speculate::policy::BanditPolicy::new();
        bandit.push_arm(); // one arm for the initial UserNgram slot
        Self { slots: vec![base], alpha: 0.10, margin_ns, margin_forwards: 0.5, bandit }
    }

    /// Register a gated proposer. REFUSES any hidden/text-bridge slot whose
    /// offline oracle verdict is not "GO". oracle_verdict = ReplayReport::verdict().
    pub fn enable_neural_slot(
        &mut self, id: ProposerId, window: usize, min_accept_rate: f32,
        requires_hidden: bool, requires_text_bridge: bool, oracle_verdict: &str,
    ) -> crate::Result<()> {
        if (requires_hidden || requires_text_bridge) && oracle_verdict != "GO" {
            return Err(crate::Error::Model(
                "gated proposer denied: oracle verdict not GO (tau<2.5)".into()));
        }
        self.slots.push(Slot {
            id, gov: SpecGovernor::new(window, min_accept_rate), cost: CostModel::new(),
            requires_hidden, requires_text_bridge, oracle_cleared: true,
        });
        self.bandit.push_arm();
        Ok(())
    }

    /// Cost-aware draft sizing from the measured verify-cost curve. Returns the
    /// B in 2..=MAX_DRAFT_LEN maximizing payoff = min(ewma_accept_len, B) -
    /// verify_cost_forwards(B), or None if no B clears margin_forwards. B=1 is
    /// skipped — it never pays (verify_cost(1)=1.0 ≥ accept≤1), which is exactly
    /// the short-draft net-negative case that sank the free market.
    fn best_payoff_b(&self, slot: &Slot) -> Option<(usize, f32)> {
        let acc = slot.cost.ewma_accept_len;
        let mut best: Option<(usize, f32)> = None;
        for b in 2..=MAX_DRAFT_LEN {
            let payoff = acc.min(b as f32) - verify_cost_forwards(b);
            if payoff > self.margin_forwards
                && best.map_or(true, |(_, bp)| payoff > bp)
            {
                best = Some((b, payoff));
            }
        }
        best
    }

    fn plan_shape(&self, slot: &Slot, ctx: &RouterCtx) -> (usize, usize) {
        let conf = (ctx.context_confidence * slot.cost.ewma_hit_frac).clamp(0.0, 1.0);
        let len = 1 + ((MAX_DRAFT_LEN - 1) as f32 * conf).round() as usize;
        (len.clamp(1, MAX_DRAFT_LEN), 1) // tree_width=1 until Phase 6
    }

    /// Two-tier: (1) governor health gate, (2) wall-clock arbiter — max positive
    /// expected_gain - margin among healthy slots. None clears ⇒ NoSpec.
    pub fn plan(&self, ctx: &RouterCtx) -> RouterPlan {
        // Cost-aware arbitration: among healthy slots, pick the (slot, B) with the
        // largest payoff = accepted_prefix - verify_cost(B). NoSpec if none clears
        // the floor — this stops EH being net-negative on short drafts / weak
        // acceptance (the eff-TPS finding). Long-exact-span proposers
        // (suffix/SAM/retrieval) win naturally: their per-slot ewma_accept_len is
        // high when they match, so they out-payoff n-gram's low-confidence tails.
        let mut best: Option<(ProposerId, usize, f32)> = None;
        for slot in &self.slots {
            if !slot.oracle_cleared { continue; }
            if slot.requires_hidden && !ctx.hidden_available { continue; }
            if !slot.gov.is_enabled() { continue; }
            if let Some((b, payoff)) = self.best_payoff_b(slot) {
                if best.map_or(true, |(_, _, bp)| payoff > bp) {
                    best = Some((slot.id, b, payoff));
                }
            }
        }
        match best {
            Some((id, draft_len, _)) => RouterPlan::Spec { id, draft_len, tree_width: 1 },
            None => RouterPlan::NoSpec,
        }
    }

    /// Feed back the cycle that ran: update EWMA + step the slot's governor
    /// (the existing g.step(na>0) contract).
    pub fn record(&mut self, id: ProposerId, o: &StepObservation) {
        let alpha = self.alpha;
        // Find the slot index first (immutable borrow), then mutate.
        let slot_idx = self.slots.iter().position(|s| s.id == id);
        if let Some(idx) = slot_idx {
            self.slots[idx].cost.update(o, alpha);
            self.slots[idx].gov.step(o.accepted > 0);
            let reward = (o.accepted as f64 / o.drafted.max(1) as f64).clamp(0.0, 1.0);
            self.bandit.update(idx, reward);
        }
    }

    /// Preserve the "keep observing while disabled" contract (qwen_dense.rs:2609):
    /// feed a skipped slot a pessimistic false so cooldown counts down.
    pub fn observe_disabled(&mut self, id: ProposerId) {
        if let Some(slot) = self.slots.iter_mut().find(|s| s.id == id) {
            slot.gov.step(false);
        }
    }

    pub fn accept_rate(&self, id: ProposerId) -> Option<f32> {
        self.slots.iter().find(|s| s.id == id).map(|s| s.gov.accept_rate())
    }

    /// Register a model-free (oracle_cleared=true) base slot alongside UserNgram.
    /// Used by the 'ud_loop to add SuffixArray for two-proposer arbitration (P1.4).
    pub fn add_free_slot(&mut self, id: ProposerId, window: usize, min_accept_rate: f32) {
        self.slots.push(Slot {
            id,
            gov: SpecGovernor::new(window, min_accept_rate),
            cost: CostModel::new(),
            requires_hidden: false,
            requires_text_bridge: false,
            oracle_cleared: true,
        });
        self.bandit.push_arm();
    }

    /// Bandit-driven plan: UCB1 arm selection over all enabled, oracle-cleared slots.
    /// Additive alongside plan() — never replaces it in production code paths.
    pub fn plan_bandit(&self, ctx: &RouterCtx) -> RouterPlan {
        let candidates: Vec<(usize, f64)> = self.slots.iter().enumerate()
            .filter(|(_, s)| s.oracle_cleared && s.gov.is_enabled())
            .filter(|(_, s)| !s.requires_hidden || ctx.hidden_available)
            .map(|(i, s)| (i, s.cost.ewma_hit_frac as f64))
            .collect();
        let Some(slot_idx) = self.bandit.pick_ucb1(&candidates) else {
            return RouterPlan::NoSpec;
        };
        let slot = &self.slots[slot_idx];
        let (draft_len, tree_width) = self.plan_shape(slot, ctx);
        RouterPlan::Spec { id: slot.id, draft_len, tree_width }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weak_accept_no_spec() {
        // Recent accepted-prefix collapses to ~1 → no B clears the verify-cost
        // payoff floor → NoSpec. This is the cure for the net-negative short-draft
        // case: the router declines to spec when it won't pay.
        let mut r = ProposalRouter::new(16, 0.35, 1.0);
        for _ in 0..50 {
            r.record(ProposerId::UserNgram, &StepObservation {
                accepted: 1, drafted: 4, ..Default::default() });
        }
        let plan = r.plan(&RouterCtx { target_ns_per_token: 1.0, context_confidence: 0.5, hidden_available: false });
        assert_eq!(plan, RouterPlan::NoSpec);
    }

    #[test]
    fn strong_accept_specs_long() {
        // Recent accepted-prefix is long (~6) → spec with a large B that clears
        // the verify-cost curve (payoff > floor).
        let mut r = ProposalRouter::new(16, 0.35, 1.0);
        for _ in 0..50 {
            r.record(ProposerId::UserNgram, &StepObservation {
                accepted: 6, drafted: 7, ..Default::default() });
        }
        match r.plan(&RouterCtx { target_ns_per_token: 1.0, context_confidence: 0.5, hidden_available: false }) {
            RouterPlan::Spec { id: ProposerId::UserNgram, draft_len, .. } =>
                assert!(draft_len >= 5, "expected long draft, got B={draft_len}"),
            other => panic!("expected long Spec, got {other:?}"),
        }
    }

    #[test]
    fn gated_neural_slot_denied_without_go() {
        let mut r = ProposalRouter::new(16, 0.35, 1.0);
        let denied = r.enable_neural_slot(ProposerId::Eagle5, 16, 0.35, true, false, "NO-GO");
        assert!(denied.is_err(), "hidden slot must be refused without an oracle GO");
        let ok = r.enable_neural_slot(ProposerId::Eagle5, 16, 0.35, true, false, "GO");
        assert!(ok.is_ok());
    }
}

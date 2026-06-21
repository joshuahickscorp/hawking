//! Wall-clock-optimizing proposal router. Generalizes SpecGovernor from one
//! optional accept-rate gate to N per-proposer hysteresis machines under a
//! wall-clock expected_gain arbiter. Pure CPU logic — the loop feeds it measured
//! ns and it returns a plan. Losslessness is independent of the router; it only
//! chooses whether/how much to propose.

use super::governor::SpecGovernor;

pub const MAX_VERIFY_BATCH: usize = 8;
pub const MAX_DRAFT_LEN: usize = MAX_VERIFY_BATCH - 1; // 7, matches k_la cap

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProposerId {
    UserNgram,
    SuffixArray,
    Eagle5,
    Rest,
    CrossTokenizer,
    Retrieval,      // Phase 2 — REST-style retrieval proposer
    Tree,           // Phase 6 — token-tree CPU fallback
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
        Self { ewma_accept_len: 1.0, ewma_draft_ns: 0.0, ewma_verify_extra_ns: 0.0,
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
    margin_ns: f32,
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
        Self { slots: vec![base], alpha: 0.10, margin_ns, bandit }
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

    fn expected_gain_ns(&self, slot: &Slot, ctx: &RouterCtx, planned_len: usize) -> f32 {
        let c = &slot.cost;
        let e_accepted = c.ewma_accept_len.min(planned_len as f32);
        let benefit = e_accepted * ctx.target_ns_per_token;
        let cost = c.ewma_draft_ns + c.ewma_verify_extra_ns + c.ewma_retok_ns + c.ewma_sync_ns;
        benefit - cost
    }

    fn plan_shape(&self, slot: &Slot, ctx: &RouterCtx) -> (usize, usize) {
        let conf = (ctx.context_confidence * slot.cost.ewma_hit_frac).clamp(0.0, 1.0);
        let len = 1 + ((MAX_DRAFT_LEN - 1) as f32 * conf).round() as usize;
        (len.clamp(1, MAX_DRAFT_LEN), 1) // tree_width=1 until Phase 6
    }

    /// Two-tier: (1) governor health gate, (2) wall-clock arbiter — max positive
    /// expected_gain - margin among healthy slots. None clears ⇒ NoSpec.
    pub fn plan(&self, ctx: &RouterCtx) -> RouterPlan {
        let mut best: Option<(ProposerId, usize, usize, f32)> = None;
        for slot in &self.slots {
            if !slot.oracle_cleared { continue; }
            if slot.requires_hidden && !ctx.hidden_available { continue; }
            if !slot.gov.is_enabled() { continue; }
            let (draft_len, tree_width) = self.plan_shape(slot, ctx);
            let gain = self.expected_gain_ns(slot, ctx, draft_len);
            if gain <= self.margin_ns { continue; }
            let score = gain - self.margin_ns;
            if best.map_or(true, |(_, _, _, bs)| score > bs) {
                best = Some((slot.id, draft_len, tree_width, score));
            }
        }
        match best {
            Some((id, draft_len, tree_width, _)) =>
                RouterPlan::Spec { id, draft_len, tree_width },
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
    fn fast_target_auto_kills_spec() {
        // target_ns_per_token tiny ⇒ benefit < margin ⇒ NoSpec (the Qwen-3B kill).
        let mut r = ProposalRouter::new(16, 0.35, 1.0);
        // Seed a non-trivial draft cost so benefit must clear it.
        r.record(ProposerId::UserNgram, &StepObservation {
            accepted: 1, drafted: 4, draft_ns: 1000, ..Default::default() });
        let plan = r.plan(&RouterCtx { target_ns_per_token: 1.0, context_confidence: 0.5, hidden_available: false });
        assert_eq!(plan, RouterPlan::NoSpec);
    }

    #[test]
    fn slow_target_enables_spec() {
        let r = ProposalRouter::new(16, 0.35, 1.0);
        let plan = r.plan(&RouterCtx { target_ns_per_token: 1_000_000.0, context_confidence: 0.9, hidden_available: false });
        assert!(matches!(plan, RouterPlan::Spec { id: ProposerId::UserNgram, .. }));
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

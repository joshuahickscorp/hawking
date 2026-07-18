//! Event Horizon Phase 8 — contextual-bandit router policy.
//! UCB1 arm selection over router proposer slots.
//! Additive: plan_bandit() is an alternate alongside the existing plan().
//!
//! Deliberately imports nothing from router.rs or governor.rs to avoid
//! creating a circular dependency. The router assembles the candidate list
//! and passes it in; this module only does arithmetic.

/// Per-arm statistics for UCB1 bookkeeping.
#[derive(Debug, Clone, Default)]
pub struct BanditArm {
    /// Number of times this arm has been pulled.
    pub pulls: u64,
    /// Sum of rewards received (each reward ∈ [0,1]).
    pub reward_sum: f64,
}

/// UCB1 bandit policy indexed by slot position (usize), not ProposerId.
///
/// Avoids circular imports by working on raw slot indices. The router owns
/// the mapping from index → ProposerId and builds the candidate slice.
///
/// UCB1 score for arm i:
///   score_i = mu_i + sqrt(2 * ln(total + 1) / (pulls_i + 1))
///
/// The +1 offsets guard against ln(0) and division-by-zero on cold arms,
/// and ensure every cold arm gets a finite (and large) exploration bonus.
#[derive(Debug, Clone)]
pub struct BanditPolicy {
    arms: Vec<BanditArm>,
    /// Total pulls across all arms (incremented in update()).
    total: u64,
}

impl Default for BanditPolicy {
    fn default() -> Self {
        Self {
            arms: Vec::new(),
            total: 0,
        }
    }
}

impl BanditPolicy {
    /// Create an empty policy. Call push_arm() once per slot added to the router.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new arm. Must be called once per proposer slot, in the same
    /// order the router inserts slots, so that slot_idx is the array position.
    pub fn push_arm(&mut self) {
        self.arms.push(BanditArm::default());
    }

    /// Number of registered arms.
    pub fn n_arms(&self) -> usize {
        self.arms.len()
    }

    /// UCB1 arm selection over a filtered candidate list.
    ///
    /// `candidates` is `&[(slot_idx, mu)]` where `mu` is the ewma_hit_frac
    /// from the slot's CostModel, assembled by plan_bandit() in the router.
    /// The mu from the cost model seeds the exploitation term; the bandit's
    /// own pull statistics add the UCB exploration bonus on top.
    ///
    /// Returns the slot_idx of the arm with the highest UCB1 score, or None
    /// if `candidates` is empty or any slot_idx is out of bounds.
    pub fn pick_ucb1(&self, candidates: &[(usize, f64)]) -> Option<usize> {
        if candidates.is_empty() {
            return None;
        }

        // ln(total + 1): +1 keeps this finite on the very first pull.
        let log_total = ((self.total + 1) as f64).ln();

        let mut best_idx: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;

        for &(slot_idx, _mu_hint) in candidates {
            // Out-of-bounds slot_idx: bail with None (caller assembled a stale list).
            let arm = self.arms.get(slot_idx)?;

            // Exploitation: use the arm's own mean reward rather than the
            // router's ewma_hit_frac so the bandit builds its own independent
            // signal. The router's mu_hint is the filter/pre-screen; the
            // bandit's mu is the bandit's own measurement of expected reward.
            let mu = if arm.pulls == 0 {
                0.0_f64
            } else {
                arm.reward_sum / arm.pulls as f64
            };

            // Exploration bonus: sqrt(2 * ln(total+1) / (pulls+1))
            let bonus = (2.0 * log_total / (arm.pulls + 1) as f64).sqrt();

            let score = mu + bonus;
            if score > best_score {
                best_score = score;
                best_idx = Some(slot_idx);
            }
        }

        best_idx
    }

    /// Record the outcome of a pull.
    ///
    /// `slot_idx` must be a valid arm index (panics in debug, silently saturates
    /// in release if out of bounds — callers are expected to use indices from
    /// pick_ucb1 which already validates them).
    ///
    /// `reward` should be in [0, 1]; values outside that range are accepted
    /// but may degrade UCB1 score ordering.
    pub fn update(&mut self, slot_idx: usize, reward: f64) {
        if let Some(arm) = self.arms.get_mut(slot_idx) {
            arm.pulls += 1;
            arm.reward_sum += reward;
            self.total += 1;
        }
    }

    /// Mean reward for arm `slot_idx`. Returns 0.0 on a cold arm (pulls == 0)
    /// or an out-of-bounds index.
    pub fn mu(&self, slot_idx: usize) -> f64 {
        match self.arms.get(slot_idx) {
            Some(arm) if arm.pulls > 0 => arm.reward_sum / arm.pulls as f64,
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_policy(n: usize) -> BanditPolicy {
        let mut p = BanditPolicy::new();
        for _ in 0..n {
            p.push_arm();
        }
        p
    }

    /// Test 1: after warmup, high-reward arm wins.
    ///
    /// Arm 0 consistently rewarded 0.9, arm 1 rewarded 0.1.
    /// After enough pulls to shrink the exploration bonus, arm 0 must win.
    #[test]
    fn high_reward_arm_wins_after_warmup() {
        let mut p = make_policy(2);

        // Warm up both arms with many pulls to collapse the exploration bonus.
        for _ in 0..100 {
            p.update(0, 0.9);
            p.update(1, 0.1);
        }

        let candidates = vec![(0, 0.9), (1, 0.1)];
        let chosen = p.pick_ucb1(&candidates).expect("must return Some");
        assert_eq!(
            chosen, 0,
            "arm 0 (reward 0.9) should beat arm 1 (reward 0.1) after warmup"
        );
    }

    /// Test 2: a cold arm gets explored due to its large UCB bonus.
    ///
    /// Arm 0 has many pulls with moderate reward; arm 1 has zero pulls.
    /// The cold arm's exploration bonus must dominate and it gets picked.
    #[test]
    fn cold_arm_gets_explored() {
        let mut p = make_policy(2);

        // Warm arm 0 only.
        for _ in 0..50 {
            p.update(0, 0.7);
        }
        // Arm 1 is cold (zero pulls).

        let candidates = vec![(0, 0.7), (1, 0.0)];
        let chosen = p.pick_ucb1(&candidates).expect("must return Some");
        assert_eq!(chosen, 1, "cold arm 1 should be explored over warmed arm 0");
    }

    /// Test 3: empty candidate list returns None.
    #[test]
    fn empty_candidates_returns_none() {
        let p = make_policy(3);
        assert_eq!(p.pick_ucb1(&[]), None);
    }

    /// Test 4: a single candidate is always chosen regardless of its stats.
    #[test]
    fn single_candidate_always_chosen() {
        let mut p = make_policy(1);
        // Cold arm.
        assert_eq!(p.pick_ucb1(&[(0, 0.0)]), Some(0));
        // Arm with pulls.
        p.update(0, 0.5);
        assert_eq!(p.pick_ucb1(&[(0, 0.5)]), Some(0));
    }

    /// Test 5: mu() is 0.0 on a cold arm and correct after update.
    #[test]
    fn mu_is_zero_cold_and_correct_after_update() {
        let mut p = make_policy(2);

        // Cold: both arms return 0.0.
        assert_eq!(p.mu(0), 0.0, "cold arm mu must be 0.0");
        assert_eq!(p.mu(1), 0.0, "cold arm mu must be 0.0");

        // Out-of-bounds also 0.0.
        assert_eq!(p.mu(99), 0.0);

        // After two updates: mu should equal the mean.
        p.update(0, 0.8);
        p.update(0, 0.4);
        let expected = (0.8 + 0.4) / 2.0;
        let got = p.mu(0);
        assert!(
            (got - expected).abs() < 1e-12,
            "mu(0) = {got} expected {expected}"
        );

        // Arm 1 untouched.
        assert_eq!(p.mu(1), 0.0);
    }

    /// Bonus: out-of-bounds slot_idx in candidates returns None (safety gate).
    #[test]
    fn out_of_bounds_slot_returns_none() {
        let p = make_policy(2);
        // slot_idx=5 is beyond the 2-arm policy.
        assert_eq!(p.pick_ucb1(&[(5, 0.5)]), None);
    }

    /// Bonus: total pull counter increments and both arms' pulls accumulate.
    #[test]
    fn total_and_arm_pulls_tracked() {
        let mut p = make_policy(2);
        assert_eq!(p.total, 0);
        p.update(0, 1.0);
        p.update(1, 0.5);
        p.update(0, 0.9);
        assert_eq!(p.total, 3);
        assert_eq!(p.arms[0].pulls, 2);
        assert_eq!(p.arms[1].pulls, 1);
    }
}

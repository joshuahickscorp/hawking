//! RLEF — Reinforcement Learning from Execution Feedback (bible §11.7).
//!
//! The reward in RLEF comes from the **execution environment** (build/test/lint
//! oracles), not from a human or a caller. This module makes that real:
//!
//!   * [`RewardConfig`] is the §11.7.2 reward-shaping table.
//!   * [`reward_for`] maps an [`ExecutionOutcome`] → a scalar reward **derived**
//!     from what the oracles reported (the previous code summed a
//!     caller-supplied `reward` field; that is gone).
//!   * [`assemble_dataset`] turns a set of attempts on the same task into the
//!     `(context, response, reward)` tuples GRPO consumes, including the
//!     group-relative advantage (§11.7.3) — real arithmetic, not a placeholder.
//!   * [`RlefDaemon`] + [`ppl_gate`] are the documented training seam: the
//!     gradient step itself is post-shell (Hawking Condense owns it), but the
//!     reward derivation, group advantage, and PPL-gate decision are real.

use hide_core::ids::RunId;
use serde::{Deserialize, Serialize};

/// What the oracles reported for one generation attempt (§11.7.2). This is the
/// *cause* of the reward — the reward is computed from it, never supplied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionOutcome {
    /// All oracles green.
    AllGreen,
    /// The build broke.
    BuildFail,
    /// Build ok, a test failed.
    TestFail,
    /// Build + tests ok, only a lint rule failed.
    LintOnly,
    /// The attempt timed out.
    Timeout,
}

/// A coarser per-signal feedback datum the harness can emit per oracle. Kept for
/// compatibility with callers that report signals one at a time; folded into an
/// [`ExecutionOutcome`] by [`ExecutionOutcome::from_signals`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackSignal {
    BuildPassed,
    BuildFailed,
    TestPassed,
    TestFailed,
    LintFailed,
    Timeout,
}

impl ExecutionOutcome {
    /// Reduce a bag of per-oracle signals into the single worst outcome (the
    /// reward reflects the most severe failure, matching the §11.7.2 ladder:
    /// build break dominates a test fail dominates a lint-only fail).
    pub fn from_signals(signals: &[FeedbackSignal]) -> Self {
        if signals.iter().any(|s| matches!(s, FeedbackSignal::Timeout)) {
            return Self::Timeout;
        }
        if signals
            .iter()
            .any(|s| matches!(s, FeedbackSignal::BuildFailed))
        {
            return Self::BuildFail;
        }
        if signals
            .iter()
            .any(|s| matches!(s, FeedbackSignal::TestFailed))
        {
            return Self::TestFail;
        }
        if signals
            .iter()
            .any(|s| matches!(s, FeedbackSignal::LintFailed))
        {
            return Self::LintOnly;
        }
        Self::AllGreen
    }
}

/// The §11.7.2 reward-shaping table.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RewardConfig {
    pub all_green: f32,
    pub build_fail: f32,
    pub test_fail: f32,
    pub lint_only: f32,
    pub timeout: f32,
}

impl Default for RewardConfig {
    fn default() -> Self {
        // The exact shaping the bible specifies (§11.7.2).
        Self {
            all_green: 1.0,
            build_fail: -1.0,
            test_fail: -0.5,
            lint_only: -0.25,
            timeout: -0.75,
        }
    }
}

/// Derive the scalar reward for an outcome from the shaping config. This is the
/// load-bearing change: reward is a pure function of execution, not an input.
pub fn reward_for(outcome: ExecutionOutcome, config: &RewardConfig) -> f32 {
    match outcome {
        ExecutionOutcome::AllGreen => config.all_green,
        ExecutionOutcome::BuildFail => config.build_fail,
        ExecutionOutcome::TestFail => config.test_fail,
        ExecutionOutcome::LintOnly => config.lint_only,
        ExecutionOutcome::Timeout => config.timeout,
    }
}

/// One generation attempt on a task: the prompt/context that produced it, the
/// response, and the execution outcome it earned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Attempt {
    pub context: String,
    pub response: String,
    pub outcome: ExecutionOutcome,
}

/// All attempts on a single task (a GRPO "group", §11.7.3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskGroup {
    pub run_id: RunId,
    pub task_id: String,
    pub attempts: Vec<Attempt>,
}

/// A training tuple with the GRPO group-relative advantage already computed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingTuple {
    pub context: String,
    pub response: String,
    /// The raw execution reward (`reward_for`).
    pub reward: f32,
    /// `(reward - group_mean) / group_std` — the GRPO advantage (§11.7.3).
    pub advantage: f32,
}

/// Turn a task group into GRPO training tuples: derive each reward from its
/// outcome, then normalize within the group to the group-relative advantage.
pub fn assemble_group(group: &TaskGroup, config: &RewardConfig) -> Vec<TrainingTuple> {
    let rewards: Vec<f32> = group
        .attempts
        .iter()
        .map(|a| reward_for(a.outcome, config))
        .collect();
    let (mean, std) = mean_std(&rewards);
    group
        .attempts
        .iter()
        .zip(&rewards)
        .map(|(a, &r)| TrainingTuple {
            context: a.context.clone(),
            response: a.response.clone(),
            reward: r,
            // std==0 (all attempts equal) → zero advantage, no gradient signal.
            advantage: if std > f32::EPSILON {
                (r - mean) / std
            } else {
                0.0
            },
        })
        .collect()
}

/// Assemble a whole batch of groups into one flat training set.
pub fn assemble_dataset(groups: &[TaskGroup], config: &RewardConfig) -> Vec<TrainingTuple> {
    groups
        .iter()
        .flat_map(|g| assemble_group(g, config))
        .collect()
}

fn mean_std(xs: &[f32]) -> (f32, f32) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let n = xs.len() as f32;
    let mean = xs.iter().sum::<f32>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}

// ============================================================================
// Daemon seam + PPL gate
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlefConfig {
    pub tasks_per_batch: u32,
    pub attempts_per_task: u32,
    pub max_grad_steps: u32,
    /// Roll back if held-out PPL degrades by more than this many nats (§11.7.2).
    pub ppl_rollback_nats: f64,
    pub lora_rank: u32,
    pub learning_rate: f64,
    pub kl_penalty: f64,
    pub reward_shape: RewardConfig,
}

impl Default for RlefConfig {
    fn default() -> Self {
        Self {
            tasks_per_batch: 20,
            attempts_per_task: 4,
            max_grad_steps: 100,
            ppl_rollback_nats: 0.5,
            lora_rank: 16,
            learning_rate: 1e-5,
            kl_penalty: 0.02,
            reward_shape: RewardConfig::default(),
        }
    }
}

/// The §11.7.4 daemon. The gradient step is post-shell (a `GradientStepper` seam
/// the trainer fills in); everything around it — dataset assembly, the PPL gate,
/// the rollback decision — is real here.
pub struct RlefDaemon {
    pub model_role: String,
    pub config: RlefConfig,
    pub ppl_baseline: f64,
}

impl RlefDaemon {
    pub fn new(model_role: impl Into<String>, ppl_baseline: f64) -> Self {
        Self {
            model_role: model_role.into(),
            config: RlefConfig::default(),
            ppl_baseline,
        }
    }

    /// Assemble the training set for one overnight batch from the executed
    /// groups. Real work; the gradient step that consumes it is the seam below.
    pub fn prepare_batch(&self, groups: &[TaskGroup]) -> Vec<TrainingTuple> {
        assemble_dataset(groups, &self.config.reward_shape)
    }

    /// The PPL gate (§11.7.2): keep the new adapter iff held-out PPL did not
    /// degrade past the rollback threshold. `current_ppl` is measured by a
    /// forward pass (the [`PplEvaluator`] seam) — this is the *decision*, which
    /// is pure.
    pub fn ppl_gate_decision(&self, current_ppl: f64) -> GateOutcome {
        if current_ppl <= self.ppl_baseline + self.config.ppl_rollback_nats {
            GateOutcome::Keep { current_ppl }
        } else {
            GateOutcome::Rollback {
                current_ppl,
                baseline: self.ppl_baseline,
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateOutcome {
    Keep { current_ppl: f64 },
    Rollback { current_ppl: f64, baseline: f64 },
}

impl GateOutcome {
    pub fn keeps(&self) -> bool {
        matches!(self, GateOutcome::Keep { .. })
    }
}

/// Seam: the actual PPL measurement (a forward pass over held-out examples). The
/// shell ships a stub; Hawking Condense provides the real implementation against
/// a loaded adapter. Kept as a trait so the gate decision is testable without a
/// model.
pub trait PplEvaluator {
    fn held_out_ppl(&self, adapter_path: &std::path::Path) -> f64;
}

/// Convenience: evaluate PPL via the seam and apply the gate in one call.
pub fn ppl_gate(
    daemon: &RlefDaemon,
    evaluator: &dyn PplEvaluator,
    adapter_path: &std::path::Path,
) -> GateOutcome {
    daemon.ppl_gate_decision(evaluator.held_out_ppl(adapter_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group(outcomes: &[ExecutionOutcome]) -> TaskGroup {
        TaskGroup {
            run_id: RunId::new(),
            task_id: "t".into(),
            attempts: outcomes
                .iter()
                .enumerate()
                .map(|(i, &o)| Attempt {
                    context: format!("ctx{i}"),
                    response: format!("resp{i}"),
                    outcome: o,
                })
                .collect(),
        }
    }

    #[test]
    fn reward_is_derived_from_outcome() {
        let cfg = RewardConfig::default();
        assert_eq!(reward_for(ExecutionOutcome::AllGreen, &cfg), 1.0);
        assert_eq!(reward_for(ExecutionOutcome::BuildFail, &cfg), -1.0);
        assert_eq!(reward_for(ExecutionOutcome::TestFail, &cfg), -0.5);
        assert_eq!(reward_for(ExecutionOutcome::LintOnly, &cfg), -0.25);
        assert_eq!(reward_for(ExecutionOutcome::Timeout, &cfg), -0.75);
    }

    #[test]
    fn signals_fold_to_worst_outcome() {
        let s = [FeedbackSignal::BuildPassed, FeedbackSignal::TestFailed];
        assert_eq!(
            ExecutionOutcome::from_signals(&s),
            ExecutionOutcome::TestFail
        );
        let s2 = [FeedbackSignal::BuildFailed, FeedbackSignal::TestFailed];
        assert_eq!(
            ExecutionOutcome::from_signals(&s2),
            ExecutionOutcome::BuildFail
        );
        assert_eq!(
            ExecutionOutcome::from_signals(&[]),
            ExecutionOutcome::AllGreen
        );
    }

    #[test]
    fn group_advantage_is_zero_mean() {
        let g = group(&[
            ExecutionOutcome::AllGreen,
            ExecutionOutcome::BuildFail,
            ExecutionOutcome::AllGreen,
            ExecutionOutcome::TestFail,
        ]);
        let tuples = assemble_group(&g, &RewardConfig::default());
        let sum_adv: f32 = tuples.iter().map(|t| t.advantage).sum();
        assert!(sum_adv.abs() < 1e-4, "advantages should sum to ~0");
        // The all-green attempts have positive advantage; the failures negative.
        assert!(tuples[0].advantage > 0.0);
        assert!(tuples[1].advantage < 0.0);
    }

    #[test]
    fn identical_group_has_no_signal() {
        let g = group(&[ExecutionOutcome::AllGreen, ExecutionOutcome::AllGreen]);
        let tuples = assemble_group(&g, &RewardConfig::default());
        assert!(tuples.iter().all(|t| t.advantage == 0.0));
    }

    struct FixedPpl(f64);
    impl PplEvaluator for FixedPpl {
        fn held_out_ppl(&self, _: &std::path::Path) -> f64 {
            self.0
        }
    }

    #[test]
    fn ppl_gate_keeps_within_threshold_rolls_back_beyond() {
        let daemon = RlefDaemon::new("hero", 10.0);
        // +0.4 nats <= 0.5 threshold → keep.
        assert!(ppl_gate(&daemon, &FixedPpl(10.4), std::path::Path::new("a")).keeps());
        // +0.6 nats > 0.5 → rollback.
        assert!(!ppl_gate(&daemon, &FixedPpl(10.6), std::path::Path::new("a")).keeps());
    }
}

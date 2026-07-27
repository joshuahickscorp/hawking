//! Orchestration patterns & the selection rule (bible ch.09 §4.2).
//!
//! Seven patterns compose ch.02 runs. The selection rule (P5, §4.2.2) gates them
//! all; the default for an ambiguous task is a *single* run. The decisive
//! heuristic: **the presence of a deterministic oracle flips the strategy from
//! coordinate to verify-and-select** — when you can write an acceptance oracle,
//! generate many and let the oracle pick; debate is reserved for the genuinely
//! oracle-less case.
//!
//! A [`Pattern`] is the executable shape (`fan_out → runs → reduce`). The crate
//! ships the decision (`choose_pattern`) + the pattern descriptors; the
//! `FleetManager` materialises a chosen pattern into a set of jobs with the right
//! footprint/dependency wiring (e.g. tournament = N footprint-overlapping jobs
//! that race; fan-out = N footprint-disjoint jobs + a reduce job depending on
//! them).

use crate::queue::{AgentJob, ConcurrencyClass, PriorityClass};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrchestrationPattern {
    SingleAgent,
    FanOutMapReduce,
    Pipeline,
    Tournament,
    PlannerWorkersMerger,
    Debate,
    SpeculativeExploration,
}

impl OrchestrationPattern {
    /// The wire/config name (matches the A.1 `run_spec.orchestration` field).
    pub fn name(self) -> &'static str {
        match self {
            OrchestrationPattern::SingleAgent => "single",
            OrchestrationPattern::FanOutMapReduce => "map_reduce",
            OrchestrationPattern::Pipeline => "pipeline",
            OrchestrationPattern::Tournament => "tournament",
            OrchestrationPattern::PlannerWorkersMerger => "planner_workers",
            OrchestrationPattern::Debate => "debate",
            OrchestrationPattern::SpeculativeExploration => "speculative",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "single" => OrchestrationPattern::SingleAgent,
            "map_reduce" | "fanout" => OrchestrationPattern::FanOutMapReduce,
            "pipeline" => OrchestrationPattern::Pipeline,
            "tournament" => OrchestrationPattern::Tournament,
            "planner_workers" => OrchestrationPattern::PlannerWorkersMerger,
            "debate" => OrchestrationPattern::Debate,
            "speculative" => OrchestrationPattern::SpeculativeExploration,
            _ => return None,
        })
    }
}

/// The characteristics of a task that drive pattern selection (§4.2.2 inputs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TaskShape {
    /// Can we write an acceptance oracle (build+test+grep) up front?
    pub has_acceptance_oracle: bool,
    /// Does the work partition into footprint-disjoint subtasks?
    pub partitions_disjoint: bool,
    /// Clean staged handoffs (design → implement → test)?
    pub staged_handoffs: bool,
    /// One hard goal with a high first-attempt failure rate?
    pub one_hard_goal: bool,
    /// Exploratory divergent approaches worth racing?
    pub exploratory: bool,
    /// Breadth task needing isolated context windows (research/investigation)?
    pub needs_breadth_isolation: bool,
    /// Subjective synthesis with no oracle?
    pub subjective_synthesis: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatternDecision {
    pub pattern: OrchestrationPattern,
    pub reason: String,
}

/// The normative selection rule (§4.2.2). The presence of a deterministic oracle
/// flips the whole strategy.
pub fn choose_pattern(shape: TaskShape) -> PatternDecision {
    let (pattern, reason) = if !shape.has_acceptance_oracle {
        if shape.needs_breadth_isolation {
            (
                OrchestrationPattern::PlannerWorkersMerger,
                "no oracle; breadth task → isolated-context workers (Anthropic 90.2% shape)",
            )
        } else if shape.subjective_synthesis {
            (
                OrchestrationPattern::Debate,
                "no oracle; subjective synthesis → debate (documented fallback, P5)",
            )
        } else {
            (
                OrchestrationPattern::SingleAgent,
                "no oracle and not separable → single-agent (the safe default, K12)",
            )
        }
    } else if shape.partitions_disjoint {
        (
            OrchestrationPattern::FanOutMapReduce,
            "oracle + disjoint footprints → fan-out/map-reduce",
        )
    } else if shape.staged_handoffs {
        (
            OrchestrationPattern::Pipeline,
            "oracle + clean staged handoffs → pipeline",
        )
    } else if shape.one_hard_goal {
        (
            OrchestrationPattern::Tournament,
            "oracle + one hard high-failure goal → tournament/best-of-N (P4)",
        )
    } else if shape.exploratory {
        (
            OrchestrationPattern::SpeculativeExploration,
            "oracle + divergent approaches → speculative exploration (free locally, P1)",
        )
    } else {
        (
            OrchestrationPattern::SingleAgent,
            "oracle but no parallel structure → single-agent",
        )
    };
    PatternDecision {
        pattern,
        reason: reason.to_string(),
    }
}

/// Materialise a chosen pattern into a set of jobs (the executable shape). Each
/// pattern wires footprint/dependency structure so the scheduler + merge funnel
/// behave correctly:
/// - **Tournament**: `width` jobs racing the same goal (oracle selects one).
/// - **Fan-out**: `width` disjoint jobs + a `reduce` job depending on all of them.
/// - **Single**: one job.
pub fn materialise(
    pattern: OrchestrationPattern,
    objective: &str,
    width: u8,
    priority: PriorityClass,
) -> Vec<AgentJob> {
    let width = width.max(1);
    match pattern {
        OrchestrationPattern::SingleAgent | OrchestrationPattern::Pipeline => {
            vec![tagged(AgentJob::new(objective, priority), pattern)]
        }
        OrchestrationPattern::Tournament
        | OrchestrationPattern::SpeculativeExploration
        | OrchestrationPattern::Debate => (0..width)
            .map(|i| {
                tagged(
                    AgentJob::new(format!("{objective} (attempt {})", i + 1), priority),
                    pattern,
                )
            })
            .collect(),
        OrchestrationPattern::FanOutMapReduce | OrchestrationPattern::PlannerWorkersMerger => {
            let mut jobs: Vec<AgentJob> = (0..width)
                .map(|i| {
                    tagged(
                        AgentJob::new(format!("{objective} (part {})", i + 1), priority),
                        pattern,
                    )
                })
                .collect();
            let child_ids: Vec<String> = jobs.iter().map(|j| j.id.clone()).collect();
            // The reduce/merger job integrates all children and runs the full
            // suite (§4.4.1). It depends on every child.
            let reduce = tagged(
                AgentJob::new(format!("{objective} (reduce)"), priority)
                    .depends_on(child_ids)
                    .with_concurrency_class(ConcurrencyClass::Model),
                pattern,
            );
            jobs.push(reduce);
            jobs
        }
    }
}

fn tagged(mut job: AgentJob, pattern: OrchestrationPattern) -> AgentJob {
    job.spec.pattern = Some(pattern.name().to_string());
    job
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_plus_disjoint_picks_fanout() {
        let d = choose_pattern(TaskShape {
            has_acceptance_oracle: true,
            partitions_disjoint: true,
            ..TaskShape::default()
        });
        assert_eq!(d.pattern, OrchestrationPattern::FanOutMapReduce);
    }

    #[test]
    fn oracle_plus_hard_goal_picks_tournament() {
        let d = choose_pattern(TaskShape {
            has_acceptance_oracle: true,
            one_hard_goal: true,
            ..TaskShape::default()
        });
        assert_eq!(d.pattern, OrchestrationPattern::Tournament);
    }

    #[test]
    fn no_oracle_ambiguous_defaults_to_single_agent() {
        let d = choose_pattern(TaskShape::default());
        assert_eq!(d.pattern, OrchestrationPattern::SingleAgent);
    }

    #[test]
    fn no_oracle_breadth_picks_planner_workers() {
        let d = choose_pattern(TaskShape {
            needs_breadth_isolation: true,
            ..TaskShape::default()
        });
        assert_eq!(d.pattern, OrchestrationPattern::PlannerWorkersMerger);
    }

    #[test]
    fn materialise_fanout_adds_reduce_depending_on_children() {
        let jobs = materialise(
            OrchestrationPattern::FanOutMapReduce,
            "port endpoints",
            3,
            PriorityClass::Normal,
        );
        assert_eq!(jobs.len(), 4); // 3 parts + 1 reduce
        let reduce = jobs.last().unwrap();
        assert_eq!(reduce.dependencies.len(), 3);
    }

    #[test]
    fn materialise_tournament_races_n_jobs_same_goal() {
        let jobs = materialise(
            OrchestrationPattern::Tournament,
            "fix the bug",
            4,
            PriorityClass::High,
        );
        assert_eq!(jobs.len(), 4);
        assert!(jobs.iter().all(|j| j.dependencies.is_empty()));
        assert!(jobs
            .iter()
            .all(|j| j.spec.pattern.as_deref() == Some("tournament")));
    }
}

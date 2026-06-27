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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatternDecision {
    pub pattern: OrchestrationPattern,
    pub reason: String,
}

pub fn choose_pattern(
    has_acceptance_oracle: bool,
    disjoint_footprints: bool,
    needs_breadth: bool,
    subjective: bool,
) -> PatternDecision {
    let pattern = if has_acceptance_oracle && disjoint_footprints {
        OrchestrationPattern::FanOutMapReduce
    } else if has_acceptance_oracle {
        OrchestrationPattern::Tournament
    } else if needs_breadth {
        OrchestrationPattern::PlannerWorkersMerger
    } else if subjective {
        OrchestrationPattern::Debate
    } else {
        OrchestrationPattern::SingleAgent
    };
    PatternDecision {
        pattern,
        reason: format!("selected {:?}", pattern),
    }
}

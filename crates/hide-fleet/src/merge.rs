use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidatePatch {
    pub job_id: String,
    pub diff_hash: String,
    pub changed_files: Vec<String>,
    pub oracle_passed: bool,
    pub score: f32,
    pub summary: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MergeDecision {
    pub winner_job_id: Option<String>,
    pub strategy: MergeStrategy,
    pub conflicts: Vec<String>,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeStrategy {
    SelectWinner,
    ThreeWay,
    Structured,
    ManualReview,
    RejectAll,
}

pub struct TournamentSelector;

impl TournamentSelector {
    pub fn select(candidates: &[CandidatePatch]) -> MergeDecision {
        let mut passing: Vec<_> = candidates.iter().filter(|c| c.oracle_passed).collect();
        passing.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(winner) = passing.first() {
            MergeDecision {
                winner_job_id: Some(winner.job_id.clone()),
                strategy: MergeStrategy::SelectWinner,
                conflicts: Vec::new(),
                reason: "oracle-passing candidate selected by score".to_string(),
            }
        } else {
            MergeDecision {
                winner_job_id: None,
                strategy: MergeStrategy::RejectAll,
                conflicts: Vec::new(),
                reason: "no candidate passed deterministic oracles".to_string(),
            }
        }
    }
}

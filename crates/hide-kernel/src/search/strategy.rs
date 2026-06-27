use crate::verify::oracle::Verdict;
use futures::future::BoxFuture;
use hide_core::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    pub summary: String,
    pub score: f32,
    pub verdicts: Vec<Verdict>,
}

pub trait SearchStrategy: Send + Sync {
    fn name(&self) -> &str;
    fn generate<'a>(&'a self, prompt: &'a str) -> BoxFuture<'a, Result<Vec<Candidate>>>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EscalationLadder {
    pub tiers: Vec<SearchTier>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchTier {
    React,
    BestOfN,
    TreeOfThoughts,
    Lats,
    Debate,
}

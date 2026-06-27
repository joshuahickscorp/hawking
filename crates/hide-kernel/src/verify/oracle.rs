use futures::future::BoxFuture;
use hide_core::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationInput {
    pub step_id: Option<String>,
    pub workspace_root: String,
    pub changed_files: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    pub status: VerdictStatus,
    pub score: f32,
    pub oracle: String,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerdictStatus {
    Pass,
    Fail,
    Inconclusive,
    Skipped,
}

pub trait Oracle: Send + Sync {
    fn name(&self) -> &str;
    fn verify<'a>(&'a self, input: &'a VerificationInput) -> BoxFuture<'a, Result<Verdict>>;
}

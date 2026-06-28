//! Self-improving prompt modules (bible §11.2).
//!
//! # POST-SHELL MOONSHOT SEAM — the DSPy / ADAS / GEPA optimization loop is NOT implemented.
//!
//! What is REAL in this module is only the **typed data model and the promotion
//! gate**: a versioned [`PromptModule`] with an immutable [`PromptVersion`]
//! history, an [`OptimizationMetric`], a `min_eval_n` floor, and
//! [`PromptModule::promote`] — which accepts a candidate version *only* if it was
//! evaluated on enough cases (`eval_n >= min_eval_n`) **and** scores strictly
//! higher than the current best. That gate is enough to store, audit, and
//! roll back prompt versions safely.
//!
//! What is **deliberately deferred** (a documented seam, per the bible's
//! moonshot tiering) is the actual *optimizer* that produces those candidate
//! versions:
//!
//!   * **DSPy-style** compile/bootstrap of few-shot demonstrations and
//!     instruction tuning from the eval set;
//!   * **ADAS** (Automated Design of Agentic Systems) meta-search over module
//!     graphs;
//!   * **GEPA** reflective prompt evolution (LLM-in-the-loop mutate + select on
//!     the [`OptimizationMetric`]).
//!
//! Each of those requires the production eval flywheel ([`crate::eval`]) plus a
//! local-model optimization budget that only exists after the shell ships. The
//! [`PromotedBy::AutoDspy`] and [`PromotedBy::Adas`] provenance variants exist so
//! that, when the loop is built, a machine-promoted version is distinguishable
//! from a [`PromotedBy::Human`] one — but nothing in this crate emits them yet.
//! No optimizer is wired; do not read the presence of these types as a claim
//! that the self-improving loop runs.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptModule {
    pub name: String,
    pub schema_version: u32,
    pub input_schema: Value,
    pub output_schema: Value,
    pub template: String,
    pub history: Vec<PromptVersion>,
    pub metric: OptimizationMetric,
    pub min_eval_n: u16,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptVersion {
    pub version: u32,
    pub template: String,
    pub score: f64,
    pub eval_n: u16,
    pub promoted_at_ms: u64,
    pub promoted_by: PromotedBy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationMetric {
    Accuracy,
    AcceptRate,
    LatencyMs,
    OraclePassRate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromotedBy {
    Human,
    AutoDspy { optimizer_run_id: String },
    Adas { run_id: String },
}

impl PromptModule {
    pub fn promote(&mut self, version: PromptVersion) -> bool {
        if version.eval_n < self.min_eval_n {
            return false;
        }
        if self
            .history
            .last()
            .map_or(true, |current| version.score > current.score)
        {
            self.template = version.template.clone();
            self.history.push(version);
            true
        } else {
            false
        }
    }
}

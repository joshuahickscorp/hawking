pub mod deterministic;
pub mod gate;
pub mod oracle;
pub mod probabilistic;

pub use crate::verify::oracle::VerificationInput;

use crate::verify::oracle::{Cost, Oracle, OracleClass, Verdict};
use hide_core::Result;
use std::collections::BTreeMap;
use std::sync::Arc;

/// A registry of named oracles. The kernel resolves a step's
/// `acceptance.oracles` ids against this, runs the resolved set ordered
/// **deterministic-first, then cheapest-first**, and feeds the verdicts to the
/// [`gate::VerificationGate`].
#[derive(Default, Clone)]
pub struct OracleSuite {
    oracles: BTreeMap<String, Arc<dyn Oracle>>,
}

impl OracleSuite {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, oracle: Arc<dyn Oracle>) {
        self.oracles.insert(oracle.name().to_string(), oracle);
    }

    pub fn get(&self, id: &str) -> Option<Arc<dyn Oracle>> {
        self.oracles.get(id).cloned()
    }

    pub fn is_empty(&self) -> bool {
        self.oracles.is_empty()
    }

    /// Resolve the requested ids and return them ordered deterministic-first,
    /// then cheap-before-expensive (so a fast `grep_ast` fails the gate before a
    /// slow `test` ever runs). Unknown ids are skipped (the caller logs them).
    pub fn resolve_ranked(&self, ids: &[String]) -> Vec<Arc<dyn Oracle>> {
        let mut resolved: Vec<Arc<dyn Oracle>> =
            ids.iter().filter_map(|id| self.get(id)).collect();
        resolved.sort_by(|a, b| {
            let class_rank = |c: OracleClass| match c {
                OracleClass::Deterministic => 0,
                OracleClass::Probabilistic => 1,
            };
            let cost_rank = |c: Cost| c as u8;
            class_rank(a.class())
                .cmp(&class_rank(b.class()))
                .then(cost_rank(a.cost_hint()).cmp(&cost_rank(b.cost_hint())))
        });
        resolved
    }

    /// Run the ranked oracle set against `input`, short-circuiting on the first
    /// deterministic Fail (no point running an expensive test after the build
    /// already broke — §4.6.4). Returns every verdict produced.
    pub async fn run(&self, ids: &[String], input: &VerificationInput) -> Result<Vec<Verdict>> {
        let mut verdicts = Vec::new();
        for oracle in self.resolve_ranked(ids) {
            let verdict = oracle.verify(input).await?;
            let short_circuit = verdict.is_deterministic()
                && verdict.status == crate::verify::oracle::VerdictStatus::Fail;
            verdicts.push(verdict);
            if short_circuit {
                break;
            }
        }
        Ok(verdicts)
    }
}

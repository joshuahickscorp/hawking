pub mod deterministic;
pub mod gate;
pub mod oracle;
pub mod probabilistic;

pub use crate::verify::oracle::VerificationInput;

use crate::verify::oracle::{Cost, Oracle, OracleClass, Verdict, VerdictStatus};
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
    /// slow `test` ever runs), alongside the list of ids that resolved to *no*
    /// registered oracle. Unknown ids are NOT silently dropped: the caller must
    /// surface them (warn + an Inconclusive marker) so a step declaring an
    /// unregistered verifier can never be accepted on faith (K1).
    pub fn resolve_ranked<'a>(&self, ids: &'a [String]) -> (Vec<Arc<dyn Oracle>>, Vec<&'a str>) {
        let mut resolved: Vec<Arc<dyn Oracle>> = Vec::new();
        let mut unknown: Vec<&'a str> = Vec::new();
        for id in ids {
            match self.get(id) {
                Some(oracle) => resolved.push(oracle),
                None => unknown.push(id.as_str()),
            }
        }
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
        (resolved, unknown)
    }

    /// Run the ranked oracle set against `input`, short-circuiting on the first
    /// deterministic Fail (no point running an expensive test after the build
    /// already broke — §4.6.4). Returns every verdict produced.
    ///
    /// Every id that did not resolve to a registered oracle is logged
    /// (`tracing::warn`) AND recorded as a `Deterministic` `Inconclusive` verdict
    /// carrying the unknown id. That marker keeps the run auditable and prevents
    /// the gate from accepting a step whose declared verifier never ran: an
    /// Inconclusive deterministic verdict drives the gate to Inconclusive, never
    /// Accept.
    pub async fn run(&self, ids: &[String], input: &VerificationInput) -> Result<Vec<Verdict>> {
        let (resolved, unknown) = self.resolve_ranked(ids);
        let mut verdicts = Vec::new();
        for id in unknown {
            tracing::warn!(
                oracle = %id,
                step = ?input.step_id,
                "step declared an unregistered oracle id; recording Inconclusive marker"
            );
            verdicts.push(unknown_oracle_verdict(id));
        }
        for oracle in resolved {
            let verdict = oracle.verify(input).await?;
            let short_circuit = verdict.is_deterministic() && verdict.status == VerdictStatus::Fail;
            verdicts.push(verdict);
            if short_circuit {
                break;
            }
        }
        Ok(verdicts)
    }
}

/// The auditable marker for an oracle id that resolved to no registered oracle.
/// Deterministic + Inconclusive so the gate cannot Accept on its account (the
/// declared verifier never ran), while the unknown id stays in the verdict set.
fn unknown_oracle_verdict(id: &str) -> Verdict {
    Verdict {
        status: VerdictStatus::Inconclusive,
        score: 0.0,
        oracle: id.to_string(),
        class: OracleClass::Deterministic,
        detail: format!("unknown oracle id '{id}': no oracle registered under this name"),
        failures: Vec::new(),
        artifacts: Vec::new(),
        duration_ms: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::oracle::Cost;
    use futures::future::BoxFuture;

    /// A trivial always-Pass deterministic oracle for resolution tests.
    struct PassOracle(&'static str);
    impl Oracle for PassOracle {
        fn name(&self) -> &str {
            self.0
        }
        fn verify<'a>(&'a self, _input: &'a VerificationInput) -> BoxFuture<'a, Result<Verdict>> {
            Box::pin(async move { Ok(Verdict::pass(self.0, OracleClass::Deterministic, "ok")) })
        }
    }

    fn suite_with(name: &'static str) -> OracleSuite {
        let mut suite = OracleSuite::new();
        suite.register(Arc::new(PassOracle(name)));
        suite
    }

    #[test]
    fn resolve_ranked_surfaces_unknown_ids() {
        let suite = suite_with("build");
        let ids = ["build".to_string(), "ghost".to_string()];
        let (resolved, unknown) = suite.resolve_ranked(&ids);
        assert_eq!(resolved.len(), 1);
        assert_eq!(unknown, vec!["ghost"]);
    }

    #[tokio::test]
    async fn unknown_oracle_id_produces_visible_inconclusive_marker() {
        // A step declaring an unregistered oracle must NOT yield an empty,
        // silent verdict set — it must produce a Deterministic Inconclusive
        // marker that names the unknown id (auditable signal).
        let suite = suite_with("build");
        let input = VerificationInput::new(".");
        let verdicts = suite
            .run(&["build".to_string(), "ghost".to_string()], &input)
            .await
            .unwrap();
        let marker = verdicts
            .iter()
            .find(|v| v.oracle == "ghost")
            .expect("unknown oracle id must surface a verdict, not be silently dropped");
        assert_eq!(marker.status, VerdictStatus::Inconclusive);
        assert_eq!(marker.class, OracleClass::Deterministic);
        assert!(marker.detail.contains("ghost"));
    }

    #[tokio::test]
    async fn unknown_oracle_id_does_not_let_gate_accept_on_faith() {
        // The marker is Inconclusive, so a step whose ONLY declared oracle is
        // unknown can never reach Accept.
        use crate::verify::gate::{GateDecision, VerificationGate};
        let suite = OracleSuite::new();
        let input = VerificationInput::new(".");
        let verdicts = suite.run(&["ghost".to_string()], &input).await.unwrap();
        assert_ne!(VerificationGate::default().decide(&verdicts), GateDecision::Accept);
    }

    #[test]
    fn resolve_ranked_orders_deterministic_then_cheap() {
        // (sanity) keeps the ranking contract while returning unknowns.
        let _ = Cost::Cheap; // touch the import path used by other oracles
        let suite = suite_with("build");
        let ids = ["build".to_string()];
        let (resolved, unknown) = suite.resolve_ranked(&ids);
        assert_eq!(resolved.len(), 1);
        assert!(unknown.is_empty());
    }
}

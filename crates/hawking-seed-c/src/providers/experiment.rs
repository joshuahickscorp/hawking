//! **Laboratory collapse.** One experiment schema describes every study; one laboratory runner consumes it
//! **through the Seed's control interface** ([`crate::state::Machine`]) — the lab does not own a
//! controller, queue, or scheduler, and there is no Python entrypoint per old experiment. Superseded
//! campaign-specific launchers are sealed and removed.

use super::provider::Context;
use crate::evidence::receipt;
use crate::record::Record;
use crate::state::{Event, Machine};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};

/// The one experiment schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    #[serde(default = "schema")]
    pub schema: String,
    pub hypothesis: String,
    pub parent: String,
    /// Tensor / expert scope [rows, cols].
    pub scope: [usize; 2],
    pub representation: String,
    /// Exact rational target rate as "num/den".
    pub rate: String,
    pub doctor_budget_bpw: f64,
    pub evaluation: String,
    pub resources: String,
    #[serde(default)]
    pub dependencies: Vec<String>,
    /// Stopping rule (e.g. "divergence <= 0.05 or 3 families tried").
    pub stopping_rule: String,
}

fn schema() -> String {
    "hawking.packs.experiment.v1".into()
}

impl Experiment {
    /// Content identity via the Seed's one canonical-JSON + sha256 engine directly (`Record::new`).
    pub fn identity(&self) -> Result<String> {
        Ok(Record::new("experiment", serde_json::to_value(self)?).identity)
    }
}

/// The outcome of running an experiment through the Seed's controller.
#[derive(Debug, Clone, Serialize)]
pub struct ExperimentRun {
    pub experiment_identity: String,
    pub final_state: String,
    pub records: usize,
    pub all_sealed: bool,
}

/// The one laboratory runner. It drives the Seed's state machine (prepare→admit→run→evaluate→seal),
/// records the experiment declaration + the provider's evidence, and never mutates state except through
/// the Machine's public control interface. `provider_evidence` is the sealed Record a provider returned.
pub fn run_experiment(
    exp: &Experiment,
    _ctx: &Context,
    root: &std::path::Path,
    provider_evidence: Record,
) -> Result<ExperimentRun> {
    let mut m = Machine::open(root)?;
    // Drive the run purely through the Seed's control interface.
    m.apply(Event::Prepare, serde_json::json!({"experiment": exp.identity()?}))?;
    m.apply(Event::Admit, serde_json::json!({"parent": exp.parent, "scope": exp.scope}))?;
    m.apply(Event::Run, serde_json::json!({"representation": exp.representation, "rate": exp.rate}))?;
    // Record the experiment declaration as a sealed condensation receipt (the Seed's ONE evidence engine).
    m.record(receipt("condensation", serde_json::to_value(exp)?))?;
    // Record the provider's own sealed evidence (it was produced without touching state).
    provider_evidence
        .verify()
        .map_err(|e| Error::Provider(format!("provider evidence not sealed: {e}")))?;
    m.record(provider_evidence)?;
    m.apply(Event::Evaluate, serde_json::json!({"stopping_rule": exp.stopping_rule}))?;
    m.apply(Event::Seal, serde_json::json!({}))?;

    // Re-open (crash/resume) and verify every record is sealed and untampered.
    let m2 = Machine::open(root)?;
    let all_sealed = m2.log.iter().all(|r| r.verify().is_ok());
    Ok(ExperimentRun {
        experiment_identity: exp.identity()?,
        final_state: format!("{:?}", m2.state),
        records: m2.log.len(),
        all_sealed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::source_decl::SourceRecord;
    use crate::gravity::Rate;

    #[test]
    fn one_runner_drives_hawking_controller_only() {
        let root = std::env::temp_dir().join(format!("nucleus-exp-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let exp = Experiment {
            schema: schema(),
            hypothesis: "ternary latent is sub-bit and repairable".into(),
            parent: "openai/gpt-oss-120b".into(),
            scope: [256, 256],
            representation: "ternary_latent".into(),
            rate: "4/5".into(),
            doctor_budget_bpw: 0.99,
            evaluation: "output_divergence".into(),
            resources: "cpu".into(),
            dependencies: vec![],
            stopping_rule: "divergence <= 0.05 or 3 families".into(),
        };
        let ctx = Context::new("exp-run", &root.to_string_lossy(), SourceRecord::local("/tmp"), Rate::new(4, 5));
        let ev = receipt("evaluation", serde_json::json!({"divergence": 0.03}));
        let run = run_experiment(&exp, &ctx, &root, ev).unwrap();
        assert_eq!(run.final_state, "Sealed");
        assert!(run.all_sealed && run.records >= 4);
    }
}

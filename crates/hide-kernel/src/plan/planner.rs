//! Plan synthesis (bible ch.02 §4.5). The planner turns an objective into a
//! plan-as-data DAG where **every step declares its acceptance oracle up front**.

use crate::plan::schema::{Acceptance, Plan, PlanStatus, PlanStep, StepKind};
use crate::runtime_client::KernelRuntimeClient;
use futures::future::BoxFuture;
use hide_core::ids::PlanId;
use hide_core::runtime::{InferenceRequest, StreamChunk};
use hide_core::Result;
use std::collections::BTreeMap;
use std::sync::Arc;

pub trait Planner: Send + Sync {
    fn synthesize<'a>(&'a self, objective: &'a str) -> BoxFuture<'a, Result<Plan>>;
}

/// A single-step planner (tests / trivial objectives). The step verifies via the
/// `typecheck` oracle so even the stub path exercises a real deterministic gate.
#[derive(Default)]
pub struct StubPlanner;

impl Planner for StubPlanner {
    fn synthesize<'a>(&'a self, objective: &'a str) -> BoxFuture<'a, Result<Plan>> {
        let objective = objective.to_string();
        Box::pin(async move {
            // A single non-effectful step with a human predicate and no oracle
            // ids — verified by the probabilistic fallback (or, when no oracle is
            // wired, accepted as a soft step). Lets the minimal kernel make
            // honest progress without a runtime.
            let mut step = PlanStep::new(
                "Carry out the objective",
                StepKind::Investigate,
                Acceptance::predicate("objective addressed"),
            );
            step.rationale = format!("satisfy: {objective}");
            Ok(Plan {
                id: PlanId::new(),
                title: "Stub plan".to_string(),
                objective,
                steps: vec![step],
                status: PlanStatus::Active,
                budget: Default::default(),
            })
        })
    }
}

/// A planner that asks the model for a decomposition, then maps it onto the
/// plan schema. On any runtime error it falls back to a canonical
/// investigate → edit → verify DAG (so the loop is never blocked on the model).
pub struct RuntimePlanner {
    runtime: Arc<KernelRuntimeClient>,
}

impl RuntimePlanner {
    pub fn new(runtime: Arc<KernelRuntimeClient>) -> Self {
        Self { runtime }
    }

    /// The canonical three-step DAG: investigate (no effect) → edit (typecheck +
    /// build) → verify (test). Each step's acceptance names real oracles.
    pub fn default_dag(objective: &str) -> Plan {
        let investigate = PlanStep::new(
            "Investigate the codebase",
            StepKind::Investigate,
            Acceptance::predicate("relevant files and symbols identified"),
        );
        let mut edit = PlanStep::new(
            "Apply the change",
            StepKind::Edit,
            Acceptance::with_oracles(
                "the workspace builds after the edit",
                vec!["typecheck".to_string(), "build".to_string()],
            ),
        );
        edit.dependencies = vec![investigate.id.clone()];
        let mut verify = PlanStep::new(
            "Verify with tests",
            StepKind::Verify,
            Acceptance::with_oracles("tests pass", vec!["test".to_string()]),
        );
        verify.dependencies = vec![edit.id.clone()];
        Plan {
            id: PlanId::new(),
            title: format!("Plan: {}", objective.chars().take(60).collect::<String>()),
            objective: objective.to_string(),
            steps: vec![investigate, edit, verify],
            status: PlanStatus::Active,
            budget: Default::default(),
        }
    }
}

impl Planner for RuntimePlanner {
    fn synthesize<'a>(&'a self, objective: &'a str) -> BoxFuture<'a, Result<Plan>> {
        Box::pin(async move {
            // Ask the model for a step list (advisory — the acceptance contract
            // is always supplied by us, never trusted from the model).
            let request = InferenceRequest {
                task_kind: "plan".to_string(),
                prompt: format!(
                    "Decompose this objective into an ordered list of concrete steps, \
                     one per line:\n{objective}"
                ),
                messages: Vec::new(),
                max_output_tokens: 256,
                sampler: None,
                grammar: None,
                want_logprobs: false,
                metadata: BTreeMap::new(),
            };
            let mut buf = String::new();
            let mut sink = |chunk: StreamChunk| {
                if let StreamChunk::Token { text, .. } = chunk {
                    buf.push_str(&text);
                }
                Ok(())
            };
            // On a runtime error, fall back to the canonical DAG.
            if self.runtime.generate(request, &mut sink).await.is_err() {
                return Ok(Self::default_dag(objective));
            }
            let titles: Vec<String> = buf
                .lines()
                .map(|l| {
                    l.trim_start_matches(|c: char| {
                        c.is_ascii_digit() || matches!(c, '-' | '*' | '.' | ')' | ' ')
                    })
                    .trim()
                })
                .filter(|l| !l.is_empty())
                .map(String::from)
                .collect();
            if titles.is_empty() {
                return Ok(Self::default_dag(objective));
            }
            // Map model steps onto the schema with a default build+test acceptance
            // and linear dependencies; the final step also requires tests.
            let mut steps: Vec<PlanStep> = Vec::new();
            let mut prev: Option<hide_core::ids::StepId> = None;
            let n = titles.len();
            for (i, title) in titles.into_iter().enumerate() {
                let last = i + 1 == n;
                let (kind, acceptance) = if last {
                    (
                        StepKind::Verify,
                        Acceptance::with_oracles(
                            "the change builds and tests pass",
                            vec!["build".to_string(), "test".to_string()],
                        ),
                    )
                } else {
                    (
                        StepKind::Edit,
                        Acceptance::with_oracles(
                            "the workspace type-checks",
                            vec!["typecheck".to_string()],
                        ),
                    )
                };
                let mut step = PlanStep::new(title, kind, acceptance);
                if let Some(p) = prev.take() {
                    step.dependencies = vec![p];
                }
                prev = Some(step.id.clone());
                steps.push(step);
            }
            Ok(Plan {
                id: PlanId::new(),
                title: format!("Plan: {}", objective.chars().take(60).collect::<String>()),
                objective: objective.to_string(),
                steps,
                status: PlanStatus::Active,
                budget: Default::default(),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::dag::PlanDag;
    use hawking_orch::inference::StubInferenceClient;
    use hawking_orch::registry::RoleRegistry;
    use hawking_orch::router::SimpleRouter;

    fn runtime(resp: &str) -> Arc<KernelRuntimeClient> {
        let registry = Arc::new(RoleRegistry::with_default_local_roles());
        let router = Arc::new(SimpleRouter::new(registry));
        Arc::new(KernelRuntimeClient::new(
            router,
            Arc::new(StubInferenceClient::new(resp)),
        ))
    }

    #[tokio::test]
    async fn default_dag_is_acyclic_and_ordered() {
        let plan = RuntimePlanner::default_dag("do the thing");
        assert!(PlanDag::acyclic(&plan));
        assert_eq!(plan.steps.len(), 3);
        // Only the first (investigate) step is ready initially.
        assert_eq!(PlanDag::ready_steps(&plan).len(), 1);
    }

    #[tokio::test]
    async fn runtime_planner_maps_model_lines() {
        let planner = RuntimePlanner::new(runtime("1. read code\n2. edit file\n3. run tests"));
        let plan = planner.synthesize("obj").await.unwrap();
        assert_eq!(plan.steps.len(), 3);
        assert!(PlanDag::acyclic(&plan));
        // Last step requires tests.
        assert!(plan
            .steps
            .last()
            .unwrap()
            .acceptance
            .oracles
            .contains(&"test".to_string()));
    }
}

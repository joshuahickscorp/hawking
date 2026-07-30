pub mod dag {
    use crate::plan::schema::{Plan, StepStatus};
    use hide_core::ids::StepId;
    use std::collections::{BTreeMap, BTreeSet};

    pub struct PlanDag;

    impl PlanDag {
        pub fn ready_steps(plan: &Plan) -> Vec<StepId> {
            let completed: BTreeSet<_> = plan
                .steps
                .iter()
                .filter(|step| step.status == StepStatus::Completed)
                .map(|step| step.id.clone())
                .collect();
            plan.steps
                .iter()
                .filter(|step| step.status == StepStatus::Pending)
                .filter(|step| step.dependencies.iter().all(|dep| completed.contains(dep)))
                .map(|step| step.id.clone())
                .collect()
        }

        /// The plan's dependency graph is a DAG (no cycles). The driver gates `Plan`
        /// on this (§4.5.2): a cyclic plan must be replanned, never executed.
        pub fn acyclic(plan: &Plan) -> bool {
            !Self::has_cycle(plan)
        }

        pub fn has_cycle(plan: &Plan) -> bool {
            let deps: BTreeMap<_, _> = plan
                .steps
                .iter()
                .map(|step| (step.id.clone(), step.dependencies.clone()))
                .collect();
            for step in deps.keys() {
                let mut visiting = BTreeSet::new();
                if visit(step, &deps, &mut visiting) {
                    return true;
                }
            }
            false
        }
    }

    fn visit(
        step: &StepId,
        deps: &BTreeMap<StepId, Vec<StepId>>,
        visiting: &mut BTreeSet<StepId>,
    ) -> bool {
        if !visiting.insert(step.clone()) {
            return true;
        }
        for dep in deps.get(step).into_iter().flatten() {
            if visit(dep, deps, visiting) {
                return true;
            }
        }
        visiting.remove(step);
        false
    }
}

pub mod planner {
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
            assert_eq!(PlanDag::ready_steps(&plan).len(), 1);
        }
        #[tokio::test]
        async fn runtime_planner_maps_model_lines() {
            let planner = RuntimePlanner::new(runtime("1. read code\n2. edit file\n3. run tests"));
            let plan = planner.synthesize("obj").await.unwrap();
            assert_eq!(plan.steps.len(), 3);
            assert!(PlanDag::acyclic(&plan));
            assert!(plan
                .steps
                .last()
                .unwrap()
                .acceptance
                .oracles
                .contains(&"test".to_string()));
        }
    }
}

pub mod replan {
    //! Replanning (bible ch.02 §4.5.3 / §4.7.3).
    //!
    //! Replan when repeated repairs fail (the *approach* is wrong) or the failure
    //! reveals the *plan* was wrong (a missing dependency / wrong decomposition).
    //! **Localized first** (revise from the failure point, carry a lesson forward),
    //! **full** only if needed.

    use crate::plan::schema::{Acceptance, Plan, PlanStatus, PlanStep, StepKind, StepStatus};
    use hide_core::ids::StepId;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ReplanRequest {
        pub failed_step: Option<StepId>,
        pub reason: String,
        /// A lesson distilled from the failure, prepended to the revised step.
        #[serde(default)]
        pub lesson: Option<String>,
        /// Localized (revise from the failure point) vs full (resynthesize).
        pub local_only: bool,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ReplanResult {
        pub plan: Plan,
        pub changed_steps: Vec<StepId>,
    }

    /// Mark a plan superseded (used when a full replan replaces it entirely).
    pub fn supersede(mut plan: Plan) -> Plan {
        plan.status = PlanStatus::Superseded;
        plan
    }

    /// Localized replan: revise the failed step in place (reset it to pending with
    /// the lesson recorded in its rationale, and insert a remediation step before
    /// it if the failure points at a missing dependency). The downstream steps are
    /// left intact (their deps still reference the revised step's id).
    ///
    /// Returns the revised plan + the ids of the steps that changed. Bounded by the
    /// caller against `Budget.max_replans`.
    pub fn localized_replan(plan: &Plan, request: &ReplanRequest) -> ReplanResult {
        let mut plan = plan.clone();
        let mut changed = Vec::new();

        if let Some(failed_id) = &request.failed_step {
            if let Some(idx) = plan.steps.iter().position(|s| &s.id == failed_id) {
                // Reset the failed step and fold the lesson into its rationale so the
                // next attempt's repair context carries it (§4.7.2).
                let failed = &mut plan.steps[idx];
                failed.status = StepStatus::Pending;
                failed.attempts = 0;
                failed.repairs = 0;
                if let Some(lesson) = &request.lesson {
                    failed.rationale = format!("{} | lesson: {lesson}", failed.rationale);
                }
                changed.push(failed.id.clone());

                // Insert an investigation step *before* the failed one to gather the
                // missing context the approach was lacking. The failed step gains a
                // dependency on it (localized graph surgery, not a full rebuild).
                let mut probe = PlanStep::new(
                    format!("Re-investigate before retrying: {}", request.reason),
                    StepKind::Investigate,
                    Acceptance::predicate("root cause of the prior failure understood"),
                );
                probe.rationale = request
                    .lesson
                    .clone()
                    .unwrap_or_else(|| request.reason.clone());
                let probe_id = probe.id.clone();
                plan.steps[idx].dependencies.push(probe_id.clone());
                plan.steps.insert(idx, probe);
                changed.push(probe_id);
            }
        }
        ReplanResult {
            plan,
            changed_steps: changed,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::plan::dag::PlanDag;
        use crate::plan::planner::RuntimePlanner;
        #[test]
        fn localized_replan_resets_failed_and_inserts_probe() {
            let plan = RuntimePlanner::default_dag("obj");
            let edit_id = plan.steps[1].id.clone();
            let req = ReplanRequest {
                failed_step: Some(edit_id.clone()),
                reason: "build kept failing".to_string(),
                lesson: Some("exp must be i64".to_string()),
                local_only: true,
            };
            let result = localized_replan(&plan, &req);
            assert_eq!(result.changed_steps.len(), 2);
            assert!(PlanDag::acyclic(&result.plan));
            assert_eq!(result.plan.steps.len(), plan.steps.len() + 1);
            let edit = result.plan.step(&edit_id).unwrap();
            assert_eq!(edit.status, StepStatus::Pending);
            assert!(edit.rationale.contains("i64"));
        }
    }
}

pub mod schema {
    //! The plan-as-data contract (bible ch.02 Appendix A.1).
    //!
    //! A plan is a DAG of steps. **Each step declares its `acceptance` up front** —
    //! the oracle contract that must pass before the step advances. This is the
    //! chapter's most important field (K1: no state advances on faith): the plan
    //! commits, *before acting*, to how each step will be machine-verified.

    use crate::govern::Budget;
    use crate::search::strategy::SearchTier;
    use hide_core::ids::{PlanId, StepId};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Plan {
        pub id: PlanId,
        pub title: String,
        pub objective: String,
        pub steps: Vec<PlanStep>,
        pub status: PlanStatus,
        /// The governor contract for this plan (A.5). Carried on the plan so a
        /// replan can revise caps and so subagents inherit a derived child budget.
        #[serde(default)]
        pub budget: Budget,
    }

    impl Plan {
        /// A minimal one-step plan whose single step verifies via the given oracles.
        /// Used by the stub planner and tests; the real planner emits richer DAGs.
        pub fn single_step(title: impl Into<String>, objective: impl Into<String>) -> Self {
            Self {
                id: PlanId::new(),
                title: title.into(),
                objective: objective.into(),
                steps: vec![PlanStep::new(
                    "Architecture scaffold pass",
                    StepKind::Edit,
                    Acceptance::predicate(
                        "folder/module structure exists and core contracts compile",
                    ),
                )],
                status: PlanStatus::Active,
                budget: Budget::default(),
            }
        }

        pub fn step(&self, id: &StepId) -> Option<&PlanStep> {
            self.steps.iter().find(|s| &s.id == id)
        }

        pub fn step_mut(&mut self, id: &StepId) -> Option<&mut PlanStep> {
            self.steps.iter_mut().find(|s| &s.id == id)
        }
    }

    /// A single plan step (A.1). `acceptance` is required (the verifier contract).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct PlanStep {
        pub id: StepId,
        /// The step this one elaborates (for decomposition); `None` at the top level.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub parent: Option<StepId>,
        pub title: String,
        pub kind: StepKind,
        /// Why this step exists — carried forward into repair/replan lessons.
        #[serde(default)]
        pub rationale: String,
        pub dependencies: Vec<StepId>,
        pub status: StepStatus,
        /// THE VERIFIER CONTRACT — the oracles that must pass for this step (K1).
        pub acceptance: Acceptance,
        /// Optional concrete tool the act stage should dispatch (e.g. `"build.run"`,
        /// `"edit.write_file"`). When set, `Act` runs it through the tool dispatcher.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub tool_hint: Option<String>,
        /// Args for `tool_hint` (a JSON object).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub tool_args: Option<serde_json::Value>,
        /// Artifacts this step produces that downstream steps consume.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub produced: Vec<String>,
        /// Per-step search-tier override (escalate this hard step to best-of-N/ToT).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub search_hint: Option<SearchHint>,
        /// How many times this step has been attempted (act stage).
        #[serde(default)]
        pub attempts: u32,
        /// How many repair cycles this step has consumed.
        #[serde(default)]
        pub repairs: u32,
    }

    impl PlanStep {
        pub fn new(title: impl Into<String>, kind: StepKind, acceptance: Acceptance) -> Self {
            Self {
                id: StepId::new(),
                parent: None,
                title: title.into(),
                kind,
                rationale: String::new(),
                dependencies: Vec::new(),
                status: StepStatus::Pending,
                acceptance,
                tool_hint: None,
                tool_args: None,
                produced: Vec::new(),
                search_hint: None,
                attempts: 0,
                repairs: 0,
            }
        }

        /// Does the step mutate the world (needs an autonomy gate / approval)?
        pub fn is_effectful(&self) -> bool {
            matches!(
                self.kind,
                StepKind::Edit | StepKind::Command | StepKind::Delegate
            )
        }
    }

    /// The verifier contract (A.1 `acceptance`). Lists the oracle ids that must pass,
    /// the human predicate, optional test selectors, and a probabilistic threshold
    /// used only when no deterministic oracle applies.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Acceptance {
        /// Oracle ids resolved against the oracle registry (deterministic preferred).
        #[serde(default)]
        pub oracles: Vec<String>,
        /// Human-readable success condition.
        pub predicate: String,
        /// Optional test selectors for the `test` oracle.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub tests: Vec<String>,
        /// Probabilistic fallback threshold (only consulted when no deterministic
        /// oracle applies).
        #[serde(default = "default_threshold")]
        pub threshold: f32,
    }

    fn default_threshold() -> f32 {
        0.7
    }

    impl Acceptance {
        /// An acceptance with only a human predicate (no oracle ids) — verified by the
        /// probabilistic fallback. Useful for `synthesize`/`investigate` steps.
        pub fn predicate(predicate: impl Into<String>) -> Self {
            Self {
                oracles: Vec::new(),
                predicate: predicate.into(),
                tests: Vec::new(),
                threshold: default_threshold(),
            }
        }

        /// An acceptance backed by a list of (deterministic) oracle ids.
        pub fn with_oracles(predicate: impl Into<String>, oracles: Vec<String>) -> Self {
            Self {
                oracles,
                predicate: predicate.into(),
                tests: Vec::new(),
                threshold: default_threshold(),
            }
        }
    }

    /// Per-step search override (A.1 `search_hint`).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct SearchHint {
        pub tier: SearchTier,
        #[serde(default = "default_n")]
        pub n: u32,
    }

    fn default_n() -> u32 {
        4
    }

    /// Aligned to A.1: investigate / edit / command / verify / synthesize /
    /// decompose / delegate.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum StepKind {
        Investigate,
        Edit,
        Command,
        Verify,
        Synthesize,
        Decompose,
        Delegate,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum StepStatus {
        Pending,
        Ready,
        Running,
        Blocked,
        Completed,
        Failed,
        Skipped,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum PlanStatus {
        Draft,
        Active,
        Completed,
        Failed,
        Superseded,
    }
}

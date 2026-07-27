//! Goals, Plans, and Agents (Bible sec 14).
//!
//! A [`Goal`] is what the user wants. A [`Plan`] is a directed acyclic graph of
//! [`PlanStep`]s that reaches it, where each step declares -- before it runs --
//! its objective, dependencies, scope, effects, expected artifacts, acceptance
//! oracle, rollback boundary, cost, and whether it is parallelizable. An
//! [`Agent`] is a worker (root or sub-agent) with a role, a lineage, a context
//! scope, an effect allowance, a budget, and a return policy.
//!
//! Everything here is a declaration, not an execution. This crate does not run
//! plans or agents; it defines the shape a runtime fills.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::ids::{AgentId, GoalId, OracleId, PlanId, SessionId, StepId};

/// A declared side effect. A step or agent lists the effects it may cause so a
/// scope check can gate it before execution. Kept protocol-local (rather than
/// reusing another crate's effect enum) so this schema authority is
/// self-contained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    ReadFs,
    WriteFs,
    Network,
    Process,
    Shell,
    Vcs,
    Environment,
    Approval,
    AgentSpawn,
    State,
    Other,
}

impl Effect {
    pub fn as_str(&self) -> &'static str {
        match self {
            Effect::ReadFs => "read_fs",
            Effect::WriteFs => "write_fs",
            Effect::Network => "network",
            Effect::Process => "process",
            Effect::Shell => "shell",
            Effect::Vcs => "vcs",
            Effect::Environment => "environment",
            Effect::Approval => "approval",
            Effect::AgentSpawn => "agent_spawn",
            Effect::State => "state",
            Effect::Other => "other",
        }
    }
}

/// A goal: the acceptance target a plan is built to reach.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Goal {
    pub id: GoalId,
    #[serde(default)]
    pub session: Option<SessionId>,
    /// The goal in one statement.
    pub statement: String,
    /// Acceptance criteria in prose (each a checkable condition).
    #[serde(default)]
    pub acceptance: Vec<String>,
    /// The oracle that ultimately grades the goal, if one is bound.
    #[serde(default)]
    pub acceptance_oracle: Option<OracleId>,
    /// Hard constraints the plan must respect.
    #[serde(default)]
    pub constraints: Vec<String>,
    pub created_ms: u64,
}

/// The blast radius a step is allowed to touch. `paths` are the files or globs
/// it may read/write; `network` says whether it may reach out.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Scope {
    #[serde(default)]
    pub paths: Vec<String>,
    #[serde(default)]
    pub network: bool,
    #[serde(default)]
    pub description: Option<String>,
}

/// How far a step's effects can be unwound if it is rejected or fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RollbackKind {
    /// Nothing to undo.
    None,
    /// Undo by restoring a checkpoint.
    Checkpoint,
    /// Undo by reloading a state capsule.
    StateCapsule,
    /// Undo by dropping a VCS stash.
    GitStash,
}

/// The rollback boundary a step commits to before mutating anything.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RollbackBoundary {
    pub kind: RollbackKind,
    /// A reference to the boundary marker (checkpoint id, capsule id, stash
    /// ref), when the kind implies one.
    #[serde(default)]
    pub reference: Option<String>,
}

impl Default for RollbackBoundary {
    fn default() -> Self {
        Self {
            kind: RollbackKind::None,
            reference: None,
        }
    }
}

/// An estimated cost. Every field is optional so a planner can declare only
/// what it can estimate. These are estimates, not measurements -- the honest
/// framing this crate holds to.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
pub struct Cost {
    #[serde(default)]
    pub tokens: Option<u64>,
    #[serde(default)]
    pub wall_ms: Option<u64>,
    #[serde(default)]
    pub usd_micros: Option<u64>,
}

/// One node in the plan DAG. Its edges are `dependencies`; a valid plan is
/// acyclic and every dependency resolves to another step in the same plan
/// (checked by [`Plan::validate_dag`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PlanStep {
    pub id: StepId,
    pub objective: String,
    #[serde(default)]
    pub dependencies: Vec<StepId>,
    pub scope: Scope,
    #[serde(default)]
    pub effects: Vec<Effect>,
    #[serde(default)]
    pub expected_artifacts: Vec<String>,
    #[serde(default)]
    pub acceptance_oracle: Option<OracleId>,
    #[serde(default)]
    pub rollback_boundary: RollbackBoundary,
    #[serde(default)]
    pub cost: Cost,
    #[serde(default)]
    pub parallelizable: bool,
}

/// A plan: a DAG of steps that reaches a goal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Plan {
    pub id: PlanId,
    #[serde(default)]
    pub goal: Option<GoalId>,
    pub steps: Vec<PlanStep>,
    pub created_ms: u64,
}

/// Why a plan is not a well-formed DAG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DagError {
    /// A dependency names a step id that is not in the plan.
    DanglingDependency { step: String, dependency: String },
    /// The dependency graph contains a cycle (the plan is not acyclic).
    Cycle,
    /// Two steps share an id.
    DuplicateStepId(String),
}

impl std::fmt::Display for DagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DagError::DanglingDependency { step, dependency } => {
                write!(f, "step {step:?} depends on unknown step {dependency:?}")
            }
            DagError::Cycle => write!(f, "plan dependency graph contains a cycle"),
            DagError::DuplicateStepId(id) => write!(f, "duplicate step id {id:?}"),
        }
    }
}

impl std::error::Error for DagError {}

impl Plan {
    /// Check that the steps form a valid DAG: unique ids, every dependency
    /// resolves within the plan, and no cycles. Deterministic and model-free.
    pub fn validate_dag(&self) -> Result<(), DagError> {
        use std::collections::{HashMap, HashSet};

        let mut index: HashMap<&str, usize> = HashMap::new();
        for (i, step) in self.steps.iter().enumerate() {
            if index.insert(step.id.as_str(), i).is_some() {
                return Err(DagError::DuplicateStepId(step.id.as_str().to_string()));
            }
        }
        for step in &self.steps {
            for dep in &step.dependencies {
                if !index.contains_key(dep.as_str()) {
                    return Err(DagError::DanglingDependency {
                        step: step.id.as_str().to_string(),
                        dependency: dep.as_str().to_string(),
                    });
                }
            }
        }

        // Iterative DFS three-color cycle detection.
        #[derive(Clone, Copy, PartialEq)]
        enum Color {
            White,
            Gray,
            Black,
        }
        let mut color = vec![Color::White; self.steps.len()];
        for start in 0..self.steps.len() {
            if color[start] != Color::White {
                continue;
            }
            // stack of (node, whether we are entering or leaving)
            let mut stack: Vec<(usize, bool)> = vec![(start, false)];
            let mut on_path: HashSet<usize> = HashSet::new();
            while let Some((node, leaving)) = stack.pop() {
                if leaving {
                    color[node] = Color::Black;
                    on_path.remove(&node);
                    continue;
                }
                if color[node] == Color::Black {
                    continue;
                }
                color[node] = Color::Gray;
                on_path.insert(node);
                stack.push((node, true));
                for dep in &self.steps[node].dependencies {
                    let next = index[dep.as_str()];
                    if on_path.contains(&next) {
                        return Err(DagError::Cycle);
                    }
                    if color[next] == Color::White {
                        stack.push((next, false));
                    }
                }
            }
        }
        Ok(())
    }
}

/// How a sub-agent's results return to its parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReturnPolicy {
    /// Return only a summary of what happened.
    Summary,
    /// Return the full transcript of the sub-agent's turns.
    FullTranscript,
    /// Return the produced artifacts only.
    Artifacts,
    /// Return a state capsule the parent can rebind.
    StateCapsule,
}

/// The model policy an agent runs under.
///
/// DEFERRED_MODEL_REQUIRED: this only names a policy; binding a policy to an
/// actual model and honoring `allow_fallback` at inference time is the job of a
/// model-bearing runtime and is not implemented or claimed here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ModelPolicy {
    /// A named policy (for example `"default"`, `"cheap"`, `"frontier"`).
    pub policy: String,
    #[serde(default)]
    pub tier: Option<String>,
    #[serde(default)]
    pub allow_fallback: bool,
}

/// What a sub-agent is allowed to see of its parent's context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ContextScope {
    #[serde(default)]
    pub include: Vec<String>,
    #[serde(default)]
    pub exclude: Vec<String>,
    #[serde(default)]
    pub inherit_parent: bool,
}

/// The resource budget an agent may spend.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
pub struct Budget {
    #[serde(default)]
    pub tokens: Option<u64>,
    #[serde(default)]
    pub wall_ms: Option<u64>,
    #[serde(default)]
    pub tool_calls: Option<u32>,
    #[serde(default)]
    pub usd_micros: Option<u64>,
}

/// An agent: a worker in the agent tree (Bible sec 14). The root agent has no
/// parent; sub-agents record their parent and the host records their children,
/// so the tree is walkable from either end.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Agent {
    pub id: AgentId,
    pub role: String,
    #[serde(default)]
    pub parent: Option<AgentId>,
    #[serde(default)]
    pub children: Vec<AgentId>,
    pub model_policy: ModelPolicy,
    pub context_scope: ContextScope,
    #[serde(default)]
    pub effects: Vec<Effect>,
    #[serde(default)]
    pub budget: Budget,
    pub return_policy: ReturnPolicy,
}

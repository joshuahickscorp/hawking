//! Headless HIDE agent kernel (bible ch.02).
//!
//! The kernel is the deterministic brain above the model: sessions, plan-as-data,
//! budget governance, verification boundaries, and replay-safe event emission.
//! The [`AgentKernel`] owns the long-lived components (planner, oracle suite,
//! verification gate, governor, runtime client, tool dispatcher, codebase
//! grounding) and drives the FSM one transition at a time.

pub mod checkpoint;
pub mod cooperate;
pub mod govern;
pub mod machine;
pub mod plan;
pub mod projection;
pub mod runtime_client;
pub mod search;
pub mod session;
pub mod skills;
pub mod subagent;
pub mod tools;
pub mod verify;

use crate::govern::{Autonomy, Governor};
use crate::machine::driver::AgentDriver;
use crate::machine::effects::Mode;
use crate::machine::state::AgentState;
use crate::plan::planner::{Planner, RuntimePlanner, StubPlanner};
use crate::runtime_client::KernelRuntimeClient;
use crate::verify::deterministic::ProcessOracle;
use crate::verify::gate::VerificationGate;
use crate::verify::OracleSuite;
use hawking_context::{CompileInput, ContextCompiler, ContextProfile};
use hawking_index::CodeIndex;
use hide_core::event::{NewEvent, UserIntentEvent};
use hide_core::ids::{ModelId, RunId, SessionId};
use hide_core::permission::{PermissionPolicy, StaticPermissionEngine};
use hide_core::persistence::DynEventLog;
use hide_core::runtime::{ModelArchitecture, ModelDescriptor};
use hide_core::tool::{ToolDispatcher, ToolRegistry};
use hide_core::Result;
use parking_lot::Mutex;
use serde_json::json;
use std::sync::Arc;

/// Codebase grounding: the context compiler over the code index (imports the
/// `hawking-context` + `hawking-index` crates the audit flagged as
/// declared-but-unused). `compile(task)` returns the manifest hash that grounds
/// a step.
pub struct Grounding {
    index: Arc<dyn CodeIndex>,
    profile: ContextProfile,
    model: ModelDescriptor,
}

impl Grounding {
    pub fn new(index: Arc<dyn CodeIndex>) -> Self {
        Self {
            index,
            profile: ContextProfile::coding_default(8192),
            model: ModelDescriptor {
                id: ModelId::new(),
                name: "kernel-grounding".to_string(),
                architecture: ModelArchitecture::Transformer,
                context_tokens: 8192,
                tokenizer_signature: "hawking-local".to_string(),
                footprint_mb: 0,
            },
        }
    }

    /// Compile context for a task and return the manifest content hash.
    pub async fn compile(&self, task: &str) -> Result<Option<String>> {
        let mut compiler = ContextCompiler::new();
        compiler.add_source(hawking_context::sources::CodeIndexContextSource::new(
            self.index.clone(),
            8,
        ));
        let compiled = compiler
            .compile(CompileInput {
                profile: self.profile.clone(),
                model: self.model.clone(),
                task: task.to_string(),
            })
            .await?;
        // Derive a stable manifest hash from the retained span ids (provenance).
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"hide-kernel-grounding-v1\0");
        for span in &compiled.manifest.retained {
            hasher.update(span.id.as_bytes());
            hasher.update(b"\0");
        }
        Ok(Some(format!("blake3:{}", hasher.finalize().to_hex())))
    }
}

/// The agent kernel. Construct with [`AgentKernel::new`] for the
/// minimal (stub planner, no oracles) configuration that `hide-backend`
/// consumes, or with [`AgentKernel::builder`] for a fully-wired kernel.
pub struct AgentKernel {
    events: DynEventLog,
    planner: Arc<dyn Planner>,
    suite: OracleSuite,
    gate: VerificationGate,
    governor: Mutex<Governor>,
    runtime: Option<Arc<KernelRuntimeClient>>,
    dispatcher: Option<Arc<ToolDispatcher>>,
    grounding: Option<Arc<Grounding>>,
    workspace_root: String,
    mode: Mode,
}

impl AgentKernel {
    /// The minimal kernel `hide-backend` constructs: a stub planner, an empty
    /// oracle suite (the gate is then probabilistic-inconclusive, never a false
    /// Pass), and no runtime/tools. Drives the FSM through its lifecycle.
    pub fn new(events: DynEventLog) -> Self {
        Self {
            events,
            planner: Arc::new(StubPlanner),
            suite: OracleSuite::new(),
            gate: VerificationGate::default(),
            governor: Mutex::new(Governor::default()),
            runtime: None,
            dispatcher: None,
            grounding: None,
            workspace_root: ".".to_string(),
            mode: Mode::Live,
        }
    }

    pub fn builder(events: DynEventLog) -> KernelBuilder {
        KernelBuilder::new(events)
    }

    pub async fn start_run(
        &self,
        session_id: SessionId,
        objective: impl Into<String>,
    ) -> Result<AgentState> {
        let objective = objective.into();
        let run_id = RunId::new();
        self.events
            .append(
                NewEvent::user_intent(
                    session_id.clone(),
                    UserIntentEvent {
                        intent: "submit_turn".to_string(),
                        args: json!({ "objective": objective }),
                    },
                )
                .with_run(run_id.clone()),
            )
            .await?;
        Ok(AgentState::new(session_id, run_id, objective))
    }

    pub async fn step(&self, state: &mut AgentState) -> Result<()> {
        // Take a working copy of the governor so the (non-async) lock is never
        // held across the `await` (the governor's only cross-step state is its
        // autonomy + a pending interrupt, both cheap to round-trip).
        let mut governor = self.governor.lock().clone();
        let result = {
            let mut driver = AgentDriver {
                events: self.events.clone(),
                planner: self.planner.as_ref(),
                suite: &self.suite,
                gate: &self.gate,
                governor: &mut governor,
                runtime: self.runtime.as_deref(),
                dispatcher: self.dispatcher.as_deref(),
                grounding: self.grounding.as_deref(),
                workspace_root: self.workspace_root.clone(),
                mode: self.mode,
            };
            driver.step(state).await
        };
        // Write the (interrupt-consumed) governor back; preserve any interrupt the
        // host injected concurrently during the await.
        let mut live = self.governor.lock();
        live.autonomy = governor.autonomy;
        if governor.pending_interrupt.is_none() {
            // the driver consumed its interrupt; keep any newly-injected one.
        } else {
            live.pending_interrupt = governor.pending_interrupt;
        }
        result
    }

    /// Inject an interrupt (Abort/Pause/Steer) consumed on the next transition.
    pub fn interrupt(&self, interrupt: crate::govern::Interrupt) {
        self.governor.lock().interrupt(interrupt);
    }
}

/// Builder for a fully-wired kernel.
pub struct KernelBuilder {
    events: DynEventLog,
    planner: Option<Arc<dyn Planner>>,
    suite: OracleSuite,
    gate: VerificationGate,
    autonomy: Autonomy,
    runtime: Option<Arc<KernelRuntimeClient>>,
    dispatcher: Option<Arc<ToolDispatcher>>,
    grounding: Option<Arc<Grounding>>,
    workspace_root: String,
    mode: Mode,
}

impl KernelBuilder {
    pub fn new(events: DynEventLog) -> Self {
        Self {
            events,
            planner: None,
            suite: OracleSuite::new(),
            gate: VerificationGate::default(),
            autonomy: Autonomy::FullAuto,
            runtime: None,
            dispatcher: None,
            grounding: None,
            workspace_root: ".".to_string(),
            mode: Mode::Live,
        }
    }

    pub fn workspace_root(mut self, root: impl Into<String>) -> Self {
        self.workspace_root = root.into();
        self
    }

    pub fn autonomy(mut self, autonomy: Autonomy) -> Self {
        self.autonomy = autonomy;
        self
    }

    pub fn mode(mut self, mode: Mode) -> Self {
        self.mode = mode;
        self
    }

    pub fn planner(mut self, planner: Arc<dyn Planner>) -> Self {
        self.planner = Some(planner);
        self
    }

    pub fn gate(mut self, gate: VerificationGate) -> Self {
        self.gate = gate;
        self
    }

    pub fn oracle_suite(mut self, suite: OracleSuite) -> Self {
        self.suite = suite;
        self
    }

    pub fn grounding(mut self, grounding: Arc<Grounding>) -> Self {
        self.grounding = Some(grounding);
        self
    }

    /// Wire the runtime client (model). Also installs a [`RuntimePlanner`] if no
    /// planner has been set yet.
    pub fn runtime(mut self, runtime: Arc<KernelRuntimeClient>) -> Self {
        if self.planner.is_none() {
            self.planner = Some(Arc::new(RuntimePlanner::new(runtime.clone())));
        }
        self.runtime = Some(runtime);
        self
    }

    /// Wire a tool dispatcher (effectful steps + shelling oracles).
    pub fn dispatcher(mut self, dispatcher: Arc<ToolDispatcher>) -> Self {
        self.dispatcher = Some(dispatcher);
        self
    }

    /// Convenience: register the standard deterministic process oracles
    /// (build/typecheck/test/lint) against the given dispatcher.
    pub fn with_standard_oracles(mut self, dispatcher: Arc<ToolDispatcher>) -> Self {
        self.suite
            .register(Arc::new(ProcessOracle::build(dispatcher.clone())));
        self.suite
            .register(Arc::new(ProcessOracle::typecheck(dispatcher.clone())));
        self.suite
            .register(Arc::new(ProcessOracle::test(dispatcher.clone())));
        self.suite
            .register(Arc::new(ProcessOracle::lint(dispatcher.clone())));
        if self.dispatcher.is_none() {
            self.dispatcher = Some(dispatcher);
        }
        self
    }

    pub fn build(self) -> AgentKernel {
        AgentKernel {
            events: self.events,
            planner: self.planner.unwrap_or_else(|| Arc::new(StubPlanner)),
            suite: self.suite,
            gate: self.gate,
            governor: Mutex::new(Governor::new(self.autonomy)),
            runtime: self.runtime,
            dispatcher: self.dispatcher,
            grounding: self.grounding,
            workspace_root: self.workspace_root,
            mode: self.mode,
        }
    }
}

/// Build a permission-allow-all tool dispatcher over the builtin catalog rooted
/// at `workspace_root`. Used by tests and simple hosts; production wires the
/// real permission engine + sandbox config.
pub fn allow_all_dispatcher(workspace_root: impl Into<String>) -> Arc<ToolDispatcher> {
    let registry = Arc::new(ToolRegistry::default());
    hide_tools::register_builtin_tools_with(
        &registry,
        hide_tools::ShellConfig {
            workspace_root: Some(workspace_root.into()),
            disable_sandbox: true,
            ..Default::default()
        },
    );
    Arc::new(ToolDispatcher::new(
        registry,
        Arc::new(StaticPermissionEngine::new(PermissionPolicy {
            default_decision: hide_core::types::Decision::Allow,
            rules: Vec::new(),
            risk_gates: Vec::new(),
        })),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::event::InMemoryEventLog;

    #[tokio::test]
    async fn kernel_can_drive_minimal_run_to_done() {
        let log = Arc::new(InMemoryEventLog::new());
        let kernel = AgentKernel::new(log.clone());
        let mut state = kernel
            .start_run(SessionId::new(), "scaffold the thing")
            .await
            .unwrap();
        for _ in 0..40 {
            if state.phase.is_terminal() {
                break;
            }
            kernel.step(&mut state).await.unwrap();
        }
        assert!(state.phase.is_terminal());
        assert!(log.len() >= 5);
    }
}

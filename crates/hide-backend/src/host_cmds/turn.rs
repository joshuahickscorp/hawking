use crate::approval::{ApprovalDecision, ApprovalHub};
use crate::commands::CommandRouter;
use crate::connectors::{register_backend_connectors, ConnectorRegistry, ConnectorStatus};
use crate::initialize::{ClientCapabilities, ClientInfo, ConnectionRegistry, InitializeResponse};
use crate::interrupt::InterruptHub;
use crate::live_thread::LiveThread;
use crate::memory::{
    MemoryDraft, MemoryLedger, MemoryRecord, MemoryRevalidation, MemoryScope, MemoryStatus,
    PrivacyClass, RevalidateTarget,
};
use crate::policy::{
    derive_policy_decision, tool_declared_effects, PolicyDecision, PolicyDecisionRecord,
};
use crate::process::{ProcessState, ProcessSupervisor, StartSpec};
use crate::replay::BackendReplayService;
use crate::rewind::{self, CheckpointCoverage, FileChange, ForkPoint, RewindTarget, StateRef};
use crate::security::SecurityServices;
use crate::services::{
    BackendCapabilities, BackendServices, Budget, CheckpointRecord, CheckpointStore,
    EnvironmentNode, EnvironmentSwitch, GoalOutcome, GoalRecord, GoalStatus, GoalStore,
    GoalVerdict, JobRecord, JobStatus, JobStore, RepoNode, SharedBackend, Trigger, TriggerEvent,
    TrustState, WorkspaceEdge, WorkspaceEdgeKind, WorkspaceGraph, WorkspaceStore,
};
use crate::supervisor::{RuntimeSupervisor, SupervisorConfig};
use crate::surfaces::SurfaceGraphService;
use crate::tools::{build_default_tool_dispatcher, build_default_tool_registry};
use crate::ui_bus::UiEventBus;
use hide_core::api::{Intent, IntentAck, UiEvent, UiEventKind};
use hide_core::event::{Event, NewEvent, ToolCallEvent, ToolResultEvent};
use hide_core::ids::{EventId, RunId, SessionId, StepId};
use hide_core::observability::{HealthCheck, HealthReport, HealthStatus};
use hide_core::runtime::{ModelRole, RuntimeSupervisorState};
use hide_core::tool::{ToolCall, ToolDispatcher, ToolRegistry, ToolResult, ToolSpec, ToolStatus};
use hide_core::Result;
use hide_fleet::manager::KernelRunLauncher;
use hide_fleet::{
    AgentJob, ConcurrencyClass, FleetConfig, FleetGovernor, FleetManager, OsResourceProbe,
    PriorityClass,
};
use hide_kernel::govern::{Autonomy, Interrupt};
use hide_kernel::machine::state::{AgentState, ApprovalRequest, Phase};
use hide_kernel::session::SessionProjection;
use hide_kernel::{AgentKernel, Grounding};
// Bible Book IX sec 28-29 / sec 78.1 #6: the deterministic verification plane.
// The colliding names (`Verdict`, `VerificationInput`, `Oracle`) are qualified
// as `hide_kernel::verify_plane::*` at their (few) use sites so the function-local
// `hide_kernel::verify::oracle::*` imports in the goal path and the tests keep
// their meaning; only the non-colliding types are imported here.
use super::*;
use hide_kernel::verify_plane::{
    Finding, GateDecision, ReviewRole, ReviewRoleProfile, SourceFile, StaticAnalysisOracle,
    TieredVerdict, VerificationReceipt, VerificationTier,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

impl BackendHost {
    pub async fn steer_run(
        &self,
        run_id: RunId,
        instruction: impl Into<String>,
        session: Option<SessionId>,
    ) -> Result<Event> {
        let instruction = instruction.into();
        // 1. Signal the running kernel (same hub Cancel/Pause/Resume ride).
        self.interrupts.signal(
            run_id.clone(),
            Interrupt::Steer {
                instruction: instruction.clone(),
            },
        );
        // 2. Durable steer event (audit + projection), tagged with the run.
        let session = session.unwrap_or_else(|| self.commands.control_session().clone());
        let event = self
            .services
            .event_log
            .append(
                NewEvent::system(
                    session.clone(),
                    "turn.steer",
                    json!({ "run_id": run_id.as_str(), "instruction": instruction }),
                )
                .with_run(run_id.clone()),
            )
            .await?;
        // 3. Surface it on Wire-B so the transcript shows the redirect.
        self.ui_bus.publish(UiEvent {
            seq: event.seq,
            session_id: Some(session),
            kind: UiEventKind::Custom(json!({
                "kind": "turn_steer",
                "run_id": run_id.as_str(),
                "instruction": instruction,
            })),
        });
        Ok(event)
    }

    /// Dispatch a durable Memory / Goal-eval / Workspace-trust / Environment-switch
    /// custom intent to the corresponding tested host method (bible sec 21-22, 14,
    /// 35). These built methods were unreachable from the typed FE because
    /// `handle_intent` had no custom-name arm for them. Payload shapes:
    ///
    /// * `memory_add`             -> a MemoryDraft: `{ scope: {kind,id}, claim,
    ///   source, author, confidence?, citations?, invalidation?, privacy?,
    ///   expiry_ms? }`
    /// * `memory_supersede`       -> `{ old_id, replacement: <MemoryDraft> }`
    /// * `memory_record_outcome`  -> `{ memory_id, success: bool }`
    /// * `memory_revalidate`      -> `{ memory_id | scope: {kind,id}, repo_root? }`
    /// * `goal_evaluate`          -> `{ session_id }`
    /// * `workspace_set_repo_trust` -> `{ repo_id, trust: "trusted"|"untrusted" }`
    /// * `environment_switch`     -> `{ session_id, env_id, reason? }`
    ///
    /// Each arm routes to the existing method (never re-implements its logic) and
    /// surfaces the domain change on Wire-B; `environment_switch`/`goal_evaluate`
    /// already emit their own durable events, so those are not double-recorded.
    pub fn build_turn_kernel(
        &self,
        base_url: String,
        session_id: SessionId,
        run_id: RunId,
    ) -> AgentKernel {
        use crate::model_provider::{HttpModelProvider, ModelProviderInferenceClient};
        use hawking_orch::inference::InferenceClient;
        use hawking_orch::router::SimpleRouter;
        use hide_kernel::runtime_client::KernelRuntimeClient;

        let inference: Arc<dyn InferenceClient> = Arc::new(ModelProviderInferenceClient::new(
            HttpModelProvider::new(base_url),
        ));
        let runtime = Arc::new(KernelRuntimeClient::new(
            Arc::new(SimpleRouter::new(self.services.role_registry.clone())),
            inference,
        ));

        let dispatcher = self.build_turn_dispatcher(session_id, Some(run_id));
        let grounding = Arc::new(Grounding::new(self.services.code_index.clone()));

        AgentKernel::builder(self.services.event_log.clone())
            .workspace_root(
                self.services
                    .config
                    .workspace_root
                    .to_string_lossy()
                    .to_string(),
            )
            .autonomy(turn_kernel_autonomy())
            .grounding(grounding)
            // `.runtime(..)` installs a `RuntimePlanner` since no planner is set.
            .runtime(runtime)
            .dispatcher(dispatcher.clone())
            .with_standard_oracles(dispatcher)
            .build()
    }

    /// The dispatcher a turn's tools go through: the REAL permission engine (config-driven, NOT
    /// `allow_all_dispatcher`), with the SAME [`DispatchRecorder`] the host's own dispatcher
    /// carries, bound to this turn's session and run.
    ///
    /// The kernel holds this object directly, so binding the attribution HERE is what makes an
    /// agent edit produce a `tool.call`/`tool.result` pair and an addressable diff hunk. It is
    /// bound rather than ambient because a task-local would not survive the kernel spawning a task.
    pub fn build_turn_dispatcher(
        &self,
        session_id: SessionId,
        run_id: Option<RunId>,
    ) -> Arc<ToolDispatcher> {
        let bound = crate::tools::DispatchContext::unverified_model(session_id, run_id);
        Arc::new(
            crate::tools::build_task_tool_dispatcher(
                &self.services.config,
                self.tools.clone(),
                Some(bound.clone()),
            )
            .with_observer(Arc::new(DispatchRecorder::bound_to(
                self.services.clone(),
                self.ui_bus.clone(),
                bound,
            ))),
        )
    }

    /// Spawn the generation for an accepted `SubmitTurn`: route it at the live
    /// runtime and stream tokens onto Wire-B. The run's `run_id` is registered
    /// so `CancelRun`/`PauseRun` reach it via the shared `InterruptHub`. When no
    /// runtime is online (no model configured, or it is not yet `Ready`), this
    /// publishes a `RuntimeStatus`/`Error` UiEvent instead of generating, so the
    /// FE shows "model offline", never a fake token.
    pub fn build_fleet_kernel(&self, session_id: SessionId) -> AgentKernel {
        use crate::model_provider::{HttpModelProvider, ModelProviderInferenceClient};
        use hawking_orch::inference::{InferenceClient, StubInferenceClient};
        use hawking_orch::router::SimpleRouter;
        use hide_kernel::runtime_client::KernelRuntimeClient;

        let inference: Arc<dyn InferenceClient> = if let Some(url) = self.runtime_base_url() {
            Arc::new(ModelProviderInferenceClient::new(HttpModelProvider::new(
                url,
            )))
        } else {
            // Offline: RuntimePlanner.synthesize falls through to default_dag
            // on empty/failed generation — still a real plan, not StubPlanner.
            Arc::new(StubInferenceClient::new(""))
        };
        let runtime = Arc::new(KernelRuntimeClient::new(
            Arc::new(SimpleRouter::new(self.services.role_registry.clone())),
            inference,
        ));
        let dispatcher = self.build_turn_dispatcher(session_id, None);
        AgentKernel::builder(self.services.event_log.clone())
            .workspace_root(
                self.services
                    .config
                    .workspace_root
                    .to_string_lossy()
                    .to_string(),
            )
            // Fleet has no interactive approver on this path: FullAuto so the
            // RuntimePlanner DAG can progress. Oracles still gate correctness.
            .autonomy(Autonomy::FullAuto)
            // `.runtime(..)` installs RuntimePlanner when no planner is set.
            .runtime(runtime)
            .dispatcher(dispatcher.clone())
            .with_standard_oracles(dispatcher)
            .build()
    }

    /// Generate against a (supervised) runtime through the kernel's runtime-client
    /// seam and publish the completion onto the push Wire-B bus.
    ///
    /// This is the host's end-to-end generation path: a `KernelRuntimeClient`
    /// (router + the host's HTTP `ModelProvider`, adapted to the orch
    /// `InferenceClient` seam) produces tokens; each token batch is published -
    /// with coalescing - onto the broadcast bus, then flushed at stream end. The
    /// returned string is the full completion (for callers that also want it
    /// inline). `base_url` is the supervised serve's base (from the
    /// `RuntimeSupervisor`).
    pub async fn generate_and_publish(
        &self,
        session_id: SessionId,
        base_url: impl Into<String>,
        prompt: impl Into<String>,
    ) -> Result<String> {
        use crate::model_provider::{HttpModelProvider, ModelProviderInferenceClient};

        let provider = HttpModelProvider::new(base_url);
        let inference: Arc<dyn hawking_orch::inference::InferenceClient> =
            Arc::new(ModelProviderInferenceClient::new(provider));
        // Both generation entry points funnel through `run_turn_core` so the live
        // path and this one build the SAME real request (compiled context + real
        // history + a derived budget) and can never drift. This twin skips the
        // per-step / post-turn live-manifest telemetry (no run/interrupt wiring).
        let outcome = run_turn_core(
            inference,
            self.services.event_log.clone(),
            self.services.role_registry.clone(),
            self.services.code_index.clone(),
            self.services.memory_store.clone(),
            self.services.classed_memory.clone(),
            self.ui_bus.clone(),
            session_id,
            prompt.into(),
            None,
            None,
            self.services.repo_instructions.clone(),
        )
        .await?;
        Ok(outcome.completion)
    }
}

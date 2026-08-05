use crate::approval::{ApprovalDecision, ApprovalHub};
use crate::commands::CommandRouter;
use crate::connectors::{register_backend_connectors, ConnectorRegistry, ConnectorStatus};
use crate::hcli_sources::HcliSourceContext;
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

/// A durable, explicit-runtime model turn for HCLI and other headless callers.
///
/// Unlike the historical [`BackendHost::generate_and_publish`] convenience
/// method, this result proves that the user prompt was first recorded in the
/// session log. That makes subsequent calls to the same named session a real
/// contextual conversation rather than a completion-only sequence.
#[derive(Debug, Clone, Serialize)]
pub struct HcliTurnResult {
    pub session_id: SessionId,
    /// The exact durable user-intent event written before inference began.
    pub intent_event_id: EventId,
    pub intent_event_seq: u64,
    /// The target-verified assistant history event written after successful
    /// inference. This is absent only when generation itself errors, in which
    /// case this method returns an error instead of a partial success result.
    pub assistant_event_id: EventId,
    /// The durable runtime-generation event sequence, suitable for correlating
    /// Wire-B token streaming with the terminal HCLI response.
    pub stream_id: String,
    pub completion: String,
    /// Raw metrics from the model runtime. Optional fields mean the runtime did
    /// not expose them; callers must not derive a complete-forward TPS without
    /// both `decode_ms` and `completed_decode_forwards`.
    pub generation_stats: hide_core::runtime::GenerationStats,
    pub complete_forward_tps: Option<f64>,
    /// Metadata-only receipt for an explicit local evidence selection. This is
    /// present only when the caller supplied a source pack; the derivative text
    /// itself is never echoed in the terminal result.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_context: Option<Value>,
}

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
        self.build_kernel_for_runtime(base_url, session_id, Some(run_id), None, false)
    }

    /// Build the full agent kernel for a non-interactive headless run.
    ///
    /// The agent mints its own `RunId` in `AgentKernel::start_run`, so this
    /// path deliberately leaves the dispatch attribution unbound at construction
    /// time. The verified model-tool executor then receives and verifies the
    /// actual generated run id on each call. Binding a guessed id here would make
    /// the audit trail internally inconsistent.
    pub fn build_headless_kernel(
        &self,
        base_url: String,
        session_id: SessionId,
        runtime_output_cap: Option<usize>,
        compact_model_prompts: bool,
    ) -> AgentKernel {
        self.build_kernel_for_runtime(
            base_url,
            session_id,
            None,
            runtime_output_cap,
            compact_model_prompts,
        )
    }

    fn build_kernel_for_runtime(
        &self,
        base_url: String,
        session_id: SessionId,
        run_id: Option<RunId>,
        runtime_output_cap: Option<usize>,
        compact_model_prompts: bool,
    ) -> AgentKernel {
        use crate::model_provider::{HttpModelProvider, ModelProviderInferenceClient};
        use hawking_orch::inference::InferenceClient;
        use hawking_orch::router::SimpleRouter;
        use hide_kernel::runtime_client::KernelRuntimeClient;

        let provider = HttpModelProvider::new(base_url);
        let inference: Arc<dyn InferenceClient> = match runtime_output_cap.filter(|cap| *cap > 0) {
            Some(cap) => Arc::new(ModelProviderInferenceClient::with_max_output_tokens(
                provider, cap,
            )),
            None => Arc::new(ModelProviderInferenceClient::new(provider)),
        };
        let runtime = Arc::new(KernelRuntimeClient::new(
            Arc::new(SimpleRouter::new(self.services.role_registry.clone())),
            inference,
        ));

        let dispatcher = self.build_turn_dispatcher(session_id, run_id);
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
            .compact_model_prompts(compact_model_prompts)
            // `.runtime(..)` installs a `RuntimePlanner` since no planner is set.
            .runtime(runtime)
            .dispatcher(dispatcher.clone())
            .verified_model_tool_executor(self.verified_model_tool_executor())
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

    /// The only kernel-facing authority allowed to execute a parsed model tool
    /// call. It uses the host's shared dispatcher (not the per-turn unverified
    /// one) so the executor can establish the durable, action-bound
    /// target-verification context before dispatch.
    pub fn verified_model_tool_executor(
        &self,
    ) -> Arc<dyn hide_kernel::tools::VerifiedModelToolExecutor> {
        Arc::new(crate::tools::DirectTargetModelToolExecutor::new(
            self.services.clone(),
            self.dispatcher.clone(),
        ))
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
            .verified_model_tool_executor(self.verified_model_tool_executor())
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
        let outcome = self
            .generate_and_publish_outcome(
                session_id,
                base_url.into(),
                prompt.into(),
                None,
                None,
                None,
            )
            .await?;
        Ok(outcome.completion)
    }

    /// Run one externally selected local model endpoint as a durable HCLI
    /// conversation turn. The explicit URL avoids coupling a CLI client to the
    /// optional `HIDE_MODEL_WEIGHTS` supervisor, while the recorded
    /// `SubmitTurn` preserves the full user/assistant history for the next
    /// call. This method deliberately does *not* call `handle_intent`, because
    /// that method would also start the supervisor-owned generation path.
    pub async fn hcli_turn(
        &self,
        session_id: SessionId,
        base_url: impl Into<String>,
        prompt: impl Into<String>,
    ) -> Result<HcliTurnResult> {
        self.hcli_turn_with_output_cap(session_id, base_url, prompt, None)
            .await
    }

    /// As [`Self::hcli_turn`], with an explicit caller-requested output cap.
    /// The cap only narrows the model window derived by `run_turn_core`; it can
    /// never make a packed context claim a larger usable window.
    pub async fn hcli_turn_with_output_cap(
        &self,
        session_id: SessionId,
        base_url: impl Into<String>,
        prompt: impl Into<String>,
        requested_output_cap: Option<usize>,
    ) -> Result<HcliTurnResult> {
        self.hcli_turn_with_output_cap_and_source_context(
            session_id,
            base_url,
            prompt,
            requested_output_cap,
            None,
        )
        .await
    }

    /// As [`Self::hcli_turn_with_output_cap`], with a single explicit,
    /// bounded pack of local object-store derivatives. The pack is injected
    /// into this invocation's real native prompt as untrusted reference
    /// material; it is not appended to durable user history or implicitly
    /// carried to a later turn.
    pub async fn hcli_turn_with_output_cap_and_source_context(
        &self,
        session_id: SessionId,
        base_url: impl Into<String>,
        prompt: impl Into<String>,
        requested_output_cap: Option<usize>,
        source_context: Option<HcliSourceContext>,
    ) -> Result<HcliTurnResult> {
        let prompt = prompt.into();
        let base_url = base_url.into();
        // An explicitly selected HCLI endpoint may have a much smaller native
        // window than the product-default coding role.  Consult the endpoint
        // before compiling durable context so the packer does not knowingly
        // send an over-window prompt and rely on the runtime to truncate it.
        // `None` remains an honest fallback for legacy/unreachable endpoints.
        let live_ceiling = crate::model_provider::HttpModelProvider::new(base_url.clone())
            .get_context_info()
            .await
            .and_then(|info| {
                let ceiling = info.ctx_len_effective.or(info.ctx_len_native)?;
                Some((
                    info.recurrent_state_bytes,
                    info.ctx_len_native,
                    ceiling,
                    info.max_output_tokens,
                ))
            });
        let ack = self
            .commands
            .handle(Intent::SubmitTurn {
                session_id: session_id.clone(),
                text: prompt.clone(),
                attachments: Vec::new(),
            })
            .await?;
        if !ack.accepted {
            return Err(hide_core::error::HideError::PolicyDenied(
                ack.message
                    .unwrap_or_else(|| "HCLI turn was refused before logging".to_string()),
            ));
        }
        let intent_event_seq = ack.event_seq.ok_or_else(|| {
            hide_core::error::HideError::PolicyDenied(
                "accepted HCLI turn was missing its durable intent event".to_string(),
            )
        })?;
        let intent_event_id = self
            .services
            .event_log
            .scan(Some(session_id.clone()), None, None)
            .await?
            .into_iter()
            .find(|event| event.seq == intent_event_seq)
            .map(|event| event.id)
            .ok_or_else(|| {
                hide_core::error::HideError::PolicyDenied(
                    "accepted HCLI turn intent event could not be recovered from the durable log"
                        .to_string(),
                )
            })?;
        let outcome = self
            .generate_and_publish_outcome(
                session_id.clone(),
                base_url,
                prompt,
                requested_output_cap,
                source_context.as_ref(),
                live_ceiling,
            )
            .await?;
        let complete_forward_tps = match (
            outcome.generation_stats.decode_ms,
            outcome.generation_stats.completed_decode_forwards,
        ) {
            (Some(decode_ms), Some(forwards)) if decode_ms > 0.0 && forwards > 0 => {
                Some(forwards as f64 / (decode_ms / 1_000.0))
            }
            _ => None,
        };
        let source_context_disposition = outcome.source_context_disposition;
        Ok(HcliTurnResult {
            session_id,
            intent_event_id,
            intent_event_seq,
            assistant_event_id: outcome.assistant_event_id,
            stream_id: outcome.stream_seq.to_string(),
            completion: outcome.completion,
            generation_stats: outcome.generation_stats,
            complete_forward_tps,
            source_context: source_context.as_ref().map(|context| {
                let mut receipt = context.receipt_json();
                let (target, model_prompt_omitted, note) = match source_context_disposition {
                    SourceContextDisposition::Injected => (
                        "durable_native_turn_prompt",
                        false,
                        "Only this turn receives the selected model-facing derivatives. The durable context.compiled event stores the same metadata-only selection receipt.",
                    ),
                    SourceContextDisposition::OmittedWholeBlockForLiveWindow => (
                        "none",
                        true,
                        "The complete selected evidence block plus the reconstructed native prompt and response reserve did not fit the observed compact context, so it was omitted rather than truncated. Its selection metadata is preserved in the durable context.compiled event.",
                    ),
                    SourceContextDisposition::NotRequested => (
                        "none",
                        true,
                        "No non-empty selected evidence block was available for this turn.",
                    ),
                };
                receipt["injection"] = json!({
                    "status": source_context_disposition.as_str(),
                    "target": target,
                    "model_prompt_omitted": model_prompt_omitted,
                    "persisted_as_user_history": false,
                    "note": note,
                });
                receipt
            }),
        })
    }

    async fn generate_and_publish_outcome(
        &self,
        session_id: SessionId,
        base_url: String,
        prompt: String,
        requested_output_cap: Option<usize>,
        source_context: Option<&HcliSourceContext>,
        live_ceiling: Option<(Option<usize>, Option<usize>, usize, Option<usize>)>,
    ) -> Result<TurnOutcome> {
        use crate::model_provider::{HttpModelProvider, ModelProviderInferenceClient};

        let provider = HttpModelProvider::new(base_url);
        let inference: Arc<dyn hawking_orch::inference::InferenceClient> =
            Arc::new(ModelProviderInferenceClient::new(provider));
        // Both generation entry points funnel through `run_turn_core` so the live
        // path and this one build the SAME real request (compiled context + real
        // history + a derived budget) and can never drift. This twin skips the
        // per-step / post-turn live-manifest telemetry (no run/interrupt wiring).
        run_turn_core(
            inference,
            self.services.event_log.clone(),
            self.services.role_registry.clone(),
            self.services.code_index.clone(),
            self.services.memory_store.clone(),
            self.services.classed_memory.clone(),
            self.ui_bus.clone(),
            session_id,
            prompt.into(),
            live_ceiling,
            None,
            self.services.repo_instructions.clone(),
            requested_output_cap,
            source_context,
        )
        .await
    }
}

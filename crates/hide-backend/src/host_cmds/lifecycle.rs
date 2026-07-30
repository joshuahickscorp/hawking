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
    pub fn open_workspace(workspace_root: impl Into<PathBuf>) -> Result<Self> {
        Self::from_services(BackendServices::open_workspace(workspace_root)?)
    }

    pub fn from_services(services: BackendServices) -> Result<Self> {
        let services = Arc::new(services);
        let tools = Arc::new(build_default_tool_registry());
        // W1: register configured MCP servers into the live tool registry at boot.
        // Source: `.hide/mcp.json` (array of `McpServerDescriptor`) under the
        // workspace root — same `.hide/` layout every other durable workspace
        // artifact uses. A server that fails to start is logged + evented and
        // does NOT fail host boot (`register_mcp_servers` already returns
        // per-server results).
        register_mcp_servers_at_boot(&services, &tools);
        let ui_bus = Arc::new(UiEventBus::default());
        // RECORDED at construction, so there is no such thing as a dispatch through this host that
        // produces no tool events and no reviewable diff.
        let dispatcher = Arc::new(
            build_default_tool_dispatcher(&services.config, tools.clone()).with_observer(Arc::new(
                DispatchRecorder::new(services.clone(), ui_bus.clone()),
            )),
        );
        let connectors = Arc::new(ConnectorRegistry::default());
        register_backend_connectors(&connectors, &services);
        let interrupts = Arc::new(InterruptHub::default());
        let runtime = Self::maybe_boot_runtime(&services);
        // Re-register the runtime connector now that the supervisor exists, so its `state` method
        // is a real read of the engine instead of a guess from the static role registry.
        connectors.register(crate::connectors::runtime_connector(
            &services,
            runtime.clone(),
        ));
        // One surface graph bound to the host primary session. All three lenses
        // share that session id; handoffs never mint a parallel session.
        let primary = services.session();
        let surfaces = Arc::new(SurfaceGraphService::for_session(
            &primary,
            services.event_log.clone(),
            ui_bus.clone(),
        ));
        // Publish the initial projection so FE navigation can bind on connect.
        surfaces.publish_view();
        Ok(Self {
            commands: CommandRouter::with_interrupts(
                services.event_log.clone(),
                interrupts.clone(),
            ),
            replay: BackendReplayService::new(
                services.event_log.clone(),
                services.projection_store.clone(),
            ),
            services,
            connectors,
            tools,
            dispatcher,
            security: SecurityServices::default(),
            processes: Arc::new(ProcessSupervisor::new(ui_bus.clone())),
            ui_bus,
            interrupts,
            approvals: Arc::new(ApprovalHub::default()),
            gate_book: Arc::new(GateBook::default()),
            runtime,
            connections: Arc::new(ConnectionRegistry::default()),
            surfaces,
        })
    }

    /// Construct + (in the background) boot the runtime supervisor, GATED behind
    /// the `HIDE_MODEL_WEIGHTS` env var. When unset (the headless/test default)
    /// this returns `None` and NO server is ever spawned, so the ~410 unit tests
    /// stay model-free. When set to a weights path, the `RuntimeSupervisor` is
    /// built for `hawking serve --weights <path>` and `boot()` is spawned on the
    /// current tokio runtime so construction stays synchronous and NON-FATAL: a
    /// missing binary, a bad path, or a `/healthz` that never comes up just
    /// leaves the supervisor in `Failed`/`Booting`; the host is still returned
    /// and fully usable (it will report "model offline" rather than fake a
    /// token). Related env:
    /// * `HIDE_MODEL_ADDR` — bind (default `127.0.0.1:8745`, distinct from
    ///   hide-serve's 8744)
    /// * `HIDE_HAWKING_BIN` — path to the `hawking` binary (default: `hawking`
    ///   on `PATH`)
    /// * `HIDE_MODEL_BOOT_TIMEOUT_SECS` — wait for `/healthz` (default 300)
    pub(crate) fn maybe_boot_runtime(
        services: &Arc<BackendServices>,
    ) -> Option<Arc<RuntimeSupervisor>> {
        let weights = std::env::var("HIDE_MODEL_WEIGHTS").ok()?;
        if weights.trim().is_empty() {
            return None;
        }
        let bind = std::env::var("HIDE_MODEL_ADDR")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "127.0.0.1:8745".to_string());
        let layout = services.layout();
        let cfg = SupervisorConfig::for_hawking_serve(
            bind,
            &services.config.workspace_root,
            &weights,
            layout.hide_dir.join("runtime.lock"),
        );
        let supervisor = Arc::new(RuntimeSupervisor::for_hawking_serve(cfg));
        // Boot in the background so construction is sync + non-fatal. If we are
        // not inside a tokio runtime (a sync test that set the env var), skip the
        // spawn but still hand back the (Down) supervisor: health/status report
        // it honestly and generation surfaces "model offline".
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let sup = supervisor.clone();
            handle.spawn(async move {
                if let Err(e) = sup.boot().await {
                    // Non-fatal: the supervisor already transitioned to Failed and
                    // recorded the reason; just surface it (consistent with the
                    // supervisor's own eprintln! diagnostics).
                    eprintln!("warning: runtime supervisor boot failed (non-fatal): {e}");
                }
            });
        }
        Some(supervisor)
    }

    /// Subscribe to the live push UiEvent stream (Wire-B). Ordered; a lagging
    /// subscriber gets a `Lagged` signal rather than stalling the host.
    pub fn subscribe_ui(&self) -> tokio::sync::broadcast::Receiver<UiEvent> {
        self.ui_bus.subscribe()
    }

    /// The push UiEvent bus (for callers that want to publish/coalesce directly).
    pub fn ui_bus(&self) -> &Arc<UiEventBus> {
        &self.ui_bus
    }

    /// The interrupt hub control intents signal onto (shared with the kernel).
    pub fn interrupts(&self) -> &Arc<InterruptHub> {
        &self.interrupts
    }

    /// The approval hub `approve_effect`/`deny_effect` intents deposit onto
    /// (shared with the running kernel turn). A paused effectful step drains it
    /// to resume or skip.
    pub fn approvals(&self) -> &Arc<ApprovalHub> {
        &self.approvals
    }

    /// The supervised runtime's state (`None` when no model is configured, i.e.
    /// `HIDE_MODEL_WEIGHTS` unset). Surfaced so the FE's `RuntimeStatus` can
    /// reflect down/booting/ready/degraded/failed.
    pub fn runtime_state(&self) -> Option<RuntimeSupervisorState> {
        self.runtime.as_ref().map(|s| s.state())
    }

    /// The base URL of the supervised runtime, but only when it is `Ready`. A
    /// `None` here means "no model online to generate against", so the caller
    /// surfaces that as a `RuntimeStatus`/`Error` UiEvent rather than faking a
    /// token. When Ready, also installs [`HttpEmbeddingClient`] on the sqlite
    /// code index so hybrid search's semantic leg is real (never a silent stub).
    pub fn open_live_thread(&self, session: SessionId) -> LiveThread {
        LiveThread::open(session, self.services.event_log.clone())
    }

    /// Handle a client Initialize (Stage 4 capability negotiation, Codex mechanism
    /// 5). Records the connection's negotiated `capabilities` (the experimental-api
    /// gate + the opt-out notification method set) keyed by `connection_id`, and
    /// returns the server-info reply. The stored capabilities are consulted in the
    /// notification emit path ([`Self::notification_for_connection`]). The
    /// `ClientInfo` is accepted per the handshake but not retained (only the
    /// negotiation levers drive server behavior). Model-free.
    pub fn initialize(
        &self,
        connection_id: impl Into<String>,
        _client: ClientInfo,
        capabilities: ClientCapabilities,
    ) -> InitializeResponse {
        self.connections.initialize(connection_id, capabilities);
        InitializeResponse {
            user_agent: format!("hide-backend/{}", env!("CARGO_PKG_VERSION")),
            workspace_root: self.services.config.workspace_root.display().to_string(),
            platform_family: std::env::consts::FAMILY.to_string(),
            platform_os: std::env::consts::OS.to_string(),
        }
    }

    /// The per-connection capability registry (Stage 4 Initialize handshake). The
    /// notification emit path consults it to suppress opted-out methods.
    pub fn connections(&self) -> &ConnectionRegistry {
        &self.connections
    }

    /// Promote a LIVE interactive run to a durable BACKGROUND JOB (Stage 4
    /// background promotion) WITHOUT restarting it: the still-running run keeps its
    /// `run_id` and its tokio task, so it survives a client disconnect. A durable
    /// [`JobRecord`] bound to that run id is created (status `Running`, a Manual
    /// wake trigger), so a fresh host recovers it and a reconnecting client can
    /// find, inspect, steer, pause, stop, fork, and resume-in-foreground the SAME
    /// run. Also appends a `run.promoted` event tying the run to the job on the
    /// session log. Reuses `job_create` (never a second store); model-free.
    pub async fn status(&self) -> BackendStatus {
        BackendStatus {
            workspace_root: self.services.config.workspace_root.clone(),
            capabilities: self.services.capabilities.clone(),
            connectors: self.connectors.statuses().await,
            tools: self.tools.specs(),
            model_roles: self.services.role_registry.all(),
            runtime: self.runtime_state(),
        }
    }

    pub async fn health(&self) -> HealthReport {
        let mut checks = Vec::new();
        let layout = self.services.layout();
        checks.push(path_check("hide_dir", &layout.hide_dir));
        checks.push(path_check("event_log", &layout.event_log));
        checks.push(path_check("blobs", &layout.blobs));
        checks.push(path_check("projections", &layout.projections));
        checks.push(path_check("kv", &layout.kv));
        checks.push(count_check("tools", self.tools.specs().len()));
        checks.push(count_check(
            "model_roles",
            self.services.role_registry.all().len(),
        ));
        for connector in self.connectors.statuses().await {
            checks.push(HealthCheck {
                name: format!("connector:{}", connector.id),
                status: if connector.healthy {
                    HealthStatus::Ok
                } else {
                    HealthStatus::Failed
                },
                detail: connector.detail,
            });
        }
        // Surface the runtime supervisor state so the FE's RuntimeStatus
        // reflects down/booting/ready/degraded/failed. When NO model is
        // configured (the headless default) the runtime is simply absent and we
        // report `Ok` with a "not configured" note: a missing model is not a
        // health failure of the host. A configured-but-not-ready runtime maps to
        // Degraded (still booting) or Failed (crashed past its restart cap).
        let (rt_status, rt_detail) = match self.runtime_state() {
            None => (HealthStatus::Ok, "not configured".to_string()),
            Some(RuntimeSupervisorState::Ready) => (HealthStatus::Ok, "ready".to_string()),
            Some(RuntimeSupervisorState::Failed) => (HealthStatus::Failed, "failed".to_string()),
            Some(other) => (HealthStatus::Degraded, format!("{other:?}").to_lowercase()),
        };
        checks.push(HealthCheck {
            name: "runtime".to_string(),
            status: rt_status,
            detail: rt_detail,
        });
        let status = if checks
            .iter()
            .any(|check| check.status == HealthStatus::Failed)
        {
            HealthStatus::Failed
        } else if checks
            .iter()
            .any(|check| check.status == HealthStatus::Degraded)
        {
            HealthStatus::Degraded
        } else {
            HealthStatus::Ok
        };
        HealthReport {
            component: "hide-backend".to_string(),
            status,
            checks,
        }
    }
}

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
use hide_kernel::verify_plane::{
    Finding, GateDecision, ReviewRole, ReviewRoleProfile, SourceFile, StaticAnalysisOracle,
    TieredVerdict, VerificationReceipt, VerificationTier,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[path = "host_types.rs"]
mod host_types;
pub use host_types::*;

#[path = "host_support_0.rs"]
mod host_support_0;
#[allow(unused_imports)]
pub(crate) use host_support_0::*;

#[path = "host_support_1.rs"]
mod host_support_1;
#[allow(unused_imports)]
pub(crate) use host_support_1::*;

#[path = "host_support_2.rs"]
mod host_support_2;
#[allow(unused_imports)]
pub(crate) use host_support_2::*;

#[path = "host_support_3.rs"]
mod host_support_3;
pub use host_support_3::BackendStatus;
#[allow(unused_imports)]
pub(crate) use host_support_3::*;

// Host command surface — semantic modules (replaces mechanical host_ops_N splits).
#[path = "host_cmds/intent_effects.rs"]
mod host_intent_effects;
#[path = "host_cmds/intent_entry.rs"]
mod host_intent_entry;
#[path = "host_cmds/intent_handlers.rs"]
mod host_intent_handlers;
#[path = "host_cmds/jobs_memory.rs"]
mod host_jobs_memory;
#[path = "host_cmds/lifecycle.rs"]
mod host_lifecycle;
#[path = "host_cmds/tools_workspace.rs"]
mod host_tools_workspace;
#[path = "host_cmds/turn.rs"]
mod host_turn;
pub use host_turn::HcliTurnResult;
#[path = "host_cmds/verify_checkpoint.rs"]
mod host_verify_checkpoint;

pub struct BackendHost {
    pub services: SharedBackend,
    pub connectors: Arc<ConnectorRegistry>,
    pub tools: Arc<ToolRegistry>,
    /// The one permission-gated, verifying applier. A frontend save reaches it the way the agent's
    /// edits do, through [`BackendHost::dispatch_tool`] (no second write channel).
    pub dispatcher: Arc<ToolDispatcher>,
    pub security: SecurityServices,
    pub replay: BackendReplayService,
    commands: CommandRouter,
    /// The push Wire-B bus (broadcast + coalescing). The pull `ui_events` API is
    /// retained for replay/catch-up; this is the live path.
    ui_bus: Arc<UiEventBus>,
    /// Shared with the CommandRouter so control intents reach running runs.
    interrupts: Arc<InterruptHub>,
    /// The per-run approval mailbox. A live kernel turn under bounded
    /// `SuggestOnly` autonomy PAUSES on an effectful step; an `approve_effect`/
    /// `deny_effect` intent deposits the decision here, and the running turn
    /// drains it to resume (approve) or skip the step (deny). Without this, an
    /// effectful turn spun `Paused` until the Governor aborted at the step cap.
    approvals: Arc<ApprovalHub>,
    /// Genuinely destructive commands ([`dangerous_command`]) are not dropped - they are parked
    /// here under a unique gate id and surfaced as a `SecurityGate` UiEvent. An `approve_gate`
    /// intent with that id releases and runs the command; `deny_gate` drops it.
    gate_book: Arc<GateBook>,
    /// The supervised `hawking serve` runtime, present only when a model is
    /// configured (`HIDE_MODEL_WEIGHTS` set). `None` keeps the host fully usable
    /// headless: the ~410 unit tests never spawn a server. When present, its
    /// state machine (`Down -> Booting -> Ready -> Degraded -> Failed`) is
    /// surfaced through `health()`/`status()`, and `base_url()` (once `Ready`)
    /// is where `SubmitTurn` generation is routed.
    runtime: Option<Arc<RuntimeSupervisor>>,
    /// The session-aware terminal process surface (Trace D). Terminal commands and
    /// long-lived service processes run sandbox-confined here, stream incrementally,
    /// persist across navigation, and can be captured as durable artifacts.
    processes: Arc<ProcessSupervisor>,
    /// Per-connection negotiated capabilities (Stage 4 Initialize handshake): the
    /// experimental-api gate and the opt-out notification method set, consulted in
    /// the notification emit path so a connection never receives a class of pushes
    /// it opted out of.
    connections: Arc<ConnectionRegistry>,
    /// One session, three lenses (YOU / CHAT / IDE). Surfaces share this graph;
    /// they do not each construct their own. Handoff capsules carry claims only.
    surfaces: Arc<SurfaceGraphService>,
}

/// Load MCP server descriptors for host boot.
///
/// Chosen source: `<workspace>/.hide/mcp.json` — a JSON array of
/// [`hide_kernel::tooling::mcp::McpServerDescriptor`]. Matches the existing workspace
/// layout (every durable host artifact already lives under `.hide/`) and reuses
/// the descriptor type's own serde, so there is no parallel config schema.
/// Absent / unreadable / invalid files yield an empty list (no MCP, no error).
pub(super) fn load_mcp_descriptors(
    workspace_root: &Path,
) -> Vec<hide_kernel::tooling::mcp::McpServerDescriptor> {
    let path = workspace_root.join(".hide").join("mcp.json");
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) => return Vec::new(),
    };
    match serde_json::from_slice(&bytes) {
        Ok(descs) => descs,
        Err(e) => {
            eprintln!(
                "warning: ignoring invalid MCP config at {}: {e}",
                path.display()
            );
            Vec::new()
        }
    }
}

/// Drive an async future from the sync `from_services` path. Prefer the current
/// tokio runtime (multi-thread tests use `block_in_place`); otherwise spin a
/// short-lived current-thread runtime.
pub(super) fn block_on_async<F: std::future::Future>(fut: F) -> F::Output {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(fut)),
        Err(_) => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("mcp boot runtime")
            .block_on(fut),
    }
}

/// Connect configured MCP servers and register their tools into `tools`.
/// Per-server failures are logged + recorded as `mcp.server_failed` events and
/// never abort host construction.
pub(super) fn register_mcp_servers_at_boot(services: &BackendServices, tools: &ToolRegistry) {
    let descriptors = load_mcp_descriptors(&services.config.workspace_root);
    if descriptors.is_empty() {
        return;
    }
    let log = services.event_log.clone();
    let session = services.session();
    block_on_async(async move {
        let results = hide_kernel::tooling::mcp::register_mcp_servers(&descriptors, tools).await;
        for reg in results {
            let (kind, payload) = match &reg.error {
                Some(err) => {
                    eprintln!(
                        "warning: MCP server '{}' failed to register (non-fatal): {err}",
                        reg.server_id
                    );
                    (
                        "mcp.server_failed",
                        json!({
                            "server_id": reg.server_id,
                            "error": err,
                        }),
                    )
                }
                None => (
                    "mcp.server_registered",
                    json!({
                        "server_id": reg.server_id,
                        "tools": reg.tools,
                    }),
                ),
            };
            let _ = log
                .append(NewEvent::system(session.clone(), kind, payload))
                .await;
        }
    });
}

#[cfg(test)]
#[path = "host_live_manifest_tests.rs"]
mod live_manifest_tests;

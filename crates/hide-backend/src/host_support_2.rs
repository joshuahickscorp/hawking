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

/// Attach capability + rot + meter to a compiled manifest so the durable
/// `context.compiled` event and any projection carry auditable numbers.
///
/// `tokens_estimated` is `true` when packing used the `chars/4` heuristic rather
/// than a real tokenizer — the meter must never claim tokenizer-true counts then.
pub(crate) fn seal_compiled_manifest(
    manifest: &mut hawking_context::ContextManifest,
    capability: hawking_context::ContextCapability,
    live: Option<&hawking_context::ManifestLive>,
    tokens_estimated: bool,
) {
    use hawking_context::{detect_context_rot, ContextMeter, RotThresholds};
    let occupancy = live.map(|l| l.occupancy);
    let watermark = live.map(|l| l.watermark);
    let fidelity = live.and_then(|l| l.recall_fidelity);
    let rot = detect_context_rot(
        manifest,
        occupancy,
        watermark,
        fidelity,
        RotThresholds::default(),
    );
    let meter = ContextMeter::from_parts(
        &capability,
        manifest.used_tokens,
        tokens_estimated,
        live,
        Some(&rot),
    );
    manifest.capability = Some(capability);
    manifest.rot = Some(rot);
    manifest.meter = Some(meter);
}

/// JSON payload for the durable `context.compiled` marker: compile stats plus
/// the honest capability / rot / meter picture (so a later audit never has to
/// re-infer whether a number was measured).
pub(crate) fn context_compiled_payload(
    manifest: &hawking_context::ContextManifest,
    out_budget: Option<usize>,
    path: &str,
    run_id: Option<&str>,
) -> serde_json::Value {
    let mut body = json!({
        "used_tokens": manifest.used_tokens,
        "retained": manifest.retained.len(),
        "dropped": manifest.dropped.len(),
        "path": path,
        "capability": manifest.capability,
        "rot": manifest.rot,
        "meter": manifest.meter,
        // Hard rule, restated on every durable record.
        "native_is_not_usable": true,
    });
    if let Some(b) = out_budget {
        body["budget"] = json!(b);
    }
    if let Some(id) = run_id {
        body["run_id"] = json!(id);
    }
    body
}

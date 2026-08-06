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

/// What is parked at the security gate awaiting an `approve_gate` / `deny_gate` decision.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PendingAction {
    /// A terminal command classified dangerous. Runs SANDBOX-confined on release.
    Command {
        argv: Vec<String>,
        cwd: Option<String>,
    },
    /// A custom intent whose `CommandSpec` declares `ApprovalPolicy::Ask`. Recorded in the
    /// event log already; its EFFECT runs only once released.
    Intent { name: String, payload: Value },
}

/// A bounded book of commands parked at the security gate, keyed by gate id. Bounded so a never-
/// answered gate cannot leak unboundedly: past `CAP` the book REFUSES to park anything more, and
/// the caller is told its action was not held. It used to evict the oldest entry, which silently
/// dropped a pending approval on the floor and turned a later approve of it into a no-op the
/// frontend read as success. Human-approved gates are rare, so a small `Vec` under a `Mutex` is
/// ample. Gate ids are `command:<n>` (monotonic), unique so concurrent gates never collide.
#[derive(Default)]
pub(crate) struct GateBook {
    inner: std::sync::Mutex<Vec<(String, PendingAction)>>,
}

impl GateBook {
    pub(crate) const CAP: usize = 32;

    /// Park an action and return its fresh gate id, or `None` when `CAP` decisions are already
    /// outstanding (fail closed: nothing is parked, so nothing is silently lost).
    pub(crate) fn hold(&self, action: PendingAction) -> Option<String> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static GATE_SEQ: AtomicU64 = AtomicU64::new(1);
        let mut g = self.inner.lock().unwrap();
        if g.len() >= Self::CAP {
            return None;
        }
        let gate = format!("command:{}", GATE_SEQ.fetch_add(1, Ordering::Relaxed));
        g.push((gate.clone(), action));
        Some(gate)
    }

    /// Remove and return the action parked under `gate` (approve path). `None` if unknown.
    pub(crate) fn take(&self, gate: &str) -> Option<PendingAction> {
        let mut g = self.inner.lock().unwrap();
        g.iter().position(|(k, _)| k == gate).map(|i| g.remove(i).1)
    }

    /// Drop the command parked under `gate` (deny path). Returns whether one was parked.
    pub(crate) fn remove(&self, gate: &str) -> bool {
        let mut g = self.inner.lock().unwrap();
        match g.iter().position(|(k, _)| k == gate) {
            Some(i) => {
                g.remove(i);
                true
            }
            None => false,
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }
}

/// Classify a command as genuinely destructive / system-level. Returns `Some(reason)` to block, `None`
/// to allow. Conservative: ordinary dev commands (build, test, git, `rm -rf node_modules`) pass; only
/// privilege escalation, filesystem destroyers, recursive deletes of a system/home path, remote code
/// piped into a shell, and fork bombs are caught.
pub(crate) fn dangerous_command(argv: &[String]) -> Option<&'static str> {
    let prog = argv.first().map(|s| s.as_str()).unwrap_or("");
    let j = argv.join(" ").to_lowercase();
    if prog == "sudo" || prog == "doas" {
        return Some("runs as administrator");
    }
    if prog == "mkfs" || j.contains("mkfs.") {
        return Some("formats a filesystem");
    }
    if prog == "dd" && j.contains("of=/dev/") {
        return Some("writes raw to a device");
    }
    if prog == "rm"
        && (j.contains("-rf") || j.contains("-fr") || (j.contains("-r") && j.contains("-f")))
    {
        if j.contains(" /") || j.contains(" ~") || j.contains(" /*") {
            return Some("recursively deletes a system path");
        }
    }
    if (j.contains("curl ") || j.contains("wget "))
        && (j.contains("| sh") || j.contains("|sh") || j.contains("| bash") || j.contains("|bash"))
    {
        return Some("pipes a remote script into a shell");
    }
    if j.contains(":(){") || j.contains(":|:&") {
        return Some("fork bomb");
    }
    if (prog == "chmod" || prog == "chown")
        && j.contains("-r")
        && (j.contains(" /") || j.contains(" ~"))
    {
        return Some("recursively changes permissions on a system path");
    }
    None
}

// Run a command in the workspace and stream stdout/stderr back as tool_progress (the terminal renders
// them). Confined to the workspace root. A real command runner, not a full interactive PTY. The
// security gate is applied UPSTREAM (in `spawn_command_run`), so reaching here means the command is
// either inherently safe or was user-approved via the gate round-trip.
pub(crate) async fn exec_command_streamed(
    ui_bus: Arc<UiEventBus>,
    root: PathBuf,
    argv: Vec<String>,
    cwd: Option<String>,
) {
    use std::sync::atomic::{AtomicU64, Ordering};
    use tokio::io::AsyncBufReadExt;
    static SHELL_SEQ: AtomicU64 = AtomicU64::new(1);
    let call_id = format!("shell:{}", SHELL_SEQ.fetch_add(1, Ordering::Relaxed));
    let line = |bus: &Arc<UiEventBus>, message: String| {
        bus.publish(UiEvent {
            seq: 0,
            session_id: None,
            kind: UiEventKind::ToolProgress {
                call_id: call_id.clone(),
                message,
                event_id: None,
            },
        });
    };

    // Confine the cwd to the workspace root (reject any escape).
    let dir = match &cwd {
        Some(c) if !c.contains("..") => root.join(c.trim_start_matches('/')),
        _ => root.clone(),
    };

    let mut command = tokio::process::Command::new(&argv[0]);
    command
        .args(&argv[1..])
        .current_dir(&dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    let mut child = match command.spawn() {
        Ok(c) => c,
        Err(e) => {
            line(&ui_bus, format!("{}: {}", argv[0], e));
            return;
        }
    };

    let mut readers = Vec::new();
    if let Some(out) = child.stdout.take() {
        let bus = ui_bus.clone();
        let cid = call_id.clone();
        readers.push(tokio::spawn(async move {
            let mut lines = tokio::io::BufReader::new(out).lines();
            while let Ok(Some(l)) = lines.next_line().await {
                bus.publish(UiEvent {
                    seq: 0,
                    session_id: None,
                    kind: UiEventKind::ToolProgress {
                        call_id: cid.clone(),
                        message: l,
                        event_id: None,
                    },
                });
            }
        }));
    }
    if let Some(err) = child.stderr.take() {
        let bus = ui_bus.clone();
        let cid = call_id.clone();
        readers.push(tokio::spawn(async move {
            let mut lines = tokio::io::BufReader::new(err).lines();
            while let Ok(Some(l)) = lines.next_line().await {
                bus.publish(UiEvent {
                    seq: 0,
                    session_id: None,
                    kind: UiEventKind::ToolProgress {
                        call_id: cid.clone(),
                        message: l,
                        event_id: None,
                    },
                });
            }
        }));
    }
    let status = child.wait().await;
    for r in readers {
        let _ = r.await;
    }
    match status {
        Ok(s) if s.success() => line(&ui_bus, "exit 0".to_string()),
        Ok(s) => line(&ui_bus, format!("exit {}", s.code().unwrap_or(-1))),
        Err(e) => line(&ui_bus, format!("wait failed: {e}")),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendStatus {
    pub workspace_root: PathBuf,
    pub capabilities: BackendCapabilities,
    pub connectors: Vec<ConnectorStatus>,
    pub tools: Vec<ToolSpec>,
    pub model_roles: Vec<ModelRole>,
    /// The supervised runtime's state, or `None` when no model is configured
    /// (`HIDE_MODEL_WEIGHTS` unset). Lets the FE reflect down/booting/ready/
    /// degraded/failed.
    #[serde(default)]
    pub runtime: Option<RuntimeSupervisorState>,
}

pub(crate) fn tool_result_summary(result: &ToolResult) -> String {
    if let Some(error) = &result.error {
        return format!("{}: {}", error.code, error.message);
    }
    if let Some(value) = &result.structured_content {
        return value.to_string();
    }
    format!("{:?}", result.status)
}

/// Extract a permission-engine target from a tool call's args: a filesystem
/// `path`, else the first `argv` token, else the tool id. Used by the policy
/// ledger's engine consultation ([`BackendHost::permission_verdict_for`]).
pub(crate) fn policy_target_from_args(tool_id: &str, args: &Value) -> String {
    if let Some(path) = args.get("path").and_then(|value| value.as_str()) {
        return path.to_string();
    }
    if let Some(first) = args
        .get("argv")
        .and_then(|value| value.as_array())
        .and_then(|argv| argv.first())
        .and_then(|value| value.as_str())
    {
        return first.to_string();
    }
    tool_id.to_string()
}

pub(crate) fn path_check(name: &str, path: &std::path::Path) -> HealthCheck {
    let exists = path.exists();
    HealthCheck {
        name: name.to_string(),
        status: if exists {
            HealthStatus::Ok
        } else {
            HealthStatus::Failed
        },
        detail: if exists {
            path.display().to_string()
        } else {
            format!("missing {}", path.display())
        },
    }
}

pub(crate) fn count_check(name: &str, count: usize) -> HealthCheck {
    HealthCheck {
        name: name.to_string(),
        status: if count == 0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Ok
        },
        detail: count.to_string(),
    }
}

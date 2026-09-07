//! The ONE command registry (consolidation gate task 2).
//!
//! HIDE has one semantic object model ([`crate::model`]) and one wire protocol
//! ([`crate::protocol`]). This module adds the third leg of the authority: the
//! single canonical table of every user-invocable command, so that every surface
//! (a toolbar button, a keyboard shortcut, a context-menu item, the command
//! palette, a chat action, an IDE gesture, an ACP peer, or the SDK) resolves the
//! SAME command from the SAME table instead of each surface re-declaring its own
//! bindings. A command names a capability, the control(s) it appears under, the
//! selection and capabilities it needs, its effects and approval policy, how it
//! is undone, and how it reaches the backend.
//!
//! Owner decision: this lives in `hide-protocol` (the existing schema authority)
//! and is projected by `hide-sdk` codegen, rather than in a new
//! `hide-command-registry` crate. A new crate would add a compile unit and a
//! naming-symmetry temptation with nothing behind it; the catalog is a schemars
//! table exactly like [`Method`](crate::protocol::Method), so it belongs where
//! the other wire shapes already live. See
//! `docs/hide-impl/consolidation/HIDE_COMMAND_REGISTRY_SPEC.md`.
//!
//! Model-free: this is a static table plus deterministic checks over it. It runs
//! no model, opens no socket, and produces no runtime bytes.
//!
//! # Seeding
//!
//! [`command_catalog`] is seeded from the ranked census priority in
//! `HIDE_BACKEND_WITHOUT_SURFACE_REPORT.md`: each command maps a REAL host
//! capability to a REAL existing control (never a new button). The bindings are
//! grounded in `crates/hide-core/src/api.rs` (the `Intent` variants), the
//! host-handled custom names in `crates/hide-backend/src/host.rs`, and the
//! `Method` set in [`crate::protocol`].

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::plan::Effect;

/// A UI surface a command can appear on. The closed set of places the shipped
/// shell renders controls (Bible surfaces, plus the palette as the universal
/// fallback surface).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Surface {
    /// Private general-purpose personal AI lens (YOU).
    You,
    Chat,
    Ide,
    Home,
    ContextStack,
    StatusBar,
    StateTimeline,
    Terminal,
    DiffReview,
    Settings,
    Fleet,
    Palette,
    Editor,
}

/// The domain a command belongs to. The coverage test asserts the priority
/// domains are all present, so this doubles as the campaign's progress ledger.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    /// Submit, cancel, pause, resume a turn.
    Turn,
    /// Accept or reject a proposed diff.
    Diff,
    /// Scrub, fork, and checkpoint the state timeline.
    Timeline,
    /// Deterministic verification (static analysis, receipts).
    Verify,
    /// Side-chat create and merge.
    SideChat,
    /// Integrity-verified checkpoints.
    Checkpoint,
    /// Durable outcome-governed memory.
    Memory,
    /// Durable goal plus acceptance test.
    Goal,
    /// Mid-turn steering.
    Steer,
    /// Multi-repo workspace and trust.
    Workspace,
    /// Per-session environment switch.
    Environment,
    /// Transcript search.
    Search,
    /// Open a file or navigate the editor.
    File,
    /// Run a shell command.
    Terminal,
    /// Plan step approve / edit / reorder / skip / repair.
    Plan,
    /// Durable background job promotion and foreground resume.
    Background,
    /// Cross-surface claim-only handoffs (YOU↔CHAT↔IDE).
    Handoff,
    /// Active surface lens switch on the shared session.
    Surface,
}

/// A declared effect class for a command. This mirrors the protocol effect
/// classes exactly: it reuses [`crate::plan::Effect`], the schema authority's
/// own effect enum (ReadFs, WriteFs, Network, Process, Shell, Vcs, Environment,
/// Approval, AgentSpawn, State, Other).
///
/// Note on the other `Effect` in the tree: `hide-extension-registry` owns a
/// SECURITY least-privilege effect ranking (Read, Write, GitMutation, Execute,
/// Process, Network, SecretAccess, ExternalMutation, Irreversible, Privileged).
/// That is a different taxonomy for capability resolution and does not derive
/// `JsonSchema`; reusing it here would add a cross-crate dependency and a schema
/// mismatch. The command registry mirrors the protocol effect classes instead,
/// so `EffectClass` is a plain alias of the in-crate protocol effect.
pub type EffectClass = Effect;

/// Whether the command needs a live selection before it can run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RequiredSelection {
    None,
    Text,
    File,
    Hunk,
    PlanStep,
    Any,
}

/// The approval gate a command passes through before it takes effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalPolicy {
    /// Runs immediately.
    Auto,
    /// Prompts the human for a decision.
    Ask,
    /// Must run inside a sandbox.
    RequireSandbox,
    /// Never allowed from this surface.
    Deny,
}

/// How a command's effect is unwound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum UndoStrategy {
    /// Nothing to undo.
    None,
    /// Apply an inverse operation.
    Inverse,
    /// Restore a checkpoint.
    Checkpoint,
    /// Reject the pending proposal.
    Reject,
}

/// How a surface reaches the backend for this command.
///
/// - [`BackendBinding::Intent`] names a real `hide-core` `Intent` variant
///   (`crates/hide-core/src/api.rs`) for the HCLI backend bridge.
/// - [`BackendBinding::Custom`] names an `Intent::Custom{name}` the HCLI host
///   handles and exposes through [`WIRE_CUSTOM_NAMES`]. There is no pending
///   tier: a name the host does not handle is not on the contract at all.
/// - [`BackendBinding::Rpc`] names an elevated capability: either a real
///   [`Method`](crate::protocol::Method) string or a census-confirmed host
///   capability in [`HOST_CAPABILITIES`].
/// - [`BackendBinding::LocalOnly`] is a client-local action with no backend call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", content = "target", rename_all = "snake_case")]
pub enum BackendBinding {
    Intent(String),
    Custom(String),
    Rpc(String),
    LocalOnly,
}

/// One command in the ONE registry: a capability mapped to the control(s) it
/// appears under, with everything a surface needs to render, gate, invoke, and
/// undo it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CommandSpec {
    /// Stable unique id (the capability name), the key every surface resolves on.
    pub id: String,
    /// Human title for menus and the palette.
    pub title: String,
    /// One-line description of what the command does.
    pub description: String,
    pub category: Category,
    /// The surface this command lives on first.
    pub primary_surface: Surface,
    /// Every surface that offers the command.
    pub available_surfaces: Vec<Surface>,
    /// The selection kind the command needs before it can run.
    pub required_selection: RequiredSelection,
    /// Negotiated server capabilities the command requires (see
    /// [`ServerCapabilities`](crate::protocol::ServerCapabilities)).
    pub required_capabilities: Vec<String>,
    /// The effect classes the command may cause.
    pub effects: Vec<EffectClass>,
    pub approval_policy: ApprovalPolicy,
    /// A keyboard shortcut (`Mod` is Cmd on macOS, Ctrl elsewhere), if bound.
    pub keyboard_shortcut: Option<String>,
    /// Whether the command palette lists it.
    pub command_palette: bool,
    /// Whether a context menu offers it.
    pub context_menu: bool,
    /// A toolbar button id it binds to, if any.
    pub toolbar_binding: Option<String>,
    pub backend_binding: BackendBinding,
    pub undo_strategy: UndoStrategy,
    /// The receipt kind the command emits, if it seals one.
    pub receipt_kind: Option<String>,
    /// A telemetry event name, if the command is instrumented.
    pub telemetry: Option<String>,
}

/// The `Intent` variant tags the HCLI backend bridge accepts.
/// Mirror of the snake_case `#[serde(tag = "type")]` names in
/// `crates/hide-core/src/api.rs` (the `custom` escape hatch is excluded; custom
/// names are validated separately). Kept as a mirror the way `wire.ts` mirrors
/// `api.rs`; the integrity test uses it to reject an invented Intent binding.
pub const INTENT_NAMES: &[&str] = &[
    "submit_turn",
    "cancel_run",
    "pause_run",
    "resume_run",
    "accept_diff",
    "reject_diff",
    "scrub_to_event",
    "fork_session",
    "open_file",
    "run_command",
];

/// The canonical custom names for the HCLI/HIDE protocol. Every `Custom`
/// binding must be in here, and every name in here has an arm in
/// `crates/hide-backend/src/host.rs` `HANDLED_CUSTOM_NAMES` (asserted there).
pub const WIRE_CUSTOM_NAMES: &[&str] = &[
    "pty_input",
    "pty_resize",
    "run_search",
    "revert_diff",
    "save_file",
    "redirect_run",
    "approve_plan",
    "edit_plan_step",
    "reorder_plan",
    "approve_gate",
    "deny_gate",
    "new_session",
    "open_session",
    "create_worktree",
    "create_side_chat",
    "merge_side_chat",
    "goal_set",
    "goal_clear",
    "checkpoint_create",
    "checkpoint_restore",
    "approve_effect",
    "deny_effect",
    "skip_step",
    "repair_step",
    "checkpoint_rewind",
    "checkpoint_replay",
    "checkpoint_fork",
    "checkpoint_compare",
    "checkpoint_inspect",
    "promote_run",
    "resume_run_foreground",
    "memory_add",
    "memory_supersede",
    "memory_record_outcome",
    "memory_revalidate",
    "goal_evaluate",
    "workspace_set_repo_trust",
    "environment_switch",
    // The StatusBar Problems counter's producer. `handle_static_analysis_intent`
    // is the host arm; the CommandSpec below binds Custom, so the counter has a
    // way to fill itself instead of only ever reading a projection nothing wrote.
    "run_static_analysis",
    // The task-scoped write lease (crates/hide-backend/src/tools.rs). `grant_write_lease` is
    // `Ask`, so its effect is held at the security gate and only a human approval installs the
    // lease; `revoke_write_lease` is `Auto`, because a de-escalation may never need approval.
    "grant_write_lease",
    "revoke_write_lease",
    // The three process controls. `handle_process_intent` is the host arm; each addresses ONE
    // named process, so `process` is required (guessing "the latest" would stop the wrong one).
    "attach_process",
    "stop_process",
    "capture_process_artifact",
    // The sealed diff review receipt over a diff the HCLI backend produced.
    "export_review_receipt",
    // YOU / CHAT / IDE shared session graph (claim-only handoffs).
    "switch_surface",
    "handoff_create",
    "handoff_receive",
];

/// Census-confirmed host capabilities reachable over the elevated protocol that
/// are NOT also a [`Method`](crate::protocol::Method) string. An `Rpc` binding
/// must be a real `Method` OR one of these. No command binds `Rpc` at all any
/// more: `run_static_analysis` was the last bound to one of THESE and is now
/// `Custom`, dispatched over the HCLI backend bridge by
/// `handle_static_analysis_intent`, and `goal_get` (the last `Rpc` row of any
/// kind) is retired because HCLI has no `/rpc` client to dispatch it with. This
/// list stays as the
/// census record of the elevated surface.
/// Host refs inline:
/// - `run_static_analysis`     host.rs:1373
/// - `memory_add`              host.rs:1870
/// - `memory_supersede`        host.rs:1906
/// - `memory_record_outcome`   host.rs:1931
/// - `memory_revalidate`       host.rs:1957
/// - `memory_list`             host.rs:1885
/// - `goal_evaluate`           host.rs:1572
/// - `workspace_set_repo_trust` host.rs:1136
/// - `environment_switch`      host.rs:1193
pub const HOST_CAPABILITIES: &[&str] = &[
    "run_static_analysis",
    "memory_add",
    "memory_supersede",
    "memory_record_outcome",
    "memory_revalidate",
    "memory_list",
    "goal_evaluate",
    "workspace_set_repo_trust",
    "environment_switch",
];

/// How a catalog row reaches the backend. Live rows use only Intent and Custom;
/// Rpc and LocalOnly remain on [`BackendBinding`] for the schema but are unused.
#[derive(Clone, Copy)]
enum Bind {
    Intent(&'static str),
    Custom(&'static str),
}

/// Common identity and routing fields for one command. Non-default security,
/// effect, undo, selection, capability, shortcut, and policy fields live in the
/// override match inside `command_catalog`, not here.
struct Row {
    id: &'static str,
    title: &'static str,
    description: &'static str,
    category: Category,
    primary: Surface,
    surfaces: &'static [Surface],
    bind: Bind,
}

/// Build one common-field row. Argument order is fixed:
/// id, title, description, category, primary, surfaces, bind.
macro_rules! row {
    ($id:expr, $title:expr, $desc:expr, $cat:expr, $pri:expr, $surf:expr, $bind:expr) => {
        Row {
            id: $id,
            title: $title,
            description: $desc,
            category: $cat,
            primary: $pri,
            surfaces: $surf,
            bind: $bind,
        }
    };
}

fn caps(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

/// Common identity/routing table in declaration order (stable golden order).
const ROWS: &[Row] = {
    use Category as C;
    use Surface as S;
    &[
    row!("submit_turn", "Send message",
        "Submit the composer text as a new turn.",
        C::Turn, S::Chat, &[S::Chat, S::Home, S::Palette], Bind::Intent("submit_turn")),
    row!("cancel_run", "Cancel run",
        "Cancel the running turn.",
        C::Turn, S::Chat, &[S::Chat, S::StatusBar, S::Palette], Bind::Intent("cancel_run")),
    row!("pause_run", "Pause run",
        "Pause the running turn.",
        C::Turn, S::Chat, &[S::Chat, S::Palette], Bind::Intent("pause_run")),
    row!("resume_run", "Resume run",
        "Resume a paused turn.",
        C::Turn, S::Chat, &[S::Chat, S::Palette], Bind::Intent("resume_run")),
    row!("accept_diff", "Accept diff",
        "Apply a proposed diff hunk to the working tree.",
        C::Diff, S::DiffReview, &[S::DiffReview, S::Editor, S::Ide], Bind::Intent("accept_diff")),
    row!("reject_diff", "Reject diff",
        "Reject a proposed hunk, restoring that file on disk.",
        C::Diff, S::DiffReview, &[S::DiffReview, S::Editor, S::Ide], Bind::Intent("reject_diff")),
    row!("fork_session", "Fork from here",
        "Fork a new session from the selected timeline event.",
        C::Timeline, S::StateTimeline, &[S::StateTimeline, S::Palette], Bind::Intent("fork_session")),
    row!("open_file", "Open file",
        "Open a file in the editor.",
        C::File, S::Ide, &[S::Ide, S::Editor, S::Palette], Bind::Intent("open_file")),
    row!("run_command", "Run command",
        "Run a shell command in the terminal.",
        C::Terminal, S::Terminal, &[S::Terminal, S::Palette], Bind::Intent("run_command")),
    row!("run_static_analysis", "Run static analysis",
        "Run Tier-1 deterministic static analysis and show real problem counts.",
        C::Verify, S::StatusBar, &[S::StatusBar, S::ContextStack, S::Palette], Bind::Custom("run_static_analysis")),
    row!("create_side_chat", "New side chat",
        "Open a side chat thread that can later merge back.",
        C::SideChat, S::Chat, &[S::Chat, S::Palette], Bind::Custom("create_side_chat")),
    row!("merge_side_chat", "Merge side chat",
        "Merge a side chat's summary back into the main thread.",
        C::SideChat, S::Chat, &[S::Chat, S::Palette], Bind::Custom("merge_side_chat")),
    row!("checkpoint_create", "Create checkpoint",
        "Seal an integrity-verified restore point on the timeline.",
        C::Checkpoint, S::StateTimeline, &[S::StateTimeline, S::Palette], Bind::Custom("checkpoint_create")),
    row!("checkpoint_restore", "Restore checkpoint",
        "Restore the session to a sealed checkpoint.",
        C::Checkpoint, S::StateTimeline, &[S::StateTimeline, S::Palette], Bind::Custom("checkpoint_restore")),
    row!("memory_add", "Add memory note",
        "Store a durable outcome-governed note the agent keeps.",
        C::Memory, S::ContextStack, &[S::ContextStack, S::Editor, S::Palette], Bind::Custom("memory_add")),
    row!("memory_supersede", "Supersede memory",
        "Replace a stale memory while keeping its history.",
        C::Memory, S::ContextStack, &[S::ContextStack, S::Palette], Bind::Custom("memory_supersede")),
    row!("memory_record_outcome", "Record memory outcome",
        "Report that a remembered fact was right or wrong so it self-quarantines.",
        C::Memory, S::ContextStack, &[S::ContextStack, S::Palette], Bind::Custom("memory_record_outcome")),
    row!("memory_revalidate", "Revalidate memory",
        "Re-check a memory's citations against the repo on disk.",
        C::Memory, S::ContextStack, &[S::ContextStack, S::Palette], Bind::Custom("memory_revalidate")),
    row!("goal_set", "Set goal",
        "Set a durable goal and acceptance criteria for the session.",
        C::Goal, S::Home, &[S::Home, S::Chat, S::Palette], Bind::Custom("goal_set")),
    row!("goal_clear", "Clear goal",
        "Clear the session's goal.",
        C::Goal, S::Home, &[S::Home, S::Palette], Bind::Custom("goal_clear")),
    row!("goal_evaluate", "Evaluate goal",
        "Run the deterministic acceptance check for the goal.",
        C::Goal, S::Home, &[S::Home, S::Chat, S::Palette], Bind::Custom("goal_evaluate")),
    row!("steer", "Steer turn",
        "Redirect the running turn mid-flight via the interrupt hub.",
        C::Steer, S::Chat, &[S::Chat, S::Palette], Bind::Custom("redirect_run")),
    row!("workspace_set_repo_trust", "Set repo trust",
        "Trust a repo so its instructions and policy can activate.",
        C::Workspace, S::Home, &[S::Home, S::Settings, S::Palette], Bind::Custom("workspace_set_repo_trust")),
    row!("environment_switch", "Switch environment",
        "Switch the session's dev, prod, or sandbox environment.",
        C::Environment, S::Home, &[S::Home, S::StatusBar, S::Settings, S::Palette], Bind::Custom("environment_switch")),
    row!("run_search", "Search transcript",
        "Search the session transcript by literal or structured query, over the intent channel.",
        C::Search, S::Palette, &[S::Palette, S::Chat, S::Ide], Bind::Custom("run_search")),
    row!("checkpoint_rewind", "Rewind to checkpoint",
        "Rewind code, conversation, or both to a checkpoint on a fresh child session.",
        C::Checkpoint, S::StateTimeline, &[S::StateTimeline, S::Palette], Bind::Custom("checkpoint_rewind")),
    row!("checkpoint_replay", "Replay from checkpoint",
        "Re-apply the recorded history from a checkpoint forward onto a new lineage.",
        C::Checkpoint, S::StateTimeline, &[S::StateTimeline, S::Palette], Bind::Custom("checkpoint_replay")),
    row!("checkpoint_fork", "Fork from checkpoint",
        "Branch an ephemeral session seeded only with a checkpoint's inherited prefix.",
        C::Checkpoint, S::StateTimeline, &[S::StateTimeline, S::Palette], Bind::Custom("checkpoint_fork")),
    row!("checkpoint_compare", "Compare checkpoint",
        "Show the file-level code differences against a checkpoint or another session.",
        C::Checkpoint, S::StateTimeline, &[S::StateTimeline, S::Palette], Bind::Custom("checkpoint_compare")),
    row!("checkpoint_inspect", "Inspect checkpoint",
        "Verify a checkpoint's integrity and coverage.",
        C::Checkpoint, S::StateTimeline, &[S::StateTimeline, S::Palette], Bind::Custom("checkpoint_inspect")),
    row!("approve_plan", "Approve plan",
        "Approve a plan step, or the whole plan when no step is selected.",
        C::Plan, S::ContextStack, &[S::ContextStack, S::Chat, S::Palette], Bind::Custom("approve_plan")),
    row!("edit_plan_step", "Edit plan step",
        "Edit the text of a plan step.",
        C::Plan, S::ContextStack, &[S::ContextStack, S::Chat, S::Palette], Bind::Custom("edit_plan_step")),
    row!("reorder_plan", "Reorder plan",
        "Reorder the plan's steps to a new permutation.",
        C::Plan, S::ContextStack, &[S::ContextStack, S::Chat, S::Palette], Bind::Custom("reorder_plan")),
    row!("skip_step", "Skip step",
        "Skip a plan step with a recorded reason.",
        C::Plan, S::ContextStack, &[S::ContextStack, S::Chat, S::Palette], Bind::Custom("skip_step")),
    row!("repair_step", "Repair step",
        "Re-open a failed plan step so it can be retried.",
        C::Plan, S::ContextStack, &[S::ContextStack, S::Chat, S::Palette], Bind::Custom("repair_step")),
    row!("promote_run", "Run in background",
        "Promote a live interactive run to a durable background job without restarting it.",
        C::Background, S::StatusBar, &[S::StatusBar, S::Home, S::Palette], Bind::Custom("promote_run")),
    row!("resume_run_foreground", "Resume in foreground",
        "Reattach a reconnecting client to a promoted run and resume it in the foreground.",
        C::Background, S::StatusBar, &[S::StatusBar, S::Home, S::Palette], Bind::Custom("resume_run_foreground")),
    row!("pty_input", "Terminal input",
        "Write input bytes to the live terminal process's stdin.",
        C::Terminal, S::Terminal, &[S::Terminal, S::Palette], Bind::Custom("pty_input")),
    row!("pty_resize", "Terminal resize",
        "Record the live terminal process's column and row geometry.",
        C::Terminal, S::Terminal, &[S::Terminal, S::Palette], Bind::Custom("pty_resize")),
    row!("attach_process", "Attach to process",
        "Re-attach to a running process and replay its buffered output into the terminal.",
        C::Terminal, S::Terminal, &[S::Terminal, S::Palette], Bind::Custom("attach_process")),
    row!("stop_process", "Stop process",
        "Stop a running process: terminate its group, then kill it after a short grace.",
        C::Terminal, S::Terminal, &[S::Terminal, S::Palette], Bind::Custom("stop_process")),
    row!("capture_process_artifact", "Capture process output",
        "Preserve a process's captured output as a durable artifact in the blob store.",
        C::Terminal, S::Terminal, &[S::Terminal, S::Palette], Bind::Custom("capture_process_artifact")),
    row!("export_review_receipt", "Export review receipt",
        "Seal a diff's hunks and their verification receipts into a durable review receipt.",
        C::Diff, S::DiffReview, &[S::DiffReview, S::Ide, S::Palette], Bind::Custom("export_review_receipt")),
    row!("new_session", "New session",
        "Start a fresh session thread from the launcher or the New-chat menu.",
        C::SideChat, S::Home, &[S::Home, S::Chat, S::Palette], Bind::Custom("new_session")),
    row!("revert_diff", "Revert diff",
        "Revert a whole diff on disk once its hunks are already decided.",
        C::Diff, S::DiffReview, &[S::DiffReview, S::Ide, S::Palette], Bind::Custom("revert_diff")),
    row!("save_file", "Save file",
        "Write the open editor buffer to disk through the permission-gated applier.",
        C::File, S::Editor, &[S::Editor, S::Ide], Bind::Custom("save_file")),
    row!("create_worktree", "Create worktree",
        "Request an isolated git worktree on a fresh branch (the host holds it at a gate).",
        C::Workspace, S::Home, &[S::Home, S::Palette], Bind::Custom("create_worktree")),
    row!("open_session", "Open session",
        "Reopen a recorded session and republish its transcript.",
        C::SideChat, S::Home, &[S::Home, S::Palette], Bind::Custom("open_session")),
    row!("approve_gate", "Approve held command",
        "Release a command the security gate is holding so it runs.",
        C::Terminal, S::Chat, &[S::Chat, S::Terminal, S::Palette], Bind::Custom("approve_gate")),
    row!("deny_gate", "Deny held command",
        "Drop a command the security gate is holding so it never runs.",
        C::Terminal, S::Chat, &[S::Chat, S::Terminal, S::Palette], Bind::Custom("deny_gate")),
    row!("approve_effect", "Approve effectful step",
        "Let a paused effectful step in the running turn proceed.",
        C::Plan, S::Chat, &[S::Chat, S::StateTimeline, S::Palette], Bind::Custom("approve_effect")),
    row!("deny_effect", "Deny effectful step",
        "Skip a paused effectful step in the running turn.",
        C::Plan, S::Chat, &[S::Chat, S::StateTimeline, S::Palette], Bind::Custom("deny_effect")),
    row!("grant_write_lease", "Grant write lease",
        "Let this task edit files inside a declared, trusted scope without asking per file.",
        C::Workspace, S::StatusBar, &[S::StatusBar, S::Home, S::Palette], Bind::Custom("grant_write_lease")),
    row!("revoke_write_lease", "Revoke write lease",
        "End the active write lease so workspace writes ask for approval again.",
        C::Workspace, S::StatusBar, &[S::StatusBar, S::Palette], Bind::Custom("revoke_write_lease")),
    row!("switch_surface", "Switch surface",
        "Change the active lens (YOU, CHAT, or IDE) on the shared session. Does not mint a new session or move capability.",
        C::Surface, S::You, &[S::You, S::Chat, S::Ide, S::Home, S::Palette], Bind::Custom("switch_surface")),
    row!("handoff_create", "Create handoff",
        "Seal a typed claim-only capsule from the active surface to another. Never transports authority.",
        C::Handoff, S::You, &[S::You, S::Chat, S::Ide, S::Palette], Bind::Custom("handoff_create")),
    row!("handoff_receive", "Receive handoff",
        "Open a sealed capsule into its target lens on the same session. Receiver capability is unchanged.",
        C::Handoff, S::Chat, &[S::You, S::Chat, S::Ide, S::Palette], Bind::Custom("handoff_receive")),
    ]
};

/// The canonical command table: the ONE registry every surface resolves from.
///
/// Ordering is stable (declaration order) so the serialized golden is byte
/// stable. Seeded from the ranked census priority (verify, side chat,
/// checkpoints, memory, goals, steer, workspace trust) plus environment switch,
/// transcript search, and the already-working core intents.
pub fn command_catalog() -> Vec<CommandSpec> {
    ROWS.iter()
        .map(|row| {
            let mut spec = CommandSpec {
                id: row.id.to_string(),
                title: row.title.to_string(),
                description: row.description.to_string(),
                category: row.category,
                primary_surface: row.primary,
                available_surfaces: row.surfaces.to_vec(),
                required_selection: RequiredSelection::None,
                required_capabilities: Vec::new(),
                effects: Vec::new(),
                approval_policy: ApprovalPolicy::Auto,
                keyboard_shortcut: None,
                command_palette: true,
                context_menu: false,
                toolbar_binding: None,
                backend_binding: match row.bind {
                    Bind::Intent(t) => BackendBinding::Intent(t.to_string()),
                    Bind::Custom(t) => BackendBinding::Custom(t.to_string()),
                },
                undo_strategy: UndoStrategy::None,
                receipt_kind: None,
                telemetry: None,
            };
            // Explicit non-default security/effect/undo/selection/capability/
            // shortcut/policy authority (pinned by security_policies_match_s4_matrix).

            use ApprovalPolicy::*;
            use Effect as E;
            use RequiredSelection as Sel;
            use UndoStrategy as U;

            match spec.id.as_str() {
                // -- shortcuts / chrome only -----------------------------------------
                "submit_turn" => {
                    // BC-HIDE_SESSION-015: approval stays Auto (default).
                    spec.keyboard_shortcut = Some("Mod+Enter".into());
                    spec.toolbar_binding = Some("composer.send".into());
                    spec.telemetry = Some("turn.submit".into());
                }
                "cancel_run" => {
                    spec.keyboard_shortcut = Some("Mod+.".into());
                }
                // -- diffs -----------------------------------------------------------
                "accept_diff" => {
                    spec.keyboard_shortcut = Some("Mod+Enter".into());
                    spec.command_palette = false;
                    spec.context_menu = true;
                    spec.effects = vec![E::WriteFs];
                    spec.undo_strategy = U::Reject;
                    spec.receipt_kind = Some("patch".into());
                    spec.required_selection = Sel::Hunk;
                }
                // Reject writes (inverse hunk restore); not Ask (policy follows effect gate).
                "reject_diff" => {
                    spec.keyboard_shortcut = Some("Mod+Backspace".into());
                    spec.command_palette = false;
                    spec.context_menu = true;
                    spec.required_selection = Sel::Hunk;
                    spec.effects = vec![E::WriteFs, E::State];
                    spec.undo_strategy = U::Inverse;
                }
                "fork_session" => {
                    spec.effects = vec![E::State];
                    spec.required_capabilities = caps(&["state"]);
                }
                "open_file" => {
                    spec.context_menu = true;
                    spec.effects = vec![E::ReadFs];
                    spec.required_selection = Sel::File;
                }
                // RequireSandbox exactly on run_command.
                "run_command" => {
                    spec.approval_policy = RequireSandbox;
                    spec.effects = vec![E::Shell, E::Process];
                }
                "run_static_analysis" => {
                    spec.effects = vec![E::ReadFs, E::Process];
                    spec.receipt_kind = Some("verification_receipt".into());
                }
                "create_side_chat" => {
                    spec.keyboard_shortcut = Some("Mod+Shift+N".into());
                    spec.toolbar_binding = Some("chat.new".into());
                    spec.effects = vec![E::State];
                    spec.required_capabilities = caps(&["subscriptions"]);
                }
                "merge_side_chat" | "goal_set" | "goal_clear" | "reorder_plan" => {
                    spec.effects = vec![E::State];
                    spec.undo_strategy = U::Inverse;
                }
                "checkpoint_create" => {
                    spec.effects = vec![E::State];
                    spec.required_capabilities = caps(&["checkpoints", "state"]);
                    spec.receipt_kind = Some("checkpoint".into());
                }
                // Ask exactly on checkpoint_restore.
                "checkpoint_restore" => {
                    spec.approval_policy = Ask;
                    spec.effects = vec![E::State];
                    spec.required_capabilities = caps(&["checkpoints", "state"]);
                    spec.undo_strategy = U::Checkpoint;
                }
                "memory_add" => {
                    spec.context_menu = true;
                    spec.effects = vec![E::State];
                    spec.undo_strategy = U::Inverse;
                    spec.required_selection = Sel::Text;
                }
                "memory_supersede" => {
                    spec.effects = vec![E::State];
                    spec.undo_strategy = U::Inverse;
                    spec.required_selection = Sel::Text;
                }
                "memory_record_outcome"
                | "approve_plan"
                | "promote_run"
                | "resume_run_foreground"
                | "new_session"
                | "open_session"
                | "revoke_write_lease" => {
                    spec.effects = vec![E::State];
                }
                "memory_revalidate" => {
                    spec.effects = vec![E::ReadFs, E::State];
                }
                "goal_evaluate" => {
                    spec.effects = vec![E::Process];
                    spec.receipt_kind = Some("verification_receipt".into());
                }
                "steer" => {
                    spec.keyboard_shortcut = Some("Mod+/".into());
                    spec.toolbar_binding = Some("composer.steer".into());
                    spec.required_capabilities = caps(&["streaming"]);
                    spec.required_selection = Sel::Text;
                }
                // Ask exactly on workspace_set_repo_trust.
                "workspace_set_repo_trust" => {
                    spec.approval_policy = Ask;
                    spec.effects = vec![E::State];
                    spec.undo_strategy = U::Inverse;
                }
                "environment_switch" => {
                    spec.effects = vec![E::Environment];
                    spec.undo_strategy = U::Inverse;
                }
                "run_search" => {
                    spec.effects = vec![E::ReadFs];
                }
                // Ask exactly on checkpoint_rewind.
                "checkpoint_rewind" => {
                    spec.approval_policy = Ask;
                    spec.effects = vec![E::State, E::WriteFs];
                    spec.required_capabilities = caps(&["checkpoints", "state"]);
                    spec.undo_strategy = U::Checkpoint;
                }
                "checkpoint_replay" | "checkpoint_fork" => {
                    spec.effects = vec![E::State];
                    spec.required_capabilities = caps(&["checkpoints", "state"]);
                }
                "checkpoint_compare" | "checkpoint_inspect" => {
                    spec.effects = vec![E::ReadFs];
                    spec.required_capabilities = caps(&["checkpoints"]);
                }
                "edit_plan_step" | "skip_step" => {
                    spec.effects = vec![E::State];
                    spec.undo_strategy = U::Inverse;
                    spec.required_selection = Sel::PlanStep;
                }
                "repair_step" => {
                    spec.effects = vec![E::State];
                    spec.required_selection = Sel::PlanStep;
                }
                "pty_input" | "attach_process" | "stop_process" => {
                    spec.effects = vec![E::Process];
                }
                "capture_process_artifact" => {
                    spec.effects = vec![E::Process, E::State];
                    spec.receipt_kind = Some("artifact".into());
                }
                "export_review_receipt" => {
                    spec.effects = vec![E::State];
                    spec.receipt_kind = Some("diff_review_receipt".into());
                }
                // Ask exactly on revert_diff.
                "revert_diff" => {
                    spec.approval_policy = Ask;
                    spec.effects = vec![E::WriteFs, E::State];
                    spec.undo_strategy = U::Inverse;
                }
                "save_file" => {
                    spec.keyboard_shortcut = Some("Mod+S".into());
                    spec.command_palette = false;
                    spec.required_selection = Sel::File;
                    spec.effects = vec![E::WriteFs];
                }
                // Ask exactly on create_worktree.
                "create_worktree" => {
                    spec.approval_policy = Ask;
                    spec.effects = vec![E::Vcs, E::Process, E::WriteFs];
                }
                "approve_gate" | "deny_gate" | "approve_effect" | "deny_effect" => {
                    spec.effects = vec![E::Approval];
                }
                // Ask exactly on grant_write_lease (no WriteFs on the grant itself).
                "grant_write_lease" => {
                    spec.approval_policy = Ask;
                    spec.effects = vec![E::Approval, E::State];
                }
                "switch_surface" => {
                    spec.effects = vec![E::State];
                    spec.keyboard_shortcut = Some("Mod+Shift+U".into());
                    spec.toolbar_binding = Some("surface.you".into());
                    spec.telemetry = Some("surface.switch".into());
                }
                "handoff_create" => {
                    spec.effects = vec![E::State];
                    spec.telemetry = Some("handoff.create".into());
                }
                "handoff_receive" => {
                    spec.effects = vec![E::State];
                    spec.telemetry = Some("handoff.receive".into());
                }
                // Pure defaults: pause_run, resume_run, pty_resize.
                _ => {}
            }
            spec
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Method;
    use std::collections::BTreeSet;
    #[test]
    fn catalog_is_non_empty_with_unique_ids() {
        let catalog = command_catalog();
        assert!(!catalog.is_empty(), "the catalog must not be empty");
        let mut ids = BTreeSet::new();
        for spec in &catalog {
            assert!(
                ids.insert(spec.id.as_str()),
                "duplicate command id: {}",
                spec.id
            );
        }
    }
    #[test]
    fn every_command_has_a_shortcut_or_lives_in_the_palette() {
        for spec in command_catalog() {
            assert!(spec.keyboard_shortcut.is_some() || spec.command_palette);
        }
    }
    #[test]
    fn backend_bindings_resolve_to_real_targets() {
        let intents: BTreeSet<&str> = INTENT_NAMES.iter().copied().collect();
        let live_custom: BTreeSet<&str> = WIRE_CUSTOM_NAMES.iter().copied().collect();
        let host_caps: BTreeSet<&str> = HOST_CAPABILITIES.iter().copied().collect();
        let methods: BTreeSet<&str> = Method::ALL.iter().map(|m| m.as_str()).collect();
        for spec in command_catalog() {
            match &spec.backend_binding {
                BackendBinding::Intent(name) => assert!(
                    intents.contains(name.as_str()),
                    "{}: Intent({name}) is not a real api.rs Intent",
                    spec.id
                ),
                BackendBinding::Custom(name) => assert!(
                    live_custom.contains(name.as_str()),
                    "{}: Custom({name}) is not a live wire.ts custom name",
                    spec.id
                ),
                BackendBinding::Rpc(name) => assert!(
                    methods.contains(name.as_str()) || host_caps.contains(name.as_str()),
                    "{}: Rpc({name}) is neither a real Method nor a census host capability",
                    spec.id
                ),
                BackendBinding::LocalOnly => {}
            }
        }
    }
    #[test]
    fn every_live_custom_name_has_a_command() {
        let bound: BTreeSet<String> = command_catalog()
            .into_iter()
            .filter_map(|s| match s.backend_binding {
                BackendBinding::Custom(name) => Some(name),
                _ => None,
            })
            .collect();
        for name in WIRE_CUSTOM_NAMES {
            assert!(
                bound.contains(*name),
                "live custom name with no CommandSpec: {name}"
            );
        }
    }
    #[test]
    fn wire_custom_names_are_canonical_and_unique() {
        let names: BTreeSet<&str> = WIRE_CUSTOM_NAMES.iter().copied().collect();
        assert_eq!(names.len(), WIRE_CUSTOM_NAMES.len());
    }
    #[test]
    fn every_writing_command_declares_the_write_and_its_undo() {
        let writers = [
            ("accept_diff", true),
            ("reject_diff", true),
            ("revert_diff", true),
            ("checkpoint_rewind", true),
            ("create_worktree", false),
            ("save_file", false),
        ];
        let catalog = command_catalog();
        for (id, undoable) in writers {
            let spec = catalog
                .iter()
                .find(|s| s.id == id)
                .unwrap_or_else(|| panic!("{id} is missing from the catalog"));
            assert!(spec.effects.contains(&Effect::WriteFs));
            assert_eq!(spec.undo_strategy != UndoStrategy::None, undoable);
        }
    }
    #[test]
    fn catalog_covers_the_seven_priority_domains() {
        let categories: BTreeSet<Category> = command_catalog().iter().map(|s| s.category).collect();
        for required in [
            Category::Verify,
            Category::SideChat,
            Category::Checkpoint,
            Category::Memory,
            Category::Goal,
            Category::Steer,
            Category::Workspace,
        ] {
            assert!(
                categories.contains(&required),
                "catalog is missing priority domain: {required:?}"
            );
        }
    }
    #[test]
    fn specs_round_trip_through_serde_json() {
        for spec in command_catalog() {
            let json = serde_json::to_string(&spec).expect("serialize");
            let back: CommandSpec = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, spec, "a command spec must survive a serde round trip");
        }
    }
    #[test]
    fn catalog_data_carries_no_en_or_em_dashes() {
        let json = serde_json::to_string(&command_catalog()).unwrap();
        assert!(!json.contains('\u{2013}') && !json.contains('\u{2014}'));
    }

    /// Complete Ask / RequireSandbox matrix. Any unlisted elevated policy fails.
    #[test]
    fn security_policies_match_s4_matrix() {
        const ASK: &[&str] = &[
            "checkpoint_restore",
            "workspace_set_repo_trust",
            "checkpoint_rewind",
            "revert_diff",
            "create_worktree",
            "grant_write_lease",
        ];
        const SANDBOX: &[&str] = &["run_command"];
        let ask: BTreeSet<&str> = ASK.iter().copied().collect();
        let sandbox: BTreeSet<&str> = SANDBOX.iter().copied().collect();
        let catalog = command_catalog();
        for id in ASK {
            let spec = catalog
                .iter()
                .find(|s| s.id == *id)
                .unwrap_or_else(|| panic!("missing {id}"));
            assert_eq!(spec.approval_policy, ApprovalPolicy::Ask, "{id}");
        }
        for id in SANDBOX {
            let spec = catalog
                .iter()
                .find(|s| s.id == *id)
                .unwrap_or_else(|| panic!("missing {id}"));
            assert_eq!(spec.approval_policy, ApprovalPolicy::RequireSandbox, "{id}");
        }
        for spec in &catalog {
            match spec.approval_policy {
                ApprovalPolicy::Ask => assert!(
                    ask.contains(spec.id.as_str()),
                    "unlisted Ask policy on {}",
                    spec.id
                ),
                ApprovalPolicy::RequireSandbox => assert!(
                    sandbox.contains(spec.id.as_str()),
                    "unlisted RequireSandbox policy on {}",
                    spec.id
                ),
                ApprovalPolicy::Auto => {}
                other => panic!(
                    "unexpected elevated policy {other:?} on {}; only Ask/RequireSandbox/Auto allowed",
                    spec.id
                ),
            }
        }
        // submit_turn remains Auto (BC-HIDE_SESSION-015).
        let submit = catalog.iter().find(|s| s.id == "submit_turn").unwrap();
        assert_eq!(submit.approval_policy, ApprovalPolicy::Auto);
        // reject_diff remains WriteFs+State with inverse undo (not Ask).
        let reject = catalog.iter().find(|s| s.id == "reject_diff").unwrap();
        assert!(reject.effects.contains(&Effect::WriteFs));
        assert!(reject.effects.contains(&Effect::State));
        assert_eq!(reject.undo_strategy, UndoStrategy::Inverse);
        assert_eq!(reject.approval_policy, ApprovalPolicy::Auto);
    }
}

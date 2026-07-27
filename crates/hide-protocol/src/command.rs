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
///   (`crates/hide-core/src/api.rs`); the shipped FE already posts these to
///   `/v1/hide/intent`.
/// - [`BackendBinding::Custom`] names an `Intent::Custom{name}` the host already
///   handles AND `wire.ts` `CUSTOM_NAMES` exposes, mirrored here as
///   [`WIRE_CUSTOM_NAMES`]. There is no pending tier: a name the host does not
///   handle is not on the contract at all.
/// - [`BackendBinding::Rpc`] names an elevated capability: either a real
///   [`Method`](crate::protocol::Method) string or a census-confirmed host
///   capability in [`HOST_CAPABILITIES`].
/// - [`BackendBinding::LocalOnly`] is a pure FE action with no backend call.
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

/// The `Intent` variant tags the shipped FE already posts to `/v1/hide/intent`.
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

/// The live custom names, an EXACT mirror of `app/src/wire.ts` `CUSTOM_NAMES`
/// (the `wire_custom_names_mirror_wire_ts` test reads that file and compares, so
/// this cannot silently go stale again). Every `Custom` binding must be in here,
/// and every name in here has an arm in `crates/hide-backend/src/host.rs`
/// `HANDLED_CUSTOM_NAMES` (asserted there).
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
    // The sealed diff review receipt over a diff the app itself produced.
    "export_review_receipt",
];

/// Census-confirmed host capabilities reachable over the elevated protocol that
/// are NOT also a [`Method`](crate::protocol::Method) string. An `Rpc` binding
/// must be a real `Method` OR one of these. No command binds `Rpc` at all any
/// more: `run_static_analysis` was the last bound to one of THESE and is now
/// `Custom`, dispatched over `/v1/hide/intent` by `handle_static_analysis_intent`,
/// and `goal_get` (the last `Rpc` row of any kind) is retired because this
/// frontend has no `/rpc` client to dispatch it with. This list stays as the
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

/// A spec with lazy defaults; each command overrides only the fields that differ
/// via struct-update syntax. Defaults: palette-visible, auto-approved, no
/// selection, no effects, nothing to undo.
fn base(
    id: &str,
    title: &str,
    description: &str,
    category: Category,
    primary_surface: Surface,
    available_surfaces: Vec<Surface>,
    backend_binding: BackendBinding,
) -> CommandSpec {
    CommandSpec {
        id: id.to_string(),
        title: title.to_string(),
        description: description.to_string(),
        category,
        primary_surface,
        available_surfaces,
        required_selection: RequiredSelection::None,
        required_capabilities: Vec::new(),
        effects: Vec::new(),
        approval_policy: ApprovalPolicy::Auto,
        keyboard_shortcut: None,
        command_palette: true,
        context_menu: false,
        toolbar_binding: None,
        backend_binding,
        undo_strategy: UndoStrategy::None,
        receipt_kind: None,
        telemetry: None,
    }
}

fn caps(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

/// The canonical command table: the ONE registry every surface resolves from.
///
/// Ordering is stable (declaration order) so the serialized golden is byte
/// stable. Seeded from the ranked census priority (verify, side chat,
/// checkpoints, memory, goals, steer, workspace trust) plus environment switch,
/// transcript search, and the already-working core intents.
pub fn command_catalog() -> Vec<CommandSpec> {
    use ApprovalPolicy::*;
    use BackendBinding as B;
    use Category as C;
    use RequiredSelection as Sel;
    use Surface as S;
    use UndoStrategy as U;

    vec![
        // -- core turn control (already working: real Intent variants) --------
        CommandSpec {
            keyboard_shortcut: Some("Mod+Enter".into()),
            toolbar_binding: Some("composer.send".into()),
            telemetry: Some("turn.submit".into()),
            ..base(
                "submit_turn",
                "Send message",
                "Submit the composer text as a new turn.",
                C::Turn,
                S::Chat,
                vec![S::Chat, S::Home, S::Palette],
                B::Intent("submit_turn".into()),
            )
        },
        CommandSpec {
            keyboard_shortcut: Some("Mod+.".into()),
            ..base(
                "cancel_run",
                "Cancel run",
                "Cancel the running turn.",
                C::Turn,
                S::Chat,
                vec![S::Chat, S::StatusBar, S::Palette],
                B::Intent("cancel_run".into()),
            )
        },
        base(
            "pause_run",
            "Pause run",
            "Pause the running turn.",
            C::Turn,
            S::Chat,
            vec![S::Chat, S::Palette],
            B::Intent("pause_run".into()),
        ),
        base(
            "resume_run",
            "Resume run",
            "Resume a paused turn.",
            C::Turn,
            S::Chat,
            vec![S::Chat, S::Palette],
            B::Intent("resume_run".into()),
        ),
        // -- diff review (already working) -----------------------------------
        CommandSpec {
            keyboard_shortcut: Some("Mod+Enter".into()),
            command_palette: false,
            context_menu: true,
            effects: vec![Effect::WriteFs],
            undo_strategy: U::Reject,
            receipt_kind: Some("patch".into()),
            required_selection: Sel::Hunk,
            ..base(
                "accept_diff",
                "Accept diff",
                "Apply a proposed diff hunk to the working tree.",
                C::Diff,
                S::DiffReview,
                vec![S::DiffReview, S::Editor, S::Ide],
                B::Intent("accept_diff".into()),
            )
        },
        // Rejecting WRITES: it inverse-writes the hunk back to the working tree, so the row says
        // so. With no `hunk_id` the host reads it as the whole diff, which is the same effect
        // `revert_diff` declares `Ask` for, and `effect_command` holds it at the same gate: the
        // policy follows the effect, not the wire name that carried it.
        CommandSpec {
            keyboard_shortcut: Some("Mod+Backspace".into()),
            command_palette: false,
            context_menu: true,
            required_selection: Sel::Hunk,
            effects: vec![Effect::WriteFs, Effect::State],
            undo_strategy: U::Inverse,
            ..base(
                "reject_diff",
                "Reject diff",
                "Reject a proposed hunk, restoring that file on disk.",
                C::Diff,
                S::DiffReview,
                vec![S::DiffReview, S::Editor, S::Ide],
                B::Intent("reject_diff".into()),
            )
        },
        // -- timeline (already working) --------------------------------------
        CommandSpec {
            effects: vec![Effect::State],
            required_capabilities: caps(&["state"]),
            ..base(
                "fork_session",
                "Fork from here",
                "Fork a new session from the selected timeline event.",
                C::Timeline,
                S::StateTimeline,
                vec![S::StateTimeline, S::Palette],
                B::Intent("fork_session".into()),
            )
        },
        // -- editor / terminal (already working) -----------------------------
        // The `Mod+P` this used to declare was bound NOWHERE: a chord carries no
        // path, so `required_selection: File` keeps `open_file` out of every
        // derived key map, and Mod+P is the palette's own chord (store.ts
        // SHELL_COMMANDS `toggle.palette`). A catalog that advertises a chord
        // nothing binds is the drift this registry exists to remove, so it is
        // dropped rather than faked. Open a file from the explorer, a diff chip,
        // or the palette's file search.
        CommandSpec {
            context_menu: true,
            effects: vec![Effect::ReadFs],
            required_selection: Sel::File,
            ..base(
                "open_file",
                "Open file",
                "Open a file in the editor.",
                C::File,
                S::Ide,
                vec![S::Ide, S::Editor, S::Palette],
                B::Intent("open_file".into()),
            )
        },
        CommandSpec {
            approval_policy: RequireSandbox,
            effects: vec![Effect::Shell, Effect::Process],
            ..base(
                "run_command",
                "Run command",
                "Run a shell command in the terminal.",
                C::Terminal,
                S::Terminal,
                vec![S::Terminal, S::Palette],
                B::Intent("run_command".into()),
            )
        },
        // -- verify -> StatusBar Problems (census priority 1) ----------------
        CommandSpec {
            effects: vec![Effect::ReadFs, Effect::Process],
            receipt_kind: Some("verification_receipt".into()),
            ..base(
                "run_static_analysis",
                "Run static analysis",
                "Run Tier-1 deterministic static analysis and show real problem counts.",
                C::Verify,
                S::StatusBar,
                vec![S::StatusBar, S::ContextStack, S::Palette],
                B::Custom("run_static_analysis".into()),
            )
        },
        // -- side chat -> dead New-chat buttons (census priority 2) ----------
        CommandSpec {
            keyboard_shortcut: Some("Mod+Shift+N".into()),
            toolbar_binding: Some("chat.new".into()),
            effects: vec![Effect::State],
            required_capabilities: caps(&["subscriptions"]),
            ..base(
                "create_side_chat",
                "New side chat",
                "Open a side chat thread that can later merge back.",
                C::SideChat,
                S::Chat,
                vec![S::Chat, S::Palette],
                B::Custom("create_side_chat".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::State],
            undo_strategy: U::Inverse,
            ..base(
                "merge_side_chat",
                "Merge side chat",
                "Merge a side chat's summary back into the main thread.",
                C::SideChat,
                S::Chat,
                vec![S::Chat, S::Palette],
                B::Custom("merge_side_chat".into()),
            )
        },
        // -- checkpoints -> StateTimeline (census priority 3) ----------------
        CommandSpec {
            effects: vec![Effect::State],
            required_capabilities: caps(&["checkpoints", "state"]),
            receipt_kind: Some("checkpoint".into()),
            ..base(
                "checkpoint_create",
                "Create checkpoint",
                "Seal an integrity-verified restore point on the timeline.",
                C::Checkpoint,
                S::StateTimeline,
                vec![S::StateTimeline, S::Palette],
                B::Custom("checkpoint_create".into()),
            )
        },
        CommandSpec {
            approval_policy: Ask,
            effects: vec![Effect::State],
            required_capabilities: caps(&["checkpoints", "state"]),
            undo_strategy: U::Checkpoint,
            ..base(
                "checkpoint_restore",
                "Restore checkpoint",
                "Restore the session to a sealed checkpoint.",
                C::Checkpoint,
                S::StateTimeline,
                vec![S::StateTimeline, S::Palette],
                B::Custom("checkpoint_restore".into()),
            )
        },
        // -- memory -> ContextStack Memory stratum (census priority 4) -------
        CommandSpec {
            context_menu: true,
            effects: vec![Effect::State],
            undo_strategy: U::Inverse,
            required_selection: Sel::Text,
            ..base(
                "memory_add",
                "Add memory note",
                "Store a durable outcome-governed note the agent keeps.",
                C::Memory,
                S::ContextStack,
                vec![S::ContextStack, S::Editor, S::Palette],
                B::Custom("memory_add".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::State],
            undo_strategy: U::Inverse,
            required_selection: Sel::Text,
            ..base(
                "memory_supersede",
                "Supersede memory",
                "Replace a stale memory while keeping its history.",
                C::Memory,
                S::ContextStack,
                vec![S::ContextStack, S::Palette],
                B::Custom("memory_supersede".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::State],
            ..base(
                "memory_record_outcome",
                "Record memory outcome",
                "Report that a remembered fact was right or wrong so it self-quarantines.",
                C::Memory,
                S::ContextStack,
                vec![S::ContextStack, S::Palette],
                B::Custom("memory_record_outcome".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::ReadFs, Effect::State],
            ..base(
                "memory_revalidate",
                "Revalidate memory",
                "Re-check a memory's citations against the repo on disk.",
                C::Memory,
                S::ContextStack,
                vec![S::ContextStack, S::Palette],
                B::Custom("memory_revalidate".into()),
            )
        },
        // -- goals -> HomeComposer goal field (census priority 5) ------------
        CommandSpec {
            effects: vec![Effect::State],
            undo_strategy: U::Inverse,
            ..base(
                "goal_set",
                "Set goal",
                "Set a durable goal and acceptance criteria for the session.",
                C::Goal,
                S::Home,
                vec![S::Home, S::Chat, S::Palette],
                B::Custom("goal_set".into()),
            )
        },
        // RETIRED: `goal_get`. It was the catalog's last `Rpc` row, and this frontend speaks
        // `/v1/hide/intent` only, so no surface could dispatch it while it still declared
        // `command_palette: true`: the palette advertised a row it can never run. Same reason
        // `search_transcript` was collapsed. `goal/get` remains a real elevated-protocol Method
        // (rpc.rs Method::GoalGet over BackendHost::goal_get); it is simply not a UI command.
        CommandSpec {
            effects: vec![Effect::State],
            undo_strategy: U::Inverse,
            ..base(
                "goal_clear",
                "Clear goal",
                "Clear the session's goal.",
                C::Goal,
                S::Home,
                vec![S::Home, S::Palette],
                B::Custom("goal_clear".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::Process],
            receipt_kind: Some("verification_receipt".into()),
            ..base(
                "goal_evaluate",
                "Evaluate goal",
                "Run the deterministic acceptance check for the goal.",
                C::Goal,
                S::Home,
                vec![S::Home, S::Chat, S::Palette],
                B::Custom("goal_evaluate".into()),
            )
        },
        // -- steer -> SteerBar (census priority 6, the true end-to-end hole) --
        CommandSpec {
            keyboard_shortcut: Some("Mod+/".into()),
            toolbar_binding: Some("composer.steer".into()),
            required_capabilities: caps(&["streaming"]),
            required_selection: Sel::Text,
            ..base(
                "steer",
                "Steer turn",
                "Redirect the running turn mid-flight via the interrupt hub.",
                C::Steer,
                S::Chat,
                vec![S::Chat, S::Palette],
                B::Custom("redirect_run".into()),
            )
        },
        // -- workspace trust -> Add-folder flow (census priority 7) ----------
        CommandSpec {
            approval_policy: Ask,
            effects: vec![Effect::State],
            undo_strategy: U::Inverse,
            ..base(
                "workspace_set_repo_trust",
                "Set repo trust",
                "Trust a repo so its instructions and policy can activate.",
                C::Workspace,
                S::Home,
                vec![S::Home, S::Settings, S::Palette],
                B::Custom("workspace_set_repo_trust".into()),
            )
        },
        // -- environment switch -> SideBar popover ---------------------------
        CommandSpec {
            effects: vec![Effect::Environment],
            undo_strategy: U::Inverse,
            ..base(
                "environment_switch",
                "Switch environment",
                "Switch the session's dev, prod, or sandbox environment.",
                C::Environment,
                S::Home,
                vec![S::Home, S::StatusBar, S::Settings, S::Palette],
                B::Custom("environment_switch".into()),
            )
        },
        // -- transcript search -> the ONE search box (palette / Explorer field) --
        // COLLAPSED from two ids for one capability: `search_transcript`, bound
        // `Rpc(item/list)` and carrying a `Mod+Shift+F` that nothing could ever
        // register (an Rpc binding is undispatchable from this FE, and a bare
        // chord carries no query), plus this row. The host answers `run_search`,
        // `search` and `search_transcript` on the SAME arm
        // (host.rs handle_search_intent), so one command is the honest count.
        // Search opens with the palette chord (Mod+P) and the query comes from
        // the box; literal + structured filters, semantic stays
        // DEFERRED_MODEL_REQUIRED.
        CommandSpec {
            effects: vec![Effect::ReadFs],
            ..base(
                "run_search",
                "Search transcript",
                "Search the session transcript by literal or structured query, over the intent channel.",
                C::Search,
                S::Palette,
                vec![S::Palette, S::Chat, S::Ide],
                B::Custom("run_search".into()),
            )
        },
        // -- checkpoint rewind / replay / fork / compare / inspect (Trace E) -----
        // Custom names the host already handles (handle_goal_checkpoint_intent);
        // each acts on a sealed checkpoint from the StateTimeline.
        CommandSpec {
            approval_policy: Ask,
            effects: vec![Effect::State, Effect::WriteFs],
            required_capabilities: caps(&["checkpoints", "state"]),
            undo_strategy: U::Checkpoint,
            ..base(
                "checkpoint_rewind",
                "Rewind to checkpoint",
                "Rewind code, conversation, or both to a checkpoint on a fresh child session.",
                C::Checkpoint,
                S::StateTimeline,
                vec![S::StateTimeline, S::Palette],
                B::Custom("checkpoint_rewind".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::State],
            required_capabilities: caps(&["checkpoints", "state"]),
            ..base(
                "checkpoint_replay",
                "Replay from checkpoint",
                "Re-apply the recorded history from a checkpoint forward onto a new lineage.",
                C::Checkpoint,
                S::StateTimeline,
                vec![S::StateTimeline, S::Palette],
                B::Custom("checkpoint_replay".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::State],
            required_capabilities: caps(&["checkpoints", "state"]),
            ..base(
                "checkpoint_fork",
                "Fork from checkpoint",
                "Branch an ephemeral session seeded only with a checkpoint's inherited prefix.",
                C::Checkpoint,
                S::StateTimeline,
                vec![S::StateTimeline, S::Palette],
                B::Custom("checkpoint_fork".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::ReadFs],
            required_capabilities: caps(&["checkpoints"]),
            ..base(
                "checkpoint_compare",
                "Compare checkpoint",
                "Show the file-level code differences against a checkpoint or another session.",
                C::Checkpoint,
                S::StateTimeline,
                vec![S::StateTimeline, S::Palette],
                B::Custom("checkpoint_compare".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::ReadFs],
            required_capabilities: caps(&["checkpoints"]),
            ..base(
                "checkpoint_inspect",
                "Inspect checkpoint",
                "Verify a checkpoint's integrity and coverage.",
                C::Checkpoint,
                S::StateTimeline,
                vec![S::StateTimeline, S::Palette],
                B::Custom("checkpoint_inspect".into()),
            )
        },
        // -- plan step approve / edit / reorder / skip / repair (plan domain) ----
        // The PlanCard gestures on the ContextStack / Chat plan surface. Custom
        // names routed through host.rs handle_plan_intent; they mutate the durable
        // plan record and republish the `plan` projection.
        CommandSpec {
            effects: vec![Effect::State],
            ..base(
                "approve_plan",
                "Approve plan",
                "Approve a plan step, or the whole plan when no step is selected.",
                C::Plan,
                S::ContextStack,
                vec![S::ContextStack, S::Chat, S::Palette],
                B::Custom("approve_plan".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::State],
            undo_strategy: U::Inverse,
            required_selection: Sel::PlanStep,
            ..base(
                "edit_plan_step",
                "Edit plan step",
                "Edit the text of a plan step.",
                C::Plan,
                S::ContextStack,
                vec![S::ContextStack, S::Chat, S::Palette],
                B::Custom("edit_plan_step".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::State],
            undo_strategy: U::Inverse,
            ..base(
                "reorder_plan",
                "Reorder plan",
                "Reorder the plan's steps to a new permutation.",
                C::Plan,
                S::ContextStack,
                vec![S::ContextStack, S::Chat, S::Palette],
                B::Custom("reorder_plan".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::State],
            undo_strategy: U::Inverse,
            required_selection: Sel::PlanStep,
            ..base(
                "skip_step",
                "Skip step",
                "Skip a plan step with a recorded reason.",
                C::Plan,
                S::ContextStack,
                vec![S::ContextStack, S::Chat, S::Palette],
                B::Custom("skip_step".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::State],
            required_selection: Sel::PlanStep,
            ..base(
                "repair_step",
                "Repair step",
                "Re-open a failed plan step so it can be retried.",
                C::Plan,
                S::ContextStack,
                vec![S::ContextStack, S::Chat, S::Palette],
                B::Custom("repair_step".into()),
            )
        },
        // -- background job promotion / foreground resume (Stage 4, Trace G) -----
        // Custom names the host already handles (handle_background_intent). Pause,
        // stop, and fork of a promoted run reuse pause_run / cancel_run /
        // fork_session, which already route by run id.
        CommandSpec {
            effects: vec![Effect::State],
            ..base(
                "promote_run",
                "Run in background",
                "Promote a live interactive run to a durable background job without restarting it.",
                C::Background,
                S::StatusBar,
                vec![S::StatusBar, S::Home, S::Palette],
                B::Custom("promote_run".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::State],
            ..base(
                "resume_run_foreground",
                "Resume in foreground",
                "Reattach a reconnecting client to a promoted run and resume it in the foreground.",
                C::Background,
                S::StatusBar,
                vec![S::StatusBar, S::Home, S::Palette],
                B::Custom("resume_run_foreground".into()),
            )
        },
        // -- terminal input + process control (all wired) ------------------------
        // Custom names (live in wire.ts) the host routes through
        // handle_process_intent: pty_input writes bytes to a live process's stdin,
        // pty_resize records its geometry, and attach_process / stop_process /
        // capture_process_artifact reach the host methods of the same names. Those
        // three had no wire trigger at all, so a client could start a sandboxed
        // process and then not re-attach to it, stop it, or keep its output.
        CommandSpec {
            effects: vec![Effect::Process],
            ..base(
                "pty_input",
                "Terminal input",
                "Write input bytes to the live terminal process's stdin.",
                C::Terminal,
                S::Terminal,
                vec![S::Terminal, S::Palette],
                B::Custom("pty_input".into()),
            )
        },
        base(
            "pty_resize",
            "Terminal resize",
            "Record the live terminal process's column and row geometry.",
            C::Terminal,
            S::Terminal,
            vec![S::Terminal, S::Palette],
            B::Custom("pty_resize".into()),
        ),
        CommandSpec {
            effects: vec![Effect::Process],
            ..base(
                "attach_process",
                "Attach to process",
                "Re-attach to a running process and replay its buffered output into the terminal.",
                C::Terminal,
                S::Terminal,
                vec![S::Terminal, S::Palette],
                B::Custom("attach_process".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::Process],
            ..base(
                "stop_process",
                "Stop process",
                "Stop a running process: terminate its group, then kill it after a short grace.",
                C::Terminal,
                S::Terminal,
                vec![S::Terminal, S::Palette],
                B::Custom("stop_process".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::Process, Effect::State],
            receipt_kind: Some("artifact".into()),
            ..base(
                "capture_process_artifact",
                "Capture process output",
                "Preserve a process's captured output as a durable artifact in the blob store.",
                C::Terminal,
                S::Terminal,
                vec![S::Terminal, S::Palette],
                B::Custom("capture_process_artifact".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::State],
            receipt_kind: Some("diff_review_receipt".into()),
            ..base(
                "export_review_receipt",
                "Export review receipt",
                "Seal a diff's hunks and their verification receipts into a durable review receipt.",
                C::Diff,
                S::DiffReview,
                vec![S::DiffReview, S::Ide, S::Palette],
                B::Custom("export_review_receipt".into()),
            )
        },
        // -- contract reconciliation: host-handled names that had NO CommandSpec ---
        // Each was already dispatched raw by a surface (so it had no palette row and
        // no shortcut parity) and each has a real arm in host.rs `handle_intent`.
        CommandSpec {
            effects: vec![Effect::State],
            ..base(
                "new_session",
                "New session",
                "Start a fresh session thread from the launcher or the New-chat menu.",
                C::SideChat,
                S::Home,
                vec![S::Home, S::Chat, S::Palette],
                B::Custom("new_session".into()),
            )
        },
        CommandSpec {
            approval_policy: Ask,
            effects: vec![Effect::WriteFs, Effect::State],
            undo_strategy: U::Inverse,
            ..base(
                "revert_diff",
                "Revert diff",
                "Revert a whole diff on disk once its hunks are already decided.",
                C::Diff,
                S::DiffReview,
                vec![S::DiffReview, S::Ide, S::Palette],
                B::Custom("revert_diff".into()),
            )
        },
        // RETIRED: `edit_hunk`. It read only `diff_id` + `hunk_id` and routed to the SAME
        // `apply_hunk` as `accept_diff{hunk_id}`, so it was one capability under two ids, and its
        // description promised an edited body the host never received. Accept the hunk instead.
        //
        // The editor save. It used to be a raw `fs.write_file` connector call bound inside Monaco:
        // no catalog row, no keyboard table entry, and the permission refusal thrown away. It is a
        // command now, so it goes through the ONE dispatch spine and a refused write is held at the
        // approval gate like every other refused effect. Not in the palette: the buffer being saved
        // lives in the editor, and a palette gesture carries no buffer (the argument rule).
        CommandSpec {
            keyboard_shortcut: Some("Mod+S".into()),
            command_palette: false,
            required_selection: Sel::File,
            effects: vec![Effect::WriteFs],
            ..base(
                "save_file",
                "Save file",
                "Write the open editor buffer to disk through the permission-gated applier.",
                C::File,
                S::Editor,
                vec![S::Editor, S::Ide],
                B::Custom("save_file".into()),
            )
        },
        // -- contract cleanup: the last eight host-handled names with no spec ----
        // Every one of these already had a REAL arm in host.rs and a REAL gesture in
        // the app, and every one of those gestures built its own `Intent::Custom`
        // because the registry did not carry the command. That is exactly the
        // per-surface binding this registry exists to end, so they are declared here
        // and the surfaces resolve them through `runCommand` like everything else.
        // Argument-carrying rows stay out of the palette by the argument rule
        // (`REQUIRED_ARGS` in app/src/store.ts), not by a second visibility flag.
        CommandSpec {
            approval_policy: Ask,
            effects: vec![Effect::Vcs, Effect::Process, Effect::WriteFs],
            ..base(
                "create_worktree",
                "Create worktree",
                "Request an isolated git worktree on a fresh branch (the host holds it at a gate).",
                C::Workspace,
                S::Home,
                vec![S::Home, S::Palette],
                B::Custom("create_worktree".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::State],
            ..base(
                "open_session",
                "Open session",
                "Reopen a recorded session and republish its transcript.",
                C::SideChat,
                S::Home,
                vec![S::Home, S::Palette],
                B::Custom("open_session".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::Approval],
            ..base(
                "approve_gate",
                "Approve held command",
                "Release a command the security gate is holding so it runs.",
                C::Terminal,
                S::Chat,
                vec![S::Chat, S::Terminal, S::Palette],
                B::Custom("approve_gate".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::Approval],
            ..base(
                "deny_gate",
                "Deny held command",
                "Drop a command the security gate is holding so it never runs.",
                C::Terminal,
                S::Chat,
                vec![S::Chat, S::Terminal, S::Palette],
                B::Custom("deny_gate".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::Approval],
            ..base(
                "approve_effect",
                "Approve effectful step",
                "Let a paused effectful step in the running turn proceed.",
                C::Plan,
                S::Chat,
                vec![S::Chat, S::StateTimeline, S::Palette],
                B::Custom("approve_effect".into()),
            )
        },
        CommandSpec {
            effects: vec![Effect::Approval],
            ..base(
                "deny_effect",
                "Deny effectful step",
                "Skip a paused effectful step in the running turn.",
                C::Plan,
                S::Chat,
                vec![S::Chat, S::StateTimeline, S::Palette],
                B::Custom("deny_effect".into()),
            )
        },
        // -- the task-scoped write lease -------------------------------------
        // The shipped policy asks before every workspace write, which is right for one stray edit
        // and wrong for an approved implementation task: the agent's own edits were refused too, so
        // the diff store stayed empty. The lease is the approval, taken ONCE per task and bounded by
        // a declared scope. It is `Ask` precisely because the grant IS the human decision; it
        // declares no `write_fs` of its own because it performs no write, it widens the policy for
        // the writes the ordinary edit path already declares.
        CommandSpec {
            approval_policy: Ask,
            effects: vec![Effect::Approval, Effect::State],
            ..base(
                "grant_write_lease",
                "Grant write lease",
                "Let this task edit files inside a declared, trusted scope without asking per file.",
                C::Workspace,
                S::StatusBar,
                vec![S::StatusBar, S::Home, S::Palette],
                B::Custom("grant_write_lease".into()),
            )
        },
        // Auto, and it stays Auto: taking permission away may never wait on permission.
        CommandSpec {
            effects: vec![Effect::State],
            ..base(
                "revoke_write_lease",
                "Revoke write lease",
                "End the active write lease so workspace writes ask for approval again.",
                C::Workspace,
                S::StatusBar,
                vec![S::StatusBar, S::Palette],
                B::Custom("revoke_write_lease".into()),
            )
        },
    ]
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

    /// Parity invariant: every command is reachable. No orphan actions: each has
    /// a keyboard shortcut OR is listed in the command palette.
    #[test]
    fn every_command_has_a_shortcut_or_lives_in_the_palette() {
        for spec in command_catalog() {
            assert!(
                spec.keyboard_shortcut.is_some() || spec.command_palette,
                "orphan command (no shortcut and not in the palette): {}",
                spec.id
            );
        }
    }

    /// Backend-binding integrity: nothing is silently invented.
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

    /// The other direction of the same contract: a live custom name with no
    /// `CommandSpec` is a capability the palette, the shortcut map and the SDK
    /// cannot see, so every surface that wants it hand-builds an intent. Eight
    /// names were in exactly that state before the contract-cleanup stage.
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

    /// The mirror must BE a mirror. The previous guard compared two Rust consts
    /// and so passed while [`WIRE_CUSTOM_NAMES`] drifted 17 names behind
    /// `wire.ts`; this one reads the TypeScript contract itself, in order, so
    /// either file changing alone fails here.
    #[test]
    fn wire_custom_names_mirror_wire_ts() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../app/src/wire.ts")
            .canonicalize()
            .expect("app/src/wire.ts must sit at the workspace root");
        let src = std::fs::read_to_string(&path).expect("read wire.ts");
        let body = src
            .split_once("export const CUSTOM_NAMES = [")
            .expect("wire.ts declares CUSTOM_NAMES")
            .1
            .split_once("] as const;")
            .expect("CUSTOM_NAMES is a closed array")
            .0;
        // One quoted name per line; comment lines carry no quotes.
        let names: Vec<&str> = body
            .lines()
            .filter_map(|l| l.trim().strip_prefix('"'))
            .filter_map(|l| l.split_once('"'))
            .map(|(name, _)| name)
            .collect();
        assert_eq!(
            names, WIRE_CUSTOM_NAMES,
            "WIRE_CUSTOM_NAMES has drifted from app/src/wire.ts CUSTOM_NAMES"
        );
    }

    /// Truth in the authority: every row whose host path writes the working tree declares the
    /// write, and says whether that write can be unwound. `reject_diff` declared NO effects and NO
    /// undo while inverse-writing files, which is also what made its `auto` policy next to
    /// `revert_diff`'s `ask` look deliberate rather than a hole.
    #[test]
    fn every_writing_command_declares_the_write_and_its_undo() {
        // (catalog id, whether the host can unwind the write it makes)
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
            assert!(
                spec.effects.contains(&Effect::WriteFs),
                "{id} writes the working tree but does not declare write_fs"
            );
            assert_eq!(
                spec.undo_strategy != UndoStrategy::None,
                undoable,
                "{id}: undo_strategy must say truthfully whether the write can be unwound"
            );
        }
    }

    /// Coverage: the catalog contains at least the seven priority domains.
    #[test]
    fn catalog_covers_the_seven_priority_domains() {
        let categories: BTreeSet<Category> =
            command_catalog().iter().map(|s| s.category).collect();
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

    /// No added string carries an en or em dash (house rule).
    #[test]
    fn catalog_data_carries_no_en_or_em_dashes() {
        let json = serde_json::to_string(&command_catalog()).unwrap();
        assert!(
            !json.contains('\u{2013}') && !json.contains('\u{2014}'),
            "catalog data must use plain hyphens only"
        );
    }
}

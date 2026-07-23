//! Backend composition layer for HIDE — the runnable, headless host (bible
//! ch.01 host/process model + ch.07 Wire-A/Wire-B).
//!
//! This is the non-frontend host boundary. It composes every sibling crate into
//! a [`BackendHost`], and — as of WP-11 — it is a *runnable host*, not just a
//! composition facade:
//!
//! * [`supervisor::RuntimeSupervisor`] spawns + supervises the `hawking serve`
//!   child (state machine + `/healthz` poll + backoff + `runtime.lock`).
//! * [`model_provider::HttpModelProvider`] lets the kernel generate against that
//!   live runtime over HTTP (T5 — no engine-crate link).
//! * [`ui_bus::UiEventBus`] is the push Wire-B (broadcast + render coalescing +
//!   bounded backpressure); the pull `ui_events` API is retained for replay.
//! * [`commands::CommandRouter`] validates + can *reject* intents, and control
//!   intents signal a running run via the [`interrupt::InterruptHub`].
//! * [`replay::BackendReplayService`] adds time-travel (`scrub_to_event` /
//!   `fork_session`).
//! * `hide-fleet` is now a load-bearing dep: [`BackendHost::fleet_run`] schedules
//!   a parallel kernel run via `FleetManager`.
//!
//! ## Deferred seams (documented, not built)
//!
//! * **Tauri transport** — [`commands::CommandRouter::handle`] is kept
//!   transport-agnostic; a future `#[tauri::command]` wraps it behind
//!   `invoke('hide_intent')`. The shell adds no `tauri` dep.
//! * **WASM plugin host** — `hide_core::plugin::ExtensionRegistry` stays the
//!   descriptor registry; the wasmtime component host is post-shell. No
//!   `wasmtime` dep.

pub mod approval;
pub mod commands;
pub mod compat_instructions;
pub mod connectors;
pub mod digest;
pub mod host;
pub mod initialize;
pub mod interrupt;
pub mod live_thread;
pub mod memory;
pub mod model_provider;
pub mod plan_domain;
pub mod policy;
pub mod process;
pub mod program;
pub mod replay;
pub mod rewind;
pub mod rpc;
pub mod security;
pub mod services;
pub mod supervisor;
pub mod tools;
pub mod tq_metadata;
pub mod ui_bus;

pub use approval::{ApprovalDecision, ApprovalHub};
pub use commands::CommandRouter;
pub use compat_instructions::{
    resolve_repo_instructions, resolve_repo_instructions_for_root, CompatInstructionsSource,
    LoadedInstruction, ResolvedInstructions,
};
pub use connectors::{Connector, ConnectorRegistry, ConnectorStatus};
pub use host::{
    BackendHost, BackendStatus, EvidenceLink, SideChatResult, StaticAnalysisReceipt,
};
pub use initialize::{ClientCapabilities, ClientInfo, ConnectionRegistry, InitializeResponse};
pub use interrupt::InterruptHub;
pub use live_thread::{LiveThread, LiveThreadInitGuard, THREAD_PERSISTED_KIND};
pub use memory::{
    CitationResolution, MemoryDraft, MemoryLedger, MemoryRecord, MemoryRevalidation, MemoryScope,
    MemoryStatus, PrivacyClass, RevalidateTarget,
};
pub use model_provider::{GenerateRoute, HttpModelProvider};
pub use policy::{
    derive_policy_decision, tool_declared_effects, PolicyDecision, PolicyDecisionRecord,
};
pub use process::{ProcessState, ProcessStatus, ProcessSupervisor, StartSpec};
pub use program::{
    default_program_limits, HostProgramHandles, ProgramRunError, ProgramRunResult,
};
pub use replay::{BackendReplayService, TranscriptHit, TranscriptQuery};
pub use rewind::{
    CheckpointCoverage, ChangeStatus, FileChange, ForkPoint, GoalRef, ReceiptScope, RewindTarget,
    StateRef,
};
pub use rpc::{ui_event_to_notification, RpcResult};
pub use services::{
    BackendCapabilities, BackendServices, CheckpointRecord, CheckpointStore, ConversationEdge,
    ConversationGraph, ConversationNode, GoalOutcome, GoalRecord, GoalStatus, GoalStore,
    GoalVerdict, SessionRecord, SessionRegistry, SessionRelationship,
};
pub use supervisor::{
    ProcessLauncher, RuntimeChild, RuntimeLauncher, RuntimeSupervisor, SupervisorConfig,
};
pub use ui_bus::UiEventBus;

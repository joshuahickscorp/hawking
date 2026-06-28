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

pub mod commands;
pub mod connectors;
pub mod host;
pub mod interrupt;
pub mod model_provider;
pub mod replay;
pub mod security;
pub mod services;
pub mod supervisor;
pub mod tools;
pub mod ui_bus;

pub use commands::CommandRouter;
pub use connectors::{Connector, ConnectorRegistry, ConnectorStatus};
pub use host::{BackendHost, BackendStatus};
pub use interrupt::InterruptHub;
pub use model_provider::{GenerateRoute, HttpModelProvider};
pub use replay::BackendReplayService;
pub use services::{BackendCapabilities, BackendServices, SessionRegistry};
pub use supervisor::{
    ProcessLauncher, RuntimeChild, RuntimeLauncher, RuntimeSupervisor, SupervisorConfig,
};
pub use ui_bus::UiEventBus;

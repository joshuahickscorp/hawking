//! Backend composition layer for HIDE.
//!
//! This is the non-frontend host boundary: a future Tauri command layer or
//! headless CLI can instantiate these services, register connectors, and route
//! intents without learning each subsystem's internal layout.

pub mod commands;
pub mod connectors;
pub mod host;
pub mod replay;
pub mod security;
pub mod services;
pub mod tools;

pub use connectors::{Connector, ConnectorRegistry, ConnectorStatus};
pub use host::{BackendHost, BackendStatus};
pub use replay::BackendReplayService;
pub use services::{BackendCapabilities, BackendServices};

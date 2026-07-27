//! Shared HIDE contracts.
//!
//! This crate intentionally contains data models, traits, and small in-memory
//! scaffolds only. Runtime-heavy pieces live behind traits so the Tauri host,
//! headless kernel tests, and future CLI can all share the same architecture.

pub mod api;
pub mod config;
pub mod error;
pub mod event;
pub mod ids;
pub mod migration;
pub mod observability;
pub mod permission;
pub mod persistence;
pub mod plugin;
pub mod project;
pub mod runtime;
pub mod security;
pub mod supervision;
pub mod tool;
pub mod types;

pub use error::{HideError, Result};

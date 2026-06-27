//! HIDE living-index scaffold.
//!
//! This crate owns query contracts and an in-memory reference implementation.
//! Persistent SQLite/tree-sitter/LSP/vector backends can slot in behind the
//! same traits without changing agent or tool code.

pub mod daemon;
pub mod graph;
pub mod merkle;
pub mod query;
pub mod semantic;
pub mod store;

pub use query::{CodeIndex, InMemoryCodeIndex, SearchQuery, SearchResult};

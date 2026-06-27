//! Builtin HIDE tools.
//!
//! The dispatcher and permission model live in `hide-core`; this crate provides
//! concrete builtin tool implementations and protocol bridges that can be
//! registered by the backend host.

pub mod fs;
pub mod git;
pub mod mcp;
pub mod registry;
pub mod shell;

pub use registry::register_builtin_tools;

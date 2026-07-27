//! Builtin HIDE tools (bible ch.03).
//!
//! The dispatcher and permission model live in `hide-core`; this crate provides
//! concrete builtin tool implementations and the MCP host/client bridge.
//!
//! Module map:
//! * [`fs`] — read/list/write/stat/glob/watch (§4.6.1), `bytes_ref` spill.
//! * [`edit`] — the tiered verifying applier: search_replace / apply_patch /
//!   write_file (§4.7), with optimistic-concurrency `base_hash`.
//! * [`shell`] — sandboxed `shell.run` with a timeout watchdog (§4.8); `shell.plan`.
//! * [`proc`] — `test.run`/`build.run`/`compile.check`, EXEC_NONZERO-as-data.
//! * [`search`] — `search.text` (ignore-walker + regex).
//! * [`git`] — status/diff/log/commit + the worktree trio (§4.6.6).
//! * [`mcp`] — JSON-RPC 2.0 MCP client over stdio + Streamable HTTP (§4.10).

pub mod common;
pub mod edit;
pub mod fs;
pub mod git;
pub mod mcp;
pub mod memory;
pub mod proc;
pub mod registry;
pub mod search;
pub mod shell;
pub mod spec_helpers;

pub use registry::{register_builtin_tools, register_builtin_tools_with};
pub use shell::ShellConfig;

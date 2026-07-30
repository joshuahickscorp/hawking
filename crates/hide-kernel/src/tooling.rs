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

pub use registry::{register_builtin_tools, register_builtin_tools_with};
pub use shell::ShellConfig;

#[path = "tooling_common.rs"]
pub mod common;
#[path = "tooling_edit.rs"]
pub mod edit;
#[path = "tooling_fs.rs"]
pub mod fs;
#[path = "tooling_git.rs"]
pub mod git;
#[path = "tooling_mcp.rs"]
pub mod mcp;
#[path = "tooling_memory.rs"]
pub mod memory;
#[path = "tooling_proc.rs"]
pub mod proc;
#[path = "tooling_registry.rs"]
pub mod registry;
#[path = "tooling_search.rs"]
pub mod search;
#[path = "tooling_shell.rs"]
pub mod shell;
#[path = "tooling_spec_helpers.rs"]
pub mod spec_helpers;

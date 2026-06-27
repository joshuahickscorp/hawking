//! HIDE context and memory substrate.
//!
//! This is the shell-side compiler described in `docs/hide-bible/04-*`: it
//! ranks sources, packs a token budget, and emits a replayable manifest.

pub mod budget;
pub mod compiler;
pub mod kv;
pub mod manifest;
pub mod memory;
pub mod profiles;
pub mod sources;

pub use compiler::{CompiledContext, ContextCandidate, ContextCompiler, ContextSource};
pub use manifest::{ContextManifest, ContextSpan};

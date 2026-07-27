//! hide-extension-registry: one unified capability registry.
//!
//! HIDE can call many kinds of extension: tools, skills, plugins, hooks, MCP and
//! ACP endpoints, subagents, rules, commands, oracles, browser actions, and
//! integrations. Rather than a separate loader and permission story for each,
//! this crate holds all of them behind one ABI (Bible sec 24). It is entirely
//! model-free: capabilities are declared, indexed, ranked, and enforced by
//! deterministic logic over their manifests, with no network and no inference.
//!
//! The three ideas it exists to enforce:
//!
//! - Progressive disclosure. A planner first sees a compact index of each
//!   capability (id, kind, description, scopes) and pays for a full schema only
//!   when it explicitly loads one. Resolving candidates never materializes a
//!   schema; a monotonic load counter makes that checkable.
//!
//! - Honest effects. A capability may not do more than it declares. If its
//!   network policy, secret policy, or scopes imply an effect that its `effects`
//!   list omits, registration is rejected.
//!
//! - Pinned provenance. An id is unique, can be pinned to a version and a
//!   source commit, and can be revoked. A revoked capability leaves resolution
//!   and cannot be loaded, and its id cannot be silently reused.
//!
//! ```
//! use hide_extension_registry::{
//!     Registry, CapabilityManifest, CapabilityKind, Effect, Scope, ResolveQuery,
//! };
//!
//! let mut reg = Registry::new();
//! let mut m = CapabilityManifest::new("fs.read", "1.0.0", CapabilityKind::Tool, "hide");
//! m.description = "read a file from the repository".to_string();
//! m.scopes = vec![Scope::Repo];
//! m.effects = vec![Effect::Read];
//! reg.register(m).unwrap();
//!
//! let ranked = reg.resolve_for(&ResolveQuery::new().task("read file").kind(CapabilityKind::Tool));
//! assert_eq!(ranked[0].entry.id, "fs.read");
//! assert_eq!(reg.schema_load_count(), 0); // resolution never loaded a schema
//! ```

pub mod builtin_tools;
pub mod error;
pub mod index;
pub mod manifest;
pub mod registry;

pub use builtin_tools::{build_builtin_tool_registry, register_builtin_tools};
pub use error::{RegistryError, Result};
pub use index::CompactEntry;
pub use manifest::{
    CapabilityKind, CapabilityManifest, ContextCost, Effect, FullSchema, NetworkPolicy, Provenance,
    SandboxReq, Scope, SchemaRef, SecretPolicy,
};
pub use registry::{PinSpec, RankedCandidate, Registry, ResolveQuery};

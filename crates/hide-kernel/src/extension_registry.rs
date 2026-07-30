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
//! use hide_kernel::extension_registry::{
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

pub use builtin_tools::{build_builtin_tool_registry, register_builtin_tools};
pub use error::{RegistryError, Result};
pub use index::CompactEntry;
pub use manifest::{
    CapabilityKind, CapabilityManifest, ContextCost, Effect, FullSchema, NetworkPolicy, Provenance,
    SandboxReq, SchemaRef, Scope, SecretPolicy,
};
pub use registry::{PinSpec, RankedCandidate, Registry, ResolveQuery};

pub mod builtin_tools {
    //! Bridge: the `hide-tools` builtin tool catalog as typed capability manifests.
    //!
    //! `hide-tools` owns the concrete builtin tools (fs, edit, shell, proc, search,
    //! git, memory) and the MCP host bridge. Registering the catalog into a
    //! `hide_core::tool::ToolRegistry` yields one `ToolSpec` per tool, and that spec
    //! carries the metadata this module derives effects from:
    //!
    //! * `capabilities_required` - the coarse capability class the tool needs
    //!   (`fs.read`, `fs.write`, `git.read`, `git.write`, `shell.exec`,
    //!   `index.read`, `mcp.call`).
    //! * `annotations` - `read_only`, `destructive`, `open_world`.
    //!
    //! We turn that honest, tool-supplied metadata into one [`CapabilityManifest`]
    //! per tool so the unified registry (Bible sec 24) can rank, disclose, and
    //! effect-account the builtins the same way it does every other capability.
    //! Effects are DERIVED from each spec, never guessed. The full JSON schema of a
    //! tool rides in the manifest's `SchemaRef` payload and is materialized only by
    //! an explicit `load_full_schema`, so the compact index and effect queries stay
    //! schema-free (progressive disclosure).
    //!
    //! Derived effect mapping (see [`derive_effects`]):
    //!
    //! | tool(s)                                                   | effects            |
    //! |-----------------------------------------------------------|--------------------|
    //! | fs.read/list/stat/glob/watch, search.text                 | Read               |
    //! | git.status/diff/log/worktree.list                         | Read               |
    //! | shell.plan (validate only, does not execute)              | Read               |
    //! | fs.write, edit.search_replace/apply_patch/write_file      | Write              |
    //! | git.commit, git.worktree.add, git.worktree.remove         | GitMutation        |
    //! | test.run, build.run, compile.check (bounded exec)         | Execute            |
    //! | shell.run (open-world exec)                                | Execute + Process  |
    //! | memory (view reads, mutating commands write)              | Read + Write       |
    //! | mcp host bridge (reaches an external server)               | Read + Network + ExternalMutation |

    use crate::extension_registry::manifest::{
        CapabilityKind, CapabilityManifest, Effect, NetworkPolicy, Provenance, SandboxReq,
        SchemaRef, Scope, SecretPolicy,
    };
    use crate::extension_registry::registry::Registry;
    use hide_core::tool::{ToolRegistry, ToolSpec};

    /// Every builtin manifest is provided by, and provenanced to, `hide-tools`.
    const PROVIDER: &str = "hide-tools";
    /// The builtins are first-party HIDE code, not an external license.
    const LICENSE: &str = "internal";

    /// Build a fresh [`Registry`] populated with one [`CapabilityManifest`] for
    /// every builtin tool in the `hide-tools` catalog (plus the MCP host bridge).
    pub fn build_builtin_tool_registry() -> Registry {
        let mut reg = Registry::new();
        register_builtin_tools(&mut reg);
        reg
    }

    /// Register the builtin tool catalog into an existing [`Registry`].
    ///
    /// The catalog is the source of truth: `crate::tooling::register_builtin_tools`
    /// populates a `ToolRegistry`, and we bridge each resulting `ToolSpec` into a
    /// typed manifest. Registration goes through [`Registry::register`], so the
    /// registry's honest-effects invariant is enforced on every derived manifest (a
    /// mapping that implied but omitted a network or secret effect would be
    /// rejected here, not silently accepted).
    pub fn register_builtin_tools(reg: &mut Registry) {
        let tool_registry = ToolRegistry::default();
        crate::tooling::register_builtin_tools(&tool_registry);

        for spec in tool_registry.specs() {
            reg.register(manifest_for_spec(&spec))
                .expect("derived builtin tool manifest must satisfy registry invariants");
        }

        // The MCP host is not a static catalog entry: per-server tools are minted
        // dynamically by crate::tooling::mcp::mcp_tool_to_hide_spec. Register a single
        // manifest for the bridge capability itself so a planner can see and
        // effect-account MCP reach-out up front, without a live server.
        reg.register(mcp_bridge_manifest())
            .expect("mcp bridge manifest must satisfy registry invariants");
    }

    /// Project one `hide-tools` `ToolSpec` down to a typed [`CapabilityManifest`].
    fn manifest_for_spec(spec: &ToolSpec) -> CapabilityManifest {
        let effects = derive_effects(spec);
        // A tool that executes or spawns a process must run isolated; everything
        // else (pure reads, file writes, git mutations) runs in-process.
        let needs_sandbox = effects
            .iter()
            .any(|e| matches!(e, Effect::Execute | Effect::Process));

        let version = if spec.version.trim().is_empty() {
            "0.1.0".to_string()
        } else {
            spec.version.clone()
        };

        let mut m =
            CapabilityManifest::new(spec.name.clone(), version, CapabilityKind::Tool, PROVIDER);
        m.description = spec.description.clone();
        m.license = LICENSE.to_string();
        m.effects = effects;
        m.scopes = vec![Scope::Repo];
        m.sandbox = if needs_sandbox {
            SandboxReq::Subprocess
        } else {
            SandboxReq::None
        };
        m.network = NetworkPolicy::Deny;
        m.secrets = SecretPolicy::None;
        // Progressive disclosure: the heavy JSON schema rides in the SchemaRef
        // payload and is parsed only by an explicit load_full_schema. The compact
        // index and the effect accessors never touch it.
        let input_raw = spec.input_schema.to_string();
        m.context_cost.schema_tokens = estimate_tokens(&input_raw, spec.output_schema.as_ref());
        m.input_schema_ref =
            SchemaRef::with_raw(format!("hide-tools://{}/input", spec.name), input_raw);
        m.output_schema_ref = match &spec.output_schema {
            Some(out) => SchemaRef::with_raw(
                format!("hide-tools://{}/output", spec.name),
                out.to_string(),
            ),
            None => SchemaRef::new(format!("hide-tools://{}/output", spec.name)),
        };
        m.provenance = Provenance {
            source: PROVIDER.to_string(),
            commit: None,
            license: LICENSE.to_string(),
        };
        m
    }

    /// Derive the honest effect set for a tool from its spec metadata.
    ///
    /// The rule is keyed on the spec, in this order:
    /// 1. `memory` is dual-mode: `view` reads, the mutating commands write.
    /// 2. a `read_only` tool performs no mutation whatever its capability class
    ///    (this catches shell.plan, which requires `shell.exec` but only validates).
    /// 3. otherwise classify by the coarse capability the tool requires.
    fn derive_effects(spec: &ToolSpec) -> Vec<Effect> {
        if spec.name == "memory" {
            return vec![Effect::Read, Effect::Write];
        }
        if spec.annotations.read_only {
            return vec![Effect::Read];
        }
        let cap = spec
            .capabilities_required
            .first()
            .map(String::as_str)
            .unwrap_or("");
        match cap {
            "fs.write" => vec![Effect::Write],
            "git.write" => vec![Effect::GitMutation],
            "shell.exec" => {
                // open-world exec (shell.run) can spawn and control arbitrary
                // processes; bounded exec (test/build/compile, open_world=false)
                // executes a fixed command without that reach.
                if spec.annotations.open_world {
                    vec![Effect::Execute, Effect::Process]
                } else {
                    vec![Effect::Execute]
                }
            }
            // The MCP proxy reads listings/results, reaches an external endpoint,
            // and the remote tool can mutate an external system.
            "mcp.call" => vec![Effect::Read, Effect::Network, Effect::ExternalMutation],
            // Defensive default: an un-annotated non-read tool is treated as a
            // writer, never silently downgraded to read-only.
            _ => vec![Effect::Write],
        }
    }

    /// The MCP host-bridge capability. Modeled as a `CapabilityKind::Mcp` manifest
    /// derived from the transport semantics in `crate::tooling::mcp`: the client speaks
    /// JSON-RPC over stdio (spawned subprocess) or Streamable HTTP (network), so it
    /// reaches out (Network) to an untrusted external server whose tools can mutate
    /// external state (ExternalMutation). It runs isolated and denies no host by
    /// policy (endpoints are configured per server).
    fn mcp_bridge_manifest() -> CapabilityManifest {
        let mut m = CapabilityManifest::new("mcp", "0.1.0", CapabilityKind::Mcp, PROVIDER);
        m.description =
            "MCP host bridge: JSON-RPC over stdio or Streamable HTTP to external MCP servers"
                .to_string();
        m.license = LICENSE.to_string();
        m.effects = vec![Effect::Read, Effect::Network, Effect::ExternalMutation];
        m.scopes = vec![Scope::Repo, Scope::Network("*".to_string())];
        m.sandbox = SandboxReq::Subprocess;
        // Network is granted (this is the one builtin that reaches out); declaring
        // NetworkPolicy without Effect::Network would be rejected at registration.
        m.network = NetworkPolicy::Any;
        m.secrets = SecretPolicy::None;
        m.provenance = Provenance {
            source: PROVIDER.to_string(),
            commit: None,
            license: LICENSE.to_string(),
        };
        m
    }

    /// A coarse token estimate for a tool's schema, used only to rank load cost.
    fn estimate_tokens(input_raw: &str, output: Option<&serde_json::Value>) -> u32 {
        let out_len = output.map(|o| o.to_string().len()).unwrap_or(0);
        ((input_raw.len() + out_len) / 4) as u32
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::extension_registry::registry::ResolveQuery;
        const CATALOG_IDS: &[&str] = &[
            "fs.read",
            "fs.list",
            "fs.write",
            "fs.stat",
            "fs.glob",
            "fs.watch",
            "edit.search_replace",
            "edit.apply_patch",
            "edit.write_file",
            "shell.run",
            "shell.plan",
            "test.run",
            "build.run",
            "compile.check",
            "search.text",
            "git.status",
            "git.diff",
            "git.log",
            "git.commit",
            "git.worktree.add",
            "git.worktree.remove",
            "git.worktree.list",
            "memory",
        ];
        #[test]
        fn registers_every_builtin_tool() {
            let reg = build_builtin_tool_registry();
            for id in CATALOG_IDS {
                assert!(reg.contains(id), "missing builtin manifest for {id}");
                assert_eq!(reg.kind(id).unwrap(), CapabilityKind::Tool, "kind for {id}");
                let prov = reg.provenance(id).unwrap();
                assert_eq!(prov.source, "hide-tools", "provenance source for {id}");
                assert_eq!(prov.license, "internal", "provenance license for {id}");
            }
            assert_eq!(reg.active_len(), CATALOG_IDS.len() + 1);
        }
        #[test]
        fn fs_read_is_read_only() {
            let reg = build_builtin_tool_registry();
            let effects = reg.declared_effects("fs.read").unwrap();
            assert_eq!(effects, vec![Effect::Read]);
            assert!(!effects.contains(&Effect::Write));
            assert!(!effects.contains(&Effect::Execute));
            assert!(!effects.contains(&Effect::Process));
            assert!(!reg.requires_sandbox("fs.read").unwrap());
        }
        #[test]
        fn edit_tool_declares_write() {
            let reg = build_builtin_tool_registry();
            for id in ["edit.search_replace", "edit.apply_patch", "edit.write_file"] {
                let effects = reg.declared_effects(id).unwrap();
                assert!(effects.contains(&Effect::Write), "{id} must declare Write");
                assert_ne!(effects, vec![Effect::Read], "{id} must not be read-only");
            }
        }
        #[test]
        fn shell_run_is_execute_process_and_sandboxed() {
            let reg = build_builtin_tool_registry();
            let effects = reg.declared_effects("shell.run").unwrap();
            assert!(
                effects.contains(&Effect::Execute),
                "shell.run declares Execute"
            );
            assert!(
                effects.contains(&Effect::Process),
                "shell.run declares Process"
            );
            assert!(
                reg.requires_sandbox("shell.run").unwrap(),
                "shell.run must require sandbox isolation"
            );
        }
        #[test]
        fn bounded_exec_is_execute_and_sandboxed_but_not_process() {
            let reg = build_builtin_tool_registry();
            for id in ["test.run", "build.run", "compile.check"] {
                let effects = reg.declared_effects(id).unwrap();
                assert!(effects.contains(&Effect::Execute), "{id} declares Execute");
                assert!(
                    !effects.contains(&Effect::Process),
                    "{id} is not open-world"
                );
                assert!(reg.requires_sandbox(id).unwrap(), "{id} must be sandboxed");
            }
        }
        #[test]
        fn git_mutation_tool_declares_git_mutation() {
            let reg = build_builtin_tool_registry();
            for id in ["git.commit", "git.worktree.add", "git.worktree.remove"] {
                let effects = reg.declared_effects(id).unwrap();
                assert!(
                    effects.contains(&Effect::GitMutation),
                    "{id} must declare GitMutation"
                );
                assert!(!reg.requires_sandbox(id).unwrap(), "{id} sandbox");
            }
        }
        #[test]
        fn shell_plan_and_git_reads_are_read_only() {
            let reg = build_builtin_tool_registry();
            for id in [
                "shell.plan",
                "git.status",
                "git.diff",
                "git.log",
                "git.worktree.list",
                "search.text",
            ] {
                assert_eq!(
                    reg.declared_effects(id).unwrap(),
                    vec![Effect::Read],
                    "{id}"
                );
                assert!(!reg.requires_sandbox(id).unwrap(), "{id} sandbox");
            }
        }
        #[test]
        fn memory_declares_both_read_and_write() {
            let reg = build_builtin_tool_registry();
            let effects = reg.declared_effects("memory").unwrap();
            assert!(effects.contains(&Effect::Read), "memory reads (view)");
            assert!(
                effects.contains(&Effect::Write),
                "memory writes (mutations)"
            );
        }
        #[test]
        fn no_mutating_tool_is_silently_read_only() {
            let reg = build_builtin_tool_registry();
            let mutators = [
                "fs.write",
                "edit.search_replace",
                "edit.apply_patch",
                "edit.write_file",
                "git.commit",
                "git.worktree.add",
                "git.worktree.remove",
                "shell.run",
                "test.run",
                "build.run",
                "compile.check",
                "memory",
            ];
            for id in mutators {
                let effects = reg.declared_effects(id).unwrap();
                assert_ne!(
                    effects,
                    vec![Effect::Read],
                    "{id} performs mutation but is declared read-only"
                );
                let executes = effects
                    .iter()
                    .any(|e| matches!(e, Effect::Execute | Effect::Process));
                if executes {
                    assert!(
                        reg.requires_sandbox(id).unwrap(),
                        "{id} executes unsandboxed"
                    );
                }
            }
        }
        #[test]
        fn progressive_disclosure_effects_without_schema_load() {
            let reg = build_builtin_tool_registry();
            assert_eq!(reg.schema_load_count(), 0);
            let _ = reg.index();
            let _ = reg.declared_effects("shell.run").unwrap();
            let _ = reg.scope_allows("fs.read", &Scope::Repo).unwrap();
            let _ = reg.resolve_for(
                &ResolveQuery::new()
                    .task("edit file")
                    .kind(CapabilityKind::Tool),
            );
            assert_eq!(
                reg.schema_load_count(),
                0,
                "effect and index queries must stay schema-free"
            );
            for entry in reg.index() {
                if entry.id == "fs.read" {
                    assert_eq!(entry.kind, CapabilityKind::Tool);
                }
            }
            let full = reg.load_full_schema("fs.read").unwrap();
            assert_eq!(reg.schema_load_count(), 1);
            let input = full.input.expect("fs.read carries an input schema");
            assert_eq!(input["properties"]["path"]["type"], "string");
        }
        #[test]
        fn scope_allows_repository() {
            let reg = build_builtin_tool_registry();
            for id in CATALOG_IDS {
                assert!(
                    reg.scope_allows(id, &Scope::Repo).unwrap(),
                    "{id} must be scoped to the repository"
                );
            }
        }
        #[test]
        fn mcp_bridge_declares_network_and_external_mutation() {
            let reg = build_builtin_tool_registry();
            let effects = reg.declared_effects("mcp").unwrap();
            assert!(effects.contains(&Effect::Network), "mcp reaches out");
            assert!(
                effects.contains(&Effect::ExternalMutation),
                "mcp remote tools mutate external state"
            );
            assert_eq!(reg.kind("mcp").unwrap(), CapabilityKind::Mcp);
            assert!(reg.requires_sandbox("mcp").unwrap());
        }
    }
}

pub mod error {
    use thiserror::Error;

    use crate::extension_registry::manifest::Effect;

    /// Errors surfaced by the capability registry. Registration is strict on the
    /// safety-relevant paths (duplicate ids, undeclared effects, pin violations)
    /// because those are the invariants the registry exists to hold.
    #[derive(Debug, Error)]
    pub enum RegistryError {
        #[error("capability id {0:?} is already registered")]
        DuplicateId(String),

        #[error("no capability registered with id {0:?}")]
        NotFound(String),

        #[error("capability {0:?} has been revoked")]
        Revoked(String),

        #[error("capability {id:?} declares effects {missing} that its policies and scopes require but the effects list omits")]
        UndeclaredEffects { id: String, missing: EffectList },

        #[error("capability {id:?} violates pin: {detail}")]
        PinViolation { id: String, detail: String },

        #[error("capability {id:?} is invalid: {detail}")]
        InvalidManifest { id: String, detail: String },

        #[error("schema parse error for capability {id:?} ({which}): {source}")]
        Schema {
            id: String,
            which: &'static str,
            #[source]
            source: serde_json::Error,
        },
    }

    pub type Result<T> = std::result::Result<T, RegistryError>;

    /// A comma-joined list of effect names, used in error messages so the missing
    /// declarations are legible without a custom Display on `Vec<Effect>`.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct EffectList(pub Vec<Effect>);

    impl std::fmt::Display for EffectList {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let names: Vec<&str> = self.0.iter().map(|e| e.as_str()).collect();
            write!(f, "[{}]", names.join(", "))
        }
    }
}

pub mod index {
    //! The compact metadata index.
    //!
    //! Progressive disclosure means a planner first sees only enough to decide
    //! whether a capability is worth loading. That view is a [`CompactEntry`]: id,
    //! kind, description, and scopes, and nothing heavier. In particular it carries
    //! no schema and no raw schema text, so building or returning the index never
    //! materializes a full schema. The full schema is reached only through an
    //! explicit [`hide_kernel::extension_registry::Registry::load_full_schema`] call.

    use crate::extension_registry::manifest::{CapabilityKind, CapabilityManifest, Scope};

    /// The disclosed-up-front view of one capability. Deliberately excludes the
    /// input/output schemas, effects, sandbox, network and secret policies, and
    /// provenance. Those come from the narrow enforcement accessors or from a full
    /// schema load, so the cheap index stays cheap.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct CompactEntry {
        pub id: String,
        pub kind: CapabilityKind,
        pub description: String,
        pub scopes: Vec<Scope>,
    }

    impl CompactEntry {
        /// Project a manifest down to its compact view. This copies only the four
        /// disclosed fields; the schema refs are not touched.
        pub fn from_manifest(m: &CapabilityManifest) -> Self {
            CompactEntry {
                id: m.id.clone(),
                kind: m.kind,
                description: m.description.clone(),
                scopes: m.scopes.clone(),
            }
        }
    }
}

pub mod manifest {
    //! Capability manifest types.
    //!
    //! A [`CapabilityManifest`] is the single declaration a provider files for any
    //! extension that HIDE can call: a tool, a skill, a plugin, a hook, an MCP or
    //! ACP endpoint, a subagent, a rule, a command, an oracle, a browser action, or
    //! an integration. The manifest carries lightweight metadata (indexed and
    //! disclosed up front) and references to heavier input/output schemas that are
    //! only materialized on demand. It also states, up front and checkably, what the
    //! capability is allowed to do: its effects, scopes, sandbox requirement,
    //! network policy, secret policy, and provenance.

    use serde::{Deserialize, Serialize};

    /// The category of an extension. One registry holds all of them so that
    /// resolution, effect accounting, and progressive disclosure work the same way
    /// regardless of how a capability is implemented.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum CapabilityKind {
        Tool,
        Skill,
        Plugin,
        Hook,
        Mcp,
        Acp,
        Subagent,
        Rule,
        Command,
        Oracle,
        Browser,
        Integration,
    }

    impl CapabilityKind {
        pub fn as_str(&self) -> &'static str {
            match self {
                CapabilityKind::Tool => "tool",
                CapabilityKind::Skill => "skill",
                CapabilityKind::Plugin => "plugin",
                CapabilityKind::Hook => "hook",
                CapabilityKind::Mcp => "mcp",
                CapabilityKind::Acp => "acp",
                CapabilityKind::Subagent => "subagent",
                CapabilityKind::Rule => "rule",
                CapabilityKind::Command => "command",
                CapabilityKind::Oracle => "oracle",
                CapabilityKind::Browser => "browser",
                CapabilityKind::Integration => "integration",
            }
        }
    }

    /// A single side effect a capability may produce. Effects are declared, not
    /// inferred at call time, so a planner can reason about a capability before
    /// invoking it. Everything except [`Effect::Read`] counts as elevated for
    /// least-privilege ranking.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum Effect {
        Read,
        Write,
        /// Mutates version-control state (commits, refs, worktrees). A specialized
        /// write whose blast radius is the repository's git history and working
        /// tree layout, not just a single file's bytes.
        GitMutation,
        Execute,
        /// Spawns or controls an operating-system process. Distinct from
        /// [`Effect::Execute`]: a tool can execute one fixed, bounded command
        /// without the open-world reach to spawn and manage arbitrary processes.
        Process,
        Network,
        SecretAccess,
        ExternalMutation,
        Irreversible,
        Privileged,
    }

    impl Effect {
        pub fn as_str(&self) -> &'static str {
            match self {
                Effect::Read => "Read",
                Effect::Write => "Write",
                Effect::GitMutation => "GitMutation",
                Effect::Execute => "Execute",
                Effect::Process => "Process",
                Effect::Network => "Network",
                Effect::SecretAccess => "SecretAccess",
                Effect::ExternalMutation => "ExternalMutation",
                Effect::Irreversible => "Irreversible",
                Effect::Privileged => "Privileged",
            }
        }

        /// Everything above a plain read is elevated. Resolution ranks lower-effect
        /// capabilities first so a task is served by the least-privilege option.
        pub fn is_elevated(&self) -> bool {
            !matches!(self, Effect::Read)
        }
    }

    /// A resource or role scope a capability is bound to. Scopes are matched with
    /// coverage rules (see [`Scope::covers`]) so a broad grant satisfies a narrow
    /// request without listing every leaf.
    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum Scope {
        /// The whole working repository.
        Repo,
        /// A filesystem path prefix (a parent path covers its descendants).
        Filesystem(String),
        /// A network host (`"*"` covers any host).
        Network(String),
        /// A named environment variable (`"*"` covers any).
        Env(String),
        /// A named secret (`"*"` covers any).
        Secret(String),
        /// An operator role this capability is offered to.
        Role(String),
        /// An opaque capability-defined scope, matched by equality.
        Custom(String),
    }

    impl Scope {
        /// Whether a held scope (`self`) covers a requested scope (`want`).
        pub fn covers(&self, want: &Scope) -> bool {
            use Scope::*;
            match (self, want) {
                (Repo, Repo) => true,
                (Filesystem(a), Filesystem(b)) => path_prefix_covers(a, b),
                (Network(a), Network(b)) => a == "*" || a == b,
                (Env(a), Env(b)) => a == "*" || a == b,
                (Secret(a), Secret(b)) => a == "*" || a == b,
                (Role(a), Role(b)) => a == b,
                (Custom(a), Custom(b)) => a == b,
                _ => false,
            }
        }
    }

    /// True when path `a` is a prefix directory of path `b` (component aware).
    fn path_prefix_covers(a: &str, b: &str) -> bool {
        let a = a.trim_end_matches('/');
        let b = b.trim_end_matches('/');
        if a.is_empty() {
            // An empty (root) grant covers everything.
            return true;
        }
        b == a || b.starts_with(&format!("{a}/"))
    }

    /// How isolated a capability must run.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
    pub enum SandboxReq {
        /// Runs in-process with no added isolation.
        #[default]
        None,
        /// Must run in an isolated subprocess.
        Subprocess,
        /// Must run in a container.
        Container,
        /// Must run in a virtual machine.
        Vm,
    }

    impl SandboxReq {
        pub fn requires_isolation(&self) -> bool {
            !matches!(self, SandboxReq::None)
        }
    }

    /// What network access a capability is permitted. Denied by default: a
    /// capability that wants the network has to say so, and saying so implies the
    /// [`Effect::Network`] declaration.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub enum NetworkPolicy {
        /// No network access.
        Deny,
        /// Access limited to the listed hosts.
        AllowList(Vec<String>),
        /// Unrestricted network access.
        Any,
    }

    impl Default for NetworkPolicy {
        fn default() -> Self {
            NetworkPolicy::Deny
        }
    }

    impl NetworkPolicy {
        pub fn grants_network(&self) -> bool {
            !matches!(self, NetworkPolicy::Deny)
        }
    }

    /// What secret material a capability may read. None by default; anything else
    /// implies the [`Effect::SecretAccess`] declaration.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub enum SecretPolicy {
        /// No secret access.
        None,
        /// Access limited to the listed named secrets.
        Named(Vec<String>),
        /// Access to any secret.
        Any,
    }

    impl Default for SecretPolicy {
        fn default() -> Self {
            SecretPolicy::None
        }
    }

    impl SecretPolicy {
        pub fn grants_secret(&self) -> bool {
            !matches!(self, SecretPolicy::None)
        }
    }

    /// The cost of pulling a capability's full schema into context. Progressive
    /// disclosure trades this off: the compact index is cheap, and the caller pays
    /// `schema_tokens` only when it actually loads the schema.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
    pub struct ContextCost {
        pub schema_tokens: u32,
    }

    /// Where a capability came from, for pinning and audit.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Provenance {
        pub source: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub commit: Option<String>,
        pub license: String,
    }

    /// A reference to a schema document. The `uri` is always present and cheap; the
    /// `raw` payload is the heavy schema text that the registry only parses when
    /// [`crate::extension_registry::Registry::load_full_schema`] is called. Keeping the raw text out of
    /// the compact index and out of every accessor except the explicit load is what
    /// makes disclosure progressive.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct SchemaRef {
        pub uri: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub raw: Option<String>,
    }

    impl SchemaRef {
        pub fn new(uri: impl Into<String>) -> Self {
            SchemaRef {
                uri: uri.into(),
                raw: None,
            }
        }

        pub fn with_raw(uri: impl Into<String>, raw: impl Into<String>) -> Self {
            SchemaRef {
                uri: uri.into(),
                raw: Some(raw.into()),
            }
        }
    }

    /// The full, on-demand schema view. Produced only by an explicit load so the
    /// heavy JSON never rides along with metadata resolution.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct FullSchema {
        pub input_uri: String,
        pub output_uri: String,
        pub input: Option<serde_json::Value>,
        pub output: Option<serde_json::Value>,
    }

    /// The single unified declaration for any capability the registry can serve.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct CapabilityManifest {
        pub id: String,
        pub version: String,
        pub kind: CapabilityKind,
        pub provider: String,
        pub license: String,
        /// Short human description, surfaced in the compact index for disclosure and
        /// used as a haystack when ranking against a task query.
        #[serde(default)]
        pub description: String,
        pub effects: Vec<Effect>,
        pub input_schema_ref: SchemaRef,
        pub output_schema_ref: SchemaRef,
        pub scopes: Vec<Scope>,
        #[serde(default)]
        pub sandbox: SandboxReq,
        #[serde(default)]
        pub network: NetworkPolicy,
        #[serde(default)]
        pub secrets: SecretPolicy,
        #[serde(default)]
        pub context_cost: ContextCost,
        pub provenance: Provenance,
    }

    impl CapabilityManifest {
        /// A minimal manifest with denied network, no secrets, no sandbox, and a
        /// single read effect. Tests and callers override fields from here.
        pub fn new(
            id: impl Into<String>,
            version: impl Into<String>,
            kind: CapabilityKind,
            provider: impl Into<String>,
        ) -> Self {
            let id = id.into();
            CapabilityManifest {
                input_schema_ref: SchemaRef::new(format!("schema://{id}/input")),
                output_schema_ref: SchemaRef::new(format!("schema://{id}/output")),
                id,
                version: version.into(),
                kind,
                provider: provider.into(),
                license: "MIT".to_string(),
                description: String::new(),
                effects: vec![Effect::Read],
                scopes: Vec::new(),
                sandbox: SandboxReq::None,
                network: NetworkPolicy::Deny,
                secrets: SecretPolicy::None,
                context_cost: ContextCost::default(),
                provenance: Provenance {
                    source: "builtin".to_string(),
                    commit: None,
                    license: "MIT".to_string(),
                },
            }
        }

        /// The effects that this manifest's policies and scopes require it to hold.
        /// A network policy that grants access requires [`Effect::Network`]; a secret
        /// policy that grants access requires [`Effect::SecretAccess`]; a network or
        /// secret scope requires the matching effect. Order is stable and de-duped.
        pub fn implied_effects(&self) -> Vec<Effect> {
            let mut out: Vec<Effect> = Vec::new();
            let push = |e: Effect, out: &mut Vec<Effect>| {
                if !out.contains(&e) {
                    out.push(e);
                }
            };
            if self.network.grants_network() {
                push(Effect::Network, &mut out);
            }
            if self.secrets.grants_secret() {
                push(Effect::SecretAccess, &mut out);
            }
            for s in &self.scopes {
                match s {
                    Scope::Network(_) => push(Effect::Network, &mut out),
                    Scope::Secret(_) => push(Effect::SecretAccess, &mut out),
                    _ => {}
                }
            }
            out
        }

        /// Implied effects that the manifest fails to declare. Empty means the
        /// declaration is complete; a non-empty result is a registration blocker.
        pub fn undeclared_effects(&self) -> Vec<Effect> {
            self.implied_effects()
                .into_iter()
                .filter(|e| !self.effects.contains(e))
                .collect()
        }

        /// Whether this manifest's scopes cover a requested scope.
        pub fn scope_allows(&self, want: &Scope) -> bool {
            self.scopes.iter().any(|held| held.covers(want))
        }

        /// Whether this manifest is offered to a given role. A manifest with no role
        /// scope is offered to every role; one with role scopes is offered only to
        /// the roles it names.
        pub fn offered_to_role(&self, role: &str) -> bool {
            let mut has_role_scope = false;
            for s in &self.scopes {
                if let Scope::Role(r) = s {
                    has_role_scope = true;
                    if r == role {
                        return true;
                    }
                }
            }
            !has_role_scope
        }

        /// Whether this manifest explicitly names a role scope for `role`.
        pub fn declares_role(&self, role: &str) -> bool {
            self.scopes
                .iter()
                .any(|s| matches!(s, Scope::Role(r) if r == role))
        }

        /// Count of elevated (non-read) effects, used for least-privilege ranking.
        pub fn elevated_effect_count(&self) -> usize {
            self.effects.iter().filter(|e| e.is_elevated()).count()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn filesystem_prefix_coverage() {
            let held = Scope::Filesystem("src".to_string());
            assert!(held.covers(&Scope::Filesystem("src/lib.rs".to_string())));
            assert!(held.covers(&Scope::Filesystem("src".to_string())));
            assert!(!held.covers(&Scope::Filesystem("tests".to_string())));
            assert!(!held.covers(&Scope::Filesystem("srcextra".to_string())));
        }
        #[test]
        fn network_wildcard_coverage() {
            let any = Scope::Network("*".to_string());
            assert!(any.covers(&Scope::Network("example.com".to_string())));
            let one = Scope::Network("example.com".to_string());
            assert!(one.covers(&Scope::Network("example.com".to_string())));
            assert!(!one.covers(&Scope::Network("evil.com".to_string())));
        }
        #[test]
        fn implied_effects_from_policies_and_scopes() {
            let mut m = CapabilityManifest::new("x", "1.0.0", CapabilityKind::Tool, "p");
            m.network = NetworkPolicy::Any;
            m.secrets = SecretPolicy::Named(vec!["token".to_string()]);
            m.scopes = vec![Scope::Network("api.example.com".to_string())];
            let implied = m.implied_effects();
            assert!(implied.contains(&Effect::Network));
            assert!(implied.contains(&Effect::SecretAccess));
            assert_eq!(implied.iter().filter(|e| **e == Effect::Network).count(), 1);
        }
        #[test]
        fn undeclared_effects_detected() {
            let mut m = CapabilityManifest::new("x", "1.0.0", CapabilityKind::Tool, "p");
            m.network = NetworkPolicy::Any;
            assert_eq!(m.undeclared_effects(), vec![Effect::Network]);
            m.effects.push(Effect::Network);
            assert!(m.undeclared_effects().is_empty());
        }
        #[test]
        fn role_offering() {
            let mut m = CapabilityManifest::new("x", "1.0.0", CapabilityKind::Tool, "p");
            assert!(m.offered_to_role("reviewer"));
            m.scopes = vec![Scope::Role("admin".to_string())];
            assert!(m.offered_to_role("admin"));
            assert!(!m.offered_to_role("reviewer"));
            assert!(m.declares_role("admin"));
        }
    }
}

pub mod registry {
    //! The unified capability registry.
    //!
    //! One [`Registry`] holds every kind of capability behind one ABI. It enforces
    //! the invariants the rest of HIDE relies on:
    //! - a capability may not register an effect it does not declare (undeclared
    //!   effects that its policies or scopes imply are a hard registration error),
    //! - a capability that executes or spawns a process must declare sandbox
    //!   isolation,
    //! - ids are unique,
    //! - a pinned id must match its pinned version and provenance,
    //! - the full schema is disclosed only on request,
    //! - revoked capabilities disappear from resolution and cannot be loaded.

    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::extension_registry::error::{EffectList, RegistryError, Result};
    use crate::extension_registry::index::CompactEntry;
    use crate::extension_registry::manifest::{
        CapabilityKind, CapabilityManifest, Effect, FullSchema, SchemaRef, Scope,
    };

    /// A pin binds an id to an expected version and/or provenance. Any field left
    /// `None` is unconstrained. Registration that names a pinned id must satisfy
    /// every constrained field.
    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct PinSpec {
        pub version: Option<String>,
        pub source: Option<String>,
        pub commit: Option<String>,
    }

    impl PinSpec {
        /// Explain the first mismatch between this pin and a manifest, or `None` if
        /// the manifest satisfies the pin.
        fn mismatch(&self, m: &CapabilityManifest) -> Option<String> {
            if let Some(v) = &self.version {
                if &m.version != v {
                    return Some(format!("version {:?} != pinned {:?}", m.version, v));
                }
            }
            if let Some(s) = &self.source {
                if &m.provenance.source != s {
                    return Some(format!(
                        "provenance source {:?} != pinned {:?}",
                        m.provenance.source, s
                    ));
                }
            }
            if let Some(c) = &self.commit {
                match &m.provenance.commit {
                    Some(mc) if mc == c => {}
                    Some(mc) => return Some(format!("provenance commit {mc:?} != pinned {c:?}")),
                    None => return Some(format!("provenance commit missing, pinned {c:?}")),
                }
            }
            None
        }
    }

    /// A resolution request. Every field is optional or empty by default; an
    /// all-default query matches every active capability.
    #[derive(Debug, Clone, Default)]
    pub struct ResolveQuery {
        /// Free text; whitespace-split keywords are matched against id and
        /// description to rank relevance. Does not filter, only ranks.
        pub task: Option<String>,
        /// Restrict to one kind.
        pub kind: Option<CapabilityKind>,
        /// Restrict to capabilities offered to this role.
        pub role: Option<String>,
        /// Every listed scope must be covered by the capability's declared scopes.
        pub required_scopes: Vec<Scope>,
    }

    impl ResolveQuery {
        pub fn new() -> Self {
            Self::default()
        }
        pub fn task(mut self, t: impl Into<String>) -> Self {
            self.task = Some(t.into());
            self
        }
        pub fn kind(mut self, k: CapabilityKind) -> Self {
            self.kind = Some(k);
            self
        }
        pub fn role(mut self, r: impl Into<String>) -> Self {
            self.role = Some(r.into());
            self
        }
        pub fn require_scope(mut self, s: Scope) -> Self {
            self.required_scopes.push(s);
            self
        }
    }

    /// A candidate returned from [`Registry::resolve_for`], carrying its compact
    /// entry plus the ranking factors that placed it. The list is returned already
    /// sorted best-first, so callers can just take the head.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct RankedCandidate {
        pub entry: CompactEntry,
        /// Count of task keywords found in id or description.
        pub task_matches: usize,
        /// Whether the capability names the queried role explicitly.
        pub role_match: bool,
        /// Count of elevated (non-read) effects; fewer ranks higher.
        pub elevated_effects: usize,
        /// Schema load cost in tokens; cheaper ranks higher.
        pub schema_tokens: u32,
    }

    struct Entry {
        manifest: CapabilityManifest,
        revoked: bool,
    }

    /// The registry. Not `Clone` (it owns a monotonic schema-load counter used to
    /// prove that resolution never eagerly loads schemas).
    #[derive(Default)]
    pub struct Registry {
        entries: BTreeMap<String, Entry>,
        pins: BTreeMap<String, PinSpec>,
        schema_loads: AtomicU64,
    }

    impl Registry {
        pub fn new() -> Self {
            Self::default()
        }

        /// Register a capability. Fails on a duplicate id, on any effect the
        /// manifest's policies or scopes imply but the effects list omits, on empty
        /// required identity fields, and on a pin violation.
        pub fn register(&mut self, manifest: CapabilityManifest) -> Result<()> {
            if manifest.id.trim().is_empty() {
                return Err(RegistryError::InvalidManifest {
                    id: manifest.id.clone(),
                    detail: "id is empty".to_string(),
                });
            }
            if manifest.version.trim().is_empty() {
                return Err(RegistryError::InvalidManifest {
                    id: manifest.id.clone(),
                    detail: "version is empty".to_string(),
                });
            }
            if self.entries.contains_key(&manifest.id) {
                return Err(RegistryError::DuplicateId(manifest.id.clone()));
            }
            let missing = manifest.undeclared_effects();
            if !missing.is_empty() {
                return Err(RegistryError::UndeclaredEffects {
                    id: manifest.id.clone(),
                    missing: EffectList(missing),
                });
            }
            // Sandbox honesty. Registration is the one gate EVERY capability
            // crosses, including any manifest minted from an on-disk or otherwise
            // foreign declaration, so a manifest that claims it executes or spawns
            // a process while requiring no isolation is refused here rather than
            // trusted until call time. The builtin bridge already sets Subprocess
            // for exactly these effects; this makes that a checked invariant.
            if manifest
                .effects
                .iter()
                .any(|e| matches!(e, Effect::Execute | Effect::Process))
                && !manifest.sandbox.requires_isolation()
            {
                return Err(RegistryError::InvalidManifest {
                    id: manifest.id.clone(),
                    detail: "declares Execute or Process but requires no sandbox isolation"
                        .to_string(),
                });
            }
            if let Some(pin) = self.pins.get(&manifest.id) {
                if let Some(detail) = pin.mismatch(&manifest) {
                    return Err(RegistryError::PinViolation {
                        id: manifest.id.clone(),
                        detail,
                    });
                }
            }
            self.entries.insert(
                manifest.id.clone(),
                Entry {
                    manifest,
                    revoked: false,
                },
            );
            Ok(())
        }

        /// Pin an id to an expected version and/or provenance. If the id is already
        /// registered, the current manifest is validated against the pin
        /// immediately (a retroactive pin that the live manifest violates is
        /// rejected). Future registrations of this id must also satisfy the pin.
        pub fn pin(&mut self, id: impl Into<String>, pin: PinSpec) -> Result<()> {
            let id = id.into();
            if let Some(e) = self.entries.get(&id) {
                if let Some(detail) = pin.mismatch(&e.manifest) {
                    return Err(RegistryError::PinViolation { id, detail });
                }
            }
            self.pins.insert(id, pin);
            Ok(())
        }

        /// Revoke a capability. It stays in the map (so its id cannot be silently
        /// reused) but is excluded from the index, from resolution, and from schema
        /// loads. Returns an error if the id was never registered.
        pub fn revoke(&mut self, id: &str) -> Result<()> {
            match self.entries.get_mut(id) {
                Some(e) => {
                    e.revoked = true;
                    Ok(())
                }
                None => Err(RegistryError::NotFound(id.to_string())),
            }
        }

        /// Whether an id is present (registered, possibly revoked).
        pub fn contains(&self, id: &str) -> bool {
            self.entries.contains_key(id)
        }

        /// Whether an id is present and revoked.
        pub fn is_revoked(&self, id: &str) -> bool {
            self.entries.get(id).map(|e| e.revoked).unwrap_or(false)
        }

        /// Number of active (non-revoked) capabilities.
        pub fn active_len(&self) -> usize {
            self.entries.values().filter(|e| !e.revoked).count()
        }

        /// The compact metadata index over active capabilities, ordered by id. This
        /// discloses only id, kind, description, and scopes; it never materializes a
        /// schema.
        pub fn index(&self) -> Vec<CompactEntry> {
            self.entries
                .values()
                .filter(|e| !e.revoked)
                .map(|e| CompactEntry::from_manifest(&e.manifest))
                .collect()
        }

        /// Resolve ranked candidates for a task, role, and scope requirement.
        ///
        /// Hard filters: kind (if set), role offering (if set), and full coverage of
        /// every required scope. Revoked capabilities never appear. Ranking, applied
        /// after filtering, is fully deterministic:
        /// 1. more matched task keywords first,
        /// 2. explicit role match first,
        /// 3. fewer elevated effects first (least privilege),
        /// 4. lower schema-token cost first,
        /// 5. id ascending as the final tie-break.
        pub fn resolve_for(&self, q: &ResolveQuery) -> Vec<RankedCandidate> {
            let keywords = split_keywords(q.task.as_deref());
            let mut out: Vec<RankedCandidate> = Vec::new();

            for e in self.entries.values() {
                if e.revoked {
                    continue;
                }
                let m = &e.manifest;
                if let Some(k) = q.kind {
                    if m.kind != k {
                        continue;
                    }
                }
                if let Some(role) = &q.role {
                    if !m.offered_to_role(role) {
                        continue;
                    }
                }
                if !q.required_scopes.iter().all(|s| m.scope_allows(s)) {
                    continue;
                }

                let haystack = format!("{} {}", m.id, m.description).to_lowercase();
                let task_matches = keywords.iter().filter(|k| haystack.contains(*k)).count();
                let role_match = q
                    .role
                    .as_deref()
                    .map(|r| m.declares_role(r))
                    .unwrap_or(false);

                out.push(RankedCandidate {
                    entry: CompactEntry::from_manifest(m),
                    task_matches,
                    role_match,
                    elevated_effects: m.elevated_effect_count(),
                    schema_tokens: m.context_cost.schema_tokens,
                });
            }

            out.sort_by(|a, b| {
                b.task_matches
                    .cmp(&a.task_matches)
                    .then(b.role_match.cmp(&a.role_match))
                    .then(a.elevated_effects.cmp(&b.elevated_effects))
                    .then(a.schema_tokens.cmp(&b.schema_tokens))
                    .then(a.entry.id.cmp(&b.entry.id))
            });
            out
        }

        /// Load and parse the full input/output schema for a capability. This is the
        /// only path that touches the heavy schema text, and it bumps the schema
        /// load counter so callers can prove resolution stayed lazy. Fails if the id
        /// is unknown, if it is revoked, or if a raw schema fails to parse.
        pub fn load_full_schema(&self, id: &str) -> Result<FullSchema> {
            let e = self
                .entries
                .get(id)
                .ok_or_else(|| RegistryError::NotFound(id.to_string()))?;
            if e.revoked {
                return Err(RegistryError::Revoked(id.to_string()));
            }
            let m = &e.manifest;
            let input = parse_schema(&m.input_schema_ref, id, "input")?;
            let output = parse_schema(&m.output_schema_ref, id, "output")?;
            self.schema_loads.fetch_add(1, Ordering::Relaxed);
            Ok(FullSchema {
                input_uri: m.input_schema_ref.uri.clone(),
                output_uri: m.output_schema_ref.uri.clone(),
                input,
                output,
            })
        }

        /// How many full-schema loads have happened. Zero after any number of
        /// registrations, index builds, and resolutions: those never load a schema.
        pub fn schema_load_count(&self) -> u64 {
            self.schema_loads.load(Ordering::Relaxed)
        }

        // --- enforcement helpers ------------------------------------------------

        /// The declared effects of a capability.
        pub fn declared_effects(&self, id: &str) -> Result<Vec<Effect>> {
            Ok(self.active(id)?.effects.clone())
        }

        /// Whether a capability requires sandbox isolation.
        pub fn requires_sandbox(&self, id: &str) -> Result<bool> {
            Ok(self.active(id)?.sandbox.requires_isolation())
        }

        /// Whether a capability's declared scopes cover a requested scope.
        pub fn scope_allows(&self, id: &str, scope: &Scope) -> Result<bool> {
            Ok(self.active(id)?.scope_allows(scope))
        }

        /// The pinned identity a capability was registered under (version and
        /// provenance), for audit and pinning.
        pub fn version(&self, id: &str) -> Result<String> {
            Ok(self.active(id)?.version.clone())
        }

        /// The provenance of a capability.
        pub fn provenance(
            &self,
            id: &str,
        ) -> Result<crate::extension_registry::manifest::Provenance> {
            Ok(self.active(id)?.provenance.clone())
        }

        /// The kind of a capability.
        pub fn kind(&self, id: &str) -> Result<CapabilityKind> {
            Ok(self.active(id)?.kind)
        }

        /// The context cost (schema token estimate) of a capability.
        pub fn context_cost(&self, id: &str) -> Result<u32> {
            Ok(self.active(id)?.context_cost.schema_tokens)
        }

        /// Borrow the active (non-revoked) manifest for `id`.
        fn active(&self, id: &str) -> Result<&CapabilityManifest> {
            let e = self
                .entries
                .get(id)
                .ok_or_else(|| RegistryError::NotFound(id.to_string()))?;
            if e.revoked {
                return Err(RegistryError::Revoked(id.to_string()));
            }
            Ok(&e.manifest)
        }
    }

    fn split_keywords(task: Option<&str>) -> Vec<String> {
        match task {
            None => Vec::new(),
            Some(t) => t
                .split_whitespace()
                .map(|w| w.to_lowercase())
                .filter(|w| !w.is_empty())
                .collect(),
        }
    }

    fn parse_schema(
        r: &SchemaRef,
        id: &str,
        which: &'static str,
    ) -> Result<Option<serde_json::Value>> {
        match &r.raw {
            None => Ok(None),
            Some(text) => {
                let v = serde_json::from_str(text).map_err(|source| RegistryError::Schema {
                    id: id.to_string(),
                    which,
                    source,
                })?;
                Ok(Some(v))
            }
        }
    }
}

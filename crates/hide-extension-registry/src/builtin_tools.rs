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

use crate::manifest::{
    CapabilityKind, CapabilityManifest, Effect, NetworkPolicy, Provenance, SandboxReq, Scope,
    SchemaRef, SecretPolicy,
};
use crate::registry::Registry;
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
/// The catalog is the source of truth: `hide_tools::register_builtin_tools`
/// populates a `ToolRegistry`, and we bridge each resulting `ToolSpec` into a
/// typed manifest. Registration goes through [`Registry::register`], so the
/// registry's honest-effects invariant is enforced on every derived manifest (a
/// mapping that implied but omitted a network or secret effect would be
/// rejected here, not silently accepted).
pub fn register_builtin_tools(reg: &mut Registry) {
    let tool_registry = ToolRegistry::default();
    hide_tools::register_builtin_tools(&tool_registry);

    for spec in tool_registry.specs() {
        reg.register(manifest_for_spec(&spec))
            .expect("derived builtin tool manifest must satisfy registry invariants");
    }

    // The MCP host is not a static catalog entry: per-server tools are minted
    // dynamically by hide_tools::mcp::mcp_tool_to_hide_spec. Register a single
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
        Some(out) => {
            SchemaRef::with_raw(format!("hide-tools://{}/output", spec.name), out.to_string())
        }
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
/// derived from the transport semantics in `hide_tools::mcp`: the client speaks
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
    use crate::registry::ResolveQuery;

    /// The full builtin catalog id set, mirrored from
    /// `hide_tools::registry::register_builtin_tools`. If hide-tools grows a tool
    /// this list is what the bridge must keep covering.
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
            // Every builtin is bridged as a Tool.
            assert_eq!(reg.kind(id).unwrap(), CapabilityKind::Tool, "kind for {id}");
            // Honest provenance.
            let prov = reg.provenance(id).unwrap();
            assert_eq!(prov.source, "hide-tools", "provenance source for {id}");
            assert_eq!(prov.license, "internal", "provenance license for {id}");
        }
        // 23 catalog tools + the mcp host bridge.
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
        // A pure read never needs the sandbox.
        assert!(!reg.requires_sandbox("fs.read").unwrap());
    }

    #[test]
    fn edit_tool_declares_write() {
        let reg = build_builtin_tool_registry();
        for id in ["edit.search_replace", "edit.apply_patch", "edit.write_file"] {
            let effects = reg.declared_effects(id).unwrap();
            assert!(effects.contains(&Effect::Write), "{id} must declare Write");
            // An edit tool is never declared read-only.
            assert_ne!(effects, vec![Effect::Read], "{id} must not be read-only");
        }
    }

    #[test]
    fn shell_run_is_execute_process_and_sandboxed() {
        let reg = build_builtin_tool_registry();
        let effects = reg.declared_effects("shell.run").unwrap();
        assert!(effects.contains(&Effect::Execute), "shell.run declares Execute");
        assert!(effects.contains(&Effect::Process), "shell.run declares Process");
        assert!(
            reg.requires_sandbox("shell.run").unwrap(),
            "shell.run must require sandbox isolation"
        );
    }

    #[test]
    fn bounded_exec_is_execute_and_sandboxed_but_not_process() {
        // proc tools execute a fixed command: Execute (sandboxed) without the
        // open-world Process reach that shell.run carries.
        let reg = build_builtin_tool_registry();
        for id in ["test.run", "build.run", "compile.check"] {
            let effects = reg.declared_effects(id).unwrap();
            assert!(effects.contains(&Effect::Execute), "{id} declares Execute");
            assert!(!effects.contains(&Effect::Process), "{id} is not open-world");
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
            // Git mutation is not a process execution, so no sandbox is required.
            assert!(!reg.requires_sandbox(id).unwrap(), "{id} sandbox");
        }
    }

    #[test]
    fn shell_plan_and_git_reads_are_read_only() {
        // shell.plan requires shell.exec but only validates; it must not be
        // mistaken for an executor. Same for the git read trio.
        let reg = build_builtin_tool_registry();
        for id in [
            "shell.plan",
            "git.status",
            "git.diff",
            "git.log",
            "git.worktree.list",
            "search.text",
        ] {
            assert_eq!(reg.declared_effects(id).unwrap(), vec![Effect::Read], "{id}");
            assert!(!reg.requires_sandbox(id).unwrap(), "{id} sandbox");
        }
    }

    #[test]
    fn memory_declares_both_read_and_write() {
        // memory is dual-mode: honest declaration carries both, never just one.
        let reg = build_builtin_tool_registry();
        let effects = reg.declared_effects("memory").unwrap();
        assert!(effects.contains(&Effect::Read), "memory reads (view)");
        assert!(effects.contains(&Effect::Write), "memory writes (mutations)");
    }

    #[test]
    fn no_mutating_tool_is_silently_read_only() {
        // Spot check the honesty floor: nothing that mutates is declared as a
        // bare read, and nothing that executes forgets its sandbox.
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
                assert!(reg.requires_sandbox(id).unwrap(), "{id} executes unsandboxed");
            }
        }
    }

    #[test]
    fn progressive_disclosure_effects_without_schema_load() {
        let reg = build_builtin_tool_registry();
        // Building, indexing, resolving, and querying effects must never load a
        // full schema.
        assert_eq!(reg.schema_load_count(), 0);
        let _ = reg.index();
        let _ = reg.declared_effects("shell.run").unwrap();
        let _ = reg.scope_allows("fs.read", &Scope::Repo).unwrap();
        let _ = reg.resolve_for(&ResolveQuery::new().task("edit file").kind(CapabilityKind::Tool));
        assert_eq!(
            reg.schema_load_count(),
            0,
            "effect and index queries must stay schema-free"
        );
        // The compact index carries no schema payload at all.
        for entry in reg.index() {
            // CompactEntry structurally cannot hold a schema; assert it still
            // resolves the id/kind we expect for a known tool.
            if entry.id == "fs.read" {
                assert_eq!(entry.kind, CapabilityKind::Tool);
            }
        }
        // Only an explicit load materializes the schema, and it returns the real
        // hide-tools input schema.
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

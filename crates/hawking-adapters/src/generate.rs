//! Generate family documentation, JSON schemas, CLI surface (commands/help/
//! completion), SDK types, HIDE capability declarations, Fabric declarations,
//! schema migrations, and the root JSON deliverables from the registry.
//!
//! Pattern mirrors `hide-sdk-codegen`: pure deterministic strings, checked-in
//! goldens under `generated/`, drift test fails on diff. **Do not add a second
//! codegen system** — extend this generator.

use std::path::{Path, PathBuf};

use crate::bridge_surface::{bridge_surface_document, bridge_surface_json};
use crate::export::{
    adapter_abi_json, adapter_registry_json, capability_matrix_json, migration_map_json,
    test_matrix_json,
};
use crate::registry::builtin_registry;
use crate::support_level::SupportLevel;
use crate::{ABI_SCHEMA, REGISTRY_SCHEMA};

/// One generated artifact (relative path under the crate + contents).
#[derive(Debug, Clone)]
pub struct GeneratedArtifact {
    pub relative_path: &'static str,
    pub contents: String,
}

/// Produce every generated artifact in a stable order.
pub fn generate_all() -> Vec<GeneratedArtifact> {
    vec![
        GeneratedArtifact {
            // Counted source golden (not under generated/ exclusion).
            relative_path: "goldens/families.md",
            contents: family_docs_md(),
        },
        GeneratedArtifact {
            relative_path: "generated/registry.schema.json",
            contents: registry_json_schema(),
        },
        GeneratedArtifact {
            relative_path: "generated/schemas/adapters.schema.json",
            contents: registry_json_schema(),
        },
        GeneratedArtifact {
            relative_path: "generated/schemas/artifacts.schema.json",
            contents: artifacts_json_schema(),
        },
        GeneratedArtifact {
            relative_path: "generated/schemas/profiles.schema.json",
            contents: profiles_json_schema(),
        },
        GeneratedArtifact {
            relative_path: "generated/schemas/runtime_capabilities.schema.json",
            contents: runtime_capabilities_json_schema(),
        },
        GeneratedArtifact {
            relative_path: "generated/schemas/events.schema.json",
            contents: events_json_schema(),
        },
        GeneratedArtifact {
            relative_path: "generated/schemas/fabric_placement.schema.json",
            contents: fabric_placement_json_schema(),
        },
        GeneratedArtifact {
            relative_path: "generated/schemas/tool_effects.schema.json",
            contents: tool_effects_json_schema(),
        },
        GeneratedArtifact {
            relative_path: "generated/cli_validate.json",
            contents: cli_validate_json(),
        },
        GeneratedArtifact {
            relative_path: "generated/cli_surface.json",
            contents: cli_surface_json(),
        },
        GeneratedArtifact {
            relative_path: "generated/cli_completion.bash",
            contents: cli_completion_bash(),
        },
        GeneratedArtifact {
            relative_path: "generated/cli_completion.zsh",
            contents: cli_completion_zsh(),
        },
        GeneratedArtifact {
            // Counted source golden (not under generated/ exclusion).
            relative_path: "goldens/sdk_types.d.ts",
            contents: sdk_types_ts(),
        },
        GeneratedArtifact {
            relative_path: "generated/hide_capabilities.json",
            contents: hide_capabilities_json(),
        },
        GeneratedArtifact {
            relative_path: "generated/fabric_declarations.json",
            contents: fabric_declarations_json(),
        },
        GeneratedArtifact {
            relative_path: "generated/HAWKING_SCHEMA_MIGRATIONS.json",
            contents: schema_migrations_json(),
        },
        GeneratedArtifact {
            relative_path: "generated/HAWKING_ADAPTER_ABI.json",
            contents: adapter_abi_json(),
        },
        GeneratedArtifact {
            relative_path: "generated/HAWKING_ADAPTER_REGISTRY.json",
            contents: adapter_registry_json(),
        },
        GeneratedArtifact {
            relative_path: "generated/HAWKING_ADAPTER_CAPABILITY_MATRIX.json",
            contents: capability_matrix_json(),
        },
        GeneratedArtifact {
            relative_path: "generated/HAWKING_ADAPTER_TEST_MATRIX.json",
            contents: test_matrix_json(),
        },
        GeneratedArtifact {
            relative_path: "generated/HAWKING_ADAPTER_MIGRATION_MAP.json",
            contents: migration_map_json(),
        },
        GeneratedArtifact {
            relative_path: "generated/HAWKING_CANONICAL_EVENTS.json",
            contents: hawking_events::canonical_events_json(),
        },
        GeneratedArtifact {
            relative_path: "generated/HAWKING_BRIDGE_SURFACE.json",
            contents: bridge_surface_json(),
        },
        GeneratedArtifact {
            relative_path: "generated/HAWKING_CLI_SURFACE.json",
            contents: cli_surface_json(),
        },
    ]
}

/// Published adapter deliverable basenames (canonical location:
/// `crates/hawking-adapters/generated/<name>`). Used by drift tests to ensure
/// these names are not re-published at the repo root.
pub fn repo_root_artifacts() -> Vec<(&'static str, String)> {
    vec![
        ("HAWKING_ADAPTER_ABI.json", adapter_abi_json()),
        ("HAWKING_ADAPTER_REGISTRY.json", adapter_registry_json()),
        (
            "HAWKING_ADAPTER_CAPABILITY_MATRIX.json",
            capability_matrix_json(),
        ),
        ("HAWKING_ADAPTER_TEST_MATRIX.json", test_matrix_json()),
        ("HAWKING_ADAPTER_MIGRATION_MAP.json", migration_map_json()),
        (
            "HAWKING_CANONICAL_EVENTS.json",
            hawking_events::canonical_events_json(),
        ),
        ("HAWKING_BRIDGE_SURFACE.json", bridge_surface_json()),
        ("HAWKING_CLI_SURFACE.json", cli_surface_json()),
        ("HAWKING_SCHEMA_MIGRATIONS.json", schema_migrations_json()),
    ]
}

/// Write all artifacts under `crate_root` (the hawking-adapters package dir).
pub fn write_all(crate_root: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut written = Vec::new();
    for art in generate_all() {
        let path = crate_root.join(art.relative_path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, &art.contents)?;
        written.push(path);
    }
    Ok(written)
}

fn pretty(doc: serde_json::Value) -> String {
    let mut s = serde_json::to_string_pretty(&doc).expect("json serializes");
    s.push('\n');
    s
}

// ---------------------------------------------------------------------------
// Family docs
// ---------------------------------------------------------------------------

fn family_docs_md() -> String {
    let r = builtin_registry();
    let mut out = String::from(
        "# Hawking model-family adapter registry\n\n\
         Generated from `hawking-adapters` — do not hand-edit.\n\n\
         **No family is PRODUCTION today.**\n\n\
         | Family | Level | Executes | Serve-registered | Module |\n\
         |---|---|---|---|---|\n",
    );
    for d in r.families() {
        out.push_str(&format!(
            "| {} | {} | {} | {} | `{}` |\n",
            d.display_name,
            d.level.as_str(),
            d.executes,
            d.serve_registered,
            d.module
        ));
    }
    out.push_str("\n## Gaps\n\n");
    for d in r.families() {
        out.push_str(&format!("### {}\n\n", d.id));
        for g in d.gaps {
            out.push_str(&format!("- {g}\n"));
        }
        out.push('\n');
    }
    // Single trailing newline only (git diff --check rejects blank line at EOF).
    while out.ends_with("\n\n") {
        out.pop();
    }
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------
// JSON schemas
// ---------------------------------------------------------------------------

fn registry_json_schema() -> String {
    let grades: Vec<&str> = SupportLevel::all().iter().map(|g| g.as_str()).collect();
    let schema = serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": REGISTRY_SCHEMA,
        "title": "HawkingAdapterRegistry",
        "type": "object",
        "required": ["schema", "families", "support_levels"],
        "properties": {
            "schema": { "const": REGISTRY_SCHEMA },
            "abi_schema": { "const": ABI_SCHEMA },
            "support_levels": {
                "type": "array",
                "items": { "type": "string", "enum": grades }
            },
            "families": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "id", "aliases", "level", "evidence", "module",
                        "executes", "serve_registered", "gaps", "abi"
                    ],
                    "properties": {
                        "id": { "type": "string" },
                        "aliases": { "type": "array", "items": { "type": "string" } },
                        "level": {
                            "type": "string",
                            "enum": [
                                "DECLARED",
                                "SOURCE_HEADER_VALIDATED",
                                "SYNTHETIC_PARITY",
                                "REAL_TENSOR_DECODE",
                                "SMALL_REAL_CHECKPOINT",
                                "FULL_PARENT_VALIDATED",
                                "PRODUCTION"
                            ]
                        },
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["path", "claim", "kind"],
                                "properties": {
                                    "path": { "type": "string" },
                                    "claim": { "type": "string" },
                                    "kind": { "type": "string" }
                                }
                            }
                        },
                        "abi": {
                            "type": "object",
                            "description": "Full family ABI; every field value or null+reason"
                        }
                    }
                }
            }
        }
    });
    pretty(schema)
}

fn artifacts_json_schema() -> String {
    let r = builtin_registry();
    let mut codecs = std::collections::BTreeSet::new();
    for d in r.families() {
        if let Some(vs) = d.abi.artifact_codecs.values {
            for v in vs {
                codecs.insert(*v);
            }
        }
    }
    // Canonical codec ids the surface understands (registry may use free text).
    let canonical = ["gguf", "gravity", "safetensors", "tq", "synthetic_twin"];
    for c in canonical {
        codecs.insert(c);
    }
    let codec_list: Vec<&str> = codecs.into_iter().collect();
    pretty(serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "hawking.artifacts.v1",
        "title": "HawkingArtifact",
        "description": "Loadable model artifact descriptors derived from the adapter registry codecs.",
        "type": "object",
        "required": ["codec", "path", "family_id"],
        "properties": {
            "codec": {
                "type": "string",
                "description": "Artifact codec. Free-text values from family ABI are allowed; known canonical ids are listed.",
                "examples": codec_list
            },
            "path": { "type": "string" },
            "family_id": { "type": "string" },
            "shard_index": { "type": ["integer", "null"], "minimum": 0 },
            "digest": { "type": ["string", "null"] },
            "serve_registered": { "type": "boolean" }
        }
    }))
}

fn profiles_json_schema() -> String {
    pretty(serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "hawking.profiles.v1",
        "title": "HawkingRuntimeAndKernelProfiles",
        "description": "Runtime lever profiles (CLI --profile) and kernel-profile JSON shape.",
        "definitions": {
            "RuntimeProfile": {
                "type": "string",
                "enum": ["default", "fast", "race", "efficient", "exact"]
            },
            "KernelProfile": {
                "type": "object",
                "required": ["schema_version", "profile_id", "profile_name", "model_arch", "selected"],
                "properties": {
                    "schema_version": { "type": "integer", "const": 1 },
                    "profile_id": { "type": "string" },
                    "profile_name": { "type": "string" },
                    "model_id": { "type": "string" },
                    "model_arch": { "type": "string" },
                    "device_name": { "type": "string" },
                    "shader_hash": { "type": "string" },
                    "selected": { "type": "object" },
                    "evidence": { "type": "object" }
                }
            }
        },
        "oneOf": [
            { "$ref": "#/definitions/RuntimeProfile" },
            { "$ref": "#/definitions/KernelProfile" }
        ]
    }))
}

fn runtime_capabilities_json_schema() -> String {
    let r = builtin_registry();
    let families: Vec<_> = r
        .families()
        .map(|d| {
            serde_json::json!({
                "family": d.id,
                "level": d.level.as_str(),
                "executes": d.executes,
                "serve_registered": d.serve_registered,
                "providers": {
                    "cpu": d.abi.providers.cpu.value,
                    "metal": d.abi.providers.metal.value,
                    "cuda": d.abi.providers.cuda.value,
                }
            })
        })
        .collect();
    pretty(serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "hawking.runtime_capabilities.v1",
        "title": "HawkingRuntimeCapabilities",
        "type": "object",
        "required": ["schema", "families", "runtime_profiles", "bridge_endpoints"],
        "properties": {
            "schema": { "const": "hawking.runtime_capabilities.v1" },
            "families": {
                "type": "array",
                "description": "Per-family execute/serve/provider capability snapshot from the registry."
            },
            "runtime_profiles": {
                "type": "array",
                "items": { "type": "string", "enum": ["default", "fast", "race", "efficient", "exact"] }
            },
            "bridge_endpoints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["endpoint", "status"],
                    "properties": {
                        "endpoint": { "type": "string" },
                        "status": {
                            "type": "string",
                            "enum": ["live", "partial", "not_implemented"]
                        }
                    }
                }
            }
        },
        "examples": [{
            "schema": "hawking.runtime_capabilities.v1",
            "families": families,
            "runtime_profiles": ["default", "fast", "race", "efficient", "exact"],
            "bridge_endpoints": bridge_endpoints_json()
        }]
    }))
}

fn events_json_schema() -> String {
    let cats: Vec<&str> = hawking_events::Category::all()
        .iter()
        .map(|c| c.as_str())
        .collect();
    let you_kinds: Vec<&str> = hawking_events::YOU_EVENTS.iter().map(|s| s.kind).collect();
    pretty(serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "hawking.events.canonical.schema.v1",
        "title": "CanonicalEvent",
        "description": "Durable product event envelope. YOU events use the same schema on the same bus.",
        "type": "object",
        "required": ["id", "seq", "session_id", "kind", "surface", "subsystem", "verification", "category"],
        "properties": {
            "id": { "type": "string", "description": "Stable event id" },
            "seq": { "type": "integer", "minimum": 0, "description": "Monotone sequence within session" },
            "session_id": { "type": "string" },
            "kind": { "type": "string" },
            "surface": {
                "type": "string",
                "enum": ["you", "chat", "ide", "bridge", "terminal", "mcp", "sdk", "serve", "system"]
            },
            "subsystem": { "type": "string" },
            "verification": {
                "type": "string",
                "enum": ["target_verified", "provisional"],
                "description": "Load-bearing Draft/Verified wall; provisional must not render as final"
            },
            "category": { "type": "string", "enum": cats },
            "payload": { "type": "object" }
        },
        "you_event_kinds": you_kinds
    }))
}

fn fabric_placement_json_schema() -> String {
    pretty(serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "hawking.fabric.placement.v1",
        "title": "FabricPlacement",
        "description": "Fabric placement declaration derived from family ABI fabric_partition_boundaries.",
        "type": "object",
        "required": ["family", "placement", "serve_registered"],
        "properties": {
            "family": { "type": "string" },
            "placement": {
                "type": "string",
                "enum": ["local_serve_eligible", "not_serve_registered", "declared_only"]
            },
            "serve_registered": { "type": "boolean" },
            "fabric_partition": { "type": ["string", "null"] },
            "fabric_partition_null_reason": { "type": ["string", "null"] },
            "event_kind": { "const": "fabric.placement" }
        }
    }))
}

fn tool_effects_json_schema() -> String {
    pretty(serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "hawking.tool_effects.v1",
        "title": "ToolEffectSet",
        "description": "Tool effect request shape (hide-core EffectSet) exposed for SDK consumers.",
        "type": "object",
        "required": ["effects"],
        "properties": {
            "effects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["kind", "target", "risk"],
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["read", "write", "delete", "execute", "network", "model", "plugin", "unknown"]
                        },
                        "target": { "type": "string" },
                        "bytes_hash": { "type": ["string", "null"] },
                        "risk": {
                            "type": "string",
                            "enum": ["trivial", "low", "medium", "high", "critical"]
                        },
                        "metadata": {
                            "type": "object",
                            "additionalProperties": { "type": "string" }
                        }
                    }
                }
            }
        }
    }))
}

// ---------------------------------------------------------------------------
// CLI validation + surface + completion
// ---------------------------------------------------------------------------

fn cli_validate_json() -> String {
    let r = builtin_registry();
    let rules: Vec<_> = r
        .families()
        .map(|d| {
            serde_json::json!({
                "family": d.id,
                "aliases": d.aliases,
                "max_level": d.level.as_str(),
                "require_evidence_when_above": SupportLevel::Declared.as_str(),
                "require_evidence_kind": crate::abi::required_evidence_kind(d.level)
                    .map(|k| k.as_str()),
                "forbid_production": true,
                "executes": d.executes,
                "serve_registered": d.serve_registered,
                "evidence_paths": d.evidence.iter().map(|e| e.path).collect::<Vec<_>>(),
                "abi_fields_required": crate::abi::ABI_FIELD_NAMES,
            })
        })
        .collect();
    pretty(serde_json::json!({
        "schema": "hawking.adapters.cli_validate.v1",
        "rules": rules,
        "global": {
            "forbid_production": true,
            "note": "A level asserted without backing evidence of the grade-named kind must fail validation. Every ABI field must be present or null with a reason."
        }
    }))
}

/// CLI commands, help text, and adapter-derived flags — sole generated CLI surface.
fn cli_surface_json() -> String {
    let r = builtin_registry();
    let family_ids: Vec<&str> = r.families().map(|d| d.id).collect();
    let family_aliases: Vec<&str> = r
        .families()
        .flat_map(|d| d.aliases.iter().copied())
        .collect();
    let grades: Vec<&str> = SupportLevel::all().iter().map(|g| g.as_str()).collect();

    let family_rows: Vec<_> = r
        .families()
        .map(|d| {
            serde_json::json!({
                "id": d.id,
                "aliases": d.aliases,
                "level": d.level.as_str(),
                "executes": d.executes,
                "serve_registered": d.serve_registered,
                "help": format!(
                    "{} — support {}{}{}",
                    d.display_name,
                    d.level.as_str(),
                    if d.executes { "; executes" } else { "; does not execute" },
                    if d.serve_registered { "; serve-registered" } else { "" }
                ),
            })
        })
        .collect();

    // Core hawking subcommands (kept in lockstep with crates/hawking/src/main.rs Cmd).
    // Adapter-derived commands are additive and generated from the registry.
    let base_commands = vec![
        serde_json::json!({
            "name": "serve",
            "help": "Start the OpenAI-compatible HTTP server.",
            "flags": [
                {"name": "--weights", "help": "Path to a GGUF (or a single .gravity shard file)."},
                {"name": "--gravity", "help": "Serve a sealed .gravity artifact (directory or shard). Also HAWKING_GRAVITY."},
                {"name": "--addr", "help": "Listen address.", "default": "0.0.0.0:8080"},
                {"name": "--profile", "help": "Runtime lever profile.", "values": ["default","fast","race","efficient","exact"]},
            ]
        }),
        serde_json::json!({
            "name": "generate",
            "help": "One-shot generation to stdout.",
            "flags": [
                {"name": "--weights", "help": "Path to model weights."},
                {"name": "--profile", "help": "Runtime lever profile.", "values": ["default","fast","race","efficient","exact"]},
            ]
        }),
        serde_json::json!({
            "name": "adapters",
            "help": "Inspect the model-family adapter registry (generated from hawking-adapters; cannot drift).",
            "subcommands": [
                {
                    "name": "list",
                    "help": "List families with honest support grades (no PRODUCTION today).",
                    "flags": []
                },
                {
                    "name": "show",
                    "help": "Show full ABI + evidence for one family.",
                    "flags": [
                        {"name": "family", "positional": true, "values": family_ids, "help": "Family id or alias."}
                    ]
                },
                {
                    "name": "validate",
                    "help": "Validate that every family's level is backed by evidence of the grade-named kind.",
                    "flags": []
                },
            ]
        }),
        serde_json::json!({
            "name": "doctor",
            "help": "Inspect model size, KV-cache budget, current RSS, and per-Mac fit."
        }),
        serde_json::json!({
            "name": "version",
            "help": "Print version and the model id, if a weights path is given."
        }),
    ];

    pretty(serde_json::json!({
        "schema": "hawking.cli.surface.v1",
        "generated_from": "crates/hawking-adapters FamilyRegistry + bridge surface + runtime profiles",
        "note": "This is the sole CLI surface document. Shell completion and help for adapter-related commands are derived here so the CLI cannot invent families or grades the registry does not own.",
        "binary": "hawking",
        "global_flags": [
            {
                "name": "--profile",
                "help": "Named lever bundle (default/fast/race/efficient/exact). Explicit HAWKING_QWEN_* env always wins.",
                "values": ["default", "fast", "race", "efficient", "exact"]
            }
        ],
        "commands": base_commands,
        "adapter_families": family_rows,
        "completion": {
            "family_ids": family_ids,
            "family_aliases": family_aliases,
            "support_levels": grades,
            "runtime_profiles": ["default", "fast", "race", "efficient", "exact"],
            "artifact_flags": ["--weights", "--gravity"],
        },
        "honesty": {
            "no_family_is_production": true,
            "responses_and_messages_not_implemented": true,
            "bridge_endpoints": bridge_endpoints_json(),
        }
    }))
}

fn cli_completion_bash() -> String {
    let r = builtin_registry();
    let families: Vec<&str> = r
        .families()
        .flat_map(|d| std::iter::once(d.id).chain(d.aliases.iter().copied()))
        .collect();
    let family_list = families.join(" ");
    let profiles = "default fast race efficient exact";
    let cmds = "serve generate tokenize bench autotune doctor profile-rank stats version adapters adapters-list adapters-show adapters-validate";

    format!(
        r#"# Generated by hawking-adapters-codegen — do not hand-edit.
# Source: eval "$(hawking-adapters-codegen --print-bash-completion)" or source this file.
_hawking_complete() {{
  local cur prev
  COMPREPLY=()
  cur="${{COMP_WORDS[COMP_CWORD]}}"
  prev="${{COMP_WORDS[COMP_CWORD-1]}}"
  local cmds="{cmds}"
  local profiles="{profiles}"
  local families="{family_list}"
  local grades="{grades}"

  if [[ ${{COMP_CWORD}} -eq 1 ]]; then
    COMPREPLY=( $(compgen -W "${{cmds}}" -- "${{cur}}") )
    return 0
  fi
  case "${{prev}}" in
    --profile)
      COMPREPLY=( $(compgen -W "${{profiles}}" -- "${{cur}}") )
      return 0
      ;;
    show|adapters-show)
      COMPREPLY=( $(compgen -W "${{families}}" -- "${{cur}}") )
      return 0
      ;;
    --weights|--gravity|--kernel-profile|--hardware-profile|--profile-json)
      COMPREPLY=( $(compgen -f -- "${{cur}}") )
      return 0
      ;;
  esac
  case "${{COMP_WORDS[1]}}" in
    adapters)
      if [[ ${{COMP_CWORD}} -eq 2 ]]; then
        COMPREPLY=( $(compgen -W "list show validate" -- "${{cur}}") )
      elif [[ ${{COMP_WORDS[2]}} == show ]]; then
        COMPREPLY=( $(compgen -W "${{families}}" -- "${{cur}}") )
      fi
      ;;
    *)
      COMPREPLY=( $(compgen -W "--profile --weights --gravity --help" -- "${{cur}}") )
      ;;
  esac
}}
complete -F _hawking_complete hawking
"#,
        cmds = cmds,
        profiles = profiles,
        family_list = family_list,
        grades = SupportLevel::all()
            .iter()
            .map(|g| g.as_str())
            .collect::<Vec<_>>()
            .join(" "),
    )
}

fn cli_completion_zsh() -> String {
    let r = builtin_registry();
    let families: Vec<&str> = r
        .families()
        .flat_map(|d| std::iter::once(d.id).chain(d.aliases.iter().copied()))
        .collect();
    let family_list = families.join(" ");
    format!(
        r#"#compdef hawking
# Generated by hawking-adapters-codegen — do not hand-edit.
_hawking() {{
  local -a commands profiles families
  commands=(
    'serve:Start the OpenAI-compatible HTTP server'
    'generate:One-shot generation to stdout'
    'tokenize:CPU-only tokenizer parity diagnostic'
    'bench:Run a benchmark suite'
    'autotune:Deterministically select a kernel/runtime profile'
    'doctor:Inspect model size and fit'
    'version:Print version'
    'adapters:Inspect the model-family adapter registry'
  )
  profiles=(default fast race efficient exact)
  families=({family_list})

  _arguments -C \
    '(-h --help)'{{-h,--help}}'[help]' \
    '--profile[runtime lever profile]:profile:(${{profiles}})' \
    '1: :->cmd' \
    '*::arg:->args'

  case $state in
    cmd)
      _describe -t commands 'hawking command' commands
      ;;
    args)
      case $words[1] in
        adapters)
          if (( CURRENT == 2 )); then
            _values 'adapters subcommand' list show validate
          elif [[ $words[2] == show ]]; then
            _values 'family' ${{families}}
          fi
          ;;
        serve|generate)
          _arguments \
            '--weights[weights path]:file:_files' \
            '--gravity[gravity artifact]:file:_files' \
            '--profile[runtime profile]:profile:(${{profiles}})'
          ;;
      esac
      ;;
  esac
}}
compdef _hawking hawking
"#,
        family_list = family_list,
    )
}

// ---------------------------------------------------------------------------
// SDK TypeScript
// ---------------------------------------------------------------------------

fn sdk_types_ts() -> String {
    let r = builtin_registry();
    let mut out = String::from(
        "/* Generated by hawking-adapters-codegen — do not hand-edit. */\n\
         /* Covers: adapters, artifacts, profiles, runtime capabilities, events, Fabric placement, tool effects. */\n\n",
    );

    // --- Adapters ---
    out.push_str(
        "export type SupportLevel =\n\
           | \"DECLARED\"\n\
           | \"SOURCE_HEADER_VALIDATED\"\n\
           | \"SYNTHETIC_PARITY\"\n\
           | \"REAL_TENSOR_DECODE\"\n\
           | \"SMALL_REAL_CHECKPOINT\"\n\
           | \"FULL_PARENT_VALIDATED\"\n\
           | \"PRODUCTION\";\n\n\
         export type EvidenceKind =\n\
           | \"description\"\n\
           | \"source_header\"\n\
           | \"synthetic_parity\"\n\
           | \"real_tensor_decode\"\n\
           | \"small_checkpoint_run\"\n\
           | \"full_parent_validation\"\n\
           | \"production_receipt\";\n\n\
         export type FamilyId =\n",
    );
    let ids: Vec<_> = r.families().map(|d| d.id).collect();
    for (i, id) in ids.iter().enumerate() {
        let sep = if i + 1 == ids.len() { ";" } else { "" };
        out.push_str(&format!("  | \"{id}\"{sep}\n"));
    }
    out.push_str(
        "\nexport interface FamilyEvidence {\n\
         \tpath: string;\n\
         \tclaim: string;\n\
         \tkind: EvidenceKind;\n\
         }\n\n\
         export interface AbiField<T = string> {\n\
         \tvalue: T | null;\n\
         \tnull_reason: string | null;\n\
         }\n\n\
         export interface FamilyAdapterEntry {\n\
         \tid: FamilyId;\n\
         \taliases: string[];\n\
         \tdisplayName: string;\n\
         \tlevel: SupportLevel;\n\
         \tevidence: FamilyEvidence[];\n\
         \tmodule: string;\n\
         \texecutes: boolean;\n\
         \tserveRegistered: boolean;\n\
         \tgaps: string[];\n\
         }\n\n\
         export const FAMILY_ADAPTERS: FamilyAdapterEntry[] = [\n",
    );
    for d in r.families() {
        out.push_str("  {\n");
        out.push_str(&format!("    id: \"{}\",\n", d.id));
        out.push_str("    aliases: [");
        for (i, a) in d.aliases.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!("\"{a}\""));
        }
        out.push_str("],\n");
        out.push_str(&format!(
            "    displayName: \"{}\",\n",
            d.display_name.replace('"', "\\\"")
        ));
        out.push_str(&format!("    level: \"{}\",\n", d.level.as_str()));
        out.push_str("    evidence: [\n");
        for e in d.evidence {
            out.push_str(&format!(
                "      {{ path: \"{}\", claim: \"{}\", kind: \"{}\" }},\n",
                e.path,
                e.claim.replace('"', "\\\""),
                e.kind.as_str()
            ));
        }
        out.push_str("    ],\n");
        out.push_str(&format!(
            "    module: \"{}\",\n",
            d.module.replace('"', "\\\"")
        ));
        out.push_str(&format!("    executes: {},\n", d.executes));
        out.push_str(&format!("    serveRegistered: {},\n", d.serve_registered));
        out.push_str("    gaps: [\n");
        for g in d.gaps {
            out.push_str(&format!("      \"{}\",\n", g.replace('"', "\\\"")));
        }
        out.push_str("    ],\n");
        out.push_str("  },\n");
    }
    out.push_str("];\n\n");

    // --- Artifacts ---
    out.push_str(
        "export type ArtifactCodec = string;\n\n\
         export interface ArtifactRef {\n\
         \tcodec: ArtifactCodec;\n\
         \tpath: string;\n\
         \tfamilyId: FamilyId;\n\
         \tshardIndex?: number | null;\n\
         \tdigest?: string | null;\n\
         \tserveRegistered?: boolean;\n\
         }\n\n",
    );

    // --- Profiles ---
    out.push_str(
        "export type RuntimeProfile = \"default\" | \"fast\" | \"race\" | \"efficient\" | \"exact\";\n\n\
         export interface KernelProfile {\n\
         \tschema_version: 1;\n\
         \tprofile_id: string;\n\
         \tprofile_name: string;\n\
         \tmodel_id?: string;\n\
         \tmodel_arch: string;\n\
         \tdevice_name?: string;\n\
         \tshader_hash?: string;\n\
         \tselected: Record<string, unknown>;\n\
         \tevidence?: Record<string, unknown>;\n\
         }\n\n\
         export const RUNTIME_PROFILES: RuntimeProfile[] = [\"default\", \"fast\", \"race\", \"efficient\", \"exact\"];\n\n",
    );

    // --- Runtime capabilities ---
    out.push_str(
        "export type BridgeEndpointStatus = \"live\" | \"partial\" | \"not_implemented\";\n\n\
         export interface BridgeEndpoint {\n\
         \tendpoint: string;\n\
         \tstatus: BridgeEndpointStatus;\n\
         \tentry_path: string;\n\
         \ttests: string[];\n\
         }\n\n\
         export interface RuntimeCapabilities {\n\
         \tschema: \"hawking.runtime_capabilities.v1\";\n\
         \tfamilies: FamilyAdapterEntry[];\n\
         \truntimeProfiles: RuntimeProfile[];\n\
         \tbridgeEndpoints: BridgeEndpoint[];\n\
         }\n\n",
    );

    // --- Events ---
    out.push_str(
        "export type ContentVerification = \"target_verified\" | \"provisional\";\n\n\
         export type ProducingSurface =\n\
           | \"you\" | \"chat\" | \"ide\" | \"bridge\" | \"terminal\"\n\
           | \"mcp\" | \"sdk\" | \"serve\" | \"system\";\n\n\
         export type EventCategory =\n",
    );
    let cats = hawking_events::Category::all();
    for (i, c) in cats.iter().enumerate() {
        let sep = if i + 1 == cats.len() { ";" } else { "" };
        out.push_str(&format!("  | \"{}\"{sep}\n", c.as_str()));
    }
    out.push_str(
        "\nexport interface CanonicalEventEnvelope {\n\
         \tid: string;\n\
         \tseq: number;\n\
         \tsession_id: string;\n\
         \tsurface: ProducingSurface;\n\
         \tsubsystem: string;\n\
         \tverification: ContentVerification;\n\
         \tcategory: EventCategory;\n\
         \tkind: string;\n\
         \tpayload: Record<string, unknown>;\n\
         }\n\n\
         export type YouEventName =\n",
    );
    for (i, s) in hawking_events::YOU_EVENTS.iter().enumerate() {
        let sep = if i + 1 == hawking_events::YOU_EVENTS.len() {
            ";"
        } else {
            ""
        };
        out.push_str(&format!("  | \"{}\"{sep}\n", s.event.as_pascal()));
    }
    out.push_str("\nexport type YouEventKind =\n");
    for (i, s) in hawking_events::YOU_EVENTS.iter().enumerate() {
        let sep = if i + 1 == hawking_events::YOU_EVENTS.len() {
            ";"
        } else {
            ""
        };
        out.push_str(&format!("  | \"{}\"{sep}\n", s.kind));
    }
    out.push_str("\nexport const YOU_EVENTS: { event: YouEventName; kind: YouEventKind; category: EventCategory; defaultVerification: ContentVerification }[] = [\n");
    for s in hawking_events::YOU_EVENTS {
        out.push_str(&format!(
            "  {{ event: \"{}\", kind: \"{}\", category: \"{}\", defaultVerification: \"{}\" }},\n",
            s.event.as_pascal(),
            s.kind,
            s.category.as_str(),
            s.default_verification.as_str(),
        ));
    }
    out.push_str("];\n\n");

    // --- Fabric placement ---
    out.push_str(
        "export type FabricPlacementKind = \"local_serve_eligible\" | \"not_serve_registered\" | \"declared_only\";\n\n\
         export interface FabricPlacement {\n\
         \tfamily: FamilyId;\n\
         \tplacement: FabricPlacementKind;\n\
         \tserveRegistered: boolean;\n\
         \tfabricPartition: string | null;\n\
         \tfabricPartitionNullReason: string | null;\n\
         \teventKind: \"fabric.placement\";\n\
         }\n\n\
         export const FABRIC_PLACEMENTS: FabricPlacement[] = [\n",
    );
    for d in r.families() {
        let placement = if d.serve_registered {
            "local_serve_eligible"
        } else if d.executes {
            "not_serve_registered"
        } else {
            "declared_only"
        };
        let part = match d.abi.fabric_partition_boundaries.value {
            Some(v) => format!("\"{}\"", v.replace('"', "\\\"")),
            None => "null".into(),
        };
        let reason = match d.abi.fabric_partition_boundaries.null_reason {
            Some(v) => format!("\"{}\"", v.replace('"', "\\\"")),
            None => "null".into(),
        };
        out.push_str(&format!(
            "  {{ family: \"{}\", placement: \"{}\", serveRegistered: {}, fabricPartition: {}, fabricPartitionNullReason: {}, eventKind: \"fabric.placement\" }},\n",
            d.id, placement, d.serve_registered, part, reason
        ));
    }
    out.push_str("];\n\n");

    // --- Tool effects ---
    out.push_str(
        "export type EffectKind = \"read\" | \"write\" | \"delete\" | \"execute\" | \"network\" | \"model\" | \"plugin\" | \"unknown\";\n\
         export type RiskLevel = \"trivial\" | \"low\" | \"medium\" | \"high\" | \"critical\";\n\n\
         export interface ToolEffect {\n\
         \tkind: EffectKind;\n\
         \ttarget: string;\n\
         \tbytes_hash?: string | null;\n\
         \trisk: RiskLevel;\n\
         \tmetadata?: Record<string, string>;\n\
         }\n\n\
         export interface ToolEffectSet {\n\
         \teffects: ToolEffect[];\n\
         }\n",
    );

    out
}

// ---------------------------------------------------------------------------
// HIDE capabilities + Fabric
// ---------------------------------------------------------------------------

fn hide_capabilities_json() -> String {
    let r = builtin_registry();
    let caps: Vec<_> = r
        .families()
        .map(|d| {
            serde_json::json!({
                "id": format!("model_family.{}", d.id),
                "kind": "model_family",
                "level": d.level.as_str(),
                "aliases": d.aliases,
                "executes": d.executes,
                "serve_registered": d.serve_registered,
                "description": format!("{} — {}", d.display_name, d.level.as_str()),
            })
        })
        .collect();
    pretty(serde_json::json!({
        "schema": "hawking.hide.model_family_capabilities.v1",
        "capabilities": caps,
    }))
}

fn fabric_declarations_json() -> String {
    let cats: Vec<_> = hawking_events::all_categories()
        .iter()
        .map(|c| {
            serde_json::json!({
                "category": c.as_str(),
                "kind": hawking_events::kind_for_category(*c),
            })
        })
        .collect();
    let r = builtin_registry();
    let families: Vec<_> = r
        .families()
        .map(|d| {
            serde_json::json!({
                "family": d.id,
                "serve_registered": d.serve_registered,
                "placement": if d.serve_registered { "local_serve_eligible" } else if d.executes { "not_serve_registered" } else { "declared_only" },
                "fabric_partition": d.abi.fabric_partition_boundaries.value,
                "fabric_partition_null_reason": d.abi.fabric_partition_boundaries.null_reason,
            })
        })
        .collect();
    pretty(serde_json::json!({
        "schema": "hawking.fabric.declarations.v1",
        "note": "Declarations only — Fabric implementation is a parallel lane.",
        "event_categories": cats,
        "family_placement": families,
    }))
}

// ---------------------------------------------------------------------------
// Schema migrations (versioned upgrade paths)
// ---------------------------------------------------------------------------

fn schema_migrations_json() -> String {
    pretty(serde_json::json!({
        "schema": "hawking.schema_migrations.v1",
        "note": "Versioned upgrade paths for generated schemas. A consumer pinned to an older version uses these steps rather than breaking silently.",
        "schemas": [
            {
                "id": "hawking.adapters.registry",
                "current": "v2",
                "path": "crates/hawking-adapters/generated/HAWKING_ADAPTER_REGISTRY.json",
                "migrations": [
                    {
                        "from": "v1",
                        "to": "v2",
                        "breaking": true,
                        "steps": [
                            "Rename support grade fields to uppercase SupportLevel enum strings",
                            "Require full ABI object per family (every field present or null with null_reason)",
                            "Add evidence[].kind of the grade-named EvidenceKind"
                        ],
                        "compat": "v1 consumers must re-read against v2; there is no silent field map"
                    }
                ]
            },
            {
                "id": "hawking.adapters.abi",
                "current": "v1",
                "path": "crates/hawking-adapters/generated/HAWKING_ADAPTER_ABI.json",
                "migrations": []
            },
            {
                "id": "hawking.events.canonical",
                "current": "v1",
                "path": "crates/hawking-adapters/generated/HAWKING_CANONICAL_EVENTS.json",
                "migrations": [
                    {
                        "from": "v1-pre-you",
                        "to": "v1",
                        "breaking": false,
                        "steps": [
                            "Add envelope field surface (producing surface) alongside subsystem",
                            "Add eight YOU categories (you_objects … you_handoff) and seventeen you.* kinds",
                            "Add you_events array documenting ObjectAdded … HandoffCreated",
                            "category_count becomes 24; primary_kinds gains YOU primaries"
                        ],
                        "compat": "Events without surface default via subsystem.default_surface(); pre-YOU consumers ignore unknown categories"
                    }
                ]
            },
            {
                "id": "hawking.bridge.surface",
                "current": "v1",
                "path": "crates/hawking-adapters/generated/HAWKING_BRIDGE_SURFACE.json",
                "migrations": []
            },
            {
                "id": "hawking.cli.surface",
                "current": "v1",
                "path": "crates/hawking-adapters/generated/HAWKING_CLI_SURFACE.json",
                "migrations": [
                    {
                        "from": "none",
                        "to": "v1",
                        "breaking": false,
                        "steps": [
                            "Introduce generated CLI surface (commands, help, completion tokens) from FamilyRegistry",
                            "Adapter family ids/aliases/grades become completion sources"
                        ],
                        "compat": "New artifact; no prior consumer"
                    }
                ]
            },
            {
                "id": "hawking.artifacts",
                "current": "v1",
                "path": "crates/hawking-adapters/generated/schemas/artifacts.schema.json",
                "migrations": []
            },
            {
                "id": "hawking.profiles",
                "current": "v1",
                "path": "crates/hawking-adapters/generated/schemas/profiles.schema.json",
                "migrations": []
            },
            {
                "id": "hawking.runtime_capabilities",
                "current": "v1",
                "path": "crates/hawking-adapters/generated/schemas/runtime_capabilities.schema.json",
                "migrations": []
            },
            {
                "id": "hawking.fabric.placement",
                "current": "v1",
                "path": "crates/hawking-adapters/generated/schemas/fabric_placement.schema.json",
                "migrations": []
            },
            {
                "id": "hawking.tool_effects",
                "current": "v1",
                "path": "crates/hawking-adapters/generated/schemas/tool_effects.schema.json",
                "migrations": []
            }
        ]
    }))
}

fn bridge_endpoints_json() -> Vec<serde_json::Value> {
    bridge_surface_document()["endpoints"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

//! Generate family documentation, JSON schemas, CLI validation, SDK types,
//! HIDE capability declarations, Fabric declarations, and the five root
//! deliverables from the registry.
//!
//! Pattern mirrors `hide-sdk-codegen`: pure deterministic strings, checked-in
//! goldens under `generated/`, drift test fails on diff. **Do not add a second
//! codegen system** — extend this generator.

use std::path::{Path, PathBuf};

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
            relative_path: "generated/families.md",
            contents: family_docs_md(),
        },
        GeneratedArtifact {
            relative_path: "generated/registry.schema.json",
            contents: registry_json_schema(),
        },
        GeneratedArtifact {
            relative_path: "generated/cli_validate.json",
            contents: cli_validate_json(),
        },
        GeneratedArtifact {
            relative_path: "generated/sdk_types.d.ts",
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
    ]
}

/// Repo-root deliverables written by the codegen binary.
pub fn repo_root_artifacts() -> Vec<(&'static str, String)> {
    vec![
        ("HAWKING_ADAPTER_ABI.json", adapter_abi_json()),
        ("HAWKING_ADAPTER_REGISTRY.json", adapter_registry_json()),
        (
            "HAWKING_ADAPTER_CAPABILITY_MATRIX.json",
            capability_matrix_json(),
        ),
        ("HAWKING_ADAPTER_TEST_MATRIX.json", test_matrix_json()),
        (
            "HAWKING_ADAPTER_MIGRATION_MAP.json",
            migration_map_json(),
        ),
        (
            "HAWKING_CANONICAL_EVENTS.json",
            hawking_events::canonical_events_json(),
        ),
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
    out
}

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
    let mut s = serde_json::to_string_pretty(&schema).unwrap();
    s.push('\n');
    s
}

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
    let doc = serde_json::json!({
        "schema": "hawking.adapters.cli_validate.v1",
        "rules": rules,
        "global": {
            "forbid_production": true,
            "note": "A level asserted without backing evidence of the grade-named kind must fail validation. Every ABI field must be present or null with a reason."
        }
    });
    let mut s = serde_json::to_string_pretty(&doc).unwrap();
    s.push('\n');
    s
}

fn sdk_types_ts() -> String {
    let r = builtin_registry();
    let mut out = String::from(
        "/* Generated by hawking-adapters-codegen — do not hand-edit. */\n\n\
         export type SupportLevel =\n\
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
    out.push_str("];\n");
    out
}

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
    let doc = serde_json::json!({
        "schema": "hawking.hide.model_family_capabilities.v1",
        "capabilities": caps,
    });
    let mut s = serde_json::to_string_pretty(&doc).unwrap();
    s.push('\n');
    s
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
                "placement": if d.serve_registered { "local_serve_eligible" } else { "not_serve_registered" },
                "fabric_partition": d.abi.fabric_partition_boundaries.value,
                "fabric_partition_null_reason": d.abi.fabric_partition_boundaries.null_reason,
            })
        })
        .collect();
    let doc = serde_json::json!({
        "schema": "hawking.fabric.declarations.v1",
        "note": "Declarations only — Fabric implementation is a parallel lane.",
        "event_categories": cats,
        "family_placement": families,
    });
    let mut s = serde_json::to_string_pretty(&doc).unwrap();
    s.push('\n');
    s
}

//! Export registry + derived matrices from the sole FamilyRegistry.
//!
//! Generated documents (not hand-maintained):
//! - `HAWKING_ADAPTER_ABI.json` — ABI schema
//! - `HAWKING_ADAPTER_REGISTRY.json` — full family registry
//! - `HAWKING_ADAPTER_CAPABILITY_MATRIX.json`
//! - `HAWKING_ADAPTER_TEST_MATRIX.json`
//! - `HAWKING_ADAPTER_MIGRATION_MAP.json`

use serde_json::{json, Value};

use crate::abi::{
    required_evidence_kind, AbiField, AbiListField, FamilyDescriptor, ABI_FIELD_NAMES,
};
use crate::registry::builtin_registry;
use crate::support_level::SupportLevel;
use crate::{ABI_SCHEMA, REGISTRY_SCHEMA};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn field_json(f: &AbiField) -> Value {
    match (f.value, f.null_reason) {
        (Some(v), _) => json!({ "value": v, "null_reason": null }),
        (None, Some(r)) => json!({ "value": null, "null_reason": r }),
        (None, None) => json!({ "value": null, "null_reason": "INCOMPLETE_FIELD" }),
    }
}

fn list_field_json(f: &AbiListField) -> Value {
    match (f.values, f.null_reason) {
        (Some(vs), _) => json!({ "value": vs, "null_reason": null }),
        (None, Some(r)) => json!({ "value": null, "null_reason": r }),
        (None, None) => json!({ "value": null, "null_reason": "INCOMPLETE_FIELD" }),
    }
}

fn family_json(d: &FamilyDescriptor) -> Value {
    let cl = &d.abi.context_limits;
    json!({
        "id": d.id,
        "aliases": d.aliases,
        "display_name": d.display_name,
        "level": d.level.as_str(),
        "evidence": d.evidence.iter().map(|e| json!({
            "path": e.path,
            "claim": e.claim,
            "kind": e.kind.as_str(),
        })).collect::<Vec<_>>(),
        "module": d.module,
        "executes": d.executes,
        "serve_registered": d.serve_registered,
        "gaps": d.gaps,
        "abi": {
            "source_config_classes": list_field_json(&d.abi.source_config_classes),
            "tensor_namespace_rules": field_json(&d.abi.tensor_namespace_rules),
            "tokenizer": field_json(&d.abi.tokenizer),
            "chat_template": field_json(&d.abi.chat_template),
            "attention_or_state": field_json(&d.abi.attention_or_state),
            "topology": field_json(&d.abi.topology),
            "normalization": field_json(&d.abi.normalization),
            "positional_encoding": field_json(&d.abi.positional_encoding),
            "kv_or_state_format": field_json(&d.abi.kv_or_state_format),
            "context_limits": {
                "max_context_tokens": cl.max_context_tokens,
                "validation_status": cl.validation_status,
                "null_reason": cl.null_reason,
            },
            "artifact_codecs": list_field_json(&d.abi.artifact_codecs),
            "providers": {
                "cpu": field_json(&d.abi.providers.cpu),
                "metal": field_json(&d.abi.providers.metal),
                "cuda": field_json(&d.abi.providers.cuda),
            },
            "fabric_partition_boundaries": field_json(&d.abi.fabric_partition_boundaries),
            "capability_limitations": d.abi.capability_limitations,
            "source_precision_classes": list_field_json(&d.abi.source_precision_classes),
            "parity_evidence": d.evidence.iter().map(|e| json!({
                "path": e.path,
                "claim": e.claim,
                "kind": e.kind.as_str(),
            })).collect::<Vec<_>>(),
        }
    })
}

fn pretty(doc: Value) -> String {
    let mut s = serde_json::to_string_pretty(&doc).expect("json serializes");
    s.push('\n');
    s
}

// ---------------------------------------------------------------------------
// ABI schema document
// ---------------------------------------------------------------------------

pub fn adapter_abi_document() -> Value {
    let grades: Vec<Value> = SupportLevel::all()
        .iter()
        .map(|g| {
            let req = required_evidence_kind(*g).map(|k| k.as_str());
            json!({
                "grade": g.as_str(),
                "required_evidence_kind": req,
                "meaning": grade_meaning(*g),
            })
        })
        .collect();

    let fields: Vec<Value> = ABI_FIELD_NAMES
        .iter()
        .map(|name| {
            json!({
                "name": name,
                "required": true,
                "null_allowed": !matches!(*name, "family_id" | "aliases" | "support_level" | "parity_evidence" | "capability_limitations"),
                "null_requires_reason": true,
                "description": field_description(name),
            })
        })
        .collect();

    json!({
        "schema": ABI_SCHEMA,
        "title": "HawkingAdapterAbi",
        "note": "One canonical adapter ABI. Families declare every field; generators emit matrices. No family is PRODUCTION today.",
        "support_grades": grades,
        "fields": fields,
        "promotion_rule": "Promotion requires the evidence the grade names. No family becomes PRODUCTION because its shapes look familiar.",
        "canonical_home": {
            "crate": "crates/hawking-adapters",
            "registry": "crates/hawking-adapters/src/registry.rs",
            "family_modules": "crates/hawking-adapters/src/families.rs",
            "codegen": "hawking-adapters-codegen (same golden/drift pattern as hide-sdk-codegen; not a second codegen system)"
        }
    })
}

fn grade_meaning(g: SupportLevel) -> &'static str {
    match g {
        SupportLevel::Declared => "described; nothing parsed, nothing executes",
        SupportLevel::SourceHeaderValidated => {
            "real official config/tokenizer/safetensors header parsed and mapped"
        }
        SupportLevel::SyntheticParity => "matches a deterministic reference on a synthetic twin",
        SupportLevel::RealTensorDecode => "at least one real tensor decoded from a real checkpoint",
        SupportLevel::SmallRealCheckpoint => {
            "a real small checkpoint of the family runs end to end"
        }
        SupportLevel::FullParentValidated => "a real full-size parent validated",
        SupportLevel::Production => "served, under test, with a standing parity receipt",
    }
}

fn field_description(name: &str) -> &'static str {
    match name {
        "family_id" => "Canonical family id (stable string key)",
        "aliases" => "Alternate ids / arch strings that resolve to this family",
        "source_config_classes" => "Source/config class identifiers",
        "tensor_namespace_rules" => "Tensor naming / packing rules",
        "tokenizer" => "Tokenizer identity or protocol",
        "chat_template" => "Chat template identity",
        "attention_or_state" => "Attention or recurrent state mechanism",
        "topology" => "Dense / MoE topology",
        "normalization" => "Normalization class",
        "positional_encoding" => "Positional encoding class",
        "kv_or_state_format" => "KV cache or state format",
        "context_limits" => "Context limits and their validation status",
        "artifact_codecs" => "Ingestible artifact codecs",
        "providers" => "CPU / Metal / CUDA provider availability",
        "fabric_partition_boundaries" => "Fabric partition / placement boundaries",
        "capability_limitations" => "Known capability limitations",
        "source_precision_classes" => "Source precision / quant classes",
        "parity_evidence" => "Named parity / grade evidence entries",
        "support_level" => "Honest support grade on the seven-step ladder",
        _ => "",
    }
}

pub fn adapter_abi_json() -> String {
    pretty(adapter_abi_document())
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

pub fn adapter_registry_document() -> Value {
    let r = builtin_registry();
    let families: Vec<Value> = r.families().map(family_json).collect();

    json!({
        "schema": REGISTRY_SCHEMA,
        "note": "No family is PRODUCTION today. Levels are never inflated from a code reading alone. Extended ABI: every field present or null with reason.",
        "support_levels": SupportLevel::all().iter().map(|g| g.as_str()).collect::<Vec<_>>(),
        "canonical_home": "crates/hawking-adapters — sole metadata/ABI authority; execution remains load_engine + gravity_engine",
        "authorities_not_sole": authorities_not_sole(),
        "families": families,
    })
}

fn authorities_not_sole() -> Value {
    json!([
        {
            "name": "load_engine",
            "path": "crates/hawking-core/src/model/mod.rs",
            "role": "GGUF live dispatch (llama, deepseek2, qwen, qwen-moe, rwkv7) + gravity",
            "canonical_after_migration": "remains sole GGUF+gravity *execution* factory; support-level/ABI metadata lives in hawking-adapters"
        },
        {
            "name": "gravity_engine",
            "path": "crates/hawking-core/src/model/gravity_engine.rs",
            "role": "live .gravity for llama + glm_moe_dsa only",
            "canonical_after_migration": "remains sole .gravity execution path; family ABI fields for glm/llama gravity live in hawking-adapters"
        },
        {
            "name": "seed-c ArchAdapter",
            "path": "crates/hawking-seed-c/src/providers/adapters.rs",
            "role": "declarative plan summary — does not execute",
            "canonical_after_migration": "plan-summary metadata; family support grade + full ABI is hawking-adapters"
        },
        {
            "name": "PRODUCTION_EXECUTION_ADAPTER_REGISTRY",
            "path": "tools/condense/glm52_worker.py",
            "role": "empty by contract (fail-closed)",
            "canonical_after_migration": "stays empty until a real PRODUCTION grade is earned in hawking-adapters"
        },
        {
            "name": "hawking-adapters-extra",
            "path": "packs/hawking-adapters-extra.json",
            "role": "gemma2/phi3/mixtral/mamba2/olmoe extracted off-tree",
            "canonical_after_migration": "pack remains the off-tree code home; families declared under hawking-adapters"
        },
        {
            "name": "hawking-adapters FamilyRegistry",
            "path": "crates/hawking-adapters/src/registry.rs",
            "role": "THIS crate — sole honest support-level + full ABI index (metadata ABI, not a second runtime)"
        },
        {
            "name": "hawking-orch adapters",
            "path": "crates/hawking-orch/src/adapters.rs",
            "role": "LoRA language/task selection — different concept",
            "canonical_after_migration": "unchanged; not architecture-family ABI"
        }
    ])
}

pub fn adapter_registry_json() -> String {
    pretty(adapter_registry_document())
}

// ---------------------------------------------------------------------------
// Capability matrix: family × capability, with grade backing each cell
// ---------------------------------------------------------------------------

const CAPABILITIES: &[&str] = &[
    "execute_forward",
    "serve_registered",
    "gguf_codec",
    "gravity_codec",
    "safetensors_codec",
    "cpu_provider",
    "metal_provider",
    "cuda_provider",
    "tokenizer_declared",
    "chat_template_declared",
    "dense_topology",
    "moe_topology",
    "attention_or_state_declared",
    "fabric_partition_declared",
    "source_header_validated",
    "synthetic_parity",
    "real_tensor_decode",
    "small_real_checkpoint",
    "full_parent_validated",
    "production",
];

pub fn capability_matrix_document() -> Value {
    let r = builtin_registry();
    let mut rows = Vec::new();
    for d in r.families() {
        let mut caps = serde_json::Map::new();
        for cap in CAPABILITIES {
            caps.insert((*cap).to_string(), capability_cell(d, cap));
        }
        rows.push(json!({
            "family": d.id,
            "support_level": d.level.as_str(),
            "capabilities": caps,
        }));
    }
    json!({
        "schema": "hawking.adapters.capability_matrix.v1",
        "note": "Each cell carries the family's support grade as backing. Grade is never inflated to claim a capability the evidence does not name.",
        "capabilities": CAPABILITIES,
        "rows": rows,
    })
}

fn capability_cell(d: &FamilyDescriptor, cap: &str) -> Value {
    let level = d.level.as_str();
    let evidence_paths: Vec<&str> = d.evidence.iter().map(|e| e.path).collect();

    let (present, note) = match cap {
        "execute_forward" => (Some(d.executes), "descriptor.executes"),
        "serve_registered" => (Some(d.serve_registered), "descriptor.serve_registered"),
        "gguf_codec" => (
            Some(list_contains(&d.abi.artifact_codecs, "gguf")),
            "abi.artifact_codecs",
        ),
        "gravity_codec" => (
            Some(list_contains(&d.abi.artifact_codecs, "gravity")),
            "abi.artifact_codecs",
        ),
        "safetensors_codec" => (
            Some(list_contains(&d.abi.artifact_codecs, "safetensors")),
            "abi.artifact_codecs",
        ),
        "cpu_provider" => (
            Some(provider_available(&d.abi.providers.cpu)),
            "abi.providers.cpu",
        ),
        "metal_provider" => (
            Some(provider_available(&d.abi.providers.metal)),
            "abi.providers.metal",
        ),
        "cuda_provider" => (
            Some(provider_available(&d.abi.providers.cuda)),
            "abi.providers.cuda",
        ),
        "tokenizer_declared" => (Some(d.abi.tokenizer.value.is_some()), "abi.tokenizer"),
        "chat_template_declared" => (
            Some(d.abi.chat_template.value.is_some()),
            "abi.chat_template",
        ),
        "dense_topology" => (
            Some(field_mentions(&d.abi.topology, "dense")),
            "abi.topology",
        ),
        "moe_topology" => (
            Some(field_mentions(&d.abi.topology, "MoE") || field_mentions(&d.abi.topology, "moe")),
            "abi.topology",
        ),
        "attention_or_state_declared" => (
            Some(d.abi.attention_or_state.value.is_some()),
            "abi.attention_or_state",
        ),
        "fabric_partition_declared" => (
            Some(d.abi.fabric_partition_boundaries.value.is_some()),
            "abi.fabric_partition_boundaries",
        ),
        "source_header_validated" => (
            Some(d.level >= SupportLevel::SourceHeaderValidated),
            "support_level ladder",
        ),
        "synthetic_parity" => (
            Some(d.level >= SupportLevel::SyntheticParity),
            "support_level ladder",
        ),
        "real_tensor_decode" => (
            Some(d.level >= SupportLevel::RealTensorDecode),
            "support_level ladder",
        ),
        "small_real_checkpoint" => (
            Some(d.level >= SupportLevel::SmallRealCheckpoint),
            "support_level ladder",
        ),
        "full_parent_validated" => (
            Some(d.level >= SupportLevel::FullParentValidated),
            "support_level ladder",
        ),
        "production" => (Some(false), "no family is PRODUCTION today"),
        _ => (None, "unknown capability"),
    };

    json!({
        "present": present,
        "backing_grade": level,
        "note": note,
        "evidence_paths": evidence_paths,
    })
}

fn list_contains(f: &AbiListField, needle: &str) -> bool {
    f.values
        .map(|vs| {
            vs.iter().any(|v| {
                v.to_ascii_lowercase()
                    .contains(&needle.to_ascii_lowercase())
            })
        })
        .unwrap_or(false)
}

fn field_mentions(f: &AbiField, needle: &str) -> bool {
    f.value
        .map(|v| {
            v.to_ascii_lowercase()
                .contains(&needle.to_ascii_lowercase())
        })
        .unwrap_or(false)
}

fn provider_available(f: &AbiField) -> bool {
    f.value
        .map(|v| {
            let l = v.to_ascii_lowercase();
            l.contains("available") || l.contains("partial")
        })
        .unwrap_or(false)
}

pub fn capability_matrix_json() -> String {
    pretty(capability_matrix_document())
}

// ---------------------------------------------------------------------------
// Test matrix: family × test, exist vs missing
// ---------------------------------------------------------------------------

/// Known / desired tests per family. Paths are checked on disk at generation time
/// when possible; the matrix records existence.
pub fn test_matrix_document() -> Value {
    let r = builtin_registry();
    let root = crate::evidence::workspace_root();
    let mut rows = Vec::new();

    for d in r.families() {
        let mut tests = Vec::new();
        // Evidence paths that look like tests
        for e in d.evidence {
            let is_test = e.path.contains("/tests/") || e.path.ends_with("_test.py");
            let exists = crate::evidence::resolve_logical_path(&root, e.path).exists();
            tests.push(json!({
                "path": e.path,
                "role": "evidence",
                "kind": e.kind.as_str(),
                "claim": e.claim,
                "exists": exists,
                "is_test": is_test,
                "status": if exists { "present" } else { "missing" },
            }));
        }
        // Desired grade-ladder tests that may be missing
        for desired in desired_tests_for(d) {
            let already = tests.iter().any(|t| {
                t.get("path")
                    .and_then(|p| p.as_str())
                    .map(|p| p == desired.path)
                    .unwrap_or(false)
            });
            if already {
                continue;
            }
            let exists = crate::evidence::resolve_logical_path(&root, desired.path).exists();
            tests.push(json!({
                "path": desired.path,
                "role": "desired",
                "kind": desired.kind,
                "claim": desired.claim,
                "exists": exists,
                "is_test": true,
                "status": if exists { "present" } else { "missing" },
            }));
        }
        let present = tests.iter().filter(|t| t["status"] == "present").count();
        let missing = tests.iter().filter(|t| t["status"] == "missing").count();
        rows.push(json!({
            "family": d.id,
            "support_level": d.level.as_str(),
            "tests_present": present,
            "tests_missing": missing,
            "tests": tests,
        }));
    }

    json!({
        "schema": "hawking.adapters.test_matrix.v1",
        "note": "Generated from registry evidence + desired grade-ladder tests. Missing entries are not invented as present.",
        "rows": rows,
    })
}

struct DesiredTest {
    path: &'static str,
    kind: &'static str,
    claim: &'static str,
}

fn desired_tests_for(d: &FamilyDescriptor) -> Vec<DesiredTest> {
    // Grade-ladder gaps: if family is below a grade, note the missing promotion test.
    let mut out = Vec::new();
    if d.level < SupportLevel::SourceHeaderValidated {
        out.push(DesiredTest {
            path: "MISSING:source_header_validation_test",
            kind: "source_header",
            claim: "no SOURCE_HEADER_VALIDATED test registered for this family",
        });
    }
    if d.level < SupportLevel::SyntheticParity {
        out.push(DesiredTest {
            path: "MISSING:synthetic_parity_test",
            kind: "synthetic_parity",
            claim: "no SYNTHETIC_PARITY test registered for this family",
        });
    }
    if d.level < SupportLevel::RealTensorDecode {
        out.push(DesiredTest {
            path: "MISSING:real_tensor_decode_test",
            kind: "real_tensor_decode",
            claim: "no REAL_TENSOR_DECODE test registered for this family",
        });
    }
    if d.level < SupportLevel::SmallRealCheckpoint {
        out.push(DesiredTest {
            path: "MISSING:small_real_checkpoint_test",
            kind: "small_checkpoint_run",
            claim: "no SMALL_REAL_CHECKPOINT e2e test registered for this family",
        });
    }
    if d.level < SupportLevel::FullParentValidated {
        out.push(DesiredTest {
            path: "MISSING:full_parent_validation_test",
            kind: "full_parent_validation",
            claim: "no FULL_PARENT_VALIDATED receipt/test registered for this family",
        });
    }
    // PRODUCTION is always missing today
    out.push(DesiredTest {
        path: "MISSING:production_parity_receipt",
        kind: "production_receipt",
        claim: "no PRODUCTION standing parity receipt (forbidden until deliberately lifted)",
    });
    out
}

pub fn test_matrix_json() -> String {
    pretty(test_matrix_document())
}

// ---------------------------------------------------------------------------
// Migration map: old authority → canonical, per call site
// ---------------------------------------------------------------------------

pub fn migration_map_document() -> Value {
    json!({
        "schema": "hawking.adapters.migration_map.v1",
        "note": "Architecture-family metadata/ABI consolidates on hawking-adapters. Execution factories stay put. Do not invent a second runtime registry.",
        "canonical": {
            "metadata_abi": {
                "name": "hawking-adapters FamilyRegistry",
                "path": "crates/hawking-adapters/src/registry.rs",
                "owns": ["support_level", "family ABI fields", "capability matrix", "test matrix", "evidence honesty"]
            },
            "execution_gguf": {
                "name": "load_engine",
                "path": "crates/hawking-core/src/model/mod.rs",
                "owns": ["GGUF open", "arch string dispatch", "Engine construction"]
            },
            "execution_gravity": {
                "name": "gravity_engine",
                "path": "crates/hawking-core/src/model/gravity_engine.rs",
                "owns": [".gravity open", "llama + glm_moe_dsa forward"]
            }
        },
        "mappings": [
            {
                "old_authority": "load_engine",
                "path": "crates/hawking-core/src/model/mod.rs",
                "call_sites": [
                    "crates/hawking-serve/src/lib.rs",
                    "crates/hawking-core/src/engine.rs",
                    "crates/hawking-core/tests/*"
                ],
                "role_was": "implicit sole authority for what families exist + execute",
                "canonical_for": ["execution_gguf", "execution_gravity_dispatch"],
                "not_canonical_for": ["support_level", "full_abi_metadata"],
                "migration": "Keep load_engine as sole GGUF+gravity factory. Family support grades and ABI fields are read from hawking-adapters; do not re-derive them from arch-string match arms."
            },
            {
                "old_authority": "gravity_engine",
                "path": "crates/hawking-core/src/model/gravity_engine.rs",
                "call_sites": [
                    "crates/hawking-core/src/model/mod.rs",
                    "crates/hawking-core/tests/gravity_*"
                ],
                "role_was": "live .gravity execute for llama + glm_moe_dsa",
                "canonical_for": ["execution_gravity"],
                "not_canonical_for": ["support_level", "full_abi_metadata"],
                "migration": "Remains execution path. GLM/llama gravity ABI + grade live under families.rs rows glm / llama."
            },
            {
                "old_authority": "seed-c ArchAdapter",
                "path": "crates/hawking-seed-c/src/providers/adapters.rs",
                "call_sites": [
                    "crates/hawking-seed-c/src/providers/adapters.rs::builtins",
                    "PROVIDER_FOUNDRY / ladder consumers"
                ],
                "role_was": "declarative plan-summary descriptor (does not execute)",
                "canonical_for": ["plan_summary_metadata"],
                "not_canonical_for": ["support_level", "execution", "production_claims"],
                "migration": "ArchAdapter stays plan-only. When a consumer needs family grade or full ABI, use hawking-adapters. Do not treat ArchAdapter presence as SOURCE_HEADER_VALIDATED or higher."
            },
            {
                "old_authority": "PRODUCTION_EXECUTION_ADAPTER_REGISTRY",
                "path": "tools/condense/glm52_worker.py",
                "call_sites": [
                    "tools/condense/glm52_worker.py"
                ],
                "role_was": "empty-by-contract production registry (fail-closed)",
                "canonical_for": ["fail_closed_production_gate"],
                "not_canonical_for": ["family_declaration"],
                "migration": "Keep empty until a family earns PRODUCTION in hawking-adapters with a standing parity receipt. Never populate from shape familiarity."
            },
            {
                "old_authority": "hawking-adapters-extra",
                "path": "packs/hawking-adapters-extra.json",
                "call_sites": [
                    "crates/hawking-core/src/model/mod.rs (error strings / pack references)",
                    "crates/hawking-core/tests/gemma2_smoke.rs",
                    "crates/hawking-core/tests/phi3_smoke.rs",
                    "crates/hawking-core/tests/mamba2_smoke.rs"
                ],
                "role_was": "off-tree extracted engines",
                "canonical_for": ["off_tree_engine_pack"],
                "not_canonical_for": ["support_level_index"],
                "migration": "Pack remains code home for gemma2/phi3/mixtral/mamba2/olmoe. Families are DECLARED under hawking-adapters with pack paths as evidence."
            },
            {
                "old_authority": "hawking-orch adapters",
                "path": "crates/hawking-orch/src/adapters.rs",
                "call_sites": [
                    "crates/hawking-orch (LoRA selection)"
                ],
                "role_was": "language/task LoRA selection policy",
                "canonical_for": ["lora_selection"],
                "not_canonical_for": ["architecture_family_abi"],
                "migration": "No migration — different concept. Do not merge LoRA adapters into FamilyRegistry."
            },
            {
                "old_authority": "Python condense family adapters",
                "path": "tools/condense/*_adapter.py",
                "call_sites": [
                    "tools/condense/glm52_adapter.py",
                    "tools/condense/qwen3_moe_adapter.py",
                    "tools/condense/deepseek_v4_adapter.py",
                    "tools/condense/gravity_execution_adapter.py"
                ],
                "role_was": "campaign-side condense/forge helpers",
                "canonical_for": ["campaign_condense_helpers"],
                "not_canonical_for": ["shipping_support_level"],
                "migration": "Campaign tools may keep local helpers; shipping honesty for family grade is hawking-adapters only."
            }
        ]
    })
}

pub fn migration_map_json() -> String {
    pretty(migration_map_document())
}

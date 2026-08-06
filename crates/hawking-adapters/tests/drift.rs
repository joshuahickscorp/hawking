use hawking_adapters::generate::{generate_all, repo_root_artifacts};
use std::path::PathBuf;
fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
fn workspace_root() -> PathBuf {
    let mut dir = crate_root();
    dir.pop(); // crates
    dir.pop(); // root
    dir
}
#[test]
fn generated_artifacts_match_checked_in() {
    let root = crate_root();
    let mut failures = Vec::new();
    for art in generate_all() {
        let path = root.join(art.relative_path);
        if !path.exists() {
            failures.push(format!(
                "missing checked-in artifact {} — run: cargo run -p hawking-adapters --bin hawking-adapters-codegen -j 4",
                art.relative_path
            ));
            continue;
        }
        let on_disk = std::fs::read_to_string(&path).expect("read golden");
        if on_disk != art.contents {
            failures.push(format!(
                "drift in {} ({} bytes on disk vs {} generated) — re-run hawking-adapters-codegen",
                art.relative_path,
                on_disk.len(),
                art.contents.len()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "generated artifact drift:\n{}",
        failures.join("\n")
    );
}
#[test]
fn adapter_deliverables_live_under_generated_only() {
    let workspace = workspace_root();
    let crate_dir = crate_root();
    let mut failures = Vec::new();
    for (name, contents) in repo_root_artifacts() {
        let root_path = workspace.join(name);
        if root_path.exists() {
            failures.push(format!(
                "repo-root duplicate must not exist: {name} (canonical path is crates/hawking-adapters/generated/{name})"
            ));
        }
        let gen_path = crate_dir.join("generated").join(name);
        if !gen_path.exists() {
            failures.push(format!(
                "missing generated deliverable generated/{name} — run hawking-adapters-codegen"
            ));
            continue;
        }
        let on_disk = std::fs::read_to_string(&gen_path).expect("read generated deliverable");
        if on_disk != contents {
            failures.push(format!(
                "drift in generated/{name} ({} bytes on disk vs {} generated)",
                on_disk.len(),
                contents.len()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "adapter deliverable placement/drift:\n{}",
        failures.join("\n")
    );
}
#[test]
fn regenerating_produces_no_diff() {
    generated_artifacts_match_checked_in();
    adapter_deliverables_live_under_generated_only();
}
#[test]
fn you_events_present_in_canonical_export() {
    let doc: serde_json::Value =
        serde_json::from_str(&hawking_events::canonical_events_json()).unwrap();
    let you = doc
        .get("you_events")
        .expect("you_events block in HAWKING_CANONICAL_EVENTS");
    assert_eq!(you["count"], 17);
    let events = you["events"].as_array().expect("events array");
    assert_eq!(events.len(), 17);
    let names: Vec<&str> = events
        .iter()
        .map(|e| e["event"].as_str().unwrap())
        .collect();
    for expected in [
        "ObjectAdded",
        "ObjectProcessed",
        "MemoryProposed",
        "MemoryCommitted",
        "MemoryCorrected",
        "ConnectorRead",
        "ConnectorWriteProposed",
        "ResearchStarted",
        "SourceCaptured",
        "ClaimVerified",
        "AutomationCreated",
        "AutomationRan",
        "SwarmCreated",
        "AgentDelegated",
        "AgentResult",
        "ProjectUpdated",
        "HandoffCreated",
    ] {
        assert!(
            names.contains(&expected),
            "missing YOU event {expected} in export"
        );
    }
    let envelope = doc["chosen_model"]["envelope_fields"]
        .as_array()
        .expect("envelope_fields");
    for field in [
        "id",
        "seq",
        "session_id",
        "surface",
        "subsystem",
        "verification",
    ] {
        assert!(
            envelope.iter().any(|v| v.as_str() == Some(field)),
            "envelope missing {field}"
        );
    }
    assert_eq!(doc["category_count"], 24);
}
#[test]
fn cli_surface_lists_registry_families() {
    let surface = generate_all()
        .into_iter()
        .find(|a| a.relative_path == "generated/cli_surface.json")
        .expect("cli_surface generated");
    let doc: serde_json::Value = serde_json::from_str(&surface.contents).unwrap();
    let families = doc["adapter_families"].as_array().unwrap();
    let r = hawking_adapters::builtin_registry();
    assert_eq!(families.len(), r.families().count());
    for d in r.families() {
        assert!(families
            .iter()
            .any(|f| f["id"] == d.id && f["level"] == d.level.as_str()));
    }
    let honesty = &doc["honesty"]["bridge_endpoints"];
    for ep in ["POST /v1/responses", "POST /v1/messages"] {
        let row = honesty
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["endpoint"] == ep)
            .unwrap_or_else(|| panic!("missing {ep}"));
        assert_eq!(row["status"], "not_implemented");
    }
}
#[test]
fn schema_migrations_cover_new_surfaces() {
    let mig = generate_all()
        .into_iter()
        .find(|a| a.relative_path == "generated/HAWKING_SCHEMA_MIGRATIONS.json")
        .expect("schema migrations generated");
    let doc: serde_json::Value = serde_json::from_str(&mig.contents).unwrap();
    let ids: Vec<&str> = doc["schemas"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["id"].as_str().unwrap())
        .collect();
    for need in [
        "hawking.adapters.registry",
        "hawking.events.canonical",
        "hawking.cli.surface",
        "hawking.artifacts",
        "hawking.profiles",
        "hawking.runtime_capabilities",
        "hawking.fabric.placement",
        "hawking.tool_effects",
    ] {
        assert!(ids.contains(&need), "migrations missing {need}");
    }
}
#[test]
fn sdk_types_cover_required_domains() {
    let ts = generate_all()
        .into_iter()
        .find(|a| a.relative_path == "goldens/sdk_types.d.ts")
        .expect("sdk_types")
        .contents;
    for needle in [
        "export type FamilyId",
        "export interface ArtifactRef",
        "export type RuntimeProfile",
        "export interface RuntimeCapabilities",
        "export interface CanonicalEventEnvelope",
        "export type YouEventName",
        "export interface FabricPlacement",
        "export interface ToolEffectSet",
        "export type ContentVerification",
    ] {
        assert!(ts.contains(needle), "sdk_types.d.ts missing {needle}");
    }
    assert!(ts.contains("ObjectAdded"));
    assert!(ts.contains("HandoffCreated"));
    assert_eq!(ts.matches("export type YouEventName").count(), 1);
}
#[test]
fn honest_family_grades_match_evidence_ladder() {
    let expected = [
        ("deepseek", "SOURCE_HEADER_VALIDATED"),
        ("gemma", "DECLARED"),
        ("glm", "SMALL_REAL_CHECKPOINT"),
        ("kimi", "SYNTHETIC_PARITY"),
        ("llama", "SOURCE_HEADER_VALIDATED"),
        ("minimax", "DECLARED"),
        ("mistral_mixtral", "SOURCE_HEADER_VALIDATED"),
        ("phi", "DECLARED"),
        ("qwen", "SOURCE_HEADER_VALIDATED"),
        ("state_space", "DECLARED"),
    ];
    let r = hawking_adapters::builtin_registry();
    for (id, level) in expected {
        let d = r.get(id).unwrap_or_else(|| panic!("missing family {id}"));
        assert_eq!(
            d.level.as_str(),
            level,
            "family {id} grade drifted from the honest evidence ladder"
        );
    }
}

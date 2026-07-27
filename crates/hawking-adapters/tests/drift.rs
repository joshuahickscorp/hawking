//! Drift test: regenerating schemas/types produces no diff against goldens.
//!
//! Also checks repo-root deliverables when present, so hand-edits to
//! `HAWKING_ADAPTER_*.json` fail until `hawking-adapters-codegen` is re-run.

use std::path::PathBuf;

use hawking_adapters::generate::{generate_all, repo_root_artifacts};

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
fn repo_root_deliverables_match_registry() {
    let root = workspace_root();
    let mut failures = Vec::new();
    for (name, contents) in repo_root_artifacts() {
        // Bridge surface is optional lockstep; events + adapter deliverables required.
        if name == "HAWKING_CANONICAL_EVENTS.json" {
            // Still check if present; codegen writes it.
        }
        let path = root.join(name);
        if !path.exists() {
            failures.push(format!(
                "missing repo-root deliverable {name} — run hawking-adapters-codegen"
            ));
            continue;
        }
        let on_disk = std::fs::read_to_string(&path).expect("read root deliverable");
        if on_disk != contents {
            failures.push(format!(
                "drift in repo-root {name} ({} bytes on disk vs {} generated)",
                on_disk.len(),
                contents.len()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "repo-root deliverable drift:\n{}",
        failures.join("\n")
    );
}

#[test]
fn regenerating_produces_no_diff() {
    // Semantic alias of the two checks above: one assertion site for the contract.
    generated_artifacts_match_checked_in();
    repo_root_deliverables_match_registry();
}

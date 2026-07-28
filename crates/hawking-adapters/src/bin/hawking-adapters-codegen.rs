//! Codegen entry: regenerate checked-in adapter/event/CLI/schema artifacts.
//!
//! Mirrors `hide-sdk-codegen`: pure deterministic write of goldens; drift tests
//! fail when regeneration would change bytes. **This is the one adapter
//! codegen** — extend it; do not fork a second system.
//!
//! Writes:
//! - `crates/hawking-adapters/generated/*` (schemas, CLI surface, completion,
//!   SDK types, matrices, events, bridge, migrations)
//! - repo-root: `HAWKING_ADAPTER_{ABI,REGISTRY,CAPABILITY_MATRIX,TEST_MATRIX,MIGRATION_MAP}.json`
//! - repo-root: `HAWKING_CANONICAL_EVENTS.json`, `HAWKING_BRIDGE_SURFACE.json`,
//!   `HAWKING_CLI_SURFACE.json`, `HAWKING_SCHEMA_MIGRATIONS.json`

use std::path::PathBuf;

use anyhow::{Context, Result};

fn main() -> Result<()> {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let written = hawking_adapters::generate::write_all(&crate_root)
        .context("write generated/ under hawking-adapters")?;
    for p in &written {
        println!(
            "wrote {} ({} bytes)",
            p.display(),
            std::fs::metadata(p)?.len()
        );
    }

    let repo_root = crate_root
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .context("resolve repo root")?;

    for (name, contents) in hawking_adapters::generate::repo_root_artifacts() {
        let path = repo_root.join(name);
        std::fs::write(&path, &contents)
            .with_context(|| format!("write {}", path.display()))?;
        println!("wrote {} ({} bytes)", path.display(), contents.len());
    }

    Ok(())
}

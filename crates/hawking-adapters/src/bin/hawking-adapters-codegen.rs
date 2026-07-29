//! Codegen entry: regenerate checked-in adapter/event/CLI/schema artifacts.
//!
//! Mirrors `hide-sdk-codegen`: pure deterministic write of goldens; drift tests
//! fail when regeneration would change bytes. **This is the one adapter
//! codegen** — extend it; do not fork a second system.
//!
//! Writes only `crates/hawking-adapters/generated/*` (schemas, CLI surface,
//! completion, SDK types, matrices, events, bridge, migrations). Repo-root
//! duplicates are not published — one location only.

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

    Ok(())
}

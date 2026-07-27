//! Codegen entry point: regenerate the committed protocol artifacts from the
//! ONE schema source.
//!
//! Writes these files under the crate:
//!
//! - `goldens/protocol.schema.json` - the protocol JSON Schema bundle;
//! - `goldens/protocol.d.ts` - the generated frontend TypeScript types;
//! - `goldens/command_catalog.json` - the serialized ONE command registry;
//! - `goldens/commands.d.ts` - the CommandSpec type plus the catalog array;
//! - `fixtures/events.json` - the canonical event fixtures.
//!
//! The golden-file tests regenerate these in memory and compare against the
//! committed copies, so a protocol change that would break the frontend fails
//! the build until these are refreshed by running this binary.
//!
//! Model-free: pure deterministic codegen, no network and no model.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};

fn main() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let goldens = root.join("goldens");
    let fixtures = root.join("fixtures");
    fs::create_dir_all(&goldens).context("create goldens dir")?;
    fs::create_dir_all(&fixtures).context("create fixtures dir")?;

    let artifacts: [(PathBuf, String); 5] = [
        (
            goldens.join("protocol.schema.json"),
            hide_sdk::schema::protocol_schema_json(),
        ),
        (
            goldens.join("protocol.d.ts"),
            hide_sdk::ts::protocol_typescript(),
        ),
        (
            goldens.join("command_catalog.json"),
            hide_sdk::command::command_catalog_json(),
        ),
        (
            goldens.join("commands.d.ts"),
            hide_sdk::command::command_typescript(),
        ),
        (
            fixtures.join("events.json"),
            hide_sdk::fixtures::events_json(),
        ),
    ];

    for (path, contents) in &artifacts {
        fs::write(path, contents).with_context(|| format!("write {}", path.display()))?;
        println!("wrote {} ({} bytes)", path.display(), contents.len());
    }
    Ok(())
}

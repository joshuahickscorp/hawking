//! Print a deterministic, header-only source byte receipt.
//!
//! Usage:
//! `cargo run -p hawking-core --example gravity_safetensors_inventory -- /path/to/model`
//! This never reads a tensor payload and does not admit or execute a model.

use hawking_core::safetensors_inventory::SafetensorsInventory;
use serde_json::json;
use std::{env, process::ExitCode};

fn main() -> ExitCode {
    let Some(root) = env::args_os().nth(1) else {
        eprintln!("usage: gravity_safetensors_inventory <model-directory>");
        return ExitCode::from(2);
    };
    match SafetensorsInventory::scan_dir(&root) {
        Ok(inventory) => {
            let shards = inventory
                .shards
                .iter()
                .map(|shard| {
                    json!({
                        "path": shard.path,
                        "file_bytes": shard.file_bytes,
                        "header_bytes": shard.header_bytes,
                        "payload_bytes": shard.payload_bytes,
                        "tensor_count": shard.tensors.len(),
                    })
                })
                .collect::<Vec<_>>();
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "schema": "HAWKING_GRAVITY_SAFETENSORS_INVENTORY_V1",
                    "scope": "header_only; payloads_not_read; not_admission_or_execution",
                    "root": inventory.root,
                    "file_bytes": inventory.file_bytes,
                    "header_bytes": inventory.header_bytes,
                    "payload_bytes": inventory.payload_bytes,
                    "tensor_count": inventory.tensor_count,
                    "shard_count": inventory.shards.len(),
                    "shards": shards,
                }))
                .expect("serializing a primitive inventory receipt cannot fail")
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("gravity safetensors inventory refused: {error}");
            ExitCode::from(1)
        }
    }
}

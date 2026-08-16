//! Open a packed Q80 mixed catalog and print the on-disk BPW ledger.
//!
//! Does not generate tokens and does not run Metal.

use hawking_core::model::qwen80_mixed_catalog::{
    Qwen80MixedStreamingCatalog, QWEN80_MIXED_EXPECTED_TENSOR_COUNT,
};
use serde_json::json;
use std::env;
use std::path::PathBuf;
use std::process;

fn main() {
    let root = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(Qwen80MixedStreamingCatalog::default_root_hint())
        });
    if !root.is_absolute() {
        eprintln!("usage: ascension_qwen80_mixed_catalog_inspect ABSOLUTE_ROOT");
        process::exit(2);
    }
    let catalog = Qwen80MixedStreamingCatalog::open(&root).unwrap_or_else(|error| {
        eprintln!("qwen80 mixed catalog: {error}");
        process::exit(2);
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "root": catalog.root,
            "manifest_path": catalog.manifest_path,
            "manifest_seal_sha256": catalog.manifest_seal_sha256,
            "catalog_sha256": catalog.catalog_sha256,
            "tensor_count": catalog.tensor_count(),
            "expected_full_tensor_count": QWEN80_MIXED_EXPECTED_TENSOR_COUNT,
            "complete_physical_bpw": catalog.complete_physical_bpw,
            "tensor_payload_bytes": catalog.tensor_payload_bytes,
            "segment_count": catalog.segments().len(),
            "full_catalog": catalog.tensor_count() == QWEN80_MIXED_EXPECTED_TENSOR_COUNT,
            "claim_boundary": {
                "did_not_generate": true,
                "did_not_run_metal": true,
                "packing_is_not_a_coherence_claim": true,
            },
        }))
        .expect("json")
    );
}

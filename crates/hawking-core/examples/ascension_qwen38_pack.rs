//! Language-only Qwen3.8 pack: skip vision, fuse split in_proj, uniform Q4.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core --example ascension_qwen38_pack
//! workspace/ops/build/rust/release-fast/examples/ascension_qwen38_pack \
//!   --source /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16 \
//!   --root /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1
//! ```

use hawking_core::model::qwen38_pack::{pack_qwen38_language_uniform_q4, Qwen38PackRequest};
use serde_json::json;
use std::env;
use std::path::PathBuf;
use std::process;

fn usage() -> &'static str {
    "usage: ascension_qwen38_pack --source DIR --root DIR"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("ascension_qwen38_pack: {message}");
    process::exit(2);
}

fn main() {
    let mut source = None;
    let mut root = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--source" => {
                source = Some(PathBuf::from(
                    args.next().unwrap_or_else(|| fail(usage())),
                ));
            }
            "--root" => {
                root = Some(PathBuf::from(
                    args.next().unwrap_or_else(|| fail(usage())),
                ));
            }
            other => fail(format!("unknown argument {other}; {}", usage())),
        }
    }
    let source = source.unwrap_or_else(|| fail(usage()));
    let root = root.unwrap_or_else(|| fail(usage()));
    let report = pack_qwen38_language_uniform_q4(&Qwen38PackRequest {
        source_dir: source,
        output_root: root,
    })
    .unwrap_or_else(|error| fail(error));
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "ok",
            "manifest_path": report.manifest_path,
            "tensor_count": report.tensor_count,
            "q4_tensors": report.q4_tensors,
            "f32_tensors": report.f32_tensors,
            "source_weight_elements": report.source_weight_elements,
            "tensor_payload_bytes": report.tensor_payload_bytes,
            "complete_physical_bpw": report.complete_physical_bpw,
            "fused_in_proj_layers": report.fused_in_proj_layers,
            "skipped_vision_tensors": report.skipped_vision_tensors,
            "min_q4_cosine": report.min_q4_cosine,
        }))
        .expect("report JSON")
    );
}

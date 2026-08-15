//! Stream-pack Qwen3-Coder-Next as uniform Q4 group-64 (HQ30UQ4).
//!
//! W3.T4 full-catalog entrypoint. Reads one source tensor at a time from the
//! sealed safetensors shards, writes one `.hq80uq4` payload, then evicts the
//! BF16/f32 workspace. Does not hold the ~160 GB source and ~42 GB Q4 catalog
//! resident together. Does not generate tokens, touch the Q30 path, or run
//! Metal.
//!
//! Expected full-catalog output: 74,391 tensors, ~42 GB
//! (`79.67e9 * 4.256 / 8`), under
//! `quality-candidates/uniform-q4-group64-v1`.
//!
//! ```text
//! CARGO_TARGET_DIR=<worktree>/target-t4 cargo run --release -p hawking-core \
//!   --example ascension_qwen80_uniform_q4_repack -j6 -- \
//!   --revalidation ABSOLUTE_PATH/QWEN80_CURRENT_SOURCE_SHARD_REVALIDATION.json \
//!   --root ABSOLUTE_PATH/quality-candidates/uniform-q4-group64-v1 \
//!   --workers 6
//! ```
//!
//! Omit `--allow-partial-catalog`. W3.T4 must pack the full 74,391-tensor
//! source index. This example refuses a short index unless that flag is set
//! (authoring / unit-test fixtures only).

use hawking_core::model::qwen_complete_binary::{
    pack_qwen80_uniform_q4, Qwen80UniformQ4PackRequest, QWEN80_UNIFORM_Q4_CAMPAIGN_BPW,
    QWEN80_UNIFORM_Q4_EXPECTED_TENSOR_COUNT, UNIFORM_Q4_NOMINAL_BPW,
};
use serde_json::json;
use std::env;
use std::path::PathBuf;
use std::process;

fn usage() -> &'static str {
    "usage: ascension_qwen80_uniform_q4_repack \
        --revalidation ABSOLUTE_PATH \
        --root ABSOLUTE_PATH \
        [--workers 1..N] [--allow-partial-catalog]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("ascension_qwen80_uniform_q4_repack: {message}");
    process::exit(2);
}

fn main() {
    let mut revalidation = None;
    let mut root = None;
    let mut workers = 6usize;
    let mut allow_partial = false;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--allow-partial-catalog" => allow_partial = true,
            "--revalidation" => {
                let value = args
                    .next()
                    .unwrap_or_else(|| fail(format!("missing value for {flag}; {}", usage())));
                if revalidation.replace(PathBuf::from(value)).is_some() {
                    fail(format!("--revalidation was supplied more than once; {}", usage()));
                }
            }
            "--root" => {
                let value = args
                    .next()
                    .unwrap_or_else(|| fail(format!("missing value for {flag}; {}", usage())));
                if root.replace(PathBuf::from(value)).is_some() {
                    fail(format!("--root was supplied more than once; {}", usage()));
                }
            }
            "--workers" => {
                let value = args
                    .next()
                    .unwrap_or_else(|| fail(format!("missing value for {flag}; {}", usage())));
                workers = value.parse().unwrap_or_else(|_| {
                    fail("--workers must be a positive integer");
                });
                if workers == 0 {
                    fail("--workers must be at least 1");
                }
            }
            other => fail(format!("unknown argument {other:?}; {}", usage())),
        }
    }
    let revalidation = revalidation.unwrap_or_else(|| fail(usage()));
    let root = root.unwrap_or_else(|| fail(usage()));
    if !revalidation.is_absolute() || !root.is_absolute() {
        fail("--revalidation and --root must be absolute paths");
    }

    let report = pack_qwen80_uniform_q4(&Qwen80UniformQ4PackRequest {
        output_root: root,
        revalidation_path: revalidation,
        require_full_qwen80_catalog: !allow_partial,
        workers,
    })
    .unwrap_or_else(|error| fail(error));

    let physical_bpw = if report.source_weight_elements == 0 {
        0.0
    } else {
        (report.tensor_payload_bytes as f64 * 8.0) / report.source_weight_elements as f64
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "ok",
            "manifest_path": report.manifest_path,
            "terminal_path": report.terminal_path,
            "manifest_seal_sha256": report.manifest_seal_sha256,
            "terminal_seal_sha256": report.terminal_seal_sha256,
            "tensor_count": report.tensor_count,
            "expected_full_tensor_count": QWEN80_UNIFORM_Q4_EXPECTED_TENSOR_COUNT,
            "source_weight_elements": report.source_weight_elements,
            "tensor_payload_bytes": report.tensor_payload_bytes,
            "mean_component_cosine": report.mean_component_cosine,
            "nominal_codec_bpw": UNIFORM_Q4_NOMINAL_BPW,
            "campaign_bpw": QWEN80_UNIFORM_Q4_CAMPAIGN_BPW,
            "payload_only_physical_bpw": physical_bpw,
            "full_catalog": report.tensor_count == QWEN80_UNIFORM_Q4_EXPECTED_TENSOR_COUNT,
            "streaming_contract": "one_tensor_range_read_quantize_write_evict",
            "claim_boundary": {
                "did_not_resident_load_160gb_bf16_and_42gb_q4": true,
                "did_not_generate": true,
                "did_not_change_q30_path_or_runtime": true,
            },
        }))
        .expect("report JSON")
    );
}

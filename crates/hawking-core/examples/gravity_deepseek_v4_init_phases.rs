//! Init-phase breakdown for the DSV4F sealed artifact. No Metal, no token.
//!
//!   HAWKING_STARTUP_TIMING=1 cargo run --profile release-fast -p hawking-core \
//!     --example gravity_deepseek_v4_init_phases -- --artifact <gravity>

use hawking_core::gravity_deepseek_v4::DeepSeekV4FullStreamReader;
use hawking_core::gravity_deepseek_v4_layer_source_anchors::verify_deepseek_v4_layer_source_anchors;
use hawking_core::gravity_deepseek_v4_streamed_forward::{
    discover_sealed_dsv4f_artifact, open_admitted_dsv4f_reader,
};
use serde_json::json;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    hawking_core::startup_timing::mark_process_start();
    let mut artifact = None;
    let mut write_index = false;
    let mut compare = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--artifact" => {
                artifact = Some(PathBuf::from(args.next().ok_or("--artifact needs a path")?))
            }
            "--verify" => {
                let mode = args.next().ok_or("--verify needs full|admission")?;
                hawking_core::gravity_deepseek_v4::DeepSeekV4VerifyMode::parse(&mode)
                    .map_err(|e| format!("{e}"))?;
                std::env::set_var("HAWKING_DSV4F_VERIFY", mode);
            }
            "--write-index" => write_index = true,
            "--compare" => compare = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let artifact = match artifact {
        Some(path) => path,
        None => discover_sealed_dsv4f_artifact()
            .ok_or("no --artifact given and no sealed full-43-layer-stream.gravity found")?,
    };
    eprintln!("dsv4f init phases: artifact={}", artifact.display());
    let wall = Instant::now();
    let (reader, admission) = open_admitted_dsv4f_reader(&artifact)?;
    let anchors = hawking_core::startup_timing::time_ms_result("layer_source_anchors", || {
        verify_deepseek_v4_layer_source_anchors(&reader)
    })?;
    let _ = anchors.layer(0)?;
    let embed = reader.tensor_metadata("embed.weight")?;
    let mut index_seal = None;
    if write_index {
        let written = reader.write_artifact_index_from_admission(&admission.source_path)?;
        eprintln!(
            "wrote artifact index {} bytes={} wall_ms={}",
            written.path.display(),
            written.bytes,
            written.wall_ms
        );
        index_seal = Some(json!({
            "path": written.path,
            "bytes": written.bytes,
            "wall_ms": written.wall_ms,
            "tensor_count": written.tensor_count,
            "chunk_count": written.chunk_count,
            "index_seal_sha256": written.index_seal_sha256,
        }));
    }
    let mut compare_result = None;
    if compare {
        let other = if reader.chunk_verification_stats().artifact_index_loaded {
            std::env::set_var("HAWKING_DSV4F_INDEX", "0");
            let view =
                hawking_core::gravity_deepseek_v4_streamed_forward::prepare_sealed_admission_root(
                    &admission.source_path,
                )?;
            DeepSeekV4FullStreamReader::admit(&view.path)?
        } else {
            DeepSeekV4FullStreamReader::try_admit_from_artifact_index(
                &admission.source_path,
                hawking_core::gravity_deepseek_v4::DeepSeekV4VerifyMode::Admission,
            )?
            .ok_or("compare requires a loadable artifact index (pass --write-index first)")?
        };
        reader.structural_map_eq(&other)?;
        compare_result = Some(json!({
            "status": "STRUCTURALLY_IDENTICAL",
            "tensor_count": reader.tensor_count(),
            "chunk_count": reader.chunk_count(),
            "left_index_loaded": reader.chunk_verification_stats().artifact_index_loaded,
            "right_index_loaded": other.chunk_verification_stats().artifact_index_loaded,
        }));
        eprintln!("tensor map structurally identical between index and JSON paths");
    }
    let wall_ms = wall.elapsed().as_millis();
    hawking_core::startup_timing::emit_stderr_json();
    let snap = hawking_core::startup_timing::snapshot();
    let stats = reader.chunk_verification_stats();
    let out = json!({
        "schema": "hawking.gravity.deepseek_v4.init_phases.v1",
        "artifact": artifact.display().to_string(),
        "admission_view": admission.view,
        "tensor_count": reader.tensor_count(),
        "chunk_count": reader.chunk_count(),
        "embed_bytes": embed.bytes,
        "admission_receipt_loaded": stats.admission_receipt_loaded,
        "artifact_index_loaded": stats.artifact_index_loaded,
        "index": index_seal,
        "compare": compare_result,
        "wall_ms": wall_ms,
        "startup_timing": snap.to_json(),
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

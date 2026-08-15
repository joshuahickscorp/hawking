//! Host-CPU streamed BOS forward over the sealed DeepSeek-V4-Flash 43-layer
//! source stream.
//!
//! This is **not** native, not an Engine, and not a TPS/coherence claim.
//! It loads one operator, executes it, frees it, checkpoints the HC residual,
//! and records measured peak RSS.
//!
//! Usage:
//!   cargo run -p hawking-core --release --example gravity_deepseek_v4_streamed_forward -- \
//!     --artifact /path/to/full-43-layer-stream.gravity \
//!     --out receipts/dsv4f_streamed_forward_l0_l42_receipt.json \
//!     --checkpoint receipts/dsv4f_streamed_forward.ckpt.json \
//!     [--max-layer 42] [--resume] [--skip-head] [--metal]
//!
//! If `--artifact` is omitted, the example searches HAWKING_DSV4F_ARTIFACT
//! and the known sealed locations.

use hawking_core::gravity_deepseek_v4_streamed_forward::{
    discover_sealed_dsv4f_artifact, run_streamed_forward, StreamedForwardConfig,
    STREAMED_EXECUTION_PATH,
};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    artifact: PathBuf,
    out: PathBuf,
    checkpoint: Option<PathBuf>,
    max_layer: usize,
    resume: bool,
    skip_head: bool,
    metal: bool,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut artifact = None;
    let mut out = None;
    let mut checkpoint = None;
    let mut max_layer = 42usize;
    let mut resume = false;
    let mut skip_head = false;
    let mut metal = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--artifact" => {
                artifact = Some(PathBuf::from(args.next().ok_or("--artifact needs a path")?))
            }
            "--out" => out = Some(PathBuf::from(args.next().ok_or("--out needs a path")?)),
            "--checkpoint" => {
                checkpoint = Some(PathBuf::from(
                    args.next().ok_or("--checkpoint needs a path")?,
                ))
            }
            "--max-layer" => {
                max_layer = args
                    .next()
                    .ok_or("--max-layer needs a value")?
                    .parse()
                    .map_err(|_| "--max-layer must be an integer")?;
            }
            "--resume" => resume = true,
            "--skip-head" => skip_head = true,
            "--metal" => metal = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let artifact = match artifact {
        Some(path) => path,
        None => discover_sealed_dsv4f_artifact()
            .ok_or("no --artifact given and no sealed full-43-layer-stream.gravity found")?,
    };
    Ok(Args {
        artifact,
        out: out.ok_or("--out is required")?,
        checkpoint,
        max_layer,
        resume,
        skip_head,
        metal,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    if args.max_layer > 42 {
        return Err("--max-layer must be in 0..42".into());
    }
    let wall = Instant::now();
    eprintln!(
        "dsv4f streamed forward: path={STREAMED_EXECUTION_PATH} artifact={} max_layer={} resume={} head={} metal={}",
        args.artifact.display(),
        args.max_layer,
        args.resume,
        !args.skip_head,
        args.metal
    );

    let mut config = StreamedForwardConfig::for_layers(args.max_layer)?;
    config.checkpoint_path = args.checkpoint.clone();
    config.resume = args.resume;
    config.compute_final_head = !args.skip_head;
    config.use_metal = args.metal;

    let report = run_streamed_forward(&args.artifact, config)?;
    let mut receipt = report.to_receipt_json();
    receipt["example_wall_ms"] = serde_json::json!(wall.elapsed().as_millis());
    let encoded = serde_json::to_vec_pretty(&receipt)?;
    let digest = format!("{:x}", Sha256::digest(&encoded));
    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let tmp = args.out.with_extension("json.tmp");
    {
        let mut file = fs::File::create(&tmp)?;
        file.write_all(&encoded)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    fs::rename(&tmp, &args.out)?;

    eprintln!(
        "deepest_layer={:?} layers={:?} peak_rss_bytes={} peak_weight_bytes={} rss_ok={} weight_ok={} greedy={:?} metal_dispatches={} fallbacks={} stop={:?} receipt_sha256={digest} out={}",
        report.deepest_layer,
        report.layers_executed,
        report.peak_rss_bytes,
        report.peak_weight_resident_bytes,
        report.rss_within_bound,
        report.weight_within_bound,
        report.greedy.as_ref().map(|g| (g.token_id, g.logit)),
        report.honesty.metal_dispatches,
        report.honesty.fallbacks,
        report.stop_reason,
        args.out.display()
    );
    eprintln!("operator_profile:");
    for row in report.operator_profile.to_sorted_rows() {
        eprintln!(
            "  {:>8.3}s  {:>6.2}%  {:>6}  {}",
            row.seconds, row.percent, row.calls, row.name
        );
    }
    if let Some(reason) = report.stop_reason {
        return Err(format!("streamed forward stopped: {reason}").into());
    }
    if report.deepest_layer != Some(args.max_layer) {
        let head_only_from_complete_checkpoint =
            args.resume && report.layers_executed.is_empty() && report.greedy.is_some();
        if !head_only_from_complete_checkpoint {
            return Err(format!(
                "streamed forward deepest_layer {:?} != requested {}",
                report.deepest_layer, args.max_layer
            )
            .into());
        }
    }
    Ok(())
}

//! Minimum complete native BOS token graph for DeepSeek-V4-Flash.
//!
//! This is not an Engine, BASE_TRUE_TPS, coherence, or tournament claim.
//! It builds one BOS token with a compact top-6 device worklist and batched
//! Metal command buffers, then records measured wall time, dispatches, RSS,
//! and the parity triple against the host CPU oracle.
//!
//! Usage:
//!   cargo run -p hawking-core --release --example gravity_deepseek_v4_native_token_graph -- \
//!     [--artifact /path/to/full-43-layer-stream.gravity] \
//!     [--out receipts/dsv4f_native_token_graph_l0_l42_receipt.json] \
//!     [--max-layer 42] [--skip-head] [--verify full|admission]

use hawking_core::gravity_deepseek_v4_native_token_graph::{
    run_native_bos_token, ORACLE_GREEDY_LOGIT, ORACLE_GREEDY_TOKEN_ID, ORACLE_HC_BF16_SHA256,
};
use hawking_core::gravity_deepseek_v4_streamed_forward::discover_sealed_dsv4f_artifact;
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

struct Args {
    artifact: PathBuf,
    out: Option<PathBuf>,
    max_layer: usize,
    skip_head: bool,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut artifact = None;
    let mut out = None;
    let mut max_layer = 42usize;
    let mut skip_head = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--artifact" => {
                artifact = Some(PathBuf::from(args.next().ok_or("--artifact needs a path")?))
            }
            "--out" => out = Some(PathBuf::from(args.next().ok_or("--out needs a path")?)),
            "--max-layer" => {
                max_layer = args
                    .next()
                    .ok_or("--max-layer needs a value")?
                    .parse()
                    .map_err(|_| "--max-layer must be an integer")?;
            }
            "--skip-head" => skip_head = true,
            "--verify" => {
                let mode = args.next().ok_or("--verify needs full|admission")?;
                hawking_core::gravity_deepseek_v4::DeepSeekV4VerifyMode::parse(&mode)
                    .map_err(|e| format!("{e}"))?;
                std::env::set_var("HAWKING_DSV4F_VERIFY", mode);
            }
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
        out,
        max_layer,
        skip_head,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    hawking_core::startup_timing::mark_process_start();
    let args = parse_args()?;
    eprintln!(
        "dsv4f native token graph: artifact={} max_layer={} head={}",
        args.artifact.display(),
        args.max_layer,
        !args.skip_head
    );
    let report = run_native_bos_token(&args.artifact, args.max_layer, !args.skip_head)?;
    let json = serde_json::to_vec_pretty(&report.to_receipt_json())?;
    let digest = format!("{:x}", Sha256::digest(&json));
    if let Some(out) = args.out.as_ref() {
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = fs::File::create(out)?;
        file.write_all(&json)?;
        file.write_all(b"\n")?;
    }
    if let Some(ledger) = report.token_ns_ledger.as_ref() {
        let ledger_path = args
            .out
            .as_ref()
            .and_then(|path| path.parent().map(|parent| parent.join("DSV4F_TOKEN_NS_LEDGER.json")))
            .unwrap_or_else(|| PathBuf::from("receipts/DSV4F_TOKEN_NS_LEDGER.json"));
        if let Some(parent) = ledger_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let ledger_json = serde_json::to_vec_pretty(ledger)?;
        let mut file = fs::File::create(&ledger_path)?;
        file.write_all(&ledger_json)?;
        file.write_all(b"\n")?;
        eprintln!("wrote {}", ledger_path.display());
    }
    let greedy = report.greedy.as_ref();
    eprintln!(
        "deepest_layer={:?} layers={:?} wall_ms={} init_ms={} body_ms={} s/token={:.4} (body {:.4}) peak_rss_bytes={} peak_weight_bytes={} metal_dispatches={} command_buffers={} sync_points={} readbacks={} buffer_creates={} buffer_rebinds={} fallbacks={} host_expert_gather={} host_expert_output_readback={} host_route_id_readback={} hash_invocations={} admission_trust_hits={} bytes_hashed={} admission_receipt_loaded={} artifact_index_loaded={} hc_sha={} greedy={:?} oracle=(token={} logit={} sha={}) stop={:?} receipt_sha256={digest}",
        report.deepest_layer,
        report.layers_executed,
        report.wall_ms,
        report.init_ms,
        report.body_ms,
        report.wall_ms as f64 / 1000.0,
        report.body_ms as f64 / 1000.0,
        report.peak_rss_bytes,
        report.peak_weight_resident_bytes,
        report.counters.metal_dispatches,
        report.counters.command_buffers,
        report.counters.total_sync_points,
        report.counters.total_readbacks,
        report.counters.total_buffer_creations,
        report.counters.total_buffer_rebinds,
        report.counters.fallbacks,
        report.counters.host_expert_gather,
        report.counters.host_expert_output_readback,
        report.counters.host_route_id_readback,
        report.chunk_verification.hash_invocations,
        report.chunk_verification.admission_trust_hits,
        report.chunk_verification.bytes_hashed,
        report.chunk_verification.admission_receipt_loaded,
        report.chunk_verification.artifact_index_loaded,
        report.hc_bf16_sha256,
        greedy.map(|g| (g.token_id, g.logit)),
        ORACLE_GREEDY_TOKEN_ID,
        ORACLE_GREEDY_LOGIT,
        ORACLE_HC_BF16_SHA256,
        report.stop_reason,
    );
    if let Some(ledger) = report.token_ns_ledger.as_ref() {
        eprintln!(
            "token_ns_ledger diagnosis={:?} body_ns={} host_exclusive_ns={} metal_gpu_ns={} metal_wait_ns={} verify_ns={} cbs={} isolated={} inter_cb_gap_ns={} gpu_idle_frac={:?} encoders={} proof={}",
            ledger.diagnosis,
            ledger.body_ns,
            ledger.metal_vs_host.host_exclusive_ns,
            ledger.metal_vs_host.metal_gpu_ns,
            ledger.metal_vs_host.metal_wait_ns,
            ledger.verify_ns,
            ledger.metal_vs_host.production_command_buffers,
            ledger.isolated_kernels.len(),
            ledger.gpu_gaps.inter_cb_device_gap_ns,
            ledger.gpu_gaps.gpu_idle_fraction_of_span,
            ledger.gpu_gaps.production_encoders,
            ledger.diagnosis_proof,
        );
    }
    hawking_core::startup_timing::emit_stderr_json();
    if let Some(path) = args.out.as_ref() {
        eprintln!("wrote {}", path.display());
    }
    if let Some(reason) = report.stop_reason.as_ref() {
        return Err(reason.clone().into());
    }
    Ok(())
}

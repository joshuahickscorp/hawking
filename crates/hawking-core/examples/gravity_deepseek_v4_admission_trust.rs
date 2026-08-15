//! One-time parallel SHA-256 admission pass for a sealed DeepSeek-V4 stream.
//!
//! Pays the full per-chunk hash once and writes a sealed receipt. Prefers
//! `<artifact>/.hawking-admission.json`; if that directory is not writable
//! (this host's Downloads tree is TCC-locked) it falls back to
//! `~/.cache/hawking/dsv4f-admission/<manifest_seal>.json`. Subsequent
//! token reads with `HAWKING_DSV4F_VERIFY=admission` skip hashing chunks
//! whose cheap identity still matches.
//!
//! ```sh
//! cargo run --profile release-fast -p hawking-core --example gravity_deepseek_v4_admission_trust -- \
//!   --artifact /absolute/path/to/full-43-layer-stream.gravity \
//!   --out /absolute/path/to/receipts/dsv4f_admission_pass.json
//! ```

use hawking_core::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, DeepSeekV4VerifyMode, ADMISSION_TRUST_SCHEMA,
};
use hawking_core::gravity_deepseek_v4_streamed_forward::{
    discover_sealed_dsv4f_artifact, prepare_sealed_admission_root,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    artifact: PathBuf,
    out: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let wall = Instant::now();
    eprintln!(
        "dsv4f admission trust: artifact={} threads hint={}",
        args.artifact.display(),
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    );

    let admission = prepare_sealed_admission_root(&args.artifact)?;
    let reader = DeepSeekV4FullStreamReader::admit_with_verify_mode(
        &admission.path,
        DeepSeekV4VerifyMode::Full,
    )?;
    let seal = reader.seal_admission_trust_at(&admission.source_path)?;
    let stats = reader.chunk_verification_stats();

    let receipt = json!({
        "schema": "hawking.gravity.deepseek_v4.admission_trust_pass.v1",
        "status": "PASS_ADMISSION_TRUST_SEAL",
        "admission_receipt_schema": ADMISSION_TRUST_SCHEMA,
        "artifact": {
            "path": admission.source_path,
            "admission_view": admission.view,
            "manifest_seal_sha256": reader.manifest_seal_sha256(),
            "content_addressed_chunk_sha256": reader.content_addressed_chunk_sha256(),
        },
        "seal": {
            "path": seal.path,
            "table_sha256": seal.table_sha256,
            "seal_sha256": seal.seal_sha256,
            "verifier_version": seal.verifier_version,
            "chunk_count": seal.chunk_count,
            "total_bytes": seal.total_bytes,
            "bytes_hashed": seal.bytes_hashed,
            "threads": seal.threads,
            "hash_wall_ms": seal.hash_wall_ms,
            "seal_wall_ms": seal.wall_ms,
        },
        "reader_stats_after_seal": {
            "hash_invocations": stats.hash_invocations,
            "bytes_hashed": stats.bytes_hashed,
            "chunks_verified": stats.chunks_verified,
        },
        "example_wall_ms": wall.elapsed().as_millis(),
        "claim": "one-time full SHA-256 admission; not a token, forward, Engine, or TPS result",
    });
    let encoded = serde_json::to_vec_pretty(&receipt)?;
    let digest = format!("{:x}", Sha256::digest(&encoded));
    if let Some(out) = args.out.as_ref() {
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp = out.with_extension("json.tmp");
        {
            let mut file = fs::File::create(&tmp)?;
            file.write_all(&encoded)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
        }
        fs::rename(&tmp, out)?;
        eprintln!("wrote {}", out.display());
    }
    eprintln!(
        "status=PASS_ADMISSION_TRUST_SEAL receipt={} chunks={} bytes={} threads={} hash_wall_ms={} seal_wall_ms={} example_wall_ms={} digest={digest}",
        seal.path.display(),
        seal.chunk_count,
        seal.total_bytes,
        seal.threads,
        seal.hash_wall_ms,
        seal.wall_ms,
        wall.elapsed().as_millis(),
    );
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut artifact = None;
    let mut out = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--artifact" => {
                artifact = Some(PathBuf::from(args.next().ok_or("--artifact needs a path")?))
            }
            "--out" => out = Some(PathBuf::from(args.next().ok_or("--out needs a path")?)),
            "--help" | "-h" => {
                println!(
                    "usage: gravity_deepseek_v4_admission_trust [--artifact <full Gravity dir>] [--out <receipt.json>]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let artifact = match artifact {
        Some(path) => path,
        None => discover_sealed_dsv4f_artifact()
            .ok_or("no --artifact given and no sealed full-43-layer-stream.gravity found")?,
    };
    Ok(Args { artifact, out })
}

//! Native read-only control for Flash PLE address and source-row identity.
//!
//! This resolves the exact hashed row sequence through hawking-core and
//! range-reads only those BF16 rows. It does not evaluate PLE projections,
//! load a model, start Metal, or authorize a teacher/runtime.

use hawking_core::flash_ple::{
    FlashPleAddressConfig, FlashPleAddressContract, FlashPleLookupLayout,
};
use hawking_core::gravity_deepseek_v4::canonical_json;
use hawking_core::model::qwen80_source_bf16_layer_major::SourceBf16Index;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const SCHEMA: &str = "hawking.flash.ple_source_row_control.v1";
const STATUS: &str = "NATIVE_RUST_PLE_ADDRESS_AND_SOURCE_ROW_CONTROL__NO_PLE_NUMERICAL_EXECUTION";
const EXPECTED_CONFIG_SHA256: &str =
    "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b";
const EXPECTED_INDEX_SHA256: &str =
    "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de";
const SPLIT_PARTS: usize = 128;

#[derive(Debug)]
struct Args {
    root: PathBuf,
    tokens: Vec<u64>,
    out: PathBuf,
}

fn required(value: Option<String>, flag: &str) -> Result<String, Box<dyn Error>> {
    value.ok_or_else(|| format!("missing required {flag}").into())
}

fn parse_tokens(raw: &str) -> Result<Vec<u64>, Box<dyn Error>> {
    let values = raw
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::parse::<u64>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err("--tokens must contain at least one token ID".into());
    }
    Ok(values)
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut root = None;
    let mut tokens = None;
    let mut out = None;
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--root" => root = argv.next(),
            "--tokens" => tokens = argv.next(),
            "--out" => out = argv.next(),
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok(Args {
        root: PathBuf::from(required(root, "--root")?),
        tokens: parse_tokens(&required(tokens, "--tokens")?)?,
        out: PathBuf::from(required(out, "--out")?),
    })
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn file_binding(path: &Path, expected_sha256: &str) -> Result<Value, Box<dyn Error>> {
    if path.is_symlink() || !path.is_file() {
        return Err(format!(
            "required source identity file is absent or symlinked: {}",
            path.display()
        )
        .into());
    }
    let bytes = fs::read(path)?;
    let observed = sha256_bytes(&bytes);
    if observed != expected_sha256 {
        return Err(format!(
            "source identity differs for {}: expected={expected_sha256} observed={observed}",
            path.display()
        )
        .into());
    }
    Ok(json!({"path": path, "sha256": observed, "bytes": bytes.len()}))
}

fn seal(document: &mut Value) -> Result<String, Box<dyn Error>> {
    let object = document
        .as_object_mut()
        .ok_or("PLE source-row receipt must be a JSON object")?;
    if object.contains_key("seal_sha256") {
        return Err("PLE source-row receipt already contains a seal".into());
    }
    let digest = sha256_bytes(&canonical_json(document));
    document["seal_sha256"] = Value::String(digest.clone());
    Ok(digest)
}

fn write_new(path: &Path, document: &Value) -> Result<(), Box<dyn Error>> {
    if path.is_symlink() {
        return Err(format!("receipt path may not be a symlink: {}", path.display()).into());
    }
    let parent = path.parent().ok_or("receipt path has no parent")?;
    fs::create_dir_all(parent)?;
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    file.write_all(&serde_json::to_vec_pretty(document)?)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

fn run(args: Args) -> Result<(Value, String), Box<dyn Error>> {
    let started = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let root = fs::canonicalize(&args.root)?;
    let config_binding = file_binding(&root.join("config.json"), EXPECTED_CONFIG_SHA256)?;
    let index_binding = file_binding(
        &root.join("model.safetensors.index.json"),
        EXPECTED_INDEX_SHA256,
    )?;
    let contract =
        FlashPleAddressContract::from_config(FlashPleAddressConfig::qwen38_flash_next_layer1())?;
    let source = SourceBf16Index::open(&root)?;
    let layout = FlashPleLookupLayout::from_source_index(&source, &contract, SPLIT_PARTS)?;
    let mut state = contract.new_state();
    let accesses = contract.trace_segment(&mut state, &args.tokens)?;
    let mut token_records = Vec::with_capacity(accesses.len());
    let mut whole_payload = Sha256::new();
    for access in &accesses {
        let rows = layout.read_token_embedding_bf16(&source, &contract, access)?;
        let mut embedding = Vec::with_capacity(contract.config.ple_embed_dim * 2);
        let row_records = rows
            .into_iter()
            .map(|row| {
                whole_payload.update(&row.bf16_bytes);
                embedding.extend_from_slice(&row.bf16_bytes);
                json!({
                    "global_head": row.global_head,
                    "combined_row": row.combined_row,
                    "shard_ordinal": row.shard_ordinal,
                    "tensor": row.tensor_name,
                    "local_row": row.local_row,
                    "row_bytes": row.bf16_bytes.len(),
                    "payload_sha256": sha256_bytes(&row.bf16_bytes),
                })
            })
            .collect::<Vec<_>>();
        if embedding.len() != contract.config.ple_embed_dim * 2 {
            return Err("ordered PLE rows did not reconstruct one complete BF16 embedding".into());
        }
        token_records.push(json!({
            "session_token_index": access.session_token_index,
            "token_id": access.token_id,
            "embedding_bf16_bytes": embedding.len(),
            "embedding_bf16_sha256": sha256_bytes(&embedding),
            "rows": row_records,
        }));
    }
    let finished = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let expected_payload_bytes = args.tokens.len() * contract.ngram_heads() * layout.row_bytes;
    if source.bytes_read_total() != expected_payload_bytes as u64 {
        return Err(format!(
            "source payload accounting differs: observed={} expected={expected_payload_bytes}",
            source.bytes_read_total()
        )
        .into());
    }
    let mut receipt = json!({
        "schema": SCHEMA,
        "status": STATUS,
        "started_unix_ns": started,
        "finished_unix_ns": finished,
        "elapsed_ns": finished - started,
        "source": {
            "root": root,
            "model": "Qwen/Qwen3.8-Flash-Next",
            "pinned_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
            "config": config_binding,
            "safetensors_index": index_binding,
        },
        "address_contract": {
            "source_layer_index": contract.config.source_layer_index,
            "source_ple_layer_id_one_indexed": contract.config.source_layer_index + 1,
            "vocab_size": contract.config.vocab_size,
            "eos_token_id": contract.config.eos_token_id,
            "ngram_size": contract.config.ngram_size,
            "heads_per_ngram": contract.config.heads_per_ngram,
            "head_count": contract.ngram_heads(),
            "head_dim": contract.head_dim(),
            "multipliers": contract.multipliers,
            "head_vocab_sizes": contract.head_vocab_sizes,
            "head_offsets": contract.head_offsets,
            "total_vocab_size": contract.total_vocab_size,
            "padded_vocab_size": contract.padded_vocab_size,
            "eos_aware_two_token_state": true,
        },
        "layout": {
            "shard_count": layout.shards.len(),
            "combined_rows": layout.combined_rows,
            "row_bytes": layout.row_bytes,
            "full_lookup_tensor_materialized": false,
        },
        "tokens": token_records,
        "payload_accounting": {
            "source_range_reads": args.tokens.len() * contract.ngram_heads(),
            "source_payload_bytes_read": source.bytes_read_total(),
            "expected_source_payload_bytes": expected_payload_bytes,
            "ordered_all_token_rows_sha256": format!("{:x}", whole_payload.finalize()),
        },
        "model_loaded": false,
        "gpu_or_metal_started": false,
        "ple_numerical_execution_performed": false,
        "promotion_allowed": false,
        "claim_boundary": "Native Rust owns exact PLE address/history and range-read row identity for the pinned source. This does not evaluate projections, norms, gating, convolution, or injection; establish source/full-model parity; construct a compact PLE/NR; measure EBPW/TPS/capability; deploy; promote Pulsar; or retire Kimi.",
    });
    let digest = seal(&mut receipt)?;
    write_new(&args.out, &receipt)?;
    Ok((receipt, digest))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let out = args.out.clone();
    let (_, seal) = run(args)?;
    println!(
        "status={STATUS} receipt={} seal_sha256={seal}",
        out.display()
    );
    Ok(())
}

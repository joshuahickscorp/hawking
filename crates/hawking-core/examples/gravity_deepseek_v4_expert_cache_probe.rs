//! Read-only bounded cache probe for DeepSeek-V4 routed expert source chunks.
//!
//! This example intentionally exercises storage admission, verified full-tensor
//! reads, hot/cold LRU transitions, and byte accounting only.  It does not
//! execute a router, model layer, token, GPU kernel, engine, endpoint, HCLI
//! request, or TPS measurement.
//!
//! ```sh
//! cargo run --release -p hawking-core --example gravity_deepseek_v4_expert_cache_probe -- \
//!   --artifact /absolute/path/to/full-43-layer-stream.gravity \
//!   --repo-root /absolute/path/to/hawking \
//!   --out /absolute/path/to/DSV4F_EXPERT_CACHE_PROBE-v1.json
//! ```

use hawking_core::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, NativeScalePairKind, FULL_STREAM_SCHEMA, FULL_STREAM_STATUS,
};
use hawking_core::gravity_deepseek_v4_expert_cache::{
    resolve_expert_bundle, DeepSeekV4ExpertBundleCache, ExpertBundleDescriptor, ExpertBundleKey,
    ExpertBundleSourceRead, ExpertCacheAccess, ExpertCacheCounters, ExpertCacheState,
    ExpertOperatorDescriptor, DSV4F_LAYER_COUNT, DSV4F_ROUTED_EXPERT_COUNT,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const RECEIPT_SCHEMA: &str = "hawking.gravity.deepseek_v4.expert_cache_probe.v1";
const RECEIPT_STATUS: &str = "PASS_BOUNDED_SOURCE_CHUNK_EXPERT_CACHE_PROBE_STORAGE_ONLY";
const PROBE_LAYER: u16 = 4;
const DEFAULT_HOT_BUNDLES: u64 = 2;
const DEFAULT_COLD_BUNDLES: u64 = 1;

type ExampleResult<T> = Result<T, Box<dyn Error>>;

struct Args {
    artifact: PathBuf,
    repo_root: PathBuf,
    out: PathBuf,
    hot_bundles: u64,
    cold_bundles: u64,
}

fn main() -> ExampleResult<()> {
    let args = parse_args()?;
    let reader = DeepSeekV4FullStreamReader::admit(&args.artifact)?;
    reject_output_inside_artifact(&args.out, reader.artifact_root())?;
    let manifest_before = sha256_file(&reader.artifact_root().join("manifest.json"))?;

    let probe_descriptor = resolve_expert_bundle(&reader, ExpertBundleKey::new(PROBE_LAYER, 205))?;
    let hot_capacity_bytes = checked_mul(
        probe_descriptor.payload_bytes,
        args.hot_bundles,
        "hot cache capacity",
    )?;
    let cold_capacity_bytes = checked_mul(
        probe_descriptor.payload_bytes,
        args.cold_bundles,
        "cold cache capacity",
    )?;
    let mut cache = DeepSeekV4ExpertBundleCache::new(hot_capacity_bytes, cold_capacity_bytes)?;

    // These calls run no payload read: they prove that a future router cannot
    // request a coordinate outside the sealed 43x256 source body.
    let invalid_layer_rejected =
        resolve_expert_bundle(&reader, ExpertBundleKey::new(DSV4F_LAYER_COUNT, 0)).is_err();
    let invalid_expert_rejected = resolve_expert_bundle(
        &reader,
        ExpertBundleKey::new(PROBE_LAYER, DSV4F_ROUTED_EXPERT_COUNT),
    )
    .is_err();
    if !invalid_layer_rejected || !invalid_expert_rejected {
        return Err(failure(
            "expert cache did not reject an out-of-body coordinate",
        ));
    }

    // This compact, non-router script makes both source fills and cache
    // transitions observable: cold prefetch, promotion, hot filling,
    // demotion, cold eviction, a hot prefetch hit, and a re-read after
    // eviction.  It never claims that these IDs came from an actual router.
    let mut accesses = Vec::new();
    run_prefetch(
        &mut cache,
        &reader,
        ExpertBundleKey::new(PROBE_LAYER, 205),
        &mut accesses,
    )?;
    run_acquire(
        &mut cache,
        &reader,
        ExpertBundleKey::new(PROBE_LAYER, 205),
        &mut accesses,
    )?;
    run_acquire(
        &mut cache,
        &reader,
        ExpertBundleKey::new(PROBE_LAYER, 168),
        &mut accesses,
    )?;
    run_prefetch(
        &mut cache,
        &reader,
        ExpertBundleKey::new(PROBE_LAYER, 53),
        &mut accesses,
    )?;
    run_acquire(
        &mut cache,
        &reader,
        ExpertBundleKey::new(PROBE_LAYER, 98),
        &mut accesses,
    )?;
    run_acquire(
        &mut cache,
        &reader,
        ExpertBundleKey::new(PROBE_LAYER, 205),
        &mut accesses,
    )?;
    run_prefetch(
        &mut cache,
        &reader,
        ExpertBundleKey::new(PROBE_LAYER, 205),
        &mut accesses,
    )?;
    run_prefetch(
        &mut cache,
        &reader,
        ExpertBundleKey::new(PROBE_LAYER, 12),
        &mut accesses,
    )?;
    run_acquire(
        &mut cache,
        &reader,
        ExpertBundleKey::new(PROBE_LAYER, 53),
        &mut accesses,
    )?;
    cache.assert_invariants()?;

    let final_state = cache.state();
    let stats = cache.counters();
    assert_probe_accounting(&accesses, &probe_descriptor, &final_state, stats)?;
    let manifest_after = sha256_file(&reader.artifact_root().join("manifest.json"))?;
    if manifest_before != manifest_after || manifest_before != reader.manifest_file_sha256() {
        return Err(failure(
            "artifact manifest content changed during a read-only cache probe",
        ));
    }

    let unsigned = json!({
        "schema": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "artifact": {
            "root": reader.artifact_root().display().to_string(),
            "manifest_schema": FULL_STREAM_SCHEMA,
            "manifest_status": FULL_STREAM_STATUS,
            "manifest_seal_sha256": reader.manifest_seal_sha256(),
            "manifest_file_sha256_before": manifest_before,
            "manifest_file_sha256_after": manifest_after,
            "restart_seal_sha256": reader.restart_seal_sha256(),
            "source": {
                "repository": reader.source_identity().repository,
                "revision": reader.source_identity().revision,
                "source_parent_retained": false,
            },
            "reader_admission": {
                "tensor_count": reader.tensor_count(),
                "source_native_tensor_bytes": reader.tensor_bytes(),
                "content_addressed_chunk_count": reader.chunk_count(),
                "native_fp4_pair_count": reader.native_scale_pair_count_for(NativeScalePairKind::Fp4E2M1fnX2),
                "all_named_tensor_and_chunk_paths_admitted_before_probe": true,
                "all_payload_reads_used_verified_full_tensor_reader": true,
            },
            "pinned_source_metadata_sha256": {
                "inference_model_py": reader.source_metadata_asset_sha256("inference/model.py")?,
                "inference_kernel_py": reader.source_metadata_asset_sha256("inference/kernel.py")?,
                "inference_convert_py": reader.source_metadata_asset_sha256("inference/convert.py")?,
            },
        },
        "source_hashes": source_hashes(&args.repo_root)?,
        "probe_configuration": {
            "layer": PROBE_LAYER,
            "hot_capacity_bundles": args.hot_bundles,
            "cold_capacity_bundles": args.cold_bundles,
            "exact_bundle_payload_bytes": probe_descriptor.payload_bytes,
            "hot_capacity_bytes": hot_capacity_bytes,
            "cold_capacity_bytes": cold_capacity_bytes,
            "resident_payload_byte_ceiling": checked_add(hot_capacity_bytes, cold_capacity_bytes, "combined resident capacity")?,
            "full_tensor_read_only": true,
            "source_parent_files_opened": false,
            "artifact_mutated": false,
        },
        "exact_native_fp4_bundle_contract": descriptor_json(&probe_descriptor),
        "rejection_checks": {
            "out_of_range_layer_rejected_before_any_payload_read": invalid_layer_rejected,
            "out_of_range_expert_rejected_before_any_payload_read": invalid_expert_rejected,
            "non_v4_source_identity_rejected_by_admitted_reader_and_cache_identity_guard": true,
            "non_fp4_or_wrong_w1_w2_w3_pair_geometry_rejected_by_cache_resolver": true,
        },
        "script": {
            "origin": "fixed storage-only cache transition probe; IDs are not route results",
            "actions": accesses,
            "final_state": state_json(&final_state),
            "counters": counters_json(stats),
        },
        "storage_accounting": {
            "source_bundle_loads": stats.source_bundle_loads,
            "source_tensor_reads": stats.source_tensor_reads,
            "source_chunk_reads": stats.source_chunk_reads,
            "actual_payload_bytes_returned": stats.source_payload_bytes_returned,
            "actual_verified_chunk_bytes": stats.source_verified_chunk_bytes,
            "resident_hot_bytes": final_state.hot_resident_bytes,
            "resident_cold_bytes": final_state.cold_resident_bytes,
            "resident_total_bytes": checked_add(final_state.hot_resident_bytes, final_state.cold_resident_bytes, "resident byte total")?,
            "cache_payload_persisted_after_process": false,
            "cache_payload_written_to_artifact": false,
        },
        "reproduction": {
            "actual_command": reproduction_command(&args),
            "checks": [
                "DeepSeekV4FullStreamReader::admit validates sealed manifest, pinned source identity, complete tensor mappings, content-addressed regular non-symlink paths, and source-native scale pairs.",
                "Every cache fill calls DeepSeekV4FullStreamReader::read_verified_full for w1, w1.scale, w2, w2.scale, w3, and w3.scale; every touched chunk is SHA-256 checked before its bytes are retained.",
                "Cache invariants check tier capacities, map/byte consistency, one-to-one LRU membership, and no duplicate hot/cold residency after every probe action.",
                "The artifact manifest SHA-256 is checked before and after the probe; no artifact file is opened for write.",
                "The resulting receipt must pass lab.receipts.verify using the campaign canonical seal family."
            ],
        },
        "claim_boundary": {
            "storage_cache_primitive_only": true,
            "source_chunk_backed": true,
            "full_43_layer_engine_created": false,
            "router_executed": false,
            "route_ids_are_model_router_outputs": false,
            "expert_matvec_executed": false,
            "model_forward_tokens": 0,
            "gpu_dispatches": 0,
            "command_buffers": 0,
            "hcli_endpoint_started": false,
            "base_true_tps_measured": false,
            "claim": "bounded source-chunk expert residency/prefetch storage probe only; not a DeepSeek-V4 runtime, route execution, forward, HCLI, GPU, parity, or TPS result",
        },
    });
    let receipt = seal(unsigned)?;
    write_new_receipt(&args.out, &receipt)?;
    let seal = receipt
        .get("seal_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| failure("sealed receipt has no seal_sha256"))?;
    println!(
        "status={RECEIPT_STATUS} receipt={} seal_sha256={seal}",
        args.out.display()
    );
    Ok(())
}

fn run_prefetch(
    cache: &mut DeepSeekV4ExpertBundleCache,
    reader: &DeepSeekV4FullStreamReader,
    key: ExpertBundleKey,
    accesses: &mut Vec<Value>,
) -> ExampleResult<()> {
    let access = cache.prefetch(reader, key)?;
    if cache.resident(key).is_none() {
        return Err(failure(
            "prefetch returned without resident source-native bytes",
        ));
    }
    accesses.push(access_json(reader, "prefetch", &access)?);
    Ok(())
}

fn run_acquire(
    cache: &mut DeepSeekV4ExpertBundleCache,
    reader: &DeepSeekV4FullStreamReader,
    key: ExpertBundleKey,
    accesses: &mut Vec<Value>,
) -> ExampleResult<()> {
    let access = cache.acquire(reader, key)?;
    if cache.resident(key).is_none() {
        return Err(failure(
            "demand acquire returned without resident source-native bytes",
        ));
    }
    accesses.push(access_json(reader, "acquire", &access)?);
    Ok(())
}

fn assert_probe_accounting(
    accesses: &[Value],
    descriptor: &ExpertBundleDescriptor,
    final_state: &ExpertCacheState,
    stats: ExpertCacheCounters,
) -> ExampleResult<()> {
    let source_reads = accesses
        .iter()
        .filter(|access| !access["source_read"].is_null())
        .count() as u64;
    if source_reads == 0 || stats.source_bundle_loads != source_reads {
        return Err(failure(
            "source load counter differs from recorded cache-fill accesses",
        ));
    }
    let expected_payload_bytes = checked_mul(
        descriptor.payload_bytes,
        source_reads,
        "expected probe source payload bytes",
    )?;
    let expected_chunk_bytes = checked_mul(
        descriptor.verified_chunk_bytes_per_fill,
        source_reads,
        "expected probe verified chunk bytes",
    )?;
    let expected_tensor_reads = checked_mul(source_reads, 6, "expected source tensor reads")?;
    let expected_chunk_reads = checked_mul(
        source_reads,
        u64::try_from(descriptor.source_chunk_read_count_per_fill)?,
        "expected source chunk reads",
    )?;
    if stats.source_payload_bytes_returned != expected_payload_bytes
        || stats.source_verified_chunk_bytes != expected_chunk_bytes
        || stats.source_tensor_reads != expected_tensor_reads
        || stats.source_chunk_reads != expected_chunk_reads
    {
        return Err(failure("cache source byte/read accounting does not close"));
    }
    if final_state.hot_resident_bytes > final_state.hot_capacity_bytes
        || final_state.cold_resident_bytes > final_state.cold_capacity_bytes
        || final_state.hot_keys_lru_to_mru.is_empty()
        || final_state.cold_keys_lru_to_mru.is_empty()
    {
        return Err(failure(
            "probe did not preserve the configured bounded hot/cold state",
        ));
    }
    Ok(())
}

fn descriptor_json(descriptor: &ExpertBundleDescriptor) -> Value {
    json!({
        "key": key_json(descriptor.key),
        "payload_bytes": descriptor.payload_bytes,
        "verified_chunk_bytes_per_full_fill": descriptor.verified_chunk_bytes_per_fill,
        "source_chunk_read_count_per_full_fill": descriptor.source_chunk_read_count_per_fill,
        "operators": descriptor.operators.iter().map(operator_json).collect::<Vec<_>>(),
    })
}

fn operator_json(operator: &ExpertOperatorDescriptor) -> Value {
    json!({
        "operator": operator.operator.as_str(),
        "weight": {
            "name": operator.weight_name,
            "dtype": "I8",
            "bytes": operator.weight_bytes,
        },
        "scale": {
            "name": operator.scale_name,
            "dtype": "F8_E8M0",
            "bytes": operator.scale_bytes,
        },
        "source_shard": operator.source_shard,
        "representation": operator.representation.as_str(),
        "geometry": {
            "out_rows": operator.out_rows,
            "packed_k": operator.packed_k,
            "logical_k": operator.logical_k,
            "scale_rows": operator.scale_rows,
            "scale_cols": operator.scale_cols,
        },
    })
}

fn access_json(
    reader: &DeepSeekV4FullStreamReader,
    action: &str,
    access: &ExpertCacheAccess,
) -> ExampleResult<Value> {
    Ok(json!({
        "action": action,
        "key": key_json(access.key),
        "result": access.result.as_str(),
        "source_read": match &access.source_read {
            Some(source_read) => source_read_json(reader, source_read)?,
            None => Value::Null,
        },
        "state_after": state_json(&access.state_after),
    }))
}

fn source_read_json(
    reader: &DeepSeekV4FullStreamReader,
    source_read: &ExpertBundleSourceRead,
) -> ExampleResult<Value> {
    let mut paths = Vec::with_capacity(source_read.chunk_paths.len());
    for path in &source_read.chunk_paths {
        let absolute = reader.artifact_root().join(&path.chunk_relpath);
        let metadata = fs::symlink_metadata(&absolute)?;
        if metadata.file_type().is_symlink()
            || !metadata.file_type().is_file()
            || metadata.len() != path.bytes
        {
            return Err(failure(
                "source read path changed after verified reader access",
            ));
        }
        paths.push(json!({
            "tensor": path.tensor_name,
            "tensor_role": path.tensor_role,
            "chunk_relative_path": path.chunk_relpath,
            "chunk_absolute_path": absolute.display().to_string(),
            "chunk_sha256": path.chunk_sha256,
            "bytes": path.bytes,
            "regular_non_symlink_after_reader_read": true,
        }));
    }
    Ok(json!({
        "key": key_json(source_read.key),
        "reader_operation": "DeepSeekV4FullStreamReader::read_verified_full",
        "payload_bytes_returned": source_read.payload_bytes_returned,
        "verified_chunk_bytes": source_read.verified_chunk_bytes,
        "source_chunk_read_count": source_read.source_chunk_read_count,
        "paths": paths,
        "all_touched_chunks_sha256_verified_before_bytes_retained": true,
    }))
}

fn state_json(state: &ExpertCacheState) -> Value {
    json!({
        "hot_capacity_bytes": state.hot_capacity_bytes,
        "cold_capacity_bytes": state.cold_capacity_bytes,
        "hot_resident_bytes": state.hot_resident_bytes,
        "cold_resident_bytes": state.cold_resident_bytes,
        "hot_keys_lru_to_mru": state.hot_keys_lru_to_mru.iter().copied().map(key_json).collect::<Vec<_>>(),
        "cold_keys_lru_to_mru": state.cold_keys_lru_to_mru.iter().copied().map(key_json).collect::<Vec<_>>(),
        "counters": counters_json(state.counters),
    })
}

fn counters_json(counters: ExpertCacheCounters) -> Value {
    json!({
        "demand_requests": counters.demand_requests,
        "prefetch_requests": counters.prefetch_requests,
        "demand_hot_hits": counters.demand_hot_hits,
        "demand_cold_hits": counters.demand_cold_hits,
        "demand_misses": counters.demand_misses,
        "prefetch_hot_hits": counters.prefetch_hot_hits,
        "prefetch_cold_hits": counters.prefetch_cold_hits,
        "prefetch_misses": counters.prefetch_misses,
        "promotions": counters.promotions,
        "hot_demotions": counters.hot_demotions,
        "hot_evictions": counters.hot_evictions,
        "cold_evictions": counters.cold_evictions,
        "demand_source_loads": counters.demand_source_loads,
        "prefetch_source_loads": counters.prefetch_source_loads,
        "source_bundle_loads": counters.source_bundle_loads,
        "source_tensor_reads": counters.source_tensor_reads,
        "source_chunk_reads": counters.source_chunk_reads,
        "source_payload_bytes_returned": counters.source_payload_bytes_returned,
        "source_verified_chunk_bytes": counters.source_verified_chunk_bytes,
    })
}

fn key_json(key: ExpertBundleKey) -> Value {
    json!({"layer": key.layer, "expert": key.expert})
}

fn source_hashes(repo_root: &Path) -> ExampleResult<BTreeMap<String, String>> {
    let mut hashes = BTreeMap::new();
    for relative in [
        "crates/hawking-core/src/gravity_deepseek_v4.rs",
        "crates/hawking-core/src/gravity_deepseek_v4_expert_cache.rs",
        "crates/hawking-core/src/lib.rs",
        "crates/hawking-core/examples/gravity_deepseek_v4_expert_cache_probe.rs",
        "Cargo.lock",
    ] {
        hashes.insert(relative.to_owned(), sha256_file(&repo_root.join(relative))?);
    }
    Ok(hashes)
}

fn reproduction_command(args: &Args) -> String {
    format!(
        "cargo run --release -p hawking-core --example gravity_deepseek_v4_expert_cache_probe -- --artifact {} --repo-root {} --out {} --hot-bundles {} --cold-bundles {}",
        args.artifact.display(),
        args.repo_root.display(),
        args.out.display(),
        args.hot_bundles,
        args.cold_bundles,
    )
}

fn parse_args() -> ExampleResult<Args> {
    let mut artifact = None;
    let mut repo_root = None;
    let mut out = None;
    let mut hot_bundles = DEFAULT_HOT_BUNDLES;
    let mut cold_bundles = DEFAULT_COLD_BUNDLES;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--artifact" => artifact = args.next().map(PathBuf::from),
            "--repo-root" => repo_root = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            "--hot-bundles" => hot_bundles = parse_positive_u64(args.next(), "--hot-bundles")?,
            "--cold-bundles" => cold_bundles = parse_positive_u64(args.next(), "--cold-bundles")?,
            "--help" | "-h" => {
                println!(
                    "usage: gravity_deepseek_v4_expert_cache_probe --artifact <absolute full Gravity dir> --repo-root <absolute hawking repo> --out <absolute receipt.json> [--hot-bundles N] [--cold-bundles N]"
                );
                std::process::exit(0);
            }
            other => return Err(failure(format!("unknown argument {other:?}"))),
        }
    }
    let artifact = artifact.ok_or_else(|| failure("--artifact is required"))?;
    let repo_root = repo_root.ok_or_else(|| failure("--repo-root is required"))?;
    let out = out.ok_or_else(|| failure("--out is required"))?;
    if !artifact.is_absolute() || !repo_root.is_absolute() || !out.is_absolute() {
        return Err(failure(
            "--artifact, --repo-root, and --out must be absolute paths",
        ));
    }
    if !repo_root.is_dir() {
        return Err(failure("--repo-root must be an existing directory"));
    }
    Ok(Args {
        artifact,
        repo_root,
        out,
        hot_bundles,
        cold_bundles,
    })
}

fn parse_positive_u64(value: Option<String>, label: &str) -> ExampleResult<u64> {
    let parsed = value
        .ok_or_else(|| failure(format!("{label} needs a value")))?
        .parse::<u64>()?;
    if parsed == 0 {
        return Err(failure(format!("{label} must be positive")));
    }
    Ok(parsed)
}

fn reject_output_inside_artifact(out: &Path, artifact_root: &Path) -> ExampleResult<()> {
    let parent = out
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or_else(|| failure("--out needs a parent directory"))?;
    let canonical_parent = fs::canonicalize(parent)?;
    let candidate = canonical_parent.join(
        out.file_name()
            .ok_or_else(|| failure("--out needs a filename"))?,
    );
    if candidate.starts_with(artifact_root) {
        return Err(failure(
            "--out must remain outside the immutable full Gravity artifact",
        ));
    }
    Ok(())
}

fn write_new_receipt(path: &Path, receipt: &Value) -> ExampleResult<()> {
    if path.exists() {
        return Err(failure(format!(
            "refusing to overwrite existing expert cache receipt {}",
            path.display()
        )));
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or_else(|| failure("--out needs a parent directory"))?;
    fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| failure("--out filename must be UTF-8"))?;
    let temporary = parent.join(format!(
        ".{name}.{}.expert-cache-probe.tmp",
        std::process::id()
    ));
    let encoded = serde_json::to_vec_pretty(receipt)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    if let Err(error) = file
        .write_all(&encoded)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all())
    {
        let _ = fs::remove_file(&temporary);
        return Err(Box::new(error));
    }
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(Box::new(error));
    }
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn seal(mut value: Value) -> ExampleResult<Value> {
    if value
        .as_object()
        .ok_or_else(|| failure("receipt root must be a JSON object"))?
        .contains_key("seal_sha256")
    {
        return Err(failure("receipt unexpectedly already contains a seal"));
    }
    // This receipt intentionally contains only strings, booleans, integers,
    // arrays, and sorted JSON maps; serde_json's compact output therefore
    // matches the campaign's Python canonical JSON form.  The emitted file is
    // independently checked with lab.receipts.verify by the caller.
    let digest = sha256(&serde_json::to_vec(&value)?);
    value
        .as_object_mut()
        .expect("object was checked above")
        .insert("seal_sha256".to_owned(), Value::String(digest));
    Ok(value)
}

fn sha256_file(path: &Path) -> ExampleResult<String> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(failure(format!(
            "expected a regular non-symlink file for hashing: {}",
            path.display()
        )));
    }
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn checked_mul(left: u64, right: u64, label: &str) -> ExampleResult<u64> {
    left.checked_mul(right)
        .ok_or_else(|| failure(format!("{label} overflow")))
}

fn checked_add(left: u64, right: u64, label: &str) -> ExampleResult<u64> {
    left.checked_add(right)
        .ok_or_else(|| failure(format!("{label} overflow")))
}

fn failure(message: impl Into<String>) -> Box<dyn Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}

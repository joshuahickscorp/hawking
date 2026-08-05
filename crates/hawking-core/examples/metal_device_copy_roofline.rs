//! Bounded Metal device-copy roofline probe for Apple M3 systems.
//!
//! This deliberately measures **only** `MTLBlitCommandEncoder` buffer copies.
//! It does not open a Gravity artifact, compile or run a model shader, execute
//! a token, invoke HCLI, or report a model/kernel/TPS result.  The source and
//! destination buffers are private GPU buffers; a tiny shared readback buffer
//! validates sampled output bytes after the measured copy sequence.
//!
//! The one DSV4F value admitted into the receipt is the exact static
//! source-layout body-weight contract from the sealed expert-residency receipt.
//! Arithmetic against the measured *payload copy* ceiling is explicitly an
//! ideal upper-bound comparator, never a physical DSV4F traffic or TPS claim.
//!
//! ```sh
//! cargo run --release -p hawking-core --example metal_device_copy_roofline -- \
//!   --static-expert-residency-receipt /absolute/path/to/static-expert-residency-receipt-v2.json \
//!   --out /absolute/path/to/DSV4F_METAL_DEVICE_COPY_ROOFLINE.json
//! ```

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("metal_device_copy_roofline requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
mod macos {
    use metal::objc::{msg_send, sel, sel_impl};
    use metal::{Device, MTLCommandBufferStatus, MTLResourceOptions, NSRange};
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::error::Error;
    use std::fs::{self, OpenOptions};
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    const RECEIPT_SCHEMA: &str = "hawking.gravity.deepseek_v4.metal_device_copy_roofline.v1";
    const RECEIPT_STATUS: &str = "PASS_REAL_M3_METAL_DEVICE_COPY_CEILING_NOT_MODEL_KERNEL_OR_TPS";
    const EXPECTED_STATIC_RESIDENCY_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.static_expert_residency.v1";
    const EXPECTED_STATIC_RESIDENCY_STATUS: &str =
        "SEALED_STATIC_FULL_STREAM_EXPERT_RESIDENCY_CONTRACT_RUNTIME_PENDING";
    const EXPECTED_STATIC_RESIDENCY_SEAL: &str =
        "bb1a77fea9b7de127f936f5b138559a52a9f96ec7b611e5ed8cd05d3ad4fe2da";
    const EXPECTED_FULL_MANIFEST_SEAL: &str =
        "da28df8300aed5d557b3e433acc496ee19055ce57f584c6ddf619982e7e15f62";
    const EXPECTED_REPOSITORY: &str = "deepseek-ai/DeepSeek-V4-Flash";
    const EXPECTED_REVISION: &str = "60d8d70770c6776ff598c94bb586a859a38244f1";
    const STATIC_BODY_LOGICAL_BYTES_PER_DECODE_TOKEN: u64 = 10_158_240_088;
    const MIB: u64 = 1024 * 1024;
    const DEFAULT_SIZES_MIB: &[u64] = &[128, 256, 512];
    const MAX_SIZES: usize = 3;
    const MIN_SIZE_MIB: u64 = 64;
    const MAX_SIZE_MIB: u64 = 512;
    const DEFAULT_WARMUPS: usize = 3;
    const DEFAULT_TRIALS: usize = 9;
    const MAX_WARMUPS: usize = 12;
    const MAX_TRIALS: usize = 31;
    const SAMPLE_BYTES: usize = 64 * 1024;
    const SAMPLE_COUNT: usize = 4;

    type ProbeResult<T> = Result<T, Box<dyn Error>>;

    fn failure(message: impl Into<String>) -> Box<dyn Error> {
        std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
    }

    struct Args {
        out: PathBuf,
        static_expert_residency_receipt: PathBuf,
        sizes_mib: Vec<u64>,
        warmups: usize,
        trials: usize,
    }

    struct StaticContract {
        path: PathBuf,
        file_sha256: String,
        seal_sha256: String,
        body_logical_bytes_per_decode_token: u64,
        full_manifest_seal_sha256: String,
    }

    #[derive(Clone)]
    struct CopySample {
        ordinal: usize,
        gpu_start_ns: u64,
        gpu_end_ns: u64,
        gpu_duration_ns: u64,
        payload_bytes: u64,
        physical_copy_traffic_bytes: u64,
        payload_gib_per_s: f64,
        physical_copy_traffic_gib_per_s: f64,
        host_encode_us: u64,
        host_submit_us: u64,
        host_wait_us: u64,
        host_wall_us: u64,
        command_buffers: u64,
        blit_encoders: u64,
        copy_operations: u64,
        cpu_visible_waits: u64,
    }

    struct SizeResult {
        size_mib: u64,
        size_bytes: u64,
        sample_offsets: Vec<u64>,
        warmups: usize,
        trials: Vec<CopySample>,
        initialization_command_buffers: u64,
        initialization_blit_encoders: u64,
        initialization_cpu_visible_waits: u64,
        reseed_command_buffers: u64,
        reseed_blit_encoders: u64,
        reseed_cpu_visible_waits: u64,
        verification_command_buffers: u64,
        verification_blit_encoders: u64,
        verification_cpu_visible_waits: u64,
        source_pattern_sha256: String,
        expected_readback_sha256: String,
        actual_readback_sha256: String,
    }

    pub fn main() -> ProbeResult<()> {
        let args = parse_args()?;
        let contract = read_static_contract(&args.static_expert_residency_receipt)?;
        let device = Device::system_default()
            .ok_or_else(|| failure("Metal system-default device is unavailable"))?;
        let device_name = device.name().to_string();
        if !device_name.contains("M3") {
            return Err(failure(format!(
                "this bounded roofline probe is restricted to Apple M3; found {device_name:?}"
            )));
        }
        if !device.has_unified_memory() {
            return Err(failure(
                "this bounded probe requires a unified-memory Apple Silicon Metal device",
            ));
        }
        let queue = device.new_command_queue();
        let mut results = Vec::with_capacity(args.sizes_mib.len());
        for size_mib in &args.sizes_mib {
            results.push(run_size(
                &device,
                &queue,
                *size_mib,
                args.warmups,
                args.trials,
            )?);
        }

        let mut per_size = Vec::with_capacity(results.len());
        let mut best_median_payload_gib_per_s = 0.0_f64;
        let mut best_median_traffic_gib_per_s = 0.0_f64;
        let mut best_size_mib = 0_u64;
        for result in &results {
            let mut payloads = result
                .trials
                .iter()
                .map(|sample| sample.payload_gib_per_s)
                .collect::<Vec<_>>();
            let mut traffics = result
                .trials
                .iter()
                .map(|sample| sample.physical_copy_traffic_gib_per_s)
                .collect::<Vec<_>>();
            let mut gpu_duration_ns = result
                .trials
                .iter()
                .map(|sample| sample.gpu_duration_ns as f64)
                .collect::<Vec<_>>();
            let mut host_wait_us = result
                .trials
                .iter()
                .map(|sample| sample.host_wait_us as f64)
                .collect::<Vec<_>>();
            payloads.sort_by(f64::total_cmp);
            traffics.sort_by(f64::total_cmp);
            gpu_duration_ns.sort_by(f64::total_cmp);
            host_wait_us.sort_by(f64::total_cmp);
            let median_payload = percentile(&payloads, 0.50)?;
            let median_traffic = percentile(&traffics, 0.50)?;
            if median_payload > best_median_payload_gib_per_s {
                best_median_payload_gib_per_s = median_payload;
                best_median_traffic_gib_per_s = median_traffic;
                best_size_mib = result.size_mib;
            }
            let totals = measured_totals(result)?;
            per_size.push(json!({
                "copy_size_mib": result.size_mib,
                "copy_size_bytes": result.size_bytes,
                "private_source_bytes": result.size_bytes,
                "private_destination_bytes": result.size_bytes,
                "measured_trials": result.trials.len(),
                "warmup_copies": result.warmups,
                "gpu_timestamp_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime only",
                "measured_topology": totals,
                "gpu_duration_ns": summary(&gpu_duration_ns)?,
                "payload_copy_gib_per_s": summary(&payloads)?,
                "read_plus_write_copy_traffic_gib_per_s": summary(&traffics)?,
                "host_wait_us": summary(&host_wait_us)?,
                "output_integrity": {
                    "sample_count": SAMPLE_COUNT,
                    "sample_bytes_each": SAMPLE_BYTES,
                    "sample_offsets": result.sample_offsets,
                    "source_pattern_sha256": result.source_pattern_sha256,
                    "expected_readback_sha256": result.expected_readback_sha256,
                    "actual_readback_sha256": result.actual_readback_sha256,
                    "exact_match": result.expected_readback_sha256 == result.actual_readback_sha256,
                    "meaning": "sampled private destination bytes were copied to shared readback after the measured sequence and exactly match the post-warmup source reseed pattern",
                },
                "setup_topology": {
                    "initialization": {
                        "command_buffers": result.initialization_command_buffers,
                        "blit_encoders": result.initialization_blit_encoders,
                        "cpu_visible_waits": result.initialization_cpu_visible_waits,
                    },
                    "post_warmup_source_reseed": {
                        "command_buffers": result.reseed_command_buffers,
                        "blit_encoders": result.reseed_blit_encoders,
                        "cpu_visible_waits": result.reseed_cpu_visible_waits,
                    },
                    "output_readback": {
                        "command_buffers": result.verification_command_buffers,
                        "blit_encoders": result.verification_blit_encoders,
                        "cpu_visible_waits": result.verification_cpu_visible_waits,
                    },
                },
                "measured_trials_detail": result.trials.iter().map(copy_sample_json).collect::<Vec<_>>(),
            }));
        }
        if best_median_payload_gib_per_s <= 0.0 {
            return Err(failure(
                "no positive median GPU timestamped copy bandwidth was measured",
            ));
        }
        let best_median_payload_bytes_per_second =
            best_median_payload_gib_per_s * (1024_f64 * 1024_f64 * 1024_f64);
        let ideal_body_only_seconds_per_token = contract.body_logical_bytes_per_decode_token as f64
            / best_median_payload_bytes_per_second;
        let static_body_payload_bytes_per_second_at_100_tps =
            contract.body_logical_bytes_per_decode_token as f64 * 100.0;
        let static_body_requirement_fraction_of_best_copy_ceiling =
            static_body_payload_bytes_per_second_at_100_tps / best_median_payload_bytes_per_second;

        let created_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| failure(format!("system clock predates Unix epoch: {error}")))?
            .as_millis() as u64;
        let unsigned = json!({
            "schema": RECEIPT_SCHEMA,
            "status": RECEIPT_STATUS,
            "created_unix_ms": created_unix_ms,
            "scope": {
                "device_copy_only": true,
                "operation": "one private MTLBuffer -> private MTLBuffer MTLBlitCommandEncoder copy per warmup/trial",
                "gpu_compute_dispatches": 0,
                "gpu_model_kernels": 0,
                "gravity_artifact_opened": false,
                "deepseek_v4_weights_opened": false,
                "deepseek_v4_forward_tokens": 0,
                "hcli_endpoint_started": false,
                "base_true_tps_measured": false,
                "claim": "This is a device-copy bandwidth ceiling only. It is not a model bandwidth measurement, kernel benchmark, runtime result, source/CPU parity result, HCLI result, or TPS result.",
            },
            "reproduction": {
                "example": "cargo run --release -p hawking-core --example metal_device_copy_roofline",
                "sizes_mib": args.sizes_mib,
                "warmups_per_size": args.warmups,
                "gpu_timestamped_clean_trials_per_size": args.trials,
                "buffer_storage": "MTLStorageModePrivate for source/destination; MTLStorageModeShared only for bounded pattern staging/readback",
                "timing_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime only; host encode/submit/wait/wall are reported separately and never substituted for GPU duration",
                "copy_bandwidth_numerator": {
                    "payload_bytes": "N bytes copied source->destination per measured trial",
                    "read_plus_write_physical_copy_traffic_bytes": "2N bytes per measured trial; reported separately so the convention is explicit",
                },
                "measurement_guardrails": {
                    "max_copy_size_mib": MAX_SIZE_MIB,
                    "max_sizes": MAX_SIZES,
                    "min_measured_trials": 5,
                    "source_reseed_after_warmup": true,
                    "no_empty_measured_command_buffers": true,
                    "receipt_written_only_after_all_gpu_timestamps_and_integrity_checks_pass": true,
                },
            },
            "metal": {
                "device": device_name,
                "registry_id": device.registry_id(),
                "unified_memory": device.has_unified_memory(),
                "recommended_max_working_set_size_bytes": device.recommended_max_working_set_size(),
                "max_transfer_rate_bytes_per_second_reported_by_device": device.max_transfer_rate(),
                "measured_gpu_compute_dispatches": 0,
                "fallback": false,
                "fallback_count": 0,
            },
            "static_dsv4f_source_layout_comparator": {
                "receipt_path": contract.path,
                "receipt_file_sha256": contract.file_sha256,
                "receipt_schema": EXPECTED_STATIC_RESIDENCY_SCHEMA,
                "receipt_status": EXPECTED_STATIC_RESIDENCY_STATUS,
                "receipt_seal_sha256": contract.seal_sha256,
                "full_manifest_seal_sha256": contract.full_manifest_seal_sha256,
                "body_selected_weight_logical_bytes_per_decode_token": contract.body_logical_bytes_per_decode_token,
                "physical_active_bytes_per_token": "NOT_MEASURED_NO_NATIVE_RUNTIME",
                "comparator": {
                    "best_median_payload_copy_gib_per_s": best_median_payload_gib_per_s,
                    "best_median_read_plus_write_copy_traffic_gib_per_s": best_median_traffic_gib_per_s,
                    "best_size_mib": best_size_mib,
                    "ideal_body_only_seconds_per_token_if_every_static_logical_byte_used_best_payload_copy_rate": ideal_body_only_seconds_per_token,
                    "ideal_body_only_ms_per_token_if_every_static_logical_byte_used_best_payload_copy_rate": ideal_body_only_seconds_per_token * 1000.0,
                    "static_body_logical_payload_bytes_per_second_at_100_tps": static_body_payload_bytes_per_second_at_100_tps,
                    "static_body_requirement_fraction_of_best_payload_copy_ceiling_at_100_tps": static_body_requirement_fraction_of_best_copy_ceiling,
                },
                "strict_interpretation": "The static value is source-layout logical selected body weight bytes, not measured runtime traffic. The arithmetic is an ideal upper-bound comparator only; it does not establish cache behavior, active physical bytes, operation cost, synchronization cost, real model bandwidth, latency, or any TPS result.",
            },
            "sizes": per_size,
            "next_boundary": "A registered 43-layer source-native DSV4F runtime with complete-token GPU counters is required before this copy ceiling can be connected to physical active bytes, runtime bandwidth, p99 latency, or BASE_TRUE_TPS.",
        });
        let receipt = seal(unsigned)?;
        write_new_receipt(&args.out, &receipt)?;
        let seal = receipt
            .get("seal_sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| failure("sealed receipt lacks seal_sha256"))?;
        println!(
            "status={RECEIPT_STATUS} receipt={} seal_sha256={seal}",
            args.out.display()
        );
        Ok(())
    }

    fn parse_args() -> ProbeResult<Args> {
        let mut out = None::<PathBuf>;
        let mut static_expert_residency_receipt = None::<PathBuf>;
        let mut sizes_mib = DEFAULT_SIZES_MIB.to_vec();
        let mut warmups = DEFAULT_WARMUPS;
        let mut trials = DEFAULT_TRIALS;
        let mut args = std::env::args().skip(1);
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--out" => out = args.next().map(PathBuf::from),
                "--static-expert-residency-receipt" => {
                    static_expert_residency_receipt = args.next().map(PathBuf::from)
                }
                "--sizes-mib" => {
                    let raw = args
                        .next()
                        .ok_or_else(|| failure("--sizes-mib needs a comma-separated value"))?;
                    sizes_mib = raw
                        .split(',')
                        .map(|value| {
                            value.parse::<u64>().map_err(|error| {
                                failure(format!("invalid --sizes-mib element {value:?}: {error}"))
                            })
                        })
                        .collect::<ProbeResult<Vec<_>>>()?;
                }
                "--warmups" => {
                    warmups = args
                        .next()
                        .ok_or_else(|| failure("--warmups needs a value"))?
                        .parse::<usize>()
                        .map_err(|error| failure(format!("invalid --warmups: {error}")))?;
                }
                "--trials" => {
                    trials = args
                        .next()
                        .ok_or_else(|| failure("--trials needs a value"))?
                        .parse::<usize>()
                        .map_err(|error| failure(format!("invalid --trials: {error}")))?;
                }
                "--help" | "-h" => {
                    println!(
                        "usage: metal_device_copy_roofline --static-expert-residency-receipt <absolute v2 receipt.json> --out <absolute receipt.json> [--sizes-mib 128,256,512] [--warmups 3] [--trials 9]"
                    );
                    std::process::exit(0);
                }
                other => return Err(failure(format!("unknown argument {other:?}"))),
            }
        }
        let out = out.ok_or_else(|| failure("--out is required"))?;
        let static_expert_residency_receipt = static_expert_residency_receipt
            .ok_or_else(|| failure("--static-expert-residency-receipt is required"))?;
        if !out.is_absolute() || !static_expert_residency_receipt.is_absolute() {
            return Err(failure(
                "--out and --static-expert-residency-receipt must be absolute paths",
            ));
        }
        if sizes_mib.is_empty() || sizes_mib.len() > MAX_SIZES {
            return Err(failure(format!(
                "--sizes-mib must contain 1..={MAX_SIZES} bounded sizes"
            )));
        }
        let mut sorted = sizes_mib.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != sizes_mib.len()
            || sizes_mib.iter().any(|size| {
                *size < MIN_SIZE_MIB || *size > MAX_SIZE_MIB || *size % MIN_SIZE_MIB != 0
            })
        {
            return Err(failure(format!(
                "each --sizes-mib value must be unique, a multiple of {MIN_SIZE_MIB}, and within {MIN_SIZE_MIB}..={MAX_SIZE_MIB}"
            )));
        }
        if warmups == 0 || warmups > MAX_WARMUPS {
            return Err(failure(format!(
                "--warmups must be within 1..={MAX_WARMUPS}"
            )));
        }
        if !(5..=MAX_TRIALS).contains(&trials) {
            return Err(failure(format!("--trials must be within 5..={MAX_TRIALS}")));
        }
        Ok(Args {
            out,
            static_expert_residency_receipt,
            sizes_mib,
            warmups,
            trials,
        })
    }

    fn read_static_contract(path: &Path) -> ProbeResult<StaticContract> {
        let raw = fs::read(path).map_err(|error| {
            failure(format!(
                "cannot read static expert-residency receipt {}: {error}",
                path.display()
            ))
        })?;
        let file_sha256 = sha256(&raw);
        let mut value: Value = serde_json::from_slice(&raw).map_err(|error| {
            failure(format!(
                "static expert-residency receipt is not JSON {}: {error}",
                path.display()
            ))
        })?;
        let recorded_seal = value
            .as_object_mut()
            .ok_or_else(|| failure("static expert-residency receipt root must be an object"))?
            .remove("seal_sha256")
            .and_then(|seal| seal.as_str().map(str::to_owned))
            .ok_or_else(|| failure("static expert-residency receipt lacks seal_sha256"))?;
        let computed_seal = sha256(&serde_json::to_vec(&value)?);
        if recorded_seal != computed_seal || recorded_seal != EXPECTED_STATIC_RESIDENCY_SEAL {
            return Err(failure(format!(
                "static expert-residency receipt seal mismatch: recorded={recorded_seal} computed={computed_seal} expected={EXPECTED_STATIC_RESIDENCY_SEAL}"
            )));
        }
        check_string(&value, "/schema", EXPECTED_STATIC_RESIDENCY_SCHEMA)?;
        check_string(&value, "/status", EXPECTED_STATIC_RESIDENCY_STATUS)?;
        check_string(
            &value,
            "/source_binding/full_manifest_seal_sha256",
            EXPECTED_FULL_MANIFEST_SEAL,
        )?;
        check_string(&value, "/source_binding/repository", EXPECTED_REPOSITORY)?;
        check_string(&value, "/source_binding/revision", EXPECTED_REVISION)?;
        let body_logical_bytes = pointer_u64(
            &value,
            "/static_active_byte_summary/body_selected_weight_logical_bytes_per_decode_token",
        )?;
        if body_logical_bytes != STATIC_BODY_LOGICAL_BYTES_PER_DECODE_TOKEN {
            return Err(failure(format!(
                "static body logical-byte contract mismatch: {body_logical_bytes} != {STATIC_BODY_LOGICAL_BYTES_PER_DECODE_TOKEN}"
            )));
        }
        Ok(StaticContract {
            path: path.to_path_buf(),
            file_sha256,
            seal_sha256: recorded_seal,
            body_logical_bytes_per_decode_token: body_logical_bytes,
            full_manifest_seal_sha256: EXPECTED_FULL_MANIFEST_SEAL.to_owned(),
        })
    }

    fn check_string(value: &Value, pointer: &str, expected: &str) -> ProbeResult<()> {
        let found = value
            .pointer(pointer)
            .and_then(Value::as_str)
            .ok_or_else(|| failure(format!("missing string at static receipt {pointer}")))?;
        if found != expected {
            return Err(failure(format!(
                "static receipt value mismatch at {pointer}: expected {expected:?}, found {found:?}"
            )));
        }
        Ok(())
    }

    fn pointer_u64(value: &Value, pointer: &str) -> ProbeResult<u64> {
        value
            .pointer(pointer)
            .and_then(Value::as_u64)
            .ok_or_else(|| failure(format!("missing u64 at static receipt {pointer}")))
    }

    fn run_size(
        device: &Device,
        queue: &metal::CommandQueue,
        size_mib: u64,
        warmups: usize,
        trials: usize,
    ) -> ProbeResult<SizeResult> {
        let size_bytes = size_mib
            .checked_mul(MIB)
            .ok_or_else(|| failure("copy size overflow"))?;
        let size_usize =
            usize::try_from(size_bytes).map_err(|_| failure("copy size does not fit usize"))?;
        if size_usize < SAMPLE_BYTES * SAMPLE_COUNT {
            return Err(failure(
                "copy size is too small for bounded integrity samples",
            ));
        }
        let private = MTLResourceOptions::StorageModePrivate;
        let shared = MTLResourceOptions::StorageModeShared;
        let source = device.new_buffer(size_bytes, private);
        let destination = device.new_buffer(size_bytes, private);
        let pattern_a = deterministic_pattern(0x73d0_01e4_9b3a_2f51);
        let pattern_b = deterministic_pattern(0x26e5_c487_7a91_0d3b);
        if pattern_a == pattern_b {
            return Err(failure(
                "deterministic integrity patterns unexpectedly collide",
            ));
        }
        let staging_a = device.new_buffer(SAMPLE_BYTES as u64, shared);
        let staging_b = device.new_buffer(SAMPLE_BYTES as u64, shared);
        write_shared_buffer(&staging_a, &pattern_a);
        write_shared_buffer(&staging_b, &pattern_b);
        let readback_bytes = SAMPLE_BYTES
            .checked_mul(SAMPLE_COUNT)
            .ok_or_else(|| failure("readback size overflow"))?;
        let readback = device.new_buffer(readback_bytes as u64, shared);
        let offsets = sample_offsets(size_bytes)?;

        initialize_buffers(
            queue,
            &source,
            &destination,
            &staging_a,
            &offsets,
            size_bytes,
            size_mib,
        )?;
        for ordinal in 0..warmups {
            let _ = timed_copy(
                queue,
                &source,
                &destination,
                size_bytes,
                size_mib,
                "warmup",
                ordinal,
            )?;
        }
        // This happens after warmup so the destination contains pattern A and
        // the source contains pattern B at the audited samples. The first
        // measured copy must therefore make a real source->destination change
        // for the final readback check to pass.
        reseed_source_after_warmup(queue, &source, &staging_b, &offsets, size_mib)?;
        let mut measured = Vec::with_capacity(trials);
        for ordinal in 0..trials {
            measured.push(timed_copy(
                queue,
                &source,
                &destination,
                size_bytes,
                size_mib,
                "measured",
                ordinal,
            )?);
        }
        let actual_readback =
            readback_integrity(queue, &destination, &readback, &offsets, size_mib)?;
        let mut expected = Vec::with_capacity(readback_bytes);
        for _ in 0..SAMPLE_COUNT {
            expected.extend_from_slice(&pattern_b);
        }
        let expected_readback_sha256 = sha256(&expected);
        let actual_readback_sha256 = sha256(&actual_readback);
        if actual_readback != expected {
            return Err(failure(format!(
                "private-buffer output integrity mismatch at {size_mib} MiB: expected={} actual={}",
                expected_readback_sha256, actual_readback_sha256
            )));
        }
        Ok(SizeResult {
            size_mib,
            size_bytes,
            sample_offsets: offsets,
            warmups,
            trials: measured,
            initialization_command_buffers: 1,
            initialization_blit_encoders: 1,
            initialization_cpu_visible_waits: 1,
            reseed_command_buffers: 1,
            reseed_blit_encoders: 1,
            reseed_cpu_visible_waits: 1,
            verification_command_buffers: 1,
            verification_blit_encoders: 1,
            verification_cpu_visible_waits: 1,
            source_pattern_sha256: sha256(&pattern_b),
            expected_readback_sha256,
            actual_readback_sha256,
        })
    }

    fn initialize_buffers(
        queue: &metal::CommandQueue,
        source: &metal::Buffer,
        destination: &metal::Buffer,
        staging: &metal::Buffer,
        offsets: &[u64],
        size_bytes: u64,
        size_mib: u64,
    ) -> ProbeResult<()> {
        let command = queue.new_command_buffer();
        command.set_label(&format!("hawking.device-copy-roofline.init.{size_mib}MiB"));
        let blit = command.new_blit_command_encoder();
        blit.set_label("hawking.device-copy-roofline.init.blit");
        let range = NSRange {
            location: 0,
            length: size_bytes,
        };
        blit.fill_buffer(source, range, 0);
        blit.fill_buffer(destination, range, 0);
        for offset in offsets {
            blit.copy_from_buffer(staging, 0, source, *offset, SAMPLE_BYTES as u64);
        }
        blit.end_encoding();
        commit_and_wait(command, "initialization")
    }

    fn reseed_source_after_warmup(
        queue: &metal::CommandQueue,
        source: &metal::Buffer,
        staging: &metal::Buffer,
        offsets: &[u64],
        size_mib: u64,
    ) -> ProbeResult<()> {
        let command = queue.new_command_buffer();
        command.set_label(&format!(
            "hawking.device-copy-roofline.reseed.{size_mib}MiB"
        ));
        let blit = command.new_blit_command_encoder();
        blit.set_label("hawking.device-copy-roofline.reseed.blit");
        for offset in offsets {
            blit.copy_from_buffer(staging, 0, source, *offset, SAMPLE_BYTES as u64);
        }
        blit.end_encoding();
        commit_and_wait(command, "post-warmup source reseed")
    }

    fn timed_copy(
        queue: &metal::CommandQueue,
        source: &metal::Buffer,
        destination: &metal::Buffer,
        size_bytes: u64,
        size_mib: u64,
        phase: &str,
        ordinal: usize,
    ) -> ProbeResult<CopySample> {
        let wall_started = Instant::now();
        let encode_started = Instant::now();
        let command = queue.new_command_buffer();
        command.set_label(&format!(
            "hawking.device-copy-roofline.{phase}.{size_mib}MiB.{ordinal}"
        ));
        let blit = command.new_blit_command_encoder();
        blit.set_label("hawking.device-copy-roofline.measured-blit");
        blit.copy_from_buffer(source, 0, destination, 0, size_bytes);
        blit.end_encoding();
        let host_encode_us = encode_started.elapsed().as_micros() as u64;
        let submit_started = Instant::now();
        command.commit();
        let host_submit_us = submit_started.elapsed().as_micros() as u64;
        let wait_started = Instant::now();
        command.wait_until_completed();
        let host_wait_us = wait_started.elapsed().as_micros() as u64;
        ensure_completed(command, phase)?;
        let (gpu_start_ns, gpu_end_ns, gpu_duration_ns) = gpu_timestamp_ns(command)?;
        let host_wall_us = wall_started.elapsed().as_micros() as u64;
        let seconds = gpu_duration_ns as f64 / 1_000_000_000.0;
        let payload_gib_per_s = size_bytes as f64 / seconds / (1024_f64 * 1024_f64 * 1024_f64);
        let physical_copy_traffic_bytes = size_bytes
            .checked_mul(2)
            .ok_or_else(|| failure("copy traffic byte count overflow"))?;
        let physical_copy_traffic_gib_per_s =
            physical_copy_traffic_bytes as f64 / seconds / (1024_f64 * 1024_f64 * 1024_f64);
        if !payload_gib_per_s.is_finite() || !physical_copy_traffic_gib_per_s.is_finite() {
            return Err(failure("non-finite GPU timestamped copy bandwidth"));
        }
        Ok(CopySample {
            ordinal,
            gpu_start_ns,
            gpu_end_ns,
            gpu_duration_ns,
            payload_bytes: size_bytes,
            physical_copy_traffic_bytes,
            payload_gib_per_s,
            physical_copy_traffic_gib_per_s,
            host_encode_us,
            host_submit_us,
            host_wait_us,
            host_wall_us,
            command_buffers: 1,
            blit_encoders: 1,
            copy_operations: 1,
            cpu_visible_waits: 1,
        })
    }

    fn readback_integrity(
        queue: &metal::CommandQueue,
        destination: &metal::Buffer,
        readback: &metal::Buffer,
        offsets: &[u64],
        size_mib: u64,
    ) -> ProbeResult<Vec<u8>> {
        let command = queue.new_command_buffer();
        command.set_label(&format!(
            "hawking.device-copy-roofline.readback.{size_mib}MiB"
        ));
        let blit = command.new_blit_command_encoder();
        blit.set_label("hawking.device-copy-roofline.readback.blit");
        for (index, source_offset) in offsets.iter().enumerate() {
            blit.copy_from_buffer(
                destination,
                *source_offset,
                readback,
                (index * SAMPLE_BYTES) as u64,
                SAMPLE_BYTES as u64,
            );
        }
        blit.end_encoding();
        commit_and_wait(command, "output readback")?;
        let mut bytes = vec![0_u8; SAMPLE_BYTES * SAMPLE_COUNT];
        let bytes_len = bytes.len();
        unsafe {
            bytes.copy_from_slice(std::slice::from_raw_parts(
                readback.contents() as *const u8,
                bytes_len,
            ));
        }
        Ok(bytes)
    }

    fn commit_and_wait(command: &metal::CommandBufferRef, label: &str) -> ProbeResult<()> {
        command.commit();
        command.wait_until_completed();
        ensure_completed(command, label)
    }

    fn ensure_completed(command: &metal::CommandBufferRef, label: &str) -> ProbeResult<()> {
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(failure(format!(
                "Metal command buffer did not complete successfully for {label}: {:?}",
                command.status()
            )));
        }
        Ok(())
    }

    /// GPUStartTime/GPUEndTime are not wrapped by metal 0.29. This direct
    /// selector is read only after `wait_until_completed`; a missing or invalid
    /// timestamp is a hard failure, never replaced by host wait time.
    fn gpu_timestamp_ns(command: &metal::CommandBufferRef) -> ProbeResult<(u64, u64, u64)> {
        let (start, end): (f64, f64) = unsafe {
            (
                msg_send![command, GPUStartTime],
                msg_send![command, GPUEndTime],
            )
        };
        if !(start.is_finite() && end.is_finite() && start > 0.0 && end > start) {
            return Err(failure(format!(
                "completed Metal command buffer lacks valid GPU timestamps: start={start:?} end={end:?}"
            )));
        }
        let duration_ns = ((end - start) * 1_000_000_000.0).round() as u64;
        if duration_ns == 0 {
            return Err(failure("completed Metal GPU duration rounded to zero"));
        }
        Ok((
            (start * 1_000_000_000.0).round() as u64,
            (end * 1_000_000_000.0).round() as u64,
            duration_ns,
        ))
    }

    fn deterministic_pattern(mut state: u64) -> Vec<u8> {
        let mut pattern = Vec::with_capacity(SAMPLE_BYTES);
        for _ in 0..SAMPLE_BYTES {
            state ^= state << 7;
            state ^= state >> 9;
            state ^= state << 8;
            pattern.push((state >> 17) as u8);
        }
        pattern
    }

    fn write_shared_buffer(buffer: &metal::Buffer, bytes: &[u8]) {
        unsafe {
            (buffer.contents() as *mut u8).copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
        }
    }

    fn sample_offsets(size_bytes: u64) -> ProbeResult<Vec<u64>> {
        let alignment = SAMPLE_BYTES as u64;
        let mut offsets = vec![
            0,
            (size_bytes / 3 / alignment) * alignment,
            ((size_bytes * 2 / 3) / alignment) * alignment,
            size_bytes
                .checked_sub(alignment)
                .ok_or_else(|| failure("copy size is smaller than an integrity sample"))?,
        ];
        offsets.sort_unstable();
        offsets.dedup();
        if offsets.len() != SAMPLE_COUNT
            || offsets
                .iter()
                .any(|offset| offset.saturating_add(alignment) > size_bytes)
        {
            return Err(failure(
                "could not derive four bounded non-overlapping integrity samples",
            ));
        }
        Ok(offsets)
    }

    fn measured_totals(result: &SizeResult) -> ProbeResult<Value> {
        let mut payload_bytes = 0_u64;
        let mut physical_traffic_bytes = 0_u64;
        let mut command_buffers = 0_u64;
        let mut blit_encoders = 0_u64;
        let mut copy_operations = 0_u64;
        let mut waits = 0_u64;
        for sample in &result.trials {
            payload_bytes = payload_bytes
                .checked_add(sample.payload_bytes)
                .ok_or_else(|| failure("payload total overflow"))?;
            physical_traffic_bytes = physical_traffic_bytes
                .checked_add(sample.physical_copy_traffic_bytes)
                .ok_or_else(|| failure("traffic total overflow"))?;
            command_buffers += sample.command_buffers;
            blit_encoders += sample.blit_encoders;
            copy_operations += sample.copy_operations;
            waits += sample.cpu_visible_waits;
        }
        if command_buffers != result.trials.len() as u64
            || blit_encoders != command_buffers
            || copy_operations != command_buffers
            || waits != command_buffers
        {
            return Err(failure(
                "measured command-buffer/blit/copy/wait accounting did not reconcile",
            ));
        }
        Ok(json!({
            "bytes_read_from_private_source": payload_bytes,
            "bytes_written_to_private_destination": payload_bytes,
            "read_plus_write_physical_copy_traffic_bytes": physical_traffic_bytes,
            "command_buffers": command_buffers,
            "blit_encoders": blit_encoders,
            "compute_encoders": 0,
            "gpu_compute_dispatches": 0,
            "blit_copy_operations": copy_operations,
            "cpu_visible_waits": waits,
            "empty_command_buffers": 0,
            "accounting_reconciled": true,
        }))
    }

    fn copy_sample_json(sample: &CopySample) -> Value {
        json!({
            "ordinal": sample.ordinal,
            "gpu_start_ns": sample.gpu_start_ns,
            "gpu_end_ns": sample.gpu_end_ns,
            "gpu_duration_ns": sample.gpu_duration_ns,
            "payload_bytes_read_from_private_source": sample.payload_bytes,
            "payload_bytes_written_to_private_destination": sample.payload_bytes,
            "read_plus_write_physical_copy_traffic_bytes": sample.physical_copy_traffic_bytes,
            "payload_copy_gib_per_s": sample.payload_gib_per_s,
            "read_plus_write_copy_traffic_gib_per_s": sample.physical_copy_traffic_gib_per_s,
            "host_encode_us": sample.host_encode_us,
            "host_submit_us": sample.host_submit_us,
            "host_wait_us": sample.host_wait_us,
            "host_wall_us": sample.host_wall_us,
            "command_buffers": sample.command_buffers,
            "blit_encoders": sample.blit_encoders,
            "compute_encoders": 0,
            "gpu_compute_dispatches": 0,
            "blit_copy_operations": sample.copy_operations,
            "cpu_visible_waits": sample.cpu_visible_waits,
        })
    }

    fn percentile(sorted: &[f64], quantile: f64) -> ProbeResult<f64> {
        if sorted.is_empty() || !(0.0..=1.0).contains(&quantile) {
            return Err(failure("invalid percentile input"));
        }
        let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
        sorted
            .get(index)
            .copied()
            .ok_or_else(|| failure("percentile index is out of bounds"))
    }

    fn summary(sorted: &[f64]) -> ProbeResult<Value> {
        if sorted.is_empty() {
            return Err(failure("cannot summarize empty sample set"));
        }
        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        Ok(json!({
            "count": sorted.len(),
            "min": sorted.first().copied().unwrap_or_default(),
            "p50": percentile(sorted, 0.50)?,
            "p95": percentile(sorted, 0.95)?,
            "p99": percentile(sorted, 0.99)?,
            "max": sorted.last().copied().unwrap_or_default(),
            "mean": mean,
            "percentile_method": "nearest index round((n-1)*q) over GPU-timestamped trials",
        }))
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn seal(mut value: Value) -> ProbeResult<Value> {
        let object = value
            .as_object_mut()
            .ok_or_else(|| failure("receipt root must be an object"))?;
        if object.contains_key("seal_sha256") {
            return Err(failure("receipt unexpectedly already has a seal"));
        }
        let seal = sha256(&serde_json::to_vec(&value)?);
        value
            .as_object_mut()
            .expect("receipt root was checked above")
            .insert("seal_sha256".to_owned(), Value::String(seal));
        Ok(value)
    }

    fn write_new_receipt(path: &Path, receipt: &Value) -> ProbeResult<()> {
        if path.exists() {
            return Err(failure(format!(
                "refusing to overwrite existing roofline receipt {}",
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
            ".{name}.{}.metal-device-copy-roofline.tmp",
            std::process::id()
        ));
        let bytes = serde_json::to_vec_pretty(receipt)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| failure(format!("cannot create receipt temporary file: {error}")))?;
        if let Err(error) = file
            .write_all(&bytes)
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
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::main()
}

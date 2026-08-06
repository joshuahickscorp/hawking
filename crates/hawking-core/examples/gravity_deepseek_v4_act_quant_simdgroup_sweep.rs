//! Byte-exact Metal sweep for the bounded DeepSeek-V4-Flash `act_quant` stage.
//!
//! This is a deliberately narrow source-linear component experiment.  It uses
//! the exact deterministic BF16 `[4096]` input bound to
//! `layers.0.attn.wq_a`, checks every output E4M3FN activation byte and every
//! UE8M0 scale byte against the canonical CPU oracle v2, and compares an
//! optional SIMDgroup/block-parallel candidate against the previously sealed
//! 5.967 ms serial authority stage.
//!
//! It is **not** a V4 runtime, a model forward, token generation, HCLI, or a
//! BASE_TRUE_TPS benchmark.  The candidate remains a model.linear component
//! QAT candidate until a future full-runtime parity integration explicitly
//! proves otherwise.
//!
//! ```sh
//! cargo run --release -p hawking-core --example gravity_deepseek_v4_act_quant_simdgroup_sweep -- \
//!   --artifact /absolute/path/to/full-43-layer-stream.gravity \
//!   --cpu-oracle /absolute/path/to/DSV4F_ACT_QUANT_WQ_A_CPU_ORACLE-v2.json \
//!   --authority-receipt /absolute/path/to/DSV4F_MODEL_LINEAR_METAL_COMPONENT_PARITY-v1.json \
//!   --out /absolute/path/to/DSV4F_ACT_QUANT_SIMDGROUP_SWEEP.json
//! ```

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(
        std::io::Error::other("gravity_deepseek_v4_act_quant_simdgroup_sweep requires macOS Metal")
            .into(),
    )
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::gravity_deepseek_v4::{
        DeepSeekV4FullStreamReader, NativeScalePairKind, FULL_STREAM_SCHEMA, FULL_STREAM_STATUS,
    };
    use hawking_core::gravity_deepseek_v4_act_quant::{
        act_quant_bf16_ue8m0, deterministic_wq_a_input_bf16, verify_source_algorithm_anchors,
        ACT_QUANT_BLOCK, LAYER0_WQ_A_COLS, LAYER0_WQ_A_ROWS, LAYER0_WQ_A_SCALE, LAYER0_WQ_A_WEIGHT,
    };
    use hawking_core::metal::{
        MetalContext, MetalDispatchTiming, PhysicalTraceGuard, PhysicalTraceIdentity,
    };
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::cmp::Ordering;
    use std::error::Error;
    use std::fs::{self, File, OpenOptions};
    use std::io::Write;
    use std::path::{Path, PathBuf};

    const RECEIPT_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.act_quant_simdgroup_component_sweep.v1";
    const RECEIPT_STATUS: &str = "PASS_REAL_METAL_ACT_QUANT_BYTE_EXACT_SWEEP_NOT_FULL_RUNTIME";
    const CPU_ORACLE_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.act_quant_fp8_wq_a_cpu_algorithm_oracle.v1";
    const CPU_ORACLE_STATUS: &str =
        "PASS_SOURCE_DERIVED_CPU_ALGORITHM_ORACLE_NOT_INDEPENDENT_SOURCE_RUNTIME_PARITY";
    const CPU_ORACLE_V2_BASENAME: &str = "DSV4F_ACT_QUANT_WQ_A_CPU_ORACLE-v2.json";
    const AUTHORITY_RECEIPT_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.model_linear_fp8_act_quant_metal_component_parity.v1";
    const AUTHORITY_RECEIPT_STATUS: &str =
        "PASS_REAL_METAL_MODEL_LINEAR_COMPONENT_PARITY_NOT_FULL_RUNTIME";
    const AUTHORITY_KERNEL: &str = "deepseek_v4_act_quant_bf16_ue8m0_authority";
    const CANDIDATE_KERNEL: &str = "deepseek_v4_act_quant_bf16_ue8m0_simdgroup_block_candidate";
    const FIXED_AUTHORITY_GPU_US: u64 = 5_967;
    const DEFAULT_WARMUPS: usize = 8;
    const DEFAULT_TRIALS: usize = 25;
    const SIMD_WIDTH: u32 = 32;
    const BLOCKS: u32 = (LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK) as u32;
    const THREAD_LADDER: &[u32] = &[32, 64, 128, 256, 512, 1024, 2048];
    const VECTOR_LADDER: &[u32] = &[1, 2, 4, 8];

    type SweepResult<T> = Result<T, Box<dyn Error>>;

    struct Args {
        artifact: PathBuf,
        cpu_oracle: PathBuf,
        authority_receipt: PathBuf,
        out: PathBuf,
        warmups: usize,
        trials: usize,
    }

    #[derive(Clone)]
    struct CpuOracleBinding {
        path: PathBuf,
        file_sha256: String,
        seal_sha256: String,
        input_sha256_bf16_le: String,
        activation_sha256: String,
        scale_sha256: String,
    }

    #[derive(Clone)]
    struct AuthorityReceiptBinding {
        path: PathBuf,
        file_sha256: String,
        seal_sha256: String,
        gpu_duration_us: u64,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Geometry {
        threads_x: u32,
        vector_width: u32,
    }

    #[derive(Default)]
    struct TopologyCounts {
        dispatches: u64,
        command_buffers: u64,
        compute_encoders: u64,
        waits: u64,
    }

    struct Timings {
        gpu_us: Vec<u64>,
        encode_us: Vec<u64>,
        submit_us: Vec<u64>,
        wait_us: Vec<u64>,
        host_wall_us: Vec<u64>,
        gpu_intervals_ns: Vec<[u64; 2]>,
    }

    impl Timings {
        fn new(trials: usize) -> Self {
            Self {
                gpu_us: Vec::with_capacity(trials),
                encode_us: Vec::with_capacity(trials),
                submit_us: Vec::with_capacity(trials),
                wait_us: Vec::with_capacity(trials),
                host_wall_us: Vec::with_capacity(trials),
                gpu_intervals_ns: Vec::with_capacity(trials),
            }
        }

        fn record(&mut self, timing: &MetalDispatchTiming) -> SweepResult<()> {
            let gpu_us = timing
                .gpu_duration_us
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    failure(
                        "measured dispatch has no positive completed-command-buffer GPU timestamp",
                    )
                })?;
            let start_ns = timing
                .gpu_start_ns
                .ok_or_else(|| failure("measured dispatch has no GPU start timestamp"))?;
            let end_ns = timing
                .gpu_end_ns
                .ok_or_else(|| failure("measured dispatch has no GPU end timestamp"))?;
            if end_ns <= start_ns {
                return Err(failure(
                    "measured dispatch GPU timestamp interval is not positive",
                ));
            }
            self.gpu_us.push(gpu_us);
            self.encode_us.push(timing.encode_us);
            self.submit_us.push(timing.submit_us);
            self.wait_us.push(timing.wait_us);
            self.host_wall_us.push(timing.host_wall_us);
            self.gpu_intervals_ns.push([start_ns, end_ns]);
            Ok(())
        }
    }

    fn failure(message: impl Into<String>) -> Box<dyn Error> {
        std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
    }

    fn parse_positive(value: Option<String>, flag: &str) -> SweepResult<usize> {
        let value = value.ok_or_else(|| failure(format!("{flag} needs a value")))?;
        let parsed = value
            .parse::<usize>()
            .map_err(|_| failure(format!("{flag} must be a positive integer")))?;
        if parsed == 0 {
            return Err(failure(format!("{flag} must be positive")));
        }
        Ok(parsed)
    }

    fn parse_args() -> SweepResult<Args> {
        let mut artifact = None::<PathBuf>;
        let mut cpu_oracle = None::<PathBuf>;
        let mut authority_receipt = None::<PathBuf>;
        let mut out = None::<PathBuf>;
        let mut warmups = DEFAULT_WARMUPS;
        let mut trials = DEFAULT_TRIALS;
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--artifact" => artifact = arguments.next().map(PathBuf::from),
                "--cpu-oracle" => cpu_oracle = arguments.next().map(PathBuf::from),
                "--authority-receipt" => authority_receipt = arguments.next().map(PathBuf::from),
                "--out" => out = arguments.next().map(PathBuf::from),
                "--warmups" => warmups = parse_positive(arguments.next(), "--warmups")?,
                "--trials" => trials = parse_positive(arguments.next(), "--trials")?,
                "--help" | "-h" => {
                    println!(
                        "usage: gravity_deepseek_v4_act_quant_simdgroup_sweep --artifact <absolute full Gravity dir> --cpu-oracle <absolute DSV4F_ACT_QUANT_WQ_A_CPU_ORACLE-v2.json> --authority-receipt <absolute DSV4F_MODEL_LINEAR_METAL_COMPONENT_PARITY-v1.json> --out <absolute receipt.json> [--warmups N] [--trials N]"
                    );
                    std::process::exit(0);
                }
                other => return Err(failure(format!("unknown argument {other:?}"))),
            }
        }
        let artifact = artifact.ok_or_else(|| failure("--artifact is required"))?;
        let cpu_oracle = cpu_oracle.ok_or_else(|| failure("--cpu-oracle is required"))?;
        let authority_receipt =
            authority_receipt.ok_or_else(|| failure("--authority-receipt is required"))?;
        let out = out.ok_or_else(|| failure("--out is required"))?;
        if !artifact.is_absolute()
            || !cpu_oracle.is_absolute()
            || !authority_receipt.is_absolute()
            || !out.is_absolute()
        {
            return Err(failure("all path arguments must be absolute"));
        }
        Ok(Args {
            artifact,
            cpu_oracle,
            authority_receipt,
            out,
            warmups,
            trials,
        })
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn sha256_join(parts: &[&str]) -> String {
        let mut digest = Sha256::new();
        for part in parts {
            digest.update(part.as_bytes());
            digest.update([0]);
        }
        format!("{:x}", digest.finalize())
    }

    fn u16_le_bytes(values: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<u16>());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn regular_file(path: &Path, label: &str) -> SweepResult<PathBuf> {
        let metadata = fs::symlink_metadata(path)
            .map_err(|error| failure(format!("cannot inspect {label}: {error}")))?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            return Err(failure(format!(
                "{label} must be a regular non-symlink file"
            )));
        }
        Ok(fs::canonicalize(path)?)
    }

    fn read_verified_sealed_json(
        path: &Path,
        label: &str,
    ) -> SweepResult<(PathBuf, String, String, Value)> {
        let canonical_path = regular_file(path, label)?;
        let raw = fs::read(&canonical_path)?;
        let file_sha256 = sha256(&raw);
        let mut value: Value = serde_json::from_slice(&raw)
            .map_err(|error| failure(format!("{label} is not JSON: {error}")))?;
        let seal = value
            .as_object_mut()
            .ok_or_else(|| failure(format!("{label} root must be an object")))?
            .remove("seal_sha256")
            .and_then(|value| value.as_str().map(str::to_owned))
            .ok_or_else(|| failure(format!("{label} has no string seal_sha256")))?;
        if seal.len() != 64
            || !seal
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || sha256(&canonical_json(&value)) != seal
        {
            return Err(failure(format!("{label} canonical seal does not verify")));
        }
        value
            .as_object_mut()
            .expect("JSON object was checked")
            .insert("seal_sha256".to_owned(), Value::String(seal.clone()));
        Ok((canonical_path, file_sha256, seal, value))
    }

    fn value_at<'a>(value: &'a Value, pointer: &str, label: &str) -> SweepResult<&'a Value> {
        value
            .pointer(pointer)
            .ok_or_else(|| failure(format!("{label} lacks {pointer}")))
    }

    fn string_at<'a>(value: &'a Value, pointer: &str, label: &str) -> SweepResult<&'a str> {
        value_at(value, pointer, label)?
            .as_str()
            .ok_or_else(|| failure(format!("{label} {pointer} is not a string")))
    }

    fn u64_at(value: &Value, pointer: &str, label: &str) -> SweepResult<u64> {
        value_at(value, pointer, label)?
            .as_u64()
            .ok_or_else(|| failure(format!("{label} {pointer} is not an unsigned integer")))
    }

    fn bool_at(value: &Value, pointer: &str, label: &str) -> SweepResult<bool> {
        value_at(value, pointer, label)?
            .as_bool()
            .ok_or_else(|| failure(format!("{label} {pointer} is not a boolean")))
    }

    fn validate_cpu_oracle(
        path: &Path,
        reader: &DeepSeekV4FullStreamReader,
        input_bf16: &[u16],
        activation: &[u8],
        scales: &[u8],
    ) -> SweepResult<CpuOracleBinding> {
        if path.file_name().and_then(|name| name.to_str()) != Some(CPU_ORACLE_V2_BASENAME) {
            return Err(failure(
                "only canonical DSV4F_ACT_QUANT_WQ_A_CPU_ORACLE-v2.json can bind this sweep",
            ));
        }
        let (path, file_sha256, seal_sha256, value) =
            read_verified_sealed_json(path, "canonical CPU oracle v2")?;
        let label = "canonical CPU oracle v2";
        if string_at(&value, "/schema", label)? != CPU_ORACLE_SCHEMA
            || string_at(&value, "/status", label)? != CPU_ORACLE_STATUS
            || string_at(&value, "/artifact/manifest_schema", label)? != FULL_STREAM_SCHEMA
            || string_at(&value, "/artifact/manifest_status", label)? != FULL_STREAM_STATUS
            || string_at(&value, "/artifact/manifest_seal_sha256", label)?
                != reader.manifest_seal_sha256()
            || string_at(&value, "/artifact/manifest_file_sha256", label)?
                != reader.manifest_file_sha256()
            || string_at(&value, "/artifact/restart_receipt_seal_sha256", label)?
                != reader.restart_seal_sha256()
            || string_at(&value, "/artifact/source/repository", label)?
                != reader.source_identity().repository
            || string_at(&value, "/artifact/source/revision", label)?
                != reader.source_identity().revision
            || bool_at(&value, "/artifact/source/source_parent_retained", label)?
        {
            return Err(failure(
                "canonical CPU oracle v2 does not bind the admitted source-evicted full Gravity artifact",
            ));
        }
        let input_sha256_bf16_le = sha256(&u16_le_bytes(input_bf16));
        let activation_sha256 = sha256(activation);
        let scale_sha256 = sha256(scales);
        if string_at(&value, "/input/sha256_bf16_le", label)? != input_sha256_bf16_le
            || u64_at(&value, "/input/length", label)? != input_bf16.len() as u64
            || string_at(&value, "/act_quant/activation_sha256", label)? != activation_sha256
            || string_at(&value, "/act_quant/scale_sha256", label)? != scale_sha256
            || u64_at(&value, "/act_quant/activation_bytes", label)? != activation.len() as u64
            || u64_at(&value, "/act_quant/scale_bytes", label)? != scales.len() as u64
            || u64_at(&value, "/act_quant/block_size", label)? != ACT_QUANT_BLOCK as u64
        {
            return Err(failure(
                "canonical CPU oracle v2 differs from the direct source-derived act_quant recomputation",
            ));
        }
        Ok(CpuOracleBinding {
            path,
            file_sha256,
            seal_sha256,
            input_sha256_bf16_le,
            activation_sha256,
            scale_sha256,
        })
    }

    fn validate_authority_receipt(
        path: &Path,
        reader: &DeepSeekV4FullStreamReader,
        cpu_oracle: &CpuOracleBinding,
    ) -> SweepResult<AuthorityReceiptBinding> {
        let (path, file_sha256, seal_sha256, value) =
            read_verified_sealed_json(path, "fixed 5.967 ms authority receipt")?;
        let label = "fixed 5.967 ms authority receipt";
        if string_at(&value, "/schema", label)? != AUTHORITY_RECEIPT_SCHEMA
            || string_at(&value, "/status", label)? != AUTHORITY_RECEIPT_STATUS
            || string_at(&value, "/artifact/manifest_seal_sha256", label)?
                != reader.manifest_seal_sha256()
            || string_at(&value, "/artifact/manifest_file_sha256", label)?
                != reader.manifest_file_sha256()
            || string_at(
                &value,
                "/canonical_cpu_oracle_v2/receipt_seal_sha256",
                label,
            )? != cpu_oracle.seal_sha256
            || string_at(
                &value,
                "/canonical_cpu_oracle_v2/input_sha256_bf16_le",
                label,
            )? != cpu_oracle.input_sha256_bf16_le
            || string_at(&value, "/canonical_cpu_oracle_v2/activation_sha256", label)?
                != cpu_oracle.activation_sha256
            || string_at(
                &value,
                "/canonical_cpu_oracle_v2/activation_scale_sha256",
                label,
            )? != cpu_oracle.scale_sha256
            || string_at(&value, "/gpu_act_quant/kernel", label)? != AUTHORITY_KERNEL
            || !bool_at(
                &value,
                "/gpu_act_quant/activation_bytewise_cpu_oracle_match",
                label,
            )?
            || !bool_at(
                &value,
                "/gpu_act_quant/scale_bytewise_cpu_oracle_match",
                label,
            )?
            || bool_at(&value, "/gpu_act_quant/fallback", label)?
        {
            return Err(failure(
                "authority receipt is not the canonical byte-exact source-linear act_quant stage",
            ));
        }
        let gpu_duration_us = u64_at(&value, "/gpu_act_quant/dispatch/gpu_duration_us", label)?;
        if gpu_duration_us != FIXED_AUTHORITY_GPU_US {
            return Err(failure(format!(
                "authority receipt GPU duration must be the fixed {FIXED_AUTHORITY_GPU_US} us stage, found {gpu_duration_us} us"
            )));
        }
        if u64_at(&value, "/gpu_act_quant/dispatch/compute_dispatches", label)? != 1
            || u64_at(&value, "/gpu_act_quant/dispatch/command_buffers", label)? != 1
            || u64_at(&value, "/gpu_act_quant/dispatch/compute_encoders", label)? != 1
        {
            return Err(failure(
                "authority receipt did not record exactly one real GPU act_quant dispatch",
            ));
        }
        Ok(AuthorityReceiptBinding {
            path,
            file_sha256,
            seal_sha256,
            gpu_duration_us,
        })
    }

    fn check_timing(timing: &MetalDispatchTiming, counts: &mut TopologyCounts) -> SweepResult<()> {
        if timing.compute_dispatches != 1
            || timing.command_buffers != 1
            || timing.compute_encoders != 1
        {
            return Err(failure(
                "sweep dispatch did not use exactly one command buffer, encoder, and GPU dispatch",
            ));
        }
        counts.dispatches += 1;
        counts.command_buffers += 1;
        counts.compute_encoders += 1;
        counts.waits += 1; // dispatch_threads_timed always completed the command buffer.
        Ok(())
    }

    fn timing_summary(values: &[u64]) -> SweepResult<Value> {
        if values.is_empty() {
            return Err(failure(
                "timing summary requires at least one measured sample",
            ));
        }
        let mut ordered = values.to_vec();
        ordered.sort_unstable();
        let percentile = |numerator: usize| -> u64 {
            let index = (ordered.len() * numerator).div_ceil(100).saturating_sub(1);
            ordered[index]
        };
        let sum: u128 = ordered.iter().map(|value| u128::from(*value)).sum();
        Ok(json!({
            "count": ordered.len(),
            "min_us": ordered[0],
            "p50_us": percentile(50),
            "p95_us": percentile(95),
            "p99_us": percentile(99),
            "max_us": ordered[ordered.len() - 1],
            "mean_us": format!("{:.6}", sum as f64 / ordered.len() as f64),
            "samples_us": ordered,
        }))
    }

    fn timing_json(timings: &Timings) -> SweepResult<Value> {
        Ok(json!({
            "gpu_timestamp_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime",
            "gpu_duration": timing_summary(&timings.gpu_us)?,
            "host_encode_duration": timing_summary(&timings.encode_us)?,
            "host_submit_duration": timing_summary(&timings.submit_us)?,
            "host_wait_duration": timing_summary(&timings.wait_us)?,
            "host_wall_duration": timing_summary(&timings.host_wall_us)?,
            "measured_gpu_timestamp_intervals_ns": timings.gpu_intervals_ns,
        }))
    }

    fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &u32) {
        encoder.set_bytes(
            index,
            std::mem::size_of::<u32>() as u64,
            value as *const u32 as *const _,
        );
    }

    fn read_gpu_bytes(buffer: &metal::Buffer, length: usize) -> SweepResult<Vec<u8>> {
        if buffer.length() < length as u64 {
            return Err(failure(
                "Metal output buffer is smaller than requested byte readback",
            ));
        }
        Ok(unsafe { std::slice::from_raw_parts(buffer.contents() as *const u8, length).to_vec() })
    }

    fn run_authority(
        context: &MetalContext,
        input: &metal::Buffer,
        activation: &metal::Buffer,
        scales: &metal::Buffer,
        cols: u32,
        warmups: usize,
        trials: usize,
        counts: &mut TopologyCounts,
    ) -> SweepResult<Timings> {
        let mut timings = Timings::new(trials);
        for dispatch_index in 0..(warmups + trials) {
            let timing = context.dispatch_threads_timed(
                AUTHORITY_KERNEL,
                (BLOCKS, 1, 1),
                (SIMD_WIDTH, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(activation), 0);
                    encoder.set_buffer(2, Some(scales), 0);
                    set_u32(encoder, 3, &cols);
                },
            )?;
            check_timing(&timing, counts)?;
            if dispatch_index >= warmups {
                timings.record(&timing)?;
            }
        }
        Ok(timings)
    }

    fn run_candidate(
        context: &MetalContext,
        input: &metal::Buffer,
        activation: &metal::Buffer,
        scales: &metal::Buffer,
        cols: u32,
        geometry: Geometry,
        warmups: usize,
        trials: usize,
        counts: &mut TopologyCounts,
    ) -> SweepResult<Timings> {
        if geometry.threads_x < SIMD_WIDTH || geometry.threads_x % SIMD_WIDTH != 0 {
            return Err(failure(
                "candidate threads_x is not an integral SIMDgroup count",
            ));
        }
        let blocks_per_threadgroup = geometry.threads_x / SIMD_WIDTH;
        let threadgroups = BLOCKS.div_ceil(blocks_per_threadgroup);
        let grid_threads = threadgroups * geometry.threads_x;
        let mut timings = Timings::new(trials);
        for dispatch_index in 0..(warmups + trials) {
            let timing = context.dispatch_threads_timed(
                CANDIDATE_KERNEL,
                (grid_threads, 1, 1),
                (geometry.threads_x, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(activation), 0);
                    encoder.set_buffer(2, Some(scales), 0);
                    set_u32(encoder, 3, &cols);
                    set_u32(encoder, 4, &geometry.threads_x);
                    set_u32(encoder, 5, &geometry.vector_width);
                },
            )?;
            check_timing(&timing, counts)?;
            if dispatch_index >= warmups {
                timings.record(&timing)?;
            }
        }
        Ok(timings)
    }

    fn bool_match(expected: &[u8], observed: &[u8]) -> SweepResult<()> {
        if expected.len() != observed.len() {
            return Err(failure("byte parity operands have different lengths"));
        }
        if expected != observed {
            let first = expected
                .iter()
                .zip(observed)
                .position(|(left, right)| left != right)
                .unwrap_or(0);
            return Err(failure(format!(
                "GPU output differs from canonical CPU oracle byte at index {first}: expected={:#04x} observed={:#04x}",
                expected[first], observed[first]
            )));
        }
        Ok(())
    }

    fn geometry_json(geometry: Geometry) -> Value {
        let blocks_per_threadgroup = geometry.threads_x / SIMD_WIDTH;
        let threadgroups = BLOCKS.div_ceil(blocks_per_threadgroup);
        let active_lanes_per_block = match geometry.vector_width {
            1 => 32,
            2 => 32,
            4 => 32,
            8 => 16,
            _ => 0,
        };
        json!({
            "threads": [geometry.threads_x, 1, 1],
            "simdgroup_width": SIMD_WIDTH,
            "simdgroups_per_threadgroup": blocks_per_threadgroup,
            "source_blocks_per_threadgroup": blocks_per_threadgroup,
            "source_blocks_total": BLOCKS,
            "threadgroups": [threadgroups, 1, 1],
            "grid_threads": [threadgroups * geometry.threads_x, 1, 1],
            "block_width_elements": ACT_QUANT_BLOCK,
            "packed_vector_width_bf16_elements": geometry.vector_width,
            "active_lanes_per_source_block": active_lanes_per_block,
            "threadgroup_memory_bytes": 32,
            "register_pressure": "not exposed by Metal API; vector width is recorded, not inferred as occupancy",
            "occupancy": "not exposed by Metal API; not inferred",
            "pipeline_overlap": "none; each component dispatch uses an explicit completed-command-buffer wait",
        })
    }

    fn candidate_order(
        left: &(Geometry, u64, u64, u64),
        right: &(Geometry, u64, u64, u64),
    ) -> Ordering {
        (
            left.1,
            left.2,
            left.3,
            left.0.threads_x,
            left.0.vector_width,
        )
            .cmp(&(
                right.1,
                right.2,
                right.3,
                right.0.threads_x,
                right.0.vector_width,
            ))
    }

    fn seal(mut value: Value) -> SweepResult<(Value, String)> {
        if !value.is_object() || value.get("seal_sha256").is_some() {
            return Err(failure("sweep receipt must be an unsealed JSON object"));
        }
        let seal = sha256(&canonical_json(&value));
        value
            .as_object_mut()
            .expect("receipt object was checked")
            .insert("seal_sha256".to_owned(), Value::String(seal.clone()));
        Ok((value, seal))
    }

    fn write_new_receipt(path: &Path, receipt: &Value) -> SweepResult<()> {
        if path.exists() {
            return Err(failure(format!(
                "refusing to overwrite existing act_quant sweep receipt {}",
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
            ".{name}.{}.act-quant-simdgroup-sweep.tmp",
            std::process::id()
        ));
        let bytes = serde_json::to_vec_pretty(receipt)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| failure(format!("cannot create temporary sweep receipt: {error}")))?;
        if let Err(error) = file
            .write_all(&bytes)
            .and_then(|_| file.write_all(b"\n"))
            .and_then(|_| file.sync_all())
        {
            let _ = fs::remove_file(&temporary);
            return Err(Box::new(error));
        }
        if let Err(error) = fs::hard_link(&temporary, path) {
            let _ = fs::remove_file(&temporary);
            return Err(failure(format!(
                "refusing to overwrite or link sweep receipt {}: {error}",
                path.display()
            )));
        }
        fs::remove_file(&temporary)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    }

    pub fn run() -> SweepResult<()> {
        let args = parse_args()?;
        let reader = DeepSeekV4FullStreamReader::admit(&args.artifact)?;
        let anchors = verify_source_algorithm_anchors(&reader)?;
        let pair = reader.native_scale_pair(LAYER0_WQ_A_WEIGHT)?;
        if pair.kind != NativeScalePairKind::Fp8E4M3fn
            || pair.weight.name != LAYER0_WQ_A_WEIGHT
            || pair.scale.name != LAYER0_WQ_A_SCALE
            || pair.weight.shape.as_slice() != [LAYER0_WQ_A_ROWS as u64, LAYER0_WQ_A_COLS as u64]
            || pair.scale.shape.as_slice()
                != [
                    (LAYER0_WQ_A_ROWS / ACT_QUANT_BLOCK) as u64,
                    (LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK) as u64,
                ]
            || pair.logical_k != LAYER0_WQ_A_COLS as u64
            || pair.out_rows != LAYER0_WQ_A_ROWS as u64
        {
            return Err(failure(
                "admitted layer-0 WQ-A pair does not preserve the source-native FP8 geometry",
            ));
        }
        let input_bf16 = deterministic_wq_a_input_bf16();
        if input_bf16.len() != LAYER0_WQ_A_COLS || BLOCKS != 32 {
            return Err(failure("fixed layer-0 act_quant input geometry changed"));
        }
        let cpu = act_quant_bf16_ue8m0(&input_bf16)?;
        if cpu.activation_e4m3fn.len() != LAYER0_WQ_A_COLS
            || cpu.scales_e8m0fnu.len() != BLOCKS as usize
        {
            return Err(failure("direct CPU act_quant output geometry changed"));
        }
        let cpu_oracle = validate_cpu_oracle(
            &args.cpu_oracle,
            &reader,
            &input_bf16,
            &cpu.activation_e4m3fn,
            &cpu.scales_e8m0fnu,
        )?;
        let authority_receipt =
            validate_authority_receipt(&args.authority_receipt, &reader, &cpu_oracle)?;

        let input_bytes = u16_le_bytes(&input_bf16);
        let context = MetalContext::new_with_trace(true)?;
        let device = context.device_name();
        let authority_pipeline = context.pipeline(AUTHORITY_KERNEL)?;
        let candidate_pipeline = context.pipeline(CANDIDATE_KERNEL)?;
        let authority_thread_execution_width = authority_pipeline.thread_execution_width() as u64;
        let candidate_thread_execution_width = candidate_pipeline.thread_execution_width() as u64;
        let candidate_max_total_threads =
            candidate_pipeline.max_total_threads_per_threadgroup() as u64;
        drop(authority_pipeline);
        drop(candidate_pipeline);

        let input_buffer = context.new_buffer_with_bytes_checked(&input_bytes)?;
        let activation_buffer = context.new_buffer_checked(LAYER0_WQ_A_COLS)?;
        let scale_buffer = context.new_buffer_checked(BLOCKS as usize)?;
        let cols = LAYER0_WQ_A_COLS as u32;
        let run_nonce = sha256_join(&[
            reader.manifest_seal_sha256(),
            &cpu_oracle.seal_sha256,
            &authority_receipt.seal_sha256,
            &cpu_oracle.input_sha256_bf16_le,
            "act_quant_simdgroup_component_sweep_v1",
        ]);
        let interval_id = sha256_join(&[&run_nonce, AUTHORITY_KERNEL, CANDIDATE_KERNEL]);
        let trace_identity = PhysicalTraceIdentity::new(
            interval_id.clone(),
            run_nonce.clone(),
            "act_quant_simdgroup_component_sweep".to_owned(),
            "model_linear_qat_candidate_only".to_owned(),
            Some(1),
            0,
        )?;
        let physical_trace = PhysicalTraceGuard::begin(trace_identity)?;
        let mut counts = TopologyCounts::default();

        let authority_timings = run_authority(
            &context,
            &input_buffer,
            &activation_buffer,
            &scale_buffer,
            cols,
            args.warmups,
            args.trials,
            &mut counts,
        )?;
        let authority_activation = read_gpu_bytes(&activation_buffer, LAYER0_WQ_A_COLS)?;
        let authority_scales = read_gpu_bytes(&scale_buffer, BLOCKS as usize)?;
        bool_match(&cpu.activation_e4m3fn, &authority_activation)?;
        bool_match(&cpu.scales_e8m0fnu, &authority_scales)?;
        let authority_timing = timing_json(&authority_timings)?;
        let authority_p50 = u64_at(
            &authority_timing,
            "/gpu_duration/p50_us",
            "fresh authority sweep",
        )?;

        let mut candidates = Vec::with_capacity(THREAD_LADDER.len() * VECTOR_LADDER.len());
        let mut passing = Vec::<(Geometry, u64, u64, u64)>::new();
        for &threads_x in THREAD_LADDER {
            for &vector_width in VECTOR_LADDER {
                let geometry = Geometry {
                    threads_x,
                    vector_width,
                };
                let geometry_value = geometry_json(geometry);
                if u64::from(threads_x) > candidate_max_total_threads {
                    candidates.push(json!({
                        "geometry": geometry_value,
                        "status": "UNSUPPORTED_MAX_TOTAL_THREADS",
                        "not_dispatched_reason": "requested threadgroup exceeds the compiled candidate pipeline maximum",
                        "pipeline_max_total_threads_per_threadgroup": candidate_max_total_threads,
                        "gpu_dispatches": 0,
                        "command_buffers": 0,
                        "compute_encoders": 0,
                        "cpu_visible_waits": 0,
                        "fallback": false,
                        "fallback_reason": null,
                    }));
                    continue;
                }
                let timings = run_candidate(
                    &context,
                    &input_buffer,
                    &activation_buffer,
                    &scale_buffer,
                    cols,
                    geometry,
                    args.warmups,
                    args.trials,
                    &mut counts,
                )?;
                let activation = read_gpu_bytes(&activation_buffer, LAYER0_WQ_A_COLS)?;
                let scales = read_gpu_bytes(&scale_buffer, BLOCKS as usize)?;
                bool_match(&cpu.activation_e4m3fn, &activation)?;
                bool_match(&cpu.scales_e8m0fnu, &scales)?;
                let timing = timing_json(&timings)?;
                let p50 = u64_at(&timing, "/gpu_duration/p50_us", "candidate timings")?;
                let p95 = u64_at(&timing, "/gpu_duration/p95_us", "candidate timings")?;
                let p99 = u64_at(&timing, "/gpu_duration/p99_us", "candidate timings")?;
                passing.push((geometry, p50, p95, p99));
                let iterations = args.warmups + args.trials;
                candidates.push(json!({
                    "geometry": geometry_value,
                    "status": "PASS_GPU_TIMESTAMPED_BYTE_EXACT_CPU_ORACLE_V2",
                    "candidate_scope": "model.linear act_quant component QAT candidate only",
                    "warmup_dispatches": args.warmups,
                    "measured_gpu_timestamped_dispatches": args.trials,
                    "timing": timing,
                    "activation": {
                        "dtype": "F8_E4M3FN",
                        "bytes": activation.len(),
                        "sha256": sha256(&activation),
                        "bytewise_canonical_cpu_oracle_v2_match": true,
                    },
                    "scale": {
                        "dtype": "F8_E8M0FNU",
                        "bytes": scales.len(),
                        "sha256": sha256(&scales),
                        "bytewise_canonical_cpu_oracle_v2_match": true,
                    },
                    "logical_bytes_read_per_dispatch": input_bytes.len(),
                    "logical_bytes_written_per_dispatch": activation.len() + scales.len(),
                    "fp_operations": "exact source E4M3FN finite-table encoder; operation count is data-dependent implementation work and not inferred as FLOPs",
                    "integer_or_bit_operations": "UE8M0 exponent extraction plus E4M3FN finite-table search; not independently hardware-counter exposed",
                    "speedup_vs_fixed_5967us_authority_gpu_p50": format!("{:.6}", FIXED_AUTHORITY_GPU_US as f64 / p50 as f64),
                    "speedup_vs_same_run_authority_gpu_p50": format!("{:.6}", authority_p50 as f64 / p50 as f64),
                    "gpu_dispatches": iterations,
                    "command_buffers": iterations,
                    "compute_encoders": iterations,
                    "cpu_visible_waits": iterations,
                    "empty_command_buffers": 0,
                    "fallback": false,
                    "fallback_reason": null,
                }));
            }
        }

        let physical_counts = physical_trace.counts();
        drop(physical_trace);
        let (buffers_created, bytes_allocated, commits) = context.drain_stats();
        let trace_samples = context.drain_trace();
        if passing.is_empty() {
            return Err(failure(
                "no optional act_quant candidate passed byte-exact CPU oracle v2",
            ));
        }
        if physical_counts.command_count != counts.dispatches
            || physical_counts.encoder_count != counts.dispatches
            || commits as u64 != counts.dispatches
            || trace_samples.len() as u64 != counts.dispatches
            || counts.command_buffers != counts.dispatches
            || counts.compute_encoders != counts.dispatches
            || counts.waits != counts.dispatches
        {
            return Err(failure(
                "act_quant sweep command-buffer/encoder/dispatch/wait accounting did not reconcile",
            ));
        }
        let winner = passing
            .iter()
            .min_by(|left, right| candidate_order(left, right))
            .copied()
            .ok_or_else(|| failure("act_quant candidate sweep has no winner"))?;
        let deepest = passing
            .iter()
            .max_by_key(|(geometry, _, _, _)| geometry.threads_x)
            .copied()
            .ok_or_else(|| failure("act_quant candidate sweep has no deepest stable rung"))?;

        let unsigned = json!({
            "schema": RECEIPT_SCHEMA,
            "status": RECEIPT_STATUS,
            "scope": {
                "component": "model.linear act_quant only: fixed layer-0 wq_a BF16[4096] input",
                "source_derived_algorithm": true,
                "model_linear_qat_candidate_only": true,
                "not_a_runtime_kernel_promotion": true,
                "not_a_full_model_load": true,
                "not_a_full_model_forward": true,
                "not_token_execution_or_generation": true,
                "not_hcli_endpoint": true,
                "not_base_true_tps_measurement": true,
                "not_registered_43_layer_runtime_adapter": true,
            },
            "artifact": {
                "path": reader.artifact_root().display().to_string(),
                "manifest_schema": FULL_STREAM_SCHEMA,
                "manifest_status": FULL_STREAM_STATUS,
                "manifest_file_sha256": reader.manifest_file_sha256(),
                "manifest_seal_sha256": reader.manifest_seal_sha256(),
                "restart_receipt_seal_sha256": reader.restart_seal_sha256(),
                "source_parent_retained": false,
            },
            "source": {
                "repository": reader.source_identity().repository,
                "revision": reader.source_identity().revision,
                "source_hashes": {
                    "inference/model.py": anchors.inference_model_py_sha256,
                    "inference/kernel.py": anchors.inference_kernel_py_sha256,
                    "inference/config.json": anchors.inference_config_json_sha256,
                    "config.json": anchors.model_config_json_sha256,
                },
                "bounded_tensor_contract": {
                    "weight": LAYER0_WQ_A_WEIGHT,
                    "weight_shape": [LAYER0_WQ_A_ROWS, LAYER0_WQ_A_COLS],
                    "weight_dtype": "F8_E4M3",
                    "weight_scale": LAYER0_WQ_A_SCALE,
                    "weight_scale_shape": [LAYER0_WQ_A_ROWS / ACT_QUANT_BLOCK, LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK],
                    "weight_scale_dtype": "F8_E8M0",
                    "pair_admitted_from_sealed_full_stream": true,
                    "parent_safetensors_materialized": false,
                },
            },
            "canonical_cpu_oracle_v2": {
                "path": cpu_oracle.path,
                "file_sha256": cpu_oracle.file_sha256,
                "receipt_seal_sha256": cpu_oracle.seal_sha256,
                "receipt_seal_verified": true,
                "input_sha256_bf16_le": cpu_oracle.input_sha256_bf16_le,
                "activation_sha256": cpu_oracle.activation_sha256,
                "scale_sha256": cpu_oracle.scale_sha256,
                "direct_act_quant_recomputed_and_matches": true,
            },
            "fixed_before_authority_receipt": {
                "path": authority_receipt.path,
                "file_sha256": authority_receipt.file_sha256,
                "receipt_seal_sha256": authority_receipt.seal_sha256,
                "kernel": AUTHORITY_KERNEL,
                "gpu_duration_us": authority_receipt.gpu_duration_us,
                "gpu_duration_ms": "5.967",
                "byte_exact_activation_and_scale_vs_canonical_cpu_oracle_v2": true,
                "fallback": false,
            },
            "input": {
                "kind": "deterministic_exact_bf16_bitpattern_vector_v1",
                "captured_from_model_forward": false,
                "dtype": "BF16",
                "length": input_bf16.len(),
                "sha256_bf16_le": sha256(&input_bytes),
            },
            "fresh_authority_resweep": {
                "kernel": AUTHORITY_KERNEL,
                "geometry": { "grid_threads": [BLOCKS, 1, 1], "threadgroup_threads": [SIMD_WIDTH, 1, 1] },
                "warmup_dispatches": args.warmups,
                "measured_gpu_timestamped_dispatches": args.trials,
                "timing": authority_timing,
                "activation": { "sha256": sha256(&authority_activation), "bytewise_canonical_cpu_oracle_v2_match": true },
                "scale": { "sha256": sha256(&authority_scales), "bytewise_canonical_cpu_oracle_v2_match": true },
                "gpu_dispatches": args.warmups + args.trials,
                "command_buffers": args.warmups + args.trials,
                "compute_encoders": args.warmups + args.trials,
                "cpu_visible_waits": args.warmups + args.trials,
                "empty_command_buffers": 0,
                "fallback": false,
                "fallback_reason": null,
            },
            "optional_simdgroup_block_candidates": {
                "kernel": CANDIDATE_KERNEL,
                "candidate_rule": "A candidate is admitted only after activation and UE8M0 scale output byte-for-byte equal the canonical CPU oracle v2; no CPU fallback is used for a passing GPU result.",
                "thread_ladder": THREAD_LADDER,
                "vector_width_ladder_bf16_elements": VECTOR_LADDER,
                "candidates": candidates,
                "winner": {
                    "selection": "lowest measured GPU p50, then p95, then p99, then lower threads and vector width",
                    "geometry": geometry_json(winner.0),
                    "gpu_p50_us": winner.1,
                    "gpu_p95_us": winner.2,
                    "gpu_p99_us": winner.3,
                    "speedup_vs_fixed_5967us_authority_gpu_p50": format!("{:.6}", FIXED_AUTHORITY_GPU_US as f64 / winner.1 as f64),
                    "promotion": "MODEL_LINEAR_COMPONENT_QAT_CANDIDATE_ONLY_NOT_RUNTIME_PROMOTION",
                },
                "deepest_stable_thread_rung": {
                    "geometry": geometry_json(deepest.0),
                    "gpu_p50_us": deepest.1,
                    "gpu_p95_us": deepest.2,
                    "gpu_p99_us": deepest.3,
                },
            },
            "metal": {
                "device": device,
                "pipelines_precompiled_before_warmup": true,
                "authority_pipeline_thread_execution_width": authority_thread_execution_width,
                "candidate_pipeline_thread_execution_width": candidate_thread_execution_width,
                "candidate_pipeline_max_total_threads_per_threadgroup": candidate_max_total_threads,
                "buffers_created": buffers_created,
                "bytes_allocated": bytes_allocated,
                "gpu_dispatches": counts.dispatches,
                "command_buffers": counts.command_buffers,
                "compute_encoders": counts.compute_encoders,
                "cpu_visible_waits": counts.waits,
                "empty_command_buffers": 0,
                "physical_trace_command_buffers": physical_counts.command_count,
                "physical_trace_compute_encoders": physical_counts.encoder_count,
                "trace_samples": trace_samples.len(),
                "fallback": false,
                "fallback_count": 0,
                "cpu_used_only_for_direct_oracle_recomputation_and_post_dispatch_byte_comparison": true,
            },
            "logical_bytes": {
                "read_bf16_per_dispatch": input_bytes.len(),
                "written_e4m3fn_plus_e8m0_per_dispatch": LAYER0_WQ_A_COLS + BLOCKS as usize,
                "source_weight_or_model_state_read": 0,
                "note": "This QAT-only component benchmark does not execute the following FP8 projection or any model state.",
            },
            "physical_trace": {
                "interval_id": interval_id,
                "run_nonce": run_nonce,
                "phase": "act_quant_simdgroup_component_sweep",
                "role": "model_linear_qat_candidate_only",
            },
            "claim_boundary": "Real Metal GPU timestamps and byte-exact activation/scale parity are established only for this fixed source-derived model.linear act_quant component. This does not establish a V4 runtime, a full model forward, a token loop, HCLI behavior, or BASE_TRUE_TPS.",
        });
        let (receipt, seal) = seal(unsigned)?;
        write_new_receipt(&args.out, &receipt)?;
        println!(
            "{}",
            serde_json::to_string(&json!({
                "status": RECEIPT_STATUS,
                "receipt": args.out,
                "seal_sha256": seal,
                "winner_threads": winner.0.threads_x,
                "winner_vector_width": winner.0.vector_width,
                "winner_gpu_p50_us": winner.1,
            }))?
        );
        Ok(())
    }

    fn canonical_json(value: &Value) -> Vec<u8> {
        let mut output = Vec::new();
        write_canonical_json(&mut output, value);
        output
    }

    fn write_canonical_json(output: &mut Vec<u8>, value: &Value) {
        match value {
            Value::Null => output.extend_from_slice(b"null"),
            Value::Bool(true) => output.extend_from_slice(b"true"),
            Value::Bool(false) => output.extend_from_slice(b"false"),
            Value::Number(number) => output.extend_from_slice(number.to_string().as_bytes()),
            Value::String(string) => output.extend_from_slice(
                serde_json::to_string(string)
                    .expect("JSON string serialization is infallible")
                    .as_bytes(),
            ),
            Value::Array(values) => {
                output.push(b'[');
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        output.push(b',');
                    }
                    write_canonical_json(output, value);
                }
                output.push(b']');
            }
            Value::Object(object) => {
                output.push(b'{');
                let mut keys: Vec<&String> = object.keys().collect();
                keys.sort();
                for (index, key) in keys.into_iter().enumerate() {
                    if index != 0 {
                        output.push(b',');
                    }
                    output.extend_from_slice(
                        serde_json::to_string(key)
                            .expect("JSON string serialization is infallible")
                            .as_bytes(),
                    );
                    output.push(b':');
                    write_canonical_json(output, &object[key]);
                }
                output.push(b'}');
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

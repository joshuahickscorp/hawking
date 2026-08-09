//! Source-bound packed binary sign + scale Metal matvec component probe.
//!
//! The probe reads one real BF16 `model.layers.0.mlp.gate.weight` tensor from
//! each locally pinned Qwen source, packs it into a 1-bit sign stream plus one
//! FP16 absmax scale per 128 weights, and checks one direct Metal matvec
//! against a CPU oracle of that *same packed representation*. It is only an
//! operator receipt: no decoder, model token loop, capability test, or model
//! TPS is measured or implied.
//!
//! Run from the repository root:
//! `cargo run --release -p hawking-core --example ascension_qwen_binary_sign_scale_metal_probe -- --out workspace/campaign/records/ascension-sandbox/physical/kernel/QWEN_BINARY_SIGN_SCALE_MATVEC_METAL_COMPONENT_PROBE.json`

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
mod macos {
    use half::{bf16, f16};
    use hawking_core::kernels::qwen_binary_sign_scale_matvec_component_tcb;
    use hawking_core::metal::{MetalContext, TokenCommandBuffer};
    use metal::Buffer;
    use serde::Serialize;
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::env;
    use std::error::Error;
    use std::fs::{self, File};
    use std::io::{Read, Seek, SeekFrom};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    const SCHEMA: &str = "hawking.ascension.qwen_binary_sign_scale_metal_component_probe.v1";
    const TENSOR_NAME: &str = "model.layers.0.mlp.gate.weight";
    const GROUP_SIZE: usize = 128;
    const WARMUP_RUNS: usize = 3;
    const MEASURED_RUNS: usize = 24;
    const MAX_SAFETENSORS_HEADER_BYTES: u64 = 64 * 1024 * 1024;

    struct Args {
        qwen30_dir: PathBuf,
        qwen80_dir: PathBuf,
        out: PathBuf,
    }

    #[derive(Clone, Copy)]
    struct LaneSpec {
        model_id: &'static str,
        architecture: &'static str,
    }

    #[derive(Serialize)]
    struct SourceTensorReport {
        model_directory: String,
        config_path: String,
        config_sha256: String,
        safetensors_index_path: String,
        safetensors_shard_path: String,
        safetensors_header_sha256: String,
        tensor_name: &'static str,
        dtype: &'static str,
        shape: [usize; 2],
        tensor_data_offset_bytes_in_shard: u64,
        tensor_data_bytes: usize,
        source_tensor_sha256: String,
    }

    #[derive(Serialize)]
    struct PackingReport {
        group_size: usize,
        sign_bit_order: &'static str,
        sign_payload_bytes: usize,
        fp16_scale_count: usize,
        fp16_scale_payload_bytes: usize,
        packed_payload_bytes: usize,
        effective_payload_bits_per_source_weight: f64,
        source_to_packed_weight_rmse: f64,
        source_to_packed_weight_max_abs_error: f32,
    }

    #[derive(Serialize)]
    struct TimingReport {
        timing_authority: &'static str,
        warmup_runs: usize,
        measured_runs: usize,
        host_wall_us_min: f64,
        host_wall_us_p50: f64,
        host_wall_us_p95: f64,
        host_wall_us_max: f64,
        component_matvecs_per_second_from_host_wall_p50: f64,
        packed_weight_multiply_accumulates_per_second_from_host_wall_p50: f64,
    }

    #[derive(Serialize)]
    struct ParityReport {
        cpu_oracle_definition: &'static str,
        max_abs_output_error: f32,
        output_error_tolerance: f32,
        rows_within_tolerance: bool,
    }

    #[derive(Serialize)]
    struct LaneReport {
        model_id: &'static str,
        architecture: &'static str,
        source: SourceTensorReport,
        packing: PackingReport,
        direct_metal_dispatch: bool,
        dispatches_per_component_matvec: usize,
        parity: ParityReport,
        timing: TimingReport,
    }

    struct LoadedTensor {
        source: SourceTensorReport,
        raw_bf16: Vec<u8>,
        rows: usize,
        cols: usize,
    }

    struct PackedBinary {
        signs: Vec<u8>,
        scales_bits: Vec<u16>,
        source_to_packed_weight_rmse: f64,
        source_to_packed_weight_max_abs_error: f32,
    }

    fn repository_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
    }

    fn parse_args() -> Result<Args, Box<dyn Error>> {
        let root = repository_root();
        let mut qwen30_dir =
            root.join("workspace/campaign/records/runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct");
        let mut qwen80_dir = root.join("workspace/campaign/records/runs/qwen-80b/Qwen3-Coder-Next");
        let mut out = None;
        let mut values = env::args().skip(1);
        while let Some(flag) = values.next() {
            let value = values
                .next()
                .ok_or_else(|| format!("missing value after {flag}"))?;
            match flag.as_str() {
                "--qwen30-dir" => qwen30_dir = PathBuf::from(value),
                "--qwen80-dir" => qwen80_dir = PathBuf::from(value),
                "--out" => out = Some(PathBuf::from(value)),
                _ => return Err(format!("unknown argument {flag}").into()),
            }
        }
        Ok(Args {
            qwen30_dir,
            qwen80_dir,
            out: out.ok_or("--out is required")?,
        })
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn required_usize(value: &Value, field: &str) -> Result<usize, Box<dyn Error>> {
        value
            .get(field)
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .ok_or_else(|| format!("configuration missing unsigned {field}").into())
    }

    fn validate_config(
        config: &Value,
        expected_architecture: &str,
    ) -> Result<usize, Box<dyn Error>> {
        let architectures = config
            .get("architectures")
            .and_then(Value::as_array)
            .ok_or("configuration missing architectures")?;
        if !architectures
            .iter()
            .any(|value| value.as_str() == Some(expected_architecture))
        {
            return Err(format!(
                "configuration architectures do not contain expected {expected_architecture}"
            )
            .into());
        }
        required_usize(config, "hidden_size")
    }

    fn read_tensor(spec: LaneSpec, model_directory: &Path) -> Result<LoadedTensor, Box<dyn Error>> {
        let config_path = model_directory.join("config.json");
        let config_bytes = fs::read(&config_path)?;
        let config: Value = serde_json::from_slice(&config_bytes)?;
        let hidden_size = validate_config(&config, spec.architecture)?;

        let index_path = model_directory.join("model.safetensors.index.json");
        let index: Value = serde_json::from_slice(&fs::read(&index_path)?)?;
        let shard_name = index
            .get("weight_map")
            .and_then(Value::as_object)
            .and_then(|weights| weights.get(TENSOR_NAME))
            .and_then(Value::as_str)
            .ok_or_else(|| format!("safetensors index does not map {TENSOR_NAME}"))?;
        let shard_path = model_directory.join(shard_name);
        let mut shard = File::open(&shard_path)?;
        let mut header_size_bytes = [0u8; 8];
        shard.read_exact(&mut header_size_bytes)?;
        let header_size = u64::from_le_bytes(header_size_bytes);
        if header_size == 0 || header_size > MAX_SAFETENSORS_HEADER_BYTES {
            return Err(format!("invalid safetensors header length {header_size}").into());
        }
        let mut header_bytes = vec![0u8; header_size as usize];
        shard.read_exact(&mut header_bytes)?;
        let header: Value = serde_json::from_slice(&header_bytes)?;
        let entry = header
            .get(TENSOR_NAME)
            .and_then(Value::as_object)
            .ok_or_else(|| format!("safetensors header does not contain {TENSOR_NAME}"))?;
        if entry.get("dtype").and_then(Value::as_str) != Some("BF16") {
            return Err("component probe accepts BF16 source tensors only".into());
        }
        let shape_values = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or("source tensor missing shape")?;
        if shape_values.len() != 2 {
            return Err("source tensor must be rank two for matvec component probe".into());
        }
        let rows = shape_values[0]
            .as_u64()
            .ok_or("source tensor row count is not unsigned")? as usize;
        let cols = shape_values[1]
            .as_u64()
            .ok_or("source tensor column count is not unsigned")? as usize;
        if rows == 0 || cols == 0 || cols != hidden_size {
            return Err(format!(
                "source tensor shape [{rows}, {cols}] does not bind to non-zero hidden_size {hidden_size}"
            )
            .into());
        }
        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or("source tensor missing data_offsets")?;
        if offsets.len() != 2 {
            return Err("source tensor data_offsets must contain exactly two values".into());
        }
        let data_start = offsets[0]
            .as_u64()
            .ok_or("source tensor data start is not unsigned")?;
        let data_end = offsets[1]
            .as_u64()
            .ok_or("source tensor data end is not unsigned")?;
        let element_count = rows
            .checked_mul(cols)
            .ok_or("source tensor element count overflows usize")?;
        let tensor_data_bytes = element_count
            .checked_mul(2)
            .ok_or("source tensor byte count overflows usize")?;
        if data_end.checked_sub(data_start) != Some(tensor_data_bytes as u64) {
            return Err("source tensor BF16 data range does not match declared shape".into());
        }
        let tensor_data_offset_bytes_in_shard = 8u64
            .checked_add(header_size)
            .and_then(|offset| offset.checked_add(data_start))
            .ok_or("source tensor file offset overflows u64")?;
        shard.seek(SeekFrom::Start(tensor_data_offset_bytes_in_shard))?;
        let mut raw_bf16 = vec![0u8; tensor_data_bytes];
        shard.read_exact(&mut raw_bf16)?;
        Ok(LoadedTensor {
            source: SourceTensorReport {
                model_directory: model_directory.display().to_string(),
                config_path: config_path.display().to_string(),
                config_sha256: sha256_hex(&config_bytes),
                safetensors_index_path: index_path.display().to_string(),
                safetensors_shard_path: shard_path.display().to_string(),
                safetensors_header_sha256: sha256_hex(&header_bytes),
                tensor_name: TENSOR_NAME,
                dtype: "BF16",
                shape: [rows, cols],
                tensor_data_offset_bytes_in_shard,
                tensor_data_bytes,
                source_tensor_sha256: sha256_hex(&raw_bf16),
            },
            raw_bf16,
            rows,
            cols,
        })
    }

    fn bf16_at(raw_bf16: &[u8], element: usize) -> f32 {
        let byte = element * 2;
        bf16::from_bits(u16::from_le_bytes([raw_bf16[byte], raw_bf16[byte + 1]])).to_f32()
    }

    fn pack_binary_sign_scale(
        raw_bf16: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<PackedBinary, Box<dyn Error>> {
        let elements = rows
            .checked_mul(cols)
            .ok_or("packed source dimensions overflow usize")?;
        if raw_bf16.len() != elements * 2 {
            return Err("raw BF16 source byte length disagrees with shape".into());
        }
        let groups_per_row = cols.div_ceil(GROUP_SIZE);
        let mut signs = vec![0u8; elements.div_ceil(8)];
        let mut scales_bits = Vec::with_capacity(rows * groups_per_row);
        let mut squared_error = 0.0f64;
        let mut max_abs_error = 0.0f32;
        for row in 0..rows {
            for group in 0..groups_per_row {
                let group_start = group * GROUP_SIZE;
                let group_end = (group_start + GROUP_SIZE).min(cols);
                let row_base = row * cols;
                let mut max_abs = 0.0f32;
                for col in group_start..group_end {
                    max_abs = max_abs.max(bf16_at(raw_bf16, row_base + col).abs());
                }
                let scale = f16::from_f32(max_abs);
                if !scale.to_f32().is_finite() {
                    return Err(
                        "source weight absmax cannot be represented as finite FP16 scale".into(),
                    );
                }
                let reconstructed_scale = scale.to_f32();
                scales_bits.push(scale.to_bits());
                for col in group_start..group_end {
                    let flat = row_base + col;
                    let value = bf16_at(raw_bf16, flat);
                    if value >= 0.0 {
                        signs[flat >> 3] |= 1u8 << (flat & 7);
                    }
                    let reconstructed = if value >= 0.0 {
                        reconstructed_scale
                    } else {
                        -reconstructed_scale
                    };
                    let error = value - reconstructed;
                    squared_error += f64::from(error) * f64::from(error);
                    max_abs_error = max_abs_error.max(error.abs());
                }
            }
        }
        Ok(PackedBinary {
            signs,
            scales_bits,
            source_to_packed_weight_rmse: (squared_error / elements as f64).sqrt(),
            source_to_packed_weight_max_abs_error: max_abs_error,
        })
    }

    fn deterministic_input(cols: usize) -> Vec<f32> {
        (0..cols)
            .map(|index| ((index * 71 % 509) as f32 - 254.0) / 509.0)
            .collect()
    }

    fn cpu_packed_matvec(
        packed: &PackedBinary,
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Vec<f32> {
        let groups_per_row = cols.div_ceil(GROUP_SIZE);
        let mut output = vec![0.0f32; rows];
        for row in 0..rows {
            let mut sum = 0.0f32;
            for col in 0..cols {
                let flat = row * cols + col;
                let scale =
                    f16::from_bits(packed.scales_bits[row * groups_per_row + col / GROUP_SIZE])
                        .to_f32();
                let signed_scale = if ((packed.signs[flat >> 3] >> (flat & 7)) & 1) != 0 {
                    scale
                } else {
                    -scale
                };
                sum += signed_scale * input[col];
            }
            output[row] = sum;
        }
        output
    }

    fn dispatch_component(
        metal: &MetalContext,
        signs: &Buffer,
        scales: &Buffer,
        input: &Buffer,
        output: &Buffer,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let mut tcb = TokenCommandBuffer::new(metal);
        qwen_binary_sign_scale_matvec_component_tcb(
            &mut tcb, signs, scales, input, output, rows, cols, GROUP_SIZE,
        )?;
        if tcb.dispatch_count() != 1 {
            return Err(
                "packed binary matvec component must encode exactly one Metal dispatch".into(),
            );
        }
        tcb.commit_and_wait()?;
        Ok(unsafe { std::slice::from_raw_parts(output.contents() as *const f32, rows).to_vec() })
    }

    fn percentile_us(mut samples: Vec<f64>, percentile: f64) -> f64 {
        samples.sort_by(|left, right| left.total_cmp(right));
        let index = ((samples.len() - 1) as f64 * percentile).ceil() as usize;
        samples[index]
    }

    fn run_lane(
        metal: &MetalContext,
        spec: LaneSpec,
        model_directory: &Path,
    ) -> Result<LaneReport, Box<dyn Error>> {
        let loaded = read_tensor(spec, model_directory)?;
        let packed = pack_binary_sign_scale(&loaded.raw_bf16, loaded.rows, loaded.cols)?;
        let input_values = deterministic_input(loaded.cols);
        let expected = cpu_packed_matvec(&packed, loaded.rows, loaded.cols, &input_values);
        let signs = metal.new_buffer_with_bytes_checked(&packed.signs)?;
        let scales =
            metal.new_buffer_with_bytes_checked(bytemuck::cast_slice(&packed.scales_bits))?;
        let input = metal.new_buffer_with_bytes_checked(bytemuck::cast_slice(&input_values))?;
        let output = metal.new_buffer_checked(loaded.rows * std::mem::size_of::<f32>())?;
        let observed = dispatch_component(
            metal,
            &signs,
            &scales,
            &input,
            &output,
            loaded.rows,
            loaded.cols,
        )?;
        let max_abs_output_error = expected
            .iter()
            .zip(&observed)
            .map(|(expected, observed)| (expected - observed).abs())
            .fold(0.0f32, f32::max);
        const OUTPUT_ERROR_TOLERANCE: f32 = 2.0e-3;
        if max_abs_output_error > OUTPUT_ERROR_TOLERANCE {
            return Err(format!(
                "{} packed binary sign+scale Metal parity failed: max_abs_output_error={max_abs_output_error}",
                spec.model_id
            )
            .into());
        }

        for _ in 0..WARMUP_RUNS {
            let _ = dispatch_component(
                metal,
                &signs,
                &scales,
                &input,
                &output,
                loaded.rows,
                loaded.cols,
            )?;
        }
        let mut samples_us = Vec::with_capacity(MEASURED_RUNS);
        for _ in 0..MEASURED_RUNS {
            let started = Instant::now();
            let _ = dispatch_component(
                metal,
                &signs,
                &scales,
                &input,
                &output,
                loaded.rows,
                loaded.cols,
            )?;
            samples_us.push(started.elapsed().as_secs_f64() * 1_000_000.0);
        }
        let min = samples_us.iter().copied().fold(f64::INFINITY, f64::min);
        let max = samples_us.iter().copied().fold(0.0f64, f64::max);
        let p50 = percentile_us(samples_us.clone(), 0.50);
        let p95 = percentile_us(samples_us, 0.95);
        let packed_payload_bytes = packed.signs.len() + packed.scales_bits.len() * 2;
        let source_weights = loaded.rows * loaded.cols;
        Ok(LaneReport {
            model_id: spec.model_id,
            architecture: spec.architecture,
            source: loaded.source,
            packing: PackingReport {
                group_size: GROUP_SIZE,
                sign_bit_order: "row-major, least-significant-bit first",
                sign_payload_bytes: packed.signs.len(),
                fp16_scale_count: packed.scales_bits.len(),
                fp16_scale_payload_bytes: packed.scales_bits.len() * 2,
                packed_payload_bytes,
                effective_payload_bits_per_source_weight: (packed_payload_bytes * 8) as f64
                    / source_weights as f64,
                source_to_packed_weight_rmse: packed.source_to_packed_weight_rmse,
                source_to_packed_weight_max_abs_error: packed.source_to_packed_weight_max_abs_error,
            },
            direct_metal_dispatch: true,
            dispatches_per_component_matvec: 1,
            parity: ParityReport {
                cpu_oracle_definition:
                    "CPU matvec over the same packed sign bits and FP16-rounded group scales",
                max_abs_output_error,
                output_error_tolerance: OUTPUT_ERROR_TOLERANCE,
                rows_within_tolerance: true,
            },
            timing: TimingReport {
                timing_authority:
                    "host wall time for one completed component command buffer; not GPU-only timing and not token timing",
                warmup_runs: WARMUP_RUNS,
                measured_runs: MEASURED_RUNS,
                host_wall_us_min: min,
                host_wall_us_p50: p50,
                host_wall_us_p95: p95,
                host_wall_us_max: max,
                component_matvecs_per_second_from_host_wall_p50: 1_000_000.0 / p50,
                packed_weight_multiply_accumulates_per_second_from_host_wall_p50:
                    source_weights as f64 * 1_000_000.0 / p50,
            },
        })
    }

    fn write_receipt(out: &Path, report: &Value) -> Result<(), Box<dyn Error>> {
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        let file_name = out
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("receipt output path must have a UTF-8 file name")?;
        let temporary = out.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));
        fs::write(
            &temporary,
            format!("{}\n", serde_json::to_string_pretty(report)?),
        )?;
        fs::rename(temporary, out)?;
        Ok(())
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let args = parse_args()?;
        let metal = MetalContext::new()?;
        let qwen30 = run_lane(
            &metal,
            LaneSpec {
                model_id: "Qwen3-Coder-30B-A3B-Instruct",
                architecture: "Qwen3MoeForCausalLM",
            },
            &args.qwen30_dir,
        )?;
        let qwen80 = run_lane(
            &metal,
            LaneSpec {
                model_id: "Qwen3-Coder-Next-80B",
                architecture: "Qwen3NextForCausalLM",
            },
            &args.qwen80_dir,
        )?;
        let report = json!({
            "schema": SCHEMA,
            "status": "PASS_SOURCE_BOUND_PACKED_BINARY_SIGN_SCALE_MATVEC_COMPONENT_NOT_FULL_MODEL_NOT_TPS_GATE",
            "device": metal.device_name(),
            "lanes": [qwen30, qwen80],
            "claim_boundary": {
                "each_lane_reads_one_real_raw_bf16_gate_tensor_and_records_its_sha256": true,
                "each_lane_executes_one_direct_metal_packed_binary_sign_scale_matvec_component": true,
                "parity_is_against_cpu_execution_of_the_same_lossy_packed_representation": true,
                "source_to_packed_weight_error_is_not_model_quality_or_capability_evidence": true,
                "qkv_projection_attention_deltanet_moe_residual_norm_lm_head_and_token_loop_not_run": true,
                "component_matvecs_per_second_is_not_tokens_per_second": true,
                "not_a_full_model_or_manager_qualification": true,
                "not_a_100_tps_kernel_operational_receipt": true,
                "not_a_tg3_333_tps_receipt": true,
                "not_a_tournament_admission_or_winner_selection_receipt": true
            }
        });
        write_receipt(&args.out, &report)?;
        println!("{}", serde_json::to_string(&report)?);
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

//! Source-bound all-row Qwen30 direct-packed gate/up/SwiGLU component candidate.
//!
//! The experiment opens exactly two admitted `HQ30G1B1` routed-expert
//! projections, computes a packed CPU authority, verifies Metal control and
//! fused paths against it, and optionally measures only the completed
//! component command buffers.  It deliberately does not load raw BF16,
//! construct decoded weights, call MPS, run a layer/token/model, or calculate
//! TPS.  The production runtime does not select this candidate.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "ascension_qwen30_direct_packed_gate_up_swiglu_fused requires macOS Metal",
    )
    .into())
}

#[cfg(target_os = "macos")]
mod macos {
    use half::f16;
    use hawking_core::model::qwen_complete_binary::{
        admit_complete_binary_artifact, parse_complete_binary_header, CompleteBinaryAdmission,
        QwenCompleteBinaryModel,
    };
    use metal::{CompileOptions, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
    use serde_json::{json, Value};
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    const SCHEMA: &str = "hawking.ascension.qwen30_direct_packed_gate_up_swiglu_fused_component.v1";
    const PARITY_STATUS: &str =
        "PASS_DIRECT_PACKED_QWEN30_GATE_UP_SWIGLU_FUSED_CPU_METAL_PARITY_COMPONENT_ONLY";
    const BENCHMARK_STATUS: &str =
        "PASS_DIRECT_PACKED_QWEN30_GATE_UP_SWIGLU_FUSED_COMPONENT_BENCHMARK_NOT_MODEL_TPS";
    const GROUP_SIZE: usize = 128;
    const EXPERT_ROWS: usize = 768;
    const HIDDEN_COLS: usize = 2048;
    const DEFAULT_WARMUPS: usize = 3;
    const DEFAULT_TRIALS: usize = 11;
    const MAX_TRIALS: usize = 31;
    const TOLERANCE: f32 = 4.0e-3;
    const SHADER_SOURCE: &str =
        include_str!("../shaders/qwen_direct_packed_gate_up_swiglu_fused.metal");

    type Probe<T> = Result<T, Box<dyn Error>>;

    struct Args {
        manifest: PathBuf,
        expected_manifest_seal_sha256: String,
        expected_source_audit_seal_sha256: String,
        expected_source_revision: String,
        layer: usize,
        expert: usize,
        warmups: usize,
        trials: usize,
        benchmark: bool,
        out: PathBuf,
    }

    struct PackedProjection {
        name: String,
        signs: Vec<u8>,
        scales: Vec<u8>,
        rows: usize,
        cols: usize,
        groups: usize,
        header_bytes: usize,
        payload_bytes: usize,
        payload_sha256: String,
        artifact_path: String,
    }

    struct Buffers {
        gate_signs: metal::Buffer,
        gate_scales: metal::Buffer,
        up_signs: metal::Buffer,
        up_scales: metal::Buffer,
        input: metal::Buffer,
        baseline_gate: metal::Buffer,
        baseline_up: metal::Buffer,
        baseline_activation: metal::Buffer,
        fused_activation: metal::Buffer,
    }

    struct Runner {
        device: Device,
        queue: metal::CommandQueue,
        baseline_matvec: metal::ComputePipelineState,
        baseline_swiglu: metal::ComputePipelineState,
        fused: metal::ComputePipelineState,
    }

    fn failure(message: impl Into<String>) -> Box<dyn Error> {
        std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
    }

    fn usage() -> &'static str {
        "usage: ascension_qwen30_direct_packed_gate_up_swiglu_fused \\
            --manifest ABSOLUTE_PATH \\
            --expected-manifest-seal-sha256 SHA256 \\
            --expected-source-audit-seal-sha256 SHA256 \\
            --expected-source-revision REVISION \\
            --out ABSOLUTE_PATH [--layer N] [--expert N] [--warmups N] [--trials N] [--benchmark]"
    }

    fn required<T>(value: Option<T>, flag: &str) -> Probe<T> {
        value.ok_or_else(|| failure(format!("missing {flag}; {}", usage())))
    }

    fn parse_usize(value: &str, flag: &str) -> Probe<usize> {
        value.parse::<usize>().map_err(|_| {
            failure(format!(
                "{flag} must be an unsigned decimal integer; {}",
                usage()
            ))
        })
    }

    fn parse_args() -> Probe<Args> {
        let mut manifest = None;
        let mut expected_manifest_seal_sha256 = None;
        let mut expected_source_audit_seal_sha256 = None;
        let mut expected_source_revision = None;
        let mut layer = 0usize;
        let mut expert = 0usize;
        let mut warmups = DEFAULT_WARMUPS;
        let mut trials = DEFAULT_TRIALS;
        let mut benchmark = false;
        let mut out = None;
        let mut values = env::args().skip(1);
        while let Some(flag) = values.next() {
            if flag == "--benchmark" {
                if benchmark {
                    return Err(failure(format!(
                        "--benchmark supplied more than once; {}",
                        usage()
                    )));
                }
                benchmark = true;
                continue;
            }
            let value = values
                .next()
                .ok_or_else(|| failure(format!("missing value for {flag:?}; {}", usage())))?;
            match flag.as_str() {
                "--manifest" => {
                    if manifest.replace(PathBuf::from(value)).is_some() {
                        return Err(failure(format!(
                            "--manifest supplied more than once; {}",
                            usage()
                        )));
                    }
                }
                "--expected-manifest-seal-sha256" => {
                    if expected_manifest_seal_sha256.replace(value).is_some() {
                        return Err(failure(format!(
                            "--expected-manifest-seal-sha256 supplied more than once; {}",
                            usage()
                        )));
                    }
                }
                "--expected-source-audit-seal-sha256" => {
                    if expected_source_audit_seal_sha256.replace(value).is_some() {
                        return Err(failure(format!(
                            "--expected-source-audit-seal-sha256 supplied more than once; {}",
                            usage()
                        )));
                    }
                }
                "--expected-source-revision" => {
                    if expected_source_revision.replace(value).is_some() {
                        return Err(failure(format!(
                            "--expected-source-revision supplied more than once; {}",
                            usage()
                        )));
                    }
                }
                "--layer" => layer = parse_usize(&value, "--layer")?,
                "--expert" => expert = parse_usize(&value, "--expert")?,
                "--warmups" => warmups = parse_usize(&value, "--warmups")?,
                "--trials" => trials = parse_usize(&value, "--trials")?,
                "--out" => {
                    if out.replace(PathBuf::from(value)).is_some() {
                        return Err(failure(format!(
                            "--out supplied more than once; {}",
                            usage()
                        )));
                    }
                }
                _ => return Err(failure(format!("unsupported option {flag:?}; {}", usage()))),
            }
        }
        let manifest = required(manifest, "--manifest")?;
        let out = required(out, "--out")?;
        if !manifest.is_absolute() || !out.is_absolute() {
            return Err(failure("--manifest and --out must be absolute paths"));
        }
        if warmups == 0 || trials == 0 || trials > MAX_TRIALS {
            return Err(failure(format!(
                "warmups must be positive and trials must be in 1..={MAX_TRIALS}"
            )));
        }
        if layer >= 48 || expert >= 128 {
            return Err(failure(
                "Qwen30 fused candidate requires layer < 48 and expert < 128",
            ));
        }
        Ok(Args {
            manifest,
            expected_manifest_seal_sha256: required(
                expected_manifest_seal_sha256,
                "--expected-manifest-seal-sha256",
            )?,
            expected_source_audit_seal_sha256: required(
                expected_source_audit_seal_sha256,
                "--expected-source-audit-seal-sha256",
            )?,
            expected_source_revision: required(
                expected_source_revision,
                "--expected-source-revision",
            )?,
            layer,
            expert,
            warmups,
            trials,
            benchmark,
            out,
        })
    }

    fn projection(
        artifact: &hawking_core::model::qwen_complete_binary::CompleteBinaryArtifact,
        name: String,
    ) -> Probe<PackedProjection> {
        let tensor = artifact
            .tensor(&name)
            .map_err(|error| failure(error.to_string()))?;
        let payload = artifact
            .read_tensor_payload(&name)
            .map_err(|error| failure(error.to_string()))?;
        let header =
            parse_complete_binary_header(&payload).map_err(|error| failure(error.to_string()))?;
        if header.group_size != GROUP_SIZE
            || header.shape != [EXPERT_ROWS, HIDDEN_COLS]
            || header.groups != EXPERT_ROWS * (HIDDEN_COLS / GROUP_SIZE)
        {
            return Err(failure(format!(
                "{name} is not the exact Qwen30 expert geometry [768, 2048] at group_size=128"
            )));
        }
        let scales = payload[header.scale_offset..header.sign_offset].to_vec();
        let signs = payload[header.sign_offset..header.payload_bytes].to_vec();
        Ok(PackedProjection {
            name,
            signs,
            scales,
            rows: header.shape[0],
            cols: header.shape[1],
            groups: header.groups,
            header_bytes: header.scale_offset,
            payload_bytes: header.payload_bytes,
            payload_sha256: tensor.artifact_sha256.clone(),
            artifact_path: tensor.artifact_path.display().to_string(),
        })
    }

    fn direct_value(projection: &PackedProjection, index: usize) -> f32 {
        let group = index / GROUP_SIZE;
        let bit = index % GROUP_SIZE;
        let scale_offset = group * std::mem::size_of::<u16>();
        let scale = f16::from_bits(u16::from_le_bytes([
            projection.scales[scale_offset],
            projection.scales[scale_offset + 1],
        ]))
        .to_f32();
        let positive =
            ((projection.signs[group * (GROUP_SIZE / 8) + bit / 8] >> (bit % 8)) & 1) != 0;
        if positive {
            scale
        } else {
            -scale
        }
    }

    fn cpu_matvec(projection: &PackedProjection, input: &[f32]) -> Vec<f32> {
        (0..projection.rows)
            .map(|row| {
                let mut sum = 0.0f32;
                for column in 0..projection.cols {
                    sum = direct_value(projection, row * projection.cols + column)
                        .mul_add(input[column], sum);
                }
                sum
            })
            .collect()
    }

    fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
        gate.iter()
            .zip(up)
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect()
    }

    fn deterministic_input(cols: usize) -> Vec<f32> {
        (0..cols)
            .map(|index| ((index * 71 % 509) as f32 - 254.0) / 509.0)
            .collect()
    }

    fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
        encoder.set_bytes(
            index,
            std::mem::size_of::<u32>() as u64,
            &value as *const u32 as *const _,
        );
    }

    fn read_f32(buffer: &metal::Buffer, count: usize) -> Probe<Vec<f32>> {
        if buffer.length() < (count * std::mem::size_of::<f32>()) as u64 {
            return Err(failure("output buffer is too short"));
        }
        Ok(unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, count).to_vec() })
    }

    impl Runner {
        fn new() -> Probe<Self> {
            let device =
                Device::system_default().ok_or_else(|| failure("Metal device unavailable"))?;
            if !device.has_unified_memory() {
                return Err(failure(
                    "candidate requires unified-memory Apple Silicon Metal",
                ));
            }
            let options = CompileOptions::new();
            let library = device
                .new_library_with_source(SHADER_SOURCE, &options)
                .map_err(failure)?;
            let baseline_matvec_function = library
                .get_function("qwen_direct_packed_gate_up_baseline_matvec", None)
                .map_err(failure)?;
            let baseline_swiglu_function = library
                .get_function("qwen_direct_packed_gate_up_baseline_swiglu", None)
                .map_err(failure)?;
            let fused_function = library
                .get_function("qwen_direct_packed_gate_up_swiglu_fused_candidate", None)
                .map_err(failure)?;
            Ok(Self {
                queue: device.new_command_queue(),
                baseline_matvec: device
                    .new_compute_pipeline_state_with_function(&baseline_matvec_function)
                    .map_err(failure)?,
                baseline_swiglu: device
                    .new_compute_pipeline_state_with_function(&baseline_swiglu_function)
                    .map_err(failure)?,
                fused: device
                    .new_compute_pipeline_state_with_function(&fused_function)
                    .map_err(failure)?,
                device,
            })
        }

        fn byte_buffer(&self, values: &[u8]) -> metal::Buffer {
            let buffer = self
                .device
                .new_buffer(values.len() as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    buffer.contents() as *mut u8,
                    values.len(),
                )
            };
            buffer
        }

        fn f32_buffer(&self, values: &[f32]) -> metal::Buffer {
            let buffer = self.device.new_buffer(
                (values.len() * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr() as *const u8,
                    buffer.contents() as *mut u8,
                    values.len() * std::mem::size_of::<f32>(),
                )
            };
            buffer
        }

        fn buffers(
            &self,
            gate: &PackedProjection,
            up: &PackedProjection,
            input: &[f32],
        ) -> Buffers {
            Buffers {
                gate_signs: self.byte_buffer(&gate.signs),
                gate_scales: self.byte_buffer(&gate.scales),
                up_signs: self.byte_buffer(&up.signs),
                up_scales: self.byte_buffer(&up.scales),
                input: self.f32_buffer(input),
                baseline_gate: self.f32_buffer(&vec![0.0; gate.rows]),
                baseline_up: self.f32_buffer(&vec![0.0; gate.rows]),
                baseline_activation: self.f32_buffer(&vec![0.0; gate.rows]),
                fused_activation: self.f32_buffer(&vec![0.0; gate.rows]),
            }
        }

        fn complete(command: &metal::CommandBufferRef, label: &str) -> Probe<()> {
            command.commit();
            command.wait_until_completed();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(failure(format!(
                    "{label} did not complete: {:?}",
                    command.status()
                )));
            }
            Ok(())
        }

        fn baseline(&self, buffers: &Buffers, rows: usize, cols: usize) -> Probe<()> {
            let command = self.queue.new_command_buffer();
            for (signs, scales, output) in [
                (
                    &buffers.gate_signs,
                    &buffers.gate_scales,
                    &buffers.baseline_gate,
                ),
                (&buffers.up_signs, &buffers.up_scales, &buffers.baseline_up),
            ] {
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.baseline_matvec);
                encoder.set_buffer(0, Some(signs), 0);
                encoder.set_buffer(1, Some(scales), 0);
                encoder.set_buffer(2, Some(&buffers.input), 0);
                encoder.set_buffer(3, Some(output), 0);
                set_u32(encoder, 4, rows as u32);
                set_u32(encoder, 5, cols as u32);
                set_u32(encoder, 6, GROUP_SIZE as u32);
                encoder.dispatch_threads(
                    MTLSize::new(rows as u64, 1, 1),
                    MTLSize::new(rows.min(256) as u64, 1, 1),
                );
                encoder.end_encoding();
            }
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.baseline_swiglu);
            encoder.set_buffer(0, Some(&buffers.baseline_gate), 0);
            encoder.set_buffer(1, Some(&buffers.baseline_up), 0);
            encoder.set_buffer(2, Some(&buffers.baseline_activation), 0);
            set_u32(encoder, 3, rows as u32);
            encoder.dispatch_threads(
                MTLSize::new(rows as u64, 1, 1),
                MTLSize::new(rows.min(256) as u64, 1, 1),
            );
            encoder.end_encoding();
            Self::complete(
                command,
                "three-dispatch direct-packed gate/up/SwiGLU baseline",
            )
        }

        fn fused(&self, buffers: &Buffers, rows: usize, cols: usize) -> Probe<()> {
            let command = self.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.fused);
            encoder.set_buffer(0, Some(&buffers.gate_signs), 0);
            encoder.set_buffer(1, Some(&buffers.gate_scales), 0);
            encoder.set_buffer(2, Some(&buffers.up_signs), 0);
            encoder.set_buffer(3, Some(&buffers.up_scales), 0);
            encoder.set_buffer(4, Some(&buffers.input), 0);
            encoder.set_buffer(5, Some(&buffers.fused_activation), 0);
            set_u32(encoder, 6, rows as u32);
            set_u32(encoder, 7, cols as u32);
            set_u32(encoder, 8, GROUP_SIZE as u32);
            encoder.dispatch_threads(
                MTLSize::new(rows as u64, 1, 1),
                MTLSize::new(rows.min(256) as u64, 1, 1),
            );
            encoder.end_encoding();
            Self::complete(
                command,
                "one-dispatch direct-packed gate/up/SwiGLU fused candidate",
            )
        }
    }

    fn percentile(mut values: Vec<f64>, fraction: f64) -> Probe<f64> {
        if values.is_empty() {
            return Err(failure("cannot summarize an empty sample series"));
        }
        values.sort_by(f64::total_cmp);
        let index = ((values.len() - 1) as f64 * fraction).ceil() as usize;
        Ok(values[index])
    }

    fn summary(values: &[f64]) -> Probe<Value> {
        Ok(json!({
            "sample_count": values.len(),
            "host_wall_us_min": values.iter().copied().fold(f64::INFINITY, f64::min),
            "host_wall_us_p50": percentile(values.to_vec(), 0.50)?,
            "host_wall_us_p95": percentile(values.to_vec(), 0.95)?,
            "host_wall_us_max": values.iter().copied().fold(0.0, f64::max),
            "timing_authority": "host wall around one completed direct-packed component command buffer; not GPU-only timing and not model/token timing",
        }))
    }

    fn max_error(expected: &[f32], actual: &[f32]) -> f32 {
        expected
            .iter()
            .zip(actual)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max)
    }

    fn byte_ledger(gate: &PackedProjection, up: &PackedProjection) -> Probe<Value> {
        if gate.rows != up.rows || gate.cols != up.cols || gate.groups != up.groups {
            return Err(failure("gate/up geometry diverged before byte ledger"));
        }
        let f32_bytes = std::mem::size_of::<f32>();
        let rows = gate.rows;
        let cols = gate.cols;
        let output_bytes = rows
            .checked_mul(f32_bytes)
            .ok_or_else(|| failure("expert output byte count overflow"))?;
        let input_per_projection = rows
            .checked_mul(cols)
            .and_then(|value| value.checked_mul(f32_bytes))
            .ok_or_else(|| failure("logical input stream byte count overflow"))?;
        let pair_payload_bytes = gate
            .payload_bytes
            .checked_add(up.payload_bytes)
            .ok_or_else(|| failure("pair payload byte count overflow"))?;
        let pair_body_bytes = gate
            .signs
            .len()
            .checked_add(gate.scales.len())
            .and_then(|value| value.checked_add(up.signs.len()))
            .and_then(|value| value.checked_add(up.scales.len()))
            .ok_or_else(|| failure("pair packed body byte count overflow"))?;
        let baseline_workspace = output_bytes
            .checked_mul(3)
            .ok_or_else(|| failure("baseline workspace byte count overflow"))?;
        let candidate_workspace = output_bytes;
        let baseline_source_level_f32_reads = input_per_projection
            .checked_mul(2)
            .ok_or_else(|| failure("baseline input load byte count overflow"))?;
        Ok(json!({
            "authority": "exact admitted buffer geometry and source-level materialization ledger; it is not a cache model, hardware bandwidth measurement, or TPS claim",
            "geometry": {
                "rows": rows,
                "cols": cols,
                "elements_per_projection": rows * cols,
                "group_size": GROUP_SIZE,
                "groups_per_projection": gate.groups,
                "groups_per_row": cols / GROUP_SIZE,
                "sign_bit_order": "little",
                "scale_dtype": "float16",
            },
            "exact_resident_admitted_payload_bytes": {
                "gate_header": gate.header_bytes,
                "gate_scales": gate.scales.len(),
                "gate_signs": gate.signs.len(),
                "gate_total": gate.payload_bytes,
                "up_header": up.header_bytes,
                "up_scales": up.scales.len(),
                "up_signs": up.signs.len(),
                "up_total": up.payload_bytes,
                "pair_total_including_headers": pair_payload_bytes,
                "pair_direct_packed_body_excluding_headers": pair_body_bytes,
            },
            "exact_f32_workspace_bytes": {
                "baseline_gate_up_activation": baseline_workspace,
                "candidate_activation_only": candidate_workspace,
                "candidate_eliminates_materialized_gate_and_up": baseline_workspace - candidate_workspace,
            },
            "source_level_operand_ledger": {
                "baseline_three_dispatch_input_f32_load_bytes": baseline_source_level_f32_reads,
                "candidate_one_dispatch_input_f32_load_bytes": input_per_projection,
                "candidate_input_f32_load_bytes_eliminated": baseline_source_level_f32_reads - input_per_projection,
                "baseline_direct_packed_weight_body_is_not_decoded": true,
                "candidate_direct_packed_weight_body_is_not_decoded": true,
                "hardware_cache_or_bandwidth_inference_forbidden": true,
            },
            "exact_command_topology": {
                "baseline": {"command_buffers": 1, "compute_dispatches": 3, "gate_output_materialized": true, "up_output_materialized": true, "activation_output_materialized": true},
                "candidate": {"command_buffers": 1, "compute_dispatches": 1, "gate_output_materialized": false, "up_output_materialized": false, "activation_output_materialized": true},
            },
        }))
    }

    fn atomic_json(path: &Path, document: &Value) -> Probe<()> {
        if path.exists() {
            return Err(failure(format!(
                "refusing to overwrite existing component receipt {}",
                path.display()
            )));
        }
        let parent = path
            .parent()
            .ok_or_else(|| failure("--out has no parent directory"))?;
        fs::create_dir_all(parent)?;
        let temporary = path.with_file_name(format!(
            ".{}.{}.tmp",
            path.file_name()
                .and_then(|v| v.to_str())
                .ok_or_else(|| failure("--out name is not UTF-8"))?,
            std::process::id()
        ));
        fs::write(
            &temporary,
            format!("{}\n", serde_json::to_string_pretty(document)?),
        )?;
        fs::rename(temporary, path)?;
        Ok(())
    }

    pub fn run() -> Probe<()> {
        let args = parse_args()?;
        let admission = CompleteBinaryAdmission {
            model: QwenCompleteBinaryModel::Qwen30Coder,
            expected_manifest_seal_sha256: args.expected_manifest_seal_sha256.clone(),
            expected_source_audit_seal_sha256: args.expected_source_audit_seal_sha256.clone(),
            expected_source_revision: args.expected_source_revision.clone(),
        };
        let artifact = admit_complete_binary_artifact(&args.manifest, &admission)
            .map_err(|error| failure(error.to_string()))?;
        let prefix = format!("model.layers.{}.mlp.experts.{}", args.layer, args.expert);
        let gate = projection(&artifact, format!("{prefix}.gate_proj.weight"))?;
        let up = projection(&artifact, format!("{prefix}.up_proj.weight"))?;
        if gate.rows != up.rows || gate.cols != up.cols {
            return Err(failure(
                "admitted gate/up projections have mismatched geometry",
            ));
        }

        let input = deterministic_input(gate.cols);
        let cpu_gate = cpu_matvec(&gate, &input);
        let cpu_up = cpu_matvec(&up, &input);
        let cpu_activation = swiglu(&cpu_gate, &cpu_up);
        let runner = Runner::new()?;
        let buffers = runner.buffers(&gate, &up, &input);

        runner.baseline(&buffers, gate.rows, gate.cols)?;
        let baseline_gate = read_f32(&buffers.baseline_gate, gate.rows)?;
        let baseline_up = read_f32(&buffers.baseline_up, gate.rows)?;
        let baseline_activation = read_f32(&buffers.baseline_activation, gate.rows)?;
        runner.fused(&buffers, gate.rows, gate.cols)?;
        let fused_activation = read_f32(&buffers.fused_activation, gate.rows)?;

        let baseline_gate_error = max_error(&cpu_gate, &baseline_gate);
        let baseline_up_error = max_error(&cpu_up, &baseline_up);
        let baseline_activation_error = max_error(&cpu_activation, &baseline_activation);
        let fused_activation_error = max_error(&cpu_activation, &fused_activation);
        let fused_vs_baseline_error = max_error(&baseline_activation, &fused_activation);
        if baseline_gate_error > TOLERANCE
            || baseline_up_error > TOLERANCE
            || baseline_activation_error > TOLERANCE
            || fused_activation_error > TOLERANCE
            || fused_vs_baseline_error > TOLERANCE
        {
            return Err(failure(format!(
                "direct-packed all-row CPU/Metal parity failed: gate={baseline_gate_error}, up={baseline_up_error}, baseline_activation={baseline_activation_error}, fused_activation={fused_activation_error}, fused_vs_baseline={fused_vs_baseline_error}, tolerance={TOLERANCE}"
            )));
        }

        let timing = if args.benchmark {
            for _ in 0..args.warmups {
                runner.baseline(&buffers, gate.rows, gate.cols)?;
                runner.fused(&buffers, gate.rows, gate.cols)?;
            }
            let mut baseline_us = Vec::with_capacity(args.trials);
            let mut fused_us = Vec::with_capacity(args.trials);
            for _ in 0..args.trials {
                let started = Instant::now();
                runner.baseline(&buffers, gate.rows, gate.cols)?;
                baseline_us.push(started.elapsed().as_secs_f64() * 1_000_000.0);
                let started = Instant::now();
                runner.fused(&buffers, gate.rows, gate.cols)?;
                fused_us.push(started.elapsed().as_secs_f64() * 1_000_000.0);
            }
            let baseline_summary = summary(&baseline_us)?;
            let fused_summary = summary(&fused_us)?;
            let baseline_p50 = baseline_summary["host_wall_us_p50"]
                .as_f64()
                .ok_or_else(|| failure("baseline p50 missing"))?;
            let fused_p50 = fused_summary["host_wall_us_p50"]
                .as_f64()
                .ok_or_else(|| failure("fused p50 missing"))?;
            json!({
                "benchmark_ran": true,
                "warmups": args.warmups,
                "trials": args.trials,
                "baseline_three_dispatch": baseline_summary,
                "candidate_one_dispatch": fused_summary,
                "p50_component_host_wall_delta_us": fused_p50 - baseline_p50,
                "p50_component_host_wall_speedup_ratio": baseline_p50 / fused_p50,
                "not_a_model_or_token_rate": true,
            })
        } else {
            json!({
                "benchmark_ran": false,
                "reason": "parity-only invocation; --benchmark is explicit so a shared GPU is never silently benchmarked",
                "not_a_model_or_token_rate": true,
            })
        };

        let status = if args.benchmark {
            BENCHMARK_STATUS
        } else {
            PARITY_STATUS
        };
        let document = json!({
            "schema": SCHEMA,
            "status": status,
            "binding": {
                "manifest_path": args.manifest,
                "manifest_seal_sha256": artifact.manifest_seal_sha256,
                "source_audit_seal_sha256": artifact.source_audit_seal_sha256,
                "source_revision": artifact.source_revision,
                "admitted_tensor_count": artifact.tensors.len(),
            },
            "candidate": {
                "id": "qwen30-direct-packed-all-row-gate-up-swiglu-fused",
                "shader": "crates/hawking-core/shaders/qwen_direct_packed_gate_up_swiglu_fused.metal",
                "layer": args.layer,
                "expert": args.expert,
                "all_rows_processed": gate.rows,
                "direct_packed_layout": "HQ30G1B1, group_size=128, FP16 scales plus LSB-first sign bits",
                "baseline_component": "two direct packed matvecs plus separately materialized SwiGLU",
                "candidate_component": "one direct packed gate/up reduction plus in-kernel SwiGLU output",
            },
            "tensors": {
                "gate": {"name": gate.name, "artifact_path": gate.artifact_path, "artifact_sha256": gate.payload_sha256, "shape": [gate.rows, gate.cols], "groups": gate.groups, "header_bytes": gate.header_bytes, "scale_bytes": gate.scales.len(), "sign_bytes": gate.signs.len(), "payload_bytes": gate.payload_bytes},
                "up": {"name": up.name, "artifact_path": up.artifact_path, "artifact_sha256": up.payload_sha256, "shape": [up.rows, up.cols], "groups": up.groups, "header_bytes": up.header_bytes, "scale_bytes": up.scales.len(), "sign_bytes": up.signs.len(), "payload_bytes": up.payload_bytes},
            },
            "parity": {
                "cpu_oracle": "Rust f32 direct matvec and SwiGLU over the exact admitted sign-bit/FP16-scale buffers",
                "input": "deterministic f32 vector; no source BF16 weights, decoded weight body, MPS, raw model, or alternate model path",
                "tolerance_max_abs": TOLERANCE,
                "baseline_gate_max_abs_error": baseline_gate_error,
                "baseline_up_max_abs_error": baseline_up_error,
                "baseline_swiglu_max_abs_error": baseline_activation_error,
                "fused_swiglu_max_abs_error": fused_activation_error,
                "fused_vs_baseline_swiglu_max_abs_error": fused_vs_baseline_error,
                "all_within_tolerance": true,
            },
            "geometry_and_byte_ledger": byte_ledger(&gate, &up)?,
            "timing": timing,
            "integration_notes": {
                "runtime_was_not_modified": true,
                "candidate_replaces_only_one_selected_route's gate/up/SwiGLU triple": true,
                "required_before_runtime_selection": [
                    "wire a separately named runtime candidate path with route-major output offsets",
                    "prove all selected experts across early, middle, late, hot, and cold layers against the current direct packed control path",
                    "re-run full-token native capability and numerical checks on the exact artifact",
                    "re-run a fresh complete-token latency profile with unambiguous >=98 percent host-stage coverage",
                    "run clean HCLI generation and official TPS gates; this component receipt cannot promote any of those gates",
                ],
                "rollback": "retain the existing independent direct-packed gate, up, and SwiGLU dispatches as the control path until every listed gate passes",
            },
            "claim_boundary": {
                "opens_only_admitted_direct_binary_qwen30_payloads": true,
                "decoded_weight_materialization": false,
                "raw_bf16_or_mps_full_model_opened": false,
                "full_layer_token_generation_hcli_capability_or_tps_claim": false,
                "tg_or_tournament_receipt": false,
                "component_only": true,
            },
        });
        atomic_json(&args.out, &document)?;
        println!("{}", serde_json::to_string(&document)?);
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

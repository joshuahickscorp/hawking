//! Source-bound Qwen30 direct-packed gate/up command-topology candidate.
//!
//! This is intentionally a small, isolated kernel experiment.  It admits the
//! exact current Qwen30 complete-binary artifact, opens two routed-expert
//! projections from that pack, and compares a two-dispatch baseline with a
//! one-dispatch paired candidate.  CPU parity is over the *same admitted
//! sign-bit/FP16-scale direct binary values*.  No raw BF16 tensor is opened,
//! no MPS path exists, and no model TPS is calculated.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(
        std::io::Error::other("ascension_qwen30_direct_packed_gate_up_pair requires macOS Metal")
            .into(),
    )
}

#[cfg(target_os = "macos")]
mod macos {
    use half::f16;
    use hawking_core::model::qwen_complete_binary::{
        admit_complete_binary_artifact, parse_complete_binary_header, CompleteBinaryAdmission,
        QwenCompleteBinaryModel,
    };
    use metal::{CompileOptions, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
    use serde_json::json;
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;

    const SCHEMA: &str = "hawking.ascension.qwen30_direct_packed_gate_up_pair_component.v1";
    const STATUS: &str =
        "PASS_DIRECT_PACKED_QWEN30_GATE_UP_PAIR_COMPONENT_CPU_PARITY_NOT_MODEL_TPS";
    const GROUP_SIZE: usize = 128;
    const DEFAULT_WARMUPS: usize = 3;
    const DEFAULT_TRIALS: usize = 11;
    const MAX_TRIALS: usize = 31;
    const TOLERANCE: f32 = 2.0e-3;
    const SHADER_SOURCE: &str = include_str!("../shaders/qwen_direct_packed_gate_up_pair.metal");

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
        out: PathBuf,
    }

    struct PackedProjection {
        name: String,
        signs: Vec<u8>,
        scales: Vec<u8>,
        rows: usize,
        cols: usize,
        payload_sha256: String,
        artifact_path: String,
    }

    struct Buffers {
        gate_signs: metal::Buffer,
        gate_scales: metal::Buffer,
        up_signs: metal::Buffer,
        up_scales: metal::Buffer,
        input: metal::Buffer,
        gate_output: metal::Buffer,
        up_output: metal::Buffer,
    }

    struct Runner {
        device: Device,
        queue: metal::CommandQueue,
        baseline: metal::ComputePipelineState,
        paired: metal::ComputePipelineState,
    }

    fn failure(message: impl Into<String>) -> Box<dyn Error> {
        std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
    }

    fn usage() -> &'static str {
        "usage: ascension_qwen30_direct_packed_gate_up_pair \\
            --manifest ABSOLUTE_PATH \\
            --expected-manifest-seal-sha256 SHA256 \\
            --expected-source-audit-seal-sha256 SHA256 \\
            --expected-source-revision REVISION \\
            --out ABSOLUTE_PATH [--layer N] [--expert N] [--warmups N] [--trials N]"
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
        let mut out = None;
        let mut values = env::args().skip(1);
        while let Some(flag) = values.next() {
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
                "Qwen30 direct candidate requires layer < 48 and expert < 128",
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
        if header.group_size != GROUP_SIZE || header.shape.len() != 2 || header.shape[1] != 2048 {
            return Err(failure(format!(
                "{name} direct packed geometry is not [rows, 2048] with group_size=128"
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
            payload_sha256: tensor.artifact_sha256.clone(),
            artifact_path: tensor.artifact_path.display().to_string(),
        })
    }

    fn direct_value(projection: &PackedProjection, index: usize) -> f32 {
        let group = index / GROUP_SIZE;
        let bit = index % GROUP_SIZE;
        let scale_offset = group * 2;
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
            let baseline_function = library
                .get_function("qwen_direct_packed_matvec_baseline", None)
                .map_err(failure)?;
            let paired_function = library
                .get_function("qwen_direct_packed_gate_up_pair_candidate", None)
                .map_err(failure)?;
            Ok(Self {
                queue: device.new_command_queue(),
                baseline: device
                    .new_compute_pipeline_state_with_function(&baseline_function)
                    .map_err(failure)?,
                paired: device
                    .new_compute_pipeline_state_with_function(&paired_function)
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
                gate_output: self.f32_buffer(&vec![0.0; gate.rows]),
                up_output: self.f32_buffer(&vec![0.0; up.rows]),
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
                    &buffers.gate_output,
                ),
                (&buffers.up_signs, &buffers.up_scales, &buffers.up_output),
            ] {
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.baseline);
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
            Self::complete(command, "two-dispatch baseline")
        }

        fn paired(&self, buffers: &Buffers, rows: usize, cols: usize) -> Probe<()> {
            let command = self.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.paired);
            encoder.set_buffer(0, Some(&buffers.gate_signs), 0);
            encoder.set_buffer(1, Some(&buffers.gate_scales), 0);
            encoder.set_buffer(2, Some(&buffers.up_signs), 0);
            encoder.set_buffer(3, Some(&buffers.up_scales), 0);
            encoder.set_buffer(4, Some(&buffers.input), 0);
            encoder.set_buffer(5, Some(&buffers.gate_output), 0);
            encoder.set_buffer(6, Some(&buffers.up_output), 0);
            set_u32(encoder, 7, rows as u32);
            set_u32(encoder, 8, cols as u32);
            set_u32(encoder, 9, GROUP_SIZE as u32);
            encoder.dispatch_threads(
                MTLSize::new(rows as u64, 1, 1),
                MTLSize::new(rows.min(256) as u64, 1, 1),
            );
            encoder.end_encoding();
            Self::complete(command, "one-dispatch paired candidate")
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

    fn summary(values: &[f64]) -> Probe<serde_json::Value> {
        Ok(json!({
            "sample_count": values.len(),
            "host_wall_us_min": values.iter().copied().fold(f64::INFINITY, f64::min),
            "host_wall_us_p50": percentile(values.to_vec(), 0.50)?,
            "host_wall_us_p95": percentile(values.to_vec(), 0.95)?,
            "host_wall_us_max": values.iter().copied().fold(0.0, f64::max),
            "timing_authority": "host wall around one completed component command buffer; not GPU-only timing and not model/token timing",
        }))
    }

    fn max_error(expected: &[f32], actual: &[f32]) -> f32 {
        expected
            .iter()
            .zip(actual)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max)
    }

    fn atomic_json(path: &std::path::Path, document: &serde_json::Value) -> Probe<()> {
        let parent = path
            .parent()
            .ok_or_else(|| failure("--out has no parent directory"))?;
        fs::create_dir_all(parent)?;
        let temporary = path.with_file_name(format!(
            ".{}.{}.tmp",
            path.file_name()
                .and_then(|v| v.to_str())
                .ok_or_else(|| failure("--out name is not UTF-8"))?,
            process_id()
        ));
        fs::write(
            &temporary,
            format!("{}\n", serde_json::to_string_pretty(document)?),
        )?;
        fs::rename(temporary, path)?;
        Ok(())
    }

    fn process_id() -> u32 {
        std::process::id()
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
                "admitted gate/up direct projections have mismatched geometry",
            ));
        }
        let input = deterministic_input(gate.cols);
        let cpu_gate = cpu_matvec(&gate, &input);
        let cpu_up = cpu_matvec(&up, &input);
        let runner = Runner::new()?;
        let buffers = runner.buffers(&gate, &up, &input);

        runner.baseline(&buffers, gate.rows, gate.cols)?;
        let baseline_gate = read_f32(&buffers.gate_output, gate.rows)?;
        let baseline_up = read_f32(&buffers.up_output, up.rows)?;
        runner.paired(&buffers, gate.rows, gate.cols)?;
        let paired_gate = read_f32(&buffers.gate_output, gate.rows)?;
        let paired_up = read_f32(&buffers.up_output, up.rows)?;
        let baseline_error =
            max_error(&cpu_gate, &baseline_gate).max(max_error(&cpu_up, &baseline_up));
        let paired_error = max_error(&cpu_gate, &paired_gate).max(max_error(&cpu_up, &paired_up));
        if baseline_error > TOLERANCE || paired_error > TOLERANCE {
            return Err(failure(format!(
                "direct-packed CPU parity failed: baseline={baseline_error}, paired={paired_error}, tolerance={TOLERANCE}"
            )));
        }

        for _ in 0..args.warmups {
            runner.baseline(&buffers, gate.rows, gate.cols)?;
            runner.paired(&buffers, gate.rows, gate.cols)?;
        }
        let mut baseline_us = Vec::with_capacity(args.trials);
        let mut paired_us = Vec::with_capacity(args.trials);
        for _ in 0..args.trials {
            let started = Instant::now();
            runner.baseline(&buffers, gate.rows, gate.cols)?;
            baseline_us.push(started.elapsed().as_secs_f64() * 1_000_000.0);
            let started = Instant::now();
            runner.paired(&buffers, gate.rows, gate.cols)?;
            paired_us.push(started.elapsed().as_secs_f64() * 1_000_000.0);
        }
        let baseline_summary = summary(&baseline_us)?;
        let paired_summary = summary(&paired_us)?;
        let baseline_p50 = baseline_summary["host_wall_us_p50"]
            .as_f64()
            .ok_or_else(|| failure("baseline p50 missing"))?;
        let paired_p50 = paired_summary["host_wall_us_p50"]
            .as_f64()
            .ok_or_else(|| failure("paired p50 missing"))?;
        let document = json!({
            "schema": SCHEMA,
            "status": STATUS,
            "binding": {
                "manifest_path": args.manifest,
                "manifest_seal_sha256": artifact.manifest_seal_sha256,
                "source_audit_seal_sha256": artifact.source_audit_seal_sha256,
                "source_revision": artifact.source_revision,
                "admitted_tensor_count": artifact.tensors.len(),
            },
            "candidate": {
                "id": "qwen30-direct-packed-gate-up-pair-command-topology",
                "shader": "crates/hawking-core/shaders/qwen_direct_packed_gate_up_pair.metal",
                "layer": args.layer,
                "expert": args.expert,
                "baseline_command_topology": {"command_buffers": 1, "compute_dispatches": 2, "projection_pairs": 1},
                "candidate_command_topology": {"command_buffers": 1, "compute_dispatches": 1, "projection_pairs": 1},
                "direct_packed_layout": "HQ30G1B1, group_size=128, FP16 scales plus sign bits",
            },
            "tensors": {
                "gate": {"name": gate.name, "artifact_path": gate.artifact_path, "artifact_sha256": gate.payload_sha256, "shape": [gate.rows, gate.cols], "sign_bytes": gate.signs.len(), "scale_bytes": gate.scales.len()},
                "up": {"name": up.name, "artifact_path": up.artifact_path, "artifact_sha256": up.payload_sha256, "shape": [up.rows, up.cols], "sign_bytes": up.signs.len(), "scale_bytes": up.scales.len()},
            },
            "parity": {
                "cpu_oracle": "Rust f32 matvec over the exact admitted direct binary sign bits and FP16 scales",
                "input": "deterministic f32 vector, no source BF16 input or weight fallback",
                "tolerance": TOLERANCE,
                "baseline_max_abs_error": baseline_error,
                "paired_max_abs_error": paired_error,
                "baseline_within_tolerance": true,
                "paired_within_tolerance": true,
            },
            "timing": {
                "warmups": args.warmups,
                "trials": args.trials,
                "baseline_two_dispatch": baseline_summary,
                "candidate_one_dispatch": paired_summary,
                "p50_component_host_wall_delta_us": paired_p50 - baseline_p50,
                "p50_component_host_wall_speedup_ratio": baseline_p50 / paired_p50,
                "not_a_model_or_token_rate": true,
            },
            "claim_boundary": {
                "opens_only_admitted_direct_binary_qwen30_payloads": true,
                "raw_bf16_and_mps_full_model_not_opened": true,
                "cpu_oracle_is_only_same_packed_representation_parity": true,
                "no_qwen_layer_token_generation_hcli_capability_or_tps": true,
                "integration_requires_separate_runtime_parity_and_complete_token_reprofile": true,
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

//! One-shot, non-serving CPU/device parity gate for Qwen30's HQ30GR2 L0/E0
//! sparse gate/up candidate.
//!
//! The executable has exactly two modes. `cpu-oracle` re-admits the isolated
//! candidate and emits a scalar exact-format HQ30GR2 reference for one sealed
//! L0 router-input vector; it creates no Metal context. `device-parity` first
//! re-establishes that CPU reference, then creates the separately typed Metal
//! runtime and dispatches only the sparse L0/E0 gate/up/SwiGLU pair. It never
//! executes a complete layer/token, logits, sampler, HCLI, or server path.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("qwen30 HQ30GR2 sparse gate/up parity requires macOS Metal");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::model::qwen30_complete_runtime::{
        Qwen30CompleteRuntimeOptions, Qwen30GateUpSwiGluKernel, Qwen30PackedMatvecKernel,
        Qwen30QualityRepackNativeDiagnosticRuntime,
    };
    use hawking_core::model::qwen30_quality_repack_diagnostic::{
        Qwen30QualityRepackDiagnosticCatalog, QWEN30_QUALITY_REPACK_SPARSE_GATE_UP_KERNEL,
    };
    use hawking_core::model::qwen_complete_binary::{
        admit_qwen30_quality_repack_artifact, Qwen30QualityRepackAdmission,
    };
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::env;
    use std::fs::{self, File, OpenOptions};
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};
    use std::process;

    const RESULT_SCHEMA: &str = "hawking.ascension.qwen30_hq30gr2_sparse_gate_up_device_parity.v1";
    const CPU_STATUS: &str = "EARNED_HQ30GR2_CPU_FORMAT_ORACLE_NOT_DEVICE_OR_RUNTIME";
    const DEVICE_PASS_STATUS: &str =
        "EARNED_HQ30GR2_SPARSE_GATE_UP_CPU_DEVICE_PARITY_NOT_LAYER_OR_RUNTIME";
    const DEVICE_REFUSAL_STATUS: &str = "REFUSED_HQ30GR2_SPARSE_GATE_UP_CPU_DEVICE_PARITY_DIVERGED";
    const INPUT_VALUES: usize = 2048;
    const OUTPUT_VALUES: usize = 768;
    const MAX_ABS_ERROR: f64 = 1.0e-2;
    const MAX_REL_ERROR: f64 = 2.0e-3;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Mode {
        CpuOracle,
        DeviceParity,
    }

    impl Mode {
        fn parse(value: &str) -> Result<Self, String> {
            match value {
                "cpu-oracle" => Ok(Self::CpuOracle),
                "device-parity" => Ok(Self::DeviceParity),
                _ => Err(format!(
                    "unsupported --mode {value:?}; expected cpu-oracle or device-parity; {}",
                    usage()
                )),
            }
        }

        fn name(self) -> &'static str {
            match self {
                Self::CpuOracle => "cpu-oracle",
                Self::DeviceParity => "device-parity",
            }
        }
    }

    struct Arguments {
        mode: Mode,
        manifest: PathBuf,
        admission: Qwen30QualityRepackAdmission,
        input_f32le: PathBuf,
        expected_input_sha256: String,
        cpu_activation_f64le: Option<PathBuf>,
        expected_cpu_activation_sha256: Option<String>,
        output_dir: PathBuf,
        max_seq_len: usize,
    }

    fn usage() -> &'static str {
        "usage: ascension_qwen30_hq30gr2_sparse_gate_up_device_parity \\
            --mode cpu-oracle|device-parity \\
            --manifest ABSOLUTE_PATH \\
            --expected-manifest-seal-sha256 SHA256 \\
            --expected-source-audit-seal-sha256 SHA256 \\
            --expected-source-revision REVISION \\
            --expected-revalidation-path ABSOLUTE_PATH \\
            --expected-revalidation-seal-sha256 SHA256 \\
            --expected-selection-path ABSOLUTE_PATH \\
            --expected-selection-seal-sha256 SHA256 \\
            --expected-source-snapshot-path ABSOLUTE_PATH \\
            --expected-source-snapshot-seal-sha256 SHA256 \\
            --expected-terminal-path ABSOLUTE_PATH \\
            --expected-terminal-seal-sha256 SHA256 \\
            --input-f32le ABSOLUTE_PATH --expected-input-sha256 SHA256 \\
            --output-dir ABSOLUTE_PATH [--max-seq-len N] \\
            [--cpu-activation-f64le ABSOLUTE_PATH --expected-cpu-activation-sha256 SHA256]"
    }

    fn required<T>(value: Option<T>, flag: &str) -> Result<T, String> {
        value.ok_or_else(|| format!("missing {flag}; {}", usage()))
    }

    fn absolute(path: PathBuf, flag: &str) -> Result<PathBuf, String> {
        if !path.is_absolute() {
            return Err(format!("{flag} must be an absolute path; {}", usage()));
        }
        Ok(path)
    }

    fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
        value
            .parse::<usize>()
            .map_err(|_| format!("{flag} must be an unsigned decimal integer; {}", usage()))
    }

    fn lower_sha256(value: &str, flag: &str) -> Result<String, String> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(format!("{flag} must be lowercase SHA-256; {}", usage()));
        }
        Ok(value.to_owned())
    }

    fn parse_arguments() -> Result<Arguments, String> {
        let mut mode = None;
        let mut manifest = None;
        let mut expected_manifest_seal_sha256 = None;
        let mut expected_source_audit_seal_sha256 = None;
        let mut expected_source_revision = None;
        let mut expected_revalidation_path = None;
        let mut expected_revalidation_seal_sha256 = None;
        let mut expected_selection_path = None;
        let mut expected_selection_seal_sha256 = None;
        let mut expected_source_snapshot_path = None;
        let mut expected_source_snapshot_seal_sha256 = None;
        let mut expected_terminal_path = None;
        let mut expected_terminal_seal_sha256 = None;
        let mut input_f32le = None;
        let mut expected_input_sha256 = None;
        let mut cpu_activation_f64le = None;
        let mut expected_cpu_activation_sha256 = None;
        let mut output_dir = None;
        let mut max_seq_len = 512usize;
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {flag:?}; {}", usage()))?;
            macro_rules! unique {
                ($slot:ident, $value:expr) => {
                    if $slot.replace($value).is_some() {
                        return Err(format!("{flag} was supplied more than once; {}", usage()));
                    }
                };
            }
            match flag.as_str() {
                "--mode" => unique!(mode, Mode::parse(&value)?),
                "--manifest" => unique!(manifest, PathBuf::from(value)),
                "--expected-manifest-seal-sha256" => {
                    unique!(expected_manifest_seal_sha256, lower_sha256(&value, &flag)?)
                }
                "--expected-source-audit-seal-sha256" => {
                    unique!(
                        expected_source_audit_seal_sha256,
                        lower_sha256(&value, &flag)?
                    )
                }
                "--expected-source-revision" => unique!(expected_source_revision, value),
                "--expected-revalidation-path" => {
                    unique!(expected_revalidation_path, PathBuf::from(value))
                }
                "--expected-revalidation-seal-sha256" => {
                    unique!(
                        expected_revalidation_seal_sha256,
                        lower_sha256(&value, &flag)?
                    )
                }
                "--expected-selection-path" => {
                    unique!(expected_selection_path, PathBuf::from(value))
                }
                "--expected-selection-seal-sha256" => {
                    unique!(expected_selection_seal_sha256, lower_sha256(&value, &flag)?)
                }
                "--expected-source-snapshot-path" => {
                    unique!(expected_source_snapshot_path, PathBuf::from(value))
                }
                "--expected-source-snapshot-seal-sha256" => {
                    unique!(
                        expected_source_snapshot_seal_sha256,
                        lower_sha256(&value, &flag)?
                    )
                }
                "--expected-terminal-path" => unique!(expected_terminal_path, PathBuf::from(value)),
                "--expected-terminal-seal-sha256" => {
                    unique!(expected_terminal_seal_sha256, lower_sha256(&value, &flag)?)
                }
                "--input-f32le" => unique!(input_f32le, PathBuf::from(value)),
                "--expected-input-sha256" => {
                    unique!(expected_input_sha256, lower_sha256(&value, &flag)?)
                }
                "--cpu-activation-f64le" => unique!(cpu_activation_f64le, PathBuf::from(value)),
                "--expected-cpu-activation-sha256" => {
                    unique!(expected_cpu_activation_sha256, lower_sha256(&value, &flag)?)
                }
                "--output-dir" => unique!(output_dir, PathBuf::from(value)),
                "--max-seq-len" => max_seq_len = parse_usize(&value, "--max-seq-len")?,
                _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
            }
        }
        let mode = required(mode, "--mode")?;
        if max_seq_len == 0 {
            return Err("--max-seq-len must be positive".into());
        }
        let cpu_paths_present =
            cpu_activation_f64le.is_some() || expected_cpu_activation_sha256.is_some();
        if mode == Mode::CpuOracle && cpu_paths_present {
            return Err("cpu-oracle accepts no --cpu-activation-f64le binding".into());
        }
        if mode == Mode::DeviceParity
            && (!cpu_activation_f64le.is_some() || !expected_cpu_activation_sha256.is_some())
        {
            return Err("device-parity requires both --cpu-activation-f64le and --expected-cpu-activation-sha256".into());
        }
        let admission = Qwen30QualityRepackAdmission {
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
            expected_revalidation_path: absolute(
                required(expected_revalidation_path, "--expected-revalidation-path")?,
                "--expected-revalidation-path",
            )?,
            expected_revalidation_seal_sha256: required(
                expected_revalidation_seal_sha256,
                "--expected-revalidation-seal-sha256",
            )?,
            expected_selection_path: absolute(
                required(expected_selection_path, "--expected-selection-path")?,
                "--expected-selection-path",
            )?,
            expected_selection_seal_sha256: required(
                expected_selection_seal_sha256,
                "--expected-selection-seal-sha256",
            )?,
            expected_source_snapshot_path: absolute(
                required(
                    expected_source_snapshot_path,
                    "--expected-source-snapshot-path",
                )?,
                "--expected-source-snapshot-path",
            )?,
            expected_source_snapshot_seal_sha256: required(
                expected_source_snapshot_seal_sha256,
                "--expected-source-snapshot-seal-sha256",
            )?,
            expected_terminal_path: absolute(
                required(expected_terminal_path, "--expected-terminal-path")?,
                "--expected-terminal-path",
            )?,
            expected_terminal_seal_sha256: required(
                expected_terminal_seal_sha256,
                "--expected-terminal-seal-sha256",
            )?,
        };
        Ok(Arguments {
            mode,
            manifest: absolute(required(manifest, "--manifest")?, "--manifest")?,
            admission,
            input_f32le: absolute(required(input_f32le, "--input-f32le")?, "--input-f32le")?,
            expected_input_sha256: required(expected_input_sha256, "--expected-input-sha256")?,
            cpu_activation_f64le: cpu_activation_f64le
                .map(|path| absolute(path, "--cpu-activation-f64le"))
                .transpose()?,
            expected_cpu_activation_sha256,
            output_dir: absolute(required(output_dir, "--output-dir")?, "--output-dir")?,
            max_seq_len,
        })
    }

    fn sha256_bytes(bytes: &[u8]) -> String {
        let mut digest = Sha256::new();
        digest.update(bytes);
        format!("{:x}", digest.finalize())
    }

    fn sha256_file(path: &Path) -> Result<String, String> {
        let mut file =
            File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
        let mut digest = Sha256::new();
        let mut chunk = [0u8; 1024 * 1024];
        loop {
            let read = file
                .read(&mut chunk)
                .map_err(|error| format!("cannot hash {}: {error}", path.display()))?;
            if read == 0 {
                break;
            }
            digest.update(&chunk[..read]);
        }
        Ok(format!("{:x}", digest.finalize()))
    }

    fn current_executable_sha256() -> Result<String, String> {
        let path = env::current_exe()
            .map_err(|error| format!("cannot resolve parity executable: {error}"))?;
        sha256_file(&path)
    }

    fn read_f32le(path: &Path, expected_sha256: &str) -> Result<Vec<f32>, String> {
        let bytes = fs::read(path)
            .map_err(|error| format!("cannot read F32LE input {}: {error}", path.display()))?;
        let actual = sha256_bytes(&bytes);
        if actual != expected_sha256 {
            return Err(format!(
                "F32LE input SHA-256 differs: observed={actual} expected={expected_sha256}"
            ));
        }
        if bytes.len() != INPUT_VALUES * std::mem::size_of::<f32>() {
            return Err(format!(
                "F32LE input has {} bytes, expected {} for {INPUT_VALUES} values",
                bytes.len(),
                INPUT_VALUES * std::mem::size_of::<f32>()
            ));
        }
        let values = bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_bits(u32::from_le_bytes(chunk.try_into().expect("exact f32"))))
            .collect::<Vec<_>>();
        if values.iter().any(|value| !value.is_finite()) {
            return Err("F32LE input contains a non-finite value".into());
        }
        Ok(values)
    }

    fn read_f64le(path: &Path, expected_sha256: &str) -> Result<Vec<f64>, String> {
        let bytes = fs::read(path)
            .map_err(|error| format!("cannot read F64LE CPU oracle {}: {error}", path.display()))?;
        let actual = sha256_bytes(&bytes);
        if actual != expected_sha256 {
            return Err(format!(
                "F64LE CPU oracle SHA-256 differs: observed={actual} expected={expected_sha256}"
            ));
        }
        if bytes.len() != OUTPUT_VALUES * std::mem::size_of::<f64>() {
            return Err(format!(
                "F64LE CPU oracle has {} bytes, expected {} for {OUTPUT_VALUES} values",
                bytes.len(),
                OUTPUT_VALUES * std::mem::size_of::<f64>()
            ));
        }
        let values = bytes
            .chunks_exact(std::mem::size_of::<f64>())
            .map(|chunk| f64::from_bits(u64::from_le_bytes(chunk.try_into().expect("exact f64"))))
            .collect::<Vec<_>>();
        if values.iter().any(|value| !value.is_finite()) {
            return Err("F64LE CPU oracle contains a non-finite value".into());
        }
        Ok(values)
    }

    fn f64le(values: &[f64]) -> Result<Vec<u8>, String> {
        if values.len() != OUTPUT_VALUES || values.iter().any(|value| !value.is_finite()) {
            return Err("CPU oracle activation is not exactly 768 finite F64 values".into());
        }
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f64>());
        for value in values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        Ok(bytes)
    }

    fn f32le(values: &[f32]) -> Result<Vec<u8>, String> {
        if values.len() != OUTPUT_VALUES || values.iter().any(|value| !value.is_finite()) {
            return Err("device activation is not exactly 768 finite F32 values".into());
        }
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
        for value in values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        Ok(bytes)
    }

    fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| format!("refusing to replace {}: {error}", path.display()))?;
        file.write_all(bytes)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
        file.sync_all()
            .map_err(|error| format!("cannot sync {}: {error}", path.display()))
    }

    fn write_new_json(path: &Path, value: &Value) -> Result<(), String> {
        let mut bytes = serde_json::to_vec_pretty(value)
            .map_err(|error| format!("cannot serialize result JSON: {error}"))?;
        bytes.push(b'\n');
        write_new(path, &bytes)
    }

    fn require_output_dir(path: &Path) -> Result<(), String> {
        if !path.is_dir() {
            return Err(format!(
                "--output-dir {} must be an existing isolated run directory",
                path.display()
            ));
        }
        Ok(())
    }

    fn binding(arguments: &Arguments, executable_sha256: &str, input_sha256: &str) -> Value {
        json!({
            "candidate_manifest_path": arguments.manifest,
            "candidate_manifest_seal_sha256": arguments.admission.expected_manifest_seal_sha256,
            "source_audit_seal_sha256": arguments.admission.expected_source_audit_seal_sha256,
            "source_revision": arguments.admission.expected_source_revision,
            "revalidation": {
                "path": arguments.admission.expected_revalidation_path,
                "seal_sha256": arguments.admission.expected_revalidation_seal_sha256,
            },
            "selection": {
                "path": arguments.admission.expected_selection_path,
                "seal_sha256": arguments.admission.expected_selection_seal_sha256,
            },
            "source_snapshot": {
                "path": arguments.admission.expected_source_snapshot_path,
                "seal_sha256": arguments.admission.expected_source_snapshot_seal_sha256,
            },
            "terminal": {
                "path": arguments.admission.expected_terminal_path,
                "seal_sha256": arguments.admission.expected_terminal_seal_sha256,
            },
            "input_f32le": {
                "path": arguments.input_f32le,
                "sha256": input_sha256,
                "values": INPUT_VALUES,
            },
            "runtime_executable_sha256": executable_sha256,
            "fixed_topology": {
                "all_unchanged_projections": "scalar_direct_packed_control",
                "selected_sparse_pair": "L0/E0 gate+up HQ30GR2",
                "kernel": QWEN30_QUALITY_REPACK_SPARSE_GATE_UP_KERNEL,
                "no_direct_fallback_for_sparse_pair": true,
                "no_bf16_or_dense_weight_path": true,
            },
        })
    }

    fn cpu_catalog_and_oracle(
        arguments: &Arguments,
        input: &[f32],
    ) -> Result<(Qwen30QualityRepackDiagnosticCatalog, Vec<f64>), String> {
        let candidate =
            admit_qwen30_quality_repack_artifact(&arguments.manifest, &arguments.admission)
                .map_err(|error| format!("HQ30GR2 candidate admission refused: {error}"))?;
        let catalog = Qwen30QualityRepackDiagnosticCatalog::from_admitted(candidate)
            .map_err(|error| format!("HQ30GR2 typed catalog refused: {error}"))?;
        let oracle = catalog
            .sparse_gate_up_cpu_oracle_f64(
                &input
                    .iter()
                    .map(|value| f64::from(*value))
                    .collect::<Vec<_>>(),
            )
            .map_err(|error| format!("HQ30GR2 CPU oracle refused: {error}"))?;
        Ok((catalog, oracle))
    }

    fn comparison(device: &[f32], cpu: &[f64]) -> Result<Value, String> {
        if device.len() != OUTPUT_VALUES || cpu.len() != OUTPUT_VALUES {
            return Err("CPU/device activation lengths differ from the typed 768-row ABI".into());
        }
        let mut max_abs = 0.0f64;
        let mut max_rel = 0.0f64;
        let mut max_abs_row = 0usize;
        let mut max_rel_row = 0usize;
        for (row, (&device_value, &cpu_value)) in device.iter().zip(cpu).enumerate() {
            if !device_value.is_finite() || !cpu_value.is_finite() {
                return Err(format!("CPU/device parity contains non-finite row {row}"));
            }
            let absolute = (f64::from(device_value) - cpu_value).abs();
            let relative = absolute / cpu_value.abs().max(1.0);
            if absolute > max_abs {
                max_abs = absolute;
                max_abs_row = row;
            }
            if relative > max_rel {
                max_rel = relative;
                max_rel_row = row;
            }
        }
        Ok(json!({
            "cpu_reference": "immutable_admission_snapshot_hq30gr2_scalar_f64",
            "device_reference": "typed_direct_base_plus_sorted_sparse_residual_metal_f32",
            "values_compared": OUTPUT_VALUES,
            "max_abs_error": max_abs,
            "max_abs_error_row": max_abs_row,
            "max_rel_error": max_rel,
            "max_rel_error_row": max_rel_row,
            "max_abs_error_allowed": MAX_ABS_ERROR,
            "max_rel_error_allowed": MAX_REL_ERROR,
            "passes": max_abs <= MAX_ABS_ERROR && max_rel <= MAX_REL_ERROR,
            "cpu_precision_vs_device_precision": "f64_reference_vs_f32_device; tolerance is explicit",
        }))
    }

    fn fail(detail: impl AsRef<str>) -> ! {
        eprintln!(
            "qwen30 HQ30GR2 sparse gate/up parity refused: {}",
            detail.as_ref()
        );
        process::exit(2);
    }

    fn run() -> Result<Value, String> {
        let arguments = parse_arguments()?;
        require_output_dir(&arguments.output_dir)?;
        let executable_sha256 = current_executable_sha256()?;
        let input = read_f32le(&arguments.input_f32le, &arguments.expected_input_sha256)?;
        let input_sha256 = sha256_file(&arguments.input_f32le)?;
        let (catalog, cpu_oracle) = cpu_catalog_and_oracle(&arguments, &input)?;
        let cpu_bytes = f64le(&cpu_oracle)?;
        let cpu_sha256 = sha256_bytes(&cpu_bytes);
        let dispatch = catalog
            .sparse_gate_up_dispatch()
            .map_err(|error| format!("HQ30GR2 typed dispatch refused: {error}"))?;
        if dispatch.rows != OUTPUT_VALUES || dispatch.cols != INPUT_VALUES {
            return Err("typed HQ30GR2 dispatch has unexpected gate/up geometry".into());
        }
        let common = json!({
            "schema": RESULT_SCHEMA,
            "mode": arguments.mode.name(),
            "binding": binding(&arguments, &executable_sha256, &input_sha256),
            "typed_catalog": {
                "verified_payload_count": catalog.verified_payload_count(),
                "direct_tensor_count": catalog.direct_tensor_count(),
                "sparse_residual_tensor_count": catalog.sparse_residual_tensor_count(),
                "sparse_gate_up_dispatch": {
                    "kernel_name": dispatch.kernel_name,
                    "rows": dispatch.rows,
                    "cols": dispatch.cols,
                    "group_size": dispatch.group_size,
                    "gate_residual_count": dispatch.gate_residual_count,
                    "up_residual_count": dispatch.up_residual_count,
                    "exact_non_fma_scalar_order_required": dispatch.exact_non_fma_scalar_order_required,
                    "direct_fallback_for_sparse_residual_forbidden": dispatch.direct_fallback_for_sparse_residual_forbidden,
                },
            },
            "cpu_oracle": {
                "activation_f64le_sha256": cpu_sha256,
                "activation_values": OUTPUT_VALUES,
                "admission_snapshot_only": true,
                "raw_bf16_or_dense_weight_path": false,
            },
            "claim_boundary": {
                "not_a_complete_layer_or_full_token": true,
                "no_logits_sampler_generation_hcli_or_server": true,
                "not_coherence_tps_tg_capability_manager_or_tournament": true,
            },
        });
        match arguments.mode {
            Mode::CpuOracle => {
                let activation_path = arguments.output_dir.join("cpu-activation.f64le");
                write_new(&activation_path, &cpu_bytes)?;
                let mut result = common;
                let object = result
                    .as_object_mut()
                    .ok_or_else(|| "internal result must be an object".to_string())?;
                object.insert("status".into(), Value::String(CPU_STATUS.into()));
                object.insert(
                    "cpu_oracle_output".into(),
                    json!({
                        "path": activation_path,
                        "sha256": cpu_sha256,
                        "values": OUTPUT_VALUES,
                    }),
                );
                let result_path = arguments.output_dir.join("result.json");
                write_new_json(&result_path, &result)?;
                Ok(result)
            }
            Mode::DeviceParity => {
                let cpu_path = arguments
                    .cpu_activation_f64le
                    .as_ref()
                    .expect("device parity requires CPU oracle path");
                let expected_cpu_sha = arguments
                    .expected_cpu_activation_sha256
                    .as_deref()
                    .expect("device parity requires CPU oracle SHA");
                let recorded_cpu = read_f64le(cpu_path, expected_cpu_sha)?;
                let recorded_bytes = f64le(&recorded_cpu)?;
                if recorded_bytes != cpu_bytes || expected_cpu_sha != cpu_sha256 {
                    return Err(
                        "current exact CPU oracle differs from the protected pre-Metal CPU output"
                            .into(),
                    );
                }
                // Drop the CPU-only catalog before the one typed runtime constructs a
                // Metal context. The runtime re-admits the same protected candidate,
                // then uploads only L0/E0 sparse organs plus lazy direct controls.
                drop(catalog);
                let mut runtime = Qwen30QualityRepackNativeDiagnosticRuntime::load(
                    &arguments.manifest,
                    &arguments.admission,
                    Qwen30CompleteRuntimeOptions {
                        max_seq_len: arguments.max_seq_len,
                        trace_dispatch: true,
                        packed_matvec_kernel: Qwen30PackedMatvecKernel::SerialControl,
                        gate_up_swiglu_kernel: Qwen30GateUpSwiGluKernel::ThreeDispatchControl,
                    },
                )
                .map_err(|error| format!("typed HQ30GR2 device runtime refused: {error}"))?;
                if runtime.artifact_manifest_seal()
                    != arguments.admission.expected_manifest_seal_sha256
                {
                    return Err(
                        "typed HQ30GR2 runtime manifest seal differs from protected admission"
                            .into(),
                    );
                }
                if runtime.sparse_gate_up_dispatch() != &dispatch {
                    return Err(
                        "typed HQ30GR2 runtime dispatch differs from pre-Metal CPU contract".into(),
                    );
                }
                let device = runtime
                    .sparse_gate_up_device_pair_for_input_diagnostic(&input)
                    .map_err(|error| format!("typed HQ30GR2 device pair refused: {error}"))?;
                let compare = comparison(&device, &recorded_cpu)?;
                let passes = compare.get("passes").and_then(Value::as_bool) == Some(true);
                let device_bytes = f32le(&device)?;
                let device_path = arguments.output_dir.join("device-activation.f32le");
                write_new(&device_path, &device_bytes)?;
                let profiler = runtime.drain_profiler();
                let mut result = common;
                let object = result
                    .as_object_mut()
                    .ok_or_else(|| "internal result must be an object".to_string())?;
                object.insert(
                    "status".into(),
                    Value::String(
                        if passes {
                            DEVICE_PASS_STATUS
                        } else {
                            DEVICE_REFUSAL_STATUS
                        }
                        .into(),
                    ),
                );
                object.insert(
                    "protected_cpu_oracle".into(),
                    json!({
                        "path": cpu_path,
                        "sha256": expected_cpu_sha,
                        "recomputed_current_sha256": cpu_sha256,
                    }),
                );
                object.insert("device_parity".into(), compare);
                object.insert(
                    "device_output".into(),
                    json!({
                        "path": device_path,
                        "sha256": sha256_bytes(&device_bytes),
                        "values": OUTPUT_VALUES,
                    }),
                );
                object.insert(
                    "device_execution".into(),
                    json!({
                        "metal_context_created": true,
                        "kernel": QWEN30_QUALITY_REPACK_SPARSE_GATE_UP_KERNEL,
                        "only_selected_l0_e0_gate_up_swiglu_pair_executed": true,
                        "all_layers_executed": false,
                        "full_token_executed": false,
                        "dispatch_samples": profiler.dispatch_samples.len(),
                        "command_buffers_committed": profiler.command_buffers_committed,
                    }),
                );
                let result_path = arguments.output_dir.join("result.json");
                write_new_json(&result_path, &result)?;
                Ok(result)
            }
        }
    }

    pub fn main() {
        match run() {
            Ok(result) => println!(
                "{}",
                serde_json::to_string(&result).expect("parity result must serialize")
            ),
            Err(error) => fail(error),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn only_two_non_serving_modes_are_accepted() {
            assert_eq!(Mode::parse("cpu-oracle").unwrap(), Mode::CpuOracle);
            assert_eq!(Mode::parse("device-parity").unwrap(), Mode::DeviceParity);
            assert!(Mode::parse("generate").is_err());
        }

        #[test]
        fn comparison_requires_both_explicit_tolerances() {
            let cpu = vec![1.0f64; OUTPUT_VALUES];
            let device = vec![1.0f32; OUTPUT_VALUES];
            assert_eq!(
                comparison(&device, &cpu).unwrap()["passes"],
                Value::Bool(true)
            );
            let device = vec![2.0f32; OUTPUT_VALUES];
            assert_eq!(
                comparison(&device, &cpu).unwrap()["passes"],
                Value::Bool(false)
            );
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos::main();
}

//! Bounded, fail-closed native Metal entrypoint for the admitted Qwen30 pack.
//!
//! This executable has deliberately separate structural and execution modes:
//! `preflight` establishes only the sealed artifact/config/tokenizer/catalog
//! binding; `forward-token` executes one real complete native token; and
//! `generate-greedy` executes a bounded prompt prefill plus feedback loop.
//! None computes or reports TPS, HCLI, capability, TG, or tournament results.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("qwen30 complete native runtime requires macOS Metal");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::model::qwen30_complete_runtime::{
        preflight_complete_runtime, Qwen30CompleteNativeRuntime, Qwen30CompleteRuntimeOptions,
        Qwen30GateUpSwiGluKernel, Qwen30NativeGreedyStep, Qwen30NativeProfilerSnapshot,
        Qwen30PackedMatvecKernel,
    };
    use hawking_core::model::qwen_complete_binary::{
        CompleteBinaryAdmission, Qwen30ActivationWeightedSvdAdmission, Qwen30UniformQ4Admission,
        Qwen30UniformQnAdmission, QwenCompleteBinaryModel, UniformQnBits,
        QWEN30_ACTIVATION_WEIGHTED_SVD_SCHEMA, QWEN30_COMPLETE_BINARY_SCHEMA,
        QWEN30_UNIFORM_Q4_SCHEMA,
    };
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    use std::env;
    use std::fs::File;
    use std::io::Read;
    use std::path::PathBuf;
    use std::process;

    const RESULT_SCHEMA: &str = "hawking.ascension.qwen30_complete_native_runtime_result.v1";

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Mode {
        Preflight,
        ForwardToken,
        GenerateGreedy,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum PromptTemplate {
        /// Raw text is retained only for isolated tokenizer/runtime diagnosis;
        /// it is never the template-bound manager/HCLI path.
        RawTextDiagnostic,
        /// The exact source Jinja's validated one-user/no-tools branch.
        SourceUserChat,
    }

    impl PromptTemplate {
        fn parse(value: &str) -> Result<Self, String> {
            match value {
                "raw-text-diagnostic" => Ok(Self::RawTextDiagnostic),
                "source-user-chat" => Ok(Self::SourceUserChat),
                _ => Err(format!(
                    "unsupported --prompt-template {value:?}; expected raw-text-diagnostic or source-user-chat; {}",
                    usage()
                )),
            }
        }

        fn receipt_name(self) -> &'static str {
            match self {
                Self::RawTextDiagnostic => "raw_text_diagnostic",
                Self::SourceUserChat => "source_user_chat_template",
            }
        }
    }

    impl Mode {
        fn parse(value: &str) -> Result<Self, String> {
            match value {
                "preflight" => Ok(Self::Preflight),
                "forward-token" => Ok(Self::ForwardToken),
                "generate-greedy" => Ok(Self::GenerateGreedy),
                _ => Err(format!("unsupported --mode {value:?}; {}", usage())),
            }
        }

        fn name(self) -> &'static str {
            match self {
                Self::Preflight => "preflight",
                Self::ForwardToken => "forward-token",
                Self::GenerateGreedy => "generate-greedy",
            }
        }
    }

    fn parse_packed_matvec_kernel(value: &str) -> Result<Qwen30PackedMatvecKernel, String> {
        match value {
            "control" | "col32-candidate" => Ok(Qwen30PackedMatvecKernel::ScalarControl),
            "serial-control" => Ok(Qwen30PackedMatvecKernel::SerialControl),
            "simdgroup-candidate" => Ok(Qwen30PackedMatvecKernel::SimdgroupCandidate),
            "rowblock-2" => Ok(Qwen30PackedMatvecKernel::RowBlock2),
            "rowblock-4" => Ok(Qwen30PackedMatvecKernel::RowBlock4),
            "rowblock-8" => Ok(Qwen30PackedMatvecKernel::RowBlock8),
            _ => Err(format!(
                "unsupported --packed-matvec-kernel {value:?}; expected control, serial-control, simdgroup-candidate, rowblock-2, rowblock-4, or rowblock-8; {}",
                usage()
            )),
        }
    }

    fn parse_gate_up_swiglu_kernel(value: &str) -> Result<Qwen30GateUpSwiGluKernel, String> {
        match value {
            "control" => Ok(Qwen30GateUpSwiGluKernel::ThreeDispatchControl),
            "fused-candidate" => Ok(Qwen30GateUpSwiGluKernel::FusedCandidate),
            "fused-candidate-device-parity" => Ok(
                Qwen30GateUpSwiGluKernel::FusedCandidateWithDeviceControlParity,
            ),
            "paired-scalar-order-candidate-device-parity" => Ok(
                Qwen30GateUpSwiGluKernel::PairedScalarOrderCandidateWithDeviceControlParity,
            ),
            "paired-scalar-order-production-no-parity" => Ok(
                Qwen30GateUpSwiGluKernel::PairedScalarOrderProductionNoParity,
            ),
            _ => Err(format!(
                "unsupported --gate-up-swiglu-kernel {value:?}; expected control, fused-candidate, fused-candidate-device-parity, paired-scalar-order-candidate-device-parity, or paired-scalar-order-production-no-parity; {}",
                usage()
            )),
        }
    }

    struct Arguments {
        manifest: PathBuf,
        expected_manifest_seal_sha256: String,
        expected_source_audit_seal_sha256: String,
        expected_source_revision: String,
        /// Required only for the mixed HQ30G1B1 + HGRAVS01 candidate schema.
        activation_weighted: Option<ActivationWeightedBindings>,
        uniform_q4: Option<UniformQ4Bindings>,
        uniform_qn: Option<UniformQnBindings>,
        mode: Mode,
        token_id: Option<u32>,
        prompt: Option<String>,
        max_new_tokens: usize,
        max_seq_len: usize,
        trace_dispatch: bool,
        packed_matvec_kernel: Qwen30PackedMatvecKernel,
        gate_up_swiglu_kernel: Qwen30GateUpSwiGluKernel,
        prompt_template: PromptTemplate,
    }

    #[derive(Clone, Debug)]
    struct UniformQ4Bindings {
        expected_revalidation_path: PathBuf,
        expected_revalidation_seal_sha256: String,
        expected_terminal_path: PathBuf,
        expected_terminal_seal_sha256: String,
    }

    #[derive(Clone, Debug)]
    struct UniformQnBindings {
        bits: u32,
        expected_revalidation_path: PathBuf,
        expected_revalidation_seal_sha256: String,
        expected_terminal_path: PathBuf,
        expected_terminal_seal_sha256: String,
    }

    #[derive(Clone, Debug)]
    struct ActivationWeightedBindings {
        expected_revalidation_path: PathBuf,
        expected_revalidation_seal_sha256: String,
        expected_selection_path: PathBuf,
        expected_selection_seal_sha256: String,
        expected_source_snapshot_path: PathBuf,
        expected_source_snapshot_seal_sha256: String,
        expected_terminal_path: PathBuf,
        expected_terminal_seal_sha256: String,
        expected_activation_capture_sha256: String,
    }

    fn usage() -> &'static str {
        "usage: ascension_qwen30_complete_native_runtime \\
            --manifest ABSOLUTE_PATH \\
            --expected-manifest-seal-sha256 SHA256 \\
            --expected-source-audit-seal-sha256 SHA256 \\
            --expected-source-revision REVISION \\
            --mode preflight|forward-token|generate-greedy \\
            [--token-id ID] [--prompt TEXT] [--max-new-tokens N] \\
            [--max-seq-len N] [--trace-dispatch] [--prompt-template raw-text-diagnostic|source-user-chat] \
            [--packed-matvec-kernel control|serial-control|simdgroup-candidate|rowblock-2|rowblock-4|rowblock-8] \
            [--gate-up-swiglu-kernel control|fused-candidate|fused-candidate-device-parity|paired-scalar-order-candidate-device-parity|paired-scalar-order-production-no-parity] \
            [--expected-revalidation-path PATH --expected-revalidation-seal-sha256 SHA256 \
             --expected-selection-path PATH --expected-selection-seal-sha256 SHA256 \
             --expected-source-snapshot-path PATH --expected-source-snapshot-seal-sha256 SHA256 \
             --expected-terminal-path PATH --expected-terminal-seal-sha256 SHA256 \
             --expected-activation-capture-sha256 SHA256]"
    }

    fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
        value
            .parse::<usize>()
            .map_err(|_| format!("{flag} must be an unsigned decimal integer; {}", usage()))
    }

    fn parse_u32(value: &str, flag: &str) -> Result<u32, String> {
        value
            .parse::<u32>()
            .map_err(|_| format!("{flag} must be an unsigned decimal integer; {}", usage()))
    }

    fn required<T>(value: Option<T>, flag: &str) -> Result<T, String> {
        value.ok_or_else(|| format!("missing {flag}; {}", usage()))
    }

    fn parse_arguments() -> Result<Arguments, String> {
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
        let mut expected_activation_capture_sha256 = None;
        let mut mode = None;
        let mut token_id = None;
        let mut prompt = None;
        let mut max_new_tokens = 2usize;
        let mut max_seq_len = 256usize;
        let mut trace_dispatch = false;
        let mut packed_matvec_kernel = Qwen30PackedMatvecKernel::RowBlock4;
        let mut gate_up_swiglu_kernel = Qwen30GateUpSwiGluKernel::ThreeDispatchControl;
        let mut prompt_template = PromptTemplate::RawTextDiagnostic;
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
            if flag == "--trace-dispatch" {
                if trace_dispatch {
                    return Err(format!(
                        "--trace-dispatch was supplied more than once; {}",
                        usage()
                    ));
                }
                trace_dispatch = true;
                continue;
            }
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {flag:?}; {}", usage()))?;
            match flag.as_str() {
                "--manifest" => {
                    if manifest.replace(PathBuf::from(value)).is_some() {
                        return Err(format!(
                            "--manifest was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-manifest-seal-sha256" => {
                    if expected_manifest_seal_sha256.replace(value).is_some() {
                        return Err(format!(
                            "--expected-manifest-seal-sha256 was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-source-audit-seal-sha256" => {
                    if expected_source_audit_seal_sha256.replace(value).is_some() {
                        return Err(format!(
                            "--expected-source-audit-seal-sha256 was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-source-revision" => {
                    if expected_source_revision.replace(value).is_some() {
                        return Err(format!(
                            "--expected-source-revision was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-revalidation-path" => {
                    if expected_revalidation_path
                        .replace(PathBuf::from(value))
                        .is_some()
                    {
                        return Err(format!(
                            "--expected-revalidation-path was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-revalidation-seal-sha256" => {
                    if expected_revalidation_seal_sha256.replace(value).is_some() {
                        return Err(format!(
                            "--expected-revalidation-seal-sha256 was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-selection-path" => {
                    if expected_selection_path
                        .replace(PathBuf::from(value))
                        .is_some()
                    {
                        return Err(format!(
                            "--expected-selection-path was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-selection-seal-sha256" => {
                    if expected_selection_seal_sha256.replace(value).is_some() {
                        return Err(format!(
                            "--expected-selection-seal-sha256 was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-source-snapshot-path" => {
                    if expected_source_snapshot_path
                        .replace(PathBuf::from(value))
                        .is_some()
                    {
                        return Err(format!(
                            "--expected-source-snapshot-path was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-source-snapshot-seal-sha256" => {
                    if expected_source_snapshot_seal_sha256.replace(value).is_some() {
                        return Err(format!(
                            "--expected-source-snapshot-seal-sha256 was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-terminal-path" => {
                    if expected_terminal_path
                        .replace(PathBuf::from(value))
                        .is_some()
                    {
                        return Err(format!(
                            "--expected-terminal-path was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-terminal-seal-sha256" => {
                    if expected_terminal_seal_sha256.replace(value).is_some() {
                        return Err(format!(
                            "--expected-terminal-seal-sha256 was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-activation-capture-sha256" => {
                    if expected_activation_capture_sha256.replace(value).is_some() {
                        return Err(format!(
                            "--expected-activation-capture-sha256 was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--mode" => {
                    if mode.replace(Mode::parse(&value)?).is_some() {
                        return Err(format!("--mode was supplied more than once; {}", usage()));
                    }
                }
                "--token-id" => {
                    if token_id.replace(parse_u32(&value, "--token-id")?).is_some() {
                        return Err(format!(
                            "--token-id was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--prompt" => {
                    if prompt.replace(value).is_some() {
                        return Err(format!("--prompt was supplied more than once; {}", usage()));
                    }
                }
                "--max-new-tokens" => max_new_tokens = parse_usize(&value, "--max-new-tokens")?,
                "--max-seq-len" => max_seq_len = parse_usize(&value, "--max-seq-len")?,
                "--packed-matvec-kernel" => {
                    packed_matvec_kernel = parse_packed_matvec_kernel(&value)?
                }
                "--gate-up-swiglu-kernel" => {
                    gate_up_swiglu_kernel = parse_gate_up_swiglu_kernel(&value)?
                }
                "--prompt-template" => prompt_template = PromptTemplate::parse(&value)?,
                _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
            }
        }
        let manifest = required(manifest, "--manifest")?;
        if !manifest.is_absolute() {
            return Err("--manifest must be an absolute path".into());
        }
        let mode = required(mode, "--mode")?;
        match mode {
            Mode::Preflight => {
                if token_id.is_some() || prompt.is_some() {
                    return Err("preflight accepts neither --token-id nor --prompt".into());
                }
            }
            Mode::ForwardToken => {
                required(token_id, "--token-id for forward-token")?;
                if prompt.is_some() {
                    return Err("forward-token accepts --token-id, not --prompt".into());
                }
            }
            Mode::GenerateGreedy => {
                required(prompt.as_ref(), "--prompt for generate-greedy")?;
                if token_id.is_some() {
                    return Err("generate-greedy accepts --prompt, not --token-id".into());
                }
                if max_new_tokens == 0 {
                    return Err("--max-new-tokens must be positive for generate-greedy".into());
                }
            }
        }
        if max_seq_len == 0 {
            return Err("--max-seq-len must be positive".into());
        }
        let aw_only = expected_selection_path.is_some()
            || expected_selection_seal_sha256.is_some()
            || expected_source_snapshot_path.is_some()
            || expected_source_snapshot_seal_sha256.is_some()
            || expected_activation_capture_sha256.is_some();
        let reval_or_terminal = expected_revalidation_path.is_some()
            || expected_revalidation_seal_sha256.is_some()
            || expected_terminal_path.is_some()
            || expected_terminal_seal_sha256.is_some();
        let (activation_weighted, uniform_q4, uniform_qn) = if aw_only {
            let revalidation_path =
                required(expected_revalidation_path, "--expected-revalidation-path")?;
            let selection_path = required(expected_selection_path, "--expected-selection-path")?;
            let snapshot_path =
                required(expected_source_snapshot_path, "--expected-source-snapshot-path")?;
            let terminal_path = required(expected_terminal_path, "--expected-terminal-path")?;
            for (flag, path) in [
                ("--expected-revalidation-path", &revalidation_path),
                ("--expected-selection-path", &selection_path),
                ("--expected-source-snapshot-path", &snapshot_path),
                ("--expected-terminal-path", &terminal_path),
            ] {
                if !path.is_absolute() {
                    return Err(format!("{flag} must be an absolute path"));
                }
            }
            (
                Some(ActivationWeightedBindings {
                    expected_revalidation_path: revalidation_path,
                    expected_revalidation_seal_sha256: required(
                        expected_revalidation_seal_sha256,
                        "--expected-revalidation-seal-sha256",
                    )?,
                    expected_selection_path: selection_path,
                    expected_selection_seal_sha256: required(
                        expected_selection_seal_sha256,
                        "--expected-selection-seal-sha256",
                    )?,
                    expected_source_snapshot_path: snapshot_path,
                    expected_source_snapshot_seal_sha256: required(
                        expected_source_snapshot_seal_sha256,
                        "--expected-source-snapshot-seal-sha256",
                    )?,
                    expected_terminal_path: terminal_path,
                    expected_terminal_seal_sha256: required(
                        expected_terminal_seal_sha256,
                        "--expected-terminal-seal-sha256",
                    )?,
                    expected_activation_capture_sha256: required(
                        expected_activation_capture_sha256,
                        "--expected-activation-capture-sha256",
                    )?,
                }),
                None,
                None,
            )
        } else if reval_or_terminal {
            let revalidation_path =
                required(expected_revalidation_path, "--expected-revalidation-path")?;
            let terminal_path = required(expected_terminal_path, "--expected-terminal-path")?;
            for (flag, path) in [
                ("--expected-revalidation-path", &revalidation_path),
                ("--expected-terminal-path", &terminal_path),
            ] {
                if !path.is_absolute() {
                    return Err(format!("{flag} must be an absolute path"));
                }
            }
            let uq = UniformQ4Bindings {
                expected_revalidation_path: revalidation_path.clone(),
                expected_revalidation_seal_sha256: required(
                    expected_revalidation_seal_sha256.clone(),
                    "--expected-revalidation-seal-sha256",
                )?,
                expected_terminal_path: terminal_path.clone(),
                expected_terminal_seal_sha256: required(
                    expected_terminal_seal_sha256.clone(),
                    "--expected-terminal-seal-sha256",
                )?,
            };
            // Same bindings work for Qn; bits chosen later from schema.
            let un = UniformQnBindings {
                bits: 0,
                expected_revalidation_path: revalidation_path,
                expected_revalidation_seal_sha256: uq.expected_revalidation_seal_sha256.clone(),
                expected_terminal_path: terminal_path,
                expected_terminal_seal_sha256: uq.expected_terminal_seal_sha256.clone(),
            };
            (None, Some(uq), Some(un))
        } else {
            (None, None, None)
        };
        Ok(Arguments {
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
            activation_weighted,
            uniform_q4,
            uniform_qn,
            mode,
            token_id,
            prompt,
            max_new_tokens,
            max_seq_len,
            trace_dispatch,
            packed_matvec_kernel,
            gate_up_swiglu_kernel,
            prompt_template,
        })
    }

    fn admission(arguments: &Arguments) -> CompleteBinaryAdmission {
        CompleteBinaryAdmission {
            model: QwenCompleteBinaryModel::Qwen30Coder,
            expected_manifest_seal_sha256: arguments.expected_manifest_seal_sha256.clone(),
            expected_source_audit_seal_sha256: arguments.expected_source_audit_seal_sha256.clone(),
            expected_source_revision: arguments.expected_source_revision.clone(),
        }
    }


    fn uniform_q4_admission(arguments: &Arguments) -> Result<Qwen30UniformQ4Admission, String> {
        let uq = arguments.uniform_q4.as_ref().ok_or_else(|| {
            "uniform Q4 requires --expected-revalidation-path/seal and --expected-terminal-path/seal".to_string()
        })?;
        Ok(Qwen30UniformQ4Admission {
            expected_manifest_seal_sha256: arguments.expected_manifest_seal_sha256.clone(),
            expected_source_audit_seal_sha256: arguments.expected_source_audit_seal_sha256.clone(),
            expected_source_revision: arguments.expected_source_revision.clone(),
            expected_revalidation_path: uq.expected_revalidation_path.clone(),
            expected_revalidation_seal_sha256: uq.expected_revalidation_seal_sha256.clone(),
            expected_terminal_path: uq.expected_terminal_path.clone(),
            expected_terminal_seal_sha256: uq.expected_terminal_seal_sha256.clone(),
        })
    }

    fn uniform_qn_admission(arguments: &Arguments, bits: UniformQnBits) -> Result<Qwen30UniformQnAdmission, String> {
        let un = arguments.uniform_qn.as_ref().ok_or_else(|| {
            "uniform Qn requires --expected-revalidation-path/seal and --expected-terminal-path/seal".to_string()
        })?;
        Ok(Qwen30UniformQnAdmission {
            bits,
            expected_manifest_seal_sha256: arguments.expected_manifest_seal_sha256.clone(),
            expected_source_audit_seal_sha256: arguments.expected_source_audit_seal_sha256.clone(),
            expected_source_revision: arguments.expected_source_revision.clone(),
            expected_revalidation_path: un.expected_revalidation_path.clone(),
            expected_revalidation_seal_sha256: un.expected_revalidation_seal_sha256.clone(),
            expected_terminal_path: un.expected_terminal_path.clone(),
            expected_terminal_seal_sha256: un.expected_terminal_seal_sha256.clone(),
        })
    }

    fn activation_weighted_admission(arguments: &Arguments) -> Result<Qwen30ActivationWeightedSvdAdmission, String> {
        let aw = arguments.activation_weighted.as_ref().ok_or_else(|| {
            "activation-weighted SVD candidate requires --expected-revalidation-path/seal, --expected-selection-path/seal, --expected-source-snapshot-path/seal, --expected-terminal-path/seal, and --expected-activation-capture-sha256".to_string()
        })?;
        Ok(Qwen30ActivationWeightedSvdAdmission {
            expected_manifest_seal_sha256: arguments.expected_manifest_seal_sha256.clone(),
            expected_source_audit_seal_sha256: arguments.expected_source_audit_seal_sha256.clone(),
            expected_source_revision: arguments.expected_source_revision.clone(),
            expected_revalidation_path: aw.expected_revalidation_path.clone(),
            expected_revalidation_seal_sha256: aw.expected_revalidation_seal_sha256.clone(),
            expected_selection_path: aw.expected_selection_path.clone(),
            expected_selection_seal_sha256: aw.expected_selection_seal_sha256.clone(),
            expected_source_snapshot_path: aw.expected_source_snapshot_path.clone(),
            expected_source_snapshot_seal_sha256: aw.expected_source_snapshot_seal_sha256.clone(),
            expected_terminal_path: aw.expected_terminal_path.clone(),
            expected_terminal_seal_sha256: aw.expected_terminal_seal_sha256.clone(),
            expected_activation_capture_sha256: aw.expected_activation_capture_sha256.clone(),
        })
    }

    fn peek_manifest_schema(path: &std::path::Path) -> Result<String, String> {
        let raw = std::fs::read(path)
            .map_err(|error| format!("cannot read manifest {}: {error}", path.display()))?;
        let value: Value = serde_json::from_slice(&raw)
            .map_err(|error| format!("manifest is not JSON: {error}"))?;
        value
            .get("schema")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| "manifest lacks a string schema field".into())
    }

    fn duration_us(value: std::time::Duration) -> u64 {
        u64::try_from(value.as_micros()).unwrap_or(u64::MAX)
    }

    fn current_executable_sha256() -> Result<String, String> {
        let path = env::current_exe()
            .map_err(|error| format!("cannot resolve current runtime executable: {error}"))?;
        let mut file = File::open(&path).map_err(|error| {
            format!(
                "cannot open current runtime executable {}: {error}",
                path.display()
            )
        })?;
        let mut digest = Sha256::new();
        let mut chunk = [0u8; 1024 * 1024];
        loop {
            let read = file.read(&mut chunk).map_err(|error| {
                format!(
                    "cannot hash current runtime executable {}: {error}",
                    path.display()
                )
            })?;
            if read == 0 {
                break;
            }
            digest.update(&chunk[..read]);
        }
        Ok(format!("{:x}", digest.finalize()))
    }

    fn source_user_template_json(runtime: &Qwen30CompleteNativeRuntime, applied: bool) -> Value {
        let source = runtime.source_user_chat_template();
        json!({
            "mode": "source_user_chat_template",
            "source_template_bound": true,
            "applied_to_prompt": applied,
            "source_template_path": source.source_template_path,
            "source_template_sha256": source.source_template_sha256,
            "tokenizer_config_path": source.tokenizer_config_path,
            "tokenizer_config_sha256": source.tokenizer_config_sha256,
            "supported_message_shape": "one_user_message_no_system_no_tools",
        })
    }

    fn step_json(step: &Qwen30NativeGreedyStep) -> Value {
        let gate_up_swiglu_device_control_parity = step
            .gate_up_swiglu_device_control_parity
            .as_ref()
            .map(|parity| {
                json!({
                    "layers_compared": parity.layers_compared,
                    "routed_experts_compared": parity.routed_experts_compared,
                    "activation_values_compared": parity.activation_values_compared,
                    "max_abs_error": parity.max_abs_error,
                    "tolerance_max_abs": parity.tolerance_max_abs,
                    "device_buffer_comparison_only": true,
                })
            })
            .unwrap_or(Value::Null);
        json!({
            "position": step.position,
            "sampled_token_id": step.token_id,
            "elapsed_us_diagnostic_not_tps": duration_us(step.elapsed),
            "command_buffers": step.command_buffers,
            "metal_dispatches": step.metal_dispatches,
            "host_route_id_readbacks": step.host_route_id_readbacks,
            "host_sample_id_readbacks": step.host_sample_id_readbacks,
            "host_stage_interval_count": step.host_stage_intervals.len(),
            "gate_up_swiglu_device_control_parity": gate_up_swiglu_device_control_parity,
        })
    }

    fn generation_gate_up_swiglu_parity_json(
        generation: &hawking_core::model::qwen30_complete_runtime::Qwen30NativeGeneration,
    ) -> Value {
        let all_steps = generation
            .prefill_steps
            .iter()
            .chain(generation.steps.iter());
        let mut full_forwards = 0usize;
        let mut layers_compared = 0usize;
        let mut routed_experts_compared = 0usize;
        let mut activation_values_compared = 0usize;
        let mut max_abs_error = 0.0f32;
        let mut tolerance = None::<f32>;
        let mut absent = 0usize;
        for step in all_steps {
            full_forwards = full_forwards.saturating_add(1);
            let Some(parity) = step.gate_up_swiglu_device_control_parity.as_ref() else {
                absent = absent.saturating_add(1);
                continue;
            };
            layers_compared = layers_compared.saturating_add(parity.layers_compared);
            routed_experts_compared =
                routed_experts_compared.saturating_add(parity.routed_experts_compared);
            activation_values_compared =
                activation_values_compared.saturating_add(parity.activation_values_compared);
            max_abs_error = max_abs_error.max(parity.max_abs_error);
            if let Some(expected) = tolerance {
                if expected.to_bits() != parity.tolerance_max_abs.to_bits() {
                    return json!({
                        "enabled": true,
                        "valid": false,
                        "reason": "fused gate/up parity tolerance changed between native forwards",
                    });
                }
            } else {
                tolerance = Some(parity.tolerance_max_abs);
            }
        }
        if full_forwards == 0 || absent == full_forwards {
            return Value::Null;
        }
        json!({
            "enabled": true,
            "valid": absent == 0 && tolerance.is_some(),
            "full_model_forwards_compared": full_forwards.saturating_sub(absent),
            "full_model_forwards_without_device_parity": absent,
            "layers_compared": layers_compared,
            "routed_experts_compared": routed_experts_compared,
            "activation_values_compared": activation_values_compared,
            "max_abs_error": max_abs_error,
            "tolerance_max_abs": tolerance,
            "all_selected_route_major_activations_compared_on_device": absent == 0,
            "raw_bf16_or_cpu_model_oracle_not_used": true,
            "claim_boundary": "candidate numerical parity diagnostic only; not a runtime, HCLI, TPS, TG, capability, or tournament receipt",
        })
    }

    fn profiler_json(
        snapshot: Qwen30NativeProfilerSnapshot,
        expected_complete_token_dispatch_samples: Option<usize>,
        host_stage_step: Option<&Qwen30NativeGreedyStep>,
    ) -> Value {
        // Preserve the ordered device trace for the separate profile-token
        // stage. Grouped totals below are convenient, but cannot prove that
        // every expected decoder dispatch occurred in its architecture-valid
        // order. This remains instrumentation evidence only, never TPS.
        let raw_dispatch_samples = snapshot
            .dispatch_samples
            .iter()
            .map(|sample| {
                json!({
                    "kernel_name": sample.kernel_name,
                    "wall_us": sample.wall_us,
                    "gpu_us": sample.gpu_us,
                    "gpu_start_ns": sample.gpu_start_ns,
                    "gpu_end_ns": sample.gpu_end_ns,
                })
            })
            .collect::<Vec<_>>();
        let mut grouped: BTreeMap<&str, (u64, u64, u64, u64)> = BTreeMap::new();
        for sample in snapshot.dispatch_samples {
            let row = grouped.entry(sample.kernel_name).or_insert((0, 0, 0, 0));
            row.0 = row.0.saturating_add(1);
            row.1 = row.1.saturating_add(sample.wall_us);
            if let Some(gpu_us) = sample.gpu_us {
                row.2 = row.2.saturating_add(1);
                row.3 = row.3.saturating_add(gpu_us);
            }
        }
        let total_encoding_wall_us = grouped
            .values()
            .fold(0u64, |sum, row| sum.saturating_add(row.1));
        let total_sampled_gpu_us = grouped
            .values()
            .fold(0u64, |sum, row| sum.saturating_add(row.3));
        let gpu_timing_sample_count = raw_dispatch_samples
            .iter()
            .filter(|sample| sample.get("gpu_us").and_then(Value::as_u64).is_some())
            .count();
        let complete_token_gpu_profile_coverage_earned = expected_complete_token_dispatch_samples
            .map(|expected| {
                raw_dispatch_samples.len() == expected
                    && gpu_timing_sample_count == expected
                    && std::env::var("HAWKING_TCB_TRACE").ok().as_deref() == Some("gpu_prod")
            });
        // Host-wall attribution comes from the same `forward_token_greedy`
        // timer as `execution.step.elapsed_us_diagnostic_not_tps`.  It is
        // intentionally present only for the one-token diagnostic mode; a
        // normal generation/server path does not collect interval telemetry.
        let (
            host_stage_timer_origin,
            host_stage_intervals,
            host_stage_covered_us,
            host_stage_interval_coverage_earned,
        ) = match host_stage_step {
            Some(step) => {
                let total_us = duration_us(step.elapsed);
                let mut previous_end = 0u64;
                let mut covered_us = 0u64;
                let mut valid = total_us > 0;
                let intervals = step
                    .host_stage_intervals
                    .iter()
                    .map(|interval| {
                        if interval.start_us < previous_end
                            || interval.end_us < interval.start_us
                            || interval.end_us > total_us
                        {
                            valid = false;
                        }
                        previous_end = interval.end_us;
                        covered_us = covered_us
                            .saturating_add(interval.end_us.saturating_sub(interval.start_us));
                        json!({
                            "bucket": interval.bucket,
                            "label": interval.label,
                            "start_us": interval.start_us,
                            "end_us": interval.end_us,
                        })
                    })
                    .collect::<Vec<_>>();
                let coverage_earned =
                    valid && covered_us.saturating_mul(100) >= total_us.saturating_mul(98);
                (
                    Value::String("complete_token_runtime_start".to_string()),
                    Value::Array(intervals),
                    Value::from(covered_us),
                    Value::Bool(coverage_earned),
                )
            }
            None => (Value::Null, Value::Null, Value::Null, Value::Null),
        };
        let kernels = grouped
            .into_iter()
            .map(|(kernel_name, (dispatches, encoding_wall_us, gpu_samples, gpu_us_sum))| {
                json!({
                    "kernel": kernel_name,
                    "dispatches": dispatches,
                    "encoding_wall_us": encoding_wall_us,
                    "encoding_wall_share_percent": if total_encoding_wall_us == 0 { Value::Null } else { json!(encoding_wall_us as f64 * 100.0 / total_encoding_wall_us as f64) },
                    "gpu_timing_samples": gpu_samples,
                    "gpu_us_sum_when_available": if gpu_samples == 0 { Value::Null } else { json!(gpu_us_sum) },
                    "gpu_sampled_share_percent_when_available": if gpu_samples == 0 || total_sampled_gpu_us == 0 { Value::Null } else { json!(gpu_us_sum as f64 * 100.0 / total_sampled_gpu_us as f64) },
                })
            })
            .collect::<Vec<_>>();
        json!({
            "diagnostic_only_not_clean_tps": true,
            "tcb_trace_mode_requested": std::env::var("HAWKING_TCB_TRACE").ok(),
            "total_encoding_wall_us": total_encoding_wall_us,
            "total_sampled_gpu_us_when_available": if total_sampled_gpu_us == 0 { Value::Null } else { json!(total_sampled_gpu_us) },
            "dispatch_sample_count": raw_dispatch_samples.len(),
            "gpu_timing_sample_count": gpu_timing_sample_count,
            "expected_complete_token_dispatch_samples": expected_complete_token_dispatch_samples,
            "complete_token_gpu_profile_coverage_earned": complete_token_gpu_profile_coverage_earned,
            "host_stage_timer_origin": host_stage_timer_origin,
            "host_stage_intervals": host_stage_intervals,
            "host_stage_covered_us": host_stage_covered_us,
            "host_stage_interval_coverage_earned": host_stage_interval_coverage_earned,
            "ordered_dispatch_samples": raw_dispatch_samples,
            "buffers_created": snapshot.buffers_created,
            "bytes_allocated": snapshot.bytes_allocated,
            "command_buffers_committed": snapshot.command_buffers_committed,
            "kernels": kernels,
        })
    }

    fn print_json(mut value: Value) {
        // Attach process-local startup phase timers when HAWKING_STARTUP_TIMING=1.
        // Always emit to stderr (never stdout — stdout is the receipt).
        hawking_core::startup_timing::emit_stderr_json();
        let snap = hawking_core::startup_timing::snapshot();
        if snap.enabled {
            if let Some(object) = value.as_object_mut() {
                object.insert("startup_timing".into(), snap.to_json());
            }
        }
        println!(
            "{}",
            serde_json::to_string(&value).expect("runtime result must serialize")
        );
    }

    fn fail(detail: impl AsRef<str>) -> ! {
        // Surface partial startup phases even on fail-closed exits.
        hawking_core::startup_timing::emit_stderr_json();
        eprintln!(
            "qwen30 complete native runtime refused: {}",
            detail.as_ref()
        );
        process::exit(2);
    }

    pub fn run() {
        hawking_core::startup_timing::mark_process_start();
        let arguments = parse_arguments().unwrap_or_else(|error| fail(error));
        let admission = admission(&arguments);
        let schema = peek_manifest_schema(&arguments.manifest).unwrap_or_else(|error| fail(error));
        let is_activation_weighted = schema == QWEN30_ACTIVATION_WEIGHTED_SVD_SCHEMA;
        let is_uniform_q4 = schema == QWEN30_UNIFORM_Q4_SCHEMA;
        let is_uniform_q3 = schema == "hawking.ascension.qwen30_uniform_q3_group128_candidate.v1";
        let is_uniform_q2 = schema == "hawking.ascension.qwen30_uniform_q2_group128_candidate.v1";
        let is_uniform_qn = is_uniform_q2 || is_uniform_q3;
        if is_activation_weighted && arguments.activation_weighted.is_none() {
            fail("manifest schema is activation-weighted SVD; supply AW handoff bindings");
        }
        if !is_activation_weighted && arguments.activation_weighted.is_some() {
            fail("activation-weighted handoff bindings supplied for non-AW schema");
        }
        if is_uniform_q4 && arguments.uniform_q4.is_none() {
            fail("uniform Q4 schema needs revalidation+terminal seals");
        }
        if is_uniform_qn && arguments.uniform_qn.is_none() {
            fail("uniform Qn schema needs revalidation+terminal seals");
        }
        if !is_activation_weighted
            && !is_uniform_q4
            && !is_uniform_qn
            && schema != QWEN30_COMPLETE_BINARY_SCHEMA
        {
            fail(format!("unsupported Qwen30 complete-native manifest schema {schema:?}"));
        }
        let runtime_executable_sha256 =
            current_executable_sha256().unwrap_or_else(|error| fail(error));
        if arguments.mode == Mode::Preflight {
            if is_activation_weighted {
                let aw = activation_weighted_admission(&arguments)
                    .unwrap_or_else(|error| fail(error));
                let artifact =
                    hawking_core::model::qwen_complete_binary::admit_qwen30_activation_weighted_svd_artifact(
                        &arguments.manifest,
                        &aw,
                    )
                    .unwrap_or_else(|error| fail(error.to_string()));
                print_json(json!({
                    "schema": RESULT_SCHEMA,
                    "status": "EARNED_QWEN30_ACTIVATION_WEIGHTED_SVD_NATIVE_ADMISSION_PREFLIGHT_NOT_TOKEN_EXECUTION",
                    "mode": arguments.mode.name(),
                    "runtime_executable_sha256": runtime_executable_sha256,
                    "preflight": {
                        "manifest_path": artifact.manifest_path,
                        "manifest_seal_sha256": artifact.manifest_seal_sha256,
                        "source_revision": artifact.source_revision,
                        "tensor_count": artifact.tensors.len(),
                        "tensor_payload_bytes": artifact.tensor_payload_bytes,
                        "source_weight_elements": artifact.source_weight_elements,
                        "selected_hgravs01_organs": artifact.selected_hgravs_organs.len(),
                        "verified_payload_count": artifact.verified_payload_count(),
                        "complete_verified_payload_cache_at_admission": artifact.has_complete_verified_payload_cache(),
                        "mixed_layout": "HQ30G1B1+HGRAVS01",
                        "hgravs01_executes_natively_as_two_stage_low_rank_matvec": true,
                        "dense_reconstruction_on_token_path": false,
                        "preflight_payload_snapshots_are_process_local": true,
                    },
                    "claim_boundary": {
                        "strict_mixed_artifact_admission_only": true,
                        "no_full_token_has_executed": true,
                        "not_generation_capability_hcli_clean_tps_tg_or_tournament_qualification": true,
                    },
                }));
                return;
            }
            let preflight = preflight_complete_runtime(&arguments.manifest, &admission)
                .unwrap_or_else(|error| fail(error.to_string()));
            print_json(json!({
                "schema": RESULT_SCHEMA,
                "status": "EARNED_QWEN30_DIRECT_PACKED_NATIVE_RUNTIME_PREFLIGHT_NOT_TOKEN_EXECUTION",
                "mode": arguments.mode.name(),
                "runtime_executable_sha256": runtime_executable_sha256,
                "preflight": {
                    "manifest_path": preflight.manifest_path,
                    "manifest_seal_sha256": preflight.manifest_seal_sha256,
                    "source_revision": preflight.source_revision,
                    "config_path": preflight.config_path,
                    "config_sha256": preflight.config_sha256,
                    "tokenizer_path": preflight.tokenizer_path,
                    "tokenizer_sha256": preflight.tokenizer_sha256,
                    "tokenizer_addressable_vocab": preflight.tokenizer_addressable_vocab,
                    "source_user_chat_template_path": preflight.source_user_chat_template_path,
                    "source_user_chat_template_sha256": preflight.source_user_chat_template_sha256,
                    "tokenizer_config_path": preflight.tokenizer_config_path,
                    "tokenizer_config_sha256": preflight.tokenizer_config_sha256,
                    "source_user_chat_template_bound": true,
                    "complete_exact_tensor_catalog_bound": true,
                    "tensor_count": preflight.tensor_count,
                    "tensor_payload_bytes": preflight.tensor_payload_bytes,
                    "source_weight_elements": preflight.source_weight_elements,
                    "direct_layout_group_size": preflight.direct_layout_group_size,
                    "verified_payload_count": preflight.verified_payload_count,
                    "complete_verified_payload_cache_at_admission": preflight.complete_verified_payload_cache_at_admission,
                    "preflight_payload_snapshots_are_process_local": true,
                },
                "claim_boundary": {
                    "strict_direct_packed_artifact_config_tokenizer_catalog_binding_only": true,
                    "no_full_token_has_executed": true,
                    "not_generation_capability_hcli_clean_tps_tg_or_tournament_qualification": true,
                },
            }));
            return;
        }

        let options = Qwen30CompleteRuntimeOptions {
            max_seq_len: arguments.max_seq_len,
            trace_dispatch: arguments.trace_dispatch,
            packed_matvec_kernel: if is_uniform_q4 || is_uniform_qn {
                Qwen30PackedMatvecKernel::SerialControl
            } else {
                arguments.packed_matvec_kernel
            },
            gate_up_swiglu_kernel: if is_uniform_q4 || is_uniform_qn {
                Qwen30GateUpSwiGluKernel::ThreeDispatchControl
            } else {
                arguments.gate_up_swiglu_kernel
            },
        };
        let mut runtime = hawking_core::startup_timing::time_ms("startup:runtime_load_total", || {
            if is_uniform_q4 {
                let uq = uniform_q4_admission(&arguments).unwrap_or_else(|error| fail(error));
                Qwen30CompleteNativeRuntime::load_uniform_q4(&arguments.manifest, &uq, options)
                    .unwrap_or_else(|error| fail(error.to_string()))
            } else if is_uniform_qn {
                let bits = if is_uniform_q3 { UniformQnBits::Three } else { UniformQnBits::Two };
                let un = uniform_qn_admission(&arguments, bits).unwrap_or_else(|error| fail(error));
                Qwen30CompleteNativeRuntime::load_uniform_qn(&arguments.manifest, &un, options)
                    .unwrap_or_else(|error| fail(error.to_string()))
            } else if is_activation_weighted {
                let aw =
                    activation_weighted_admission(&arguments).unwrap_or_else(|error| fail(error));
                Qwen30CompleteNativeRuntime::load_activation_weighted_svd(
                    &arguments.manifest,
                    &aw,
                    options,
                )
                .unwrap_or_else(|error| fail(error.to_string()))
            } else {
                Qwen30CompleteNativeRuntime::load(&arguments.manifest, &admission, options)
                    .unwrap_or_else(|error| fail(error.to_string()))
            }
        });
        // Exclude constructor allocation from the bounded execution profile.
        let _ = runtime.drain_profiler();
        let runtime_binding = json!({
            "manifest_seal_sha256": runtime.artifact_manifest_seal(),
            "source_revision": runtime.config.source_revision,
            "architecture": "Qwen3MoeForCausalLM",
            "layers": runtime.config.layers,
            "hidden_size": runtime.config.hidden,
            "attention_heads": runtime.config.attention_heads,
            "key_value_heads": runtime.config.key_value_heads,
            "head_dim": runtime.config.head_dim,
            "experts": runtime.config.experts,
            "experts_per_token": runtime.config.experts_per_token,
            "model_vocab_size": runtime.config.vocab_size,
            "tokenizer_addressable_vocab": runtime.tokenizer_addressable_vocab(),
            "native_max_seq_len": runtime.max_seq_len(),
            "metal_only": true,
            "raw_bf16_loader_not_opened": true,
            "raw_bf16_teacher_not_runtime_participant": true,
            "model_alone": true,
            "no_host_model_math_fallback": true,
            "fallback_count": 0,
            "activation_weighted_svd": {
                "enabled": runtime.has_activation_weighted_svd_organs(),
                "hgravs01_executes_natively_as_two_stage_low_rank_matvec": runtime.has_activation_weighted_svd_organs(),
                "dense_reconstruction_on_token_path": false,
            },
            "immutable_complete_payload_catalog": {
                "validated_during_process_admission": true,
                "verified_payload_count": runtime.verified_payload_count(),
                "expected_complete_tensor_count": 18867,
                "complete_verified_payload_cache": runtime.has_complete_verified_payload_cache(),
                "payload_access_path": if runtime.has_activation_weighted_svd_organs() {
                    "immutable_admission_verified_mixed_hq30g1b1_hgravs01_snapshot"
                } else {
                    "immutable_admission_verified_direct_snapshot"
                },
                "per_token_payload_sha256_rescan": false,
                "full_artifact_revalidation_required_on_process_restart": true,
            },
            "source_user_chat_template": source_user_template_json(&runtime, false),
            "packed_matvec_kernel": runtime.packed_matvec_kernel().receipt_name(),
            "gate_up_swiglu_kernel": runtime.gate_up_swiglu_kernel().receipt_name(),
        });

        match arguments.mode {
            Mode::ForwardToken => {
                let input_token_id = arguments.token_id.expect("validated forward-token id");
                // Production-mode cost ledger (HAWKING_COST_LEDGER=1): 18 buckets +
                // DeviceTimeline existed but nothing ever opened a token, so
                // is_recording() was always false. gpu_prod cannot answer this -- it
                // early-returns from begin_serial_group, giving one encoder per dispatch.
                let ledger_open = hawking_core::cost_ledger::begin_token();
                let step = runtime
                    .forward_token_greedy(input_token_id)
                    .unwrap_or_else(|error| fail(error.to_string()));
                let cost_ledger_json = if ledger_open {
                    hawking_core::cost_ledger::end_token()
                        .map(|report| report.to_json_value())
                        .unwrap_or(Value::Null)
                } else {
                    Value::Null
                };
                // `step.metal_dispatches` counts graph dispatches; native
                // first-token residency also lazily decodes four RMS vectors
                // per layer plus final norm (48 * 4 + 1) through distinct
                // Metal command buffers. Keep the expected trace cardinality
                // explicit so a truncated counter-sample run fails closed.
                let expected_profile_dispatches = step.metal_dispatches.saturating_add(193);
                let profiler = profiler_json(
                    runtime.drain_profiler(),
                    Some(expected_profile_dispatches),
                    Some(&step),
                );
                print_json(json!({
                    "schema": RESULT_SCHEMA,
                    "status": if is_activation_weighted {
                        "EARNED_QWEN30_MIXED_HQ30G1B1_HGRAVS01_NATIVE_METAL_FULL_TOKEN_EXECUTED_UNQUALIFIED"
                    } else {
                        "EARNED_QWEN30_DIRECT_PACKED_NATIVE_METAL_FULL_TOKEN_EXECUTED_UNQUALIFIED"
                    },
                    "mode": arguments.mode.name(),
                    "runtime_executable_sha256": runtime_executable_sha256,
                    "runtime_binding": runtime_binding,
                    "cost_ledger": cost_ledger_json,
                    "execution": {
                        "input_token_id": input_token_id,
                        "all_layers_executed": true,
                        "all_48_layers_executed": true,
                        "fallback_count": 0,
                        "final_norm_lm_head_device_argmax_executed": true,
                        "step": step_json(&step),
                    },
                    "profiler": profiler,
                    "claim_boundary": {
                        "one_direct_packed_native_full_token_executed": true,
                        "not_prompt_dependent_generation_or_coherence_evidence": true,
                        "not_hcli_clean_tps_tg_capability_or_tournament_qualification": true,
                    },
                }));
            }
            Mode::GenerateGreedy => {
                let prompt = arguments
                    .prompt
                    .as_deref()
                    .expect("validated generation prompt");
                let (generation, prompt_template) = hawking_core::startup_timing::time_ms(
                    "startup:generation_total",
                    || match arguments.prompt_template {
                        PromptTemplate::RawTextDiagnostic => (
                            runtime
                                .generate_greedy(prompt, arguments.max_new_tokens)
                                .unwrap_or_else(|error| fail(error.to_string())),
                            json!({
                                "mode": PromptTemplate::RawTextDiagnostic.receipt_name(),
                                "source_template_bound": false,
                                "applied_to_prompt": false,
                                "claim_boundary": "raw text diagnostic only; not eligible for template-bound runtime promotion",
                            }),
                        ),
                        PromptTemplate::SourceUserChat => (
                            runtime
                                .generate_source_user_chat_greedy(prompt, arguments.max_new_tokens)
                                .unwrap_or_else(|error| fail(error.to_string())),
                            source_user_template_json(&runtime, true),
                        ),
                    },
                );
                let profiler = profiler_json(runtime.drain_profiler(), None, None);
                let steps = generation.steps.iter().map(step_json).collect::<Vec<_>>();
                let gate_up_swiglu_device_control_parity =
                    generation_gate_up_swiglu_parity_json(&generation);
                let full_forward_count = generation
                    .prompt_token_ids
                    .len()
                    .saturating_add(generation.steps.len());
                print_json(json!({
                    "schema": RESULT_SCHEMA,
                    "status": "EARNED_QWEN30_DIRECT_PACKED_NATIVE_GREEDY_AUTOREGRESSIVE_EXECUTED_UNQUALIFIED",
                    "mode": arguments.mode.name(),
                    "runtime_executable_sha256": runtime_executable_sha256,
                    "runtime_binding": runtime_binding,
                    "execution": {
                        "all_48_layers_executed_for_each_forward": true,
                        "final_norm_lm_head_device_argmax_executed": true,
                        "prompt_template": prompt_template,
                        "prompt_token_ids": generation.prompt_token_ids,
                        "completion_token_ids": generation.completion_token_ids,
                        "completion_text_unscored": generation.completion_text,
                        "ended_on_eog": generation.ended_on_eog,
                        "prompt_full_forwards": generation.prompt_token_ids.len(),
                        "completion_feedback_full_forwards": generation.steps.len(),
                        "full_model_forward_count": full_forward_count,
                        "autoregressive_feedback_executed": !generation.steps.is_empty(),
                        "steps": steps,
                        "gate_up_swiglu_device_control_parity": gate_up_swiglu_device_control_parity,
                    },
                    "profiler": profiler,
                    "claim_boundary": {
                        "direct_packed_native_greedy_execution_only": true,
                        "coherence_prompt_dependence_capability_and_hcli_are_not_yet_scored": true,
                        "diagnostic_timing_is_not_clean_tps": true,
                        "not_tg_or_tournament_qualification": true,
                    },
                }));
            }
            Mode::Preflight => unreachable!("handled before native runtime construction"),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn production_paired_scalar_order_cli_is_separate_from_diagnostic_parity() {
            assert_eq!(
                parse_gate_up_swiglu_kernel("paired-scalar-order-production-no-parity").unwrap(),
                Qwen30GateUpSwiGluKernel::PairedScalarOrderProductionNoParity,
            );
            assert_eq!(
                parse_gate_up_swiglu_kernel("paired-scalar-order-candidate-device-parity").unwrap(),
                Qwen30GateUpSwiGluKernel::PairedScalarOrderCandidateWithDeviceControlParity,
            );
            assert!(parse_gate_up_swiglu_kernel("paired-scalar-order").is_err());
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos::run();
}

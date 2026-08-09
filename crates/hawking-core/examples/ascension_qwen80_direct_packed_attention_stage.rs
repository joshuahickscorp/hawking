//! Direct-packed Qwen3-Coder-Next layer-3 GQA component parity probe.
//!
//! The probe admits the sealed Qwen80 complete-binary catalog, then uses only
//! its compact sign/FP16-group-scale tensors for one deterministic two-token
//! layer-3 full-attention fixture. It covers Q/K/V/O projections, Q/K
//! residual-RMSNorm, partial RoPE, a two-entry GQA KV cache, causal attention,
//! and the source q_proj sigmoid gate. The CPU path is only a parity oracle
//! over that same compact representation; it never opens raw BF16 weights.
//!
//! This is deliberately a mixer-stage proof. It is not a complete decoder
//! layer, generation, HCLI, capability, BASE_TRUE_TPS, TG rung, or tournament
//! qualification result.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::kernels::{mha_decode_f32_tcb, qwen_binary_sign_scale_matvec_component_tcb};
    use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
    use hawking_core::model::qwen80_complete_runtime::{
        Qwen80CompleteArtifactCatalog, Qwen80LayerKind,
    };
    use hawking_core::model::qwen_complete_binary::{
        decode_complete_binary_f32, parse_complete_binary_header, CompleteBinaryAdmission,
        QwenCompleteBinaryModel,
    };
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::env;
    use std::fs::{self, OpenOptions};
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::process;
    use std::time::{SystemTime, UNIX_EPOCH};

    const LAYER: usize = 3;
    const HIDDEN: usize = 2048;
    const QUERY_HEADS: usize = 16;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const QUERY_DIM: usize = QUERY_HEADS * HEAD_DIM;
    const KV_DIM: usize = KV_HEADS * HEAD_DIM;
    const Q_PROJ_ROWS: usize = QUERY_DIM * 2;
    const ROTARY_DIM: usize = 64;
    const GROUP_SIZE: usize = 128;
    const ROPE_THETA: f32 = 5_000_000.0;
    const RMS_EPSILON: f32 = 1.0e-6;
    const POSITIONS: usize = 2;

    const RESULT_SCHEMA: &str =
        "hawking.ascension.qwen80_direct_packed_layer3_gqa_attention_stage.v1";
    const CPU_STATUS: &str =
        "QWEN80_ADMITTED_DIRECT_PACKED_LAYER3_GQA_CPU_ORACLE_READY_METAL_LEASE_REQUIRED";
    const METAL_STATUS: &str =
        "EARNED_QWEN80_ADMITTED_DIRECT_PACKED_LAYER3_GQA_TWO_TOKEN_COMPONENT_STAGE_NOT_COMPLETE_LAYER_OR_TOKEN";

    trait StageSetScalar {
        fn stage_set_u32(&self, index: u64, value: u32);
        fn stage_set_f32(&self, index: u64, value: f32);
    }

    impl StageSetScalar for ::metal::ComputeCommandEncoderRef {
        #[inline(always)]
        fn stage_set_u32(&self, index: u64, value: u32) {
            self.set_bytes(
                index,
                std::mem::size_of::<u32>() as u64,
                &value as *const u32 as *const _,
            );
        }

        #[inline(always)]
        fn stage_set_f32(&self, index: u64, value: f32) {
            self.set_bytes(
                index,
                std::mem::size_of::<f32>() as u64,
                &value as *const f32 as *const _,
            );
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Mode {
        CpuOracle,
        Metal,
    }

    struct Arguments {
        manifest: PathBuf,
        expected_manifest_seal_sha256: String,
        expected_source_audit_seal_sha256: String,
        expected_source_revision: String,
        capture_dir: PathBuf,
        mode: Mode,
    }

    struct DecodedMatrix {
        rows: usize,
        cols: usize,
        values: Vec<f32>,
    }

    struct CpuToken {
        q_projection: Vec<f32>,
        k_projection: Vec<f32>,
        v_projection: Vec<f32>,
        query: Vec<f32>,
        key: Vec<f32>,
        value: Vec<f32>,
        attention: Vec<f32>,
        gated: Vec<f32>,
        output: Vec<f32>,
    }

    struct CpuStage {
        tokens: Vec<CpuToken>,
    }

    struct GpuPackedTensor {
        signs: PinnedBuffer,
        scales: PinnedBuffer,
    }

    #[derive(Default)]
    struct ErrorLedger {
        q_projection: f32,
        k_projection: f32,
        v_projection: f32,
        q_norm_rope: f32,
        k_norm_rope_cache: f32,
        value_cache: f32,
        gqa_attention: f32,
        sigmoid_gate: f32,
        o_projection: f32,
    }

    fn usage() -> &'static str {
        concat!(
            "usage: ascension_qwen80_direct_packed_attention_stage \\\n",
            "    --manifest ABSOLUTE_PATH \\\n",
            "    --expected-manifest-seal-sha256 SHA256 \\\n",
            "    --expected-source-audit-seal-sha256 SHA256 \\\n",
            "    --expected-source-revision REVISION \\\n",
            "    --capture-dir NEW_ABSOLUTE_DIRECTORY \\\n",
            "    [--mode cpu-oracle|metal]"
        )
    }

    fn parse_arguments() -> Result<Arguments, String> {
        let mut manifest = None;
        let mut expected_manifest_seal_sha256 = None;
        let mut expected_source_audit_seal_sha256 = None;
        let mut expected_source_revision = None;
        let mut capture_dir = None;
        let mut mode = Mode::CpuOracle;
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
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
                "--capture-dir" => {
                    if capture_dir.replace(PathBuf::from(value)).is_some() {
                        return Err(format!(
                            "--capture-dir was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--mode" => {
                    mode = match value.as_str() {
                        "cpu-oracle" => Mode::CpuOracle,
                        "metal" => Mode::Metal,
                        _ => return Err(format!("unsupported --mode {value:?}; {}", usage())),
                    };
                }
                _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
            }
        }
        let manifest = manifest.ok_or_else(|| format!("missing --manifest; {}", usage()))?;
        if !manifest.is_absolute() {
            return Err("--manifest must be an absolute path".into());
        }
        let capture_dir =
            capture_dir.ok_or_else(|| format!("missing --capture-dir; {}", usage()))?;
        if !capture_dir.is_absolute() {
            return Err("--capture-dir must be an absolute path".into());
        }
        Ok(Arguments {
            manifest,
            expected_manifest_seal_sha256: expected_manifest_seal_sha256
                .ok_or_else(|| format!("missing --expected-manifest-seal-sha256; {}", usage()))?,
            expected_source_audit_seal_sha256: expected_source_audit_seal_sha256.ok_or_else(
                || format!("missing --expected-source-audit-seal-sha256; {}", usage()),
            )?,
            expected_source_revision: expected_source_revision
                .ok_or_else(|| format!("missing --expected-source-revision; {}", usage()))?,
            capture_dir,
            mode,
        })
    }

    fn mode_name(mode: Mode) -> &'static str {
        match mode {
            Mode::CpuOracle => "cpu-oracle",
            Mode::Metal => "metal",
        }
    }

    /// Create one exclusive, durable attempt directory before the catalog is
    /// opened. A later receipt is only trusted if this creation succeeded.
    fn begin_capture(arguments: &Arguments) -> Result<(), String> {
        let parent = arguments
            .capture_dir
            .parent()
            .ok_or_else(|| "--capture-dir must have a parent directory".to_string())?;
        if !parent.is_dir() {
            return Err(format!(
                "--capture-dir parent {:?} is not an existing directory",
                parent
            ));
        }
        fs::create_dir(&arguments.capture_dir).map_err(|error| {
            format!(
                "refusing to reuse or create non-exclusive --capture-dir {:?}: {error}",
                arguments.capture_dir
            )
        })?;
        let started_unix_millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("system clock is before Unix epoch: {error}"))?
            .as_millis();
        let invocation = json!({
            "schema": "hawking.ascension.qwen80_direct_packed_layer3_gqa_capture.v1",
            "status": "STARTED_QWEN80_DIRECT_PACKED_LAYER3_GQA_COMPONENT_ATTEMPT",
            "started_unix_millis": started_unix_millis,
            "mode": mode_name(arguments.mode),
            "manifest": arguments.manifest,
            "expected_manifest_seal_sha256": arguments.expected_manifest_seal_sha256,
            "expected_source_audit_seal_sha256": arguments.expected_source_audit_seal_sha256,
            "expected_source_revision": arguments.expected_source_revision,
            "claim_boundary": {
                "component_stage_only": true,
                "not_a_complete_layer_decoder_generation_hcli_or_tps_result": true,
            },
        });
        write_new_atomic(
            &arguments.capture_dir,
            "invocation.json",
            &serde_json::to_vec_pretty(&invocation).map_err(|error| error.to_string())?,
        )
    }

    /// Atomically publish a new named file without replacing any prior attempt
    /// evidence. The capture directory is created exclusively, then receipt is
    /// written last as the completion marker.
    fn write_new_atomic(capture_dir: &Path, name: &str, contents: &[u8]) -> Result<(), String> {
        let target = capture_dir.join(name);
        let temporary = capture_dir.join(format!(".{name}.{}.tmp", process::id()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| format!("cannot create capture temporary {:?}: {error}", temporary))?;
        if let Err(error) = file.write_all(contents).and_then(|_| file.sync_all()) {
            let _ = fs::remove_file(&temporary);
            return Err(format!(
                "cannot durably write capture temporary {:?}: {error}",
                temporary
            ));
        }
        drop(file);
        if let Err(error) = fs::hard_link(&temporary, &target) {
            let _ = fs::remove_file(&temporary);
            return Err(format!(
                "cannot atomically publish new capture {:?} from {:?}: {error}",
                target, temporary
            ));
        }
        fs::remove_file(&temporary)
            .map_err(|error| format!("cannot retire capture temporary {:?}: {error}", temporary))
    }

    fn failure_result(arguments: &Arguments, error: &str) -> Value {
        json!({
            "schema": RESULT_SCHEMA,
            "status": "REFUSED_QWEN80_DIRECT_PACKED_LAYER3_GQA_COMPONENT_ATTEMPT_ERROR",
            "mode": mode_name(arguments.mode),
            "error": error,
            "source_binding": {
                "manifest": arguments.manifest,
                "expected_manifest_seal_sha256": arguments.expected_manifest_seal_sha256,
                "expected_source_audit_seal_sha256": arguments.expected_source_audit_seal_sha256,
                "expected_source_revision": arguments.expected_source_revision,
            },
            "claim_boundary": {
                "component_stage_only": true,
                "no_cpu_or_metal_parity_is_claimed": true,
                "does_not_execute_a_complete_layer_or_decoder": true,
                "does_not_generate_tokens_expose_hcli_or_measure_tps": true,
            },
        })
    }

    /// Publish structured stdout, stderr, and receipt files. The receipt exists
    /// only after both stream files were atomically written, making it the
    /// completion marker for a future quiet-lease invocation.
    fn finalize_capture(
        arguments: &Arguments,
        stage_result: Result<Value, String>,
    ) -> Result<(Value, Option<String>), String> {
        let (mut result, failure) = match stage_result {
            Ok(result) => (result, None),
            Err(error) => (failure_result(arguments, &error), Some(error)),
        };
        {
            let object = result
                .as_object_mut()
                .ok_or_else(|| "stage result must be a JSON object".to_string())?;
            object.insert(
                "durable_capture".into(),
                json!({
                    "directory": arguments.capture_dir,
                    "invocation_file": "invocation.json",
                    "stdout_file": "stdout.jsonl",
                    "stderr_file": "stderr.log",
                    "receipt_file": "receipt.json",
                    "receipt_is_completion_marker": true,
                    "all_files_are_create_new_and_non_overwriting": true,
                }),
            );
        }
        let unsigned = serde_json::to_vec(&result).map_err(|error| error.to_string())?;
        let digest = format!("{:x}", Sha256::digest(&unsigned));
        {
            let object = result
                .as_object_mut()
                .ok_or_else(|| "stage result must be a JSON object".to_string())?;
            object.insert("receipt_unsigned_json_sha256".into(), json!(digest));
            object.insert(
                "receipt_digest_algorithm".into(),
                json!("sha256(serde_json::to_vec(unsigned_receipt))"),
            );
        }
        let mut stdout = serde_json::to_vec(&result).map_err(|error| error.to_string())?;
        stdout.push(10);
        let stderr = failure
            .as_ref()
            .map(|error| format!("{error}\n").into_bytes())
            .unwrap_or_default();
        // Streams must appear before the receipt completion marker.
        write_new_atomic(&arguments.capture_dir, "stdout.jsonl", &stdout)?;
        write_new_atomic(&arguments.capture_dir, "stderr.log", &stderr)?;
        let mut receipt = serde_json::to_vec_pretty(&result).map_err(|error| error.to_string())?;
        receipt.push(10);
        write_new_atomic(&arguments.capture_dir, "receipt.json", &receipt)?;
        Ok((result, failure))
    }

    fn bytes_for_f32(elements: usize, label: &str) -> Result<usize, String> {
        elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("{label} byte count overflows usize"))
    }

    fn fixture_hidden(seed: usize) -> Vec<f32> {
        (0..HIDDEN)
            .map(|index| {
                let value = ((index * 97 + seed * 43 + 29) % 8191) as f32 - 4095.0;
                value / 8191.0
            })
            .collect()
    }

    fn check_finite(values: &[f32], label: &str) -> Result<(), String> {
        if let Some((index, value)) = values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "{label} contains non-finite value {value} at {index}"
            ));
        }
        Ok(())
    }

    fn max_abs_error(expected: &[f32], observed: &[f32], label: &str) -> Result<f32, String> {
        if expected.len() != observed.len() {
            return Err(format!(
                "{label} parity length mismatch: expected {}, observed {}",
                expected.len(),
                observed.len()
            ));
        }
        let mut maximum = 0.0f32;
        for (index, (&expected, &observed)) in expected.iter().zip(observed).enumerate() {
            if !expected.is_finite() || !observed.is_finite() {
                return Err(format!(
                    "{label} parity has non-finite values at {index}: expected={expected}, observed={observed}"
                ));
            }
            maximum = maximum.max((expected - observed).abs());
        }
        Ok(maximum)
    }

    fn require_parity(label: &str, error: f32, tolerance: f32) -> Result<(), String> {
        if error > tolerance {
            return Err(format!(
                "{label} direct-packed Metal parity failed: max_abs_error={error}, tolerance={tolerance}"
            ));
        }
        Ok(())
    }

    fn decode_matrix(
        catalog: &Qwen80CompleteArtifactCatalog,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<DecodedMatrix, String> {
        let payload = catalog
            .read_direct_tensor_payload(name)
            .map_err(|error| error.to_string())?;
        let (header, values) =
            decode_complete_binary_f32(&payload).map_err(|error| error.to_string())?;
        if header.shape.as_slice() != [rows, cols] || header.group_size != GROUP_SIZE {
            return Err(format!(
                "direct packed tensor {name:?} has shape {:?}/group {}, expected [{rows}, {cols}]/{}",
                header.shape, header.group_size, GROUP_SIZE
            ));
        }
        if values.len() != rows * cols {
            return Err(format!(
                "direct packed tensor {name:?} decoded {} values, expected {}",
                values.len(),
                rows * cols
            ));
        }
        check_finite(&values, name)?;
        Ok(DecodedMatrix { rows, cols, values })
    }

    fn decode_vector(
        catalog: &Qwen80CompleteArtifactCatalog,
        name: &str,
        elements: usize,
    ) -> Result<Vec<f32>, String> {
        let payload = catalog
            .read_direct_tensor_payload(name)
            .map_err(|error| error.to_string())?;
        let (header, values) =
            decode_complete_binary_f32(&payload).map_err(|error| error.to_string())?;
        if header.shape.as_slice() != [elements] || header.group_size != GROUP_SIZE {
            return Err(format!(
                "direct packed vector {name:?} has shape {:?}/group {}, expected [{elements}]/{}",
                header.shape, header.group_size, GROUP_SIZE
            ));
        }
        if values.len() != elements {
            return Err(format!(
                "direct packed vector {name:?} decoded {} values, expected {elements}",
                values.len()
            ));
        }
        check_finite(&values, name)?;
        Ok(values)
    }

    fn packed_matvec(
        matrix: &DecodedMatrix,
        input: &[f32],
        label: &str,
    ) -> Result<Vec<f32>, String> {
        if input.len() != matrix.cols {
            return Err(format!(
                "{label} input length {} does not equal packed matrix columns {}",
                input.len(),
                matrix.cols
            ));
        }
        let mut output = vec![0.0f32; matrix.rows];
        for row in 0..matrix.rows {
            let weights = &matrix.values[row * matrix.cols..(row + 1) * matrix.cols];
            let mut sum = 0.0f32;
            for (&weight, &value) in weights.iter().zip(input) {
                sum += weight * value;
            }
            if !sum.is_finite() {
                return Err(format!(
                    "{label} compact CPU oracle is non-finite at row {row}"
                ));
            }
            output[row] = sum;
        }
        Ok(output)
    }

    fn source_norm_rope(
        raw: &[f32],
        weights: &[f32],
        heads: usize,
        sequence_slot: usize,
        label: &str,
    ) -> Result<Vec<f32>, String> {
        if raw.len() != heads * HEAD_DIM || weights.len() != HEAD_DIM {
            return Err(format!(
                "{label} received raw={} weights={} for heads={heads}, expected {}/{}",
                raw.len(),
                weights.len(),
                heads * HEAD_DIM,
                HEAD_DIM
            ));
        }
        let mut output = vec![0.0f32; raw.len()];
        for head in 0..heads {
            let base = head * HEAD_DIM;
            let variance = raw[base..base + HEAD_DIM]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / HEAD_DIM as f32;
            let inverse_rms = (variance + RMS_EPSILON).sqrt().recip();
            for dimension in 0..HEAD_DIM {
                // Qwen3-NextRMSNorm is norm(x) * (1 + learned_weight).
                output[base + dimension] =
                    raw[base + dimension] * inverse_rms * (1.0 + weights[dimension]);
            }
            // Qwen3Next's RoPE is non-interleaved rotate_half on the first
            // partial_rotary_factor * head_dim = 64 dimensions.
            let before = output[base..base + ROTARY_DIM].to_vec();
            for dimension in 0..(ROTARY_DIM / 2) {
                let inverse_frequency =
                    ROPE_THETA.powf(-2.0 * dimension as f32 / ROTARY_DIM as f32);
                let angle = sequence_slot as f32 * inverse_frequency;
                let cosine = angle.cos();
                let sine = angle.sin();
                output[base + dimension] =
                    before[dimension] * cosine - before[dimension + ROTARY_DIM / 2] * sine;
                output[base + dimension + ROTARY_DIM / 2] =
                    before[dimension + ROTARY_DIM / 2] * cosine + before[dimension] * sine;
            }
        }
        check_finite(&output, label)?;
        Ok(output)
    }

    fn query_from_interleaved_q_projection(q_projection: &[f32]) -> Result<Vec<f32>, String> {
        if q_projection.len() != Q_PROJ_ROWS {
            return Err(format!(
                "layer-3 q_proj has {} outputs, expected {Q_PROJ_ROWS}",
                q_projection.len()
            ));
        }
        let mut query = vec![0.0f32; QUERY_DIM];
        for head in 0..QUERY_HEADS {
            let source = head * 2 * HEAD_DIM;
            let destination = head * HEAD_DIM;
            query[destination..destination + HEAD_DIM]
                .copy_from_slice(&q_projection[source..source + HEAD_DIM]);
        }
        Ok(query)
    }

    fn gate_from_interleaved_q_projection(q_projection: &[f32]) -> Result<Vec<f32>, String> {
        if q_projection.len() != Q_PROJ_ROWS {
            return Err(format!(
                "layer-3 q_proj has {} outputs, expected {Q_PROJ_ROWS}",
                q_projection.len()
            ));
        }
        let mut gate = vec![0.0f32; QUERY_DIM];
        for head in 0..QUERY_HEADS {
            let source = head * 2 * HEAD_DIM + HEAD_DIM;
            let destination = head * HEAD_DIM;
            gate[destination..destination + HEAD_DIM]
                .copy_from_slice(&q_projection[source..source + HEAD_DIM]);
        }
        Ok(gate)
    }

    fn gqa_attention(
        query: &[f32],
        key_cache: &[f32],
        value_cache: &[f32],
        sequence_length: usize,
    ) -> Result<Vec<f32>, String> {
        if query.len() != QUERY_DIM
            || key_cache.len() < sequence_length * KV_DIM
            || value_cache.len() < sequence_length * KV_DIM
            || sequence_length == 0
        {
            return Err("Qwen80 GQA CPU oracle received an invalid exact cache geometry".into());
        }
        let group = QUERY_HEADS / KV_HEADS;
        let scale = (HEAD_DIM as f32).sqrt().recip();
        let mut output = vec![0.0f32; QUERY_DIM];
        for head in 0..QUERY_HEADS {
            let query_base = head * HEAD_DIM;
            let kv_head = head / group;
            let mut scores = Vec::with_capacity(sequence_length);
            for token in 0..sequence_length {
                let cache_base = (token * KV_HEADS + kv_head) * HEAD_DIM;
                let dot = (0..HEAD_DIM)
                    .map(|dimension| {
                        query[query_base + dimension] * key_cache[cache_base + dimension]
                    })
                    .sum::<f32>();
                scores.push(dot * scale);
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let normalizer = scores
                .iter()
                .map(|score| (score - maximum).exp())
                .sum::<f32>();
            if !normalizer.is_finite() || normalizer <= 0.0 {
                return Err(format!(
                    "Qwen80 GQA CPU oracle invalid softmax at head {head}"
                ));
            }
            for (token, score) in scores.iter().enumerate() {
                let probability = (*score - maximum).exp() / normalizer;
                let cache_base = (token * KV_HEADS + kv_head) * HEAD_DIM;
                for dimension in 0..HEAD_DIM {
                    output[query_base + dimension] +=
                        probability * value_cache[cache_base + dimension];
                }
            }
        }
        check_finite(&output, "Qwen80 GQA CPU attention")?;
        Ok(output)
    }

    fn run_cpu_stage(catalog: &Qwen80CompleteArtifactCatalog) -> Result<CpuStage, String> {
        const Q: &str = "model.layers.3.self_attn.q_proj.weight";
        const K: &str = "model.layers.3.self_attn.k_proj.weight";
        const V: &str = "model.layers.3.self_attn.v_proj.weight";
        const O: &str = "model.layers.3.self_attn.o_proj.weight";
        const Q_NORM: &str = "model.layers.3.self_attn.q_norm.weight";
        const K_NORM: &str = "model.layers.3.self_attn.k_norm.weight";

        let q = decode_matrix(catalog, Q, Q_PROJ_ROWS, HIDDEN)?;
        let k = decode_matrix(catalog, K, KV_DIM, HIDDEN)?;
        let v = decode_matrix(catalog, V, KV_DIM, HIDDEN)?;
        let o = decode_matrix(catalog, O, HIDDEN, QUERY_DIM)?;
        let q_norm = decode_vector(catalog, Q_NORM, HEAD_DIM)?;
        let k_norm = decode_vector(catalog, K_NORM, HEAD_DIM)?;
        let inputs = [fixture_hidden(31), fixture_hidden(167)];
        let mut key_cache = vec![0.0f32; POSITIONS * KV_DIM];
        let mut value_cache = vec![0.0f32; POSITIONS * KV_DIM];
        let mut tokens = Vec::with_capacity(POSITIONS);

        for (position, input) in inputs.iter().enumerate() {
            let q_projection = packed_matvec(&q, input, "layer-3 q_proj")?;
            let k_projection = packed_matvec(&k, input, "layer-3 k_proj")?;
            let v_projection = packed_matvec(&v, input, "layer-3 v_proj")?;
            let query_raw = query_from_interleaved_q_projection(&q_projection)?;
            let query = source_norm_rope(
                &query_raw,
                &q_norm,
                QUERY_HEADS,
                position,
                "layer-3 source q_norm + partial RoPE",
            )?;
            let key = source_norm_rope(
                &k_projection,
                &k_norm,
                KV_HEADS,
                position,
                "layer-3 source k_norm + partial RoPE",
            )?;
            let cache_start = position * KV_DIM;
            key_cache[cache_start..cache_start + KV_DIM].copy_from_slice(&key);
            value_cache[cache_start..cache_start + KV_DIM].copy_from_slice(&v_projection);
            let attention = gqa_attention(&query, &key_cache, &value_cache, position + 1)?;
            let gate = gate_from_interleaved_q_projection(&q_projection)?;
            let gated = attention
                .iter()
                .zip(&gate)
                .map(|(&value, &gate)| value * (1.0 + (-gate).exp()).recip())
                .collect::<Vec<_>>();
            check_finite(&gated, "layer-3 source q_proj sigmoid gate")?;
            let output = packed_matvec(&o, &gated, "layer-3 o_proj")?;
            tokens.push(CpuToken {
                q_projection,
                k_projection,
                v_projection,
                query,
                key,
                value: value_cache[cache_start..cache_start + KV_DIM].to_vec(),
                attention,
                gated,
                output,
            });
        }
        Ok(CpuStage { tokens })
    }

    fn assert_catalog_geometry(catalog: &Qwen80CompleteArtifactCatalog) -> Result<(), String> {
        let config = &catalog.config;
        if config
            .layer_kind(LAYER)
            .map_err(|error| error.to_string())?
            != Qwen80LayerKind::FullAttention
            || config.hidden != HIDDEN
            || config.attention_heads != QUERY_HEADS
            || config.key_value_heads != KV_HEADS
            || config.attention_head_dim != HEAD_DIM
            || config.rope_theta().to_bits() != ROPE_THETA.to_bits()
            || config.partial_rotary_factor().to_bits() != 0.25f32.to_bits()
            || config.rms_norm_eps().to_bits() != RMS_EPSILON.to_bits()
        {
            return Err(
                "admitted Qwen80 catalog drifted from exact layer-3 full-GQA geometry".into(),
            );
        }
        let required = [
            (
                "model.layers.3.self_attn.q_proj.weight",
                vec![Q_PROJ_ROWS, HIDDEN],
            ),
            (
                "model.layers.3.self_attn.k_proj.weight",
                vec![KV_DIM, HIDDEN],
            ),
            (
                "model.layers.3.self_attn.v_proj.weight",
                vec![KV_DIM, HIDDEN],
            ),
            (
                "model.layers.3.self_attn.o_proj.weight",
                vec![HIDDEN, QUERY_DIM],
            ),
            ("model.layers.3.self_attn.q_norm.weight", vec![HEAD_DIM]),
            ("model.layers.3.self_attn.k_norm.weight", vec![HEAD_DIM]),
        ];
        for (name, shape) in required {
            let header = catalog
                .direct_tensor_header(name)
                .map_err(|error| error.to_string())?;
            if header.shape != shape || header.group_size != GROUP_SIZE {
                return Err(format!(
                    "admitted layer-3 tensor {name:?} has shape {:?}/group {}, expected {:?}/{}",
                    header.shape, header.group_size, shape, GROUP_SIZE
                ));
            }
        }
        Ok(())
    }

    fn upload_packed_tensor(
        context: &MetalContext,
        catalog: &Qwen80CompleteArtifactCatalog,
        name: &str,
        expected_shape: &[usize],
    ) -> Result<GpuPackedTensor, String> {
        let payload = catalog
            .read_direct_tensor_payload(name)
            .map_err(|error| error.to_string())?;
        let header = parse_complete_binary_header(&payload).map_err(|error| error.to_string())?;
        if header.shape.as_slice() != expected_shape || header.group_size != GROUP_SIZE {
            return Err(format!(
                "GPU upload refused {name:?}: shape {:?}/group {}, expected {:?}/{}",
                header.shape, header.group_size, expected_shape, GROUP_SIZE
            ));
        }
        let scales = payload
            .get(header.scale_offset..header.sign_offset)
            .ok_or_else(|| format!("GPU upload {name:?} has truncated scale section"))?;
        let signs = payload
            .get(header.sign_offset..header.payload_bytes)
            .ok_or_else(|| format!("GPU upload {name:?} has truncated sign section"))?;
        if scales.len() != header.groups * std::mem::size_of::<u16>()
            || signs.len() != header.groups * (GROUP_SIZE / 8)
        {
            return Err(format!(
                "GPU upload {name:?} compact sections violate admitted header"
            ));
        }
        Ok(GpuPackedTensor {
            signs: context
                .new_buffer_with_bytes_checked(signs)
                .map_err(|error| error.to_string())?,
            scales: context
                .new_buffer_with_bytes_checked(scales)
                .map_err(|error| error.to_string())?,
        })
    }

    fn snapshot_f32(
        buffer: &PinnedBuffer,
        elements: usize,
        label: &str,
    ) -> Result<Vec<f32>, String> {
        let bytes = bytes_for_f32(elements, label)?;
        if buffer.length() < bytes as u64 {
            return Err(format!(
                "{label} snapshot needs {bytes} bytes but buffer only has {}",
                buffer.length()
            ));
        }
        Ok(unsafe {
            std::slice::from_raw_parts(buffer.contents() as *const f32, elements).to_vec()
        })
    }

    fn update_ledger(
        ledger: &mut ErrorLedger,
        cpu: &CpuToken,
        q_projection: &[f32],
        k_projection: &[f32],
        v_projection: &[f32],
        query: &[f32],
        key_cache: &[f32],
        value_cache: &[f32],
        position: usize,
        attention: &[f32],
        gated: &[f32],
        output: &[f32],
    ) -> Result<(), String> {
        let cache_start = position * KV_DIM;
        ledger.q_projection =
            ledger
                .q_projection
                .max(max_abs_error(&cpu.q_projection, q_projection, "q_proj")?);
        ledger.k_projection =
            ledger
                .k_projection
                .max(max_abs_error(&cpu.k_projection, k_projection, "k_proj")?);
        ledger.v_projection =
            ledger
                .v_projection
                .max(max_abs_error(&cpu.v_projection, v_projection, "v_proj")?);
        ledger.q_norm_rope =
            ledger
                .q_norm_rope
                .max(max_abs_error(&cpu.query, query, "q_norm + partial RoPE")?);
        ledger.k_norm_rope_cache = ledger.k_norm_rope_cache.max(max_abs_error(
            &cpu.key,
            &key_cache[cache_start..cache_start + KV_DIM],
            "k_norm + partial RoPE cache row",
        )?);
        ledger.value_cache = ledger.value_cache.max(max_abs_error(
            &cpu.value,
            &value_cache[cache_start..cache_start + KV_DIM],
            "value cache row",
        )?);
        ledger.gqa_attention =
            ledger
                .gqa_attention
                .max(max_abs_error(&cpu.attention, attention, "causal GQA")?);
        ledger.sigmoid_gate =
            ledger
                .sigmoid_gate
                .max(max_abs_error(&cpu.gated, gated, "q_proj sigmoid gate")?);
        ledger.o_projection =
            ledger
                .o_projection
                .max(max_abs_error(&cpu.output, output, "o_proj")?);
        Ok(())
    }

    fn run_metal_stage(
        catalog: &Qwen80CompleteArtifactCatalog,
        cpu: &CpuStage,
    ) -> Result<(String, ErrorLedger, usize), String> {
        const Q: &str = "model.layers.3.self_attn.q_proj.weight";
        const K: &str = "model.layers.3.self_attn.k_proj.weight";
        const V: &str = "model.layers.3.self_attn.v_proj.weight";
        const O: &str = "model.layers.3.self_attn.o_proj.weight";
        const Q_NORM: &str = "model.layers.3.self_attn.q_norm.weight";
        const K_NORM: &str = "model.layers.3.self_attn.k_norm.weight";

        if cpu.tokens.len() != POSITIONS {
            return Err("CPU oracle did not produce the exact two-token cache fixture".into());
        }
        let context = MetalContext::new_with_trace(false).map_err(|error| error.to_string())?;
        let q = upload_packed_tensor(&context, catalog, Q, &[Q_PROJ_ROWS, HIDDEN])?;
        let k = upload_packed_tensor(&context, catalog, K, &[KV_DIM, HIDDEN])?;
        let v = upload_packed_tensor(&context, catalog, V, &[KV_DIM, HIDDEN])?;
        let o = upload_packed_tensor(&context, catalog, O, &[HIDDEN, QUERY_DIM])?;
        let q_norm = upload_packed_tensor(&context, catalog, Q_NORM, &[HEAD_DIM])?;
        let k_norm = upload_packed_tensor(&context, catalog, K_NORM, &[HEAD_DIM])?;

        let input = context
            .new_buffer_checked(bytes_for_f32(HIDDEN, "layer-3 fixture input")?)
            .map_err(|error| error.to_string())?;
        let q_projection = context
            .new_buffer_checked(bytes_for_f32(Q_PROJ_ROWS, "layer-3 q projection")?)
            .map_err(|error| error.to_string())?;
        let k_projection = context
            .new_buffer_checked(bytes_for_f32(KV_DIM, "layer-3 k projection")?)
            .map_err(|error| error.to_string())?;
        let v_projection = context
            .new_buffer_checked(bytes_for_f32(KV_DIM, "layer-3 v projection")?)
            .map_err(|error| error.to_string())?;
        let query = context
            .new_buffer_checked(bytes_for_f32(QUERY_DIM, "layer-3 normalized query")?)
            .map_err(|error| error.to_string())?;
        let key_cache = context
            .new_buffer_checked(bytes_for_f32(POSITIONS * KV_DIM, "layer-3 K cache")?)
            .map_err(|error| error.to_string())?;
        let value_cache = context
            .new_buffer_checked(bytes_for_f32(POSITIONS * KV_DIM, "layer-3 V cache")?)
            .map_err(|error| error.to_string())?;
        let attention = context
            .new_buffer_checked(bytes_for_f32(QUERY_DIM, "layer-3 causal GQA output")?)
            .map_err(|error| error.to_string())?;
        let gated = context
            .new_buffer_checked(bytes_for_f32(QUERY_DIM, "layer-3 attention gate")?)
            .map_err(|error| error.to_string())?;
        let output = context
            .new_buffer_checked(bytes_for_f32(HIDDEN, "layer-3 o_proj output")?)
            .map_err(|error| error.to_string())?;

        let inputs = [fixture_hidden(31), fixture_hidden(167)];
        let mut ledger = ErrorLedger::default();
        let mut dispatches = 0usize;
        for (position, fixture) in inputs.iter().enumerate() {
            MetalContext::write_buffer_bytes(&input, bytemuck::cast_slice(fixture));
            let mut token = TokenCommandBuffer::new(&context);
            qwen_binary_sign_scale_matvec_component_tcb(
                &mut token,
                &q.signs,
                &q.scales,
                &input,
                &q_projection,
                Q_PROJ_ROWS,
                HIDDEN,
                GROUP_SIZE,
            )
            .map_err(|error| error.to_string())?;
            qwen_binary_sign_scale_matvec_component_tcb(
                &mut token,
                &k.signs,
                &k.scales,
                &input,
                &k_projection,
                KV_DIM,
                HIDDEN,
                GROUP_SIZE,
            )
            .map_err(|error| error.to_string())?;
            qwen_binary_sign_scale_matvec_component_tcb(
                &mut token,
                &v.signs,
                &v.scales,
                &input,
                &v_projection,
                KV_DIM,
                HIDDEN,
                GROUP_SIZE,
            )
            .map_err(|error| error.to_string())?;
            token
                .dispatch_threads(
                    "qwen80_attention_qk_norm_rope_cache",
                    (QUERY_HEADS as u32, 1, 1),
                    (QUERY_HEADS as u32, 1, 1),
                    |encoder| {
                        encoder.set_buffer(0, Some(&q_projection), 0);
                        encoder.set_buffer(1, Some(&k_projection), 0);
                        encoder.set_buffer(2, Some(&v_projection), 0);
                        encoder.set_buffer(3, Some(&q_norm.signs), 0);
                        encoder.set_buffer(4, Some(&q_norm.scales), 0);
                        encoder.set_buffer(5, Some(&k_norm.signs), 0);
                        encoder.set_buffer(6, Some(&k_norm.scales), 0);
                        encoder.set_buffer(7, Some(&query), 0);
                        encoder.set_buffer(8, Some(&key_cache), 0);
                        encoder.set_buffer(9, Some(&value_cache), 0);
                        encoder.stage_set_u32(10, position as u32);
                        encoder.stage_set_u32(11, QUERY_HEADS as u32);
                        encoder.stage_set_u32(12, KV_HEADS as u32);
                        encoder.stage_set_u32(13, HEAD_DIM as u32);
                        encoder.stage_set_u32(14, ROTARY_DIM as u32);
                        encoder.stage_set_u32(15, GROUP_SIZE as u32);
                        encoder.stage_set_f32(16, ROPE_THETA);
                        encoder.stage_set_f32(17, RMS_EPSILON);
                    },
                )
                .map_err(|error| error.to_string())?;
            mha_decode_f32_tcb(
                &mut token,
                &query,
                &key_cache,
                0,
                &value_cache,
                0,
                &attention,
                position + 1,
                HEAD_DIM,
                QUERY_HEADS,
                KV_HEADS,
            )
            .map_err(|error| error.to_string())?;
            token
                .dispatch_threads(
                    "qwen80_attention_apply_sigmoid_gate",
                    (QUERY_DIM as u32, 1, 1),
                    (256, 1, 1),
                    |encoder| {
                        encoder.set_buffer(0, Some(&attention), 0);
                        encoder.set_buffer(1, Some(&q_projection), 0);
                        encoder.set_buffer(2, Some(&gated), 0);
                        encoder.stage_set_u32(3, QUERY_DIM as u32);
                        encoder.stage_set_u32(4, HEAD_DIM as u32);
                    },
                )
                .map_err(|error| error.to_string())?;
            qwen_binary_sign_scale_matvec_component_tcb(
                &mut token, &o.signs, &o.scales, &gated, &output, HIDDEN, QUERY_DIM, GROUP_SIZE,
            )
            .map_err(|error| error.to_string())?;
            dispatches += token.dispatch_count();
            token.commit_and_wait().map_err(|error| error.to_string())?;

            update_ledger(
                &mut ledger,
                &cpu.tokens[position],
                &snapshot_f32(&q_projection, Q_PROJ_ROWS, "q projection")?,
                &snapshot_f32(&k_projection, KV_DIM, "k projection")?,
                &snapshot_f32(&v_projection, KV_DIM, "v projection")?,
                &snapshot_f32(&query, QUERY_DIM, "normalized query")?,
                &snapshot_f32(&key_cache, POSITIONS * KV_DIM, "key cache")?,
                &snapshot_f32(&value_cache, POSITIONS * KV_DIM, "value cache")?,
                position,
                &snapshot_f32(&attention, QUERY_DIM, "causal GQA")?,
                &snapshot_f32(&gated, QUERY_DIM, "sigmoid gate")?,
                &snapshot_f32(&output, HIDDEN, "o projection")?,
            )?;
        }

        require_parity("layer-3 q projection", ledger.q_projection, 5.0e-4)?;
        require_parity("layer-3 k projection", ledger.k_projection, 5.0e-4)?;
        require_parity("layer-3 v projection", ledger.v_projection, 5.0e-4)?;
        require_parity("layer-3 q_norm + RoPE", ledger.q_norm_rope, 2.0e-4)?;
        require_parity(
            "layer-3 k_norm + RoPE cache append",
            ledger.k_norm_rope_cache,
            2.0e-4,
        )?;
        require_parity("layer-3 V cache append", ledger.value_cache, 5.0e-4)?;
        require_parity("layer-3 causal GQA", ledger.gqa_attention, 3.0e-4)?;
        require_parity(
            "layer-3 attention sigmoid gate",
            ledger.sigmoid_gate,
            2.0e-5,
        )?;
        require_parity("layer-3 o projection", ledger.o_projection, 2.0e-3)?;

        Ok((context.device_name(), ledger, dispatches))
    }

    fn run(arguments: &Arguments) -> Result<Value, String> {
        let admission = CompleteBinaryAdmission {
            model: QwenCompleteBinaryModel::Qwen80CoderNext,
            expected_manifest_seal_sha256: arguments.expected_manifest_seal_sha256.clone(),
            expected_source_audit_seal_sha256: arguments.expected_source_audit_seal_sha256.clone(),
            expected_source_revision: arguments.expected_source_revision.clone(),
        };
        // One protected complete-artifact admission must happen before this
        // component ever opens a compact payload.
        let catalog = Qwen80CompleteArtifactCatalog::load(&arguments.manifest, &admission)
            .map_err(|error| error.to_string())?;
        assert_catalog_geometry(&catalog)?;
        let cpu = run_cpu_stage(&catalog)?;
        let base = json!({
            "schema": RESULT_SCHEMA,
            "artifact": {
                "manifest": catalog.manifest_path(),
                "manifest_seal_sha256": catalog.manifest_seal(),
                "source_revision": catalog.config.source_revision,
                "tensor_count": catalog.tensor_count(),
                "tensor_payload_bytes": catalog.tensor_payload_bytes(),
                "source_weight_elements": catalog.source_weight_elements(),
            },
            "source_bound_layer": {
                "layer": LAYER,
                "kind": "full_attention",
                "tensors": {
                    "q_proj": "model.layers.3.self_attn.q_proj.weight",
                    "k_proj": "model.layers.3.self_attn.k_proj.weight",
                    "v_proj": "model.layers.3.self_attn.v_proj.weight",
                    "o_proj": "model.layers.3.self_attn.o_proj.weight",
                    "q_norm": "model.layers.3.self_attn.q_norm.weight",
                    "k_norm": "model.layers.3.self_attn.k_norm.weight",
                },
                "geometry": {
                    "hidden": HIDDEN,
                    "query_heads": QUERY_HEADS,
                    "kv_heads": KV_HEADS,
                    "gqa_queries_per_kv_head": QUERY_HEADS / KV_HEADS,
                    "head_dim": HEAD_DIM,
                    "q_proj_shape": [Q_PROJ_ROWS, HIDDEN],
                    "k_proj_shape": [KV_DIM, HIDDEN],
                    "v_proj_shape": [KV_DIM, HIDDEN],
                    "o_proj_shape": [HIDDEN, QUERY_DIM],
                    "q_proj_source_layout": "[16 heads][Q(256), gate(256)]",
                    "qk_norm_shape": [HEAD_DIM],
                    "partial_rotary_dimensions": ROTARY_DIM,
                    "rope_theta": ROPE_THETA,
                    "rms_norm_epsilon": RMS_EPSILON,
                    "cache_layout": "[sequence][2 KV heads][256] f32",
                    "fixture_positions": [0, 1],
                },
            },
            "cpu_oracle": {
                "uses_only_admitted_compact_sign_and_fp16_group_scale_payload_decode": true,
                "opens_no_raw_bf16_or_mps_shadow_model": true,
                "two_token_causal_cache_exercised": true,
                "source_q_norm_is_residual_scale_one_plus_weight": true,
                "source_q_proj_gate_is_sigmoid_of_each_head_local_second_256_rows": true,
            },
            "claim_boundary": {
                "component_stage_only": true,
                "does_not_execute_input_layernorm_residual_post_attention_norm_or_moe": true,
                "does_not_execute_complete_layer_or_48_layer_decoder": true,
                "does_not_generate_tokens_expose_hcli_or_measure_tps": true,
                "does_not_claim_100_tps_tg3_or_tournament_qualification": true,
            },
        });
        let result = match arguments.mode {
            Mode::CpuOracle => {
                let mut object = base
                    .as_object()
                    .cloned()
                    .ok_or("base report must be an object")?;
                object.insert("status".into(), json!(CPU_STATUS));
                object.insert(
                    "metal_execution".into(),
                    json!({
                        "performed": false,
                        "reason": "explicit quiet GPU lease required; --mode metal is intentionally opt-in",
                    }),
                );
                serde_json::Value::Object(object)
            }
            Mode::Metal => {
                let (device, ledger, dispatches) = run_metal_stage(&catalog, &cpu)?;
                let mut object = base
                    .as_object()
                    .cloned()
                    .ok_or("base report must be an object")?;
                object.insert("status".into(), json!(METAL_STATUS));
                object.insert(
                    "metal_execution".into(),
                    json!({
                        "performed": true,
                        "device": device,
                        "two_token_command_buffers": POSITIONS,
                        "compute_dispatches": dispatches,
                        "direct_packed_projection_kernels": ["q_proj", "k_proj", "v_proj", "o_proj"],
                        "native_attention_kernels": [
                            "qwen80_attention_qk_norm_rope_cache",
                            "mha_decode_f32",
                            "qwen80_attention_apply_sigmoid_gate",
                        ],
                        "max_abs_error": {
                            "q_projection": ledger.q_projection,
                            "k_projection": ledger.k_projection,
                            "v_projection": ledger.v_projection,
                            "q_norm_rope": ledger.q_norm_rope,
                            "k_norm_rope_cache": ledger.k_norm_rope_cache,
                            "value_cache": ledger.value_cache,
                            "gqa_attention": ledger.gqa_attention,
                            "sigmoid_gate": ledger.sigmoid_gate,
                            "o_projection": ledger.o_projection,
                        },
                    }),
                );
                serde_json::Value::Object(object)
            }
        };
        Ok(result)
    }

    pub fn main() -> Result<(), Box<dyn std::error::Error>> {
        let arguments = parse_arguments().map_err(std::io::Error::other)?;
        begin_capture(&arguments).map_err(std::io::Error::other)?;
        let (result, failure) =
            finalize_capture(&arguments, run(&arguments)).map_err(std::io::Error::other)?;
        println!(
            "{}",
            serde_json::to_string(&result).map_err(std::io::Error::other)?
        );
        if let Some(error) = failure {
            return Err(std::io::Error::other(error).into());
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::main()
}

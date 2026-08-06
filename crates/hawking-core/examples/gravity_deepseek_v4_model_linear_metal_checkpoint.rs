//! GPU source-linear parity checkpoint for one DeepSeek-V4-Flash component.
//!
//! This is deliberately narrower than a model runtime.  It runs exactly one
//! sealed source tensor pair, `layers.0.attn.wq_a`, with one deterministic
//! BF16 row through the pinned source algorithm shape:
//!
//! ```text
//! BF16 [4096]
//!   -> GPU act_quant(block=128, UE8M0) -> E4M3FN [4096] + E8M0FNU [32]
//!   -> GPU FP8 weighted projection using activation *and* source weight scales
//!   -> FP32 [1024]
//! ```
//!
//! The completed receipt binds the immutable full-stream artifact, exact
//! source-code hashes, source chunk digests, and the canonical CPU oracle v2.
//! It is a **model.linear component parity checkpoint only**: it does not
//! execute embedding, mHC, attention, a token loop, a full model, HCLI, or a
//! TPS benchmark.
//!
//! ```sh
//! cargo run --release -p hawking-core --example gravity_deepseek_v4_model_linear_metal_checkpoint -- \
//!   --artifact /absolute/path/to/full-43-layer-stream.gravity \
//!   --cpu-oracle /absolute/path/to/DSV4F_ACT_QUANT_WQ_A_CPU_ORACLE-v2.json \
//!   --out /absolute/path/to/DSV4F_MODEL_LINEAR_METAL_COMPONENT_PARITY.json
//! ```

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "gravity_deepseek_v4_model_linear_metal_checkpoint requires macOS Metal",
    )
    .into())
}

#[cfg(target_os = "macos")]
pub mod macos {
    use half::bf16;
    use hawking_core::gravity_deepseek_v4::{
        DeepSeekV4FullStreamReader, DeepSeekV4TensorMetadata, NativeScalePairKind,
        FULL_STREAM_SCHEMA, FULL_STREAM_STATUS,
    };
    use hawking_core::gravity_deepseek_v4_act_quant::{
        deterministic_wq_a_input_bf16, layer0_wq_a_cpu_oracle, verify_source_algorithm_anchors,
        Layer0WqACpuOracleResult, ACT_QUANT_BLOCK, LAYER0_WQ_A_COLS, LAYER0_WQ_A_ROWS,
        LAYER0_WQ_A_SCALE, LAYER0_WQ_A_WEIGHT,
    };
    use hawking_core::metal::{
        MetalBatchTiming, MetalContext, MetalDispatchTiming, PhysicalTraceGuard,
        PhysicalTraceIdentity,
    };
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::error::Error;
    use std::fs::{self, File, OpenOptions};
    use std::io::Write;
    use std::path::{Path, PathBuf};

    const RECEIPT_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.model_linear_fp8_act_quant_metal_component_parity.v1";
    const RECEIPT_STATUS: &str = "PASS_REAL_METAL_MODEL_LINEAR_COMPONENT_PARITY_NOT_FULL_RUNTIME";
    const CPU_ORACLE_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.act_quant_fp8_wq_a_cpu_algorithm_oracle.v1";
    const CPU_ORACLE_STATUS: &str =
        "PASS_SOURCE_DERIVED_CPU_ALGORITHM_ORACLE_NOT_INDEPENDENT_SOURCE_RUNTIME_PARITY";
    const CPU_ORACLE_V2_BASENAME: &str = "DSV4F_ACT_QUANT_WQ_A_CPU_ORACLE-v2.json";
    const ACT_QUANT_KERNEL: &str = "deepseek_v4_act_quant_bf16_ue8m0_authority";
    const AUTHORITY_KERNEL: &str = "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority";
    const SIMD_CANDIDATE_KERNEL: &str =
        "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_simdgroup_v4_splitk_candidate";
    const QAT_SIMDGROUP_KERNEL: &str = "deepseek_v4_act_quant_bf16_ue8m0_simdgroup_block_candidate";
    const QAT_WINNER_THREADS: u32 = 32;
    const QAT_WINNER_VECTOR_WIDTH: u32 = 2;
    const QAT_V2_WARMUPS: usize = 8;
    const QAT_V2_TRIALS: usize = 25;
    const OUTPUT_ABS_TOLERANCE: f32 = 1.0e-4;
    const OUTPUT_REL_TOLERANCE: f32 = 1.0e-4;

    type ProbeResult<T> = Result<T, Box<dyn Error>>;

    struct Args {
        artifact: PathBuf,
        cpu_oracle: PathBuf,
        out: PathBuf,
    }

    struct CpuOracleBinding {
        path: PathBuf,
        file_sha256: String,
        seal_sha256: String,
        input_sha256_bf16_le: String,
        activation_sha256: String,
        scale_sha256: String,
        output_fp32_sha256: String,
        output_bf16_sha256: String,
    }

    fn failure(message: impl Into<String>) -> Box<dyn Error> {
        std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
    }

    pub fn run() -> ProbeResult<()> {
        let args = parse_args()?;
        let reader = DeepSeekV4FullStreamReader::admit(&args.artifact)?;
        let anchors = verify_source_algorithm_anchors(&reader)?;
        if reader.source_identity().repository != "deepseek-ai/DeepSeek-V4-Flash"
            || reader.source_identity().revision != "60d8d70770c6776ff598c94bb586a859a38244f1"
        {
            return Err(failure(
                "model-linear checkpoint reader did not admit the pinned DeepSeek-V4 source",
            ));
        }

        let (weight_meta, scale_meta) = {
            let pair = reader.native_scale_pair(LAYER0_WQ_A_WEIGHT)?;
            if pair.kind != NativeScalePairKind::Fp8E4M3fn
                || pair.weight.name != LAYER0_WQ_A_WEIGHT
                || pair.scale.name != LAYER0_WQ_A_SCALE
                || pair.weight.shape.as_slice()
                    != [LAYER0_WQ_A_ROWS as u64, LAYER0_WQ_A_COLS as u64]
                || pair.scale.shape.as_slice()
                    != [
                        (LAYER0_WQ_A_ROWS / ACT_QUANT_BLOCK) as u64,
                        (LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK) as u64,
                    ]
                || pair.logical_k != LAYER0_WQ_A_COLS as u64
                || pair.out_rows != LAYER0_WQ_A_ROWS as u64
            {
                return Err(failure(
                    "layer-0 WQ-A changed from the pinned source-linear FP8 geometry",
                ));
            }
            (pair.weight.clone(), pair.scale.clone())
        };

        let input_bf16 = deterministic_wq_a_input_bf16();
        if input_bf16.len() != LAYER0_WQ_A_COLS {
            return Err(failure(
                "deterministic BF16 input does not match WQ-A source K",
            ));
        }
        let input_bf16_le = u16_le_bytes(&input_bf16);
        let cpu = layer0_wq_a_cpu_oracle(&reader, &input_bf16)?;
        let cpu_oracle = validate_cpu_oracle_receipt(&args.cpu_oracle, &reader, &input_bf16, &cpu)?;

        // Re-read through the bounded reader for GPU upload.  Each call hashes
        // every touched content-addressed source chunk before returning bytes;
        // this never reconstructs a safetensors parent source file.
        let weights =
            reader.read_verified_full(LAYER0_WQ_A_WEIGHT, LAYER0_WQ_A_ROWS * LAYER0_WQ_A_COLS)?;
        let weight_scales = reader.read_verified_full(
            LAYER0_WQ_A_SCALE,
            (LAYER0_WQ_A_ROWS / ACT_QUANT_BLOCK) * (LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK),
        )?;
        if weights.len() != LAYER0_WQ_A_ROWS * LAYER0_WQ_A_COLS
            || weight_scales.len()
                != (LAYER0_WQ_A_ROWS / ACT_QUANT_BLOCK) * (LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK)
        {
            return Err(failure(
                "bounded source reads returned an unexpected WQ-A FP8 payload size",
            ));
        }

        let context = MetalContext::new_with_trace(true)?;
        let device_name = context.device_name();
        // Resolve all pipelines before the measured dispatches.  The receipt
        // therefore separates lazy compilation from completed GPU timestamps.
        let act_quant_pipeline = context.pipeline(ACT_QUANT_KERNEL)?;
        let authority_pipeline = context.pipeline(AUTHORITY_KERNEL)?;
        let candidate_pipeline = context.pipeline(SIMD_CANDIDATE_KERNEL)?;
        let act_quant_pipeline_thread_execution_width =
            act_quant_pipeline.thread_execution_width() as u64;
        let authority_pipeline_thread_execution_width =
            authority_pipeline.thread_execution_width() as u64;
        let candidate_pipeline_thread_execution_width =
            candidate_pipeline.thread_execution_width() as u64;
        let candidate_pipeline_max_total_threads =
            candidate_pipeline.max_total_threads_per_threadgroup() as u32;
        drop(act_quant_pipeline);
        drop(authority_pipeline);
        drop(candidate_pipeline);

        let candidate_threads = (candidate_pipeline_max_total_threads.min(256) / 32) * 32;
        if candidate_threads < 32 {
            return Err(failure(
                "Metal SIMDgroup candidate pipeline cannot admit one 32-lane SIMDgroup",
            ));
        }

        let input_buffer = context.new_buffer_with_bytes_checked(&input_bf16_le)?;
        let weight_buffer = context.new_buffer_with_bytes_checked(&weights)?;
        let weight_scale_buffer = context.new_buffer_with_bytes_checked(&weight_scales)?;
        let activation_buffer = context.new_buffer_checked(LAYER0_WQ_A_COLS)?;
        let activation_scale_buffer =
            context.new_buffer_checked(LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK)?;
        let authority_output_buffer =
            context.new_buffer_checked(LAYER0_WQ_A_ROWS * std::mem::size_of::<f32>())?;
        let candidate_output_buffer =
            context.new_buffer_checked(LAYER0_WQ_A_ROWS * std::mem::size_of::<f32>())?;

        let rows = LAYER0_WQ_A_ROWS as u32;
        let cols = LAYER0_WQ_A_COLS as u32;
        let scale_cols = (LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK) as u32;
        let run_nonce = sha256_join(&[
            reader.manifest_seal_sha256(),
            &weight_meta.segments[0].sha256,
            &scale_meta.segments[0].sha256,
            &cpu_oracle.seal_sha256,
            &cpu_oracle.input_sha256_bf16_le,
            "model_linear_gpu_checkpoint_v1",
        ]);
        let interval_id = sha256_join(&[
            &run_nonce,
            ACT_QUANT_KERNEL,
            AUTHORITY_KERNEL,
            SIMD_CANDIDATE_KERNEL,
        ]);
        let trace_identity = PhysicalTraceIdentity::new(
            interval_id.clone(),
            run_nonce.clone(),
            "model_linear_component_parity".to_owned(),
            "act_quant_then_fp8_projection".to_owned(),
            Some(1),
            0,
        )?;
        let physical_trace = PhysicalTraceGuard::begin(trace_identity)?;

        let act_quant_timing = context.dispatch_threads_timed(
            ACT_QUANT_KERNEL,
            (scale_cols, 1, 1),
            (32, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&input_buffer), 0);
                encoder.set_buffer(1, Some(&activation_buffer), 0);
                encoder.set_buffer(2, Some(&activation_scale_buffer), 0);
                set_u32(encoder, 3, &cols);
            },
        )?;
        require_completed_dispatch(&act_quant_timing, "GPU act_quant")?;
        let gpu_activation = read_gpu_bytes(&activation_buffer, LAYER0_WQ_A_COLS)?;
        let gpu_activation_scales =
            read_gpu_bytes(&activation_scale_buffer, LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK)?;
        if gpu_activation != cpu.quantized_input.activation_e4m3fn {
            return Err(failure(
                "GPU act_quant E4M3FN bytes differ from the canonical CPU oracle; no receipt emitted",
            ));
        }
        if gpu_activation_scales != cpu.quantized_input.scales_e8m0fnu {
            return Err(failure(
                "GPU act_quant UE8M0 scale bytes differ from the canonical CPU oracle; no receipt emitted",
            ));
        }
        if sha256(&gpu_activation) != cpu_oracle.activation_sha256
            || sha256(&gpu_activation_scales) != cpu_oracle.scale_sha256
        {
            return Err(failure(
                "GPU act_quant byte hashes differ from the sealed CPU oracle v2 binding",
            ));
        }

        let authority_timing = context.dispatch_threads_timed(
            AUTHORITY_KERNEL,
            (rows, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&weight_buffer), 0);
                encoder.set_buffer(1, Some(&weight_scale_buffer), 0);
                encoder.set_buffer(2, Some(&activation_buffer), 0);
                encoder.set_buffer(3, Some(&activation_scale_buffer), 0);
                encoder.set_buffer(4, Some(&authority_output_buffer), 0);
                set_u32(encoder, 5, &rows);
                set_u32(encoder, 6, &cols);
                set_u32(encoder, 7, &scale_cols);
            },
        )?;
        require_completed_dispatch(&authority_timing, "GPU source-linear authority projection")?;
        let authority_output = read_gpu_f32(&authority_output_buffer, LAYER0_WQ_A_ROWS)?;
        let authority_parity = f32_parity(&cpu.output.fp32, &authority_output)?;
        if authority_parity.get("status").and_then(Value::as_str) != Some("PASS") {
            return Err(failure(
                "GPU source-linear authority output failed canonical CPU oracle parity; no receipt emitted",
            ));
        }

        // The optional kernel gets the exact device-generated activation and
        // scale buffers.  It is selected only if it clears the same CPU-oracle
        // comparison; otherwise there is no host/CPU fallback, only the
        // independently recorded serial GPU authority result.
        let candidate_timing = context.dispatch_threads_timed(
            SIMD_CANDIDATE_KERNEL,
            (candidate_threads, rows, 1),
            (candidate_threads, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&weight_buffer), 0);
                encoder.set_buffer(1, Some(&weight_scale_buffer), 0);
                encoder.set_buffer(2, Some(&activation_buffer), 0);
                encoder.set_buffer(3, Some(&activation_scale_buffer), 0);
                encoder.set_buffer(4, Some(&candidate_output_buffer), 0);
                set_u32(encoder, 5, &rows);
                set_u32(encoder, 6, &cols);
                set_u32(encoder, 7, &scale_cols);
                set_u32(encoder, 8, &candidate_threads);
            },
        )?;
        require_completed_dispatch(&candidate_timing, "GPU source-linear SIMDgroup candidate")?;
        let candidate_output = read_gpu_f32(&candidate_output_buffer, LAYER0_WQ_A_ROWS)?;
        let candidate_parity = f32_parity(&cpu.output.fp32, &candidate_output)?;
        let candidate_passed =
            candidate_parity.get("status").and_then(Value::as_str) == Some("PASS");
        let (selected_kernel, selected_output, selected_parity, candidate_selection_reason) =
            if candidate_passed {
                (
                    SIMD_CANDIDATE_KERNEL,
                    &candidate_output,
                    &candidate_parity,
                    "optional SIMDgroup candidate passed the same canonical CPU-oracle threshold and is preferred for this bounded component checkpoint",
                )
            } else {
                (
                    AUTHORITY_KERNEL,
                    &authority_output,
                    &authority_parity,
                    "optional SIMDgroup candidate did not pass the declared canonical CPU-oracle threshold; serial GPU authority remains selected without CPU fallback",
                )
            };

        let physical_counts = physical_trace.counts();
        drop(physical_trace);
        let (buffers_created, bytes_allocated, commits) = context.drain_stats();
        let trace_samples = context.drain_trace();
        if physical_counts.command_count != 3
            || physical_counts.encoder_count != 3
            || commits != 3
            || trace_samples.len() != 3
        {
            return Err(failure(
                "model-linear checkpoint command/encoder/trace accounting did not match three real GPU stages",
            ));
        }
        let expected_dispatches = act_quant_timing.compute_dispatches
            + authority_timing.compute_dispatches
            + candidate_timing.compute_dispatches;
        if expected_dispatches != 3 {
            return Err(failure(
                "model-linear checkpoint did not execute three real GPU dispatches",
            ));
        }

        let selected_bf16 = selected_output
            .iter()
            .copied()
            .map(|value| bf16::from_f32(value).to_bits())
            .collect::<Vec<_>>();
        let selected_bf16_sha256 = sha256(&u16_le_bytes(&selected_bf16));
        let selected_bf16_hash_match = selected_bf16_sha256 == cpu_oracle.output_bf16_sha256;
        let unsigned = json!({
            "schema": RECEIPT_SCHEMA,
            "status": RECEIPT_STATUS,
            "scope": {
                "component": "model.linear only: layers.0.attn.wq_a",
                "model_linear_component_only": true,
                "not_embedding": true,
                "not_mhc": true,
                "not_attention": true,
                "not_router_or_expert_execution": true,
                "not_full_model_load": true,
                "not_full_model_forward": true,
                "not_token_execution_or_generation": true,
                "not_hcli_endpoint": true,
                "not_base_true_tps_measurement": true,
                "not_registered_43_layer_runtime_adapter": true,
            },
            "artifact": {
                "path": reader.artifact_root().display().to_string(),
                "full_stream_schema": FULL_STREAM_SCHEMA,
                "full_stream_status": FULL_STREAM_STATUS,
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
                "source_tensor_chunk_bindings": {
                    "weight": tensor_binding_json(&weight_meta),
                    "weight_scale": tensor_binding_json(&scale_meta),
                    "touched_chunks_sha256_verified_before_gpu_upload": true,
                    "parent_safetensors_materialized": false,
                },
            },
            "canonical_cpu_oracle_v2": {
                "path": cpu_oracle.path.display().to_string(),
                "file_sha256": cpu_oracle.file_sha256,
                "receipt_seal_sha256": cpu_oracle.seal_sha256,
                "receipt_seal_verified": true,
                "input_sha256_bf16_le": cpu_oracle.input_sha256_bf16_le,
                "activation_sha256": cpu_oracle.activation_sha256,
                "activation_scale_sha256": cpu_oracle.scale_sha256,
                "output_fp32_sha256_le": cpu_oracle.output_fp32_sha256,
                "output_bf16_sha256_le": cpu_oracle.output_bf16_sha256,
                "direct_cpu_oracle_recomputed_and_matches_sealed_v2": true,
            },
            "input": {
                "kind": "deterministic_exact_bf16_bitpattern_vector_v1",
                "captured_from_model_forward": false,
                "dtype": "BF16",
                "length": input_bf16.len(),
                "sha256_bf16_le": sha256(&input_bf16_le),
            },
            "gpu_act_quant": {
                "kernel": ACT_QUANT_KERNEL,
                "source_algorithm": "act_quant(block_size=128, scale_fmt=ue8m0, E4M3FN cast with finite clamp[-448,448])",
                "input_dtype": "BF16",
                "activation_dtype": "F8_E4M3FN",
                "activation_bytes": gpu_activation.len(),
                "activation_sha256": sha256(&gpu_activation),
                "activation_bytewise_cpu_oracle_match": true,
                "scale_dtype": "F8_E8M0FNU",
                "scale_bytes": gpu_activation_scales.len(),
                "scale_sha256": sha256(&gpu_activation_scales),
                "scale_bytewise_cpu_oracle_match": true,
                "dispatch": timing_json(&act_quant_timing),
                "fallback": false,
                "fallback_reason": Value::Null,
            },
            "gpu_fp8_weighted_projection": {
                "source_algorithm": "FP8 block GEMV: E4M3FN activation×weight dot accumulated per K=128 block, then activation E8M0FNU×weight E8M0FNU scaling",
                "rows": LAYER0_WQ_A_ROWS,
                "cols": LAYER0_WQ_A_COLS,
                "weight_scale_shape": [LAYER0_WQ_A_ROWS / ACT_QUANT_BLOCK, LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK],
                "authority": {
                    "kernel": AUTHORITY_KERNEL,
                    "dispatch": timing_json(&authority_timing),
                    "output_fp32_sha256_le": sha256(&f32_le_bytes(&authority_output)),
                    "cpu_oracle_parity": authority_parity,
                    "output_bf16_sha256_le": sha256(&u16_le_bytes(&authority_output.iter().copied().map(|value| bf16::from_f32(value).to_bits()).collect::<Vec<_>>())),
                },
                "optional_simdgroup_candidate": {
                    "kernel": SIMD_CANDIDATE_KERNEL,
                    "dispatch_geometry": { "grid": [candidate_threads, rows, 1], "threadgroup": [candidate_threads, 1, 1] },
                    "pipeline_thread_execution_width": candidate_pipeline_thread_execution_width,
                    "dispatch": timing_json(&candidate_timing),
                    "output_fp32_sha256_le": sha256(&f32_le_bytes(&candidate_output)),
                    "cpu_oracle_parity": candidate_parity,
                    "passes_same_parity_gate": candidate_passed,
                },
                "selected_kernel": selected_kernel,
                "selection_reason": candidate_selection_reason,
                "selected_output_fp32_sha256_le": sha256(&f32_le_bytes(selected_output)),
                "selected_cpu_oracle_parity": selected_parity,
                "selected_output_bf16_sha256_le": selected_bf16_sha256,
                "selected_output_bf16_hash_matches_cpu_oracle": selected_bf16_hash_match,
                "fallback": false,
                "fallback_reason": Value::Null,
            },
            "metal": {
                "device": device_name,
                "pipelines_precompiled_before_measured_dispatches": true,
                "act_quant_pipeline_thread_execution_width": act_quant_pipeline_thread_execution_width,
                "authority_pipeline_thread_execution_width": authority_pipeline_thread_execution_width,
                "candidate_pipeline_thread_execution_width": candidate_pipeline_thread_execution_width,
                "candidate_pipeline_max_total_threads_per_threadgroup": candidate_pipeline_max_total_threads,
                "buffers_created": buffers_created,
                "bytes_allocated": bytes_allocated,
                "gpu_dispatches": expected_dispatches,
                "command_buffers": 3,
                "compute_encoders": 3,
                "cpu_visible_waits": 3,
                "empty_command_buffers": 0,
                "physical_trace_command_buffers": physical_counts.command_count,
                "physical_trace_compute_encoders": physical_counts.encoder_count,
                "trace_samples": trace_samples.len(),
                "fallback": false,
                "fallback_count": 0,
                "cpu_used_only_for_source_oracle_and_post_dispatch_comparison": true,
            },
            "logical_bytes": {
                "act_quant_read_bf16": input_bf16_le.len(),
                "act_quant_written_e4m3fn_plus_e8m0": gpu_activation.len() + gpu_activation_scales.len(),
                "each_projection_read_weight_plus_weight_scale_plus_activation_plus_activation_scale": weights.len() + weight_scales.len() + gpu_activation.len() + gpu_activation_scales.len(),
                "each_projection_written_fp32": LAYER0_WQ_A_ROWS * std::mem::size_of::<f32>(),
            },
            "physical_trace": {
                "interval_id": interval_id,
                "run_nonce": run_nonce,
                "phase": "model_linear_component_parity",
                "role": "act_quant_then_fp8_projection",
            },
            "claim_boundary": "This is a real GPU source-linear component parity checkpoint only. It does NOT establish embedding, mHC, attention, routing, expert execution, a full V4 runtime, token generation, HCLI behavior, or BASE_TRUE_TPS.",
        });
        let (receipt, seal) = seal(unsigned)?;
        write_new_receipt(&args.out, &receipt)?;
        println!(
            "{}",
            serde_json::to_string(&json!({
                "status": RECEIPT_STATUS,
                "receipt": args.out,
                "seal_sha256": seal,
                "selected_kernel": selected_kernel,
                "gpu_dispatches": expected_dispatches,
                "candidate_passed": candidate_passed,
            }))?
        );
        Ok(())
    }

    struct QatV2Args {
        artifact: PathBuf,
        cpu_oracle: PathBuf,
        predecessor_v1: PathBuf,
        out: PathBuf,
        warmups: usize,
        trials: usize,
    }

    struct PredecessorV1Binding {
        path: PathBuf,
        file_sha256: String,
        seal_sha256: String,
        act_quant_gpu_duration_us: u64,
    }

    #[derive(Default)]
    struct TimingSeries {
        gpu_us: Vec<u64>,
        encode_us: Vec<u64>,
        submit_us: Vec<u64>,
        wait_us: Vec<u64>,
        host_wall_us: Vec<u64>,
        gpu_intervals_ns: Vec<[u64; 2]>,
    }

    impl TimingSeries {
        fn with_capacity(count: usize) -> Self {
            Self {
                gpu_us: Vec::with_capacity(count),
                encode_us: Vec::with_capacity(count),
                submit_us: Vec::with_capacity(count),
                wait_us: Vec::with_capacity(count),
                host_wall_us: Vec::with_capacity(count),
                gpu_intervals_ns: Vec::with_capacity(count),
            }
        }

        fn record_fields(
            &mut self,
            gpu_duration_us: Option<u64>,
            gpu_start_ns: Option<u64>,
            gpu_end_ns: Option<u64>,
            encode_us: u64,
            submit_us: u64,
            wait_us: u64,
            host_wall_us: u64,
        ) -> ProbeResult<()> {
            let gpu_duration_us = gpu_duration_us.filter(|value| *value > 0).ok_or_else(|| {
                failure("timed model.linear chain dispatch has no positive GPU timestamp")
            })?;
            let gpu_start_ns = gpu_start_ns
                .ok_or_else(|| failure("timed model.linear chain dispatch has no GPU start"))?;
            let gpu_end_ns = gpu_end_ns
                .ok_or_else(|| failure("timed model.linear chain dispatch has no GPU end"))?;
            if gpu_end_ns <= gpu_start_ns {
                return Err(failure(
                    "timed model.linear chain dispatch has a non-positive GPU interval",
                ));
            }
            self.gpu_us.push(gpu_duration_us);
            self.encode_us.push(encode_us);
            self.submit_us.push(submit_us);
            self.wait_us.push(wait_us);
            self.host_wall_us.push(host_wall_us);
            self.gpu_intervals_ns.push([gpu_start_ns, gpu_end_ns]);
            Ok(())
        }

        fn record_dispatch(&mut self, timing: &MetalDispatchTiming) -> ProbeResult<()> {
            self.record_fields(
                timing.gpu_duration_us,
                timing.gpu_start_ns,
                timing.gpu_end_ns,
                timing.encode_us,
                timing.submit_us,
                timing.wait_us,
                timing.host_wall_us,
            )
        }

        fn record_batch(&mut self, timing: &MetalBatchTiming) -> ProbeResult<()> {
            self.record_fields(
                timing.gpu_duration_us,
                timing.gpu_start_ns,
                timing.gpu_end_ns,
                timing.encode_us,
                timing.submit_us,
                timing.wait_us,
                timing.host_wall_us,
            )
        }
    }

    fn parse_qat_v2_positive(value: Option<String>, flag: &str) -> ProbeResult<usize> {
        let value = value.ok_or_else(|| failure(format!("{flag} needs a value")))?;
        let parsed = value
            .parse::<usize>()
            .map_err(|_| failure(format!("{flag} must be a positive integer")))?;
        if parsed == 0 {
            return Err(failure(format!("{flag} must be positive")));
        }
        Ok(parsed)
    }

    fn parse_qat_v2_args() -> ProbeResult<QatV2Args> {
        let mut artifact = None::<PathBuf>;
        let mut cpu_oracle = None::<PathBuf>;
        let mut predecessor_v1 = None::<PathBuf>;
        let mut out = None::<PathBuf>;
        let mut warmups = QAT_V2_WARMUPS;
        let mut trials = QAT_V2_TRIALS;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--artifact" => artifact = args.next().map(PathBuf::from),
                "--cpu-oracle" => cpu_oracle = args.next().map(PathBuf::from),
                "--predecessor-v1" => predecessor_v1 = args.next().map(PathBuf::from),
                "--out" => out = args.next().map(PathBuf::from),
                "--warmups" => warmups = parse_qat_v2_positive(args.next(), "--warmups")?,
                "--trials" => trials = parse_qat_v2_positive(args.next(), "--trials")?,
                "--help" | "-h" => {
                    println!(
                        "usage: gravity_deepseek_v4_model_linear_qat_v2 --artifact <absolute full Gravity dir> --cpu-oracle <absolute DSV4F_ACT_QUANT_WQ_A_CPU_ORACLE-v2.json> --predecessor-v1 <absolute DSV4F_MODEL_LINEAR_METAL_COMPONENT_PARITY-v1.json> --out <absolute DSV4F_MODEL_LINEAR_METAL_COMPONENT_PARITY-v2.json> [--warmups N] [--trials N]"
                    );
                    std::process::exit(0);
                }
                other => return Err(failure(format!("unknown argument {other:?}"))),
            }
        }
        let artifact = artifact.ok_or_else(|| failure("--artifact is required"))?;
        let cpu_oracle = cpu_oracle.ok_or_else(|| failure("--cpu-oracle is required"))?;
        let predecessor_v1 =
            predecessor_v1.ok_or_else(|| failure("--predecessor-v1 is required"))?;
        let out = out.ok_or_else(|| failure("--out is required"))?;
        if !artifact.is_absolute()
            || !cpu_oracle.is_absolute()
            || !predecessor_v1.is_absolute()
            || !out.is_absolute()
        {
            return Err(failure("all QAT v2 paths must be absolute"));
        }
        Ok(QatV2Args {
            artifact,
            cpu_oracle,
            predecessor_v1,
            out,
            warmups,
            trials,
        })
    }

    fn bool_at(value: &Value, path: &[&str]) -> ProbeResult<bool> {
        value_at(value, path)?.as_bool().ok_or_else(|| {
            failure(format!(
                "receipt JSON path {} is not a boolean",
                path.join(".")
            ))
        })
    }

    fn validate_predecessor_v1_receipt(
        path: &Path,
        reader: &DeepSeekV4FullStreamReader,
        cpu_oracle: &CpuOracleBinding,
    ) -> ProbeResult<PredecessorV1Binding> {
        if path.file_name().and_then(|name| name.to_str())
            != Some("DSV4F_MODEL_LINEAR_METAL_COMPONENT_PARITY-v1.json")
        {
            return Err(failure(
                "--predecessor-v1 must be the immutable DSV4F_MODEL_LINEAR_METAL_COMPONENT_PARITY-v1.json receipt",
            ));
        }
        let meta = fs::symlink_metadata(path)?;
        if meta.file_type().is_symlink() || !meta.file_type().is_file() {
            return Err(failure(
                "--predecessor-v1 must be a regular non-symlink file",
            ));
        }
        let canonical_path = fs::canonicalize(path)?;
        let raw = fs::read(&canonical_path)?;
        let file_sha256 = sha256(&raw);
        let mut value: Value = serde_json::from_slice(&raw)
            .map_err(|error| failure(format!("predecessor v1 receipt is not JSON: {error}")))?;
        let seal_sha256 = value
            .as_object_mut()
            .ok_or_else(|| failure("predecessor v1 receipt root is not an object"))?
            .remove("seal_sha256")
            .and_then(|value| value.as_str().map(str::to_owned))
            .ok_or_else(|| failure("predecessor v1 receipt lacks seal_sha256"))?;
        if !is_sha256(&seal_sha256) || sha256(&canonical_json(&value)) != seal_sha256 {
            return Err(failure(
                "predecessor v1 receipt canonical seal does not verify",
            ));
        }
        value
            .as_object_mut()
            .expect("object was checked")
            .insert("seal_sha256".to_owned(), Value::String(seal_sha256.clone()));
        if text_at(&value, &["schema"])?
            != "hawking.gravity.deepseek_v4.model_linear_fp8_act_quant_metal_component_parity.v1"
            || text_at(&value, &["status"])?
                != "PASS_REAL_METAL_MODEL_LINEAR_COMPONENT_PARITY_NOT_FULL_RUNTIME"
            || text_at(&value, &["artifact", "manifest_seal_sha256"])?
                != reader.manifest_seal_sha256()
            || text_at(&value, &["artifact", "manifest_file_sha256"])?
                != reader.manifest_file_sha256()
            || text_at(&value, &["canonical_cpu_oracle_v2", "receipt_seal_sha256"])?
                != cpu_oracle.seal_sha256
            || text_at(&value, &["gpu_act_quant", "kernel"])? != ACT_QUANT_KERNEL
            || !bool_at(
                &value,
                &["gpu_act_quant", "activation_bytewise_cpu_oracle_match"],
            )?
            || !bool_at(
                &value,
                &["gpu_act_quant", "scale_bytewise_cpu_oracle_match"],
            )?
            || bool_at(&value, &["gpu_act_quant", "fallback"])?
        {
            return Err(failure(
                "predecessor v1 does not bind the same byte-exact source-linear component authority",
            ));
        }
        let act_quant_gpu_duration_us =
            number_at(&value, &["gpu_act_quant", "dispatch", "gpu_duration_us"])?;
        if act_quant_gpu_duration_us != 5_967 {
            return Err(failure(format!(
                "predecessor v1 authority stage must be 5967us, observed {act_quant_gpu_duration_us}us"
            )));
        }
        Ok(PredecessorV1Binding {
            path: canonical_path,
            file_sha256,
            seal_sha256,
            act_quant_gpu_duration_us,
        })
    }

    fn require_completed_batch(timing: &MetalBatchTiming, stage: &str) -> ProbeResult<()> {
        if timing.command_buffers != 1
            || timing.compute_encoders != 1
            || timing.compute_dispatches != 2
            || timing.gpu_duration_us.unwrap_or(0) == 0
            || timing.gpu_start_ns.is_none()
            || timing.gpu_end_ns.is_none()
        {
            return Err(failure(format!(
                "{stage} did not complete as one GPU-timestamped command buffer with one ordered compute encoder/two dispatches",
            )));
        }
        Ok(())
    }

    fn summary_json(values: &[u64]) -> ProbeResult<Value> {
        if values.is_empty() {
            return Err(failure("timing summary needs at least one measured sample"));
        }
        let mut ordered = values.to_vec();
        ordered.sort_unstable();
        let percentile = |percent: usize| {
            let index = (ordered.len() * percent).div_ceil(100).saturating_sub(1);
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

    fn timing_series_json(series: &TimingSeries) -> ProbeResult<Value> {
        Ok(json!({
            "timestamp_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime",
            "gpu_duration": summary_json(&series.gpu_us)?,
            "host_encode_duration": summary_json(&series.encode_us)?,
            "host_submit_duration": summary_json(&series.submit_us)?,
            "host_wait_duration": summary_json(&series.wait_us)?,
            "host_wall_duration": summary_json(&series.host_wall_us)?,
            "measured_gpu_timestamp_intervals_ns": series.gpu_intervals_ns,
        }))
    }

    /// Run the v2 checkpoint from the dedicated wrapper example.  V1's CLI,
    /// receipt schema, and sealed receipt remain untouched; this function is
    /// deliberately additive so the QAT child can be compared to its exact
    /// predecessor rather than silently replacing it.
    pub fn run_qat_v2() -> ProbeResult<()> {
        let args = parse_qat_v2_args()?;
        let reader = DeepSeekV4FullStreamReader::admit(&args.artifact)?;
        let anchors = verify_source_algorithm_anchors(&reader)?;
        let (weight_meta, scale_meta) = {
            let pair = reader.native_scale_pair(LAYER0_WQ_A_WEIGHT)?;
            if pair.kind != NativeScalePairKind::Fp8E4M3fn
                || pair.weight.name != LAYER0_WQ_A_WEIGHT
                || pair.scale.name != LAYER0_WQ_A_SCALE
                || pair.weight.shape.as_slice()
                    != [LAYER0_WQ_A_ROWS as u64, LAYER0_WQ_A_COLS as u64]
                || pair.scale.shape.as_slice()
                    != [
                        (LAYER0_WQ_A_ROWS / ACT_QUANT_BLOCK) as u64,
                        (LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK) as u64,
                    ]
                || pair.logical_k != LAYER0_WQ_A_COLS as u64
                || pair.out_rows != LAYER0_WQ_A_ROWS as u64
            {
                return Err(failure(
                    "layer-0 WQ-A changed from the sealed source-native FP8 geometry",
                ));
            }
            (pair.weight.clone(), pair.scale.clone())
        };
        let input_bf16 = deterministic_wq_a_input_bf16();
        let input_bf16_le = u16_le_bytes(&input_bf16);
        if input_bf16.len() != LAYER0_WQ_A_COLS {
            return Err(failure("QAT v2 input does not match layer-0 WQ-A K"));
        }
        let cpu = layer0_wq_a_cpu_oracle(&reader, &input_bf16)?;
        let cpu_oracle = validate_cpu_oracle_receipt(&args.cpu_oracle, &reader, &input_bf16, &cpu)?;
        let predecessor =
            validate_predecessor_v1_receipt(&args.predecessor_v1, &reader, &cpu_oracle)?;

        // GPU upload reads the same source-bound tensor pair as v1.  The full
        // parent safetensors files are never materialized.
        let weights =
            reader.read_verified_full(LAYER0_WQ_A_WEIGHT, LAYER0_WQ_A_ROWS * LAYER0_WQ_A_COLS)?;
        let weight_scales = reader.read_verified_full(
            LAYER0_WQ_A_SCALE,
            (LAYER0_WQ_A_ROWS / ACT_QUANT_BLOCK) * (LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK),
        )?;
        let rows = LAYER0_WQ_A_ROWS as u32;
        let cols = LAYER0_WQ_A_COLS as u32;
        let scale_cols = (LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK) as u32;

        let context = MetalContext::new_with_trace(true)?;
        let device_name = context.device_name();
        let qat_pipeline = context.pipeline(QAT_SIMDGROUP_KERNEL)?;
        let projection_pipeline = context.pipeline(SIMD_CANDIDATE_KERNEL)?;
        let qat_pipeline_thread_execution_width = qat_pipeline.thread_execution_width() as u64;
        let qat_pipeline_max_total_threads =
            qat_pipeline.max_total_threads_per_threadgroup() as u32;
        let projection_pipeline_thread_execution_width =
            projection_pipeline.thread_execution_width() as u64;
        let projection_pipeline_max_total_threads =
            projection_pipeline.max_total_threads_per_threadgroup() as u32;
        drop(qat_pipeline);
        drop(projection_pipeline);
        if qat_pipeline_max_total_threads < QAT_WINNER_THREADS {
            return Err(failure(
                "QAT winner threadgroup is not supported by this Metal pipeline",
            ));
        }
        let projection_threads = (projection_pipeline_max_total_threads.min(256) / 32) * 32;
        if projection_threads < 32 {
            return Err(failure(
                "projection SIMDgroup candidate cannot admit one 32-lane SIMDgroup",
            ));
        }
        let qat_blocks = (LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK) as u32;
        let qat_grid_threads = qat_blocks * QAT_WINNER_THREADS;

        let input_buffer = context.new_buffer_with_bytes_checked(&input_bf16_le)?;
        let weight_buffer = context.new_buffer_with_bytes_checked(&weights)?;
        let weight_scale_buffer = context.new_buffer_with_bytes_checked(&weight_scales)?;
        let activation_buffer = context.new_buffer_checked(LAYER0_WQ_A_COLS)?;
        let activation_scale_buffer =
            context.new_buffer_checked(LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK)?;
        let output_buffer =
            context.new_buffer_checked(LAYER0_WQ_A_ROWS * std::mem::size_of::<f32>())?;

        let run_nonce = sha256_join(&[
            reader.manifest_seal_sha256(),
            &weight_meta.segments[0].sha256,
            &scale_meta.segments[0].sha256,
            &cpu_oracle.seal_sha256,
            &predecessor.seal_sha256,
            "model_linear_qat_v2_gpu_chain",
        ]);

        // Timestamped separated topology: two command buffers and two waits
        // per measured chain.  This is an honest staging baseline, not a
        // pretend fused graph.
        let separated_interval_id = sha256_join(&[&run_nonce, "separated_qat_projection"]);
        let separated_trace = PhysicalTraceGuard::begin(PhysicalTraceIdentity::new(
            separated_interval_id.clone(),
            run_nonce.clone(),
            "model_linear_qat_v2_separated".to_owned(),
            "qat_then_fp8_projection_two_command_buffers".to_owned(),
            Some(1),
            0,
        )?)?;
        let mut qat_series = TimingSeries::with_capacity(args.trials);
        let mut projection_series = TimingSeries::with_capacity(args.trials);
        let mut separated_chain_gpu_us = Vec::with_capacity(args.trials);
        let mut separated_chain_host_wall_us = Vec::with_capacity(args.trials);
        for iteration in 0..(args.warmups + args.trials) {
            let qat_timing = context.dispatch_threads_timed(
                QAT_SIMDGROUP_KERNEL,
                (qat_grid_threads, 1, 1),
                (QAT_WINNER_THREADS, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&input_buffer), 0);
                    encoder.set_buffer(1, Some(&activation_buffer), 0);
                    encoder.set_buffer(2, Some(&activation_scale_buffer), 0);
                    set_u32(encoder, 3, &cols);
                    set_u32(encoder, 4, &QAT_WINNER_THREADS);
                    set_u32(encoder, 5, &QAT_WINNER_VECTOR_WIDTH);
                },
            )?;
            require_completed_dispatch(&qat_timing, "QAT winner act_quant")?;
            let projection_timing = context.dispatch_threads_timed(
                SIMD_CANDIDATE_KERNEL,
                (projection_threads, rows, 1),
                (projection_threads, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&weight_buffer), 0);
                    encoder.set_buffer(1, Some(&weight_scale_buffer), 0);
                    encoder.set_buffer(2, Some(&activation_buffer), 0);
                    encoder.set_buffer(3, Some(&activation_scale_buffer), 0);
                    encoder.set_buffer(4, Some(&output_buffer), 0);
                    set_u32(encoder, 5, &rows);
                    set_u32(encoder, 6, &cols);
                    set_u32(encoder, 7, &scale_cols);
                    set_u32(encoder, 8, &projection_threads);
                },
            )?;
            require_completed_dispatch(&projection_timing, "QAT winner FP8 projection")?;
            if iteration >= args.warmups {
                qat_series.record_dispatch(&qat_timing)?;
                projection_series.record_dispatch(&projection_timing)?;
                separated_chain_gpu_us.push(
                    qat_timing.gpu_duration_us.expect("checked")
                        + projection_timing.gpu_duration_us.expect("checked"),
                );
                separated_chain_host_wall_us
                    .push(qat_timing.host_wall_us + projection_timing.host_wall_us);
            }
        }
        let separated_activation = read_gpu_bytes(&activation_buffer, LAYER0_WQ_A_COLS)?;
        let separated_scales =
            read_gpu_bytes(&activation_scale_buffer, LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK)?;
        if separated_activation != cpu.quantized_input.activation_e4m3fn
            || separated_scales != cpu.quantized_input.scales_e8m0fnu
            || sha256(&separated_activation) != cpu_oracle.activation_sha256
            || sha256(&separated_scales) != cpu_oracle.scale_sha256
        {
            return Err(failure(
                "separated QAT winner output is not byte-exact to canonical CPU oracle v2",
            ));
        }
        let separated_output = read_gpu_f32(&output_buffer, LAYER0_WQ_A_ROWS)?;
        let separated_parity = f32_parity(&cpu.output.fp32, &separated_output)?;
        if separated_parity.get("status").and_then(Value::as_str) != Some("PASS") {
            return Err(failure(
                "separated QAT-to-FP8 projection chain failed canonical CPU projection parity",
            ));
        }
        let separated_bf16 = separated_output
            .iter()
            .copied()
            .map(|value| bf16::from_f32(value).to_bits())
            .collect::<Vec<_>>();
        let separated_bf16_sha256 = sha256(&u16_le_bytes(&separated_bf16));
        if separated_bf16_sha256 != cpu_oracle.output_bf16_sha256 {
            return Err(failure(
                "separated QAT-to-FP8 projection BF16 output differs from canonical CPU oracle",
            ));
        }
        let separated_physical_counts = separated_trace.counts();
        drop(separated_trace);
        let separated_trace_samples = context.drain_trace();
        let per_topology_iteration = args.warmups + args.trials;
        if separated_physical_counts.command_count != per_topology_iteration as u64 * 2
            || separated_physical_counts.encoder_count != per_topology_iteration as u64 * 2
            || separated_trace_samples.len() != per_topology_iteration * 2
        {
            return Err(failure(
                "separated QAT chain physical command/encoder/trace accounting mismatch",
            ));
        }

        // One command buffer with one ordered compute encoder.  The QAT
        // candidate writes the device activation/scales, a resource-scoped
        // Metal barrier seals that GPU dependency, and the projection
        // candidate consumes those exact same device buffers.  There is no
        // CPU wait, readback, copy, or host activation/routing between stages.
        let batch_interval_id = sha256_join(&[&run_nonce, "single_cb_gpu_handoff"]);
        let batch_trace = PhysicalTraceGuard::begin(PhysicalTraceIdentity::new(
            batch_interval_id.clone(),
            run_nonce.clone(),
            "model_linear_qat_v2_one_command_buffer".to_owned(),
            "gpu_internal_qat_to_fp8_projection_handoff".to_owned(),
            Some(1),
            0,
        )?)?;
        let mut batch_series = TimingSeries::with_capacity(args.trials);
        for iteration in 0..(args.warmups + args.trials) {
            let batch_timing = context.dispatch_batch_timed(|batch| {
                let barrier_resources: [&metal::ResourceRef; 2] =
                    [&**activation_buffer, &**activation_scale_buffer];
                batch.dispatch_threads_pair_in_one_encoder(
                    QAT_SIMDGROUP_KERNEL,
                    (qat_grid_threads, 1, 1),
                    (QAT_WINNER_THREADS, 1, 1),
                    |encoder| {
                        encoder.set_buffer(0, Some(&input_buffer), 0);
                        encoder.set_buffer(1, Some(&activation_buffer), 0);
                        encoder.set_buffer(2, Some(&activation_scale_buffer), 0);
                        set_u32(encoder, 3, &cols);
                        set_u32(encoder, 4, &QAT_WINNER_THREADS);
                        set_u32(encoder, 5, &QAT_WINNER_VECTOR_WIDTH);
                    },
                    &barrier_resources,
                    SIMD_CANDIDATE_KERNEL,
                    (projection_threads, rows, 1),
                    (projection_threads, 1, 1),
                    |encoder| {
                        encoder.set_buffer(0, Some(&weight_buffer), 0);
                        encoder.set_buffer(1, Some(&weight_scale_buffer), 0);
                        encoder.set_buffer(2, Some(&activation_buffer), 0);
                        encoder.set_buffer(3, Some(&activation_scale_buffer), 0);
                        encoder.set_buffer(4, Some(&output_buffer), 0);
                        set_u32(encoder, 5, &rows);
                        set_u32(encoder, 6, &cols);
                        set_u32(encoder, 7, &scale_cols);
                        set_u32(encoder, 8, &projection_threads);
                    },
                )?;
                Ok(())
            })?;
            require_completed_batch(&batch_timing, "one-command-buffer QAT-to-FP8 chain")?;
            if iteration >= args.warmups {
                batch_series.record_batch(&batch_timing)?;
            }
        }
        let batch_activation = read_gpu_bytes(&activation_buffer, LAYER0_WQ_A_COLS)?;
        let batch_scales =
            read_gpu_bytes(&activation_scale_buffer, LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK)?;
        if batch_activation != cpu.quantized_input.activation_e4m3fn
            || batch_scales != cpu.quantized_input.scales_e8m0fnu
            || sha256(&batch_activation) != cpu_oracle.activation_sha256
            || sha256(&batch_scales) != cpu_oracle.scale_sha256
        {
            return Err(failure(
                "one-command-buffer QAT winner output is not byte-exact to canonical CPU oracle v2",
            ));
        }
        let batch_output = read_gpu_f32(&output_buffer, LAYER0_WQ_A_ROWS)?;
        let batch_parity = f32_parity(&cpu.output.fp32, &batch_output)?;
        if batch_parity.get("status").and_then(Value::as_str) != Some("PASS") {
            return Err(failure(
                "one-command-buffer QAT-to-FP8 projection chain failed canonical CPU projection parity",
            ));
        }
        let batch_bf16 = batch_output
            .iter()
            .copied()
            .map(|value| bf16::from_f32(value).to_bits())
            .collect::<Vec<_>>();
        let batch_bf16_sha256 = sha256(&u16_le_bytes(&batch_bf16));
        if batch_bf16_sha256 != cpu_oracle.output_bf16_sha256 {
            return Err(failure(
                "one-command-buffer QAT-to-FP8 projection BF16 output differs from canonical CPU oracle",
            ));
        }
        let batch_physical_counts = batch_trace.counts();
        drop(batch_trace);
        let batch_trace_samples = context.drain_trace();
        if batch_physical_counts.command_count != per_topology_iteration as u64
            || batch_physical_counts.encoder_count != per_topology_iteration as u64
            || batch_trace_samples.len() != per_topology_iteration
        {
            return Err(failure(
                "one-encoder/one-command-buffer QAT chain physical command/encoder/trace accounting mismatch",
            ));
        }
        let (buffers_created, bytes_allocated, commits) = context.drain_stats();
        let expected_total_commits = per_topology_iteration * 3;
        if commits != expected_total_commits {
            return Err(failure(format!(
                "QAT v2 Metal commit count {} differs from expected {}",
                commits, expected_total_commits
            )));
        }

        let qat_timing = timing_series_json(&qat_series)?;
        let projection_timing = timing_series_json(&projection_series)?;
        let separated_chain_gpu = summary_json(&separated_chain_gpu_us)?;
        let separated_chain_host_wall = summary_json(&separated_chain_host_wall_us)?;
        let batch_timing = timing_series_json(&batch_series)?;
        let separated_gpu_p50 = number_at(&separated_chain_gpu, &["p50_us"])?;
        let separated_host_p50 = number_at(&separated_chain_host_wall, &["p50_us"])?;
        let batch_gpu_p50 = number_at(&batch_timing, &["gpu_duration", "p50_us"])?;
        let batch_host_p50 = number_at(&batch_timing, &["host_wall_duration", "p50_us"])?;
        let unsigned = json!({
            "schema": "hawking.gravity.deepseek_v4.model_linear_fp8_qat_metal_component_parity.v2",
            "status": "PASS_REAL_METAL_QAT_MODEL_LINEAR_COMPONENT_PARITY_NOT_FULL_RUNTIME",
            "scope": {
                "component": "model.linear only: layers.0.attn.wq_a",
                "model_linear_component_only": true,
                "qat_candidate_only": true,
                "not_embedding": true,
                "not_mhc": true,
                "not_attention": true,
                "not_router_or_expert_execution": true,
                "not_full_model_load": true,
                "not_full_model_forward": true,
                "not_token_execution_or_generation": true,
                "not_hcli_endpoint": true,
                "not_base_true_tps_measurement": true,
                "not_registered_43_layer_runtime_adapter": true,
            },
            "predecessor_v1": {
                "path": predecessor.path,
                "file_sha256": predecessor.file_sha256,
                "seal_sha256": predecessor.seal_sha256,
                "sealed_authority_act_quant_gpu_duration_us": predecessor.act_quant_gpu_duration_us,
                "v1_receipt_preserved_unchanged": true,
            },
            "artifact": {
                "path": reader.artifact_root().display().to_string(),
                "full_stream_schema": FULL_STREAM_SCHEMA,
                "full_stream_status": FULL_STREAM_STATUS,
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
                "source_tensor_chunk_bindings": {
                    "weight": tensor_binding_json(&weight_meta),
                    "weight_scale": tensor_binding_json(&scale_meta),
                    "touched_chunks_sha256_verified_before_gpu_upload": true,
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
                "activation_scale_sha256": cpu_oracle.scale_sha256,
                "output_fp32_sha256_le": cpu_oracle.output_fp32_sha256,
                "output_bf16_sha256_le": cpu_oracle.output_bf16_sha256,
                "direct_cpu_oracle_recomputed_and_matches_sealed_v2": true,
            },
            "input": {
                "kind": "deterministic_exact_bf16_bitpattern_vector_v1",
                "captured_from_model_forward": false,
                "dtype": "BF16",
                "length": input_bf16.len(),
                "sha256_bf16_le": sha256(&input_bf16_le),
            },
            "qat_winner": {
                "kernel": QAT_SIMDGROUP_KERNEL,
                "source_algorithm": "act_quant(block_size=128, scale_fmt=ue8m0, finite E4M3FN clamp[-448,448])",
                "winner_from_receipt": "DSV4F_ACT_QUANT_SIMDGROUP_SWEEP-v1.json",
                "threads": [QAT_WINNER_THREADS, 1, 1],
                "vector_width_bf16_elements": QAT_WINNER_VECTOR_WIDTH,
                "grid_threads": [qat_grid_threads, 1, 1],
                "activation_dtype": "F8_E4M3FN",
                "scale_dtype": "F8_E8M0FNU",
                "activation_and_scale_byte_exact_cpu_oracle_v2_required": true,
                "fallback": false,
            },
            "projection": {
                "kernel": SIMD_CANDIDATE_KERNEL,
                "source_algorithm": "FP8 block GEMV: E4M3FN activation×weight K=128 dot then activation E8M0FNU×weight E8M0FNU scaling",
                "threads": [projection_threads, 1, 1],
                "grid_threads": [projection_threads, rows, 1],
                "output_cpu_oracle_parity_required": true,
                "output_bf16_hash_required": true,
                "fallback": false,
            },
            "separated_timestamped_chain": {
                "topology": "two completed command buffers: QAT then projection",
                "cpu_readback_or_wait_between_gpu_stages": true,
                "note": "The explicit per-stage timestamp baseline waits after QAT before issuing the projection; it is retained only as a command-topology comparison, not a runtime fused graph.",
                "warmup_chains": args.warmups,
                "measured_chains": args.trials,
                "qat_timing": qat_timing,
                "projection_timing": projection_timing,
                "chain_gpu_duration": separated_chain_gpu,
                "chain_host_wall_duration": separated_chain_host_wall,
                "activation_sha256": sha256(&separated_activation),
                "activation_bytewise_cpu_oracle_v2_match": true,
                "scale_sha256": sha256(&separated_scales),
                "scale_bytewise_cpu_oracle_v2_match": true,
                "projection_output_fp32_sha256_le": sha256(&f32_le_bytes(&separated_output)),
                "projection_cpu_oracle_parity": separated_parity,
                "projection_output_bf16_sha256_le": separated_bf16_sha256,
                "projection_output_bf16_hash_matches_cpu_oracle": true,
                "gpu_dispatches": per_topology_iteration * 2,
                "command_buffers": per_topology_iteration * 2,
                "compute_encoders": per_topology_iteration * 2,
                "cpu_visible_waits": per_topology_iteration * 2,
                "empty_command_buffers": 0,
                "fallback": false,
            },
            "one_command_buffer_gpu_handoff": {
                "technical_validity": "PASS: one ordered compute encoder in one committed MTLCommandBuffer; a resource-scoped Metal barrier makes QAT writes visible to the projection dispatch without host handoff.",
                "topology": "one completed command buffer with one ordered compute encoder/two dispatches and a resource-scoped GPU barrier",
                "cpu_readback_or_wait_between_gpu_stages": false,
                "gpu_internal_intermediate_buffers": ["activation_e4m3fn[4096]", "activation_scales_e8m0[32]"],
                "warmup_chains": args.warmups,
                "measured_chains": args.trials,
                "chain_timing": batch_timing,
                "activation_sha256": sha256(&batch_activation),
                "activation_bytewise_cpu_oracle_v2_match": true,
                "scale_sha256": sha256(&batch_scales),
                "scale_bytewise_cpu_oracle_v2_match": true,
                "projection_output_fp32_sha256_le": sha256(&f32_le_bytes(&batch_output)),
                "projection_cpu_oracle_parity": batch_parity,
                "projection_output_bf16_sha256_le": batch_bf16_sha256,
                "projection_output_bf16_hash_matches_cpu_oracle": true,
                "gpu_dispatches": per_topology_iteration * 2,
                "command_buffers": per_topology_iteration,
                "compute_encoders": per_topology_iteration,
                "cpu_visible_waits": per_topology_iteration,
                "empty_command_buffers": 0,
                "fallback": false,
            },
            "command_topology_comparison": {
                "separated_gpu_p50_us": separated_gpu_p50,
                "one_command_buffer_gpu_p50_us": batch_gpu_p50,
                "separated_host_wall_p50_us": separated_host_p50,
                "one_command_buffer_host_wall_p50_us": batch_host_p50,
                "gpu_p50_speedup_one_command_buffer_over_separated": format!("{:.6}", separated_gpu_p50 as f64 / batch_gpu_p50 as f64),
                "host_wall_p50_speedup_one_command_buffer_over_separated": format!("{:.6}", separated_host_p50 as f64 / batch_host_p50 as f64),
                "command_buffers_per_chain": { "separated": 2, "one_command_buffer": 1 },
                "cpu_visible_waits_per_chain": { "separated": 2, "one_command_buffer": 1 },
                "interpretation": "GPU timestamps measure command-buffer execution only; host-wall comparison exposes the explicit submission/wait topology. No full-runtime promotion follows from this component result.",
            },
            "metal": {
                "device": device_name,
                "pipelines_precompiled_before_warmup": true,
                "qat_pipeline_thread_execution_width": qat_pipeline_thread_execution_width,
                "qat_pipeline_max_total_threads_per_threadgroup": qat_pipeline_max_total_threads,
                "projection_pipeline_thread_execution_width": projection_pipeline_thread_execution_width,
                "projection_pipeline_max_total_threads_per_threadgroup": projection_pipeline_max_total_threads,
                "buffers_created": buffers_created,
                "bytes_allocated": bytes_allocated,
                "total_gpu_dispatches": per_topology_iteration * 4,
                "total_command_buffers": expected_total_commits,
                "total_compute_encoders": per_topology_iteration * 3,
                "total_cpu_visible_waits": expected_total_commits,
                "total_empty_command_buffers": 0,
                "separated_physical_trace_command_buffers": separated_physical_counts.command_count,
                "separated_physical_trace_compute_encoders": separated_physical_counts.encoder_count,
                "separated_trace_samples": separated_trace_samples.len(),
                "one_command_buffer_physical_trace_command_buffers": batch_physical_counts.command_count,
                "one_command_buffer_physical_trace_compute_encoders": batch_physical_counts.encoder_count,
                "one_command_buffer_trace_samples": batch_trace_samples.len(),
                "fallback": false,
                "fallback_count": 0,
                "cpu_used_only_for_source_oracle_recomputation_and_post_chain_comparison": true,
            },
            "logical_bytes_per_chain": {
                "qat_read_bf16": input_bf16_le.len(),
                "qat_written_e4m3fn_plus_e8m0": LAYER0_WQ_A_COLS + LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK,
                "projection_read_weight_plus_scale_plus_device_activation_plus_device_scale": weights.len() + weight_scales.len() + LAYER0_WQ_A_COLS + LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK,
                "projection_written_fp32": LAYER0_WQ_A_ROWS * std::mem::size_of::<f32>(),
                "one_command_buffer_host_intermediate_copy_bytes": 0,
            },
            "physical_trace": {
                "run_nonce": run_nonce,
                "separated_interval_id": separated_interval_id,
                "one_command_buffer_interval_id": batch_interval_id,
            },
            "claim_boundary": "This is a real source-derived layer-0 model.linear GPU component parity and command-topology measurement. It does not establish a DeepSeek-V4 runtime, full model forward, token generation, HCLI behavior, or BASE_TRUE_TPS.",
        });
        let (receipt, seal_sha256) = seal(unsigned)?;
        write_new_receipt(&args.out, &receipt)?;
        println!(
            "{}",
            serde_json::to_string(&json!({
                "status": "PASS_REAL_METAL_QAT_MODEL_LINEAR_COMPONENT_PARITY_NOT_FULL_RUNTIME",
                "receipt": args.out,
                "seal_sha256": seal_sha256,
                "one_command_buffer_gpu_p50_us": batch_gpu_p50,
                "one_command_buffer_host_wall_p50_us": batch_host_p50,
            }))?
        );
        Ok(())
    }

    fn parse_args() -> ProbeResult<Args> {
        let mut artifact = None::<PathBuf>;
        let mut cpu_oracle = None::<PathBuf>;
        let mut out = None::<PathBuf>;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--artifact" => artifact = args.next().map(PathBuf::from),
                "--cpu-oracle" => cpu_oracle = args.next().map(PathBuf::from),
                "--out" => out = args.next().map(PathBuf::from),
                "--help" | "-h" => {
                    println!(
                        "usage: gravity_deepseek_v4_model_linear_metal_checkpoint --artifact <absolute full Gravity dir> --cpu-oracle <absolute DSV4F_ACT_QUANT_WQ_A_CPU_ORACLE-v2.json> --out <absolute receipt.json>"
                    );
                    std::process::exit(0);
                }
                other => return Err(failure(format!("unknown argument {other:?}"))),
            }
        }
        let artifact = artifact.ok_or_else(|| failure("--artifact is required"))?;
        let cpu_oracle = cpu_oracle.ok_or_else(|| failure("--cpu-oracle is required"))?;
        let out = out.ok_or_else(|| failure("--out is required"))?;
        if !artifact.is_absolute() || !cpu_oracle.is_absolute() || !out.is_absolute() {
            return Err(failure(
                "--artifact, --cpu-oracle, and --out must be absolute paths",
            ));
        }
        Ok(Args {
            artifact,
            cpu_oracle,
            out,
        })
    }

    fn validate_cpu_oracle_receipt(
        path: &Path,
        reader: &DeepSeekV4FullStreamReader,
        input_bf16: &[u16],
        cpu: &Layer0WqACpuOracleResult,
    ) -> ProbeResult<CpuOracleBinding> {
        if path.file_name().and_then(|name| name.to_str()) != Some(CPU_ORACLE_V2_BASENAME) {
            return Err(failure(
                "only the canonical DSV4F_ACT_QUANT_WQ_A_CPU_ORACLE-v2.json may bind this GPU checkpoint",
            ));
        }
        let meta = fs::symlink_metadata(path)?;
        if meta.file_type().is_symlink() || !meta.file_type().is_file() {
            return Err(failure("--cpu-oracle must be a regular non-symlink file"));
        }
        let canonical_path = fs::canonicalize(path)?;
        let raw = fs::read(&canonical_path)?;
        let file_sha256 = sha256(&raw);
        let mut value: Value = serde_json::from_slice(&raw).map_err(|error| {
            failure(format!("canonical CPU oracle receipt is not JSON: {error}"))
        })?;
        let seal_sha256 = value
            .as_object_mut()
            .ok_or_else(|| failure("canonical CPU oracle receipt root is not an object"))?
            .remove("seal_sha256")
            .and_then(|value| value.as_str().map(str::to_owned))
            .ok_or_else(|| failure("canonical CPU oracle receipt has no string seal_sha256"))?;
        if !is_sha256(&seal_sha256) || sha256(&canonical_json(&value)) != seal_sha256 {
            return Err(failure(
                "canonical CPU oracle v2 receipt seal does not verify under its declared canonical JSON rule",
            ));
        }
        value
            .as_object_mut()
            .expect("object was checked")
            .insert("seal_sha256".to_owned(), Value::String(seal_sha256.clone()));
        if text_at(&value, &["schema"])? != CPU_ORACLE_SCHEMA
            || text_at(&value, &["status"])? != CPU_ORACLE_STATUS
        {
            return Err(failure(
                "--cpu-oracle is not the canonical passing source-derived CPU oracle v2 schema/status",
            ));
        }
        if text_at(&value, &["artifact", "manifest_schema"])? != FULL_STREAM_SCHEMA
            || text_at(&value, &["artifact", "manifest_status"])? != FULL_STREAM_STATUS
            || text_at(&value, &["artifact", "manifest_seal_sha256"])?
                != reader.manifest_seal_sha256()
            || text_at(&value, &["artifact", "manifest_file_sha256"])?
                != reader.manifest_file_sha256()
            || text_at(&value, &["artifact", "restart_receipt_seal_sha256"])?
                != reader.restart_seal_sha256()
            || text_at(&value, &["artifact", "source", "repository"])?
                != reader.source_identity().repository
            || text_at(&value, &["artifact", "source", "revision"])?
                != reader.source_identity().revision
        {
            return Err(failure(
                "canonical CPU oracle v2 does not bind the admitted full Gravity artifact/source",
            ));
        }
        let input_sha256 = sha256(&u16_le_bytes(input_bf16));
        let activation_sha256 = sha256(&cpu.quantized_input.activation_e4m3fn);
        let scale_sha256 = sha256(&cpu.quantized_input.scales_e8m0fnu);
        let output_fp32_sha256 = sha256(&f32_le_bytes(&cpu.output.fp32));
        let output_bf16_sha256 = sha256(&u16_le_bytes(&cpu.output.bf16_bits));
        if text_at(&value, &["input", "sha256_bf16_le"])? != input_sha256
            || number_at(&value, &["input", "length"])? != input_bf16.len() as u64
            || text_at(&value, &["act_quant", "activation_sha256"])? != activation_sha256
            || text_at(&value, &["act_quant", "scale_sha256"])? != scale_sha256
            || number_at(&value, &["act_quant", "activation_bytes"])?
                != cpu.quantized_input.activation_e4m3fn.len() as u64
            || number_at(&value, &["act_quant", "scale_bytes"])?
                != cpu.quantized_input.scales_e8m0fnu.len() as u64
            || text_at(&value, &["cpu_fp8_gemv", "output_fp32_le_sha256"])? != output_fp32_sha256
            || text_at(&value, &["cpu_fp8_gemv", "output_bf16_le_sha256"])? != output_bf16_sha256
            || number_at(&value, &["cpu_fp8_gemv", "output_fp32_count"])?
                != cpu.output.fp32.len() as u64
            || number_at(&value, &["cpu_fp8_gemv", "output_bf16_count"])?
                != cpu.output.bf16_bits.len() as u64
        {
            return Err(failure(
                "direct CPU oracle recomputation differs from canonical CPU oracle v2 hashes or geometry",
            ));
        }
        let receipt_scales = value_at(&value, &["act_quant", "scale_e8m0fnu_bytes"])?
            .as_array()
            .ok_or_else(|| failure("CPU oracle v2 scale_e8m0fnu_bytes is not an array"))?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| u8::try_from(value).ok())
                    .ok_or_else(|| failure("CPU oracle v2 has an invalid E8M0 scale byte"))
            })
            .collect::<ProbeResult<Vec<_>>>()?;
        if receipt_scales != cpu.quantized_input.scales_e8m0fnu {
            return Err(failure(
                "canonical CPU oracle v2 raw E8M0 scale vector differs from direct recomputation",
            ));
        }
        Ok(CpuOracleBinding {
            path: canonical_path,
            file_sha256,
            seal_sha256,
            input_sha256_bf16_le: input_sha256,
            activation_sha256,
            scale_sha256,
            output_fp32_sha256,
            output_bf16_sha256,
        })
    }

    fn value_at<'a>(value: &'a Value, path: &[&str]) -> ProbeResult<&'a Value> {
        let mut current = value;
        for key in path {
            current = current.get(*key).ok_or_else(|| {
                failure(format!("receipt is missing JSON path {}", path.join(".")))
            })?;
        }
        Ok(current)
    }

    fn text_at<'a>(value: &'a Value, path: &[&str]) -> ProbeResult<&'a str> {
        value_at(value, path)?.as_str().ok_or_else(|| {
            failure(format!(
                "receipt JSON path {} is not a string",
                path.join(".")
            ))
        })
    }

    fn number_at(value: &Value, path: &[&str]) -> ProbeResult<u64> {
        value_at(value, path)?.as_u64().ok_or_else(|| {
            failure(format!(
                "receipt JSON path {} is not an unsigned integer",
                path.join(".")
            ))
        })
    }

    fn require_completed_dispatch(timing: &MetalDispatchTiming, stage: &str) -> ProbeResult<()> {
        if timing.compute_dispatches != 1
            || timing.command_buffers != 1
            || timing.compute_encoders != 1
            || timing.gpu_duration_us.unwrap_or(0) == 0
            || timing.gpu_start_ns.is_none()
            || timing.gpu_end_ns.is_none()
        {
            return Err(failure(format!(
                "{stage} did not complete as one GPU-timestamped command buffer/encoder/dispatch",
            )));
        }
        Ok(())
    }

    fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &u32) {
        encoder.set_bytes(
            index,
            std::mem::size_of::<u32>() as u64,
            value as *const u32 as *const _,
        );
    }

    fn read_gpu_bytes(buffer: &metal::Buffer, length: usize) -> ProbeResult<Vec<u8>> {
        if buffer.length() < length as u64 {
            return Err(failure(
                "Metal buffer is smaller than requested GPU byte readback",
            ));
        }
        Ok(unsafe { std::slice::from_raw_parts(buffer.contents() as *const u8, length).to_vec() })
    }

    fn read_gpu_f32(buffer: &metal::Buffer, count: usize) -> ProbeResult<Vec<f32>> {
        let bytes = count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| failure("GPU FP32 readback byte count overflow"))?;
        if buffer.length() < bytes as u64 {
            return Err(failure(
                "Metal buffer is smaller than requested GPU FP32 readback",
            ));
        }
        let output =
            unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, count).to_vec() };
        if output.iter().any(|value| !value.is_finite()) {
            return Err(failure(
                "GPU source-linear projection produced a non-finite output",
            ));
        }
        Ok(output)
    }

    fn f32_parity(cpu: &[f32], gpu: &[f32]) -> ProbeResult<Value> {
        if cpu.is_empty() || cpu.len() != gpu.len() {
            return Err(failure(
                "CPU/GPU output vectors have incompatible source-linear geometry",
            ));
        }
        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        let mut sum_abs = 0.0_f64;
        let mut worst = 0usize;
        let mut failures = 0usize;
        for (index, (&expected, &observed)) in cpu.iter().zip(gpu).enumerate() {
            if !expected.is_finite() || !observed.is_finite() {
                return Err(failure(
                    "CPU/GPU source-linear parity encountered a non-finite value",
                ));
            }
            let abs = (expected - observed).abs();
            let rel = abs / expected.abs().max(1.0e-6);
            if abs > max_abs {
                max_abs = abs;
                worst = index;
            }
            max_rel = max_rel.max(rel);
            sum_abs += f64::from(abs);
            if abs > OUTPUT_ABS_TOLERANCE + OUTPUT_REL_TOLERANCE * expected.abs() {
                failures += 1;
            }
        }
        Ok(json!({
            "status": if failures == 0 { "PASS" } else { "FAIL" },
            "comparison": "per-output abs_error <= 1e-4 + 1e-4 * abs(canonical_cpu_oracle_fp32)",
            "failing_outputs": failures,
            "max_abs_error_f32": max_abs.to_string(),
            "mean_abs_error_f64": (sum_abs / cpu.len() as f64).to_string(),
            "max_relative_error_f32": max_rel.to_string(),
            "worst_output_row": worst,
            "cpu_value_at_worst_row_f32": cpu[worst].to_string(),
            "gpu_value_at_worst_row_f32": gpu[worst].to_string(),
        }))
    }

    fn timing_json(timing: &MetalDispatchTiming) -> Value {
        json!({
            "authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime",
            "gpu_duration_us": timing.gpu_duration_us.expect("timing was checked before receipt construction"),
            "gpu_start_ns": timing.gpu_start_ns.expect("timing was checked before receipt construction"),
            "gpu_end_ns": timing.gpu_end_ns.expect("timing was checked before receipt construction"),
            "pipeline_lookup_us": timing.pipeline_lookup_us,
            "encode_us": timing.encode_us,
            "submit_us": timing.submit_us,
            "wait_us": timing.wait_us,
            "host_wall_us": timing.host_wall_us,
            "command_buffers": timing.command_buffers,
            "compute_encoders": timing.compute_encoders,
            "compute_dispatches": timing.compute_dispatches,
        })
    }

    fn tensor_binding_json(tensor: &DeepSeekV4TensorMetadata) -> Value {
        json!({
            "name": tensor.name,
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "bytes": tensor.bytes,
            "source_file_start": tensor.source_file_start,
            "source_file_end": tensor.source_file_end,
            "source_shard": tensor.source_shard,
            "segments": tensor.segments.iter().map(|segment| json!({
                "bytes": segment.bytes,
                "chunk_relpath": segment.chunk_relpath,
                "sha256": segment.sha256,
                "source_file_start": segment.source_file_start,
                "source_file_end": segment.source_file_end,
                "tensor_start": segment.tensor_start,
                "tensor_end": segment.tensor_end,
                "row_start": segment.row_start,
                "row_count": segment.row_count,
            })).collect::<Vec<_>>(),
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

    fn is_sha256(value: &str) -> bool {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }

    fn u16_le_bytes(values: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<u16>());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
        for value in values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        bytes
    }

    fn canonical_json(value: &Value) -> Vec<u8> {
        let mut out = Vec::new();
        write_canonical_json(&mut out, value);
        out
    }

    fn write_canonical_json(out: &mut Vec<u8>, value: &Value) {
        match value {
            Value::Null => out.extend_from_slice(b"null"),
            Value::Bool(true) => out.extend_from_slice(b"true"),
            Value::Bool(false) => out.extend_from_slice(b"false"),
            Value::Number(number) => out.extend_from_slice(number.to_string().as_bytes()),
            Value::String(string) => out.extend_from_slice(
                serde_json::to_string(string)
                    .expect("JSON string serialization is infallible")
                    .as_bytes(),
            ),
            Value::Array(values) => {
                out.push(b'[');
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        out.push(b',');
                    }
                    write_canonical_json(out, value);
                }
                out.push(b']');
            }
            Value::Object(object) => {
                let mut keys = object.keys().collect::<Vec<_>>();
                keys.sort();
                out.push(b'{');
                for (index, key) in keys.into_iter().enumerate() {
                    if index != 0 {
                        out.push(b',');
                    }
                    out.extend_from_slice(
                        serde_json::to_string(key)
                            .expect("JSON string serialization is infallible")
                            .as_bytes(),
                    );
                    out.push(b':');
                    write_canonical_json(out, &object[key]);
                }
                out.push(b'}');
            }
        }
    }

    fn seal(mut receipt: Value) -> ProbeResult<(Value, String)> {
        if !receipt.is_object() || receipt.get("seal_sha256").is_some() {
            return Err(failure(
                "checkpoint receipt must be an unsealed JSON object",
            ));
        }
        let seal_sha256 = sha256(&canonical_json(&receipt));
        receipt
            .as_object_mut()
            .expect("receipt object was checked")
            .insert("seal_sha256".to_owned(), Value::String(seal_sha256.clone()));
        Ok((receipt, seal_sha256))
    }

    fn write_new_receipt(path: &Path, receipt: &Value) -> ProbeResult<()> {
        if path.exists() {
            return Err(failure(format!(
                "refusing to overwrite existing source-linear Metal checkpoint receipt {}",
                path.display(),
            )));
        }
        let parent = path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .ok_or_else(|| failure("--out needs a parent directory"))?;
        fs::create_dir_all(parent)?;
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| failure("--out filename must be UTF-8"))?;
        let temporary = parent.join(format!(
            ".{filename}.{}.model-linear-metal.tmp",
            std::process::id(),
        ));
        let bytes = serde_json::to_vec_pretty(receipt)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| {
                failure(format!(
                    "cannot create checkpoint receipt temporary: {error}"
                ))
            })?;
        if let Err(error) = file
            .write_all(&bytes)
            .and_then(|_| file.write_all(b"\n"))
            .and_then(|_| file.sync_all())
        {
            let _ = fs::remove_file(&temporary);
            return Err(Box::new(error));
        }
        drop(file);
        if let Err(error) = fs::hard_link(&temporary, path) {
            let _ = fs::remove_file(&temporary);
            return Err(failure(format!(
                "refusing to overwrite or link source-linear checkpoint receipt {}: {error}",
                path.display(),
            )));
        }
        fs::remove_file(&temporary)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

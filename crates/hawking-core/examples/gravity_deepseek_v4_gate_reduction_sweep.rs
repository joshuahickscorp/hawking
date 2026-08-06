//! Isolated DeepSeek-V4-Flash P0 Gate reduction-association sweep.
//!
//! This probe intentionally stages only one frozen BF16[4096] Gate input and
//! the admitted layer-0 BF16[256,4096] Gate matrix.  It is not connected to
//! P6, a layer forward, a causal cache, HCLI, token generation, or TPS.  The
//! existing P6A serial Gate kernel is dispatched only as a reproducibility
//! control; C1-C7 are uniquely named candidate kernels in `moe.metal`.
//!
//! A source-calibration shard is required before Metal is constructed, so an
//! absent or malformed Torch target cannot cause a GPU run.  The output keeps
//! only hashes and aggregate metrics: neither source target logits nor device
//! candidate logits are retained.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(
        std::io::Error::other("gravity_deepseek_v4_gate_reduction_sweep requires macOS Metal")
            .into(),
    )
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::gravity_deepseek_v4::DeepSeekV4FullStreamReader;
    use hawking_core::gravity_deepseek_v4_layer0_moe::{
        layer0_hash_route_f64_authority_for_token, LAYER0_FFN_GATE_WEIGHT, ROUTED_EXPERTS,
    };
    use hawking_core::gravity_deepseek_v4_layer0_prefix::{HIDDEN_SIZE, PREFIX_TOKEN_ID};
    use hawking_core::metal::{
        MetalBatchTiming, MetalContext, PhysicalTraceCounts, PhysicalTraceGuard,
        PhysicalTraceIdentity,
    };
    use hawking_core::numeric_parity::{score_pair, ulp_distance_f32, Bounds, PairedScore};
    use serde::Serialize;
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::error::Error;
    use std::fs::{self, File, OpenOptions};
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    const SWEEP_SCHEMA: &str = "hawking.gravity.deepseek_v4.p0_gate_reduction_sweep.v1";
    const SWEEP_STATUS: &str =
        "UNSEALED_REAL_METAL_P0_GATE_REDUCTION_SWEEP_DIAGNOSTIC_NOT_RUNTIME_NOT_TPS";
    const TRACE_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.p7_layer0_position0_gate_input_trace.v1";
    const TRACE_STATUS: &str = "UNSEALED_POST_COMPLETION_BOS_FFN_NORM_GATE_INPUT_TRACE_NON_RECEIPT";
    const CALIBRATION_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.p0_gate_torch_f32_calibration_shard.v1";
    const CALIBRATION_STATUS: &str =
        "UNSEALED_QUALIFIED_SOURCE_CPU_TORCH_F32_GATE_TARGET_NON_RECEIPT";
    const CONTROL_KERNEL: &str = "deepseek_v4_p6a_gate_bf16_matvec_authority";
    const C1_KERNEL: &str = "deepseek_v4_p0_gate_reduction_c1_serial_fma_candidate";
    const C2_KERNEL: &str = "deepseek_v4_p0_gate_reduction_c2_strided4_fma_candidate";
    const C3_KERNEL: &str = "deepseek_v4_p0_gate_reduction_c3_block128_fma_candidate";
    const C4_KERNEL: &str = "deepseek_v4_p0_gate_reduction_c4_simd32_fma_candidate";
    const C5_KERNEL: &str = "deepseek_v4_p0_gate_reduction_c5_strided32_fma_candidate";
    const C6_KERNEL: &str = "deepseek_v4_p0_gate_reduction_c6_block256x16_fma_candidate";
    const C7_KERNEL: &str = "deepseek_v4_p0_gate_reduction_c7_block64x64_fma_candidate";
    const GATE_ROWS: usize = ROUTED_EXPERTS;
    const GATE_COLS: usize = HIDDEN_SIZE;
    const GATE_INPUT_BYTES: usize = GATE_COLS * std::mem::size_of::<u16>();
    const GATE_WEIGHT_BYTES: usize = GATE_ROWS * GATE_INPUT_BYTES;
    const GATE_LOGIT_BYTES: usize = GATE_ROWS * std::mem::size_of::<f32>();
    const WARMUP_TRIALS: usize = 2;
    const CLEAN_TRIALS: usize = 5;
    const SIMDGROUP_WIDTH: u32 = 32;
    const DIRECT_TORCH_MAX_REL: f64 = 1.0e-4;
    const DIRECT_TORCH_REL_L2: f64 = 1.0e-5;
    const DIRECT_TORCH_TOP_K: usize = 5;

    type ProbeResult<T> = Result<T, Box<dyn Error>>;

    struct Args {
        artifact: PathBuf,
        trace: PathBuf,
        source_calibration: PathBuf,
        out: PathBuf,
    }

    #[derive(Clone)]
    struct ArtifactBinding {
        manifest_seal_sha256: String,
        manifest_file_sha256: String,
        restart_receipt_seal_sha256: String,
    }

    struct TraceInput {
        file_sha256: String,
        path: PathBuf,
        artifact: ArtifactBinding,
        bf16_le: Vec<u8>,
        bf16_sha256: String,
        baseline_gate_logits_sha256: String,
        gate_weight_name: String,
        gate_weight_sha256: String,
    }

    struct SourceCalibration {
        file_sha256: String,
        path: PathBuf,
        status: String,
        artifact: ArtifactBinding,
        trace_file_sha256: String,
        trace_input_sha256: String,
        gate_weight_name: String,
        gate_weight_sha256: String,
        target_f32: Vec<f32>,
        target_sha256: String,
    }

    #[derive(Clone, Copy)]
    enum CandidateGeometry {
        ScalarRows,
        SimdgroupRows,
    }

    #[derive(Clone, Copy)]
    struct Candidate {
        id: &'static str,
        kernel: &'static str,
        arithmetic: &'static str,
        geometry: CandidateGeometry,
        control: bool,
    }

    #[derive(Serialize)]
    struct F32PairMetrics {
        elements: usize,
        f32_bit_exact: bool,
        f32_bit_mismatch_elements: usize,
        max_abs: f64,
        max_relative_reference_floor_1e_minus_12: f64,
        relative_l2: f64,
        ulp_median: f64,
        ulp_p95: f64,
        ulp_p99: f64,
        ulp_max: f64,
        greedy_argmax_reference: usize,
        greedy_argmax_candidate: usize,
        greedy_argmax_exact_match: bool,
        top_k: usize,
        top_k_reference: Vec<usize>,
        top_k_candidate: Vec<usize>,
        top_k_exact_match: bool,
    }

    #[derive(Serialize)]
    struct Distribution {
        samples: usize,
        min: u64,
        p50: u64,
        p95: u64,
        p99: u64,
        max: u64,
    }

    #[derive(Serialize)]
    struct OptionalDistribution {
        available_samples: usize,
        distribution: Option<Distribution>,
    }

    #[derive(Serialize)]
    struct TimingSummary {
        warmup_trials: usize,
        clean_trials: usize,
        clean_host_wall_us: Distribution,
        clean_wait_us: Distribution,
        clean_encode_us: Distribution,
        clean_submit_us: Distribution,
        clean_pipeline_lookup_us: Distribution,
        clean_gpu_duration_us: OptionalDistribution,
    }

    #[derive(Serialize)]
    struct CandidateTopology {
        command_buffers: u64,
        compute_encoders: u64,
        compute_dispatches: u64,
        cpu_visible_waits: u64,
        expected_runs: u64,
        physical: PhysicalTraceCounts,
        physical_matches_command_topology: bool,
        metal_trace_samples: usize,
        metal_trace_samples_all_dispatch_batches: bool,
        context_commits: usize,
        context_buffers_created_during_runs: usize,
        context_bytes_allocated_during_runs: usize,
    }

    #[derive(Serialize)]
    struct CandidateGeometryReport {
        grid_threads: [u32; 3],
        threads_per_threadgroup: [u32; 3],
        thread_execution_width: u64,
        max_total_threads_per_threadgroup: u64,
        simdgroup_width_required: Option<u32>,
    }

    #[derive(Serialize)]
    struct CandidateResult {
        id: &'static str,
        kernel: &'static str,
        control: bool,
        arithmetic: &'static str,
        geometry: CandidateGeometryReport,
        clean_output_sha256_f32_le: Vec<String>,
        clean_output_hashes_all_equal: bool,
        source_target_f32_comparison: F32PairMetrics,
        direct_torch_compatibility_under_declared_f32_bounds: bool,
        numeric_parity_v2_1_source_and_device_vs_live_fp64: PairedScore,
        source_target_vs_live_fp64_pass: bool,
        candidate_vs_live_fp64_pass: bool,
        promotion_eligible_within_this_probe: bool,
        timing: TimingSummary,
        topology: CandidateTopology,
        fallback: bool,
        raw_candidate_logits_retained: usize,
    }

    pub fn run() -> ProbeResult<()> {
        let args = parse_args()?;
        validate_new_output_path(&args.out)?;

        // All file/schema/source checks occur before MetalContext construction
        // or any buffer allocation. An absent calibration therefore cannot be
        // mistaken for a failed GPU candidate.
        let trace = load_trace(&args.trace)?;
        let calibration = load_source_calibration(&args.source_calibration, &trace)?;
        let reader = DeepSeekV4FullStreamReader::admit(&args.artifact)?;
        validate_artifact_bindings(&reader, &trace, &calibration)?;
        let gate_weight = load_verified_gate_weight(&reader, &trace, &calibration)?;
        let input_bf16 = decode_bf16_le(&trace.bf16_le)?;
        let f64_authority =
            layer0_hash_route_f64_authority_for_token(&reader, PREFIX_TOKEN_ID, &input_bf16)?;
        if f64_authority.logits_f64.len() != GATE_ROWS {
            return Err(failure(
                "live FP64 Gate authority has an invalid logit geometry",
            ));
        }

        let v21_bounds = gate_v21_bounds();
        let source_target_v21 = score_pair(
            &calibration.target_f32,
            &calibration.target_f32,
            &f64_authority.logits_f64,
            &v21_bounds,
        );
        if !source_target_v21.pass {
            return Err(failure(
                "source calibration does not satisfy Numeric Parity V2.1 against the live independent FP64 Gate authority; refusing GPU sweep",
            ));
        }

        let context = MetalContext::new_with_trace(true)?;
        let control_and_candidates = [
            Candidate {
                id: "CONTROL_P6A_SERIAL_NONFUSED",
                kernel: CONTROL_KERNEL,
                control: true,
                arithmetic: "existing P6A serial BF16->F32 product then F32 add, increasing columns; reproducibility control only",
                geometry: CandidateGeometry::ScalarRows,
            },
            Candidate {
                id: "C1_SERIAL_PRECISE_FMA",
                kernel: C1_KERNEL,
                control: false,
                arithmetic: "one row/thread, increasing columns, explicit metal::precise::fma",
                geometry: CandidateGeometry::ScalarRows,
            },
            Candidate {
                id: "C2_STRIDED4_PRECISE_FMA",
                kernel: C2_KERNEL,
                control: false,
                arithmetic: "four interleaved FMA partials then deterministic lane0..lane3 fold",
                geometry: CandidateGeometry::ScalarRows,
            },
            Candidate {
                id: "C3_BLOCK128_PRECISE_FMA",
                kernel: C3_KERNEL,
                control: false,
                arithmetic: "thirty-two contiguous 128-product FMA partials then deterministic increasing-block fold",
                geometry: CandidateGeometry::ScalarRows,
            },
            Candidate {
                id: "C4_SIMD32_PRECISE_FMA",
                kernel: C4_KERNEL,
                control: false,
                arithmetic: "one 32-lane SIMDgroup/row; strided precise FMA partials then Metal simd_sum tree",
                geometry: CandidateGeometry::SimdgroupRows,
            },
            Candidate {
                id: "C5_STRIDED32_PRECISE_FMA_ORDERED",
                kernel: C5_KERNEL,
                control: false,
                arithmetic: "thirty-two interleaved FMA partials then deterministic increasing-lane fold",
                geometry: CandidateGeometry::ScalarRows,
            },
            Candidate {
                id: "C6_BLOCK256X16_PRECISE_FMA",
                kernel: C6_KERNEL,
                control: false,
                arithmetic: "sixteen contiguous 256-product FMA partials then deterministic increasing-block fold",
                geometry: CandidateGeometry::ScalarRows,
            },
            Candidate {
                id: "C7_BLOCK64X64_PRECISE_FMA",
                kernel: C7_KERNEL,
                control: false,
                arithmetic: "sixty-four contiguous 64-product FMA partials then deterministic increasing-block fold",
                geometry: CandidateGeometry::ScalarRows,
            },
        ];

        // Compile every candidate before reset of the trace counters. This
        // makes timing/topology evidence exclude lazy pipeline compilation.
        for candidate in &control_and_candidates {
            let pipeline = context.pipeline(candidate.kernel)?;
            validate_candidate_pipeline(candidate, &pipeline)?;
        }
        let input_buffer = context.new_buffer_with_bytes_checked(&trace.bf16_le)?;
        let weight_buffer = context.new_buffer_with_bytes_checked(&gate_weight)?;
        let output_buffer = context.new_buffer_checked(GATE_LOGIT_BYTES)?;
        let _ = context.drain_trace();
        let _ = context.drain_stats();

        let run_nonce = run_nonce(&reader, &trace, &calibration)?;
        let mut results = Vec::with_capacity(control_and_candidates.len());
        for (candidate_index, candidate) in control_and_candidates.iter().enumerate() {
            let result = execute_candidate(
                &context,
                candidate,
                &input_buffer,
                &weight_buffer,
                &output_buffer,
                &calibration.target_f32,
                &f64_authority.logits_f64,
                &v21_bounds,
                &run_nonce,
                candidate_index,
            )?;
            if candidate.control
                && (!result.clean_output_hashes_all_equal
                    || result
                        .clean_output_sha256_f32_le
                        .first()
                        .map_or(true, |hash| hash != &trace.baseline_gate_logits_sha256))
            {
                return Err(failure(
                    "isolated P6A Gate control did not reproduce the frozen trace Gate-logit hash; refusing to publish candidate results",
                ));
            }
            results.push(result);
        }

        let candidate_results = &results[1..];
        let all_candidate_v21_pass = candidate_results
            .iter()
            .all(|result| result.candidate_vs_live_fp64_pass);
        let any_candidate_v21_pass = candidate_results
            .iter()
            .any(|result| result.candidate_vs_live_fp64_pass);
        let all_candidate_hashes_stable = candidate_results
            .iter()
            .all(|result| result.clean_output_hashes_all_equal);

        let unsigned = json!({
            "schema": SWEEP_SCHEMA,
            "status": SWEEP_STATUS,
            "unsealed": true,
            "receipt_promoted": false,
            "is_runtime": false,
            "is_hcli": false,
            "base_true_tps": "not_evaluated",
            "claim_boundary": "A real-Metal, Gate-only reduction-association diagnostic over one frozen P0 BF16[4096] input and one admitted BF16[256,4096] Gate matrix. It does not execute P6 routing, experts, mHC, attention, a causal cache, a model layer, a generated token, an HCLI endpoint, or TPS benchmark. Candidate outputs are transient readbacks; this file retains only hashes and aggregate comparisons.",
            "artifact": {
                "path": reader.artifact_root().display().to_string(),
                "manifest_seal_sha256": reader.manifest_seal_sha256(),
                "manifest_file_sha256": reader.manifest_file_sha256(),
                "restart_receipt_seal_sha256": reader.restart_seal_sha256(),
                "source_repository": reader.source_identity().repository,
                "source_revision": reader.source_identity().revision,
                "source_parent_retained": false,
            },
            "frozen_input_trace": {
                "path": trace.path.display().to_string(),
                "file_sha256": trace.file_sha256,
                "schema": TRACE_SCHEMA,
                "layer": 0,
                "token_id": PREFIX_TOKEN_ID,
                "token_position": 0,
                "dtype": "BF16",
                "shape": [GATE_COLS],
                "byte_count": GATE_INPUT_BYTES,
                "sha256_bf16_le": trace.bf16_sha256,
                "p4a_predecessor": "P4A_EXACT_ATTENTION_ONLY as declared by the frozen trace; this isolated probe replays its post-completion Gate input and does not re-execute attention",
            },
            "source_calibration": {
                "path": calibration.path.display().to_string(),
                "file_sha256": calibration.file_sha256,
                "schema": CALIBRATION_SCHEMA,
                "status": calibration.status,
                "target_dtype": "F32",
                "target_shape": [GATE_ROWS],
                "target_byte_count": GATE_LOGIT_BYTES,
                "target_sha256_f32_le": calibration.target_sha256,
                "raw_target_retained_in_this_output": false,
            },
            "gate_tensor": {
                "name": LAYER0_FFN_GATE_WEIGHT,
                "dtype": "BF16",
                "shape": [GATE_ROWS, GATE_COLS],
                "byte_count": GATE_WEIGHT_BYTES,
                "sha256": trace.gate_weight_sha256,
            },
            "numeric_parity_v2_1": {
                "bounds": v21_bounds,
                "reference": "live independent FP64 Gate authority reconstructed from the admitted artifact and frozen BF16 input",
                "source_target_self_check": source_target_v21,
            },
            "direct_torch_compatibility": {
                "reference": "bounded raw F32[256] Torch F.linear calibration target, retained only in the input shard",
                "max_relative_reference_floor_1e_minus_12_ceiling": DIRECT_TORCH_MAX_REL,
                "relative_l2_ceiling": DIRECT_TORCH_REL_L2,
                "exact_greedy_argmax_required": true,
                "exact_top_k": DIRECT_TORCH_TOP_K,
                "meaning": "This is a separate source-target compatibility signal. It never replaces the independent FP64 Numeric Parity V2.1 gate.",
            },
            "compile": {
                "metal_context": "MetalContext::new_with_trace(true)",
                "compile_math_mode": "baseline_default_metal_compile_options",
                "candidate_kernels_are_not_registered_in_p6_or_any_runtime_seam": true,
                "candidate_shader_file": "crates/hawking-core/shaders/moe.metal",
            },
            "control": results[0],
            "candidates": candidate_results,
            "summary": {
                "candidate_count": candidate_results.len(),
                "clean_trials_per_candidate": CLEAN_TRIALS,
                "warmup_trials_per_candidate": WARMUP_TRIALS,
                "all_candidate_clean_hashes_stable": all_candidate_hashes_stable,
                "any_candidate_numeric_parity_v2_1_pass": any_candidate_v21_pass,
                "all_candidate_numeric_parity_v2_1_pass": all_candidate_v21_pass,
                "promotion_statement": "No candidate is promoted by this diagnostic. Any later P6 integration must preserve this exact trace/artifact/calibration binding and separately prove the whole P0 route, MoE, and child-state ladder.",
            },
            "storage_policy": {
                "raw_frozen_input_payloads_retained_in_output": 0,
                "raw_source_target_payloads_retained_in_output": 0,
                "raw_candidate_logits_retained_in_output": 0,
                "raw_weight_payloads_retained_in_output": 0,
                "retained_observations": "artifact/trace/calibration bindings, output hashes, dispatch/topology counters, timing summaries, and aggregate Numeric Parity V2.1 metrics only",
            },
        });
        let mut output = decimal_strings(unsigned);
        let canonical_unsigned_sha256 = sha256(&canonical_json(&output));
        output
            .as_object_mut()
            .ok_or_else(|| failure("sweep output root is not an object"))?
            .insert(
                "canonical_unsigned_sha256".to_owned(),
                Value::String(canonical_unsigned_sha256),
            );
        let rendered = String::from_utf8(canonical_json(&output))?;
        write_new_output(&args.out, &rendered)?;
        println!("{rendered}");
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_candidate(
        context: &MetalContext,
        candidate: &Candidate,
        input: &metal::Buffer,
        weights: &metal::Buffer,
        output: &metal::Buffer,
        source_target: &[f32],
        fp64_reference: &[f64],
        bounds: &Bounds,
        base_nonce: &str,
        candidate_index: usize,
    ) -> ProbeResult<CandidateResult> {
        let pipeline = context.pipeline(candidate.kernel)?;
        validate_candidate_pipeline(candidate, &pipeline)?;
        let (grid, tg) = candidate_geometry(candidate.geometry)?;
        let trace_identity = PhysicalTraceIdentity::new(
            sha256_join(&[base_nonce, candidate.id, "interval"]),
            sha256_join(&[base_nonce, candidate.id, "run"]),
            format!("p0_gate_{}", candidate.id.to_ascii_lowercase()),
            "isolated_gate_reduction".to_owned(),
            Some(1),
            candidate_index,
        )?;
        let _ = context.drain_trace();
        let _ = context.drain_stats();
        let physical_trace = PhysicalTraceGuard::begin(trace_identity)?;
        let mut all_timings = Vec::with_capacity(WARMUP_TRIALS + CLEAN_TRIALS);
        let mut clean_timings = Vec::with_capacity(CLEAN_TRIALS);
        let mut clean_hashes = Vec::with_capacity(CLEAN_TRIALS);
        let mut first_clean_output = None;
        for trial in 0..(WARMUP_TRIALS + CLEAN_TRIALS) {
            let timing = context.dispatch_batch_timed(|batch| {
                dispatch_gate_candidate(batch, candidate.kernel, input, weights, output, grid, tg)
            })?;
            let values = read_gpu_f32(output, GATE_ROWS)?;
            if trial >= WARMUP_TRIALS {
                let output_hash = sha256(&f32_le_bytes(&values));
                clean_hashes.push(output_hash);
                if first_clean_output.is_none() {
                    first_clean_output = Some(values);
                }
                clean_timings.push(timing);
            }
            all_timings.push(timing);
        }
        let physical = physical_trace.counts();
        drop(physical_trace);
        let trace_samples = context.drain_trace();
        let (buffers_created, bytes_allocated, commits) = context.drain_stats();
        let candidate_output =
            first_clean_output.ok_or_else(|| failure("candidate recorded no clean output"))?;
        let clean_output_hashes_all_equal = clean_hashes
            .first()
            .is_some_and(|first| clean_hashes.iter().all(|value| value == first));
        let paired_v21 = score_pair(source_target, &candidate_output, fp64_reference, bounds);
        let source_target_f32_comparison = f32_pair_metrics(source_target, &candidate_output)?;
        let direct_torch_compatibility = clean_output_hashes_all_equal
            && source_target_f32_comparison.max_relative_reference_floor_1e_minus_12
                <= DIRECT_TORCH_MAX_REL
            && source_target_f32_comparison.relative_l2 <= DIRECT_TORCH_REL_L2
            && source_target_f32_comparison.greedy_argmax_exact_match
            && source_target_f32_comparison.top_k_exact_match;
        let command_buffers = all_timings
            .iter()
            .map(|timing| timing.command_buffers)
            .sum::<u64>();
        let compute_encoders = all_timings
            .iter()
            .map(|timing| timing.compute_encoders)
            .sum::<u64>();
        let compute_dispatches = all_timings
            .iter()
            .map(|timing| timing.compute_dispatches)
            .sum::<u64>();
        let expected_runs = (WARMUP_TRIALS + CLEAN_TRIALS) as u64;
        if command_buffers != expected_runs
            || compute_encoders != expected_runs
            || compute_dispatches != expected_runs
            || physical.command_count != expected_runs
            || physical.encoder_count != expected_runs
            || commits as u64 != expected_runs
            || buffers_created != 0
            || bytes_allocated != 0
            || trace_samples.len() != expected_runs as usize
        {
            return Err(failure(format!(
                "{} topology accounting differs from its one-dispatch-per-run contract",
                candidate.id
            )));
        }
        let trace_samples_all_dispatch_batches = trace_samples
            .iter()
            .all(|sample| sample.kernel_name == "dispatch_batch");
        if !trace_samples_all_dispatch_batches {
            return Err(failure(format!(
                "{} received an unexpected trace sample outside its isolated batch dispatches",
                candidate.id
            )));
        }
        let source_target_vs_live_fp64_pass = paired_v21.host.pass;
        let candidate_vs_live_fp64_pass = paired_v21.device.pass;
        Ok(CandidateResult {
            id: candidate.id,
            kernel: candidate.kernel,
            control: candidate.control,
            arithmetic: candidate.arithmetic,
            geometry: CandidateGeometryReport {
                grid_threads: [grid.0, grid.1, grid.2],
                threads_per_threadgroup: [tg.0, tg.1, tg.2],
                thread_execution_width: pipeline.thread_execution_width() as u64,
                max_total_threads_per_threadgroup: pipeline.max_total_threads_per_threadgroup()
                    as u64,
                simdgroup_width_required: matches!(
                    candidate.geometry,
                    CandidateGeometry::SimdgroupRows
                )
                .then_some(SIMDGROUP_WIDTH),
            },
            clean_output_sha256_f32_le: clean_hashes,
            clean_output_hashes_all_equal,
            source_target_f32_comparison,
            direct_torch_compatibility_under_declared_f32_bounds: direct_torch_compatibility,
            numeric_parity_v2_1_source_and_device_vs_live_fp64: paired_v21,
            source_target_vs_live_fp64_pass,
            candidate_vs_live_fp64_pass,
            promotion_eligible_within_this_probe: !candidate.control
                && clean_output_hashes_all_equal
                && candidate_vs_live_fp64_pass
                && direct_torch_compatibility,
            timing: timing_summary(&clean_timings)?,
            topology: CandidateTopology {
                command_buffers,
                compute_encoders,
                compute_dispatches,
                cpu_visible_waits: expected_runs,
                expected_runs,
                physical,
                physical_matches_command_topology: physical.command_count == command_buffers
                    && physical.encoder_count == compute_encoders,
                metal_trace_samples: trace_samples.len(),
                metal_trace_samples_all_dispatch_batches: trace_samples_all_dispatch_batches,
                context_commits: commits,
                context_buffers_created_during_runs: buffers_created,
                context_bytes_allocated_during_runs: bytes_allocated,
            },
            fallback: false,
            raw_candidate_logits_retained: 0,
        })
    }

    fn candidate_geometry(
        geometry: CandidateGeometry,
    ) -> ProbeResult<((u32, u32, u32), (u32, u32, u32))> {
        let rows = u32::try_from(GATE_ROWS).map_err(|_| failure("Gate rows do not fit u32"))?;
        match geometry {
            CandidateGeometry::ScalarRows => Ok(((rows, 1, 1), (SIMDGROUP_WIDTH, 1, 1))),
            CandidateGeometry::SimdgroupRows => Ok((
                (
                    rows.checked_mul(SIMDGROUP_WIDTH)
                        .ok_or_else(|| failure("SIMDgroup Gate grid overflow"))?,
                    1,
                    1,
                ),
                (SIMDGROUP_WIDTH, 1, 1),
            )),
        }
    }

    fn validate_candidate_pipeline(
        candidate: &Candidate,
        pipeline: &metal::ComputePipelineState,
    ) -> ProbeResult<()> {
        if pipeline.max_total_threads_per_threadgroup() < u64::from(SIMDGROUP_WIDTH) {
            return Err(failure(format!(
                "{} cannot admit its required {}-thread group",
                candidate.kernel, SIMDGROUP_WIDTH
            )));
        }
        if matches!(candidate.geometry, CandidateGeometry::SimdgroupRows)
            && pipeline.thread_execution_width() != u64::from(SIMDGROUP_WIDTH)
        {
            return Err(failure(format!(
                "{} requires a 32-thread SIMDgroup but reports execution width {}",
                candidate.kernel,
                pipeline.thread_execution_width()
            )));
        }
        Ok(())
    }

    fn dispatch_gate_candidate(
        batch: &mut hawking_core::metal::CommandBatch<'_>,
        kernel: &str,
        input: &metal::Buffer,
        weights: &metal::Buffer,
        output: &metal::Buffer,
        grid: (u32, u32, u32),
        tg: (u32, u32, u32),
    ) -> hawking_core::Result<()> {
        let rows = GATE_ROWS as u32;
        let cols = GATE_COLS as u32;
        batch.dispatch_threads(kernel, grid, tg, |encoder| {
            encoder.set_buffer(0, Some(weights), 0);
            encoder.set_buffer(1, Some(input), 0);
            encoder.set_buffer(2, Some(output), 0);
            set_u32(encoder, 3, &rows);
            set_u32(encoder, 4, &cols);
        })
    }

    fn load_trace(path: &Path) -> ProbeResult<TraceInput> {
        let raw = read_absolute_regular(path, "P0 Gate input trace")?;
        let file_sha256 = sha256(&raw);
        let value: Value = serde_json::from_slice(&raw)?;
        expect_string(&value, &["schema"], "trace schema", TRACE_SCHEMA)?;
        expect_string(&value, &["status"], "trace status", TRACE_STATUS)?;
        expect_u64(&value, &["trace_binding", "layer"], "trace layer", 0)?;
        expect_u64(
            &value,
            &["trace_binding", "token_id"],
            "trace token id",
            PREFIX_TOKEN_ID,
        )?;
        expect_u64(
            &value,
            &["trace_binding", "token_position"],
            "trace token position",
            0,
        )?;
        expect_bool(
            &value,
            &["trace_binding", "post_completion_readback_only"],
            "trace post-completion flag",
            true,
        )?;
        expect_bool(
            &value,
            &["trace_binding", "trace_does_not_feed_graph"],
            "trace non-feeding flag",
            true,
        )?;
        expect_bool(
            &value,
            &["trace_binding", "p4a_attention_predecessor_exact"],
            "trace P4A exact flag",
            true,
        )?;
        expect_string(
            &value,
            &["trace_binding", "p4a_attention_predecessor_label"],
            "trace P4A exact label",
            "P4A_EXACT_ATTENTION_ONLY",
        )?;
        expect_bool(
            &value,
            &[
                "trace_binding",
                "real_graph_completed_before_trace_emission",
            ],
            "trace real graph completion flag",
            true,
        )?;
        expect_bool(
            &value,
            &["trace_binding", "trace_does_not_modify_graph_counters"],
            "trace non-mutating counter flag",
            true,
        )?;
        let raw_payload = required_object(&value, &["raw_payload"], "trace raw payload")?;
        expect_string(raw_payload, &["dtype"], "trace raw dtype", "BF16")?;
        expect_string(
            raw_payload,
            &["byte_order"],
            "trace raw byte order",
            "little_endian",
        )?;
        expect_string(
            raw_payload,
            &["encoding"],
            "trace raw encoding",
            "lowercase_hex_raw_bf16_le",
        )?;
        expect_shape(
            raw_payload,
            &["shape"],
            &[GATE_COLS as u64],
            "trace raw shape",
        )?;
        expect_u64(
            raw_payload,
            &["element_count"],
            "trace raw element count",
            GATE_COLS as u64,
        )?;
        expect_u64(
            raw_payload,
            &["byte_count"],
            "trace raw byte count",
            GATE_INPUT_BYTES as u64,
        )?;
        let bf16_sha256 = required_string(raw_payload, &["sha256"], "trace raw SHA-256")?;
        let bf16_le = decode_lowercase_hex(
            &required_string(raw_payload, &["data"], "trace raw data")?,
            GATE_INPUT_BYTES,
            "trace raw BF16 payload",
        )?;
        if sha256(&bf16_le) != bf16_sha256 {
            return Err(failure("trace raw BF16 payload SHA-256 mismatch"));
        }
        expect_string(
            &value,
            &["input_output_sha256", "p6_gate_input_ffn_norm_bf16_le"],
            "trace P6 Gate input SHA-256",
            &bf16_sha256,
        )?;
        let baseline_gate_logits_sha256 = required_string(
            &value,
            &["input_output_sha256", "p6_gate_output_logits_f32_le"],
            "trace baseline Gate output SHA-256",
        )?;
        let artifact = parse_artifact_binding(&value, &["artifact"], "trace artifact")?;
        expect_bool(
            &value,
            &["artifact", "source_parent_retained"],
            "trace source parent retained flag",
            false,
        )?;
        let gate_weight_name = required_string(
            &value,
            &["model_source", "p6_gate_route_bindings", "gate_weight_name"],
            "trace Gate tensor name",
        )?;
        let gate_weight_sha256 = required_string(
            &value,
            &[
                "model_source",
                "p6_gate_route_bindings",
                "gate_weight_sha256",
            ],
            "trace Gate tensor SHA-256",
        )?;
        if gate_weight_name != LAYER0_FFN_GATE_WEIGHT {
            return Err(failure(
                "trace binds a Gate tensor other than layers.0.ffn.gate.weight",
            ));
        }
        expect_u64(
            &value,
            &["privacy_and_storage_bound", "raw_payload_count"],
            "trace raw payload count",
            1,
        )?;
        for field in [
            "raw_source_weight_payloads",
            "raw_other_activation_payloads",
            "raw_gate_output_payloads",
            "raw_route_weight_payloads",
        ] {
            expect_u64(
                &value,
                &["privacy_and_storage_bound", field],
                "trace privacy zero-payload field",
                0,
            )?;
        }
        Ok(TraceInput {
            file_sha256,
            path: path.to_owned(),
            artifact,
            bf16_le,
            bf16_sha256,
            baseline_gate_logits_sha256,
            gate_weight_name,
            gate_weight_sha256,
        })
    }

    fn load_source_calibration(path: &Path, trace: &TraceInput) -> ProbeResult<SourceCalibration> {
        let raw = read_absolute_regular(path, "P0 Gate source calibration")?;
        let file_sha256 = sha256(&raw);
        let value: Value = serde_json::from_slice(&raw)?;
        expect_string(
            &value,
            &["schema"],
            "source calibration schema",
            CALIBRATION_SCHEMA,
        )?;
        expect_bool(
            &value,
            &["unsealed"],
            "source calibration unsealed flag",
            true,
        )?;
        expect_bool(
            &value,
            &["receipt_promoted"],
            "source calibration receipt-promoted flag",
            false,
        )?;
        expect_string(
            &value,
            &["status"],
            "source calibration status",
            CALIBRATION_STATUS,
        )?;
        if count_object_key(&value, "data") != 1 {
            return Err(failure(
                "source calibration must contain exactly one raw data field: its bounded F32[256] target",
            ));
        }
        let status = CALIBRATION_STATUS.to_owned();
        expect_string(
            &value,
            &["trace_binding", "schema"],
            "source calibration trace schema",
            TRACE_SCHEMA,
        )?;
        expect_string(
            &value,
            &["trace_binding", "status"],
            "source calibration trace status",
            TRACE_STATUS,
        )?;
        expect_bool(
            &value,
            &["trace_binding", "immutable_existing_trace"],
            "source calibration immutable trace flag",
            true,
        )?;
        expect_bool(
            &value,
            &["trace_binding", "raw_p0_input_copied_into_this_shard"],
            "source calibration no copied input flag",
            false,
        )?;
        let trace_file_sha256 = required_string(
            &value,
            &["trace_binding", "file_sha256"],
            "source calibration trace file SHA-256",
        )?;
        let trace_input_sha256 = required_string(
            &value,
            &["trace_binding", "raw_gate_input_payload_sha256"],
            "source calibration trace input SHA-256",
        )?;
        if trace_file_sha256 != trace.file_sha256 || trace_input_sha256 != trace.bf16_sha256 {
            return Err(failure(
                "source calibration is not bound to the supplied frozen Gate trace/input",
            ));
        }
        expect_string(
            &value,
            &["trace_binding", "recorded_metal_gate_logits_f32_le_sha256"],
            "source calibration frozen baseline Gate-logit SHA-256",
            &trace.baseline_gate_logits_sha256,
        )?;
        expect_u64(
            &value,
            &["trace_binding", "layer"],
            "source calibration layer",
            0,
        )?;
        expect_u64(
            &value,
            &["trace_binding", "token_id"],
            "source calibration token id",
            PREFIX_TOKEN_ID,
        )?;
        expect_u64(
            &value,
            &["trace_binding", "token_position"],
            "source calibration token position",
            0,
        )?;
        let artifact =
            parse_artifact_binding(&value, &["artifact_binding"], "source calibration artifact")?;
        let gate_binding = value
            .get("source_tensor_binding")
            .or_else(|| value.pointer("/source_tensor_bindings/gate"))
            .ok_or_else(|| failure("source calibration is missing source_tensor_binding/gate"))?;
        expect_string(
            gate_binding,
            &["name"],
            "source calibration Gate tensor name",
            LAYER0_FFN_GATE_WEIGHT,
        )?;
        expect_string(
            gate_binding,
            &["dtype"],
            "source calibration Gate tensor dtype",
            "BF16",
        )?;
        expect_shape(
            gate_binding,
            &["shape"],
            &[GATE_ROWS as u64, GATE_COLS as u64],
            "source calibration Gate tensor shape",
        )?;
        expect_u64(
            gate_binding,
            &["bytes"],
            "source calibration Gate tensor bytes",
            GATE_WEIGHT_BYTES as u64,
        )?;
        let gate_weight_name =
            required_string(gate_binding, &["name"], "source calibration Gate name")?;
        let gate_weight_sha256 = required_string(
            gate_binding,
            &["logical_tensor_sha256"],
            "source calibration Gate SHA-256",
        )?;
        if gate_weight_name != trace.gate_weight_name
            || gate_weight_sha256 != trace.gate_weight_sha256
        {
            return Err(failure(
                "source calibration Gate tensor binding differs from the frozen trace binding",
            ));
        }
        let raw_target = required_object(&value, &["raw_f32_le"], "source calibration raw target")?;
        expect_string(raw_target, &["dtype"], "source target dtype", "F32")?;
        expect_string(
            raw_target,
            &["byte_order"],
            "source target byte order",
            "little_endian",
        )?;
        expect_string(
            raw_target,
            &["encoding"],
            "source target encoding",
            "lowercase_hex_raw_f32_le",
        )?;
        expect_shape(
            raw_target,
            &["shape"],
            &[GATE_ROWS as u64],
            "source target shape",
        )?;
        expect_u64(
            raw_target,
            &["element_count"],
            "source target element count",
            GATE_ROWS as u64,
        )?;
        expect_u64(
            raw_target,
            &["byte_count"],
            "source target byte count",
            GATE_LOGIT_BYTES as u64,
        )?;
        let target_sha256 = required_string(raw_target, &["sha256"], "source target SHA-256")?;
        let target_bytes = decode_lowercase_hex(
            &required_string(raw_target, &["data"], "source target data")?,
            GATE_LOGIT_BYTES,
            "source F32 target",
        )?;
        if sha256(&target_bytes) != target_sha256 {
            return Err(failure("source target F32 SHA-256 mismatch"));
        }
        let target_f32 = decode_f32_le(&target_bytes)?;
        if target_f32.len() != GATE_ROWS || target_f32.iter().any(|value| !value.is_finite()) {
            return Err(failure("source target is not a finite F32[256] vector"));
        }
        Ok(SourceCalibration {
            file_sha256,
            path: path.to_owned(),
            status,
            artifact,
            trace_file_sha256,
            trace_input_sha256,
            gate_weight_name,
            gate_weight_sha256,
            target_f32,
            target_sha256,
        })
    }

    fn validate_artifact_bindings(
        reader: &DeepSeekV4FullStreamReader,
        trace: &TraceInput,
        calibration: &SourceCalibration,
    ) -> ProbeResult<()> {
        for (label, binding) in [
            ("frozen trace", &trace.artifact),
            ("source calibration", &calibration.artifact),
        ] {
            if binding.manifest_seal_sha256 != reader.manifest_seal_sha256()
                || binding.manifest_file_sha256 != reader.manifest_file_sha256()
                || binding.restart_receipt_seal_sha256 != reader.restart_seal_sha256()
            {
                return Err(failure(format!(
                    "{label} artifact binding does not match the admitted full Gravity stream",
                )));
            }
        }
        if calibration.trace_file_sha256 != trace.file_sha256
            || calibration.trace_input_sha256 != trace.bf16_sha256
            || calibration.gate_weight_name != trace.gate_weight_name
            || calibration.gate_weight_sha256 != trace.gate_weight_sha256
        {
            return Err(failure(
                "calibration-to-trace binding changed after calibration parsing",
            ));
        }
        Ok(())
    }

    fn load_verified_gate_weight(
        reader: &DeepSeekV4FullStreamReader,
        trace: &TraceInput,
        calibration: &SourceCalibration,
    ) -> ProbeResult<Vec<u8>> {
        let metadata = reader.tensor_metadata(LAYER0_FFN_GATE_WEIGHT)?;
        if metadata.dtype != "BF16"
            || metadata.shape.as_slice() != [GATE_ROWS as u64, GATE_COLS as u64]
            || metadata.bytes != GATE_WEIGHT_BYTES as u64
        {
            return Err(failure(
                "admitted Gate tensor metadata differs from BF16[256,4096]",
            ));
        }
        let bytes = reader.read_verified_full(LAYER0_FFN_GATE_WEIGHT, GATE_WEIGHT_BYTES)?;
        if bytes.len() != GATE_WEIGHT_BYTES {
            return Err(failure("verified Gate matrix length is invalid"));
        }
        let digest = sha256(&bytes);
        if digest != trace.gate_weight_sha256 || digest != calibration.gate_weight_sha256 {
            return Err(failure(
                "verified Gate matrix SHA-256 differs from trace/calibration",
            ));
        }
        Ok(bytes)
    }

    fn parse_artifact_binding(
        value: &Value,
        path: &[&str],
        label: &str,
    ) -> ProbeResult<ArtifactBinding> {
        let binding = required_object(value, path, label)?;
        Ok(ArtifactBinding {
            manifest_seal_sha256: required_string(
                binding,
                &["manifest_seal_sha256"],
                "artifact manifest seal",
            )?,
            manifest_file_sha256: required_string(
                binding,
                &["manifest_file_sha256"],
                "artifact manifest file hash",
            )?,
            restart_receipt_seal_sha256: required_string(
                binding,
                &["restart_receipt_seal_sha256"],
                "artifact restart receipt seal",
            )?,
        })
    }

    fn gate_v21_bounds() -> Bounds {
        Bounds {
            max_meaningful_rel: 1.0e-4,
            ..Bounds::continuous_only()
        }
    }

    fn timing_summary(clean: &[MetalBatchTiming]) -> ProbeResult<TimingSummary> {
        if clean.len() != CLEAN_TRIALS {
            return Err(failure(
                "candidate did not retain exactly five clean timings",
            ));
        }
        let gpu: Vec<u64> = clean
            .iter()
            .filter_map(|timing| timing.gpu_duration_us)
            .collect();
        Ok(TimingSummary {
            warmup_trials: WARMUP_TRIALS,
            clean_trials: CLEAN_TRIALS,
            clean_host_wall_us: distribution(clean.iter().map(|timing| timing.host_wall_us))?,
            clean_wait_us: distribution(clean.iter().map(|timing| timing.wait_us))?,
            clean_encode_us: distribution(clean.iter().map(|timing| timing.encode_us))?,
            clean_submit_us: distribution(clean.iter().map(|timing| timing.submit_us))?,
            clean_pipeline_lookup_us: distribution(
                clean.iter().map(|timing| timing.pipeline_lookup_us),
            )?,
            clean_gpu_duration_us: OptionalDistribution {
                available_samples: gpu.len(),
                distribution: if gpu.is_empty() {
                    None
                } else {
                    Some(distribution(gpu)?)
                },
            },
        })
    }

    fn distribution(values: impl IntoIterator<Item = u64>) -> ProbeResult<Distribution> {
        let mut values = values.into_iter().collect::<Vec<_>>();
        if values.is_empty() {
            return Err(failure("cannot summarize an empty timing series"));
        }
        values.sort_unstable();
        let n = values.len();
        let percentile = |fraction: f64| -> u64 {
            let index = ((fraction * (n.saturating_sub(1)) as f64).round() as usize).min(n - 1);
            values[index]
        };
        Ok(Distribution {
            samples: n,
            min: values[0],
            p50: percentile(0.50),
            p95: percentile(0.95),
            p99: percentile(0.99),
            max: values[n - 1],
        })
    }

    fn f32_pair_metrics(reference: &[f32], candidate: &[f32]) -> ProbeResult<F32PairMetrics> {
        if reference.len() != candidate.len() || reference.is_empty() {
            return Err(failure("F32 comparison needs equal non-empty vectors"));
        }
        let mut mismatch = 0usize;
        let mut max_abs = 0.0_f64;
        let mut max_rel = 0.0_f64;
        let mut numerator = 0.0_f64;
        let mut denominator = 0.0_f64;
        let mut ulps = Vec::with_capacity(reference.len());
        for (&reference, &candidate) in reference.iter().zip(candidate) {
            if !reference.is_finite() || !candidate.is_finite() {
                return Err(failure("F32 comparison received a non-finite value"));
            }
            if reference.to_bits() != candidate.to_bits() {
                mismatch += 1;
            }
            let difference = (f64::from(candidate) - f64::from(reference)).abs();
            max_abs = max_abs.max(difference);
            max_rel = max_rel.max(difference / f64::from(reference).abs().max(1.0e-12));
            numerator += difference * difference;
            denominator += f64::from(reference) * f64::from(reference);
            ulps.push(ulp_distance_f32(reference, candidate) as f64);
        }
        ulps.sort_by(f64::total_cmp);
        let n = ulps.len();
        let percentile = |fraction: f64| -> f64 {
            let index = ((fraction * (n.saturating_sub(1)) as f64).round() as usize).min(n - 1);
            ulps[index]
        };
        let greedy_argmax_reference = descending_indices(reference, 1)?[0];
        let greedy_argmax_candidate = descending_indices(candidate, 1)?[0];
        let top_k_reference = descending_indices(reference, DIRECT_TORCH_TOP_K)?;
        let top_k_candidate = descending_indices(candidate, DIRECT_TORCH_TOP_K)?;
        Ok(F32PairMetrics {
            elements: reference.len(),
            f32_bit_exact: mismatch == 0,
            f32_bit_mismatch_elements: mismatch,
            max_abs,
            max_relative_reference_floor_1e_minus_12: max_rel,
            relative_l2: numerator.sqrt() / denominator.sqrt().max(f64::MIN_POSITIVE),
            ulp_median: percentile(0.50),
            ulp_p95: percentile(0.95),
            ulp_p99: percentile(0.99),
            ulp_max: ulps[n - 1],
            greedy_argmax_reference,
            greedy_argmax_candidate,
            greedy_argmax_exact_match: greedy_argmax_reference == greedy_argmax_candidate,
            top_k: DIRECT_TORCH_TOP_K,
            top_k_exact_match: top_k_reference == top_k_candidate,
            top_k_reference,
            top_k_candidate,
        })
    }

    fn descending_indices(values: &[f32], top_k: usize) -> ProbeResult<Vec<usize>> {
        if top_k == 0 || top_k > values.len() || values.iter().any(|value| !value.is_finite()) {
            return Err(failure("cannot rank an invalid F32 Gate vector"));
        }
        let mut indices = (0..values.len()).collect::<Vec<_>>();
        indices.sort_by(|&left, &right| {
            values[right]
                .total_cmp(&values[left])
                .then_with(|| left.cmp(&right))
        });
        indices.truncate(top_k);
        Ok(indices)
    }

    fn read_gpu_f32(buffer: &metal::Buffer, count: usize) -> ProbeResult<Vec<f32>> {
        let bytes = count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| failure("GPU F32 readback size overflow"))?;
        if buffer.length() < bytes as u64 {
            return Err(failure("GPU output buffer is too small for Gate logits"));
        }
        let values =
            unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, count) }.to_vec();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(failure("GPU Gate output includes a non-finite value"));
        }
        Ok(values)
    }

    fn decode_bf16_le(bytes: &[u8]) -> ProbeResult<Vec<u16>> {
        if bytes.len() != GATE_INPUT_BYTES {
            return Err(failure("BF16 Gate input byte length is invalid"));
        }
        Ok(bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect())
    }

    fn decode_f32_le(bytes: &[u8]) -> ProbeResult<Vec<f32>> {
        if bytes.len() != GATE_LOGIT_BYTES {
            return Err(failure("F32 Gate target byte length is invalid"));
        }
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| {
                f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            })
            .collect::<Vec<_>>();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(failure("F32 Gate target includes a non-finite value"));
        }
        Ok(values)
    }

    fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn run_nonce(
        reader: &DeepSeekV4FullStreamReader,
        trace: &TraceInput,
        calibration: &SourceCalibration,
    ) -> ProbeResult<String> {
        let unix_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| failure("system clock precedes Unix epoch"))?
            .as_nanos()
            .to_string();
        Ok(sha256_join(&[
            reader.manifest_seal_sha256(),
            &trace.file_sha256,
            &calibration.file_sha256,
            &std::process::id().to_string(),
            &unix_ns,
            "dsv4f-p0-gate-reduction-sweep-v1",
        ]))
    }

    fn parse_args() -> ProbeResult<Args> {
        let mut artifact = None;
        let mut trace = None;
        let mut source_calibration = None;
        let mut out = None;
        let mut args = std::env::args_os().skip(1);
        while let Some(flag) = args.next() {
            match flag.to_string_lossy().as_ref() {
                "--artifact" => artifact = args.next().map(PathBuf::from),
                "--trace" => trace = args.next().map(PathBuf::from),
                "--source-calibration" => source_calibration = args.next().map(PathBuf::from),
                "--out" => out = args.next().map(PathBuf::from),
                "--help" | "-h" => {
                    println!(
                        "usage: gravity_deepseek_v4_gate_reduction_sweep --artifact <absolute full Gravity dir> --trace <absolute P0 Gate trace.json> --source-calibration <absolute Torch F32 target shard.json> --out <absolute new unsealed diagnostic.json>"
                    );
                    std::process::exit(0);
                }
                other => return Err(failure(format!("unknown argument {other}"))),
            }
        }
        let artifact = artifact.ok_or_else(|| failure("--artifact is required"))?;
        let trace = trace.ok_or_else(|| failure("--trace is required"))?;
        let source_calibration =
            source_calibration.ok_or_else(|| failure("--source-calibration is required"))?;
        let out = out.ok_or_else(|| failure("--out is required"))?;
        for (flag, path) in [
            ("--artifact", &artifact),
            ("--trace", &trace),
            ("--source-calibration", &source_calibration),
            ("--out", &out),
        ] {
            if !path.is_absolute() {
                return Err(failure(format!("{flag} must be an absolute path")));
            }
        }
        Ok(Args {
            artifact,
            trace,
            source_calibration,
            out,
        })
    }

    fn read_absolute_regular(path: &Path, label: &str) -> ProbeResult<Vec<u8>> {
        if !path.is_absolute() {
            return Err(failure(format!("{label} path must be absolute")));
        }
        let metadata = fs::symlink_metadata(path)?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            return Err(failure(format!(
                "{label} must be a regular non-symlink file"
            )));
        }
        Ok(fs::read(path)?)
    }

    fn validate_new_output_path(path: &Path) -> ProbeResult<()> {
        if !path.is_absolute() {
            return Err(failure("--out must be absolute"));
        }
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .ok_or_else(|| failure("--out requires a parent directory"))?;
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty() && *name != "." && *name != "..")
            .ok_or_else(|| failure("--out must name a normal UTF-8 file"))?;
        if name.is_empty() || path.exists() {
            return Err(failure(format!(
                "refusing to overwrite existing Gate reduction sweep output {}",
                path.display()
            )));
        }
        if parent.exists() {
            let metadata = fs::symlink_metadata(parent)?;
            if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
                return Err(failure("--out parent must be a non-symlink directory"));
            }
        }
        Ok(())
    }

    fn write_new_output(path: &Path, rendered: &str) -> ProbeResult<()> {
        validate_new_output_path(path)?;
        let parent = path
            .parent()
            .ok_or_else(|| failure("output path has no parent"))?;
        fs::create_dir_all(parent)?;
        let parent_metadata = fs::symlink_metadata(parent)?;
        if parent_metadata.file_type().is_symlink() || !parent_metadata.file_type().is_dir() {
            return Err(failure("output parent became an unsafe directory"));
        }
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| failure("output filename is not UTF-8"))?;
        let temporary = parent.join(format!(".{name}.{}.gate-sweep.tmp", std::process::id()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        if let Err(error) = file
            .write_all(rendered.as_bytes())
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
                "atomically publish Gate sweep output: {error}"
            )));
        }
        fs::remove_file(&temporary)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    }

    fn required_value<'a>(value: &'a Value, path: &[&str], label: &str) -> ProbeResult<&'a Value> {
        let mut current = value;
        for key in path {
            current = current
                .get(*key)
                .ok_or_else(|| failure(format!("missing {label} at {}", path.join("."))))?;
        }
        Ok(current)
    }

    fn required_object<'a>(value: &'a Value, path: &[&str], label: &str) -> ProbeResult<&'a Value> {
        let object = required_value(value, path, label)?;
        if !object.is_object() {
            return Err(failure(format!("{label} must be a JSON object")));
        }
        Ok(object)
    }

    fn required_string(value: &Value, path: &[&str], label: &str) -> ProbeResult<String> {
        required_value(value, path, label)?
            .as_str()
            .filter(|text| !text.is_empty())
            .map(str::to_owned)
            .ok_or_else(|| failure(format!("{label} must be a non-empty string")))
    }

    fn expect_string(value: &Value, path: &[&str], label: &str, expected: &str) -> ProbeResult<()> {
        let observed = required_string(value, path, label)?;
        if observed != expected {
            return Err(failure(format!(
                "{label} mismatch: expected {expected:?}, got {observed:?}"
            )));
        }
        Ok(())
    }

    fn expect_u64(value: &Value, path: &[&str], label: &str, expected: u64) -> ProbeResult<()> {
        let observed = required_value(value, path, label)?
            .as_u64()
            .ok_or_else(|| failure(format!("{label} must be an unsigned integer")))?;
        if observed != expected {
            return Err(failure(format!(
                "{label} mismatch: expected {expected}, got {observed}"
            )));
        }
        Ok(())
    }

    fn expect_bool(value: &Value, path: &[&str], label: &str, expected: bool) -> ProbeResult<()> {
        let observed = required_value(value, path, label)?
            .as_bool()
            .ok_or_else(|| failure(format!("{label} must be boolean")))?;
        if observed != expected {
            return Err(failure(format!(
                "{label} mismatch: expected {expected}, got {observed}"
            )));
        }
        Ok(())
    }

    fn expect_shape(
        value: &Value,
        path: &[&str],
        expected: &[u64],
        label: &str,
    ) -> ProbeResult<()> {
        let observed = required_value(value, path, label)?
            .as_array()
            .ok_or_else(|| failure(format!("{label} must be an integer array")))?
            .iter()
            .map(|dimension| {
                dimension
                    .as_u64()
                    .ok_or_else(|| failure(format!("{label} contains a non-integer dimension")))
            })
            .collect::<ProbeResult<Vec<_>>>()?;
        if observed.as_slice() != expected {
            return Err(failure(format!(
                "{label} mismatch: expected {expected:?}, got {observed:?}"
            )));
        }
        Ok(())
    }

    fn decode_lowercase_hex(
        input: &str,
        expected_bytes: usize,
        label: &str,
    ) -> ProbeResult<Vec<u8>> {
        if input.len() != expected_bytes * 2
            || !input
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(failure(format!(
                "{label} must be exactly lowercase hex for {expected_bytes} bytes"
            )));
        }
        input
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let high = hex_nibble(pair[0])
                    .ok_or_else(|| failure(format!("{label} contains invalid hex")))?;
                let low = hex_nibble(pair[1])
                    .ok_or_else(|| failure(format!("{label} contains invalid hex")))?;
                Ok((high << 4) | low)
            })
            .collect()
    }

    fn hex_nibble(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            _ => None,
        }
    }

    fn count_object_key(value: &Value, key: &str) -> usize {
        match value {
            Value::Array(values) => values
                .iter()
                .map(|value| count_object_key(value, key))
                .sum(),
            Value::Object(values) => {
                usize::from(values.contains_key(key))
                    + values
                        .values()
                        .map(|value| count_object_key(value, key))
                        .sum::<usize>()
            }
            _ => 0,
        }
    }

    fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &u32) {
        encoder.set_bytes(
            index,
            std::mem::size_of::<u32>() as u64,
            value as *const u32 as *const _,
        );
    }

    fn decimal_strings(value: Value) -> Value {
        match value {
            Value::Number(number) if number.is_i64() || number.is_u64() => Value::Number(number),
            Value::Number(number) => Value::String(number.to_string()),
            Value::Array(values) => Value::Array(values.into_iter().map(decimal_strings).collect()),
            Value::Object(values) => Value::Object(
                values
                    .into_iter()
                    .map(|(key, value)| (key, decimal_strings(value)))
                    .collect(),
            ),
            other => other,
        }
    }

    fn canonical_json(value: &Value) -> Vec<u8> {
        let mut output = Vec::new();
        write_canonical_json(&mut output, value);
        output
    }

    fn write_canonical_json(output: &mut Vec<u8>, value: &Value) {
        match value {
            Value::Null => output.extend_from_slice(b"null"),
            Value::Bool(value) => output.extend_from_slice(if *value { b"true" } else { b"false" }),
            Value::Number(value) => output.extend_from_slice(value.to_string().as_bytes()),
            Value::String(value) => {
                output.extend_from_slice(serde_json::to_string(value).unwrap().as_bytes())
            }
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
            Value::Object(values) => {
                output.push(b'{');
                let mut keys = values.keys().collect::<Vec<_>>();
                keys.sort();
                for (index, key) in keys.into_iter().enumerate() {
                    if index != 0 {
                        output.push(b',');
                    }
                    output.extend_from_slice(serde_json::to_string(key).unwrap().as_bytes());
                    output.push(b':');
                    write_canonical_json(output, &values[key]);
                }
                output.push(b'}');
            }
        }
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

    fn failure(message: impl Into<String>) -> Box<dyn Error> {
        std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

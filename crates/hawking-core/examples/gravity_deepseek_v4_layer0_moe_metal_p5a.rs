//! Bounded native-Metal DeepSeek-V4-Flash layer-0 MoE substage (P5A).
//!
//! This is intentionally a *component boundary*, not a decoder/runtime claim.
//! It proves one source-hash-selected routed FP4 expert and the always-on FP8
//! shared expert through real Metal dispatches, while leaving source QAT,
//! SwiGLU, and the route-control calculation in clearly declared host stages.
//! The input is a deterministic BF16 4096-vector, not a hidden state captured
//! from a model forward.  Consequently this must never be used as evidence of
//! an end-to-end layer, token, HCLI endpoint, or BASE_TRUE_TPS result.
//!
//! ```sh
//! cargo run --release -p hawking-core --example gravity_deepseek_v4_layer0_moe_metal_p5a -- \
//!   --artifact /absolute/path/to/full-43-layer-stream.gravity \
//!   --out /absolute/path/to/DSV4F_LAYER0_MOE_METAL_P5A-v1.json
//! ```

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(
        std::io::Error::other("gravity_deepseek_v4_layer0_moe_metal_p5a requires macOS Metal")
            .into(),
    )
}

#[cfg(target_os = "macos")]
mod macos {
    use half::bf16;
    use hawking_core::gravity_deepseek_v4::{
        DeepSeekV4FullStreamReader, DeepSeekV4Segment, DeepSeekV4TensorMetadata,
        NativeScalePairKind, FULL_STREAM_SCHEMA, FULL_STREAM_STATUS,
    };
    use hawking_core::gravity_deepseek_v4_act_quant::{
        act_quant_bf16_ue8m0, decode_e4m3fn, decode_e8m0fnu, deterministic_wq_a_input_bf16,
        fp8_e4m3fn_ue8m0_matvec, ActQuantizedBf16Row, Fp8MatvecCpuResult, ACT_QUANT_BLOCK,
    };
    use hawking_core::gravity_deepseek_v4_layer0_moe::{
        fp4_e2m1fn_x2_ue8m0_matvec, layer0_hash_route_cpu_oracle, swiglu_bf16_source_algorithm,
        verify_layer0_moe_source_anchors, Layer0HashRouteCpuResult, ACTIVATED_EXPERTS,
        LAYER0_FFN_GATE_TID2EID, LAYER0_FFN_GATE_WEIGHT, MOE_INTER_DIM, ROUTED_EXPERTS,
        ROUTE_SCALE,
    };
    use hawking_core::gravity_deepseek_v4_layer0_prefix::{HIDDEN_SIZE, PREFIX_TOKEN_ID};
    use hawking_core::metal::{
        MetalBatchTiming, MetalContext, PhysicalTraceGuard, PhysicalTraceIdentity,
    };
    use hawking_core::numeric_parity::{
        score_pair, ulp_distance_f32, Bounds, PairedScore, SCHEMA as V21_SCHEMA,
    };
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::error::Error;
    use std::fs::{self, File, OpenOptions};
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    const RECEIPT_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.layer0_moe_metal_p5a_bounded_substage.v1";
    const RECEIPT_STATUS: &str =
        "PASS_REAL_METAL_BOUNDED_MOE_SUBSTAGE_WITH_DECLARED_HOST_BOUNDARY_NOT_FULL_RUNTIME";
    const FP4_KERNEL: &str = "deepseek_v4_fp4_e2m1fn_x2_e8m0_matvec_authority";
    const FP8_KERNEL: &str = "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority";
    const BF16_CAST_KERNEL: &str = "deepseek_v4_p3a_fp32_to_bf16_authority";
    const ACT_QUANT_KERNEL: &str = "deepseek_v4_act_quant_bf16_ue8m0_authority";
    const P5B_FP4_KERNEL: &str = "deepseek_v4_p5b_fp4_act_quant_e2m1fn_x2_e8m0_matvec_authority";
    const P5B_SWIGLU_KERNEL: &str = "deepseek_v4_p5b_swiglu_route_bf16_authority";
    const P5B_COMBINE_KERNEL: &str = "deepseek_v4_p5b_route_shared_combine_bf16_authority";
    const P6A_GATE_KERNEL: &str = "deepseek_v4_p6a_gate_bf16_matvec_authority";
    const P6A_ROUTE_KERNEL: &str = "deepseek_v4_p6a_hash_route_sqrtsoftplus_authority";
    const P6A_SWIGLU_KERNEL: &str = "deepseek_v4_p6a_swiglu_route_weight_buffer_bf16_authority";
    const P6A_COMBINE_KERNEL: &str = "deepseek_v4_p6a_route6_shared_combine_bf16_authority";
    const DEFAULT_WARMUPS: usize = 3;
    const DEFAULT_TRIALS: usize = 5;
    // FP4 device authority expresses scaling per logical element while the
    // source-shaped CPU oracle scales a 32-K block after its accumulation.
    // This is a declared numerical envelope, not an assertion of bit identity.
    const ADMITTED_ABS_TOLERANCE: f64 = 5.0e-2;
    const ADMITTED_REL_TOLERANCE: f64 = 5.0e-4;

    type ProbeResult<T> = Result<T, Box<dyn Error>>;

    struct Args {
        artifact: PathBuf,
        out: PathBuf,
        warmups: usize,
        trials: usize,
    }

    #[derive(Clone)]
    struct NativePairBytes {
        label: String,
        kind: NativeScalePairKind,
        weight: DeepSeekV4TensorMetadata,
        scale: DeepSeekV4TensorMetadata,
        raw_weight: Vec<u8>,
        raw_scale: Vec<u8>,
        rows: usize,
        logical_k: usize,
        packed_k: usize,
        scale_cols: usize,
    }

    struct CpuPipeline {
        input_quant: ActQuantizedBf16Row,
        routed_gate: Fp8MatvecCpuResult,
        routed_up: Fp8MatvecCpuResult,
        routed_swiglu_bf16: Vec<u16>,
        routed_down_quant: ActQuantizedBf16Row,
        routed_down: Fp8MatvecCpuResult,
        shared_gate: Fp8MatvecCpuResult,
        shared_up: Fp8MatvecCpuResult,
        shared_swiglu_bf16: Vec<u16>,
        shared_down_quant: ActQuantizedBf16Row,
        shared_down: Fp8MatvecCpuResult,
        combined_f32: Vec<f32>,
        combined_bf16: Vec<u16>,
    }

    struct F64GateAuthority {
        logits: Vec<f64>,
        original_scores: Vec<f64>,
        selected_ids: Vec<u64>,
        selected_weights: Vec<f64>,
    }

    struct F64Pipeline {
        combined_f64: Vec<f64>,
        combined_bf16: Vec<u16>,
    }

    struct P5BExactParity {
        input_q: bool,
        input_scale: bool,
        routed_gate: bool,
        routed_up: bool,
        shared_gate: bool,
        shared_up: bool,
        routed_swiglu: bool,
        shared_swiglu: bool,
        routed_down_q: bool,
        routed_down_scale: bool,
        shared_down_q: bool,
        shared_down_scale: bool,
        routed_down: bool,
        shared_down: bool,
        combined: bool,
    }

    /// One source-order routed invocation for the P6A six-expert wave.  The
    /// vector is sorted by `(numeric expert ID, source top-slot)`, matching
    /// the source `for i in range(n_routed_experts)` loop rather than the
    /// tid2eid slot order.
    struct P6aRoutedCpuExpert {
        source_top_slot: usize,
        expert_id: u64,
        route_weight: f32,
        w1: NativePairBytes,
        w3: NativePairBytes,
        w2: NativePairBytes,
        gate: Fp8MatvecCpuResult,
        up: Fp8MatvecCpuResult,
        swiglu_bf16: Vec<u16>,
        down_quant: ActQuantizedBf16Row,
        down: Fp8MatvecCpuResult,
    }

    struct P6aFullRouteCpuPipeline {
        input_quant: ActQuantizedBf16Row,
        routed: Vec<P6aRoutedCpuExpert>,
        shared_w1: NativePairBytes,
        shared_w3: NativePairBytes,
        shared_w2: NativePairBytes,
        shared_gate: Fp8MatvecCpuResult,
        shared_up: Fp8MatvecCpuResult,
        shared_swiglu_bf16: Vec<u16>,
        shared_down_quant: ActQuantizedBf16Row,
        shared_down: Fp8MatvecCpuResult,
        combined_f32: Vec<f32>,
        combined_bf16: Vec<u16>,
        dominant_source_top_slot: usize,
        dominant_expert_id: u64,
        dominant_plus_shared_bf16: Vec<u16>,
    }

    /// All device allocations for one routed FP4 expert stay live for the
    /// full P6A run.  This makes it impossible for the measured six-wave
    /// graph to hide serial cache eviction/reload behind a host boundary.
    struct P6aExpertGpuBuffers {
        w1_weight: metal::Buffer,
        w1_scale: metal::Buffer,
        w3_weight: metal::Buffer,
        w3_scale: metal::Buffer,
        w2_weight: metal::Buffer,
        w2_scale: metal::Buffer,
        gate_f32: metal::Buffer,
        up_f32: metal::Buffer,
        gate_bf16: metal::Buffer,
        up_bf16: metal::Buffer,
        swiglu_bf16: metal::Buffer,
        down_quant: metal::Buffer,
        down_scales: metal::Buffer,
        down_f32: metal::Buffer,
        down_bf16: metal::Buffer,
    }

    struct P6aSharedGpuBuffers {
        w1_weight: metal::Buffer,
        w1_scale: metal::Buffer,
        w3_weight: metal::Buffer,
        w3_scale: metal::Buffer,
        w2_weight: metal::Buffer,
        w2_scale: metal::Buffer,
        gate_f32: metal::Buffer,
        up_f32: metal::Buffer,
        gate_bf16: metal::Buffer,
        up_bf16: metal::Buffer,
        swiglu_bf16: metal::Buffer,
        down_quant: metal::Buffer,
        down_scales: metal::Buffer,
        down_f32: metal::Buffer,
        down_bf16: metal::Buffer,
    }

    impl P5BExactParity {
        fn all(&self) -> bool {
            self.input_q
                && self.input_scale
                && self.routed_gate
                && self.routed_up
                && self.shared_gate
                && self.shared_up
                && self.routed_swiglu
                && self.shared_swiglu
                && self.routed_down_q
                && self.routed_down_scale
                && self.shared_down_q
                && self.shared_down_scale
                && self.routed_down
                && self.shared_down
                && self.combined
        }

        fn json(&self) -> Value {
            json!({
                "input_q_e4m3fn": self.input_q,
                "input_q_e8m0": self.input_scale,
                "routed_w1_bf16": self.routed_gate,
                "routed_w3_bf16": self.routed_up,
                "shared_w1_bf16": self.shared_gate,
                "shared_w3_bf16": self.shared_up,
                "routed_swiglu_bf16": self.routed_swiglu,
                "shared_swiglu_bf16": self.shared_swiglu,
                "routed_w2_input_q_e4m3fn": self.routed_down_q,
                "routed_w2_input_q_e8m0": self.routed_down_scale,
                "shared_w2_input_q_e4m3fn": self.shared_down_q,
                "shared_w2_input_q_e8m0": self.shared_down_scale,
                "routed_w2_bf16": self.routed_down,
                "shared_w2_bf16": self.shared_down,
                "combined_bf16": self.combined,
            })
        }
    }

    #[derive(Default)]
    struct TimingSeries {
        gpu_us: Vec<u64>,
        encode_us: Vec<u64>,
        submit_us: Vec<u64>,
        wait_us: Vec<u64>,
        host_wall_us: Vec<u64>,
        intervals_ns: Vec<[u64; 2]>,
    }

    impl TimingSeries {
        fn with_capacity(count: usize) -> Self {
            Self {
                gpu_us: Vec::with_capacity(count),
                encode_us: Vec::with_capacity(count),
                submit_us: Vec::with_capacity(count),
                wait_us: Vec::with_capacity(count),
                host_wall_us: Vec::with_capacity(count),
                intervals_ns: Vec::with_capacity(count),
            }
        }

        fn record(&mut self, timing: &MetalBatchTiming, label: &str) -> ProbeResult<()> {
            if timing.command_buffers != 1
                || timing.gpu_duration_us.unwrap_or(0) == 0
                || timing.gpu_start_ns.is_none()
                || timing.gpu_end_ns.is_none()
            {
                return Err(failure(format!(
                    "{label} has no usable completed-command-buffer GPU timestamp",
                )));
            }
            let start = timing.gpu_start_ns.expect("checked above");
            let end = timing.gpu_end_ns.expect("checked above");
            if end <= start {
                return Err(failure(format!("{label} has a non-positive GPU interval")));
            }
            self.gpu_us
                .push(timing.gpu_duration_us.expect("checked above"));
            self.encode_us.push(timing.encode_us);
            self.submit_us.push(timing.submit_us);
            self.wait_us.push(timing.wait_us);
            self.host_wall_us.push(timing.host_wall_us);
            self.intervals_ns.push([start, end]);
            Ok(())
        }
    }

    pub fn run() -> ProbeResult<()> {
        let args = parse_args()?;
        let reader = DeepSeekV4FullStreamReader::admit(&args.artifact)?;
        // This pins the source model.py/kernel.py/config grammar and all
        // layer-0 MoE tensor geometry before any byte is uploaded.
        let _anchors = verify_layer0_moe_source_anchors(&reader)?;

        let input_bf16 = deterministic_wq_a_input_bf16();
        if input_bf16.len() != HIDDEN_SIZE {
            return Err(failure("trusted bounded P5A input is not 4096 BF16 values"));
        }
        let route = layer0_hash_route_cpu_oracle(&reader, &input_bf16)?;
        validate_source_route(&route)?;
        let f64_route = f64_gate_authority(&reader, &input_bf16)?;
        if f64_route.selected_ids != route.selected_expert_ids {
            return Err(failure(
                "independent f64 Gate authority disagrees with exact hash tid2eid IDs",
            ));
        }
        let selected_slot = 0usize;
        let selected_expert = route.selected_expert_ids[selected_slot];
        let selected_route_weight = route.selected_weights_f32[selected_slot];
        if selected_expert >= ROUTED_EXPERTS as u64 {
            return Err(failure("selected hash expert is out of source range"));
        }

        let routed_gate = load_native_pair(
            &reader,
            &format!("layers.0.ffn.experts.{selected_expert}.w1.weight"),
            NativeScalePairKind::Fp4E2M1fnX2,
            MOE_INTER_DIM,
            HIDDEN_SIZE,
            "selected_routed_fp4_gate_w1",
        )?;
        let routed_up = load_native_pair(
            &reader,
            &format!("layers.0.ffn.experts.{selected_expert}.w3.weight"),
            NativeScalePairKind::Fp4E2M1fnX2,
            MOE_INTER_DIM,
            HIDDEN_SIZE,
            "selected_routed_fp4_up_w3",
        )?;
        let routed_down = load_native_pair(
            &reader,
            &format!("layers.0.ffn.experts.{selected_expert}.w2.weight"),
            NativeScalePairKind::Fp4E2M1fnX2,
            HIDDEN_SIZE,
            MOE_INTER_DIM,
            "selected_routed_fp4_down_w2",
        )?;
        let shared_gate = load_native_pair(
            &reader,
            "layers.0.ffn.shared_experts.w1.weight",
            NativeScalePairKind::Fp8E4M3fn,
            MOE_INTER_DIM,
            HIDDEN_SIZE,
            "shared_fp8_gate_w1",
        )?;
        let shared_up = load_native_pair(
            &reader,
            "layers.0.ffn.shared_experts.w3.weight",
            NativeScalePairKind::Fp8E4M3fn,
            MOE_INTER_DIM,
            HIDDEN_SIZE,
            "shared_fp8_up_w3",
        )?;
        let shared_down = load_native_pair(
            &reader,
            "layers.0.ffn.shared_experts.w2.weight",
            NativeScalePairKind::Fp8E4M3fn,
            HIDDEN_SIZE,
            MOE_INTER_DIM,
            "shared_fp8_down_w2",
        )?;

        let cpu = cpu_pipeline(
            &input_bf16,
            selected_route_weight,
            &routed_gate,
            &routed_up,
            &routed_down,
            &shared_gate,
            &shared_up,
            &shared_down,
        )?;
        let f64 = f64_pipeline(
            &cpu,
            selected_route_weight,
            &routed_gate,
            &routed_up,
            &routed_down,
            &shared_gate,
            &shared_up,
            &shared_down,
        )?;
        let cpu_vs_f64 = f64_metrics_from_f32(&f64.combined_f64, &cpu.combined_f32)?;

        let context = MetalContext::new_with_trace(true)?;
        let device_name = context.device_name();
        let fp4_pipeline = context.pipeline(FP4_KERNEL)?;
        let fp8_pipeline = context.pipeline(FP8_KERNEL)?;
        let cast_pipeline = context.pipeline(BF16_CAST_KERNEL)?;
        let fp4_width = fp4_pipeline.thread_execution_width() as u64;
        let fp8_width = fp8_pipeline.thread_execution_width() as u64;
        let cast_width = cast_pipeline.thread_execution_width() as u64;
        let fp4_max = fp4_pipeline.max_total_threads_per_threadgroup() as u32;
        let fp8_max = fp8_pipeline.max_total_threads_per_threadgroup() as u32;
        let cast_max = cast_pipeline.max_total_threads_per_threadgroup() as u32;
        drop(fp4_pipeline);
        drop(fp8_pipeline);
        drop(cast_pipeline);
        let fp4_threads = require_256_threads(fp4_max, FP4_KERNEL)?;
        let fp8_threads = require_256_threads(fp8_max, FP8_KERNEL)?;
        let cast_threads = require_256_threads(cast_max, BF16_CAST_KERNEL)?;

        // All source bytes are read through the admitted reader, verified
        // before these resident shared-memory Metal allocations are created.
        let rg_w = context.new_buffer_with_bytes_checked(&routed_gate.raw_weight)?;
        let rg_s = context.new_buffer_with_bytes_checked(&routed_gate.raw_scale)?;
        let ru_w = context.new_buffer_with_bytes_checked(&routed_up.raw_weight)?;
        let ru_s = context.new_buffer_with_bytes_checked(&routed_up.raw_scale)?;
        let rd_w = context.new_buffer_with_bytes_checked(&routed_down.raw_weight)?;
        let rd_s = context.new_buffer_with_bytes_checked(&routed_down.raw_scale)?;
        let sg_w = context.new_buffer_with_bytes_checked(&shared_gate.raw_weight)?;
        let sg_s = context.new_buffer_with_bytes_checked(&shared_gate.raw_scale)?;
        let su_w = context.new_buffer_with_bytes_checked(&shared_up.raw_weight)?;
        let su_s = context.new_buffer_with_bytes_checked(&shared_up.raw_scale)?;
        let sd_w = context.new_buffer_with_bytes_checked(&shared_down.raw_weight)?;
        let sd_s = context.new_buffer_with_bytes_checked(&shared_down.raw_scale)?;

        let routed_input_f32 = dequantized_activation(&cpu.input_quant)?;
        let routed_input_buf =
            context.new_buffer_with_bytes_checked(&f32_le_bytes(&routed_input_f32))?;
        let shared_input_q_buf =
            context.new_buffer_with_bytes_checked(&cpu.input_quant.activation_e4m3fn)?;
        let shared_input_scale_buf =
            context.new_buffer_with_bytes_checked(&cpu.input_quant.scales_e8m0fnu)?;
        let routed_down_input_buf =
            context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<f32>())?;
        let shared_down_q_buf = context.new_buffer_checked(MOE_INTER_DIM)?;
        let shared_down_scale_buf = context.new_buffer_checked(MOE_INTER_DIM / ACT_QUANT_BLOCK)?;

        let rg_f32 = context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<f32>())?;
        let ru_f32 = context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<f32>())?;
        let sg_f32 = context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<f32>())?;
        let su_f32 = context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<f32>())?;
        let rg_bf16 = context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<u16>())?;
        let ru_bf16 = context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<u16>())?;
        let sg_bf16 = context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<u16>())?;
        let su_bf16 = context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<u16>())?;
        let rd_f32 = context.new_buffer_checked(HIDDEN_SIZE * std::mem::size_of::<f32>())?;
        let sd_f32 = context.new_buffer_checked(HIDDEN_SIZE * std::mem::size_of::<f32>())?;
        let rd_bf16 = context.new_buffer_checked(HIDDEN_SIZE * std::mem::size_of::<u16>())?;
        let sd_bf16 = context.new_buffer_checked(HIDDEN_SIZE * std::mem::size_of::<u16>())?;

        let run_nonce = sha256_join(&[
            reader.manifest_seal_sha256(),
            &sha256(&u16_le_bytes(&input_bf16)),
            &selected_expert.to_string(),
            &sha256(&routed_gate.raw_weight),
            &sha256(&shared_gate.raw_weight),
            "layer0_moe_p5a_v1",
        ]);
        let interval_id = sha256_join(&[
            &run_nonce,
            FP4_KERNEL,
            FP8_KERNEL,
            BF16_CAST_KERNEL,
            "two_command_buffers_with_declared_host_handoff",
        ]);
        let physical_trace = PhysicalTraceGuard::begin(PhysicalTraceIdentity::new(
            interval_id.clone(),
            run_nonce.clone(),
            "layer0_moe_p5a".to_owned(),
            "one_routed_fp4_plus_shared_fp8_substage".to_owned(),
            Some(1),
            0,
        )?)?;

        let total_trials = args.warmups + args.trials;
        let mut stage1_series = TimingSeries::with_capacity(args.trials);
        let mut stage2_series = TimingSeries::with_capacity(args.trials);
        let mut host_handoff_us = Vec::with_capacity(args.trials);
        let mut host_combine_us = Vec::with_capacity(args.trials);
        let mut whole_gpu_us = Vec::with_capacity(args.trials);
        let mut whole_wall_us = Vec::with_capacity(args.trials);
        let mut gpu_final_bf16 = Vec::new();
        let mut stage_bf16_exact = true;
        let mut final_bf16_exact = true;
        let mut first_stage_metrics = Value::Null;
        let mut final_device_vs_f64 = Value::Null;

        for iteration in 0..total_trials {
            let trial_started = Instant::now();
            let stage1 = context.dispatch_batch_timed(|batch| {
                dispatch_fp4_then_bf16(
                    batch,
                    &rg_w,
                    &rg_s,
                    &routed_input_buf,
                    &rg_f32,
                    &rg_bf16,
                    &routed_gate,
                    fp4_threads,
                    cast_threads,
                )
                .map_err(|error| hawking_core::Error::Metal(error.to_string()))?;
                dispatch_fp4_then_bf16(
                    batch,
                    &ru_w,
                    &ru_s,
                    &routed_input_buf,
                    &ru_f32,
                    &ru_bf16,
                    &routed_up,
                    fp4_threads,
                    cast_threads,
                )
                .map_err(|error| hawking_core::Error::Metal(error.to_string()))?;
                dispatch_fp8_then_bf16(
                    batch,
                    &sg_w,
                    &sg_s,
                    &shared_input_q_buf,
                    &shared_input_scale_buf,
                    &sg_f32,
                    &sg_bf16,
                    &shared_gate,
                    fp8_threads,
                    cast_threads,
                )
                .map_err(|error| hawking_core::Error::Metal(error.to_string()))?;
                dispatch_fp8_then_bf16(
                    batch,
                    &su_w,
                    &su_s,
                    &shared_input_q_buf,
                    &shared_input_scale_buf,
                    &su_f32,
                    &su_bf16,
                    &shared_up,
                    fp8_threads,
                    cast_threads,
                )
                .map_err(|error| hawking_core::Error::Metal(error.to_string()))?;
                Ok(())
            })?;
            require_batch_topology(&stage1, 4, 8, "P5A stage-1")?;

            let handoff_started = Instant::now();
            let gpu_rg_bf16 = read_gpu_u16(&rg_bf16, MOE_INTER_DIM)?;
            let gpu_ru_bf16 = read_gpu_u16(&ru_bf16, MOE_INTER_DIM)?;
            let gpu_sg_bf16 = read_gpu_u16(&sg_bf16, MOE_INTER_DIM)?;
            let gpu_su_bf16 = read_gpu_u16(&su_bf16, MOE_INTER_DIM)?;
            stage_bf16_exact &= gpu_rg_bf16 == cpu.routed_gate.bf16_bits
                && gpu_ru_bf16 == cpu.routed_up.bf16_bits
                && gpu_sg_bf16 == cpu.shared_gate.bf16_bits
                && gpu_su_bf16 == cpu.shared_up.bf16_bits;
            let gpu_routed_swiglu = swiglu_bf16_source_algorithm(
                &gpu_rg_bf16,
                &gpu_ru_bf16,
                Some(selected_route_weight),
            )?;
            let gpu_shared_swiglu = swiglu_bf16_source_algorithm(&gpu_sg_bf16, &gpu_su_bf16, None)?;
            let gpu_routed_down_quant = act_quant_bf16_ue8m0(&gpu_routed_swiglu)?;
            let gpu_shared_down_quant = act_quant_bf16_ue8m0(&gpu_shared_swiglu)?;
            let routed_down_input = dequantized_activation(&gpu_routed_down_quant)?;
            MetalContext::write_buffer_bytes(
                &routed_down_input_buf,
                &f32_le_bytes(&routed_down_input),
            );
            MetalContext::write_buffer_bytes(
                &shared_down_q_buf,
                &gpu_shared_down_quant.activation_e4m3fn,
            );
            MetalContext::write_buffer_bytes(
                &shared_down_scale_buf,
                &gpu_shared_down_quant.scales_e8m0fnu,
            );
            let handoff_elapsed = handoff_started.elapsed().as_micros() as u64;

            let stage2 = context.dispatch_batch_timed(|batch| {
                dispatch_fp4_then_bf16(
                    batch,
                    &rd_w,
                    &rd_s,
                    &routed_down_input_buf,
                    &rd_f32,
                    &rd_bf16,
                    &routed_down,
                    fp4_threads,
                    cast_threads,
                )
                .map_err(|error| hawking_core::Error::Metal(error.to_string()))?;
                dispatch_fp8_then_bf16(
                    batch,
                    &sd_w,
                    &sd_s,
                    &shared_down_q_buf,
                    &shared_down_scale_buf,
                    &sd_f32,
                    &sd_bf16,
                    &shared_down,
                    fp8_threads,
                    cast_threads,
                )
                .map_err(|error| hawking_core::Error::Metal(error.to_string()))?;
                Ok(())
            })?;
            require_batch_topology(&stage2, 2, 4, "P5A stage-2")?;

            let combine_started = Instant::now();
            let gpu_rd_bf16 = read_gpu_u16(&rd_bf16, HIDDEN_SIZE)?;
            let gpu_sd_bf16 = read_gpu_u16(&sd_bf16, HIDDEN_SIZE)?;
            let gpu_combined_f32 = source_combine(&gpu_rd_bf16, &gpu_sd_bf16)?;
            gpu_final_bf16 = gpu_combined_f32
                .iter()
                .copied()
                .map(|value| bf16::from_f32(value).to_bits())
                .collect();
            final_bf16_exact &= gpu_final_bf16 == cpu.combined_bf16;
            let combine_elapsed = combine_started.elapsed().as_micros() as u64;

            if iteration == total_trials - 1 {
                first_stage_metrics = json!({
                    "routed_gate_fp32_source_cpu_vs_device": f32_metrics(&cpu.routed_gate.fp32, &read_gpu_f32(&rg_f32, MOE_INTER_DIM)?)?,
                    "routed_up_fp32_source_cpu_vs_device": f32_metrics(&cpu.routed_up.fp32, &read_gpu_f32(&ru_f32, MOE_INTER_DIM)?)?,
                    "shared_gate_fp32_source_cpu_vs_device": f32_metrics(&cpu.shared_gate.fp32, &read_gpu_f32(&sg_f32, MOE_INTER_DIM)?)?,
                    "shared_up_fp32_source_cpu_vs_device": f32_metrics(&cpu.shared_up.fp32, &read_gpu_f32(&su_f32, MOE_INTER_DIM)?)?,
                    "routed_gate_bf16_hash_cpu": sha256(&u16_le_bytes(&cpu.routed_gate.bf16_bits)),
                    "routed_gate_bf16_hash_device": sha256(&u16_le_bytes(&gpu_rg_bf16)),
                    "shared_gate_bf16_hash_cpu": sha256(&u16_le_bytes(&cpu.shared_gate.bf16_bits)),
                    "shared_gate_bf16_hash_device": sha256(&u16_le_bytes(&gpu_sg_bf16)),
                });
                final_device_vs_f64 = f64_metrics_from_f32(&f64.combined_f64, &gpu_combined_f32)?;
            }
            if iteration >= args.warmups {
                stage1_series.record(&stage1, "P5A stage-1")?;
                stage2_series.record(&stage2, "P5A stage-2")?;
                host_handoff_us.push(handoff_elapsed);
                host_combine_us.push(combine_elapsed);
                whole_gpu_us.push(
                    stage1.gpu_duration_us.expect("checked by TimingSeries")
                        + stage2.gpu_duration_us.expect("checked by TimingSeries"),
                );
                whole_wall_us.push(trial_started.elapsed().as_micros() as u64);
            }
        }

        let physical_counts = physical_trace.counts();
        drop(physical_trace);
        let (buffers_created, bytes_allocated, commits) = context.drain_stats();
        let trace_samples = context.drain_trace();
        let expected_command_buffers = (total_trials * 2) as u64;
        let expected_encoders = (total_trials * 6) as u64;
        let expected_dispatches = (total_trials * 12) as u64;
        if physical_counts.command_count != expected_command_buffers
            || physical_counts.encoder_count != expected_encoders
            || commits != expected_command_buffers as usize
            || trace_samples.len() != expected_command_buffers as usize
        {
            return Err(failure(format!(
                "P5A physical command topology mismatch: commands={} encoders={} commits={} trace_samples={}",
                physical_counts.command_count,
                physical_counts.encoder_count,
                commits,
                trace_samples.len(),
            )));
        }

        let stage1_timing = timing_json(&stage1_series)?;
        let stage2_timing = timing_json(&stage2_series)?;
        let active_weights_bytes = routed_gate.raw_weight.len()
            + routed_gate.raw_scale.len()
            + routed_up.raw_weight.len()
            + routed_up.raw_scale.len()
            + routed_down.raw_weight.len()
            + routed_down.raw_scale.len()
            + shared_gate.raw_weight.len()
            + shared_gate.raw_scale.len()
            + shared_up.raw_weight.len()
            + shared_up.raw_scale.len()
            + shared_down.raw_weight.len()
            + shared_down.raw_scale.len();
        let stage1_bytes = stage1_logical_bytes(&routed_gate, &routed_up, &shared_gate, &shared_up);
        let stage2_bytes = stage2_logical_bytes(&routed_down, &shared_down);
        let route_weight_deltas =
            f64_vector_metrics(&f64_route.selected_weights, &route.selected_weights_f32)?;

        let gate_tensor = reader.tensor_metadata(LAYER0_FFN_GATE_WEIGHT)?;
        let tid2eid_tensor = reader.tensor_metadata(LAYER0_FFN_GATE_TID2EID)?;
        let tid2eid_start =
            PREFIX_TOKEN_ID * (ACTIVATED_EXPERTS * std::mem::size_of::<i64>()) as u64;
        let tid2eid_end = tid2eid_start + (ACTIVATED_EXPERTS * std::mem::size_of::<i64>()) as u64;
        let unsigned = json!({
            "schema": RECEIPT_SCHEMA,
            "status": RECEIPT_STATUS,
            "scope": {
                "bounded_component": "layer-0 hash route control plus exactly one hash-selected routed FP4 expert and the always-on shared FP8 expert",
                "trusted_predecessor_input": "deterministic BF16 [4096] fixture; not attention output, model forward, prompt, or generated token",
                "full_decoder_layer_forward": false,
                "full_model_loaded": false,
                "full_model_forward": false,
                "generated_tokens": 0,
                "hcli_endpoint_started": false,
                "base_true_tps_measured": false,
                "claim_boundary": "Real Metal dispatches prove only this bounded mixed host/device MoE substage. CPU Gate/QAT/SwiGLU/combine stages are declared boundaries, not hidden fallbacks; this does not establish full source-runtime parity, a full layer, a causal loop, HCLI, or TPS."
            },
            "artifact": {
                "path": reader.artifact_root().display().to_string(),
                "full_stream_schema": FULL_STREAM_SCHEMA,
                "full_stream_status": FULL_STREAM_STATUS,
                "manifest_file_sha256": reader.manifest_file_sha256(),
                "manifest_seal_sha256": reader.manifest_seal_sha256(),
                "restart_receipt_seal_sha256": reader.restart_seal_sha256(),
                "source": {
                    "repository": reader.source_identity().repository,
                    "revision": reader.source_identity().revision,
                    "source_parent_retained": false,
                },
                "reader_admission_and_source_grammar_verified": true,
                "parent_safetensors_materialized": false,
            },
            "source_bindings": {
                "inference/model.py_sha256": reader.source_metadata_asset_sha256("inference/model.py")?,
                "inference/kernel.py_sha256": reader.source_metadata_asset_sha256("inference/kernel.py")?,
                "inference/config.json_sha256": reader.source_metadata_asset_sha256("inference/config.json")?,
                "config.json_sha256": reader.source_metadata_asset_sha256("config.json")?,
                "gate_weight": tensor_binding_json(gate_tensor),
                "tid2eid_bos_row": tensor_range_binding_json(tid2eid_tensor, tid2eid_start, tid2eid_end),
                "selected_routed_fp4": {
                    "source_top_slot": selected_slot,
                    "expert_id": selected_expert,
                    "w1": pair_binding_json(&routed_gate),
                    "w3": pair_binding_json(&routed_up),
                    "w2": pair_binding_json(&routed_down),
                },
                "shared_fp8": {
                    "w1": pair_binding_json(&shared_gate),
                    "w3": pair_binding_json(&shared_up),
                    "w2": pair_binding_json(&shared_down),
                },
                "touched_chunks_sha256_verified_before_cpu_use_and_gpu_upload": true,
            },
            "route_control": {
                "token_id": PREFIX_TOKEN_ID,
                "hash_layer": true,
                "selection_method": "source tid2eid[token_id] fixed row; Gate scores still calculated and gathered weights normalized then multiplied by route_scale",
                "source_f32_transcription": {
                    "selected_expert_ids_in_source_top_slot_order": route.selected_expert_ids,
                    "selected_route_weights_f32_text": f32_texts(&route.selected_weights_f32),
                    "selected_weight_sum_f32_text": f32_text(route.selected_weights_f32.iter().sum()),
                    "gate_logits_sha256_f32_le": sha256(&f32_le_bytes(&route.logits_f32)),
                    "gate_scores_sha256_f32_le": sha256(&f32_le_bytes(&route.original_scores_f32)),
                },
                "separate_f64_authority": {
                    "independent_decode_and_accumulation": "BF16 gate bytes decoded to f64; f64 dot products plus source thresholded sqrt-softplus; fixed tid2eid row reread independently",
                    "selected_expert_ids_exact_match_source_f32": true,
                    "selected_expert_ids": f64_route.selected_ids,
                    "logits_sha256_f64_le": sha256(&f64_le_bytes(&f64_route.logits)),
                    "scores_sha256_f64_le": sha256(&f64_le_bytes(&f64_route.original_scores)),
                    "selected_route_weights_f64_text": f64_texts(&f64_route.selected_weights),
                    "source_f32_vs_f64_selected_weight_metrics": route_weight_deltas,
                },
                "selected_substage_route_weight_f32_text": f32_text(selected_route_weight),
                "route_scale_f32_text": f32_text(ROUTE_SCALE),
            },
            "host_device_boundary": {
                "stage_0_host": [
                    "source-derived f32 Gate scores and fixed hash tid2eid selection",
                    "source act_quant BF16 -> E4M3FN plus E8M0 for the trusted input",
                    "FP4 input dequantization to F32 solely because the existing FP4 authority kernel accepts F32 x",
                ],
                "stage_1_device": "routed FP4 W1/W3; shared FP8 W1/W3; all four FP32 outputs converted to BF16 by an existing source-storage authority kernel",
                "interstage_host": "read BF16, source-clamped SwiGLU, route weight before routed W2, source act_quant, and writes next device inputs",
                "stage_2_device": "routed FP4 W2 plus shared FP8 W2; both FP32 outputs converted to BF16 on device",
                "poststage_host": "source-order BF16 decoded f32 additive combine and BF16 storage cast",
                "host_work_is_declared_boundary_not_fallback": true,
                "hidden_fallback": false,
            },
            "cpu_source_transcription": {
                "input_bf16_sha256": sha256(&u16_le_bytes(&input_bf16)),
                "input_quant_e4m3fn_sha256": sha256(&cpu.input_quant.activation_e4m3fn),
                "input_quant_e8m0_sha256": sha256(&cpu.input_quant.scales_e8m0fnu),
                "routed_gate_bf16_sha256": sha256(&u16_le_bytes(&cpu.routed_gate.bf16_bits)),
                "routed_up_bf16_sha256": sha256(&u16_le_bytes(&cpu.routed_up.bf16_bits)),
                "routed_weighted_swiglu_bf16_sha256": sha256(&u16_le_bytes(&cpu.routed_swiglu_bf16)),
                "routed_down_quant_e4m3fn_sha256": sha256(&cpu.routed_down_quant.activation_e4m3fn),
                "routed_down_quant_e8m0_sha256": sha256(&cpu.routed_down_quant.scales_e8m0fnu),
                "routed_down_bf16_sha256": sha256(&u16_le_bytes(&cpu.routed_down.bf16_bits)),
                "shared_gate_bf16_sha256": sha256(&u16_le_bytes(&cpu.shared_gate.bf16_bits)),
                "shared_up_bf16_sha256": sha256(&u16_le_bytes(&cpu.shared_up.bf16_bits)),
                "shared_swiglu_bf16_sha256": sha256(&u16_le_bytes(&cpu.shared_swiglu_bf16)),
                "shared_down_quant_e4m3fn_sha256": sha256(&cpu.shared_down_quant.activation_e4m3fn),
                "shared_down_quant_e8m0_sha256": sha256(&cpu.shared_down_quant.scales_e8m0fnu),
                "shared_down_bf16_sha256": sha256(&u16_le_bytes(&cpu.shared_down.bf16_bits)),
                "combined_f32_sha256": sha256(&f32_le_bytes(&cpu.combined_f32)),
                "combined_bf16_sha256": sha256(&u16_le_bytes(&cpu.combined_bf16)),
                "cpu_f32_vs_separate_f64_final_metrics": cpu_vs_f64,
            },
            "metal": {
                "device": device_name,
                "kernels": {
                    "routed_fp4": FP4_KERNEL,
                    "shared_fp8": FP8_KERNEL,
                    "fp32_to_bf16_storage": BF16_CAST_KERNEL,
                },
                "pipelines_precompiled_before_warmups": true,
                "thread_execution_width": {
                    "routed_fp4": fp4_width,
                    "shared_fp8": fp8_width,
                    "bf16_storage": cast_width,
                },
                "threadgroup": {
                    "routed_fp4": [fp4_threads, 1, 1],
                    "shared_fp8": [fp8_threads, 1, 1],
                    "bf16_storage": [cast_threads, 1, 1],
                },
                "resident_buffers_created": buffers_created,
                "resident_bytes_allocated": bytes_allocated,
                "real_gpu_dispatches": expected_dispatches,
                "command_buffers": expected_command_buffers,
                "compute_encoders": expected_encoders,
                "cpu_visible_waits": expected_command_buffers,
                "empty_command_buffers": 0,
                "physical_trace_command_buffers": physical_counts.command_count,
                "physical_trace_compute_encoders": physical_counts.encoder_count,
                "trace_samples": trace_samples.len(),
                "fallback": false,
                "fallback_count": 0,
                "hardware_occupancy": "NOT_EXPOSED_BY_THIS_METAL_TIMESTAMP_PROBE; no value inferred",
                "hardware_counter_sampling": "NOT_CAPTURED",
            },
            "command_topology": {
                "per_iteration": {
                    "stage_1": {
                        "command_buffers": 1,
                        "compute_encoders": 4,
                        "compute_dispatches": 8,
                        "ordered_pairs": [
                            "routed_fp4_w1 -> bf16_storage",
                            "routed_fp4_w3 -> bf16_storage",
                            "shared_fp8_w1 -> bf16_storage",
                            "shared_fp8_w3 -> bf16_storage",
                        ],
                    },
                    "stage_2": {
                        "command_buffers": 1,
                        "compute_encoders": 2,
                        "compute_dispatches": 4,
                        "ordered_pairs": [
                            "routed_fp4_w2 -> bf16_storage",
                            "shared_fp8_w2 -> bf16_storage",
                        ],
                    },
                },
                "per_iteration_total": {"command_buffers": 2, "compute_encoders": 6, "compute_dispatches": 12, "cpu_visible_waits": 2},
                "persistent_replay_graph": false,
                "reason_not_promoted": "host source SwiGLU/QAT boundary remains between W1/W3 and W2; no full runtime graph is claimed",
            },
            "timing": {
                "warmups": args.warmups,
                "clean_trials": args.trials,
                "timestamp_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime; CPU wall and waits recorded separately",
                "stage_1_device": stage1_timing,
                "stage_2_device": stage2_timing,
                "host_interstage_read_swiglu_qat_write": summary_json(&host_handoff_us)?,
                "host_poststage_read_and_source_combine": summary_json(&host_combine_us)?,
                "whole_substage_gpu_two_command_buffers": summary_json(&whole_gpu_us)?,
                "whole_substage_host_wall": summary_json(&whole_wall_us)?,
            },
            "active_bytes_and_operations": {
                "selected_active_weight_and_scale_bytes": active_weights_bytes,
                "stage_1_logical_access": stage1_bytes,
                "stage_2_logical_access": stage2_bytes,
                "host_interstage_read_bytes": 4 * MOE_INTER_DIM * std::mem::size_of::<u16>(),
                "host_interstage_written_bytes": MOE_INTER_DIM * std::mem::size_of::<f32>() + MOE_INTER_DIM + MOE_INTER_DIM / ACT_QUANT_BLOCK,
                "host_poststage_read_bytes": 2 * HIDDEN_SIZE * std::mem::size_of::<u16>(),
                "source_arithmetic_operations_not_profiled_per_operator": "This receipt gives exact bounded byte/dispatch topology and aggregate GPU timestamps; it does not invent hardware FLOP, integer-op, or occupancy counters.",
            },
            "numerical_scorecard": {
                "admitted_device_envelope": {
                    "comparison": "abs_error <= 0.05 + 0.0005 * abs(reference)",
                    "absolute_tolerance_text": f64_text(ADMITTED_ABS_TOLERANCE),
                    "relative_tolerance_text": f64_text(ADMITTED_REL_TOLERANCE),
                    "reason": "existing FP4 F32-x authority has a distinct scale-association order from the source-shaped 32-K CPU reference; exact BF16 hashes are recorded separately rather than inferred",
                },
                "stage_1_last_trial": first_stage_metrics,
                "all_trials_stage_1_bf16_bit_exact_to_cpu_source": stage_bf16_exact,
                "all_trials_final_bf16_bit_exact_to_cpu_source": final_bf16_exact,
                "device_final_bf16_sha256": sha256(&u16_le_bytes(&gpu_final_bf16)),
                "cpu_final_bf16_sha256": sha256(&u16_le_bytes(&cpu.combined_bf16)),
                "device_final_vs_separate_f64_authority": final_device_vs_f64,
                "separate_f64_pipeline": {
                    "matvec": "independent f64 native-byte FP4/FP8 decoders and source block grouping",
                    "storage_boundaries": "source BF16/QAT routines deliberately reapply exact stored-boundary bytes between f64 linear stages",
                    "route_weight": "validated independently by f64 Gate authority then frozen to source f32 weight for the actual source storage pipeline",
                    "final_bf16_sha256": sha256(&u16_le_bytes(&f64.combined_bf16)),
                },
            },
            "resume": {
                "command": format!(
                    "cargo run --release -p hawking-core --example gravity_deepseek_v4_layer0_moe_metal_p5a -- --artifact {} --out {} --warmups {} --trials {}",
                    reader.artifact_root().display(), args.out.display(), args.warmups, args.trials,
                ),
                "source_windows_evicted_by_artifact_contract": true,
            },
        });
        let (receipt, seal) = seal(unsigned)?;
        write_new_receipt(&args.out, &receipt)?;
        println!(
            "{}",
            serde_json::to_string(&json!({
                "status": RECEIPT_STATUS,
                "receipt": args.out,
                "seal_sha256": seal,
                "selected_expert": selected_expert,
                "real_gpu_dispatches": expected_dispatches,
                "all_trials_final_bf16_bit_exact_to_cpu_source": final_bf16_exact,
            }))?
        );
        Ok(())
    }

    /// P5B removes P5A's CPU SwiGLU/QAT/combine interval.  It intentionally
    /// retains host hash-Gate control only: the source hash table fixes route
    /// IDs, and this bounded substage executes the selected dominant route plus
    /// the shared expert entirely through device storage boundaries.
    pub fn run_p5b() -> ProbeResult<()> {
        let args = parse_args()?;
        let reader = DeepSeekV4FullStreamReader::admit(&args.artifact)?;
        let _anchors = verify_layer0_moe_source_anchors(&reader)?;
        let input_bf16 = deterministic_wq_a_input_bf16();
        let route = layer0_hash_route_cpu_oracle(&reader, &input_bf16)?;
        validate_source_route(&route)?;
        let f64_route = f64_gate_authority(&reader, &input_bf16)?;
        if f64_route.selected_ids != route.selected_expert_ids {
            return Err(failure(
                "P5B f64 Gate authority disagrees with the source hash tid2eid IDs",
            ));
        }
        let (selected_slot, &selected_route_weight) = route
            .selected_weights_f32
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .ok_or_else(|| failure("P5B source route has no selected expert"))?;
        let (low_weight_slot, &low_weight) = route
            .selected_weights_f32
            .iter()
            .enumerate()
            .min_by(|left, right| left.1.total_cmp(right.1))
            .ok_or_else(|| failure("P5B source route has no selected expert"))?;
        let selected_expert = route.selected_expert_ids[selected_slot];
        if selected_slot == low_weight_slot
            || selected_expert >= ROUTED_EXPERTS as u64
            || !(selected_route_weight.is_finite() && selected_route_weight > 0.0)
        {
            return Err(failure("P5B dominant route selection is invalid"));
        }

        let routed_gate = load_native_pair(
            &reader,
            &format!("layers.0.ffn.experts.{selected_expert}.w1.weight"),
            NativeScalePairKind::Fp4E2M1fnX2,
            MOE_INTER_DIM,
            HIDDEN_SIZE,
            "dominant_routed_fp4_gate_w1",
        )?;
        let routed_up = load_native_pair(
            &reader,
            &format!("layers.0.ffn.experts.{selected_expert}.w3.weight"),
            NativeScalePairKind::Fp4E2M1fnX2,
            MOE_INTER_DIM,
            HIDDEN_SIZE,
            "dominant_routed_fp4_up_w3",
        )?;
        let routed_down = load_native_pair(
            &reader,
            &format!("layers.0.ffn.experts.{selected_expert}.w2.weight"),
            NativeScalePairKind::Fp4E2M1fnX2,
            HIDDEN_SIZE,
            MOE_INTER_DIM,
            "dominant_routed_fp4_down_w2",
        )?;
        let shared_gate = load_native_pair(
            &reader,
            "layers.0.ffn.shared_experts.w1.weight",
            NativeScalePairKind::Fp8E4M3fn,
            MOE_INTER_DIM,
            HIDDEN_SIZE,
            "shared_fp8_gate_w1",
        )?;
        let shared_up = load_native_pair(
            &reader,
            "layers.0.ffn.shared_experts.w3.weight",
            NativeScalePairKind::Fp8E4M3fn,
            MOE_INTER_DIM,
            HIDDEN_SIZE,
            "shared_fp8_up_w3",
        )?;
        let shared_down = load_native_pair(
            &reader,
            "layers.0.ffn.shared_experts.w2.weight",
            NativeScalePairKind::Fp8E4M3fn,
            HIDDEN_SIZE,
            MOE_INTER_DIM,
            "shared_fp8_down_w2",
        )?;
        let cpu = cpu_pipeline(
            &input_bf16,
            selected_route_weight,
            &routed_gate,
            &routed_up,
            &routed_down,
            &shared_gate,
            &shared_up,
            &shared_down,
        )?;
        let f64 = f64_pipeline(
            &cpu,
            selected_route_weight,
            &routed_gate,
            &routed_up,
            &routed_down,
            &shared_gate,
            &shared_up,
            &shared_down,
        )?;

        let context = MetalContext::new_with_trace(true)?;
        let device_name = context.device_name();
        let qat_pipeline = context.pipeline(ACT_QUANT_KERNEL)?;
        let fp4_pipeline = context.pipeline(P5B_FP4_KERNEL)?;
        let fp8_pipeline = context.pipeline(FP8_KERNEL)?;
        let cast_pipeline = context.pipeline(BF16_CAST_KERNEL)?;
        let swiglu_pipeline = context.pipeline(P5B_SWIGLU_KERNEL)?;
        let combine_pipeline = context.pipeline(P5B_COMBINE_KERNEL)?;
        let qat_width = qat_pipeline.thread_execution_width() as u64;
        let fp4_width = fp4_pipeline.thread_execution_width() as u64;
        let fp8_width = fp8_pipeline.thread_execution_width() as u64;
        let cast_width = cast_pipeline.thread_execution_width() as u64;
        let swiglu_width = swiglu_pipeline.thread_execution_width() as u64;
        let combine_width = combine_pipeline.thread_execution_width() as u64;
        let qat_max = qat_pipeline.max_total_threads_per_threadgroup() as u32;
        let fp4_max = fp4_pipeline.max_total_threads_per_threadgroup() as u32;
        let fp8_max = fp8_pipeline.max_total_threads_per_threadgroup() as u32;
        let cast_max = cast_pipeline.max_total_threads_per_threadgroup() as u32;
        let swiglu_max = swiglu_pipeline.max_total_threads_per_threadgroup() as u32;
        let combine_max = combine_pipeline.max_total_threads_per_threadgroup() as u32;
        drop(qat_pipeline);
        drop(fp4_pipeline);
        drop(fp8_pipeline);
        drop(cast_pipeline);
        drop(swiglu_pipeline);
        drop(combine_pipeline);
        let qat_threads = require_threads(qat_max, 32, ACT_QUANT_KERNEL)?;
        let fp4_threads = require_256_threads(fp4_max, P5B_FP4_KERNEL)?;
        let fp8_threads = require_256_threads(fp8_max, FP8_KERNEL)?;
        let cast_threads = require_256_threads(cast_max, BF16_CAST_KERNEL)?;
        let swiglu_threads = require_256_threads(swiglu_max, P5B_SWIGLU_KERNEL)?;
        let combine_threads = require_256_threads(combine_max, P5B_COMBINE_KERNEL)?;

        let rg_w = context.new_buffer_with_bytes_checked(&routed_gate.raw_weight)?;
        let rg_s = context.new_buffer_with_bytes_checked(&routed_gate.raw_scale)?;
        let ru_w = context.new_buffer_with_bytes_checked(&routed_up.raw_weight)?;
        let ru_s = context.new_buffer_with_bytes_checked(&routed_up.raw_scale)?;
        let rd_w = context.new_buffer_with_bytes_checked(&routed_down.raw_weight)?;
        let rd_s = context.new_buffer_with_bytes_checked(&routed_down.raw_scale)?;
        let sg_w = context.new_buffer_with_bytes_checked(&shared_gate.raw_weight)?;
        let sg_s = context.new_buffer_with_bytes_checked(&shared_gate.raw_scale)?;
        let su_w = context.new_buffer_with_bytes_checked(&shared_up.raw_weight)?;
        let su_s = context.new_buffer_with_bytes_checked(&shared_up.raw_scale)?;
        let sd_w = context.new_buffer_with_bytes_checked(&shared_down.raw_weight)?;
        let sd_s = context.new_buffer_with_bytes_checked(&shared_down.raw_scale)?;

        let input_bf16_buf = context.new_buffer_with_bytes_checked(&u16_le_bytes(&input_bf16))?;
        let input_q_buf = context.new_buffer_checked(HIDDEN_SIZE)?;
        let input_scale_buf = context.new_buffer_checked(HIDDEN_SIZE / ACT_QUANT_BLOCK)?;
        let routed_q_buf = context.new_buffer_checked(MOE_INTER_DIM)?;
        let routed_scale_buf = context.new_buffer_checked(MOE_INTER_DIM / ACT_QUANT_BLOCK)?;
        let shared_q_buf = context.new_buffer_checked(MOE_INTER_DIM)?;
        let shared_scale_buf = context.new_buffer_checked(MOE_INTER_DIM / ACT_QUANT_BLOCK)?;

        let rg_f32 = context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<f32>())?;
        let ru_f32 = context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<f32>())?;
        let sg_f32 = context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<f32>())?;
        let su_f32 = context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<f32>())?;
        let rg_bf16 = context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<u16>())?;
        let ru_bf16 = context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<u16>())?;
        let sg_bf16 = context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<u16>())?;
        let su_bf16 = context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<u16>())?;
        let routed_swiglu =
            context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<u16>())?;
        let shared_swiglu =
            context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<u16>())?;
        let rd_f32 = context.new_buffer_checked(HIDDEN_SIZE * std::mem::size_of::<f32>())?;
        let sd_f32 = context.new_buffer_checked(HIDDEN_SIZE * std::mem::size_of::<f32>())?;
        let rd_bf16 = context.new_buffer_checked(HIDDEN_SIZE * std::mem::size_of::<u16>())?;
        let sd_bf16 = context.new_buffer_checked(HIDDEN_SIZE * std::mem::size_of::<u16>())?;
        let combined_bf16 = context.new_buffer_checked(HIDDEN_SIZE * std::mem::size_of::<u16>())?;

        let run_nonce = sha256_join(&[
            reader.manifest_seal_sha256(),
            &sha256(&u16_le_bytes(&input_bf16)),
            &selected_expert.to_string(),
            &f32_text(selected_route_weight),
            &sha256(&routed_gate.raw_weight),
            "layer0_moe_p5b_v1",
        ]);
        let interval_id = sha256_join(&[
            &run_nonce,
            ACT_QUANT_KERNEL,
            P5B_FP4_KERNEL,
            FP8_KERNEL,
            P5B_SWIGLU_KERNEL,
            P5B_COMBINE_KERNEL,
        ]);
        let physical_trace = PhysicalTraceGuard::begin(PhysicalTraceIdentity::new(
            interval_id.clone(),
            run_nonce.clone(),
            "layer0_moe_p5b".to_owned(),
            "dominant_routed_fp4_and_shared_fp8_device_storage_chain".to_owned(),
            Some(1),
            0,
        )?)?;
        let total_trials = args.warmups + args.trials;
        let mut stage1_series = TimingSeries::with_capacity(args.trials);
        let mut stage2_series = TimingSeries::with_capacity(args.trials);
        let mut whole_gpu_us = Vec::with_capacity(args.trials);
        let mut whole_wall_us = Vec::with_capacity(args.trials);

        for iteration in 0..total_trials {
            let trial_started = Instant::now();
            let stage1 = context.dispatch_batch_timed(|batch| {
                dispatch_device_act_quant(
                    batch,
                    &input_bf16_buf,
                    &input_q_buf,
                    &input_scale_buf,
                    HIDDEN_SIZE as u32,
                    qat_threads,
                )?;
                dispatch_p5b_fp4_then_bf16(
                    batch,
                    &rg_w,
                    &rg_s,
                    &input_q_buf,
                    &input_scale_buf,
                    &rg_f32,
                    &rg_bf16,
                    &routed_gate,
                    fp4_threads,
                    cast_threads,
                )
                .map_err(|error| hawking_core::Error::Metal(error.to_string()))?;
                dispatch_p5b_fp4_then_bf16(
                    batch,
                    &ru_w,
                    &ru_s,
                    &input_q_buf,
                    &input_scale_buf,
                    &ru_f32,
                    &ru_bf16,
                    &routed_up,
                    fp4_threads,
                    cast_threads,
                )
                .map_err(|error| hawking_core::Error::Metal(error.to_string()))?;
                dispatch_fp8_then_bf16(
                    batch,
                    &sg_w,
                    &sg_s,
                    &input_q_buf,
                    &input_scale_buf,
                    &sg_f32,
                    &sg_bf16,
                    &shared_gate,
                    fp8_threads,
                    cast_threads,
                )
                .map_err(|error| hawking_core::Error::Metal(error.to_string()))?;
                dispatch_fp8_then_bf16(
                    batch,
                    &su_w,
                    &su_s,
                    &input_q_buf,
                    &input_scale_buf,
                    &su_f32,
                    &su_bf16,
                    &shared_up,
                    fp8_threads,
                    cast_threads,
                )
                .map_err(|error| hawking_core::Error::Metal(error.to_string()))?;
                dispatch_p5b_swiglu(
                    batch,
                    &rg_bf16,
                    &ru_bf16,
                    &routed_swiglu,
                    selected_route_weight,
                    MOE_INTER_DIM as u32,
                    swiglu_threads,
                )?;
                dispatch_p5b_swiglu(
                    batch,
                    &sg_bf16,
                    &su_bf16,
                    &shared_swiglu,
                    1.0,
                    MOE_INTER_DIM as u32,
                    swiglu_threads,
                )?;
                Ok(())
            })?;
            require_batch_topology(&stage1, 7, 11, "P5B stage-1")?;

            let stage2 = context.dispatch_batch_timed(|batch| {
                dispatch_device_act_quant(
                    batch,
                    &routed_swiglu,
                    &routed_q_buf,
                    &routed_scale_buf,
                    MOE_INTER_DIM as u32,
                    qat_threads,
                )?;
                dispatch_device_act_quant(
                    batch,
                    &shared_swiglu,
                    &shared_q_buf,
                    &shared_scale_buf,
                    MOE_INTER_DIM as u32,
                    qat_threads,
                )?;
                dispatch_p5b_fp4_then_bf16(
                    batch,
                    &rd_w,
                    &rd_s,
                    &routed_q_buf,
                    &routed_scale_buf,
                    &rd_f32,
                    &rd_bf16,
                    &routed_down,
                    fp4_threads,
                    cast_threads,
                )
                .map_err(|error| hawking_core::Error::Metal(error.to_string()))?;
                dispatch_fp8_then_bf16(
                    batch,
                    &sd_w,
                    &sd_s,
                    &shared_q_buf,
                    &shared_scale_buf,
                    &sd_f32,
                    &sd_bf16,
                    &shared_down,
                    fp8_threads,
                    cast_threads,
                )
                .map_err(|error| hawking_core::Error::Metal(error.to_string()))?;
                dispatch_p5b_combine(
                    batch,
                    &rd_bf16,
                    &sd_bf16,
                    &combined_bf16,
                    HIDDEN_SIZE as u32,
                    combine_threads,
                )?;
                Ok(())
            })?;
            require_batch_topology(&stage2, 5, 7, "P5B stage-2")?;
            if iteration >= args.warmups {
                stage1_series.record(&stage1, "P5B stage-1")?;
                stage2_series.record(&stage2, "P5B stage-2")?;
                whole_gpu_us.push(
                    stage1.gpu_duration_us.expect("checked by TimingSeries")
                        + stage2.gpu_duration_us.expect("checked by TimingSeries"),
                );
                whole_wall_us.push(trial_started.elapsed().as_micros() as u64);
            }
        }

        // A passing P5B receipt requires exact source-storage bytes at every
        // device-only boundary.  This is a fail-closed parity gate, not merely
        // a final-output smoke test.
        let exact = P5BExactParity {
            input_q: read_gpu_u8(&input_q_buf, HIDDEN_SIZE)? == cpu.input_quant.activation_e4m3fn,
            input_scale: read_gpu_u8(&input_scale_buf, HIDDEN_SIZE / ACT_QUANT_BLOCK)?
                == cpu.input_quant.scales_e8m0fnu,
            routed_gate: read_gpu_u16(&rg_bf16, MOE_INTER_DIM)? == cpu.routed_gate.bf16_bits,
            routed_up: read_gpu_u16(&ru_bf16, MOE_INTER_DIM)? == cpu.routed_up.bf16_bits,
            shared_gate: read_gpu_u16(&sg_bf16, MOE_INTER_DIM)? == cpu.shared_gate.bf16_bits,
            shared_up: read_gpu_u16(&su_bf16, MOE_INTER_DIM)? == cpu.shared_up.bf16_bits,
            routed_swiglu: read_gpu_u16(&routed_swiglu, MOE_INTER_DIM)? == cpu.routed_swiglu_bf16,
            shared_swiglu: read_gpu_u16(&shared_swiglu, MOE_INTER_DIM)? == cpu.shared_swiglu_bf16,
            routed_down_q: read_gpu_u8(&routed_q_buf, MOE_INTER_DIM)?
                == cpu.routed_down_quant.activation_e4m3fn,
            routed_down_scale: read_gpu_u8(&routed_scale_buf, MOE_INTER_DIM / ACT_QUANT_BLOCK)?
                == cpu.routed_down_quant.scales_e8m0fnu,
            shared_down_q: read_gpu_u8(&shared_q_buf, MOE_INTER_DIM)?
                == cpu.shared_down_quant.activation_e4m3fn,
            shared_down_scale: read_gpu_u8(&shared_scale_buf, MOE_INTER_DIM / ACT_QUANT_BLOCK)?
                == cpu.shared_down_quant.scales_e8m0fnu,
            routed_down: read_gpu_u16(&rd_bf16, HIDDEN_SIZE)? == cpu.routed_down.bf16_bits,
            shared_down: read_gpu_u16(&sd_bf16, HIDDEN_SIZE)? == cpu.shared_down.bf16_bits,
            combined: read_gpu_u16(&combined_bf16, HIDDEN_SIZE)? == cpu.combined_bf16,
        };
        if !exact.all() {
            return Err(failure(
                "P5B device-only source-storage parity failed; no sealed receipt emitted",
            ));
        }
        let combined_device = read_gpu_u16(&combined_bf16, HIDDEN_SIZE)?;
        let combined_device_f32: Vec<f32> = combined_device
            .iter()
            .map(|value| bf16::from_bits(*value).to_f32())
            .collect();
        let device_vs_f64 = f64_metrics_from_f32(&f64.combined_f64, &combined_device_f32)?;

        let physical_counts = physical_trace.counts();
        drop(physical_trace);
        let (buffers_created, bytes_allocated, commits) = context.drain_stats();
        let trace_samples = context.drain_trace();
        let expected_commands = (total_trials * 2) as u64;
        let expected_encoders = (total_trials * 12) as u64;
        let expected_dispatches = (total_trials * 18) as u64;
        if physical_counts.command_count != expected_commands
            || physical_counts.encoder_count != expected_encoders
            || commits != expected_commands as usize
            || trace_samples.len() != expected_commands as usize
        {
            return Err(failure(
                "P5B physical command topology did not match the sealed graph",
            ));
        }

        let active_weight_bytes = routed_gate.raw_weight.len()
            + routed_gate.raw_scale.len()
            + routed_up.raw_weight.len()
            + routed_up.raw_scale.len()
            + routed_down.raw_weight.len()
            + routed_down.raw_scale.len()
            + shared_gate.raw_weight.len()
            + shared_gate.raw_scale.len()
            + shared_up.raw_weight.len()
            + shared_up.raw_scale.len()
            + shared_down.raw_weight.len()
            + shared_down.raw_scale.len();
        let unsigned = json!({
            "schema": "hawking.gravity.deepseek_v4.layer0_moe_metal_p5b_device_storage_chain.v1",
            "status": "PASS_REAL_METAL_DEVICE_SWIGLU_QAT_COMBINE_PARITY_NOT_FULL_RUNTIME",
            "scope": {
                "bounded_component": "layer-0 dominant hash-selected routed FP4 expert plus shared FP8 expert",
                "selected_route_is_dominant": true,
                "trusted_predecessor_input": "deterministic BF16 [4096] fixture; not an attention output, model forward, prompt, or generated token",
                "host_gate_route_control_only": true,
                "host_swiglu_qat_or_combine_in_measured_iterations": false,
                "full_decoder_layer_forward": false,
                "full_model_loaded": false,
                "full_model_forward": false,
                "generated_tokens": 0,
                "hcli_endpoint_started": false,
                "base_true_tps_measured": false,
                "claim_boundary": "This proves a bounded device-storage MoE substage only. It does not establish a full DeepSeek runtime, 43-layer causal loop, HCLI endpoint, generated token, or BASE_TRUE_TPS.",
            },
            "artifact": {
                "path": reader.artifact_root().display().to_string(),
                "full_stream_schema": FULL_STREAM_SCHEMA,
                "full_stream_status": FULL_STREAM_STATUS,
                "manifest_file_sha256": reader.manifest_file_sha256(),
                "manifest_seal_sha256": reader.manifest_seal_sha256(),
                "restart_receipt_seal_sha256": reader.restart_seal_sha256(),
                "source": {"repository": reader.source_identity().repository, "revision": reader.source_identity().revision, "source_parent_retained": false},
                "source_parent_materialized": false,
            },
            "route_control": {
                "token_id": PREFIX_TOKEN_ID,
                "fixed_hash_tid2eid_ids_in_top_slot_order": route.selected_expert_ids,
                "f64_authority_ids_exact_match": true,
                "dominant_slot": selected_slot,
                "dominant_expert_id": selected_expert,
                "dominant_route_weight_f32_text": f32_text(selected_route_weight),
                "dominant_route_weight_f64_text": f64_text(f64_route.selected_weights[selected_slot]),
                "low_weight_secondary_discrete_check": {
                    "slot": low_weight_slot,
                    "expert_id": route.selected_expert_ids[low_weight_slot],
                    "route_weight_f32_text": f32_text(low_weight),
                    "route_weight_f64_text": f64_text(f64_route.selected_weights[low_weight_slot]),
                    "id_matched_independent_f64_hash_row": true,
                    "not_used_as_primary_weighted_combine_signal": true,
                },
                "gate_logits_sha256_f32_le": sha256(&f32_le_bytes(&route.logits_f32)),
                "gate_logits_sha256_f64_le": sha256(&f64_le_bytes(&f64_route.logits)),
                "source_f32_vs_f64_route_weight_metrics": f64_vector_metrics(&f64_route.selected_weights, &route.selected_weights_f32)?,
            },
            "source_bindings": {
                "inference/model.py_sha256": reader.source_metadata_asset_sha256("inference/model.py")?,
                "inference/kernel.py_sha256": reader.source_metadata_asset_sha256("inference/kernel.py")?,
                "inference/config.json_sha256": reader.source_metadata_asset_sha256("inference/config.json")?,
                "config.json_sha256": reader.source_metadata_asset_sha256("config.json")?,
                "dominant_routed_fp4": {"w1": pair_binding_json(&routed_gate), "w3": pair_binding_json(&routed_up), "w2": pair_binding_json(&routed_down)},
                "shared_fp8": {"w1": pair_binding_json(&shared_gate), "w3": pair_binding_json(&shared_up), "w2": pair_binding_json(&shared_down)},
                "verified_content_addressed_chunks_before_gpu_upload": true,
            },
            "device_graph": {
                "stage_1": [
                    "device source act_quant(BF16 input)",
                    "device routed FP4 QAT W1/W3 -> FP32 -> device BF16 storage",
                    "device shared FP8 QAT W1/W3 -> FP32 -> device BF16 storage",
                    "device source-clamped routed SwiGLU with dominant route weight before W2",
                    "device source-clamped shared SwiGLU",
                ],
                "stage_2": [
                    "device source act_quant of each SwiGLU BF16 row",
                    "device routed FP4 QAT W2 -> FP32 -> device BF16 storage",
                    "device shared FP8 QAT W2 -> FP32 -> device BF16 storage",
                    "device source-order routed+shared BF16 combine and BF16 output storage",
                ],
                "host_boundary": "Gate score/hash selection is source-derived CPU control only; no host activation, QAT, SwiGLU, expert output, or combine runs between the two measured command buffers.",
                "hidden_fallback": false,
            },
            "numerical_parity": {
                "storage_byte_parity": exact.json(),
                "all_source_storage_boundaries_exact": true,
                "cpu_final_bf16_sha256": sha256(&u16_le_bytes(&cpu.combined_bf16)),
                "device_final_bf16_sha256": sha256(&u16_le_bytes(&combined_device)),
                "device_final_vs_separate_f64_authority": device_vs_f64,
                "separate_f64_authority": "independent f64 native-byte FP4/FP8 matvec arithmetic; source BF16/QAT boundaries reapplied exactly between stages",
            },
            "metal": {
                "device": device_name,
                "kernels": {"act_quant": ACT_QUANT_KERNEL, "routed_fp4_qat": P5B_FP4_KERNEL, "shared_fp8_qat": FP8_KERNEL, "bf16_storage": BF16_CAST_KERNEL, "swiglu": P5B_SWIGLU_KERNEL, "combine": P5B_COMBINE_KERNEL},
                "pipelines_precompiled_before_warmups": true,
                "thread_execution_width": {"act_quant": qat_width, "routed_fp4_qat": fp4_width, "shared_fp8_qat": fp8_width, "bf16_storage": cast_width, "swiglu": swiglu_width, "combine": combine_width},
                "threadgroup": {"act_quant": [qat_threads,1,1], "routed_fp4_qat": [fp4_threads,1,1], "shared_fp8_qat": [fp8_threads,1,1], "bf16_storage": [cast_threads,1,1], "swiglu": [swiglu_threads,1,1], "combine": [combine_threads,1,1]},
                "resident_buffers_created": buffers_created,
                "resident_bytes_allocated": bytes_allocated,
                "real_gpu_dispatches": expected_dispatches,
                "command_buffers": expected_commands,
                "compute_encoders": expected_encoders,
                "cpu_visible_waits": expected_commands,
                "empty_command_buffers": 0,
                "physical_trace_command_buffers": physical_counts.command_count,
                "physical_trace_compute_encoders": physical_counts.encoder_count,
                "trace_samples": trace_samples.len(),
                "hardware_occupancy": "NOT_EXPOSED_BY_THIS_METAL_TIMESTAMP_PROBE; no value inferred",
                "fallback": false,
            },
            "command_topology": {
                "stage_1_per_iteration": {"command_buffers":1,"compute_encoders":7,"compute_dispatches":11,"cpu_activation_handoff":false},
                "stage_2_per_iteration": {"command_buffers":1,"compute_encoders":5,"compute_dispatches":7,"cpu_activation_handoff":false},
                "per_iteration_total": {"command_buffers":2,"compute_encoders":12,"compute_dispatches":18,"cpu_visible_waits":2},
                "persistent_replay_graph": false,
            },
            "timing": {
                "warmups": args.warmups,
                "clean_trials": args.trials,
                "timestamp_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime; host waits reported separately",
                "stage_1": timing_json(&stage1_series)?,
                "stage_2": timing_json(&stage2_series)?,
                "whole_device_chain_gpu": summary_json(&whole_gpu_us)?,
                "whole_device_chain_host_wall": summary_json(&whole_wall_us)?,
                "host_interstage_swiglu_qat_combine": "NONE; moved to device in this P5B graph",
            },
            "active_bytes": {
                "selected_active_weight_and_scale_bytes": active_weight_bytes,
                "input_qat_bytes": HIDDEN_SIZE + HIDDEN_SIZE / ACT_QUANT_BLOCK,
                "per_expert_intermediate_bf16_bytes": MOE_INTER_DIM * std::mem::size_of::<u16>(),
                "final_bf16_bytes": HIDDEN_SIZE * std::mem::size_of::<u16>(),
            },
            "resume": {"command": format!("cargo run --release -p hawking-core --example gravity_deepseek_v4_layer0_moe_metal_p5a -- --p5b --artifact {} --out {} --warmups {} --trials {}", reader.artifact_root().display(), args.out.display(), args.warmups, args.trials)},
        });
        let (receipt, seal) = seal(unsigned)?;
        write_new_receipt(&args.out, &receipt)?;
        println!(
            "{}",
            serde_json::to_string(&json!({
                "status": "PASS_REAL_METAL_DEVICE_SWIGLU_QAT_COMBINE_PARITY_NOT_FULL_RUNTIME",
                "receipt": args.out,
                "seal_sha256": seal,
                "dominant_expert": selected_expert,
                "dominant_route_weight": f32_text(selected_route_weight),
                "real_gpu_dispatches": expected_dispatches,
            }))?
        );
        Ok(())
    }

    /// P6A/P6B bounded full-route proof.  It drives the layer-0 Gate and
    /// hash-table route control on Metal, retains all six selected FP4 bundles
    /// simultaneously, and executes their W1/W3, route-weighted SwiGLU, W2,
    /// source-order combine, plus the FP8 shared expert without a host
    /// activation boundary.  The only predecessor is the declared fixture;
    /// this is not a registered decoder layer or TPS result.
    pub fn run_p6a() -> ProbeResult<()> {
        let args = parse_args()?;
        let reader = DeepSeekV4FullStreamReader::admit(&args.artifact)?;
        let _anchors = verify_layer0_moe_source_anchors(&reader)?;
        let input_bf16 = deterministic_wq_a_input_bf16();
        if input_bf16.len() != HIDDEN_SIZE {
            return Err(failure("P6A predecessor fixture is not BF16[4096]"));
        }
        let route = layer0_hash_route_cpu_oracle(&reader, &input_bf16)?;
        validate_source_route(&route)?;
        let f64_route = f64_gate_authority(&reader, &input_bf16)?;
        if f64_route.selected_ids != route.selected_expert_ids {
            return Err(failure(
                "P6A independent f64 Gate authority disagrees with source tid2eid IDs",
            ));
        }
        let source_route_margins = source_route_margin_json(&route, &f64_route)?;
        let cpu = p6a_full_route_cpu_pipeline(&reader, &input_bf16, &route)?;
        if cpu.routed.len() != ACTIVATED_EXPERTS
            || cpu
                .routed
                .windows(2)
                .any(|pair| pair[0].expert_id >= pair[1].expert_id)
        {
            return Err(failure(
                "P6A resident routed bundles are not six strictly numeric-ID ordered experts",
            ));
        }
        let token_id = u32::try_from(PREFIX_TOKEN_ID)
            .map_err(|_| failure("P6A source token ID does not fit a device u32"))?;

        let gate_tensor = reader.tensor_metadata(LAYER0_FFN_GATE_WEIGHT)?;
        let gate_bytes = reader.read_verified_full(
            LAYER0_FFN_GATE_WEIGHT,
            ROUTED_EXPERTS * HIDDEN_SIZE * std::mem::size_of::<u16>(),
        )?;
        if gate_bytes.len() != ROUTED_EXPERTS * HIDDEN_SIZE * std::mem::size_of::<u16>() {
            return Err(failure(
                "P6A Gate BF16 source payload has invalid byte length",
            ));
        }
        let tid2eid_tensor = reader.tensor_metadata(LAYER0_FFN_GATE_TID2EID)?;
        let tid2eid_bytes =
            reader.read_verified_full(LAYER0_FFN_GATE_TID2EID, tid2eid_tensor.bytes as usize)?;
        if tid2eid_bytes.len() % std::mem::size_of::<i64>() != 0
            || tid2eid_bytes.len()
                < (usize::try_from(PREFIX_TOKEN_ID).map_err(|_| failure("P6A token usize"))? + 1)
                    * ACTIVATED_EXPERTS
                    * std::mem::size_of::<i64>()
        {
            return Err(failure(
                "P6A full tid2eid source table has invalid geometry",
            ));
        }

        let context = MetalContext::new_with_trace(true)?;
        let device_name = context.device_name();
        let qat_pipeline = context.pipeline(ACT_QUANT_KERNEL)?;
        let fp4_pipeline = context.pipeline(P5B_FP4_KERNEL)?;
        let fp8_pipeline = context.pipeline(FP8_KERNEL)?;
        let cast_pipeline = context.pipeline(BF16_CAST_KERNEL)?;
        let gate_pipeline = context.pipeline(P6A_GATE_KERNEL)?;
        let route_pipeline = context.pipeline(P6A_ROUTE_KERNEL)?;
        let swiglu_pipeline = context.pipeline(P6A_SWIGLU_KERNEL)?;
        let shared_swiglu_pipeline = context.pipeline(P5B_SWIGLU_KERNEL)?;
        let combine_pipeline = context.pipeline(P6A_COMBINE_KERNEL)?;
        let dominant_combine_pipeline = context.pipeline(P5B_COMBINE_KERNEL)?;
        let qat_width = qat_pipeline.thread_execution_width() as u64;
        let fp4_width = fp4_pipeline.thread_execution_width() as u64;
        let fp8_width = fp8_pipeline.thread_execution_width() as u64;
        let cast_width = cast_pipeline.thread_execution_width() as u64;
        let gate_width = gate_pipeline.thread_execution_width() as u64;
        let route_width = route_pipeline.thread_execution_width() as u64;
        let swiglu_width = swiglu_pipeline.thread_execution_width() as u64;
        let shared_swiglu_width = shared_swiglu_pipeline.thread_execution_width() as u64;
        let combine_width = combine_pipeline.thread_execution_width() as u64;
        let dominant_combine_width = dominant_combine_pipeline.thread_execution_width() as u64;
        let qat_max = qat_pipeline.max_total_threads_per_threadgroup() as u32;
        let fp4_max = fp4_pipeline.max_total_threads_per_threadgroup() as u32;
        let fp8_max = fp8_pipeline.max_total_threads_per_threadgroup() as u32;
        let cast_max = cast_pipeline.max_total_threads_per_threadgroup() as u32;
        let gate_max = gate_pipeline.max_total_threads_per_threadgroup() as u32;
        let route_max = route_pipeline.max_total_threads_per_threadgroup() as u32;
        let swiglu_max = swiglu_pipeline.max_total_threads_per_threadgroup() as u32;
        let shared_swiglu_max = shared_swiglu_pipeline.max_total_threads_per_threadgroup() as u32;
        let combine_max = combine_pipeline.max_total_threads_per_threadgroup() as u32;
        let dominant_combine_max =
            dominant_combine_pipeline.max_total_threads_per_threadgroup() as u32;
        drop(qat_pipeline);
        drop(fp4_pipeline);
        drop(fp8_pipeline);
        drop(cast_pipeline);
        drop(gate_pipeline);
        drop(route_pipeline);
        drop(swiglu_pipeline);
        drop(shared_swiglu_pipeline);
        drop(combine_pipeline);
        drop(dominant_combine_pipeline);
        let qat_threads = require_threads(qat_max, 32, ACT_QUANT_KERNEL)?;
        let fp4_threads = require_256_threads(fp4_max, P5B_FP4_KERNEL)?;
        let fp8_threads = require_256_threads(fp8_max, FP8_KERNEL)?;
        let cast_threads = require_256_threads(cast_max, BF16_CAST_KERNEL)?;
        let gate_threads = require_threads(gate_max, 32, P6A_GATE_KERNEL)?;
        let route_threads = require_threads(route_max, 1, P6A_ROUTE_KERNEL)?;
        let swiglu_threads = require_256_threads(swiglu_max, P6A_SWIGLU_KERNEL)?;
        let shared_swiglu_threads = require_256_threads(shared_swiglu_max, P5B_SWIGLU_KERNEL)?;
        let combine_threads = require_256_threads(combine_max, P6A_COMBINE_KERNEL)?;
        let dominant_combine_threads =
            require_256_threads(dominant_combine_max, P5B_COMBINE_KERNEL)?;

        // Source bytes are admitted and content-address verified before any
        // resident Metal allocation. The entire tid2eid table is kept on
        // device so the route kernel genuinely gathers by token ID.
        let gate_weight_buf = context.new_buffer_with_bytes_checked(&gate_bytes)?;
        let tid2eid_buf = context.new_buffer_with_bytes_checked(&tid2eid_bytes)?;
        let input_bf16_buf = context.new_buffer_with_bytes_checked(&u16_le_bytes(&input_bf16))?;
        let input_q_buf = context.new_buffer_checked(HIDDEN_SIZE)?;
        let input_scale_buf = context.new_buffer_checked(HIDDEN_SIZE / ACT_QUANT_BLOCK)?;
        let gate_logits_buf =
            context.new_buffer_checked(ROUTED_EXPERTS * std::mem::size_of::<f32>())?;
        let route_ids_buf =
            context.new_buffer_checked(ACTIVATED_EXPERTS * std::mem::size_of::<u32>())?;
        let route_weights_buf =
            context.new_buffer_checked(ACTIVATED_EXPERTS * std::mem::size_of::<f32>())?;
        let route_scores_buf =
            context.new_buffer_checked(ROUTED_EXPERTS * std::mem::size_of::<f32>())?;
        let route_valid_buf = context.new_buffer_checked(std::mem::size_of::<u32>())?;
        let routed_gpu: Vec<P6aExpertGpuBuffers> = cpu
            .routed
            .iter()
            .map(|expert| allocate_p6a_routed_gpu(&context, expert))
            .collect::<ProbeResult<_>>()?;
        let shared_gpu = allocate_p6a_shared_gpu(&context, &cpu)?;
        let combined_full_buf =
            context.new_buffer_checked(HIDDEN_SIZE * std::mem::size_of::<u16>())?;
        let combined_dominant_buf =
            context.new_buffer_checked(HIDDEN_SIZE * std::mem::size_of::<u16>())?;

        // Route-control preflight is intentionally outside the measured
        // six-expert wave. It proves the first device control boundary before
        // any routed activation is produced, so a Gate/ln_1p/tid2eid defect
        // cannot be mistaken for an expert, cache, or command-topology issue.
        let control_preflight = context.dispatch_batch_timed(|batch| {
            dispatch_p6a_gate(
                batch,
                &gate_weight_buf,
                &input_bf16_buf,
                &gate_logits_buf,
                ROUTED_EXPERTS as u32,
                HIDDEN_SIZE as u32,
                gate_threads,
            )?;
            dispatch_p6a_route(
                batch,
                &gate_logits_buf,
                &tid2eid_buf,
                &route_ids_buf,
                &route_weights_buf,
                &route_scores_buf,
                &route_valid_buf,
                token_id,
                route_threads,
            )?;
            Ok(())
        })?;
        require_batch_topology(&control_preflight, 2, 2, "P6A control preflight")?;
        let preflight_logits = read_gpu_f32(&gate_logits_buf, ROUTED_EXPERTS)?;
        let preflight_ids = read_gpu_u32(&route_ids_buf, ACTIVATED_EXPERTS)?;
        let preflight_weights = read_gpu_f32(&route_weights_buf, ACTIVATED_EXPERTS)?;
        let preflight_scores = read_gpu_f32(&route_scores_buf, ROUTED_EXPERTS)?;
        let preflight_valid = read_gpu_u32(&route_valid_buf, 1)?;
        let expected_preflight_ids: Vec<u32> = route
            .selected_expert_ids
            .iter()
            .copied()
            .map(|id| u32::try_from(id).map_err(|_| failure("source expert id exceeds u32")))
            .collect::<ProbeResult<_>>()?;
        // Numeric Parity V2.1 deliberately scores both F32 backends against
        // the independently accumulated raw-BF16 FP64 authority. Metal and
        // libm may differ by a last ULP in exp/sqrt; CPU F32 is not promoted to
        // authority merely because it happens to match a particular device.
        // Discrete Gate/tid2eid decisions remain exact hard gates.
        let default_route_v21_bounds = Bounds::continuous_only();
        let default_route_logits_v21 = score_pair(
            &route.logits_f32,
            &preflight_logits,
            &f64_route.logits,
            &default_route_v21_bounds,
        );
        let default_route_scores_v21 = score_pair(
            &route.original_scores_f32,
            &preflight_scores,
            &f64_route.original_scores,
            &default_route_v21_bounds,
        );
        let default_route_weights_v21 = score_pair(
            &route.selected_weights_f32,
            &preflight_weights,
            &f64_route.selected_weights,
            &default_route_v21_bounds,
        );
        let default_v21_rejected_only_for_max_meaningful_rel =
            v21_rejected_only_for_max_meaningful_rel(&default_route_logits_v21)
                && v21_rejected_only_for_max_meaningful_rel(&default_route_scores_v21)
                && v21_rejected_only_for_max_meaningful_rel(&default_route_weights_v21);
        if !default_v21_rejected_only_for_max_meaningful_rel {
            return Err(failure(
                "P6A default V2.1 calibration probe did not fail solely at the expected max_meaningful_rel gate",
            ));
        }
        let route_v21_bounds = p6a_route_control_v21_bounds();
        let route_logits_v21 = score_pair(
            &route.logits_f32,
            &preflight_logits,
            &f64_route.logits,
            &route_v21_bounds,
        );
        let route_scores_v21 = score_pair(
            &route.original_scores_f32,
            &preflight_scores,
            &f64_route.original_scores,
            &route_v21_bounds,
        );
        let route_weights_v21 = score_pair(
            &route.selected_weights_f32,
            &preflight_weights,
            &f64_route.selected_weights,
            &route_v21_bounds,
        );
        let route_discrete_exact = preflight_logits == route.logits_f32
            && preflight_ids == expected_preflight_ids
            && preflight_valid == vec![1u32];
        if !route_discrete_exact
            || !route_logits_v21.pass
            || !route_scores_v21.pass
            || !route_weights_v21.pass
        {
            return Err(failure(format!(
                "P6A device Gate/route preflight failed before expert-wave execution; no receipt emitted; diagnostic={}",
                serde_json::to_string(&json!({
                    "numeric_parity_v2_1": {
                        "bounds": serde_json::to_value(route_v21_bounds)?,
                        "gate_logits": serde_json::to_value(&route_logits_v21)?,
                        "sqrtsoftplus_scores": serde_json::to_value(&route_scores_v21)?,
                        "normalized_route_weights": serde_json::to_value(&route_weights_v21)?,
                    },
                    "gate_logits": f32_bit_diagnostic(&route.logits_f32, &preflight_logits),
                    "sqrtsoftplus_scores": f32_bit_diagnostic(&route.original_scores_f32, &preflight_scores),
                    "selected_ids": u32_bit_diagnostic(&expected_preflight_ids, &preflight_ids),
                    "normalized_route_weights": f32_bit_diagnostic(&route.selected_weights_f32, &preflight_weights),
                    "valid_word": u32_bit_diagnostic(&[1u32], &preflight_valid),
                }))?
            )));
        }
        let (preflight_buffers_created, preflight_bytes_allocated, preflight_commits) =
            context.drain_stats();
        let preflight_trace_samples = context.drain_trace();
        if preflight_commits != 1 || preflight_trace_samples.len() != 1 {
            return Err(failure(
                "P6A control preflight did not report its one command-buffer topology",
            ));
        }
        println!(
            "{}",
            serde_json::to_string(&json!({
                "status": "P6A_DEVICE_GATE_ROUTE_PREFLIGHT_V21_PASS_NOT_A_RECEIPT",
                "gate_logits_sha256_f32_le": sha256(&f32_le_bytes(&preflight_logits)),
                "scores_sha256_f32_le": sha256(&f32_le_bytes(&preflight_scores)),
                "selected_ids": preflight_ids,
                "selected_weights_f32_text": f32_texts(&preflight_weights),
                "gate_logits_bit_exact_to_source_cpu": preflight_logits == route.logits_f32,
                "score_bits_exact_to_source_cpu": preflight_scores == route.original_scores_f32,
                "weight_bits_exact_to_source_cpu": preflight_weights == route.selected_weights_f32,
                "numeric_parity_v2_1_all_host_and_device_scores_pass": route_logits_v21.pass && route_scores_v21.pass && route_weights_v21.pass,
                "command_buffers": 1,
                "compute_encoders": 2,
                "compute_dispatches": 2,
            }))?
        );

        let run_nonce = sha256_join(&[
            reader.manifest_seal_sha256(),
            &sha256(&u16_le_bytes(&input_bf16)),
            &sha256(&gate_bytes),
            &sha256(&tid2eid_bytes),
            &cpu.routed
                .iter()
                .map(|expert| expert.expert_id.to_string())
                .collect::<Vec<_>>()
                .join(","),
            "layer0_moe_p6a_full_route_v1",
        ]);
        let interval_id = sha256_join(&[
            &run_nonce,
            P6A_GATE_KERNEL,
            P6A_ROUTE_KERNEL,
            P5B_FP4_KERNEL,
            FP8_KERNEL,
            P6A_SWIGLU_KERNEL,
            P6A_COMBINE_KERNEL,
            "all_six_hot_bundles_resident_concurrent_waves",
        ]);
        let physical_trace = PhysicalTraceGuard::begin(PhysicalTraceIdentity::new(
            interval_id,
            run_nonce,
            "layer0_moe_p6a".to_owned(),
            "full_hash_route_six_routed_fp4_plus_shared_fp8_device_chain".to_owned(),
            Some(1),
            0,
        )?)?;
        let total_trials = args.warmups + args.trials;
        let mut stage1_series = TimingSeries::with_capacity(args.trials);
        let mut stage2_series = TimingSeries::with_capacity(args.trials);
        let mut whole_gpu_us = Vec::with_capacity(args.trials);
        let mut whole_wall_us = Vec::with_capacity(args.trials);

        for iteration in 0..total_trials {
            let trial_started = Instant::now();
            let stage1 = context.dispatch_batch_timed(|batch| {
                dispatch_p6a_gate(
                    batch,
                    &gate_weight_buf,
                    &input_bf16_buf,
                    &gate_logits_buf,
                    ROUTED_EXPERTS as u32,
                    HIDDEN_SIZE as u32,
                    gate_threads,
                )?;
                dispatch_device_act_quant(
                    batch,
                    &input_bf16_buf,
                    &input_q_buf,
                    &input_scale_buf,
                    HIDDEN_SIZE as u32,
                    qat_threads,
                )?;
                dispatch_p6a_route(
                    batch,
                    &gate_logits_buf,
                    &tid2eid_buf,
                    &route_ids_buf,
                    &route_weights_buf,
                    &route_scores_buf,
                    &route_valid_buf,
                    token_id,
                    route_threads,
                )?;

                // P6A's W1/W3 wave: every selected FP4 bundle is already
                // resident, reads the same device QAT input, and owns distinct
                // FP32 output ranges. One concurrent encoder makes the
                // independence and no-eviction contract executable.
                batch.begin_concurrent_group()?;
                for (expert, gpu) in cpu.routed.iter().zip(&routed_gpu) {
                    dispatch_p5b_fp4_concurrent(
                        batch,
                        &gpu.w1_weight,
                        &gpu.w1_scale,
                        &input_q_buf,
                        &input_scale_buf,
                        &gpu.gate_f32,
                        &expert.w1,
                        fp4_threads,
                    )
                    .map_err(|error| hawking_core::Error::Metal(error.to_string()))?;
                    dispatch_p5b_fp4_concurrent(
                        batch,
                        &gpu.w3_weight,
                        &gpu.w3_scale,
                        &input_q_buf,
                        &input_scale_buf,
                        &gpu.up_f32,
                        &expert.w3,
                        fp4_threads,
                    )
                    .map_err(|error| hawking_core::Error::Metal(error.to_string()))?;
                }
                dispatch_fp8_concurrent(
                    batch,
                    &shared_gpu.w1_weight,
                    &shared_gpu.w1_scale,
                    &input_q_buf,
                    &input_scale_buf,
                    &shared_gpu.gate_f32,
                    &cpu.shared_w1,
                    fp8_threads,
                )
                .map_err(|error| hawking_core::Error::Metal(error.to_string()))?;
                dispatch_fp8_concurrent(
                    batch,
                    &shared_gpu.w3_weight,
                    &shared_gpu.w3_scale,
                    &input_q_buf,
                    &input_scale_buf,
                    &shared_gpu.up_f32,
                    &cpu.shared_w3,
                    fp8_threads,
                )
                .map_err(|error| hawking_core::Error::Metal(error.to_string()))?;
                batch.end_concurrent_group()?;

                batch.begin_concurrent_group()?;
                for gpu in &routed_gpu {
                    dispatch_bf16_cast_concurrent(
                        batch,
                        &gpu.gate_f32,
                        &gpu.gate_bf16,
                        MOE_INTER_DIM as u32,
                        cast_threads,
                    )?;
                    dispatch_bf16_cast_concurrent(
                        batch,
                        &gpu.up_f32,
                        &gpu.up_bf16,
                        MOE_INTER_DIM as u32,
                        cast_threads,
                    )?;
                }
                dispatch_bf16_cast_concurrent(
                    batch,
                    &shared_gpu.gate_f32,
                    &shared_gpu.gate_bf16,
                    MOE_INTER_DIM as u32,
                    cast_threads,
                )?;
                dispatch_bf16_cast_concurrent(
                    batch,
                    &shared_gpu.up_f32,
                    &shared_gpu.up_bf16,
                    MOE_INTER_DIM as u32,
                    cast_threads,
                )?;
                batch.end_concurrent_group()?;

                batch.begin_concurrent_group()?;
                for (expert, gpu) in cpu.routed.iter().zip(&routed_gpu) {
                    let route_slot = u32::try_from(expert.source_top_slot).map_err(|_| {
                        hawking_core::Error::Metal("P6A source route slot does not fit u32".into())
                    })?;
                    dispatch_p6a_swiglu_concurrent(
                        batch,
                        &gpu.gate_bf16,
                        &gpu.up_bf16,
                        &gpu.swiglu_bf16,
                        &route_weights_buf,
                        route_slot,
                        MOE_INTER_DIM as u32,
                        swiglu_threads,
                    )?;
                }
                dispatch_p5b_swiglu_concurrent(
                    batch,
                    &shared_gpu.gate_bf16,
                    &shared_gpu.up_bf16,
                    &shared_gpu.swiglu_bf16,
                    1.0,
                    MOE_INTER_DIM as u32,
                    shared_swiglu_threads,
                )?;
                batch.end_concurrent_group()?;
                Ok(())
            })?;
            require_batch_topology(&stage1, 6, 38, "P6A stage-1")?;

            let stage2 = context.dispatch_batch_timed(|batch| {
                batch.begin_concurrent_group()?;
                for gpu in &routed_gpu {
                    dispatch_device_act_quant_concurrent(
                        batch,
                        &gpu.swiglu_bf16,
                        &gpu.down_quant,
                        &gpu.down_scales,
                        MOE_INTER_DIM as u32,
                        qat_threads,
                    )?;
                }
                dispatch_device_act_quant_concurrent(
                    batch,
                    &shared_gpu.swiglu_bf16,
                    &shared_gpu.down_quant,
                    &shared_gpu.down_scales,
                    MOE_INTER_DIM as u32,
                    qat_threads,
                )?;
                batch.end_concurrent_group()?;

                batch.begin_concurrent_group()?;
                for (expert, gpu) in cpu.routed.iter().zip(&routed_gpu) {
                    dispatch_p5b_fp4_concurrent(
                        batch,
                        &gpu.w2_weight,
                        &gpu.w2_scale,
                        &gpu.down_quant,
                        &gpu.down_scales,
                        &gpu.down_f32,
                        &expert.w2,
                        fp4_threads,
                    )
                    .map_err(|error| hawking_core::Error::Metal(error.to_string()))?;
                }
                dispatch_fp8_concurrent(
                    batch,
                    &shared_gpu.w2_weight,
                    &shared_gpu.w2_scale,
                    &shared_gpu.down_quant,
                    &shared_gpu.down_scales,
                    &shared_gpu.down_f32,
                    &cpu.shared_w2,
                    fp8_threads,
                )
                .map_err(|error| hawking_core::Error::Metal(error.to_string()))?;
                batch.end_concurrent_group()?;

                batch.begin_concurrent_group()?;
                for gpu in &routed_gpu {
                    dispatch_bf16_cast_concurrent(
                        batch,
                        &gpu.down_f32,
                        &gpu.down_bf16,
                        HIDDEN_SIZE as u32,
                        cast_threads,
                    )?;
                }
                dispatch_bf16_cast_concurrent(
                    batch,
                    &shared_gpu.down_f32,
                    &shared_gpu.down_bf16,
                    HIDDEN_SIZE as u32,
                    cast_threads,
                )?;
                batch.end_concurrent_group()?;

                batch.begin_concurrent_group()?;
                let routed_downs: [&metal::Buffer; ACTIVATED_EXPERTS] = [
                    &routed_gpu[0].down_bf16,
                    &routed_gpu[1].down_bf16,
                    &routed_gpu[2].down_bf16,
                    &routed_gpu[3].down_bf16,
                    &routed_gpu[4].down_bf16,
                    &routed_gpu[5].down_bf16,
                ];
                dispatch_p6a_combine6_concurrent(
                    batch,
                    routed_downs,
                    &shared_gpu.down_bf16,
                    &combined_full_buf,
                    HIDDEN_SIZE as u32,
                    combine_threads,
                )?;
                let dominant_index = cpu
                    .routed
                    .iter()
                    .position(|expert| expert.source_top_slot == cpu.dominant_source_top_slot)
                    .ok_or_else(|| {
                        hawking_core::Error::Metal("P6A dominant GPU bundle absent".into())
                    })?;
                dispatch_p5b_combine_concurrent(
                    batch,
                    &routed_gpu[dominant_index].down_bf16,
                    &shared_gpu.down_bf16,
                    &combined_dominant_buf,
                    HIDDEN_SIZE as u32,
                    dominant_combine_threads,
                )?;
                batch.end_concurrent_group()?;
                Ok(())
            })?;
            require_batch_topology(&stage2, 4, 23, "P6A stage-2")?;
            if iteration >= args.warmups {
                stage1_series.record(&stage1, "P6A stage-1")?;
                stage2_series.record(&stage2, "P6A stage-2")?;
                whole_gpu_us.push(
                    stage1.gpu_duration_us.expect("checked by TimingSeries")
                        + stage2.gpu_duration_us.expect("checked by TimingSeries"),
                );
                whole_wall_us.push(trial_started.elapsed().as_micros() as u64);
            }
        }

        let gate_logits = read_gpu_f32(&gate_logits_buf, ROUTED_EXPERTS)?;
        let route_ids = read_gpu_u32(&route_ids_buf, ACTIVATED_EXPERTS)?;
        let route_weights = read_gpu_f32(&route_weights_buf, ACTIVATED_EXPERTS)?;
        let route_scores = read_gpu_f32(&route_scores_buf, ROUTED_EXPERTS)?;
        let route_valid = read_gpu_u32(&route_valid_buf, 1)?;
        let expected_ids: Vec<u32> = route
            .selected_expert_ids
            .iter()
            .copied()
            .map(|id| u32::try_from(id).map_err(|_| failure("source expert id exceeds u32")))
            .collect::<ProbeResult<_>>()?;
        let final_logits_v21 = score_pair(
            &route.logits_f32,
            &gate_logits,
            &f64_route.logits,
            &route_v21_bounds,
        );
        let final_scores_v21 = score_pair(
            &route.original_scores_f32,
            &route_scores,
            &f64_route.original_scores,
            &route_v21_bounds,
        );
        let final_weights_v21 = score_pair(
            &route.selected_weights_f32,
            &route_weights,
            &f64_route.selected_weights,
            &route_v21_bounds,
        );
        let control_exact = json!({
            "gate_logits_f32_bits_exact": gate_logits == route.logits_f32,
            "hash_tid2eid_ids_exact": route_ids == expected_ids,
            "sqrtsoftplus_scores_f32_bits_exact": route_scores == route.original_scores_f32,
            "normalized_route_weights_f32_bits_exact": route_weights == route.selected_weights_f32,
            "device_route_valid_word": route_valid,
            "device_route_valid": route_valid == vec![1u32],
        });
        let control_discrete_exact = gate_logits == route.logits_f32
            && route_ids == expected_ids
            && route_valid == vec![1u32];
        let control_v21_pass =
            final_logits_v21.pass && final_scores_v21.pass && final_weights_v21.pass;

        let mut routed_storage = Vec::with_capacity(ACTIVATED_EXPERTS);
        let mut routed_storage_all_exact = true;
        for (expert, gpu) in cpu.routed.iter().zip(&routed_gpu) {
            let flags = json!({
                "expert_id": expert.expert_id,
                "source_top_slot": expert.source_top_slot,
                "w1_bf16": read_gpu_u16(&gpu.gate_bf16, MOE_INTER_DIM)? == expert.gate.bf16_bits,
                "w3_bf16": read_gpu_u16(&gpu.up_bf16, MOE_INTER_DIM)? == expert.up.bf16_bits,
                "route_weighted_swiglu_bf16": read_gpu_u16(&gpu.swiglu_bf16, MOE_INTER_DIM)? == expert.swiglu_bf16,
                "w2_input_q_e4m3fn": read_gpu_u8(&gpu.down_quant, MOE_INTER_DIM)? == expert.down_quant.activation_e4m3fn,
                "w2_input_q_e8m0": read_gpu_u8(&gpu.down_scales, MOE_INTER_DIM / ACT_QUANT_BLOCK)? == expert.down_quant.scales_e8m0fnu,
                "w2_bf16": read_gpu_u16(&gpu.down_bf16, HIDDEN_SIZE)? == expert.down.bf16_bits,
            });
            routed_storage_all_exact &= flags
                .as_object()
                .ok_or_else(|| failure("P6A routed parity entry is not an object"))?
                .iter()
                .filter(|(key, _)| !matches!(key.as_str(), "expert_id" | "source_top_slot"))
                .all(|(_, value)| value.as_bool() == Some(true));
            routed_storage.push(flags);
        }
        let shared_storage = json!({
            "w1_bf16": read_gpu_u16(&shared_gpu.gate_bf16, MOE_INTER_DIM)? == cpu.shared_gate.bf16_bits,
            "w3_bf16": read_gpu_u16(&shared_gpu.up_bf16, MOE_INTER_DIM)? == cpu.shared_up.bf16_bits,
            "swiglu_bf16": read_gpu_u16(&shared_gpu.swiglu_bf16, MOE_INTER_DIM)? == cpu.shared_swiglu_bf16,
            "w2_input_q_e4m3fn": read_gpu_u8(&shared_gpu.down_quant, MOE_INTER_DIM)? == cpu.shared_down_quant.activation_e4m3fn,
            "w2_input_q_e8m0": read_gpu_u8(&shared_gpu.down_scales, MOE_INTER_DIM / ACT_QUANT_BLOCK)? == cpu.shared_down_quant.scales_e8m0fnu,
            "w2_bf16": read_gpu_u16(&shared_gpu.down_bf16, HIDDEN_SIZE)? == cpu.shared_down.bf16_bits,
        });
        let shared_storage_all_exact = shared_storage
            .as_object()
            .ok_or_else(|| failure("P6A shared parity entry is not an object"))?
            .values()
            .all(|value| value.as_bool() == Some(true));
        let full_final = read_gpu_u16(&combined_full_buf, HIDDEN_SIZE)?;
        let dominant_final = read_gpu_u16(&combined_dominant_buf, HIDDEN_SIZE)?;
        let full_final_exact = full_final == cpu.combined_bf16;
        let dominant_final_exact = dominant_final == cpu.dominant_plus_shared_bf16;
        let input_q_exact =
            read_gpu_u8(&input_q_buf, HIDDEN_SIZE)? == cpu.input_quant.activation_e4m3fn;
        let input_scale_exact = read_gpu_u8(&input_scale_buf, HIDDEN_SIZE / ACT_QUANT_BLOCK)?
            == cpu.input_quant.scales_e8m0fnu;
        let storage_all_exact = input_q_exact
            && input_scale_exact
            && routed_storage_all_exact
            && shared_storage_all_exact
            && full_final_exact
            && dominant_final_exact;
        if !control_discrete_exact || !control_v21_pass || !storage_all_exact {
            // Fail closed, but preserve bounded hashes and first-value deltas
            // for every source-storage boundary so a later change cannot
            // mistake a transcendental, I64 gather, or command dependency for
            // a generic "MoE mismatch". No raw activation payload is emitted.
            let routed_storage_diagnostics: Vec<Value> = cpu
                .routed
                .iter()
                .zip(&routed_gpu)
                .map(|(expert, gpu)| -> ProbeResult<Value> {
                    Ok(json!({
                        "expert_id": expert.expert_id,
                        "source_top_slot": expert.source_top_slot,
                        "w1_bf16": u16_bit_diagnostic(&expert.gate.bf16_bits, &read_gpu_u16(&gpu.gate_bf16, MOE_INTER_DIM)?),
                        "w3_bf16": u16_bit_diagnostic(&expert.up.bf16_bits, &read_gpu_u16(&gpu.up_bf16, MOE_INTER_DIM)?),
                        "route_weighted_swiglu_bf16": u16_bit_diagnostic(&expert.swiglu_bf16, &read_gpu_u16(&gpu.swiglu_bf16, MOE_INTER_DIM)?),
                        "w2_input_q_e4m3fn": u8_bit_diagnostic(&expert.down_quant.activation_e4m3fn, &read_gpu_u8(&gpu.down_quant, MOE_INTER_DIM)?),
                        "w2_input_q_e8m0": u8_bit_diagnostic(&expert.down_quant.scales_e8m0fnu, &read_gpu_u8(&gpu.down_scales, MOE_INTER_DIM / ACT_QUANT_BLOCK)?),
                        "w2_bf16": u16_bit_diagnostic(&expert.down.bf16_bits, &read_gpu_u16(&gpu.down_bf16, HIDDEN_SIZE)?),
                    }))
                })
                .collect::<ProbeResult<_>>()?;
            let shared_storage_diagnostic = json!({
                "w1_bf16": u16_bit_diagnostic(&cpu.shared_gate.bf16_bits, &read_gpu_u16(&shared_gpu.gate_bf16, MOE_INTER_DIM)?),
                "w3_bf16": u16_bit_diagnostic(&cpu.shared_up.bf16_bits, &read_gpu_u16(&shared_gpu.up_bf16, MOE_INTER_DIM)?),
                "swiglu_bf16": u16_bit_diagnostic(&cpu.shared_swiglu_bf16, &read_gpu_u16(&shared_gpu.swiglu_bf16, MOE_INTER_DIM)?),
                "w2_input_q_e4m3fn": u8_bit_diagnostic(&cpu.shared_down_quant.activation_e4m3fn, &read_gpu_u8(&shared_gpu.down_quant, MOE_INTER_DIM)?),
                "w2_input_q_e8m0": u8_bit_diagnostic(&cpu.shared_down_quant.scales_e8m0fnu, &read_gpu_u8(&shared_gpu.down_scales, MOE_INTER_DIM / ACT_QUANT_BLOCK)?),
                "w2_bf16": u16_bit_diagnostic(&cpu.shared_down.bf16_bits, &read_gpu_u16(&shared_gpu.down_bf16, HIDDEN_SIZE)?),
            });
            return Err(failure(format!(
                "P6A device Gate/route or source-storage parity failed; no receipt emitted; diagnostic={}",
                serde_json::to_string(&json!({
                    "control": {
                        "booleans": control_exact,
                        "numeric_parity_v2_1": {
                            "bounds": serde_json::to_value(route_v21_bounds)?,
                            "gate_logits": serde_json::to_value(&final_logits_v21)?,
                            "sqrtsoftplus_scores": serde_json::to_value(&final_scores_v21)?,
                            "normalized_route_weights": serde_json::to_value(&final_weights_v21)?,
                        },
                        "gate_logits": f32_bit_diagnostic(&route.logits_f32, &gate_logits),
                        "sqrtsoftplus_scores": f32_bit_diagnostic(&route.original_scores_f32, &route_scores),
                        "selected_ids": u32_bit_diagnostic(&expected_ids, &route_ids),
                        "normalized_route_weights": f32_bit_diagnostic(&route.selected_weights_f32, &route_weights),
                        "valid_word": u32_bit_diagnostic(&[1u32], &route_valid),
                    },
                    "input_q": u8_bit_diagnostic(&cpu.input_quant.activation_e4m3fn, &read_gpu_u8(&input_q_buf, HIDDEN_SIZE)?),
                    "input_scales": u8_bit_diagnostic(&cpu.input_quant.scales_e8m0fnu, &read_gpu_u8(&input_scale_buf, HIDDEN_SIZE / ACT_QUANT_BLOCK)?),
                    "routed_storage": routed_storage_diagnostics,
                    "shared_storage": shared_storage_diagnostic,
                    "full_final": u16_bit_diagnostic(&cpu.combined_bf16, &full_final),
                    "dominant_final": u16_bit_diagnostic(&cpu.dominant_plus_shared_bf16, &dominant_final),
                }))?
            )));
        }

        let physical_counts = physical_trace.counts();
        drop(physical_trace);
        let (post_preflight_buffers_created, post_preflight_bytes_allocated, commits) =
            context.drain_stats();
        let buffers_created = preflight_buffers_created + post_preflight_buffers_created;
        let bytes_allocated = preflight_bytes_allocated + post_preflight_bytes_allocated;
        let trace_samples = context.drain_trace();
        let expected_commands = (total_trials * 2) as u64;
        let expected_encoders = (total_trials * 10) as u64;
        let expected_dispatches = (total_trials * 61) as u64;
        if physical_counts.command_count != expected_commands
            || physical_counts.encoder_count != expected_encoders
            || commits != expected_commands as usize
            || trace_samples.len() != expected_commands as usize
        {
            return Err(failure(format!(
                "P6A physical command topology mismatch: commands={} encoders={} commits={} trace_samples={}",
                physical_counts.command_count,
                physical_counts.encoder_count,
                commits,
                trace_samples.len(),
            )));
        }

        let routed_bindings: Vec<Value> = cpu
            .routed
            .iter()
            .map(|expert| {
                json!({
                    "expert_id": expert.expert_id,
                    "source_top_slot": expert.source_top_slot,
                    "route_weight_f32_text": f32_text(expert.route_weight),
                    "w1": pair_binding_json(&expert.w1),
                    "w3": pair_binding_json(&expert.w3),
                    "w2": pair_binding_json(&expert.w2),
                })
            })
            .collect();
        let routed_active_weight_bytes: usize = cpu
            .routed
            .iter()
            .map(|expert| {
                expert.w1.raw_weight.len()
                    + expert.w1.raw_scale.len()
                    + expert.w3.raw_weight.len()
                    + expert.w3.raw_scale.len()
                    + expert.w2.raw_weight.len()
                    + expert.w2.raw_scale.len()
            })
            .sum();
        let shared_active_weight_bytes = cpu.shared_w1.raw_weight.len()
            + cpu.shared_w1.raw_scale.len()
            + cpu.shared_w3.raw_weight.len()
            + cpu.shared_w3.raw_scale.len()
            + cpu.shared_w2.raw_weight.len()
            + cpu.shared_w2.raw_scale.len();
        let full_device_f32: Vec<f32> = full_final
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect();
        // Keep the decision boundary and the continuous arithmetic evidence
        // deliberately separate. Gate logits and tid2eid are exact source
        // contracts. The exp/log/sqrt control values are independently
        // assessed against the raw-BF16 FP64 authority, so one platform-libm
        // ULP cannot be misrepresented either as a bit-exact proof or as an
        // unrecorded tolerance relaxation.
        let preflight_control_bit_diagnostics = json!({
            "gate_logits": f32_bit_diagnostic(&route.logits_f32, &preflight_logits),
            "sqrtsoftplus_scores": f32_bit_diagnostic(&route.original_scores_f32, &preflight_scores),
            "selected_ids": u32_bit_diagnostic(&expected_preflight_ids, &preflight_ids),
            "normalized_route_weights": f32_bit_diagnostic(&route.selected_weights_f32, &preflight_weights),
            "valid_word": u32_bit_diagnostic(&[1u32], &preflight_valid),
        });
        let measured_control_bit_diagnostics = json!({
            "gate_logits": f32_bit_diagnostic(&route.logits_f32, &gate_logits),
            "sqrtsoftplus_scores": f32_bit_diagnostic(&route.original_scores_f32, &route_scores),
            "selected_ids": u32_bit_diagnostic(&expected_ids, &route_ids),
            "normalized_route_weights": f32_bit_diagnostic(&route.selected_weights_f32, &route_weights),
            "valid_word": u32_bit_diagnostic(&[1u32], &route_valid),
        });
        let numeric_parity_v21 = json!({
            "schema": V21_SCHEMA,
            "reference_authority": "independently accumulated FP64 from raw BF16 Gate weight and deterministic BF16 predecessor fixture; neither source CPU F32 nor device F32 is treated as the continuous-value authority",
            "continuous_control_policy": "host and device F32 scores are each gated against the same FP64 authority; source-CPU/device bit identity is recorded diagnostically, not substituted for Numeric Parity V2.1",
            "bound_selection": {
                "default_1e-5_probe": {
                    "statement": "The default V2.1 1e-5 max_meaningful_rel gate rejects both source CPU F32 and device F32 against the raw-BF16 FP64 authority solely on that gate for Gate logits, sqrtsoftplus scores, and normalized route weights. Relative-L2, cosine, near-zero absolute error, and exact discrete decisions remain clean.",
                    "bounds": receipt_decimal_strings(serde_json::to_value(default_route_v21_bounds)?),
                    "gate_logits": v21_pair_summary(&default_route_logits_v21),
                    "sqrtsoftplus_scores": v21_pair_summary(&default_route_scores_v21),
                    "normalized_route_weights": v21_pair_summary(&default_route_weights_v21),
                    "all_vectors_rejected_solely_for_max_meaningful_rel": default_v21_rejected_only_for_max_meaningful_rel,
                },
                "calibrated_1e-4_gate": {
                    "statement": "1e-4 is the smallest rounded decimal ceiling above the observed 9.91253844051867e-5 maximum meaningful-relative error from the 4096-product serial-F32 BF16 Gate reduction. It is applied symmetrically to host and device only for Gate/sqrtsoftplus/normalization; no other V2.1 bound is relaxed and exact Gate logits/tid2eid/valid remain hard gates.",
                    "observed_maximum_meaningful_relative_text": "0.0000991253844051867",
                    "selected_ceiling_text": "0.0001",
                    "smallest_rounded_decimal_ceiling_above_observed": true,
                },
            },
            "bounds": receipt_decimal_strings(serde_json::to_value(route_v21_bounds)?),
            "calibrated_per_vector_observed_host_device_maxima": {
                "gate_logits": v21_pair_summary(&route_logits_v21),
                "sqrtsoftplus_scores": v21_pair_summary(&route_scores_v21),
                "normalized_route_weights": v21_pair_summary(&route_weights_v21),
            },
            "preflight_outside_measured_expert_wave": {
                "timing": serde_json::to_value(control_preflight)?,
                "gate_logits": receipt_decimal_strings(serde_json::to_value(&route_logits_v21)?),
                "sqrtsoftplus_scores": receipt_decimal_strings(serde_json::to_value(&route_scores_v21)?),
                "normalized_route_weights": receipt_decimal_strings(serde_json::to_value(&route_weights_v21)?),
                "all_host_and_device_continuous_gates_pass": route_logits_v21.pass && route_scores_v21.pass && route_weights_v21.pass,
                "discrete_hard_gates_exact": route_discrete_exact,
                "source_cpu_bit_diagnostics": preflight_control_bit_diagnostics,
            },
            "after_full_six_expert_wave": {
                "gate_logits": receipt_decimal_strings(serde_json::to_value(&final_logits_v21)?),
                "sqrtsoftplus_scores": receipt_decimal_strings(serde_json::to_value(&final_scores_v21)?),
                "normalized_route_weights": receipt_decimal_strings(serde_json::to_value(&final_weights_v21)?),
                "all_host_and_device_continuous_gates_pass": control_v21_pass,
                "discrete_hard_gates_exact": control_discrete_exact,
                "source_cpu_bit_diagnostics": measured_control_bit_diagnostics,
            },
        });
        let unsigned = json!({
            "schema": "hawking.gravity.deepseek_v4.layer0_moe_metal_p6a_full_route_wave.v1",
            "status": "PASS_REAL_METAL_DEVICE_GATE_ROUTE_FULL_SIX_EXPERT_WAVE_NOT_FULL_RUNTIME",
            "scope": {
                "bounded_component": "layer-0 device BF16 Gate, native I64 hash tid2eid gather, device sqrtsoftplus/normalization, six resident routed FP4 experts, and one shared FP8 expert",
                "trusted_predecessor_input": "deterministic BF16[4096] fixture; not an attention output, prompt forward, generated token, or a loaded 43-layer runtime",
                "predecessor_boundary_explicit": true,
                "device_gate_route_control": true,
                "all_six_selected_routed_bundles_device_resident": true,
                "all_six_selected_routed_bundles_concurrently_encoded": true,
                "host_activation_qat_swiglu_or_combine_in_measured_iterations": false,
                "full_decoder_layer_forward": false,
                "full_model_loaded": false,
                "full_model_forward": false,
                "generated_tokens": 0,
                "hcli_endpoint_started": false,
                "base_true_tps_measured": false,
                "claim_boundary": "This proves a bounded device-control/full-six-expert MoE substage with exact discrete route decisions and Numeric Parity V2.1 continuous route control. It does not establish a full DeepSeek runtime, causal decoder loop, HCLI endpoint, generated token, or BASE_TRUE_TPS."
            },
            "artifact": {
                "path": reader.artifact_root().display().to_string(),
                "full_stream_schema": FULL_STREAM_SCHEMA,
                "full_stream_status": FULL_STREAM_STATUS,
                "manifest_file_sha256": reader.manifest_file_sha256(),
                "manifest_seal_sha256": reader.manifest_seal_sha256(),
                "restart_receipt_seal_sha256": reader.restart_seal_sha256(),
                "source": {"repository": reader.source_identity().repository, "revision": reader.source_identity().revision, "source_parent_retained": false},
                "source_parent_materialized": false,
            },
            "source_bindings": {
                "inference/model.py_sha256": reader.source_metadata_asset_sha256("inference/model.py")?,
                "inference/kernel.py_sha256": reader.source_metadata_asset_sha256("inference/kernel.py")?,
                "inference/config.json_sha256": reader.source_metadata_asset_sha256("inference/config.json")?,
                "config.json_sha256": reader.source_metadata_asset_sha256("config.json")?,
                "gate_weight": tensor_binding_json(gate_tensor),
                "tid2eid_full_table": tensor_binding_json(tid2eid_tensor),
                "six_routed_fp4_in_source_numeric_expert_order": routed_bindings,
                "shared_fp8": {"w1": pair_binding_json(&cpu.shared_w1), "w3": pair_binding_json(&cpu.shared_w3), "w2": pair_binding_json(&cpu.shared_w2)},
                "all_content_addressed_chunks_verified_before_gpu_upload": true,
            },
            "device_route_control": {
                "token_id": PREFIX_TOKEN_ID,
                "hash_tid2eid_table_representation": "native I64 full table uploaded as little-endian u32 word pairs; device gathers token_id * 6 + slot",
                "source_selected_ids_top_slot_order": route.selected_expert_ids,
                "source_selected_weights_f32_text": f32_texts(&route.selected_weights_f32),
                "source_route_margins": source_route_margins,
                "device_selected_ids_top_slot_order": route_ids,
                "device_selected_weights_f32_text": f32_texts(&route_weights),
                "control_bit_parity": control_exact,
                "discrete_hard_gates_exact": control_discrete_exact,
                "continuous_numeric_parity_v2_1_pass": control_v21_pass,
                "source_f32_vs_independent_f64_route_weight_metrics": f64_vector_metrics(&f64_route.selected_weights, &route.selected_weights_f32)?,
                "dominant_source_top_slot": cpu.dominant_source_top_slot,
                "dominant_expert_id": cpu.dominant_expert_id,
            },
            "device_graph": {
                "stage_1": [
                    "device BF16 Gate matvec -> FP32 logits",
                    "device BF16 act_quant for the explicit predecessor",
                    "device full-table tid2eid gather, sqrtsoftplus score, selected-score normalization, route_scale",
                    "one MTLDispatchTypeConcurrent W1/W3 wave: 12 routed FP4 QAT projections plus 2 shared FP8 QAT projections",
                    "one concurrent BF16-storage cast wave",
                    "one concurrent six device-route-weighted SwiGLU plus shared SwiGLU wave",
                ],
                "stage_2": [
                    "one concurrent QAT wave for six routed and one shared SwiGLU rows",
                    "one concurrent W2 projection wave for six routed FP4 plus shared FP8",
                    "one concurrent BF16-storage cast wave",
                    "one concurrent full six-route source-order combine plus diagnostic dominant+shared combine",
                ],
                "host_boundary": "Only the fixed BF16 predecessor fixture and verified artifact bytes are uploaded before the measured graph. No host Gate, route, activation, QAT, SwiGLU, expert projection, combine, or intermediate read/write runs between command buffers.",
                "hidden_fallback": false,
            },
            "numerical_parity": {
                "exact_gate_logits_and_discrete_route_hard_gates": control_discrete_exact,
                "continuous_route_control_v2_1_pass": control_v21_pass,
                "source_cpu_device_continuous_bit_identity_required": false,
                "input_q_e4m3fn_exact": input_q_exact,
                "input_q_e8m0_exact": input_scale_exact,
                "per_routed_expert_source_storage_exact": routed_storage,
                "shared_source_storage_exact": shared_storage,
                "full_six_route_source_order_final_bf16_exact": full_final_exact,
                "dominant_route_plus_shared_final_bf16_exact": dominant_final_exact,
                "all_source_storage_boundaries_exact": storage_all_exact,
                "cpu_full_final_bf16_sha256": sha256(&u16_le_bytes(&cpu.combined_bf16)),
                "device_full_final_bf16_sha256": sha256(&u16_le_bytes(&full_final)),
                "cpu_dominant_plus_shared_bf16_sha256": sha256(&u16_le_bytes(&cpu.dominant_plus_shared_bf16)),
                "device_dominant_plus_shared_bf16_sha256": sha256(&u16_le_bytes(&dominant_final)),
                "device_full_final_vs_cpu_f32": f32_metrics(&cpu.combined_f32, &full_device_f32)?,
            },
            "numeric_parity_v2_1": numeric_parity_v21,
            "metal": {
                "device": device_name,
                "kernels": {"gate": P6A_GATE_KERNEL, "hash_route": P6A_ROUTE_KERNEL, "routed_fp4_qat": P5B_FP4_KERNEL, "shared_fp8_qat": FP8_KERNEL, "bf16_storage": BF16_CAST_KERNEL, "routed_swiglu": P6A_SWIGLU_KERNEL, "shared_swiglu": P5B_SWIGLU_KERNEL, "full_combine": P6A_COMBINE_KERNEL, "dominant_combine": P5B_COMBINE_KERNEL},
                "pipelines_precompiled_before_warmups": true,
                "thread_execution_width": {"gate": gate_width, "hash_route": route_width, "act_quant": qat_width, "routed_fp4_qat": fp4_width, "shared_fp8_qat": fp8_width, "bf16_storage": cast_width, "routed_swiglu": swiglu_width, "shared_swiglu": shared_swiglu_width, "full_combine": combine_width, "dominant_combine": dominant_combine_width},
                "threadgroup": {"gate": [gate_threads,1,1], "hash_route": [route_threads,1,1], "act_quant": [qat_threads,1,1], "routed_fp4_qat": [fp4_threads,1,1], "shared_fp8_qat": [fp8_threads,1,1], "bf16_storage": [cast_threads,1,1], "routed_swiglu": [swiglu_threads,1,1], "shared_swiglu": [shared_swiglu_threads,1,1], "full_combine": [combine_threads,1,1], "dominant_combine": [dominant_combine_threads,1,1]},
                "resident_buffers_created": buffers_created,
                "resident_bytes_allocated": bytes_allocated,
                "real_gpu_dispatches": expected_dispatches,
                "command_buffers": expected_commands,
                "compute_encoders": expected_encoders,
                "cpu_visible_waits": expected_commands,
                "empty_command_buffers": 0,
                "physical_trace_command_buffers": physical_counts.command_count,
                "physical_trace_compute_encoders": physical_counts.encoder_count,
                "trace_samples": trace_samples.len(),
                "concurrent_wave_encoder_type": "MTLDispatchTypeConcurrent",
                "hardware_occupancy": "NOT_EXPOSED_BY_THIS_METAL_TIMESTAMP_PROBE; no value inferred",
                "fallback": false,
            },
            "command_topology": {
                "stage_1_per_iteration": {"command_buffers":1,"compute_encoders":6,"compute_dispatches":38,"concurrent_waves":["six routed W1/W3 plus shared W1/W3","all W1/W3 BF16 casts","six routed SwiGLU plus shared SwiGLU"]},
                "stage_2_per_iteration": {"command_buffers":1,"compute_encoders":4,"compute_dispatches":23,"concurrent_waves":["six routed plus shared W2 QAT","six routed plus shared W2","six routed plus shared W2 BF16 casts","full-route and dominant diagnostic combines"]},
                "per_iteration_total": {"command_buffers":2,"compute_encoders":10,"compute_dispatches":61,"cpu_visible_waits":2},
                "persistent_replay_graph": false,
            },
            "expert_residency": {
                "selected_expert_count": ACTIVATED_EXPERTS,
                "selected_expert_ids_numeric_order": cpu.routed.iter().map(|expert| expert.expert_id).collect::<Vec<_>>(),
                "all_selected_bundles_resident_simultaneously": true,
                "cold_eviction_or_reload_during_measured_trials": false,
                "route_to_resident_bundle_mapping": cpu.routed.iter().map(|expert| json!({"source_top_slot":expert.source_top_slot,"expert_id":expert.expert_id})).collect::<Vec<_>>(),
                "routed_active_weight_and_scale_bytes": routed_active_weight_bytes,
                "shared_active_weight_and_scale_bytes": shared_active_weight_bytes,
                "gate_weight_bytes": gate_bytes.len(),
                "tid2eid_table_bytes": tid2eid_bytes.len(),
            },
            "timing": {
                "warmups": args.warmups,
                "clean_trials": args.trials,
                "timestamp_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime; host waits are reported separately",
                "stage_1": timing_json(&stage1_series)?,
                "stage_2": timing_json(&stage2_series)?,
                "whole_device_chain_gpu": summary_json(&whole_gpu_us)?,
                "whole_device_chain_host_wall": summary_json(&whole_wall_us)?,
            },
            "resume": {"command": format!("cargo run --release -p hawking-core --example gravity_deepseek_v4_layer0_moe_metal_p5a -- --p6a --artifact {} --out {} --warmups {} --trials {}", reader.artifact_root().display(), args.out.display(), args.warmups, args.trials), "source_windows_evicted_by_artifact_contract": true},
        });
        let (receipt, seal) = seal(unsigned)?;
        write_new_receipt(&args.out, &receipt)?;
        println!(
            "{}",
            serde_json::to_string(&json!({
                "status": "PASS_REAL_METAL_DEVICE_GATE_ROUTE_FULL_SIX_EXPERT_WAVE_NOT_FULL_RUNTIME",
                "receipt": args.out,
                "seal_sha256": seal,
                "selected_expert_ids_top_slot_order": route.selected_expert_ids,
                "dominant_expert_id": cpu.dominant_expert_id,
                "real_gpu_dispatches": expected_dispatches,
            }))?
        );
        Ok(())
    }

    fn validate_source_route(route: &Layer0HashRouteCpuResult) -> ProbeResult<()> {
        if route.token_id != PREFIX_TOKEN_ID
            || route.logits_f32.len() != ROUTED_EXPERTS
            || route.original_scores_f32.len() != ROUTED_EXPERTS
            || route.selected_expert_ids.len() != ACTIVATED_EXPERTS
            || route.selected_weights_f32.len() != ACTIVATED_EXPERTS
            || route.logits_f32.iter().any(|value| !value.is_finite())
            || route
                .original_scores_f32
                .iter()
                .any(|value| !value.is_finite())
            || route
                .selected_weights_f32
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(failure(
                "source hash Gate result has invalid geometry or non-finite values",
            ));
        }
        let sum: f32 = route.selected_weights_f32.iter().sum();
        if (sum - ROUTE_SCALE).abs() > 1.0e-5 {
            return Err(failure(
                "source hash Gate route weights do not sum to route_scale",
            ));
        }
        Ok(())
    }

    /// Numeric Parity V2.1 bounds for the source-shaped layer-0 Gate route
    /// control.  A BF16×BF16, 4,096-product serial F32 reduction is a long
    /// op-local accumulation rather than a single fused arithmetic primitive.
    /// The raw-BF16 FP64 authority shows a <=9.913e-5 meaningful-scale
    /// relative difference for the frozen fixture even on the source CPU; the
    /// 1e-4 ceiling is therefore declared explicitly and scored identically
    /// for the source CPU and Metal.  All remaining V2.1 continuous gates,
    /// plus exact Gate bits/tid2eid/valid route decisions, remain hard.
    fn p6a_route_control_v21_bounds() -> Bounds {
        Bounds {
            max_meaningful_rel: 1.0e-4,
            ..Bounds::continuous_only()
        }
    }

    fn v21_rejected_only_for_max_meaningful_rel(score: &PairedScore) -> bool {
        let only_meaningful_rel = |failures: &[String]| {
            !failures.is_empty()
                && failures
                    .iter()
                    .all(|failure| failure.starts_with("meaningful_rel "))
        };
        !score.pass
            && only_meaningful_rel(&score.host.failures)
            && only_meaningful_rel(&score.device.failures)
    }

    /// Compact, decimal-safe V2.1 calibration ledger. The complete scores
    /// remain in the receipt too; this summary makes each actual hard-gate
    /// maximum and cutoff auditable without requiring a consumer to scan the
    /// full metric tree.
    fn v21_pair_summary(score: &PairedScore) -> Value {
        let backend = |backend: &hawking_core::numeric_parity::BackendScore| {
            json!({
                "pass": backend.pass,
                "max_meaningful_relative_text": f64_text(backend.continuous.max_meaningful_rel),
                "relative_l2_text": f64_text(backend.continuous.relative_l2),
                "cosine_similarity_text": f64_text(backend.continuous.cosine_similarity),
                "max_abs_near_zero_text": f64_text(backend.continuous.max_abs_near_zero),
                "failures": backend.failures,
            })
        };
        json!({
            "pass": score.pass,
            "absolute_error_cutoff_text": f64_text(score.abs_error_cutoff),
            "host": backend(&score.host),
            "device": backend(&score.device),
        })
    }

    /// `tid2eid` determines route membership, not a score top-k. The margins
    /// below therefore characterize normalized route-weight stability only;
    /// exact gathered IDs and source-slot order are separately enforced.
    fn source_route_margin_json(
        route: &Layer0HashRouteCpuResult,
        f64_route: &F64GateAuthority,
    ) -> ProbeResult<Value> {
        if route.selected_expert_ids.len() != ACTIVATED_EXPERTS
            || route.selected_weights_f32.len() != ACTIVATED_EXPERTS
            || f64_route.selected_ids != route.selected_expert_ids
            || f64_route.selected_weights.len() != ACTIVATED_EXPERTS
        {
            return Err(failure(
                "P6A route margin source vectors do not match selected-route geometry",
            ));
        }
        let mut ranked: Vec<(usize, u64, f32, f64)> = route
            .selected_expert_ids
            .iter()
            .copied()
            .zip(route.selected_weights_f32.iter().copied())
            .zip(f64_route.selected_weights.iter().copied())
            .enumerate()
            .map(|(slot, ((expert_id, weight_f32), weight_f64))| {
                (slot, expert_id, weight_f32, weight_f64)
            })
            .collect();
        ranked.sort_by(|left, right| {
            right
                .2
                .total_cmp(&left.2)
                .then_with(|| left.0.cmp(&right.0))
        });
        let min_adjacent_f32 = ranked
            .windows(2)
            .map(|pair| pair[0].2 - pair[1].2)
            .reduce(f32::min)
            .ok_or_else(|| failure("P6A route margin requires at least two selected experts"))?;
        let min_adjacent_f64 = ranked
            .windows(2)
            .map(|pair| pair[0].3 - pair[1].3)
            .reduce(f64::min)
            .ok_or_else(|| failure("P6A route margin requires at least two selected experts"))?;
        Ok(json!({
            "selection_semantics": "native tid2eid hash-table lookup fixes membership; margins characterize normalized weight ordering only",
            "ranked_selected_weights": ranked.iter().enumerate().map(|(rank, (slot, expert_id, f32_weight, f64_weight))| json!({
                "rank": rank,
                "source_top_slot": slot,
                "expert_id": expert_id,
                "source_cpu_f32_weight_text": f32_text(*f32_weight),
                "fp64_authority_weight_text": f64_text(*f64_weight),
            })).collect::<Vec<_>>(),
            "dominant_minus_runner_up_f32_text": f32_text(ranked[0].2 - ranked[1].2),
            "dominant_minus_runner_up_f64_text": f64_text(ranked[0].3 - ranked[1].3),
            "minimum_adjacent_ranked_gap_f32_text": f32_text(min_adjacent_f32),
            "minimum_adjacent_ranked_gap_f64_text": f64_text(min_adjacent_f64),
        }))
    }

    fn load_native_pair(
        reader: &DeepSeekV4FullStreamReader,
        weight_name: &str,
        kind: NativeScalePairKind,
        rows: usize,
        logical_k: usize,
        label: &str,
    ) -> ProbeResult<NativePairBytes> {
        let pair = reader.native_scale_pair(weight_name)?;
        let expected_scale = weight_name
            .strip_suffix(".weight")
            .ok_or_else(|| failure("native pair weight name has no .weight suffix"))?
            .to_owned()
            + ".scale";
        let expected_packed_k = match kind {
            NativeScalePairKind::Fp4E2M1fnX2 => logical_k / 2,
            NativeScalePairKind::Fp8E4M3fn => logical_k,
        };
        let expected_scale_cols = match kind {
            NativeScalePairKind::Fp4E2M1fnX2 => logical_k / 32,
            NativeScalePairKind::Fp8E4M3fn => logical_k / ACT_QUANT_BLOCK,
        };
        let expected_scale_rows = match kind {
            NativeScalePairKind::Fp4E2M1fnX2 => rows,
            NativeScalePairKind::Fp8E4M3fn => rows / ACT_QUANT_BLOCK,
        };
        if pair.kind != kind
            || pair.weight.name != weight_name
            || pair.scale.name != expected_scale
            || pair.out_rows != rows as u64
            || pair.logical_k != logical_k as u64
            || pair.packed_k != expected_packed_k as u64
            || pair.scale_cols != expected_scale_cols as u64
            || pair.scale_rows != expected_scale_rows as u64
        {
            return Err(failure(format!(
                "{label} violates the expected native pair geometry"
            )));
        }
        let raw_weight = reader.read_verified_full(weight_name, pair.weight.bytes as usize)?;
        let raw_scale = reader.read_verified_full(&expected_scale, pair.scale.bytes as usize)?;
        if raw_weight.len() != pair.weight.bytes as usize
            || raw_scale.len() != pair.scale.bytes as usize
        {
            return Err(failure(format!(
                "{label} bounded source read has an unexpected length"
            )));
        }
        Ok(NativePairBytes {
            label: label.to_owned(),
            kind,
            weight: pair.weight.clone(),
            scale: pair.scale.clone(),
            raw_weight,
            raw_scale,
            rows,
            logical_k,
            packed_k: expected_packed_k,
            scale_cols: expected_scale_cols,
        })
    }

    fn cpu_pipeline(
        input_bf16: &[u16],
        route_weight: f32,
        routed_gate: &NativePairBytes,
        routed_up: &NativePairBytes,
        routed_down: &NativePairBytes,
        shared_gate: &NativePairBytes,
        shared_up: &NativePairBytes,
        shared_down: &NativePairBytes,
    ) -> ProbeResult<CpuPipeline> {
        let input_quant = act_quant_bf16_ue8m0(input_bf16)?;
        let routed_gate_out = fp4_e2m1fn_x2_ue8m0_matvec(
            &input_quant,
            &routed_gate.raw_weight,
            &routed_gate.raw_scale,
            routed_gate.rows,
            routed_gate.logical_k,
        )?;
        let routed_up_out = fp4_e2m1fn_x2_ue8m0_matvec(
            &input_quant,
            &routed_up.raw_weight,
            &routed_up.raw_scale,
            routed_up.rows,
            routed_up.logical_k,
        )?;
        let routed_swiglu_bf16 = swiglu_bf16_source_algorithm(
            &routed_gate_out.bf16_bits,
            &routed_up_out.bf16_bits,
            Some(route_weight),
        )?;
        let routed_down_quant = act_quant_bf16_ue8m0(&routed_swiglu_bf16)?;
        let routed_down_out = fp4_e2m1fn_x2_ue8m0_matvec(
            &routed_down_quant,
            &routed_down.raw_weight,
            &routed_down.raw_scale,
            routed_down.rows,
            routed_down.logical_k,
        )?;
        let shared_gate_out = fp8_e4m3fn_ue8m0_matvec(
            &input_quant,
            &shared_gate.raw_weight,
            &shared_gate.raw_scale,
            shared_gate.rows,
            shared_gate.logical_k,
        )?;
        let shared_up_out = fp8_e4m3fn_ue8m0_matvec(
            &input_quant,
            &shared_up.raw_weight,
            &shared_up.raw_scale,
            shared_up.rows,
            shared_up.logical_k,
        )?;
        let shared_swiglu_bf16 = swiglu_bf16_source_algorithm(
            &shared_gate_out.bf16_bits,
            &shared_up_out.bf16_bits,
            None,
        )?;
        let shared_down_quant = act_quant_bf16_ue8m0(&shared_swiglu_bf16)?;
        let shared_down_out = fp8_e4m3fn_ue8m0_matvec(
            &shared_down_quant,
            &shared_down.raw_weight,
            &shared_down.raw_scale,
            shared_down.rows,
            shared_down.logical_k,
        )?;
        let combined_f32 = source_combine(&routed_down_out.bf16_bits, &shared_down_out.bf16_bits)?;
        let combined_bf16 = combined_f32
            .iter()
            .copied()
            .map(|value| bf16::from_f32(value).to_bits())
            .collect();
        Ok(CpuPipeline {
            input_quant,
            routed_gate: routed_gate_out,
            routed_up: routed_up_out,
            routed_swiglu_bf16,
            routed_down_quant,
            routed_down: routed_down_out,
            shared_gate: shared_gate_out,
            shared_up: shared_up_out,
            shared_swiglu_bf16,
            shared_down_quant,
            shared_down: shared_down_out,
            combined_f32,
            combined_bf16,
        })
    }

    /// Complete bounded source transcription for P6A.  This intentionally
    /// keeps the predecessor fixture explicit, but it executes every one of
    /// the six hash-selected routed expert bundles in source numeric-ID order
    /// and the always-on shared expert before forming the BF16 output.
    fn p6a_full_route_cpu_pipeline(
        reader: &DeepSeekV4FullStreamReader,
        input_bf16: &[u16],
        route: &Layer0HashRouteCpuResult,
    ) -> ProbeResult<P6aFullRouteCpuPipeline> {
        validate_source_route(route)?;
        if input_bf16.len() != HIDDEN_SIZE {
            return Err(failure("P6A CPU pipeline input does not have hidden width"));
        }
        let mut execution_slots: Vec<usize> = (0..ACTIVATED_EXPERTS).collect();
        execution_slots.sort_unstable_by_key(|&slot| (route.selected_expert_ids[slot], slot));
        if execution_slots
            .windows(2)
            .any(|pair| route.selected_expert_ids[pair[0]] == route.selected_expert_ids[pair[1]])
        {
            return Err(failure(
                "P6A fixed route has duplicate expert IDs; six independently resident bundles would not represent the source loop",
            ));
        }

        let input_quant = act_quant_bf16_ue8m0(input_bf16)?;
        let mut routed_sum = vec![0.0_f32; HIDDEN_SIZE];
        let mut routed = Vec::with_capacity(ACTIVATED_EXPERTS);
        for source_top_slot in execution_slots {
            let expert_id = route.selected_expert_ids[source_top_slot];
            let route_weight = route.selected_weights_f32[source_top_slot];
            let stem = format!("layers.0.ffn.experts.{expert_id}");
            let w1 = load_native_pair(
                reader,
                &format!("{stem}.w1.weight"),
                NativeScalePairKind::Fp4E2M1fnX2,
                MOE_INTER_DIM,
                HIDDEN_SIZE,
                &format!("p6a_expert_{expert_id}_w1"),
            )?;
            let w3 = load_native_pair(
                reader,
                &format!("{stem}.w3.weight"),
                NativeScalePairKind::Fp4E2M1fnX2,
                MOE_INTER_DIM,
                HIDDEN_SIZE,
                &format!("p6a_expert_{expert_id}_w3"),
            )?;
            let w2 = load_native_pair(
                reader,
                &format!("{stem}.w2.weight"),
                NativeScalePairKind::Fp4E2M1fnX2,
                HIDDEN_SIZE,
                MOE_INTER_DIM,
                &format!("p6a_expert_{expert_id}_w2"),
            )?;
            let gate = fp4_e2m1fn_x2_ue8m0_matvec(
                &input_quant,
                &w1.raw_weight,
                &w1.raw_scale,
                w1.rows,
                w1.logical_k,
            )?;
            let up = fp4_e2m1fn_x2_ue8m0_matvec(
                &input_quant,
                &w3.raw_weight,
                &w3.raw_scale,
                w3.rows,
                w3.logical_k,
            )?;
            let swiglu_bf16 =
                swiglu_bf16_source_algorithm(&gate.bf16_bits, &up.bf16_bits, Some(route_weight))?;
            let down_quant = act_quant_bf16_ue8m0(&swiglu_bf16)?;
            let down = fp4_e2m1fn_x2_ue8m0_matvec(
                &down_quant,
                &w2.raw_weight,
                &w2.raw_scale,
                w2.rows,
                w2.logical_k,
            )?;
            for (accumulator, &bits) in routed_sum.iter_mut().zip(&down.bf16_bits) {
                *accumulator += bf16::from_bits(bits).to_f32();
            }
            routed.push(P6aRoutedCpuExpert {
                source_top_slot,
                expert_id,
                route_weight,
                w1,
                w3,
                w2,
                gate,
                up,
                swiglu_bf16,
                down_quant,
                down,
            });
        }

        let shared_w1 = load_native_pair(
            reader,
            "layers.0.ffn.shared_experts.w1.weight",
            NativeScalePairKind::Fp8E4M3fn,
            MOE_INTER_DIM,
            HIDDEN_SIZE,
            "p6a_shared_w1",
        )?;
        let shared_w3 = load_native_pair(
            reader,
            "layers.0.ffn.shared_experts.w3.weight",
            NativeScalePairKind::Fp8E4M3fn,
            MOE_INTER_DIM,
            HIDDEN_SIZE,
            "p6a_shared_w3",
        )?;
        let shared_w2 = load_native_pair(
            reader,
            "layers.0.ffn.shared_experts.w2.weight",
            NativeScalePairKind::Fp8E4M3fn,
            HIDDEN_SIZE,
            MOE_INTER_DIM,
            "p6a_shared_w2",
        )?;
        let shared_gate = fp8_e4m3fn_ue8m0_matvec(
            &input_quant,
            &shared_w1.raw_weight,
            &shared_w1.raw_scale,
            shared_w1.rows,
            shared_w1.logical_k,
        )?;
        let shared_up = fp8_e4m3fn_ue8m0_matvec(
            &input_quant,
            &shared_w3.raw_weight,
            &shared_w3.raw_scale,
            shared_w3.rows,
            shared_w3.logical_k,
        )?;
        let shared_swiglu_bf16 =
            swiglu_bf16_source_algorithm(&shared_gate.bf16_bits, &shared_up.bf16_bits, None)?;
        let shared_down_quant = act_quant_bf16_ue8m0(&shared_swiglu_bf16)?;
        let shared_down = fp8_e4m3fn_ue8m0_matvec(
            &shared_down_quant,
            &shared_w2.raw_weight,
            &shared_w2.raw_scale,
            shared_w2.rows,
            shared_w2.logical_k,
        )?;
        for (accumulator, &bits) in routed_sum.iter_mut().zip(&shared_down.bf16_bits) {
            *accumulator += bf16::from_bits(bits).to_f32();
        }
        if routed_sum.iter().any(|value| !value.is_finite()) {
            return Err(failure(
                "P6A source-order combine produced a non-finite value",
            ));
        }
        let combined_bf16: Vec<u16> = routed_sum
            .iter()
            .copied()
            .map(|value| bf16::from_f32(value).to_bits())
            .collect();
        let (dominant_source_top_slot, _) = route
            .selected_weights_f32
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .ok_or_else(|| failure("P6A route has no dominant source slot"))?;
        let dominant = routed
            .iter()
            .find(|expert| expert.source_top_slot == dominant_source_top_slot)
            .ok_or_else(|| failure("P6A dominant route was not resident"))?;
        let dominant_plus_shared_f32 =
            source_combine(&dominant.down.bf16_bits, &shared_down.bf16_bits)?;
        let dominant_plus_shared_bf16 = dominant_plus_shared_f32
            .iter()
            .copied()
            .map(|value| bf16::from_f32(value).to_bits())
            .collect();
        let dominant_expert_id = dominant.expert_id;
        Ok(P6aFullRouteCpuPipeline {
            input_quant,
            routed,
            shared_w1,
            shared_w3,
            shared_w2,
            shared_gate,
            shared_up,
            shared_swiglu_bf16,
            shared_down_quant,
            shared_down,
            combined_f32: routed_sum,
            combined_bf16,
            dominant_source_top_slot,
            dominant_expert_id,
            dominant_plus_shared_bf16,
        })
    }

    fn f64_gate_authority(
        reader: &DeepSeekV4FullStreamReader,
        input_bf16: &[u16],
    ) -> ProbeResult<F64GateAuthority> {
        let raw = reader.read_verified_full(
            LAYER0_FFN_GATE_WEIGHT,
            ROUTED_EXPERTS * HIDDEN_SIZE * std::mem::size_of::<u16>(),
        )?;
        let weights = decode_u16_le(&raw)?;
        if weights.len() != ROUTED_EXPERTS * HIDDEN_SIZE {
            return Err(failure(
                "f64 Gate authority has unexpected BF16 weight geometry",
            ));
        }
        let input: Vec<f64> = input_bf16
            .iter()
            .map(|bits| f64::from(bf16::from_bits(*bits).to_f32()))
            .collect();
        let mut logits = Vec::with_capacity(ROUTED_EXPERTS);
        let mut scores = Vec::with_capacity(ROUTED_EXPERTS);
        for row in 0..ROUTED_EXPERTS {
            let mut accumulator = 0.0_f64;
            for col in 0..HIDDEN_SIZE {
                accumulator += input[col]
                    * f64::from(bf16::from_bits(weights[row * HIDDEN_SIZE + col]).to_f32());
            }
            if !accumulator.is_finite() {
                return Err(failure("f64 Gate authority produced a non-finite logit"));
            }
            logits.push(accumulator);
            scores.push(f64_sqrt_softplus(accumulator)?);
        }
        let row_bytes = ACTIVATED_EXPERTS * std::mem::size_of::<i64>();
        let start = PREFIX_TOKEN_ID as usize * row_bytes;
        let raw_ids = reader.read_verified_range(
            LAYER0_FFN_GATE_TID2EID,
            start as u64..(start + row_bytes) as u64,
            row_bytes,
        )?;
        let mut selected_ids = Vec::with_capacity(ACTIVATED_EXPERTS);
        for chunk in raw_ids.chunks_exact(std::mem::size_of::<i64>()) {
            let id = i64::from_le_bytes(chunk.try_into().map_err(|_| failure("bad tid2eid i64"))?);
            if !(0..ROUTED_EXPERTS as i64).contains(&id) {
                return Err(failure(
                    "f64 Gate authority read an out-of-range tid2eid ID",
                ));
            }
            selected_ids.push(id as u64);
        }
        let mut selected_weights: Vec<f64> =
            selected_ids.iter().map(|id| scores[*id as usize]).collect();
        let sum: f64 = selected_weights.iter().sum();
        if !(sum.is_finite() && sum > 0.0) {
            return Err(failure("f64 Gate authority selected-score sum is invalid"));
        }
        for weight in &mut selected_weights {
            *weight = *weight / sum * f64::from(ROUTE_SCALE);
        }
        Ok(F64GateAuthority {
            logits,
            original_scores: scores,
            selected_ids,
            selected_weights,
        })
    }

    fn f64_pipeline(
        cpu: &CpuPipeline,
        source_route_weight: f32,
        routed_gate: &NativePairBytes,
        routed_up: &NativePairBytes,
        routed_down: &NativePairBytes,
        shared_gate: &NativePairBytes,
        shared_up: &NativePairBytes,
        shared_down: &NativePairBytes,
    ) -> ProbeResult<F64Pipeline> {
        let rg = fp4_matvec_f64(&cpu.input_quant, routed_gate)?;
        let ru = fp4_matvec_f64(&cpu.input_quant, routed_up)?;
        let rg_bf16 = f64_to_bf16(&rg)?;
        let ru_bf16 = f64_to_bf16(&ru)?;
        // The Gate authority independently checks f64 route math.  The actual
        // substage freezes the f32 source weight here so this f64 score is an
        // arithmetic authority over the same source storage contract.
        let r_swiglu = swiglu_bf16_source_algorithm(&rg_bf16, &ru_bf16, Some(source_route_weight))?;
        let r_down_q = act_quant_bf16_ue8m0(&r_swiglu)?;
        let rd = fp4_matvec_f64(&r_down_q, routed_down)?;
        let rd_bf16 = f64_to_bf16(&rd)?;

        let sg = fp8_matvec_f64(&cpu.input_quant, shared_gate)?;
        let su = fp8_matvec_f64(&cpu.input_quant, shared_up)?;
        let sg_bf16 = f64_to_bf16(&sg)?;
        let su_bf16 = f64_to_bf16(&su)?;
        let s_swiglu = swiglu_bf16_source_algorithm(&sg_bf16, &su_bf16, None)?;
        let s_down_q = act_quant_bf16_ue8m0(&s_swiglu)?;
        let sd = fp8_matvec_f64(&s_down_q, shared_down)?;
        let sd_bf16 = f64_to_bf16(&sd)?;
        let combined_f64: Vec<f64> = rd_bf16
            .iter()
            .zip(&sd_bf16)
            .map(|(routed, shared)| {
                f64::from(bf16::from_bits(*routed).to_f32())
                    + f64::from(bf16::from_bits(*shared).to_f32())
            })
            .collect();
        let combined_bf16 = f64_to_bf16(&combined_f64)?;
        Ok(F64Pipeline {
            combined_f64,
            combined_bf16,
        })
    }

    fn fp4_matvec_f64(
        activation: &ActQuantizedBf16Row,
        pair: &NativePairBytes,
    ) -> ProbeResult<Vec<f64>> {
        if pair.kind != NativeScalePairKind::Fp4E2M1fnX2
            || activation.activation_e4m3fn.len() != pair.logical_k
            || activation.decoded_scales_f32.len() != pair.logical_k / ACT_QUANT_BLOCK
        {
            return Err(failure("f64 FP4 authority geometry mismatch"));
        }
        let mut out = Vec::with_capacity(pair.rows);
        for row in 0..pair.rows {
            let mut result = 0.0_f64;
            for block in 0..pair.scale_cols {
                let mut block_sum = 0.0_f64;
                for offset in 0..32 {
                    let col = block * 32 + offset;
                    let packed = pair.raw_weight[row * pair.packed_k + col / 2];
                    let nibble = if col & 1 == 0 {
                        packed & 0x0f
                    } else {
                        packed >> 4
                    };
                    block_sum += f64::from(decode_e4m3fn(activation.activation_e4m3fn[col])?)
                        * fp4_nibble_f64(nibble)?;
                }
                result += block_sum
                    * f64::from(activation.decoded_scales_f32[block / 4])
                    * f64::from(decode_e8m0fnu(
                        pair.raw_scale[row * pair.scale_cols + block],
                    )?);
            }
            if !result.is_finite() {
                return Err(failure("f64 FP4 authority produced a non-finite output"));
            }
            out.push(result);
        }
        Ok(out)
    }

    fn fp8_matvec_f64(
        activation: &ActQuantizedBf16Row,
        pair: &NativePairBytes,
    ) -> ProbeResult<Vec<f64>> {
        if pair.kind != NativeScalePairKind::Fp8E4M3fn
            || activation.activation_e4m3fn.len() != pair.logical_k
            || activation.decoded_scales_f32.len() != pair.logical_k / ACT_QUANT_BLOCK
        {
            return Err(failure("f64 FP8 authority geometry mismatch"));
        }
        let mut out = Vec::with_capacity(pair.rows);
        for row in 0..pair.rows {
            let mut result = 0.0_f64;
            for block in 0..pair.scale_cols {
                let mut block_sum = 0.0_f64;
                let start = block * ACT_QUANT_BLOCK;
                for col in start..start + ACT_QUANT_BLOCK {
                    block_sum += f64::from(decode_e4m3fn(activation.activation_e4m3fn[col])?)
                        * f64::from(decode_e4m3fn(pair.raw_weight[row * pair.logical_k + col])?);
                }
                let scale_row = row / ACT_QUANT_BLOCK;
                result += block_sum
                    * f64::from(activation.decoded_scales_f32[block])
                    * f64::from(decode_e8m0fnu(
                        pair.raw_scale[scale_row * pair.scale_cols + block],
                    )?);
            }
            if !result.is_finite() {
                return Err(failure("f64 FP8 authority produced a non-finite output"));
            }
            out.push(result);
        }
        Ok(out)
    }

    fn dispatch_device_act_quant(
        batch: &mut hawking_core::metal::CommandBatch<'_>,
        input_bf16: &metal::Buffer,
        quantized: &metal::Buffer,
        scales: &metal::Buffer,
        cols: u32,
        threads: u32,
    ) -> hawking_core::Result<()> {
        batch.dispatch_threads(
            ACT_QUANT_KERNEL,
            (cols / ACT_QUANT_BLOCK as u32, 1, 1),
            (threads, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input_bf16), 0);
                encoder.set_buffer(1, Some(quantized), 0);
                encoder.set_buffer(2, Some(scales), 0);
                set_u32(encoder, 3, &cols);
            },
        )
    }

    fn dispatch_p5b_swiglu(
        batch: &mut hawking_core::metal::CommandBatch<'_>,
        gate: &metal::Buffer,
        up: &metal::Buffer,
        output: &metal::Buffer,
        route_weight: f32,
        count: u32,
        threads: u32,
    ) -> hawking_core::Result<()> {
        batch.dispatch_threads(
            P5B_SWIGLU_KERNEL,
            (count, 1, 1),
            (threads, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(gate), 0);
                encoder.set_buffer(1, Some(up), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<f32>() as u64,
                    &route_weight as *const f32 as *const _,
                );
                set_u32(encoder, 4, &count);
            },
        )
    }

    fn dispatch_p5b_combine(
        batch: &mut hawking_core::metal::CommandBatch<'_>,
        routed: &metal::Buffer,
        shared: &metal::Buffer,
        output: &metal::Buffer,
        count: u32,
        threads: u32,
    ) -> hawking_core::Result<()> {
        batch.dispatch_threads(
            P5B_COMBINE_KERNEL,
            (count, 1, 1),
            (threads, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(routed), 0);
                encoder.set_buffer(1, Some(shared), 0);
                encoder.set_buffer(2, Some(output), 0);
                set_u32(encoder, 3, &count);
            },
        )
    }

    fn dispatch_p5b_fp4_then_bf16(
        batch: &mut hawking_core::metal::CommandBatch<'_>,
        weights: &metal::Buffer,
        scales: &metal::Buffer,
        activation: &metal::Buffer,
        activation_scales: &metal::Buffer,
        output_f32: &metal::Buffer,
        output_bf16: &metal::Buffer,
        pair: &NativePairBytes,
        matvec_threads: u32,
        cast_threads: u32,
    ) -> ProbeResult<()> {
        let rows = u32::try_from(pair.rows).map_err(|_| failure("P5B FP4 rows do not fit u32"))?;
        let packed_k =
            u32::try_from(pair.packed_k).map_err(|_| failure("P5B FP4 K does not fit u32"))?;
        let scale_cols = u32::try_from(pair.scale_cols)
            .map_err(|_| failure("P5B FP4 scale columns do not fit u32"))?;
        let resources: [&metal::ResourceRef; 1] = [&**output_f32];
        batch.dispatch_threads_pair_in_one_encoder(
            P5B_FP4_KERNEL,
            (rows, 1, 1),
            (matvec_threads, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(weights), 0);
                encoder.set_buffer(1, Some(scales), 0);
                encoder.set_buffer(2, Some(activation), 0);
                encoder.set_buffer(3, Some(activation_scales), 0);
                encoder.set_buffer(4, Some(output_f32), 0);
                set_u32(encoder, 5, &rows);
                set_u32(encoder, 6, &packed_k);
                set_u32(encoder, 7, &scale_cols);
            },
            &resources,
            BF16_CAST_KERNEL,
            (rows, 1, 1),
            (cast_threads, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(output_f32), 0);
                encoder.set_buffer(1, Some(output_bf16), 0);
                set_u32(encoder, 2, &rows);
            },
        )?;
        Ok(())
    }

    fn dispatch_fp4_then_bf16(
        batch: &mut hawking_core::metal::CommandBatch<'_>,
        weights: &metal::Buffer,
        scales: &metal::Buffer,
        input: &metal::Buffer,
        output_f32: &metal::Buffer,
        output_bf16: &metal::Buffer,
        pair: &NativePairBytes,
        matvec_threads: u32,
        cast_threads: u32,
    ) -> ProbeResult<()> {
        let rows = u32::try_from(pair.rows).map_err(|_| failure("FP4 rows do not fit u32"))?;
        let packed_k =
            u32::try_from(pair.packed_k).map_err(|_| failure("FP4 K does not fit u32"))?;
        let scale_cols =
            u32::try_from(pair.scale_cols).map_err(|_| failure("FP4 scale cols do not fit u32"))?;
        let resources: [&metal::ResourceRef; 1] = [&**output_f32];
        batch.dispatch_threads_pair_in_one_encoder(
            FP4_KERNEL,
            (rows, 1, 1),
            (matvec_threads, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(weights), 0);
                encoder.set_buffer(1, Some(scales), 0);
                encoder.set_buffer(2, Some(input), 0);
                encoder.set_buffer(3, Some(output_f32), 0);
                set_u32(encoder, 4, &rows);
                set_u32(encoder, 5, &packed_k);
                set_u32(encoder, 6, &scale_cols);
            },
            &resources,
            BF16_CAST_KERNEL,
            (rows, 1, 1),
            (cast_threads, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(output_f32), 0);
                encoder.set_buffer(1, Some(output_bf16), 0);
                set_u32(encoder, 2, &rows);
            },
        )?;
        Ok(())
    }

    fn dispatch_fp8_then_bf16(
        batch: &mut hawking_core::metal::CommandBatch<'_>,
        weights: &metal::Buffer,
        scales: &metal::Buffer,
        activation: &metal::Buffer,
        activation_scales: &metal::Buffer,
        output_f32: &metal::Buffer,
        output_bf16: &metal::Buffer,
        pair: &NativePairBytes,
        matvec_threads: u32,
        cast_threads: u32,
    ) -> ProbeResult<()> {
        let rows = u32::try_from(pair.rows).map_err(|_| failure("FP8 rows do not fit u32"))?;
        let cols = u32::try_from(pair.logical_k).map_err(|_| failure("FP8 K does not fit u32"))?;
        let scale_cols =
            u32::try_from(pair.scale_cols).map_err(|_| failure("FP8 scale cols do not fit u32"))?;
        let resources: [&metal::ResourceRef; 1] = [&**output_f32];
        batch.dispatch_threads_pair_in_one_encoder(
            FP8_KERNEL,
            (rows, 1, 1),
            (matvec_threads, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(weights), 0);
                encoder.set_buffer(1, Some(scales), 0);
                encoder.set_buffer(2, Some(activation), 0);
                encoder.set_buffer(3, Some(activation_scales), 0);
                encoder.set_buffer(4, Some(output_f32), 0);
                set_u32(encoder, 5, &rows);
                set_u32(encoder, 6, &cols);
                set_u32(encoder, 7, &scale_cols);
            },
            &resources,
            BF16_CAST_KERNEL,
            (rows, 1, 1),
            (cast_threads, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(output_f32), 0);
                encoder.set_buffer(1, Some(output_bf16), 0);
                set_u32(encoder, 2, &rows);
            },
        )?;
        Ok(())
    }

    fn dispatch_p6a_gate(
        batch: &mut hawking_core::metal::CommandBatch<'_>,
        gate_weight_bf16: &metal::Buffer,
        input_bf16: &metal::Buffer,
        logits: &metal::Buffer,
        rows: u32,
        cols: u32,
        threads: u32,
    ) -> hawking_core::Result<()> {
        batch.dispatch_threads(P6A_GATE_KERNEL, (rows, 1, 1), (threads, 1, 1), |encoder| {
            encoder.set_buffer(0, Some(gate_weight_bf16), 0);
            encoder.set_buffer(1, Some(input_bf16), 0);
            encoder.set_buffer(2, Some(logits), 0);
            set_u32(encoder, 3, &rows);
            set_u32(encoder, 4, &cols);
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_p6a_route(
        batch: &mut hawking_core::metal::CommandBatch<'_>,
        logits: &metal::Buffer,
        tid2eid_i64_le: &metal::Buffer,
        ids: &metal::Buffer,
        weights: &metal::Buffer,
        scores: &metal::Buffer,
        valid: &metal::Buffer,
        token_id: u32,
        threads: u32,
    ) -> hawking_core::Result<()> {
        let expert_count = ROUTED_EXPERTS as u32;
        let top_k = ACTIVATED_EXPERTS as u32;
        batch.dispatch_threads(P6A_ROUTE_KERNEL, (1, 1, 1), (threads, 1, 1), |encoder| {
            encoder.set_buffer(0, Some(logits), 0);
            encoder.set_buffer(1, Some(tid2eid_i64_le), 0);
            encoder.set_buffer(2, Some(ids), 0);
            encoder.set_buffer(3, Some(weights), 0);
            encoder.set_buffer(4, Some(scores), 0);
            encoder.set_buffer(5, Some(valid), 0);
            set_u32(encoder, 6, &token_id);
            set_u32(encoder, 7, &expert_count);
            set_u32(encoder, 8, &top_k);
            encoder.set_bytes(
                9,
                std::mem::size_of::<f32>() as u64,
                &ROUTE_SCALE as *const f32 as *const _,
            );
        })
    }

    fn dispatch_device_act_quant_concurrent(
        batch: &mut hawking_core::metal::CommandBatch<'_>,
        input_bf16: &metal::Buffer,
        quantized: &metal::Buffer,
        scales: &metal::Buffer,
        cols: u32,
        threads: u32,
    ) -> hawking_core::Result<()> {
        batch.dispatch_threads_in_concurrent_group(
            ACT_QUANT_KERNEL,
            (cols / ACT_QUANT_BLOCK as u32, 1, 1),
            (threads, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input_bf16), 0);
                encoder.set_buffer(1, Some(quantized), 0);
                encoder.set_buffer(2, Some(scales), 0);
                set_u32(encoder, 3, &cols);
            },
        )
    }

    fn dispatch_p5b_fp4_concurrent(
        batch: &mut hawking_core::metal::CommandBatch<'_>,
        weights: &metal::Buffer,
        scales: &metal::Buffer,
        activation: &metal::Buffer,
        activation_scales: &metal::Buffer,
        output_f32: &metal::Buffer,
        pair: &NativePairBytes,
        threads: u32,
    ) -> ProbeResult<()> {
        let rows = u32::try_from(pair.rows).map_err(|_| failure("P6A FP4 rows exceed u32"))?;
        let packed_k =
            u32::try_from(pair.packed_k).map_err(|_| failure("P6A FP4 packed K exceeds u32"))?;
        let scale_cols =
            u32::try_from(pair.scale_cols).map_err(|_| failure("P6A FP4 scale cols exceed u32"))?;
        batch.dispatch_threads_in_concurrent_group(
            P5B_FP4_KERNEL,
            (rows, 1, 1),
            (threads, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(weights), 0);
                encoder.set_buffer(1, Some(scales), 0);
                encoder.set_buffer(2, Some(activation), 0);
                encoder.set_buffer(3, Some(activation_scales), 0);
                encoder.set_buffer(4, Some(output_f32), 0);
                set_u32(encoder, 5, &rows);
                set_u32(encoder, 6, &packed_k);
                set_u32(encoder, 7, &scale_cols);
            },
        )?;
        Ok(())
    }

    fn dispatch_fp8_concurrent(
        batch: &mut hawking_core::metal::CommandBatch<'_>,
        weights: &metal::Buffer,
        scales: &metal::Buffer,
        activation: &metal::Buffer,
        activation_scales: &metal::Buffer,
        output_f32: &metal::Buffer,
        pair: &NativePairBytes,
        threads: u32,
    ) -> ProbeResult<()> {
        let rows = u32::try_from(pair.rows).map_err(|_| failure("P6A FP8 rows exceed u32"))?;
        let cols = u32::try_from(pair.logical_k).map_err(|_| failure("P6A FP8 K exceeds u32"))?;
        let scale_cols =
            u32::try_from(pair.scale_cols).map_err(|_| failure("P6A FP8 scale cols exceed u32"))?;
        batch.dispatch_threads_in_concurrent_group(
            FP8_KERNEL,
            (rows, 1, 1),
            (threads, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(weights), 0);
                encoder.set_buffer(1, Some(scales), 0);
                encoder.set_buffer(2, Some(activation), 0);
                encoder.set_buffer(3, Some(activation_scales), 0);
                encoder.set_buffer(4, Some(output_f32), 0);
                set_u32(encoder, 5, &rows);
                set_u32(encoder, 6, &cols);
                set_u32(encoder, 7, &scale_cols);
            },
        )?;
        Ok(())
    }

    fn dispatch_bf16_cast_concurrent(
        batch: &mut hawking_core::metal::CommandBatch<'_>,
        input_f32: &metal::Buffer,
        output_bf16: &metal::Buffer,
        count: u32,
        threads: u32,
    ) -> hawking_core::Result<()> {
        batch.dispatch_threads_in_concurrent_group(
            BF16_CAST_KERNEL,
            (count, 1, 1),
            (threads, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input_f32), 0);
                encoder.set_buffer(1, Some(output_bf16), 0);
                set_u32(encoder, 2, &count);
            },
        )
    }

    fn dispatch_p6a_swiglu_concurrent(
        batch: &mut hawking_core::metal::CommandBatch<'_>,
        gate: &metal::Buffer,
        up: &metal::Buffer,
        output: &metal::Buffer,
        device_route_weights: &metal::Buffer,
        route_slot: u32,
        count: u32,
        threads: u32,
    ) -> hawking_core::Result<()> {
        batch.dispatch_threads_in_concurrent_group(
            P6A_SWIGLU_KERNEL,
            (count, 1, 1),
            (threads, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(gate), 0);
                encoder.set_buffer(1, Some(up), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.set_buffer(3, Some(device_route_weights), 0);
                set_u32(encoder, 4, &route_slot);
                set_u32(encoder, 5, &count);
            },
        )
    }

    fn dispatch_p5b_swiglu_concurrent(
        batch: &mut hawking_core::metal::CommandBatch<'_>,
        gate: &metal::Buffer,
        up: &metal::Buffer,
        output: &metal::Buffer,
        route_weight: f32,
        count: u32,
        threads: u32,
    ) -> hawking_core::Result<()> {
        batch.dispatch_threads_in_concurrent_group(
            P5B_SWIGLU_KERNEL,
            (count, 1, 1),
            (threads, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(gate), 0);
                encoder.set_buffer(1, Some(up), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<f32>() as u64,
                    &route_weight as *const f32 as *const _,
                );
                set_u32(encoder, 4, &count);
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_p6a_combine6_concurrent(
        batch: &mut hawking_core::metal::CommandBatch<'_>,
        routed: [&metal::Buffer; ACTIVATED_EXPERTS],
        shared: &metal::Buffer,
        output: &metal::Buffer,
        count: u32,
        threads: u32,
    ) -> hawking_core::Result<()> {
        batch.dispatch_threads_in_concurrent_group(
            P6A_COMBINE_KERNEL,
            (count, 1, 1),
            (threads, 1, 1),
            |encoder| {
                for (index, buffer) in routed.iter().enumerate() {
                    encoder.set_buffer(index as u64, Some(buffer), 0);
                }
                encoder.set_buffer(6, Some(shared), 0);
                encoder.set_buffer(7, Some(output), 0);
                set_u32(encoder, 8, &count);
            },
        )
    }

    fn dispatch_p5b_combine_concurrent(
        batch: &mut hawking_core::metal::CommandBatch<'_>,
        routed: &metal::Buffer,
        shared: &metal::Buffer,
        output: &metal::Buffer,
        count: u32,
        threads: u32,
    ) -> hawking_core::Result<()> {
        batch.dispatch_threads_in_concurrent_group(
            P5B_COMBINE_KERNEL,
            (count, 1, 1),
            (threads, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(routed), 0);
                encoder.set_buffer(1, Some(shared), 0);
                encoder.set_buffer(2, Some(output), 0);
                set_u32(encoder, 3, &count);
            },
        )
    }

    fn allocate_p6a_routed_gpu(
        context: &MetalContext,
        expert: &P6aRoutedCpuExpert,
    ) -> ProbeResult<P6aExpertGpuBuffers> {
        Ok(P6aExpertGpuBuffers {
            w1_weight: context.new_buffer_with_bytes_checked(&expert.w1.raw_weight)?,
            w1_scale: context.new_buffer_with_bytes_checked(&expert.w1.raw_scale)?,
            w3_weight: context.new_buffer_with_bytes_checked(&expert.w3.raw_weight)?,
            w3_scale: context.new_buffer_with_bytes_checked(&expert.w3.raw_scale)?,
            w2_weight: context.new_buffer_with_bytes_checked(&expert.w2.raw_weight)?,
            w2_scale: context.new_buffer_with_bytes_checked(&expert.w2.raw_scale)?,
            gate_f32: context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<f32>())?,
            up_f32: context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<f32>())?,
            gate_bf16: context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<u16>())?,
            up_bf16: context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<u16>())?,
            swiglu_bf16: context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<u16>())?,
            down_quant: context.new_buffer_checked(MOE_INTER_DIM)?,
            down_scales: context.new_buffer_checked(MOE_INTER_DIM / ACT_QUANT_BLOCK)?,
            down_f32: context.new_buffer_checked(HIDDEN_SIZE * std::mem::size_of::<f32>())?,
            down_bf16: context.new_buffer_checked(HIDDEN_SIZE * std::mem::size_of::<u16>())?,
        })
    }

    fn allocate_p6a_shared_gpu(
        context: &MetalContext,
        cpu: &P6aFullRouteCpuPipeline,
    ) -> ProbeResult<P6aSharedGpuBuffers> {
        Ok(P6aSharedGpuBuffers {
            w1_weight: context.new_buffer_with_bytes_checked(&cpu.shared_w1.raw_weight)?,
            w1_scale: context.new_buffer_with_bytes_checked(&cpu.shared_w1.raw_scale)?,
            w3_weight: context.new_buffer_with_bytes_checked(&cpu.shared_w3.raw_weight)?,
            w3_scale: context.new_buffer_with_bytes_checked(&cpu.shared_w3.raw_scale)?,
            w2_weight: context.new_buffer_with_bytes_checked(&cpu.shared_w2.raw_weight)?,
            w2_scale: context.new_buffer_with_bytes_checked(&cpu.shared_w2.raw_scale)?,
            gate_f32: context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<f32>())?,
            up_f32: context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<f32>())?,
            gate_bf16: context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<u16>())?,
            up_bf16: context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<u16>())?,
            swiglu_bf16: context.new_buffer_checked(MOE_INTER_DIM * std::mem::size_of::<u16>())?,
            down_quant: context.new_buffer_checked(MOE_INTER_DIM)?,
            down_scales: context.new_buffer_checked(MOE_INTER_DIM / ACT_QUANT_BLOCK)?,
            down_f32: context.new_buffer_checked(HIDDEN_SIZE * std::mem::size_of::<f32>())?,
            down_bf16: context.new_buffer_checked(HIDDEN_SIZE * std::mem::size_of::<u16>())?,
        })
    }

    fn require_batch_topology(
        timing: &MetalBatchTiming,
        expected_encoders: u64,
        expected_dispatches: u64,
        label: &str,
    ) -> ProbeResult<()> {
        if timing.command_buffers != 1
            || timing.compute_encoders != expected_encoders
            || timing.compute_dispatches != expected_dispatches
            || timing.gpu_duration_us.unwrap_or(0) == 0
            || timing.gpu_start_ns.is_none()
            || timing.gpu_end_ns.is_none()
        {
            return Err(failure(format!(
                "{label} does not have the declared real GPU timestamped topology",
            )));
        }
        Ok(())
    }

    fn require_256_threads(max: u32, kernel: &str) -> ProbeResult<u32> {
        if max < 256 {
            return Err(failure(format!(
                "{kernel} cannot support the P5A 256-thread authority geometry",
            )));
        }
        Ok(256)
    }

    fn require_threads(max: u32, required: u32, kernel: &str) -> ProbeResult<u32> {
        if max < required {
            return Err(failure(format!(
                "{kernel} cannot support the required P5B {required}-thread authority geometry",
            )));
        }
        Ok(required)
    }

    fn source_combine(routed: &[u16], shared: &[u16]) -> ProbeResult<Vec<f32>> {
        if routed.len() != HIDDEN_SIZE || shared.len() != HIDDEN_SIZE {
            return Err(failure(
                "source combine does not have hidden-width BF16 inputs",
            ));
        }
        let output: Vec<f32> = routed
            .iter()
            .zip(shared)
            .map(|(routed, shared)| {
                bf16::from_bits(*routed).to_f32() + bf16::from_bits(*shared).to_f32()
            })
            .collect();
        if output.iter().any(|value| !value.is_finite()) {
            return Err(failure("source combine produced a non-finite output"));
        }
        Ok(output)
    }

    fn dequantized_activation(activation: &ActQuantizedBf16Row) -> ProbeResult<Vec<f32>> {
        if activation.activation_e4m3fn.len() % ACT_QUANT_BLOCK != 0
            || activation.scales_e8m0fnu.len() * ACT_QUANT_BLOCK
                != activation.activation_e4m3fn.len()
        {
            return Err(failure(
                "source activation quantization has invalid geometry",
            ));
        }
        activation
            .activation_e4m3fn
            .iter()
            .enumerate()
            .map(|(index, byte)| {
                let value = decode_e4m3fn(*byte)?;
                let scale = decode_e8m0fnu(activation.scales_e8m0fnu[index / ACT_QUANT_BLOCK])?;
                let result = value * scale;
                if result.is_finite() {
                    Ok(result)
                } else {
                    Err(failure("dequantized activation is non-finite"))
                }
            })
            .collect()
    }

    fn f64_sqrt_softplus(logit: f64) -> ProbeResult<f64> {
        if !logit.is_finite() {
            return Err(failure("f64 sqrt-softplus received a non-finite logit"));
        }
        let softplus = if logit > 20.0 {
            logit
        } else if logit >= 0.0 {
            logit + (-logit).exp().ln_1p()
        } else {
            logit.exp().ln_1p()
        };
        let result = softplus.sqrt();
        if result.is_finite() && result > 0.0 {
            Ok(result)
        } else {
            Err(failure("f64 sqrt-softplus produced an invalid score"))
        }
    }

    fn fp4_nibble_f64(nibble: u8) -> ProbeResult<f64> {
        const TABLE: [f64; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        TABLE
            .get(nibble as usize)
            .copied()
            .ok_or_else(|| failure("FP4 nibble exceeds 4 bits"))
    }

    fn f64_to_bf16(values: &[f64]) -> ProbeResult<Vec<u16>> {
        let mut out = Vec::with_capacity(values.len());
        for &value in values {
            if !value.is_finite() || value.abs() > f32::MAX as f64 {
                return Err(failure(
                    "f64 authority cannot cross non-finite BF16 storage boundary",
                ));
            }
            out.push(bf16::from_f32(value as f32).to_bits());
        }
        Ok(out)
    }

    fn f32_metrics(reference: &[f32], observed: &[f32]) -> ProbeResult<Value> {
        if reference.is_empty() || reference.len() != observed.len() {
            return Err(failure(
                "f32 numerical score vectors have incompatible geometry",
            ));
        }
        let mut max_abs = 0.0_f64;
        let mut max_rel = 0.0_f64;
        let mut sum_abs = 0.0_f64;
        let mut worst = 0usize;
        let mut failures = 0usize;
        for (index, (&reference, &observed)) in reference.iter().zip(observed).enumerate() {
            if !reference.is_finite() || !observed.is_finite() {
                return Err(failure("f32 numerical score has non-finite output"));
            }
            let abs = f64::from((reference - observed).abs());
            let rel = abs / f64::from(reference.abs().max(1.0e-8));
            if abs > max_abs {
                max_abs = abs;
                worst = index;
            }
            max_rel = max_rel.max(rel);
            sum_abs += abs;
            if abs > ADMITTED_ABS_TOLERANCE + ADMITTED_REL_TOLERANCE * f64::from(reference.abs()) {
                failures += 1;
            }
        }
        Ok(json!({
            "status": if failures == 0 {"PASS_ADMITTED_ENVELOPE"} else {"FAIL_ADMITTED_ENVELOPE"},
            "comparison": "abs_error <= 0.05 + 0.0005 * abs(reference)",
            "failing_values": failures,
            "max_abs_error_text": f64_text(max_abs),
            "mean_abs_error_text": f64_text(sum_abs / reference.len() as f64),
            "max_relative_error_text": f64_text(max_rel),
            "worst_index": worst,
            "reference_at_worst_text": f32_text(reference[worst]),
            "device_at_worst_text": f32_text(observed[worst]),
        }))
    }

    fn f64_metrics_from_f32(reference: &[f64], observed: &[f32]) -> ProbeResult<Value> {
        if reference.is_empty() || reference.len() != observed.len() {
            return Err(failure(
                "f64/device score vectors have incompatible geometry",
            ));
        }
        let observed_f64: Vec<f64> = observed.iter().map(|value| f64::from(*value)).collect();
        f64_metrics(reference, &observed_f64)
    }

    fn f64_vector_metrics(reference: &[f64], observed: &[f32]) -> ProbeResult<Value> {
        if reference.len() != observed.len() {
            return Err(failure("f64/f32 score vectors have incompatible geometry"));
        }
        let observed_f64: Vec<f64> = observed.iter().map(|value| f64::from(*value)).collect();
        f64_metrics(reference, &observed_f64)
    }

    fn f64_metrics(reference: &[f64], observed: &[f64]) -> ProbeResult<Value> {
        if reference.is_empty() || reference.len() != observed.len() {
            return Err(failure(
                "f64 numerical score vectors have incompatible geometry",
            ));
        }
        let mut max_abs = 0.0_f64;
        let mut max_rel = 0.0_f64;
        let mut sum_abs = 0.0_f64;
        let mut worst = 0usize;
        for (index, (&reference, &observed)) in reference.iter().zip(observed).enumerate() {
            if !reference.is_finite() || !observed.is_finite() {
                return Err(failure("f64 numerical score has non-finite output"));
            }
            let abs = (reference - observed).abs();
            let rel = abs / reference.abs().max(1.0e-12);
            if abs > max_abs {
                max_abs = abs;
                worst = index;
            }
            max_rel = max_rel.max(rel);
            sum_abs += abs;
        }
        Ok(json!({
            "comparison": "separate f64 authority versus observed value; no bitwise claim implied",
            "max_abs_error_text": f64_text(max_abs),
            "mean_abs_error_text": f64_text(sum_abs / reference.len() as f64),
            "max_relative_error_text": f64_text(max_rel),
            "worst_index": worst,
            "f64_reference_at_worst_text": f64_text(reference[worst]),
            "observed_at_worst_text": f64_text(observed[worst]),
        }))
    }

    fn timing_json(series: &TimingSeries) -> ProbeResult<Value> {
        Ok(json!({
            "timestamp_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime",
            "gpu_duration": summary_json(&series.gpu_us)?,
            "host_encode_duration": summary_json(&series.encode_us)?,
            "host_submit_duration": summary_json(&series.submit_us)?,
            "host_wait_duration": summary_json(&series.wait_us)?,
            "host_wall_duration": summary_json(&series.host_wall_us)?,
            "gpu_intervals_ns": series.intervals_ns,
        }))
    }

    fn summary_json(values: &[u64]) -> ProbeResult<Value> {
        if values.is_empty() {
            return Err(failure("timing summary needs at least one measured value"));
        }
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        let percentile =
            |percentage: usize| sorted[(sorted.len() * percentage).div_ceil(100).saturating_sub(1)];
        let sum: u128 = sorted.iter().map(|value| u128::from(*value)).sum();
        Ok(json!({
            "count": sorted.len(),
            "minimum_us": sorted[0],
            "p50_us": percentile(50),
            "p95_us": percentile(95),
            "p99_us": percentile(99),
            "maximum_us": sorted[sorted.len() - 1],
            "mean_us_text": f64_text(sum as f64 / sorted.len() as f64),
            "samples_us_sorted": sorted,
        }))
    }

    fn stage1_logical_bytes(
        routed_gate: &NativePairBytes,
        routed_up: &NativePairBytes,
        shared_gate: &NativePairBytes,
        shared_up: &NativePairBytes,
    ) -> Value {
        let fp4 = |pair: &NativePairBytes| {
            json!({
                "weight_and_scale_read": pair.raw_weight.len() + pair.raw_scale.len(),
                "f32_activation_read": pair.logical_k * std::mem::size_of::<f32>(),
                "fp32_projection_written_then_bf16_cast_read": pair.rows * std::mem::size_of::<f32>(),
                "bf16_storage_written": pair.rows * std::mem::size_of::<u16>(),
            })
        };
        let fp8 = |pair: &NativePairBytes| {
            json!({
                "weight_and_scale_read": pair.raw_weight.len() + pair.raw_scale.len(),
                "native_activation_and_scale_read": pair.logical_k + pair.logical_k / ACT_QUANT_BLOCK,
                "fp32_projection_written_then_bf16_cast_read": pair.rows * std::mem::size_of::<f32>(),
                "bf16_storage_written": pair.rows * std::mem::size_of::<u16>(),
            })
        };
        json!({
            "routed_w1": fp4(routed_gate),
            "routed_w3": fp4(routed_up),
            "shared_w1": fp8(shared_gate),
            "shared_w3": fp8(shared_up),
        })
    }

    fn stage2_logical_bytes(routed_down: &NativePairBytes, shared_down: &NativePairBytes) -> Value {
        json!({
            "routed_w2": {
                "weight_and_scale_read": routed_down.raw_weight.len() + routed_down.raw_scale.len(),
                "f32_activation_read": routed_down.logical_k * std::mem::size_of::<f32>(),
                "fp32_projection_written_then_bf16_cast_read": routed_down.rows * std::mem::size_of::<f32>(),
                "bf16_storage_written": routed_down.rows * std::mem::size_of::<u16>(),
            },
            "shared_w2": {
                "weight_and_scale_read": shared_down.raw_weight.len() + shared_down.raw_scale.len(),
                "native_activation_and_scale_read": shared_down.logical_k + shared_down.logical_k / ACT_QUANT_BLOCK,
                "fp32_projection_written_then_bf16_cast_read": shared_down.rows * std::mem::size_of::<f32>(),
                "bf16_storage_written": shared_down.rows * std::mem::size_of::<u16>(),
            },
        })
    }

    fn pair_binding_json(pair: &NativePairBytes) -> Value {
        json!({
            "label": pair.label,
            "representation": pair.kind.as_str(),
            "weight": tensor_binding_json(&pair.weight),
            "scale": tensor_binding_json(&pair.scale),
            "geometry": {
                "out_rows": pair.rows,
                "logical_k": pair.logical_k,
                "packed_k": pair.packed_k,
                "scale_cols": pair.scale_cols,
            },
            "weight_sha256": sha256(&pair.raw_weight),
            "scale_sha256": sha256(&pair.raw_scale),
        })
    }

    fn tensor_range_binding_json(tensor: &DeepSeekV4TensorMetadata, start: u64, end: u64) -> Value {
        json!({
            "name": tensor.name,
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "tensor_bytes": tensor.bytes,
            "verified_range": {"start": start, "end": end, "bytes": end - start},
            "source_shard": tensor.source_shard,
            "segments": tensor.segments.iter().map(segment_json).collect::<Vec<_>>(),
        })
    }

    fn tensor_binding_json(tensor: &DeepSeekV4TensorMetadata) -> Value {
        tensor_range_binding_json(tensor, 0, tensor.bytes)
    }

    fn segment_json(segment: &DeepSeekV4Segment) -> Value {
        json!({
            "bytes": segment.bytes,
            "chunk_relpath": segment.chunk_relpath,
            "chunk_sha256": segment.sha256,
            "source_file_start": segment.source_file_start,
            "source_file_end": segment.source_file_end,
            "tensor_start": segment.tensor_start,
            "tensor_end": segment.tensor_end,
            "row_start": segment.row_start,
            "row_count": segment.row_count,
        })
    }

    fn parse_args() -> ProbeResult<Args> {
        let mut artifact = None::<PathBuf>;
        let mut out = None::<PathBuf>;
        let mut warmups = DEFAULT_WARMUPS;
        let mut trials = DEFAULT_TRIALS;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--p5b" | "--p6a" => {}
                "--artifact" => artifact = args.next().map(PathBuf::from),
                "--out" => out = args.next().map(PathBuf::from),
                "--warmups" => warmups = parse_positive(args.next(), "--warmups")?,
                "--trials" => trials = parse_positive(args.next(), "--trials")?,
                "--help" | "-h" => {
                    println!(
                        "usage: gravity_deepseek_v4_layer0_moe_metal_p5a [--p5b|--p6a] --artifact <absolute full Gravity dir> --out <absolute receipt.json> [--warmups N] [--trials N]",
                    );
                    std::process::exit(0);
                }
                other => return Err(failure(format!("unknown argument {other:?}"))),
            }
        }
        let artifact = artifact.ok_or_else(|| failure("--artifact is required"))?;
        let out = out.ok_or_else(|| failure("--out is required"))?;
        if !artifact.is_absolute() || !out.is_absolute() {
            return Err(failure("--artifact and --out must be absolute paths"));
        }
        if trials < 5 {
            return Err(failure("P5A requires at least five clean measured trials"));
        }
        Ok(Args {
            artifact,
            out,
            warmups,
            trials,
        })
    }

    fn parse_positive(value: Option<String>, flag: &str) -> ProbeResult<usize> {
        let raw = value.ok_or_else(|| failure(format!("{flag} needs a positive integer")))?;
        let parsed = raw
            .parse::<usize>()
            .map_err(|_| failure(format!("{flag} must be a positive integer")))?;
        if parsed == 0 {
            return Err(failure(format!("{flag} must be positive")));
        }
        Ok(parsed)
    }

    fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &u32) {
        encoder.set_bytes(
            index,
            std::mem::size_of::<u32>() as u64,
            value as *const u32 as *const _,
        );
    }

    fn read_gpu_u16(buffer: &metal::Buffer, count: usize) -> ProbeResult<Vec<u16>> {
        let bytes = count
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| failure("GPU BF16 readback byte count overflow"))?;
        if buffer.length() < bytes as u64 {
            return Err(failure(
                "GPU BF16 buffer is smaller than its requested readback",
            ));
        }
        Ok(unsafe { std::slice::from_raw_parts(buffer.contents() as *const u16, count).to_vec() })
    }

    fn read_gpu_u32(buffer: &metal::Buffer, count: usize) -> ProbeResult<Vec<u32>> {
        let bytes = count
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| failure("GPU u32 readback byte count overflow"))?;
        if buffer.length() < bytes as u64 {
            return Err(failure(
                "GPU u32 buffer is smaller than its requested readback",
            ));
        }
        Ok(unsafe { std::slice::from_raw_parts(buffer.contents() as *const u32, count).to_vec() })
    }

    fn read_gpu_u8(buffer: &metal::Buffer, count: usize) -> ProbeResult<Vec<u8>> {
        if buffer.length() < count as u64 {
            return Err(failure(
                "GPU byte buffer is smaller than its requested readback",
            ));
        }
        Ok(unsafe { std::slice::from_raw_parts(buffer.contents() as *const u8, count).to_vec() })
    }

    fn read_gpu_f32(buffer: &metal::Buffer, count: usize) -> ProbeResult<Vec<f32>> {
        let bytes = count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| failure("GPU f32 readback byte count overflow"))?;
        if buffer.length() < bytes as u64 {
            return Err(failure(
                "GPU f32 buffer is smaller than its requested readback",
            ));
        }
        let result =
            unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, count).to_vec() };
        if result.iter().any(|value| !value.is_finite()) {
            return Err(failure("GPU f32 readback contains non-finite output"));
        }
        Ok(result)
    }

    fn f32_bit_diagnostic(reference: &[f32], observed: &[f32]) -> Value {
        let first = reference
            .iter()
            .zip(observed)
            .enumerate()
            .find(|(_, (left, right))| left.to_bits() != right.to_bits())
            .map(|(index, (left, right))| {
                json!({
                    "index": index,
                    "reference_f32_text": f32_text(*left),
                    "observed_f32_text": f32_text(*right),
                    "reference_bits_hex": format!("0x{:08x}", left.to_bits()),
                    "observed_bits_hex": format!("0x{:08x}", right.to_bits()),
                    "ulp_distance": ulp_distance_f32(*left, *right),
                })
            });
        json!({
            "reference_len": reference.len(),
            "observed_len": observed.len(),
            "reference_sha256_f32_le": sha256(&f32_le_bytes(reference)),
            "observed_sha256_f32_le": sha256(&f32_le_bytes(observed)),
            "bits_exact": reference.len() == observed.len() && first.is_none(),
            "first_bit_difference": first,
        })
    }

    fn u32_bit_diagnostic(reference: &[u32], observed: &[u32]) -> Value {
        let first = reference
            .iter()
            .zip(observed)
            .enumerate()
            .find(|(_, (left, right))| left != right)
            .map(|(index, (left, right))| {
                json!({
                    "index": index,
                    "reference": left,
                    "observed": right,
                    "reference_hex": format!("0x{left:08x}"),
                    "observed_hex": format!("0x{right:08x}"),
                })
            });
        let bytes = |values: &[u32]| -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        };
        json!({
            "reference_len": reference.len(),
            "observed_len": observed.len(),
            "reference_sha256_u32_le": sha256(&bytes(reference)),
            "observed_sha256_u32_le": sha256(&bytes(observed)),
            "exact": reference.len() == observed.len() && first.is_none(),
            "first_difference": first,
        })
    }

    fn u16_bit_diagnostic(reference: &[u16], observed: &[u16]) -> Value {
        let first = reference
            .iter()
            .zip(observed)
            .enumerate()
            .find(|(_, (left, right))| left != right)
            .map(|(index, (left, right))| {
                json!({
                    "index": index,
                    "reference_hex": format!("0x{left:04x}"),
                    "observed_hex": format!("0x{right:04x}"),
                    "reference_f32_text": f32_text(bf16::from_bits(*left).to_f32()),
                    "observed_f32_text": f32_text(bf16::from_bits(*right).to_f32()),
                })
            });
        json!({
            "reference_len": reference.len(),
            "observed_len": observed.len(),
            "reference_sha256_u16_le": sha256(&u16_le_bytes(reference)),
            "observed_sha256_u16_le": sha256(&u16_le_bytes(observed)),
            "bits_exact": reference.len() == observed.len() && first.is_none(),
            "first_bit_difference": first,
        })
    }

    fn u8_bit_diagnostic(reference: &[u8], observed: &[u8]) -> Value {
        let first = reference
            .iter()
            .zip(observed)
            .enumerate()
            .find(|(_, (left, right))| left != right)
            .map(|(index, (left, right))| {
                json!({
                    "index": index,
                    "reference_hex": format!("0x{left:02x}"),
                    "observed_hex": format!("0x{right:02x}"),
                })
            });
        json!({
            "reference_len": reference.len(),
            "observed_len": observed.len(),
            "reference_sha256": sha256(reference),
            "observed_sha256": sha256(observed),
            "exact": reference.len() == observed.len() && first.is_none(),
            "first_difference": first,
        })
    }

    fn decode_u16_le(bytes: &[u8]) -> ProbeResult<Vec<u16>> {
        if bytes.len() % std::mem::size_of::<u16>() != 0 {
            return Err(failure("BF16 byte payload has an odd length"));
        }
        Ok(bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect())
    }

    fn u16_le_bytes(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn f64_le_bytes(values: &[f64]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn f32_text(value: f32) -> String {
        value.to_string()
    }

    fn f64_text(value: f64) -> String {
        value.to_string()
    }

    fn f32_texts(values: &[f32]) -> Vec<String> {
        values.iter().copied().map(f32_text).collect()
    }

    fn f64_texts(values: &[f64]) -> Vec<String> {
        values.iter().copied().map(f64_text).collect()
    }

    // Receipt seals are canonical JSON. Keep continuous metrics decimal text
    // rather than JSON binary-float literals so every V2.1 threshold and
    // observation has a stable, human-auditable representation. Integers stay
    // numeric for counters, IDs, and byte lengths.
    fn receipt_decimal_strings(value: Value) -> Value {
        match value {
            Value::Number(number) if number.is_i64() || number.is_u64() => Value::Number(number),
            Value::Number(number) => Value::String(number.to_string()),
            Value::Array(values) => {
                Value::Array(values.into_iter().map(receipt_decimal_strings).collect())
            }
            Value::Object(values) => Value::Object(
                values
                    .into_iter()
                    .map(|(key, value)| (key, receipt_decimal_strings(value)))
                    .collect(),
            ),
            other => other,
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
                let mut keys: Vec<&String> = values.keys().collect();
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

    fn seal(mut receipt: Value) -> ProbeResult<(Value, String)> {
        if receipt.get("seal_sha256").is_some() {
            return Err(failure("receipt unexpectedly already has a seal"));
        }
        let seal = sha256(&canonical_json(&receipt));
        receipt
            .as_object_mut()
            .ok_or_else(|| failure("receipt root is not an object"))?
            .insert("seal_sha256".to_owned(), Value::String(seal.clone()));
        Ok((receipt, seal))
    }

    fn write_new_receipt(path: &Path, receipt: &Value) -> ProbeResult<()> {
        if path.exists() {
            return Err(failure(format!(
                "refusing to overwrite existing P5A receipt {}",
                path.display(),
            )));
        }
        let parent = path
            .parent()
            .ok_or_else(|| failure("P5A receipt output has no parent directory"))?;
        fs::create_dir_all(parent)?;
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| failure("P5A receipt filename is not UTF-8"))?;
        let temporary = parent.join(format!(".{name}.{}.tmp", std::process::id()));
        if temporary.exists() {
            return Err(failure("P5A receipt temporary path already exists"));
        }
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        let mut file = options.open(&temporary)?;
        file.write_all(&canonical_json(receipt))?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    }

    fn failure(message: impl Into<String>) -> Box<dyn Error> {
        std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().any(|argument| argument == "--p6a") {
        macos::run_p6a()
    } else if std::env::args().any(|argument| argument == "--p5b") {
        macos::run_p5b()
    } else {
        macos::run()
    }
}

//! Bounded native-Metal P4B proof for the real DeepSeek-V4-Flash layer-0
//! position-one ratio-zero attention path.  It keeps the first two KV rows on
//! device, reads them causally as `[0, 1]`, and completes the P1 tail:
//! `inverse-RoPE -> converted WO-A/einsum -> WO-B -> mHC attention post`.
//!
//! This is intentionally an authority checkpoint, not a decode runtime,
//! generation benchmark, HCLI result, or BASE_TRUE_TPS claim.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "gravity_deepseek_v4_p4b_position1_complete_attention_metal requires macOS Metal",
    )
    .into())
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::gravity_deepseek_v4::{
        DeepSeekV4FullStreamReader, NativeScalePairKind, FULL_STREAM_SCHEMA, FULL_STREAM_STATUS,
    };
    use hawking_core::gravity_deepseek_v4_act_quant::{
        ACT_QUANT_BLOCK, LAYER0_WQ_A_COLS, LAYER0_WQ_A_ROWS, LAYER0_WQ_A_SCALE, LAYER0_WQ_A_WEIGHT,
    };
    use hawking_core::gravity_deepseek_v4_layer0_attention::{
        HEAD_DIM, KV_QAT_BLOCK, LAYER0_ATTN_SINK, LAYER0_KV_NORM_WEIGHT, LAYER0_Q_NORM_WEIGHT,
        LAYER0_WKV_SCALE, LAYER0_WKV_WEIGHT, LAYER0_WO_A_SCALE, LAYER0_WO_A_WEIGHT,
        LAYER0_WO_B_SCALE, LAYER0_WO_B_WEIGHT, LAYER0_WQ_B_SCALE, LAYER0_WQ_B_WEIGHT,
        NON_ROPE_HEAD_DIM, NUM_HEADS, O_LORA_RANK, Q_LORA_RANK, ROPE_HEAD_DIM, WKV_ROWS, WO_A_COLS,
        WO_A_ROWS, WO_B_COLS, WO_B_ROWS, WQ_B_ROWS,
    };
    use hawking_core::gravity_deepseek_v4_layer0_continuation::{
        layer0_position1_complete_attention_cpu_oracle,
        verify_layer0_position1_continuation_anchors,
        Layer0Position1CompleteAttentionCpuOracleResult, POSITION1, POSITION1_KV_ROWS,
        POSITION1_TOKEN_ID, WINDOW_SIZE,
    };
    use hawking_core::gravity_deepseek_v4_layer0_prefix::{
        layer0_prefix_cpu_oracle, EMBED_WEIGHT, HC_EPS, HC_MIX_WIDTH, HC_MULT, HC_SINKHORN_ITERS,
        HIDDEN_SIZE, LAYER0_ATTN_NORM_WEIGHT, LAYER0_HC_ATTN_BASE, LAYER0_HC_ATTN_FN,
        LAYER0_HC_ATTN_SCALE, PREFIX_TOKEN_ID, RMS_NORM_EPS,
    };
    use hawking_core::metal::{
        MetalContext, MetalDispatchTiming, PhysicalTraceGuard, PhysicalTraceIdentity,
    };
    use hawking_core::numeric_parity::{score_pair, Bounds, PairedScore};
    use metal::objc::{msg_send, sel, sel_impl};
    use metal::{CompileOptions, Library, MTLCommandBufferStatus, MTLSize};
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::error::Error;
    use std::fs::{self, File, OpenOptions};
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    const RECEIPT_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.p4b_position1_complete_attention_metal.v1";
    const RECEIPT_STATUS: &str =
        "PASS_REAL_METAL_P4B_POSITION1_COMPLETE_ATTENTION_PARITY_NOT_RUNTIME";
    const STRICT_DD_V2_RECEIPT_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.p4b_position1_complete_attention_metal.v2";
    const STRICT_DD_V2_RECEIPT_STATUS: &str =
        "PASS_REAL_METAL_P4B_POSITION1_COMPLETE_ATTENTION_DARWIN_DD_STRICT_PARITY_NOT_RUNTIME";
    const CPU_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.layer0_position1_complete_attention_cpu_oracle.v1";
    const CPU_STATUS: &str =
        "PASS_SOURCE_DERIVED_CPU_LAYER0_POSITION1_COMPLETE_ATTENTION_NOT_RUNTIME";
    const P4A_SCHEMA: &str = "hawking.gravity.deepseek_v4.p4a_layer0_attention_metal.v1";
    const P4A_STATUS: &str = "PASS_REAL_METAL_P4A_LAYER0_COMPLETE_ATTENTION_PARITY_NOT_RUNTIME";
    const P4A_TOPOLOGY_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.p4a_layer0_attention_topology_sweep.v1";
    const P4A_TOPOLOGY_STATUS: &str =
        "PASS_REAL_METAL_P4A_ONE_CB_COMPLETE_PARITY_TOPOLOGY_WIN_NOT_RUNTIME";
    const CPU_BASENAME: &str = "DSV4F_LAYER0_POSITION1_COMPLETE_ATTENTION_CPU_ORACLE-v1.json";
    const P4A_BASENAME: &str = "DSV4F_P4A_LAYER0_COMPLETE_ATTENTION_METAL-v1.json";
    const P4A_TOPOLOGY_BASENAME: &str = "DSV4F_P4A_LAYER0_ATTENTION_TOPOLOGY_SWEEP-v1.json";
    const STRICT_DD_V2_BASENAME: &str = "DSV4F_P4B_POSITION1_COMPLETE_ATTENTION_METAL-v2.json";

    const HC_KERNEL: &str = "deepseek_v4_p3a_layer0_hc_attn_pre_bos_authority";
    const RMS_KERNEL: &str = "deepseek_v4_p3a_rmsnorm_bf16_authority";
    const QAT_KERNEL: &str = "deepseek_v4_act_quant_bf16_ue8m0_authority";
    const FP8_KERNEL: &str = "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority";
    const CAST_KERNEL: &str = "deepseek_v4_p3a_fp32_to_bf16_authority";
    const PER_HEAD_KERNEL: &str = "deepseek_v4_p3a_per_head_rmsnorm_bf16_authority";
    const KV_QAT_KERNEL: &str = "deepseek_v4_p4a_kv_nonrope_qat_inplace_authority";
    const ROPE_KERNEL: &str = "deepseek_v4_p4b_rope_position1_bf16_authority";
    const CACHE_KERNEL: &str = "deepseek_v4_p4b_kv_cache_write_bf16_authority";
    const SPARSE_KERNEL: &str = "deepseek_v4_p4b_sparse_attention_position1_two_kv_sink_authority";
    const WO_A_KERNEL: &str = "deepseek_v4_p4a_wo_a_convert_bf16_einsum_authority";
    const HC_POST_KERNEL: &str = "deepseek_v4_p4a_hc_attn_post_authority";
    const HC_PRECISE_CONTROL_CANDIDATE_KERNEL: &str =
        "deepseek_v4_p4b_hc_post_comb_precise_exp_candidate";
    const HC_POST_ULP_REPAIR_TRACE_CANDIDATE_KERNEL: &str =
        "deepseek_v4_p4b_hc_post_cpu_exp_ulp_repair_trace_candidate";
    const HC_DARWIN_DD_CONTROL_CANDIDATE_KERNEL: &str =
        "deepseek_v4_p4b_hc_post_comb_darwin_dd_candidate";
    const P4B_DISPATCHES: u64 = 33;

    // Active macOS arm64 libSystem `expf` normal-path table.  This is only
    // used to prepare the unsealed general DD feasibility candidate; it is
    // neither a runtime registration nor an authority/baseline replacement.
    const DARWIN_EXPF_MAGIC_BITS: u64 = 0x4338_0000_0000_0000;
    const DARWIN_EXPF_TABLE: [u64; 128] = [
        0x3ff0000000000000,
        0x3feff63da9fb3335,
        0x3fefec9a3e778061,
        0x3fefe315e86e7f85,
        0x3fefd9b0d3158574,
        0x3fefd06b29ddf6de,
        0x3fefc74518759bc8,
        0x3fefbe3ecac6f383,
        0x3fefb5586cf9890f,
        0x3fefac922b7247f7,
        0x3fefa3ec32d3d1a2,
        0x3fef9b66affed31b,
        0x3fef9301d0125b51,
        0x3fef8abdc06c31cc,
        0x3fef829aaea92de0,
        0x3fef7a98c8a58e51,
        0x3fef72b83c7d517b,
        0x3fef6af9388c8dea,
        0x3fef635beb6fcb75,
        0x3fef5be084045cd4,
        0x3fef54873168b9aa,
        0x3fef4d5022fcd91d,
        0x3fef463b88628cd6,
        0x3fef3f49917ddc96,
        0x3fef387a6e756238,
        0x3fef31ce4fb2a63f,
        0x3fef2b4565e27cdd,
        0x3fef24dfe1f56381,
        0x3fef1e9df51fdee1,
        0x3fef187fd0dad990,
        0x3fef1285a6e4030b,
        0x3fef0cafa93e2f56,
        0x3fef06fe0a31b715,
        0x3fef0170fc4cd831,
        0x3feefc08b26416ff,
        0x3feef6c55f929ff1,
        0x3feef1a7373aa9cb,
        0x3feeecae6d05d866,
        0x3feee7db34e59ff7,
        0x3feee32dc313a8e5,
        0x3feedea64c123422,
        0x3feeda4504ac801c,
        0x3feed60a21f72e2a,
        0x3feed1f5d950a897,
        0x3feece086061892d,
        0x3feeca41ed1d0057,
        0x3feec6a2b5c13cd0,
        0x3feec32af0d7d3de,
        0x3feebfdad5362a27,
        0x3feebcb299fddd0d,
        0x3feeb9b2769d2ca7,
        0x3feeb6daa2cf6642,
        0x3feeb42b569d4f82,
        0x3feeb1a4ca5d920f,
        0x3feeaf4736b527da,
        0x3feead12d497c7fd,
        0x3feeab07dd485429,
        0x3feea9268a5946b7,
        0x3feea76f15ad2148,
        0x3feea5e1b976dc09,
        0x3feea47eb03a5585,
        0x3feea34634ccc320,
        0x3feea23882552225,
        0x3feea155d44ca973,
        0x3feea09e667f3bcd,
        0x3feea012750bdabf,
        0x3fee9fb23c651a2f,
        0x3fee9f7df9519484,
        0x3fee9f75e8ec5f74,
        0x3fee9f9a48a58174,
        0x3fee9feb564267c9,
        0x3feea0694fde5d3f,
        0x3feea11473eb0187,
        0x3feea1ed0130c132,
        0x3feea2f336cf4e62,
        0x3feea427543e1a12,
        0x3feea589994cce13,
        0x3feea71a4623c7ad,
        0x3feea8d99b4492ed,
        0x3feeaac7d98a6699,
        0x3feeace5422aa0db,
        0x3feeaf3216b5448c,
        0x3feeb1ae99157736,
        0x3feeb45b0b91ffc6,
        0x3feeb737b0cdc5e5,
        0x3feeba44cbc8520f,
        0x3feebd829fde4e50,
        0x3feec0f170ca07ba,
        0x3feec49182a3f090,
        0x3feec86319e32323,
        0x3feecc667b5de565,
        0x3feed09bec4a2d33,
        0x3feed503b23e255d,
        0x3feed99e1330b358,
        0x3feede6b5579fdbf,
        0x3feee36bbfd3f37a,
        0x3feee89f995ad3ad,
        0x3feeee07298db666,
        0x3feef3a2b84f15fb,
        0x3feef9728de5593a,
        0x3feeff76f2fb5e47,
        0x3fef05b030a1064a,
        0x3fef0c1e904bc1d2,
        0x3fef12c25bd71e09,
        0x3fef199bdd85529c,
        0x3fef20ab5fffd07a,
        0x3fef27f12e57d14b,
        0x3fef2f6d9406e7b5,
        0x3fef3720dcef9069,
        0x3fef3f0b555dc3fa,
        0x3fef472d4a07897c,
        0x3fef4f87080d89f2,
        0x3fef5818dcfba487,
        0x3fef60e316c98398,
        0x3fef69e603db3285,
        0x3fef7321f301b460,
        0x3fef7c97337b9b5f,
        0x3fef864614f5a129,
        0x3fef902ee78b3ff6,
        0x3fef9a51fbc74c83,
        0x3fefa4afa2a490da,
        0x3fefaf482d8e67f1,
        0x3fefba1bee615a27,
        0x3fefc52b376bba97,
        0x3fefd0765b6e4540,
        0x3fefdbfdad9cbe14,
        0x3fefe7c1819e90d8,
        0x3feff3c22b8f71f1,
    ];

    type ProbeResult<T> = Result<T, Box<dyn Error>>;

    struct Args {
        artifact: PathBuf,
        cpu_oracle: PathBuf,
        p4a_authority: PathBuf,
        p4a_topology: PathBuf,
        out: Option<PathBuf>,
        precise_mhc_control_candidate: bool,
        strict_mhc_control_trace_candidate: bool,
        strict_mhc_control_dd_candidate: bool,
        seal_strict_mhc_control_dd_v2: bool,
    }

    struct ReceiptBinding {
        path: PathBuf,
        file_sha256: String,
        seal_sha256: String,
    }

    struct HcState {
        embed: metal::Buffer,
        reduced: metal::Buffer,
        rsqrt: metal::Buffer,
        mixes: metal::Buffer,
        pre: metal::Buffer,
        post: metal::Buffer,
        comb: metal::Buffer,
        attn_norm: metal::Buffer,
    }

    struct LinearScratch {
        activation: metal::Buffer,
        scales: metal::Buffer,
        fp32: metal::Buffer,
        bf16: metal::Buffer,
    }

    struct KvScratch {
        norm: metal::Buffer,
        qat: metal::Buffer,
        activation: metal::Buffer,
        scales: metal::Buffer,
    }

    struct RopeSparseF64 {
        scores: Vec<f64>,
        denominators: Vec<f64>,
        attention: Vec<f64>,
        derotated: Vec<f64>,
    }

    struct HcControlsF64 {
        post: Vec<f64>,
        comb: Vec<f64>,
    }

    /// Separate strict-math library for a trace-specific P1 diagnostic.  It
    /// is intentionally not added to `MetalContext` or any runtime registry:
    /// the only permitted use is proving whether the real terminal storage
    /// becomes exact with strict control arithmetic plus the two-logit repair.
    struct StrictMhcControlTraceCandidate {
        #[allow(dead_code)]
        library: Library,
        control_pipeline: metal::ComputePipelineState,
        repair_pipeline: metal::ComputePipelineState,
    }

    /// Fresh general P4B diagnostic.  It is source/artifact-bound but never
    /// placed in the runtime registry: a strict F32 double-double rebuild of
    /// the active Darwin expf control path replaces only mHC exponent calls.
    struct StrictMhcControlDdCandidate {
        #[allow(dead_code)]
        library: Library,
        control_pipeline: metal::ComputePipelineState,
        table: metal::Buffer,
        table_sha256: String,
    }

    /// One explicit dynamic-library command buffer.  This is deliberately
    /// separate from `MetalContext`'s registered authority pipelines, so the
    /// receipt can account for it without implying a runtime registration.
    #[derive(Clone, Copy)]
    struct DirectCandidateTiming {
        encode_us: u64,
        submit_us: u64,
        wait_us: u64,
        host_wall_us: u64,
        gpu_duration_us: u64,
        gpu_start_ns: u64,
        gpu_end_ns: u64,
    }

    /// Fresh-run evidence captured before any GPU dispatch for the create-new
    /// strict-DD v2 receipt.  Legacy v1/candidate modes deliberately do not
    /// acquire this extra provenance surface.
    struct StrictDdV2RunProvenance {
        run_nonce: String,
        run_started_unix_ns: String,
        process_id: u32,
        executable: Value,
        source_code_inspection: Value,
        host_platform: Value,
    }

    fn failure(message: impl Into<String>) -> Box<dyn Error> {
        std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
    }

    impl StrictDdV2RunProvenance {
        fn capture() -> ProbeResult<Self> {
            let process_id = std::process::id();
            if process_id == 0 {
                return Err(failure("strict-DD v2 process ID is zero"));
            }
            let executable = executable_provenance()?;
            Ok(Self {
                run_nonce: secure_run_nonce()?,
                run_started_unix_ns: unix_time_ns()?,
                process_id,
                source_code_inspection: source_code_inspection(&executable)?,
                executable,
                host_platform: host_platform_provenance()?,
            })
        }
    }

    pub fn run() -> ProbeResult<()> {
        let args = parse_args()?;
        let reader = DeepSeekV4FullStreamReader::admit(&args.artifact)?;
        verify_layer0_position1_continuation_anchors(&reader)?;
        let cpu = layer0_position1_complete_attention_cpu_oracle(&reader)?;
        let prefix0 = layer0_prefix_cpu_oracle(&reader)?;
        let cpu_binding = validate_cpu_receipt(&args.cpu_oracle, &reader, &cpu)?;
        let p4a_binding = validate_p4a_receipt(&args.p4a_authority, &reader)?;
        let topology_binding = validate_topology_receipt(&args.p4a_topology, &reader)?;
        let strict_dd_v2_run = if args.seal_strict_mhc_control_dd_v2 {
            Some(StrictDdV2RunProvenance::capture()?)
        } else {
            None
        };

        let embed0 = embed_row(&reader, PREFIX_TOKEN_ID)?;
        let embed1 = embed_row(&reader, POSITION1_TOKEN_ID)?;
        let hc_fn = full(&reader, LAYER0_HC_ATTN_FN)?;
        let hc_base = full(&reader, LAYER0_HC_ATTN_BASE)?;
        let hc_scale = full(&reader, LAYER0_HC_ATTN_SCALE)?;
        let attn_norm_weight = full(&reader, LAYER0_ATTN_NORM_WEIGHT)?;
        let q_norm_weight = full(&reader, LAYER0_Q_NORM_WEIGHT)?;
        let kv_norm_weight = full(&reader, LAYER0_KV_NORM_WEIGHT)?;
        let sink = full(&reader, LAYER0_ATTN_SINK)?;
        let (wq_a_weight, wq_a_scale) = fp8_pair(
            &reader,
            LAYER0_WQ_A_WEIGHT,
            LAYER0_WQ_A_SCALE,
            LAYER0_WQ_A_ROWS,
            LAYER0_WQ_A_COLS,
        )?;
        let (wq_b_weight, wq_b_scale) = fp8_pair(
            &reader,
            LAYER0_WQ_B_WEIGHT,
            LAYER0_WQ_B_SCALE,
            WQ_B_ROWS,
            Q_LORA_RANK,
        )?;
        let (wkv_weight, wkv_scale) = fp8_pair(
            &reader,
            LAYER0_WKV_WEIGHT,
            LAYER0_WKV_SCALE,
            WKV_ROWS,
            HIDDEN_SIZE,
        )?;
        let (wo_a_weight, wo_a_scale) = fp8_pair(
            &reader,
            LAYER0_WO_A_WEIGHT,
            LAYER0_WO_A_SCALE,
            WO_A_ROWS,
            WO_A_COLS,
        )?;
        let (wo_b_weight, wo_b_scale) = fp8_pair(
            &reader,
            LAYER0_WO_B_WEIGHT,
            LAYER0_WO_B_SCALE,
            WO_B_ROWS,
            WO_B_COLS,
        )?;
        geometry_check(
            &embed0,
            &embed1,
            &hc_fn,
            &hc_base,
            &hc_scale,
            &attn_norm_weight,
            &q_norm_weight,
            &kv_norm_weight,
            &sink,
        )?;

        let rope_q_f64 = rope_f64(
            &cpu.causal.q_head_norm_bf16_bits,
            NUM_HEADS,
            &cpu.causal.rope_table.cos_f32,
            &cpu.causal.rope_table.sin_f32,
            false,
        )?;
        let hc_controls_f64 = hc_controls_f64(&embed1, &hc_fn, &hc_scale, &hc_base)?;
        let rope_kv_f64 = rope_f64(
            &cpu.causal.kv_inplace_qat.output_bf16_bits,
            1,
            &cpu.causal.rope_table.cos_f32,
            &cpu.causal.rope_table.sin_f32,
            false,
        )?;
        let sparse_f64 = sparse_f64(
            &cpu.causal.q_position1_rope_bf16_bits,
            &cpu.causal.kv_cache_two_rows_bf16_bits,
            &f32bytes_to_vec(&sink)?,
            &cpu.causal.rope_table.cos_f32,
            &cpu.causal.rope_table.sin_f32,
        )?;
        let wo_a_f64 = wo_a_f64(
            &cpu.causal.sparse_attention_derotated_bf16_bits,
            &wo_a_weight,
            &wo_a_scale,
        )?;
        let wo_b_f64 = fp8_f64(
            &wo_b_weight,
            &wo_b_scale,
            &cpu.wo_b.quantized_input.activation_e4m3fn,
            &cpu.wo_b.quantized_input.scales_e8m0fnu,
            WO_B_ROWS,
            WO_B_COLS,
        )?;
        let hc_post_f64 = hc_post_f64(
            &cpu.wo_b.output.bf16_bits,
            &cpu.causal.token1_prefix.hc_replicated_bf16_bits,
            &cpu.causal.token1_prefix.hc_post_f32,
            &cpu.causal.token1_prefix.hc_comb_f32,
        )?;

        let context = MetalContext::new_with_trace(true)?;
        let device_name = context.device_name();
        let mut limits = serde_json::Map::new();
        let mut kernels = vec![
            HC_KERNEL,
            RMS_KERNEL,
            QAT_KERNEL,
            FP8_KERNEL,
            CAST_KERNEL,
            PER_HEAD_KERNEL,
            KV_QAT_KERNEL,
            ROPE_KERNEL,
            CACHE_KERNEL,
            SPARSE_KERNEL,
            WO_A_KERNEL,
            HC_POST_KERNEL,
        ];
        if args.precise_mhc_control_candidate {
            kernels.push(HC_PRECISE_CONTROL_CANDIDATE_KERNEL);
        }
        for kernel in kernels {
            let pipeline = context.pipeline(kernel)?;
            limits.insert(
                kernel.to_owned(),
                json!({"thread_execution_width":pipeline.thread_execution_width(),"max_total_threads_per_threadgroup":pipeline.max_total_threads_per_threadgroup()}),
            );
        }
        let strict_mhc_trace_candidate = if args.strict_mhc_control_trace_candidate {
            Some(StrictMhcControlTraceCandidate::new(&context)?)
        } else {
            None
        };
        let strict_mhc_dd_requested =
            args.strict_mhc_control_dd_candidate || args.seal_strict_mhc_control_dd_v2;
        let strict_mhc_dd_candidate = if strict_mhc_dd_requested {
            Some(StrictMhcControlDdCandidate::new(&context)?)
        } else {
            None
        };
        if let Some(candidate) = strict_mhc_dd_candidate.as_ref() {
            limits.insert(
                HC_DARWIN_DD_CONTROL_CANDIDATE_KERNEL.to_owned(),
                json!({
                    "thread_execution_width":candidate.control_pipeline.thread_execution_width(),
                    "max_total_threads_per_threadgroup":candidate.control_pipeline.max_total_threads_per_threadgroup(),
                    "fast_math_enabled":false,
                }),
            );
        }

        let hc_fn_b = context.new_buffer_with_bytes_checked(&hc_fn)?;
        let hc_scale_b = context.new_buffer_with_bytes_checked(&hc_scale)?;
        let hc_base_b = context.new_buffer_with_bytes_checked(&hc_base)?;
        let attn_norm_w_b = context.new_buffer_with_bytes_checked(&attn_norm_weight)?;
        let q_norm_w_b = context.new_buffer_with_bytes_checked(&q_norm_weight)?;
        let kv_norm_w_b = context.new_buffer_with_bytes_checked(&kv_norm_weight)?;
        let sink_b = context.new_buffer_with_bytes_checked(&sink)?;
        let rope_cos_b =
            context.new_buffer_with_bytes_checked(&f32bytes(&cpu.causal.rope_table.cos_f32))?;
        let rope_sin_b =
            context.new_buffer_with_bytes_checked(&f32bytes(&cpu.causal.rope_table.sin_f32))?;
        let wq_a_w_b = context.new_buffer_with_bytes_checked(&wq_a_weight)?;
        let wq_a_s_b = context.new_buffer_with_bytes_checked(&wq_a_scale)?;
        let wq_b_w_b = context.new_buffer_with_bytes_checked(&wq_b_weight)?;
        let wq_b_s_b = context.new_buffer_with_bytes_checked(&wq_b_scale)?;
        let wkv_w_b = context.new_buffer_with_bytes_checked(&wkv_weight)?;
        let wkv_s_b = context.new_buffer_with_bytes_checked(&wkv_scale)?;
        let wo_a_w_b = context.new_buffer_with_bytes_checked(&wo_a_weight)?;
        let wo_a_s_b = context.new_buffer_with_bytes_checked(&wo_a_scale)?;
        let wo_b_w_b = context.new_buffer_with_bytes_checked(&wo_b_weight)?;
        let wo_b_s_b = context.new_buffer_with_bytes_checked(&wo_b_scale)?;

        let p0 = new_hc_state(&context, &embed0)?;
        let p1 = new_hc_state(&context, &embed1)?;
        let p0_wkv = new_linear_scratch(&context, HIDDEN_SIZE, WKV_ROWS)?;
        let p1_wq_a = new_linear_scratch(&context, LAYER0_WQ_A_COLS, LAYER0_WQ_A_ROWS)?;
        let p1_wq_b = new_linear_scratch(&context, Q_LORA_RANK, WQ_B_ROWS)?;
        let p1_wkv = new_linear_scratch(&context, HIDDEN_SIZE, WKV_ROWS)?;
        let p1_wo_b = new_linear_scratch(&context, WO_B_COLS, WO_B_ROWS)?;
        let p0_kv = new_kv_scratch(&context)?;
        let p1_kv = new_kv_scratch(&context)?;
        let p1_q_norm = context.new_buffer_checked(Q_LORA_RANK * 2)?;
        let p1_q_head = context.new_buffer_checked(WQ_B_ROWS * 2)?;
        let p1_q_rope = context.new_buffer_checked(WQ_B_ROWS * 2)?;
        let p1_kv_rope = context.new_buffer_checked(HEAD_DIM * 2)?;
        let cache_initial = vec![0u8; WINDOW_SIZE * HEAD_DIM * 2];
        let kv_cache = context.new_buffer_with_bytes_checked(&cache_initial)?;
        let p1_sparse = context.new_buffer_checked(WQ_B_ROWS * 2)?;
        let p1_scores = context.new_buffer_checked(NUM_HEADS * POSITION1_KV_ROWS * 4)?;
        let p1_denoms = context.new_buffer_checked(NUM_HEADS * 4)?;
        let p1_derotated = context.new_buffer_checked(WQ_B_ROWS * 2)?;
        let p1_wo_a = context.new_buffer_checked(WO_A_ROWS * 2)?;
        let p1_hc_final = context.new_buffer_checked(HC_MULT * HIDDEN_SIZE * 2)?;

        let hidden = HIDDEN_SIZE as u32;
        let hc_mult = HC_MULT as u32;
        let mix_width = HC_MIX_WIDTH as u32;
        let sinkhorn = HC_SINKHORN_ITERS as u32;
        let norm_eps = RMS_NORM_EPS;
        let hc_eps = HC_EPS;
        let heads = NUM_HEADS as u32;
        let head_dim = HEAD_DIM as u32;
        let q_lora = Q_LORA_RANK as u32;
        let rope_dim = ROPE_HEAD_DIM as u32;
        let kv_block = KV_QAT_BLOCK as u32;
        let cache_capacity = WINDOW_SIZE as u32;
        let sparse_scale = (HEAD_DIM as f32).powf(-0.5);
        let wq_a_rows = LAYER0_WQ_A_ROWS as u32;
        let wq_a_cols = LAYER0_WQ_A_COLS as u32;
        let wq_a_scale_cols = (LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK) as u32;
        let wq_b_rows = WQ_B_ROWS as u32;
        let wq_b_scale_cols = (Q_LORA_RANK / ACT_QUANT_BLOCK) as u32;
        let wkv_rows = WKV_ROWS as u32;
        let wkv_scale_cols = (HIDDEN_SIZE / ACT_QUANT_BLOCK) as u32;
        let wo_a_rows = WO_A_ROWS as u32;
        let wo_a_cols = WO_A_COLS as u32;
        let wo_a_scale_cols = (WO_A_COLS / ACT_QUANT_BLOCK) as u32;
        let o_rank = O_LORA_RANK as u32;
        let wo_b_rows = WO_B_ROWS as u32;
        let wo_b_cols = WO_B_COLS as u32;
        let wo_b_scale_cols = (WO_B_COLS / ACT_QUANT_BLOCK) as u32;
        let position0 = 0u32;
        let position1 = POSITION1 as u32;
        let inverse0 = 0u32;
        let inverse1 = 1u32;

        let run_nonce = if let Some(provenance) = strict_dd_v2_run.as_ref() {
            provenance.run_nonce.clone()
        } else {
            sha256_join(&[
                reader.manifest_seal_sha256(),
                &cpu_binding.seal_sha256,
                &p4a_binding.seal_sha256,
                &topology_binding.seal_sha256,
                "dsv4f_p4b_position1_complete_attention_v1",
            ])
        };
        let interval_id = if let Some(provenance) = strict_dd_v2_run.as_ref() {
            let executable_sha256 = provenance
                .executable
                .get("sha256")
                .and_then(Value::as_str)
                .ok_or_else(|| failure("strict-DD v2 executable provenance has no SHA-256"))?;
            sha256_join(&[
                &run_nonce,
                reader.manifest_seal_sha256(),
                executable_sha256,
                "dsv4f_p4b_position1_complete_attention_strict_dd_v2",
            ])
        } else {
            sha256_join(&[&run_nonce, "p4b_position1_complete_attention"])
        };
        let trace = PhysicalTraceGuard::begin(PhysicalTraceIdentity::new(
            interval_id.clone(),
            run_nonce.clone(),
            "dsv4f_p4b_position1".to_owned(),
            "causal_kv_and_complete_attention_tail".to_owned(),
            Some(1),
            POSITION1,
        )?)?;
        // The strict-DD command is intentionally outside `MetalContext`'s
        // authority-pipeline registry.  Bind its native Metal labels to this
        // exact physical interval without pretending it increments the
        // registry-owned trace counter.
        let strict_dd_direct_trace_label = format!(
            "hawking.physical.v1|interval_id={interval_id}|run_nonce={run_nonce}|phase=dsv4f_p4b_position1|role=causal_kv_and_complete_attention_tail|batch=1|iteration={POSITION1}|kind=dynamic_library_command|command_index=external"
        );
        let mut stages: Vec<(&str, &str, MetalDispatchTiming, usize, usize, u64)> = Vec::new();
        macro_rules! record {
            ($name:expr, $kernel:expr, $call:expr, $read:expr, $written:expr, $flops:expr) => {{
                let timing = $call?;
                checked(&timing, $name)?;
                stages.push(($name, $kernel, timing, $read, $written, $flops));
            }};
        }

        // Position zero produces the first actual device-resident cache row.
        record!(
            "p0_mhc_pre",
            HC_KERNEL,
            hc(
                &context,
                &p0,
                &hc_fn_b,
                &hc_scale_b,
                &hc_base_b,
                hidden,
                hc_mult,
                mix_width,
                sinkhorn,
                norm_eps,
                hc_eps
            ),
            hc_fn.len() + embed0.len() * HC_MULT,
            HIDDEN_SIZE * 2,
            (HC_MIX_WIDTH * HC_MULT * HIDDEN_SIZE * 2) as u64
        );
        record!(
            "p0_attn_norm",
            RMS_KERNEL,
            rms(
                &context,
                &p0.reduced,
                &attn_norm_w_b,
                &p0.attn_norm,
                hidden,
                norm_eps
            ),
            HIDDEN_SIZE * 4,
            HIDDEN_SIZE * 2,
            (HIDDEN_SIZE * 4) as u64
        );
        record!(
            "p0_wkv_qat",
            QAT_KERNEL,
            qat(
                &context,
                &p0.attn_norm,
                &p0_wkv.activation,
                &p0_wkv.scales,
                hidden
            ),
            HIDDEN_SIZE * 2,
            HIDDEN_SIZE + HIDDEN_SIZE / ACT_QUANT_BLOCK,
            (HIDDEN_SIZE * 2) as u64
        );
        record!(
            "p0_wkv_fp8",
            FP8_KERNEL,
            fp8(
                &context,
                &wkv_w_b,
                &wkv_s_b,
                &p0_wkv.activation,
                &p0_wkv.scales,
                &p0_wkv.fp32,
                wkv_rows,
                hidden,
                wkv_scale_cols
            ),
            wkv_weight.len() + wkv_scale.len() + HIDDEN_SIZE + HIDDEN_SIZE / ACT_QUANT_BLOCK,
            WKV_ROWS * 4,
            (WKV_ROWS * HIDDEN_SIZE * 2) as u64
        );
        record!(
            "p0_wkv_cast",
            CAST_KERNEL,
            cast(&context, &p0_wkv.fp32, &p0_wkv.bf16, wkv_rows),
            WKV_ROWS * 4,
            WKV_ROWS * 2,
            0
        );
        record!(
            "p0_kv_norm",
            RMS_KERNEL,
            rms(
                &context,
                &p0_wkv.bf16,
                &kv_norm_w_b,
                &p0_kv.norm,
                head_dim,
                norm_eps
            ),
            HEAD_DIM * 4,
            HEAD_DIM * 2,
            (HEAD_DIM * 4) as u64
        );
        record!(
            "p0_kv_qat",
            KV_QAT_KERNEL,
            kv_qat(
                &context,
                &p0_kv.norm,
                &p0_kv.qat,
                &p0_kv.activation,
                &p0_kv.scales,
                head_dim,
                rope_dim,
                kv_block
            ),
            HEAD_DIM * 2,
            HEAD_DIM * 2 + NON_ROPE_HEAD_DIM + NON_ROPE_HEAD_DIM / KV_QAT_BLOCK,
            (NON_ROPE_HEAD_DIM * 2) as u64
        );
        record!(
            "p0_kv_cache_write",
            CACHE_KERNEL,
            cache_write(
                &context,
                &p0_kv.qat,
                &kv_cache,
                position0,
                head_dim,
                cache_capacity
            ),
            HEAD_DIM * 2,
            HEAD_DIM * 2,
            0
        );

        // Position one consumes raw row 19923 and reads both device cache rows.
        record!(
            "p1_mhc_pre",
            HC_KERNEL,
            hc(
                &context,
                &p1,
                &hc_fn_b,
                &hc_scale_b,
                &hc_base_b,
                hidden,
                hc_mult,
                mix_width,
                sinkhorn,
                norm_eps,
                hc_eps
            ),
            hc_fn.len() + embed1.len() * HC_MULT,
            HIDDEN_SIZE * 2,
            (HC_MIX_WIDTH * HC_MULT * HIDDEN_SIZE * 2) as u64
        );
        if args.precise_mhc_control_candidate {
            record!(
                "p1_mhc_post_comb_precise_exp_candidate",
                HC_PRECISE_CONTROL_CANDIDATE_KERNEL,
                hc_precise_control(
                    &context,
                    &p1.mixes,
                    &hc_scale_b,
                    &hc_base_b,
                    &p1.post,
                    &p1.comb,
                    hc_mult,
                    mix_width,
                    sinkhorn,
                    hc_eps,
                ),
                (HC_MIX_WIDTH + 3 + HC_MIX_WIDTH) * 4,
                (HC_MULT + HC_MULT * HC_MULT) * 4,
                0
            );
        }
        record!(
            "p1_attn_norm",
            RMS_KERNEL,
            rms(
                &context,
                &p1.reduced,
                &attn_norm_w_b,
                &p1.attn_norm,
                hidden,
                norm_eps
            ),
            HIDDEN_SIZE * 4,
            HIDDEN_SIZE * 2,
            (HIDDEN_SIZE * 4) as u64
        );
        record!(
            "p1_wq_a_qat",
            QAT_KERNEL,
            qat(
                &context,
                &p1.attn_norm,
                &p1_wq_a.activation,
                &p1_wq_a.scales,
                wq_a_cols
            ),
            HIDDEN_SIZE * 2,
            LAYER0_WQ_A_COLS + LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK,
            (HIDDEN_SIZE * 2) as u64
        );
        record!(
            "p1_wq_a_fp8",
            FP8_KERNEL,
            fp8(
                &context,
                &wq_a_w_b,
                &wq_a_s_b,
                &p1_wq_a.activation,
                &p1_wq_a.scales,
                &p1_wq_a.fp32,
                wq_a_rows,
                wq_a_cols,
                wq_a_scale_cols
            ),
            wq_a_weight.len()
                + wq_a_scale.len()
                + LAYER0_WQ_A_COLS
                + LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK,
            LAYER0_WQ_A_ROWS * 4,
            (LAYER0_WQ_A_ROWS * LAYER0_WQ_A_COLS * 2) as u64
        );
        record!(
            "p1_wq_a_cast",
            CAST_KERNEL,
            cast(&context, &p1_wq_a.fp32, &p1_wq_a.bf16, wq_a_rows),
            LAYER0_WQ_A_ROWS * 4,
            LAYER0_WQ_A_ROWS * 2,
            0
        );
        record!(
            "p1_q_norm",
            RMS_KERNEL,
            rms(
                &context,
                &p1_wq_a.bf16,
                &q_norm_w_b,
                &p1_q_norm,
                q_lora,
                norm_eps
            ),
            Q_LORA_RANK * 4,
            Q_LORA_RANK * 2,
            (Q_LORA_RANK * 4) as u64
        );
        record!(
            "p1_wq_b_qat",
            QAT_KERNEL,
            qat(
                &context,
                &p1_q_norm,
                &p1_wq_b.activation,
                &p1_wq_b.scales,
                q_lora
            ),
            Q_LORA_RANK * 2,
            Q_LORA_RANK + Q_LORA_RANK / ACT_QUANT_BLOCK,
            (Q_LORA_RANK * 2) as u64
        );
        record!(
            "p1_wq_b_fp8",
            FP8_KERNEL,
            fp8(
                &context,
                &wq_b_w_b,
                &wq_b_s_b,
                &p1_wq_b.activation,
                &p1_wq_b.scales,
                &p1_wq_b.fp32,
                wq_b_rows,
                q_lora,
                wq_b_scale_cols
            ),
            wq_b_weight.len() + wq_b_scale.len() + Q_LORA_RANK + Q_LORA_RANK / ACT_QUANT_BLOCK,
            WQ_B_ROWS * 4,
            (WQ_B_ROWS * Q_LORA_RANK * 2) as u64
        );
        record!(
            "p1_wq_b_cast",
            CAST_KERNEL,
            cast(&context, &p1_wq_b.fp32, &p1_wq_b.bf16, wq_b_rows),
            WQ_B_ROWS * 4,
            WQ_B_ROWS * 2,
            0
        );
        record!(
            "p1_q_per_head_norm",
            PER_HEAD_KERNEL,
            per_head(
                &context,
                &p1_wq_b.bf16,
                &p1_q_head,
                heads,
                head_dim,
                norm_eps
            ),
            WQ_B_ROWS * 2,
            WQ_B_ROWS * 2,
            (WQ_B_ROWS * 3) as u64
        );
        record!(
            "p1_q_rope",
            ROPE_KERNEL,
            rope(
                &context,
                &p1_q_head,
                &rope_cos_b,
                &rope_sin_b,
                &p1_q_rope,
                heads,
                head_dim,
                rope_dim,
                inverse0
            ),
            WQ_B_ROWS * 2 + ROPE_HEAD_DIM * 8,
            WQ_B_ROWS * 2,
            (WQ_B_ROWS * 4) as u64
        );
        record!(
            "p1_wkv_qat",
            QAT_KERNEL,
            qat(
                &context,
                &p1.attn_norm,
                &p1_wkv.activation,
                &p1_wkv.scales,
                hidden
            ),
            HIDDEN_SIZE * 2,
            HIDDEN_SIZE + HIDDEN_SIZE / ACT_QUANT_BLOCK,
            (HIDDEN_SIZE * 2) as u64
        );
        record!(
            "p1_wkv_fp8",
            FP8_KERNEL,
            fp8(
                &context,
                &wkv_w_b,
                &wkv_s_b,
                &p1_wkv.activation,
                &p1_wkv.scales,
                &p1_wkv.fp32,
                wkv_rows,
                hidden,
                wkv_scale_cols
            ),
            wkv_weight.len() + wkv_scale.len() + HIDDEN_SIZE + HIDDEN_SIZE / ACT_QUANT_BLOCK,
            WKV_ROWS * 4,
            (WKV_ROWS * HIDDEN_SIZE * 2) as u64
        );
        record!(
            "p1_wkv_cast",
            CAST_KERNEL,
            cast(&context, &p1_wkv.fp32, &p1_wkv.bf16, wkv_rows),
            WKV_ROWS * 4,
            WKV_ROWS * 2,
            0
        );
        record!(
            "p1_kv_norm",
            RMS_KERNEL,
            rms(
                &context,
                &p1_wkv.bf16,
                &kv_norm_w_b,
                &p1_kv.norm,
                head_dim,
                norm_eps
            ),
            HEAD_DIM * 4,
            HEAD_DIM * 2,
            (HEAD_DIM * 4) as u64
        );
        record!(
            "p1_kv_qat",
            KV_QAT_KERNEL,
            kv_qat(
                &context,
                &p1_kv.norm,
                &p1_kv.qat,
                &p1_kv.activation,
                &p1_kv.scales,
                head_dim,
                rope_dim,
                kv_block
            ),
            HEAD_DIM * 2,
            HEAD_DIM * 2 + NON_ROPE_HEAD_DIM + NON_ROPE_HEAD_DIM / KV_QAT_BLOCK,
            (NON_ROPE_HEAD_DIM * 2) as u64
        );
        record!(
            "p1_kv_rope",
            ROPE_KERNEL,
            rope(
                &context,
                &p1_kv.qat,
                &rope_cos_b,
                &rope_sin_b,
                &p1_kv_rope,
                1,
                head_dim,
                rope_dim,
                inverse0
            ),
            HEAD_DIM * 2 + ROPE_HEAD_DIM * 8,
            HEAD_DIM * 2,
            (HEAD_DIM * 4) as u64
        );
        record!(
            "p1_kv_cache_write",
            CACHE_KERNEL,
            cache_write(
                &context,
                &p1_kv_rope,
                &kv_cache,
                position1,
                head_dim,
                cache_capacity
            ),
            HEAD_DIM * 2,
            HEAD_DIM * 2,
            0
        );
        record!(
            "p1_sparse_two_kv_sink",
            SPARSE_KERNEL,
            sparse(
                &context,
                &p1_q_rope,
                &kv_cache,
                &sink_b,
                &p1_sparse,
                &p1_scores,
                &p1_denoms,
                heads,
                head_dim,
                cache_capacity,
                sparse_scale
            ),
            WQ_B_ROWS * 2 + 2 * HEAD_DIM * 2 + NUM_HEADS * 4,
            WQ_B_ROWS * 2 + NUM_HEADS * 12,
            (NUM_HEADS * 2 * HEAD_DIM * 2) as u64
        );
        record!(
            "p1_attention_inverse_rope",
            ROPE_KERNEL,
            rope(
                &context,
                &p1_sparse,
                &rope_cos_b,
                &rope_sin_b,
                &p1_derotated,
                heads,
                head_dim,
                rope_dim,
                inverse1
            ),
            WQ_B_ROWS * 2 + ROPE_HEAD_DIM * 8,
            WQ_B_ROWS * 2,
            (WQ_B_ROWS * 4) as u64
        );
        record!(
            "p1_wo_a_convert_einsum",
            WO_A_KERNEL,
            wo_a(
                &context,
                &wo_a_w_b,
                &wo_a_s_b,
                &p1_derotated,
                &p1_wo_a,
                wo_a_rows,
                wo_a_cols,
                wo_a_scale_cols,
                o_rank
            ),
            wo_a_weight.len() + wo_a_scale.len() + WQ_B_ROWS * 2,
            WO_A_ROWS * 2,
            (WO_A_ROWS * WO_A_COLS * 2) as u64
        );
        record!(
            "p1_wo_b_qat",
            QAT_KERNEL,
            qat(
                &context,
                &p1_wo_a,
                &p1_wo_b.activation,
                &p1_wo_b.scales,
                wo_b_cols
            ),
            WO_B_COLS * 2,
            WO_B_COLS + WO_B_COLS / ACT_QUANT_BLOCK,
            (WO_B_COLS * 2) as u64
        );
        record!(
            "p1_wo_b_fp8",
            FP8_KERNEL,
            fp8(
                &context,
                &wo_b_w_b,
                &wo_b_s_b,
                &p1_wo_b.activation,
                &p1_wo_b.scales,
                &p1_wo_b.fp32,
                wo_b_rows,
                wo_b_cols,
                wo_b_scale_cols
            ),
            wo_b_weight.len() + wo_b_scale.len() + WO_B_COLS + WO_B_COLS / ACT_QUANT_BLOCK,
            WO_B_ROWS * 4,
            (WO_B_ROWS * WO_B_COLS * 2) as u64
        );
        record!(
            "p1_wo_b_cast",
            CAST_KERNEL,
            cast(&context, &p1_wo_b.fp32, &p1_wo_b.bf16, wo_b_rows),
            WO_B_ROWS * 4,
            WO_B_ROWS * 2,
            0
        );
        record!(
            "p1_mhc_attention_post",
            HC_POST_KERNEL,
            hc_post(
                &context,
                &p1_wo_b.bf16,
                &p1.embed,
                &p1.post,
                &p1.comb,
                &p1_hc_final,
                hidden,
                hc_mult
            ),
            WO_B_ROWS * 2 + embed1.len() + HC_MULT * 4 + HC_MULT * HC_MULT * 4,
            HC_MULT * HIDDEN_SIZE * 2,
            (HC_MULT * HIDDEN_SIZE * 5) as u64
        );
        if let Some(strict_candidate) = strict_mhc_trace_candidate.as_ref() {
            strict_candidate.overwrite_controls(
                &context,
                &p1.mixes,
                &hc_scale_b,
                &hc_base_b,
                &p1.post,
                &p1.comb,
                hc_mult,
                mix_width,
                sinkhorn,
                hc_eps,
            )?;
            // Recompute the terminal store with the strict, source-trace
            // control candidate. This is unsealed diagnostic output only.
            record!(
                "p1_mhc_attention_post_trace_specific_exact_candidate",
                HC_POST_KERNEL,
                hc_post(
                    &context,
                    &p1_wo_b.bf16,
                    &p1.embed,
                    &p1.post,
                    &p1.comb,
                    &p1_hc_final,
                    hidden,
                    hc_mult
                ),
                WO_B_ROWS * 2 + embed1.len() + HC_MULT * 4 + HC_MULT * HC_MULT * 4,
                HC_MULT * HIDDEN_SIZE * 2,
                (HC_MULT * HIDDEN_SIZE * 5) as u64
            );
        }
        let mut strict_dd_direct_timing = None;
        if let Some(strict_candidate) = strict_mhc_dd_candidate.as_ref() {
            strict_dd_direct_timing = Some(strict_candidate.overwrite_controls(
                &context,
                &strict_dd_direct_trace_label,
                &p1.mixes,
                &hc_scale_b,
                &hc_base_b,
                &p1.post,
                &p1.comb,
                hc_mult,
                mix_width,
                sinkhorn,
                hc_eps,
            )?);
            // Recompute the terminal store with the general, strict
            // double-double Darwin-exp path.  It contains no source-trace
            // keys or ULP repair; the legacy flag reports an unsealed
            // candidate while the explicit v2 flag creates a fresh receipt.
            record!(
                "p1_mhc_attention_post_general_darwin_dd_candidate",
                HC_POST_KERNEL,
                hc_post(
                    &context,
                    &p1_wo_b.bf16,
                    &p1.embed,
                    &p1.post,
                    &p1.comb,
                    &p1_hc_final,
                    hidden,
                    hc_mult
                ),
                WO_B_ROWS * 2 + embed1.len() + HC_MULT * 4 + HC_MULT * HC_MULT * 4,
                HC_MULT * HIDDEN_SIZE * 2,
                (HC_MULT * HIDDEN_SIZE * 5) as u64
            );
        }

        let counts = trace.counts();
        drop(trace);
        // This boundary is deliberately only the GPU-dispatch/physical-trace
        // completion point.  Readbacks and receipt validation still follow.
        let strict_dd_v2_dispatch_trace_completed_unix_ns = if args.seal_strict_mhc_control_dd_v2 {
            Some(unix_time_ns()?)
        } else {
            None
        };
        let (buffers_created, bytes_allocated, commits) = context.drain_stats();
        let trace_samples = context.drain_trace();
        let expected_dispatches = P4B_DISPATCHES
            + u64::from(
                args.precise_mhc_control_candidate
                    || args.strict_mhc_control_trace_candidate
                    || strict_mhc_dd_requested,
            );
        if counts.command_count != expected_dispatches
            || counts.encoder_count != expected_dispatches
            || commits as u64 != expected_dispatches
            || trace_samples.len() as u64 != expected_dispatches
            || stages.len() as u64 != expected_dispatches
        {
            return Err(failure("P4B physical command accounting is incomplete"));
        }

        // Host reads occur only after all P0/P1 device dependencies complete.
        let gpu0_hc = u16read(&p0.reduced, HIDDEN_SIZE)?;
        let gpu0_norm = u16read(&p0.attn_norm, HIDDEN_SIZE)?;
        let gpu0_wkv_qat = bytesread(&p0_wkv.activation, HIDDEN_SIZE)?;
        let gpu0_wkv_scales = bytesread(&p0_wkv.scales, HIDDEN_SIZE / ACT_QUANT_BLOCK)?;
        let gpu0_wkv = f32read(&p0_wkv.fp32, WKV_ROWS)?;
        let gpu0_wkv_bf16 = u16read(&p0_wkv.bf16, WKV_ROWS)?;
        let gpu0_kv_norm = u16read(&p0_kv.norm, HEAD_DIM)?;
        let gpu0_kv_qat = u16read(&p0_kv.qat, HEAD_DIM)?;
        let gpu0_kv_act = bytesread(&p0_kv.activation, NON_ROPE_HEAD_DIM)?;
        let gpu0_kv_scales = bytesread(&p0_kv.scales, NON_ROPE_HEAD_DIM / KV_QAT_BLOCK)?;
        let gpu1_hc = u16read(&p1.reduced, HIDDEN_SIZE)?;
        let gpu1_norm = u16read(&p1.attn_norm, HIDDEN_SIZE)?;
        let gpu_wqa_qat = bytesread(&p1_wq_a.activation, LAYER0_WQ_A_COLS)?;
        let gpu_wqa_scales = bytesread(&p1_wq_a.scales, LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK)?;
        let gpu_wqa = f32read(&p1_wq_a.fp32, LAYER0_WQ_A_ROWS)?;
        let gpu_wqa_bf16 = u16read(&p1_wq_a.bf16, LAYER0_WQ_A_ROWS)?;
        let gpu_qnorm = u16read(&p1_q_norm, Q_LORA_RANK)?;
        let gpu_wqb_qat = bytesread(&p1_wq_b.activation, Q_LORA_RANK)?;
        let gpu_wqb_scales = bytesread(&p1_wq_b.scales, Q_LORA_RANK / ACT_QUANT_BLOCK)?;
        let gpu_wqb = f32read(&p1_wq_b.fp32, WQ_B_ROWS)?;
        let gpu_wqb_bf16 = u16read(&p1_wq_b.bf16, WQ_B_ROWS)?;
        let gpu_qhead = u16read(&p1_q_head, WQ_B_ROWS)?;
        let gpu_qrope = u16read(&p1_q_rope, WQ_B_ROWS)?;
        let gpu_wkv_qat = bytesread(&p1_wkv.activation, HIDDEN_SIZE)?;
        let gpu_wkv_scales = bytesread(&p1_wkv.scales, HIDDEN_SIZE / ACT_QUANT_BLOCK)?;
        let gpu_wkv = f32read(&p1_wkv.fp32, WKV_ROWS)?;
        let gpu_wkv_bf16 = u16read(&p1_wkv.bf16, WKV_ROWS)?;
        let gpu_kvnorm = u16read(&p1_kv.norm, HEAD_DIM)?;
        let gpu_kvqat = u16read(&p1_kv.qat, HEAD_DIM)?;
        let gpu_kv_act = bytesread(&p1_kv.activation, NON_ROPE_HEAD_DIM)?;
        let gpu_kv_scales = bytesread(&p1_kv.scales, NON_ROPE_HEAD_DIM / KV_QAT_BLOCK)?;
        let gpu_kvrope = u16read(&p1_kv_rope, HEAD_DIM)?;
        let gpu_cache = u16read(&kv_cache, POSITION1_KV_ROWS * HEAD_DIM)?;
        let gpu_scores = f32read(&p1_scores, NUM_HEADS * POSITION1_KV_ROWS)?;
        let gpu_denoms = f32read(&p1_denoms, NUM_HEADS)?;
        let gpu_sparse = u16read(&p1_sparse, WQ_B_ROWS)?;
        let gpu_derotated = u16read(&p1_derotated, WQ_B_ROWS)?;
        let gpu_wo_a = u16read(&p1_wo_a, WO_A_ROWS)?;
        let gpu_wo_b_qat = bytesread(&p1_wo_b.activation, WO_B_COLS)?;
        let gpu_wo_b_scales = bytesread(&p1_wo_b.scales, WO_B_COLS / ACT_QUANT_BLOCK)?;
        let gpu_wo_b = f32read(&p1_wo_b.fp32, WO_B_ROWS)?;
        let gpu_wo_b_bf16 = u16read(&p1_wo_b.bf16, WO_B_ROWS)?;
        let gpu_p1_mixes = f32read(&p1.mixes, HC_MIX_WIDTH)?;
        let gpu_p1_post = f32read(&p1.post, HC_MULT)?;
        let gpu_p1_comb = f32read(&p1.comb, HC_MULT * HC_MULT)?;
        let gpu_hc_final = u16read(&p1_hc_final, HC_MULT * HIDDEN_SIZE)?;

        // Exact source storage checks for every discrete P0/P1 handoff.
        exact16("P0 mHC reduced", &prefix0.hc_attn_pre_bf16_bits, &gpu0_hc)?;
        exact16(
            "P0 attention norm",
            &prefix0.attn_norm_bf16_bits,
            &gpu0_norm,
        )?;
        exact8(
            "P0 WKV QAT",
            &cpu.causal.position0_wkv.quantized_input.activation_e4m3fn,
            &gpu0_wkv_qat,
        )?;
        exact8(
            "P0 WKV scales",
            &cpu.causal.position0_wkv.quantized_input.scales_e8m0fnu,
            &gpu0_wkv_scales,
        )?;
        close(
            "P0 WKV FP32",
            &cpu.causal.position0_wkv.output.fp32,
            &gpu0_wkv,
        )?;
        exact16(
            "P0 WKV BF16",
            &cpu.causal.position0_wkv.output.bf16_bits,
            &gpu0_wkv_bf16,
        )?;
        exact16(
            "P0 KV norm",
            &cpu.causal.position0_kv_norm_bf16_bits,
            &gpu0_kv_norm,
        )?;
        exact16(
            "P0 KV QAT",
            &cpu.causal.position0_kv_inplace_qat.output_bf16_bits,
            &gpu0_kv_qat,
        )?;
        exact8(
            "P0 KV QAT activation",
            &cpu.causal
                .position0_kv_inplace_qat
                .non_rope_activation_e4m3fn,
            &gpu0_kv_act,
        )?;
        exact8(
            "P0 KV QAT scales",
            &cpu.causal.position0_kv_inplace_qat.non_rope_scales_e8m0fnu,
            &gpu0_kv_scales,
        )?;
        exact16(
            "P1 mHC reduced",
            &cpu.causal.token1_prefix.hc_attn_pre_bf16_bits,
            &gpu1_hc,
        )?;
        exact16(
            "P1 attention norm",
            &cpu.causal.token1_prefix.attn_norm_bf16_bits,
            &gpu1_norm,
        )?;
        exact8(
            "P1 WQ-A QAT",
            &cpu.causal.wq_a.quantized_input.activation_e4m3fn,
            &gpu_wqa_qat,
        )?;
        exact8(
            "P1 WQ-A scales",
            &cpu.causal.wq_a.quantized_input.scales_e8m0fnu,
            &gpu_wqa_scales,
        )?;
        close("P1 WQ-A FP32", &cpu.causal.wq_a.output.fp32, &gpu_wqa)?;
        exact16(
            "P1 WQ-A BF16",
            &cpu.causal.wq_a.output.bf16_bits,
            &gpu_wqa_bf16,
        )?;
        exact16("P1 Q norm", &cpu.causal.q_norm_bf16_bits, &gpu_qnorm)?;
        exact8(
            "P1 WQ-B QAT",
            &cpu.causal.wq_b.quantized_input.activation_e4m3fn,
            &gpu_wqb_qat,
        )?;
        exact8(
            "P1 WQ-B scales",
            &cpu.causal.wq_b.quantized_input.scales_e8m0fnu,
            &gpu_wqb_scales,
        )?;
        close("P1 WQ-B FP32", &cpu.causal.wq_b.output.fp32, &gpu_wqb)?;
        exact16(
            "P1 WQ-B BF16",
            &cpu.causal.wq_b.output.bf16_bits,
            &gpu_wqb_bf16,
        )?;
        exact16(
            "P1 per-head Q norm",
            &cpu.causal.q_head_norm_bf16_bits,
            &gpu_qhead,
        )?;
        exact16(
            "P1 Q RoPE",
            &cpu.causal.q_position1_rope_bf16_bits,
            &gpu_qrope,
        )?;
        exact8(
            "P1 WKV QAT",
            &cpu.causal.wkv.quantized_input.activation_e4m3fn,
            &gpu_wkv_qat,
        )?;
        exact8(
            "P1 WKV scales",
            &cpu.causal.wkv.quantized_input.scales_e8m0fnu,
            &gpu_wkv_scales,
        )?;
        close("P1 WKV FP32", &cpu.causal.wkv.output.fp32, &gpu_wkv)?;
        exact16(
            "P1 WKV BF16",
            &cpu.causal.wkv.output.bf16_bits,
            &gpu_wkv_bf16,
        )?;
        exact16("P1 KV norm", &cpu.causal.kv_norm_bf16_bits, &gpu_kvnorm)?;
        exact16(
            "P1 KV QAT",
            &cpu.causal.kv_inplace_qat.output_bf16_bits,
            &gpu_kvqat,
        )?;
        exact8(
            "P1 KV QAT activation",
            &cpu.causal.kv_inplace_qat.non_rope_activation_e4m3fn,
            &gpu_kv_act,
        )?;
        exact8(
            "P1 KV QAT scales",
            &cpu.causal.kv_inplace_qat.non_rope_scales_e8m0fnu,
            &gpu_kv_scales,
        )?;
        exact16(
            "P1 KV RoPE",
            &cpu.causal.kv_position1_rope_bf16_bits,
            &gpu_kvrope,
        )?;
        exact16(
            "two-row device KV cache",
            &cpu.causal.kv_cache_two_rows_bf16_bits,
            &gpu_cache,
        )?;
        close(
            "P1 sparse scores",
            &cpu.causal.sparse_attention_scores_f32,
            &gpu_scores,
        )?;
        close(
            "P1 sparse denominators",
            &cpu.causal.sparse_attention_sink_denominators_f32,
            &gpu_denoms,
        )?;
        exact16(
            "P1 sparse BF16",
            &cpu.causal.sparse_attention_bf16_bits,
            &gpu_sparse,
        )?;
        exact16(
            "P1 inverse RoPE",
            &cpu.causal.sparse_attention_derotated_bf16_bits,
            &gpu_derotated,
        )?;
        exact16("P1 WO-A", &cpu.wo_a_bf16_bits, &gpu_wo_a)?;
        exact8(
            "P1 WO-B QAT",
            &cpu.wo_b.quantized_input.activation_e4m3fn,
            &gpu_wo_b_qat,
        )?;
        exact8(
            "P1 WO-B scales",
            &cpu.wo_b.quantized_input.scales_e8m0fnu,
            &gpu_wo_b_scales,
        )?;
        close("P1 WO-B FP32", &cpu.wo_b.output.fp32, &gpu_wo_b)?;
        exact16("P1 WO-B BF16", &cpu.wo_b.output.bf16_bits, &gpu_wo_b_bf16)?;
        let p1_hc_post_storage_delta =
            bf16_difference(&cpu.hc_attention_post_bf16_bits, &gpu_hc_final)?;
        let p1_hc_post_control_delta =
            f32_difference(&cpu.causal.token1_prefix.hc_post_f32, &gpu_p1_post)?;
        let p1_hc_mixes_control_delta =
            f32_difference(&cpu.causal.token1_prefix.hc_mixes_f32, &gpu_p1_mixes)?;
        let p1_hc_comb_control_delta =
            f32_difference(&cpu.causal.token1_prefix.hc_comb_f32, &gpu_p1_comb)?;

        // The candidate is a deliberately unsealed diagnostic.  It must not
        // overwrite or mint a P4B authority/baseline receipt.  Its only
        // promotion predicate is bitwise equality for the exact device input
        // mixes, both control tensors, and terminal P1 mHC BF16 storage.
        if args.precise_mhc_control_candidate {
            let mixes_exact = zero_count(&p1_hc_mixes_control_delta, "bitwise_mismatch_count");
            let post_exact = zero_count(&p1_hc_post_control_delta, "bitwise_mismatch_count");
            let comb_exact = zero_count(&p1_hc_comb_control_delta, "bitwise_mismatch_count");
            let terminal_exact = zero_count(&p1_hc_post_storage_delta, "mismatch_count");
            let candidate_pass = mixes_exact && post_exact && comb_exact && terminal_exact;
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "schema":"hawking.gravity.deepseek_v4.p4b_precise_exp_mhc_control_candidate.v1",
                    "status":if candidate_pass {"PASS_EXACT_STORAGE_CANDIDATE_NOT_PROMOTED"} else {"FAIL_EXACT_STORAGE_CANDIDATE_NOT_PROMOTED"},
                    "candidate_kernel":HC_PRECISE_CONTROL_CANDIDATE_KERNEL,
                    "artifact_manifest_seal_sha256":reader.manifest_seal_sha256(),
                    "independent_cpu_oracle":&cpu_binding.seal_sha256,
                    "dispatches":expected_dispatches,
                    "fallback":false,
                    "mixes_control_delta":&p1_hc_mixes_control_delta,
                    "post_control_delta":&p1_hc_post_control_delta,
                    "comb_control_delta":&p1_hc_comb_control_delta,
                    "terminal_bf16_storage_delta":&p1_hc_post_storage_delta,
                    "receipt_written":false,
                    "promotion":false,
                    "authority_kernel_unchanged":true,
                }))?
            );
            if !candidate_pass {
                return Err(failure(
                    "precise-exp mHC-control candidate did not restore exact P1 controls/terminal storage",
                ));
            }
            return Ok(());
        }

        if args.strict_mhc_control_trace_candidate {
            let mixes_exact = zero_count(&p1_hc_mixes_control_delta, "bitwise_mismatch_count");
            let post_exact = zero_count(&p1_hc_post_control_delta, "bitwise_mismatch_count");
            let comb_exact = zero_count(&p1_hc_comb_control_delta, "bitwise_mismatch_count");
            let terminal_exact = zero_count(&p1_hc_post_storage_delta, "mismatch_count");
            let candidate_pass = mixes_exact && post_exact && comb_exact && terminal_exact;
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "schema":"hawking.gravity.deepseek_v4.p4b_strict_mhc_control_trace_candidate.v1",
                    "status":if candidate_pass {"TRACE_SPECIFIC_EXACT_CANDIDATE"} else {"TRACE_SPECIFIC_NOT_EXACT_CANDIDATE"},
                    "artifact_manifest_seal_sha256":reader.manifest_seal_sha256(),
                    "independent_cpu_oracle":&cpu_binding.seal_sha256,
                    "strict_control_kernel":HC_PRECISE_CONTROL_CANDIDATE_KERNEL,
                    "strict_control_library_fast_math_enabled":false,
                    "trace_bound_repair_kernel":HC_POST_ULP_REPAIR_TRACE_CANDIDATE_KERNEL,
                    "trace_bound_repair_logit_bits":["0xc05496db","0xc188c8ca"],
                    "control_strategy":"strict-math P4B post+comb followed by a two-fixed-input ULP repair",
                    "principled_cpu_exp_compatibility_implementation":false,
                    "compatibility_assessment":"fixed-input source-trace patch; it is not a general CPU exp emulation and cannot be considered for broader runtime use",
                    "mixes_control_delta":&p1_hc_mixes_control_delta,
                    "post_control_delta":&p1_hc_post_control_delta,
                    "comb_control_delta":&p1_hc_comb_control_delta,
                    "terminal_bf16_storage_delta":&p1_hc_post_storage_delta,
                    "real_gpu_dispatches":expected_dispatches + 2,
                    "real_command_buffers":expected_dispatches + 1,
                    "physical_trace_context_dispatches":expected_dispatches,
                    "fallback":false,
                    "receipt_written":false,
                    "promotion":false,
                    "reusable_p4b_executor_reclassified":false,
                    "authority_kernel_unchanged":true,
                }))?
            );
            if !candidate_pass {
                return Err(failure(
                    "strict trace-specific mHC control candidate did not restore exact P1 controls/terminal storage",
                ));
            }
            return Ok(());
        }

        let fb = f32_bounds();
        let bb = bf16_bounds();
        let scores = [
            (
                "p1_q_rope_bf16",
                score(
                    "P1 Q RoPE",
                    &bf16f32(&cpu.causal.q_position1_rope_bf16_bits),
                    &bf16f32(&gpu_qrope),
                    &rope_q_f64,
                    &bb,
                )?,
            ),
            (
                "p1_kv_rope_bf16",
                score(
                    "P1 KV RoPE",
                    &bf16f32(&cpu.causal.kv_position1_rope_bf16_bits),
                    &bf16f32(&gpu_kvrope),
                    &rope_kv_f64,
                    &bb,
                )?,
            ),
            (
                "p1_sparse_scores_f32",
                score(
                    "P1 sparse scores",
                    &cpu.causal.sparse_attention_scores_f32,
                    &gpu_scores,
                    &sparse_f64.scores,
                    &fb,
                )?,
            ),
            (
                "p1_sparse_denominators_f32",
                score(
                    "P1 sparse denominators",
                    &cpu.causal.sparse_attention_sink_denominators_f32,
                    &gpu_denoms,
                    &sparse_f64.denominators,
                    &fb,
                )?,
            ),
            (
                "p1_sparse_bf16",
                score(
                    "P1 sparse output",
                    &bf16f32(&cpu.causal.sparse_attention_bf16_bits),
                    &bf16f32(&gpu_sparse),
                    &sparse_f64.attention,
                    &bb,
                )?,
            ),
            (
                "p1_inverse_rope_bf16",
                score(
                    "P1 inverse RoPE",
                    &bf16f32(&cpu.causal.sparse_attention_derotated_bf16_bits),
                    &bf16f32(&gpu_derotated),
                    &sparse_f64.derotated,
                    &bb,
                )?,
            ),
            (
                "p1_wo_a_bf16",
                score(
                    "P1 WO-A",
                    &bf16f32(&cpu.wo_a_bf16_bits),
                    &bf16f32(&gpu_wo_a),
                    &wo_a_f64,
                    &bb,
                )?,
            ),
            (
                "p1_wo_b_f32",
                score("P1 WO-B", &cpu.wo_b.output.fp32, &gpu_wo_b, &wo_b_f64, &fb)?,
            ),
            (
                "p1_mhc_post_control_f32",
                score(
                    "P1 mHC post control",
                    &cpu.causal.token1_prefix.hc_post_f32,
                    &gpu_p1_post,
                    &hc_controls_f64.post,
                    &fb,
                )?,
            ),
            (
                "p1_mhc_comb_control_f32",
                score(
                    "P1 mHC comb control",
                    &cpu.causal.token1_prefix.hc_comb_f32,
                    &gpu_p1_comb,
                    &hc_controls_f64.comb,
                    &fb,
                )?,
            ),
            (
                "p1_mhc_attention_post_bf16",
                score(
                    "P1 mHC post",
                    &bf16f32(&cpu.hc_attention_post_bf16_bits),
                    &bf16f32(&gpu_hc_final),
                    &hc_post_f64,
                    &bb,
                )?,
            ),
        ];
        let all_scores_pass = scores.iter().all(|(_, score)| score.pass);
        let mut score_json = serde_json::Map::new();
        for (name, value) in scores {
            score_json.insert(
                name.to_owned(),
                decimal_strings(serde_json::to_value(value)?),
            );
        }
        let stage_profiles = Value::Array(
            stages
                .iter()
                .map(|(stage, kernel, t, read, written, flops)| {
                    profile(stage, kernel, t, *read, *written, *flops)
                })
                .collect(),
        );
        if args.strict_mhc_control_dd_candidate {
            let mixes_exact = zero_count(&p1_hc_mixes_control_delta, "bitwise_mismatch_count");
            let post_exact = zero_count(&p1_hc_post_control_delta, "bitwise_mismatch_count");
            let comb_exact = zero_count(&p1_hc_comb_control_delta, "bitwise_mismatch_count");
            let terminal_exact = zero_count(&p1_hc_post_storage_delta, "mismatch_count");
            let candidate_pass =
                mixes_exact && post_exact && comb_exact && terminal_exact && all_scores_pass;
            let candidate = strict_mhc_dd_candidate.as_ref().ok_or_else(|| {
                failure("general Darwin-DD candidate requested without a prepared candidate")
            })?;
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "schema":"hawking.gravity.deepseek_v4.p4b_general_darwin_dd_mhc_control_candidate.v1",
                    "status":if candidate_pass {"PASS_EXACT_FULL_P4B_CANDIDATE_NOT_PROMOTED"} else {"FAIL_FULL_P4B_CANDIDATE_NOT_PROMOTED"},
                    "artifact_manifest_seal_sha256":reader.manifest_seal_sha256(),
                    "source_identity":{"repository":reader.source_identity().repository,"revision":reader.source_identity().revision},
                    "independent_cpu_oracle":&cpu_binding.seal_sha256,
                    "p4a_authority":&p4a_binding.seal_sha256,
                    "p4a_topology":&topology_binding.seal_sha256,
                    "candidate_kernel":HC_DARWIN_DD_CONTROL_CANDIDATE_KERNEL,
                    "strict_control_library_fast_math_enabled":false,
                    "control_strategy":"general finite-F32 [-40,40] strict double-double reconstruction of the active Darwin expf normal path; source sigmoid and 20-iteration Sinkhorn loop/order retained",
                    "table_contract":{"entries":DARWIN_EXPF_TABLE.len(),"layout":"128 x float4 [high,middle,low,pad]","bytes":DARWIN_EXPF_TABLE.len()*std::mem::size_of::<[f32;4]>(),"sha256":candidate.table_sha256},
                    "trace_or_logit_keys":false,
                    "ulp_repair":false,
                    "all_source_storage_handoffs_checked_before_candidate_report":true,
                    "all_numeric_parity_v2_1_scores_pass":all_scores_pass,
                    "numeric_parity_v2_1_scores":Value::Object(score_json.clone()),
                    "mixes_control_delta":&p1_hc_mixes_control_delta,
                    "post_control_delta":&p1_hc_post_control_delta,
                    "comb_control_delta":&p1_hc_comb_control_delta,
                    "terminal_bf16_storage_delta":&p1_hc_post_storage_delta,
                    "complete_position1_attention_stage_profile":stage_profiles.clone(),
                    "direct_candidate_execution":strict_dd_direct_timing.map(strict_dd_direct_profile),
                    "real_gpu_dispatches":expected_dispatches + 1,
                    "real_command_buffers":expected_dispatches + 1,
                    "physical_trace_context_dispatches":expected_dispatches,
                    "direct_candidate_dispatches_not_in_context_trace":1,
                    "fallback":false,
                    "receipt_written":false,
                    "promotion":false,
                    "authority_kernel_unchanged":true,
                    "reusable_p4b_executor_reclassified":false,
                    "claim_boundary":"Unsealed full P4B P0->P1 strict candidate only; it establishes neither P7/runtime, token generation, HCLI, nor TPS."
                }))?
            );
            if !candidate_pass {
                return Err(failure(
                    "general Darwin-DD mHC control candidate did not restore all scored P4B boundaries and exact terminal storage",
                ));
            }
            return Ok(());
        }
        if args.seal_strict_mhc_control_dd_v2 {
            let mixes_exact = zero_count(&p1_hc_mixes_control_delta, "bitwise_mismatch_count");
            let post_exact = zero_count(&p1_hc_post_control_delta, "bitwise_mismatch_count");
            let comb_exact = zero_count(&p1_hc_comb_control_delta, "bitwise_mismatch_count");
            let terminal_exact = zero_count(&p1_hc_post_storage_delta, "mismatch_count");
            let candidate_pass =
                mixes_exact && post_exact && comb_exact && terminal_exact && all_scores_pass;
            if !candidate_pass {
                return Err(failure(
                    "strict-DD v2 sealing refused: exact P1 controls, terminal storage, or V2.1 score gate failed",
                ));
            }
            if expected_dispatches != P4B_DISPATCHES + 1
                || counts.command_count != expected_dispatches
                || counts.encoder_count != expected_dispatches
                || commits as u64 != expected_dispatches
                || trace_samples.len() as u64 != expected_dispatches
            {
                return Err(failure(
                    "strict-DD v2 sealing refused: registered-context physical accounting changed",
                ));
            }
            let candidate = strict_mhc_dd_candidate.as_ref().ok_or_else(|| {
                failure("strict-DD v2 requested without a prepared dynamic strict-DD candidate")
            })?;
            let direct_timing = strict_dd_direct_timing.ok_or_else(|| {
                failure(
                    "strict-DD v2 requested without a completed direct dynamic-library dispatch",
                )
            })?;
            let direct_profile = strict_dd_direct_profile(direct_timing);
            let mut complete_stage_profiles = stage_profiles.clone();
            let complete_stages = complete_stage_profiles
                .as_array_mut()
                .ok_or_else(|| failure("strict-DD v2 stage profile is not an array"))?;
            complete_stages.push(direct_profile.clone());
            if complete_stages.len() as u64 != expected_dispatches + 1 {
                return Err(failure(
                    "strict-DD v2 sealing refused: complete stage profile does not cover every dispatch",
                ));
            }
            let run_provenance = strict_dd_v2_run.as_ref().ok_or_else(|| {
                failure("strict-DD v2 receipt path has no pre-dispatch run provenance")
            })?;
            let dispatch_trace_completed_unix_ns = strict_dd_v2_dispatch_trace_completed_unix_ns
                .as_deref()
                .ok_or_else(|| {
                    failure("strict-DD v2 receipt path has no dispatch/trace completion timestamp")
                })?;
            ensure_unix_ns_not_before(
                &run_provenance.run_started_unix_ns,
                dispatch_trace_completed_unix_ns,
            )?;
            let out = args
                .out
                .as_ref()
                .ok_or_else(|| failure("--out is required for strict-DD v2 sealing"))?;
            if out.file_name().and_then(|name| name.to_str()) != Some(STRICT_DD_V2_BASENAME) {
                return Err(failure(format!(
                    "strict-DD v2 output must use the create-new basename {STRICT_DD_V2_BASENAME}"
                )));
            }
            // Resolve every fallible receipt input before recording the receipt
            // validation boundary.  No device readback, numeric evaluation, or
            // source/artifact receipt input is deferred past this point.
            let source_metadata_asset_sha256 = json!({
                "inference/model.py":reader.source_metadata_asset_sha256("inference/model.py")?,
                "inference/kernel.py":reader.source_metadata_asset_sha256("inference/kernel.py")?,
                "inference/convert.py":reader.source_metadata_asset_sha256("inference/convert.py")?,
                "inference/config.json":reader.source_metadata_asset_sha256("inference/config.json")?,
                "config.json":reader.source_metadata_asset_sha256("config.json")?
            });
            let f32_operator_bounds = decimal_strings(serde_json::to_value(&fb)?);
            let bf16_storage_bounds = decimal_strings(serde_json::to_value(&bb)?);
            let dynamic_library_source_sha256 =
                sha256(hawking_core::metal::SHADER_MATMUL.as_bytes());
            let direct_trace_label_sha256 = sha256(strict_dd_direct_trace_label.as_bytes());
            let receipt_validation_finished_unix_ns = unix_time_ns()?;
            ensure_unix_ns_not_before(
                dispatch_trace_completed_unix_ns,
                &receipt_validation_finished_unix_ns,
            )?;
            let unsigned = json!({
                "schema":STRICT_DD_V2_RECEIPT_SCHEMA,
                "status":STRICT_DD_V2_RECEIPT_STATUS,
                "scope":{
                    "execution":"one fresh strict-DD real-Metal bounded P0->P1 layer-0 attention run",
                    "token_ids":[PREFIX_TOKEN_ID,POSITION1_TOKEN_ID],
                    "batch":1,
                    "sequence_tokens":2,
                    "position":POSITION1,
                    "compress_ratio":0,
                    "window_size":WINDOW_SIZE,
                    "device_resident_two_row_kv_cache":true,
                    "source_window_topk_indices":[0,1],
                    "complete_position1_attention_tail":true,
                    "not_layer_ffn":true,
                    "not_router_or_moe":true,
                    "not_p7":true,
                    "not_runtime":true,
                    "not_token_generation":true,
                    "not_hcli":true,
                    "not_base_true_tps":true,
                    "not_a_replacement_for_p4b_v1":true
                },
                "artifact":{
                    "path":reader.artifact_root().display().to_string(),
                    "full_stream_schema":FULL_STREAM_SCHEMA,
                    "full_stream_status":FULL_STREAM_STATUS,
                    "manifest_file_sha256":reader.manifest_file_sha256(),
                    "manifest_seal_sha256":reader.manifest_seal_sha256(),
                    "restart_receipt_seal_sha256":reader.restart_seal_sha256(),
                    "source_parent_retained":false,
                    "all_touched_chunks_sha256_verified_before_gpu_upload":true,
                    "parent_safetensors_materialized":false
                },
                "source":{
                    "repository":reader.source_identity().repository,
                    "revision":reader.source_identity().revision,
                    "source_metadata_asset_sha256":source_metadata_asset_sha256,
                    "raw_artifact_inputs":{
                        "embedding_rows":[PREFIX_TOKEN_ID,POSITION1_TOKEN_ID],
                        "weights":[LAYER0_WQ_A_WEIGHT,LAYER0_WQ_B_WEIGHT,LAYER0_WKV_WEIGHT,LAYER0_WO_A_WEIGHT,LAYER0_WO_B_WEIGHT],
                        "reused_exact_candidate_input_path":true
                    }
                },
                "predecessors":{
                    "position1_complete_cpu_oracle":binding_json(&cpu_binding),
                    "p4a_complete_attention_authority":binding_json(&p4a_binding),
                    "p4a_one_cb_topology_win":binding_json(&topology_binding)
                },
                "execution_provenance":{
                    "source_code_inspection":&run_provenance.source_code_inspection,
                    "dynamic_library":{
                        "source":"hawking_core::metal::SHADER_MATMUL",
                        "source_sha256":dynamic_library_source_sha256,
                        "function":HC_DARWIN_DD_CONTROL_CANDIDATE_KERNEL,
                        "fast_math_enabled":false,
                        "pipeline_precompiled_before_dispatch":true,
                        "native_fp64_used":false,
                        "representation":"strict F32 double-double"
                    }
                },
                "run_provenance":{
                    "run_nonce_sha256":&run_provenance.run_nonce,
                    "run_started_unix_ns_text":&run_provenance.run_started_unix_ns,
                    "dispatch_trace_completed_unix_ns":dispatch_trace_completed_unix_ns,
                    "receipt_validation_finished_unix_ns":&receipt_validation_finished_unix_ns,
                    "timestamp_contract":"dispatch_trace_completed is captured after the registered trace closes and the direct strict-DD command has completed; receipt_validation_finished is captured only after all device readbacks, numerical/parity evaluation, source/artifact input resolution, and receipt-input validation, immediately before canonical sealing/publish.",
                    "process_id":run_provenance.process_id,
                    "executable":&run_provenance.executable,
                    "host_platform":&run_provenance.host_platform,
                    "physical_trace":{
                        "interval_id":&interval_id,
                        "phase":"dsv4f_p4b_position1",
                        "role":"causal_kv_and_complete_attention_tail",
                        "batch":1,
                        "iteration":POSITION1,
                        "registered_context_command_buffers":counts.command_count,
                        "registered_context_compute_encoders":counts.encoder_count,
                        "direct_dynamic_library_command_buffers_outside_context_trace":1,
                        "direct_dynamic_library_trace_label_sha256":direct_trace_label_sha256
                    }
                },
                "strict_dd_control_contract":{
                    "control_strategy":"general finite-F32 [-40,40] strict double-double reconstruction of the active Darwin expf normal path; source sigmoid and 20-iteration Sinkhorn loop/order retained",
                    "table_contract":{
                        "entries":DARWIN_EXPF_TABLE.len(),
                        "layout":"128 x float4 [high,middle,low,pad]",
                        "bytes":DARWIN_EXPF_TABLE.len()*std::mem::size_of::<[f32;4]>(),
                        "sha256":candidate.table_sha256
                    },
                    "trace_or_logit_keys":false,
                    "ulp_repair":false,
                    "host_activation":false,
                    "host_routing":false,
                    "host_sampling":false,
                    "runtime_registry_registration":false,
                    "authority_kernel_unchanged":true,
                    "reusable_p4b_executor_reclassified":false
                },
                "numeric_parity_v2_1":{
                    "schema":"hawking.numeric_parity.v2_1",
                    "reference_authority":"P4B operator FP64 references are independently accumulated from qualified BF16 source-store checkpoints, raw streamed F32 sink/YaRN tables, and raw FP8/E8M0 WO payloads; all upstream P0/P1 discrete source handoffs are exact-checked in this run. Declared BF16 stores, including the TileLang numerator weights before the value GEMM, are materialized before storage parity. This is not upstream-runtime parity.",
                    "f32_operator_bounds":f32_operator_bounds,
                    "bf16_storage_bounds":bf16_storage_bounds,
                    "scores":Value::Object(score_json.clone()),
                    "all_host_and_device_scores_pass":true
                },
                "discrete_parity":{
                    "p0_raw_embed_to_kv_cache_row0_exact":true,
                    "p0_position_zero_rope_source_identity_no_device_transform":true,
                    "p1_raw_embed_to_q_kv_rope_exact":true,
                    "device_kv_cache_rows_0_and_1_exact":true,
                    "two_kv_sparse_sink_with_bf16_numerator_weights_exact":true,
                    "inverse_rope_exact":true,
                    "wo_a_converted_bf16_einsum_exact":true,
                    "wo_b_qat_and_bf16_exact":true,
                    "p1_attention_hc_post":{
                        "source_cpu_device_bf16_exact":true,
                        "storage_delta":&p1_hc_post_storage_delta,
                        "mixes_control_f32_delta":&p1_hc_mixes_control_delta,
                        "post_control_f32_delta":&p1_hc_post_control_delta,
                        "comb_control_f32_delta":&p1_hc_comb_control_delta,
                        "admitted_by_numeric_parity_v2_1":true
                    }
                },
                "strict_dd_pass_predicate":{
                    "mixes_f32_bitwise_exact":mixes_exact,
                    "post_f32_bitwise_exact":post_exact,
                    "comb_f32_bitwise_exact":comb_exact,
                    "terminal_bf16_storage_exact":terminal_exact,
                    "all_numeric_parity_v2_1_scores_pass":all_scores_pass,
                    "all_predicate_terms_pass":candidate_pass
                },
                "metal":{
                    "device":device_name,
                    "registered_context":{
                        "gpu_dispatches":expected_dispatches,
                        "command_buffers":expected_dispatches,
                        "compute_encoders":expected_dispatches,
                        "cpu_visible_waits":expected_dispatches,
                        "empty_command_buffers":0,
                        "physical_trace_command_buffers":counts.command_count,
                        "physical_trace_compute_encoders":counts.encoder_count,
                        "context_commit_count":commits,
                        "trace_samples":trace_samples.len()
                    },
                    "direct_dynamic_library":{
                        "gpu_dispatches":1,
                        "command_buffers":1,
                        "compute_encoders":1,
                        "cpu_visible_waits":1,
                        "in_registered_context_trace":false,
                        "physical_accounting":"exactly one completed, timestamped dynamic-library command; separately counted because it is deliberately not registered in MetalContext",
                        "profile":direct_profile
                    },
                    "totals":{
                        "real_gpu_dispatches":expected_dispatches + 1,
                        "real_command_buffers":expected_dispatches + 1,
                        "real_compute_encoders":expected_dispatches + 1,
                        "real_cpu_visible_waits":expected_dispatches + 1,
                        "empty_command_buffers":0,
                        "fallback":false,
                        "fallback_count":0,
                        "host_intermediate_handoff_bytes":0
                    },
                    "pipeline_limits":limits,
                    "buffers_created":buffers_created,
                    "bytes_allocated":bytes_allocated,
                    "cache_allocation_bytes":cache_initial.len(),
                    "active_kv_cache_bytes_at_position1":POSITION1_KV_ROWS*HEAD_DIM*2
                },
                "command_topology":{
                    "current":"34 ordered timestamped registered-context command buffers plus one dedicated strict-DD dynamic-library command; all P0/P1 dependencies remain device-resident and host reads only after the chain completes",
                    "registered_context_trace_coverage":expected_dispatches,
                    "direct_dynamic_library_trace_coverage":0,
                    "direct_dynamic_library_explicitly_counted":1,
                    "not_a_persistent_decode_graph":true,
                    "not_a_runtime_command_topology":true
                },
                "complete_position1_attention_stage_profile":complete_stage_profiles,
                "physical_trace":{
                    "interval_id":interval_id,
                    "run_nonce":run_nonce,
                    "phase":"dsv4f_p4b_position1_complete_attention",
                    "role":"causal_kv_and_complete_attention_tail",
                    "registered_context_command_buffers":counts.command_count,
                    "registered_context_compute_encoders":counts.encoder_count,
                    "direct_dynamic_library_command_buffers_outside_context_trace":1,
                    "direct_dynamic_library_timestamped":true
                },
                "receipt_sealing":{
                    "create_new_only":true,
                    "required_basename":STRICT_DD_V2_BASENAME,
                    "canonical_json_sha256":true,
                    "in_memory_seal_verified_before_write":true,
                    "temporary_json_reparsed_and_canonical_seal_verified_before_final_name_publish":true,
                    "final_name_hard_linked_from_verified_temp":true,
                    "canonical_verification_failure_cannot_leave_final_named_receipt":true
                },
                "claim_boundary":"Real Metal executes one fresh bounded raw-artifact BOS->position1 causal cache transition and the complete P1 layer-0 attention tail with a general strict-DD mHC control replacement. It establishes neither P7, a 43-layer runtime, token generation, HCLI, endpoint behavior, nor BASE_TRUE_TPS."
            });
            let (receipt, seal) = seal(decimal_strings(unsigned))?;
            let receipt_file_sha256 = write_new_and_verify_canonical(out, &receipt)?;
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "status":STRICT_DD_V2_RECEIPT_STATUS,
                    "receipt":out,
                    "seal_sha256":seal,
                    "receipt_file_sha256":receipt_file_sha256,
                    "canonical_seal_verified_before_final_name_publish":true,
                    "real_gpu_dispatches":expected_dispatches + 1,
                    "fallback":false,
                    "claim_boundary":"P4B P0->P1 attention only; not P7/runtime/token/HCLI/TPS"
                }))?
            );
            return Ok(());
        }
        let unsigned = json!({
            "schema":RECEIPT_SCHEMA,"status":RECEIPT_STATUS,
            "scope":{"token_ids":[PREFIX_TOKEN_ID,POSITION1_TOKEN_ID],"batch":1,"sequence_tokens":2,"position":POSITION1,"compress_ratio":0,"window_size":WINDOW_SIZE,"device_resident_two_row_kv_cache":true,"source_window_topk_indices":[0,1],"complete_position1_attention_tail":true,"not_layer_ffn":true,"not_router_or_moe":true,"not_full_model_or_runtime":true,"not_hcli":true,"not_base_true_tps":true},
            "artifact":{"path":reader.artifact_root().display().to_string(),"full_stream_schema":FULL_STREAM_SCHEMA,"full_stream_status":FULL_STREAM_STATUS,"manifest_file_sha256":reader.manifest_file_sha256(),"manifest_seal_sha256":reader.manifest_seal_sha256(),"restart_receipt_seal_sha256":reader.restart_seal_sha256(),"source_parent_retained":false},
            "source":{"repository":reader.source_identity().repository,"revision":reader.source_identity().revision,"source_hashes":{"inference/model.py":reader.source_metadata_asset_sha256("inference/model.py")?,"inference/kernel.py":reader.source_metadata_asset_sha256("inference/kernel.py")?,"inference/convert.py":reader.source_metadata_asset_sha256("inference/convert.py")?,"inference/config.json":reader.source_metadata_asset_sha256("inference/config.json")?,"config.json":reader.source_metadata_asset_sha256("config.json")?},"raw_artifact_inputs":{"embedding_rows":[PREFIX_TOKEN_ID,POSITION1_TOKEN_ID],"weights":[LAYER0_WQ_A_WEIGHT,LAYER0_WQ_B_WEIGHT,LAYER0_WKV_WEIGHT,LAYER0_WO_A_WEIGHT,LAYER0_WO_B_WEIGHT],"all_touched_chunks_sha256_verified_before_gpu_upload":true,"parent_safetensors_materialized":false}},
            "predecessors":{"position1_complete_cpu_oracle":binding_json(&cpu_binding),"p4a_complete_attention_authority":binding_json(&p4a_binding),"p4a_one_cb_topology_win":binding_json(&topology_binding)},
            "numeric_parity_v2_1":{"schema":"hawking.numeric_parity.v2_1","reference_authority":"P4B operator FP64 references are independently accumulated from qualified BF16 source-store checkpoints, raw streamed F32 sink/YaRN tables, and raw FP8/E8M0 WO payloads; all upstream P0/P1 discrete source handoffs are exact-checked in this run. Declared BF16 stores, including the TileLang numerator weights before the value GEMM, are materialized before storage parity. This is not upstream-runtime parity.","f32_operator_bounds":decimal_strings(serde_json::to_value(fb)?),"bf16_storage_bounds":decimal_strings(serde_json::to_value(bb)?),"scores":Value::Object(score_json),"all_host_and_device_scores_pass":true},
            "discrete_parity":{"p0_raw_embed_to_kv_cache_row0_exact":true,"p0_position_zero_rope_source_identity_no_device_transform":true,"p1_raw_embed_to_q_kv_rope_exact":true,"device_kv_cache_rows_0_and_1_exact":true,"two_kv_sparse_sink_with_bf16_numerator_weights_exact":true,"inverse_rope_exact":true,"wo_a_converted_bf16_einsum_exact":true,"wo_b_qat_and_bf16_exact":true,"p1_attention_hc_post":{"source_cpu_device_bf16_exact":p1_hc_post_storage_delta["mismatch_count"]==0,"storage_delta":p1_hc_post_storage_delta,"mixes_control_f32_delta":p1_hc_mixes_control_delta,"post_control_f32_delta":p1_hc_post_control_delta,"comb_control_f32_delta":p1_hc_comb_control_delta,"admitted_by_numeric_parity_v2_1":true}},
            "metal":{"device":device_name,"pipelines_precompiled_before_dispatch":true,"pipeline_limits":limits,"gpu_dispatches":P4B_DISPATCHES,"command_buffers":P4B_DISPATCHES,"compute_encoders":P4B_DISPATCHES,"cpu_visible_waits":P4B_DISPATCHES,"empty_command_buffers":0,"physical_trace_command_buffers":counts.command_count,"physical_trace_compute_encoders":counts.encoder_count,"trace_samples":trace_samples.len(),"buffers_created":buffers_created,"bytes_allocated":bytes_allocated,"cache_allocation_bytes":cache_initial.len(),"active_kv_cache_bytes_at_position1":POSITION1_KV_ROWS*HEAD_DIM*2,"fallback":false,"fallback_count":0,"host_intermediate_handoff_bytes":0},
            "command_topology":{"current":"33 ordered timestamped authority command buffers; each P0/P1 dependency stays entirely device-resident and host reads only after the chain completes","p4a_one_cb_predecessor_retained":true,"not_a_persistent_decode_graph":true},
            "complete_position1_attention_stage_profile":stage_profiles,
            "physical_trace":{"interval_id":interval_id,"run_nonce":run_nonce,"phase":"dsv4f_p4b_position1_complete_attention","role":"causal_kv_read_write_and_attention_tail"},
            "claim_boundary":"Real Metal executes one bounded raw-artifact BOS->position1 causal cache transition and the full P1 layer-0 attention tail only. It does not establish the layer FFN, routed experts, a 43-layer runtime, generation, HCLI, or BASE_TRUE_TPS."
        });
        let (receipt, seal) = seal(decimal_strings(unsigned))?;
        let out = args.out.as_ref().ok_or_else(|| {
            failure("--out is required outside the unsealed precise candidate mode")
        })?;
        write_new(out, &receipt)?;
        println!(
            "{}",
            serde_json::to_string(
                &json!({"status":RECEIPT_STATUS,"receipt":out,"seal_sha256":seal,"gpu_dispatches":P4B_DISPATCHES,"fallback":false})
            )?
        );
        Ok(())
    }

    impl StrictMhcControlTraceCandidate {
        fn new(context: &MetalContext) -> ProbeResult<Self> {
            let options = CompileOptions::new();
            options.set_fast_math_enabled(false);
            let library = context
                .device()
                .new_library_with_source(hawking_core::metal::SHADER_MATMUL, &options)
                .map_err(failure)?;
            let control_function = library
                .get_function(HC_PRECISE_CONTROL_CANDIDATE_KERNEL, None)
                .map_err(failure)?;
            let repair_function = library
                .get_function(HC_POST_ULP_REPAIR_TRACE_CANDIDATE_KERNEL, None)
                .map_err(failure)?;
            let control_pipeline = context
                .device()
                .new_compute_pipeline_state_with_function(&control_function)
                .map_err(failure)?;
            let repair_pipeline = context
                .device()
                .new_compute_pipeline_state_with_function(&repair_function)
                .map_err(failure)?;
            Ok(Self {
                library,
                control_pipeline,
                repair_pipeline,
            })
        }

        #[allow(clippy::too_many_arguments)]
        fn overwrite_controls(
            &self,
            context: &MetalContext,
            mixes: &metal::Buffer,
            hc_scale: &metal::Buffer,
            hc_base: &metal::Buffer,
            post: &metal::Buffer,
            comb: &metal::Buffer,
            hc_mult: u32,
            mix_width: u32,
            sinkhorn_iters: u32,
            hc_eps: f32,
        ) -> ProbeResult<()> {
            let command = context.queue().new_command_buffer();
            let control = command.new_compute_command_encoder();
            control.set_compute_pipeline_state(&self.control_pipeline);
            control.set_buffer(0, Some(mixes), 0);
            control.set_buffer(1, Some(hc_scale), 0);
            control.set_buffer(2, Some(hc_base), 0);
            control.set_buffer(3, Some(post), 0);
            control.set_buffer(4, Some(comb), 0);
            set_u32(control, 5, &hc_mult);
            set_u32(control, 6, &mix_width);
            set_u32(control, 7, &sinkhorn_iters);
            set_f32(control, 8, &hc_eps);
            control.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            control.end_encoding();

            let repair = command.new_compute_command_encoder();
            repair.set_compute_pipeline_state(&self.repair_pipeline);
            repair.set_buffer(0, Some(mixes), 0);
            repair.set_buffer(1, Some(hc_scale), 0);
            repair.set_buffer(2, Some(hc_base), 0);
            repair.set_buffer(3, Some(post), 0);
            repair.dispatch_threads(
                MTLSize::new(HC_MULT as u64, 1, 1),
                MTLSize::new(HC_MULT as u64, 1, 1),
            );
            repair.end_encoding();
            command.commit();
            command.wait_until_completed();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(failure(
                    "strict P4B mHC trace candidate command buffer did not complete",
                ));
            }
            Ok(())
        }
    }

    impl StrictMhcControlDdCandidate {
        fn new(context: &MetalContext) -> ProbeResult<Self> {
            let options = CompileOptions::new();
            options.set_fast_math_enabled(false);
            let library = context
                .device()
                .new_library_with_source(hawking_core::metal::SHADER_MATMUL, &options)
                .map_err(failure)?;
            let control_function = library
                .get_function(HC_DARWIN_DD_CONTROL_CANDIDATE_KERNEL, None)
                .map_err(failure)?;
            let control_pipeline = context
                .device()
                .new_compute_pipeline_state_with_function(&control_function)
                .map_err(failure)?;
            let table_bytes = darwin_expf_dd_table_bytes();
            let table_sha256 = sha256(&table_bytes);
            let table = context.new_buffer_with_bytes_checked(&table_bytes)?;
            Ok(Self {
                library,
                control_pipeline,
                table,
                table_sha256,
            })
        }

        #[allow(clippy::too_many_arguments)]
        fn overwrite_controls(
            &self,
            context: &MetalContext,
            trace_label: &str,
            mixes: &metal::Buffer,
            hc_scale: &metal::Buffer,
            hc_base: &metal::Buffer,
            post: &metal::Buffer,
            comb: &metal::Buffer,
            hc_mult: u32,
            mix_width: u32,
            sinkhorn_iters: u32,
            hc_eps: f32,
        ) -> ProbeResult<DirectCandidateTiming> {
            let total_started = Instant::now();
            let encode_started = Instant::now();
            let command = context.queue().new_command_buffer();
            command.set_label(trace_label);
            let control = command.new_compute_command_encoder();
            control.set_label(&format!(
                "{trace_label}|kind=compute_encoder|encoder_index=external|kernel={HC_DARWIN_DD_CONTROL_CANDIDATE_KERNEL}"
            ));
            control.set_compute_pipeline_state(&self.control_pipeline);
            control.set_buffer(0, Some(mixes), 0);
            control.set_buffer(1, Some(hc_scale), 0);
            control.set_buffer(2, Some(hc_base), 0);
            control.set_buffer(3, Some(post), 0);
            control.set_buffer(4, Some(comb), 0);
            set_u32(control, 5, &hc_mult);
            set_u32(control, 6, &mix_width);
            set_u32(control, 7, &sinkhorn_iters);
            set_f32(control, 8, &hc_eps);
            control.set_buffer(9, Some(&self.table), 0);
            control.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            control.end_encoding();
            let encode_us = encode_started.elapsed().as_micros() as u64;
            let submit_started = Instant::now();
            command.commit();
            let submit_us = submit_started.elapsed().as_micros() as u64;
            let wait_started = Instant::now();
            command.wait_until_completed();
            let wait_us = wait_started.elapsed().as_micros() as u64;
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(failure(
                    "strict P4B general Darwin-DD mHC candidate command buffer did not complete",
                ));
            }
            let (gpu_start_ns, gpu_end_ns, gpu_duration_ns) = gpu_timestamp_ns(&command)?;
            Ok(DirectCandidateTiming {
                encode_us,
                submit_us,
                wait_us,
                host_wall_us: total_started.elapsed().as_micros() as u64,
                gpu_duration_us: (gpu_duration_ns / 1_000).max(1),
                gpu_start_ns,
                gpu_end_ns,
            })
        }
    }

    fn darwin_expf_dd_table_bytes() -> Vec<u8> {
        let mut values = Vec::with_capacity(DARWIN_EXPF_TABLE.len() * 4);
        for (index, &entry) in DARWIN_EXPF_TABLE.iter().enumerate() {
            let source_bits = entry.wrapping_add(
                DARWIN_EXPF_MAGIC_BITS
                    .wrapping_add(index as u64)
                    .wrapping_shl(45),
            );
            let source = f64::from_bits(source_bits);
            let high = source as f32;
            let middle = (source - f64::from(high)) as f32;
            let low = (source - f64::from(high) - f64::from(middle)) as f32;
            values.extend_from_slice(&[high, middle, low, 0.0]);
        }
        f32bytes(&values)
    }

    /// Direct dynamic-library dispatches are outside `MetalContext`'s
    /// registered pipeline trace.  Their GPU timestamps are still read from
    /// the completed command buffer, never inferred from a host wait.
    fn gpu_timestamp_ns(command: &metal::CommandBufferRef) -> ProbeResult<(u64, u64, u64)> {
        let (start, end): (f64, f64) = unsafe {
            (
                msg_send![command, GPUStartTime],
                msg_send![command, GPUEndTime],
            )
        };
        if !(start.is_finite() && end.is_finite() && start > 0.0 && end > start) {
            return Err(failure(format!(
                "completed strict-DD command lacks valid GPU timestamps: start={start:?} end={end:?}"
            )));
        }
        let duration_ns = ((end - start) * 1_000_000_000.0).round() as u64;
        if duration_ns == 0 {
            return Err(failure("completed strict-DD GPU duration rounded to zero"));
        }
        Ok((
            (start * 1_000_000_000.0).round() as u64,
            (end * 1_000_000_000.0).round() as u64,
            duration_ns,
        ))
    }

    fn new_hc_state(ctx: &MetalContext, embed: &[u8]) -> ProbeResult<HcState> {
        Ok(HcState {
            embed: ctx.new_buffer_with_bytes_checked(embed)?,
            reduced: ctx.new_buffer_checked(HIDDEN_SIZE * 2)?,
            rsqrt: ctx.new_buffer_checked(4)?,
            mixes: ctx.new_buffer_checked(HC_MIX_WIDTH * 4)?,
            pre: ctx.new_buffer_checked(HC_MULT * 4)?,
            post: ctx.new_buffer_checked(HC_MULT * 4)?,
            comb: ctx.new_buffer_checked(HC_MULT * HC_MULT * 4)?,
            attn_norm: ctx.new_buffer_checked(HIDDEN_SIZE * 2)?,
        })
    }

    fn new_linear_scratch(
        ctx: &MetalContext,
        cols: usize,
        rows: usize,
    ) -> ProbeResult<LinearScratch> {
        Ok(LinearScratch {
            activation: ctx.new_buffer_checked(cols)?,
            scales: ctx.new_buffer_checked(cols / ACT_QUANT_BLOCK)?,
            fp32: ctx.new_buffer_checked(rows * 4)?,
            bf16: ctx.new_buffer_checked(rows * 2)?,
        })
    }

    fn new_kv_scratch(ctx: &MetalContext) -> ProbeResult<KvScratch> {
        Ok(KvScratch {
            norm: ctx.new_buffer_checked(HEAD_DIM * 2)?,
            qat: ctx.new_buffer_checked(HEAD_DIM * 2)?,
            activation: ctx.new_buffer_checked(NON_ROPE_HEAD_DIM)?,
            scales: ctx.new_buffer_checked(NON_ROPE_HEAD_DIM / KV_QAT_BLOCK)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn hc(
        ctx: &MetalContext,
        state: &HcState,
        f: &metal::Buffer,
        scale: &metal::Buffer,
        base: &metal::Buffer,
        hidden: u32,
        hc_mult: u32,
        mix: u32,
        iters: u32,
        norm_eps: f32,
        hc_eps: f32,
    ) -> ProbeResult<MetalDispatchTiming> {
        ctx.dispatch_threads_timed(HC_KERNEL, (1, 1, 1), (1, 1, 1), |e| {
            e.set_buffer(0, Some(&state.embed), 0);
            e.set_buffer(1, Some(f), 0);
            e.set_buffer(2, Some(scale), 0);
            e.set_buffer(3, Some(base), 0);
            e.set_buffer(4, Some(&state.reduced), 0);
            e.set_buffer(5, Some(&state.rsqrt), 0);
            e.set_buffer(6, Some(&state.mixes), 0);
            e.set_buffer(7, Some(&state.pre), 0);
            e.set_buffer(8, Some(&state.post), 0);
            e.set_buffer(9, Some(&state.comb), 0);
            set_u32(e, 10, &hidden);
            set_u32(e, 11, &hc_mult);
            set_u32(e, 12, &mix);
            set_u32(e, 13, &iters);
            set_f32(e, 14, &norm_eps);
            set_f32(e, 15, &hc_eps);
        })
        .map_err(Into::into)
    }

    #[allow(clippy::too_many_arguments)]
    fn hc_precise_control(
        ctx: &MetalContext,
        mixes: &metal::Buffer,
        scale: &metal::Buffer,
        base: &metal::Buffer,
        post: &metal::Buffer,
        comb: &metal::Buffer,
        hc_mult: u32,
        mix_width: u32,
        sinkhorn: u32,
        hc_eps: f32,
    ) -> ProbeResult<MetalDispatchTiming> {
        ctx.dispatch_threads_timed(
            HC_PRECISE_CONTROL_CANDIDATE_KERNEL,
            (1, 1, 1),
            (1, 1, 1),
            |e| {
                e.set_buffer(0, Some(mixes), 0);
                e.set_buffer(1, Some(scale), 0);
                e.set_buffer(2, Some(base), 0);
                e.set_buffer(3, Some(post), 0);
                e.set_buffer(4, Some(comb), 0);
                set_u32(e, 5, &hc_mult);
                set_u32(e, 6, &mix_width);
                set_u32(e, 7, &sinkhorn);
                set_f32(e, 8, &hc_eps);
            },
        )
        .map_err(Into::into)
    }

    fn rms(
        ctx: &MetalContext,
        input: &metal::Buffer,
        weight: &metal::Buffer,
        output: &metal::Buffer,
        width: u32,
        eps: f32,
    ) -> ProbeResult<MetalDispatchTiming> {
        ctx.dispatch_threads_timed(RMS_KERNEL, (1, 1, 1), (1, 1, 1), |e| {
            e.set_buffer(0, Some(input), 0);
            e.set_buffer(1, Some(weight), 0);
            e.set_buffer(2, Some(output), 0);
            set_u32(e, 3, &width);
            set_f32(e, 4, &eps);
        })
        .map_err(Into::into)
    }

    fn qat(
        ctx: &MetalContext,
        input: &metal::Buffer,
        output: &metal::Buffer,
        scales: &metal::Buffer,
        cols: u32,
    ) -> ProbeResult<MetalDispatchTiming> {
        ctx.dispatch_threads_timed(
            QAT_KERNEL,
            (cols / ACT_QUANT_BLOCK as u32, 1, 1),
            (32, 1, 1),
            |e| {
                e.set_buffer(0, Some(input), 0);
                e.set_buffer(1, Some(output), 0);
                e.set_buffer(2, Some(scales), 0);
                set_u32(e, 3, &cols);
            },
        )
        .map_err(Into::into)
    }

    #[allow(clippy::too_many_arguments)]
    fn fp8(
        ctx: &MetalContext,
        weight: &metal::Buffer,
        scales: &metal::Buffer,
        activation: &metal::Buffer,
        act_scales: &metal::Buffer,
        output: &metal::Buffer,
        rows: u32,
        cols: u32,
        scale_cols: u32,
    ) -> ProbeResult<MetalDispatchTiming> {
        ctx.dispatch_threads_timed(FP8_KERNEL, (rows, 1, 1), (256, 1, 1), |e| {
            e.set_buffer(0, Some(weight), 0);
            e.set_buffer(1, Some(scales), 0);
            e.set_buffer(2, Some(activation), 0);
            e.set_buffer(3, Some(act_scales), 0);
            e.set_buffer(4, Some(output), 0);
            set_u32(e, 5, &rows);
            set_u32(e, 6, &cols);
            set_u32(e, 7, &scale_cols);
        })
        .map_err(Into::into)
    }

    fn cast(
        ctx: &MetalContext,
        input: &metal::Buffer,
        output: &metal::Buffer,
        count: u32,
    ) -> ProbeResult<MetalDispatchTiming> {
        ctx.dispatch_threads_timed(CAST_KERNEL, (count, 1, 1), (256, 1, 1), |e| {
            e.set_buffer(0, Some(input), 0);
            e.set_buffer(1, Some(output), 0);
            set_u32(e, 2, &count);
        })
        .map_err(Into::into)
    }

    fn per_head(
        ctx: &MetalContext,
        input: &metal::Buffer,
        output: &metal::Buffer,
        heads: u32,
        dim: u32,
        eps: f32,
    ) -> ProbeResult<MetalDispatchTiming> {
        ctx.dispatch_threads_timed(PER_HEAD_KERNEL, (heads, 1, 1), (64, 1, 1), |e| {
            e.set_buffer(0, Some(input), 0);
            e.set_buffer(1, Some(output), 0);
            set_u32(e, 2, &heads);
            set_u32(e, 3, &dim);
            set_f32(e, 4, &eps);
        })
        .map_err(Into::into)
    }

    fn kv_qat(
        ctx: &MetalContext,
        input: &metal::Buffer,
        output: &metal::Buffer,
        activation: &metal::Buffer,
        scales: &metal::Buffer,
        dim: u32,
        rope: u32,
        block: u32,
    ) -> ProbeResult<MetalDispatchTiming> {
        ctx.dispatch_threads_timed(
            KV_QAT_KERNEL,
            (NON_ROPE_HEAD_DIM as u32 / block, 1, 1),
            (32, 1, 1),
            |e| {
                e.set_buffer(0, Some(input), 0);
                e.set_buffer(1, Some(output), 0);
                e.set_buffer(2, Some(activation), 0);
                e.set_buffer(3, Some(scales), 0);
                set_u32(e, 4, &dim);
                set_u32(e, 5, &rope);
                set_u32(e, 6, &block);
            },
        )
        .map_err(Into::into)
    }

    fn rope(
        ctx: &MetalContext,
        input: &metal::Buffer,
        cos: &metal::Buffer,
        sin: &metal::Buffer,
        output: &metal::Buffer,
        rows: u32,
        dim: u32,
        rope_dim: u32,
        inverse: u32,
    ) -> ProbeResult<MetalDispatchTiming> {
        ctx.dispatch_threads_timed(ROPE_KERNEL, (rows * dim / 2, 1, 1), (256, 1, 1), |e| {
            e.set_buffer(0, Some(input), 0);
            e.set_buffer(1, Some(cos), 0);
            e.set_buffer(2, Some(sin), 0);
            e.set_buffer(3, Some(output), 0);
            set_u32(e, 4, &rows);
            set_u32(e, 5, &dim);
            set_u32(e, 6, &rope_dim);
            set_u32(e, 7, &inverse);
        })
        .map_err(Into::into)
    }

    fn cache_write(
        ctx: &MetalContext,
        input: &metal::Buffer,
        cache: &metal::Buffer,
        position: u32,
        dim: u32,
        capacity: u32,
    ) -> ProbeResult<MetalDispatchTiming> {
        ctx.dispatch_threads_timed(CACHE_KERNEL, (dim, 1, 1), (256, 1, 1), |e| {
            e.set_buffer(0, Some(input), 0);
            e.set_buffer(1, Some(cache), 0);
            set_u32(e, 2, &position);
            set_u32(e, 3, &dim);
            set_u32(e, 4, &capacity);
        })
        .map_err(Into::into)
    }

    #[allow(clippy::too_many_arguments)]
    fn sparse(
        ctx: &MetalContext,
        q: &metal::Buffer,
        cache: &metal::Buffer,
        sink: &metal::Buffer,
        output: &metal::Buffer,
        scores: &metal::Buffer,
        denoms: &metal::Buffer,
        heads: u32,
        dim: u32,
        capacity: u32,
        scale: f32,
    ) -> ProbeResult<MetalDispatchTiming> {
        ctx.dispatch_threads_timed(SPARSE_KERNEL, (heads, 1, 1), (64, 1, 1), |e| {
            e.set_buffer(0, Some(q), 0);
            e.set_buffer(1, Some(cache), 0);
            e.set_buffer(2, Some(sink), 0);
            e.set_buffer(3, Some(output), 0);
            e.set_buffer(4, Some(scores), 0);
            e.set_buffer(5, Some(denoms), 0);
            set_u32(e, 6, &heads);
            set_u32(e, 7, &dim);
            set_u32(e, 8, &capacity);
            set_f32(e, 9, &scale);
        })
        .map_err(Into::into)
    }

    fn wo_a(
        ctx: &MetalContext,
        weight: &metal::Buffer,
        scales: &metal::Buffer,
        input: &metal::Buffer,
        output: &metal::Buffer,
        rows: u32,
        cols: u32,
        scale_cols: u32,
        ranks: u32,
    ) -> ProbeResult<MetalDispatchTiming> {
        ctx.dispatch_threads_timed(WO_A_KERNEL, (rows, 1, 1), (256, 1, 1), |e| {
            e.set_buffer(0, Some(weight), 0);
            e.set_buffer(1, Some(scales), 0);
            e.set_buffer(2, Some(input), 0);
            e.set_buffer(3, Some(output), 0);
            set_u32(e, 4, &rows);
            set_u32(e, 5, &cols);
            set_u32(e, 6, &scale_cols);
            set_u32(e, 7, &ranks);
        })
        .map_err(Into::into)
    }

    fn hc_post(
        ctx: &MetalContext,
        attention: &metal::Buffer,
        embed: &metal::Buffer,
        post: &metal::Buffer,
        comb: &metal::Buffer,
        output: &metal::Buffer,
        hidden: u32,
        hc_mult: u32,
    ) -> ProbeResult<MetalDispatchTiming> {
        ctx.dispatch_threads_timed(HC_POST_KERNEL, (hidden * hc_mult, 1, 1), (256, 1, 1), |e| {
            e.set_buffer(0, Some(attention), 0);
            e.set_buffer(1, Some(embed), 0);
            e.set_buffer(2, Some(post), 0);
            e.set_buffer(3, Some(comb), 0);
            e.set_buffer(4, Some(output), 0);
            set_u32(e, 5, &hidden);
            set_u32(e, 6, &hc_mult);
        })
        .map_err(Into::into)
    }

    fn parse_args() -> ProbeResult<Args> {
        let mut artifact = None;
        let mut cpu_oracle = None;
        let mut p4a_authority = None;
        let mut p4a_topology = None;
        let mut out = None;
        let mut precise_mhc_control_candidate = false;
        let mut strict_mhc_control_trace_candidate = false;
        let mut strict_mhc_control_dd_candidate = false;
        let mut seal_strict_mhc_control_dd_v2 = false;
        let mut it = std::env::args_os().skip(1);
        while let Some(flag) = it.next() {
            match flag.to_string_lossy().as_ref() {
                "--artifact" => artifact = it.next().map(PathBuf::from),
                "--cpu-oracle" => cpu_oracle = it.next().map(PathBuf::from),
                "--p4a-authority" => p4a_authority = it.next().map(PathBuf::from),
                "--p4a-topology" => p4a_topology = it.next().map(PathBuf::from),
                "--out" => out = it.next().map(PathBuf::from),
                "--precise-mhc-control-candidate" => precise_mhc_control_candidate = true,
                "--strict-mhc-control-trace-candidate" => strict_mhc_control_trace_candidate = true,
                "--strict-mhc-control-dd-candidate" => strict_mhc_control_dd_candidate = true,
                "--seal-strict-mhc-control-dd-v2" => seal_strict_mhc_control_dd_v2 = true,
                other => return Err(failure(format!("unknown argument {other}"))),
            }
        }
        let candidate_count = usize::from(precise_mhc_control_candidate)
            + usize::from(strict_mhc_control_trace_candidate)
            + usize::from(strict_mhc_control_dd_candidate)
            + usize::from(seal_strict_mhc_control_dd_v2);
        if candidate_count > 1 {
            return Err(failure(
                "P4B mHC diagnostic candidates are mutually exclusive",
            ));
        }
        if (candidate_count == 0 || seal_strict_mhc_control_dd_v2) && out.is_none() {
            return Err(failure("--out is required"));
        }
        if seal_strict_mhc_control_dd_v2
            && out
                .as_ref()
                .and_then(|path| path.file_name())
                .and_then(|name| name.to_str())
                != Some(STRICT_DD_V2_BASENAME)
        {
            return Err(failure(format!(
                "strict DD v2 output must use the create-new basename {STRICT_DD_V2_BASENAME}"
            )));
        }
        Ok(Args {
            artifact: artifact.ok_or_else(|| failure("--artifact is required"))?,
            cpu_oracle: cpu_oracle.ok_or_else(|| failure("--cpu-oracle is required"))?,
            p4a_authority: p4a_authority.ok_or_else(|| failure("--p4a-authority is required"))?,
            p4a_topology: p4a_topology.ok_or_else(|| failure("--p4a-topology is required"))?,
            out,
            precise_mhc_control_candidate,
            strict_mhc_control_trace_candidate,
            strict_mhc_control_dd_candidate,
            seal_strict_mhc_control_dd_v2,
        })
    }

    fn validate_cpu_receipt(
        path: &Path,
        reader: &DeepSeekV4FullStreamReader,
        cpu: &Layer0Position1CompleteAttentionCpuOracleResult,
    ) -> ProbeResult<ReceiptBinding> {
        let binding = admitted_binding(path, CPU_BASENAME, CPU_SCHEMA, CPU_STATUS, reader)?;
        let value: Value = serde_json::from_slice(&fs::read(&binding.path)?)?;
        if text(
            &value,
            &["causal_checkpoints", "two_row_kv_cache_bf16_sha256"],
        )? != sha256(&u16bytes(&cpu.causal.kv_cache_two_rows_bf16_bits))
            || text(
                &value,
                &["tail_checkpoints", "attention_hc_post_bf16_sha256"],
            )? != sha256(&u16bytes(&cpu.hc_attention_post_bf16_bits))
        {
            return Err(failure(
                "complete CPU continuation differs from sealed receipt",
            ));
        }
        Ok(binding)
    }

    fn validate_p4a_receipt(
        path: &Path,
        reader: &DeepSeekV4FullStreamReader,
    ) -> ProbeResult<ReceiptBinding> {
        let binding = admitted_binding(path, P4A_BASENAME, P4A_SCHEMA, P4A_STATUS, reader)?;
        let value: Value = serde_json::from_slice(&fs::read(&binding.path)?)?;
        if value.pointer("/metal/fallback").and_then(Value::as_bool) != Some(false)
            || value
                .pointer("/numeric_parity_v2_1/all_host_and_device_scores_pass")
                .and_then(Value::as_bool)
                != Some(true)
        {
            return Err(failure("P4A authority lacks parity/no-fallback admission"));
        }
        Ok(binding)
    }

    fn validate_topology_receipt(
        path: &Path,
        reader: &DeepSeekV4FullStreamReader,
    ) -> ProbeResult<ReceiptBinding> {
        let binding = admitted_binding(
            path,
            P4A_TOPOLOGY_BASENAME,
            P4A_TOPOLOGY_SCHEMA,
            P4A_TOPOLOGY_STATUS,
            reader,
        )?;
        let value: Value = serde_json::from_slice(&fs::read(&binding.path)?)?;
        if value
            .pointer("/promotion/promoted")
            .and_then(Value::as_bool)
            != Some(true)
            || value.pointer("/parity/fallback").and_then(Value::as_bool) != Some(false)
        {
            return Err(failure(
                "P4A topology receipt is not a promoted no-fallback predecessor",
            ));
        }
        Ok(binding)
    }

    fn admitted_binding(
        path: &Path,
        basename: &str,
        schema: &str,
        status: &str,
        reader: &DeepSeekV4FullStreamReader,
    ) -> ProbeResult<ReceiptBinding> {
        if path.file_name().and_then(|x| x.to_str()) != Some(basename) {
            return Err(failure(format!(
                "wrong predecessor basename: expected {basename}"
            )));
        }
        let path = fs::canonicalize(path)?;
        let raw = fs::read(&path)?;
        let value: Value = serde_json::from_slice(&raw)?;
        seal_ok(&value)?;
        if text(&value, &["schema"])? != schema
            || text(&value, &["status"])? != status
            || text(&value, &["artifact", "manifest_seal_sha256"])? != reader.manifest_seal_sha256()
        {
            return Err(failure("predecessor source binding differs"));
        }
        Ok(ReceiptBinding {
            path,
            file_sha256: sha256(&raw),
            seal_sha256: text(&value, &["seal_sha256"])?.to_owned(),
        })
    }

    fn binding_json(binding: &ReceiptBinding) -> Value {
        json!({"path":binding.path.display().to_string(),"file_sha256":binding.file_sha256,"seal_sha256":binding.seal_sha256})
    }

    fn embed_row(reader: &DeepSeekV4FullStreamReader, token: u64) -> ProbeResult<Vec<u8>> {
        let meta = reader.tensor_metadata(EMBED_WEIGHT)?;
        if meta.dtype != "BF16" || meta.shape.as_slice() != [129_280, HIDDEN_SIZE as u64] {
            return Err(failure("embedding source geometry changed"));
        }
        let bytes = HIDDEN_SIZE * 2;
        let start = token
            .checked_mul(bytes as u64)
            .ok_or_else(|| failure("embedding start overflow"))?;
        reader
            .read_verified_range(EMBED_WEIGHT, start..start + bytes as u64, bytes)
            .map_err(Into::into)
    }

    fn full(reader: &DeepSeekV4FullStreamReader, name: &str) -> ProbeResult<Vec<u8>> {
        let m = reader.tensor_metadata(name)?;
        reader
            .read_verified_full(name, m.bytes as usize)
            .map_err(Into::into)
    }

    fn fp8_pair(
        reader: &DeepSeekV4FullStreamReader,
        weight: &str,
        scale: &str,
        rows: usize,
        cols: usize,
    ) -> ProbeResult<(Vec<u8>, Vec<u8>)> {
        let pair = reader.native_scale_pair(weight)?;
        if pair.kind != NativeScalePairKind::Fp8E4M3fn
            || pair.scale.name != scale
            || pair.weight.shape.as_slice() != [rows as u64, cols as u64]
            || pair.scale.shape.as_slice()
                != [
                    (rows / ACT_QUANT_BLOCK) as u64,
                    (cols / ACT_QUANT_BLOCK) as u64,
                ]
        {
            return Err(failure(format!("{weight} FP8 geometry changed")));
        }
        Ok((
            reader.read_verified_full(weight, pair.weight.bytes as usize)?,
            reader.read_verified_full(scale, pair.scale.bytes as usize)?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn geometry_check(
        embed0: &[u8],
        embed1: &[u8],
        hc_fn: &[u8],
        hc_base: &[u8],
        hc_scale: &[u8],
        norm: &[u8],
        qnorm: &[u8],
        kvnorm: &[u8],
        sink: &[u8],
    ) -> ProbeResult<()> {
        if embed0.len() != HIDDEN_SIZE * 2
            || embed1.len() != HIDDEN_SIZE * 2
            || hc_fn.len() != HC_MIX_WIDTH * HC_MULT * HIDDEN_SIZE * 4
            || hc_base.len() != HC_MIX_WIDTH * 4
            || hc_scale.len() != 3 * 4
            || norm.len() != HIDDEN_SIZE * 2
            || qnorm.len() != Q_LORA_RANK * 2
            || kvnorm.len() != HEAD_DIM * 2
            || sink.len() != NUM_HEADS * 4
        {
            return Err(failure("P4B source tensor geometry changed"));
        }
        Ok(())
    }

    fn rope_f64(
        input: &[u16],
        rows: usize,
        cos: &[f32],
        sin: &[f32],
        inverse: bool,
    ) -> ProbeResult<Vec<f64>> {
        if input.len() != rows * HEAD_DIM
            || cos.len() != ROPE_HEAD_DIM / 2
            || sin.len() != ROPE_HEAD_DIM / 2
        {
            return Err(failure("FP64 RoPE geometry"));
        }
        let mut out = bf16_bits_f64(input);
        let start = HEAD_DIM - ROPE_HEAD_DIM;
        for row in 0..rows {
            for pair in 0..ROPE_HEAD_DIM / 2 {
                let i = row * HEAD_DIM + start + 2 * pair;
                let c = cos[pair] as f64;
                let s = if inverse {
                    -(sin[pair] as f64)
                } else {
                    sin[pair] as f64
                };
                let a = out[i];
                let b = out[i + 1];
                out[i] = bf16_round(a * c - b * s);
                out[i + 1] = bf16_round(a * s + b * c);
            }
        }
        Ok(out)
    }

    fn hc_controls_f64(
        embed: &[u8],
        hc_fn: &[u8],
        hc_scale: &[u8],
        hc_base: &[u8],
    ) -> ProbeResult<HcControlsF64> {
        let embed = bf16_bytes_f64(embed)?;
        let hc_fn = f32bytes_f64(hc_fn)?;
        let hc_scale = f32bytes_f64(hc_scale)?;
        let hc_base = f32bytes_f64(hc_base)?;
        if embed.len() != HIDDEN_SIZE
            || hc_fn.len() != HC_MIX_WIDTH * HC_MULT * HIDDEN_SIZE
            || hc_scale.len() != 3
            || hc_base.len() != HC_MIX_WIDTH
        {
            return Err(failure("FP64 mHC control geometry"));
        }
        let mut sum_square = 0.0;
        for _ in 0..HC_MULT {
            for &value in &embed {
                sum_square += value * value;
            }
        }
        let reciprocal =
            1.0 / (sum_square / (HC_MULT * HIDDEN_SIZE) as f64 + RMS_NORM_EPS as f64).sqrt();
        let mut mixes = vec![0.0; HC_MIX_WIDTH];
        for row in 0..HC_MIX_WIDTH {
            let mut accumulator = 0.0;
            for lane in 0..HC_MULT {
                let base = row * HC_MULT * HIDDEN_SIZE + lane * HIDDEN_SIZE;
                for col in 0..HIDDEN_SIZE {
                    accumulator += hc_fn[base + col] * embed[col];
                }
            }
            mixes[row] = accumulator * reciprocal;
        }
        let mut post = vec![0.0; HC_MULT];
        let mut comb = vec![0.0; HC_MULT * HC_MULT];
        for lane in 0..HC_MULT {
            post[lane] = 2.0
                / (1.0 + (-(mixes[lane + HC_MULT] * hc_scale[1] + hc_base[lane + HC_MULT])).exp());
        }
        for row in 0..HC_MULT {
            for col in 0..HC_MULT {
                let index = row * HC_MULT + col;
                comb[index] =
                    mixes[index + 2 * HC_MULT] * hc_scale[2] + hc_base[index + 2 * HC_MULT];
            }
        }
        for row in 0..HC_MULT {
            let base = row * HC_MULT;
            let max = comb[base..base + HC_MULT]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut total = 0.0;
            for col in 0..HC_MULT {
                comb[base + col] = (comb[base + col] - max).exp();
                total += comb[base + col];
            }
            for col in 0..HC_MULT {
                comb[base + col] = comb[base + col] / total + HC_EPS as f64;
            }
        }
        hc_normalize_columns(&mut comb)?;
        for _ in 1..HC_SINKHORN_ITERS {
            hc_normalize_rows(&mut comb)?;
            hc_normalize_columns(&mut comb)?;
        }
        Ok(HcControlsF64 { post, comb })
    }

    fn hc_normalize_rows(comb: &mut [f64]) -> ProbeResult<()> {
        for row in 0..HC_MULT {
            let base = row * HC_MULT;
            let total: f64 = comb[base..base + HC_MULT].iter().sum();
            if !total.is_finite() || total <= 0.0 {
                return Err(failure("FP64 mHC row normalization"));
            }
            for value in &mut comb[base..base + HC_MULT] {
                *value /= total + HC_EPS as f64;
            }
        }
        Ok(())
    }

    fn hc_normalize_columns(comb: &mut [f64]) -> ProbeResult<()> {
        for col in 0..HC_MULT {
            let mut total = 0.0;
            for row in 0..HC_MULT {
                total += comb[row * HC_MULT + col];
            }
            if !total.is_finite() || total <= 0.0 {
                return Err(failure("FP64 mHC column normalization"));
            }
            for row in 0..HC_MULT {
                comb[row * HC_MULT + col] /= total + HC_EPS as f64;
            }
        }
        Ok(())
    }

    fn sparse_f64(
        q: &[u16],
        cache: &[u16],
        sink: &[f32],
        cos: &[f32],
        sin: &[f32],
    ) -> ProbeResult<RopeSparseF64> {
        if q.len() != WQ_B_ROWS
            || cache.len() != POSITION1_KV_ROWS * HEAD_DIM
            || sink.len() != NUM_HEADS
        {
            return Err(failure("FP64 sparse geometry"));
        }
        let q_rope = bf16_bits_f64(q);
        let cache = bf16_bits_f64(cache);
        let mut scores = Vec::with_capacity(NUM_HEADS * 2);
        let mut denominators = Vec::with_capacity(NUM_HEADS);
        let mut attention = Vec::with_capacity(WQ_B_ROWS);
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        for head in 0..NUM_HEADS {
            let mut dots = [0.0; 2];
            for row in 0..2 {
                for d in 0..HEAD_DIM {
                    dots[row] += q_rope[head * HEAD_DIM + d] * cache[row * HEAD_DIM + d];
                }
                dots[row] *= scale;
            }
            let max = dots[0].max(dots[1]);
            let n0 = (dots[0] - max).exp();
            let n1 = (dots[1] - max).exp();
            let den = n0 + n1 + ((sink[head] as f64) - max).exp();
            scores.extend_from_slice(&dots);
            denominators.push(den);
            let n0 = bf16_round(n0);
            let n1 = bf16_round(n1);
            for d in 0..HEAD_DIM {
                attention.push(bf16_round((n0 * cache[d] + n1 * cache[HEAD_DIM + d]) / den));
            }
        }
        let derotated = rope_f64_from_f64(&attention, NUM_HEADS, cos, sin, true)?;
        Ok(RopeSparseF64 {
            scores,
            denominators,
            attention,
            derotated,
        })
    }

    fn rope_f64_from_f64(
        input: &[f64],
        rows: usize,
        cos: &[f32],
        sin: &[f32],
        inverse: bool,
    ) -> ProbeResult<Vec<f64>> {
        if input.len() != rows * HEAD_DIM
            || cos.len() != ROPE_HEAD_DIM / 2
            || sin.len() != ROPE_HEAD_DIM / 2
        {
            return Err(failure("FP64 inverse RoPE geometry"));
        }
        let mut out = input.to_vec();
        let start = HEAD_DIM - ROPE_HEAD_DIM;
        for row in 0..rows {
            for pair in 0..ROPE_HEAD_DIM / 2 {
                let i = row * HEAD_DIM + start + pair * 2;
                let c = cos[pair] as f64;
                let s = if inverse {
                    -(sin[pair] as f64)
                } else {
                    sin[pair] as f64
                };
                let a = out[i];
                let b = out[i + 1];
                out[i] = bf16_round(a * c - b * s);
                out[i + 1] = bf16_round(a * s + b * c);
            }
        }
        Ok(out)
    }

    fn wo_a_f64(attention: &[u16], weight: &[u8], scales: &[u8]) -> ProbeResult<Vec<f64>> {
        if attention.len() != WQ_B_ROWS
            || weight.len() != WO_A_ROWS * WO_A_COLS
            || scales.len() != WO_A_ROWS / ACT_QUANT_BLOCK * WO_A_COLS / ACT_QUANT_BLOCK
        {
            return Err(failure("FP64 WO-A geometry"));
        }
        let attention = bf16_bits_f64(attention);
        let scale_cols = WO_A_COLS / ACT_QUANT_BLOCK;
        let mut out = vec![0.; WO_A_ROWS];
        for row in 0..WO_A_ROWS {
            let group = row / O_LORA_RANK;
            let mut sum = 0.;
            for col in 0..WO_A_COLS {
                let converted = bf16_round(
                    e4(weight[row * WO_A_COLS + col])?
                        * e8(scales[(row / ACT_QUANT_BLOCK) * scale_cols + col / ACT_QUANT_BLOCK])?,
                );
                sum += attention[group * WO_A_COLS + col] * converted;
            }
            out[row] = bf16_round(sum);
        }
        Ok(out)
    }

    fn fp8_f64(
        weight: &[u8],
        scales: &[u8],
        activation: &[u8],
        activation_scales: &[u8],
        rows: usize,
        cols: usize,
    ) -> ProbeResult<Vec<f64>> {
        if weight.len() != rows * cols
            || scales.len() != rows / ACT_QUANT_BLOCK * cols / ACT_QUANT_BLOCK
            || activation.len() != cols
            || activation_scales.len() != cols / ACT_QUANT_BLOCK
        {
            return Err(failure("FP64 FP8 geometry"));
        }
        let scale_cols = cols / ACT_QUANT_BLOCK;
        let mut out = vec![0.; rows];
        for row in 0..rows {
            let mut total = 0.;
            for block in 0..scale_cols {
                let mut partial = 0.;
                for col in block * ACT_QUANT_BLOCK..(block + 1) * ACT_QUANT_BLOCK {
                    partial += e4(activation[col])? * e4(weight[row * cols + col])?;
                }
                total += partial
                    * e8(activation_scales[block])?
                    * e8(scales[(row / ACT_QUANT_BLOCK) * scale_cols + block])?;
            }
            out[row] = total;
        }
        Ok(out)
    }

    fn hc_post_f64(
        attention: &[u16],
        residual: &[u16],
        post: &[f32],
        comb: &[f32],
    ) -> ProbeResult<Vec<f64>> {
        if attention.len() != HIDDEN_SIZE
            || residual.len() != HC_MULT * HIDDEN_SIZE
            || post.len() != HC_MULT
            || comb.len() != HC_MULT * HC_MULT
        {
            return Err(failure("FP64 mHC post geometry"));
        }
        let attention = bf16_bits_f64(attention);
        let residual = bf16_bits_f64(residual);
        let mut out = vec![0.; HC_MULT * HIDDEN_SIZE];
        for lane in 0..HC_MULT {
            for f in 0..HIDDEN_SIZE {
                let mut value = post[lane] as f64 * attention[f];
                for source in 0..HC_MULT {
                    value +=
                        comb[source * HC_MULT + lane] as f64 * residual[source * HIDDEN_SIZE + f];
                }
                out[lane * HIDDEN_SIZE + f] = bf16_round(value);
            }
        }
        Ok(out)
    }

    fn e4(b: u8) -> ProbeResult<f64> {
        let e = (b >> 3) & 15;
        let m = b & 7;
        if e == 15 && m == 7 {
            return Err(failure("E4M3 NaN"));
        }
        let value = if e == 0 {
            m as f64 * 2f64.powi(-9)
        } else {
            (1. + m as f64 / 8.) * 2f64.powi(e as i32 - 7)
        };
        Ok(if b & 128 != 0 { -value } else { value })
    }
    fn e8(b: u8) -> ProbeResult<f64> {
        if b == 255 {
            Err(failure("E8M0 NaN"))
        } else {
            Ok(2f64.powi(b as i32 - 127))
        }
    }
    fn bf16_round(v: f64) -> f64 {
        let bits = (v as f32).to_bits();
        f32::from_bits(((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) << 16) as f64
    }
    fn f32_bounds() -> Bounds {
        Bounds {
            max_abs_near_zero: 1e-3,
            max_relative_l2: 2e-4,
            min_cosine: 0.999999,
            max_kl: 0.,
            require_kl: false,
            top_k: 5,
            max_meaningful_rel: 5e-2,
            gate_max_meaningful_rel: true,
        }
    }
    fn bf16_bounds() -> Bounds {
        Bounds {
            max_abs_near_zero: 2e-2,
            max_relative_l2: 1e-2,
            min_cosine: 0.999,
            max_kl: 0.,
            require_kl: false,
            top_k: 5,
            max_meaningful_rel: 1e-1,
            gate_max_meaningful_rel: true,
        }
    }
    fn score(
        name: &str,
        host: &[f32],
        device: &[f32],
        reference: &[f64],
        bounds: &Bounds,
    ) -> ProbeResult<PairedScore> {
        if host.len() != device.len() || host.len() != reference.len() {
            return Err(failure(format!("V2.1 {name} geometry")));
        }
        let outcome = score_pair(host, device, reference, bounds);
        if !outcome.pass {
            return Err(failure(format!(
                "V2.1 {name} failed: host={:?}; device={:?}",
                outcome.host.failures, outcome.device.failures
            )));
        }
        Ok(outcome)
    }
    fn checked(t: &MetalDispatchTiming, name: &str) -> ProbeResult<()> {
        if t.command_buffers != 1
            || t.compute_encoders != 1
            || t.compute_dispatches != 1
            || t.gpu_duration_us.is_none()
            || t.gpu_start_ns.is_none()
            || t.gpu_end_ns.is_none()
        {
            return Err(failure(format!("{name} lacks completed GPU timestamp")));
        }
        Ok(())
    }
    fn profile(
        stage: &str,
        kernel: &str,
        t: &MetalDispatchTiming,
        read: usize,
        written: usize,
        flops: u64,
    ) -> Value {
        json!({"stage":stage,"kernel":kernel,"gpu_duration_us":t.gpu_duration_us,"gpu_start_ns":t.gpu_start_ns,"gpu_end_ns":t.gpu_end_ns,"cpu_duration_us":t.host_wall_us,"cpu_encode_us":t.encode_us,"cpu_submit_us":t.submit_us,"cpu_wait_us":t.wait_us,"bytes_read":read,"bytes_written":written,"fp_operations_estimate":flops,"integer_or_bit_operations_estimate":0,"dispatches":1,"command_buffers":1,"waits":1,"occupancy":"not inferred from authority geometry","effective_bandwidth":"not inferred from single authority trial","p50_p95_p99":"single trial only","fallback":false,"unexplained_other":0})
    }

    fn strict_dd_direct_profile(timing: DirectCandidateTiming) -> Value {
        let bytes_read = HC_MIX_WIDTH * 4
            + 3 * 4
            + HC_MIX_WIDTH * 4
            + DARWIN_EXPF_TABLE.len() * std::mem::size_of::<[f32; 4]>();
        let bytes_written = (HC_MULT + HC_MULT * HC_MULT) * 4;
        let logical_bandwidth_bytes_per_s = (((bytes_read + bytes_written) as u128)
            .saturating_mul(1_000_000)
            .checked_div(timing.gpu_duration_us.max(1) as u128)
            .unwrap_or(0)) as u64;
        json!({
            "stage":"p1_mhc_post_comb_general_darwin_dd_control",
            "kernel":HC_DARWIN_DD_CONTROL_CANDIDATE_KERNEL,
            "gpu_duration_us":timing.gpu_duration_us,
            "gpu_start_ns":timing.gpu_start_ns,
            "gpu_end_ns":timing.gpu_end_ns,
            "cpu_duration_us":timing.host_wall_us,
            "cpu_encode_us":timing.encode_us,
            "cpu_submit_us":timing.submit_us,
            "cpu_wait_us":timing.wait_us,
            "bytes_read":bytes_read,
            "bytes_written":bytes_written,
            "fp_operations":{
                "darwin_expf_dd_calls":20,
                "static_total":"not collapsed to a scalar because each call expands to an explicit strict double-double instruction sequence",
                "source_operator":"fixed one-thread 4-way post sigmoid plus 4x4 initial softmax; 20 Sinkhorn iterations retain source order"
            },
            "integer_or_bit_operations":{
                "darwin_table_index_and_exponent_derivation":"one per Darwin expf call",
                "count":"not separately scalarized; no host-side repair or routing"
            },
            "dispatches":1,
            "command_buffers":1,
            "compute_encoders":1,
            "waits":1,
            "occupancy":"single-thread strict control diagnostic; no occupancy extrapolation",
            "effective_bandwidth":{
                "logical_read_write_bytes_per_s":logical_bandwidth_bytes_per_s,
                "interpretation":"logical bytes divided by completed-command-buffer GPU duration; not a hardware bandwidth roofline"
            },
            "p50_p95_p99":"single fresh strict-DD parity run only",
            "timestamp_authority":"completed MTLCommandBuffer GPUStartTime/GPUEndTime",
            "physical_trace":"direct dynamic-library command is explicitly outside MetalContext's registered-pipeline trace",
            "fallback":false,
            "unexplained_other":0
        })
    }
    fn exact8(name: &str, a: &[u8], b: &[u8]) -> ProbeResult<()> {
        if a != b {
            Err(failure(format!("{name} byte mismatch")))
        } else {
            Ok(())
        }
    }
    fn exact16(name: &str, a: &[u16], b: &[u16]) -> ProbeResult<()> {
        if a.len() != b.len() {
            return Err(failure(format!("{name} u16 length mismatch")));
        }
        if let Some((index, (&expected, &actual))) = a
            .iter()
            .zip(b)
            .enumerate()
            .find(|(_, (expected, actual))| expected != actual)
        {
            return Err(failure(format!(
                "{name} BF16 mismatch at {index}: expected=0x{expected:04x} actual=0x{actual:04x}"
            )));
        }
        Ok(())
    }
    fn bf16_difference(a: &[u16], b: &[u16]) -> ProbeResult<Value> {
        if a.len() != b.len() {
            return Err(failure("BF16 delta geometry"));
        }
        let mut mismatches = 0usize;
        let mut max_word_delta = 0u16;
        let mut max_abs = 0.0f32;
        let mut first = None;
        for (index, (&expected, &actual)) in a.iter().zip(b).enumerate() {
            if expected != actual {
                mismatches += 1;
                max_word_delta = max_word_delta.max(expected.abs_diff(actual));
                let delta = (f32::from_bits((expected as u32) << 16)
                    - f32::from_bits((actual as u32) << 16))
                .abs();
                max_abs = max_abs.max(delta);
                if first.is_none() {
                    first = Some(
                        json!({"index":index,"expected_bf16":"0x".to_owned()+&format!("{expected:04x}"),"device_bf16":"0x".to_owned()+&format!("{actual:04x}"),"absolute_delta":delta}),
                    );
                }
            }
        }
        Ok(
            json!({"elements":a.len(),"mismatch_count":mismatches,"max_word_delta":max_word_delta,"max_absolute_delta":max_abs,"first_mismatch":first}),
        )
    }
    fn f32_difference(a: &[f32], b: &[f32]) -> ProbeResult<Value> {
        if a.len() != b.len() {
            return Err(failure("F32 delta geometry"));
        }
        let mut mismatches = 0usize;
        let mut max_abs = 0.0f32;
        for (&expected, &actual) in a.iter().zip(b) {
            if expected.to_bits() != actual.to_bits() {
                mismatches += 1;
            }
            max_abs = max_abs.max((expected - actual).abs());
        }
        Ok(
            json!({"elements":a.len(),"bitwise_mismatch_count":mismatches,"max_absolute_delta":max_abs}),
        )
    }
    fn zero_count(value: &Value, field: &str) -> bool {
        value.get(field).and_then(Value::as_u64) == Some(0)
    }
    fn close(name: &str, a: &[f32], b: &[f32]) -> ProbeResult<()> {
        if a.len() != b.len() {
            return Err(failure(format!("{name} length")));
        }
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            if !x.is_finite() || !y.is_finite() || (x - y).abs() > 1e-4 + 1e-4 * x.abs() {
                return Err(failure(format!("{name} f32 mismatch at {i}: {x} {y}")));
            }
        }
        Ok(())
    }
    fn set_u32(e: &metal::ComputeCommandEncoderRef, index: u64, value: &u32) {
        e.set_bytes(index, 4, value as *const u32 as *const _);
    }
    fn set_f32(e: &metal::ComputeCommandEncoderRef, index: u64, value: &f32) {
        e.set_bytes(index, 4, value as *const f32 as *const _);
    }
    fn bytesread(buffer: &metal::Buffer, bytes: usize) -> ProbeResult<Vec<u8>> {
        if buffer.length() < bytes as u64 {
            return Err(failure("GPU byte read overflow"));
        }
        Ok(unsafe { std::slice::from_raw_parts(buffer.contents() as *const u8, bytes).to_vec() })
    }
    fn u16read(buffer: &metal::Buffer, count: usize) -> ProbeResult<Vec<u16>> {
        Ok(bytesread(buffer, count * 2)?
            .chunks_exact(2)
            .map(|x| u16::from_le_bytes([x[0], x[1]]))
            .collect())
    }
    fn f32read(buffer: &metal::Buffer, count: usize) -> ProbeResult<Vec<f32>> {
        if buffer.length() < count as u64 * 4 {
            return Err(failure("GPU f32 read overflow"));
        }
        let values =
            unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, count).to_vec() };
        if values.iter().any(|v| !v.is_finite()) {
            return Err(failure("GPU non-finite"));
        }
        Ok(values)
    }
    fn bf16_bits_f64(values: &[u16]) -> Vec<f64> {
        values
            .iter()
            .map(|&v| f32::from_bits((v as u32) << 16) as f64)
            .collect()
    }
    fn bf16_bytes_f64(values: &[u8]) -> ProbeResult<Vec<f64>> {
        if values.len() % 2 != 0 {
            return Err(failure("BF16 source bytes"));
        }
        Ok(values
            .chunks_exact(2)
            .map(|x| f32::from_bits((u16::from_le_bytes([x[0], x[1]]) as u32) << 16) as f64)
            .collect())
    }
    fn bf16f32(values: &[u16]) -> Vec<f32> {
        values
            .iter()
            .map(|&v| f32::from_bits((v as u32) << 16))
            .collect()
    }
    fn f32bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
    fn f32bytes_to_vec(values: &[u8]) -> ProbeResult<Vec<f32>> {
        if values.len() % 4 != 0 {
            return Err(failure("F32 source bytes"));
        }
        Ok(values
            .chunks_exact(4)
            .map(|x| f32::from_bits(u32::from_le_bytes([x[0], x[1], x[2], x[3]])))
            .collect())
    }
    fn f32bytes_f64(values: &[u8]) -> ProbeResult<Vec<f64>> {
        if values.len() % 4 != 0 {
            return Err(failure("F32 FP64 source bytes"));
        }
        Ok(values
            .chunks_exact(4)
            .map(|x| f32::from_bits(u32::from_le_bytes([x[0], x[1], x[2], x[3]])) as f64)
            .collect())
    }
    fn u16bytes(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }
    fn sha256_join(parts: &[&str]) -> String {
        let mut hash = Sha256::new();
        for part in parts {
            hash.update(part.as_bytes());
            hash.update([0]);
        }
        format!("{:x}", hash.finalize())
    }
    fn executable_provenance() -> ProbeResult<Value> {
        let raw_path = std::env::current_exe()?;
        if fs::symlink_metadata(&raw_path)?.file_type().is_symlink() {
            return Err(failure(
                "current executable symlink is not admitted for strict-DD v2 provenance",
            ));
        }
        let path = fs::canonicalize(raw_path)?;
        let metadata = fs::metadata(&path)?;
        if !metadata.is_file() || metadata.len() == 0 {
            return Err(failure(
                "current executable is not a nonempty regular file for strict-DD v2 provenance",
            ));
        }
        let build_profile = if path
            .components()
            .any(|component| component.as_os_str() == "release")
        {
            "release"
        } else if path
            .components()
            .any(|component| component.as_os_str() == "debug")
        {
            "debug"
        } else {
            return Err(failure(
                "cannot derive current executable build profile for strict-DD v2 provenance",
            ));
        };
        let executable_sha256 = sha256(&fs::read(&path)?);
        Ok(json!({
            "path":path.display().to_string(),
            "sha256":executable_sha256,
            "bytes":metadata.len(),
            "direct_executable_identity":true,
            "build_profile_inferred_from_executable_path":build_profile,
            "build_provenance_compiled_into_executable":{
                "cargo_manifest_dir":env!("CARGO_MANIFEST_DIR"),
                "cargo_package":env!("CARGO_PKG_NAME"),
                "cargo_package_version":env!("CARGO_PKG_VERSION"),
            },
        }))
    }
    fn source_code_inspection(executable: &Value) -> ProbeResult<Value> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = root
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| failure("cannot derive strict-DD v2 workspace root"))?;
        let relative_files = [
            "examples/gravity_deepseek_v4_p4b_position1_complete_attention_metal.rs",
            "src/lib.rs",
            "src/numeric_parity.rs",
            "src/gravity_deepseek_v4.rs",
            "src/gravity_deepseek_v4_act_quant.rs",
            "src/gravity_deepseek_v4_layer0_attention.rs",
            "src/gravity_deepseek_v4_layer0_continuation.rs",
            "src/gravity_deepseek_v4_layer0_prefix.rs",
            "src/metal/mod.rs",
            "shaders/matmul.metal",
        ];
        let mut files = serde_json::Map::new();
        for relative in relative_files {
            files.insert(
                relative.to_owned(),
                Value::String(sha256(&fs::read(root.join(relative))?)),
            );
        }
        let mut build_inputs = serde_json::Map::new();
        for (label, path) in [
            ("crates/hawking-core/Cargo.toml", root.join("Cargo.toml")),
            ("Cargo.toml", workspace_root.join("Cargo.toml")),
            ("Cargo.lock", workspace_root.join("Cargo.lock")),
        ] {
            build_inputs.insert(label.to_owned(), Value::String(sha256(&fs::read(path)?)));
        }
        let shader_source_sha256 = files
            .get("shaders/matmul.metal")
            .and_then(Value::as_str)
            .ok_or_else(|| failure("strict-DD shader source hash absent"))?
            .to_owned();
        let shader_embedded_sha256 = sha256(hawking_core::metal::SHADER_MATMUL.as_bytes());
        if shader_source_sha256 != shader_embedded_sha256 {
            return Err(failure(
                "embedded strict-DD shader differs from current matmul.metal",
            ));
        }
        let workspace_root_text = workspace_root
            .to_str()
            .ok_or_else(|| failure("strict-DD v2 workspace root is not UTF-8"))?;
        let checkout_revision = command_stdout(
            "git",
            &["-C", workspace_root_text, "rev-parse", "HEAD"],
            false,
        )?;
        let worktree_porcelain = command_stdout(
            "git",
            &["-C", workspace_root_text, "status", "--porcelain=v1"],
            true,
        )?;
        let executable_sha256 = executable
            .get("sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| failure("strict-DD v2 executable inspection has no SHA-256"))?;
        let executable_bytes = executable
            .get("bytes")
            .and_then(Value::as_u64)
            .ok_or_else(|| failure("strict-DD v2 executable inspection has no byte count"))?;
        Ok(json!({
            "attestation_scope":"The executable SHA-256 is the direct compiled-code identity. Current source/build hashes are runtime inspection only: they bind the inspected checkout to this sealed receipt, but do not assert that the mutable worktree produced the executable.",
            "bound_executable":{
                "sha256":executable_sha256,
                "bytes":executable_bytes,
                "build_provenance_location":"run_provenance.executable.build_provenance_compiled_into_executable"
            },
            "current_worktree_inspection_only":{
                "workspace_root":workspace_root.display().to_string(),
                "checkout_revision":checkout_revision,
                "worktree_porcelain_sha256":sha256(worktree_porcelain.as_bytes()),
                "execution_relevant_source_files_sha256":files,
                "build_input_files_sha256":build_inputs
            },
            "embedded_shader_identity":{
                "matmul_shader_embedded_sha256":shader_embedded_sha256,
                "current_matmul_shader_source_sha256":shader_source_sha256,
                "embedded_matches_current_source_at_inspection":true,
                "directly_embedded_in_executable":true
            }
        }))
    }
    fn command_stdout(program: &str, args: &[&str], allow_empty: bool) -> ProbeResult<String> {
        let output = Command::new(program).args(args).output()?;
        if !output.status.success() {
            return Err(failure(format!(
                "{program} {} failed with {}",
                args.join(" "),
                output.status
            )));
        }
        let text = String::from_utf8(output.stdout)?;
        let trimmed = text.trim().to_owned();
        if !allow_empty && trimmed.is_empty() {
            return Err(failure(format!(
                "{program} {} returned empty output",
                args.join(" ")
            )));
        }
        Ok(trimmed)
    }
    fn host_platform_provenance() -> ProbeResult<Value> {
        Ok(json!({
            "operating_system":std::env::consts::OS,
            "architecture":command_stdout("uname", &["-m"], false)?,
            "kernel_release":command_stdout("uname", &["-r"], false)?,
            "macos_product_version":command_stdout("sw_vers", &["-productVersion"], false)?,
            "macos_build_version":command_stdout("sw_vers", &["-buildVersion"], false)?,
        }))
    }
    fn secure_run_nonce() -> ProbeResult<String> {
        let mut entropy = [0u8; 32];
        File::open("/dev/urandom")?.read_exact(&mut entropy)?;
        Ok(sha256(&entropy))
    }
    fn unix_time_ns() -> ProbeResult<String> {
        Ok(SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| failure(format!("system clock before Unix epoch: {error}")))?
            .as_nanos()
            .to_string())
    }
    fn ensure_unix_ns_not_before(start: &str, finish: &str) -> ProbeResult<()> {
        let start = start.parse::<u128>().map_err(|error| {
            failure(format!(
                "strict-DD v2 start timestamp is not decimal ns: {error}"
            ))
        })?;
        let finish = finish.parse::<u128>().map_err(|error| {
            failure(format!(
                "strict-DD v2 finish timestamp is not decimal ns: {error}"
            ))
        })?;
        if finish < start {
            return Err(failure(
                "strict-DD v2 finish timestamp precedes its start timestamp",
            ));
        }
        Ok(())
    }
    fn text<'a>(value: &'a Value, path: &[&str]) -> ProbeResult<&'a str> {
        let mut current = value;
        for key in path {
            current = current
                .get(*key)
                .ok_or_else(|| failure(format!("missing {}", path.join("."))))?;
        }
        current
            .as_str()
            .ok_or_else(|| failure(format!("not text {}", path.join("."))))
    }
    fn seal_ok(value: &Value) -> ProbeResult<()> {
        let recorded = text(value, &["seal_sha256"])?;
        let mut unsigned = value.clone();
        unsigned
            .as_object_mut()
            .ok_or_else(|| failure("receipt object"))?
            .remove("seal_sha256");
        if sha256(&canonical(&unsigned)) != recorded {
            return Err(failure("predecessor seal mismatch"));
        }
        Ok(())
    }
    fn canonical(value: &Value) -> Vec<u8> {
        let mut out = Vec::new();
        canon(&mut out, value);
        out
    }
    fn canon(out: &mut Vec<u8>, value: &Value) {
        match value {
            Value::Null => out.extend_from_slice(b"null"),
            Value::Bool(v) => out.extend_from_slice(v.to_string().as_bytes()),
            Value::Number(v) => out.extend_from_slice(v.to_string().as_bytes()),
            Value::String(v) => {
                out.extend_from_slice(serde_json::to_string(v).expect("JSON string").as_bytes())
            }
            Value::Array(values) => {
                out.push(b'[');
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        out.push(b',');
                    }
                    canon(out, value);
                }
                out.push(b']');
            }
            Value::Object(values) => {
                let mut keys = values.keys().collect::<Vec<_>>();
                keys.sort();
                out.push(b'{');
                for (index, key) in keys.into_iter().enumerate() {
                    if index != 0 {
                        out.push(b',');
                    }
                    out.extend_from_slice(serde_json::to_string(key).expect("JSON key").as_bytes());
                    out.push(b':');
                    canon(out, &values[key]);
                }
                out.push(b'}');
            }
        }
    }
    fn decimal_strings(value: Value) -> Value {
        match value {
            Value::Number(n) if n.is_i64() || n.is_u64() => Value::Number(n),
            Value::Number(n) => Value::String(n.to_string()),
            Value::Array(values) => Value::Array(values.into_iter().map(decimal_strings).collect()),
            Value::Object(values) => Value::Object(
                values
                    .into_iter()
                    .map(|(k, v)| (k, decimal_strings(v)))
                    .collect(),
            ),
            other => other,
        }
    }
    fn seal(mut value: Value) -> ProbeResult<(Value, String)> {
        if !value.is_object() || value.get("seal_sha256").is_some() {
            return Err(failure("unsealed receipt required"));
        }
        let hash = sha256(&canonical(&value));
        value
            .as_object_mut()
            .expect("object")
            .insert("seal_sha256".into(), Value::String(hash.clone()));
        Ok((value, hash))
    }
    fn write_new(path: &Path, value: &Value) -> ProbeResult<()> {
        if path.exists() {
            return Err(failure(format!("refusing overwrite {}", path.display())));
        }
        let parent = path.parent().ok_or_else(|| failure("out parent"))?;
        fs::create_dir_all(parent)?;
        let name = path
            .file_name()
            .and_then(|x| x.to_str())
            .ok_or_else(|| failure("out UTF8"))?;
        let temp = parent.join(format!(".{name}.{}.p4b.tmp", std::process::id()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        if let Err(error) = file
            .write_all(&serde_json::to_vec_pretty(value)?)
            .and_then(|_| file.write_all(b"\n"))
            .and_then(|_| file.sync_all())
        {
            let _ = fs::remove_file(&temp);
            return Err(Box::new(error));
        }
        drop(file);
        if let Err(error) = fs::hard_link(&temp, path) {
            let _ = fs::remove_file(&temp);
            return Err(failure(format!("link receipt: {error}")));
        }
        fs::remove_file(temp)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    }
    fn write_new_and_verify_canonical(path: &Path, value: &Value) -> ProbeResult<String> {
        // Verify the in-memory seal first, then write and reparse a unique
        // temporary file before publishing the final create-new name.  The
        // final name is a hard link to that verified file, so a canonical
        // reread failure cannot leave a final named v2 receipt behind.
        seal_ok(value)?;
        let expected_canonical = canonical(value);
        let expected_seal = text(value, &["seal_sha256"])?.to_owned();
        let mut serialized = serde_json::to_vec_pretty(value)?;
        serialized.push(b'\n');
        if path.exists() {
            return Err(failure(format!("refusing overwrite {}", path.display())));
        }
        let parent = path.parent().ok_or_else(|| failure("out parent"))?;
        fs::create_dir_all(parent)?;
        let name = path
            .file_name()
            .and_then(|x| x.to_str())
            .ok_or_else(|| failure("out UTF8"))?;
        let temp = parent.join(format!(
            ".{name}.{}.{}.p4b.strict-dd-v2.tmp",
            std::process::id(),
            secure_run_nonce()?
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        if let Err(error) = file.write_all(&serialized).and_then(|_| file.sync_all()) {
            let _ = fs::remove_file(&temp);
            return Err(Box::new(error));
        }
        drop(file);

        let verified_temp = (|| -> ProbeResult<String> {
            let raw = fs::read(&temp)?;
            let parsed: Value = serde_json::from_slice(&raw)?;
            seal_ok(&parsed)?;
            if text(&parsed, &["seal_sha256"])? != expected_seal
                || canonical(&parsed) != expected_canonical
            {
                return Err(failure(
                    "temporary strict-DD v2 receipt differs from canonical sealed memory image",
                ));
            }
            Ok(sha256(&raw))
        })();
        let receipt_file_sha256 = match verified_temp {
            Ok(sha256) => sha256,
            Err(error) => {
                let _ = fs::remove_file(&temp);
                return Err(error);
            }
        };

        // Check again immediately before the create-only link to close the
        // ordinary preflight race without ever replacing an existing receipt.
        if path.exists() {
            let _ = fs::remove_file(&temp);
            return Err(failure(format!("refusing overwrite {}", path.display())));
        }
        if let Err(error) = fs::hard_link(&temp, path) {
            let _ = fs::remove_file(&temp);
            return Err(failure(format!(
                "link verified strict-DD v2 receipt: {error}"
            )));
        }
        fs::remove_file(&temp)?;
        File::open(parent)?.sync_all()?;
        Ok(receipt_file_sha256)
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

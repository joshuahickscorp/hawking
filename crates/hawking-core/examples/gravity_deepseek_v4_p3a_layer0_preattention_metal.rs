//! Bounded native-Metal P3A checkpoint for DeepSeek-V4-Flash layer 0.
//!
//! This executable intentionally stops before KV, attention, routing, MoE,
//! token sampling, a decode loop, or HCLI.  It proves a much narrower real
//! device chain, using only sealed streamed source payloads:
//!
//! ```text
//! tokenizer-bound BOS / position 0 / ratio 0
//!   embed -> mHC pre + Sinkhorn -> attn RMSNorm
//!   -> source QAT -> WQ-A -> Q RMSNorm
//!   -> source QAT -> WQ-B -> BF16 copy -> per-head Q RMSNorm
//! ```
//!
//! CPU is used solely for the separately sealed source-algorithm oracle and
//! after-completion comparison.  Device intermediate buffers remain on GPU
//! across the chain; there is no host activation/routing fallback.
//!
//! ```sh
//! cargo run --release -p hawking-core --example gravity_deepseek_v4_p3a_layer0_preattention_metal -- \
//!   --artifact /absolute/path/to/full-43-layer-stream.gravity \
//!   --cpu-attention-oracle /absolute/path/to/DSV4F_LAYER0_ATTENTION_CPU_ORACLE-v1.json \
//!   --out /absolute/path/to/DSV4F_P3A_LAYER0_PREATTENTION_METAL-v1.json
//! ```

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "gravity_deepseek_v4_p3a_layer0_preattention_metal requires macOS Metal",
    )
    .into())
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::gravity_deepseek_v4::{
        DeepSeekV4FullStreamReader, DeepSeekV4TensorMetadata, NativeScalePairKind,
        FULL_STREAM_SCHEMA, FULL_STREAM_STATUS,
    };
    use hawking_core::gravity_deepseek_v4_act_quant::{
        ACT_QUANT_BLOCK, LAYER0_WQ_A_COLS, LAYER0_WQ_A_ROWS, LAYER0_WQ_A_SCALE, LAYER0_WQ_A_WEIGHT,
    };
    use hawking_core::gravity_deepseek_v4_layer0_attention::{
        layer0_attention_cpu_oracle, verify_layer0_attention_source_anchors, LAYER0_Q_NORM_WEIGHT,
        LAYER0_WQ_B_SCALE, LAYER0_WQ_B_WEIGHT, NUM_HEADS, Q_LORA_RANK, WQ_B_ROWS,
    };
    use hawking_core::gravity_deepseek_v4_layer0_prefix::{
        EMBED_WEIGHT, HC_EPS, HC_FLAT_WIDTH, HC_MIX_WIDTH, HC_MULT, HC_SINKHORN_ITERS, HIDDEN_SIZE,
        LAYER0_ATTN_NORM_WEIGHT, LAYER0_HC_ATTN_BASE, LAYER0_HC_ATTN_FN, LAYER0_HC_ATTN_SCALE,
        PREFIX_TOKEN_ID, RMS_NORM_EPS,
    };
    use hawking_core::metal::{
        MetalContext, MetalDispatchTiming, PhysicalTraceGuard, PhysicalTraceIdentity,
    };
    use hawking_core::numeric_parity::{rmsnorm_f64, score_pair, Bounds, PairedScore};
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::error::Error;
    use std::fs::{self, File, OpenOptions};
    use std::io::Write;
    use std::path::{Path, PathBuf};

    const RECEIPT_SCHEMA: &str = "hawking.gravity.deepseek_v4.p3a_layer0_preattention_metal.v1";
    const RECEIPT_STATUS: &str = "PASS_REAL_METAL_P3A_LAYER0_PREATTENTION_PARITY_NOT_RUNTIME";
    const CPU_ORACLE_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.layer0_attention_cpu_algorithm_oracle.v1";
    const CPU_ORACLE_STATUS: &str =
        "PASS_SOURCE_DERIVED_CPU_LAYER0_ATTENTION_NOT_INDEPENDENT_UPSTREAM_RUNTIME_PARITY";
    const CPU_ORACLE_BASENAME: &str = "DSV4F_LAYER0_ATTENTION_CPU_ORACLE-v1.json";

    const HC_KERNEL: &str = "deepseek_v4_p3a_layer0_hc_attn_pre_bos_authority";
    const RMS_KERNEL: &str = "deepseek_v4_p3a_rmsnorm_bf16_authority";
    const QAT_KERNEL: &str = "deepseek_v4_act_quant_bf16_ue8m0_authority";
    const FP8_KERNEL: &str = "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority";
    const CAST_KERNEL: &str = "deepseek_v4_p3a_fp32_to_bf16_authority";
    const PER_HEAD_KERNEL: &str = "deepseek_v4_p3a_per_head_rmsnorm_bf16_authority";
    const P3A_DISPATCHES: u64 = 10;

    type ProbeResult<T> = Result<T, Box<dyn Error>>;

    struct Args {
        artifact: PathBuf,
        cpu_attention_oracle: PathBuf,
        out: PathBuf,
    }

    struct CpuOracleBinding {
        path: PathBuf,
        file_sha256: String,
        seal_sha256: String,
    }

    struct HcF64Reference {
        flat_rsqrt: f64,
        mixes: Vec<f64>,
        pre: Vec<f64>,
        post: Vec<f64>,
        comb: Vec<f64>,
        reduced: Vec<f64>,
    }

    fn failure(message: impl Into<String>) -> Box<dyn Error> {
        std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
    }

    pub fn run() -> ProbeResult<()> {
        let args = parse_args()?;
        let reader = DeepSeekV4FullStreamReader::admit(&args.artifact)?;
        let anchors = verify_layer0_attention_source_anchors(&reader)?;
        if reader.source_identity().repository != "deepseek-ai/DeepSeek-V4-Flash"
            || reader.source_identity().revision != "60d8d70770c6776ff598c94bb586a859a38244f1"
        {
            return Err(failure(
                "P3A reader did not admit the pinned DeepSeek-V4-Flash source identity",
            ));
        }

        let embed_meta = expect_tensor(
            &reader,
            EMBED_WEIGHT,
            "BF16",
            &[129_280, HIDDEN_SIZE as u64],
        )?;
        let hc_fn_meta = expect_tensor(
            &reader,
            LAYER0_HC_ATTN_FN,
            "F32",
            &[HC_MIX_WIDTH as u64, HC_FLAT_WIDTH as u64],
        )?;
        let hc_base_meta =
            expect_tensor(&reader, LAYER0_HC_ATTN_BASE, "F32", &[HC_MIX_WIDTH as u64])?;
        let hc_scale_meta = expect_tensor(&reader, LAYER0_HC_ATTN_SCALE, "F32", &[3])?;
        let attn_norm_meta = expect_tensor(
            &reader,
            LAYER0_ATTN_NORM_WEIGHT,
            "BF16",
            &[HIDDEN_SIZE as u64],
        )?;
        let q_norm_meta =
            expect_tensor(&reader, LAYER0_Q_NORM_WEIGHT, "BF16", &[Q_LORA_RANK as u64])?;
        let (wq_a_meta, wq_a_scale_meta) = fp8_pair_metadata(
            &reader,
            LAYER0_WQ_A_WEIGHT,
            LAYER0_WQ_A_SCALE,
            LAYER0_WQ_A_ROWS,
            LAYER0_WQ_A_COLS,
        )?;
        let (wq_b_meta, wq_b_scale_meta) = fp8_pair_metadata(
            &reader,
            LAYER0_WQ_B_WEIGHT,
            LAYER0_WQ_B_SCALE,
            WQ_B_ROWS,
            Q_LORA_RANK,
        )?;

        // The CPU algorithm oracle is an independently sealed predecessor
        // binding, not the numerical authority.  We recompute it directly
        // from the streamed source before doing any GPU work, and then score
        // both source-CPU and Metal f32 outputs against separate f64 paths.
        let cpu = layer0_attention_cpu_oracle(&reader)?;
        let cpu_oracle = validate_cpu_attention_oracle(&args.cpu_attention_oracle, &reader, &cpu)?;

        let row_bytes = HIDDEN_SIZE
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| failure("BOS embedding row byte count overflow"))?;
        let embed = reader.read_verified_range(EMBED_WEIGHT, 0..row_bytes as u64, row_bytes)?;
        let hc_fn = reader.read_verified_full(LAYER0_HC_ATTN_FN, hc_fn_meta.bytes as usize)?;
        let hc_base =
            reader.read_verified_full(LAYER0_HC_ATTN_BASE, hc_base_meta.bytes as usize)?;
        let hc_scale =
            reader.read_verified_full(LAYER0_HC_ATTN_SCALE, hc_scale_meta.bytes as usize)?;
        let attn_norm =
            reader.read_verified_full(LAYER0_ATTN_NORM_WEIGHT, attn_norm_meta.bytes as usize)?;
        let q_norm = reader.read_verified_full(LAYER0_Q_NORM_WEIGHT, q_norm_meta.bytes as usize)?;
        let wq_a_weight =
            reader.read_verified_full(LAYER0_WQ_A_WEIGHT, wq_a_meta.bytes as usize)?;
        let wq_a_scale =
            reader.read_verified_full(LAYER0_WQ_A_SCALE, wq_a_scale_meta.bytes as usize)?;
        let wq_b_weight =
            reader.read_verified_full(LAYER0_WQ_B_WEIGHT, wq_b_meta.bytes as usize)?;
        let wq_b_scale =
            reader.read_verified_full(LAYER0_WQ_B_SCALE, wq_b_scale_meta.bytes as usize)?;
        if embed.len() != row_bytes
            || hc_fn.len() != HC_MIX_WIDTH * HC_FLAT_WIDTH * std::mem::size_of::<f32>()
            || hc_base.len() != HC_MIX_WIDTH * std::mem::size_of::<f32>()
            || hc_scale.len() != 3 * std::mem::size_of::<f32>()
            || attn_norm.len() != row_bytes
            || q_norm.len() != Q_LORA_RANK * std::mem::size_of::<u16>()
            || wq_a_weight.len() != LAYER0_WQ_A_ROWS * LAYER0_WQ_A_COLS
            || wq_a_scale.len()
                != (LAYER0_WQ_A_ROWS / ACT_QUANT_BLOCK) * (LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK)
            || wq_b_weight.len() != WQ_B_ROWS * Q_LORA_RANK
            || wq_b_scale.len() != (WQ_B_ROWS / ACT_QUANT_BLOCK) * (Q_LORA_RANK / ACT_QUANT_BLOCK)
        {
            return Err(failure(
                "P3A bounded source uploads changed from the pinned geometry",
            ));
        }

        let hc_reference = hc_attn_pre_f64_reference(&embed, &hc_fn, &hc_scale, &hc_base)?;
        let attn_norm_reference = rmsnorm_f64(
            &bf16_le_f64(&u16_le_bytes(&cpu.prefix.hc_attn_pre_bf16_bits))?,
            &bf16_le_f64(&attn_norm)?,
            RMS_NORM_EPS as f64,
        )
        .map_err(failure)?;
        let wq_a_reference = fp8_linear_f64_reference(
            &wq_a_weight,
            &wq_a_scale,
            &cpu.wq_a.quantized_input.activation_e4m3fn,
            &cpu.wq_a.quantized_input.scales_e8m0fnu,
            LAYER0_WQ_A_ROWS,
            LAYER0_WQ_A_COLS,
        )?;
        let q_norm_reference = rmsnorm_f64(
            &bf16_le_f64(&u16_le_bytes(&cpu.wq_a.output.bf16_bits))?,
            &bf16_le_f64(&q_norm)?,
            RMS_NORM_EPS as f64,
        )
        .map_err(failure)?;
        let wq_b_reference = fp8_linear_f64_reference(
            &wq_b_weight,
            &wq_b_scale,
            &cpu.wq_b.quantized_input.activation_e4m3fn,
            &cpu.wq_b.quantized_input.scales_e8m0fnu,
            WQ_B_ROWS,
            Q_LORA_RANK,
        )?;
        let q_head_reference = per_head_rmsnorm_f64(
            &bf16_le_f64(&u16_le_bytes(&cpu.wq_b.output.bf16_bits))?,
            NUM_HEADS,
            WQ_B_ROWS / NUM_HEADS,
            RMS_NORM_EPS as f64,
        )?;

        let context = MetalContext::new_with_trace(true)?;
        let device_name = context.device_name();
        let mut pipeline_widths = serde_json::Map::new();
        for kernel in [
            HC_KERNEL,
            RMS_KERNEL,
            QAT_KERNEL,
            FP8_KERNEL,
            CAST_KERNEL,
            PER_HEAD_KERNEL,
        ] {
            let pipeline = context.pipeline(kernel)?;
            pipeline_widths.insert(
                kernel.to_owned(),
                json!({
                    "thread_execution_width": pipeline.thread_execution_width(),
                    "max_total_threads_per_threadgroup": pipeline.max_total_threads_per_threadgroup(),
                }),
            );
        }

        // All stages below receive source bytes or previous device buffers;
        // no output is read or CPU-copied until every dispatch has completed.
        let embed_buffer = context.new_buffer_with_bytes_checked(&embed)?;
        let hc_fn_buffer = context.new_buffer_with_bytes_checked(&hc_fn)?;
        let hc_scale_buffer = context.new_buffer_with_bytes_checked(&hc_scale)?;
        let hc_base_buffer = context.new_buffer_with_bytes_checked(&hc_base)?;
        let hc_reduced_buffer = context.new_buffer_checked(row_bytes)?;
        let hc_rsqrt_buffer = context.new_buffer_checked(std::mem::size_of::<f32>())?;
        let hc_mixes_buffer =
            context.new_buffer_checked(HC_MIX_WIDTH * std::mem::size_of::<f32>())?;
        let hc_pre_buffer = context.new_buffer_checked(HC_MULT * std::mem::size_of::<f32>())?;
        let hc_post_buffer = context.new_buffer_checked(HC_MULT * std::mem::size_of::<f32>())?;
        let hc_comb_buffer =
            context.new_buffer_checked(HC_MULT * HC_MULT * std::mem::size_of::<f32>())?;
        let attn_norm_weight_buffer = context.new_buffer_with_bytes_checked(&attn_norm)?;
        let attn_norm_output_buffer = context.new_buffer_checked(row_bytes)?;
        let wq_a_weight_buffer = context.new_buffer_with_bytes_checked(&wq_a_weight)?;
        let wq_a_scale_buffer = context.new_buffer_with_bytes_checked(&wq_a_scale)?;
        let wq_a_activation_buffer = context.new_buffer_checked(LAYER0_WQ_A_COLS)?;
        let wq_a_activation_scale_buffer =
            context.new_buffer_checked(LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK)?;
        let wq_a_fp32_output_buffer =
            context.new_buffer_checked(LAYER0_WQ_A_ROWS * std::mem::size_of::<f32>())?;
        let wq_a_bf16_output_buffer =
            context.new_buffer_checked(LAYER0_WQ_A_ROWS * std::mem::size_of::<u16>())?;
        let q_norm_weight_buffer = context.new_buffer_with_bytes_checked(&q_norm)?;
        let q_norm_output_buffer =
            context.new_buffer_checked(Q_LORA_RANK * std::mem::size_of::<u16>())?;
        let wq_b_weight_buffer = context.new_buffer_with_bytes_checked(&wq_b_weight)?;
        let wq_b_scale_buffer = context.new_buffer_with_bytes_checked(&wq_b_scale)?;
        let wq_b_activation_buffer = context.new_buffer_checked(Q_LORA_RANK)?;
        let wq_b_activation_scale_buffer =
            context.new_buffer_checked(Q_LORA_RANK / ACT_QUANT_BLOCK)?;
        let wq_b_fp32_output_buffer =
            context.new_buffer_checked(WQ_B_ROWS * std::mem::size_of::<f32>())?;
        let wq_b_bf16_output_buffer =
            context.new_buffer_checked(WQ_B_ROWS * std::mem::size_of::<u16>())?;
        let q_head_output_buffer =
            context.new_buffer_checked(WQ_B_ROWS * std::mem::size_of::<u16>())?;

        let run_nonce = sha256_join(&[
            reader.manifest_seal_sha256(),
            &cpu_oracle.seal_sha256,
            &sha256(&embed),
            &sha256(&wq_a_weight),
            &sha256(&wq_b_weight),
            "dsv4f_p3a_layer0_preattention_metal_v1",
        ]);
        let interval_id = sha256_join(&[&run_nonce, "p3a_mhc_q_projection_chain"]);
        let physical_trace = PhysicalTraceGuard::begin(PhysicalTraceIdentity::new(
            interval_id.clone(),
            run_nonce.clone(),
            "dsv4f_p3a_layer0_preattention".to_owned(),
            "mhc_norm_q_projection".to_owned(),
            Some(1),
            0,
        )?)?;

        let hidden = HIDDEN_SIZE as u32;
        let hc_mult = HC_MULT as u32;
        let mix_width = HC_MIX_WIDTH as u32;
        let sinkhorn_iters = HC_SINKHORN_ITERS as u32;
        let norm_eps = RMS_NORM_EPS;
        let hc_eps = HC_EPS;
        let q_lora_rank = Q_LORA_RANK as u32;
        let heads = NUM_HEADS as u32;
        let head_dim = (WQ_B_ROWS / NUM_HEADS) as u32;
        let wq_a_rows = LAYER0_WQ_A_ROWS as u32;
        let wq_a_cols = LAYER0_WQ_A_COLS as u32;
        let wq_a_scale_cols = (LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK) as u32;
        let wq_b_rows = WQ_B_ROWS as u32;
        let wq_b_cols = Q_LORA_RANK as u32;
        let wq_b_scale_cols = (Q_LORA_RANK / ACT_QUANT_BLOCK) as u32;

        let hc_timing =
            context.dispatch_threads_timed(HC_KERNEL, (1, 1, 1), (1, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(&embed_buffer), 0);
                encoder.set_buffer(1, Some(&hc_fn_buffer), 0);
                encoder.set_buffer(2, Some(&hc_scale_buffer), 0);
                encoder.set_buffer(3, Some(&hc_base_buffer), 0);
                encoder.set_buffer(4, Some(&hc_reduced_buffer), 0);
                encoder.set_buffer(5, Some(&hc_rsqrt_buffer), 0);
                encoder.set_buffer(6, Some(&hc_mixes_buffer), 0);
                encoder.set_buffer(7, Some(&hc_pre_buffer), 0);
                encoder.set_buffer(8, Some(&hc_post_buffer), 0);
                encoder.set_buffer(9, Some(&hc_comb_buffer), 0);
                set_u32(encoder, 10, &hidden);
                set_u32(encoder, 11, &hc_mult);
                set_u32(encoder, 12, &mix_width);
                set_u32(encoder, 13, &sinkhorn_iters);
                set_f32(encoder, 14, &norm_eps);
                set_f32(encoder, 15, &hc_eps);
            })?;
        require_completed_dispatch(&hc_timing, "GPU mHC pre")?;

        let attn_norm_timing =
            context.dispatch_threads_timed(RMS_KERNEL, (1, 1, 1), (1, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(&hc_reduced_buffer), 0);
                encoder.set_buffer(1, Some(&attn_norm_weight_buffer), 0);
                encoder.set_buffer(2, Some(&attn_norm_output_buffer), 0);
                set_u32(encoder, 3, &hidden);
                set_f32(encoder, 4, &norm_eps);
            })?;
        require_completed_dispatch(&attn_norm_timing, "GPU attention RMSNorm")?;

        let wq_a_qat_timing = context.dispatch_threads_timed(
            QAT_KERNEL,
            (wq_a_scale_cols, 1, 1),
            (32, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&attn_norm_output_buffer), 0);
                encoder.set_buffer(1, Some(&wq_a_activation_buffer), 0);
                encoder.set_buffer(2, Some(&wq_a_activation_scale_buffer), 0);
                set_u32(encoder, 3, &wq_a_cols);
            },
        )?;
        require_completed_dispatch(&wq_a_qat_timing, "GPU WQ-A source QAT")?;

        let wq_a_timing = context.dispatch_threads_timed(
            FP8_KERNEL,
            (wq_a_rows, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&wq_a_weight_buffer), 0);
                encoder.set_buffer(1, Some(&wq_a_scale_buffer), 0);
                encoder.set_buffer(2, Some(&wq_a_activation_buffer), 0);
                encoder.set_buffer(3, Some(&wq_a_activation_scale_buffer), 0);
                encoder.set_buffer(4, Some(&wq_a_fp32_output_buffer), 0);
                set_u32(encoder, 5, &wq_a_rows);
                set_u32(encoder, 6, &wq_a_cols);
                set_u32(encoder, 7, &wq_a_scale_cols);
            },
        )?;
        require_completed_dispatch(&wq_a_timing, "GPU WQ-A FP8 projection")?;

        let wq_a_cast_timing = context.dispatch_threads_timed(
            CAST_KERNEL,
            (wq_a_rows, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&wq_a_fp32_output_buffer), 0);
                encoder.set_buffer(1, Some(&wq_a_bf16_output_buffer), 0);
                set_u32(encoder, 2, &wq_a_rows);
            },
        )?;
        require_completed_dispatch(&wq_a_cast_timing, "GPU WQ-A FP32-to-BF16 handoff")?;

        let q_norm_timing =
            context.dispatch_threads_timed(RMS_KERNEL, (1, 1, 1), (1, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(&wq_a_bf16_output_buffer), 0);
                encoder.set_buffer(1, Some(&q_norm_weight_buffer), 0);
                encoder.set_buffer(2, Some(&q_norm_output_buffer), 0);
                set_u32(encoder, 3, &q_lora_rank);
                set_f32(encoder, 4, &norm_eps);
            })?;
        require_completed_dispatch(&q_norm_timing, "GPU Q RMSNorm")?;

        let wq_b_qat_timing = context.dispatch_threads_timed(
            QAT_KERNEL,
            (wq_b_scale_cols, 1, 1),
            (32, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&q_norm_output_buffer), 0);
                encoder.set_buffer(1, Some(&wq_b_activation_buffer), 0);
                encoder.set_buffer(2, Some(&wq_b_activation_scale_buffer), 0);
                set_u32(encoder, 3, &wq_b_cols);
            },
        )?;
        require_completed_dispatch(&wq_b_qat_timing, "GPU WQ-B source QAT")?;

        let wq_b_timing = context.dispatch_threads_timed(
            FP8_KERNEL,
            (wq_b_rows, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&wq_b_weight_buffer), 0);
                encoder.set_buffer(1, Some(&wq_b_scale_buffer), 0);
                encoder.set_buffer(2, Some(&wq_b_activation_buffer), 0);
                encoder.set_buffer(3, Some(&wq_b_activation_scale_buffer), 0);
                encoder.set_buffer(4, Some(&wq_b_fp32_output_buffer), 0);
                set_u32(encoder, 5, &wq_b_rows);
                set_u32(encoder, 6, &wq_b_cols);
                set_u32(encoder, 7, &wq_b_scale_cols);
            },
        )?;
        require_completed_dispatch(&wq_b_timing, "GPU WQ-B FP8 projection")?;

        let wq_b_cast_timing = context.dispatch_threads_timed(
            CAST_KERNEL,
            (wq_b_rows, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&wq_b_fp32_output_buffer), 0);
                encoder.set_buffer(1, Some(&wq_b_bf16_output_buffer), 0);
                set_u32(encoder, 2, &wq_b_rows);
            },
        )?;
        require_completed_dispatch(&wq_b_cast_timing, "GPU WQ-B FP32-to-BF16 handoff")?;

        let q_head_timing = context.dispatch_threads_timed(
            PER_HEAD_KERNEL,
            (heads, 1, 1),
            (64, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&wq_b_bf16_output_buffer), 0);
                encoder.set_buffer(1, Some(&q_head_output_buffer), 0);
                set_u32(encoder, 2, &heads);
                set_u32(encoder, 3, &head_dim);
                set_f32(encoder, 4, &norm_eps);
            },
        )?;
        require_completed_dispatch(&q_head_timing, "GPU per-head Q RMSNorm")?;

        let physical_counts = physical_trace.counts();
        drop(physical_trace);
        let (buffers_created, bytes_allocated, commits) = context.drain_stats();
        let trace_samples = context.drain_trace();
        if physical_counts.command_count != P3A_DISPATCHES
            || physical_counts.encoder_count != P3A_DISPATCHES
            || commits as u64 != P3A_DISPATCHES
            || trace_samples.len() as u64 != P3A_DISPATCHES
        {
            return Err(failure("P3A physical command/encoder/trace accounting changed from ten timestamped device stages"));
        }

        let gpu_hc_reduced = read_gpu_u16(&hc_reduced_buffer, HIDDEN_SIZE)?;
        let gpu_hc_rsqrt = read_gpu_f32(&hc_rsqrt_buffer, 1)?;
        let gpu_hc_mixes = read_gpu_f32(&hc_mixes_buffer, HC_MIX_WIDTH)?;
        let gpu_hc_pre = read_gpu_f32(&hc_pre_buffer, HC_MULT)?;
        let gpu_hc_post = read_gpu_f32(&hc_post_buffer, HC_MULT)?;
        let gpu_hc_comb = read_gpu_f32(&hc_comb_buffer, HC_MULT * HC_MULT)?;
        let gpu_attn_norm = read_gpu_u16(&attn_norm_output_buffer, HIDDEN_SIZE)?;
        let gpu_wq_a_activation = read_gpu_bytes(&wq_a_activation_buffer, LAYER0_WQ_A_COLS)?;
        let gpu_wq_a_activation_scales = read_gpu_bytes(
            &wq_a_activation_scale_buffer,
            LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK,
        )?;
        let gpu_wq_a_fp32 = read_gpu_f32(&wq_a_fp32_output_buffer, LAYER0_WQ_A_ROWS)?;
        let gpu_wq_a_bf16 = read_gpu_u16(&wq_a_bf16_output_buffer, LAYER0_WQ_A_ROWS)?;
        let gpu_q_norm = read_gpu_u16(&q_norm_output_buffer, Q_LORA_RANK)?;
        let gpu_wq_b_activation = read_gpu_bytes(&wq_b_activation_buffer, Q_LORA_RANK)?;
        let gpu_wq_b_activation_scales =
            read_gpu_bytes(&wq_b_activation_scale_buffer, Q_LORA_RANK / ACT_QUANT_BLOCK)?;
        let gpu_wq_b_fp32 = read_gpu_f32(&wq_b_fp32_output_buffer, WQ_B_ROWS)?;
        let gpu_wq_b_bf16 = read_gpu_u16(&wq_b_bf16_output_buffer, WQ_B_ROWS)?;
        let gpu_q_head = read_gpu_u16(&q_head_output_buffer, WQ_B_ROWS)?;

        exact_u16(
            "mHC reduced BF16",
            &cpu.prefix.hc_attn_pre_bf16_bits,
            &gpu_hc_reduced,
        )?;
        exact_f32(
            "mHC flat rsqrt",
            &[cpu.prefix.hc_flat_rsqrt],
            &gpu_hc_rsqrt,
            1.0e-5,
            1.0e-5,
        )?;
        exact_f32(
            "mHC mixes",
            &cpu.prefix.hc_mixes_f32,
            &gpu_hc_mixes,
            1.0e-4,
            1.0e-4,
        )?;
        exact_f32(
            "mHC pre",
            &cpu.prefix.hc_pre_f32,
            &gpu_hc_pre,
            1.0e-5,
            1.0e-5,
        )?;
        exact_f32(
            "mHC post",
            &cpu.prefix.hc_post_f32,
            &gpu_hc_post,
            1.0e-5,
            1.0e-5,
        )?;
        exact_f32(
            "mHC comb",
            &cpu.prefix.hc_comb_f32,
            &gpu_hc_comb,
            1.0e-5,
            1.0e-5,
        )?;
        exact_u16(
            "attention RMSNorm BF16",
            &cpu.prefix.attn_norm_bf16_bits,
            &gpu_attn_norm,
        )?;
        exact_bytes(
            "WQ-A source QAT activation",
            &cpu.wq_a.quantized_input.activation_e4m3fn,
            &gpu_wq_a_activation,
        )?;
        exact_bytes(
            "WQ-A source QAT scale",
            &cpu.wq_a.quantized_input.scales_e8m0fnu,
            &gpu_wq_a_activation_scales,
        )?;
        exact_f32(
            "WQ-A FP32",
            &cpu.wq_a.output.fp32,
            &gpu_wq_a_fp32,
            1.0e-4,
            1.0e-4,
        )?;
        exact_u16(
            "WQ-A BF16 handoff",
            &cpu.wq_a.output.bf16_bits,
            &gpu_wq_a_bf16,
        )?;
        exact_u16("Q RMSNorm BF16", &cpu.q_norm_bf16_bits, &gpu_q_norm)?;
        exact_bytes(
            "WQ-B source QAT activation",
            &cpu.wq_b.quantized_input.activation_e4m3fn,
            &gpu_wq_b_activation,
        )?;
        exact_bytes(
            "WQ-B source QAT scale",
            &cpu.wq_b.quantized_input.scales_e8m0fnu,
            &gpu_wq_b_activation_scales,
        )?;
        exact_f32(
            "WQ-B FP32",
            &cpu.wq_b.output.fp32,
            &gpu_wq_b_fp32,
            1.0e-4,
            1.0e-4,
        )?;
        exact_u16(
            "WQ-B BF16 handoff",
            &cpu.wq_b.output.bf16_bits,
            &gpu_wq_b_bf16,
        )?;
        exact_u16(
            "per-head Q RMSNorm BF16",
            &cpu.q_head_norm_bf16_bits,
            &gpu_q_head,
        )?;
        exact_u16(
            "position-zero RoPE identity",
            &gpu_q_head,
            &cpu.q_position0_rope_bf16_bits,
        )?;

        let f32_bounds = p3a_f32_bounds();
        let bf16_bounds = p3a_bf16_bounds();
        let numeric_mhc_rsqrt = required_score(
            "mHC flat rsqrt",
            &[cpu.prefix.hc_flat_rsqrt],
            &gpu_hc_rsqrt,
            &[hc_reference.flat_rsqrt],
            &f32_bounds,
        )?;
        let numeric_mhc_mixes = required_score(
            "mHC mixes",
            &cpu.prefix.hc_mixes_f32,
            &gpu_hc_mixes,
            &hc_reference.mixes,
            &f32_bounds,
        )?;
        let numeric_mhc_pre = required_score(
            "mHC pre",
            &cpu.prefix.hc_pre_f32,
            &gpu_hc_pre,
            &hc_reference.pre,
            &f32_bounds,
        )?;
        let numeric_mhc_post = required_score(
            "mHC post",
            &cpu.prefix.hc_post_f32,
            &gpu_hc_post,
            &hc_reference.post,
            &f32_bounds,
        )?;
        let numeric_mhc_comb = required_score(
            "mHC comb",
            &cpu.prefix.hc_comb_f32,
            &gpu_hc_comb,
            &hc_reference.comb,
            &f32_bounds,
        )?;
        let numeric_mhc_reduced = required_score(
            "mHC reduced BF16",
            &bf16_bits_f32(&cpu.prefix.hc_attn_pre_bf16_bits),
            &bf16_bits_f32(&gpu_hc_reduced),
            &hc_reference.reduced,
            &bf16_bounds,
        )?;
        let numeric_attn_norm = required_score(
            "attention RMSNorm BF16",
            &bf16_bits_f32(&cpu.prefix.attn_norm_bf16_bits),
            &bf16_bits_f32(&gpu_attn_norm),
            &attn_norm_reference,
            &bf16_bounds,
        )?;
        let numeric_wq_a = required_score(
            "WQ-A FP8 projection",
            &cpu.wq_a.output.fp32,
            &gpu_wq_a_fp32,
            &wq_a_reference,
            &f32_bounds,
        )?;
        let numeric_q_norm = required_score(
            "Q RMSNorm BF16",
            &bf16_bits_f32(&cpu.q_norm_bf16_bits),
            &bf16_bits_f32(&gpu_q_norm),
            &q_norm_reference,
            &bf16_bounds,
        )?;
        let numeric_wq_b = required_score(
            "WQ-B FP8 projection",
            &cpu.wq_b.output.fp32,
            &gpu_wq_b_fp32,
            &wq_b_reference,
            &f32_bounds,
        )?;
        let numeric_q_head = required_score(
            "per-head Q RMSNorm BF16",
            &bf16_bits_f32(&cpu.q_head_norm_bf16_bits),
            &bf16_bits_f32(&gpu_q_head),
            &q_head_reference,
            &bf16_bounds,
        )?;
        // The repository's receipt verifier canonicalizes JSON with Python.
        // Preserve all V2.1 floating values as decimal strings so the signed
        // cross-language receipt has no implementation-specific float-format
        // spelling while integer counters remain JSON numbers.
        let f32_bounds_json = receipt_decimal_strings(serde_json::to_value(f32_bounds)?);
        let bf16_bounds_json = receipt_decimal_strings(serde_json::to_value(bf16_bounds)?);
        let numeric_mhc_rsqrt_json =
            receipt_decimal_strings(serde_json::to_value(numeric_mhc_rsqrt)?);
        let numeric_mhc_mixes_json =
            receipt_decimal_strings(serde_json::to_value(numeric_mhc_mixes)?);
        let numeric_mhc_pre_json = receipt_decimal_strings(serde_json::to_value(numeric_mhc_pre)?);
        let numeric_mhc_post_json =
            receipt_decimal_strings(serde_json::to_value(numeric_mhc_post)?);
        let numeric_mhc_comb_json =
            receipt_decimal_strings(serde_json::to_value(numeric_mhc_comb)?);
        let numeric_mhc_reduced_json =
            receipt_decimal_strings(serde_json::to_value(numeric_mhc_reduced)?);
        let numeric_attn_norm_json =
            receipt_decimal_strings(serde_json::to_value(numeric_attn_norm)?);
        let numeric_wq_a_json = receipt_decimal_strings(serde_json::to_value(numeric_wq_a)?);
        let numeric_q_norm_json = receipt_decimal_strings(serde_json::to_value(numeric_q_norm)?);
        let numeric_wq_b_json = receipt_decimal_strings(serde_json::to_value(numeric_wq_b)?);
        let numeric_q_head_json = receipt_decimal_strings(serde_json::to_value(numeric_q_head)?);

        let stage_timings = json!([
            stage_profile(
                "mhc_attn_pre_sinkhorn",
                HC_KERNEL,
                &hc_timing,
                hc_fn.len() + hc_scale.len() + hc_base.len() + embed.len() * HC_MULT,
                row_bytes + 4 + 24 * 4 + 4 * 4 + 4 * 4 + 16 * 4,
                HC_MIX_WIDTH as u64 * HC_FLAT_WIDTH as u64 * 2
                    + HIDDEN_SIZE as u64 * HC_MULT as u64 * 2,
                0
            ),
            stage_profile(
                "attn_rmsnorm",
                RMS_KERNEL,
                &attn_norm_timing,
                row_bytes * 2,
                row_bytes,
                HIDDEN_SIZE as u64 * 4,
                0
            ),
            stage_profile(
                "wq_a_act_quant",
                QAT_KERNEL,
                &wq_a_qat_timing,
                row_bytes,
                LAYER0_WQ_A_COLS + LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK,
                LAYER0_WQ_A_COLS as u64 * 2,
                LAYER0_WQ_A_COLS as u64
            ),
            stage_profile(
                "wq_a_fp8_matvec",
                FP8_KERNEL,
                &wq_a_timing,
                wq_a_weight.len()
                    + wq_a_scale.len()
                    + LAYER0_WQ_A_COLS
                    + LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK,
                LAYER0_WQ_A_ROWS * 4,
                (LAYER0_WQ_A_ROWS * LAYER0_WQ_A_COLS * 2) as u64,
                LAYER0_WQ_A_ROWS as u64 * LAYER0_WQ_A_COLS as u64
            ),
            stage_profile(
                "wq_a_fp32_to_bf16",
                CAST_KERNEL,
                &wq_a_cast_timing,
                LAYER0_WQ_A_ROWS * 4,
                LAYER0_WQ_A_ROWS * 2,
                0,
                LAYER0_WQ_A_ROWS as u64
            ),
            stage_profile(
                "q_rmsnorm",
                RMS_KERNEL,
                &q_norm_timing,
                Q_LORA_RANK * 4,
                Q_LORA_RANK * 2,
                Q_LORA_RANK as u64 * 4,
                0
            ),
            stage_profile(
                "wq_b_act_quant",
                QAT_KERNEL,
                &wq_b_qat_timing,
                Q_LORA_RANK * 2,
                Q_LORA_RANK + Q_LORA_RANK / ACT_QUANT_BLOCK,
                Q_LORA_RANK as u64 * 2,
                Q_LORA_RANK as u64
            ),
            stage_profile(
                "wq_b_fp8_matvec",
                FP8_KERNEL,
                &wq_b_timing,
                wq_b_weight.len() + wq_b_scale.len() + Q_LORA_RANK + Q_LORA_RANK / ACT_QUANT_BLOCK,
                WQ_B_ROWS * 4,
                (WQ_B_ROWS * Q_LORA_RANK * 2) as u64,
                WQ_B_ROWS as u64 * Q_LORA_RANK as u64
            ),
            stage_profile(
                "wq_b_fp32_to_bf16",
                CAST_KERNEL,
                &wq_b_cast_timing,
                WQ_B_ROWS * 4,
                WQ_B_ROWS * 2,
                0,
                WQ_B_ROWS as u64
            ),
            stage_profile(
                "q_per_head_rmsnorm",
                PER_HEAD_KERNEL,
                &q_head_timing,
                WQ_B_ROWS * 2,
                WQ_B_ROWS * 2,
                WQ_B_ROWS as u64 * 3,
                0
            ),
        ]);

        let unsigned = json!({
            "schema": RECEIPT_SCHEMA,
            "status": RECEIPT_STATUS,
            "scope": {
                "bounded_rung": "P3A layer-0 pre-attention only",
                "tokenizer_bound_token_id": PREFIX_TOKEN_ID,
                "batch": 1,
                "sequence_tokens": 1,
                "position": 0,
                "compress_ratio": 0,
                "position_zero_rope_identity_checked": true,
                "not_kv_or_sparse_attention": true,
                "not_indexer_or_router": true,
                "not_moe": true,
                "not_full_model_load_or_forward": true,
                "not_token_generation": true,
                "not_hcli_endpoint": true,
                "not_base_true_tps_measurement": true,
                "not_registered_runtime": true,
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
                    "inference/model.py": anchors.prefix.act_quant.inference_model_py_sha256,
                    "inference/kernel.py": anchors.prefix.act_quant.inference_kernel_py_sha256,
                    "inference/config.json": anchors.prefix.act_quant.inference_config_json_sha256,
                    "config.json": anchors.prefix.act_quant.model_config_json_sha256,
                    "inference/convert.py": anchors.inference_convert_py_sha256,
                    "tokenizer.json": anchors.prefix.tokenizer_json_sha256,
                    "tokenizer_config.json": anchors.prefix.tokenizer_config_json_sha256,
                },
                "source_tensor_chunk_bindings": {
                    "embed_bos_row": tensor_binding_json(&embed_meta),
                    "hc_attn_fn": tensor_binding_json(&hc_fn_meta),
                    "hc_attn_base": tensor_binding_json(&hc_base_meta),
                    "hc_attn_scale": tensor_binding_json(&hc_scale_meta),
                    "attn_norm_weight": tensor_binding_json(&attn_norm_meta),
                    "wq_a_weight": tensor_binding_json(&wq_a_meta),
                    "wq_a_scale": tensor_binding_json(&wq_a_scale_meta),
                    "q_norm_weight": tensor_binding_json(&q_norm_meta),
                    "wq_b_weight": tensor_binding_json(&wq_b_meta),
                    "wq_b_scale": tensor_binding_json(&wq_b_scale_meta),
                    "all_touched_source_chunks_sha256_verified_before_gpu_upload": true,
                    "parent_safetensors_materialized": false,
                },
            },
            "sealed_cpu_attention_oracle": {
                "path": cpu_oracle.path.display().to_string(),
                "file_sha256": cpu_oracle.file_sha256,
                "receipt_seal_sha256": cpu_oracle.seal_sha256,
                "receipt_seal_verified": true,
                "direct_source_cpu_oracle_recomputed_and_matches_p3a_stage_hashes": true,
                "role": "discrete/transcribed-source comparison only; not the sole numerical authority",
            },
            "numeric_parity_v2_1": {
                "schema": "hawking.numeric_parity.v2_1",
                "reference_authority": "Separately accumulated FP64 references from raw streamed BF16/F32/FP8/E8M0 source payloads and discrete device QAT bytes. Neither candidate is produced by casting source-CPU f32 output to FP64.",
                "f32_operator_bounds": f32_bounds_json,
                "bf16_storage_bounds": bf16_bounds_json,
                "mhc_flat_rsqrt_f32": numeric_mhc_rsqrt_json,
                "mhc_mixes_f32": numeric_mhc_mixes_json,
                "mhc_pre_f32": numeric_mhc_pre_json,
                "mhc_post_f32": numeric_mhc_post_json,
                "mhc_comb_f32": numeric_mhc_comb_json,
                "mhc_reduced_bf16": numeric_mhc_reduced_json,
                "attn_rmsnorm_bf16": numeric_attn_norm_json,
                "wq_a_fp8_projection_f32": numeric_wq_a_json,
                "q_rmsnorm_bf16": numeric_q_norm_json,
                "wq_b_fp8_projection_f32": numeric_wq_b_json,
                "q_per_head_rmsnorm_bf16": numeric_q_head_json,
                "all_host_and_device_scores_pass": true,
            },
            "discrete_parity": {
                "mhc_reduced_bf16_byte_exact": true,
                "attn_rmsnorm_bf16_byte_exact": true,
                "wq_a_qat_activation_and_scale_byte_exact": true,
                "wq_a_bf16_handoff_byte_exact": true,
                "q_rmsnorm_bf16_byte_exact": true,
                "wq_b_qat_activation_and_scale_byte_exact": true,
                "wq_b_bf16_handoff_byte_exact": true,
                "q_per_head_rmsnorm_bf16_byte_exact": true,
                "position_zero_rope_bf16_identity_byte_exact": true,
            },
            "metal": {
                "device": device_name,
                "pipelines_precompiled_before_measured_dispatches": true,
                "pipeline_geometry_limits": pipeline_widths,
                "gpu_dispatches": P3A_DISPATCHES,
                "command_buffers": P3A_DISPATCHES,
                "compute_encoders": P3A_DISPATCHES,
                "cpu_visible_waits": P3A_DISPATCHES,
                "empty_command_buffers": 0,
                "physical_trace_command_buffers": physical_counts.command_count,
                "physical_trace_compute_encoders": physical_counts.encoder_count,
                "trace_samples": trace_samples.len(),
                "buffers_created": buffers_created,
                "bytes_allocated": bytes_allocated,
                "fallback": false,
                "fallback_count": 0,
                "host_activation_or_routing": false,
                "host_intermediate_handoff_bytes": 0,
                "host_used_only_for_source_oracle_and_post_chain_comparison": true,
            },
            "command_topology": {
                "topology": "ten ordered, timestamped command buffers for stage-local P3A diagnostics; shared device buffers carry every intermediate between stages",
                "reason_not_replayed_graph": "P3A is a numerical-authority rung, not a promoted decode graph; per-stage GPU timestamp accountability is retained deliberately",
                "command_buffers_per_bounded_p3a_rung": P3A_DISPATCHES,
                "cpu_waits_are_for_completed_timestamp_capture_not_host_data_handoff": true,
            },
            "complete_p3a_stage_profile": stage_timings,
            "runtime_boundary": "This is real native Metal execution of the bounded layer-0 pre-attention source path. It does not establish a V4 runtime, attention completion, token generation, HCLI behavior, or BASE_TRUE_TPS.",
            "physical_trace": {
                "interval_id": interval_id,
                "run_nonce": run_nonce,
                "phase": "dsv4f_p3a_layer0_preattention",
                "role": "mhc_norm_q_projection",
            },
        });
        let (receipt, seal_sha256) = seal(unsigned)?;
        write_new_receipt(&args.out, &receipt)?;
        println!(
            "{}",
            serde_json::to_string(&json!({
                "status": RECEIPT_STATUS,
                "receipt": args.out,
                "seal_sha256": seal_sha256,
                "gpu_dispatches": P3A_DISPATCHES,
                "fallback": false,
            }))?
        );
        Ok(())
    }

    fn parse_args() -> ProbeResult<Args> {
        let mut artifact = None;
        let mut cpu_attention_oracle = None;
        let mut out = None;
        let mut values = std::env::args_os().skip(1);
        while let Some(flag) = values.next() {
            match flag.to_string_lossy().as_ref() {
                "--artifact" => artifact = values.next().map(PathBuf::from),
                "--cpu-attention-oracle" => cpu_attention_oracle = values.next().map(PathBuf::from),
                "--out" => out = values.next().map(PathBuf::from),
                "--help" | "-h" => {
                    return Err(failure(
                        "usage: --artifact PATH --cpu-attention-oracle PATH --out PATH",
                    ))
                }
                other => return Err(failure(format!("unknown argument {other}"))),
            }
        }
        Ok(Args {
            artifact: artifact.ok_or_else(|| failure("--artifact is required"))?,
            cpu_attention_oracle: cpu_attention_oracle
                .ok_or_else(|| failure("--cpu-attention-oracle is required"))?,
            out: out.ok_or_else(|| failure("--out is required"))?,
        })
    }

    fn expect_tensor(
        reader: &DeepSeekV4FullStreamReader,
        name: &str,
        dtype: &str,
        shape: &[u64],
    ) -> ProbeResult<DeepSeekV4TensorMetadata> {
        let tensor = reader.tensor_metadata(name)?.clone();
        if tensor.dtype != dtype || tensor.shape.as_slice() != shape {
            return Err(failure(format!(
                "{name} source dtype/shape {:?}/{:?} differs from expected {dtype:?}/{shape:?}",
                tensor.dtype, tensor.shape
            )));
        }
        Ok(tensor)
    }

    fn fp8_pair_metadata(
        reader: &DeepSeekV4FullStreamReader,
        weight_name: &str,
        scale_name: &str,
        rows: usize,
        cols: usize,
    ) -> ProbeResult<(DeepSeekV4TensorMetadata, DeepSeekV4TensorMetadata)> {
        let pair = reader.native_scale_pair(weight_name)?;
        if pair.kind != NativeScalePairKind::Fp8E4M3fn
            || pair.weight.name != weight_name
            || pair.scale.name != scale_name
            || pair.weight.shape.as_slice() != [rows as u64, cols as u64]
            || pair.scale.shape.as_slice()
                != [
                    (rows / ACT_QUANT_BLOCK) as u64,
                    (cols / ACT_QUANT_BLOCK) as u64,
                ]
            || pair.logical_k != cols as u64
            || pair.out_rows != rows as u64
        {
            return Err(failure(format!(
                "{weight_name} native FP8/E8M0 source geometry changed from P3A contract"
            )));
        }
        Ok((pair.weight.clone(), pair.scale.clone()))
    }

    fn validate_cpu_attention_oracle(
        path: &Path,
        reader: &DeepSeekV4FullStreamReader,
        cpu: &hawking_core::gravity_deepseek_v4_layer0_attention::Layer0AttentionCpuOracleResult,
    ) -> ProbeResult<CpuOracleBinding> {
        if path.file_name().and_then(|name| name.to_str()) != Some(CPU_ORACLE_BASENAME) {
            return Err(failure(format!(
                "only the canonical {CPU_ORACLE_BASENAME} may bind P3A"
            )));
        }
        let path = fs::canonicalize(path)?;
        let raw = fs::read(&path)?;
        let value: Value = serde_json::from_slice(&raw).map_err(|error| {
            failure(format!("sealed CPU attention oracle is not JSON: {error}"))
        })?;
        let seal_sha256 = text_at(&value, &["seal_sha256"])?.to_owned();
        if !is_sha256(&seal_sha256)
            || sha256(&canonical_json(&without_seal(&value)?)) != seal_sha256
        {
            return Err(failure(
                "sealed CPU attention oracle canonical seal does not verify",
            ));
        }
        if text_at(&value, &["schema"])? != CPU_ORACLE_SCHEMA
            || text_at(&value, &["status"])? != CPU_ORACLE_STATUS
            || text_at(&value, &["artifact", "manifest_seal_sha256"])?
                != reader.manifest_seal_sha256()
            || text_at(&value, &["artifact", "source", "repository"])?
                != reader.source_identity().repository
            || text_at(&value, &["artifact", "source", "revision"])?
                != reader.source_identity().revision
        {
            return Err(failure(
                "sealed CPU attention oracle does not bind the admitted full-stream artifact/source",
            ));
        }
        let checks = [
            (
                &[
                    "intermediate_receipts",
                    "prefix_continuity",
                    "embedding",
                    "sha256_bf16_le",
                ][..],
                sha256(&u16_le_bytes(&cpu.prefix.embed_bf16_bits)),
            ),
            (
                &[
                    "intermediate_receipts",
                    "prefix_continuity",
                    "hc_attn_pre",
                    "mixes",
                    "sha256_f32_le",
                ][..],
                sha256(&f32_le_bytes(&cpu.prefix.hc_mixes_f32)),
            ),
            (
                &[
                    "intermediate_receipts",
                    "prefix_continuity",
                    "hc_attn_pre",
                    "reduced_bf16",
                    "sha256_bf16_le",
                ][..],
                sha256(&u16_le_bytes(&cpu.prefix.hc_attn_pre_bf16_bits)),
            ),
            (
                &[
                    "intermediate_receipts",
                    "prefix_continuity",
                    "attn_norm_output_and_wq_a_input",
                    "sha256_bf16_le",
                ][..],
                sha256(&u16_le_bytes(&cpu.prefix.attn_norm_bf16_bits)),
            ),
            (
                &[
                    "intermediate_receipts",
                    "q_path",
                    "wq_a_fp8_linear",
                    "output_fp32",
                    "sha256_f32_le",
                ][..],
                sha256(&f32_le_bytes(&cpu.wq_a.output.fp32)),
            ),
            (
                &[
                    "intermediate_receipts",
                    "q_path",
                    "wq_a_fp8_linear",
                    "output_bf16",
                    "sha256_bf16_le",
                ][..],
                sha256(&u16_le_bytes(&cpu.wq_a.output.bf16_bits)),
            ),
            (
                &[
                    "intermediate_receipts",
                    "q_path",
                    "q_norm",
                    "sha256_bf16_le",
                ][..],
                sha256(&u16_le_bytes(&cpu.q_norm_bf16_bits)),
            ),
            (
                &[
                    "intermediate_receipts",
                    "q_path",
                    "wq_b_fp8_linear",
                    "output_fp32",
                    "sha256_f32_le",
                ][..],
                sha256(&f32_le_bytes(&cpu.wq_b.output.fp32)),
            ),
            (
                &[
                    "intermediate_receipts",
                    "q_path",
                    "wq_b_fp8_linear",
                    "output_bf16",
                    "sha256_bf16_le",
                ][..],
                sha256(&u16_le_bytes(&cpu.wq_b.output.bf16_bits)),
            ),
            (
                &[
                    "intermediate_receipts",
                    "q_path",
                    "per_head_rmsnorm",
                    "sha256_bf16_le",
                ][..],
                sha256(&u16_le_bytes(&cpu.q_head_norm_bf16_bits)),
            ),
        ];
        for (path, expected) in checks {
            if text_at(&value, path)? != expected {
                return Err(failure(format!(
                    "direct source CPU oracle diverges from sealed attention stage {}",
                    path.join(".")
                )));
            }
        }
        Ok(CpuOracleBinding {
            path,
            file_sha256: sha256(&raw),
            seal_sha256,
        })
    }

    fn hc_attn_pre_f64_reference(
        embed: &[u8],
        hc_fn: &[u8],
        hc_scale: &[u8],
        hc_base: &[u8],
    ) -> ProbeResult<HcF64Reference> {
        let embed = bf16_le_f64(embed)?;
        let hc_fn = f32_le_f64(hc_fn)?;
        let hc_scale = f32_le_f64(hc_scale)?;
        let hc_base = f32_le_f64(hc_base)?;
        if embed.len() != HIDDEN_SIZE
            || hc_fn.len() != HC_MIX_WIDTH * HC_FLAT_WIDTH
            || hc_scale.len() != 3
            || hc_base.len() != HC_MIX_WIDTH
        {
            return Err(failure("FP64 mHC source reference geometry is invalid"));
        }
        let mut sum_square = 0.0_f64;
        for _lane in 0..HC_MULT {
            for &value in &embed {
                sum_square += value * value;
            }
        }
        let flat_rsqrt = 1.0 / (sum_square / HC_FLAT_WIDTH as f64 + RMS_NORM_EPS as f64).sqrt();
        let mut mixes = vec![0.0_f64; HC_MIX_WIDTH];
        for row in 0..HC_MIX_WIDTH {
            let mut accumulator = 0.0_f64;
            for lane in 0..HC_MULT {
                let base = row * HC_FLAT_WIDTH + lane * HIDDEN_SIZE;
                for feature in 0..HIDDEN_SIZE {
                    accumulator += hc_fn[base + feature] * embed[feature];
                }
            }
            mixes[row] = accumulator * flat_rsqrt;
        }
        let mut pre = vec![0.0_f64; HC_MULT];
        let mut post = vec![0.0_f64; HC_MULT];
        let mut comb = vec![0.0_f64; HC_MULT * HC_MULT];
        for lane in 0..HC_MULT {
            pre[lane] =
                1.0 / (1.0 + (-(mixes[lane] * hc_scale[0] + hc_base[lane])).exp()) + HC_EPS as f64;
            post[lane] = 2.0
                * (1.0
                    / (1.0
                        + (-(mixes[lane + HC_MULT] * hc_scale[1] + hc_base[lane + HC_MULT]))
                            .exp()));
        }
        for row in 0..HC_MULT {
            for column in 0..HC_MULT {
                let index = row * HC_MULT + column;
                let source_index = index + HC_MULT * 2;
                comb[index] = mixes[source_index] * hc_scale[2] + hc_base[source_index];
            }
        }
        for row in 0..HC_MULT {
            let start = row * HC_MULT;
            let row_max = comb[start..start + HC_MULT]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut row_sum = 0.0_f64;
            for column in 0..HC_MULT {
                let index = start + column;
                comb[index] = (comb[index] - row_max).exp();
                row_sum += comb[index];
            }
            for column in 0..HC_MULT {
                let index = start + column;
                comb[index] = comb[index] / row_sum + HC_EPS as f64;
            }
        }
        normalize_columns_f64(&mut comb)?;
        for _ in 1..HC_SINKHORN_ITERS {
            normalize_rows_f64(&mut comb)?;
            normalize_columns_f64(&mut comb)?;
        }
        let mut reduced = vec![0.0_f64; HIDDEN_SIZE];
        for feature in 0..HIDDEN_SIZE {
            for lane in 0..HC_MULT {
                reduced[feature] += pre[lane] * embed[feature];
            }
        }
        Ok(HcF64Reference {
            flat_rsqrt,
            mixes,
            pre,
            post,
            comb,
            reduced,
        })
    }

    fn normalize_rows_f64(comb: &mut [f64]) -> ProbeResult<()> {
        for row in 0..HC_MULT {
            let start = row * HC_MULT;
            let sum: f64 = comb[start..start + HC_MULT].iter().sum();
            if !(sum.is_finite() && sum > 0.0) {
                return Err(failure("FP64 mHC Sinkhorn row sum is invalid"));
            }
            for value in &mut comb[start..start + HC_MULT] {
                *value /= sum + HC_EPS as f64;
            }
        }
        Ok(())
    }

    fn normalize_columns_f64(comb: &mut [f64]) -> ProbeResult<()> {
        for column in 0..HC_MULT {
            let mut sum = 0.0_f64;
            for row in 0..HC_MULT {
                sum += comb[row * HC_MULT + column];
            }
            if !(sum.is_finite() && sum > 0.0) {
                return Err(failure("FP64 mHC Sinkhorn column sum is invalid"));
            }
            for row in 0..HC_MULT {
                let index = row * HC_MULT + column;
                comb[index] /= sum + HC_EPS as f64;
            }
        }
        Ok(())
    }

    fn fp8_linear_f64_reference(
        weight: &[u8],
        weight_scales: &[u8],
        activation: &[u8],
        activation_scales: &[u8],
        rows: usize,
        cols: usize,
    ) -> ProbeResult<Vec<f64>> {
        if rows == 0
            || cols == 0
            || cols % ACT_QUANT_BLOCK != 0
            || rows % ACT_QUANT_BLOCK != 0
            || weight.len() != rows * cols
            || weight_scales.len() != (rows / ACT_QUANT_BLOCK) * (cols / ACT_QUANT_BLOCK)
            || activation.len() != cols
            || activation_scales.len() != cols / ACT_QUANT_BLOCK
        {
            return Err(failure(
                "FP64 source FP8 linear reference geometry is invalid",
            ));
        }
        let scale_cols = cols / ACT_QUANT_BLOCK;
        let mut output = vec![0.0_f64; rows];
        for row in 0..rows {
            let mut row_accumulator = 0.0_f64;
            for block in 0..scale_cols {
                let start = block * ACT_QUANT_BLOCK;
                let mut block_accumulator = 0.0_f64;
                for column in start..start + ACT_QUANT_BLOCK {
                    block_accumulator +=
                        e4m3fn_f64(activation[column])? * e4m3fn_f64(weight[row * cols + column])?;
                }
                row_accumulator += block_accumulator
                    * (e8m0fnu_f64(activation_scales[block])?
                        * e8m0fnu_f64(
                            weight_scales[(row / ACT_QUANT_BLOCK) * scale_cols + block],
                        )?);
            }
            output[row] = row_accumulator;
        }
        Ok(output)
    }

    fn per_head_rmsnorm_f64(
        input: &[f64],
        heads: usize,
        head_dim: usize,
        eps: f64,
    ) -> ProbeResult<Vec<f64>> {
        if heads == 0 || head_dim == 0 || input.len() != heads * head_dim || !(eps > 0.0) {
            return Err(failure("FP64 per-head RMSNorm geometry is invalid"));
        }
        let mut output = Vec::with_capacity(input.len());
        for head in 0..heads {
            let row = &input[head * head_dim..(head + 1) * head_dim];
            let sum_square: f64 = row.iter().map(|value| value * value).sum();
            let reciprocal = 1.0 / (sum_square / head_dim as f64 + eps).sqrt();
            output.extend(row.iter().map(|value| value * reciprocal));
        }
        Ok(output)
    }

    fn e4m3fn_f64(bits: u8) -> ProbeResult<f64> {
        let exponent = (bits >> 3) & 0x0f;
        let mantissa = bits & 0x07;
        if exponent == 0x0f && mantissa == 0x07 {
            return Err(failure("FP64 source FP8 reference encountered E4M3FN NaN"));
        }
        let magnitude = if exponent == 0 {
            (mantissa as f64) * 2f64.powi(-9)
        } else {
            (1.0 + mantissa as f64 / 8.0) * 2f64.powi(exponent as i32 - 7)
        };
        Ok(if (bits & 0x80) != 0 {
            -magnitude
        } else {
            magnitude
        })
    }

    fn e8m0fnu_f64(bits: u8) -> ProbeResult<f64> {
        if bits == 0xff {
            return Err(failure("FP64 source FP8 reference encountered E8M0FNU NaN"));
        }
        Ok(2f64.powi(bits as i32 - 127))
    }

    fn p3a_f32_bounds() -> Bounds {
        // The source operations are local but include 16K mHC reductions and
        // FP8 block reductions.  These bounds still gate relative L2,
        // cosine, meaningful-scale max error, and exact top-k/argmax while
        // permitting normal source f32 accumulation-order variance against
        // independently accumulated FP64 values.
        Bounds {
            max_abs_near_zero: 1.0e-3,
            max_relative_l2: 2.0e-4,
            min_cosine: 0.999_999,
            max_kl: 0.0,
            require_kl: false,
            top_k: 5,
            max_meaningful_rel: 5.0e-2,
            gate_max_meaningful_rel: true,
        }
    }

    fn p3a_bf16_bounds() -> Bounds {
        // These candidates are deliberately BF16 storage checkpoints.  The
        // strict discrete gate above proves CPU/device identical BF16 bytes;
        // V2.1 here separately quantifies their distance from an FP64
        // pre-rounding authority without treating BF16 rounding as f32 drift.
        Bounds {
            max_abs_near_zero: 2.0e-2,
            max_relative_l2: 1.0e-2,
            min_cosine: 0.999,
            max_kl: 0.0,
            require_kl: false,
            top_k: 5,
            max_meaningful_rel: 1.0e-1,
            gate_max_meaningful_rel: true,
        }
    }

    fn required_score(
        name: &str,
        host: &[f32],
        device: &[f32],
        reference: &[f64],
        bounds: &Bounds,
    ) -> ProbeResult<PairedScore> {
        if host.len() != device.len() || host.len() != reference.len() {
            return Err(failure(format!(
                "Numeric Parity V2.1 {name} geometry mismatch"
            )));
        }
        let score = score_pair(host, device, reference, bounds);
        if !score.pass {
            return Err(failure(format!(
                "Numeric Parity V2.1 {name} failed against separately accumulated FP64 reference: host={:?}; device={:?}",
                score.host.failures, score.device.failures
            )));
        }
        Ok(score)
    }

    fn stage_profile(
        stage: &str,
        kernel: &str,
        timing: &MetalDispatchTiming,
        bytes_read: usize,
        bytes_written: usize,
        fp_operations: u64,
        integer_or_bit_operations: u64,
    ) -> Value {
        json!({
            "stage": stage,
            "kernel": kernel,
            "gpu_duration_us": timing.gpu_duration_us,
            "gpu_start_ns": timing.gpu_start_ns,
            "gpu_end_ns": timing.gpu_end_ns,
            "cpu_duration_us": timing.host_wall_us,
            "cpu_encode_us": timing.encode_us,
            "cpu_submit_us": timing.submit_us,
            "cpu_wait_us": timing.wait_us,
            "bytes_read": bytes_read,
            "bytes_written": bytes_written,
            "fp_operations_estimate": fp_operations,
            "integer_or_bit_operations_estimate": integer_or_bit_operations,
            "dispatches": timing.compute_dispatches,
            "command_buffers": timing.command_buffers,
            "waits": timing.command_buffers,
            "occupancy": "not inferred from a single bounded authority dispatch",
            "effective_bandwidth": "not inferred; source-authority geometry is not a throughput candidate",
            "p50_p95_p99": "single trial only; no percentile claim",
            "fallback": false,
            "unexplained_other": 0,
        })
    }

    fn require_completed_dispatch(timing: &MetalDispatchTiming, stage: &str) -> ProbeResult<()> {
        if timing.command_buffers != 1
            || timing.compute_encoders != 1
            || timing.compute_dispatches != 1
            || timing.gpu_duration_us.is_none()
            || timing.gpu_start_ns.is_none()
            || timing.gpu_end_ns.is_none()
        {
            return Err(failure(format!(
                "{stage} did not complete with one real GPU-timestamped command buffer/encoder/dispatch"
            )));
        }
        Ok(())
    }

    fn exact_bytes(name: &str, expected: &[u8], observed: &[u8]) -> ProbeResult<()> {
        if expected != observed {
            return Err(failure(format!(
                "{name} differs from the source CPU oracle; no P3A receipt emitted"
            )));
        }
        Ok(())
    }

    fn exact_u16(name: &str, expected: &[u16], observed: &[u16]) -> ProbeResult<()> {
        exact_bytes(name, &u16_le_bytes(expected), &u16_le_bytes(observed))
    }

    fn exact_f32(
        name: &str,
        expected: &[f32],
        observed: &[f32],
        absolute: f32,
        relative: f32,
    ) -> ProbeResult<()> {
        if expected.len() != observed.len() || expected.is_empty() {
            return Err(failure(format!("{name} CPU/GPU f32 geometry is invalid")));
        }
        for (index, (&expected, &observed)) in expected.iter().zip(observed).enumerate() {
            if !expected.is_finite()
                || !observed.is_finite()
                || (expected - observed).abs() > absolute + relative * expected.abs()
            {
                return Err(failure(format!(
                    "{name} CPU/GPU f32 parity failed at {index}: expected {expected}, observed {observed}"
                )));
            }
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

    fn set_f32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &f32) {
        encoder.set_bytes(
            index,
            std::mem::size_of::<f32>() as u64,
            value as *const f32 as *const _,
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

    fn read_gpu_u16(buffer: &metal::Buffer, count: usize) -> ProbeResult<Vec<u16>> {
        let bytes = count
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| failure("GPU BF16 readback byte count overflow"))?;
        let raw = read_gpu_bytes(buffer, bytes)?;
        Ok(raw
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect())
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
            return Err(failure("GPU P3A stage produced a non-finite FP32 output"));
        }
        Ok(output)
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

    fn bf16_le_f64(bytes: &[u8]) -> ProbeResult<Vec<f64>> {
        if bytes.len() % 2 != 0 {
            return Err(failure("BF16 FP64 reference payload has odd byte length"));
        }
        let output = bytes
            .chunks_exact(2)
            .map(|chunk| {
                f32::from_bits((u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16) as f64
            })
            .collect::<Vec<_>>();
        if output.iter().any(|value| !value.is_finite()) {
            return Err(failure("BF16 FP64 reference contains a non-finite value"));
        }
        Ok(output)
    }

    fn f32_le_f64(bytes: &[u8]) -> ProbeResult<Vec<f64>> {
        if bytes.len() % 4 != 0 {
            return Err(failure("F32 FP64 reference payload is not aligned"));
        }
        let output = bytes
            .chunks_exact(4)
            .map(|chunk| {
                f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])) as f64
            })
            .collect::<Vec<_>>();
        if output.iter().any(|value| !value.is_finite()) {
            return Err(failure("F32 FP64 reference contains a non-finite value"));
        }
        Ok(output)
    }

    fn bf16_bits_f32(bits: &[u16]) -> Vec<f32> {
        bits.iter()
            .map(|bits| f32::from_bits((*bits as u32) << 16))
            .collect()
    }

    fn u16_le_bytes(values: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        bytes
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

    fn without_seal(value: &Value) -> ProbeResult<Value> {
        let mut copy = value.clone();
        let object = copy
            .as_object_mut()
            .ok_or_else(|| failure("receipt root is not a JSON object"))?;
        if object.remove("seal_sha256").is_none() {
            return Err(failure("receipt has no seal_sha256"));
        }
        Ok(copy)
    }

    /// Canonical receipt values must round-trip under both serde_json and the
    /// repository's Python JSON canonicalizer.  Integer counters stay native
    /// JSON numbers; all non-integer numerical diagnostics are strings.
    fn receipt_decimal_strings(value: Value) -> Value {
        match value {
            Value::Number(number) if number.is_i64() || number.is_u64() => Value::Number(number),
            Value::Number(number) => Value::String(number.to_string()),
            Value::Array(values) => {
                Value::Array(values.into_iter().map(receipt_decimal_strings).collect())
            }
            Value::Object(object) => Value::Object(
                object
                    .into_iter()
                    .map(|(key, value)| (key, receipt_decimal_strings(value)))
                    .collect(),
            ),
            other => other,
        }
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
            return Err(failure("P3A receipt must be an unsealed JSON object"));
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
                "refusing to overwrite existing P3A receipt {}",
                path.display()
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
        let temporary = parent.join(format!(".{filename}.{}.p3a.tmp", std::process::id()));
        let bytes = serde_json::to_vec_pretty(receipt)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| failure(format!("cannot create P3A receipt temporary: {error}")))?;
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
                "refusing to overwrite or link P3A receipt {}: {error}",
                path.display()
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

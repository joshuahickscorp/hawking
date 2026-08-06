//! Bounded native-Metal P4A proof: the complete DeepSeek-V4-Flash layer-0
//! attention block at the real tokenizer BOS, position zero, ratio zero.
//!
//! It is intentionally neither a 43-layer runtime nor a decode/HCLI/TPS
//! claim.  The only executed body is:
//!
//! ```text
//! embed -> mHC pre -> norm -> Q and KV paths -> one-KV sparse attention/sink
//! -> converted-WO-A grouped BF16 einsum -> WO-B -> mHC attention post
//! ```
//!
//! All intermediate tensors move directly between device buffers.  The CPU
//! source oracle is used only before and after the GPU run; each numerical
//! score compares source-CPU and device results to a separately accumulated
//! FP64 reference from raw streamed payloads.  Where the source specifies a
//! BF16 output store, that store boundary is explicitly materialized in the
//! FP64 reference before discrete parity is measured.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "gravity_deepseek_v4_p4a_layer0_attention_metal requires macOS Metal",
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
        layer0_attention_cpu_oracle, verify_layer0_attention_source_anchors,
        Layer0AttentionCpuOracleResult, HEAD_DIM, KV_QAT_BLOCK, LAYER0_ATTN_SINK,
        LAYER0_KV_NORM_WEIGHT, LAYER0_Q_NORM_WEIGHT, LAYER0_WKV_SCALE, LAYER0_WKV_WEIGHT,
        LAYER0_WO_A_SCALE, LAYER0_WO_A_WEIGHT, LAYER0_WO_B_SCALE, LAYER0_WO_B_WEIGHT,
        LAYER0_WQ_B_SCALE, LAYER0_WQ_B_WEIGHT, NON_ROPE_HEAD_DIM, NUM_HEADS, O_LORA_RANK,
        Q_LORA_RANK, ROPE_HEAD_DIM, WKV_ROWS, WO_A_COLS, WO_A_ROWS, WO_B_COLS, WO_B_ROWS,
        WQ_B_ROWS,
    };
    use hawking_core::gravity_deepseek_v4_layer0_prefix::{
        EMBED_WEIGHT, HC_EPS, HC_FLAT_WIDTH, HC_MIX_WIDTH, HC_MULT, HC_SINKHORN_ITERS, HIDDEN_SIZE,
        LAYER0_ATTN_NORM_WEIGHT, LAYER0_HC_ATTN_BASE, LAYER0_HC_ATTN_FN, LAYER0_HC_ATTN_SCALE,
        PREFIX_TOKEN_ID, RMS_NORM_EPS,
    };
    use hawking_core::metal::{
        MetalBatchTiming, MetalContext, MetalDispatchTiming, PhysicalTraceGuard,
        PhysicalTraceIdentity,
    };
    use hawking_core::numeric_parity::{rmsnorm_f64, score_pair, Bounds, PairedScore};
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::error::Error;
    use std::fs::{self, File, OpenOptions};
    use std::io::Write;
    use std::path::{Path, PathBuf};

    const RECEIPT_SCHEMA: &str = "hawking.gravity.deepseek_v4.p4a_layer0_attention_metal.v1";
    const RECEIPT_STATUS: &str = "PASS_REAL_METAL_P4A_LAYER0_COMPLETE_ATTENTION_PARITY_NOT_RUNTIME";
    const CPU_ORACLE_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.layer0_attention_cpu_algorithm_oracle.v1";
    const CPU_ORACLE_STATUS: &str =
        "PASS_SOURCE_DERIVED_CPU_LAYER0_ATTENTION_NOT_INDEPENDENT_UPSTREAM_RUNTIME_PARITY";
    const P3A_SCHEMA: &str = "hawking.gravity.deepseek_v4.p3a_layer0_preattention_metal.v1";
    const P3A_STATUS: &str = "PASS_REAL_METAL_P3A_LAYER0_PREATTENTION_PARITY_NOT_RUNTIME";
    const CPU_ORACLE_BASENAME: &str = "DSV4F_LAYER0_ATTENTION_CPU_ORACLE-v1.json";
    const P3A_BASENAME: &str = "DSV4F_P3A_LAYER0_PREATTENTION_METAL-v1.json";
    const P4A_BASENAME: &str = "DSV4F_P4A_LAYER0_COMPLETE_ATTENTION_METAL-v1.json";
    const P4A_TOPOLOGY_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.p4a_layer0_attention_topology_sweep.v1";
    const P4A_TOPOLOGY_TRIALS_MIN: usize = 5;
    const P4A_TOPOLOGY_TRIALS_DEFAULT: usize = 7;
    const P4A_TOPOLOGY_WARMUPS_DEFAULT: usize = 2;

    const HC_KERNEL: &str = "deepseek_v4_p3a_layer0_hc_attn_pre_bos_authority";
    const RMS_KERNEL: &str = "deepseek_v4_p3a_rmsnorm_bf16_authority";
    const QAT_KERNEL: &str = "deepseek_v4_act_quant_bf16_ue8m0_authority";
    const FP8_KERNEL: &str = "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority";
    const CAST_KERNEL: &str = "deepseek_v4_p3a_fp32_to_bf16_authority";
    const PER_HEAD_KERNEL: &str = "deepseek_v4_p3a_per_head_rmsnorm_bf16_authority";
    const KV_QAT_KERNEL: &str = "deepseek_v4_p4a_kv_nonrope_qat_inplace_authority";
    const SPARSE_KERNEL: &str = "deepseek_v4_p4a_sparse_attention_position0_sink_authority";
    const WO_A_KERNEL: &str = "deepseek_v4_p4a_wo_a_convert_bf16_einsum_authority";
    const HC_POST_KERNEL: &str = "deepseek_v4_p4a_hc_attn_post_authority";
    const P4A_DISPATCHES: u64 = 21;

    type ProbeResult<T> = Result<T, Box<dyn Error>>;

    struct Args {
        artifact: PathBuf,
        cpu_attention_oracle: PathBuf,
        p3a_precursor: PathBuf,
        out: PathBuf,
        topology_out: Option<PathBuf>,
        topology_trials: usize,
        topology_warmups: usize,
    }

    struct ReceiptBinding {
        path: PathBuf,
        file_sha256: String,
        seal_sha256: String,
    }

    struct HcF64 {
        post: Vec<f64>,
        comb: Vec<f64>,
    }

    struct SparseF64 {
        scores: Vec<f64>,
        denominators: Vec<f64>,
        output: Vec<f64>,
    }

    struct TopologyPending {
        out: PathBuf,
        authority: ReceiptBinding,
        warmups: usize,
        trials: usize,
        baseline_gpu_us: Vec<u64>,
        baseline_host_wall_us: Vec<u64>,
        candidate_gpu_us: Vec<u64>,
        candidate_host_wall_us: Vec<u64>,
        candidate_encode_us: Vec<u64>,
        candidate_submit_us: Vec<u64>,
        candidate_wait_us: Vec<u64>,
        physical_command_buffers: u64,
        physical_compute_encoders: u64,
        trace_samples: usize,
        commits: usize,
        buffers_created: usize,
        bytes_allocated: usize,
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
            return Err(failure("P4A requires the pinned DeepSeek-V4-Flash source"));
        }
        let topology_authority = if let Some(topology_out) = args.topology_out.as_ref() {
            if topology_out == &args.out {
                return Err(failure("--topology-out must differ from --out"));
            }
            if topology_out.exists() {
                return Err(failure(format!(
                    "refusing overwrite {}",
                    topology_out.display()
                )));
            }
            Some(validate_p4a_authority_receipt(&args.out, &reader)?)
        } else {
            None
        };

        let cpu = layer0_attention_cpu_oracle(&reader)?;
        let cpu_binding = validate_cpu_receipt(&args.cpu_attention_oracle, &reader, &cpu)?;
        let p3a_binding = validate_p3a_receipt(&args.p3a_precursor, &reader)?;

        let embed_meta = reader.tensor_metadata(EMBED_WEIGHT)?.clone();
        let embed = reader.read_verified_range(
            EMBED_WEIGHT,
            0..(HIDDEN_SIZE * std::mem::size_of::<u16>()) as u64,
            HIDDEN_SIZE * std::mem::size_of::<u16>(),
        )?;
        let (hc_fn_meta, hc_fn) = full(&reader, LAYER0_HC_ATTN_FN)?;
        let (hc_base_meta, hc_base) = full(&reader, LAYER0_HC_ATTN_BASE)?;
        let (hc_scale_meta, hc_scale) = full(&reader, LAYER0_HC_ATTN_SCALE)?;
        let (attn_norm_meta, attn_norm) = full(&reader, LAYER0_ATTN_NORM_WEIGHT)?;
        let (q_norm_meta, q_norm) = full(&reader, LAYER0_Q_NORM_WEIGHT)?;
        let (kv_norm_meta, kv_norm) = full(&reader, LAYER0_KV_NORM_WEIGHT)?;
        let (sink_meta, sink) = full(&reader, LAYER0_ATTN_SINK)?;
        let (wq_a_meta, wq_a_scale_meta, wq_a_weight, wq_a_scale) = fp8_pair(
            &reader,
            LAYER0_WQ_A_WEIGHT,
            LAYER0_WQ_A_SCALE,
            LAYER0_WQ_A_ROWS,
            LAYER0_WQ_A_COLS,
        )?;
        let (wq_b_meta, wq_b_scale_meta, wq_b_weight, wq_b_scale) = fp8_pair(
            &reader,
            LAYER0_WQ_B_WEIGHT,
            LAYER0_WQ_B_SCALE,
            WQ_B_ROWS,
            Q_LORA_RANK,
        )?;
        let (wkv_meta, wkv_scale_meta, wkv_weight, wkv_scale) = fp8_pair(
            &reader,
            LAYER0_WKV_WEIGHT,
            LAYER0_WKV_SCALE,
            WKV_ROWS,
            HIDDEN_SIZE,
        )?;
        let (wo_a_meta, wo_a_scale_meta, wo_a_weight, wo_a_scale) = fp8_pair(
            &reader,
            LAYER0_WO_A_WEIGHT,
            LAYER0_WO_A_SCALE,
            WO_A_ROWS,
            WO_A_COLS,
        )?;
        let (wo_b_meta, wo_b_scale_meta, wo_b_weight, wo_b_scale) = fp8_pair(
            &reader,
            LAYER0_WO_B_WEIGHT,
            LAYER0_WO_B_SCALE,
            WO_B_ROWS,
            WO_B_COLS,
        )?;
        let expected_embed_bytes = HIDDEN_SIZE * 2;
        if embed.len() != expected_embed_bytes
            || hc_fn.len() != HC_MIX_WIDTH * HC_FLAT_WIDTH * 4
            || hc_base.len() != HC_MIX_WIDTH * 4
            || hc_scale.len() != 3 * 4
            || attn_norm.len() != HIDDEN_SIZE * 2
            || q_norm.len() != Q_LORA_RANK * 2
            || kv_norm.len() != HEAD_DIM * 2
            || sink.len() != NUM_HEADS * 4
        {
            return Err(failure("P4A source tensor geometry changed"));
        }

        // Separate raw-payload FP64 references.  These never obtain their
        // values by re-promoting source CPU f32 outputs.
        let hc_f64 = hc_pre_f64(&embed, &hc_fn, &hc_scale, &hc_base)?;
        let wkv_f64 = fp8_f64(
            &wkv_weight,
            &wkv_scale,
            &cpu.wkv.quantized_input.activation_e4m3fn,
            &cpu.wkv.quantized_input.scales_e8m0fnu,
            WKV_ROWS,
            HIDDEN_SIZE,
        )?;
        let kv_norm_f64 = bf16_store_ref(
            rmsnorm_f64(
                &bf16_bits_f64(&cpu.wkv.output.bf16_bits),
                &bf16_bytes_f64(&kv_norm)?,
                RMS_NORM_EPS as f64,
            )
            .map_err(failure)?,
        );
        let kv_qat_f64 = kv_qat_f64(
            &cpu.kv_norm_bf16_bits,
            &cpu.kv_inplace_qat.non_rope_activation_e4m3fn,
            &cpu.kv_inplace_qat.non_rope_scales_e8m0fnu,
        )?;
        let sparse_f64 = sparse_f64(
            &cpu.q_position0_rope_bf16_bits,
            &cpu.kv_position0_rope_bf16_bits,
            &f32_bytes_f64(&sink)?,
        )?;
        let wo_a_f64 = wo_a_f64(
            &cpu.sparse_attention_derotated_bf16_bits,
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
        let hc_post_f64 = hc_post_f64(&cpu.wo_b.output.bf16_bits, &embed, &hc_f64)?;

        let context = MetalContext::new_with_trace(true)?;
        let device_name = context.device_name();
        let mut pipeline_limits = serde_json::Map::new();
        for kernel in [
            HC_KERNEL,
            RMS_KERNEL,
            QAT_KERNEL,
            FP8_KERNEL,
            CAST_KERNEL,
            PER_HEAD_KERNEL,
            KV_QAT_KERNEL,
            SPARSE_KERNEL,
            WO_A_KERNEL,
            HC_POST_KERNEL,
        ] {
            let pipeline = context.pipeline(kernel)?;
            pipeline_limits.insert(
                kernel.to_owned(),
                json!({
                    "thread_execution_width": pipeline.thread_execution_width(),
                    "max_total_threads_per_threadgroup": pipeline.max_total_threads_per_threadgroup(),
                }),
            );
        }

        // P3A precursor buffers.
        let embed_b = context.new_buffer_with_bytes_checked(&embed)?;
        let hc_fn_b = context.new_buffer_with_bytes_checked(&hc_fn)?;
        let hc_scale_b = context.new_buffer_with_bytes_checked(&hc_scale)?;
        let hc_base_b = context.new_buffer_with_bytes_checked(&hc_base)?;
        let hc_reduced_b = context.new_buffer_checked(HIDDEN_SIZE * 2)?;
        let hc_rsqrt_b = context.new_buffer_checked(4)?;
        let hc_mixes_b = context.new_buffer_checked(HC_MIX_WIDTH * 4)?;
        let hc_pre_b = context.new_buffer_checked(HC_MULT * 4)?;
        let hc_post_b = context.new_buffer_checked(HC_MULT * 4)?;
        let hc_comb_b = context.new_buffer_checked(HC_MULT * HC_MULT * 4)?;
        let attn_norm_w_b = context.new_buffer_with_bytes_checked(&attn_norm)?;
        let attn_norm_out_b = context.new_buffer_checked(HIDDEN_SIZE * 2)?;
        let wq_a_w_b = context.new_buffer_with_bytes_checked(&wq_a_weight)?;
        let wq_a_s_b = context.new_buffer_with_bytes_checked(&wq_a_scale)?;
        let wq_a_act_b = context.new_buffer_checked(LAYER0_WQ_A_COLS)?;
        let wq_a_act_s_b = context.new_buffer_checked(LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK)?;
        let wq_a_out_b = context.new_buffer_checked(LAYER0_WQ_A_ROWS * 4)?;
        let wq_a_bf16_b = context.new_buffer_checked(LAYER0_WQ_A_ROWS * 2)?;
        let q_norm_w_b = context.new_buffer_with_bytes_checked(&q_norm)?;
        let q_norm_out_b = context.new_buffer_checked(Q_LORA_RANK * 2)?;
        let wq_b_w_b = context.new_buffer_with_bytes_checked(&wq_b_weight)?;
        let wq_b_s_b = context.new_buffer_with_bytes_checked(&wq_b_scale)?;
        let wq_b_act_b = context.new_buffer_checked(Q_LORA_RANK)?;
        let wq_b_act_s_b = context.new_buffer_checked(Q_LORA_RANK / ACT_QUANT_BLOCK)?;
        let wq_b_out_b = context.new_buffer_checked(WQ_B_ROWS * 4)?;
        let wq_b_bf16_b = context.new_buffer_checked(WQ_B_ROWS * 2)?;
        let q_head_b = context.new_buffer_checked(WQ_B_ROWS * 2)?;

        // KV, sparse, WO-A/WO-B, and mHC-post continuation buffers.
        let wkv_w_b = context.new_buffer_with_bytes_checked(&wkv_weight)?;
        let wkv_s_b = context.new_buffer_with_bytes_checked(&wkv_scale)?;
        let wkv_act_b = context.new_buffer_checked(HIDDEN_SIZE)?;
        let wkv_act_s_b = context.new_buffer_checked(HIDDEN_SIZE / ACT_QUANT_BLOCK)?;
        let wkv_out_b = context.new_buffer_checked(WKV_ROWS * 4)?;
        let wkv_bf16_b = context.new_buffer_checked(WKV_ROWS * 2)?;
        let kv_norm_w_b = context.new_buffer_with_bytes_checked(&kv_norm)?;
        let kv_norm_out_b = context.new_buffer_checked(HEAD_DIM * 2)?;
        let kv_qat_out_b = context.new_buffer_checked(HEAD_DIM * 2)?;
        let kv_qat_act_b = context.new_buffer_checked(NON_ROPE_HEAD_DIM)?;
        let kv_qat_s_b = context.new_buffer_checked(NON_ROPE_HEAD_DIM / KV_QAT_BLOCK)?;
        let sink_b = context.new_buffer_with_bytes_checked(&sink)?;
        let sparse_out_b = context.new_buffer_checked(WQ_B_ROWS * 2)?;
        let sparse_scores_b = context.new_buffer_checked(NUM_HEADS * 4)?;
        let sparse_denoms_b = context.new_buffer_checked(NUM_HEADS * 4)?;
        let wo_a_w_b = context.new_buffer_with_bytes_checked(&wo_a_weight)?;
        let wo_a_s_b = context.new_buffer_with_bytes_checked(&wo_a_scale)?;
        let wo_a_out_b = context.new_buffer_checked(WO_A_ROWS * 2)?;
        let wo_b_w_b = context.new_buffer_with_bytes_checked(&wo_b_weight)?;
        let wo_b_s_b = context.new_buffer_with_bytes_checked(&wo_b_scale)?;
        let wo_b_act_b = context.new_buffer_checked(WO_B_COLS)?;
        let wo_b_act_s_b = context.new_buffer_checked(WO_B_COLS / ACT_QUANT_BLOCK)?;
        let wo_b_out_b = context.new_buffer_checked(WO_B_ROWS * 4)?;
        let wo_b_bf16_b = context.new_buffer_checked(WO_B_ROWS * 2)?;
        let hc_final_b = context.new_buffer_checked(HC_MULT * HIDDEN_SIZE * 2)?;

        let run_nonce = sha256_join(&[
            reader.manifest_seal_sha256(),
            &cpu_binding.seal_sha256,
            &p3a_binding.seal_sha256,
            &sha256(&embed),
            &sha256(&wo_a_weight),
            &sha256(&wo_b_weight),
            "dsv4f_p4a_complete_layer0_attention_v1",
        ]);
        let interval_id = sha256_join(&[&run_nonce, "p4a_complete_layer0_attention_chain"]);
        let trace = PhysicalTraceGuard::begin(PhysicalTraceIdentity::new(
            interval_id.clone(),
            run_nonce.clone(),
            "dsv4f_p4a_layer0_attention".to_owned(),
            "complete_attention_block".to_owned(),
            Some(1),
            0,
        )?)?;

        let hidden = HIDDEN_SIZE as u32;
        let hc_mult = HC_MULT as u32;
        let mix_width = HC_MIX_WIDTH as u32;
        let sinkhorn = HC_SINKHORN_ITERS as u32;
        let norm_eps = RMS_NORM_EPS;
        let hc_eps = HC_EPS;
        let q_lora = Q_LORA_RANK as u32;
        let heads = NUM_HEADS as u32;
        let head_dim = HEAD_DIM as u32;
        let wq_a_rows = LAYER0_WQ_A_ROWS as u32;
        let wq_a_cols = LAYER0_WQ_A_COLS as u32;
        let wq_a_scale_cols = (LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK) as u32;
        let wq_b_rows = WQ_B_ROWS as u32;
        let wq_b_scale_cols = (Q_LORA_RANK / ACT_QUANT_BLOCK) as u32;
        let wkv_rows = WKV_ROWS as u32;
        let wkv_scale_cols = (HIDDEN_SIZE / ACT_QUANT_BLOCK) as u32;
        let kv_block = KV_QAT_BLOCK as u32;
        let rope_dim = ROPE_HEAD_DIM as u32;
        let sparse_scale = (HEAD_DIM as f32).powf(-0.5);
        let wo_a_rows = WO_A_ROWS as u32;
        let wo_a_cols = WO_A_COLS as u32;
        let wo_a_scale_cols = (WO_A_COLS / ACT_QUANT_BLOCK) as u32;
        let o_rank = O_LORA_RANK as u32;
        let wo_b_rows = WO_B_ROWS as u32;
        let wo_b_cols = WO_B_COLS as u32;
        let wo_b_scale_cols = (WO_B_COLS / ACT_QUANT_BLOCK) as u32;

        // P3A is recomputed here in the same physical trace. No output is
        // read to CPU until the full P4A attention chain has completed.
        let hc_t = context.dispatch_threads_timed(HC_KERNEL, (1, 1, 1), (1, 1, 1), |e| {
            e.set_buffer(0, Some(&embed_b), 0);
            e.set_buffer(1, Some(&hc_fn_b), 0);
            e.set_buffer(2, Some(&hc_scale_b), 0);
            e.set_buffer(3, Some(&hc_base_b), 0);
            e.set_buffer(4, Some(&hc_reduced_b), 0);
            e.set_buffer(5, Some(&hc_rsqrt_b), 0);
            e.set_buffer(6, Some(&hc_mixes_b), 0);
            e.set_buffer(7, Some(&hc_pre_b), 0);
            e.set_buffer(8, Some(&hc_post_b), 0);
            e.set_buffer(9, Some(&hc_comb_b), 0);
            set_u32(e, 10, &hidden);
            set_u32(e, 11, &hc_mult);
            set_u32(e, 12, &mix_width);
            set_u32(e, 13, &sinkhorn);
            set_f32(e, 14, &norm_eps);
            set_f32(e, 15, &hc_eps);
        })?;
        checked(&hc_t, "mHC pre")?;
        let attn_norm_t =
            context.dispatch_threads_timed(RMS_KERNEL, (1, 1, 1), (1, 1, 1), |e| {
                e.set_buffer(0, Some(&hc_reduced_b), 0);
                e.set_buffer(1, Some(&attn_norm_w_b), 0);
                e.set_buffer(2, Some(&attn_norm_out_b), 0);
                set_u32(e, 3, &hidden);
                set_f32(e, 4, &norm_eps);
            })?;
        checked(&attn_norm_t, "attention RMSNorm")?;
        let wq_a_qat_t = qat(
            &context,
            &attn_norm_out_b,
            &wq_a_act_b,
            &wq_a_act_s_b,
            wq_a_cols,
        )?;
        let wq_a_t = fp8(
            &context,
            &wq_a_w_b,
            &wq_a_s_b,
            &wq_a_act_b,
            &wq_a_act_s_b,
            &wq_a_out_b,
            wq_a_rows,
            wq_a_cols,
            wq_a_scale_cols,
        )?;
        let wq_a_cast_t = cast(&context, &wq_a_out_b, &wq_a_bf16_b, wq_a_rows)?;
        let q_norm_t = context.dispatch_threads_timed(RMS_KERNEL, (1, 1, 1), (1, 1, 1), |e| {
            e.set_buffer(0, Some(&wq_a_bf16_b), 0);
            e.set_buffer(1, Some(&q_norm_w_b), 0);
            e.set_buffer(2, Some(&q_norm_out_b), 0);
            set_u32(e, 3, &q_lora);
            set_f32(e, 4, &norm_eps);
        })?;
        checked(&q_norm_t, "Q RMSNorm")?;
        let wq_b_qat_t = qat(&context, &q_norm_out_b, &wq_b_act_b, &wq_b_act_s_b, q_lora)?;
        let wq_b_t = fp8(
            &context,
            &wq_b_w_b,
            &wq_b_s_b,
            &wq_b_act_b,
            &wq_b_act_s_b,
            &wq_b_out_b,
            wq_b_rows,
            q_lora,
            wq_b_scale_cols,
        )?;
        let wq_b_cast_t = cast(&context, &wq_b_out_b, &wq_b_bf16_b, wq_b_rows)?;
        let q_head_t =
            context.dispatch_threads_timed(PER_HEAD_KERNEL, (heads, 1, 1), (64, 1, 1), |e| {
                e.set_buffer(0, Some(&wq_b_bf16_b), 0);
                e.set_buffer(1, Some(&q_head_b), 0);
                set_u32(e, 2, &heads);
                set_u32(e, 3, &head_dim);
                set_f32(e, 4, &norm_eps);
            })?;
        checked(&q_head_t, "per-head Q RMSNorm")?;

        let wkv_qat_t = qat(&context, &attn_norm_out_b, &wkv_act_b, &wkv_act_s_b, hidden)?;
        let wkv_t = fp8(
            &context,
            &wkv_w_b,
            &wkv_s_b,
            &wkv_act_b,
            &wkv_act_s_b,
            &wkv_out_b,
            wkv_rows,
            hidden,
            wkv_scale_cols,
        )?;
        let wkv_cast_t = cast(&context, &wkv_out_b, &wkv_bf16_b, wkv_rows)?;
        let kv_norm_t = context.dispatch_threads_timed(RMS_KERNEL, (1, 1, 1), (1, 1, 1), |e| {
            e.set_buffer(0, Some(&wkv_bf16_b), 0);
            e.set_buffer(1, Some(&kv_norm_w_b), 0);
            e.set_buffer(2, Some(&kv_norm_out_b), 0);
            set_u32(e, 3, &head_dim);
            set_f32(e, 4, &norm_eps);
        })?;
        checked(&kv_norm_t, "KV RMSNorm")?;
        let kv_qat_t = context.dispatch_threads_timed(
            KV_QAT_KERNEL,
            (NON_ROPE_HEAD_DIM as u32 / kv_block, 1, 1),
            (32, 1, 1),
            |e| {
                e.set_buffer(0, Some(&kv_norm_out_b), 0);
                e.set_buffer(1, Some(&kv_qat_out_b), 0);
                e.set_buffer(2, Some(&kv_qat_act_b), 0);
                e.set_buffer(3, Some(&kv_qat_s_b), 0);
                set_u32(e, 4, &head_dim);
                set_u32(e, 5, &rope_dim);
                set_u32(e, 6, &kv_block);
            },
        )?;
        checked(&kv_qat_t, "KV non-RoPE QAT")?;
        let sparse_t =
            context.dispatch_threads_timed(SPARSE_KERNEL, (heads, 1, 1), (64, 1, 1), |e| {
                e.set_buffer(0, Some(&q_head_b), 0);
                e.set_buffer(1, Some(&kv_qat_out_b), 0);
                e.set_buffer(2, Some(&sink_b), 0);
                e.set_buffer(3, Some(&sparse_out_b), 0);
                e.set_buffer(4, Some(&sparse_scores_b), 0);
                e.set_buffer(5, Some(&sparse_denoms_b), 0);
                set_u32(e, 6, &heads);
                set_u32(e, 7, &head_dim);
                set_f32(e, 8, &sparse_scale);
            })?;
        checked(&sparse_t, "position-zero sparse attention/sink")?;
        let wo_a_t =
            context.dispatch_threads_timed(WO_A_KERNEL, (wo_a_rows, 1, 1), (256, 1, 1), |e| {
                e.set_buffer(0, Some(&wo_a_w_b), 0);
                e.set_buffer(1, Some(&wo_a_s_b), 0);
                e.set_buffer(2, Some(&sparse_out_b), 0);
                e.set_buffer(3, Some(&wo_a_out_b), 0);
                set_u32(e, 4, &wo_a_rows);
                set_u32(e, 5, &wo_a_cols);
                set_u32(e, 6, &wo_a_scale_cols);
                set_u32(e, 7, &o_rank);
            })?;
        checked(&wo_a_t, "WO-A conversion/einsum")?;
        let wo_b_qat_t = qat(&context, &wo_a_out_b, &wo_b_act_b, &wo_b_act_s_b, wo_b_cols)?;
        let wo_b_t = fp8(
            &context,
            &wo_b_w_b,
            &wo_b_s_b,
            &wo_b_act_b,
            &wo_b_act_s_b,
            &wo_b_out_b,
            wo_b_rows,
            wo_b_cols,
            wo_b_scale_cols,
        )?;
        let wo_b_cast_t = cast(&context, &wo_b_out_b, &wo_b_bf16_b, wo_b_rows)?;
        let hc_post_t = context.dispatch_threads_timed(
            HC_POST_KERNEL,
            (hidden * hc_mult, 1, 1),
            (256, 1, 1),
            |e| {
                e.set_buffer(0, Some(&wo_b_bf16_b), 0);
                e.set_buffer(1, Some(&embed_b), 0);
                e.set_buffer(2, Some(&hc_post_b), 0);
                e.set_buffer(3, Some(&hc_comb_b), 0);
                e.set_buffer(4, Some(&hc_final_b), 0);
                set_u32(e, 5, &hidden);
                set_u32(e, 6, &hc_mult);
            },
        )?;
        checked(&hc_post_t, "mHC attention post")?;

        let counts = trace.counts();
        drop(trace);
        let (buffers_created, bytes_allocated, commits) = context.drain_stats();
        let trace_samples = context.drain_trace();
        if counts.command_count != P4A_DISPATCHES
            || counts.encoder_count != P4A_DISPATCHES
            || commits as u64 != P4A_DISPATCHES
            || trace_samples.len() as u64 != P4A_DISPATCHES
        {
            return Err(failure("P4A command topology accounting is incomplete"));
        }

        // This macro deliberately retains every authority-stage dispatch and
        // buffer binding.  The topology candidate below changes *only* the
        // command-buffer boundary: 21 ordered compute encoders are encoded
        // into one completed command buffer.  It does not assert an unsafe
        // same-encoder dependency or a concurrent wave.
        macro_rules! p4a_complete_chain {
            ($dispatch:ident) => {{
                $dispatch!(HC_KERNEL, (1, 1, 1), (1, 1, 1), |e| {
                    e.set_buffer(0, Some(&embed_b), 0);
                    e.set_buffer(1, Some(&hc_fn_b), 0);
                    e.set_buffer(2, Some(&hc_scale_b), 0);
                    e.set_buffer(3, Some(&hc_base_b), 0);
                    e.set_buffer(4, Some(&hc_reduced_b), 0);
                    e.set_buffer(5, Some(&hc_rsqrt_b), 0);
                    e.set_buffer(6, Some(&hc_mixes_b), 0);
                    e.set_buffer(7, Some(&hc_pre_b), 0);
                    e.set_buffer(8, Some(&hc_post_b), 0);
                    e.set_buffer(9, Some(&hc_comb_b), 0);
                    set_u32(e, 10, &hidden);
                    set_u32(e, 11, &hc_mult);
                    set_u32(e, 12, &mix_width);
                    set_u32(e, 13, &sinkhorn);
                    set_f32(e, 14, &norm_eps);
                    set_f32(e, 15, &hc_eps);
                });
                $dispatch!(RMS_KERNEL, (1, 1, 1), (1, 1, 1), |e| {
                    e.set_buffer(0, Some(&hc_reduced_b), 0);
                    e.set_buffer(1, Some(&attn_norm_w_b), 0);
                    e.set_buffer(2, Some(&attn_norm_out_b), 0);
                    set_u32(e, 3, &hidden);
                    set_f32(e, 4, &norm_eps);
                });
                $dispatch!(
                    QAT_KERNEL,
                    (wq_a_cols / ACT_QUANT_BLOCK as u32, 1, 1),
                    (32, 1, 1),
                    |e| {
                        e.set_buffer(0, Some(&attn_norm_out_b), 0);
                        e.set_buffer(1, Some(&wq_a_act_b), 0);
                        e.set_buffer(2, Some(&wq_a_act_s_b), 0);
                        set_u32(e, 3, &wq_a_cols);
                    }
                );
                $dispatch!(FP8_KERNEL, (wq_a_rows, 1, 1), (256, 1, 1), |e| {
                    e.set_buffer(0, Some(&wq_a_w_b), 0);
                    e.set_buffer(1, Some(&wq_a_s_b), 0);
                    e.set_buffer(2, Some(&wq_a_act_b), 0);
                    e.set_buffer(3, Some(&wq_a_act_s_b), 0);
                    e.set_buffer(4, Some(&wq_a_out_b), 0);
                    set_u32(e, 5, &wq_a_rows);
                    set_u32(e, 6, &wq_a_cols);
                    set_u32(e, 7, &wq_a_scale_cols);
                });
                $dispatch!(CAST_KERNEL, (wq_a_rows, 1, 1), (256, 1, 1), |e| {
                    e.set_buffer(0, Some(&wq_a_out_b), 0);
                    e.set_buffer(1, Some(&wq_a_bf16_b), 0);
                    set_u32(e, 2, &wq_a_rows);
                });
                $dispatch!(RMS_KERNEL, (1, 1, 1), (1, 1, 1), |e| {
                    e.set_buffer(0, Some(&wq_a_bf16_b), 0);
                    e.set_buffer(1, Some(&q_norm_w_b), 0);
                    e.set_buffer(2, Some(&q_norm_out_b), 0);
                    set_u32(e, 3, &q_lora);
                    set_f32(e, 4, &norm_eps);
                });
                $dispatch!(
                    QAT_KERNEL,
                    (q_lora / ACT_QUANT_BLOCK as u32, 1, 1),
                    (32, 1, 1),
                    |e| {
                        e.set_buffer(0, Some(&q_norm_out_b), 0);
                        e.set_buffer(1, Some(&wq_b_act_b), 0);
                        e.set_buffer(2, Some(&wq_b_act_s_b), 0);
                        set_u32(e, 3, &q_lora);
                    }
                );
                $dispatch!(FP8_KERNEL, (wq_b_rows, 1, 1), (256, 1, 1), |e| {
                    e.set_buffer(0, Some(&wq_b_w_b), 0);
                    e.set_buffer(1, Some(&wq_b_s_b), 0);
                    e.set_buffer(2, Some(&wq_b_act_b), 0);
                    e.set_buffer(3, Some(&wq_b_act_s_b), 0);
                    e.set_buffer(4, Some(&wq_b_out_b), 0);
                    set_u32(e, 5, &wq_b_rows);
                    set_u32(e, 6, &q_lora);
                    set_u32(e, 7, &wq_b_scale_cols);
                });
                $dispatch!(CAST_KERNEL, (wq_b_rows, 1, 1), (256, 1, 1), |e| {
                    e.set_buffer(0, Some(&wq_b_out_b), 0);
                    e.set_buffer(1, Some(&wq_b_bf16_b), 0);
                    set_u32(e, 2, &wq_b_rows);
                });
                $dispatch!(PER_HEAD_KERNEL, (heads, 1, 1), (64, 1, 1), |e| {
                    e.set_buffer(0, Some(&wq_b_bf16_b), 0);
                    e.set_buffer(1, Some(&q_head_b), 0);
                    set_u32(e, 2, &heads);
                    set_u32(e, 3, &head_dim);
                    set_f32(e, 4, &norm_eps);
                });
                $dispatch!(
                    QAT_KERNEL,
                    (hidden / ACT_QUANT_BLOCK as u32, 1, 1),
                    (32, 1, 1),
                    |e| {
                        e.set_buffer(0, Some(&attn_norm_out_b), 0);
                        e.set_buffer(1, Some(&wkv_act_b), 0);
                        e.set_buffer(2, Some(&wkv_act_s_b), 0);
                        set_u32(e, 3, &hidden);
                    }
                );
                $dispatch!(FP8_KERNEL, (wkv_rows, 1, 1), (256, 1, 1), |e| {
                    e.set_buffer(0, Some(&wkv_w_b), 0);
                    e.set_buffer(1, Some(&wkv_s_b), 0);
                    e.set_buffer(2, Some(&wkv_act_b), 0);
                    e.set_buffer(3, Some(&wkv_act_s_b), 0);
                    e.set_buffer(4, Some(&wkv_out_b), 0);
                    set_u32(e, 5, &wkv_rows);
                    set_u32(e, 6, &hidden);
                    set_u32(e, 7, &wkv_scale_cols);
                });
                $dispatch!(CAST_KERNEL, (wkv_rows, 1, 1), (256, 1, 1), |e| {
                    e.set_buffer(0, Some(&wkv_out_b), 0);
                    e.set_buffer(1, Some(&wkv_bf16_b), 0);
                    set_u32(e, 2, &wkv_rows);
                });
                $dispatch!(RMS_KERNEL, (1, 1, 1), (1, 1, 1), |e| {
                    e.set_buffer(0, Some(&wkv_bf16_b), 0);
                    e.set_buffer(1, Some(&kv_norm_w_b), 0);
                    e.set_buffer(2, Some(&kv_norm_out_b), 0);
                    set_u32(e, 3, &head_dim);
                    set_f32(e, 4, &norm_eps);
                });
                $dispatch!(
                    KV_QAT_KERNEL,
                    (NON_ROPE_HEAD_DIM as u32 / kv_block, 1, 1),
                    (32, 1, 1),
                    |e| {
                        e.set_buffer(0, Some(&kv_norm_out_b), 0);
                        e.set_buffer(1, Some(&kv_qat_out_b), 0);
                        e.set_buffer(2, Some(&kv_qat_act_b), 0);
                        e.set_buffer(3, Some(&kv_qat_s_b), 0);
                        set_u32(e, 4, &head_dim);
                        set_u32(e, 5, &rope_dim);
                        set_u32(e, 6, &kv_block);
                    }
                );
                $dispatch!(SPARSE_KERNEL, (heads, 1, 1), (64, 1, 1), |e| {
                    e.set_buffer(0, Some(&q_head_b), 0);
                    e.set_buffer(1, Some(&kv_qat_out_b), 0);
                    e.set_buffer(2, Some(&sink_b), 0);
                    e.set_buffer(3, Some(&sparse_out_b), 0);
                    e.set_buffer(4, Some(&sparse_scores_b), 0);
                    e.set_buffer(5, Some(&sparse_denoms_b), 0);
                    set_u32(e, 6, &heads);
                    set_u32(e, 7, &head_dim);
                    set_f32(e, 8, &sparse_scale);
                });
                $dispatch!(WO_A_KERNEL, (wo_a_rows, 1, 1), (256, 1, 1), |e| {
                    e.set_buffer(0, Some(&wo_a_w_b), 0);
                    e.set_buffer(1, Some(&wo_a_s_b), 0);
                    e.set_buffer(2, Some(&sparse_out_b), 0);
                    e.set_buffer(3, Some(&wo_a_out_b), 0);
                    set_u32(e, 4, &wo_a_rows);
                    set_u32(e, 5, &wo_a_cols);
                    set_u32(e, 6, &wo_a_scale_cols);
                    set_u32(e, 7, &o_rank);
                });
                $dispatch!(
                    QAT_KERNEL,
                    (wo_b_cols / ACT_QUANT_BLOCK as u32, 1, 1),
                    (32, 1, 1),
                    |e| {
                        e.set_buffer(0, Some(&wo_a_out_b), 0);
                        e.set_buffer(1, Some(&wo_b_act_b), 0);
                        e.set_buffer(2, Some(&wo_b_act_s_b), 0);
                        set_u32(e, 3, &wo_b_cols);
                    }
                );
                $dispatch!(FP8_KERNEL, (wo_b_rows, 1, 1), (256, 1, 1), |e| {
                    e.set_buffer(0, Some(&wo_b_w_b), 0);
                    e.set_buffer(1, Some(&wo_b_s_b), 0);
                    e.set_buffer(2, Some(&wo_b_act_b), 0);
                    e.set_buffer(3, Some(&wo_b_act_s_b), 0);
                    e.set_buffer(4, Some(&wo_b_out_b), 0);
                    set_u32(e, 5, &wo_b_rows);
                    set_u32(e, 6, &wo_b_cols);
                    set_u32(e, 7, &wo_b_scale_cols);
                });
                $dispatch!(CAST_KERNEL, (wo_b_rows, 1, 1), (256, 1, 1), |e| {
                    e.set_buffer(0, Some(&wo_b_out_b), 0);
                    e.set_buffer(1, Some(&wo_b_bf16_b), 0);
                    set_u32(e, 2, &wo_b_rows);
                });
                $dispatch!(HC_POST_KERNEL, (hidden * hc_mult, 1, 1), (256, 1, 1), |e| {
                    e.set_buffer(0, Some(&wo_b_bf16_b), 0);
                    e.set_buffer(1, Some(&embed_b), 0);
                    e.set_buffer(2, Some(&hc_post_b), 0);
                    e.set_buffer(3, Some(&hc_comb_b), 0);
                    e.set_buffer(4, Some(&hc_final_b), 0);
                    set_u32(e, 5, &hidden);
                    set_u32(e, 6, &hc_mult);
                });
            }};
        }

        let topology_pending = if let (Some(topology_out), Some(authority)) =
            (args.topology_out.as_ref(), topology_authority)
        {
            let total_trials = args.topology_warmups + args.topology_trials;
            let topology_nonce = sha256_join(&[
                &run_nonce,
                "p4a_complete_attention_one_cb_ordered_encoder_sweep_v1",
            ]);
            let topology_interval = sha256_join(&[&topology_nonce, "paired_topology_trials"]);
            let topology_trace = PhysicalTraceGuard::begin(PhysicalTraceIdentity::new(
                topology_interval,
                topology_nonce,
                "dsv4f_p4a_layer0_attention_topology".to_owned(),
                "same_model_21cb_vs_1cb_21ordered_encoders".to_owned(),
                Some(1),
                0,
            )?)?;
            let run_separate_trial = || -> ProbeResult<(u64, u64)> {
                let mut gpu_us = 0u64;
                let mut host_wall_us = 0u64;
                macro_rules! direct_measure {
                    ($kernel:expr, $grid:expr, $tg:expr, |$encoder:ident| $encode:block) => {{
                        let timing =
                            context
                                .dispatch_threads_timed($kernel, $grid, $tg, |$encoder| $encode)?;
                        checked(&timing, $kernel)?;
                        gpu_us += timing.gpu_duration_us.ok_or_else(|| {
                            failure("completed direct P4A timing lacks GPU duration")
                        })?;
                        host_wall_us += timing.host_wall_us;
                    }};
                }
                p4a_complete_chain!(direct_measure);
                Ok((gpu_us, host_wall_us))
            };
            let run_one_cb_trial = || -> ProbeResult<MetalBatchTiming> {
                Ok(context.dispatch_batch_timed(|batch| {
                    macro_rules! batch_measure {
                        ($kernel:expr, $grid:expr, $tg:expr, |$encoder:ident| $encode:block) => {{
                            batch.dispatch_threads($kernel, $grid, $tg, |$encoder| $encode)?;
                        }};
                    }
                    p4a_complete_chain!(batch_measure);
                    Ok(())
                })?)
            };
            let mut baseline_gpu_us = Vec::with_capacity(args.topology_trials);
            let mut baseline_host_wall_us = Vec::with_capacity(args.topology_trials);
            let mut candidate_gpu_us = Vec::with_capacity(args.topology_trials);
            let mut candidate_host_wall_us = Vec::with_capacity(args.topology_trials);
            let mut candidate_encode_us = Vec::with_capacity(args.topology_trials);
            let mut candidate_submit_us = Vec::with_capacity(args.topology_trials);
            let mut candidate_wait_us = Vec::with_capacity(args.topology_trials);
            for iteration in 0..total_trials {
                let (base_gpu, base_host_wall) = run_separate_trial()?;
                let candidate = run_one_cb_trial()?;
                if candidate.command_buffers != 1
                    || candidate.compute_encoders != P4A_DISPATCHES
                    || candidate.compute_dispatches != P4A_DISPATCHES
                    || candidate.gpu_duration_us.is_none()
                    || candidate.gpu_start_ns.is_none()
                    || candidate.gpu_end_ns.is_none()
                {
                    return Err(failure(
                        "one-CB P4A topology candidate did not expose complete completed-CB timing/accounting",
                    ));
                }
                if iteration >= args.topology_warmups {
                    baseline_gpu_us.push(base_gpu);
                    baseline_host_wall_us.push(base_host_wall);
                    candidate_gpu_us.push(candidate.gpu_duration_us.expect("checked above"));
                    candidate_host_wall_us.push(candidate.host_wall_us);
                    candidate_encode_us.push(candidate.encode_us);
                    candidate_submit_us.push(candidate.submit_us);
                    candidate_wait_us.push(candidate.wait_us);
                }
            }
            let topology_counts = topology_trace.counts();
            drop(topology_trace);
            let topology_trace_samples = context.drain_trace();
            let (topology_buffers_created, topology_bytes_allocated, topology_commits) =
                context.drain_stats();
            let expected_commands = total_trials as u64 * (P4A_DISPATCHES + 1);
            let expected_encoders = total_trials as u64 * (P4A_DISPATCHES * 2);
            if topology_counts.command_count != expected_commands
                || topology_counts.encoder_count != expected_encoders
                || topology_trace_samples.len() as u64 != expected_commands
                || topology_commits as u64 != expected_commands
            {
                return Err(failure(format!(
                    "P4A topology physical accounting mismatch: commands={}/{} encoders={}/{} trace={}/{} commits={}/{}",
                    topology_counts.command_count,
                    expected_commands,
                    topology_counts.encoder_count,
                    expected_encoders,
                    topology_trace_samples.len(),
                    expected_commands,
                    topology_commits,
                    expected_commands,
                )));
            }
            Some(TopologyPending {
                out: topology_out.clone(),
                authority,
                warmups: args.topology_warmups,
                trials: args.topology_trials,
                baseline_gpu_us,
                baseline_host_wall_us,
                candidate_gpu_us,
                candidate_host_wall_us,
                candidate_encode_us,
                candidate_submit_us,
                candidate_wait_us,
                physical_command_buffers: topology_counts.command_count,
                physical_compute_encoders: topology_counts.encoder_count,
                trace_samples: topology_trace_samples.len(),
                commits: topology_commits,
                buffers_created: topology_buffers_created,
                bytes_allocated: topology_bytes_allocated,
            })
        } else {
            None
        };

        let gpu_hc = u16read(&hc_reduced_b, HIDDEN_SIZE)?;
        let gpu_attn_norm = u16read(&attn_norm_out_b, HIDDEN_SIZE)?;
        let gpu_wq_a_act = bytesread(&wq_a_act_b, LAYER0_WQ_A_COLS)?;
        let gpu_wq_a_s = bytesread(&wq_a_act_s_b, LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK)?;
        let gpu_wq_a = f32read(&wq_a_out_b, LAYER0_WQ_A_ROWS)?;
        let gpu_wq_a_b = u16read(&wq_a_bf16_b, LAYER0_WQ_A_ROWS)?;
        let gpu_qnorm = u16read(&q_norm_out_b, Q_LORA_RANK)?;
        let gpu_wq_b_act = bytesread(&wq_b_act_b, Q_LORA_RANK)?;
        let gpu_wq_b_s = bytesread(&wq_b_act_s_b, Q_LORA_RANK / ACT_QUANT_BLOCK)?;
        let gpu_wq_b = f32read(&wq_b_out_b, WQ_B_ROWS)?;
        let gpu_q = u16read(&q_head_b, WQ_B_ROWS)?;
        let gpu_wkv_act = bytesread(&wkv_act_b, HIDDEN_SIZE)?;
        let gpu_wkv_s = bytesread(&wkv_act_s_b, HIDDEN_SIZE / ACT_QUANT_BLOCK)?;
        let gpu_wkv = f32read(&wkv_out_b, WKV_ROWS)?;
        let gpu_wkv_b = u16read(&wkv_bf16_b, WKV_ROWS)?;
        let gpu_kvnorm = u16read(&kv_norm_out_b, HEAD_DIM)?;
        let gpu_kv_qat = u16read(&kv_qat_out_b, HEAD_DIM)?;
        let gpu_kv_act = bytesread(&kv_qat_act_b, NON_ROPE_HEAD_DIM)?;
        let gpu_kv_s = bytesread(&kv_qat_s_b, NON_ROPE_HEAD_DIM / KV_QAT_BLOCK)?;
        let gpu_sparse = f32read(&sparse_scores_b, NUM_HEADS)?;
        let gpu_denoms = f32read(&sparse_denoms_b, NUM_HEADS)?;
        let gpu_attn = u16read(&sparse_out_b, WQ_B_ROWS)?;
        let gpu_woa = u16read(&wo_a_out_b, WO_A_ROWS)?;
        let gpu_wob_act = bytesread(&wo_b_act_b, WO_B_COLS)?;
        let gpu_wob_s = bytesread(&wo_b_act_s_b, WO_B_COLS / ACT_QUANT_BLOCK)?;
        let gpu_wob = f32read(&wo_b_out_b, WO_B_ROWS)?;
        let gpu_wob_b = u16read(&wo_b_bf16_b, WO_B_ROWS)?;
        let gpu_final = u16read(&hc_final_b, HC_MULT * HIDDEN_SIZE)?;

        exact16("mHC reduced", &cpu.prefix.hc_attn_pre_bf16_bits, &gpu_hc)?;
        exact16(
            "attention norm",
            &cpu.prefix.attn_norm_bf16_bits,
            &gpu_attn_norm,
        )?;
        exact8(
            "WQ-A QAT activation",
            &cpu.wq_a.quantized_input.activation_e4m3fn,
            &gpu_wq_a_act,
        )?;
        exact8(
            "WQ-A QAT scale",
            &cpu.wq_a.quantized_input.scales_e8m0fnu,
            &gpu_wq_a_s,
        )?;
        close("WQ-A FP32", &cpu.wq_a.output.fp32, &gpu_wq_a)?;
        exact16("WQ-A BF16", &cpu.wq_a.output.bf16_bits, &gpu_wq_a_b)?;
        exact16("Q norm", &cpu.q_norm_bf16_bits, &gpu_qnorm)?;
        exact8(
            "WQ-B QAT activation",
            &cpu.wq_b.quantized_input.activation_e4m3fn,
            &gpu_wq_b_act,
        )?;
        exact8(
            "WQ-B QAT scale",
            &cpu.wq_b.quantized_input.scales_e8m0fnu,
            &gpu_wq_b_s,
        )?;
        close("WQ-B FP32", &cpu.wq_b.output.fp32, &gpu_wq_b)?;
        exact16("Q head norm", &cpu.q_head_norm_bf16_bits, &gpu_q)?;
        exact16(
            "Q position-zero rope",
            &cpu.q_position0_rope_bf16_bits,
            &gpu_q,
        )?;
        exact8(
            "WKV QAT activation",
            &cpu.wkv.quantized_input.activation_e4m3fn,
            &gpu_wkv_act,
        )?;
        exact8(
            "WKV QAT scale",
            &cpu.wkv.quantized_input.scales_e8m0fnu,
            &gpu_wkv_s,
        )?;
        close("WKV FP32", &cpu.wkv.output.fp32, &gpu_wkv)?;
        exact16("WKV BF16", &cpu.wkv.output.bf16_bits, &gpu_wkv_b)?;
        exact16("KV norm", &cpu.kv_norm_bf16_bits, &gpu_kvnorm)?;
        exact8(
            "KV QAT activation",
            &cpu.kv_inplace_qat.non_rope_activation_e4m3fn,
            &gpu_kv_act,
        )?;
        exact8(
            "KV QAT scale",
            &cpu.kv_inplace_qat.non_rope_scales_e8m0fnu,
            &gpu_kv_s,
        )?;
        exact16(
            "KV QAT output",
            &cpu.kv_inplace_qat.output_bf16_bits,
            &gpu_kv_qat,
        )?;
        close(
            "sparse scores",
            &cpu.sparse_attention_scores_f32,
            &gpu_sparse,
        )?;
        close(
            "sparse sink denominators",
            &cpu.sparse_attention_sink_denominators_f32,
            &gpu_denoms,
        )?;
        exact16(
            "sparse attention",
            &cpu.sparse_attention_bf16_bits,
            &gpu_attn,
        )?;
        exact16(
            "attention position-zero derotation",
            &cpu.sparse_attention_derotated_bf16_bits,
            &gpu_attn,
        )?;
        exact16("WO-A converted BF16 einsum", &cpu.wo_a_bf16_bits, &gpu_woa)?;
        exact8(
            "WO-B QAT activation",
            &cpu.wo_b.quantized_input.activation_e4m3fn,
            &gpu_wob_act,
        )?;
        exact8(
            "WO-B QAT scale",
            &cpu.wo_b.quantized_input.scales_e8m0fnu,
            &gpu_wob_s,
        )?;
        close("WO-B FP32", &cpu.wo_b.output.fp32, &gpu_wob)?;
        exact16("WO-B BF16", &cpu.wo_b.output.bf16_bits, &gpu_wob_b)?;
        exact16(
            "mHC attention post",
            &cpu.hc_attn_post_bf16_bits,
            &gpu_final,
        )?;

        let fb = f32_bounds();
        let bb = bf16_bounds();
        let scores = [
            (
                "wkv_fp8_f32",
                score("WKV FP8", &cpu.wkv.output.fp32, &gpu_wkv, &wkv_f64, &fb)?,
            ),
            (
                "kv_rmsnorm_bf16",
                score(
                    "KV RMSNorm",
                    &bf16f32(&cpu.kv_norm_bf16_bits),
                    &bf16f32(&gpu_kvnorm),
                    &kv_norm_f64,
                    &bb,
                )?,
            ),
            (
                "kv_qat_bf16",
                score(
                    "KV QAT",
                    &bf16f32(&cpu.kv_inplace_qat.output_bf16_bits),
                    &bf16f32(&gpu_kv_qat),
                    &kv_qat_f64,
                    &bb,
                )?,
            ),
            (
                "sparse_scores_f32",
                score(
                    "Sparse scores",
                    &cpu.sparse_attention_scores_f32,
                    &gpu_sparse,
                    &sparse_f64.scores,
                    &fb,
                )?,
            ),
            (
                "sparse_sink_denominators_f32",
                score(
                    "Sparse denominators",
                    &cpu.sparse_attention_sink_denominators_f32,
                    &gpu_denoms,
                    &sparse_f64.denominators,
                    &fb,
                )?,
            ),
            (
                "sparse_attention_bf16",
                score(
                    "Sparse output",
                    &bf16f32(&cpu.sparse_attention_bf16_bits),
                    &bf16f32(&gpu_attn),
                    &sparse_f64.output,
                    &bb,
                )?,
            ),
            (
                "wo_a_converted_bf16_einsum",
                score(
                    "WO-A",
                    &bf16f32(&cpu.wo_a_bf16_bits),
                    &bf16f32(&gpu_woa),
                    &wo_a_f64,
                    &bb,
                )?,
            ),
            (
                "wo_b_fp8_f32",
                score("WO-B", &cpu.wo_b.output.fp32, &gpu_wob, &wo_b_f64, &fb)?,
            ),
            (
                "mhc_attention_post_bf16",
                score(
                    "mHC post",
                    &bf16f32(&cpu.hc_attn_post_bf16_bits),
                    &bf16f32(&gpu_final),
                    &hc_post_f64,
                    &bb,
                )?,
            ),
        ];
        let mut score_json = serde_json::Map::new();
        for (name, score) in scores {
            score_json.insert(
                name.to_owned(),
                decimal_strings(serde_json::to_value(score)?),
            );
        }

        let stage_profiles = json!([
            profile(
                "mhc_pre_sinkhorn",
                HC_KERNEL,
                &hc_t,
                hc_fn.len() + embed.len() * HC_MULT,
                HIDDEN_SIZE * 2,
                HC_MIX_WIDTH as u64 * HC_FLAT_WIDTH as u64 * 2
            ),
            profile(
                "attn_rmsnorm",
                RMS_KERNEL,
                &attn_norm_t,
                HIDDEN_SIZE * 4,
                HIDDEN_SIZE * 2,
                HIDDEN_SIZE as u64 * 4
            ),
            profile(
                "wq_a_qat",
                QAT_KERNEL,
                &wq_a_qat_t,
                HIDDEN_SIZE * 2,
                LAYER0_WQ_A_COLS + 32,
                HIDDEN_SIZE as u64 * 2
            ),
            profile(
                "wq_a_fp8",
                FP8_KERNEL,
                &wq_a_t,
                wq_a_weight.len() + wq_a_scale.len() + 4128,
                LAYER0_WQ_A_ROWS * 4,
                (LAYER0_WQ_A_ROWS * LAYER0_WQ_A_COLS * 2) as u64
            ),
            profile(
                "wq_a_cast",
                CAST_KERNEL,
                &wq_a_cast_t,
                LAYER0_WQ_A_ROWS * 4,
                LAYER0_WQ_A_ROWS * 2,
                0
            ),
            profile(
                "q_rmsnorm",
                RMS_KERNEL,
                &q_norm_t,
                Q_LORA_RANK * 4,
                Q_LORA_RANK * 2,
                Q_LORA_RANK as u64 * 4
            ),
            profile(
                "wq_b_qat",
                QAT_KERNEL,
                &wq_b_qat_t,
                Q_LORA_RANK * 2,
                Q_LORA_RANK + 8,
                Q_LORA_RANK as u64 * 2
            ),
            profile(
                "wq_b_fp8",
                FP8_KERNEL,
                &wq_b_t,
                wq_b_weight.len() + wq_b_scale.len() + 1032,
                WQ_B_ROWS * 4,
                (WQ_B_ROWS * Q_LORA_RANK * 2) as u64
            ),
            profile(
                "wq_b_cast",
                CAST_KERNEL,
                &wq_b_cast_t,
                WQ_B_ROWS * 4,
                WQ_B_ROWS * 2,
                0
            ),
            profile(
                "q_head_rmsnorm",
                PER_HEAD_KERNEL,
                &q_head_t,
                WQ_B_ROWS * 2,
                WQ_B_ROWS * 2,
                WQ_B_ROWS as u64 * 3
            ),
            profile(
                "wkv_qat",
                QAT_KERNEL,
                &wkv_qat_t,
                HIDDEN_SIZE * 2,
                HIDDEN_SIZE + 32,
                HIDDEN_SIZE as u64 * 2
            ),
            profile(
                "wkv_fp8",
                FP8_KERNEL,
                &wkv_t,
                wkv_weight.len() + wkv_scale.len() + 4128,
                WKV_ROWS * 4,
                (WKV_ROWS * HIDDEN_SIZE * 2) as u64
            ),
            profile(
                "wkv_cast",
                CAST_KERNEL,
                &wkv_cast_t,
                WKV_ROWS * 4,
                WKV_ROWS * 2,
                0
            ),
            profile(
                "kv_rmsnorm",
                RMS_KERNEL,
                &kv_norm_t,
                HEAD_DIM * 4,
                HEAD_DIM * 2,
                HEAD_DIM as u64 * 4
            ),
            profile(
                "kv_nonrope_qat",
                KV_QAT_KERNEL,
                &kv_qat_t,
                HEAD_DIM * 2,
                HEAD_DIM * 2 + NON_ROPE_HEAD_DIM + 7,
                NON_ROPE_HEAD_DIM as u64 * 2
            ),
            profile(
                "sparse_attention_sink",
                SPARSE_KERNEL,
                &sparse_t,
                WQ_B_ROWS * 2 + HEAD_DIM * 2 + NUM_HEADS * 4,
                WQ_B_ROWS * 2 + NUM_HEADS * 8,
                (WQ_B_ROWS + NUM_HEADS) as u64 * 2
            ),
            profile(
                "wo_a_convert_einsum",
                WO_A_KERNEL,
                &wo_a_t,
                wo_a_weight.len() + wo_a_scale.len() + WQ_B_ROWS * 2,
                WO_A_ROWS * 2,
                (WO_A_ROWS * WO_A_COLS * 2) as u64
            ),
            profile(
                "wo_b_qat",
                QAT_KERNEL,
                &wo_b_qat_t,
                WO_B_COLS * 2,
                WO_B_COLS + 64,
                WO_B_COLS as u64 * 2
            ),
            profile(
                "wo_b_fp8",
                FP8_KERNEL,
                &wo_b_t,
                wo_b_weight.len() + wo_b_scale.len() + WO_B_COLS + 64,
                WO_B_ROWS * 4,
                (WO_B_ROWS * WO_B_COLS * 2) as u64
            ),
            profile(
                "wo_b_cast",
                CAST_KERNEL,
                &wo_b_cast_t,
                WO_B_ROWS * 4,
                WO_B_ROWS * 2,
                0
            ),
            profile(
                "mhc_attention_post",
                HC_POST_KERNEL,
                &hc_post_t,
                WO_B_ROWS * 2 + embed.len() + HC_MULT * 4 + HC_MULT * HC_MULT * 4,
                HC_MULT * HIDDEN_SIZE * 2,
                (HC_MULT * HIDDEN_SIZE * 5) as u64
            ),
        ]);

        let tensors = json!({
            "embed_bos_row": tensor_json(&embed_meta), "hc_fn": tensor_json(&hc_fn_meta), "hc_base": tensor_json(&hc_base_meta), "hc_scale": tensor_json(&hc_scale_meta), "attn_norm": tensor_json(&attn_norm_meta), "q_norm": tensor_json(&q_norm_meta), "kv_norm": tensor_json(&kv_norm_meta), "attn_sink": tensor_json(&sink_meta), "wq_a_weight": tensor_json(&wq_a_meta), "wq_a_scale": tensor_json(&wq_a_scale_meta), "wq_b_weight": tensor_json(&wq_b_meta), "wq_b_scale": tensor_json(&wq_b_scale_meta), "wkv_weight": tensor_json(&wkv_meta), "wkv_scale": tensor_json(&wkv_scale_meta), "wo_a_weight": tensor_json(&wo_a_meta), "wo_a_scale": tensor_json(&wo_a_scale_meta), "wo_b_weight": tensor_json(&wo_b_meta), "wo_b_scale": tensor_json(&wo_b_scale_meta), "all_touched_chunks_sha256_verified_before_gpu_upload": true, "parent_safetensors_materialized": false,
        });
        let unsigned = json!({
            "schema": RECEIPT_SCHEMA, "status": RECEIPT_STATUS,
            "scope": {"token_id": PREFIX_TOKEN_ID,"batch":1,"sequence_tokens":1,"position":0,"compress_ratio":0,"one_selected_causal_kv":true,"all_device_intermediate_chain":true,"not_layer_ffn":true,"not_moe_or_router":true,"not_full_model_or_runtime":true,"not_hcli":true,"not_base_true_tps":true},
            "artifact": {"path":reader.artifact_root().display().to_string(),"full_stream_schema":FULL_STREAM_SCHEMA,"full_stream_status":FULL_STREAM_STATUS,"manifest_file_sha256":reader.manifest_file_sha256(),"manifest_seal_sha256":reader.manifest_seal_sha256(),"restart_receipt_seal_sha256":reader.restart_seal_sha256(),"source_parent_retained":false},
            "source": {"repository":reader.source_identity().repository,"revision":reader.source_identity().revision,"source_hashes":{"inference/model.py":anchors.prefix.act_quant.inference_model_py_sha256,"inference/kernel.py":anchors.prefix.act_quant.inference_kernel_py_sha256,"inference/config.json":anchors.prefix.act_quant.inference_config_json_sha256,"config.json":anchors.prefix.act_quant.model_config_json_sha256,"inference/convert.py":anchors.inference_convert_py_sha256},"tensor_chunk_bindings":tensors},
            "predecessors": {"sealed_cpu_attention_oracle":{"path":cpu_binding.path.display().to_string(),"file_sha256":cpu_binding.file_sha256,"seal_sha256":cpu_binding.seal_sha256,"direct_source_cpu_recomputed":true},"p3a_preattention":{"path":p3a_binding.path.display().to_string(),"file_sha256":p3a_binding.file_sha256,"seal_sha256":p3a_binding.seal_sha256,"role":"validated precursor only; P4A independently recomputes its device path"}},
            "numeric_parity_v2_1": {"schema":"hawking.numeric_parity.v2_1","reference_authority":"separately accumulated FP64 from raw streamed BF16/F32/FP8/E8M0 payloads; declared BF16 output-store boundaries are explicitly materialized before discrete activation parity; source CPU f32 is not the sole authority","f32_operator_bounds":decimal_strings(serde_json::to_value(fb)?),"bf16_storage_bounds":decimal_strings(serde_json::to_value(bb)?),"scores":Value::Object(score_json.clone()),"all_host_and_device_scores_pass":true},
            "discrete_parity":{"p3a_q_path_bf16_and_qat_exact":true,"wkv_qat_and_bf16_exact":true,"kv_norm_and_64wide_nonrope_qat_exact":true,"position_zero_rope_identity_device_noop":true,"sparse_sink_output_exact":true,"wo_a_conversion_semantics_and_grouped_bf16_einsum_exact":true,"wo_b_qat_and_bf16_exact":true,"mhc_attention_post_exact":true},
            "metal":{"device":device_name,"pipelines_precompiled_before_dispatch":true,"pipeline_limits":pipeline_limits,"gpu_dispatches":P4A_DISPATCHES,"command_buffers":P4A_DISPATCHES,"compute_encoders":P4A_DISPATCHES,"cpu_visible_waits":P4A_DISPATCHES,"empty_command_buffers":0,"physical_trace_command_buffers":counts.command_count,"physical_trace_compute_encoders":counts.encoder_count,"trace_samples":trace_samples.len(),"buffers_created":buffers_created,"bytes_allocated":bytes_allocated,"fallback":false,"fallback_count":0,"host_intermediate_handoff_bytes":0},
            "command_topology":{"current":"21 ordered timestamped authority command buffers; no CPU intermediate read/copy/activation/routing","fusion_status":"not promoted: P4A preserves per-stage timestamp/accountability before any same-model command-buffer fusion sweep","next_required_comparison":"replayable one-CB/two-CB topology must preserve this exact discrete and V2.1 gate before any topology win is promoted"},
            "complete_layer0_attention_stage_profile":stage_profiles,
            "physical_trace":{"interval_id":interval_id,"run_nonce":run_nonce,"phase":"dsv4f_p4a_layer0_attention","role":"complete_attention_block"},
            "claim_boundary":"Real Metal executes one complete layer-0 attention block at BOS/position0/ratio0 only. This does not establish the layer FFN, routed experts, a full 43-layer runtime, generation, HCLI, or BASE_TRUE_TPS."
        });
        if let Some(pending) = topology_pending {
            let baseline_gpu = timing_summary(&pending.baseline_gpu_us)?;
            let baseline_wall = timing_summary(&pending.baseline_host_wall_us)?;
            let candidate_gpu = timing_summary(&pending.candidate_gpu_us)?;
            let candidate_wall = timing_summary(&pending.candidate_host_wall_us)?;
            let candidate_encode = timing_summary(&pending.candidate_encode_us)?;
            let candidate_submit = timing_summary(&pending.candidate_submit_us)?;
            let candidate_wait = timing_summary(&pending.candidate_wait_us)?;
            let base_wall_p50 = percentile_u64(&pending.baseline_host_wall_us, 50)?;
            let base_wall_p99 = percentile_u64(&pending.baseline_host_wall_us, 99)?;
            let candidate_wall_p50 = percentile_u64(&pending.candidate_host_wall_us, 50)?;
            let candidate_wall_p99 = percentile_u64(&pending.candidate_host_wall_us, 99)?;
            // A topology reduction is only promoted on completed-command
            // wall time: a lower count with worse p99 is explicitly rejected.
            let promoted =
                candidate_wall_p50 < base_wall_p50 && candidate_wall_p99 <= base_wall_p99;
            let topology_status = if promoted {
                "PASS_REAL_METAL_P4A_ONE_CB_COMPLETE_PARITY_TOPOLOGY_WIN_NOT_RUNTIME"
            } else {
                "PASS_REAL_METAL_P4A_ONE_CB_COMPLETE_PARITY_NONPROMOTED_NOT_RUNTIME"
            };
            let topology_unsigned = json!({
                "schema":P4A_TOPOLOGY_SCHEMA,
                "status":topology_status,
                "scope":{"token_id":PREFIX_TOKEN_ID,"batch":1,"sequence_tokens":1,"position":0,"compress_ratio":0,"complete_layer0_attention":true,"same_model_same_device_buffers":true,"candidate_only_changes_command_buffer_boundary":true,"not_layer_ffn":true,"not_moe_or_router":true,"not_full_model_or_runtime":true,"not_hcli":true,"not_base_true_tps":true},
                "artifact":{"path":reader.artifact_root().display().to_string(),"manifest_seal_sha256":reader.manifest_seal_sha256(),"source_parent_retained":false},
                "source":{"repository":reader.source_identity().repository,"revision":reader.source_identity().revision,"source_hashes":{"inference/model.py":anchors.prefix.act_quant.inference_model_py_sha256,"inference/kernel.py":anchors.prefix.act_quant.inference_kernel_py_sha256,"inference/convert.py":anchors.inference_convert_py_sha256}},
                "predecessor_authority":{"path":pending.authority.path.display().to_string(),"file_sha256":pending.authority.file_sha256,"seal_sha256":pending.authority.seal_sha256,"role":"retained per-stage P4A authority receipt; never replaced by this lower-command candidate"},
                "parity":{"candidate_complete_chain_cpu_discrete_exact":true,"candidate_complete_chain_v2_1_host_and_device_vs_raw_payload_f64":true,"v2_1_scores":Value::Object(score_json),"fallback":false,"host_intermediate_handoff_bytes":0},
                "trial_protocol":{"warmups":pending.warmups,"clean_trials":pending.trials,"paired_order_per_iteration":"21-CB authority baseline then 1-CB candidate","timing_authority":"completed MTLCommandBuffer GPUStartTime/GPUEndTime plus host wall time","pipeline_precompiled_before_trials":true},
                "topologies":{"baseline":{"command_buffers":P4A_DISPATCHES,"compute_encoders":P4A_DISPATCHES,"compute_dispatches":P4A_DISPATCHES,"cpu_visible_waits":P4A_DISPATCHES,"gpu_duration_sum_of_completed_stage_intervals":baseline_gpu,"host_wall_sum":baseline_wall},"candidate":{"command_buffers":1,"compute_encoders":P4A_DISPATCHES,"compute_dispatches":P4A_DISPATCHES,"cpu_visible_waits":1,"ordered_encoder_dependency":"each dispatch remains in a distinct ordered compute encoder; no unproven same-encoder or concurrent dependency is claimed","gpu_duration_completed_command_buffer_interval":candidate_gpu,"host_wall":candidate_wall,"encode":candidate_encode,"submit":candidate_submit,"wait":candidate_wait}},
                "physical_trace":{"command_buffers":pending.physical_command_buffers,"compute_encoders":pending.physical_compute_encoders,"trace_samples":pending.trace_samples,"commits":pending.commits,"buffers_created_during_trials":pending.buffers_created,"bytes_allocated_during_trials":pending.bytes_allocated},
                "promotion":{"promoted":promoted,"rule":"candidate host-wall p50 must improve and candidate host-wall p99 must not regress","baseline_host_wall_p50_us":base_wall_p50,"baseline_host_wall_p99_us":base_wall_p99,"candidate_host_wall_p50_us":candidate_wall_p50,"candidate_host_wall_p99_us":candidate_wall_p99,"next_if_promoted":"retain the P4A per-stage authority receipt and test position-1 causal KV state/read-write (P4B); do not treat this as a full runtime or TPS result","next_if_not_promoted":"retain the P4A per-stage authority receipt and do not substitute this lower-command topology"},
                "claim_boundary":"This is a bounded command-topology comparison for the already sealed BOS/position0/ratio0 complete attention block. It does not establish persistent replay, full-layer execution, generation, HCLI, or BASE_TRUE_TPS."
            });
            let (topology_receipt, topology_seal) = seal(decimal_strings(topology_unsigned))?;
            write_new(&pending.out, &topology_receipt)?;
            println!(
                "{}",
                serde_json::to_string(
                    &json!({"status":topology_status,"receipt":pending.out,"seal_sha256":topology_seal,"promoted":promoted,"fallback":false})
                )?
            );
        } else {
            let (receipt, seal) = seal(unsigned)?;
            write_new(&args.out, &receipt)?;
            println!(
                "{}",
                serde_json::to_string(
                    &json!({"status":RECEIPT_STATUS,"receipt":args.out,"seal_sha256":seal,"gpu_dispatches":P4A_DISPATCHES,"fallback":false})
                )?
            );
        }
        Ok(())
    }

    fn qat(
        ctx: &MetalContext,
        input: &metal::Buffer,
        out: &metal::Buffer,
        scales: &metal::Buffer,
        cols: u32,
    ) -> ProbeResult<MetalDispatchTiming> {
        let t = ctx.dispatch_threads_timed(
            QAT_KERNEL,
            (cols / ACT_QUANT_BLOCK as u32, 1, 1),
            (32, 1, 1),
            |e| {
                e.set_buffer(0, Some(input), 0);
                e.set_buffer(1, Some(out), 0);
                e.set_buffer(2, Some(scales), 0);
                set_u32(e, 3, &cols);
            },
        )?;
        checked(&t, "source QAT")?;
        Ok(t)
    }
    fn fp8(
        ctx: &MetalContext,
        w: &metal::Buffer,
        s: &metal::Buffer,
        a: &metal::Buffer,
        as_: &metal::Buffer,
        out: &metal::Buffer,
        rows: u32,
        cols: u32,
        scale_cols: u32,
    ) -> ProbeResult<MetalDispatchTiming> {
        let t = ctx.dispatch_threads_timed(FP8_KERNEL, (rows, 1, 1), (256, 1, 1), |e| {
            e.set_buffer(0, Some(w), 0);
            e.set_buffer(1, Some(s), 0);
            e.set_buffer(2, Some(a), 0);
            e.set_buffer(3, Some(as_), 0);
            e.set_buffer(4, Some(out), 0);
            set_u32(e, 5, &rows);
            set_u32(e, 6, &cols);
            set_u32(e, 7, &scale_cols);
        })?;
        checked(&t, "FP8 projection")?;
        Ok(t)
    }
    fn cast(
        ctx: &MetalContext,
        input: &metal::Buffer,
        out: &metal::Buffer,
        count: u32,
    ) -> ProbeResult<MetalDispatchTiming> {
        let t = ctx.dispatch_threads_timed(CAST_KERNEL, (count, 1, 1), (256, 1, 1), |e| {
            e.set_buffer(0, Some(input), 0);
            e.set_buffer(1, Some(out), 0);
            set_u32(e, 2, &count);
        })?;
        checked(&t, "FP32-to-BF16 cast")?;
        Ok(t)
    }

    fn parse_args() -> ProbeResult<Args> {
        let mut artifact = None;
        let mut cpu_attention_oracle = None;
        let mut p3a_precursor = None;
        let mut out = None;
        let mut topology_out = None;
        let mut topology_trials = P4A_TOPOLOGY_TRIALS_DEFAULT;
        let mut topology_warmups = P4A_TOPOLOGY_WARMUPS_DEFAULT;
        let mut it = std::env::args_os().skip(1);
        while let Some(flag) = it.next() {
            match flag.to_string_lossy().as_ref() {
                "--artifact" => artifact = it.next().map(PathBuf::from),
                "--cpu-attention-oracle" => cpu_attention_oracle = it.next().map(PathBuf::from),
                "--p3a-precursor" => p3a_precursor = it.next().map(PathBuf::from),
                "--out" => out = it.next().map(PathBuf::from),
                "--topology-out" => topology_out = it.next().map(PathBuf::from),
                "--topology-trials" => {
                    topology_trials = it
                        .next()
                        .ok_or_else(|| failure("--topology-trials needs a value"))?
                        .to_string_lossy()
                        .parse()
                        .map_err(|_| failure("--topology-trials must be an integer"))?;
                }
                "--topology-warmups" => {
                    topology_warmups = it
                        .next()
                        .ok_or_else(|| failure("--topology-warmups needs a value"))?
                        .to_string_lossy()
                        .parse()
                        .map_err(|_| failure("--topology-warmups must be an integer"))?;
                }
                other => return Err(failure(format!("unknown argument {other}"))),
            }
        }
        if topology_out.is_some() && topology_trials < P4A_TOPOLOGY_TRIALS_MIN {
            return Err(failure(format!(
                "P4A topology sweep requires at least {P4A_TOPOLOGY_TRIALS_MIN} clean trials"
            )));
        }
        Ok(Args {
            artifact: artifact.ok_or_else(|| failure("--artifact is required"))?,
            cpu_attention_oracle: cpu_attention_oracle
                .ok_or_else(|| failure("--cpu-attention-oracle is required"))?,
            p3a_precursor: p3a_precursor.ok_or_else(|| failure("--p3a-precursor is required"))?,
            out: out.ok_or_else(|| failure("--out is required"))?,
            topology_out,
            topology_trials,
            topology_warmups,
        })
    }
    fn full(
        reader: &DeepSeekV4FullStreamReader,
        name: &str,
    ) -> ProbeResult<(DeepSeekV4TensorMetadata, Vec<u8>)> {
        let m = reader.tensor_metadata(name)?.clone();
        let b = reader.read_verified_full(name, m.bytes as usize)?;
        Ok((m, b))
    }
    fn fp8_pair(
        reader: &DeepSeekV4FullStreamReader,
        w: &str,
        s: &str,
        rows: usize,
        cols: usize,
    ) -> ProbeResult<(
        DeepSeekV4TensorMetadata,
        DeepSeekV4TensorMetadata,
        Vec<u8>,
        Vec<u8>,
    )> {
        let p = reader.native_scale_pair(w)?;
        if p.kind != NativeScalePairKind::Fp8E4M3fn
            || p.scale.name != s
            || p.weight.shape.as_slice() != [rows as u64, cols as u64]
            || p.scale.shape.as_slice()
                != [
                    (rows / ACT_QUANT_BLOCK) as u64,
                    (cols / ACT_QUANT_BLOCK) as u64,
                ]
        {
            return Err(failure(format!("{w} FP8 pair geometry changed")));
        }
        let wb = reader.read_verified_full(w, p.weight.bytes as usize)?;
        let sb = reader.read_verified_full(s, p.scale.bytes as usize)?;
        Ok((p.weight.clone(), p.scale.clone(), wb, sb))
    }

    fn validate_cpu_receipt(
        path: &Path,
        reader: &DeepSeekV4FullStreamReader,
        cpu: &Layer0AttentionCpuOracleResult,
    ) -> ProbeResult<ReceiptBinding> {
        if path.file_name().and_then(|x| x.to_str()) != Some(CPU_ORACLE_BASENAME) {
            return Err(failure("wrong CPU attention oracle basename"));
        }
        let path = fs::canonicalize(path)?;
        let raw = fs::read(&path)?;
        let v: Value = serde_json::from_slice(&raw)?;
        seal_ok(&v)?;
        if text(&v, &["schema"])? != CPU_ORACLE_SCHEMA
            || text(&v, &["status"])? != CPU_ORACLE_STATUS
            || text(&v, &["artifact", "manifest_seal_sha256"])? != reader.manifest_seal_sha256()
        {
            return Err(failure("CPU oracle source binding differs"));
        }
        if text(
            &v,
            &[
                "intermediate_receipts",
                "o_path",
                "wo_b_fp8_linear",
                "output_bf16",
                "sha256_bf16_le",
            ],
        )? != sha256(&u16bytes(&cpu.wo_b.output.bf16_bits))
            || text(
                &v,
                &[
                    "intermediate_receipts",
                    "mhc_attn_post",
                    "output",
                    "sha256_bf16_le",
                ],
            )? != sha256(&u16bytes(&cpu.hc_attn_post_bf16_bits))
        {
            return Err(failure(
                "direct CPU oracle differs from sealed attention receipt",
            ));
        }
        Ok(ReceiptBinding {
            path,
            file_sha256: sha256(&raw),
            seal_sha256: text(&v, &["seal_sha256"])?.to_owned(),
        })
    }
    fn validate_p3a_receipt(
        path: &Path,
        reader: &DeepSeekV4FullStreamReader,
    ) -> ProbeResult<ReceiptBinding> {
        if path.file_name().and_then(|x| x.to_str()) != Some(P3A_BASENAME) {
            return Err(failure("wrong P3A predecessor basename"));
        }
        let path = fs::canonicalize(path)?;
        let raw = fs::read(&path)?;
        let v: Value = serde_json::from_slice(&raw)?;
        seal_ok(&v)?;
        if text(&v, &["schema"])? != P3A_SCHEMA
            || text(&v, &["status"])? != P3A_STATUS
            || text(&v, &["artifact", "manifest_seal_sha256"])? != reader.manifest_seal_sha256()
            || v.pointer("/metal/fallback").and_then(Value::as_bool) != Some(false)
        {
            return Err(failure(
                "P3A predecessor is not the admitted passing device precursor",
            ));
        }
        Ok(ReceiptBinding {
            path,
            file_sha256: sha256(&raw),
            seal_sha256: text(&v, &["seal_sha256"])?.to_owned(),
        })
    }

    /// Topology measurements must be anchored to an already sealed P4A
    /// authority run.  This prevents a lower-command candidate from replacing
    /// the per-stage receipt that established correctness.
    fn validate_p4a_authority_receipt(
        path: &Path,
        reader: &DeepSeekV4FullStreamReader,
    ) -> ProbeResult<ReceiptBinding> {
        if path.file_name().and_then(|x| x.to_str()) != Some(P4A_BASENAME) {
            return Err(failure("wrong P4A authority receipt basename"));
        }
        let path = fs::canonicalize(path)?;
        let raw = fs::read(&path)?;
        let v: Value = serde_json::from_slice(&raw)?;
        seal_ok(&v)?;
        if text(&v, &["schema"])? != RECEIPT_SCHEMA
            || text(&v, &["status"])? != RECEIPT_STATUS
            || text(&v, &["artifact", "manifest_seal_sha256"])? != reader.manifest_seal_sha256()
            || v.pointer("/metal/fallback").and_then(Value::as_bool) != Some(false)
            || v.pointer("/numeric_parity_v2_1/all_host_and_device_scores_pass")
                .and_then(Value::as_bool)
                != Some(true)
        {
            return Err(failure(
                "P4A authority receipt is not an admitted complete-parity Metal run",
            ));
        }
        Ok(ReceiptBinding {
            path,
            file_sha256: sha256(&raw),
            seal_sha256: text(&v, &["seal_sha256"])?.to_owned(),
        })
    }

    fn hc_pre_f64(embed: &[u8], fn_b: &[u8], scale_b: &[u8], base_b: &[u8]) -> ProbeResult<HcF64> {
        let x = bf16_bytes_f64(embed)?;
        let f = f32_bytes_f64(fn_b)?;
        let s = f32_bytes_f64(scale_b)?;
        let b = f32_bytes_f64(base_b)?;
        if x.len() != HIDDEN_SIZE
            || f.len() != HC_MIX_WIDTH * HC_FLAT_WIDTH
            || s.len() != 3
            || b.len() != HC_MIX_WIDTH
        {
            return Err(failure("FP64 mHC geometry invalid"));
        }
        let mut ss = 0.;
        for _ in 0..HC_MULT {
            for &v in &x {
                ss += v * v;
            }
        }
        let r = 1. / (ss / HC_FLAT_WIDTH as f64 + RMS_NORM_EPS as f64).sqrt();
        let mut mix = vec![0.; HC_MIX_WIDTH];
        for row in 0..HC_MIX_WIDTH {
            let mut a = 0.;
            for lane in 0..HC_MULT {
                let off = row * HC_FLAT_WIDTH + lane * HIDDEN_SIZE;
                for col in 0..HIDDEN_SIZE {
                    a += f[off + col] * x[col];
                }
            }
            mix[row] = a * r;
        }
        let mut post = vec![0.; HC_MULT];
        let mut comb = vec![0.; HC_MULT * HC_MULT];
        for i in 0..HC_MULT {
            post[i] = 2. / (1. + (-(mix[i + HC_MULT] * s[1] + b[i + HC_MULT])).exp());
        }
        for row in 0..HC_MULT {
            for col in 0..HC_MULT {
                let i = row * HC_MULT + col;
                comb[i] = mix[i + 2 * HC_MULT] * s[2] + b[i + 2 * HC_MULT];
            }
        }
        for row in 0..HC_MULT {
            let st = row * HC_MULT;
            let m = comb[st..st + HC_MULT]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.;
            for c in 0..HC_MULT {
                comb[st + c] = (comb[st + c] - m).exp();
                sum += comb[st + c];
            }
            for c in 0..HC_MULT {
                comb[st + c] = comb[st + c] / sum + HC_EPS as f64;
            }
        }
        norm_cols(&mut comb)?;
        for _ in 1..HC_SINKHORN_ITERS {
            norm_rows(&mut comb)?;
            norm_cols(&mut comb)?;
        }
        Ok(HcF64 { post, comb })
    }
    fn norm_rows(x: &mut [f64]) -> ProbeResult<()> {
        for r in 0..HC_MULT {
            let st = r * HC_MULT;
            let s: f64 = x[st..st + HC_MULT].iter().sum();
            if s <= 0. || !s.is_finite() {
                return Err(failure("FP64 Sinkhorn row"));
            }
            for v in &mut x[st..st + HC_MULT] {
                *v /= s + HC_EPS as f64;
            }
        }
        Ok(())
    }
    fn norm_cols(x: &mut [f64]) -> ProbeResult<()> {
        for c in 0..HC_MULT {
            let mut s = 0.;
            for r in 0..HC_MULT {
                s += x[r * HC_MULT + c];
            }
            if s <= 0. || !s.is_finite() {
                return Err(failure("FP64 Sinkhorn column"));
            }
            for r in 0..HC_MULT {
                x[r * HC_MULT + c] /= s + HC_EPS as f64;
            }
        }
        Ok(())
    }
    fn fp8_f64(
        w: &[u8],
        ws: &[u8],
        a: &[u8],
        as_: &[u8],
        rows: usize,
        cols: usize,
    ) -> ProbeResult<Vec<f64>> {
        if rows % 128 != 0
            || cols % 128 != 0
            || w.len() != rows * cols
            || ws.len() != rows / 128 * cols / 128
            || a.len() != cols
            || as_.len() != cols / 128
        {
            return Err(failure("FP64 FP8 geometry"));
        }
        let sc = cols / 128;
        let mut o = vec![0.; rows];
        for r in 0..rows {
            let mut acc = 0.;
            for bl in 0..sc {
                let mut ba = 0.;
                for c in bl * 128..(bl + 1) * 128 {
                    ba += e4(a[c])? * e4(w[r * cols + c])?;
                }
                acc += ba * e8(as_[bl])? * e8(ws[(r / 128) * sc + bl])?;
            }
            o[r] = acc;
        }
        Ok(o)
    }
    fn bf16_store_ref(values: Vec<f64>) -> Vec<f64> {
        values.into_iter().map(bf16_round).collect()
    }
    fn kv_qat_f64(input: &[u16], a: &[u8], s: &[u8]) -> ProbeResult<Vec<f64>> {
        if input.len() != HEAD_DIM
            || a.len() != NON_ROPE_HEAD_DIM
            || s.len() != NON_ROPE_HEAD_DIM / KV_QAT_BLOCK
        {
            return Err(failure("FP64 KV QAT geometry"));
        }
        let mut o = bf16_bits_f64(input);
        for i in 0..NON_ROPE_HEAD_DIM {
            o[i] = bf16_round(e4(a[i])? * e8(s[i / KV_QAT_BLOCK])?);
        }
        Ok(o)
    }
    // The sparse operator's declared output representation is BF16.  Retain
    // the raw-payload f64 score/denominator reference above, then apply the
    // same explicit source store boundary before discrete activation parity.
    // Without this materialization, values that collapse to the same BF16
    // payload can spuriously reorder a top-k diagnostic.
    fn sparse_f64(q: &[u16], kv: &[u16], sink: &[f64]) -> ProbeResult<SparseF64> {
        if q.len() != WQ_B_ROWS || kv.len() != HEAD_DIM || sink.len() != NUM_HEADS {
            return Err(failure("FP64 sparse geometry"));
        }
        let q = bf16_bits_f64(q);
        let kv = bf16_bits_f64(kv);
        let scale = 1. / (HEAD_DIM as f64).sqrt();
        let mut scores = Vec::with_capacity(NUM_HEADS);
        let mut denominators = Vec::with_capacity(NUM_HEADS);
        let mut output = Vec::with_capacity(WQ_B_ROWS);
        for h in 0..NUM_HEADS {
            let mut dot = 0.;
            for d in 0..HEAD_DIM {
                dot += q[h * HEAD_DIM + d] * kv[d];
            }
            let score = dot * scale;
            let den = 1. + (sink[h] - score).exp();
            scores.push(score);
            denominators.push(den);
            for &v in &kv {
                output.push(bf16_round(v / den));
            }
        }
        Ok(SparseF64 {
            scores,
            denominators,
            output,
        })
    }
    fn wo_a_f64(attn: &[u16], w: &[u8], s: &[u8]) -> ProbeResult<Vec<f64>> {
        if attn.len() != WQ_B_ROWS
            || w.len() != WO_A_ROWS * WO_A_COLS
            || s.len() != WO_A_ROWS / 128 * WO_A_COLS / 128
        {
            return Err(failure("FP64 WO-A geometry"));
        }
        let a = bf16_bits_f64(attn);
        let sc = WO_A_COLS / 128;
        let mut out = vec![0.; WO_A_ROWS];
        for row in 0..WO_A_ROWS {
            let g = row / O_LORA_RANK;
            let mut acc = 0.;
            for c in 0..WO_A_COLS {
                let converted =
                    bf16_round(e4(w[row * WO_A_COLS + c])? * e8(s[(row / 128) * sc + c / 128])?);
                acc += a[g * WO_A_COLS + c] * converted;
            }
            out[row] = bf16_round(acc);
        }
        Ok(out)
    }
    fn hc_post_f64(attn: &[u16], embed: &[u8], hc: &HcF64) -> ProbeResult<Vec<f64>> {
        if attn.len() != HIDDEN_SIZE
            || hc.post.len() != HC_MULT
            || hc.comb.len() != HC_MULT * HC_MULT
        {
            return Err(failure("FP64 mHC post geometry"));
        }
        let a = bf16_bits_f64(attn);
        let x = bf16_bytes_f64(embed)?;
        let mut o = vec![0.; HC_MULT * HIDDEN_SIZE];
        for k in 0..HC_MULT {
            for f in 0..HIDDEN_SIZE {
                let mut v = hc.post[k] * a[f];
                for j in 0..HC_MULT {
                    v += hc.comb[j * HC_MULT + k] * x[f];
                }
                o[k * HIDDEN_SIZE + f] = bf16_round(v);
            }
        }
        Ok(o)
    }
    fn e4(b: u8) -> ProbeResult<f64> {
        let e = (b >> 3) & 15;
        let m = b & 7;
        if e == 15 && m == 7 {
            return Err(failure("E4M3 NaN"));
        }
        let v = if e == 0 {
            m as f64 * 2f64.powi(-9)
        } else {
            (1. + m as f64 / 8.) * 2f64.powi(e as i32 - 7)
        };
        Ok(if b & 128 != 0 { -v } else { v })
    }
    fn e8(b: u8) -> ProbeResult<f64> {
        if b == 255 {
            return Err(failure("E8M0 NaN"));
        }
        Ok(2f64.powi(b as i32 - 127))
    }
    fn bf16_round(v: f64) -> f64 {
        let f = v as f32;
        let bits = f.to_bits();
        let out = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
        f32::from_bits(out << 16) as f64
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
    fn score(name: &str, h: &[f32], d: &[f32], r: &[f64], b: &Bounds) -> ProbeResult<PairedScore> {
        if h.len() != d.len() || h.len() != r.len() {
            return Err(failure(format!("V2.1 {name} geometry")));
        }
        let x = score_pair(h, d, r, b);
        if !x.pass {
            return Err(failure(format!(
                "V2.1 {name} failed: host={:?}; device={:?}",
                x.host.failures, x.device.failures
            )));
        }
        Ok(x)
    }

    fn checked(t: &MetalDispatchTiming, stage: &str) -> ProbeResult<()> {
        if t.command_buffers != 1
            || t.compute_encoders != 1
            || t.compute_dispatches != 1
            || t.gpu_duration_us.is_none()
            || t.gpu_start_ns.is_none()
            || t.gpu_end_ns.is_none()
        {
            return Err(failure(format!("{stage} lacks completed GPU timestamp")));
        }
        Ok(())
    }

    fn percentile_u64(values: &[u64], percentile: u64) -> ProbeResult<u64> {
        if values.is_empty() || percentile > 100 {
            return Err(failure("invalid timing percentile request"));
        }
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        let rank = ((percentile as usize * (sorted.len() - 1) + 99) / 100).min(sorted.len() - 1);
        Ok(sorted[rank])
    }

    fn timing_summary(values: &[u64]) -> ProbeResult<Value> {
        if values.is_empty() {
            return Err(failure("empty timing series"));
        }
        let min = *values.iter().min().expect("checked nonempty");
        let max = *values.iter().max().expect("checked nonempty");
        let mean = values.iter().map(|&value| value as f64).sum::<f64>() / values.len() as f64;
        Ok(json!({
            "samples_us":values,
            "min_us":min,
            "p50_us":percentile_u64(values,50)?,
            "p95_us":percentile_u64(values,95)?,
            "p99_us":percentile_u64(values,99)?,
            "max_us":max,
            "mean_us":mean,
        }))
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
    fn exact8(name: &str, a: &[u8], b: &[u8]) -> ProbeResult<()> {
        if a != b {
            return Err(failure(format!("{name} byte mismatch")));
        }
        Ok(())
    }
    fn exact16(name: &str, a: &[u16], b: &[u16]) -> ProbeResult<()> {
        exact8(name, &u16bytes(a), &u16bytes(b))
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
    fn set_u32(e: &metal::ComputeCommandEncoderRef, i: u64, v: &u32) {
        e.set_bytes(i, 4, v as *const u32 as *const _)
    }
    fn set_f32(e: &metal::ComputeCommandEncoderRef, i: u64, v: &f32) {
        e.set_bytes(i, 4, v as *const f32 as *const _)
    }
    fn bytesread(b: &metal::Buffer, n: usize) -> ProbeResult<Vec<u8>> {
        if b.length() < n as u64 {
            return Err(failure("GPU byte read overflow"));
        }
        Ok(unsafe { std::slice::from_raw_parts(b.contents() as *const u8, n).to_vec() })
    }
    fn u16read(b: &metal::Buffer, n: usize) -> ProbeResult<Vec<u16>> {
        let x = bytesread(b, n * 2)?;
        Ok(x.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    }
    fn f32read(b: &metal::Buffer, n: usize) -> ProbeResult<Vec<f32>> {
        if b.length() < n as u64 * 4 {
            return Err(failure("GPU f32 read overflow"));
        }
        let x = unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n).to_vec() };
        if x.iter().any(|v| !v.is_finite()) {
            return Err(failure("GPU non-finite"));
        }
        Ok(x)
    }
    fn bf16_bits_f64(x: &[u16]) -> Vec<f64> {
        x.iter()
            .map(|&v| f32::from_bits((v as u32) << 16) as f64)
            .collect()
    }
    fn bf16_bytes_f64(x: &[u8]) -> ProbeResult<Vec<f64>> {
        if x.len() % 2 != 0 {
            return Err(failure("BF16 bytes"));
        }
        Ok(x.chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64)
            .collect())
    }
    fn f32_bytes_f64(x: &[u8]) -> ProbeResult<Vec<f64>> {
        if x.len() % 4 != 0 {
            return Err(failure("F32 bytes"));
        }
        Ok(x.chunks_exact(4)
            .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])) as f64)
            .collect())
    }
    fn bf16f32(x: &[u16]) -> Vec<f32> {
        x.iter()
            .map(|&v| f32::from_bits((v as u32) << 16))
            .collect()
    }
    fn tensor_json(t: &DeepSeekV4TensorMetadata) -> Value {
        json!({"name":t.name,"dtype":t.dtype,"shape":t.shape,"bytes":t.bytes,"segments":t.segments.iter().map(|s|json!({"bytes":s.bytes,"chunk_relpath":s.chunk_relpath,"sha256":s.sha256,"source_file_start":s.source_file_start,"source_file_end":s.source_file_end,"tensor_start":s.tensor_start,"tensor_end":s.tensor_end,"row_start":s.row_start,"row_count":s.row_count})).collect::<Vec<_>>()})
    }
    fn u16bytes(x: &[u16]) -> Vec<u8> {
        let mut o = Vec::with_capacity(x.len() * 2);
        for v in x {
            o.extend_from_slice(&v.to_le_bytes())
        }
        o
    }
    fn sha256(x: &[u8]) -> String {
        format!("{:x}", Sha256::digest(x))
    }
    fn sha256_join(xs: &[&str]) -> String {
        let mut d = Sha256::new();
        for x in xs {
            d.update(x.as_bytes());
            d.update([0])
        }
        format!("{:x}", d.finalize())
    }
    fn text<'a>(v: &'a Value, p: &[&str]) -> ProbeResult<&'a str> {
        let mut x = v;
        for k in p {
            x = x
                .get(*k)
                .ok_or_else(|| failure(format!("missing {}", p.join("."))))?;
        }
        x.as_str()
            .ok_or_else(|| failure(format!("not text {}", p.join("."))))
    }
    fn seal_ok(v: &Value) -> ProbeResult<()> {
        let s = text(v, &["seal_sha256"])?;
        let mut x = v.clone();
        x.as_object_mut()
            .ok_or_else(|| failure("receipt object"))?
            .remove("seal_sha256");
        if sha256(&canonical(&x)) != s {
            return Err(failure("predecessor seal mismatch"));
        }
        Ok(())
    }
    fn canonical(v: &Value) -> Vec<u8> {
        let mut out = Vec::new();
        canon(&mut out, v);
        out
    }

    fn canon(out: &mut Vec<u8>, value: &Value) {
        match value {
            Value::Null => out.extend_from_slice(b"null"),
            Value::Bool(true) => out.extend_from_slice(b"true"),
            Value::Bool(false) => out.extend_from_slice(b"false"),
            Value::Number(number) => out.extend_from_slice(number.to_string().as_bytes()),
            Value::String(string) => out.extend_from_slice(
                serde_json::to_string(string)
                    .expect("JSON string")
                    .as_bytes(),
            ),
            Value::Array(values) => {
                out.push(b'[');
                for (index, child) in values.iter().enumerate() {
                    if index != 0 {
                        out.push(b',');
                    }
                    canon(out, child);
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
                    out.extend_from_slice(serde_json::to_string(key).expect("JSON key").as_bytes());
                    out.push(b':');
                    canon(out, &object[key]);
                }
                out.push(b'}');
            }
        }
    }
    fn decimal_strings(v: Value) -> Value {
        match v {
            Value::Number(n) if n.is_i64() || n.is_u64() => Value::Number(n),
            Value::Number(n) => Value::String(n.to_string()),
            Value::Array(a) => Value::Array(a.into_iter().map(decimal_strings).collect()),
            Value::Object(m) => Value::Object(
                m.into_iter()
                    .map(|(k, v)| (k, decimal_strings(v)))
                    .collect(),
            ),
            x => x,
        }
    }
    fn seal(mut v: Value) -> ProbeResult<(Value, String)> {
        if !v.is_object() || v.get("seal_sha256").is_some() {
            return Err(failure("unsealed receipt required"));
        }
        let s = sha256(&canonical(&v));
        v.as_object_mut()
            .unwrap()
            .insert("seal_sha256".into(), Value::String(s.clone()));
        Ok((v, s))
    }
    fn write_new(path: &Path, v: &Value) -> ProbeResult<()> {
        if path.exists() {
            return Err(failure(format!("refusing overwrite {}", path.display())));
        }
        let parent = path.parent().ok_or_else(|| failure("out parent"))?;
        fs::create_dir_all(parent)?;
        let name = path
            .file_name()
            .and_then(|x| x.to_str())
            .ok_or_else(|| failure("out UTF8"))?;
        let tmp = parent.join(format!(".{name}.{}.p4a.tmp", std::process::id()));
        let mut f = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        if let Err(e) = f
            .write_all(&serde_json::to_vec_pretty(v)?)
            .and_then(|_| f.write_all(b"\n"))
            .and_then(|_| f.sync_all())
        {
            let _ = fs::remove_file(&tmp);
            return Err(Box::new(e));
        }
        drop(f);
        if let Err(e) = fs::hard_link(&tmp, path) {
            let _ = fs::remove_file(&tmp);
            return Err(failure(format!("link receipt: {e}")));
        }
        fs::remove_file(&tmp)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

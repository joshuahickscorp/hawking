// Batched prefill for Qwen38HybridDecodeSession. Included into `mod device`
// so it can see private workspace / weight types. Decode `step` is unchanged.

const QWEN38_PREFILL_GEMM_TG: u32 = 128;
const QWEN38_PREFILL_GEMM_ROWS_PER_TG: u32 = 32;
const QWEN38_PREFILL_GEMM_SHMEM_BYTES: u64 = 5120 * 4;
const QWEN38_PREFILL_GQA_ATTN_SHMEM_BYTES: u64 = 16 * 8 * 4;
const QWEN38_PREFILL_MAX_BATCH: usize = QWEN38_PREFILL_CHUNK;

#[derive(Clone, Debug, Default)]
struct Qwen38PrefillTiming {
    gpu_ns: Option<u64>,
    wait_ns: u64,
    encode_ns: u64,
    submit_ns: u64,
    dispatches: u64,
    active_weight_bytes: u64,
}

impl Qwen38PrefillTiming {
    fn accumulate(&mut self, timing: &CommandBufferTiming, active_weight_bytes: u64) {
        self.gpu_ns = match (self.gpu_ns, timing.gpu_ns) {
            (Some(a), Some(b)) => a.checked_add(b),
            (None, Some(b)) if self.dispatches == 0 => Some(b),
            (None, Some(_)) => None,
            (other, _) => other,
        };
        self.wait_ns = self.wait_ns.saturating_add(timing.wait_ns);
        self.encode_ns = self.encode_ns.saturating_add(timing.encode_ns);
        self.submit_ns = self.submit_ns.saturating_add(timing.submit_ns);
        self.dispatches = self.dispatches.saturating_add(timing.dispatches);
        self.active_weight_bytes = self
            .active_weight_bytes
            .saturating_add(active_weight_bytes);
    }
}

impl Qwen38PrefillWorkspace {
    fn allocate(ctx: &MetalContext, cap: usize) -> Result<Self> {
        if cap == 0 {
            return Err(Error::Model("qwen38 prefill chunk cap must be positive".into()));
        }
        let layout = Qwen38DeltaNetLayout::source_exact();
        let f32b = |n: usize| {
            n.checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| Error::Model("qwen38 prefill workspace overflow".into()))
        };
        let hidden = f32b(cap.checked_mul(QWEN38_HIDDEN).ok_or_else(|| {
            Error::Model("qwen38 prefill hidden overflow".into())
        })?)?;
        let mid = f32b(cap.checked_mul(QWEN38_INTERMEDIATE).ok_or_else(|| {
            Error::Model("qwen38 prefill mlp overflow".into())
        })?)?;
        let qkvz = f32b(cap.checked_mul(layout.qkvz_rows()).ok_or_else(|| {
            Error::Model("qwen38 prefill qkvz overflow".into())
        })?)?;
        let ba = f32b(cap.checked_mul(layout.ba_rows()).ok_or_else(|| {
            Error::Model("qwen38 prefill ba overflow".into())
        })?)?;
        let value = f32b(cap.checked_mul(layout.value_elements()).ok_or_else(|| {
            Error::Model("qwen38 prefill value overflow".into())
        })?)?;
        let q_proj = f32b(
            cap.checked_mul(QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM * 2)
                .ok_or_else(|| Error::Model("qwen38 prefill q_proj overflow".into()))?,
        )?;
        let kv = f32b(
            cap.checked_mul(QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM)
                .ok_or_else(|| Error::Model("qwen38 prefill kv overflow".into()))?,
        )?;
        let query = f32b(
            cap.checked_mul(QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM)
                .ok_or_else(|| Error::Model("qwen38 prefill query overflow".into()))?,
        )?;
        Ok(Self {
            cap,
            tokens: ctx.new_buffer_checked(cap * std::mem::size_of::<u32>())?,
            hidden: ctx.new_buffer_checked(hidden)?,
            normalized: ctx.new_buffer_checked(hidden)?,
            mixer: ctx.new_buffer_checked(hidden)?,
            first_residual: ctx.new_buffer_checked(hidden)?,
            down: ctx.new_buffer_checked(hidden)?,
            gate: ctx.new_buffer_checked(mid)?,
            up: ctx.new_buffer_checked(mid)?,
            act: ctx.new_buffer_checked(mid)?,
            qkvz: ctx.new_buffer_checked(qkvz)?,
            ba: ctx.new_buffer_checked(ba)?,
            repeated_q: ctx.new_buffer_checked(value)?,
            repeated_k: ctx.new_buffer_checked(value)?,
            conv_v: ctx.new_buffer_checked(value)?,
            z: ctx.new_buffer_checked(value)?,
            rec_out: ctx.new_buffer_checked(value)?,
            gated: ctx.new_buffer_checked(value)?,
            q_proj: ctx.new_buffer_checked(q_proj)?,
            k_proj: ctx.new_buffer_checked(kv)?,
            v_proj: ctx.new_buffer_checked(kv)?,
            query: ctx.new_buffer_checked(query)?,
            attn: ctx.new_buffer_checked(query)?,
            gated_attn: ctx.new_buffer_checked(query)?,
        })
    }
}

impl Qwen38HybridDecodeSession {
    fn prefill_chunk_cap() -> usize {
        qwen38_prefill_chunk_tokens()
    }

    fn ensure_prefill_workspace(&mut self) -> Result<()> {
        let cap = Self::prefill_chunk_cap().min(self.max_seq_len);
        if let Some(existing) = &self.prefill {
            if existing.cap >= cap {
                return Ok(());
            }
        }
        self.prefill = Some(Qwen38PrefillWorkspace::allocate(&self.context, cap)?);
        Ok(())
    }

    fn gemm_grid(rows: u32) -> (u32, u32, u32) {
        (
            rows.div_ceil(QWEN38_PREFILL_GEMM_ROWS_PER_TG)
                .saturating_mul(QWEN38_PREFILL_GEMM_TG)
                .max(QWEN38_PREFILL_GEMM_TG),
            1,
            1,
        )
    }

    fn encode_prefill_gemm(
        &self,
        tcb: &mut TokenCommandBuffer<'_>,
        name: &str,
        input: &PinnedBuffer,
        output: &PinnedBuffer,
        n_tokens: u32,
    ) -> Result<()> {
        if n_tokens == 0 {
            return Ok(());
        }
        if n_tokens > QWEN38_PREFILL_MAX_BATCH as u32 {
            return Err(Error::Model(format!(
                "qwen38 prefill GEMM batch {n_tokens} exceeds {}",
                QWEN38_PREFILL_MAX_BATCH
            )));
        }
        if let Some(MixedGpuWeight::Affine(body)) = self.weights.mixed.get(name) {
            let biases = body.biases.as_ref().ok_or_else(|| {
                mixed_error(format!("{name} prefill GEMM needs affine scale+bias, not q2f"))
            })?;
            if body.group_size != 64 {
                return Err(mixed_error(format!(
                    "{name} prefill GEMM requires group 64, got {}",
                    body.group_size
                )));
            }
            self.record_affine_weight(body);
            let rows = body.rows;
            let cols = body.cols;
            return tcb.dispatch_threads(
                "qwen38_prefill_affine_q2_g64_gemm_mma_n64",
                Self::gemm_grid(rows),
                (QWEN38_PREFILL_GEMM_TG, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&body.codes), 0);
                    enc.set_buffer(1, Some(&body.scales), 0);
                    enc.set_buffer(2, Some(biases), 0);
                    enc.set_buffer(3, Some(input), 0);
                    enc.set_buffer(4, Some(output), 0);
                    set_u32(enc, 5, rows);
                    set_u32(enc, 6, cols);
                    set_u32(enc, 7, n_tokens);
                    enc.set_threadgroup_memory_length(0, QWEN38_PREFILL_GEMM_SHMEM_BYTES);
                },
            );
        }
        if let Some(weight) = self.weights.q4.get(name) {
            if weight.group_size != 64 {
                return Err(Error::Model(format!(
                    "{name} prefill GEMM requires Q4 group 64, got {}",
                    weight.group_size
                )));
            }
            self.record_q4_weight(weight);
            let rows = weight.rows as u32;
            let cols = weight.cols as u32;
            return tcb.dispatch_threads(
                "qwen38_prefill_q4_g64_gemm_mma_n64",
                Self::gemm_grid(rows),
                (QWEN38_PREFILL_GEMM_TG, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&weight.codes), 0);
                    enc.set_buffer(1, Some(&weight.scales), 0);
                    enc.set_buffer(2, Some(input), 0);
                    enc.set_buffer(3, Some(output), 0);
                    set_u32(enc, 4, rows);
                    set_u32(enc, 5, cols);
                    set_u32(enc, 6, n_tokens);
                    enc.set_threadgroup_memory_length(0, QWEN38_PREFILL_GEMM_SHMEM_BYTES);
                },
            );
        }
        Err(mixed_error(format!(
            "prefill GEMM missing affine/Q4 {name}; refusing a silent matvec fallback"
        )))
    }

    fn encode_prefill_rmsnorm(
        &self,
        tcb: &mut TokenCommandBuffer<'_>,
        input: &PinnedBuffer,
        weight_name: &str,
        output: &PinnedBuffer,
        n_tokens: u32,
    ) -> Result<()> {
        let weight = self.f32(weight_name)?;
        let tg = if self.rmsnorm_tg > 0 { self.rmsnorm_tg } else { 256 };
        let hidden = QWEN38_HIDDEN as u32;
        tcb.dispatch_threads(
            "qwen38_prefill_rmsnorm_f32",
            (tg * n_tokens, 1, 1),
            (tg, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(input), 0);
                enc.set_buffer(1, Some(weight), 0);
                enc.set_buffer(2, Some(output), 0);
                set_u32(enc, 3, hidden);
                enc.set_bytes(4, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                set_u32(enc, 5, n_tokens);
                enc.set_threadgroup_memory_length(0, u64::from(tg) * 4);
            },
        )
    }

    fn encode_prefill_add_residual_rmsnorm(
        &self,
        tcb: &mut TokenCommandBuffer<'_>,
        residual_in: &PinnedBuffer,
        delta: &PinnedBuffer,
        residual_out: &PinnedBuffer,
        weight_name: &str,
        x_norm: &PinnedBuffer,
        n_tokens: u32,
    ) -> Result<()> {
        let weight = self.f32(weight_name)?;
        let tg = if self.rmsnorm_tg > 0 { self.rmsnorm_tg } else { 256 };
        let hidden = QWEN38_HIDDEN as u32;
        tcb.dispatch_threads(
            "qwen38_prefill_add_residual_rmsnorm_f32",
            (tg * n_tokens, 1, 1),
            (tg, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(residual_in), 0);
                enc.set_buffer(1, Some(delta), 0);
                enc.set_buffer(2, Some(residual_out), 0);
                enc.set_buffer(3, Some(weight), 0);
                enc.set_buffer(4, Some(x_norm), 0);
                set_u32(enc, 5, hidden);
                enc.set_bytes(6, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                set_u32(enc, 7, n_tokens);
                enc.set_threadgroup_memory_length(0, u64::from(tg) * 4);
            },
        )
    }

    fn encode_prefill_add(
        &self,
        tcb: &mut TokenCommandBuffer<'_>,
        residual: &PinnedBuffer,
        delta: &PinnedBuffer,
        output: &PinnedBuffer,
        n_f32: u32,
    ) -> Result<()> {
        tcb.dispatch_threads(
            "qwen38_prefill_add_residual_f32",
            (n_f32, 1, 1),
            (256, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(residual), 0);
                enc.set_buffer(1, Some(delta), 0);
                enc.set_buffer(2, Some(output), 0);
                set_u32(enc, 3, n_f32);
            },
        )
    }

    fn encode_prefill_embed(
        &self,
        tcb: &mut TokenCommandBuffer<'_>,
        n_tokens: u32,
    ) -> Result<()> {
        let pf = self.prefill.as_ref().ok_or_else(|| {
            Error::Model("qwen38 prefill workspace missing at embed".into())
        })?;
        const EMBED: &str = "language_model.model.embed_tokens.weight";
        let weight = self.q4(EMBED)?;
        if weight.rows != QWEN38_VOCAB || weight.cols != QWEN38_HIDDEN {
            return Err(Error::Model("qwen38 embed shape drifted".into()));
        }
        self.record_q4_weight(weight);
        let hidden = QWEN38_HIDDEN as u32;
        let vocab = QWEN38_VOCAB as u32;
        let group = weight.group_size as u32;
        tcb.dispatch_threads(
            "qwen38_prefill_q4_embed",
            (hidden, n_tokens, 1),
            (256, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&weight.codes), 0);
                enc.set_buffer(1, Some(&weight.scales), 0);
                enc.set_buffer(2, Some(&pf.tokens), 0);
                enc.set_buffer(3, Some(&pf.hidden), 0);
                set_u32(enc, 4, n_tokens);
                set_u32(enc, 5, hidden);
                set_u32(enc, 6, vocab);
                set_u32(enc, 7, group);
            },
        )
    }

    fn encode_prefill_deltanet(
        &self,
        tcb: &mut TokenCommandBuffer<'_>,
        layer: usize,
        n_tokens: u32,
    ) -> Result<()> {
        let pf = self.prefill.as_ref().ok_or_else(|| {
            Error::Model("qwen38 prefill workspace missing at deltanet".into())
        })?;
        let layout = Qwen38DeltaNetLayout::source_exact();
        let slot = self.deltanet_state_slot(layer)?;
        let conv_off = (slot * layout.conv_state_elements() * 4) as u64;
        let rec_off = (slot * layout.recurrent_state_elements() * 4) as u64;
        if !(self.fuse_add_rmsnorm && layer > 0) {
            self.encode_prefill_rmsnorm(
                tcb,
                &pf.hidden,
                self.layer_name(layer, "input_layernorm.weight"),
                &pf.normalized,
                n_tokens,
            )?;
        }
        self.encode_prefill_gemm(
            tcb,
            self.layer_name(layer, "linear_attn.in_proj_qkvz.weight"),
            &pf.normalized,
            &pf.qkvz,
            n_tokens,
        )?;
        self.encode_prefill_gemm(
            tcb,
            self.layer_name(layer, "linear_attn.in_proj_ba.weight"),
            &pf.normalized,
            &pf.ba,
            n_tokens,
        )?;
        let conv_w = self.f32(self.layer_name(layer, "linear_attn.conv1d.weight"))?;
        let qkvz_stride = layout.qkvz_rows() as u32;
        let value_stride = layout.value_elements() as u32;
        tcb.dispatch_threads(
            "qwen38_prefill_qkvz_rearrange_conv",
            (256, layout.key_heads as u32, 1),
            (256, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&pf.qkvz), 0);
                enc.set_buffer(1, Some(conv_w), 0);
                enc.set_buffer(2, Some(&self.workspace.conv_state), conv_off);
                enc.set_buffer(3, Some(&pf.repeated_q), 0);
                enc.set_buffer(4, Some(&pf.repeated_k), 0);
                enc.set_buffer(5, Some(&pf.conv_v), 0);
                enc.set_buffer(6, Some(&pf.z), 0);
                set_u32(enc, 7, n_tokens);
                set_u32(enc, 8, qkvz_stride);
                set_u32(enc, 9, value_stride);
                enc.set_bytes(10, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                enc.set_threadgroup_memory_length(0, 4 * 256 * 4);
            },
        )?;
        let a_log = self.f32(self.layer_name(layer, "linear_attn.A_log"))?;
        let dt_bias = self.f32(self.layer_name(layer, "linear_attn.dt_bias"))?;
        let ba_stride = layout.ba_rows() as u32;
        tcb.dispatch_threads(
            "qwen38_prefill_gated_delta_ba_f4",
            (layout.key_head_dim as u32, layout.value_heads as u32, 32),
            (layout.key_head_dim as u32, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&self.workspace.rec_state), rec_off);
                enc.set_buffer(1, Some(&pf.repeated_q), 0);
                enc.set_buffer(2, Some(&pf.repeated_k), 0);
                enc.set_buffer(3, Some(&pf.conv_v), 0);
                enc.set_buffer(4, Some(&pf.ba), 0);
                enc.set_buffer(5, Some(a_log), 0);
                enc.set_buffer(6, Some(dt_bias), 0);
                enc.set_buffer(7, Some(&pf.rec_out), 0);
                set_u32(enc, 8, n_tokens);
                set_u32(enc, 9, value_stride);
                set_u32(enc, 10, ba_stride);
                enc.set_threadgroup_memory_length(0, 512);
            },
        )?;
        let norm_w = self.f32(self.layer_name(layer, "linear_attn.norm.weight"))?;
        let dn_tg = if self.dn_rmsnorm_tg > 0 {
            self.dn_rmsnorm_tg
        } else {
            256
        };
        tcb.dispatch_threads(
            "qwen38_prefill_gated_rmsnorm",
            (layout.value_heads as u32 * dn_tg, 1, 1),
            (dn_tg, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&pf.rec_out), 0);
                enc.set_buffer(1, Some(&pf.z), 0);
                enc.set_buffer(2, Some(norm_w), 0);
                enc.set_buffer(3, Some(&pf.gated), 0);
                set_u32(enc, 4, n_tokens);
                set_u32(enc, 5, value_stride);
                set_u32(enc, 6, layout.value_heads as u32);
                set_u32(enc, 7, layout.value_head_dim as u32);
                enc.set_bytes(8, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                enc.set_threadgroup_memory_length(0, u64::from(dn_tg) * 4);
            },
        )?;
        self.encode_prefill_gemm(
            tcb,
            self.layer_name(layer, "linear_attn.out_proj.weight"),
            &pf.gated,
            &pf.mixer,
            n_tokens,
        )?;
        self.encode_prefill_mixer_residual(tcb, layer, n_tokens)
    }

    fn encode_prefill_gqa(
        &self,
        tcb: &mut TokenCommandBuffer<'_>,
        layer: usize,
        n_tokens: u32,
    ) -> Result<()> {
        let pf = self.prefill.as_ref().ok_or_else(|| {
            Error::Model("qwen38 prefill workspace missing at gqa".into())
        })?;
        if self.position.saturating_add(n_tokens as usize) > self.max_seq_len {
            return Err(Error::Model(format!(
                "qwen38 GQA prefill position {}+{} exceeds max_seq_len {}",
                self.position, n_tokens, self.max_seq_len
            )));
        }
        let slot = self.gqa_state_slot(layer)?;
        let slot_elems = self.max_seq_len * QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM;
        let cache_off = (slot * slot_elems * 4) as u64;
        if !(self.fuse_add_rmsnorm && layer > 0) {
            self.encode_prefill_rmsnorm(
                tcb,
                &pf.hidden,
                self.layer_name(layer, "input_layernorm.weight"),
                &pf.normalized,
                n_tokens,
            )?;
        }
        self.encode_prefill_gemm(
            tcb,
            self.layer_name(layer, "self_attn.q_proj.weight"),
            &pf.normalized,
            &pf.q_proj,
            n_tokens,
        )?;
        self.encode_prefill_gemm(
            tcb,
            self.layer_name(layer, "self_attn.k_proj.weight"),
            &pf.normalized,
            &pf.k_proj,
            n_tokens,
        )?;
        self.encode_prefill_gemm(
            tcb,
            self.layer_name(layer, "self_attn.v_proj.weight"),
            &pf.normalized,
            &pf.v_proj,
            n_tokens,
        )?;
        let q_norm = self.f32(self.layer_name(layer, "self_attn.q_norm.weight"))?;
        let k_norm = self.f32(self.layer_name(layer, "self_attn.k_norm.weight"))?;
        let rope_tg = if self.rope_tg > 0 { self.rope_tg } else { 256 };
        let pos0 = self.position as u32;
        let q_stride = (QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM * 2) as u32;
        let kv_stride = (QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM) as u32;
        let query_stride = (QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM) as u32;
        let cache_seq_stride = kv_stride;
        tcb.dispatch_threads(
            "qwen38_prefill_gqa_rope_cache",
            (n_tokens * rope_tg, QWEN38_GQA_HEADS as u32, 1),
            (rope_tg, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&pf.q_proj), 0);
                enc.set_buffer(1, Some(&pf.k_proj), 0);
                enc.set_buffer(2, Some(&pf.v_proj), 0);
                enc.set_buffer(3, Some(q_norm), 0);
                enc.set_buffer(4, Some(k_norm), 0);
                enc.set_buffer(5, Some(&pf.query), 0);
                enc.set_buffer(6, Some(&self.workspace.gqa_key), cache_off);
                enc.set_buffer(7, Some(&self.workspace.gqa_value), cache_off);
                set_u32(enc, 8, pos0);
                set_u32(enc, 9, n_tokens);
                set_u32(enc, 10, q_stride);
                set_u32(enc, 11, kv_stride);
                set_u32(enc, 12, query_stride);
                set_u32(enc, 13, cache_seq_stride);
                enc.set_threadgroup_memory_length(0, u64::from(rope_tg) * 4);
            },
        )?;
        tcb.dispatch_threads(
            "qwen38_prefill_gqa_attn",
            (n_tokens * 256, QWEN38_GQA_HEADS as u32, 1),
            (256, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&pf.query), 0);
                enc.set_buffer(1, Some(&self.workspace.gqa_key), cache_off);
                enc.set_buffer(2, Some(&self.workspace.gqa_value), cache_off);
                enc.set_buffer(3, Some(&pf.attn), 0);
                set_u32(enc, 4, pos0);
                set_u32(enc, 5, n_tokens);
                set_u32(enc, 6, query_stride);
                set_u32(enc, 7, cache_seq_stride);
                enc.set_threadgroup_memory_length(0, QWEN38_PREFILL_GQA_ATTN_SHMEM_BYTES);
            },
        )?;
        tcb.dispatch_threads(
            "qwen38_prefill_sigmoid_gate",
            (query_stride, n_tokens, 1),
            (256, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&pf.attn), 0);
                enc.set_buffer(1, Some(&pf.q_proj), 0);
                enc.set_buffer(2, Some(&pf.gated_attn), 0);
                set_u32(enc, 3, n_tokens);
                set_u32(enc, 4, query_stride);
                set_u32(enc, 5, q_stride);
            },
        )?;
        self.encode_prefill_gemm(
            tcb,
            self.layer_name(layer, "self_attn.o_proj.weight"),
            &pf.gated_attn,
            &pf.mixer,
            n_tokens,
        )?;
        self.encode_prefill_mixer_residual(tcb, layer, n_tokens)
    }

    fn encode_prefill_mixer_residual(
        &self,
        tcb: &mut TokenCommandBuffer<'_>,
        layer: usize,
        n_tokens: u32,
    ) -> Result<()> {
        let pf = self.prefill.as_ref().ok_or_else(|| {
            Error::Model("qwen38 prefill workspace missing at mixer residual".into())
        })?;
        if self.fuse_add_rmsnorm {
            self.encode_prefill_add_residual_rmsnorm(
                tcb,
                &pf.hidden,
                &pf.mixer,
                &pf.first_residual,
                self.layer_name(layer, "post_attention_layernorm.weight"),
                &pf.normalized,
                n_tokens,
            )
        } else {
            self.encode_prefill_add(
                tcb,
                &pf.hidden,
                &pf.mixer,
                &pf.first_residual,
                n_tokens * QWEN38_HIDDEN as u32,
            )
        }
    }

    fn encode_prefill_mlp(
        &self,
        tcb: &mut TokenCommandBuffer<'_>,
        layer: usize,
        n_tokens: u32,
    ) -> Result<()> {
        let pf = self.prefill.as_ref().ok_or_else(|| {
            Error::Model("qwen38 prefill workspace missing at mlp".into())
        })?;
        if !self.fuse_add_rmsnorm {
            self.encode_prefill_rmsnorm(
                tcb,
                &pf.first_residual,
                self.layer_name(layer, "post_attention_layernorm.weight"),
                &pf.normalized,
                n_tokens,
            )?;
        }
        self.encode_prefill_gemm(
            tcb,
            self.layer_name(layer, "mlp.gate_proj.weight"),
            &pf.normalized,
            &pf.gate,
            n_tokens,
        )?;
        self.encode_prefill_gemm(
            tcb,
            self.layer_name(layer, "mlp.up_proj.weight"),
            &pf.normalized,
            &pf.up,
            n_tokens,
        )?;
        let n = n_tokens * QWEN38_INTERMEDIATE as u32;
        tcb.dispatch_threads(
            "qwen38_prefill_swiglu_f32",
            (n, 1, 1),
            (256, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&pf.gate), 0);
                enc.set_buffer(1, Some(&pf.up), 0);
                enc.set_buffer(2, Some(&pf.act), 0);
                set_u32(enc, 3, n);
            },
        )?;
        self.encode_prefill_gemm(
            tcb,
            self.layer_name(layer, "mlp.down_proj.weight"),
            &pf.act,
            &pf.down,
            n_tokens,
        )?;
        if self.fuse_add_rmsnorm {
            let next = self.next_norm_weight_name(layer).to_owned();
            self.encode_prefill_add_residual_rmsnorm(
                tcb,
                &pf.first_residual,
                &pf.down,
                &pf.hidden,
                &next,
                &pf.normalized,
                n_tokens,
            )
        } else {
            self.encode_prefill_add(
                tcb,
                &pf.first_residual,
                &pf.down,
                &pf.hidden,
                n_tokens * QWEN38_HIDDEN as u32,
            )
        }
    }

    fn encode_prefill_terminal(
        &self,
        tcb: &mut TokenCommandBuffer<'_>,
        n_tokens: u32,
    ) -> Result<()> {
        let pf = self.prefill.as_ref().ok_or_else(|| {
            Error::Model("qwen38 prefill workspace missing at terminal".into())
        })?;
        if !self.fuse_add_rmsnorm {
            self.encode_prefill_rmsnorm(
                tcb,
                &pf.hidden,
                "language_model.model.norm.weight",
                &pf.normalized,
                n_tokens,
            )?;
        }
        let last = n_tokens.saturating_sub(1);
        let hidden = QWEN38_HIDDEN as u32;
        tcb.dispatch_threads(
            "qwen38_prefill_copy_row",
            (hidden, 1, 1),
            (256, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&pf.normalized), 0);
                enc.set_buffer(1, Some(&self.workspace.normalized), 0);
                set_u32(enc, 2, last);
                set_u32(enc, 3, hidden);
            },
        )?;
        self.encode_named_matvec(
            tcb,
            "language_model.lm_head.weight",
            &self.workspace.normalized,
            &self.workspace.logits,
        )?;
        sample_argmax_f32_tcb(
            tcb,
            &self.workspace.logits,
            &self.workspace.sampled,
            QWEN38_VOCAB,
        )
    }

    fn encode_prefill_chunk(
        &self,
        tcb: &mut TokenCommandBuffer<'_>,
        n_tokens: u32,
        last_chunk: bool,
    ) -> Result<()> {
        self.encode_prefill_embed(tcb, n_tokens)?;
        for layer in 0..QWEN38_LAYERS {
            match self.mixer_kind(layer)? {
                Qwen38MixerKind::DeltaNet => self.encode_prefill_deltanet(tcb, layer, n_tokens)?,
                Qwen38MixerKind::Gqa => self.encode_prefill_gqa(tcb, layer, n_tokens)?,
            }
            self.encode_prefill_mlp(tcb, layer, n_tokens)?;
        }
        if last_chunk {
            self.encode_prefill_terminal(tcb, n_tokens)?;
        }
        Ok(())
    }

    /// Consume `prompt` as GEMM chunks. Leaves conv / recurrent / KV state
    /// as `step` would have, `position == prompt.len()`, and the first new
    /// token in `workspace.sampled`.
    fn prefill_prompt(&mut self, prompt: &[u32]) -> Result<(u32, Qwen38PrefillTiming)> {
        if prompt.is_empty() {
            return Err(Error::Model("qwen38 prompt is empty".into()));
        }
        if prompt.len() > self.max_seq_len {
            return Err(Error::Model(format!(
                "qwen38 prompt {} exceeds max_seq_len {}",
                prompt.len(),
                self.max_seq_len
            )));
        }
        self.ensure_prefill_workspace()?;
        let cap = self.prefill.as_ref().map(|p| p.cap).unwrap_or(1);
        let mut timing = Qwen38PrefillTiming::default();
        let mut offset = 0usize;
        while offset < prompt.len() {
            let n = (prompt.len() - offset).min(cap);
            let last_chunk = offset + n == prompt.len();
            {
                let pf = self.prefill.as_ref().ok_or_else(|| {
                    Error::Model("qwen38 prefill workspace missing".into())
                })?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        prompt[offset..offset + n].as_ptr(),
                        pf.tokens.contents() as *mut u32,
                        n,
                    );
                }
            }
            self.reset_active_weight_bytes();
            let mut tcb = TokenCommandBuffer::new(&self.context);
            self.enable_dispatch_name_trace(&mut tcb);
            if self.serial_token_encoder {
                tcb.begin_serial_group()?;
            }
            self.encode_prefill_chunk(&mut tcb, n as u32, last_chunk)?;
            if self.serial_token_encoder {
                tcb.end_serial_group()?;
            }
            let harvested = tcb.structural_kernel_names().map(|names| names.to_vec());
            let chunk_timing = tcb.commit_and_wait_timed()?;
            self.harvest_dispatch_names(harvested);
            timing.accumulate(&chunk_timing, self.last_active_weight_bytes());
            self.position = self.position.saturating_add(n);
            offset += n;
        }
        let sampled = unsafe { *(self.workspace.sampled.contents() as *const u32) };
        Ok((sampled, timing))
    }
}

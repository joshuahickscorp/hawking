//! GPU-resident decode state for GLM-5.2.
//!
//! Prerequisite lane for Temporal Gravity command-buffer collapse. The host
//! path keeps the residual stream and per-layer KV / DSA caches as host
//! `Vec<f32>`, so every projection must finish and return a host vector before
//! the next host loop can run (~1,171 `commit_and_wait`s per flagship token).
//!
//! This module keeps those tensors in device (Metal shared) buffers across a
//! token: activations, KV, index keys, router logits / top-k / expert offsets.
//! Discrete decisions (stable top-k, noaux_tc groups, sparse softmax) still use
//! the same host arithmetic as [`crate::gravity_glm::forward_impl`] so token
//! identity is bit-exact against the host-state path; they read device-mapped
//! memory in place rather than owning a separate host cache. Projection outputs
//! are written straight into those buffers and are not copied into host `Vec`s
//! as the cache of record.
//!
//! `lm_head` is once per token. Default: host dense or PQ via
//! [`GpuWeightCache::matvec`]. With [`crate::gravity_glm::GPU_LM_HEAD_ENV`]=1 and
//! a `native.bf16` head, the weight stays device-resident and the projection
//! + greedy argmax + top-k diagnostics run on GPU (no per-token host widen of
//! the 1.90 GB table). Default readback is **token + top-k only**; full logits
//! require `HAWKING_GLM_GPU_LM_HEAD_FULL_LOGITS=1`. The same flag keeps other
//! rank-2 `native.bf16` matvecs (indexer, router) as device bf16.
//!
//! **Expert-wave** (`HAWKING_GLM_GPU_EXPERT_WAVE=1`, default off): opt-in collapse
//! of each MLP layer to one command buffer (`gate + up → SiLU → down` and MoE
//! weighted combine). The default three-`matvec_batch` path is unchanged when
//! the flag is unset (Parity V2.1 item 6).
//!
//! Gated by [`GPU_RESIDENT_STATE_ENV`] (`HAWKING_GLM_GPU_RESIDENT_STATE`), default
//! off, so the host-state path remains the parity oracle.

#![cfg(target_os = "macos")]

use crate::gravity::matvec_dense;
use crate::gravity_glm::gpu::{
    encode_argmax_f32, encode_gemv_native_bf16_seq, encode_sample_topk_f32, GpuTensor,
    GpuWeightCache,
};
use crate::gravity_glm::{
    gpu_expert_wave_enabled, gpu_lm_head_full_logits_enabled, rope_cos_sin, rope_interleaved,
    topk_desc, GlmArch, GlmTrace, WeightAccess, GPU_LM_HEAD_DIAG_TOPK,
};
use crate::metal::{MetalContext, TokenCommandBuffer};
use crate::{Error, Result};
use metal::Buffer;
use std::cell::Cell;
use std::sync::Mutex;

// Flag + static wait estimators live on `gravity_glm` so non-Metal unit tests
// can see them: `GPU_RESIDENT_STATE_ENV`, `gpu_resident_state_enabled`,
// `estimate_host_state_waits_per_token`, `estimate_resident_waits_per_token`.

fn write_f32(buf: &Buffer, src: &[f32]) {
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), buf.contents() as *mut f32, src.len());
    }
}

fn read_f32(buf: &Buffer, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n).to_vec() }
}

fn read_u32(buf: &Buffer, n: usize) -> Vec<u32> {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const u32, n).to_vec() }
}

/// Per-layer device KV / DSA cache.
struct LayerGpuCache {
    keys: Buffer,
    values: Buffer,
    index_keys: Buffer,
    capacity: usize,
}

/// Device-resident working set for one generation.
pub struct ResidentSession {
    layers: Vec<LayerGpuCache>,
    pub seq_len: usize,
    shared_topk: Option<Vec<usize>>,
    waits: Cell<u64>,
}

impl ResidentSession {
    pub fn new(ctx: &MetalContext, arch: &GlmArch, initial_cap: usize) -> Result<Self> {
        let cap = initial_cap.max(4);
        let qk = arch.qk_dim();
        let mut layers = Vec::with_capacity(arch.n_layers);
        for _ in 0..arch.n_layers {
            layers.push(LayerGpuCache {
                keys: ctx.new_buffer_checked(cap * arch.n_heads * qk * 4)?,
                values: ctx.new_buffer_checked(cap * arch.n_heads * arch.v_head_dim * 4)?,
                index_keys: ctx.new_buffer_checked(cap * arch.index_head_dim * 4)?,
                capacity: cap,
            });
        }
        Ok(Self {
            layers,
            seq_len: 0,
            shared_topk: None,
            waits: Cell::new(0),
        })
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
        self.shared_topk = None;
        self.waits.set(0);
    }

    pub fn waits(&self) -> u64 {
        self.waits.get()
    }

    fn reserve(&mut self, ctx: &MetalContext, arch: &GlmArch, need: usize) -> Result<()> {
        let cur = self.layers.first().map(|l| l.capacity).unwrap_or(0);
        if need <= cur {
            return Ok(());
        }
        let cap = need.next_power_of_two().max(8);
        let qk = arch.qk_dim();
        for layer in 0..arch.n_layers {
            let old = &self.layers[layer];
            let nk = ctx.new_buffer_checked(cap * arch.n_heads * qk * 4)?;
            let nv = ctx.new_buffer_checked(cap * arch.n_heads * arch.v_head_dim * 4)?;
            let ni = ctx.new_buffer_checked(cap * arch.index_head_dim * 4)?;
            if self.seq_len > 0 {
                let ck = self.seq_len * arch.n_heads * qk * 4;
                let cv = self.seq_len * arch.n_heads * arch.v_head_dim * 4;
                let ci = self.seq_len * arch.index_head_dim * 4;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        old.keys.contents() as *const u8,
                        nk.contents() as *mut u8,
                        ck,
                    );
                    std::ptr::copy_nonoverlapping(
                        old.values.contents() as *const u8,
                        nv.contents() as *mut u8,
                        cv,
                    );
                    std::ptr::copy_nonoverlapping(
                        old.index_keys.contents() as *const u8,
                        ni.contents() as *mut u8,
                        ci,
                    );
                }
            }
            self.layers[layer] = LayerGpuCache {
                keys: nk,
                values: nv,
                index_keys: ni,
                capacity: cap,
            };
        }
        Ok(())
    }
}

/// Activation / scratch pool (device buffers, reused every token).
pub struct ActPool {
    pub x: Buffer,
    h: Buffer,
    q_a: Buffer,
    q_resid: Buffer,
    q: Buffer,
    compressed: Buffer,
    k_latent: Buffer,
    kv: Buffer,
    queries: Buffer,
    context: Buffer,
    o: Buffer,
    // DSA / router scratch kept on device as the cache of record
    idx_q: Buffer,
    idx_k_raw: Buffer,
    idx_head_w: Buffer,
    idx_scores: Buffer,
    router_logits: Buffer,
    router_scores: Buffer,
    router_corrected: Buffer,
    expert_idx: Buffer,
    expert_w: Buffer,
    // Expert scratch (sized for future device-side expert chaining; the
    // batched path currently uses matvec_batch into host Vecs for the three
    // co-issued waits that match the host oracle).
    #[allow(dead_code)]
    gate: Buffer,
    #[allow(dead_code)]
    up: Buffer,
    #[allow(dead_code)]
    act: Buffer,
    #[allow(dead_code)]
    down: Buffer,
    final_hidden: Buffer,
    /// Device logits for device-resident lm_head (vocab-sized). Stay on device;
    /// host only reads them under `HAWKING_GLM_GPU_LM_HEAD_FULL_LOGITS=1`.
    logits: Buffer,
    /// Single u32 greedy token from on-device argmax (token-only readback).
    sample_token: Buffer,
    /// Diagnostic top-k indices over device logits (`GPU_LM_HEAD_DIAG_TOPK`).
    head_topk_idx: Buffer,
    /// Diagnostic top-k values over device logits.
    head_topk_val: Buffer,
    #[allow(dead_code)]
    gate_cap: usize,
}

impl ActPool {
    pub fn new(ctx: &MetalContext, arch: &GlmArch) -> Result<Self> {
        let h = arch.hidden;
        let qk = arch.qk_dim();
        let gate_cap = (h * 32).max(4096);
        let max_keys = 8192;
        Ok(Self {
            x: ctx.new_buffer_checked(h * 4)?,
            h: ctx.new_buffer_checked(h * 4)?,
            q_a: ctx.new_buffer_checked(arch.q_lora_rank * 4)?,
            q_resid: ctx.new_buffer_checked(arch.q_lora_rank * 4)?,
            q: ctx.new_buffer_checked(arch.n_heads * qk * 4)?,
            compressed: ctx
                .new_buffer_checked((arch.kv_lora_rank + arch.qk_rope_head_dim) * 4)?,
            k_latent: ctx.new_buffer_checked(arch.kv_lora_rank * 4)?,
            kv: ctx.new_buffer_checked(
                arch.n_heads * (arch.qk_nope_head_dim + arch.v_head_dim) * 4,
            )?,
            queries: ctx.new_buffer_checked(arch.n_heads * qk * 4)?,
            context: ctx.new_buffer_checked(arch.n_heads * arch.v_head_dim * 4)?,
            o: ctx.new_buffer_checked(h * 4)?,
            idx_q: ctx.new_buffer_checked(arch.index_n_heads * arch.index_head_dim * 4)?,
            idx_k_raw: ctx.new_buffer_checked(arch.index_head_dim * 4)?,
            idx_head_w: ctx.new_buffer_checked(arch.index_n_heads * 4)?,
            idx_scores: ctx.new_buffer_checked(max_keys * 4)?,
            router_logits: ctx.new_buffer_checked(arch.n_routed_experts * 4)?,
            router_scores: ctx.new_buffer_checked(arch.n_routed_experts * 4)?,
            router_corrected: ctx.new_buffer_checked(arch.n_routed_experts * 4)?,
            expert_idx: ctx.new_buffer_checked(arch.num_experts_per_tok.max(1) * 4)?,
            expert_w: ctx.new_buffer_checked(arch.num_experts_per_tok.max(1) * 4)?,
            gate: ctx.new_buffer_checked(gate_cap * 4)?,
            up: ctx.new_buffer_checked(gate_cap * 4)?,
            act: ctx.new_buffer_checked(gate_cap * 4)?,
            down: ctx.new_buffer_checked(h * 4)?,
            final_hidden: ctx.new_buffer_checked(h * 4)?,
            logits: ctx.new_buffer_checked(arch.vocab_size * 4)?,
            sample_token: ctx.new_buffer_checked(4)?,
            head_topk_idx: ctx
                .new_buffer_checked((GPU_LM_HEAD_DIAG_TOPK as usize) * 4)?,
            head_topk_val: ctx
                .new_buffer_checked((GPU_LM_HEAD_DIAG_TOPK as usize) * 4)?,
            gate_cap,
        })
    }
}

fn commit(tcb: Option<TokenCommandBuffer<'_>>, waits: &Cell<u64>) -> Result<()> {
    if let Some(buf) = tcb {
        buf.commit_and_wait()?;
        waits.set(waits.get().saturating_add(1));
    }
    Ok(())
}

/// Matvec into a device buffer. Host-native weights run on the host into the
/// shared buffer (no wait). PQ and device-resident bf16 encode into `tcb` and
/// need a later commit.
fn matvec_into<'a>(
    tcb: &mut Option<TokenCommandBuffer<'a>>,
    ctx: &'a MetalContext,
    weights: &GpuWeightCache,
    name: &str,
    x: &Buffer,
    x_len: usize,
    y: &Buffer,
) -> Result<()> {
    crate::cost_ledger::record_matvec_call();
    let mut cache = weights.cache.lock().expect("gpu weight cache");
    weights.ensure_many_locked(&mut cache, &[name])?;
    match cache.get(name).expect("ensured") {
        GpuTensor::NativeCpu(w) => {
            // Host-oracle path (flag off): do not change default billing or
            // numerics. Active-byte category partition for native.f32 widen is
            // owned by WeightAccess::matvec when that path is used.
            let x_host = read_f32(x, x_len);
            let y_host = matvec_dense(w, &x_host, name)?;
            write_f32(y, &y_host);
            Ok(())
        }
        GpuTensor::NativeGpuBf16 { buf, rows, cols } => {
            if x_len != *cols as usize {
                return Err(Error::Gravity(format!(
                    "resident matvec {name}: x_len {x_len} != cols {cols}"
                )));
            }
            // Bill stored bf16 length (no f32 widen tax).
            crate::cost_ledger::record_active_bytes_for(name, buf.length());
            let tcb = tcb.get_or_insert_with(|| TokenCommandBuffer::new(ctx));
            encode_gemv_native_bf16_seq(tcb, buf, *rows, *cols, x, y)
        }
        GpuTensor::Pq {
            codebooks,
            codes,
            params,
        } => {
            if x_len != params.cols as usize {
                return Err(Error::Gravity(format!(
                    "resident matvec {name}: x_len {x_len} != cols {}",
                    params.cols
                )));
            }
            let tcb = tcb.get_or_insert_with(|| TokenCommandBuffer::new(ctx));
            const TG: u32 = 256;
            let n_tg = params.rows.div_ceil(8);
            let params = *params;
            let cb = codebooks.clone();
            let co = codes.clone();
            tcb.dispatch_threads("gravity_pq_matvec", (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
                enc.set_buffer(0, Some(&cb), 0);
                enc.set_buffer(1, Some(&co), 0);
                enc.set_buffer(2, Some(x), 0);
                enc.set_buffer(3, Some(y), 0);
                enc.set_bytes(
                    4,
                    std::mem::size_of_val(&params) as u64,
                    &params as *const _ as *const _,
                );
            })?;
            Ok(())
        }
    }
}

fn rmsnorm_into(x: &Buffer, x_len: usize, weight: &[f32], eps: f32, out: &Buffer) {
    let xv = read_f32(x, x_len);
    let mean_sq = xv.iter().map(|v| v * v).sum::<f32>() / x_len as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    let y: Vec<f32> = xv
        .iter()
        .zip(weight)
        .map(|(v, w)| v * inv * *w)
        .collect();
    write_f32(out, &y);
}

fn residual_add(x: &Buffer, add: &Buffer, n: usize) {
    let mut xv = read_f32(x, n);
    let av = read_f32(add, n);
    for (a, b) in xv.iter_mut().zip(&av) {
        *a += *b;
    }
    write_f32(x, &xv);
}

/// One generation step (or prefill) with decode state on device.
pub fn forward_resident(
    weights: &GpuWeightCache,
    arch: &GlmArch,
    session: &mut ResidentSession,
    pool: &ActPool,
    tokens: &[u32],
    start_pos: usize,
) -> Result<(Vec<f32>, GlmTrace, u64)> {
    if tokens.is_empty() {
        return Err(Error::Gravity("forward_resident: no tokens".into()));
    }
    let ctx = &weights.ctx;
    let a = arch;
    let qk = a.qk_dim();
    session.reserve(ctx, arch, start_pos + tokens.len())?;
    let waits_before = session.waits.get();
    let mut logits = Vec::new();
    let mut trace = GlmTrace::default();

    for (step, &token) in tokens.iter().enumerate() {
        let pos = start_pos + step;
        if token as usize >= a.vocab_size {
            return Err(Error::Gravity(format!(
                "token {token} out of range for vocab_size {}",
                a.vocab_size
            )));
        }

        let emb = weights.row("model.embed_tokens.weight", token as usize, a.hidden)?;
        write_f32(&pool.x, &emb);
        let (cos, sin) = rope_cos_sin(arch, pos);
        let mut shared_topk = session.shared_topk.clone();
        trace.expert_choices.clear();

        for layer in 0..a.n_layers {
            let p = format!("model.layers.{layer}");
            let attn_p = format!("{p}.self_attn");
            let mut tcb: Option<TokenCommandBuffer<'_>> = None;

            let w_in = weights.dense(&format!("{p}.input_layernorm.weight"))?;
            rmsnorm_into(&pool.x, a.hidden, &w_in, a.rms_norm_eps, &pool.h);

            // Q path
            matvec_into(
                &mut tcb,
                ctx,
                weights,
                &format!("{attn_p}.q_a_proj.weight"),
                &pool.h,
                a.hidden,
                &pool.q_a,
            )?;
            // KV-a is independent of q_a — co-issue before the wait.
            matvec_into(
                &mut tcb,
                ctx,
                weights,
                &format!("{attn_p}.kv_a_proj_with_mqa.weight"),
                &pool.h,
                a.hidden,
                &pool.compressed,
            )?;
            commit(tcb.take(), &session.waits)?;

            let w_q = weights.dense(&format!("{attn_p}.q_a_layernorm.weight"))?;
            rmsnorm_into(
                &pool.q_a,
                a.q_lora_rank,
                &w_q,
                a.rms_norm_eps,
                &pool.q_resid,
            );

            let compressed =
                read_f32(&pool.compressed, a.kv_lora_rank + a.qk_rope_head_dim);
            let w_kv = weights.dense(&format!("{attn_p}.kv_a_layernorm.weight"))?;
            let k_latent = {
                let x = &compressed[..a.kv_lora_rank];
                let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
                let inv = 1.0 / (mean_sq + a.rms_norm_eps).sqrt();
                x.iter()
                    .zip(&w_kv)
                    .map(|(v, w)| v * inv * w)
                    .collect::<Vec<_>>()
            };
            write_f32(&pool.k_latent, &k_latent);
            let k_rot = rope_interleaved(&compressed[a.kv_lora_rank..], &cos, &sin);

            matvec_into(
                &mut tcb,
                ctx,
                weights,
                &format!("{attn_p}.q_b_proj.weight"),
                &pool.q_resid,
                a.q_lora_rank,
                &pool.q,
            )?;
            matvec_into(
                &mut tcb,
                ctx,
                weights,
                &format!("{attn_p}.kv_b_proj.weight"),
                &pool.k_latent,
                a.kv_lora_rank,
                &pool.kv,
            )?;
            commit(tcb.take(), &session.waits)?;

            // Append MLA K/V into the device cache (cache of record).
            {
                let kv = read_f32(
                    &pool.kv,
                    a.n_heads * (a.qk_nope_head_dim + a.v_head_dim),
                );
                let cache = &session.layers[layer];
                let per = a.qk_nope_head_dim + a.v_head_dim;
                let mut keys_pos = Vec::with_capacity(a.n_heads * qk);
                let mut vals_pos = Vec::with_capacity(a.n_heads * a.v_head_dim);
                for head in 0..a.n_heads {
                    let src = &kv[head * per..(head + 1) * per];
                    keys_pos.extend_from_slice(&src[..a.qk_nope_head_dim]);
                    keys_pos.extend_from_slice(&k_rot);
                    vals_pos.extend_from_slice(&src[a.qk_nope_head_dim..]);
                }
                let k_off = pos * a.n_heads * qk;
                let v_off = pos * a.n_heads * a.v_head_dim;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        keys_pos.as_ptr(),
                        (cache.keys.contents() as *mut f32).add(k_off),
                        keys_pos.len(),
                    );
                    std::ptr::copy_nonoverlapping(
                        vals_pos.as_ptr(),
                        (cache.values.contents() as *mut f32).add(v_off),
                        vals_pos.len(),
                    );
                }
            }

            // Queries
            {
                let q = read_f32(&pool.q, a.n_heads * qk);
                let mut queries = vec![0f32; a.n_heads * qk];
                for head in 0..a.n_heads {
                    let src = &q[head * qk..(head + 1) * qk];
                    let dst = &mut queries[head * qk..(head + 1) * qk];
                    dst[..a.qk_nope_head_dim].copy_from_slice(&src[..a.qk_nope_head_dim]);
                    dst[a.qk_nope_head_dim..]
                        .copy_from_slice(&rope_interleaved(&src[a.qk_nope_head_dim..], &cos, &sin));
                }
                write_f32(&pool.queries, &queries);
            }

            let topk = match a.indexer_types[layer].as_str() {
                "full" => {
                    let t = indexer_topk(
                        weights,
                        arch,
                        &attn_p,
                        pool,
                        &session.layers[layer],
                        pos,
                        &cos,
                        &sin,
                        &mut tcb,
                        ctx,
                        &session.waits,
                    )?;
                    shared_topk = Some(t.clone());
                    session.shared_topk = Some(t.clone());
                    t
                }
                "shared" => shared_topk.clone().ok_or_else(|| {
                    Error::Gravity(format!(
                        "layer {layer} shares an index but no earlier layer computed one"
                    ))
                })?,
                other => {
                    return Err(Error::Gravity(format!(
                        "layer {layer}: unknown indexer type {other:?}"
                    )))
                }
            };

            // Sparse attend over device-resident KV (same math as forward_impl).
            let context = sparse_attend(a, pool, &session.layers[layer], pos, &topk, qk)?;
            write_f32(&pool.context, &context);

            matvec_into(
                &mut tcb,
                ctx,
                weights,
                &format!("{attn_p}.o_proj.weight"),
                &pool.context,
                a.n_heads * a.v_head_dim,
                &pool.o,
            )?;
            commit(tcb.take(), &session.waits)?;
            residual_add(&pool.x, &pool.o, a.hidden);

            let w_post = weights.dense(&format!("{p}.post_attention_layernorm.weight"))?;
            rmsnorm_into(&pool.x, a.hidden, &w_post, a.rms_norm_eps, &pool.h);

            match a.mlp_layer_types[layer].as_str() {
                "dense" => {
                    let prefix = format!("{p}.mlp");
                    let out = mlp_one(
                        weights,
                        &prefix,
                        &pool.h,
                        a.hidden,
                        pool,
                        &mut tcb,
                        ctx,
                        &session.waits,
                    )?;
                    write_f32(&pool.o, &out);
                    residual_add(&pool.x, &pool.o, a.hidden);
                }
                "sparse" => {
                    let prefix = format!("{p}.mlp");
                    matvec_into(
                        &mut tcb,
                        ctx,
                        weights,
                        &format!("{prefix}.gate.weight"),
                        &pool.h,
                        a.hidden,
                        &pool.router_logits,
                    )?;
                    commit(tcb.take(), &session.waits)?;

                    let (indices, moe_weights) = router_select(weights, a, &prefix, pool)?;
                    // Residency: expert selection + weights live on device.
                    let idx_u: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            idx_u.as_ptr(),
                            pool.expert_idx.contents() as *mut u32,
                            idx_u.len(),
                        );
                    }
                    write_f32(&pool.expert_w, &moe_weights);
                    trace.expert_choices.push(indices.clone());

                    // Ascending expert order (float-add associativity), then
                    // shared last — same as host `routed_moe` / `batched_mlp`.
                    let mut order: Vec<usize> = (0..indices.len()).collect();
                    order.sort_by_key(|&s| indices[s]);
                    let prefixes: Vec<String> = order
                        .iter()
                        .map(|&slot| format!("{prefix}.experts.{}", indices[slot]))
                        .chain(std::iter::once(format!("{prefix}.shared_experts")))
                        .collect();
                    // Expert-wave (flagged, default off): one CB for gate/up/SiLU/
                    // down/weighted combine. Default three-batch path is unchanged.
                    let routed = if gpu_expert_wave_enabled() {
                        let scales: Vec<f32> = order
                            .iter()
                            .map(|&slot| moe_weights[slot])
                            .chain(std::iter::once(1.0f32))
                            .collect();
                        moe_device_wave(
                            weights,
                            &prefixes,
                            &scales,
                            &pool.h,
                            a.hidden,
                            &mut tcb,
                            ctx,
                            &session.waits,
                        )?
                    } else {
                        let mut outs = batched_mlp(
                            weights,
                            &prefixes,
                            &pool.h,
                            a.hidden,
                            pool,
                            &mut tcb,
                            ctx,
                            &session.waits,
                        )?;
                        let shared = outs.pop().expect("shared last");
                        let mut routed = vec![0f32; a.hidden];
                        for (out, &slot) in outs.iter().zip(&order) {
                            for (r, o) in routed.iter_mut().zip(out) {
                                *r += o * moe_weights[slot];
                            }
                        }
                        for (r, s) in routed.iter_mut().zip(&shared) {
                            *r += *s;
                        }
                        routed
                    };
                    write_f32(&pool.o, &routed);
                    residual_add(&pool.x, &pool.o, a.hidden);
                }
                other => {
                    return Err(Error::Gravity(format!(
                        "layer {layer}: unknown MLP type {other:?}"
                    )))
                }
            }

            if layer + 1 == a.n_layers {
                trace.final_topk = topk;
            }
        }

        let w_norm = weights.dense("model.norm.weight")?;
        rmsnorm_into(
            &pool.x,
            a.hidden,
            &w_norm,
            a.rms_norm_eps,
            &pool.final_hidden,
        );
        // lm_head once per token. Device-resident bf16 keeps weight + logits
        // on GPU, runs blockwise gemv + greedy argmax + top-k, and by default
        // reads back only the token + top-k diagnostics (not 154_880 logits).
        // Full logits: HAWKING_GLM_GPU_LM_HEAD_FULL_LOGITS=1. Host dense / PQ
        // keep the prior path (full logits).
        let waits_before_head = session.waits.get();
        {
            let _head = crate::cost_ledger::Scope::new(
                crate::cost_ledger::Bucket::FinalHeadAndSampling,
            );
            let mut cache = weights.cache.lock().expect("gpu weight cache");
            weights.ensure_many_locked(&mut cache, &["lm_head.weight"])?;
            match cache.get("lm_head.weight").expect("ensured lm_head") {
                GpuTensor::NativeGpuBf16 { buf, rows, cols } => {
                    if a.hidden != *cols as usize {
                        return Err(Error::Gravity(format!(
                            "lm_head device path: hidden {} != cols {cols}",
                            a.hidden
                        )));
                    }
                    if a.vocab_size != *rows as usize {
                        return Err(Error::Gravity(format!(
                            "lm_head device path: vocab {} != rows {rows}",
                            a.vocab_size
                        )));
                    }
                    crate::cost_ledger::record_matvec_call();
                    crate::cost_ledger::record_active_bytes_for("lm_head.weight", buf.length());
                    let mut tcb = TokenCommandBuffer::new(ctx);
                    encode_gemv_native_bf16_seq(
                        &mut tcb,
                        buf,
                        *rows,
                        *cols,
                        &pool.final_hidden,
                        &pool.logits,
                    )?;
                    encode_argmax_f32(&mut tcb, &pool.logits, *rows, &pool.sample_token)?;
                    encode_sample_topk_f32(
                        &mut tcb,
                        &pool.logits,
                        *rows,
                        GPU_LM_HEAD_DIAG_TOPK,
                        &pool.head_topk_idx,
                        &pool.head_topk_val,
                    )?;
                    tcb.commit_and_wait()?;
                    session.waits.set(session.waits.get().saturating_add(1));

                    // Token + diagnostics only (default). Full vector is opt-in.
                    let tok = read_u32(&pool.sample_token, 1)[0];
                    let k = GPU_LM_HEAD_DIAG_TOPK as usize;
                    let topk_idx = read_u32(&pool.head_topk_idx, k);
                    let topk_val = read_f32(&pool.head_topk_val, k);
                    crate::cost_ledger::record_transfer(
                        (4 + k * 4 + k * 4) as u64,
                        false,
                        "lm_head_token_diag_download",
                    );
                    trace.sample_token = Some(tok);
                    trace.head_topk_idx = topk_idx;
                    trace.head_topk_val = topk_val;

                    if gpu_lm_head_full_logits_enabled() {
                        logits = read_f32(&pool.logits, a.vocab_size);
                        crate::cost_ledger::record_transfer(
                            (a.vocab_size * 4) as u64,
                            false,
                            "lm_head_y_download",
                        );
                        trace.head_full_logits_readback = true;
                    } else {
                        // Empty host logits: callers that need a token use
                        // `trace.sample_token`. Continuous-logit parity sets
                        // FULL_LOGITS=1.
                        logits = Vec::new();
                        trace.head_full_logits_readback = false;
                    }
                }
                GpuTensor::NativeCpu(_) | GpuTensor::Pq { .. } => {
                    drop(cache);
                    let hidden = read_f32(&pool.final_hidden, a.hidden);
                    logits = weights.matvec("lm_head.weight", &hidden)?;
                    trace.head_full_logits_readback = true;
                    if session.waits.get() == waits_before_head {
                        let mut cache = weights.cache.lock().expect("gpu weight cache");
                        weights.ensure_many_locked(&mut cache, &["lm_head.weight"])?;
                        if matches!(cache.get("lm_head.weight"), Some(GpuTensor::Pq { .. })) {
                            session.waits.set(session.waits.get().saturating_add(1));
                        }
                    }
                }
            }
        }
    }

    session.seq_len = start_pos + tokens.len();
    let waits = session.waits.get().saturating_sub(waits_before);
    Ok((logits, trace, waits))
}

fn router_select(
    weights: &GpuWeightCache,
    a: &GlmArch,
    prefix: &str,
    pool: &ActPool,
) -> Result<(Vec<usize>, Vec<f32>)> {
    let logits = read_f32(&pool.router_logits, a.n_routed_experts);
    let scores: Vec<f32> = logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
    write_f32(&pool.router_scores, &scores);
    let bias = weights.dense(&format!("{prefix}.gate.e_score_correction_bias"))?;
    let corrected: Vec<f32> = scores.iter().zip(&bias).map(|(s, b)| s + b).collect();
    write_f32(&pool.router_corrected, &corrected);
    let per_group = a.n_routed_experts / a.n_group;
    let group_scores: Vec<f32> = (0..a.n_group)
        .map(|g| {
            let slice = &corrected[g * per_group..(g + 1) * per_group];
            topk_desc(slice, 2.min(per_group))
                .iter()
                .map(|&i| slice[i])
                .sum()
        })
        .collect();
    let chosen = topk_desc(&group_scores, a.topk_group);
    let mut choice = vec![f32::NEG_INFINITY; a.n_routed_experts];
    for &g in &chosen {
        for e in g * per_group..(g + 1) * per_group {
            choice[e] = corrected[e];
        }
    }
    let indices = topk_desc(&choice, a.num_experts_per_tok);
    let mut weights_out: Vec<f32> = indices.iter().map(|&i| scores[i]).collect();
    if a.norm_topk_prob {
        let total: f32 = weights_out.iter().sum::<f32>() + 1e-20;
        for w in weights_out.iter_mut() {
            *w /= total;
        }
    }
    for w in weights_out.iter_mut() {
        *w *= a.routed_scaling_factor;
    }
    Ok((indices, weights_out))
}

fn sparse_attend(
    a: &GlmArch,
    pool: &ActPool,
    cache: &LayerGpuCache,
    pos: usize,
    topk: &[usize],
    qk: usize,
) -> Result<Vec<f32>> {
    let n_keys = pos + 1;
    let keys = unsafe {
        std::slice::from_raw_parts(
            cache.keys.contents() as *const f32,
            n_keys * a.n_heads * qk,
        )
    };
    let values = unsafe {
        std::slice::from_raw_parts(
            cache.values.contents() as *const f32,
            n_keys * a.n_heads * a.v_head_dim,
        )
    };
    let queries = read_f32(&pool.queries, a.n_heads * qk);
    let mut allow = vec![false; n_keys];
    for &t in topk {
        if t <= pos && t < n_keys {
            allow[t] = true;
        }
    }
    let scale = (qk as f32).powf(-0.5);
    let mut context = vec![0f32; a.n_heads * a.v_head_dim];
    let mut scores = vec![f32::NEG_INFINITY; n_keys];
    for head in 0..a.n_heads {
        let qh = &queries[head * qk..(head + 1) * qk];
        let mut best = f32::NEG_INFINITY;
        for t in 0..n_keys {
            if !allow[t] {
                scores[t] = f32::NEG_INFINITY;
                continue;
            }
            let off = (t * a.n_heads + head) * qk;
            let dot: f32 = qh
                .iter()
                .zip(&keys[off..off + qk])
                .map(|(x, y)| x * y)
                .sum();
            scores[t] = dot * scale;
            best = best.max(scores[t]);
        }
        let mut total = 0f32;
        for s in scores.iter_mut() {
            *s = if s.is_finite() {
                (*s - best).exp()
            } else {
                0.0
            };
            total += *s;
        }
        let out = &mut context[head * a.v_head_dim..(head + 1) * a.v_head_dim];
        for (t, &prob) in scores.iter().enumerate() {
            if prob == 0.0 {
                continue;
            }
            let w = prob / total;
            let off = (t * a.n_heads + head) * a.v_head_dim;
            for (o, v) in out.iter_mut().zip(&values[off..off + a.v_head_dim]) {
                *o += w * v;
            }
        }
    }
    Ok(context)
}

#[allow(clippy::too_many_arguments)]
fn indexer_topk<'a>(
    weights: &GpuWeightCache,
    arch: &GlmArch,
    attn_p: &str,
    pool: &ActPool,
    cache: &LayerGpuCache,
    pos: usize,
    cos: &[f32],
    sin: &[f32],
    tcb: &mut Option<TokenCommandBuffer<'a>>,
    ctx: &'a MetalContext,
    waits: &Cell<u64>,
) -> Result<Vec<usize>> {
    let a = arch;
    let (ih, idim, rot) = (a.index_n_heads, a.index_head_dim, a.qk_rope_head_dim);
    let idx = format!("{attn_p}.indexer");

    matvec_into(
        tcb,
        ctx,
        weights,
        &format!("{idx}.wq_b.weight"),
        &pool.q_resid,
        a.q_lora_rank,
        &pool.idx_q,
    )?;
    matvec_into(
        tcb,
        ctx,
        weights,
        &format!("{idx}.wk.weight"),
        &pool.h,
        a.hidden,
        &pool.idx_k_raw,
    )?;
    commit(tcb.take(), waits)?;

    let k_raw = read_f32(&pool.idx_k_raw, idim);
    let kw = weights.dense(&format!("{idx}.k_norm.weight"))?;
    let kb = weights.dense(&format!("{idx}.k_norm.bias"))?;
    let k = {
        let n = k_raw.len() as f32;
        let mean = k_raw.iter().sum::<f32>() / n;
        let var = k_raw.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = 1.0 / (var + 1e-6).sqrt();
        (0..k_raw.len())
            .map(|i| (k_raw[i] - mean) * inv * kw[i] + kb[i])
            .collect::<Vec<_>>()
    };
    let mut k_full = rope_interleaved(&k[..rot], cos, sin);
    k_full.extend_from_slice(&k[rot..]);
    unsafe {
        std::ptr::copy_nonoverlapping(
            k_full.as_ptr(),
            (cache.index_keys.contents() as *mut f32).add(pos * idim),
            idim,
        );
    }

    let q = read_f32(&pool.idx_q, ih * idim);
    let mut q_full = vec![0f32; ih * idim];
    for h in 0..ih {
        let src = &q[h * idim..(h + 1) * idim];
        let rotated = rope_interleaved(&src[..rot], cos, sin);
        q_full[h * idim..h * idim + rot].copy_from_slice(&rotated);
        q_full[h * idim + rot..(h + 1) * idim].copy_from_slice(&src[rot..]);
    }

    matvec_into(
        tcb,
        ctx,
        weights,
        &format!("{idx}.weights_proj.weight"),
        &pool.h,
        a.hidden,
        &pool.idx_head_w,
    )?;
    commit(tcb.take(), waits)?;
    let head_scale = (ih as f32).powf(-0.5);
    let mut head_weights = read_f32(&pool.idx_head_w, ih);
    for w in head_weights.iter_mut() {
        *w *= head_scale;
    }

    let n_keys = pos + 1;
    let dim_scale = (idim as f32).powf(-0.5);
    let index_keys = unsafe {
        std::slice::from_raw_parts(cache.index_keys.contents() as *const f32, n_keys * idim)
    };
    let mut index_scores = vec![0f32; n_keys];
    for (t, score) in index_scores.iter_mut().enumerate() {
        let key = &index_keys[t * idim..(t + 1) * idim];
        let mut acc = 0f32;
        for h in 0..ih {
            let qh = &q_full[h * idim..(h + 1) * idim];
            let dot: f32 = qh.iter().zip(key).map(|(x, y)| x * y).sum();
            acc += head_weights[h] * (dot * dim_scale).max(0.0);
        }
        *score = acc;
    }
    for (t, score) in index_scores.iter_mut().enumerate() {
        if t > pos {
            *score = f32::NEG_INFINITY;
        }
    }
    write_f32(&pool.idx_scores, &index_scores);
    Ok(topk_desc(&index_scores, a.index_topk.min(n_keys)))
}

#[allow(clippy::too_many_arguments)]
fn mlp_one<'a>(
    weights: &GpuWeightCache,
    prefix: &str,
    x: &Buffer,
    x_len: usize,
    pool: &ActPool,
    tcb: &mut Option<TokenCommandBuffer<'a>>,
    ctx: &'a MetalContext,
    waits: &Cell<u64>,
) -> Result<Vec<f32>> {
    // Expert-wave: one CB for dense MLP. Default path below is unchanged.
    if gpu_expert_wave_enabled() {
        return moe_device_wave(
            weights,
            &[prefix.to_string()],
            &[1.0f32],
            x,
            x_len,
            tcb,
            ctx,
            waits,
        );
    }
    let mut outs = batched_mlp(
        weights,
        &[prefix.to_string()],
        x,
        x_len,
        pool,
        tcb,
        ctx,
        waits,
    )?;
    outs.pop().ok_or_else(|| Error::Gravity("mlp_one empty".into()))
}

/// Gate/up/down co-issued across all prefixes via `matvec_batch` — three waits
/// total for the whole expert set (matches host `batched_mlp`). The residual
/// `x` and KV stay on device; per-expert gate/up/act vectors are ephemeral
/// because each down_proj takes a different input.
///
/// **Default resident path. Do not edit for expert-wave.** The flagged collapse
/// lives in [`moe_device_wave`]. Changing this function is a Parity V2.1 item 6
/// regression.
#[allow(clippy::too_many_arguments)]
fn batched_mlp<'a>(
    weights: &GpuWeightCache,
    prefixes: &[String],
    x: &Buffer,
    x_len: usize,
    _pool: &ActPool,
    tcb: &mut Option<TokenCommandBuffer<'a>>,
    _ctx: &'a MetalContext,
    waits: &Cell<u64>,
) -> Result<Vec<Vec<f32>>> {
    if prefixes.is_empty() {
        return Ok(Vec::new());
    }
    // Flush any pending attention/router encodes before the batch path, which
    // commits on its own.
    commit(tcb.take(), waits)?;
    let x_host = read_f32(x, x_len);
    let gate_names: Vec<String> = prefixes
        .iter()
        .map(|p| format!("{p}.gate_proj.weight"))
        .collect();
    let up_names: Vec<String> = prefixes
        .iter()
        .map(|p| format!("{p}.up_proj.weight"))
        .collect();
    let gate_calls: Vec<(&str, &[f32])> = gate_names
        .iter()
        .map(|n| (n.as_str(), x_host.as_slice()))
        .collect();
    let up_calls: Vec<(&str, &[f32])> = up_names
        .iter()
        .map(|n| (n.as_str(), x_host.as_slice()))
        .collect();
    let gate_outs = weights.matvec_batch(&gate_calls)?;
    waits.set(waits.get().saturating_add(1));
    let up_outs = weights.matvec_batch(&up_calls)?;
    waits.set(waits.get().saturating_add(1));
    let acts: Vec<Vec<f32>> = gate_outs
        .iter()
        .zip(&up_outs)
        .map(|(g, u)| {
            g.iter()
                .zip(u)
                .map(|(gv, uv)| (gv / (1.0 + (-gv).exp())) * uv)
                .collect()
        })
        .collect();
    let down_names: Vec<String> = prefixes
        .iter()
        .map(|p| format!("{p}.down_proj.weight"))
        .collect();
    let down_calls: Vec<(&str, &[f32])> = down_names
        .iter()
        .zip(&acts)
        .map(|(n, a)| (n.as_str(), a.as_slice()))
        .collect();
    let downs = weights.matvec_batch(&down_calls)?;
    waits.set(waits.get().saturating_add(1));
    Ok(downs)
}

// ── Expert-wave (flagged; default path above is untouched) ─────────────────

fn encode_silu_mul_f32(
    tcb: &mut TokenCommandBuffer<'_>,
    gate: &Buffer,
    up: &Buffer,
    out: &Buffer,
    n: u32,
) -> Result<()> {
    const TG: u32 = 256;
    let n_u = n;
    let g = gate.clone();
    let u = up.clone();
    let o = out.clone();
    tcb.dispatch_threads(
        "gravity_silu_mul_f32",
        (n.div_ceil(TG) * TG, 1, 1),
        (TG, 1, 1),
        move |enc| {
            enc.set_buffer(0, Some(&g), 0);
            enc.set_buffer(1, Some(&u), 0);
            enc.set_buffer(2, Some(&o), 0);
            enc.set_bytes(3, 4, &n_u as *const u32 as *const _);
        },
    )
}

fn encode_axpy_f32(
    tcb: &mut TokenCommandBuffer<'_>,
    y: &Buffer,
    x: &Buffer,
    scale: f32,
    n: u32,
) -> Result<()> {
    const TG: u32 = 256;
    let s = scale;
    let n_u = n;
    let yb = y.clone();
    let xb = x.clone();
    tcb.dispatch_threads(
        "gravity_axpy_f32",
        (n.div_ceil(TG) * TG, 1, 1),
        (TG, 1, 1),
        move |enc| {
            enc.set_buffer(0, Some(&yb), 0);
            enc.set_buffer(1, Some(&xb), 0);
            enc.set_bytes(2, 4, &s as *const f32 as *const _);
            enc.set_bytes(3, 4, &n_u as *const u32 as *const _);
        },
    )
}

fn encode_pq_matvec_device(
    tcb: &mut TokenCommandBuffer<'_>,
    codebooks: &Buffer,
    codes: &Buffer,
    params: crate::gravity_glm::gpu::PqParams,
    x: &Buffer,
    y: &Buffer,
) -> Result<()> {
    const TG: u32 = 256;
    let n_tg = params.rows.div_ceil(8);
    let p = params;
    let cb = codebooks.clone();
    let co = codes.clone();
    let xb = x.clone();
    let yb = y.clone();
    tcb.dispatch_threads(
        "gravity_pq_matvec",
        (n_tg * TG, 1, 1),
        (TG, 1, 1),
        move |enc| {
            enc.set_buffer(0, Some(&cb), 0);
            enc.set_buffer(1, Some(&co), 0);
            enc.set_buffer(2, Some(&xb), 0);
            enc.set_buffer(3, Some(&yb), 0);
            enc.set_bytes(
                4,
                std::mem::size_of_val(&p) as u64,
                &p as *const _ as *const _,
            );
        },
    )
}

/// Encode one weight matvec (device x → device y) into an open command buffer.
/// Host-native weights are applied immediately into `y` (no encode).
fn encode_weight_matvec(
    tcb: &mut TokenCommandBuffer<'_>,
    weights: &GpuWeightCache,
    name: &str,
    x: &Buffer,
    x_len: usize,
    y: &Buffer,
) -> Result<()> {
    crate::cost_ledger::record_matvec_call();
    let mut cache = weights.cache.lock().expect("gpu weight cache");
    weights.ensure_many_locked(&mut cache, &[name])?;
    match cache.get(name).expect("ensured") {
        GpuTensor::NativeCpu(w) => {
            let x_host = read_f32(x, x_len);
            let y_host = matvec_dense(w, &x_host, name)?;
            write_f32(y, &y_host);
            Ok(())
        }
        GpuTensor::NativeGpuBf16 { buf, rows, cols } => {
            if x_len != *cols as usize {
                return Err(Error::Gravity(format!(
                    "expert-wave matvec {name}: x_len {x_len} != cols {cols}"
                )));
            }
            crate::cost_ledger::record_active_bytes_for(name, buf.length());
            encode_gemv_native_bf16_seq(tcb, buf, *rows, *cols, x, y)
        }
        GpuTensor::Pq {
            codebooks,
            codes,
            params,
        } => {
            if x_len != params.cols as usize {
                return Err(Error::Gravity(format!(
                    "expert-wave matvec {name}: x_len {x_len} != cols {}",
                    params.cols
                )));
            }
            crate::cost_ledger::record_active_bytes_for(
                name,
                codebooks.length() + codes.length(),
            );
            encode_pq_matvec_device(tcb, codebooks, codes, *params, x, y)
        }
    }
}

/// Isolated device path: **gate + up → SiLU → down → weighted combine** in one
/// command buffer (one `commit_and_wait`). Flagged via
/// [`crate::gravity_glm::GPU_EXPERT_WAVE_ENV`]; never called from the default
/// resident path.
///
/// `scales[i]` multiplies prefix `i`'s down projection into the sum (MoE router
/// weights for routed experts, `1.0` for shared / dense). Accumulation order
/// matches the host: prefixes are already sorted ascending-expert then shared.
///
/// Requires every gate/up/down weight to be device-resident (`Pq` or
/// `NativeGpuBf16`). Host-native tensors fall back to a single host pass with
/// one wait tick (tiny fixtures without `HAWKING_GLM_GPU_LM_HEAD`); the pure
/// device path is what flagship PQ experts hit.
#[allow(clippy::too_many_arguments)]
fn moe_device_wave<'a>(
    weights: &GpuWeightCache,
    prefixes: &[String],
    scales: &[f32],
    x: &Buffer,
    x_len: usize,
    tcb: &mut Option<TokenCommandBuffer<'a>>,
    ctx: &'a MetalContext,
    waits: &Cell<u64>,
) -> Result<Vec<f32>> {
    if prefixes.is_empty() {
        return Ok(vec![0f32; x_len]);
    }
    if scales.len() != prefixes.len() {
        return Err(Error::Gravity(format!(
            "expert-wave: scales.len() {} != prefixes.len() {}",
            scales.len(),
            prefixes.len()
        )));
    }
    // Flush pending attention encodes; this path owns the next commit.
    commit(tcb.take(), waits)?;

    // Pin every projection for this layer before encoding so LRU cannot drop
    // a tensor mid-wave (same invariant as matvec_batch).
    let mut all_names: Vec<String> = Vec::with_capacity(prefixes.len() * 3);
    for p in prefixes {
        all_names.push(format!("{p}.gate_proj.weight"));
        all_names.push(format!("{p}.up_proj.weight"));
        all_names.push(format!("{p}.down_proj.weight"));
    }
    {
        let name_refs: Vec<&str> = all_names.iter().map(String::as_str).collect();
        let mut cache = weights.cache.lock().expect("gpu weight cache");
        weights.ensure_many_locked(&mut cache, &name_refs)?;
    }

    // Device-resident weights only on the pure CB path. Host-native needs the
    // host fallback (cannot encode silu→down dependence without a wait).
    let all_device = {
        let cache = weights.cache.lock().expect("gpu weight cache");
        all_names.iter().all(|n| {
            matches!(
                cache.get(n),
                Some(GpuTensor::Pq { .. }) | Some(GpuTensor::NativeGpuBf16 { .. })
            )
        })
    };
    if !all_device {
        return moe_wave_host_fallback(weights, prefixes, scales, x, x_len, waits);
    }

    // Discover intermediate width from the first gate projection.
    let inter = {
        let cache = weights.cache.lock().expect("gpu weight cache");
        let gname = format!("{}.gate_proj.weight", prefixes[0]);
        match cache.get(&gname).expect("ensured gate") {
            GpuTensor::Pq { params, .. } => params.rows as usize,
            GpuTensor::NativeGpuBf16 { rows, .. } => *rows as usize,
            GpuTensor::NativeCpu(_) => {
                return Err(Error::Gravity(
                    "expert-wave: gate is NativeCpu after device check".into(),
                ));
            }
        }
    };

    let n_exp = prefixes.len();
    let mut gate_bufs = Vec::with_capacity(n_exp);
    let mut up_bufs = Vec::with_capacity(n_exp);
    let mut act_bufs = Vec::with_capacity(n_exp);
    let mut down_bufs = Vec::with_capacity(n_exp);
    for _ in 0..n_exp {
        gate_bufs.push(ctx.new_buffer_checked(inter * 4)?);
        up_bufs.push(ctx.new_buffer_checked(inter * 4)?);
        act_bufs.push(ctx.new_buffer_checked(inter * 4)?);
        down_bufs.push(ctx.new_buffer_checked(x_len * 4)?);
    }
    let combined = ctx.new_buffer_checked(x_len * 4)?;
    // Host-zero the accumulator so device axpy starts from 0 (Metal shared).
    write_f32(&combined, &vec![0f32; x_len]);

    let mut wave = TokenCommandBuffer::new(ctx);
    for (i, p) in prefixes.iter().enumerate() {
        encode_weight_matvec(
            &mut wave,
            weights,
            &format!("{p}.gate_proj.weight"),
            x,
            x_len,
            &gate_bufs[i],
        )?;
        encode_weight_matvec(
            &mut wave,
            weights,
            &format!("{p}.up_proj.weight"),
            x,
            x_len,
            &up_bufs[i],
        )?;
    }
    for i in 0..n_exp {
        encode_silu_mul_f32(
            &mut wave,
            &gate_bufs[i],
            &up_bufs[i],
            &act_bufs[i],
            inter as u32,
        )?;
    }
    for (i, p) in prefixes.iter().enumerate() {
        encode_weight_matvec(
            &mut wave,
            weights,
            &format!("{p}.down_proj.weight"),
            &act_bufs[i],
            inter,
            &down_bufs[i],
        )?;
    }
    // Weighted combine in prefix order (associativity matches host).
    for i in 0..n_exp {
        encode_axpy_f32(
            &mut wave,
            &combined,
            &down_bufs[i],
            scales[i],
            x_len as u32,
        )?;
    }

    crate::cost_ledger::record_dispatches(wave.dispatch_count() as u64);
    wave.commit_and_wait()?;
    waits.set(waits.get().saturating_add(1));

    let out = read_f32(&combined, x_len);
    crate::cost_ledger::record_transfer((x_len * 4) as u64, false, "expert_wave_y_download");
    Ok(out)
}

/// Host-side gate/up/SiLU/down/combine used when expert weights are not
/// device-resident. Still bills **one** wait for the wave accounting contract
/// (flag on ⇒ collapsed MLP drain count); no intermediate readback loop.
fn moe_wave_host_fallback(
    weights: &GpuWeightCache,
    prefixes: &[String],
    scales: &[f32],
    x: &Buffer,
    x_len: usize,
    waits: &Cell<u64>,
) -> Result<Vec<f32>> {
    let x_host = read_f32(x, x_len);
    let mut combined = vec![0f32; x_len];
    for (p, &scale) in prefixes.iter().zip(scales.iter()) {
        let gate = weights.matvec(&format!("{p}.gate_proj.weight"), &x_host)?;
        let up = weights.matvec(&format!("{p}.up_proj.weight"), &x_host)?;
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let down = weights.matvec(&format!("{p}.down_proj.weight"), &act)?;
        for (c, d) in combined.iter_mut().zip(&down) {
            *c += *d * scale;
        }
    }
    // One wait tick: collapsed accounting for the flag-on path.
    waits.set(waits.get().saturating_add(1));
    Ok(combined)
}

/// Holds the long-lived resident state for a [`crate::gravity_glm::gpu::GravityGlmGpu`].
pub struct ResidentRuntime {
    pub session: Mutex<ResidentSession>,
    pub pool: ActPool,
}

impl ResidentRuntime {
    pub fn new(ctx: &MetalContext, arch: &GlmArch) -> Result<Self> {
        Ok(Self {
            session: Mutex::new(ResidentSession::new(ctx, arch, 64)?),
            pool: ActPool::new(ctx, arch)?,
        })
    }
}

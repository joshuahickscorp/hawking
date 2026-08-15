//! Layer-major BF16 SOURCE forward + activation capture for Qwen3-Coder-Next (Q80).
//!
//! Resource contract:
//! * Does **not** resident-load the ~148 GiB BF16 source.
//! * Generation and capture both stream one layer's weights at a time via
//!   safetensors range-reads, then free them.
//! * Capture inverts the loop (layer-major): load layer N, push ALL corpus
//!   tokens through it, write routes + retained hiddens, free that layer's
//!   retained hidden rows, then free the layer weights. Only one layer's
//!   hidden payloads are resident at a time; route membership stays complete.
//!
//! Operator semantics are **reused** from `qwen80_complete_runtime`:
//! residual RMSNorm `(1+w)`, Gated DeltaNet recurrence, GQA q/k norm + partial
//! RoPE + causal attention + sigmoid gate, top-10 `norm_topk_prob` router,
//! SwiGLU experts, shared expert + sigmoid gate combine. Only the weight
//! backend differs (BF16 GEMV vs packed complete-binary).

use crate::artifact::widen_native;
use crate::kernels::{add_inplace, argmax_f32, silu_mul};
use crate::model::qwen80_complete_runtime::{
    qwen80_gqa_apply_sigmoid_gate, qwen80_gqa_causal_attention,
    qwen80_gqa_query_from_interleaved_q_projection, qwen80_gqa_source_norm_rope,
    source_qwen80_ba_to_decay_beta, source_qwen80_causal_conv_step_dense,
    source_qwen80_gated_rms_norm, source_qwen80_l2_normalize, source_qwen80_recurrent_deltanet,
    source_qwen80_residual_rms_norm, source_qwen80_split_linear_qkvz, source_qwen80_topk_router,
    Qwen80CanonicalGqaLayout, Qwen80CanonicalLinearDeltaNetLayout,
};
use crate::{Error, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
#[cfg(not(unix))]
use std::io::{Seek, SeekFrom};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

pub const QWEN80_LAYERS: usize = 48;
pub const QWEN80_HIDDEN: usize = 2048;
pub const QWEN80_FULL_ATTN_HEADS: usize = 16;
pub const QWEN80_FULL_ATTN_KV_HEADS: usize = 2;
pub const QWEN80_FULL_ATTN_HEAD_DIM: usize = 256;
pub const QWEN80_EXPERTS: usize = 512;
pub const QWEN80_TOP_K: usize = 10;
pub const QWEN80_MOE_INTERMEDIATE: usize = 512;
pub const QWEN80_SHARED_EXPERT_INTERMEDIATE: usize = 512;
pub const QWEN80_VOCAB: usize = 151_936;
pub const QWEN80_TOKENIZER_VOCAB: usize = 151_669;
pub const QWEN80_FULL_ATTENTION_INTERVAL: usize = 4;
pub const QWEN80_RMS_EPS: f32 = 1.0e-6;
/// Soft upper bound: single-digit GiB. Approaching 148 GiB means resident load.
pub const STREAMED_PEAK_RSS_HARD_CAP_BYTES: u64 = 16 * 1024 * 1024 * 1024;
/// Contract estimate for per-layer expert BF16 payload (~3 GiB).
pub const PER_LAYER_EXPERT_BF16_BYTES: u64 = 3 * 1024 * 1024 * 1024;

/// Default retained router-input rows per (layer, expert) under first-N retention.
///
/// Chosen so organs see enough fit rows to pass the null test (failure is sharp
/// below ~16 rows; flattens above ~32). At N=64 × 512 experts the worst-case
/// unique rows/layer is 32768 (≈256 MiB of f32@2048). Retained hidden payloads
/// are flushed and freed at the end of each layer, so only that per-layer
/// footprint is resident — not × [`QWEN80_LAYERS`].
pub const DEFAULT_MAX_HIDDEN_TOKENS_PER_EXPERT: usize = 64;

/// Worst-case unique retained rows at one layer under first-N (no multi-route credit).
#[inline]
pub fn worst_case_unique_rows_per_layer(max_hidden_tokens_per_expert: usize) -> usize {
    max_hidden_tokens_per_expert.saturating_mul(QWEN80_EXPERTS)
}

/// Worst-case resident retained-hidden bytes at one layer (`f32` × hidden).
///
/// Per-layer flush means this is the capture RAM budget: it is **not**
/// multiplied by [`QWEN80_LAYERS`].
#[inline]
pub fn worst_case_retained_hidden_bytes_per_layer(max_hidden_tokens_per_expert: usize) -> usize {
    worst_case_unique_rows_per_layer(max_hidden_tokens_per_expert)
        .saturating_mul(QWEN80_HIDDEN)
        .saturating_mul(4)
}

/// `true` iff first-N = `n` fits under [`STREAMED_PEAK_RSS_HARD_CAP_BYTES`].
///
/// `n == 0` is refused (the CLI requires a positive per-expert quota).
#[inline]
pub fn max_hidden_tokens_per_expert_within_streamed_cap(n: usize) -> bool {
    n > 0
        && worst_case_retained_hidden_bytes_per_layer(n)
            <= STREAMED_PEAK_RSS_HARD_CAP_BYTES as usize
}

/// Stderr progress line printed by the layer-major capture driver.
pub fn format_capture_progress(
    probe_count: usize,
    total_tokens: usize,
    max_hidden_tokens_per_expert: usize,
    source_tensor_count: usize,
) -> String {
    let worst = worst_case_unique_rows_per_layer(max_hidden_tokens_per_expert).min(total_tokens);
    let mib = (worst.saturating_mul(QWEN80_HIDDEN).saturating_mul(4) as f64) / (1024.0 * 1024.0);
    format!(
        "capture: {probe_count} probes, {total_tokens} tokens, per-expert first-N={max_hidden_tokens_per_expert} (worst-case {worst} rows/layer ≈{mib:.1} MiB f32@2048); source tensors={source_tensor_count}"
    )
}

/// On-disk relative path for one retained hidden row. Consumers resolve this
/// against the capture output directory (`hidden/L{{layer}}/{{probe}}/{{pos}}.f32le`).
pub fn retained_hidden_relative_path(layer: usize, probe_id: &str, position: usize) -> String {
    format!("hidden/L{layer:02}/{probe_id}/{position:06}.f32le")
}

/// Write one retained hidden row as little-endian f32. Refuses to overwrite.
pub fn write_retained_hidden_f32le(path: &Path, values: &[f32]) -> Result<(String, usize)> {
    let parent = path.parent().ok_or_else(|| {
        model_err(format!(
            "hidden capture path has no parent: {}",
            path.display()
        ))
    })?;
    std::fs::create_dir_all(parent).map_err(|e| {
        model_err(format!(
            "cannot create hidden capture directory {}: {e}",
            parent.display()
        ))
    })?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|e| model_err(format!("cannot create hidden {}: {e}", path.display())))?;
    let mut digest = Sha256::new();
    for value in values {
        let bytes = value.to_le_bytes();
        file.write_all(&bytes)
            .map_err(|e| model_err(format!("cannot write hidden {}: {e}", path.display())))?;
        digest.update(bytes);
    }
    file.flush()
        .map_err(|e| model_err(format!("cannot flush hidden {}: {e}", path.display())))?;
    Ok((format!("{:x}", digest.finalize()), values.len() * 4))
}

/// Deterministic per-expert first-N retention for one token.
///
/// Walk tokens in global order. For each top-k expert that still has fewer than
/// `max_per_expert` retained members, credit this token to that expert. Retain
/// the token's router-input hidden if any expert still needed a slot.
///
/// This is deliberately not a random reservoir: the same corpus + same N yields
/// byte-identical retention across runs. Route membership is independent and
/// stays complete for every token.
pub fn credit_expert_first_n_retention(
    expert_retained: &mut [usize],
    selected_expert_ids: &[u32],
    max_per_expert: usize,
) -> bool {
    if max_per_expert == 0 || selected_expert_ids.is_empty() {
        return false;
    }
    let mut retain = false;
    for &expert in selected_expert_ids {
        let e = expert as usize;
        if e >= expert_retained.len() {
            continue;
        }
        if expert_retained[e] < max_per_expert {
            expert_retained[e] += 1;
            retain = true;
        }
    }
    retain
}

/// Bytes of retained hidden payloads currently sitting in `captures`.
///
/// After a per-layer flush this is the in-memory footprint of one layer (or
/// zero once that layer has been released). It must not grow with layer index.
pub fn resident_retained_hidden_bytes(captures: &[Vec<Vec<LayerTokenCapture>>]) -> usize {
    captures
        .iter()
        .flatten()
        .flatten()
        .map(|cap| cap.router_input_hidden.len().saturating_mul(4))
        .sum()
}

/// Drop retained hidden payloads for `layer_idx`. Route membership is left
/// intact; [`LayerTokenCapture::hidden_retained`] still records whether a row
/// was kept. Returns the number of `f32` elements freed.
pub fn release_layer_retained_hiddens(
    captures: &mut [Vec<Vec<LayerTokenCapture>>],
    layer_idx: usize,
) -> usize {
    let mut freed = 0usize;
    for probe in captures.iter_mut() {
        for token in probe.iter_mut() {
            for cap in token.iter_mut() {
                if cap.layer == layer_idx {
                    freed = freed.saturating_add(cap.router_input_hidden.len());
                    cap.router_input_hidden = Vec::new();
                }
            }
        }
    }
    freed
}

/// Apply per-expert first-N retention and append one layer of captures.
///
/// `all_router_in` is `[token, hidden]` in the same global order as
/// `token_index` / `routes`. Expert-retained counters reset each call (each
/// layer). Does not touch residual streams.
pub fn append_retained_layer_captures(
    captures: &mut [Vec<Vec<LayerTokenCapture>>],
    token_index: &[(usize, usize)],
    routes: &mut [(Vec<u32>, Vec<f32>)],
    all_router_in: &[f32],
    layer_idx: usize,
    max_hidden_tokens_per_expert: usize,
) -> Result<()> {
    if all_router_in.len() != token_index.len().saturating_mul(QWEN80_HIDDEN) {
        return Err(model_err("router-input/token_index length mismatch"));
    }
    if routes.len() != token_index.len() {
        return Err(model_err("routes/token_index length mismatch"));
    }
    let mut expert_retained = vec![0usize; QWEN80_EXPERTS];
    for (t, &(pi, pos)) in token_index.iter().enumerate() {
        if pi >= captures.len() || pos >= captures[pi].len() {
            return Err(model_err(format!(
                "token_index ({pi},{pos}) out of capture bounds"
            )));
        }
        let (ids, weights) = std::mem::take(&mut routes[t]);
        let retain = credit_expert_first_n_retention(
            &mut expert_retained,
            &ids,
            max_hidden_tokens_per_expert,
        );
        let hidden = if retain {
            all_router_in[t * QWEN80_HIDDEN..(t + 1) * QWEN80_HIDDEN].to_vec()
        } else {
            Vec::new()
        };
        captures[pi][pos].push(LayerTokenCapture {
            layer: layer_idx,
            selected_expert_ids: ids,
            normalized_route_weights: weights,
            router_input_hidden: hidden,
            hidden_retained: retain,
        });
    }
    Ok(())
}

/// Invoke `on_flush` (write this layer's retained rows) then drop those
/// payloads so they are not resident when the next layer loads.
pub fn flush_and_release_layer_hiddens<F>(
    captures: &mut [Vec<Vec<LayerTokenCapture>>],
    layer_idx: usize,
    on_flush: Option<&mut F>,
) -> Result<usize>
where
    F: FnMut(usize, &mut [Vec<Vec<LayerTokenCapture>>]) -> Result<()>,
{
    if let Some(cb) = on_flush {
        cb(layer_idx, captures)?;
    }
    Ok(release_layer_retained_hiddens(captures, layer_idx))
}

fn model_err(msg: impl Into<String>) -> Error {
    Error::Model(msg.into())
}

#[inline]
fn bf16_le_to_f32(bytes: &[u8]) -> f32 {
    debug_assert!(bytes.len() >= 2);
    f32::from_bits((u16::from_le_bytes([bytes[0], bytes[1]]) as u32) << 16)
}

/// BF16-LE → f32 widen of a row-major `[rows, cols]` matrix into a fresh buffer.
pub fn widen_bf16_mat(weight_le: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
    let n = rows
        .checked_mul(cols)
        .ok_or_else(|| model_err("widen_bf16_mat size overflow"))?;
    let mut out = vec![0.0f32; n];
    widen_bf16_into(weight_le, rows, cols, &mut out)?;
    Ok(out)
}

/// BF16-LE → f32 widen into a caller-owned buffer (avoids per-call alloc when reused).
pub fn widen_bf16_into(weight_le: &[u8], rows: usize, cols: usize, out: &mut [f32]) -> Result<()> {
    let n = rows
        .checked_mul(cols)
        .ok_or_else(|| model_err("widen_bf16_into size overflow"))?;
    let expect = n
        .checked_mul(2)
        .ok_or_else(|| model_err("widen_bf16_into byte overflow"))?;
    if weight_le.len() < expect {
        return Err(model_err(format!(
            "widen_bf16_into weight bytes {} < {expect}",
            weight_le.len()
        )));
    }
    if out.len() < n {
        return Err(model_err(format!(
            "widen_bf16_into out len {} < {n}",
            out.len()
        )));
    }
    // Memory-bandwidth bound. Parallelise only for large tensors; avoid
    // thread-spawn storms when widening many experts × 3 mats per layer.
    const PARALLEL_THRESHOLD: usize = 512 * 1024; // elements
    if n < PARALLEL_THRESHOLD {
        for i in 0..n {
            let b = i * 2;
            out[i] = bf16_le_to_f32(&weight_le[b..b + 2]);
        }
        return Ok(());
    }
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(4)
        .clamp(1, 8);
    let chunk = n.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (t, out_chunk) in out[..n].chunks_mut(chunk).enumerate() {
            let base = t * chunk;
            let w = weight_le;
            scope.spawn(move || {
                for (i, o) in out_chunk.iter_mut().enumerate() {
                    let idx = base + i;
                    let b = idx * 2;
                    *o = bf16_le_to_f32(&w[b..b + 2]);
                }
            });
        }
    });
    Ok(())
}

/// One expert's gate/up/down for its member tokens.
///
/// Widens BF16 → f32 once, then runs **per-token** `gemm_f32` (n_batch=1).
/// Multi-token sgemm (M>1) can use a different Accelerate micro-kernel than
/// M=1 and reassociate enough to flip borderline top-k after ~30 layers; the
/// scalar capture path uses M=1 via `gemv_bf16`→`gemm_bf16`, so we match that.
/// Amortised widen still removes the on-the-fly BF16 tax; parallel experts
/// supply the throughput.
///
/// Writes unweighted down outputs into `expert_down_out` (n × h). Residual
/// scatter stays with the caller (route-slot order).
fn expert_batched_down(
    expert: &ExpertWeights,
    members: &[(usize, f32)],
    all_router_in: &[f32],
    h: usize,
    inter: usize,
    expert_down_out: &mut [f32],
    _x_g: &mut [f32],
    gu_out: &mut [f32],
    act: &mut [f32],
    w_gu: &mut [f32],
    w_down: &mut [f32],
) -> Result<()> {
    let n = members.len();
    if n == 0 {
        return Ok(());
    }
    if expert_down_out.len() < n * h {
        return Err(model_err("expert_batched_down output too small"));
    }

    // Widen once; run gate/up/down as separate M=1 matvecs (matches scalar
    // swiglu_mlp_bf16 → gemv_bf16 → gemm_bf16 geometry exactly).
    let (w_gate, w_up) = w_gu[..2 * inter * h].split_at_mut(inter * h);
    widen_bf16_into(&expert.gate, inter, h, w_gate)?;
    widen_bf16_into(&expert.up, inter, h, w_up)?;
    let w_down = &mut w_down[..h * inter];
    widen_bf16_into(&expert.down, h, inter, w_down)?;

    let (gate_scratch, up_scratch) = gu_out[..2 * inter].split_at_mut(inter);
    let act_scratch = &mut act[..inter];
    for (i, &(t, _)) in members.iter().enumerate() {
        let x = &all_router_in[t * h..(t + 1) * h];
        gemm_f32(w_gate, inter, h, x, 1, gate_scratch)?;
        gemm_f32(w_up, inter, h, x, 1, up_scratch)?;
        silu_mul(gate_scratch, up_scratch, act_scratch);
        gemm_f32(
            w_down,
            h,
            inter,
            act_scratch,
            1,
            &mut expert_down_out[i * h..(i + 1) * h],
        )?;
    }
    Ok(())
}

/// Parallel per-expert MoE over the flat token corpus at one layer (routed experts only).
///
/// Expert GEMMs run in parallel; residual scatter is serial and applies each
/// token's experts in **route top-k slot order** so float accumulation matches
/// scalar `moe_combine` (route membership stays bitwise-stable).
fn moe_routed_experts_parallel(
    layer: &mut LoadedLayer,
    expert_members: &[Vec<(usize, f32)>],
    routes: &[(Vec<u32>, Vec<f32>)],
    all_router_in: &[f32],
    total_tokens: usize,
    h: usize,
    inter: usize,
    moe_out: &mut [f32],
) -> Result<()> {
    // Active experts only.
    let active: Vec<usize> = (0..QWEN80_EXPERTS)
        .filter(|&e| !expert_members[e].is_empty())
        .collect();
    for e in 0..QWEN80_EXPERTS {
        if expert_members[e].is_empty() {
            layer.experts[e].gate = Vec::new();
            layer.experts[e].up = Vec::new();
            layer.experts[e].down = Vec::new();
        }
    }
    if active.is_empty() {
        return Ok(());
    }

    // Per-expert unweighted down outputs (empty for inactive).
    let mut expert_down: Vec<Vec<f32>> = (0..QWEN80_EXPERTS).map(|_| Vec::new()).collect();
    let max_by_mem = if total_tokens > 40_000 {
        2usize
    } else if total_tokens > 4_000 {
        4
    } else {
        8
    };
    let n_workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, max_by_mem)
        .min(active.len());

    let err: Mutex<Option<String>> = Mutex::new(None);
    // Each worker returns (expert_id, down_out) pairs for its shard.
    let mut worker_results: Vec<Vec<(usize, Vec<f32>)>> =
        (0..n_workers).map(|_| Vec::new()).collect();
    {
        let experts: &[ExpertWeights] = &layer.experts;
        std::thread::scope(|scope| {
            let chunk = active.len().div_ceil(n_workers);
            for (wi, result_slot) in worker_results.iter_mut().enumerate() {
                let start = wi * chunk;
                if start >= active.len() {
                    break;
                }
                let end = (start + chunk).min(active.len());
                let my_experts = &active[start..end];
                let expert_members = expert_members;
                let all_router_in = all_router_in;
                let err = &err;
                scope.spawn(move || {
                    let max_n = my_experts
                        .iter()
                        .map(|&e| expert_members[e].len())
                        .max()
                        .unwrap_or(0);
                    let mut x_g = vec![0.0f32; max_n * h];
                    let mut gu_out = vec![0.0f32; max_n * 2 * inter];
                    let mut act = vec![0.0f32; max_n * inter];
                    let mut w_gu = vec![0.0f32; 2 * inter * h];
                    let mut w_down = vec![0.0f32; h * inter];
                    let mut local: Vec<(usize, Vec<f32>)> = Vec::with_capacity(my_experts.len());
                    for &e in my_experts {
                        let n = expert_members[e].len();
                        let mut out = vec![0.0f32; n * h];
                        if let Err(err_e) = expert_batched_down(
                            &experts[e],
                            &expert_members[e],
                            all_router_in,
                            h,
                            inter,
                            &mut out,
                            &mut x_g,
                            &mut gu_out,
                            &mut act,
                            &mut w_gu,
                            &mut w_down,
                        ) {
                            if let Ok(mut g) = err.lock() {
                                *g = Some(err_e.to_string());
                            }
                            return;
                        }
                        local.push((e, out));
                    }
                    *result_slot = local;
                });
            }
        });
    }
    if let Some(msg) = err.into_inner().unwrap_or(None) {
        return Err(model_err(msg));
    }
    for wr in worker_results {
        for (e, out) in wr {
            expert_down[e] = out;
        }
    }

    // local_of[e][token] = row in expert_down[e]
    let mut local_of: Vec<HashMap<usize, usize>> = vec![HashMap::new(); QWEN80_EXPERTS];
    for &e in &active {
        for (local_i, &(t, _)) in expert_members[e].iter().enumerate() {
            local_of[e].insert(t, local_i);
        }
    }

    // Scatter in route-slot order per token (matches scalar moe_combine).
    for t in 0..total_tokens {
        let (ids, weights) = &routes[t];
        let dst = &mut moe_out[t * h..(t + 1) * h];
        for (&eid, &w) in ids.iter().zip(weights.iter()) {
            let e = eid as usize;
            let local_i = *local_of[e]
                .get(&t)
                .ok_or_else(|| model_err(format!("token {t} missing local row for expert {e}")))?;
            let src = &expert_down[e][local_i * h..(local_i + 1) * h];
            for j in 0..h {
                dst[j] += src[j] * w;
            }
        }
    }

    for e in 0..QWEN80_EXPERTS {
        layer.experts[e].gate = Vec::new();
        layer.experts[e].up = Vec::new();
        layer.experts[e].down = Vec::new();
        expert_down[e] = Vec::new();
    }
    Ok(())
}

/// Shared expert SwiGLU over all tokens + per-token sigmoid gate values.
/// Returns `(shared_down [T*h], gate_vals [T])` so the caller can add
/// `shared_down * gate` **after** routed experts (scalar `moe_combine` order).
///
/// Per-token M=1 matvecs after one widen — matches scalar `swiglu_mlp_bf16`.
fn shared_expert_batched(
    layer: &LoadedLayer,
    all_router_in: &[f32],
    total_tokens: usize,
    h: usize,
    inter: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if total_tokens == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let w_gate = widen_bf16_mat(&layer.shared_gate, inter, h)?;
    let w_up = widen_bf16_mat(&layer.shared_up, inter, h)?;
    let w_down = widen_bf16_mat(&layer.shared_down, h, inter)?;
    let w_sg = widen_bf16_mat(&layer.shared_expert_gate, 1, h)?;

    let mut shared_down = vec![0.0f32; total_tokens * h];
    let mut gate_vals = vec![0.0f32; total_tokens];
    let mut gate_buf = vec![0.0f32; inter];
    let mut up_buf = vec![0.0f32; inter];
    let mut act_buf = vec![0.0f32; inter];
    let mut sg_logit = [0.0f32; 1];
    for t in 0..total_tokens {
        let x = &all_router_in[t * h..(t + 1) * h];
        gemm_f32(&w_gate, inter, h, x, 1, &mut gate_buf)?;
        gemm_f32(&w_up, inter, h, x, 1, &mut up_buf)?;
        silu_mul(&gate_buf, &up_buf, &mut act_buf);
        gemm_f32(
            &w_down,
            h,
            inter,
            &act_buf,
            1,
            &mut shared_down[t * h..(t + 1) * h],
        )?;
        gemm_f32(&w_sg, 1, h, x, 1, &mut sg_logit)?;
        let gate_val = 1.0 / (1.0 + (-sg_logit[0]).exp());
        if !gate_val.is_finite() || !(0.0..=1.0).contains(&gate_val) {
            return Err(model_err("shared expert gate sigmoid invalid"));
        }
        gate_vals[t] = gate_val;
    }
    Ok((shared_down, gate_vals))
}

#[cfg(target_os = "macos")]
mod accelerate_gemm {
    const CBLAS_ROW_MAJOR: i32 = 101;
    const CBLAS_NO_TRANS: i32 = 111;
    const CBLAS_TRANS: i32 = 112;

    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        fn cblas_sgemm(
            order: i32,
            transa: i32,
            transb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }

    /// Batched matmul: `out[b] = W @ x[b]` where `W` is row-major `[rows, cols]`.
    pub fn gemm_w_times_x(
        w: &[f32],
        rows: usize,
        cols: usize,
        x: &[f32],
        n_batch: usize,
        out: &mut [f32],
    ) {
        debug_assert_eq!(w.len(), rows * cols);
        debug_assert_eq!(x.len(), n_batch * cols);
        debug_assert_eq!(out.len(), n_batch * rows);
        if n_batch == 0 || rows == 0 || cols == 0 {
            return;
        }
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                n_batch as i32,
                rows as i32,
                cols as i32,
                1.0,
                x.as_ptr(),
                cols as i32,
                w.as_ptr(),
                cols as i32,
                0.0,
                out.as_mut_ptr(),
                rows as i32,
            );
        }
    }

    /// `Out = scores @ V` with scores `[seq, seq]` and V `[seq, head_dim]`.
    pub fn gemm_scores_times_v(
        scores: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        out: &mut [f32],
    ) {
        debug_assert_eq!(scores.len(), seq_len * seq_len);
        debug_assert_eq!(v.len(), seq_len * head_dim);
        debug_assert_eq!(out.len(), seq_len * head_dim);
        if seq_len == 0 || head_dim == 0 {
            return;
        }
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                seq_len as i32,
                head_dim as i32,
                seq_len as i32,
                1.0,
                scores.as_ptr(),
                seq_len as i32,
                v.as_ptr(),
                head_dim as i32,
                0.0,
                out.as_mut_ptr(),
                head_dim as i32,
            );
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod accelerate_gemm {
    pub fn gemm_w_times_x(
        w: &[f32],
        rows: usize,
        cols: usize,
        x: &[f32],
        n_batch: usize,
        out: &mut [f32],
    ) {
        for b in 0..n_batch {
            let xb = &x[b * cols..(b + 1) * cols];
            let ob = &mut out[b * rows..(b + 1) * rows];
            for r in 0..rows {
                let row = &w[r * cols..(r + 1) * cols];
                let mut acc = 0.0f32;
                for c in 0..cols {
                    acc += row[c] * xb[c];
                }
                ob[r] = acc;
            }
        }
    }

    pub fn gemm_scores_times_v(
        scores: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        out: &mut [f32],
    ) {
        for b in 0..seq_len {
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for t in 0..seq_len {
                    acc += scores[b * seq_len + t] * v[t * head_dim + d];
                }
                out[b * head_dim + d] = acc;
            }
        }
    }
}

/// Causal GQA prefill for a full sequence via Accelerate GEMM for QKᵀ and PV.
/// Layouts match `qwen80_gqa_causal_attention`:
///   q:     (seq, query_heads * head_dim)
///   k/v:   (seq, kv_heads * head_dim)
///   out:   (seq, query_heads * head_dim)
fn gqa_prefill_causal(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    layout: &Qwen80CanonicalGqaLayout,
    seq_len: usize,
    out: &mut [f32],
) -> Result<()> {
    if seq_len == 0 {
        return Ok(());
    }
    let n_heads = layout.query_heads;
    let n_kv_heads = layout.key_value_heads;
    let head_dim = layout.head_dim;
    let q_dim = layout.query_dim;
    let kv_dim = layout.kv_dim;
    if q.len() != seq_len * q_dim
        || k.len() != seq_len * kv_dim
        || v.len() != seq_len * kv_dim
        || out.len() != seq_len * q_dim
    {
        return Err(model_err("gqa_prefill_causal geometry mismatch"));
    }
    // Short sequences: decode-step loop avoids scratch alloc and matches the
    // single-token path bit-for-bit (no GEMM reassociation).
    if seq_len <= 32 {
        for pos in 0..seq_len {
            let attn = qwen80_gqa_causal_attention(
                &q[pos * q_dim..(pos + 1) * q_dim],
                &k[..(pos + 1) * kv_dim],
                &v[..(pos + 1) * kv_dim],
                pos + 1,
                layout,
            )?;
            out[pos * q_dim..(pos + 1) * q_dim].copy_from_slice(&attn);
        }
        return Ok(());
    }

    let group_size = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut q_h = vec![0.0f32; seq_len * head_dim];
    let mut k_h = vec![0.0f32; seq_len * head_dim];
    let mut v_h = vec![0.0f32; seq_len * head_dim];
    let mut scores = vec![0.0f32; seq_len * seq_len];
    let mut ctx = vec![0.0f32; seq_len * head_dim];

    for h in 0..n_heads {
        let kv_h = h / group_size;
        for t in 0..seq_len {
            let qs = t * q_dim + h * head_dim;
            let ks = t * kv_dim + kv_h * head_dim;
            q_h[t * head_dim..(t + 1) * head_dim].copy_from_slice(&q[qs..qs + head_dim]);
            k_h[t * head_dim..(t + 1) * head_dim].copy_from_slice(&k[ks..ks + head_dim]);
            v_h[t * head_dim..(t + 1) * head_dim].copy_from_slice(&v[ks..ks + head_dim]);
        }
        // scores[b,t] = k[t] · q[b]
        accelerate_gemm::gemm_w_times_x(&k_h, seq_len, head_dim, &q_h, seq_len, &mut scores);
        for b in 0..seq_len {
            let row = &mut scores[b * seq_len..(b + 1) * seq_len];
            let mut max_v = f32::NEG_INFINITY;
            for t in 0..=b {
                row[t] *= scale;
                if row[t] > max_v {
                    max_v = row[t];
                }
            }
            for t in (b + 1)..seq_len {
                row[t] = 0.0;
            }
            let mut sum = 0.0f32;
            for t in 0..=b {
                let e = (row[t] - max_v).exp();
                row[t] = e;
                sum += e;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for t in 0..=b {
                row[t] *= inv;
            }
            for t in (b + 1)..seq_len {
                row[t] = 0.0;
            }
        }
        accelerate_gemm::gemm_scores_times_v(&scores, &v_h, seq_len, head_dim, &mut ctx);
        for t in 0..seq_len {
            let os = t * q_dim + h * head_dim;
            out[os..os + head_dim].copy_from_slice(&ctx[t * head_dim..(t + 1) * head_dim]);
        }
    }
    if out.iter().any(|v| !v.is_finite()) {
        return Err(model_err("gqa_prefill_causal produced non-finite output"));
    }
    Ok(())
}

/// F32 GEMM: `out[b] = W @ x[b]` for `b in 0..n_batch`.
pub fn gemm_f32(
    w: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    n_batch: usize,
    out: &mut [f32],
) -> Result<()> {
    if w.len() < rows * cols {
        return Err(model_err(format!(
            "gemm_f32 W len {} < rows*cols {}",
            w.len(),
            rows * cols
        )));
    }
    if x.len() != n_batch * cols || out.len() != n_batch * rows {
        return Err(model_err(format!(
            "gemm_f32 geometry: x={} out={} n_batch={n_batch} rows={rows} cols={cols}",
            x.len(),
            out.len()
        )));
    }
    accelerate_gemm::gemm_w_times_x(w, rows, cols, x, n_batch, out);
    Ok(())
}

/// Row-major BF16 GEMM: widen once, then `out[b] = W @ x[b]`.
pub fn gemm_bf16(
    weight_le: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    n_batch: usize,
    out: &mut [f32],
) -> Result<()> {
    let w = widen_bf16_mat(weight_le, rows, cols)?;
    gemm_f32(&w, rows, cols, x, n_batch, out)
}

/// Row-major GEMV: `out = W @ x` with W stored as little-endian BF16 rows.
///
/// Large mats (lm_head, big projections): widen once and hit Accelerate GEMM.
/// Small projections: on-the-fly convert keeps the decode working set lean.
pub fn gemv_bf16(
    weight_le: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    if x.len() != cols || out.len() != rows {
        return Err(model_err(format!(
            "gemv_bf16 geometry: x={} out={} rows={rows} cols={cols}",
            x.len(),
            out.len()
        )));
    }
    let expect = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| model_err("gemv_bf16 size overflow"))?;
    if weight_le.len() < expect {
        return Err(model_err(format!(
            "gemv_bf16 weight bytes {} < {expect}",
            weight_le.len()
        )));
    }
    const WIDEN_THRESHOLD_ELEMS: usize = 256 * 1024;
    if rows.saturating_mul(cols) >= WIDEN_THRESHOLD_ELEMS {
        return gemm_bf16(weight_le, rows, cols, x, 1, out);
    }
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 16);
    let chunk = rows.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (t, out_chunk) in out.chunks_mut(chunk).enumerate() {
            let row0 = t * chunk;
            let w = weight_le;
            let x = x;
            scope.spawn(move || {
                for (i, o) in out_chunk.iter_mut().enumerate() {
                    let row = row0 + i;
                    if row >= rows {
                        break;
                    }
                    let base = row * cols * 2;
                    let mut acc = 0.0f32;
                    let mut c = 0usize;
                    while c + 4 <= cols {
                        acc += bf16_le_to_f32(&w[base + c * 2..]) * x[c]
                            + bf16_le_to_f32(&w[base + (c + 1) * 2..]) * x[c + 1]
                            + bf16_le_to_f32(&w[base + (c + 2) * 2..]) * x[c + 2]
                            + bf16_le_to_f32(&w[base + (c + 3) * 2..]) * x[c + 3];
                        c += 4;
                    }
                    while c < cols {
                        acc += bf16_le_to_f32(&w[base + c * 2..]) * x[c];
                        c += 1;
                    }
                    *o = acc;
                }
            });
        }
    });
    Ok(())
}

/// Row-major GEMV with f32 rows (after a one-shot BF16 widen of a small tensor).
pub fn gemv_f32_rows(
    w: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    if x.len() != cols || out.len() != rows || w.len() < rows * cols {
        return Err(model_err(format!(
            "gemv_f32 geometry: w={} x={} out={} rows={rows} cols={cols}",
            w.len(),
            x.len(),
            out.len()
        )));
    }
    gemm_f32(w, rows, cols, x, 1, out)
}

#[derive(Clone, Debug)]
struct TensorLoc {
    shard: PathBuf,
    data_offset: u64,
    nbytes: usize,
    shape: Vec<usize>,
    dtype: String,
}

/// Index over the source BF16 safetensors shards. Headers only; payloads range-read.
pub struct SourceBf16Index {
    pub model_dir: PathBuf,
    map: HashMap<String, TensorLoc>,
    handles: Mutex<HashMap<PathBuf, File>>,
    /// Cumulative payload bytes successfully range-read.
    pub bytes_read: Mutex<u64>,
}

impl SourceBf16Index {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let index_path = model_dir.join("model.safetensors.index.json");
        let index_bytes = std::fs::read(&index_path).map_err(|e| {
            model_err(format!(
                "cannot read safetensors index {}: {e}",
                index_path.display()
            ))
        })?;
        let index: Value = serde_json::from_slice(&index_bytes)
            .map_err(|e| model_err(format!("safetensors index is not JSON: {e}")))?;
        let weight_map = index
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| model_err("safetensors index lacks weight_map"))?;

        let mut by_shard: HashMap<String, Vec<String>> = HashMap::new();
        for (name, shard_v) in weight_map {
            let shard = shard_v
                .as_str()
                .ok_or_else(|| model_err(format!("weight_map entry {name} is not a string")))?;
            by_shard
                .entry(shard.to_string())
                .or_default()
                .push(name.clone());
        }

        let mut map = HashMap::new();
        for (shard_name, names) in by_shard {
            let shard_path = model_dir.join(&shard_name);
            let header = read_safetensors_header(&shard_path)?;
            let header_len = header.header_nbytes;
            for name in names {
                let info = header
                    .tensors
                    .get(&name)
                    .ok_or_else(|| model_err(format!("shard {shard_name} lacks tensor {name}")))?;
                if info.dtype != "BF16" && info.dtype != "BFLOAT16" {
                    return Err(model_err(format!(
                        "tensor {name} dtype {} is not BF16",
                        info.dtype
                    )));
                }
                let (begin, end) = info.data_offsets;
                if end < begin {
                    return Err(model_err(format!(
                        "tensor {name} has inverted data_offsets"
                    )));
                }
                let nbytes = (end - begin) as usize;
                map.insert(
                    name,
                    TensorLoc {
                        shard: shard_path.clone(),
                        data_offset: 8 + header_len + begin,
                        nbytes,
                        shape: info.shape.clone(),
                        dtype: info.dtype.clone(),
                    },
                );
            }
        }
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            map,
            handles: Mutex::new(HashMap::new()),
            bytes_read: Mutex::new(0),
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.map.len()
    }

    pub fn bytes_read_total(&self) -> u64 {
        *self.bytes_read.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn require(&self, name: &str) -> Result<&TensorLoc> {
        self.map
            .get(name)
            .ok_or_else(|| model_err(format!("source index lacks tensor {name}")))
    }

    /// Range-read a tensor's raw BF16 payload. Does not keep other tensors resident.
    ///
    /// On Unix this uses `pread` (`read_exact_at`) so concurrent callers on the
    /// same shard do not need to serialize seeks. That is what makes parallel
    /// expert loads in [`LoadedLayer::load`] safe and fast.
    pub fn read_raw(&self, name: &str) -> Result<Vec<u8>> {
        let loc = self.require(name)?;
        // Avoid zero-fill: pread overwrites every byte.
        let mut buf = Vec::with_capacity(loc.nbytes);
        unsafe {
            buf.set_len(loc.nbytes);
        }
        self.read_raw_into(loc, name, &mut buf)?;
        Ok(buf)
    }

    fn ensure_shard_open(&self, shard: &Path) -> Result<()> {
        let mut handles = self
            .handles
            .lock()
            .map_err(|_| model_err("source shard handle map poisoned"))?;
        if !handles.contains_key(shard) {
            let f = File::open(shard)
                .map_err(|e| model_err(format!("cannot open shard {}: {e}", shard.display())))?;
            handles.insert(shard.to_path_buf(), f);
        }
        Ok(())
    }

    fn read_raw_into(&self, loc: &TensorLoc, name: &str, buf: &mut [u8]) -> Result<()> {
        if buf.len() < loc.nbytes {
            return Err(model_err(format!(
                "read_raw_into buffer {} < {}",
                buf.len(),
                loc.nbytes
            )));
        }
        self.ensure_shard_open(&loc.shard)?;
        // Clone the fd under the lock, then release so concurrent preads do not
        // serialize on the handle map.
        let file = {
            let handles = self
                .handles
                .lock()
                .map_err(|_| model_err("source shard handle map poisoned"))?;
            let f = handles
                .get(&loc.shard)
                .ok_or_else(|| model_err(format!("shard {} not open", loc.shard.display())))?;
            f.try_clone().map_err(|e| {
                model_err(format!(
                    "cannot clone shard handle {}: {e}",
                    loc.shard.display()
                ))
            })?
        };
        #[cfg(unix)]
        {
            file.read_exact_at(&mut buf[..loc.nbytes], loc.data_offset)
                .map_err(|e| {
                    model_err(format!(
                        "range-read {} ({} bytes) from {} @ {}: {e}",
                        name,
                        loc.nbytes,
                        loc.shard.display(),
                        loc.data_offset
                    ))
                })?;
        }
        #[cfg(not(unix))]
        {
            let mut f = file;
            f.seek(SeekFrom::Start(loc.data_offset)).map_err(|e| {
                model_err(format!(
                    "seek {} @ {}: {e}",
                    loc.shard.display(),
                    loc.data_offset
                ))
            })?;
            f.read_exact(&mut buf[..loc.nbytes]).map_err(|e| {
                model_err(format!(
                    "range-read {} ({} bytes) from {}: {e}",
                    name,
                    loc.nbytes,
                    loc.shard.display()
                ))
            })?;
        }
        if let Ok(mut br) = self.bytes_read.lock() {
            *br = br.saturating_add(loc.nbytes as u64);
        }
        Ok(())
    }

    /// Parallel range-read of many tensors (used to load a layer's 512 experts).
    pub fn read_raw_many(&self, names: &[String]) -> Result<Vec<Vec<u8>>> {
        if names.is_empty() {
            return Ok(Vec::new());
        }
        for name in names {
            let loc = self.require(name)?;
            self.ensure_shard_open(&loc.shard)?;
        }
        let n = names.len();
        let n_workers = std::thread::available_parallelism()
            .map(|t| t.get())
            .unwrap_or(4)
            .clamp(1, 16)
            .min(n);
        let chunk = n.div_ceil(n_workers);
        let mut out: Vec<Option<Vec<u8>>> = (0..n).map(|_| None).collect();
        let err: Mutex<Option<String>> = Mutex::new(None);
        std::thread::scope(|scope| {
            for (wi, out_chunk) in out.chunks_mut(chunk).enumerate() {
                let base = wi * chunk;
                let names = names;
                let err = &err;
                let index = self;
                scope.spawn(move || {
                    for (i, slot) in out_chunk.iter_mut().enumerate() {
                        let name = &names[base + i];
                        match index.read_raw(name) {
                            Ok(buf) => *slot = Some(buf),
                            Err(e) => {
                                if let Ok(mut g) = err.lock() {
                                    *g = Some(e.to_string());
                                }
                                return;
                            }
                        }
                    }
                });
            }
        });
        if let Some(msg) = err.into_inner().unwrap_or(None) {
            return Err(model_err(msg));
        }
        out.into_iter()
            .map(|o| o.ok_or_else(|| model_err("parallel read_raw_many missing result")))
            .collect()
    }

    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        let raw = self.read_raw(name)?;
        widen_native("native.bf16", &raw)
    }

    /// Read a single embedding row without loading the full embedding table.
    pub fn embed_row(&self, token: u32) -> Result<Vec<f32>> {
        if token as usize >= QWEN80_VOCAB {
            return Err(model_err(format!("token {token} outside vocabulary")));
        }
        let loc = self.require("model.embed_tokens.weight")?;
        if loc.shape != [QWEN80_VOCAB, QWEN80_HIDDEN] {
            return Err(model_err(format!(
                "embed_tokens shape {:?} is not [{QWEN80_VOCAB}, {QWEN80_HIDDEN}]",
                loc.shape
            )));
        }
        let row_bytes = QWEN80_HIDDEN * 2;
        let offset = loc
            .data_offset
            .checked_add(
                (token as u64)
                    .checked_mul(row_bytes as u64)
                    .ok_or_else(|| model_err("embed row offset overflow"))?,
            )
            .ok_or_else(|| model_err("embed absolute offset overflow"))?;
        self.ensure_shard_open(&loc.shard)?;
        let file = {
            let handles = self
                .handles
                .lock()
                .map_err(|_| model_err("source shard handle map poisoned"))?;
            let f = handles
                .get(&loc.shard)
                .ok_or_else(|| model_err(format!("shard {} not open", loc.shard.display())))?;
            f.try_clone()
                .map_err(|e| model_err(format!("cannot clone embed shard handle: {e}")))?
        };
        let mut buf = vec![0u8; row_bytes];
        #[cfg(unix)]
        {
            file.read_exact_at(&mut buf, offset)
                .map_err(|e| model_err(format!("read embed row {token}: {e}")))?;
        }
        #[cfg(not(unix))]
        {
            let mut f = file;
            f.seek(SeekFrom::Start(offset))
                .map_err(|e| model_err(format!("seek embed row {token}: {e}")))?;
            f.read_exact(&mut buf)
                .map_err(|e| model_err(format!("read embed row {token}: {e}")))?;
        }
        if let Ok(mut br) = self.bytes_read.lock() {
            *br = br.saturating_add(row_bytes as u64);
        }
        widen_native("native.bf16", &buf)
    }
}

struct SafetensorsHeader {
    header_nbytes: u64,
    tensors: HashMap<String, SafetensorsTensorInfo>,
}

struct SafetensorsTensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (u64, u64),
}

fn read_safetensors_header(path: &Path) -> Result<SafetensorsHeader> {
    let mut file =
        File::open(path).map_err(|e| model_err(format!("cannot open {}: {e}", path.display())))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf).map_err(|e| {
        model_err(format!(
            "cannot read header length of {}: {e}",
            path.display()
        ))
    })?;
    let header_nbytes = u64::from_le_bytes(len_buf);
    if header_nbytes == 0 || header_nbytes > 64 * 1024 * 1024 {
        return Err(model_err(format!(
            "implausible safetensors header length {header_nbytes} in {}",
            path.display()
        )));
    }
    let mut raw = vec![0u8; header_nbytes as usize];
    file.read_exact(&mut raw)
        .map_err(|e| model_err(format!("cannot read header of {}: {e}", path.display())))?;
    let value: Value = serde_json::from_slice(&raw).map_err(|e| {
        model_err(format!(
            "safetensors header JSON invalid in {}: {e}",
            path.display()
        ))
    })?;
    let object = value.as_object().ok_or_else(|| {
        model_err(format!(
            "safetensors header is not an object in {}",
            path.display()
        ))
    })?;
    let mut tensors = HashMap::new();
    for (name, info_v) in object {
        if name == "__metadata__" {
            continue;
        }
        let info = info_v
            .as_object()
            .ok_or_else(|| model_err(format!("tensor {name} header is not an object")))?;
        let dtype = info
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| model_err(format!("tensor {name} lacks dtype")))?
            .to_string();
        let shape = info
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| model_err(format!("tensor {name} lacks shape")))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| model_err(format!("tensor {name} has non-integer shape")))
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = info
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| model_err(format!("tensor {name} lacks data_offsets")))?;
        if offsets.len() != 2 {
            return Err(model_err(format!(
                "tensor {name} data_offsets is not a pair"
            )));
        }
        let begin = offsets[0]
            .as_u64()
            .ok_or_else(|| model_err(format!("tensor {name} data_offsets[0] invalid")))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| model_err(format!("tensor {name} data_offsets[1] invalid")))?;
        tensors.insert(
            name.clone(),
            SafetensorsTensorInfo {
                dtype,
                shape,
                data_offsets: (begin, end),
            },
        );
    }
    Ok(SafetensorsHeader {
        header_nbytes,
        tensors,
    })
}

fn layer_name(layer: usize, suffix: &str) -> String {
    format!("model.layers.{layer}.{suffix}")
}

fn expert_name(layer: usize, expert: usize, role: &str) -> String {
    format!("model.layers.{layer}.mlp.experts.{expert}.{role}.weight")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerKind {
    LinearDeltaNet,
    FullAttentionGqa,
}

pub fn layer_kind(layer: usize) -> Result<LayerKind> {
    if layer >= QWEN80_LAYERS {
        return Err(model_err(format!("layer {layer} out of range")));
    }
    // Source: (layer + 1) % full_attention_interval == 0 => GQA
    if (layer + 1) % QWEN80_FULL_ATTENTION_INTERVAL == 0 {
        Ok(LayerKind::FullAttentionGqa)
    } else {
        Ok(LayerKind::LinearDeltaNet)
    }
}

pub struct ExpertWeights {
    pub gate: Vec<u8>,
    pub up: Vec<u8>,
    pub down: Vec<u8>,
}

/// Mixer + MoE weights for one layer, held only for that layer's corpus pass.
pub struct LoadedLayer {
    pub layer: usize,
    pub kind: LayerKind,
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    // DeltaNet
    pub in_proj_qkvz: Option<Vec<u8>>,
    pub in_proj_ba: Option<Vec<u8>>,
    pub conv1d: Option<Vec<f32>>,
    pub a_log: Option<Vec<f32>>,
    pub dt_bias: Option<Vec<f32>>,
    pub gated_rms_norm: Option<Vec<f32>>,
    pub out_proj_linear: Option<Vec<u8>>,
    // GQA
    pub q_proj: Option<Vec<u8>>,
    pub k_proj: Option<Vec<u8>>,
    pub v_proj: Option<Vec<u8>>,
    pub o_proj: Option<Vec<u8>>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    // MoE
    pub router: Vec<u8>,
    pub shared_gate: Vec<u8>,
    pub shared_up: Vec<u8>,
    pub shared_down: Vec<u8>,
    pub shared_expert_gate: Vec<u8>,
    pub experts: Vec<ExpertWeights>,
    pub resident_bytes: u64,
    pub load_secs: f64,
}

impl LoadedLayer {
    pub fn load(index: &SourceBf16Index, layer: usize) -> Result<Self> {
        let t0 = Instant::now();
        let kind = layer_kind(layer)?;
        let input_layernorm = index.read_f32(&layer_name(layer, "input_layernorm.weight"))?;
        let post_attention_layernorm =
            index.read_f32(&layer_name(layer, "post_attention_layernorm.weight"))?;
        let router = index.read_raw(&layer_name(layer, "mlp.gate.weight"))?;
        let shared_gate =
            index.read_raw(&layer_name(layer, "mlp.shared_expert.gate_proj.weight"))?;
        let shared_up = index.read_raw(&layer_name(layer, "mlp.shared_expert.up_proj.weight"))?;
        let shared_down =
            index.read_raw(&layer_name(layer, "mlp.shared_expert.down_proj.weight"))?;
        let shared_expert_gate =
            index.read_raw(&layer_name(layer, "mlp.shared_expert_gate.weight"))?;

        let mut resident = (input_layernorm.len() + post_attention_layernorm.len()) * 4
            + router.len()
            + shared_gate.len()
            + shared_up.len()
            + shared_down.len()
            + shared_expert_gate.len();

        let mut in_proj_qkvz = None;
        let mut in_proj_ba = None;
        let mut conv1d = None;
        let mut a_log = None;
        let mut dt_bias = None;
        let mut gated_rms_norm = None;
        let mut out_proj_linear = None;
        let mut q_proj = None;
        let mut k_proj = None;
        let mut v_proj = None;
        let mut o_proj = None;
        let mut q_norm = None;
        let mut k_norm = None;

        match kind {
            LayerKind::LinearDeltaNet => {
                let qkvz = index.read_raw(&layer_name(layer, "linear_attn.in_proj_qkvz.weight"))?;
                let ba = index.read_raw(&layer_name(layer, "linear_attn.in_proj_ba.weight"))?;
                let conv = index.read_f32(&layer_name(layer, "linear_attn.conv1d.weight"))?;
                let al = index.read_f32(&layer_name(layer, "linear_attn.A_log"))?;
                let dt = index.read_f32(&layer_name(layer, "linear_attn.dt_bias"))?;
                let gn = index.read_f32(&layer_name(layer, "linear_attn.norm.weight"))?;
                let op = index.read_raw(&layer_name(layer, "linear_attn.out_proj.weight"))?;
                resident += qkvz.len()
                    + ba.len()
                    + conv.len() * 4
                    + al.len() * 4
                    + dt.len() * 4
                    + gn.len() * 4
                    + op.len();
                in_proj_qkvz = Some(qkvz);
                in_proj_ba = Some(ba);
                conv1d = Some(conv);
                a_log = Some(al);
                dt_bias = Some(dt);
                gated_rms_norm = Some(gn);
                out_proj_linear = Some(op);
            }
            LayerKind::FullAttentionGqa => {
                let q = index.read_raw(&layer_name(layer, "self_attn.q_proj.weight"))?;
                let k = index.read_raw(&layer_name(layer, "self_attn.k_proj.weight"))?;
                let v = index.read_raw(&layer_name(layer, "self_attn.v_proj.weight"))?;
                let o = index.read_raw(&layer_name(layer, "self_attn.o_proj.weight"))?;
                let qn = index.read_f32(&layer_name(layer, "self_attn.q_norm.weight"))?;
                let kn = index.read_f32(&layer_name(layer, "self_attn.k_norm.weight"))?;
                resident += q.len() + k.len() + v.len() + o.len() + qn.len() * 4 + kn.len() * 4;
                q_proj = Some(q);
                k_proj = Some(k);
                v_proj = Some(v);
                o_proj = Some(o);
                q_norm = Some(qn);
                k_norm = Some(kn);
            }
        }

        // 512 experts × gate/up/down: parallel pread across shards (the 28%→high
        // CPU fix — sequential range-reads starve the cores on this many tensors).
        let mut expert_names = Vec::with_capacity(QWEN80_EXPERTS * 3);
        for expert in 0..QWEN80_EXPERTS {
            expert_names.push(expert_name(layer, expert, "gate_proj"));
            expert_names.push(expert_name(layer, expert, "up_proj"));
            expert_names.push(expert_name(layer, expert, "down_proj"));
        }
        let mut expert_payloads = index.read_raw_many(&expert_names)?;
        if expert_payloads.len() != QWEN80_EXPERTS * 3 {
            return Err(model_err(format!(
                "expert payload count {} != {}",
                expert_payloads.len(),
                QWEN80_EXPERTS * 3
            )));
        }
        let mut experts = Vec::with_capacity(QWEN80_EXPERTS);
        let mut payloads = expert_payloads.drain(..);
        for _ in 0..QWEN80_EXPERTS {
            let gate = payloads.next().unwrap();
            let up = payloads.next().unwrap();
            let down = payloads.next().unwrap();
            resident += gate.len() + up.len() + down.len();
            experts.push(ExpertWeights { gate, up, down });
        }

        Ok(Self {
            layer,
            kind,
            input_layernorm,
            post_attention_layernorm,
            in_proj_qkvz,
            in_proj_ba,
            conv1d,
            a_log,
            dt_bias,
            gated_rms_norm,
            out_proj_linear,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            router,
            shared_gate,
            shared_up,
            shared_down,
            shared_expert_gate,
            experts,
            resident_bytes: resident as u64,
            load_secs: t0.elapsed().as_secs_f64(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct DeltaNetState {
    pub conv_state: Vec<f32>,
    pub recurrent_state: Vec<f32>,
}

impl DeltaNetState {
    pub fn zero(layout: &Qwen80CanonicalLinearDeltaNetLayout) -> Result<Self> {
        Ok(Self {
            conv_state: vec![0.0; layout.conv_state_elements()?],
            recurrent_state: vec![0.0; layout.recurrent_state_elements()?],
        })
    }
}

#[derive(Clone, Debug)]
pub struct GqaState {
    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>,
    pub max_seq: usize,
}

impl GqaState {
    pub fn new(max_seq: usize, layout: &Qwen80CanonicalGqaLayout) -> Self {
        Self {
            key_cache: vec![0.0; max_seq * layout.kv_dim],
            value_cache: vec![0.0; max_seq * layout.kv_dim],
            max_seq,
        }
    }
}

/// Per-token capture surface for one layer (matches complete-binary capture).
///
/// After per-layer flush, [`Self::router_input_hidden`] is empty even when
/// [`Self::hidden_retained`] is true — the row has been written and freed.
/// Route membership stays complete for every token.
#[derive(Clone, Debug)]
pub struct LayerTokenCapture {
    pub layer: usize,
    pub selected_expert_ids: Vec<u32>,
    pub normalized_route_weights: Vec<f32>,
    pub router_input_hidden: Vec<f32>,
    /// Whether first-N kept this token's hidden at this layer. Survives flush.
    pub hidden_retained: bool,
}

pub type ProbeHidden = Vec<f32>;

fn swiglu_mlp_bf16(
    gate_w: &[u8],
    up_w: &[u8],
    down_w: &[u8],
    x: &[f32],
    intermediate: usize,
    gate_buf: &mut [f32],
    up_buf: &mut [f32],
    act_buf: &mut [f32],
    down_buf: &mut [f32],
) -> Result<()> {
    gemv_bf16(gate_w, intermediate, QWEN80_HIDDEN, x, gate_buf)?;
    gemv_bf16(up_w, intermediate, QWEN80_HIDDEN, x, up_buf)?;
    silu_mul(gate_buf, up_buf, act_buf);
    gemv_bf16(down_w, QWEN80_HIDDEN, intermediate, act_buf, down_buf)?;
    Ok(())
}

fn moe_combine(
    layer: &LoadedLayer,
    router_input: &[f32],
    moe_combined: &mut [f32],
    router_logits: &mut [f32],
    gate: &mut [f32],
    up: &mut [f32],
    act: &mut [f32],
    down: &mut [f32],
    shared_gate: &mut [f32],
    shared_up: &mut [f32],
    shared_act: &mut [f32],
    shared_down: &mut [f32],
    shared_gate_logit: &mut [f32],
) -> Result<(Vec<u32>, Vec<f32>)> {
    // Shared MLP first (source SparseMoeBlock order), then router + routed.
    swiglu_mlp_bf16(
        &layer.shared_gate,
        &layer.shared_up,
        &layer.shared_down,
        router_input,
        QWEN80_SHARED_EXPERT_INTERMEDIATE,
        shared_gate,
        shared_up,
        shared_act,
        shared_down,
    )?;
    gemv_bf16(
        &layer.router,
        QWEN80_EXPERTS,
        QWEN80_HIDDEN,
        router_input,
        router_logits,
    )?;
    let route = source_qwen80_topk_router(router_logits)?;
    moe_combined.fill(0.0);
    for (slot, (&eid, &w)) in route.ids.iter().zip(route.weights.iter()).enumerate() {
        let expert = layer
            .experts
            .get(eid as usize)
            .ok_or_else(|| model_err(format!("route expert {eid} out of range")))?;
        swiglu_mlp_bf16(
            &expert.gate,
            &expert.up,
            &expert.down,
            router_input,
            QWEN80_MOE_INTERMEDIATE,
            gate,
            up,
            act,
            down,
        )?;
        for i in 0..QWEN80_HIDDEN {
            moe_combined[i] += down[i] * w;
        }
        let _ = slot;
    }
    gemv_bf16(
        &layer.shared_expert_gate,
        1,
        QWEN80_HIDDEN,
        router_input,
        shared_gate_logit,
    )?;
    let gate_val = 1.0 / (1.0 + (-shared_gate_logit[0]).exp());
    if !gate_val.is_finite() || !(0.0..=1.0).contains(&gate_val) {
        return Err(model_err("shared expert gate sigmoid invalid"));
    }
    for i in 0..QWEN80_HIDDEN {
        moe_combined[i] += shared_down[i] * gate_val;
    }
    let ids = route.ids.iter().map(|&id| id as u32).collect();
    let weights = route.weights.to_vec();
    Ok((ids, weights))
}

/// One DeltaNet mixer step through first residual (reuses packed-oracle maths).
fn deltanet_mixer_step(
    layer: &LoadedLayer,
    hidden: &[f32],
    state: &mut DeltaNetState,
    layout: &Qwen80CanonicalLinearDeltaNetLayout,
) -> Result<Vec<f32>> {
    let qkvz_w = layer
        .in_proj_qkvz
        .as_ref()
        .ok_or_else(|| model_err("missing in_proj_qkvz"))?;
    let ba_w = layer
        .in_proj_ba
        .as_ref()
        .ok_or_else(|| model_err("missing in_proj_ba"))?;
    let conv_w = layer
        .conv1d
        .as_ref()
        .ok_or_else(|| model_err("missing conv1d"))?;
    let a_log = layer
        .a_log
        .as_ref()
        .ok_or_else(|| model_err("missing A_log"))?;
    let dt_bias = layer
        .dt_bias
        .as_ref()
        .ok_or_else(|| model_err("missing dt_bias"))?;
    let gated_norm = layer
        .gated_rms_norm
        .as_ref()
        .ok_or_else(|| model_err("missing gated_rms_norm"))?;
    let out_proj = layer
        .out_proj_linear
        .as_ref()
        .ok_or_else(|| model_err("missing linear out_proj"))?;

    let input_rms = source_qwen80_residual_rms_norm(hidden, &layer.input_layernorm)?;
    let qkvz_rows = layout.qkvz_projection_elements()?;
    let ba_rows = layout.ba_projection_elements()?;
    let mut projected_qkvz = vec![0.0f32; qkvz_rows];
    let mut projected_ba = vec![0.0f32; ba_rows];
    gemv_bf16(
        qkvz_w,
        qkvz_rows,
        QWEN80_HIDDEN,
        &input_rms,
        &mut projected_qkvz,
    )?;
    gemv_bf16(ba_w, ba_rows, QWEN80_HIDDEN, &input_rms, &mut projected_ba)?;

    let (raw_query, raw_key, raw_value, z) =
        source_qwen80_split_linear_qkvz(&projected_qkvz, layout)?;
    let mut mixed_qkv = Vec::with_capacity(layout.conv_channels);
    mixed_qkv.extend_from_slice(&raw_query);
    mixed_qkv.extend_from_slice(&raw_key);
    mixed_qkv.extend_from_slice(&raw_value);
    let (convolved_qkv, next_conv) =
        source_qwen80_causal_conv_step_dense(&mixed_qkv, &state.conv_state, conv_w, layout)?;
    let raw_query_len = layout.key_elements()?;
    let raw_key_len = raw_query_len;
    let raw_value_len = layout.value_elements()?;
    let convolved_query = &convolved_qkv[..raw_query_len];
    let convolved_key = &convolved_qkv[raw_query_len..raw_query_len + raw_key_len];
    let convolved_value = &convolved_qkv[raw_query_len + raw_key_len..];
    if convolved_value.len() != raw_value_len {
        return Err(model_err("convolution value geometry broken"));
    }
    let convolved_value = convolved_value.to_vec();

    let mut repeated_query = vec![0.0f32; raw_value_len];
    let mut repeated_key = vec![0.0f32; raw_value_len];
    for value_head in 0..layout.value_heads {
        let key_head = value_head / layout.value_heads_per_key_head;
        let mut query_head = convolved_query
            [key_head * layout.key_head_dim..(key_head + 1) * layout.key_head_dim]
            .to_vec();
        let mut key_head_values = convolved_key
            [key_head * layout.key_head_dim..(key_head + 1) * layout.key_head_dim]
            .to_vec();
        source_qwen80_l2_normalize(&mut query_head, (layout.key_head_dim as f32).sqrt().recip())?;
        source_qwen80_l2_normalize(&mut key_head_values, 1.0)?;
        let destination = value_head * layout.key_head_dim;
        repeated_query[destination..destination + layout.key_head_dim].copy_from_slice(&query_head);
        repeated_key[destination..destination + layout.key_head_dim]
            .copy_from_slice(&key_head_values);
    }
    let (decay, beta) = source_qwen80_ba_to_decay_beta(&projected_ba, a_log, dt_bias, layout)?;
    let recurrent_output = source_qwen80_recurrent_deltanet(
        &mut state.recurrent_state,
        &repeated_query,
        &repeated_key,
        &convolved_value,
        &decay,
        &beta,
        layout,
    )?;
    state.conv_state = next_conv;
    let repeated_gated_norm_weight = (0..layout.value_heads)
        .flat_map(|_| gated_norm.iter().copied())
        .collect::<Vec<_>>();
    let gated_output = source_qwen80_gated_rms_norm(
        &recurrent_output,
        &z,
        &repeated_gated_norm_weight,
        layout.value_heads,
        layout.value_head_dim,
    )?;
    let mut mixer_output = vec![0.0f32; QWEN80_HIDDEN];
    gemv_bf16(
        out_proj,
        QWEN80_HIDDEN,
        raw_value_len,
        &gated_output,
        &mut mixer_output,
    )?;
    let mut residual = hidden.to_vec();
    add_inplace(&mut residual, &mixer_output);
    if residual.iter().any(|v| !v.is_finite()) {
        return Err(model_err("DeltaNet residual non-finite"));
    }
    Ok(residual)
}

/// One GQA mixer step through first residual.
fn gqa_mixer_step(
    layer: &LoadedLayer,
    hidden: &[f32],
    state: &mut GqaState,
    position: usize,
    layout: &Qwen80CanonicalGqaLayout,
) -> Result<Vec<f32>> {
    if position >= state.max_seq {
        return Err(model_err(format!(
            "GQA position {position} exceeds max_seq {}",
            state.max_seq
        )));
    }
    let q_proj = layer
        .q_proj
        .as_ref()
        .ok_or_else(|| model_err("missing q_proj"))?;
    let k_proj = layer
        .k_proj
        .as_ref()
        .ok_or_else(|| model_err("missing k_proj"))?;
    let v_proj = layer
        .v_proj
        .as_ref()
        .ok_or_else(|| model_err("missing v_proj"))?;
    let o_proj = layer
        .o_proj
        .as_ref()
        .ok_or_else(|| model_err("missing o_proj"))?;
    let q_norm = layer
        .q_norm
        .as_ref()
        .ok_or_else(|| model_err("missing q_norm"))?;
    let k_norm = layer
        .k_norm
        .as_ref()
        .ok_or_else(|| model_err("missing k_norm"))?;

    let input_rms = source_qwen80_residual_rms_norm(hidden, &layer.input_layernorm)?;
    let mut q_projection = vec![0.0f32; layout.q_proj_rows];
    let mut k_projection = vec![0.0f32; layout.kv_dim];
    let mut v_projection = vec![0.0f32; layout.kv_dim];
    gemv_bf16(
        q_proj,
        layout.q_proj_rows,
        QWEN80_HIDDEN,
        &input_rms,
        &mut q_projection,
    )?;
    gemv_bf16(
        k_proj,
        layout.kv_dim,
        QWEN80_HIDDEN,
        &input_rms,
        &mut k_projection,
    )?;
    gemv_bf16(
        v_proj,
        layout.kv_dim,
        QWEN80_HIDDEN,
        &input_rms,
        &mut v_projection,
    )?;

    let query_raw = qwen80_gqa_query_from_interleaved_q_projection(&q_projection, layout)?;
    let query = qwen80_gqa_source_norm_rope(
        &query_raw,
        q_norm,
        layout.query_heads,
        layout.head_dim,
        layout.rotary_dim,
        position,
        "GQA q_norm + partial RoPE",
    )?;
    let key_row = qwen80_gqa_source_norm_rope(
        &k_projection,
        k_norm,
        layout.key_value_heads,
        layout.head_dim,
        layout.rotary_dim,
        position,
        "GQA k_norm + partial RoPE",
    )?;
    let cache_start = position
        .checked_mul(layout.kv_dim)
        .ok_or_else(|| model_err("GQA cache start overflow"))?;
    let cache_end = cache_start
        .checked_add(layout.kv_dim)
        .ok_or_else(|| model_err("GQA cache end overflow"))?;
    state.key_cache[cache_start..cache_end].copy_from_slice(&key_row);
    state.value_cache[cache_start..cache_end].copy_from_slice(&v_projection);
    let sequence_length = position + 1;
    let attention = qwen80_gqa_causal_attention(
        &query,
        &state.key_cache,
        &state.value_cache,
        sequence_length,
        layout,
    )?;
    let gated = qwen80_gqa_apply_sigmoid_gate(&attention, &q_projection, layout)?;
    let mut mixer_output = vec![0.0f32; QWEN80_HIDDEN];
    gemv_bf16(
        o_proj,
        QWEN80_HIDDEN,
        layout.query_dim,
        &gated,
        &mut mixer_output,
    )?;
    let mut residual = hidden.to_vec();
    add_inplace(&mut residual, &mixer_output);
    if residual.iter().any(|v| !v.is_finite()) {
        return Err(model_err("GQA residual non-finite"));
    }
    Ok(residual)
}

/// Batched GQA mixer over a full probe sequence: one Q/K/V/O GEMM, sgemm causal
/// attention. This is the change that dominated wall clock on Q30 long probes;
/// same story here on GQA layers (3,7,...,47).
fn gqa_mixer_prefill_probe(
    layer: &LoadedLayer,
    hidden: &mut [f32],
    seq_len: usize,
    layout: &Qwen80CanonicalGqaLayout,
) -> Result<()> {
    if seq_len == 0 {
        return Ok(());
    }
    let h = QWEN80_HIDDEN;
    if hidden.len() != seq_len * h {
        return Err(model_err("gqa_mixer_prefill_probe hidden geometry"));
    }
    let q_proj = layer
        .q_proj
        .as_ref()
        .ok_or_else(|| model_err("missing q_proj"))?;
    let k_proj = layer
        .k_proj
        .as_ref()
        .ok_or_else(|| model_err("missing k_proj"))?;
    let v_proj = layer
        .v_proj
        .as_ref()
        .ok_or_else(|| model_err("missing v_proj"))?;
    let o_proj = layer
        .o_proj
        .as_ref()
        .ok_or_else(|| model_err("missing o_proj"))?;
    let q_norm = layer
        .q_norm
        .as_ref()
        .ok_or_else(|| model_err("missing q_norm"))?;
    let k_norm = layer
        .k_norm
        .as_ref()
        .ok_or_else(|| model_err("missing k_norm"))?;

    // Widen dense projections once; per-position M=1 matvecs match scalar
    // `gqa_mixer_step` / gemv_bf16→gemm_bf16. Batched attention (sgemm QKᵀ/PV)
    // is applied over the full sequence after Q/K/V are materialised — that is
    // the dominant prefill win on long probes.
    let q_w = widen_bf16_mat(q_proj, layout.q_proj_rows, h)?;
    let k_w = widen_bf16_mat(k_proj, layout.kv_dim, h)?;
    let v_w = widen_bf16_mat(v_proj, layout.kv_dim, h)?;
    let o_w = widen_bf16_mat(o_proj, h, layout.query_dim)?;

    let mut q_projection = vec![0.0f32; seq_len * layout.q_proj_rows];
    let mut k_cache = vec![0.0f32; seq_len * layout.kv_dim];
    let mut v_cache = vec![0.0f32; seq_len * layout.kv_dim];
    let mut query = vec![0.0f32; seq_len * layout.query_dim];

    for pos in 0..seq_len {
        let rin = source_qwen80_residual_rms_norm(
            &hidden[pos * h..(pos + 1) * h],
            &layer.input_layernorm,
        )?;
        let q_row = &mut q_projection[pos * layout.q_proj_rows..(pos + 1) * layout.q_proj_rows];
        let k_row = &mut k_cache[pos * layout.kv_dim..(pos + 1) * layout.kv_dim];
        let v_row = &mut v_cache[pos * layout.kv_dim..(pos + 1) * layout.kv_dim];
        gemm_f32(&q_w, layout.q_proj_rows, h, &rin, 1, q_row)?;
        gemm_f32(&k_w, layout.kv_dim, h, &rin, 1, k_row)?;
        gemm_f32(&v_w, layout.kv_dim, h, &rin, 1, v_row)?;

        let query_raw = qwen80_gqa_query_from_interleaved_q_projection(q_row, layout)?;
        let q_normed = qwen80_gqa_source_norm_rope(
            &query_raw,
            q_norm,
            layout.query_heads,
            layout.head_dim,
            layout.rotary_dim,
            pos,
            "GQA q_norm + partial RoPE",
        )?;
        query[pos * layout.query_dim..(pos + 1) * layout.query_dim].copy_from_slice(&q_normed);

        let k_normed = qwen80_gqa_source_norm_rope(
            k_row,
            k_norm,
            layout.key_value_heads,
            layout.head_dim,
            layout.rotary_dim,
            pos,
            "GQA k_norm + partial RoPE",
        )?;
        k_row.copy_from_slice(&k_normed);
    }
    drop(q_w);
    drop(k_w);
    drop(v_w);

    let mut attn = vec![0.0f32; seq_len * layout.query_dim];
    gqa_prefill_causal(&query, &k_cache, &v_cache, layout, seq_len, &mut attn)?;
    drop(query);
    drop(k_cache);
    drop(v_cache);

    let mut gated = vec![0.0f32; seq_len * layout.query_dim];
    for pos in 0..seq_len {
        let g = qwen80_gqa_apply_sigmoid_gate(
            &attn[pos * layout.query_dim..(pos + 1) * layout.query_dim],
            &q_projection[pos * layout.q_proj_rows..(pos + 1) * layout.q_proj_rows],
            layout,
        )?;
        gated[pos * layout.query_dim..(pos + 1) * layout.query_dim].copy_from_slice(&g);
    }
    drop(attn);
    drop(q_projection);

    for pos in 0..seq_len {
        let mut mixer_row = vec![0.0f32; h];
        gemm_f32(
            &o_w,
            h,
            layout.query_dim,
            &gated[pos * layout.query_dim..(pos + 1) * layout.query_dim],
            1,
            &mut mixer_row,
        )?;
        add_inplace(&mut hidden[pos * h..(pos + 1) * h], &mixer_row);
        if hidden[pos * h..(pos + 1) * h]
            .iter()
            .any(|v| !v.is_finite())
        {
            return Err(model_err("GQA residual non-finite"));
        }
    }
    Ok(())
}

/// Run one loaded layer over one probe sequence (causal within the probe).
pub fn forward_layer_probe(
    layer: &LoadedLayer,
    hidden: &mut ProbeHidden,
    seq_len: usize,
    // Global token offset of this probe, and the stride of the retained hidden
    // subsample. `retain_stride <= 1` keeps every token (the previous behaviour).
    probe_token_offset: usize,
    retain_stride: usize,
) -> Result<Vec<LayerTokenCapture>> {
    if seq_len == 0 {
        return Ok(Vec::new());
    }
    if hidden.len() != seq_len * QWEN80_HIDDEN {
        return Err(model_err(format!(
            "probe hidden len {} != seq {seq_len} * {QWEN80_HIDDEN}",
            hidden.len()
        )));
    }

    let linear_layout = Qwen80CanonicalLinearDeltaNetLayout::source_exact();
    linear_layout.validate()?;
    let gqa_layout = Qwen80CanonicalGqaLayout::source_exact();
    gqa_layout.validate()?;

    let mut delta_state = DeltaNetState::zero(&linear_layout)?;
    let mut gqa_state = GqaState::new(seq_len, &gqa_layout);
    let mut captures = Vec::with_capacity(seq_len);

    let mut router_logits = vec![0.0f32; QWEN80_EXPERTS];
    let mut gate = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
    let mut up = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
    let mut act = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
    let mut down = vec![0.0f32; QWEN80_HIDDEN];
    let mut shared_gate = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
    let mut shared_up = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
    let mut shared_act = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
    let mut shared_down = vec![0.0f32; QWEN80_HIDDEN];
    let mut shared_gate_logit = vec![0.0f32; 1];
    let mut moe_combined = vec![0.0f32; QWEN80_HIDDEN];

    for pos in 0..seq_len {
        let x_slice = &hidden[pos * QWEN80_HIDDEN..(pos + 1) * QWEN80_HIDDEN];
        let x_in = x_slice.to_vec();
        let first_residual = match layer.kind {
            LayerKind::LinearDeltaNet => {
                deltanet_mixer_step(layer, &x_in, &mut delta_state, &linear_layout)?
            }
            LayerKind::FullAttentionGqa => {
                gqa_mixer_step(layer, &x_in, &mut gqa_state, pos, &gqa_layout)?
            }
        };
        let router_input =
            source_qwen80_residual_rms_norm(&first_residual, &layer.post_attention_layernorm)?;
        let (ids, weights) = moe_combine(
            layer,
            &router_input,
            &mut moe_combined,
            &mut router_logits,
            &mut gate,
            &mut up,
            &mut act,
            &mut down,
            &mut shared_gate,
            &mut shared_up,
            &mut shared_act,
            &mut shared_down,
            &mut shared_gate_logit,
        )?;
        let mut out = first_residual;
        add_inplace(&mut out, &moe_combined);
        if out.iter().any(|v| !v.is_finite()) {
            return Err(model_err(format!(
                "layer {} pos {pos} second residual non-finite",
                layer.layer
            )));
        }
        hidden[pos * QWEN80_HIDDEN..(pos + 1) * QWEN80_HIDDEN].copy_from_slice(&out);
        // Same defect Q30 had: this hidden is QWEN80_HIDDEN f32 = 8 KiB per token per
        // layer, and retaining it for every token costs total_tokens * 8 KiB PER LAYER
        // for the whole run. Route membership stays complete (small, and the fit needs
        // full coverage); only the hidden is bounded, by a deterministic stride so the
        // retained set is reproducible and layer-independent.
        let flat_index = probe_token_offset + pos;
        let retain = retain_stride <= 1 || (flat_index % retain_stride) == 0;
        captures.push(LayerTokenCapture {
            layer: layer.layer,
            selected_expert_ids: ids,
            normalized_route_weights: weights,
            router_input_hidden: if retain { router_input } else { Vec::new() },
            hidden_retained: retain,
        });
    }
    Ok(captures)
}

/// Embed every probe's tokens into residual streams (range-read rows only).
pub fn embed_probes(
    index: &SourceBf16Index,
    probes: &[(String, Vec<u32>)],
) -> Result<Vec<ProbeHidden>> {
    let mut out = Vec::with_capacity(probes.len());
    for (_, tokens) in probes {
        let mut h = Vec::with_capacity(tokens.len() * QWEN80_HIDDEN);
        for &tok in tokens {
            let row = index.embed_row(tok)?;
            h.extend_from_slice(&row);
        }
        out.push(h);
    }
    Ok(out)
}

/// Final RMSNorm + lm_head logits for the last residual of a sequence.
pub fn logits_from_final_hidden(index: &SourceBf16Index, hidden: &[f32]) -> Result<Vec<f32>> {
    if hidden.len() != QWEN80_HIDDEN {
        return Err(model_err("final hidden width mismatch"));
    }
    let norm_w = index.read_f32("model.norm.weight")?;
    let normed = source_qwen80_residual_rms_norm(hidden, &norm_w)?;
    // lm_head ~593 MiB BF16 — load, matvec, free.
    let lm_head = index.read_raw("lm_head.weight")?;
    let mut logits = vec![0.0f32; QWEN80_VOCAB];
    gemv_bf16(&lm_head, QWEN80_VOCAB, QWEN80_HIDDEN, &normed, &mut logits)?;
    drop(lm_head);
    // Mask source-reserved tail (tokenizer vocab 151669 .. 151935).
    for logit in logits.iter_mut().skip(QWEN80_TOKENIZER_VOCAB) {
        *logit = f32::NEG_INFINITY;
    }
    Ok(logits)
}

/// Timing / bandwidth telemetry for one full layer-major pass.
#[derive(Clone, Debug, Default)]
pub struct StreamTelemetry {
    pub layers: usize,
    pub tokens: usize,
    pub weight_bytes_read: u64,
    pub load_secs: f64,
    pub compute_secs: f64,
    pub wall_secs: f64,
    pub max_layer_resident_bytes: u64,
    pub peak_rss_bytes: u64,
}

impl StreamTelemetry {
    pub fn stream_gib_per_s(&self) -> f64 {
        if self.load_secs <= 0.0 {
            return 0.0;
        }
        (self.weight_bytes_read as f64) / self.load_secs / (1024.0 * 1024.0 * 1024.0)
    }

    /// Tokens for which compute wall would equal load wall at this measured rate.
    /// Corpus sizes up to this are "free" (I/O bound).
    pub fn free_corpus_crossover_tokens(&self) -> f64 {
        if self.compute_secs <= 0.0 || self.tokens == 0 {
            return f64::INFINITY;
        }
        // At fixed stream: load_secs is independent of tokens; compute scales linearly.
        // Crossover when compute_secs(t) = load_secs => t = load_secs / (compute_secs/tokens)
        let compute_per_token = self.compute_secs / self.tokens as f64;
        if compute_per_token <= 0.0 {
            return f64::INFINITY;
        }
        self.load_secs / compute_per_token
    }
}

/// Layer-major full forward over all probes.
///
/// Per layer:
/// 1. Mixer phase — GQA: batched Q/K/V/O + sgemm causal attention per probe;
///    DeltaNet: sequential recurrence within each probe (batch across probes
///    only via the shared MoE phase, never across time).
/// 2. Collect every token's post-attention RMSNorm (router input) into one matrix.
/// 3. One router GEMM over the whole corpus at this layer.
/// 4. Shared expert one batched SwiGLU; routed experts gather/GEMM/scatter in
///    parallel; widen only active experts.
/// 5. Retains router-input hiddens under **per-expert first-N** (see
///    [`credit_expert_first_n_retention`]): the first `max_hidden_tokens_per_expert`
///    tokens that route to expert E keep their hidden for that layer. Full route
///    membership is always recorded.
/// 6. Invokes `on_layer_flush` (caller writes this layer's retained rows) and
///    then drops those hidden payloads before layer `L+1` loads. The returned
///    captures keep complete routes; `router_input_hidden` is empty after flush.
///
/// Retention must happen *after* routing is known, so a precomputed global
/// (probe, position) set cannot express per-expert quotas. Both DeltaNet and
/// GQA layers run the same MoE router (512 experts, top-10); only the mixer
/// differs. Expert-retained counters reset each layer.
pub fn capture_all_layers(
    index: &SourceBf16Index,
    probes: &[(String, Vec<u32>)],
    hiddens: &mut [ProbeHidden],
    max_hidden_tokens_per_expert: usize,
    mut on_layer: Option<&mut dyn FnMut(usize, &LoadedLayer, &StreamTelemetry)>,
    mut on_layer_flush: Option<
        &mut dyn FnMut(usize, &mut [Vec<Vec<LayerTokenCapture>>]) -> Result<()>,
    >,
) -> Result<(Vec<Vec<Vec<LayerTokenCapture>>>, StreamTelemetry)> {
    if hiddens.len() != probes.len() {
        return Err(model_err("hiddens/probes length mismatch"));
    }
    let total_tokens: usize = probes.iter().map(|(_, t)| t.len()).sum();
    let mut captures: Vec<Vec<Vec<LayerTokenCapture>>> = probes
        .iter()
        .map(|(_, toks)| {
            (0..toks.len())
                .map(|_| Vec::with_capacity(QWEN80_LAYERS))
                .collect()
        })
        .collect();

    let wall0 = Instant::now();
    let bytes0 = index.bytes_read_total();
    let mut telem = StreamTelemetry {
        layers: QWEN80_LAYERS,
        tokens: total_tokens,
        ..Default::default()
    };

    let h = QWEN80_HIDDEN;
    let inter = QWEN80_MOE_INTERMEDIATE;
    let shared_inter = QWEN80_SHARED_EXPERT_INTERMEDIATE;
    let linear_layout = Qwen80CanonicalLinearDeltaNetLayout::source_exact();
    linear_layout.validate()?;
    let gqa_layout = Qwen80CanonicalGqaLayout::source_exact();
    gqa_layout.validate()?;

    let mut all_router_in = vec![0.0f32; total_tokens * h];
    let mut router_logits = vec![0.0f32; total_tokens * QWEN80_EXPERTS];
    let mut moe_out = vec![0.0f32; total_tokens * h];

    for layer_idx in 0..QWEN80_LAYERS {
        let load_t0 = Instant::now();
        let mut layer = LoadedLayer::load(index, layer_idx)?;
        telem.load_secs += load_t0.elapsed().as_secs_f64();
        telem.max_layer_resident_bytes = telem.max_layer_resident_bytes.max(layer.resident_bytes);
        if let Some(cb) = on_layer.as_mut() {
            cb(layer_idx, &layer, &telem);
        }
        let comp_t0 = Instant::now();

        // token_index[t] = (probe_i, pos); flat order is the first-N retention domain.
        let mut token_index: Vec<(usize, usize)> = Vec::with_capacity(total_tokens);
        let mut flat_t = 0usize;

        // --- Phase 1: mixer per probe, collect router inputs. ---
        match layer.kind {
            LayerKind::LinearDeltaNet => {
                // Sequential WITHIN each probe (recurrence). Across probes is free
                // because each probe has independent DeltaNet state.
                for (pi, (_, tokens)) in probes.iter().enumerate() {
                    let seq_len = tokens.len();
                    if seq_len == 0 {
                        continue;
                    }
                    let hidden = &mut hiddens[pi];
                    if hidden.len() != seq_len * h {
                        return Err(model_err(format!(
                            "layer {layer_idx} probe {pi}: hidden len {} != {seq_len}*{h}",
                            hidden.len()
                        )));
                    }
                    let mut state = DeltaNetState::zero(&linear_layout)?;
                    for pos in 0..seq_len {
                        let x_in = hidden[pos * h..(pos + 1) * h].to_vec();
                        let first = deltanet_mixer_step(&layer, &x_in, &mut state, &linear_layout)?;
                        hidden[pos * h..(pos + 1) * h].copy_from_slice(&first);
                        let rin = source_qwen80_residual_rms_norm(
                            &first,
                            &layer.post_attention_layernorm,
                        )?;
                        all_router_in[flat_t * h..(flat_t + 1) * h].copy_from_slice(&rin);
                        token_index.push((pi, pos));
                        flat_t += 1;
                    }
                }
            }
            LayerKind::FullAttentionGqa => {
                for (pi, (_, tokens)) in probes.iter().enumerate() {
                    let seq_len = tokens.len();
                    if seq_len == 0 {
                        continue;
                    }
                    let hidden = &mut hiddens[pi];
                    if hidden.len() != seq_len * h {
                        return Err(model_err(format!(
                            "layer {layer_idx} probe {pi}: hidden len {} != {seq_len}*{h}",
                            hidden.len()
                        )));
                    }
                    gqa_mixer_prefill_probe(&layer, hidden, seq_len, &gqa_layout)?;
                    for pos in 0..seq_len {
                        let rin = source_qwen80_residual_rms_norm(
                            &hidden[pos * h..(pos + 1) * h],
                            &layer.post_attention_layernorm,
                        )?;
                        all_router_in[flat_t * h..(flat_t + 1) * h].copy_from_slice(&rin);
                        token_index.push((pi, pos));
                        flat_t += 1;
                    }
                }
            }
        }
        debug_assert_eq!(flat_t, total_tokens);

        // Free mixer BF16 payloads before MoE (experts still needed).
        layer.in_proj_qkvz = None;
        layer.in_proj_ba = None;
        layer.out_proj_linear = None;
        layer.q_proj = None;
        layer.k_proj = None;
        layer.v_proj = None;
        layer.o_proj = None;

        // --- Phase 2: router — widen once, per-token M=1 matvec (matches scalar). ---
        let router_w = widen_bf16_mat(&layer.router, QWEN80_EXPERTS, h)?;
        layer.router = Vec::new();
        for t in 0..total_tokens {
            gemm_f32(
                &router_w,
                QWEN80_EXPERTS,
                h,
                &all_router_in[t * h..(t + 1) * h],
                1,
                &mut router_logits[t * QWEN80_EXPERTS..(t + 1) * QWEN80_EXPERTS],
            )?;
        }
        drop(router_w);

        let mut routes: Vec<(Vec<u32>, Vec<f32>)> = Vec::with_capacity(total_tokens);
        let mut expert_members: Vec<Vec<(usize, f32)>> = vec![Vec::new(); QWEN80_EXPERTS];
        for t in 0..total_tokens {
            let route = source_qwen80_topk_router(
                &router_logits[t * QWEN80_EXPERTS..(t + 1) * QWEN80_EXPERTS],
            )?;
            let ids: Vec<u32> = route.ids.iter().map(|&id| id as u32).collect();
            let weights = route.weights.to_vec();
            for (&e, &w) in ids.iter().zip(weights.iter()) {
                expert_members[e as usize].push((t, w));
            }
            routes.push((ids, weights));
        }

        // --- Phase 3: shared expert (compute only) + parallel routed GEMMs,
        //     then residual combine in scalar order: routed (route-slot order)
        //     then shared*gate. ---
        moe_out.fill(0.0);
        let (shared_down, gate_vals) =
            shared_expert_batched(&layer, &all_router_in, total_tokens, h, shared_inter)?;
        // Drop shared BF16 after use.
        layer.shared_gate = Vec::new();
        layer.shared_up = Vec::new();
        layer.shared_down = Vec::new();
        layer.shared_expert_gate = Vec::new();

        moe_routed_experts_parallel(
            &mut layer,
            &expert_members,
            &routes,
            &all_router_in,
            total_tokens,
            h,
            inter,
            &mut moe_out,
        )?;
        // shared * sigmoid(gate) last — matches scalar moe_combine.
        for t in 0..total_tokens {
            let g = gate_vals[t];
            let src = &shared_down[t * h..(t + 1) * h];
            let dst = &mut moe_out[t * h..(t + 1) * h];
            for j in 0..h {
                dst[j] += src[j] * g;
            }
        }

        // Apply MoE residual, then record captures. Hidden retention is decided
        // here, after top-k routing is known: first N tokens (global order) that
        // route to expert E keep their router-input hidden for E's organ fit. A
        // token is kept if any of its top-k experts still has an open slot;
        // co-routing may therefore give popular experts more than N rows, which
        // is a floor-not-ceiling. Route membership is always recorded.
        // expert_retained is sized from QWEN80_EXPERTS (512), never a Q30 constant.
        for (t, &(pi, pos)) in token_index.iter().enumerate() {
            add_inplace(
                &mut hiddens[pi][pos * h..(pos + 1) * h],
                &moe_out[t * h..(t + 1) * h],
            );
            if hiddens[pi][pos * h..(pos + 1) * h]
                .iter()
                .any(|v| !v.is_finite())
            {
                return Err(model_err(format!(
                    "layer {layer_idx} probe {pi} pos {pos} second residual non-finite"
                )));
            }
        }
        append_retained_layer_captures(
            &mut captures,
            &token_index,
            &mut routes,
            &all_router_in,
            layer_idx,
            max_hidden_tokens_per_expert,
        )?;
        // Write this layer's retained rows (caller) and drop the payloads
        // before the next layer's weights load.
        if let Some(cb) = on_layer_flush.as_mut() {
            cb(layer_idx, &mut captures)?;
        }
        release_layer_retained_hiddens(&mut captures, layer_idx);
        debug_assert_eq!(resident_retained_hidden_bytes(&captures), 0);

        telem.compute_secs += comp_t0.elapsed().as_secs_f64();
        drop(layer);
        telem.peak_rss_bytes = telem.peak_rss_bytes.max(peak_rss_bytes());
        if telem.peak_rss_bytes > STREAMED_PEAK_RSS_HARD_CAP_BYTES {
            return Err(model_err(format!(
                "peak RSS {} exceeds streamed hard cap {}; refuse (looks like resident load)",
                telem.peak_rss_bytes, STREAMED_PEAK_RSS_HARD_CAP_BYTES
            )));
        }
    }
    telem.weight_bytes_read = index.bytes_read_total().saturating_sub(bytes0);
    telem.wall_secs = wall0.elapsed().as_secs_f64();
    telem.peak_rss_bytes = telem.peak_rss_bytes.max(peak_rss_bytes());
    Ok((captures, telem))
}

/// Stateful single-token step through all 48 layers (one full weight stream).
/// Used by the autoregressive generation loop after prompt prefill.
fn decode_one_token_stream(
    index: &SourceBf16Index,
    mut hidden: Vec<f32>,
    position: usize,
    max_seq: usize,
    delta_states: &mut [Option<DeltaNetState>],
    gqa_states: &mut [Option<GqaState>],
) -> Result<Vec<f32>> {
    if hidden.len() != QWEN80_HIDDEN {
        return Err(model_err("decode hidden width mismatch"));
    }
    let linear_layout = Qwen80CanonicalLinearDeltaNetLayout::source_exact();
    linear_layout.validate()?;
    let gqa_layout = Qwen80CanonicalGqaLayout::source_exact();
    gqa_layout.validate()?;

    let mut router_logits = vec![0.0f32; QWEN80_EXPERTS];
    let mut gate = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
    let mut up = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
    let mut act = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
    let mut down = vec![0.0f32; QWEN80_HIDDEN];
    let mut shared_gate = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
    let mut shared_up = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
    let mut shared_act = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
    let mut shared_down = vec![0.0f32; QWEN80_HIDDEN];
    let mut shared_gate_logit = vec![0.0f32; 1];
    let mut moe_combined = vec![0.0f32; QWEN80_HIDDEN];

    for layer_idx in 0..QWEN80_LAYERS {
        let layer = LoadedLayer::load(index, layer_idx)?;
        let first_residual = match layer.kind {
            LayerKind::LinearDeltaNet => {
                let state = delta_states[layer_idx]
                    .get_or_insert_with(|| DeltaNetState::zero(&linear_layout).expect("layout ok"));
                deltanet_mixer_step(&layer, &hidden, state, &linear_layout)?
            }
            LayerKind::FullAttentionGqa => {
                let state = gqa_states[layer_idx]
                    .get_or_insert_with(|| GqaState::new(max_seq, &gqa_layout));
                gqa_mixer_step(&layer, &hidden, state, position, &gqa_layout)?
            }
        };
        let router_input =
            source_qwen80_residual_rms_norm(&first_residual, &layer.post_attention_layernorm)?;
        let _ = moe_combine(
            &layer,
            &router_input,
            &mut moe_combined,
            &mut router_logits,
            &mut gate,
            &mut up,
            &mut act,
            &mut down,
            &mut shared_gate,
            &mut shared_up,
            &mut shared_act,
            &mut shared_down,
            &mut shared_gate_logit,
        )?;
        hidden = first_residual;
        add_inplace(&mut hidden, &moe_combined);
        drop(layer);
        if peak_rss_bytes() > STREAMED_PEAK_RSS_HARD_CAP_BYTES {
            return Err(model_err(format!(
                "peak RSS {} exceeds streamed hard cap during decode",
                peak_rss_bytes()
            )));
        }
    }
    Ok(hidden)
}

/// Prefill a prompt by processing every position layer-major (one weight stream).
fn prefill_prompt_stream(
    index: &SourceBf16Index,
    token_ids: &[u32],
    max_seq: usize,
    delta_states: &mut [Option<DeltaNetState>],
    gqa_states: &mut [Option<GqaState>],
) -> Result<Vec<f32>> {
    let probes = vec![("prefill".to_string(), token_ids.to_vec())];
    let mut hiddens = embed_probes(index, &probes)?;
    let linear_layout = Qwen80CanonicalLinearDeltaNetLayout::source_exact();
    linear_layout.validate()?;
    let gqa_layout = Qwen80CanonicalGqaLayout::source_exact();
    gqa_layout.validate()?;

    for layer_idx in 0..QWEN80_LAYERS {
        let layer = LoadedLayer::load(index, layer_idx)?;
        // Process the single probe sequence, capturing states for decode.
        let seq_len = token_ids.len();
        let hidden = &mut hiddens[0];
        match layer.kind {
            LayerKind::LinearDeltaNet => {
                let state = delta_states[layer_idx]
                    .get_or_insert_with(|| DeltaNetState::zero(&linear_layout).expect("layout ok"));
                // Re-run with state we keep (forward_layer_probe uses local state).
                let mut router_logits = vec![0.0f32; QWEN80_EXPERTS];
                let mut gate = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
                let mut up = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
                let mut act = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
                let mut down = vec![0.0f32; QWEN80_HIDDEN];
                let mut shared_gate = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
                let mut shared_up = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
                let mut shared_act = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
                let mut shared_down = vec![0.0f32; QWEN80_HIDDEN];
                let mut shared_gate_logit = vec![0.0f32; 1];
                let mut moe_combined = vec![0.0f32; QWEN80_HIDDEN];
                for pos in 0..seq_len {
                    let x_in = hidden[pos * QWEN80_HIDDEN..(pos + 1) * QWEN80_HIDDEN].to_vec();
                    let first = deltanet_mixer_step(&layer, &x_in, state, &linear_layout)?;
                    let rin =
                        source_qwen80_residual_rms_norm(&first, &layer.post_attention_layernorm)?;
                    let _ = moe_combine(
                        &layer,
                        &rin,
                        &mut moe_combined,
                        &mut router_logits,
                        &mut gate,
                        &mut up,
                        &mut act,
                        &mut down,
                        &mut shared_gate,
                        &mut shared_up,
                        &mut shared_act,
                        &mut shared_down,
                        &mut shared_gate_logit,
                    )?;
                    let mut out = first;
                    add_inplace(&mut out, &moe_combined);
                    hidden[pos * QWEN80_HIDDEN..(pos + 1) * QWEN80_HIDDEN].copy_from_slice(&out);
                }
            }
            LayerKind::FullAttentionGqa => {
                let state = gqa_states[layer_idx]
                    .get_or_insert_with(|| GqaState::new(max_seq, &gqa_layout));
                let mut router_logits = vec![0.0f32; QWEN80_EXPERTS];
                let mut gate = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
                let mut up = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
                let mut act = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
                let mut down = vec![0.0f32; QWEN80_HIDDEN];
                let mut shared_gate = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
                let mut shared_up = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
                let mut shared_act = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
                let mut shared_down = vec![0.0f32; QWEN80_HIDDEN];
                let mut shared_gate_logit = vec![0.0f32; 1];
                let mut moe_combined = vec![0.0f32; QWEN80_HIDDEN];
                for pos in 0..seq_len {
                    let x_in = hidden[pos * QWEN80_HIDDEN..(pos + 1) * QWEN80_HIDDEN].to_vec();
                    let first = gqa_mixer_step(&layer, &x_in, state, pos, &gqa_layout)?;
                    let rin =
                        source_qwen80_residual_rms_norm(&first, &layer.post_attention_layernorm)?;
                    let _ = moe_combine(
                        &layer,
                        &rin,
                        &mut moe_combined,
                        &mut router_logits,
                        &mut gate,
                        &mut up,
                        &mut act,
                        &mut down,
                        &mut shared_gate,
                        &mut shared_up,
                        &mut shared_act,
                        &mut shared_down,
                        &mut shared_gate_logit,
                    )?;
                    let mut out = first;
                    add_inplace(&mut out, &moe_combined);
                    hidden[pos * QWEN80_HIDDEN..(pos + 1) * QWEN80_HIDDEN].copy_from_slice(&out);
                }
            }
        }
        drop(layer);
        if peak_rss_bytes() > STREAMED_PEAK_RSS_HARD_CAP_BYTES {
            return Err(model_err(format!(
                "peak RSS {} exceeds streamed hard cap during prefill",
                peak_rss_bytes()
            )));
        }
    }
    let h = &hiddens[0];
    let n = h.len() / QWEN80_HIDDEN;
    Ok(h[(n - 1) * QWEN80_HIDDEN..n * QWEN80_HIDDEN].to_vec())
}

/// Greedy decode with the source one-user chat template.
pub fn greedy_decode_user_prompt(
    index: &SourceBf16Index,
    tokenizer_path: &Path,
    user_text: &str,
    max_new_tokens: usize,
) -> Result<GreedyDecodeResult> {
    use tokenizers::Tokenizer;

    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
        model_err(format!(
            "cannot load tokenizer {}: {e}",
            tokenizer_path.display()
        ))
    })?;
    // Source chat template for a single user turn without tools/system:
    // <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
    let rendered = format!("<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n");
    let encoding = tokenizer
        .encode(rendered.as_str(), false)
        .map_err(|e| model_err(format!("tokenizer encode failed: {e}")))?;
    let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();
    if token_ids.is_empty() {
        return Err(model_err("chat-template encoding produced no tokens"));
    }
    let prompt_len = token_ids.len();
    let max_seq = prompt_len + max_new_tokens + 8;
    let mut delta_states: Vec<Option<DeltaNetState>> = (0..QWEN80_LAYERS).map(|_| None).collect();
    let mut gqa_states: Vec<Option<GqaState>> = (0..QWEN80_LAYERS).map(|_| None).collect();

    let mut last_hidden = prefill_prompt_stream(
        index,
        &token_ids,
        max_seq,
        &mut delta_states,
        &mut gqa_states,
    )?;
    let mut generated = Vec::new();
    let eos = [151645u32, 151643u32];

    for step in 0..max_new_tokens {
        let logits = logits_from_final_hidden(index, &last_hidden)?;
        let next = argmax_f32(&logits);
        generated.push(next);
        if eos.contains(&next) {
            break;
        }
        token_ids.push(next);
        let position = prompt_len + step;
        let emb = index.embed_row(next)?;
        last_hidden = decode_one_token_stream(
            index,
            emb,
            position,
            max_seq,
            &mut delta_states,
            &mut gqa_states,
        )?;
    }

    let cont_text = tokenizer
        .decode(&generated, true)
        .map_err(|e| model_err(format!("tokenizer decode failed: {e}")))?;
    Ok(GreedyDecodeResult {
        prompt_token_count: prompt_len,
        generated_token_ids: generated,
        continuation_text: cont_text,
        rendered_prompt: rendered,
        peak_rss_bytes: peak_rss_bytes(),
        weight_bytes_read: index.bytes_read_total(),
    })
}

#[derive(Clone, Debug)]
pub struct GreedyDecodeResult {
    pub prompt_token_count: usize,
    pub generated_token_ids: Vec<u32>,
    pub continuation_text: String,
    pub rendered_prompt: String,
    pub peak_rss_bytes: u64,
    pub weight_bytes_read: u64,
}

/// True when the top-1 continuation is a coherent capital-of-France answer.
pub fn is_coherent_paris_continuation(text: &str) -> bool {
    let t = text.trim_start();
    if t.contains("Wien") || t.to_ascii_lowercase().contains("swiper") {
        return false;
    }
    // Degenerate single-token loops / pure punctuation
    if t.len() > 8 {
        let first = t.chars().next().unwrap_or('\0');
        if t.chars().filter(|c| *c == first).count() > t.len() * 3 / 4 {
            return false;
        }
    }
    let lower = t.to_ascii_lowercase();
    lower.starts_with("paris")
        || lower.starts_with(" paris")
        || lower.starts_with("the capital of france is paris")
        || lower.starts_with("**paris")
        || lower.contains("paris is the capital")
}

/// Process peak RSS in bytes (macOS: `ru_maxrss` is already bytes).
pub fn peak_rss_bytes() -> u64 {
    #[cfg(unix)]
    {
        #[cfg(target_os = "macos")]
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct TimeVal {
            tv_sec: i64,
            tv_usec: i32,
            _pad: i32,
        }
        #[cfg(not(target_os = "macos"))]
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct TimeVal {
            tv_sec: i64,
            tv_usec: i64,
        }
        #[repr(C)]
        struct Rusage {
            ru_utime: TimeVal,
            ru_stime: TimeVal,
            ru_maxrss: i64,
            ru_ixrss: i64,
            ru_idrss: i64,
            ru_isrss: i64,
            ru_minflt: i64,
            ru_majflt: i64,
            _pad: [i64; 8],
        }
        extern "C" {
            fn getrusage(who: i32, usage: *mut Rusage) -> i32;
        }
        const RUSAGE_SELF: i32 = 0;
        #[cfg(target_os = "macos")]
        let zero_tv = TimeVal {
            tv_sec: 0,
            tv_usec: 0,
            _pad: 0,
        };
        #[cfg(not(target_os = "macos"))]
        let zero_tv = TimeVal {
            tv_sec: 0,
            tv_usec: 0,
        };
        let mut u = Rusage {
            ru_utime: zero_tv,
            ru_stime: zero_tv,
            ru_maxrss: 0,
            ru_ixrss: 0,
            ru_idrss: 0,
            ru_isrss: 0,
            ru_minflt: 0,
            ru_majflt: 0,
            _pad: [0; 8],
        };
        let rc = unsafe { getrusage(RUSAGE_SELF, &mut u) };
        if rc != 0 {
            return 0;
        }
        u.ru_maxrss.max(0) as u64
    }
    #[cfg(not(unix))]
    {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_kinds_match_source_interval() {
        assert_eq!(layer_kind(0).unwrap(), LayerKind::LinearDeltaNet);
        assert_eq!(layer_kind(2).unwrap(), LayerKind::LinearDeltaNet);
        assert_eq!(layer_kind(3).unwrap(), LayerKind::FullAttentionGqa);
        assert_eq!(layer_kind(7).unwrap(), LayerKind::FullAttentionGqa);
        assert_eq!(layer_kind(47).unwrap(), LayerKind::FullAttentionGqa);
        let gqa: Vec<_> = (0..48)
            .filter(|&l| layer_kind(l).unwrap() == LayerKind::FullAttentionGqa)
            .collect();
        assert_eq!(gqa, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]);
    }

    #[test]
    fn paris_coherence_accepts_variants() {
        assert!(is_coherent_paris_continuation(" Paris"));
        assert!(is_coherent_paris_continuation("Paris is the capital"));
        assert!(!is_coherent_paris_continuation(" Wien swiper swiper"));
        assert!(!is_coherent_paris_continuation("swiper Wien"));
    }

    #[test]
    fn layouts_validate() {
        Qwen80CanonicalLinearDeltaNetLayout::source_exact()
            .validate()
            .unwrap();
        Qwen80CanonicalGqaLayout::source_exact().validate().unwrap();
    }

    #[test]
    fn per_expert_first_n_retention_is_deterministic_and_bounded() {
        // Synthetic routes against Q80's expert table size (512) and top-10
        // shape: tokens cycle through disjoint expert pairs so each expert's
        // first-N set is unambiguous. (Only 8 experts are touched; the rest
        // stay at zero — bookkeeping is still sized from QWEN80_EXPERTS.)
        let max_n = 3usize;
        let mut counts = vec![0usize; QWEN80_EXPERTS];
        let routes: Vec<Vec<u32>> = (0..20u32).map(|t| vec![t % 4, 4 + (t % 4)]).collect();
        let mut retained_mask = Vec::new();
        for ids in &routes {
            retained_mask.push(credit_expert_first_n_retention(&mut counts, ids, max_n));
        }
        // Touched experts 0..7 should have been credited exactly max_n times.
        for e in 0..8 {
            assert_eq!(counts[e], max_n, "expert {e} retained count");
        }
        for e in 8..QWEN80_EXPERTS {
            assert_eq!(counts[e], 0, "untouched expert {e}");
        }
        // First 3 tokens that hit expert 0 (t=0,4,8) must be retained; later ones
        // that only hit already-full experts must not.
        assert!(retained_mask[0]);
        assert!(retained_mask[4]);
        assert!(retained_mask[8]);
        // After experts 0 and 4 are full (3 credits each from t=0,4,8), token 12
        // routes to the same pair and must be dropped.
        assert!(!retained_mask[12]);
        // Re-running the same sequence must produce the same mask (determinism).
        let mut counts2 = vec![0usize; QWEN80_EXPERTS];
        let mask2: Vec<bool> = routes
            .iter()
            .map(|ids| credit_expert_first_n_retention(&mut counts2, ids, max_n))
            .collect();
        assert_eq!(retained_mask, mask2);
        assert_eq!(counts, counts2);
    }

    #[test]
    fn per_expert_first_n_zero_retains_nothing() {
        let mut counts = vec![0usize; QWEN80_EXPERTS];
        // top-10 shaped id list (Q80 routing width), still zero when N=0.
        let ids: Vec<u32> = (0..QWEN80_TOP_K as u32).collect();
        assert!(!credit_expert_first_n_retention(&mut counts, &ids, 0));
        assert!(counts.iter().all(|&c| c == 0));
    }

    fn empty_captures(probes: &[(String, Vec<u32>)]) -> Vec<Vec<Vec<LayerTokenCapture>>> {
        probes
            .iter()
            .map(|(_, toks)| (0..toks.len()).map(|_| Vec::new()).collect())
            .collect()
    }

    /// Synthetic top-10 routes + distinct hidden rows for a small probe set.
    /// Hidden row `t` is filled with `layer as f32 + t as f32 / 1000` so two
    /// runs at the same N are byte-comparable and layers are not interchangeable.
    fn synthetic_layer_inputs(
        token_count: usize,
        layer_idx: usize,
    ) -> (Vec<(usize, usize)>, Vec<(Vec<u32>, Vec<f32>)>, Vec<f32>) {
        let token_index: Vec<(usize, usize)> = (0..token_count).map(|pos| (0, pos)).collect();
        let routes: Vec<(Vec<u32>, Vec<f32>)> = (0..token_count)
            .map(|t| {
                let ids: Vec<u32> = (0..QWEN80_TOP_K as u32)
                    .map(|k| (t as u32 * 3 + k) % QWEN80_EXPERTS as u32)
                    .collect();
                let weights = vec![0.1f32; QWEN80_TOP_K];
                (ids, weights)
            })
            .collect();
        let mut all_router_in = vec![0.0f32; token_count * QWEN80_HIDDEN];
        for t in 0..token_count {
            let fill = layer_idx as f32 + (t as f32) / 1000.0;
            for x in all_router_in[t * QWEN80_HIDDEN..(t + 1) * QWEN80_HIDDEN].iter_mut() {
                *x = fill;
            }
        }
        (token_index, routes, all_router_in)
    }

    fn write_layer_if_retained(
        output_dir: &std::path::Path,
        probe_id: &str,
        token_count: usize,
        layer_idx: usize,
        captures: &[Vec<Vec<LayerTokenCapture>>],
    ) -> Result<()> {
        for pos in 0..token_count {
            let cap = &captures[0][pos][layer_idx];
            if !cap.hidden_retained {
                continue;
            }
            let rel = retained_hidden_relative_path(layer_idx, probe_id, pos);
            write_retained_hidden_f32le(&output_dir.join(rel), &cap.router_input_hidden)?;
        }
        Ok(())
    }

    #[test]
    fn per_layer_budget_guard_admits_384_and_512_and_refuses_over_cap() {
        let cap = STREAMED_PEAK_RSS_HARD_CAP_BYTES as usize;
        let bytes_per_n = QWEN80_EXPERTS
            .saturating_mul(QWEN80_HIDDEN)
            .saturating_mul(4);
        assert!(bytes_per_n > 0);
        let max_admitted = cap / bytes_per_n;
        let before_layers_factor_ceiling = cap
            / (QWEN80_EXPERTS
                .saturating_mul(QWEN80_LAYERS)
                .saturating_mul(QWEN80_HIDDEN)
                .saturating_mul(4));

        assert!(
            max_hidden_tokens_per_expert_within_streamed_cap(384),
            "N=384 must fit the per-layer budget"
        );
        assert!(
            max_hidden_tokens_per_expert_within_streamed_cap(512),
            "N=512 must fit the per-layer budget"
        );
        assert!(
            max_hidden_tokens_per_expert_within_streamed_cap(max_admitted),
            "derived boundary N={max_admitted} must be admitted (budget == floor(cap/bytes_per_n))"
        );
        assert!(
            !max_hidden_tokens_per_expert_within_streamed_cap(max_admitted + 1),
            "N={} must exceed the per-layer cap",
            max_admitted + 1
        );
        assert!(
            !max_hidden_tokens_per_expert_within_streamed_cap(0),
            "N=0 is not a valid quota"
        );
        // The old ×48 guard capped N at 85. Per-layer must raise that ceiling.
        assert!(
            max_admitted > before_layers_factor_ceiling,
            "per-layer ceiling {max_admitted} should exceed the retired ×layers ceiling {before_layers_factor_ceiling}"
        );
        assert!(
            max_admitted >= 512,
            "per-layer ceiling {max_admitted} must admit the rank-192 fit target N=512"
        );
        assert_eq!(
            worst_case_retained_hidden_bytes_per_layer(max_admitted),
            max_admitted * bytes_per_n
        );
        assert!(worst_case_retained_hidden_bytes_per_layer(max_admitted) <= cap);
        assert!(worst_case_retained_hidden_bytes_per_layer(max_admitted + 1) > cap);
        // Constants this task must not change.
        assert_eq!(DEFAULT_MAX_HIDDEN_TOKENS_PER_EXPERT, 64);
        assert_eq!(STREAMED_PEAK_RSS_HARD_CAP_BYTES, 16 * 1024 * 1024 * 1024);
    }

    #[test]
    fn retained_hiddens_do_not_accumulate_across_layers() {
        // Structural: after each layer is appended, only that layer's hidden
        // payloads are resident; after flush they are gone. A test that only
        // inspected the final on-disk tree would pass even if this release
        // were missing.
        let token_count = 24usize;
        let n_layers = 8usize;
        let n = 4usize;
        let probes = vec![("probe0".to_string(), vec![1u32; token_count])];
        let mut captures = empty_captures(&probes);
        let mut after_append = Vec::with_capacity(n_layers);
        for layer_idx in 0..n_layers {
            let (token_index, mut routes, all_router_in) =
                synthetic_layer_inputs(token_count, layer_idx);
            append_retained_layer_captures(
                &mut captures,
                &token_index,
                &mut routes,
                &all_router_in,
                layer_idx,
                n,
            )
            .expect("append");
            let resident = resident_retained_hidden_bytes(&captures);
            after_append.push(resident);
            // Only this layer may hold hidden payloads.
            for probe in &captures {
                for token in probe {
                    for cap in token {
                        if cap.layer != layer_idx {
                            assert!(
                                cap.router_input_hidden.is_empty(),
                                "layer {} still resident while appending layer {layer_idx}",
                                cap.layer
                            );
                        }
                    }
                }
            }
            let freed = release_layer_retained_hiddens(&mut captures, layer_idx);
            assert!(freed > 0, "layer {layer_idx} should have retained rows");
            assert_eq!(
                resident_retained_hidden_bytes(&captures),
                0,
                "layer {layer_idx} hiddens must be freed before the next layer"
            );
            // Routes stay complete for every token at every finished layer.
            for token in &captures[0] {
                assert_eq!(token.len(), layer_idx + 1);
                let cap = &token[layer_idx];
                assert_eq!(cap.selected_expert_ids.len(), QWEN80_TOP_K);
                assert_eq!(cap.normalized_route_weights.len(), QWEN80_TOP_K);
                assert!(cap.router_input_hidden.is_empty());
            }
        }
        // Equal per-layer residency (same token count, same N) — must not grow
        // with L. If release were skipped, after_append[L] would be (L+1)×base.
        let base = after_append[0];
        assert!(base > 0);
        for (layer_idx, &bytes) in after_append.iter().enumerate() {
            assert_eq!(
                bytes, base,
                "resident hidden bytes after append of layer {layer_idx} grew with L"
            );
        }
        assert!(
            base < n_layers.saturating_mul(base) / 2 + 1,
            "sanity: one-layer footprint must be far below the accumulated {n_layers}-layer total"
        );
    }

    #[test]
    fn same_probe_set_same_n_is_byte_identical_across_two_runs() {
        let token_count = 16usize;
        let n_layers = 4usize;
        let n = 3usize;
        let probe_id = "det0";
        let probes = vec![(probe_id.to_string(), vec![7u32; token_count])];

        let run = |dir: &std::path::Path| {
            let mut captures = empty_captures(&probes);
            for layer_idx in 0..n_layers {
                let (token_index, mut routes, all_router_in) =
                    synthetic_layer_inputs(token_count, layer_idx);
                append_retained_layer_captures(
                    &mut captures,
                    &token_index,
                    &mut routes,
                    &all_router_in,
                    layer_idx,
                    n,
                )
                .expect("append");
                let mut flush = |layer: usize, caps: &mut [Vec<Vec<LayerTokenCapture>>]| {
                    write_layer_if_retained(dir, probe_id, token_count, layer, caps)
                };
                flush_and_release_layer_hiddens(&mut captures, layer_idx, Some(&mut flush))
                    .expect("flush");
            }
            captures
        };

        let dir_a = tempfile::tempdir().expect("tempdir a");
        let dir_b = tempfile::tempdir().expect("tempdir b");
        let caps_a = run(dir_a.path());
        let caps_b = run(dir_b.path());

        // Route membership + retain mask identical.
        assert_eq!(caps_a.len(), caps_b.len());
        for (pa, pb) in caps_a.iter().zip(caps_b.iter()) {
            for (ta, tb) in pa.iter().zip(pb.iter()) {
                assert_eq!(ta.len(), tb.len());
                for (ca, cb) in ta.iter().zip(tb.iter()) {
                    assert_eq!(ca.layer, cb.layer);
                    assert_eq!(ca.selected_expert_ids, cb.selected_expert_ids);
                    assert_eq!(ca.normalized_route_weights, cb.normalized_route_weights);
                    assert_eq!(ca.hidden_retained, cb.hidden_retained);
                    assert!(ca.router_input_hidden.is_empty());
                    assert!(cb.router_input_hidden.is_empty());
                }
            }
        }

        // On-disk retained rows are byte-identical.
        let collect = |root: &std::path::Path| -> Vec<(String, Vec<u8>)> {
            let mut files = Vec::new();
            for layer in 0..n_layers {
                for pos in 0..token_count {
                    let rel = retained_hidden_relative_path(layer, probe_id, pos);
                    let path = root.join(&rel);
                    if path.is_file() {
                        files.push((rel, std::fs::read(path).expect("read hidden")));
                    }
                }
            }
            files.sort_by(|a, b| a.0.cmp(&b.0));
            files
        };
        let files_a = collect(dir_a.path());
        let files_b = collect(dir_b.path());
        assert!(!files_a.is_empty(), "expected retained hidden files");
        assert_eq!(files_a, files_b);
    }

    #[test]
    fn n384_and_n512_synthetic_flush_runs_and_records_peak_rss() {
        // Reduced configuration: no 148 GiB source. Same retain+flush+write
        // path the capture uses, at the N values the CLI must now admit.
        for &n in &[384usize, 512usize] {
            assert!(
                max_hidden_tokens_per_expert_within_streamed_cap(n),
                "N={n} must be admitted"
            );
            let token_count = 8usize;
            let n_layers = 4usize;
            let probe_id = "nrun";
            let probes = vec![(probe_id.to_string(), vec![3u32; token_count])];
            let dir = tempfile::tempdir().expect("tempdir");
            let mut captures = empty_captures(&probes);
            eprintln!(
                "{}",
                format_capture_progress(probes.len(), token_count, n, 0)
            );
            for layer_idx in 0..n_layers {
                let (token_index, mut routes, all_router_in) =
                    synthetic_layer_inputs(token_count, layer_idx);
                append_retained_layer_captures(
                    &mut captures,
                    &token_index,
                    &mut routes,
                    &all_router_in,
                    layer_idx,
                    n,
                )
                .expect("append");
                let resident = resident_retained_hidden_bytes(&captures);
                assert!(
                    resident <= worst_case_retained_hidden_bytes_per_layer(n),
                    "N={n} layer {layer_idx} resident {resident} exceeded per-layer budget"
                );
                let mut flush = |layer: usize, caps: &mut [Vec<Vec<LayerTokenCapture>>]| {
                    write_layer_if_retained(dir.path(), probe_id, token_count, layer, caps)
                };
                flush_and_release_layer_hiddens(&mut captures, layer_idx, Some(&mut flush))
                    .expect("flush");
                assert_eq!(resident_retained_hidden_bytes(&captures), 0);
            }
            let peak = peak_rss_bytes();
            eprintln!(
                "synthetic N={n} peak_rss_bytes={peak} ({:.3} GiB); hard cap {}",
                peak as f64 / (1024.0 * 1024.0 * 1024.0),
                STREAMED_PEAK_RSS_HARD_CAP_BYTES
            );
            assert!(
                peak <= STREAMED_PEAK_RSS_HARD_CAP_BYTES,
                "synthetic N={n} peak RSS {peak} exceeds streamed hard cap"
            );
            // Every token still has complete top-k routes for every layer.
            for token in &captures[0] {
                assert_eq!(token.len(), n_layers);
                for cap in token {
                    assert_eq!(cap.selected_expert_ids.len(), QWEN80_TOP_K);
                }
            }
        }
    }

    /// Manual gate: set `QWEN80_SOURCE_DIR` and run with `--ignored`.
    /// Compares scalar [`forward_layer_probe`] routes vs batched [`capture_all_layers`]
    /// on one short probe for every layer (bitwise expert-id identity).
    #[test]
    #[ignore = "requires QWEN80_SOURCE_DIR and ~minutes of weight streaming"]
    fn route_membership_scalar_vs_batched_identity() {
        let dir = std::path::PathBuf::from(
            std::env::var("QWEN80_SOURCE_DIR").expect("QWEN80_SOURCE_DIR"),
        );
        let index = SourceBf16Index::open(&dir).expect("open source");
        // Fixed short chat-template-ish token sequence (15 tokens).
        let probes = vec![(
            "route_check".to_string(),
            vec![
                151644u32, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644,
                77091, 198,
            ],
        )];
        let seq = probes[0].1.len();

        let mut h_scalar = embed_probes(&index, &probes).expect("embed scalar");
        let mut scalar_by_layer_token: Vec<Vec<Vec<u32>>> =
            vec![vec![Vec::new(); seq]; QWEN80_LAYERS];
        for layer_idx in 0..QWEN80_LAYERS {
            let layer = LoadedLayer::load(&index, layer_idx).expect("load scalar");
            let caps =
                forward_layer_probe(&layer, &mut h_scalar[0], seq, 0, 1).expect("scalar forward");
            for (pos, cap) in caps.into_iter().enumerate() {
                scalar_by_layer_token[layer_idx][pos] = cap.selected_expert_ids;
            }
            drop(layer);
            eprintln!("route_check scalar layer {layer_idx}/{QWEN80_LAYERS}");
        }

        let mut h_batch = embed_probes(&index, &probes).expect("embed batch");
        // N large enough that short-probe retention does not drop routes' tokens;
        // identity check is on expert ids only.
        let (batch_caps, _) = capture_all_layers(
            &index,
            &probes,
            &mut h_batch,
            DEFAULT_MAX_HIDDEN_TOKENS_PER_EXPERT,
            None,
            None,
        )
        .expect("batched");

        let mut mismatches = 0usize;
        for layer_idx in 0..QWEN80_LAYERS {
            for pos in 0..seq {
                let b = &batch_caps[0][pos][layer_idx].selected_expert_ids;
                let s = &scalar_by_layer_token[layer_idx][pos];
                if b != s {
                    mismatches += 1;
                    if mismatches <= 8 {
                        eprintln!(
                            "ROUTE MISMATCH layer={layer_idx} pos={pos} scalar={s:?} batch={b:?}"
                        );
                    }
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "route membership not bitwise identical ({mismatches} token-layers)"
        );
        eprintln!("ROUTE IDENTITY PASS: {QWEN80_LAYERS} layers × {seq} tokens");
    }
}

//! Layer-tile batched execution for DSV4F activation-X capture.
//!
//! The driver stays layer-major. Inside a layer the token dimension is a
//! GEMM (dense organs) or a per-expert grouped GEMM (routed). Grouping is
//! the inverse of `dsv4f_pack_worklist`: that kernel sorts one token's top-6
//! by (expert_id, slot); here we invert the same pairs into per-expert token
//! rows and issue one GEMM per nonempty expert. Retention is applied later
//! by the unchanged first-N writer.

use std::collections::BTreeMap;
use std::time::Instant;

use half::bf16;

use crate::gravity_deepseek_v4::DeepSeekV4FullStreamReader;
use crate::gravity_deepseek_v4_act_quant::{
    act_quant_bf16_ue8m0, decode_e4m3fn, decode_e8m0fnu, fp8_e4m3fn_ue8m0_matvec, ACT_QUANT_BLOCK,
};
use crate::gravity_deepseek_v4_layer0_attention::{
    hc_attn_post_source_algorithm, kv_non_rope_inplace_qat_source_algorithm,
    per_head_rms_norm_bf16_source_algorithm, position_zero_rope_identity,
    rms_norm_bf16_source_algorithm, sparse_attention_position_zero_source_algorithm, HEAD_DIM,
    KV_QAT_BLOCK, NUM_HEADS, O_GROUPS, O_LORA_RANK, Q_LORA_RANK, ROPE_HEAD_DIM, WKV_ROWS,
    WO_A_COLS, WO_A_ROWS, WO_B_COLS, WO_B_ROWS, WQ_B_ROWS,
};
use crate::gravity_deepseek_v4_layer0_moe::{
    fp4_e2m1fn_x2_ue8m0_matvec, swiglu_bf16_source_algorithm, ACTIVATED_EXPERTS, FP4_BLOCK,
    MOE_INTER_DIM, ROUTED_EXPERTS,
};
use crate::gravity_deepseek_v4_layer0_prefix::{
    hc_attn_pre_source_algorithm, HC_EPS, HC_FLAT_WIDTH, HC_MIX_WIDTH, HC_SINKHORN_ITERS,
    HIDDEN_SIZE, RMS_NORM_EPS,
};
use crate::gravity_deepseek_v4_layer_source_anchors::{
    DeepSeekV4LayerCommonTensor, DeepSeekV4LayerControlProjection, DeepSeekV4LayerExpertProjection,
    DeepSeekV4LayerGateMode, DeepSeekV4LayerMhcStage, DeepSeekV4LayerSourceAnchor,
};
use crate::gravity_deepseek_v4_streamed_native::StreamedNativeSession;
use crate::Result;

use super::{
    gravity, hash_route_from_logits, learned_bias_route, read_bf16_tracked, read_f32_tracked,
    OperatorProfile, ResidentLedger, StreamedLayerCapture,
};

use crate::gravity_deepseek_v4::NativeScalePairKind;

/// Extra tile working-set budget used to pick the default `--token-tile`.
/// Sequential capture peaked at 1.51 GiB; 2 GiB of tile activations stays
/// far under the 16 GiB RSS ceiling.
const TILE_RSS_BUDGET_BYTES: usize = 2 * 1024 * 1024 * 1024;
const PER_TOKEN_TILE_BYTES: usize = 256 * 1024;
const DEFAULT_TILE_CAP: usize = 2048;

pub fn default_token_tile(n_tokens: usize) -> usize {
    let cap = (TILE_RSS_BUDGET_BYTES / PER_TOKEN_TILE_BYTES)
        .max(32)
        .min(DEFAULT_TILE_CAP);
    n_tokens.min(cap).max(1)
}

/// Inverse of `dsv4f_pack_worklist`: one token's top-6 is sorted by
/// (expert_id, original slot). Collecting those pairs across the tile and
/// grouping by expert_id (BTreeMap = ascending id) yields the same per-token
/// combine order as the existing single-token loop.
pub fn pack_expert_token_worklists(
    selected_ids: &[Vec<u64>],
    selected_weights: &[Vec<f32>],
) -> Result<Vec<(u64, Vec<(usize, f32)>)>> {
    if selected_ids.len() != selected_weights.len() {
        return Err(gravity("expert worklist id/weight batch length mismatch"));
    }
    let mut by_expert: BTreeMap<u64, Vec<(usize, f32)>> = BTreeMap::new();
    for (token, (ids, weights)) in selected_ids.iter().zip(selected_weights).enumerate() {
        if ids.len() != ACTIVATED_EXPERTS || weights.len() != ACTIVATED_EXPERTS {
            return Err(gravity(format!(
                "token {token} route membership is {} ids / {} weights",
                ids.len(),
                weights.len()
            )));
        }
        let mut slots: Vec<usize> = (0..ACTIVATED_EXPERTS).collect();
        slots.sort_unstable_by_key(|&slot| (ids[slot], slot));
        for slot in slots {
            by_expert
                .entry(ids[slot])
                .or_default()
                .push((token, weights[slot]));
        }
    }
    Ok(by_expert.into_iter().collect())
}

/// Run one layer for a tile of independent position-0 tokens.
pub fn execute_layer_tile(
    reader: &DeepSeekV4FullStreamReader,
    layer: &DeepSeekV4LayerSourceAnchor,
    hc_in: &[Vec<u16>],
    token_ids: &[u64],
    ledger: &mut ResidentLedger,
    profile: &mut OperatorProfile,
    metal: Option<&mut StreamedNativeSession>,
) -> Result<Vec<StreamedLayerCapture>> {
    let n = hc_in.len();
    if n == 0 || n != token_ids.len() {
        return Err(gravity("layer tile token/hc length mismatch"));
    }
    for (t, hc) in hc_in.iter().enumerate() {
        if hc.len() != HC_FLAT_WIDTH {
            return Err(gravity(format!(
                "tile token {t} HC width {} != {HC_FLAT_WIDTH}",
                hc.len()
            )));
        }
    }
    let mut metal = metal;
    let attn_hc = execute_attention_tile(
        reader,
        layer,
        hc_in,
        ledger,
        profile,
        metal.as_deref_mut(),
    )?;
    execute_moe_tile(reader, layer, &attn_hc, token_ids, ledger, profile, metal)
}

fn execute_attention_tile(
    reader: &DeepSeekV4FullStreamReader,
    layer: &DeepSeekV4LayerSourceAnchor,
    hc_in: &[Vec<u16>],
    ledger: &mut ResidentLedger,
    profile: &mut OperatorProfile,
    metal: Option<&mut StreamedNativeSession>,
) -> Result<Vec<Vec<u16>>> {
    let n = hc_in.len();
    let mut metal = metal;
    let mhc = layer.mhc_binding(DeepSeekV4LayerMhcStage::Attention);
    let t_io = Instant::now();
    let hc_fn = read_f32_tracked(
        reader,
        ledger,
        &mhc.fn_tensor.name,
        HC_MIX_WIDTH * HC_FLAT_WIDTH,
    )?;
    let hc_base = read_f32_tracked(reader, ledger, &mhc.base_tensor.name, HC_MIX_WIDTH)?;
    let hc_scale = read_f32_tracked(reader, ledger, &mhc.scale_tensor.name, 3)?;
    profile.add("streaming_io", t_io.elapsed());
    let t_mhc = Instant::now();
    let mut posts = Vec::with_capacity(n);
    let mut combs = Vec::with_capacity(n);
    let mut reduceds = Vec::with_capacity(n);
    for hc in hc_in {
        let (_, _, _, post_f32, comb_f32, reduced) = hc_attn_pre_source_algorithm(
            hc,
            &hc_fn,
            &hc_scale,
            &hc_base,
            RMS_NORM_EPS,
            HC_EPS,
            HC_SINKHORN_ITERS,
        )?;
        posts.push(post_f32);
        combs.push(comb_f32);
        reduceds.push(reduced);
    }
    profile.add("mhc_attn_pre", t_mhc.elapsed());
    ledger.release(&mhc.fn_tensor.name)?;
    ledger.release(&mhc.base_tensor.name)?;
    ledger.release(&mhc.scale_tensor.name)?;
    drop(hc_fn);
    drop(hc_base);
    drop(hc_scale);

    let attn_norm = layer.common_tensor(DeepSeekV4LayerCommonTensor::AttentionNorm);
    let t_io = Instant::now();
    let attn_norm_w = read_bf16_tracked(reader, ledger, &attn_norm.name, HIDDEN_SIZE)?;
    profile.add("streaming_io", t_io.elapsed());
    let t_norm = Instant::now();
    let mut attn_norm_rows = Vec::with_capacity(n);
    for reduced in &reduceds {
        attn_norm_rows.push(rms_norm_bf16_source_algorithm(
            reduced,
            &attn_norm_w,
            HIDDEN_SIZE,
            RMS_NORM_EPS,
        )?);
    }
    profile.add("attn_norm", t_norm.elapsed());
    ledger.release(&attn_norm.name)?;
    drop(attn_norm_w);
    drop(reduceds);
    let attn_norm_packed = stack_u16(&attn_norm_rows);

    let wq_a = layer.control_pair(DeepSeekV4LayerControlProjection::WqA);
    let wq_a_out = fp8_linear_batched_tracked(
        reader,
        ledger,
        profile,
        metal.as_deref_mut(),
        "mla_wq_a",
        &wq_a.weight.name,
        &wq_a.scale.name,
        Q_LORA_RANK,
        HIDDEN_SIZE,
        &attn_norm_packed,
        n,
    )?;

    let q_norm = layer.common_tensor(DeepSeekV4LayerCommonTensor::AttentionQNorm);
    let t_io = Instant::now();
    let q_norm_w = read_bf16_tracked(reader, ledger, &q_norm.name, Q_LORA_RANK)?;
    profile.add("streaming_io", t_io.elapsed());
    let t_norm = Instant::now();
    let mut q_norm_rows = Vec::with_capacity(n);
    for t in 0..n {
        let row = &wq_a_out[t * Q_LORA_RANK..(t + 1) * Q_LORA_RANK];
        q_norm_rows.push(rms_norm_bf16_source_algorithm(
            row,
            &q_norm_w,
            Q_LORA_RANK,
            RMS_NORM_EPS,
        )?);
    }
    profile.add("q_norm", t_norm.elapsed());
    ledger.release(&q_norm.name)?;
    drop(q_norm_w);
    drop(wq_a_out);
    let q_norm_packed = stack_u16(&q_norm_rows);
    drop(q_norm_rows);

    let wq_b = layer.control_pair(DeepSeekV4LayerControlProjection::WqB);
    let wq_b_out = fp8_linear_batched_tracked(
        reader,
        ledger,
        profile,
        metal.as_deref_mut(),
        "mla_wq_b",
        &wq_b.weight.name,
        &wq_b.scale.name,
        WQ_B_ROWS,
        Q_LORA_RANK,
        &q_norm_packed,
        n,
    )?;
    drop(q_norm_packed);

    let t_head = Instant::now();
    let mut q_rope_rows = Vec::with_capacity(n);
    for t in 0..n {
        let row = &wq_b_out[t * WQ_B_ROWS..(t + 1) * WQ_B_ROWS];
        let q_head =
            per_head_rms_norm_bf16_source_algorithm(row, NUM_HEADS, HEAD_DIM, RMS_NORM_EPS)?;
        q_rope_rows.push(position_zero_rope_identity(
            &q_head,
            NUM_HEADS,
            HEAD_DIM,
            ROPE_HEAD_DIM,
        )?);
    }
    profile.add("q_head_norm_rope", t_head.elapsed());
    drop(wq_b_out);

    let wkv = layer.control_pair(DeepSeekV4LayerControlProjection::Wkv);
    let wkv_out = fp8_linear_batched_tracked(
        reader,
        ledger,
        profile,
        metal.as_deref_mut(),
        "mla_wkv",
        &wkv.weight.name,
        &wkv.scale.name,
        WKV_ROWS,
        HIDDEN_SIZE,
        &attn_norm_packed,
        n,
    )?;
    drop(attn_norm_packed);
    drop(attn_norm_rows);

    let kv_norm = layer.common_tensor(DeepSeekV4LayerCommonTensor::AttentionKvNorm);
    let t_io = Instant::now();
    let kv_norm_w = read_bf16_tracked(reader, ledger, &kv_norm.name, HEAD_DIM)?;
    profile.add("streaming_io", t_io.elapsed());
    let t_kv = Instant::now();
    let mut kv_rope_rows = Vec::with_capacity(n);
    for t in 0..n {
        let row = &wkv_out[t * WKV_ROWS..(t + 1) * WKV_ROWS];
        let kv_norm_row = rms_norm_bf16_source_algorithm(row, &kv_norm_w, HEAD_DIM, RMS_NORM_EPS)?;
        let kv_qat = kv_non_rope_inplace_qat_source_algorithm(
            &kv_norm_row,
            HEAD_DIM,
            ROPE_HEAD_DIM,
            KV_QAT_BLOCK,
        )?;
        kv_rope_rows.push(position_zero_rope_identity(
            &kv_qat.output_bf16_bits,
            1,
            HEAD_DIM,
            ROPE_HEAD_DIM,
        )?);
    }
    profile.add("kv_norm_qat_rope", t_kv.elapsed());
    ledger.release(&kv_norm.name)?;
    drop(kv_norm_w);
    drop(wkv_out);

    let sink = layer.common_tensor(DeepSeekV4LayerCommonTensor::AttentionSink);
    let t_io = Instant::now();
    let sink_f32 = read_f32_tracked(reader, ledger, &sink.name, NUM_HEADS)?;
    profile.add("streaming_io", t_io.elapsed());
    let t_attn = Instant::now();
    let mut attn_derotated = Vec::with_capacity(n);
    for t in 0..n {
        let (_, _, attn_out) = sparse_attention_position_zero_source_algorithm(
            &q_rope_rows[t],
            &kv_rope_rows[t],
            &sink_f32,
            NUM_HEADS,
            HEAD_DIM,
        )?;
        attn_derotated.push(position_zero_rope_identity(
            &attn_out,
            NUM_HEADS,
            HEAD_DIM,
            ROPE_HEAD_DIM,
        )?);
    }
    profile.add("sparse_attn_bos", t_attn.elapsed());
    ledger.release(&sink.name)?;
    drop(sink_f32);
    drop(q_rope_rows);
    drop(kv_rope_rows);
    let attn_packed = stack_u16(&attn_derotated);
    drop(attn_derotated);

    let wo_a = layer.control_pair(DeepSeekV4LayerControlProjection::WoA);
    let wo_a_out = wo_a_einsum_batched_tracked(
        reader,
        ledger,
        profile,
        metal.as_deref_mut(),
        &wo_a.weight.name,
        &wo_a.scale.name,
        &attn_packed,
        n,
    )?;
    drop(attn_packed);

    let wo_b = layer.control_pair(DeepSeekV4LayerControlProjection::WoB);
    let wo_b_out = fp8_linear_batched_tracked(
        reader,
        ledger,
        profile,
        metal.as_deref_mut(),
        "mla_wo_b",
        &wo_b.weight.name,
        &wo_b.scale.name,
        WO_B_ROWS,
        WO_B_COLS,
        &wo_a_out,
        n,
    )?;
    drop(wo_a_out);

    let t_post = Instant::now();
    let mut out = Vec::with_capacity(n);
    for t in 0..n {
        let row = &wo_b_out[t * WO_B_ROWS..(t + 1) * WO_B_ROWS];
        out.push(hc_attn_post_source_algorithm(
            row,
            &hc_in[t],
            &posts[t],
            &combs[t],
        )?);
    }
    profile.add("mhc_attn_post", t_post.elapsed());
    Ok(out)
}

fn execute_moe_tile(
    reader: &DeepSeekV4FullStreamReader,
    layer: &DeepSeekV4LayerSourceAnchor,
    attn_hc: &[Vec<u16>],
    token_ids: &[u64],
    ledger: &mut ResidentLedger,
    profile: &mut OperatorProfile,
    metal: Option<&mut StreamedNativeSession>,
) -> Result<Vec<StreamedLayerCapture>> {
    let n = attn_hc.len();
    let mut metal = metal;
    let mhc = layer.mhc_binding(DeepSeekV4LayerMhcStage::FeedForward);
    let t_io = Instant::now();
    let hc_fn = read_f32_tracked(
        reader,
        ledger,
        &mhc.fn_tensor.name,
        HC_MIX_WIDTH * HC_FLAT_WIDTH,
    )?;
    let hc_base = read_f32_tracked(reader, ledger, &mhc.base_tensor.name, HC_MIX_WIDTH)?;
    let hc_scale = read_f32_tracked(reader, ledger, &mhc.scale_tensor.name, 3)?;
    profile.add("streaming_io", t_io.elapsed());
    let t_mhc = Instant::now();
    let mut posts = Vec::with_capacity(n);
    let mut combs = Vec::with_capacity(n);
    let mut reduceds = Vec::with_capacity(n);
    for hc in attn_hc {
        let (_, _, _, post_f32, comb_f32, reduced) = hc_attn_pre_source_algorithm(
            hc,
            &hc_fn,
            &hc_scale,
            &hc_base,
            RMS_NORM_EPS,
            HC_EPS,
            HC_SINKHORN_ITERS,
        )?;
        posts.push(post_f32);
        combs.push(comb_f32);
        reduceds.push(reduced);
    }
    profile.add("mhc_ffn_pre", t_mhc.elapsed());
    ledger.release(&mhc.fn_tensor.name)?;
    ledger.release(&mhc.base_tensor.name)?;
    ledger.release(&mhc.scale_tensor.name)?;
    drop(hc_fn);
    drop(hc_base);
    drop(hc_scale);

    let ffn_norm = layer.common_tensor(DeepSeekV4LayerCommonTensor::FeedForwardNorm);
    let t_io = Instant::now();
    let ffn_norm_w = read_bf16_tracked(reader, ledger, &ffn_norm.name, HIDDEN_SIZE)?;
    profile.add("streaming_io", t_io.elapsed());
    let t_norm = Instant::now();
    let mut ffn_norm_rows = Vec::with_capacity(n);
    for reduced in &reduceds {
        ffn_norm_rows.push(rms_norm_bf16_source_algorithm(
            reduced,
            &ffn_norm_w,
            HIDDEN_SIZE,
            RMS_NORM_EPS,
        )?);
    }
    profile.add("ffn_norm", t_norm.elapsed());
    ledger.release(&ffn_norm.name)?;
    drop(ffn_norm_w);
    drop(reduceds);
    let ffn_norm_packed = stack_u16(&ffn_norm_rows);

    let gate = layer.gate_binding();
    let logits_packed = gate_logits_batched_tracked(
        reader,
        ledger,
        profile,
        metal.as_deref_mut(),
        &gate.score_weight.name,
        &ffn_norm_packed,
        n,
    )?;
    let t_route = Instant::now();
    let mut selected_ids = Vec::with_capacity(n);
    let mut selected_weights = Vec::with_capacity(n);
    match layer.gate_mode {
        DeepSeekV4LayerGateMode::HashTokenIdToExpertIds => {
            for t in 0..n {
                let logits = &logits_packed[t * ROUTED_EXPERTS..(t + 1) * ROUTED_EXPERTS];
                let (ids, weights) = hash_route_from_logits(
                    reader,
                    ledger,
                    &gate.route_data.name,
                    token_ids[t],
                    logits,
                )?;
                selected_ids.push(ids);
                selected_weights.push(weights);
            }
        }
        DeepSeekV4LayerGateMode::LearnedScoresWithSelectionBias => {
            let t_io = Instant::now();
            let bias = read_f32_tracked(reader, ledger, &gate.route_data.name, ROUTED_EXPERTS)?;
            profile.add("streaming_io", t_io.elapsed());
            for t in 0..n {
                let logits = &logits_packed[t * ROUTED_EXPERTS..(t + 1) * ROUTED_EXPERTS];
                let (ids, weights) = learned_bias_route(logits, &bias)?;
                selected_ids.push(ids);
                selected_weights.push(weights);
            }
            ledger.release(&gate.route_data.name)?;
            drop(bias);
        }
    }
    profile.add("route", t_route.elapsed());
    drop(logits_packed);

    let worklists = pack_expert_token_worklists(&selected_ids, &selected_weights)?;
    let mut moe_sum = vec![0.0_f32; n * HIDDEN_SIZE];
    for (expert_id, members) in worklists {
        let m = members.len();
        let mut packed = vec![0u16; m * HIDDEN_SIZE];
        for (row, &(token, _)) in members.iter().enumerate() {
            packed[row * HIDDEN_SIZE..(row + 1) * HIDDEN_SIZE]
                .copy_from_slice(&ffn_norm_rows[token]);
        }
        let weights: Vec<f32> = members.iter().map(|&(_, w)| w).collect();
        let down = routed_expert_batched_tracked(
            reader,
            ledger,
            profile,
            metal.as_deref_mut(),
            layer,
            expert_id,
            &weights,
            &packed,
            m,
        )?;
        for (row, &(token, _)) in members.iter().enumerate() {
            let src = &down[row * HIDDEN_SIZE..(row + 1) * HIDDEN_SIZE];
            let dest = &mut moe_sum[token * HIDDEN_SIZE..(token + 1) * HIDDEN_SIZE];
            for (acc, &bits) in dest.iter_mut().zip(src) {
                *acc += bf16::from_bits(bits).to_f32();
            }
        }
    }

    let shared = shared_expert_batched_tracked(
        reader,
        ledger,
        profile,
        metal,
        layer,
        &ffn_norm_packed,
        n,
    )?;
    let t_combine = Instant::now();
    for t in 0..n {
        let src = &shared[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
        let dest = &mut moe_sum[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
        for (acc, &bits) in dest.iter_mut().zip(src) {
            *acc += bf16::from_bits(bits).to_f32();
        }
        if dest.iter().any(|v| !v.is_finite()) {
            return Err(gravity(format!(
                "layer {} token {t} MoE combine produced a non-finite value",
                layer.layer
            )));
        }
    }
    profile.add("moe_combine", t_combine.elapsed());

    let t_post = Instant::now();
    let mut captures = Vec::with_capacity(n);
    for t in 0..n {
        let moe_bf16: Vec<u16> = moe_sum[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE]
            .iter()
            .map(|&v| bf16::from_f32(v).to_bits())
            .collect();
        let h_post_ffn_norm_f32: Vec<f32> = ffn_norm_rows[t]
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect();
        if h_post_ffn_norm_f32.iter().any(|v| !v.is_finite()) {
            return Err(gravity(format!(
                "layer {} token {t} h_post_ffn_norm contains a non-finite value",
                layer.layer
            )));
        }
        let next_hc = hc_attn_post_source_algorithm(
            &moe_bf16,
            &attn_hc[t],
            &posts[t],
            &combs[t],
        )?;
        let selected_expert_ids = selected_ids[t]
            .iter()
            .map(|&id| {
                u32::try_from(id).map_err(|_| gravity(format!("expert id {id} exceeds u32")))
            })
            .collect::<Result<Vec<_>>>()?;
        captures.push(StreamedLayerCapture {
            next_hc_bf16: next_hc,
            selected_expert_ids,
            normalized_route_weights: selected_weights[t].clone(),
            h_post_ffn_norm: h_post_ffn_norm_f32,
        });
    }
    profile.add("mhc_ffn_post", t_post.elapsed());
    Ok(captures)
}

fn stack_u16(rows: &[Vec<u16>]) -> Vec<u16> {
    let mut out = Vec::with_capacity(rows.iter().map(|r| r.len()).sum());
    for row in rows {
        out.extend_from_slice(row);
    }
    out
}

fn fp8_linear_batched_tracked(
    reader: &DeepSeekV4FullStreamReader,
    ledger: &mut ResidentLedger,
    profile: &mut OperatorProfile,
    metal: Option<&mut StreamedNativeSession>,
    bucket: &str,
    weight_name: &str,
    scale_name: &str,
    output_rows: usize,
    logical_k: usize,
    input: &[u16],
    batch: usize,
) -> Result<Vec<u16>> {
    if batch == 0 || input.len() != batch * logical_k {
        return Err(gravity(format!(
            "{bucket}: batched input length {} != {batch}*{logical_k}",
            input.len()
        )));
    }
    let pair = reader.native_scale_pair(weight_name)?;
    let scale_rows = output_rows / ACT_QUANT_BLOCK;
    let scale_cols = logical_k / ACT_QUANT_BLOCK;
    if pair.kind != NativeScalePairKind::Fp8E4M3fn
        || pair.scale.name != scale_name
        || pair.out_rows != output_rows as u64
        || pair.logical_k != logical_k as u64
    {
        return Err(gravity(format!(
            "{weight_name} is not the expected native FP8/E8M0 pair"
        )));
    }
    let t_io = Instant::now();
    let weights = reader.read_verified_full_view(weight_name, output_rows * logical_k)?;
    ledger.acquire(weight_name, weights.len())?;
    let scales = reader.read_verified_full_view(scale_name, scale_rows * scale_cols)?;
    ledger.acquire(scale_name, scales.len())?;
    profile.add("streaming_io", t_io.elapsed());
    if let Some(session) = metal {
        let t_metal = Instant::now();
        match session.fp8_linear_batched(
            input,
            weights.as_bytes(),
            scales.as_bytes(),
            output_rows,
            logical_k,
            batch,
        ) {
            Ok(output) => {
                profile.add(bucket, t_metal.elapsed());
                ledger.release(weight_name)?;
                ledger.release(scale_name)?;
                drop(weights);
                drop(scales);
                return Ok(output);
            }
            Err(_) => {
                // Keep the organ on the existing Metal GEMV path. Not a CPU
                // fallback — the fallback counter stays 0 if GEMV succeeds.
                let mut out = Vec::with_capacity(batch * output_rows);
                for t in 0..batch {
                    let row_in = &input[t * logical_k..(t + 1) * logical_k];
                    match session.fp8_linear(
                        row_in,
                        weights.as_bytes(),
                        scales.as_bytes(),
                        output_rows,
                        logical_k,
                    ) {
                        Ok(row) => out.extend(row),
                        Err(error) => {
                            session.record_fallback(format!("{bucket}: {error}"));
                            let quantized = act_quant_bf16_ue8m0(row_in)?;
                            let cpu = fp8_e4m3fn_ue8m0_matvec(
                                &quantized,
                                weights.as_bytes(),
                                scales.as_bytes(),
                                output_rows,
                                logical_k,
                            )?;
                            out.extend(cpu.bf16_bits);
                        }
                    }
                }
                profile.add(bucket, t_metal.elapsed());
                ledger.release(weight_name)?;
                ledger.release(scale_name)?;
                drop(weights);
                drop(scales);
                return Ok(out);
            }
        }
    }
    let t_cpu = Instant::now();
    let mut out = Vec::with_capacity(batch * output_rows);
    for t in 0..batch {
        let quantized = act_quant_bf16_ue8m0(&input[t * logical_k..(t + 1) * logical_k])?;
        let cpu = fp8_e4m3fn_ue8m0_matvec(
            &quantized,
            weights.as_bytes(),
            scales.as_bytes(),
            output_rows,
            logical_k,
        )?;
        out.extend(cpu.bf16_bits);
    }
    profile.add(bucket, t_cpu.elapsed());
    ledger.release(weight_name)?;
    ledger.release(scale_name)?;
    drop(weights);
    drop(scales);
    Ok(out)
}

fn fp4_linear_batched_tracked(
    reader: &DeepSeekV4FullStreamReader,
    ledger: &mut ResidentLedger,
    profile: &mut OperatorProfile,
    metal: Option<&mut StreamedNativeSession>,
    bucket: &str,
    weight_name: &str,
    scale_name: &str,
    output_rows: usize,
    logical_k: usize,
    input: &[u16],
    batch: usize,
) -> Result<Vec<u16>> {
    if batch == 0 || input.len() != batch * logical_k {
        return Err(gravity(format!(
            "{bucket}: batched input length {} != {batch}*{logical_k}",
            input.len()
        )));
    }
    let pair = reader.native_scale_pair(weight_name)?;
    let packed_k = logical_k / 2;
    let scale_cols = logical_k / FP4_BLOCK;
    if pair.kind != NativeScalePairKind::Fp4E2M1fnX2
        || pair.scale.name != scale_name
        || pair.out_rows != output_rows as u64
        || pair.logical_k != logical_k as u64
    {
        return Err(gravity(format!(
            "{weight_name} is not the expected native FP4/E8M0 pair"
        )));
    }
    let t_io = Instant::now();
    let weights = reader.read_verified_full_view(weight_name, output_rows * packed_k)?;
    ledger.acquire(weight_name, weights.len())?;
    let scales = reader.read_verified_full_view(scale_name, output_rows * scale_cols)?;
    ledger.acquire(scale_name, scales.len())?;
    profile.add("streaming_io", t_io.elapsed());
    if let Some(session) = metal {
        let t_metal = Instant::now();
        match session.fp4_linear_batched(
            input,
            weights.as_bytes(),
            scales.as_bytes(),
            output_rows,
            logical_k,
            batch,
        ) {
            Ok(output) => {
                profile.add(bucket, t_metal.elapsed());
                ledger.release(weight_name)?;
                ledger.release(scale_name)?;
                drop(weights);
                drop(scales);
                return Ok(output);
            }
            Err(_) => {
                let mut out = Vec::with_capacity(batch * output_rows);
                for t in 0..batch {
                    let row_in = &input[t * logical_k..(t + 1) * logical_k];
                    match session.fp4_linear(
                        row_in,
                        weights.as_bytes(),
                        scales.as_bytes(),
                        output_rows,
                        logical_k,
                    ) {
                        Ok(row) => out.extend(row),
                        Err(error) => {
                            session.record_fallback(format!("{bucket}: {error}"));
                            let quantized = act_quant_bf16_ue8m0(row_in)?;
                            let cpu = fp4_e2m1fn_x2_ue8m0_matvec(
                                &quantized,
                                weights.as_bytes(),
                                scales.as_bytes(),
                                output_rows,
                                logical_k,
                            )?;
                            out.extend(cpu.bf16_bits);
                        }
                    }
                }
                profile.add(bucket, t_metal.elapsed());
                ledger.release(weight_name)?;
                ledger.release(scale_name)?;
                drop(weights);
                drop(scales);
                return Ok(out);
            }
        }
    }
    let t_cpu = Instant::now();
    let mut out = Vec::with_capacity(batch * output_rows);
    for t in 0..batch {
        let quantized = act_quant_bf16_ue8m0(&input[t * logical_k..(t + 1) * logical_k])?;
        let cpu = fp4_e2m1fn_x2_ue8m0_matvec(
            &quantized,
            weights.as_bytes(),
            scales.as_bytes(),
            output_rows,
            logical_k,
        )?;
        out.extend(cpu.bf16_bits);
    }
    profile.add(bucket, t_cpu.elapsed());
    ledger.release(weight_name)?;
    ledger.release(scale_name)?;
    drop(weights);
    drop(scales);
    Ok(out)
}

fn wo_a_einsum_batched_tracked(
    reader: &DeepSeekV4FullStreamReader,
    ledger: &mut ResidentLedger,
    profile: &mut OperatorProfile,
    metal: Option<&mut StreamedNativeSession>,
    weight_name: &str,
    scale_name: &str,
    attention: &[u16],
    batch: usize,
) -> Result<Vec<u16>> {
    let attn_width = NUM_HEADS * HEAD_DIM;
    if batch == 0 || attention.len() != batch * attn_width {
        return Err(gravity("WO-A batched input is not [batch, 64, 512] BF16"));
    }
    let pair = reader.native_scale_pair(weight_name)?;
    if pair.kind != NativeScalePairKind::Fp8E4M3fn
        || pair.scale.name != scale_name
        || pair.out_rows != WO_A_ROWS as u64
        || pair.logical_k != WO_A_COLS as u64
    {
        return Err(gravity(format!(
            "{weight_name} is not the expected WO-A FP8 pair"
        )));
    }
    let t_io = Instant::now();
    let weights = reader.read_verified_full_view(weight_name, WO_A_ROWS * WO_A_COLS)?;
    ledger.acquire(weight_name, weights.len())?;
    let scales = reader.read_verified_full_view(
        scale_name,
        (WO_A_ROWS / ACT_QUANT_BLOCK) * (WO_A_COLS / ACT_QUANT_BLOCK),
    )?;
    ledger.acquire(scale_name, scales.len())?;
    profile.add("streaming_io", t_io.elapsed());
    if let Some(session) = metal {
        let t_metal = Instant::now();
        match session.wo_a_einsum_batched(attention, weights.as_bytes(), scales.as_bytes(), batch)
        {
            Ok(output) => {
                profile.add("mla_wo_a", t_metal.elapsed());
                ledger.release(weight_name)?;
                ledger.release(scale_name)?;
                drop(weights);
                drop(scales);
                return Ok(output);
            }
            Err(_) => {
                let mut out = Vec::with_capacity(batch * WO_A_ROWS);
                for t in 0..batch {
                    let row = &attention[t * attn_width..(t + 1) * attn_width];
                    match session.wo_a_einsum(row, weights.as_bytes(), scales.as_bytes()) {
                        Ok(bits) => out.extend(bits),
                        Err(error) => {
                            session.record_fallback(format!("mla_wo_a: {error}"));
                            out.extend(cpu_wo_a_one(row, weights.as_bytes(), scales.as_bytes())?);
                        }
                    }
                }
                profile.add("mla_wo_a", t_metal.elapsed());
                ledger.release(weight_name)?;
                ledger.release(scale_name)?;
                drop(weights);
                drop(scales);
                return Ok(out);
            }
        }
    }
    let t_cpu = Instant::now();
    let mut out = Vec::with_capacity(batch * WO_A_ROWS);
    for t in 0..batch {
        out.extend(cpu_wo_a_one(
            &attention[t * attn_width..(t + 1) * attn_width],
            weights.as_bytes(),
            scales.as_bytes(),
        )?);
    }
    profile.add("mla_wo_a", t_cpu.elapsed());
    ledger.release(weight_name)?;
    ledger.release(scale_name)?;
    drop(weights);
    drop(scales);
    Ok(out)
}

fn cpu_wo_a_one(attention: &[u16], weights: &[u8], scales: &[u8]) -> Result<Vec<u16>> {
    let input: Vec<f32> = attention
        .iter()
        .map(|bits| bf16::from_bits(*bits).to_f32())
        .collect();
    if input.iter().any(|v| !v.is_finite()) {
        return Err(gravity("WO-A attention input is non-finite"));
    }
    let scale_cols = WO_A_COLS / ACT_QUANT_BLOCK;
    let mut output = Vec::with_capacity(WO_A_ROWS);
    for group in 0..O_GROUPS {
        let input_group = &input[group * WO_A_COLS..(group + 1) * WO_A_COLS];
        for rank in 0..O_LORA_RANK {
            let row = group * O_LORA_RANK + rank;
            let mut acc = 0.0_f32;
            for column in 0..WO_A_COLS {
                let raw = weights[row * WO_A_COLS + column];
                let scale_index = (row / ACT_QUANT_BLOCK) * scale_cols + column / ACT_QUANT_BLOCK;
                let converted =
                    bf16::from_f32(decode_e4m3fn(raw)? * decode_e8m0fnu(scales[scale_index])?)
                        .to_f32();
                if !converted.is_finite() {
                    return Err(gravity("WO-A conversion produced a non-finite BF16 weight"));
                }
                acc += input_group[column] * converted;
            }
            if !acc.is_finite() {
                return Err(gravity("WO-A grouped einsum produced a non-finite value"));
            }
            output.push(bf16::from_f32(acc).to_bits());
        }
    }
    Ok(output)
}

fn gate_logits_batched_tracked(
    reader: &DeepSeekV4FullStreamReader,
    ledger: &mut ResidentLedger,
    profile: &mut OperatorProfile,
    metal: Option<&mut StreamedNativeSession>,
    weight_name: &str,
    input_bf16: &[u16],
    batch: usize,
) -> Result<Vec<f32>> {
    if batch == 0 || input_bf16.len() != batch * HIDDEN_SIZE {
        return Err(gravity("gate batched input is not BF16[batch, 4096]"));
    }
    let t_io = Instant::now();
    let weights = read_bf16_tracked(reader, ledger, weight_name, ROUTED_EXPERTS * HIDDEN_SIZE)?;
    profile.add("streaming_io", t_io.elapsed());
    if let Some(session) = metal {
        let t_metal = Instant::now();
        match session.gate_logits_batched(input_bf16, &weights, batch) {
            Ok(logits) => {
                profile.add("gate_logits_metal", t_metal.elapsed());
                ledger.release(weight_name)?;
                drop(weights);
                return Ok(logits);
            }
            Err(_) => {
                let mut out = Vec::with_capacity(batch * ROUTED_EXPERTS);
                for t in 0..batch {
                    let row = &input_bf16[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
                    match session.gate_logits(row, &weights) {
                        Ok(logits) => out.extend(logits),
                        Err(error) => {
                            session.record_fallback(format!("gate_logits: {error}"));
                            out.extend(cpu_gate_one(row, &weights)?);
                        }
                    }
                }
                profile.add("gate_logits_metal", t_metal.elapsed());
                ledger.release(weight_name)?;
                drop(weights);
                return Ok(out);
            }
        }
    }
    let t_cpu = Instant::now();
    let mut out = Vec::with_capacity(batch * ROUTED_EXPERTS);
    for t in 0..batch {
        out.extend(cpu_gate_one(
            &input_bf16[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE],
            &weights,
        )?);
    }
    profile.add("gate_logits_cpu", t_cpu.elapsed());
    ledger.release(weight_name)?;
    drop(weights);
    Ok(out)
}

fn cpu_gate_one(input_bf16: &[u16], weights: &[u16]) -> Result<Vec<f32>> {
    let input: Vec<f32> = input_bf16
        .iter()
        .map(|bits| bf16::from_bits(*bits).to_f32())
        .collect();
    if input.iter().any(|v| !v.is_finite()) {
        return Err(gravity("Gate input contains a non-finite BF16 value"));
    }
    let mut logits = Vec::with_capacity(ROUTED_EXPERTS);
    for row in 0..ROUTED_EXPERTS {
        let mut acc = 0.0_f32;
        let wrow = &weights[row * HIDDEN_SIZE..(row + 1) * HIDDEN_SIZE];
        for (&activation, &weight_bits) in input.iter().zip(wrow) {
            let weight = bf16::from_bits(weight_bits).to_f32();
            if !weight.is_finite() {
                return Err(gravity("Gate weight contains a non-finite BF16 value"));
            }
            acc += activation * weight;
        }
        if !acc.is_finite() {
            return Err(gravity("Gate logit is non-finite"));
        }
        logits.push(acc);
    }
    Ok(logits)
}

fn routed_expert_batched_tracked(
    reader: &DeepSeekV4FullStreamReader,
    ledger: &mut ResidentLedger,
    profile: &mut OperatorProfile,
    metal: Option<&mut StreamedNativeSession>,
    layer: &DeepSeekV4LayerSourceAnchor,
    expert_id: u64,
    route_weights: &[f32],
    input: &[u16],
    batch: usize,
) -> Result<Vec<u16>> {
    let mut metal = metal;
    if route_weights.len() != batch {
        return Err(gravity("routed expert batch/weight length mismatch"));
    }
    let w1 = layer.routed_expert_pair(expert_id as usize, DeepSeekV4LayerExpertProjection::W1)?;
    let w3 = layer.routed_expert_pair(expert_id as usize, DeepSeekV4LayerExpertProjection::W3)?;
    let w2 = layer.routed_expert_pair(expert_id as usize, DeepSeekV4LayerExpertProjection::W2)?;
    let gate = fp4_linear_batched_tracked(
        reader,
        ledger,
        profile,
        metal.as_deref_mut(),
        "routed_w1",
        &w1.weight.name,
        &w1.scale.name,
        MOE_INTER_DIM,
        HIDDEN_SIZE,
        input,
        batch,
    )?;
    let up = fp4_linear_batched_tracked(
        reader,
        ledger,
        profile,
        metal.as_deref_mut(),
        "routed_w3",
        &w3.weight.name,
        &w3.scale.name,
        MOE_INTER_DIM,
        HIDDEN_SIZE,
        input,
        batch,
    )?;
    let t_swiglu = Instant::now();
    let mut swiglu = Vec::with_capacity(batch * MOE_INTER_DIM);
    for t in 0..batch {
        let g = &gate[t * MOE_INTER_DIM..(t + 1) * MOE_INTER_DIM];
        let u = &up[t * MOE_INTER_DIM..(t + 1) * MOE_INTER_DIM];
        swiglu.extend(swiglu_bf16_source_algorithm(g, u, Some(route_weights[t]))?);
    }
    profile.add("swiglu_routed", t_swiglu.elapsed());
    drop(gate);
    drop(up);
    fp4_linear_batched_tracked(
        reader,
        ledger,
        profile,
        metal,
        "routed_w2",
        &w2.weight.name,
        &w2.scale.name,
        HIDDEN_SIZE,
        MOE_INTER_DIM,
        &swiglu,
        batch,
    )
}

fn shared_expert_batched_tracked(
    reader: &DeepSeekV4FullStreamReader,
    ledger: &mut ResidentLedger,
    profile: &mut OperatorProfile,
    metal: Option<&mut StreamedNativeSession>,
    layer: &DeepSeekV4LayerSourceAnchor,
    input: &[u16],
    batch: usize,
) -> Result<Vec<u16>> {
    let mut metal = metal;
    let w1 = layer.shared_expert_pair(DeepSeekV4LayerExpertProjection::W1);
    let w3 = layer.shared_expert_pair(DeepSeekV4LayerExpertProjection::W3);
    let w2 = layer.shared_expert_pair(DeepSeekV4LayerExpertProjection::W2);
    let gate = fp8_linear_batched_tracked(
        reader,
        ledger,
        profile,
        metal.as_deref_mut(),
        "shared_w1",
        &w1.weight.name,
        &w1.scale.name,
        MOE_INTER_DIM,
        HIDDEN_SIZE,
        input,
        batch,
    )?;
    let up = fp8_linear_batched_tracked(
        reader,
        ledger,
        profile,
        metal.as_deref_mut(),
        "shared_w3",
        &w3.weight.name,
        &w3.scale.name,
        MOE_INTER_DIM,
        HIDDEN_SIZE,
        input,
        batch,
    )?;
    let t_swiglu = Instant::now();
    let mut swiglu = Vec::with_capacity(batch * MOE_INTER_DIM);
    for t in 0..batch {
        let g = &gate[t * MOE_INTER_DIM..(t + 1) * MOE_INTER_DIM];
        let u = &up[t * MOE_INTER_DIM..(t + 1) * MOE_INTER_DIM];
        swiglu.extend(swiglu_bf16_source_algorithm(g, u, None)?);
    }
    profile.add("swiglu_shared", t_swiglu.elapsed());
    drop(gate);
    drop(up);
    fp8_linear_batched_tracked(
        reader,
        ledger,
        profile,
        metal,
        "shared_w2",
        &w2.weight.name,
        &w2.scale.name,
        HIDDEN_SIZE,
        MOE_INTER_DIM,
        &swiglu,
        batch,
    )
}

#[cfg(test)]
mod tests {
    use super::pack_expert_token_worklists;

    #[test]
    fn pack_expert_worklists_is_inverse_of_per_token_sort() {
        let ids = vec![vec![5, 1, 9, 2, 7, 3], vec![2, 8, 1, 4, 6, 0]];
        let weights = vec![
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            vec![0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ];
        let lists = pack_expert_token_worklists(&ids, &weights).unwrap();
        let experts: Vec<u64> = lists.iter().map(|(e, _)| *e).collect();
        assert_eq!(experts, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let e1 = lists.iter().find(|(e, _)| *e == 1).unwrap();
        assert_eq!(e1.1, vec![(0, 0.2), (1, 0.4)]);
        let e2 = lists.iter().find(|(e, _)| *e == 2).unwrap();
        assert_eq!(e2.1, vec![(0, 0.4), (1, 0.6)]);
    }
}

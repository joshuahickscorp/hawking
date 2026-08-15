//! Isolated Qwen3-Coder-Next layer-3 direct-packed GQA two-token CPU oracle.
//!
//! This target deliberately consumes only three small existing evidence
//! documents: the sealed outer layer-3 GQA fixture receipt, its inner receipt,
//! and the source-template/source-config authority.  It never opens a model
//! payload, source sidecar, tokenizer, gravity artifact, registry, runtime,
//! watcher, server, lease, or Metal object.
//!
//! The numerical vectors are deterministic *synthetic* direct-packed sign-bit
//! plus group-scale fixtures.  They exercise the exact source-shaped layer-3
//! operator order over two positions:
//!
//! ```text
//! Q/K/V projection -> Q/K RMSNorm + partial RoPE -> append K/V
//! -> causal softmax -> source Q-gate sigmoid -> O projection -> rollback
//! ```
//!
//! Therefore the emitted boundary hashes are a future device-child parity ABI,
//! not a model forward, a full layer, a token, a decoder, HCLI, TPS, TG, or
//! server result.
//!
//! ```text
//! cargo run -p hawking-core --example ascension_qwen80_direct_packed_gqa_two_token_cpu_oracle -- \
//!   --layer3-inner-receipt /absolute/path/inner/receipt.json \
//!   --layer3-outer-receipt /absolute/path/outer-terminal-receipt.json \
//!   --source-template-authority /absolute/path/QWEN80_SOURCE_TEMPLATE_AB_AUTHORITY.json \
//!   --out /absolute/new/path/QWEN80_GQA_TWO_TOKEN_CPU_ORACLE.json
//! ```

use hawking_core::model::qwen80_complete_runtime::{
    qwen80_layer_kind, Qwen80LayerKind, QWEN80_COMPLETE_NATIVE_MAX_CONTEXT,
};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;

const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_direct_packed_gqa_two_token_cpu_oracle.v1";
const RESULT_STATUS: &str = "CAPTURED_SYNTHETIC_DIRECT_PACKED_QWEN80_LAYER3_GQA_TWO_TOKEN_CPU_ORACLE_COMPONENT_ONLY_NOT_RUNTIME";
const LAYER3_INNER_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_layer3_gqa_attention_stage.v1";
const LAYER3_INNER_STATUS: &str =
    "EARNED_QWEN80_ADMITTED_DIRECT_PACKED_LAYER3_GQA_TWO_TOKEN_COMPONENT_STAGE_NOT_COMPLETE_LAYER_OR_TOKEN";
const LAYER3_OUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_layer3_gqa_outer_launcher.v1";
const LAYER3_OUTER_STATUS: &str = "CAPTURED_QWEN80_LAYER3_GQA_OUTER_TERMINAL_CHILD_ZERO_INNER_RECEIPT_PRESENT_NOT_A_PARITY_OR_RUNTIME_CLAIM";
const LAYER3_OUTER_SEAL: &str = "b0b16468df4b42ae8b02de076de72850bdd244f9dd488be9432f8e61bbe0a44e";
const SOURCE_TEMPLATE_SCHEMA: &str =
    "hawking.ascension.qwen80_source_template_ab_authority_preflight.v1";
const SOURCE_TEMPLATE_STATUS: &str = "PREPARED_NOT_EXECUTED_QWEN80_SOURCE_TEMPLATE_AB_AUTHORITY";
const SOURCE_TEMPLATE_PREIMAGE_SHA256: &str =
    "a69fe1487ebac4d1c4b2c6d0e92fcbd803e17b7442ba4c59964ec391e811a6a8";
const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const SOURCE_CONFIG_SHA256: &str =
    "a7b8098d3b05777f12bb5677a26bf1240a1bb09def1b06b29e6be86cae2e84f8";
const MANIFEST_SEAL_SHA256: &str =
    "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b";

const LAYER: usize = 3;
const POSITIONS: usize = 2;
const HIDDEN: usize = 2_048;
const QUERY_HEADS: usize = 16;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const QUERY_DIM: usize = QUERY_HEADS * HEAD_DIM;
const KV_DIM: usize = KV_HEADS * HEAD_DIM;
const Q_PROJ_ROWS: usize = QUERY_DIM * 2;
const ROTARY_DIM: usize = 64;
const GROUP_SIZE: usize = 128;
const ROPE_THETA: f32 = 5_000_000.0;
const RMS_EPSILON: f32 = 1.0e-6;

#[derive(Clone, Debug, Serialize)]
struct FileEvidence {
    path: String,
    bytes: usize,
    sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct BoundFixtureAuthority {
    sealed_layer3_outer_receipt: FileEvidence,
    layer3_inner_component_receipt: FileEvidence,
    source_template_authority: FileEvidence,
    layer3_outer_seal_sha256: &'static str,
    layer3_inner_unsigned_json_sha256: String,
    source_config_sha256: &'static str,
    source_revision: &'static str,
    source_geometry_checked_via_public_schedule_resolver: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProjectionRole {
    Q,
    K,
    V,
    O,
}

impl ProjectionRole {
    fn label(self) -> &'static str {
        match self {
            Self::Q => "q_projection_query_and_gate_rows",
            Self::K => "k_projection",
            Self::V => "v_projection",
            Self::O => "o_projection",
        }
    }

    fn tensor_name(self) -> &'static str {
        match self {
            Self::Q => "model.layers.3.self_attn.q_proj.weight",
            Self::K => "model.layers.3.self_attn.k_proj.weight",
            Self::V => "model.layers.3.self_attn.v_proj.weight",
            Self::O => "model.layers.3.self_attn.o_proj.weight",
        }
    }

    fn seed(self) -> u64 {
        match self {
            Self::Q => 0x47b4_9be2_1d8f_0a13,
            Self::K => 0x9c61_2ad0_45e7_b813,
            Self::V => 0x2d5a_7cc1_9f83_e641,
            Self::O => 0xd6f0_1449_ab32_7c9e,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct ProjectionAbi {
    role: &'static str,
    tensor_name: &'static str,
    shape: Vec<usize>,
    direct_packed_group_size: usize,
    compact_payload_format: &'static str,
    synthetic_fixture_payload: bool,
    source_artifact_payload_opened: bool,
}

fn source_projection_abis() -> Vec<ProjectionAbi> {
    [
        (ProjectionRole::Q, Q_PROJ_ROWS, HIDDEN),
        (ProjectionRole::K, KV_DIM, HIDDEN),
        (ProjectionRole::V, KV_DIM, HIDDEN),
        (ProjectionRole::O, HIDDEN, QUERY_DIM),
    ]
    .into_iter()
    .map(|(role, rows, columns)| ProjectionAbi {
        role: role.label(),
        tensor_name: role.tensor_name(),
        shape: vec![rows, columns],
        direct_packed_group_size: GROUP_SIZE,
        compact_payload_format: "direct_binary_sign_bits_plus_fp16_group_scales",
        synthetic_fixture_payload: true,
        source_artifact_payload_opened: false,
    })
    .collect()
}

/// Synthetic packed signs/scales intentionally remain algorithmic instead of
/// allocating, decoding, or consulting a real model payload.  The group and
/// sign-bit accesses mirror the compact ABI a future device child must use.
#[derive(Clone, Debug)]
struct SyntheticDirectPackedMatrix {
    role: ProjectionRole,
    rows: usize,
    columns: usize,
    group_size: usize,
}

impl SyntheticDirectPackedMatrix {
    fn new(
        role: ProjectionRole,
        rows: usize,
        columns: usize,
        group_size: usize,
    ) -> Result<Self, String> {
        if rows == 0 || columns == 0 || group_size == 0 || columns % group_size != 0 {
            return Err(
                "synthetic direct-packed matrix has invalid rows/columns/group size".into(),
            );
        }
        Ok(Self {
            role,
            rows,
            columns,
            group_size,
        })
    }

    fn source_shaped(role: ProjectionRole) -> Result<Self, String> {
        let (rows, columns) = match role {
            ProjectionRole::Q => (Q_PROJ_ROWS, HIDDEN),
            ProjectionRole::K | ProjectionRole::V => (KV_DIM, HIDDEN),
            ProjectionRole::O => (HIDDEN, QUERY_DIM),
        };
        Self::new(role, rows, columns, GROUP_SIZE)
    }

    fn groups_per_row(&self) -> usize {
        self.columns / self.group_size
    }

    fn mix(mut value: u64) -> u64 {
        value ^= value >> 30;
        value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value ^= value >> 27;
        value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn sign_word(&self, row: usize, group: usize, word: usize) -> u64 {
        Self::mix(
            self.role.seed()
                ^ (row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ (group as u64).wrapping_mul(0xd1b5_4a32_d192_ed03)
                ^ (word as u64).wrapping_mul(0x94d0_49bb_1331_11eb),
        )
    }

    fn group_scale(&self, row: usize, group: usize) -> f32 {
        let mixed = Self::mix(
            self.role.seed()
                ^ (row as u64).wrapping_mul(0x632b_b1e7_23d5_6c61)
                ^ (group as u64).wrapping_mul(0x8cb9_2ba7_2f3d_8dd7),
        );
        // Values are exactly representable binary fractions; this remains a
        // deterministic synthetic scale, not an artifact FP16 decode.
        (1 + (mixed % 23) as u32) as f32 / 1_024.0
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn element(&self, row: usize, column: usize) -> Result<f32, String> {
        if row >= self.rows || column >= self.columns {
            return Err("synthetic direct-packed element index is outside matrix geometry".into());
        }
        let group = column / self.group_size;
        let in_group = column % self.group_size;
        let word = self.sign_word(row, group, in_group / 64);
        let positive = ((word >> (in_group % 64)) & 1) != 0;
        let scale = self.group_scale(row, group);
        Ok(if positive { scale } else { -scale })
    }

    fn matvec(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        if input.len() != self.columns {
            return Err(format!(
                "{} direct-packed projection input has {} elements, expected {}",
                self.role.label(),
                input.len(),
                self.columns
            ));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err("synthetic direct-packed projection rejects non-finite input".into());
        }
        let mut output = vec![0.0f32; self.rows];
        for row in 0..self.rows {
            let mut sum = 0.0f32;
            for group in 0..self.groups_per_row() {
                let scale = self.group_scale(row, group);
                let first = group * self.group_size;
                let mut word = self.sign_word(row, group, 0);
                for in_group in 0..self.group_size {
                    if in_group == 64 {
                        word = self.sign_word(row, group, 1);
                    }
                    let value = input[first + in_group];
                    if ((word >> (in_group % 64)) & 1) != 0 {
                        sum += scale * value;
                    } else {
                        sum -= scale * value;
                    }
                }
            }
            if !sum.is_finite() {
                return Err(format!(
                    "{} synthetic direct-packed projection produced non-finite row {row}",
                    self.role.label()
                ));
            }
            output[row] = sum;
        }
        Ok(output)
    }
}

#[derive(Clone, Debug)]
struct KvSnapshot {
    key: Vec<f32>,
    value: Vec<f32>,
    committed_positions: usize,
}

#[derive(Clone, Debug)]
struct FixtureKvCache {
    key: Vec<f32>,
    value: Vec<f32>,
    committed_positions: usize,
}

impl FixtureKvCache {
    fn empty() -> Self {
        Self {
            key: vec![0.0; POSITIONS * KV_DIM],
            value: vec![0.0; POSITIONS * KV_DIM],
            committed_positions: 0,
        }
    }

    fn snapshot(&self) -> KvSnapshot {
        KvSnapshot {
            key: self.key.clone(),
            value: self.value.clone(),
            committed_positions: self.committed_positions,
        }
    }

    fn append(&mut self, position: usize, key: &[f32], value: &[f32]) -> Result<(), String> {
        if position != self.committed_positions {
            return Err(format!(
                "K/V append position {position} does not equal next committed position {}",
                self.committed_positions
            ));
        }
        if position >= POSITIONS || key.len() != KV_DIM || value.len() != KV_DIM {
            return Err("K/V append violates the bounded [2][2][256] fixture geometry".into());
        }
        if key.iter().chain(value).any(|entry| !entry.is_finite()) {
            return Err("K/V append rejects non-finite vectors".into());
        }
        let offset = position * KV_DIM;
        self.key[offset..offset + KV_DIM].copy_from_slice(key);
        self.value[offset..offset + KV_DIM].copy_from_slice(value);
        self.committed_positions += 1;
        Ok(())
    }

    fn causal_key_value(&self, position: usize) -> Result<(&[f32], &[f32]), String> {
        if position >= POSITIONS || self.committed_positions != position + 1 {
            return Err(format!(
                "causal read at position {position} requires K/V append-before-read through that position; committed_positions={}",
                self.committed_positions
            ));
        }
        let elements = (position + 1) * KV_DIM;
        Ok((&self.key[..elements], &self.value[..elements]))
    }

    fn restore(&mut self, snapshot: &KvSnapshot) -> Result<(), String> {
        if snapshot.key.len() != self.key.len()
            || snapshot.value.len() != self.value.len()
            || snapshot.committed_positions > POSITIONS
        {
            return Err("rollback snapshot does not match the bounded K/V fixture geometry".into());
        }
        self.key.copy_from_slice(&snapshot.key);
        self.value.copy_from_slice(&snapshot.value);
        self.committed_positions = snapshot.committed_positions;
        Ok(())
    }

    fn active_key(&self) -> &[f32] {
        &self.key[..self.committed_positions * KV_DIM]
    }

    fn active_value(&self) -> &[f32] {
        &self.value[..self.committed_positions * KV_DIM]
    }
}

fn synthetic_hidden(position: usize) -> Vec<f32> {
    (0..HIDDEN)
        .map(|index| {
            let numerator = ((index * 97 + position * 43 + 29) % 8_191) as f32 - 4_095.0;
            numerator / 8_191.0
        })
        .collect()
}

fn synthetic_norm_weights(kind: ProjectionRole) -> Result<Vec<f32>, String> {
    if kind != ProjectionRole::Q && kind != ProjectionRole::K {
        return Err("only Q and K have source GQA RMSNorm weights".into());
    }
    let seed = kind.seed() as usize;
    Ok((0..HEAD_DIM)
        .map(|index| (((index * 29 + seed % 97 + 7) % 61) as f32 - 30.0) / 4_096.0)
        .collect())
}

fn finite(values: &[f32], label: &str) -> Result<(), String> {
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "{label} contains non-finite value {value} at {index}"
        ));
    }
    Ok(())
}

/// Qwen3-Next Q/K normalization and non-interleaved partial RoPE, using the
/// same `norm(x) * (1 + learned_weight)` and 64-dimension rotate-half rules as
/// the source-bound layer-3 component fixture.
fn source_norm_partial_rope(
    raw: &[f32],
    weights: &[f32],
    heads: usize,
    position: usize,
    label: &str,
) -> Result<Vec<f32>, String> {
    if raw.len() != heads * HEAD_DIM || weights.len() != HEAD_DIM || position >= POSITIONS {
        return Err(format!(
            "{label} has invalid source-shaped norm/RoPE geometry"
        ));
    }
    let mut output = vec![0.0f32; raw.len()];
    for head in 0..heads {
        let base = head * HEAD_DIM;
        let variance = raw[base..base + HEAD_DIM]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / HEAD_DIM as f32;
        let inverse_rms = (variance + RMS_EPSILON).sqrt().recip();
        for dimension in 0..HEAD_DIM {
            output[base + dimension] =
                raw[base + dimension] * inverse_rms * (1.0 + weights[dimension]);
        }
        let before = output[base..base + ROTARY_DIM].to_vec();
        for dimension in 0..(ROTARY_DIM / 2) {
            let inverse_frequency = ROPE_THETA.powf(-2.0 * dimension as f32 / ROTARY_DIM as f32);
            let angle = position as f32 * inverse_frequency;
            let cosine = angle.cos();
            let sine = angle.sin();
            output[base + dimension] =
                before[dimension] * cosine - before[dimension + ROTARY_DIM / 2] * sine;
            output[base + dimension + ROTARY_DIM / 2] =
                before[dimension + ROTARY_DIM / 2] * cosine + before[dimension] * sine;
        }
    }
    finite(&output, label)?;
    Ok(output)
}

fn query_from_interleaved_q_projection(q_projection: &[f32]) -> Result<Vec<f32>, String> {
    if q_projection.len() != Q_PROJ_ROWS {
        return Err("Q projection does not have source [16][Q(256),gate(256)] rows".into());
    }
    let mut query = vec![0.0f32; QUERY_DIM];
    for head in 0..QUERY_HEADS {
        let source = head * 2 * HEAD_DIM;
        let destination = head * HEAD_DIM;
        query[destination..destination + HEAD_DIM]
            .copy_from_slice(&q_projection[source..source + HEAD_DIM]);
    }
    Ok(query)
}

fn gate_from_interleaved_q_projection(q_projection: &[f32]) -> Result<Vec<f32>, String> {
    if q_projection.len() != Q_PROJ_ROWS {
        return Err("Q projection does not have source [16][Q(256),gate(256)] rows".into());
    }
    let mut gate = vec![0.0f32; QUERY_DIM];
    for head in 0..QUERY_HEADS {
        let source = head * 2 * HEAD_DIM + HEAD_DIM;
        let destination = head * HEAD_DIM;
        gate[destination..destination + HEAD_DIM]
            .copy_from_slice(&q_projection[source..source + HEAD_DIM]);
    }
    Ok(gate)
}

struct CausalAttention {
    probabilities: Vec<f32>,
    output: Vec<f32>,
}

fn causal_gqa_attention(
    query: &[f32],
    cache: &FixtureKvCache,
    position: usize,
) -> Result<CausalAttention, String> {
    if query.len() != QUERY_DIM {
        return Err("causal GQA query does not have source [16][256] geometry".into());
    }
    let (keys, values) = cache.causal_key_value(position)?;
    let sequence_length = position + 1;
    let scale = (HEAD_DIM as f32).sqrt().recip();
    let queries_per_kv_head = QUERY_HEADS / KV_HEADS;
    let mut probabilities = Vec::with_capacity(QUERY_HEADS * sequence_length);
    let mut output = vec![0.0f32; QUERY_DIM];
    for head in 0..QUERY_HEADS {
        let query_base = head * HEAD_DIM;
        let kv_head = head / queries_per_kv_head;
        let mut scores = Vec::with_capacity(sequence_length);
        for token in 0..sequence_length {
            let key_base = (token * KV_HEADS + kv_head) * HEAD_DIM;
            let dot = (0..HEAD_DIM)
                .map(|dimension| query[query_base + dimension] * keys[key_base + dimension])
                .sum::<f32>();
            scores.push(dot * scale);
        }
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let normalizer = scores
            .iter()
            .map(|score| (*score - maximum).exp())
            .sum::<f32>();
        if !normalizer.is_finite() || normalizer <= 0.0 {
            return Err(format!(
                "causal GQA softmax is invalid at query head {head}"
            ));
        }
        for (token, score) in scores.iter().enumerate() {
            let probability = (*score - maximum).exp() / normalizer;
            probabilities.push(probability);
            let value_base = (token * KV_HEADS + kv_head) * HEAD_DIM;
            for dimension in 0..HEAD_DIM {
                output[query_base + dimension] += probability * values[value_base + dimension];
            }
        }
    }
    finite(&probabilities, "causal GQA softmax probabilities")?;
    finite(&output, "causal GQA output")?;
    Ok(CausalAttention {
        probabilities,
        output,
    })
}

fn sigmoid_gate(values: &[f32]) -> Result<Vec<f32>, String> {
    let output = values
        .iter()
        .map(|value| (1.0 + (-value).exp()).recip())
        .collect::<Vec<_>>();
    finite(&output, "source Q projection sigmoid gate")?;
    Ok(output)
}

fn sha256_f32le(values: &[f32]) -> Result<String, String> {
    finite(values, "f32 little-endian SHA-256 input")?;
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[derive(Clone, Debug, Serialize)]
struct BoundaryHash {
    name: &'static str,
    shape: Vec<usize>,
    f32le_sha256: String,
}

fn boundary(name: &'static str, shape: Vec<usize>, values: &[f32]) -> Result<BoundaryHash, String> {
    let elements = shape.iter().product::<usize>();
    if elements != values.len() {
        return Err(format!(
            "boundary {name} shape {:?} has {elements} elements, observed {}",
            shape,
            values.len()
        ));
    }
    Ok(BoundaryHash {
        name,
        shape,
        f32le_sha256: sha256_f32le(values)?,
    })
}

#[derive(Clone, Debug, Serialize)]
struct TokenBoundaryHashes {
    position: usize,
    source_hidden: BoundaryHash,
    q_projection: BoundaryHash,
    k_projection: BoundaryHash,
    v_projection: BoundaryHash,
    query_after_q_norm_partial_rope: BoundaryHash,
    key_after_k_norm_partial_rope: BoundaryHash,
    active_key_cache_after_append: BoundaryHash,
    active_value_cache_after_append: BoundaryHash,
    causal_softmax_probabilities: BoundaryHash,
    causal_attention: BoundaryHash,
    q_projection_sigmoid_gate: BoundaryHash,
    gated_attention: BoundaryHash,
    o_projection: BoundaryHash,
}

#[derive(Clone, Debug, Serialize)]
struct RollbackBoundaryHashes {
    initial_active_key: BoundaryHash,
    initial_active_value: BoundaryHash,
    active_key_after_two_appends: BoundaryHash,
    active_value_after_two_appends: BoundaryHash,
    restored_key_after_rollback_to_position_zero: BoundaryHash,
    restored_value_after_rollback_to_position_zero: BoundaryHash,
    restored_key_after_rollback_to_empty: BoundaryHash,
    restored_value_after_rollback_to_empty: BoundaryHash,
    rollback_restored_position_zero_exactly: bool,
    rollback_restored_empty_exactly: bool,
}

#[derive(Clone, Debug, Serialize)]
struct DeviceParitySchema {
    schema: &'static str,
    element_encoding: &'static str,
    comparison_rule: &'static str,
    required_command_order: Vec<&'static str>,
    required_readback_boundaries: Vec<&'static str>,
    cache_rollback_must_restore_exact_f32le_hashes: bool,
}

#[derive(Clone, Debug, Serialize)]
struct FutureDeviceInputs {
    component_scope: &'static str,
    source_projection_abi: Vec<ProjectionAbi>,
    required_norm_tensors: Vec<&'static str>,
    source_bound_hidden_inputs: Vec<&'static str>,
    caller_owned_session_state: Vec<&'static str>,
    same_command_buffer_requirements: Vec<&'static str>,
    device_child_must_not_claim: Vec<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
struct ExecutionBoundary {
    fixture_component_only: bool,
    deterministic_synthetic_weights_only: bool,
    real_upstream_hidden_used: bool,
    model_artifact_or_payload_opened: bool,
    source_sidecar_opened: bool,
    runtime_registry_watcher_server_or_lease_touched: bool,
    metal_or_gpu_invoked: bool,
    full_layer_token_decoder_hcli_tps_or_tg_claimed: bool,
}

#[derive(Clone, Debug, Serialize)]
struct OracleReport {
    schema: &'static str,
    status: &'static str,
    source_fixture_authority: BoundFixtureAuthority,
    source_geometry: SourceGeometry,
    synthetic_direct_packed_fixture: SyntheticFixtureDescription,
    per_token_boundary_hashes: Vec<TokenBoundaryHashes>,
    rollback_boundary_hashes: RollbackBoundaryHashes,
    future_device_parity_schema: DeviceParitySchema,
    exact_future_device_inputs: FutureDeviceInputs,
    execution_boundary: ExecutionBoundary,
    unsealed_preimage_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct SourceGeometry {
    layer: usize,
    layer_kind: &'static str,
    fixture_positions: Vec<usize>,
    hidden_size: usize,
    query_heads: usize,
    kv_heads: usize,
    queries_per_kv_head: usize,
    head_dim: usize,
    q_proj_shape: Vec<usize>,
    k_proj_shape: Vec<usize>,
    v_proj_shape: Vec<usize>,
    o_proj_shape: Vec<usize>,
    qk_norm_shape: Vec<usize>,
    partial_rotary_dimensions: usize,
    rope_theta: f32,
    rms_norm_epsilon: f32,
    active_and_rollback_kv_shape: Vec<usize>,
}

#[derive(Clone, Debug, Serialize)]
struct SyntheticFixtureDescription {
    payload_kind: &'static str,
    source_artifact_payload_opened: bool,
    source_hidden_is_synthetic: bool,
    direct_packed_group_size: usize,
    q_projection_layout: &'static str,
    exact_operator_order: Vec<&'static str>,
}

fn source_geometry() -> SourceGeometry {
    SourceGeometry {
        layer: LAYER,
        layer_kind: "full_attention",
        fixture_positions: vec![0, 1],
        hidden_size: HIDDEN,
        query_heads: QUERY_HEADS,
        kv_heads: KV_HEADS,
        queries_per_kv_head: QUERY_HEADS / KV_HEADS,
        head_dim: HEAD_DIM,
        q_proj_shape: vec![Q_PROJ_ROWS, HIDDEN],
        k_proj_shape: vec![KV_DIM, HIDDEN],
        v_proj_shape: vec![KV_DIM, HIDDEN],
        o_proj_shape: vec![HIDDEN, QUERY_DIM],
        qk_norm_shape: vec![HEAD_DIM],
        partial_rotary_dimensions: ROTARY_DIM,
        rope_theta: ROPE_THETA,
        rms_norm_epsilon: RMS_EPSILON,
        active_and_rollback_kv_shape: vec![POSITIONS, KV_HEADS, HEAD_DIM],
    }
}

fn validate_source_geometry() -> Result<(), String> {
    if QWEN80_COMPLETE_NATIVE_MAX_CONTEXT < POSITIONS
        || qwen80_layer_kind(LAYER).map_err(|error| error.to_string())?
            != Qwen80LayerKind::FullAttention
        || QUERY_HEADS % KV_HEADS != 0
        || ROTARY_DIM != HEAD_DIM / 4
        || GROUP_SIZE == 0
        || HIDDEN % GROUP_SIZE != 0
        || QUERY_DIM % GROUP_SIZE != 0
    {
        return Err("public Qwen80 source schedule or source-shaped GQA geometry drifted".into());
    }
    for (slot, expected_layer) in (0..12).map(|slot| (slot, slot * 4 + 3)) {
        if qwen80_layer_kind(expected_layer).map_err(|error| error.to_string())?
            != Qwen80LayerKind::FullAttention
        {
            return Err(format!("Qwen80 GQA schedule drifted at slot {slot}"));
        }
    }
    Ok(())
}

fn run_synthetic_two_token_fixture(
) -> Result<(Vec<TokenBoundaryHashes>, RollbackBoundaryHashes), String> {
    validate_source_geometry()?;
    let q = SyntheticDirectPackedMatrix::source_shaped(ProjectionRole::Q)?;
    let k = SyntheticDirectPackedMatrix::source_shaped(ProjectionRole::K)?;
    let v = SyntheticDirectPackedMatrix::source_shaped(ProjectionRole::V)?;
    let o = SyntheticDirectPackedMatrix::source_shaped(ProjectionRole::O)?;
    let q_norm = synthetic_norm_weights(ProjectionRole::Q)?;
    let k_norm = synthetic_norm_weights(ProjectionRole::K)?;
    let mut cache = FixtureKvCache::empty();
    let initial = cache.snapshot();
    let initial_key = boundary(
        "initial_active_key",
        vec![0, KV_HEADS, HEAD_DIM],
        cache.active_key(),
    )?;
    let initial_value = boundary(
        "initial_active_value",
        vec![0, KV_HEADS, HEAD_DIM],
        cache.active_value(),
    )?;
    let mut after_position_zero = None;
    let mut token_hashes = Vec::with_capacity(POSITIONS);

    for position in 0..POSITIONS {
        let hidden = synthetic_hidden(position);
        let q_projection = q.matvec(&hidden)?;
        let k_projection = k.matvec(&hidden)?;
        let v_projection = v.matvec(&hidden)?;
        let raw_query = query_from_interleaved_q_projection(&q_projection)?;
        let query = source_norm_partial_rope(
            &raw_query,
            &q_norm,
            QUERY_HEADS,
            position,
            "source Q RMSNorm + partial RoPE",
        )?;
        let key = source_norm_partial_rope(
            &k_projection,
            &k_norm,
            KV_HEADS,
            position,
            "source K RMSNorm + partial RoPE",
        )?;
        // Exact state sequencing is intentional: no causal read may occur
        // until the current K and V have both been appended.
        cache.append(position, &key, &v_projection)?;
        let attention = causal_gqa_attention(&query, &cache, position)?;
        let gate = sigmoid_gate(&gate_from_interleaved_q_projection(&q_projection)?)?;
        let gated = attention
            .output
            .iter()
            .zip(&gate)
            .map(|(attention, gate)| attention * gate)
            .collect::<Vec<_>>();
        finite(&gated, "source GQA gated attention")?;
        let output = o.matvec(&gated)?;

        token_hashes.push(TokenBoundaryHashes {
            position,
            source_hidden: boundary("source_hidden", vec![HIDDEN], &hidden)?,
            q_projection: boundary(
                "q_projection_interleaved_query_gate",
                vec![QUERY_HEADS, 2, HEAD_DIM],
                &q_projection,
            )?,
            k_projection: boundary("k_projection", vec![KV_HEADS, HEAD_DIM], &k_projection)?,
            v_projection: boundary("v_projection", vec![KV_HEADS, HEAD_DIM], &v_projection)?,
            query_after_q_norm_partial_rope: boundary(
                "query_after_q_norm_partial_rope",
                vec![QUERY_HEADS, HEAD_DIM],
                &query,
            )?,
            key_after_k_norm_partial_rope: boundary(
                "key_after_k_norm_partial_rope",
                vec![KV_HEADS, HEAD_DIM],
                &key,
            )?,
            active_key_cache_after_append: boundary(
                "active_key_cache_after_append",
                vec![position + 1, KV_HEADS, HEAD_DIM],
                cache.active_key(),
            )?,
            active_value_cache_after_append: boundary(
                "active_value_cache_after_append",
                vec![position + 1, KV_HEADS, HEAD_DIM],
                cache.active_value(),
            )?,
            causal_softmax_probabilities: boundary(
                "causal_softmax_probabilities",
                vec![QUERY_HEADS, position + 1],
                &attention.probabilities,
            )?,
            causal_attention: boundary(
                "causal_attention",
                vec![QUERY_HEADS, HEAD_DIM],
                &attention.output,
            )?,
            q_projection_sigmoid_gate: boundary(
                "q_projection_sigmoid_gate",
                vec![QUERY_HEADS, HEAD_DIM],
                &gate,
            )?,
            gated_attention: boundary("gated_attention", vec![QUERY_HEADS, HEAD_DIM], &gated)?,
            o_projection: boundary("o_projection", vec![HIDDEN], &output)?,
        });
        if position == 0 {
            after_position_zero = Some(cache.snapshot());
        }
    }

    let key_after_two_appends = boundary(
        "active_key_after_two_appends",
        vec![POSITIONS, KV_HEADS, HEAD_DIM],
        cache.active_key(),
    )?;
    let value_after_two_appends = boundary(
        "active_value_after_two_appends",
        vec![POSITIONS, KV_HEADS, HEAD_DIM],
        cache.active_value(),
    )?;
    let position_zero_snapshot = after_position_zero.ok_or_else(|| {
        "two-token fixture did not retain the position-zero rollback snapshot".to_string()
    })?;
    cache.restore(&position_zero_snapshot)?;
    let restored_position_zero_key = boundary(
        "restored_key_after_rollback_to_position_zero",
        vec![1, KV_HEADS, HEAD_DIM],
        cache.active_key(),
    )?;
    let restored_position_zero_value = boundary(
        "restored_value_after_rollback_to_position_zero",
        vec![1, KV_HEADS, HEAD_DIM],
        cache.active_value(),
    )?;
    let rollback_position_zero_exact = cache.key == position_zero_snapshot.key
        && cache.value == position_zero_snapshot.value
        && cache.committed_positions == position_zero_snapshot.committed_positions;
    cache.restore(&initial)?;
    let restored_empty_key = boundary(
        "restored_key_after_rollback_to_empty",
        vec![0, KV_HEADS, HEAD_DIM],
        cache.active_key(),
    )?;
    let restored_empty_value = boundary(
        "restored_value_after_rollback_to_empty",
        vec![0, KV_HEADS, HEAD_DIM],
        cache.active_value(),
    )?;
    let rollback_empty_exact = cache.key == initial.key
        && cache.value == initial.value
        && cache.committed_positions == initial.committed_positions;

    Ok((
        token_hashes,
        RollbackBoundaryHashes {
            initial_active_key: initial_key,
            initial_active_value: initial_value,
            active_key_after_two_appends: key_after_two_appends,
            active_value_after_two_appends: value_after_two_appends,
            restored_key_after_rollback_to_position_zero: restored_position_zero_key,
            restored_value_after_rollback_to_position_zero: restored_position_zero_value,
            restored_key_after_rollback_to_empty: restored_empty_key,
            restored_value_after_rollback_to_empty: restored_empty_value,
            rollback_restored_position_zero_exactly: rollback_position_zero_exact,
            rollback_restored_empty_exactly: rollback_empty_exact,
        },
    ))
}

fn parity_schema() -> DeviceParitySchema {
    DeviceParitySchema {
        schema: "hawking.ascension.qwen80_direct_packed_gqa_two_token_device_child_parity.v1",
        element_encoding: "f32_little_endian_contiguous_row_major",
        comparison_rule: "future device readback SHA-256 must exactly equal every named CPU boundary hash; no tolerance-only promotion",
        required_command_order: vec![
            "direct_packed_q_projection",
            "direct_packed_k_projection",
            "direct_packed_v_projection",
            "q_rms_norm_residual_scale_one_plus_weight",
            "q_partial_rope_non_interleaved_rotate_half_first_64",
            "k_rms_norm_residual_scale_one_plus_weight",
            "k_partial_rope_non_interleaved_rotate_half_first_64",
            "snapshot_caller_owned_active_kv_to_rollback",
            "append_key_at_current_position",
            "append_value_at_current_position",
            "causal_softmax_read_positions_zero_through_current_only",
            "q_projection_head_local_second_256_row_sigmoid_gate",
            "direct_packed_o_projection",
            "copy_named_boundary_readbacks",
            "rollback_caller_owned_active_kv_exactly",
        ],
        required_readback_boundaries: vec![
            "q_projection_interleaved_query_gate",
            "k_projection",
            "v_projection",
            "query_after_q_norm_partial_rope",
            "key_after_k_norm_partial_rope",
            "active_key_cache_after_append",
            "active_value_cache_after_append",
            "causal_softmax_probabilities",
            "causal_attention",
            "q_projection_sigmoid_gate",
            "gated_attention",
            "o_projection",
            "restored_key_after_rollback_to_position_zero",
            "restored_value_after_rollback_to_position_zero",
            "restored_key_after_rollback_to_empty",
            "restored_value_after_rollback_to_empty",
        ],
        cache_rollback_must_restore_exact_f32le_hashes: true,
    }
}

fn future_device_inputs() -> FutureDeviceInputs {
    FutureDeviceInputs {
        component_scope: "one source-bound full-attention layer-3 GQA mixer component over exactly two caller-authorized positions",
        source_projection_abi: source_projection_abis(),
        required_norm_tensors: vec![
            "model.layers.3.self_attn.q_norm.weight [256]",
            "model.layers.3.self_attn.k_norm.weight [256]",
        ],
        source_bound_hidden_inputs: vec![
            "two independently source-bound [2048] hidden vectors for positions 0 and 1",
            "authority proving layer-3 input provenance, session identity, and position",
        ],
        caller_owned_session_state: vec![
            "active K cache [2,2,256] f32 for layer 3 / GQA slot 0",
            "active V cache [2,2,256] f32 for layer 3 / GQA slot 0",
            "non-aliasing rollback K cache [2,2,256] f32",
            "non-aliasing rollback V cache [2,2,256] f32",
        ],
        same_command_buffer_requirements: vec![
            "all projection, norm/RoPE, append, causal-read, gate, O-projection, readback, and rollback commands encode on one caller-owned token command buffer per position",
            "current K/V append must happen before the causal read at that position",
            "the device child must retain every named f32 readback until host SHA-256 comparison completes",
        ],
        device_child_must_not_claim: vec![
            "full residual or MoE completion",
            "complete layer, token, decoder, generation, HCLI, TPS, TG, or tournament qualification",
            "source BF16/MPS fallback or a synthetic hidden state as model evidence",
        ],
    }
}

fn synthetic_fixture_description() -> SyntheticFixtureDescription {
    SyntheticFixtureDescription {
        payload_kind:
            "deterministic_algorithmic_synthetic_direct_binary_sign_bits_plus_binary_group_scales",
        source_artifact_payload_opened: false,
        source_hidden_is_synthetic: true,
        direct_packed_group_size: GROUP_SIZE,
        q_projection_layout: "[16 query heads][query 256 rows, sigmoid-gate 256 rows]",
        exact_operator_order: parity_schema().required_command_order,
    }
}

fn unsealed_preimage_sha256<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn value_at<'a>(value: &'a Value, path: &str) -> Result<&'a Value, String> {
    let mut current = value;
    for segment in path.split('.') {
        current = current.get(segment).ok_or_else(|| {
            format!("required authority field {path:?} is missing at {segment:?}")
        })?;
    }
    Ok(current)
}

fn require_string(value: &Value, path: &str, expected: &str) -> Result<(), String> {
    if value_at(value, path)?.as_str() != Some(expected) {
        return Err(format!(
            "authority field {path:?} drifted from its required value"
        ));
    }
    Ok(())
}

fn require_usize(value: &Value, path: &str, expected: usize) -> Result<(), String> {
    if value_at(value, path)?.as_u64() != Some(expected as u64) {
        return Err(format!(
            "authority field {path:?} drifted from its required value"
        ));
    }
    Ok(())
}

fn require_bool(value: &Value, path: &str, expected: bool) -> Result<(), String> {
    if value_at(value, path)?.as_bool() != Some(expected) {
        return Err(format!(
            "authority field {path:?} drifted from its required value"
        ));
    }
    Ok(())
}

fn require_usize_array(value: &Value, path: &str, expected: &[usize]) -> Result<(), String> {
    let observed = value_at(value, path)?
        .as_array()
        .ok_or_else(|| format!("authority field {path:?} must be an array"))?
        .iter()
        .map(|entry| entry.as_u64().and_then(|entry| usize::try_from(entry).ok()))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| format!("authority field {path:?} must be an unsigned-integer array"))?;
    if observed != expected {
        return Err(format!("authority field {path:?} geometry drifted"));
    }
    Ok(())
}

fn canonical_regular(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be an absolute path"));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot inspect {label} {path:?}: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{label} must be a regular, non-symlink file"));
    }
    path.canonicalize()
        .map_err(|error| format!("cannot canonicalize {label} {path:?}: {error}"))
}

fn load_json_evidence(path: &Path, label: &str) -> Result<(Value, FileEvidence), String> {
    let canonical = canonical_regular(path, label)?;
    let bytes = fs::read(&canonical)
        .map_err(|error| format!("cannot read {label} {canonical:?}: {error}"))?;
    let document: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("cannot parse {label} {canonical:?}: {error}"))?;
    if !document.is_object() {
        return Err(format!("{label} must be a JSON object"));
    }
    Ok((
        document,
        FileEvidence {
            path: canonical.display().to_string(),
            bytes: bytes.len(),
            sha256: format!("{:x}", Sha256::digest(bytes)),
        },
    ))
}

fn validate_inner_layer3_receipt(inner: &Value) -> Result<(), String> {
    require_string(inner, "schema", LAYER3_INNER_SCHEMA)?;
    require_string(inner, "status", LAYER3_INNER_STATUS)?;
    require_string(inner, "artifact.manifest_seal_sha256", MANIFEST_SEAL_SHA256)?;
    require_string(inner, "artifact.source_revision", SOURCE_REVISION)?;
    require_usize(inner, "source_bound_layer.layer", LAYER)?;
    require_string(inner, "source_bound_layer.kind", "full_attention")?;
    require_usize(inner, "source_bound_layer.geometry.hidden", HIDDEN)?;
    require_usize(
        inner,
        "source_bound_layer.geometry.query_heads",
        QUERY_HEADS,
    )?;
    require_usize(inner, "source_bound_layer.geometry.kv_heads", KV_HEADS)?;
    require_usize(inner, "source_bound_layer.geometry.head_dim", HEAD_DIM)?;
    require_usize_array(
        inner,
        "source_bound_layer.geometry.fixture_positions",
        &[0, 1],
    )?;
    require_usize_array(
        inner,
        "source_bound_layer.geometry.q_proj_shape",
        &[Q_PROJ_ROWS, HIDDEN],
    )?;
    require_usize_array(
        inner,
        "source_bound_layer.geometry.k_proj_shape",
        &[KV_DIM, HIDDEN],
    )?;
    require_usize_array(
        inner,
        "source_bound_layer.geometry.v_proj_shape",
        &[KV_DIM, HIDDEN],
    )?;
    require_usize_array(
        inner,
        "source_bound_layer.geometry.o_proj_shape",
        &[HIDDEN, QUERY_DIM],
    )?;
    require_usize_array(
        inner,
        "source_bound_layer.geometry.qk_norm_shape",
        &[HEAD_DIM],
    )?;
    require_usize(
        inner,
        "source_bound_layer.geometry.partial_rotary_dimensions",
        ROTARY_DIM,
    )?;
    require_string(
        inner,
        "source_bound_layer.tensors.q_proj",
        ProjectionRole::Q.tensor_name(),
    )?;
    require_string(
        inner,
        "source_bound_layer.tensors.k_proj",
        ProjectionRole::K.tensor_name(),
    )?;
    require_string(
        inner,
        "source_bound_layer.tensors.v_proj",
        ProjectionRole::V.tensor_name(),
    )?;
    require_string(
        inner,
        "source_bound_layer.tensors.o_proj",
        ProjectionRole::O.tensor_name(),
    )?;
    require_bool(inner, "cpu_oracle.two_token_causal_cache_exercised", true)?;
    require_bool(
        inner,
        "cpu_oracle.uses_only_admitted_compact_sign_and_fp16_group_scale_payload_decode",
        true,
    )?;
    require_bool(inner, "metal_execution.performed", true)?;
    require_bool(inner, "claim_boundary.component_stage_only", true)?;
    require_bool(
        inner,
        "claim_boundary.does_not_execute_complete_layer_or_48_layer_decoder",
        true,
    )?;
    require_bool(
        inner,
        "claim_boundary.does_not_generate_tokens_expose_hcli_or_measure_tps",
        true,
    )?;
    Ok(())
}

fn validate_outer_layer3_receipt(
    outer: &Value,
    outer_evidence: &FileEvidence,
    inner: &Value,
    inner_evidence: &FileEvidence,
) -> Result<(), String> {
    require_string(outer, "schema", LAYER3_OUTER_SCHEMA)?;
    require_string(outer, "status", LAYER3_OUTER_STATUS)?;
    require_string(outer, "seal_sha256", LAYER3_OUTER_SEAL)?;
    require_string(outer, "inner_probe_capture.schema", LAYER3_INNER_SCHEMA)?;
    require_string(outer, "inner_probe_capture.status", LAYER3_INNER_STATUS)?;
    require_string(outer, "inner_probe_capture.sha256", &inner_evidence.sha256)?;
    require_string(
        outer,
        "inner_probe_capture.receipt_unsigned_json_sha256",
        value_at(inner, "receipt_unsigned_json_sha256").and_then(|value| {
            value
                .as_str()
                .ok_or_else(|| "inner receipt digest is not a string".into())
        })?,
    )?;
    require_string(
        outer,
        "source_binding.expected_manifest_seal_sha256",
        MANIFEST_SEAL_SHA256,
    )?;
    require_string(
        outer,
        "source_binding.expected_source_revision",
        SOURCE_REVISION,
    )?;
    require_bool(outer, "claim_boundary.outer_terminal_capture_only", true)?;
    require_bool(
        outer,
        "claim_boundary.does_not_execute_a_complete_layer_or_decoder",
        true,
    )?;
    if outer_evidence.sha256.is_empty() {
        return Err("outer receipt file digest is unexpectedly empty".into());
    }
    Ok(())
}

fn validate_source_template_authority(source: &Value) -> Result<(), String> {
    require_string(source, "schema", SOURCE_TEMPLATE_SCHEMA)?;
    require_string(source, "status", SOURCE_TEMPLATE_STATUS)?;
    require_string(
        source,
        "unsealed_preimage_sha256",
        SOURCE_TEMPLATE_PREIMAGE_SHA256,
    )?;
    require_string(source, "source_authority.model_id", MODEL_ID)?;
    require_string(source, "source_authority.model_key", MODEL_KEY)?;
    require_string(
        source,
        "source_authority.source_repository",
        SOURCE_REPOSITORY,
    )?;
    require_string(source, "source_authority.source_revision", SOURCE_REVISION)?;
    require_string(
        source,
        "source_authority.source_config_sha256",
        SOURCE_CONFIG_SHA256,
    )?;
    require_usize(source, "source_authority.max_position_embeddings", 262_144)?;
    require_bool(
        source,
        "execution_boundary.live_packed_artifact_scan_performed",
        false,
    )?;
    require_bool(
        source,
        "execution_boundary.raw_weight_or_gravity_payload_opened",
        false,
    )?;
    require_bool(
        source,
        "execution_boundary.metal_device_or_dispatch_performed",
        false,
    )?;
    require_bool(
        source,
        "execution_boundary.model_or_decoder_execution_performed",
        false,
    )?;
    require_bool(
        source,
        "execution_boundary.server_or_hcli_execution_performed",
        false,
    )?;
    require_bool(
        source,
        "execution_boundary.tps_or_tg_measurement_performed",
        false,
    )?;
    Ok(())
}

fn bind_authorities(arguments: &Arguments) -> Result<BoundFixtureAuthority, String> {
    let (inner, inner_evidence) =
        load_json_evidence(&arguments.layer3_inner_receipt, "--layer3-inner-receipt")?;
    let (outer, outer_evidence) =
        load_json_evidence(&arguments.layer3_outer_receipt, "--layer3-outer-receipt")?;
    let (source, source_evidence) = load_json_evidence(
        &arguments.source_template_authority,
        "--source-template-authority",
    )?;
    validate_inner_layer3_receipt(&inner)?;
    validate_outer_layer3_receipt(&outer, &outer_evidence, &inner, &inner_evidence)?;
    validate_source_template_authority(&source)?;
    validate_source_geometry()?;
    let inner_unsigned_sha = value_at(&inner, "receipt_unsigned_json_sha256")?
        .as_str()
        .ok_or_else(|| "inner receipt unsigned digest is not a string".to_string())?
        .to_owned();
    Ok(BoundFixtureAuthority {
        sealed_layer3_outer_receipt: outer_evidence,
        layer3_inner_component_receipt: inner_evidence,
        source_template_authority: source_evidence,
        layer3_outer_seal_sha256: LAYER3_OUTER_SEAL,
        layer3_inner_unsigned_json_sha256: inner_unsigned_sha,
        source_config_sha256: SOURCE_CONFIG_SHA256,
        source_revision: SOURCE_REVISION,
        source_geometry_checked_via_public_schedule_resolver: true,
    })
}

struct Arguments {
    layer3_inner_receipt: PathBuf,
    layer3_outer_receipt: PathBuf,
    source_template_authority: PathBuf,
    out: Option<PathBuf>,
}

fn usage() -> &'static str {
    concat!(
        "usage: ascension_qwen80_direct_packed_gqa_two_token_cpu_oracle \\\n+    --layer3-inner-receipt ABSOLUTE_PATH \\\n+    --layer3-outer-receipt ABSOLUTE_PATH \\\n+    --source-template-authority ABSOLUTE_PATH \\\n+    [--out ABSOLUTE_NEW_FILE]"
    )
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut inner = None;
    let mut outer = None;
    let mut source = None;
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag:?}; {}", usage()))?;
        let path = PathBuf::from(value);
        match flag.as_str() {
            "--layer3-inner-receipt" => {
                if inner.replace(path).is_some() {
                    return Err(format!(
                        "--layer3-inner-receipt was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--layer3-outer-receipt" => {
                if outer.replace(path).is_some() {
                    return Err(format!(
                        "--layer3-outer-receipt was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--source-template-authority" => {
                if source.replace(path).is_some() {
                    return Err(format!(
                        "--source-template-authority was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--out" => {
                if out.replace(path).is_some() {
                    return Err(format!("--out was supplied more than once; {}", usage()));
                }
            }
            _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
        }
    }
    let inner = inner.ok_or_else(|| format!("missing --layer3-inner-receipt; {}", usage()))?;
    let outer = outer.ok_or_else(|| format!("missing --layer3-outer-receipt; {}", usage()))?;
    let source =
        source.ok_or_else(|| format!("missing --source-template-authority; {}", usage()))?;
    if !inner.is_absolute() || !outer.is_absolute() || !source.is_absolute() {
        return Err("all authority inputs must be absolute paths".into());
    }
    if let Some(path) = &out {
        if !path.is_absolute() {
            return Err("--out must be an absolute path".into());
        }
    }
    Ok(Arguments {
        layer3_inner_receipt: inner,
        layer3_outer_receipt: outer,
        source_template_authority: source,
        out,
    })
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "--out must have an existing parent directory".to_string())?;
    if !parent.is_dir() {
        return Err(format!(
            "--out parent {parent:?} is not an existing directory"
        ));
    }
    if path.exists() {
        return Err(format!("refusing to overwrite existing --out {path:?}"));
    }
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("qwen80-gqa-oracle"),
        process::id()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| format!("cannot create output temporary {temporary:?}: {error}"))?;
    if let Err(error) = file.write_all(bytes).and_then(|_| file.sync_all()) {
        let _ = fs::remove_file(&temporary);
        return Err(format!(
            "cannot write output temporary {temporary:?}: {error}"
        ));
    }
    drop(file);
    if let Err(error) = fs::hard_link(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(format!("cannot publish new output {path:?}: {error}"));
    }
    fs::remove_file(&temporary)
        .map_err(|error| format!("cannot retire output temporary {temporary:?}: {error}"))
}

fn build_report(authority: BoundFixtureAuthority) -> Result<OracleReport, String> {
    let (per_token_boundary_hashes, rollback_boundary_hashes) = run_synthetic_two_token_fixture()?;
    if !rollback_boundary_hashes.rollback_restored_position_zero_exactly
        || !rollback_boundary_hashes.rollback_restored_empty_exactly
    {
        return Err("synthetic GQA cache rollback did not restore an exact snapshot".into());
    }
    let report_without_digest = (
        RESULT_SCHEMA,
        RESULT_STATUS,
        &authority,
        source_geometry(),
        synthetic_fixture_description(),
        &per_token_boundary_hashes,
        &rollback_boundary_hashes,
        parity_schema(),
        future_device_inputs(),
    );
    let digest = unsealed_preimage_sha256(&report_without_digest)?;
    Ok(OracleReport {
        schema: RESULT_SCHEMA,
        status: RESULT_STATUS,
        source_fixture_authority: authority,
        source_geometry: source_geometry(),
        synthetic_direct_packed_fixture: synthetic_fixture_description(),
        per_token_boundary_hashes,
        rollback_boundary_hashes,
        future_device_parity_schema: parity_schema(),
        exact_future_device_inputs: future_device_inputs(),
        execution_boundary: ExecutionBoundary {
            fixture_component_only: true,
            deterministic_synthetic_weights_only: true,
            real_upstream_hidden_used: false,
            model_artifact_or_payload_opened: false,
            source_sidecar_opened: false,
            runtime_registry_watcher_server_or_lease_touched: false,
            metal_or_gpu_invoked: false,
            full_layer_token_decoder_hcli_tps_or_tg_claimed: false,
        },
        unsealed_preimage_sha256: digest,
    })
}

fn main() {
    let result = (|| -> Result<(), String> {
        let arguments = parse_arguments()?;
        let authority = bind_authorities(&arguments)?;
        let report = build_report(authority)?;
        let mut bytes = serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?;
        bytes.push(b'\n');
        if let Some(path) = arguments.out.as_deref() {
            write_new(path, &bytes)?;
        }
        print!("{}", String::from_utf8_lossy(&bytes));
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!("ascension_qwen80_direct_packed_gqa_two_token_cpu_oracle: {error}");
        process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_schedule_authority_keeps_layer_three_as_gqa() {
        validate_source_geometry().expect("exact Qwen80 source schedule must accept layer 3 GQA");
        assert_eq!(
            qwen80_layer_kind(3).unwrap(),
            Qwen80LayerKind::FullAttention
        );
        assert_eq!(
            qwen80_layer_kind(2).unwrap(),
            Qwen80LayerKind::LinearAttention
        );
        assert_eq!(
            qwen80_layer_kind(47).unwrap(),
            Qwen80LayerKind::FullAttention
        );
    }

    #[test]
    fn synthetic_direct_packed_matvec_matches_its_bit_and_scale_materialization() {
        let matrix = SyntheticDirectPackedMatrix::new(ProjectionRole::Q, 3, 8, 2).unwrap();
        let input = [0.5, -0.25, 0.75, -0.125, 0.25, 0.5, -0.5, 1.0];
        let projected = matrix.matvec(&input).unwrap();
        for row in 0..matrix.rows {
            let materialized = (0..matrix.columns)
                .map(|column| matrix.element(row, column).unwrap() * input[column])
                .sum::<f32>();
            assert_eq!(projected[row].to_bits(), materialized.to_bits());
        }
    }

    #[test]
    fn causal_read_requires_append_and_rollback_restores_the_exact_snapshot() {
        let mut cache = FixtureKvCache::empty();
        let query = vec![0.125; QUERY_DIM];
        assert!(causal_gqa_attention(&query, &cache, 0).is_err());
        let empty = cache.snapshot();
        let key0 = vec![0.25; KV_DIM];
        let value0 = vec![0.5; KV_DIM];
        cache.append(0, &key0, &value0).unwrap();
        let first = causal_gqa_attention(&query, &cache, 0).unwrap();
        assert_eq!(first.probabilities.len(), QUERY_HEADS);
        assert!(first
            .probabilities
            .iter()
            .all(|probability| probability.to_bits() == 1.0f32.to_bits()));
        let position_zero = cache.snapshot();
        cache
            .append(1, &vec![-0.25; KV_DIM], &vec![0.75; KV_DIM])
            .unwrap();
        let second = causal_gqa_attention(&query, &cache, 1).unwrap();
        assert_eq!(second.probabilities.len(), QUERY_HEADS * 2);
        cache.restore(&position_zero).unwrap();
        assert_eq!(cache.key, position_zero.key);
        assert_eq!(cache.value, position_zero.value);
        assert_eq!(cache.committed_positions, 1);
        cache.restore(&empty).unwrap();
        assert_eq!(cache.key, empty.key);
        assert_eq!(cache.value, empty.value);
        assert_eq!(cache.committed_positions, 0);
    }

    #[test]
    fn source_norm_rope_only_rotates_the_partial_prefix() {
        let raw = (0..HEAD_DIM)
            .map(|index| index as f32 / 256.0)
            .collect::<Vec<_>>();
        let weights = vec![0.0; HEAD_DIM];
        let position_zero = source_norm_partial_rope(&raw, &weights, 1, 0, "test").unwrap();
        let position_one = source_norm_partial_rope(&raw, &weights, 1, 1, "test").unwrap();
        assert_ne!(position_zero[..ROTARY_DIM], position_one[..ROTARY_DIM]);
        assert_eq!(position_zero[ROTARY_DIM..], position_one[ROTARY_DIM..]);
    }

    #[test]
    fn full_two_token_fixture_retains_every_future_device_boundary_and_rolls_back() {
        let (tokens, rollback) = run_synthetic_two_token_fixture().unwrap();
        assert_eq!(tokens.len(), POSITIONS);
        for (position, token) in tokens.iter().enumerate() {
            assert_eq!(token.position, position);
            assert_eq!(token.active_key_cache_after_append.shape[0], position + 1);
            assert_eq!(token.active_value_cache_after_append.shape[0], position + 1);
            assert_eq!(
                token.causal_softmax_probabilities.shape,
                vec![QUERY_HEADS, position + 1]
            );
            assert_eq!(token.o_projection.shape, vec![HIDDEN]);
            assert_eq!(token.o_projection.f32le_sha256.len(), 64);
        }
        assert!(rollback.rollback_restored_position_zero_exactly);
        assert!(rollback.rollback_restored_empty_exactly);
        assert_eq!(parity_schema().required_readback_boundaries.len(), 16);
        assert_eq!(future_device_inputs().source_projection_abi.len(), 4);
    }
}

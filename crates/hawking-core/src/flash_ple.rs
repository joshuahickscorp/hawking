//! Deterministic Qwen4-Exp PLE address and token-history owner.
//!
//! PLE is more than this address plane: native execution still needs admitted
//! lookup payloads, projections, grouped RMS norms, nonlinear gating, and the
//! dilated convolution. This module owns only the stable integer mechanics
//! that select ordered lookup rows and carry the two-token lexical state.
//! It performs no source read, model execution, Metal dispatch, or admission.

use crate::model::qwen80_source_bf16_layer_major::SourceBf16Index;
use crate::{Error, Result as HawkingResult};

const PRIME_1: u64 = 10_007;
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashPleAddressConfig {
    pub vocab_size: u64,
    pub eos_token_id: u64,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub ngram_vocab_size_base: u64,
    pub make_vocab_size_divisible_by: u64,
    pub ple_embed_dim: usize,
    pub ple_layer_index: usize,
    /// Zero-indexed decoder-layer index. The source config stores one-indexed
    /// PLE layer IDs, so Flash's `ple_layer_ids=[2]` resolves to layer 1.
    pub source_layer_index: usize,
    pub seed: u64,
}

impl FlashPleAddressConfig {
    /// Exact static geometry from the pinned Qwen3.8-Flash-Next source config.
    pub fn qwen38_flash_next_layer1() -> Self {
        Self {
            vocab_size: 248_320,
            eos_token_id: 248_044,
            ngram_size: 3,
            heads_per_ngram: 8,
            ngram_vocab_size_base: 20_000_000,
            make_vocab_size_divisible_by: 128,
            ple_embed_dim: 2_560,
            ple_layer_index: 0,
            source_layer_index: 1,
            seed: 1_234,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashPleAddressContract {
    pub config: FlashPleAddressConfig,
    pub multipliers: Vec<u64>,
    pub head_vocab_sizes: Vec<u64>,
    pub head_offsets: Vec<u64>,
    pub total_vocab_size: u64,
    pub padded_vocab_size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashPleRowAccess {
    pub ngram_order: usize,
    pub head_within_ngram: usize,
    pub global_head: usize,
    pub combined_row: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashPleTokenAccess {
    pub session_token_index: u64,
    pub segment_token_index: usize,
    pub token_id: u64,
    pub accesses: Vec<FlashPleRowAccess>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashPleAddressState {
    previous_context: Vec<u64>,
    session_token_index: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashPleLookupShard {
    pub ordinal: usize,
    pub tensor_name: String,
    pub global_start: u64,
    pub rows: u64,
    pub columns: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashPleLookupLayout {
    pub shards: Vec<FlashPleLookupShard>,
    pub combined_rows: u64,
    pub row_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashPleSourceRow {
    pub global_head: usize,
    pub combined_row: u64,
    pub shard_ordinal: usize,
    pub tensor_name: String,
    pub local_row: u64,
    pub bf16_bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashPleBf16Support {
    pub key_proj: Vec<u8>,
    pub value_proj: Vec<u8>,
    pub norm_key: Vec<u8>,
    pub norm_query: Vec<u8>,
    pub norm_conv: Vec<u8>,
    pub conv: Vec<u8>,
    pub hidden_size: usize,
    pub hc_count: usize,
    pub embed_dim: usize,
    pub conv_kernel_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlashPleNumericalState {
    /// Row-major `[state_len, hc_count * hidden_size]` F32 history.
    pub conv_state: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlashPleNumericalOutput {
    /// Row-major `[tokens, hc_count * hidden_size]` PLE injection.
    pub injection: Vec<f32>,
    pub state: FlashPleNumericalState,
}

fn model_error(message: impl Into<String>) -> Error {
    Error::Model(message.into())
}

fn bf16_at(bytes: &[u8], index: usize) -> f32 {
    let offset = index * 2;
    f32::from_bits((u16::from_le_bytes([bytes[offset], bytes[offset + 1]]) as u32) << 16)
}

fn decode_i64_le(bytes: &[u8]) -> Result<Vec<u64>, String> {
    if !bytes.len().is_multiple_of(8) {
        return Err("PLE I64 control has a partial element".into());
    }
    bytes
        .chunks_exact(8)
        .map(|chunk| {
            let value = i64::from_le_bytes(chunk.try_into().expect("eight-byte chunk"));
            u64::try_from(value).map_err(|_| "PLE I64 control contains a negative value".into())
        })
        .collect()
}

/// Read one source-declared I64 PLE address control.  Shape and raw bytes are
/// not sufficient evidence of integer semantics: a byte-compatible U64/F64
/// auxiliary tensor must fail before its payload is decoded as I64.
fn read_source_i64_control(
    index: &SourceBf16Index,
    name: &str,
    expected: &[u64],
) -> HawkingResult<()> {
    let dtype = index.aux_tensor_dtype(name)?;
    if dtype != "I64" {
        return Err(model_error(format!(
            "PLE address control {name} has dtype {dtype}, expected I64"
        )));
    }
    if index.aux_tensor_shape(name)? != [expected.len()] {
        return Err(model_error(format!(
            "PLE address control {name} has incompatible shape"
        )));
    }
    let observed = decode_i64_le(&index.read_aux_raw(name)?).map_err(model_error)?;
    if observed != expected {
        return Err(model_error(format!(
            "PLE address control {name} differs from the derived contract"
        )));
    }
    Ok(())
}

fn matvec_bf16(weights: &[u8], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|row| {
            let mut sum = 0.0f32;
            for col in 0..cols {
                sum += bf16_at(weights, row * cols + col) * input[col];
            }
            sum
        })
        .collect()
}

fn grouped_rms_norm_bf16(
    values: &[f32],
    weights: &[u8],
    group_size: usize,
    eps: f32,
) -> Result<Vec<f32>, String> {
    if values.is_empty()
        || values.len() * 2 != weights.len()
        || !values.len().is_multiple_of(group_size)
        || group_size == 0
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err("PLE grouped RMSNorm geometry or epsilon is invalid".into());
    }
    let mut output = vec![0.0f32; values.len()];
    for (group_index, group) in values.chunks_exact(group_size).enumerate() {
        let mean_square = group.iter().map(|value| value * value).sum::<f32>() / group_size as f32;
        let inverse_rms = (mean_square + eps).sqrt().recip();
        let base = group_index * group_size;
        for index in 0..group_size {
            output[base + index] =
                group[index] * inverse_rms * (1.0 + bf16_at(weights, base + index));
        }
    }
    Ok(output)
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(SPLITMIX_GAMMA);
    value = (value ^ (value >> 30)).wrapping_mul(SPLITMIX_M1);
    value = (value ^ (value >> 27)).wrapping_mul(SPLITMIX_M2);
    value ^ (value >> 31)
}

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut divisor = 3u64;
    while divisor <= value / divisor {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

fn nth_prime_after(start: u64, count: usize) -> Result<u64, String> {
    let mut candidate = start;
    for _ in 0..count {
        candidate = candidate
            .checked_add(1)
            .ok_or_else(|| "PLE prime search overflowed".to_string())?;
        while !is_prime(candidate) {
            candidate = candidate
                .checked_add(1)
                .ok_or_else(|| "PLE prime search overflowed".to_string())?;
        }
    }
    Ok(candidate)
}

impl FlashPleAddressContract {
    pub fn from_config(config: FlashPleAddressConfig) -> Result<Self, String> {
        if config.vocab_size == 0
            || config.ngram_size < 2
            || config.heads_per_ngram == 0
            || config.ngram_vocab_size_base < 2
            || config.make_vocab_size_divisible_by == 0
        {
            return Err("invalid PLE vocabulary, n-gram, or head geometry".into());
        }
        if config.eos_token_id >= config.vocab_size {
            return Err("PLE EOS token is outside the vocabulary".into());
        }
        let ngram_heads = (config.ngram_size - 1)
            .checked_mul(config.heads_per_ngram)
            .ok_or_else(|| "PLE head count overflowed".to_string())?;
        if config.ple_embed_dim == 0 || !config.ple_embed_dim.is_multiple_of(ngram_heads) {
            return Err("PLE embedding width must divide exactly into its heads".into());
        }

        let multiplier_max = (i64::MAX as u64) / config.vocab_size;
        let half_bound = (multiplier_max / 2).max(1);
        let base_seed = config
            .seed
            .wrapping_add(PRIME_1.wrapping_mul(config.ple_layer_index as u64));
        let multipliers = (0..config.ngram_size)
            .map(|index| {
                let seeded = base_seed
                    .wrapping_add(SPLITMIX_GAMMA.wrapping_mul((index as u64).wrapping_add(1)));
                2 * (splitmix64(seeded) % half_bound) + 1
            })
            .collect::<Vec<_>>();

        let mut head_vocab_sizes = Vec::with_capacity(ngram_heads);
        let mut head_offsets = Vec::with_capacity(ngram_heads);
        let mut total_vocab_size = 0u64;
        for head_index in 0..ngram_heads {
            let global_head = config
                .ple_layer_index
                .checked_mul(ngram_heads)
                .and_then(|value| value.checked_add(head_index))
                .ok_or_else(|| "PLE global head index overflowed".to_string())?;
            let size = nth_prime_after(config.ngram_vocab_size_base - 1, global_head + 1)?;
            head_offsets.push(total_vocab_size);
            head_vocab_sizes.push(size);
            total_vocab_size = total_vocab_size
                .checked_add(size)
                .ok_or_else(|| "PLE lookup vocabulary overflowed".to_string())?;
        }
        let divisor = config.make_vocab_size_divisible_by;
        let padded_vocab_size = total_vocab_size
            .checked_add(divisor - 1)
            .ok_or_else(|| "PLE padded vocabulary overflowed".to_string())?
            / divisor
            * divisor;
        Ok(Self {
            config,
            multipliers,
            head_vocab_sizes,
            head_offsets,
            total_vocab_size,
            padded_vocab_size,
        })
    }

    pub fn context_len(&self) -> usize {
        self.config.ngram_size - 1
    }

    pub fn ngram_heads(&self) -> usize {
        self.context_len() * self.config.heads_per_ngram
    }

    pub fn head_dim(&self) -> usize {
        self.config.ple_embed_dim / self.ngram_heads()
    }

    pub fn new_state(&self) -> FlashPleAddressState {
        FlashPleAddressState {
            previous_context: vec![self.config.eos_token_id; self.context_len()],
            session_token_index: 0,
        }
    }

    fn shifted_token(&self, history: &[u64], position: usize, shift: usize) -> u64 {
        if shift == 0 {
            return history[position];
        }
        let previous_eos = history[..position]
            .iter()
            .rposition(|token| *token == self.config.eos_token_id);
        let segment_start = previous_eos.map_or(0, |index| index + 1);
        if position - segment_start >= shift && position >= shift {
            history[position - shift]
        } else {
            self.config.eos_token_id
        }
    }

    pub fn trace_segment(
        &self,
        state: &mut FlashPleAddressState,
        tokens: &[u64],
    ) -> Result<Vec<FlashPleTokenAccess>, String> {
        if state.previous_context.len() != self.context_len() {
            return Err("PLE token-history state has incompatible geometry".into());
        }
        if tokens.iter().any(|token| *token >= self.config.vocab_size) {
            return Err("PLE token sequence contains an ID outside the vocabulary".into());
        }
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let mut history = state.previous_context.clone();
        history.extend_from_slice(tokens);
        let mut traced = Vec::with_capacity(tokens.len());
        for (segment_token_index, token_id) in tokens.iter().copied().enumerate() {
            let position = self.context_len() + segment_token_index;
            let shifted = (0..self.config.ngram_size)
                .map(|shift| self.shifted_token(&history, position, shift))
                .collect::<Vec<_>>();
            let mut accesses = Vec::with_capacity(self.ngram_heads());
            for ngram_order in 2..=self.config.ngram_size {
                let start = (ngram_order - 2) * self.config.heads_per_ngram;
                let stop = start + self.config.heads_per_ngram;
                let mut mixed_id = shifted[0]
                    .checked_mul(self.multipliers[0])
                    .ok_or_else(|| "PLE mixed token ID overflowed".to_string())?;
                for (prior, shifted_token) in shifted.iter().enumerate().take(ngram_order).skip(1) {
                    mixed_id ^= shifted_token
                        .checked_mul(self.multipliers[prior])
                        .ok_or_else(|| "PLE mixed token ID overflowed".to_string())?;
                }
                for global_head in start..stop {
                    let combined_row = mixed_id % self.head_vocab_sizes[global_head]
                        + self.head_offsets[global_head];
                    accesses.push(FlashPleRowAccess {
                        ngram_order,
                        head_within_ngram: global_head - start,
                        global_head,
                        combined_row,
                    });
                }
            }
            traced.push(FlashPleTokenAccess {
                session_token_index: state.session_token_index,
                segment_token_index,
                token_id,
                accesses,
            });
            state.session_token_index = state
                .session_token_index
                .checked_add(1)
                .ok_or_else(|| "PLE session token index overflowed".to_string())?;
        }
        let keep_from = history.len() - self.context_len();
        state.previous_context = history[keep_from..].to_vec();
        Ok(traced)
    }
}

impl FlashPleLookupLayout {
    /// Resolve the checkpoint's physically concatenated PLE shards using only
    /// admitted safetensors headers. No payload is read by this constructor.
    pub fn from_source_index(
        index: &SourceBf16Index,
        contract: &FlashPleAddressContract,
        split_parts: usize,
    ) -> HawkingResult<Self> {
        if split_parts == 0 {
            return Err(model_error("PLE lookup layout requires at least one shard"));
        }
        let columns = contract.head_dim();
        let mut shards = Vec::with_capacity(split_parts);
        let mut combined_rows = 0u64;
        for ordinal in 0..split_parts {
            let tensor_name = format!(
                "model.language_model.layers.{}.ple.ple_embedding.ngram_embedding.shard_{}.weight",
                contract.config.source_layer_index, ordinal
            );
            let shape = index.tensor_shape(&tensor_name)?;
            if shape.len() != 2 || shape[0] == 0 || shape[1] != columns {
                return Err(model_error(format!(
                    "PLE lookup shard {ordinal} has shape {shape:?}, expected [rows,{columns}]"
                )));
            }
            let rows = u64::try_from(shape[0])
                .map_err(|_| model_error("PLE lookup shard row count does not fit u64"))?;
            let expected_bytes = shape[0]
                .checked_mul(columns)
                .and_then(|value| value.checked_mul(2))
                .ok_or_else(|| model_error("PLE lookup shard byte count overflowed"))?;
            if index.tensor_nbytes(&tensor_name)? != expected_bytes {
                return Err(model_error(format!(
                    "PLE lookup shard {ordinal} byte count differs from BF16 geometry"
                )));
            }
            shards.push(FlashPleLookupShard {
                ordinal,
                tensor_name,
                global_start: combined_rows,
                rows,
                columns,
            });
            combined_rows = combined_rows
                .checked_add(rows)
                .ok_or_else(|| model_error("PLE combined row count overflowed"))?;
        }
        if combined_rows != contract.padded_vocab_size {
            return Err(model_error(format!(
                "PLE checkpoint rows {combined_rows} differ from address contract {}",
                contract.padded_vocab_size
            )));
        }
        Ok(Self {
            shards,
            combined_rows,
            row_bytes: columns * 2,
        })
    }

    pub fn locate(&self, combined_row: u64) -> HawkingResult<(&FlashPleLookupShard, u64)> {
        let shard = self
            .shards
            .partition_point(|candidate| candidate.global_start <= combined_row)
            .checked_sub(1)
            .and_then(|index| self.shards.get(index))
            .ok_or_else(|| {
                model_error(format!(
                    "PLE combined row is outside the checkpoint: {combined_row}"
                ))
            })?;
        let local_row = combined_row - shard.global_start;
        if local_row >= shard.rows {
            return Err(model_error(format!(
                "PLE combined row is outside resolved shard: {combined_row}"
            )));
        }
        Ok((shard, local_row))
    }

    /// Range-read the 16 ordered BF16 rows for one already-derived token.
    /// The giant table is never materialized; every access remains explicit.
    pub fn read_token_embedding_bf16(
        &self,
        index: &SourceBf16Index,
        contract: &FlashPleAddressContract,
        token: &FlashPleTokenAccess,
    ) -> HawkingResult<Vec<FlashPleSourceRow>> {
        if token.accesses.len() != contract.ngram_heads() {
            return Err(model_error(
                "PLE token access does not contain every ordered head",
            ));
        }
        let mut rows = Vec::with_capacity(token.accesses.len());
        for (expected_head, access) in token.accesses.iter().enumerate() {
            if access.global_head != expected_head {
                return Err(model_error(
                    "PLE token access heads are not in canonical order",
                ));
            }
            let head_start = contract.head_offsets[expected_head];
            let head_end = head_start + contract.head_vocab_sizes[expected_head];
            if access.combined_row < head_start || access.combined_row >= head_end {
                return Err(model_error(format!(
                    "PLE row {} escaped global head {expected_head}",
                    access.combined_row
                )));
            }
            let (shard, local_row) = self.locate(access.combined_row)?;
            let byte_offset = usize::try_from(local_row)
                .ok()
                .and_then(|row| row.checked_mul(self.row_bytes))
                .ok_or_else(|| model_error("PLE source row byte offset overflowed"))?;
            let bf16_bytes =
                index.read_raw_range(&shard.tensor_name, byte_offset, self.row_bytes)?;
            rows.push(FlashPleSourceRow {
                global_head: expected_head,
                combined_row: access.combined_row,
                shard_ordinal: shard.ordinal,
                tensor_name: shard.tensor_name.clone(),
                local_row,
                bf16_bytes,
            });
        }
        Ok(rows)
    }
}

impl FlashPleBf16Support {
    /// Load the bounded non-table PLE support and verify the checkpoint's I64
    /// address controls against Hawking's independently derived contract.
    pub fn load_source(
        index: &SourceBf16Index,
        contract: &FlashPleAddressContract,
        hidden_size: usize,
        hc_count: usize,
        conv_kernel_size: usize,
    ) -> HawkingResult<Self> {
        if hidden_size == 0 || hc_count == 0 || conv_kernel_size == 0 {
            return Err(model_error("PLE numerical geometry must be positive"));
        }
        let hc_hidden = hidden_size
            .checked_mul(hc_count)
            .ok_or_else(|| model_error("PLE HC width overflowed"))?;
        let prefix = format!(
            "model.language_model.layers.{}.ple.",
            contract.config.source_layer_index
        );
        let read_bf16 = |suffix: &str, expected: &[usize]| -> HawkingResult<Vec<u8>> {
            let name = format!("{prefix}{suffix}");
            if index.tensor_shape(&name)? != expected {
                return Err(model_error(format!(
                    "PLE support tensor {name} has incompatible shape"
                )));
            }
            let bytes = index.read_raw(&name)?;
            if bytes.len() != expected.iter().product::<usize>() * 2 {
                return Err(model_error(format!(
                    "PLE support tensor {name} has incompatible BF16 bytes"
                )));
            }
            Ok(bytes)
        };
        let read_i64 = |suffix: &str, expected: &[u64]| -> HawkingResult<()> {
            let name = format!("{prefix}ple_embedding.{suffix}");
            read_source_i64_control(index, &name, expected)
        };
        read_i64("layer_multipliers", &contract.multipliers)?;
        read_i64("ngram_heads_offsets", &contract.head_offsets)?;
        read_i64("ngram_heads_vocab_sizes", &contract.head_vocab_sizes)?;
        Ok(Self {
            key_proj: read_bf16(
                "key_proj.weight",
                &[hc_hidden, contract.config.ple_embed_dim],
            )?,
            value_proj: read_bf16(
                "value_proj.weight",
                &[hidden_size, contract.config.ple_embed_dim],
            )?,
            norm_key: read_bf16("norm_key.weight", &[hc_hidden])?,
            norm_query: read_bf16("norm_query.weight", &[hc_hidden])?,
            norm_conv: read_bf16("norm_conv.weight", &[hc_hidden])?,
            conv: read_bf16("conv1d.weight", &[hc_hidden, 1, conv_kernel_size])?,
            hidden_size,
            hc_count,
            embed_dim: contract.config.ple_embed_dim,
            conv_kernel_size,
        })
    }

    pub fn zero_state(&self, ngram_size: usize) -> Result<FlashPleNumericalState, String> {
        if ngram_size < 2 {
            return Err("PLE convolution dilation must be at least two".into());
        }
        let state_len = (self.conv_kernel_size - 1)
            .checked_mul(ngram_size)
            .ok_or_else(|| "PLE convolution state length overflowed".to_string())?;
        Ok(FlashPleNumericalState {
            conv_state: vec![0.0; state_len * self.hidden_size * self.hc_count],
        })
    }

    /// F32 source-formula oracle over admitted BF16 support and embeddings.
    /// This is a correctness owner, not the final Metal performance path.
    pub fn evaluate_f32(
        &self,
        hidden_states: &[f32],
        embeddings: &[f32],
        token_count: usize,
        ngram_size: usize,
        eps: f32,
        mut state: FlashPleNumericalState,
    ) -> Result<FlashPleNumericalOutput, String> {
        let hc_hidden = self.hidden_size * self.hc_count;
        if token_count == 0
            || hidden_states.len() != token_count * hc_hidden
            || embeddings.len() != token_count * self.embed_dim
            || self.key_proj.len() != hc_hidden * self.embed_dim * 2
            || self.value_proj.len() != self.hidden_size * self.embed_dim * 2
            || self.norm_key.len() != hc_hidden * 2
            || self.norm_query.len() != hc_hidden * 2
            || self.norm_conv.len() != hc_hidden * 2
            || self.conv.len() != hc_hidden * self.conv_kernel_size * 2
        {
            return Err("PLE numerical input or support geometry is inconsistent".into());
        }
        let state_len = (self.conv_kernel_size - 1)
            .checked_mul(ngram_size)
            .ok_or_else(|| "PLE convolution state length overflowed".to_string())?;
        if state.conv_state.len() != state_len * hc_hidden {
            return Err("PLE convolution state geometry is inconsistent".into());
        }
        let mut injection = Vec::with_capacity(token_count * hc_hidden);
        for token_index in 0..token_count {
            let hidden = &hidden_states[token_index * hc_hidden..(token_index + 1) * hc_hidden];
            let embedding =
                &embeddings[token_index * self.embed_dim..(token_index + 1) * self.embed_dim];
            let key = matvec_bf16(&self.key_proj, hc_hidden, self.embed_dim, embedding);
            let value = matvec_bf16(
                &self.value_proj,
                self.hidden_size,
                self.embed_dim,
                embedding,
            );
            let key_normed = grouped_rms_norm_bf16(&key, &self.norm_key, self.hidden_size, eps)?;
            let query_normed =
                grouped_rms_norm_bf16(hidden, &self.norm_query, self.hidden_size, eps)?;
            let mut gated = vec![0.0f32; hc_hidden];
            for stream in 0..self.hc_count {
                let base = stream * self.hidden_size;
                let dot = (0..self.hidden_size)
                    .map(|index| key_normed[base + index] * query_normed[base + index])
                    .sum::<f32>();
                let raw_gate = dot / (self.hidden_size as f32).sqrt();
                let signed_sqrt = raw_gate.signum() * raw_gate.abs().max(1.0e-6).sqrt();
                let gate = sigmoid(signed_sqrt);
                for index in 0..self.hidden_size {
                    gated[base + index] = gate * value[index];
                }
            }
            let normed = grouped_rms_norm_bf16(&gated, &self.norm_conv, self.hidden_size, eps)?;
            for feature in 0..hc_hidden {
                let mut convolved = 0.0f32;
                for kernel_index in 0..self.conv_kernel_size {
                    let input = if kernel_index + 1 == self.conv_kernel_size {
                        normed[feature]
                    } else {
                        state.conv_state[kernel_index * ngram_size * hc_hidden + feature]
                    };
                    convolved +=
                        bf16_at(&self.conv, feature * self.conv_kernel_size + kernel_index) * input;
                }
                injection.push(gated[feature] + convolved * sigmoid(convolved));
            }
            state.conv_state.rotate_left(hc_hidden);
            let tail = state.conv_state.len() - hc_hidden;
            state.conv_state[tail..].copy_from_slice(&normed);
        }
        if injection.iter().any(|value| !value.is_finite())
            || state.conv_state.iter().any(|value| !value.is_finite())
        {
            return Err("PLE numerical result contains non-finite values".into());
        }
        Ok(FlashPleNumericalOutput { injection, state })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Map, Value};
    use std::fs::File;
    use std::io::Write;

    const SYNTHETIC_I64_CONTROL: &str =
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers";

    fn synthetic_i64_control_index(dtype: &str, values: &[u64]) -> tempfile::TempDir {
        let directory = tempfile::tempdir().expect("tempdir");
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|value| {
                i64::try_from(*value)
                    .expect("non-negative I64")
                    .to_le_bytes()
            })
            .collect();
        let mut descriptors = Map::new();
        descriptors.insert(
            SYNTHETIC_I64_CONTROL.to_string(),
            json!({
                "dtype": dtype,
                "shape": [values.len()],
                "data_offsets": [0, payload.len()],
            }),
        );
        let header = serde_json::to_vec(&Value::Object(descriptors)).expect("serialize header");
        let shard_name = "model-00001-of-00001.safetensors";
        let mut shard = File::create(directory.path().join(shard_name)).expect("create shard");
        shard
            .write_all(&(header.len() as u64).to_le_bytes())
            .expect("write header length");
        shard.write_all(&header).expect("write header");
        shard.write_all(&payload).expect("write payload");
        std::fs::write(
            directory.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({"weight_map": {SYNTHETIC_I64_CONTROL: shard_name}}))
                .expect("serialize index"),
        )
        .expect("write index");
        directory
    }

    fn contract() -> FlashPleAddressContract {
        FlashPleAddressContract::from_config(FlashPleAddressConfig::qwen38_flash_next_layer1())
            .unwrap()
    }

    fn rows(records: &[FlashPleTokenAccess]) -> Vec<Vec<u64>> {
        records
            .iter()
            .map(|record| {
                record
                    .accesses
                    .iter()
                    .map(|access| access.combined_row)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn pinned_contract_matches_independent_source_formula_fixture() {
        let contract = contract();
        assert_eq!(
            contract.multipliers,
            vec![23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
        );
        assert_eq!(
            contract.head_vocab_sizes,
            vec![
                20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
                20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
                20_000_161, 20_000_171,
            ]
        );
        assert_eq!(contract.total_vocab_size, 320_001_446);
        assert_eq!(contract.padded_vocab_size, 320_001_536);
        assert_eq!(contract.head_dim(), 160);
        assert_eq!(contract.config.source_layer_index, 1);
    }

    #[test]
    fn bos_addresses_match_independent_python_trace() {
        let contract = contract();
        let mut state = contract.new_state();
        let traced = contract.trace_segment(&mut state, &[5_423]).unwrap();
        assert_eq!(
            rows(&traced)[0],
            vec![
                14_372_315,
                21_827_640,
                59_280_967,
                63_888_239,
                80_283_410,
                106_543_326,
                126_678_265,
                141_582_367,
                173_802_232,
                195_724_011,
                204_213_856,
                225_545_542,
                259_482_876,
                274_655_901,
                299_988_367,
                308_709_461,
            ]
        );
    }

    #[test]
    fn chunked_history_equals_one_segment() {
        let contract = contract();
        let tokens = [5_423, 799, 4_581, 3_817, 13];
        let mut whole_state = contract.new_state();
        let whole = contract.trace_segment(&mut whole_state, &tokens).unwrap();
        let mut chunked_state = contract.new_state();
        let mut chunked = contract
            .trace_segment(&mut chunked_state, &tokens[..1])
            .unwrap();
        chunked.extend(
            contract
                .trace_segment(&mut chunked_state, &tokens[1..3])
                .unwrap(),
        );
        chunked.extend(
            contract
                .trace_segment(&mut chunked_state, &tokens[3..])
                .unwrap(),
        );
        assert_eq!(rows(&whole), rows(&chunked));
        assert_eq!(whole_state, chunked_state);
        assert_eq!(whole_state.session_token_index, 5);
    }

    #[test]
    fn eos_resets_lexical_history_exactly() {
        let contract = contract();
        let mut state = contract.new_state();
        let traced = contract
            .trace_segment(&mut state, &[5_423, 248_044, 799])
            .unwrap();
        assert_eq!(
            rows(&traced)[2],
            vec![
                15_698_239,
                28_014_771,
                57_928_220,
                60_012_744,
                91_419_177,
                102_689_088,
                130_345_018,
                155_288_140,
                176_838_907,
                199_470_072,
                208_753_500,
                232_150_606,
                246_371_794,
                261_822_088,
                287_245_481,
                316_410_161,
            ]
        );
    }

    #[test]
    fn rejects_invalid_geometry_and_out_of_range_tokens() {
        let mut invalid = FlashPleAddressConfig::qwen38_flash_next_layer1();
        invalid.ple_embed_dim = 2_559;
        assert!(FlashPleAddressContract::from_config(invalid).is_err());
        let contract = contract();
        let mut state = contract.new_state();
        assert!(contract.trace_segment(&mut state, &[248_320]).is_err());
    }

    #[test]
    fn physical_layout_lookup_refuses_gaps_and_bounds() {
        let layout = FlashPleLookupLayout {
            shards: vec![
                FlashPleLookupShard {
                    ordinal: 0,
                    tensor_name: "s0".into(),
                    global_start: 0,
                    rows: 3,
                    columns: 2,
                },
                FlashPleLookupShard {
                    ordinal: 1,
                    tensor_name: "s1".into(),
                    global_start: 3,
                    rows: 2,
                    columns: 2,
                },
            ],
            combined_rows: 5,
            row_bytes: 4,
        };
        assert_eq!(layout.locate(0).unwrap().1, 0);
        assert_eq!(layout.locate(4).unwrap().1, 1);
        assert!(layout.locate(5).is_err());
    }

    #[test]
    fn source_i64_address_controls_reject_byte_compatible_non_i64_dtypes_before_read() {
        let expected = [23u64, 47u64, 89u64];
        let valid = synthetic_i64_control_index("I64", &expected);
        let valid_index = SourceBf16Index::open(valid.path()).expect("open I64 fixture");
        read_source_i64_control(&valid_index, SYNTHETIC_I64_CONTROL, &expected)
            .expect("I64 control must be admitted");
        assert_eq!(valid_index.bytes_read_total(), 24);

        let incompatible = synthetic_i64_control_index("U64", &expected);
        let incompatible_index =
            SourceBf16Index::open(incompatible.path()).expect("open U64 fixture");
        let error = read_source_i64_control(&incompatible_index, SYNTHETIC_I64_CONTROL, &expected)
            .expect_err("U64 bytes must not be decoded as an I64 source control");
        assert!(error.to_string().contains("dtype U64, expected I64"));
        assert_eq!(incompatible_index.bytes_read_total(), 0);
    }
}

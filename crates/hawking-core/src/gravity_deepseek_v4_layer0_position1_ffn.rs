//! Bounded source-derived CPU successor for DeepSeek-V4 layer 0, position 1.
//!
//! The sealed position-one complete-attention CPU oracle ends with a four-lane
//! BF16 mHC residual state.  This module consumes that exact state through the
//! source `hc_ffn_pre`, FFN norm, Gate/hash route for token `19923` (`Hello`),
//! six native FP4 routed experts, the native FP8 shared expert, source-order
//! F32 combine, and `hc_ffn_post`.
//!
//! It is deliberately a bounded CPU source-algorithm checkpoint, not a
//! registered decoder layer, Metal path, causal runtime, endpoint, generated
//! token, or TPS measurement.

use crate::gravity_deepseek_v4::DeepSeekV4FullStreamReader;
use crate::gravity_deepseek_v4_layer0_continuation::{
    layer0_position1_complete_attention_cpu_oracle, verify_layer0_position1_continuation_anchors,
    Layer0Position1CompleteAttentionCpuOracleResult, POSITION1_TOKEN_ID,
};
use crate::gravity_deepseek_v4_layer0_moe::{
    layer0_moe_successor_cpu_oracle, verify_layer0_moe_source_anchors,
    Layer0MoeSuccessorCpuOracleResult,
};
use crate::gravity_deepseek_v4_layer0_prefix::HC_FLAT_WIDTH;
use crate::{Error, Result};

/// Complete position-one FFN successor result.  `complete_attention` retains
/// the bounded causal predecessor so receipt producers can hash-bind the
/// exact four-lane state passed to `ffn` without persisting raw activations.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0Position1FullFfnCpuOracleResult {
    pub complete_attention: Layer0Position1CompleteAttentionCpuOracleResult,
    pub ffn: Layer0MoeSuccessorCpuOracleResult,
}

/// Verify both halves of the position-one source contract: the tokenizer and
/// causal-attention continuation, then the layer-0 hash-routed MoE grammar.
/// This fails closed before any FFN source tensor is interpreted.
pub fn verify_layer0_position1_full_ffn_source_anchors(
    reader: &DeepSeekV4FullStreamReader,
) -> Result<()> {
    verify_layer0_position1_continuation_anchors(reader)?;
    verify_layer0_moe_source_anchors(reader)?;
    Ok(())
}

/// Execute the real source-derived position-one layer-0 FFN successor.
///
/// The complete attention predecessor is recomputed from the admitted stream
/// rather than replaced by an arbitrary activation.  The MoE successor then
/// receives its exact `HC_MULT * HIDDEN_SIZE` BF16 output and indexes the
/// source `tid2eid` row using the verified `Hello` token ID.
pub fn layer0_position1_full_ffn_cpu_oracle(
    reader: &DeepSeekV4FullStreamReader,
) -> Result<Layer0Position1FullFfnCpuOracleResult> {
    verify_layer0_position1_full_ffn_source_anchors(reader)?;
    let complete_attention = layer0_position1_complete_attention_cpu_oracle(reader)?;
    if complete_attention.causal.token1_id != POSITION1_TOKEN_ID
        || complete_attention.hc_attention_post_bf16_bits.len() != HC_FLAT_WIDTH
    {
        return Err(position1(
            "position-one complete-attention predecessor does not expose the verified four-lane HC state",
        ));
    }
    let ffn = layer0_moe_successor_cpu_oracle(
        reader,
        POSITION1_TOKEN_ID,
        &complete_attention.hc_attention_post_bf16_bits,
    )?;
    if ffn.route.token_id != POSITION1_TOKEN_ID {
        return Err(position1(
            "position-one MoE successor did not retain the Hello tid2eid row identity",
        ));
    }
    Ok(Layer0Position1FullFfnCpuOracleResult {
        complete_attention,
        ffn,
    })
}

fn position1(message: impl Into<String>) -> Error {
    Error::Gravity(format!(
        "DeepSeek-V4 layer-0 position-one FFN: {}",
        message.into()
    ))
}

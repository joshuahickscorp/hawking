//! Qwen3-MoE forward pass. Standard MHA attention, no MLA. Top-8 of
//! 128 routed experts; no shared expert. Validates that the MoE
//! kernel pack isn't DeepSeek-shaped.
//!
//! The complete Qwen decoder remains unavailable.  The route topology and
//! Metal route dispatch below are deliberately usable independently of that
//! future decoder so a Qwen-specific kernel cannot be confused with a
//! DeepSeek-shaped path.  They are component primitives only: they do not
//! load weights, execute attention or experts, expose an Engine, or measure
//! model TPS.

use crate::engine::{Engine, EngineConfig, GenStats, GenerateRequest, StreamEvent};
use crate::{Error, Result};
use serde_json::Value;
use std::path::Path;

/// Exact routed-expert topology for a Qwen manager candidate.
///
/// This is intentionally narrower than a model configuration.  The values
/// identify the direct Metal router launch only; they do not claim a complete
/// model implementation or source-body admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QwenMoERouteTopology {
    pub model_id: &'static str,
    pub architecture: &'static str,
    pub n_experts: usize,
    pub top_k: usize,
}

pub const QWEN30_CODER_ROUTE_TOPOLOGY: QwenMoERouteTopology = QwenMoERouteTopology {
    model_id: "Qwen3-Coder-30B-A3B-Instruct",
    architecture: "Qwen3MoeForCausalLM",
    n_experts: 128,
    top_k: 8,
};

pub const QWEN80_CODER_NEXT_ROUTE_TOPOLOGY: QwenMoERouteTopology = QwenMoERouteTopology {
    model_id: "Qwen3-Coder-Next-80B",
    architecture: "Qwen3NextForCausalLM",
    n_experts: 512,
    top_k: 10,
};

/// Exact grouped-query attention geometry for Qwen3-Coder-30B-A3B-Instruct.
///
/// This binds the existing direct Metal GQA primitive to the official Qwen
/// shape.  It is intentionally an operator contract rather than a decoder
/// configuration: RoPE, projections, cache management, residuals, MoE, and
/// the token loop remain separate runtime obligations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QwenGqaAttentionTopology {
    pub model_id: &'static str,
    pub architecture: &'static str,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

pub const QWEN30_CODER_GQA_TOPOLOGY: QwenGqaAttentionTopology = QwenGqaAttentionTopology {
    model_id: "Qwen3-Coder-30B-A3B-Instruct",
    architecture: "Qwen3MoeForCausalLM",
    n_heads: 32,
    n_kv_heads: 4,
    head_dim: 128,
};

impl QwenMoERouteTopology {
    /// Reject dimensions that cannot be represented by the current direct
    /// route kernel.  This is a kernel-shape validation, not a model admission
    /// or performance gate.
    pub fn validate(self) -> Result<()> {
        if self.n_experts == 0 || self.n_experts > 512 {
            return Err(Error::Model(format!(
                "{} router n_experts={} outside supported direct-kernel range 1..=512",
                self.model_id, self.n_experts
            )));
        }
        if self.top_k == 0 || self.top_k > self.n_experts {
            return Err(Error::Model(format!(
                "{} router top_k={} must be in 1..={}",
                self.model_id, self.top_k, self.n_experts
            )));
        }
        Ok(())
    }
}

impl QwenGqaAttentionTopology {
    /// Validate the fixed Qwen30 GQA shape before a component dispatch.
    pub fn validate(self) -> Result<()> {
        if self.n_heads != 32 || self.n_kv_heads != 4 || self.head_dim != 128 {
            return Err(Error::Model(format!(
                "{} requires exact Qwen30 GQA geometry 32 query heads / 4 KV heads / dim 128, got {}/{}/{}",
                self.model_id, self.n_heads, self.n_kv_heads, self.head_dim
            )));
        }
        if self.n_heads % self.n_kv_heads != 0 {
            return Err(Error::Model(
                "Qwen30 GQA query heads must divide evenly by KV heads".into(),
            ));
        }
        Ok(())
    }
}

fn required_usize(value: &Value, path: &str) -> Result<usize> {
    value
        .pointer(path)
        .and_then(Value::as_u64)
        .map(|number| number as usize)
        .ok_or_else(|| Error::Model(format!("Qwen router configuration missing unsigned {path}")))
}

fn required_architecture(value: &Value, path: &str, expected: &str) -> Result<()> {
    let architectures = value
        .pointer(path)
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Model(format!("Qwen router configuration missing array {path}")))?;
    if architectures
        .iter()
        .any(|entry| entry.as_str() == Some(expected))
    {
        Ok(())
    } else {
        Err(Error::Model(format!(
            "Qwen router architecture at {path} does not contain {expected}"
        )))
    }
}

/// Bind the 30B router launch dimensions to its locally present official
/// configuration.  The caller retains responsibility for source integrity and
/// model-body qualification.
pub fn qwen30_route_topology_from_hf_config(config: &Value) -> Result<QwenMoERouteTopology> {
    required_architecture(
        config,
        "/architectures",
        QWEN30_CODER_ROUTE_TOPOLOGY.architecture,
    )?;
    if required_usize(config, "/num_experts")? != QWEN30_CODER_ROUTE_TOPOLOGY.n_experts
        || required_usize(config, "/num_experts_per_tok")? != QWEN30_CODER_ROUTE_TOPOLOGY.top_k
    {
        return Err(Error::Model(
            "Qwen30 router configuration disagrees with the exact 128-expert/top-8 kernel topology"
                .into(),
        ));
    }
    Ok(QWEN30_CODER_ROUTE_TOPOLOGY)
}

/// Bind the Qwen30 direct GQA component to all of the source configuration
/// dimensions that define its cache layout.  This rejects a merely
/// Qwen-looking configuration before any Metal dispatch can be described as a
/// Qwen30 component result.
pub fn qwen30_gqa_topology_from_hf_config(config: &Value) -> Result<QwenGqaAttentionTopology> {
    required_architecture(
        config,
        "/architectures",
        QWEN30_CODER_GQA_TOPOLOGY.architecture,
    )?;
    let actual = (
        required_usize(config, "/num_attention_heads")?,
        required_usize(config, "/num_key_value_heads")?,
        required_usize(config, "/head_dim")?,
    );
    let expected = (
        QWEN30_CODER_GQA_TOPOLOGY.n_heads,
        QWEN30_CODER_GQA_TOPOLOGY.n_kv_heads,
        QWEN30_CODER_GQA_TOPOLOGY.head_dim,
    );
    if actual != expected {
        return Err(Error::Model(format!(
            "Qwen30 attention configuration disagrees with exact 32/4/128 GQA topology: got {}/{}/{}, expected {}/{}/{}",
            actual.0, actual.1, actual.2, expected.0, expected.1, expected.2
        )));
    }
    QWEN30_CODER_GQA_TOPOLOGY.validate()?;
    Ok(QWEN30_CODER_GQA_TOPOLOGY)
}

/// Bind the 80B router launch dimensions to the strictly metadata-only source
/// admission record.  It intentionally accepts no weights and does not change
/// the rule that 80B body acquisition follows qualified 30B promotion.
pub fn qwen80_route_topology_from_metadata(metadata: &Value) -> Result<QwenMoERouteTopology> {
    required_architecture(
        metadata,
        "/architecture/architectures",
        QWEN80_CODER_NEXT_ROUTE_TOPOLOGY.architecture,
    )?;
    if required_usize(metadata, "/architecture/num_experts")?
        != QWEN80_CODER_NEXT_ROUTE_TOPOLOGY.n_experts
        || required_usize(metadata, "/architecture/num_experts_per_tok")?
            != QWEN80_CODER_NEXT_ROUTE_TOPOLOGY.top_k
    {
        return Err(Error::Model(
            "Qwen80 metadata disagrees with the exact 512-expert/top-10 kernel topology".into(),
        ));
    }
    Ok(QWEN80_CODER_NEXT_ROUTE_TOPOLOGY)
}

/// Dispatch exactly one Qwen router operation with a direct Metal kernel.
///
/// This is deliberately not wired into [`QwenMoE`] because the model engine is
/// still unimplemented.  The function is kept here so a measured route probe
/// has a typed Qwen binding rather than using generic dimensions ad hoc.
#[cfg(target_os = "macos")]
pub fn dispatch_qwen_router_component(
    metal: &crate::metal::MetalContext,
    topology: QwenMoERouteTopology,
    logits: &[f32],
) -> Result<(Vec<u32>, Vec<f32>)> {
    topology.validate()?;
    if logits.len() != topology.n_experts {
        return Err(Error::Model(format!(
            "{} router expected {} logits, received {}",
            topology.model_id,
            topology.n_experts,
            logits.len()
        )));
    }
    let logits_buffer = metal.new_buffer_with_bytes_checked(bytemuck::cast_slice(logits))?;
    let ids_buffer = metal.new_buffer_checked(topology.top_k * std::mem::size_of::<u32>())?;
    let weights_buffer = metal.new_buffer_checked(topology.top_k * std::mem::size_of::<f32>())?;
    let mut tcb = crate::metal::TokenCommandBuffer::new(metal);
    crate::kernels::moe_topk_gate_tcb(
        &mut tcb,
        &logits_buffer,
        &ids_buffer,
        &weights_buffer,
        topology.n_experts,
        topology.top_k,
    )?;
    tcb.commit_and_wait()?;

    // All buffers use StorageModeShared, so their contents are visible once
    // the command buffer completes.  Copy before the buffers are dropped.
    let ids = unsafe {
        std::slice::from_raw_parts(ids_buffer.contents() as *const u32, topology.top_k).to_vec()
    };
    let weights = unsafe {
        std::slice::from_raw_parts(weights_buffer.contents() as *const f32, topology.top_k).to_vec()
    };
    Ok((ids, weights))
}

/// Dispatch one source-shaped Qwen30 cached-decode GQA attention component.
///
/// This uses the real Metal GQA operator with the exact 32/4/128 Qwen30
/// geometry.  Inputs are caller-owned component data after Q/K projection and
/// RoPE; this function deliberately does not turn a successful attention
/// result into a complete decoder, a model TPS result, or a manager gate.
#[cfg(target_os = "macos")]
pub fn dispatch_qwen30_gqa_attention_component(
    metal: &crate::metal::MetalContext,
    topology: QwenGqaAttentionTopology,
    query: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    sequence_length: usize,
) -> Result<Vec<f32>> {
    topology.validate()?;
    if sequence_length == 0 {
        return Err(Error::Model(
            "Qwen30 GQA component requires a nonempty KV cache".into(),
        ));
    }
    let query_elements = topology.n_heads * topology.head_dim;
    let kv_elements = sequence_length * topology.n_kv_heads * topology.head_dim;
    if query.len() != query_elements
        || key_cache.len() != kv_elements
        || value_cache.len() != kv_elements
    {
        return Err(Error::Model(format!(
            "Qwen30 GQA component shape mismatch: q={} expected={query_elements}; k={} v={} expected={kv_elements}",
            query.len(), key_cache.len(), value_cache.len()
        )));
    }
    let mut output = vec![0.0f32; query_elements];
    crate::kernels::mha_decode_f32_metal(
        metal,
        query,
        key_cache,
        value_cache,
        sequence_length,
        topology.head_dim,
        topology.n_heads,
        topology.n_kv_heads,
        &mut output,
    )?;
    Ok(output)
}

pub struct QwenMoE {
    pub model_id: String,
}

impl Engine for QwenMoE {
    fn load(_weights: &Path, _config: EngineConfig) -> Result<Self> {
        Err(Error::Unimplemented(
            "qwen-moe: lands in Phase 3 (DeepSeek-V2-Lite ships first)",
        ))
    }

    fn generate(
        &mut self,
        _req: GenerateRequest,
        _sink: &mut dyn FnMut(StreamEvent),
    ) -> Result<GenStats> {
        Err(Error::Unimplemented("qwen-moe forward"))
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn model_arch(&self) -> &str {
        "qwen2"
    }

    fn forward_tokens_for_test(
        &mut self,
        _tokens: &[u32],
        _positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        Err(Error::Unimplemented("qwen-moe forward_tokens"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn qwen30_route_topology_requires_exact_source_shape() {
        let config = json!({
            "architectures": ["Qwen3MoeForCausalLM"],
            "num_experts": 128,
            "num_experts_per_tok": 8,
        });
        assert_eq!(
            qwen30_route_topology_from_hf_config(&config).unwrap(),
            QWEN30_CODER_ROUTE_TOPOLOGY
        );
        let wrong = json!({
            "architectures": ["Qwen3MoeForCausalLM"],
            "num_experts": 128,
            "num_experts_per_tok": 10,
        });
        assert!(qwen30_route_topology_from_hf_config(&wrong).is_err());
    }

    #[test]
    fn qwen30_gqa_topology_requires_exact_source_shape() {
        let config = json!({
            "architectures": ["Qwen3MoeForCausalLM"],
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
        });
        assert_eq!(
            qwen30_gqa_topology_from_hf_config(&config).unwrap(),
            QWEN30_CODER_GQA_TOPOLOGY
        );
        let wrong = json!({
            "architectures": ["Qwen3MoeForCausalLM"],
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        });
        assert!(qwen30_gqa_topology_from_hf_config(&wrong).is_err());
    }

    #[test]
    fn qwen80_route_topology_accepts_only_exact_metadata_shape() {
        let metadata = json!({
            "architecture": {
                "architectures": ["Qwen3NextForCausalLM"],
                "num_experts": 512,
                "num_experts_per_tok": 10,
            }
        });
        assert_eq!(
            qwen80_route_topology_from_metadata(&metadata).unwrap(),
            QWEN80_CODER_NEXT_ROUTE_TOPOLOGY
        );
        let wrong = json!({
            "architecture": {
                "architectures": ["Qwen3NextForCausalLM"],
                "num_experts": 256,
                "num_experts_per_tok": 10,
            }
        });
        assert!(qwen80_route_topology_from_metadata(&wrong).is_err());
    }
}

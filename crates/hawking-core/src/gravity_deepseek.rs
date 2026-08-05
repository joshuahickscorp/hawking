//! DeepSeek-V2-family `.gravity` architecture contract.
//!
//! DeepSeek is not a dense Llama-family model with a different header.  Its
//! decode state is multi-head latent attention (MLA), and its feed-forward
//! path is a routed MoE with a shared expert.  Keeping those facts in an
//! explicit contract prevents the adapter registry from accidentally sending
//! a DeepSeek artifact through the dense Llama path.  The contract is also
//! the boundary for a future executable MLA/MoE backend: every field that
//! changes the function of the model is required in the artifact header and
//! no value is guessed from a model name.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Mutex;

use crate::gravity::GravityWeights;
use crate::{Error, Result};

fn required_u64(a: &serde_json::Value, aliases: &[&str]) -> Result<u64> {
    aliases
        .iter()
        .find_map(|key| a.get(*key).and_then(serde_json::Value::as_u64))
        .ok_or_else(|| {
            Error::Gravity(format!(
                "architecture missing integer field (accepted aliases: {})",
                aliases.join(", ")
            ))
        })
}

fn optional_u64(a: &serde_json::Value, aliases: &[&str]) -> Option<u64> {
    aliases
        .iter()
        .find_map(|key| a.get(*key).and_then(serde_json::Value::as_u64))
}

fn required_f64(a: &serde_json::Value, aliases: &[&str]) -> Result<f64> {
    aliases
        .iter()
        .find_map(|key| a.get(*key).and_then(serde_json::Value::as_f64))
        .ok_or_else(|| {
            Error::Gravity(format!(
                "architecture missing numeric field (accepted aliases: {})",
                aliases.join(", ")
            ))
        })
}

fn required_bool(a: &serde_json::Value, aliases: &[&str]) -> Result<bool> {
    aliases
        .iter()
        .find_map(|key| a.get(*key).and_then(serde_json::Value::as_bool))
        .ok_or_else(|| {
            Error::Gravity(format!(
                "architecture missing boolean field (accepted aliases: {})",
                aliases.join(", ")
            ))
        })
}

fn required_string(a: &serde_json::Value, aliases: &[&str]) -> Result<String> {
    aliases
        .iter()
        .find_map(|key| {
            a.get(*key)
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        })
        .ok_or_else(|| {
            Error::Gravity(format!(
                "architecture missing string field (accepted aliases: {})",
                aliases.join(", ")
            ))
        })
}

fn rope_theta(a: &serde_json::Value) -> Result<f64> {
    a.get("rope_parameters")
        .and_then(|value| value.get("rope_theta"))
        .and_then(serde_json::Value::as_f64)
        .or_else(|| a.get("rope_theta").and_then(serde_json::Value::as_f64))
        .ok_or_else(|| Error::Gravity("architecture rope_theta missing".into()))
}

/// The two functional organs that distinguish DeepSeek-V2 from a dense
/// decoder.  These names are deliberately public so packers and receipts can
/// state the executable grammar without duplicating string literals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekAttentionKind {
    MultiHeadLatent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekLayerKind {
    Dense,
    SparseMoe,
}

/// Header-level architecture needed by an executable DeepSeek Gravity
/// adapter.  Values are canonicalized from either Hugging Face-style names or
/// the equivalent DeepSeek GGUF names emitted by the packer.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekGravityArch {
    pub model_type: String,
    pub n_layers: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub dense_intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub first_k_dense_layers: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub topk_method: String,
    pub scoring_func: String,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
}

impl DeepSeekGravityArch {
    /// Parse the canonical Gravity architecture block.
    ///
    /// The aliases are only for packer interoperability.  Defaults are not
    /// used for model-function fields: a missing routing or MLA field is a
    /// malformed artifact, not permission to run a guessed model.
    pub fn from_header(extra: &serde_json::Value) -> Result<Self> {
        let a = extra
            .get("architecture")
            .ok_or_else(|| Error::Gravity("shard header has no `architecture`".into()))?;
        let model_type = a.get("model_type").and_then(serde_json::Value::as_str);
        if !matches!(model_type, Some("deepseek2") | Some("deepseek_v2")) {
            return Err(Error::Gravity(format!(
                "gravity_deepseek: architecture.model_type is {model_type:?}, expected \"deepseek2\" or \"deepseek_v2\""
            )));
        }

        let arch = Self {
            model_type: model_type.expect("model_type matched above").to_string(),
            n_layers: required_u64(a, &["num_hidden_layers", "block_count"])? as usize,
            hidden: required_u64(a, &["hidden_size", "embedding_length"])? as usize,
            n_heads: required_u64(a, &["num_attention_heads", "head_count"])? as usize,
            n_kv_heads: required_u64(a, &["num_key_value_heads", "head_count_kv"])? as usize,
            // Some DeepSeek-V2-Lite exports carry `q_lora_rank: null` and
            // materialize a direct q_proj instead.  Zero is the explicit
            // direct-query variant; nonzero selects q_a/q_b MLA query.
            q_lora_rank: optional_u64(a, &["q_lora_rank"]).unwrap_or(0) as usize,
            kv_lora_rank: required_u64(a, &["kv_lora_rank"])? as usize,
            qk_nope_head_dim: required_u64(a, &["qk_nope_head_dim"])? as usize,
            qk_rope_head_dim: required_u64(a, &["qk_rope_head_dim"])? as usize,
            v_head_dim: required_u64(a, &["v_head_dim"])? as usize,
            dense_intermediate_size: required_u64(a, &["intermediate_size", "feed_forward_length"])?
                as usize,
            moe_intermediate_size: required_u64(
                a,
                &["moe_intermediate_size", "expert_feed_forward_length"],
            )? as usize,
            n_routed_experts: required_u64(
                a,
                &["n_routed_experts", "expert_count", "num_local_experts"],
            )? as usize,
            n_shared_experts: required_u64(a, &["n_shared_experts", "expert_shared_count"])?
                as usize,
            num_experts_per_tok: required_u64(a, &["num_experts_per_tok", "expert_used_count"])?
                as usize,
            first_k_dense_layers: required_u64(
                a,
                &["first_k_dense_replace", "leading_dense_block_count"],
            )? as usize,
            n_group: required_u64(a, &["n_group"])? as usize,
            topk_group: required_u64(a, &["topk_group"])? as usize,
            topk_method: required_string(a, &["topk_method"])?,
            scoring_func: required_string(a, &["scoring_func"])?,
            norm_topk_prob: required_bool(a, &["norm_topk_prob"])?,
            routed_scaling_factor: required_f64(a, &["routed_scaling_factor"])? as f32,
            vocab_size: required_u64(a, &["vocab_size"])? as usize,
            rope_theta: rope_theta(a)? as f32,
            rms_norm_eps: required_f64(a, &["rms_norm_eps", "rms_norm_epsilon"])? as f32,
            max_seq_len: required_u64(a, &["max_position_embeddings", "context_length"])? as usize,
        };
        arch.validate()?;
        Ok(arch)
    }

    /// Validate geometry that would otherwise produce a plausible but wrong
    /// decode.  This is intentionally independent of any tensor payload.
    pub fn validate(&self) -> Result<()> {
        let positive = [
            ("num_hidden_layers", self.n_layers),
            ("hidden_size", self.hidden),
            ("num_attention_heads", self.n_heads),
            ("num_key_value_heads", self.n_kv_heads),
            ("kv_lora_rank", self.kv_lora_rank),
            ("qk_nope_head_dim", self.qk_nope_head_dim),
            ("qk_rope_head_dim", self.qk_rope_head_dim),
            ("v_head_dim", self.v_head_dim),
            ("vocab_size", self.vocab_size),
            ("max_seq_len", self.max_seq_len),
        ];
        if let Some((name, _value)) = positive.into_iter().find(|(_, value)| *value == 0) {
            return Err(Error::Gravity(format!(
                "deepseek architecture.{name} must be positive"
            )));
        }
        if self.qk_rope_head_dim % 2 != 0 {
            return Err(Error::Gravity(
                "deepseek architecture.qk_rope_head_dim must be even for interleaved RoPE".into(),
            ));
        }
        if self.n_heads % self.n_kv_heads != 0 {
            return Err(Error::Gravity(format!(
                "deepseek num_attention_heads {} is not divisible by num_key_value_heads {}",
                self.n_heads, self.n_kv_heads
            )));
        }
        if self.first_k_dense_layers > self.n_layers {
            return Err(Error::Gravity(format!(
                "deepseek first_k_dense_replace {} exceeds {} layers",
                self.first_k_dense_layers, self.n_layers
            )));
        }
        if self.n_routed_experts == 0 || self.num_experts_per_tok == 0 {
            return Err(Error::Gravity(
                "deepseek routed MoE must have positive expert counts".into(),
            ));
        }
        if self.num_experts_per_tok > self.n_routed_experts {
            return Err(Error::Gravity(format!(
                "deepseek num_experts_per_tok {} exceeds n_routed_experts {}",
                self.num_experts_per_tok, self.n_routed_experts
            )));
        }
        if self.n_group == 0 || self.n_routed_experts % self.n_group != 0 {
            return Err(Error::Gravity(format!(
                "deepseek {} routed experts do not divide into {} groups",
                self.n_routed_experts, self.n_group
            )));
        }
        if self.topk_group == 0 || self.topk_group > self.n_group {
            return Err(Error::Gravity(format!(
                "deepseek topk_group {} is outside 1..={}",
                self.topk_group, self.n_group
            )));
        }
        let experts_per_group = self.n_routed_experts / self.n_group;
        if self.topk_group.saturating_mul(experts_per_group) < self.num_experts_per_tok {
            return Err(Error::Gravity(format!(
                "deepseek topk_group {} can expose only {} experts, below num_experts_per_tok {}",
                self.topk_group,
                self.topk_group.saturating_mul(experts_per_group),
                self.num_experts_per_tok
            )));
        }
        if !matches!(
            self.topk_method.as_str(),
            "greedy" | "gready" | "group_limited_greedy"
        ) {
            return Err(Error::Gravity(format!(
                "deepseek2 Gravity runtime does not support topk_method {:?}",
                self.topk_method
            )));
        }
        if !self.routed_scaling_factor.is_finite() || self.routed_scaling_factor <= 0.0 {
            return Err(Error::Gravity(
                "deepseek routed_scaling_factor must be finite and positive".into(),
            ));
        }
        if self.scoring_func != "softmax" {
            return Err(Error::Gravity(format!(
                "deepseek2 Gravity runtime only admits scoring_func=softmax, got {:?}",
                self.scoring_func
            )));
        }
        Ok(())
    }

    pub fn attention_kind(&self) -> DeepSeekAttentionKind {
        DeepSeekAttentionKind::MultiHeadLatent
    }

    pub fn layer_kind(&self, layer: usize) -> Result<DeepSeekLayerKind> {
        if layer >= self.n_layers {
            return Err(Error::Gravity(format!(
                "deepseek layer {layer} outside 0..{}",
                self.n_layers
            )));
        }
        Ok(if layer < self.first_k_dense_layers {
            DeepSeekLayerKind::Dense
        } else {
            DeepSeekLayerKind::SparseMoe
        })
    }

    /// The compact MLA state held per cached token when values are stored as
    /// f32.  This is a logical state number, not a TPS claim.
    pub fn compact_kv_bytes_per_token(&self, bytes_per_value: usize) -> usize {
        (self.kv_lora_rank + self.qk_rope_head_dim).saturating_mul(bytes_per_value)
    }

    /// Canonical tensor names consumed by the future executable adapter.  A
    /// packer may read GGUF names, but the artifact ABI is normalized here so
    /// the runtime never branches on source-export spelling.
    pub fn required_tensor_names(&self, layer: usize) -> Result<Vec<String>> {
        let kind = self.layer_kind(layer)?;
        let p = format!("model.layers.{layer}");
        let a = format!("{p}.self_attn");
        let mut names = vec![
            "model.embed_tokens.weight".into(),
            "model.norm.weight".into(),
            "lm_head.weight".into(),
            format!("{p}.input_layernorm.weight"),
            format!("{p}.post_attention_layernorm.weight"),
            format!("{a}.kv_a_proj_with_mqa.weight"),
            format!("{a}.kv_a_layernorm.weight"),
            format!("{a}.kv_b_proj.weight"),
            format!("{a}.o_proj.weight"),
        ];
        if self.q_lora_rank == 0 {
            names.push(format!("{a}.q_proj.weight"));
        } else {
            names.extend([
                format!("{a}.q_a_proj.weight"),
                format!("{a}.q_a_layernorm.weight"),
                format!("{a}.q_b_proj.weight"),
            ]);
        }
        match kind {
            DeepSeekLayerKind::Dense => {
                names.extend([
                    format!("{p}.mlp.gate_proj.weight"),
                    format!("{p}.mlp.up_proj.weight"),
                    format!("{p}.mlp.down_proj.weight"),
                ]);
            }
            DeepSeekLayerKind::SparseMoe => {
                names.push(format!("{p}.mlp.gate.weight"));
                for expert in 0..self.n_routed_experts {
                    names.extend([
                        format!("{p}.mlp.experts.{expert}.gate_proj.weight"),
                        format!("{p}.mlp.experts.{expert}.up_proj.weight"),
                        format!("{p}.mlp.experts.{expert}.down_proj.weight"),
                    ]);
                }
                for suffix in ["gate_proj", "up_proj", "down_proj"] {
                    names.push(format!("{p}.mlp.shared_experts.{suffix}.weight"));
                }
            }
        }
        Ok(names)
    }

    /// Fail closed before a pack is admitted if any canonical tensor is
    /// absent.  The returned list is bounded so a malformed 64-expert header
    /// cannot flood a caller's diagnostic.
    pub fn validate_tensor_names(&self, present: &HashSet<String>, layer: usize) -> Result<()> {
        let missing: Vec<String> = self
            .required_tensor_names(layer)?
            .into_iter()
            .filter(|name| !present.contains(name))
            .take(8)
            .collect();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(Error::Gravity(format!(
                "deepseek layer {layer} missing canonical tensors (first {}): {}",
                missing.len(),
                missing.join(", ")
            )))
        }
    }
}

#[derive(Default)]
struct DeepSeekLayerCache {
    /// Expanded keys, `[position][head][qk_nope + qk_rope]`.
    keys: Vec<f32>,
    /// Expanded values, `[position][head][v_head_dim]`.
    values: Vec<f32>,
}

struct DeepSeekSession {
    caches: Vec<DeepSeekLayerCache>,
}

impl DeepSeekSession {
    fn new(arch: &DeepSeekGravityArch) -> Self {
        Self {
            caches: (0..arch.n_layers)
                .map(|_| DeepSeekLayerCache::default())
                .collect(),
        }
    }

    fn reset(&mut self) {
        for cache in &mut self.caches {
            *cache = DeepSeekLayerCache::default();
        }
    }
}

/// CPU authority captured at functional boundaries for a single token.
///
/// This is intentionally a diagnostic surface, not a serving representation:
/// every vector is host-readable so a future resident MLA/MoE backend can
/// compare one layer at a time before it is allowed to replace the oracle.
#[derive(Debug, Clone, Default)]
pub struct DeepSeekLayerStageTrace {
    pub layer: usize,
    pub input: Vec<f32>,
    pub attn_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub kv_a: Vec<f32>,
    pub c_kv: Vec<f32>,
    pub k_pe: Vec<f32>,
    pub context: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub after_attention: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub router_logits: Vec<f32>,
    pub routes: Vec<(usize, f32)>,
    pub ffn_out: Vec<f32>,
    pub output: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct DeepSeekTokenStageTrace {
    pub token: u32,
    pub position: usize,
    pub layers: Vec<DeepSeekLayerStageTrace>,
    pub final_norm: Vec<f32>,
    pub logits: Vec<f32>,
}

/// Correctness-first DeepSeek Gravity runtime.
///
/// This is deliberately the small, readable oracle path: it consumes the
/// canonical Gravity tensor ABI directly, keeps an expanded CPU KV cache, and
/// executes MLA plus routed/shared SwiGLU.  It is not a TG speed claim.  The
/// future Metal backend must match this path before it is allowed to replace
/// it in the registry.
pub struct GravityDeepSeek {
    pub arch: DeepSeekGravityArch,
    weights: GravityWeights,
    session: Mutex<DeepSeekSession>,
}

impl GravityDeepSeek {
    pub fn open(path: &Path, verify_hash: bool) -> Result<Self> {
        let weights = GravityWeights::open(path, verify_hash)?;
        Self::from_weights(weights)
    }

    pub fn open_dir(dir: &Path, verify_hash: bool) -> Result<Self> {
        let weights = GravityWeights::open_dir(dir, verify_hash)?;
        Self::from_weights(weights)
    }

    fn from_weights(weights: GravityWeights) -> Result<Self> {
        let arch = DeepSeekGravityArch::from_header(&weights.header)?;
        let present: HashSet<String> = weights.tensor_names().into_iter().collect();
        // Validate every layer's ABI before the first token.  This keeps a
        // partially packed model from failing only after a long prefill.
        for layer in 0..arch.n_layers {
            arch.validate_tensor_names(&present, layer)?;
        }
        Ok(Self {
            session: Mutex::new(DeepSeekSession::new(&arch)),
            arch,
            weights,
        })
    }

    pub fn forward(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut session = self.session.lock().expect("deepseek session mutex");
        session.reset();
        forward_impl(&self.weights, &self.arch, &mut session, tokens, 0, None)
    }

    pub fn forward_at(&self, tokens: &[u32], start_pos: usize) -> Result<Vec<f32>> {
        let mut session = self.session.lock().expect("deepseek session mutex");
        forward_impl(
            &self.weights,
            &self.arch,
            &mut session,
            tokens,
            start_pos,
            None,
        )
    }

    /// Run the CPU authority and retain every load-bearing MLA/MoE boundary.
    /// The session is reset exactly like [`Self::forward`], so the returned
    /// trace is suitable for a same-prompt, same-position Metal comparison.
    pub fn forward_with_stage_trace(
        &self,
        tokens: &[u32],
    ) -> Result<(Vec<f32>, Vec<DeepSeekTokenStageTrace>)> {
        let mut session = self.session.lock().expect("deepseek session mutex");
        session.reset();
        let mut trace = Vec::with_capacity(tokens.len());
        let logits = forward_impl(
            &self.weights,
            &self.arch,
            &mut session,
            tokens,
            0,
            Some(&mut trace),
        )?;
        Ok((logits, trace))
    }
}

fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
    if x.len() != weight.len() || x.is_empty() {
        return Err(Error::Gravity(format!(
            "deepseek rmsnorm shape {} != {}",
            x.len(),
            weight.len()
        )));
    }
    let mean_sq = x.iter().map(|value| value * value).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    Ok(x.iter()
        .zip(weight)
        .map(|(value, scale)| value * inv * scale)
        .collect())
}

fn silu_mul(gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
    if gate.len() != up.len() {
        return Err(Error::Gravity(format!(
            "deepseek SwiGLU shape {} != {}",
            gate.len(),
            up.len()
        )));
    }
    Ok(gate
        .iter()
        .zip(up)
        .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
        .collect())
}

fn rope_interleaved(v: &[f32], pos: usize, theta: f32) -> Result<Vec<f32>> {
    if v.len() % 2 != 0 {
        return Err(Error::Gravity(
            "deepseek RoPE input must have even width".into(),
        ));
    }
    let half = v.len() / 2;
    let mut out = vec![0.0f32; v.len()];
    for i in 0..half {
        let inv_freq = 1.0f32 / theta.powf(2.0 * i as f32 / v.len() as f32);
        let angle = pos as f32 * inv_freq;
        let (sin, cos) = angle.sin_cos();
        let first = v[2 * i];
        let second = v[2 * i + 1];
        out[i] = first * cos - second * sin;
        out[half + i] = second * cos + first * sin;
    }
    Ok(out)
}

fn dense_mlp(weights: &GravityWeights, prefix: &str, x: &[f32]) -> Result<Vec<f32>> {
    let gate = weights.matvec(&format!("{prefix}.gate_proj.weight"), x)?;
    let up = weights.matvec(&format!("{prefix}.up_proj.weight"), x)?;
    let act = silu_mul(&gate, &up)?;
    weights.matvec(&format!("{prefix}.down_proj.weight"), &act)
}

fn topk_desc_with_epsilon(values: &[f32], k: usize, tie_epsilon: f32) -> Vec<usize> {
    // Do not express epsilon ties as a sort comparator: a tolerance relation
    // is intentionally non-transitive (a~b and b~c need not imply a~c),
    // which would violate Rust's total-order contract and can panic.  This
    // masked best-of-K scan is the same grammar as the Metal kernel and is
    // deterministic for every finite/NaN input.
    let mut available = vec![true; values.len()];
    let mut out = Vec::with_capacity(k.min(values.len()));
    for _ in 0..k.min(values.len()) {
        let mut best: Option<usize> = None;
        for i in 0..values.len() {
            if !available[i] {
                continue;
            }
            let replace = match best {
                None => true,
                Some(j) => {
                    let vi = values[i];
                    let vj = values[j];
                    let tied = tie_epsilon > 0.0
                        && vi.is_finite()
                        && vj.is_finite()
                        && (vi - vj).abs() <= tie_epsilon;
                    (vi > vj && !tied) || (tied && i < j)
                }
            };
            if replace {
                best = Some(i);
            }
        }
        let Some(index) = best else { break };
        available[index] = false;
        out.push(index);
    }
    out
}

/// DeepSeek-V2's router: softmax scores, optional group-limited selection,
/// then either top-k normalization or the configured route scale.  This is
/// separate from GLM's sigmoid/correction-bias `noaux_tc` router.
fn deepseek_routes(arch: &DeepSeekGravityArch, logits: &mut [f32]) -> Result<Vec<(usize, f32)>> {
    if logits.len() != arch.n_routed_experts {
        return Err(Error::Gravity(format!(
            "deepseek router produced {} logits, expected {}",
            logits.len(),
            arch.n_routed_experts
        )));
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0f32;
    for value in logits.iter_mut() {
        *value = (*value - max_logit).exp();
        total += *value;
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::Gravity(
            "deepseek router softmax is non-finite".into(),
        ));
    }
    for value in logits.iter_mut() {
        *value /= total;
    }

    let tie_epsilon = crate::moe::route_tie_epsilon();
    let mut allowed = vec![true; logits.len()];
    if arch.topk_method == "group_limited_greedy" {
        let per_group = arch.n_routed_experts / arch.n_group;
        let group_scores: Vec<f32> = (0..arch.n_group)
            .map(|group| {
                logits[group * per_group..(group + 1) * per_group]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect();
        let chosen_groups = topk_desc_with_epsilon(&group_scores, arch.topk_group, tie_epsilon);
        allowed.fill(false);
        for group in chosen_groups {
            for expert in group * per_group..(group + 1) * per_group {
                allowed[expert] = true;
            }
        }
    }

    let mut masked = logits.to_vec();
    for (score, is_allowed) in masked.iter_mut().zip(allowed) {
        if !is_allowed {
            *score = f32::NEG_INFINITY;
        }
    }
    let indices = topk_desc_with_epsilon(&masked, arch.num_experts_per_tok, tie_epsilon);
    let mut routes: Vec<(usize, f32)> = indices
        .into_iter()
        .map(|index| (index, logits[index]))
        .collect();
    if arch.num_experts_per_tok > 1 && arch.norm_topk_prob {
        let denom = routes.iter().map(|(_, weight)| *weight).sum::<f32>() + 1e-20;
        for (_, weight) in routes.iter_mut() {
            *weight /= denom;
        }
    } else {
        for (_, weight) in routes.iter_mut() {
            *weight *= arch.routed_scaling_factor;
        }
    }
    Ok(routes)
}

fn routed_mlp_with_trace(
    weights: &GravityWeights,
    arch: &DeepSeekGravityArch,
    prefix: &str,
    x: &[f32],
    mut trace: Option<&mut (Vec<f32>, Vec<(usize, f32)>)>,
) -> Result<(Vec<f32>, Vec<f32>, Vec<(usize, f32)>)> {
    let mut logits = weights.matvec(&format!("{prefix}.gate.weight"), x)?;
    // Keep the pre-softmax router values for stage parity. `deepseek_routes`
    // normalizes its input in place, so cloning after that call would label
    // probabilities as logits and make CPU/Metal boundary comparisons lie.
    let raw_router_logits = logits.clone();
    let routes = deepseek_routes(arch, &mut logits)?;
    if let Some((dst_logits, dst_routes)) = trace.as_deref_mut() {
        dst_logits.clear();
        dst_logits.extend_from_slice(&raw_router_logits);
        dst_routes.clear();
        dst_routes.extend_from_slice(&routes);
    }
    let mut out = vec![0.0f32; x.len()];
    for (expert, weight) in routes {
        let expert_prefix = format!("{prefix}.experts.{expert}");
        let expert_out = dense_mlp(weights, &expert_prefix, x)?;
        for (dst, value) in out.iter_mut().zip(expert_out) {
            // `deepseek_routes` owns the V2 route-scale application when
            // `norm_topk_prob` is false.  Applying it here as well would
            // square the configured scale and silently change the model.
            *dst += value * weight;
        }
    }
    if arch.n_shared_experts > 0 {
        // DeepSeek-V2 stores the always-on shared experts as one fused
        // projection whose intermediate width is n_shared * moe_width.
        let shared = dense_mlp(weights, &format!("{prefix}.shared_experts"), x)?;
        for (dst, value) in out.iter_mut().zip(shared) {
            *dst += value;
        }
    }
    Ok((
        out,
        raw_router_logits,
        trace.map(|(_, routes)| routes.clone()).unwrap_or_default(),
    ))
}

fn forward_impl(
    weights: &GravityWeights,
    arch: &DeepSeekGravityArch,
    session: &mut DeepSeekSession,
    tokens: &[u32],
    start_pos: usize,
    mut trace_out: Option<&mut Vec<DeepSeekTokenStageTrace>>,
) -> Result<Vec<f32>> {
    if tokens.is_empty() {
        return Err(Error::Gravity("deepseek forward: no tokens".into()));
    }
    let qk_dim = arch.qk_nope_head_dim + arch.qk_rope_head_dim;
    let per_head_kv = arch.qk_nope_head_dim + arch.v_head_dim;
    let mut logits = Vec::new();

    for (offset, &token) in tokens.iter().enumerate() {
        let pos = start_pos + offset;
        let tracing = trace_out.is_some();
        let mut token_trace = tracing.then(|| DeepSeekTokenStageTrace {
            token,
            position: pos,
            ..Default::default()
        });
        if token as usize >= arch.vocab_size {
            return Err(Error::Gravity(format!(
                "deepseek token {token} out of range for vocab_size {}",
                arch.vocab_size
            )));
        }
        let mut x = weights.row("model.embed_tokens.weight", token as usize, arch.hidden)?;
        if x.len() != arch.hidden {
            return Err(Error::Gravity(format!(
                "deepseek embedding row {} != hidden {}",
                x.len(),
                arch.hidden
            )));
        }

        for layer in 0..arch.n_layers {
            let p = format!("model.layers.{layer}");
            let attn = format!("{p}.self_attn");
            let layer_input = if tracing { x.clone() } else { Vec::new() };
            let h = rmsnorm(
                &x,
                &weights.dense(&format!("{p}.input_layernorm.weight"))?,
                arch.rms_norm_eps,
            )?;
            let layer_attn_norm = if tracing { h.clone() } else { Vec::new() };

            let q = if arch.q_lora_rank == 0 {
                weights.matvec(&format!("{attn}.q_proj.weight"), &h)?
            } else {
                let q_a = weights.matvec(&format!("{attn}.q_a_proj.weight"), &h)?;
                let q_a_norm = rmsnorm(
                    &q_a,
                    &weights.dense(&format!("{attn}.q_a_layernorm.weight"))?,
                    arch.rms_norm_eps,
                )?;
                weights.matvec(&format!("{attn}.q_b_proj.weight"), &q_a_norm)?
            };
            let layer_q = if tracing { q.clone() } else { Vec::new() };

            let compressed = weights.matvec(&format!("{attn}.kv_a_proj_with_mqa.weight"), &h)?;
            if compressed.len() < arch.kv_lora_rank + arch.qk_rope_head_dim {
                return Err(Error::Gravity(format!(
                    "deepseek MLA compressed projection has {} values, expected at least {}",
                    compressed.len(),
                    arch.kv_lora_rank + arch.qk_rope_head_dim
                )));
            }
            let layer_kv_a = if tracing {
                compressed.clone()
            } else {
                Vec::new()
            };
            let k_latent = rmsnorm(
                &compressed[..arch.kv_lora_rank],
                &weights.dense(&format!("{attn}.kv_a_layernorm.weight"))?,
                arch.rms_norm_eps,
            )?;
            let layer_c_kv = if tracing {
                k_latent.clone()
            } else {
                Vec::new()
            };
            let k_rope = rope_interleaved(
                &compressed[arch.kv_lora_rank..arch.kv_lora_rank + arch.qk_rope_head_dim],
                pos,
                arch.rope_theta,
            )?;
            let layer_k_pe = if tracing { k_rope.clone() } else { Vec::new() };
            let kv = weights.matvec(&format!("{attn}.kv_b_proj.weight"), &k_latent)?;
            if q.len() != arch.n_heads * qk_dim || kv.len() != arch.n_heads * per_head_kv {
                return Err(Error::Gravity(format!(
                    "deepseek MLA projection shape q={} kv={}, expected q={} kv={}",
                    q.len(),
                    kv.len(),
                    arch.n_heads * qk_dim,
                    arch.n_heads * per_head_kv
                )));
            }

            let cache = &mut session.caches[layer];
            for head in 0..arch.n_heads {
                let q_src = &q[head * qk_dim..(head + 1) * qk_dim];
                let q_rope =
                    rope_interleaved(&q_src[arch.qk_nope_head_dim..], pos, arch.rope_theta)?;
                let kv_src = &kv[head * per_head_kv..(head + 1) * per_head_kv];
                cache
                    .keys
                    .extend_from_slice(&kv_src[..arch.qk_nope_head_dim]);
                cache.keys.extend_from_slice(&k_rope);
                cache
                    .values
                    .extend_from_slice(&kv_src[arch.qk_nope_head_dim..]);

                // The attention loop below consumes the just-appended cache;
                // keep the query in a temporary row-major buffer to avoid
                // reconstructing it for every cached position.
                let _ = q_rope;
            }

            let mut context = vec![0.0f32; arch.n_heads * arch.v_head_dim];
            let n_keys = cache.keys.len() / (arch.n_heads * qk_dim);
            for head in 0..arch.n_heads {
                let q_src = &q[head * qk_dim..(head + 1) * qk_dim];
                let q_rope =
                    rope_interleaved(&q_src[arch.qk_nope_head_dim..], pos, arch.rope_theta)?;
                let mut query = vec![0.0f32; qk_dim];
                query[..arch.qk_nope_head_dim].copy_from_slice(&q_src[..arch.qk_nope_head_dim]);
                query[arch.qk_nope_head_dim..].copy_from_slice(&q_rope);

                let mut scores = vec![f32::NEG_INFINITY; n_keys];
                let scale = (qk_dim as f32).powf(-0.5);
                let mut best = f32::NEG_INFINITY;
                for (t, score) in scores.iter_mut().enumerate() {
                    let off = (t * arch.n_heads + head) * qk_dim;
                    let dot = query
                        .iter()
                        .zip(&cache.keys[off..off + qk_dim])
                        .map(|(a, b)| a * b)
                        .sum::<f32>();
                    *score = dot * scale;
                    best = best.max(*score);
                }
                let mut total = 0.0f32;
                for score in &mut scores {
                    *score = (*score - best).exp();
                    total += *score;
                }
                let out = &mut context[head * arch.v_head_dim..(head + 1) * arch.v_head_dim];
                for (t, score) in scores.into_iter().enumerate() {
                    let probability = score / total;
                    let off = (t * arch.n_heads + head) * arch.v_head_dim;
                    for (dst, value) in out
                        .iter_mut()
                        .zip(&cache.values[off..off + arch.v_head_dim])
                    {
                        *dst += probability * value;
                    }
                }
            }
            let layer_context = if tracing { context.clone() } else { Vec::new() };

            let attn_out = weights.matvec(&format!("{attn}.o_proj.weight"), &context)?;
            if attn_out.len() != arch.hidden {
                return Err(Error::Gravity(format!(
                    "deepseek o_proj output {} != hidden {}",
                    attn_out.len(),
                    arch.hidden
                )));
            }
            let layer_attn_out = if tracing {
                attn_out.clone()
            } else {
                Vec::new()
            };
            for (value, update) in x.iter_mut().zip(attn_out) {
                *value += update;
            }
            let layer_after_attention = if tracing { x.clone() } else { Vec::new() };

            let h2 = rmsnorm(
                &x,
                &weights.dense(&format!("{p}.post_attention_layernorm.weight"))?,
                arch.rms_norm_eps,
            )?;
            let layer_ffn_norm = if tracing { h2.clone() } else { Vec::new() };
            let mut router_capture = tracing.then(|| (Vec::new(), Vec::new()));
            let ffn = match arch.layer_kind(layer)? {
                DeepSeekLayerKind::Dense => dense_mlp(weights, &format!("{p}.mlp"), &h2)?,
                DeepSeekLayerKind::SparseMoe => {
                    let (out, _, _) = routed_mlp_with_trace(
                        weights,
                        arch,
                        &format!("{p}.mlp"),
                        &h2,
                        router_capture.as_mut(),
                    )?;
                    out
                }
            };
            if ffn.len() != arch.hidden {
                return Err(Error::Gravity(format!(
                    "deepseek MLP output {} != hidden {}",
                    ffn.len(),
                    arch.hidden
                )));
            }
            let layer_ffn_out = if tracing { ffn.clone() } else { Vec::new() };
            for (value, update) in x.iter_mut().zip(ffn) {
                *value += update;
            }
            if let Some(ref mut trace) = token_trace {
                let (router_logits, routes) = router_capture.take().unwrap_or_default();
                trace.layers.push(DeepSeekLayerStageTrace {
                    layer,
                    input: layer_input,
                    attn_norm: layer_attn_norm,
                    q: layer_q,
                    kv_a: layer_kv_a,
                    c_kv: layer_c_kv,
                    k_pe: layer_k_pe,
                    context: layer_context,
                    attn_out: layer_attn_out,
                    after_attention: layer_after_attention,
                    ffn_norm: layer_ffn_norm,
                    router_logits,
                    routes,
                    ffn_out: layer_ffn_out,
                    output: x.clone(),
                });
            }
        }

        let final_hidden = rmsnorm(&x, &weights.dense("model.norm.weight")?, arch.rms_norm_eps)?;
        logits = weights.matvec("lm_head.weight", &final_hidden)?;
        if let Some(mut trace) = token_trace.take() {
            trace.final_norm = final_hidden;
            trace.logits = logits.clone();
            if let Some(out) = trace_out.as_deref_mut() {
                out.push(trace);
            }
        }
    }
    Ok(logits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::io::Write;

    fn header() -> serde_json::Value {
        serde_json::json!({
            "architecture": {
                "model_type": "deepseek2",
                "num_hidden_layers": 4,
                "hidden_size": 2048,
                "num_attention_heads": 16,
                "num_key_value_heads": 1,
                "q_lora_rank": 1536,
                "kv_lora_rank": 512,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "v_head_dim": 128,
                "intermediate_size": 10944,
                "moe_intermediate_size": 1408,
                "n_routed_experts": 64,
                "n_shared_experts": 2,
                "num_experts_per_tok": 6,
                "first_k_dense_replace": 1,
                "n_group": 8,
                "topk_group": 3,
                "topk_method": "group_limited_greedy",
                "scoring_func": "softmax",
                "norm_topk_prob": false,
                "routed_scaling_factor": 1.0,
                "vocab_size": 102400,
                "rope_theta": 10000.0,
                "rms_norm_eps": 0.000001,
                "max_position_embeddings": 16384
            }
        })
    }

    #[test]
    fn parses_mla_and_moe_geometry_without_defaults() {
        let arch = DeepSeekGravityArch::from_header(&header()).expect("valid DeepSeek header");
        assert_eq!(
            arch.attention_kind(),
            DeepSeekAttentionKind::MultiHeadLatent
        );
        assert_eq!(arch.layer_kind(0).unwrap(), DeepSeekLayerKind::Dense);
        assert_eq!(arch.layer_kind(1).unwrap(), DeepSeekLayerKind::SparseMoe);
        assert_eq!(arch.compact_kv_bytes_per_token(4), (512 + 64) * 4);
    }

    #[test]
    fn wrong_architecture_is_refused() {
        let mut value = header();
        value["architecture"]["model_type"] = serde_json::json!("llama");
        let error = DeepSeekGravityArch::from_header(&value).unwrap_err();
        assert!(error.to_string().contains("expected \"deepseek2\""));
    }

    #[test]
    fn missing_routing_field_is_refused_instead_of_guessed() {
        let mut value = header();
        value["architecture"]
            .as_object_mut()
            .unwrap()
            .remove("n_group");
        let error = DeepSeekGravityArch::from_header(&value).unwrap_err();
        assert!(error.to_string().contains("n_group"));
    }

    #[test]
    fn non_softmax_router_is_refused_by_the_v2_runtime() {
        let mut value = header();
        value["architecture"]["scoring_func"] = serde_json::json!("sigmoid");
        let error = DeepSeekGravityArch::from_header(&value).unwrap_err();
        assert!(error.to_string().contains("scoring_func=softmax"));
    }

    #[test]
    fn group_limited_softmax_routes_only_inside_selected_groups() {
        let arch = DeepSeekGravityArch::from_header(&header()).unwrap();
        let mut logits = vec![-10.0f32; arch.n_routed_experts];
        logits[8] = 9.0;
        logits[16] = 8.0;
        logits[24] = 7.0;
        logits[0] = 6.0;
        let routes = super::deepseek_routes(&arch, &mut logits).unwrap();
        assert_eq!(routes.len(), arch.num_experts_per_tok);
        let groups: std::collections::HashSet<usize> = routes
            .iter()
            .map(|(expert, _)| expert / (arch.n_routed_experts / arch.n_group))
            .collect();
        assert!(groups.len() <= arch.topk_group);
    }

    #[test]
    fn route_tie_epsilon_prefers_lower_expert_id_without_changing_clear_winner() {
        let values = [0.5005f32, 0.5, 0.1, 0.05];
        assert_eq!(super::topk_desc_with_epsilon(&values, 2, 0.0), vec![0, 1]);
        assert_eq!(super::topk_desc_with_epsilon(&values, 2, 0.001), vec![0, 1]);

        let reversed = [0.5f32, 0.5005, 0.1, 0.05];
        assert_eq!(super::topk_desc_with_epsilon(&reversed, 2, 0.0), vec![1, 0]);
        assert_eq!(
            super::topk_desc_with_epsilon(&reversed, 2, 0.001),
            vec![0, 1]
        );

        let clear = [0.9f32, 0.5, 0.1, 0.05];
        assert_eq!(super::topk_desc_with_epsilon(&clear, 2, 0.001), vec![0, 1]);
    }

    #[test]
    fn canonical_tensor_map_distinguishes_dense_and_sparse_layers() {
        let arch = DeepSeekGravityArch::from_header(&header()).unwrap();
        let dense = arch.required_tensor_names(0).unwrap();
        assert!(dense.iter().any(|n| n.ends_with("mlp.gate_proj.weight")));
        assert!(!dense.iter().any(|n| n.contains("experts.0")));
        let sparse = arch.required_tensor_names(1).unwrap();
        assert!(sparse.iter().any(|n| n.ends_with("mlp.gate.weight")));
        assert!(sparse.iter().any(|n| n.contains("experts.63.down_proj")));
        assert!(sparse
            .iter()
            .any(|n| n.ends_with("shared_experts.down_proj.weight")));
    }

    #[test]
    fn null_query_rank_selects_the_direct_query_tensor_abi() {
        let mut value = header();
        value["architecture"]["q_lora_rank"] = serde_json::Value::Null;
        let arch = DeepSeekGravityArch::from_header(&value).unwrap();
        assert_eq!(arch.q_lora_rank, 0);
        let names = arch.required_tensor_names(0).unwrap();
        assert!(names.iter().any(|name| name.ends_with("q_proj.weight")));
        assert!(!names.iter().any(|name| name.ends_with("q_a_proj.weight")));
    }

    #[test]
    fn tensor_presence_check_reports_missing_without_claiming_runtime() {
        let arch = DeepSeekGravityArch::from_header(&header()).unwrap();
        let mut present = HashSet::new();
        present.insert("model.embed_tokens.weight".to_string());
        let error = arch.validate_tensor_names(&present, 1).unwrap_err();
        assert!(error.to_string().contains("missing canonical tensors"));
    }

    #[test]
    fn route_scale_is_applied_once_when_topk_is_not_normalized() {
        let mut value = header();
        value["architecture"]["norm_topk_prob"] = serde_json::json!(false);
        value["architecture"]["routed_scaling_factor"] = serde_json::json!(1.5);
        let arch = DeepSeekGravityArch::from_header(&value).unwrap();
        let mut logits = vec![0.0f32; arch.n_routed_experts];
        let routes = super::deepseek_routes(&arch, &mut logits).unwrap();
        assert_eq!(routes.len(), arch.num_experts_per_tok);
        for (_, weight) in routes {
            assert!((weight - 1.5 / arch.n_routed_experts as f32).abs() < 1e-6);
        }
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn matrix(rows: usize, cols: usize, seed: f32) -> Vec<f32> {
        (0..rows * cols)
            .map(|index| {
                let row = index / cols;
                let col = index % cols;
                // Keep the fixture deterministic and non-degenerate while
                // making it easy to inspect a failed logit manually.
                (seed + row as f32 * 0.07 + col as f32 * 0.013).sin() * 0.2
            })
            .collect()
    }

    fn write_tiny_deepseek() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("fixture tempdir");
        let path = dir.path().join("tiny-deepseek.gravity");
        let mut tensors: Vec<(String, Vec<u64>, Vec<u8>)> = Vec::new();
        let mut add = |name: String, shape: Vec<u64>, values: Vec<f32>| {
            tensors.push((name, shape, f32_bytes(&values)));
        };
        let hidden = 4usize;
        let vocab = 5usize;
        let qk = 3usize;
        let kv = 4usize;
        let vdim = 2usize;
        let dense_mid = 4usize;
        let moe_mid = 3usize;

        add(
            "model.embed_tokens.weight".into(),
            vec![vocab as u64, hidden as u64],
            matrix(vocab, hidden, 0.11),
        );
        add(
            "model.norm.weight".into(),
            vec![hidden as u64],
            vec![1.0; hidden],
        );
        add(
            "lm_head.weight".into(),
            vec![vocab as u64, hidden as u64],
            matrix(vocab, hidden, 0.73),
        );

        for layer in 0..2usize {
            let p = format!("model.layers.{layer}");
            let a = format!("{p}.self_attn");
            add(
                format!("{p}.input_layernorm.weight"),
                vec![hidden as u64],
                vec![1.0; hidden],
            );
            add(
                format!("{p}.post_attention_layernorm.weight"),
                vec![hidden as u64],
                vec![1.0; hidden],
            );
            add(
                format!("{a}.q_proj.weight"),
                vec![qk as u64, hidden as u64],
                matrix(qk, hidden, 0.19 + layer as f32),
            );
            add(
                format!("{a}.kv_a_proj_with_mqa.weight"),
                vec![kv as u64, hidden as u64],
                matrix(kv, hidden, 0.23 + layer as f32),
            );
            add(
                format!("{a}.kv_a_layernorm.weight"),
                vec![2],
                vec![1.0, 1.0],
            );
            add(
                format!("{a}.kv_b_proj.weight"),
                vec![3, 2],
                matrix(3, 2, 0.29 + layer as f32),
            );
            add(
                format!("{a}.o_proj.weight"),
                vec![hidden as u64, vdim as u64],
                matrix(hidden, vdim, 0.31 + layer as f32),
            );

            if layer == 0 {
                add(
                    format!("{p}.mlp.gate_proj.weight"),
                    vec![dense_mid as u64, hidden as u64],
                    matrix(dense_mid, hidden, 0.37),
                );
                add(
                    format!("{p}.mlp.up_proj.weight"),
                    vec![dense_mid as u64, hidden as u64],
                    matrix(dense_mid, hidden, 0.41),
                );
                add(
                    format!("{p}.mlp.down_proj.weight"),
                    vec![hidden as u64, dense_mid as u64],
                    matrix(hidden, dense_mid, 0.43),
                );
            } else {
                add(
                    format!("{p}.mlp.gate.weight"),
                    vec![2, hidden as u64],
                    matrix(2, hidden, 0.47),
                );
                for expert in 0..2usize {
                    let ep = format!("{p}.mlp.experts.{expert}");
                    add(
                        format!("{ep}.gate_proj.weight"),
                        vec![moe_mid as u64, hidden as u64],
                        matrix(moe_mid, hidden, 0.53 + expert as f32),
                    );
                    add(
                        format!("{ep}.up_proj.weight"),
                        vec![moe_mid as u64, hidden as u64],
                        matrix(moe_mid, hidden, 0.59 + expert as f32),
                    );
                    add(
                        format!("{ep}.down_proj.weight"),
                        vec![hidden as u64, moe_mid as u64],
                        matrix(hidden, moe_mid, 0.61 + expert as f32),
                    );
                }
                let sp = format!("{p}.mlp.shared_experts");
                add(
                    format!("{sp}.gate_proj.weight"),
                    vec![moe_mid as u64, hidden as u64],
                    matrix(moe_mid, hidden, 0.67),
                );
                add(
                    format!("{sp}.up_proj.weight"),
                    vec![moe_mid as u64, hidden as u64],
                    matrix(moe_mid, hidden, 0.71),
                );
                add(
                    format!("{sp}.down_proj.weight"),
                    vec![hidden as u64, moe_mid as u64],
                    matrix(hidden, moe_mid, 0.79),
                );
            }
        }

        let mut offset = 0u64;
        let mut descriptors = Vec::with_capacity(tensors.len());
        let mut body = Vec::new();
        for (name, shape, payload) in &tensors {
            let digest = format!("{:x}", Sha256::digest(payload));
            descriptors.push(serde_json::json!({
                "name": name,
                "codec": "native.f32",
                "offset": offset,
                "bytes": payload.len() as u64,
                "sha256": digest,
                "shape": shape,
                "elements": payload.len() as u64 / 4,
            }));
            offset += payload.len() as u64;
            body.extend_from_slice(payload);
        }
        let header = serde_json::json!({
            "schema": "hawking.gravity.shard_header.v1",
            "format_version": 1,
            "model": {"family": "deepseek", "fixture": true},
            "architecture": {
                "model_type": "deepseek2",
                "num_hidden_layers": 2,
                "hidden_size": hidden,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "q_lora_rank": null,
                "kv_lora_rank": 2,
                "qk_nope_head_dim": 1,
                "qk_rope_head_dim": 2,
                "v_head_dim": vdim,
                "intermediate_size": dense_mid,
                "moe_intermediate_size": moe_mid,
                "n_routed_experts": 2,
                "n_shared_experts": 1,
                "num_experts_per_tok": 1,
                "first_k_dense_replace": 1,
                "n_group": 1,
                "topk_group": 1,
                "topk_method": "greedy",
                "scoring_func": "softmax",
                "norm_topk_prob": true,
                "routed_scaling_factor": 1.0,
                "vocab_size": vocab,
                "rope_theta": 10000.0,
                "rms_norm_eps": 0.00001,
                "max_position_embeddings": 8,
            },
            "tokenizer": {"fixture": true},
            "compression": {"codec": "native.f32"},
            "integrity": {
                "body_sha256": format!("{:x}", Sha256::digest(&body)),
                "tensor_count": tensors.len(),
            },
            "tensors": descriptors,
        });
        let encoded = serde_json::to_vec(&header).expect("fixture header json");
        let mut file = std::fs::File::create(&path).expect("fixture create");
        file.write_all(b"GRAVITY\0").unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        file.write_all(&(encoded.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&encoded).unwrap();
        file.write_all(&body).unwrap();
        (dir, path)
    }

    #[test]
    fn tiny_gravity_artifact_matches_incremental_stateful_forward() {
        let (_dir, path) = write_tiny_deepseek();
        let model = GravityDeepSeek::open(&path, true).expect("open tiny DeepSeek artifact");
        let full = model.forward(&[1, 2]).expect("full prefill");
        let first = model.forward_at(&[1], 0).expect("first incremental token");
        let incremental = model.forward_at(&[2], 1).expect("second incremental token");
        assert_eq!(full.len(), 5);
        assert_eq!(first.len(), 5);
        assert_eq!(incremental.len(), full.len());
        for (got, want) in incremental.iter().zip(full) {
            assert!(
                (got - want).abs() < 1e-6,
                "incremental {got} != full {want}"
            );
        }
    }

    #[test]
    fn tiny_gravity_stage_trace_covers_mla_moe_boundaries() {
        let (_dir, path) = write_tiny_deepseek();
        let model = GravityDeepSeek::open(&path, true).expect("open tiny DeepSeek artifact");
        let (logits, trace) = model
            .forward_with_stage_trace(&[1])
            .expect("stage-traced tiny DeepSeek forward");
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].token, 1);
        assert_eq!(trace[0].position, 0);
        assert_eq!(trace[0].layers.len(), 2);
        assert_eq!(trace[0].logits, logits);
        assert_eq!(trace[0].layers[0].q.len(), 3);
        assert_eq!(trace[0].layers[0].c_kv.len(), 2);
        assert_eq!(trace[0].layers[0].k_pe.len(), 2);
        assert!(trace[0].layers[0].routes.is_empty());
        assert_eq!(trace[0].layers[1].routes.len(), 1);
        assert_eq!(trace[0].layers[1].router_logits.len(), 2);
        assert_eq!(trace[0].layers[1].output.len(), 4);
        assert!(trace[0]
            .layers
            .iter()
            .flat_map(|layer| layer.output.iter())
            .all(|value| value.is_finite()));
    }
}

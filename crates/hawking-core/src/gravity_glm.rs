//! GLM-5.2 (`glm_moe_dsa`) forward pass executed directly out of `.gravity`.
//!
//! The production counterpart of `tools/condense/glm52_reference.py`, and
//! graded against it. Everything here that looks like an odd choice is a
//! choice the reference makes, and the two agree or this is wrong:
//!
//!   - RoPE is *interleaved*, and the rotated halves are **concatenated**,
//!     not scattered back to their even and odd source positions.
//!   - The router selects on sigmoid scores **plus** a correction bias, then
//!     weights with the **uncorrected** scores (`noaux_tc`).
//!   - Expert groups are chosen by the sum of each group's best two
//!     corrected scores, and only experts inside the chosen groups compete.
//!   - An IndexShare layer **reuses** the previous full layer's DSA
//!     selection. Recomputing it there is a different model.
//!   - Indexer scores are ReLU'd before the per-head weighted sum.
//!
//! Ties are broken the way `np.argsort(kind="stable")` breaks them -- lower
//! index first -- because a differently-broken tie silently selects a
//! different expert or a different key, and nothing downstream would say so.

use std::path::Path;

use crate::gravity::GravityWeights;
use crate::{Error, Result};

fn cfg_u64(v: &serde_json::Value, key: &str) -> Result<u64> {
    v.get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| Error::Gravity(format!("architecture.{key} missing or not an integer")))
}

fn cfg_f64(v: &serde_json::Value, key: &str) -> Result<f64> {
    v.get(key)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| Error::Gravity(format!("architecture.{key} missing or not a number")))
}

fn cfg_strings(v: &serde_json::Value, key: &str) -> Result<Vec<String>> {
    v.get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::Gravity(format!("architecture.{key} missing or not an array")))?
        .iter()
        .map(|e| {
            e.as_str()
                .map(str::to_string)
                .ok_or_else(|| Error::Gravity(format!("architecture.{key} has a non-string entry")))
        })
        .collect()
}

/// The GLM configuration the forward pass needs. Every field is read from
/// the artifact header; none is defaulted, because a guessed `index_topk`
/// or `topk_group` produces a model that runs and is wrong.
#[derive(Debug, Clone)]
pub struct GlmArch {
    pub n_layers: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub n_routed_experts: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// Per layer: `"full"` computes a DSA index, `"shared"` reuses the
    /// previous full layer's.
    pub indexer_types: Vec<String>,
    /// Per layer: `"dense"` or `"sparse"`.
    pub mlp_layer_types: Vec<String>,
}

impl GlmArch {
    pub fn from_header(extra: &serde_json::Value) -> Result<GlmArch> {
        let a = extra
            .get("architecture")
            .ok_or_else(|| Error::Gravity("shard header has no `architecture`".into()))?;
        let model_type = a.get("model_type").and_then(serde_json::Value::as_str);
        if model_type != Some("glm_moe_dsa") {
            return Err(Error::Gravity(format!(
                "gravity_glm: architecture.model_type is {model_type:?}, expected \"glm_moe_dsa\""
            )));
        }
        let rope_theta = a
            .get("rope_parameters")
            .and_then(|r| r.get("rope_theta"))
            .and_then(serde_json::Value::as_f64)
            .or_else(|| a.get("rope_theta").and_then(serde_json::Value::as_f64))
            .ok_or_else(|| Error::Gravity("architecture rope_theta missing".into()))?;

        let arch = GlmArch {
            n_layers: cfg_u64(a, "num_hidden_layers")? as usize,
            hidden: cfg_u64(a, "hidden_size")? as usize,
            n_heads: cfg_u64(a, "num_attention_heads")? as usize,
            q_lora_rank: cfg_u64(a, "q_lora_rank")? as usize,
            kv_lora_rank: cfg_u64(a, "kv_lora_rank")? as usize,
            qk_nope_head_dim: cfg_u64(a, "qk_nope_head_dim")? as usize,
            qk_rope_head_dim: cfg_u64(a, "qk_rope_head_dim")? as usize,
            v_head_dim: cfg_u64(a, "v_head_dim")? as usize,
            index_n_heads: cfg_u64(a, "index_n_heads")? as usize,
            index_head_dim: cfg_u64(a, "index_head_dim")? as usize,
            index_topk: cfg_u64(a, "index_topk")? as usize,
            n_routed_experts: cfg_u64(a, "n_routed_experts")? as usize,
            n_group: cfg_u64(a, "n_group")? as usize,
            topk_group: cfg_u64(a, "topk_group")? as usize,
            num_experts_per_tok: cfg_u64(a, "num_experts_per_tok")? as usize,
            norm_topk_prob: a
                .get("norm_topk_prob")
                .and_then(serde_json::Value::as_bool)
                .ok_or_else(|| Error::Gravity("architecture.norm_topk_prob missing".into()))?,
            routed_scaling_factor: cfg_f64(a, "routed_scaling_factor")? as f32,
            vocab_size: cfg_u64(a, "vocab_size")? as usize,
            rms_norm_eps: cfg_f64(a, "rms_norm_eps")? as f32,
            rope_theta: rope_theta as f32,
            indexer_types: cfg_strings(a, "indexer_types")?,
            mlp_layer_types: cfg_strings(a, "mlp_layer_types")?,
        };
        if arch.indexer_types.len() != arch.n_layers || arch.mlp_layer_types.len() != arch.n_layers {
            return Err(Error::Gravity(format!(
                "layer schedules are {} / {} long but the model has {} layers",
                arch.indexer_types.len(),
                arch.mlp_layer_types.len(),
                arch.n_layers
            )));
        }
        if arch.indexer_types.first().map(String::as_str) == Some("shared") {
            return Err(Error::Gravity(
                "layer 0 is an IndexShare layer with no previous index to share".into(),
            ));
        }
        if arch.n_group == 0 || arch.n_routed_experts % arch.n_group != 0 {
            return Err(Error::Gravity(format!(
                "{} routed experts do not divide into {} groups",
                arch.n_routed_experts, arch.n_group
            )));
        }
        Ok(arch)
    }

    pub fn qk_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
}

fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(weight).map(|(v, w)| v * inv * w).collect()
}

/// Affine LayerNorm, used only by the DSA indexer's key projection.
fn layernorm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    (0..x.len())
        .map(|i| (x[i] - mean) * inv * weight[i] + bias[i])
        .collect()
}

fn silu_mul(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
        .collect()
}

/// Descending top-k with `np.argsort(kind="stable")` tie-breaking: equal
/// values keep ascending index order. NaN never wins, matching the way the
/// reference's `-inf` masking removes a candidate entirely.
fn topk_desc(values: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

/// GLM's interleaved RoPE for one `rotary_dim`-wide vector.
///
/// The trap: the rotated first and second components are **concatenated**,
/// so output element `i` is not input element `i` rotated. `cos`/`sin` are
/// `rotary_dim/2` long.
fn rope_interleaved(v: &[f32], cos: &[f32], sin: &[f32]) -> Vec<f32> {
    let half = v.len() / 2;
    let mut out = vec![0f32; v.len()];
    for i in 0..half {
        let first = v[2 * i];
        let second = v[2 * i + 1];
        out[i] = first * cos[i] - second * sin[i];
        out[half + i] = second * cos[i] + first * sin[i];
    }
    out
}

/// Per-layer attention state. GLM caches the assembled `qk_dim` keys and
/// `v_head_dim` values per head, plus the DSA indexer's own key stream.
#[derive(Default)]
struct LayerCache {
    /// `[pos][head][qk_dim]` flattened.
    keys: Vec<f32>,
    /// `[pos][head][v_head_dim]` flattened.
    values: Vec<f32>,
    /// `[pos][index_head_dim]` flattened. Only full-indexer layers fill it.
    index_keys: Vec<f32>,
}

/// A `.gravity` shard loaded as an executable GLM-5.2 model.
pub struct GravityGlm {
    pub arch: GlmArch,
    weights: GravityWeights,
}

/// Everything one token's forward produced that a caller may want to check
/// without re-running it.
#[derive(Debug, Clone, Default)]
pub struct GlmTrace {
    /// DSA keys selected by the last layer that computed an index.
    pub final_topk: Vec<usize>,
    /// Per sparse layer, the experts the router chose for this token.
    pub expert_choices: Vec<Vec<usize>>,
}

impl GravityGlm {
    /// Open a single-shard `.gravity` artifact — the semantic fixture, which
    /// is small enough to decode eagerly.
    pub fn open(path: &Path, verify_hash: bool) -> Result<GravityGlm> {
        let weights = GravityWeights::open(path, verify_hash)?;
        let arch = GlmArch::from_header(&weights.header)?;
        Ok(GravityGlm { arch, weights })
    }

    /// Open a multi-shard flagship model — every `model-*.gravity` file
    /// under `dir`, indexed but not decoded. Only 8 of 256 experts activate
    /// per layer, so eager decode of every shard would waste ~32x the
    /// necessary work and, at this scale, exceed physical memory; see
    /// [`crate::gravity::GravityWeights::open_dir`].
    pub fn open_dir(dir: &Path, verify_hash: bool) -> Result<GravityGlm> {
        let weights = GravityWeights::open_dir(dir, verify_hash)?;
        let arch = GlmArch::from_header(&weights.header)?;
        Ok(GravityGlm { arch, weights })
    }

    /// RoPE cos/sin for one position: `qk_rope_head_dim/2` of each. The
    /// reference builds a `rotary_dim`-wide table by concatenating the
    /// frequencies with themselves and then takes the first half, which is
    /// exactly these values.
    fn rope_cos_sin(&self, pos: usize) -> (Vec<f32>, Vec<f32>) {
        let rot = self.arch.qk_rope_head_dim;
        let half = rot / 2;
        let mut cos = vec![0f32; half];
        let mut sin = vec![0f32; half];
        for i in 0..half {
            // f32 throughout: the reference forces float32 here, and matching
            // it in f64 would drift from the thing being reproduced.
            let inv_freq = 1.0f32 / self.arch.rope_theta.powf(2.0 * i as f32 / rot as f32);
            let theta = pos as f32 * inv_freq;
            cos[i] = theta.cos();
            sin[i] = theta.sin();
        }
        (cos, sin)
    }

    /// The DSA indexer: which cached keys this token is allowed to attend to.
    #[allow(clippy::too_many_arguments)]
    fn indexer_topk(
        &self,
        prefix: &str,
        hidden: &[f32],
        q_resid: &[f32],
        cache: &mut LayerCache,
        pos: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<Vec<usize>> {
        let a = &self.arch;
        let (ih, idim, rot) = (a.index_n_heads, a.index_head_dim, a.qk_rope_head_dim);
        let idx = format!("{prefix}.indexer");

        let q = self.weights.matvec(&format!("{idx}.wq_b.weight"), q_resid)?;
        if q.len() != ih * idim {
            return Err(Error::Gravity(format!(
                "indexer wq_b produced {} values, expected {ih} heads * {idim}",
                q.len()
            )));
        }
        let k_raw = self.weights.matvec(&format!("{idx}.wk.weight"), hidden)?;
        let k = layernorm(
            &k_raw,
            &self.weights.dense(&format!("{idx}.k_norm.weight"))?,
            &self.weights.dense(&format!("{idx}.k_norm.bias"))?,
            1e-6,
        );

        // Rotate the leading `rot` dims, keep the tail unrotated.
        let mut k_full = rope_interleaved(&k[..rot], cos, sin);
        k_full.extend_from_slice(&k[rot..]);
        cache.index_keys.extend_from_slice(&k_full);
        let n_keys = cache.index_keys.len() / idim;

        let mut q_full = vec![0f32; ih * idim];
        for h in 0..ih {
            let src = &q[h * idim..(h + 1) * idim];
            let rotated = rope_interleaved(&src[..rot], cos, sin);
            q_full[h * idim..h * idim + rot].copy_from_slice(&rotated);
            q_full[h * idim + rot..(h + 1) * idim].copy_from_slice(&src[rot..]);
        }

        let head_scale = (ih as f32).powf(-0.5);
        let mut head_weights = self
            .weights
            .matvec(&format!("{idx}.weights_proj.weight"), hidden)?;
        for w in head_weights.iter_mut() {
            *w *= head_scale;
        }

        let dim_scale = (idim as f32).powf(-0.5);
        let mut index_scores = vec![0f32; n_keys];
        for (t, score) in index_scores.iter_mut().enumerate() {
            let key = &cache.index_keys[t * idim..(t + 1) * idim];
            let mut acc = 0f32;
            for h in 0..ih {
                let qh = &q_full[h * idim..(h + 1) * idim];
                let dot: f32 = qh.iter().zip(key).map(|(a, b)| a * b).sum();
                // ReLU before the weighted sum: a head that dislikes a key
                // contributes nothing rather than voting against it.
                acc += head_weights[h] * (dot * dim_scale).max(0.0);
            }
            *score = acc;
        }
        // Causal: this token cannot index a key from the future. Positions
        // beyond `pos` only exist during batched prefill.
        for (t, score) in index_scores.iter_mut().enumerate() {
            if t > pos {
                *score = f32::NEG_INFINITY;
            }
        }
        Ok(topk_desc(&index_scores, a.index_topk.min(n_keys)))
    }

    /// GLM's `noaux_tc` router: select on corrected scores inside the
    /// winning expert groups, weight with the uncorrected ones.
    fn router(&self, prefix: &str, hidden: &[f32]) -> Result<(Vec<usize>, Vec<f32>)> {
        let a = &self.arch;
        let logits = self
            .weights
            .matvec(&format!("{prefix}.gate.weight"), hidden)?;
        let scores: Vec<f32> = logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
        let bias = self
            .weights
            .dense(&format!("{prefix}.gate.e_score_correction_bias"))?;
        if bias.len() != a.n_routed_experts || scores.len() != a.n_routed_experts {
            return Err(Error::Gravity(format!(
                "router shape mismatch: {} logits, {} bias values, {} experts",
                scores.len(),
                bias.len(),
                a.n_routed_experts
            )));
        }
        let corrected: Vec<f32> = scores.iter().zip(bias).map(|(s, b)| s + b).collect();

        // A group's strength is the sum of its best two corrected scores.
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
        let chosen: Vec<usize> = topk_desc(&group_scores, a.topk_group);

        // Only experts inside a chosen group compete.
        let mut choice = vec![f32::NEG_INFINITY; a.n_routed_experts];
        for &g in &chosen {
            for e in g * per_group..(g + 1) * per_group {
                choice[e] = corrected[e];
            }
        }
        let indices = topk_desc(&choice, a.num_experts_per_tok);
        let mut weights: Vec<f32> = indices.iter().map(|&i| scores[i]).collect();
        if a.norm_topk_prob {
            let total: f32 = weights.iter().sum::<f32>() + 1e-20;
            for w in weights.iter_mut() {
                *w /= total;
            }
        }
        for w in weights.iter_mut() {
            *w *= a.routed_scaling_factor;
        }
        Ok((indices, weights))
    }

    fn dense_mlp(&self, prefix: &str, x: &[f32]) -> Result<Vec<f32>> {
        let gate = self.weights.matvec(&format!("{prefix}.gate_proj.weight"), x)?;
        let up = self.weights.matvec(&format!("{prefix}.up_proj.weight"), x)?;
        self.weights
            .matvec(&format!("{prefix}.down_proj.weight"), &silu_mul(&gate, &up))
    }

    fn routed_moe(&self, prefix: &str, x: &[f32]) -> Result<(Vec<f32>, Vec<usize>)> {
        let (indices, weights) = self.router(prefix, x)?;
        // The reference accumulates in ascending expert order, so match it:
        // float addition is not associative and the artifact is graded to
        // 1e-4, not to "close enough".
        let mut order: Vec<usize> = (0..indices.len()).collect();
        order.sort_by_key(|&s| indices[s]);

        let mut routed = vec![0f32; x.len()];
        for &slot in &order {
            let out = self.dense_mlp(&format!("{prefix}.experts.{}", indices[slot]), x)?;
            for (r, o) in routed.iter_mut().zip(&out) {
                *r += o * weights[slot];
            }
        }
        let shared = self.dense_mlp(&format!("{prefix}.shared_experts"), x)?;
        for (r, s) in routed.iter_mut().zip(&shared) {
            *r += s;
        }
        Ok((routed, indices))
    }

    /// Run `tokens` from an empty cache and return the logits after the last
    /// one, plus a trace of what the routers and the indexer chose.
    pub fn forward(&self, tokens: &[u32]) -> Result<(Vec<f32>, GlmTrace)> {
        if tokens.is_empty() {
            return Err(Error::Gravity("forward: no tokens".into()));
        }
        let a = &self.arch;
        let qk_dim = a.qk_dim();
        let mut caches: Vec<LayerCache> = (0..a.n_layers).map(|_| LayerCache::default()).collect();
        let mut logits = Vec::new();
        let mut trace = GlmTrace::default();

        for (pos, &token) in tokens.iter().enumerate() {
            if token as usize >= a.vocab_size {
                return Err(Error::Gravity(format!(
                    "token {token} out of range for vocab_size {}",
                    a.vocab_size
                )));
            }
            let mut x = self
                .weights
                .row("model.embed_tokens.weight", token as usize, a.hidden)?;
            let (cos, sin) = self.rope_cos_sin(pos);
            let mut shared_topk: Option<Vec<usize>> = None;
            trace.expert_choices.clear();

            for layer in 0..a.n_layers {
                let p = format!("model.layers.{layer}");
                let attn_p = format!("{p}.self_attn");
                let h = rmsnorm(
                    &x,
                    &self.weights.dense(&format!("{p}.input_layernorm.weight"))?,
                    a.rms_norm_eps,
                );

                // Queries through the low-rank path.
                let q_a = self.weights.matvec(&format!("{attn_p}.q_a_proj.weight"), &h)?;
                let q_resid = rmsnorm(
                    &q_a,
                    &self.weights
                        .dense(&format!("{attn_p}.q_a_layernorm.weight"))?,
                    a.rms_norm_eps,
                );
                let q = self
                    .weights
                    .matvec(&format!("{attn_p}.q_b_proj.weight"), &q_resid)?;

                // Keys and values through the compressed MLA latent. The
                // rope part is shared across heads (MQA on that slice).
                let compressed = self
                    .weights
                    .matvec(&format!("{attn_p}.kv_a_proj_with_mqa.weight"), &h)?;
                let k_latent = rmsnorm(
                    &compressed[..a.kv_lora_rank],
                    &self.weights
                        .dense(&format!("{attn_p}.kv_a_layernorm.weight"))?,
                    a.rms_norm_eps,
                );
                let k_rot = rope_interleaved(&compressed[a.kv_lora_rank..], &cos, &sin);
                let kv = self
                    .weights
                    .matvec(&format!("{attn_p}.kv_b_proj.weight"), &k_latent)?;

                let per_head_kv = a.qk_nope_head_dim + a.v_head_dim;
                let cache = &mut caches[layer];
                for head in 0..a.n_heads {
                    let src = &kv[head * per_head_kv..(head + 1) * per_head_kv];
                    cache
                        .keys
                        .extend_from_slice(&src[..a.qk_nope_head_dim]);
                    cache.keys.extend_from_slice(&k_rot);
                    cache
                        .values
                        .extend_from_slice(&src[a.qk_nope_head_dim..]);
                }

                // Assemble this token's queries: unrotated nope half, then
                // the interleaved-rotated rope half.
                let mut queries = vec![0f32; a.n_heads * qk_dim];
                for head in 0..a.n_heads {
                    let src = &q[head * qk_dim..(head + 1) * qk_dim];
                    let dst = &mut queries[head * qk_dim..(head + 1) * qk_dim];
                    dst[..a.qk_nope_head_dim].copy_from_slice(&src[..a.qk_nope_head_dim]);
                    dst[a.qk_nope_head_dim..]
                        .copy_from_slice(&rope_interleaved(&src[a.qk_nope_head_dim..], &cos, &sin));
                }

                let topk = match a.indexer_types[layer].as_str() {
                    "full" => {
                        let t = self.indexer_topk(&attn_p, &h, &q_resid, cache, pos, &cos, &sin)?;
                        shared_topk = Some(t.clone());
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

                // Attend only to selected keys at or before this position.
                let n_keys = cache.keys.len() / (a.n_heads * qk_dim);
                let mut allow = vec![false; n_keys];
                for &t in &topk {
                    if t <= pos && t < n_keys {
                        allow[t] = true;
                    }
                }

                let scale = (qk_dim as f32).powf(-0.5);
                let mut context = vec![0f32; a.n_heads * a.v_head_dim];
                let mut scores = vec![f32::NEG_INFINITY; n_keys];
                for head in 0..a.n_heads {
                    let qh = &queries[head * qk_dim..(head + 1) * qk_dim];
                    let mut best = f32::NEG_INFINITY;
                    for t in 0..n_keys {
                        if !allow[t] {
                            scores[t] = f32::NEG_INFINITY;
                            continue;
                        }
                        let off = (t * a.n_heads + head) * qk_dim;
                        let dot: f32 = qh
                            .iter()
                            .zip(&cache.keys[off..off + qk_dim])
                            .map(|(a, b)| a * b)
                            .sum();
                        scores[t] = dot * scale;
                        best = best.max(scores[t]);
                    }
                    let mut total = 0f32;
                    for s in scores.iter_mut() {
                        *s = if s.is_finite() { (*s - best).exp() } else { 0.0 };
                        total += *s;
                    }
                    let out = &mut context[head * a.v_head_dim..(head + 1) * a.v_head_dim];
                    for (t, &prob) in scores.iter().enumerate() {
                        if prob == 0.0 {
                            continue;
                        }
                        let w = prob / total;
                        let off = (t * a.n_heads + head) * a.v_head_dim;
                        for (o, v) in out
                            .iter_mut()
                            .zip(&cache.values[off..off + a.v_head_dim])
                        {
                            *o += w * v;
                        }
                    }
                }

                let attn_out = self
                    .weights
                    .matvec(&format!("{attn_p}.o_proj.weight"), &context)?;
                for (xv, o) in x.iter_mut().zip(&attn_out) {
                    *xv += o;
                }

                let h2 = rmsnorm(
                    &x,
                    &self.weights
                        .dense(&format!("{p}.post_attention_layernorm.weight"))?,
                    a.rms_norm_eps,
                );
                let mlp_out = match a.mlp_layer_types[layer].as_str() {
                    "dense" => self.dense_mlp(&format!("{p}.mlp"), &h2)?,
                    "sparse" => {
                        let (out, experts) = self.routed_moe(&format!("{p}.mlp"), &h2)?;
                        trace.expert_choices.push(experts);
                        out
                    }
                    other => {
                        return Err(Error::Gravity(format!(
                            "layer {layer}: unknown MLP type {other:?}"
                        )))
                    }
                };
                for (xv, m) in x.iter_mut().zip(&mlp_out) {
                    *xv += m;
                }

                if layer + 1 == a.n_layers {
                    trace.final_topk = topk;
                }
            }

            let final_hidden = rmsnorm(
                &x,
                &self.weights.dense("model.norm.weight")?,
                a.rms_norm_eps,
            );
            logits = self.weights.matvec("lm_head.weight", &final_hidden)?;
        }
        Ok((logits, trace))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference's own selfcheck pins position zero to `[x0,x2,x1,x3]`
    /// for a 4-wide vector -- the concatenated layout, not the scattered
    /// one. If this passes with a scatter implementation, the test is wrong.
    #[test]
    fn interleaved_rope_position_zero_is_the_concatenated_layout() {
        let v = [0f32, 1.0, 2.0, 3.0];
        let got = rope_interleaved(&v, &[1.0, 1.0], &[0.0, 0.0]);
        assert_eq!(got, vec![0.0, 2.0, 1.0, 3.0]);
    }

    #[test]
    fn topk_desc_breaks_ties_toward_the_lower_index() {
        assert_eq!(topk_desc(&[1.0, 3.0, 3.0, 0.0], 2), vec![1, 2]);
        assert_eq!(topk_desc(&[f32::NEG_INFINITY, 0.5], 1), vec![1]);
    }
}

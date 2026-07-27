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

/// What the forward pass needs from wherever weights actually live: CPU
/// decode-on-call (`GravityWeights`) or a GPU-resident lazy cache
/// ([`gpu::GpuWeightCache`]). The orchestration below — MLA, DSA, the
/// router, MoE dispatch — is the part that took real work to get bit-exact
/// against the oracle; it is written once against this trait and both
/// backends run the identical logic, rather than risking two copies
/// drifting apart.
pub trait WeightAccess {
    fn dense(&self, name: &str) -> Result<Vec<f32>>;
    fn matvec(&self, name: &str, x: &[f32]) -> Result<Vec<f32>>;
    fn row(&self, name: &str, index: usize, cols: usize) -> Result<Vec<f32>>;

    /// `calls.len()` independent matvecs, dispatched together where the
    /// backend can. The default just loops `matvec` -- correct for any
    /// backend, but only the GPU one overrides it to batch every call into
    /// one command buffer instead of paying a synchronous round trip per
    /// matvec. That round trip is what a routed MoE layer's experts are
    /// dominated by: each is independent of the others, so nothing about
    /// correctness requires paying for eight of them one at a time.
    fn matvec_batch(&self, calls: &[(&str, &[f32])]) -> Result<Vec<Vec<f32>>> {
        calls.iter().map(|&(name, x)| self.matvec(name, x)).collect()
    }
}

impl WeightAccess for GravityWeights {
    fn dense(&self, name: &str) -> Result<Vec<f32>> {
        GravityWeights::dense(self, name)
    }
    fn matvec(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        GravityWeights::matvec(self, name, x)
    }
    fn row(&self, name: &str, index: usize, cols: usize) -> Result<Vec<f32>> {
        GravityWeights::row(self, name, index, cols)
    }
}

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
pub(crate) fn topk_desc(values: &[f32], k: usize) -> Vec<usize> {
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
pub(crate) fn rope_interleaved(v: &[f32], cos: &[f32], sin: &[f32]) -> Vec<f32> {
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

/// The growing per-layer caches for one generation. Held across calls by
/// whichever caller needs incremental decode ([`gpu::GravityGlmGpu`]); a
/// one-shot caller ([`GravityGlm::forward`]) makes a fresh one and throws it
/// away.
struct GlmSession {
    caches: Vec<LayerCache>,
}

impl GlmSession {
    fn new(arch: &GlmArch) -> GlmSession {
        GlmSession {
            caches: (0..arch.n_layers).map(|_| LayerCache::default()).collect(),
        }
    }

    /// Drop everything back to an empty cache, for a caller starting a new
    /// request on a model it keeps resident between requests.
    fn reset(&mut self) {
        for c in &mut self.caches {
            *c = LayerCache::default();
        }
    }
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
    /// Greedy token from the device lm_head path (token-only readback).
    /// Present when [`GPU_LM_HEAD_ENV`] is on and the head ran on device.
    pub sample_token: Option<u32>,
    /// On-device top-k indices over logits (diagnostics; empty when not computed).
    pub head_topk_idx: Vec<u32>,
    /// On-device top-k values over logits (diagnostics; empty when not computed).
    pub head_topk_val: Vec<f32>,
    /// True when the full vocab logit vector was read back to the host.
    /// Default device path leaves this false (token + diagnostics only).
    pub head_full_logits_readback: bool,
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

    /// Run `tokens` from an empty cache and return the logits after the last
    /// one, plus a trace of what the routers and the indexer chose.
    pub fn forward(&self, tokens: &[u32]) -> Result<(Vec<f32>, GlmTrace)> {
        let mut session = GlmSession::new(&self.arch);
        forward_impl(&self.weights, &self.arch, &mut session, tokens, 0)
    }
}

/// RoPE cos/sin for one position: `qk_rope_head_dim/2` of each. The
/// reference builds a `rotary_dim`-wide table by concatenating the
/// frequencies with themselves and then takes the first half, which is
/// exactly these values.
pub(crate) fn rope_cos_sin(arch: &GlmArch, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let rot = arch.qk_rope_head_dim;
    let half = rot / 2;
    let mut cos = vec![0f32; half];
    let mut sin = vec![0f32; half];
    for i in 0..half {
        // f32 throughout: the reference forces float32 here, and matching
        // it in f64 would drift from the thing being reproduced.
        let inv_freq = 1.0f32 / arch.rope_theta.powf(2.0 * i as f32 / rot as f32);
        let theta = pos as f32 * inv_freq;
        cos[i] = theta.cos();
        sin[i] = theta.sin();
    }
    (cos, sin)
}

/// The DSA indexer: which cached keys this token is allowed to attend to.
#[allow(clippy::too_many_arguments)]
fn indexer_topk(
    weights: &dyn WeightAccess,
    arch: &GlmArch,
    prefix: &str,
    hidden: &[f32],
    q_resid: &[f32],
    cache: &mut LayerCache,
    pos: usize,
    cos: &[f32],
    sin: &[f32],
) -> Result<Vec<usize>> {
    let a = arch;
    let (ih, idim, rot) = (a.index_n_heads, a.index_head_dim, a.qk_rope_head_dim);
    let idx = format!("{prefix}.indexer");

    let q = weights.matvec(&format!("{idx}.wq_b.weight"), q_resid)?;
    if q.len() != ih * idim {
        return Err(Error::Gravity(format!(
            "indexer wq_b produced {} values, expected {ih} heads * {idim}",
            q.len()
        )));
    }
    let k_raw = weights.matvec(&format!("{idx}.wk.weight"), hidden)?;
    let k = layernorm(
        &k_raw,
        &weights.dense(&format!("{idx}.k_norm.weight"))?,
        &weights.dense(&format!("{idx}.k_norm.bias"))?,
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
    let mut head_weights = weights.matvec(&format!("{idx}.weights_proj.weight"), hidden)?;
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
fn router(
    weights: &dyn WeightAccess,
    arch: &GlmArch,
    prefix: &str,
    hidden: &[f32],
) -> Result<(Vec<usize>, Vec<f32>)> {
    use crate::cost_ledger::{self, Bucket};
    let _route = cost_ledger::Scope::new(Bucket::Routing);
    let a = arch;
    let logits = weights.matvec(&format!("{prefix}.gate.weight"), hidden)?;
    let scores: Vec<f32> = logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
    let bias = weights.dense(&format!("{prefix}.gate.e_score_correction_bias"))?;
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

fn dense_mlp(weights: &dyn WeightAccess, prefix: &str, x: &[f32]) -> Result<Vec<f32>> {
    let gate = weights.matvec(&format!("{prefix}.gate_proj.weight"), x)?;
    let up = weights.matvec(&format!("{prefix}.up_proj.weight"), x)?;
    weights.matvec(&format!("{prefix}.down_proj.weight"), &silu_mul(&gate, &up))
}

/// `dense_mlp` for every prefix in `prefixes` against the same `x`, batched
/// into three round trips (gate, up, down) instead of three per prefix. An
/// MoE layer's experts are mutually independent -- none reads another's
/// output -- so nothing about correctness requires visiting them one at a
/// time; only `down_proj`'s input differs per expert, and even that batches,
/// since `matvec_batch` takes its own `x` per call.
fn batched_mlp(weights: &dyn WeightAccess, prefixes: &[String], x: &[f32]) -> Result<Vec<Vec<f32>>> {
    let gate_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.gate_proj.weight")).collect();
    let up_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.up_proj.weight")).collect();
    let gate_calls: Vec<(&str, &[f32])> = gate_names.iter().map(|n| (n.as_str(), x)).collect();
    let up_calls: Vec<(&str, &[f32])> = up_names.iter().map(|n| (n.as_str(), x)).collect();
    let gates = weights.matvec_batch(&gate_calls)?;
    let ups = weights.matvec_batch(&up_calls)?;

    let hidden: Vec<Vec<f32>> = gates.iter().zip(&ups).map(|(g, u)| silu_mul(g, u)).collect();
    let down_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.down_proj.weight")).collect();
    let down_calls: Vec<(&str, &[f32])> = down_names
        .iter()
        .zip(&hidden)
        .map(|(n, h)| (n.as_str(), h.as_slice()))
        .collect();
    weights.matvec_batch(&down_calls)
}

fn routed_moe(
    weights: &dyn WeightAccess,
    arch: &GlmArch,
    prefix: &str,
    x: &[f32],
) -> Result<(Vec<f32>, Vec<usize>)> {
    use crate::cost_ledger::{self, Bucket};

    let (indices, moe_weights) = router(weights, arch, prefix, x)?;
    // The reference accumulates in ascending expert order, so match it:
    // float addition is not associative and the artifact is graded to
    // 1e-4, not to "close enough". Batching changes when a matvec's bytes
    // cross the bus, never the order results are summed in below.
    let mut order: Vec<usize> = (0..indices.len()).collect();
    order.sort_by_key(|&s| indices[s]);

    // Routed experts and the shared expert are co-batched into three CBs
    // (gate/up/down). Metal exclusive time lands in metal_* buckets; the
    // scopes below capture the CPU orchestration around those dispatches
    // and the weighted residual accumulate. Shared is last in `prefixes`.
    let prefixes: Vec<String> = order
        .iter()
        .map(|&slot| format!("{prefix}.experts.{}", indices[slot]))
        .chain(std::iter::once(format!("{prefix}.shared_experts")))
        .collect();

    let mut outs = {
        // Parent scope is routed_experts for the co-batch; shared_experts
        // only owns the final residual add so we do not invent a clean
        // GPU split that the runtime does not actually perform.
        let _routed = cost_ledger::Scope::new(Bucket::RoutedExperts);
        batched_mlp(weights, &prefixes, x)?
    };
    let shared = outs.pop().expect("prefixes has the shared expert last");

    let mut routed = {
        let _routed = cost_ledger::Scope::new(Bucket::RoutedExperts);
        let mut routed = vec![0f32; x.len()];
        cost_ledger::record_allocation((routed.len() * 4) as u64);
        for (out, &slot) in outs.iter().zip(&order) {
            for (r, o) in routed.iter_mut().zip(out) {
                *r += o * moe_weights[slot];
            }
        }
        routed
    };
    {
        let _shared = cost_ledger::Scope::new(Bucket::SharedExperts);
        for (r, s) in routed.iter_mut().zip(&shared) {
            *r += s;
        }
    }
    Ok((routed, indices))
}

/// Run `tokens` starting at `start_pos` against `session`'s cache and return
/// the logits after the last one, plus a trace of what the routers and the
/// indexer chose. Shared by [`GravityGlm::forward`] (CPU, always starts a
/// fresh session at position 0) and the GPU-resident path, which reuses one
/// session across a whole generation so incremental decode never repeats
/// work a previous call already did.
fn forward_impl(
    weights: &dyn WeightAccess,
    arch: &GlmArch,
    session: &mut GlmSession,
    tokens: &[u32],
    start_pos: usize,
) -> Result<(Vec<f32>, GlmTrace)> {
    use crate::cost_ledger::{self, Bucket};

    if tokens.is_empty() {
        return Err(Error::Gravity("forward: no tokens".into()));
    }
    let a = arch;
    let qk_dim = a.qk_dim();
    let mut logits = Vec::new();
    let mut trace = GlmTrace::default();

    // Gate geometry for active routed-expert bytes (projection byte size is
    // filled in live by the GPU cache when known; default is the sealed figure).
    cost_ledger::set_geometry_active_bytes(cost_ledger::geometry_active_bytes(
        a.n_layers,
        a.num_experts_per_tok,
        None,
    ));

    for (i, &token) in tokens.iter().enumerate() {
        let pos = start_pos + i;
        if token as usize >= a.vocab_size {
            return Err(Error::Gravity(format!(
                "token {token} out of range for vocab_size {}",
                a.vocab_size
            )));
        }
        let mut x = {
            let _orch = cost_ledger::Scope::new(Bucket::CpuOrchestration);
            weights.row("model.embed_tokens.weight", token as usize, a.hidden)?
        };
        let (cos, sin) = {
            let _orch = cost_ledger::Scope::new(Bucket::CpuOrchestration);
            rope_cos_sin(arch, pos)
        };
        let mut shared_topk: Option<Vec<usize>> = None;
        trace.expert_choices.clear();

        for layer in 0..a.n_layers {
            let p = format!("model.layers.{layer}");
            let attn_p = format!("{p}.self_attn");

            // Attention + IndexShare: projections, DSA indexer, sparse attend,
            // o_proj residual. Nested metal/verify buckets steal exclusive time.
            let topk = {
                let _attn = cost_ledger::Scope::new(Bucket::AttentionAndIndexShare);
                let h = rmsnorm(
                    &x,
                    &weights.dense(&format!("{p}.input_layernorm.weight"))?,
                    a.rms_norm_eps,
                );

                // Queries through the low-rank path.
                let q_a = weights.matvec(&format!("{attn_p}.q_a_proj.weight"), &h)?;
                let q_resid = rmsnorm(
                    &q_a,
                    &weights.dense(&format!("{attn_p}.q_a_layernorm.weight"))?,
                    a.rms_norm_eps,
                );
                let q = weights.matvec(&format!("{attn_p}.q_b_proj.weight"), &q_resid)?;

                // Keys and values through the compressed MLA latent. The
                // rope part is shared across heads (MQA on that slice).
                let compressed =
                    weights.matvec(&format!("{attn_p}.kv_a_proj_with_mqa.weight"), &h)?;
                let k_latent = rmsnorm(
                    &compressed[..a.kv_lora_rank],
                    &weights.dense(&format!("{attn_p}.kv_a_layernorm.weight"))?,
                    a.rms_norm_eps,
                );
                let k_rot = rope_interleaved(&compressed[a.kv_lora_rank..], &cos, &sin);
                let kv = weights.matvec(&format!("{attn_p}.kv_b_proj.weight"), &k_latent)?;

                {
                    let _kv = cost_ledger::Scope::new(Bucket::KvUpdate);
                    let per_head_kv = a.qk_nope_head_dim + a.v_head_dim;
                    let cache = &mut session.caches[layer];
                    for head in 0..a.n_heads {
                        let src = &kv[head * per_head_kv..(head + 1) * per_head_kv];
                        cache.keys.extend_from_slice(&src[..a.qk_nope_head_dim]);
                        cache.keys.extend_from_slice(&k_rot);
                        cache.values.extend_from_slice(&src[a.qk_nope_head_dim..]);
                    }
                }

                // Assemble this token's queries: unrotated nope half, then
                // the interleaved-rotated rope half.
                let mut queries = vec![0f32; a.n_heads * qk_dim];
                cost_ledger::record_allocation((queries.len() * 4) as u64);
                for head in 0..a.n_heads {
                    let src = &q[head * qk_dim..(head + 1) * qk_dim];
                    let dst = &mut queries[head * qk_dim..(head + 1) * qk_dim];
                    dst[..a.qk_nope_head_dim].copy_from_slice(&src[..a.qk_nope_head_dim]);
                    dst[a.qk_nope_head_dim..]
                        .copy_from_slice(&rope_interleaved(&src[a.qk_nope_head_dim..], &cos, &sin));
                }

                let topk = match a.indexer_types[layer].as_str() {
                    "full" => {
                        let t = indexer_topk(
                            weights, arch, &attn_p, &h, &q_resid, &mut session.caches[layer], pos,
                            &cos, &sin,
                        )?;
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
                let cache = &session.caches[layer];
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
                cost_ledger::record_allocation((context.len() * 4 + scores.len() * 4) as u64);
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
                        for (o, v) in out
                            .iter_mut()
                            .zip(&cache.values[off..off + a.v_head_dim])
                        {
                            *o += w * v;
                        }
                    }
                }

                let attn_out = weights.matvec(&format!("{attn_p}.o_proj.weight"), &context)?;
                for (xv, o) in x.iter_mut().zip(&attn_out) {
                    *xv += o;
                }
                topk
            };

            let h2 = {
                let _orch = cost_ledger::Scope::new(Bucket::CpuOrchestration);
                rmsnorm(
                    &x,
                    &weights.dense(&format!("{p}.post_attention_layernorm.weight"))?,
                    a.rms_norm_eps,
                )
            };
            let mlp_out = match a.mlp_layer_types[layer].as_str() {
                "dense" => {
                    let _orch = cost_ledger::Scope::new(Bucket::CpuOrchestration);
                    dense_mlp(weights, &format!("{p}.mlp"), &h2)?
                }
                "sparse" => {
                    let (out, experts) = routed_moe(weights, arch, &format!("{p}.mlp"), &h2)?;
                    trace.expert_choices.push(experts);
                    out
                }
                other => {
                    return Err(Error::Gravity(format!(
                        "layer {layer}: unknown MLP type {other:?}"
                    )))
                }
            };
            {
                let _orch = cost_ledger::Scope::new(Bucket::CpuOrchestration);
                for (xv, m) in x.iter_mut().zip(&mlp_out) {
                    *xv += m;
                }
            }

            if layer + 1 == a.n_layers {
                trace.final_topk = topk;
            }
        }

        {
            let _head = cost_ledger::Scope::new(Bucket::FinalHeadAndSampling);
            let final_hidden = rmsnorm(&x, &weights.dense("model.norm.weight")?, a.rms_norm_eps);
            logits = weights.matvec("lm_head.weight", &final_hidden)?;
        }
    }
    Ok((logits, trace))
}

/// Opt-in GPU-resident decode state (activations, KV, router/top-k/expert
/// offsets). Default off — the host-state path remains the parity oracle.
/// See [`estimate_host_state_waits_per_token`] / [`estimate_resident_waits_per_token`].
pub const GPU_RESIDENT_STATE_ENV: &str = "HAWKING_GLM_GPU_RESIDENT_STATE";

/// Whether [`GPU_RESIDENT_STATE_ENV`] requests the resident decode path.
pub fn gpu_resident_state_enabled() -> bool {
    crate::env_on(GPU_RESIDENT_STATE_ENV)
}

/// Opt-in device-resident native.bf16 matvec + GPU head sampling for GLM.
///
/// Default off — host dense matvec remains the parity oracle. When set:
/// - every rank-2 `native.bf16` matvec target (flagship: `lm_head.weight`,
///   indexer projections, router `gate.weight`) is uploaded once as raw bf16
///   (no host widen), kept under the GPU weight-cache budget, and projected
///   with `gemv_native_bf16_seq` (bf16 input, sequential f32 accumulate);
/// - on the resident path the head runs blockwise logits + argmax + top-k on
///   device and **reads back only the token plus top-k diagnostics** (not the
///   154,880-element logit vector). Full logits require
///   [`GPU_LM_HEAD_FULL_LOGITS_ENV`]=1 (parity / debug only).
///
/// Integrates with the existing `GpuWeightCache` / resident-state path; does
/// not invent a second cache. Default resident path with this flag unset is
/// unchanged (Parity V2.1 item 6).
pub const GPU_LM_HEAD_ENV: &str = "HAWKING_GLM_GPU_LM_HEAD";

/// Opt-in full vocab logit readback on the device lm_head path.
///
/// Default off. When unset under [`GPU_LM_HEAD_ENV`], the resident head returns
/// an empty logit vector and fills [`GlmTrace::sample_token`] /
/// [`GlmTrace::head_topk_idx`] instead. Set to `1` for continuous-logit parity
/// against the host oracle / FP64 authority.
pub const GPU_LM_HEAD_FULL_LOGITS_ENV: &str = "HAWKING_GLM_GPU_LM_HEAD_FULL_LOGITS";

/// Default diagnostic top-k size for device head sampling (exact decision set).
pub const GPU_LM_HEAD_DIAG_TOPK: u32 = 5;

/// Whether [`GPU_LM_HEAD_ENV`] requests the device native-bf16 path.
pub fn gpu_lm_head_enabled() -> bool {
    crate::env_on(GPU_LM_HEAD_ENV)
}

/// Whether the device head should also pull the full logit vector to the host.
pub fn gpu_lm_head_full_logits_enabled() -> bool {
    crate::env_on(GPU_LM_HEAD_FULL_LOGITS_ENV)
}

/// Opt-in **expert-wave** device path for resident GLM MoE / dense MLP.
///
/// Default **off**. When set with [`GPU_RESIDENT_STATE_ENV`]=1, each MLP layer
/// encodes `gate + up → SiLU → down` (and MoE weighted combine) into **one**
/// command buffer instead of three `matvec_batch` commits with a host silu
/// readback between them. Static target for the `batched_mlp` bucket alone:
/// **234 → 78** drains/token on the flagship schedule (3 waits/layer → 1).
///
/// Isolation (Parity V2.1 item 6): the default resident path is byte-identical
/// and untouched when this flag is unset. Sealed Math-Preserve unchanged.
/// Numerics under this flag use V2.1 (FP64 authority, continuous metrics +
/// exact discrete decisions) — not bit-identical Metal-vs-libm `exp`.
pub const GPU_EXPERT_WAVE_ENV: &str = "HAWKING_GLM_GPU_EXPERT_WAVE";

/// Whether [`GPU_EXPERT_WAVE_ENV`] requests the collapsed expert-wave path.
pub fn gpu_expert_wave_enabled() -> bool {
    crate::env_on(GPU_EXPERT_WAVE_ENV)
}

/// Static `commit_and_wait` count on the host-state GPU path (default).
/// Matches the measured ~1,171 figure on the flagship schedule.
pub fn estimate_host_state_waits_per_token(arch: &GlmArch) -> u64 {
    let mut waits = 0u64;
    for layer in 0..arch.n_layers {
        waits += 5; // q_a, q_b, kv_a, kv_b, o_proj
        if arch.indexer_types[layer] == "full" {
            waits += 3; // wq_b, wk, weights_proj
        }
        match arch.mlp_layer_types[layer].as_str() {
            "dense" => waits += 3,
            "sparse" => waits += 1 + 3, // router + 3 expert batches
            _ => {}
        }
    }
    waits + 1 // lm_head
}

/// Static `commit_and_wait` count on the resident-state path.
///
/// Attention projections that share a dependency rank are co-issued (q_a with
/// kv_a, q_b with kv_b); expert gate/up/down stay three batched waits like the
/// host path, with the residual / KV living on device between them. Live counts
/// come from the resident forward's wait counter when a device is available.
///
/// **Default path only.** When [`gpu_expert_wave_enabled`] the MLP portion
/// collapses — see [`estimate_resident_expert_wave_waits_per_token`]. This
/// function's numbers must stay fixed (Parity V2.1 item 6).
pub fn estimate_resident_waits_per_token(arch: &GlmArch) -> u64 {
    let mut waits = 0u64;
    for layer in 0..arch.n_layers {
        // Co-issued q_a+kv_a, then q_b+kv_b, then o_proj.
        waits += 3;
        if arch.indexer_types[layer] == "full" {
            // wq_b+wk together, then weights_proj
            waits += 2;
        }
        match arch.mlp_layer_types[layer].as_str() {
            "dense" => waits += 2, // gate+up, then down
            "sparse" => {
                // router + three expert batches (gate/up/down, routed+shared)
                waits += 1 + 3;
            }
            _ => {}
        }
    }
    waits + 1 // lm_head boundary
}

/// Static drains from `batched_mlp` alone (gate / up / down commits).
///
/// Default: **3 per layer** (dense and sparse). Expert-wave: **1 per layer**.
/// Flagship (78 layers): **234 → 78**. Labelled static; live counts need a
/// device and the wait counter on `forward_resident_counted`.
pub fn estimate_batched_mlp_drains_per_token(arch: &GlmArch, expert_wave: bool) -> u64 {
    let per_layer = if expert_wave { 1u64 } else { 3u64 };
    (arch.n_layers as u64).saturating_mul(per_layer)
}

/// Resident-state wait estimate with the expert-wave MLP collapse enabled.
///
/// Same attention / indexer / router / head accounting as
/// [`estimate_resident_waits_per_token`]; MLP is one commit per layer (dense
/// or sparse) instead of three expert batches / two dense stages.
pub fn estimate_resident_expert_wave_waits_per_token(arch: &GlmArch) -> u64 {
    let mut waits = 0u64;
    for layer in 0..arch.n_layers {
        waits += 3; // co-issued attention
        if arch.indexer_types[layer] == "full" {
            waits += 2;
        }
        match arch.mlp_layer_types[layer].as_str() {
            "dense" => waits += 1, // fused gate+up+silu+down
            "sparse" => waits += 1 + 1, // router + fused expert wave
            _ => {}
        }
    }
    waits + 1 // lm_head boundary
}

/// Env var for the GPU weight-cache byte budget. Explicit (bytes, not
/// inferred from free RAM). Unset → [`DEFAULT_GPU_WEIGHT_CACHE_BUDGET_BYTES`].
pub const GPU_WEIGHT_CACHE_BUDGET_ENV: &str = "HAWKING_GRAVITY_GPU_CACHE_BUDGET_BYTES";

/// Default residency budget: 32 GiB. Enough for many MoE tokens of working
/// set on a 96 GiB unified-memory host without claiming the whole machine.
/// Override with [`GPU_WEIGHT_CACHE_BUDGET_ENV`].
pub const DEFAULT_GPU_WEIGHT_CACHE_BUDGET_BYTES: u64 = 32 * 1024 * 1024 * 1024;

/// Resolve the GPU weight-cache budget from the environment, or the default.
pub fn gpu_weight_cache_budget_bytes() -> Result<u64> {
    match std::env::var(GPU_WEIGHT_CACHE_BUDGET_ENV) {
        Ok(raw) => {
            let bytes: u64 = raw.parse().map_err(|_| {
                Error::Gravity(format!(
                    "{GPU_WEIGHT_CACHE_BUDGET_ENV}={raw:?} is not an integer byte count"
                ))
            })?;
            if bytes == 0 {
                return Err(Error::Gravity(format!(
                    "{GPU_WEIGHT_CACHE_BUDGET_ENV} must be > 0 (got 0)"
                )));
            }
            Ok(bytes)
        }
        Err(std::env::VarError::NotPresent) => Ok(DEFAULT_GPU_WEIGHT_CACHE_BUDGET_BYTES),
        Err(std::env::VarError::NotUnicode(_)) => Err(Error::Gravity(format!(
            "{GPU_WEIGHT_CACHE_BUDGET_ENV} is not valid UTF-8"
        ))),
    }
}

/// Snapshot of GPU weight-cache residency. Surfaced on the model and in the
/// BASE_TRUE_TPS receipt so a long run has an explicit number rather than
/// "whatever the HashMap happened to hold".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuWeightCacheStats {
    pub budget_bytes: u64,
    pub resident_bytes: u64,
    pub high_water_bytes: u64,
    pub entries: usize,
    pub evictions: u64,
}

/// Byte-budgeted LRU of named tensors. Platform-independent so the eviction
/// policy can be unit-tested without Metal. The GPU path holds one of these
/// under a mutex and never evicts under a concurrent encode: `matvec` /
/// `matvec_batch` keep the guard across dispatch, which is synchronous.
///
/// **Pinning:** [`BoundedLru::admit_pinned`] takes a pin set of names that
/// must stay resident for the current call (an MoE layer's ~8×3 projections).
/// Eviction only considers entries outside that set, so a batch can never
/// punch a hole in itself. If the pinned working set alone exceeds the
/// budget, admission fails loudly instead of thrashing.
#[derive(Debug)]
pub struct BoundedLru<T> {
    map: std::collections::HashMap<String, LruEntry<T>>,
    budget_bytes: u64,
    resident_bytes: u64,
    high_water_bytes: u64,
    clock: u64,
    evictions: u64,
}

#[derive(Debug)]
struct LruEntry<T> {
    value: T,
    bytes: u64,
    last_tick: u64,
}

impl<T> BoundedLru<T> {
    /// Build a cache with an explicit byte budget. `budget_bytes == 0` is
    /// rejected: a zero budget can never admit a tensor and would only thrash.
    pub fn new(budget_bytes: u64) -> Result<Self> {
        if budget_bytes == 0 {
            return Err(Error::Gravity(
                "GPU weight cache budget must be > 0".into(),
            ));
        }
        Ok(Self {
            map: std::collections::HashMap::new(),
            budget_bytes,
            resident_bytes: 0,
            high_water_bytes: 0,
            clock: 0,
            evictions: 0,
        })
    }

    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn high_water_bytes(&self) -> u64 {
        self.high_water_bytes
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Option<&T> {
        self.map.get(name).map(|e| &e.value)
    }

    pub fn stats(&self) -> GpuWeightCacheStats {
        GpuWeightCacheStats {
            budget_bytes: self.budget_bytes,
            resident_bytes: self.resident_bytes,
            high_water_bytes: self.high_water_bytes,
            entries: self.map.len(),
            evictions: self.evictions,
        }
    }

    /// Mark `name` as most-recently used. No-op if absent.
    pub fn touch(&mut self, name: &str) {
        if let Some(e) = self.map.get_mut(name) {
            self.clock = self.clock.saturating_add(1);
            e.last_tick = self.clock;
        }
    }

    /// Admit `items` while pinning every name in `pin`.
    ///
    /// - Existing pin members are touched (LRU refresh) and not re-inserted.
    /// - Items whose name is already resident are dropped (caller lost a race
    ///   or passed a duplicate); the resident copy is touched.
    /// - Room is made only by evicting entries **outside** `pin`.
    /// - If the pinned working set (already-resident pin members + new items)
    ///   exceeds the budget, returns `Err` without mutating residency for the
    ///   new items — never thrash inside the pin set.
    pub fn admit_pinned(
        &mut self,
        items: Vec<(String, T, u64)>,
        pin: &std::collections::HashSet<String>,
    ) -> Result<()> {
        for name in pin {
            self.touch(name);
        }

        let mut to_insert = Vec::with_capacity(items.len());
        for (name, value, bytes) in items {
            if self.map.contains_key(&name) {
                self.touch(&name);
                continue;
            }
            if bytes > self.budget_bytes {
                return Err(Error::Gravity(format!(
                    "tensor {name} ({bytes} B) alone exceeds GPU weight cache budget \
                     ({} B); raise {GPU_WEIGHT_CACHE_BUDGET_ENV}",
                    self.budget_bytes
                )));
            }
            to_insert.push((name, value, bytes));
        }

        if to_insert.is_empty() {
            return Ok(());
        }

        let new_bytes: u64 = to_insert.iter().map(|(_, _, b)| *b).sum();
        let mut pinned_bytes = new_bytes;
        for name in pin {
            if let Some(e) = self.map.get(name) {
                pinned_bytes = pinned_bytes.saturating_add(e.bytes);
            }
        }
        if pinned_bytes > self.budget_bytes {
            return Err(Error::Gravity(format!(
                "pinned GPU weight working set ({pinned_bytes} B) exceeds cache budget \
                 ({} B); raise {GPU_WEIGHT_CACHE_BUDGET_ENV} or shrink the batch",
                self.budget_bytes
            )));
        }

        while self.resident_bytes.saturating_add(new_bytes) > self.budget_bytes {
            if !self.evict_one_unpinned(pin) {
                // pinned_bytes check above should make this unreachable.
                return Err(Error::Gravity(format!(
                    "GPU weight cache cannot free enough bytes for a {new_bytes} B admission \
                     under budget {} B (pinned set blocked eviction)",
                    self.budget_bytes
                )));
            }
        }

        for (name, value, bytes) in to_insert {
            self.clock = self.clock.saturating_add(1);
            self.map.insert(
                name,
                LruEntry {
                    value,
                    bytes,
                    last_tick: self.clock,
                },
            );
            self.resident_bytes = self.resident_bytes.saturating_add(bytes);
            if self.resident_bytes > self.high_water_bytes {
                self.high_water_bytes = self.resident_bytes;
            }
        }
        Ok(())
    }

    /// Evict the least-recently-used entry whose name is not in `pin`.
    /// Returns whether an entry was removed.
    fn evict_one_unpinned(&mut self, pin: &std::collections::HashSet<String>) -> bool {
        let victim = self
            .map
            .iter()
            .filter(|(k, _)| !pin.contains(k.as_str()))
            .min_by_key(|(_, e)| e.last_tick)
            .map(|(k, _)| k.clone());
        let Some(name) = victim else {
            return false;
        };
        if let Some(e) = self.map.remove(&name) {
            self.resident_bytes = self.resident_bytes.saturating_sub(e.bytes);
            self.evictions = self.evictions.saturating_add(1);
            true
        } else {
            false
        }
    }
}

/// GPU-resident execution: the same [`forward_impl`] orchestration above,
/// against weights lazily uploaded to Metal buffers instead of decoded on
/// the CPU on every call.
#[cfg(target_os = "macos")]
pub mod gpu {
    use super::*;
    use crate::gravity::{matvec_dense, parse_pq_header, pq_sections, widen_native, PqHeader};
    use crate::metal::{MetalContext, TokenCommandBuffer};
    use metal::Buffer;
    use std::collections::HashSet;
    use std::sync::Mutex;

    /// Mirror of `GravityPQParams` in `shaders/gravity_pq.metal`: eight
    /// `uint`s in declaration order, `#[repr(C)]` so a pointer cast is a
    /// valid `set_bytes` payload. The same layout as
    /// `gravity_llama::gpu`'s copy -- both dispatch the same kernel -- kept
    /// as a second small struct rather than a shared one neither
    /// architecture module privately owns.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub(crate) struct PqParams {
        pub(crate) dim: u32,
        pub(crate) subspaces: u32,
        pub(crate) sub: u32,
        pub(crate) card: u32,
        pub(crate) rows: u32,
        pub(crate) cols: u32,
        pub(crate) nchunk: u32,
        pub(crate) bits: u32,
    }

    impl PqParams {
        fn from_header(h: &PqHeader) -> PqParams {
            PqParams {
                dim: h.d as u32,
                subspaces: h.s as u32,
                sub: h.sub as u32,
                card: h.card as u32,
                rows: h.rows,
                cols: h.cols,
                nchunk: h.nchunk,
                bits: h.bits as u32,
            }
        }
    }

    /// One tensor resident for matvec.
    ///
    /// - `Pq`: gravity-pq codebooks+codes on device.
    /// - `NativeCpu`: native tensors widened to f32 on the host (norms go
    ///   through [`GravityWeights::dense`], not this path; this is for
    ///   native matvec targets when [`super::GPU_LM_HEAD_ENV`] is off, or
    ///   non-bf16 / non-rank-2 natives).
    /// - `NativeGpuBf16`: raw `native.bf16` matrix uploaded once and kept
    ///   device-resident under the weight-cache budget. Covers every rank-2
    ///   `native.bf16` matvec target when the flag is on (lm_head, indexer,
    ///   router) so the f32 widen tax is not billed as active bytes.
    pub(crate) enum GpuTensor {
        Pq {
            codebooks: Buffer,
            codes: Buffer,
            params: PqParams,
        },
        NativeCpu(Vec<f32>),
        NativeGpuBf16 {
            buf: Buffer,
            rows: u32,
            cols: u32,
        },
    }

    /// A [`WeightAccess`] backend that uploads each `gravity-pq` tensor to
    /// the device on first use and keeps it under a **byte-budgeted LRU**.
    ///
    /// Unbounded residency was justified by "a short run touches 8 of 256
    /// experts per layer", but a real generation's routing diversity grows
    /// the resident set toward the whole routed artifact (~82 GB on
    /// General-R0). The budget is explicit
    /// ([`super::GPU_WEIGHT_CACHE_BUDGET_ENV`]); eviction is LRU outside the
    /// current call's pin set so a `matvec_batch` can never lose a tensor
    /// an earlier admission in the same batch just installed.
    ///
    /// The cache mutex is held across `dispatch_pq_matvec(_batch)`, which is
    /// synchronous (encode, commit, wait). Dropping an evicted `Buffer`
    /// after that wait is safe: Metal command buffers retain the resources
    /// they use for the lifetime of the encode.
    pub struct GpuWeightCache {
        pub(crate) ctx: MetalContext,
        pub(crate) weights: GravityWeights,
        pub(crate) cache: Mutex<BoundedLru<GpuTensor>>,
    }

    impl GpuWeightCache {
        /// Resident byte size of a cached tensor.
        /// `Pq` = codebooks + codes buffer lengths; `NativeCpu` = f32 vec × 4;
        /// `NativeGpuBf16` = raw bf16 buffer length (no host widen).
        fn tensor_bytes(t: &GpuTensor) -> u64 {
            match t {
                GpuTensor::Pq {
                    codebooks, codes, ..
                } => codebooks.length() + codes.length(),
                GpuTensor::NativeCpu(v) => (v.len() as u64).saturating_mul(4),
                GpuTensor::NativeGpuBf16 { buf, .. } => buf.length(),
            }
        }

        /// Decode + upload one tensor. Does not touch the cache.
        fn load_tensor(&self, name: &str) -> Result<(GpuTensor, u64)> {
            use crate::cost_ledger::{self, Bucket};

            let (codec, blob, shape) = self.weights.raw_payload_with_shape(name)?;
            let entry = if codec == "gravity-pq" {
                let (h, cb, codes_padded) = {
                    let _decode = cost_ledger::Scope::new(Bucket::PackedIndexDecode);
                    let h = parse_pq_header(&blob)?;
                    if h.rotate != 0 {
                        return Err(Error::Gravity(format!(
                            "tensor {name}: rotated gravity-pq artifacts (rotate=1) are not yet \
                             supported on the GPU path"
                        )));
                    }
                    let (cb, codes) = pq_sections(&blob)?;
                    // Four bytes of tail padding so the kernel's whole-word read
                    // at the last index's byte offset stays in bounds.
                    let mut codes_padded = Vec::with_capacity(codes.len() + 4);
                    codes_padded.extend_from_slice(codes);
                    codes_padded.extend_from_slice(&[0u8; 4]);
                    (h, cb, codes_padded)
                };
                let (codebooks, codes) = {
                    let _xfer = cost_ledger::Scope::new(Bucket::HostDeviceTransfer);
                    let codebooks = self.ctx.new_buffer_with_bytes_checked(cb)?;
                    let codes = self.ctx.new_buffer_with_bytes_checked(&codes_padded)?;
                    cost_ledger::record_transfer(cb.len() as u64, true, "pq_codebooks_upload");
                    cost_ledger::record_transfer(
                        codes_padded.len() as u64,
                        true,
                        "pq_codes_upload",
                    );
                    cost_ledger::record_allocation(cb.len() as u64 + codes_padded.len() as u64);
                    (codebooks, codes)
                };
                GpuTensor::Pq {
                    codebooks,
                    codes,
                    params: PqParams::from_header(&h),
                }
            } else if codec.starts_with("native.") {
                // Device-resident native.bf16: upload raw bytes once (no host
                // widen). Flagship lm_head is 1.90 GB; indexer + router add
                // the rest of the native f32 widen tax (~2.53 GB of the
                // surplus). Rank-2 only — norms/biases stay host dense().
                // Gated by HAWKING_GLM_GPU_LM_HEAD; default path untouched.
                if super::gpu_lm_head_enabled() && codec == "native.bf16" && shape.len() == 2 {
                    let rows = shape[0] as u32;
                    let cols = shape[1] as u32;
                    let expect = (rows as u64)
                        .saturating_mul(cols as u64)
                        .saturating_mul(2);
                    if blob.len() as u64 != expect {
                        return Err(Error::Gravity(format!(
                            "tensor {name}: bf16 payload {} B != rows*cols*2 ({expect})",
                            blob.len()
                        )));
                    }
                    let xfer_tag = if name == "lm_head.weight" || name.ends_with("lm_head.weight") {
                        "lm_head_bf16_upload"
                    } else if name.contains(".indexer.") {
                        "indexer_bf16_upload"
                    } else if name.contains("gate.weight") {
                        "router_bf16_upload"
                    } else {
                        "native_bf16_upload"
                    };
                    let buf = {
                        let _xfer = cost_ledger::Scope::new(Bucket::HostDeviceTransfer);
                        let b = self.ctx.new_buffer_with_bytes_checked(&blob)?;
                        cost_ledger::record_transfer(blob.len() as u64, true, xfer_tag);
                        cost_ledger::record_allocation(blob.len() as u64);
                        b
                    };
                    GpuTensor::NativeGpuBf16 { buf, rows, cols }
                } else {
                    let _decode = cost_ledger::Scope::new(Bucket::PackedIndexDecode);
                    GpuTensor::NativeCpu(widen_native(&codec, &blob)?)
                }
            } else {
                return Err(Error::Gravity(format!(
                    "tensor {name}: unsupported codec {codec:?}"
                )));
            };
            let bytes = Self::tensor_bytes(&entry);
            cost_ledger::record_first_touch_load_bytes(bytes);
            Ok((entry, bytes))
        }

        /// Admit every name in the call set under one lock, pinning the whole
        /// set so mid-batch eviction cannot create a hole. Holds `cache`
        /// across load+admit; load is disk I/O but the alternative (ensure
        /// then re-lock) reintroduces a TOCTOU where another thread's
        /// admission could evict between ensure and use.
        pub(crate) fn ensure_many_locked(
            &self,
            cache: &mut BoundedLru<GpuTensor>,
            names: &[&str],
        ) -> Result<()> {
            let pin: HashSet<String> = names.iter().map(|s| (*s).to_string()).collect();
            let mut prepared = Vec::new();
            let mut seen_missing: HashSet<String> = HashSet::new();
            for &name in names {
                if cache.contains(name) {
                    cache.touch(name);
                    continue;
                }
                if !seen_missing.insert(name.to_string()) {
                    continue;
                }
                let (tensor, bytes) = self.load_tensor(name)?;
                prepared.push((name.to_string(), tensor, bytes));
            }
            cache.admit_pinned(prepared, &pin)
        }

        pub fn stats(&self) -> GpuWeightCacheStats {
            self.cache
                .lock()
                .expect("gpu weight cache mutex")
                .stats()
        }
    }

    impl WeightAccess for GpuWeightCache {
        // Norm weights and biases: small, natively carried, touched every
        // layer -- decoding them on the CPU each call is cheaper than the
        // round trip a GPU read-back would cost. `GravityWeights::dense`
        // memoizes the widened vector (and one-shot hash verification) so
        // this path no longer re-hashes the same bytes every token.
        fn dense(&self, name: &str) -> Result<Vec<f32>> {
            self.weights.dense(name)
        }

        // The embedding table's row lookup: one row, once per token. Also
        // not worth a device-resident path. Shares the native dense memo
        // for small tensors; large tables only memoize verification.
        fn row(&self, name: &str, index: usize, cols: usize) -> Result<Vec<f32>> {
            self.weights.row(name, index, cols)
        }

        fn matvec(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
            use crate::cost_ledger;
            cost_ledger::record_matvec_call();
            let mut cache = self.cache.lock().expect("gpu weight cache mutex");
            self.ensure_many_locked(&mut cache, &[name])?;
            match cache.get(name).expect("ensure just inserted it") {
                GpuTensor::NativeCpu(w) => {
                    // Widened f32 residency (native.bf16 artifacts pay a 2×
                    // traffic tax vs stored bytes). Category partition is what
                    // makes the 4×-vs-geometry figure explainable.
                    cost_ledger::record_active_bytes_for(name, (w.len() * 4) as u64);
                    matvec_dense(w, x, name)
                }
                GpuTensor::NativeGpuBf16 { buf, rows, cols } => {
                    // Bill stored bf16 size, not the 2× f32 widen tax.
                    cost_ledger::record_active_bytes_for(name, buf.length());
                    dispatch_gemv_native_bf16_seq(&self.ctx, buf, *rows, *cols, x)
                }
                GpuTensor::Pq {
                    codebooks,
                    codes,
                    params,
                } => {
                    // Device buffer lengths: codebooks + codes (+4 pad on
                    // codes at upload). Not the mmap slice, not page size.
                    let bytes = codebooks.length() + codes.length();
                    cost_ledger::record_active_bytes_for(name, bytes);
                    dispatch_pq_matvec(&self.ctx, codebooks, codes, *params, x)
                }
            }
        }

        /// The batching that makes an MoE layer affordable: every call this
        /// token's routed experts need goes into one command buffer instead
        /// of one synchronous round trip apiece. A command buffer's fixed
        /// submission/wait cost is what a straight per-matvec `dispatch`
        /// pays 8-9x over for a layer's worth of experts, regardless of how
        /// cheap the kernel itself is.
        ///
        /// Admission pins the whole name set before any eviction runs, so a
        /// budget hit cannot drop a tensor this batch just uploaded.
        fn matvec_batch(&self, calls: &[(&str, &[f32])]) -> Result<Vec<Vec<f32>>> {
            use crate::cost_ledger;
            cost_ledger::record_matvec_batch(calls.len() as u64);
            let names: Vec<&str> = calls.iter().map(|&(n, _)| n).collect();
            let mut cache = self.cache.lock().expect("gpu weight cache mutex");
            self.ensure_many_locked(&mut cache, &names)?;

            let mut results: Vec<Option<Vec<f32>>> = vec![None; calls.len()];
            let mut gpu_calls: Vec<(usize, &Buffer, &Buffer, PqParams, &[f32])> = Vec::new();
            for (i, &(name, x)) in calls.iter().enumerate() {
                match cache.get(name).expect("ensure just inserted it") {
                    GpuTensor::NativeCpu(w) => {
                        cost_ledger::record_active_bytes_for(name, (w.len() * 4) as u64);
                        results[i] = Some(matvec_dense(w, x, name)?);
                    }
                    GpuTensor::NativeGpuBf16 { buf, rows, cols } => {
                        // Device bf16: once-per-token for lm_head; indexer /
                        // router also land here under the same flag.
                        cost_ledger::record_active_bytes_for(name, buf.length());
                        results[i] =
                            Some(dispatch_gemv_native_bf16_seq(&self.ctx, buf, *rows, *cols, x)?);
                    }
                    GpuTensor::Pq {
                        codebooks,
                        codes,
                        params,
                    } => {
                        cost_ledger::record_active_bytes_for(
                            name,
                            codebooks.length() + codes.length(),
                        );
                        gpu_calls.push((i, codebooks, codes, *params, x));
                    }
                }
            }

            if !gpu_calls.is_empty() {
                let pq_calls: Vec<(&Buffer, &Buffer, PqParams, &[f32])> = gpu_calls
                    .iter()
                    .map(|&(_, cb, co, params, x)| (cb, co, params, x))
                    .collect();
                let outs = dispatch_pq_matvec_batch(&self.ctx, &pq_calls)?;
                for (&(i, ..), y) in gpu_calls.iter().zip(outs) {
                    results[i] = Some(y);
                }
            }

            results
                .into_iter()
                .enumerate()
                .map(|(i, r)| {
                    r.ok_or_else(|| Error::Gravity(format!("matvec_batch: no result for call {i}")))
                })
                .collect()
        }
    }

    /// Device-resident `native.bf16` GEMV: weights already on GPU; upload `x`,
    /// sequential accumulate per row (bit-identical to host widen+matvec),
    /// read logits back. Also used from the resident path via the buffer API.
    /// Public for parity tests and micro-benchmarks.
    pub fn dispatch_gemv_native_bf16_seq(
        ctx: &MetalContext,
        weight: &Buffer,
        rows: u32,
        cols: u32,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        use crate::cost_ledger::{self, Bucket};
        use std::time::Instant;

        if x.len() != cols as usize {
            return Err(Error::Gravity(format!(
                "gemv_native_bf16_seq: x.len() {} != cols {cols}",
                x.len()
            )));
        }
        let expect = (rows as u64).saturating_mul(cols as u64).saturating_mul(2);
        if weight.length() < expect {
            return Err(Error::Gravity(format!(
                "gemv_native_bf16_seq: weight buffer {} B < rows*cols*2 ({expect})",
                weight.length()
            )));
        }
        let x_bytes = std::mem::size_of_val(x) as u64;
        let y_bytes = rows as usize * std::mem::size_of::<f32>();
        let (x_buf, y_buf) = {
            let _xfer = cost_ledger::Scope::new(Bucket::HostDeviceTransfer);
            let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
            let y_buf = ctx.new_buffer(y_bytes);
            cost_ledger::record_transfer(x_bytes, true, "lm_head_x_upload");
            cost_ledger::record_allocation(x_bytes + y_bytes as u64);
            (x_buf, y_buf)
        };
        let rows_u = rows;
        let cols_u = cols;
        // One thread per output row; TG size 256 packs the grid densely.
        const TG: u32 = 256;
        let n_tg = rows.div_ceil(TG);
        let grid = (n_tg * TG, 1, 1);
        let tg = (TG, 1, 1);

        if cost_ledger::is_recording() {
            let t_encode = Instant::now();
            let mut tcb = TokenCommandBuffer::new(ctx);
            tcb.dispatch_threads("gemv_native_bf16_seq", grid, tg, |enc| {
                enc.set_buffer(0, Some(weight), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&y_buf), 0);
                enc.set_bytes(3, 4, &rows_u as *const u32 as *const _);
                enc.set_bytes(4, 4, &cols_u as *const u32 as *const _);
            })?;
            cost_ledger::add_duration(Bucket::MetalEncode, t_encode.elapsed());
            cost_ledger::record_dispatches(tcb.dispatch_count() as u64);
            tcb.commit_and_wait_split()?;
        } else {
            ctx.dispatch_threads("gemv_native_bf16_seq", grid, tg, |enc| {
                enc.set_buffer(0, Some(weight), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&y_buf), 0);
                enc.set_bytes(3, 4, &rows_u as *const u32 as *const _);
                enc.set_bytes(4, 4, &cols_u as *const u32 as *const _);
            })?;
        }

        let y = {
            let _xfer = cost_ledger::Scope::new(Bucket::HostDeviceTransfer);
            let y_ptr = y_buf.contents() as *const f32;
            let y = unsafe { std::slice::from_raw_parts(y_ptr, rows as usize) }.to_vec();
            cost_ledger::record_transfer(y_bytes as u64, false, "lm_head_y_download");
            cost_ledger::record_allocation(y_bytes as u64);
            y
        };
        Ok(y)
    }

    /// Encode bf16 GEMV into an existing command buffer (device x → device y).
    /// Caller commits. Used by the resident path so final_hidden never leaves
    /// the device for the projection.
    pub(crate) fn encode_gemv_native_bf16_seq(
        tcb: &mut TokenCommandBuffer<'_>,
        weight: &Buffer,
        rows: u32,
        cols: u32,
        x: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        const TG: u32 = 256;
        let n_tg = rows.div_ceil(TG);
        let grid = (n_tg * TG, 1, 1);
        let tg = (TG, 1, 1);
        let rows_u = rows;
        let cols_u = cols;
        let w = weight.clone();
        let xb = x.clone();
        let yb = y.clone();
        tcb.dispatch_threads("gemv_native_bf16_seq", grid, tg, move |enc| {
            enc.set_buffer(0, Some(&w), 0);
            enc.set_buffer(1, Some(&xb), 0);
            enc.set_buffer(2, Some(&yb), 0);
            enc.set_bytes(3, 4, &rows_u as *const u32 as *const _);
            enc.set_bytes(4, 4, &cols_u as *const u32 as *const _);
        })?;
        Ok(())
    }

    /// Encode greedy argmax over device logits → single u32 token buffer.
    pub(crate) fn encode_argmax_f32(
        tcb: &mut TokenCommandBuffer<'_>,
        logits: &Buffer,
        n: u32,
        token_out: &Buffer,
    ) -> Result<()> {
        const TG: u32 = 256;
        let n_u = n;
        let lg = logits.clone();
        let tok = token_out.clone();
        tcb.dispatch_threads("sample_argmax_f32", (TG, 1, 1), (TG, 1, 1), move |enc| {
            enc.set_buffer(0, Some(&lg), 0);
            enc.set_buffer(1, Some(&tok), 0);
            enc.set_bytes(2, 4, &n_u as *const u32 as *const _);
            enc.set_threadgroup_memory_length(0, (TG as u64) * 4);
            enc.set_threadgroup_memory_length(1, (TG as u64) * 4);
        })?;
        Ok(())
    }

    /// Encode parallel top-k over device logits (diagnostics + decision set).
    /// `k` must be ≤ 64 (Metal `sample_topk` limit). Outputs indices + values.
    pub(crate) fn encode_sample_topk_f32(
        tcb: &mut TokenCommandBuffer<'_>,
        logits: &Buffer,
        n: u32,
        k: u32,
        topk_idx: &Buffer,
        topk_val: &Buffer,
    ) -> Result<()> {
        const TG: u32 = 256;
        let n_u = n;
        let k_u = k.min(64);
        let lg = logits.clone();
        let idx = topk_idx.clone();
        let val = topk_val.clone();
        tcb.dispatch_threads("sample_topk", (TG, 1, 1), (TG, 1, 1), move |enc| {
            enc.set_buffer(0, Some(&lg), 0);
            enc.set_buffer(1, Some(&idx), 0);
            enc.set_buffer(2, Some(&val), 0);
            enc.set_bytes(3, 4, &n_u as *const u32 as *const _);
            enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
            enc.set_threadgroup_memory_length(0, (TG as u64) * 4);
            enc.set_threadgroup_memory_length(1, (TG as u64) * 4);
            // selected[64] for exclusion across rounds
            enc.set_threadgroup_memory_length(2, 64 * 4);
        })?;
        Ok(())
    }

    /// One `gravity_pq_matvec` dispatch against already-resident codebooks
    /// and codes: upload `x`, run, read `y` back. Same kernel and launch
    /// shape as [`crate::gravity::pq_matvec_metal`], which additionally
    /// uploads the codebooks and codes on every call -- right for a parity
    /// test against a single payload, wasteful for a weight a token
    /// revisits every layer.
    fn dispatch_pq_matvec(
        ctx: &MetalContext,
        codebooks: &Buffer,
        codes: &Buffer,
        params: PqParams,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        use crate::cost_ledger::{self, Bucket};
        use std::time::Instant;

        if x.len() != params.cols as usize {
            return Err(Error::Gravity(format!(
                "gpu matvec: x.len() {} != cols {}",
                x.len(),
                params.cols
            )));
        }
        let x_bytes = std::mem::size_of_val(x) as u64;
        let y_bytes = params.rows as usize * std::mem::size_of::<f32>();
        let (x_buf, y_buf) = {
            let _xfer = cost_ledger::Scope::new(Bucket::HostDeviceTransfer);
            let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
            let y_buf = ctx.new_buffer(y_bytes);
            cost_ledger::record_transfer(x_bytes, true, "matvec_x_upload");
            cost_ledger::record_allocation(x_bytes + y_bytes as u64);
            (x_buf, y_buf)
        };

        // One SIMD group (32 lanes) per output row, 8 SIMD groups (256
        // threads) per threadgroup; the kernel guards `row >= rows` for the
        // boundary threadgroup.
        const TG: u32 = 256;
        let n_tg = params.rows.div_ceil(8);
        // When the cost ledger is recording, encode into a TCB and split
        // submit vs synchronize so the three Metal buckets are distinct.
        // When off, the existing single-dispatch path is unchanged.
        if cost_ledger::is_recording() {
            let t_encode = Instant::now();
            let mut tcb = TokenCommandBuffer::new(ctx);
            tcb.dispatch_threads("gravity_pq_matvec", (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
                enc.set_buffer(0, Some(codebooks), 0);
                enc.set_buffer(1, Some(codes), 0);
                enc.set_buffer(2, Some(&x_buf), 0);
                enc.set_buffer(3, Some(&y_buf), 0);
                enc.set_bytes(
                    4,
                    std::mem::size_of::<PqParams>() as u64,
                    &params as *const PqParams as *const _,
                );
            })?;
            cost_ledger::add_duration(Bucket::MetalEncode, t_encode.elapsed());
            cost_ledger::record_dispatches(tcb.dispatch_count() as u64);
            tcb.commit_and_wait_split()?;
        } else {
            ctx.dispatch_threads("gravity_pq_matvec", (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
                enc.set_buffer(0, Some(codebooks), 0);
                enc.set_buffer(1, Some(codes), 0);
                enc.set_buffer(2, Some(&x_buf), 0);
                enc.set_buffer(3, Some(&y_buf), 0);
                enc.set_bytes(
                    4,
                    std::mem::size_of::<PqParams>() as u64,
                    &params as *const PqParams as *const _,
                );
            })?;
        }

        let y = {
            let _xfer = cost_ledger::Scope::new(Bucket::HostDeviceTransfer);
            let y_ptr = y_buf.contents() as *const f32;
            let y = unsafe { std::slice::from_raw_parts(y_ptr, params.rows as usize) }.to_vec();
            cost_ledger::record_transfer(y_bytes as u64, false, "matvec_y_download");
            cost_ledger::record_allocation(y_bytes as u64);
            y
        };
        Ok(y)
    }

    /// `calls.len()` independent `gravity_pq_matvec` dispatches into one
    /// [`TokenCommandBuffer`], one `commit_and_wait` for the whole batch.
    /// Every call gets its own `x`/`y` buffer pair -- `down_proj`'s calls
    /// each take a different expert's `silu_mul(gate, up)`, so a shared `x`
    /// would be wrong, not just less general.
    fn dispatch_pq_matvec_batch(
        ctx: &MetalContext,
        calls: &[(&Buffer, &Buffer, PqParams, &[f32])],
    ) -> Result<Vec<Vec<f32>>> {
        use crate::cost_ledger::{self, Bucket};
        use std::time::Instant;

        if calls.is_empty() {
            return Ok(Vec::new());
        }
        let mut x_bufs = Vec::with_capacity(calls.len());
        let mut y_bufs = Vec::with_capacity(calls.len());
        let mut y_lens = Vec::with_capacity(calls.len());
        {
            let _xfer = cost_ledger::Scope::new(Bucket::HostDeviceTransfer);
            for &(_, _, params, x) in calls {
                if x.len() != params.cols as usize {
                    return Err(Error::Gravity(format!(
                        "gpu matvec_batch: x.len() {} != cols {}",
                        x.len(),
                        params.cols
                    )));
                }
                let x_bytes = std::mem::size_of_val(x) as u64;
                let y_bytes = params.rows as usize * std::mem::size_of::<f32>();
                x_bufs.push(ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x)));
                y_bufs.push(ctx.new_buffer(y_bytes));
                y_lens.push(params.rows as usize);
                cost_ledger::record_transfer(x_bytes, true, "matvec_batch_x_upload");
                cost_ledger::record_allocation(x_bytes + y_bytes as u64);
            }
        }

        if cost_ledger::is_recording() {
            // Manual TCB so encode / submit / synchronize are separate lines.
            const TG: u32 = 256;
            let t_encode = Instant::now();
            let mut tcb = TokenCommandBuffer::new(ctx);
            for (i, &(codebooks, codes, params, _)) in calls.iter().enumerate() {
                let n_tg = params.rows.div_ceil(8);
                tcb.dispatch_threads(
                    "gravity_pq_matvec",
                    (n_tg * TG, 1, 1),
                    (TG, 1, 1),
                    |enc| {
                        enc.set_buffer(0, Some(codebooks), 0);
                        enc.set_buffer(1, Some(codes), 0);
                        enc.set_buffer(2, Some(&x_bufs[i]), 0);
                        enc.set_buffer(3, Some(&y_bufs[i]), 0);
                        enc.set_bytes(
                            4,
                            std::mem::size_of::<PqParams>() as u64,
                            &params as *const PqParams as *const _,
                        );
                    },
                )?;
            }
            let n_disp = tcb.dispatch_count() as u64;
            cost_ledger::add_duration(Bucket::MetalEncode, t_encode.elapsed());
            cost_ledger::record_dispatches(n_disp);

            // commit_and_wait is one submit + one sync for the whole batch.
            // Split by timing commit vs wait via the TCB's internal path when
            // possible; here we re-implement the commit/wait pair.
            tcb.commit_and_wait_split()?;
        } else {
            let mut tcb = TokenCommandBuffer::new(ctx);
            const TG: u32 = 256;
            for (i, &(codebooks, codes, params, _)) in calls.iter().enumerate() {
                let n_tg = params.rows.div_ceil(8);
                tcb.dispatch_threads(
                    "gravity_pq_matvec",
                    (n_tg * TG, 1, 1),
                    (TG, 1, 1),
                    |enc| {
                        enc.set_buffer(0, Some(codebooks), 0);
                        enc.set_buffer(1, Some(codes), 0);
                        enc.set_buffer(2, Some(&x_bufs[i]), 0);
                        enc.set_buffer(3, Some(&y_bufs[i]), 0);
                        enc.set_bytes(
                            4,
                            std::mem::size_of::<PqParams>() as u64,
                            &params as *const PqParams as *const _,
                        );
                    },
                )?;
            }
            tcb.commit_and_wait()?;
        }

        let _xfer = cost_ledger::Scope::new(Bucket::HostDeviceTransfer);
        Ok(calls
            .iter()
            .zip(&y_bufs)
            .zip(&y_lens)
            .map(|((&(_, _, params, _), y_buf), &rows)| {
                let y_ptr = y_buf.contents() as *const f32;
                let y = unsafe { std::slice::from_raw_parts(y_ptr, rows) }.to_vec();
                cost_ledger::record_transfer(
                    (params.rows as usize * std::mem::size_of::<f32>()) as u64,
                    false,
                    "matvec_batch_y_download",
                );
                cost_ledger::record_allocation((y.len() * 4) as u64);
                y
            })
            .collect())
    }

    /// A `.gravity` GLM-5.2 model with weights lazily resident on the GPU.
    ///
    /// Default: host-state orchestration ([`forward_impl`]) — KV and the
    /// residual stream are host `Vec<f32>`, every projection round-trips.
    /// With [`super::GPU_RESIDENT_STATE_ENV`]=1 the resident path keeps
    /// activations / KV / router state on device; see
    /// [`crate::gravity_glm_resident`].
    pub struct GravityGlmGpu {
        pub arch: GlmArch,
        weights: GpuWeightCache,
        session: Mutex<GlmSession>,
        /// Present when opened with the resident-state flag (or forced).
        resident: Option<crate::gravity_glm_resident::ResidentRuntime>,
        /// Sticky per-instance override; `None` means follow the env flag at
        /// open time (already reflected in `resident.is_some()`).
        resident_enabled: bool,
    }

    impl GravityGlmGpu {
        /// Open with a context this model owns. An `Engine` must be `Send +
        /// Sync`, and a borrowed context makes that impossible to express,
        /// so the model holds its own -- same reasoning as
        /// [`crate::gravity_llama::gpu::GravityLlamaGpu`].
        pub fn open_dir(dir: &Path, verify_hash: bool) -> Result<GravityGlmGpu> {
            Self::open_dir_with(MetalContext::new()?, dir, verify_hash)
        }

        pub fn open_dir_with(
            ctx: MetalContext,
            dir: &Path,
            verify_hash: bool,
        ) -> Result<GravityGlmGpu> {
            Self::open_dir_with_budget(ctx, dir, verify_hash, gpu_weight_cache_budget_bytes()?)
        }

        /// Open with an explicit GPU weight-cache byte budget (tests and
        /// callers that do not want the env default).
        pub fn open_dir_with_budget(
            ctx: MetalContext,
            dir: &Path,
            verify_hash: bool,
            budget_bytes: u64,
        ) -> Result<GravityGlmGpu> {
            Self::open_dir_with_budget_resident(
                ctx,
                dir,
                verify_hash,
                budget_bytes,
                super::gpu_resident_state_enabled(),
            )
        }

        /// Open with an explicit resident-state choice (parity tests force
        /// both sides without fighting the process environment).
        pub fn open_dir_with_budget_resident(
            ctx: MetalContext,
            dir: &Path,
            verify_hash: bool,
            budget_bytes: u64,
            resident_enabled: bool,
        ) -> Result<GravityGlmGpu> {
            let weights = GravityWeights::open_dir(dir, verify_hash)?;
            let arch = GlmArch::from_header(&weights.header)?;
            let session = Mutex::new(GlmSession::new(&arch));
            let resident = if resident_enabled {
                Some(crate::gravity_glm_resident::ResidentRuntime::new(&ctx, &arch)?)
            } else {
                None
            };
            Ok(GravityGlmGpu {
                weights: GpuWeightCache {
                    ctx,
                    weights,
                    cache: Mutex::new(BoundedLru::new(budget_bytes)?),
                },
                arch,
                session,
                resident,
                resident_enabled,
            })
        }

        /// Current GPU weight-cache residency (budget, live bytes, high-water).
        pub fn cache_stats(&self) -> GpuWeightCacheStats {
            self.weights.stats()
        }

        /// Whether this instance is on the device-resident decode path.
        pub fn resident_state_enabled(&self) -> bool {
            self.resident_enabled
        }

        /// Live `commit_and_wait` count from the last resident generation, if any.
        pub fn last_resident_waits(&self) -> Option<u64> {
            self.resident
                .as_ref()
                .map(|r| r.session.lock().expect("resident session").waits())
        }

        /// Run `tokens` from an empty cache -- the start of a new request on
        /// a model kept resident across many of them.
        pub fn forward(&self, tokens: &[u32]) -> Result<(Vec<f32>, GlmTrace)> {
            if let Some(rt) = &self.resident {
                let mut session = rt.session.lock().expect("resident session");
                session.reset();
                let (logits, trace, _waits) = crate::gravity_glm_resident::forward_resident(
                    &self.weights,
                    &self.arch,
                    &mut session,
                    &rt.pool,
                    tokens,
                    0,
                )?;
                return Ok((logits, trace));
            }
            let mut session = self.session.lock().expect("glm session mutex");
            session.reset();
            forward_impl(&self.weights, &self.arch, &mut session, tokens, 0)
        }

        /// Continue the current request's cache from `start_pos`: decode,
        /// one new token against whatever `forward` or a previous
        /// `forward_at` already built.
        pub fn forward_at(
            &self,
            tokens: &[u32],
            start_pos: usize,
        ) -> Result<(Vec<f32>, GlmTrace)> {
            if let Some(rt) = &self.resident {
                let mut session = rt.session.lock().expect("resident session");
                let (logits, trace, _waits) = crate::gravity_glm_resident::forward_resident(
                    &self.weights,
                    &self.arch,
                    &mut session,
                    &rt.pool,
                    tokens,
                    start_pos,
                )?;
                return Ok((logits, trace));
            }
            let mut session = self.session.lock().expect("glm session mutex");
            forward_impl(&self.weights, &self.arch, &mut session, tokens, start_pos)
        }

        /// Resident forward that also returns the live wait count for this call.
        pub fn forward_resident_counted(
            &self,
            tokens: &[u32],
        ) -> Result<(Vec<f32>, GlmTrace, u64)> {
            let rt = self.resident.as_ref().ok_or_else(|| {
                Error::Gravity(
                    "forward_resident_counted requires HAWKING_GLM_GPU_RESIDENT_STATE=1".into(),
                )
            })?;
            let mut session = rt.session.lock().expect("resident session");
            session.reset();
            crate::gravity_glm_resident::forward_resident(
                &self.weights,
                &self.arch,
                &mut session,
                &rt.pool,
                tokens,
                0,
            )
        }

        /// One decode step with the per-token cost ledger recording.
        /// Requires `HAWKING_COST_LEDGER=1` or a prior
        /// [`crate::cost_ledger::set_enabled(true)`]; when the ledger is
        /// off this is identical to [`forward_at`] and returns `None` for
        /// the report.
        pub fn forward_at_with_ledger(
            &self,
            tokens: &[u32],
            start_pos: usize,
        ) -> Result<(Vec<f32>, GlmTrace, Option<crate::cost_ledger::TokenCostReport>)> {
            crate::cost_ledger::begin_token();
            let result = self.forward_at(tokens, start_pos);
            let report = crate::cost_ledger::end_token();
            let (logits, trace) = result?;
            Ok((logits, trace, report))
        }
    }
}

/// `GravityGlmGpu` must be `Send + Sync` to be served behind the `Engine`
/// trait. This fails to compile the moment that stops being true, which is
/// the only way to notice: nothing else in the crate would.
#[cfg(all(test, target_os = "macos"))]
mod gpu_bounds {
    fn _assert_send_sync<T: Send + Sync>() {}
    #[test]
    fn gravity_glm_gpu_is_send_and_sync() {
        _assert_send_sync::<super::gpu::GravityGlmGpu>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// The reference's own selfcheck pins position zero to `[x0,x2,x1,x3]`
    /// for a 4-wide vector -- the concatenated layout, not the scattered
    /// one. If this passes with a scatter implementation, the test is wrong.
    #[test]
    fn interleaved_rope_position_zero_is_the_concatenated_layout() {
        let v = [0f32, 1.0, 2.0, 3.0];
        let got = rope_interleaved(&v, &[1.0, 1.0], &[0.0, 0.0]);
        assert_eq!(got, vec![0.0, 2.0, 1.0, 3.0]);
    }

    /// Resident-state flag defaults off so the host path stays the oracle.
    #[test]
    fn resident_state_flag_defaults_off() {
        // Ensure we do not inherit a sticky env from another test in-process.
        let prev = std::env::var_os(GPU_RESIDENT_STATE_ENV);
        std::env::remove_var(GPU_RESIDENT_STATE_ENV);
        assert!(!gpu_resident_state_enabled());
        match prev {
            Some(v) => std::env::set_var(GPU_RESIDENT_STATE_ENV, v),
            None => std::env::remove_var(GPU_RESIDENT_STATE_ENV),
        }
    }

    /// Device lm_head flag defaults off so host dense matvec stays the oracle.
    #[test]
    fn gpu_lm_head_flag_defaults_off() {
        let prev = std::env::var_os(GPU_LM_HEAD_ENV);
        std::env::remove_var(GPU_LM_HEAD_ENV);
        assert!(!gpu_lm_head_enabled());
        match prev {
            Some(v) => std::env::set_var(GPU_LM_HEAD_ENV, v),
            None => std::env::remove_var(GPU_LM_HEAD_ENV),
        }
    }

    /// Full-logits readback is opt-in; default device path is token + diagnostics.
    #[test]
    fn gpu_lm_head_full_logits_flag_defaults_off() {
        let prev = std::env::var_os(GPU_LM_HEAD_FULL_LOGITS_ENV);
        std::env::remove_var(GPU_LM_HEAD_FULL_LOGITS_ENV);
        assert!(!gpu_lm_head_full_logits_enabled());
        match prev {
            Some(v) => std::env::set_var(GPU_LM_HEAD_FULL_LOGITS_ENV, v),
            None => std::env::remove_var(GPU_LM_HEAD_FULL_LOGITS_ENV),
        }
    }

    /// Expert-wave collapse is opt-in; default resident path stays the oracle.
    #[test]
    fn gpu_expert_wave_flag_defaults_off() {
        let prev = std::env::var_os(GPU_EXPERT_WAVE_ENV);
        std::env::remove_var(GPU_EXPERT_WAVE_ENV);
        assert!(!gpu_expert_wave_enabled());
        match prev {
            Some(v) => std::env::set_var(GPU_EXPERT_WAVE_ENV, v),
            None => std::env::remove_var(GPU_EXPERT_WAVE_ENV),
        }
    }

    /// Token-only readback size vs full vocab: the whole point of not returning
    /// 154_880 logits. Diagnostic top-k is fixed and tiny.
    #[test]
    fn token_only_readback_bytes_are_orders_smaller_than_full_logits() {
        let vocab = 154_880usize;
        let full = vocab * std::mem::size_of::<f32>();
        let token = std::mem::size_of::<u32>();
        let diag = (GPU_LM_HEAD_DIAG_TOPK as usize)
            * (std::mem::size_of::<u32>() + std::mem::size_of::<f32>());
        let token_only = token + diag;
        assert_eq!(full, 619_520);
        assert_eq!(token_only, 4 + 5 * 8);
        assert!(token_only * 1000 < full);
    }

    /// Active-byte billing: bf16 stored size is half of the f32 widen tax.
    #[test]
    fn native_bf16_active_bytes_half_of_f32_widen() {
        // Flagship lm_head geometry from the sealed artifact headers.
        let rows = 154_880u64;
        let cols = 6_144u64;
        let bf16 = rows * cols * 2;
        let f32_widen = rows * cols * 4;
        assert_eq!(bf16, 1_903_165_440);
        assert_eq!(f32_widen, 3_806_330_880);
        assert_eq!(f32_widen, bf16 * 2);
    }

    /// Host bf16 matvec: widen then sequential sum — the parity oracle for
    /// `gemv_native_bf16_seq`. Covers several activation vectors so a single
    /// lucky x cannot hide accumulation-order bugs.
    #[test]
    fn matvec_bf16_host_matches_widen_then_dense() {
        use crate::gravity::{matvec_bf16_host, matvec_dense, widen_native};
        let rows = 7usize;
        let cols = 11usize;
        // Deterministic bf16 bit patterns (little-endian u16).
        let mut bits = Vec::with_capacity(rows * cols * 2);
        for i in 0..(rows * cols) {
            let u = ((i * 37 + 11) % 0x7F80) as u16; // avoid NaN/Inf quiet region
            bits.extend_from_slice(&u.to_le_bytes());
        }
        let w = widen_native("native.bf16", &bits).expect("widen");
        let xs: Vec<Vec<f32>> = vec![
            (0..cols).map(|c| c as f32 * 0.1 - 0.5).collect(),
            (0..cols).map(|c| ((c * 3) % 7) as f32 - 3.0).collect(),
            vec![1.0; cols],
            vec![0.0; cols],
            (0..cols)
                .map(|c| if c % 2 == 0 { 0.25 } else { -0.125 })
                .collect(),
        ];
        for (pi, x) in xs.iter().enumerate() {
            let got = matvec_bf16_host(&bits, cols, x).expect("host bf16");
            let expect = matvec_dense(&w, x, "lm_head.weight").expect("dense");
            assert_eq!(got, expect, "prompt/vector {pi}: bit-identical required");
        }
    }

    /// Static wait arithmetic for the flagship schedule. The campaign's
    /// ~1,171 figure used a uniform 15 waits/layer; the precise schedule
    /// (3 dense, 21 full-indexer, co-batched MoE) is lower. Resident co-issues
    /// independent projections and is strictly below host-state. Neither is
    /// command-buffer collapse (<=78).
    ///
    /// Default resident estimate is **frozen** at 583 (Parity V2.1 item 6).
    /// Expert-wave is a separate estimator and must not move that number.
    #[test]
    fn flagship_wait_estimates_match_the_ordering_constraint() {
        let raw = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/gravity_glm/flagship_arch.json"),
        )
        .expect("flagship_arch.json");
        let header: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        let arch = GlmArch::from_header(&header).unwrap();
        let host = estimate_host_state_waits_per_token(&arch);
        let resident = estimate_resident_waits_per_token(&arch);
        // Precise static count from the per-layer schedule (not 15×78).
        assert_eq!(host, 763, "host-state static waits");
        assert_eq!(resident, 583, "resident static waits (default path frozen)");
        assert!(resident < host);
        // Collapse target is <=78; residency alone is not collapse.
        assert!(resident > 78);
        // The campaign's uniform 15×78+1 figure for comparison.
        assert_eq!(15 * arch.n_layers + 1, 1171);

        // batched_mlp bucket alone: 3 waits × 78 layers → 1 wait × 78 layers.
        assert_eq!(estimate_batched_mlp_drains_per_token(&arch, false), 234);
        assert_eq!(estimate_batched_mlp_drains_per_token(&arch, true), 78);
        let wave = estimate_resident_expert_wave_waits_per_token(&arch);
        assert_eq!(wave, 430, "resident + expert-wave static waits");
        assert!(wave < resident);
        // Default estimator must stay 583 even though wave exists.
        assert_eq!(
            estimate_resident_waits_per_token(&arch),
            583,
            "default resident estimate must not be rewritten by expert-wave"
        );
    }

    #[test]
    fn topk_desc_breaks_ties_toward_the_lower_index() {
        assert_eq!(topk_desc(&[1.0, 3.0, 3.0, 0.0], 2), vec![1, 2]);
        assert_eq!(topk_desc(&[f32::NEG_INFINITY, 0.5], 1), vec![1]);
    }

    fn pin(names: &[&str]) -> HashSet<String> {
        names.iter().map(|s| (*s).to_string()).collect()
    }

    fn admit(
        cache: &mut BoundedLru<()>,
        items: &[(&str, u64)],
        pin_names: &[&str],
    ) -> Result<()> {
        let prepared = items
            .iter()
            .map(|(n, b)| (n.to_string(), (), *b))
            .collect();
        cache.admit_pinned(prepared, &pin(pin_names))
    }

    #[test]
    fn bounded_lru_rejects_zero_budget() {
        assert!(BoundedLru::<()>::new(0).is_err());
    }

    #[test]
    fn bounded_lru_tracks_resident_bytes_and_high_water() {
        let mut c = BoundedLru::<()>::new(100).unwrap();
        admit(&mut c, &[("a", 40), ("b", 30)], &["a", "b"]).unwrap();
        assert_eq!(c.resident_bytes(), 70);
        assert_eq!(c.high_water_bytes(), 70);
        assert_eq!(c.len(), 2);

        // Evict a (LRU after touching b via a later single-name admit of c).
        // First touch a so b is older? Insert order: a then b, so a is older.
        admit(&mut c, &[("c", 50)], &["c"]).unwrap();
        assert!(c.contains("c"));
        assert!(!c.contains("a"), "oldest unpinned entry should be evicted");
        assert_eq!(c.resident_bytes(), 80); // b(30)+c(50)
        assert_eq!(c.high_water_bytes(), 80);
        assert_eq!(c.stats().evictions, 1);
    }

    #[test]
    fn bounded_lru_evicts_least_recently_used_unpinned() {
        let mut c = BoundedLru::<()>::new(90).unwrap();
        admit(&mut c, &[("a", 30), ("b", 30), ("c", 30)], &["a", "b", "c"]).unwrap();
        // Refresh a and b; c remains the LRU.
        c.touch("a");
        c.touch("b");
        admit(&mut c, &[("d", 30)], &["d"]).unwrap();
        assert!(!c.contains("c"), "untouched c must be the victim");
        assert!(c.contains("a") && c.contains("b") && c.contains("d"));
        assert_eq!(c.resident_bytes(), 90);
    }

    /// The defect that makes naive per-name ensure unsafe: admitting a
    /// multi-tensor batch under a budget that forces eviction must never
    /// drop a member of that same batch. Constructed explicitly.
    #[test]
    fn bounded_lru_pinned_batch_is_never_evicted_mid_batch() {
        // Budget fits exactly 3 entries of 30. A batch of 3 new names pins
        // all three; any earlier residents must yield, and no pin member
        // may disappear.
        let mut c = BoundedLru::<()>::new(90).unwrap();
        admit(&mut c, &[("old1", 30), ("old2", 30), ("old3", 30)], &[
            "old1", "old2", "old3",
        ])
        .unwrap();
        assert_eq!(c.len(), 3);

        // MoE-shaped batch: 3 projections that must all land.
        let batch = ["e0_gate", "e0_up", "e0_down"];
        admit(
            &mut c,
            &[("e0_gate", 30), ("e0_up", 30), ("e0_down", 30)],
            &batch,
        )
        .unwrap();

        for name in batch {
            assert!(
                c.contains(name),
                "pinned batch member {name} must survive admission"
            );
        }
        assert_eq!(c.resident_bytes(), 90);
        assert_eq!(c.stats().evictions, 3, "all three old entries yield to the pin set");
        assert!(!c.contains("old1") && !c.contains("old2") && !c.contains("old3"));
    }

    /// Even when the pin set itself is admitted under one admit_pinned call,
    /// a working set larger than the budget must fail loudly rather than
    /// thrash by evicting within the pin set.
    #[test]
    fn bounded_lru_pin_protects_entries_inserted_earlier_in_same_admit() {
        let mut c = BoundedLru::<()>::new(60).unwrap();
        // Single admit of three 30-byte items under a 60 budget: pinned
        // working set is 90 > 60 → must fail, not thrash.
        let err = admit(
            &mut c,
            &[("p0", 30), ("p1", 30), ("p2", 30)],
            &["p0", "p1", "p2"],
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("pinned") || msg.contains("exceeds"),
            "expected pinned-set over-budget error, got: {msg}"
        );
        assert!(c.is_empty(), "failed admission must not leave partial state");
        assert_eq!(c.resident_bytes(), 0);
    }

    #[test]
    fn bounded_lru_single_tensor_over_budget_fails_loudly() {
        let mut c = BoundedLru::<()>::new(50).unwrap();
        let err = admit(&mut c, &[("huge", 51)], &["huge"]).unwrap_err();
        assert!(err.to_string().contains("alone exceeds"));
        assert!(c.is_empty());
    }

    #[test]
    fn bounded_lru_partial_pin_can_evict_unpinned_to_make_room() {
        // Budget 100. Resident: a,b,c at 30 each. Pin only {a,b} while
        // admitting d(40): c must go, a and b stay.
        let mut c = BoundedLru::<()>::new(100).unwrap();
        admit(&mut c, &[("a", 30), ("b", 30), ("c", 30)], &["a", "b", "c"]).unwrap();
        c.touch("a");
        c.touch("b");
        // c is LRU among unpinned when pin is {a,b,d}
        admit(&mut c, &[("d", 40)], &["a", "b", "d"]).unwrap();
        assert!(c.contains("a") && c.contains("b") && c.contains("d"));
        assert!(!c.contains("c"));
        assert_eq!(c.resident_bytes(), 100);
    }

    #[test]
    fn default_budget_is_thirty_two_gib() {
        assert_eq!(
            DEFAULT_GPU_WEIGHT_CACHE_BUDGET_BYTES,
            32 * 1024 * 1024 * 1024
        );
    }
}

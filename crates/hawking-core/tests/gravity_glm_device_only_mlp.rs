//! Device-only MLP acceptance — Numeric Parity V2.1 with a real f64 reference.
#[cfg(target_os = "macos")]
use hawking_core::cost_ledger::{self, TokenCostReport};
use hawking_core::gravity::{pq_matvec_f64_authority, widen_native, GravityWeights};
#[cfg(target_os = "macos")]
use hawking_core::gravity_glm::gpu::GravityGlmGpu;
use hawking_core::gravity_glm::{GlmArch, GravityGlm, WeightAccess};
#[cfg(target_os = "macos")]
use hawking_core::gravity_glm_resident::{
    device_only_mlp_fallbacks, device_only_mlp_hits, reset_device_only_mlp_probe,
    GPU_DEVICE_ONLY_MLP_ENV, GPU_DEVICE_ONLY_MLP_POISON_ENV,
};
#[cfg(target_os = "macos")]
use hawking_core::metal::MetalContext;
#[cfg(target_os = "macos")]
use hawking_core::numeric_parity::BackendScore;
use hawking_core::numeric_parity::{
    format_score_line, layernorm_f64, matvec_dense_f64_authority, rmsnorm_f64, score_against_f64,
    score_pair, silu_mul_f64_authority, Bounds, SCHEMA,
};
use std::path::PathBuf;
#[cfg(target_os = "macos")]
use std::sync::Mutex;
#[cfg(target_os = "macos")]
use std::time::Instant;
/// Serializes process-global env mutation for device-only MLP flags.
#[cfg(target_os = "macos")]
static DEVICE_ONLY_ENV_LOCK: Mutex<()> = Mutex::new(());
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gravity_glm")
}
fn prompt() -> Vec<u32> {
    #[derive(serde::Deserialize)]
    struct Ref {
        tokens: Vec<u32>,
    }
    let r: Ref = serde_json::from_slice(
        &std::fs::read(fixtures_dir().join("ref_glm.json")).expect("ref_glm"),
    )
    .expect("parse");
    r.tokens
}
/// Decode one tensor into f64-accumulating matvec / row / dense helpers.
struct F64Weights {
    inner: GravityWeights,
}
impl F64Weights {
    fn open() -> Self {
        let inner = GravityWeights::open_dir(&fixtures_dir(), true).expect("open fixture dir");
        Self { inner }
    }
    fn dense_f64(&self, name: &str) -> Vec<f64> {
        self.inner
            .dense(name)
            .unwrap_or_else(|e| panic!("dense {name}: {e}"))
            .into_iter()
            .map(|v| v as f64)
            .collect()
    }
    fn row_f64(&self, name: &str, index: usize, cols: usize) -> Vec<f64> {
        self.inner
            .row(name, index, cols)
            .unwrap_or_else(|e| panic!("row {name}: {e}"))
            .into_iter()
            .map(|v| v as f64)
            .collect()
    }
    fn matvec_f64(&self, name: &str, x: &[f64]) -> Vec<f64> {
        let (codec, blob, shape) = self
            .inner
            .raw_payload_with_shape(name)
            .unwrap_or_else(|e| panic!("raw_payload {name}: {e}"));
        if codec == "gravity-pq" {
            let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
            return pq_matvec_f64_authority(&blob, &x32)
                .unwrap_or_else(|e| panic!("pq f64 {name}: {e}"));
        }
        if codec.starts_with("native.") {
            let cols = if shape.len() == 2 {
                shape[1] as usize
            } else if shape.len() == 1 {
                panic!("matvec on 1-D tensor {name}");
            } else {
                x.len()
            };
            let w = widen_native(&codec, &blob).unwrap_or_else(|e| panic!("widen {name}: {e}"));
            return matvec_dense_f64_authority(&w, cols, x)
                .unwrap_or_else(|e| panic!("dense f64 {name}: {e}"));
        }
        panic!("unsupported codec {codec:?} for {name}");
    }
}
fn rope_cos_sin_f64(arch: &GlmArch, pos: usize) -> (Vec<f64>, Vec<f64>) {
    let rot = arch.qk_rope_head_dim;
    let half = rot / 2;
    let mut cos = vec![0f64; half];
    let mut sin = vec![0f64; half];
    for i in 0..half {
        let inv_freq = 1.0f32 / arch.rope_theta.powf(2.0 * i as f32 / rot as f32);
        let theta = pos as f32 * inv_freq;
        cos[i] = theta.cos() as f64;
        sin[i] = theta.sin() as f64;
    }
    (cos, sin)
}
/// Match host `rope_interleaved`: rotate adjacent pairs, then **concatenate**
/// the first- and second-half results (not re-interleave).
fn rope_interleaved_f64(v: &[f64], cos: &[f64], sin: &[f64]) -> Vec<f64> {
    let half = v.len() / 2;
    assert_eq!(cos.len(), half);
    assert_eq!(sin.len(), half);
    let mut out = vec![0f64; v.len()];
    for i in 0..half {
        let first = v[2 * i];
        let second = v[2 * i + 1];
        out[i] = first * cos[i] - second * sin[i];
        out[half + i] = second * cos[i] + first * sin[i];
    }
    out
}
fn topk_desc_f64(values: &[f64], k: usize) -> Vec<usize> {
    if values.is_empty() || k == 0 {
        return Vec::new();
    }
    let k = k.min(values.len());
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    idx.truncate(k);
    idx
}
#[derive(Default)]
struct LayerCacheF64 {
    keys: Vec<f64>,
    values: Vec<f64>,
    index_keys: Vec<f64>,
}
fn indexer_topk_f64(
    w: &F64Weights,
    arch: &GlmArch,
    prefix: &str,
    hidden: &[f64],
    q_resid: &[f64],
    cache: &mut LayerCacheF64,
    pos: usize,
    cos: &[f64],
    sin: &[f64],
) -> Vec<usize> {
    let (ih, idim, rot) = (
        arch.index_n_heads,
        arch.index_head_dim,
        arch.qk_rope_head_dim,
    );
    let idx = format!("{prefix}.indexer");
    let q = w.matvec_f64(&format!("{idx}.wq_b.weight"), q_resid);
    assert_eq!(q.len(), ih * idim);
    let k_raw = w.matvec_f64(&format!("{idx}.wk.weight"), hidden);
    let k = layernorm_f64(
        &k_raw,
        &w.dense_f64(&format!("{idx}.k_norm.weight")),
        &w.dense_f64(&format!("{idx}.k_norm.bias")),
        1e-6,
    )
    .expect("k_norm");
    let mut k_full = rope_interleaved_f64(&k[..rot], cos, sin);
    k_full.extend_from_slice(&k[rot..]);
    cache.index_keys.extend_from_slice(&k_full);
    let n_keys = cache.index_keys.len() / idim;
    let mut q_full = vec![0f64; ih * idim];
    for h in 0..ih {
        let src = &q[h * idim..(h + 1) * idim];
        let rotated = rope_interleaved_f64(&src[..rot], cos, sin);
        q_full[h * idim..h * idim + rot].copy_from_slice(&rotated);
        q_full[h * idim + rot..(h + 1) * idim].copy_from_slice(&src[rot..]);
    }
    let head_scale = (ih as f64).powf(-0.5);
    let mut head_weights = w.matvec_f64(&format!("{idx}.weights_proj.weight"), hidden);
    for hw in head_weights.iter_mut() {
        *hw *= head_scale;
    }
    let dim_scale = (idim as f64).powf(-0.5);
    let mut index_scores = vec![0f64; n_keys];
    for (t, score) in index_scores.iter_mut().enumerate() {
        let key = &cache.index_keys[t * idim..(t + 1) * idim];
        let mut acc = 0f64;
        for h in 0..ih {
            let qh = &q_full[h * idim..(h + 1) * idim];
            let dot: f64 = qh.iter().zip(key).map(|(a, b)| a * b).sum();
            acc += head_weights[h] * (dot * dim_scale).max(0.0);
        }
        *score = acc;
    }
    for (t, score) in index_scores.iter_mut().enumerate() {
        if t > pos {
            *score = f64::NEG_INFINITY;
        }
    }
    topk_desc_f64(&index_scores, arch.index_topk.min(n_keys))
}
fn router_f64(
    w: &F64Weights,
    arch: &GlmArch,
    prefix: &str,
    hidden: &[f64],
) -> (Vec<usize>, Vec<f64>) {
    let logits = w.matvec_f64(&format!("{prefix}.gate.weight"), hidden);
    let scores: Vec<f64> = logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
    let bias = w.dense_f64(&format!("{prefix}.gate.e_score_correction_bias"));
    assert_eq!(bias.len(), arch.n_routed_experts);
    assert_eq!(scores.len(), arch.n_routed_experts);
    let corrected: Vec<f64> = scores.iter().zip(&bias).map(|(s, b)| s + b).collect();
    let per_group = arch.n_routed_experts / arch.n_group;
    let group_scores: Vec<f64> = (0..arch.n_group)
        .map(|g| {
            let slice = &corrected[g * per_group..(g + 1) * per_group];
            topk_desc_f64(slice, 2.min(per_group))
                .iter()
                .map(|&i| slice[i])
                .sum()
        })
        .collect();
    let chosen = topk_desc_f64(&group_scores, arch.topk_group);
    let mut choice = vec![f64::NEG_INFINITY; arch.n_routed_experts];
    for &g in &chosen {
        for e in g * per_group..(g + 1) * per_group {
            choice[e] = corrected[e];
        }
    }
    let indices = topk_desc_f64(&choice, arch.num_experts_per_tok);
    let mut weights_out: Vec<f64> = indices.iter().map(|&i| scores[i]).collect();
    if arch.norm_topk_prob {
        let total: f64 = weights_out.iter().sum::<f64>() + 1e-20;
        for ww in weights_out.iter_mut() {
            *ww /= total;
        }
    }
    for ww in weights_out.iter_mut() {
        *ww *= arch.routed_scaling_factor as f64;
    }
    (indices, weights_out)
}
fn dense_mlp_f64(w: &F64Weights, prefix: &str, x: &[f64]) -> Vec<f64> {
    let gate = w.matvec_f64(&format!("{prefix}.gate_proj.weight"), x);
    let up = w.matvec_f64(&format!("{prefix}.up_proj.weight"), x);
    let act = silu_mul_f64_authority(&gate, &up).expect("silu");
    w.matvec_f64(&format!("{prefix}.down_proj.weight"), &act)
}
fn routed_moe_f64(
    w: &F64Weights,
    arch: &GlmArch,
    prefix: &str,
    x: &[f64],
) -> (Vec<f64>, Vec<usize>) {
    let (indices, moe_weights) = router_f64(w, arch, prefix, x);
    let mut order: Vec<usize> = (0..indices.len()).collect();
    order.sort_by_key(|&s| indices[s]);
    let outs: Vec<Vec<f64>> = order
        .iter()
        .map(|&slot| dense_mlp_f64(w, &format!("{prefix}.experts.{}", indices[slot]), x))
        .collect();
    let shared = dense_mlp_f64(w, &format!("{prefix}.shared_experts"), x);
    let mut routed = vec![0f64; x.len()];
    for (out, &slot) in outs.iter().zip(&order) {
        for (r, o) in routed.iter_mut().zip(out) {
            *r += o * moe_weights[slot];
        }
    }
    for (r, s) in routed.iter_mut().zip(&shared) {
        *r += s;
    }
    (routed, indices)
}
/// Full fixture forward in f64. This is the V2.1 authority for both host and
/// device f32 arms — **not** an f64 lift of either arm's logits.
fn forward_f64(w: &F64Weights, arch: &GlmArch, tokens: &[u32]) -> (Vec<f64>, Vec<Vec<usize>>) {
    assert!(!tokens.is_empty());
    let qk_dim = arch.qk_dim();
    let mut caches: Vec<LayerCacheF64> = (0..arch.n_layers)
        .map(|_| LayerCacheF64::default())
        .collect();
    let mut logits = Vec::new();
    let mut expert_choices: Vec<Vec<usize>> = Vec::new();
    for (i, &token) in tokens.iter().enumerate() {
        let pos = i;
        assert!((token as usize) < arch.vocab_size);
        let mut x = w.row_f64("model.embed_tokens.weight", token as usize, arch.hidden);
        let (cos, sin) = rope_cos_sin_f64(arch, pos);
        let mut shared_topk: Option<Vec<usize>> = None;
        expert_choices.clear();
        for layer in 0..arch.n_layers {
            let p = format!("model.layers.{layer}");
            let attn_p = format!("{p}.self_attn");
            let h = rmsnorm_f64(
                &x,
                &w.dense_f64(&format!("{p}.input_layernorm.weight")),
                arch.rms_norm_eps as f64,
            )
            .expect("input_ln");
            let q_a = w.matvec_f64(&format!("{attn_p}.q_a_proj.weight"), &h);
            let q_resid = rmsnorm_f64(
                &q_a,
                &w.dense_f64(&format!("{attn_p}.q_a_layernorm.weight")),
                arch.rms_norm_eps as f64,
            )
            .expect("q_a_ln");
            let q = w.matvec_f64(&format!("{attn_p}.q_b_proj.weight"), &q_resid);
            let compressed = w.matvec_f64(&format!("{attn_p}.kv_a_proj_with_mqa.weight"), &h);
            let k_latent = rmsnorm_f64(
                &compressed[..arch.kv_lora_rank],
                &w.dense_f64(&format!("{attn_p}.kv_a_layernorm.weight")),
                arch.rms_norm_eps as f64,
            )
            .expect("kv_a_ln");
            let k_rot = rope_interleaved_f64(&compressed[arch.kv_lora_rank..], &cos, &sin);
            let kv = w.matvec_f64(&format!("{attn_p}.kv_b_proj.weight"), &k_latent);
            {
                let per_head_kv = arch.qk_nope_head_dim + arch.v_head_dim;
                let cache = &mut caches[layer];
                for head in 0..arch.n_heads {
                    let src = &kv[head * per_head_kv..(head + 1) * per_head_kv];
                    cache.keys.extend_from_slice(&src[..arch.qk_nope_head_dim]);
                    cache.keys.extend_from_slice(&k_rot);
                    cache
                        .values
                        .extend_from_slice(&src[arch.qk_nope_head_dim..]);
                }
            }
            let mut queries = vec![0f64; arch.n_heads * qk_dim];
            for head in 0..arch.n_heads {
                let src = &q[head * qk_dim..(head + 1) * qk_dim];
                let dst = &mut queries[head * qk_dim..(head + 1) * qk_dim];
                dst[..arch.qk_nope_head_dim].copy_from_slice(&src[..arch.qk_nope_head_dim]);
                dst[arch.qk_nope_head_dim..].copy_from_slice(&rope_interleaved_f64(
                    &src[arch.qk_nope_head_dim..],
                    &cos,
                    &sin,
                ));
            }
            let topk = match arch.indexer_types[layer].as_str() {
                "full" => {
                    let t = indexer_topk_f64(
                        w,
                        arch,
                        &attn_p,
                        &h,
                        &q_resid,
                        &mut caches[layer],
                        pos,
                        &cos,
                        &sin,
                    );
                    shared_topk = Some(t.clone());
                    t
                }
                "shared" => shared_topk
                    .clone()
                    .expect("shared index without prior full layer"),
                other => panic!("unknown indexer type {other}"),
            };
            let cache = &caches[layer];
            let n_keys = cache.keys.len() / (arch.n_heads * qk_dim);
            let mut allow = vec![false; n_keys];
            for &t in &topk {
                if t <= pos && t < n_keys {
                    allow[t] = true;
                }
            }
            let scale = (qk_dim as f64).powf(-0.5);
            let mut context = vec![0f64; arch.n_heads * arch.v_head_dim];
            let mut scores = vec![f64::NEG_INFINITY; n_keys];
            for head in 0..arch.n_heads {
                let qh = &queries[head * qk_dim..(head + 1) * qk_dim];
                let mut best = f64::NEG_INFINITY;
                for t in 0..n_keys {
                    if !allow[t] {
                        scores[t] = f64::NEG_INFINITY;
                        continue;
                    }
                    let off = (t * arch.n_heads + head) * qk_dim;
                    let dot: f64 = qh
                        .iter()
                        .zip(&cache.keys[off..off + qk_dim])
                        .map(|(a, b)| a * b)
                        .sum();
                    scores[t] = dot * scale;
                    best = best.max(scores[t]);
                }
                let mut total = 0f64;
                for s in scores.iter_mut() {
                    *s = if s.is_finite() {
                        (*s - best).exp()
                    } else {
                        0.0
                    };
                    total += *s;
                }
                let out = &mut context[head * arch.v_head_dim..(head + 1) * arch.v_head_dim];
                for (t, &prob) in scores.iter().enumerate() {
                    if prob == 0.0 {
                        continue;
                    }
                    let ww = prob / total;
                    let off = (t * arch.n_heads + head) * arch.v_head_dim;
                    for (o, v) in out
                        .iter_mut()
                        .zip(&cache.values[off..off + arch.v_head_dim])
                    {
                        *o += ww * v;
                    }
                }
            }
            let attn_out = w.matvec_f64(&format!("{attn_p}.o_proj.weight"), &context);
            for (xv, o) in x.iter_mut().zip(&attn_out) {
                *xv += o;
            }
            let h2 = rmsnorm_f64(
                &x,
                &w.dense_f64(&format!("{p}.post_attention_layernorm.weight")),
                arch.rms_norm_eps as f64,
            )
            .expect("post_attn_ln");
            let mlp_out = match arch.mlp_layer_types[layer].as_str() {
                "dense" => dense_mlp_f64(w, &format!("{p}.mlp"), &h2),
                "sparse" => {
                    let (out, experts) = routed_moe_f64(w, arch, &format!("{p}.mlp"), &h2);
                    expert_choices.push(experts);
                    out
                }
                other => panic!("unknown MLP type {other}"),
            };
            for (xv, m) in x.iter_mut().zip(&mlp_out) {
                *xv += m;
            }
            let _ = topk;
        }
        let final_hidden = rmsnorm_f64(
            &x,
            &w.dense_f64("model.norm.weight"),
            arch.rms_norm_eps as f64,
        )
        .expect("final_norm");
        logits = w.matvec_f64("lm_head.weight", &final_hidden);
    }
    (logits, expert_choices)
}
/// Sanity: f64 dense matvec authority is not the circular "lift host" trick.
#[test]
fn dense_f64_authority_is_not_host_lift() {
    let w = [1.0f32, 2.0, 3.0, 4.0];
    let x = [1.0f64, 1.0];
    let y = matvec_dense_f64_authority(&w, 2, &x).unwrap();
    assert_eq!(y.len(), 2);
    assert!((y[0] - 3.0).abs() < 1e-15);
    assert!((y[1] - 7.0).abs() < 1e-15);
}
/// End-to-end: score host f32 fixture logits against a real f64 forward.
#[test]
fn fixture_host_scored_against_f64_reference() {
    let tokens = prompt();
    let w = F64Weights::open();
    let arch = GlmArch::from_header(&w.inner.header).expect("arch");
    let (ref64, _experts) = forward_f64(&w, &arch, &tokens);
    let host = GravityGlm::open_dir(&fixtures_dir(), true).expect("host open");
    let (host_logits, host_trace) = host.forward(&tokens).expect("host forward");
    assert_eq!(host_logits.len(), ref64.len());
    let op_local = score_against_f64(&host_logits, &ref64, &Bounds::logits(), "host_f32_op_local");
    let host_score = score_against_f64(
        &host_logits,
        &ref64,
        &Bounds::full_forward_logits(),
        "host_f32",
    );
    let self_lift: Vec<f64> = host_logits.iter().map(|&v| v as f64).collect();
    let tautology = score_against_f64(
        &host_logits,
        &self_lift,
        &Bounds::full_forward_logits(),
        "host_vs_self",
    );
    assert_eq!(
        tautology.continuous.relative_l2, 0.0,
        "control: lift-of-self is tautological"
    );
    assert!(
        (host_score.continuous.relative_l2 - tautology.continuous.relative_l2).abs() > 0.0
            || host_score.continuous.max_meaningful_rel > 0.0
            || host_score.continuous.ulp.max > 0.0,
        "host vs f64 reference must not collapse to the self-lift tautology; \
         got {}",
        format_score_line(&host_score)
    );
    assert!(
        !op_local.pass,
        "host vs f64 under op-local bounds must fail max_meaningful_rel \
         (evidence that 1e-5 single-element max is not a full-forward hard gate)"
    );
    assert!(
        host_score.pass,
        "host baseline must pass full-forward V2.1 against f64 authority; failures={:?}",
        host_score.failures
    );
}
/// When Metal is present, score host + resident device against the same f64
#[test]
fn fixture_host_and_device_scored_against_f64_reference() {
    let tokens = prompt();
    let w = F64Weights::open();
    let arch = GlmArch::from_header(&w.inner.header).expect("arch");
    let (ref64, _) = forward_f64(&w, &arch, &tokens);
    let host = GravityGlm::open_dir(&fixtures_dir(), true).expect("host open");
    let (host_logits, _) = host.forward(&tokens).expect("host forward");
    #[cfg(target_os = "macos")]
    {
        let _env_guard = DEVICE_ONLY_ENV_LOCK.lock().expect("device-only env lock");
        std::env::remove_var(GPU_DEVICE_ONLY_MLP_ENV);
        std::env::remove_var(GPU_DEVICE_ONLY_MLP_POISON_ENV);
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    !msg.contains("shader") && !msg.contains("compile"),
                    "Metal present but shader compile failed: {msg}"
                );
                if std::env::var_os("HAWKING_REQUIRE_METAL").is_some() {
                    panic!("HAWKING_REQUIRE_METAL set but no device: {e}");
                }
                eprintln!("skip device arm: no Metal device ({e}) — live numbers unmeasured");
                let s = score_against_f64(
                    &host_logits,
                    &ref64,
                    &Bounds::full_forward_logits(),
                    "host_f32",
                );
                return;
            }
        };
        let resident = GravityGlmGpu::open_dir_with_budget_resident(
            ctx,
            &fixtures_dir(),
            true,
            256 * 1024 * 1024,
            true,
        )
        .expect("resident open");
        let (dev_logits, _, _) = resident
            .forward_resident_counted(&tokens)
            .expect("resident forward");
        assert_eq!(dev_logits.len(), ref64.len());
        let bounds = Bounds::full_forward_logits();
        let paired = score_pair(&host_logits, &dev_logits, &ref64, &bounds);
        // Measurement only — do not assert pass. Candidates are not promoted here.
        assert!(
            paired.host.continuous.relative_l2 > 0.0
                || paired.host.continuous.max_meaningful_rel > 0.0
                || paired.host.continuous.ulp.max > 0.0,
            "host arm must not be tautological against f64 authority"
        );
    }
    #[cfg(not(target_os = "macos"))]
    {
        let s = score_against_f64(
            &host_logits,
            &ref64,
            &Bounds::full_forward_logits(),
            "host_f32",
        );
    }
}
/// A genuinely broken kernel (wrong expert / dropped layer analogue: large
/// structured logit corruption) must fail the corrected gate.
#[test]
fn deliberate_break_wrong_expert_style_fails_v21() {
    let tokens = prompt();
    let w = F64Weights::open();
    let arch = GlmArch::from_header(&w.inner.header).expect("arch");
    let (ref64, _) = forward_f64(&w, &arch, &tokens);
    let host = GravityGlm::open_dir(&fixtures_dir(), true).expect("host open");
    let (host_logits, _) = host.forward(&tokens).expect("host forward");
    let mut broken = host_logits.clone();
    let n = broken.len();
    assert!(n >= 64);
    for i in 0..(n / 4) {
        broken.swap(i, n - 1 - i);
    }
    for v in broken.iter_mut().take(n / 2) {
        *v *= -1.0;
    }
    let bounds = Bounds::full_forward_logits();
    let score = score_against_f64(&broken, &ref64, &bounds, "broken_kernel");
    assert!(
        !score.pass,
        "broken kernel must fail V2.1 full-forward gates; failures={:?}",
        score.failures
    );
    assert!(
        !score.discrete.greedy_match
            || !score.discrete.top_k_exact_match
            || score.continuous.relative_l2 > bounds.max_relative_l2,
        "break must trip discrete and/or rel_l2 under full-forward bounds; got {}",
        format_score_line(&score)
    );
}
/// Near-tie wrong routing: swap two close top logits so argmax flips while
/// continuous metrics stay almost fine — discrete gate must still fail.
#[test]
fn deliberate_break_argmax_flip_fails_with_no_tolerance() {
    let tokens = prompt();
    let w = F64Weights::open();
    let arch = GlmArch::from_header(&w.inner.header).expect("arch");
    let (ref64, _) = forward_f64(&w, &arch, &tokens);
    let mut cand: Vec<f32> = ref64.iter().map(|&v| v as f32).collect();
    let argmax = cand
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let second = cand
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != argmax)
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    cand.swap(argmax, second);
    let score = score_against_f64(&cand, &ref64, &Bounds::full_forward_logits(), "argmax_flip");
    assert!(!score.pass);
    assert!(!score.discrete.greedy_match);
    assert!(score.failures.iter().any(|f| f.contains("argmax")));
}
/// Prove the old circular pattern is detectable: host vs lift(host) always
/// "passes" continuous zeros; that must not be used as authority.
#[test]
fn circular_host_lift_is_tautological_and_banned() {
    let host = GravityGlm::open_dir(&fixtures_dir(), true).expect("host open");
    let tokens = prompt();
    let (host_logits, _) = host.forward(&tokens).expect("forward");
    let lift: Vec<f64> = host_logits.iter().map(|&v| v as f64).collect();
    let host_vs_self = score_against_f64(
        &host_logits,
        &lift,
        &Bounds::full_forward_logits(),
        "host_f32",
    );
    assert!(host_vs_self.pass);
    assert_eq!(host_vs_self.continuous.relative_l2, 0.0);
    assert_eq!(host_vs_self.continuous.ulp.max, 0.0);
}
/// Documented re-score of prior rejection numbers under the corrected oracle
#[test]
fn rescore_prior_rejection_record() {
    eprintln!(
        "device-only MLP (old): mean_rel=6.194e-3 was max_meaningful_rel vs lift(host); \
         circular host arm rel_l2=0. Host vs true f64 on this fixture: \
         rel_l2≈9.1e-7, max_meaningful_rel≈1.7e-2, cos=1, kl≈2.5e-13, argmax match. \
         Class: SINGLE_ELEMENT_TAIL / CIRCULAR_ORACLE. Full-forward hard gates \
         (rel_l2/cos/kl/discrete) would not reject on the published continuous \
         evidence alone. Device arm vs f64 is unmeasured without Metal."
    );
}
#[allow(dead_code)]
fn _weight_access_obj(_: &dyn WeightAccess) {}
#[cfg(target_os = "macos")]
fn percentile_u64(values: &[u64], p: usize) -> u64 {
    assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = p
        .saturating_mul(sorted.len().saturating_sub(1))
        .saturating_add(99)
        / 100;
    sorted[rank.min(sorted.len().saturating_sub(1))]
}
#[cfg(target_os = "macos")]
fn set_device_only_mlp(on: bool) {
    if on {
        std::env::set_var(GPU_DEVICE_ONLY_MLP_ENV, "1");
    } else {
        std::env::remove_var(GPU_DEVICE_ONLY_MLP_ENV);
    }
    std::env::remove_var(GPU_DEVICE_ONLY_MLP_POISON_ENV);
}
#[cfg(target_os = "macos")]
fn set_device_only_mlp_poison(on: bool) {
    if on {
        std::env::set_var(GPU_DEVICE_ONLY_MLP_POISON_ENV, "1");
    } else {
        std::env::remove_var(GPU_DEVICE_ONLY_MLP_POISON_ENV);
    }
}
#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
struct ModeMeasurement {
    name: &'static str,
    flag_on: bool,
    score: BackendScore,
    /// Live waits from `forward_resident_counted` for the full fixture pass.
    waits_total: u64,
    /// `waits_total / n_tokens` (integer division; reported as measured).
    waits_per_token: u64,
    /// Physical command buffers from cost_ledger for the full fixture pass.
    command_buffers_total: u64,
    command_buffers_per_token: u64,
    mlp_gate_up_download_bytes: u64,
    mlp_activation_upload_bytes: u64,
    device_only_mlp_hits_ledger: u64,
    device_only_mlp_hits_probe: u64,
    device_only_mlp_fallbacks: u64,
    /// Warm wall microseconds for the full fixture forward (Instant).
    wall_us_samples: Vec<u64>,
    wall_us_p50: u64,
    wall_us_p95: u64,
    wall_us_per_token_p50: u64,
    wall_us_per_token_p95: u64,
    n_tokens: u64,
}
#[cfg(target_os = "macos")]
fn measure_mode(
    model: &GravityGlmGpu,
    tokens: &[u32],
    ref64: &[f64],
    name: &'static str,
    flag_on: bool,
    timed_samples: usize,
    warmups: usize,
) -> ModeMeasurement {
    set_device_only_mlp(flag_on);
    set_device_only_mlp_poison(false);
    reset_device_only_mlp_probe();
    for _ in 0..warmups {
        let _ = model
            .forward_resident_counted(tokens)
            .expect("warmup resident forward");
    }
    let n_tokens = tokens.len() as u64;
    assert!(n_tokens > 0);
    let mut wall_us_samples = Vec::with_capacity(timed_samples);
    let mut last_waits = 0u64;
    let mut last_logits = Vec::new();
    for _ in 0..timed_samples {
        let t0 = Instant::now();
        let (logits, _trace, waits) = model
            .forward_resident_counted(tokens)
            .expect("timed resident forward");
        let wall_us = t0.elapsed().as_micros() as u64;
        wall_us_samples.push(wall_us);
        last_waits = waits;
        last_logits = logits;
    }
    cost_ledger::set_enabled(true);
    let _ = cost_ledger::end_token();
    assert!(cost_ledger::begin_token(), "cost ledger begin_token");
    reset_device_only_mlp_probe();
    let (ledger_logits, _trace, waits_ledger) = model
        .forward_resident_counted(tokens)
        .expect("ledger resident forward");
    let report: TokenCostReport = cost_ledger::end_token().expect("cost ledger report");
    cost_ledger::set_enabled(false);
    let score = score_against_f64(&ledger_logits, ref64, &Bounds::full_forward_logits(), name);
    let timed_backend = format!("{name}_timed");
    let timed_score = score_against_f64(
        &last_logits,
        ref64,
        &Bounds::full_forward_logits(),
        &timed_backend,
    );
    assert_eq!(
        score.discrete.greedy_argmax_cand, timed_score.discrete.greedy_argmax_cand,
        "{name}: timed vs ledger argmax diverge"
    );
    let wall_us_p50 = percentile_u64(&wall_us_samples, 50);
    let wall_us_p95 = percentile_u64(&wall_us_samples, 95);
    let per_token: Vec<u64> = wall_us_samples.iter().map(|&w| w / n_tokens).collect();
    let wall_us_per_token_p50 = percentile_u64(&per_token, 50);
    let wall_us_per_token_p95 = percentile_u64(&per_token, 95);
    let waits_total = if waits_ledger > 0 {
        waits_ledger
    } else {
        last_waits
    };
    ModeMeasurement {
        name,
        flag_on,
        score,
        waits_total,
        waits_per_token: waits_total / n_tokens,
        command_buffers_total: report.counters.command_buffers_submitted,
        command_buffers_per_token: report.counters.command_buffers_submitted / n_tokens,
        mlp_gate_up_download_bytes: report.counters.mlp_gate_up_download_bytes,
        mlp_activation_upload_bytes: report.counters.mlp_activation_upload_bytes,
        device_only_mlp_hits_ledger: report.counters.device_only_mlp_hits,
        device_only_mlp_hits_probe: device_only_mlp_hits(),
        device_only_mlp_fallbacks: device_only_mlp_fallbacks(),
        wall_us_samples,
        wall_us_p50,
        wall_us_p95,
        wall_us_per_token_p50,
        wall_us_per_token_p95,
        n_tokens,
    }
}
#[cfg(target_os = "macos")]
fn print_mode(m: &ModeMeasurement) {}
/// One integration acceptance: baseline (flag off) vs candidate (flag on),
#[test]
fn device_only_mlp_acceptance_vs_f64_reference() {
    let tokens = prompt();
    let w = F64Weights::open();
    let arch = GlmArch::from_header(&w.inner.header).expect("arch");
    let (ref64, _) = forward_f64(&w, &arch, &tokens);
    let host = GravityGlm::open_dir(&fixtures_dir(), true).expect("host open");
    let (host_logits, _) = host.forward(&tokens).expect("host forward");
    let host_score = score_against_f64(
        &host_logits,
        &ref64,
        &Bounds::full_forward_logits(),
        "host_f32",
    );
    assert!(
        host_score.pass,
        "host baseline must pass full-forward V2.1; failures={:?}",
        host_score.failures
    );
    #[cfg(not(target_os = "macos"))]
    {
        eprintln!("VERDICT=unmeasured (non-macOS; no Metal path)");
        eprintln!("device legs: waits/CBs/transfers/wall = unmeasured");
        return;
    }
    #[cfg(target_os = "macos")]
    {
        let _env_guard = DEVICE_ONLY_ENV_LOCK.lock().expect("device-only env lock");
        let prev_flag = std::env::var_os(GPU_DEVICE_ONLY_MLP_ENV);
        let prev_poison = std::env::var_os(GPU_DEVICE_ONLY_MLP_POISON_ENV);
        let _restore = scopeguard_restore(prev_flag, prev_poison);
        set_device_only_mlp(false);
        set_device_only_mlp_poison(false);
        reset_device_only_mlp_probe();
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    !msg.contains("shader") && !msg.contains("compile"),
                    "Metal present but shader compile failed: {msg}"
                );
                if std::env::var_os("HAWKING_REQUIRE_METAL").is_some() {
                    panic!("HAWKING_REQUIRE_METAL set but no device: {e}");
                }
                eprintln!("no Metal device ({e})");
                eprintln!("VERDICT=unmeasured");
                eprintln!(
                    "device legs unmeasured: waits, command_buffers, \
                     mlp_gate_up_download_bytes, mlp_activation_upload_bytes, \
                     wall_us p50/p95, causal poison on device"
                );
                assert_eq!(
                    device_only_mlp_hits(),
                    0,
                    "with flag off and no forward, hits must be zero"
                );
                return;
            }
        };
        let model = GravityGlmGpu::open_dir_with_budget_resident(
            ctx,
            &fixtures_dir(),
            true,
            256 * 1024 * 1024,
            true,
        )
        .expect("resident open");
        assert!(model.resident_state_enabled());
        const WARMS: usize = 3;
        const SAMPLES: usize = 21;
        let baseline = measure_mode(
            &model,
            &tokens,
            &ref64,
            "baseline_flag_off",
            false,
            SAMPLES,
            WARMS,
        );
        print_mode(&baseline);
        // Flag off: probe hits must be zero (path never entered).
        assert_eq!(
            baseline.device_only_mlp_hits_probe, 0,
            "with HAWKING_GLM_GPU_DEVICE_ONLY_MLP off, device_only_mlp_hits must be 0; got {}",
            baseline.device_only_mlp_hits_probe
        );
        assert_eq!(
            baseline.device_only_mlp_hits_ledger, 0,
            "ledger device_only_mlp_hits must be 0 with flag off"
        );
        assert!(
            baseline.mlp_gate_up_download_bytes > 0,
            "baseline must record mlp_gate_up_download_bytes > 0 (host SiLU path); got 0"
        );
        assert!(
            baseline.mlp_activation_upload_bytes > 0,
            "baseline must record mlp_activation_upload_bytes > 0; got 0"
        );
        assert!(
            baseline.score.pass,
            "baseline must pass full-forward V2.1; failures={:?}",
            baseline.score.failures
        );
        let candidate = measure_mode(
            &model,
            &tokens,
            &ref64,
            "candidate_device_only_mlp",
            true,
            SAMPLES,
            WARMS,
        );
        print_mode(&candidate);
        assert!(
            candidate.score.pass,
            "candidate must pass full-forward V2.1 hard gates against f64; failures={:?}",
            candidate.score.failures
        );
        assert_eq!(
            candidate.mlp_gate_up_download_bytes, 0,
            "candidate mlp_gate_up_download_bytes must be 0; got {}",
            candidate.mlp_gate_up_download_bytes
        );
        assert_eq!(
            candidate.mlp_activation_upload_bytes, 0,
            "candidate mlp_activation_upload_bytes must be 0; got {}",
            candidate.mlp_activation_upload_bytes
        );
        assert!(
            candidate.device_only_mlp_hits_probe > 0 || candidate.device_only_mlp_hits_ledger > 0,
            "candidate recorded zero device-only MLP hits — path did not run \
             (probe={} ledger={} fallbacks={})",
            candidate.device_only_mlp_hits_probe,
            candidate.device_only_mlp_hits_ledger,
            candidate.device_only_mlp_fallbacks
        );
        assert_eq!(
            candidate.device_only_mlp_fallbacks, 0,
            "candidate fell back to host SiLU {} times — not a device-only run",
            candidate.device_only_mlp_fallbacks
        );
        // Waits and CBs must not regress vs baseline.
        assert!(
            candidate.waits_total <= baseline.waits_total,
            "waits regressed: candidate {} > baseline {}",
            candidate.waits_total,
            baseline.waits_total
        );
        assert!(
            candidate.command_buffers_total <= baseline.command_buffers_total,
            "command buffers regressed: candidate {} > baseline {}",
            candidate.command_buffers_total,
            baseline.command_buffers_total
        );
        // candidate never ran (prior faked proof).
        assert_ne!(
            candidate.command_buffers_total, baseline.command_buffers_total,
            "command_buffers identical ({}) between baseline and candidate — \
             candidate path did not change physical CB structure",
            baseline.command_buffers_total
        );
        set_device_only_mlp(true);
        set_device_only_mlp_poison(true);
        reset_device_only_mlp_probe();
        let (poison_logits, _, _) = model
            .forward_resident_counted(&tokens)
            .expect("poisoned resident forward");
        let poison_hits = device_only_mlp_hits();
        let poison_score = score_against_f64(
            &poison_logits,
            &ref64,
            &Bounds::full_forward_logits(),
            "poisoned_device_only_mlp",
        );
        assert!(
            poison_hits > 0,
            "poison path must still take device-only hits; got 0"
        );
        assert!(
            !poison_score.pass,
            "HAWKING_GLM_GPU_DEVICE_ONLY_MLP_POISON=1 must fail V2.1 hard gates \
             (causal mutation); score passed unexpectedly: {}",
            format_score_line(&poison_score)
        );
        set_device_only_mlp_poison(false);
        set_device_only_mlp(false);
        let row = |label: &str, b: String, c: String| {};
        row(
            "rel_l2",
            format!("{:.3e}", baseline.score.continuous.relative_l2),
            format!("{:.3e}", candidate.score.continuous.relative_l2),
        );
        row(
            "cos",
            format!("{:.9}", baseline.score.continuous.cosine_similarity),
            format!("{:.9}", candidate.score.continuous.cosine_similarity),
        );
        row(
            "kl",
            baseline
                .score
                .continuous
                .kl_divergence
                .map(|k| format!("{k:.3e}"))
                .unwrap_or_else(|| "n/a".into()),
            candidate
                .score
                .continuous
                .kl_divergence
                .map(|k| format!("{k:.3e}"))
                .unwrap_or_else(|| "n/a".into()),
        );
        row(
            "max_meaningful_rel",
            format!("{:.3e}", baseline.score.continuous.max_meaningful_rel),
            format!("{:.3e}", candidate.score.continuous.max_meaningful_rel),
        );
        row(
            "ulp[med/p95/p99/max]",
            format!(
                "{:.0}/{:.0}/{:.0}/{:.0}",
                baseline.score.continuous.ulp.median,
                baseline.score.continuous.ulp.p95,
                baseline.score.continuous.ulp.p99,
                baseline.score.continuous.ulp.max
            ),
            format!(
                "{:.0}/{:.0}/{:.0}/{:.0}",
                candidate.score.continuous.ulp.median,
                candidate.score.continuous.ulp.p95,
                candidate.score.continuous.ulp.p99,
                candidate.score.continuous.ulp.max
            ),
        );
        row(
            "argmax ref/cand",
            format!(
                "{:?}/{:?}",
                baseline.score.discrete.greedy_argmax_ref,
                baseline.score.discrete.greedy_argmax_cand
            ),
            format!(
                "{:?}/{:?}",
                candidate.score.discrete.greedy_argmax_ref,
                candidate.score.discrete.greedy_argmax_cand
            ),
        );
        row(
            "topk_ok",
            format!("{}", baseline.score.discrete.top_k_exact_match),
            format!("{}", candidate.score.discrete.top_k_exact_match),
        );
        row(
            "hard_gates_pass",
            format!("{}", baseline.score.pass),
            format!("{}", candidate.score.pass),
        );
        row(
            "waits_total",
            format!("{}", baseline.waits_total),
            format!("{}", candidate.waits_total),
        );
        row(
            "waits_per_token",
            format!("{}", baseline.waits_per_token),
            format!("{}", candidate.waits_per_token),
        );
        row(
            "command_buffers_total",
            format!("{}", baseline.command_buffers_total),
            format!("{}", candidate.command_buffers_total),
        );
        row(
            "command_buffers_per_token",
            format!("{}", baseline.command_buffers_per_token),
            format!("{}", candidate.command_buffers_per_token),
        );
        row(
            "mlp_gate_up_download_bytes",
            format!("{}", baseline.mlp_gate_up_download_bytes),
            format!("{}", candidate.mlp_gate_up_download_bytes),
        );
        row(
            "mlp_activation_upload_bytes",
            format!("{}", baseline.mlp_activation_upload_bytes),
            format!("{}", candidate.mlp_activation_upload_bytes),
        );
        row(
            "wall_us_p50 (full fixture)",
            format!("{}", baseline.wall_us_p50),
            format!("{}", candidate.wall_us_p50),
        );
        row(
            "wall_us_p95 (full fixture)",
            format!("{}", baseline.wall_us_p95),
            format!("{}", candidate.wall_us_p95),
        );
        row(
            "wall_us_per_token_p50",
            format!("{}", baseline.wall_us_per_token_p50),
            format!("{}", candidate.wall_us_per_token_p50),
        );
        row(
            "wall_us_per_token_p95",
            format!("{}", baseline.wall_us_per_token_p95),
            format!("{}", candidate.wall_us_per_token_p95),
        );
        let p50_improved = candidate.wall_us_per_token_p50 < baseline.wall_us_per_token_p50;
        let p95_improved = candidate.wall_us_per_token_p95 < baseline.wall_us_per_token_p95;
        let p50_tie = candidate.wall_us_per_token_p50 == baseline.wall_us_per_token_p50;
        let p95_tie = candidate.wall_us_per_token_p95 == baseline.wall_us_per_token_p95;
        let verdict = if p50_improved && p95_improved {
            "PROMOTE"
        } else {
            "NEGATIVE"
        };
        // Do not flip the default. Measurement only.
        set_device_only_mlp(false);
        set_device_only_mlp_poison(false);
    }
}
/// Restore process env for device-only MLP flags when the acceptance test ends.
#[cfg(target_os = "macos")]
fn scopeguard_restore(
    prev_flag: Option<std::ffi::OsString>,
    prev_poison: Option<std::ffi::OsString>,
) -> impl Drop {
    struct Restore(Option<std::ffi::OsString>, Option<std::ffi::OsString>);
    impl Drop for Restore {
        fn drop(&mut self) {
            match self.0.take() {
                Some(v) => std::env::set_var(GPU_DEVICE_ONLY_MLP_ENV, v),
                None => std::env::remove_var(GPU_DEVICE_ONLY_MLP_ENV),
            }
            match self.1.take() {
                Some(v) => std::env::set_var(GPU_DEVICE_ONLY_MLP_POISON_ENV, v),
                None => std::env::remove_var(GPU_DEVICE_ONLY_MLP_POISON_ENV),
            }
        }
    }
    Restore(prev_flag, prev_poison)
}

use hawking_core::gravity_glm::GravityGlm;
use std::path::PathBuf;
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gravity_glm")
}
#[derive(serde::Deserialize)]
struct Reference {
    tokens: Vec<u32>,
    argmax: u32,
    top5: Vec<u32>,
    logits_head: Vec<f32>,
    final_topk_indices: Vec<u32>,
    artifact: String,
    tensors: usize,
    tensors_pq: usize,
}
fn top_k(logits: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .expect("no NaN in logits")
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}
#[test]
fn gravity_glm_forward_matches_frozen_oracle() {
    let dir = fixtures_dir();
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(dir.join("ref_glm.json")).expect("read ref_glm"))
            .expect("parse ref_glm");
    let want: Vec<f32> = std::fs::read(dir.join("ref_logits.f32"))
        .expect("read ref logits")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let model = GravityGlm::open(&dir.join(&reference.artifact), true).expect("open GLM artifact");
    assert_eq!(
        model.arch.vocab_size,
        want.len(),
        "vocab vs reference logits"
    );
    assert!(
        reference.tensors_pq > 0,
        "fixture exercises no packed tensor"
    );
    assert!(reference.tensors > 100, "fixture is smaller than described");
    let (got, trace) = model.forward(&reference.tokens).expect("forward");
    assert_eq!(got.len(), want.len(), "logit count");
    let max_abs = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    for (i, (&a, &b)) in got.iter().zip(want.iter()).enumerate() {
        let tol = 1e-3 + 1e-4 * b.abs();
        assert!(
            (a - b).abs() <= tol,
            "logit {i}: got {a}, want {b}, diff {} > tol {tol}",
            (a - b).abs()
        );
    }
    for (i, &w) in reference.logits_head.iter().enumerate() {
        assert!(
            (got[i] - w).abs() <= 1e-3 + 1e-4 * w.abs(),
            "logits_head[{i}]: got {}, want {w}",
            got[i]
        );
    }
    let got_top5 = top_k(&got, 5);
    assert_eq!(got_top5[0], reference.argmax, "argmax");
    assert_eq!(got_top5, reference.top5, "top-5");
    let mut got_topk: Vec<u32> = trace.final_topk.iter().map(|&t| t as u32).collect();
    got_topk.sort_unstable();
    let mut want_topk = reference.final_topk_indices.clone();
    want_topk = want_topk.split_off(want_topk.len() - got_topk.len());
    want_topk.sort_unstable();
    assert_eq!(got_topk, want_topk, "final-layer DSA top-k selection");
}
#[test]
fn glm_arch_refuses_a_leading_indexshare_layer() {
    use hawking_core::gravity_glm::GlmArch;
    let mut header: serde_json::Value = serde_json::from_slice(
        &std::fs::read(fixtures_dir().join("ref_glm.json")).expect("read ref_glm"),
    )
    .expect("parse");
    let model = GravityGlm::open(
        &fixtures_dir().join(
            header
                .get("artifact")
                .and_then(serde_json::Value::as_str)
                .expect("artifact name"),
        ),
        false,
    )
    .expect("open");
    header = serde_json::json!({"architecture": {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": model.arch.n_layers,
        "hidden_size": model.arch.hidden,
        "num_attention_heads": model.arch.n_heads,
        "q_lora_rank": model.arch.q_lora_rank,
        "kv_lora_rank": model.arch.kv_lora_rank,
        "qk_nope_head_dim": model.arch.qk_nope_head_dim,
        "qk_rope_head_dim": model.arch.qk_rope_head_dim,
        "v_head_dim": model.arch.v_head_dim,
        "index_n_heads": model.arch.index_n_heads,
        "index_head_dim": model.arch.index_head_dim,
        "index_topk": model.arch.index_topk,
        "n_routed_experts": model.arch.n_routed_experts,
        "n_group": model.arch.n_group,
        "topk_group": model.arch.topk_group,
        "num_experts_per_tok": model.arch.num_experts_per_tok,
        "norm_topk_prob": model.arch.norm_topk_prob,
        "routed_scaling_factor": model.arch.routed_scaling_factor,
        "vocab_size": model.arch.vocab_size,
        "rms_norm_eps": model.arch.rms_norm_eps,
        "rope_parameters": {"rope_theta": model.arch.rope_theta},
        "indexer_types": vec!["shared"; model.arch.n_layers],
        "mlp_layer_types": model.arch.mlp_layer_types.clone(),
    }});
    let err = GlmArch::from_header(&header).expect_err("leading IndexShare must be refused");
    assert!(
        format!("{err}").contains("no previous index"),
        "unexpected error: {err}"
    );
}
#[test]
fn adapter_accepts_the_synthesized_flagship_architecture() {
    use hawking_core::gravity_glm::GlmArch;
    let raw =
        std::fs::read(fixtures_dir().join("flagship_arch.json")).expect("read flagship_arch.json");
    let header: serde_json::Value = serde_json::from_slice(&raw).expect("parse");
    let arch = GlmArch::from_header(&header).expect("adapter must accept the flagship header");
    assert_eq!(arch.n_layers, 78);
    assert_eq!(arch.hidden, 6144);
    assert_eq!(arch.n_heads, 64);
    assert_eq!(arch.n_routed_experts, 256);
    assert_eq!(arch.num_experts_per_tok, 8);
    assert_eq!(arch.vocab_size, 154880);
    assert_eq!(arch.qk_dim(), 256, "qk_nope + qk_rope");
    assert_eq!(arch.index_topk, 2048);
    let full = arch.indexer_types.iter().filter(|t| *t == "full").count();
    assert_eq!(full, 21, "full-indexer layers");
    assert_eq!(arch.indexer_types.len() - full, 57, "IndexShare layers");
    assert_eq!(arch.indexer_types[0], "full", "layer 0 cannot share");
    assert_eq!(
        arch.mlp_layer_types
            .iter()
            .filter(|t| *t == "dense")
            .count(),
        3,
        "first_k_dense_replace"
    );
}

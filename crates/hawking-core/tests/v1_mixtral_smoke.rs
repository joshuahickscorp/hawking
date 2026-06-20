use hawking_core::model::mixtral::{MixtralConfig, MixtralEngine};

#[test]
fn synthetic_mixtral_shape_smoke_is_finite() {
    let cfg = MixtralConfig::synthetic_for_test();
    assert_eq!(cfg.n_experts, 8);
    assert_eq!(cfg.top_k, 2);
    assert_eq!(cfg.hidden, cfg.n_heads * cfg.head_dim);

    let logits = MixtralEngine::synthetic_forward_shape_for_test(&cfg, 17);
    assert_eq!(logits.len(), cfg.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

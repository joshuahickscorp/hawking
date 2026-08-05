//! Unit tests for the broker-kernel A/B promotion gate.
//!
//! No Metal, no Gravity artifact, no forward-lane modules.

use hawking_core::broker_kernel_ab::{
    broker_kernel_ranking, decide_promotion, layer_cost_snapshot, AbTrialInput, BrokerScope,
    KernelFamily, PromotionVerdict, ACT_QUANT_AUTHORITY_GPU_US, FP4_MATVEC_DISPATCHES_PER_LAYER,
    L0_L1_MS_PER_LAYER, L0_L2_METAL_DISPATCHES, P6_DISPATCHES_PER_LAYER,
};

#[test]
fn ranking_is_complete_and_ordered() {
    let ranks = broker_kernel_ranking();
    assert_eq!(ranks.len(), KernelFamily::all().len());
    for (i, entry) in ranks.iter().enumerate() {
        assert_eq!(entry.rank as usize, i + 1, "ranks must be dense 1..N");
        assert!(!entry.authority_kernel.is_empty());
        assert!(!entry.current_cost.is_empty());
        assert!(!entry.tuning_levers.is_empty());
    }
}

#[test]
fn shared_vs_terra_scopes_present() {
    let ranks = broker_kernel_ranking();
    assert!(ranks.iter().any(|e| matches!(e.scope, BrokerScope::Shared)));
    assert!(ranks
        .iter()
        .any(|e| matches!(e.scope, BrokerScope::TerraDeepSeek)));
    // FP4 is Terra-native; act quant + CB topology are shared wins.
    let fp4 = ranks
        .iter()
        .find(|e| e.family == KernelFamily::Fp4ExpertMatvec)
        .unwrap();
    assert!(matches!(fp4.scope, BrokerScope::TerraDeepSeek));
    let act = ranks
        .iter()
        .find(|e| e.family == KernelFamily::ActQuant)
        .unwrap();
    assert!(matches!(act.scope, BrokerScope::Shared));
}

#[test]
fn receipt_constants_match_sealed_multi_layer_shape() {
    // Guard against silent drift of the plan's cited numbers.
    assert_eq!(L0_L2_METAL_DISPATCHES, 276);
    assert_eq!(P6_DISPATCHES_PER_LAYER, 60);
    assert_eq!(FP4_MATVEC_DISPATCHES_PER_LAYER, 18);
    assert_eq!(ACT_QUANT_AUTHORITY_GPU_US, 5_967);
    assert!(L0_L1_MS_PER_LAYER > 8_000.0 && L0_L1_MS_PER_LAYER < 10_000.0);
    let snap = layer_cost_snapshot();
    assert_eq!(snap.parity_class, "NUMERIC_PARITY_V2_1_ONLY");
    assert_eq!(snap.l0_l2_command_buffers, 26);
}

#[test]
fn reject_parity_even_when_orders_of_magnitude_faster() {
    let d = decide_promotion(AbTrialInput {
        family: KernelFamily::Fp4ExpertMatvec,
        authority_kernel: "authority".into(),
        candidate_kernel: "candidate".into(),
        parity_pass: false,
        parity_detail: Some("rel_l2 exceeded".into()),
        authority_gpu_p50_us: Some(1_000_000),
        candidate_gpu_p50_us: Some(1),
    });
    assert_eq!(d.verdict, PromotionVerdict::RejectParity);
    assert!(!d.serve_promoted);
    assert!(d.parity_detail.as_deref().unwrap().contains("rel_l2"));
}

#[test]
fn reject_no_win_when_parity_ok_but_not_faster() {
    let equal = decide_promotion(AbTrialInput {
        family: KernelFamily::Fp8ControlMatvec,
        authority_kernel: "authority".into(),
        candidate_kernel: "candidate".into(),
        parity_pass: true,
        parity_detail: None,
        authority_gpu_p50_us: Some(500),
        candidate_gpu_p50_us: Some(500),
    });
    assert_eq!(equal.verdict, PromotionVerdict::RejectNoWin);
    assert!(!equal.serve_promoted);

    let slower = decide_promotion(AbTrialInput {
        family: KernelFamily::Fp8ControlMatvec,
        authority_kernel: "authority".into(),
        candidate_kernel: "candidate".into(),
        parity_pass: true,
        parity_detail: None,
        authority_gpu_p50_us: Some(500),
        candidate_gpu_p50_us: Some(800),
    });
    assert_eq!(slower.verdict, PromotionVerdict::RejectNoWin);

    let untimed = decide_promotion(AbTrialInput {
        family: KernelFamily::ActQuant,
        authority_kernel: "authority".into(),
        candidate_kernel: "candidate".into(),
        parity_pass: true,
        parity_detail: None,
        authority_gpu_p50_us: None,
        candidate_gpu_p50_us: None,
    });
    assert_eq!(untimed.verdict, PromotionVerdict::RejectNoWin);
    assert!(untimed.speed_ratio.is_none());
}

#[test]
fn candidate_ready_never_sets_serve_promoted() {
    let d = decide_promotion(AbTrialInput {
        family: KernelFamily::ActQuant,
        authority_kernel: "deepseek_v4_act_quant_bf16_ue8m0_authority".into(),
        candidate_kernel: "deepseek_v4_act_quant_bf16_ue8m0_simdgroup_block_candidate".into(),
        parity_pass: true,
        parity_detail: None,
        authority_gpu_p50_us: Some(ACT_QUANT_AUTHORITY_GPU_US),
        candidate_gpu_p50_us: Some(ACT_QUANT_AUTHORITY_GPU_US / 2),
    });
    assert_eq!(d.verdict, PromotionVerdict::CandidateReady);
    assert!(d.speed_improved);
    assert!(d.speed_ratio.unwrap() < 1.0);
    assert!(!d.serve_promoted, "scaffold must never flip serve");
}

#[test]
fn family_parse_roundtrip() {
    for f in KernelFamily::all() {
        assert_eq!(KernelFamily::parse(f.as_str()), Some(*f));
    }
    assert!(KernelFamily::parse("not_a_kernel").is_none());
}

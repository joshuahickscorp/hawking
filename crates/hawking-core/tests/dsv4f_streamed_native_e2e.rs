//! End-to-end streamed Metal BOS decode against the sealed DSV4F artifact.
//!
//! Skips when the artifact is absent. Full 43-layer greedy check is the
//! parity gate for the opt-in Metal path.

use hawking_core::gravity_deepseek_v4_streamed_forward::{
    discover_sealed_dsv4f_artifact, run_streamed_forward, StreamedForwardConfig,
    DECLARED_PEAK_RSS_BOUND_BYTES,
};
use hawking_core::gravity_deepseek_v4_streamed_native::{
    greedy_logit_within_tolerance, ORACLE_GREEDY_LOGIT, ORACLE_GREEDY_TOKEN_ID,
};

#[test]
fn streamed_metal_e2e_greedy_token_and_residency() {
    let Some(artifact) = discover_sealed_dsv4f_artifact() else {
        eprintln!("sealed DSV4F artifact not found; metal e2e skipped");
        return;
    };
    if std::env::var("HAWKING_DSV4F_NATIVE_E2E").ok().as_deref() != Some("1") {
        eprintln!(
            "set HAWKING_DSV4F_NATIVE_E2E=1 to run the 43-layer metal e2e (skipped by default)"
        );
        return;
    }
    let mut config = StreamedForwardConfig::for_layers(42).expect("max layer");
    config.use_metal = true;
    config.compute_final_head = true;
    let report = run_streamed_forward(&artifact, config).expect("metal streamed forward");
    assert!(
        report.stop_reason.is_none(),
        "metal e2e stopped: {:?}",
        report.stop_reason
    );
    assert_eq!(report.layers_executed.len(), 43);
    assert!(!report.native);
    assert!(report.honesty.metal_dispatches > 0);
    assert_eq!(
        report.honesty.fallbacks, 0,
        "unexpected metal fallbacks: {:?}",
        report.honesty
    );
    let greedy = report.greedy.expect("greedy token");
    assert_eq!(greedy.token_id, ORACLE_GREEDY_TOKEN_ID);
    assert!(
        greedy_logit_within_tolerance(greedy.logit, ORACLE_GREEDY_LOGIT),
        "logit {} outside {} of {}",
        greedy.logit,
        0.05,
        ORACLE_GREEDY_LOGIT
    );
    assert!(report.rss_within_bound);
    assert!(report.peak_rss_bytes <= DECLARED_PEAK_RSS_BOUND_BYTES);
    assert!(report.weight_within_bound);
}

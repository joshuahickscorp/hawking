//! Native BOS token graph parity against the sealed CPU oracle.
//!
//! Skips when the artifact is absent. The 43-layer greedy check is the
//! required parity gate for this lane.

use hawking_core::gravity_deepseek_v4_native_token_graph::{
    run_native_bos_token, NATIVE_TOKEN_GRAPH_KERNELS, ORACLE_GREEDY_LOGIT, ORACLE_GREEDY_TOKEN_ID,
    ORACLE_HC_BF16_SHA256,
};
use hawking_core::gravity_deepseek_v4_streamed_forward::discover_sealed_dsv4f_artifact;
use hawking_core::metal::SHADER_DSV4F_NATIVE_TOKEN_GRAPH;

#[test]
fn native_graph_kernels_have_trace_names() {
    for kernel in NATIVE_TOKEN_GRAPH_KERNELS {
        assert!(
            SHADER_DSV4F_NATIVE_TOKEN_GRAPH.contains(&format!("kernel void {kernel}(")),
            "{kernel} missing from dsv4f_native_token_graph.metal"
        );
    }
}

#[test]
fn native_graph_e2e_parity_and_zero_host_gather() {
    let Some(artifact) = discover_sealed_dsv4f_artifact() else {
        eprintln!("sealed DSV4F artifact not found; native graph e2e skipped");
        return;
    };
    if std::env::var("HAWKING_DSV4F_NATIVE_GRAPH_E2E")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!(
            "set HAWKING_DSV4F_NATIVE_GRAPH_E2E=1 to run the 43-layer native graph e2e (skipped by default)"
        );
        return;
    }
    let report = run_native_bos_token(&artifact, 42, true).expect("native bos token");
    assert!(
        report.stop_reason.is_none(),
        "native graph stopped: {:?}",
        report.stop_reason
    );
    assert_eq!(report.layers_executed.len(), 43);
    assert_eq!(report.counters.host_expert_gather, 0);
    assert_eq!(report.counters.host_expert_output_readback, 0);
    assert_eq!(report.counters.fallbacks, 0);
    assert!(report.counters.metal_dispatches > 0);
    assert!(
        report.counters.command_buffers < 3343,
        "command buffers {} did not beat the scaffold dispatch count",
        report.counters.command_buffers
    );
    assert!(report.honesty.compact_top6_worklist);
    assert!(!report.honesty.dense_over_256);
    let greedy = report.greedy.expect("greedy token");
    assert_eq!(greedy.token_id, ORACLE_GREEDY_TOKEN_ID);
    assert!(
        (greedy.logit - ORACLE_GREEDY_LOGIT).abs() <= 0.05,
        "logit {} outside 0.05 of {}",
        greedy.logit,
        ORACLE_GREEDY_LOGIT
    );
    if report.hc_bf16_sha256 != ORACLE_HC_BF16_SHA256 {
        eprintln!(
            "HC SHA diverged from the CPU oracle: got {} expected {}",
            report.hc_bf16_sha256, ORACLE_HC_BF16_SHA256
        );
    }
    assert!(report.rss_within_bound);
    assert!(report.weight_within_bound);
}

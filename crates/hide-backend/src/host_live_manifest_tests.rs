use super::build_live_manifest;
#[test]
fn ssm_regime_carries_recall_fidelity() {
    let ssm = build_live_manifest(Some(6 * 1024 * 1024), Some(1000), 1000, 500);
    assert!(ssm.recall_fidelity.is_some());
    assert!(ssm.state_bytes.is_some());
    assert!(ssm.kv_seq_len.is_none());
    assert!(
        (ssm.occupancy - 0.5).abs() < 0.05,
        "occupancy {}",
        ssm.occupancy
    );
}
#[test]
fn transformer_regime_has_no_fidelity() {
    let tf = build_live_manifest(None, Some(4096), 4096, 1024);
    assert!(tf.recall_fidelity.is_none());
    assert!(tf.kv_seq_len.is_some());
}

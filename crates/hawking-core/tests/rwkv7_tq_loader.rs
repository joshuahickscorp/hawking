#![cfg(feature = "tq")]
#[allow(dead_code)]
fn expected_proj_names(n_layers: usize) -> Vec<String> {
    let mut names = Vec::with_capacity(n_layers * 6);
    for i in 0..n_layers {
        names.push(format!("blk.{i}.time_mix_receptance.weight"));
        names.push(format!("blk.{i}.time_mix_key.weight"));
        names.push(format!("blk.{i}.time_mix_value.weight"));
        names.push(format!("blk.{i}.time_mix_gate.weight"));
        names.push(format!("blk.{i}.channel_mix_key.weight"));
        names.push(format!("blk.{i}.channel_mix_value.weight"));
    }
    names
}
#[test]
#[ignore = "requires RWKV7_TQ_TEST_ARTIFACT env var pointing to a .tq file"]
fn tq_artifact_loads_expected_names() {
    let path = std::env::var("RWKV7_TQ_TEST_ARTIFACT")
        .expect("RWKV7_TQ_TEST_ARTIFACT must be set to run this test");
    let _ = &path; // placeholder until the real loader is available
    panic!("STUB: wire hawking_core::model::rwkv7::load_tq_artifact and check tensor names");
}
#[test]
#[ignore = "requires RWKV7_TQ_TEST_ARTIFACT env var pointing to a .tq file"]
fn tq_artifact_shapes_match_04b() {
    let path = std::env::var("RWKV7_TQ_TEST_ARTIFACT")
        .expect("RWKV7_TQ_TEST_ARTIFACT must be set to run this test");
    const N_FF: usize = 4096;
    const N_EMBD: usize = 1024;
    let _ = (path, N_FF, N_EMBD);
    panic!("STUB: wire hawking_core::model::rwkv7::load_tq_artifact and check shapes");
}
#[test]
fn tq_loader_missing_artifact_is_err() {
    let names = expected_proj_names(24);
    assert_eq!(names.len(), 144, "24 layers × 6 projections = 144 names");
    assert_eq!(names[0], "blk.0.time_mix_receptance.weight");
    assert_eq!(names[5], "blk.0.channel_mix_value.weight");
    assert_eq!(names[6], "blk.1.time_mix_receptance.weight");
    assert_eq!(
        names[143], "blk.23.channel_mix_value.weight",
        "last tensor name for 24-layer model"
    );
}

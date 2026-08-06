#![cfg(target_os = "macos")]
use hawking_core::gguf::{GgmlType, GgufFile};
use hawking_core::mixed_quant_store::{MixedQuantStore, StoreKey};
use hawking_core::quant_tier_map::{GroupKind, TierMap};
use std::path::PathBuf;
mod common;
use common::weights_path_deepseek as weights_path;
#[test]
fn build_default_tier_map_against_v2_lite_gguf() {
    let weights = weights_path();
    if !weights.exists() {
        eprintln!("skipping: V2-Lite weights missing");
        return;
    }
    let tier_path = PathBuf::from("../../artifacts/calibration/tier_maps/v2_lite_default.json");
    if !tier_path.exists() {
        eprintln!("skipping: tier map missing");
        return;
    }
    let gguf = GgufFile::open(&weights).expect("open gguf");
    let tier_map = TierMap::load(&tier_path).expect("load tier map");
    tier_map
        .validate("deepseek2", 27)
        .expect("tier map matches V2-Lite shape");
    let store = MixedQuantStore::build(
        &gguf, &tier_map, 27, 1,  // first_k_dense_layers for V2-Lite
        64, // n_routed_experts
        true,
    )
    .expect("build store");
    assert!(
        store.len_tensors() > 0,
        "tier map should have produced at least one re-quantized tensor"
    );
    let key = StoreKey::routed(4, GroupKind::Down, 0);
    let t = store.get(key).expect("layer 4 down expert 0 in store");
    assert_eq!(t.dtype, GgmlType::Q8_0);
    assert_eq!(t.n_elems, 1408 * 2048);
    assert_eq!(t.byte_size, (1408 * 2048 / 32) * 34);
    if let Some(t) = store.get(StoreKey::routed(25, GroupKind::Down, 5)) {
        assert_eq!(t.dtype, GgmlType::Q6_K);
    } else {
        eprintln!("layer 25 down already at Q6_K in source; build skipped (no-op)");
    }
    assert!(
        store.blob().len() <= 6 * 1024 * 1024 * 1024,
        "store blob {} bytes; expected ≤ 6 GiB",
        store.blob().len()
    );
}

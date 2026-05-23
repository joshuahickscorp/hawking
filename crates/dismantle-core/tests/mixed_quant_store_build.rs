//! path-to-50 lever 2 foundation smoke: build a MixedQuantStore from
//! the live V2-Lite GGUF using the default tier map and verify the
//! resulting blob byte layout matches expectations. Does NOT exercise
//! the dispatcher (gated on the kernel-buffer override wedge — see
//! reports/mixed_precision_quant_wiring_handoff.md §3.4).

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use dismantle_core::gguf::{GgmlType, GgufFile};
use dismantle_core::mixed_quant_store::{MixedQuantStore, StoreKey};
use dismantle_core::quant_tier_map::{GroupKind, TierMap};

fn weights_path() -> PathBuf {
    PathBuf::from("../../models/deepseek-v2-lite-q4.gguf")
}

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
        &gguf,
        &tier_map,
        27,
        1,  // first_k_dense_layers for V2-Lite
        64, // n_routed_experts
        true,
    )
    .expect("build store");

    // Some layers may already be at the target dtype in the GGUF (in
    // which case the build skips them as no-ops), so the count varies
    // by source-quant. Just sanity-check that *some* re-quantization
    // happened and the spot-check tensors are present.
    eprintln!(
        "mixed_quant_store: {} tensors materialized",
        store.len_tensors()
    );
    assert!(
        store.len_tensors() > 0,
        "tier map should have produced at least one re-quantized tensor"
    );

    // Spot-check: layer 4 expert 0 down should be Q8_0 per the map.
    let key = StoreKey::routed(4, GroupKind::Down, 0);
    let t = store.get(key).expect("layer 4 down expert 0 in store");
    assert_eq!(t.dtype, GgmlType::Q8_0);
    // V2-Lite moe_intermediate * hidden = 1408 * 2048 = 2_883_584 elems
    assert_eq!(t.n_elems, 1408 * 2048);
    // Q8_0: 34 bytes per 32 elems → 2_883_584 / 32 * 34 = 3_063_808 bytes
    assert_eq!(t.byte_size, (1408 * 2048 / 32) * 34);

    // Spot-check: layer 25 down → Q6_K per default map (if not native).
    if let Some(t) = store.get(StoreKey::routed(25, GroupKind::Down, 5)) {
        assert_eq!(t.dtype, GgmlType::Q6_K);
    } else {
        eprintln!("layer 25 down already at Q6_K in source; build skipped (no-op)");
    }

    // Total blob size is sanity-bounded: ~ 26 layers * 64 experts *
    // ~3 MB/expert = ~5 GB upper bound (Q8 case).
    assert!(
        store.blob().len() <= 6 * 1024 * 1024 * 1024,
        "store blob {} bytes; expected ≤ 6 GiB",
        store.blob().len()
    );

    eprintln!(
        "mixed_quant_store: {} tensors / {} MB blob",
        store.len_tensors(),
        store.blob().len() / (1024 * 1024)
    );
}

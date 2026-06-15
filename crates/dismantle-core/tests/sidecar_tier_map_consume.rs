//! Track 4.3 gate: a sidecar tier map (a) round-trips when attached to a
//! predec sidecar via `attach_tier_map_to_sidecar`, and (b) the LOADED copy's
//! resolver (`SidecarTierMap::dtype_for`) — the exact fn the loader hook
//! `honor_sidecar_tier_map` consults — reports the per-tensor override, returns
//! None for absent tensors, and validates. Pure file I/O + byte math, no Metal.

use dismantle_core::gguf::GgmlType;
use dismantle_core::sidecar::{
    attach_tier_map_to_sidecar, load_sidecar_tier_map_json, read_predec_entries,
    SidecarContents, SidecarHeader, SidecarProfile, SidecarQuality, SidecarTierEntry,
    SidecarTierMap, SidecarWriter, SIDECAR_VERSION,
};

fn predec_only_header() -> SidecarHeader {
    SidecarHeader {
        version: SIDECAR_VERSION,
        source_gguf_hash: "gguf_hash_xyz".into(),
        tokenizer_hash: "tok".into(),
        shader_hash: "shader".into(),
        bake_profile: SidecarProfile::Fast,
        contents: SidecarContents { q4k_predec_scales: true, ..Default::default() },
        quality: SidecarQuality::default(),
        bake_device: "test".into(),
        bake_time_secs: 0,
        tier_map: None,
    }
}

#[test]
fn attach_then_loader_resolver_reports_override() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("model.dismantle");

    // 1) Write a predec-ONLY sidecar (what bake_sidecar_predec produces).
    let base = SidecarWriter {
        path: path.clone(),
        predec_entries: vec![(0x1000u64, vec![1.0_f32, 2.0, 3.0]), (0x2000u64, vec![4.0_f32])],
        header: predec_only_header(),
    };
    assert!(base.write().expect("write predec sidecar") > 0);

    // 2) Attach a tier map (what the CLI does after bake).
    let tm = SidecarTierMap {
        entries: vec![
            SidecarTierEntry { tensor: "blk.0.ffn_down.weight".into(), dtype: "q6_K".into() },
            SidecarTierEntry { tensor: "blk.7.attn_v.weight".into(), dtype: "q8_0".into() },
        ],
    };
    assert!(attach_tier_map_to_sidecar(&path, tm.clone()).expect("attach") > 0);

    // 3) Read the LOADED copy back and exercise the resolver the loader hook uses.
    let (header, entries) = read_predec_entries(&path).expect("re-read");
    assert!(header.contents.mixed_quant_tier_map, "content flag must flip on attach");
    assert_eq!(entries.len(), 2, "predec entries survive the rewrite");
    let loaded = header.tier_map.expect("tier map present after attach");
    assert_eq!(loaded, tm, "tier map byte-identical after round-trip");
    assert!(loaded.validate().is_ok());
    // This is exactly what honor_sidecar_tier_map calls per GGUF tensor name:
    assert_eq!(loaded.dtype_for("blk.0.ffn_down.weight").unwrap(), Some(GgmlType::Q6_K));
    assert_eq!(loaded.dtype_for("blk.7.attn_v.weight").unwrap(), Some(GgmlType::Q8_0));
    assert_eq!(loaded.dtype_for("blk.3.attn_q.weight").unwrap(), None, "absent tensor falls through");
}

#[test]
fn bad_dtype_json_fails_the_bake() {
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("tm.json");
    std::fs::write(&p, r#"{"entries":[{"tensor":"blk.0.ffn_down.weight","dtype":"q3_K"}]}"#).unwrap();
    // q3_K is not a supported sidecar tier dtype → load+validate must error,
    // so a typo'd tier fails the bake instead of silently no-op'ing at load.
    assert!(load_sidecar_tier_map_json(&p).is_err());
}

#[test]
fn good_dtype_json_parses() {
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("tm.json");
    std::fs::write(&p, r#"{"entries":[{"tensor":"blk.0.ffn_down.weight","dtype":"q6_K"}]}"#).unwrap();
    let tm = load_sidecar_tier_map_json(&p).expect("parse good json");
    assert_eq!(tm.entries.len(), 1);
    assert_eq!(tm.dtype_for("blk.0.ffn_down.weight").unwrap(), Some(GgmlType::Q6_K));
}

use hawking_core::gguf::GgmlType;
use hawking_core::sidecar::{
    attach_tier_map_to_sidecar, load_sidecar_tier_map_json, read_predec_entries, SidecarContents,
    SidecarHeader, SidecarProfile, SidecarQuality, SidecarTierEntry, SidecarTierMap, SidecarWriter,
    SIDECAR_VERSION,
};
fn predec_only_header() -> SidecarHeader {
    SidecarHeader {
        version: SIDECAR_VERSION,
        source_gguf_hash: "gguf_hash_xyz".into(),
        tokenizer_hash: "tok".into(),
        shader_hash: "shader".into(),
        bake_profile: SidecarProfile::Fast,
        contents: SidecarContents {
            q4k_predec_scales: true,
            ..Default::default()
        },
        quality: SidecarQuality::default(),
        bake_device: "test".into(),
        bake_time_secs: 0,
        tier_map: None,
    }
}
#[test]
fn attach_then_loader_resolver_reports_override() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("model.hawking");
    let base = SidecarWriter {
        path: path.clone(),
        predec_entries: vec![
            (0x1000u64, vec![1.0_f32, 2.0, 3.0]),
            (0x2000u64, vec![4.0_f32]),
        ],
        header: predec_only_header(),
    };
    assert!(base.write().expect("write predec sidecar") > 0);
    let tm = SidecarTierMap {
        entries: vec![
            SidecarTierEntry {
                tensor: "blk.0.ffn_down.weight".into(),
                dtype: "q6_K".into(),
            },
            SidecarTierEntry {
                tensor: "blk.7.attn_v.weight".into(),
                dtype: "q8_0".into(),
            },
        ],
    };
    assert!(attach_tier_map_to_sidecar(&path, tm.clone()).expect("attach") > 0);
    let (header, entries) = read_predec_entries(&path).expect("re-read");
    assert!(
        header.contents.mixed_quant_tier_map,
        "content flag must flip on attach"
    );
    assert_eq!(entries.len(), 2, "predec entries survive the rewrite");
    let loaded = header.tier_map.expect("tier map present after attach");
    assert_eq!(loaded, tm, "tier map byte-identical after round-trip");
    assert!(loaded.validate().is_ok());
    assert_eq!(
        loaded.dtype_for("blk.0.ffn_down.weight").unwrap(),
        Some(GgmlType::Q6_K)
    );
    assert_eq!(
        loaded.dtype_for("blk.7.attn_v.weight").unwrap(),
        Some(GgmlType::Q8_0)
    );
    assert_eq!(
        loaded.dtype_for("blk.3.attn_q.weight").unwrap(),
        None,
        "absent tensor falls through"
    );
}
#[test]
fn bad_dtype_json_fails_the_bake() {
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("tm.json");
    std::fs::write(
        &p,
        r#"{"entries":[{"tensor":"blk.0.ffn_down.weight","dtype":"q3_K"}]}"#,
    )
    .unwrap();
    assert!(load_sidecar_tier_map_json(&p).is_err());
}
#[test]
fn good_dtype_json_parses() {
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("tm.json");
    std::fs::write(
        &p,
        r#"{"entries":[{"tensor":"blk.0.ffn_down.weight","dtype":"q6_K"}]}"#,
    )
    .unwrap();
    let tm = load_sidecar_tier_map_json(&p).expect("parse good json");
    assert_eq!(tm.entries.len(), 1);
    assert_eq!(
        tm.dtype_for("blk.0.ffn_down.weight").unwrap(),
        Some(GgmlType::Q6_K)
    );
}
